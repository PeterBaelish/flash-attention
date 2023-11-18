/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

#include <cuda.h>
#include <chrono>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/**
 * Copyright 2023 Joshua Bakita
 * Library to control SM masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 */

// In functions that do not return an error code, we favor terminating with an
// error rather than merely printing a warning and continuing.
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

// Layout of mask control fields to match CUDA's static global struct
struct global_sm_control {
	uint32_t enabled;
	uint64_t mask;
} __attribute__((packed));

/*** CUDA Globals Manipulation. CUDA 10.2 only ***/

// Ends up being 0x7fb7fa3408 in some binaries (CUDA 10.2, Jetson)
static struct global_sm_control* g_sm_control = NULL;

/* Find the location of CUDA's `globals` struct and the SM mask control fields
 * No symbols are exported from within `globals`, so this has to do a very
 * messy lookup, following the pattern of the assembly of `cuDeviceGetCount()`.
 * Don't call this before the CUDA library has been initialized.
 * (Note that this appears to work, even if built on CUDA > 10.2.)
 */


/*** QMD/TMD-based SM Mask Control via Debug Callback. CUDA 11+ ***/

// Tested working on CUDA x86_64 11.0-12.2.
// Tested not working on aarch64 or x86_64 10.2
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3
static uint64_t g_sm_mask = 0;
static __thread uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;
static void launchCallback(void *ukwn, int domain, int cbid, const void *in_params) {
	if (*(uint32_t*)in_params < 0x50) {
		fprintf(stderr, "Unsupported CUDA version for callback-based SM masking. Aborting...\n");
		return;
	}
	if (!**((uintptr_t***)in_params+8)) {
		fprintf(stderr, "Called with NULL halLaunchDataAllocation\n");
		return;
	}
	//fprintf(stderr, "cta: %lx\n", *(uint64_t*)(**((char***)in_params + 8) + 74));
	// TODO: Check for supported QMD version (>XXX, <4.00)
	// TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 72 bytes (rather than 8 bytes) wide
	uint32_t *lower_ptr = (uint32_t*)(**((char***)in_params + 8) + 84);
	uint32_t *upper_ptr = (uint32_t*)(**((char***)in_params + 8) + 88);
	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}
	//fprintf(stderr, "lower mask: %x\n", *lower_ptr);
	//fprintf(stderr, "upper mask: %x\n", *upper_ptr);
}

static void setup_sm_control_11() {
	int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
	int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
	uintptr_t* tbl_base;
	uint32_t my_hndl;
	// Avoid race conditions (setup can only be called once)
	if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
		return;

	cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
	uintptr_t subscribe_func_addr = *(tbl_base + 3);
	uintptr_t enable_func_addr = *(tbl_base + 6);
	subscribe = (typeof(subscribe))subscribe_func_addr;
	enable = (typeof(enable))enable_func_addr;
	int res = 0;
	res = subscribe(&my_hndl, launchCallback, NULL);
	if (res) {
		fprintf(stderr, "libsmctrl: Error subscribing to launch callback. Error %d\n", res);
		return;
	}
	res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
	if (res)
		fprintf(stderr, "libsmctrl: Error enabling launch callback. Error %d\n", res);
}

// Set default mask for all launches
static void libsmctrl_set_global_mask(uint64_t mask) {
	int ver;
	cuDriverGetVersion(&ver);
	if (ver > 10020) {
		if (!sm_control_setup_called)
			setup_sm_control_11();
		g_sm_mask = mask;
	} else { // < CUDA 10.2
		abort(1, ENOSYS, "Global masking requires at least CUDA 10.2; "
		                 "this application is using CUDA %d.%d",
		                 ver / 1000, (ver % 100));
	}
}

// Set mask for next launch from this thread
static void libsmctrl_set_next_mask(uint64_t mask) {
	if (!sm_control_setup_called)
		setup_sm_control_11();
	g_next_sm_mask = mask;
}


/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
#define CU_9_0_MASK_OFF_TX2 0x128 // CUDA 9.0 is slightly different on the TX2
// CUDA 9.0 and 9.1 use the same offset
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x24c
// CUDA 10.0, 10.1 and 10.2 use the same offset
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset

// Layout in CUDA's `stream` struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
} __attribute__((packed));

// Check if this system has a Parker SoC (TX2/PX2 chip)
// (CUDA 9.0 behaves slightly different on this platform.)
// @return 1 if detected, 0 if not, -cuda_err on error
#if __aarch64__
int detect_parker_soc() {
	int cap_major, cap_minor, err, dev_count;
	if (err = cuDeviceGetCount(&dev_count))
		return -err;
	// As CUDA devices are numbered by order of compute power, check every
	// device, in case a powerful discrete GPU is attached (such as on the
	// DRIVE PX2). We detect the Parker SoC via its unique CUDA compute
	// capability: 6.2.
	for (int i = 0; i < dev_count; i++) {
		if (err = cuDeviceGetAttribute(&cap_minor,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		                               i))
			return -err;
		if (err = cuDeviceGetAttribute(&cap_major,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		                               i))
			return -err;
		if (cap_major == 6 && cap_minor == 2)
			return 1;
	}
	return 0;
}
#endif // __aarch64__

// Should work for CUDA 8.0 through 12.1
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
static void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010: {
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
#if __aarch64__
		// Jetson TX2 offset is slightly different on CUDA 9.0.
		// Only compile the check into ARM64 builds.
		int is_parker;
		const char* err_str;
		if ((is_parker = detect_parker_soc()) < 0) {
			cuGetErrorName(-is_parker, &err_str);
			fprintf(stderr, "libsmctrl_set_stream_mask: CUDA call "
					"failed while doing compatibilty test."
			                "Error, '%s'. Not applying stream "
					"mask.\n", err_str);
		}

		if (is_parker)
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF_TX2);
#endif
		break;
	}
	case 9020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_2_MASK_OFF);
		break;
	case 10000:
	case 10010:
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_0_MASK_OFF);
		break;
	case 11000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_0_MASK_OFF);
		break;
	case 11010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_1_MASK_OFF);
		break;
	case 11020:
	case 11030:
	case 11040:
	case 11050:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_2_MASK_OFF);
		break;
	case 11060:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_6_MASK_OFF);
		break;
	case 11070:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_7_MASK_OFF);
		break;
	case 11080:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF);
		break;
	case 12000:
	case 12010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	default: {
		// For experimenting to determine the right mask offset, set the MASK_OFF
		// environment variable (positive and negative numbers are supported)
		char* mask_off_str = getenv("MASK_OFF");
		fprintf(stderr, "libsmctrl: Stream masking unsupported on this CUDA version (%d)!\n", ver);
		if (mask_off_str) {
			int off = atoi(mask_off_str);
			fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.1 base %#x "
					"(total off: %#x)\n", off, CU_12_0_MASK_OFF, CU_12_0_MASK_OFF+off);
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF + off);
		} else {
			return;
		}}
	}

	hw_mask->upper = mask >> 32;
	hw_mask->lower = mask;
}

/* INFORMATIONAL FUNCTIONS */

// Read an integer from a file in `/proc`
static int read_int_procfile(char* filename, uint64_t* out) {
	char f_data[18] = {0};
	int fd = open(filename, O_RDONLY);
	if (fd == -1)
		return errno;
	read(fd, f_data, 18);
	close(fd);
	*out = strtoll(f_data, NULL, 16);
	return 0;
}

// We support up to 12 GPCs per GPU, and up to 16 GPUs.
static uint64_t tpc_mask_per_gpc_per_dev[16][12];
// Output mask is vtpc-indexed (virtual TPC)
static int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev) {
	uint32_t i, j, vtpc_idx = 0;
	uint64_t gpc_mask, num_tpc_per_gpc, max_gpcs, gpc_tpc_mask;
	int err;
	char filename[100];
	*num_enabled_gpcs = 0;
	// Maximum number of GPCs supported for this chip
	snprintf(filename, 100, "/proc/gpu%d/num_gpcs", dev);
	if (err = read_int_procfile(filename, &max_gpcs)) {
		fprintf(stderr, "libsmctrl: nvdebug module must be loaded into kernel before "
				"using libsmctrl_get_*_info() functions\n");
		return err;
	}
	// TODO: handle arbitrary-size GPUs
	if (dev > 16 || max_gpcs > 12) {
		fprintf(stderr, "libsmctrl: GPU possibly too large for preallocated map!\n");
		return ERANGE;
	}
	// Set bit = disabled GPC
	snprintf(filename, 100, "/proc/gpu%d/gpc_mask", dev);
	if (err = read_int_procfile(filename, &gpc_mask))
		return err;
	snprintf(filename, 100, "/proc/gpu%d/num_tpc_per_gpc", dev);
	if (err = read_int_procfile(filename, &num_tpc_per_gpc))
		return err;
	// For each enabled GPC
	for (i = 0; i < max_gpcs; i++) {
		// Skip this GPC if disabled
		if ((1 << i) & gpc_mask)
			continue;
		(*num_enabled_gpcs)++;
		// Get the bitstring of TPCs disabled for this GPC
		// Set bit = disabled TPC
		snprintf(filename, 100, "/proc/gpu%d/gpc%d_tpc_mask", dev, i);
		if (err = read_int_procfile(filename, &gpc_tpc_mask))
			return err;
		uint64_t* tpc_mask = &tpc_mask_per_gpc_per_dev[dev][*num_enabled_gpcs - 1];
		*tpc_mask = 0;
		for (j = 0; j < num_tpc_per_gpc; j++) {
				// Skip disabled TPCs
				if ((1 << j) & gpc_tpc_mask)
					continue;
				*tpc_mask |= (1ull << vtpc_idx);
				vtpc_idx++;
		}
	}
	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev[dev];
	return 0;
}

static int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev) {
	uint32_t num_gpcs;
	uint64_t* tpcs_per_gpc;
	int res;
	if (res = libsmctrl_get_gpc_info(&num_gpcs, &tpcs_per_gpc, dev))
		return res;
	*num_tpcs = 0;
	for (int gpc = 0; gpc < num_gpcs; gpc++) {
		*num_tpcs += __builtin_popcountl(tpcs_per_gpc[gpc]);
	}
	return 0;
}

// @param dev Device index as understood by CUDA **can differ from nvdebug idx**
// This implementation is fragile, and could be incorrect for odd GPUs
static int libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev) {
	int num_sms, major, minor, res = 0;
	const char* err_str;
	if (res = cuInit(0))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev))
		goto abort_cuda;
	// SM masking only works on sm_35+
	if (major < 3 || (major == 3 && minor < 5))
		return ENOTSUP;
	// Everything newer than Pascal (as of Hopper) has 2 SMs per TPC, as well
	// as the P100, which is uniquely sm_60
	int sms_per_tpc;
	if (major > 6 || (major == 6 && minor == 0))
		sms_per_tpc = 2;
	else
		sms_per_tpc = 1;
	// It looks like there may be some upcoming weirdness (TPCs with only one SM?)
	// with Hopper
	if (major >= 9)
		fprintf(stderr, "libsmctrl: WARNING, TPC masking is untested on Hopper,"
				" and will likely yield incorrect results! Proceed with caution.\n");
	*num_tpcs = num_sms/sms_per_tpc;
	return 0;
abort_cuda:
	//cuGetErrorName(res, &err_str);
	//fprintf(stderr, "libsmctrl: CUDA call failed due to %s. Failing with EIO...\n", err_str);
	return EIO;
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_N, bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
    flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_even_N, Is_even_K, Return_softmax>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_N, bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_kernel_casual(Flash_fwd_params params, const int bidb, const int bidh) {
    flash::compute_attn_casual<Kernel_traits, Is_dropout, Is_causal, Is_even_N, Is_even_K, Return_softmax>(params, bidb, bidh);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    
    //printf("fuck you smem_size = %d\n", smem_size);
    //printf("y\n");

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // We also use is_even_N to set Unpadded in the BlockInfo constructor, so we need to check
    // for cu_seqlens_q as well.
    const bool is_even_N = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_N, IsEvenNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                // Will only return softmax if dropout, to reduce compilation time.
                /* auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, IsEvenNConst, IsEvenKConst, ReturnSoftmaxConst && Is_dropout>;
                // auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, IsEvenNConst, true, ReturnSoftmaxConst && Is_dropout>;*/
                if(Is_causal) {
                    dim3 grid(num_m_block, 1, 1);
                    auto kernel = &flash_fwd_kernel_casual<Kernel_traits, Is_dropout, Is_causal, IsEvenNConst, IsEvenKConst, ReturnSoftmaxConst && Is_dropout>;
                    int ctas_per_sm;
                    //if (smem_size >= 48 * 1024) {
                    //    C10_CUDA_CHECK(cudaFuncSetAttribute(
                    //        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    //}
                    //cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //    &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

                    auto b = params.b;
                    auto h = params.h;

                    cudaStream_t streams[7];
                    uint64_t mask = 0x3full;

                    for(int i = 0; i < 7; i++) {
                        cudaStreamCreate(&streams[i]);
                        libsmctrl_set_stream_mask(streams[i], ~mask);
                        mask <<= 6;
                    }

                    for (int i = 0; i < b; i++) {
                        for (int j = 0; j < h; j++) {
                            auto now = std::chrono::high_resolution_clock::now();
                            auto duration = now.time_since_epoch();
                            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
                            printf("start time of kernel(%d, %d) is %llu\n", i, j, nanoseconds);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, streams[(i*b+j)%7]>>>(params, i, j);
                        }
                    }

                    for(int i = 0; i < 7; i++) {
                        cudaStreamSynchronize(streams[i]);
                        cudaStreamDestroy(streams[i]);
                    }
                }
                else {
                    dim3 grid(num_m_block, params.b, params.h);
                    auto kernel = &flash_fwd_kernel_casual<Kernel_traits, Is_dropout, Is_causal, IsEvenNConst, IsEvenKConst, ReturnSoftmaxConst && Is_dropout>;
                    //if (smem_size >= 48 * 1024) {
                    //    C10_CUDA_CHECK(cudaFuncSetAttribute(
                    //        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    //}
                    int ctas_per_sm;
                    //cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //    &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, 0, 0);
					//cudaStreamSynchronize(stream);
                }
                //C10_CUDA_KERNEL_LAUNCH_CHECK();  
                //printf("yyy\n");
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 32;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 64;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
                // Using block size (64 x 256) is 27% slower for seqlen=2k
                // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
}

template<typename T>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 96;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            if (is_sm8x) {
                //if constexpr(!Is_causal) {
                if (0){
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // These two are always slower
            // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 128;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
                // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
                if (is_sm8x) {
                    //if constexpr(!Is_causal) {
                    if (0){
                        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                    } else {
                        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                    }
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // 1st ones are good for H100, A100
                // 2nd one is good for A6000 bc we get slightly better occupancy
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
}

template<typename T>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 160;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // For A100, H100, 128 x 32 is the fastest.
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 64 with 8 warps is the fastest for non-causal.
            if (is_sm8x) {
                //if constexpr(!Is_causal) {
                if (0){
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 192;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 224;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)) {  // 112 KB
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // We can't do 128 x 32 with 8 warps because with headdim 224, kBlockKSmem = 32.
            // If we have N = 32, there are only 1024 elements to load at once, where each load
            // is 8 elements. This means we can only use 128 threads and not 256 threads.
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // For A100, we want to run with 128 x 64 (128KB smem).
            // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
            if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 64 KB
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 96 KB
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}
