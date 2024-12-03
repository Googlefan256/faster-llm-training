use std::ffi::{c_int, c_void};

use half::bf16;

use crate::{
    bindings::{
        cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cublasScalEx,
        cudaDataType_t, cudaLaunchKernel, dim3,
    },
    util::{check_cublas_status, check_status},
    CudaHandle, Tensor,
};

extern "C" {
    fn sqrt_forward_kernel(output: *mut bf16, input: *const bf16, size: c_int) -> c_void;
}

pub fn sqrt(input: &Tensor, handle: &CudaHandle) -> anyhow::Result<Tensor> {
    let res = input.clone(handle)?;
    let block_size = 256;
    let grid_size = (res.size as u32 + block_size - 1) / block_size;
    check_status(unsafe {
        cudaLaunchKernel(
            sqrt_forward_kernel as *const c_void,
            dim3 {
                x: grid_size,
                y: 1,
                z: 1,
            },
            dim3 {
                x: block_size,
                y: 1,
                z: 1,
            },
            [
                &mut (res.ptr as *mut bf16) as *mut *mut bf16 as *mut c_void,
                &mut (input.ptr as *mut bf16) as *mut *mut bf16 as *mut c_void,
                &(res.size as i32) as *const c_int as *mut c_int as *mut c_void,
            ]
            .as_mut_ptr(),
            0,
            *handle.stream(),
        )
    })?;
    Ok(res)
}
