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

pub fn sqrt_frontend(input: &Tensor, handle: &CudaHandle) -> anyhow::Result<Tensor> {
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

pub fn sqrt_backward(
    output_grad: &Tensor,
    input: &mut Tensor,
    handle: &CudaHandle,
) -> anyhow::Result<()> {
    // The gradient of sqrt(x) is 1/(2*sqrt(x))
    let temp_tensor = Tensor::new(input.shape.clone(), 0.0, handle)?;
    sqrt_frontend(input, handle)?; // Compute sqrt(input) and store in temp_tensor
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            input.size as i32,
            &2.0_f32 as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            temp_tensor.ptr,
            cudaDataType_t::CUDA_R_32F,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            input.size as i32,
            &1.0_f32 as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            temp_tensor.ptr,
            cudaDataType_t::CUDA_R_32F,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            input.size as i32,
            &1.0_f32 as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            output_grad.ptr,
            cudaDataType_t::CUDA_R_32F,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    let input_grad = output_grad.clone(handle)?;
    check_cublas_status(unsafe {
        cublasGemmEx(
            *handle.handle(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            input.size as i32,
            1,
            1,
            &1.0_f32 as *const _ as *const c_void,
            output_grad.ptr,
            cudaDataType_t::CUDA_R_32F,
            1,
            temp_tensor.ptr,
            cudaDataType_t::CUDA_R_32F,
            1,
            &0.0_f32 as *const _ as *const c_void,
            input_grad.ptr,
            cudaDataType_t::CUDA_R_32F,
            input.size as i32,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    })?;
    input.set_grad(input_grad, handle)?;
    Ok(())
}
