use std::ffi::c_void;

use crate::{
    bindings::{cublasAxpyEx, cudaDataType_t},
    util::check_cublas_status,
    CudaHandle, Tensor,
};

pub fn add(lhs: &Tensor, rhs: &Tensor, handle: &CudaHandle) -> anyhow::Result<Tensor> {
    let res = rhs.clone(handle)?;
    check_cublas_status(unsafe {
        cublasAxpyEx(
            *handle.handle(),
            lhs.size as i32,
            &1.0_f32 as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            lhs.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            res.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    Ok(res)
}

pub fn add_backward(
    output_grad: &Tensor,
    lhs: &Tensor,
    rhs: &Tensor,
    handle: &CudaHandle,
) -> anyhow::Result<()> {
    if let Some(grad) = lhs.grad.as_ref() {
        check_cublas_status(unsafe {
            cublasAxpyEx(
                *handle.handle(),
                grad.size as i32,
                &1.0_f32 as *const _ as *const c_void,
                cudaDataType_t::CUDA_R_32F,
                output_grad.ptr,
                cudaDataType_t::CUDA_R_16BF,
                1,
                grad.ptr,
                cudaDataType_t::CUDA_R_16BF,
                1,
                cudaDataType_t::CUDA_R_32F,
            )
        })?;
    }
    if let Some(grad) = rhs.grad.as_ref() {
        check_cublas_status(unsafe {
            cublasAxpyEx(
                *handle.handle(),
                grad.size as i32,
                &1.0_f32 as *const _ as *const c_void,
                cudaDataType_t::CUDA_R_32F,
                output_grad.ptr,
                cudaDataType_t::CUDA_R_16BF,
                1,
                grad.ptr,
                cudaDataType_t::CUDA_R_16BF,
                1,
                cudaDataType_t::CUDA_R_32F,
            )
        })?;
    }

    Ok(())
}
