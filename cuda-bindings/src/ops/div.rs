use std::ffi::c_void;

use half::bf16;

use crate::{
    bindings::{cublasDotEx, cublasScalEx, cudaDataType_t},
    util::check_cublas_status,
    CudaHandle, Tensor,
};

pub fn div_frontend(lhs: &Tensor, rhs: f32, handle: &CudaHandle) -> anyhow::Result<Tensor> {
    let res = lhs.clone(handle)?;
    let reciprocal_rhs = 1.0f32 / rhs;
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            lhs.size as i32,
            &reciprocal_rhs as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            res.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    Ok(res)
}

pub fn div_backward(
    output_grad: &Tensor,
    lhs: &mut Tensor,
    rhs: f32,
    handle: &CudaHandle,
) -> anyhow::Result<bf16> {
    // Compute grad_lhs = grad_output / rhs
    let grad_lhs = output_grad.clone(handle)?;
    let reciprocal_rhs = 1.0f32 / rhs;
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            grad_lhs.size as i32,
            &reciprocal_rhs as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            grad_lhs.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;

    // Compute grad_rhs = - (lhs * grad_output) / (rhs * rhs)
    let mut dot_product = bf16::from_f32_const(0.0);
    check_cublas_status(unsafe {
        cublasDotEx(
            *handle.handle(),
            lhs.size as i32,
            lhs.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            output_grad.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            &mut dot_product as *mut _ as *mut c_void,
            cudaDataType_t::CUDA_R_16BF,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    let rhs_squared = rhs * rhs;
    let scale = -1.0f32 / rhs_squared;
    let grad_rhs = bf16::from_f32(dot_product.to_f32() * scale);
    lhs.set_grad(grad_lhs, handle)?;
    Ok(grad_rhs)
}
