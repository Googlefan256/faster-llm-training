use std::ffi::c_void;

use half::bf16;

use crate::{
    bindings::{cublasDotEx, cublasScalEx, cudaDataType_t},
    util::check_cublas_status,
    CudaHandle, Tensor,
};

pub fn mul_frontend(lhs: &Tensor, rhs: f32, handle: &CudaHandle) -> anyhow::Result<Tensor> {
    let res = lhs.clone(handle)?;
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            lhs.size as i32,
            &rhs as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            res.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    Ok(res)
}
pub fn mul_backward(
    output_grad: &Tensor,
    lhs: &mut Tensor,
    rhs: f32,
    handle: &CudaHandle,
) -> anyhow::Result<bf16> {
    // Allocate memory for grad_lhs
    let lhs_grad = output_grad.clone(handle)?;

    // Compute grad_lhs = grad_res * rhs
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            lhs_grad.size as i32,
            &rhs as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            lhs_grad.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;

    // Compute grad_rhs = sum(grad_res * lhs)
    let mut rhs_grad = bf16::from_f32_const(0.0);
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
            &mut rhs_grad as *mut _ as *mut c_void,
            cudaDataType_t::CUDA_R_16BF,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    lhs.set_grad(lhs_grad, handle)?;
    Ok(rhs_grad)
}
