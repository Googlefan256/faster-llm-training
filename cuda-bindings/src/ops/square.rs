use std::ffi::c_void;

use crate::{
    bindings::{
        cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cublasScalEx,
        cudaDataType_t,
    },
    util::check_cublas_status,
    CudaHandle, Tensor,
};

pub fn square_frontend(lhs: &Tensor, handle: &CudaHandle) -> anyhow::Result<Tensor> {
    let res = lhs.clone(handle)?;
    check_cublas_status(unsafe {
        cublasGemmEx(
            *handle.handle(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            res.size as i32,
            res.size as i32,
            res.size as i32,
            &1.0_f32 as *const _ as *const c_void,
            res.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            res.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            &0.0_f32 as *const _ as *const c_void,
            res.ptr,
            cudaDataType_t::CUDA_R_16BF,
            res.size as i32,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    })?;
    Ok(res)
}

pub fn square_backward(
    output_grad: &Tensor,
    lhs: &mut Tensor,
    handle: &CudaHandle,
) -> anyhow::Result<()> {
    let lhs_grad = Tensor::new(output_grad.shape.clone(), 0.0, handle)?;
    check_cublas_status(unsafe {
        cublasGemmEx(
            *handle.handle(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            lhs.size as i32,
            lhs.size as i32,
            lhs.size as i32,
            &1.0_f32 as *const _ as *const c_void,
            output_grad.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            lhs.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            &0.0_f32 as *const _ as *const c_void,
            lhs_grad.ptr,
            cudaDataType_t::CUDA_R_16BF,
            lhs.size as i32,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    })?;
    check_cublas_status(unsafe {
        cublasScalEx(
            *handle.handle(),
            lhs.size as i32,
            &2.0_f32 as *const _ as *const c_void,
            cudaDataType_t::CUDA_R_32F,
            lhs_grad.ptr,
            cudaDataType_t::CUDA_R_16BF,
            1,
            cudaDataType_t::CUDA_R_32F,
        )
    })?;
    lhs.set_grad(lhs_grad, handle)?;
    Ok(())
}
