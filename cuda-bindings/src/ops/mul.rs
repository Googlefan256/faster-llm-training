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
