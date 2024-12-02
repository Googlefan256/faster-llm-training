use crate::bindings::{cublasStatus_t, cudaError_t};

pub fn check_status(status: cudaError_t) -> anyhow::Result<()> {
    if status != cudaError_t::cudaSuccess {
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}

pub fn check_cublas_status(status: cublasStatus_t) -> anyhow::Result<()> {
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}
