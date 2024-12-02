use crate::bindings::{cublasStatus_t, cudaError_enum, cudaError_t};

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

pub fn check_cu_status(status: cudaError_enum) -> anyhow::Result<()> {
    if status != cudaError_enum::CUDA_SUCCESS {
        unsafe {
            let mut str = std::mem::MaybeUninit::uninit();
            crate::bindings::cuGetErrorString(status, str.as_mut_ptr());
            println!(
                "{:?}",
                std::ffi::CString::from_raw(str.assume_init() as *mut i8)
            );
        };
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}
