use std::mem::MaybeUninit;

use crate::{
    bindings::{
        cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cudaStreamCreate, cudaStreamDestroy,
        cudaStreamSynchronize, cudaStream_t,
    },
    util::{check_cublas_status, check_status},
};

pub struct CudaHandle {
    stream: cudaStream_t,
    handle: cublasHandle_t,
}

impl CudaHandle {
    pub fn new() -> anyhow::Result<Self> {
        let stream = unsafe {
            let mut stream = MaybeUninit::uninit();
            check_status(cudaStreamCreate(stream.as_mut_ptr()))?;
            stream.assume_init()
        };
        let handle = unsafe {
            let mut handle = MaybeUninit::uninit();
            check_cublas_status(cublasCreate_v2(handle.as_mut_ptr()))?;
            handle.assume_init()
        };
        Ok(Self { stream, handle })
    }
    #[inline]
    pub(crate) fn stream(&self) -> &cudaStream_t {
        &self.stream
    }
    pub(crate) fn handle(&self) -> &cublasHandle_t {
        &self.handle
    }
    pub fn drop(self) -> anyhow::Result<()> {
        check_status(unsafe { cudaStreamDestroy(self.stream) })?;
        check_cublas_status(unsafe { cublasDestroy_v2(self.handle) })?;
        Ok(())
    }
    pub fn sync(&self) -> anyhow::Result<()> {
        check_status(unsafe { cudaStreamSynchronize(self.stream) })?;
        Ok(())
    }
}
