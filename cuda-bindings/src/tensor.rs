use std::{ffi::c_void, mem::MaybeUninit};

use crate::{
    bindings::{cudaFreeAsync, cudaMallocAsync, cudaMemcpyAsync, cudaMemcpyKind},
    util::check_status,
    CudaHandle,
};
use half::bf16;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub(crate) ptr: *mut c_void,
    pub(crate) size: usize,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, value: f32, stream: &CudaHandle) -> anyhow::Result<Self> {
        let size = shape.iter().fold(1, |res, a| res * a);
        let mut ptr = MaybeUninit::uninit();
        let data = vec![bf16::from_f32_const(value); size];
        check_status(unsafe {
            cudaMallocAsync(ptr.as_mut_ptr(), size_of::<bf16>() * size, *stream.stream())
        })?;
        let ptr = unsafe { ptr.assume_init() };
        check_status(unsafe {
            cudaMemcpyAsync(
                ptr,
                data.as_ptr() as *mut c_void,
                size_of::<bf16>() * size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                *stream.stream(),
            )
        })?;
        Ok(Self { shape, ptr, size })
    }
    pub fn drop(self, stream: &CudaHandle) -> anyhow::Result<()> {
        check_status(unsafe { cudaFreeAsync(self.ptr as *mut c_void, *stream.stream()) })?;
        Ok(())
    }
    pub fn to_vec(&self, stream: &CudaHandle) -> anyhow::Result<Vec<bf16>> {
        let mut vec = vec![bf16::from_f32(0.0); self.size];
        check_status(unsafe {
            cudaMemcpyAsync(
                vec.as_mut_ptr() as *mut c_void,
                self.ptr,
                size_of::<bf16>() * self.size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                *stream.stream(),
            )
        })?;
        Ok(vec)
    }
    pub fn clone(&self, stream: &CudaHandle) -> anyhow::Result<Self> {
        let mut ptr = MaybeUninit::uninit();
        check_status(unsafe {
            cudaMallocAsync(
                ptr.as_mut_ptr(),
                size_of::<bf16>() * self.size,
                *stream.stream(),
            )
        })?;
        let ptr = unsafe { ptr.assume_init() };
        check_status(unsafe {
            cudaMemcpyAsync(
                ptr,
                self.ptr,
                size_of::<bf16>() * self.size,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                *stream.stream(),
            )
        })?;
        Ok(Self {
            shape: self.shape.clone(),
            ptr,
            size: self.size.clone(),
        })
    }
}
