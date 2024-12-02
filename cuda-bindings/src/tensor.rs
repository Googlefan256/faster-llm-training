use std::{ffi::c_void, mem::MaybeUninit};

use crate::{
    bindings::{cudaFreeAsync, cudaMallocAsync, cudaMemcpyAsync, cudaMemcpyKind},
    util::check_status,
    CudaHandle,
};
use half::bf16;

pub struct Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) ptr: *mut c_void,
    pub(crate) grad: Box<Option<Tensor>>,
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
        Ok(Self {
            shape,
            ptr,
            size,
            grad: Box::new(None),
        })
    }
    pub fn drop(mut self, stream: &CudaHandle) -> anyhow::Result<()> {
        check_status(unsafe { cudaFreeAsync(self.ptr as *mut c_void, *stream.stream()) })?;
        if let Some(grad) = self.grad.take() {
            grad.drop(stream)?;
        }
        Ok(())
    }
    pub fn to_vec(&self, stream: &CudaHandle) -> anyhow::Result<(Vec<bf16>, Option<Vec<bf16>>)> {
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
        let grad_vec = if let Some(grad) = self.grad.as_ref() {
            Some(grad.to_vec(stream)?.0)
        } else {
            None
        };
        Ok((vec, grad_vec))
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
            grad: Box::new(if let Some(grad) = self.grad.as_ref() {
                Some(grad.clone(stream)?)
            } else {
                None
            }),
        })
    }
    pub fn set_grad(&mut self, grad: Tensor, stream: &CudaHandle) -> anyhow::Result<()> {
        if let Some(grad) = self.grad.take() {
            grad.drop(stream)?;
        }
        self.grad = Box::new(Some(grad));
        Ok(())
    }
    pub fn none_grad(&mut self, stream: &CudaHandle) -> anyhow::Result<()> {
        if let Some(grad) = self.grad.take() {
            grad.drop(stream)?;
        }
        self.grad = Box::new(None);
        Ok(())
    }
}
