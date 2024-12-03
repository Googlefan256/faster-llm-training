use cuda_bindings::{ops, CudaHandle, Tensor};

pub fn rms_norm(tensor: &Tensor, handle: &CudaHandle) -> anyhow::Result<Tensor> {
    Ok(tensor.clone(handle)?)
}
