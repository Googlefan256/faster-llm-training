use autograd::AutoTensor;
use cuda_bindings::CudaHandle;
mod autograd;
mod model;

fn main() -> anyhow::Result<()> {
    let handle = CudaHandle::new()?;
    let mut lhs = AutoTensor::new(vec![1], 2.0, &handle)?;
    lhs.backward(grad_output, &handle)?;
    lhs.drop(&handle)?;
    handle.drop()?;
    Ok(())
}
