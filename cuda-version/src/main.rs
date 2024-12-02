use cuda_bindings::{ops, CudaHandle, Tensor};

fn main() -> anyhow::Result<()> {
    let handle = CudaHandle::new()?;
    let mut lhs = Tensor::new(vec![1], 0.1, &handle)?;
    let tensor = ops::mul_frontend(&lhs, 0.2, &handle)?;
    let grad = ops::mul_backward(&Tensor::new(vec![1], 1.0, &handle)?, &mut lhs, 0.2, &handle)?;
    println!("{:?}/{}", lhs.to_vec(&handle)?, grad);
    tensor.drop(&handle)?;
    lhs.drop(&handle)?;
    handle.sync()?;
    handle.drop()?;
    Ok(())
}
