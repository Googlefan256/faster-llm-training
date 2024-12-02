use cuda_bindings::{ops, CudaHandle, Tensor};

fn main() -> anyhow::Result<()> {
    let handle = CudaHandle::new()?;
    let mut lhs = Tensor::new(vec![1], 2.0, &handle)?;
    let tensor = ops::sqrt_frontend(&lhs, &handle)?;
    ops::sqrt_backward(&Tensor::new(vec![1], 1.0, &handle)?, &mut lhs, &handle)?;
    println!("{:?}", tensor.to_vec(&handle)?);
    tensor.drop(&handle)?;
    lhs.drop(&handle)?;
    handle.sync()?;
    Ok(())
}
