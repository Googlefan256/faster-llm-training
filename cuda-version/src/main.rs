use cuda_bindings::{ops, CudaHandle, Tensor};

fn main() -> anyhow::Result<()> {
    let handle = CudaHandle::new()?;
    let mut lhs = Tensor::new(vec![1], 0.1, &handle)?;
    let mut rhs = Tensor::new(vec![1], 0.2, &handle)?;
    for _i in 0..1_000_00 {
        let mut tensor = ops::add(&lhs, &rhs, &handle)?;
        tensor.set_grad(Tensor::new(vec![1], 1.0, &handle)?, &handle)?;
        ops::add_backward(&tensor, &lhs, &rhs, &handle)?;
        tensor.drop(&handle)?;
        lhs.none_grad(&handle)?;
        rhs.none_grad(&handle)?;
    }
    lhs.drop(&handle)?;
    rhs.drop(&handle)?;
    handle.sync()?;
    Ok(())
}
