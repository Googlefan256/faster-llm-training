use cuda_bindings::{ops, CudaHandle, Tensor};

pub enum Op<'a> {
    Add {
        lhs: &'a mut AutoTensor<'a>,
        rhs: &'a mut AutoTensor<'a>,
    },
}

impl<'a> Op<'a> {
    pub fn clone(&'a mut self) -> anyhow::Result<Self> {
        Ok(match self {
            Op::Add { lhs, rhs } => Op::Add {
                lhs: *lhs,
                rhs: *rhs,
            },
        })
    }
}

pub struct AutoTensor<'a> {
    tensor: Tensor,
    history: Option<Box<Op<'a>>>,
    grad: Option<Tensor>,
}

impl<'a> AutoTensor<'a> {
    pub fn new(shape: Vec<usize>, value: f32, handle: &CudaHandle) -> anyhow::Result<Self> {
        Ok(Self {
            grad: None,
            history: None,
            tensor: Tensor::new(shape, value, handle)?,
        })
    }
    pub fn backward(&mut self, grad_output: &Tensor, handle: &CudaHandle) -> anyhow::Result<()> {
        if let Some(op) = self.history.as_mut() {
            match op.as_mut() {
                Op::Add { lhs, rhs } => {
                    let (lhs_g, rhs_g) = ops::add_backward(&grad_output, handle)?;
                    lhs.acc_grad(&lhs_g, handle)?;
                    rhs.acc_grad(&rhs_g, handle)?;
                }
            }
        }
        Ok(())
    }

    fn acc_grad(&mut self, grad: &Tensor, handle: &CudaHandle) -> anyhow::Result<()> {
        match &mut self.grad {
            Some(g) => {
                *g = ops::add(g, grad, handle)?;
            }
            None => {
                self.grad = Some(grad.clone(handle)?);
            }
        }
        Ok(())
    }
    pub fn zero_grad(&mut self, handle: &CudaHandle) -> anyhow::Result<()> {
        if let Some(g) = self.grad.take() {
            // Set gradient to zero
            g.drop(handle)?;
            self.grad = None;
        }
        Ok(())
    }
    pub fn drop(mut self, handle: &CudaHandle) -> anyhow::Result<()> {
        self.zero_grad(handle)?;
        self.tensor.drop(handle)?;
        Ok(())
    }
    pub fn clone(&'a mut self, handle: &CudaHandle) -> anyhow::Result<Self> {
        Ok(Self {
            tensor: self.tensor.clone(handle)?,
            grad: if let Some(grad) = self.grad.as_ref() {
                Some(grad.clone(handle)?)
            } else {
                None
            },
            history: if let Some(history) = self.history.as_mut() {
                Some(Box::new(history.clone()?))
            } else {
                None
            },
        })
    }
}
