use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

pub struct MLP {
    pub c_fc: Linear,
    pub c_proj: Linear,
    pub n_embd: usize,
}

impl MLP {
    pub fn new(n_embd: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        Ok(Self {
            c_fc: Linear::new(vb.get(&[4 * n_embd, n_embd], "c_fc")?, None),
            c_proj: Linear::new(vb.get(&[n_embd, 4 * n_embd], "c_proj")?, None), // 0
            n_embd,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.relu()?.sqr()?;
        let x = self.c_proj.forward(&x)?;
        Ok(x)
    }
}
