use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use super::attn::{rms_norm, CausalSelfAttention};
use super::mlp::MLP;

pub struct Block {
    attn: CausalSelfAttention,
    mlp: MLP,
    lambdas_1: Tensor,
    lambdas_2: Tensor,
}

impl Block {
    pub fn new(
        vb: VarBuilder,
        n_head: usize,
        n_embd: usize,
        dev: &Device,
        base: f32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn: CausalSelfAttention::new(n_head, n_embd, vb.pp("attn"), dev, base)?,
            mlp: MLP::new(n_embd, vb.pp("mlp"))?,
            lambdas_1: vb.get(1, "l_1")?, // 1.0
            lambdas_2: vb.get(1, "l_2")?, // 0.0
        })
    }
    pub fn forward(
        &mut self,
        x: Tensor,
        v1: Option<Tensor>,
        x0: &Tensor,
    ) -> anyhow::Result<(Tensor, Option<Tensor>)> {
        let x = (x.broadcast_mul(&self.lambdas_1)? + x0.broadcast_mul(&self.lambdas_2)?)?;
        let (x1, v1) = self.attn.forward(&rms_norm(&x)?, v1)?;
        let x = (x + x1)?;
        let x = (&x + self.mlp.forward(&rms_norm(&x)?)?)?;
        Ok((x, Some(v1)))
    }
}
