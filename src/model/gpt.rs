use candle_core::{Device, Tensor};
use candle_nn::{embedding, linear, loss::cross_entropy, Embedding, Linear, Module, VarBuilder};

use super::{attn::rms_norm, block::Block};

pub struct GPT {
    wte: Embedding,
    h: Vec<Block>,
    skip_weights: Vec<Tensor>,
    lm_head: Linear, // 0
    enc_layer: usize,
    dec_layer: usize,
}

impl GPT {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        n_embd: usize,
        n_layer: usize,
        n_head: usize,
        dev: &Device,
        base: f32,
    ) -> anyhow::Result<Self> {
        let mut h = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            h.push(Block::new(vb.pp("layer").pp(i), n_head, n_embd, dev, base)?);
        }
        let enc_layer = n_layer / 2;
        let dec_layer = n_layer - enc_layer;
        let mut skip_weights = Vec::with_capacity(dec_layer);
        for i in 0..dec_layer {
            skip_weights.push(vb.pp("skip_weights").get(1, &i.to_string())?)
        }
        Ok(Self {
            wte: embedding(vocab_size, n_embd, vb.pp("wte"))?,
            h,
            skip_weights,
            lm_head: linear(n_embd, vocab_size, vb.pp("lm_head"))?,
            enc_layer,
            dec_layer,
        })
    }
    fn forward_inner(&mut self, idx: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = rms_norm(&self.wte.forward(&idx)?)?;
        let x0 = x.clone();
        let mut v1 = None;
        let mut skip_connections = Vec::with_capacity(self.enc_layer);
        for i in 0..self.enc_layer {
            (x, v1) = self.h[i].forward(x, v1, &x0)?;
            skip_connections.push(x.clone());
        }
        for i in 0..self.dec_layer {
            let skip = skip_connections.pop().unwrap();
            let weighted_skip = skip.broadcast_mul(&self.skip_weights[i])?;
            (x, v1) = self.h[self.enc_layer + i].forward((x + weighted_skip)?, v1, &x0)?;
        }
        let x = rms_norm(&x)?;
        Ok(x)
    }
    pub fn forward_infer(&mut self, idx: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.forward_inner(idx)?;
        let logits = self.lm_head.forward(&x)?;
        let logits = (30.0 * (logits / 30.0)?.tanh()?)?;
        Ok(logits)
    }
    pub fn forward_train(&mut self, idx: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.forward_inner(idx)?;
        let logits = self.lm_head.forward(&x)?;
        let logits = (30.0 * (logits / 30.0)?.tanh()?)?;
        let shifted_logits = logits.narrow(1, 0, logits.dim(1)? - 1)?.flatten_to(1)?;
        let shifted_targets = idx.narrow(1, 1, idx.dim(1)? - 1)?.flatten_to(1)?;
        Ok(cross_entropy(&shifted_logits, &shifted_targets)?)
    }
}
