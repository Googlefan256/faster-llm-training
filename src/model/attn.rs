use core::f32;
use std::f64::consts::E;

use super::rotary::Rotary;
use candle_core::{Device, Tensor};
use candle_nn::{ops::softmax, Linear, Module, VarBuilder};

#[inline]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
) -> anyhow::Result<Tensor> {
    // Validate input tensor dimensions
    let (_batch_size, _num_heads, q_seq_len, _head_dim) = q.dims4()?;
    let (_, _, k_seq_len, _) = k.dims4()?;
    let (_, _, v_seq_len, _) = v.dims4()?;
    if k_seq_len != v_seq_len {
        return Err(anyhow::anyhow!("Key and value sequence lengths must match"));
    }
    let q = (q * scale)?;
    let attn_scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let masked_scores = if causal {
        let mask = create_causal_mask(q_seq_len, k_seq_len, q.device())?;
        mask.broadcast_as(attn_scores.shape())?.where_cond(
            &Tensor::new(f64::NEG_INFINITY, mask.device())?
                .to_dtype(attn_scores.dtype())?
                .broadcast_as(attn_scores.shape())?,
            &attn_scores,
        )?
    } else {
        attn_scores
    };
    let attn_weights = softmax(&masked_scores, 3)?;
    let output = attn_weights.contiguous()?.matmul(&v.contiguous()?)?;
    Ok(output)
}

// Helper function to create a causal mask
#[inline]
fn create_causal_mask(
    q_seq_len: usize,
    k_seq_len: usize,
    device: &Device,
) -> anyhow::Result<Tensor> {
    // Create a mask where upper triangular part is 1 (to be masked out)
    let mut mask_data = vec![0_u32; q_seq_len * k_seq_len];

    for i in 0..q_seq_len {
        for j in (i + 1)..k_seq_len {
            mask_data[i * k_seq_len + j] = 1;
        }
    }

    Ok(Tensor::from_vec(mask_data, (q_seq_len, k_seq_len), device)?)
}
pub fn rms_norm(tensor: &Tensor) -> anyhow::Result<Tensor> {
    let square_mean = tensor.sqr()?.mean_keepdim(0)?;
    let denom = (square_mean + 1e-8)?.sqrt()?;
    let normalized_tensor = tensor.broadcast_div(&denom)?;
    Ok(normalized_tensor)
}

pub struct CausalSelfAttention {
    n_head: usize,
    head_dim: usize,
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_o: Linear,
    rotary: Rotary,
    lamb: Tensor,
}

impl CausalSelfAttention {
    pub fn new(
        n_head: usize,
        n_embd: usize,
        vb: VarBuilder,
        dev: &Device,
        base: f32,
    ) -> anyhow::Result<Self> {
        let head_dim = n_embd / n_head;
        assert!(n_embd % n_head == 0, "n_embd / n_head must be zero");
        let c_q = Linear::new(vb.get(&[n_embd, n_embd], "q")?, None);
        let c_k = Linear::new(vb.get(&[n_embd, n_embd], "k")?, None);
        let c_v = Linear::new(vb.get(&[n_embd, n_embd], "v")?, None);
        let c_o = Linear::new(vb.get(&[n_embd, n_embd], "o")?, None); // 0
        let rotary = Rotary::new(head_dim as u32, dev, base)?;
        let lamb = vb.get(1, "lamb")?; // 0.5
        Ok(Self {
            n_head,
            head_dim,
            c_q,
            c_k,
            c_v,
            c_o,
            rotary,
            lamb,
        })
    }
    pub fn forward(&mut self, x: &Tensor, v1: Option<Tensor>) -> anyhow::Result<(Tensor, Tensor)> {
        let b = x.dim(0)?;
        let t = x.dim(1)?;
        let q = self
            .c_q
            .forward(x)?
            .reshape(&[b, t, self.n_head, self.head_dim])?;
        let k = self
            .c_k
            .forward(x)?
            .reshape(&[b, t, self.n_head, self.head_dim])?;
        let v = self
            .c_v
            .forward(x)?
            .reshape(&[b, t, self.n_head, self.head_dim])?;
        let v1 = if let Some(v1) = v1 { v1 } else { v.clone() };
        let v = (&v.broadcast_mul(&(1.0 - &self.lamb)?)?
            + v1.reshape(v.shape())?.broadcast_mul(&self.lamb)?)?;
        let (cos, sin) = self.rotary.forward(&q)?;
        let q = Rotary::apply_rotary_emb(&rms_norm(&q)?, &cos, &sin)?;
        let k = Rotary::apply_rotary_emb(&rms_norm(&k)?, &cos, &sin)?;
        let y = flash_attn(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            1.0 / E.sqrt(),
            true,
        )?;
        let y = y.transpose(1, 2)?.reshape(x.shape())?;
        let y = self.c_o.forward(&y)?;
        Ok((y, v1))
    }
}
