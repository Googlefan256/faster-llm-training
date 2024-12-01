use candle_core::{DType, Device, Tensor};

#[inline]
fn outer(lhs: &Tensor, rhs: &Tensor) -> anyhow::Result<Tensor> {
    if lhs.dims().len() != 1 || rhs.dims().len() != 1 {
        anyhow::bail!("Both tensors must be 1-dimensional");
    }
    let lhs_reshaped = lhs.unsqueeze(1)?;
    let rhs_reshaped = rhs.unsqueeze(0)?;
    let result = lhs_reshaped.matmul(&rhs_reshaped)?;
    Ok(result)
}

pub struct Rotary {
    inv_freq: Tensor,
    seq_len_cached: Option<usize>,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
}

impl Rotary {
    pub fn new(dim: u32, dev: &Device, base: f32) -> anyhow::Result<Self> {
        let ar = (Tensor::arange_step(0, dim, 2, dev)?.to_dtype(DType::F32)? / (dim as f64))?;
        Ok(Self {
            inv_freq: (1.0_f64 / Tensor::full(base, ar.shape(), dev)?.pow(&ar)?)?,
            seq_len_cached: None,
            cos_cached: None,
            sin_cached: None,
        })
    }
    pub fn forward(&mut self, x: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let seq_len = x.shape().dim(1)?;
        if Some(seq_len) != self.seq_len_cached {
            self.seq_len_cached = Some(seq_len);
            let t =
                Tensor::arange(0, seq_len as u32, x.device())?.to_dtype(self.inv_freq.dtype())?;
            let freqs = outer(&t, &self.inv_freq)?;
            self.cos_cached = Some(freqs.cos()?.to_dtype(x.dtype())?);
            self.sin_cached = Some(freqs.sin()?.to_dtype(x.dtype())?);
        };
        return Ok((
            self.cos_cached
                .as_ref()
                .unwrap()
                .unsqueeze(0)?
                .unsqueeze(2)?,
            self.sin_cached
                .as_ref()
                .unwrap()
                .unsqueeze(0)?
                .unsqueeze(2)?,
        ));
    }
    #[inline]
    pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> anyhow::Result<Tensor> {
        // Ensure x is a 4D tensor
        assert_eq!(x.dims().len(), 4, "Input must be a 4D tensor");

        let d = x.dim(3)? / 2;

        // Split the last dimension into two halves
        let x1 = x.narrow(3, 0, d)?;
        let x2 = x.narrow(3, d, d)?;
        // Apply rotary embedding transformations
        let y1 = (x1.broadcast_mul(cos)? + x2.broadcast_mul(sin)?)?;
        let y2 = (x1.broadcast_mul(&sin.neg()?)? + x2.broadcast_mul(cos)?)?;

        // Concatenate the transformed halves
        Ok(Tensor::cat(&[y1, y2], 3)?)
    }
}
