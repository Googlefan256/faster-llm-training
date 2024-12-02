use half::bf16;

use crate::Tensor;
use once_cell::sync::Lazy;

static SQRT_2: Lazy<bf16> = Lazy::new(|| bf16::from_f32_const(2.0_f32.sqrt()));

pub fn gelu(tensor: &Tensor) -> anyhow::Result<Tensor> {
    anyhow::bail!("a")
}
