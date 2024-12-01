use candle_core::{DType, Device, Tensor};
use candle_nn::{optim::AdamW, Optimizer};
use llmrs::model::gpt::GPT;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let tok = Tokenizer::from_pretrained("openai-community/gpt2", None)
        .map_err(|e| anyhow::anyhow!(e))?;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::BF16, &dev);
    let mut gpt = GPT::new(vb, tok.get_vocab_size(true), 384, 32, 3, &dev, 10000.0)?;
    let mut optim = AdamW::new_lr(varmap.all_vars(), 1e-3)?;
    let now = std::time::Instant::now();
    for i in 0..100 {
        let idx = Tensor::new(&[[1_u32, 2, 3]], &dev)?;
        let res = gpt.forward_train(&idx)?;
        let loss: f32 = res.detach().to_dtype(DType::F32)?.to_vec0()?;
        let t = res.backward()?;
        optim.step(&t)?;
        println!("Step:{i}/Loss:{loss}/Time:{}", now.elapsed().as_millis());
    }
    Ok(())
}
