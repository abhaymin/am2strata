//! Step 0 gate: prove the mold candle fork can actually train on CPU — a 2-weight linear regression trained with a
//! `Var`, `backward()`, and `candle_nn::AdamW`, whose loss falls. If this fails, autograd is missing and training waits.

use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

#[test]
fn two_weight_linear_regression_loss_falls() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    // A clean linear system, y = 1.5*x1 - 0.7*x2 + 0.3, so gradient descent has a clear optimum.
    let xs = Tensor::new(&[[1f32, 1.], [1., -1.], [-1., 1.], [-1., -1.], [2., 0.5], [0.5, 2.]], &dev)?;
    let w_true = Tensor::new(&[[1.5f32], [-0.7]], &dev)?; // [2,1]
    let bias = Tensor::full(0.3_f32, (xs.dim(0)?, 1usize), &dev)?;
    let target = xs.matmul(&w_true)?.add(&bias)?; // [N,1]

    // Two trainable weights (one row): the model is just `w`.
    let w = Var::from_slice(&[0.1f32, -0.1], (1usize, 2usize), &dev)?;
    let mut params = ParamsAdamW::default();
    params.lr = 0.1;
    params.weight_decay = 0.0;
    let mut opt = AdamW::new(vec![w.clone()], params)?;

    let w_t = || w.as_tensor().clone().t().unwrap(); // the two weights transposed, [2,1]
    let mse = |pred: &Tensor| -> Result<f32, Box<dyn std::error::Error>> {
        Ok(pred.sub(&target)?.powf(2.0)?.mean_all()?.to_scalar::<f32>()?)
    };

    let initial_loss = mse(&xs.matmul(&w_t())?)? as f64;
    for _ in 0..500 {
        let pred = xs.matmul(&w_t())?;
        let loss = pred.sub(&target)?.powf(2.0)?.mean_all()?;
        opt.backward_step(&loss)?; // backward() + AdamW update, one step
    }
    let final_loss = mse(&xs.matmul(&w_t())?)? as f64;

    println!("two-weight linear regression: initial {initial_loss:.6} -> final {final_loss:.6}");
    assert!(final_loss < initial_loss, "loss did not fall: {initial_loss} -> {final_loss}");
    assert!(final_loss < 0.1, "loss did not converge to near-zero: {final_loss}");
    Ok(())
}
