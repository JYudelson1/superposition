use dfdx::prelude::*;
use lazy_pbar::pbar;
use rand::prelude::{StdRng, SeedableRng};

mod viz;
mod model;
mod data;
use model::{ToyModel, TrainConfig};

fn main() {
    const F: usize = 400; // Num features
    const D: usize = 30;  // Num dimensions
    const BATCH_SIZE: usize = 128;
    const SPARSITY: f32 = 0.0;
    const N_ITER: usize = 50_000;
    
    for s in pbar((0..100).map(|i| i as f32 / 100.0), 100){    
        let importance: Tensor1D<F> = data::exp_importance(0.7);
        let mut rng = StdRng::seed_from_u64(0);
        let opt: Adam<ToyModel<F, D>> = Adam::new(AdamConfig {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
        });
        let experiment = model::Experiment::DStar;

        let mut m = ToyModel::<F, D>::new();
        m.reset_params(&mut rng);
        
        m.train_loop(
            TrainConfig::<F, D, BATCH_SIZE>::new(
                s, N_ITER, importance, opt, rng, experiment
            )
        );
    }
    
}