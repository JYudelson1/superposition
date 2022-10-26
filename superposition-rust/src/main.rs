use dfdx::prelude::*;
use lazy_pbar::pbar;
use rand::prelude::{StdRng, SeedableRng};

mod viz;
mod model;
mod data;
use model::{ToyModel, TrainConfig};
use data::{Verbosity, Experiment};

fn main() {
    const F: usize = 100; // Num features
    const D: usize = 7;  // Num dimensions
    const BATCH_SIZE: usize = 1024;
    const SPARSITY: f32 = 0.0;
    const N_ITER: usize = 1_000_000;
    //TODO: Turn this into a congif.toml file
    
    for s in pbar((0..100).map(|i| i as f32 / 100.0), 100){    
        let importance: Tensor1D<F> = data::exp_importance(1.0);
        let mut rng = StdRng::seed_from_u64(0);
        let opt: Adam<ToyModel<F, D>> = Adam::new(AdamConfig {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
        });
        let experiment = Experiment::DStar;

        let mut m: ToyModel<F, D> = ToyModel::new();
        m.reset_params(&mut rng);
        
        m.train_loop(
            TrainConfig::<F, D, BATCH_SIZE>::new(
                s,
                N_ITER, 
                importance, 
                opt, 
                rng, 
                experiment,
                Verbosity::Quiet
            )
        );
    }
    
}