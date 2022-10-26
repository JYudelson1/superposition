use dfdx::prelude::*;
use lazy_pbar::pbar;
use rand::prelude::{StdRng, SeedableRng};

mod viz;
mod model;
mod data;
mod parsed_config;
use model::{ToyModel, TrainConfig};
use parsed_config::*;

fn main() {
    
    for s in pbar(SPARSITIES.into_iter()){
        // Reset these for each sparsity
        let importance: Tensor1D<F> = data::exp_importance(IMPORTANCE_DECAY_FACTOR);
        let mut rng = StdRng::seed_from_u64(SEED);
        let opt: Adam<ToyModel<F, D>> = Adam::default();
        

        let mut m: ToyModel<F, D> = ToyModel::new();
        m.reset_params(&mut rng);
        
        m.train_loop(
            TrainConfig::<F, D, BATCH_SIZE>::new(
                s,
                N_ITER, 
                importance, 
                opt, 
                rng, 
                EXPERIMENT,
                VERBOSITY
            )
        );
    }
}