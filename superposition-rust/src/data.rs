use dfdx::prelude::*;
use super::viz;

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub(crate) enum Experiment {
    WtW,
    DStar
}

#[derive(Debug, Clone)]
pub(crate) struct WtWResult<const F: usize> {
    pub(crate) wtw: [[f32; F]; F],
    pub(crate) bias: [f32; F],
    pub(crate) feature_norms: [f32; F],
    pub(crate) superpositions: [f32; F],
}

#[derive(Debug, Clone)]
pub(crate) struct DStarResult {
    pub(crate) sparsity: f32,
    pub(crate) dstar: f32
}

#[derive(Debug, Clone)]
pub(crate) enum ExperimentResult<const F: usize> {
    WtW(WtWResult<F>),
    DStar(DStarResult)
}
impl<const F: usize> ExperimentResult<F> {
    pub(crate) fn print_experiment(&self) {
        match self {
            ExperimentResult::WtW(wtw_result) => {
                println!("W^T x W Matrix:");
                viz::print_colored_matrix(&wtw_result.wtw);
                println!("Bias:");
                viz::print_colored_vector(&wtw_result.bias);
                println!("Feature Norms:");
                viz::print_colored_vector(&wtw_result.feature_norms);
                println!("Superposition Measure:");
                viz::print_colored_vector(&wtw_result.superpositions);
            },
            ExperimentResult::DStar(dstar_result) => {
                if dstar_result.sparsity == 1.0 {
                    print!("\nAt sparsity = {} (1/(1-S) = NaN), D* = {}\n\n", 
                        dstar_result.sparsity, 
                        dstar_result.dstar);
                } else {
                    print!("\nAt sparsity = {} (1/(1-S) = {}), D* = {}\n\n", 
                        dstar_result.sparsity, 
                        1.0 / (1.0 - dstar_result.sparsity),
                        dstar_result.dstar);
                }
            },
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum Verbosity {
    Full(usize),
    EndOnly,
    Quiet
}

pub(crate) fn generate_batch<const F: usize, const B: usize, R: rand::Rng>
    (s: f32, rng: &mut R) -> Tensor2D<B, F>{
    let mut data = Tensor2D::<B, F>::rand(rng);
    if s == 0.0 {return data}

    let mask = Tensor2D::<B, F>::rand(rng);
    for (i, arr) in mask.data().iter().enumerate() {
        for (j, el) in arr.iter().enumerate(){
            if *el <= s {
                data.mut_data()[i][j] = 0.0
            }   
        }
    }

    data
}
pub(crate) fn imp_loss<const F: usize, const B: usize, T: Tape>
    (pred: Tensor2D<B, F, T>, targ: &Tensor2D<B, F, NoneTape>, imp: Tensor1D<F>) -> Tensor0D<T>{
    let diff_sq = square(sub(pred, targ));
    let full_imp: Tensor2D<B, F> = imp.broadcast1();
    let adjusted_diff = mul(diff_sq, &full_imp);
    mean(adjusted_diff)
}

pub(crate) fn exp_importance<const F: usize>(base: f32) -> Tensor1D<F>{
    let mut importance: Tensor1D<F> = Tensor1D::new([0.0;F]);
    for i in 0..F{
        importance.mut_data()[i] = base.powf(i as f32);
    }
    importance
}

pub(crate) fn frobenius<const F: usize, const D: usize>(matrix: Tensor2D<F, D>) -> f32{
    *sqrt(sum(square(matrix))).data()
}