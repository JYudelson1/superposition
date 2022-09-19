use dfdx::prelude::*;

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
    *sum(square(matrix)).data()
}