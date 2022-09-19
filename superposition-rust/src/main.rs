use dfdx::prelude::*;
use rand::prelude::{StdRng, SeedableRng};
use lazy_pbar::pbar;

mod viz;

struct ToyModel<const F: usize, const D: usize>{
    weights: Tensor2D<F, D>,
    bias: Tensor1D<F>
}

impl<const F: usize, const D: usize> ToyModel<F, D> {
    fn new() -> ToyModel<F,D> {
        ToyModel {
            weights: Default::default(),
            bias: Default::default()
        }
    }
    fn wtw_data(&self) -> [[f32; F]; F] {
        *matmul_transpose(self.weights.duplicate(), &self.weights).data()
    }
    fn print_wtw(&self) {
        println!("W^T x W:");
        let wtw = self.wtw_data();
        pprint(&wtw);
    }
    fn w_norm(&self, i: usize) -> f32{
        let mut w_sq = 0.0;

        for x_i in self.weights.data()[i] {
            w_sq = w_sq + x_i.powi(2);
        }
        w_sq.powf(0.5)
    }
    fn all_feature_norms(&self) -> [f32; F]{
        let mut norms = [0.0; F];
        for i in 0..F {
            norms[i] = self.w_norm(i);
        }
        norms
    }
}
impl<const F: usize, const D: usize> 
    CanUpdateWithGradients for ToyModel<F, D> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.weights.update(grads, unused);
        self.bias.update(grads, unused);

    }
}
impl<const F: usize, const D: usize> 
    ResetParams for ToyModel<F,D> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.weights = Tensor2D::rand(rng);
        self.bias = Tensor1D::zeros();
    }
}
impl<const F: usize, const D: usize, H: Tape> 
    Module<Tensor1D<F, H>> for ToyModel<F, D> {
    type Output = Tensor1D<F, H>;
    fn forward(&self, input: Tensor1D<F, H>) -> Self::Output {
        let h = vecmat_mul(input, &self.weights);

        let x_p = vecmat_mul_transpose(h, &self.weights);

        let out = add(x_p, &self.bias);

        out.relu()
    }
    fn forward_mut(&mut self, input: Tensor1D<F, H>) -> Self::Output {
        self.forward(input)
    }
}
impl<const F: usize, const D: usize, const B: usize, H: Tape> 
    Module<Tensor2D<B, F, H>> for ToyModel<F, D> {
    type Output = Tensor2D<B, F, H>;
    fn forward(&self, input: Tensor2D<B, F, H>) -> Self::Output {
        let h = matmul(input, &self.weights);

        let (x_p, tape) = matmul_transpose(h, &self.weights).split_tape();
        
        let out: Self::Output = add(self.bias.duplicate().put_tape(tape).broadcast1(), &x_p);

        out.relu()
    }
    fn forward_mut(&mut self, input: Tensor2D<B, F, H>) -> Self::Output {
        self.forward(input)
    }
}

fn generate_batch<const F: usize, const B: usize, R: rand::Rng>
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
fn imp_loss<const F: usize, const B: usize, T: Tape>
    (pred: Tensor2D<B, F, T>, targ: &Tensor2D<B, F, NoneTape>, imp: Tensor1D<F>) -> Tensor0D<T>{
    let diff_sq = square(sub(pred, targ));
    let full_imp: Tensor2D<B, F> = imp.broadcast1();
    let adjusted_diff = mul(diff_sq, &full_imp);
    mean(adjusted_diff)
}
fn pprint<const F: usize, const D: usize>
    (arr: &[[f32; F]; D]){
    for row in arr.iter() {
        println!("{:.2?}", row)
    }
}

fn main() {
    const F: usize = 20; // Num features
    const D: usize = 5;  // Num dimensions
    const BATCH_SIZE: usize = 128;
    const SPARSITY: f32 = 0.7;
    const N_ITER: usize = 100_000;

    let mut importance: Tensor1D<F> = Tensor1D::new([0.0;F]);
    for i in 0..F{
        importance.mut_data()[i] = 0.7f32.powf(i as f32);
    }

    let mut rng = StdRng::seed_from_u64(0);
    let mut m = ToyModel::<F, D>::new();
    m.reset_params(&mut rng);
    
    let mut opt: Adam<ToyModel<F, D>> = Adam::new(AdamConfig {
        lr: 1e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
    });

    m.print_wtw();
    for i in pbar(0..N_ITER, N_ITER) {
        let data: Tensor2D<BATCH_SIZE, F> = generate_batch(SPARSITY, &mut rng);
        let out: Tensor2D<BATCH_SIZE, F, OwnedTape> = m.forward(data.trace());
        let loss = imp_loss(out, &data, importance.duplicate());
        let gradients: Gradients = loss.backward();
        match opt.update(&mut m, gradients) {
            Ok(()) => (),
            Err(error) => print!("{:#?}", error)
        }
        if i % 10_000 == 0 {
            println!("W^T x W Matrix at i = {}:", i);
            viz::print_colored_matrix(&m.wtw_data());
            println!("Bias:");
            viz::print_colored_vector(&m.bias.data());
            println!("Feature Norms:");
            viz::print_colored_vector(&m.all_feature_norms());
        }
    }
    
    viz::print_colored_matrix(&m.wtw_data());
    println!("Bias:");
    viz::print_colored_vector(&m.bias.data());
    println!("Feature Norms:");
    viz::print_colored_vector(&m.all_feature_norms());

}