use dfdx::prelude::*;
use lazy_pbar::pbar;
use rand::prelude::StdRng;

use super::data;
use super::viz;

#[derive(Debug, Clone)]
pub(crate) struct ToyModel<const F: usize, const D: usize>{
    weights: Tensor2D<F, D>,
    bias: Tensor1D<F>
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Experiment {
    WtW,
    DStar
}

#[derive(Debug)]
pub(crate) struct TrainConfig<const F: usize, const D: usize, const B: usize>{
    sparsity: f32,
    n_iter: usize,
    importance: Tensor1D<F>,
    opt: Adam<ToyModel<F, D>>,
    rng: StdRng,
    experiment: Experiment
}

impl<const F: usize, const D: usize> ToyModel<F, D> {
    pub(crate) fn new() -> ToyModel<F,D> {
        ToyModel {
            weights: Default::default(),
            bias: Default::default()
        }
    }
    fn wtw_data(&self) -> [[f32; F]; F] {
        *matmul_transpose(self.weights.duplicate(), &self.weights).data()
    }
    fn w_norm(&self, i: usize) -> f32{
        let feature: Tensor1D<D, NoneTape> = self.weights.duplicate().select(&i);
        *sqrt(sum(square(feature))).data()
    }
    fn all_feature_norms(&self) -> [f32; F]{
        let mut norms = [0.0; F];
        for i in 0..F {
            norms[i] = self.w_norm(i);
        }
        norms
    }
    fn get_superposition(&self, i: usize) -> f32{
        //TODO: Make sure this is working right
        let mut s = 0.0;
        for j in 0..F {
            if i==j {continue};
            let i_j_dot_product: Tensor1D<D> = mul(
                self.weights.duplicate().select(&i), 
                &self.weights.duplicate().select(&j));
            s = s + sum(square(i_j_dot_product)).data();
        }
        s
    }
    fn get_all_superposiiton(&self) -> [f32; F]{
        let mut superpositions = [0.0; F];
        for i in 0..F {
            superpositions[i] = self.get_superposition(i);
        }
        superpositions
    }
    fn d_star(&self) -> f32 {
        (D as f32) / (data::frobenius(self.weights.duplicate()))
    }
    fn perform_experiment<const B: usize>(&self, config: &TrainConfig<F, D, B>){
        match config.experiment {
            Experiment::WtW => {
                println!("W^T x W Matrix:");
                viz::print_colored_matrix(&self.wtw_data());
                println!("Bias:");
                viz::print_colored_vector(self.bias.data());
                println!("Feature Norms:");
                viz::print_colored_vector(&self.all_feature_norms());
                println!("Superposition Measure:");
                viz::print_colored_vector(&self.get_all_superposiiton());
            },
            Experiment::DStar => {
                print!("\nAt sparsity = {} (1/(1-S) = {}), D* = {}\n\n", 
                        config.sparsity, 
                        1.0 / (1.0 - config.sparsity),
                        self.d_star());
            }
        }
    }
    pub(crate) fn train_loop<const B: usize>(&mut self, mut config: TrainConfig<F, D, B>){
        for i in pbar(0..config.n_iter, config.n_iter) {
            let data: Tensor2D<B, F> = data::generate_batch(config.sparsity, &mut config.rng);
            let out: Tensor2D<B, F, OwnedTape> = self.forward(data.trace());
            let loss = data::imp_loss(out, &data, config.importance.duplicate());
            let gradients: Gradients = loss.backward();
            match config.opt.update(self, gradients) {
                Ok(()) => (),
                Err(error) => print!("{:#?}", error)
            }
            if i % 10_000 == 0 {
                self.perform_experiment(&config);
            }
        }
        self.perform_experiment(&config);
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

impl<const F: usize, const D: usize, const B: usize> TrainConfig<F, D, B> {
    pub(crate) fn new(sparsity: f32,
                      n_iter: usize,
                      importance: Tensor1D<F>,
                      opt: Adam<ToyModel<F, D>>,
                      rng: StdRng,
                      experiment: Experiment) -> TrainConfig<F, D, B> {
        TrainConfig { 
            sparsity: sparsity, 
            n_iter: n_iter, 
            importance: importance, 
            opt: opt, 
            rng: rng,
            experiment: experiment }
    }
}