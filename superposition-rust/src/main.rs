use dfdx::prelude::*;
use rand::prelude::*;

struct ToyModel<const F: usize, const D: usize>{
    W: Tensor2D<F, D>,
    bias: Tensor1D<F>
}

impl<const F: usize, const D: usize> ToyModel<F, D> {
    fn new<R: rand::Rng>(rng: &mut R) -> ToyModel<F,D> {
        ToyModel {
            W: Tensor2D::rand(rng),
            bias: Tensor1D::zeros()
        }
    }

    fn forward<const n: usize>(self, x: Tensor2D<n, F, OwnedTape>) -> Tensor2D<n, F, OwnedTape> {
        let h: Tensor2D<n, D, OwnedTape> = matmul(x, &self.W);

        let x_p: Tensor2D<n, F, OwnedTape> = matmul_transpose(h, &self.W);
        let bias: Tensor2D<n, F> = self.bias.broadcast1();

        let out = add(x_p, &bias);

        out.relu()
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let m = ToyModel::<3, 2>::new(&mut rng);

    let data = Tensor2D::<1,3>::new([[1.0,2.0,3.0]]).traced();
    let out = m.forward(data);
    print!("{:#?}", out.data());
}
