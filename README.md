## Replicating "Toy Models of Superposition"

Anthropic Paper is [here](https://transformer-circuits.pub/2022/toy_model/index.html).

Working through the paper, just to make sure I get it.

Todos:
- [ ] Write the model (W/ different sparsities)
  - [x] Building model
  - [ ] Optimizer + train code (taking into acct importance)
- [ ] Random data generation (taking into acct sparsity)
- [ ] Run through some of the experiments, esp:
  - [ ] Phase diagrams
  - [ ] W<sup>T</sup>W at different sparsities
  - [ ] Sparsity vs dimensions/feature ratio
- [ ] Figure out how to make those pretty graphs
- [ ] (Maybe) Replicate in Rust, using dfdx (nice bc statically typed tensor sizes, mmmm)