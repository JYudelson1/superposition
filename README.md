## Replicating "Toy Models of Superposition"

Anthropic Paper is [here](https://transformer-circuits.pub/2022/toy_model/index.html).

Working through the paper, just to make sure I get it.

Todos:
- [x] Write the model (W/ different sparsities)
  - [x] Building model
  - [x] Optimizer + train code (taking into acct importance)
- [x] Random data generation (taking into acct sparsity)
- [ ] Run through some of the experiments, esp:
  - [ ] Phase diagrams
  - [x] W<sup>T</sup>W at different sparsities
  - [ ] Sparsity vs dimensions/feature ratio
- [ ] Figure out how to make those pretty graphs
- [x] (Maybe) Replicate in Rust, using dfdx (nice bc statically typed tensor sizes, mmmm)

**EDIT**: Moving primarily to Rust, since statically sized tensors are your friend.