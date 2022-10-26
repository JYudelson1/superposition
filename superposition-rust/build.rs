use std::fs;
use std::path::Path;
use toml::Value;

fn main() {
    let dest_path = Path::new("src/parsed_config.rs");
    let config_path = Path::new("config.toml");
    let config: Value = fs::read_to_string(config_path)
                        .expect("config.toml doesn't exist!")
                        .parse::<Value>()
                        .unwrap();

    //let mut parsed_consts = String::new();

    let verbosity: String = if config["verbosity"]["print_every_n_steps"] == Value::Boolean(true) {
        format!("Full({})", config["verbosity"]["n_steps"])
    } else if config["verbosity"]["print_at_end"] == Value::Boolean(true) {
        "EndOnly".to_string()
    } else {
        "Quiet".to_string()
    };

    let sparsity_len =  match &config["experiment"]["sparsities"] {
        Value::Float(_) => 1,
        Value::Array(arr) => arr.len(),
        _ => 0
    };
   

    let parsed_consts = format!("
use super::data::Experiment;
use super::data::Verbosity;

pub(crate) const F: usize = {}; // Num features
pub(crate) const D: usize = {};  // Num dimensions
pub(crate) const BATCH_SIZE: usize = {};
pub(crate) const SPARSITIES: [f32; {}] = {};
pub(crate) const N_ITER: usize = {};
pub(crate) const SEED: u64 = {};
pub(crate) const EXPERIMENT: Experiment = Experiment::{};
pub(crate) const VERBOSITY: Verbosity = Verbosity::{};
pub(crate) const IMPORTANCE_DECAY_FACTOR: f32 = {};
    ",
    config["experiment"]["num_features"],
    config["experiment"]["num_dimensions"],
    config["training"]["batch_size"],
    sparsity_len,
    config["experiment"]["sparsities"],
    config["training"]["epochs"],
    config["training"]["randseed"],
    config["experiment"]["experiment_name"].as_str().unwrap(),
    verbosity,
    config["experiment"]["importance_decay"]
);


    fs::write(&dest_path, parsed_consts).unwrap();
    
    println!("cargo:rerun-if-changed=config.toml");
}