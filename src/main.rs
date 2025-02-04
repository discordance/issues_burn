use burn::{backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, optim::AdamConfig, tensor::backend::AutodiffBackend};

mod example;

fn main() {
    type Backend = burn_ndarray::NdArray<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    let device = burn_ndarray::NdArrayDevice::default();
    
    let artifact_dir = "/tmp/guide";

    // Create the training configuration.
    let training_config = example::TrainingConfig {
        model: example::SimpleModelConfig {
            in_features: 2,
            out_features: 1,
        },
        optimizer: AdamConfig::new(),
        num_epochs: 100,
        batch_size: 1,
        num_workers: 1,
        seed: 42,
        learning_rate: 1.0e-4,
    };

    // Call the train function with the autodiff backend.
    example::train::<AutodiffBackend>(artifact_dir, training_config, device);
}
