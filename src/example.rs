use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use burn::{
    config::Config,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    module::Module,
    nn::{Linear, LinearConfig},
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{
        metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

/// A simple model with a single linear layer for sequence regression.
/// This model processes 3D tensors of shape `[batch, timesteps, in_features]`.
#[derive(Module, Debug)]
pub struct SimpleModel<B: Backend> {
    /// Single linear layer.
    pub linear: Linear<B>,
    /// Number of output features.
    pub out_features: usize,
}

impl<B: Backend> SimpleModel<B> {
    /// Executes a forward pass through the model on sequence data.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch, timesteps, in_features]`.
    ///
    /// # Returns
    /// Output tensor of shape `[batch, timesteps, out_features]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Retrieve dimensions: [batch, timesteps, in_features].
        let shape = x.shape();

        let batch = shape.dims[0];
        let timesteps = shape.dims[1];
        let in_features = shape.dims[2];

        // Reshape to 2D tensor for the linear layer: [batch * timesteps, in_features].
        let x_reshaped = x.reshape([batch * timesteps, in_features]);
        // Apply the linear layer.
        let y = self.linear.forward(x_reshaped);

        // Reshape the output back to 3D: [batch, timesteps, out_features].
        let reshaped = y.reshape([batch, timesteps, self.out_features]);

        // @TODO Demonstration of a bug:
        // Upscale the time dimension by repeating it 2 times.
        let upscaled = reshaped.repeat_dim(1, 2);

        // Then sum over the last dimension.
        let summed = upscaled.sum_dim(2);

        summed.narrow(1, 0, timesteps)
    }

    /// Saves the model to a file.
    ///
    /// # Arguments
    /// * `path` - File path where the model will be saved.
    /// * `recorder` - Recorder for serializing model state.
    ///
    /// # Returns
    /// A `Result` indicating success or failure.
    pub fn save_file(&self, path: String, recorder: &CompactRecorder) -> Result<(), String> {
        fs::write(path, "model data").map_err(|e| e.to_string())
    }
}

/// Implements the training step for `SimpleModel` for sequence regression tasks.
/// It computes the Mean Squared Error (MSE) loss and produces 2D predictions and targets
/// for the regression output.
impl<B: AutodiffBackend> TrainStep<(Tensor<B, 3>, Tensor<B, 3>), RegressionOutput<B>>
    for SimpleModel<B>
{
    fn step(
        &self,
        (input, target): (Tensor<B, 3>, Tensor<B, 3>),
    ) -> TrainOutput<RegressionOutput<B>> {
        // Forward pass.
        let prediction = self.forward(input.clone());
        // Compute MSE loss.
        let loss = prediction
            .clone()
            .sub(target.clone())
            .powf_scalar(2.0)
            .mean();
        // Retrieve batch and timestep dimensions.
        let shape = prediction.shape();
        let batch = shape.dims[0];
        let timesteps = shape.dims[1];
        // Flatten predictions to 2D: [batch * timesteps, out_features].
        let prediction_2d = prediction.reshape([batch * timesteps, self.out_features]);
        // Flatten targets to 2D: [batch * timesteps, target_features].
        let target_shape = target.shape();
        let target_2d = target.reshape([
            target_shape.dims[0] * target_shape.dims[1],
            target_shape.dims[2],
        ]);
        // Create regression output with 2D tensors.
        let reg_output = RegressionOutput::new(loss.clone(), prediction_2d, target_2d);
        // Backpropagate and return the training output.
        TrainOutput::new(self, loss.backward(), reg_output)
    }
}

/// Implements the validation step for `SimpleModel` for sequence regression tasks.
/// It computes the Mean Squared Error (MSE) loss and produces 2D predictions and targets
/// for the regression output.
impl<B: Backend> ValidStep<(Tensor<B, 3>, Tensor<B, 3>), RegressionOutput<B>> for SimpleModel<B> {
    fn step(&self, (input, target): (Tensor<B, 3>, Tensor<B, 3>)) -> RegressionOutput<B> {
        let prediction = self.forward(input);
        let loss = prediction
            .clone()
            .sub(target.clone())
            .powf_scalar(2.0)
            .mean();
        // Retrieve batch and timestep dimensions.
        let shape = prediction.shape();
        let batch = shape.dims[0];
        let timesteps = shape.dims[1];
        // Flatten predictions to 2D.
        let prediction_2d = prediction.reshape([batch * timesteps, self.out_features]);
        // Flatten targets to 2D.
        let target_shape = target.shape();
        let target_2d = target.reshape([
            target_shape.dims[0] * target_shape.dims[1],
            target_shape.dims[2],
        ]);
        // Create regression output with 2D tensors.
        RegressionOutput::new(loss, prediction_2d, target_2d)
    }
}

/// Configuration for constructing a `SimpleModel`.
#[derive(Config, Debug)]
pub struct SimpleModelConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl SimpleModelConfig {
    /// Initializes the `SimpleModel` using the configuration.
    ///
    /// # Type Parameters
    /// * `B`: The backend type.
    ///
    /// # Arguments
    /// * `device` - The device for tensor allocation.
    ///
    /// # Returns
    /// An instance of `SimpleModel<B>`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleModel<B> {
        let linear = LinearConfig::new(self.in_features, self.out_features).init(device);
        SimpleModel {
            linear,
            out_features: self.out_features,
        }
    }
}

/// Training configuration for the simple sequence regression task.
#[derive(Config)]
pub struct TrainingConfig {
    /// Model configuration.
    pub model: SimpleModelConfig,
    /// Optimizer configuration.
    pub optimizer: AdamConfig,
    /// Number of training epochs.
    #[config(default = 1000)]
    pub num_epochs: usize,
    /// Batch size.
    #[config(default = 1)]
    pub batch_size: usize,
    /// Number of workers for data loading.
    #[config(default = 1)]
    pub num_workers: usize,
    /// Random seed.
    #[config(default = 42)]
    pub seed: u64,
    /// Learning rate.
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

/// A simple dataset for sequence regression, holding a collection of input-target pairs.
/// Each sample is a 2D tensor of shape `[timesteps, in_features]`.
#[derive(Clone, Debug)]
pub struct SimpleDataset<B: Backend> {
    /// Vector of samples, where each sample is a pair: (input, target).
    pub samples: Vec<(Tensor<B, 2>, Tensor<B, 2>)>,
}

impl<B: Backend> SimpleDataset<B> {
    /// Creates a new dataset with the provided samples.
    pub fn new(samples: Vec<(Tensor<B, 2>, Tensor<B, 2>)>) -> Self {
        Self { samples }
    }
}

/// Implements the Dataset trait for `SimpleDataset`.
impl<B: Backend> Dataset<(Tensor<B, 2>, Tensor<B, 2>)> for SimpleDataset<B> {
    fn get(&self, index: usize) -> Option<(Tensor<B, 2>, Tensor<B, 2>)> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// A simple batcher that collects a vector of samples (each of shape `[timesteps, in_features]`)
/// and concatenates them into batched tensors of shape `[batch, timesteps, in_features]`.
#[derive(Clone)]
pub struct SimpleBatcher<B: Backend> {
    /// Device for tensor operations.
    pub device: B::Device,
}

impl<B: Backend> SimpleBatcher<B> {
    /// Constructs a new `SimpleBatcher`.
    ///
    /// # Arguments
    /// * `device` - The device for tensor allocation.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

/// Implements the Batcher trait for `SimpleBatcher`.
impl<B: Backend> Batcher<(Tensor<B, 2>, Tensor<B, 2>), (Tensor<B, 3>, Tensor<B, 3>)>
    for SimpleBatcher<B>
{
    /// Batches a vector of samples by unsqueezing and concatenating along the new batch dimension.
    fn batch(&self, samples: Vec<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let (inputs, targets): (Vec<_>, Vec<_>) = samples.into_iter().unzip();
        // Unsqueeze each sample to add the batch dimension.
        let inputs: Vec<_> = inputs.into_iter().map(|x| x.unsqueeze()).collect();
        let targets: Vec<_> = targets.into_iter().map(|x| x.unsqueeze()).collect();
        let batched_inputs = Tensor::cat(inputs, 0);
        let batched_targets = Tensor::cat(targets, 0);
        (batched_inputs, batched_targets)
    }
}

/// Creates or resets the artifact directory.
///
/// # Arguments
/// * `artifact_dir` - Directory path where artifacts are stored.
fn create_artifact_dir(artifact_dir: &str) {
    let _ = fs::remove_dir_all(artifact_dir);
    fs::create_dir_all(artifact_dir).unwrap();
}

/// Trains the `SimpleModel` on a sequence regression task (y = 3x + 2) using a learner.
///
/// # Type Parameters
/// * `B`: The autodiff backend type.
///
/// # Arguments
/// * `artifact_dir` - Directory to store training artifacts.
/// * `config` - Training configuration.
/// * `device` - The device used for training.
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    // Prepare the artifact directory.
    create_artifact_dir(artifact_dir);

    // Define the number of timesteps and features for each sequence.
    let timesteps = 10;
    let n_features = 2;

    // Create training samples: each sample is a sequence of shape `[timesteps, n_features]`.
    let samples_t: Vec<(Tensor<B, 2>, Tensor<B, 2>)> = vec![
        (
            // Training Sample 1: Input sequence.
            Tensor::<B, 1>::from_floats(
                [
                    -1.0, 0.5, // timestep 1
                    -0.8, 0.6, // timestep 2
                    -0.6, 0.7, // timestep 3
                    -0.4, 0.8, // timestep 4
                    -0.2, 0.9, // timestep 5
                    0.0, 1.0, // timestep 6
                    0.2, 1.1, // timestep 7
                    0.4, 1.2, // timestep 8
                    0.6, 1.3, // timestep 9
                    0.8, 1.4, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
            // Training Sample 1: Target sequence.
            Tensor::<B, 1>::from_floats(
                [
                    -1.0, 1.0, // timestep 1
                    -0.6, 1.2, // timestep 2
                    -0.2, 1.4, // timestep 3
                    0.2, 1.6, // timestep 4
                    0.6, 1.8, // timestep 5
                    1.0, 2.0, // timestep 6
                    1.4, 2.2, // timestep 7
                    1.8, 2.4, // timestep 8
                    2.2, 2.6, // timestep 9
                    2.6, 2.8, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
        ),
        (
            // Training Sample 2: Input sequence.
            Tensor::<B, 1>::from_floats(
                [
                    0.5, 1.0, // timestep 1
                    0.7, 1.1, // timestep 2
                    0.9, 1.2, // timestep 3
                    1.1, 1.3, // timestep 4
                    1.3, 1.4, // timestep 5
                    1.5, 1.5, // timestep 6
                    1.7, 1.6, // timestep 7
                    1.9, 1.7, // timestep 8
                    2.1, 1.8, // timestep 9
                    2.3, 1.9, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
            // Training Sample 2: Target sequence.
            Tensor::<B, 1>::from_floats(
                [
                    3.5, 4.0, // timestep 1
                    3.7, 4.1, // timestep 2
                    3.9, 4.2, // timestep 3
                    4.1, 4.3, // timestep 4
                    4.3, 4.4, // timestep 5
                    4.5, 4.5, // timestep 6
                    4.7, 4.6, // timestep 7
                    4.9, 4.7, // timestep 8
                    5.1, 4.8, // timestep 9
                    5.3, 4.9, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
        ),
        (
            // Training Sample 3: Input sequence.
            Tensor::<B, 1>::from_floats(
                [
                    1.0, 0.5, // timestep 1
                    1.2, 0.7, // timestep 2
                    1.4, 0.9, // timestep 3
                    1.6, 1.1, // timestep 4
                    1.8, 1.3, // timestep 5
                    2.0, 1.5, // timestep 6
                    2.2, 1.7, // timestep 7
                    2.4, 1.9, // timestep 8
                    2.6, 2.1, // timestep 9
                    2.8, 2.3, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
            // Training Sample 3: Target sequence.
            Tensor::<B, 1>::from_floats(
                [
                    5.0, 3.0, // timestep 1
                    5.2, 3.2, // timestep 2
                    5.4, 3.4, // timestep 3
                    5.6, 3.6, // timestep 4
                    5.8, 3.8, // timestep 5
                    6.0, 4.0, // timestep 6
                    6.2, 4.2, // timestep 7
                    6.4, 4.4, // timestep 8
                    6.6, 4.6, // timestep 9
                    6.8, 4.8, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
        ),
    ];

    // Create validation samples similarly using the inner backend.
    let samples_v: Vec<(Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 2>)> = vec![
        (
            // Validation Sample 1: Input sequence.
            Tensor::<B::InnerBackend, 1>::from_floats(
                [
                    -1.1, 0.4, // timestep 1
                    -0.9, 0.5, // timestep 2
                    -0.7, 0.6, // timestep 3
                    -0.5, 0.7, // timestep 4
                    -0.3, 0.8, // timestep 5
                    -0.1, 0.9, // timestep 6
                    0.1, 1.0, // timestep 7
                    0.3, 1.1, // timestep 8
                    0.5, 1.2, // timestep 9
                    0.7, 1.3, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
            // Validation Sample 1: Target sequence.
            Tensor::<B::InnerBackend, 1>::from_floats(
                [
                    -1.2, 0.9, // timestep 1
                    -0.8, 1.0, // timestep 2
                    -0.4, 1.1, // timestep 3
                    0.0, 1.2, // timestep 4
                    0.4, 1.3, // timestep 5
                    0.8, 1.4, // timestep 6
                    1.2, 1.5, // timestep 7
                    1.6, 1.6, // timestep 8
                    2.0, 1.7, // timestep 9
                    2.4, 1.8, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
        ),
        (
            // Validation Sample 2: Input sequence.
            Tensor::<B::InnerBackend, 1>::from_floats(
                [
                    0.6, 1.1, // timestep 1
                    0.8, 1.2, // timestep 2
                    1.0, 1.3, // timestep 3
                    1.2, 1.4, // timestep 4
                    1.4, 1.5, // timestep 5
                    1.6, 1.6, // timestep 6
                    1.8, 1.7, // timestep 7
                    2.0, 1.8, // timestep 8
                    2.2, 1.9, // timestep 9
                    2.4, 2.0, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
            // Validation Sample 2: Target sequence.
            Tensor::<B::InnerBackend, 1>::from_floats(
                [
                    3.6, 4.1, // timestep 1
                    3.8, 4.2, // timestep 2
                    4.0, 4.3, // timestep 3
                    4.2, 4.4, // timestep 4
                    4.4, 4.5, // timestep 5
                    4.6, 4.6, // timestep 6
                    4.8, 4.7, // timestep 7
                    5.0, 4.8, // timestep 8
                    5.2, 4.9, // timestep 9
                    5.4, 5.0, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
        ),
        (
            // Validation Sample 3: Input sequence.
            Tensor::<B::InnerBackend, 1>::from_floats(
                [
                    1.1, 0.6, // timestep 1
                    1.3, 0.8, // timestep 2
                    1.5, 1.0, // timestep 3
                    1.7, 1.2, // timestep 4
                    1.9, 1.4, // timestep 5
                    2.1, 1.6, // timestep 6
                    2.3, 1.8, // timestep 7
                    2.5, 2.0, // timestep 8
                    2.7, 2.2, // timestep 9
                    2.9, 2.4, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
            // Validation Sample 3: Target sequence.
            Tensor::<B::InnerBackend, 1>::from_floats(
                [
                    5.1, 3.1, // timestep 1
                    5.3, 3.3, // timestep 2
                    5.5, 3.5, // timestep 3
                    5.7, 3.7, // timestep 4
                    5.9, 3.9, // timestep 5
                    6.1, 4.1, // timestep 6
                    6.3, 4.3, // timestep 7
                    6.5, 4.5, // timestep 8
                    6.7, 4.7, // timestep 9
                    6.9, 4.9, // timestep 10
                ],
                &device,
            )
            .reshape([timesteps, n_features]),
        ),
    ];

    let dataset_t = SimpleDataset::<B>::new(samples_t);
    let dataset_v = SimpleDataset::<B::InnerBackend>::new(samples_v);

    // Build data loaders for training and validation.
    let dataloader_train = DataLoaderBuilder::new(SimpleBatcher::<B>::new(device.clone()))
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_t);

    let dataloader_valid =
        DataLoaderBuilder::new(SimpleBatcher::<B::InnerBackend>::new(device.clone()))
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(dataset_v);

    // Build the learner.
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // Fit the model.
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save the trained model.
    model_trained
        .save_file(
            format!("{}/model.bin", artifact_dir),
            &CompactRecorder::new(),
        )
        .expect("Failed to save the model");
}


#[cfg(test)]
mod tests {
    // Import necessary items from the current module.
    use super::*;
    use burn::{backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, tensor::Tensor};
    use burn_ndarray::NdArray;

    /// Tests the forward pass of `SimpleModel` by verifying the output tensor shape.
    #[test]
    fn test_forward_pass_shape() {
        type Backend = NdArray<f32, i32>;
        type AutodiffBackend = Autodiff<Backend>;
        let device = burn_ndarray::NdArrayDevice::default();

        // Define dimensions for the test.
        let batch: usize = 2;
        let timesteps: usize = 3;
        let in_features: usize = 4;
        let out_features: usize = 5;

        // Initialize the model using the configuration.
        let model_config = SimpleModelConfig {
            in_features,
            out_features,
        };
        let model = model_config.init::<AutodiffBackend>(&device);

        // Create a dummy input tensor with sequential values.
        let input_values: Vec<f32> = (0..(batch * timesteps * in_features))
            .map(|x| x as f32)
            .collect();

        let input = Tensor::<AutodiffBackend, 1>::from_floats(input_values.as_slice(), &device)
            .reshape([batch, timesteps, in_features]);

        // Execute the forward pass.
        let output = model.forward(input);

        // Retrieve and check the output shape.
        let output_shape = output.shape();
        
        assert_eq!(output_shape.dims.len(), 3);
        assert_eq!(output_shape.dims[0], batch);
        assert_eq!(output_shape.dims[1], timesteps);
    }
}