"""Defines a task for training a diffusion model on MNIST."""

from dataclasses import dataclass

import ml.api as ml
import torch
import torchvision.transforms.functional as V
from torch import Tensor
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision.datasets import MNIST

from image_gen.models.diffusion import DiffusionModel


@dataclass
class DiffusionTaskConfig(ml.SupervisedLearningTaskConfig):
    num_beta_steps: int = ml.conf_field(500, help="Number of beta steps")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = DiffusionModel
Batch = tuple[Tensor, ...]
Output = Tensor
Loss = Tensor


@ml.register_task("diffusion", DiffusionTaskConfig)
class DiffusionTask(ml.SupervisedLearningTask[DiffusionTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: DiffusionTaskConfig) -> None:
        super().__init__(config)

        betas = ml.get_diffusion_beta_schedule("cosine", config.num_beta_steps, dtype=torch.float32)
        self.diff = ml.GaussianDiffusion(betas)

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        (images,) = batch
        return self.diff.loss(images, model)

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        loss = output

        if state.phase != "train":
            (images,) = batch
            max_images = 9
            generated = self.diff.sample(model, images[:max_images].shape, images.device)
            self.logger.log_images("generated", generated[0], max_images=max_images, sep=2)
            single_generation = generated[:, 0]
            self.logger.log_images("generated_single", single_generation, max_images=max_images, sep=2)

        return loss

    def get_dataset(self, phase: ml.Phase) -> Dataset[tuple[Tensor, ...]]:
        root_dir = ml.get_data_dir() / "mnist"
        mnist = MNIST(
            root=root_dir,
            train=phase == "train",
            download=not root_dir.exists(),
        )

        data = mnist.data.float()
        data = V.pad(data, [2, 2])
        data = (data - 127.5) / 127.5
        data = data.unsqueeze(1)
        return TensorDataset(data)
