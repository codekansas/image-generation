"""Defines a simple supervised learning template task.

This task is meant to be used as a template for creating new tasks. Just
change the key from ``template`` to whatever you want to name your task, and
implement the following methods:

- :meth:`run_model`
- :meth:`compute_loss`
- :meth:`get_dataset`
"""

from dataclasses import dataclass

import ml.api as ml
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from torch import Tensor
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision.datasets import MNIST

from image_gen.models.unet import UNetModel


@dataclass
class MnistTaskConfig(ml.SupervisedLearningTaskConfig):
    num_beta_steps: int = ml.conf_field(500, help="Number of beta steps")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = UNetModel
Batch = tuple[Tensor, ...]
Output = tuple[Tensor, Tensor]
Loss = Tensor


@ml.register_task("mnist", MnistTaskConfig)
class MnistTask(ml.SupervisedLearningTask[MnistTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: MnistTaskConfig) -> None:
        super().__init__(config)

        betas = ml.get_diffusion_beta_schedule("linear", config.num_beta_steps, dtype=torch.float32)
        self.diff = ml.GaussianDiffusion(betas)

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        (images,) = batch
        times = self.diff.sample_random_times(images.shape[0], device=images.device)
        q_sample, noise = self.diff.q_sample(images, times)
        pred_noise = model(q_sample, times)
        return pred_noise, noise

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        (images,), (pred_noise, noise) = batch, output
        loss = F.mse_loss(pred_noise, noise)

        def model_sample(q_sample: Tensor, t: Tensor) -> Tensor:
            x = model.forward(q_sample, t)
            return x.clamp(-1.0, 1.0)

        if state.phase != "train":
            max_images = 9
            init_noise = torch.randn_like(images[:max_images])
            generated = self.diff.p_sample_loop(model_sample, init_noise)
            self.logger.log_images("generated", generated[-1], max_images=max_images, sep=2)
            single_generation = torch.stack([g[0] for g in generated])
            self.logger.log_images("generated_single", single_generation, max_images=max_images, sep=2,)

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


if __name__ == "__main__":
    # python -m image_gen.tasks.mnist
    ml.test_task(MnistTask(MnistTaskConfig()))
