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
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST

from image_gen.models.unet import UNetModel


@dataclass
class MnistTaskConfig(ml.SupervisedLearningTaskConfig):
    num_beta_steps: int = ml.conf_field(100, help="Number of beta steps")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = UNetModel
Batch = tuple[Tensor, Tensor]
Output = tuple[Tensor, Tensor]
Loss = Tensor


def pad_images(x: PIL.Image.Image) -> PIL.Image:
    return V.pad(x, padding=[2, 2])


@ml.register_task("mnist", MnistTaskConfig)
class MnistTask(ml.SupervisedLearningTask[MnistTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: MnistTaskConfig) -> None:
        super().__init__(config)

        betas = ml.get_diffusion_beta_schedule("linear", config.num_beta_steps, dtype=torch.float32)
        self.diff = ml.GaussianDiffusion(betas)

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        images, _ = batch
        images = (images - images.mean()) / images.std()
        times = self.diff.sample_random_times(images.shape[0], device=images.device)
        q_sample, noise = self.diff.q_sample(images, times)
        pred_noise = model(q_sample, times)
        return pred_noise, noise

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        (images, _), (pred_noise, noise) = batch, output
        loss = F.mse_loss(pred_noise, noise)

        def model_sample(q_sample: Tensor, t: Tensor) -> Tensor:
            x = model.forward(q_sample, t)
            return x.clamp(-1.0, 1.0)

        if state.phase != "train":
            init_noise = torch.randn_like(images)
            generated = self.diff.p_sample_loop(model_sample, init_noise)
            self.logger.log_images("generated", generated[-1], max_images=9, sep=2)

        return loss

    def get_dataset(self, phase: ml.Phase) -> Dataset:
        root_dir = ml.get_data_dir() / "mnist"
        return MNIST(
            root=root_dir,
            train=phase == "train",
            download=not root_dir.exists(),
            transform=pad_images,
        )


if __name__ == "__main__":
    # python -m image_gen.tasks.mnist
    ml.test_task(MnistTask(MnistTaskConfig()))