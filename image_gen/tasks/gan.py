"""Defines a task for training a GAN model on MNIST."""

from dataclasses import dataclass

import ml.api as ml
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from torch import Tensor
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision.datasets import MNIST

from image_gen.models.gan import DiscriminatorModel, GeneratorModel


@dataclass
class GANTaskConfig(ml.GenerativeAdversarialNetworkTaskConfig):
    num_beta_steps: int = ml.conf_field(500, help="Number of beta steps")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Batch = tuple[Tensor, ...]
GeneratorOutput = Tensor
DiscriminatorOutput = tuple[Tensor, Tensor]
Loss = Tensor


@ml.register_task("gan", GANTaskConfig)
class GANTask(
    ml.GenerativeAdversarialNetworkTask[
        GANTaskConfig,
        GeneratorModel,
        DiscriminatorModel,
        Batch,
        GeneratorOutput,
        DiscriminatorOutput,
    ],
):
    def run_generator(self, generator: GeneratorModel, batch: Batch, state: ml.State) -> GeneratorOutput:
        (images,) = batch
        noise = torch.randn_like(images)
        gen_images = generator(noise)
        return gen_images

    def run_discriminator(
        self,
        discriminator: DiscriminatorModel,
        batch: Batch,
        gen_output: GeneratorOutput,
        state: ml.State,
    ) -> DiscriminatorOutput:
        (images,) = batch
        gen_images = gen_output
        return discriminator(images), discriminator(gen_images)

    def compute_generator_loss(
        self,
        generator: GeneratorModel,
        discriminator: DiscriminatorModel,
        batch: Batch,
        state: ml.State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
    ) -> dict[str, Tensor]:
        _, fake_pred = dis_output
        loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        return {"generator": loss}

    def compute_discriminator_loss(
        self,
        generator: GeneratorModel,
        discriminator: DiscriminatorModel,
        batch: Batch,
        state: ml.State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
    ) -> dict[str, Tensor]:
        real_pred, fake_pred = dis_output
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        return {"real": real_loss, "fake": fake_loss}

    def do_logging(
        self,
        generator: GeneratorModel,
        discriminator: DiscriminatorModel,
        batch: Batch,
        state: ml.State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
        losses: dict[str, Tensor],
    ) -> None:
        if state.phase != "train":
            (images,) = batch
            gen_images = gen_output
            max_images = 9
            self.logger.log_images("real", images, max_images=max_images, sep=2)
            self.logger.log_images("generated", gen_images, max_images=max_images, sep=2)

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
