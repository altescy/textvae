import colt
import torch


class Embedder(torch.nn.Module, colt.Registrable):
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        raise NotImplementedError
