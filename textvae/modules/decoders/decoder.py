from typing import Dict, Tuple

import colt
import torch

DecoderState = Dict[str, torch.Tensor]


class Decoder(torch.nn.Module, colt.Registrable):
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def init_state(self, encoder_output: torch.Tensor) -> DecoderState:
        raise NotImplementedError

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_output: torch.Tensor,
        previous_state: Dict[str, torch.Tensor],
    ) -> Tuple[DecoderState, torch.Tensor]:
        raise NotImplementedError
