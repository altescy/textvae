from typing import Tuple, cast

import torch

from textvae.modules.decoders.decoder import Decoder, DecoderState


@Decoder.register("lstm_cell")
class LstmCellDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self._deocder_cell = torch.nn.LSTMCell(input_size + hidden_size, hidden_size)

    def get_output_dim(self) -> int:
        return self._deocder_cell.hidden_size

    def init_state(self, encoder_output: torch.Tensor) -> DecoderState:
        assert encoder_output.dim() == 2, f"{encoder_output.dim()=}"
        assert encoder_output.size(1) == self._deocder_cell.hidden_size, f"{encoder_output.size()=}"
        return {
            "hidden": encoder_output,
            "cell": encoder_output.new_zeros(encoder_output.size()),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_output: torch.Tensor,
        previous_state: DecoderState,
    ) -> Tuple[DecoderState, torch.Tensor]:
        inputs = torch.cat([inputs, encoder_output], dim=-1)
        hidden, cell = self._deocder_cell(inputs, (previous_state["hidden"], previous_state["cell"]))
        return {"hidden": hidden, "cell": cell}, cast(torch.Tensor, hidden)
