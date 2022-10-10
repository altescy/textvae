from typing import cast

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from textvae.modules.encoders.encoder import Encoder


class PytorchRnnWrapper(Encoder):
    def __init__(self, module: torch.nn.Module) -> None:
        try:
            if not module.batch_first:
                raise ValueError("Expected batch_first=True, got batch_first=False")
        except AttributeError:
            pass

        super().__init__()

        self._module = module
        try:
            self._num_directions = 2 if module.bidirectional else 1
        except AttributeError:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return cast(int, self._module.input_size)

    def get_output_dim(self) -> int:
        return cast(int, self._module.hidden_size) * self._num_directions

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, max_length, _ = inputs.size()
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        packed, state = self._module(packed)
        if isinstance(state, tuple):
            state = state[0]
        output = state.transpose(0, 1).index_select(0, packed.unsorted_indices)
        output = output[:, -min(self._num_directions, 2) :, :]
        output = output.contiguous().view((batch_size, self.get_output_dim()))
        return cast(torch.Tensor, output)


@Encoder.register("lstm")
class LstmEncoder(PytorchRnnWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            torch.nn.LSTM(  # type: ignore[no-untyped-call]
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        )


@Encoder.register("gru")
class GruEncoder(PytorchRnnWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            torch.nn.GRU(  # type: ignore[no-untyped-call]
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        )


@Encoder.register("rnn")
class RnnEncoder(PytorchRnnWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            torch.nn.RNN(  # type: ignore[no-untyped-call]
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        )
