from typing import List, cast

import pytest
import torch

from textvae.modules.encoders import GruEncoder, LstmEncoder, PytorchRnnWrapper, RnnEncoder


def _length_to_mask(lengths: List[int]) -> torch.BoolTensor:
    max_length = max(lengths)
    mask = torch.zeros((len(lengths), max_length), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return cast(torch.BoolTensor, mask)


@pytest.mark.parametrize(
    "encoder_cls",
    [GruEncoder, LstmEncoder, RnnEncoder],
)
def test_pytorch_rnn_wrapper(encoder_cls: PytorchRnnWrapper) -> None:
    encoder = encoder_cls(input_size=16, hidden_size=8)

    inputs = torch.randn(3, 5, 16)
    mask = _length_to_mask([4, 3, 5])
    output = encoder(inputs, mask)

    assert encoder.get_output_dim() == 8
    assert output.size() == (3, encoder.get_output_dim())


@pytest.mark.parametrize(
    "encoder_cls",
    [GruEncoder, LstmEncoder, RnnEncoder],
)
def test_bidirectional_pytorch_rnn_wrapper(encoder_cls: PytorchRnnWrapper) -> None:
    encoder = encoder_cls(input_size=16, hidden_size=8, bidirectional=True)

    inputs = torch.randn(3, 5, 16)
    mask = _length_to_mask([4, 3, 5])
    output = encoder(inputs, mask)

    assert encoder.get_output_dim() == 16
    assert output.size() == (3, encoder.get_output_dim())
