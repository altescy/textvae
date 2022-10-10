from textvae.data.datamodule import Batch, TextVaeDataModule


def test_textvae_datamodule() -> None:
    datamodule = TextVaeDataModule()
    dataset = datamodule.build_dataset("tests/fixtures/data/sentences.txt", update_vocab=True)
    dataloader = datamodule.build_dataloader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    assert isinstance(batch, Batch)
    assert batch.tokens.size() == (2, 5)
    assert batch.mask.size() == (2, 5)
    assert len(datamodule.vocab) == 13
