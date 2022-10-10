TextVAE
=======

[![Actions Status](https://github.com/altescy/textvae/workflows/CI/badge.svg)](https://github.com/altescy/textvae/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/altescy/textvae)](https://github.com/altescy/textvae/blob/master/LICENSE)

VAE implementation for text generation with PyTorch

## Usage

Train VAE model:

```bash
textvae train config.json --workdir output/
```

Reconstruct texts:

```python
from textvae import TextVAE

texts = ["this is a first sentence", "this is a second sentence"]
textvae = TextVAE.from_archive("output/archive.pkl")
for reconstructed_text in textvae.reconstruct(texts):
    print(reconstructed_text)
```

Encode texts:

```python
from textvae import TextVAE

texts = ["this is a first sentence", "this is a second sentence"]
textvae = TextVAE.from_archive("output/archive.pkl")
for mean, logvar in textvae.encode(texts):
    ...
```
