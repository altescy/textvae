from importlib.metadata import version

from textvae.textvae import TextVAE

__version__ = version("textvae")
__all__ = ["TextVAE"]
