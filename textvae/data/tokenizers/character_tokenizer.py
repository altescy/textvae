from typing import List

from textvae.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("character")
class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        return list(text)
