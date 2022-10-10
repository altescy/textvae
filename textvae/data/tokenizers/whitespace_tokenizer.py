from typing import List

from textvae.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("whitespace")
class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        return text.split()
