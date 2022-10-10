from typing import List

import colt


class Tokenizer(colt.Registrable):
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError
