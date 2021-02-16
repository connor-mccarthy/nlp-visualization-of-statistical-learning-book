from typing import List

import spacy
from spacy.tokens.token import Token

from config import NLP_STOP_WORDS


def get_noun_tokens(pages: str) -> List[Token]:
    text = " ".join(pages)

    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.Defaults.stop_words.update(NLP_STOP_WORDS)
    doc = nlp(text)
    return [token for token in doc if token.pos_ == "NOUN"]
