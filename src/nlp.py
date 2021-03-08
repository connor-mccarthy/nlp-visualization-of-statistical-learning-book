from typing import List

import numpy as np
import spacy
from spacy.tokens.token import Token
from wordfreq import word_frequency

nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
nlp.max_length = 2_000_000


def is_big_word(token: Token) -> bool:
    return len(token.text) > 2


def is_alpha(token: Token) -> bool:
    return token.is_alpha


def not_stop(token: Token) -> bool:
    return not token.is_stop


def is_probably_a_word(token: Token) -> bool:
    return not token.is_oov


def has_vector(token: Token) -> bool:
    return np.sum(token.vector) != 0


def is_common_word(token: Token) -> bool:
    return word_frequency(token.text, "en", "best" < 0.0005)


filter_funcs = [is_big_word, is_alpha, not_stop, is_probably_a_word, has_vector]


def lemmatize_tokens(tokens: Token) -> Token:
    lemmatized_tokens = [token.lemma_.lower() for token in tokens]
    token_string = " ".join(lemmatized_tokens)
    doc = nlp(token_string)
    return [token for token in doc]


def get_noun_tokens(text: str, downsample_factor: int = 1) -> List[Token]:
    doc = nlp(text.lower())
    nouns = [token for token in doc if token.pos_ == "NOUN"]
    return [token for token in nouns if all(func(token) for func in filter_funcs)][
        ::downsample_factor
    ]
