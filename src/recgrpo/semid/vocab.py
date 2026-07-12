"""Token layout.

Each residual level owns a disjoint block of ids, so a token identifies both a
code and the level it came from and the decoder never has to be told where in an
item it is. The suffix block disambiguates items whose three codes collide.

    0            <pad>
    1            <bos>
    2            <eos>
    3            <sep>        end of history, start of the generated item
    4   .. 259    level 1     (256)
    260 .. 515    level 2     (256)
    516 .. 771    level 3     (256)
    772 .. 1027   suffix      (256)
"""

from __future__ import annotations

import numpy as np

PAD = 0
BOS = 1
EOS = 2
SEP = 3

N_SPECIAL = 4
CODEBOOK_SIZE = 256
N_LEVELS = 3
DEDUP_CAPACITY = 256

LEVEL_OFFSETS = tuple(N_SPECIAL + level * CODEBOOK_SIZE for level in range(N_LEVELS))
DEDUP_OFFSET = N_SPECIAL + N_LEVELS * CODEBOOK_SIZE
VOCAB_SIZE = DEDUP_OFFSET + DEDUP_CAPACITY

CODE_LEN = N_LEVELS + 1  # three learned codes plus the suffix


def codes_to_tokens(codes: np.ndarray) -> np.ndarray:
    """(n_items, 4) of per-level indices -> (n_items, 4) of vocabulary ids."""
    offsets = np.array(LEVEL_OFFSETS + (DEDUP_OFFSET,), dtype=np.int64)
    return codes.astype(np.int64) + offsets


def tokens_to_codes(tokens: np.ndarray) -> np.ndarray:
    offsets = np.array(LEVEL_OFFSETS + (DEDUP_OFFSET,), dtype=np.int64)
    return tokens.astype(np.int64) - offsets


def level_of(token: int) -> int:
    """Which position in an item sequence a token belongs to, or -1 for specials."""
    if token < N_SPECIAL:
        return -1
    return min((token - N_SPECIAL) // CODEBOOK_SIZE, CODE_LEN - 1)
