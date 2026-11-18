# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch


@dataclass
class EPICBlendMetadata:
    """Runtime metadata for EPIC blending.

    Tracks the current status and important indices across layers:
    - status 0: before check layer (normal forward)
    - status 1: check layer (importance scoring + index selection)
    - status 2: post-check layers (use previously selected imp_indices)
    """

    imp_indices: Optional[torch.Tensor] = None
    org_seq_len: Optional[int] = None
    attn_bias: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    recomp_ratio: float = 0.16
    status: int = 0

    def clean(self):
        self.imp_indices = None
        self.org_seq_len = None
        self.attn_bias = None
        self.positions = None
        self.status = 0
