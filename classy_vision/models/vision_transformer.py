#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Vision Transformer implementation from https://arxiv.org/abs/2010.11929.

References:
https://github.com/google-research/vision_transformer
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import copy
import logging
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from classy_vision.models import ClassyModel, register_model

from .lecun_normal_init import lecun_normal_init


LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block.

    From @myleott -
    There are at least three common structures.
    1) Attention is all you need had the worst one, where the layernorm came after each
        block and was in the residual path.
    2) BERT improved upon this by moving the layernorm to the beginning of each block
        (and adding an extra layernorm at the end).
    3) There's a further improved version that also moves the layernorm outside of the
        residual path, which is what this implementation does.

    Figure 1 of this paper compares versions 1 and 3:
        https://openreview.net/pdf?id=B1x8anVFPr
    Figure 7 of this paper compares versions 2 and 3 for BERT:
        https://arxiv.org/abs/1909.08053
    """

    def __init__(
        self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout_rate
        )  # uses correct initialization by default
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)

    def forward(self, input):
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout_rate,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        for i in range(num_layers):
            layers.append(
                (
                    f"layer_{i}",
                    EncoderBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                    ),
                )
            )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding  # should broadcast to the same shape
        return self.ln(self.layers(self.dropout(x)))


@register_model("vision_transformer")
class VisionTransformer(ClassyModel):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Input shape indivisible by patch size"
        assert classifier in ["token", "gap"], "Unexpected classifier mode"
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.classifier = classifier

        input_channels = 3

        # conv_proj is a more efficient version of reshaping, permuting and projecting
        # the input
        self.conv_proj = nn.Conv2d(
            input_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
        )
        self.trunk_output = nn.Identity()

        self.seq_length = seq_length
        self.init_weights()

    def init_weights(self):
        lecun_normal_init(
            self.conv_proj.weight,
            fan_in=self.conv_proj.in_channels
            * self.conv_proj.kernel_size[0]
            * self.conv_proj.kernel_size[1],
        )
        nn.init.zeros_(self.conv_proj.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config.pop("name")
        config.pop("heads", None)
        return cls(**config)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Unexpected input shape"
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> ((n_h * n_w), n, hidden_dim)
        # the self attention layer expects inputs in the format (S, N, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(2, 0, 1)

        if self.classifier == "token":
            # expand the class token to the full batch
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        x = self.encoder(x)

        if self.classifier == "token":
            # just return the output for the class token
            x = x[0, :, :]
        else:
            x = x.mean(dim=0)

        return self.trunk_output(x)

    def set_classy_state(self, state, strict=True):
        # shape of pos_embedding is (seq_length, 1, hidden_dim)
        pos_embedding = state["model"]["trunk"]["encoder.pos_embedding"]
        seq_length, n, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(
                f"Unexpected position embedding shape: {pos_embedding.shape}"
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Position embedding hidden_dim incorrect: {hidden_dim}"
                f", expected: {self.hidden_dim}"
            )
        new_seq_length = self.seq_length

        if new_seq_length != seq_length:
            # need to interpolate the weights for the position embedding
            # we do this by reshaping the positions embeddings to a 2d grid, performing
            # an interpolation in the (h, w) space and then reshaping back to a 1d grid
            if self.classifier == "token":
                # the class token embedding shouldn't be interpolated so we split it up
                seq_length -= 1
                new_seq_length -= 1
                pos_embedding_token = pos_embedding[:1, :, :]
                pos_embedding_img = pos_embedding[1:, :, :]
            else:
                pos_embedding_token = pos_embedding[:0, :, :]  # empty data
                pos_embedding_img = pos_embedding
            # (seq_length, 1, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(1, 2, 0)
            seq_length_1d = int(math.sqrt(seq_length))
            assert (
                seq_length_1d * seq_length_1d == seq_length
            ), "seq_length is not a perfect square"

            logging.info(
                "Interpolating the position embeddings from image "
                f"{seq_length_1d * self.patch_size} to size {self.image_size}"
            )

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = self.image_size // self.patch_size

            # use bicubic interpolation - it gives significantly better results in
            # the test `test_resolution_change`
            new_pos_embedding_img = torch.nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode="bicubic",
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_l)
            new_pos_embedding_img = new_pos_embedding_img.reshape(
                1, hidden_dim, new_seq_length
            )
            # (1, hidden_dim, new_seq_length) -> (new_seq_length, 1, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(2, 0, 1)
            new_pos_embedding = torch.cat(
                [pos_embedding_token, new_pos_embedding_img], dim=0
            )
            state["model"]["trunk"]["encoder.pos_embedding"] = new_pos_embedding
        super().set_classy_state(state, strict=strict)


@register_model("vit_b_32")
class ViTB32(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
    ):
        super().__init__(
            image_size=image_size,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
        )


@register_model("vit_b_16")
class ViTB16(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
    ):
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
        )


@register_model("vit_l_32")
class ViTL32(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
    ):
        super().__init__(
            image_size=image_size,
            patch_size=32,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
        )


@register_model("vit_l_16")
class ViTL16(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
    ):
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
        )


@register_model("vit_h_14")
class ViTH14(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
    ):
        super().__init__(
            image_size=image_size,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
        )
