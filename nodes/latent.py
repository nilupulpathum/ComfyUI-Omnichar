# SPDX-License-Identifier: GPL-3.0-or-later
"""Attach a character's references to conditioning, for a model that reads them as latents."""

from __future__ import annotations

import node_helpers
from omnichar_sdk import (
    FIT_MODES,
    SIZE_POLICIES,
    CharChanged,
    CharError,
    common_size,
)

from .common import CATEGORY, REFS_INPUT, fail_on_change, to_image


class OmnicharCharacterReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "refs": REFS_INPUT,
                "vae": ("VAE", {"tooltip": "Encodes each reference. Use the model's own VAE."}),
            },
            "optional": {
                "size_from": (list(SIZE_POLICIES), {"default": "first"}),
                "fit": (list(FIT_MODES), {"default": "pad"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    RETURN_NAMES = ("conditioning", "images")
    FUNCTION = "attach"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Attach every reference to the conditioning as a latent, for an edit model like FLUX.2. "
        "Replaces a chain of Set Reference Latent nodes, and takes as many references as the "
        "character has. Feed it to the positive and the negative conditioning both. "
        "The images output carries the same references as one batch, for a model whose text "
        "encoder takes reference images in its own slots, like Qwen-Image 2.1 or Krea 2."
    )

    def attach(self, conditioning, refs, vae, size_from="first", fit="pad"):
        try:
            if not refs:
                raise CharError(
                    "Decode Character sent no references. Check its reference set and maximum."
                )
            # One append of the whole list, which is how ComfyUI's own multi-reference nodes do it.
            latents = [vae.encode(to_image([ref.open()])[:, :, :, :3]) for ref in refs]
            # The same resolved list as one batch, so a text encoder's image slots receive
            # exactly the references the prompt's numbers name.
            size = common_size(refs, size_from)
            images = to_image([ref.fit(size, fit) for ref in refs])
        except CharChanged as error:
            raise fail_on_change(error) from error
        return (
            node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": latents}, append=True
            ),
            images,
        )
