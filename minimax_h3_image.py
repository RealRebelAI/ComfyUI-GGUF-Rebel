# (c) RealRebelAI contributors || Apache-2.0
"""GGUF -> native ComfyUI metadata bridge for MiniMax H3 Image.

ComfyUI detects MiniMax H3 Image from checkpoint tensor names and reads
pdd_steps / pdd_sigmas from safetensors-style ``config`` JSON metadata.
This bridge supplies that config for H3 Image GGUFs without affecting other
model architectures.
"""

import json

# MiniMax H3 Image's official six-step PDD grid (pdd_config.json).
DEFAULT_PDD_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)


def is_minimax_h3_image(state_dict):
    """Identify the H3 Image tensor family, excluding H3 video checkpoints."""
    return (
        "video_patch_proj.weight" in state_dict
        and "condition_proj.weight" in state_dict
        and "final_layer.video_out.weight" in state_dict
        and "audio_patch_proj.weight" not in state_dict
        and "blocks.0.attn.qkv_proj.weight" in state_dict
    )


def prepare_minimax_h3_image_metadata(state_dict, metadata):
    """Supply native ComfyUI PDD config for H3 Image, preserving existing data.

    Prefer per-file GGUF PDD metadata. Fall back to the official six-step
    grid for pre-existing GGUFs that have no embedded PDD fields.
    """
    if not is_minimax_h3_image(state_dict):
        return metadata

    metadata = dict(metadata or {})
    raw_config = metadata.get("config")
    if raw_config:
        config = json.loads(raw_config)
        if not isinstance(config, dict):
            raise ValueError("MiniMax H3 Image GGUF config must be a JSON object")
    else:
        config = {}

    transformer = dict(config.get("transformer") or {})
    grid = transformer.get("pdd_sigmas")
    if grid is None:
        grid = metadata.get("minimax_h3_image.pdd_sigmas")
    if grid is None:
        grid = DEFAULT_PDD_SIGMAS
    grid = [float(v) for v in grid]
    if len(grid) < 2 or grid[0] != 1.0 or grid[-1] != 0.0 or any(a <= b for a, b in zip(grid, grid[1:])):
        raise ValueError("Invalid MiniMax H3 Image PDD sigma grid in GGUF metadata")

    steps = transformer.get("pdd_steps")
    if steps is None:
        steps = metadata.get("minimax_h3_image.pdd_steps", len(grid) - 1)
    steps = int(steps)
    if steps != len(grid) - 1:
        raise ValueError(f"MiniMax H3 Image PDD steps ({steps}) do not match sigma grid ({len(grid) - 1})")

    transformer["pdd_steps"] = steps
    transformer["pdd_sigmas"] = grid
    config["transformer"] = transformer
    metadata["config"] = json.dumps(config)
    return metadata
