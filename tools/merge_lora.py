#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.model.skeletonllm_chat import InternVLChatModel

try:
    from transformers.modeling_utils import load_state_dict as hf_load_state_dict
except Exception:
    hf_load_state_dict = None

argparse = argparse.ArgumentParser()
argparse.add_argument('input_path', type=str, help='Path to the input model')
argparse.add_argument('output_path', type=str, help='Path to the output model')
args = argparse.parse_args()

print('Loading model...')
model = InternVLChatModel.from_pretrained(
    args.input_path, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16).eval()
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

# Eagerly instantiate skeleton renderer and reload its weights from shards
if getattr(model.config, 'use_skeleton', False):
    try:
        # Ensure the lazy module is constructed with correct H/W/device
        skel_mod = model._get_skeleton_renderer()
        # Also instantiate the inner differentiable renderer so that its params exist
        try:
            dev = getattr(model, 'device', None)
            if dev is None:
                dev = next(model.parameters()).device
            _ = skel_mod._ensure_renderer(dev)
        except Exception as e:
            print(f'Warning: failed to ensure inner renderer: {e}')

        # Collect all weight shard files (bin/safetensors) from input_path
        weight_files = []
        for idx_name in ['pytorch_model.bin.index.json', 'model.safetensors.index.json']:
            idx_file = os.path.join(args.input_path, idx_name)
            if os.path.exists(idx_file):
                try:
                    index = json.load(open(idx_file))
                    files = sorted(set(os.path.join(args.input_path, f) for f in index.get('weight_map', {}).values()))
                    weight_files.extend(files)
                except Exception as e:
                    print(f'Warning: failed to parse index file {idx_file}: {e}')

        if not weight_files:
            # Fallback: try common single-file patterns
            weight_files = sorted(glob.glob(os.path.join(args.input_path, 'pytorch_model*.bin'))) \
                         + sorted(glob.glob(os.path.join(args.input_path, 'model*.safetensors'))) \
                         + sorted(glob.glob(os.path.join(args.input_path, '*.safetensors')))

        reloaded = 0
        for wf in weight_files:
            try:
                if hf_load_state_dict is not None:
                    sd = hf_load_state_dict(wf)
                else:
                    if wf.endswith('.safetensors'):
                        try:
                            from safetensors.torch import load_file as safe_load_file  # type: ignore
                            sd = safe_load_file(wf)
                        except Exception:
                            sd = torch.load(wf, map_location='cpu')
                    else:
                        sd = torch.load(wf, map_location='cpu')
                _incompat = model.load_state_dict(sd, strict=False)
                reloaded += 1
            except Exception as e:
                print(f'Warning: failed to reload shard {wf}: {e}')
        if reloaded == 0:
            print('Warning: no extra shards reloaded; skeleton weights may be missing in output.')
        else:
            # Log whether skeleton params exist now
            num_skel_params = sum(1 for n, _ in model.named_parameters() if n.startswith('_skeleton_renderer_module'))
            print(f'Skeleton renderer instantiated; shards reloaded: {reloaded}; skeleton params: {num_skel_params}')
    except Exception as e:
        print(f'Warning: failed to instantiate/reload skeleton renderer: {e}')

def _collect_weight_files(base_dir):
    files = []
    for idx_name in ['pytorch_model.bin.index.json', 'model.safetensors.index.json']:
        idx_file = os.path.join(base_dir, idx_name)
        if os.path.exists(idx_file):
            try:
                index = json.load(open(idx_file))
                part_files = sorted(set(os.path.join(base_dir, f) for f in index.get('weight_map', {}).values()))
                files.extend(part_files)
            except Exception as e:
                print(f'Warning: failed to parse index file {idx_file}: {e}')
    if not files:
        files = sorted(glob.glob(os.path.join(base_dir, 'pytorch_model*.bin'))) \
             + sorted(glob.glob(os.path.join(base_dir, 'model*.safetensors'))) \
             + sorted(glob.glob(os.path.join(base_dir, '*.safetensors')))
    return files

def _reload_from_files(model_obj, files):
    reloaded_local = 0
    for wf in files:
        try:
            # IMPORTANT: avoid hf_load_state_dict here to prevent meta tensors during reload
            if wf.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file as safe_load_file  # type: ignore
                    sd = safe_load_file(wf)
                except Exception:
                    sd = torch.load(wf, map_location='cpu')
            else:
                sd = torch.load(wf, map_location='cpu')
            _ = model_obj.load_state_dict(sd, strict=False)
            reloaded_local += 1
        except Exception as e:
            print(f'Warning: failed to reload shard {wf}: {e}')
    return reloaded_local

# If LoRA flags are set but modules are not PEFT-wrapped yet, wrap and reload to populate LoRA weights
weight_files_all = _collect_weight_files(args.input_path)

# Capture what the input checkpoint expects, before the merge resets the flags.
_expected_llm_lora = int(getattr(model.config, 'use_llm_lora', 0) or 0)
_expected_use_skeleton = bool(getattr(model.config, 'use_skeleton', False))

if getattr(model.config, 'use_backbone_lora', 0):
    if not hasattr(model.vision_model, 'merge_and_unload'):
        try:
            model.wrap_backbone_lora(r=model.config.use_backbone_lora, lora_alpha=2 * model.config.use_backbone_lora)
            _ = _reload_from_files(model, weight_files_all)
        except Exception as e:
            print(f'Warning: failed to wrap/reload backbone LoRA: {e}')
    if hasattr(model.vision_model, 'merge_and_unload'):
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
        model.config.use_backbone_lora = 0

if getattr(model.config, 'use_llm_lora', 0):
    if not hasattr(model.language_model, 'merge_and_unload'):
        try:
            model.wrap_llm_lora(r=model.config.use_llm_lora, lora_alpha=2 * model.config.use_llm_lora)
            _ = _reload_from_files(model, weight_files_all)
        except Exception as e:
            print(f'Warning: failed to wrap/reload LLM LoRA: {e}')
    if hasattr(model.language_model, 'merge_and_unload'):
        model.language_model.merge_and_unload()
        model.language_model = model.language_model.model
        model.config.use_llm_lora = 0

# Verify the merge actually populated what we expect, instead of exiting 0 on a broken merge.
if _expected_use_skeleton:
    n_skel = sum(1 for n, _ in model.named_parameters() if n.startswith('_skeleton_renderer_module'))
    if n_skel == 0:
        raise RuntimeError(
            'Merge produced no DrAction renderer parameters; the input checkpoint likely lacks '
            '_skeleton_renderer_module.* shards. Aborting to avoid saving a renderer-less model.')
if _expected_llm_lora and int(getattr(model.config, 'use_llm_lora', 0) or 0):
    raise RuntimeError(
        'LLM LoRA was present in the input config but merge_and_unload did not run. '
        'Aborting to avoid silently dropping the LoRA deltas.')
residual_lora = [n for n, _ in model.named_parameters() if '.lora_' in n]
if residual_lora:
    raise RuntimeError(
        f'{len(residual_lora)} LoRA parameters remain after merge (e.g. {residual_lora[0]}). '
        'merge_and_unload did not fold them in. Aborting.')
print(f'Merge verified (use_skeleton={_expected_use_skeleton}, expected_llm_lora={_expected_llm_lora}, residual_lora=0).')

print('Saving model...')
# Use safe_serialization=False to avoid errors from tensor sharing in base configs
model.save_pretrained(args.output_path, safe_serialization=False)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)
print('Done!')
