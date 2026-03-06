import os
from glob import glob
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
from tqdm import tqdm
import numpy as np


def _get_nested_attr(obj, path):
    parts = path.split('.')
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def _set_nested_attr(obj, path, value):
    parts = path.split('.')
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def _load_tensor(tensor_data) -> mx.array:
    if isinstance(tensor_data, mx.array):
        return tensor_data
    return mx.array(np.array(tensor_data))


def _find_module_for_param(model, param_name):
    parts = param_name.rsplit('.', 1)
    if len(parts) == 2:
        module_path, attr_name = parts
        module = _get_nested_attr(model, module_path)
        return module, attr_name
    return model, param_name


def load_embedding_from_target(model, target_path, target_hidden_size=None, draft_hidden_size=None):
    if target_hidden_size is not None and draft_hidden_size is not None:
        if target_hidden_size != draft_hidden_size:
            print(f"[load_model] Skipping target embeddings: target hidden_size={target_hidden_size} != draft hidden_size={draft_hidden_size}")
            return False

    target_keys = ["model.embed_tokens.weight", "embed_tokens.weight"]

    safetensor_files = glob(os.path.join(target_path, "*.safetensors"))
    for file in safetensor_files:
        try:
            with safe_open(file, "numpy") as f:
                keys = list(f.keys())
                for key in target_keys:
                    if key in keys:
                        print(f"[load_model] Found embedding {key} in {file}")
                        tensor = mx.array(f.get_tensor(key))
                        module, attr = _find_module_for_param(model, "model.embed_tokens")
                        if hasattr(module, 'weight_loader'):
                            module.weight_loader(tensor)
                        else:
                            module.weight = tensor
                        return True
        except Exception as e:
            print(f"[load_model] Error reading safetensor {file}: {e}")
            continue

    return False


def load_eagle_model(model, path, packed_modules_mapping, target_path=None, target_hidden_size=None):
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    state_dict = {}

    if safetensor_files:
        for file in safetensor_files:
            try:
                with safe_open(file, "numpy") as f:
                    for key in f.keys():
                        state_dict[key] = mx.array(f.get_tensor(key))
                print(f"[load_model] Loaded {len(state_dict)} weights from {file}")
                break
            except Exception as e:
                print(f"[load_model] Error reading safetensor {file}: {e}")
                continue

    if len(state_dict) == 0:
        bin_file = os.path.join(path, "pytorch_model.bin")
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"No safetensors or pytorch_model.bin found at {path}")
        import torch
        torch_state = torch.load(bin_file, map_location="cpu")
        for k, v in torch_state.items():
            state_dict[k] = mx.array(v.numpy())

    if hasattr(model, 'd2t') and 'd2t' in state_dict:
        d2t_tensor = state_dict['d2t']
        model.d2t = {i: int(d2t_tensor[i].item()) for i in range(d2t_tensor.shape[0])}
        model.d2t_tensor = d2t_tensor.astype(mx.int32)
        print(f"[load_model] Loaded d2t dictionary with {len(model.d2t)} entries")

    if hasattr(model, 't2d') and 't2d' in state_dict:
        t2d_tensor = state_dict['t2d']
        model.t2d = {i: int(t2d_tensor[i].item()) for i in range(t2d_tensor.shape[0])}
        model.t2d_tensor = t2d_tensor.astype(mx.int32)
        print(f"[load_model] Loaded t2d dictionary with {len(model.t2d)} entries")

    found_embed_tokens = any('embed_tokens' in k for k in state_dict.keys())

    if not found_embed_tokens:
        if target_path:
            draft_hidden_size = model.config.hidden_size if hasattr(model, 'config') else None
            print(f"[load_model] 'embed_tokens' not found in draft weights. Loading from target: {target_path}")
            if not load_embedding_from_target(model, target_path, target_hidden_size, draft_hidden_size):
                raise ValueError(f"[load_model] Could not load embeddings from target or draft model")

    for weight_name, loaded_weight in tqdm(state_dict.items(), desc="Loading EAGLE3 weights"):
        if weight_name in ['d2t', 't2d']:
            continue

        is_packed = False
        for k, (v, shard_id) in packed_modules_mapping.items():
            if k in weight_name:
                param_name = weight_name.replace(k, v)
                module_path = param_name.rsplit('.', 1)[0]
                module = _get_nested_attr(model, module_path)
                module.weight_loader(loaded_weight, shard_id)
                is_packed = True
                break

        if is_packed:
            continue

        if weight_name == 'midlayer.hidden_norm.weight':
            param_name = 'model.layer.conditioning_feature_ln.weight'
        elif weight_name.startswith('midlayer.'):
            param_name = weight_name.replace('midlayer.', 'model.layer.')
        elif weight_name == 'norm.weight':
            param_name = 'final_norm.weight'
        elif weight_name == 'embed_tokens.weight':
            param_name = 'model.embed_tokens.weight'
        else:
            param_name = weight_name

        module_path = param_name.rsplit('.', 1)[0]
        attr_name = param_name.rsplit('.', 1)[1]
        module = _get_nested_attr(model, module_path)
        if hasattr(module, 'weight_loader') and attr_name == 'weight':
            module.weight_loader(loaded_weight)
        else:
            setattr(module, attr_name, loaded_weight)


def load_safetensors_model(model, path, packed_modules_mapping):
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    for file in tqdm(safetensor_files, desc="Loading model files"):
        with safe_open(file, "numpy") as f:
            for weight_name in f.keys():
                loaded_weight = mx.array(f.get_tensor(weight_name))
                is_packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        module_path = param_name.rsplit('.', 1)[0]
                        module = _get_nested_attr(model, module_path)
                        module.weight_loader(loaded_weight, shard_id)
                        is_packed = True
                        break

                if not is_packed:
                    parts = weight_name.rsplit('.', 1)
                    module_path = parts[0]
                    attr_name = parts[1] if len(parts) > 1 else weight_name
                    module = _get_nested_attr(model, module_path)
                    if hasattr(module, 'weight_loader') and attr_name == 'weight':
                        module.weight_loader(loaded_weight)
                    else:
                        setattr(module, attr_name, loaded_weight)


def load_model(model, path, target_path=None, target_hidden_size=None):
    print(f"[load_model] loading model from {path}")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    is_eagle = 'eagle' in path.lower()

    if is_eagle:
        load_eagle_model(model, path, packed_modules_mapping, target_path=target_path, target_hidden_size=target_hidden_size)
    else:
        load_safetensors_model(model, path, packed_modules_mapping)

    print(f"[load_model] finished loading model from {path}")
