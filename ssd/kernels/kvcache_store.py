import mlx.core as mx


def store_kvcache(key: mx.array, value: mx.array, k_cache: mx.array, v_cache: mx.array, slot_mapping: mx.array):
    N = key.shape[0]
    D = k_cache.shape[1]
    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    mx.eval(slot_mapping)
    for i in range(N):
        slot = int(slot_mapping[i].item())
        if slot >= 0:
            k_cache[slot] = key_flat[i]
            v_cache[slot] = value_flat[i]
