import mlx.core as mx

_mask_cache = {
    'glue_and_rec_mask_hit': None,
    'glue_and_rec_mask_miss': None,
    'diag_components': None,
    'ones_tensor': None,
    'cached_params': None,
}


def get_mask_iter_i(i: int, prefix_len: int, K: int, F: int) -> mx.array:
    q_len = F * (K + 1)
    prefix_mask = mx.ones((q_len, prefix_len))
    row_idx = mx.arange(K + 1)
    col_idx = mx.arange(K + 1)
    tril = (col_idx.reshape(1, -1) <= row_idx.reshape(-1, 1)).astype(mx.int32)
    glue_and_rec_mask = mx.repeat(tril, repeats=F, axis=0)
    diags = [mx.eye(q_len) for _ in range(i + 1)]
    mask = mx.concatenate([prefix_mask, glue_and_rec_mask] + diags, axis=1)
    return mask.astype(mx.bool_)


def _precompute_mask_components(K: int, F: int, max_step: int, max_context_len: int, fan_out_list: list[int], fan_out_list_miss: list[int]):
    fan_out_t = mx.array(fan_out_list, dtype=mx.int32)
    fan_out_t_miss = mx.array(fan_out_list_miss, dtype=mx.int32)
    mx.eval(fan_out_t, fan_out_t_miss)

    tril = (mx.arange(K + 1).reshape(1, -1) <= mx.arange(K + 1).reshape(-1, 1)).astype(mx.int32)

    rows_hit = []
    for pos in range(K + 1):
        count = int(fan_out_t[pos].item())
        for _ in range(count):
            rows_hit.append(tril[pos])
    glue_and_rec_mask_hit = mx.stack(rows_hit)

    rows_miss = []
    for pos in range(K + 1):
        count = int(fan_out_t_miss[pos].item())
        for _ in range(count):
            rows_miss.append(tril[pos])
    glue_and_rec_mask_miss = mx.stack(rows_miss)

    MQ_LEN = glue_and_rec_mask_hit.shape[0]

    diag_components = {}
    for step in range(max_step + 1):
        diags = [mx.eye(MQ_LEN) for _ in range(step + 1)]
        if diags:
            diag_components[step] = mx.concatenate(diags, axis=1)
        else:
            diag_components[step] = mx.zeros((MQ_LEN, 0))

    ones_tensor = mx.ones((MQ_LEN, max_context_len))

    return glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor


def _get_custom_mask_optimized(context_lens, step, K, F, B, glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor, cache_hits):
    MQ_LEN = glue_and_rec_mask_hit.shape[0]
    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = tree_decode_added + glue_added

    mx.eval(context_lens, cache_hits)
    masks = []
    for b in range(B):
        prefix_len = int(context_lens[b].item()) - ttl_added
        prefix_mask = ones_tensor[:, :prefix_len]
        if int(cache_hits[b].item()) == 1:
            glue_and_rec_mask = glue_and_rec_mask_hit
        else:
            glue_and_rec_mask = glue_and_rec_mask_miss
        mask = mx.concatenate([prefix_mask, glue_and_rec_mask, diag_components[step]], axis=1)
        masks.append(mask.reshape(-1))

    return mx.concatenate(masks, axis=0).astype(mx.bool_)


def get_custom_mask_cached(config, context_lens, step, K, F, B, fan_out_list, fan_out_list_miss, cache_hits):
    global _mask_cache

    max_step = K + 1
    current_params = (K, F, max_step, config.max_model_len, tuple(fan_out_list), tuple(fan_out_list_miss))

    if (_mask_cache['cached_params'] is None or _mask_cache['cached_params'] != current_params):
        glue_hit, glue_miss, diag_comps, ones_t = _precompute_mask_components(
            K, F, max_step, config.max_model_len, fan_out_list, fan_out_list_miss,
        )
        _mask_cache['glue_and_rec_mask_hit'] = glue_hit
        _mask_cache['glue_and_rec_mask_miss'] = glue_miss
        _mask_cache['diag_components'] = diag_comps
        _mask_cache['ones_tensor'] = ones_t
        _mask_cache['cached_params'] = current_params

    return _get_custom_mask_optimized(
        context_lens, step, K, F, B,
        _mask_cache['glue_and_rec_mask_hit'],
        _mask_cache['glue_and_rec_mask_miss'],
        _mask_cache['diag_components'],
        _mask_cache['ones_tensor'],
        cache_hits,
    )


def get_custom_mask_vectorized(config, context_lens, step, K, B, cache_hits):
    fan_out_list = config.fan_out_list
    fan_out_list_miss = config.fan_out_list_miss
    fan_out_t = mx.array(fan_out_list, dtype=mx.int32)
    fan_out_t_miss = mx.array(fan_out_list_miss, dtype=mx.int32)
    mx.eval(fan_out_t, fan_out_t_miss)
    MQ_LEN = sum(fan_out_list)

    tril = (mx.arange(K + 1).reshape(1, -1) <= mx.arange(K + 1).reshape(-1, 1)).astype(mx.bool_)

    rows_hit = []
    for pos in range(K + 1):
        count = int(fan_out_t[pos].item())
        for _ in range(count):
            rows_hit.append(tril[pos])
    glue_hit = mx.stack(rows_hit)

    rows_miss = []
    for pos in range(K + 1):
        count = int(fan_out_t_miss[pos].item())
        for _ in range(count):
            rows_miss.append(tril[pos])
    glue_miss = mx.stack(rows_miss)

    eye = mx.eye(MQ_LEN, dtype=mx.bool_)
    diag_part = mx.concatenate([eye for _ in range(step + 1)], axis=1)

    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = tree_decode_added + glue_added

    mx.eval(context_lens, cache_hits)
    masks = []
    for b in range(B):
        ctx_len = int(context_lens[b].item())
        prefix_len = ctx_len - ttl_added
        prefix_mask = mx.ones((MQ_LEN, prefix_len), dtype=mx.bool_)
        if int(cache_hits[b].item()) == 1:
            glue = glue_hit
        else:
            glue = glue_miss
        mask = mx.concatenate([prefix_mask, glue, diag_part], axis=1)
        masks.append(mask.reshape(-1))

    return mx.concatenate(masks, axis=0).astype(mx.bool_)


def get_custom_mask(config, context_lens, step, K, F, B, cache_hits):
    if B <= 8:
        return get_custom_mask_cached(config, context_lens, step, K, F, B, fan_out_list=config.fan_out_list, fan_out_list_miss=config.fan_out_list_miss, cache_hits=cache_hits)
    else:
        return get_custom_mask_vectorized(config, context_lens, step, K, B, cache_hits)
