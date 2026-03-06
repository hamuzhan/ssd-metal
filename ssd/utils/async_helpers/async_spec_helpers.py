import mlx.core as mx
from ssd.config import Config


def compute_megaspec_lookahead(MQ_LEN: int, K: int) -> int:
    return K + 1 + K * MQ_LEN


def make_glue_decode_input_ids(
    draft_tokens: mx.array,
    rec_tokens: mx.array,
) -> mx.array:
    assert draft_tokens.shape[0] == rec_tokens.shape[0]
    out = mx.concatenate([mx.expand_dims(rec_tokens, axis=1), draft_tokens], axis=1).reshape(-1)
    return out


def get_forked_recovery_tokens_from_logits(config: Config, logits: mx.array, cache_hits: mx.array, returned_tokens: mx.array, tokenizer):
    B, _, V_actual = logits.shape
    K = config.speculate_k
    fan_out_list = config.fan_out_list
    fan_out_list_miss = config.fan_out_list_miss
    assert cache_hits.shape == (B,)
    assert logits.shape[0] == B and logits.shape[1] == K + 1
    assert len(fan_out_list) == K + 1
    assert returned_tokens.shape == (B, K + 1)

    one_hot_mask = mx.one_hot(returned_tokens[:, 1:].astype(mx.int32), V_actual)
    neg_inf_mask = mx.where(one_hot_mask > 0, float('-inf'), 0.0)
    logits_first_k = logits[:, :-1, :] + neg_inf_mask
    logits = mx.concatenate([logits_first_k, logits[:, -1:, :]], axis=1)

    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    topk_idx = mx.argpartition(-logits, kth=k_max - 1, axis=-1)[:, :, :k_max]
    topk_vals = mx.take_along_axis(logits, topk_idx, axis=-1)
    sorted_order = mx.argsort(-topk_vals, axis=-1)
    topk_idx = mx.take_along_axis(topk_idx, sorted_order, axis=-1)

    hit_counts = mx.array(fan_out_list, dtype=mx.int32)
    miss_counts = mx.array(fan_out_list_miss, dtype=mx.int32)
    ch_bool = cache_hits.astype(mx.bool_).reshape(B, 1)
    counts_b = mx.where(
        ch_bool,
        mx.broadcast_to(hit_counts.reshape(1, -1), (B, K + 1)),
        mx.broadcast_to(miss_counts.reshape(1, -1), (B, K + 1)),
    )

    mx.eval(counts_b, topk_idx)
    results = []
    for b in range(B):
        row_tokens = []
        for pos in range(K + 1):
            count = int(counts_b[b, pos].item())
            row_tokens.append(topk_idx[b, pos, :count])
        results.append(mx.concatenate(row_tokens))
    idxs_flat = mx.stack(results)
    assert idxs_flat.shape == (B, sum(fan_out_list))

    return idxs_flat


def apply_sampler_x_rescaling(probs: mx.array, sampler_x: float, F: int) -> mx.array:
    sorted_probs = mx.sort(probs, axis=-1)
    threshold = mx.expand_dims(sorted_probs[..., -(F + 1)], axis=-1)
    mask = probs >= threshold
    result = mx.where(mask, probs * sampler_x, probs)
    result = result / mx.sum(result, axis=-1, keepdims=True)
    return result
