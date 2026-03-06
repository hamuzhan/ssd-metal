import mlx.core as mx
from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling


def _gumbel_sample(probs: mx.array) -> mx.array:
    u = mx.random.uniform(shape=probs.shape)
    scores = probs / (-mx.log(u + 1e-10) + 1e-10)
    return mx.argmax(scores, axis=-1)


def verify(
    logits_p: mx.array,
    logits_q: mx.array,
    speculations: mx.array,
    temperatures_target: mx.array,
    temperatures_draft: mx.array,
    cache_hits: mx.array | None = None,
    sampler_x: float | None = None,
    async_fan_out: int | None = None,
    jit_speculate: bool = False,
) -> tuple[list[list[int]], list[int]]:
    B, Kp1, V = logits_p.shape
    K = Kp1 - 1

    draft_tokens = speculations[:, 1:]
    preds_p = mx.argmax(logits_p, axis=-1)

    matches = draft_tokens == preds_p[:, :-1]
    any_mismatch = mx.logical_not(matches).any(axis=1)
    first_mismatch = mx.argmax(mx.logical_not(matches).astype(mx.int32), axis=1)
    accept_greedy = mx.where(any_mismatch, first_mismatch, mx.full(first_mismatch.shape, K, dtype=mx.int32))

    batch_idx = mx.arange(B)
    mx.eval(accept_greedy)
    rec_greedy = mx.array([int(preds_p[b, int(accept_greedy[b].item())].item()) for b in range(B)], dtype=mx.int32)

    temps_t = temperatures_target
    temps_q = temperatures_draft

    base_ratio_rows = mx.logical_or(temps_t > 0, temps_q > 0)
    if jit_speculate:
        ratio_rows = base_ratio_rows
    else:
        if cache_hits is not None:
            ratio_rows = mx.logical_and(base_ratio_rows, cache_hits.astype(mx.bool_))
        else:
            ratio_rows = mx.zeros_like(base_ratio_rows)

    mx.eval(ratio_rows)
    do_any_ratio = bool(ratio_rows.any().item())
    need_p_probs = bool((temps_t > 0).any().item()) or do_any_ratio

    probs_p = None
    if need_p_probs:
        logits_p_f = logits_p.astype(mx.float32)
        t = mx.maximum(mx.expand_dims(mx.expand_dims(temps_t, axis=-1), axis=-1), 1e-8)
        scaled_logits = logits_p_f / t
        probs_p = mx.softmax(scaled_logits, axis=-1)
        zero_t = temps_t == 0
        mx.eval(zero_t)
        if bool(zero_t.any().item()):
            argmax_p = mx.argmax(logits_p_f, axis=-1)
            greedy_onehot = mx.one_hot(argmax_p, V).astype(mx.float32)
            probs_p = mx.where(mx.expand_dims(mx.expand_dims(zero_t, -1), -1), greedy_onehot, probs_p)

    if do_any_ratio:
        logits_q_f = logits_q.astype(mx.float32)
        tq = mx.maximum(mx.expand_dims(mx.expand_dims(temps_q, axis=-1), axis=-1), 1e-8)
        probs_q = mx.softmax(logits_q_f / tq, axis=-1)

        if sampler_x is not None:
            assert async_fan_out is not None
            probs_q = apply_sampler_x_rescaling(probs_q, sampler_x, async_fan_out)

        p_all = probs_p[:, :K, :]
        draft_idx = mx.expand_dims(draft_tokens, axis=-1)
        mx.eval(draft_idx, p_all, probs_q)
        p_vals = mx.take_along_axis(p_all, draft_idx, axis=-1).squeeze(-1)
        q_vals = mx.take_along_axis(probs_q, draft_idx, axis=-1).squeeze(-1)

        accept_probs = mx.minimum(p_vals / (q_vals + 1e-10), 1.0)
        rand = mx.random.uniform(shape=accept_probs.shape)
        accepts = rand <= accept_probs

        rej_any = mx.logical_not(accepts).any(axis=1)
        first_rej = mx.argmax(mx.logical_not(accepts).astype(mx.int32), axis=1)
        accept_ratio = mx.where(rej_any, first_rej, mx.full(first_rej.shape, K, dtype=mx.int32))

        accept_until = mx.where(ratio_rows, accept_ratio, accept_greedy)
    else:
        accept_until = accept_greedy

    mx.eval(accept_until)

    if probs_p is None:
        rec_final_list = rec_greedy.tolist()
    else:
        rec_final = mx.zeros((B,), dtype=mx.int32)
        rec_list = []
        for b in range(B):
            au = int(accept_until[b].item())
            t_val = float(temps_t[b].item())
            if t_val == 0:
                rec_list.append(int(preds_p[b, au].item()))
            else:
                p_fb = probs_p[b, au]
                if do_any_ratio and bool(ratio_rows[b].item()) and au < K:
                    q_safe = min(au, K - 1)
                    q_fb = probs_q[b, q_safe]
                    adj = mx.maximum(p_fb - q_fb, 0.0)
                    s = adj.sum()
                    if float(s.item()) > 0:
                        dist = adj / s
                    else:
                        dist = p_fb / p_fb.sum()
                    rec_list.append(int(_gumbel_sample(mx.expand_dims(dist, 0)).item()))
                else:
                    p_sum = p_fb.sum()
                    dist = p_fb / p_sum
                    rec_list.append(int(_gumbel_sample(mx.expand_dims(dist, 0)).item()))
        rec_final_list = rec_list

    accepted_suffixes: list[list[int]] = []
    mx.eval(speculations, draft_tokens)
    starts = speculations[:, 0].tolist()
    counts = accept_until.tolist()

    for b in range(B):
        n = counts[b]
        suffix = [starts[b]] + draft_tokens[b, :n].tolist()
        accepted_suffixes.append(suffix)

    return accepted_suffixes, rec_final_list
