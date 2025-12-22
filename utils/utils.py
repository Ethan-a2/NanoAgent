import mlx.core as mx


def top_p_cal(probs, p):
    # https://cyrilzakka.github.io/llm-playbook/nested/topp.html
    sorted_indices = mx.argsort(probs)
    cum_probs = mx.cumsum(sorted_indices)
    valid_mask = cum_probs >= (1-p)
    return valid_mask

def min_p_cal(probs, p):
    # Find max probability per batch
    max_probs = mx.max(probs, axis=-1, keepdims=True)
    # Mask tokens below min_p * max_prob
    threshold = max_probs * p
    mask = probs >= threshold
    # Zero out filtered probs
    return mask

def top_k_cal(probs, k):
    topk_vals = mx.topk(probs, k+1)
    min_top_k = mx.min(topk_vals)
    mask = probs > min_top_k
    return mask

def sampler(
    logits: mx.array,
    min_p: float = None,
    top_p: float = None,
    top_k: int = None,
    temperature: float = 1.0,

) -> mx.array:
    """
    Min-p sampling (relative probability threshold).

    Args:
        logits: [vocab_size] or [batch, vocab_size]
        min_p: probability floor relative to max prob (e.g. 0.05)
        temperature: sampling temperature

    Returns:
        Sampled token indices
    """
    # Apply temperature
    # logits = logits / max(temperature, 1e-6)
    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    if top_p:
        top_p_mask = top_p_cal(probs, top_p)
    else:
        top_p_mask = 1
    if min_p:
        min_p_mask = min_p_cal(probs, min_p)
    else:
        min_p_mask = 1
    if top_k:
        top_k_mask = top_k_cal(probs, top_k)
    else:
        top_k_mask = 1
    
    probs = probs * top_p_mask * min_p_mask * top_k_mask
    # Renormalize
    probs = probs / mx.sum(
        probs, axis=-1, keepdims=True
    )
    # Sample
    _logits = mx.log(probs) / temperature
    return mx.random.categorical(_logits)


