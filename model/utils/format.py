import re
# ═══════════════════════════════════════════════════════════════
# Decode helper (fixed: clamp bins, skip short/empty segments)
# ═══════════════════════════════════════════════════════════════
def decode_prediction(pred: str, duration: float, num_bins: int = 300):
    """
    Parse model output into [{timestamp, sentence}].

    FIX: expects "<time=X><time=Y> sentence" (no space between time tokens).
    FIX: clamp bins to [0, num_bins-1] to avoid timestamp > duration.
    FIX: skip segments where start >= end or sentence too short.
    """
    # FIX: pattern matches "<time=X><time=Y> sentence" (no space between tokens)
    pattern = r"<time=(\d+)><time=(\d+)>\s*([^<]+)"
    results = []
    for match in re.finditer(pattern, pred):
        start_bin = int(match.group(1))
        end_bin   = int(match.group(2))
        sentence  = match.group(3).strip()

        # Clamp to valid range
        start_bin = max(0, min(start_bin, num_bins - 1))
        end_bin   = max(0, min(end_bin,   num_bins - 1))

        start = start_bin / (num_bins - 1) * duration
        end   = end_bin   / (num_bins - 1) * duration

        # Fix reversed timestamps
        if start > end:
            start, end = end, start

        # Skip noise segments
        if end - start < 0.5:
            continue
        if len(sentence.split()) < 3:
            continue

        results.append({
            "timestamp": [round(start, 2), round(end, 2)],
            "sentence":  sentence,
        })
    return results