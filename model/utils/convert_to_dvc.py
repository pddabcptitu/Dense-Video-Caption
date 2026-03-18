import math

def time_to_token(t, duration, num_bins):
    return round(t / duration * (num_bins - 1))


def convert_activitynet(sample, num_bins=100):
    duration = sample["duration"]
    timestamps = sample["timestamps"]
    sentences = sample["sentences"]

    output = []

    for (start, end), text in zip(timestamps, sentences):
        start_token = time_to_token(start, duration, num_bins)
        end_token = time_to_token(end, duration, num_bins)

        if end_token <= start_token:
            continue

        output.append(f"<times={start_token}> <times={end_token}> {text}")

    return " ".join(output)