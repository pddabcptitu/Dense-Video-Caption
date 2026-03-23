from transformers import T5Tokenizer

def get_tokenizer(path, num_bins=0):
    tok = T5Tokenizer.from_pretrained(path)
    if num_bins:
        tok.add_tokens([f"<time={i}>" for i in range(num_bins)])
    return tok