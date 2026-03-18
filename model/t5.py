from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def load_tokenizer(local_path=None, num_bins=0):
    local_path = local_path if local_path is not None else "google-t5/t5-base"
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    
    # Tạo danh sách token thời gian: <times=0>, <times=1>, ...
    new_tokens = [f'<times={i}>' for i in range(num_bins)]
    print(new_tokens)
    if new_tokens:
        # Thêm vào tokenizer
        tokenizer.add_tokens(new_tokens)
    return tokenizer

def load_model(tokenizer, local_path=None):
    local_path = local_path if local_path is not None else "google-t5/t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
    
    # Quan trọng: Truyền len(tokenizer) đã được thêm tokens mới
    model.resize_token_embeddings(len(tokenizer))
    
    return model

def save_tokenizer(tokenizer, dir_=r'model/t5_finetune'):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    tokenizer.save_pretrained(dir_)

def save_model(model, dir_=r'model/t5_finetune'):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    model.save_pretrained(dir_)