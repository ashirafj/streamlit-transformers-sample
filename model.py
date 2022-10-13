
from transformers import T5Tokenizer, AutoModelForCausalLM
import torch

# GPU が利用可能な場合には GPU を使用する
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 モデルを読み込み
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").to(DEVICE)

# 文章生成用の関数を定義
def generate_sentences(prompt, max_length, num_return_sequencces):
    # prompt: 元となる文章
    # max_length: 生成する最大文字数
    # num_return_sequences: 生成する文章の数

    # 文字列をトークン化して ID に変換
    encoding = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    # 続きの文章を生成
    outputs = model.generate(encoding, do_sample=True, max_length=max_length, num_return_sequences=num_return_sequencces)

    # outputs は ID で表されているので、単語に戻す
    decoded_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return decoded_sentences
