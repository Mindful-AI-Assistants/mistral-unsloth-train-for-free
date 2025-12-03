# Minimal inference example using a Hugging Face tokenizer and either the MinimalLM scaffold or HF model.
import argparse
import torch
from transformers import AutoTokenizer
from model import MinimalLM, load_hf_model

def greedy_generate(model, tokenizer, prompt, max_new_tokens=32, device="cpu"):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = torch.cat([tokens, next_token], dim=1)
        return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--hf_model", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) or args.device=="cuda" else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if args.hf_model:
        model = load_hf_model(args.hf_model, device=device)
    else:
        model = MinimalLM(vocab_size=tokenizer.vocab_size, d_model=256, n_layers=2, n_head=4, d_ff=1024)
    model = model.to(device)
    print(greedy_generate(model, tokenizer, args.prompt, device=device))

if __name__ == "__main__":
    main()
