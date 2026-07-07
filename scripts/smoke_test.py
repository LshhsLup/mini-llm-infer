import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./models/qwen3-0.6b"


def main():
    # ---- 第1步：检查环境 ----
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # ---- 第2步：加载tokenizer和模型 ----
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # ---- 第3步：构造输入 ----
    messages = [{"role": "user", "content": "用C++实现hello world。"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, 
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # ---- 第4步：生成 ----
    print("Running generate()...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t1 = time.time()

    # ---- 第5步：解码输出 + 统计 ----
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    print("=" * 50)
    print("Output:", text)
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Tokens generated: {generated.shape[0]}")
    print(f"Tokens/s: {generated.shape[0] / (t1 - t0):.2f}")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()