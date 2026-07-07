"""
下载模型到本地 models/ 目录。
用法:
    python scripts/01_download_model.py
    python scripts/01_download_model.py --model Qwen/Qwen3-0.6B --out models/Qwen3-0.6B
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

def main():
    parser = argparse.ArgumentParser(description="下载HuggingFace模型到本地目录")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace模型repo id, 默认: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="本地保存目录, 默认根据模型名自动生成到 models/ 下"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace access token(私有/受限模型需要, 可选)"
    )
    args = parser.parse_args()

    if args.out is None:
        # 自动生成目录名, 例如 Qwen/Qwen3-0.6B -> models/Qwen3-0.6B
        model_short_name = args.model.split("/")[-1].lower()
        out_dir = Path("models") / model_short_name
    else:
        out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"下载模型: {args.model}")
    print(f"保存到: {out_dir.resolve()}")

    snapshot_download(
        repo_id=args.model,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        token=args.token,
        # 只下推理需要的文件, 跳过训练用的额外文件(比如.bin如果有.safetensors就跳过.bin)
        allow_patterns=[
            "*.json", "*.safetensors", "*.model",
            "*.txt", "tokenizer*", "*.jinja"
        ],
    )

    print("下载完成, 目录内容:")
    for f in sorted(out_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:40s} {size_mb:.1f} MB")

if __name__ == "__main__":
    main()