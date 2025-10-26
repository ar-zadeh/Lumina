from mlx_lm import convert

# print(help(convert))
repo = "./mlx_model_llama_nemotron_8b_v1"
upload_repo = "bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)