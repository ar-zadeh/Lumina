from mlx_lm import generate, load

model, tokenizer = load("./mlx_models/mlx_model_nemotron_8b_quantized") #4bit quantized
# model, tokenizer = load("./mlx_models/mlx_model_llama_nemotron_8b_v1")

prompt = "What's the weather in SF?"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)