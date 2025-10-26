from mlx_lm import generate, load

model, tokenizer = load("mlx-community/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct-4bit")

prompt = 'Imagine you are in pokemon Emerald. Come up with the few key actions you would take starting from the game screen'

if tokenizer.chat_template is not None:
    messages = [{'role': "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    
response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=2048)