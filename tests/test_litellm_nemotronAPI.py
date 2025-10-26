from agents.extensions.models.litellm_model import LitellmModel
from agents import Agent, Runner, set_tracing_disabled

set_tracing_disabled(True)
import asyncio

model = LitellmModel(
    model="openai/mlx_models/mlx_model_nemotron_8b_quantized",
    api_key="fake_key_for_ollama",
    base_url="http://localhost:8080/v1"
)

prompt = "What are the key features of the latest version of the Nemotron model?"

agent = Agent(
    name="NemotronInfoAgent",
    model=model,
    instructions="Answer the following question about the Nemotron model:",
)

async def test_nemotron_info_agent():
    response = await Runner.run(agent, input=prompt)
    print("Agent Response:", response)
    assert "Nemotron" in response, "Response should contain information about Nemotron model"
    assert "features" in response, "Response should mention features of the Nemotron model"
    
if __name__ == "__main__":
    asyncio.run(test_nemotron_info_agent())