# note that you need to run the local mlx server first
# python -m mlx_lm server --model ../mlx_models/mlx_model_nemotron_8b_quantized --host 0.0.0.0 --port 8080 --max-tokens=32000
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from agents import set_default_openai_client, Agent, Runner, set_tracing_disabled, set_default_openai_api, OpenAIChatCompletionsModel, function_tool
from agents.extensions.models.litellm_model import LitellmModel


EXTRACTION_LLM_MODEL = "qwen3:0.6b"

## Approach 4: Using LitellmModel with custom configuration
# os.environ["LOCAL_API_URL"] = "http://localhost:8080/v1"
# os.environ["LOCAL_API_KEY"] = "fake"
set_tracing_disabled(True)
litellm_model = LitellmModel(
    model="openai/default_model", ## this is very important to use default_model argument otherwise it will load ANOTHER MODEL and doesn't use what the server has.
    api_key="fake_key_for_ollama",
    base_url="http://localhost:8080/v1"
)

@function_tool
async def get_weather_info(location: str) -> str:
    """
    A function to get weather information for a given location. This is a placeholder function that simulates fetching weather data.
    
    Args:
        location (str): The location for which to get the weather information.

    Returns:
        str: Simulated weather information for the location.
    """
    return f"Weather information for {location}: Sunny, 25°C"

agent = Agent(
    name="NemotronInfoAgent",
    model=litellm_model,
    instructions="Answer the following question:",
    tools=[get_weather_info]
)

async def main():
    prompt = "What is the weather like in New York?"
    response = await Runner.run(agent, input=prompt)
    print("Agent Response:", response)
    assert "weather" in response, "Response should contain information about the weather"
    assert "New York" in response, "Response should mention New York"


## Approach 3: Using AsyncOpenAI directly with custom configuration
### PROBLEM: TOOLCALLING does not work, because it doesn't parse it.
# custom_client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key='fake_key_for_ollama')
# # set_default_openai_client(custom_client, use_for_tracing=False)
# # set_default_openai_api("chat_completions")
# set_tracing_disabled(True)

# @function_tool
# def get_weather_info(location: str) -> str:
#     """
#     A function to get weather information for a given location. This is a placeholder function that simulates fetching weather data.
    
#     Args:
#         location (str): The location for which to get the weather information.

#     Returns:
#         str: Simulated weather information for the location.
#     """
#     return f"Weather information for {location}: Sunny, 25°C"

# agent = Agent(
#     name="NemotronInfoAgent",
#     model=OpenAIChatCompletionsModel(model="../mlx_models/mlx_model_nemotron_8b_quantized", openai_client=custom_client),
#     instructions="Answer the following question about the Nemotron model:",
#     tools=[get_weather_info]
# )

# async def main():
#     prompt = "What are the key features of the latest version of the Nemotron model?"
#     response = await Runner.run(agent, prompt)
#     print("Agent Response:", response)
#     assert "Nemotron" in response, "Response should contain information about Nemotron model"
#     assert "features" in response, "Response should mention features of the Nemotron model"


## Approach 2: Using ChatOpenAI with custom configuration
# chat = ChatOpenAI(
#     openai_api_base = "http://localhost:8080/v1",
#     model = "../mlx_models/mlx_model_nemotron_8b_quantized",
#     temperature = 0.6,
#     api_key='fake_key_for_ollama'
# )
# messages = [
#     SystemMessage(content="You are an onboarding assistant."),
#     HumanMessage(content="Welcome our new customer!")
# ]

# response = chat.invoke(messages)
# print(response)


## Approach 1: Using ChatOpenAI with custom configuration
# async def get_chat_completion(messages, model="../mlx_models/mlx_model_nemotron_8b_quantized"):
#     response = await chat2.chat.completions.create(
#         messages=messages,
#         model=model,
#     )
#     return response.choices[0].message.content

# chat2 = AsyncOpenAI(
#     base_url = "http://localhost:8080/v1",
#     api_key='fake_key_for_ollama'
# )


# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"},
# ]
# async def main():
#     completion = await get_chat_completion(messages)
#     print(completion)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())