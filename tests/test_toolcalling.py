import json

from typing_extensions import TypedDict, Any
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.models.litellm_provider import LitellmProvider
from agents import Agent, FunctionTool, RunContextWrapper, function_tool, set_tracing_disabled, Runner


set_tracing_disabled(True)
# litellm_model = LitellmModel(
#     model="openai/default_model", ## this is very important to use default_model argument otherwise it will load ANOTHER MODEL and doesn't use what the server has.
#     api_key="fake_key_for_ollama",
#     base_url="http://localhost:8080/v1"
# )

EXTRACTION_LLM_MODEL = "qwen3:0.6b"
# ollama_model = LitellmProvider().get_model(f'ollama_chat/{EXTRACTION_LLM_MODEL}')
# ollama_model = LitellmProvider().get_model(f'openai/default_model')

api_key = input("Enter your Gemini API key: ")
gemini_model = LitellmModel(model='gemini/gemini-2.5-flash', api_key=api_key)

class Location(TypedDict):
    lat: float
    long: float

@function_tool  
async def fetch_weather(location: Location) -> str:
    
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "sunny"


@function_tool(name_override="fetch_data")  
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<file contents>"


agent = Agent(
    name="Assistant",
    model = gemini_model,
    tools=[fetch_weather, read_file],  
)

# for tool in agent.tools:
#     if isinstance(tool, FunctionTool):
#         print(tool.name)
#         print(tool.description)
#         print(json.dumps(tool.params_json_schema, indent=2))
#         print()
        
async def main():
    location = {"lat": 40.7128, "long": -74.0060}
    weather = await Runner.run(agent, f"What's the weather like in {location}?")
    print(f"The weather is: {weather}")

    file_contents = await Runner.run(agent, f"Read the file at example.txt")
    print(f"File contents: {file_contents}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())