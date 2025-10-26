import asyncio

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel # didn't seem to work.
from agents.extensions.models.litellm_provider import LitellmProvider

set_tracing_disabled(True)
llm_model = "qwen3:0.6b"

@function_tool
def get_time():
    import time
    return time.ctime()

async def main():
    agent = Agent(
        name = 'assistant', 
        instructions = 'You are a helpful assistant.',
        model = LitellmProvider().get_model(f'ollama_chat/{llm_model}'),
        tools = [get_time],
    )
    
    result = await Runner.run(agent, "What is the current time?")
    
    print(result.final_output)
    
if __name__ == '__main__':
    asyncio.run(main())