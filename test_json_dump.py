from data_agents.agents_collection import *
from agents import Runner

import asyncio


async def main():
    with open("debug_pydantic_prompt.txt", "r") as f:
        prompt = f.read()
        
    result = await Runner.run(pydantic_agent, prompt)
    print("Pydantic Model Recommendation:")
    # print(type(result.final_output))
    print(result.final_output.model_dump_json(indent=2))

if __name__ == '__main__':
    asyncio.run(main())