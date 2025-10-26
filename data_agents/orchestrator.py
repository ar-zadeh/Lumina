from .agents_collection import *
from agents import set_tracing_disabled, Runner
import asyncio

set_tracing_disabled(True)

async def main():
    user_input = input("Enter your extraction intention: ")
    result = await Runner.run(schema_agent, user_input)
    print("Schema Recommendation:")
    print(type(result.final_output))
    print(result.final_output.model_dump_json(indent=2))
    
if __name__ == '__main__':
    asyncio.run(main())