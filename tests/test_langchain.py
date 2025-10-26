"""
one of the challenges i ran into is that i wanted my model to break things down into several tasks, and then solve them all
and then aggregate them using a final organizer agent. initially i tried to do this all in one agent, but the model kept
failing to generate all required parts. 

so I broke it down into multiple agents, and an orchestrator, but still the output was very finicky and unreliable.
meaning that sometimes the model would decide to output results itself.
or it would use all tools but missed the organizer step.
or the organizer would fail.
"""


import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

## define state (the information it needs to track during processing)
class PydanticGenerationState(TypedDict):
    # task description
    task_description: str
    
    # task schema
    task_schema: Dict[str, Any]
    
    # pydantic model code
    model_code: Optional[str]
    
    # extraction prompt
    extraction_prompt: Optional[str]
    
    # field mappings
    field_mappings: List[Dict[str, Any]]
    
    # keep track of conversation history for Agent
    messages: List[Dict[str, Any]]
    

model = ChatOpenAI(
    openai_api_base = "http://localhost:8080/v1",
    model = "mlx_model_nemotron_8b_quantized",
    temperature = 0.6,
    api_key='fake_key_for_ollama'
)

def read_task_description(state: PydanticGenerationState):
    task_description = state["task_description"]

    print("Alfred is processing the task description:", task_description)
    
    # no state changes needed here
    return {}

def generate_extraction_prompt(state: PydanticGenerationState):
    
    print("Alfred is generating the extraction prompt...")
    task_description = state["task_description"]
    task_schema = state["task_schema"]

    prompt = f"Generate a detailed extraction prompt for the following task: {task_description}\n"
    prompt += f"Here's the schema user wants to extract: {task_schema}\n"

    # call the model
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "extraction_prompt": response.content,
        "messages": new_messages
    }
    
def generate_pydantic_model(state: PydanticGenerationState):
    
    print("Alfred is generating the Pydantic model code...")
    task_description = state["task_description"]
    task_schema = state["task_schema"]
    task_extraction_prompt = state.get("extraction_prompt", "")

    prompt = f"Generate a Pydantic model for the following extraction task: {task_description}\n"
    prompt += f"Here is the schema to extract: {task_schema}\n"
    if task_extraction_prompt:
        prompt += f"Here is the extraction prompt to guide you: {task_extraction_prompt}\n"
    prompt += "Include proper type hints, Field descriptions, and validators where appropriate."

    # call the model
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "model_code": response.content,
        "messages": new_messages
    }
    
def generate_field_mappings(state: PydanticGenerationState):
    task_description = state["task_description"]
    task_schema = state["task_schema"]

    prompt = f"Generate field mappings for the following extraction task: {task_description}\n"
    prompt += f"Here is the schema to extract: {task_schema}\n"
    prompt += "Provide a list of field names with their descriptions."

    # call the model
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "field_mappings": response.content,
        "messages": new_messages
    }
    
def organize_results(state: PydanticGenerationState):
    model_code = state.get("model_code", "")
    extraction_prompt = state.get("extraction_prompt", "")
    field_mappings = state.get("field_mappings", "")

    print("\n" + "="*50)
    print("Generated Pydantic Model:\n", model_code)
    print("\nExtraction Prompt:\n", extraction_prompt)
    print("\nField Mappings:\n", field_mappings)
    print("="*50 + "\n")
    
    return {}

# Define routing logic -- we don't have any; we want to run all steps sequentially

# Create the State Graph and Define Edges
code_graph = StateGraph(PydanticGenerationState)

# Add nodes
code_graph.add_node("read_task_description", read_task_description)
code_graph.add_node("generate_extraction_prompt", generate_extraction_prompt)
code_graph.add_node("generate_pydantic_model", generate_pydantic_model)
code_graph.add_node("generate_field_mappings", generate_field_mappings)
code_graph.add_node("organize_results", organize_results)

# start the edges
code_graph.add_edge(START, "read_task_description")

# Add edges - to define the flow
code_graph.add_edge("read_task_description", "generate_extraction_prompt")
code_graph.add_edge("generate_extraction_prompt", "generate_pydantic_model")
code_graph.add_edge("generate_pydantic_model", "generate_field_mappings")
code_graph.add_edge("generate_field_mappings", "organize_results")
code_graph.add_edge("organize_results", END)

# compile the graph
compiled_graph = code_graph.compile()

print(compiled_graph.get_graph().draw_ascii())



# result = compiled_graph.invoke({
#     "task_description": "Extract temperature and pressure data from scientific articles.",
#     "task_schema": {
#         "temperature": "float - The temperature value in Celsius.",
#         "pressure": "float - The pressure value in Pascals."
#     },
#     "model_code": None,
#     "extraction_prompt": None,
#     "field_mappings": [],
#     "messages": [],
# })

# print("Final State:", result)