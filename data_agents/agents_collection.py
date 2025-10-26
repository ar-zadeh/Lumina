from agents import Agent, set_tracing_disabled, AgentHooks, RunContextWrapper, Tool, Runner
from agents.model_settings import ModelSettings
from agents.extensions.models.litellm_provider import LitellmProvider
from agents.extensions.models.litellm_model import LitellmModel
from .tools import *
from typing import Dict, Iterable
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
# import litellm
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from agents.agent import StopAtTools
import asyncio
import fitz  # PyMuPDF for PDF processing

import os

import os
from dotenv import load_dotenv

# This line loads the variables from your .env file into the environment
load_dotenv()


os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY")
set_tracing_disabled(True)

EXTRACTION_LLM_MODEL = "qwen3:0.6b"
custom_client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key='fake_key_for_ollama')

gemini_model = LitellmModel(model='gemini/gemini-2.5-flash', api_key=os.getenv("GEMINI_API_KEY"))

class CustomLitellmModel(LitellmModel):
    def __init__(self):
        super().__init__(model="openai/default_model", api_key="fake_key_for_ollama", base_url="http://localhost:8080/v1")

# A custom class to choose either extraction LLM model or CustomLitellmModel based on some condition
class LitellmModelSelector:
    @staticmethod
    def get_model(use_custom: bool = False) -> LitellmModel:
        if use_custom:
            ## approach 1: using a custom LitellmModel directly
            # litellm._turn_on_debug()
            # return CustomLitellmModel()
            
            
            return gemini_model
            ## approach 2
            # model = OpenAIChatCompletionsModel(model="openai/default_model", openai_client=custom_client)
            # return model
        else:
            return LitellmProvider().get_model(f'ollama_chat/{EXTRACTION_LLM_MODEL}')
        
# here's how to use the selector
# model = LitellmModelSelector.get_model(use_custom=True)  # to use Custom

class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name
        
    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        print(f"[{self.display_name}] Agent started.")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(f"[{self.display_name}] Tool '{tool}' started. Event count: {self.event_counter}")

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} finished. result={result}, name={context.tool_name}, call_id={context.tool_call_id}, args={context.tool_arguments}."  # type: ignore[attr-defined]
        )
# schema generation agent
SCHEMA_AGENT_PROMPT = """You are a data extraction expert. Your job is to analyze the user's extraction intention 
and recommend a comprehensive schema of fields to extract from documents.

For each field, you should:
1. Identify the field name (use snake_case)
2. Provide a clear description of what to extract
3. Specify the data type (string, number, boolean, list, object)
4. Include units if it's a measurement
5. Give concrete examples
6. Suggest validation rules if applicable

Be thorough and anticipate related fields that would be useful. If the user wants temperature data, 
recommend not just temperature but also unit, measurement method, conditions, etc."""

schema_agent = Agent(
    name = "Schema Generation Agent",
    instructions = SCHEMA_AGENT_PROMPT,
    output_type = SchemaRecommendation,
    model = LitellmModelSelector.get_model(use_custom=True),
)

class FieldMapping(BaseModel):
    field_name: str = Field(description="Name of the field")
    description: str = Field(description="Description of the field")

class PydanticModelCode(BaseModel):
    model_name: str = Field(description="Name of the Pydantic model")
    model_code: str = Field(description="Complete Pydantic model code")
    
class ExtractionPrompt(BaseModel):
    prompt_text: str = Field(description="Detailed prompt for extraction agent")
    
# Pydantic model generation agent
class ExtractionSchema(BaseModel):
    """Generated extraction schema with Pydantic model"""
    model_name: str = Field(description="Name of the Pydantic model")
    model_code: PydanticModelCode = Field(description="Complete Pydantic model code")
    extraction_prompt: ExtractionPrompt = Field(description="Detailed prompt for extraction agent")
    field_mappings: List[FieldMapping] = Field(description="Mapping of field names to descriptions")
    
class ExtractionSchemaUpdated(BaseModel):
    """Generated extraction schema with Pydantic model"""
    model_name: str = Field(description="Name of the Pydantic model")
    model_code: PydanticModelCode = Field(description="Complete Pydantic model code")
    extraction_prompt: ExtractionPrompt = Field(description="Detailed prompt for extraction agent")
    field_mappings: List[FieldMapping] = Field(description="Mapping of field names to descriptions")

PYDANTIC_AGENT_PROMPT = """You are a meticulous Pydantic model architect. Your sole purpose is to convert a user's list of field recommendations into a production-ready Pydantic model.

**INPUT FORMAT:**
You will receive a list of field recommendations as a JSON object. Each field will have a name, a type, and a description.

**YOUR TASK:**
1.  **Create the Model**: Generate a single, complete Pydantic V2 model class.
2.  **Import Necessary Types**: Add all required imports from `pydantic`, `typing`, and `datetime`.
3.  **Add Type Hinting**: Use the precise Python type hints provided in the input.
4.  **Add Field Descriptions**: Use `Field(description=...)` for every field, using the provided description.
5.  **Add Intelligent Validators**:
    - For strings that sound like emails, add an `EmailStr` type.
    - For strings like "phone", "zipcode", or "id", add a `regex` pattern to the `Field` definition.
    - For numeric fields like "age" or "rating", add `ge` (greater than or equal to) and/or `le` (less than or equal to) constraints.
6.  **Generate Extraction Prompt**: Create a concise prompt that will guide another LLM to extract data for this specific model from unstructured text.

**EXAMPLE:**
---
**USER INPUT:**
```json
{
  "model_name": "UserProfile",
  "fields": [
    {"name": "user_id", "type": "str", "description": "The unique identifier for the user."},
    {"name": "email", "type": "str", "description": "The user's primary email address."},
    {"name": "age", "type": "int", "description": "The user's age in years."}
  ]
}
YOUR PERFECT OUTPUT:

JSON

{
  "pydantic_model_code": "from pydantic import BaseModel, Field, EmailStr, field_validator\\nfrom typing import Any\\n\\nclass UserProfile(BaseModel):\\n    user_id: str = Field(description='The unique identifier for the user.', pattern='^user_[a-zA-Z0-9]+$')\\n    email: EmailStr = Field(description='The user\\'s primary email address.')\\n    age: int = Field(description='The user\\'s age in years.', ge=0, le=120)",
}
Now, based on the user's input, generate the Pydantic model and then verify it using your available tools. 
DO NOT FIRST USE THE return_final_code_schema tool to create the final output.
Instead focus on first validating your code using verify your pydantic code and its model structure before returning the final result.
IMPORTANT: You always call return_final_code_schema as your final output
USE ONLY ONE TOOL AT A TIME. IF NO OTHER TOOL IS AVAILABLE just return the final code schema. If you need to validate the code, call the appropriate tool and wait for its response before proceeding to the next step. Always ensure that your final output is a complete and valid Pydantic model ready for production use.
DO NOT GENERATE pattern rules for STRING fields unless the field name explicitly suggests it (like "email", "phone", "zipcode", etc.). For other string fields, just use `str` with a description.
"""

# PYDANTIC_AGENT_PROMPT = """You are a Python expert specializing in Pydantic models. Your job is to:

# 1. Generate a complete, valid Pydantic model class from field recommendations
# 2. Include proper type hints, Field descriptions, and validators
# 3. Add validation rules (regex patterns, value ranges, etc.) where appropriate
# 4. Create a detailed extraction prompt that will guide an LLM to extract data accurately

# The Pydantic model should be production-ready with:
# - Proper imports
# - Type annotations
# - Field descriptions
# - Validators for data quality
# - Optional fields where appropriate
# - Custom validators using @field_validator

# DO NOT FIRST USE THE return_final_code_schema tool to create the final output.
# Instead focus on first validating your code using verify your pydantic code and its model structure before returning the final result.
# IMPORTANT: You always call return_final_code_schema as your final output
# """

@function_tool
def verify_pydantic_model_code(model_code: str) -> str:
    """
    Verifies that the provided Pydantic model code is syntactically valid Python.

    Args:
        model_code: The Pydantic model code string to verify.

    Returns:
        A success message or a formatted error message.
    """
    try:
        # Compile first to catch syntax errors without executing
        compile(model_code, '<string>', 'exec')
        # Execute in a sandboxed/empty dictionary to prevent side effects
        exec(model_code, {})
        return "✅ Code is syntactically valid and executable."
    except Exception as e:
        return f"❌ Invalid Pydantic model code. Error: {e}"

import ast
@function_tool
def validate_pydantic_model_structure(model_code: str, required_fields: list[str]) -> str:
    """
    Validates that the generated Pydantic model code contains all the required fields.

    Args:
        model_code: The Pydantic model code to validate.
        required_fields: A list of field names that must be present in the model.

    Returns:
        A success message or a message detailing the missing fields.
    """
    try:
        # Parse the code into an Abstract Syntax Tree
        tree = ast.parse(model_code)
        
        # Find the class definition in the code
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # We found the class, now let's find the fields defined in it
                defined_fields = set()
                for body_item in node.body:
                    # Fields are defined as Annotated Assignments (e.g., name: str)
                    if isinstance(body_item, ast.AnnAssign):
                        defined_fields.add(body_item.target.id)
                
                # Check if all required fields are present
                missing_fields = set(required_fields) - defined_fields
                if not missing_fields:
                    return f"✅ Validation successful. All {len(required_fields)} required fields are present."
                else:
                    return f"❌ Validation failed. Missing fields: {', '.join(missing_fields)}"

        return "❌ Validation failed. No class definition found in the code."

    except Exception as e:
        return f"❌ Error during structural validation: {e}"
    
@function_tool
def return_final_code_schema(model_name: str, model_code: str):
    """
    Generates the final code schema for the Pydantic model and extraction prompt.

    Args:
        model_name: The name of the Pydantic model.
        model_code: The complete Pydantic model code.
    Returns:
        A dictionary containing the model name, model code, and extraction prompt.
    """

    return {
        "model_name": model_name,
        "model_code": model_code
    }

### we cannot force the agent to use a custom_schema and also do tool calling
### so we need to resort to another approach: https://github.com/openai/openai-agents-python/issues/1778
pydantic_code_agent = Agent(
    name = "Pydantic Code Generation Agent",
    instructions = PYDANTIC_AGENT_PROMPT,
    # output_type = PydanticModelCode,
    hooks = CustomAgentHooks(display_name="Pydantic Code Agent"),
    model = LitellmModelSelector.get_model(use_custom=True),
    tool_use_behavior = StopAtTools(stop_at_tool_names=["return_final_code_schema"]),
    # tools = [verify_pydantic_model_code, validate_pydantic_model_structure, return_final_code_schema],
    tools = [return_final_code_schema],
)

extraction_prompt_agent = Agent(
    name = "Extraction Prompt Generation Agent",
    instructions = "You are an expert in crafting detailed extraction prompts for LLMs based on provided field recommendations. "
                   "Your task is to generate a comprehensive extraction prompt that guides the LLM to accurately identify and extract each field. "
                   "Ensure the prompt includes clear instructions, examples, and any necessary context to improve extraction accuracy.",
    output_type = ExtractionPrompt,
    hooks = CustomAgentHooks(display_name="Extraction Prompt Agent"),
    model = LitellmModelSelector.get_model(use_custom=True)
)

field_mapping_agent = Agent(
    name = "Field Mapping Generation Agent",
    instructions = "You are an expert in generating field mappings for data extraction tasks. "
                   "Your task is to create a mapping of field names to their descriptions based on the provided field recommendations. "
                   "Ensure each mapping is clear and concise to facilitate accurate data extraction.",
    output_type = List[FieldMapping],
    hooks = CustomAgentHooks(display_name="Field Mapping Agent"),
    model = LitellmModelSelector.get_model(use_custom=True)
)


### Data Extraction Agent ###
def _synchronous_convert_to_markdown(file_path: str) -> str | None:
    """The actual blocking conversion logic."""
    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            all_text = "".join(page.get_text("text") + "\n\n" for page in doc)
            doc.close()
            return all_text
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"❌ Error converting {os.path.basename(file_path)}: {e}")
        return None

async def convert_to_markdown_async(file_path: str) -> str | None:
    """Asynchronously convert a file to markdown format."""
    return await asyncio.to_thread(_synchronous_convert_to_markdown, file_path)

async def extract_from_single_document(
    content: str, 
    filename: str, 
    extraction_prompt: str, 
    DynamicExtractionModel: type[BaseModel]
) -> List[dict]:
    """
    Creates and runs a dedicated agent for a single document.
    """
    try:
        # 1. Define the extraction agent dynamically inside the worker
        #    This allows us to set the dynamic Pydantic model as the output type.
        ExtractionAgent = Agent(
            name=f"Data Extractor for {filename}",
            instructions=extraction_prompt,
            output_type=Iterable[DynamicExtractionModel], # We expect a list of results!
            hooks=CustomAgentHooks(display_name=f"Extractor ({filename})"),
            model=LitellmModelSelector.get_model(use_custom=True),
        )

        # 2. Run the agent
        results = await Runner.run(ExtractionAgent, input=content)

        # 3. Process the results and add the source document
        processed_results = []
        for instance in results.final_output:
            data_dict = instance.model_dump()
            data_dict["_source_document"] = filename
            processed_results.append(data_dict)
        
        print(f"✅ Successfully extracted {len(processed_results)} items from {filename}")
        return processed_results

    except Exception as e:
        print(f"❌ Failed to process {filename}. Error: {e}")
        return [] # Return an empty list on failure

async def process_file_pipeline(file_path: str, DynamicExtractionModel: type[BaseModel], extraction_prompt: str) -> List[dict]:
    """
    A single pipeline worker that converts a file and then starts its extraction.
    """
    filename = os.path.basename(file_path)
    print(f"🔄 Starting processing for {filename}...")

    # Step 1: Convert to markdown asynchronously
    # check if the file is already converted to markdown to avoid redundant processing
    converted_path = os.path.join("converted_markdown", f"{os.path.splitext(filename)[0]}.md")
    if os.path.exists(converted_path):
        print(f"📄 Found existing markdown for {filename}. Loading from {converted_path}...")
        with open(converted_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        print(f"📄 No existing markdown found for {filename}. Converting...")
        content = await convert_to_markdown_async(file_path)
    
    # Save the converted content to re-use for debugging or future reference
    if content is not None:
        converted_path = os.path.join("converted_markdown", f"{os.path.splitext(filename)[0]}.md")
        os.makedirs("converted_markdown", exist_ok=True)
        with open(converted_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Converted {filename} to markdown and saved to {converted_path}")
    else:
        print(f"❌ Conversion failed for {filename}. No markdown file created.")
        return [] # Return an empty list if conversion fails


    # Step 2: Run the extraction agent
    return await extract_from_single_document(
        content=content,
        filename=filename,
        extraction_prompt=extraction_prompt,
        DynamicExtractionModel=DynamicExtractionModel
    )


async def main():
    user_input = "Extract temperature and pressure data from scientific articles."
    result = await Runner.run(test_agent, user_input)
    print("Schema Recommendation:")
    print(result.model_dump_json(indent=2))
        
        
if __name__ == "__main__":
    # test agent
    test_agent = Agent(
        name = "Test Schema Agent",
        instructions = "You are a data extraction expert. Given an extraction intention, recommend a schema of fields to extract.",
        output_type = SchemaRecommendation,
        model = CustomLitellmModel(),
    )

    asyncio.run(main())