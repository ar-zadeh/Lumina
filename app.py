"""
Gradio UI for Agentic Document Extraction System
Beautiful, intuitive interface with step-by-step workflow
"""
import ast
import gradio as gr
import json
import pandas as pd
from typing import *
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import pydantic

import asyncio
from agents import Runner
from data_agents.agents_collection import *
from data_agents.rag_agent import RAGSystem

from dotenv import load_dotenv

# This line loads the variables from your .env file into the environment
load_dotenv()

# --- Helper function to format your structured data into text chunks ---
def create_text_chunks_from_data(extracted_data: List[Dict[str, Any]]) -> List[str]:
    """Converts a list of extracted dictionaries into formatted string chunks for RAG."""
    chunks = []
    for i, record in enumerate(extracted_data):
        # Create a clean, readable string from each record
        chunk_text = f"Record ID: {i}\n"
        chunk_text += "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in record.items())
        chunks.append(chunk_text)
    return chunks

### Handle timeouts gracefully
from random import random
import functools

class MaxRetriesExceededError(Exception):
    """Exception raised when a function fails after all retry attempts."""
    pass

def async_retry(max_retries: int = 3, timeout_seconds: int = 30, initial_delay: int = 2):
    """
    A decorator to add timeout and exponential backoff retry logic
    to an asynchronous function.
    """
    def decorator(func):
        @functools.wraps(func) # Preserves the original function's metadata
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    # Run the wrapped function with a timeout
                    return await asyncio.wait_for(
                        func(*args, **kwargs), 
                        timeout=timeout_seconds
                    )

                except asyncio.TimeoutError:
                    print(f"Attempt {attempt + 1}/{max_retries} for '{func.__name__}' timed out.")
                    
                    # If this was the last attempt, raise the custom exception
                    if attempt == max_retries - 1:
                        raise MaxRetriesExceededError(
                            f"Function '{func.__name__}' failed after {max_retries} attempts."
                        )

                    # Calculate and wait for the backoff delay
                    delay = (initial_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"Waiting {delay:.2f} seconds before retrying...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

@async_retry(max_retries=3, timeout_seconds=60)
async def run_agent_gracefully(agent, input_text):
    """
    A simple wrapper function to run the agent.
    The @async_retry decorator automatically handles timeouts and retries for this call.
    """
    print("Running the agent...")
    return await Runner.run(agent, input=input_text)

# Import your agents (assuming they're in the same directory)
# from extraction_framework import ExtractionOrchestrator
# from rag_agent import RAGAgent

# ============================================================================
# UI State Management
# ============================================================================

class AppState:
    """Manages application state across the UI"""
    def __init__(self):
        self.orchestrator = None  # ExtractionOrchestrator()
        self.rag_agent = None  # RAGAgent()
        self.current_recommendations = None
        self.extraction_goal = None
        self.current_schema = None
        self.extracted_data = None
        self.indexed = False
        self.uploaded_files = []
        self.rag_system = None

# Global state
app_state = AppState()

# ============================================================================
# Theme and Styling
# ============================================================================

custom_css = """
.step-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
}

.success-box {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 15px;
    border-radius: 8px;
    color: white;
    margin: 10px 0;
}

.info-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 15px;
    border-radius: 8px;
    color: white;
    margin: 10px 0;
}

.warning-box {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 15px;
    border-radius: 8px;
    color: white;
    margin: 10px 0;
}

.field-card {
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background: #f8f9fa;
}

.stat-card {
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.query-result {
    background: white;
    border-left: 4px solid #667eea;
    padding: 20px;
    margin: 15px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Custom gradio component styling */
.gradio-container {
    font-family: 'Inter', sans-serif;
}

.tab-nav button {
    font-size: 16px !important;
    padding: 12px 24px !important;
}
"""

# ============================================================================
# Step 1: File Upload & Intention
# ============================================================================

def handle_file_upload(files, intention_text):
    """Handle document uploads and extraction intention"""
    if not files or len(files) == 0:
        return (
            "⚠️ Please upload at least one document.",
            gr.update(interactive=False),
            None
        )
    
    if not intention_text or intention_text.strip() == "":
        return (
            "⚠️ Please describe what you want to extract.",
            gr.update(interactive=False),
            None
        )
    
    # The file objects returned by gr.Files are temporary file wrappers.
    # We should store their paths for later use.
    file_paths = [f.name for f in files]
    app_state.uploaded_files = file_paths
    app_state.extraction_goal = intention_text
    
    # Create file summary using os module
    file_list = "\n".join([f"  • {os.path.basename(f)} ({os.path.getsize(f) / 1024:.1f} KB)" for f in file_paths])
    
    summary = f"""
✅ **Upload Successful!**

📁 **Files Uploaded:** {len(files)}
{file_list}

🎯 **Extraction Goal:**
{intention_text}

**Next Step:** Click "Generate Schema Recommendations" at the top of next section to proceed.
    """
    
    return (
        summary,
        gr.update(interactive=True),  # Enable next button
        intention_text  # Pass intention to next step
    )
    

# ============================================================================
# Step 2: Schema Recommendations
# ============================================================================

async def generate_recommendations(intention_text):
    """Generate field recommendations using Agent 1"""
    if not intention_text:
        return "❌ No extraction intention provided.", None, gr.update(interactive=False)
    
    try:
        # Call Agent 1
        ## use real agent
        recommendations = await Runner.run(schema_agent, intention_text)
        app_state.current_recommendations = recommendations
        print(recommendations)
        # parse recommendations to JSON
        recommendations = json.loads(recommendations.final_output.model_dump_json())
  
        app_state.current_recommendations = recommendations
        
        # Format recommendations for display
        fields_md = "## 📋 Recommended Fields\n\n"
        
        for i, field in enumerate(recommendations['recommended_fields'], 1):
            unit_text = f" ({field['unit']})" if field['unit'] else ""
            validation_text = f"\n  - **Validation:** {field['validation_rules']}" if field['validation_rules'] else ""
            
            fields_md += f"""
### {i}. `{field['field_name']}` - *{field['data_type']}*{unit_text}

**Description:** {field['description']}

**Example:** `{field['example']}`{validation_text}

---
"""
        
        fields_md += f"\n\n**💡 Additional Notes:**\n{recommendations['additional_notes']}"
        
        # Create editable dataframe
        df = pd.DataFrame([
            {
                "Field Name": f["field_name"],
                "Type": f["data_type"],
                "Unit": f["unit"] or "N/A",
                "Description": f["description"],
                "Keep": True  # Checkbox column
            }
            for f in recommendations['recommended_fields']
        ])
        
        return (
            fields_md,
            df,
            gr.update(interactive=True)  # Enable next button
        )
        
    except Exception as e:
        return f"❌ Error generating recommendations: {str(e)}", None, gr.update(interactive=False)

def modify_schema(dataframe):
    """Allow user to modify recommended schema"""
    if dataframe is None or len(dataframe) == 0:
        return "No schema to modify."
    
    # Filter for kept fields
    kept_fields = dataframe[dataframe['Keep'] == True]
    
    summary = f"""
✅ **Schema Modified**

Selected {len(kept_fields)} out of {len(dataframe)} fields.

**Next Step:** Click "Generate Pydantic Model" to continue.
    """
    
    return summary

# ============================================================================
# Step 3: Generate Pydantic Model
# ============================================================================
async def generate_pydantic_model(schema_df: pd.DataFrame):
    """Generate Pydantic model using Agent 2"""
    if not app_state.current_recommendations:
        return "❌ No recommendations available.", "", gr.update(interactive=False)
    
    if schema_df is None or schema_df.empty:
        return "❌ Schema table is empty. Please review recommendations first.", "", "", gr.update(interactive=False)

    print("DF is ", schema_df)
    
    # Get the set of field names the user decided to keep from the UI table.
    kept_field_names = set(schema_df[schema_df['Keep'] == True]['Field Name'])
    print("Kept field names: ", kept_field_names)
    # Filter the original, detailed recommendations list based on the user's choices.
    fields_to_include_list = [
        field for field in app_state.current_recommendations['recommended_fields']
        if field['field_name'] in kept_field_names
    ]
    
    
    # Check if any fields are left after filtering
    if not fields_to_include_list:
         return "❌ No fields were selected to keep. Please check at least one box in the schema table.", "", "", gr.update(interactive=False)

    fields_to_include = json.dumps(fields_to_include_list, indent=2)
    
    # print("\nCurrent Recommendations Fields:\n ", app_state.current_recommendations['recommended_fields'])
    ### prepare the prompt for agent 2
    fields_to_include = json.dumps([field for field in app_state.current_recommendations['recommended_fields']], indent=2)
    
    PYDANTIC_MODEL_TEMPLATE = f"""Generate a Pydantic model for this extraction task:

    Extraction Goal: {app_state.extraction_goal}

    Fields to include:
    {fields_to_include}

    Additional Notes: {app_state.current_recommendations['additional_notes']}

    Provide:
    0. A brief description of the extraction task aka Pydantic model name
    1. Complete Pydantic model code (with imports)
    2. A detailed extraction prompt that explains how to identify and extract each field
    3. Field mappings for reference"""


    try:
        # Call Agent 2 and it sub agents
        try:
            prompt_res = await run_agent_gracefully(extraction_prompt_agent, f"Task: {app_state.extraction_goal}\nFields: {fields_to_include}\nNotes: {app_state.current_recommendations['additional_notes']}")
        except MaxRetriesExceededError as e:
            print(f"Error: {str(e)}")
            return f"❌ Failed to generate extraction prompt after multiple attempts. Please try again later.", "", gr.update(interactive=False)
        prompt_result = prompt_res.final_output
        print("here's prompt result: ", prompt_result)
        
        # pydantic code generation agent
        try:
            code_res = await run_agent_gracefully(pydantic_code_agent, f"Task: {app_state.extraction_goal}\nFields: {fields_to_include}\nNotes: {app_state.current_recommendations['additional_notes']}\nExtraction Prompt: {prompt_result.prompt_text}")
        except MaxRetriesExceededError as e:
            print(f"Error: {str(e)}")
            return f"❌ Failed to generate Pydantic model after multiple attempts. Please try again later.", "", gr.update(interactive=False)
        code_result_str = code_res.final_output
        print("here's code result: ", code_result_str)
        code_result_dict = ast.literal_eval(code_result_str) # convert string representation of dict to actual dict
        code_result = PydanticModelCode(**code_result_dict) # convert dict to Pydantic model instance
    

        # parse recommendations to JSON
        app_state.current_schema = {
            "model_name": code_result.model_name,
            "model_code": code_result.model_code,
            "extraction_prompt": prompt_result.prompt_text,
        }
        
        
        summary = f"""
✅ **Pydantic Model Generated**

**Model Name:** MaterialSynthesis

**Fields:** {len(app_state.current_recommendations['recommended_fields'])}

The model includes:
- Type validation
- Field descriptions
- Custom validators
- Optional field handling

**Next Step:** Click "Extract Data" to process your documents.
        """
        
        return (
            summary,
            code_result.model_code,
            prompt_result.prompt_text,
            gr.update(interactive=True)
        )
        
    except Exception as e:
        return f"❌ Error generating model: {str(e)}", "", "", gr.update(interactive=False)

# ============================================================================
# Step 4: Extract Data
# ============================================================================

async def extract_data(progress=gr.Progress()):
    """
    Extract data in parallel by creating a task for each document.
    """
    if not app_state.uploaded_files:
        return "❌ No files uploaded.", None, None, gr.update(interactive=False)
    if not app_state.current_schema or 'model_code' not in app_state.current_schema:
        return "❌ Extraction schema not generated.", None, None, gr.update(interactive=False)

    progress(0, desc="Initializing extraction...")

    # --- THIS IS THE SECTION TO CHANGE ---
    
    # 1. Create a safe, controlled global environment for exec()
    #    This is our "toolbox" with everything the Pydantic code might need.
    custom_globals = {
        "BaseModel": BaseModel,
        "Field": Field,
        "Literal": Literal,
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "Any": Any,
    }

    # 2. Dynamically create the Pydantic model within this controlled environment
    try:
        namespace = {} # This will store the new class definition
        exec(
            app_state.current_schema['model_code'], 
            custom_globals, # Pass the toolbox as the global scope
            namespace       # The local scope where the new class will be defined
        )
        ExtractionModel = namespace[app_state.current_schema['model_name']]
    except Exception as e:
        # Now, if it fails, the error will be more specific and understandable
        return f"❌ Error creating Pydantic model: {e}", None, None, gr.update(interactive=False)

    # Dynamically create the Pydantic model from the generated code
    try:
        namespace = {}
        exec(app_state.current_schema['model_code'], globals(), namespace)
        ExtractionModel = namespace[app_state.current_schema['model_name']]
    except Exception as e:
        return f"❌ Error creating Pydantic model: {e}", None, None, gr.update(interactive=False)

    # Create a list of async tasks, one for each document
    tasks = []
    file_count = len(app_state.uploaded_files)
    
    for i, file_path in enumerate(app_state.uploaded_files):
        progress(i / file_count, desc=f"Scheduling extraction for {os.path.basename(file_path)}...")
        
        task = asyncio.create_task(
            process_file_pipeline(
                file_path, ExtractionModel, app_state.current_schema['extraction_prompt']    
            )
            
        )
        tasks.append(task)
    
    list_of_results = await asyncio.gather(*tasks)
    
    # Flatten the list of lists into a single list of all extractions
    all_extractions = [item for sublist in list_of_results for item in sublist]
    app_state.extracted_data = all_extractions
    
    if not app_state.extracted_data:
        return "✅ Extraction complete, but no data was found.", None, None, gr.update(interactive=True)

    # --- Create DataFrame and Visualization (same as before) ---
    progress(0.9, desc="Processing results...")
    df = pd.DataFrame(app_state.extracted_data)
    fig = None # placeholder for visualization
    
    # # Create visualization
    # fig = px.scatter(
    #     df,
    #     x="synthesis_temperature",
    #     y="particle_size",
    #     color="synthesis_method",
    #     size="reaction_time",
    #     hover_data=["material_name"],
    #     title="Synthesis Conditions vs Particle Size",
    #     labels={
    #         "synthesis_temperature": "Temperature (°C)",
    #         "particle_size": "Particle Size (nm)"
    #     }
    # )
    # fig.update_layout(
    #     template="plotly_white",
    #     height=500
    # )
    
    progress(1.0, desc="Complete!")
    
    summary = f"""
✅ **Extraction Complete!**

**Records Extracted:** {len(app_state.extracted_data)}
**Documents Processed:** {len(app_state.uploaded_files)}

**Next Steps:**
1. Review the extracted data in the table below
2. Download as JSON or CSV if needed
3. Proceed to "Query Data" tab to ask questions
    """
    
    return (
        summary,
        df,
        fig,
        gr.update(interactive=True)
    )
    

def export_data(format_choice):
    """Export extracted data by writing to a temporary file
    and returning the file path to gradio to handle the download."""
    if not app_state.extracted_data:
        return None
    
    # Define the file extension based on user choice
    suffix = ".json" if format_choice == "JSON" else ".csv"
    
    # Create a named temporary file that won't be deleted immediately
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=suffix, encoding='utf-8') as temp_f:
        if format_choice == "JSON":
            json.dump(app_state.extracted_data, temp_f, indent=2)
        else:  # CSV
            df = pd.DataFrame(app_state.extracted_data)
            df.to_csv(temp_f.name, index=False)
            
        # Return the path of the temporary file you just created
        return temp_f.name
# ============================================================================
# Step 5: Query Data (RAG)
# ============================================================================
async def index_data_for_rag(csv_file, progress=gr.Progress()):
    """
    Initializes RAG and indexes data from either an uploaded CSV or the extraction pipeline.
    """
    data_to_index = None
    source_description = ""

    # --- NEW: Logic to determine the data source ---
    if csv_file is not None:
        try:
            # Source 1: User uploaded a CSV file
            progress(0.1, desc=f"Reading {os.path.basename(csv_file.name)}...")
            df = pd.read_csv(csv_file.name)
            data_to_index = df.to_dict(orient='records') # Convert DataFrame to list of dicts
            source_description = f"uploaded file '{os.path.basename(csv_file.name)}'"
        except Exception as e:
            return f"❌ Error reading CSV file: {e}", gr.update(interactive=False)
            
    elif app_state.extracted_data:
        # Source 2: Fallback to data from the extraction pipeline
        data_to_index = app_state.extracted_data
        source_description = "the extraction pipeline"
    
    else:
        # No data source available
        return "❌ No data to index. Either upload a CSV or run the extraction pipeline first.", gr.update(interactive=False)

    try:
        progress(0.2, desc="Initializing RAG System...")
        if app_state.rag_system is None:
            app_state.rag_system = RAGSystem(
                embed_api_key=os.getenv("NVIDIA_EMBED_API_KEY"),
                rerank_api_key=os.getenv("NVIDIA_RERANK_API_KEY"),
                gemini_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
            )

        progress(0.4, desc=f"Preparing {len(data_to_index)} records for indexing...")
        chunks = create_text_chunks_from_data(data_to_index)
        
        if not chunks:
            return "⚠️ No data could be processed for indexing.", gr.update(interactive=False)

        progress(0.7, desc=f"Indexing {len(chunks)} records...")
        await asyncio.to_thread(app_state.rag_system.index_documents, chunks, batch_size=10)
        
        app_state.indexed = True
        progress(1.0, desc="Indexed!")
        
        return (
            f"✅ **Data from {source_description} Indexed!**\n\nYou can now ask questions.",
            gr.update(interactive=True)
        )
        
    except Exception as e:
        return f"❌ Error indexing data: {str(e)}", gr.update(interactive=False)
    
async def query_data(query_text: str, num_results: int):
    """Queries the indexed data using the full RAG pipeline."""
    if not app_state.indexed:
        return "❌ Please index data first.", None, None, "Status: Not Indexed"
    if not query_text or not query_text.strip():
        return "⚠️ Please enter a question.", None, None, "Status: Waiting for query"

    try:
        # Run the potentially long-running pipeline in a thread
        pipeline_result = await asyncio.to_thread(
            app_state.rag_system.run_pipeline,
            query=query_text,
            retrieve_top_k=20,       # Retrieve more to give the reranker options
            rerank_top_k=num_results # Rerank and keep the number from the slider
        )

        # Format the final answer for display
        answer = pipeline_result.get("answer", "No answer could be generated.")
        answer_md = f"## 💬 Answer\n\n{answer}"

        # Format reranked results into a DataFrame
        reranked_results = pipeline_result.get("reranked_results", [])
        records_df = pd.DataFrame(reranked_results)
        # Select and rename columns for clarity in the UI
        if not records_df.empty:
            records_df = records_df[['text', 'rerank_score']].rename(
                columns={'text': 'Relevant Context', 'rerank_score': 'Relevance Score'}
            )

        # Create the confidence gauge from the top rerank score (as a proxy)
        top_score = reranked_results[0]['rerank_score'] if reranked_results else 0
        # value of top_score can vary between -inf to inf so we need to normalize it to a 0-1 range for the gauge. Assuming scores are between -10 and 10 for simplicity:
        min_score = min(reranked_results, key=lambda x: x['rerank_score'])['rerank_score'] if reranked_results else 0
        max_score = max(reranked_results, key=lambda x: x['rerank_score'])['rerank_score'] if reranked_results else 0
        if max_score - min_score > 0:
            normalized_score = (top_score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5
        top_score = normalized_score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top_score * 100,
            title={'text': "Top Document Relevance"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        fig.update_layout(height=300)

        return (
            answer_md,
            records_df,
            fig,
            f"Query processed successfully using {len(reranked_results)} relevant records."
        )

    except Exception as e:
        return f"❌ Error processing query: {str(e)}", None, None, "Status: Error"
# ============================================================================
# Main Gradio Interface
# ============================================================================

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="AI Document Extraction") as demo:
        
        gr.Markdown("""
        # 🤖 AI-Powered Document Extraction System
        ### Extract structured data from unstructured documents using multi-agent AI
        """)
        
        with gr.Tabs() as tabs:
            
            # ================================================================
            # TAB 1: EXTRACTION PIPELINE
            # ================================================================
            with gr.Tab("📄 Extract Data", id=0):
                
                gr.Markdown("""
                ## Step-by-Step Extraction Process
                Follow these steps to extract structured data from your documents.
                """)
                
                # Step 1: Upload
                with gr.Accordion("**Step 1:** Upload Documents & Define Goal", open=True):
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_upload = gr.Files(
                                label="📁 Upload Documents",
                                file_types=[".pdf", ".csv", ".txt", ".md"],
                                file_count="multiple"
                            )
                            intention_input = gr.Textbox(
                                label="🎯 What do you want to extract?",
                                placeholder="E.g., Extract material synthesis information including temperatures, methods, and results...",
                                lines=4
                            )
                            upload_btn = gr.Button("📤 Upload & Proceed", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            upload_status = gr.Markdown("⏳ Waiting for upload...")
                    
                    intention_state = gr.State()
                
                # Step 2: Recommendations
                with gr.Accordion("**Step 2:** Review Schema Recommendations", open=False) as accordion_2:
                    recommend_btn = gr.Button("🔮 Generate Schema Recommendations", interactive=False, size="lg")
                    recommendations_display = gr.Markdown()
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ✏️ Customize Schema (Optional)")
                            schema_table = gr.Dataframe(
                                label="Edit fields - uncheck 'Keep' to remove fields",
                                interactive=True,
                                wrap=True
                            )
                            modify_btn = gr.Button("💾 Apply Changes")
                            modification_status = gr.Markdown()
                
                # Step 3: Generate Model
                with gr.Accordion("**Step 3:** Generate Pydantic Model", open=False) as accordion_3:
                    generate_model_btn = gr.Button("⚙️ Generate Pydantic Model", interactive=False, size="lg")
                    model_status = gr.Markdown()
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Generated Model Code")
                            model_code_display = gr.Code(language="python", label="Pydantic Model")
                        with gr.Column():
                            gr.Markdown("#### Extraction Prompt")
                            prompt_display = gr.Textbox(label="Prompt for Extraction Agent", lines=10)
                
                # Step 4: Extract
                with gr.Accordion("**Step 4:** Extract Data from Documents", open=False) as accordion_4:
                    extract_btn = gr.Button("🚀 Extract Data", interactive=False, size="lg", variant="primary")
                    extraction_status = gr.Markdown()
                    
                    with gr.Row():
                        with gr.Column():
                            extracted_table = gr.Dataframe(
                                label="📊 Extracted Data",
                                wrap=True,
                                interactive=False
                            )
                        
                        with gr.Column():
                            extraction_viz = gr.Plot(label="📈 Data Visualization")
                    
                    with gr.Row():
                        export_format = gr.Radio(
                            choices=["JSON", "CSV"],
                            value="JSON",
                            label="Export Format"
                        )
                        export_btn = gr.Button("💾 Prepare the file")
                        # with gr.Column():
                    export_output = gr.File(label="Available File for Download")
            
            # ================================================================
            # TAB 2: QUERY DATA (RAG)
            # ================================================================
            with gr.Tab("🔍 Query Data", id=1):
                
                gr.Markdown("""
                ## Ask Questions About Your Extracted Data
                You can either use the data from the 'Extract Data' tab or upload a new CSV file below to start querying.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # --- NEW: Add a dedicated CSV upload component ---
                        csv_upload = gr.File(
                            label="📄 Upload CSV to Query",
                            file_types=[".csv"]
                        )
                        index_btn = gr.Button("🗄️ Index Data for Queries", size="lg", variant="primary")
                        index_status = gr.Markdown("⏳ Data not indexed yet. Click above to index.")
            
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="💬 Ask a Question",
                            placeholder="E.g., What materials were synthesized at temperatures above 400°C?",
                            lines=3
                        )
                        num_results_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of records to retrieve"
                        )
                        query_btn = gr.Button("🔍 Search", size="lg", interactive=False)
                    
                    with gr.Column(scale=1):
                        confidence_gauge = gr.Plot(label="Confidence Score")
                
                query_result = gr.Markdown(label="Answer")
                relevant_records = gr.Dataframe(label="📄 Relevant Records", wrap=True)
                query_status = gr.Markdown()
                
                gr.Markdown("### 💡 Example Questions")
                example_queries = gr.Examples(
                    examples=[
                        ["What is the average synthesis temperature?"],
                        ["Which synthesis method produces the smallest particles?"],
                        ["List all materials synthesized using hydrothermal methods."],
                        ["What's the relationship between temperature and particle size?"],
                        ["Which materials are suitable for photocatalysis?"]
                    ],
                    inputs=query_input
                )
            
            # ================================================================
            # TAB 3: ANALYTICS DASHBOARD
            # ================================================================
            # with gr.Tab("📊 Analytics", id=2):
                
            #     gr.Markdown("## 📈 Data Analytics Dashboard")
                
            #     if app_state.extracted_data:
            #         df = pd.DataFrame(app_state.extracted_data)
                    
            #         with gr.Row():
            #             with gr.Column():
            #                 gr.Markdown(f"""
            #                 <div class="stat-card">
            #                     <h2>{len(df)}</h2>
            #                     <p>Total Records</p>
            #                 </div>
            #                 """)
            #             with gr.Column():
            #                 gr.Markdown(f"""
            #                 <div class="stat-card">
            #                     <h2>{len(df.columns)}</h2>
            #                     <p>Fields Extracted</p>
            #                 </div>
            #                 """)
            #             with gr.Column():
            #                 gr.Markdown(f"""
            #                 <div class="stat-card">
            #                     <h2>{len(app_state.uploaded_files)}</h2>
            #                     <p>Documents Processed</p>
            #                 </div>
            #                 """)
                    
            #         # More analytics here...
            #         gr.Markdown("*Complete extraction to see detailed analytics*")
            #     else:
            #         gr.Markdown("⏳ *No data available yet. Complete the extraction process first.*")
        
        # ================================================================
        # EVENT HANDLERS
        # ================================================================
        
        # Step 1: Upload
        upload_btn.click(
            fn=handle_file_upload,
            inputs=[file_upload, intention_input],
            outputs=[upload_status, recommend_btn, intention_state]
        )
        
        # Step 2: Recommendations
        recommend_btn.click(
            fn=generate_recommendations,
            inputs=[intention_state],
            outputs=[recommendations_display, schema_table, generate_model_btn]
        )
        
        modify_btn.click(
            fn=modify_schema,
            inputs=[schema_table],
            outputs=[modification_status]
        )
        
        # Step 3: Generate Model
        generate_model_btn.click(
            fn=generate_pydantic_model,
            inputs=[schema_table],
            outputs=[model_status, model_code_display, prompt_display, extract_btn]
        )
        
        # Step 4: Extract
        extract_btn.click(
            fn=extract_data,
            outputs=[extraction_status, extracted_table, extraction_viz, index_btn]
        )
        
        # Export
        export_btn.click(
            fn=export_data,
            inputs=[export_format],
            outputs=[export_output]
        )
        
        # RAG: Index
        index_btn.click(
            fn=index_data_for_rag,
            inputs=[csv_upload],  # <-- Pass the new file upload component as input
            outputs=[index_status, query_btn]
        )
        
        # RAG: Query
        query_btn.click(
            fn=query_data,
            inputs=[query_input, num_results_slider],
            outputs=[query_result, relevant_records, confidence_gauge, query_status]
        )
    
    return demo

# ============================================================================
# Launch Application
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
)   