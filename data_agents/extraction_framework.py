"""
Agentic Document Extraction Framework using OpenAI Agents SDK
Extracts structured data from unstructured documents (PDF, CSV, etc.)
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, create_model
from openai import OpenAI
import PyPDF2
import pandas as pd

client = OpenAI()

# ============================================================================
# Agent 1: Schema Recommendation Agent
# ============================================================================

class FieldRecommendation(BaseModel):
    """Recommended field for extraction"""
    field_name: str = Field(description="Name of the field to extract")
    description: str = Field(description="What this field represents")
    data_type: str = Field(description="Expected data type (string, number, boolean, list, object)")
    unit: Optional[str] = Field(default=None, description="Unit of measurement if applicable")
    example: str = Field(description="Example of what this field might contain")
    validation_rules: Optional[str] = Field(default=None, description="Any validation constraints")

class SchemaRecommendation(BaseModel):
    """Complete schema recommendation"""
    extraction_goal: str = Field(description="User's extraction intention")
    recommended_fields: List[FieldRecommendation]
    additional_notes: str = Field(description="Additional guidance for extraction")

class SchemaRecommendationAgent:
    """Agent 1: Recommends extraction schema based on user intention"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = """You are a data extraction expert. Your job is to analyze the user's extraction intention 
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

    def recommend_schema(self, user_intention: str) -> SchemaRecommendation:
        """Generate schema recommendations based on user's extraction goal"""
        
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""User wants to extract: {user_intention}
                
Please recommend a comprehensive extraction schema with all relevant fields."""}
            ],
            response_format=SchemaRecommendation
        )
        
        return response.choices[0].message.parsed

# ============================================================================
# Agent 2: Pydantic Model Generation Agent
# ============================================================================

class ExtractionSchema(BaseModel):
    """Generated extraction schema with Pydantic model"""
    model_name: str = Field(description="Name of the Pydantic model")
    model_code: str = Field(description="Complete Pydantic model code")
    extraction_prompt: str = Field(description="Detailed prompt for extraction agent")
    field_mappings: Dict[str, str] = Field(description="Mapping of field names to descriptions")

class ModelGenerationAgent:
    """Agent 2: Generates Pydantic models and extraction prompts"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = """You are a Python expert specializing in Pydantic models. Your job is to:

1. Generate a complete, valid Pydantic model class from field recommendations
2. Include proper type hints, Field descriptions, and validators
3. Add validation rules (regex patterns, value ranges, etc.) where appropriate
4. Create a detailed extraction prompt that will guide an LLM to extract data accurately

The Pydantic model should be production-ready with:
- Proper imports
- Type annotations
- Field descriptions
- Validators for data quality
- Optional fields where appropriate
- Custom validators using @field_validator"""

    def generate_model(self, schema_recommendation: SchemaRecommendation) -> ExtractionSchema:
        """Generate Pydantic model and extraction prompt from recommendations"""
        
        # Create prompt for model generation
        fields_json = json.dumps([field.model_dump() for field in schema_recommendation.recommended_fields], indent=2)
        
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Generate a Pydantic model for this extraction task:

Extraction Goal: {schema_recommendation.extraction_goal}

Fields to include:
{fields_json}

Additional Notes: {schema_recommendation.additional_notes}

Provide:
1. Complete Pydantic model code (with imports)
2. A detailed extraction prompt that explains how to identify and extract each field
3. Field mappings for reference"""}
            ],
            response_format=ExtractionSchema
        )
        
        return response.choices[0].message.parsed

# ============================================================================
# Agent 3: Extraction Agent
# ============================================================================

class ExtractionAgent:
    """Agent 3: Extracts structured data from documents"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        
    def extract_from_document(
        self, 
        document_content: str,
        pydantic_model: type[BaseModel],
        extraction_prompt: str,
        document_name: str = "document"
    ) -> BaseModel:
        """Extract structured data from a single document"""
        
        system_prompt = f"""You are a precise data extraction agent. Your job is to extract structured information 
from documents according to the provided schema.

{extraction_prompt}

Extract all relevant information accurately. If a field cannot be determined from the document, use null/None.
Be precise with numbers, units, and technical terms."""

        user_prompt = f"""Document: {document_name}

Content:
{document_content}

Please extract all relevant information according to the schema."""

        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=pydantic_model
        )
        
        return response.choices[0].message.parsed
    
    def extract_from_documents(
        self,
        documents: List[Dict[str, str]],
        pydantic_model: type[BaseModel],
        extraction_prompt: str
    ) -> List[BaseModel]:
        """Extract from multiple documents"""
        
        results = []
        for doc in documents:
            result = self.extract_from_document(
                document_content=doc['content'],
                pydantic_model=pydantic_model,
                extraction_prompt=extraction_prompt,
                document_name=doc.get('name', 'unknown')
            )
            results.append(result)
        
        return results

# ============================================================================
# Document Readers
# ============================================================================

class DocumentReader:
    """Utility class to read various document formats"""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def read_csv(file_path: str) -> str:
        """Read CSV and convert to string representation"""
        df = pd.read_csv(file_path)
        return df.to_string()
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        """Read plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def read_document(file_path: str) -> Dict[str, str]:
        """Read document based on extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            content = DocumentReader.read_pdf(file_path)
        elif extension == '.csv':
            content = DocumentReader.read_csv(file_path)
        elif extension in ['.txt', '.md']:
            content = DocumentReader.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return {
            'name': path.name,
            'content': content,
            'path': str(path)
        }

# ============================================================================
# Orchestrator - Ties Everything Together
# ============================================================================

class ExtractionOrchestrator:
    """Main orchestrator that coordinates all three agents"""
    
    def __init__(self):
        self.schema_agent = SchemaRecommendationAgent()
        self.model_agent = ModelGenerationAgent()
        self.extraction_agent = ExtractionAgent()
        self.reader = DocumentReader()
    
    def run_extraction_pipeline(
        self,
        user_intention: str,
        document_paths: List[str],
        review_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Complete extraction pipeline
        
        Args:
            user_intention: What the user wants to extract
            document_paths: List of document file paths
            review_recommendations: If True, returns recommendations for user review
            
        Returns:
            Dictionary containing recommendations, schema, and extracted data
        """
        
        print("=" * 80)
        print("STEP 1: Generating Schema Recommendations")
        print("=" * 80)
        
        # Step 1: Get schema recommendations
        recommendations = self.schema_agent.recommend_schema(user_intention)
        
        print(f"\n📋 Extraction Goal: {recommendations.extraction_goal}")
        print(f"\n✅ Recommended {len(recommendations.recommended_fields)} fields:")
        for field in recommendations.recommended_fields:
            print(f"  • {field.field_name} ({field.data_type})")
            print(f"    → {field.description}")
            if field.unit:
                print(f"    → Unit: {field.unit}")
            print(f"    → Example: {field.example}")
            if field.validation_rules:
                print(f"    → Validation: {field.validation_rules}")
            print()
        
        print(f"📝 Notes: {recommendations.additional_notes}\n")
        
        if review_recommendations:
            return {
                'step': 'recommendations',
                'recommendations': recommendations,
                'message': 'Review recommendations. Call continue_extraction() to proceed.'
            }
        
        # Step 2: Generate Pydantic model
        print("=" * 80)
        print("STEP 2: Generating Pydantic Model")
        print("=" * 80)
        
        extraction_schema = self.model_agent.generate_model(recommendations)
        
        print(f"\n📦 Model Name: {extraction_schema.model_name}")
        print(f"\n🔧 Generated Model Code:")
        print(extraction_schema.model_code)
        print(f"\n📄 Extraction Prompt:")
        print(extraction_schema.extraction_prompt)
        
        # Create the actual Pydantic model dynamically
        pydantic_model = self._create_dynamic_model(recommendations, extraction_schema.model_name)
        
        # Step 3: Read documents
        print("\n" + "=" * 80)
        print("STEP 3: Reading Documents")
        print("=" * 80)
        
        documents = []
        for path in document_paths:
            print(f"\n📄 Reading: {path}")
            doc = self.reader.read_document(path)
            documents.append(doc)
            print(f"  ✓ Loaded {len(doc['content'])} characters")
        
        # Step 4: Extract from all documents
        print("\n" + "=" * 80)
        print("STEP 4: Extracting Data")
        print("=" * 80)
        
        extracted_data = self.extraction_agent.extract_from_documents(
            documents=documents,
            pydantic_model=pydantic_model,
            extraction_prompt=extraction_schema.extraction_prompt
        )
        
        print(f"\n✅ Extracted data from {len(extracted_data)} documents")
        
        return {
            'recommendations': recommendations,
            'schema': extraction_schema,
            'pydantic_model': pydantic_model,
            'documents': documents,
            'extracted_data': extracted_data,
            'results_json': [item.model_dump() for item in extracted_data]
        }
    
    def _create_dynamic_model(self, recommendations: SchemaRecommendation, model_name: str) -> type[BaseModel]:
        """Create a dynamic Pydantic model from recommendations"""
        
        fields = {}
        for field in recommendations.recommended_fields:
            # Map string types to Python types
            type_map = {
                'string': str,
                'number': float,
                'integer': int,
                'boolean': bool,
                'list': List[str],
                'object': Dict[str, Any]
            }
            
            field_type = type_map.get(field.data_type.lower(), str)
            
            # Make field optional with None as default
            fields[field.field_name] = (
                Optional[field_type],
                Field(default=None, description=field.description)
            )
        
        return create_model(model_name, **fields)

# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use the framework"""
    
    # Initialize orchestrator
    orchestrator = ExtractionOrchestrator()
    
    # Define what you want to extract
    user_intention = """
    I want to extract material composition and synthesis information from materials science papers.
    Specifically, I need to capture:
    - What materials were fabricated or synthesized
    - The synthesis process and methods used
    - Temperature conditions (with units)
    - Results and properties achieved
    """
    
    # Documents to process (replace with actual paths)
    document_paths = [
        "paper1.pdf",
        "paper2.pdf",
        "experimental_data.csv"
    ]
    
    # Run the complete pipeline
    results = orchestrator.run_extraction_pipeline(
        user_intention=user_intention,
        document_paths=document_paths,
        review_recommendations=False  # Set to True to review before extraction
    )
    
    # Access results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for i, data in enumerate(results['extracted_data'], 1):
        print(f"\n📄 Document {i}: {results['documents'][i-1]['name']}")
        print(json.dumps(data.model_dump(), indent=2))
    
    # Export to JSON
    with open('extracted_data.json', 'w') as f:
        json.dump(results['results_json'], f, indent=2)
    
    print("\n✅ Results exported to extracted_data.json")

if __name__ == "__main__":
    # Example: Just get recommendations first
    orchestrator = ExtractionOrchestrator()
    
    intention = "Extract detailed temperature information from experimental chemistry papers"
    recommendations = orchestrator.schema_agent.recommend_schema(intention)
    
    print("Recommended Fields:")
    for field in recommendations.recommended_fields:
        print(f"\n{field.field_name}:")
        print(f"  Type: {field.data_type}")
        print(f"  Description: {field.description}")
        if field.unit:
            print(f"  Unit: {field.unit}")
        print(f"  Example: {field.example}")