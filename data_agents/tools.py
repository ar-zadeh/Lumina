from agents import function_tool
from typing import List, Optional
from pydantic import BaseModel, Field


## Schema generation agent
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
    
