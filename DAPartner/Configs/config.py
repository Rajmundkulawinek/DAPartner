from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

import os

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
BASE_PROMPTS = BASE_DIR / ".." / "Prompts" / "Dimensional"
GLOBAL_FILES = BASE_DIR / ".." / "Configs" 

# ---------------------------------------------------------------------------
# 2. LLM models
# ---------------------------------------------------------------------------
load_dotenv()
llm_point_critical_columns_in_source = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Collection = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Collection_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Analysis = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Analysis_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Entity_Level_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Entity_Level = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Columns_Level_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Columns_Level = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Create_Modeling_Feed_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Create_Modeling_Feed = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Scripts_Creation_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Scripts_Creation = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Entity_Level_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Entity_Level = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Columns_Level_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Columns_Level = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Create_Modeling_Feed_Detect_State = ChatOpenAI(model_name="gpt-5-mini") #gpt-5-codex
llm_Model_Fact_Create_Modeling_Feed = ChatOpenAI(model_name="gpt-5-mini") #gpt-5-codex
llm_Model_Fact_Scripts_Creation_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Scripts_Creation = ChatOpenAI(model_name="gpt-5-mini")


# ---------------------------------------------------------------------------
# 3. Files
# ---------------------------------------------------------------------------
with open(BASE_PROMPTS / "Requirements_Colection.txt", encoding="utf-8") as f:
    REQUIREMENTS_COLLECTION_PROMPT_TEXT = f.read().strip()
REQUIREMENTS_COLLECTION_PROMPT_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_COLLECTION_PROMPT_TEXT)

with open(BASE_PROMPTS / "Requirements_Colection_Detect_State.txt", encoding="utf-8") as f:
    REQUIREMENTS_COLLECTION_PROMPT_Detect_State_TEXT = f.read().strip()
REQUIREMENTS_COLLECTION_PROMPT_Detect_State_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_COLLECTION_PROMPT_Detect_State_TEXT)

with open(BASE_PROMPTS / "Requirements_Analysis.txt", encoding="utf-8") as f:
    REQUIREMENTS_ANALYSIS_PROMPT_TEXT = f.read().strip()
REQUIREMENTS_ANALYSIS_PROMPT_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_ANALYSIS_PROMPT_TEXT)
# Wywołanie: REQUIREMENTS_ANALYSIS_PROMPT = REQUIREMENTS_ANALYSIS_PROMPT_TEMPLATE.format(table_list='xxx')

with open(BASE_PROMPTS / "Requirements_Analysis_Detect_State.txt", encoding="utf-8") as f:
    REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEXT = f.read().strip()
REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEXT)

with open(BASE_PROMPTS / "Point_Critical_Columns_In_Source.txt", encoding="utf-8") as f:
    Point_Critical_Columns_In_Source_Text = f.read().strip()
Point_Critical_Columns_In_Source_Template = PromptTemplate.from_template(Point_Critical_Columns_In_Source_Text)

with open(GLOBAL_FILES / "List_available_dimensions.json", encoding="utf-8") as f:
    LIST_AVAILABLE_DIMENSIONS = f.read().strip()

with open(GLOBAL_FILES / "SAP_2_Snowflake_data_types_mapping.csv", encoding="utf-8") as f:
    SAP_2_Snowflake_data_types_mapping_file_content = f.read().strip()

# Model_Dimension_Columns_Level_Detect_State
with open(BASE_PROMPTS / "Model_Dimension_Columns_Level_Detect_State.txt", encoding="utf-8") as f:
    Model_Dimension_Columns_Level_Detect_State_Text = f.read().strip()
Model_Dimension_Columns_Level_Detect_State_Template = PromptTemplate.from_template(Model_Dimension_Columns_Level_Detect_State_Text)

# Model_Dimension_Columns_Level
with open(BASE_PROMPTS / "Model_Dimension_Columns_Level.txt", encoding="utf-8") as f:
    Model_Dimension_Columns_Level_Text = f.read().strip()
Model_Dimension_Columns_Level_Template = PromptTemplate.from_template(Model_Dimension_Columns_Level_Text)

# Model_Dimension_Create_Modeling_Feed_Detect_State
with open(BASE_PROMPTS / "Model_Dimension_Create_Modeling_Feed_Detect_State.txt", encoding="utf-8") as f:
    Model_Dimension_Create_Modeling_Feed_Detect_State_Text = f.read().strip()
Model_Dimension_Create_Modeling_Feed_Detect_State_Template = PromptTemplate.from_template(Model_Dimension_Create_Modeling_Feed_Detect_State_Text)

# Model_Dimension_Create_Modeling_Feed
with open(BASE_PROMPTS / "Model_Dimension_Create_Modeling_Feed.txt", encoding="utf-8") as f:
    Model_Dimension_Create_Modeling_Feed_Text = f.read().strip()
Model_Dimension_Create_Modeling_Feed_Template = PromptTemplate.from_template(Model_Dimension_Create_Modeling_Feed_Text)

# Model_Dimension_Entity_Level_Detect_State
with open(BASE_PROMPTS / "Model_Dimension_Entity_Level_Detect_State.txt", encoding="utf-8") as f:
    Model_Dimension_Entity_Level_Detect_State_Text = f.read().strip()
Model_Dimension_Entity_Level_Detect_State_Template = PromptTemplate.from_template(Model_Dimension_Entity_Level_Detect_State_Text)

# Model_Dimension_Entity_Level
with open(BASE_PROMPTS / "Model_Dimension_Entity_Level.txt", encoding="utf-8") as f:
    Model_Dimension_Entity_Level_Text = f.read().strip()
Model_Dimension_Entity_Level_Template = PromptTemplate.from_template(Model_Dimension_Entity_Level_Text)

# Model_Dimension_Scripts_Creation_Detect_State
with open(BASE_PROMPTS / "Model_Dimension_Scripts_Creation_Detect_State.txt", encoding="utf-8") as f:
    Model_Dimension_Scripts_Creation_Detect_State_Text = f.read().strip()
Model_Dimension_Scripts_Creation_Detect_State_Template = PromptTemplate.from_template(Model_Dimension_Scripts_Creation_Detect_State_Text)

# Model_Dimension_Scripts_Creation
with open(BASE_PROMPTS / "Model_Dimension_Scripts_Creation.txt", encoding="utf-8") as f:
    Model_Dimension_Scripts_Creation_Text = f.read().strip()
Model_Dimension_Scripts_Creation_Template = PromptTemplate.from_template(Model_Dimension_Scripts_Creation_Text)

# Model_Dimension_Scripts_Creation
with open(BASE_PROMPTS / "Model_Dimension_Scripts_Creation.txt", encoding="utf-8") as f:
    Model_Dimension_Scripts_Creation_Text = f.read().strip()
Model_Dimension_Scripts_Creation_Template = PromptTemplate.from_template(Model_Dimension_Scripts_Creation_Text)

# Model_Fact_Entity_Level_Detect_State
with open(BASE_PROMPTS / "Model_Fact_Entity_Level_Detect_State.txt", encoding="utf-8") as f:
    Model_Fact_Entity_Level_Detect_State_Text = f.read().strip()
Model_Fact_Entity_Level_Detect_State_Template = PromptTemplate.from_template(Model_Fact_Entity_Level_Detect_State_Text)

# Model_Fact_Entity_Level
with open(BASE_PROMPTS / "Model_Fact_Entity_Level.txt", encoding="utf-8") as f:
    Model_Fact_Entity_Level_Text = f.read().strip()
Model_Fact_Entity_Level_Template = PromptTemplate.from_template(Model_Fact_Entity_Level_Text)

# Model_Fact_Columns_Level_Detect_State
with open(BASE_PROMPTS / "Model_Fact_Columns_Level_Detect_State.txt", encoding="utf-8") as f:
    Model_Fact_Columns_Level_Detect_State_Text = f.read().strip()
Model_Fact_Columns_Level_Detect_State_Template = PromptTemplate.from_template(Model_Fact_Columns_Level_Detect_State_Text)

# Model_Fact_Columns_Level
with open(BASE_PROMPTS / "Model_Fact_Columns_Level.txt", encoding="utf-8") as f:
    Model_Fact_Columns_Level_Text = f.read().strip()
Model_Fact_Columns_Level_Template = PromptTemplate.from_template(Model_Fact_Columns_Level_Text)

# Model_Fact_Scripts_Creation_Detect_State
with open(BASE_PROMPTS / "Model_Fact_Scripts_Creation_Detect_State.txt", encoding="utf-8") as f:
    Model_Fact_Scripts_Creation_Detect_State_Text = f.read().strip()
Model_Fact_Scripts_Creation_Detect_State_Template = PromptTemplate.from_template(Model_Fact_Scripts_Creation_Detect_State_Text)

# Model_Fact_Scripts_Creation
with open(BASE_PROMPTS / "Model_Fact_Scripts_Creation.txt", encoding="utf-8") as f:
    Model_Fact_Scripts_Creation_Text = f.read().strip()
Model_Fact_Scripts_Creation_Template = PromptTemplate.from_template(Model_Fact_Scripts_Creation_Text)

# Model_Fact_Create_Modeling_Feed
with open(BASE_PROMPTS / "Model_Fact_Create_Modeling_Feed.txt", encoding="utf-8") as f:
    Model_Fact_Create_Modeling_Feed_Text = f.read().strip()
Model_Fact_Create_Modeling_Feed_Template = PromptTemplate.from_template(Model_Fact_Create_Modeling_Feed_Text)

# Model_Fact_Create_Modeling_Feed_Detect_State
with open(BASE_PROMPTS / "Model_Fact_Create_Modeling_Feed_Detect_State.txt", encoding="utf-8") as f:
    Model_Fact_Create_Modeling_Feed_Detect_State_Text = f.read().strip()
Model_Fact_Create_Modeling_Feed_Detect_State_Template = PromptTemplate.from_template(Model_Fact_Create_Modeling_Feed_Detect_State_Text)

