
from typing import List, Optional
from pydantic import BaseModel, Field
from rich import print as rprint

class AvailableDimension(BaseModel):
    dimension_name:                 str  # mandatory
    main_source_table:              List[str]  # mandatory
    other_source_tables:            Optional[List[str]] = Field(default=None)
    business_key_on_source_side:    Optional[List[str]] = Field(default=None)
    surrogate_key:                  Optional[str] = Field(default=None)
    business_key_on_dimension_side: Optional[List[str]] = Field(default=None)

class AvailableDimensionEssence(BaseModel):
    dimension_name:                 str  # mandatory
    main_source_table:              List[str]  # mandatory

class ColumnDefinition(BaseModel):
    order:              Optional[int] = Field(default=None, description="Columns order in dimension" )
    column_name:        Optional[str] = Field(default=None, description="name of the collumn in target dimension" )
    column_source:      Optional[str] = Field(default=None, description="name of the collumn in source table or calculation represent column value" )
    load_logic:         Optional[str] = Field(default=None, description="Destcription how to load data in ETL process" )
    data_type:          Optional[str] = Field(default=None, description="Column data type in final data warehouse" )
    length:             Optional[int] = Field(default=None)
    precision:          Optional[int] = Field(default=None)
    column_comment:     Optional[str] = Field(default=None, description="Comment give light on busines meaining of keep values" )
    nullable:           Optional[bool] = Field(default=False)
    PII:                Optional[bool] = Field(default=False,  description="yes - if field contains personal informations" )
    column_confidentiality_level:   Optional[str] = Field(default=None, description="Data confidentiality clasyfication: C1 - Highly Confidential, C2 - Confidential, C3 - Internal, C4 - Public")
    PK:                 Optional[bool] = Field(default=False, description="Information if field is primary key")
    UK:                 Optional[bool] = Field(default=False, description="Information if field is part of unique key")
    FK:                 Optional[str] = Field(default=None, description="Information if field is foreign key and to which [table].[column] is referring")

class ContextFact(BaseModel):
    context_fact_name:              		str  # mandatory
    context_fact_comment:      			    Optional[str] = Field(default=None, description="Business description of fact with information about main source table" )
    context_fact_source:      			    List[str] = Field(description="Name of main fact which is feeding curent fact" ) 
    context_fact_filtration:                bool = Field(default=False, description="Where condition which filtering proper subset for fact from main table" )
    context_fact_rls_collumns:   			Optional[List[str]] = Field(default=None, description="Which columns will allow row level security filtration (rls)" )
    context_fact_column_to_take:          	Optional[List[ColumnDefinition]] = Field(default=None)
    context_fact_unique_key:             	Optional[str] = Field(default=None, description="Point which column from source table is the busines key" )
    context_fact_ddl:                       Optional[str] = Field(default=None, description="Definition of DDL script" )
    context_fact_sql:                       Optional[str] = Field(default=None, description="Definition of SQL script" )
    context_fact_feed_2_modeling_tool:      Optional[str] = Field(default=None, description="Definition of model which will feed designe tool" )
    context_fact_design_on_entity_level_approved:   bool = Field(default=False, description="Information if user approved designe for this fact" )
    context_fact_design_on_columns_level_approved:  bool = Field(default=False, description="Information if user approved designe for this fact" )
    context_fact_feed_2_modeling_tool_approved:     bool = Field(default=False, description="Scripts: feed_2_modeling_tool has approved by user" )
    context_fact_scripts_approved:                  bool = Field(default=False, description="Scripts: ddl, sql has approved by user" )    

class FactToCreate(BaseModel):
    fact_name:              				str  # mandatory
    fact_comment:      				        Optional[str] = Field(default=None, description="Business description of fact with information about main source table" )
    fact_main_source_table:      			List[str]  # mandatory
    fact_other_source_tables:    			Optional[List[str]] = Field(default=None)
    fact_rls_collumns:   			        Optional[List[str]] = Field(default=None, description="Which columns will allow row level security filtration (rls)" )
    fact_connected_dimensions:              Optional[List[str]] = Field(default=None)
    fact_column_list:            			Optional[List[ColumnDefinition]] = Field(default=None)
    fact_unique_key:             			Optional[str] = Field(default=None, description="Point which column from source table is the busines key" )
    fact_type:           			        Optional[str] = Field(default=None, description="Type of history: transactional(no history), ranges(valid_from, valid_to), snapshot(valid_in), accumulating(valid_from)" )
    fact_critical_columns_analyze_txt:      Optional[str] = Field(default=None, description="Sugestia którą kolumnę należy uwzględnić w fakcie i dlaczego" )
    fact_source_tables_analyze_txt:         Optional[str] = Field(default=None, description="Analizy na poziomie kolumn wszystkich tabel źródłowych dostarczone przez SAP eksperta lub analityka" )    
    fact_ddl:                               Optional[str] = Field(default=None, description="Definition of DDL script" )
    fact_sql:                               Optional[str] = Field(default=None, description="Definition of SQL script" )
    fact_feed_2_modeling_tool:              Optional[str] = Field(default=None, description="Definition of model which will feed designe tool" )
    fact_design_on_entity_level_approved:   bool = Field(default=False, description="Information if user approved designe for this fact" )
    fact_design_on_columns_level_approved:  bool = Field(default=False, description="Information if user approved designe for this fact" )
    fact_feed_2_modeling_tool_approved:     bool = Field(default=False, description="Scripts: feed_2_modeling_tool has approved by user" )
    fact_scripts_approved:                  bool = Field(default=False, description="Scripts: ddl, sql has approved by user" )

class DimensionToCreate(BaseModel):
    dimension_name:                         str  # mandatory
    dimension_comment:                      Optional[str]  = Field(default=None, description="Business description of dimension with information about main source table" )
    main_source_table:                      List[str]  # mandatory
    business_key_on_source_side:            Optional[List[str]] = Field(default=None, description="Point which column from source table is the busines key" )
    other_source_tables:                    Optional[List[str]] = Field(default=None, description="Point other tables which need to build compex dimension" )
    surrogate_key:                          Optional[str] = Field(default=None, description="Point surrodate key in final dim. Usually dim_<name>_key" )
    business_key_on_dimension_side:         Optional[List[str]] = Field(default=None, description="Point columns which are representing busines key on dimension side" )    
    history_type:                           Optional[str] = Field(default=None, description="Type of history: SCD1 - just current data, SCD2 - historization" )
    detailed_column_list:                   Optional[List[ColumnDefinition]] = Field(default=None, description="List of details informations about final dimension columns. For each column it is seperate object" )
    critical_columns_analyze_txt:           Optional[str] = Field(default=None, description="Sugestia którą kolumnę należy uwzględnić w wymiarze i dlaczego" )
    source_tables_analyze_txt:              Optional[str] = Field(default=None, description="Analizy na poziomie kolumn wszystkich tabel źródłowych dostarczone przez SAP eksperta lub analityka" )    
    ddl:                                    Optional[str] = Field(default=None, description="Definition of DDL script" )
    sql:                                    Optional[str] = Field(default=None, description="Definition of SQL script" )
    feed_2_modeling_tool:                   Optional[str] = Field(default=None, description="Definition of model which will feed designe tool" )
    design_on_entity_level_approved:        bool = Field(default=False, description="Information if user approved designe for this dimension" )
    design_on_columns_level_approved:       bool = Field(default=False, description="Information if user approved designe for this dimension" )
    feed_2_modeling_tool_approved:          bool = Field(default=False, description="Scripts: feed_2_modeling_tool has approved by user" )
    scripts_approved:                       bool = Field(default=False, description="Scripts: ddl, sql has approved by user" )
  
# MAIN class: DesignState
class DesignState(BaseModel):
    required_source_tables:             List[str] = Field( default_factory=list, description="Tabele źródłowe przekazane przez użytkownika do analizy" )
    additional_source_tables:           Optional[List[str]] = Field( default_factory=list, description="Tabele źródłowe zaproponowane i zaakceptowane w wyniku analizy" )
    required_source_tables_approved:    bool = Field( default=False, description="Czy user zaakceptował inicjalną listę tabel żródłowych?" )
    additional_source_tables_approved:  bool = Field( default=False, description="Czy user zaakceptował dodatkową listę tabel żródłowych- zaproponowaną mu na czacie?" )
    objects_to_create_approved :        bool = Field( default=False, description="Czy user zaakceptował listę faktów / wymiarów do zamodelowania?")
    last_user_message:                  Optional[str] = Field(default=None)
    last_5_messages:                    List[str] = Field( default_factory=list, description="Ostatnie 10 wiadomości (user + AI)" )
    available_dimensions:               List[AvailableDimension] = Field(default_factory=list, description="Szczegółowe informacje o obiektach już istniejących w hurtowni" )
    available_dimensions_essence:       List[AvailableDimensionEssence] = Field(default_factory=list, description="Najważniejsze informacje o obiektach już istniejących w whurtowni" )
    facts_to_create:                    List[FactToCreate] = Field(default_factory=list, description="Lista faktów do stworzenia wraz z ich szczegółami" )
    context_facts_to_create:            List[ContextFact] = Field(default_factory=list, description="Lista faktów kontekstowych do stworzenia wraz z ich szczegółami" )
    dimensions_to_create:               List[DimensionToCreate] = Field(default_factory=list, description="Lista wymiarów do stworzenia wraz z ich szczegółami" )
    currently_modeled_object:           Optional[str] = Field(default=None, description="Fact or dimension which is actually modeled" ) 
    source_table_analyze:               Optional[dict[str, str]] = Field(default_factory=dict, description="Analiza tabeli źródłowej dostarczona przez SAP eksperta lub analityka na bazie wskazanego pliku" )
    SAP_2_Snowflake_data_types:         Optional[str] = None
    awaiting_input_for:                 Optional[str] = None
    modeling_style:                     Optional[str] = Field(default=None, description="Modeling style: all columns from source table modeled in fact/dimension or traditiona clear fact/dimension")
