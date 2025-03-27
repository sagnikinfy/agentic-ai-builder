from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm_keyfile = "infy_auto.json"
llm_project = "upheld-caldron-411606"
creds_llm = service_account.Credentials.from_service_account_file(llm_keyfile)

llm = ChatVertexAI(safety_settings=safety_settings, project = llm_project, 
                   credentials=creds_llm, location="us-central1",
                    model_name= 'gemini-1.5-flash-preview-0514',
                    temperature= 0.0,
                    top_p=0.8,
                    top_k=40,
                    verbose= True,
                    convert_system_message_to_human=False,
                    streaming = True, 
                    max_output_tokens = 8000)

from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Literal, Annotated, Sequence

def get_prompt(inp):
    return f"""-Goal-
    Given a schema of a table and an input query, generate correct Google SQL query for BigQuery to get desired output from the table.
    Below is the table and field definiton schema.

    -SCHEMA-
    Table name = `apigee-infosys.metrics.metrics_trends`

    Field name - 'case_number'
    Type - INTEGER
    Description - This field refers to case numbers of cases.

    Field name - 'ldap'
    Type - STRING
    Description - This field refers to user ids of TSR or case owner. TSR is an individual from support team who handles the cases.

    Field name - 'specialization'
    Type - STRING
    Description - This field refers to the subject in which a TSR is specialized. This field can have null values.

    Field name - 'product'
    Type - STRING
    Description - This field refers the product name under which the case was created.This field can have null values.

    Field name - 'service_level'
    Type - STRING
    Description - This field refers to the service level or segment of the customer.This field can have null values.

    Field name - 'priority'
    Type - STRING
    Description - This field refers to priority of the case. 

    Field name - 'name'
    Type - STRING
    Description - This field refers to organization or company name or customer name that opened the case with the support.This field can have null values.

    Field name - 'shard'
    Type - STRING
    Description - This field refers to the shard where a TSR belongs to.This field can have null values.

    Field name - 'team'
    Type - STRING
    Description - This field refers to the team or site where a TSR belongs to.This field can have null values.

    Field name - 'has_escalation'
    Type - BOOLEAN
    Description - This field checks if the case was escalated or not.

    Field name - 'escalation_date'
    Type - DATE
    Description - This field refers to the escalation date of a case if there is an escalation. This field can have null values.

    Field name - 'case_created_week'
    Type - DATE
    Description - This field refers to the first day of the week in which the case was created.

    Field name - 'escalated_week'
    Type - DATE
    Description - This field refers to the first day of the week in which the case was escalated (only if there is an escalation). This field can have null values.

    Field name - 'escalated_hour'
    Type - INTEGER
    Description - This field refers to the hour value when the case was escalated (only if there is an escalation). This field can have null values.

    Field name - 'escalation_owner'
    Type - STRING
    Description - This field refers to the case owner at the time of escalation (only if there is an escalation).This field can have null values.

    Field name - 'owner_tunure'
    Type - FLOAT
    Description - This field refers to experience of the escalated case owner in month (only if there is an escalation).This field can have null values.
    
    Field name - 'case_status'
    Type - STRING
    Description - This field refers to the status of the case.

    Field name - 'ces'
    Type - STRING
    Description - This field refers to the customer feedback - This field can have 'CSAT', 'DSAT' or null values. CSAT refer to positive feedback and DSAT refers to negative feedback.

    Field name - 'survey_channel'
    Type - STRING
    Description - This field refers to the survey channel. This field can have 'chat', 'email', 'phone' or null values.

    Field name - 'survey_date'
    Type - DATE
    Description - This field refers to the date when survey was received.

    Field name - 'survey_week'
    Type - DATE
    Description - This field refers to the start day of the week when the survey was received.


    -INSTRUCTION-
    1. The escalation week can be different from the case creation week.
    2. Escalation rate for a week should be the total number of cases escalated on that week out of total number of cases created on that week.
    3. Based of the above schema of the table, output will be a correct Google SQL query for BigQuery query.
    4. Output will be BigQuery SQL only. Do not add any additional information.

    ######################
    -Examples-
    ######################

    Example 1:
    Input query : what is the escalation rate for the week 2024-10-27?
    #############

    Output : SELECT
    SAFE_DIVIDE((COUNTIF(esc.escalated_week = DATE('2024-10-27')) , COUNTIF(esc.case_created_week = DATE('2024-10-27'))) AS escalation_rate
    FROM
    `apigee-infosys.metrics.metrics_trends` AS esc
    #################################################################

    Example 2:
    Input query : what is the escalation rate by specialization for the week 2024-10-27?
    ##############
    Output : SELECT
    esc.specialization
    SAFE_DIVIDE((COUNTIF(esc.escalated_week = DATE('2024-10-27')) , COUNTIF(esc.case_created_week = DATE('2024-10-27'))) AS escalation_rate
    FROM
    `apigee-infosys.metrics.metrics_trends` AS esc
    Group By 1
    #################################################################
    
    Example 3:
    Input query : what is the ces for the week 2024-10-27?
    ##############
    Output : select 
    safe_divide(
      countif(c.ces='CSAT' and c.survey_week = DATE('2024-10-27')),
      countif(c.ces is not null and c.survey_week = DATE('2024-10-27'))
    ) as ces_pct
    FROM
    `apigee-infosys.metrics.metrics_trends` AS c
    #################################################################
    
    Example 4:
    Input query : what is the ces by specialization for the week 2024-10-27?
    ##############
    Output : select 
    c.specialization,
    safe_divide(
    countif(c.ces='CSAT' and c.survey_week = DATE('2024-10-27')),
    countif(c.ces is not null and c.survey_week = DATE('2024-10-27'))
    ) as ces_rate
    FROM
    `apigee-infosys.metrics.metrics_trends` AS c
    group by 1
    #################################################################

    #############################
    -Real Data-
    ######################
    Input query: {inp}
    ######################
    Output:
    """

@tool
def sql_query_generation(query: Annotated[str, "User's query about various metrics trends"]) -> str:
    """
    Takes a query about various metrics trends regarding 'escalations', 'CES', 'TSR', 'specialization', 'shard', 'customer', 'team' etc and returns a SQL query that defines the answer.
    Do not call this tool more than once.
    
    Args:
        query (string): User's query query about various metrics trends regarding 'escalations', 'CES', 'TSR', 'specialization', 'shard', 'customer', 'team' etc
        
    Returns:
        Output SQL query (SQL query).
    
    """
    
    sql = llm.predict(get_prompt(query))
    sql = sql.lstrip("```sql").rstrip("\n```").strip()
    
    return f"Output SQL query:\n {sql}"
