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
                    
                    
from skills.reasoning_rag import RAG
from typing import Annotated

@tool
def generate_FMR_from_similar_cases(issue_desc: Annotated[str, "Description of the issue"], case_numbers: Annotated[str, "comma separated similar case numbers"]) -> str:
    """
    Takes issue description and similar case numbers, generates 'first meaningful response' or 'FMR'.
    You have a deatiled issue description and an array of similar case numbers. Your task is to generate FMR from knowledge base.
    Do not call this tool more than once.
        
    Args:
        issue_desc (string): Input issue description.
        case_numbers (string): Comma separated similar case numbers.

    Returns:
        first meaningful respose message or FMR message (string).
    """
    
    
    cn = []
    try:
        cn = [x.strip() for x in eval(case_numbers)]
    except Exception as e:
        cn = re.findall(r'\d+(?:,\d+)?', case_numbers)
        
    cn = [int(x) for x in cn]
    
    FMR = "Generated FMR:\n"
    
    final_out = RAG(cn, issue_desc, "FMR")
    
    return FMR + final_out