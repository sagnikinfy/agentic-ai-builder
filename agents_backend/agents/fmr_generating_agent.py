from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage

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

def create_tool_agent(llm: ChatVertexAI, tools: list, system_prompt: str):
    """
    Helper function to create agents with custom tools and system prompt
    
    Args:
        llm (ChatVertexAI): LLM for the agent
        tools (list): list of tools the agent will use
        system_prompt (str): text describing specific agent purpose

    Returns:
        executor (AgentExecutor): Runnable for the agent created.
    """
    
    system_prompt_template = PromptTemplate(

                template= system_prompt + """
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
                Use the following context to answer your query 
                if available: \n {agent_history} \n
                """,
                input_variables=["agent_history"],
            )

    system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt_template)

    prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt,
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, 
                return_intermediate_steps= True, verbose = False)
    return executor
    
### tool calls
from skills.fetch_issue_description_from_casenumber import fetch_issue_description_from_casenumber
from skills.similar_case_search import similar_case_search
from skills.generate_FMR_from_similar_cases import generate_FMR_from_similar_cases
fmr_generating_agent = create_tool_agent(llm=llm, tools = [fetch_issue_description_from_casenumber,similar_case_search,generate_FMR_from_similar_cases], system_prompt = '''You are a helpful assistant who replies to questions about generating 'first meaningful response' or 'FMR' for a 
        given case number. 
        You have to follow below steps to generate FMR:
        1. First. fetch issue description for the given case number.
        2. Then, find similar case numbers which are similar to the issue description.
        3. Finally, using similar case numbers and the issue description generate the FMR or first meaningful response from knowledge base.
        
        Use your tools to complete the request. 
        If you do not have a tool to answer the question, say so.''')