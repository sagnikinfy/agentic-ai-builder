from helper import *
import os
import json
from typing import List, Any
#from langchain_core.tools import tool
#from langchain_google_vertexai import ChatVertexAI
#from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
#from google.oauth2 import service_account
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
#from langchain.agents import AgentExecutor, create_tool_calling_agent
#from langchain_core.messages import AIMessage, HumanMessage

def agent_logic_builder(agent_tag: str, skill_tags: List[str], system_prompt: str) -> str:
    """
    Agent code generation
    """
    with open("agents/agent_template.txt") as f:
        code = f.read()
    tools_list = "["
    for i in skill_tags:
        code += f"\nfrom skills.{i} import {i}"
        tools_list += f"{i},"
    tools_list = tools_list[:-1] + "]"
        
        
    code += f"\n{agent_tag} = create_tool_agent(llm=llm, tools = {tools_list}, system_prompt = '''{system_prompt}''')"
    #print(code)
    #code += f"out = {agent_tag}.invoke({{'messages': [HumanMessage(content='Hi')], 'agent_history': []}})
    return code




def create_graph(agent_tag: str) -> str:
    """
    Execution flow graph creation
    """
    code = """from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
import vertexai
from langchain_core.messages.base import BaseMessage
import operator
import functools
from typing_extensions import TypedDict, List, Union, Tuple, Dict
from typing import Literal, Annotated, Sequence
import httpx
import json
import pandas as pd
import asyncio
import re
from google.cloud import storage
from google.cloud import bigquery
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from enum import Enum
from pydantic import BaseModel
from json import JSONDecodeError
import langgraph
from langgraph.types import Command,interrupt
from langgraph.graph import StateGraph, START, END, MessagesState

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm_keyfile = "xxx.json"
llm_project = "xxxxx"
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
    """
    code += f"\nfrom agents.{agent_tag} import {agent_tag}\n"

    code += f"""
class State(MessagesState):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_history: Annotated[Sequence[BaseMessage], operator.add]
        
@tool 
def llm_verification_tool(prompt: Annotated[str, "prompt given by user"]):
    '''
    Takes a prompt and reply the answer

    Args:
        prompt (string)

    Returns:
        Updated response (string)
    '''
    return llm.invoke(prompt)
        
def create_tool_agent(llm: ChatVertexAI, tools: list, system_prompt: str):
    '''
    Helper function to create agents with custom tools and system prompt

    Args:
        llm (ChatVertexAI): LLM for the agent
        tools (list): list of tools the agent will use
        system_prompt (str): text describing specific agent purpose

    Returns:
        executor (AgentExecutor): Runnable for the agent created.
    '''

    system_prompt_template = PromptTemplate(

                template= system_prompt + '''
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
                Use the following context to answer your query 
                if available: \n {{agent_history}} \n
                ''',
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
        
system_prompt_verify = '''
        You are a helpful assistant who replies on user's prompt.
        Use your tools to complete the request. If you do not have a tool to complete the request, say so. 
'''

verify_agent = create_tool_agent(llm, [llm_verification_tool], system_prompt_verify)

def crew_node_verify(state, agent, name):
    input = {{'messages': [state['messages'][-1]], 'agent_history' : state['agent_history']}}
    result = agent.invoke(input)
    return {{"agent_history": [AIMessage(content= result["output"], 
                                        additional_kwargs= {{'intermediate_steps' : result['intermediate_steps']}}, name=name)]}}

def crew_node(state, agent, name):
    input = {{'messages': [state['messages'][-1]], 'agent_history' : []}}
    result = agent.invoke(input)
    return {{"agent_history": [AIMessage(content= result["output"], 
                                        additional_kwargs= {{'intermediate_steps' : result['intermediate_steps']}}, name=name)]}}
                                     
        
def human_review_node(state) -> Command[Literal["verify_node", "{agent_tag}_node"]]:
    #last_message = state['agent_history'][-1].content
    human_review = interrupt(
        {{
            "question": "Is this correct?",
        }}
    )

    review_action = human_review["action"]
    review_data = human_review.get("data")

    if review_action == "continue":
        return Command(goto=END)

    elif review_action == "regnerate":
        return Command(goto="{agent_tag}_node", update={{"messages" : [
            HumanMessage(content=review_data)
        ]}})

    elif review_action == "update":
        return Command(goto="verify_node", update={{"messages" : [
            HumanMessage(content=review_data)
        ]}})

def route_after_llm(state) -> Literal[END, "human_review_node"]:
    if not state['agent_history'][-1].additional_kwargs['intermediate_steps']:
        return END
    else:
        return "human_review_node"

verify_node = functools.partial(crew_node_verify, agent=verify_agent, name = "verify")
    """
    
    code += f"\n{agent_tag}_node = functools.partial(crew_node, agent={agent_tag}, name = '{agent_tag}')\n"
    code += f"""
def create_graph(memory):
    builder = StateGraph(State)
    builder.add_node("{agent_tag}_node",{agent_tag}_node)
    builder.add_node("verify_node",verify_node)
    builder.add_node("human_review_node",human_review_node)
    builder.add_edge(START, "{agent_tag}_node")
    builder.add_conditional_edges("{agent_tag}_node", route_after_llm)
    builder.add_conditional_edges("verify_node", route_after_llm)
    graph = builder.compile(checkpointer=memory)
    return graph
    """
    return code
    
    
def create_or_update_agent(agent_tag: str, skill_tags: List[str], system_prompt: str, desc: str, mode: str) -> str:
    """
    Create or update an agent
    """
    if check_if_exists(agent_tag, "agents") and mode == "insert":
        return "This agent is already exists, please try creating unique agents"
    else:
        ## write agent codes
        code = agent_logic_builder(agent_tag, skill_tags, system_prompt)
        ## load to local file
        if os.path.isfile(f"agents/{agent_tag}.py"):
            os.remove(f"agents/{agent_tag}.py")
 
        with open(f"agents/{agent_tag}.py", "w") as f:
            f.write(code)
        ## load to storage
        blob = bucket.blob(f"agents/{agent_tag}.py")
        blob.upload_from_string(code)
        ## update dictionary
        with open("agents/agents.json") as f:
            agentsets = json.loads(f.read())
        agentsets[agent_tag] = {"prompt" : system_prompt, "tools": skill_tags, "desc" : desc}
        with open("agents/agents.json", "w") as f:
            f.write(json.dumps(agentsets))
            
        ## create graph
        graph_code = create_graph(agent_tag)
        if os.path.isfile(f"graphs/{agent_tag}_graph.py"):
            os.remove(f"graphs/{agent_tag}_graph.py")
            
        with open(f"graphs/{agent_tag}_graph.py", "w") as f:
            f.write(graph_code)

        ## upload graph to bucket
        blob = bucket.blob(f"graphs/{agent_tag}_graph.py")
        blob.upload_from_string(graph_code)
        
        ## update workflows
        if mode == "insert":
            new_workflow = f"\nfrom graphs.{agent_tag}_graph import create_graph as {agent_tag}_graph\nworkflows['{agent_tag}']={agent_tag}_graph(memory)\n"
            with open(f"workflows.py", "a") as f:
                f.write(new_workflow)
                     
        return "updated"

    
def delete_agent(agent_tag: str) -> str:
    """
    Delete an agent
    """
    ## Delete from bucket
    blob = bucket.blob(f"agents/{agent_tag}.py")
    blob.delete()

    blob = bucket.blob(f"graphs/{agent_tag}_graph.py")
    blob.delete()
    ## Delete from workflow
    with open("workflows.py") as f:
        workflows = f.read()  
    string = f"from graphs.{agent_tag}_graph import create_graph as {agent_tag}_graph\nworkflows['{agent_tag}']={agent_tag}_graph(memory)"
    idx = workflows.find(string)
    if idx >= 0:
        with open("workflows.py", "w") as f:
            f.write(workflows[0:idx] + workflows[idx + len(string) : ])
    ## Delete from lacal
    if os.path.isfile(f"agents/{agent_tag}.py"):
        os.remove(f"agents/{agent_tag}.py")
    ## Delete from dictionary
    with open("agents/agents.json") as f:
        agentsets = json.loads(f.read())
    agentsets.pop(agent_tag)
    with open("agents/agents.json", "w") as f:
        f.write(json.dumps(agentsets))
    ## Delete from graph
    if os.path.isfile(f"graphs/{agent_tag}_graph.py"):
        os.remove(f"graphs/{agent_tag}_graph.py")
    
    return "deleted"
    

    
def test_agent(agent_tag: str, test_query: str) -> str:
    """
    Test an agent
    """
    blob = bucket.get_blob(f"agents/{agent_tag}.py")
    
    data = blob.download_as_string().decode() + f"\nout = {agent_tag}.invoke({{'messages': [HumanMessage(content='''{test_query}''')], 'agent_history': []}})\nreturn out"
    header = "def ex():\n"
    data = "    ".join(('\n'+data.lstrip()).splitlines(True))
    header += data+"\n"
    header += "out = ex()"
    #print(header)
    #data = blob.download_as_string().decode()
    loc = {}
    #exec(header, globals(), loc)
    #print(f"=========== {loc}")
    #out = loc['out']['output']
    #return out
    try:
        exec(header, globals(), loc)
        out = loc['out']['output']
        return out
    except Exception as e:
        return e
    

def fetch_agent(agent_tag: str) -> List[str]:
    """
    Fetch an agent details
    """
    with open("agents/agents.json") as f:
         agentsets = json.loads(f.read())
    prompt = agentsets[agent_tag]["prompt"]
    tools = agentsets[agent_tag]["tools"]
    return [prompt, tools]


def fetch_all_agents() -> Any:
    """
    Fetch all agents
    """
    with open("agents/agents.json") as f:
         agentsets = json.loads(f.read())
    out = []
    for i in agentsets.keys():
        out.append({
            "tag" : i,
            "desc" : agentsets[i]["desc"]
        })
    return out