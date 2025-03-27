from langchain_google_vertexai import ChatVertexAI
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
    
from agents.fmr_generating_agent import fmr_generating_agent

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
                if available: 
 {agent_history} 

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
    input = {'messages': [state['messages'][-1]], 'agent_history' : state['agent_history']}
    result = agent.invoke(input)
    return {"agent_history": [AIMessage(content= result["output"], 
                                        additional_kwargs= {'intermediate_steps' : result['intermediate_steps']}, name=name)]}

def crew_node(state, agent, name):
    input = {'messages': [state['messages'][-1]], 'agent_history' : []}
    result = agent.invoke(input)
    return {"agent_history": [AIMessage(content= result["output"], 
                                        additional_kwargs= {'intermediate_steps' : result['intermediate_steps']}, name=name)]}
                                     
        
def human_review_node(state) -> Command[Literal["verify_node", "fmr_generating_agent_node"]]:
    #last_message = state['agent_history'][-1].content
    human_review = interrupt(
        {
            "question": "Is this correct?",
        }
    )

    review_action = human_review["action"]
    review_data = human_review.get("data")

    if review_action == "continue":
        return Command(goto=END)

    elif review_action == "regnerate":
        return Command(goto="fmr_generating_agent_node", update={"messages" : [
            HumanMessage(content=review_data)
        ]})

    elif review_action == "update":
        return Command(goto="verify_node", update={"messages" : [
            HumanMessage(content=review_data)
        ]})

def route_after_llm(state) -> Literal[END, "human_review_node"]:
    if not state['agent_history'][-1].additional_kwargs['intermediate_steps']:
        return END
    else:
        return "human_review_node"

verify_node = functools.partial(crew_node_verify, agent=verify_agent, name = "verify")
    
fmr_generating_agent_node = functools.partial(crew_node, agent=fmr_generating_agent, name = 'fmr_generating_agent')

def create_graph(memory):
    builder = StateGraph(State)
    builder.add_node("fmr_generating_agent_node",fmr_generating_agent_node)
    builder.add_node("verify_node",verify_node)
    builder.add_node("human_review_node",human_review_node)
    builder.add_edge(START, "fmr_generating_agent_node")
    builder.add_conditional_edges("fmr_generating_agent_node", route_after_llm)
    builder.add_conditional_edges("verify_node", route_after_llm)
    graph = builder.compile(checkpointer=memory)
    return graph
    