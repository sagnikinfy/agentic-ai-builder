import json
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from agent_handler import *
from tool_handler import *
from req_schema import *
from supervisor import supervisor
from workflows import workflows
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command


app = FastAPI()

@app.post("/create_agent/")
async def create_agent(body: CreateAgent):
    res = {}
    try:
        out = create_or_update_agent(agent_tag = body.agent_tag , skill_tags = body.skill_tags, system_prompt = body.system_prompt, 
                                     mode = body.mode)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.post("/delete_agent/")
async def del_agent(body: AgentTag):
    res = {}
    try:
        out = delete_agent(agent_tag = body.agent_tag)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.post("/test_agent/")
async def test_agent_with_prompt(body: AgentTag):
    res = {}
    try:
        out = test_agent(agent_tag = body.agent_tag, test_query = body.prompt)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.post("/get_agent/")
async def get_agent(body: AgentTag):
    res = {}
    try:
        out = fetch_agent(agent_tag = body.agent_tag)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))



@app.post("/create_tool/")
async def create_tool(body: CreateTool):
    res = {}
    try:
        out = create_or_update_skill(skill_tag = body.skill_tag, code = body.code, mode = body.mode, desc = body.desc)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))



@app.post("/delete_tool/")
async def delete_tool(body: SkillTag):
    res = {}
    try:
        out = delete_skill(skill_tag = body.skill_tag)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))



@app.post("/test_tool/")
async def test_tool(body: SkillTag):
    res = {}
    try:
        out = test_skill(skill_tag = body.skill_tag, test_query = body.prompt)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.post("/get_tool_insert/")
async def get_tool_insert(body: SkillTag):
    res = {}
    try:
        out = show_template_insert(skill_tag = body.skill_tag, desc = body.desc)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.post("/get_tool_update/")
async def get_tool_uodate(body = SkillTag):
    res = {}
    try:
        out = show_template_update(skill_tag = body.skill_tag)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()  
    user_uuid = None  
    try:
        while True:
            data = await websocket.receive_text()  

            try:
                payload = json.loads(data)
                user_uuid = payload.get("uuid")
                message = payload.get("message")
                mode = payload.get("mode")   ## update, regen, continue
                agent_name = payload.get("agent")
                reply = payload.get("reply")
                first_msg = payload.get("first_msg")
                goto = ""
                
                if message:
                    ## if mode is init and agent_name is null then call supervisor first then execute graph
                    if mode == "init" and not agent_name:
                        goto = supervisor(message)
                    else:
                        goto = agent_name
                        
                    if goto == 'FINISH':
                        await websocket.send_text(json.dumps({"not_found" : "No agent available for this query"}))
                        
                    else:
                        
                        graph = workflows[f'{goto}']
                        thread = {"configurable": {"thread_id": user_uuid}}
                        print(f"Pending Executions! {thread}")
                        print(graph.get_state(thread).next)
                        input_data = None

                        if mode == "init":
                            input_data = {
                                    "messages": [
                                        HumanMessage(content=message)
                                    ], "agent_history": []
                                }
                        elif mode == "update":
                            followup_prompt = f"""
                            Below is the conversation of user and agent. User askes a question and agent answers it. User can ask follow up question of change or modify the answer. Your task is to take the role of the agent and response to the user and continue the conversation.

                            User: {first_msg}\n
                            Agent: {reply}\n
                            User: {message}\n
                            Agent:
                            """
                            input_data = Command(resume={"action": "update", "data" : followup_prompt})

                        elif mode == "regnerate":
                            input_data = Command(resume={"action": "regnerate", "data" : followup_prompt})

                        else:
                            input_data = Command(resume={"action": "continue"})

                        out_response = ""
                        is_correct = False

                        for event in graph.stream(input_data, thread, stream_mode="updates"):
                            print(event)
                            print("======")
                            if f"{goto}" in event:
                                out_response = event[f'{goto}']['agent_history'][0].content
                            elif "verify" in event:
                                out_response = event['verify']['agent_history'][0].content
                            elif "__interrupt__" in event:
                                is_correct = True
                            

                        out_response = json.dumps({"resp": out_response, "agent" : goto, "is_followup": is_correct})
                        await websocket.send_text(out_response)
                                 
                    
            except KeyError as e:
                print(f"Agent not found - {e}")
                await websocket.send_text(json.dumps({"error" : f"Agent not found - {e}"}))
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if user_uuid:
            print("Closing connection.")
        try:
            await websocket.close()
        except RuntimeError as e:
            print(f"WebSocket close error: {e}")
            


