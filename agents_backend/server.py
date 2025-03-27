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
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def agent_execution_log(e, ag):
    steps = e[f'{ag}_node']['agent_history'][0].additional_kwargs['intermediate_steps']
    out = f"Agent invoked - {ag}\n\nTool details:\n"
    counter = 1
    for i in steps:
        out += f"===========================\n{counter}. Tool name: {i[0].tool}\n===========================\nLog data: {i[0].log}\nOut: {i[1]}\n==========================\n"
        counter += 1
    return out

@app.post("/create_agent/")
async def create_agent(body: CreateAgent):
    res = {}
    try:
        out = create_or_update_agent(agent_tag = body.agent_tag , skill_tags = body.skill_tags, system_prompt = body.system_prompt, 
                                     desc = body.desc,
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
    #out = test_agent(agent_tag = body.agent_tag, test_query = body.prompt)
    #res["success"] = out
    #return JSONResponse(content=jsonable_encoder(res))
    try:
        out = test_agent(agent_tag = body.agent_tag, test_query = body.prompt)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.post("/get_agent_tag/")
async def get_agent(body: AgentTag):
    res = {}
    try:
        out = fetch_agent(agent_tag = body.agent_tag)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))
    

@app.get("/get_agent/")
async def get_all_agents():
    res = {}
    try:
        out = fetch_all_agents()
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    return JSONResponse(content=jsonable_encoder(res))


@app.get("/get_tools/")
async def get_all_tools():
    res = {}
    try:
        out = fetch_all_skills()
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
    #out = test_skill(skill_tag = body.skill_tag, test_query = body.prompt)
    try:
        out = test_skill(skill_tag = body.skill_tag, test_query = body.prompt)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    print(res)
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
async def get_tool_update(body: SkillTag):
    res = {}
    try:
        out = show_template_update(skill_tag = body.skill_tag, desc = body.desc)
        res["success"] = out
    except Exception as e:
        res["error"] = str(e)
    print(res)
    return JSONResponse(content=jsonable_encoder(res))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()  
    user_uuid = None  
    try:
        while True:
            data = await websocket.receive_text()  
            #print(data)

            try:
                payload = json.loads(data)
                print(payload);
                user_uuid = payload.get("uuid")
                message = payload.get("message")
                mode = payload.get("mode")   ## update, regen, continue
                agent_name = payload.get("agent")
                reply = payload.get("reply")
                first_msg = payload.get("first_msg")
                check_follow_up = payload.get("followup")
                goto = ""
                
                if message:
                    ## Get the agent from supoervisor
                    #if mode == "init" and not agent_name:
                    if not check_follow_up:
                        goto = supervisor(message)
                    else:
                        goto = agent_name

                    ###################################

                    ## If legit agent found then execute graph
                        
                    if goto == 'FINISH':
                        await websocket.send_text(json.dumps({"error" : "No agent available for this query", "is_followup" : False}))
                        
                    else:

                        ## Valid agent found
                        
                        graph = workflows[f'{goto}']
                        thread = {"configurable": {"thread_id": user_uuid}}
                        #print(f"Pending Executions! {thread}")
                        #print(graph.get_state(thread).next)
                        input_data = None

                        if mode == "init":
                            input_data = {
                                    "messages": [
                                        HumanMessage(content=message)
                                    ]
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
                            input_data = Command(resume={"action": "regnerate", "data" : first_msg})

                        else:
                            input_data = Command(resume={"action": "continue"})

                        #out_response = reply.replace("\n\n\nDoes it answer to your query ?","")
                        out_response = "";
                        is_correct = False
                        for event in graph.stream(input_data, thread, stream_mode="updates"):
                            print(event)
                            print("======")
                            if f"{goto}_node" in event:
                                out_response = event[f'{goto}_node']['agent_history'][0].content
                                execution_log = agent_execution_log(event, goto)
                                await websocket.send_text(json.dumps({"log" : execution_log}))
                            elif "verify_node" in event:
                                out_response = event['verify_node']['agent_history'][0].content
                            elif "__interrupt__" in event:
                                out_response += "\n\n\nDoes it answer to your query ?"
                                is_correct = True
                            
                        if mode == "init":
                            out_response = json.dumps({"resp": out_response, "agent" : goto, "is_followup": is_correct, "firstq" : first_msg})
                        else:
                            out_response = json.dumps({"resp": out_response, "agent" : goto, "is_followup": is_correct})
                        await websocket.send_text(out_response)
                                 
                    
            except KeyError as e:
                print(f"Agent not found - {e}")
                await websocket.send_text(json.dumps({"error" : f"Agent not found - {e}", "is_followup" : False}))
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if user_uuid:
            print("Closing connection.")
        try:
            await websocket.close()
        except RuntimeError as e:
            print(f"WebSocket close error: {e}")
            


