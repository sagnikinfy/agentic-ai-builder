from pydantic import BaseModel
from typing import List

class CreateAgent(BaseModel):
    agent_tag: str
    skill_tags: List[str]
    system_prompt: str
    desc : str | None = None
    mode: str
    
class AgentTag(BaseModel):
    agent_tag: str
    prompt: str | None = None
    
    
class CreateTool(BaseModel):
    skill_tag: str
    code: str
    mode: str
    desc: str
    
class SkillTag(BaseModel):
    skill_tag: str
    prompt: str | None = None
    desc: str | None = None