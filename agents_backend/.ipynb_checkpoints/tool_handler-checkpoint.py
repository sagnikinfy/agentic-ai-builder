from helper import *
import os
import json
from typing import List

def show_template_insert(skill_tag: str, desc: str) -> str:
    """
    While creating a new skill this template will be displayed to the frontend
    """
    if check_if_exists(skill_tag, "tools"):
        return "This skill is already exists, please try creating unique skills"
    with open("skills/tool_format.txt") as f:
        template = f.read()
    return template.format(func = skill_tag, desc = desc)


def show_template_update(skill_tag: str) -> List[str]:
    """
    While updating a skill this template will be displayed to the frontend
    """
    blob = bucket.get_blob(f"tools/{skill_tag}.py")
    if not blob:
        return f"Specific skill/tool : {skill_tag} not found"
    data = blob.download_as_string().decode()
    with open("skills/skills.json") as f:
        skillsets = json.loads(f.read())
        
    desc = skillsets.get(skill_tag, None)
    if not desc:
        return f"Specific skill/tool : {skill_tag} not found"
    return [data, desc]


def create_or_update_skill(skill_tag: str, code: str, mode: str, desc: str) -> str:
    """
    Create or update a skill/tool
    """
    if check_if_exists(skill_tag, "tools") and mode == "insert":
        return "This skill is already exists, please try creating unique skills"
    else:
        ## load to local file
        if os.path.isfile(f"skills/{skill_tag}.py"):
            os.remove(f"skills/{skill_tag}.py")
        with open(f"skills/{skill_tag}.py", "w") as f:
            f.write(code)
        ## load to storage
        blob = bucket.blob(f"tools/{skill_tag}.py")
        blob.upload_from_string(code)
        ## update dictionary
        with open("skills/skills.json") as f:
            skillsets = json.loads(f.read())
        skillsets[skill_tag] = desc
        with open("skills/skills.json", "w") as f:
            f.write(json.dumps(skillsets))
        return "updated"
    
    
def delete_skill(skill_tag: str) -> str:
    """
    Delete a skill/tool
    """
    ## Delete from bucket
    blob = bucket.blob(f"tools/{skill_tag}.py")
    blob.delete()
    ## Delete from lacal
    if os.path.isfile(f"skills/{skill_tag}.py"):
        os.remove(f"skills/{skill_tag}.py")
    ## Delete from dictionary
    with open("skills/skills.json") as f:
        skillsets = json.loads(f.read())
    skillsets.pop(skill_tag)
    with open("skills/skills.json", "w") as f:
        f.write(json.dumps(skillsets))
    return "deleted"


def test_skill(skill_tag: str, test_query: str) -> str:
    """
    Test a skill/tool
    """
    blob = bucket.get_blob(f"tools/{skill_tag}.py")
    data = blob.download_as_string().decode() + f"\nout = {skill_tag}.invoke('''{test_query}''')"
    #data = blob.download_as_string().decode()
    loc = {}
    try:
        exec(data, globals(), loc)
        out = loc['out'].content
        return out
    except Exception as e:
        return e