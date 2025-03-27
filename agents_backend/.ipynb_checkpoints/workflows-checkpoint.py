workflows = {}
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

from graphs.simple_agent_graph import create_graph as simple_agent_graph
workflows['simple_agent']=simple_agent_graph(memory)

