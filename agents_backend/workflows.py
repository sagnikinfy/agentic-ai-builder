workflows = {}
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()


from graphs.simple_agent_graph import create_graph as simple_agent_graph
workflows['simple_agent']=simple_agent_graph(memory)

from graphs.priority_agent_graph import create_graph as priority_agent_graph
workflows['priority_agent']=priority_agent_graph(memory)

from graphs.metrics_trends_analysis_agent_graph import create_graph as metrics_trends_analysis_agent_graph
workflows['metrics_trends_analysis_agent']=metrics_trends_analysis_agent_graph(memory)


from graphs.fmr_generating_agent_graph import create_graph as fmr_generating_agent_graph
workflows['fmr_generating_agent']=fmr_generating_agent_graph(memory)
