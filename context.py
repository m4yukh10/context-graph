import getpass
import os
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()
model = init_chat_model("llama3-8b-8192", model_provider="groq")   
class AgentState(TypedDict):
    name: str
    age: int
    rating: int
    answer: str
    memory: str
    decision: str
    model: str
def model_call(state: AgentState) -> AgentState:
    state['model'] = f"this is the model used by {state['name']}"
    return state

def answers(state: AgentState) -> AgentState:
    state['answer'] = f"hello there {state['name']}"
    state['rating'] += 1
    
    return state

def memory(state: AgentState) -> AgentState:
    state['memory'] = f"memory updated for {state['name']}"
    state['age'] += 1
    
    return state

def decision(state: AgentState):
    if state['decision'] == "?":
        return "answer"
    elif state['decision'] == "///":
        return "memories"    
graph = StateGraph(AgentState)
graph.add_node("model", model_call)
graph.add_node("answers", answers)
graph.add_node("memory", memory)
graph.add_node("decision", decision)


graph.add_edge(START, "model")
graph.add_edge("model", "decision")
graph.add_conditional_edges(
    "decision",
    decision,
    {
        "answer": "answers",
        "memories": "memory",
    }
    
)

graph.add_edge("memory", "model")
graph.add_edge("answers", "model")

graph.compile()
