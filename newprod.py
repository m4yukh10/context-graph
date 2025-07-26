import os
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()
model = init_chat_model("llama3-8b-8192", model_provider="groq")

class AgentState(TypedDict):
    context: str
    prompt: str
    answer: str
    facts: List[str]
    summary: str
    mode: str  
    input_queue: List[str]  

def input_node(state: AgentState) -> AgentState:
    if not state['input_queue']:
        state['mode'] = 'done'
        state['prompt'] = ''
        return state
    user_input = state['input_queue'].pop(0)
    if user_input.strip().lower() == 'done':
        state['mode'] = 'done'
        state['prompt'] = ''
    else:
        state['prompt'] = user_input
        state['mode'] = 'ask' if user_input.strip().endswith('?') else 'tell'
    return state

def decision_node(state: AgentState) -> AgentState:
    
    return state

def decision_router(state: AgentState):
    if state['mode'] == 'ask':
        return 'answer'
    elif state['mode'] == 'tell':
        return 'store_fact'
    elif state['mode'] == 'done':
        return 'summary'
    else:
        return 'input'

def answer_node(state: AgentState) -> AgentState:
    
    context = state['context']
    if state['facts']:
        context += '\nAdditional facts: ' + '; '.join(state['facts'])
    prompt = f"Answer the following question based on the context.\nQuestion: {state['prompt']}\nContext: {context}"
    response = model([HumanMessage(content=prompt)])
    state['answer'] = response.content
    return state

def store_fact_node(state: AgentState) -> AgentState:
    
    fact = state['prompt'].strip()
    if fact:
        state['facts'].append(fact)
    state['answer'] = ''
    return state

def summary_node(state: AgentState) -> AgentState:
    
    summary_prompt = f"Summarize the following facts and context in a brief paragraph.\nContext: {state['context']}\nFacts: {'; '.join(state['facts'])}"
    response = model([HumanMessage(content=summary_prompt)])
    state['summary'] = response.content
    return state

graph = StateGraph(AgentState)
graph.add_node('input', input_node)
graph.add_node('decision', decision_node)
graph.add_node('answer', answer_node)
graph.add_node('store_fact', store_fact_node)
graph.add_node('summary', summary_node)

graph.add_edge(START, 'input')
graph.add_edge('input', 'decision')
graph.add_conditional_edges(
    'decision',
    decision_router,  
    {
        'answer': 'answer',
        'store_fact': 'store_fact',
        'summary': 'summary',
        'input': 'input',
    }
)
graph.add_edge('answer', 'input')
graph.add_edge('store_fact', 'input')
graph.add_edge('summary', END)
graph.compile()


if __name__ == "__main__":
    demo_context = "Today's meeting is about the Q2 product launch. Alice is the project manager. The deadline is June 30."
    demo_inputs = [
        "Who is the project manager?",
        "The budget is $50,000.",
        "What is the deadline?",
        "We will use agile methodology.",
        "done"
    ]
    init_state = {
        'context': demo_context,
        'prompt': '',
        'answer': '',
        'facts': [],
        'summary': '',
        'mode': '',
        'input_queue': demo_inputs.copy(),
    }
    compiled = graph.compile()
    final_state = compiled.invoke(init_state)
    print('Facts:', final_state['facts'])
    print('Summary:', final_state['summary'])
