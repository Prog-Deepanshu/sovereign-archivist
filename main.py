import json
import asyncio
from typing import TypedDict, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

# --- SCHEMAS ---
class SearchQueries(BaseModel):
    queries: List[str] = Field(description="Exactly 3 distinct search queries.")

class ResearchState(TypedDict):
    topic: str
    queries: List[str]
    raw_data: List[str]
    critique: str
    final_report: str

# --- MODELS ---
model = ChatOllama(model="llama3", temperature=0)
writer_model = ChatOllama(model="mistral", temperature=0.7)
search_tool = DuckDuckGoSearchRun()

# --- AGENT NODES ---
async def strategist(state: ResearchState):
    structured_llm = model.with_structured_output(SearchQueries)
    prompt = f"Break down the topic '{state['topic']}' into 3 deep search queries."
    response = await structured_llm.ainvoke(prompt)
    return {"queries": response.queries}

async def scout(state: ResearchState):
    results = []
    for q in state["queries"]:
        try:
            res = await asyncio.to_thread(search_tool.run, q)
            results.append(f"Query: {q}\nResult: {res}\n")
        except:
            results.append(f"Search failed for: {q}")
    return {"raw_data": results}

async def fact_checker(state: ResearchState):
    data = "\n".join(state["raw_data"])
    prompt = f"Identify lies or contradictions in this data:\n\n{data}"
    critique = await model.ainvoke(prompt)
    return {"critique": critique.content}

async def writer(state: ResearchState):
    data = "\n".join(state["raw_data"])
    prompt = f"Topic: {state['topic']}\nData: {data}\nCritique: {state['critique']}\nWrite a deep-dive Markdown report."
    report = await writer_model.ainvoke(prompt)
    return {"final_report": report.content}

# --- GRAPH ---
workflow = StateGraph(ResearchState)
workflow.add_node("strategist", strategist)
workflow.add_node("scout", scout)
workflow.add_node("fact_checker", fact_checker)
workflow.add_node("writer", writer)
workflow.set_entry_point("strategist")
workflow.add_edge("strategist", "scout")
workflow.add_edge("scout", "fact_checker")
workflow.add_edge("fact_checker", "writer")
workflow.add_edge("writer", END)
app_graph = workflow.compile()

# --- SERVER ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def stream_logic(topic: str):
    async for event in app_graph.astream({"topic": topic}):
        for node, data in event.items():
            yield f"data: {json.dumps({'node': node})}\n\n"
            if node == "writer":
                yield f"data: {json.dumps({'report': data['final_report']})}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/research")
async def research(topic: str = Query(...)):
    return StreamingResponse(stream_logic(topic), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

