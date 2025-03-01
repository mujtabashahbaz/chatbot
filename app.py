from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict

# Import LangGraph chatbot logic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Define the State with custom fields
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph builder
memory = MemorySaver()
graph_builder = StateGraph(State)

# Define tools
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Initialize the LLM with OpenAI and bind tools
llm = ChatOpenAI(model="gpt-4o", api_key="your_openai_api_key_here")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1, "Only one tool call is supported at a time."
    return {"messages": [message]}

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges for tool usage
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END},
)

# Return to chatbot after tool execution
graph_builder.add_edge("tools", "chatbot")

# Set the start and end points
graph_builder.add_edge(START, "chatbot")

# Compile the graph with checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Define the FastAPI app
app = FastAPI()

# Pydantic model for request body
class ChatRequest(BaseModel):
    user_input: str
    thread_id: str

# Pydantic model for response
class ChatResponse(BaseModel):
    assistant_response: str

# API endpoint for chatbot interaction
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    events = graph.stream({"messages": [{"role": "user", "content": request.user_input}]}, config, stream_mode="values")
    
    # Extract the assistant's response
    assistant_response = ""
    for event in events:
        if "messages" in event:
            assistant_response = event["messages"][-1].content
    
    return ChatResponse(assistant_response=assistant_response)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}