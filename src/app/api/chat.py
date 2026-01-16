from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.chat.agent import AgentRouter
from src.domain.chat.models import AgentStep, AnswerAction, RefuseAction

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    steps: list[AgentStep]

@router.post("/", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Interact with the Agentic RAG system.
    """
    agent = AgentRouter()
    steps = []
    final_answer = ""
    
    async for step in agent.run(request.query):
        steps.append(step)
        if step.action:
            if isinstance(step.action, AnswerAction):
                final_answer = step.action.answer
            elif isinstance(step.action, RefuseAction):
                final_answer = f"Refused: {step.action.reason}"
                
    return ChatResponse(
        answer=final_answer or "No answer generated.",
        steps=steps
    )
