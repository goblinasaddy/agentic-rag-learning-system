import json
import time
from typing import List, AsyncGenerator
from litellm import completion
from src.app.core.config import settings
from src.domain.chat.models import (
    AgentState, AgentStep, AgentAction, 
    RetrieveAction, SummarizeAction, ClarifyAction, AnswerAction, RefuseAction
)
from src.domain.chat.tools import AgentTools
from src.domain.chat.guardrails import ConfidenceGuard

from typing import Optional

class AgentRouter:
    def __init__(self, tools: Optional[AgentTools] = None):
        self.tools = tools or AgentTools()
        self.guard = ConfidenceGuard()
        self.history: List[AgentStep] = []
        self.max_steps = 5
        self.model = settings.LLM_MODEL
        
    def _build_system_prompt(self) -> str:
        return """You are a specific, reliable Agentic Document Assistant.
your Goal: Answer user queries using ONLY the provided tools.
You have access to:
1. retrieve_context(query): Search documents.
2. summarize_docs(ids): Summarize specific docs.
3. clarify(question): Ask user for details.
4. refuse(reason): If you cannot answer.

RULES:
- ALWAYS 'retrieve' first if you need information.
- If retrieval is empty/irrelevant, try identifying why, or clarify.
- If you have enough info, use 'answer'.
- Be concise.
- Provide a 'rationale' for your decision.
- **VERSION CHECK**: checks context for '[WARNING: OUTDATED VERSION]'. If found, you MUST mention in your answer that the information might be outdated. Prefer valid documents if available.

RESPONSE FORMAT:
You MUST output a single JSON object.
Examples:
{"action_type": "retrieve", "query": "search query", "rationale": "reasoning"}
{"action_type": "answer", "answer": "final response", "rationale": "reasoning", "citations": ["doc1.pdf"], "confidence_score": 0.9}
{"action_type": "refuse", "reason": "why refusing", "rationale": "reasoning"}
"""

    async def run(self, user_query: str) -> AsyncGenerator[AgentStep, None]:
        """
        Main loop:
        1. User Query -> Thinking (Plan)
        2. Action Loop (Thought -> Tool -> Observation)
        3. Answer
        """
        step_count = 0
        current_context = ""
        
        # Initial State
        yield AgentStep(
            state=AgentState.THINKING,
            thought="Received query. Planning next step.",
            timestamp=time.time()
        )
        
        while step_count < self.max_steps:
            step_count += 1
            
            # 1. Prepare Messages for LLM
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            
            # Add history
            for step in self.history:
                if step.action:
                    messages.append({"role": "assistant", "content": f"Action: {step.action.action_type}\nRationale: {step.action.rationale}"})
                if step.observation:
                    messages.append({"role": "user", "content": f"Tool Output: {step.observation}"})
            
            # Current Input
            messages.append({"role": "user", "content": f"User Query: {user_query}\nCurrent Context: {current_context}"})
            
            # 2. Call LLM for Decision
            # We use LiteLLM's response_format or function calling if model supports it.
            # Ideally we use Pydantic object for structured output.
            # Since we are "FOSS/Local", we might be using Llama3 via Ollama which supports JSON mode well.
            
            try:
                response = completion(
                    model=self.model,
                    messages=messages,
                    base_url=settings.OLLAMA_BASE_URL if "ollama" in self.model else None,
                    response_format={"type": "json_object"} 
                )
                content = response.choices[0].message.content
                
                # Parse Decision (Expect JSON matching one of the Action schemas)
                # In robust implementation, we'd use a parser library. 
                # Here we trust the prompt + JSON mode.
                decision_data = json.loads(content)
                
                # Naive polymorphic parsing based on 'action_type'
                action_type = decision_data.get("action_type")
                
                if action_type == "retrieve":
                    action = RetrieveAction(**decision_data)
                elif action_type == "answer":
                    action = AnswerAction(**decision_data)
                elif action_type == "clarify":
                    action = ClarifyAction(**decision_data)
                elif action_type == "refuse":
                    action = RefuseAction(**decision_data)
                else:
                    # Fallback
                    action = RefuseAction(reason="Invalid action generated", rationale="LLM failure")
                
            except Exception as e:
                yield AgentStep(state=AgentState.REFUSING, thought=f"LLM Error: {e}", timestamp=time.time())
                return

            # 3. Create Step & Execute
            step = AgentStep(
                state=AgentState.ANALYZING, # Interim state
                thought=f"Decided to {action.action_type}",
                action=action,
                timestamp=time.time()
            )
            
            yield step
            self.history.append(step)
            
            # 4. Execute Action
            if isinstance(action, RetrieveAction):
                step.state = AgentState.RETRIEVING
                observation = await self.tools.retrieve_context(action.query)
                step.observation = observation
                current_context += f"\nRetrieval for '{action.query}':\n{observation}"
                # Loop continues
                
            elif isinstance(action, AnswerAction):
                # GUARDRAIL CHECK
                validated_action = self.guard.evaluate_answer(action, current_context)
                
                # Update step action if it changed (e.g. to Refuse)
                step.action = validated_action
                
                if isinstance(validated_action, RefuseAction):
                     step.state = AgentState.REFUSING
                     step.thought += f" (Guardrail: Refused due to {validated_action.reason})"
                else:
                    step.state = AgentState.DONE
                return # Done
                
            elif isinstance(action, ClarifyAction):
                step.state = AgentState.CLARIFYING
                return # Done (wait for user)
                
            elif isinstance(action, RefuseAction):
                step.state = AgentState.REFUSING
                return # Done
                
        # Max steps reached
        yield AgentStep(
            state=AgentState.REFUSING,
            thought="Max steps reached.",
            action=RefuseAction(reason="Too many steps", rationale="Could not resolve query in time."),
            timestamp=time.time()
        )
