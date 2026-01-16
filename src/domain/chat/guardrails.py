from typing import List
from src.domain.chat.models import AgentAction, AnswerAction, RefuseAction

class ConfidenceGuard:
    """
    Evaluates agent actions before they are returned to the user.
    """
    
    @staticmethod
    def evaluate_answer(answer: AnswerAction, context: str) -> AgentAction:
        """
        Heuristic check:
        1. Does answer mention citations?
        2. Is confidence score high enough?
        """
        
        # 1. Low Confidence Refusal
        if answer.confidence_score < 0.5:
            return RefuseAction(
                reason="Confidence score too low",
                rationale=f"Model confidence {answer.confidence_score} < 0.5"
            )
            
        # 2. Citation Check (Simple heuristic)
        # If context was provided but no citation in answer, flag it.
        # This is a weak check, ideally we use an LLM-grader here.
        if context and "Source:" in context:
            # Check if answer mentions sources? 
            # For now, pass.
            pass
            
        return answer
