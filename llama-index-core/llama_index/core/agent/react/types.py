"""Base types for ReAct agent."""

from abc import abstractmethod
from typing import Dict

from llama_index.core.bridge.pydantic import BaseModel


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""


class ActionReasoningStep(BaseReasoningStep):
    """Action Reasoning step."""

    thought: str
    action: str
    action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return (
            # f"Thought: {self.thought}\nAction: {self.action}\n"
            # f"Action Input: {self.action_input}"
            #Rana: Changing return to JIVA-EKE
            f"JIVA-EKE Analysis: {self.thought}\n JIVA-EKE Action: {self.action}\n"
            f"JIVA-EKE Action Input: {self.action_input}"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ObservationReasoningStep(BaseReasoningStep):
    """Observation reasoning step."""

    observation: str

    def get_content(self) -> str:
        """Get content."""
        # return f"Observation: {self.observation}"
        #Rana: Changing return to JIVA-EKE
        return f"JIVA-EKE Findings: {self.observation}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ResponseReasoningStep(BaseReasoningStep):
    """Response reasoning step."""

    thought: str
    response: str
    is_streaming: bool = False

    def get_content(self) -> str:
        """Get content."""
        if self.is_streaming:
            return (
                # f"Thought: {self.thought}\n"
                # f"Answer (Starts With): {self.response} ..."
                #Rana: Changing return to JIVA-EKE
                f"JIVA-EKE Analysis: {self.thought}\n"
                f"JIVA-EKE Answer (Starts With): {self.response} ..."
            )
        else:
            # return f"Thought: {self.thought}\n" f"Answer: {self.response}"
            #Rana: Changing return to JIVA-EKE
            return f"JIVA-EKE Analysis: {self.thought}\n" f"JIVA-EKE Answer: {self.response}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return True
