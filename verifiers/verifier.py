from dataclasses import dataclass, field
from typing import NamedTuple, List, Optional
from abc import ABC, abstractmethod
from .language import VerifierLanguage

@dataclass
class Message:
    level: str
    file_name: str
    line_num: int
    column_num: int
    text: str

# Adapted from lean_cmd_executor.py
class Goal(NamedTuple):
    hypotheses: List[str]
    inference: str

    def format_message(self) -> str:
        message_str = f"[GOAL] \n" + self.inference + "\n[HYPOTHESES]\n"
        for hypothesis in self.hypotheses:
            message_str += "[HYPOTHESIS] " + hypothesis
        return message_str
    
    def __eq__(self, other: 'Goal'):
        if not isinstance(other, Goal):
            return False
        return self.inference == other.inference and set(self.hypotheses) == set(other.hypotheses) # Compares equality of lists up to permutation and duplicates
        # TODO: This may still fail to recognize some "actually equal" goals. For instance, if one goal results from casework, it may have an additional "case 0" etc. which doesn't really make it different.

class ProofState(NamedTuple):
    state_str: str
    goals: List[Goal]

    @classmethod
    def empty(cls: 'ProofState'):
        return ProofState("", [])

@dataclass
class VerificationResult:
    state: Optional[str] = None
    messages: List[Message] = field(default_factory=list)

EmptyResult = VerificationResult()


class Verifier(ABC):
    # Every field defaulting to NotImplemented
    # should be overridden in any "non-abstract" subclass
    language: VerifierLanguage = NotImplemented

    def __init__(self):
        if self.language == NotImplemented:
            raise TypeError("Initializing a verifier with an unknown language.")

    @abstractmethod
    def verify(self, proof: str) -> VerificationResult:
        pass

    # !!TODO: require a method for giving Goal-PartialProofArrivingAtGoal pairs
    #@abstractmethod
    #def a

# Unit test code
if __name__ == "__main__":
    v = Verifier() # Should raise an error