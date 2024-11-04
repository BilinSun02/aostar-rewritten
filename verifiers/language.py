from abc import ABC, abstractmethod
from typing import Type, Generic, TypeVar

ProofStep_T = TypeVar('ProofStep_T')

class ProofSegment(ABC):
    """
    A complete proof, parts of a proof, or just a proof step.
    """

    def __init__(self):
        if self.language_name == NotImplemented:
            raise TypeError("Initializing a language without a name.")
    @abstractmethod
    def __add__(self, other: 'ProofSegment') -> 'ProofSegment':
        # Concatenate two proof steps together into one proof step.
        assert isinstance(other, ProofSegment)
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
class VerifierLanguage(ABC, Generic[ProofStep_T]):
    # Every field defaulting to NotImplemented
    # should be overridden in any "non-abstract" subclass
    language_name: str = NotImplemented # Name in natural language

    @classmethod
    @abstractmethod
    def complete_proof(self, proof_segment: ProofSegment) -> str:
        """
        Given a proof segment, supply dummy parts (e.g. `sorry` in Lean 3/4)
        and closing (e.g. `end` in Lean 3) to get a proof that can be run on
        the verifier.
        """
        pass

    @classmethod
    @abstractmethod
    def predict_proof_step(cls, proof: str) -> ProofStep_T:
        """
        Interact with model to predict the next proof step.
        """
        pass