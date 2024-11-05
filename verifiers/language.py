from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type
from dataclasses import dataclass

@dataclass(frozen=True)
class ProofSegment(ABC):
    """
    A complete proof, the beginnig parts of a proof,
    or just a proof step.
    """

    @abstractmethod
    def __add__(self, other: 'ProofSegment') -> 'ProofSegment':
        #Concatenate two proof steps together into one proof step.
        assert isinstance(other, ProofSegment)
        pass

    def __radd__(self, other: 'ProofSegment') -> 'ProofSegment':
        assert isinstance(other, ProofSegment)
        return other.__add__(self)
    
    @abstractmethod
    def __str__(self) -> str:
        """
        Intended to return a human readable format.
        NOT guaranteed to be directly runnable on the verifier.
        """
        pass

    @property
    @abstractmethod
    def indicates_abandonment(self) -> bool:
        """
        The prompt may allow the proof predictor to abanson the
        current proof state (e.g. if the predictor realizes the
        state is unprovable) by outputing specific outputs (e.g.
        `sorry` for Lean 3 and 4).
        """
        pass

class LanguageServer[ProofSegment_T: ProofSegment](ABC):
    # Every field defaulting to NotImplemented
    # should be overridden in any "non-abstract" subclass
    language_name: str = NotImplemented # Name in natural language

    def __init__(self):
        if self.language_name == NotImplemented:
            raise TypeError("Initializing a verifier language with no name.")

    @classmethod
    @abstractmethod
    def complete_proof(self, proof_segment: ProofSegment_T) -> str:
        """
        Given a proof segment, supply dummy parts (e.g. `sorry` 4), if
        needed, to get a proof that can be run on the verifier.
        """
        pass

    #@classmethod
    #@abstractmethod
    #def predict_proof_step(cls,
    #    proof: str,
    #    comment: str
    #) -> ProofSegment_T:
    #    """
    #    Interact with model to predict the next proof step.
    #    """
    #    pass
    # !!TODO: consider the interface for this
    # Presumably similar to GPTPrompter.prompt_for_tactics

# Unit test code
if __name__ == "__main__":
    @dataclass(frozen=True)
    class TestPS(ProofSegment):
        proof: str
        def __add__(self, other):
            pass
        def __str__(self):
            return self.proof

    d = {TestPS("dummy proof"): []}
    # Should not raise, because TestPS should also be frozen,
    # i.e. immutable, making it usable for indices.
    print(d)