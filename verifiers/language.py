from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type
from dataclasses import dataclass

from llms.common import LLMAccess

@dataclass(frozen=True)
class ProofSegment(ABC):
    """
    A complete proof, the beginnig parts of a proof,
    or just a proof step.
    """

    @classmethod
    @abstractmethod
    def empty_proof(cls) -> 'ProofSegment':
        pass

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

class VerifierLanguage(ABC):
    # Every field defaulting to NotImplemented
    # should be overridden in any "non-abstract" subclass
    language_name: str = None # Name in natural language
    proof_segment_type: Type[ProofSegment] = None

    def __init__(self):
        assert self.language_name is not None,\
            "Initializing a verifier language with no name."
        assert self.proof_segment_type is not None,\
            "Initializing a verifier language without rules for" +\
            " how to handle proof segments."

    @abstractmethod
    def close_proof(self, proof_segment: ProofSegment) -> str:
        """
        Given a proof segment, supply dummy parts (e.g. `end` in Lean)
        to get a proof that, assuming proof_segment is otherwise complete,
        can be run on the verifier.
        """
        pass

    @abstractmethod
    def predict_proof_step(
        self,
        proof_segment: ProofSegment,
        comment: str,
        llm_access: LLMAccess
    ) -> ProofSegment:
        """
        Interact with model to predict the next proof step.
        `comment` is not expect to conform to the syntax of
        the verifier language. Hence, it's the responsibility
        of this function to possibly wrap it.
        """
        pass


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