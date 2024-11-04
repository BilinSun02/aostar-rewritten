from .language import *
from dataclasses import dataclass

@dataclass(frozen=True)
class Lean3ProofSegment(ProofSegment):
    body: str
    necessary_import: str = None
    # Had to sidestep the `import` keyword of Python ;-)

    def __add__(self, other: 'Lean3ProofSegment') -> 'Lean3ProofSegment':
        # Concatenate two proof steps together into one proof step.
        assert isinstance(other, Lean3ProofSegment)
        return Lean3ProofSegment(
            self.body + other.body,
            self.necessary_import + other.necessary_import
        )

    def __str__(self) -> str:
        return self.necessary_import + self.body

class Lean3(VerifierLanguage[Lean3ProofSegment]):
    language_name: str = "Lean 3"
     # !!TODO

    @classmethod
    def complete_proof(self, proof_segment: Lean3ProofSegment) -> str:
        # There's nothing to do for Lean 3: `lean` will happily run
        # an incomplete proof.
        return str(proof_segment)

    # !!TODO: implement predict_proof_step