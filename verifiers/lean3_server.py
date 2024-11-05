from .language import *
from llms.prompts import *
from dataclasses import dataclass

@dataclass(frozen=True)
class Lean3ProofSegment(ProofSegment):
    tactics: str
    imports: str = None
    # Had to sidestep the `import` keyword of Python ;-)

    def __add__(self, other: 'Lean3ProofSegment') -> 'Lean3ProofSegment':
        # Concatenate two proof steps together into one proof step.
        # !!!TODO: check the indentation of `other.tactics`
        assert isinstance(other, Lean3ProofSegment)

        if self.imports and self.imports.endswith('\n'):
            imports = self.imports + other.imports
        else:
            imports = self.imports + '\n' + other.imports

        if self.tactics and self.tactics.endswith('\n'):
            tactics = self.tactics + other.tactics
        else:
            tactics = self.tactics + '\n' + other.tactics

        return Lean3ProofSegment(tactics, imports)

    def __str__(self) -> str:
        return self.imports + self.tactics

    def indicates_abandonment(self) -> bool:
        # This hardcodes "sorry" to mean "Abandon the goal."
        # Un-hardcode this if we need to use "sorry" in the future.
        return "sorry" in self.tactics

class Lean3Server(LanguageServer[Lean3ProofSegment]):
    language_name: str = "Lean 3"
     # !!TODO

    @classmethod
    def complete_proof(self, proof_segment: Lean3ProofSegment) -> str:
        # `lean` will happily run an incomplete proof.
        # In fact, we need to remove "end"s at the end
        # to get useful goal information.
        proof_str: str = proof_segment.imports + '\n' + proof_segment.tactics
        return self.remove_end_line(proof_str)

    @staticmethod
    def remove_end_line(string:str) -> str:
        """
        Given a multiline string, remove any line that has only "end" modulo whitespaces
        """
        lines = string.split('\n')
        purged_lines = [line for line in lines if line.strip() != "end"]
        purged_string = '\n'.join(purged_lines)
        return purged_string

    # !!TODO: implement predict_proof_step
    def predict_proof_step(self, proof_segment: Lean3ProofSegment) -> str:
        messages[0]["content"] += prompt_message_token_limit
        messages[1]["content"] = goals + avoid_steps
        raise NotImplementedError