from .language import *
from llms.prompts import *
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

class Lean3Server(LanguageServer[Lean3ProofSegment]):
    language_name: str = "Lean 3"
     # !!TODO

    @classmethod
    def complete_proof(self, proof_segment: Lean3ProofSegment) -> str:
        # There's nothing to do for Lean 3: `lean` will happily run
        # an incomplete proof.
        return str(proof_segment)

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

        messages = copy.deepcopy(messages_skeleton)
        messages[0]["content"] = prompt_message_introduction
        if self.include_input_format_prompt:
            messages[0]["content"] += prompt_message_input_format
        if self.think_aloud:
            messages[0]["content"] += prompt_message_output_format_with_thoughts
        else:
            messages[0]["content"] += prompt_message_output_format_wo_thoughts
        messages[0]["content"] += prompt_message_token_limit
        messages[1]["content"] = goals + avoid_steps
        raise NotImplementedError