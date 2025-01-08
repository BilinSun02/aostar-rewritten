import re
from dataclasses import dataclass

from .language import *
from .verifier import Verifier, ProofState, Message
from .lean4_verifier import Lean4Verifier
from llms.prompts import *
from llms.common import LLMAccess

@dataclass(frozen=True)
class Lean4ProofSegment(ProofSegment):
    tactics: str
    imports: str = ""
    # Had to sidestep the `import` keyword of Python ;-)

    @classmethod
    def empty_proof(cls) -> 'Lean4ProofSegment':
        return Lean4ProofSegment("", "")

    def __add__(self, other: 'Lean4ProofSegment') -> 'Lean4ProofSegment':
        # Concatenate two proof steps together into one proof step.
        # !!!TODO: check the indentation of `other.tactics`
        assert isinstance(other, Lean4ProofSegment)

        if self.imports and self.imports.endswith('\n'):
            imports = self.imports + other.imports
        else:
            imports = self.imports + '\n' + other.imports

        if self.tactics and self.tactics.endswith('\n'):
            tactics = self.tactics + other.tactics
        else:
            tactics = self.tactics + '\n' + other.tactics

        return Lean4ProofSegment(tactics, imports)

    def __str__(self) -> str:
        return self.imports + '\n' + self.tactics

    @property
    def indicates_abandonment(self) -> bool:
        # This hardcodes "sorry" to mean "Abandon the goal."
        # Un-hardcode this if we need to use "sorry" in the future.
        return "sorry" in self.tactics

class Lean4Server(VerifierLanguage):
    language_name: str = "Lean 4"
    proof_segment_type: Type[ProofSegment] = Lean4ProofSegment
    verifier: Verifier = Lean4Verifier()

    def complete_proof(self, proof_segment: Lean4ProofSegment) -> str:
        proof_str: str = proof_segment.imports + '\n' + proof_segment.tactics
        # !!TODO: may also need `sorry`
        return proof_str

    def close_proof(self, proof_segment: ProofSegment) -> str:
        proof_str: str = proof_segment.imports + '\n' + proof_segment.tactics
        return proof_str

    def predict_proof_step(
        self,
        proof_segment: Lean4ProofSegment,
        comment: str,
        llm_access: LLMAccess
    ) -> str:
        message_body = f"""
/-
The following, up to "--[EOF]", was an incomplete Lean 4 proof.
An expert picked up from there and completed the proof.
The expert first planned out the proof, and kept thoughts
as comments of the form
"--[THOUGHTS]..."
before writing up any actual tactics.
The expert was unable to add anything to the beginning
of the document, in particular any `import` statements.
To make up for this, the expert would added a comment of
the following form, if necessary, before the line that
depends on the import:
"--[IMPORT]import xxx"
so that the reader can add the `import`s to the beginning
to get a runnable proof. (Note that `import xxx` should
be runnable as a Lean statement and is not in natural language.)
-/
{proof_segment.imports}
{proof_segment.tactics}
/-
{comment}
[END]
-/
"""
        response = llm_access.complete(message_body)   
        # Get lines up to the first non-comment
        response_lines = response.splitlines()
        comment_line_pattern = r'^\s*--.*$'
        response_lines_up_to_first_non_comment = []
        for line in response_lines:
            response_lines_up_to_first_non_comment.append(line)
            if line and not re.match(comment_line_pattern, line):
                break
        # TODO: instead of just getting one line as such,
        # try to find as many lines as runnable.

        tactics = '\n'.join(response_lines_up_to_first_non_comment)
        imports = '\n'.join(re.findall(r'^\s*--\[IMPORT\].*$', tactics, re.MULTILINE))

        ## Some empirical patchwork
        ## The LLM may end the completed part also with "--[END]"
        #response = re.sub(r'^\s*--\[END\].*$', '', response, flags=re.MULTILINE)

        return Lean4ProofSegment(tactics, imports)

# Unit test code
if __name__ == "__main__":
    from llms.gpt_access import GptAccess
    # !!TODO: find good example
    thm_statement = Lean4ProofSegment("""
""", "")
    gpt_access = GptAccess(model_name="gpt-4o-mini", )
    #gpt_access = GptAccess(model_name="gpt-4o", )
    server = Lean4Server()
    print(server.predict_proof_step(thm_statement, thm_statement))