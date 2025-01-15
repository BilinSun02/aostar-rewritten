import re
from dataclasses import dataclass
from typing import List, Tuple

from .language import *
from .verifier import Verifier
from .lean3_verifier import Lean3Verifier
from llms.prompts import *
from llms.common import LLMAccess
from .string_operations import replace_at_indices

lean3_comment_or_blank_line_pattern = r'^\s*(--.*)?$'

@dataclass(frozen=True)
class Lean3ProofSegment(ProofSegment):
    tactics: str
    imports: str = ""
    # Had to sidestep the `import` keyword of Python ;-)

    @classmethod
    def empty_proof(cls) -> 'Lean3ProofSegment':
        return Lean3ProofSegment("", "")

    def __add__(self, other: 'Lean3ProofSegment') -> 'Lean3ProofSegment':
        assert isinstance(other, Lean3ProofSegment)

        if self.imports and not self.imports.endswith('\n'):
            imports = self.imports + '\n' + other.imports
        else:
            imports = self.imports + other.imports

        if self.tactics and not self.tactics.endswith('\n'):
            tactics = self.tactics + '\n' + other.tactics
        else:
            tactics = self.tactics + other.tactics

        return Lean3ProofSegment(tactics, imports)

    def __str__(self) -> str:
        return self.imports + '\n' + self.tactics

    @property
    def indicates_abandonment(self) -> bool:
        # This hardcodes "sorry" to mean "Abandon the goal."
        # Un-hardcode this if we need to use "sorry" in the future.
        return "sorry" in self.tactics

class Lean3Server(VerifierLanguage):
    language_name: str = "Lean 3"
    proof_segment_type: Type[ProofSegment] = Lean3ProofSegment
    verifier: Verifier = Lean3Verifier()

    @staticmethod
    def normalize_comments_and_indentation(tactics_str: str) -> str:
        """
        Convert comments into the `--` format,
        and remove all indentation before `--` or actual tactics.
        (Spaces will need to be added back later to assemble into a proof.)
        """

        # First work on the `/- ... -/` comments, which have strange behaviros:
        # Unlike the "greedy" manner of C where `/* /* */` is considered closed,
        # Lean 3 and 4 do not consider `/- /- -/` closed, until it's completed
        # to `/- /- -/ -/`.
        block_comment_level = 0
        idx = 1
        replacements : List[Tuple[Tuple[int, int], str]] = []
        while idx < len(tactics_str):
            if tactics_str[idx-1:idx+1] == "/-":
                block_comment_level += 1
                replacements.append(((idx-1, idx+1), "--"))
                idx += 2
            elif tactics_str[idx-1:idx+1] == "-/":
                #assert block_comment_level > 0
                    # Maybe not part of a comment?
                if block_comment_level == 1:
                    # Closing block. A tactic may ensue on the same line.
                    # We just break the possible tactic onto its own line
                    # to avoid dealing with the weird indentation system of
                    # Lean, which counts "-/" towards indentation.
                    block_comment_level -= 1
                    replacements.append(((idx-1, idx+1), "\n"))
                elif block_comment_level > 1:
                    block_comment_level -= 1
                    replacements.append(((idx-1, idx+1), ""))
                idx += 2
            elif block_comment_level == 0 and tactics_str[idx-1:idx+1] == "--":
                # Any subsequent `/-` in this line should be ignored
                while idx < len(tactics_str) and tactics_str[idx] != '\n':
                    idx += 1
            elif block_comment_level > 0 and tactics_str[idx-1] == '\n':
                replacements.append(((idx, idx), "--"))
                idx += 1
            else:
                idx += 1

        assert block_comment_level == 0, "Block comments not closed"
        tactics_str = replace_at_indices(tactics_str, replacements)

        # Now normalize the indentation
        indented_tactics = []
        indent_level = 0
        indent_space = ' '  # Single space works for Lean 4

        # Split the source code into lines
        lines = tactics_str.splitlines()

        for line in lines:
            stripped_line = line.strip()
            # One part being either a brace or a string w/o braces
            parts = []
            current_part = ''
 
            for char in stripped_line:
                if char in ['{', '}']:
                    if current_part:
                        parts.append(current_part)
                        current_part = ''
                    parts.append(char)
                else:
                    current_part += char

            if current_part:
                parts.append(current_part)

            for part in parts:
                part = part.strip()
                if part == '{':
                    indented_tactics.append(indent_space * indent_level + '{')
                    indent_level += 1
                elif part == '}':
                    indent_level -= 1
                    indented_tactics.append(indent_space * indent_level + '}')
                elif part:  # Non-empty part (not a brace)
                    indented_tactics.append(indent_space * indent_level + part)

        return '\n'.join(indented_tactics)

    def close_proof(self, proof_segment: Lean3ProofSegment) -> str:
        proof_str: str = proof_segment.imports + '\n' + proof_segment.tactics
        proof_str += '\nend'
        return proof_str

    @staticmethod
    def remove_end_line(string: str) -> str:
        """
        Given a multiline string, remove any line that has only "end" modulo whitespaces
        Useful for when the LLM adds `end` at the end of a proof; we want to add `end`
        ourselves.
        """
        lines = string.split('\n')
        purged_lines = [line for line in lines if line.strip() != "end"]
        purged_string = '\n'.join(purged_lines)
        return purged_string

    def predict_proof_step(
        self,
        proof_segment: Lean3ProofSegment,
        comment: str,
        llm_access: LLMAccess
    ) -> str:
        message_body = f"""
/-
The following, up to "--[EOF]", was an incomplete Lean 3 proof.
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
-/
--[EOF]
"""
        response = llm_access.complete(message_body)   
        response = self.normalize_comments_and_indentation(response)
        # Get lines up to the first non-comment
        response_lines = response.splitlines()
        response_lines_up_to_first_non_comment = []
        for line in response_lines:
            response_lines_up_to_first_non_comment.append(line)
            if line and not re.match(lean3_comment_or_blank_line_pattern, line):
                break
        # TODO: instead of just getting one line as such,
        # try to find as many lines as runnable.

        tactics = '\n'.join(response_lines_up_to_first_non_comment)
        imports = '\n'.join(re.findall(
            r'^\s*--\[IMPORT\].*$',
            tactics,
        re.MULTILINE))
        # !!TODO: check if import detection works

        ## Some empirical patchwork
        ## The LLM may end the completed part also with "--[END]"
        #response = re.sub(r'^\s*--\[END\].*$', '', response, flags=re.MULTILINE)

        return Lean3ProofSegment(tactics, imports)

# Unit test code
if __name__ == "__main__":
    from llms.gpt_access import GptAccess
    thm_statement = Lean3ProofSegment("""
theorem a_plus_b_b_plus_a
(a b : ℕ) : a + b = b + a :=
begin
""", "import data.nat.basic")
    gpt_access = GptAccess(model_name="gpt-4o-mini", )
    #gpt_access = GptAccess(model_name="gpt-4o", )
    server = Lean3Server()
    print(server.predict_proof_step(thm_statement, thm_statement))