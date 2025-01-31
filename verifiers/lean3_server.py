import re
from dataclasses import dataclass
from typing import List, Tuple, Type
import multiprocessing as mp

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

        # Lean 3 does not care about indentation.
        # Just concatenate by newline.
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
    verifier: Type[Verifier] = Lean3Verifier

    @staticmethod
    def standardize_comments_and_indentation(tactics_str: str) -> str:
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
        indent_space = ' '  # Necessary for Lean 4, optional for Lean 3

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
        if llm_access.follows_instructions:
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
        else: # The model wouldn't quite understand our comments anyway
            message_body = proof_segment.imports + '\n' + proof_segment.tactics
        response = llm_access.complete(message_body)   
        response = self.standardize_comments_and_indentation(response)
        response_lines = response.splitlines()
        non_empty_cutoff : int = None
            # The index of the first non-comment line
        compile_cutoff : int = None
            # The max. line index up to which the code compiles
        for idx, line in enumerate(response_lines):
            if line and not re.match(lean3_comment_or_blank_line_pattern, line):
                non_empty_cutoff = idx
                break
        if non_empty_cutoff is None:
            raise ValueError(f"No tactic in LLM {response=}")
                # We could just try prompting the LLM again,
                # but more likely something is wrong with the LLM,
                # with the prompt, or with parsing.
        else:
            test_proofs = []
            # Count down to non_empty_cutoff + 1 (inclusive)
            # This boundary is desirable: if even including one nonempty line
            # renders the code non-compilable, then the response is
            # "totally wrong".
            for idx in range(len(response_lines), non_empty_cutoff, -1):
                if not re.match(lean3_comment_or_blank_line_pattern, response_lines[idx-1]):
                    # Nothing to check about a comment
                    test_proofs.append('\n'.join(response_lines[:idx]))

            with mp.Pool() as pool:
                test_results = pool.map(self.verifier().verify, test_proofs)

            accepted_proof : str = None
            for idx, result in enumerate(test_results):
                if not any(map(lambda m: m.severity == 'error', result.messages)):
                    accepted_proof = test_proofs[idx]
                    break

            if accepted_proof is None:
                # Does not compile at all. Just pass the full response
                # and the search algorithm will know this attempts fails.
                accepted_proof = response

        tactics = accepted_proof
        imports = '\n'.join(re.findall(
            r'^.*(?<=--\[IMPORT\])(.*?)$',
            tactics,
            re.MULTILINE
        ))

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
