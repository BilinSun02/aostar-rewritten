import re
from dataclasses import dataclass
from typing import List, Tuple

from .language import *
from .verifier import Verifier
from .lean4_verifier import Lean4Verifier
from llms.prompts import *
from llms.common import LLMAccess
from .string_operations import replace_at_indices

lean4_comment_or_blank_line_pattern = r'^\s*(--.*)?$'

# The required Lean 4 theorem statement "initial segment" format:
# Following `theorem ... := by`, there must be the line ` skip`.
# This no-op serves two purposes: (1) the theorem statement needs
# it to compile successfully on the Lean 4 repl; (2) It sets the
# indentation (to be one space).
@dataclass(frozen=True)
class Lean4ProofSegment(ProofSegment):
    tactics: str
    imports: str = ""
    # Had to sidestep the `import` keyword of Python ;-)

    @classmethod
    def empty_proof(cls) -> 'Lean4ProofSegment':
        return Lean4ProofSegment("", "")

    @property
    def last_line_indentation(self) -> str:
        # Find the last non-empty, non-comment line of self.tactics
        for line in reversed(self.tactics.splitlines()):
            if not re.match(
                lean4_comment_or_blank_line_pattern,
                line,
                re.MULTILINE
            ):
                return re.match(r"^\s*", line).group(0)

        return ""

    def normalize_indentation(self, other_tactics_str) -> str:
        return '\n'.join(map(
            lambda line: self.last_line_indentation + line,
            other_tactics_str.splitlines()
        ))

    def __add__(self, other: 'Lean4ProofSegment') -> 'Lean4ProofSegment':
        assert isinstance(other, Lean4ProofSegment)

        if self.imports and not self.imports.endswith('\n'):
            imports = self.imports + '\n' + other.imports
        else:
            imports = self.imports + other.imports

        other_tactics_str = self.normalize_indentation(other.tactics)
        if self.tactics and not self.tactics.endswith('\n'):
            tactics = self.tactics + '\n' + other_tactics_str
        else:
            tactics = self.tactics + other_tactics_str

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

    # Not currently used
    #@staticmethod
    #def get_last_indentation(multiline_string):
    #    # Get the last non-empty line
    #    lines = [line for line in multiline_string.splitlines() if line.strip()]
    #    last_line = lines[-1] if lines else ""
    #    indentation_string = last_line[:len(last_line) - len(last_line.lstrip())]
    #    return indentation_string

    @staticmethod
    def first_line_indentation(tactics_str: str) -> str:
        # Find the first non-empty, non-comment line of self.tactics
        for line in tactics_str.splitlines():
            if not re.match(
                lean4_comment_or_blank_line_pattern,
                line,
                re.MULTILINE
            ):
                return re.match(r"^\s*", line).group(0)

        return ""

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
        indent_space = ' '  # Single space works for Lean 4

        lines = tactics_str.splitlines()

        for line in lines:
            if re.match(lean4_comment_or_blank_line_pattern, line):
                indented_tactics.append(line)
                continue

            while line:
                # If a line has braces in it,
                # we chop it up into multiple lines so that each brace occupies
                # a whole line while anything in between braces inhabit their own
                # lines separate from the lines of the braces.
                split_regex = r'^(\{|\}|[^\{\}]*)(.*)$'
                split_line, remainder = re.match(split_regex, line).groups()
                split_line = split_line.strip()
                remainder = remainder.strip()

                if split_line:
                    if split_line == '{'\
                        or (split_line.startswith('|') and "=>" in split_line):
                        # The latter happens when it's part of casework
                        indented_tactics.append(
                            indent_space * indent_level + split_line
                        )
                        indent_level += 1
                    elif split_line == '}':
                        indent_level -= 1
                        indented_tactics.append(
                            indent_space * indent_level + split_line
                        )
                    elif split_line:  # Non-empty part (not a brace)
                        indented_tactics.append(
                            indent_space * indent_level + split_line
                        )
                
                line = remainder

        return '\n'.join(indented_tactics)


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
-/
--[EOF]
"""
        response = llm_access.complete(message_body)   
        if "·" in response:
            raise NotImplementedError("LLM response contains '·'. \
                                      We can't parse this yet.")
                # Because it will be a bit involved to deal with the indentation
        response = self.standardize_comments_and_indentation(response)
        response_lines = response.splitlines()
        non_empty_cutoff : int = None
            # The index of the first non-comment line
        compile_cutoff : int = None
            # The max. line index up to which the code compiles
        for idx, line in enumerate(response_lines):
            if line and not re.match(lean4_comment_or_blank_line_pattern, line):
                non_empty_cutoff = idx
                break
        if non_empty_cutoff is None:
            raise ValueError("No tactic in LLM response.")
                # We could just try prompting the LLM again,
                # but more likely something is wrong with the LLM,
                # with the prompt, or with parsing.
        else:
            # Count down to non_empty_cutoff + 1
            # This boundary is desirable: if even including one nonempty line
            # renders the code non-compilable, then the response is
            # "totally wrong".
            for idx in range(len(response_lines), non_empty_cutoff, -1):
                if re.match(lean4_comment_or_blank_line_pattern,
                            response_lines[idx-1]):
                    continue # Nothing to check about a comment
                test_proof = '\n'.join(response_lines[:idx])
                test_result = self.verifier.verify(test_proof)
                if not any(map(
                    lambda m: m.severity == 'error',
                    test_result.messages
                )):
                    compile_cutoff = idx
                    break
            if compile_cutoff is None:
                # Does not compile at all. Just pass the full response
                # and the search algorithm will know this attempts fails.
                compile_cutoff = len(response_lines)

        tactics = '\n'.join(response_lines[:compile_cutoff])
        imports = '\n'.join(re.findall(
            r'^.*(?<=--\[IMPORT\])(.*?)$',
            tactics,
            re.MULTILINE
        ))

        ## Some empirical patchwork
        ## The LLM may end the completed part also with "--[END]"
        #response = re.sub(r'^\s*--\[END\].*$', '', response, flags=re.MULTILINE)

        return Lean4ProofSegment(tactics, imports)

# Unit test code
if __name__ == "__main__":
    from llms.gpt_access import GptAccess
    thm_statement = Lean4ProofSegment("""
theorem a_plus_b_b_plus_a
(a b : Nat) : a + b = b + a := by
""", "")
    gpt_access = GptAccess(model_name="gpt-4o-mini", )
    #gpt_access = GptAccess(model_name="gpt-4o", )
    server = Lean4Server()
    print(server.predict_proof_step(thm_statement, thm_statement))
