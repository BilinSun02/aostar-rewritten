import re
from dataclasses import dataclass
from typing import List, Tuple, Type
import multiprocessing as mp

from .language import *
from .verifier import Verifier
from .lean4_verifier import Lean4Verifier
from llms.prompts import *
from llms.common import LLMAccess
from utils.string_operations import ConcatSafeStr, replace_at_indices

lean4_comment_or_blank_line_pattern = r'^\s*(--.*)?$'

@dataclass(frozen=True)
class Lean4ProofSegment(ProofSegment):
    tactics: ConcatSafeStr
    imports: ConcatSafeStr = ""

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
        return Lean4ProofSegment(
                self.tactics + other.tactics,
                self.imports + other.imports
        )

    def __str__(self) -> str:
        return self.imports + self.tactics

    @property
    def indicates_abandonment(self) -> bool:
        # This hardcodes "sorry" to mean "Abandon the goal."
        # Un-hardcode this if we need to use "sorry" in the future.
        return "sorry" in self.tactics

class Lean4Server(VerifierLanguage):
    language_name: str = "Lean 4"
    proof_segment_type: Type[ProofSegment] = Lean4ProofSegment
    verifier: Type[Verifier] = Lean4Verifier

    def close_proof(self, proof_segment: ProofSegment) -> str:
        indt = proof_segment.last_line_indentation
        if not indt:
            indt = ' '
        # Lean 4 refuses to compile empty `by` statements, i.e. one
        # without any tactic in it. And when the statement is not
        # empty, an extra `skip` no-op doesn't break anything.
        return proof_segment.imports + proof_segment.tactics + indt + 'skip'

    def predict_proof_step(
        self,
        proof_segment: Lean4ProofSegment,
        comment: str,
        llm_access: LLMAccess
    ) -> str:
        if llm_access.follows_instructions:
            message_body = f"""
/-
The following, up to "--[EOF]", is an incomplete Lean 4 proof.
Pick up from there and completed the proof. First plan out the
proof, and keep thoughts as comments of the form "--[THOUGHTS]..."
before writing up any actual tactics. As you are unable to add
anything to the beginning of the document, in particular, any
`import` statements, add a comment of the following form, if
necessary, before the line that depends on the import, e.g.
"--[IMPORT]import Mathlib.Data.Nat.Prime"
and a post-processing program will add the `import`s for you.

Otherwise, your response will be simply concatenated with the
given proof segment and run on Lean. In particular,
(1) No natural language or otherwise extraneous text may appear
in comments. Your [IMPORT] statement should also not be written
using natural language.
(2) You are responsible for providing appropriate indentation.
You may need nonzero indentation starting from the first line.
-/
{proof_segment.imports}
{proof_segment.tactics}
/-
{comment}
-/
--[EOF]
"""
        else: # The model wouldn't quite understand our comments anyway
            message_body = str(proof_segment)
        response = llm_access.complete(message_body)   

        imports = '\n'.join(re.findall(
            r'^.*(?<=--\[IMPORT\])(.*?)$',
            response,
            re.MULTILINE
        ))
        if imports and not imports.endswith('\n'):
            imports += '\n'

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
            raise ValueError(f"No tactic in LLM {response=}")
                # We could just try prompting the LLM again,
                # but more likely something is wrong with the LLM,
                # with the prompt, or with parsing.

        candidates: List[str] = []
        # Count down to non_empty_cutoff + 1 (inclusive)
        # This boundary is desirable: if even including one nonempty line
        # renders the code non-compilable, then the response is
        # "totally wrong".
        for idx in range(len(response_lines), non_empty_cutoff, -1):
            if not re.match(lean4_comment_or_blank_line_pattern, response_lines[idx-1]):
                # Nothing to check about a comment
                candidates.append('\n'.join(response_lines[:idx]))

        test_proofs = map(
                lambda seg: imports + str(proof_segment) + '\n' + seg,
                candidates
        )
        with mp.Pool() as pool:
            test_results = pool.map(self.verifier().verify, test_proofs)

        tactics : str = None
        for idx, result in enumerate(test_results):
            if not any(map(lambda m: m.severity == 'error', result.messages)):
                tactics = candidates[idx]
                break

        if tactics is None:
            # Does not compile at all. Just pass the full response
            # and the search algorithm will know this attempts fails.
            tactics = response
        if tactics and not tactics.endswith('\n'):
            tactics += '\n'

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
