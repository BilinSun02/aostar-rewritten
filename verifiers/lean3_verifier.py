from .verifier import *
import os, tempfile
import re
import typing
from subprocess import Popen, PIPE, STDOUT
from typing import Tuple, List, Optional
from .language import ProofState, EmptyProofState

lean3_proof_state_separator = "⊢"
lean3_proof_state_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
lean3_has_state_message = 'tactic failed, there are unsolved goals\nstate:'
lean3_goal_regex = rf"([\s|\S]*?){lean3_proof_state_separator}([\s|\S]*)"

# Class adapted from lean_cmd_server.py from the COPRA codebase
# Some methods are from lean_cmd_executor.py from the COPRA codebase
class Lean3Verifier(Verifier):
    def __init__(
        self,
        max_memory_in_mibs: int = 40000,
        timeout_in_secs: int = 60,
        lean_cwd: str = './testbed/src', # The root for the "Lean project"
            # TODO: Change whenever the source code hierarchy changes
    ):
        super().__init__()
        assert lean_cwd is not None, "lean_cwd must be provided"
        assert os.path.isdir(lean_cwd), "lean_cwd must be a valid directory"
        self.max_memory_in_mibs = max_memory_in_mibs
        self.lean_cwd = lean_cwd
        self.timeout_in_secs: int = timeout_in_secs

    def run_file_on_lean(
        self,
        filepath: str,
    ) -> VerificationResult:
        full_path = os.path.join(self.lean_cwd, filepath)
        assert os.path.isfile(full_path), f"filepath must be a valid file: {filepath}"
        lean_cmd = f'lean --memory={self.max_memory_in_mibs} {filepath}'
        process = Popen(
            lean_cmd, 
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = self.lean_cwd, 
            bufsize = 1, 
            universal_newlines = True)
        # Start the process, and wait for it to finish
        process.wait(timeout=self.timeout_in_secs)
        # Get the output
        output = process.stdout.read()
        # Kill the process if it is still running
        process.kill()
        # Return the output
        if len(output) == 0:
            return EmptyResult
        else:
            return self.parse_output(full_path, output) 
    
    def parse_output(self, full_path: str, output: str) -> VerificationResult:
        # AbsFilePath:Line:Column: [waring|error]: Message
        # First get absolute path from full path
        abs_path = os.path.abspath(full_path) + ':'
        messages = output.split(abs_path)
        messages = [msg for msg in messages if len(msg) > 0] # Remove empty strings
        final_messages : typing.List[Message] = []
        state : Optional[ProofState] = None
        msg_unparsed : List[str] = []
        for msg in messages:
            # Get rid of line number and column number
            try:
                line_num_str, col_num_str, level_str, text = msg.split(':', 3)
                line_num_str = line_num_str.strip()
                col_num_str = col_num_str.strip()
                severity_str = level_str.strip()
                text = text.strip()
                line_num = int(line_num_str)
                col_num = int(col_num_str)
                severity = severity_str.lower()
                if severity == 'error' and text.startswith(lean3_has_state_message):
                    unparsed_state = text[len(lean3_has_state_message):]
                    state = self.parse_proof_state(unparsed_state)
                else:
                    final_messages.append(Message(severity, text, line_num, col_num))
            except:
                msg_unparsed.append(msg)
                pass
        if len(final_messages) > 0:
            # Sort messages by line number
            final_messages.sort(key=lambda msg: msg.begin_line_num)
        last_line_num = 0 if len(final_messages) == 0 else final_messages[-1].begin_line_num
        # Now add the unparsed messages
        for msg in msg_unparsed:
            final_messages.append(Message('info', msg, last_line_num, 0))
        # re-sort
        final_messages.sort(key=lambda msg: msg.begin>line_num)
        return VerificationResult(state, final_messages)

    def verify(
        self,
        proof: str,
    ) -> Tuple[ProofState, List[Message]]:
        with tempfile.NamedTemporaryFile() as temp_file:
            # Use the temporary file
            temp_file.write(proof.encode('utf-8'))
            temp_file.seek(0)
            response = self.run_file_on_lean(temp_file.name)
            return (
                self.parse_proof_state(response.state),
                response.messages
            )

    def parse_proof_state(
        self,
        proof_state_str: str
    ) -> ProofState:
        if not proof_state_str or proof_state_str == "no goals":
            return EmptyProofState
        if lean3_proof_state_separator not in proof_state_str:
            raise ValueError(f"Invalid {proof_state_str=}")
        proof_state_str = proof_state_str.strip()
        proof_state_str += "\n\n"
        all_matches = re.findall(lean3_proof_state_regex, proof_state_str, re.MULTILINE)
        goal_strs = []
        total_goal_cnt = 0
        for _, goal_cnt, goal_str in all_matches:
            if len(goal_cnt) > 0:
                total_goal_cnt = int(goal_cnt)
            goal_str = goal_str.strip()
            goal_strs.append(goal_str)
        if total_goal_cnt > 0:
            assert len(goal_strs) == total_goal_cnt, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
        else:
            assert len(goal_strs) == 1, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
            total_goal_cnt = 1
        assert len(goal_strs) == total_goal_cnt, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
        goals = []
        for goal_str in goal_strs:
            goal = self.parse_goal(goal_str)
            goals.append(goal)
        return ProofState(proof_state_str, goals)

    def parse_goal(self, goal_str: str) -> Goal:
        goal_str = goal_str.strip()
        goal = ""
        hyps_infs = re.findall(lean3_goal_regex, goal_str, re.MULTILINE)
        assert len(hyps_infs) == 1, f"Found zero or more than one goal in the goal string: {goal_str}"
        hypotheses_str, inference = hyps_infs[0]
        hypotheses_str = hypotheses_str.strip()
        inference = inference.strip()
        hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
        # Get rid of all the empty hypotheses
        hypotheses = [hyp for hyp in hypotheses if len(hyp) > 0]
        goal = Goal(hypotheses, inference)
        return goal

# Unit test code
if __name__ == "__main__":
    v = Lean3Verifier()
    #print(v.parse_proof_state("test⊢string"))
    #print(type(v.parse_proof_state("test⊢string")))

    my_proof = """ # Test error outputs
theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :=
begin
    nonsense
end
"""

#    my_proof = """ # Test multiple goals (comment out `simp` lines)
#mutual def even, odd
#with even : nat → bool
#| 0     := tt
#| (a+1) := odd a
#with odd : nat → bool
#| 0     := ff
#| (a+1) := even a
#
#lemma even_eq_not_odd : ∀ a, even a = bnot (odd a) :=
#begin
#  intro a, induction a,
#  --simp [even, odd],
#  --simp [*, even, odd]
#end
#"""

#    my_proof = """
#theorem inequality_chain
#(a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :=
#begin
#  apply trans,
#end
#"""
    print(v.verify(my_proof))