# !! TODO: remove this file in the future as it's been superceded by lean3.py
from verifiers.verifiers import Message, Goal, ProofState
from .lean3_cmd_server import LeanCmdServer
from typing import Tuple, List, Dict, Any
import re
import tempfile

# Adapted from lean_cmd_executor.py from the COPRA codebase
def run_proof_on_lean(
    proof: str,
    lean_cwd: str = './testbed', # The root for the "Lean project"
        # TODO: Change whenever the source code hierarchy changes
    max_memory_in_mib: int = 40000,
    timeout_in_secs: int = 60
) -> Tuple[ProofState, List[Message]]:
    with tempfile.NamedTemporaryFile() as temp_file:
        # Use the temporary file
        temp_file.write(proof.encode('utf-8'))
        temp_file.seek(0)
        lean_server = LeanCmdServer(
            memory_in_mibs = max_memory_in_mib,
            lean_cwd = lean_cwd,
        )
        response = lean_server.run(temp_file.name, timeout_in_secs=timeout_in_secs)
        return (parse_proof_state_human_readable(response.state), response.messages)

# Adapted from lean_cmd_executor.py from the COPRA codebase
proof_state_separator = "⊢"
proof_state_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
def parse_proof_state_human_readable(proof_state_str: str) -> ProofState:
    if not proof_state_str or proof_state_str == "no goals":
        return ProofState.empty()
    if proof_state_separator not in proof_state_str:
        raise ValueError(f"Invalid {proof_state_str=}")
    proof_state_str = proof_state_str.strip()
    proof_state_str += "\n\n"
    all_matches = re.findall(proof_state_regex, proof_state_str, re.MULTILINE)
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
        goal = parse_goal(goal_str)
        goals.append(goal)
    return ProofState(proof_state_str, goals)

# Adapted from lean_cmd_executor.py from the COPRA codebase
goal_regex = rf"([\s|\S]*?){proof_state_separator}([\s|\S]*)"
def parse_goal(goal_str: str):
    goal_str = goal_str.strip()
    goal = ""
    hyps_goals = re.findall(goal_regex, goal_str, re.MULTILINE)
    assert len(hyps_goals) == 1, f"Found more than one goal in the goal string: {goal_str}"
    hypotheses_str, goal = hyps_goals[0]
    hypotheses_str = hypotheses_str.strip()
    goal = goal.strip()
    hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
    # Get rid of all the empty hypotheses
    hypotheses = [hyp for hyp in hypotheses if len(hyp) > 0]
    goal = Goal(hypotheses, goal)
    return goal

if __name__ == "__main__":
    #print(parse_proof_state_human_readable("test⊢string"))
    #print(type(parse_proof_state_human_readable("test⊢string")))

    #logging.basicConfig(filename='lean_executor.log', filemode='w', level=logging.INFO)
    ##os.chdir(root_dir)
    #project = "data/test/lean_proj"
    #file = "data/test/lean_proj/src/simpler.lean"
    #with LeanOneoffExec(file, project) as lean_exec:
    #    print(lean_exec())

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
    print(run_proof_on_lean(my_proof, lean_cwd="./testbed/src"))