from lean_cmd_server import LeanCmdServer
from lean_cmd_executor import Lean3Executor, ProofContext, Obligation
from lean_parse_utils import LeanLineByLineReader
import re
import logging
import os

# TODO: refactor this so that we don't need to use this as an environment.
# This will require us to ditch Lean3Executor as that needs to be used
# as an environment too
class LeanOneoffExec:
    def __init__(self, file_path: str, project_root: str = '.'):
        self.lean_stdin_reader = LeanLineByLineReader(file_path)
        self.lean_exec : Lean3Executor = Lean3Executor(
            project_root=project_root,
            use_human_readable_proof_context=True, 
            proof_step_iter=self.lean_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.lean_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.lean_exec.__exit__(exc_type, exc_value, traceback)
    
    def __call__(self):
        self.lean_exec.run_to_finish()
        return self.lean_exec.proof_context.all_goals

def run_proof_on_lean(proof: str, max_memory_in_mib:int=4000, project_root:str="."): # max_memory_in_mib was 40000 in lean_cmd_executor
    temp_file_name = "temp_proof.lean"
    #temp_file_name = "data/test/lean_proj2/temp_proof.lean"
    with open(temp_file_name, "w") as f:
        f.write(proof)
    lean_server = LeanCmdServer(memory_in_mibs=max_memory_in_mib, cwd=project_root, debug=False)
    response = lean_server.run(temp_file_name, 60)
    return (parse_proof_context_human_readable(response.state), response.messages)

# Adapted from lean_cmd_executor.py
def parse_proof_context_human_readable(proof_context_str: str) -> ProofContext:
    if proof_context_str is None or len(proof_context_str) == 0 or Lean3Executor.proof_context_separator not in proof_context_str:
        return None
    if proof_context_str == "no goals":
        return ProofContext.empty()
    proof_context_str = proof_context_str.strip()
    proof_context_str += "\n\n"
    all_matches = re.findall(Lean3Executor.proof_context_regex, proof_context_str, re.MULTILINE)
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
    return ProofContext(goals, [], [], [])

# Adapted from lean_cmd_executor.py
def parse_goal(goal_str: str):
    goal_str = goal_str.strip()
    goal = ""
    hyps_goals = re.findall(Lean3Executor.goal_regex, goal_str, re.MULTILINE)
    assert len(hyps_goals) == 1, f"Found more than one goal in the goal string: {goal_str}"
    hypotheses_str, goal = hyps_goals[0]
    hypotheses_str = hypotheses_str.strip()
    goal = goal.strip()
    hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
    # Get rid of all the empty hypotheses
    hypotheses = [hyp for hyp in hypotheses if len(hyp) > 0]
    goal = Obligation(hypotheses, goal)
    return goal

if __name__ == "__main__":
    #print(parse_proof_context_human_readable("test⊢string"))
    #print(type(parse_proof_context_human_readable("test⊢string")))

    #logging.basicConfig(filename='lean_executor.log', filemode='w', level=logging.INFO)
    ##os.chdir(root_dir)
    #project = "data/test/lean_proj"
    #file = "data/test/lean_proj/src/simpler.lean"
    #with LeanOneoffExec(file, project) as lean_exec:
    #    print(lean_exec())

    my_proof = """
theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :=
begin
end
"""

    #os.chdir("data/test/lean_proj2")
#    my_proof = """
#theorem inequality_chain
#(a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :=
#begin
#  apply trans,
#end
#"""
    #print(run_proof_on_lean(my_proof, project_root="/home/billion/Projects/aostar_rewritten/data/test/lean_proj2")) # project_root should have leanpkg.toml
    print(run_proof_on_lean(my_proof, project_root=".")) # project_root should have leanpkg.toml