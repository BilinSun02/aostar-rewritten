from lean_cmd_server import LeanCmdServer
from typing import NamedTuple, Tuple, List, Dict, Any
from lean_cmd_server import Message
import re
import os
import datetime, random # For generating random temp file names

# Adapted from lean_cmd_executor.py
class Goal(NamedTuple):
    hypotheses: List[str]
    inference: str

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"hypotheses": self.hypotheses,
                "inference": self.inference}
    
    def format_message(self) -> str:
        message_str = f"[GOAL] \n" + self.inference + "\n[HYPOTHESES]\n"
        for hypothesis in self.hypotheses:
            message_str += "[HYPOTHESIS] " + hypothesis
        return message_str
    
    def __eq__(self, other: 'Goal'):
        if not isinstance(other, Goal):
            return False
        return self.inference == other.inference and set(self.hypotheses) == set(other.hypotheses) # Compares equality of lists up to permutation and duplicates
        # TODO: This may still fail to recognize some "actually equal" goals. For instance, if one goal results from casework, it may have an additional "case 0" etc. which doesn't really make it different.

# Adapted from lean_cmd_executor.py
#TODO: either make better use of the other attributes of this class
# or just eliminate this class
class ProofContext(NamedTuple):
    fg_goals: List[Goal]
    bg_goals: List[Goal]
    shelved_goals: List[Goal]
    given_up_goals: List[Goal]

    @classmethod
    def empty(cls: 'ProofContext'):
        return ProofContext([], [], [], [])

    @classmethod
    def from_dict(cls, data):
        fg_goals = list(map(Goal.from_dict, data["fg_goals"]))
        bg_goals = list(map(Goal.from_dict, data["bg_goals"]))
        shelved_goals = list(map(Goal.from_dict, data["shelved_goals"]))
        given_up_goals = list(map(Goal.from_dict,
                                  data["given_up_goals"]))
        return cls(fg_goals, bg_goals, shelved_goals, given_up_goals)

    def to_dict(self) -> Dict[str, Any]:
        return {"fg_goals": list(map(Goal.to_dict, self.fg_goals)),
                "bg_goals": list(map(Goal.to_dict, self.bg_goals)),
                "shelved_goals": list(map(Goal.to_dict,
                                          self.shelved_goals)),
                "given_up_goals": list(map(Goal.to_dict,
                                           self.given_up_goals))}

    @property
    def all_goals(self) -> List[Goal]:
        return self.fg_goals + self.bg_goals + \
            self.shelved_goals + self.given_up_goals

    @property
    def focused_goal(self) -> str:
        if self.fg_goals:
            return self.fg_goals[0].inference
        else:
            return ""

    @property
    def focused_hyps(self) -> List[str]:
        if self.fg_goals:
            return self.fg_goals[0].hypotheses
        else:
            return []


def run_proof_on_lean(
    proof: str,
    lean_cwd: str = './testbed', # The root for the "Lean project"
        # Change whenever the source code hierarchy changes
    max_memory_in_mib: int = 40000,
    timeout_in_secs: int = 60
) -> Tuple[ProofContext, List[Message]]:
    # Borrowed the idea of randomizing the name of the temp file here
    # from the copra codebase
    ticks = datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
    random_num = str(random.randint(0, 100000000))
    temp_file_name = os.path.join("src", f"temptodel_{ticks}_{random_num}.lean")
    with open(os.path.join(lean_cwd, temp_file_name), "w") as f:
        f.write(proof)
    try:
        lean_server = LeanCmdServer(
            memory_in_mibs = max_memory_in_mib,
            lean_cwd = lean_cwd,
        )
        response = lean_server.run(temp_file_name, timeout_in_secs=timeout_in_secs)
        return (parse_proof_context_human_readable(response.state), response.messages)
    finally:
        os.remove(os.path.join(lean_cwd, temp_file_name))
        pass


# Adapted from lean_cmd_executor.py
proof_context_separator = "⊢"
proof_context_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
def parse_proof_context_human_readable(proof_context_str: str) -> ProofContext:
    if not proof_context_str or proof_context_str == "no goals":
        return ProofContext.empty()
    if proof_context_separator not in proof_context_str:
        raise ValueError(f"Invalid {proof_context_str=}")
    proof_context_str = proof_context_str.strip()
    proof_context_str += "\n\n"
    all_matches = re.findall(proof_context_regex, proof_context_str, re.MULTILINE)
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
goal_regex = rf"([\s|\S]*?){proof_context_separator}([\s|\S]*)"
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
    print(run_proof_on_lean(my_proof, lean_cwd="./testbed/src"))