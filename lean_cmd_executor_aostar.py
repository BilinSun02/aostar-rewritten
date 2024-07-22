from lean_cmd_server import LeanCmdServer
import typing
import re
import os

# Adapted from lean_cmd_executor.py
class Obligation(typing.NamedTuple):
    hypotheses: typing.List[str]
    inference: str

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {"hypotheses": self.hypotheses,
                "inference": self.inference}
    
    def format_message(self, goal_idx: int) -> str:
        message_str = f"[GOAL] {goal_idx}\n" + self.inference + "\n[HYPOTHESES]\n"
        for hypothesis in self.hypotheses:
            message_str += "[HYPOTHESIS] " + hypothesis
        return message_str

def run_proof_on_lean(proof: str, max_memory_in_mib:int=4000, project_root:str="."): # max_memory_in_mib was 40000 in lean_cmd_executor
    temp_file_name = os.path.join(project_root, "src/temp_proof.lean")
    #temp_file_name = "data/test/lean_proj2/temp_proof.lean"
    with open(temp_file_name, "w") as f:
        f.write(proof)
    lean_server = LeanCmdServer(memory_in_mibs=max_memory_in_mib, cwd=project_root, debug=False)
    response = lean_server.run(temp_file_name, 60)
    return (parse_proof_context_human_readable(response.state), response.messages)

# Adapted from lean_cmd_executor.py
#TODO: either make better use of the other attributes of this class
# or just eliminate this class
class ProofContext(typing.NamedTuple):
    fg_goals: typing.List[Obligation]
    bg_goals: typing.List[Obligation]
    shelved_goals: typing.List[Obligation]
    given_up_goals: typing.List[Obligation]

    @classmethod
    def empty(cls: typing.Type['ProofContext']):
        return ProofContext([], [], [], [])

    @classmethod
    def from_dict(cls, data):
        fg_goals = list(map(Obligation.from_dict, data["fg_goals"]))
        bg_goals = list(map(Obligation.from_dict, data["bg_goals"]))
        shelved_goals = list(map(Obligation.from_dict, data["shelved_goals"]))
        given_up_goals = list(map(Obligation.from_dict,
                                  data["given_up_goals"]))
        return cls(fg_goals, bg_goals, shelved_goals, given_up_goals)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {"fg_goals": list(map(Obligation.to_dict, self.fg_goals)),
                "bg_goals": list(map(Obligation.to_dict, self.bg_goals)),
                "shelved_goals": list(map(Obligation.to_dict,
                                          self.shelved_goals)),
                "given_up_goals": list(map(Obligation.to_dict,
                                           self.given_up_goals))}

    @property
    def all_goals(self) -> typing.List[Obligation]:
        return self.fg_goals + self.bg_goals + \
            self.shelved_goals + self.given_up_goals

    @property
    def focused_goal(self) -> str:
        if self.fg_goals:
            return self.fg_goals[0].inference
        else:
            return ""

    @property
    def focused_hyps(self) -> typing.List[str]:
        if self.fg_goals:
            return self.fg_goals[0].hypotheses
        else:
            return []

# Adapted from lean_cmd_executor.py
proof_context_separator = "⊢"
proof_context_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
def parse_proof_context_human_readable(proof_context_str: str) -> ProofContext:
    if proof_context_str is None or len(proof_context_str) == 0 or proof_context_separator not in proof_context_str:
        return None
    if proof_context_str == "no goals":
        return ProofContext.empty()
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
    print(run_proof_on_lean(my_proof, project_root="./testbed/src"))