from typing import Optional, List
from dataclasses import dataclass, field
from heapq import heapify, heappop
from prompt_gpt import prompt_for_tactics
#from prompt_human import prompt_for_tactics
from lean_cmd_executor import Obligation
from lean_cmd_executor_aostar import run_proof_on_lean
from enum import Enum, auto

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='aostar.log', encoding='utf-8', level=logging.DEBUG)

@dataclass
class ANDNodeInfo:
    '''
    An AND node itself is a child of an OR node, and so represents a proof step.
    '''
    proof_step: str

    def __str__(self) -> str:
        return self.proof_step
    
    def __eq__(self, other: 'ANDNodeInfo') -> bool:
        # Considered equal if the proof step is equal
        # For motivation, see comments for Node.__eq__
        return self.proof_step == other.proof_step

@dataclass
class ORNodeInfo:
    '''
    An OR node itself is a child of an AND node, and so represents a goal.
    '''
    goal: Obligation # TODO: I want to deprecate the word "obligation", but it is still used in existing codebase

    def __str__(self) -> str:
        return "("+") (".join(self.goal.hypotheses) + ") : (" + self.goal.goal + ")"

    def __eq__(self, other: 'ORNodeInfo') -> bool:
        # Considered equal if hypotheses and goals are equal
        # For motivation, see comments for Node.__eq__
        return str(self) == str(other)

class NodeState(Enum):
    PENDING = auto()
    SOLVED = auto()
    FAILED = auto()

@dataclass
class Node:
    parent: 'Node'
    node_info: ANDNodeInfo | ORNodeInfo
    estimation: float # The more positive, the more costly
    children: List['Node'] | None
    state: NodeState = NodeState.PENDING
    @property
    def solved(self) -> bool:
        return self.state == NodeState.SOLVED
    
    def __post_init__(self):
        self.pending_children = self.children.copy() # Children that are not solved or failed

    def __lt__(self, other: 'Node') -> bool:
        # For heapq
        return self.estimation < other.estimation
    
    def __eq__(self, other: 'Node') -> bool:
        # For removing failed nodes from future searches
        # To do that, we look up the failed nodes from the parent's pending_children list
        # so defining equality to be equality of goals/states would suffice
        return self.node_info == other.node_info

def heuristic(node: Node) -> Node:
    return 1

def init_node(parent: Node, node_info: ANDNodeInfo | ORNodeInfo) -> Node:
    node = Node(parent=parent, node_info=node_info, estimation=heuristic(node_info), children=[])
    if parent is not None:
        parent.children.append(node)
        parent.pending_children.append(node)
    return node

def backtrack(node: Node) -> None:
    """
    Recursively notify parents of updated children.
    """
    assert node.children, "Backtracking on a node without children"
    #print(f"Backtracking on node {node=}, which was {node.solved=} before backtracking")

    match node.node_info:
        case ANDNodeInfo(_):
            node.estimation = sum(child.estimation for child in node.children)
            if any(child.state == NodeState.FAILED for child in node.children):
                node.state = NodeState.FAILED
            elif all(child.state == NodeState.SOLVED for child in node.children):
                node.state = NodeState.SOLVED
        case ORNodeInfo(_):
            node.estimation = min(child.estimation for child in node.children)
            if all(child.state == NodeState.FAILED for child in node.children):
                node.state = NodeState.FAILED
            elif any(child.state == NodeState.SOLVED for child in node.children):
                node.state = NodeState.SOLVED
        case _:
            raise TypeError(f"Unknown node information type: {type(node.node_info)}")
    if node.parent is not None:
        if node.state != NodeState.PENDING: # Recently solved or failed
            node.parent.pending_children = [peer for peer in node.parent.pending_children if peer != node]
        backtrack(node.parent)

def expand(node: Node, proof_so_far: str) -> None:
    """
    "Work" on the current node.
    """
    assert not node.children, "Expanding a non-leaf node"
    assert isinstance(node.node_info, ORNodeInfo), "The current implementation assumes only OR nodes are `expand()`ed"

    ## Now format the proof_context into what we need for prompting the LLM
    #goal_statement = "[GOALS]\n"
    #idx = 1
    #for goal in proof_context.fg_goals:
    #    goal_statement += goal.format_message(idx)
    #    idx += 1
    #return goal_statement 
    tactics = prompt_for_tactics("[GOALS]\n" + node.node_info.goal.format_message(1))
    for tactic in tactics:
        and_node = init_node(parent=node, node_info=ANDNodeInfo(proof_step=tactic))
        run_lean_proof_context, run_lean_messages = run_proof_on_lean(proof_so_far + standardize_indentation(tactic) + "\nend") #TODO: this assumes indentation for tactic is 4
        if any(msg.level == 'error' for msg in run_lean_messages): # Presumably Lean syntactic errors
            and_node.state = NodeState.FAILED
            and_node.estimation = float("inf")
        elif run_lean_proof_context: # No syntactic errors, and goal(s) remain to prove
            #print(f"{run_lean_results.fg_goals=}")
            for obligation in run_lean_proof_context.fg_goals: # TODO: I want to deprecate the word "obligation", but it is still used in existing codebase
                init_node(parent=and_node, node_info=ORNodeInfo(goal=obligation))
                # The estimation for each such OR node will be inited to 1
            and_node.estimation = len(run_lean_proof_context.fg_goals) # TODO: This needs to be changed when we change heuristics
        else: # No syntactic errors and no goals to prove
            node.state = NodeState.SOLVED
            node.estimation = 0
    backtrack(node)

def standardize_indentation(string:str, indent_amount:int=4) -> str:
    lines = string.split('\n')
    standardized_lines = []
    for line in lines:
        if line.strip():
            standardized_line = ' ' * indent_amount + line.lstrip()
            standardized_lines.append(standardized_line)
        else:
            standardized_lines.append(line)
    standardized_string = '\n'.join(standardized_lines)
    return standardized_string

def find(node: Node, proof_so_far: str) -> None:
    #print(f"find() visits {str(node.node_info)=}")
    if not node.children:
        expand(node, proof_so_far)
    else:
        match node.node_info:
            case ANDNodeInfo(proof_step=s):
                print(f"{node.parent=}")
                if node.parent is not None:
                    # Unless the current node is the root,
                    # the tactic here needs to be indented
                    # TODO: check my assumption that all lines are indented by 4
                    s = standardize_indentation(s, 4)
                    s += "\n" # for good measure
                proof_so_far += s
            case ORNodeInfo(_):
                pass
            case _:
                raise TypeError(f"Unknown node information type: {type(node.node_info)}")
        best_child = min(node.pending_children)
        find(best_child, proof_so_far)

def ao_star(theorem_statement: str) -> None: # TODO: the naming of some variables assume the thing to prove is a "theorem", but in practice it may well be an example etc.
    theorem_statement += "\nbegin\n"
    root = init_node(parent=None, node_info=ANDNodeInfo(proof_step=theorem_statement))
    root_proof_context, root_lean_messages = run_proof_on_lean(theorem_statement + "end")
    assert all(not msg.level == 'error' for msg in root_lean_messages), f"Problems in the theorem statement:\n{root_lean_messages}"
    root_goals = root_proof_context.fg_goals # If the assert above fails, this can't even work as root_proof_context would be `None`
    # root_goals should be a singleton: Lean just gets the theorem statement so the one goal is the conclusion of the theorem
    assert len(root_goals) == 1, "It's unexpected that the theorem statement already begets not exactly one goal." # Let me know if my assumption is wrong
    root_goal = root_goals[0]
    root_sole_child = init_node(parent=root, node_info=ORNodeInfo(goal = root_goal))
    while root.state == NodeState.PENDING:
        find(root, "")
    match root.state:
        case NodeState.SOLVED:
            print("Proof search successful.")
        case NodeState.FAILED:
            print("Proof search unsuccessful.")
        case _:
            raise ValueError("Proof search ended with an unexpected state")
    present_solution(root)

# See https://stackoverflow.com/a/287944
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def present_solution(node: Node, prefix: str = '', is_last: bool = True):
    # Prints the search tree in a format similar to the Linux `tree` command.
    # Parts that are used towards the final solution are in green or blue or boldface
    # others, black and non-bold
    boldmarker = bcolors.BOLD
    match node.node_info:
        case ANDNodeInfo(_):
            colormarker = bcolors.OKGREEN
        case ORNodeInfo(_):
            colormarker = bcolors.OKBLUE
    endmarker = bcolors.ENDC

    connector = '└── ' if is_last else '├── '
    if node.solved:
        print(prefix + colormarker + connector + endmarker
              + boldmarker + str(node.node_info) + endmarker)
        new_prefix = prefix + colormarker + ('    ' if is_last else '│   ') + endmarker
    else:
        print(prefix + connector + str(node.node_info))
        new_prefix = prefix + ('    ' if is_last else '│   ')

    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        present_solution(child, new_prefix, is_last_child)

if __name__ == "__main__":
    # Test driving code
    task = "theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :="
    ao_star(task)