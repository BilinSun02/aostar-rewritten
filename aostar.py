from typing import Optional, List
from dataclasses import dataclass, field
from heapq import heapify, heappop
from prompt_gpt import prompt_for_tactics
#from prompt_human import prompt_for_tactics
from lean_cmd_executor import Obligation
from lean_cmd_executor_aostar import run_proof_on_lean
from enum import Enum, auto
import datetime
import os
import argparse
import logging

# The root for the Lean project
project_root = os.path.join(os.getcwd(), "testbed")


@dataclass
class ANDNodeInfo:
    '''
    An AND node itself is a child of an OR node, and so represents a proof step.
    '''
    proof_step: str
    supplementary_info: str = field(default_factory=str) # E.g. additional necessary imports for tactics

    def __str__(self) -> str:
        return self.proof_step + " " + self.supplementary_info
    
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
        return "("+") (".join(self.goal.hypotheses) + ") : (" + self.goal.inference + ")"

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
    children: List['Node']
    state: NodeState = NodeState.PENDING
    estimation: float = None # The more positive, the more costly
    @property
    def solved(self) -> bool:
        return self.state == NodeState.SOLVED
    
    def __post_init__(self):
        self.pending_children = self.children.copy() # To hold PENDING and FAILED children
        if not self.estimation:
            self.estimation = heuristic(self)
        if self.parent is not None:
            self.parent.children.append(node)
            self.parent.pending_children.append(node)

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
    return node

def backtrack(node: Node) -> None:
    """
    Recursively notify parents of updated children.
    """
    assert node.children, "Backtracking on a node without children"
    self.logger.debug(f"Backtracking on node {node=}, which was {node.solved=} before backtracking")

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
    message = "[GOALS]\n" + node.node_info.goal.format_message(1) # TODO: support it when there is more than one goal
    self.logger.info(f"Prompting for tactics with {message=}")

    tactics = []
    avoid_steps = "[AVOID STEPS]\n"
    class Done(Exception): # Workaround to break nested loops. Alas, I want goto back
        pass
    try:
        while len(tactics) < 3: # TODO: this was [EXPAND NUM]. Make this customizable.
            tactics_import_pairs_to_try = prompt_for_tactics(message, avoid_steps=avoid_steps, n_tactics=1)
            for tactic, necessary_import in tactics_import_pairs_to_try:
                if tactic in tactics:
                    self.logger.warning("The LLM keeps producing the same tactic despite instructions not to do so.")
                    continue
                run_lean_proof_context, run_lean_messages = run_proof_on_lean(necessary_import + "\n" + proof_so_far + standardize_indentation(tactic) + "\nend", project_root=project_root) #TODO: this assumes indentation for tactic is 4
                self.logger.debug(f"Running {tactic=} returns\n"
                            +f"{run_lean_messages=} and\n"
                            +f"{run_lean_proof_context=}")
                and_node = Node(parent=node, node_info=ANDNodeInfo(proof_step=tactic, supplementary_info=necessary_import), children=[])
                if any(msg.level == 'error' for msg in run_lean_messages): # Presumably Lean syntactic errors
                    # Let the LLM avoid it
                    avoid_steps += "[STEP]" + tactic + "\n[ERROR]"
                    avoid_steps += "\n".join(("Error:" + msg.text) for msg in run_lean_messages if msg.level == 'error')
                    avoid_steps += "[END ERROR]"
                    and_node.state = NodeState.FAILED
                    and_node.estimation = float("inf")
                else: # No syntactic errors
                    avoid_steps += "[STEP]" + tactic + "\n[ERROR]"
                    avoid_steps += "This tactic has been suggested by others. You should come up with a novel tactic.\n"
                    avoid_steps += "[END ERROR]"
                    tactics.append(tactic)
                    if not run_lean_proof_context: # If this list is empty, we have no goals to prove; we are done
                        and_node.state = NodeState.SOLVED
                        and_node.estimation = 0
                        raise Done("As we assume the current node is an OR, we're done") 
                    else:
                        self.logger.debug(f"After running {tactic=}, {run_lean_proof_context.fg_goals=}")
                        for obligation in run_lean_proof_context.fg_goals: # TODO: I want to deprecate the word "obligation", but it is still used in existing codebase
                            Node(parent=and_node, node_info=ORNodeInfo(goal=obligation))
                            # We just initialize it without needing to use (or bind a name) to it yet.
                            # We don't worry about this being collected as garbage since the parent's
                            # children attribute will have a reference to it, as per how Node.__init__() is defined.
                        and_node.estimation = len(run_lean_proof_context.fg_goals) # TODO: This needs to be changed when we change heuristics
    except Done:
        pass

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
    self.logger.debug(f"find() visits the node {str(node.node_info)=}")
    if not node.children:
        expand(node, proof_so_far)
    else:
        match node.node_info:
            case ANDNodeInfo(proof_step=s):
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

def ao_star(theorem_statement: str, logger: logging.Logger) -> None: # TODO: the naming of some variables assume the thing to prove is a "theorem", but in practice it may well be an example etc.
    self.logger = logger # This will then be used by other functions in aostar.py.

    theorem_statement += "\nbegin\n"
    root = init_node(parent=None, node_info=ANDNodeInfo(proof_step=theorem_statement))
    root_proof_context, root_lean_messages = run_proof_on_lean(theorem_statement + "end", project_root=project_root)
    assert all(not msg.level == 'error' for msg in root_lean_messages), f"Problems in the theorem statement:\n{root_lean_messages}"
    root_goals = root_proof_context.fg_goals # If the assert above fails, this can't even work as root_proof_context would be `None`
    # root_goals should be a singleton: Lean just gets the theorem statement so the one goal is the conclusion of the theorem
    assert len(root_goals) == 1, "It's unexpected that the theorem statement already begets not exactly one goal." # Let me know if my assumption is wrong
    root_goal = root_goals[0]
    root_sole_child = init_node(parent=root, node_info=ORNodeInfo(goal = root_goal))

    self.logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search started.')
    try:
        while root.state == NodeState.PENDING:
            find(root, "")
    except KeyboardInterrupt:
        self.logger.info("Proof search interrupted by user.")
    except BaseException as e:
        self.logger.error(repr(e))
    # Whether or not we had an exception, go on to print the proof search tree

    match root.state:
        case NodeState.SOLVED:
            self.logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search successful.')
        case NodeState.FAILED:
            self.logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search unsuccessful.')
        case NodeState.PENDING:
            self.logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search did not finish.')
        case _:
            self.logger.error(f"Proof search ended with an unexpected state {root.state=}")
    self.logger.info("Showing solved nodes:\n" + present_solution(root))
    self.logger.info("The above may include ANSI escape codes for colors. Make sure to use a compatible terminal emulator or editor.")
    self.logger.info(f"Proof search incurred {prompt_for_tactics.gpt_token_counter} tokens in total.")

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

def present_solution(node: Node, prefix: str = '', is_last: bool = True) -> str:
    # Prints the search tree in a format similar to the Linux `tree` command.
    # Parts that are solved are in green or blue or boldface
    # Others, black and non-bold
    solution_str = prefix
    next_prefix  = prefix

    boldmarker = bcolors.BOLD
    match node.node_info:
        case ANDNodeInfo(_):
            colormarker = bcolors.OKGREEN
        case ORNodeInfo(_):
            colormarker = bcolors.OKBLUE
    endmarker = bcolors.ENDC

    connector = '└── ' if is_last else '├── '
    if not node.solved:
        solution_str += connector
        solution_str += str(node.node_info).replace("\n", "\\n")
        next_prefix  += ('    ' if is_last else '│   ')
    else:
        solution_str += colormarker + connector                                + endmarker
        solution_str += boldmarker  + str(node.node_info).replace("\n", "\\n") + endmarker
        next_prefix  += colormarker + ('    ' if is_last else '│   ')          + endmarker
    solution_str += '\n'

    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        solution_str += present_solution(child, next_prefix, is_last_child)
    
    return solution_str

if __name__ == "__main__":
    # Test driving code
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str, default='logs/proof_search.log')
    args = parser.parse_args()
    log_path = args.log_path
    global_logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG, filemode="w")

    #task = "theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :="
    task = "theorem inequality_chain (a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :="
#    task = """import data.nat.basic
#theorem amc12a_2015_p10
#  (x y : ℤ)
#  (h₀ : 0 < y)
#  (h₁ : y < x)
#  (h₂ : x + y + (x * y) = 80) :
#  x = 26 :=
#"""
    ao_star(task, global_logger)