from typing import Optional, List
from dataclasses import dataclass, field
from heapq import heapify, heappop
#from prompt_gpt import prompt_for_tactics, prompt_for_triviality
from prompt_human import prompt_for_tactics, prompt_for_triviality

@dataclass
class ANDNodeInfo:
    '''
    An AND node itself is a child of an OR node, and so represents a proof step.
    '''
    proof_step: str

@dataclass
class ORNodeInfo:
    '''
    An OR node itself is a child of an AND node, and so represents a goal.
    '''
    goal: str

'''
TODO: move this
An node type enum to represent the type of the node in the graph
AND node: all children must be solved for the parent to be solved. In other words, a Proof Plan node
OR node: at least one child must be solved for the parent to be solved. In other words, an Obligation node
'''

@dataclass
class Node:
    parent: 'Node'
    node_info: ANDNodeInfo | ORNodeInfo
    estimation: float
    children: List['Node'] | None
    solved: bool = False
    
    def __post_init__(self):
        self.unsolved_children = self.children.copy()
        heapify(self.unsolved_children) # Always safe: heapify is idempotent

    def __lt__(self, other: 'Node'):
        # For heapq
        return self.estimation < other.estimation

def heuristic(node: Node) -> Node:
    return 1

def init_node(parent: Node, node_info: ANDNodeInfo | ORNodeInfo) -> Node:
    node = Node(parent=parent, node_info=node_info, estimation=heuristic(node_info), children=[], solved=False)
    if parent is not None:
        parent.children.append(node)
    return node

def backtrack(node: Node) -> None:
    """
    Recursively notify parents of updated children.
    """
    assert node.children, "Backtracking on a node without children"
    # The following code assumes children is non-empty.
    # For our implementation we don't need to backtrack on leaf nodes anyways.
    match node.node_info:
        case ANDNodeInfo(_):
            node.estimation = sum(child.estimation for child in node.children)
            node.solved = all(child.solved for child in node.children)
        case ORNodeInfo(_):
            node.estimation = min(child.estimation for child in node.children)
            node.solved = any(child.solved for child in node.children)
        case _:
            raise TypeError(f"Unknown node information type: {type(node.node_info)}")
    if node.parent is not None:
        backtrack(node.parent)

def expand(node: Node, proof_so_far: str) -> None:
    """
    "Work" on the current node.
    """
    assert not node.children, "Expanding a non-leaf node"
    assert isinstance(node.node_info, ORNodeInfo), "The current implementation assumes only OR nodes are `expand()`ed"

    ## Placeholder for prompting the LLM to check if the node is trivially solvable
    #is_trivially_solvable = prompt_for_triviality(node.obligation)
    #if is_trivially_solvable:
    #    node.solved = True
    #    node.estimation = 0
    #    if node.parent:
    #        backtrack(node.parent)
    #else:
    if True: # TODO: the current implementation does need to specially trivial cases, right?
        tactics = prompt_for_tactics(node.node_info.goal)  # Replace with actual LLM generation of tactics
        for tactic in tactics:
            and_node = init_node(parent=node, node_info=ANDNodeInfo(proof_step=tactic))
            for goal in run_lean_results: #TODO: implement run_lean_results
                init_node(parent=and_node, node_info=ORNodeInfo(goal=goal))
        backtrack(node)

def find(node: Node, proof_so_far: str) -> None:
    if not node.children:
        expand(node)
    else:
        print(f"looking into unsolved children for node with {node.obligation=}")
        best_child = heappop(node.unsolved_children)
        match best_child.node_info:
            case ANDNodeInfo(proof_step=s):
                proof_so_for += s
            case ORNodeInfo(_):
                pass
            case _:
                raise TypeError(f"Unknown node information type: {type(node.node_info)}")
        find(best_child, proof_so_far)

def ao_star(goal):
    root = init_node(parent=None, node_info=ORNodeInfo(goal=goal))
    while not root.solved:
        find(root)

if __name__ == "__main__":
    # Test driving code
    task = "[root]"
    ao_star(task)