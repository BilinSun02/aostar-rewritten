from typing import Optional, List
from dataclasses import dataclass, field
from heapq import heapify, heappop
#from prompt_gpt import prompt_for_tactics, prompt_for_triviality
from prompt_human import prompt_for_tactics, prompt_for_triviality

@dataclass
class Node:
    parent: 'Node'
    nodetype: str
    obligation: str
    estimation: float
    children: List['Node'] | None
    solved: bool = False
    
    def __post_init__(self):
        self.unsolved_children = self.children.copy()
        heapify(self.unsolved_children) # Always safe: heapify is idempotent
        #print(f"heapifying unsolved children for node with {self.obligation=}")

    def __lt__(self, other: 'Node'):
        # For heapq
        return self.estimation < other.estimation

def heuristic(node: Node) -> Node:
    return 1

def init_node(parent: Node, nodetype: Optional[str], obligation: Optional[str]) -> Node:
    node = Node(parent=parent, nodetype=nodetype, obligation=obligation, estimation=heuristic(obligation), children=[], solved=False)
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
    if node.nodetype == "OR":
        node.estimation = min(child.estimation for child in node.children)
        node.solved = any(child.solved for child in node.children)
    else:  # AND node
        node.estimation = sum(child.estimation for child in node.children)
        node.solved = all(child.solved for child in node.children)
    if node.parent is not None:
        backtrack(node.parent)

def expand(node) -> None:
    """
    "Work" on the current node.
    """
    assert not node.children, "Expanding a non-leaf node"
    node.nodetype = "OR"

    # Placeholder for prompting the LLM to check if the node is trivially solvable
    is_trivially_solvable = prompt_for_triviality(node.obligation)
    if is_trivially_solvable:
        node.solved = True
        node.estimation = 0
        if node.parent:
            backtrack(node.parent)
    else:
        # Placeholder for prompting the LLM and obtaining tactics
        tactics = prompt_for_tactics(node.obligation)  # Replace with actual LLM generation of tactics
        for tactic in tactics:
            and_node = init_node(parent=node, nodetype="AND", obligation=', and'.join(tactic))
            for obligation in tactic:
                init_node(parent=and_node, nodetype="OR", obligation=obligation)
        backtrack(node)

def find(node):
    if not node.children:
        expand(node)
    else:
        print(f"looking into unsolved children for node with {node.obligation=}")
        best_child = heappop(node.unsolved_children)
        find(best_child)

def ao_star(task):
    root = init_node(parent=None, nodetype="OR", obligation=task)
    while not root.solved:
        find(root)

if __name__ == "__main__":
    # Test driving code
    task = "[root]"
    ao_star(task)