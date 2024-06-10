from typing import Optional, List
from dataclasses import dataclass
from heapq import heapify
from prompt_gpt import prompt_for_tactics, prompt_for_triviality

@dataclass
class Node:
    parent: Optional['Node'] = None
    nodetype: Optional[str]
    obligation: Optional[str]
    estimation: Optional[float]
    children: List['Node']
    solved: bool = False
    
    def __post_init__(self):
        if  self.nodetype == "OR":
            self.estimation = heuristic(self)
        heapify(self.children) # Always safe: heapify is idempotent

    def __lt__(self, other):
        # For heapq
        return self.estimation < other.estimation

def heuristic(node: Node) -> Node:
    return 1

def init_node(parent: Node, nodetype: Optional[str], obligation: Optional[str]) -> Node:
    node = Node(parent=parent, nodetype=nodetype, obligation=obligation)
    if parent is not None:
        parent.children.append(node)
    return node

def backtrack(node: Node) -> Node:
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

def expand(node):
    assert not node.children, "Expanding a non-leaf node"
    node.nodetype = "OR"

    # Placeholder for prompting the LLM to check if the node is trivially solvable
    is_trivially_solvable = prompt_for_triviality(node.obligation)
    if is_trivially_solvable:
        node.solved = True
        node.estimation = 0
        if node.parent is not None:
            backtrack(node.parent)
    else:
        # Placeholder for prompting the LLM and obtaining tactics
        tactics = []  # Replace with actual LLM generation of tactics
        for tactic in tactics:
            and_node = Node()
            init_node(and_node, parent=node, nodetype="AND", obligation=tactic)
            for obligation in tactic:
                or_node = Node()
                init_node(or_node, parent=and_node, nodetype="OR", obligation=obligation)
        backtrack(node)

def find(node):
    if not node.children:
        expand(node)
    else:
        best_child = None
        best_estimate = float('inf')
        for child in node.children:
            if child.estimation < best_estimate:
                best_child = child
                best_estimate = child.estimation
        find(best_child)

def ao_star(task):
    root = init_node(parent=None, nodetype="OR", obligation=task)
    while not root.solved:
        find(root)

if __name__ == "__main__":
    # Test driving code
    task = "a"
    ao_star(task)