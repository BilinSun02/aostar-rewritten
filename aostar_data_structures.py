from dataclasses import dataclass, field
from typing import Optional, List, Callable
from lean_cmd_executor_aostar import Obligation
from enum import Enum, auto

@dataclass
class ANDNodeInfo:
    '''
    An AND node itself is a child of an OR node, and so represents a proof step.
    '''
    proof_step: str
    necessary_import: str = field(default_factory=str) # E.g. additional necessary imports for tactics

    def __str__(self) -> str:
        return self.proof_step + " " + self.necessary_import
    
    def __eq__(self, other: 'ANDNodeInfo') -> bool:
        # Considered equal if the proof step is equal
        # For motivation, see comments for Node.__eq__
        return self.proof_step == other.proof_step

@dataclass
class PendingANDNodeInfo: #TODO: the meaning of Pending here is not quite related to NodeState.PENDING; change the naming of either to improve clarity
    '''
    A PendingAND node itself is a child of an OR node.
    Expanding it results in a new AND node that is guaranteed 
    to be distinct from previous ones (and possibly some AND
    nodes that fail to be distinct or fail to compile).
    Expansion of a PendingAND node does NOT eliminate itself--
    it can be expanded in the future to create more AND nodes.
    '''
    distinct_tried_tactics: List[str] = field(default_factory=list)
    avoid_steps_str: str = field(default_factory=str("[AVOID STEPS]\n")) # !! TODO: check if default str should be set this way

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
    parent: Optional['Node']
    node_type: ANDNodeInfo | ORNodeInfo | PendingANDNodeInfo
    heuristic: Callable[['Node'], float]
    children: List['Node'] = field(default_factory=list)
    state: NodeState = NodeState.PENDING
    hide_from_visualization: bool = False
    @property
    def solved(self) -> bool:
        return self.state == NodeState.SOLVED
    @property
    def estimation(self) -> float:
        # The more positive, the more difficult
        # While not hard-coded, it's recommended to ensure this is non-negative
        # and put it at +infty if the search is essentially infeasible
        return self.heuristic(self)
    
    def __post_init__(self):
        self.pending_children = self.children.copy() # To hold PENDING children
        if self.parent is not None:
            self.parent.children.append(self)
            self.parent.pending_children.append(self)

    def __lt__(self, other: 'Node') -> bool:
        # For heapq
        return self.estimation < other.estimation
    
    def __eq__(self, other: 'Node') -> bool:
        # For removing failed nodes from future searches
        # To do that, we look up the failed nodes from the parent's pending_children list
        # so defining equality to be equality of goals/states would suffice
        return self.node_type == other.node_type