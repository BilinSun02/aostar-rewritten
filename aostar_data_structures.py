from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, List
from enum import Enum, auto

from lean_cmd_server import Message
from lean_cmd_executor_aostar import Goal
#from prompt_human import prompt_for_tactics


class NodeState(Enum):
    ACTIVE = auto()
    SOLVED = auto()
    FAILED = auto()

class NodeDetailedState(Enum):
    ACTIVE = auto()
    SOLVED = auto()
    FAILED_DUE_TO_CHILDREN = auto() # All children failed or, in the case of an OR node, there's no child
    DOESNT_COMPILE = auto()
    ABANDONED = auto() # Too hard or blantantly wrong
    IS_REPETITIVE = auto()
    NO_PROGRESS = auto()

@dataclass
class Node(ABC):
    parents: Optional['Node'] = field(default_factory=list, init=False)
    # Only OR nodes may have more than 1 parent, an adaptation made to avoid repetitive labor trying to solve the same goal multiple times
    children: List['Node'] = field(default_factory=list, init=False)
    # `parents` and `children` should normally only be changed by `add_child`
    expanded: bool = field(default=False, init=False)
    hide_from_visualization: bool = field(default=False, init=False)
    _detailed_state: NodeDetailedState = field(default=NodeDetailedState.ACTIVE, init=False)
    # `init=False` avoids "non-default argument follows default argument" for derived classes: https://stackoverflow.com/a/58525728

    @property
    def detailed_state(self) -> NodeDetailedState:
        return self._detailed_state

    @detailed_state.setter
    def detailed_state(self, value: NodeDetailedState):
        # To guard against accidentally assigning a NodeState
        assert isinstance(value, NodeDetailedState), f"detailed_state must be an instance of NodeDetailedState, not {type(value)}"
        self._detailed_state = value

    @property
    def state(self) -> NodeState:
        match self.detailed_state:
            case NodeDetailedState.ACTIVE:
                return NodeState.ACTIVE
            case NodeDetailedState.SOLVED:
                return NodeState.SOLVED
            case NodeDetailedState.FAILED_DUE_TO_CHILDREN |\
                 NodeDetailedState.DOESNT_COMPILE |\
                 NodeDetailedState.ABANDONED |\
                 NodeDetailedState.IS_REPETITIVE |\
                 NodeDetailedState.NO_PROGRESS:
                return NodeState.FAILED
            case _:
                raise TypeError(f"Unrecognized node {self.detailed_state=}")

    @property
    def solved(self) -> bool:
        return self.state == NodeState.SOLVED
    
    @property
    def ancestors(self) -> List['Node']:
        if len(self.parents) == 0:
            return []
        return [ancestor for parent in self.parents for ancestor in (parent.ancestors + [parent])]

    @property
    def root(self) -> 'Node':
        if not self.parents:
            return self
        else:
            assert all(parent.root is self.parents[0].root for parent in self.parents), "Something really bad must have happened. There's no unique root in the tree."
            return self.parents[0].root

    def __post_init__(self):
        pass
    
    @abstractmethod
    def __eq__(self, other: 'Node') -> bool:
        return self.parents == other.parents and self.detailed_state == other.detailed_state
        # TODO: reconsider whether we want to compare parents

    def add_child(self, node: 'Node') -> None:
        self.children.append(node)
        node.parents.append(self)

    @property
    def active_children(self) -> List['Node']:
        return [child for child in self.children if child.state == NodeState.ACTIVE]

@dataclass
class ANDNode(Node):
    '''
    An AND node itself is a child of an OR node, and so represents a proof step.
    '''
    proof_step: str # One or more tactics
    necessary_import: str = field(default="") # E.g. additional necessary imports for tactics
    error_messages: List[Message] = field(default_factory=list, init=False) # Nonempty if the tactic failed to compile
    _proof_so_far: str = field(default="", init=False) # Nonempty if the tactic failed to compile

    def __str__(self) -> str:
        return self.proof_step + " " + self.necessary_import

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, ANDNode):
            return False
        # Considered equal if the proof step is equal
        # For motivation, see comments for Node.__eq__
        return Node.__eq__(self, other) and self.proof_step == other.proof_step

@dataclass
class AbandonedANDNode(ANDNode):
    expanded: bool = field(default=True, init=False) # Overrides the default value `False` in `Node`: there's no point to expand it anymore.
    _detailed_state: NodeDetailedState = field(default=NodeDetailedState.ABANDONED, init=False) # Overrides the default value `ACTIVE` in `Node`

@dataclass
class RepetitiveANDNode(ANDNode):
    expanded: bool = field(default=True, init=False) # Overrides the default value `False` in `Node`: there's no point to expand it anymore.
    _detailed_state: NodeDetailedState = field(default=NodeDetailedState.IS_REPETITIVE, init=False) # Overrides the default value `ACTIVE` in `Node`

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

@dataclass
class MERISTEMNode(Node):
    '''
    A MERISTEM node itself is a child of an OR node.
    Expanding it results in a new AND node that is guaranteed 
    to be distinct from previous ones (and possibly some AND
    nodes that fail to be distinct or fail to compile).
    Expansion of a MERISTEM node does NOT eliminate itself--
    it can be expanded in the future to create more AND nodes.
    '''
    hide_from_visualization: bool = field(default=True, init=False) # Overrides the default value `True` in `Node`
    distinct_tried_tactic_import_pairs: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.hide_from_visualization, "MERISTEM nodes should be hidden"

    def __str__(self) -> str:
        assert len(self.parents) == 1, "A MERISTEMNode must be a child of exactly one OR node"
        return f"MERISTEM child of {self.parents[0]}"

    @property
    def parent_OR_node(self) -> 'ORNode':
        assert len(self.parents) == 1, "A MERISTEMNode must be a child of exactly one OR node"
        return self.parents[0]

    @property
    def avoid_steps_str(self) -> str:
        avoid_steps_str = "[AVOID STEPS]\n"
        for peer in self.parent_OR_node.children:
            if isinstance(peer, ANDNode):
                avoid_steps_str += "[STEP]" + peer.proof_step + "\n[ERROR]" # This is technically missing the import part, but close enough
                if peer.error_messages:
                    avoid_steps_str += "\n".join(msg.text for msg in peer.error_messages)
                elif peer.detailed_state == NodeDetailedState.IS_REPETITIVE:
                    avoid_steps_str += "You have repeatedly suggested this tactic. Do NOT suggest it again."
                elif peer.detailed_state == NodeDetailedState.NO_PROGRESS:
                    avoid_steps_str += "This tactic compiles fine but leads to an undesirable goal. " +\
                                       "Usually this means this tactic (or this tactic together with further steps) " +\
                                       "lead to exactly the same goal as where it started, i.e. no progress is made."
                else:
                    avoid_steps_str += "This tactic is known. You are tasked to come up with a novel tactic."
                avoid_steps_str += "[END ERROR]\n"
        return avoid_steps_str

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, MERISTEMNode):
            return False
        return Node.__eq__(self, other)

@dataclass
class ORNode(Node):
    '''
    An OR node itself is a child of an AND node, and so represents a goal.
    '''
    goal: Goal

    def __str__(self) -> str:
        return "("+") (".join(self.goal.hypotheses) + ") : (" + self.goal.inference + ")"

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, ORNode):
            return False
        # Considered equal if hypotheses and goals are equal
        # For motivation, see comments for Node.__eq__
        return Node.__eq__(self, other) and str(self) == str(other)