from dataclasses import dataclass, field
from typing import Optional, List
from lean_cmd_executor_aostar import Obligation, run_proof_on_lean
from enum import Enum, auto
from abc import ABC, abstractmethod
from logging import Logger
from prompt_gpt import prompt_for_tactics
#from prompt_human import prompt_for_tactics


class NodeState(Enum):
    UNSOLVED = auto()
    SOLVED = auto()
    FAILED = auto()

@dataclass
class Node(ABC):
    parent: Optional['Node']
    children: List['Node']# = field(default_factory=list) # Commented out to avoid "non-default argument follows default argument" for derived classes
    state: NodeState# = NodeState.UNSOLVED
    expanded: bool# = False
    hide_from_visualization: bool# = False

    @property
    def solved(self) -> bool:
        return self.state == NodeState.SOLVED
    
    def __post_init__(self):
        self.unsolved_children = self.children.copy() # To hold UNSOLVED children
        if self.parent is not None:
            self.parent.children.append(self)
            if self.state == NodeState.UNSOLVED:
                self.parent.unsolved_children.append(self)
    
    @abstractmethod
    def __eq__(self, other: 'Node') -> bool:
        # For when we're removing solved or failed nodes from future searches.
        # To remove a node we need to first look up the node unsolved_children list
        # so we just define equality to be equality of goals/states.
        pass

    @abstractmethod
    def expand(self, proof_so_far: str, logger: Logger) -> None:
        pass

    def remove_child(self, to_remove: 'Node') -> None:
        self.unsolved_children = [child for child in self.unsolved_children if child != to_remove]

@dataclass
class ANDNode(Node):
    '''
    An AND node itself is a child of an OR node, and so represents a proof step.
    '''
    proof_step: str
    necessary_import: str = field(default_factory=str) # E.g. additional necessary imports for tactics

    def __str__(self) -> str:
        return self.proof_step + " " + self.necessary_import

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, ANDNode):
            return False
        # Considered equal if the proof step is equal
        # For motivation, see comments for Node.__eq__
        return self.proof_step == other.proof_step

    def expand(self, proof_so_far: str, logger: Logger) -> None:
        self.expanded = True
        # For now, every AND is guaranteed to have been
        # inited earlier (see the MERISTEMNodeInfo case)

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
    distinct_tried_tactics: List[str] = field(default_factory=list)
    avoid_steps_str: str = field(default = "[AVOID STEPS]\n")

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.hide_from_visualization, "MERISTEM nodes should be hidden"

    def __str__(self) -> str:
        return f"MERISTEM child of {self.parent}"

    def expand(self, proof_so_far: str, logger: Logger) -> None:
        assert not self.expanded, f"Node {self} has already been expanded"
        # We do NOT mark MERISTEM nodes as expanded

        assert self.parent is not None, "A MERISTEMNode must be a child under an OR node"
        parent_OR_node = self.parent

        message = "[GOALS]\n" + parent_OR_node.goal.format_message(1) # TODO: support it when there is more than one goal
        print_friendly_avoid_steps_str = str(self.avoid_steps_str).replace('\n', '\\n')
        logger.info(f"Prompting for tactics with {message=} with cautions {print_friendly_avoid_steps_str}")
        tactics_import_pairs_to_try = prompt_for_tactics(message, avoid_steps=self.avoid_steps_str, n_tactics=1)
        for tactic, necessary_import in tactics_import_pairs_to_try:
            # Create an AND node even before checking if it's repetitive or not compiling,
            # so we can see at the end how many times the LLM repeats itself
            and_node = ANDNode(
                parent = parent_OR_node,
                children = [],
                state = NodeState.UNSOLVED,
                expanded = False,
                hide_from_visualization = False,
                proof_step = tactic,
                necessary_import = necessary_import
            )
            tactics = self.distinct_tried_tactics
            if tactic in tactics:
                logger.warning(f"The LLM repeatedly produces {tactic=} despite instructions not to do so.")
                and_node.state = NodeState.FAILED # TODO: maybe make this a KILLED state?
                parent_OR_node.remove_child(and_node)
                self.avoid_steps_str += "[STEP]" + tactic + "\n[ERROR]"
                self.avoid_steps_str += "This tactic has been repeatedly suggested. Be careful not to suggest it again.\n"
                self.avoid_steps_str += "[END ERROR]"
                continue

            run_lean_proof_context, run_lean_messages = run_proof_on_lean(necessary_import + "\n" + proof_so_far + standardize_indentation(tactic) + "\nend") # TODO: this assumes indentation for tactic is 2
            logger.debug(f"Running {tactic=} returns\n"
                        +f"{run_lean_messages=} and\n"
                        +f"{run_lean_proof_context=}")
            logger.debug(f"After running {tactic=}, {run_lean_proof_context.fg_goals=}")
            if any(msg.level == 'error' for msg in run_lean_messages): # Presumably Lean syntactic errors
                logger.info(f"The LLM suggested {tactic=} which failed to compile.")
                # Let the LLM avoid it
                tactics.append(tactic)
                #self.distinct_tried_tactics = tactics # Not necessary; lists are mutable
                self.avoid_steps_str += "[STEP]" + tactic + "\n[ERROR]"
                self.avoid_steps_str += "\n".join(("Error:" + msg.text) for msg in run_lean_messages if msg.level == 'error')
                self.avoid_steps_str += "[END ERROR]"
                and_node.state = NodeState.FAILED
                parent_OR_node.remove_child(and_node)
                continue

            # If we reach here, we have a distinct new tactic that compiles
            logger.info(f"The LLM suggested {tactic=} which is adopted.")

            # Avoid the same tactic in the future
            tactics.append(tactic)
            #self.distinct_tried_tactics = tactics # Not necessary; lists are mutable
            self.avoid_steps_str += "[STEP]" + tactic + "\n[ERROR]"
            self.avoid_steps_str += "This tactic has been suggested by others. You should come up with a novel tactic.\n"
            self.avoid_steps_str += "[END ERROR]"
            if not run_lean_proof_context: # If this list is empty, we have no goals to prove; we are done
                and_node.state = NodeState.SOLVED
            else:
                for obligation in run_lean_proof_context.fg_goals: # TODO: I want to deprecate the word "obligation", but it is still used in existing codebase
                    ORNode(
                        parent = and_node,
                        children = [],
                        state = NodeState.UNSOLVED,
                        expanded = False,
                        hide_from_visualization = False,
                        goal = obligation
                    )
                    # We just initialize it without needing to use (or bind a name) to it yet.
                    # We don't worry about this being collected as garbage since the parent's
                    # children attribute will have a reference to it, as per how Node.__init__() is defined.

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, MERISTEMNode):
            return False
        return self.parent == other.parent

@dataclass
class ORNode(Node):
    '''
    An OR node itself is a child of an AND node, and so represents a goal.
    '''
    goal: Obligation # TODO: I want to deprecate the word "obligation", but it is still used in existing codebase

    def __str__(self) -> str:
        return "("+") (".join(self.goal.hypotheses) + ") : (" + self.goal.inference + ")"

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, ORNode):
            return False
        # Considered equal if hypotheses and goals are equal
        # For motivation, see comments for Node.__eq__
        return str(self) == str(other)

    def expand(self, proof_so_far: str, logger: Logger) -> None:
        assert not self.expanded, f"Node {self} has already been expanded"
        self.expanded = True
        MERISTEMNode(
            parent = self,
            children = [],
            state = NodeState.UNSOLVED,
            expanded = False,
            hide_from_visualization = True,
            distinct_tried_tactics = [],
            avoid_steps_str = "[AVOID STEPS]\n"
        )