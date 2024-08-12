from dataclasses import dataclass, field
from typing import Optional, List
from lean_cmd_executor_aostar import Obligation, run_proof_on_lean
from enum import Enum, auto
from abc import ABC, abstractmethod
from logging import Logger
from prompt_gpt import prompt_for_tactics
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

@dataclass
class Node(ABC):
    parent: Optional['Node']
    children: List['Node']# = field(default_factory=list)
    expanded: bool# = False
    hide_from_visualization: bool# = False
    # Defaults commented out to avoid "non-default argument follows default argument" for derived classes
    _detailed_state: NodeDetailedState = field(default=NodeDetailedState.ACTIVE, init=False)
    # When field with default is excluded from init though, it's fine: https://stackoverflow.com/a/58525728

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
                 NodeDetailedState.IS_REPETITIVE:
                return NodeState.FAILED
            case _:
                raise TypeError(f"Unrecognized node {self.detailed_state=}")

    @property
    def solved(self) -> bool:
        return self.state == NodeState.SOLVED

    def __post_init__(self):
        self.unsolved_children = self.children.copy() # To hold ACTIVE children
        if self.parent is not None:
            self.parent.children.append(self)
            if self.state == NodeState.ACTIVE:
                self.parent.unsolved_children.append(self)
    
    @abstractmethod
    def __eq__(self, other: 'Node') -> bool:
        # E.g. for when we're removing solved or failed nodes from future searches.
        # To remove a node we need to first look up the node unsolved_children list
        # which uses __eq__().
        return self.parent == other.parent and self.detailed_state == other.detailed_state

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
        return Node.__eq__(self, other) and self.proof_step == other.proof_step

    def expand(self, proof_so_far: str, logger: Logger) -> None:
        self.expanded = True
        # For now, every AND is guaranteed to have been
        # inited earlier (see MERISTEMNode.expand())

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
    distinct_tried_tactic_import_pairs: List[str] = field(default_factory=list)
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

        message = "[GOALS]\n" + parent_OR_node.goal.format_message() # TODO: this now only includes the immediate parent's goal. There may be other goals floating around. Consider also including those so the LLM has better contexts.
        print_friendly_avoid_steps_str = str(self.avoid_steps_str).replace('\n', '\\n')
        logger.info(f"Prompting for tactics with {message=} with cautions {print_friendly_avoid_steps_str}")
        tactics_import_pairs_to_try = prompt_for_tactics(message, avoid_steps=self.avoid_steps_str, n_tactics=1)
        for tactic, necessary_import in tactics_import_pairs_to_try:
            # Create an AND node even before checking if it's repetitive or not compiling,
            # so we can see at the end how many times the LLM repeats itself
            and_node = ANDNode(
                parent = parent_OR_node,
                children = [],
                expanded = False,
                hide_from_visualization = False,
                proof_step = tactic,
                necessary_import = necessary_import
            )
            if "sorry" in tactic:
                # TODO: This hardcodes "sorry" to mean "the goal was abandoned". Un-hardcode this in the future if we need to use "sorry" in the future.
                and_node.expanded = True
                and_node.detailed_state = NodeDetailedState.ABANDONED
                self.expanded = True
                self.detailed_state = NodeDetailedState.ABANDONED
            tactic_imports = self.distinct_tried_tactic_import_pairs
            if (tactic, necessary_import) in tactic_imports:
                logger.warning(f"The LLM repeatedly produces {tactic=} despite instructions not to do so.")
                and_node.expanded = True
                and_node.detailed_state = NodeDetailedState.IS_REPETITIVE
                parent_OR_node.remove_child(and_node)
                #self.avoid_steps_str += "[STEP]" + tactic + "\n[ERROR]"
                #self.avoid_steps_str += "This tactic has been repeatedly suggested. Be careful not to suggest it again.\n"
                #self.avoid_steps_str += "[END ERROR]"
                # TODO: This will pile up EVERY time the same tactic is suggested, making the prompt prohibitingly long.
                # We better use better prompting tricks or higher temperature to get around this.
                continue

            proof_to_run = necessary_import + "\n" + proof_so_far + standardize_indentation(tactic) + "\nend"
            run_lean_proof_context, run_lean_messages = run_proof_on_lean(proof_to_run) # TODO: this assumes indentation for tactic is 2
            logger.debug(f"Running {tactic=} returns\n"
                        +f"{run_lean_messages=} and\n"
                        +f"{run_lean_proof_context=}")
            logger.debug(f"After running {tactic=}, {run_lean_proof_context.fg_goals=}")
            error_msgs = [msg for msg in run_lean_messages if msg.level == 'error']
            if len(error_msgs) > 0: # Presumably Lean syntactic errors
                logger.info(
                    f"The LLM suggested {tactic=} which failed to compile. Error messages:\n" +
                    "\n".join(msg.text for msg in error_msgs) + '\n' +
                    "Full proof:\n" + proof_to_run + '\n'
                )
                # Let the LLM avoid it
                tactic_imports.append((tactic, necessary_import))
                #self.distinct_tried_tactics = tactic_imports # Not necessary; lists are mutable
                self.avoid_steps_str += "[STEP]" + tactic + "\n[ERROR]"
                self.avoid_steps_str += "\n".join(("Error:" + msg.text) for msg in run_lean_messages if msg.level == 'error')
                self.avoid_steps_str += "[END ERROR]"
                and_node.expanded = True
                and_node.detailed_state = NodeDetailedState.DOESNT_COMPILE
                parent_OR_node.remove_child(and_node)
                continue

            # If we reach here, we have a distinct new tactic that compiles
            logger.info(f"The LLM suggested {tactic=} which is adopted.")

            # Avoid the same tactic in the future
            tactic_imports.append((tactic, necessary_import))
            #self.distinct_tried_tactics = tactic_imports # Not necessary; lists are mutable
            self.avoid_steps_str += "[STEP]" + tactic + "\n[ERROR]"
            self.avoid_steps_str += "This tactic has been suggested by others. You are taked to come up with a novel tactic.\n"
            self.avoid_steps_str += "[END ERROR]"
            # TODO: maintain a dict using tactic_import pairs as indices and avoid_step_str as key,
            # and dynamically build avoid_steps_str from that.
            # The current implementation simply accumulates this str and it's unfeasible to change past contents.
            if not run_lean_proof_context: # If this list is empty, we have no goals to prove; we are done
                and_node.detailed_state = NodeDetailedState.SOLVED
            else:
                for obligation in run_lean_proof_context.fg_goals: # TODO: I want to deprecate the word "obligation", but it is still used in existing codebase
                    ORNode(
                        parent = and_node,
                        children = [],
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
        return Node.__eq__(self, other)

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
        return Node.__eq__(self, other) and str(self) == str(other)

    def expand(self, proof_so_far: str, logger: Logger) -> None:
        assert not self.expanded, f"Node {self} has already been expanded"
        self.expanded = True
        MERISTEMNode(
            parent = self,
            children = [],
            expanded = False,
            hide_from_visualization = True,
            distinct_tried_tactic_import_pairs = [],
            avoid_steps_str = "[AVOID STEPS]\n"
        )