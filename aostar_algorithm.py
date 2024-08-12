import os, datetime, argparse, pickle, traceback, re
from typing import Callable, Final
from logging import Logger
from threading import Thread

from aostar_data_structures import *
from lean_cmd_executor_aostar import run_proof_on_lean
from search_tree_visualization import present_search_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='logs/proof_search.log', help='Where to store the log')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Where to load the search tree')
    parser.add_argument('--dump_checkpoint_path', type=str, default='logs/proof_search_tree.pth.tar', help='Where to save the search tree')
    parser.add_argument('--present_search_tree_file_path', type=str, default='logs/proof_search_tree.txt', help='Where to print the search tree in real time in a readable form')
    args = parser.parse_args()

    log_path: Final[str] = args.log_path
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG, filemode="w")

    if args.load_checkpoint_path:
        load_checkpoint_path: Final[str] = os.path.abspath(args.load_checkpoint_path)
    else:
        load_checkpoint_path = None

    if args.dump_checkpoint_path:
        dump_checkpoint_path: Final[str] = os.path.abspath(args.dump_checkpoint_path)
    else:
        dump_checkpoint_path = None

    if args.present_search_tree_file_path:
        present_search_tree_file_path: Final[str] = os.path.abspath(args.present_search_tree_file_path)
    else:
        present_search_tree_file_path = None

else:
    pass
    # The functions in this script still expect the the above variables to be available (initialized)
    # outside the functions body. Any external caller will be responsible for setting the variables.


def backtrack(node: Node, logger: Logger) -> None:
    """
    Recursively notify parents of updated children.
    """
    # All updates to estimation are now removed from backtrack(),
    # as Node.estimate will now be computed
    # TODO: computing estimate every time is more expensive. Add back some way to cache estimate values
    if not node.expanded:
        # This function looks at children to update the current node.
        # As a corner case, when run on a node w/o children, pass to the parent
        backtrack(node.parent, logger)
        return

    print_friendly_node_str = str(node).replace("\n", "\\n")
    logger.debug(f"Backtracking on node {print_friendly_node_str}, which was {node.solved=} before backtracking")

    match node:
        case ANDNode(_):
            if any(child.state == NodeState.FAILED for child in node.children):
                node.detailed_state = NodeDetailedState.FAILED_DUE_TO_CHILDREN
            elif all(child.state == NodeState.SOLVED for child in node.children):
                node.detailed_state = NodeDetailedState.SOLVED
        case ORNode(_):
            if all(child.state == NodeState.FAILED for child in node.children):
                node.detailed_state = NodeDetailedState.FAILED_DUE_TO_CHILDREN
            elif any(child.state == NodeState.SOLVED for child in node.children):
                node.detailed_state = NodeDetailedState.SOLVED
        case MERISTEMNode(_):
            pass # Nothing to update--a MERISTEM node is always ACTIVE
        case _:
            raise TypeError(f"Unknown node type: {type(node)}")

    if node.parent is not None:
        if node.state != NodeState.ACTIVE: # Recently solved or failed
            node.parent.remove_child(node)
        backtrack(node.parent, logger)

def find(
    node: Node,
    proof_so_far: str,
    estimate: Callable[[Node], float],
    logger: Logger
) -> None:
    print_friendly_node_str = str(node).replace("\n", "\\n")
    logger.debug(f"find() visits the node {print_friendly_node_str}, which currently has a cost estimate of {estimate(node)}")
    if not node.expanded:
        node.expand(proof_so_far, logger)
        backtrack(node, logger)
    else:
        match node:
            case ANDNode(proof_step=s, necessary_import=i):
                if node.parent is not None:
                    # Unless the current node is the root,
                    # the tactic here needs to be indented
                    # TODO: check my assumption that all lines are indented by 2
                    s = standardize_indentation(s, 2)
                s += "\n" # for good measure
                if i:
                    proof_so_far = i + '\n' + proof_so_far
                proof_so_far += s
            case ORNode(_):
                pass
            case MERISTEMNode(_):
                raise RuntimeError("A MERISTEMNode failed to be a leaf node. Check the implementation for mistakes.")
            case _:
                raise TypeError(f"Unknown node type: {type(node)}")
        
        logger.debug("Costs of children:\n" +\
            '\n'.join(f"{str(child)} has cost estimate {estimate(child)}" for child in node.children)
        )
        best_child = min(node.unsolved_children, key=estimate)
        find(best_child, proof_so_far, estimate, logger)

def ao_star(
    theorem_statement: str, # Not necessarily a Lean "theorem"; can also be an "example" etc.
    estimate: Callable[[Node], float],
    logger: Logger,
    load_checkpoint_path: Optional[str],
    dump_checkpoint_path: Optional[str],
    present_search_tree_file_path: Optional[str]
) -> None:
    if present_search_tree_file_path.endswith('.html'):
        logger.warning("present_search_tree_file_path ends with .html "
            "(which is NOT recommended since it's the ANSI escape code version of the search tree "
            "that will be written to present_search_tree_file_path). "
            "Will skip saving the HTML version."
        )
        present_search_tree_ANSI_file_path = present_search_tree_file_path
    elif present_search_tree_file_path:
        present_search_tree_ANSI_file_path = present_search_tree_file_path
        present_search_tree_HTML_file_path = os.path.splitext(present_search_tree_ANSI_file_path)[0] + '.html'

    if load_checkpoint_path:
        with open(load_checkpoint_path, 'rb') as f:
            root = pickle.load(f)
        if root.proof_step != theorem_statement:
            logger.warning(f"{theorem_statement=}\n differs from that of the root:\n{root.proof_step}")
    else:
        # Remove blank lines at the end of the string, so logs are more concise
        theorem_statement = re.sub(r'\s*\n\s*$', '', theorem_statement, flags=re.MULTILINE)
        theorem_statement += "\nbegin"
        root = ANDNode(
            parent = None,
            children = [],
            expanded = False,
            hide_from_visualization = False,
            proof_step = theorem_statement,
            necessary_import = ""
        )
        root_proof_context, root_lean_messages = run_proof_on_lean(theorem_statement + "\nend", max_memory_in_mib=3000)
        assert all(not msg.level == 'error' for msg in root_lean_messages), f"Problems in the theorem statement:\n{root_lean_messages}"
        root_goals = root_proof_context.fg_goals # If the assert above fails, this can't even work as root_proof_context would be `None`
        # root_goals should be a singleton: Lean just gets the theorem statement so the one goal is the conclusion of the theorem
        assert len(root_goals) == 1, "It's unexpected that the theorem statement already begets not exactly one goal." # Let me know if my assumption is wrong
        root_goal = root_goals[0]
        ORNode(
            parent = root, 
            children = [],
            expanded = False,
            hide_from_visualization = False,
            goal = root_goal
        )

    logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search started.')
    try:
        while root.state == NodeState.ACTIVE:
            find(root, "", estimate, logger)
            # Trick to prevent the saving process from being interrupted by KeyboardInterrupt
            # Found at https://stackoverflow.com/a/842567
            if dump_checkpoint_path:
                save_thread = Thread(target=serialize_tree, args=(root, dump_checkpoint_path))
                save_thread.start()
                save_thread.join()
            if present_search_tree_ANSI_file_path:
                with open(present_search_tree_ANSI_file_path, 'w') as f:
                    f.write(present_search_tree(
                        root,
                        style = 'ANSI',
                        is_part_of_solution = root.state == NodeState.ACTIVE
                    ))
                if present_search_tree_HTML_file_path:
                    with open(present_search_tree_HTML_file_path, 'w') as f:
                        f.write(present_search_tree(
                            root,
                            style = 'HTML',
                            is_part_of_solution = root.state == NodeState.ACTIVE
                        ))
    except KeyboardInterrupt:
        logger.info("Proof search interrupted by user.")
    except BaseException:
        logger.error(traceback.format_exc())
    # Whether or not we had an exception, go on to print the proof search tree

    proof_str = ""
    match root.state:
        case NodeState.SOLVED:
            proof_str = collect_solution(root, "")
            logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search successful:\n' + proof_str)
        case NodeState.FAILED:
            logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search unsuccessful.')
        case NodeState.ACTIVE:
            logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search did not finish.')
        case _:
            logger.error(f"Proof search ended with an unexpected state {root.state=}")
    logger.info("Proof search tree:\n" + present_search_tree(
        root,
        style = 'plain',
        is_part_of_solution = root.state == NodeState.ACTIVE
    ))
    logger.info("The above includes Unicode characters. Make sure to use a compatible terminal emulator or editor.")
    logger.info(f"Proof search incurred {prompt_for_tactics.gpt_token_counter} tokens in total, ")
    return proof_str

def serialize_tree(root: Node, file: str) -> None:
    with open(file, 'wb') as f:
        pickle.dump(root, f)

def collect_solution(node: Node, proof_so_far: str) -> str:
    assert node.solved, f"{node=} is not solved"
    match node:
        case ANDNode(proof_step=proof_step, necessary_import=necessary_import):
            if node.parent is not None:
                # Unless the current node is the root,
                # the tactic here needs to be indented
                # TODO: check my assumption that all lines are indented by 4
                necessary_import = standardize_indentation(necessary_import, 0)
                proof_step = standardize_indentation(proof_step, 4)
                proof_str = necessary_import + '\n' + proof_so_far + proof_step + '\n'
                for child in node.children:
                    proof_str = collect_solution(child, proof_str)
                    # The recursive call will check the children are each solved
            else: # root Node
                proof_str = necessary_import + proof_so_far + proof_step + '\n'
                for child in node.children:
                    proof_str = collect_solution(child, proof_str)
                    # The recursive call will check the children are each solved
                proof_str += "end"
        case ORNode(_):
            proof_str = proof_so_far
            properly_settled = False
            for child in node.children:
                if child.solved:
                    proof_str = collect_solution(child, proof_str)
                    properly_settled = True
                    break # TODO: If more than one proof is found, print all possibilities
            if not properly_settled:
                raise RuntimeError("OR node {node} has no solved child")
        case MERISTEMNode(_):
            raise RuntimeError("A MERISTEM node should never be part of a solution")
        case _:
            raise TypeError(f"Unknown node type: {type(node)}")
    return proof_str

if __name__ == "__main__":
    # Test driving code

    theorem_statement = "theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :="
    #theorem_statement = "theorem inequality_chain (a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :="
#    theorem_statement = """import data.nat.basic
#theorem amc12a_2015_p10
#  (x y : ℤ)
#  (h₀ : 0 < y)
#  (h₁ : y < x)
#  (h₂ : x + y + (x * y) = 80) :
#  x = 26 :=
#"""

    def unexpanded_heuristic(node: Node) -> float: # Results in naive DFS
        match node:
            case ANDNode(_):
                return 1
            case ORNode(_):
                return 0
            case MERISTEMNode(_):
                return 0

    def cost(node: Node) -> float: # Results in naive DFS
        return 0

    def estimate(node: Node) -> float:
        if not node.expanded:
            match node.state:
                case NodeState.FAILED:
                    return float("inf")
                case NodeState.SOLVED:
                    return unexpanded_heuristic(node)
                case NodeState.ACTIVE:
                    pass # More calculation to do
                case _:
                    raise TypeError(f"Unrecognized node state: {node.state}")
        match node:
            case ANDNode(_):
                return cost(node) + sum(map(estimate, node.children))
            case ORNode(_):
                return cost(node) + min(map(estimate, node.children))

    print(ao_star(
        theorem_statement,
        estimate,
        logger,
        load_checkpoint_path,
        dump_checkpoint_path,
        present_search_tree_file_path
    ))
    # Note that checkpoint dumps produced by running aostar_algorithm.py standalone
    # can't be used by e.g. aostar_wrappers.py, due to a pickle issue on the `Node` etc.
    # see https://stackoverflow.com/q/50394432 for more details.