import os, datetime, argparse, pickle, traceback, re, logging
from typing import Callable, Final
from threading import Thread

from data_structures import *
from lean3_cmd_executor_aostar import run_proof_on_lean
from search_tree_visualization import present_search_tree
from prompt_gpt import GPTPrompter, GPTCircuitBreak

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

null_logger = logging.getLogger('null_logger')
null_logger.addHandler(logging.NullHandler())

def backtrack(
    node: Node,
    logger: logging.Logger,
    nodes_history: Optional[list[tuple[Node, NodeDetailedState]]] = None
) -> list[tuple[Node, NodeDetailedState]]:
    """
    Recursively notify parents of updated children.
    Returns a list of all nodes that are visited, whether they are changed,
    together with their states before the change. 
    """
    if not nodes_history:
        nodes_history = list()
    nodes_history.append((node, node.detailed_state))

    if not node.expanded:
        # As a corner case, when run on an unexpanded node, pass to the parents,
        # since the children are unknown yet and we can't update the current node based on them.
        for parent in node.parents:
            backtrack(parent, logger, nodes_history) # nodes_history is mutated during this call
        return nodes_history

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
        case MERISTEMNode():
            pass # Nothing to update--a MERISTEM node can't have children
        case _:
            raise TypeError(f"Unknown node type: {type(node)}")

    for parent in node.parents:
        if not any(parent is tup[0] for tup in nodes_history): # Without this check, we may not only duplicate the parent, but worse yet run into infinite recursions due to loops
            backtrack(parent, logger, nodes_history) # nodes_history is mutated during this call
    
    return nodes_history

def expand(
    node: Node,
    proof_so_far: str,
    prompter: GPTPrompter,
    logger: logging.Logger
) -> None:
    assert not node.expanded, f"Node {node} has already been expanded"
    if not isinstance(node, MERISTEMNode):
        node.expanded = True

    match node:
        case ANDNode(proof_step=p, necessary_import=n):
            node.expanded = True

            proof_to_run = n + "\n" + proof_so_far + standardize_indentation(p) + "\nend"
            run_lean_proof_context, run_lean_messages = run_proof_on_lean(proof_to_run) # TODO: this assumes indentation for tactic is 2
            logger.debug(f"Running the tactic {p} returns\n" +\
                        f"{run_lean_messages=} and\n" +\
                        f"{run_lean_proof_context=}")
            logger.debug(f"Running the tactic {p} leads to goals {run_lean_proof_context.fg_goals}")

            node.error_messages = [msg for msg in run_lean_messages if msg.level == 'error']
            if len(node.error_messages) > 0: # Presumably Lean syntax errors
                logger.info(
                    f"The tactic {p} failed to compile. Error messages:\n" +
                    "\n".join(msg.text for msg in node.error_messages) + '\n' +
                    "Full proof:\n" + proof_to_run + '\n'
                )
                node.detailed_state = NodeDetailedState.DOESNT_COMPILE
            else:
                logger.info(f"The tactic {p} compiles without a problem.")

                if not run_lean_proof_context.fg_goals: # If this list is empty, we have no goals to prove; we are done
                    node.detailed_state = NodeDetailedState.SOLVED
                else:
                    # For each goal, we first check if the goal is already in the tree
                    # TODO: this is recomputed every time. Consider caching.
                    def find_OR_node_with_goal_thats_a_descendant_of(node: Node, goal: Goal) -> Optional[ORNode]:
                        if isinstance(node, ORNode) and node.goal == goal:
                            return node
                        if node.state == NodeState.ACTIVE: # Guards against infinite loops. Nodes that would lead to loops would have been marked NO_PROGRESS before this line.
                            for child in node.children:
                                result = find_OR_node_with_goal_thats_a_descendant_of(child, goal)
                                if result is not None:
                                    return result
                        return None
                    
                    def exists_path(orig: Node, dest: Node, without: Node) -> bool:
                        assert orig in dest.ancestors, f"Node {orig} is not an ancestor of {dest}"
                        if dest is orig:
                            return True
                        if dest is without:
                            return False
                        return any(exists_path(orig, parent, without) for parent in dest.parents)

                    for goal in run_lean_proof_context.fg_goals:
                        already_existing_OR_node = find_OR_node_with_goal_thats_a_descendant_of(node.root, goal)
                        if already_existing_OR_node:
                            node.add_child(already_existing_OR_node)
                        else:
                            node.add_child(ORNode(goal=goal))
        case ORNode(_):
            node.add_child(MERISTEMNode())
        case MERISTEMNode(avoid_steps_str=a, distinct_tried_tactic_import_pairs=d, parent_OR_node=p): # Yes, this works even though `avoid_steps_str` is a `@property`
            assert not node.expanded, f"Node {node} has already been expanded"
            # We do NOT mark MERISTEM nodes as expanded

            message = "[GOALS]\n" + p.goal.format_message() # TODO: this now only includes the immediate parent's goal. Consider also including ancestors' so the LLM has better contexts.
            print_friendly_avoid_steps_str = str(a).replace('\n', '\\n')
            logger.info(f"Prompting for tactics with {message=} with cautions {print_friendly_avoid_steps_str}")
            tactics_import_pairs_to_try = prompter.prompt_for_tactics(message, avoid_steps=a)

            for tactic, necessary_import in tactics_import_pairs_to_try:
                if "sorry" in tactic: # TODO: This hardcodes "sorry" to mean "the goal was abandoned". Un-hardcode this in the future if we need to use "sorry" in the future.
                    logger.warning(f"The LLM decides to abandon the goal {p.goal}.")
                    node.add_child(AbandonedANDNode(
                        proof_step = tactic,
                        necessary_import = necessary_import
                    ))
                    node.expanded = True
                    node.detailed_state = NodeDetailedState.ABANDONED
                    break
                elif (tactic, necessary_import) in d:
                    logger.warning(f"The LLM repeatedly produces {tactic=} despite instructions not to do so.")
                    # Still create an AND node
                    # so we can see at the end how many times the LLM repeats itself
                    p.add_child(RepetitiveANDNode(
                        proof_step = tactic,
                        necessary_import = necessary_import
                    ))
                else:
                    # If we reach here, we have a distinct new tactic
                    # Let the LLM avoid suggesting the same tactic in the future
                    d.append((tactic, necessary_import))

                    AND_peer = ANDNode(
                        proof_step = tactic,
                        necessary_import = necessary_import
                    )
                    p.add_child(AND_peer)

                    expand(AND_peer, proof_so_far, prompter, logger)
                    # This is a nontrivial optimization--we expand AND nodes whenever they are created, rather than wait for `find()` to visit the AND node.
                    # This is because expanding AND nodes is relatively cheap, involving only running Lean on the local machine.
                    # TODO: allow this to be turned off
        case _:
            raise TypeError(f"Unknown node type: {type(node)}")

def find(
    node: Node,
    proof_so_far: str,
    estimate: Callable[[Node], float],
    prompter: GPTPrompter,
    logger: logging.Logger,
) -> None:
    print_friendly_node_str = str(node).replace("\n", "\\n")
    logger.debug(f"find() visits the node {print_friendly_node_str}, which currently has a cost estimate of {estimate(node)}")
    if not node.expanded:
        expand(node, proof_so_far, prompter, logger)
        backtrack(node, logger)
    else:
        nodes_temporarily_marked_NO_PROGRESS = list()
        match node:
            case ANDNode(proof_step=s, necessary_import=i):
                if node.parents:
                    # Unless the current node is the root,
                    # the tactic here needs to be indented
                    # TODO: check my assumption that all lines are indented by 2
                    s = standardize_indentation(s, 2)
                s += "\n" # for good measure
                if i:
                    proof_so_far = i + '\n' + proof_so_far
                proof_so_far += s
            case ORNode(goal=g):
                def disable_descendants_with_goal(node: Node, goal: Goal) -> None:
                    nonlocal nodes_temporarily_marked_NO_PROGRESS
                    for child in node.children:
                        if child.state == NodeState.ACTIVE: # Guards against infinite loops. Nodes that would lead to loops would have been marked NO_PROGRESS before this line.
                            if isinstance(child, ORNode) and child.goal == goal:
                                child.detailed_state = NodeDetailedState.NO_PROGRESS
                                nodes_temporarily_marked_NO_PROGRESS += backtrack(child, null_logger)
                                #disable_descendants_with_goal(child, goal) # This would result in infinite recursion
                            else:
                                disable_descendants_with_goal(child, goal)
                disable_descendants_with_goal(node, g)
            case MERISTEMNode():
                raise RuntimeError("A MERISTEMNode failed to be a leaf node. Check the implementation for mistakes.")
            case _:
                raise TypeError(f"Unknown node type: {type(node)}")
        
        logger.debug("Costs of children:\n" +\
            '\n'.join(f"{str(child)} has cost estimate {estimate(child)}" for child in node.children)
        )

        best_child = min(node.active_children, key=estimate)
        find(best_child, proof_so_far, estimate, prompter, logger)
        for changed_node, original_state in nodes_temporarily_marked_NO_PROGRESS:
            changed_node.detailed_state = original_state

def ao_star(
    theorem_statement: str, # Not necessarily a Lean "theorem"; can also be an "example" etc.
    estimate: Callable[[Node], float],
    prompter: GPTPrompter,
    logger: logging.Logger,
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
        logger.info("Loaded checkpoint from " + load_checkpoint_path)
        if root.proof_step != theorem_statement:
            logger.warning(f"The given {theorem_statement=}\n differs from that of the root in the checkpoint:\n{root.proof_step}")
    else:
        # Remove blank lines at the end of the string, so logs are more concise
        theorem_statement = re.sub(r'\s*\n\s*$', '', theorem_statement, flags=re.MULTILINE)
        theorem_statement += "\nbegin"
        root = ANDNode(
            proof_step = theorem_statement,
            necessary_import = ""
        )
        root.root = root
        expand(root, "", prompter, logger)
        assert root.state != NodeState.FAILED, f"Problems in the theorem statement:\n{root.error_messages}"
        assert len(root.children) == 1, "It's unexpected that the theorem statement already begets not exactly one goal." # Let me know if my assumption is wrong

    logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search started.')
    try:
        while root.state == NodeState.ACTIVE:
            find(root, "", estimate, prompter, logger)
            # Trick to prevent the saving process from being interrupted by KeyboardInterrupt
            # Found at https://stackoverflow.com/a/842567
            if dump_checkpoint_path:
                save_thread = Thread(target=serialize_tree, args=(root, dump_checkpoint_path))
                save_thread.start()
                save_thread.join()
            if present_search_tree_ANSI_file_path:
                with open(present_search_tree_ANSI_file_path, 'w') as f:
                    f.write(present_search_tree(root, style = 'ANSI'))
                if present_search_tree_HTML_file_path:
                    with open(present_search_tree_HTML_file_path, 'w') as f:
                        f.write(present_search_tree(root, style = 'HTML'))
    except KeyboardInterrupt:
        logger.info("Proof search interrupted by user.")
    except GPTCircuitBreak as e:
        logger.info(str(e))
    except BaseException:
        logger.error(traceback.format_exc())
    # Whether or not we had an exception, go on to print the proof search tree and other stats

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
    logger.info("Proof search tree:\n" + present_search_tree(root, style = 'plain'))
    logger.info("The above includes Unicode characters. Make sure to use a compatible terminal emulator or editor.")
    logger.info(f"{calculate_expansion_rate(root):.2%} of expanded AND nodes compiled fine.")
    logger.info(prompter.token_and_cost_stats)
    return proof_str

def serialize_tree(root: Node, file: str) -> None:
    with open(file, 'wb') as f:
        pickle.dump(root, f)

def calculate_expansion_rate(root: Node) -> float:
    """
    Calculate the proportion of AND nodes that "successfully" expand,
    i.e. AND nodes with expanded==True and detailed_state!=DOESNT_COMPILE
    """
    compiling_count, expanded_count = 0, 0
    counted_nodes = []
    def traverse(node: Node) -> None:
        nonlocal compiling_count, expanded_count
        if isinstance(node, ANDNode):
            if not any(node is counted_node for counted_node in counted_nodes):
            # `node not in counted_nodes` won't work because that uses `==` rather than `is`
                counted_nodes.append(node) # This is necessary because we no longer have a proper tree. Running DFS can visit some nodes more than once.
                if node.expanded:
                    expanded_count += 1
                    if node.detailed_state != NodeDetailedState.DOESNT_COMPILE:
                        compiling_count += 1
        for child in node.children:
            traverse(child)

    traverse(root)

    if compiling_count == 0:
        return 0.0

    return compiling_count / expanded_count

def collect_solution(node: Node, proof_so_far: str) -> str:
    assert node.solved, f"{node=} is not solved"
    match node:
        case ANDNode(proof_step=proof_step, necessary_import=necessary_import):
            if node.parents:
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
                raise RuntimeError(f"OR node {node} has no solved child")
        case MERISTEMNode():
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

    BFS_width = 3

    def cost(node: Node) -> float:
        match node:
            case ANDNode(_):
                return 1
            case ORNode(_):
                return 0
            case _:
                raise NotImplementedError(f"Unable to put an estimate on {node=}")

    def unexpanded_heuristic(node: Node) -> float:
        match node:
            case MERISTEMNode() if len(node.parents[0].children) <  BFS_width + 1:
                # Expansion on the parent OR node has begun but hasn't finished
                # The + 1 accommodates this MERISTEM itself in addition to its AND peers
                # Force the algorithm to resume expanding (until self.BFS_width many are produced)
                return -float("inf")
            case MERISTEMNode() if len(node.parents[0].children) >= BFS_width + 1:
                # Expansion on the parent OR node has finished
                return  float("inf") # Must not expand anymore
            case _:
                return cost(node)

    def estimate(node: Node) -> float:
        match node:
            case Node() if not node.expanded:
                return unexpanded_heuristic(node)
            case Node() if node.state == NodeState.FAILED:
                return float("inf")
            case ANDNode(_) if node.expanded:
                return cost(node) +\
                        sum(estimate(child) for child in node.children)
            case ORNode(_) if node.expanded:
                return cost(node) +\
                        min(estimate(child) for child in node.children)
            case _:
                raise NotImplementedError(f"Unable to put an estimate on {node=}")

    prompter = GPTPrompter(think_aloud=True, model_name="gpt-4o-mini")

    print(ao_star(
        theorem_statement,
        estimate,
        prompter,
        logger,
        load_checkpoint_path,
        dump_checkpoint_path,
        present_search_tree_file_path
    ))
    # Note that checkpoint dumps produced by running `algorithm.py` standalone
    # can't be used by e.g. wrappers.py, due to a pickle issue on the `Node` etc.
    # see https://stackoverflow.com/q/50394432 for more details.