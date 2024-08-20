import logging, datetime, os, re, traceback
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed


from typing import Tuple, Type, Generator, Iterator, Final
from omegaconf import DictConfig, OmegaConf

from lean_cmd_executor_aostar import run_proof_on_lean
from custom_logger import create_logger
from algorithm import ao_star, NodeState
from data_structures import *
from prompt_gpt import GPTPrompter


@dataclass
class AOStarSolver(ABC):
    """
    Very thin wrapper around functions in `algorithm.py`
    This class is here to
        (1) allow customization of heuristic() in a OOP fashion (namely, by overridding)
        (2) set up a variables accessible to `algorithm.py` functions,
            without polluting the global namespace
    """

    theorem_statement: str
    prompter: GPTPrompter
    logger: logging.Logger
    load_checkpoint_path: Optional[str]
    dump_checkpoint_path: Optional[str]
    present_search_tree_file_path: Optional[str]

    def __post_init__(self):
        pass

    @abstractmethod
    def estimate(self, node: Node) -> float:
        pass
    # Note that while AOStarSolver.heuristic has signature Callable[[AOStarSolver, Node], float], while
    # self.heuristic on any instance self has signature Callable[[Node], float] according to Python's
    # method objects' __get__() rules, so self.heuristic can be passed to a Node but not AOStarSolver.heuristic

    def solve(self):
        return ao_star(
            self.theorem_statement,
            self.estimate,
            self.prompter,
            self.logger,
            self.load_checkpoint_path,
            self.dump_checkpoint_path,
            self.present_search_tree_file_path
        )


class AOStarDummySolver(AOStarSolver):
    # For unit testing `AOStarBatchSolver`s
    def estimate(self, node: Node) -> float:
        return 1
    def solve(self):
        self.logger.info("AOStarDummySolver.solve being called")
        return "dummy proof"


@dataclass
class AOStarCostBasedSolver(AOStarSolver):
    """
    When `estimate` is based on some fixed `cost` node valuation
    """

    @abstractmethod
    def cost(self, node: Node) -> float:
        pass

    @abstractmethod
    def unexpanded_heuristic(self, node: Node) -> float:
        pass

    def estimate(self, node: Node) -> float:
        match node:
            case Node() if not node.expanded:
                return self.unexpanded_heuristic(node)
            case Node() if node.state == NodeState.FAILED:
                return float("inf")
            case ANDNode(_) if node.expanded:
                return self.cost(node) +\
                        sum(self.estimate(child) for child in node.children)
            case ORNode(_) if node.expanded:
                return self.cost(node) +\
                        min(self.estimate(child) for child in node.children)
            case _:
                raise NotImplementedError(f"Unable to put an estimate on {node=}")


@dataclass
class AOStarWidthBoundedBFSSolver(AOStarCostBasedSolver):
    """
    Under each OR node, attempt to generate exactly BFS_width
    many tactics (i.e. AND nodes)
    """
    BFS_width: int = 3 # TODO: the current AOStarBatchSolver implementation gives no way to customize this

    def __post_init__(self):
        super().__post_init__()
        assert self.BFS_width >= 0, "The BFS width can't be negative"
        assert self.BFS_width > 0, "The BFS width can't be zero"

    def cost(self, node: Node) -> float:
        match node:
            case ANDNode(_):
                return 1
            case ORNode(_):
                return 0
            case _:
                raise NotImplementedError(f"Unable to put an estimate on {node=}")

    def unexpanded_heuristic(self, node: Node) -> float:
        match node:
            case MERISTEMNode() if len(node.parents[0].children) <  self.BFS_width + 1:
                # Expansion on the parent OR node has begun but hasn't finished
                # The + 1 accommodates this MERISTEM itself in addition to its AND peers
                # Force the algorithm to resume expanding (until self.BFS_width many are produced)
                return -float("inf")
            case MERISTEMNode() if len(node.parents[0].children) >= self.BFS_width + 1:
                # Expansion on the parent OR node has finished
                return  float("inf") # Must not expand anymore
            case _:
                return self.cost(node)


@dataclass
class AOStarZigzagSolver(AOStarCostBasedSolver):
    """
    See my algorithm writeup
    """
    repeated_tactics_count: bool = True
    bad_tactics_count: bool = False
    # TODO: the current AOStarBatchSolver implementation gives no way to customize these

    def cost(self, node: Node) -> float:
        match node:
            case ANDNode(_):
                return 1
            case ORNode(_):
                return 0
            case _:
                raise RuntimeError(f"{type(node)} node is not expected to be expanded.")

    def unexpanded_heuristic(self, node: Node) -> float:
        def counts(node: Node) -> bool:
            if node.detailed_state in [NodeDetailedState.IS_REPETITIVE, NodeDetailedState.NO_PROGRESS] and not self.repeated_tactics_count:
                return False
            if node.detailed_state == NodeDetailedState.DOESNT_COMPILE and not self.bad_tactics_count:
                return False
            return True

        match node:
            case MERISTEMNode():
                return len([child for child in node.parents[0].children if counts(child)])
            case _:
                return self.cost(node)


class AOStarQuadraticZigzagSolver(AOStarZigzagSolver):
    """
    Compared to AOStarZigzagSolver, this one uses a quadratic cost function
    thus encouraging deeper search
    """
    def unexpanded_heuristic(self, node: Node) -> float:
        match node:
            case MERISTEMNode():
                if self.repeated_tactics_count:
                    l = len(node.parents[0].children)
                else:
                    l = len(node.distinct_tried_tactics)
                return l * l
            case _:
                return self.cost(node)


class AOStarExponentialZigzagSolver(AOStarZigzagSolver):
    """
    Compared to AOStarZigzagSolver, this one uses a exponential cost function
    thus encouraging deeper search
    """
    def unexpanded_heuristic(self, node: Node) -> float:
        match node:
            case MERISTEMNode():
                if self.repeated_tactics_count:
                    l = len(node.parents[0].children)
                else:
                    l = len(node.distinct_tried_tactics)
                return 2 ** l
            case _:
                return self.cost(node)


# TODO: Analyze actual search trees and consider when linear, qudratic or exponential is the best.


@dataclass
class AOStarBatchSolver(ABC):
    max_threads: int
    solver_type: Type[AOStarSolver]
    use_dir: Optional[str] # Use (or reuse) a directory
    think_aloud: bool
    model_name: str
    budget_per_problem: int
    identifier_str: str = field(
        default_factory =
            lambda: datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"),
        init = False
    )
    output_dir: str = field(init=False)
    main_logger: logging.Logger = field(init=False)
    main_log_file_path: str = field(init=False)

    def __post_init__(self):
        os.makedirs("logs", exist_ok=True)
        if self.use_dir:
            self.output_dir = self.use_dir
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = f"logs/run_{self.identifier_str}"
            try:
                os.makedirs(self.output_dir, exist_ok=False)
            except FileExistsError as e:
                self.main_logger.critical(f"The directory {self.output_dir} already exists. Aborting to avoid overwriting or race conditions.")
                raise

        self.main_log_file_path = os.path.join(self.output_dir, "main.log")
        self.main_logger = create_logger("main", self.main_log_file_path)

    @abstractmethod
    def all_theorems(self) -> Iterator[Tuple[str, str, str]]:
        """
        Yields a tuple `(thm_statement, thm_group, thm_name)` each time

        `thm_statement` needs to be "almost complete": once we add
            `"  sorry\nend"` to it, it should be able to compile as a lean
            file. In particular, the imports should be included.
        `thm_group` will be used to create a subdirectory under self.output_dir
        `thm_name` will be used in log and dump file names for the theorem
        """
        # yield thm_statement, thm_group, thm_name
        pass

    def solve_all(self) -> None:
        total_token_count = 0
        total_cost = 0 # In cents

        def solve_theorem(idx, thm_statement, thm_group, thm_name):
            one_based_idx = idx + 1
            thm_identifier = str(one_based_idx) + " " \
                             + "".join(x if x.isalnum() else "_" for x in thm_name)
                             # Sanitize the theorem name to make it a valid filename
            group_dir = os.path.join(self.output_dir, thm_group)
            os.makedirs(group_dir, exist_ok=True)
            log_file_path                 = os.path.join(group_dir, thm_identifier + ".log"     )
            proof_file_path               = os.path.join(group_dir, thm_identifier + ".lean"    )
            dump_checkpoint_path          = os.path.join(group_dir, thm_identifier + ".pth.tar" )
            present_search_tree_file_path = os.path.join(group_dir, thm_identifier + "_tree.txt")

            prompter = GPTPrompter(
                model_name = self.model_name,
                think_aloud = self.think_aloud,
                budget = self.budget_per_problem
            )
            logger = create_logger(
                logger_name = thm_group + "." + thm_name,
                # This is a "dot-separated hierarchical name", in
                # the words of the `logging` module documentation
                log_file_path = log_file_path,
                logging_level = logging.INFO
            )
            if os.path.isfile(dump_checkpoint_path):
                load_checkpoing_path = dump_checkpoint_path
                self.main_logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search started for {thm_group}.{thm_name}. ' +\
                    f"Relevant log can be found at {log_file_path}.")
            else:
                load_checkpoing_path = None
                self.main_logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Proof search restarted for {thm_group}.{thm_name}. ' +\
                    f"Log will be appended to {log_file_path}. Please note that previous token counts aren't carried over.")
            solver = self.solver_type(
                theorem_statement = thm_statement,
                prompter = prompter,
                logger = logger,
                load_checkpoint_path = load_checkpoing_path,
                dump_checkpoint_path = dump_checkpoint_path,
                present_search_tree_file_path = present_search_tree_file_path
            )

            try:
                proof = solver.solve() # This may throw, but the **main** thread doesn't see the exception until `future.result()` is called.
                with open(proof_file_path, "w") as f:
                    f.write(proof)
            except Exception as e: # Don't let one failure stop the others
                idx, thm_statement, thm_group, thm_name = thm_info
                self.main_logger.error(f"Failed to solve theorem {thm_name}: {repr(e)}")
                self.main_logger.info(f"{traceback.format_exc()=}")
            finally:
                self.main_logger.info(f"LLM usage stats for proof search on {thm_name}: {prompter.token_and_cost_stats}")

        try:
            for idx, (thm_statement, thm_group, thm_name) in enumerate(self.all_theorems()):
                with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                    future_to_theorem = {
                        executor.submit(solve_theorem, idx, thm_statement, thm_group, thm_name): (idx, thm_statement, thm_group, thm_name)
                        for idx, (thm_statement, thm_group, thm_name) in enumerate(self.all_theorems())
                    }
                    
                    for future in as_completed(future_to_theorem):
                        thm_info = future_to_theorem[future]
                        future.result()  # Will raise an exception if the function raised
        except KeyboardInterrupt:
            self.main_logger.info("Keyboard interrupt detected. Aborting proof search on all theorems.")
            # TODO: send a KeyboardInterrupt to all the threads so they can terminate gracefully.
            return


@dataclass
class AOStarSingleFileSolver(AOStarBatchSolver):
    """
    Runs all `theorem`s and `examples` in the found file.
    """

    lean_file_path: str
    skip_integrity_check: bool = False

    def __post_init__(self):
        super().__post_init__()
        with open(self.lean_file_path, "r") as f:
            self.lean_file_contents = f.read()
        self.lean_file_name = os.path.basename(self.lean_file_path)
        if not self.skip_integrity_check:
            # Preliminary check that the lean file compiles. This is necessary because
            # self.all_theorems will depend on each theorem in the lean file
            # being complete (i.e., having `theorem` (or example), `begin` and `end`)
            self.main_logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Started checking integrity of lean files.')
            _, preliminary_run_messages = run_proof_on_lean(self.lean_file_contents)
            assert all(not msg.level == 'error' for msg in preliminary_run_messages), f"Problems in the theorem statement:\n{preliminary_run_messages}"
            self.main_logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Checked file {self.lean_file_name} and it compiled.')

    @staticmethod
    def find_imports(lean_file_contents) -> str:
        import_regex = r"^[ \t]*import(?:[ \t]+(?:\w+(?:\.\w+)*))+[ \t]*$"
        import_match = re.compile(import_regex, re.MULTILINE)
        return '\n'.join(import_match.findall(lean_file_contents))

    def all_theorems(self) -> Generator[Tuple[str, str, str, str], None, None]:
        # regex adapted from copra lean_executor.py
        theorem_regex = r"(((theorem\s+([\w+|\d+]*))|example)(?:\s+?)([\S|\s]*?):=[\S|\s]*?)(begin|by|calc)"
        theorem_match = re.compile(theorem_regex, re.MULTILINE)
        """
        Theorems and examples are matched like the following:
        "example : c * b * a = b * (a * c) := by" -> (f1='example : c * b * a = b * (a * c) := ', 'example', '', '', ' : c * b * a = b * (a * c) ', 'by')
        "theorem add_zero (a : R) : a + 0 = a := by" -> ('theorem add_zero (a : R) : a + 0 = a := ', 'theorem add_zero', 'theorem add_zero', 'add_zero', ' (a : R) : a + 0 = a ', 'by')
        """

        imports = self.find_imports(self.lean_file_contents)
        for match_obj in theorem_match.finditer(self.lean_file_contents):
            thm_wo_import, _, _, thm_name, _, _ = match_obj.groups()
            thm_statement = imports + '\n' + thm_wo_import
            yield thm_statement, self.lean_file_name, thm_name


@dataclass
class AOStarCopraYAMLSolver(AOStarBatchSolver):
    """
    Runs selected `theorem`s from specified files in a copra benchmark yaml.
    Note that as `example`s don't have names, they can't be indexed and
    therefore can't be selected by such a yaml at all.
    """

    cfg: DictConfig
    """
    Expects a DictConfig object extracted from a yaml configure file as exemplified
    by `config/benchmark/barebones_example.yaml`. More detailed explanations:
    1. Superfluous keys (compared to `barebones_example.yaml`) are simply ignored;
        see miniF2F_curriculum for what's acceptible.
    2. With the yaml file as `cfg`, `cfg.datasets[0].project` should be the Lean 3
        project, i.e. where we put `leanpkg.toml`. Be sure to run `leanpkg configure`
        and `leanpkg build` there before running `wrappers.py`.
    3. With the yaml file as `cfg`,
        `os.path.join(cfg.datasets[0].project, cfg.datasets[0].files[i].path)`
        should lead to an actual Lean 3 file, for each $i$.
    4. With the yaml file as `cfg`,
        `os.path.join(cfg.datasets[0].project, cfg.datasets[0].files[i].theorems)`,
        a list of strings, should specify a subset of all named theorems in the file,
        for each $i$.
    """
    skip_integrity_check: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.skip_integrity_check:
            # Preliminary check that the lean file compiles. This is necessary because
            # self.all_theorems will depend on each theorem in the lean file
            # being complete (i.e., having `theorem` (or example), `begin` and `end`)
            self.main_logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: Started checking integrity of lean files.')
            for dataset in self.cfg.datasets:
                for lean_file in dataset.files:
                    lean_file_path = os.path.join(dataset.project, lean_file.path)
                    lean_file_full_path = os.path.abspath(lean_file_path)
                    with open(lean_file_full_path, "r") as f:
                        lean_file_contents = f.read()
                    _, preliminary_run_messages = run_proof_on_lean(
                        lean_file_contents,
                        timeout_in_secs = 300 # Increase this further if needed
                    )
                    assert all(not msg.level == 'error' for msg in preliminary_run_messages),\
                        f"Problems in the theorem statements in {lean_file_full_path}:\n{preliminary_run_messages}"
            self.main_logger.info(f'{datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")}: All lean files succeeessfully compiled.')
            # The above would check if the lean files compile, like AOStarSingleFileSolver does
            # If the lean file is a large dataset file, this could take very long.
            # Use the following implementation to check only the thms selected by the yaml.
            # WARNING: it may need more debugging
            #theorem_regex = r"((theorem\s+([\w+|\d+]*))(?:\s+?)([\S|\s]*?):=)"
            #theorem_match = re.compile(theorem_regex, re.MULTILINE)
            #for dataset in self.cfg.datasets:
            #    for lean_file in dataset.files:
            #        current_file_theorems = OmegaConf.to_container(lean_file.theorems)
            #        lean_file_path = os.path.join(dataset.project, lean_file.path)
            #        with open(lean_file_path, "r") as f:
            #            lean_file_contents = f.read()

            #        imports = self.find_imports(lean_file_contents)
            #        os.makedirs(os.path.dirname(lean_file_path), exist_ok=True)
            #        for match_obj in theorem_match.finditer(lean_file_contents):
            #            thm_wo_import = match_obj.groups()[0]
            #            thm_name = match_obj.groups()[2]
            #            if thm_name in current_file_theorems:
            #                thm_statement = imports + '\n' + thm_wo_import + "\nbegin\n  sorry\nend"
            #                _, preliminary_run_messages = run_proof_on_lean(
            #                    thm_statement,
            #                    max_memory_in_mib = 3000
            #                )
            #                assert all(not msg.level == 'error' for msg in preliminary_run_messages), f"Problems in the theorem statement:\n{preliminary_run_messages}"
    
    @staticmethod
    def find_imports(lean_file_contents) -> str:
        import_regex = r"^[ \t]*import(?:[ \t]+(?:\w+(?:\.\w+)*))+[ \t]*$"
        import_match = re.compile(import_regex, re.MULTILINE)
        return '\n'.join(import_match.findall(lean_file_contents))
    
    def all_theorems(self) -> Generator[Tuple[str, str, str, str], None, None]:
        # regex adapted from copra lean_executor.py
        theorem_regex = r"(((theorem\s+([\w+|\d+]*))|example)(?:\s+?)([\S|\s]*?):=[\S|\s]*?)(begin|by|calc)"
        theorem_match = re.compile(theorem_regex, re.MULTILINE)
        """
        Theorems and examples are matched like the following:
        "example : c * b * a = b * (a * c) := by" -> (f1='example : c * b * a = b * (a * c) := ', 'example', '', '', ' : c * b * a = b * (a * c) ', 'by')
        "theorem add_zero (a : R) : a + 0 = a := by" -> ('theorem add_zero (a : R) : a + 0 = a := ', 'theorem add_zero', 'theorem add_zero', 'add_zero', ' (a : R) : a + 0 = a ', 'by')
        """

        for dataset in self.cfg.datasets:
            for lean_file in dataset.files:
                current_file_theorems = OmegaConf.to_container(lean_file.theorems)
                lean_file_path = os.path.join(dataset.project, lean_file.path)
                with open(lean_file_path, "r") as f:
                    lean_file_contents = f.read()

                imports = self.find_imports(lean_file_contents)
                os.makedirs(os.path.dirname(lean_file_path), exist_ok=True)
                for match_obj in theorem_match.finditer(lean_file_contents):
                    thm_wo_import, _, _, thm_name, _, _ = match_obj.groups()
                    if thm_name in current_file_theorems:
                        thm_statement = imports + '\n' + thm_wo_import
                        yield thm_statement, lean_file_path, thm_name

if __name__ == "__main__":
    if False:
        # Test driving code for AOStarWidthBoundedBFSSolver
        theorem_statement = "theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :="
        load_checkpoint_path = None
        prompter = GPTPrompter()
        logger = logging.getLogger(__name__)
        log_file_path = "logs/wrappers_test.log"
        logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG, filemode="w")
        #load_checkpoint_path = "logs/wrappers_test.pth.tar"
        load_checkpoint_path = None
        dump_checkpoint_path = "logs/wrappers_test.pth.tar"
        present_search_tree_file_path = "logs/wrappers_test.txt"
        solver = AOStarWidthBoundedBFSSolver(
            theorem_statement,
            prompter,
            logger,
            load_checkpoint_path,
            dump_checkpoint_path,
            present_search_tree_file_path
        )
        proof_str = solver.solve()
        if proof_str:
            print(proof_str)
            logger.info("The discovered proof: \n" + proof_str)
    elif False:
        # Test driving code for AOStarSingleFileSolver
        lean_file_path = "/home/billion/Projects/aostar-rewritten/testbed/src/simple2.lean"
        #use_dir = "/home/billion/Projects/aostar-rewritten/logs/run_2024-Aug-20-20-17-57"
        use_dir = None
        #solver_type = AOStarDummySolver
        #solver_type = AOStarWidthBoundedBFSSolver
        #solver_type = AOStarZigzagSolver
        #solver_type = AOStarQuadraticZigzagSolver
        solver_type = AOStarExponentialZigzagSolver
        batch_solver = AOStarSingleFileSolver(
            max_threads = 2,
            solver_type = solver_type,
            use_dir = use_dir,
            think_aloud = False,
            model_name = "gpt-4o-2024-08-06",
            budget_per_problem = 1,
            lean_file_path = lean_file_path
        )
        batch_solver.solve_all()
    elif True:
        # Test driving code for AOStarCopraYAMLSolver
        #solver_type = AOStarDummySolver
        #solver_type = AOStarWidthBoundedBFSSolver
        #solver_type = AOStarQuadraticZigzagSolver
        solver_type = AOStarExponentialZigzagSolver
        cfg = OmegaConf.load("config/benchmark/miniF2F_test_subset_subset2.yaml")
        batch_solver = AOStarCopraYAMLSolver(
            max_threads = 4,
            solver_type = solver_type,
            use_dir = None,
            think_aloud = True,
            #model_name = "gpt-4",
            model_name = "gpt-4o-2024-08-06",
            budget_per_problem = 1,
            cfg = cfg,
            skip_integrity_check = True
        )
        batch_solver.solve_all()