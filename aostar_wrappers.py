import logging
from typing import Tuple, Type, Generator, Iterator
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from aostar_algorithm import ao_star, NodeState
from aostar_data_structures import *
import datetime
import os
from lean_cmd_executor_aostar import run_proof_on_lean
import re
from custom_logger import create_logger
from omegaconf import DictConfig, OmegaConf


@dataclass
class AOStarSolver(ABC):
    """
    Very thin wrapper around functions in aostar_algorithm
    This class is here to
        (1) allow customization of heuristic() in a OOP fashion (namely, by overridding)
        (2) set up a variables accessible to aostar_algorithm functions,
            without polluting the global namespace
    """

    theorem_statement: str
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
            case Node(_) if not node.expanded:
                return self.unexpanded_heuristic(node)
            case Node(_) if node.state == NodeState.FAILED:
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
            case MERISTEMNode(_) if len(node.parent.children) <  self.BFS_width + 1:
                # Expansion on the parent OR node has begun but hasn't finished
                # The + 1 accommodates this MERISTEM itself in addition to its AND peers
                # Force the algorithm to resume expanding (until self.BFS_width many are produced)
                return -float("inf")
            case MERISTEMNode(_) if len(node.parent.children) >= self.BFS_width + 1:
                # Expansion on the parent OR node has finished
                return  float("inf") # Must not expand anymore
            case _:
                return self.cost(node)


class AOStarZigzagSolver(AOStarCostBasedSolver):
    """
    See my algorithm writeup
    """

    def cost(self, node: Node) -> float:
        match node:
            case ANDNode(_):
                return 1
            case ORNode(_):
                return 0
            case _:
                raise RuntimeError(f"{type(node)} node is not expected to be expanded.")

    def unexpanded_heuristic(self, node: Node) -> float:
        match node:
            case MERISTEMNode(_):
                return len(node.distinct_tried_tactics)
            case _:
                return self.cost(node)


class AOStarQuadraticZigzagSolver(AOStarCostBasedSolver):
    """
    Compared to AOStarZigzagSolver, this one uses a quadratic cost function
    thus encouraging deeper search
    """

    def cost(self, node: Node) -> float:
        match node:
            case ANDNode(_):
                return 1
            case ORNode(_):
                return 0
            case _:
                raise RuntimeError(f"{type(node)} node is not expected to be expanded.")

    def unexpanded_heuristic(self, node: Node) -> float:
        match node:
            case MERISTEMNode(_):
                l = len(node.distinct_tried_tactics)
                return l * l
            case _:
                return self.cost(node)


@dataclass
class AOStarBatchSolver(ABC):
    solver_type: Type[AOStarSolver]

    def __post_init__(self):
        self.identifier_str = datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
        os.makedirs("logs", exist_ok=True)
        self.output_dir = f"logs/run_{self.identifier_str}"
        try:
            os.makedirs(self.output_dir)
        except FileExistsError as e:
            self.main_logger.critical(f"The directory {self.output_dir} already exists. Aborting to avoid overwriting or race conditions.")
            raise

        self.main_log_file_path = os.path.join(self.output_dir, "main.log")
        self.main_logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename = self.main_log_file_path,
            encoding = 'utf-8',
            level = logging.DEBUG,
            filemode = "w"
        )
    
    @abstractmethod
    def all_theorems(self) -> Iterator[Tuple[str, str, str]]:
        """
        Yields a tuple `(thm_statement, thm_group, thm_name)` each time

        `thm_statement` needs to be "almost complete": once we add
            `"    sorry\nend"` to it, it should be able to compile as a lean
            file. In particular, the imports should be included.
        `thm_group` will be used to create a subdirectory unfrt self.output_dir
        `thm_name` will be used in log and dump file names for the theorem
        """
        # yield thm_statement, thm_group, thm_name
        pass

    def solve_all(self):
        for idx, (thm_statement, thm_group, thm_name) in enumerate(self.all_theorems()):
            one_based_idx = idx + 1
            thm_identifier = str(one_based_idx) + " " \
                             + "".join(x if x.isalnum() else "_" for x in thm_name)
                             # Sanitize the theorem name to make it a valid filename
            group_dir = os.path.join(self.output_dir, thm_group)
            os.makedirs(group_dir, exist_ok=True)
            log_file_path                 = os.path.join(group_dir, thm_identifier + ".log"    )
            proof_file_path               = os.path.join(group_dir, thm_identifier + ".lean"   )
            dump_checkpoint_path          = os.path.join(group_dir, thm_identifier + ".pth.tar")
            present_search_tree_file_path = os.path.join(group_dir, thm_identifier + "_tree.txt")

            logger = create_logger(
                logger_name = thm_group + "." + thm_name,
                # This is a "dot-separated hierarchical name", in
                # the words of the `logging` module documentation
                log_file_path = log_file_path,
                logging_level = logging.DEBUG
            )
            solver = self.solver_type(
                theorem_statement = thm_statement,
                logger = logger,
                load_checkpoint_path = None,
                dump_checkpoint_path = dump_checkpoint_path,
                present_search_tree_file_path = present_search_tree_file_path
            )
            try:
                proof = solver.solve()
                with open(proof_file_path, "w") as f:
                    f.write(proof)
            except: # Don't let one failure stop the others
                pass


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
        and `leanpkg build` there before running `aostar_wrappers.py`.
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
        logger = logging.getLogger(__name__)
        log_file_path = "logs/aostar_wrappers_test.log"
        logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG, filemode="w")
        load_checkpoint_path = "logs/aostar_wrappers_test.pth.tar"
        dump_checkpoint_path = "logs/aostar_wrappers_test.pth.tar"
        present_search_tree_file_path = "logs/aostar_wrappers_test.txt"
        solver = AOStarWidthBoundedBFSSolver(
            theorem_statement,
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
        #solver = AOStarDummySolver
        #solver = AOStarWidthBoundedBFSSolver
        solver = AOStarZigzagSolver
        batch_solver = AOStarSingleFileSolver(solver, lean_file_path)
        batch_solver.solve_all()
    elif True:
        # Test driving code for AOStarCopraYAMLSolver
        #solver = AOStarDummySolver
        #solver = AOStarWidthBoundedBFSSolver
        solver = AOStarQuadraticZigzagSolver
        cfg = OmegaConf.load("config/benchmark/miniF2F_test_subset.yaml")
        batch_solver = AOStarCopraYAMLSolver(solver, cfg, skip_integrity_check=True)
        batch_solver.solve_all()