import logging
from typing import List, Tuple, Type, Generator, Iterator
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import aostaralgorithm
from aostaralgorithm import ao_star
import datetime
import os
from lean_cmd_executor_aostar import run_proof_on_lean
import re
from custom_logger import create_logger

@dataclass
class AOStarTheoremSolver(ABC):
    """
    Very thin wrapper around functions in aostaralgorithm
    This class is here to
        (1) allow customization of heuristic() in a OOP fashion (namely, by overridding)
        (2) set up a variables accessible to aostaralgorithm functions,
            without polluting the global namespace
    """

    theorem_statement: str
    project_root: str
    logger: logging.Logger
    load_file_path: str
    dump_file_path: str

    @abstractmethod
    def heuristic(self, node: aostaralgorithm.Node) -> float:
        return 1
    # Note that while AOStarTheoremSolver.heuristic has signature Callable[[AOStarTheoremSolver, Node], float], while
    # self.heuristic on any instance self has signature Callable[[Node], float] according to Python's
    # method objects' __get__() rules, so self.heuristic can be passed to a Node but not AOStarTheoremSolver.heuristic

    def solve(self):
        aostaralgorithm.__dict__.update(asdict(self))
        """ Explanation for the last line:
        We can't just run aostaralgorithm.ao_star(self.theorem_statement) since,
        as explained in aostaralgorithm.py, a few variables like load_file_path
        need to be defined before we can call ao_star. Not even
        ```Python
        global load_file_path
        load_file_path = "some/path"
        ```
        in AOStarTheoremSolver would work, because when ao_star is imported, it will use
        variables from the aostaralgorithm *module* scope, *not* the global scope.
        Hence, we have to inject the variables into the module scope.
        """
        aostaralgorithm.heuristic = self.heuristic
        return ao_star(self.theorem_statement)

class AOStarBFSTheoremSolver(AOStarTheoremSolver):
    def heuristic(self, node: aostaralgorithm.Node) -> float:
        return 1

class AOStarDummyTheoremSolver(AOStarTheoremSolver):
    # For unit testing purposes
    def heuristic(self, node: aostaralgorithm.Node) -> float:
        return 1
    def solve(self):
        self.logger.info("AOStarDummyTheoremSolver.solve being called")
        return "dummy proof"

@dataclass
class AOStarBatchSolver(ABC):
    solver_type: Type[AOStarTheoremSolver]
    project_root: str

    def __post_init__(self):
        self.identifier_str = datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
        self.output_dir = f"logs/run_{self.identifier_str}"
        os.makedirs(self.output_dir, exist_ok=True) # !! TODO: it's not OK if this dir already exists
        # !! TODO: does makedirs ensure `logs` exists in the first place?

        self.main_log_file_path = os.path.join(self.output_dir, "main.log")
        self.main_logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename = self.main_log_file_path,
            encoding = 'utf-8',
            level = logging.DEBUG,
            filemode = "w"
        )
    
    @abstractmethod
    def all_theorems(self) -> Iterator[Tuple[str, str, str]]: # !! TODO: or Generator?
        """
        Yields a tuple `(thm_statement, thm_group, thm_name)` each time where
        `thm_statement` needs to be "almost complete": once we add
            `"    sorry\nend"` to it, it should be able to compile as a lean
            file. In particular, the imports should be included.
        `thm_group` will be used to create a subdirectory unfrt self.output_dir
        `thm_name` will be used in log and dump file names for the theorem
        """
        # yield thm_statement, thm_group, thm_name
        pass

    def solve_all(self):
        for thm_statement, thm_group, thm_name in self.all_theorems(): # !! TODO: check if we can use enumerate() on generators
            group_dir = os.path.join(self.output_dir, thm_group)
            os.makedirs(group_dir, exist_ok=True) # !! TODO: it's not OK if this dir already exists
            log_file_path   = os.path.join(group_dir, thm_name + ".log"    )
            proof_file_path = os.path.join(group_dir, thm_name + ".lean"   )
            dump_file_path  = os.path.join(group_dir, thm_name + ".pth.tar")

            # !! TODO: confirm whether logger name uses "." to denote subserviency
            logger = create_logger(
                logger_name = thm_group + "." + thm_name,
                log_file_path = log_file_path,
                logging_level = logging.DEBUG
            )
            solver = self.solver_type(
                theorem_statement = thm_statement,
                project_root = self.project_root,
                logger = logger,
                load_file_path = None,
                dump_file_path = dump_file_path
            )
            proof = solver.solve()
            with open(proof_file_path, "w") as f:
                f.write(proof)

#@dataclass
class AOStarSingleFileSolver(AOStarBatchSolver):
    # !! TODO: look up how to derive a dataclass
    #solver_type: Type[AOStarTheoremSolver]
    #project_root: str
    #lean_file_path: str
    def __init__(self, solver_type: Type[AOStarTheoremSolver], project_root: str, lean_file_path: str):
        super().__init__(solver_type=solver_type, project_root=project_root)
        self.lean_file_path = lean_file_path # !! TODO: move to dataclass class vars?
    #def __post_init__(self):
        with open(self.lean_file_path, "r") as f:
            self.lean_file_contents = f.read()
        self.lean_file_name = self.lean_file_path.split("/")[-1] # !! TODO: is there a better way (presumably from os) to do this?
        # Preliminary check that the lean file compiles. This is necessary because
        # self.all_theorems will depend on each theorem in the lean file
        # being complete (i.e., having `theorem` (or example), `begin` and `end`)
        _, preliminary_run_messages = run_proof_on_lean(
            self.lean_file_contents,
            project_root = project_root,
            max_memory_in_mib = 3000
        )
        assert all(not msg.level == 'error' for msg in preliminary_run_messages), f"Problems in the theorem statement:\n{preliminary_run_messages}"

    def find_imports(self) -> str:
        import_regex = r"^[ \t]*import(?:[ \t]+(?:\w+(?:\.\w+)*))+[ \t]*$"
        import_match = re.compile(import_regex, re.MULTILINE)
        return '\n'.join(import_match.findall(self.lean_file_contents))

    def all_theorems(self) -> Iterator[Tuple[str, str, str]]: # !! TODO: or Generator?
        # regex adapted from copra lean_executor.py
        theorem_regex = r"(((theorem\s+([\w+|\d+]*))|example)(?:\s+?)([\S|\s]*?):=[\S|\s]*?)(begin|by|calc)"
        theorem_match = re.compile(theorem_regex, re.MULTILINE)
        """
        Theorems and examples are matched like the following:
        "example : c * b * a = b * (a * c) := by" -> (f1='example : c * b * a = b * (a * c) := ', 'example', '', '', ' : c * b * a = b * (a * c) ', 'by')
        "theorem add_zero (a : R) : a + 0 = a := by" -> ('theorem add_zero (a : R) : a + 0 = a := ', 'theorem add_zero', 'theorem add_zero', 'add_zero', ' (a : R) : a + 0 = a ', 'by')
        """

        imports = self.find_imports()
        # !! TODO: what's the difference btwn a generator & an iterator? Is the following for valid?
        # !! TODO: figure out how to use a Match object
        #for thm_wo_import, _, _, thm_name, _, _ in theorem_match.finditer(self.lean_file_contents):
        for thm_wo_import, _, _, thm_name, _, _ in theorem_match.findall(self.lean_file_contents):
            thm_statement = imports + '\n' + thm_wo_import
            # !! TODO: does the following yield tuple syntax work?
            yield thm_statement, self.lean_file_name, thm_name

#@dataclass
class AOStarYAMLSelectedSolver:
    pass
# !! TODO: implement

if __name__ == "__main__":
    if False:
        # Test driving code for AOStarBFSTheoremSolver
        theorem_statement = "theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :="
        project_root = 'testbed'
        load_file_path = None
        logger = logging.getLogger(__name__)
        log_file_path = "logs/aostarwrapper_test.log"
        logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG, filemode="w")
        load_file_path = "logs/aostarwrapper_test.pth.tar"
        dump_file_path = "logs/aostarwrapper_test.pth.tar"
        solver = AOStarBFSTheoremSolver(theorem_statement, project_root, logger, load_file_path, dump_file_path)
        proof_str = solver.solve()
        if proof_str:
            print(proof_str)
            logger.info("The discovered proof: \n" + proof_str)
    elif True:
        # Test driving code for AOStarSingleFileSolver
        lean_file_path = "/home/billion/Projects/aostar-rewritten/testbed/src/simple2.lean"
        project_root = 'testbed'
        #solver = AOStarBFSTheoremSolver
        solver = AOStarDummyTheoremSolver
        batch_solver = AOStarSingleFileSolver(solver, project_root, lean_file_path)
        batch_solver.solve_all()