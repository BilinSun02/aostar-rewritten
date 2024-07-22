import logging
from typing import List, Tuple, Type
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import aostaralgorithm
from aostaralgorithm import ao_star
import datetime
import os
from lean_cmd_executor_aostar import run_proof_on_lean
import re

@dataclass
class AOStarSolver(ABC):
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
    # Note that while AOStarSolver.heuristic has signature Callable[[AOStarSolver, Node], float], while
    # self.heuristic on any instance self has signature Callable[[Node], float] according to Python's
    # method objects' __get__() rules, so self.heuristic can be passed to a Node but not AOStarSolver.heuristic

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
        in AOStarSolver would work, because when ao_star is imported, it will use
        variables from the aostaralgorithm *module* scope, *not* the global scope.
        Hence, we have to inject the variables into the module scope.
        """
        aostaralgorithm.heuristic = self.heuristic
        return ao_star(self.theorem_statement)

class AOStarBFSSolver(AOStarSolver):
    def heuristic(self, node: aostaralgorithm.Node) -> float:
        return 1

@dataclass
class AOStarBatchSolver:
    lean_file_path: str
    solver_type: Type[AOStarSolver]
    project_root: str

    def __post_init__(self):
        self.identifier_str = datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
        self.main_log_dir = f"logs/run_{self.identifier_str}"
        os.makedirs(self.main_log_dir, exist_ok=True)
        self.main_checkpoint_dir = f"checkpoints/run_{self.identifier_str}"
        os.makedirs(self.main_checkpoint_dir, exist_ok=True)

        main_log_path = self.main_log_dir + "main.log"
        self.main_logger = logging.getLogger(__name__)
        logging.basicConfig(filename=main_log_path, encoding='utf-8', level=logging.DEBUG, filemode="w")
        
        with open(self.lean_file_path, "r") as f:
            self.lean_file_contents = f.read()
        # Preliminary check that the lean file compiles
        # self.find_all_theorems will depend on each theorem in the lean file
        # being complete (i.e., having `theorem` (or example), `begin` and `end`)
        _, preliminary_run_messages = run_proof_on_lean(self.lean_file_contents, project_root=project_root, max_memory_in_mib=3000)
        assert all(not msg.level == 'error' for msg in preliminary_run_messages), f"Problems in the theorem statement:\n{preliminary_run_messages}"
        self.all_theorems_parsed = self.find_all_theorems()
        self.imports = self.find_imports()

    def find_imports(self) -> str:
        import_regex = r"^[ \t]*import(?:[ \t]+(?:\w+(?:\.\w+)*))+[ \t]*$"
        import_match = re.compile(import_regex, re.MULTILINE)
        return '\n'.join(import_match.findall(self.lean_file_contents))

    def find_all_theorems(self) -> List[Tuple[str,str,str,str,str,str]]:
        # Parts come from copra lean_executor.py
        theorem_regex = r"(((theorem ([\w+|\d+]*))|example)([\S|\s]*?):=[\S|\s]*?)(begin|by|calc)"
        theorem_match = re.compile(theorem_regex, re.MULTILINE)
        """
        Theorems and examples are matched like the following:
        "example : c * b * a = b * (a * c) := by" -> (f1='example : c * b * a = b * (a * c) := ', 'example', '', '', ' : c * b * a = b * (a * c) ', 'by')
        "theorem add_zero (a : R) : a + 0 = a := by" -> ('theorem add_zero (a : R) : a + 0 = a := ', 'theorem add_zero', 'theorem add_zero', 'add_zero', ' (a : R) : a + 0 = a ', 'by')
        """
        return theorem_match.findall(self.lean_file_contents)

    def solve_all(self):
        for idx, tup in enumerate(self.all_theorems_parsed):
            theorem_statement, theorem_head, _, theorem_name, _, _ = tup
            if theorem_name:
                log_file_name   = str(idx) + " " + theorem_name + ".log"
                proof_file_name = str(idx) + " " + theorem_name + ".lean"
                dump_file_name  = str(idx) + " " + theorem_name + ".pth.tar"
            else:
                log_file_name   = str(idx) + ".log"
                proof_file_name = str(idx) + ".lean"
                dump_file_name  = str(idx) + ".pth.tar"
            log_path = os.path.join(self.main_log_dir, log_file_name)
            logger = logging.getLogger(str(idx))
            logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG, filemode="w")
            solver = self.solver_type(
                theorem_statement = self.imports + '\n' + theorem_statement,
                project_root = self.project_root,
                logger = logger,
                load_file_path = None,
                dump_file_path = dump_file_name
            )
            proof = solver.solve()
            with open(proof_file_name, "w") as f:
                f.write(proof)

if __name__ == "__main__":
    if False:
        # Test driving code for AOStarBFSSolver
        theorem_statement = "theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :="
        project_root = 'testbed'
        load_file_path = None
        logger = logging.getLogger(__name__)
        log_path = "logs/aostarwrapper_test.log"
        logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG, filemode="w")
        load_file_path = "checkpoints/aostarwrapper_test.pth.tar"
        dump_file_path = "checkpoints/aostarwrapper_test.pth.tar"
        solver = AOStarBFSSolver(theorem_statement, project_root, logger, load_file_path, dump_file_path)
        proof_str = solver.solve()
        if proof_str:
            print(proof_str)
            logger.info("The discovered proof: \n" + proof_str)
    else:
        # Test driving code for AOStarBatchSolver
        lean_file_path = "/home/billion/Projects/aostar-rewritten/testbed/src/simple.lean"
        project_root = 'testbed'
        batch_solver = AOStarBatchSolver(lean_file_path, AOStarBFSSolver, project_root)
        batch_solver.solve_all()