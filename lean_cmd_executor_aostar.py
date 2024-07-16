from lean_cmd_executor import Lean3Executor
from lean_parse_utils import LeanLineByLineReader
import logging

class LeanOneoffExec:
    def __init__(self, file_path: str, project_root: str = '.'):
        self.lean_stdin_reader = LeanLineByLineReader(file_path)
        self.lean_exec : Lean3Executor = Lean3Executor(
            project_root=project_root,
            use_human_readable_proof_context=True, 
            proof_step_iter=self.lean_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.lean_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.lean_exec.__exit__(exc_type, exc_value, traceback)
    
    def __call__(self):
        self.lean_exec.run_to_finish()
        return self.lean_exec.proof_context.all_goals

if __name__ == "__main__":
    logging.basicConfig(filename='lean_executor.log', filemode='w', level=logging.INFO)
    #os.chdir(root_dir)
    project = "data/test/lean_proj"
    file = "data/test/lean_proj/src/simpler.lean"
    with LeanOneoffExec(file, project) as lean_exec:
        print(lean_exec())
