#!/usr/bin/env python3
# Uses DeepSeek-Prover v1.5
# !!!TODO: (1) move the model file to the aostar-rewritten dir
# !!!TODO: (2) write README for setting up the env with ds support

from .common import LLMAccess
from rpc import RPCClient
import socket
from contextlib import closing
import subprocess

# Taken from https://stackoverflow.com/a/45690594
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class DeepSeekProverAccess(LLMAccess):
    incurs_cost: bool = False

    def __init__(self) -> None:
        super().__init__("DeepSeekProverAccess")
        self.port = find_free_port()
        self.rpc_server_process = subprocess.Popen(
            f"python -m llms.DeepSeek_Prover_server --port {self.port}",
            shell = True
            # Won't block because stdout=stderr=stdin=None
        )
        self.rpc_client = RPCClient(
            host = 'localhost',
            port = self.port
        )

    def complete(self, prompt: str) -> str:
        return self.rpc_client.process(prompt)

if __name__ == "__main__":
    prompt = r'''/-- This is a complete Lean 4 proof written by an expert,
interspersed with thoughts kept as comments. --/
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
'''

    ds_access = DeepSeekProverAccess()
    print(ds_access.complete(prompt))
    pass
