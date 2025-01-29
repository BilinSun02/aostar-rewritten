#!/usr/bin/env python3
# Uses DeepSeek-Prover v1.5
# !!!TODO: (1) move the model file to the aostar-rewritten dir
# !!!TODO: (2) write README for setting up the env with ds support

import socket
from contextlib import closing
import subprocess

from .common import LLMAccess
from .rpc import RPCClient

DSPROVER_DEFAULT_PORT = 6626 # Screw "WAGO Service and Update"

# Taken from https://stackoverflow.com/a/45690594
def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class DeepSeekProverAccess(LLMAccess):
    incurs_cost: bool = False

    def __init__(self) -> None:
        super().__init__("DeepSeekProverAccess")
        self.rpc_client = RPCClient(
            host = 'localhost',
            port = DSPROVER_DEFAULT_PORT
        )

        try:
            # Test if there's an existing server running
            self.rpc_client.connect()
            self.rpc_client.disconnect()
            self.rpc_server_process = None
            #print(f"Using existing at port {DSPROVER_DEFAULT_PORT}")
        except Exception as e:
            #print(f"Failed to connect to existing instance: {e.__repr__()}")
            #print("Running new instance")
            self.port = _find_free_port()
            self.rpc_client.port = self.port
            self.rpc_server_process = subprocess.Popen(
                f"python -m llms.DeepSeek_Prover_server "+\
                    f"--port {self.port} --host 'localhost'",
                shell = True,
                # Won't block because the following are set to None:
                stdin = None,
                stdout = None,
                stderr = None,
            )

    def complete(self, prompt: str) -> str:
        return self.rpc_client.query(prompt)

if __name__ == "__main__":
    # Test driving code
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
