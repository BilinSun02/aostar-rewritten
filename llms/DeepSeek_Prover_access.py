#!/usr/bin/env python3
# Uses DeepSeek-Prover v1.5
# !!!TODO: (1) move the model file to the aostar-rewritten dir
# !!!TODO: (2) write README for setting up the env with ds support

from .common import LLMAccess, CostCircuitBreak
from vllm import LLM, SamplingParams
from rpc import RPCServer, RPCClient
import socket
from contextlib import closing

class DeepSeekProverRPCServer(RPCServer):
    def __init__(self) -> None:
        model_name = "../2dsmodel/DeepSeek-Prover-V1.5/deepseek-ai/DeepSeek-Prover-V1.5-RL" # !!TODO: move
        self.model = LLM(
            model = model_name,
            max_num_batched_tokens = 8192,
            seed = 1,
            trust_remote_code = True
        )
        self.sampling_params = SamplingParams(
            temperature = 1.0,
            max_tokens = 2048,
            top_p = 0.95,
            n = 1,
        )

    def process(self, s: str) -> str:
        model_outputs = self.model.generate(
            prompt,
            self.sampling_params,
            use_tqdm = True,
        )
        return model_outputs[0].outputs[0].text

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
        self.rpc_server = DeepSeekProverRPCServer( # !!!!TODO: move to subprocess
            host = 'localhost',
            port = self.port
        )
        self.rpc_client = RPCClient(
            host = 'localhost',
            port = self.port
        )

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
