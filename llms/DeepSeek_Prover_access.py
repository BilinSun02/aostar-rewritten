#!/usr/bin/env python3
# Uses DeepSeek-Prover v1.5
# !!!TODO: (1) move the model file to the aostar-rewritten dir
# !!!TODO: (2) write README for setting up the env with ds support

from .common import LLMAccess, CostCircuitBreak
from vllm import LLM, SamplingParams

class DeepSeekProverAccess(LLMAccess):
    incurs_cost: bool = False

    def __init__(self) -> None:
        model_name = "../2dsmodel/DeepSeek-Prover-V1.5/deepseek-ai/DeepSeek-Prover-V1.5-RL" # !!TODO: move
        self.model = LLM(model=model_name, max_num_batched_tokens=8192, seed=1, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=2048,
            top_p=0.95,
            n=1,
        )

    def complete(self,
        prompt: str,
        max_tokens: int = 1000
    ) -> str:
        model_outputs = model.generate(
            prompt,
            sampling_params,
            use_tqdm=True,
        )
        return model_outputs[0].outputs[0].text

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

    ds_access = DeepSeekProverAccess(model_name)
    print(ds_access.complete(prompt))
    pass
