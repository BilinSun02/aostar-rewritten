# This file is only intended to be a demo and is not a dependency of any other file.
from gpt_access import GptAccess

template_message = [
        {
            "role": "system",
            "content": \
"""
You are a proficient formal theorem-proving agent in Lean 3. You can predict the next proof step given the current proof state. The proof state is described in the following format:
1. All the goals are described under `[GOALS]` keyword. Each goal in this `[GOALS]` section is started by `[GOAL] i`, where i is a positive integer. Under each `[GOAL]`, the goal is described as a human-readable serialized version of the proof state as shown while running lean command.
2. Each goal comes with zero or more hypotheses in a section that starts with `[HYPOTHESES]`. This section consists of zero or more `[HYPOTHESIS]` lines, each of which starts one hypothesis. Two optional sections `[DEFINITIONS]` and `[THEOREMS]` describe the relevant definitions of symbols used in that goal and some theorems or lemmas that might help simplify the goal. Each definition under `[DEFINITIONS]` starts with the prefix `[DEFINITION] i`. Each theorem/lemma within `[THEOREMS]` starts with the prefix `[THEOREM]`.
3. Optional lines started by `[INFORMAL-THEOREM]` and `[INFORMAL-PROOFS]` describe the proofs and theorems in natural language. The proof is for the whole theorem along with the theorem statement rather than just the proof state. For example, `[INFORMAL-THEOREM]\nThe sum of two even numbers is even.\n[INFORMAL-PROOFS]\nSuppose a and b are even numbers. Then there exist integers m and n such that a = 2 * m and b = 2 * n. Then a + b = 2 * m + 2 * n = 2 * (m + n). Since m + n is an integer, a + b is even...` This is for intuitive understanding only and may not even always be correct. Any of your work should still be based solely on the proof state as started by `[GOALS]`.
4. Finally, a `[STEPS]` section describes proof steps used so far. Each proof step starts with the prefix `[STEP]` (note the absense of an `i` index), and is a valid Lean proof. For example, `[STEPS][STEP]rw h₁ at h₂,[STEP]{linarith},`. Do not generate `[STEP]` in *your* response as it is only used to tell you how far the proofs have progressed.
5. An optional `[INCORRECT STEPS]` section collects proof steps that should NOT be generated. Use this as a hint for not generating these proof steps that are known to fail. For example, `[INCORRECT STEPS][STEP]apply h₁,[STEP]rw ←h₁`. **DO NOT** generate these `[INCORRECT STEPS]` under the same states. If the states have changed, you may regenerate them again.
6. An optional `[LAST STEP]` line describes the proof step generated last time. If the proof step was incorrect, then it is also followed by error message from Lean environment. For example, `[LAST STEP]linarith,\n[ERROR MESSAGE]linarith failed to find a contradiction\nstate:\nx y : ℝ,\nh₁ : x = 3 - 2 * y,\nh₂ : 2 * x - y = 1\n⊢ false`. If the proof-step was correct then it is followed by the keyword `[SUCCESS]`. For example, `[LAST STEP]linarith,[SUCCESS]`. Don't generate the last proof step again if it was NOT successful.
7. Sometimes there can be errors in the format of the generated response. This is reported using the keyword `[ERROR]` followed by the error message. For example, `[ERROR]\nInvalid response:\n'Great! The proof is complete.', \nStopping Reason: 'stop'.\n Please respond only in the format specified.[END]`. This means that the response generated by you was not in the specified format. Please follow the specified format strictly.
8. Try not to prove a theorem in one shot. An optional `[EXPAND NUM]` value specifies the number of proofs to try. If `[EXPAND NUM]` is not supplied, provide one proof.
9. One goal may be set to be distinguished using the alternative keyword `[FOCUSED GOAL]`. In this case, completely focus on proving the `[FOCUSED GOAL]`, and do not generate anything not directly related to the `[FOCUSED GOAL]`.

Your response should consist of distinct proofs, the quantity of which is specified by `[EXPAND NUM]` as previously discussed. Syntactically, it consists of `[RUN TACTIC]` followed by the proof steps that will help progressively simplify the current proof state, then followed by `[END]`. For example, `[RUN TACTIC]induction c,[END]`.

Please take a note of the following: 
1. Follow the specified format strictly. Your only response will be to generate [RUN TACTIC] lines, and make sure to end each such line with the keyword [END]. Make sure you are not generating any other keywords in between [RUN TACTIC] and [END] such as [STEP].
2. Make sure to always capitalize the keywords, i.e., a [RUN TACTIC] is right but [Run Tactic] is strictly wrong.
3. Proof steps must compile correctly in Lean 3, and each proof should compile on its own. In particular, one proof cannot reference hypotheses or variables defined in another proof.
"""
        },
        {
            'role': 'user',
            'content': \
"""
Goals to prove:
[GOALS]
[GOAL] 1
x % 2 = 0 → x * x % 2 = 0
[HYPOTHESES]
[HYPOTHESIS] x : ℕ

[INFORMAL-THEOREM]
theorem mod_arith_1
(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=

[INFORMAL-PROOF]
To prove this theorem, we will use the property that if x is even, then x * x is also even. An even number is defined as a number that is divisible by 2, which means it can be expressed as 2 * k for some integer k. The statement x % 2 = 0 asserts that x is even.

Let's proceed with the proof in Lean:
```lean
theorem mod_arith_1 (x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
  -- Assume x is even, i.e., x % 2 = 0
  intro h,
  -- Since x is even, there exists some k such that x = 2 * k
  have k_def : ∃ k, x = 2 * k := exists_eq_mul_right_of_dvd (nat.dvd_of_mod_eq_zero h),
  -- Let's use this k to express x
  cases k_def with k hk,
  -- Now we rewrite x as 2 * k and expand (2 * k) * (2 * k)
  rw hk,
  -- After expansion, we get 4 * k * k
  calc (2 * k) * (2 * k) = 4 * (k * k) : by ring
  ... = 2 * (2 * (k * k)) : by rw ←mul_assoc
  -- The result is clearly a multiple of 2, hence it is even
  ... % 2 = 0 : by rw nat.mul_mod_right
end
```
In this proof, we first introduce our assumption that x is even. Then we express x as 2 * k for some k using the fact that x is divisible by 2. We then rewrite x in terms of k and expand the expression (2 * k) * (2 * k) to 4 * k * k, which is clearly a multiple of 2. Finally, we conclude that (x * x) % 2 = 0, which completes the proof.

[THEOREMS] 1
[THEOREM] complex.sin_two_pi :  sin (2 * π) = 0
[THEOREM] nat.digits_aux_zero : (b : ℕ) (h : 2 ≤ b) : digits_aux b h 0 = []
[THEOREM] int.mod_two_ne_one :  ¬ n % 2 = 1 ↔ n % 2 = 0
[THEOREM] int.mod_two_ne_zero :  ¬ n % 2 = 0 ↔ n % 2 = 1
[THEOREM] nat.mod_two_ne_one :  ¬ n % 2 = 1 ↔ n % 2 = 0
[THEOREM] nat.mod_two_ne_zero :  ¬ n % 2 = 0 ↔ n % 2 = 1
[THEOREM] nat.eq_zero_of_mul_eq_zero :  ∀ {n m : ℕ}, n * m = 0 → n = 0 ∨ m = 0 | 0        m
[END]
[FOCUSED GOAL]: hypotheses: x : ℕ goal: x % 2 = 0 → (x * x * x) % 2 = 0
[EXPAND NUM]: 5
"""
        },
    ]

openai_access = GptAccess(model_name="gpt-3.5-turbo")
print("printing complete chat:")
print(openai_access.complete_chat(template_message, max_tokens=100, n=2, temperature=0.8))
