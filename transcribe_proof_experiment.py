from gpt_access import *

#openai_access = GptAccess(model_name="gpt-4o-mini")
# gpt-4o-mini would tell me that 104 ≥ 1000. Let's not use it.
openai_access = GptAccess(model_name="gpt-4o-2024-08-06")
# print(openai_access.get_models())
messages = [
    {
        "role": "system",
        "content": """
You are a professional mathematics assistant who converts an informal proof into a Lean 3 proof.
"""
    },
    {
        "role": "system",
        "name": "user",
        "content": r"""
[GOAL]
f : ℕ → ℕ → ℝ,
h₀ : ∀ (x : ℕ), 0 < x → f x x = ↑x,
h₁ : ∀ (x y : ℕ), 0 < x ∧ 0 < y → f x y = f y x,
h₂ : ∀ (x y : ℕ), 0 < x ∧ 0 < y → (↑x + ↑y) * f x y = ↑y * f x (x + y)
⊢ f 14 52 = 364

[INFORMAL PROOF]
Letting $z = x + y$, we may rewrite h₂ as $(z - x) * f(x, z) = z * f(x, (z - x))$, or $f(x, z) = (z / (z - x)) * f(x, (z - x))$. Hence
\[
\begin{align*}
  f(14, 52) &= (52 / (52 - 14)) * f(14, (52 - 14)) \\
            &= (52 / (52 - 14)) * ((52 - 14) / (52 - 28)) * f(14, (52 - 28)) \\
            &= (52 / (52 - 14)) * ((52 - 14) / (52 - 28)) * ((52 - 28) / (52 - 42)) * f(14, (52 - 42)) \\
            &= (52 / (52 - 42)) * f(14, (52 - 42)) \\
            &= (52 / 10) * f(14, 10) \\
            &= (52 / 10) * f(10, 14) \tag*{by h₁}\\
            &= (52 / 10) * (14 / (14 - 10)) * f(10, (14 - 10)) \\
            &= (52 / 10) * (14 / 4) * f(10, 4) \\
            &= (52 / 10) * (14 / 4) * f(4, 10) \tag*{by h₁}\\
            &= (52 / 10) * (14 / 4) * (10 / 2) * f(2, 4) \\
            &= (52 / 10) * (14 / 4) * (10 / 2) * (4 / 2) * f(2, 2) \\
            &= (52 / 10) * (14 / 4) * (10 / 2) * (4 / 2) * 2 \tag*{by h₀}\\
            &= 364
\end{align*}
\]
""",
    },
]
print("printing complete chat:")
print(openai_access.complete_chat(messages, max_tokens=16384, n=1, temperature=0.8))