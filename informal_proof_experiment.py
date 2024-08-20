from gpt_access import *

#openai_access = GptAccess(model_name="gpt-4o-mini")
# gpt-4o-mini would tell me that 104 ≥ 1000. Let's not use it.
#openai_access = GptAccess(model_name="gpt-4o-2024-08-06")
openai_access = GptAccess(model_name="gpt-4")
# print(openai_access.get_models())
messages = [
    {
        "role": "system",
        "content": """
You are a professional mathematics assistant who interprets a Lean goal in English and reasons in natural language step-by-step to discover a proof.
Keep each "step" focused on as small a topic as possible. Label each step with a number. Try to make each new step self-complete by completely stating what the step aims to solve or builds upon, by referring to previous steps.
Make bold intermediate conjectures and attempt to prove them, but abandon those that are false, and shelve those that are too difficult or unlikely to contribute. Come back to the shelved steps if work on other steps also turn difficult or aimless.
When the Lean goal is finished, end the steps and write a informal summary on how to distill the steps into a proof.
Then write a proof in natural language that does not reference the steps above.
Make sure your proof can be transcribed line-by-line into a Lean proof: for each line (or each few lines), suggest a Lean 3 tactic that represents what's going on, such that if we collect all the tactics, they should compile as a Lean proof. Any informal language or reasoning step that can't be easily transcribed into a Lean tactic is not allowed; common examples include "and so on": except perhaps with careful rewriting into casework or induction, there is no corresponding Lean tactic, even if what to do is clear to a human. Avoid such language that fails to be specific and rigorous, even if this necessitates pedantic language.
"""
    },
    {
        "role": "system",
        "name": "example_user",
        "content": """
f : ℕ → ℕ → ℝ,
h₀ : ∀ (x : ℕ), 0 < x → f x x = ↑x,
h₁ : ∀ (x y : ℕ), 0 < x ∧ 0 < y → f x y = f y x,
h₂ : ∀ (x y : ℕ), 0 < x ∧ 0 < y → (↑x + ↑y) * f x y = ↑y * f x (x + y)
⊢ f 14 52 = 364
""",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": r"""
[INFORMAL STEPS]
(1) The goal involves a function $f$ on two positive integers.
(2) h₀ states $f(x, x) = x$.
(3) h₁ states the function is symmetric in its two arguments.
(4) h₂ is a functional equation relating $(x + y) * f(x, y)$ and $y * f(x, (x + y))$.
(5) We need to show that $f(14, 52) = 364$.
(6) As (2), (3) and (4) do not directly define $f$ but only describe $f$, it's not clear whether there's one unique $f$ satisfying (2), (3) and (4) or there is a family of satisfactory functions. We take the tentative route of guessing what $f$ can be, and then determine whether the guessed result is unique.
(7) Since $364 = 14 * 52 / 2$, we conjecture that $f$ is a quadratic form.
(8) However, given (2), (7) can't be true.
(9) Another function that produces the same result is $f = \lcm$, the least common multiple.
(10) The guess (9) is also in line with (2) and (3).
(11) It is not immediately clear whether (9) satisfies (4).
(12) We attempt to show that (9) satisfies (4).
(13) To establish (12), we need to show that $(x + y) * \lcm(x, y) = y * \lcm(x, (x + y))$.
(14) It would be easier if we can express $\lcm(x, y)$ in another way so the left hand side and the right hand side can be manipulated and simplified.
(15) If we can express $\lcm(x, y)$ as a product with the factor $y$, then the $y$ can be cancelled from both sides of the equation.
(16) For (15), we can write $\lcm(x, y) = x * y / \gcd(x, y)$.
(17) Substitute (16) into (4), and we get $(x + y) * x * y / \gcd(x, y) = y * \lcm(x, (x + y))$.
(18) Cancel $y$ from (17) and we get $(x + y) * x / \gcd(x, y) = \lcm(x, (x + y))$.
(19) Similar to (16), we can express $\lcm(x, (x + y))$ in the right hand side as $x * (x + y) / \gcd(x, (x + y))$, which looks promising since the result involves $x$ and $x + y$ which are in the left hand side.
(20) Substitute (19) into (18), and we get $(x + y) * x / \gcd(x, y) = x * (x + y) / \gcd(x, (x + y))$.
(21) Cancel $x$ and $x + y$ from (20), and we get $\gcd(x, y) = \gcd(x, (x + y))$. To prove (13), it suffices to show this.
(22) To establish (21), it suffices to show that $d \mid x \wedge d \mid y \Leftrightarrow d \mid x \wedge d \mid (x + y)$.
(23) To establish the $\Rightarrow$ part in (22), we need to show that if $d \mid x$ and $d \mid y$ then $d \mid x$ and $d \mid (x + y)$.
(24) $d \mid x$ in (23) is in the hypotheses, so to establish (23), we only need to show that, if $d \mid x$ and $d \mid y$ then $d \mid (x + y)$.
(25) To establish (24), we note that $d \mid x$ implies $x = d * m$ for some integer multiple $m$.
(26) Similar to (25), $d \mid y$ implies $y = d * n$ for some integer multiple $n$.
(27) Add together (25) and (26), and we get $(x + y) = d * (m + n)$.
(28) As $m$ and $n$ from (25) and (26) are integers, we get that $m + n$ in (27) is also an integer.
(29) (27) and (28) tegether show that $d \mid (x + y)$.
(30) (29) establishes (25), which establishes (24), which establishes (23), which establishes the $\Rightarrow$ part of (22). The $\Leftarrow$ part of (22) remains.
(31) To establish the $\Leftarrow$ part in (21), we need to show that if $d \mid x$ and $d \mid (x + y)$ then $d \mid x$ and $d \mid y$.
(32) $d \mid x$ in (31) is in the hypotheses, so to establish (31), we only need to show that, if $d \mid x$ and $d \mid (x + y)$ then $d \mid y$.
(33) To establish (32), we note that $d \mid x$ implies $x = d * m$ for some integer multiple $m$.
(34) Similar to (33), $d \mid (x + y)$ implies $(x + y) = d * n$ for some integer multiple $n$.
(35) Subtract (33) from (34), and we get $y = d * (n - m)$.
(36) As $m$ and $n$ from (33) and (34) are integers, we get that $n - m$ in (35) is also an integer.
(37) (35) and (36) tegether show that $d \mid y$.
(38) (37) establishes (33), which establishes (32), which establishes (31), which establishes the $\Leftarrow$ part of (22).
(39) (30) and (38) tegether establish (22) in full. (22) establishes (21), which establishes (13), which establishes (12), which states that (9) satisfies (4).
(40) As discussed in (10) and (30), (9) satisfies all of (2), (3) and (4). Thus, the function guessed in (9), $\lcm$, is a valid guess for (6). We continue along (6) and attempt to determine whether it is unique.
(41) For (40), we first tentatively consider the possibility that $\lcm$ is unique, i.e. there are no other functions satisfying (2), (3) and (4).
(42) To establish (41), we need to show from (2), (3) and (4) that $f$ necessarily have the same output as $\lcm$ on all inputs.
(43) To establish (42), it helps to reduce the case on larger inputs to the case on smaller inputs, whether we explicitly resort to induction.
(44) For (43), (2) does not seem helpful as it does not reduce inputs. It may only possibly be used as a base case, so we shelve this for now.
(45) For (43), (3) also does not seem helpful as it does not reduce inputs. We shelve this for now.
(46) For (43), (4) also does not seem helpful, because the relation actually relates outputs on small inputs to outputs on even larger inputs.
(47) From (44), (45) and (46), it seems that all three clues we have towards proving (43) are hard to follow. Therefore, (41) seems to be difficult. It's possible that the negation of (41) is true, so we should attempt that too.
(48) As per (47), we try to guess another function that satisfy (2), (3) and (4).
(49) For (48), another function that satisfies (2) "idempotence" and (3) symmetry is the first-order polynomial $f(x, y) = (x + y) / 2$. Verifying (4) involves some calculation so let's shelve it. What else can we say about this degree-1 polynomial?
(50) Building on (49), an observation is that $f(x, y) = (x + y) / 2$ from (49) obviously does not satisfy (5); the output fall far behind. If it does satisfy (4), then it would be a counterexample to (5). We look more closely at whether it satisfies (4).
(51) For (50), we substitute $f(x, y) = (x + y) / 2$ into (4). This gives $(x + y) * (x + y) / 2 \overset{?}{=} y * (x + (x + y)) / 2$. This equality clearly does not hold. Thus, $f(x, y) = (x + y) / 2$ does not satisfy (4).
(52) (51) shows (49) and (50) do not give a counterexample. While not a proof, this hints at the correctness of (5), unless more counterevidence appears. When looking for new guesses, we consider checking whether the function satisfies (5), 
(53) (52) suggests we should get back to (48) and try again.
(54) However, (48) now seems to be difficult, because a function that simultaneously satisfying (1) and (5) is likely not a polynomial unless we fit a polynomial really hard. A polynomial obtained that way isn't likely to pass (4) anyway. We should probably not look for polynomials.
(55) In addition to (54), we note that $\lcm$ is not a paramtrized function, and so cannot be easily tweaked into a different function that satisfies (2), (3) and (4).
(56) Given the lack of progress during (49) through (55), we conclude that (48) is more difficult than we expected.
(57) (56) suggests that we shelve the possibility of (48) and reconsider (41) which (48) was trying to avoid.
(58) For (57), (2) and (3) still don't seem to be able to contribute any nontrivial manipulation on the input, but (4) does, so we focus on (4).
(59) Building on (58) and (43), we consider if it's possible to transform (4) to relate inputs to **smaller** inputs.
(60) The symmetry of the equality relation comes to our aid. (59) is actually possible if we view (4) backwards: when we read (4) as $y * f(x, (x + y)) = (x + y) * f(x, y)$, we relate the larger input $(x, (x + y))$ to the smaller input $(x, y)$.
(61) Letting $z = x + y$, the relation in (60) can be written as $(z - x) * f(x, z) = z * f(x, (z - x))$, or $f(x, z) = (z / (z - x)) * f(x, (z - x))$. This gives us a way to calculate the output on larger inputs from the output on smaller inputs.
(62) We may repeat the transformation in (61) to get $f(x, z) = (z / (z - x)) * f(x, (z - x)) = (z / (z - x)) * ((z - x) / (z - 2x)) * f(x, (z - 2x)) = (z / (z - 2x)) * f(x, (z - 2x))$, and so on, until the second operand is a positive integer smaller than the first; but then we can appeal to (3) to swap the operands, so we can further reduce the inputs.
(63) The process described in (62) will only have to stop when the two operands are equal, whence subtracting the first ooperand from the second results in 0, an input that is not governed by (3) and (4).
(64) Hence the function output at the stopping condition outlined in (63) completely determines the output at any other input (which can be reduced according to 62). Yet the output at (63), i.e. when both operands are equal, is already completely determined by (2).
(65) From (64), we conclude that the function output is in fact completely determined by (2), (3) and (4) for every input. Therefore, $\lcm$ is indeed the only valid value for $f$.
(66) (65) shows that we only need to check $f = \lcm$ for (5), which is a trivial calculation. We are done.

[INFORMAL SUMMARY]
In hind sight, since the inputs in (5) are small, we can in fact get away by just applying the algorithm outlined by (61) through (63), without bothering to show that $f$ is $\lcm$.

[PROOF]
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
    {
        "role": "user",
        "content": r"""
theorem aime_1984_p7
  (f : ℤ → ℤ)
  (h₀ : ∀ n, 1000 ≤ n → f n = n - 3)
  (h₁ : ∀ n, n < 1000 → f n = f (f (n + 5))) :
  f 84 = 997 :=
"""
    },
]
print("printing complete chat:")
#print(openai_access.complete_chat(messages, max_tokens=16384, n=1, temperature=0))
print(openai_access.complete_chat(messages, max_tokens=4096, n=1, temperature=0))