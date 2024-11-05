# Don't forget to format `language`
prompt_message_introduction = \
"""
You are a proficient formal theorem-proving agent in {language}. You can predict the next proof step given the current proof state.

"""

prompt_message_input_format = \
"""
The proof state is described in the following format:
1. A [PROOF] section contains a given partial proof leading to the current proof state.
2. A [GOALS] section collects the unsolved goals, each goal started by [GOAL].
3. Each goal comes with a [HYPOTHESES] section of zero or more [HYPOTHESIS] lines. Two optional sections [DEFINITIONS] and [THEOREMS] include possibly relevant definitions and theorems.
4. An [AVOID STEPS] section collects proof steps that you should avoid. This section has zero or more independent [STEP] lines. Each step optionally comes with an [ERROR] message.

"""

prompt_message_output_format_wo_thoughts = \
"""
Your response should consist of one proof step attempt, e.g., "[RUN TACTIC]induction c,[END TACTIC]".
Do NOT aim to produce a full proof in [RUN TACTIC].Aim to make the step in [RUN TACTIC] minimal. For instance, "[RUN TACTIC]rw [h'],[END TACTIC]" is preferred over "[RUN TACTIC] rw [h', ← mul_assoc, h, mul_assoc],[END TACTIC]" and "[RUN TACTIC]rw [h'], simp,[END TACTIC]".
If you are very certain the goal cannot be proven (e.g. "1 % 2 = 0") without an equally wrong hypothesis that might have allowed you to use "exfalso", then you may use "sorry".
You cannot assume any library not imported in the piece given to you. You may optionally include one [IMPORT] statement, e.g. "[IMPORT]import tactic.linarith[END IMPORT]", after [END TACTIC].

"""

prompt_message_output_format_with_thoughts = \
"""
Your response should consist of one proof step attempt. Start with a section begun with [THOUGHTS] and ending with [END THOUGHTS], in which you rephrase the goal in natural language, reflect over why each failed attempt in [AVOID STEPS] failed, and informally discuss how you would correctly approach the goal, leading up to a concrete tactic. This section will not be read by others and can be as concise as you yourself can understand. Then write up a section begun with [RUN TACTIC] and ending with [END TACTIC], in which you provide one tactic to advance the current proof state. For example, "[THOUGHTS]The goal states that the sum of 1 to $n$ equals $\\frac{n(n+1)}{2}$. No failed attempts yet. No axiom apparently applicable. Will try induction.[END THOUGHTS]\n[RUN TACTIC]induction n,[END TACTIC]".
You may plan ahead for multiple tactics in [THOUGHTS], but do NOT aim to produce a full proof in [RUN TACTIC].Aim to make the step in [RUN TACTIC] minimal. For instance, "[RUN TACTIC]rw [h'],[END TACTIC]" is preferred over "[RUN TACTIC] rw [h', ← mul_assoc],[END TACTIC]" and "[RUN TACTIC]rw [h'], simp,[END TACTIC]".
If you are very certain the goal cannot be proven (e.g. "1 % 2 = 0") without an equally wrong hypothesis that might have allowed you to use "exfalso", then you may use "sorry".
You cannot assume any library not imported in the piece given to you. You may optionally include one [IMPORT] statement, e.g. "[IMPORT]import tactic.linarith[END IMPORT]", after [END TACTIC].

"""

# TODO: the prompt still needs improvement.
# 1. The example seems to mislead GPT to use induction for a problem as simple as a+b=b+a. Either add more explanation or simply remove the example.
# 2. GPT fails to be very concise, and still produces full English sentences. See what we can do.
# 3. When think_aloud is enabled, GPT tends to produce too many steps at one time which can easily fail to compile. Even if I include "The step may have slightly more than one tactic, but not more than three; there's no need to solve the problem in one step.", it sometimes produces a lot of tactics in one step. Hence I have to tell it to only have one tactic per step for now.
# 4. This hardcodes "sorry" to mean "the goal was abandoned". Un-hardcode this in the future if we need to use "sorry" in the future.

#response_token_limit = 4096 # Maximum GPT-4o supports. Interestingly GPT-4o-mini supports more.
prompt_message_token_limit = "Make sure your response do not exceed {response_token_limit=}."
# Don't forget to format `response_token_limit`