from typing import List
from gpt_access import GptAccess
import copy
import re

class GPTCircuitBreak(Exception):
    @staticmethod
    def cost_per_1M_tokens(model_name: str) -> int:
        """
        Unit: cents
        """
        # Data from https://openai.com/api/pricing/
        # Accurate as of Aug 10, 2024
        match model_name:
            case "gpt-4o-mini":
                return 15 # cents per million tokens
            case "gpt-4o":
                return 500
            case "gpt-4o-2024-08-06":
                return 250
            case "gpt-4":
                return 3000
            case "gpt-3.5-turbo-0125":
                return 50
            case _: # Default to the cost of the most expensive, gpt-4-32k
                return 6000

    @staticmethod
    def token_limit(model_name: str, budget: int) -> int:
        """
        Calculate admissible token usage within budget (budget unit: cents)
        """
        rate = GPTCircuitBreak.cost_per_1M_tokens(model_name)
        return int((budget / rate) * 1000000)

    @staticmethod
    def calc_cost(model_name: str, n_tokens: int) -> int:
        rate = GPTCircuitBreak.cost_per_1M_tokens(model_name)
        return int(n_tokens * rate / 1000000)

template_messages = [
        {
            "role": "system",
            "content": \
"""
You are a proficient formal theorem-proving agent in Lean 3. You can predict the next proof step given the current proof state. The proof state is described in the following format:
1. All the goals are described under `[GOALS]` keyword. Each goal in this `[GOALS]` section is started by `[GOAL] i`, where i is a positive integer. Under each `[GOAL]`, the goal is described as a human-readable serialized version of the proof state as shown while running lean command.
2. Each goal comes with zero or more hypotheses in a section that starts with `[HYPOTHESES]`. This section consists of zero or more `[HYPOTHESIS]` lines, each of which starts one hypothesis. Two optional sections `[DEFINITIONS]` and `[THEOREMS]` describe the relevant definitions of symbols used in that goal and some theorems or lemmas that might help simplify the goal. Each definition under `[DEFINITIONS]` starts with the prefix `[DEFINITION] i`. Each theorem/lemma within `[THEOREMS]` starts with the prefix `[THEOREM]`.
3. Finally, a `[STEPS]` section describes proof steps used so far. Each proof step starts with the prefix `[STEP]` (note the absense of an `i` index), and is a valid Lean proof. For example, `[STEPS][STEP]rw h₁ at h₂,[STEP]{linarith},`. Do not generate `[STEP]` in *your* response as it is only used to tell you how far the proofs have progressed.
4. An optional `[AVOID STEPS]` section collects proof steps that should NOT be generated. This section has zero or more independent `[STEP]` lines. For example, `[INCORRECT STEPS][STEP]apply h₁,[STEP]rw ←h₁`. Use this as a hint for not generating these proof steps that have been tried previously. **DO NOT** generate these `[INCORRECT STEPS]` under the same states. If the states have changed, you may regenerate them again.
5. Each `[STEP]` under the `[AVOID STEPS]` optionally comes with an `[ERROR]` message. For example, `[ERROR]\nInvalid response:\n'We thus finish the proof.', \nStopping Reason: 'stop'.\n Please respond only in the format specified.[END ERROR]`.

Your response should consist of one proof step attempt. Syntactically, it consists of `[RUN TACTIC]` followed by one proof step that you believe will help advance the current proof state, then followed by `[END TACTIC]`. For example, `[RUN TACTIC]induction c,[END TACTIC]`.
You are not allowed to assume any library not `import`ed in the piece given to you. You have the opportunity to include one `[IMPORT]` statement. For example, `[RUN TACTIC]linarith,[END TACTIC][IMPORT]import tactic.linarith[END IMPORT]`. What you provide in this section will be prepended to the existing proof parts, so make sure it compiles.
"""
        },
        {
            'role': 'user',
            'content': 
"""
"""
        },
    ]

def remove_end_line(string:str) ->str:
    """
    Given a multiline string, remove any line that has only "end" modulo whitespaces
    TODO: check if we do want to remove ALL end lines
    """
    lines = string.split('\n')
    purged_lines = [line for line in lines if line.strip() != "end"]
    purged_string = '\n'.join(purged_lines)
    return purged_string

def prompt_for_tactics(
    goals: str,
    avoid_steps: str = "[AVOID STEPS]",
    n_tactics: int = 5,
    model_name: str = "gpt-4o",
    budget: int = 500
) -> List[str]:
    """
    `goals` should furnish the [GOALS] section
    `avoid_steps` should furnish the [AVOID STEPS] section
    Returns a list of (`tactic`, `necessary_import`) pairs suggested by the LLM.
    Each `tactic` should compile when appended to the existing proof
    once we prepend the `necessary_import` to the existing proof
    """
    #openai_access = GptAccess(model_name="gpt-3.5-turbo")
    openai_access = GptAccess(model_name="gpt-4o-mini")
    messages = copy.deepcopy(template_messages)
    messages[1]["content"] = goals + avoid_steps
    gpt_tactics = []

    while len(gpt_tactics) < n_tactics:
        gpt_response = openai_access.complete_chat(messages, max_tokens=200, n=1, temperature=0.2)

        """
        Memo: a typical GPT response looks like this:
        ([{'role': 'assistant', 'content': '...', 'finish_reason': 'stop'}, {'role': 'assistant', 'content': '...', 'finish_reason': 'stop'}], {'prompt_tokens': 1980, 'completion_tokens': 139, 'total_tokens': 2119, 'reason': 'stop'})
        where the length of the list (the first element of the outermost tuple) is specified by the parameter `n` of `openai_access.complete_chat()`
        """
        assert len(gpt_response[0]) == 1, "Accidentally sampling too many or too few responses from the LLM."
        for gpt_message in gpt_response[0]:
            gpt_message_str = gpt_message['content']
            pattern = r'\[RUN TACTIC\](.*?)\[END TACTIC\](?:.*?\[IMPORT\](.*?)\[END IMPORT\])?'
            tactics_with_imports = re.findall(pattern, gpt_message_str, re.DOTALL)
            # `tactics_with_imports` is the list of tuples almost meeting the docstring's need
            # We will only need to doctor the tactics a bit
            for tactic, necessary_import in tactics_with_imports:
                # Sometimes GPT thinks it's done and puts `end`
                # However, that would break our program, as our program furnishes an `end` automatically
                tactic = remove_end_line(tactic)
                gpt_tactics.append((tactic, necessary_import))
                avoid_steps += "[STEP]" + tactic + "\n"
                avoid_steps += "[ERROR]This tactic has been suggested by others. You should come up with a novel tactic.[END ERROR]\n"
                messages[1]["content"] = goals + avoid_steps

        #if not hasattr(prompt_for_tactics, 'gpt_token_counter'):
        #    prompt_for_tactics.gpt_token_counter = 0
        # Moved to outside the function body...
        prompt_for_tactics.gpt_token_counter += gpt_response[1]['total_tokens']
        if prompt_for_tactics.gpt_token_counter >= GPTCircuitBreak.token_limit(model_name, budget):
            raise GPTCircuitBreak(
                f"LLM token count reached {prompt_for_tactics.gpt_token_counter}, "
                f"incurring a cost of {GPTCircuitBreak.calc_cost(model_name, prompt_for_tactics.gpt_token_counter)} cents. "
                "Terminating the program so that costs don't go out of hand."
            )

    return gpt_tactics

# ... so that this will be run even if prompt_for_tactics is never called. This matters because
# ao_star from aostar_algorithm.py may try to access this without calling prompt_for_tactics,
# e.g. when a completely solved search tree is loaded from a file
if not hasattr(prompt_for_tactics, 'gpt_token_counter'):
    prompt_for_tactics.gpt_token_counter = 0