from typing import List
from gpt_access import GptAccess
import copy
import re

class GPTCircuitBreak(Exception):
    @staticmethod
    def cost_per_1M_prompt_tokens(model_name: str) -> int:
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
            case "gpt-4-0613":
                return 3000
            case "gpt-3.5-turbo-0125":
                return 50
            case _: # Default to the cost of the most expensive, gpt-4-32k
                return 6000

    @staticmethod
    def cost_per_1M_completion_tokens(model_name: str) -> int:
        """
        Unit: cents
        """
        # Data from https://openai.com/api/pricing/
        # Accurate as of Aug 10, 2024
        match model_name:
            case "gpt-4o-mini":
                return 60 # cents per million tokens
            case "gpt-4o":
                return 1500
            case "gpt-4o-2024-08-06":
                return 1000
            case "gpt-4":
                return 6000
            case "gpt-4-0613":
                return 6000
            case "gpt-3.5-turbo-0125":
                return 150
            case _: # Default to the cost of the most expensive, gpt-4-32k
                return 12000

    @staticmethod
    def calc_cost(model_name: str, n_tokens: int) -> int:
        rate = GPTCircuitBreak.cost_per_1M_tokens(model_name)
        return int(n_tokens * rate / 1000000)

prompt_message_introduction = \
"""
You are a proficient formal theorem-proving agent in Lean 3. You can predict the next proof step given the current proof state.

"""

prompt_message_input_format = \
"""
The proof state is described in the following format:
1. Goals are in a [GOALS] section. Each goal in this [GOALS] section is started by [GOAL]. Under each [GOAL], the goal is described as a serialized version of the proof state as shown while running lean command.
2. Each goal comes with a [HYPOTHESES] section which consists of zero or more [HYPOTHESIS] lines. Two optional sections [DEFINITIONS] and [THEOREMS] include possibly relevant definitions and theorems.
3. A [STEPS] section shows proof [STEP]s used so far.
4. An optional [AVOID STEPS] section collects proof steps that you should avoid. This section has zero or more independent [STEP] lines. Each step optionally comes with an [ERROR] message.

"""

prompt_message_output_format_wo_thoughts = \
"""
Your response should consist of one proof step attempt. Syntactically, it consists of [RUN TACTIC] followed by one proof step that you believe will help advance the current proof state, then followed by [END TACTIC]. For example, "[RUN TACTIC]induction c,[END TACTIC]".
You may plan ahead for multiple tactics in [THOUGHTS], but do NOT aim to produce a full proof in [RUN TACTIC].Include only one Lean tactic. For instance, "[RUN TACTIC]rw [h],[END TACTIC]" is acceptable, but "[RUN TACTIC]rw [h], simp,[END TACTIC]" is strictly prohibited.
If you are very certain the goal cannot be proven (e.g. "1 % 2 = 0") without an equally wrong hypothesis that might have allowed you to use "exfalso", then you may use "sorry".
You cannot assume any library not imported in the piece given to you. You may optionally include one [IMPORT] statement, e.g. "[IMPORT]import tactic.linarith[END IMPORT]", after [END TACTIC].

"""

prompt_message_output_format_with_thoughts = \
"""
Your response should consist of one proof step attempt. Start with a section begun with [THOUGHTS] and ending with [END THOUGHTS], in which you rephrase the goal in natural language, reflect over why each failed attempt in [AVOID STEPS] failed, and informally discuss how you would correctly approach the goal, leading up to a concrete tactic. This section will not be read by others and can be as concise as you yourself can understand. Then write up a section begun with [RUN TACTIC] and ending with [END TACTIC], in which you provide one tactic to advance the current proof state. For example, "[THOUGHTS]The goal states that the sum of 1 to $n$ equals $\\frac{n(n+1)}{2}$. No failed attempts yet. No axiom apparently applicable. Will try induction.[END THOUGHTS]\n[RUN TACTIC]induction n,[END TACTIC]".
You may plan ahead for multiple tactics in [THOUGHTS], but do NOT aim to produce a full proof in [RUN TACTIC].Include only one Lean tactic. For instance, "[RUN TACTIC]rw [h],[END TACTIC]" is acceptable, but "[RUN TACTIC]rw [h], simp,[END TACTIC]" is strictly prohibited.
If you are very certain the goal cannot be proven (e.g. "1 % 2 = 0") without an equally wrong hypothesis that might have allowed you to use "exfalso", then you may use "sorry".
You cannot assume any library not imported in the piece given to you. You may optionally include one [IMPORT] statement, e.g. "[IMPORT]import tactic.linarith[END IMPORT]", after [END TACTIC].
"""

# TODO: the prompt still needs improvement.
# 1. The example seems to mislead GPT to use induction for a problem as simple as a+b=b+a. Either add more explanation or simply remove the example.
# 2. GPT fails to be very concise, and still produces full English sentences. See what we can do.
# 3. When think_aloud is enabled, GPT tends to produce too many steps at one time which can easily fail to compile. Even if I include "The step may have slightly more than one tactic, but not more than three; there's no need to solve the problem in one step.", it sometimes produces a lot of tactics in one step. Hence I have to tell it to only have one tactic per step for now.
# 4. This hardcodes "sorry" to mean "the goal was abandoned". Un-hardcode this in the future if we need to use "sorry" in the future.

response_token_limit = 4096 # Maximum GPT-4o supports. Interestingly GPT-4o-mini supports more.

prompt_message_token_limit = f"Make sure your response do not exceed {response_token_limit=}."

messages_skeleton = [
    {
        "role": "system",
        "content": ""
    },
    {
        'role': 'user',
        'content': ""
    },
]

def remove_end_line(string:str) -> str:
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
    n_tactics: int = 1,
    think_aloud: bool = False,
    include_input_format_prompt: bool = False,
    model_name: str = "gpt-4o",
    budget: int = 200, # In cents
) -> List[str]:
    """
    `goals` should furnish the [GOALS] section
    `avoid_steps` should furnish the [AVOID STEPS] section
    Returns a list of (`tactic`, `necessary_import`) pairs suggested by the LLM.
    Each `tactic` should compile when appended to the existing proof
    once we prepend the `necessary_import` to the existing proof
    """
    if n_tactics != 1:
        raise NotImplementedError("It's not currently supported to prompt for more than one tactic at a time.") # TODO: implement this.
    #openai_access = GptAccess(model_name="gpt-3.5-turbo")
    openai_access = GptAccess(model_name=model_name)
    messages = copy.deepcopy(messages_skeleton)
    messages[0]["content"] = prompt_message_introduction
    if include_input_format_prompt:
        messages[0]["content"] += prompt_message_input_format
    if think_aloud:
        messages[0]["content"] += prompt_message_output_format_with_thoughts
    else:
        messages[0]["content"] += prompt_message_output_format_wo_thoughts
    messages[0]["content"] += prompt_message_token_limit
    messages[1]["content"] = goals + avoid_steps
    gpt_tactics = []

    while len(gpt_tactics) < n_tactics:
        gpt_response = openai_access.complete_chat(messages, max_tokens=response_token_limit, n=1, temperature=0.2)

        """
        Memo: a typical GPT response looks like this:
        ([{'role': 'assistant', 'content': '...', 'finish_reason': 'stop'}, {'role': 'assistant', 'content': '...', 'finish_reason': 'stop'}], {'prompt_tokens': 1980, 'completion_tokens': 139, 'total_tokens': 2119, 'reason': 'stop'})
        where the length of the list (the first element of the outermost tuple) is specified by the parameter `n` of `openai_access.complete_chat()`
        """
        assert len(gpt_response[0]) == 1, "Accidentally sampling too many or too few responses from the LLM."
        for gpt_message in gpt_response[0]:
            gpt_message_str = gpt_message['content']
            if not think_aloud: # Add a dummy so the regex also works
                gpt_message_str = "[THOUGHTS][END THOUGHTS]\n" + gpt_message_str
            pattern = r'\[THOUGHTS\](.*?)\[END THOUGHTS\](?:.*?)\[RUN TACTIC\](.*?)\[END TACTIC\](?:.*?\[IMPORT\](.*?)\[END IMPORT\])?'
            tactics_with_imports = re.findall(pattern, gpt_message_str, re.DOTALL)
            # `tactics_with_imports` is the list of tuples almost meeting the docstring's need
            # We will only need to doctor the tactics a bit
            for thoughts, tactic, necessary_import in tactics_with_imports:
                if think_aloud:
                    print(thoughts+'\n\n') # TODO: return this to the caller, not just print out
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
        prompt_for_tactics.gpt_cost_counter += (gpt_response[1]['prompt_tokens']/1000000) * GPTCircuitBreak.cost_per_1M_prompt_tokens(model_name)
        prompt_for_tactics.gpt_cost_counter += (gpt_response[1]['completion_tokens']/1000000) * GPTCircuitBreak.cost_per_1M_completion_tokens(model_name)
        if prompt_for_tactics.gpt_cost_counter >= budget:
            raise GPTCircuitBreak(
                f"LLM token count reached {prompt_for_tactics.gpt_token_counter}, "
                f"incurring a cost of {prompt_for_tactics.gpt_cost_counter} cents. "
                "Terminating the program so that costs don't go out of hand."
            )

    return gpt_tactics

# ... so that this will be run even if prompt_for_tactics is never called. This matters because
# ao_star from aostar_algorithm.py may try to access this without calling prompt_for_tactics,
# e.g. when a completely solved search tree is loaded from a file
if not hasattr(prompt_for_tactics, 'gpt_token_counter'):
    prompt_for_tactics.gpt_token_counter = 0
if not hasattr(prompt_for_tactics, 'gpt_cost_counter'):
    prompt_for_tactics.gpt_cost_counter = 0