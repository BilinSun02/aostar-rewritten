from typing import List, Tuple
from abc import ABC, abstractmethod
#from gpt_access import GptAccess
from dataclasses import dataclass, field
import re

class CostCircuitBreak(Exception):
    pass

class LLMAccess(ABC):
    def __init__(self, 
        model_name: str,
        budget_in_cents: int = 100,
    ) -> None:
        self.model_name = model_name
        self.budget_in_cents = budget_in_cents
        self.cost_in_cents = 0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    @abstractmethod
    def complete(self,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        """
        Text prediction: given prompt, predict text following
        the prompt until the predicted end of the contents.
        Note that this is similar to `openai.Completion` and
        *not* openai.ChatCompletion.
        If a model does not support text prediction, it's on
        the write of the LLMAccess subclass to "emulate" it,
        e.g. by wrapping it in a "chat" completion session.
        """
        pass

# !!TODO: change references to singular "tactic" to "tactics"
@dataclass
class GPTPrompter:
    gpt_token_counter: float = field(default=0.0, init=False)
    gpt_cost_counter: float = field(default=0.0, init=False)
    n_tactics: int = 1
    think_aloud: bool = False
    include_input_format_prompt: bool = False
    model_name: str = "gpt-4o-2024-08-06"
    #model_name: str = "gpt-4o-mini"
    budget: int = 200 # In cents

    def prompt_for_tactics(
        self,
        goals: str,
        avoid_steps: str = "[AVOID STEPS]",
    ) -> List[Tuple[str, str]]:
        """
        `goals` should furnish the [GOALS] section
        `avoid_steps` should furnish the [AVOID STEPS] section
        Returns a list of (`tactic`, `imports`) pairs suggested by the LLM.
        Each `tactic` should compile when appended to the existing proof
        once we prepend the `imports` to the existing proof
        """
        if self.n_tactics != 1:
            raise NotImplementedError("It's not currently supported to prompt for more than one tactic at a time.") # TODO: implement this.
        openai_access = GptAccess(model_name=self.model_name)
        messages = copy.deepcopy(messages_skeleton)
        messages[0]["content"] = prompt_message_introduction
        if self.include_input_format_prompt:
            messages[0]["content"] += prompt_message_input_format
        if self.think_aloud:
            messages[0]["content"] += prompt_message_output_format_with_thoughts
        else:
            messages[0]["content"] += prompt_message_output_format_wo_thoughts
        messages[0]["content"] += prompt_message_token_limit
        messages[1]["content"] = goals + avoid_steps
        gpt_tactics: List[Tuple[str, str]] = [] # tactic-import pairs

        while len(gpt_tactics) < self.n_tactics:
            gpt_response = openai_access.complete_chat(messages, max_tokens=response_token_limit, n=1, temperature=0.2)

            """
            Memo: a typical GPT response looks like this:
            ([{'role': 'assistant', 'content': '...', 'finish_reason': 'stop'}, {'role': 'assistant', 'content': '...', 'finish_reason': 'stop'}], {'prompt_tokens': 1980, 'completion_tokens': 139, 'total_tokens': 2119, 'reason': 'stop'})
            where the length of the list (the first element of the outermost tuple) is specified by the parameter `n` of `openai_access.complete_chat()`
            """
            assert len(gpt_response[0]) == 1, "Accidentally sampling too many or too few responses from the LLM."
            for gpt_message in gpt_response[0]:
                gpt_message_str = gpt_message['content']
                if not self.think_aloud: # Add a dummy so the regex also works
                    gpt_message_str = "[THOUGHTS][END THOUGHTS]\n" + gpt_message_str
                pattern = r'\[THOUGHTS\](.*?)\[END THOUGHTS\](?:.*?)\[RUN TACTIC\](.*?)\[END TACTIC\](?:.*?\[IMPORT\](.*?)\[END IMPORT\])?'
                tactics_with_imports = re.findall(pattern, gpt_message_str, re.DOTALL)
                # `tactics_with_imports` is the list of tuples almost meeting the docstring's need
                # We will only need to doctor the tactics a bit
                for thoughts, tactic, imports in tactics_with_imports:
                    if self.think_aloud:
                        print(thoughts+'\n\n') # TODO: return this to the caller, not just print out
                    # Sometimes GPT thinks it's done and puts `end`
                    # However, that would break our program, as our program furnishes an `end` automatically
                    tactic = remove_end_line(tactic)
                    gpt_tactics.append((tactic, imports))
                    avoid_steps += "[STEP]" + tactic + "\n"
                    avoid_steps += "[ERROR]This tactic has been suggested by others. You should come up with a novel tactic.[END ERROR]\n"
                    messages[1]["content"] = goals + avoid_steps

        return gpt_tactics

    @property
    def token_and_cost_stats(self) -> str:
        return f"Total token count so far: {self.gpt_token_counter}; cost: ${self.gpt_cost_counter/100:.2f}"

if __name__ == "__main__":
    # Test driving code
    goals = "[GOALS]\n[GOAL] \n1 + x = 3\n[HYPOTHESES]\n[HYPOTHESIS] x = 2\n"
    avoid_steps = "[AVOID STEPS]\n"
    prompter = GPTPrompter(
        think_aloud = False,
        model_name = "gpt-4o-mini",
    )
    print(prompter.prompt_for_tactics(goals, avoid_steps))
    print(prompter.token_and_cost_stats)
