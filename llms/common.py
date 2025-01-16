from abc import ABC, abstractmethod

class CostCircuitBreak(Exception):
    pass

class LLMAccess(ABC):
    incurs_cost: bool

    def __init__(self, 
        model_name: str,
        budget_in_cents: int = 100,
    ) -> None:
        assert self.incurs_cost is not None, "LLMAccess subclasses must specify whether they incur cost"
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

    @property
    def cost_stats(self) -> str:
        if self.incurs_cost:
            return f"{self.usage['prompt_tokens']} prompt tokens and " +\
                   f"{self.usage['completion_tokens']} completion tokens " +\
                   f"incurred ${self.cost_in_cents/100:.2f} dollars in cost."
        else:
            return "No cost incurred."
