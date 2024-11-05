#!/usr/bin/env python3
# Uses deprecated OpenAI <1.0.0 APIs
# Adapted from the copra codebase

import sys, os
root_dir = f"{__file__.split('gpt_access')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import openai
import typing
from .common import LLMAccess, CostCircuitBreak
import copy

# Data from https://openai.com/api/pricing/
# and from https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-two
# Accurate as of Nov 4, 2024
gpt_model_info ={
    "gpt-3.5-turbo-0125": {
        "cents_per_1M_prompt_tokens": 50,
        "cents_per_1M_completion_tokens": 150,
        "token_limit_per_min": 200_000, 
        "request_limit_per_min" : 3_500, 
        "max_token_per_prompt" : int(3.75*2**10) # less than 4k because additional tokens are added at times
    },
    "gpt-4": {
        "cents_per_1M_prompt_tokens": 3000,
        "cents_per_1M_completion_tokens": 6000,
        "token_limit_per_min": 10_000,
        "request_limit_per_min": 500,
        "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
    },
    "gpt-4-turbo": {
        "cents_per_1M_prompt_tokens": 1000,
        "cents_per_1M_completion_tokens": 3000,
        "token_limit_per_min": 30_000,
        "request_limit_per_min": 500,
        "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
    },
    "gpt-4o": {
        "cents_per_1M_prompt_tokens": 250,
        "cents_per_1M_completion_tokens": 1000,
        "token_limit_per_min": 30_000,
        "request_limit_per_min": 500,
        "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
    },
    "gpt-4o-mini": {
        "cents_per_1M_prompt_tokens": 15,
        "cents_per_1M_completion_tokens": 60,
        "token_limit_per_min": 200_000,
        "request_limit_per_min": 500,
        "max_token_per_prompt": int(1.2*10**5) # less than 128k because additional tokens are added at times
    },
}

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

class GptAccess(LLMAccess):
    def __init__(self, 
        model_name: str,
        budget_in_cents: int = 100,
        secret_filepath: str = ".secrets/openai_key.json"
    ) -> None:
        super().__init__(model_name=model_name, budget_in_cents=budget_in_cents)
        assert secret_filepath.endswith(".json"), "Secret filepath must be a .json file"
        assert os.path.exists(secret_filepath), "Secret filepath does not exist"
        self.secret_filepath = secret_filepath
        self._load_secret()

    def complete_prompt(self, 
        prompt: str, 
        n: int = 1, 
        max_tokens: int = 5, 
        temperature: float = 0.25, 
        top_p: float = 1.0, 
        frequency_penalty: float = 0.0, 
        presence_penalty: float = 0.0, 
        stop: list = ["\n"],
        logprobs: int = 0) -> typing.List[typing.Tuple[str, float]]:
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            logprobs=logprobs
            # best_of=n
        )
        usage = response.usage
        self.usage["prompt_tokens"] += usage.prompt_tokens
        self.usage["completion_tokens"] += usage.completion_tokens
        self.usage["total_tokens"] += usage.total_tokens        
        self.cost_in_cents += int(usage.prompt_tokens * gpt_model_info[self.model_name]["cents_per_1M_prompt_tokens"] / 1000000)
        self.cost_in_cents += int(usage.completion_tokens * gpt_model_info[self.model_name]["cents_per_1M_completion_tokens"] / 1000000)

        resp = [(obj.text, sum(obj.logprobs.token_logprobs)) for obj in response.choices]
        resp.sort(key=lambda x: x[1], reverse=True)
        return resp

    def complete_chat(self,
            messages: typing.List[str],
            n: int = 1,
            max_tokens: int = 25,
            # temperature: float = 0.25, # TODO: verify again if a high temperature is indeed needed
            temperature: float = 0.25,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            stop: list = ["\n"]) -> typing.Tuple[list, dict]:
        self.check_budget()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            # messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            # stop=stop,
            n=n
        )
        usage = response.usage
        self.usage["prompt_tokens"] += usage.prompt_tokens
        self.usage["completion_tokens"] += usage.completion_tokens
        self.usage["total_tokens"] += usage.total_tokens
        self.cost_in_cents += int(usage.prompt_tokens * gpt_model_info[self.model_name]["cents_per_1M_prompt_tokens"] / 1000000)
        self.cost_in_cents += int(usage.completion_tokens * gpt_model_info[self.model_name]["cents_per_1M_completion_tokens"] / 1000000)
        # The actual cost is slightly higher:
        # The name of the user etc. also count towards the tokens.
        # See here for more details: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        return_responses = [{"role": choice.message.role, "content": choice.message.content} for choice in response.choices]
        for i in range(len(return_responses) - 1):
            return_responses[i]["finish_reason"] = "stop"
        if len(response.choices) > 0:
            return_responses[-1]["finish_reason"] = response.choices[-1].finish_reason
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "reason": response.choices[-1].finish_reason if len(response.choices) > 0 else "stop"
        }
        return return_responses, usage_dict
    
    def check_budget(self):
        if self.cost_in_cents > self.budget_in_cents:
            raise CostCircuitBreak(
                f"LLM token count reached {self.usage['total_tokens']}, "
                f"incurring a cost of {self.cost_in_cents} cents. "
                "Terminating the program so that costs don't go out of hand."
            )

    def _load_secret(self) -> None:
        with open(self.secret_filepath, "r") as f:
            secret = json.load(f)
            # openai.organization = secret["organization"]
            openai.api_key = secret["api_key"]
        pass

if __name__ == "__main__":
    os.chdir(root_dir)
    openai_access = GptAccess(model_name="gpt-3.5-turbo-0125")
    # openai_access = GptAccess(model_name="gpt-4")
    # openai_access = GptAccess(model_name="davinci")
    # print(openai_access.get_models())
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
        {
            "role": "user",
            "content": "Our idea seems to be scooped, don't know how to change direction now."
        }
    ]
    print("printing complete chat:")
    print(openai_access.complete_chat(messages, max_tokens=15, n=2, temperature=0.8))
    pass
