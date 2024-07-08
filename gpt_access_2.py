#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('gpt_access')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import json
import openai
import typing
import tiktoken
import logging

class GptAccess(object):
    gpt_model_info ={
        "gpt-3.5-turbo": {
            "token_limit_per_min": 45000, 
            "request_limit_per_min" : 3400, 
            "max_token_per_prompt" : int(3.75*2**10) # less than 4k because additional tokens are added at times
        },
        "gpt-4": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
        },
        "gpt-4-0314": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
        },
        "gpt-4-0613": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
        },
        "gpt-4-1106-preview": {
            "token_limit_per_min": 150000,
            "request_limit_per_min": 20,
            "max_token_per_prompt": int(1.2*10**5) # less than 128k because additional tokens are added at times
        },
        'codellama/CodeLlama-7b-Instruct-hf': {
            "token_limit_per_min": 10**6,
            "request_limit_per_min": 10**6,
            "max_token_per_prompt": int(13.75*2**10) # less than 16k because additional tokens are added at times
        },
        'EleutherAI/llemma_7b': {
            "token_limit_per_min": 10**6,
            "request_limit_per_min": 10**6,
            "max_token_per_prompt": int(13.75*2**10) # less than 16k because additional tokens are added at times
        },
        'morph-labs/morph-prover-v0-7b': {
            "token_limit_per_min": 10**6,
            "request_limit_per_min": 10**6,
            "max_token_per_prompt": int(13.75*2**10) # less than 16k because additional tokens are added at times            
        }
    }
    def __init__(self, 
        secret_filepath: str = ".secrets/openai_key.json",
        model_name: typing.Optional[str] = None) -> None:
        assert secret_filepath.endswith(".json"), "Secret filepath must be a .json file"
        assert os.path.exists(secret_filepath), "Secret filepath does not exist"
        self.secret_filepath = secret_filepath
        self._load_secret()
        self.models_supported = openai.Model.list().data
        self.models_supported_name = [model.id for model in self.models_supported]
        if model_name is not None:
            assert model_name in self.models_supported_name, f"Model name {model_name} not supported"
            self.model_name = model_name
        self.is_open_ai_model = True
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        pass

    def get_models(self) -> list:
        return self.models_supported
    
    def complete_prompt(self, 
        prompt: str, 
        model: typing.Optional[str] = None,
        n: int = 1, 
        max_tokens: int = 5, 
        temperature: float = 0.25, 
        top_p: float = 1.0, 
        frequency_penalty: float = 0.0, 
        presence_penalty: float = 0.0, 
        stop: list = ["\n"],
        logprobs: int = 0) -> typing.List[typing.Tuple[str, float]]:
        model = self.model_name if model is None else model
        response = openai.Completion.create(
            model=model,
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
        if self.is_open_ai_model:
            usagae = response.usage
            self.usage["prompt_tokens"] += usagae.prompt_tokens
            self.usage["completion_tokens"] += usagae.completion_tokens
            self.usage["total_tokens"] += usagae.total_tokens        
        resp = [(obj.text, sum(obj.logprobs.token_logprobs)) for obj in response.choices]
        resp.sort(key=lambda x: x[1], reverse=True)
        return resp

    def complete_chat(self,
            messages: typing.List[str],
            model: typing.Optional[str] = None,
            n: int = 1,
            max_tokens: int = 25,
            # temperature: float = 0.25, #TODO: verify again if a high temperature is indeed needed
            temperature: float = 0.25,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            stop: list = ["\n"]) -> typing.Tuple[list, dict]:
        model = self.model_name if model is None else model
        if self.is_open_ai_model:
            # TODO: only uses the first and the last message for now, ignoring all the example messages as they do not accurately generate multiple tactics in one go
            messages_temp = [messages[0], messages[-1]]
            # logger = logging.getLogger("__main__")
            

            # logger.info(f"using temperature: {temperature}")
            # logger.info(f"using frequency_penalty: {frequency_penalty}")
            # logger.info(f"using presence_penalty: {presence_penalty}")
            # logger.info(f"using top-p: {top_p}")

            # print("using temporary messages for multiple responses: ", messages_temp)
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages_temp,
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
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                # top_p=top_p,
                stop=stop,
                n=n
            )
            usage = response.usage
            self.usage["prompt_tokens"] += usage.prompt_tokens
            self.usage["completion_tokens"] += usage.completion_tokens
            self.usage["total_tokens"] += usage.total_tokens
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
    
    def num_tokens_from_messages(self, messages, model=None):
        # Model name is like "gpt-3.5-turbo-0613"
        model = model if model is not None else self.model_name
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _load_secret(self) -> None:
        with open(self.secret_filepath, "r") as f:
            secret = json.load(f)
            # openai.organization = secret["organization"]
            openai.api_key = secret["api_key"]
        pass

    def get_usage(self) -> dict:
        return self.usage

if __name__ == "__main__":
    os.chdir(root_dir)
    openai_access = GptAccess(model_name="gpt-3.5-turbo")
    # openai_access = GptAccess(model_name="gpt-4")
    # openai_access = GptAccess(model_name="davinci")
    # print(openai_access.get_models())
    messages = [
        {
            "role": "system",
            "content": \
"""
You are a proficient formal theorem-proving agent in Lean 3. You can predict the next proof step given the current proof state. The proof state is described in the following format:
1. All the goals are described under [GOALS] keyword. Each goal within the [GOALS] is described under the keyword [GOAL] i, where i is a positive integer. For example, [GOAL] 1, [GOAL] 2, etc.
2. Within each [GOAL] i keyword, the goal is described as a human-readable serialized version of the proof state as shown while running lean command. Each goal, might also accompany some hypotheses, which are described under the keyword [HYPOTHESES] i. Each hypothesis within [HYPOTHESES], starts with the prefix [HYPOTHESIS]. Apart from the goal and hypothesis, some OPTIONAL keywords like [DEFINITIONS] i and [THEOREMS] i are also present which describe the relevant definitions of symbols used in that goal, and some possible useful theorems or lemmas which might help in simplifying the goal. Each definition within [DEFINITIONS] starts with the prefix [DEFINITION], similarly, each theorem/lemma within [THEOREMS] starts with the prefix [THEOREM].
3. Optional keywords like [INFORMAL-THEOREM] and [INFORMAL-PROOFS] which will describe the proofs and theorems in natural language. The proof is described in for the whole theorem along with the theorem statement rather than just the proof state. For example, [INFORMAL-THEOREM]\nThe sum of two even numbers is even.\n[INFORMAL-PROOFS]\nSuppose a and b are even numbers. Then there exist integers m and n such that a = 2 * m and b = 2 * n. Then a + b = 2 * m + 2 * n = 2 * (m + n). Since m + n is an integer, a + b is even.. The whole proof is described in natural language, and the proof state is not described. The proof state is described in the [GOALS] keyword which should be used to generate the next proof step. Please do not take the informal proof too rigorously and only use it as a reference. It may simply be wrong.
4. Finally, [STEPS] keyword is used to describe proof-steps used so far. Each proof step starts with the prefix [STEP], and is a valid Lean tactic. For example, [STEPS][STEP]rw h₁ at h₂,[STEP]{linarith},. Do not generate [STEP] in your response as it is only used to tell you how far the proofs have progressed.
5. Sometimes, [INCORRECT STEPS] keyword optionally used to describe proof-steps which should NOT be generated. Use this as a hint for not generating these proof-steps again as they failed previously. For example, [INCORRECT STEPS][STEP]apply h₁,[STEP]rw ←h₁. **DO NOT** generate these [INCORRECT STEPS] under the same states. If the states have changed, you may regenerate some incorrect steps.
6. There is also an optional [LAST STEP] keyword which describes the proof-step generated last time. If the proof-step was incorrect, then it is also followed by error message from Lean environment. For example, [LAST STEP]linarith,\n[ERROR MESSAGE]linarith failed to find a contradiction\nstate:\nx y : ℝ,\nh₁ : x = 3 - 2 * y,\nh₂ : 2 * x - y = 1\n⊢ false. If the proof-step was correct then it is followed by the keyword [SUCCESS]. For example, [LAST STEP]linarith,[SUCCESS]. Don't generate the last proof-step again if it was NOT successful.
7. Sometimes there can be errors in the format of the generated response. This is reported using the keyword [ERROR] followed by the error message. For example, [ERROR]\nInvalid response:\n'Great! The proof is complete.', \nStopping Reason: 'stop'.\n Please respond only in the format specified.[END]. This means that the response generated by you was not in the specified format. Please follow the specified format strictly.
8. Whenever you are asked to prove one theorem, keep in mind that you are not asked to prove one theorem in one shot. Specifically, when [EXPAND NUM] is ued to tell you, in parallel, how many tactics you should try in a breadth first search manner. Do not make it sequential where the later tactics depend on the first few tactics.
9. Completely focus your self on proving the [FOCUSED GOAL]. In other words, try not to generate anything not related to the current [FOCUSED GOAL]

If you think you know the next proof step, then start your response with [RUN TACTIC] followed by the next proof-step which will help in simplifying the current proof state. For example, [RUN TACTIC]induction c,[END]. Generate 5 parallel steps at a time, meaning that these steps are all possible tactics that can transform the current state into an easier state. Do not generate multiple unparallel steps at a time because you cannot see the intermediate results. Make sure that the proof step is valid and compiles correctly in Lean 3.

You can refer to the example conversation to understand the response format better. It might also contain some similar proof states and their corresponding proof-steps.

 Please take a note of the following: 
 1. Make sure to end all your responses with the keyword [END]. Follow the specified format strictly. Your only response will be to generate [RUN TACTIC] followed by the proof tactics that you have in mind and then followed by the keyword [END]. Make sure you are not generating any other keywords in between [RUN TACTIC] and [END] such as [STEP].
 2. While generating [RUN TACTIC] keyword, do NOT generate the tactics mentioned under [INCORRECT STEPS] in the proof state description because they are failed tactics which have been tried earlier. Similary do NOT generate the last tactic if it was NOT successful. Re-generating proof-steps which mentioned in [INCORRECT STEPS] or failed [LAST STEPS] will lead to backtracking and early termination of proof search. 
 3. Do NOT finish the proof in one shot ending with end. Always go step by step. Ideally individual tactics are NOT long, ~~so~~ don't generate too many tokens, unless necessary. Generating single step in parallel allows the user to give more proof state after each step, which will help you in writing correct proof-steps.
 4. [EXPAND NUM]  indicates the number of tactics or steps that you are asked to generate for the next layer.
 5. When asked to generate multiple tactics (say 5), generate exactly 5 tactics, each of which starts with [RUN TACTIC] and ends with [END]. Make sure there are exactly 5 of them. Again, make sure you are generating 5 tactics in parallel (top [EXPAND NUM] tactics that can be used to translate the current focused goal). 
 6. [FOCUSED GOAL] indicates the goal that you must focus to prove.
 7. Make sure to always capitalize the keywords, i.e., a [RUN TACTIC] is right but [Run Tactic] is strictly wrong. Similarly, [END] is right but [End] is definitely wrong. Try to be sensitive with this grammar even there are multiple tactics that you are proposing
"""
        },
        {
            'role': 'user',
            'content': \
"""
Goals to prove:\n[GOALS]\n[GOAL] 1\nx % 2 = 0 → x * x % 2 = 0\n[HYPOTHESES] 1\n[HYPOTHESIS] x : ℕ\n\n[INFORMAL-THEOREM]\ntheorem mod_arith_1\n(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=\n\n[INFORMAL-PROOF]\nTo prove this theorem, we will use the property that if x is even, then x * x is also even. An even number is defined as a number that is divisible by 2, which means it can be expressed as 2 * k for some integer k. The statement x % 2 = 0 asserts that x is even.\n\nLet's proceed with the proof in Lean:\n\n
        lean\ntheorem mod_arith_1 (x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=\nbegin\n  -- Assume x is even, i.e., x % 2 = 0\n  intro h,\n  -- Since x is even, there exists some k such that x = 2 * k\n  have k_def : ∃ k, x = 2 * k := exists_eq_mul_right_of_dvd (nat.dvd_of_mod_eq_zero h),\n  -- Let's use this k to express x\n  cases k_def with k hk,\n  -- Now we rewrite x as 2 * k and expand (2 * k) * (2 * k)\n  rw hk,\n  -- After expansion, we get 4 * k * k\n  calc (2 * k) * (2 * k) = 4 * (k * k) : by ring\n  ... = 2 * (2 * (k * k)) : by rw ←mul_assoc\n  -- The result is clearly a multiple of 2, hence it is even\n  ... % 2 = 0 : by rw nat.mul_mod_right\nend\n
        \n\nIn this proof, we first introduce our assumption that x is even. Then we express x as 2 * k for some k using the fact that x is divisible by 2. We then rewrite x in terms of k and expand the expression (2 * k) * (2 * k) to 4 * k * k, which is clearly a multiple of 2. Finally, we conclude that (x * x) % 2 = 0, which completes the proof.\n[THEOREMS] 1\n[THEOREM] complex.sin_two_pi :  sin (2 * π) = 0\n[THEOREM] nat.digits_aux_zero : (b : ℕ) (h : 2 ≤ b) : digits_aux b h 0 = []\n[THEOREM] int.mod_two_ne_one :  ¬ n % 2 = 1 ↔ n % 2 = 0\n[THEOREM] int.mod_two_ne_zero :  ¬ n % 2 = 0 ↔ n % 2 = 1\n[THEOREM] nat.mod_two_ne_one :  ¬ n % 2 = 1 ↔ n % 2 = 0\n[THEOREM] nat.mod_two_ne_zero :  ¬ n % 2 = 0 ↔ n % 2 = 1\n[THEOREM] nat.eq_zero_of_mul_eq_zero :  ∀ {n m : ℕ}, n * m = 0 → n = 0 ∨ m = 0 | 0        m\n[END]\n[FOCUSED GOAL]: hypotheses: x : ℕgoal: x % 2 = 0 → x * x % 2 = 0\n[EXPAND NUM]: 5
"""
        },
    ]
    print("printing complete chat:")
    print(openai_access.complete_chat(messages, max_tokens=15, n=2, temperature=0.8))
    pass
