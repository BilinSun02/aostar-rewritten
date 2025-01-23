from vllm import LLM, SamplingParams
from .rpc import RPCServer, RPCMessage, RPCMessageKind
from typing import Any

class DeepSeekProverRPCServer(RPCServer):
    def __init__(self) -> None:
        model_name = "../2dsmodel/DeepSeek-Prover-V1.5/deepseek-ai/DeepSeek-Prover-V1.5-RL" # !!TODO: move
        self.model = LLM(
            model = model_name,
            max_num_batched_tokens = 8192,
            seed = 1,
            trust_remote_code = True
        )
        self.sampling_params = SamplingParams(
            temperature = 1.0,
            max_tokens = 2048,
            top_p = 0.95,
            n = 1,
        )

    def process(self, s: str) -> str:
        model_outputs = self.model.generate(
            s,
            self.sampling_params,
            use_tqdm = True,
        )[0].outputs[0].text
        return model_outputs

if __name__ == "__main__":
    # This is NOT a unit test. This file is intended to be run
    # directly and serve as a server.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    DeepSeekProverRPCServer(host = args.host, port = args.port)