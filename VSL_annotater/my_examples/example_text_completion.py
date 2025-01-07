# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Optional

import fire

from llama_models.llama3.reference_impl.generation import Llama
from termcolor import cprint


def run_main(
    ckpt_dir: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: int = 1024,
    model_parallel_size: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    prompts = [
        """\
Dumbledore: I should have known that you would be here...Professor McGonagall.
The cat meows, sniffs out and the camera pans back to a wall. The cats shadow is seen progressing into a human. 
There are footsteps and MINERVA MCGONAGALL is revealed.
McGonagall: Good evening, Professor Dumbledore. Are the rumours true, Albus?
Dumbledore: I'm afraid so, Professor. The good, and the bad.
McGonagall: And the boy?
Dumbledore: Hagrid is bringing him.
McGonagall: Do you think it wise to trust Hagrid with something as important as this?""",
    ]
    for prompt in prompts:
        result = generator.text_completion(
            prompt,
            temperature=0.9,
            top_p=0.9,
            max_gen_len=max_gen_len,
            logprobs=False,
        )

        cprint(f"{prompt}", end="")
        cprint(f"{result.generation}", color="yellow")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
