from typing import Optional
import fire
from llama_models.llama3.api.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama

def interactive_chat(
    ckpt_dir: str,
    temperature: float = 0.9,
    top_p: float = 0.6,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    """
    Interactive chat with the Llama3.2-3B model using the terminal.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    print("Welcome to the interactive chat! Type 'exit' to quit.\n")

    # Initial dialog list with a system message
    dialog = [
        RawMessage(role="system", content="You are a helpful assistant. Respond thoughtfully to user queries.")
    ]
    result = generator.chat_completion(
        dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Get and print the assistant's response
    assistant_message = result.generation
    print(f"Assistant: {assistant_message.content}\n")


    while True:
        # Get user input
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        user_input = [RawMessage(role="user", content=user_input, stop_reason=StopReason.end_of_turn)]

        # Generate response
        result = generator.chat_completion(
            user_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # Get and print the assistant's response
        assistant_message = result.generation
        print(f"Assistant: {assistant_message.content}\n")
        dialog.append(assistant_message)


def main():
    fire.Fire(interactive_chat)

if __name__ == "__main__":
    main()
