from typing import Optional
import fire
from llama_models.llama3.api.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama


def interactive_chat(
    ckpt_dir: str,
    temperature: float = 0.9,
    top_p: float = 0.6,
    max_seq_len: int = 8192,
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

    # Access the tokenizer from the generator
    tokenizer = generator.tokenizer

    print("Welcome to the interactive chat! Type 'exit' to quit.\n")

    # Initial dialog list with a system message
    dialog = [
        RawMessage(role="system", content="You are a helpful assistant. Respond thoughtfully to user queries.")
    ]

    while True:
        # Get user input
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # # Calculate and print token length of user input
        # user_token_count = len(tokenizer.encode(user_input))
        

        user_input_raw = RawMessage(role="user", content=user_input, stop_reason=StopReason.end_of_turn)
        # Calculate and print token length of user input
        user_token_count = len(tokenizer.encode(s=user_input, bos=True, eos=True))
        print(f"[INFO] User input token length: {user_token_count}")
        print("Length os user input: ", len(user_input))
        print(user_input)
        # Append user message to dialog
        dialog.append(user_input_raw)

        # Generate response
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # Get and print the assistant's response
        assistant_message = result.generation
        assistant_token_count = len(tokenizer.encode(assistant_message.content, bos=True, eos=True))
        print(f"[INFO] Assistant response token length: {assistant_token_count}")

        print(f"Assistant: {assistant_message.content}\n")

        # Append the assistant message to the dialog
        dialog.append(assistant_message)


def main():
    fire.Fire(interactive_chat)


if __name__ == "__main__":
    main()
