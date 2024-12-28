from typing import Optional
import fire
from llama_models.llama3.api.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama
import json
import re
def clean_narrative(narrative_json):
    for video_id in list(narrative_json.keys()): # 38737402-19bd-4689-9e74-3af391b15feb
        narrations_passes = narrative_json[video_id]
        keys = list(narrations_passes.keys())
        keys.remove('status')
        # print(keys)
        chunk_list = narrations_passes[keys[0]]
        for chunk in chunk_list['narrations']:
            del chunk["timestamp_frame"]
            del chunk["_unmapped_timestamp_sec"]
            del chunk['annotation_uid']
    return narrative_json

def llama_converter(
    ckpt_dir: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_seq_len: int = 5100,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 4096,
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
    tokenizer = generator.tokenizer

    # Read ex_narrative json
    # Open and read the JSON file
    with open('/home/ldapusers/gogebakan/AML/llama-models/llama_models/scripts/ex_nar.json', 'r') as file:
        ex_nar = json.load(file)
        ex_nar = clean_narrative(ex_nar)
        ex_nar = json.dumps(ex_nar, indent=None).strip()


    # Read ex_nlw.json
    with open('/home/ldapusers/gogebakan/AML/llama-models/llama_models/scripts/ex_nlq.json', 'r') as file:
        ex_nlq = json.load(file)
        ex_nlq = json.dumps(ex_nlq, indent=None).strip()


    # Read task_narration
    with open('/home/ldapusers/gogebakan/AML/llama-models/llama_models/scripts/task_narration.json', 'r') as file:
        task_narration = json.load(file)
        task_narration = clean_narrative(task_narration)
        task_narration = json.dumps(task_narration, indent=None).strip()


    json_template = """{
    
    "language_queries": [
        {
            "video_start_sec": <query_start_time_in_seconds>,
            "video_end_sec": <query_end_time_in_seconds>,
            "query": "<query-question>"
        }
    ]}
    """

    chunk_template = """{
        "video_start_sec": <query_start_time_in_seconds>,
        "video_end_sec": <query_end_time_in_seconds>,
        "query": "<query-question>"
    }"""

    system_prompt = f"""
    You are an advanced assistant helping a Machine Learning engineer to preprocess data. Your task is to:
    1. Understand the structure of input narrative JSON files, which contain time-stamped narrative data describing events in a video.
    2. Generate a JSON output strictly following this structure: {json_template}.
    3. Populate the "language_queries" field with 10 unique questions in the format of {chunk_template}.
    4. Ensure questions are contextually relevant, temporally aligned with the input, and unique in content.
    5. DO NOT add any additional text when generating JSON output—respond only with the structured JSON.

    Each "query" in "language_queries" should:
    - Reflect an understanding of the event described in the narrative.
    - Use timestamps from the input file to maintain temporal relevance.
    """

    first_prompt = f"""
    I need to extract meaningful and unique template questions from a narrative file describing events in a video. Here's the context:
    1. The video is shot from a first-person perspective and captures daily life moments.
    2. You will generate questions based on narratives provided in JSON format.
    3. The narrative input and expected output formats are as follows:
        - `narration_1_1.json`: Example input narrative file with time-stamped descriptions.
        - `nlq_val_1_1.json`: Example output JSON file with questions and timestamps derived from the input.
        - `narration_1_2.json`: New input file for which you need to generate 10 unique questions.
    4. Your task:
        - Understand the context and temporal alignment in the input.
        - Generate 10 diverse, unique questions that match the input narrative's context and time stamps.
        - Output the JSON in the format of `nlq_val_1_1.json` without any extra text.
        - Generated question queries shouldnt contain any information about time stamps
        - Avoid questions whose answers are "yes" or "no"
    """


    user_token_count = len(tokenizer.encode(s=first_prompt, bos=True, eos=True))
    print(f"First prompt token number: {user_token_count}")

    # Initial dialog list with a system message
    dialog = [RawMessage(role="system",content=system_prompt
        ),
        RawMessage(
            role="user",
            content=first_prompt,
            stop_reason=StopReason.end_of_message
        ),
        RawMessage(
            role="assistant",
            content="Sure, can you provide me narration_1_1.json?",
            stop_reason=StopReason.end_of_turn
        ),
        RawMessage(
            role="user",
            content=ex_nar,
            stop_reason=StopReason.end_of_message
        ),
        RawMessage(
            role="assistant",
            content="Thanks for the narration_1_1.json. Can you provide me nlq_val_1_1.json so that I can understand the relationship between input and the desired output?",
            stop_reason=StopReason.end_of_turn
        ),
        RawMessage(
            role="user",
            content=ex_nlq,
            stop_reason=StopReason.end_of_message
        ),
        RawMessage(
            role="assistant",
            content="Thanks, I get that. Now, please give me the example narration JSON (narration_1_2.json). I will generate a JSON script that includes 10 unique questions with corresponding timestamps in the same format as nlq_val_1_1.json. I will provide you JSON without any additional messages.",
            stop_reason=StopReason.end_of_turn
        ),
        RawMessage(
            role="user",
            content=task_narration,
            stop_reason=StopReason.end_of_message)
            ]
    print("Token number of messages")
    for msg in dialog:
        token_count = len(tokenizer.encode(s=msg.content, bos=True, eos=True))
        print(f"{msg.role}, {token_count}")


    result = generator.chat_completion(
        dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Get and print the assistant's response
    assistant_message = result.generation
    print(f"Assistant: {assistant_message.content}\n")
    user_token_count = len(tokenizer.encode(s=assistant_message.content, bos=True, eos=True))
    print(f"dump_strt token number: {user_token_count}")

    dump_str = "{" + assistant_message.content.split("{",1)[-1]
    # dump_str = dump_str.replace("\\n","")
    # dump_str = dump_str.replace("\\","")

    try:
        dump_str = json.loads(dump_str)
    except:
        print("Cannot loads")

    with open('output.json', 'w') as outfile:
        json.dump(dump_str, outfile)


def main():
    fire.Fire(llama_converter)

if __name__ == "__main__":
    main()
