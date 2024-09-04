from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset as HFDataset
from unsloth import FastLanguageModel
import torch
import json
import argparse
import random
random.seed(42)


prompt = """You are about to analyze segments of academic texts where authors cite other papers. 

### Instruction
Your task is to identify and articulate the nature of the relationship between the citing paper and the cited works. Focus on understanding why the author has chosen to reference these works and in what context. Use the following guidelines to structure your response:

- Identify the Citation: Start by locating the citation within the text. Citations might appear in various formats, such as numbers in brackets, author names, or footnotes.

- Contextual Clues: Read the sentences or paragraphs surrounding the citation to gather context. Pay attention to verbs and phrases that might indicate the relationship, such as "builds upon," "contradicts," "extends," "employs," or "reviews."

- Classify the Relationship: Based on the context and the way the cited work is discussed, classify the relationship into one of the following categories:

    - Supporting Evidence: The cited work provides foundational data, theories, or methodologies that support the claims or hypotheses of the citing paper.
    - Methodological Basis: The citing paper adopts or adapts the methods, techniques, or procedures from the cited work.
    - Theoretical Foundation: The citing paper leverages theories, models, or frameworks established in the cited work to underpin its own research.
    - Data Source: The citation is used to acknowledge the origin of a dataset, model, or specific information that the citing paper utilizes in its research or analysis. This category highlights the reliance on external data or pre-existing models as a foundational element for the study conducted in the citing paper.
    - Extension or Continuation: The citing paper expands upon the research of the cited work, exploring new dimensions, contexts, or variables.

For each citation, provide a brief summary in the format of JSON that includes:

- The citation number (as presented in the text).
- The category of the relationship (from the classifications provided above).
- A short explanation of how the cited work contributes to the citing paper.

### Input:
{}

### Response:
{}"""

def train_citation_model(args):
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        cache_dir = "../models",
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        
        inputs       = examples["Input"]
        outputs      = examples["Output"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = prompt.format(input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }



    def train():

        data_path = args.data_path
        with open(data_path, 'r') as f:
            list_data_dict = json.load(f)
        for sample in list_data_dict:
            sample["Output"] = str(sample["Output"])

        citation_dataset = HFDataset.from_list(list_data_dict)
        citation_dataset = citation_dataset.map(formatting_prompts_func, batched = True,)


        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = citation_dataset,
            # data_collator=citation_data_collator,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 30,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none",
            ),
        )

        #@title Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer.train()

        model.save_pretrained(args.output_path) # Local saving

    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='unsloth/gemma-1.1-7b-it-bnb-4bit')
    parser.add_argument('--data_path', type=str, default='../outputs/generated_data.json')
    parser.add_argument('--output_path', type=str, default='../outputs/Gemma_7b_Citation')
    args = parser.parse_args()
    train_citation_model(args)