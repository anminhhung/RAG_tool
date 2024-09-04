from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.groq import Groq
from llama_index.llms.vllm import Vllm
from vllm import SamplingParams
from tqdm import tqdm
import json
from datasets import load_dataset
from utils.helper import has_citation, parse_json
from GemmaResearchAssistant.src.prompts import DEFAULT_CITATION_INFER_PROMPT_TEMPLATE as prompt_template
import random

from dotenv import load_dotenv, dotenv_values
random.seed(42)
import argparse
import os
from vllm import SamplingParams
load_dotenv()


def load_model(llm_service, model_name):

    if llm_service == "openai":
        llm = OpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    elif llm_service == "hf":
        from unsloth import FastLanguageModel

        max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            token = os.getenv("HUGGINGFACE_API_KEY"), # use one if using gated models like meta-llama/Llama-2-7b-hf
            cache_dir = "../models",
        )
        FastLanguageModel.for_inference(model) 

        llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
    elif llm_service=="vllm":
        llm = Vllm(model=model_name, temperature=0.8, tensor_parallel_size=2)
    elif llm_service == "groq":
        llm = Groq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
    return llm

def generate_relationships(args):
    sentence_splitter = SentenceSplitter.from_defaults(chunk_size=256, chunk_overlap=0)
    llm = load_model(args.service, args.model_name)
    generated_data = []
    all_articles = []

    if args.load_local:
        with open(f'{args.output_path}/parsed_arxiv_papers.json', 'r') as f:
            all_articles = json.load(f)
    else:
        all_articles = load_dataset(args.dataset_name)['train']
        all_articles = all_articles.to_list()

    print("TOTAL len of all articles: ",  len(all_articles))

    for article in tqdm(all_articles[400:800], total=len(all_articles[400:800])):
        chunks = []
        article['citation_data'] = []
        for section in article['sections']:
            if len(section['publication_ref']) > 0:
                res = sentence_splitter.split_text(section['text'])
                # chunks += res
                chunks += [c for c in res if has_citation(c)]
        if args.service == "vllm":
            sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
            completion = llm._client.generate([prompt_template.format(input=chunk) for chunk in chunks], sampling_params)
            citation_data = [parse_json(c.outputs[0].text) for c in completion]
            generated_data += [{'Input': chunk, 'Output': output} for chunk, output in zip(chunks, citation_data)]
            article['citation_data'] += citation_data
        else:
            for chunk in chunks:
                try:
                    completion = llm.complete(prompt_template.format(input=chunk))
                    citation_data = parse_json(completion.text)
                    generated_data.append({'Input': chunk, 'Output': citation_data})

                    article['citation_data'] += citation_data
                except Exception as e:
                    print(f'Error in chunk: {e}')
                    continue
        # save generated_data
        with open(f'{args.output_path}/generated_data.json', 'w') as f:
            json.dump(generated_data, f)

        # save new article data
        with open(f'{args.output_path}/parsed_article_citation.json', 'w') as f:
            json.dump(all_articles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BachNgoHParsedArxivPapers_12k')
    parser.add_argument('--load_local', type=bool, default=False)
    parser.add_argument('--service', type=str, default='hf')
    parser.add_argument('--model_name', type=str, default='Gemma_7b_Citation_v2')
    parser.add_argument('--output_path', type=str, default='../outputs')
    args = parser.parse_args()
    generate_relationships(args)
            
