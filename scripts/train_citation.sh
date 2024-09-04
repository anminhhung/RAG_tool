CUDA_VISIBLE_DEVICES=3 python3 finetuning/citation_generator_ft.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --data_path './outputs/generated_data.json' \
    --output_path './outputs/llama3_8b_citation'
