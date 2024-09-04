CUDA_VISIBLE_DEVICES=2,3 python3 generate_data.py \
    --dataset_name="BachNgoH/ParsedArxivPapers_12k" \
    --load_local=True \
    --service="groq" \
    --model_name="llama3-8b-8192" \
    --output_path="./outputs" \
