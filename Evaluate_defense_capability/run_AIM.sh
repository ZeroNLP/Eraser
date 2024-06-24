cd ./start/

CUDA_VISIBLE_DEVICES=2 python AIM.py \
    --model_path "../model/Llama2-7b-chat-hf/" \
    --lora_path "../model/Eraser_Llama2_7b_Lora/" \
    --output_path "../output/Eraser_Llama2_7b_Lora.json" \
    --data_path "../datasets/AdvBench.csv"\
    --restore_index 0
