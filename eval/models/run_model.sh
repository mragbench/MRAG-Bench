
# export HF_HOME="./cache"
# export HF_HUB_CACHE="./cache"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=5

python3 eval/models/llava_one_vision.py \
    --answers-file "llava_one_vision_gt_rag_results.jsonl" \
    --use_rag True \
    --use_retrieved_examples False 