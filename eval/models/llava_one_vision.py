import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from PIL import Image
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataloader import bench_data_loader 

def eval_model(args):

    ans_file = open(args.answers_file, "w")
    
    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)

    model.eval()

    for item in bench_data_loader(args, image_placeholder=DEFAULT_IMAGE_TOKEN):
        
        qs = item['question']
        conv_template = "qwen_1_5"
        args.conv_mode = "qwen_1_5"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        image_tensors = process_images(item['image_files'], image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

        image_sizes = [image.size for image in item['image_files']]

        with torch.inference_mode():
            cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs = text_outputs[0]

        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "qs_id": item['id'],
                                   "prompt": item['prompt'],
                                   "output": outputs,
                                   "gt_answer": item['answer'],
                                   "shortuuid": ans_id,
                                   "model_id": 'llava_one_vision_qwen_7b',
                                   "gt_choice": item['gt_choice'],
                                   "scenario": item['scenario'],
                                   "aspect": item['aspect'],
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    ############# added for mrag benchmark ####################
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--use_rag", type=lambda x: x.lower() == 'true', default=False, help="Use RAG")
    parser.add_argument("--use_retrieved_examples", type=lambda x: x.lower() == 'true', default=False, help="Use retrieved examples")

    args = parser.parse_args()

    eval_model(args)

