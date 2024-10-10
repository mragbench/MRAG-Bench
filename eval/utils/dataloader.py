import os
import json
import math
import io
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset

def bench_data_loader(args, image_placeholder="<image>"):
    """ 
    Data loader for benchmarking models
    Args:
        args: arguments
        image_placeholder: placeholder string for image
    Returns:
        generator: a generator that yields data (queries, image paths, ...) for each sample
    """
    # Data
    mrag_bench = load_dataset("uclanlp/MRAG-Bench", split="test")
    
    for item in tqdm(mrag_bench):
        
        qs_id = item['id'] 
        qs = item['question']
        ans = item['answer']
        gt_choice = item['answer_choice']
        scenario = item['scenario']
        choices_A = item['A']
        choices_B = item['B']
        choices_C = item['C']
        choices_D = item['D']
        gt_images = item['gt_images']
        gt_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images]
        
        image = item['image'].convert("RGB") 

        if scenario == 'Incomplete':
            gt_images = [gt_images[0]]
        
        ### our evaluation instuction for all the models 
        if not args.use_rag: 
            prompt = f"Answer with the option's letter from the given choices directly. {image_placeholder}\n"
            image_files = [image]
        else: 
            image_files = [image] + gt_images
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}\n"
            if scenario == 'Incomplete':
                prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}{image_placeholder}\n"

        qs += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}"
        prompt_question_part = qs
        prompt_instruction_part = prompt
        qs = prompt + qs
        
        if args.use_rag: 
            if args.use_retrieved_examples:
                retrieved_images = item['retrieved_images']
                retrieved_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in retrieved_images]
                if scenario == 'Incomplete':
                    retrieved_images = [retrieved_images[0]]
                image_files = [image] + retrieved_images
    
        cur_prompt = args.extra_prompt + qs

        yield {
            "id": qs_id, 
            "question": qs, 
            "image_files": image_files, 
            "prompt": cur_prompt,
            "answer": ans,
            "gt_choice": gt_choice,
            "scenario": scenario,
            "prompt_question_part": prompt_question_part,
            "prompt_instruction_part": prompt_instruction_part,
            "aspect": item['aspect']
        }
