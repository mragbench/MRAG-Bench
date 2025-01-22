
# MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models

[**🌐 Homepage**](https://mragbench.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/uclanlp/MRAG-Bench) | [**📖 Paper**](https://arxiv.org/abs/2410.08182) | [**💻 Evaluation**](https://github.com/mragbench/MRAG-Bench) 



## News

* 🔥 [2025-01-22] MRAG-Bench is accepted at ICLR 2025.
* Todo: Coming, integrate MRAG-Bench to [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), enabling rapid evaluation on Large Vision Language Models.
* [2024-11-05] As many people requesting, we release the image corpus [here](https://drive.google.com/file/d/1atwkNXH3aEtCLuqimZoB1Mifj5CwL3CL/view?usp=sharing) for retrieval.
* 🔥 [2024-10-10] MRAG-Bench evaluation code is released.
* 🔥 [2024-10-10] MRAG-Bench is released.


## Intro

MRAG-Bench consists of 16,130 images and 1,353 human-annotated multiple-choice questions across 9 distinct scenarios,  providing a robust and systematic evaluation of Large Vision Language Model (LVLM)’s vision-centric multimodal retrieval-augmented generation (RAG) abilities.

<img src="https://gordonhu608.github.io/images/mragbench_teaser.png" width="1000" />


## Results

Evaluated upon 10 open-source and 4 proprietary LVLMs, our results show that all LVLMs exhibit greater improvements when augmented with images compared to textual knowledge. Notably, the top-performing model, GPT-4o, faces challenges in effectively leveraging retrieved knowledge, achieving only a 5.82% improvement with ground-truth information, in contrast to a 33.16% improvement observed in human participants. These findings highlight the importance of MRAG-Bench in encouraging the community to enhance LVLMs' ability to utilize retrieved visual knowledge more effectively.

<img src="https://gordonhu608.github.io/images/mragbench_qual.png" width="800" />


## Load Dataset

```python
from datasets import load_dataset
mrag_bench = load_dataset("uclanlp/MRAG-Bench", split="test")
```

## Evaluation 

We provide an example evaluation code for LLaVA-OneVision-7B. First, install llava-onevision environment following [here](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision_Tutorials.ipynb). Please refer to our [scripts](eval/models/run_model.sh) for setting the model output path, use rag option and use retrieved examples option. By default, use rag means use ground-truth rag examples. Then run, 

```shell
bash eval/models/run_model.sh 
```

With model's results file, then please run

```python
python eval/score.py -i "path to results file"
```

For most models, our [automatic](eval/utils/automatic_extract.py) pipeline can handle the answer extraction job. However, in cases when gpt based answer extration is needed, please set your openai api key [here](eval/utils/gpt_extract.py#L14). We use openai==0.28.1 version for sending request. 

## Contact

* Wenbo Hu: whu@cs.ucla.edu

## Citation
```
@article{hu2024mragbench,
  title={MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models},
  author={Hu, Wenbo and Gu, Jia-Chen and Dou, Zi-Yi and Fayyaz, Mohsen and Lu, Pan and Chang, Kai-Wei and Peng, Nanyun},
  journal={arXiv preprint arXiv:2410.08182},
  year={2024}
}
```