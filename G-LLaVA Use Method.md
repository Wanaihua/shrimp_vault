

# G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model

This repository contains the code and data for the paper titled "G-LLaVA: Solving Geometric Problem with Multi-Modal Large
Language Model".

[Paper](https://arxiv.org/pdf/2312.11370.pdf), [Dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main) , Models([G-LLaVA-7B](https://huggingface.co/renjiepi/G-LLaVA-7B), [G-LLaVA-13B](https://huggingface.co/renjiepi/G-LLaVA-13B))


# 以下方法采用G-LLaVA-7B模型

## 先电脑下载Node.js以及Git.
## 再到电脑项目保存路径文件夹右键git bash here,输入
```
git clone https://github.com/pipilurj/G-LLaVA.git
```
## 安装项目依赖
```
cd G-LLaVA
conda create -n gllava python=3.10 -y
conda activate gllava
pip install -e .
pip install deepspeed
```

## 训练数据集准备

Download our [dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main).

Place the data under playground/data.
Here is the data structure:
```
playground/data/
├── images/
│   ├── geo3k/
│   ├── geoqa_plus/
│   ├── test/
├── alignment.json
├── qa_tuning.json
├── test_question.jsonl
├── test_answers.jsonl
```

## 第一步对齐训练
This stage enables the model to better interpret the content of geometric figures.
```
bash scripts/run_alignment.sh
```

## 第二步微调训练
This stage equips the model with stronger ability for solving geometry problems.

```
bash scripts/run_qa.sh
```

## 