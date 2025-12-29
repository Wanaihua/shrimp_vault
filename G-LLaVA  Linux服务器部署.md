

# G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model

This repository contains the code and data for the paper titled "G-LLaVA: Solving Geometric Problem with Multi-Modal Large
Language Model".

[Paper](https://arxiv.org/pdf/2312.11370.pdf), [Dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main) , Models([G-LLaVA-7B](https://huggingface.co/renjiepi/G-LLaVA-7B), [G-LLaVA-13B](https://huggingface.co/renjiepi/G-LLaVA-13B))


# 以下方法采用G-LLaVA-7B模型

## 后续项目保存路径为：/home/lj/wanaihua/，可根据自行修改。


## cd到项目保存路径文件夹输入
```
cd /home/lj/wanaihua/
git clone https://github.com/pipilurj/G-LLaVA.git
```

## 创建conda环境，安装项目依赖（conda默认已安装）
```
cd G-LLaVA
conda create -n gllava python=3.10 -y
conda activate gllava
pip install -e .
pip install deepspeed
```

## LLaVa-7B模型下载(修改modelscope缓存地址，安装在playground/data 下)
```
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

export MODELSCOPE_CACHE=/home/lj/wanaihua/G-LLaVA/playground/data

modelscope download --model huangjianuo/llava-v1.5-7b
```

## VIT视觉编码器安装(export仅在当前终端窗口有效，重新打开终端需重新设置缓存地址)
```
modelscope download --model openai-mirror/clip-vit-large-patch14-336
```

## 训练数据集准备

Download our [dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main).(此处需要挂vpn连接huggingface下载)

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


## 修改运行脚本run_alignment.sh
```
# Enable offline mode to prevent any network calls.

export HF_HUB_OFFLINE=1

export TRANSFORMERS_OFFLINE=1

  

# Allow overriding vision tower via env var `VISION_TOWER_PATH`.

# Default to your local CLIP directory.

VISION_TOWER="${VISION_TOWER_PATH:-本地vit路径}"

export VISION_TOWER_PATH="$VISION_TOWER"

  
# 0,1 代表双卡运行，多卡为0,1,2，...
deepspeed --include=localhost:0,1 gllava/train/train.py \

                                            --mm_projector_lr 1e-5 \

                                            --deepspeed ./scripts/zero3.json \

                                            --model_name_or_path llava-v1.5-7b本地路径 \

                                            --version v1 \

                                            --data_path ./playground/data/alignment.json \

                                            --image_folder playground/data/images \

                                            --vision_tower "${VISION_TOWER}" \

                                            --mm_projector_type mlp2x_gelu \

                                            --mm_vision_select_layer -2 \

                                            --mm_use_im_start_end False \

                                            --mm_use_im_patch_token False \

                                            --image_aspect_ratio pad \

                                            --group_by_modality_length True \

                                            --bf16 True \

                                            --output_dir ./checkpoints/llava1.5_7b_with_alignment \

                                            --num_train_epochs 2 \

                                            --per_device_train_batch_size 6 \

                                            --per_device_eval_batch_size 4 \

                                            --gradient_accumulation_steps 1 \

                                            --evaluation_strategy "no" \

                                            --save_strategy "steps" \

                                            --save_steps 50000 \

                                            --save_total_limit 1 \

                                            --learning_rate 1e-5 \

                                            --weight_decay 0. \

                                            --warmup_ratio 0.03 \

                                            --lr_scheduler_type "cosine" \

                                            --logging_steps 1 \

                                            --tf32 True \

                                            --model_max_length 2048 \

                                            --gradient_checkpointing True \

                                            --dataloader_num_workers 4 \

                                            --lazy_preprocess True \

                                            --freeze_backbone
```

## 第一步对齐
This stage enables the model to better interpret the content of geometric figures.
```
bash scripts/run_alignment.sh
```

## 对齐方法运行顺序
```
train.py->llava_arch.py(initialize_vision_modules)视觉编码器模型侧启动->
train.py(make_supervised_data_module)数据侧预处理->
train.py(LazySupervisedDataset)提供 “图像 + 文本” 对齐的单样本数据->
train.py(DataCollatorForSupervisedDataset)解决数据维度对齐与效率优化问题->
LLaVATrainer(驱动模型前向)->llava_llama.py(forward)->   
llava_arch.py(prepare_inputs_labels_for_multimodal)图像特征与文本嵌入的对齐与拼接。->
得到outputs，接着过lm_head计算logits与可选的loss.
```

## 第二步微调(流程与对齐一致，采取训练数据集不一致)
This stage equips the model with stronger ability for solving geometry problems.

```
bash scripts/run_qa.sh
```

## 附件（测试inputs_embeds对模型输出的影响）
### 可切入测试文件test_inputs_embeds.py,放在scripts文件夹下

```
import argparse
import os
import torch

from transformers import AutoConfig, AutoModelForCausalLM
from gllava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/lj/wanaihua/G-LLaVA/playground/data/models/huangjianuo/llava-v1.5-7b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=str, default="0,1", help="Comma-separated CUDA device indices for DataParallel")
    parser.add_argument("--data-parallel", action="store_true", help="Use nn.DataParallel across specified devices")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", type=str, default="64,128,256")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    
    dev_arg = args.device
    if torch.cuda.is_available() and "cuda" in dev_arg:
        if "," in dev_arg or "，" in dev_arg:
            dev_arg = dev_arg.split(",")[0].split("，")[0]
        if ":" in dev_arg:
            try:
                _ = int(dev_arg.split(":")[1])
                device = torch.device(dev_arg)
            except Exception:
                device = torch.device("cuda")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Loading model from {args.model_path} ...")

    cfg = None
    model_type = None
    
    try:
        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        model_type = getattr(cfg, "model_type", None)
    except Exception as e:
        print(f"Warn: AutoConfig load failed: {e}. Will attempt model load directly.")

    # Load model according to type
    if model_type == "llava" or model_type == "llama":
        vision_tower_path = "/home/lj/wanaihua/G-LLaVA/playground/data/models/openai-mirror/clip-vit-large-patch14-336"
        setattr(cfg, "mm_vision_tower", vision_tower_path)
        setattr(cfg, "mm_vision_select_layer", -2)
        setattr(cfg, "mm_use_im_start_end", False)
        setattr(cfg, "mm_use_im_patch_token", False)
        model = LlavaLlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, config=cfg).to(device)
        llava_mode = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        llava_mode = False
    model.eval()

    dp_ids = None

    if args.data_parallel and torch.cuda.is_available():
        devs = [d.strip() for d in args.devices.replace("，", ",").split(",") if d.strip()]
        try:
            dp_ids = [int(d) for d in devs]
        except Exception:
            dp_ids = [0, 1]
        print(f"Using DataParallel on devices: {dp_ids}")

    if llava_mode:
        base = model.get_model()
        if args.data_parallel and torch.cuda.is_available():
            base = torch.nn.DataParallel(base, device_ids=dp_ids)
    else:
        base = model
    hidden_size = (model.module.config.hidden_size if isinstance(model, torch.nn.DataParallel) else getattr(model.config, "hidden_size", None))

    if hidden_size is None:
        try:
            hidden_size = model.get_input_embeddings().embedding_dim
        except Exception:
            raise RuntimeError("Unable to determine hidden_size from model config or embeddings.")
    dtype = next(model.parameters()).dtype
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    for L in seq_lens:
        B = args.batch_size
        print(f"\nTesting inputs_embeds with shape (B={B}, L={L}, H={hidden_size})")
        inputs_embeds = torch.randn(B, L, hidden_size, dtype=dtype, device=device)
        attention_mask = torch.ones(B, L, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = base(
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        print(f"last_hidden_state: {tuple(outputs.last_hidden_state.shape)}")
        
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            k, v = outputs.past_key_values[0]
            print(f"past_key_values[0][0] (key): {tuple(k.shape)}")
            print(f"past_key_values[0][1] (value): {tuple(v.shape)}"
  
if __name__ == "__main__":
    main()
```
## 启动指令
```
python3 scripts/test_inputs_embeds.py --seq-lens 32,64 --batch-size 2 --device cuda:0 --data-parallel --devices 0,1
```