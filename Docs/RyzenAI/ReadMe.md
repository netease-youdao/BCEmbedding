# Getting started with Ryzen AI EP Context Cache

This is an example showing how to export and compile and run the maidalun1020/bce-embedding-base_v1 on AMD's Ryzen AI NPU by new convenient EP Context Cache with ease of usage (start from Ryzen AI 1.5). Validated on AMD Ryzen™ AI 5 340 with 50 NPU TOPS https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-5-340.html



# Activate Ryzen AI conda environment

```bash
#Install Ryzen AI msi with relative NPU driver from https://ryzenai.docs.amd.com/en/latest/inst.html
conda activate ryzen-ai-1.x
```

# Export model to onnx

```bash

pip install accelerate

python export_to_onnx.py --model maidalun1020/bce-embedding-base_v1 --token_length 512 --output_dir bce_onnx_export --opset 17 --model_inputs "input_ids,attention_mask" --model_outputs "token_embeddings,sentence_embedding"

#exported onnx model in bce_onnx_export\bce-embedding-base_v1.onnx
move bce_onnx_export\bce-embedding-base_v1.onnx .\
```

# Compile model and Generate EP Context Cache directly for NPU compatibility

```bash

python .\compile.py  .\bce-embedding-base_v1.onnx

subpartition path = c:\work\PR\bce-embedding-base_v1\vaiml_par_0\0
[Vitis AI EP] No. of Operators :   CPU    13  VAIML   453
[Vitis AI EP] No. of Subgraphs : VAIML     1

# Offload most of Operators to AMD Ryzen AI NPU,  fully leverage 50 NPU TOPS and reduce the CPU/iGPU workload directly

```

# Run NPU inference with better performance with good enough precision

```bash

# Run NPU inference with sanity testing
python .\run_npu_only.py --input_model .\bce-embedding-base_v1_ctx.onnx



# Run NPU inference with public dataset and show powerful performance compared with CPU with good enough precision
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install setuptools==75.1.0
pip install c_mteb==1.1.1
pip install huggingface_hub[hf_xet]

#  NPU 13.70it/s VS CPU 2.60it/s = 5.2X , also accuray is good enough with ndcg and l2_norm

python .\test_perf_accuray.py

100%|███████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:36<00:00,  2.60it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:07<00:00, 13.70it/s]
0.9992787
0.037882492
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:39<00:00,  2.51it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:07<00:00, 12.69it/s]
0.99933785
0.036111623
{'DuRetrieval': {'mteb_version': '1.1.1', 'dataset_revision': None, 'mteb_dataset_name': 'DuRetrieval', 'dev': {'ndcg_at_1': 0.95833, 'ndcg_at_3': 0.97148, 'ndcg_at_5': 0.97551, 'ndcg_at_10': 0.97852, 'ndcg_at_100': 0.97852, 'ndcg_at_1000': 0.97852, 'map_at_1': 0.9375, 'map_at_3': 0.96875, 'map_at_5': 0.97083, 'map_at_10': 0.97188, 'map_at_100': 0.97188, 'map_at_1000': 0.97188, 'recall_at_1': 0.9375, 'recall_at_3': 0.97917, 'recall_at_5': 0.98958, 'recall_at_10': 1.0, 'recall_at_100': 1.0, 'recall_at_1000': 1.0, 'precision_at_1': 0.95833, 'precision_at_3': 0.34028, 'precision_at_5': 0.20625, 'precision_at_10': 0.10417, 'precision_at_100': 0.01042, 'precision_at_1000': 0.00104, 'mrr_at_1': 0.95833, 'mrr_at_3': 0.96875, 'mrr_at_5': 0.97083, 'mrr_at_10': 0.97188, 'mrr_at_100': 0.97188, 'mrr_at_1000': 0.97188, 'evaluation_time': 91.94}}}
DuRetrieval ndcg_at_10: 0.97852

```