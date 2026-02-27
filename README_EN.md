<div align="center">

# SOAR-Toolkit

<a href="README.md">中文</a> | <a href="README_EN.md">English</a>

</div>

---

This competition focuses on optimizing the inference performance of the OpenBMB MiniCPM-SALA model. Participants are required to optimize based on the officially provided MiniCPM-SALA model on the specified hardware environment, and are not allowed to submit or replace any base models. To facilitate self-testing for participants, we provide the following evaluation tools to help everyone verify and plan reasonably. The main tools include:

# Base Image

## Image Download

We provide a base image (with built-in SGLang and necessary dependencies) capable of running the model and performing inference, used for local development, debugging, and evaluation self-testing.

```bash
# Domestic Download (Aliyun ACR)
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest

# Overseas Download (Github)
docker pull ghcr.io/openbmb/soar-toolkit:latest
```

## MiniCPM-SALA Model Download

### Method 1: Download via Hugging Face
Before downloading, please install the official Hugging Face CLI tool using the following command.
```bash
pip install huggingface_hub
```
Download the complete model repository to the specified path folder `./models`
```bash
huggingface-cli download OpenBMB/MiniCPM-SALA --local-dir ./models
```

### Method 2: Download via ModelScope
Before downloading, please install ModelScope using the following command.
```bash
pip install modelscope
```
Download the complete model repository to the specified path folder `./models`
```bash
modelscope download --model OpenBMB/MiniCPM-SALA --local_dir ./models
```

## Container Mount Address
Model: `/models/MiniCPM-SALA`

## Container Startup Script Reference

### Common docker run parameters
*(This content cannot be displayed outside of Feishu documents temporarily)*

### Environment Variables
*(This content cannot be displayed outside of Feishu documents temporarily)*

```bash
# Reference docker run command
docker run -d \
  --name soar-sglang-server \
  --gpus 'device=0' \
  -p 30000:30000 \
  -e SGLANG_SERVER_ARGS=[optional/custom deployment parameters, uses model default parameters when not set]'--xxxxx' \
  -v ~/models/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
  modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest 
```
> There may be differences between local and online evaluation environments. The final score is subject to the official evaluation environment.

### Notes
- Please use hyphenated parameter names in `SGLANG_SERVER_ARGS`: for example `--dense-as-sparse`, do not write `--dense_as_sparse` (the image does not automatically convert underscores).

# Evaluation Environment
To facilitate participants in carrying out targeted optimization work, the evaluation environment is described as follows:
- The total size of all files submitted by participants must not exceed 2GB.
- The resource limit for a single evaluation task is: 20 CPUs, 128 GiB memory.
- The maximum execution time for a single evaluation task is 5 hours (excluding queue waiting time).
- Hardware environment: This competition uniformly uses NVIDIA high-end RTX PRO single-card GPUs for evaluation.

# Model Correctness Evaluation
To verify that the optimization of inference code by participants does not affect the model's performance in terms of correctness, we evaluate by testing the model's score on a specific dataset. Here we publicly release the dataset `perf_public_set.jsonl` used for correctness evaluation and the script `eval_model.py` used for correctness evaluation. Participants can also use this dataset for self-checking.

## perf_public_set.jsonl
Download address: https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_dataset/perf_public_set.jsonl

This dataset contains multiple-choice questions or information extraction questions of different lengths, capable of comprehensively testing the overall performance of the model. MiniCPM-SALA scores about 82±2 points on this dataset, and we take 80 points as the baseline score. During the evaluation of the code submitted by participants, we will verify the relative score of the model on this dataset compared to the baseline score to judge whether the model capability has declined during the modification process. The file contains the following fields:
- `task`: Task type
- `question`: Input prompt text
- `gold`: Reference answer/keyword list, etc. (meaning varies by task type)

Example:
```json
{"question":"...question text...", "task":"mcq", "gold":"B"}
```
> To avoid possible score cheating, we will prepare a private set `perf_private_set.jsonl` internally. The length distribution and tasks of the two datasets are consistent, and the scores in the original model inference results are similar. It is mainly used to check whether the model has a large gap between the two datasets to ensure the fairness of the competition.

## eval_model.py
Download address: https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_model.py

`eval_model.py` will call the started SGLang inference service to give the model's evaluation score on correctness according to different evaluation task types. Finally, `ori_accuracy` represents the original score of the model on the dataset, and `overall_accuracy` represents the score relative to the above baseline score (80 points), judging whether the inference code affects the correctness effect of the model. First, you need to start the SGLang service and pass in the `api_base` used by the model:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32 
```
Parameter description (common):
- `--api_base`: SGLang service address
- `--model_path`: Model path
- `--data_path`: Dataset path
- `--concurrency`: (optional) Number of concurrent requests
- `--num_samples`: (optional) Maximum number of evaluation samples (can perform few-shot testing during debugging)
- `--verbose`: (optional) Print more detailed information for each sample

# Model Speed Evaluation

## bench_serving.sh
Download method: https://github.com/OpenBMB/SOAR-Toolkit/blob/main/bench_serving.sh

This script uses the sglang official `bench_serving` tool to run all evaluation requests at 3 concurrency levels respectively, recording the Benchmark Duration. Passing the dataset path in the corresponding level can complete the test of the corresponding level. If the dataset path is not input, the test of that level can be skipped. The relevant parameters and descriptions are as follows:

| Parameter | Required | Description | Example |
| :--- | :--- | :--- | :--- |
| API_BASE | Yes | Model service address | http://127.0.0.1:30000 |
| SPEED_DATA_S1 | No | S1 level dataset (concurrency=1), can pass JSONL path | /path/to/speech.jsonl (skip this test when not set) |
| SPEED_DATA_S8 | No | S8 level dataset (concurrency=8), can pass JSONL path | /path/to/speech.jsonl (skip this test when not set) |
| SPEED_DATA_SMAX | No | Smax level dataset (no concurrency limit), can pass JSONL path | /path/to/speech.jsonl (skip this test when not set) |

To ensure the validity and fairness of the competition results, the dataset used for speed testing in the competition is not provided here for the time being. The length distribution of the questions can refer to the competition questions. Our testing method is to test the speed of the model through fixed model input and output. Participants can construct `.jsonl` files through the following fields for self-testing:
```json
{"question": "question content...", "model_response": "model response content..."}
```
Usage example:
```bash
export SPEED_DATA_S1=/path/to/speech.jsonl
export SPEED_DATA_S8=/path/to/speech.jsonl
export SPEED_DATA_SMAX=/path/to/speech.jsonl

bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

# Submission Instructions

## Submission Method Upgrade
To provide participants with higher freedom, we have upgraded the submission method—participants can now freely customize the runtime environment and model preprocessing workflow.

## Submission Requirements
Participants need to package all codes and resources into a `.tar.gz` file, which must contain:

| File | Required | Description |
| :--- | :--- | :--- |
| `prepare_env.sh` | Required | Environment build script |
| `prepare_model.sh` | Optional | Model preprocessing script |
| Other code/resources | As needed | Organized by participants |

## Execution Process
1. **Environment Build**: After the platform starts the base environment, it will automatically execute the `prepare_env.sh` provided by the participant to install the dependencies and configurations required by the participant on top of the base environment.
2. **Model Preprocessing (if provided)**: After the environment is ready, the platform will call `prepare_model.sh` to process the original model and output it to the specified path for subsequent inference phase use.

`prepare_model.sh` must support the following two parameters:
```bash
bash prepare_model.sh --input <original model path> --output <processed model path>
```

| Parameter | Description |
| :--- | :--- |
| `--input` | The original model path provided by the platform, the script reads the model from this path |
| `--output` | The output path specified by the platform, the script must write the processed model to this path |

## Notes
- The base environment uses `uv` as the package manager. When executing operations like `pip install`, please use `uv pip install` instead, and be sure to fully test locally before submitting.
- **Migration Hint for Old Submission Method**: Participants who previously submitted using the wheel package method, please adapt the original logic to the new `prepare_env.sh` + `prepare_model.sh` method. For example, migrate the installation command of the wheel to `prepare_env.sh` for execution, and the original model processing logic can be migrated to `prepare_model.sh`. The old wheel submission method will no longer be supported.

# Submission Demo

This directory is a minimal runnable submission example, demonstrating how to organize the `prepare_env.sh` + `prepare_model.sh` submission package according to platform requirements.

## Directory Structure
```
├── prepare_env.sh          # Required — Environment build script
├── prepare_model.sh        # Optional — Model preprocessing entry
├── preprocess_model.py     # Python script called by prepare_model.sh
└── sglang/python/          # Custom sglang source code (editable install)
```

## File Descriptions

### prepare_env.sh (Required)
The platform automatically executes this script after the base environment starts. This demo does two things:
1. Use `uv pip install --no-deps -e ./sglang/python` to install custom sglang in editable mode, replacing the built-in version in the image.
2. Append inference startup parameters via `export SGLANG_SERVER_ARGS` (example adds `--log-level info`).

```bash
uv pip install --no-deps -e ./sglang/python
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"
```
> Note: `prepare_env.sh` will be sourced into the platform main script, so exported environment variables can take effect directly.

### prepare_model.sh (Optional)
The platform calls this script after the environment is ready, the interface is fixed as:
```bash
bash prepare_model.sh --input <original model path> --output <processed model path>
```
Both paths are provided by the platform, and participants do not need to care about the specific mount location in the container. This demo only does simple model file copying without any quantization or conversion.
In actual competition, you can implement preprocessing logic such as quantization (GPTQ, AWQ, etc.), pruning, weight fusion, etc. in `preprocess_model.py`.

### sglang/python/
Custom sglang source code directory. Through editable install, the platform will use the code in this directory to replace the built-in sglang in the image, and participants can modify the implementation of the inference engine here.

## Extension Examples

| Scenario | Modification Point |
| :--- | :--- |
| Install extra pip packages | Add `uv pip install xxx` in `prepare_env.sh` |
| Custom inference parameters | Modify `SGLANG_SERVER_ARGS` in `prepare_env.sh` |
| GPTQ Quantization | Implement GPTQ packaging in `preprocess_model.py`, append `--quantization gptq` in `prepare_env.sh` |
| Model Pruning/Distillation | Implement in `preprocess_model.py`, output to `--output` directory |

# Technical Path Guidance

This competition encourages participants to explore optimization around model inference performance, including but not limited to technical directions such as quantization compression and speculative sampling. To facilitate participants' understanding and practice, we provide the following two reference technical ideas.

## Path 1: Quantization Acceleration
**Optional Path: GPTQ W4A16 + Marlin Kernel + FP8 KV Cache**

Quantize model weights to 4-bit (W4A16), use Marlin high-performance dequantization GEMM Kernel to accelerate Linear calculation; Quantize KV Cache to FP8 to reduce video memory bandwidth bottleneck in Decode phase. The quantization tool (GPTQModel) works out of the box, SGLang has perfect support for GPTQ + Marlin, and participants only need to submit the quantization script to quantize on the evaluation machine on site.

**Workflow:**
1. **Write Quantization Script**: Use GPTQModel to perform W4A16 quantization (group_size=128) on SALA FP16 weights, and the script serves as a submission.
2. **Verify Correctness**: Start with `--quantization gptq_marlin`, confirm accuracy > 97%; if the drop is severe, fallback to W8.
3. **Enable KV Cache FP8**: `--kv-cache-dtype fp8_e5m2`, significant benefits in long context scenarios. Note that Lightning Attention layer uses independent linear attention state, optimization path is different.
4. **Advanced Tuning**: Adjust Marlin Kernel's tile / warp configuration for 6000D.

**Core files that may need reading and modification:**
- **Model and Loading**
  - `python/sglang/srt/models/minicpm_sala.py` — SALA model definition
  - `python/sglang/srt/model_loader/loader.py`, `weight_utils.py` — Weight loading and quantization mapping
  - `python/sglang/srt/server_args.py` — Startup parameters (`--quantization`, `--kv-cache-dtype`)
- **Quantization Methods**
  - `python/sglang/srt/layers/quantization/gptq.py` — GPTQ linear layer and Marlin scheduling
  - `python/sglang/srt/layers/quantization/__init__.py` — Quantization method registry
  - `python/sglang/srt/layers/linear.py` — Main Linear layer, quantization method injection point
- **CUDA Operators**
  - `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`, `marlin_template.h` — Marlin W4A16 GEMM Kernel
  - `sgl-kernel/csrc/gemm/gptq/gptq_kernel.cu` — GPTQ dequantization Kernel
- **KV Cache Quantization**
  - `python/sglang/srt/layers/quantization/kv_cache.py` — KV Cache quantization logic

## Path 2: Speculative Sampling
**Optional Path: EAGLE3 Multi-layer Draft Head (Requires Algorithm Innovation)**

EAGLE3 uses a lightweight Draft Head to predict candidate tokens using the target model's hidden states, and then validates them all at once by the target model. The Head parameter size is extremely small (tens of MB), meeting the 2GB limit.

**Lightning Attention Compatibility Challenge**: Some layers of SALA use Lightning Attention (Gated Delta Rule linear attention), whose recursive calculation essentially does not support tree-like causal masks, so the traditional tree verification mechanism cannot take effect directly. SGLang has preliminary integration (`is_target_verify` mode is handled in `hybrid_linear_attn_backend.py`), but participants may still need to innovate at the algorithm level, and innovative solutions are encouraged.

**Workflow:**
1. **Train EAGLE3 Head**: Refer to the EAGLE repository, use SALA to collect hidden states to train Draft Head, weights as submission.
2. **Model Adaptation**: Refer to `llama_eagle3.py` to create EAGLE3 model file for SALA.
3. **Solve Lightning Attention Verification Problem**: Core difficulty, need to modify verification logic to adapt to linear attention layer.
4. **Start Verification**: `--speculative-algorithm EAGLE3 --speculative-draft-model-path <path> --speculative-num-draft-tokens 5`.

**Core files that may need reading and modification:**
- **EAGLE3 Pipeline**
  - `python/sglang/srt/speculative/multilayer_eagleworker.py` — EAGLE3 Draft Worker main loop
  - `python/sglang/srt/speculative/eagle_utils.py` — Tree mask construction and Verify function (key point for adapting linear attention)
  - `python/sglang/srt/speculative/eagle_info.py` — Verify / Draft data structure
  - `python/sglang/srt/speculative/multi_layer_eagle_utils.py` — EAGLE3 Triton Kernel
- **Model Adaptation**
  - `python/sglang/srt/models/llama_eagle3.py` — Reference template: LLaMA EAGLE3 implementation
  - `python/sglang/srt/models/minicpm_sala.py` — SALA target model (need to create EAGLE3 version based on this)
- **Lightning Attention (Key to understanding verification compatibility)**
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` — Hybrid linear attention Backend (preliminary spec integration exists)
  - `python/sglang/srt/layers/radix_linear_attention.py` — Linear attention layer interface
  - `python/sglang/srt/layers/attention/fla/chunk.py`, `fused_recurrent.py` — Linear attention Triton Kernel
  - `python/sglang/jit_kernel/cutedsl_gdn.py` — GDN CUDA Kernel
- **CUDA Operators**
  - `sgl-kernel/csrc/speculative/eagle_utils.cu` — Tree construction + Verify Kernel
  - `sgl-kernel/csrc/speculative/speculative_sampling.cu` — Sampling Kernel

# Quantization Demo

## Scope and Positioning
This example provides a standard W4A16 quantization access reference link, using RTN strategy to verify the model's loadability and basic inference capability in the platform environment.
**Please note that this codebase serves only as a reference example for running through the process, and does not represent the final performance optimization plan or accuracy benchmark.**

## Access Specifications
- **Weight Format**: Preprocessing products must strictly follow the GPTQ structure standard (including qweight, scales, qzeros, g_idx, etc.), and ensure that the model directory contains an independent `quantize_config.json` configuration file to ensure that SGLang correctly identifies the quantization path.
- **Precision Alignment**: To ensure operator compatibility, weights other than the quantized layers need to be uniformly converted to float16.
- **Environment Adaptation**: The running script has preset `--quantization gptq` and `--dtype float16` parameters, and has adapted the float16 branch for MiniCPM's sparse attention mechanism. It is recommended to keep `--disable-cuda-graph` enabled to ensure stability for the first access.

## Running Tips
Limited by the relatively basic implementation logic of this example, the model is prone to generating ultra-long generation sequences during the inference process, which in turn triggers timeout phenomena.
Therefore, this code is for process run-through reference only and is not recommended to be submitted directly as the final result.
