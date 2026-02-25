<div align="center">

# SOAR-Toolkit

<a href="README.md">中文</a> | <a href="README_EN.md">English</a>

</div>

---

This competition focuses on optimizing the inference performance of the OpenBMB MiniCPM-SALA model. Participants must optimize based on the officially provided MiniCPM-SALA model on designated hardware environments. Submission or replacement of any base model is not permitted. To help participants with self-testing, we provide the following evaluation tools for verification and planning. These mainly include:

# Base Image

## Image Download

We provide a base image capable of running the model and performing inference (with SGLang and necessary dependencies built in), for local development, debugging, and self-evaluation.

```bash
# Download in China (Alibaba Cloud ACR)
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest

# Download overseas (GitHub)
docker pull ghcr.io/openbmb/soar-toolkit:latest
```



## MiniCPM-SALA Model Download

### Method 1: Download via Hugging Face
Before downloading, install the official Hugging Face CLI tool with the following command:

```plain text
pip install huggingface_hub
```

Download the complete model repository to the specified path `./models`:

```plain text
huggingface-cli download OpenBMB/MiniCPM-SALA --local-dir ./models
```

### Method 2: Download via ModelScope
Before downloading, install ModelScope with the following command:

```plain text
pip install modelscope
```

Download the complete model repository to the specified path `./models`:

```plain text
modelscope download --model OpenBMB/MiniCPM-SALA --local_dir ./models
```



## Container Mount Path

Model: `/models/MiniCPM-SALA`



## Container Startup Script Reference

### Common `docker run` Parameters (Docker General Options)

### Environment Variables (for public serving startup)

```bash
# Reference docker startup command
docker run -d \
  --name soar-sglang-server \
  --gpus 'device=0' \
  -p 30000:30000 \
  -e SGLANG_SERVER_ARGS=[optional/custom deployment parameters, uses model defaults if not set]'--xxxxx' \
  -v ~/models/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
  modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest 
```

> There may be differences between local and online evaluation environments. Final scores are based on the official evaluation environment.



### Notes

* In `SGLANG_SERVER_ARGS`, please use hyphenated parameter names: e.g., `--dense-as-sparse`, not `--dense_as_sparse` (the image does not automatically convert underscores).



# Model Correctness Evaluation

To verify that participants' optimizations of the inference code do not affect the model's correctness, we evaluate the model's scores on specific datasets. We publicly release the dataset `perf_public_set.jsonl` and the evaluation script `eval_model.py` used for correctness evaluation. Participants can also use this dataset for self-verification.

## perf\_public\_set.jsonl

Download: <https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_dataset/perf_public_set.jsonl>

This dataset contains multiple-choice and information extraction questions of varying lengths, comprehensively testing the overall performance of the model. During evaluation of submitted code, we will verify the model's score on this dataset to determine whether model capability has degraded during modifications. The file contains the following fields:

* **`task`**: Task type

* **`question`**: Input prompt text

* **`gold`**: Reference answer / keyword list, etc. (meaning varies by task type)

Example:

```json
{"question":"...question text...", "task":"mcq", "gold":"B"}
```

> To prevent potential score manipulation, we maintain an internal private set `perf_private_set.jsonl`. Both datasets have the same length distribution and tasks, and their scores are similar on the original model inference. The private set is primarily used to detect whether the model performs significantly differently on the two datasets, ensuring competition fairness.

## eval\_model.py

Download: https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval\_model.py

`eval_model.py` evaluates model correctness scores by calling a running SGLang inference service, based on different evaluation task types. First, start the SGLang service and pass in the `api_base` used by the model:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32 
```

Parameter description (commonly used):

* **`--api_base`**: SGLang service address

* **`--model_path`**: Model path

* **`--data_path`**: Dataset path

* **`--concurrency`**: (optional) Number of concurrent requests

* **`--num_samples`**: (optional) Maximum number of evaluation samples (can be used for few-sample testing during debugging)

* **`--verbose`**: (optional) Print more detailed information for each sample



# Model Speed Evaluation

## bench\_serving.sh

Download: https://github.com/OpenBMB/SOAR-Toolkit/blob/main/bench\_serving.sh

This script uses the official sglang `bench_serving` tool to run all evaluation requests at 3 concurrency levels and record the Benchmark Duration. By passing in the dataset path at the corresponding level, you can complete testing for that level. If no dataset path is provided, that level's test will be skipped. The parameters and descriptions are as follows:

| **Parameter**     | **Required** | **Description**                                        | **Example**                                              |
| ----------------- | ------------ | ------------------------------------------------------ | -------------------------------------------------------- |
| API\_BASE         | Yes          | Model service address                                  | <http://127.0.0.1:30000>                                 |
| SPEED\_DATA\_S1   | No           | S1 level dataset (concurrency=1), accepts JSONL path   | /path/to/speech.jsonl (skipped if not set)               |
| SPEED\_DATA\_S8   | No           | S8 level dataset (concurrency=8), accepts JSONL path   | /path/to/speech.jsonl (skipped if not set)               |
| SPEED\_DATA\_SMAX | No           | Smax level dataset (no concurrency limit), JSONL path  | /path/to/speech.jsonl (skipped if not set)               |

To ensure the validity and fairness of competition results, the dataset used for speed testing in the competition is not publicly provided at this time. The question length distribution can be referenced from the competition problem statement. Our testing method uses fixed model inputs and outputs to measure inference speed. Participants can construct a `.jsonl` file with the following fields for self-testing:

```python
{"question": "question content...", "model_response": "model response content..."}
```

Usage example:

```bash
export SPEED_DATA_S1=/path/to/speech.jsonl
export SPEED_DATA_S8=/path/to/speech.jsonl
export SPEED_DATA_SMAX=/path/to/speech.jsonl

bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

# Model Quantization / Speculative Sampling

This competition encourages participants to apply quantization or speculative sampling to the model. We will provide entry points and script demos in subsequent updates.

# Build Notes

The default build process of the upstream `sgl-kernel` repository pulls and packages multiple third-party components (DeepGEMM, Flash Attention, Triton Kernels, etc.). **The inference pipeline for this track (MiniCPM-SALA) only depends on the operator implementations in the `sgl_kernel/` directory.** The aforementioned third-party components will not be called at runtime.

The evaluation environment's validation script **only allows files with the `sgl_kernel/` and `sgl_kernel.libs/` prefixes**. Wheels containing other top-level directories (such as `deep_gemm/`, `flash_attn_origin/`, `triton_kernels/`, `include/`, `lib/`, etc.) **will fail validation and be immediately marked as submission failures**.

Therefore, participants **must** make the following adjustments to `CMakeLists.txt` before building the wheel:

1. Remove or comment out `FetchContent_Declare` / `FetchContent_Populate` directives related to `deep_gemm`, `flash_attn_origin`, and `triton_kernels`

2. Remove or comment out the corresponding `add_library`, `target_link_libraries`, `install(TARGETS ...)`, and `install(DIRECTORY ...)` directives

The final wheel file structure should only contain:

* `sgl_kernel/` — Core operator build artifacts (required)

* `sgl_kernel.libs/` — Dynamic library dependencies (if any, auto-generated by build tools)

* `sgl_kernel-*.dist-info/` — Package metadata (auto-generated)
