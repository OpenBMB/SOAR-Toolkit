<div align="center">

# SOAR-Toolkit

<a href="README.md">中文</a> | <a href="README_EN.md">English</a>

</div>

---

本次比赛围绕 OpenBMB MiniCPM-SALA 模型的推理性能优化展开。参赛者需在指定的硬件环境上基于官方提供的 MiniCPM-SALA 模型进行优化，不得提交或替换任何基座模型。为了方便各位参赛选手进行自测，我们提供了以下评测工具，帮助大家进行检验和合理规划。主要包括：

# 基础镜像

## 镜像下载

我们提供了可运行模型并进行推理的基础镜像（内置 SGLang 及必要依赖），用于本地开发、调试与评测自测。

```bash
# 国内下载（阿里云ACR）
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest

# 海外下载（Github）
docker pull ghcr.io/openbmb/soar-toolkit:latest
```

## MiniCPM-SALA 模型下载

### 方式一：通过 Hugging Face 下载
在下载前，请先通过如下命令安装 Hugging Face 官方 CLI 工具。
```bash
pip install huggingface_hub
```
下载完整模型库到指定路径文件夹./models
```bash
huggingface-cli download OpenBMB/MiniCPM-SALA --local-dir ./models
```

### 方式二：通过 ModelScope 下载
在下载前，请先通过如下命令安装 ModelScope。
```bash
pip install modelscope
```
下载完整模型库到指定路径文件夹./models
```bash
modelscope download --model OpenBMB/MiniCPM-SALA --local_dir ./models
```

## 容器内挂载地址
模型：`/models/MiniCPM-SALA`

## 容器启动脚本参考

### docker run 常用参数

| 参数 | 作用 | 示例 |
| :--- | :--- | :--- |
| `--gpus` | 选择可见 GPU | `--gpus '"device=0"'` |
| `-v <host>:<container>:ro` | 挂载目录/文件到容器（只读） | `-v /path/to/MiniCPM-SALA:/models/MiniCPM-SALA:ro` |
| `-p <host_port>:<container_port>` | 端口映射 | `-p 30000:30000` |
| `-e KEY=VALUE` | 传入环境变量（用于改启动参数） | `-e SGLANG_SERVER_ARGS='...'` |
| `--name <name>` | 容器命名（便于 `docker logs`） | `--name minicpm_sglang` |
| `-d` | 后台运行 | `-d` |
| `--rm` | 容器退出后自动删除 | `--rm` |

### 环境变量

| 环境变量 | 默认值 | 含义 / 对应 `sglang.launch_server` | 示例 |
| :--- | :--- | :--- | :--- |
| `MODEL_PATH` | `/models/MiniCPM-SALA` | `--model-path` | `-e MODEL_PATH=/models/MiniCPM-SALA` |
| `HOST` | `0.0.0.0` | `--host` | `-e HOST=0.0.0.0` |
| `PORT` | `30000` | `--port` | `-e PORT=30000` |
| `SGLANG_SERVER_ARGS` | `--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse` | 若设置则覆盖默认参数，建议单引号包裹 | `-e SGLANG_SERVER_ARGS='--disable-radix-cache ...'` |

```bash
# 参考docker启动命令
docker run -d \
  --name soar-sglang-server \
  --gpus 'device=0' \
  -p 30000:30000 \
  -e SGLANG_SERVER_ARGS=[optional/可自定义部署参数，未设定时使用模型默认参数]'--xxxxx' \
  -v ~/models/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
  modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest 
```
> 本地与线上评测环境可能存在差异，最终成绩以官方评测环境为准。

### 注意事项
- SGLANG_SERVER_ARGS 里请用连字符参数名：例如 `--dense-as-sparse`，不要写 `--dense_as_sparse`（镜像不做下划线自动转换）。

# 评测环境
为便于参赛选手有针对性地开展优化工作，现对评测环境说明如下：
- 参赛选手提交的全部文件总大小不得超过 2GB。
- 单次评测任务的资源上限为：20 CPU、128 GiB 内存。
- 单次评测任务的最长执行时间为 5 小时（不包含排队等待时间）。
- 硬件环境：本次比赛统一采用 NVIDIA 高端 RTX PRO 单卡 GPU 进行评测。

# 模型正确性评测
为了验证选手们对推理代码的优化不会影响模型在正确性上的表现，我们通过测试模型在特定数据集上的得分来进行评估。这里我们公开评测正确性所用的数据集`perf_public_set.jsonl`以及用于评测正确性的脚本`eval_model.py`，选手们也可以通过该数据集进行自查。

## perf_public_set.jsonl
下载地址：https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_dataset/perf_public_set.jsonl

本数据集包含不同长度的选择题或者信息提取题目，能够综合测试模型的整体性能表现。MiniCPM-SALA 在该数据集上的得分约在 82±2 分，我们取 80 分作为基准分。在对选手提交的代码进行评估的过程中，我们会验证模型在本数据集上的得分相对于基准分的相对分数，以此来判断模型能力在修改过程中是否会有所下降。该文件包含以下字段：
- `task`：任务类型
- `question`：输入 prompt 文本
- `gold`：参考答案/关键词列表等（不同任务类型含义不同）

示例：
```json
{"question":"...题目文本...", "task":"mcq", "gold":"B"}
```
> 为避免可能存在的刷分行为，我们会在内部准备一个私有集`perf_private_set.jsonl`。两个数据集的长度分布和任务一致，在原始模型推理结果中分数相近，主要用于检查模型是否会在两个数据集上存在较大的差距，保证比赛的公平性。

## eval_model.py
下载地址：https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_model.py

`eval_model.py` 会通过调用已启动的 SGLang 推理服务，根据不同评测任务类型，给出模型在正确性上的评测分数，最后得到的`ori_accuracy`表示模型在该数据集上的原始得分，得到`overall_accuracy`表示相对于上述基准分（80 分）的分数，判断推理代码是否会影响模型的正确性效果。首先需要启动 SGLang 服务，并传入模型所使用的api_base：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32 
```
参数说明（常用）：
- `--api_base`：SGLang 服务地址
- `--model_path`：模型路径
- `--data_path`：数据集路径
- `--concurrency`：（optional）并发请求数
- `--num_samples`：（optional）最多评测样本数（调试时可以进行少样本测试）
- `--verbose`：（optional）打印每条样本更详细的信息

# 模型速度评测

## bench_serving.sh
下载方式：https://github.com/OpenBMB/SOAR-Toolkit/blob/main/bench_serving.sh

本脚本使用 sglang 官方 bench_serving 工具，在 3 档并发度下分别跑完所有评测请求，记录 Benchmark Duration。在对应档位传入数据集路径可以完成对应档位的测试，未输入数据集路径的可跳过该档位的测试，相关传参及说明对应如下：

| 参数 | 必填 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| API_BASE | 是 | 模型服务地址 | http://127.0.0.1:30000 |
| SPEED_DATA_S1 | 否 | S1 档位数据集（并发=1），可传入JSONL 路径 | /path/to/speech.jsonl（未设定时跳过该项测试） |
| SPEED_DATA_S8 | 否 | S8 档位数据集（并发=8），可传入JSONL 路径 | /path/to/speech.jsonl（未设定时跳过该项测试） |
| SPEED_DATA_SMAX | 否 | Smax 档位数据集（不设并发上限），可传入JSONL 路径 | /path/to/speech.jsonl（未设定时跳过该项测试） |

为了保证比赛结果的有效性和公平，这里暂不提供比赛中用于速度测试的数据集，题目长度分布可参考赛题。我们测试的方式是通过固定的模型输入和输出来对模型的速度进行测试，选手们可以通过以下字段构造 .jsonl 文件传入进行自测：
```json
{"question": "问题内容...", "model_response": "模型回答内容..."}
```
使用实例：
```bash
export SPEED_DATA_S1=/path/to/speech.jsonl
export SPEED_DATA_S8=/path/to/speech.jsonl
export SPEED_DATA_SMAX=/path/to/speech.jsonl

bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

# 提交说明

## 提交方式升级
为给选手提供更高的自由度，我们对提交方式进行了升级——选手现在可以自由定制运行环境与模型预处理流程。

## 提交要求
选手需将所有代码及资源打包为 `.tar.gz` 文件，其中须包含：

| 文件 | 是否必须 | 说明 |
| :--- | :--- | :--- |
| `prepare_env.sh` | 必须 | 环境构建脚本 |
| `prepare_model.sh` | 可选 | 模型预处理脚本 |
| 其他代码/资源 | 按需 | 选手自行组织 |

## 执行流程
1. **环境构建**：平台启动基础环境后，将自动执行选手提供的 `prepare_env.sh`，在基础环境之上安装选手所需的依赖与配置。
2. **模型预处理（如提供）**：环境就绪后，平台将调用 `prepare_model.sh`，对原始模型进行处理并输出至指定路径，供后续推理阶段使用。

`prepare_model.sh` 须支持以下两个参数：
```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```

| 参数 | 说明 |
| :--- | :--- |
| `--input` | 平台提供的原始模型路径，脚本从该路径读取模型 |
| `--output` | 平台指定的输出路径，脚本须将处理后的模型写入该路径 |

## 注意事项
- 基础环境使用 `uv` 作为包管理器。执行 `pip install` 等操作时，请使用 `uv pip install` 替代，并务必在本地充分测试后再提交。
- **旧提交方式迁移提示**：此前采用 wheel 包方式提交的选手，请自行将原有逻辑适配到新的 `prepare_env.sh` + `prepare_model.sh` 方式。例如，将 wheel 的安装命令迁移至 `prepare_env.sh` 中执行即可，原有的模型处理逻辑可迁移至 `prepare_model.sh`。旧的 wheel 提交方式将不再支持。

# 提交 Demo

下载地址：[demo-sala.tar.gz](demos/demo-sala.tar.gz)

本目录是一个最小可运行的提交示例，演示如何按照平台要求组织 `prepare_env.sh` + `prepare_model.sh` 提交包。

## 目录结构
```
├── prepare_env.sh          # 必须 — 环境构建脚本
├── prepare_model.sh        # 可选 — 模型预处理入口
├── preprocess_model.py     # prepare_model.sh 调用的 Python 脚本
└── sglang/python/          # 自定义 sglang 源码（editable install）
```

## 各文件说明

### prepare_env.sh（必须）
平台在基础环境启动后自动执行此脚本。本 demo 中做了两件事：
1. 用 `uv pip install --no-deps -e ./sglang/python` 将自定义 sglang 以 editable 模式安装，替换镜像内置版本
2. 通过 `export SGLANG_SERVER_ARGS` 追加推理启动参数（示例中添加了 `--log-level info`）

```bash
uv pip install --no-deps -e ./sglang/python
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"
```
> 注意：`prepare_env.sh` 会被 source 进入平台主脚本，因此 export 的环境变量可以直接生效。

### prepare_model.sh（可选）
平台在环境就绪后调用此脚本，接口固定为：
```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```
两个路径均由平台提供，选手无需关心容器内的具体挂载位置。本 demo 中仅做简单的模型文件复制，不做任何量化或转换。
实际参赛时，可以在 `preprocess_model.py` 中实现量化（GPTQ、AWQ 等）、剪枝、权重融合等预处理逻辑。

### sglang/python/
自定义的 sglang 源码目录。通过 editable install，平台会使用此目录下的代码替代镜像内置 sglang，选手可以在此修改推理引擎的实现。

## 扩展示例

| 场景 | 修改点 |
| :--- | :--- |
| 安装额外 pip 包 | `prepare_env.sh` 中添加 `uv pip install xxx` |
| 自定义推理参数 | `prepare_env.sh` 中修改 `SGLANG_SERVER_ARGS` |
| GPTQ 量化 | `preprocess_model.py` 中实现 GPTQ 打包，`prepare_env.sh` 中追加 `--quantization gptq` |
| 模型剪枝/蒸馏 | `preprocess_model.py` 中实现，输出到 `--output` 目录 |

# 技术路径指引

本次比赛鼓励选手围绕模型推理性能进行优化探索，包括但不限于量化压缩、投机采样等技术方向。为便于参赛者理解和实践，我们提供以下两种可参考的技术思路。

## 路径一：量化加速
**可选路径：GPTQ W4A16 + Marlin Kernel + FP8 KV Cache**

将模型权重量化为 4-bit（W4A16），利用 Marlin 高性能反量化 GEMM Kernel 加速 Linear 计算；KV Cache 量化为 FP8 减少 Decode 阶段显存带宽瓶颈。量化工具（GPTQModel）开箱即用，SGLang 对 GPTQ + Marlin 支持完善，选手只需提交量化脚本在评测机上现场量化。

**工作流程：**
1. **编写量化脚本**：使用 GPTQModel 对 SALA FP16 权重做 W4A16 量化（group_size=128），脚本作为提交物。
2. **验证正确性**：`--quantization gptq_marlin` 启动，确认 accuracy > 97%；掉点严重可回退 W8。
3. **开启 KV Cache FP8**：`--kv-cache-dtype fp8_e5m2`，长上下文场景收益显著。注意 Lightning Attention 层使用独立线性注意力状态，优化路径不同。
4. **进阶调优**：针对 6000D 调整 Marlin Kernel 的 tile / warp 配置。

**可能需要阅读和修改的核心文件：**
- **模型与加载**
  - `python/sglang/srt/models/minicpm_sala.py` — SALA 模型定义
  - `python/sglang/srt/model_loader/loader.py`、`weight_utils.py` — 权重加载与量化映射
  - `python/sglang/srt/server_args.py` — 启动参数（`--quantization`、`--kv-cache-dtype`）
- **量化方法**
  - `python/sglang/srt/layers/quantization/gptq.py` — GPTQ 线性层与 Marlin 调度
  - `python/sglang/srt/layers/quantization/__init__.py` — 量化方法注册表
  - `python/sglang/srt/layers/linear.py` — 主 Linear 层，量化方法注入点
- **CUDA 算子**
  - `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`、`marlin_template.h` — Marlin W4A16 GEMM Kernel
  - `sgl-kernel/csrc/gemm/gptq/gptq_kernel.cu` — GPTQ 反量化 Kernel
- **KV Cache 量化**
  - `python/sglang/srt/layers/quantization/kv_cache.py` — KV Cache 量化逻辑

## 路径二：投机采样
**可选路径：EAGLE3 多层 Draft Head（需算法创新）**

EAGLE3 通过轻量 Draft Head 利用目标模型隐藏状态预测候选 token，再由目标模型一次性验证。Head 参数量极小（几十 MB），满足 2GB 限制。

**Lightning Attention 兼容性挑战**: SALA 部分层使用 Lightning Attention（Gated Delta Rule 线性注意力），其递推计算本质上不支持树状因果掩码，传统树验证机制无法直接生效。SGLang 已有初步集成（`hybrid_linear_attn_backend.py` 中处理了 `is_target_verify` 模式），但选手仍可能需要在算法层面创新，鼓励创新方案。

**工作流程：**
1. **训练 EAGLE3 Head**：参考 EAGLE 仓库，用 SALA 收集隐藏状态训练 Draft Head，权重作为提交物。
2. **模型适配**：参考 `llama_eagle3.py` 为 SALA 创建 EAGLE3 模型文件。
3. **解决 Lightning Attention 验证问题**：核心难点，需修改验证逻辑适配线性注意力层。
4. **启动验证**：`--speculative-algorithm EAGLE3 --speculative-draft-model-path <path> --speculative-num-draft-tokens 5`。

**可能需要阅读和修改的核心文件：**
- **EAGLE3 Pipeline**
  - `python/sglang/srt/speculative/multilayer_eagleworker.py` — EAGLE3 Draft Worker 主循环
  - `python/sglang/srt/speculative/eagle_utils.py` — 树掩码构建与 Verify 函数（适配线性注意力的重点）
  - `python/sglang/srt/speculative/eagle_info.py` — Verify / Draft 数据结构
  - `python/sglang/srt/speculative/multi_layer_eagle_utils.py` — EAGLE3 Triton Kernel
- **模型适配**
  - `python/sglang/srt/models/llama_eagle3.py` — 参考模板：LLaMA EAGLE3 实现
  - `python/sglang/srt/models/minicpm_sala.py` — SALA 目标模型（需据此创建 EAGLE3 版本）
- **Lightning Attention（理解验证兼容性的关键）**
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` — 混合线性注意力 Backend（已有初步 spec 集成）
  - `python/sglang/srt/layers/radix_linear_attention.py` — 线性注意力层接口
  - `python/sglang/srt/layers/attention/fla/chunk.py`、`fused_recurrent.py` — 线性注意力 Triton Kernel
  - `python/sglang/jit_kernel/cutedsl_gdn.py` — GDN CUDA Kernel
- **CUDA 算子**
  - `sgl-kernel/csrc/speculative/eagle_utils.cu` — 树构建 + Verify Kernel
  - `sgl-kernel/csrc/speculative/speculative_sampling.cu` — Sampling Kernel

# 量化 Demo

下载地址：[demo-quant.tar.gz](demos/demo-quant.tar.gz)

## 适用范围与定位
本示例提供了一套标准的 W4A16 量化接入参考链路，采用 RTN 策略验证模型在平台环境下的可加载性与基础推理能力。
**请注意，此代码库仅作为流程跑通的参考范例，不代表最终的性能优化方案或精度基准。**

## 接入规范
- **权重格式**：预处理产物需严格遵循 GPTQ 结构标准（包含 qweight, scales, qzeros, g_idx 等），并确保模型目录中包含独立的 `quantize_config.json` 配置文件，以保证 SGLang 正确识别量化路径。
- **精度对齐**：为确保算子兼容性，除被量化层外的其余权重需统一转换为 float16。
- **环境适配**：运行脚本已预置 `--quantization gptq` 与 `--dtype float16` 参数，并针对 MiniCPM 的稀疏注意力机制进行了 float16 分支适配。建议保持 `--disable-cuda-graph` 开启以确保首次接入的稳定性。

## 运行提示
受限于本示例较为基础的实现逻辑，模型在推理过程中容易产生超长生成序列，进而引发超时现象。
因此，本代码仅供流程跑通参考，不建议直接作为最终成绩提交。
