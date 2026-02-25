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

```plain&#x20;text
pip install huggingface_hub
```

下载完整模型库到指定路径文件夹./models

```plain&#x20;text
huggingface-cli download OpenBMB/MiniCPM-SALA --local-dir ./models
```

### 方式二：通过 ModelScope 下载
在下载前，请先通过如下命令安装 ModelScope。

```plain&#x20;text
pip install modelscope
```

下载完整模型库到指定路径文件夹./models

```plain&#x20;text
modelscope download --model OpenBMB/MiniCPM-SALA --local_dir ./models
```



## 容器内挂载地址

模型：/models/MiniCPM-SALA



## 容器启动脚本参考

### docker run 常用参数（Docker 通用选项）

### 环境变量（public serving 启动用）



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

* SGLANG\_SERVER\_ARGS 里请用连字符参数名：例如 --dense-as-sparse，不要写 --dense\_as\_sparse（镜像不做下划线自动转换）。



# 模型正确性评测

为了验证选手们对推理代码的优化不会影响模型在正确性上的表现，我们通过测试模型在特定数据集上的得分来进行评估。这里我们公开评测正确性所用的数据集`perf_public_set.jsonl`以及用于评测正确性的脚本`eval_model.py`，选手们也可以通过该数据集进行自查。

## perf\_public\_set.jsonl

下载地址：<https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_dataset/perf_public_set.jsonl>

本数据集包含不同长度的选择题或者信息提取题目，能够综合测试模型的整体性能表现。在对选手提交的代码进行评估的过程中，我们会验证模型在本数据集上的得分来判断模型能力在修改过程中是否会有所下降。该文件包含以下字段：

* **`task`**：任务类型

* **`question`**：输入 prompt 文本

* **`gold`**：参考答案/关键词列表等（不同任务类型含义不同）

示例：

```json
{"question":"...题目文本...", "task":"mcq", "gold":"B"}
```

> 为避免可能存在的刷分行为，我们会在内部准备一个私有集`perf_private_set.jsonl`。两个数据集的长度分布和任务一致，在原始模型推理结果中分数相近，主要用于检查模型是否会在两个数据集上存在较大的差距，保证比赛的公平性。

## eval\_model.py

下载地址：https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_model.py

`eval_model.py` 会通过调用已启动的 SGLang 推理服务，根据不同评测任务类型，给出模型在正确性上的评测分数。首先需要启动 SGLang 服务，并传入模型所使用的api\_base：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32 
```

参数说明（常用）：

* **`--api_base`**：SGLang 服务地址

* **`--model_path`**：模型路径

* **`--data_path`**：数据集路径

* **`--concurrency`**：（optional）并发请求数

* **`--num_samples`**：（optional）最多评测样本数（调试时可以进行少样本测试）

* **`--verbose`**：（optional）打印每条样本更详细的信息



# 模型速度评测

## bench\_serving.sh

下载方式：https://github.com/OpenBMB/SOAR-Toolkit/blob/main/bench_serving.sh

本脚本使用 sglang 官方 bench\_serving 工具，在 3 档并发度下分别跑完所有评测请求，记录 Benchmark Duration。在对应档位传入数据集路径可以完成对应档位的测试，未输入数据集路径的可跳过该档位的测试，相关传参及说明对应如下：

| **参数**            | **必填** | **说明**                         | **示例**                            |
| ----------------- | ------ | ------------------------------ | --------------------------------- |
| API\_BASE         | 是      | 模型 服务 地址                       | <http://127.0.0.1:30000>          |
| SPEED\_DATA\_S1   | 否      | S1 档位数据集（并发=1），可传入JSONL 路径     | /path/to/speech.jsonl（未设定时跳过该项测试） |
| SPEED\_DATA\_S8   | 否      | S8 档位数据集（并发=8），可传入JSONL 路径     | /path/to/speech.jsonl（未设定时跳过该项测试） |
| SPEED\_DATA\_SMAX | 否      | Smax 档位数据集（不设并发上限），可传入JSONL 路径 | /path/to/speech.jsonl（未设定时跳过该项测试） |

为了保证比赛结果的有效性和公平，这里暂不提供比赛中用于速度测试的数据集，题目长度分布可参考赛题。我们测试的方式是通过固定的模型输入和输出来对模型的速度进行测试，选手们可以通过以下字段构造 .jsonl 文件传入进行自测：

```python
{"question": "问题内容...", "model_response": "模型回答内容..."}
```

使用实例：

```bash
export SPEED_DATA_S1=/path/to/speech.jsonl
export SPEED_DATA_S8=/path/to/speech.jsonl
export SPEED_DATA_SMAX=/path/to/speech.jsonl

bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

# 模型量化/投机采样

本次比赛鼓励选手对模型进行量化或者投机采样，我们会在后续提供入口以及脚本 Demo。

# 构建注意事项

sgl-kernel 上游仓库的默认构建流程会拉取并打包多个第三方组件（DeepGEMM、Flash Attention、Triton Kernels 等）。**本赛道（MiniCPM-SALA）的推理流程仅依赖 `sgl_kernel/` 目录下的算子实现**，上述第三方组件在运行时不会被调用。

评测环境的校验脚本**仅放行 `sgl_kernel/` 与 `sgl_kernel.libs/` 前缀的文件**，包含其他顶层目录（如 `deep_gemm/`、`flash_attn_origin/`、`triton_kernels/`、`include/`、`lib/` 等）的 wheel **将无法通过校验，直接判定提交失败**。

因此，选手在构建 wheel 前**必须**对 `CMakeLists.txt` 做如下调整：

1. 移除或注释 `deep_gemm`、`flash_attn_origin`、`triton_kernels` 相关的 `FetchContent_Declare` / `FetchContent_Populate` 指令

2. 移除或注释对应的 `add_library`、`target_link_libraries`、`install(TARGETS ...)` 及 `install(DIRECTORY ...)` 指令

最终 wheel 的文件结构应仅包含：

* sgl\_kernel/ # 核心算子编译产物（必需）

* sgl\_kernel.libs/ # 动态库依赖（如有，由构建工具自动生成）

* sgl\_kernel-*.dist-info/ # 包元信息（自动生成）*

