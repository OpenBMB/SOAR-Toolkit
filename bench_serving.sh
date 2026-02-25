#!/bin/bash
# ============================================================
# benchmark_duration 评测脚本（基于 sglang.bench_serving）
#
# 使用 sglang 官方 bench_serving 工具，在 3 档并发度下
# 分别跑完所有评测请求，记录 Benchmark Duration。
#
# 用法（容器内由 entrypoint.sh 调用）:
#   bash /app/bench_serving.sh <API_BASE>
#
# 环境变量（可选）:
#   SPEED_DATA_S1   - S1(并发度1) 数据集路径（不设则跳过该项测试）
#   SPEED_DATA_S8   - S8(并发度8) 数据集路径（不设则跳过该项测试）
#   SPEED_DATA_SMAX - Smax(不设并发上限) 数据集路径（不设则跳过该项测试）
#
# 参数:
#   API_BASE     - SGLang API 地址，如 http://127.0.0.1:30000
# ============================================================
set -e

API_BASE="${1:?用法: bash bench_serving.sh <API_BASE>}"

# 解析 host 和 port
HOST=$(echo "${API_BASE}" | sed -E 's|https?://||' | cut -d: -f1)
PORT=$(echo "${API_BASE}" | sed -E 's|https?://||' | cut -d: -f2)

echo "[bench_serving] API: ${API_BASE} (host=${HOST}, port=${PORT})"
DATA_S1="${SPEED_DATA_S1:-}"
DATA_S8="${SPEED_DATA_S8:-}"
DATA_SMAX="${SPEED_DATA_SMAX:-}"

echo "[bench_serving] 数据集:"
[ -n "${DATA_S1}" ] && echo "  S1: ${DATA_S1}" || echo "  S1: (未指定，跳过)"
[ -n "${DATA_S8}" ] && echo "  S8: ${DATA_S8}" || echo "  S8: (未指定，跳过)"
[ -n "${DATA_SMAX}" ] && echo "  Smax: ${DATA_SMAX}" || echo "  Smax: (未指定，跳过)"

if [ -z "${DATA_S1}" ] && [ -z "${DATA_S8}" ] && [ -z "${DATA_SMAX}" ]; then
    echo "[bench_serving] 未指定任何速度评测数据集，跳过 benchmark_duration"
    echo '{"S1":0,"S8":0,"Smax":0}'
    exit 0
fi

# ============================================================
# Step 1: 转换数据集格式（逐档位）
# speed_*.jsonl -> custom 格式（conversations）
# ============================================================
convert_dataset() {
    local input_file="$1"
    local output_file="$2"

    python3 - "$input_file" "$output_file" <<'PY'
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

def get_prompt(item):
    # 兼容不同字段命名（历史数据/新数据）
    return item.get("question") or item.get("input") or item.get("prompt")

n = 0
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line_num, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        prompt = get_prompt(item)
        if prompt is None:
            raise KeyError(f"Missing prompt field at line {line_num}. Need one of: input/question/prompt")

        converted = {
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": item.get("model_response", "placeholder")},
            ]
        }
        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
        n += 1

print(n)
PY
}

# ============================================================
# Step 2: 在 3 档并发度下分别运行 bench_serving
# ============================================================
RESULT_JSON='{"S1":0,"S8":0,"Smax":0}'

BENCH_CONFIGS=""
[ -n "${DATA_S1}" ] && BENCH_CONFIGS="${BENCH_CONFIGS} S1:1"
[ -n "${DATA_S8}" ] && BENCH_CONFIGS="${BENCH_CONFIGS} S8:8"
[ -n "${DATA_SMAX}" ] && BENCH_CONFIGS="${BENCH_CONFIGS} Smax:"

for CONFIG in ${BENCH_CONFIGS}; do
    LABEL=$(echo "${CONFIG}" | cut -d: -f1)
    CONC=$(echo "${CONFIG}" | cut -d: -f2)

    # 选择该档位的数据集
    if [ "${LABEL}" = "S1" ]; then
        DATASET_PATH="${DATA_S1}"
    elif [ "${LABEL}" = "S8" ]; then
        DATASET_PATH="${DATA_S8}"
    elif [ "${LABEL}" = "Smax" ]; then
        DATASET_PATH="${DATA_SMAX}"
    fi

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "  [${LABEL}] 数据集文件不存在: ${DATASET_PATH}，跳过"
        continue
    fi

    # 转换数据集（每档位独立转换，避免相互影响）
    CONVERTED_DATA="/tmp/bench_eval_data_${LABEL}.jsonl"
    echo "[bench_serving] [${LABEL}] 数据集: ${DATASET_PATH}"
    convert_dataset "${DATASET_PATH}" "${CONVERTED_DATA}" >/dev/null
    NUM_PROMPTS=$(wc -l < "${CONVERTED_DATA}" 2>/dev/null || echo "0")
    echo "[bench_serving] [${LABEL}] 转换完成: ${NUM_PROMPTS} 条 -> ${CONVERTED_DATA}"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    if [ -n "${CONC}" ]; then
        echo "  [${LABEL}] 开始测试 - 并发度: ${CONC}, 共 ${NUM_PROMPTS} 条请求"
    else
        echo "  [${LABEL}] 开始测试 - 无并发上限, 共 ${NUM_PROMPTS} 条请求"
    fi
    echo "────────────────────────────────────────────────────────────"

    if [ "${NUM_PROMPTS}" = "0" ]; then
        echo "  [${LABEL}] 无有效请求（NUM_PROMPTS=0），跳过该档位"
        DURATION="0"
        RESULT_JSON=$(python3 -c "
import json
result = json.loads('${RESULT_JSON}')
result['${LABEL}'] = float('${DURATION}')
print(json.dumps(result))
")
        continue
    fi

    # 构建 bench_serving 命令
    BENCH_CMD="python3 -m sglang.bench_serving \
        --backend sglang \
        --host ${HOST} \
        --port ${PORT} \
        --dataset-name custom \
        --dataset-path ${CONVERTED_DATA} \
        --num-prompts ${NUM_PROMPTS} \
        --flush-cache"

    # 有并发限制时加 --max-concurrency，否则不加（全部请求同时发送）
    if [ -n "${CONC}" ]; then
        BENCH_CMD="${BENCH_CMD} --max-concurrency ${CONC}"
    fi

    BENCH_OUTPUT=$(eval ${BENCH_CMD} 2>&1) || true

    echo "${BENCH_OUTPUT}"

    # 提取 Benchmark duration (s): 行的值
    DURATION=$(echo "${BENCH_OUTPUT}" | grep -oP 'Benchmark duration \(s\):\s+\K[0-9.]+' || echo "0")

    echo "  [${LABEL}] Benchmark duration: ${DURATION}s"

    # 追加到结果 JSON
    RESULT_JSON=$(python3 -c "
import json
result = json.loads('${RESULT_JSON}')
result['${LABEL}'] = float('${DURATION}')
print(json.dumps(result))
")
done

# ============================================================
# Step 3: 输出汇总
# ============================================================
echo ""
echo "============================================================"
echo "  Benchmark Duration 汇总"
echo "============================================================"

python3 -c "
import json
result = json.loads('${RESULT_JSON}')
for key in ['S1', 'S8', 'Smax']:
    print(f'  {key:>4s}: {result.get(key, 0):>10.2f}s')
"

echo "============================================================"

# 最后一行输出 JSON（供 entrypoint.sh 捕获）
echo "${RESULT_JSON}"
