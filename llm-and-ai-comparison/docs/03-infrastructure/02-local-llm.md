# ローカル LLM — Ollama・llama.cpp・量子化

> ローカル LLM は大規模言語モデルを自社サーバーやローカルマシン上で実行する手法であり、データプライバシー、レイテンシ、コスト、オフライン動作の要件を満たすための重要な選択肢である。

## この章で学ぶこと

1. **ローカル実行の仕組みと量子化** — GGUF、GPTQ、AWQ による効率化とメモリ・品質のトレードオフ
2. **主要ツールの使い方** — Ollama、llama.cpp、vLLM、TGI の実践的な導入手順
3. **GPU/CPU 選定とパフォーマンス最適化** — ハードウェア要件、推論速度の向上手法

---

## 1. ローカル LLM の全体像

```
┌──────────────────────────────────────────────────────────┐
│          ローカル LLM 実行のエコシステム                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  モデル取得          量子化/最適化       実行環境          │
│  ┌──────────┐       ┌──────────┐      ┌─────────────┐  │
│  │Hugging   │       │ GGUF     │      │ Ollama      │  │
│  │Face Hub  │──────▶│ GPTQ     │─────▶│ llama.cpp   │  │
│  │          │       │ AWQ      │      │ vLLM        │  │
│  └──────────┘       │ GGML     │      │ TGI         │  │
│                     └──────────┘      └──────┬──────┘  │
│                                              │          │
│                                              ▼          │
│  ハードウェア                                            │
│  ┌──────────────────────────────────────┐              │
│  │ GPU: NVIDIA (CUDA) / Apple Silicon   │              │
│  │ CPU: x86_64 / ARM (推論可能)         │              │
│  │ RAM: 8GB〜 (モデルサイズ依存)        │              │
│  └──────────────────────────────────────┘              │
│                                                          │
│  API 提供                                                │
│  ┌──────────────────────────────────────┐              │
│  │ OpenAI 互換 API (localhost:11434等)   │              │
│  │ → 既存コードをそのまま流用可能        │              │
│  └──────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 量子化の仕組み

### 2.1 量子化とは

```
┌──────────────────────────────────────────────────────────┐
│           量子化によるモデルサイズ削減                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  FP32 (32bit浮動小数点)                                  │
│  ████████████████████████████████  → 1パラメータ = 4B    │
│  7B モデル: 28GB                                         │
│                                                          │
│  FP16 / BF16 (16bit)                                    │
│  ████████████████                  → 1パラメータ = 2B    │
│  7B モデル: 14GB                                         │
│                                                          │
│  INT8 (8bit整数)                                         │
│  ████████                          → 1パラメータ = 1B    │
│  7B モデル: 7GB                                          │
│                                                          │
│  INT4 (4bit整数)                                         │
│  ████                              → 1パラメータ = 0.5B  │
│  7B モデル: 3.5GB                                        │
│                                                          │
│  品質への影響:                                            │
│  FP16 → INT8: ほぼ劣化なし (-0.1〜0.5%)                 │
│  FP16 → INT4: 軽微な劣化 (-1〜3%)                       │
│  FP16 → INT2: 顕著な劣化 (-5〜15%)                      │
└──────────────────────────────────────────────────────────┘
```

### 2.2 量子化形式の比較

| 形式 | ビット幅 | 実行環境 | 品質 | 速度 | 主な用途 |
|------|---------|---------|------|------|---------|
| GGUF (Q4_K_M) | 4-5 bit | CPU + GPU | 高 | 中 | Ollama, llama.cpp |
| GGUF (Q8_0) | 8 bit | CPU + GPU | 最高 | 遅 | 品質重視 |
| GGUF (Q2_K) | 2-3 bit | CPU + GPU | 低 | 最速 | メモリ極小環境 |
| GPTQ | 4 bit | GPU 必須 | 高 | 高速 | vLLM, TGI |
| AWQ | 4 bit | GPU 必須 | 最高 | 高速 | vLLM (推奨) |
| EETQ | 8 bit | GPU 必須 | 最高 | 高速 | TGI |
| bitsandbytes | 4/8 bit | GPU 必須 | 高 | 中 | Transformers 統合 |

---

## 3. Ollama

### 3.1 インストールと基本操作

```bash
# macOS / Linux: インストール
curl -fsSL https://ollama.com/install.sh | sh

# モデルのダウンロードと実行
ollama pull llama3.1:8b          # 8Bモデル (4.7GB)
ollama pull qwen2.5:7b           # Qwen 7B
ollama pull deepseek-r1:8b       # DeepSeek R1 蒸留版

# インタラクティブ実行
ollama run llama3.1:8b

# 利用可能モデル一覧
ollama list
```

### 3.2 Ollama API (OpenAI 互換)

```python
from openai import OpenAI

# Ollama は OpenAI 互換 API を提供
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 任意の値でOK
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "日本語で回答してください。"},
        {"role": "user", "content": "Pythonのリスト内包表記を説明してください。"},
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)

# ストリーミング
stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "RAGとは？"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3.3 Modelfile でカスタマイズ

```dockerfile
# Modelfile: カスタムモデル定義
FROM llama3.1:8b

# システムプロンプト設定
SYSTEM """
あなたはPythonプログラミングの専門家です。
コードは常にPEP 8準拠で、型ヒント付きで記述してください。
"""

# パラメータ調整
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER stop "<|eot_id|>"
```

```bash
# カスタムモデルのビルドと実行
ollama create python-expert -f Modelfile
ollama run python-expert
```

---

## 4. llama.cpp

### 4.1 ビルドと実行

```bash
# ビルド (macOS - Metal対応)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# NVIDIA GPU 対応ビルド
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# サーバーモード (OpenAI互換API)
./build/bin/llama-server \
    -m models/llama-3.1-8b-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 8192 \       # コンテキスト長
    -ngl 99 \       # GPUにオフロードするレイヤー数
    --threads 8     # CPUスレッド数
```

### 4.2 Python バインディング

```python
from llama_cpp import Llama

# モデル読み込み
llm = Llama(
    model_path="models/llama-3.1-8b-q4_k_m.gguf",
    n_ctx=8192,        # コンテキスト長
    n_gpu_layers=-1,   # 全レイヤーをGPUに (-1 = 全部)
    n_threads=8,       # CPUスレッド数
    verbose=False,
)

# 推論
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "日本語で回答してください。"},
        {"role": "user", "content": "量子コンピュータの原理を説明してください。"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(output["choices"][0]["message"]["content"])
```

---

## 5. vLLM (高スループット推論)

### 5.1 vLLM サーバー

```bash
# インストール
pip install vllm

# サーバー起動 (OpenAI互換)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \    # GPU数
    --gpu-memory-utilization 0.9
```

```python
# vLLM サーバーへのアクセス (OpenAI互換)
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 5.2 バッチ推論 (オフライン)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 大量プロンプトを一括処理 (バッチ推論で高効率)
prompts = [f"質問{i}に回答してください" for i in range(1000)]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text[:100])
```

---

## 6. ハードウェア要件

```
┌──────────────────────────────────────────────────────────┐
│         モデルサイズ別のハードウェア要件                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  モデル        量子化    VRAM/RAM     推奨GPU/CPU        │
│  ─────        ─────    ─────────    ─────────          │
│  3B (Phi-3)   Q4       ~2GB         MacBook Air M1     │
│  7-8B         Q4       ~4GB         RTX 3060 12GB      │
│  7-8B         Q8       ~8GB         RTX 4070 12GB      │
│  7-8B         FP16     ~16GB        RTX 4090 24GB      │
│  13-14B       Q4       ~8GB         RTX 4070 Ti 16GB   │
│  34B          Q4       ~20GB        RTX 4090 24GB      │
│  70B          Q4       ~40GB        A100 80GB / 2xRTX  │
│  70B          FP16     ~140GB       2xA100 80GB        │
│  405B         Q4       ~240GB       8xA100 80GB        │
│                                                          │
│  Apple Silicon:                                          │
│  M1 (8GB)   → 7B Q4 まで快適                            │
│  M2 Pro (16GB) → 13B Q4 まで快適                        │
│  M3 Max (48GB) → 34B Q4 / 70B Q3 可能                  │
│  M3 Ultra (192GB) → 70B FP16 可能                       │
└──────────────────────────────────────────────────────────┘
```

---

## 7. 比較表

### 7.1 推論エンジン比較

| 特徴 | Ollama | llama.cpp | vLLM | TGI |
|------|--------|-----------|------|-----|
| 導入容易性 | 最高 | 中 | 中 | 中 |
| CPU 推論 | 対応 | 最適 | 非推奨 | 非推奨 |
| GPU 推論 | 対応 | 対応 | 最適 | 最適 |
| Apple Silicon | 最適 | 最適 | 限定的 | 非対応 |
| 量子化形式 | GGUF | GGUF | GPTQ/AWQ/FP16 | GPTQ/AWQ/EETQ |
| バッチ推論 | 限定的 | 限定的 | 最適 | 最適 |
| OpenAI互換 | 対応 | 対応 | 対応 | 対応 |
| マルチGPU | N/A | 限定的 | 最適 | 最適 |
| 用途 | 個人開発 | カスタマイズ | プロダクション | プロダクション |

### 7.2 量子化品質比較 (Llama 3.1 8B 基準)

| 量子化 | サイズ | MMLU | 推論速度 (tok/s) | 推奨度 |
|--------|-------|------|------------------|--------|
| FP16 | 16GB | 68.4 | 40 (A100) | 品質最優先 |
| Q8_0 | 8.5GB | 68.2 | 55 (A100) | バランス |
| Q5_K_M | 5.7GB | 67.8 | 70 (A100) | 高品質+小型 |
| Q4_K_M | 4.9GB | 67.1 | 80 (A100) | 推奨 |
| Q3_K_M | 3.9GB | 65.5 | 90 (A100) | メモリ重視 |
| Q2_K | 3.2GB | 60.2 | 100 (A100) | 最小限 |

---

## 8. Docker でのデプロイ

```python
# docker-compose.yml
"""
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - huggingface-cache:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --dtype bfloat16
      --max-model-len 8192
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama-data:
  huggingface-cache:
"""
```

---

## 9. アンチパターン

### アンチパターン 1: GPU メモリ不足での強制実行

```bash
# NG: 24GB GPUで70Bモデル (FP16) を実行しようとする
# → OOM (Out of Memory) エラー、またはスワッピングで極端に低速

# OK: 適切な量子化レベルを選択
ollama pull llama3.1:70b-q4_0  # Q4量子化: ~40GB必要
# さらに: GPU-CPU分割 (一部レイヤーをCPUで実行)
llama-server -m model.gguf -ngl 30  # 30レイヤーだけGPU
```

### アンチパターン 2: ローカル LLM で API モデルと同等の品質を期待

```
# NG: "7B Q4 モデルで GPT-4o と同じ品質が出るはず"
# → 7B Q4 は GPT-4o の 1/100 以下のパラメータ規模

# OK: 適切な期待値設定
# - 7B Q4: 簡単な質疑応答、分類、要約には十分
# - 70B Q4: GPT-3.5 Turbo レベルの品質
# - 用途を絞ってファインチューニングすれば小型でも高精度
```

---

## 10. FAQ

### Q1: Apple Silicon (M1/M2/M3) でローカル LLM は実用的か?

十分に実用的。Metal フレームワーク経由で GPU アクセラレーションが効き、
M2 Pro (16GB) で 7B Q4 が 30-40 tok/s、M3 Max (48GB) で 34B Q4 が 15-20 tok/s 程度。
Ollama が Apple Silicon を最もよくサポートしている。

### Q2: ローカル LLM と API モデルの損益分岐点は?

RTX 4090 (¥25万) で 7B モデルを運用する場合、GPT-4o mini API のコストと比較して、
月間約 50 万リクエスト以上でローカルが有利になる (電気代含む)。
ただし、運用工数 (モデル更新、障害対応) を考慮すると、100 万リクエスト以上が現実的な損益分岐点。

### Q3: ファインチューニングしたモデルをローカルで実行するには?

LoRA / QLoRA でファインチューニング → LoRA アダプタをマージ → GGUF に変換 → Ollama で実行。
`llama.cpp` の `convert` スクリプトで HuggingFace 形式 → GGUF 変換が可能。
Ollama では `FROM` に HuggingFace モデルを指定する Modelfile で直接実行もできる。

---

## まとめ

| 項目 | 推奨 |
|------|------|
| 個人開発・試用 | Ollama (最も簡単) |
| CPU推論 | llama.cpp (GGUF Q4_K_M) |
| GPU高スループット | vLLM (AWQ/GPTQ) |
| Apple Silicon | Ollama + GGUF |
| 推奨量子化 | Q4_K_M (品質・サイズのバランス最良) |
| 推奨モデル (日本語) | Qwen 2.5 7B / 14B |
| 最低ハードウェア | 8GB RAM + 4コアCPU (7B Q4) |

---

## 次に読むべきガイド

- [03-evaluation.md](./03-evaluation.md) — ローカルモデルの品質評価
- [01-vector-databases.md](./01-vector-databases.md) — ローカル LLM + ベクトル DB で完全ローカル RAG
- [../01-models/03-open-source.md](../01-models/03-open-source.md) — OSS モデルの選定

---

## 参考文献

1. Ollama, "Documentation," https://ollama.com/
2. Gerganov, "llama.cpp," https://github.com/ggerganov/llama.cpp
3. vLLM, "Documentation," https://docs.vllm.ai/
4. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023
5. Lin et al., "AWQ: Activation-aware Weight Quantization," MLSys 2024
