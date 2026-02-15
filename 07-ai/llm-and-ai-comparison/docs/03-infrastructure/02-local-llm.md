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

### 1.1 ローカル LLM を選ぶべきケース

ローカル LLM は万能ではなく、特定の条件下で大きな優位性を持つ。以下のディシジョンマトリクスで判断する。

```
┌──────────────────────────────────────────────────────────┐
│      ローカル LLM vs クラウド API 判断マトリクス            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  要件                      ローカル  API   判断基準      │
│  ──────                    ──────   ───   ──────────    │
│  データ機密性が最優先       ★★★     ★    医療・金融等  │
│  オフライン環境で動作       ★★★     ☆    軍事・工場等  │
│  低レイテンシ (<50ms)      ★★★     ★    ゲーム・RT系  │
│  月間100万req以上          ★★★     ★★   コスト逆転点  │
│  最新モデルを常に使いたい   ☆       ★★★  API有利     │
│  インフラ管理を避けたい     ☆       ★★★  チーム規模   │
│  GPT-4o級の品質が必須      ★       ★★★  大型モデル   │
│  カスタマイズが必要         ★★★     ★    FT・LoRA     │
│                                                          │
│  ★★★=非常に有利  ★★=有利  ★=まあまあ  ☆=不向き     │
└──────────────────────────────────────────────────────────┘
```

### 1.2 ローカル LLM のアーキテクチャパターン

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DeploymentPattern(Enum):
    """ローカル LLM のデプロイパターン"""
    SINGLE_GPU = "single_gpu"          # 単一 GPU (RTX 4090 等)
    MULTI_GPU = "multi_gpu"            # マルチ GPU (テンソル並列)
    CPU_ONLY = "cpu_only"              # CPU のみ (GGUF Q4)
    HYBRID = "hybrid"                  # GPU + CPU オフロード
    APPLE_SILICON = "apple_silicon"    # Apple Silicon (Metal)
    EDGE = "edge"                      # エッジデバイス (Raspberry Pi等)

@dataclass
class LocalLLMConfig:
    """ローカル LLM 構成定義"""
    model_name: str
    model_size_b: float                 # パラメータ数 (Billion)
    quantization: str                   # "Q4_K_M", "Q8_0", "GPTQ", "AWQ"
    deployment_pattern: DeploymentPattern
    max_context_length: int = 8192
    gpu_layers: int = -1                # -1 = 全レイヤー GPU
    threads: int = 8
    batch_size: int = 512

    @property
    def estimated_vram_gb(self) -> float:
        """推定 VRAM/RAM 使用量を計算"""
        quant_multiplier = {
            "FP16": 2.0,
            "Q8_0": 1.1,
            "Q5_K_M": 0.75,
            "Q4_K_M": 0.65,
            "Q3_K_M": 0.5,
            "Q2_K": 0.4,
            "GPTQ": 0.6,
            "AWQ": 0.6,
        }
        multiplier = quant_multiplier.get(self.quantization, 0.65)
        return self.model_size_b * multiplier

    @property
    def fits_in_gpu(self) -> bool:
        """指定された GPU に収まるか判定"""
        common_gpus = {
            "RTX_3060_12GB": 12,
            "RTX_4070_12GB": 12,
            "RTX_4090_24GB": 24,
            "A100_40GB": 40,
            "A100_80GB": 80,
        }
        return self.estimated_vram_gb < 24  # RTX 4090 基準

# 使用例: 各構成の VRAM 見積もり
configs = [
    LocalLLMConfig("Llama-3.1-8B", 8, "Q4_K_M", DeploymentPattern.SINGLE_GPU),
    LocalLLMConfig("Qwen-2.5-14B", 14, "Q4_K_M", DeploymentPattern.SINGLE_GPU),
    LocalLLMConfig("Llama-3.1-70B", 70, "Q4_K_M", DeploymentPattern.MULTI_GPU),
    LocalLLMConfig("Phi-3-mini", 3.8, "Q4_K_M", DeploymentPattern.EDGE),
]

for config in configs:
    print(f"{config.model_name:20s} {config.quantization:8s} "
          f"→ {config.estimated_vram_gb:.1f} GB "
          f"(GPU適合: {config.fits_in_gpu})")
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

### 2.3 GGUF 量子化の詳細

GGUF (GPT-Generated Unified Format) は llama.cpp が定義したモデルフォーマットで、CPU・GPU 双方での推論をサポートする。量子化バリエーションの名前の読み方を理解することが重要。

```
┌──────────────────────────────────────────────────────────┐
│          GGUF 量子化名の読み方                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Q4_K_M の解読:                                          │
│  │ │ │                                                  │
│  │ │ └── M = Medium (品質・サイズのバランス)              │
│  │ │     S = Small (より小型)                            │
│  │ │     L = Large (より高品質)                          │
│  │ │                                                    │
│  │ └──── K = K-Quant 方式 (改良型量子化)                 │
│  │       0 = 旧方式 (対称量子化)                         │
│  │                                                      │
│  └────── 4 = 基本ビット幅 (4bit)                         │
│                                                          │
│  推奨順位 (品質/サイズバランス):                          │
│  1. Q4_K_M  → 最もバランスが良い (推奨)                  │
│  2. Q5_K_M  → やや高品質 + やや大型                      │
│  3. Q3_K_M  → メモリ節約優先                             │
│  4. Q6_K    → 高品質指向                                 │
│  5. Q8_0    → 品質最優先 (FP16 に近い)                   │
│  6. Q2_K    → 最小サイズ (品質犠牲大)                    │
└──────────────────────────────────────────────────────────┘
```

### 2.4 GPTQ と AWQ の違い

```python
# GPTQ 量子化: キャリブレーションデータ依存の後量子化
# → 特定のデータセットを使って最適な量子化パラメータを決定
# → GPU 推論に最適化

# AWQ (Activation-aware Weight Quantization):
# → 「重要な重み」を特定し、それらを保護しながら量子化
# → GPTQ より高品質な場合が多い

# 比較表
"""
特性          GPTQ                    AWQ
──────       ──────                  ──────
量子化手法    レイヤーごとの最適化     活性化ベースの重要度分析
品質          高い                    やや高い（GPTQ以上）
速度          高速                    高速
キャリブ      必要 (128-256サンプル)  必要 (少量でOK)
互換性        vLLM, TGI, AutoGPTQ    vLLM, TGI, AutoAWQ
推奨場面      一般的なGPU推論         品質重視のGPU推論
"""
```

### 2.5 自分で量子化を実行する

```python
# AutoGPTQ を使った GPTQ 量子化
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
output_dir = "llama-3.1-8b-gptq-4bit"

# 量子化設定
quantize_config = BaseQuantizeConfig(
    bits=4,                 # 量子化ビット数
    group_size=128,         # グループサイズ (小さいほど高品質、大きいほど高速)
    desc_act=True,          # 活性化に基づくソート (品質向上)
    damp_percent=0.1,       # ダンピング (数値安定性)
)

# トークナイザとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
)

# キャリブレーションデータの準備
calibration_data = [
    tokenizer(text, return_tensors="pt")
    for text in [
        "The meaning of life is",
        "機械学習とは、データから",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        # ... 128-256 サンプル推奨
    ]
]

# 量子化の実行 (数十分〜数時間)
model.quantize(calibration_data)

# 保存
model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"量子化完了: {output_dir}")
```

```python
# AWQ 量子化
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
output_dir = "llama-3.1-8b-awq-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAWQForCausalLM.from_pretrained(model_name)

# AWQ 量子化設定
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",  # GEMM = GPU最適化
}

# 量子化実行
model.quantize(
    tokenizer,
    quant_config=quant_config,
)

# 保存
model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)
```

### 2.6 HuggingFace → GGUF 変換

```bash
# llama.cpp の convert スクリプトを使用
cd llama.cpp

# HuggingFace 形式 → GGUF (FP16)
python convert_hf_to_gguf.py \
    /path/to/huggingface-model \
    --outfile model-fp16.gguf \
    --outtype f16

# GGUF の量子化 (FP16 → Q4_K_M)
./build/bin/llama-quantize \
    model-fp16.gguf \
    model-q4_k_m.gguf \
    Q4_K_M

# 量子化結果の確認
./build/bin/llama-quantize --help
# 利用可能な量子化タイプ:
#   Q2_K, Q3_K_S, Q3_K_M, Q3_K_L,
#   Q4_0, Q4_1, Q4_K_S, Q4_K_M,
#   Q5_0, Q5_1, Q5_K_S, Q5_K_M,
#   Q6_K, Q8_0, F16, F32

# imatrix (重要度行列) を使った高品質量子化
./build/bin/llama-imatrix \
    -m model-fp16.gguf \
    -f calibration_data.txt \
    -o imatrix.dat

./build/bin/llama-quantize \
    --imatrix imatrix.dat \
    model-fp16.gguf \
    model-q4_k_m-imat.gguf \
    Q4_K_M
# → imatrix 使用で Q4 の品質が Q5 に近づく
```

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

### 3.4 Ollama の高度な管理

```bash
# モデルの詳細情報
ollama show llama3.1:8b

# モデルの削除 (ストレージ解放)
ollama rm llama3.1:8b

# GPU メモリ使用量の確認
ollama ps

# 実行中のモデルの一覧
ollama list --running

# 環境変数によるカスタマイズ
export OLLAMA_HOST=0.0.0.0:11434           # バインドアドレス
export OLLAMA_MODELS=/data/ollama/models   # モデル保存先
export OLLAMA_NUM_PARALLEL=4               # 同時リクエスト数
export OLLAMA_MAX_LOADED_MODELS=2          # 同時ロードモデル数
export OLLAMA_KEEP_ALIVE=5m                # モデルのメモリ保持時間
```

### 3.5 Ollama でカスタム GGUF を使う

```dockerfile
# Modelfile: カスタム GGUF モデル
FROM /path/to/my-custom-model-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|eot_id|>"
PARAMETER num_ctx 8192
```

```bash
# ビルドと実行
ollama create my-custom -f Modelfile
ollama run my-custom
```

### 3.6 Ollama Python ライブラリ (ネイティブ)

```python
import ollama

# Ollama ネイティブ API
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": "量子コンピュータの原理を簡潔に説明してください。"}
    ],
)
print(response["message"]["content"])

# ストリーミング
stream = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Rustの所有権を説明"}],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

# Embeddings
embeddings = ollama.embeddings(
    model="nomic-embed-text",
    prompt="日本語のテキストを埋め込む",
)
print(f"次元数: {len(embeddings['embedding'])}")

# モデル一覧
models = ollama.list()
for model in models["models"]:
    print(f"{model['name']:30s} {model['size'] / 1e9:.1f} GB")
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

### 4.3 llama.cpp サーバーの詳細設定

```bash
# 高度なサーバー設定
./build/bin/llama-server \
    -m models/llama-3.1-8b-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 16384 \              # コンテキスト長
    -ngl 99 \               # GPU レイヤー数
    --threads 8 \           # CPU スレッド数
    --threads-batch 8 \     # バッチ処理用 CPU スレッド
    --batch-size 2048 \     # バッチサイズ
    --ubatch-size 512 \     # マイクロバッチサイズ
    --cont-batching \       # 連続バッチ処理 (スループット向上)
    --flash-attn \          # Flash Attention (メモリ節約 + 高速化)
    --mlock \               # メモリロック (スワップ防止)
    --no-mmap \             # メモリマップ無効化 (大きいモデル用)
    --parallel 4 \          # 同時リクエスト処理数
    --metrics               # Prometheus メトリクス有効化
```

### 4.4 llama-cpp-python の高度な使い方

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llama3ChatHandler

# Function Calling 対応
llm = Llama(
    model_path="models/llama-3.1-8b-q4_k_m.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
    chat_format="llama-3",
    chat_handler=Llama3ChatHandler(),
)

# JSON Mode (Structured Output)
import json

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "JSONで回答してください。"},
        {"role": "user", "content": "日本の3大都市の人口を教えて"},
    ],
    response_format={"type": "json_object"},
    temperature=0.1,
)
result = json.loads(output["choices"][0]["message"]["content"])
print(json.dumps(result, ensure_ascii=False, indent=2))

# Grammar による出力制約 (GBNF 形式)
grammar_text = r'''
root   ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws number "}"
string ::= "\"" [a-zA-Z ]+ "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
'''

from llama_cpp import LlamaGrammar
grammar = LlamaGrammar.from_string(grammar_text)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "架空の人物のプロフィールを作成"}],
    grammar=grammar,
)
# → 必ず {"name": "...", "age": ...} 形式で出力
```

### 4.5 speculative decoding (投機的デコード)

```bash
# 大きなモデルの推論を小さなモデルで加速
# Draft model (小): 高速に候補トークンを生成
# Target model (大): 候補を検証

./build/bin/llama-speculative \
    -m models/llama-3.1-70b-q4_k_m.gguf \      # ターゲット (大)
    -md models/llama-3.1-8b-q4_k_m.gguf \       # ドラフト (小)
    --draft 8 \                                   # ドラフトトークン数
    -ngl 99 \
    -c 4096 \
    -p "Explain quantum computing in detail."

# 原理:
# 1. ドラフトモデルが 8 トークンを高速生成
# 2. ターゲットモデルが 8 トークンを一括検証
# 3. 一致するトークンはそのまま採用
# 4. 不一致の時点から再生成
# → 2-3 倍の高速化が期待できる
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

### 5.3 vLLM の高度な設定

```bash
# プロダクション向け vLLM 設定
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --tensor-parallel-size 2 \          # 2 GPU でテンソル並列
    --pipeline-parallel-size 1 \        # パイプライン並列
    --gpu-memory-utilization 0.90 \     # GPU メモリ使用率
    --max-num-seqs 256 \                # 最大同時シーケンス数
    --max-num-batched-tokens 8192 \     # 最大バッチトークン数
    --enable-prefix-caching \           # プレフィックスキャッシュ (KV Cache共有)
    --quantization awq \                # AWQ 量子化モデル使用時
    --enforce-eager \                   # CUDA Graph 無効 (デバッグ用)
    --disable-log-requests              # リクエストログ抑制 (本番用)
```

```python
# vLLM で LoRA アダプタを動的に切り替え
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ベースモデル + LoRA 対応で起動
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=4,  # 同時に4つの LoRA をロード可能
)

# LoRA アダプタの指定
lora_request = LoRARequest(
    lora_name="japanese-qa",
    lora_int_id=1,
    lora_local_path="/path/to/lora-adapter",
)

# LoRA 付きで推論
outputs = llm.generate(
    ["日本の首都は？"],
    SamplingParams(temperature=0.1, max_tokens=128),
    lora_request=lora_request,
)

# 別の LoRA に切り替え
lora_code = LoRARequest("code-gen", 2, "/path/to/code-lora")
outputs_code = llm.generate(
    ["def fibonacci(n):"],
    SamplingParams(temperature=0.2, max_tokens=256),
    lora_request=lora_code,
)
```

### 5.4 vLLM vs TGI 比較

```
┌──────────────────────────────────────────────────────────┐
│           vLLM vs TGI 詳細比較                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  特性           vLLM                  TGI                │
│  ────           ────                  ───                │
│  開発元         UC Berkeley           Hugging Face       │
│  言語           Python                Rust + Python      │
│  バッチ処理     PagedAttention        Continuous Batch   │
│  メモリ効率     PagedAttention        Token Streaming    │
│  量子化対応     GPTQ/AWQ/FP8         GPTQ/AWQ/EETQ     │
│  テンソル並列   対応 (最大8GPU)       対応               │
│  LoRA対応       対応 (動的切替)       対応               │
│  スループット   非常に高い            高い               │
│  ドキュメント   良好                  良好               │
│  Docker対応     公式イメージあり      公式イメージあり   │
│  Kubernetes     対応                  対応               │
│                                                          │
│  推奨:                                                   │
│  - 最大スループット → vLLM                                │
│  - HuggingFace エコシステム → TGI                        │
│  - Apple Silicon → どちらも非推奨 (Ollama推奨)           │
└──────────────────────────────────────────────────────────┘
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

### 6.1 GPU 選定ガイド

```python
# GPU 選定の意思決定ツール
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUSpec:
    name: str
    vram_gb: int
    bandwidth_gb_s: int   # メモリ帯域幅 (推論速度に直結)
    price_usd: int        # 概算価格
    power_w: int          # 消費電力
    fp16_tflops: float    # FP16 演算性能

gpu_catalog = {
    "RTX_3060":    GPUSpec("RTX 3060 12GB", 12, 360, 300, 170, 12.7),
    "RTX_4060_Ti": GPUSpec("RTX 4060 Ti 16GB", 16, 288, 450, 165, 22.1),
    "RTX_4070_Ti": GPUSpec("RTX 4070 Ti 16GB", 16, 504, 600, 285, 40.1),
    "RTX_4090":    GPUSpec("RTX 4090 24GB", 24, 1008, 1600, 450, 82.6),
    "A100_40GB":   GPUSpec("A100 40GB", 40, 1555, 10000, 250, 77.9),
    "A100_80GB":   GPUSpec("A100 80GB", 80, 2039, 15000, 300, 77.9),
    "H100_80GB":   GPUSpec("H100 80GB", 80, 3350, 30000, 700, 267.6),
    "L40S":        GPUSpec("L40S 48GB", 48, 864, 7000, 350, 91.6),
}

def recommend_gpu(model_size_b: float, quant: str = "Q4_K_M") -> list:
    """モデルサイズから推奨GPUを返す"""
    quant_factor = {
        "FP16": 2.0, "Q8_0": 1.1, "Q4_K_M": 0.65,
        "GPTQ": 0.6, "AWQ": 0.6,
    }
    required_vram = model_size_b * quant_factor.get(quant, 0.65) * 1.2  # 20%マージン

    suitable = []
    for key, gpu in gpu_catalog.items():
        if gpu.vram_gb >= required_vram:
            # tok/s 概算 (帯域幅 / (VRAM使用量))
            est_tok_s = gpu.bandwidth_gb_s / (required_vram * 2) * 10
            suitable.append({
                "gpu": gpu.name,
                "vram": gpu.vram_gb,
                "required": f"{required_vram:.1f} GB",
                "est_tok_s": f"~{est_tok_s:.0f} tok/s",
                "price": f"${gpu.price_usd:,}",
            })

    return sorted(suitable, key=lambda x: x["vram"])

# 例
print("=== 8B Q4_K_M ===")
for gpu in recommend_gpu(8, "Q4_K_M"):
    print(f"  {gpu['gpu']:25s} VRAM:{gpu['vram']}GB "
          f"必要:{gpu['required']} 速度:{gpu['est_tok_s']} "
          f"価格:{gpu['price']}")

print("\n=== 70B Q4_K_M ===")
for gpu in recommend_gpu(70, "Q4_K_M"):
    print(f"  {gpu['gpu']:25s} VRAM:{gpu['vram']}GB "
          f"必要:{gpu['required']} 速度:{gpu['est_tok_s']} "
          f"価格:{gpu['price']}")
```

### 6.2 Apple Silicon でのパフォーマンス最適化

```bash
# Apple Silicon 最適化の確認
# Metal が有効かどうか
ollama run llama3.1:8b --verbose 2>&1 | grep -i metal

# llama.cpp でのメモリ帯域テスト
./build/bin/llama-bench \
    -m models/llama-3.1-8b-q4_k_m.gguf \
    -p 512 \       # プロンプトトークン数
    -n 128 \       # 生成トークン数
    -ngl 99        # 全レイヤー GPU (Metal)

# 結果例 (M3 Max 48GB):
# prompt eval: 1234.56 tok/s
# generation:  38.91 tok/s
```

```python
# Apple Silicon での最適設定
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama-3.1-8b-q4_k_m.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,      # Metal で全レイヤー処理
    n_threads=1,           # Apple Silicon ではスレッド数を減らす方が速い場合あり
    n_batch=512,
    use_mlock=True,        # メモリロック
    verbose=False,
)

# パフォーマンス測定
import time

prompt = "日本語で100文字程度の物語を書いてください。"
start = time.time()
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
)
elapsed = time.time() - start
tokens = output["usage"]["completion_tokens"]
print(f"生成トークン: {tokens}, 時間: {elapsed:.2f}s, 速度: {tokens/elapsed:.1f} tok/s")
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

```yaml
# docker-compose.yml
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
```

### 8.1 Kubernetes でのデプロイ

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  labels:
    app: vllm-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          command:
            - "python"
            - "-m"
            - "vllm.entrypoints.openai.api_server"
            - "--model"
            - "Qwen/Qwen2.5-7B-Instruct"
            - "--dtype"
            - "bfloat16"
            - "--max-model-len"
            - "8192"
            - "--gpu-memory-utilization"
            - "0.90"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "32Gi"
            requests:
              nvidia.com/gpu: 1
              memory: "16Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-server
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "80"
```

### 8.2 プロダクション向けリバースプロキシ設定

```nginx
# nginx.conf — ローカル LLM のリバースプロキシ
upstream vllm_backend {
    least_conn;  # 最小接続数によるロードバランシング
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 443 ssl;
    server_name llm.example.com;

    ssl_certificate     /etc/ssl/certs/llm.crt;
    ssl_certificate_key /etc/ssl/private/llm.key;

    # レート制限
    limit_req_zone $binary_remote_addr zone=llm:10m rate=10r/s;

    location /v1/ {
        limit_req zone=llm burst=20 nodelay;

        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # ストリーミング対応
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;

        # タイムアウト設定 (LLM は応答に時間がかかる)
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        # リクエストサイズ制限
        client_max_body_size 10M;
    }

    # ヘルスチェック
    location /health {
        proxy_pass http://vllm_backend/health;
    }
}
```

---

## 9. パフォーマンス最適化

### 9.1 推論速度のボトルネック分析

```
┌──────────────────────────────────────────────────────────┐
│          LLM 推論のボトルネック分析                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  推論は 2 つのフェーズに分かれる:                         │
│                                                          │
│  1. Prefill (プロンプト処理)                              │
│     - 入力トークン全体を一括処理                         │
│     - GPU 演算がボトルネック (compute-bound)              │
│     - 高速化: バッチサイズ増加、Flash Attention           │
│                                                          │
│  2. Decode (トークン生成)                                │
│     - 1 トークンずつ逐次生成                             │
│     - メモリ帯域幅がボトルネック (memory-bound)           │
│     - 高速化: 量子化、KV Cache最適化                     │
│                                                          │
│  推論速度の計算式 (概算):                                │
│  Decode tok/s ≈ メモリ帯域幅(GB/s) / モデルサイズ(GB)   │
│                                                          │
│  例:                                                     │
│  RTX 4090 (1008 GB/s) + 8B Q4 (4.9GB)                  │
│  → 1008 / 4.9 ≈ 205 tok/s (理論上限)                   │
│  → 実測 80-120 tok/s (オーバーヘッド含む)                │
│                                                          │
│  A100 80GB (2039 GB/s) + 8B Q4 (4.9GB)                  │
│  → 2039 / 4.9 ≈ 416 tok/s (理論上限)                   │
│  → 実測 150-250 tok/s                                   │
└──────────────────────────────────────────────────────────┘
```

### 9.2 KV Cache の最適化

```python
# KV Cache はコンテキスト長に比例してメモリを消費
# 8B モデル、4096 コンテキストでの KV Cache サイズ:
# = layers * 2 (K+V) * heads * head_dim * context_len * dtype_size
# = 32 * 2 * 32 * 128 * 4096 * 2 (FP16)
# ≈ 2 GB

# vLLM の PagedAttention で効率化
# → KV Cache をページ単位で管理し、未使用部分を解放
# → メモリ効率が 2-4 倍向上

# llama.cpp での KV Cache 設定
"""
./build/bin/llama-server \
    -m model.gguf \
    -c 8192 \         # コンテキスト長 (KV Cache サイズに直結)
    --cache-type-k q8_0 \   # KV Cache の K を Q8 に量子化
    --cache-type-v q8_0 \   # KV Cache の V を Q8 に量子化
    # → KV Cache のメモリ使用量が約半分に
"""
```

### 9.3 パフォーマンス計測ツール

```python
import time
import statistics
from typing import Callable

def benchmark_model(
    generate_fn: Callable,
    prompts: list[str],
    warmup: int = 3,
    iterations: int = 10,
) -> dict:
    """ローカル LLM のパフォーマンスベンチマーク"""
    # ウォームアップ
    for _ in range(warmup):
        generate_fn(prompts[0])

    # 計測
    ttft_list = []  # Time to First Token
    tps_list = []   # Tokens per Second
    total_list = [] # Total Time

    for prompt in prompts[:iterations]:
        start = time.perf_counter()

        # 最初のトークンまでの時間を計測
        first_token_time = None
        total_tokens = 0

        for token in generate_fn(prompt, stream=True):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
                ttft_list.append(first_token_time)
            total_tokens += 1

        total_time = time.perf_counter() - start
        total_list.append(total_time)

        if total_tokens > 1:
            # TTFT 以降の生成速度
            gen_time = total_time - first_token_time
            tps = (total_tokens - 1) / gen_time if gen_time > 0 else 0
            tps_list.append(tps)

    return {
        "TTFT (ms)": {
            "mean": statistics.mean(ttft_list) * 1000,
            "p50": statistics.median(ttft_list) * 1000,
            "p95": sorted(ttft_list)[int(len(ttft_list) * 0.95)] * 1000,
        },
        "TPS (tok/s)": {
            "mean": statistics.mean(tps_list),
            "p50": statistics.median(tps_list),
            "min": min(tps_list),
        },
        "Total (s)": {
            "mean": statistics.mean(total_list),
            "p50": statistics.median(total_list),
        },
    }
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と解決策

```python
# === 問題 1: OOM (Out of Memory) ===
# エラー: "CUDA out of memory" / "Killed" (Linux OOM Killer)

# 解決策:
troubleshoot_oom = """
1. モデルサイズの確認
   $ ollama show llama3.1:8b --modelfile
   → VRAM 要件を確認

2. GPU メモリ使用状況の確認
   $ nvidia-smi
   → 他のプロセスが GPU を使用していないか

3. 量子化レベルを下げる
   Q8_0 (8GB) → Q4_K_M (4.9GB) → Q3_K_M (3.9GB)

4. GPU レイヤー数を減らす (CPU オフロード)
   llama-server -m model.gguf -ngl 20  # 一部だけGPU

5. コンテキスト長を短縮
   -c 8192 → -c 4096 → -c 2048

6. バッチサイズを削減
   --batch-size 2048 → --batch-size 512
"""

# === 問題 2: 推論が遅い ===
troubleshoot_slow = """
1. GPU が使われているか確認
   $ nvidia-smi  # GPU 使用率が 0% なら CPU で実行されている
   → -ngl パラメータを確認 (-1 で全レイヤー GPU)

2. Metal が有効か確認 (Apple Silicon)
   $ ollama run llama3.1 --verbose 2>&1 | grep metal

3. CPU スレッド数の最適化
   NVIDIA GPU: --threads $(nproc)  # 全コア
   Apple Silicon: --threads 1      # 1スレッドが最速の場合あり

4. メモリマッピングの確認
   --mlock  # メモリロックでスワップ防止
   --no-mmap  # メモリマップ無効 (大モデル)

5. Flash Attention の有効化
   --flash-attn  # llama.cpp サーバー
"""

# === 問題 3: 出力品質が悪い ===
troubleshoot_quality = """
1. 量子化レベルの確認
   Q2_K は大幅な品質劣化あり → Q4_K_M 以上推奨

2. チャットテンプレートの確認
   Ollama: Modelfile の TEMPLATE が正しいか
   llama.cpp: --chat-template パラメータ

3. システムプロンプトの設定
   日本語タスクには日本語システムプロンプトが必須

4. Temperature/Top-p の調整
   正確性重視: temperature=0.1, top_p=0.9
   創造性重視: temperature=0.8, top_p=0.95

5. モデル選択の見直し
   日本語 → Qwen 2.5 を優先
   英語 → Llama 3.1 が安定
"""
```

### 10.2 デバッグユーティリティ

```python
import subprocess
import json
from typing import Optional

class LocalLLMDebugger:
    """ローカル LLM のデバッグ・診断ツール"""

    @staticmethod
    def check_gpu_status() -> dict:
        """GPU の状態を確認"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append({
                        "name": parts[0],
                        "vram_total_mb": int(parts[1]),
                        "vram_used_mb": int(parts[2]),
                        "utilization_pct": int(parts[3]),
                    })
                return {"status": "ok", "gpus": gpus}
            return {"status": "error", "message": result.stderr}
        except FileNotFoundError:
            return {"status": "no_nvidia_gpu", "message": "nvidia-smi not found"}

    @staticmethod
    def check_ollama_status() -> dict:
        """Ollama の状態を確認"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                models = []
                for line in result.stdout.strip().split("\n")[1:]:  # ヘッダー除外
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
                return {"status": "ok", "models": models}
            return {"status": "error", "message": result.stderr}
        except Exception as e:
            return {"status": "not_running", "message": str(e)}

    @staticmethod
    def estimate_requirements(model_size_b: float, context_length: int = 8192) -> dict:
        """モデル実行に必要なリソースを見積もり"""
        quant_options = {}
        for quant, factor in [("FP16", 2.0), ("Q8_0", 1.1), ("Q5_K_M", 0.75),
                               ("Q4_K_M", 0.65), ("Q3_K_M", 0.5)]:
            model_mem = model_size_b * factor
            # KV Cache 概算 (layer * 2 * heads * head_dim * ctx * dtype)
            kv_cache_gb = (model_size_b / 7) * 32 * 2 * 32 * 128 * context_length * 2 / 1e9
            total = model_mem + kv_cache_gb
            quant_options[quant] = {
                "model_gb": round(model_mem, 1),
                "kv_cache_gb": round(kv_cache_gb, 1),
                "total_gb": round(total, 1),
            }
        return quant_options

# 使用例
debugger = LocalLLMDebugger()

print("=== GPU 状態 ===")
print(json.dumps(debugger.check_gpu_status(), indent=2))

print("\n=== Ollama 状態 ===")
print(json.dumps(debugger.check_ollama_status(), indent=2))

print("\n=== 8B モデルの要件 ===")
reqs = debugger.estimate_requirements(8, 8192)
for quant, info in reqs.items():
    print(f"  {quant:8s}: モデル {info['model_gb']:.1f}GB + "
          f"KV Cache {info['kv_cache_gb']:.1f}GB = "
          f"合計 {info['total_gb']:.1f}GB")
```

---

## 11. アンチパターン

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

### アンチパターン 3: セキュリティ無視のデプロイ

```python
# NG: ローカル LLM サーバーを認証なしで外部公開
"""
ollama serve  # デフォルトで 0.0.0.0:11434
# → インターネットから誰でもアクセス可能!
"""

# OK: 認証とネットワーク制限を設定
"""
# 1. ローカルホストのみにバインド
export OLLAMA_HOST=127.0.0.1:11434

# 2. リバースプロキシで認証を追加
# nginx で API Key 認証

# 3. ファイアウォールで制限
ufw allow from 192.168.1.0/24 to any port 11434
ufw deny 11434
"""
```

---

## 12. FAQ

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

### Q4: 複数モデルを同時に実行できるか?

Ollama では `OLLAMA_MAX_LOADED_MODELS` 環境変数で同時ロードモデル数を設定可能。
VRAM に余裕があれば 2-3 モデルの同時ロードが可能。
vLLM では複数の `--model` は非対応だが、複数インスタンスを異なるポートで起動する方法がある。
用途に応じてルーティングする構成が実用的。

### Q5: GGUF と GPTQ/AWQ はどう使い分ける?

CPU 推論 or Apple Silicon → GGUF 一択。NVIDIA GPU のみの環境で最大スループットを求めるなら GPTQ/AWQ。
vLLM を使う場合は AWQ が品質・速度のバランスが最良。
Ollama / llama.cpp を使う場合は GGUF (Q4_K_M) が最も安定。

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
6. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
7. NVIDIA, "TensorRT-LLM," https://github.com/NVIDIA/TensorRT-LLM
