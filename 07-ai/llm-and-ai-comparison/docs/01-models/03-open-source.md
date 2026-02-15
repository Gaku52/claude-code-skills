# オープンソース LLM — Llama・Mistral・Qwen と OSS エコシステム

> オープンソース LLM はモデル重みが公開され、自由にダウンロード・カスタマイズ・デプロイできる大規模言語モデル群であり、Meta Llama、Mistral AI、Alibaba Qwen を三大勢力としてプロプライエタリモデルに迫る性能を実現している。

## この章で学ぶこと

1. **主要オープンソース LLM の特徴と差異** — Llama 3、Mistral/Mixtral、Qwen 2.5 の設計思想・性能・ライセンス
2. **OSS LLM の選定基準** — パラメータサイズ、言語対応、ライセンス、ファインチューニング容易性
3. **実運用におけるデプロイと最適化** — 量子化、推論サーバー、コスト最適化の実践手法

---

## 1. 主要オープンソース LLM 概観

```
┌──────────────────────────────────────────────────────────┐
│            オープンソース LLM エコシステム (2024-2025)      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Meta (Llama)          Mistral AI          Alibaba       │
│  ┌──────────┐         ┌──────────┐       ┌──────────┐   │
│  │ Llama 3  │         │ Mistral  │       │ Qwen 2.5 │   │
│  │ 8B / 70B │         │ 7B       │       │ 0.5B-72B │   │
│  │ / 405B   │         │ Mixtral  │       │ Coder    │   │
│  └──────────┘         │ 8x7B     │       │ VL / Audio│  │
│                       │ 8x22B    │       └──────────┘   │
│  Google (Gemma)       │ Large 2  │       Microsoft      │
│  ┌──────────┐         └──────────┘       ┌──────────┐   │
│  │ Gemma 2  │                            │ Phi-3/4  │   │
│  │ 2B / 9B  │         DeepSeek           │ mini/med │   │
│  │ / 27B    │         ┌──────────┐       └──────────┘   │
│  └──────────┘         │ V3 / R1  │                      │
│                       │ 671B MoE │                      │
│                       └──────────┘                      │
└──────────────────────────────────────────────────────────┘
```

### 1.1 オープンソースと「オープンウェイト」の違い

```
┌──────────────────────────────────────────────────────────┐
│        オープン度合いの分類                                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  フルオープンソース (Truly Open Source)                    │
│  ├── モデル重み: 公開                                     │
│  ├── 訓練コード: 公開                                     │
│  ├── 訓練データ: 公開 (または詳細記述)                     │
│  ├── ライセンス: OSI 承認ライセンス (MIT, Apache 2.0 等)  │
│  └── 例: OLMo (AI2), Pythia (EleutherAI)                 │
│                                                          │
│  オープンウェイト (Open Weight)                           │
│  ├── モデル重み: 公開                                     │
│  ├── 訓練コード: 一部公開 / 非公開                        │
│  ├── 訓練データ: 非公開                                   │
│  ├── ライセンス: 独自ライセンス (使用制限あり)            │
│  └── 例: Llama 3 (Meta), Gemma (Google)                  │
│                                                          │
│  実質的にオープンソース (Permissive Open)                 │
│  ├── モデル重み: 公開                                     │
│  ├── ライセンス: Apache 2.0 / MIT                        │
│  ├── 商用利用: 制限なし                                   │
│  └── 例: Qwen 2.5 (Apache 2.0), DeepSeek (MIT)          │
│                                                          │
│  注意: 「オープンソース LLM」は厳密にはオープンウェイトが │
│  多いが、業界慣行として「オープンソース」と呼ばれる        │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Llama (Meta)

### 2.1 Llama 3 シリーズ

```python
# Llama 3 をHugging Face Transformers で利用
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "あなたは有能なアシスタントです。"},
    {"role": "user", "content": "Pythonのデコレータを説明してください。"},
]

input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt"
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2.2 Llama の特徴

| 特徴 | 詳細 |
|------|------|
| パラメータ | 8B / 70B / 405B |
| コンテキスト長 | 128K トークン |
| 訓練データ | 15T+ トークン (多言語) |
| ライセンス | Llama 3 Community License (月間 7 億ユーザー未満は無料) |
| 対応言語 | 英語中心 + 多言語 (日本語は中程度) |
| 特筆事項 | 405B は OSS 最大級、GPT-4 レベル |

### 2.3 Llama のアーキテクチャ詳細

```
┌──────────────────────────────────────────────────────────┐
│            Llama 3 アーキテクチャの特徴                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Transformer Decoder-Only                                │
│  ├── Grouped Query Attention (GQA)                      │
│  │   └── Key-Value ヘッドを共有しメモリ削減              │
│  ├── RoPE (Rotary Position Embeddings)                  │
│  │   └── 128K まで拡張されたコンテキスト長               │
│  ├── SwiGLU 活性化関数                                   │
│  │   └── ReLU より高品質な勾配伝搬                       │
│  └── RMSNorm (Pre-Normalization)                        │
│      └── LayerNorm より計算効率が良い                    │
│                                                          │
│  トークナイザ:                                           │
│  ├── tiktoken ベース (128K 語彙)                         │
│  ├── Llama 2 の 32K から大幅拡張                         │
│  └── 多言語のトークン効率が改善                          │
│                                                          │
│  訓練特性:                                               │
│  ├── 15T+ トークンの大規模コーパス                       │
│  ├── 405B は 16K 台の H100 GPU で訓練                    │
│  ├── DPO (Direct Preference Optimization) でアラインメント│
│  └── ツール使用・コード生成を後学習で強化                │
└──────────────────────────────────────────────────────────┘
```

### 2.4 Llama を使ったファインチューニング実装

```python
# LoRA を使った Llama 3 のファインチューニング
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 4bit量子化でモデルをロード
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 設定
lora_config = LoraConfig(
    r=16,                        # LoRA のランク
    lora_alpha=32,               # スケーリング係数
    target_modules=[             # 適用対象レイヤー
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 訓練可能パラメータの確認
model.print_trainable_parameters()
# → trainable params: 41,943,040 || all params: 8,030,261,248 || 0.52%

# データセットの準備
dataset = load_dataset("json", data_files="train_data.jsonl")

def format_instruction(example):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
あなたは有能なアシスタントです。<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""

# 訓練実行
training_args = TrainingArguments(
    output_dir="./llama3-ft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    formatting_func=format_instruction,
    max_seq_length=2048,
)

trainer.train()
model.save_pretrained("./llama3-ft-lora")
```

---

## 3. Mistral AI

### 3.1 Mixtral (MoE アーキテクチャ)

```python
# Mixtral 8x7B の利用例 (vLLM)
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,  # 2 GPU で分割
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,
)

outputs = llm.generate(
    ["Rustの所有権システムを解説してください。"],
    sampling_params,
)

for output in outputs:
    print(output.outputs[0].text)
```

### 3.2 Mistral モデルラインナップ

```
┌─────────────────────────────────────────────────┐
│          Mistral AI モデル系譜                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Dense Models:                                  │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ Mistral 7B  │  │ Mistral     │              │
│  │ (2023/09)   │  │ Large 2     │              │
│  │ 7.3B params │  │ (2024/07)   │              │
│  └─────────────┘  │ 123B params │              │
│                    └─────────────┘              │
│  MoE Models:                                    │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ Mixtral     │  │ Mixtral     │              │
│  │ 8x7B       │  │ 8x22B      │              │
│  │ (2023/12)   │  │ (2024/04)   │              │
│  │ 46.7B total │  │ 176B total  │              │
│  │ 12.9B act.  │  │ 39B active  │              │
│  └─────────────┘  └─────────────┘              │
│                                                 │
│  Special Purpose:                               │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ Codestral   │  │ Mistral     │              │
│  │ (Code)      │  │ Nemo (12B)  │              │
│  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────┘
```

### 3.3 MoE (Mixture of Experts) アーキテクチャの詳細

```
┌──────────────────────────────────────────────────────────┐
│            MoE (Mixture of Experts) の仕組み              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  通常の Transformer:                                     │
│  入力 → Attention → FFN (全パラメータ使用) → 出力        │
│  └── 計算コスト: O(全パラメータ数)                        │
│                                                          │
│  MoE Transformer:                                        │
│  入力 → Attention → Router → Expert 2つだけ活性化 → 出力 │
│                      │                                   │
│                      ├── Expert 1: ✗ (非活性)            │
│                      ├── Expert 2: ✓ (活性)              │
│                      ├── Expert 3: ✗                     │
│                      ├── Expert 4: ✗                     │
│                      ├── Expert 5: ✓ (活性)              │
│                      ├── Expert 6: ✗                     │
│                      ├── Expert 7: ✗                     │
│                      └── Expert 8: ✗                     │
│                                                          │
│  Mixtral 8x7B の場合:                                    │
│  ├── 総パラメータ: 46.7B (8 Expert × 約5B + 共有層)      │
│  ├── 活性パラメータ: 12.9B (2 Expert のみ)               │
│  ├── 推論速度: 12.9B 相当 (7B Dense の約2倍)             │
│  ├── 品質: 70B Dense モデルに匹敵                        │
│  └── メリット: 高品質 + 低推論コスト                      │
│                                                          │
│  Router の学習:                                           │
│  ├── Load Balancing Loss で Expert 利用を均等化            │
│  ├── Top-k (通常 k=2) の Expert を選択                   │
│  └── ソフトマックスで重み付け合成                         │
└──────────────────────────────────────────────────────────┘
```

### 3.4 Mistral API の利用

```python
# Mistral API (OpenAI互換形式)
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_MISTRAL_API_KEY",
    base_url="https://api.mistral.ai/v1",
)

response = client.chat.completions.create(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": "あなたはプログラミングの専門家です。"},
        {"role": "user", "content": "Go言語のgoroutineとチャネルの使い方を教えてください。"},
    ],
    temperature=0.7,
    max_tokens=2048,
)

print(response.choices[0].message.content)

# Function Calling も対応
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "株価を取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "銘柄コード"},
                },
                "required": ["symbol"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Appleの株価を教えて"}],
    tools=tools,
    tool_choice="auto",
)
```

---

## 4. Qwen (Alibaba Cloud)

### 4.1 Qwen 2.5 の利用

```python
# Qwen 2.5 — 日本語性能が高いOSSモデル
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "あなたは日本語に堪能なアシスタントです。"},
    {"role": "user", "content": "日本の四季について俳句を3つ作ってください。"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 4.2 Qwen の特筆事項

- **日本語・中国語の性能が極めて高い** (CJK 言語圏で最強クラス)
- サイズバリエーションが豊富 (0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B)
- Qwen2.5-Coder: コード特化モデル
- Qwen-VL: 視覚言語モデル
- Apache 2.0 ライセンス (最も自由度が高い)

### 4.3 Qwen マルチモーダルモデルの利用

```python
# Qwen-VL (Vision-Language) の利用
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 画像付きプロンプト
image = Image.open("diagram.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "この図を詳しく説明してください。"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### 4.4 Qwen2.5-Coder によるコード生成

```python
# Qwen2.5-Coder — コード生成特化モデル
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

messages = [
    {"role": "system", "content": "あなたは熟練のソフトウェアエンジニアです。"},
    {"role": "user", "content": """
以下の要件でPythonのクラスを実装してください:
- 名前: AsyncRateLimiter
- Token Bucket アルゴリズムによるレート制限
- asyncio 対応
- 設定可能なレート (リクエスト/秒) とバースト数
"""},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. DeepSeek

### 5.1 DeepSeek-R1 (推論特化)

```python
# DeepSeek-R1 は思考過程を明示的に出力
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_KEY",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "271は素数ですか？証明してください。"}
    ]
)

# reasoning_content に思考過程が含まれる
print("思考過程:", response.choices[0].message.reasoning_content)
print("最終回答:", response.choices[0].message.content)
```

### 5.2 DeepSeek-V3 の技術的特徴

```
┌──────────────────────────────────────────────────────────┐
│            DeepSeek-V3 技術詳細                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  アーキテクチャ: MoE (Mixture of Experts)                │
│  ├── 総パラメータ: 671B                                  │
│  ├── 活性パラメータ: 37B (各トークンで)                   │
│  ├── Expert 数: 256 (うち 8 が活性化)                     │
│  └── 共有 Expert: 1 (全トークンで常に活性化)              │
│                                                          │
│  革新的技術:                                              │
│  ├── Multi-head Latent Attention (MLA)                   │
│  │   └── KV キャッシュを圧縮し推論効率向上               │
│  ├── FP8 混合精度訓練                                    │
│  │   └── 訓練コストを大幅削減 ($5.5M で訓練完了)         │
│  └── 負荷分散なしの Expert ルーティング                   │
│      └── Auxiliary Loss を不要にした新手法                │
│                                                          │
│  性能:                                                   │
│  ├── MMLU: 87.1 (GPT-4o 級)                             │
│  ├── MATH: 90.2 (数学で GPT-4o を上回る)                 │
│  ├── コスト: 入力 $0.27/1M, 出力 $1.10/1M (激安)        │
│  └── API: OpenAI 互換形式で利用可能                       │
│                                                          │
│  DeepSeek-R1:                                            │
│  ├── 推論特化モデル (o1 対抗)                             │
│  ├── Chain-of-Thought を明示的に出力                      │
│  ├── 蒸留版: 1.5B / 7B / 8B / 14B / 32B / 70B           │
│  └── MIT ライセンス (完全自由)                            │
└──────────────────────────────────────────────────────────┘
```

### 5.3 DeepSeek-R1 蒸留モデルのローカル利用

```python
# DeepSeek-R1 の蒸留版をローカルで実行 (Ollama)
import subprocess
import requests

# Ollama でモデルをダウンロード・実行
# $ ollama pull deepseek-r1:7b

# API 経由で利用
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "deepseek-r1:7b",
        "messages": [
            {"role": "user", "content": "フィボナッチ数列の第100項を求めるアルゴリズムを説明してください。"}
        ],
        "stream": False,
    },
)

result = response.json()
print(result["message"]["content"])
```

---

## 6. その他の注目 OSS モデル

### 6.1 Gemma (Google)

```python
# Gemma 2 — Google の軽量高性能モデル
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# Gemma 2 の特徴:
# - 知識蒸留 (Knowledge Distillation) で小型化
# - Sliding Window Attention + Global Attention の交互使用
# - 2B / 9B / 27B のサイズバリエーション
# - Gemma License (研究・商用利用可、再配布時にライセンス添付必要)
```

### 6.2 Phi (Microsoft)

```python
# Phi-3/4 — Microsoft の小型高性能モデル
# Phi-3 Mini (3.8B) は同サイズで最高性能クラス

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

# Phi の特徴:
# - 高品質な「教科書レベル」のデータで訓練
# - 小型でもGPT-3.5を上回る性能
# - MIT ライセンス
# - 128K コンテキスト版も存在
# - Phi-4 (14B) は Qwen 2.5 72B と同等性能を主張
```

### 6.3 日本語特化モデル

```
┌──────────────────────────────────────────────────────────┐
│          日本語特化 / 日本語強化 OSS モデル                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  CyberAgent CALM3 (68B)                                  │
│  ├── サイバーエージェント開発                              │
│  ├── Apache 2.0 ライセンス                                │
│  ├── 日本語コーパスで追加事前学習                          │
│  └── 日本語ベンチマーク (JGLUE) で高性能                  │
│                                                          │
│  ELYZA Llama 日本語シリーズ                               │
│  ├── Llama ベースに日本語ファインチューニング              │
│  ├── ELYZA-tasks-100 で評価                               │
│  └── 比較的小型で実用的                                   │
│                                                          │
│  PLaMo (Preferred Networks)                              │
│  ├── 日本発のフルスクラッチ LLM                           │
│  ├── 日本語・英語のバイリンガル訓練                       │
│  └── 研究用途が中心                                       │
│                                                          │
│  Swallow (東京工業大学 + 産総研)                          │
│  ├── Llama ベースの日本語継続事前学習                      │
│  ├── 7B / 13B / 70B                                      │
│  └── 日本語ベンチマークで高い性能                         │
│                                                          │
│  Tanuki (松尾研究室)                                     │
│  ├── 日本語 Web コーパスで訓練                             │
│  └── 研究目的で公開                                       │
└──────────────────────────────────────────────────────────┘
```

---

## 7. モデル比較表

### 7.1 性能・スペック比較

| モデル | パラメータ | MoE | コンテキスト | 日本語 | ライセンス |
|--------|----------|-----|------------|--------|-----------|
| Llama 3.1 405B | 405B | No | 128K | 中 | Community |
| Llama 3.1 70B | 70B | No | 128K | 中 | Community |
| Mixtral 8x22B | 176B/39B活性 | Yes | 64K | 中 | Apache 2.0 |
| Qwen 2.5 72B | 72B | No | 128K | 高 | Apache 2.0 |
| DeepSeek-V3 | 671B/37B活性 | Yes | 128K | 中 | MIT |
| Gemma 2 27B | 27B | No | 8K | 低 | Gemma License |
| Phi-3 Medium | 14B | No | 128K | 低 | MIT |

### 7.2 ユースケース別推奨

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| 日本語チャットボット | Qwen 2.5 (7B-72B) | 日本語性能最高 |
| コード生成 | DeepSeek-Coder / Qwen-Coder | コード特化訓練 |
| 数学・推論 | DeepSeek-R1 | Chain-of-Thought 推論特化 |
| エッジデバイス | Phi-3 mini / Gemma 2B | 軽量で高性能 |
| 汎用・最高精度 | Llama 3.1 405B | OSS 最大パラメータ |
| コスト最適化 | Mixtral 8x7B | MoE で低推論コスト |

### 7.3 VRAM 要件とハードウェア選定

| モデルサイズ | FP16 | INT8 | INT4 (GPTQ/AWQ) | 推奨 GPU |
|------------|------|------|-----------------|---------|
| 3B | 6GB | 3GB | 2GB | RTX 3060 12GB |
| 7-8B | 16GB | 8GB | 4GB | RTX 4070 12GB |
| 14B | 28GB | 14GB | 8GB | RTX 4090 24GB |
| 32B | 64GB | 32GB | 16GB | A100 40GB |
| 70B | 140GB | 70GB | 35GB | 2×A100 80GB |
| 405B | 810GB | 405GB | 200GB | 8×A100 80GB |

---

## 8. 実運用デプロイ

### 8.1 量子化による最適化

```python
# GPTQ 量子化モデルの利用
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-3-8B-Instruct-GPTQ",
    device_map="auto",
    trust_remote_code=True,
)

# メモリ使用量の比較
# FP16:  8B モデル → 約 16GB VRAM
# INT8:  8B モデル → 約  8GB VRAM
# INT4:  8B モデル → 約  4GB VRAM (GPTQ/AWQ)
# GGUF:  8B モデル → 約  4-6GB (llama.cpp、CPU可)
```

### 8.2 推論サーバーの選定と構築

```python
# vLLM — 高性能推論サーバー
# $ pip install vllm
# $ vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000

from openai import OpenAI

# vLLM は OpenAI 互換 API を提供
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM ではダミーでOK
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

```
┌──────────────────────────────────────────────────────────┐
│          推論サーバー比較                                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  vLLM:                                                   │
│  ├── PagedAttention でメモリ効率最高                      │
│  ├── Continuous Batching でスループット最大化              │
│  ├── OpenAI 互換 API                                     │
│  ├── Tensor/Pipeline Parallelism 対応                    │
│  └── 推奨: 本番サーバー用途                               │
│                                                          │
│  Text Generation Inference (TGI):                        │
│  ├── Hugging Face 公式                                    │
│  ├── Docker コンテナで簡単デプロイ                        │
│  ├── Flash Attention 2 対応                               │
│  └── 推奨: HF エコシステム利用時                          │
│                                                          │
│  Ollama:                                                 │
│  ├── ワンコマンドで起動 (ollama run llama3.1)             │
│  ├── GGUF 形式で CPU/GPU 両対応                           │
│  ├── macOS / Linux / Windows 対応                        │
│  └── 推奨: ローカル開発・プロトタイプ                     │
│                                                          │
│  llama.cpp:                                              │
│  ├── C/C++ 実装で依存関係最小                             │
│  ├── CPU 推論に最適化 (AVX, ARM NEON)                     │
│  ├── Apple Silicon (Metal) 対応                           │
│  └── 推奨: エッジ / 組み込み / CPU 環境                   │
│                                                          │
│  スループット比較 (8B モデル, A100):                      │
│  ├── vLLM:    ~2000 tokens/s                              │
│  ├── TGI:     ~1500 tokens/s                              │
│  ├── Ollama:  ~100 tokens/s (GPU)                         │
│  └── llama.cpp: ~30 tokens/s (CPU)                        │
└──────────────────────────────────────────────────────────┘
```

### 8.3 Docker を使った本番デプロイ

```dockerfile
# vLLM を使った本番デプロイ用 Dockerfile
FROM vllm/vllm-openai:latest

# モデルのダウンロード (事前ダウンロードも可)
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.9

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_NAME}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  llm-server:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model Qwen/Qwen2.5-7B-Instruct
      --max-model-len 8192
      --host 0.0.0.0
      --port 8000
    volumes:
      - model-cache:/root/.cache/huggingface

volumes:
  model-cache:
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

```
┌──────────────────────────────────────────────────────────┐
│          OSS LLM デプロイのトラブルシューティング           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題 1: CUDA Out of Memory                              │
│  ├── 原因: モデルが GPU メモリに収まらない                │
│  ├── 解決策 1: 量子化モデルを使用 (GPTQ/AWQ/GGUF)        │
│  ├── 解決策 2: gpu_memory_utilization を下げる (0.8等)    │
│  ├── 解決策 3: max_model_len を短くする                   │
│  └── 解決策 4: tensor_parallel_size を増やす (複数GPU)    │
│                                                          │
│  問題 2: モデルのダウンロードが遅い / 失敗する            │
│  ├── 原因: HuggingFace Hub の帯域制限                     │
│  ├── 解決策 1: HF_TOKEN を設定してゲート付きモデルに対応  │
│  ├── 解決策 2: huggingface-cli download でプリフェッチ    │
│  └── 解決策 3: ミラー (hf-mirror.com 等) を利用           │
│                                                          │
│  問題 3: 推論速度が遅い                                   │
│  ├── 原因 1: Flash Attention 未使用                       │
│  │   └── pip install flash-attn でインストール            │
│  ├── 原因 2: バッチサイズが小さい                         │
│  │   └── max_num_seqs を増やす                            │
│  └── 原因 3: KV キャッシュが不足                          │
│      └── block_size を調整                                │
│                                                          │
│  問題 4: 出力品質が低い (英語は良いが日本語が悪い)        │
│  ├── 原因: モデルの日本語訓練データが不足                 │
│  ├── 解決策 1: Qwen 2.5 など日本語に強いモデルに変更      │
│  ├── 解決策 2: 日本語データでファインチューニング          │
│  └── 解決策 3: System Prompt で「日本語で回答」を明示     │
│                                                          │
│  問題 5: ライセンス違反のリスク                           │
│  ├── Llama: 月間 7億 MAU 以上で要連絡                     │
│  ├── Gemma: 再配布時ライセンス添付必須                    │
│  └── 安全策: Apache 2.0 / MIT モデルを選択               │
└──────────────────────────────────────────────────────────┘
```

### 9.2 性能ベンチマークの実施

```python
# 自社環境での推論性能ベンチマーク
import time
import statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def benchmark_model(
    model: str,
    prompts: list[str],
    max_tokens: int = 256,
    num_runs: int = 3,
) -> dict:
    """推論性能のベンチマーク"""
    latencies = []
    throughputs = []

    for prompt in prompts:
        for _ in range(num_runs):
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            elapsed = time.time() - start

            output_tokens = response.usage.completion_tokens
            latencies.append(elapsed)
            throughputs.append(output_tokens / elapsed)

    return {
        "model": model,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_latency_ms": statistics.median(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "avg_throughput_tps": statistics.mean(throughputs),
        "total_requests": len(latencies),
    }

# ベンチマーク実行
test_prompts = [
    "Pythonでバブルソートを実装してください。",
    "機械学習とディープラーニングの違いを説明してください。",
    "日本の歴史について500文字程度で要約してください。",
]

result = benchmark_model("Qwen/Qwen2.5-7B-Instruct", test_prompts)
print(f"平均レイテンシ: {result['avg_latency_ms']:.0f}ms")
print(f"P95 レイテンシ: {result['p95_latency_ms']:.0f}ms")
print(f"平均スループット: {result['avg_throughput_tps']:.1f} tokens/s")
```

---

## 10. アンチパターン

### アンチパターン 1: ライセンス不確認での商用利用

```
# NG: ライセンスを確認せず商用プロダクトに組み込む

Llama 3 Community License:
  → 月間アクティブユーザー 7 億人以上の場合は Meta への連絡が必要
  → 出力を他の LLM の訓練に使用することを禁止

Gemma License:
  → 再配布時にライセンス条件の添付が必要

# OK: Apache 2.0 ライセンスのモデルを選択
Qwen 2.5, Mixtral → 商用利用に制限なし
DeepSeek → MIT ライセンスで最も自由
```

### アンチパターン 2: モデルサイズだけで選定

```
# NG: "大きいほど良い" という単純な判断
model = "llama-3.1-405b"  # 16台のA100が必要...

# OK: タスク特性に応じた選定
# 分類・抽出 → 7B-8B で十分な場合が多い
# 創造的文章 → 70B クラスが有効
# ファインチューニング → 小さいモデルの方が現実的
# コスト重視 → MoE モデル (Mixtral) が有利
```

### アンチパターン 3: 量子化なしでのデプロイ

```python
# NG: 本番環境で FP16 のまま大型モデルを運用
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    torch_dtype=torch.float16,  # 140GB VRAM 必要
)

# OK: 用途に応じた量子化を適用
# 品質重視 → INT8 (AWQ): 品質低下 1-2%、メモリ半減
# コスト重視 → INT4 (GPTQ): 品質低下 3-5%、メモリ 1/4
# CPU 実行 → GGUF Q4_K_M: llama.cpp で CPU 推論可能
```

### アンチパターン 4: セキュリティ対策なしでの公開

```python
# NG: 認証なしで LLM API を公開
# vllm serve model --host 0.0.0.0  # インターネットから誰でもアクセス可能

# OK: 適切な認証・レート制限・プロンプトインジェクション対策
# 1. リバースプロキシ (nginx) で認証を追加
# 2. API キーによるアクセス制御
# 3. レート制限 (リクエスト数/秒)
# 4. 入力長の制限
# 5. 出力フィルタリング (有害コンテンツ検出)
```

---

## 11. ベストプラクティス

### 11.1 モデル選定チェックリスト

```
┌──────────────────────────────────────────────────────────┐
│          OSS LLM 選定チェックリスト                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  □ ライセンス確認                                        │
│    ├── 商用利用可能か                                     │
│    ├── 再配布条件は                                       │
│    └── 出力の利用制限は                                   │
│                                                          │
│  □ 性能要件の確認                                        │
│    ├── 対象言語での品質 (日本語なら Qwen 推奨)            │
│    ├── 必要な推論速度 (tokens/s)                          │
│    └── 許容できる品質低下 (量子化の影響)                  │
│                                                          │
│  □ インフラ要件の確認                                    │
│    ├── 利用可能な GPU / メモリ                             │
│    ├── スケーラビリティ要件                               │
│    └── 可用性・冗長性要件                                 │
│                                                          │
│  □ 運用要件の確認                                        │
│    ├── モデル更新の頻度と手順                             │
│    ├── モニタリング・アラート                             │
│    └── 障害時の復旧手順                                   │
│                                                          │
│  □ セキュリティ要件の確認                                │
│    ├── データがクラウドに出ない要件                       │
│    ├── プロンプトインジェクション対策                     │
│    └── 出力フィルタリング要件                             │
└──────────────────────────────────────────────────────────┘
```

### 11.2 段階的な導入戦略

```
Phase 1: 評価 (1-2週間)
├── 候補モデル 3-5 個を Ollama でローカル評価
├── 自社タスクの評価データセット (50-100問) で品質比較
└── 量子化レベル (FP16/INT8/INT4) 別の品質・速度比較

Phase 2: プロトタイプ (2-4週間)
├── vLLM/TGI で推論サーバー構築
├── 既存アプリケーションとの統合テスト
└── 負荷テスト・レイテンシ測定

Phase 3: 本番デプロイ (2-4週間)
├── Docker/Kubernetes でコンテナ化
├── モニタリング・ロギング設定
├── オートスケーリング設定
└── フォールバック戦略 (API モデルへの切り替え)

Phase 4: 運用・改善 (継続)
├── 品質モニタリング (LLM-as-a-Judge)
├── ファインチューニングの検討
├── モデル更新時の評価パイプライン
└── コスト最適化 (量子化レベル調整、バッチ最適化)
```

---

## 12. FAQ

### Q1: オープンソース LLM と API ベースのモデルはどう使い分ける?

データプライバシーが重要な場合、レイテンシ要件が厳しい場合、大量推論でコスト最適化したい場合は OSS モデルの自前デプロイが有利。
一方、最新モデルを常に利用したい場合や、インフラ管理を避けたい場合は API ベースが適切。
ハイブリッド構成 (機密データは OSS、それ以外は API) も有効な戦略。

### Q2: ファインチューニングするなら何パラメータのモデルが良い?

一般的に 7B-14B クラスが最もコストパフォーマンスが高い。
LoRA/QLoRA を使えば消費者向け GPU (RTX 4090 / 24GB) でも 7B モデルのファインチューニングが可能。
70B 以上は複数 GPU が必須で、ファインチューニングコストも大幅に増加する。

### Q3: 日本語タスクに最適な OSS モデルは?

2025 年時点では Qwen 2.5 シリーズが日本語性能で最高クラス。
日本発のモデルとしては、CyberAgent の CALM3、Preferred Networks の PLaMo、
ELYZA の Llama 日本語ファインチューンなどがある。
用途が限定的なら、日本語特化の小型モデルの方が汎用大型モデルより高精度な場合もある。

### Q4: OSS モデルの品質はプロプライエタリモデルにどこまで迫っている?

DeepSeek-V3 や Llama 3.1 405B は GPT-4o とほぼ同等のベンチマークスコアを達成。
特定タスク (数学ではDeepSeek-R1が o1 と同等、コードでは Qwen-Coder が高性能) では
プロプライエタリモデルを上回る場合もある。
ただし、総合的な指示追従、安全性、ハルシネーション制御ではまだ差がある。

### Q5: MoE モデルと Dense モデルのどちらを選ぶべき?

推論コストを重視するなら MoE (Mixtral, DeepSeek-V3)。同等品質で推論 FLOPs が少ない。
ファインチューニングのしやすさを重視するなら Dense (Llama, Qwen)。MoE のファインチューニングは
Expert 間のバランス調整が難しく、ノウハウが少ない。
デプロイの簡単さでも Dense が有利 (MoE は総パラメータ分のメモリが必要)。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 三大勢力 | Meta Llama、Mistral AI、Alibaba Qwen |
| 注目モデル | DeepSeek-R1 (推論)、Phi-3 (軽量) |
| 最大モデル | Llama 3.1 405B (Dense)、DeepSeek-V3 671B (MoE) |
| 日本語最強 | Qwen 2.5 シリーズ |
| コスト最適 | MoE モデル (Mixtral、DeepSeek-V3) |
| ライセンス推奨 | Apache 2.0 (Qwen, Mixtral) / MIT (DeepSeek) |
| デプロイ手段 | vLLM、TGI、Ollama、llama.cpp |
| ファインチューニング推奨サイズ | 7B-14B (LoRA/QLoRA で消費者GPU可) |

---

## 次に読むべきガイド

- [04-model-comparison.md](./04-model-comparison.md) — 全モデル横断ベンチマーク比較
- [../../03-infrastructure/02-local-llm.md](../03-infrastructure/02-local-llm.md) — ローカル LLM のデプロイ実践
- [../../03-infrastructure/03-evaluation.md](../03-infrastructure/03-evaluation.md) — LLM 評価手法

---

## 参考文献

1. Dubey et al., "The Llama 3 Herd of Models," arXiv:2407.21783, 2024
2. Jiang et al., "Mixtral of Experts," arXiv:2401.04088, 2024
3. Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024
4. DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024
5. Hugging Face, "Open LLM Leaderboard," https://huggingface.co/spaces/open-llm-leaderboard
6. Abdin et al., "Phi-3 Technical Report," arXiv:2404.14219, 2024
7. Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size," arXiv:2408.00118, 2024
