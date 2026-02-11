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

---

## 6. モデル比較表

### 6.1 性能・スペック比較

| モデル | パラメータ | MoE | コンテキスト | 日本語 | ライセンス |
|--------|----------|-----|------------|--------|-----------|
| Llama 3.1 405B | 405B | No | 128K | 中 | Community |
| Llama 3.1 70B | 70B | No | 128K | 中 | Community |
| Mixtral 8x22B | 176B/39B活性 | Yes | 64K | 中 | Apache 2.0 |
| Qwen 2.5 72B | 72B | No | 128K | 高 | Apache 2.0 |
| DeepSeek-V3 | 671B/37B活性 | Yes | 128K | 中 | MIT |
| Gemma 2 27B | 27B | No | 8K | 低 | Gemma License |
| Phi-3 Medium | 14B | No | 128K | 低 | MIT |

### 6.2 ユースケース別推奨

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| 日本語チャットボット | Qwen 2.5 (7B-72B) | 日本語性能最高 |
| コード生成 | DeepSeek-Coder / Qwen-Coder | コード特化訓練 |
| 数学・推論 | DeepSeek-R1 | Chain-of-Thought 推論特化 |
| エッジデバイス | Phi-3 mini / Gemma 2B | 軽量で高性能 |
| 汎用・最高精度 | Llama 3.1 405B | OSS 最大パラメータ |
| コスト最適化 | Mixtral 8x7B | MoE で低推論コスト |

---

## 7. 実運用デプロイ

### 7.1 量子化による最適化

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

---

## 8. アンチパターン

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

---

## 9. FAQ

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
