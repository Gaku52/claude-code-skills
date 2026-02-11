# ファインチューニング — モデルをタスクに特化させる技法

> LoRA、QLoRA、RLHF、DPO など、LLM を自分のデータ・タスクに最適化するための主要手法を実践的に解説する。

## この章で学ぶこと

1. **LoRA / QLoRA** による効率的なパラメータ調整の仕組みと実装
2. **RLHF と DPO** によるアラインメント手法の原理と選択基準
3. **実践的なファインチューニング**のワークフローと評価方法

---

## 1. LoRA (Low-Rank Adaptation)

### ASCII 図解 1: LoRA の仕組み

```
通常のファインチューニング:
┌─────────────┐
│ 全パラメータ W │  ← 全て更新 (数十億パラメータ)
│  (d × d)     │     GPU メモリ大量消費
└─────────────┘

LoRA:
┌─────────────┐
│ 元の重み W    │  ← 凍結 (更新しない)
│  (d × d)     │
└──────┬──────┘
       │
       + (加算)
       │
┌──────┴──────┐
│  ΔW = B × A  │  ← 低ランク行列のみ更新
│              │
│ B: (d × r)   │  r << d (例: r=8, d=4096)
│ A: (r × d)   │
│              │  パラメータ数: 2 × d × r
│ 例: 2×4096×8 │  = 65,536 (元の0.01%以下)
└─────────────┘
```

### コード例 1: LoRA でのファインチューニング (PEFT)

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# ベースモデルのロード
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # ランク (8-64 が一般的)
    lora_alpha=32,                 # スケーリング係数
    lora_dropout=0.05,             # ドロップアウト率
    target_modules=[               # 適用する層
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# LoRA を適用
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.52%
```

### コード例 2: QLoRA (4bit量子化 + LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4bit 量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # 二重量子化
)

# QLoRA: 4bit量子化モデル + LoRA
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA を適用（上記と同じ設定）
model = get_peft_model(model, lora_config)
# VRAM: 70B → ~40GB (QLoRA) vs ~140GB (フル精度)
```

### ASCII 図解 2: ファインチューニング手法の GPU メモリ比較

```
GPU VRAM 使用量 (Llama 3.1 8B の場合)

フル FT:     ████████████████████████████████████  ~60GB
             全パラメータ更新 + 勾配 + オプティマイザ

LoRA (fp16): ████████████████████░░░░░░░░░░░░░░░  ~20GB
             モデル(fp16) + LoRA勾配のみ

QLoRA (4bit):████████████░░░░░░░░░░░░░░░░░░░░░░░  ~8GB
             モデル(4bit) + LoRA勾配(bf16)

推論のみ:    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~5GB
             モデル(4bit)のみ

             0    10   20   30   40   50   60 GB
```

---

## 2. RLHF と DPO

### ASCII 図解 3: RLHF vs DPO のワークフロー

```
RLHF (Reinforcement Learning from Human Feedback):
┌─────────┐    ┌──────────┐    ┌──────────┐
│ SFT モデル│ →  │ 応答生成  │ →  │ 人間評価  │
└─────────┘    └──────────┘    └────┬─────┘
                                     │
                                     ▼
┌─────────┐    ┌──────────┐    ┌──────────┐
│ 最終モデル│ ←  │ PPO 学習  │ ←  │報酬モデル │
└─────────┘    └──────────┘    └──────────┘
                 (不安定)        (別途学習が必要)

DPO (Direct Preference Optimization):
┌─────────┐    ┌──────────┐    ┌──────────┐
│ SFT モデル│ →  │ 応答ペア  │ →  │ 人間評価  │
└─────────┘    │ 生成      │    │ (好み順) │
               └──────────┘    └────┬─────┘
                                     │
                                     ▼
┌─────────┐                   ┌──────────┐
│ 最終モデル│ ←─────────────── │ DPO 損失  │
└─────────┘   (直接最適化)     │ で学習    │
               (安定)          └──────────┘
               報酬モデル不要
```

### コード例 3: DPO でのアラインメント学習

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 好みデータセットの形式
# {"prompt": "...", "chosen": "良い応答", "rejected": "悪い応答"}
dataset = load_dataset("your-org/preference-dataset")

# DPO 設定
training_args = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,              # KLペナルティの強さ
    loss_type="sigmoid",   # "sigmoid", "hinge", "ipo"
    logging_steps=10,
    save_steps=100,
    bf16=True,
)

# DPO トレーナー
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # PEFT使用時は自動でリファレンスモデル生成
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./dpo-final")
```

### コード例 4: 学習データの準備（SFT 形式）

```python
from datasets import Dataset

# SFT 用の指示-応答データ
sft_data = [
    {
        "instruction": "以下のPythonコードのバグを修正してください。",
        "input": "def add(a, b):\n    return a - b",
        "output": "def add(a, b):\n    return a + b\n\n# 修正: 減算(-) を加算(+) に変更しました。"
    },
    {
        "instruction": "SQLクエリを最適化してください。",
        "input": "SELECT * FROM users WHERE name LIKE '%田中%'",
        "output": "SELECT id, name, email FROM users WHERE name LIKE '%田中%'\n\n-- 改善点:\n-- 1. SELECT * を必要なカラムに限定\n-- 2. LIKE前方一致の場合はインデックスが効くが、中間一致は全件走査"
    },
]

# チャットテンプレートに変換
def format_chat(example):
    messages = [
        {"role": "system", "content": "あなたは優秀なプログラミングアシスタントです。"},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list(sft_data).map(format_chat)
```

### コード例 5: OpenAI でのファインチューニング（API経由）

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. 学習データの準備 (JSONL 形式)
training_data = [
    {
        "messages": [
            {"role": "system", "content": "技術文書を日本語で要約するアシスタント"},
            {"role": "user", "content": "以下の文書を要約してください: ..."},
            {"role": "assistant", "content": "要約: ..."}
        ]
    },
    # ... 数十〜数百例
]

with open("training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 2. ファイルアップロード
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 3. ファインチューニングジョブ作成
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 1.8,
        "batch_size": 4,
    }
)

# 4. ステータス確認
status = client.fine_tuning.jobs.retrieve(job.id)
print(f"Status: {status.status}")
# ファインチューニング済みモデル: ft:gpt-4o-mini:org-name::job-id
```

---

### 比較表 1: ファインチューニング手法の比較

| 手法 | GPU メモリ | 学習速度 | 品質 | 実装難易度 | コスト |
|------|-----------|---------|------|-----------|--------|
| フル FT | 非常に高い | 遅い | 最高 | 高い | 非常に高い |
| LoRA | 中程度 | 速い | 高い | 中程度 | 中程度 |
| QLoRA | 低い | 速い | 高い | 中程度 | 低い |
| API FT (OpenAI) | 不要 | 中程度 | 中〜高 | 低い | 従量制 |
| プロンプトチューニング | 非常に低い | 非常に速い | 中程度 | 低い | 低い |

### 比較表 2: RLHF vs DPO の詳細比較

| 項目 | RLHF | DPO |
|------|------|-----|
| 報酬モデル | 必要（別途学習） | 不要 |
| 学習安定性 | 不安定（PPOの調整困難） | 安定 |
| 計算コスト | 高い（3モデル並行） | 中程度（2モデル） |
| データ要件 | 比較ペア + 報酬ラベル | 比較ペアのみ |
| 性能 | 高い（調整成功時） | RLHF に匹敵 |
| 実装難易度 | 非常に高い | 中程度 |
| 採用例 | GPT-4, Claude | Llama 3, Zephyr |

---

## アンチパターン

### アンチパターン 1: データ品質を無視した大量データ投入

```
誤: 低品質データ10万件でファインチューニング
  → ノイズを学習、ハルシネーション増加、品質低下

正: 高品質データを厳選
  - 1000件の高品質データ > 10万件の低品質データ
  - 必ず人手でデータ品質を検証
  - 多様なパターンをカバーする代表的な例を選ぶ
```

### アンチパターン 2: ファインチューニングの過信

```
誤: まずファインチューニングから始める
  → 時間とコストの浪費

正: 段階的にアプローチ
  1. まずプロンプトエンジニアリングで解決を試みる
  2. Few-shot 例で改善を試みる
  3. RAG でコンテキスト提供を試みる
  4. それでも不十分な場合にファインチューニング
  → "FT は最後の手段" が基本原則
```

---

## FAQ

### Q1: LoRA のランク r はいくつに設定すべきですか？

**A:** 一般的には r=8〜32 が良い出発点です。タスクが複雑なほど大きなランクが必要ですが、r=64 以上は過学習リスクが高まります。実験的に r=8, 16, 32 で比較し、検証セットの性能で決定するのがベストです。`lora_alpha` は通常 `r` の2倍に設定します。

### Q2: ファインチューニングにはどのくらいのデータが必要ですか？

**A:** タスクによりますが、一般的な目安は以下の通りです。分類タスク: 100〜500例、スタイル調整: 500〜2000例、専門知識の注入: 1000〜5000例、複雑な推論: 5000〜50000例。ただし、データの品質と多様性がデータ量より重要です。

### Q3: ファインチューニングとRAGはどちらを選ぶべきですか？

**A:** 目的に応じて選択します。「動作・スタイルの変更」にはファインチューニング、「知識の追加」にはRAGが適しています。ファインチューニングは一度学習すれば推論時のコストが変わらず、RAG は最新情報を動的に提供できます。多くの場合、両方を組み合わせるのが最適です。

---

## まとめ

| 項目 | 要点 |
|------|------|
| LoRA | 低ランク行列で効率的にモデルを適応、VRAM を大幅削減 |
| QLoRA | 4bit 量子化 + LoRA で 8B モデルを 1 GPU で学習可能 |
| RLHF | 報酬モデルと PPO で人間の好みに合わせる（高性能だが不安定） |
| DPO | 報酬モデル不要で直接最適化（安定・簡単） |
| データ品質 | 量より質が重要、1000件の良質データが10万件に勝つ |
| 段階的アプローチ | プロンプト → Few-shot → RAG → FT の順で検討 |

---

## 次に読むべきガイド

- [../01-models/04-model-comparison.md](../01-models/04-model-comparison.md) — モデル比較とベンチマーク
- [../02-applications/01-rag.md](../02-applications/01-rag.md) — RAG の実装
- [../03-infrastructure/02-local-llm.md](../03-infrastructure/02-local-llm.md) — ローカルLLMと量子化

---

## 参考文献

1. Hu, E. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. https://arxiv.org/abs/2106.09685
2. Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." *NeurIPS 2023*. https://arxiv.org/abs/2305.14314
3. Rafailov, R. et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*. https://arxiv.org/abs/2305.18290
4. Ouyang, L. et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS 2022*. https://arxiv.org/abs/2203.02155
