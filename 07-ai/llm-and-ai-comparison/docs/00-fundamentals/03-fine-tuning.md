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

## 2. LoRA の詳細設計と最適化

### 2.1 LoRA ハイパーパラメータの影響

```
┌──────────────────────────────────────────────────────────┐
│         LoRA ハイパーパラメータの設計空間                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  r (ランク):                                              │
│  ─────────                                               │
│  小さい (4-8)     → パラメータ少、軽量、過学習しにくい     │
│  中程度 (16-32)   → 一般的な推奨値、バランス良好           │
│  大きい (64-128)  → 表現力高、過学習リスク、メモリ増加     │
│                                                          │
│  lora_alpha (スケーリング):                               │
│  ─────────────────────                                   │
│  ΔW の寄与 = (lora_alpha / r) × B × A                   │
│  → alpha/r 比が実効的な学習率スケールを決定               │
│  → 一般的に alpha = 2 × r (例: r=16, alpha=32)          │
│  → alpha が大きすぎると学習不安定                         │
│                                                          │
│  target_modules:                                          │
│  ──────────────                                          │
│  q_proj, v_proj のみ    → 最小構成、軽量                  │
│  + k_proj, o_proj       → 標準構成                       │
│  + gate/up/down_proj    → 全アテンション+FFN (推奨)       │
│  + embed/lm_head        → 最大構成 (稀にしか使わない)     │
│                                                          │
│  lora_dropout:                                            │
│  ────────────                                            │
│  0.0    → ドロップアウトなし (データ量多い場合)            │
│  0.05   → 軽微な正則化 (推奨デフォルト)                   │
│  0.1+   → 強い正則化 (小規模データセット向け)             │
└──────────────────────────────────────────────────────────┘
```

### 2.2 target_modules の選定実験

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 異なる target_modules 設定の比較
configs = {
    "minimal": {
        "target_modules": ["q_proj", "v_proj"],
        "r": 16,
        "lora_alpha": 32,
    },
    "standard": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "r": 16,
        "lora_alpha": 32,
    },
    "full": {
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "r": 16,
        "lora_alpha": 32,
    },
}

for name, config_params in configs.items():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        **config_params,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{name:10s}: trainable={trainable:>12,} ({trainable/total:.2%})")

# 出力例:
# minimal   : trainable=  13,107,200 (0.16%)
# standard  : trainable=  26,214,400 (0.32%)
# full      : trainable=  41,943,040 (0.52%)
```

### 2.3 LoRA の数学的背景

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """LoRA レイヤーの教育的実装"""

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 元の重みを凍結
        for param in self.original.parameters():
            param.requires_grad = False

        in_dim = original_layer.in_features
        out_dim = original_layer.out_features

        # 低ランク行列 A と B
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # A はランダム初期化、B はゼロ初期化
        # → 学習開始時は ΔW = 0 (元のモデルと同一)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 元の出力 + LoRA の低ランク近似
        original_output = self.original(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return original_output + self.scaling * lora_output

    def merge_weights(self):
        """推論時に LoRA 重みを元の重みにマージ (推論高速化)"""
        delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.original.weight.data += delta_w
        return self.original


# 使用例
linear = nn.Linear(4096, 4096)
lora_linear = LoRALayer(linear, r=16, alpha=32)

# 訓練可能パラメータ数
trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora_linear.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.4%})")
# Trainable: 131,072 / 16,908,288 (0.7754%)
```

---

## 3. SFT (Supervised Fine-Tuning) の完全ワークフロー

### 3.1 データセット準備

```python
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 方法1: ローカルデータから作成
raw_data = [
    {
        "instruction": "以下のPythonコードのバグを修正してください。",
        "input": "def add(a, b):\n    return a - b",
        "output": "def add(a, b):\n    return a + b\n\n# 修正: 減算(-) を加算(+) に変更しました。"
    },
    {
        "instruction": "SQLクエリを最適化してください。",
        "input": "SELECT * FROM users WHERE name LIKE '%田中%'",
        "output": (
            "SELECT id, name, email FROM users WHERE name LIKE '%田中%'\n\n"
            "-- 改善点:\n"
            "-- 1. SELECT * を必要なカラムに限定\n"
            "-- 2. LIKE前方一致の場合はインデックスが効くが、中間一致は全件走査"
        )
    },
    # ... 数百〜数千件
]

# チャットテンプレートに変換
def format_chat(example):
    messages = [
        {"role": "system", "content": "あなたは優秀なプログラミングアシスタントです。"},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list(raw_data).map(format_chat)

# 方法2: Hugging Face Hub からロード
dataset_hf = load_dataset("kunishou/databricks-dolly-15k-ja")


# 方法3: JSONL ファイルから
import json

def load_jsonl(filepath: str) -> Dataset:
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)
```

### 3.2 SFTTrainer による学習

```python
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
import torch

# モデルとトークナイザ
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA 用量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Flash Attention 2
)

# LoRA 設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# 学習設定
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,     # 実効バッチサイズ = 4 × 4 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    max_seq_length=2048,
    dataset_text_field="text",
    gradient_checkpointing=True,       # メモリ節約
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",          # メモリ効率的なオプティマイザ
    report_to="wandb",                 # Weights & Biases でモニタリング
)

# トレーナー
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)

# 学習実行
trainer.train()

# モデル保存
trainer.save_model("./output/final")
tokenizer.save_pretrained("./output/final")
```

### 3.3 学習曲線の監視と早期停止

```python
from transformers import TrainerCallback
import matplotlib.pyplot as plt

class LossMonitorCallback(TrainerCallback):
    """学習曲線をリアルタイムで監視"""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)

            # 過学習検出: eval_loss が連続3回上昇
            if len(self.eval_losses) >= 3:
                if (self.eval_losses[-1] > self.eval_losses[-2] >
                    self.eval_losses[-3]):
                    print("WARNING: 過学習の兆候を検出。学習停止を検討してください。")

    def plot(self, save_path: str = "loss_curve.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_losses, label="Train Loss", alpha=0.7)
        if self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses, label="Eval Loss",
                     marker="o", linewidth=2)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Fine-tuning Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"学習曲線を {save_path} に保存しました")


# 使用例
loss_monitor = LossMonitorCallback()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
    callbacks=[loss_monitor],
)

trainer.train()
loss_monitor.plot()
```

---

## 4. RLHF と DPO

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

### 4.1 DPO の数学的背景

```
┌──────────────────────────────────────────────────────────┐
│         DPO 損失関数の直感的理解                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  DPO Loss = -log σ(β × (log π(y_w|x)/π_ref(y_w|x)     │
│                      - log π(y_l|x)/π_ref(y_l|x)))      │
│                                                          │
│  ここで:                                                  │
│  π     = 学習中のポリシー (モデル)                        │
│  π_ref = リファレンスポリシー (SFTモデル)                 │
│  y_w   = 人間が好んだ応答 (winner/chosen)                │
│  y_l   = 人間が好まなかった応答 (loser/rejected)         │
│  β     = KL ペナルティの強さ                              │
│  σ     = シグモイド関数                                   │
│                                                          │
│  直感:                                                    │
│  → chosen の確率を上げ、rejected の確率を下げる           │
│  → β が大きいと SFT モデルからの逸脱を制限               │
│  → β が小さいと自由に最適化 (過学習リスク)               │
│                                                          │
│  β の推奨値:                                              │
│  ├── 0.1  → 標準 (多くの場合に有効)                      │
│  ├── 0.05 → 積極的最適化 (データ品質が高い場合)          │
│  └── 0.5  → 保守的 (SFTモデルの品質維持重視)             │
└──────────────────────────────────────────────────────────┘
```

### 4.2 好みデータセットの作成方法

```python
from datasets import Dataset
import json

def create_preference_dataset(
    sft_model,
    tokenizer,
    prompts: list[str],
    n_responses: int = 4,
    temperature: float = 0.8,
) -> Dataset:
    """SFT モデルから好みデータセットを自動生成"""
    preference_data = []

    for prompt in prompts:
        # 複数の応答を生成
        responses = []
        for _ in range(n_responses):
            inputs = tokenizer(prompt, return_tensors="pt").to(sft_model.device)
            outputs = sft_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

        # LLM-as-a-Judge で評価 (GPT-4o を使用)
        from openai import OpenAI
        client = OpenAI()

        judge_prompt = f"""
以下の質問に対する{n_responses}つの回答を評価し、
最も良い回答と最も悪い回答を選んでください。

質問: {prompt}

回答:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(responses))}

JSON形式で出力: {{"best": <番号>, "worst": <番号>, "reason": "<理由>"}}
"""

        judge_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(judge_response.choices[0].message.content)

        preference_data.append({
            "prompt": prompt,
            "chosen": responses[result["best"] - 1],
            "rejected": responses[result["worst"] - 1],
        })

    return Dataset.from_list(preference_data)


# 手動でのデータ作成テンプレート
manual_preference = [
    {
        "prompt": "Pythonでリストの重複を除去する方法は？",
        "chosen": (
            "リストの重複除去にはいくつかの方法があります:\n\n"
            "1. `set()` を使う方法（最もシンプル）:\n"
            "```python\n"
            "unique = list(set(original_list))\n"
            "```\n"
            "注意: 順序が保持されません。\n\n"
            "2. `dict.fromkeys()` で順序を保持:\n"
            "```python\n"
            "unique = list(dict.fromkeys(original_list))\n"
            "```\n"
        ),
        "rejected": "set使えばいいです。",
    },
    # ... 数百件
]
```

### 4.3 ORPO (Odds Ratio Preference Optimization)

```python
from trl import ORPOTrainer, ORPOConfig

# ORPO: SFT と DPO を同時に行う手法
# → SFT ステップが不要、より効率的
orpo_config = ORPOConfig(
    output_dir="./orpo-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
    beta=0.1,              # ORPO のオッズ比パラメータ
    logging_steps=10,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
```

---

## 5. OpenAI / API 経由のファインチューニング

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

### 5.1 OpenAI ファインチューニングのベストプラクティス

```python
import json
from pathlib import Path

class OpenAIFTDataValidator:
    """OpenAI ファインチューニングデータの検証ツール"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = []
        with open(filepath) as f:
            for line in f:
                self.data.append(json.loads(line))

    def validate(self) -> dict:
        """データの品質チェック"""
        issues = []
        stats = {
            "total_examples": len(self.data),
            "total_tokens": 0,
            "avg_tokens": 0,
            "max_tokens": 0,
            "min_tokens": 0,
        }

        token_counts = []
        for i, example in enumerate(self.data):
            messages = example.get("messages", [])

            # 必須フィールドチェック
            if not messages:
                issues.append(f"行 {i}: messages が空")
                continue

            roles = [m["role"] for m in messages]

            # システムプロンプトの一貫性
            if roles[0] == "system":
                system_content = messages[0]["content"]
            else:
                issues.append(f"行 {i}: system メッセージなし")

            # assistant 応答の存在確認
            if "assistant" not in roles:
                issues.append(f"行 {i}: assistant 応答なし")

            # トークン数の概算 (1トークン ≈ 4文字)
            total_chars = sum(len(m["content"]) for m in messages)
            est_tokens = total_chars // 4
            token_counts.append(est_tokens)

        if token_counts:
            stats["total_tokens"] = sum(token_counts)
            stats["avg_tokens"] = sum(token_counts) // len(token_counts)
            stats["max_tokens"] = max(token_counts)
            stats["min_tokens"] = min(token_counts)

        # 推奨チェック
        if len(self.data) < 10:
            issues.append("WARNING: 例が10件未満。最低50件を推奨")
        elif len(self.data) < 50:
            issues.append("NOTE: 例が50件未満。100件以上を推奨")

        # コスト概算 (gpt-4o-mini の FT 料金: $3.00/1M training tokens)
        cost_per_epoch = (stats["total_tokens"] / 1_000_000) * 3.00
        stats["estimated_cost_per_epoch"] = f"${cost_per_epoch:.2f}"
        stats["estimated_cost_3_epochs"] = f"${cost_per_epoch * 3:.2f}"

        return {"stats": stats, "issues": issues}


# 使用例
validator = OpenAIFTDataValidator("training_data.jsonl")
report = validator.validate()
print("=== データ検証レポート ===")
for key, value in report["stats"].items():
    print(f"  {key}: {value}")
if report["issues"]:
    print("\n問題点:")
    for issue in report["issues"]:
        print(f"  - {issue}")
```

### 5.2 ファインチューニング済みモデルの評価

```python
from openai import OpenAI

client = OpenAI()

def compare_base_vs_ft(
    base_model: str,
    ft_model: str,
    test_prompts: list[dict],
) -> list[dict]:
    """ベースモデルとFTモデルの比較評価"""
    results = []

    for test in test_prompts:
        # ベースモデルの回答
        base_resp = client.chat.completions.create(
            model=base_model,
            messages=test["messages"],
            max_tokens=500,
            temperature=0,
        )

        # FTモデルの回答
        ft_resp = client.chat.completions.create(
            model=ft_model,
            messages=test["messages"],
            max_tokens=500,
            temperature=0,
        )

        # LLM-as-a-Judge で比較
        judge_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""以下の2つの回答を比較評価してください。

質問: {test['messages'][-1]['content']}

回答A (ベース): {base_resp.choices[0].message.content}

回答B (FT): {ft_resp.choices[0].message.content}

JSON: {{"winner": "A" | "B" | "tie", "reason": "<理由>", "score_a": 1-5, "score_b": 1-5}}"""
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )

        import json
        result = json.loads(judge_resp.choices[0].message.content)
        result["prompt"] = test["messages"][-1]["content"][:100]
        results.append(result)

    # 集計
    wins = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        wins[r["winner"]] += 1

    print(f"ベース勝利: {wins['A']}, FT勝利: {wins['B']}, 引き分け: {wins['tie']}")
    return results
```

---

## 6. モデルのマージとエクスポート

### 6.1 LoRA アダプタのマージ

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# LoRA アダプタをベースモデルにマージ
model = AutoPeftModelForCausalLM.from_pretrained(
    "./output/final",             # LoRA アダプタのパス
    torch_dtype="auto",
    device_map="auto",
)

merged_model = model.merge_and_unload()

# マージ後のモデルを保存
merged_model.save_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./output/final")
tokenizer.save_pretrained("./merged_model")

print("マージ完了。LoRA なしで推論可能になりました。")
```

### 6.2 GGUF 形式への変換 (ローカル推論用)

```bash
# llama.cpp の convert スクリプトで GGUF に変換
python llama.cpp/convert_hf_to_gguf.py \
    ./merged_model \
    --outtype bf16 \
    --outfile ./merged_model.gguf

# 量子化 (4bit)
./llama.cpp/build/bin/llama-quantize \
    ./merged_model.gguf \
    ./merged_model-q4_k_m.gguf \
    Q4_K_M

# Ollama で利用
cat > Modelfile << 'EOF'
FROM ./merged_model-q4_k_m.gguf

SYSTEM """あなたは専門的なアシスタントです。"""

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
EOF

ollama create my-finetuned -f Modelfile
ollama run my-finetuned
```

### 6.3 Hugging Face Hub へのアップロード

```python
from huggingface_hub import HfApi

api = HfApi()

# リポジトリ作成
api.create_repo("your-org/my-finetuned-model", private=True)

# アップロード
api.upload_folder(
    folder_path="./merged_model",
    repo_id="your-org/my-finetuned-model",
    commit_message="Upload fine-tuned Llama 3.1 8B",
)

# または LoRA アダプタのみアップロード (軽量)
api.upload_folder(
    folder_path="./output/final",
    repo_id="your-org/my-lora-adapter",
    commit_message="Upload LoRA adapter",
)
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と対処法

```
┌──────────────────────────────────────────────────────────┐
│       ファインチューニング トラブルシューティング             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題 1: Loss が下がらない                                │
│  原因:                                                    │
│  - 学習率が低すぎる/高すぎる                              │
│  - データ形式がモデルのチャットテンプレートと不一致         │
│  - データ品質が低い                                       │
│  対処:                                                    │
│  - 学習率を 1e-5 ~ 5e-4 の範囲で調整                    │
│  - tokenizer.apply_chat_template() を使用               │
│  - データを10件サンプリングして目視確認                    │
│                                                          │
│  問題 2: 過学習 (eval_loss 上昇)                          │
│  原因:                                                    │
│  - データ量に対してエポック数が多すぎる                    │
│  - LoRA ランクが大きすぎる                                │
│  対処:                                                    │
│  - エポック数を減らす (1-3 が一般的)                      │
│  - LoRA r を小さくする (8-16)                             │
│  - dropout を増やす (0.1-0.2)                             │
│  - データ量を増やす                                       │
│                                                          │
│  問題 3: CUDA Out of Memory                               │
│  原因: GPU メモリ不足                                      │
│  対処:                                                    │
│  - batch_size を半分にする                                 │
│  - gradient_accumulation_steps を倍にする                  │
│  - gradient_checkpointing=True にする                     │
│  - QLoRA (4bit) に切り替える                              │
│  - max_seq_length を短くする                              │
│                                                          │
│  問題 4: 生成品質が低下                                    │
│  原因:                                                    │
│  - カタストロフィック・フォゲッティング                    │
│  - 学習データのバイアス                                    │
│  対処:                                                    │
│  - 学習率を下げる                                         │
│  - LoRA r を小さくする                                    │
│  - 汎用データも混ぜる (10-20%)                            │
│  - DPO の β を大きくする (0.3-0.5)                       │
│                                                          │
│  問題 5: チャットテンプレート不一致                         │
│  原因: 学習時と推論時でテンプレートが異なる                │
│  対処:                                                    │
│  - tokenizer.apply_chat_template() を常に使用            │
│  - 特殊トークン (BOS, EOS) の処理を統一                  │
│  - Ollama 等で Modelfile のテンプレートを正確に設定       │
└──────────────────────────────────────────────────────────┘
```

### 7.2 デバッグ用コード

```python
def debug_training_data(dataset, tokenizer, n_samples: int = 5):
    """学習データのデバッグ"""
    print("=== 学習データ確認 ===")
    for i, example in enumerate(dataset.select(range(n_samples))):
        text = example.get("text", "")
        tokens = tokenizer.encode(text)

        print(f"\n--- Example {i+1} ---")
        print(f"文字数: {len(text)}")
        print(f"トークン数: {len(tokens)}")
        print(f"最初の200文字: {text[:200]}")
        print(f"最後の200文字: {text[-200:]}")

        # 特殊トークンの確認
        special_tokens = [
            t for t in tokens
            if t in tokenizer.all_special_ids
        ]
        print(f"特殊トークン数: {len(special_tokens)}")

    # 統計情報
    all_lengths = [len(tokenizer.encode(e["text"])) for e in dataset]
    print(f"\n=== 統計 ===")
    print(f"データ件数: {len(all_lengths)}")
    print(f"平均トークン数: {sum(all_lengths) / len(all_lengths):.0f}")
    print(f"最大トークン数: {max(all_lengths)}")
    print(f"最小トークン数: {min(all_lengths)}")
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

| 項目 | RLHF | DPO | ORPO |
|------|------|-----|------|
| 報酬モデル | 必要（別途学習） | 不要 | 不要 |
| SFT ステップ | 必要 | 必要 | 不要 (統合) |
| 学習安定性 | 不安定（PPOの調整困難） | 安定 | 安定 |
| 計算コスト | 高い（3モデル並行） | 中程度（2モデル） | 低い（1モデル） |
| データ要件 | 比較ペア + 報酬ラベル | 比較ペアのみ | 比較ペアのみ |
| 性能 | 高い（調整成功時） | RLHF に匹敵 | DPO に匹敵 |
| 実装難易度 | 非常に高い | 中程度 | 低い |
| 採用例 | GPT-4, Claude | Llama 3, Zephyr | Mistral v0.3 |

### 比較表 3: 学習データ規模と品質の目安

| タスク種別 | 最小データ量 | 推奨データ量 | データ品質基準 |
|-----------|------------|------------|-------------|
| テキスト分類 | 100件 | 500-2,000件 | ラベル一貫性 >95% |
| スタイル調整 | 200件 | 1,000-3,000件 | 人手検証済み |
| 知識注入 | 500件 | 2,000-10,000件 | 事実確認済み |
| コード生成 | 300件 | 1,000-5,000件 | テスト通過確認済み |
| 複雑な推論 | 1,000件 | 5,000-50,000件 | 専門家レビュー済み |
| 対話最適化 | 500件 | 2,000-10,000件 | A/B テスト検証済み |

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

### アンチパターン 3: 学習率の固定

```python
# NG: 全タスクで同じ学習率を使用
learning_rate = 2e-4  # 常にこの値

# OK: タスクとモデルサイズに応じて調整
learning_rates = {
    "sft_7b_lora":   2e-4,   # 小〜中モデルの LoRA SFT
    "sft_70b_lora":  5e-5,   # 大モデルの LoRA SFT
    "dpo_7b":        5e-7,   # DPO は低学習率
    "dpo_70b":       1e-7,   # 大モデルの DPO はさらに低く
    "openai_ft":     1.8,    # OpenAI API の multiplier
}

# ベストプラクティス: 学習率スケジュール
# 1. warmup (5-10% のステップ) で線形に上昇
# 2. cosine decay で徐々に低下
# 3. 最終学習率は初期の 10% 程度
```

### アンチパターン 4: 評価なしのデプロイ

```python
# NG: 学習完了 → 即デプロイ
trainer.train()
deploy(model)  # 品質未確認

# OK: 段階的な評価プロセス
trainer.train()

# 1. 定量評価 (自動)
eval_results = evaluate_on_test_set(model, test_dataset)
if eval_results["score"] < baseline_score:
    raise ValueError("品質がベースラインを下回っています")

# 2. 定性評価 (人手サンプル)
samples = generate_samples(model, sample_prompts, n=20)
# 人手で確認

# 3. A/B テスト
# 既存モデルとの比較を実施

# 4. 段階的ロールアウト
# 10% のトラフィックで開始 → 問題なければ拡大
```

---

## FAQ

### Q1: LoRA のランク r はいくつに設定すべきですか？

**A:** 一般的には r=8〜32 が良い出発点です。タスクが複雑なほど大きなランクが必要ですが、r=64 以上は過学習リスクが高まります。実験的に r=8, 16, 32 で比較し、検証セットの性能で決定するのがベストです。`lora_alpha` は通常 `r` の2倍に設定します。

### Q2: ファインチューニングにはどのくらいのデータが必要ですか？

**A:** タスクによりますが、一般的な目安は以下の通りです。分類タスク: 100〜500例、スタイル調整: 500〜2000例、専門知識の注入: 1000〜5000例、複雑な推論: 5000〜50000例。ただし、データの品質と多様性がデータ量より重要です。

### Q3: ファインチューニングとRAGはどちらを選ぶべきですか？

**A:** 目的に応じて選択します。「動作・スタイルの変更」にはファインチューニング、「知識の追加」にはRAGが適しています。ファインチューニングは一度学習すれば推論時のコストが変わらず、RAG は最新情報を動的に提供できます。多くの場合、両方を組み合わせるのが最適です。

### Q4: LoRA と全パラメータ FT の品質差はどの程度ですか？

**A:** 多くのタスクで LoRA (r=16-32) は全パラメータ FT の 95-99% の性能を達成します。特にタスク固有のファインチューニング (分類、要約、コード生成など) では差がほとんど見られません。ただし、モデルの知識を大幅に書き換えるような学習 (新しい言語の習得、全く新しいドメインへの適応) では全パラメータ FT が優位なことがあります。

### Q5: QLoRA で 70B モデルを学習するにはどんなハードウェアが必要ですか？

**A:** QLoRA (4bit) + LoRA (r=16) で 70B モデルを学習するには、約 40-48GB の VRAM が必要です。A100 80GB 1台、またはA100 40GB 2台 (DeepSpeed ZeRO Stage 3) で実行可能です。バッチサイズは 1-2、gradient_accumulation で実効バッチサイズを確保します。RTX 4090 (24GB) では gradient_checkpointing + batch_size=1 で辛うじて実行できる場合がありますが、安定性の面で推奨しません。

### Q6: ファインチューニング後にモデルが「壊れた」場合の対処法は？

**A:** カタストロフィック・フォゲッティングの可能性があります。対処法: (1) 学習率を下げる (1/10 程度)、(2) エポック数を減らす (1 エポックでも効果がある場合が多い)、(3) LoRA の r を小さくする、(4) 汎用データを 10-20% 混ぜる、(5) DPO の場合は beta を大きくして SFT モデルからの逸脱を制限する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| LoRA | 低ランク行列で効率的にモデルを適応、VRAM を大幅削減 |
| QLoRA | 4bit 量子化 + LoRA で 8B モデルを 1 GPU で学習可能 |
| RLHF | 報酬モデルと PPO で人間の好みに合わせる（高性能だが不安定） |
| DPO | 報酬モデル不要で直接最適化（安定・簡単） |
| ORPO | SFT + DPO を統合した効率的手法 |
| データ品質 | 量より質が重要、1000件の良質データが10万件に勝つ |
| 段階的アプローチ | プロンプト → Few-shot → RAG → FT の順で検討 |
| 評価 | FT 前後の比較評価を必ず実施、ベースラインを記録 |
| エクスポート | LoRA マージ → GGUF 変換 → Ollama 実行のパイプライン |

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
5. Hong, J. et al. (2024). "ORPO: Monolithic Preference Optimization without Reference Model." *arXiv:2403.07691*
6. Hugging Face, "PEFT Documentation." https://huggingface.co/docs/peft
7. Hugging Face, "TRL Documentation." https://huggingface.co/docs/trl
