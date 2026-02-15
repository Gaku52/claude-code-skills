# Gemini — Google DeepMind の統合マルチモーダル LLM

> Gemini は Google DeepMind が開発したマルチモーダルネイティブ LLM であり、テキスト・画像・音声・動画・コードを単一モデルで処理する次世代アーキテクチャを採用している。

## この章で学ぶこと

1. **Gemini のアーキテクチャと特徴** — Mixture of Experts、マルチモーダルネイティブ設計、長大コンテキストウィンドウの仕組み
2. **モデルラインナップと使い分け** — Ultra / Pro / Flash / Nano の性能差・コスト・適用領域
3. **Gemini API の実践的利用** — Google AI Studio・Vertex AI 経由での統合方法とベストプラクティス

---

## 1. Gemini のアーキテクチャ

### 1.1 マルチモーダルネイティブ設計

Gemini は従来の「テキストモデル + 視覚エンコーダの後付け」とは異なり、
訓練段階から複数モダリティを統合して学習するネイティブマルチモーダルアーキテクチャを採用している。

```
┌─────────────────────────────────────────────────┐
│              Gemini アーキテクチャ概要             │
├─────────────────────────────────────────────────┤
│                                                 │
│  入力モダリティ        統合エンコーダ              │
│  ┌──────────┐                                   │
│  │ テキスト  │──┐                                │
│  └──────────┘  │    ┌──────────────────┐        │
│  ┌──────────┐  ├──▶│  Unified Decoder  │──▶出力 │
│  │  画像    │──┤    │  (Transformer)    │        │
│  └──────────┘  │    └──────────────────┘        │
│  ┌──────────┐  │                                │
│  │  音声    │──┤    特徴:                        │
│  └──────────┘  │    - 各モダリティを同一          │
│  ┌──────────┐  │      潜在空間にマッピング        │
│  │  動画    │──┤    - クロスモーダル注意機構       │
│  └──────────┘  │    - 統一トークン化              │
│  ┌──────────┐  │                                │
│  │  コード  │──┘                                │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

#### マルチモーダルネイティブと後付けアプローチの違い

```
┌────────────────────────────────────────────────────────────┐
│      後付けアプローチ (GPT-4V 初期) vs ネイティブ (Gemini)   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  後付けアプローチ:                                          │
│  ┌────────┐   ┌─────────┐   ┌──────────┐                 │
│  │ 画像   │→  │ 視覚     │→  │テキスト  │→ LLM → 出力    │
│  │        │   │ エンコーダ│   │ 変換    │                  │
│  └────────┘   └─────────┘   └──────────┘                 │
│  - 視覚情報がテキスト空間に「翻訳」される                     │
│  - モダリティ間の微妙な関係性が失われやすい                   │
│  - 音声・動画への拡張が困難                                  │
│                                                            │
│  ネイティブアプローチ (Gemini):                              │
│  ┌────────┐                                                │
│  │ 画像   │──┐                                             │
│  ├────────┤  │  ┌──────────────────┐                      │
│  │ テキスト│──┼─▶│ 統合 Transformer │→ 出力               │
│  ├────────┤  │  │ (共有パラメータ)  │                      │
│  │ 音声   │──┘  └──────────────────┘                      │
│  └────────┘                                                │
│  - 全モダリティが同一の潜在空間で学習                        │
│  - クロスモーダルな推論が自然に行える                        │
│  - 新しいモダリティの追加が比較的容易                        │
└────────────────────────────────────────────────────────────┘
```

### 1.2 Mixture of Experts (MoE)

Gemini 1.5 以降は MoE アーキテクチャを採用し、推論時に一部の Expert のみを活性化することで、
パラメータ総数に対して計算コストを大幅に削減している。

```
┌───────────────────────────────────────────────┐
│           Mixture of Experts (MoE)            │
├───────────────────────────────────────────────┤
│                                               │
│  入力トークン                                  │
│      │                                        │
│      ▼                                        │
│  ┌────────┐                                   │
│  │ Router │  ← トークンごとに Expert を選択     │
│  └────────┘                                   │
│    │    │    │                                 │
│    ▼    ▼    ▼                                 │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐   │
│  │E1│ │E2│ │E3│ │E4│ │E5│ │E6│ │E7│ │E8│   │
│  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘   │
│   ★         ★                   ★            │
│   活性化     活性化              活性化         │
│                                               │
│  ★ = 選択された Expert のみ計算               │
│  総パラメータ数 >> 活性パラメータ数             │
│  → 大容量の知識 + 低い推論コスト               │
└───────────────────────────────────────────────┘
```

#### MoE の技術的詳細

MoE アーキテクチャでは、各 Transformer ブロックの FFN (Feed-Forward Network) 層を複数の Expert に置き換え、Router ネットワークがトークンごとに最適な Expert を選択する。

```python
# MoE の概念的な実装
import torch
import torch.nn as nn

class MoELayer(nn.Module):
    """Mixture of Experts レイヤーの概念実装"""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # 各 Expert は独立した FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_experts)
        ])

        # Router: 各トークンをどの Expert に送るか決定
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape

        # Router で Expert の確率分布を計算
        router_logits = self.router(x)  # [batch, seq_len, n_experts]
        router_probs = torch.softmax(router_logits, dim=-1)

        # Top-K Expert を選択
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # 選択された Expert の出力を加重合算
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [batch, seq_len]
            weight = top_k_probs[:, :, k]         # [batch, seq_len]

            for i in range(self.n_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        return output


# Gemini のスケール感
# 例: 16 Experts, Top-2 activation
# 総パラメータ: ~1T (推定)
# 活性パラメータ: ~1T / 8 ≈ 125B (推論時)
# → GPT-4o 級の性能を 1/8 の計算コストで実現
```

#### Router の負荷分散メカニズム

```
┌──────────────────────────────────────────────────────────┐
│         Router の負荷分散問題と解決策                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題: Expert Collapse (一部の Expert に偏る)              │
│                                                          │
│  負荷分散なし:                                            │
│  E1 ████████████████  ← 全トークンがここに集中            │
│  E2 ██                                                   │
│  E3 █                                                    │
│  E4 █                                                    │
│  E5 ░ (未使用)                                           │
│  E6 ░ (未使用)                                           │
│  E7 ░ (未使用)                                           │
│  E8 ░ (未使用)                                           │
│                                                          │
│  解決策: Auxiliary Loss (補助損失)                         │
│  L_total = L_task + α × L_balance                        │
│                                                          │
│  負荷分散あり:                                            │
│  E1 ████                                                 │
│  E2 ████                                                 │
│  E3 ███                                                  │
│  E4 ████                                                 │
│  E5 ███                                                  │
│  E6 ████                                                 │
│  E7 ███                                                  │
│  E8 ████                                                 │
│  → 全 Expert が均等に活用される                            │
└──────────────────────────────────────────────────────────┘
```

### 1.3 長大コンテキストウィンドウ

Gemini 1.5 Pro は最大 200 万トークンのコンテキストウィンドウを提供する。

```python
# コンテキスト長の比較
context_windows = {
    "GPT-4o":          128_000,    # 128K
    "Claude 3.5":      200_000,    # 200K
    "Gemini 1.5 Pro":  2_000_000,  # 2M (!)
    "Gemini 1.5 Flash": 1_000_000, # 1M
}

for model, tokens in context_windows.items():
    pages = tokens // 500  # 1ページ ≈ 500 トークン
    print(f"{model}: {tokens:>10,} tokens ≈ {pages:>5,} pages")
```

#### 長コンテキストを支える技術

```
┌──────────────────────────────────────────────────────────┐
│        長コンテキスト実現のための技術スタック                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Ring Attention                                       │
│     ┌───┐ ┌───┐ ┌───┐ ┌───┐                            │
│     │GPU│→│GPU│→│GPU│→│GPU│→ ...                       │
│     │ 1 │ │ 2 │ │ 3 │ │ 4 │                            │
│     └─↑─┘ └───┘ └───┘ └─│─┘                            │
│       └──────────────────┘ (リング構造)                   │
│     → 各 GPU がコンテキストの一部を担当                    │
│     → Attention を GPU 間でリレー                         │
│                                                          │
│  2. Grouped-Query Attention (GQA)                        │
│     - Key/Value ヘッドを共有して KV キャッシュ削減          │
│     - 標準 MHA の 1/8 のメモリで同等性能                   │
│                                                          │
│  3. RoPE 位置エンコーディングの拡張                        │
│     - NTK-aware interpolation                            │
│     - YaRN (Yet another RoPE extensioN)                  │
│     → 短い学習コンテキストから長い推論コンテキストへ外挿    │
│                                                          │
│  4. KV Cache の効率化                                     │
│     - Sliding Window Attention                           │
│     - KV Cache の圧縮・量子化                             │
│     → 2M トークンでも実用的なメモリ消費                   │
└──────────────────────────────────────────────────────────┘
```

#### "Lost in the Middle" 問題と対策

```python
# 長コンテキストでの精度劣化パターンと対策

# 問題: 200万トークンのうち中間部分の情報検索精度が低下する
# (先頭と末尾の情報は記憶しやすいが、中間は記憶しにくい)

# 対策 1: 重要な情報を先頭・末尾に配置
def arrange_context_optimally(documents: list[str], query: str) -> str:
    """重要度順にソートし、先頭と末尾に重要な文書を配置"""
    # 関連度スコアで文書をランク付け
    ranked = rank_by_relevance(documents, query)

    # 交互配置: 1位→先頭, 2位→末尾, 3位→先頭寄り, ...
    arranged = []
    for i, doc in enumerate(ranked):
        if i % 2 == 0:
            arranged.insert(len(arranged) // 2, doc)
        else:
            arranged.append(doc)

    return "\n\n".join(arranged)


# 対策 2: 階層的な要約 (Map-Reduce パターン)
def hierarchical_analysis(documents: list[str], query: str) -> str:
    """大量文書を階層的に処理"""
    import google.generativeai as genai

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Step 1: 各文書を個別に要約 (Map)
    summaries = []
    for doc in documents:
        summary = model.generate_content(
            f"以下の文書を{query}の観点から要約してください:\n\n{doc}"
        )
        summaries.append(summary.text)

    # Step 2: 要約を統合 (Reduce)
    pro_model = genai.GenerativeModel("gemini-1.5-pro")
    final = pro_model.generate_content(
        f"以下の要約群を統合し、{query}に対する包括的な回答を作成してください:\n\n"
        + "\n---\n".join(summaries)
    )
    return final.text


# 対策 3: チャンク分割 + 選択的投入
def selective_context(
    all_docs: list[str],
    query: str,
    max_tokens: int = 500_000,
) -> list[str]:
    """関連文書のみ選択してコンテキストに投入"""
    # Embedding ベースで関連文書をフィルタリング
    relevant = retrieve_relevant(query, all_docs, top_k=50)

    # トークン数を推定してカット
    selected = []
    total_tokens = 0
    for doc in relevant:
        doc_tokens = estimate_tokens(doc)
        if total_tokens + doc_tokens > max_tokens:
            break
        selected.append(doc)
        total_tokens += doc_tokens

    return selected
```

#### コンテキスト長別のユースケース

```
┌──────────────────────────────────────────────────────────┐
│        コンテキスト長別の実用的なユースケース                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ~32K tokens (通常のモデル)                                │
│  ├── チャットボット対話                                    │
│  ├── 短い文書の要約                                       │
│  └── コード生成 (単一ファイル)                              │
│                                                          │
│  ~128K tokens (GPT-4o, Llama 3.1)                         │
│  ├── 論文 1-3 本の分析                                    │
│  ├── 中規模コードベースのレビュー                          │
│  └── 長い会議議事録の処理                                  │
│                                                          │
│  ~200K tokens (Claude 3.5)                                │
│  ├── 書籍 1 冊の要約・分析                                │
│  ├── 大規模コードベースの理解                              │
│  └── 法律文書の包括的レビュー                              │
│                                                          │
│  ~1M tokens (Gemini 1.5 Flash)                            │
│  ├── 複数書籍の横断分析                                    │
│  ├── 動画 30 分の全文字起こし分析                          │
│  └── 大規模データセットの直接入力                           │
│                                                          │
│  ~2M tokens (Gemini 1.5 Pro)                              │
│  ├── 書籍 3-5 冊の一括処理                                │
│  ├── 1 時間の動画分析                                     │
│  ├── 大規模ソフトウェアリポジトリの全体理解                 │
│  └── 年次報告書 10 年分の時系列分析                        │
└──────────────────────────────────────────────────────────┘
```

---

## 2. モデルラインナップ

### 2.1 モデル比較表

| モデル | パラメータ規模 | コンテキスト | 主な用途 | 料金目安 (入力/1M tokens) |
|--------|-------------|------------|---------|------------------------|
| Gemini Ultra | 最大級 | 128K | 最高精度タスク | 高価格帯 |
| Gemini 1.5 Pro | 大規模 MoE | 2M | 汎用・長文書解析 | $1.25 - $5.00 |
| Gemini 1.5 Flash | 中規模 MoE | 1M | 高速・低コスト | $0.075 - $0.30 |
| Gemini 2.0 Flash | 次世代 MoE | 1M | 最新・高速 | $0.10 - $0.40 |
| Gemini Nano | 小規模 | 32K | オンデバイス | 無料 (端末内) |

### 2.2 ユースケース別選定ガイド

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| 大規模コードベース解析 | 1.5 Pro (2M) | 長大コンテキストが必須 |
| チャットボット | 2.0 Flash | 低レイテンシ・低コスト |
| 画像+テキスト理解 | 1.5 Pro / 2.0 Flash | マルチモーダル精度 |
| 端末内 AI アシスタント | Nano | オフライン動作 |
| 研究・最高精度 | Ultra | ベンチマークトップ |
| リアルタイム翻訳 | Flash | 速度重視 |

### 2.3 モデル選定の詳細フレームワーク

```
┌──────────────────────────────────────────────────────────┐
│         Gemini モデル選定フローチャート                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  START                                                   │
│    │                                                     │
│    ├─ オフライン動作が必要？                                │
│    │   YES → Gemini Nano                                  │
│    │                                                     │
│    NO ↓                                                  │
│    ├─ 128K 以上のコンテキストが必要？                       │
│    │   YES ─┬─ 2M 必要？ → 1.5 Pro                       │
│    │        └─ 1M で十分 → 1.5 Flash (コスト重視)         │
│    │                    → 1.5 Pro (精度重視)              │
│    │                                                     │
│    NO ↓                                                  │
│    ├─ 最低レイテンシが必要？                                │
│    │   YES → 2.0 Flash                                    │
│    │                                                     │
│    NO ↓                                                  │
│    ├─ コスト最小化が最優先？                                │
│    │   YES → 1.5 Flash ($0.075/1M)                        │
│    │                                                     │
│    NO ↓                                                  │
│    └─ 最高品質が必要 → 1.5 Pro / Ultra                    │
└──────────────────────────────────────────────────────────┘
```

### 2.4 料金の詳細と最適化

```python
# Gemini の料金体系の詳細
gemini_pricing = {
    "gemini-1.5-pro": {
        "input_128k_below": 1.25,   # $/1M tokens (128K以下)
        "input_128k_above": 2.50,   # $/1M tokens (128K超)
        "output_128k_below": 5.00,
        "output_128k_above": 10.00,
        "context_caching_storage": 4.50,  # $/1M tokens/hour
        "context_caching_input": 0.3125,  # 75% 割引
    },
    "gemini-1.5-flash": {
        "input_128k_below": 0.075,
        "input_128k_above": 0.15,
        "output_128k_below": 0.30,
        "output_128k_above": 0.60,
        "context_caching_storage": 1.00,
        "context_caching_input": 0.01875,
    },
    "gemini-2.0-flash": {
        "input": 0.10,
        "output": 0.40,
    },
}


def calculate_gemini_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    use_caching: bool = False,
    cached_tokens: int = 0,
    cache_duration_hours: float = 1.0,
) -> dict:
    """Gemini のコストを詳細に計算"""
    pricing = gemini_pricing[model]

    if model == "gemini-2.0-flash":
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total": input_cost + output_cost,
        }

    # 128K 境界での料金切り替え
    threshold = 128_000
    if input_tokens <= threshold:
        input_rate = pricing["input_128k_below"]
    else:
        input_rate = pricing["input_128k_above"]

    output_rate = pricing["output_128k_below"] if input_tokens <= threshold \
        else pricing["output_128k_above"]

    if use_caching and cached_tokens > 0:
        # キャッシュ利用時の割引料金
        cache_input_cost = (cached_tokens / 1_000_000) * pricing["context_caching_input"]
        regular_input_cost = ((input_tokens - cached_tokens) / 1_000_000) * input_rate
        cache_storage_cost = (cached_tokens / 1_000_000) * pricing["context_caching_storage"] * cache_duration_hours
        input_cost = cache_input_cost + regular_input_cost + cache_storage_cost
    else:
        input_cost = (input_tokens / 1_000_000) * input_rate

    output_cost = (output_tokens / 1_000_000) * output_rate

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total": input_cost + output_cost,
    }


# 使用例: 1M トークンのドキュメント処理
result = calculate_gemini_cost(
    model="gemini-1.5-pro",
    input_tokens=1_000_000,
    output_tokens=2_000,
)
print(f"1M tokens処理コスト: ${result['total']:.2f}")

# キャッシュ利用時 (同じドキュメントに複数クエリ)
result_cached = calculate_gemini_cost(
    model="gemini-1.5-pro",
    input_tokens=1_000_000,
    output_tokens=2_000,
    use_caching=True,
    cached_tokens=950_000,
    cache_duration_hours=1.0,
)
print(f"キャッシュ利用時コスト: ${result_cached['total']:.2f}")
```

---

## 3. Gemini API の実践

### 3.1 Google AI Studio (Python SDK)

```python
import google.generativeai as genai

# API キー設定
genai.configure(api_key="YOUR_API_KEY")

# モデル初期化
model = genai.GenerativeModel("gemini-1.5-pro")

# テキスト生成
response = model.generate_content("量子コンピュータを小学生に説明して")
print(response.text)
```

### 3.2 マルチモーダル入力 (画像 + テキスト)

```python
import google.generativeai as genai
from pathlib import Path

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

# 画像読み込み
image = genai.upload_file(Path("architecture_diagram.png"))

# 画像 + テキストのプロンプト
response = model.generate_content([
    image,
    "この図のアーキテクチャを解説し、改善点を3つ提案してください。"
])
print(response.text)
```

### 3.3 ストリーミング応答

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# ストリーミングで生成
response = model.generate_content(
    "Pythonの非同期処理について詳しく解説してください。",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

### 3.4 Vertex AI 経由 (エンタープライズ)

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Vertex AI 初期化
vertexai.init(project="my-project", location="us-central1")

model = GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    "当社の売上データを分析し、来期の予測を作成してください。",
    generation_config={
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
)
print(response.text)
```

### 3.5 システム指示と安全性設定

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction="あなたは金融の専門家です。正確で慎重な回答を心がけてください。",
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
    }
)

# チャット形式
chat = model.start_chat()
response = chat.send_message("日経平均の今後の見通しについて教えてください")
print(response.text)

# 会話履歴の参照
for message in chat.history:
    print(f"{message.role}: {message.parts[0].text[:50]}...")
```

### 3.6 Function Calling (ツール使用)

```python
import google.generativeai as genai
import json

genai.configure(api_key="YOUR_API_KEY")

# ツール定義
get_weather = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_weather",
            description="指定された都市の現在の天気を取得する",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "city": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="都市名 (例: 東京, 大阪)",
                    ),
                    "unit": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["celsius", "fahrenheit"],
                        description="温度の単位",
                    ),
                },
                required=["city"],
            ),
        ),
    ]
)

model = genai.GenerativeModel(
    "gemini-1.5-pro",
    tools=[get_weather],
)

chat = model.start_chat()
response = chat.send_message("東京の天気を教えてください")

# Function Call の処理
for part in response.parts:
    if fn := part.function_call:
        print(f"関数呼び出し: {fn.name}")
        print(f"引数: {dict(fn.args)}")

        # 実際の関数を呼び出し (ここではモック)
        weather_data = {"temperature": 22, "condition": "晴れ", "humidity": 55}

        # 結果をモデルに返す
        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn.name,
                        response={"result": weather_data},
                    )
                )]
            )
        )
        print(response.text)
```

### 3.7 Context Caching (コンテキストキャッシュ)

```python
import google.generativeai as genai
from google.generativeai import caching
import datetime

genai.configure(api_key="YOUR_API_KEY")

# 大きなドキュメントをキャッシュに格納
with open("large_document.txt", "r") as f:
    document_content = f.read()

# キャッシュの作成 (最低 32,768 トークン以上が必要)
cache = caching.CachedContent.create(
    model="models/gemini-1.5-pro-002",
    display_name="my-document-cache",
    system_instruction="あなたは文書分析の専門家です。",
    contents=[document_content],
    ttl=datetime.timedelta(hours=2),  # 2時間有効
)

print(f"Cache ID: {cache.name}")
print(f"Token Count: {cache.usage_metadata.total_token_count}")

# キャッシュを使ったモデル生成 (75% 割引の入力トークン料金)
model = genai.GenerativeModel.from_cached_content(cache)

# 同じドキュメントに対して複数の質問をコスト効率良く実行
questions = [
    "この文書の主要な論点を3つ挙げてください",
    "著者の結論は何ですか？",
    "第3章の要約を作成してください",
    "この文書の矛盾点や弱点を指摘してください",
]

for q in questions:
    response = model.generate_content(q)
    print(f"\nQ: {q}")
    print(f"A: {response.text[:200]}...")

# キャッシュの削除
cache.delete()
```

### 3.8 Grounding (検索連携)

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding

vertexai.init(project="my-project", location="us-central1")

model = GenerativeModel("gemini-1.5-pro")

# Google Search Grounding を使った回答
tool = Tool.from_google_search_retrieval(
    grounding.GoogleSearchRetrieval()
)

response = model.generate_content(
    "2025年のAI規制に関する最新の動向を教えてください",
    tools=[tool],
    generation_config={"temperature": 0.1},
)

print(response.text)

# Grounding のメタデータ (ソース情報)
if hasattr(response, 'candidates'):
    for candidate in response.candidates:
        if hasattr(candidate, 'grounding_metadata'):
            metadata = candidate.grounding_metadata
            print("\n検索ソース:")
            for chunk in metadata.grounding_chunks:
                print(f"  - {chunk.web.title}: {chunk.web.uri}")
```

### 3.9 動画分析の実践

```python
import google.generativeai as genai
import time

genai.configure(api_key="YOUR_API_KEY")

# 動画のアップロード
video_file = genai.upload_file("meeting_recording.mp4")

# 処理完了待ち
while video_file.state.name == "PROCESSING":
    print("動画を処理中...")
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError("動画のアップロードに失敗しました")

print(f"動画準備完了: {video_file.uri}")

model = genai.GenerativeModel("gemini-1.5-pro")

# タイムスタンプ付きの要約
response = model.generate_content([
    video_file,
    """この会議動画を分析してください:
    1. 参加者の一覧と発言頻度
    2. 議題ごとのタイムスタンプ (MM:SS - トピック)
    3. 各議題の結論と次のアクション
    4. 未解決の課題
    日本語で回答してください。"""
])
print(response.text)

# 特定の時間帯に関する質問
response = model.generate_content([
    video_file,
    "動画の5:00-10:00の区間で議論された技術的な課題を詳しく説明してください。"
])
print(response.text)
```

### 3.10 バッチ処理と並列実行

```python
import google.generativeai as genai
import asyncio
from typing import Optional

genai.configure(api_key="YOUR_API_KEY")


async def batch_generate(
    prompts: list[str],
    model_name: str = "gemini-1.5-flash",
    max_concurrent: int = 5,
    temperature: float = 0.7,
) -> list[dict]:
    """Gemini API でバッチ処理を実行"""
    model = genai.GenerativeModel(model_name)
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def process_single(prompt: str, index: int) -> dict:
        async with semaphore:
            try:
                response = await model.generate_content_async(
                    prompt,
                    generation_config={"temperature": temperature},
                )
                return {
                    "index": index,
                    "prompt": prompt[:100],
                    "response": response.text,
                    "status": "success",
                }
            except Exception as e:
                return {
                    "index": index,
                    "prompt": prompt[:100],
                    "error": str(e),
                    "status": "error",
                }

    tasks = [process_single(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda x: x["index"])


# 使用例
prompts = [
    f"質問{i}: AIの応用分野{i}について100字で説明してください"
    for i in range(20)
]

results = asyncio.run(batch_generate(prompts, max_concurrent=10))
for r in results:
    if r["status"] == "success":
        print(f"[{r['index']}] {r['response'][:80]}...")
    else:
        print(f"[{r['index']}] ERROR: {r['error']}")
```

---

## 4. Gemini の技術的差別化要因

```
┌──────────────────────────────────────────────────────┐
│           Gemini の技術的差別化ポイント                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. 超長コンテキスト                                   │
│     └─ 200万トークン → 書籍1冊丸ごと処理可能           │
│                                                      │
│  2. ネイティブマルチモーダル                            │
│     └─ テキスト/画像/音声/動画を統一的に理解            │
│                                                      │
│  3. Google エコシステム統合                             │
│     └─ Search Grounding: Google 検索結果で回答補強     │
│     └─ Workspace 統合: Gmail, Docs, Sheets 連携       │
│                                                      │
│  4. Nano モデル (オンデバイス)                          │
│     └─ Pixel / Android 端末で直接実行                  │
│     └─ プライバシー保護 + オフライン動作                │
│                                                      │
│  5. コード生成特化                                     │
│     └─ AlphaCode 2 の知見を統合                        │
│     └─ 競技プログラミングレベルの推論能力               │
└──────────────────────────────────────────────────────┘
```

### 4.1 Google エコシステム統合の詳細

```
┌──────────────────────────────────────────────────────────┐
│         Google エコシステムとの統合ポイント                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │ Google Search │     │  Google Maps  │                  │
│  │  Grounding   │     │  Places API   │                  │
│  └──────┬───────┘     └──────┬───────┘                  │
│         │                     │                          │
│         ▼                     ▼                          │
│  ┌────────────────────────────────────┐                  │
│  │           Gemini モデル             │                  │
│  └────────────────────────────────────┘                  │
│         ▲                     ▲                          │
│         │                     │                          │
│  ┌──────┴───────┐     ┌──────┴───────┐                  │
│  │  Workspace    │     │  Firebase    │                  │
│  │  (Docs/Sheets │     │  (Firestore  │                  │
│  │   /Gmail)     │     │   /Auth)     │                  │
│  └──────────────┘     └──────────────┘                  │
│                                                          │
│  活用例:                                                  │
│  1. Gmail で受信メールの自動要約・返信案の作成             │
│  2. Google Docs で文書の自動校正・翻訳                    │
│  3. Google Sheets でデータ分析・グラフ生成                │
│  4. Google Search で最新情報に基づく回答                   │
│  5. Google Maps で位置情報連携サービス                     │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Gemini Nano: オンデバイス AI

```python
# Gemini Nano は Android の AICore API 経由で利用
# (Java/Kotlin の例)

"""
// Android での Gemini Nano 利用 (Kotlin)
import com.google.ai.client.generativeai.GenerativeModel

val generativeModel = GenerativeModel(
    modelName = "gemini-nano",
    // APIキー不要 - デバイス上で直接実行
)

// テキスト生成 (デバイス上で完結)
val response = generativeModel.generateContent("こんにちは")
println(response.text)

// Gemini Nano の利点:
// - ネットワーク不要 (飛行機モードでも動作)
// - データがデバイス外に出ない (完全なプライバシー)
// - レイテンシが極めて低い (ネットワーク遅延なし)
// - API 料金が発生しない

// 対応機能:
// - テキスト要約
// - スマートリプライ
// - 文法チェック
// - テキスト分類
"""

# Python からの Gemini Nano 利用 (Chrome の組み込み AI)
# Web 版の Gemini Nano は Chrome Canary で実験的に利用可能
"""
// JavaScript (Chrome Built-in AI)
const session = await ai.languageModel.create({
  systemPrompt: "あなたは親切なアシスタントです。"
});

const result = await session.prompt("明日の予定をリマインドして");
console.log(result);

// ストリーミング
const stream = session.promptStreaming("長い文章を生成してください");
for await (const chunk of stream) {
  console.log(chunk);
}
"""
```

---

## 5. Gemini 2.0 の新機能

### 5.1 Gemini 2.0 Flash の進化点

```
┌──────────────────────────────────────────────────────────┐
│        Gemini 2.0 Flash の新機能一覧                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. ネイティブ画像生成                                     │
│     └─ テキスト→画像生成がモデル内で完結                   │
│     └─ 画像+テキストの混合出力が可能                      │
│                                                          │
│  2. ネイティブ音声生成 (TTS)                               │
│     └─ テキスト→音声の直接生成                            │
│     └─ 多言語・多スタイル対応                              │
│                                                          │
│  3. ツール使用の強化                                       │
│     └─ Google Search Grounding                           │
│     └─ コード実行 (サーバーサイド)                         │
│     └─ Function Calling の精度向上                        │
│                                                          │
│  4. Thinking Mode (思考モード)                             │
│     └─ 段階的推論 (Chain-of-Thought)                     │
│     └─ 数学・コーディングタスクの精度大幅向上              │
│                                                          │
│  5. パフォーマンス改善                                     │
│     └─ 1.5 Flash 比で精度向上 + レイテンシ改善            │
│     └─ コスト効率のさらなる改善                            │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Thinking Mode の実装

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# Thinking Mode を使った高精度推論
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")

response = model.generate_content(
    """
    以下の数学問題を段階的に解いてください:

    ある工場で、製品Aと製品Bを生産しています。
    - 製品Aは1個あたり3時間の加工と2時間の検査が必要
    - 製品Bは1個あたり2時間の加工と4時間の検査が必要
    - 1日の加工可能時間は18時間、検査可能時間は20時間
    - 製品Aの利益は500円、製品Bの利益は400円

    利益を最大化するための最適な生産量を求めてください。
    """
)

# Thinking Mode では思考過程と最終回答が分かれる
for part in response.parts:
    print(part.text)
```

---

## 6. トラブルシューティング

### 6.1 よくあるエラーと解決策

```python
import google.generativeai as genai
from google.api_core import exceptions

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")


# エラー 1: レート制限
def handle_rate_limit(prompt: str, max_retries: int = 5):
    """指数バックオフでレート制限に対応"""
    import time
    import random

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"レート制限。{wait_time:.1f}秒待機... (試行 {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise


# エラー 2: Safety フィルタによるブロック
def handle_safety_block(prompt: str):
    """Safety フィルタの診断と対応"""
    try:
        response = model.generate_content(prompt)

        # Safety ratings の確認
        if response.prompt_feedback.block_reason:
            print(f"ブロック理由: {response.prompt_feedback.block_reason}")
            for rating in response.prompt_feedback.safety_ratings:
                print(f"  {rating.category}: {rating.probability}")
            return None

        # 候補の Safety チェック
        for candidate in response.candidates:
            if candidate.finish_reason.name == "SAFETY":
                print("回答が Safety フィルタによりブロックされました")
                for rating in candidate.safety_ratings:
                    print(f"  {rating.category}: {rating.probability}")
                return None

        return response.text

    except exceptions.InvalidArgument as e:
        print(f"不正なリクエスト: {e}")
        return None


# エラー 3: トークン数超過
def check_token_count(content: str, model_name: str = "gemini-1.5-pro"):
    """事前にトークン数をチェック"""
    model = genai.GenerativeModel(model_name)

    token_count = model.count_tokens(content)
    print(f"トークン数: {token_count.total_tokens}")

    limits = {
        "gemini-1.5-pro": 2_000_000,
        "gemini-1.5-flash": 1_000_000,
        "gemini-2.0-flash": 1_000_000,
    }

    limit = limits.get(model_name, 128_000)
    if token_count.total_tokens > limit:
        print(f"警告: トークン数が上限 ({limit:,}) を超えています")
        return False
    else:
        remaining = limit - token_count.total_tokens
        print(f"残りトークン: {remaining:,}")
        return True


# エラー 4: ファイルアップロードの失敗
def safe_upload_file(file_path: str, max_retries: int = 3):
    """ファイルアップロードの堅牢な実装"""
    import os
    import time

    # ファイルサイズチェック (最大 2GB)
    file_size = os.path.getsize(file_path)
    if file_size > 2 * 1024 * 1024 * 1024:
        raise ValueError(f"ファイルサイズが上限を超えています: {file_size / (1024**3):.1f}GB")

    for attempt in range(max_retries):
        try:
            uploaded = genai.upload_file(file_path)

            # 動画の場合は処理完了を待機
            while uploaded.state.name == "PROCESSING":
                print(f"処理中... ({attempt+1}回目)")
                time.sleep(10)
                uploaded = genai.get_file(uploaded.name)

            if uploaded.state.name == "ACTIVE":
                return uploaded
            else:
                print(f"アップロード失敗: state={uploaded.state.name}")

        except Exception as e:
            print(f"アップロードエラー (試行 {attempt+1}): {e}")
            time.sleep(5)

    raise RuntimeError(f"ファイルアップロードが{max_retries}回失敗しました")
```

### 6.2 デバッグとモニタリング

```python
import google.generativeai as genai
import json
import time
from datetime import datetime

genai.configure(api_key="YOUR_API_KEY")


class GeminiMonitor:
    """Gemini API の利用状況モニタリング"""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.call_log = []

    def generate(self, prompt: str, **kwargs) -> str:
        """モニタリング付きの生成"""
        start_time = time.time()

        try:
            # トークン数の事前チェック
            token_count = self.model.count_tokens(prompt)
            input_tokens = token_count.total_tokens

            response = self.model.generate_content(prompt, **kwargs)
            latency = time.time() - start_time

            # 出力トークン数の取得
            output_tokens = self.model.count_tokens(response.text).total_tokens

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": round(latency, 3),
                "status": "success",
                "finish_reason": response.candidates[0].finish_reason.name,
            }
            self.call_log.append(log_entry)

            return response.text

        except Exception as e:
            latency = time.time() - start_time
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "latency_seconds": round(latency, 3),
                "status": "error",
                "error": str(e),
            }
            self.call_log.append(log_entry)
            raise

    def get_statistics(self) -> dict:
        """利用統計を取得"""
        if not self.call_log:
            return {"message": "No calls recorded"}

        successful = [l for l in self.call_log if l["status"] == "success"]
        failed = [l for l in self.call_log if l["status"] == "error"]

        total_input = sum(l.get("input_tokens", 0) for l in successful)
        total_output = sum(l.get("output_tokens", 0) for l in successful)
        avg_latency = sum(l["latency_seconds"] for l in successful) / len(successful) if successful else 0

        return {
            "total_calls": len(self.call_log),
            "successful": len(successful),
            "failed": len(failed),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_latency_seconds": round(avg_latency, 3),
            "error_rate": round(len(failed) / len(self.call_log) * 100, 1),
        }


# 使用例
monitor = GeminiMonitor("gemini-1.5-flash")

prompts = [
    "AIの未来について100字で述べてください",
    "Pythonのデコレータを説明してください",
    "クイックソートのアルゴリズムを説明してください",
]

for p in prompts:
    result = monitor.generate(p)
    print(f"Response: {result[:80]}...\n")

stats = monitor.get_statistics()
print(f"\n=== 統計 ===")
print(json.dumps(stats, indent=2, ensure_ascii=False))
```

---

## 7. パフォーマンス最適化

### 7.1 レイテンシ最適化

```python
import google.generativeai as genai
import time

genai.configure(api_key="YOUR_API_KEY")


# 最適化 1: ストリーミングで TTFB を最小化
def optimized_streaming(prompt: str):
    """ストリーミングによるレイテンシ最適化"""
    model = genai.GenerativeModel("gemini-1.5-flash")

    start = time.time()
    first_token_time = None

    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if first_token_time is None:
            first_token_time = time.time()
            ttfb = first_token_time - start
            print(f"TTFB: {ttfb:.3f}s")
        print(chunk.text, end="", flush=True)

    total_time = time.time() - start
    print(f"\n総時間: {total_time:.3f}s")


# 最適化 2: 適切なモデル選択
# Flash は Pro の 3-5 倍高速
performance_comparison = {
    "gemini-1.5-pro": {
        "avg_ttfb": "1.5-3.0s",
        "tokens_per_second": "30-50",
        "best_for": "高精度タスク",
    },
    "gemini-1.5-flash": {
        "avg_ttfb": "0.3-0.8s",
        "tokens_per_second": "80-150",
        "best_for": "リアルタイム応答",
    },
    "gemini-2.0-flash": {
        "avg_ttfb": "0.2-0.6s",
        "tokens_per_second": "100-200",
        "best_for": "最新・最速",
    },
}


# 最適化 3: プロンプト最適化
def optimize_prompt(prompt: str) -> str:
    """トークン数を削減してコスト・速度を改善"""
    # 不要な空白の除去
    import re
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)
    prompt = re.sub(r' {2,}', ' ', prompt)

    # 冗長な表現の除去
    replacements = {
        "できるだけ詳しく説明してください": "詳しく説明してください",
        "以下の内容について": "",
        "よろしくお願いします": "",
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)

    return prompt.strip()


# 最適化 4: Context Caching で繰り返しコストを削減
# (前述の 3.7 セクション参照)
```

### 7.2 コスト最適化戦略

```
┌──────────────────────────────────────────────────────────┐
│         Gemini コスト最適化の 5 つの戦略                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. モデルの使い分け (Routing)                              │
│     簡単なタスク → Flash ($0.075/1M)                       │
│     複雑なタスク → Pro ($1.25/1M)                          │
│     → 平均コスト 60-80% 削減                               │
│                                                          │
│  2. Context Caching                                       │
│     同じドキュメントに複数クエリ → 75% 割引                 │
│     条件: 32K tokens 以上のキャッシュ対象                   │
│                                                          │
│  3. 出力トークンの制限                                     │
│     max_output_tokens を適切に設定                         │
│     出力は入力の 4 倍高い → 出力削減が最も効果的            │
│                                                          │
│  4. バッチ処理                                             │
│     リアルタイム不要 → バッチ API で最大 50% 割引           │
│                                                          │
│  5. プロンプト最適化                                       │
│     冗長なプロンプトを簡潔に                               │
│     Few-shot 例を必要最小限に                              │
│     システム指示を Context Caching に格納                   │
└──────────────────────────────────────────────────────────┘
```

---

## 8. アンチパターン

### アンチパターン 1: コンテキストウィンドウの浪費

```python
# NG: 200万トークンあるからと全文書を無制限に投入
all_docs = load_all_company_documents()  # 300万トークン分
response = model.generate_content(all_docs + [query])
# → コンテキスト超過エラー or 精度低下 (中間部分の "Lost in the Middle" 問題)

# OK: 関連文書のみ抽出してから投入
relevant_docs = retrieve_relevant(query, all_docs, top_k=20)
response = model.generate_content(relevant_docs + [query])
# → 必要な情報に絞ることで精度向上 + コスト削減
```

### アンチパターン 2: モデル選定ミスによるコスト爆発

```python
# NG: 簡単な分類タスクに最高性能モデルを使用
model = genai.GenerativeModel("gemini-ultra")
for item in million_items:  # 100万件処理
    result = model.generate_content(f"カテゴリ分類: {item}")
# → 膨大なコスト発生

# OK: タスク難易度に応じたモデル選択
model = genai.GenerativeModel("gemini-1.5-flash")  # 分類タスクには Flash で十分
# さらに: バッチ API を使ってコスト削減
```

### アンチパターン 3: Safety フィルタの過剰緩和

```python
# NG: 安全性設定を全て無効化
safety = {cat: "BLOCK_NONE" for cat in all_categories}
# → 有害コンテンツ生成リスク、利用規約違反の可能性

# OK: ユースケースに応じた適切な設定
safety = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",  # 医療用途など必要に応じて緩和
}
```

### アンチパターン 4: エラーハンドリングの欠如

```python
# NG: エラーハンドリングなしで本番運用
response = model.generate_content(prompt)
result = response.text  # Noneの場合にクラッシュ

# OK: 堅牢なエラーハンドリング
try:
    response = model.generate_content(prompt)

    # Safety ブロックの確認
    if not response.candidates:
        print("回答が生成されませんでした (Safety フィルタ)")
        result = fallback_response()
    elif response.candidates[0].finish_reason.name != "STOP":
        print(f"異常終了: {response.candidates[0].finish_reason.name}")
        result = fallback_response()
    else:
        result = response.text

except exceptions.ResourceExhausted:
    # レート制限 → リトライ
    result = retry_with_backoff(prompt)
except exceptions.InvalidArgument as e:
    # 不正なリクエスト → ログ記録
    log_error(e, prompt)
    result = fallback_response()
except Exception as e:
    # その他のエラー
    log_error(e, prompt)
    result = fallback_response()
```

### アンチパターン 5: Context Caching の誤用

```python
# NG: 小さなコンテンツをキャッシュ (最低 32K tokens 必要)
cache = caching.CachedContent.create(
    model="models/gemini-1.5-pro-002",
    contents=["短いテキスト"],  # 100 tokens しかない
    ttl=datetime.timedelta(hours=24),
)
# → エラー: トークン数不足

# NG: TTL を長すぎに設定 (ストレージコストが発生)
cache = caching.CachedContent.create(
    model="models/gemini-1.5-pro-002",
    contents=[huge_document],
    ttl=datetime.timedelta(days=30),  # 30日 → 膨大なストレージ費用
)

# OK: 適切なサイズとTTLの設定
cache = caching.CachedContent.create(
    model="models/gemini-1.5-pro-002",
    contents=[document_over_32k_tokens],
    ttl=datetime.timedelta(hours=2),  # 必要な時間だけ
)
# クエリ完了後に明示的に削除
cache.delete()
```

---

## 9. 他モデルとの比較と使い分け

### 9.1 Gemini vs GPT-4o vs Claude 3.5 比較

```
┌──────────────────────────────────────────────────────────┐
│         三大モデルの得意分野マッピング                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│              GPT-4o                                       │
│           ┌──────────┐                                   │
│           │ マルチ   │                                   │
│           │ モーダル │                                   │
│           │ (画像生成)│                                   │
│     ┌─────┤         ├──────┐                             │
│     │     │ Function│      │                             │
│     │     │ Calling │      │                             │
│     │     └────┬────┘      │                             │
│  Gemini       │        Claude 3.5                        │
│  ┌──────────┐ │ ┌──────────┐                             │
│  │ 長コンテ │ │ │ コード   │                             │
│  │ キスト   │ │ │ 品質     │                             │
│  │ (2M)     │ │ │ (SWE-   │                             │
│  │ 動画入力 │ │ │ bench)   │                             │
│  │ Search   │ │ │ 安全性   │                             │
│  │ Grounding│ │ │ 200K ctx │                             │
│  │ 低コスト │ │ │ 指示追従 │                             │
│  └──────────┘ │ └──────────┘                             │
│               │                                          │
│          共通の強み:                                      │
│          テキスト生成、推論、JSON出力                      │
└──────────────────────────────────────────────────────────┘
```

### 9.2 ユースケース別の具体的な推奨

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| 100ページ超の法律文書分析 | Gemini 1.5 Pro | 2M コンテキスト |
| 動画の要約・分析 | Gemini 1.5 Pro | 唯一のネイティブ動画対応 |
| コードレビュー・生成 | Claude 3.5 Sonnet | SWE-bench 最高スコア |
| 大量メール自動分類 | Gemini 1.5 Flash | 最低コスト + 十分な品質 |
| 画像付きレポート生成 | GPT-4o | 画像生成 + テキスト統合 |
| 社内データ処理 | Gemini (Vertex AI) | Google Cloud 統合 |
| リアルタイムチャット | Gemini 2.0 Flash | 最低レイテンシ |
| 数学的推論 | DeepSeek-R1 / o1 | 推論特化 |

---

## 10. FAQ

### Q1: Gemini と GPT-4o はどう使い分けるべきか?

長大な文書処理 (論文集、コードベース全体など) は Gemini の 200 万トークンコンテキストが圧倒的に有利。
一方、既存ツールチェーンとの統合性や Function Calling のエコシステム成熟度では GPT-4o に分がある。
マルチモーダルタスクでは両者とも高性能だが、動画入力は Gemini が先行している。

### Q2: Gemini Nano はどのデバイスで使えるか?

Google Pixel 8 以降、Samsung Galaxy S24 以降など、対応 Android 端末で利用可能。
AICore API を通じて呼び出す。オフラインで動作し、データが端末外に送信されない利点がある。
iOS では直接利用できないが、Chrome ブラウザ内蔵の Gemini Nano (on-device AI) が一部対応。

### Q3: Gemini API の料金体系はどうなっているか?

Google AI Studio の無料枠では 1 分あたり 15 リクエスト (1.5 Flash) / 2 リクエスト (1.5 Pro) が利用可能。
有料プランでは入力トークンと出力トークンに対してモデル別の従量課金。
128K 以下と以上でトークン単価が変わる二段階料金制を採用している点に注意。

### Q4: Search Grounding とは何か?

Gemini が Google 検索の結果をリアルタイムで参照し、回答の根拠とする機能。
ハルシネーション低減に効果的で、最新情報を含む質問に特に有効。Vertex AI で利用可能。

### Q5: Context Caching はどのような場合に使うべきか?

同一の大きなドキュメント (32K トークン以上) に対して複数回のクエリを送る場合に効果的。例えば、1つの技術文書に対して「要約」「FAQ 生成」「重要ポイント抽出」など複数の指示を出す場合、キャッシュを使うことで入力コストを 75% 削減できる。TTL は必要最小限に設定し、ストレージコストに注意する。

### Q6: Vertex AI と Google AI Studio はどう使い分けるべきか?

個人開発・プロトタイピングには Google AI Studio (API キーのみで簡単に利用可能)。
エンタープライズ利用 (SLA、データ処理契約、IAM 統合、VPC サービスコントロール) には Vertex AI。
Vertex AI は Google Cloud のセキュリティ・ガバナンス機能が利用でき、HIPAA/SOC2 準拠が必要な場合は必須。

### Q7: Gemini 2.0 と 1.5 はどちらを使うべきか?

2.0 Flash は速度・コスト・品質のバランスが最も良く、新規プロジェクトでは推奨。ただし、2M コンテキストが必要な場合は 1.5 Pro が現状唯一の選択肢。2.0 はネイティブ画像生成やツール使用の精度向上など、1.5 にない機能がある。安定性を重視する本番環境では GA (一般提供) 版を確認すること。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 開発元 | Google DeepMind |
| アーキテクチャ | Transformer (MoE) + マルチモーダルネイティブ |
| 最大コンテキスト | 200 万トークン (1.5 Pro) |
| モダリティ | テキスト、画像、音声、動画、コード |
| モデルライン | Ultra / Pro / Flash / Nano |
| API アクセス | Google AI Studio (無料枠あり)、Vertex AI |
| 差別化要因 | 超長コンテキスト、Google 統合、オンデバイス |
| 主要競合 | GPT-4o (OpenAI)、Claude (Anthropic) |
| コスト最適化 | Context Caching、Flash モデル、プロンプト最適化 |
| トラブル対策 | レート制限リトライ、Safety 診断、トークンチェック |

---

## 次に読むべきガイド

- [03-open-source.md](./03-open-source.md) — オープンソース LLM との比較
- [04-model-comparison.md](./04-model-comparison.md) — 全モデル横断比較
- [../02-applications/04-multimodal.md](../02-applications/04-multimodal.md) — マルチモーダル活用の実践

---

## 参考文献

1. Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," arXiv:2312.11805, 2023
2. Google, "Gemini API Documentation," https://ai.google.dev/docs
3. Google Cloud, "Vertex AI Gemini API," https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
4. Reid et al., "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," arXiv:2403.05530, 2024
5. Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017
6. Google, "Context Caching Guide," https://ai.google.dev/gemini-api/docs/caching
7. Google, "Grounding with Google Search," https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview
