# LLM 評価 — ベンチマーク・LMSYS・人間評価

> LLM 評価はモデルの品質を定量的・定性的に測定する体系であり、適切な評価なしにモデル選定・プロンプト改善・ファインチューニングの判断は不可能である。自動ベンチマーク、LLM-as-a-Judge、人間評価を組み合わせた多面的評価が求められる。

## この章で学ぶこと

1. **自動ベンチマークの体系と読み方** — MMLU、HumanEval、MT-Bench、Arena Elo の意味と限界
2. **LLM-as-a-Judge の実装** — GPT-4o を評価者として使う手法、バイアス対策
3. **実務での評価パイプライン構築** — 自社タスク評価、A/B テスト、継続的モニタリング

---

## 1. 評価手法の全体像

```
┌──────────────────────────────────────────────────────────┐
│           LLM 評価手法の分類体系                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  自動ベンチマーク (Static)                                │
│  ├── 知識: MMLU, ARC, HellaSwag                         │
│  ├── 推論: GSM8K, MATH, BIG-Bench Hard                  │
│  ├── コード: HumanEval, MBPP, SWE-bench                 │
│  └── 多言語: MGSM, JMMLU                                │
│                                                          │
│  LLM-as-a-Judge (Dynamic)                                │
│  ├── MT-Bench: 多ターン対話を GPT-4 が 1-10 点で評価    │
│  ├── AlpacaEval: 指示追従を GPT-4 が勝率で評価          │
│  ├── 自社評価: カスタム評価基準 + LLM 評価者             │
│  └── Pairwise: 2 つの出力を比較してどちらが良いか判定   │
│                                                          │
│  人間評価 (Gold Standard)                                 │
│  ├── LMSYS Chatbot Arena: ブラインド A/B 人間投票        │
│  ├── Elo レーティング: 対戦結果から算出                  │
│  └── 専門家レビュー: ドメイン専門家による品質評価        │
│                                                          │
│  信頼性: 人間評価 > LLM-as-a-Judge > 自動ベンチマーク   │
│  コスト: 人間評価 > LLM-as-a-Judge > 自動ベンチマーク   │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 主要ベンチマークの詳細

### 2.1 MMLU (Massive Multitask Language Understanding)

```python
# MMLU の構造
mmlu_structure = {
    "総問題数": 14042,
    "科目数": 57,
    "形式": "4択問題",
    "分野": {
        "STEM": ["数学", "物理", "化学", "CS", "工学"],
        "人文": ["歴史", "哲学", "法律"],
        "社会科学": ["経済", "政治", "心理学"],
        "その他": ["臨床医学", "会計", "マーケティング"],
    },
}

# MMLU の評価例
example = {
    "question": "光の速度に最も近い値はどれか？",
    "choices": [
        "A) 3×10^6 m/s",
        "B) 3×10^8 m/s",     # 正解
        "C) 3×10^10 m/s",
        "D) 3×10^12 m/s",
    ],
    "subject": "physics",
    "answer": "B",
}
```

### 2.2 HumanEval (コード生成)

```python
# HumanEval の評価方式: pass@k
# k個のコード生成サンプルのうち1つでもテスト通過すれば正解

def evaluate_humaneval(model, problems, n_samples=10, k=1):
    """HumanEval の pass@k 評価"""
    results = []

    for problem in problems:
        completions = [model.generate(problem["prompt"]) for _ in range(n_samples)]

        # 各生成コードをテストケースで検証
        passed = sum(1 for code in completions
                     if run_tests(code, problem["tests"]))

        # pass@k の計算 (不偏推定量)
        import math
        if passed >= k:
            pass_at_k = 1.0 - math.comb(n_samples - passed, k) / math.comb(n_samples, k)
        else:
            pass_at_k = 0.0

        results.append(pass_at_k)

    return sum(results) / len(results)
```

### 2.3 LMSYS Chatbot Arena

```
┌──────────────────────────────────────────────────────────┐
│          LMSYS Chatbot Arena の仕組み                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. ユーザーがプロンプトを入力                             │
│     │                                                    │
│  2. 2つの匿名モデルが同時に回答                          │
│     ┌─────────┐    ┌─────────┐                          │
│     │ Model A │    │ Model B │  ← モデル名は非表示      │
│     │ (匿名)  │    │ (匿名)  │                          │
│     └────┬────┘    └────┬────┘                          │
│          │              │                                │
│  3. ユーザーが「A勝利/B勝利/引き分け」を投票              │
│     │                                                    │
│  4. Elo レーティングを更新                                │
│     ├── 高Elo = 強いモデル                               │
│     └── 累計投票数: 200万+                               │
│                                                          │
│  利点:                                                    │
│  - ベンチマーク汚染の影響を受けない                       │
│  - 実際のユーザー体験に基づく                             │
│  - 継続的に更新される                                     │
│                                                          │
│  限界:                                                    │
│  - 英語中心                                               │
│  - 短い対話が多い                                         │
│  - 専門タスクのカバレッジが薄い                           │
└──────────────────────────────────────────────────────────┘
```

---

## 3. LLM-as-a-Judge の実装

### 3.1 基本的な LLM 評価

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question: str, answer: str, criteria: str) -> dict:
    """LLM を評価者として使用"""
    prompt = f"""
以下の質問に対する回答を評価してください。

<question>
{question}
</question>

<answer>
{answer}
</answer>

<evaluation_criteria>
{criteria}
</evaluation_criteria>

以下の形式で評価を出力してください:
- スコア: 1-5 の整数 (5が最高)
- 理由: 評価の根拠を2-3文で

JSON形式で出力:
{{"score": <int>, "reason": "<string>"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 再現性のため
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

# 使用例
result = llm_judge(
    question="Pythonのデコレータを説明してください",
    answer="デコレータは関数を修飾する構文糖です...",
    criteria="""
    1. 技術的正確性 (誤りがないか)
    2. 分かりやすさ (初学者が理解できるか)
    3. 具体例の有無 (コード例が含まれているか)
    4. 完全性 (重要な概念が網羅されているか)
    """,
)
print(f"スコア: {result['score']}/5")
print(f"理由: {result['reason']}")
```

### 3.2 Pairwise 比較

```python
def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """2つの回答を比較評価 (バイアス軽減版)"""
    # 順序バイアス軽減: A/B の順番を入れ替えて2回評価
    results = []

    for swap in [False, True]:
        first = answer_b if swap else answer_a
        second = answer_a if swap else answer_b

        prompt = f"""
以下の質問に対する2つの回答を比較し、どちらが優れているか判定してください。

質問: {question}

回答1:
{first}

回答2:
{second}

以下の基準で評価:
- 正確性、有用性、明確性、完全性

判定: "回答1が優れている"、"回答2が優れている"、"同等" のいずれかを選び、
理由を述べてください。

JSON: {{"winner": "1" | "2" | "tie", "reason": "<string>"}}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        # 入れ替えた場合は結果を反転
        if swap:
            if result["winner"] == "1":
                result["winner"] = "2"
            elif result["winner"] == "2":
                result["winner"] = "1"

        results.append(result)

    # 2回の結果が一致するか確認
    if results[0]["winner"] == results[1]["winner"]:
        return {"winner": results[0]["winner"], "confidence": "high",
                "reason": results[0]["reason"]}
    else:
        return {"winner": "tie", "confidence": "low",
                "reason": "順序入替で結果が変わったため判定困難"}
```

---

## 4. 評価パイプラインの構築

### 4.1 自社タスク評価データセットの作成

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalCase:
    """評価ケース"""
    id: str
    category: str
    input_text: str
    reference_answer: Optional[str]  # 正解 (あれば)
    evaluation_criteria: str
    difficulty: str  # easy / medium / hard

# 評価データセットの例
eval_dataset = [
    EvalCase(
        id="sum-001",
        category="summarization",
        input_text="[長い技術文書]",
        reference_answer="[理想的な要約]",
        evaluation_criteria="正確性、網羅性、簡潔性",
        difficulty="medium",
    ),
    EvalCase(
        id="code-001",
        category="code_generation",
        input_text="二分探索を実装してください",
        reference_answer="def binary_search(arr, target): ...",
        evaluation_criteria="正確性、効率性、可読性",
        difficulty="easy",
    ),
    # 50-100ケース推奨
]
```

### 4.2 自動評価パイプライン

```python
import asyncio
from datetime import datetime

async def run_evaluation(
    models: list[str],
    eval_dataset: list[EvalCase],
    judge_model: str = "gpt-4o",
) -> dict:
    """複数モデルの一括評価"""
    results = {}

    for model in models:
        model_results = []

        for case in eval_dataset:
            # モデルの回答を生成
            answer = await generate(model, case.input_text)

            # LLM-as-a-Judge で評価
            evaluation = await llm_judge(
                question=case.input_text,
                answer=answer,
                criteria=case.evaluation_criteria,
            )

            model_results.append({
                "case_id": case.id,
                "category": case.category,
                "score": evaluation["score"],
                "reason": evaluation["reason"],
            })

        # カテゴリ別集計
        results[model] = {
            "overall": sum(r["score"] for r in model_results) / len(model_results),
            "by_category": aggregate_by_category(model_results),
            "details": model_results,
        }

    return results

def print_leaderboard(results: dict):
    """評価結果のリーダーボード表示"""
    sorted_models = sorted(results.items(), key=lambda x: x[1]["overall"], reverse=True)

    print(f"{'モデル':30s} {'総合':>6s} {'要約':>6s} {'コード':>6s} {'QA':>6s}")
    print("-" * 60)
    for model, data in sorted_models:
        cats = data["by_category"]
        print(f"{model:30s} "
              f"{data['overall']:>6.2f} "
              f"{cats.get('summarization', 0):>6.2f} "
              f"{cats.get('code_generation', 0):>6.2f} "
              f"{cats.get('qa', 0):>6.2f}")
```

### 4.3 RAGAS (RAG 評価)

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # 回答がコンテキストに忠実か
    answer_relevancy,   # 回答が質問に関連しているか
    context_precision,  # 検索結果に正解が含まれるか
    context_recall,     # 必要な情報が検索できたか
)

# 評価データの準備
eval_data = {
    "question": ["RAGとは何ですか？", ...],
    "answer": ["RAGは検索拡張生成の略で...", ...],
    "contexts": [["RAGはRetrieval-Augmented..."], ...],
    "ground_truth": ["RAGは外部知識ベースから...", ...],
}

results = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results)
# {faithfulness: 0.85, answer_relevancy: 0.92, context_precision: 0.78, ...}
```

---

## 5. 評価のバイアスと対策

```
┌──────────────────────────────────────────────────────────┐
│          LLM-as-a-Judge の既知バイアス                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Position Bias (順序バイアス)                          │
│     → 先に提示された回答を好む傾向                       │
│     対策: A/B 順序を入れ替えて2回評価し多数決             │
│                                                          │
│  2. Verbosity Bias (冗長性バイアス)                       │
│     → 長い回答を「より良い」と判断する傾向                │
│     対策: 評価基準に「簡潔性」を明示的に含める            │
│                                                          │
│  3. Self-Enhancement Bias (自己強化バイアス)              │
│     → GPT-4 は GPT-4 の出力を好む傾向                   │
│     対策: 異なるモデル (Claude) を評価者に使う            │
│                                                          │
│  4. Capability Bias (能力限界)                            │
│     → 評価者が判断できない専門分野                       │
│     対策: ドメイン専門家による人間評価で補完              │
└──────────────────────────────────────────────────────────┘
```

---

## 6. 比較表

### 6.1 評価手法比較

| 手法 | コスト | 速度 | 信頼性 | スケーラビリティ | 適用場面 |
|------|--------|------|--------|----------------|---------|
| 自動ベンチマーク | 低 | 高速 | 中 | 高 | モデル選定の足切り |
| LLM-as-a-Judge | 中 | 中速 | 中〜高 | 高 | 日常的な品質チェック |
| 人間評価 (一般) | 高 | 低速 | 高 | 低 | 最終品質検証 |
| 専門家評価 | 最高 | 最低 | 最高 | 最低 | ドメイン固有タスク |
| A/B テスト | 中 | 低速 | 最高 | 中 | 本番環境での比較 |

### 6.2 自動評価ツール比較

| ツール | 主な評価対象 | 特徴 | 料金 |
|--------|------------|------|------|
| RAGAS | RAG パイプライン | Faithfulness, Relevancy | OSS |
| DeepEval | LLM 全般 | 14+ メトリクス | OSS |
| LangSmith | LLM アプリ | トレース + 評価 | 有料 |
| Braintrust | LLM 全般 | CI/CD 統合 | 有料 |
| HumanLoop | LLM 全般 | 人間評価統合 | 有料 |

---

## 7. アンチパターン

### アンチパターン 1: ベンチマークスコアのみで判断

```
# NG: "MMLU 88点だから最高のモデルだ"
→ MMLU は多肢選択 (知識テスト) であり、
  要約、対話、コード生成の品質とは弱い相関

→ ベンチマーク汚染 (訓練データにベンチマーク問題が混入) の
  疑いがあるモデルも存在

# OK: 自社タスクの評価データセットで実測
1. 自社タスク 100問の評価セットを作成
2. 候補モデル 3-5 個で実行
3. LLM-as-a-Judge + 人間サンプル評価
4. スコアだけでなく失敗パターンも分析
```

### アンチパターン 2: 評価基準なしの主観評価

```python
# NG: 「何となくこっちの方が良い気がする」
subjective_result = "Model A の方が良い"  # 根拠不明

# OK: 明確な評価基準 (ルーブリック) を定義
rubric = {
    "正確性": {
        5: "事実の誤りが一切ない",
        4: "軽微な不正確さが1箇所",
        3: "部分的に不正確だが主旨は正しい",
        2: "複数の事実誤認がある",
        1: "根本的に間違っている",
    },
    "有用性": {
        5: "質問に完全に答えており、追加の洞察もある",
        4: "質問に十分に答えている",
        3: "質問に部分的に答えている",
        2: "質問の一部にしか答えていない",
        1: "質問に答えていない",
    },
}
```

---

## 8. FAQ

### Q1: 評価データセットは何件必要か?

最低 50 件、推奨 100-200 件。統計的に有意な差を検出するには、カテゴリあたり 30 件以上が望ましい。
McNemar 検定で有意差検定を行う場合、p < 0.05 を得るには十分なサンプル数が必要。
少数で始めて徐々に拡充するアプローチが現実的。

### Q2: LLM-as-a-Judge の評価者モデルは何を使うべき?

GPT-4o が最も広く使われ、人間評価との一致率が高い。
ただし Self-Enhancement Bias があるため、OpenAI モデルの評価には Claude を使うなど、
評価者と被評価者を異なるプロバイダにすることが望ましい。

### Q3: 継続的なモデル評価はどう自動化する?

CI/CD パイプラインに評価ステップを組み込む。
LangSmith や Braintrust でプロンプト変更ごとに自動評価を実行。
「回帰テスト」として、過去の評価スコアを下回らないことをデプロイ条件にする。
週次/月次で Elo レーティングを更新し、モデル乗り換え判断に使う。

---

## まとめ

| 項目 | 推奨 |
|------|------|
| モデル選定の足切り | MMLU / HumanEval / Arena Elo |
| 日常的品質チェック | LLM-as-a-Judge (GPT-4o) |
| 最終品質検証 | 人間評価 (専門家) |
| RAG 評価 | RAGAS フレームワーク |
| バイアス対策 | 順序入替 + 複数評価者 |
| 評価データ規模 | 100-200 ケース (カテゴリ別 30+) |
| 継続モニタリング | CI/CD 統合 + 週次レポート |

---

## 次に読むべきガイド

- [../01-models/04-model-comparison.md](../01-models/04-model-comparison.md) — 評価結果に基づくモデル選定
- [../02-applications/00-prompt-engineering.md](../02-applications/00-prompt-engineering.md) — プロンプト改善の評価
- [../04-ethics/00-ai-safety.md](../04-ethics/00-ai-safety.md) — 安全性評価

---

## 参考文献

1. Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference," arXiv:2403.04132, 2024
2. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023
3. RAGAS, "RAG Assessment," https://docs.ragas.io/
4. Hendrycks et al., "Measuring Massive Multitask Language Understanding," ICLR 2021
5. Chen et al., "Evaluating Large Language Models Trained on Code (HumanEval)," arXiv:2107.03374, 2021
