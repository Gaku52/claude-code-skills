# エージェント評価

> ベンチマーク・成功率・コスト分析――AIエージェントの性能を定量的に測定し、継続的に改善するための評価フレームワークと手法。

## この章で学ぶこと

1. エージェント評価の多次元フレームワーク（正確性・効率性・安全性）
2. 主要ベンチマーク（SWE-bench、HumanEval、GAIA等）の理解と活用
3. 自動評価パイプラインの構築と継続的改善の実践方法

---

## 1. なぜエージェント評価が難しいのか

```
従来のLLM評価 vs エージェント評価

従来のLLM評価:
  入力 → [LLM] → 出力
  評価: 出力が正しいか？ (単一ステップ)

エージェント評価:
  入力 → [計画] → [ツール1] → [判断] → [ツール2] → ... → 出力
  評価: 最終出力 + 各ステップ + 効率 + 安全性 (多次元)

  同じ最終結果でも:
  - 3ステップで達成 → 効率的
  - 30ステップで達成 → 非効率
  - 途中で危険な操作 → 安全性に問題
```

---

## 2. 評価の多次元フレームワーク

### 2.1 評価軸

```
エージェント評価の5軸

                正確性
                 /\
                /  \
               /    \
              /      \
    効率性 __/________\__ 安全性
            \        /
             \      /
              \    /
               \  /
                \/
          堅牢性    コスト

1. 正確性 (Accuracy): タスクを正しく完了したか
2. 効率性 (Efficiency): 少ないステップ/時間で完了したか
3. 安全性 (Safety): 危険な操作をしなかったか
4. 堅牢性 (Robustness): 曖昧な入力やエラーに対処できたか
5. コスト (Cost): API費用は許容範囲か
```

### 2.2 メトリクス定義

```python
# エージェント評価メトリクスの定義
from dataclasses import dataclass

@dataclass
class AgentMetrics:
    # 正確性
    task_success_rate: float      # タスク成功率 (0-1)
    partial_completion: float     # 部分完了率 (0-1)

    # 効率性
    total_steps: int              # 総ステップ数
    tool_calls: int               # ツール呼び出し回数
    total_time_seconds: float     # 実行時間
    redundant_steps: int          # 冗長なステップ数

    # コスト
    input_tokens: int             # 入力トークン数
    output_tokens: int            # 出力トークン数
    total_cost_usd: float         # 総コスト（ドル）

    # 安全性
    unsafe_actions: int           # 安全でない操作の回数
    guardrail_triggers: int       # ガードレール発動回数

    # 堅牢性
    error_recovery_rate: float    # エラー回復率
    graceful_failures: int        # 適切な失敗処理の回数

    @property
    def cost_per_task(self) -> float:
        return self.total_cost_usd

    @property
    def efficiency_score(self) -> float:
        if self.total_steps == 0:
            return 0
        return 1 - (self.redundant_steps / self.total_steps)
```

---

## 3. 主要ベンチマーク

### 3.1 ベンチマーク一覧

```
AIエージェント主要ベンチマーク

コーディング:
  +------------------+-----------------------------------+
  | SWE-bench        | GitHubのIssueを解決               |
  | HumanEval        | コード生成の正確性                |
  | MBPP             | Python基本プログラミング          |
  +------------------+-----------------------------------+

汎用エージェント:
  +------------------+-----------------------------------+
  | GAIA             | 現実世界の複雑なタスク            |
  | AgentBench       | 多環境でのエージェント評価        |
  | WebArena         | Webブラウジングタスク             |
  +------------------+-----------------------------------+

ツール使用:
  +------------------+-----------------------------------+
  | ToolBench        | ツール選択と使用の評価            |
  | API-Bank         | API呼び出しの正確性              |
  +------------------+-----------------------------------+
```

### 3.2 ベンチマーク比較表

| ベンチマーク | 対象 | タスク数 | 評価方法 | 難易度 |
|-------------|------|---------|---------|--------|
| SWE-bench | コーディング | 2,294 | テスト通過率 | 高 |
| SWE-bench Lite | コーディング | 300 | テスト通過率 | 中-高 |
| HumanEval | コード生成 | 164 | 実行正確性 | 中 |
| GAIA | 汎用 | 466 | 最終回答一致 | 高 |
| WebArena | Webタスク | 812 | 機能的正確性 | 中-高 |
| AgentBench | 多環境 | 6,000+ | 環境依存 | 中-高 |

### 3.3 SWE-benchの実行例

```python
# SWE-bench スタイルの評価パイプライン
import subprocess
from pathlib import Path

class SWEBenchEvaluator:
    def evaluate_patch(self, repo_path: str, patch: str,
                       test_command: str) -> dict:
        """エージェントが生成したパッチを評価"""

        # 1. パッチを適用
        patch_file = Path(repo_path) / "agent.patch"
        patch_file.write_text(patch)
        apply_result = subprocess.run(
            ["git", "apply", "agent.patch"],
            cwd=repo_path, capture_output=True, text=True
        )

        if apply_result.returncode != 0:
            return {
                "success": False,
                "reason": "パッチ適用失敗",
                "error": apply_result.stderr
            }

        # 2. テスト実行
        test_result = subprocess.run(
            test_command.split(),
            cwd=repo_path, capture_output=True, text=True,
            timeout=300
        )

        # 3. 結果判定
        return {
            "success": test_result.returncode == 0,
            "tests_passed": self._count_passed(test_result.stdout),
            "tests_failed": self._count_failed(test_result.stdout),
            "output": test_result.stdout[-2000:]  # 最後の2000文字
        }
```

---

## 4. 評価パイプラインの構築

### 4.1 自動評価フレームワーク

```python
# 汎用的なエージェント評価フレームワーク
import json
import time
from typing import Callable

class AgentEvaluator:
    def __init__(self, agent_factory: Callable):
        self.agent_factory = agent_factory
        self.results = []

    def run_evaluation(self, test_cases: list[dict]) -> dict:
        """テストケースのバッチ評価"""
        for i, case in enumerate(test_cases):
            print(f"評価 {i+1}/{len(test_cases)}: {case['name']}")
            result = self._evaluate_single(case)
            self.results.append(result)

        return self._aggregate_results()

    def _evaluate_single(self, case: dict) -> dict:
        agent = self.agent_factory()
        start_time = time.time()
        start_tokens = 0

        try:
            output = agent.run(case["input"])
            elapsed = time.time() - start_time

            # 正確性チェック
            if "expected_output" in case:
                is_correct = case["checker"](output, case["expected_output"])
            elif "validator" in case:
                is_correct = case["validator"](output)
            else:
                is_correct = None

            return {
                "name": case["name"],
                "success": is_correct,
                "output": output[:500],
                "time_seconds": elapsed,
                "steps": getattr(agent, "step_count", None),
                "error": None
            }

        except Exception as e:
            return {
                "name": case["name"],
                "success": False,
                "output": None,
                "time_seconds": time.time() - start_time,
                "steps": None,
                "error": str(e)
            }

    def _aggregate_results(self) -> dict:
        total = len(self.results)
        successes = sum(1 for r in self.results if r["success"])
        times = [r["time_seconds"] for r in self.results if r["time_seconds"]]

        return {
            "total_cases": total,
            "success_rate": successes / total if total > 0 else 0,
            "avg_time": sum(times) / len(times) if times else 0,
            "max_time": max(times) if times else 0,
            "error_rate": sum(1 for r in self.results if r["error"]) / total,
            "details": self.results
        }

# 使用例
test_cases = [
    {
        "name": "簡単な計算",
        "input": "123 * 456 を計算して",
        "expected_output": "56088",
        "checker": lambda output, expected: expected in output
    },
    {
        "name": "ファイル操作",
        "input": "test.txt を作成して 'hello' と書き込んで",
        "validator": lambda output: Path("test.txt").exists()
    }
]

evaluator = AgentEvaluator(agent_factory=lambda: MyAgent())
results = evaluator.run_evaluation(test_cases)
print(f"成功率: {results['success_rate']:.1%}")
```

### 4.2 LLM-as-Judge

```python
# LLMを評価者として使う
class LLMJudge:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def evaluate(self, task: str, output: str,
                 criteria: list[str]) -> dict:
        """LLMで出力を評価"""
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
以下のタスクとその出力を評価してください。

タスク: {task}
出力: {output}

評価基準:
{criteria_text}

各基準について1-5のスコアと理由をJSON形式で出力してください。
形式: {{"criteria_name": {{"score": N, "reason": "..."}}}}
"""}]
        )
        return json.loads(response.content[0].text)

# 使用例
judge = LLMJudge()
eval_result = judge.evaluate(
    task="Pythonでクイックソートを実装して",
    output=agent_output,
    criteria=["正確性", "コードの可読性", "エラーハンドリング", "効率性"]
)
```

---

## 5. コスト分析

```
エージェントのコスト構造

+-------------------------------------------+
|  1回のタスク実行のコスト内訳               |
|                                           |
|  [LLM呼び出し] ████████████████  70%     |
|  [ツール実行]   ████              15%     |
|  [メモリ/検索]  ███               10%     |
|  [その他]       █                  5%     |
+-------------------------------------------+

コスト最適化のレバー:
1. モデル選択（Haiku vs Sonnet vs Opus）
2. ステップ数の最小化
3. コンテキストサイズの管理
4. キャッシュの活用
```

```python
# コスト追跡の実装
class CostTracker:
    # Claude 3.5 Sonnet の料金（2025年時点の概算）
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},  # per 1M tokens
        "claude-haiku-4-20250514": {"input": 0.25, "output": 1.25},
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = 0

    def track(self, response):
        usage = response.usage
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.calls += 1

    def get_cost(self, model: str) -> float:
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def summary(self, model: str) -> str:
        cost = self.get_cost(model)
        return (
            f"API呼び出し: {self.calls}回\n"
            f"入力トークン: {self.total_input_tokens:,}\n"
            f"出力トークン: {self.total_output_tokens:,}\n"
            f"推定コスト: ${cost:.4f}"
        )
```

---

## 6. 比較表

### 6.1 評価手法比較

| 手法 | 自動化 | 精度 | コスト | スケーラビリティ |
|------|--------|------|--------|----------------|
| 人手評価 | なし | 最高 | 最高 | 低 |
| LLM-as-Judge | 高 | 高 | 中 | 高 |
| 自動テスト | 完全 | 中-高 | 低 | 最高 |
| ベンチマーク | 完全 | 中 | 低 | 高 |
| A/Bテスト | 中 | 高 | 中 | 中 |

### 6.2 評価頻度と目的

| 頻度 | 目的 | 手法 |
|------|------|------|
| 毎コミット | リグレッション検出 | 自動テスト（CI） |
| 毎日 | パフォーマンス監視 | メトリクスダッシュボード |
| 毎週 | 品質トレンド | LLM-as-Judge + サンプリング |
| 毎月 | 包括的評価 | ベンチマーク + 人手評価 |
| モデル更新時 | 互換性確認 | 全テストスイート |

---

## 7. アンチパターン

### アンチパターン1: 成功率のみで評価

```
# NG: 成功率90%だけ見て「優秀」と判断
成功率: 90% ← 一見良い
しかし:
  平均コスト: $2.50/タスク ← 高すぎ
  平均時間: 5分/タスク ← 遅すぎ
  安全違反: 5% ← 危険

# OK: 多次元で評価
成功率: 85% + コスト: $0.30 + 時間: 30秒 + 安全違反: 0%
→ こちらの方が実用的に優れている可能性
```

### アンチパターン2: ベンチマーク過学習

```python
# NG: ベンチマークのテストケースに合わせてプロンプトを調整
system_prompt = """
SWE-benchのタスクの場合は...  # ← ベンチマーク固有の最適化
"""

# OK: 汎用的な能力を評価
# 独自テストケースも含めてバランスよく評価
test_suite = benchmark_cases + custom_cases + edge_cases
```

---

## 8. FAQ

### Q1: 評価の自動化はどこまで可能か？

正確性（テスト通過）とコスト（トークン数）は完全自動化可能。品質（コードの可読性、回答の有用性）は LLM-as-Judge で準自動化。安全性は自動チェック + 人手サンプリングの組み合わせが現実的。

### Q2: 最低限測定すべきメトリクスは？

**3つのコアメトリクス**:
1. **タスク成功率**: 正しく完了した割合
2. **平均コスト/タスク**: API費用
3. **平均ステップ数**: 効率の指標

この3つがあれば、改善の方向性が見える。

### Q3: A/Bテストの方法は？

エージェントA（旧版）とB（新版）に同じタスクセットを実行させ、成功率・コスト・品質を比較する。統計的有意性のために **最低50タスク** は必要。LLM-as-Judgeで品質の比較判定を行うとスケーラブル。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 評価5軸 | 正確性・効率性・安全性・堅牢性・コスト |
| ベンチマーク | SWE-bench, GAIA, HumanEval等 |
| 評価手法 | 自動テスト / LLM-as-Judge / 人手評価 |
| コスト追跡 | トークン数 x 単価で算出 |
| 核心原則 | 単一メトリクスでなく多次元で評価 |
| 最低限 | 成功率 + コスト + ステップ数 |

## 次に読むべきガイド

- [../04-production/00-deployment.md](../04-production/00-deployment.md) — 本番環境でのモニタリング
- [../04-production/01-safety.md](../04-production/01-safety.md) — 安全性の評価と確保
- [../03-applications/00-coding-agents.md](../03-applications/00-coding-agents.md) — コーディングエージェントの評価

## 参考文献

1. Jimenez, C. E. et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2023) — https://arxiv.org/abs/2310.06770
2. Mialon, G. et al., "GAIA: A Benchmark for General AI Assistants" (2023) — https://arxiv.org/abs/2311.12983
3. Zheng, L. et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023) — https://arxiv.org/abs/2306.05685
