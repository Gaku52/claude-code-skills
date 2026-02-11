# 自律エージェント

> 計画・実行・振り返り――長時間にわたり自律的にタスクを遂行するエージェントの設計パターン。目標分解、自己評価、適応的再計画の仕組みを解説する。

## この章で学ぶこと

1. 自律エージェントの計画-実行-振り返りサイクルの設計
2. 目標分解とサブゴール管理の実装パターン
3. 自律性の段階とヒューマン・イン・ザ・ループの組み込み方

---

## 1. 自律エージェントの定義

```
自律エージェントの特徴

通常のエージェント:
  ユーザー: "Xを調べて"
  → 検索→回答（数ステップで完了）

自律エージェント:
  ユーザー: "競合分析レポートを作成して"
  → 計画立案
  → 情報収集（複数ソース）
  → データ分析
  → レポート草案
  → 自己レビュー
  → 修正
  → 最終レポート
  （数十〜数百ステップ、数分〜数時間）
```

### 1.1 自律性のレベル

```
自律性の5段階

Level 0: 手動          ユーザーが全ステップ指示
Level 1: アシスト      1ステップをLLMが実行
Level 2: 半自律        複数ステップを実行、要所で確認
Level 3: 条件付き自律  ほぼ自律、重要判断のみ人間が承認
Level 4: 完全自律      目標だけ与えれば完了まで自走
                       (例: Devin, Claude Code)
```

---

## 2. 計画-実行-振り返りサイクル

### 2.1 コアアーキテクチャ

```
自律エージェントのコアループ

      +--------+
      | 目標   |
      +---+----+
          |
          v
  +-------+-------+
  |   計画 (Plan)  |←────────────+
  +-------+-------+              |
          |                      |
          v                      |
  +-------+-------+      +------+------+
  |  実行 (Act)   |----->| 振り返り    |
  +-------+-------+      | (Reflect)   |
          |               +------+------+
          v                      |
     成功? ─── YES ──→ 完了     |
          |                      |
          NO ──→ 再計画 ────────+
```

### 2.2 完全な実装

```python
# 自律エージェントの完全な実装
import anthropic
import json
import time
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class SubTask:
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    attempts: int = 0

class AutonomousAgent:
    def __init__(self, tools: list, max_steps: int = 50):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.max_steps = max_steps
        self.plan: list[SubTask] = []
        self.completed_work = []
        self.reflections = []

    def run(self, goal: str) -> str:
        """目標を受け取り、自律的に完了まで実行"""
        # Phase 1: 計画
        self.plan = self._create_plan(goal)
        print(f"計画: {len(self.plan)} サブタスク")

        for step in range(self.max_steps):
            # Phase 2: 次のサブタスクを選択・実行
            next_task = self._select_next_task()
            if next_task is None:
                break  # 全タスク完了

            print(f"Step {step}: {next_task.description}")
            result = self._execute_task(next_task)

            # Phase 3: 振り返り
            reflection = self._reflect(goal, next_task, result)
            self.reflections.append(reflection)

            # 必要なら再計画
            if reflection.get("needs_replan"):
                self.plan = self._replan(goal, reflection["reason"])

        # 最終まとめ
        return self._synthesize(goal)

    def _create_plan(self, goal: str) -> list[SubTask]:
        """目標をサブタスクに分解"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
目標: {goal}

この目標を達成するためのサブタスクを JSON 配列で出力してください。
各サブタスクは独立して実行可能で、依存関係がある場合は順序で表現。
形式: [{{"id": 1, "description": "..."}}]
"""}]
        )
        tasks_data = json.loads(response.content[0].text)
        return [SubTask(id=t["id"], description=t["description"]) for t in tasks_data]

    def _select_next_task(self) -> SubTask | None:
        """次に実行すべきサブタスクを選択"""
        for task in self.plan:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
                return task
        return None

    def _execute_task(self, task: SubTask) -> str:
        """サブタスクを実行（ツール使用あり）"""
        messages = [{"role": "user", "content": f"""
サブタスク: {task.description}
これまでの成果: {json.dumps(self.completed_work[-3:], ensure_ascii=False)}

このサブタスクを完了してください。
"""}]

        # エージェントループ（最大10ステップ）
        for _ in range(10):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                result = response.content[0].text
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_work.append({
                    "task": task.description,
                    "result": result
                })
                return result

            # ツール呼び出し処理
            tool_results = self._handle_tool_calls(response)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        task.status = TaskStatus.FAILED
        return "タスクが完了できませんでした"

    def _reflect(self, goal: str, task: SubTask, result: str) -> dict:
        """実行結果を振り返り、評価する"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
全体目標: {goal}
完了したタスク: {task.description}
結果: {result}
残りのタスク: {[t.description for t in self.plan if t.status == TaskStatus.PENDING]}

以下をJSON形式で評価してください:
- quality: "good" / "acceptable" / "poor"
- needs_replan: true/false
- reason: 再計画が必要な理由（不要ならnull）
- learning: この経験から学んだこと
"""}]
        )
        return json.loads(response.content[0].text)
```

---

## 3. 目標分解パターン

### 3.1 階層的目標分解

```
目標の階層分解

[最上位目標]
├── [サブゴール 1]
│   ├── [タスク 1.1]
│   ├── [タスク 1.2]
│   └── [タスク 1.3]
├── [サブゴール 2]
│   ├── [タスク 2.1]
│   └── [タスク 2.2]
└── [サブゴール 3]
    ├── [タスク 3.1]
    ├── [タスク 3.2]
    └── [タスク 3.3]

例: "ECサイトを構築する"
├── "DB設計をする"
│   ├── ER図を作成
│   ├── テーブル定義
│   └── マイグレーション実行
├── "APIを実装する"
│   ├── 認証API
│   └── 商品API
└── "フロントエンドを構築する"
    ├── 商品一覧ページ
    ├── カートページ
    └── 決済ページ
```

### 3.2 適応的再計画

```python
# 適応的再計画の実装
class AdaptivePlanner:
    def replan(self, goal: str, current_state: dict,
               failure_reason: str) -> list[SubTask]:
        """失敗理由を考慮して計画を再作成"""
        response = self.llm.generate(f"""
目標: {goal}

現在の状態:
- 完了済み: {current_state['completed']}
- 失敗: {current_state['failed']}
- 失敗理由: {failure_reason}

過去の振り返り:
{current_state['reflections']}

失敗から学んで、残りのタスクを再計画してください。
以前と同じアプローチは避け、代替手段を検討してください。
""")
        return self._parse_plan(response)
```

---

## 4. 自己評価メカニズム

```python
# 多角的な自己評価
class SelfEvaluator:
    def evaluate(self, goal: str, output: str) -> dict:
        """出力を多角的に評価"""

        # 観点1: 目標達成度
        completeness = self.llm.generate(f"""
目標: {goal}
出力: {output}
目標の達成度を0-100%で評価: """)

        # 観点2: 品質
        quality = self.llm.generate(f"""
出力: {output}
品質（正確性、完全性、明確性）を0-100で評価: """)

        # 観点3: 改善余地
        improvements = self.llm.generate(f"""
出力: {output}
改善可能な点を3つ挙げてください: """)

        return {
            "completeness": int(completeness),
            "quality": int(quality),
            "improvements": improvements,
            "should_improve": int(completeness) < 80 or int(quality) < 70
        }
```

```
自己評価のフロー

  [出力] → [完全性チェック] → 80%未満 → [再実行]
              |
              v 80%以上
         [品質チェック] → 70点未満 → [改善ループ]
              |
              v 70点以上
         [最終確認] → 承認 → [完了]
```

---

## 5. ヒューマン・イン・ザ・ループ

### 5.1 介入ポイントの設計

```
ヒューマン・イン・ザ・ループの介入ポイント

[計画] ──確認──> [人間の承認] ──→ [実行] ──→ [振り返り]
                                      |
                                 重要判断 ──確認──> [人間の判断]
                                      |
                                 破壊的操作 ──確認──> [人間の承認]
```

### 5.2 実装パターン

```python
# ヒューマン・イン・ザ・ループの実装
class HumanInTheLoopAgent(AutonomousAgent):
    def __init__(self, *args, approval_required: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.approval_required = approval_required or [
            "delete", "deploy", "send", "purchase", "modify_production"
        ]

    def _execute_task(self, task: SubTask) -> str:
        # タスクが承認必要リストに該当するかチェック
        needs_approval = any(
            keyword in task.description.lower()
            for keyword in self.approval_required
        )

        if needs_approval:
            print(f"\n[承認要求] タスク: {task.description}")
            approval = input("承認しますか？ (yes/no/modify): ")

            if approval == "no":
                task.status = TaskStatus.BLOCKED
                return "ユーザーにより拒否されました"
            elif approval == "modify":
                task.description = input("修正後のタスク: ")

        return super()._execute_task(task)
```

---

## 6. 自律性レベル比較

| レベル | 説明 | 人間の関与 | 適用場面 | リスク |
|--------|------|-----------|---------|--------|
| L0 手動 | 全て手動 | 100% | - | 最低 |
| L1 アシスト | 1ステップ実行 | 80% | IDE補完 | 低 |
| L2 半自律 | 複数ステップ | 50% | チャットボット | 低-中 |
| L3 条件付き | 重要判断のみ | 10-20% | コーディングエージェント | 中 |
| L4 完全自律 | 目標のみ指定 | 0-5% | 自動デプロイ | 高 |

### 代表的プロダクトの自律性レベル

| プロダクト | レベル | 特徴 |
|-----------|--------|------|
| GitHub Copilot | L1 | 行単位の補完 |
| ChatGPT | L1-L2 | ツール付き対話 |
| Claude Code | L3 | コーディング+ファイル操作 |
| Devin | L3-L4 | ソフトウェア開発の自律実行 |
| AutoGPT | L4 | 完全自律（実用性は課題） |

---

## 7. アンチパターン

### アンチパターン1: 振り返りなしの猪突猛進

```python
# NG: 結果を評価せず次に進む
for task in plan:
    result = execute(task)
    # 結果が悪くてもそのまま続行...

# OK: 各ステップで振り返りと必要に応じた再計画
for task in plan:
    result = execute(task)
    evaluation = reflect(result)
    if evaluation["quality"] == "poor":
        plan = replan(goal, evaluation["reason"])
```

### アンチパターン2: 過度な自律性（ガードレールなし）

```python
# NG: 制限なしの自律実行
agent.run("サーバーのパフォーマンスを最適化して")
# → 勝手に本番設定を変更、サービスダウン

# OK: 適切なガードレール
agent = AutonomousAgent(
    max_steps=30,                    # ステップ上限
    max_cost=5.0,                    # コスト上限（ドル）
    timeout=600,                     # タイムアウト（秒）
    forbidden_actions=["rm -rf", "DROP TABLE"],  # 禁止操作
    require_approval=["deploy", "delete"]         # 承認必須操作
)
```

---

## 8. FAQ

### Q1: 自律エージェントの実行時間の目安は？

タスクの複雑さに依存するが、現時点の目安:
- **単純タスク**（ファイル操作、検索）: 30秒〜2分
- **中程度**（コーディング、分析）: 2分〜10分
- **複雑**（設計+実装+テスト）: 10分〜1時間
- **大規模**（プロジェクト全体）: 1時間以上

長時間タスクは **チェックポイント** と **進捗通知** が必須。

### Q2: 自律エージェントが「迷走」した場合の検出方法は？

- **同一ツールの連続呼び出し**: 3回以上同じツールを同じ引数で呼び出し
- **コスト急増**: 予想コストの2倍を超えた時点
- **進捗の停滞**: 最後の成功タスクから5ステップ以上経過
- **矛盾する行動**: 作成したものをすぐ削除するなど

検出時は **自動的に一時停止し、ユーザーに判断を求める** のが安全。

### Q3: Reflexionパターンとは何か？

Reflexionは「実行→失敗→反省→再実行」のサイクルを明示的にモデル化したパターン。通常のリトライと異なり、**失敗の原因を言語化してメモリに保存し、次の試行で同じ失敗を避ける** 点が特徴。テスト駆動開発のような「赤→緑」のフィードバックループと相性が良い。

---

## まとめ

| 項目 | 内容 |
|------|------|
| コアループ | 計画 → 実行 → 振り返り → （再計画） |
| 目標分解 | 階層的にサブゴール/タスクに分解 |
| 自己評価 | 完全性・品質・改善余地の多角評価 |
| 再計画 | 失敗からの学習を次の計画に反映 |
| HITL | 重要判断ポイントで人間の承認を挟む |
| ガードレール | ステップ/コスト/時間の上限を設定 |

## 次に読むべきガイド

- [../02-implementation/03-claude-agent-sdk.md](../02-implementation/03-claude-agent-sdk.md) — Claude Agent SDKでの自律エージェント構築
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) — エージェントの評価手法
- [../04-production/01-safety.md](../04-production/01-safety.md) — 安全性とガードレール

## 参考文献

1. Shinn, N. et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) — https://arxiv.org/abs/2303.11366
2. Yao, S. et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023) — https://arxiv.org/abs/2305.10601
3. Wang, G. et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023) — https://arxiv.org/abs/2305.16291
