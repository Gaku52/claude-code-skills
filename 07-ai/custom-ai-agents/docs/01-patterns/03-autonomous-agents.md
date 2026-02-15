# 自律エージェント

> 計画・実行・振り返り――長時間にわたり自律的にタスクを遂行するエージェントの設計パターン。目標分解、自己評価、適応的再計画の仕組みを解説する。

## この章で学ぶこと

1. 自律エージェントの計画-実行-振り返りサイクルの設計
2. 目標分解とサブゴール管理の実装パターン
3. 自律性の段階とヒューマン・イン・ザ・ループの組み込み方
4. メモリシステム（短期・長期・エピソード記憶）の設計
5. 安全性ガードレールと迷走検出の実装
6. 本番運用のためのモニタリング・コスト管理・テスト手法

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

### 1.2 自律エージェントを選ぶべきか？

```
意思決定フローチャート

Q1: タスクが明確に定義されており、手順が固定か？
├─ YES → ワークフローエージェント（02-workflow-agents.md）
└─ NO  → Q2へ

Q2: タスク完了に10ステップ以上必要か？
├─ YES → Q3へ
└─ NO  → シングルエージェント（00-single-agent.md）

Q3: 実行中に状況に応じた判断の変更が必要か？
├─ YES → 自律エージェント（この章）
└─ NO  → ワークフローエージェント

Q4: 失敗時に自己修正が必要か？
├─ YES → 自律エージェント + Reflexionパターン
└─ NO  → マルチエージェント委譲パターン

Q5: セキュリティ的に許容できる操作範囲は？
├─ 制限あり → Level 2-3（ヒューマン・イン・ザ・ループ必須）
└─ 制限なし → Level 4（フルガードレール必須）
```

### 1.3 典型的なユースケース

| ユースケース | 自律性レベル | ステップ数 | 所要時間 |
|------------|-----------|----------|---------|
| コード生成+テスト | L3 | 20-50 | 5-15分 |
| 競合分析レポート作成 | L3 | 30-100 | 10-30分 |
| バグ調査+修正 | L3-L4 | 10-40 | 3-20分 |
| プロジェクト初期構築 | L3 | 50-200 | 15-60分 |
| データ分析+可視化 | L2-L3 | 15-40 | 5-20分 |
| 文書翻訳+ローカライズ | L2 | 10-30 | 3-10分 |
| インフラ構築+デプロイ | L3（承認付き） | 30-80 | 10-40分 |

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
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

@dataclass
class SubTask:
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    attempts: int = 0
    max_attempts: int = 3
    dependencies: list[int] = field(default_factory=list)
    priority: int = 0  # 0が最高

    @property
    def can_execute(self) -> bool:
        return self.status == TaskStatus.PENDING and self.attempts < self.max_attempts

@dataclass
class ExecutionTrace:
    """実行トレースの記録"""
    step: int
    task_id: int
    task_description: str
    action: str
    result: str
    reflection: dict
    tokens_used: int
    duration: float
    timestamp: float

class AutonomousAgent:
    def __init__(
        self,
        tools: list,
        max_steps: int = 50,
        max_cost: float = 5.0,
        timeout: float = 3600,
        model: str = "claude-sonnet-4-20250514"
    ):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.timeout = timeout
        self.model = model
        self.plan: list[SubTask] = []
        self.completed_work: list[dict] = []
        self.reflections: list[dict] = []
        self.traces: list[ExecutionTrace] = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = 0.0
        self.agent_id = str(uuid.uuid4())[:8]

    def run(self, goal: str) -> str:
        """目標を受け取り、自律的に完了まで実行"""
        self.start_time = time.time()
        logger.info(f"[{self.agent_id}] 目標: {goal}")

        # Phase 1: 計画
        self.plan = self._create_plan(goal)
        logger.info(f"[{self.agent_id}] 計画: {len(self.plan)} サブタスク")

        for step in range(self.max_steps):
            # ガードレールチェック
            if self._should_stop():
                logger.warning(f"[{self.agent_id}] ガードレールにより停止")
                break

            # Phase 2: 次のサブタスクを選択・実行
            next_task = self._select_next_task()
            if next_task is None:
                break  # 全タスク完了

            logger.info(f"[{self.agent_id}] Step {step}: {next_task.description}")
            step_start = time.time()
            result = self._execute_task(next_task)

            # Phase 3: 振り返り
            reflection = self._reflect(goal, next_task, result)
            self.reflections.append(reflection)

            # トレース記録
            self.traces.append(ExecutionTrace(
                step=step,
                task_id=next_task.id,
                task_description=next_task.description,
                action="execute",
                result=result[:500],
                reflection=reflection,
                tokens_used=self.total_tokens,
                duration=time.time() - step_start,
                timestamp=time.time()
            ))

            # 必要なら再計画
            if reflection.get("needs_replan"):
                logger.info(f"[{self.agent_id}] 再計画: {reflection['reason']}")
                self.plan = self._replan(goal, reflection["reason"])

        # 最終まとめ
        return self._synthesize(goal)

    def _should_stop(self) -> bool:
        """ガードレールチェック"""
        # タイムアウト
        if time.time() - self.start_time > self.timeout:
            logger.warning("タイムアウト")
            return True

        # コスト上限
        if self.total_cost > self.max_cost:
            logger.warning(f"コスト上限超過: ${self.total_cost:.2f}")
            return True

        # 迷走検出
        if self._detect_wandering():
            logger.warning("迷走検出")
            return True

        return False

    def _detect_wandering(self) -> bool:
        """エージェントの迷走を検出"""
        if len(self.traces) < 5:
            return False

        recent = self.traces[-5:]

        # 同じタスクへの連続失敗
        task_ids = [t.task_id for t in recent]
        if len(set(task_ids)) == 1:
            task = next((t for t in self.plan if t.id == task_ids[0]), None)
            if task and task.status != TaskStatus.COMPLETED:
                return True

        # 全ての最近の振り返りがpoor評価
        all_poor = all(
            t.reflection.get("quality") == "poor"
            for t in recent
        )
        if all_poor:
            return True

        return False

    def _create_plan(self, goal: str) -> list[SubTask]:
        """目標をサブタスクに分解"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
目標: {goal}

この目標を達成するためのサブタスクを JSON 配列で出力してください。
各サブタスクは独立して実行可能で、依存関係がある場合は順序で表現。

形式:
[
  {{"id": 1, "description": "...", "dependencies": [], "priority": 0}},
  {{"id": 2, "description": "...", "dependencies": [1], "priority": 1}}
]

JSONのみ出力してください。
"""}]
        )
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        self._update_cost(response.usage)

        tasks_data = json.loads(response.content[0].text)
        return [
            SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                priority=t.get("priority", 0)
            )
            for t in tasks_data
        ]

    def _select_next_task(self) -> Optional[SubTask]:
        """次に実行すべきサブタスクを選択（依存関係考慮）"""
        completed_ids = {
            t.id for t in self.plan if t.status == TaskStatus.COMPLETED
        }

        candidates = [
            t for t in self.plan
            if t.can_execute
            and all(dep in completed_ids for dep in t.dependencies)
        ]

        if not candidates:
            return None

        # 優先度順にソート
        candidates.sort(key=lambda t: t.priority)
        selected = candidates[0]
        selected.status = TaskStatus.IN_PROGRESS
        selected.attempts += 1
        return selected

    def _execute_task(self, task: SubTask) -> str:
        """サブタスクを実行（ツール使用あり）"""
        context = ""
        if self.completed_work:
            recent_work = self.completed_work[-3:]
            context = f"\nこれまでの成果:\n{json.dumps(recent_work, ensure_ascii=False, indent=2)}"

        messages = [{"role": "user", "content": f"""
サブタスク: {task.description}
{context}

このサブタスクを完了してください。
"""}]

        # エージェントループ（最大10ステップ）
        for _ in range(10):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            self._update_cost(response.usage)

            if response.stop_reason == "end_turn":
                result = response.content[0].text
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_work.append({
                    "task_id": task.id,
                    "task": task.description,
                    "result": result
                })
                return result

            # ツール呼び出し処理
            tool_results = self._handle_tool_calls(response)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        task.status = TaskStatus.FAILED
        return "タスクが最大ステップ内で完了できませんでした"

    def _handle_tool_calls(self, response) -> list[dict]:
        """ツール呼び出しを処理"""
        results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_result = self._run_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(tool_result)
                })
        return results

    def _run_tool(self, name: str, input_data: dict) -> Any:
        """ツールを実行"""
        for tool in self.tools:
            if tool.get("name") == name:
                # 実際のツール実行ロジック
                logger.info(f"ツール実行: {name}({json.dumps(input_data, ensure_ascii=False)[:100]})")
                return f"ツール {name} の実行結果"
        return f"不明なツール: {name}"

    def _reflect(self, goal: str, task: SubTask, result: str) -> dict:
        """実行結果を振り返り、評価する"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
全体目標: {goal}
完了したタスク: {task.description}
結果: {result[:1000]}
残りのタスク: {[t.description for t in self.plan if t.status == TaskStatus.PENDING]}
これまでの振り返り: {json.dumps(self.reflections[-3:], ensure_ascii=False) if self.reflections else "なし"}

以下をJSON形式で評価してください:
{{
  "quality": "good" / "acceptable" / "poor",
  "needs_replan": true/false,
  "reason": "再計画が必要な理由（不要ならnull）",
  "learning": "この経験から学んだこと",
  "confidence": 0.0-1.0
}}

JSONのみ出力してください。
"""}]
        )
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        self._update_cost(response.usage)

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {
                "quality": "acceptable",
                "needs_replan": False,
                "reason": None,
                "learning": "振り返りの解析に失敗",
                "confidence": 0.5
            }

    def _replan(self, goal: str, reason: str) -> list[SubTask]:
        """失敗理由を考慮して計画を再作成"""
        completed = [t for t in self.plan if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.plan if t.status == TaskStatus.FAILED]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
目標: {goal}

現在の状態:
- 完了済み: {[t.description for t in completed]}
- 失敗: {[t.description for t in failed]}
- 再計画理由: {reason}
- これまでの学び: {[r.get('learning', '') for r in self.reflections[-5:]]}

完了済みのタスクはそのまま保持し、残りのタスクを再計画してください。
以前と同じアプローチは避け、代替手段を検討してください。

JSON配列で出力（完了済みは含めない）:
[{{"id": N, "description": "...", "dependencies": [], "priority": 0}}]
"""}]
        )
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        self._update_cost(response.usage)

        # 完了済みタスクを保持
        new_tasks_data = json.loads(response.content[0].text)
        new_tasks = [
            SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                priority=t.get("priority", 0)
            )
            for t in new_tasks_data
        ]

        return completed + new_tasks

    def _synthesize(self, goal: str) -> str:
        """完了したタスクの結果を統合して最終出力を生成"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": f"""
目標: {goal}

完了したタスクとその結果:
{json.dumps(self.completed_work, ensure_ascii=False, indent=2)}

振り返りからの学び:
{json.dumps([r.get('learning', '') for r in self.reflections], ensure_ascii=False)}

上記の成果を統合して、目標に対する最終的な成果物を出力してください。
"""}]
        )
        return response.content[0].text

    def _update_cost(self, usage):
        """コストを更新（Sonnet基準）"""
        self.total_cost += (
            usage.input_tokens * 3.0 / 1_000_000
            + usage.output_tokens * 15.0 / 1_000_000
        )

    def get_execution_summary(self) -> dict:
        """実行サマリーを取得"""
        completed = sum(1 for t in self.plan if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.plan if t.status == TaskStatus.FAILED)
        total_time = time.time() - self.start_time

        return {
            "agent_id": self.agent_id,
            "total_tasks": len(self.plan),
            "completed": completed,
            "failed": failed,
            "total_steps": len(self.traces),
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "total_time": f"{total_time:.1f}s",
            "success_rate": f"{completed / max(len(self.plan), 1) * 100:.1f}%"
        }
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

### 3.2 HTA（Hierarchical Task Analysis）による分解

```python
# 階層的タスク分析に基づく目標分解
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HTANode:
    """HTAツリーのノード"""
    id: str
    description: str
    children: list["HTANode"] = field(default_factory=list)
    plan: str = ""  # このノードの実行計画
    is_leaf: bool = False  # 末端タスクか

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    def flatten(self) -> list["HTANode"]:
        """末端タスクをフラット化"""
        if self.is_leaf or not self.children:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.flatten())
        return result

class HTAPlanner:
    """階層的タスク分析による計画生成"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def decompose(self, goal: str, max_depth: int = 3) -> HTANode:
        """目標を階層的に分解"""
        root = HTANode(id="0", description=goal)
        self._decompose_recursive(root, depth=0, max_depth=max_depth)
        return root

    def _decompose_recursive(
        self, node: HTANode, depth: int, max_depth: int
    ):
        """再帰的に分解"""
        if depth >= max_depth:
            node.is_leaf = True
            return

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": f"""
タスク: {node.description}

このタスクを2-5個のサブタスクに分解してください。
各サブタスクは具体的で実行可能なものにしてください。

JSON配列で出力:
[
  {{"id": "1", "description": "...", "is_leaf": true/false}},
  ...
]

is_leaf=true: これ以上分解不要な具体的アクション
is_leaf=false: さらに分解可能な抽象的タスク
"""}]
        )

        children_data = json.loads(response.content[0].text)
        for child_data in children_data:
            child = HTANode(
                id=f"{node.id}.{child_data['id']}",
                description=child_data["description"],
                is_leaf=child_data.get("is_leaf", False)
            )
            node.children.append(child)

            if not child.is_leaf:
                self._decompose_recursive(child, depth + 1, max_depth)

    def to_subtasks(self, root: HTANode) -> list[SubTask]:
        """HTAツリーをSubTaskリストに変換"""
        leaves = root.flatten()
        return [
            SubTask(
                id=i + 1,
                description=leaf.description,
                priority=i
            )
            for i, leaf in enumerate(leaves)
        ]

# 使用例
planner = HTAPlanner(anthropic.Anthropic())
tree = planner.decompose("ECサイトの決済機能を実装する", max_depth=2)
tasks = planner.to_subtasks(tree)
```

### 3.3 適応的再計画

```python
# 適応的再計画の実装
class AdaptivePlanner:
    """失敗と学びから計画を適応的に更新"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.plan_history: list[list[SubTask]] = []
        self.failure_patterns: list[dict] = []

    def replan(
        self,
        goal: str,
        current_state: dict,
        failure_reason: str
    ) -> list[SubTask]:
        """失敗理由を考慮して計画を再作成"""
        # 失敗パターンを記録
        self.failure_patterns.append({
            "reason": failure_reason,
            "failed_tasks": current_state.get("failed", []),
            "timestamp": time.time()
        })

        # 過去の失敗パターンから学ぶ
        failure_summary = "\n".join(
            f"- {p['reason']}" for p in self.failure_patterns[-5:]
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
目標: {goal}

現在の状態:
- 完了済み: {current_state['completed']}
- 失敗: {current_state['failed']}
- 失敗理由: {failure_reason}

過去の失敗パターン:
{failure_summary}

振り返りからの学び:
{current_state.get('reflections', [])}

以下の原則で再計画してください:
1. 過去に失敗したアプローチは避ける
2. 代替手段を積極的に検討する
3. 各タスクをより小さく具体的にする
4. リスクの高いタスクには前段階の確認を入れる

JSON配列で出力:
[{{"id": N, "description": "...", "dependencies": [], "priority": 0}}]
"""}]
        )

        tasks_data = json.loads(response.content[0].text)
        new_plan = [
            SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                priority=t.get("priority", 0)
            )
            for t in tasks_data
        ]

        self.plan_history.append(new_plan)
        return new_plan

    def get_plan_evolution(self) -> str:
        """計画の変遷を取得"""
        lines = []
        for i, plan in enumerate(self.plan_history):
            lines.append(f"=== 計画 v{i+1} ({len(plan)}タスク) ===")
            for t in plan:
                lines.append(f"  [{t.id}] {t.description}")
        return "\n".join(lines)
```

---

## 4. メモリシステム

### 4.1 メモリの3層構造

```
自律エージェントのメモリアーキテクチャ

┌────────────────────────────────────────────┐
│            ワーキングメモリ (短期)           │
│  ・現在のタスクのコンテキスト               │
│  ・直近の会話履歴                          │
│  ・LLMのコンテキストウィンドウ内            │
│  ・容量: 数千〜数万トークン                │
├────────────────────────────────────────────┤
│           エピソード記憶 (中期)             │
│  ・過去の成功/失敗の記録                    │
│  ・学んだ教訓のリスト                      │
│  ・タスク間で引き継ぐ情報                  │
│  ・容量: JSON/テキストファイル              │
├────────────────────────────────────────────┤
│           セマンティック記憶 (長期)          │
│  ・ベクトルDBに格納                        │
│  ・類似経験の検索に使用                     │
│  ・ドメイン知識の蓄積                      │
│  ・容量: 数千〜数百万エントリ              │
└────────────────────────────────────────────┘
```

### 4.2 メモリシステムの実装

```python
# 3層メモリシステム
from dataclasses import dataclass, field
from typing import Optional
import json
import time
import hashlib

@dataclass
class MemoryEntry:
    """メモリエントリ"""
    content: str
    type: str  # "success", "failure", "learning", "fact"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0

class WorkingMemory:
    """ワーキングメモリ（短期）"""

    def __init__(self, max_items: int = 20):
        self.items: list[dict] = []
        self.max_items = max_items

    def add(self, content: str, type: str = "observation"):
        self.items.append({
            "content": content,
            "type": type,
            "timestamp": time.time()
        })
        # 容量超過時は古いものから削除
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    def get_context(self, max_tokens: int = 2000) -> str:
        """LLMに渡すコンテキストを生成"""
        context_parts = []
        total_chars = 0
        for item in reversed(self.items):
            text = f"[{item['type']}] {item['content']}"
            if total_chars + len(text) > max_tokens * 4:
                break
            context_parts.insert(0, text)
            total_chars += len(text)
        return "\n".join(context_parts)

    def clear(self):
        self.items = []

class EpisodicMemory:
    """エピソード記憶（中期）"""

    def __init__(self, filepath: str = "episodic_memory.json"):
        self.filepath = filepath
        self.episodes: list[MemoryEntry] = []
        self._load()

    def _load(self):
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                self.episodes = [
                    MemoryEntry(**ep) for ep in data
                ]
        except FileNotFoundError:
            self.episodes = []

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(
                [
                    {
                        "content": ep.content,
                        "type": ep.type,
                        "tags": ep.tags,
                        "importance": ep.importance,
                        "timestamp": ep.timestamp,
                        "access_count": ep.access_count
                    }
                    for ep in self.episodes
                ],
                f, ensure_ascii=False, indent=2
            )

    def record(self, content: str, type: str,
               tags: list[str] = None, importance: float = 0.5):
        """エピソードを記録"""
        entry = MemoryEntry(
            content=content,
            type=type,
            tags=tags or [],
            importance=importance
        )
        self.episodes.append(entry)
        self._save()

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """関連するエピソードを検索"""
        # 簡易的なキーワードマッチング
        query_words = set(query.lower().split())
        scored = []
        for ep in self.episodes:
            content_words = set(ep.content.lower().split())
            tag_words = set(w.lower() for w in ep.tags)
            overlap = len(query_words & (content_words | tag_words))
            score = overlap * ep.importance * (1 + ep.access_count * 0.1)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:top_k] if _ > 0]

        # アクセスカウント更新
        for ep in results:
            ep.access_count += 1
        self._save()

        return results

    def get_learnings(self) -> list[str]:
        """学びのリストを取得"""
        return [
            ep.content for ep in self.episodes
            if ep.type == "learning"
        ]

class SemanticMemory:
    """セマンティック記憶（長期）- ベクトルDB連携"""

    def __init__(self, collection_name: str = "agent_memory"):
        # ChromaDB等のベクトルDBを使用
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            logger.warning("chromadb未インストール: セマンティック記憶は無効")
            self.collection = None

    def store(self, content: str, metadata: dict = None):
        """知識を格納"""
        if not self.collection:
            return

        doc_id = hashlib.md5(content.encode()).hexdigest()
        self.collection.upsert(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """類似知識を検索"""
        if not self.collection:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        return [
            {
                "content": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

class AgentMemorySystem:
    """統合メモリシステム"""

    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    def remember(self, content: str, type: str, importance: float = 0.5):
        """記憶を統一的に格納"""
        self.working.add(content, type)
        self.episodic.record(content, type, importance=importance)

        if importance > 0.7:
            self.semantic.store(content, {"type": type})

    def recall_relevant(self, query: str) -> dict:
        """関連する記憶を全層から取得"""
        return {
            "working": self.working.get_context(),
            "episodic": [
                ep.content for ep in self.episodic.recall(query)
            ],
            "semantic": [
                r["content"] for r in self.semantic.search(query)
            ]
        }
```

### 4.3 メモリ統合エージェント

```python
# メモリ付き自律エージェント
class MemoryAugmentedAgent(AutonomousAgent):
    """メモリシステムを統合した自律エージェント"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = AgentMemorySystem()

    def _execute_task(self, task: SubTask) -> str:
        # 関連記憶を検索
        memories = self.memory.recall_relevant(task.description)

        # メモリをコンテキストに追加
        memory_context = ""
        if memories["episodic"]:
            memory_context += "\n関連する過去の経験:\n"
            memory_context += "\n".join(f"- {m}" for m in memories["episodic"][:3])
        if memories["semantic"]:
            memory_context += "\n関連する知識:\n"
            memory_context += "\n".join(f"- {m}" for m in memories["semantic"][:3])

        messages = [{"role": "user", "content": f"""
サブタスク: {task.description}
{memory_context}

これまでの成果:
{json.dumps(self.completed_work[-3:], ensure_ascii=False)}

このサブタスクを完了してください。
"""}]

        # 実行（親クラスのロジックと同様）
        result = self._execute_with_messages(messages)

        # 結果をメモリに記録
        self.memory.remember(
            f"タスク'{task.description}'を実行。結果: {result[:200]}",
            type="success" if task.status == TaskStatus.COMPLETED else "failure",
            importance=0.6
        )

        return result

    def _reflect(self, goal: str, task: SubTask, result: str) -> dict:
        reflection = super()._reflect(goal, task, result)

        # 学びをメモリに記録
        if reflection.get("learning"):
            self.memory.remember(
                reflection["learning"],
                type="learning",
                importance=0.8
            )

        return reflection
```

---

## 5. 自己評価メカニズム

### 5.1 多角的自己評価

```python
# 多角的な自己評価
class SelfEvaluator:
    """エージェントの出力を多角的に評価"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def evaluate(self, goal: str, output: str) -> dict:
        """出力を多角的に評価"""

        # 観点1: 目標達成度
        completeness_resp = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": f"""
目標: {goal}
出力: {output[:2000]}

目標の達成度を0-100の数値のみで回答:"""}]
        )
        completeness = int(completeness_resp.content[0].text.strip())

        # 観点2: 品質
        quality_resp = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": f"""
出力: {output[:2000]}
品質（正確性、完全性、明確性）を0-100の数値のみで回答:"""}]
        )
        quality = int(quality_resp.content[0].text.strip())

        # 観点3: 改善余地
        improvements_resp = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": f"""
出力: {output[:2000]}
改善可能な点を3つ、箇条書きで挙げてください:"""}]
        )
        improvements = improvements_resp.content[0].text

        return {
            "completeness": completeness,
            "quality": quality,
            "improvements": improvements,
            "should_improve": completeness < 80 or quality < 70,
            "overall_score": (completeness + quality) / 2
        }

    def evaluate_with_rubric(self, output: str, rubric: dict) -> dict:
        """ルーブリックに基づく評価"""
        results = {}
        for criterion, description in rubric.items():
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": f"""
出力:
{output[:1500]}

評価基準「{criterion}」: {description}

この基準に対するスコア(0-10)と根拠を以下の形式で:
スコア: N
根拠: ...
"""}]
            )
            text = response.content[0].text
            score_line = text.split("\n")[0]
            score = int(score_line.split(":")[-1].strip())
            results[criterion] = {
                "score": score,
                "feedback": text
            }

        return results
```

### 5.2 自己評価フロー

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

### 5.3 Reflexionパターンの実装

```python
# Reflexion: 言語的自己強化学習
class ReflexionAgent(AutonomousAgent):
    """Reflexionパターンを組み込んだ自律エージェント"""

    def __init__(self, *args, max_reflexion_rounds: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_reflexion_rounds = max_reflexion_rounds
        self.reflexion_memory: list[str] = []  # 言語的な経験メモリ

    def run(self, goal: str) -> str:
        """Reflexionループで実行"""
        best_result = None
        best_score = 0

        for round_num in range(self.max_reflexion_rounds):
            logger.info(f"Reflexion Round {round_num + 1}")

            # 実行
            result = self._execute_round(goal, round_num)

            # 評価
            evaluator = SelfEvaluator(self.client)
            evaluation = evaluator.evaluate(goal, result)
            score = evaluation["overall_score"]

            logger.info(
                f"  スコア: {score:.1f} "
                f"(完全性: {evaluation['completeness']}, "
                f"品質: {evaluation['quality']})"
            )

            if score > best_score:
                best_score = score
                best_result = result

            # 基準を満たしたら終了
            if not evaluation["should_improve"]:
                logger.info("  品質基準達成")
                break

            # 振り返り（Reflexion）
            reflexion = self._generate_reflexion(
                goal, result, evaluation
            )
            self.reflexion_memory.append(reflexion)
            logger.info(f"  Reflexion: {reflexion[:100]}...")

            # 計画をリセットして再試行
            self.plan = []
            self.completed_work = []

        return best_result

    def _execute_round(self, goal: str, round_num: int) -> str:
        """1ラウンドの実行"""
        # 過去のReflexionをコンテキストに含める
        reflexion_context = ""
        if self.reflexion_memory:
            reflexion_context = "\n過去の振り返り（同じ失敗を避けること）:\n"
            for i, r in enumerate(self.reflexion_memory):
                reflexion_context += f"Round {i+1}: {r}\n"

        augmented_goal = f"{goal}\n{reflexion_context}"

        self.plan = self._create_plan(augmented_goal)

        for step in range(self.max_steps):
            if self._should_stop():
                break

            next_task = self._select_next_task()
            if next_task is None:
                break

            self._execute_task(next_task)

        return self._synthesize(goal)

    def _generate_reflexion(
        self, goal: str, result: str, evaluation: dict
    ) -> str:
        """失敗からの振り返りを生成"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": f"""
目標: {goal}
実行結果: {result[:1000]}

評価:
- 完全性: {evaluation['completeness']}%
- 品質: {evaluation['quality']}点
- 改善点: {evaluation['improvements']}

なぜこの結果が不十分だったのか、
次回どうすれば改善できるか、
具体的で実行可能なアドバイスを2-3文で:
"""}]
        )
        return response.content[0].text
```

---

## 6. ヒューマン・イン・ザ・ループ

### 6.1 介入ポイントの設計

```
ヒューマン・イン・ザ・ループの介入ポイント

[計画] ──確認──> [人間の承認] ──→ [実行] ──→ [振り返り]
                                      |
                                 重要判断 ──確認──> [人間の判断]
                                      |
                                 破壊的操作 ──確認──> [人間の承認]
```

### 6.2 実装パターン

```python
# ヒューマン・イン・ザ・ループの実装
from enum import Enum

class ApprovalLevel(Enum):
    NONE = "none"          # 承認不要
    INFO = "info"          # 通知のみ
    OPTIONAL = "optional"  # 任意承認（タイムアウトで自動承認）
    REQUIRED = "required"  # 必須承認
    BLOCKING = "blocking"  # ブロッキング承認（絶対に人間の判断が必要）

class HumanInTheLoopAgent(AutonomousAgent):
    """ヒューマン・イン・ザ・ループ付き自律エージェント"""

    def __init__(
        self,
        *args,
        approval_rules: dict[str, ApprovalLevel] = None,
        approval_timeout: float = 300,  # 5分
        notification_callback=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.approval_rules = approval_rules or {
            "delete": ApprovalLevel.REQUIRED,
            "deploy": ApprovalLevel.BLOCKING,
            "send": ApprovalLevel.REQUIRED,
            "purchase": ApprovalLevel.BLOCKING,
            "modify_production": ApprovalLevel.BLOCKING,
            "install": ApprovalLevel.OPTIONAL,
            "create": ApprovalLevel.INFO,
        }
        self.approval_timeout = approval_timeout
        self.notification_callback = notification_callback
        self.approval_log: list[dict] = []

    def _get_approval_level(self, task: SubTask) -> ApprovalLevel:
        """タスクの承認レベルを判定"""
        for keyword, level in self.approval_rules.items():
            if keyword in task.description.lower():
                return level
        return ApprovalLevel.NONE

    def _request_approval(
        self, task: SubTask, level: ApprovalLevel
    ) -> tuple[bool, str]:
        """人間の承認を要求"""
        log_entry = {
            "task": task.description,
            "level": level.value,
            "timestamp": time.time(),
            "decision": None
        }

        if level == ApprovalLevel.INFO:
            # 通知のみ
            if self.notification_callback:
                self.notification_callback(
                    f"[INFO] タスク実行中: {task.description}"
                )
            log_entry["decision"] = "auto_approved"
            self.approval_log.append(log_entry)
            return True, ""

        if level == ApprovalLevel.OPTIONAL:
            # タイムアウト付き承認
            print(f"\n[承認要求(任意)] タスク: {task.description}")
            print(f"  {self.approval_timeout}秒以内に入力がない場合、自動承認されます")
            # 実際のUIではWebSocket/Slack等で通知
            try:
                import signal
                signal.alarm(int(self.approval_timeout))
                approval = input("承認しますか？ (yes/no): ").strip().lower()
                signal.alarm(0)
            except Exception:
                approval = "yes"  # タイムアウト → 自動承認

            approved = approval != "no"
            log_entry["decision"] = "approved" if approved else "rejected"
            self.approval_log.append(log_entry)
            return approved, ""

        # REQUIRED / BLOCKING
        print(f"\n{'='*50}")
        print(f"[承認必須] タスク: {task.description}")
        print(f"承認レベル: {level.value}")
        print(f"{'='*50}")
        approval = input("承認しますか？ (yes/no/modify): ").strip().lower()

        if approval == "no":
            log_entry["decision"] = "rejected"
            self.approval_log.append(log_entry)
            return False, "ユーザーにより拒否されました"
        elif approval == "modify":
            new_desc = input("修正後のタスク内容: ")
            task.description = new_desc
            log_entry["decision"] = "modified"
            log_entry["modified_to"] = new_desc
            self.approval_log.append(log_entry)
            return True, ""
        else:
            log_entry["decision"] = "approved"
            self.approval_log.append(log_entry)
            return True, ""

    def _execute_task(self, task: SubTask) -> str:
        level = self._get_approval_level(task)

        if level != ApprovalLevel.NONE:
            approved, message = self._request_approval(task, level)
            if not approved:
                task.status = TaskStatus.BLOCKED
                return message

        return super()._execute_task(task)
```

### 6.3 非同期承認（Slack/Web統合）

```python
# Slack経由の承認フロー
import asyncio
from typing import Callable, Awaitable

class AsyncApprovalSystem:
    """非同期承認システム"""

    def __init__(self):
        self.pending_approvals: dict[str, asyncio.Future] = {}

    async def request_approval(
        self,
        approval_id: str,
        task_description: str,
        timeout: float = 300
    ) -> bool:
        """承認を非同期で要求"""
        future = asyncio.get_event_loop().create_future()
        self.pending_approvals[approval_id] = future

        # Slackに通知を送信
        await self._send_slack_notification(
            f"承認要求: {task_description}\n"
            f"承認: `/approve {approval_id}`\n"
            f"拒否: `/reject {approval_id}`"
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"承認タイムアウト: {approval_id}")
            return False
        finally:
            self.pending_approvals.pop(approval_id, None)

    def handle_approval_response(self, approval_id: str, approved: bool):
        """承認応答を処理（Slackコマンドから呼ばれる）"""
        future = self.pending_approvals.get(approval_id)
        if future and not future.done():
            future.set_result(approved)

    async def _send_slack_notification(self, message: str):
        """Slack通知を送信"""
        # Slack API連携の実装
        logger.info(f"Slack通知: {message}")
```

---

## 7. 安全性ガードレール

### 7.1 多層防御

```python
# 自律エージェントのガードレール
from dataclasses import dataclass
from typing import Callable

@dataclass
class GuardRail:
    """ガードレール定義"""
    name: str
    check: Callable[[dict], bool]  # True=安全, False=危険
    action: str  # "block", "warn", "require_approval"
    description: str

class GuardedAutonomousAgent(AutonomousAgent):
    """ガードレール付き自律エージェント"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guardrails: list[GuardRail] = self._default_guardrails()
        self.violations: list[dict] = []

    def _default_guardrails(self) -> list[GuardRail]:
        return [
            GuardRail(
                name="step_limit",
                check=lambda ctx: ctx["step"] < self.max_steps,
                action="block",
                description="最大ステップ数の制限"
            ),
            GuardRail(
                name="cost_limit",
                check=lambda ctx: ctx["cost"] < self.max_cost,
                action="block",
                description="コスト上限の制限"
            ),
            GuardRail(
                name="timeout",
                check=lambda ctx: ctx["elapsed"] < self.timeout,
                action="block",
                description="実行時間の制限"
            ),
            GuardRail(
                name="forbidden_commands",
                check=lambda ctx: not any(
                    cmd in ctx.get("action", "").lower()
                    for cmd in ["rm -rf", "drop table", "format", "shutdown"]
                ),
                action="block",
                description="危険なコマンドの禁止"
            ),
            GuardRail(
                name="pii_filter",
                check=lambda ctx: not self._contains_pii(ctx.get("output", "")),
                action="warn",
                description="個人情報の出力防止"
            ),
            GuardRail(
                name="loop_detection",
                check=lambda ctx: not self._detect_wandering(),
                action="require_approval",
                description="無限ループの検出"
            ),
            GuardRail(
                name="scope_check",
                check=lambda ctx: self._is_within_scope(
                    ctx.get("action", ""), ctx.get("goal", "")
                ),
                action="warn",
                description="目標の範囲内かチェック"
            ),
        ]

    def _contains_pii(self, text: str) -> bool:
        """個人情報を含むかチェック"""
        import re
        patterns = [
            r'\b\d{3}-\d{4}-\d{4}\b',  # 電話番号
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # メール
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # クレジットカード
        ]
        return any(re.search(p, text) for p in patterns)

    def _is_within_scope(self, action: str, goal: str) -> bool:
        """アクションが目標の範囲内かチェック"""
        # 簡易的なスコープチェック
        # 本番ではLLMによる判断を推奨
        return True

    def check_guardrails(self, context: dict) -> list[dict]:
        """全ガードレールをチェック"""
        violations = []
        for rail in self.guardrails:
            try:
                if not rail.check(context):
                    violation = {
                        "guardrail": rail.name,
                        "action": rail.action,
                        "description": rail.description,
                        "timestamp": time.time()
                    }
                    violations.append(violation)
                    self.violations.append(violation)
                    logger.warning(
                        f"ガードレール違反: {rail.name} - {rail.description}"
                    )
            except Exception as e:
                logger.error(f"ガードレールチェックエラー: {rail.name} - {e}")

        return violations

    def _execute_task(self, task: SubTask) -> str:
        context = {
            "step": len(self.traces),
            "cost": self.total_cost,
            "elapsed": time.time() - self.start_time,
            "action": task.description,
            "goal": "",
        }

        violations = self.check_guardrails(context)

        for v in violations:
            if v["action"] == "block":
                task.status = TaskStatus.BLOCKED
                return f"ガードレールにより停止: {v['description']}"
            elif v["action"] == "require_approval":
                print(f"\n[ガードレール警告] {v['description']}")
                approval = input("続行しますか？ (yes/no): ")
                if approval != "yes":
                    task.status = TaskStatus.BLOCKED
                    return "ユーザーにより停止"

        return super()._execute_task(task)
```

### 7.2 迷走検出の詳細実装

```python
# 高度な迷走検出
class WanderingDetector:
    """エージェントの迷走を検出するシステム"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.action_history: list[dict] = []

    def record_action(self, action: str, result: str, success: bool):
        self.action_history.append({
            "action": action,
            "result_hash": hashlib.md5(result.encode()).hexdigest()[:8],
            "success": success,
            "timestamp": time.time()
        })

    def is_wandering(self) -> tuple[bool, str]:
        """迷走を検出し、理由を返す"""
        if len(self.action_history) < self.window_size:
            return False, ""

        recent = self.action_history[-self.window_size:]

        # パターン1: 同一アクションの繰り返し
        actions = [a["action"] for a in recent]
        most_common = max(set(actions), key=actions.count)
        if actions.count(most_common) > self.window_size * 0.7:
            return True, f"同一アクションの繰り返し: {most_common}"

        # パターン2: 同一結果の繰り返し
        results = [a["result_hash"] for a in recent]
        if len(set(results)) < 3:
            return True, "同一結果の繰り返し"

        # パターン3: 連続失敗
        failures = sum(1 for a in recent if not a["success"])
        if failures > self.window_size * 0.8:
            return True, f"連続失敗: {failures}/{self.window_size}"

        # パターン4: 作成→削除の繰り返し
        create_delete_pairs = 0
        for i in range(len(recent) - 1):
            if ("create" in recent[i]["action"].lower() and
                "delete" in recent[i+1]["action"].lower()):
                create_delete_pairs += 1
        if create_delete_pairs >= 3:
            return True, "作成→削除の繰り返し（矛盾する行動）"

        return False, ""
```

---

## 8. 自律性レベル比較

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

## 9. モニタリングとコスト管理

### 9.1 実行モニタリング

```python
# 自律エージェントのモニタリング
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

@dataclass
class AgentMonitor:
    """自律エージェントのモニタリングシステム"""

    metrics: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    alerts: list[dict] = field(default_factory=list)
    alert_thresholds: dict[str, float] = field(default_factory=lambda: {
        "step_duration_p95": 30.0,  # 秒
        "cost_per_step": 0.05,     # USD
        "failure_rate": 0.3,       # 30%
        "wandering_score": 0.7,    # 迷走スコア
    })

    def record_step(
        self,
        step: int,
        duration: float,
        cost: float,
        success: bool,
        tokens: int
    ):
        """ステップのメトリクスを記録"""
        self.metrics["duration"].append(duration)
        self.metrics["cost"].append(cost)
        self.metrics["success"].append(1.0 if success else 0.0)
        self.metrics["tokens"].append(tokens)

        # アラートチェック
        self._check_alerts(step)

    def _check_alerts(self, step: int):
        """アラート条件をチェック"""
        durations = self.metrics["duration"]
        if len(durations) >= 5:
            p95 = sorted(durations)[int(len(durations) * 0.95)]
            if p95 > self.alert_thresholds["step_duration_p95"]:
                self._add_alert("高レイテンシ", f"P95={p95:.1f}s", step)

        costs = self.metrics["cost"]
        if costs and costs[-1] > self.alert_thresholds["cost_per_step"]:
            self._add_alert(
                "高コストステップ",
                f"${costs[-1]:.4f}",
                step
            )

        successes = self.metrics["success"]
        if len(successes) >= 5:
            recent_rate = statistics.mean(successes[-5:])
            if recent_rate < (1 - self.alert_thresholds["failure_rate"]):
                self._add_alert(
                    "高失敗率",
                    f"直近5ステップ成功率={recent_rate*100:.0f}%",
                    step
                )

    def _add_alert(self, type: str, detail: str, step: int):
        alert = {
            "type": type,
            "detail": detail,
            "step": step,
            "timestamp": time.time()
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{type}]: {detail} (step {step})")

    def get_report(self) -> str:
        """モニタリングレポートを生成"""
        if not self.metrics["duration"]:
            return "データなし"

        lines = [
            "=== 自律エージェント実行レポート ===",
            f"総ステップ数: {len(self.metrics['duration'])}",
            f"総実行時間: {sum(self.metrics['duration']):.1f}s",
            f"総コスト: ${sum(self.metrics['cost']):.4f}",
            f"総トークン: {sum(self.metrics['tokens']):,}",
            f"成功率: {statistics.mean(self.metrics['success'])*100:.1f}%",
            f"",
            f"--- ステップ別統計 ---",
            f"実行時間: 平均={statistics.mean(self.metrics['duration']):.2f}s, "
            f"最大={max(self.metrics['duration']):.2f}s",
            f"コスト: 平均=${statistics.mean(self.metrics['cost']):.4f}, "
            f"最大=${max(self.metrics['cost']):.4f}",
        ]

        if self.alerts:
            lines.append(f"\n--- アラート ({len(self.alerts)}件) ---")
            for a in self.alerts[-10:]:
                lines.append(f"  [{a['type']}] {a['detail']} (step {a['step']})")

        return "\n".join(lines)
```

### 9.2 コスト最適化

```python
# コスト最適化戦略
class CostOptimizer:
    """自律エージェントのコスト最適化"""

    MODEL_TIERS = {
        "fast": {
            "model": "claude-haiku-3-20240307",
            "input_cost": 0.25,
            "output_cost": 1.25,
        },
        "balanced": {
            "model": "claude-sonnet-4-20250514",
            "input_cost": 3.0,
            "output_cost": 15.0,
        },
        "best": {
            "model": "claude-opus-4-20250514",
            "input_cost": 15.0,
            "output_cost": 75.0,
        }
    }

    @staticmethod
    def select_model_for_phase(phase: str) -> str:
        """フェーズに応じたモデルを選択"""
        phase_model_map = {
            "planning": "balanced",     # 計画は中品質で十分
            "execution": "balanced",    # 実行は中品質
            "reflection": "fast",       # 振り返りは高速で十分
            "evaluation": "balanced",   # 評価は中品質
            "synthesis": "best",        # 最終統合は高品質
            "classification": "fast",   # 分類は高速で十分
        }
        tier = phase_model_map.get(phase, "balanced")
        return CostOptimizer.MODEL_TIERS[tier]["model"]

    @staticmethod
    def estimate_cost(
        plan: list[SubTask],
        avg_tokens_per_task: int = 2000
    ) -> dict:
        """実行前にコストを見積もる"""
        # 各フェーズの推定コスト
        planning_cost = avg_tokens_per_task * 2 * 3.0 / 1_000_000  # 計画
        execution_cost = (
            len(plan) * avg_tokens_per_task * 2 * 3.0 / 1_000_000
        )
        reflection_cost = (
            len(plan) * avg_tokens_per_task * 0.25 / 1_000_000  # Haiku
        )
        synthesis_cost = avg_tokens_per_task * 3 * 15.0 / 1_000_000  # Opus

        total = planning_cost + execution_cost + reflection_cost + synthesis_cost

        return {
            "planning": f"${planning_cost:.4f}",
            "execution": f"${execution_cost:.4f}",
            "reflection": f"${reflection_cost:.4f}",
            "synthesis": f"${synthesis_cost:.4f}",
            "total_estimate": f"${total:.4f}",
            "total_with_buffer": f"${total * 1.5:.4f}",  # 50%バッファ
        }
```

---

## 10. テスト

### 10.1 単体テスト

```python
# 自律エージェントのテスト
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

class TestAutonomousAgent:
    """自律エージェントの単体テスト"""

    @pytest.fixture
    def mock_client(self):
        with patch("anthropic.Anthropic") as mock:
            yield mock.return_value

    @pytest.fixture
    def agent(self, mock_client):
        return AutonomousAgent(
            tools=[],
            max_steps=10,
            max_cost=1.0,
            timeout=60
        )

    def test_create_plan(self, agent, mock_client):
        """計画生成のテスト"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='[{"id": 1, "description": "タスク1"}, '
                 '{"id": 2, "description": "タスク2"}]'
        )]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        plan = agent._create_plan("テスト目標")
        assert len(plan) == 2
        assert plan[0].description == "タスク1"

    def test_select_next_task_respects_dependencies(self, agent):
        """依存関係を考慮したタスク選択"""
        agent.plan = [
            SubTask(id=1, description="タスク1", status=TaskStatus.COMPLETED),
            SubTask(id=2, description="タスク2", dependencies=[1]),
            SubTask(id=3, description="タスク3", dependencies=[2]),
        ]

        next_task = agent._select_next_task()
        assert next_task.id == 2  # 依存先が完了しているタスク2

    def test_select_next_task_blocks_unmet_deps(self, agent):
        """未完了の依存先がある場合はスキップ"""
        agent.plan = [
            SubTask(id=1, description="タスク1", status=TaskStatus.PENDING),
            SubTask(id=2, description="タスク2", dependencies=[1]),
        ]

        next_task = agent._select_next_task()
        assert next_task.id == 1  # 依存なしのタスク1のみ選択可能

    def test_should_stop_cost_limit(self, agent):
        """コスト上限での停止"""
        agent.total_cost = 2.0  # max_cost=1.0を超過
        agent.start_time = time.time()
        assert agent._should_stop() is True

    def test_should_stop_timeout(self, agent):
        """タイムアウトでの停止"""
        agent.start_time = time.time() - 120  # timeout=60を超過
        assert agent._should_stop() is True

class TestWanderingDetector:
    """迷走検出のテスト"""

    def test_detect_repeated_action(self):
        detector = WanderingDetector(window_size=5)
        for _ in range(5):
            detector.record_action("search", "result1", True)

        is_wandering, reason = detector.is_wandering()
        assert is_wandering
        assert "同一アクション" in reason

    def test_detect_consecutive_failures(self):
        detector = WanderingDetector(window_size=5)
        for _ in range(5):
            detector.record_action("different_action", f"result_{_}", False)

        is_wandering, reason = detector.is_wandering()
        assert is_wandering
        assert "連続失敗" in reason

    def test_no_wandering_normal_operation(self):
        detector = WanderingDetector(window_size=5)
        for i in range(5):
            detector.record_action(f"action_{i}", f"result_{i}", True)

        is_wandering, _ = detector.is_wandering()
        assert not is_wandering

class TestSelfEvaluator:
    """自己評価のテスト"""

    @pytest.fixture
    def mock_client(self):
        with patch("anthropic.Anthropic") as mock:
            yield mock.return_value

    def test_evaluate_high_quality(self, mock_client):
        evaluator = SelfEvaluator(mock_client)

        # 高スコアの応答をモック
        responses = [
            MagicMock(content=[MagicMock(text="90")]),   # completeness
            MagicMock(content=[MagicMock(text="85")]),   # quality
            MagicMock(content=[MagicMock(text="改善点1\n改善点2\n改善点3")]),
        ]
        mock_client.messages.create.side_effect = responses

        result = evaluator.evaluate("目標", "出力テキスト")
        assert result["completeness"] == 90
        assert result["quality"] == 85
        assert not result["should_improve"]

    def test_evaluate_low_quality(self, mock_client):
        evaluator = SelfEvaluator(mock_client)

        responses = [
            MagicMock(content=[MagicMock(text="50")]),
            MagicMock(content=[MagicMock(text="40")]),
            MagicMock(content=[MagicMock(text="改善必要")]),
        ]
        mock_client.messages.create.side_effect = responses

        result = evaluator.evaluate("目標", "出力テキスト")
        assert result["should_improve"]

class TestGuardrails:
    """ガードレールのテスト"""

    def test_pii_detection(self):
        agent = GuardedAutonomousAgent(tools=[], max_steps=10)

        # 電話番号
        assert agent._contains_pii("080-1234-5678") is True
        # メールアドレス
        assert agent._contains_pii("user@example.com") is True
        # 通常テキスト
        assert agent._contains_pii("安全なテキスト") is False

    def test_forbidden_commands(self):
        agent = GuardedAutonomousAgent(tools=[], max_steps=10)
        context = {
            "step": 0,
            "cost": 0,
            "elapsed": 0,
            "action": "rm -rf /important",
        }

        violations = agent.check_guardrails(context)
        blocked = [v for v in violations if v["action"] == "block"]
        assert len(blocked) > 0
```

### 10.2 統合テスト

```python
# 統合テスト
class TestAutonomousAgentIntegration:
    """自律エージェントの統合テスト"""

    @pytest.fixture
    def mock_llm_sequence(self):
        """LLM呼び出しのシーケンスをモック"""
        def create_response(text, stop="end_turn"):
            resp = MagicMock()
            resp.content = [MagicMock(text=text, type="text")]
            resp.stop_reason = stop
            resp.usage = MagicMock(input_tokens=100, output_tokens=50)
            return resp

        return create_response

    def test_full_execution_flow(self, mock_llm_sequence):
        """完全な実行フローのテスト"""
        with patch("anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value

            # 計画 → 実行 × 2 → 振り返り × 2 → 統合
            client.messages.create.side_effect = [
                # 計画
                mock_llm_sequence(
                    '[{"id":1,"description":"分析"},{"id":2,"description":"レポート"}]'
                ),
                # タスク1実行
                mock_llm_sequence("分析結果"),
                # タスク1振り返り
                mock_llm_sequence(
                    '{"quality":"good","needs_replan":false,"reason":null,'
                    '"learning":"分析完了","confidence":0.9}'
                ),
                # タスク2実行
                mock_llm_sequence("レポート完成"),
                # タスク2振り返り
                mock_llm_sequence(
                    '{"quality":"good","needs_replan":false,"reason":null,'
                    '"learning":"レポート完了","confidence":0.95}'
                ),
                # 統合
                mock_llm_sequence("最終レポート"),
            ]

            agent = AutonomousAgent(tools=[], max_steps=10)
            result = agent.run("テスト分析を実行")

            assert result == "最終レポート"
            summary = agent.get_execution_summary()
            assert summary["completed"] == 2
            assert summary["failed"] == 0
```

---

## 11. アンチパターン

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
agent = GuardedAutonomousAgent(
    tools=available_tools,
    max_steps=30,                    # ステップ上限
    max_cost=5.0,                    # コスト上限（ドル）
    timeout=600,                     # タイムアウト（秒）
)
```

### アンチパターン3: メモリの不使用

```python
# NG: 毎回ゼロから計画（過去の経験を活用しない）
agent = AutonomousAgent(tools=tools)
result = agent.run("バグを修正して")
# → 同じ失敗を何度も繰り返す

# OK: メモリを活用
agent = MemoryAugmentedAgent(tools=tools)
# 過去の成功/失敗パターンを自動的に活用
result = agent.run("バグを修正して")
```

### アンチパターン4: 単一モデルへの依存

```python
# NG: 全フェーズで最高性能モデル（コスト爆発）
agent = AutonomousAgent(model="claude-opus-4-20250514")

# OK: フェーズごとのモデル最適化
class CostAwareAgent(AutonomousAgent):
    def _create_plan(self, goal):
        # 計画にはbalancedモデル
        self.current_model = CostOptimizer.select_model_for_phase("planning")
        return super()._create_plan(goal)

    def _reflect(self, goal, task, result):
        # 振り返りにはfastモデル
        self.current_model = CostOptimizer.select_model_for_phase("reflection")
        return super()._reflect(goal, task, result)

    def _synthesize(self, goal):
        # 最終統合にはbestモデル
        self.current_model = CostOptimizer.select_model_for_phase("synthesis")
        return super()._synthesize(goal)
```

---

## 12. FAQ

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

### Q4: 自律エージェントの計画はどの程度詳細にすべきか？

初期計画は**粗い粒度**で作成し、各サブゴールの実行前に**詳細化**するのが推奨。理由:
- 全体像を早期に把握できる
- 実行中に得た情報で詳細計画を調整できる
- 過度に詳細な計画は変更コストが高い

### Q5: メモリシステムはどこまで必要か？

自律性レベルにより異なる:
- **L2（半自律）**: ワーキングメモリのみで十分
- **L3（条件付き自律）**: ワーキングメモリ + エピソード記憶
- **L4（完全自律）**: 3層全て（ワーキング + エピソード + セマンティック）

### Q6: 複数の自律エージェントを協調させる場合は？

マルチエージェントパターン（01-multi-agent.md）と組み合わせる。典型的な構成:
- **オーケストレータ（L3）**: 全体計画と進捗管理
- **ワーカーエージェント（L2-L3）**: 個別タスクの実行
- **レビューエージェント（L2）**: 成果物のチェック

---

## まとめ

| 項目 | 内容 |
|------|------|
| コアループ | 計画 → 実行 → 振り返り → （再計画） |
| 目標分解 | 階層的にサブゴール/タスクに分解（HTA） |
| 自己評価 | 完全性・品質・改善余地の多角評価 |
| Reflexion | 失敗を言語化し次の試行に活かす |
| メモリ | 短期（ワーキング）・中期（エピソード）・長期（セマンティック） |
| 再計画 | 失敗からの学習を次の計画に反映 |
| HITL | 重要判断ポイントで人間の承認を挟む |
| ガードレール | ステップ/コスト/時間の上限 + 禁止操作 + PII検出 |
| 迷走検出 | 繰り返し/連続失敗/矛盾行動のパターン検出 |
| コスト最適化 | フェーズ別モデル選択 + 事前見積り |

## 次に読むべきガイド

- [../02-implementation/03-claude-agent-sdk.md](../02-implementation/03-claude-agent-sdk.md) — Claude Agent SDKでの自律エージェント構築
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) — エージェントの評価手法
- [../04-production/01-safety.md](../04-production/01-safety.md) — 安全性とガードレール

## 参考文献

1. Shinn, N. et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) — https://arxiv.org/abs/2303.11366
2. Yao, S. et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023) — https://arxiv.org/abs/2305.10601
3. Wang, G. et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023) — https://arxiv.org/abs/2305.16291
4. Park, J.S. et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023) — https://arxiv.org/abs/2304.03442
5. Anthropic, "Building effective agents" (2024) — https://docs.anthropic.com/en/docs/build-with-claude/agentic
