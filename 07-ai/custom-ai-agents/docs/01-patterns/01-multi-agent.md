# マルチエージェント

> 協調・委任・議論――複数のAIエージェントがチームとして連携し、単体では解決困難な複雑タスクを遂行するマルチエージェントシステムの設計パターン。

## この章で学ぶこと

1. マルチエージェントの3大パターン（協調・委任・議論）の使い分け
2. エージェント間通信とタスク分配の設計方法
3. マルチエージェントシステムのデバッグと最適化手法

---

## 1. なぜマルチエージェントが必要か

```
シングルエージェントの限界

問題: "Webアプリを設計・実装・テスト・デプロイしてほしい"

シングルエージェント:
  1つのLLMが全てを担当 → 専門性の不足、コンテキスト溢れ

マルチエージェント:
  [アーキテクト] → 設計
  [コーダー]     → 実装
  [テスター]     → テスト
  [DevOps]      → デプロイ
  各エージェントが専門性を持ち、協調して完成させる
```

---

## 2. マルチエージェントの3大パターン

### 2.1 パターン全体図

```
マルチエージェント パターン

1. 協調 (Collaborative)
   [A] ←→ [B] ←→ [C]     対等な立場で協力

2. 委任 (Delegation)
   [Manager]                上位が下位にタスク振り分け
   ├── [Worker A]
   ├── [Worker B]
   └── [Worker C]

3. 議論 (Debate)
   [Proposer] → 提案
   [Critic]   → 批判      異なる視点で品質向上
   [Judge]    → 判定
```

### 2.2 協調パターン

```python
# 協調パターン: エージェントが順にタスクを処理
class CollaborativeSystem:
    def __init__(self):
        self.agents = {}
        self.pipeline = []

    def add_agent(self, name: str, agent, role: str):
        self.agents[name] = {"agent": agent, "role": role}

    def set_pipeline(self, pipeline: list[str]):
        """処理パイプラインの順序を設定"""
        self.pipeline = pipeline

    def run(self, task: str) -> str:
        result = task
        context = {"original_task": task, "intermediate_results": []}

        for agent_name in self.pipeline:
            agent_info = self.agents[agent_name]
            print(f"[{agent_name}] 処理中...")

            prompt = f"""
あなたの役割: {agent_info['role']}

元のタスク: {context['original_task']}
これまでの結果: {context['intermediate_results']}

前段階の出力:
{result}

あなたの担当部分を実行してください。
"""
            result = agent_info["agent"].generate(prompt)
            context["intermediate_results"].append({
                "agent": agent_name,
                "output": result
            })

        return result

# 使用例
system = CollaborativeSystem()
system.add_agent("researcher", researcher_llm, "情報を調査・収集する")
system.add_agent("analyst", analyst_llm, "データを分析して洞察を導く")
system.add_agent("writer", writer_llm, "分析結果をレポートにまとめる")
system.set_pipeline(["researcher", "analyst", "writer"])
report = system.run("AI市場の2025年トレンドレポートを作成")
```

### 2.3 委任パターン

```python
# 委任パターン: マネージャーがワーカーにタスクを振り分け
class DelegationSystem:
    def __init__(self, manager_llm, workers: dict):
        self.manager = manager_llm
        self.workers = workers  # {name: {"llm": ..., "skills": [...]}}

    def run(self, goal: str) -> str:
        # Step 1: マネージャーがタスクを分解
        plan = self.manager.generate(f"""
目標: {goal}

利用可能なワーカーとスキル:
{self._format_workers()}

この目標を達成するために、各ワーカーへのタスク割り当てを
JSON形式で出力してください。
""")

        assignments = json.loads(plan)
        results = {}

        # Step 2: 各ワーカーにタスクを委任
        for assignment in assignments:
            worker_name = assignment["worker"]
            task = assignment["task"]
            worker = self.workers[worker_name]

            result = worker["llm"].generate(f"""
あなたのスキル: {worker['skills']}
タスク: {task}
実行してください。
""")
            results[worker_name] = result

        # Step 3: マネージャーが結果を統合
        summary = self.manager.generate(f"""
目標: {goal}
各ワーカーの結果:
{json.dumps(results, ensure_ascii=False, indent=2)}

結果を統合して最終成果物を作成してください。
""")

        return summary
```

### 2.4 議論パターン

```python
# 議論パターン: 複数視点で品質を向上
class DebateSystem:
    def __init__(self, proposer, critic, judge):
        self.proposer = proposer
        self.critic = critic
        self.judge = judge
        self.max_rounds = 3

    def run(self, question: str) -> str:
        proposal = None

        for round_num in range(self.max_rounds):
            # 提案者が回答/改善案を提示
            if proposal is None:
                proposal = self.proposer.generate(
                    f"質問: {question}\n最善の回答を提示してください。"
                )
            else:
                proposal = self.proposer.generate(
                    f"質問: {question}\n"
                    f"前回の提案: {proposal}\n"
                    f"批判: {criticism}\n"
                    f"批判を踏まえて提案を改善してください。"
                )

            # 批判者が評価
            criticism = self.critic.generate(
                f"質問: {question}\n"
                f"提案: {proposal}\n"
                f"この提案の問題点、論理的欠陥、改善点を指摘してください。"
            )

            # 審判者が十分かを判定
            judgment = self.judge.generate(
                f"質問: {question}\n"
                f"提案: {proposal}\n"
                f"批判: {criticism}\n"
                f"提案は十分な品質か？ YES/NO で回答。"
            )

            if "YES" in judgment.upper():
                return proposal

        return proposal  # 最終提案を返す
```

---

## 3. エージェント間通信

### 3.1 通信パターン

```
通信パターン

1. 直接通信 (Direct)
   [A] ──message──> [B]

2. ブロードキャスト (Broadcast)
   [A] ──message──> [B]
       ──message──> [C]
       ──message──> [D]

3. ブラックボード (Blackboard)
   [A] ──write──> +----------+ <──read── [B]
                  | 共有メモリ |
   [C] ──write──> +----------+ <──read── [D]

4. メッセージキュー (Queue)
   [A] ──push──> [Queue] ──pop──> [B]
```

### 3.2 共有メモリパターン

```python
# ブラックボード（共有メモリ）パターン
from threading import Lock
from typing import Any

class Blackboard:
    """エージェント間の共有メモリ"""
    def __init__(self):
        self._data = {}
        self._lock = Lock()
        self._history = []

    def write(self, agent_name: str, key: str, value: Any):
        with self._lock:
            self._data[key] = value
            self._history.append({
                "agent": agent_name,
                "action": "write",
                "key": key,
                "timestamp": time.time()
            })

    def read(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def get_all(self) -> dict:
        with self._lock:
            return self._data.copy()

    def get_updates_since(self, timestamp: float) -> list:
        """指定時刻以降の更新を取得"""
        return [h for h in self._history if h["timestamp"] > timestamp]

# 使用例
board = Blackboard()

# リサーチャーが結果を書き込み
board.write("researcher", "market_data", {"size": "100B", "growth": "15%"})

# アナリストが読み取って分析
data = board.read("market_data")
board.write("analyst", "analysis", f"市場規模{data['size']}、成長率{data['growth']}")
```

---

## 4. パターン比較

### 4.1 3大パターン比較

| 観点 | 協調 | 委任 | 議論 |
|------|------|------|------|
| 構造 | フラット（対等） | 階層型（上下） | 対立型（弁証法） |
| 通信 | パイプライン/共有 | 上→下→上 | 循環 |
| 適用場面 | 工程が明確 | タスク分解可能 | 品質向上が重要 |
| スケーラビリティ | 中 | 高 | 低 |
| コスト | 中 | 中-高 | 高 |
| デバッグ容易性 | 高 | 中 | 低 |
| 代表フレームワーク | LangGraph | CrewAI | AutoGen |

### 4.2 適用タスク別推奨パターン

| タスク | 推奨パターン | 理由 |
|--------|-------------|------|
| ソフトウェア開発 | 委任 | 設計→実装→テストの工程分担 |
| 研究レポート | 協調 | 調査→分析→執筆のパイプライン |
| コードレビュー | 議論 | 複数視点での品質チェック |
| データ分析 | 委任 | 分析タスクの並列分配 |
| 意思決定支援 | 議論 | 賛否両面の検討 |
| カスタマーサポート | 委任 | ルーティング+専門対応 |

---

## 5. CrewAIでのマルチエージェント実装

```python
# CrewAIを使った本格的なマルチエージェントシステム
from crewai import Agent, Task, Crew, Process

# エージェント定義
product_manager = Agent(
    role="プロダクトマネージャー",
    goal="ユーザーニーズに基づいた機能仕様を策定する",
    backstory="SaaS企業で7年の経験を持つPM。ユーザーリサーチとデータ駆動の意思決定が得意。",
    llm="claude-sonnet-4-20250514"
)

architect = Agent(
    role="ソフトウェアアーキテクト",
    goal="スケーラブルで保守性の高いシステム設計を行う",
    backstory="大規模分散システムの設計に10年従事。マイクロサービスとクラウドネイティブの専門家。",
    llm="claude-sonnet-4-20250514"
)

developer = Agent(
    role="シニアデベロッパー",
    goal="設計に基づいて高品質なコードを実装する",
    backstory="フルスタックエンジニア。Python/TypeScript/Goに精通。TDD実践者。",
    llm="claude-sonnet-4-20250514",
    tools=[code_tool, test_tool]
)

# タスク定義
spec_task = Task(
    description="ユーザー認証機能の要件を定義する",
    expected_output="機能仕様書（ユースケース、画面遷移、API仕様）",
    agent=product_manager
)

design_task = Task(
    description="認証機能のシステム設計を行う",
    expected_output="設計書（アーキテクチャ図、DB設計、API設計）",
    agent=architect,
    context=[spec_task]
)

impl_task = Task(
    description="設計に基づいて認証APIを実装する",
    expected_output="実装コード（テスト含む）",
    agent=developer,
    context=[design_task]
)

# 実行
crew = Crew(
    agents=[product_manager, architect, developer],
    tasks=[spec_task, design_task, impl_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

---

## 6. アンチパターン

### アンチパターン1: エージェント数の膨張

```python
# NG: 必要以上のエージェントを作成
crew = Crew(agents=[
    Agent(role="リサーチャー", ...),
    Agent(role="データ収集", ...),       # リサーチャーと重複
    Agent(role="情報分析", ...),          # リサーチャーと重複
    Agent(role="ライター", ...),
    Agent(role="編集者", ...),            # ライターと重複
    Agent(role="校正者", ...),            # 編集者と重複
    Agent(role="デザイナー", ...),
    Agent(role="レビュアー", ...),
])  # 8エージェント = 高コスト + 調整コスト増大

# OK: 必要最小限のエージェント
crew = Crew(agents=[
    Agent(role="リサーチャー", ...),       # 調査+データ収集
    Agent(role="ライター/編集者", ...),    # 執筆+校正
    Agent(role="レビュアー", ...),          # 品質チェック
])  # 3エージェント = 適切な粒度
```

### アンチパターン2: 無限の議論ループ

```python
# NG: 終了条件のない議論
while True:
    proposal = proposer.generate(...)
    criticism = critic.generate(...)
    # 永遠に続く可能性

# OK: 最大ラウンド数 + 合意判定
for round in range(max_rounds := 3):
    proposal = proposer.generate(...)
    criticism = critic.generate(...)
    if judge.is_satisfactory(proposal, criticism):
        break
```

---

## 7. FAQ

### Q1: マルチエージェントのコストはどの程度か？

エージェント数 x ステップ数 x 1回あたりのトークン数 でコストが増加する。例えば3エージェントが各5ステップ実行すると、シングルエージェントの5ステップに比べて **約3倍** のAPIコストがかかる。さらにエージェント間通信のオーバーヘッドも加わる。

### Q2: エージェント間で矛盾が生じた場合の解決策は？

3つのアプローチがある:
1. **多数決**: 複数エージェントの結果を投票で決定
2. **審判エージェント**: 専用のジャッジがどちらが正しいか判断
3. **人間の介入**: 重要な判断は人間に委ねる（Human-in-the-Loop）

### Q3: マルチエージェントのテスト方法は？

- **単体テスト**: 各エージェントを個別にテスト（特定入力→期待出力）
- **結合テスト**: 2-3エージェントの連携をテスト
- **E2Eテスト**: 全体パイプラインの実行（ゴールデンデータセットで）
- **障害注入テスト**: 1つのエージェントが失敗した場合のフォールバック確認

---

## まとめ

| 項目 | 内容 |
|------|------|
| 協調パターン | 対等なエージェントがパイプラインで処理 |
| 委任パターン | マネージャーがワーカーにタスクを分配 |
| 議論パターン | 提案→批判→判定の弁証法的改善 |
| 通信方式 | 直接 / ブロードキャスト / 共有メモリ / キュー |
| 設計原則 | 最小限のエージェント数で最大の効果を |
| コスト注意 | エージェント数 x ステップ数でコスト増大 |

## 次に読むべきガイド

- [02-workflow-agents.md](./02-workflow-agents.md) — ワークフローエージェントの設計
- [03-autonomous-agents.md](./03-autonomous-agents.md) — 自律エージェントの計画と実行
- [../02-implementation/01-langgraph.md](../02-implementation/01-langgraph.md) — LangGraphでの実装

## 参考文献

1. Wu, Q. et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023) — https://arxiv.org/abs/2308.08155
2. CrewAI Documentation — https://docs.crewai.com/
3. Hong, S. et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework" (2023) — https://arxiv.org/abs/2308.00352
