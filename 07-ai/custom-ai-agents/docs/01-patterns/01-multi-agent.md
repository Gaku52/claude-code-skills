# マルチエージェント

> 協調・委任・議論――複数のAIエージェントがチームとして連携し、単体では解決困難な複雑タスクを遂行するマルチエージェントシステムの設計パターン。

## この章で学ぶこと

1. マルチエージェントの3大パターン（協調・委任・議論）の使い分け
2. エージェント間通信とタスク分配の設計方法
3. マルチエージェントシステムのデバッグと最適化手法
4. 本番運用における障害耐性、コスト管理、スケーリング戦略
5. 実務シナリオ別のマルチエージェント構成例

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

### 1.1 マルチエージェントの利点と課題

```
利点                              課題
┌─────────────────────┐      ┌─────────────────────┐
│ 専門性の分離         │      │ 通信オーバーヘッド   │
│ → 各エージェントが   │      │ → エージェント間の   │
│   得意分野に集中     │      │   メッセージ交換コスト│
├─────────────────────┤      ├─────────────────────┤
│ コンテキスト管理     │      │ エラー伝播           │
│ → 各エージェントが   │      │ → 1つの失敗が全体に  │
│   独自のコンテキスト │      │   波及するリスク     │
├─────────────────────┤      ├─────────────────────┤
│ 並行処理             │      │ デバッグの複雑さ     │
│ → 独立タスクの同時   │      │ → 複数エージェントの │
│   実行でスループット↑│      │   相互作用の追跡     │
├─────────────────────┤      ├─────────────────────┤
│ スケーラビリティ     │      │ コスト増大           │
│ → エージェント追加で │      │ → API呼び出し回数が  │
│   機能拡張が容易     │      │   エージェント数倍   │
└─────────────────────┘      └─────────────────────┘
```

### 1.2 シングルからマルチへの移行判断

```python
# マルチエージェント導入の判断基準チェッカー
from dataclasses import dataclass

@dataclass
class TaskComplexityAssessment:
    """タスクの複雑度を評価してマルチエージェントの必要性を判断"""
    task_description: str
    num_distinct_skills: int        # 必要なスキル領域数
    num_steps: int                  # 推定ステップ数
    requires_parallel: bool         # 並行処理が必要か
    requires_debate: bool           # 多角的検討が必要か
    context_window_risk: bool       # コンテキスト溢れのリスク
    quality_critical: bool          # 品質が特に重要か

    @property
    def recommendation(self) -> str:
        score = 0
        if self.num_distinct_skills >= 3:
            score += 2
        elif self.num_distinct_skills >= 2:
            score += 1

        if self.num_steps > 20:
            score += 2
        elif self.num_steps > 10:
            score += 1

        if self.requires_parallel:
            score += 2
        if self.requires_debate:
            score += 1
        if self.context_window_risk:
            score += 2
        if self.quality_critical:
            score += 1

        if score >= 5:
            return "マルチエージェント推奨"
        elif score >= 3:
            return "マルチエージェント検討"
        else:
            return "シングルエージェントで十分"

# 使用例
assessment = TaskComplexityAssessment(
    task_description="Webアプリのフルスタック開発",
    num_distinct_skills=4,     # 設計、フロント、バック、テスト
    num_steps=30,
    requires_parallel=True,
    requires_debate=False,
    context_window_risk=True,
    quality_critical=True
)
print(assessment.recommendation)
# → "マルチエージェント推奨"
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
import anthropic
import json
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class AgentMessage:
    """エージェント間メッセージ"""
    sender: str
    receiver: str
    content: Any
    message_type: str = "task_result"
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class CollaborativeSystem:
    """協調パターン: 対等なエージェントがパイプラインで連携"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.agents: dict[str, dict] = {}
        self.pipeline: list[str] = []
        self.message_log: list[AgentMessage] = []
        self._total_tokens = 0

    def add_agent(self, name: str, role: str,
                  system_prompt: str = "",
                  model: str = "claude-sonnet-4-20250514",
                  tools: list = None):
        """エージェントを追加"""
        self.agents[name] = {
            "role": role,
            "system_prompt": system_prompt or f"あなたは{role}です。",
            "model": model,
            "tools": tools or []
        }

    def set_pipeline(self, pipeline: list[str]):
        """処理パイプラインの順序を設定"""
        for name in pipeline:
            if name not in self.agents:
                raise ValueError(f"エージェント '{name}' が登録されていません")
        self.pipeline = pipeline

    def run(self, task: str) -> dict:
        """パイプラインを実行"""
        result = task
        context = {
            "original_task": task,
            "intermediate_results": [],
            "start_time": time.time()
        }

        for i, agent_name in enumerate(self.pipeline):
            agent_info = self.agents[agent_name]
            logger.info(f"[{i+1}/{len(self.pipeline)}] {agent_name} 処理中...")

            prompt = f"""
あなたの役割: {agent_info['role']}

元のタスク: {context['original_task']}

{'これまでの結果:' if context['intermediate_results'] else ''}
{self._format_previous_results(context['intermediate_results'])}

前段階の出力:
{result}

あなたの担当部分を実行してください。
次のエージェントが理解できるように、結果を構造化して出力してください。
"""
            response = self.client.messages.create(
                model=agent_info["model"],
                max_tokens=4096,
                system=agent_info["system_prompt"],
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text
            self._total_tokens += response.usage.input_tokens + response.usage.output_tokens

            # メッセージログに記録
            msg = AgentMessage(
                sender=agent_name,
                receiver=self.pipeline[i + 1] if i + 1 < len(self.pipeline) else "output",
                content=result[:500],
                metadata={
                    "step": i + 1,
                    "tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            )
            self.message_log.append(msg)

            context["intermediate_results"].append({
                "agent": agent_name,
                "role": agent_info["role"],
                "output": result[:1000]
            })

        elapsed = time.time() - context["start_time"]
        return {
            "result": result,
            "steps": len(self.pipeline),
            "total_tokens": self._total_tokens,
            "elapsed_seconds": round(elapsed, 2),
            "message_log": self.message_log
        }

    def _format_previous_results(self, results: list) -> str:
        if not results:
            return ""
        formatted = []
        for r in results[-3:]:  # 直近3件のみ
            formatted.append(f"[{r['agent']}({r['role']})]: {r['output'][:300]}")
        return "\n".join(formatted)


# 使用例
system = CollaborativeSystem()
system.add_agent(
    "researcher",
    role="情報リサーチャー",
    system_prompt="あなたは優秀なリサーチャーです。信頼できる情報源から事実を収集し、構造化して提示してください。"
)
system.add_agent(
    "analyst",
    role="データアナリスト",
    system_prompt="あなたはデータアナリストです。収集されたデータからパターンやトレンドを発見し、数値に基づく洞察を導いてください。"
)
system.add_agent(
    "writer",
    role="レポートライター",
    system_prompt="あなたはビジネスライターです。分析結果を読みやすいレポートにまとめてください。エグゼクティブサマリー付きで。"
)
system.set_pipeline(["researcher", "analyst", "writer"])
report = system.run("AI市場の2025年トレンドレポートを作成")
```

### 2.3 委任パターン

```python
# 委任パターン: マネージャーがワーカーにタスクを振り分け
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DelegationSystem:
    """委任パターン: マネージャーが計画し、ワーカーが実行"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.workers: dict[str, dict] = {}
        self.manager_system_prompt = """
あなたはプロジェクトマネージャーです。
与えられた目標を達成するために：
1. タスクを適切な粒度に分解する
2. 各ワーカーの専門性を考慮してタスクを割り当てる
3. 依存関係を特定し、実行順序を決定する
4. ワーカーの結果を統合して最終成果物を作成する
"""

    def add_worker(self, name: str, skills: list[str],
                   system_prompt: str = "",
                   model: str = "claude-sonnet-4-20250514"):
        """ワーカーエージェントを追加"""
        self.workers[name] = {
            "skills": skills,
            "system_prompt": system_prompt or f"あなたは{', '.join(skills)}のスペシャリストです。",
            "model": model
        }

    def run(self, goal: str) -> dict:
        """マネージャーが計画し、ワーカーが実行"""
        start_time = time.time()

        # Step 1: マネージャーがタスクを分解・割り当て
        plan = self._create_plan(goal)
        logger.info(f"計画: {len(plan.get('assignments', []))}タスク")

        # Step 2: 依存関係に基づいてタスクを実行
        results = self._execute_plan(plan)

        # Step 3: マネージャーが結果を統合
        final = self._integrate_results(goal, results)

        elapsed = time.time() - start_time
        return {
            "result": final,
            "plan": plan,
            "worker_results": results,
            "elapsed_seconds": round(elapsed, 2)
        }

    def _create_plan(self, goal: str) -> dict:
        """マネージャーがタスクを分解して計画を作成"""
        worker_descriptions = "\n".join(
            f"- {name}: スキル={w['skills']}"
            for name, w in self.workers.items()
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.manager_system_prompt,
            messages=[{"role": "user", "content": f"""
目標: {goal}

利用可能なワーカー:
{worker_descriptions}

この目標を達成するための計画をJSON形式で出力してください。
フォーマット:
{{
  "assignments": [
    {{
      "worker": "ワーカー名",
      "task": "具体的なタスク内容",
      "priority": 1,
      "depends_on": []
    }}
  ]
}}

注意:
- depends_onは依存するタスクのインデックス（0始まり）のリスト
- priorityは1（高）から3（低）
- 並行実行可能なタスクはdepends_onを空にする
"""}]
        )

        text = response.content[0].text
        # JSONを抽出
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # フォールバック: 各ワーカーに均等に割り当て
            return {
                "assignments": [
                    {"worker": name, "task": goal, "priority": 1, "depends_on": []}
                    for name in self.workers
                ]
            }

    def _execute_plan(self, plan: dict) -> dict:
        """依存関係を考慮してタスクを実行"""
        assignments = plan.get("assignments", [])
        results = {}
        completed = set()

        # 依存関係順にソート
        remaining = list(range(len(assignments)))

        while remaining:
            # 依存関係が解決されたタスクを実行
            executable = [
                i for i in remaining
                if all(d in completed for d in assignments[i].get("depends_on", []))
            ]

            if not executable:
                logger.error("依存関係のデッドロックを検出")
                break

            # 並行実行可能なタスクをまとめて実行
            for i in executable:
                assignment = assignments[i]
                worker_name = assignment["worker"]
                task = assignment["task"]

                if worker_name not in self.workers:
                    results[i] = {"error": f"ワーカー '{worker_name}' が見つかりません"}
                    completed.add(i)
                    remaining.remove(i)
                    continue

                worker = self.workers[worker_name]

                # 依存タスクの結果をコンテキストとして提供
                dep_context = ""
                for dep_idx in assignment.get("depends_on", []):
                    if dep_idx in results:
                        dep_result = results[dep_idx]
                        dep_task = assignments[dep_idx]["task"]
                        dep_context += f"\n[前提タスク] {dep_task}\n結果: {str(dep_result)[:500]}\n"

                response = self.client.messages.create(
                    model=worker["model"],
                    max_tokens=4096,
                    system=worker["system_prompt"],
                    messages=[{"role": "user", "content": f"""
タスク: {task}
{dep_context}
上記のタスクを実行し、結果を出力してください。
"""}]
                )

                results[i] = {
                    "worker": worker_name,
                    "task": task,
                    "result": response.content[0].text
                }
                completed.add(i)
                remaining.remove(i)

        return results

    def _integrate_results(self, goal: str, results: dict) -> str:
        """マネージャーがワーカーの結果を統合"""
        results_text = ""
        for i, result in sorted(results.items()):
            if isinstance(result, dict):
                results_text += f"""
[{result.get('worker', 'unknown')}] タスク: {result.get('task', 'N/A')}
結果:
{str(result.get('result', 'N/A'))[:1000]}
---
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.manager_system_prompt,
            messages=[{"role": "user", "content": f"""
目標: {goal}

各ワーカーの結果:
{results_text}

上記の結果を統合して、目標に対する包括的な最終成果物を作成してください。
矛盾がある場合は指摘し、最善の統合を行ってください。
"""}]
        )
        return response.content[0].text


# 使用例
delegation = DelegationSystem()
delegation.add_worker(
    "frontend_dev",
    skills=["React", "TypeScript", "CSS", "UI/UX"],
    system_prompt="あなたはフロントエンドエンジニアです。React/TypeScriptでの実装を行います。"
)
delegation.add_worker(
    "backend_dev",
    skills=["Python", "FastAPI", "PostgreSQL", "Redis"],
    system_prompt="あなたはバックエンドエンジニアです。Python/FastAPIでの実装を行います。"
)
delegation.add_worker(
    "qa_engineer",
    skills=["テスト設計", "Pytest", "Playwright", "負荷テスト"],
    system_prompt="あなたはQAエンジニアです。テスト設計と自動テストの実装を行います。"
)

result = delegation.run("ユーザー認証機能（登録・ログイン・パスワードリセット）を実装")
```

### 2.4 議論パターン

```python
# 議論パターン: 複数視点で品質を向上
class DebateSystem:
    """弁証法的な議論で回答品質を向上"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_rounds = 3
        self.debate_log: list[dict] = []

    def run(self, question: str) -> dict:
        """提案→批判→改善のサイクルで品質を向上"""
        proposal = None
        criticism = None

        for round_num in range(self.max_rounds):
            logger.info(f"議論ラウンド {round_num + 1}/{self.max_rounds}")

            # 提案者が回答/改善案を提示
            if proposal is None:
                proposal = self._generate_proposal(question)
            else:
                proposal = self._improve_proposal(question, proposal, criticism)

            # 批判者が評価
            criticism = self._generate_criticism(question, proposal)

            # ログ記録
            self.debate_log.append({
                "round": round_num + 1,
                "proposal": proposal[:500],
                "criticism": criticism[:500]
            })

            # 審判者が十分かを判定
            judgment = self._judge(question, proposal, criticism)

            if judgment["is_satisfactory"]:
                logger.info(f"ラウンド {round_num + 1} で合意に達しました")
                return {
                    "result": proposal,
                    "rounds": round_num + 1,
                    "confidence": judgment["confidence"],
                    "debate_log": self.debate_log
                }

        return {
            "result": proposal,
            "rounds": self.max_rounds,
            "confidence": "最大ラウンド到達",
            "debate_log": self.debate_log
        }

    def _generate_proposal(self, question: str) -> str:
        """初期提案を生成"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="あなたは問題解決の専門家です。論理的で包括的な回答を提示してください。根拠を明示し、具体例を含めてください。",
            messages=[{"role": "user", "content": f"質問: {question}\n\n最善の回答を提示してください。"}]
        )
        return response.content[0].text

    def _improve_proposal(self, question: str, proposal: str, criticism: str) -> str:
        """批判を踏まえて提案を改善"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="あなたは問題解決の専門家です。批判を真摯に受け止め、具体的な改善を行ってください。",
            messages=[{"role": "user", "content": f"""
質問: {question}

前回の提案:
{proposal}

受けた批判:
{criticism}

批判を踏まえて提案を改善してください。
改善点を明示し、批判への対応を具体的に示してください。
"""}]
        )
        return response.content[0].text

    def _generate_criticism(self, question: str, proposal: str) -> str:
        """批判を生成"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="""あなたは批判的思考の専門家です。提案の弱点を建設的に指摘してください。
以下の観点で評価してください:
1. 論理的整合性
2. 事実の正確性
3. 見落としている観点
4. 実現可能性
5. リスクと副作用""",
            messages=[{"role": "user", "content": f"""
質問: {question}
提案: {proposal}

この提案の問題点、論理的欠陥、改善点を指摘してください。
建設的な批判を心がけてください。
"""}]
        )
        return response.content[0].text

    def _judge(self, question: str, proposal: str, criticism: str) -> dict:
        """審判が品質を判定"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system="あなたは公平な審判です。提案と批判を客観的に評価してください。",
            messages=[{"role": "user", "content": f"""
質問: {question}
提案: {proposal[:1000]}
批判: {criticism[:1000]}

以下の形式で評価してください:
判定: [PASS/FAIL]
信頼度: [高/中/低]
理由: [1行で]
"""}]
        )

        text = response.content[0].text
        is_pass = "PASS" in text.upper()
        confidence = "高" if "高" in text else ("中" if "中" in text else "低")

        return {
            "is_satisfactory": is_pass,
            "confidence": confidence,
            "raw_judgment": text
        }
```

### 2.5 ハイブリッドパターン

```python
# 委任 + 議論のハイブリッド
class HybridSystem:
    """委任パターンの各段階に議論パターンを組み込む"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.delegation = DelegationSystem(model)
        self.debate = DebateSystem(model)
        self.client = anthropic.Anthropic()
        self.model = model

    def run(self, goal: str, critical_steps: list[int] = None) -> dict:
        """委任で分担し、重要なステップでは議論で品質確保"""
        # Step 1: 計画の策定（議論で計画を精査）
        plan_proposal = self.delegation._create_plan(goal)

        # 計画自体を議論で検証
        plan_question = f"以下の計画は目標 '{goal}' を達成するのに適切ですか？\n計画: {json.dumps(plan_proposal, ensure_ascii=False)}"
        plan_review = self.debate.run(plan_question)

        # Step 2: 各タスクを実行
        results = self.delegation._execute_plan(plan_proposal)

        # Step 3: 重要なステップの結果を議論で精査
        if critical_steps:
            for step_idx in critical_steps:
                if step_idx in results:
                    step_result = results[step_idx]
                    review_question = (
                        f"以下のタスク結果の品質は十分ですか？\n"
                        f"タスク: {step_result.get('task', 'N/A')}\n"
                        f"結果: {str(step_result.get('result', ''))[:2000]}"
                    )
                    review = self.debate.run(review_question)
                    results[step_idx]["quality_review"] = review

        # Step 4: 結果統合
        final = self.delegation._integrate_results(goal, results)

        return {
            "result": final,
            "plan": plan_proposal,
            "plan_review": plan_review,
            "worker_results": results
        }
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

5. パブリッシュ/サブスクライブ (Pub/Sub)
   [A] ──publish "topic.x"──> [Bus] ──> [B] (subscribed: topic.x)
                                    ──> [C] (subscribed: topic.*)
```

### 3.2 共有メモリパターン

```python
# ブラックボード（共有メモリ）パターン
from threading import Lock
from typing import Any

class Blackboard:
    """エージェント間の共有メモリ"""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = Lock()
        self._history: list[dict] = []
        self._subscribers: dict[str, list[callable]] = {}

    def write(self, agent_name: str, key: str, value: Any):
        """データを書き込み、サブスクライバーに通知"""
        with self._lock:
            self._data[key] = value
            self._history.append({
                "agent": agent_name,
                "action": "write",
                "key": key,
                "timestamp": time.time()
            })

        # サブスクライバーに通知
        for pattern, callbacks in self._subscribers.items():
            if self._match_pattern(pattern, key):
                for callback in callbacks:
                    callback(agent_name, key, value)

    def read(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def read_many(self, keys: list[str]) -> dict:
        """複数キーを一括読み取り"""
        with self._lock:
            return {k: self._data.get(k) for k in keys}

    def get_all(self) -> dict:
        with self._lock:
            return self._data.copy()

    def subscribe(self, key_pattern: str, callback: callable):
        """キーパターンに対するサブスクリプション"""
        if key_pattern not in self._subscribers:
            self._subscribers[key_pattern] = []
        self._subscribers[key_pattern].append(callback)

    def get_updates_since(self, timestamp: float) -> list:
        """指定時刻以降の更新を取得"""
        return [h for h in self._history if h["timestamp"] > timestamp]

    def get_agent_contributions(self, agent_name: str) -> list:
        """特定エージェントの書き込み履歴"""
        return [h for h in self._history if h["agent"] == agent_name]

    def _match_pattern(self, pattern: str, key: str) -> bool:
        """簡易パターンマッチ（*はワイルドカード）"""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return pattern == key

    def summary(self) -> dict:
        """ブラックボードの状態サマリー"""
        return {
            "total_entries": len(self._data),
            "total_writes": len(self._history),
            "agents_involved": list(set(h["agent"] for h in self._history)),
            "keys": list(self._data.keys())
        }


# ブラックボードを使ったマルチエージェントシステム
class BlackboardMultiAgent:
    """ブラックボードベースのマルチエージェントシステム"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.board = Blackboard()
        self.agents: dict[str, dict] = {}

    def add_agent(self, name: str, role: str,
                  watches: list[str] = None,
                  produces: list[str] = None):
        """エージェントを追加"""
        self.agents[name] = {
            "role": role,
            "watches": watches or [],
            "produces": produces or []
        }

    def run(self, goal: str, max_iterations: int = 10) -> dict:
        """ブラックボードを介してエージェントが協調"""
        self.board.write("system", "goal", goal)
        self.board.write("system", "status", "running")

        for iteration in range(max_iterations):
            any_progress = False

            for name, agent_info in self.agents.items():
                # エージェントが監視しているキーの値を取得
                watched_data = {}
                for key in agent_info["watches"]:
                    value = self.board.read(key)
                    if value is not None:
                        watched_data[key] = value

                # 必要なデータが揃っていない場合はスキップ
                if not watched_data:
                    continue

                # エージェントが既に結果を生成済みかチェック
                already_done = all(
                    self.board.read(k) is not None
                    for k in agent_info["produces"]
                )
                if already_done:
                    continue

                # エージェントを実行
                result = self._run_agent(name, agent_info, watched_data, goal)

                # 結果をブラックボードに書き込み
                for key in agent_info["produces"]:
                    if key in result:
                        self.board.write(name, key, result[key])
                        any_progress = True

            if not any_progress:
                break

        self.board.write("system", "status", "completed")
        return {
            "result": self.board.get_all(),
            "iterations": iteration + 1,
            "board_summary": self.board.summary()
        }

    def _run_agent(self, name: str, agent_info: dict,
                   watched_data: dict, goal: str) -> dict:
        """個別エージェントを実行"""
        context = json.dumps(watched_data, ensure_ascii=False, default=str)[:3000]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"あなたは{agent_info['role']}です。",
            messages=[{"role": "user", "content": f"""
目標: {goal}

利用可能なデータ:
{context}

あなたの成果物キー: {agent_info['produces']}

成果物をJSON形式で出力してください。
キーは上記の成果物キーを使用してください。
"""}]
        )

        text = response.content[0].text
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            return json.loads(text)
        except json.JSONDecodeError:
            return {agent_info["produces"][0]: text if agent_info["produces"] else "result"}
```

### 3.3 メッセージキューパターン

```python
from collections import deque
from threading import Lock, Event
from typing import Optional

class MessageQueue:
    """優先度付きメッセージキュー"""

    def __init__(self, max_size: int = 1000):
        self._queue: deque[AgentMessage] = deque(maxlen=max_size)
        self._lock = Lock()
        self._not_empty = Event()
        self._processed: list[AgentMessage] = []

    def push(self, message: AgentMessage, priority: int = 0):
        """メッセージを追加（priority: 0=通常、1=高、2=緊急）"""
        with self._lock:
            if priority >= 2:
                self._queue.appendleft(message)  # 先頭に挿入
            else:
                self._queue.append(message)
            self._not_empty.set()

    def pop(self, receiver: Optional[str] = None,
            timeout: float = None) -> Optional[AgentMessage]:
        """メッセージを取得（receiverでフィルタ可能）"""
        with self._lock:
            if receiver:
                # 特定受信者向けのメッセージを検索
                for i, msg in enumerate(self._queue):
                    if msg.receiver == receiver:
                        del self._queue[i]
                        self._processed.append(msg)
                        return msg
                return None
            elif self._queue:
                msg = self._queue.popleft()
                self._processed.append(msg)
                return msg
            return None

    def pending_count(self, receiver: Optional[str] = None) -> int:
        """未処理メッセージ数"""
        with self._lock:
            if receiver:
                return sum(1 for m in self._queue if m.receiver == receiver)
            return len(self._queue)

    def stats(self) -> dict:
        """キューの統計"""
        return {
            "pending": len(self._queue),
            "processed": len(self._processed),
            "senders": list(set(m.sender for m in self._processed)),
            "receivers": list(set(m.receiver for m in self._processed))
        }


class QueueBasedMultiAgent:
    """メッセージキューベースのマルチエージェントシステム"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.queue = MessageQueue()
        self.agents: dict[str, dict] = {}
        self.results: dict[str, list] = {}

    def add_agent(self, name: str, role: str, handles: list[str]):
        """エージェントを追加
        handles: 処理するメッセージタイプのリスト
        """
        self.agents[name] = {
            "role": role,
            "handles": handles
        }
        self.results[name] = []

    def send_message(self, sender: str, receiver: str,
                     content: Any, message_type: str = "task"):
        """メッセージを送信"""
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type=message_type
        )
        self.queue.push(msg)

    def process_messages(self, max_cycles: int = 20) -> dict:
        """全メッセージを処理"""
        for cycle in range(max_cycles):
            if self.queue.pending_count() == 0:
                break

            for name, agent_info in self.agents.items():
                msg = self.queue.pop(receiver=name)
                if msg and msg.message_type in agent_info["handles"]:
                    result = self._process_message(name, agent_info, msg)
                    self.results[name].append({
                        "message": msg,
                        "result": result
                    })

        return {
            "results": self.results,
            "queue_stats": self.queue.stats(),
            "cycles": cycle + 1
        }

    def _process_message(self, agent_name: str, agent_info: dict,
                         message: AgentMessage) -> str:
        """エージェントがメッセージを処理"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"あなたは{agent_info['role']}です。",
            messages=[{"role": "user", "content": f"""
送信者: {message.sender}
メッセージタイプ: {message.message_type}
内容: {message.content}

このメッセージに対して、あなたの役割に基づいて適切に処理してください。
"""}]
        )
        return response.content[0].text
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
| 障害耐性 | 低（直列） | 中（並列可） | 中 |
| 品質保証 | 各段階で検証 | 統合時に検証 | 反復で向上 |
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
| セキュリティ監査 | ハイブリッド | 委任で分担、議論で精査 |
| コンテンツ制作 | 協調 | 企画→制作→校正の直列フロー |
| 翻訳品質保証 | 議論 | 複数翻訳者の比較検討 |

### 4.3 通信パターン比較

| 通信方式 | レイテンシ | スケーラビリティ | 実装複雑度 | 適用場面 |
|---------|----------|----------------|-----------|---------|
| 直接通信 | 最低 | 低 | 最低 | 2-3エージェント |
| ブラックボード | 低 | 中 | 中 | 共有状態が必要 |
| メッセージキュー | 中 | 高 | 中 | 非同期処理 |
| Pub/Sub | 中 | 最高 | 高 | イベント駆動 |
| ブロードキャスト | 高 | 低 | 低 | 全員への通知 |

---

## 5. フレームワーク実装

### 5.1 CrewAIでのマルチエージェント

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

### 5.2 LangGraphでのマルチエージェント

```python
# LangGraphを使ったグラフベースのマルチエージェント
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class TeamState(TypedDict):
    """チーム全体の共有状態"""
    task: str
    plan: list[str]
    research: str
    draft: str
    review: str
    final: str
    messages: Annotated[list[str], operator.add]  # 累積
    current_agent: str
    iteration: int

def planner_node(state: TeamState) -> dict:
    """プランナーノード: タスクを分解"""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="あなたはプロジェクトプランナーです。",
        messages=[{"role": "user", "content": f"""
タスク: {state['task']}
このタスクを3-5ステップに分解してください。
番号付きリストで出力:
"""}]
    )
    plan_text = response.content[0].text
    steps = [line.strip() for line in plan_text.split("\n")
             if line.strip() and line.strip()[0].isdigit()]
    return {
        "plan": steps,
        "messages": [f"[Planner] {len(steps)}ステップの計画を作成"],
        "current_agent": "researcher"
    }

def researcher_node(state: TeamState) -> dict:
    """リサーチャーノード: 情報収集"""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="あなたはリサーチャーです。事実に基づいた情報を収集してください。",
        messages=[{"role": "user", "content": f"""
タスク: {state['task']}
計画: {state['plan']}

上記の計画に基づいて、必要な情報をリサーチしてください。
"""}]
    )
    return {
        "research": response.content[0].text,
        "messages": [f"[Researcher] リサーチ完了"],
        "current_agent": "writer"
    }

def writer_node(state: TeamState) -> dict:
    """ライターノード: ドラフト作成"""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system="あなたはプロフェッショナルライターです。",
        messages=[{"role": "user", "content": f"""
タスク: {state['task']}
リサーチ結果: {state['research'][:2000]}
{'レビューコメント: ' + state['review'] if state.get('review') else ''}

上記に基づいてドラフトを作成してください。
"""}]
    )
    return {
        "draft": response.content[0].text,
        "messages": [f"[Writer] ドラフト作成完了"],
        "current_agent": "reviewer"
    }

def reviewer_node(state: TeamState) -> dict:
    """レビュアーノード: 品質チェック"""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="あなたは品質レビュアーです。建設的なフィードバックを提供してください。",
        messages=[{"role": "user", "content": f"""
タスク: {state['task']}
ドラフト: {state['draft'][:2000]}

品質を評価し、以下の形式で回答:
品質: [合格/要改善]
フィードバック: [具体的な改善点]
"""}]
    )
    review = response.content[0].text
    return {
        "review": review,
        "messages": [f"[Reviewer] レビュー完了"],
        "iteration": state.get("iteration", 0) + 1
    }

def should_continue(state: TeamState) -> str:
    """レビュー結果に基づいてフローを決定"""
    if state.get("iteration", 0) >= 3:
        return "finalize"
    if state.get("review") and "合格" in state["review"]:
        return "finalize"
    return "revise"

def finalize_node(state: TeamState) -> dict:
    """最終化ノード"""
    return {
        "final": state.get("draft", ""),
        "messages": [f"[System] 最終化完了（{state.get('iteration', 0)}回のイテレーション）"]
    }

# グラフの構築
workflow = StateGraph(TeamState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("finalizer", finalize_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {"revise": "writer", "finalize": "finalizer"}
)
workflow.add_edge("finalizer", END)

# コンパイルと実行
app = workflow.compile()
result = app.invoke({
    "task": "AIエージェントの未来に関するレポートを作成",
    "plan": [],
    "research": "",
    "draft": "",
    "review": "",
    "final": "",
    "messages": [],
    "current_agent": "planner",
    "iteration": 0
})
```

---

## 6. 障害耐性設計

### 6.1 エージェント障害への対応

```python
class ResilientMultiAgent:
    """障害耐性のあるマルチエージェントシステム"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.agents: dict[str, dict] = {}
        self.fallback_agents: dict[str, str] = {}  # agent → fallback_agent

    def add_agent(self, name: str, role: str,
                  fallback: Optional[str] = None,
                  max_retries: int = 3):
        """フォールバック付きでエージェントを追加"""
        self.agents[name] = {
            "role": role,
            "max_retries": max_retries
        }
        if fallback:
            self.fallback_agents[name] = fallback

    def execute_agent(self, name: str, task: str,
                      context: dict = None) -> dict:
        """障害耐性付きでエージェントを実行"""
        agent = self.agents.get(name)
        if not agent:
            return {"error": f"エージェント '{name}' が見つかりません"}

        # リトライロジック
        for attempt in range(agent["max_retries"]):
            try:
                result = self._call_agent(name, agent, task, context)
                if result and "error" not in result:
                    return result
            except Exception as e:
                logger.warning(
                    f"{name} 失敗（{attempt + 1}/{agent['max_retries']}）: {e}"
                )
                if attempt < agent["max_retries"] - 1:
                    time.sleep(2 ** attempt)

        # フォールバックエージェントを試行
        fallback_name = self.fallback_agents.get(name)
        if fallback_name and fallback_name in self.agents:
            logger.info(f"{name} → {fallback_name} にフォールバック")
            return self.execute_agent(fallback_name, task, context)

        return {
            "error": f"{name} が{agent['max_retries']}回試行後に失敗",
            "partial_result": "フォールバックなし"
        }

    def _call_agent(self, name: str, agent: dict,
                    task: str, context: dict = None) -> dict:
        """エージェントを呼び出す"""
        context_text = json.dumps(context or {}, ensure_ascii=False, default=str)[:2000]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=f"あなたは{agent['role']}です。",
            messages=[{"role": "user", "content": f"""
タスク: {task}
コンテキスト: {context_text}

タスクを実行し、結果を出力してください。
"""}]
        )
        return {
            "agent": name,
            "result": response.content[0].text
        }
```

### 6.2 タイムアウトとサーキットブレーカー

```python
from enum import Enum
from collections import deque

class CircuitState(Enum):
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 障害中（リクエスト遮断）
    HALF_OPEN = "half_open" # 回復テスト中

class CircuitBreaker:
    """サーキットブレーカー: 連続障害時にリクエストを遮断"""

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_successes = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
        return self._state

    def allow_request(self) -> bool:
        """リクエストを許可するか"""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return True
        else:  # OPEN
            return False

    def record_success(self):
        """成功を記録"""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_max:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        else:
            self._failure_count = 0

    def record_failure(self):
        """失敗を記録"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"サーキットブレーカーOPEN: {self._failure_count}回連続失敗")


class ProtectedAgent:
    """サーキットブレーカー付きエージェント"""

    def __init__(self, name: str, role: str):
        self.client = anthropic.Anthropic()
        self.name = name
        self.role = role
        self.circuit = CircuitBreaker()

    def execute(self, task: str) -> dict:
        """サーキットブレーカー付きで実行"""
        if not self.circuit.allow_request():
            return {
                "error": f"{self.name} はサーキットブレーカーにより一時停止中",
                "circuit_state": self.circuit.state.value,
                "retry_after": self.circuit.recovery_timeout
            }

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=f"あなたは{self.role}です。",
                messages=[{"role": "user", "content": task}]
            )
            self.circuit.record_success()
            return {"agent": self.name, "result": response.content[0].text}
        except Exception as e:
            self.circuit.record_failure()
            return {"error": str(e), "circuit_state": self.circuit.state.value}
```

---

## 7. コスト管理とモニタリング

### 7.1 マルチエージェントのコスト追跡

```python
class MultiAgentCostTracker:
    """マルチエージェントシステムのコスト追跡"""

    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-20250514": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, budget_usd: float = 10.0):
        self.budget = budget_usd
        self.agent_costs: dict[str, float] = {}
        self._records: list[dict] = []

    def record(self, agent_name: str, model: str,
               input_tokens: int, output_tokens: int):
        """API呼び出しを記録"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] +
                output_tokens * pricing["output"]) / 1_000_000

        self.agent_costs[agent_name] = \
            self.agent_costs.get(agent_name, 0) + cost

        self._records.append({
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": time.time()
        })

    @property
    def total_cost(self) -> float:
        return sum(self.agent_costs.values())

    @property
    def remaining_budget(self) -> float:
        return self.budget - self.total_cost

    def is_within_budget(self) -> bool:
        return self.total_cost < self.budget

    def cost_report(self) -> dict:
        """コストレポート"""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "budget_usd": self.budget,
            "remaining_usd": round(self.remaining_budget, 4),
            "utilization": f"{self.total_cost / self.budget:.1%}",
            "agent_breakdown": {
                name: {
                    "cost_usd": round(cost, 4),
                    "share": f"{cost / self.total_cost:.1%}" if self.total_cost > 0 else "0%"
                }
                for name, cost in sorted(
                    self.agent_costs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            },
            "total_api_calls": len(self._records)
        }

    def cost_optimization_suggestions(self) -> list[str]:
        """コスト最適化の提案"""
        suggestions = []

        # 最もコストの高いエージェントを特定
        if self.agent_costs:
            max_agent = max(self.agent_costs, key=self.agent_costs.get)
            max_cost = self.agent_costs[max_agent]
            if max_cost > self.total_cost * 0.5:
                suggestions.append(
                    f"{max_agent} が総コストの{max_cost/self.total_cost:.0%}を占めています。"
                    f"Haikuモデルへの切り替えを検討してください。"
                )

        # Sonnetの使用率が高い場合
        sonnet_calls = sum(1 for r in self._records if "sonnet" in r["model"])
        if sonnet_calls > len(self._records) * 0.8:
            suggestions.append(
                "API呼び出しの80%以上がSonnetです。"
                "簡単なタスク（分類、要約）はHaikuに切り替えることで"
                "コストを1/10以下に削減できます。"
            )

        # キャッシュの活用
        suggestions.append(
            "同じ入力パターンのAPI呼び出しにキャッシュを導入すると"
            "重複コストを削減できます。"
        )

        return suggestions
```

### 7.2 実行モニタリング

```python
class MultiAgentMonitor:
    """マルチエージェントシステムのモニタリング"""

    def __init__(self):
        self._events: list[dict] = []
        self._agent_metrics: dict[str, dict] = {}

    def record_event(self, agent_name: str, event_type: str,
                     details: dict = None):
        """イベントを記録"""
        event = {
            "agent": agent_name,
            "type": event_type,
            "timestamp": time.time(),
            "details": details or {}
        }
        self._events.append(event)

        # エージェントメトリクスを更新
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = {
                "total_calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0
            }
        metrics = self._agent_metrics[agent_name]
        metrics["total_calls"] += 1
        if event_type == "success":
            metrics["successes"] += 1
        elif event_type == "failure":
            metrics["failures"] += 1
        if "elapsed_ms" in (details or {}):
            metrics["total_time_ms"] += details["elapsed_ms"]

    def dashboard(self) -> dict:
        """ダッシュボードデータ"""
        return {
            "total_events": len(self._events),
            "agents": {
                name: {
                    **metrics,
                    "success_rate": f"{metrics['successes'] / metrics['total_calls']:.1%}"
                                    if metrics['total_calls'] > 0 else "N/A",
                    "avg_time_ms": round(
                        metrics['total_time_ms'] / metrics['total_calls'], 1
                    ) if metrics['total_calls'] > 0 else 0
                }
                for name, metrics in self._agent_metrics.items()
            },
            "recent_events": self._events[-10:]
        }

    def bottleneck_analysis(self) -> dict:
        """ボトルネック分析"""
        if not self._agent_metrics:
            return {"message": "データなし"}

        # 最も遅いエージェント
        slowest = max(
            self._agent_metrics.items(),
            key=lambda x: x[1]["total_time_ms"] / max(x[1]["total_calls"], 1)
        )

        # 最も失敗の多いエージェント
        most_failures = max(
            self._agent_metrics.items(),
            key=lambda x: x[1]["failures"]
        )

        return {
            "slowest_agent": {
                "name": slowest[0],
                "avg_time_ms": round(
                    slowest[1]["total_time_ms"] / max(slowest[1]["total_calls"], 1), 1
                )
            },
            "most_failing_agent": {
                "name": most_failures[0],
                "failure_count": most_failures[1]["failures"]
            }
        }
```

---

## 8. アンチパターン

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

### アンチパターン3: エージェント間の暗黙の依存

```python
# NG: グローバル変数を介した暗黙の通信
global_state = {}

def agent_a():
    global_state["result_a"] = "..."  # どこから参照されるか不明

def agent_b():
    data = global_state["result_a"]  # agent_aが先に実行されている前提

# OK: 明示的な依存関係の定義
class ExplicitDependency:
    def run(self):
        result_a = self.agent_a.execute(task)
        result_b = self.agent_b.execute(task, depends_on={"a": result_a})
        return self.integrate(result_a, result_b)
```

### アンチパターン4: エラー情報の伝播不足

```python
# NG: エラーを握りつぶして次のエージェントに渡す
def pipeline(task):
    result_a = agent_a.run(task)  # エラーかもしれない
    result_b = agent_b.run(result_a)  # 不正な入力で連鎖的に失敗
    return result_b

# OK: エラーチェックとフォールバック
def pipeline_safe(task):
    result_a = agent_a.run(task)
    if result_a.get("error"):
        logger.error(f"Agent A 失敗: {result_a['error']}")
        result_a = fallback_agent.run(task)  # フォールバック

    result_b = agent_b.run(result_a["result"])
    if result_b.get("error"):
        return {
            "error": "パイプライン失敗",
            "failed_at": "agent_b",
            "partial_result": result_a
        }
    return result_b
```

---

## 9. テスト戦略

### 9.1 マルチエージェントのテストピラミッド

```
テストピラミッド

          /\
         /  \
        / E2E \          全体パイプラインテスト（少数）
       /------\
      / 結合    \        2-3エージェントの連携テスト（中数）
     /----------\
    / 単体テスト  \      個別エージェントのテスト（多数）
   /--------------\
```

```python
import pytest
from unittest.mock import MagicMock, patch

class TestCollaborativeSystem:
    """協調パターンのテスト"""

    def test_pipeline_order(self):
        """パイプラインが正しい順序で実行されるか"""
        system = CollaborativeSystem()
        execution_order = []

        # モックエージェントを追加
        for name in ["a", "b", "c"]:
            system.add_agent(name, role=f"Agent {name}")
        system.set_pipeline(["a", "b", "c"])

        with patch.object(system.client.messages, 'create') as mock:
            mock.return_value = MagicMock(
                content=[MagicMock(text="result")],
                usage=MagicMock(input_tokens=100, output_tokens=50)
            )
            result = system.run("test task")

        assert len(system.message_log) == 3
        assert system.message_log[0].sender == "a"
        assert system.message_log[1].sender == "b"
        assert system.message_log[2].sender == "c"

    def test_context_propagation(self):
        """コンテキストが正しく伝播されるか"""
        system = CollaborativeSystem()
        system.add_agent("first", role="First")
        system.add_agent("second", role="Second")
        system.set_pipeline(["first", "second"])

        call_args = []

        def capture_call(*args, **kwargs):
            call_args.append(kwargs)
            return MagicMock(
                content=[MagicMock(text="first_result")],
                usage=MagicMock(input_tokens=100, output_tokens=50)
            )

        with patch.object(system.client.messages, 'create', side_effect=capture_call):
            system.run("test task")

        # 2番目のエージェントが1番目の結果を受け取っているか
        second_prompt = call_args[1]["messages"][0]["content"]
        assert "first_result" in second_prompt

class TestDebateSystem:
    """議論パターンのテスト"""

    def test_max_rounds(self):
        """最大ラウンド数で停止するか"""
        system = DebateSystem()
        system.max_rounds = 2

        with patch.object(system.client.messages, 'create') as mock:
            # 常にFAIL判定を返す
            mock.return_value = MagicMock(
                content=[MagicMock(text="判定: FAIL\n信頼度: 低\n理由: 不十分")]
            )
            result = system.run("test question")

        assert result["rounds"] == 2

    def test_early_termination(self):
        """合意に達したら早期終了するか"""
        system = DebateSystem()
        system.max_rounds = 5

        call_count = 0
        def mock_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "公平な審判" in kwargs.get("system", ""):
                return MagicMock(
                    content=[MagicMock(text="判定: PASS\n信頼度: 高\n理由: 十分")]
                )
            return MagicMock(
                content=[MagicMock(text="テスト回答")]
            )

        with patch.object(system.client.messages, 'create', side_effect=mock_response):
            result = system.run("test question")

        assert result["rounds"] == 1  # 1ラウンドで合意
```

---

## 10. FAQ

### Q1: マルチエージェントのコストはどの程度か？

エージェント数 x ステップ数 x 1回あたりのトークン数 でコストが増加する。例えば3エージェントが各5ステップ実行すると、シングルエージェントの5ステップに比べて **約3倍** のAPIコストがかかる。さらにエージェント間通信のオーバーヘッドも加わる。

コスト削減のアプローチ:
- 簡単なルーティングや分類はHaikuモデルを使用
- 結果のキャッシュを活用
- 不要なエージェント間通信を最小化
- バッチ処理可能なタスクはまとめて実行

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

### Q4: エージェント数の最適解は？

経験則として:
- **2-3エージェント**: ほとんどのタスクに適切
- **4-5エージェント**: 大規模プロジェクト（ソフトウェア開発等）
- **6+エージェント**: まれに必要（複雑なシミュレーション等）

エージェント数を増やすよりも、各エージェントのツールセットを充実させる方が効果的な場合が多い。

### Q5: 非同期実行と同期実行のどちらを選ぶべき？

| 条件 | 推奨 |
|------|------|
| タスク間に依存関係あり | 同期（順次実行） |
| 独立したタスクが複数 | 非同期（並列実行） |
| レイテンシ要件が厳しい | 非同期 |
| デバッグが重要 | 同期 |
| リソース制限あり | 同期 |

---

## まとめ

| 項目 | 内容 |
|------|------|
| 協調パターン | 対等なエージェントがパイプラインで処理 |
| 委任パターン | マネージャーがワーカーにタスクを分配 |
| 議論パターン | 提案→批判→判定の弁証法的改善 |
| ハイブリッド | 委任+議論の組み合わせで品質と効率を両立 |
| 通信方式 | 直接 / ブラックボード / メッセージキュー / Pub/Sub |
| 障害耐性 | リトライ / フォールバック / サーキットブレーカー |
| コスト管理 | エージェント別コスト追跡 + モデル使い分け |
| 設計原則 | 最小限のエージェント数で最大の効果を |

## 次に読むべきガイド

- [02-workflow-agents.md](./02-workflow-agents.md) -- ワークフローエージェントの設計
- [03-autonomous-agents.md](./03-autonomous-agents.md) -- 自律エージェントの計画と実行
- [../02-implementation/01-langgraph.md](../02-implementation/01-langgraph.md) -- LangGraphでの実装

## 参考文献

1. Wu, Q. et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023) -- https://arxiv.org/abs/2308.08155
2. CrewAI Documentation -- https://docs.crewai.com/
3. Hong, S. et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework" (2023) -- https://arxiv.org/abs/2308.00352
4. LangGraph Documentation -- https://langchain-ai.github.io/langgraph/
5. Talebirad, Y. et al., "Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents" (2023) -- https://arxiv.org/abs/2306.03314
