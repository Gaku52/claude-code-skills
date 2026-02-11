# シングルエージェント

> ReActパターン、ツール選択戦略、思考の連鎖――1つのLLMが自律的にタスクを遂行するシングルエージェントの設計パターンと実装技法。

## この章で学ぶこと

1. ReActパターンの動作原理と実装方法
2. ツール選択の戦略と精度を上げるプロンプト設計
3. シングルエージェントの限界と適用範囲の判断基準

---

## 1. シングルエージェントの位置づけ

```
エージェントアーキテクチャの複雑度スペクトラム

 シンプル                                              複雑
 +--------+--------+-----------+-----------+-----------+
 | LLM    | Chain  | Single    | Multi     | Autonomous|
 | 直接   | (直列) | Agent     | Agent     | Agent     |
 | 呼出   |        | (ReAct)   | (協調)    | (自律)    |
 +--------+--------+-----------+-----------+-----------+
                    ^^^^^^^^^^^
                    この章の範囲
```

シングルエージェントは **1つのLLMインスタンスがループの中でツールを使いながらタスクを遂行する** パターン。最もバランスの取れたアーキテクチャで、多くのタスクにおいて最初に検討すべき選択肢。

---

## 2. ReActパターン

### 2.1 ReActとは

ReAct = **Re**asoning + **Act**ing。LLMに「考えてから行動する」を繰り返させるパターン。

```
ReAct ループ

  Thought ─────> Action ─────> Observation
     ^                              |
     |                              |
     +──────────────────────────────+
            (繰り返し)

  最終的に Final Answer を出力して終了
```

### 2.2 ReActの実装

```python
# ReAct パターンの完全な実装
import anthropic
import json
import re

class ReActAgent:
    def __init__(self, tools: dict, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.model = model
        self.max_steps = 10

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__}"
            for name, func in self.tools.items()
        )
        return f"""あなたはReActエージェントです。以下の形式で応答してください：

Thought: [現状の分析と次のステップの推論]
Action: [ツール名]
Action Input: [ツールへの入力（JSON）]

ツール実行後にObservationが返されるので、それを分析して次のThoughtに進んでください。
最終回答が出せる場合は以下の形式で：

Thought: [最終的な推論]
Final Answer: [ユーザーへの回答]

利用可能なツール:
{tool_descriptions}
"""

    def run(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        system = self._build_system_prompt()

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                messages=messages
            )

            text = response.content[0].text

            # Final Answer が含まれていれば終了
            if "Final Answer:" in text:
                return text.split("Final Answer:")[-1].strip()

            # Action を解析して実行
            action_match = re.search(r"Action:\s*(.+)", text)
            input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)

            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input = json.loads(input_match.group(1).strip())

                # ツール実行
                if tool_name in self.tools:
                    observation = self.tools[tool_name](**tool_input)
                else:
                    observation = f"エラー: ツール '{tool_name}' は存在しません"

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

        return "最大ステップ数に達しました。"

# 使用例
def search_web(query: str) -> str:
    """Webを検索して上位結果を返す"""
    return f"検索結果: '{query}' に関する情報..."

def calculate(expression: str) -> str:
    """数式を安全に計算する"""
    return str(eval(expression, {"__builtins__": {}}, {}))

agent = ReActAgent(tools={
    "search_web": search_web,
    "calculate": calculate
})

result = agent.run("日本のGDPは何ドル？ それは世界全体の何%？")
```

### 2.3 Function Callingベースのシングルエージェント

```python
# Function Calling を使ったよりモダンな実装
class FunctionCallingAgent:
    def __init__(self, tools: list, system_prompt: str = ""):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.system_prompt = system_prompt
        self.tool_handlers = {}

    def register_handler(self, name: str, handler):
        """ツール実行ハンドラを登録"""
        self.tool_handlers[name] = handler

    def run(self, query: str, max_steps: int = 15) -> str:
        messages = [{"role": "user", "content": query}]

        for _ in range(max_steps):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages
            )

            # 最終回答
            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            # ツール呼び出し
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = self.tool_handlers.get(block.name)
                    if handler:
                        try:
                            result = handler(**block.input)
                        except Exception as e:
                            result = f"エラー: {e}"
                    else:
                        result = f"ハンドラ未登録: {block.name}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "最大ステップ数に達しました。"

    def _extract_text(self, response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
```

---

## 3. ツール選択戦略

### 3.1 ツール選択の精度を上げる方法

```
ツール選択の精度向上テクニック

1. 説明の質          → ツールの目的・使用場面を明確に
2. パラメータの制約   → enum, min/max, デフォルト値
3. 例の提供          → 具体的な使用例をdescriptionに
4. 重複排除          → 類似ツールの統合・差別化
5. カテゴリ分け      → 関連ツールのグループ化
```

### 3.2 動的ツール選択

```python
# タスクの種類に応じてツールセットを動的に選択
class DynamicToolSelector:
    def __init__(self):
        self.tool_categories = {
            "research": [
                {"name": "web_search", ...},
                {"name": "read_webpage", ...},
                {"name": "summarize", ...}
            ],
            "coding": [
                {"name": "read_file", ...},
                {"name": "write_file", ...},
                {"name": "run_tests", ...}
            ],
            "data": [
                {"name": "query_database", ...},
                {"name": "create_chart", ...},
                {"name": "export_csv", ...}
            ]
        }

    def select_tools(self, query: str, llm) -> list:
        """クエリに基づいて最適なツールセットを選択"""
        category = llm.classify(
            query,
            categories=list(self.tool_categories.keys())
        )
        return self.tool_categories[category]
```

---

## 4. 思考パターンの比較

| パターン | 思考プロセス | ツール使用 | 適用場面 |
|----------|------------|-----------|---------|
| ReAct | Thought→Action→Observation | 毎ステップ | 汎用タスク |
| Plan-then-Execute | 計画→一括実行 | 計画後に連続 | 構造化タスク |
| Reflexion | 実行→振り返り→改善 | 実行+評価 | 品質重視タスク |
| Chain-of-Thought | 推論の連鎖 | なし/最小限 | 推論集約タスク |

### 4.1 Plan-then-Execute

```python
# 先に計画を立ててから実行するパターン
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, goal: str) -> str:
        # Step 1: 計画を立てる
        plan = self.planner.generate(f"""
目標: {goal}

この目標を達成するためのステップを番号付きリストで出力してください。
各ステップは具体的で、利用可能なツールを使って実行可能であること。
""")

        steps = self._parse_plan(plan)
        results = []

        # Step 2: 計画を順に実行
        for i, step in enumerate(steps):
            result = self.executor.execute_step(
                step=step,
                previous_results=results,
                tools=self.tools
            )
            results.append({"step": step, "result": result})

        # Step 3: 結果を統合
        return self._synthesize(goal, results)
```

---

## 5. エラーハンドリング

```
エラーハンドリングの階層

Level 1: ツール実行エラー
  → 再試行 (最大3回)

Level 2: ツール選択ミス
  → 代替ツールの提案

Level 3: 計画の失敗
  → 再計画

Level 4: 目標達成不可能
  → ユーザーに報告 + 部分的成果の返却
```

```python
# ロバストなエラーハンドリングの実装
class RobustAgent:
    def execute_with_retry(self, tool_name, args, max_retries=3):
        for attempt in range(max_retries):
            try:
                result = self.tools[tool_name](**args)
                return {"status": "success", "data": result}
            except TimeoutError:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数バックオフ
                    continue
            except ValidationError as e:
                return {
                    "status": "error",
                    "message": f"入力エラー: {e}",
                    "suggestion": "パラメータを修正して再試行してください"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"予期せぬエラー: {e}",
                    "suggestion": "代替手段を検討してください"
                }

        return {"status": "error", "message": "最大再試行回数超過"}
```

---

## 6. シングル vs マルチの判断基準

| 基準 | シングルエージェント | マルチエージェント |
|------|-------------------|------------------|
| タスク複雑度 | 中程度 | 高 |
| 専門性 | 汎用的 | 複数の専門性が必要 |
| 並行処理 | 不要 | 必要 |
| デバッグ | 容易 | 複雑 |
| コスト | 低-中 | 高 |
| レイテンシ | 低-中 | 中-高 |
| 実装工数 | 少 | 多 |

---

## 7. アンチパターン

### アンチパターン1: 過度な自律性

```python
# NG: ユーザー確認なしに破壊的操作を実行
class DangerousAgent:
    def run(self, goal):
        # いきなりファイルを削除する可能性がある
        action = self.think(goal)
        self.execute(action)  # 確認なし!

# OK: 重要な操作の前にユーザー確認を挟む
class SafeAgent:
    DESTRUCTIVE_ACTIONS = {"delete_file", "send_email", "deploy"}

    def run(self, goal):
        action = self.think(goal)
        if action.tool_name in self.DESTRUCTIVE_ACTIONS:
            if not self.confirm_with_user(action):
                return "操作がキャンセルされました"
        self.execute(action)
```

### アンチパターン2: コンテキストの浪費

```python
# NG: すべてのツール結果を全文保持
observations = []
for step in range(100):
    result = tool.execute(...)
    observations.append(result)  # 大量のデータが蓄積

# OK: 必要な情報のみ抽出して保持
observations = []
for step in range(100):
    result = tool.execute(...)
    summary = self.extract_key_info(result)  # 要約
    observations.append(summary)
```

---

## 8. FAQ

### Q1: ReActとFunction Callingのどちらを使うべき？

**Function Calling推奨**。ReActはテキストベースで出力パース（正規表現）が必要だが、Function Callingはstructured output（JSON）で確実にツール呼び出しを受け取れる。ReActは教育目的や、Function Callingに非対応のモデルで使用する。

### Q2: シングルエージェントの最大ステップ数の目安は？

タスクの複雑さに依存するが、**10-25ステップ** が一般的な上限。これ以上かかるタスクは:
- タスクの分割を検討する
- マルチエージェントに移行する
- ツールの粒度を見直す（1ツールでより多くを処理）

### Q3: エージェントが同じツールを繰り返し呼ぶ場合は？

「ループ検出」を実装する。直近N回のツール呼び出しが同じパターンならば介入する:
- エラーメッセージを改善して原因を伝える
- 代替アプローチを明示的に指示する
- 強制終了して部分的な結果を返す

---

## まとめ

| 項目 | 内容 |
|------|------|
| ReAct | Thought→Action→Observation の反復パターン |
| Function Calling | 構造化されたツール呼び出し（推奨） |
| ツール選択 | 明確な説明・動的選択・5-15個に絞る |
| エラー処理 | 再試行・代替手段・ユーザー報告の階層 |
| 適用範囲 | 中程度の複雑度で汎用的なタスク |
| 設計原則 | シンプルに始め、必要に応じて複雑化する |

## 次に読むべきガイド

- [01-multi-agent.md](./01-multi-agent.md) — マルチエージェントの協調パターン
- [02-workflow-agents.md](./02-workflow-agents.md) — ワークフローエージェントの設計
- [../02-implementation/00-langchain-agent.md](../02-implementation/00-langchain-agent.md) — LangChainでの実装

## 参考文献

1. Yao, S. et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023) — https://arxiv.org/abs/2210.03629
2. Anthropic, "Tool use best practices" — https://docs.anthropic.com/en/docs/build-with-claude/tool-use
3. Shinn, N. et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) — https://arxiv.org/abs/2303.11366
