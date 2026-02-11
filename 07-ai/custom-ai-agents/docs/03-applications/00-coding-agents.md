# コーディングエージェント

> Claude Code・Devin・Cursor――コードの理解・生成・修正・テストを自律的に行うコーディングエージェントの仕組み、設計パターン、実践的な活用法。

## この章で学ぶこと

1. コーディングエージェントのアーキテクチャと主要プロダクトの比較
2. コード生成・修正・テスト・レビューの自動化パイプラインの設計
3. コーディングエージェントの効果的な活用法と限界の理解

---

## 1. コーディングエージェントとは

```
コーディングエージェントの能力スペクトラム

  コード補完            コーディングエージェント
  (行単位)              (プロジェクト単位)
  +------+------+------+------+------+------+
  | 行   | 関数 | ファイル| 複数  | 機能  | プロジェクト|
  | 補完 | 生成 | 生成  | ファイル| 実装  | 全体    |
  +------+------+------+------+------+------+
  Copilot               Claude Code / Devin
  (受動的)               (能動的)

コーディングエージェント = LLM + ファイル操作 + コマンド実行 + 検索
```

### 1.1 コーディングエージェントの動作フロー

```
典型的なバグ修正フロー

1. [Issue理解]   : バグレポートを分析
       |
2. [コード検索]   : 関連ファイルを特定
       |
3. [原因特定]     : コードを読んで原因を推論
       |
4. [テスト作成]   : 再現テストを書く（RED）
       |
5. [修正実装]     : コードを修正（GREEN）
       |
6. [テスト実行]   : 全テスト通過を確認
       |
7. [レビュー]     : 変更の品質チェック
       |
8. [コミット]     : 変更を保存
```

---

## 2. 主要コーディングエージェント

### 2.1 プロダクト比較

| 製品 | 開発元 | 形態 | 特徴 | 自律性 |
|------|--------|------|------|--------|
| Claude Code | Anthropic | CLI | ターミナル統合、MCP対応 | L3 |
| Devin | Cognition | Web | フルスタック自律開発 | L3-L4 |
| Cursor | Cursor Inc. | IDE | VS Code fork、AI統合 | L2-L3 |
| GitHub Copilot | GitHub | IDE拡張 | 行・関数単位の補完 | L1 |
| Cline | Community | VSCode拡張 | エージェント型、MCP対応 | L2-L3 |
| Aider | Community | CLI | Git統合、ペアプロ型 | L2-L3 |

### 2.2 アーキテクチャ比較

```
Claude Code のアーキテクチャ:
+-------------------------------------------+
| ターミナル (CLI)                            |
|  +---------------------------------------+|
|  | Agent Loop                            ||
|  |  [LLM] ←→ [ツール]                    ||
|  |    |         ├── Read (ファイル読取)   ||
|  |    |         ├── Write (ファイル書込)  ||
|  |    |         ├── Bash (コマンド実行)   ||
|  |    |         ├── Grep (検索)          ||
|  |    |         ├── Glob (ファイル検索)   ||
|  |    |         └── MCP (外部ツール)     ||
|  |    |                                  ||
|  |    +── 会話履歴 + コンテキスト          ||
|  +---------------------------------------+|
+-------------------------------------------+

Cursor のアーキテクチャ:
+-------------------------------------------+
| VS Code (IDE)                              |
|  +---------------------------------------+|
|  | Composer / Chat                       ||
|  |  [LLM] ←→ [IDE統合ツール]             ||
|  |    |         ├── ファイル編集         ||
|  |    |         ├── ターミナル           ||
|  |    |         ├── コードベース検索     ||
|  |    |         └── Lint/テスト          ||
|  |    |                                  ||
|  |    +── codebase indexing              ||
|  +---------------------------------------+|
+-------------------------------------------+
```

---

## 3. コーディングエージェントの実装

### 3.1 基本的なコーディングエージェント

```python
# シンプルなコーディングエージェントの実装
import anthropic
import subprocess
import os

class CodingAgent:
    def __init__(self, workspace: str):
        self.client = anthropic.Anthropic()
        self.workspace = workspace
        self.tools = [
            {
                "name": "read_file",
                "description": "ファイルの内容を読み取る",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "ファイルに内容を書き込む",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "run_command",
                "description": "シェルコマンドを実行する（テスト、lint等）",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "search_code",
                "description": "コードベースをgrep検索する",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "file_pattern": {"type": "string", "description": "*.py, *.ts等"}
                    },
                    "required": ["pattern"]
                }
            }
        ]

    def execute_tool(self, name: str, args: dict) -> str:
        full_path = os.path.join(self.workspace, args.get("path", ""))

        if name == "read_file":
            with open(full_path) as f:
                return f.read()

        elif name == "write_file":
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(args["content"])
            return f"Written to {args['path']}"

        elif name == "run_command":
            result = subprocess.run(
                args["command"], shell=True,
                capture_output=True, text=True,
                cwd=self.workspace, timeout=60
            )
            output = result.stdout + result.stderr
            return output[:5000]

        elif name == "search_code":
            cmd = f"grep -rn '{args['pattern']}' --include='{args.get('file_pattern', '*')}' ."
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=self.workspace
            )
            return result.stdout[:5000]

        return f"Unknown tool: {name}"

    def run(self, task: str) -> str:
        system = """あなたはシニアソフトウェアエンジニアです。
コードを変更する前に必ず既存コードを読んで理解してください。
テストを書いてから実装し、全テストが通ることを確認してください。"""

        messages = [{"role": "user", "content": task}]

        for _ in range(30):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system,
                tools=self.tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                return response.content[0].text

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "最大ステップ数に達しました"
```

### 3.2 TDD（テスト駆動開発）エージェント

```python
# TDDパターンのコーディングエージェント
class TDDAgent(CodingAgent):
    def implement_feature(self, feature_description: str) -> str:
        """テスト駆動で機能を実装"""
        return self.run(f"""
以下の機能をTDD（テスト駆動開発）で実装してください。

機能: {feature_description}

手順:
1. まず既存のコードベースを調査してディレクトリ構造とコードスタイルを理解
2. テストファイルを作成（テストが失敗する状態 = RED）
3. テストを実行して失敗を確認
4. 最小限の実装でテストを通す（GREEN）
5. テストを実行して全て通ることを確認
6. 必要に応じてリファクタリング（REFACTOR）
7. 最終的に全テストが通ることを確認

重要: 各ステップで実際にテストを実行してください。
""")
```

### 3.3 コードレビューエージェント

```python
# コードレビューエージェント
class CodeReviewAgent:
    def review_diff(self, diff: str) -> str:
        """差分をレビューして改善点を指摘"""
        return self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": f"""
以下のコード差分をレビューしてください。

```diff
{diff}
```

以下の観点で評価してください:
1. **バグ**: ロジックエラー、エッジケース、null/undefined
2. **セキュリティ**: インジェクション、認証、権限
3. **パフォーマンス**: N+1問題、不要な処理
4. **可読性**: 命名、構造、コメント
5. **テスト**: テストカバレッジ、エッジケースのテスト

各問題について重要度（Critical/Warning/Info）を付けてください。
"""}]
        ).content[0].text
```

---

## 4. 効果的なプロンプト設計

```python
# コーディングエージェントのためのCLAUDE.md例
CLAUDE_MD = """
# プロジェクト固有のルール

## テクノロジースタック
- Backend: Python 3.12, FastAPI, SQLAlchemy
- Frontend: TypeScript, React 19, Tailwind CSS
- DB: PostgreSQL 16
- テスト: pytest, vitest

## コーディング規約
- Python: Black + isort + mypy strict
- TypeScript: ESLint + Prettier
- コミットメッセージ: Conventional Commits

## ディレクトリ構造
- backend/src/ : バックエンドソースコード
- backend/tests/ : バックエンドテスト
- frontend/src/ : フロントエンドソースコード
- frontend/tests/ : フロントエンドテスト

## 重要な注意事項
- DB変更時は必ずAlembicマイグレーションを作成
- APIエンドポイント追加時はOpenAPIスキーマを更新
- 環境変数は .env.example に追記
"""
```

---

## 5. パフォーマンスと限界

### 5.1 得意なタスク vs 苦手なタスク

| 得意 | 苦手 |
|------|------|
| バグ修正（明確なエラー） | アーキテクチャ設計（全体最適） |
| テスト作成 | UX/UIデザインの判断 |
| リファクタリング | ビジネスロジックの要件定義 |
| ドキュメント生成 | レガシーコードの大規模改修 |
| API実装 | パフォーマンスチューニング（計測必要） |
| 型定義・スキーマ作成 | セキュリティ監査（網羅的） |
| ボイラープレート生成 | 複数リポジトリにまたがる変更 |

### 5.2 SWE-benchスコア比較（2025年時点概算）

| エージェント | SWE-bench Lite | SWE-bench Full |
|-------------|---------------|----------------|
| Claude Code (Opus) | ~55% | ~35% |
| Devin | ~45% | ~25% |
| GPT-4o + SWE-Agent | ~35% | ~20% |
| 人間エンジニア | ~80% | ~60% |

---

## 6. アンチパターン

### アンチパターン1: コンテキスト不足での指示

```
# NG: 漠然とした指示
"バグを直して"

# OK: 十分なコンテキストを提供
"users.py の get_user 関数で、存在しないユーザーIDを渡すと
500エラーが返る。404を返すように修正して。
再現手順: curl localhost:8000/users/99999"
```

### アンチパターン2: レビューなしの自動マージ

```
# NG: エージェントの出力をそのままマージ
agent.generate_code() → git push → auto-merge

# OK: 必ず人間のレビューを挟む
agent.generate_code() → PR作成 → 人間レビュー → CI通過 → マージ
```

---

## 7. FAQ

### Q1: コーディングエージェントは人間の仕事を奪うか？

現時点では「奪う」というより「変える」。エージェントが得意なのはボイラープレート・テスト・バグ修正などの定型的タスク。人間は要件定義・アーキテクチャ設計・レビュー・ユーザー体験など高レベルな判断に注力するようになる。

### Q2: どの程度のコードベースサイズまで対応できる？

現在のコーディングエージェントはコンテキストウィンドウの制約により、**一度に扱えるのは数十ファイル程度**。大規模コードベースではRAG（コード検索）や、タスクの範囲を絞ることが重要。プロジェクト全体を理解するのではなく、関連する部分を効率的に検索する設計が必要。

### Q3: エージェントが書いたコードの品質保証は？

3段階のチェックを推奨:
1. **自動テスト**: CI/CDでの自動テスト通過
2. **静的解析**: lint, 型チェック, セキュリティスキャン
3. **人間レビュー**: アーキテクチャ整合性、ビジネスロジックの正しさ

---

## まとめ

| 項目 | 内容 |
|------|------|
| 定義 | コードの理解・生成・修正・テストを自律的に行うエージェント |
| 主要製品 | Claude Code, Devin, Cursor, Copilot, Aider |
| アーキテクチャ | LLM + ファイル操作 + コマンド実行 + コード検索 |
| 得意領域 | バグ修正、テスト作成、リファクタリング |
| 限界 | アーキテクチャ設計、大規模改修 |
| 品質保証 | 自動テスト + 静的解析 + 人間レビューの3段階 |

## 次に読むべきガイド

- [01-research-agents.md](./01-research-agents.md) — リサーチエージェント
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) — エージェントの評価
- [../04-production/01-safety.md](../04-production/01-safety.md) — コーディングエージェントの安全性

## 参考文献

1. Jimenez, C. E. et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2023) — https://arxiv.org/abs/2310.06770
2. Anthropic, "Claude Code" — https://docs.anthropic.com/en/docs/claude-code
3. Yang, J. et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (2024) — https://arxiv.org/abs/2405.15793
