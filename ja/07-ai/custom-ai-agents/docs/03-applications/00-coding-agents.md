# コーディングエージェント

> Claude Code・Devin・Cursor――コードの理解・生成・修正・テストを自律的に行うコーディングエージェントの仕組み、設計パターン、実践的な活用法。

## この章で学ぶこと

1. コーディングエージェントのアーキテクチャと主要プロダクトの比較
2. コード生成・修正・テスト・レビューの自動化パイプラインの設計
3. コーディングエージェントの効果的な活用法と限界の理解
4. カスタムコーディングエージェントの実装パターン
5. 開発ワークフローへの統合と運用ベストプラクティス


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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

### 1.2 コーディングエージェントの分類

```
自律性レベルによる分類

Level 1: コード補完
  - 行/関数単位のサジェスト
  - 人間がトリガーを引く
  - 例: GitHub Copilot (Inline)

Level 2: インタラクティブ生成
  - チャットで指示→コード生成
  - 人間が承認して適用
  - 例: Cursor Chat, Copilot Chat

Level 3: タスク自律実行
  - タスク記述→複数ファイル変更
  - ツールを自律的に使用
  - 人間はレビューで介入
  - 例: Claude Code, Aider, Cline

Level 4: エンドツーエンド自律
  - Issue→PR完成まで自律
  - テスト・CI確認まで実行
  - 人間はマージ判断のみ
  - 例: Devin, SWE-Agent
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
| Windsurf | Codeium | IDE | AI統合IDE、Cascade | L2-L3 |
| Amazon Q Developer | AWS | IDE/CLI | AWS統合、セキュリティスキャン | L2 |

### 2.2 アーキテクチャ比較

```
Claude Code のアーキテクチャ:
+-------------------------------------------+
| ターミナル (CLI)                            |
|  +---------------------------------------+|
|  | Agent Loop                            ||
|  |  [LLM] <-> [ツール]                    ||
|  |    |         +-- Read (ファイル読取)   ||
|  |    |         +-- Write (ファイル書込)  ||
|  |    |         +-- Bash (コマンド実行)   ||
|  |    |         +-- Grep (検索)          ||
|  |    |         +-- Glob (ファイル検索)   ||
|  |    |         +-- MCP (外部ツール)     ||
|  |    |                                  ||
|  |    +-- 会話履歴 + コンテキスト          ||
|  +---------------------------------------+|
+-------------------------------------------+

Cursor のアーキテクチャ:
+-------------------------------------------+
| VS Code (IDE)                              |
|  +---------------------------------------+|
|  | Composer / Chat                       ||
|  |  [LLM] <-> [IDE統合ツール]             ||
|  |    |         +-- ファイル編集         ||
|  |    |         +-- ターミナル           ||
|  |    |         +-- コードベース検索     ||
|  |    |         +-- Lint/テスト          ||
|  |    |                                  ||
|  |    +-- codebase indexing              ||
|  +---------------------------------------+|
+-------------------------------------------+
```

### 2.3 ツール設計の詳細比較

```
ツール粒度の比較

Claude Code:
  - Read: ファイル全体 or 行範囲指定
  - Write: ファイル全体の書き換え
  - Edit: 部分的な文字列置換（old_string → new_string）
  - Bash: 任意のシェルコマンド
  - Grep: ripgrepベースの高速検索
  - Glob: ファイルパターンマッチング

Cursor:
  - Edit: diff形式の適用
  - Terminal: コマンド実行
  - Codebase: セマンティック検索（インデックス済み）
  - Lint: 組み込みリンター統合

Aider:
  - file edit: unified diff形式
  - shell: コマンド実行
  - git: Git操作（add, commit）

設計のポイント:
  - ファイル編集: 全置換 vs diff適用 vs search-replace
  - 検索: テキスト検索 vs セマンティック検索 vs AST検索
  - 実行: サンドボックス有無、タイムアウト設定
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
                        "file_pattern": {
                            "type": "string",
                            "description": "*.py, *.ts等"
                        }
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
            cmd = (
                f"grep -rn '{args['pattern']}' "
                f"--include='{args.get('file_pattern', '*')}' ."
            )
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

            messages.append({
                "role": "assistant",
                "content": response.content
            })
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
    def __init__(self):
        self.client = anthropic.Anthropic()

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

    def review_pr(self, repo: str, pr_number: int) -> dict:
        """GitHub PRを包括的にレビュー"""
        # PR情報の取得
        pr_info = self._get_pr_info(repo, pr_number)
        diff = self._get_pr_diff(repo, pr_number)
        changed_files = self._get_changed_files(repo, pr_number)

        # ファイルごとのレビュー
        file_reviews = []
        for file in changed_files:
            file_diff = self._extract_file_diff(diff, file)
            review = self.review_diff(file_diff)
            file_reviews.append({
                "file": file,
                "review": review
            })

        # 全体サマリーの生成
        summary = self._generate_summary(
            pr_info, file_reviews
        )

        return {
            "summary": summary,
            "file_reviews": file_reviews,
            "approval": self._determine_approval(file_reviews)
        }

    def _determine_approval(self, reviews: list) -> str:
        """レビュー結果に基づいて承認判断"""
        has_critical = any(
            "Critical" in r["review"] for r in reviews
        )
        if has_critical:
            return "CHANGES_REQUESTED"
        return "APPROVED"
```

### 3.4 リファクタリングエージェント

```python
class RefactoringAgent(CodingAgent):
    """コードリファクタリングに特化したエージェント"""

    def refactor_function(
        self,
        file_path: str,
        function_name: str,
        goal: str
    ) -> str:
        """関数をリファクタリング"""
        return self.run(f"""
以下の関数をリファクタリングしてください。

ファイル: {file_path}
関数名: {function_name}
目標: {goal}

手順:
1. 対象の関数とその呼び出し元を全て検索して理解する
2. 既存のテストを確認する（なければ先にテストを追加）
3. テストを実行して現状のパスを確認
4. リファクタリングを実施
5. テストを実行して全てパスすることを確認
6. 呼び出し元に影響がないことを確認

重要:
- 外部インターフェース（関数シグネチャ）は変更しない
- リファクタリング前後でテスト結果が同一であること
- 一度に大きな変更をせず、小さなステップで進める
""")

    def extract_method(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        new_function_name: str
    ) -> str:
        """メソッド抽出リファクタリング"""
        return self.run(f"""
以下のコード範囲を新しい関数として抽出してください。

ファイル: {file_path}
行範囲: {start_line}-{end_line}
新関数名: {new_function_name}

手順:
1. 対象コードを読んで、必要な引数と戻り値を特定
2. 新しい関数を作成
3. 元のコードを新関数の呼び出しに置換
4. テストを実行して動作確認
""")

    def remove_code_duplication(self, directory: str) -> str:
        """コード重複の検出と解消"""
        return self.run(f"""
以下のディレクトリ内のコード重複を検出して解消してください。

ディレクトリ: {directory}

手順:
1. ディレクトリ内の全ファイルを読んで類似コードを検出
2. 重複が3箇所以上あるパターンを優先して対応
3. 共通関数/クラスを抽出して重複を解消
4. 全テストが通ることを確認
5. 変更の概要をレポートとして出力

重要:
- 1つの重複パターンずつ段階的に修正
- 各修正後にテストを実行して確認
""")
```

### 3.5 マイグレーションエージェント

```python
class MigrationAgent(CodingAgent):
    """言語・フレームワークのマイグレーションエージェント"""

    def migrate_dependency(
        self,
        old_package: str,
        new_package: str,
        migration_guide: str = ""
    ) -> str:
        """依存パッケージのマイグレーション"""
        return self.run(f"""
以下のパッケージマイグレーションを実施してください。

変更前: {old_package}
変更後: {new_package}
{f'マイグレーションガイド: {migration_guide}' if migration_guide else ''}

手順:
1. 現在のコードベースで {old_package} の使用箇所を全て検索
2. 各使用箇所の変更方法を計画
3. パッケージの依存関係を更新（package.json, requirements.txt等）
4. コードを1ファイルずつ変更
5. 各ファイル変更後にテストを実行
6. 全テストが通ることを確認
7. 変更概要のレポートを出力

重要:
- APIの互換性の違いに注意
- deprecated な機能の代替手段を使用
- 型定義の変更にも対応
""")

    def upgrade_framework(
        self,
        framework: str,
        from_version: str,
        to_version: str
    ) -> str:
        """フレームワークのバージョンアップグレード"""
        return self.run(f"""
{framework} を v{from_version} から v{to_version} にアップグレードしてください。

手順:
1. 破壊的変更のリストを確認
2. 影響を受けるファイルを特定
3. 依存関係のバージョンを更新
4. コードの互換性を修正
5. deprecated 警告を解消
6. テストを実行して確認
7. 変更ログをまとめる
""")
```

---

## 4. 効果的なプロンプト設計

### 4.1 CLAUDE.md の設計

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

### 4.2 タスク別のプロンプトテンプレート

```python
# バグ修正用プロンプト
BUG_FIX_PROMPT = """
## バグ修正リクエスト

### 症状
{症状の説明}

### 再現手順
{再現手順}

### 期待される動作
{期待される動作}

### エラーログ（あれば）
```
{エラーログ}
```

### 対応方針
1. まず再現テストを書いて失敗を確認
2. 原因を特定
3. 最小限の修正で対応
4. テスト通過を確認
5. 関連する既存テストも全て通ることを確認
"""

# 新機能実装用プロンプト
FEATURE_PROMPT = """
## 新機能実装リクエスト

### 機能概要
{機能の説明}

### 受け入れ条件
{受け入れ条件のリスト}

### 技術的な制約
{制約事項}

### 対応方針
1. 既存のコードベースを調査
2. 設計方針を報告
3. テストを先に書く（TDD）
4. 実装
5. テスト通過を確認
6. ドキュメント更新（必要に応じて）
"""

# リファクタリング用プロンプト
REFACTORING_PROMPT = """
## リファクタリングリクエスト

### 対象
{対象ファイル/モジュール}

### 目的
{リファクタリングの目的}

### 制約
- 外部インターフェースは変更しない
- 既存テストを全て通す
- パフォーマンスを悪化させない

### 対応方針
1. 現状のテストカバレッジを確認（不足していれば追加）
2. 小さなステップで段階的に変更
3. 各ステップ後にテスト実行
"""
```

### 4.3 コーディングエージェント統合のベストプラクティス

```python
# GitHub Actions でのコーディングエージェント活用例
"""
name: Agent-Assisted Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Get PR diff
        id: diff
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > pr.diff
          echo "diff_file=pr.diff" >> $GITHUB_OUTPUT

      - name: AI Code Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python scripts/ai_review.py \
            --diff-file pr.diff \
            --output review.md

      - name: Post Review Comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');
            await github.rest.pulls.createReview({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.payload.pull_request.number,
              body: review,
              event: 'COMMENT'
            });
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
| 依存パッケージ更新 | コンテキストウィンドウを超える変更 |
| コードマイグレーション | 暗黙知に依存する実装判断 |

### 5.2 SWE-benchスコア比較（2025年時点概算）

| エージェント | SWE-bench Lite | SWE-bench Full |
|-------------|---------------|----------------|
| Claude Code (Opus) | ~55% | ~35% |
| Devin | ~45% | ~25% |
| GPT-4o + SWE-Agent | ~35% | ~20% |
| 人間エンジニア | ~80% | ~60% |

### 5.3 コンテキストウィンドウの制約と対策

```
コンテキストウィンドウの効率的な利用

問題: 大規模コードベースをコンテキストに収めきれない

対策1: スマートな検索戦略
  [タスク] → [キーワード抽出] → [Grep/Glob検索]
  → [関連ファイル特定] → [必要部分のみ読み込み]

対策2: 段階的な読み込み
  1st pass: ディレクトリ構造の把握
  2nd pass: 関連ファイルの概要読み取り
  3rd pass: 修正対象の詳細読み込み

対策3: コンテキストの圧縮
  - 古い会話をサマリーに圧縮
  - 不要なファイル内容を破棄
  - 関連部分のみを保持

対策4: タスクの分割
  大きなタスクを小さなサブタスクに分割
  各サブタスクは独立したコンテキストで処理
```

### 5.4 エラー分析とデバッグ戦略

```python
class DebugAgent(CodingAgent):
    """デバッグに特化したエージェント"""

    def diagnose_error(self, error_log: str) -> str:
        """エラーログから原因を診断"""
        return self.run(f"""
以下のエラーを診断し、修正してください。

エラーログ:
```
{error_log}
```

診断手順:
1. エラーメッセージからエラーの種類を特定
2. スタックトレースから発生箇所を特定
3. 該当コードを読んでエラー原因を分析
4. 関連するコードや設定も確認
5. 修正案を提示し、実装
6. テストで修正を確認

分析結果は以下の形式で報告:
- エラーの種類:
- 発生箇所:
- 根本原因:
- 修正方法:
- 再発防止策:
""")

    def investigate_flaky_test(self, test_path: str) -> str:
        """不安定なテストの調査"""
        return self.run(f"""
以下のテストが不安定（時々失敗する）です。原因を調査してください。

テストパス: {test_path}

調査手順:
1. テストコードを読んで理解
2. テスト対象のコードを確認
3. 以下の観点で原因を調査:
   - 時間依存（タイムゾーン、タイムアウト）
   - 順序依存（テスト間の依存関係）
   - 外部依存（API、DB、ファイル）
   - 並行性（レースコンディション）
   - 環境依存（OS、バージョン）
4. テストを複数回実行して確認
5. 修正を実装
6. 修正後にテストを複数回実行して安定性を確認
""")
```

---

## 6. 開発ワークフローへの統合

### 6.1 チーム開発でのコーディングエージェント活用

```
チーム開発ワークフロー

1. Issue作成（人間）
   → 要件、受け入れ条件、技術的制約を記述

2. 実装（エージェント + 人間）
   → エージェントが初期実装
   → 人間がレビューしてフィードバック
   → エージェントが修正

3. レビュー（エージェント + 人間）
   → エージェントが自動レビュー（バグ、セキュリティ、パフォーマンス）
   → 人間がアーキテクチャ、ビジネスロジックの観点でレビュー

4. テスト（エージェント）
   → CI/CDでの自動テスト
   → エージェントがテストカバレッジを補完

5. マージ（人間）
   → 最終判断は人間が行う
```

### 6.2 コーディングエージェントの評価指標

```python
class CodingAgentEvaluator:
    """コーディングエージェントのパフォーマンス評価"""

    def evaluate_task(
        self,
        task_description: str,
        agent_output: dict,
        ground_truth: dict
    ) -> dict:
        """タスク実行結果を評価"""
        metrics = {}

        # 1. 正確性: テストの通過率
        metrics["test_pass_rate"] = self._run_tests(
            agent_output["modified_files"]
        )

        # 2. コード品質: 静的解析スコア
        metrics["lint_score"] = self._run_linter(
            agent_output["modified_files"]
        )

        # 3. 効率性: ステップ数とトークン消費
        metrics["total_steps"] = agent_output["steps"]
        metrics["total_tokens"] = agent_output["tokens"]
        metrics["total_cost"] = agent_output["cost"]

        # 4. 差分の適切さ: 不要な変更がないか
        metrics["unnecessary_changes"] = self._check_unnecessary_changes(
            agent_output["diff"],
            ground_truth.get("expected_diff")
        )

        # 5. 時間: 実行時間
        metrics["execution_time"] = agent_output["duration_seconds"]

        return metrics

    def benchmark_suite(
        self,
        agent,
        test_cases: list[dict]
    ) -> dict:
        """ベンチマークスイートの実行"""
        results = []

        for case in test_cases:
            try:
                output = agent.run(case["task"])
                metrics = self.evaluate_task(
                    case["task"], output, case["expected"]
                )
                results.append({
                    "case_id": case["id"],
                    "status": "completed",
                    "metrics": metrics
                })
            except Exception as e:
                results.append({
                    "case_id": case["id"],
                    "status": "error",
                    "error": str(e)
                })

        # 集計
        completed = [r for r in results if r["status"] == "completed"]
        return {
            "total_cases": len(test_cases),
            "completed": len(completed),
            "success_rate": (
                sum(1 for r in completed
                    if r["metrics"]["test_pass_rate"] == 1.0)
                / len(completed) if completed else 0
            ),
            "avg_steps": (
                sum(r["metrics"]["total_steps"] for r in completed)
                / len(completed) if completed else 0
            ),
            "avg_cost": (
                sum(r["metrics"]["total_cost"] for r in completed)
                / len(completed) if completed else 0
            ),
            "details": results
        }
```

---

## 7. セキュリティとガードレール

### 7.1 コーディングエージェントのセキュリティ対策

```python
class SecureCodingAgent(CodingAgent):
    """セキュリティ対策が組み込まれたコーディングエージェント"""

    # 実行禁止コマンド
    BLOCKED_COMMANDS = [
        r"rm\s+-rf\s+/",
        r"curl.*\|\s*bash",
        r"wget.*\|\s*sh",
        r"chmod\s+777",
        r"sudo\s+",
        r"eval\s*\(",
    ]

    # 書き込み禁止パス
    PROTECTED_PATHS = [
        ".env",
        ".git/",
        "credentials",
        "secrets",
        "node_modules/",
        "__pycache__/",
    ]

    def execute_tool(self, name: str, args: dict) -> str:
        # コマンドの安全性チェック
        if name == "run_command":
            for pattern in self.BLOCKED_COMMANDS:
                if re.search(pattern, args["command"]):
                    return f"セキュリティポリシーにより拒否: {args['command']}"

        # 書き込みパスのチェック
        if name == "write_file":
            for protected in self.PROTECTED_PATHS:
                if protected in args.get("path", ""):
                    return f"保護対象のパスへの書き込み拒否: {args['path']}"

        return super().execute_tool(name, args)
```

### 7.2 セキュリティチェック統合

```python
class SecurityReviewAgent:
    """セキュリティ観点でのコードレビュー"""

    SECURITY_CHECKS = {
        "sql_injection": {
            "patterns": [
                r"f\".*SELECT.*{",
                r"f'.*SELECT.*{",
                r"\.format\(.*SELECT",
                r"\+.*SELECT.*\+",
            ],
            "severity": "critical",
            "recommendation": "パラメータ化クエリを使用してください"
        },
        "xss": {
            "patterns": [
                r"innerHTML\s*=",
                r"dangerouslySetInnerHTML",
                r"document\.write\(",
            ],
            "severity": "high",
            "recommendation": "ユーザー入力をサニタイズしてください"
        },
        "hardcoded_secrets": {
            "patterns": [
                r"(api_key|apikey|secret|password|token)\s*=\s*['\"]",
                r"(AWS_ACCESS_KEY|ANTHROPIC_API_KEY)\s*=\s*['\"]",
            ],
            "severity": "critical",
            "recommendation": "環境変数またはシークレットマネージャーを使用してください"
        },
        "insecure_deserialization": {
            "patterns": [
                r"pickle\.loads?\(",
                r"yaml\.load\(",
                r"eval\(",
                r"exec\(",
            ],
            "severity": "high",
            "recommendation": "安全なデシリアライズ方法を使用してください"
        }
    }

    def scan_code(self, code: str, file_path: str) -> list[dict]:
        """コードをスキャンしてセキュリティ問題を検出"""
        findings = []

        for check_name, check in self.SECURITY_CHECKS.items():
            for pattern in check["patterns"]:
                matches = re.finditer(pattern, code)
                for match in matches:
                    line_num = code[:match.start()].count("\n") + 1
                    findings.append({
                        "check": check_name,
                        "file": file_path,
                        "line": line_num,
                        "severity": check["severity"],
                        "match": match.group(),
                        "recommendation": check["recommendation"]
                    })

        return findings
```

---

## 8. アンチパターン

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
agent.generate_code() -> git push -> auto-merge

# OK: 必ず人間のレビューを挟む
agent.generate_code() -> PR作成 -> 人間レビュー -> CI通過 -> マージ
```

### アンチパターン3: テストなしでの変更

```
# NG: テストを実行せずにコード変更
エージェントにコード修正を依頼 -> そのまま適用

# OK: テスト実行を必須にする
エージェントにコード修正を依頼
-> テスト実行（RED確認）
-> 修正適用
-> テスト実行（GREEN確認）
-> 既存テストも全パス確認
```

### アンチパターン4: 大きすぎるタスクの一括指示

```
# NG: プロジェクト全体のリファクタリングを一度に依頼
"このプロジェクト全体をTypeScriptに移行して"

# OK: 小さなタスクに分割
1. "src/utils/helpers.js をTypeScriptに変換して"
2. "src/api/users.js をTypeScriptに変換して"
3. "tsconfig.jsonを作成して、段階的な移行ができるように設定して"
```

### アンチパターン5: エージェントの出力を盲信

```
# NG: エージェントが生成したコードを無検証で採用
"エージェントが書いたから正しいはず"

# OK: 必ず検証する
1. 生成コードの静的解析（lint, type check）
2. テストの実行と結果確認
3. コードロジックの人間によるレビュー
4. セキュリティスキャン
5. パフォーマンス影響の確認
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |
---

## 9. FAQ

### Q1: コーディングエージェントは人間の仕事を奪うか？

現時点では「奪う」というより「変える」。エージェントが得意なのはボイラープレート・テスト・バグ修正などの定型的タスク。人間は要件定義・アーキテクチャ設計・レビュー・ユーザー体験など高レベルな判断に注力するようになる。エンジニアの役割は「コードを書く人」から「コードを書くエージェントを導く人」に変化しつつある。

### Q2: どの程度のコードベースサイズまで対応できる？

現在のコーディングエージェントはコンテキストウィンドウの制約により、**一度に扱えるのは数十ファイル程度**。大規模コードベースではRAG（コード検索）や、タスクの範囲を絞ることが重要。プロジェクト全体を理解するのではなく、関連する部分を効率的に検索する設計が必要。

### Q3: エージェントが書いたコードの品質保証は？

3段階のチェックを推奨:
1. **自動テスト**: CI/CDでの自動テスト通過
2. **静的解析**: lint, 型チェック, セキュリティスキャン
3. **人間レビュー**: アーキテクチャ整合性、ビジネスロジックの正しさ

### Q4: コーディングエージェントの選び方は？

以下の基準で選択:
- **タスクの自律性要件**: 補完(L1)→対話(L2)→自律(L3-4)
- **開発環境**: CLI派→Claude Code/Aider、IDE派→Cursor/Cline
- **チーム規模**: 個人→何でも可、チーム→統一ツール推奨
- **セキュリティ要件**: オンプレ必須→OSS(Aider/Cline)、クラウド可→商用ツール
- **予算**: 無料→Copilot無料枠/OSS、有料→Claude Code/Cursor Pro

### Q5: エージェントに書かせるべきでないコードは？

- **セキュリティクリティカルなコード**: 認証、暗号化、権限管理（人間の専門的レビュー必須）
- **規制対応コード**: 金融、医療（コンプライアンス確認が必要）
- **アーキテクチャの根幹**: システム設計の中核部分（人間の設計判断が必要）
- **パフォーマンスクリティカルなコード**: ベンチマーク測定が必要な箇所

---

## 10. 実践的な活用シナリオ

### 10.1 新規プロジェクトのスキャフォールディング

```python
class ScaffoldAgent(CodingAgent):
    """プロジェクトの初期構成を自動生成"""

    def scaffold_project(
        self,
        project_type: str,
        name: str,
        features: list[str]
    ) -> str:
        """プロジェクトテンプレートを生成"""
        return self.run(f"""
以下の仕様でプロジェクトの初期構成を作成してください。

プロジェクトタイプ: {project_type}
プロジェクト名: {name}
必要な機能: {', '.join(features)}

作成するもの:
1. ディレクトリ構造
2. 設定ファイル（package.json / pyproject.toml 等）
3. Docker関連ファイル（Dockerfile, docker-compose.yml）
4. CI/CD設定（.github/workflows/）
5. テスト設定
6. Lint/フォーマッター設定
7. README.md
8. .gitignore
9. サンプルコード（Hello World レベル）
10. サンプルテスト

各ファイルにはコメントで設計意図を記述してください。
""")
```

### 10.2 APIドキュメント自動生成

```python
class DocGenerationAgent(CodingAgent):
    """コードからドキュメントを自動生成"""

    def generate_api_docs(self, source_dir: str) -> str:
        """APIドキュメントを生成"""
        return self.run(f"""
以下のディレクトリのAPIコードを読んで、
OpenAPI仕様のドキュメントを生成してください。

ソースディレクトリ: {source_dir}

出力:
1. 各エンドポイントの説明
2. リクエスト/レスポンスのスキーマ
3. 認証方法
4. エラーレスポンスの一覧
5. 使用例（curlコマンド）

形式: OpenAPI 3.0 YAML
""")

    def generate_changelog(self, from_tag: str, to_tag: str) -> str:
        """Git履歴からCHANGELOGを生成"""
        return self.run(f"""
Gitの履歴から {from_tag} 以降 {to_tag} までの
変更履歴（CHANGELOG）を生成してください。

手順:
1. git log {from_tag}..{to_tag} でコミット一覧を取得
2. コミットメッセージをカテゴリ分類（feat, fix, chore等）
3. 影響するファイルから変更の重要度を判定
4. CHANGELOG.md形式で出力

出力形式: Keep a Changelog 形式
""")
```

### 10.3 テストカバレッジ改善エージェント

```python
class TestCoverageAgent(CodingAgent):
    """テストカバレッジを分析して不足を補完"""

    def improve_coverage(
        self,
        target_dir: str,
        min_coverage: float = 80.0
    ) -> str:
        """テストカバレッジを目標値まで改善"""
        return self.run(f"""
以下のディレクトリのテストカバレッジを改善してください。

対象ディレクトリ: {target_dir}
目標カバレッジ: {min_coverage}%

手順:
1. 現在のカバレッジレポートを生成して確認
2. カバレッジが低いファイル/関数を特定
3. 以下の優先度でテストを追加:
   a. カバレッジ0%の関数
   b. 条件分岐が網羅されていない関数
   c. エッジケースのテスト不足
   d. エラーハンドリングのテスト不足
4. 各テスト追加後にカバレッジを再計測
5. 目標カバレッジに到達するまで繰り返す

注意:
- カバレッジの数値だけでなく、意味のあるテストを書く
- 各テストにはテストの意図をコメントで記述
- 既存のテストスタイルに合わせる
""")
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 定義 | コードの理解・生成・修正・テストを自律的に行うエージェント |
| 主要製品 | Claude Code, Devin, Cursor, Copilot, Aider, Cline |
| アーキテクチャ | LLM + ファイル操作 + コマンド実行 + コード検索 |
| 実装パターン | TDD, コードレビュー, リファクタリング, マイグレーション |
| 得意領域 | バグ修正、テスト作成、リファクタリング |
| 限界 | アーキテクチャ設計、大規模改修、コンテキスト制約 |
| 品質保証 | 自動テスト + 静的解析 + 人間レビューの3段階 |
| セキュリティ | コマンド制限、パス保護、セキュリティスキャン |

## 次に読むべきガイド

- [01-research-agents.md](./01-research-agents.md) -- リサーチエージェント
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) -- エージェントの評価
- [../04-production/01-safety.md](../04-production/01-safety.md) -- コーディングエージェントの安全性

## 参考文献

1. Jimenez, C. E. et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2023) -- https://arxiv.org/abs/2310.06770
2. Anthropic, "Claude Code" -- https://docs.anthropic.com/en/docs/claude-code
3. Yang, J. et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (2024) -- https://arxiv.org/abs/2405.15793
4. Cursor Documentation -- https://docs.cursor.com/
5. Aider Documentation -- https://aider.chat/docs/
