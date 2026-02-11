# AIドキュメント生成 -- README、API仕様、技術文書の自動化

> AIを活用してプロジェクトのドキュメントを効率的に生成・保守する手法を学び、README・API仕様書・アーキテクチャ文書・変更履歴の自動生成パイプラインを構築して開発者体験（DX）を向上させる

## この章で学ぶこと

1. **AIドキュメント生成の基盤技術** -- LLMによるコード解析、JSDoc/docstringからの仕様書生成、コンテキスト理解の仕組み
2. **実装パイプライン** -- README自動生成、OpenAPI仕様書生成、CHANGELOG自動作成、アーキテクチャ図の生成
3. **品質管理と運用** -- 生成ドキュメントのレビュープロセス、CI/CD統合、鮮度維持の自動化戦略

---

## 1. AIドキュメント生成の全体像

### 1.1 ドキュメント生成パイプライン

```
AI ドキュメント生成パイプライン

  ソースコード           AI 処理               出力
  +----------+         +------------------+   +----------+
  | ソースコード|         | 1. コード解析     |   | README.md|
  | (*.ts,    | ------> | 2. 構造理解       | ->| API仕様  |
  |  *.py)    |         | 3. 文章生成       |   | CHANGELOG|
  +----------+         | 4. フォーマット    |   +----------+
  +----------+         |                  |   +----------+
  | コメント   | ------> |                  | ->| アーキ   |
  | JSDoc     |         +------------------+   | テクチャ図|
  | docstring |                                +----------+
  +----------+
  +----------+
  | Git履歴   | ------> [差分分析 + 要約]  --> | 変更履歴  |
  | PR/Issue  |                                +----------+
  +----------+
```

### 1.2 技術スタック

```
AI ドキュメント生成 技術マップ

  LLM / AI モデル
  ├── Claude            --- コード理解・文書生成（長文コンテキスト）
  ├── GPT-4             --- 汎用的な文書生成
  ├── GitHub Copilot    --- インラインドキュメント生成
  └── Gemini            --- 大規模コードベース解析

  ドキュメント生成ツール
  ├── TypeDoc           --- TypeScript API ドキュメント
  ├── Sphinx            --- Python ドキュメント
  ├── Swagger/OpenAPI   --- REST API 仕様書
  ├── Storybook         --- UIコンポーネントカタログ
  └── Mermaid           --- ダイアグラム生成

  CI/CD 統合
  ├── GitHub Actions    --- 自動生成・デプロイ
  ├── Pre-commit hooks  --- コミット時のドキュメントチェック
  └── Dependabot        --- 依存関係ドキュメント更新

  ホスティング
  ├── GitHub Pages      --- 静的サイト公開
  ├── Notion API        --- チームWiki連携
  └── Confluence API    --- エンタープライズWiki
```

### 1.3 ドキュメント種別と生成戦略

```
  ドキュメント種別         生成方法              更新頻度
  ──────────────────────────────────────────────────
  README.md               AI + 手動レビュー     リリースごと
  API 仕様書 (OpenAPI)     コードから自動生成    コミットごと
  CHANGELOG               Git 履歴から自動生成  リリースごと
  アーキテクチャ図          AI + 手動調整        大きな変更時
  コードコメント           Copilot + 手動       コーディング時
  オンボーディング文書      AI 初稿 + 手動改善   四半期ごと
  ADR (決定記録)           AI テンプレート + 手動 設計判断時
```

---

## 2. README 自動生成

### 2.1 コードベース解析から README を生成

```python
# AI による README 自動生成スクリプト
import os
import json
from pathlib import Path

class ReadmeGenerator:
    """プロジェクト構造を解析して README を自動生成"""

    def __init__(self, project_root: str):
        self.root = Path(project_root)
        self.analysis = {}

    def analyze_project(self) -> dict:
        """プロジェクト構造を解析"""
        self.analysis = {
            "name": self._detect_project_name(),
            "language": self._detect_language(),
            "framework": self._detect_framework(),
            "dependencies": self._parse_dependencies(),
            "scripts": self._parse_scripts(),
            "directory_structure": self._get_directory_tree(),
            "entry_points": self._find_entry_points(),
            "env_vars": self._detect_env_vars(),
            "license": self._detect_license(),
        }
        return self.analysis

    def _detect_project_name(self) -> str:
        """package.json, pyproject.toml 等からプロジェクト名を取得"""
        pkg_json = self.root / "package.json"
        if pkg_json.exists():
            data = json.loads(pkg_json.read_text())
            return data.get("name", self.root.name)

        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            # TOML パースしてプロジェクト名を取得
            import tomllib
            data = tomllib.loads(pyproject.read_text())
            return data.get("project", {}).get("name", self.root.name)

        return self.root.name

    def _detect_language(self) -> list[str]:
        """ファイル拡張子からプログラミング言語を推定"""
        extensions = {}
        for f in self.root.rglob("*"):
            if f.is_file() and not any(
                p in str(f) for p in ["node_modules", ".git", "__pycache__", "venv"]
            ):
                ext = f.suffix
                extensions[ext] = extensions.get(ext, 0) + 1

        lang_map = {
            ".ts": "TypeScript", ".tsx": "TypeScript",
            ".js": "JavaScript", ".jsx": "JavaScript",
            ".py": "Python", ".go": "Go", ".rs": "Rust",
            ".java": "Java", ".rb": "Ruby", ".swift": "Swift",
        }

        detected = []
        for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
            if ext in lang_map and lang_map[ext] not in detected:
                detected.append(lang_map[ext])
        return detected[:3]

    def _parse_dependencies(self) -> dict:
        """依存関係を解析"""
        deps = {"runtime": [], "dev": []}
        pkg_json = self.root / "package.json"
        if pkg_json.exists():
            data = json.loads(pkg_json.read_text())
            deps["runtime"] = list(data.get("dependencies", {}).keys())
            deps["dev"] = list(data.get("devDependencies", {}).keys())
        return deps

    def _parse_scripts(self) -> dict:
        """実行可能なスクリプトを解析"""
        pkg_json = self.root / "package.json"
        if pkg_json.exists():
            data = json.loads(pkg_json.read_text())
            return data.get("scripts", {})
        return {}

    def generate_readme(self) -> str:
        """解析結果から README を生成"""
        if not self.analysis:
            self.analyze_project()

        a = self.analysis
        sections = [
            f"# {a['name']}\n",
            self._generate_badges(a),
            self._generate_description(a),
            self._generate_quick_start(a),
            self._generate_installation(a),
            self._generate_usage(a),
            self._generate_project_structure(a),
            self._generate_scripts_section(a),
            self._generate_env_section(a),
            self._generate_contributing(),
            self._generate_license(a),
        ]
        return "\n".join(filter(None, sections))

    def _generate_quick_start(self, a: dict) -> str:
        """クイックスタートセクションの生成"""
        scripts = a.get("scripts", {})
        lines = ["## Quick Start\n", "```bash"]

        if "dev" in scripts:
            lines.extend([
                f"git clone <repository-url>",
                f"cd {a['name']}",
                "npm install",
                "npm run dev",
            ])
        elif a.get("language") and "Python" in a["language"]:
            lines.extend([
                f"git clone <repository-url>",
                f"cd {a['name']}",
                "pip install -e .",
            ])

        lines.append("```")
        return "\n".join(lines)
```

### 2.2 AI プロンプトによる README 改善

```python
# LLM を使った README の品質改善
README_IMPROVEMENT_PROMPT = """
あなたは優秀なテクニカルライターです。
以下の自動生成された README を改善してください。

改善基準:
1. 最初の3行でプロジェクトの価値が伝わること
2. セットアップ手順が過不足なくコピペで動くこと
3. 主要な機能が箇条書きで一覧できること
4. コントリビューション方法が明確なこと
5. ライセンスが明記されていること

自動生成 README:
{auto_generated_readme}

プロジェクト解析結果:
{project_analysis}

改善された README をマークダウン形式で出力してください。
"""

def improve_readme_with_ai(auto_readme: str, analysis: dict, client) -> str:
    """AI で README を改善"""
    prompt = README_IMPROVEMENT_PROMPT.format(
        auto_generated_readme=auto_readme,
        project_analysis=json.dumps(analysis, ensure_ascii=False, indent=2),
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

---

## 3. API 仕様書の自動生成

### 3.1 コードから OpenAPI 仕様書を生成

```python
# FastAPI の場合: 自動で OpenAPI 仕様が生成される
from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="ユーザー管理 API",
    description="ユーザーの CRUD 操作を提供する REST API",
    version="1.0.0",
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc
    openapi_url="/openapi.json", # OpenAPI JSON
)

class UserCreate(BaseModel):
    """ユーザー作成リクエスト"""
    name: str = Field(..., min_length=1, max_length=100, description="ユーザー名")
    email: str = Field(..., description="メールアドレス")
    role: str = Field(default="member", description="ロール (admin, member, viewer)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "田中太郎", "email": "tanaka@example.com", "role": "member"}
            ]
        }
    }

class UserResponse(BaseModel):
    """ユーザーレスポンス"""
    id: int = Field(..., description="ユーザーID")
    name: str = Field(..., description="ユーザー名")
    email: str = Field(..., description="メールアドレス")
    role: str = Field(..., description="ロール")

@app.post(
    "/users",
    response_model=UserResponse,
    status_code=201,
    summary="ユーザーを作成",
    description="新しいユーザーを作成します。メールアドレスは一意である必要があります。",
    tags=["users"],
)
async def create_user(user: UserCreate):
    """
    ユーザーを新規作成します。

    - **name**: 1〜100文字のユーザー名
    - **email**: 有効なメールアドレス (一意制約)
    - **role**: admin, member, viewer のいずれか
    """
    # 実装...
    return UserResponse(id=1, **user.model_dump())


@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="ユーザーを取得",
    tags=["users"],
)
async def get_user(
    user_id: int = Path(..., ge=1, description="ユーザーID"),
):
    """指定された ID のユーザー情報を取得します。"""
    # 実装...
    pass
```

### 3.2 TypeScript の型定義からドキュメント生成

```typescript
// TypeDoc 用のドキュメントコメント
// TSDoc 形式で記述すると TypeDoc が自動でドキュメント生成

/**
 * ユーザーサービス
 *
 * ユーザーの作成・取得・更新・削除を担当するサービスクラス。
 * リポジトリパターンでデータアクセスを抽象化し、
 * ビジネスロジックを集中管理する。
 *
 * @example
 * ```typescript
 * const service = new UserService(userRepository);
 * const user = await service.createUser({
 *   name: "田中太郎",
 *   email: "tanaka@example.com",
 * });
 * ```
 *
 * @see {@link UserRepository} データアクセス層
 * @see {@link UserController} コントローラー層
 */
export class UserService {
  /**
   * ユーザーを作成する
   *
   * @param input - ユーザー作成パラメータ
   * @returns 作成されたユーザーオブジェクト
   * @throws {DuplicateEmailError} メールアドレスが既に登録されている場合
   * @throws {ValidationError} 入力値が不正な場合
   */
  async createUser(input: CreateUserInput): Promise<User> {
    // 実装...
    return {} as User;
  }

  /**
   * ユーザーを検索する
   *
   * @param query - 検索条件
   * @param options - ページネーションオプション
   * @returns ページネーション付きユーザーリスト
   *
   * @example
   * ```typescript
   * const result = await service.searchUsers(
   *   { role: "admin" },
   *   { page: 1, limit: 20 }
   * );
   * console.log(result.total); // 総件数
   * console.log(result.items); // ユーザー配列
   * ```
   */
  async searchUsers(
    query: SearchQuery,
    options: PaginationOptions
  ): Promise<PaginatedResult<User>> {
    // 実装...
    return {} as PaginatedResult<User>;
  }
}
```

### 3.3 docstring からの自動生成

```python
# Python の docstring から AI でドキュメントを拡充
import ast
import inspect

class DocstringEnhancer:
    """既存の docstring を AI で拡充するツール"""

    def extract_functions(self, source_code: str) -> list[dict]:
        """ソースコードから関数情報を抽出"""
        tree = ast.parse(source_code)
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None,
                    "docstring": ast.get_docstring(node),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "lineno": node.lineno,
                }
                functions.append(func_info)

        return functions

    def generate_enhanced_docstring(self, func_info: dict, client) -> str:
        """AI で拡充された docstring を生成"""
        prompt = f"""
以下の Python 関数に対して、Google スタイルの docstring を生成してください。

関数名: {func_info['name']}
引数: {func_info['args']}
戻り値: {func_info['returns']}
既存の docstring: {func_info['docstring'] or 'なし'}

以下を含めてください:
1. 関数の説明（1-2文）
2. Args セクション（各引数の型と説明）
3. Returns セクション（戻り値の説明）
4. Raises セクション（発生しうる例外）
5. Example セクション（使用例）
"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

---

## 4. CHANGELOG 自動生成

### 4.1 Git 履歴からの CHANGELOG 生成

```python
# Conventional Commits から CHANGELOG を自動生成
import subprocess
import re
from datetime import datetime

class ChangelogGenerator:
    """Git コミット履歴から CHANGELOG を自動生成"""

    COMMIT_TYPES = {
        "feat": "Features",
        "fix": "Bug Fixes",
        "docs": "Documentation",
        "style": "Styles",
        "refactor": "Code Refactoring",
        "perf": "Performance Improvements",
        "test": "Tests",
        "build": "Build System",
        "ci": "CI",
        "chore": "Chores",
    }

    # Conventional Commit パターン
    PATTERN = re.compile(
        r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore)"
        r"(?:\((?P<scope>[^)]+)\))?"
        r"(?P<breaking>!)?"
        r": (?P<description>.+)$"
    )

    def get_commits_since_tag(self, tag: str = None) -> list[dict]:
        """指定タグ以降のコミットを取得"""
        cmd = ["git", "log", "--pretty=format:%H|%s|%an|%aI"]
        if tag:
            cmd.append(f"{tag}..HEAD")

        result = subprocess.run(cmd, capture_output=True, text=True)
        commits = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) == 4:
                hash_, subject, author, date = parts
                match = self.PATTERN.match(subject)
                if match:
                    commits.append({
                        "hash": hash_[:8],
                        "type": match.group("type"),
                        "scope": match.group("scope"),
                        "breaking": bool(match.group("breaking")),
                        "description": match.group("description"),
                        "author": author,
                        "date": date,
                    })

        return commits

    def generate_changelog(self, version: str, tag: str = None) -> str:
        """CHANGELOG マークダウンを生成"""
        commits = self.get_commits_since_tag(tag)
        today = datetime.now().strftime("%Y-%m-%d")

        lines = [f"## [{version}] - {today}\n"]

        # Breaking Changes
        breaking = [c for c in commits if c["breaking"]]
        if breaking:
            lines.append("### BREAKING CHANGES\n")
            for c in breaking:
                scope = f"**{c['scope']}**: " if c["scope"] else ""
                lines.append(f"- {scope}{c['description']} ({c['hash']})")
            lines.append("")

        # タイプ別に分類
        grouped = {}
        for c in commits:
            type_label = self.COMMIT_TYPES.get(c["type"], c["type"])
            grouped.setdefault(type_label, []).append(c)

        for type_label, type_commits in grouped.items():
            lines.append(f"### {type_label}\n")
            for c in type_commits:
                scope = f"**{c['scope']}**: " if c["scope"] else ""
                lines.append(f"- {scope}{c['description']} ({c['hash']})")
            lines.append("")

        return "\n".join(lines)


# 使用例
generator = ChangelogGenerator()
changelog = generator.generate_changelog("1.2.0", tag="v1.1.0")
print(changelog)
```

### 4.2 AI によるリリースノート生成

```python
# Git 差分を AI で要約してリリースノートを生成

RELEASE_NOTE_PROMPT = """
以下の Git コミット一覧から、エンドユーザー向けのリリースノートを生成してください。

コミット一覧:
{commits}

要件:
1. 技術的な詳細ではなく、ユーザーにとっての価値を伝える
2. 「新機能」「改善」「修正」のカテゴリに分類
3. 各項目は1-2文で簡潔に
4. 日本語で記述
"""

def generate_release_notes(commits: list[dict], client) -> str:
    """AI でリリースノートを生成"""
    commits_text = "\n".join(
        f"- [{c['type']}] {c['description']}" for c in commits
    )
    prompt = RELEASE_NOTE_PROMPT.format(commits=commits_text)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

---

## 5. CI/CD 統合

### 5.1 GitHub Actions でのドキュメント自動生成

```yaml
# .github/workflows/docs.yml
name: Generate Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全履歴を取得（CHANGELOG生成用）

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      # API ドキュメント生成
      - name: Generate API docs
        run: npx typedoc --out docs/api src/

      # OpenAPI 仕様書の整合性チェック
      - name: Validate OpenAPI spec
        run: npx @redocly/cli lint openapi.yaml

      # README の鮮度チェック
      - name: Check README freshness
        run: |
          python scripts/check_readme_freshness.py \
            --readme README.md \
            --package package.json \
            --threshold 30  # 30日以上更新なしで警告

      # ドキュメントをデプロイ
      - name: Deploy docs
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs

  check-docs-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # エクスポートされた関数のドキュメントカバレッジ
      - name: Check documentation coverage
        run: |
          python scripts/doc_coverage.py \
            --src src/ \
            --min-coverage 80 \
            --report docs-coverage.json

      - name: Comment coverage on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const coverage = JSON.parse(fs.readFileSync('docs-coverage.json'));
            const body = `## Documentation Coverage\n\n` +
              `Coverage: **${coverage.percentage}%** (${coverage.documented}/${coverage.total})\n\n` +
              `${coverage.percentage >= 80 ? '✅' : '⚠️'} ` +
              `Minimum: 80%`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body,
            });
```

### 5.2 Pre-commit フックでのドキュメントチェック

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-docstrings
        name: Check docstrings
        entry: python scripts/check_docstrings.py
        language: python
        types: [python]
        args: ['--style', 'google', '--min-length', '10']

      - id: check-readme-links
        name: Check README links
        entry: python scripts/check_links.py
        language: python
        files: '\.md$'

      - id: generate-openapi
        name: Regenerate OpenAPI spec
        entry: python scripts/generate_openapi.py
        language: python
        files: '(routes|controllers|schemas)/.*\.py$'
        pass_filenames: false
```

---

## 6. 比較表

| ドキュメント種別 | 自動化度 | AI 活用効果 | 推奨ツール | 更新頻度 |
|----------------|:-------:|:--------:|:--------:|:------:|
| README | 中 | 高 (初稿生成) | Claude + 手動 | リリースごと |
| API 仕様書 (OpenAPI) | 高 | 中 (補足生成) | FastAPI / TypeDoc | コミットごと |
| CHANGELOG | 高 | 高 (要約) | Conventional Commits | リリースごと |
| コードコメント | 中 | 高 (初稿) | Copilot | コーディング時 |
| アーキテクチャ図 | 低 | 中 (Mermaid生成) | Claude + Mermaid | 設計変更時 |
| ADR | 低 | 中 (テンプレート) | Claude + 手動 | 決定時 |

| アプローチ | 品質 | 速度 | コスト | 保守性 |
|-----------|:----:|:---:|:-----:|:-----:|
| 完全手動 | 最高 | 低 | 高 (人件費) | 低 (陳腐化) |
| AI 初稿 + 人間レビュー | 高 | 高 | 中 | 高 |
| コードから完全自動生成 | 中 | 最高 | 低 | 最高 |
| AI のみ (レビューなし) | 低〜中 | 最高 | 最低 | 中 |

---

## 7. アンチパターン

### アンチパターン 1: AI 生成ドキュメントを無検証で公開

```
BAD:
  AI が生成した API ドキュメントをそのまま公開
  → 実装と異なる記述、存在しないエンドポイントの記載
  → ハルシネーション（AI の幻覚）による誤情報
  → 利用者が誤った情報に基づいて実装し、障害発生

GOOD:
  1. AI で初稿を生成（速度向上）
  2. 実際のコードとの整合性を自動チェック
  3. テクニカルライターまたは開発者がレビュー
  4. サンプルコードは実際に動作確認
  5. CI で OpenAPI spec と実装の不整合を検出
```

### アンチパターン 2: ドキュメントの鮮度管理を怠る

```
BAD:
  プロジェクト開始時に立派な README を作成
  → 半年後、セットアップ手順が古くて動かない
  → API 仕様書が実装と乖離
  → 新メンバーが誤った情報でハマる

GOOD:
  - CI でドキュメントの最終更新日をチェック
  - package.json の変更時に README の依存関係セクションを自動更新
  - PR テンプレートに「ドキュメント更新の要否」チェックボックス
  - 月次でドキュメント鮮度レポートを生成
  - 「docs」ラベルの Issue を自動作成
```

### アンチパターン 3: 全てを1つの README に詰め込む

```
BAD:
  README.md が 2000 行超
  → セットアップ手順、API リファレンス、アーキテクチャ説明、
     トラブルシューティングが全て1ファイル
  → 必要な情報を見つけられない

GOOD:
  README.md はエントリーポイント（100行以内）にとどめる:
  - プロジェクト概要（3行）
  - クイックスタート（10行）
  - 主要機能一覧
  - 詳細ドキュメントへのリンク集
    - docs/setup.md  --- セットアップ手順
    - docs/api.md    --- API リファレンス
    - docs/architecture.md --- 設計文書
```

---

## 8. FAQ

### Q1. AI でドキュメントを生成する際の品質を確保するには？

**A.** (1) **コンテキストの提供**: ソースコード、テスト、既存ドキュメントをまとめて AI に渡すことで精度が向上する。(2) **テンプレートの活用**: プロジェクト固有のドキュメントテンプレートを定義し、AI に従わせる。(3) **自動検証**: 生成されたサンプルコードを CI で実行し、動作を確認する。(4) **段階的導入**: まず内部ドキュメント（ADR、設計メモ）から始め、精度を確認してから外部向けドキュメントに展開する。AI 生成ドキュメントの品質は、入力の品質に大きく依存する。

### Q2. ドキュメントの自動生成を CI に組み込むベストプラクティスは？

**A.** (1) **PR 時のチェック**: docstring カバレッジ、OpenAPI 整合性、リンク切れ検出を PR のチェック項目に含める。(2) **マージ時の生成**: main ブランチへのマージ時に API ドキュメントを自動再生成・デプロイする。(3) **リリース時の CHANGELOG**: タグ作成時に Conventional Commits から CHANGELOG を自動生成する。(4) **定期レポート**: 週次でドキュメント鮮度レポートを Slack に通知する。段階的に自動化範囲を広げるのが現実的。

### Q3. 小規模チームでドキュメント管理を効率化するには？

**A.** (1) **README 駆動開発**: コーディング前に README を書き、それを仕様として開発する。AI で初稿を生成すると高速。(2) **Docs as Code**: ドキュメントをコードと同じリポジトリで管理し、PR でレビューする。(3) **ADR の活用**: 設計判断を Architecture Decision Records として記録し、「なぜこの設計にしたか」を残す。(4) **自動化の優先順位**: まず CHANGELOG の自動生成、次に API ドキュメント、最後に README の鮮度管理の順で導入する。小規模チームこそ自動化の効果が大きい。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| README 生成 | プロジェクト構造を自動解析 → AI で初稿生成 → 人間がレビュー |
| API 仕様書 | FastAPI/TypeDoc で自動生成。Pydantic の型情報がそのまま仕様に |
| CHANGELOG | Conventional Commits + 自動生成。AI でリリースノートを要約 |
| CI/CD 統合 | ドキュメントカバレッジ、鮮度チェック、自動デプロイをパイプラインに組み込む |
| 品質管理 | AI 生成は初稿。必ず人間がレビューし、サンプルコードは動作確認 |
| 鮮度維持 | 自動チェック + PR テンプレート + 月次レポートで陳腐化を防止 |

---

## 次に読むべきガイド

- [AIデバッグ](./03-ai-debugging.md) -- AI を活用したデバッグ効率化
- [AIコーディング](./01-ai-coding.md) -- AI によるコード生成の実践
- [開発の未来](../03-team/02-future-of-development.md) -- AI 時代の開発プロセス展望

---

## 参考文献

1. **Docs for Developers** -- Jared Bhatt & Zachary Sarah Corleissen (Apress, 2021) -- 開発者向けドキュメント執筆ガイド
2. **Conventional Commits** -- https://www.conventionalcommits.org/ -- コミットメッセージ規約
3. **TypeDoc** -- https://typedoc.org/ -- TypeScript ドキュメント生成ツール
4. **OpenAPI Specification** -- https://spec.openapis.org/oas/latest.html -- REST API 仕様の標準
