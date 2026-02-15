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

## 9. アーキテクチャドキュメントの自動生成

### 9.1 コードベースからMermaidダイアグラムを生成

```python
# ソースコードを解析してアーキテクチャダイアグラムを自動生成

import ast
from pathlib import Path
from typing import NamedTuple

class DependencyInfo(NamedTuple):
    source: str
    target: str
    relationship: str  # "imports", "inherits", "uses"

class ArchitectureDiagramGenerator:
    """コードベースからMermaidダイアグラムを自動生成"""

    def __init__(self, project_root: str):
        self.root = Path(project_root)
        self.dependencies: list[DependencyInfo] = []
        self.modules: dict[str, dict] = {}

    def analyze_python_project(self) -> dict:
        """Pythonプロジェクトの依存関係を解析"""
        for py_file in self.root.rglob("*.py"):
            if any(skip in str(py_file) for skip in [
                "__pycache__", "node_modules", ".venv", "venv", "test"
            ]):
                continue

            relative_path = py_file.relative_to(self.root)
            module_name = str(relative_path).replace("/", ".").replace(".py", "")

            try:
                tree = ast.parse(py_file.read_text())
                imports = self._extract_imports(tree)
                classes = self._extract_classes(tree)
                functions = self._extract_functions(tree)

                self.modules[module_name] = {
                    "path": str(relative_path),
                    "imports": imports,
                    "classes": classes,
                    "functions": functions,
                    "loc": len(py_file.read_text().splitlines()),
                }

                for imp in imports:
                    self.dependencies.append(DependencyInfo(
                        source=module_name,
                        target=imp,
                        relationship="imports",
                    ))
            except SyntaxError:
                pass

        return {
            "modules": self.modules,
            "dependencies": self.dependencies,
        }

    def generate_component_diagram(self) -> str:
        """コンポーネント図をMermaid形式で生成"""
        lines = ["graph TD"]

        # モジュールをレイヤーでグループ化
        layers = self._detect_layers()

        for layer_name, modules in layers.items():
            lines.append(f"    subgraph {layer_name}")
            for mod in modules:
                class_count = len(self.modules.get(mod, {}).get("classes", []))
                func_count = len(self.modules.get(mod, {}).get("functions", []))
                label = f"{mod.split('.')[-1]}\\n({class_count}classes, {func_count}funcs)"
                lines.append(f'        {mod.replace(".", "_")}["{label}"]')
            lines.append("    end")

        # 依存関係の矢印
        for dep in self.dependencies:
            if dep.target in self.modules:
                source_id = dep.source.replace(".", "_")
                target_id = dep.target.replace(".", "_")
                lines.append(f"    {source_id} --> {target_id}")

        return "\n".join(lines)

    def generate_class_diagram(self) -> str:
        """クラス図をMermaid形式で生成"""
        lines = ["classDiagram"]

        for mod_name, mod_info in self.modules.items():
            for cls in mod_info.get("classes", []):
                cls_name = cls["name"]
                lines.append(f"    class {cls_name} {{")
                for method in cls.get("methods", []):
                    visibility = "+" if not method.startswith("_") else "-"
                    lines.append(f"        {visibility}{method}()")
                lines.append("    }")

                # 継承関係
                for base in cls.get("bases", []):
                    lines.append(f"    {base} <|-- {cls_name}")

        return "\n".join(lines)

    def generate_ai_prompt(self) -> str:
        """AI にアーキテクチャ解説を依頼するプロンプトを生成"""
        return f"""
以下のプロジェクト構造を分析し、アーキテクチャドキュメントを作成してください。

## モジュール一覧（{len(self.modules)}モジュール）
{self._format_module_summary()}

## 依存関係（{len(self.dependencies)}件）
{self._format_dependency_summary()}

## 出力形式
1. アーキテクチャ概要（3-5文）
2. レイヤー構成の説明
3. 主要コンポーネントの責務
4. データフローの説明
5. 改善提案（あれば）
"""

    def _detect_layers(self) -> dict[str, list[str]]:
        """モジュール名からレイヤーを自動検出"""
        layer_keywords = {
            "Presentation": ["controller", "handler", "view", "route", "api"],
            "Application": ["service", "usecase", "command", "query"],
            "Domain": ["model", "entity", "domain", "aggregate"],
            "Infrastructure": ["repository", "adapter", "client", "db"],
        }
        layers: dict[str, list[str]] = {}
        for mod_name in self.modules:
            mod_lower = mod_name.lower()
            placed = False
            for layer, keywords in layer_keywords.items():
                if any(kw in mod_lower for kw in keywords):
                    layers.setdefault(layer, []).append(mod_name)
                    placed = True
                    break
            if not placed:
                layers.setdefault("Other", []).append(mod_name)
        return layers

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _extract_classes(self, tree: ast.AST) -> list[dict]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                bases = [ast.unparse(b) for b in node.bases]
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "bases": bases,
                })
        return classes

    def _extract_functions(self, tree: ast.AST) -> list[str]:
        return [
            node.name for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def _format_module_summary(self) -> str:
        return "\n".join(
            f"- {name}: {info['loc']}行, "
            f"クラス{len(info['classes'])}個, 関数{len(info['functions'])}個"
            for name, info in sorted(self.modules.items())
        )

    def _format_dependency_summary(self) -> str:
        return "\n".join(
            f"- {d.source} → {d.target} ({d.relationship})"
            for d in self.dependencies[:20]
        )
```

### 9.2 ADR（Architecture Decision Records）の自動生成

```python
# AI で ADR のドラフトを自動生成

ADR_TEMPLATE_PROMPT = """
以下の設計判断について、ADR (Architecture Decision Record) を作成してください。

## 設計判断の概要
{decision_summary}

## コンテキスト
{context}

## 検討した選択肢
{options}

## ADR テンプレート（以下の形式で出力）

# ADR-{adr_number}: {title}

## ステータス
提案中 / 承認済み / 廃止

## コンテキスト
（この決定が必要になった背景・課題を記述）

## 決定
（採用した解決策を具体的に記述）

## 検討した選択肢
### 選択肢A: ...
- 利点: ...
- 欠点: ...

### 選択肢B: ...
- 利点: ...
- 欠点: ...

### 選択肢C: ...
- 利点: ...
- 欠点: ...

## 決定の根拠
（なぜこの選択肢を選んだかの理由を記述）

## 影響
- 良い影響: ...
- リスク: ...
- 移行計画: ...

## 参考情報
- 関連するADR: ...
- 参考文献: ...
"""

class ADRGenerator:
    """ADRの自動生成と管理"""

    def __init__(self, adr_dir: str = "docs/adr"):
        self.adr_dir = Path(adr_dir)
        self.adr_dir.mkdir(parents=True, exist_ok=True)

    def get_next_number(self) -> int:
        """次のADR番号を取得"""
        existing = list(self.adr_dir.glob("*.md"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split("-")[0])
                numbers.append(num)
            except (ValueError, IndexError):
                pass
        return max(numbers, default=0) + 1

    def generate_adr(self, decision: dict, client) -> str:
        """AIでADRのドラフトを生成"""
        adr_number = self.get_next_number()
        prompt = ADR_TEMPLATE_PROMPT.format(
            adr_number=adr_number,
            title=decision.get("title", ""),
            decision_summary=decision.get("summary", ""),
            context=decision.get("context", ""),
            options=decision.get("options", ""),
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        adr_content = response.content[0].text
        filename = f"{adr_number:04d}-{decision['title'].lower().replace(' ', '-')}.md"
        filepath = self.adr_dir / filename
        filepath.write_text(adr_content)

        return str(filepath)
```

---

## 10. ドキュメント鮮度モニタリング

### 10.1 自動鮮度チェックシステム

```python
# ドキュメントの鮮度を自動的に監視し、陳腐化を防止する

import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class DocFreshnessReport:
    """ドキュメント鮮度レポート"""
    file_path: str
    last_modified: datetime
    related_code_modified: datetime
    days_stale: int
    staleness_level: str  # "fresh", "aging", "stale", "critical"
    related_changes: list[str] = field(default_factory=list)

class DocFreshnessMonitor:
    """ドキュメントの鮮度を監視するシステム"""

    STALENESS_THRESHOLDS = {
        "README.md": 30,           # 30日
        "CONTRIBUTING.md": 90,     # 90日
        "docs/api/": 14,           # 14日
        "docs/architecture/": 60,  # 60日
        "CHANGELOG.md": 7,         # 7日（リリースサイクルに依存）
    }

    def check_freshness(self, doc_path: str) -> DocFreshnessReport:
        """ドキュメントの鮮度をチェック"""
        # ドキュメントの最終更新日
        doc_modified = self._get_last_modified(doc_path)

        # 関連コードの最終更新日
        related_code = self._find_related_code(doc_path)
        code_modified = max(
            (self._get_last_modified(f) for f in related_code),
            default=doc_modified,
        )

        # 鮮度の計算
        days_stale = (datetime.now() - doc_modified).days
        code_days_ahead = (code_modified - doc_modified).days

        # 鮮度レベルの判定
        threshold = self._get_threshold(doc_path)
        if code_days_ahead > threshold:
            staleness_level = "critical"
        elif code_days_ahead > threshold // 2:
            staleness_level = "stale"
        elif days_stale > threshold:
            staleness_level = "aging"
        else:
            staleness_level = "fresh"

        # 関連する変更の取得
        related_changes = self._get_changes_since(doc_modified, related_code)

        return DocFreshnessReport(
            file_path=doc_path,
            last_modified=doc_modified,
            related_code_modified=code_modified,
            days_stale=days_stale,
            staleness_level=staleness_level,
            related_changes=related_changes,
        )

    def generate_freshness_report(self, doc_paths: list[str]) -> str:
        """ドキュメント鮮度の全体レポートを生成"""
        reports = [self.check_freshness(path) for path in doc_paths]

        critical = [r for r in reports if r.staleness_level == "critical"]
        stale = [r for r in reports if r.staleness_level == "stale"]
        aging = [r for r in reports if r.staleness_level == "aging"]
        fresh = [r for r in reports if r.staleness_level == "fresh"]

        output = "# ドキュメント鮮度レポート\n\n"
        output += f"生成日時: {datetime.now().isoformat()}\n\n"
        output += f"## サマリー\n"
        output += f"- 最新: {len(fresh)}件\n"
        output += f"- 経年: {len(aging)}件\n"
        output += f"- 要更新: {len(stale)}件\n"
        output += f"- 緊急: {len(critical)}件\n\n"

        if critical:
            output += "## 緊急対応が必要なドキュメント\n\n"
            for r in critical:
                output += f"- **{r.file_path}**: "
                output += f"{r.days_stale}日前に更新、"
                output += f"関連コードは{(r.related_code_modified - r.last_modified).days}日先行\n"
                for change in r.related_changes[:3]:
                    output += f"  - {change}\n"

        return output

    def _get_last_modified(self, file_path: str) -> datetime:
        """Gitからファイルの最終更新日を取得"""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%aI", "--", file_path],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                return datetime.fromisoformat(result.stdout.strip())
        except Exception:
            pass
        return datetime.now()

    def _get_threshold(self, doc_path: str) -> int:
        """ドキュメントパスに応じた閾値を返す"""
        for pattern, threshold in self.STALENESS_THRESHOLDS.items():
            if pattern in doc_path:
                return threshold
        return 30  # デフォルト30日

    def _find_related_code(self, doc_path: str) -> list[str]:
        """ドキュメントに関連するソースコードファイルを推定"""
        related = []
        # ドキュメント内のファイル参照を解析
        # 例: README.md → src/ 配下のファイル
        # 例: docs/api/users.md → src/controllers/users.ts
        return related

    def _get_changes_since(self, since: datetime,
                           files: list[str]) -> list[str]:
        """指定日時以降の変更を取得"""
        changes = []
        for f in files:
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline",
                     f"--since={since.isoformat()}", "--", f],
                    capture_output=True, text=True, timeout=5,
                )
                if result.stdout.strip():
                    changes.extend(result.stdout.strip().split("\n"))
            except Exception:
                pass
        return changes
```

### 10.2 Slack通知との統合

```python
# ドキュメント鮮度レポートをSlackに自動通知

class DocFreshnessNotifier:
    """ドキュメント鮮度をSlackに通知"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def notify_stale_docs(self, reports: list[DocFreshnessReport]) -> None:
        """陳腐化したドキュメントをSlackに通知"""
        import requests

        stale_docs = [r for r in reports if r.staleness_level in ("stale", "critical")]
        if not stale_docs:
            return

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"ドキュメント鮮度アラート（{len(stale_docs)}件）",
                }
            },
        ]

        for doc in stale_docs[:5]:
            emoji = "🔴" if doc.staleness_level == "critical" else "🟡"
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *{doc.file_path}*\n"
                        f"最終更新: {doc.days_stale}日前 | "
                        f"関連コードとの差: "
                        f"{(doc.related_code_modified - doc.last_modified).days}日"
                    ),
                }
            })

        payload = {"blocks": blocks}
        requests.post(self.webhook_url, json=payload)
```

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
| アーキテクチャ図 | コードベース解析からMermaidダイアグラムを自動生成 |
| ADR | AI でドラフトを生成し、設計判断の記録を効率化 |

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
