# AIコードレビュー ── 自動レビュー、品質チェック

> AIを活用したコードレビューの自動化手法と品質チェックプロセスを理解し、レビューの速度と精度を大幅に向上させる体制を構築する。

---

## この章で学ぶこと

1. **AIコードレビューツールの活用** ── CodeRabbit、Claude Code等を使った自動レビューの導入方法を学ぶ
2. **レビュー観点の体系化** ── AIが検出すべき問題と人間が判断すべき問題の切り分けを理解する
3. **レビュープロセスの最適化** ── AI+人間のハイブリッドレビューで効率と品質を両立する方法を確立する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [AIテスト ── テスト生成、カバレッジ向上](./00-ai-testing.md) の内容を理解していること

---

## 1. AIコードレビューの全体像

### 1.1 レビュープロセスの変遷

```
従来のコードレビュー              AIハイブリッドレビュー
┌─────────────────┐              ┌─────────────────────┐
│                 │              │                     │
│  開発者がPR作成  │              │  開発者がPR作成      │
│      │          │              │      │              │
│      ▼          │              │      ▼              │
│  レビュアーが    │              │  AI自動レビュー      │
│  全コードを読む  │              │  (数秒で完了)        │
│  (30分-2時間)   │              │      │              │
│      │          │              │      ▼              │
│      ▼          │              │  AIが指摘した問題を  │
│  コメント記入    │              │  開発者が修正        │
│      │          │              │      │              │
│      ▼          │              │      ▼              │
│  修正→再レビュー │              │  人間レビュアーが    │
│  (繰り返し)     │              │  残り20%を確認      │
│      │          │              │  (10-20分)          │
│      ▼          │              │      │              │
│  マージ          │              │      ▼              │
│  (平均2-3日)    │              │  マージ              │
│                 │              │  (平均数時間)        │
└─────────────────┘              └─────────────────────┘
```

### 1.2 AIレビューが検出できる問題の範囲

```
┌──────────────────────────────────────────────────────┐
│          AIレビューの検出能力マップ                     │
│                                                      │
│  検出精度: 高い                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │ ・コーディング規約違反                         │    │
│  │ ・未使用の変数・import                         │    │
│  │ ・型の不整合                                   │    │
│  │ ・既知のセキュリティパターン（SQLi, XSS等）     │    │
│  │ ・パフォーマンスの一般的な問題（N+1等）         │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  検出精度: 中程度                                     │
│  ┌──────────────────────────────────────────────┐    │
│  │ ・設計パターンの不適切な使用                    │    │
│  │ ・エラーハンドリングの不備                      │    │
│  │ ・テストの不足                                 │    │
│  │ ・命名の改善提案                               │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  検出精度: 低い（人間が必要）                          │
│  ┌──────────────────────────────────────────────┐    │
│  │ ・ビジネスロジックの正しさ                      │    │
│  │ ・アーキテクチャの妥当性                        │    │
│  │ ・ユーザー体験への影響                          │    │
│  │ ・組織固有の運用ルール                          │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## 2. AIレビューツールの実装

### コード例1: CodeRabbitの設定

```yaml
# .coderabbit.yaml - CodeRabbit設定ファイル
language: "ja"  # 日本語でレビュー

reviews:
  profile: "assertive"  # 積極的にレビュー
  request_changes_workflow: true
  high_level_summary: true
  poem: false

  review_comment:
    nitpick: true
    security: true
    performance: true

  path_instructions:
    - path: "src/domain/**"
      instructions: |
        ドメイン層のレビュー:
        - 外部依存がないことを確認
        - ビジネスルールの不変条件をチェック
        - ドメインイベントが適切に発行されているか
    - path: "src/api/**"
      instructions: |
        API層のレビュー:
        - 入力バリデーションの漏れ
        - エラーレスポンスの形式統一
        - 認証・認可のチェック
    - path: "tests/**"
      instructions: |
        テストのレビュー:
        - アサーションが意味のあるものか
        - エッジケースが含まれているか
        - テストの独立性が保たれているか

chat:
  auto_reply: true
```

### コード例2: Claude Codeでのレビュー実行

```bash
# Claude Codeを使ったPRレビュー

# 方法1: git diffをレビュー
claude "git diff main...HEAD の変更をレビューして。
       以下の観点でチェック:
       1. セキュリティ: 入力検証、認証、暗号化
       2. パフォーマンス: N+1、メモリリーク、計算量
       3. 保守性: SOLID原則、命名、複雑度
       4. テスト: カバレッジ、エッジケース
       各問題に重要度(Critical/Major/Minor)をつけて"

# 方法2: GitHub PRをレビュー
claude "gh pr view 123 の変更をレビューして。
       CLAUDE.mdの規約に準拠しているかもチェックして"

# 方法3: 特定ファイルのレビュー
claude "src/services/payment.py の直近の変更をレビューして。
       特に決済ロジックのセキュリティに注目して"
```

### コード例3: カスタムレビュースクリプト

```python
#!/usr/bin/env python3
"""AI自動レビュースクリプト"""

import subprocess
import json

def get_diff() -> str:
    """PRの差分を取得"""
    result = subprocess.run(
        ["git", "diff", "main...HEAD", "--unified=5"],
        capture_output=True, text=True
    )
    return result.stdout

def get_changed_files() -> list[str]:
    """変更ファイル一覧を取得"""
    result = subprocess.run(
        ["git", "diff", "--name-only", "main...HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip().split("\n")

def categorize_changes(files: list[str]) -> dict[str, list[str]]:
    """変更ファイルをカテゴリ分け"""
    categories = {
        "domain": [], "api": [], "infra": [],
        "test": [], "config": [], "other": []
    }
    for f in files:
        if "domain" in f: categories["domain"].append(f)
        elif "api" in f or "presentation" in f: categories["api"].append(f)
        elif "infra" in f: categories["infra"].append(f)
        elif "test" in f: categories["test"].append(f)
        elif f.endswith((".yaml", ".toml", ".json")): categories["config"].append(f)
        else: categories["other"].append(f)
    return categories

def generate_review_prompt(diff: str, categories: dict) -> str:
    """カテゴリに応じたレビュープロンプトを生成"""
    prompt = f"""以下のコード変更をレビューしてください。

## 変更概要
{json.dumps(categories, indent=2, ensure_ascii=False)}

## レビュー観点
- Critical: セキュリティ脆弱性、データ損失リスク
- Major: バグ、パフォーマンス問題、設計違反
- Minor: 命名改善、コメント追加、リファクタリング提案

## 差分
```diff
{diff}
```

JSON形式で出力:
{{"findings": [{{"severity": "...", "file": "...", "line": N, "message": "..."}}]}}
"""
    return prompt

if __name__ == "__main__":
    diff = get_diff()
    files = get_changed_files()
    categories = categorize_changes(files)
    prompt = generate_review_prompt(diff, categories)

    # Claude Codeに渡して実行
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True
    )
    print(result.stdout)
```

### コード例4: GitHub Actionsでの自動レビュー

```yaml
# .github/workflows/ai-review.yml
name: AI Code Review
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
          git diff origin/main...HEAD > /tmp/pr-diff.txt
          echo "lines=$(wc -l < /tmp/pr-diff.txt)" >> $GITHUB_OUTPUT

      - name: AI Review (small PR)
        if: steps.diff.outputs.lines < 500
        uses: coderabbitai/ai-pr-reviewer@latest
        with:
          debug: false
          review_simple_changes: true
          review_comment_lgtm: false

      - name: AI Review (large PR)
        if: steps.diff.outputs.lines >= 500
        run: |
          echo "::warning::PRが大きすぎます（${lines}行）。分割を検討してください。"
          # 大きなPRは要約のみ生成
```

### コード例5: レビューコメントのテンプレート

```markdown
<!-- AI Review Comment Template -->

## AI自動レビュー結果

### Critical (即座に修正必要)
- [ ] `src/auth/login.py:42` - SQLインジェクションの可能性。
      パラメータ化クエリを使用してください。

### Major (マージ前に修正推奨)
- [ ] `src/services/order.py:128` - N+1クエリが発生しています。
      `selectinload` を使用してEager Loadingにしてください。
- [ ] `src/api/users.py:55` - 入力バリデーションが不足。
      emailフィールドのフォーマットチェックを追加してください。

### Minor (改善提案)
- [ ] `src/utils/helpers.py:12` - 関数名 `proc` は曖昧です。
      `process_payment_result` のように具体的にしてください。
- [ ] `tests/test_order.py:89` - アサーションが `is not None` のみ。
      具体的な値の検証を追加してください。

### 良い点
- テストカバレッジが85%で基準を満たしています
- ドメインモデルの設計が一貫しています
- エラーハンドリングが適切に実装されています

---
*このレビューはAIによる自動生成です。人間のレビュアーによる確認も必要です。*
```

---

## 3. レビュー効率の比較

### 3.1 レビュー手法別の比較

| 手法 | 所要時間 | 検出率 | コスト | 適用場面 |
|------|---------|--------|-------|---------|
| 人間のみ | 30-120分 | 60-70% | 高い | 設計判断が必要な変更 |
| AI+人間 | 10-30分 | 85-90% | 中程度 | 標準的なPR |
| AIのみ | 1-2分 | 50-60% | 低い | Botアカウントのコミット |
| Linter+AI | 5-10分 | 75-80% | 低い | 定型的な変更 |

### 3.2 AIレビューツール比較

| ツール | 対応プラットフォーム | 言語対応 | 料金 | 特徴 |
|--------|-------------------|---------|------|------|
| CodeRabbit | GitHub/GitLab | 多言語 | $15/月〜 | PR要約、逐行レビュー |
| Graphite | GitHub | 多言語 | 無料〜 | スタック型PRと統合 |
| Claude Code | CLI | 多言語 | 従量課金 | 深い文脈理解 |
| Amazon CodeGuru | AWS | Java/Python | 従量課金 | AWSサービス統合 |
| Bito | GitHub/GitLab | 多言語 | 無料〜 | セキュリティ重視 |

---

## 4. ハイブリッドレビューの運用

```
┌──────────────────────────────────────────────────────┐
│         ハイブリッドレビュー運用フロー                   │
│                                                      │
│  PR作成                                              │
│    │                                                 │
│    ├──► AIレビュー (自動、2分以内)                     │
│    │     ├── Critical → ブロック (マージ不可)          │
│    │     ├── Major → 修正要求                        │
│    │     └── Minor → コメント                        │
│    │                                                 │
│    ├──► 静的解析 (自動、5分以内)                      │
│    │     ├── Lint / Format / Type Check              │
│    │     └── Security Scan (SAST)                    │
│    │                                                 │
│    └──► 人間レビュー (AIレビュー後)                    │
│          ├── ビジネスロジックの正しさ                   │
│          ├── アーキテクチャの妥当性                     │
│          ├── ユーザー体験への影響                      │
│          └── AIが見逃した文脈的問題                    │
│                                                      │
│  全てパス → マージ                                    │
└──────────────────────────────────────────────────────┘
```

---

## 5. セキュリティレビューの自動化

### 5.1 セキュリティ特化レビュー設定

```python
# セキュリティ観点でのAIレビューを自動化

class SecurityReviewEngine:
    """セキュリティ特化のAIコードレビューエンジン"""

    SECURITY_PATTERNS = {
        "sql_injection": {
            "patterns": [
                r"f\".*SELECT.*{.*}\"",
                r"\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)",
                r"\+.*(?:SELECT|INSERT|UPDATE|DELETE)",
            ],
            "severity": "critical",
            "message": "SQLインジェクションの可能性があります。パラメータ化クエリを使用してください。",
            "fix_example": """
# BAD
query = f"SELECT * FROM users WHERE id = {user_id}"

# GOOD
query = "SELECT * FROM users WHERE id = :id"
result = session.execute(text(query), {"id": user_id})
""",
        },
        "xss": {
            "patterns": [
                r"dangerouslySetInnerHTML",
                r"innerHTML\s*=",
                r"document\.write\(",
            ],
            "severity": "critical",
            "message": "XSS（クロスサイトスクリプティング）の可能性があります。",
        },
        "hardcoded_secret": {
            "patterns": [
                r"(?:password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
                r"(?:AWS_SECRET|PRIVATE_KEY)\s*=\s*['\"]",
            ],
            "severity": "critical",
            "message": "ハードコードされた機密情報が含まれています。環境変数を使用してください。",
        },
        "insecure_random": {
            "patterns": [
                r"random\.random\(\)",
                r"Math\.random\(\)",
            ],
            "severity": "major",
            "message": "セキュリティ用途には暗号学的に安全な乱数生成器を使用してください。",
            "fix_example": """
# BAD
import random
token = random.randint(0, 999999)

# GOOD
import secrets
token = secrets.token_urlsafe(32)
""",
        },
        "path_traversal": {
            "patterns": [
                r"open\(.*\+.*\)",
                r"os\.path\.join\(.*request",
            ],
            "severity": "critical",
            "message": "パストラバーサルの可能性があります。入力パスを検証してください。",
        },
    }

    def review_file(self, file_content: str, filename: str) -> list[dict]:
        """ファイルをセキュリティ観点でレビュー"""
        import re
        findings = []

        for vuln_type, config in self.SECURITY_PATTERNS.items():
            for pattern in config["patterns"]:
                for i, line in enumerate(file_content.split("\n"), 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            "type": vuln_type,
                            "severity": config["severity"],
                            "file": filename,
                            "line": i,
                            "code": line.strip(),
                            "message": config["message"],
                            "fix_example": config.get("fix_example", ""),
                        })

        return findings

    def generate_security_report(self, findings: list[dict]) -> str:
        """セキュリティレビュー結果のレポートを生成"""
        if not findings:
            return "セキュリティ上の問題は検出されませんでした。"

        critical = [f for f in findings if f["severity"] == "critical"]
        major = [f for f in findings if f["severity"] == "major"]

        report = "## セキュリティレビュー結果\n\n"
        report += f"検出された問題: {len(findings)}件 "
        report += f"(Critical: {len(critical)}, Major: {len(major)})\n\n"

        if critical:
            report += "### Critical（即座に修正必要）\n\n"
            for f in critical:
                report += f"- **{f['type']}** - `{f['file']}:{f['line']}`\n"
                report += f"  {f['message']}\n"
                report += f"  ```\n  {f['code']}\n  ```\n"
                if f.get("fix_example"):
                    report += f"  修正例:\n  ```python\n{f['fix_example']}\n  ```\n"

        if major:
            report += "### Major（マージ前に修正推奨）\n\n"
            for f in major:
                report += f"- **{f['type']}** - `{f['file']}:{f['line']}`\n"
                report += f"  {f['message']}\n"

        return report
```

### 5.2 依存関係のセキュリティ監査

```python
# 依存パッケージの脆弱性チェックをAIレビューに統合

class DependencyAuditor:
    """依存関係のセキュリティ監査"""

    def audit_npm_dependencies(self, package_json_path: str) -> dict:
        """npm パッケージの脆弱性チェック"""
        import subprocess
        result = subprocess.run(
            ["npm", "audit", "--json"],
            capture_output=True, text=True,
            cwd=str(Path(package_json_path).parent),
        )

        try:
            audit_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"error": "npm audit の実行に失敗しました"}

        vulnerabilities = audit_data.get("vulnerabilities", {})
        summary = {
            "total": len(vulnerabilities),
            "critical": 0,
            "high": 0,
            "moderate": 0,
            "low": 0,
            "details": [],
        }

        for pkg_name, vuln_info in vulnerabilities.items():
            severity = vuln_info.get("severity", "unknown")
            if severity in summary:
                summary[severity] += 1
            summary["details"].append({
                "package": pkg_name,
                "severity": severity,
                "title": vuln_info.get("title", ""),
                "url": vuln_info.get("url", ""),
                "fix_available": vuln_info.get("fixAvailable", False),
            })

        return summary

    def generate_ai_review_comment(self, audit_result: dict) -> str:
        """監査結果をPRコメント形式に変換"""
        if audit_result.get("total", 0) == 0:
            return "依存関係に既知の脆弱性は見つかりませんでした。"

        comment = "## 依存関係セキュリティ監査\n\n"
        comment += f"| 深刻度 | 件数 |\n|--------|------|\n"
        comment += f"| Critical | {audit_result['critical']} |\n"
        comment += f"| High | {audit_result['high']} |\n"
        comment += f"| Moderate | {audit_result['moderate']} |\n"
        comment += f"| Low | {audit_result['low']} |\n\n"

        fixable = [d for d in audit_result["details"] if d["fix_available"]]
        if fixable:
            comment += "### 自動修正可能な脆弱性\n\n"
            comment += "`npm audit fix` で以下の脆弱性を修正できます:\n\n"
            for d in fixable:
                comment += f"- **{d['package']}** ({d['severity']}): {d['title']}\n"

        return comment
```

---

## 6. レビュー品質メトリクスの可視化

### 6.1 レビュー効率の測定

```python
# AIレビューの効果を定量的に測定するシステム

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class ReviewMetric:
    """レビューメトリクスのデータポイント"""
    pr_number: int
    pr_size: int              # 変更行数
    ai_review_time_sec: float  # AIレビュー所要時間
    human_review_time_min: float  # 人間レビュー所要時間
    ai_findings: int           # AIが検出した問題数
    human_findings: int        # 人間が検出した問題数
    ai_false_positives: int    # AI誤検出数
    ai_true_positives: int     # AI正確検出数
    time_to_merge_hours: float  # PR作成からマージまでの時間
    post_merge_bugs: int = 0   # マージ後に発見されたバグ数

@dataclass
class ReviewDashboard:
    """レビュー品質ダッシュボード"""
    metrics: list[ReviewMetric] = field(default_factory=list)

    @property
    def ai_precision(self) -> float:
        """AIレビューの精度（True Positive率）"""
        total_findings = sum(m.ai_findings for m in self.metrics)
        true_positives = sum(m.ai_true_positives for m in self.metrics)
        return true_positives / total_findings if total_findings > 0 else 0

    @property
    def avg_time_to_merge(self) -> float:
        """平均マージまでの時間（時間）"""
        if not self.metrics:
            return 0
        return sum(m.time_to_merge_hours for m in self.metrics) / len(self.metrics)

    @property
    def avg_human_review_time(self) -> float:
        """平均人間レビュー時間（分）"""
        if not self.metrics:
            return 0
        return sum(m.human_review_time_min for m in self.metrics) / len(self.metrics)

    @property
    def bug_escape_rate(self) -> float:
        """バグすり抜け率"""
        total_prs = len(self.metrics)
        if total_prs == 0:
            return 0
        prs_with_bugs = sum(1 for m in self.metrics if m.post_merge_bugs > 0)
        return prs_with_bugs / total_prs

    def generate_weekly_report(self) -> str:
        """週次レビュー品質レポートを生成"""
        return f"""
## コードレビュー品質レポート

### サマリー
- 対象PR数: {len(self.metrics)}
- AI精度: {self.ai_precision:.1%}
- 平均マージ時間: {self.avg_time_to_merge:.1f}時間
- 平均人間レビュー時間: {self.avg_human_review_time:.0f}分
- バグすり抜け率: {self.bug_escape_rate:.1%}

### AI検出内訳
- 総検出数: {sum(m.ai_findings for m in self.metrics)}
- 正確な検出: {sum(m.ai_true_positives for m in self.metrics)}
- 誤検出: {sum(m.ai_false_positives for m in self.metrics)}
- 人間のみが検出: {sum(m.human_findings for m in self.metrics)}

### トレンド
- 先週比マージ時間: {self._calc_trend('time_to_merge_hours')}
- 先週比AI精度: {self._calc_trend('ai_precision')}
"""

    def _calc_trend(self, metric_name: str) -> str:
        """トレンドを計算（簡易実装）"""
        return "改善中" if len(self.metrics) > 0 else "データ不足"
```

### 6.2 レビューコメントの分類と分析

```python
# AIレビューコメントの品質を追跡・改善

class ReviewCommentAnalyzer:
    """レビューコメントを分析し、AI設定の改善に活用"""

    def categorize_comments(self, comments: list[dict]) -> dict:
        """コメントをカテゴリ別に分類"""
        categories = {
            "security": [],
            "performance": [],
            "maintainability": [],
            "correctness": [],
            "style": [],
            "documentation": [],
            "test": [],
            "other": [],
        }

        category_keywords = {
            "security": ["セキュリティ", "脆弱性", "認証", "認可",
                         "injection", "XSS", "CSRF"],
            "performance": ["パフォーマンス", "N+1", "メモリ", "キャッシュ",
                           "インデックス", "計算量"],
            "maintainability": ["保守", "リファクタ", "SOLID", "複雑度",
                                "責務", "依存"],
            "correctness": ["バグ", "エラー", "例外", "null",
                           "境界", "競合"],
            "style": ["命名", "フォーマット", "規約", "インデント"],
            "documentation": ["ドキュメント", "コメント", "docstring",
                              "README"],
            "test": ["テスト", "カバレッジ", "アサーション", "モック"],
        }

        for comment in comments:
            text = comment.get("body", "").lower()
            categorized = False
            for cat, keywords in category_keywords.items():
                if any(kw.lower() in text for kw in keywords):
                    categories[cat].append(comment)
                    categorized = True
                    break
            if not categorized:
                categories["other"].append(comment)

        return categories

    def analyze_acceptance_rate(self, comments: list[dict]) -> dict:
        """コメントの受け入れ率を分析"""
        total = len(comments)
        accepted = sum(1 for c in comments if c.get("resolved", False))
        dismissed = sum(1 for c in comments if c.get("dismissed", False))
        pending = total - accepted - dismissed

        return {
            "total": total,
            "accepted": accepted,
            "dismissed": dismissed,
            "pending": pending,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "dismiss_rate": dismissed / total if total > 0 else 0,
        }

    def suggest_config_improvements(self, analysis: dict) -> list[str]:
        """分析結果からAI設定の改善提案を生成"""
        suggestions = []

        if analysis.get("dismiss_rate", 0) > 0.4:
            suggestions.append(
                "誤検出率が高い（40%以上）。path_instructions を見直し、"
                "プロジェクト固有のルールを追加してください。"
            )

        style_ratio = len(analysis.get("categories", {}).get("style", [])) / max(analysis.get("total", 1), 1)
        if style_ratio > 0.5:
            suggestions.append(
                "スタイル関連のコメントが50%以上。Linter/Formatter で"
                "自動修正し、AIレビューの範囲から除外してください。"
            )

        return suggestions
```

---

## 7. 高度なレビュー手法

### 7.1 アーキテクチャレベルのレビュー

```python
# 個々のファイルではなくアーキテクチャレベルでレビュー

ARCHITECTURE_REVIEW_PROMPT = """
以下のPRの変更をアーキテクチャの観点でレビューしてください。

## 変更されたファイル
{changed_files}

## レビュー観点
1. レイヤー間の依存関係は正しいか
   - ドメイン層が外部に依存していないか
   - プレゼンテーション層がインフラ層に直接依存していないか

2. 境界の整合性
   - マイクロサービス間のAPI契約は維持されているか
   - 共有データベースへの新しい依存が追加されていないか

3. 設計パターンの一貫性
   - 既存のパターン（Repository、Service、Factory等）に従っているか
   - 新しいパターンを導入する場合、その理由は妥当か

4. 拡張性とテスタビリティ
   - インターフェースが適切に定義されているか
   - 依存性注入が使われているか
   - モックしやすい設計になっているか

## 差分
{diff}

## 出力形式
アーキテクチャ上の問題を深刻度順にリストアップしてください。
各問題に対して、具体的な改善案を示してください。
"""

class ArchitectureReviewer:
    """アーキテクチャレベルのコードレビュー"""

    def __init__(self, architecture_rules: dict):
        self.rules = architecture_rules

    def check_layer_violations(self, changed_files: list[str],
                                imports: dict[str, list[str]]) -> list[dict]:
        """レイヤー間の依存関係違反を検出"""
        violations = []
        layer_order = self.rules.get("layer_order", [
            "domain", "usecase", "interface", "infrastructure"
        ])

        for file_path, file_imports in imports.items():
            file_layer = self._detect_layer(file_path)
            if not file_layer:
                continue

            file_layer_idx = layer_order.index(file_layer) if file_layer in layer_order else -1

            for imp in file_imports:
                imp_layer = self._detect_layer(imp)
                if not imp_layer:
                    continue

                imp_layer_idx = layer_order.index(imp_layer) if imp_layer in layer_order else -1

                # 内側のレイヤーが外側に依存している
                if file_layer_idx < imp_layer_idx:
                    violations.append({
                        "type": "layer_violation",
                        "severity": "major",
                        "file": file_path,
                        "import": imp,
                        "message": (
                            f"{file_layer}層が{imp_layer}層に依存しています。"
                            f"依存性逆転の原則（DIP）を適用してください。"
                        ),
                    })

        return violations

    def _detect_layer(self, path: str) -> str:
        """ファイルパスからレイヤーを推定"""
        path_lower = path.lower()
        if "domain" in path_lower or "entity" in path_lower:
            return "domain"
        elif "usecase" in path_lower or "service" in path_lower:
            return "usecase"
        elif "controller" in path_lower or "handler" in path_lower:
            return "interface"
        elif "repository" in path_lower or "adapter" in path_lower:
            return "infrastructure"
        return ""
```

### 7.2 パフォーマンスレビューの自動化

```python
# パフォーマンス観点の自動レビュー

class PerformanceReviewer:
    """パフォーマンス問題を自動検出するレビューエンジン"""

    PERFORMANCE_PATTERNS = {
        "n_plus_1": {
            "description": "N+1クエリの可能性",
            "patterns": [
                # SQLAlchemy
                r"for\s+\w+\s+in\s+\w+\.query\.",
                r"for\s+\w+\s+in\s+\w+:\s*\n\s+\w+\.\w+\.",
            ],
            "severity": "major",
            "fix": "joinedload() / selectinload() でイーガーロードに変更",
        },
        "unnecessary_serialization": {
            "description": "不要なシリアライゼーション",
            "patterns": [
                r"json\.dumps\(.*json\.loads\(",
                r"\.to_json\(\).*\.from_json\(",
            ],
            "severity": "minor",
            "fix": "オブジェクトを直接渡し、不要な変換を排除",
        },
        "unbounded_query": {
            "description": "LIMITなしのクエリ",
            "patterns": [
                r"\.all\(\)\s*$",
                r"SELECT\s+\*\s+FROM\s+\w+\s*(?!.*LIMIT)",
            ],
            "severity": "major",
            "fix": "ページネーション（LIMIT/OFFSET）を追加",
        },
        "sync_io_in_async": {
            "description": "asyncコンテキストでの同期I/O",
            "patterns": [
                r"async\s+def\s+\w+.*:\s*\n(?:.*\n)*?.*\bopen\(",
                r"async\s+def\s+\w+.*:\s*\n(?:.*\n)*?.*requests\.\w+\(",
            ],
            "severity": "major",
            "fix": "aiofiles / httpx を使用して非同期I/Oに変更",
        },
    }

    def review(self, file_content: str, filename: str) -> list[dict]:
        """パフォーマンス問題を検出"""
        import re
        findings = []

        for pattern_name, config in self.PERFORMANCE_PATTERNS.items():
            for pattern in config["patterns"]:
                matches = list(re.finditer(pattern, file_content, re.MULTILINE))
                for match in matches:
                    line_num = file_content[:match.start()].count("\n") + 1
                    findings.append({
                        "type": pattern_name,
                        "severity": config["severity"],
                        "file": filename,
                        "line": line_num,
                        "description": config["description"],
                        "fix": config["fix"],
                        "code": match.group(0)[:100],
                    })

        return findings
```

---

## アンチパターン

### アンチパターン 1: AIレビューの形骸化

```
❌ BAD: AIレビューコメントを全て無視する
   - "AIのコメントは的外れ" と思い込む
   - CriticalレベルのコメントもDismissする
   - AIレビューがCIに組み込まれているが誰も見ない

✅ GOOD: AIレビューの品質を継続改善
   - 的外れなコメントのパターンを収集
   - .coderabbit.yaml / プロンプトを改善
   - 月次でAIレビューの精度をチーム内で振り返り
   - 有用なコメントの割合を測定（目標: 70%以上）
```

### アンチパターン 2: 人間レビューの省略

```
❌ BAD: "AIがOK出したから人間レビュー不要"
   - AIは文脈的な判断が苦手
   - ビジネスロジックの正しさは人間でないと判断できない
   - 責任の所在が曖昧になる

✅ GOOD: AIと人間で役割分担
   - AI: 機械的チェック（80%の問題を検出）
   - 人間: 判断的チェック（残り20%の重要な問題）
   - 人間レビューの時間は短縮するが、省略はしない
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

## FAQ

### Q1: AIレビューで誤検出（False Positive）が多い場合の対処法は？

3つのアプローチがある。(1) ルールファイル(.coderabbit.yaml等)でプロジェクト固有のルールを定義し、誤検出パターンを除外する。(2) path_instructions でディレクトリごとのレビュー指針を設定する。(3) 誤検出をログに記録し、定期的にプロンプトを改善するフィードバックループを回す。

### Q2: 大規模なPR（1000行以上）をAIでレビューする方法は？

大きなPRはまず「分割すべき」と提案するのがベストプラクティス。それが難しい場合は、(1) ファイル単位で個別にレビュー、(2) 変更の種類（リファクタリング、新機能、バグ修正）ごとにグルーピング、(3) 要約を先に生成してからレビューの優先順位を決定する。

### Q3: AIレビューをチームに導入する際の抵抗をどう乗り越えるか？

段階的導入が鍵。(1) まず1つのリポジトリでパイロット導入し、効果を数値で示す（レビュー時間の短縮、検出したバグ数等）。(2) レビュアーの負担軽減という「味方」のポジションで提案。(3) AIレビューは「最終判断」ではなく「ドラフトレビュー」であることを明確にし、人間の権限を脅かさないことを示す。

### Q4: AIレビューの設定をプロジェクトに最適化する方法は？

3段階のアプローチが効果的。(1) **初期設定（1週目）**: デフォルト設定で開始し、全てのコメントに対して「有用」「不要」のフィードバックを記録する。(2) **チューニング（2-4週目）**: フィードバックに基づいてpath_instructionsを調整し、誤検出が多いパターンを除外する。プロジェクト固有の規約（命名規則、アーキテクチャルール等）をカスタムルールとして追加する。(3) **最適化（5週目以降）**: 月次でAI精度を計測し、新しいルールの追加や不要ルールの削除を行う。チームの合意に基づいてseverityレベルを調整する。

### Q5: レビューの自動化と人間レビューのバランスをどう取るか？

基本方針は「AIが80%の機械的チェックを担当し、人間が20%の判断的チェックに集中する」こと。具体的には、(1) AI担当: コーディング規約、セキュリティパターン、パフォーマンスアンチパターン、テストカバレッジ、未使用コード、型安全性。(2) 人間担当: ビジネスロジックの正しさ、アーキテクチャの妥当性、ユーザー体験への影響、チーム内の暗黙知との整合性、新しい設計パターンの導入判断。AIレビューが完了した状態で人間レビューを開始することで、人間は高レベルの判断に集中できる。

### Q6: マイクロサービス環境でのAIレビューの注意点は？

マイクロサービスでは (1) サービス間のAPI契約変更を検出するルールを設定する（OpenAPI仕様の差分チェック）。(2) 共有ライブラリの変更が他サービスに影響しないかをAIに確認させる。(3) データベーススキーマの変更がマイグレーションを含んでいるかチェックする。(4) 分散トランザクションやイベント駆動の整合性に関する問題は、AIの検出精度が低いため人間が重点的にレビューする。

---

## まとめ

| 項目 | 要点 |
|------|------|
| AIレビューの範囲 | 規約・セキュリティ・性能は高精度、ビジネスロジックは人間 |
| 主要ツール | CodeRabbit、Claude Code、Graphite、Amazon CodeGuru |
| 運用モデル | AI自動→修正→人間レビュー(残り20%)のハイブリッド |
| 効果 | レビュー時間60-70%短縮、検出率85-90% |
| 導入方法 | パイロット→効果測定→段階展開 |
| 注意点 | AI形骸化の防止、人間レビューの維持 |

---

## 次に読むべきガイド

- [02-ai-documentation.md](./02-ai-documentation.md) ── AIドキュメント生成
- [00-ai-testing.md](./00-ai-testing.md) ── AIテストとの統合
- [../03-team/00-ai-team-practices.md](../03-team/00-ai-team-practices.md) ── チームレビュー文化の構築

---

## 参考文献

1. CodeRabbit, "AI Code Review Documentation," 2025. https://docs.coderabbit.ai/
2. Google, "Code Review Developer Guide," 2024. https://google.github.io/eng-practices/review/
3. Microsoft, "How AI is transforming code review at Microsoft," 2024. https://devblogs.microsoft.com/
