# コードレビューチェックリスト

> コードレビューは品質保証・知識共有・チーム学習の3つの役割を担う。主観に頼らず体系的なチェックリストに基づいて効率的かつ建設的なレビューを行うための観点・プロセス・コミュニケーション手法を解説する

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| クリーンコードの基本原則 | 命名規則・関数設計・コメントの書き方 | [00-naming-conventions.md](../00-principles/00-naming-conventions.md) |
| SOLID原則 | 単一責任原則・依存性逆転 | [04-solid-principles.md](../00-principles/04-solid-principles.md) |
| テスト原則 | テストピラミッド・テストカバレッジ | [04-testing-principles.md](../01-practices/04-testing-principles.md) |
| リファクタリング | コードの臭い・技術的負債 | [03-technical-debt.md](../02-refactoring/03-technical-debt.md) |
| API設計 | REST API設計原則 | [03-api-design.md](./03-api-design.md) |

---

## この章で学ぶこと

1. **レビューの観点体系** — 正確性、可読性、保守性、セキュリティ、パフォーマンスの5軸チェックを網羅的に実施できる
2. **効率的なレビュープロセス** — PR サイズ制限、レスポンスタイム SLA、自動化との組み合わせで、レビューの効率と品質を両立できる
3. **建設的なフィードバック** — コメントの種類分類、提案型レビュー、心理的安全性の確保で、チームの成長を促進できる
4. **レビューの自動化戦略** — CI/CD パイプラインとの統合、静的解析ツールの活用で、人間の判断が必要な箇所に集中できる
5. **組織レベルのレビュー文化** — CODEOWNERS、レビューメトリクス、知識共有の仕組みを構築し、持続可能なレビュー体制を確立できる

---

## 1. レビュー観点の5軸

### 1.1 チェックリスト全体構成

```
コードレビュー 5軸チェック

  +-----------+
  | 正確性     |  ← ロジックは正しいか？エッジケースは？
  +-----------+
       |
  +-----------+
  | 可読性     |  ← 6ヶ月後の自分が理解できるか？
  +-----------+
       |
  +-----------+
  | 保守性     |  ← 変更が容易か？テストはあるか？
  +-----------+
       |
  +-----------+
  | セキュリティ|  ← 入力検証は？認証・認可は？
  +-----------+
       |
  +-----------+
  | パフォーマンス| ← N+1問題は？メモリリークは？
  +-----------+
```

```
レビューの優先度マトリクス:

  高優先度 (マージブロッカー):
  ├── バグ・ロジックエラー
  ├── セキュリティ脆弱性
  ├── データ損失の可能性
  └── 本番環境への影響

  中優先度 (修正推奨):
  ├── 設計・アーキテクチャの問題
  ├── テスト不足
  ├── パフォーマンス問題
  └── エラーハンドリング不足

  低優先度 (改善提案):
  ├── 命名の改善
  ├── コードスタイル
  ├── ドキュメント不足
  └── リファクタリングの余地
```

### 1.2 各軸の詳細チェックリスト

```python
# ===== 正確性チェック =====
correctness_checklist = [
    "ビジネスロジックが要件と一致しているか",
    "エッジケース（null、空配列、0、負数、最大値）が処理されているか",
    "エラーハンドリングが適切か（例外の種類、リカバリー）",
    "並行処理の問題はないか（レースコンディション、デッドロック）",
    "トランザクション境界は正しいか",
    "既存テストが通るか（回帰がないか）",
    "整数オーバーフロー、浮動小数点の丸め誤差は考慮されているか",
    "タイムゾーン・日付境界の処理は正しいか",
]

# ===== 可読性チェック =====
readability_checklist = [
    "変数名・関数名が意図を明確に伝えているか",
    "関数の長さは適切か（20行以内が目安）",
    "ネストが深すぎないか（3段以内が目安）",
    "コメントが「なぜ」を説明しているか（「何」はコードが語るべき）",
    "一貫した命名規則に従っているか",
    "不要なコメントやデッドコードがないか",
    "認知負荷が高くないか（1つの関数で複数の抽象レベルを混在させていないか）",
    "早期リターンパターンが使えるのに深いネストになっていないか",
]

# ===== 保守性チェック =====
maintainability_checklist = [
    "単一責任原則に従っているか（1クラス=1責務）",
    "DRY原則：重複コードはないか",
    "テストが追加されているか（新機能・バグ修正）",
    "既存のアーキテクチャパターンに従っているか",
    "依存関係の方向は正しいか（レイヤー違反がないか）",
    "マジックナンバーが定数化されているか",
    "将来の変更が予想される箇所に適切な抽象化があるか",
    "設定値がハードコードされていないか（環境変数・設定ファイル）",
]

# ===== セキュリティチェック =====
security_checklist = [
    "入力のバリデーション・サニタイズは適切か",
    "SQLインジェクション対策（パラメータバインド）",
    "XSS対策（出力エスケープ）",
    "認証・認可チェックが漏れていないか",
    "機密情報がログに出力されていないか",
    "秘密鍵・トークンがコードにハードコードされていないか",
    "CSRF対策は適切か",
    "ファイルアップロードのサイズ制限・タイプチェックは適切か",
    "レート制限は適切に設定されているか",
]

# ===== パフォーマンスチェック =====
performance_checklist = [
    "N+1 クエリ問題はないか",
    "不要なデータの取得（SELECT *）はないか",
    "ループ内でのDB/API呼び出しはないか",
    "適切なインデックスが設定されているか",
    "大量データ処理でメモリを圧迫しないか",
    "キャッシュすべきデータをキャッシュしているか",
    "不要な再レンダリング（React）がないか",
    "非同期処理にすべき重い処理が同期で実行されていないか",
]
```

### 1.3 レビュー観点のチートシート

```
┌───────────────────────────────────────────────────────┐
│          レビュー観点クイックリファレンス                    │
├───────────────────────────────────────────────────────┤
│                                                       │
│  変更が影響する箇所を特定:                               │
│  ├── この関数を呼んでいる箇所は？ (影響範囲)              │
│  ├── この変更が壊す可能性のあるテストは？                 │
│  └── 関連するドキュメントの更新は必要か？                 │
│                                                       │
│  「自分ならどう書くか」ではなく確認すること:              │
│  ├── 要件を満たしているか？                             │
│  ├── エッジケースは処理されているか？                    │
│  ├── テストは十分か？                                   │
│  └── セキュリティリスクはないか？                        │
│                                                       │
│  見落としやすいポイント:                                 │
│  ├── 削除されたコードの影響                              │
│  ├── 設定ファイルの変更                                 │
│  ├── DB マイグレーション（ロールバック可能か？）          │
│  └── 環境変数・シークレットの追加                        │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 2. レビューフローとルール

### 2.1 プロセス

```
  PR 作成
    |
    v
  [自動チェック] ← CI: lint, test, coverage, security scan
    |
    | 全パス
    v
  [セルフレビュー] ← 作者自身がまず確認
    |
    v
  [レビュー依頼] ← 1-2名のレビュアーをアサイン
    |
    +---> レビュアー確認 (目標: 24時間以内)
    |
    v
  [フィードバック]
    |
    +---> Approve → マージ
    |
    +---> Request Changes → 修正 → 再レビュー
    |
    +---> Comment → 議論 → 合意形成
```

### 2.2 セルフレビューの体系的手法

```python
# セルフレビューチェックリスト
# PR 作成後、レビュー依頼前に作者自身が確認する

self_review_checklist = {
    "デバッグコード": [
        "print / console.log / debugger が残っていないか",
        "TODO / FIXME / HACK コメントが意図的か確認",
        "テスト用のハードコード値が残っていないか",
    ],
    "差分の確認": [
        "不要な変更（フォーマットのみの差分）が混ざっていないか",
        "意図しないファイルが含まれていないか (.env, node_modules)",
        "コミットメッセージは変更内容を正確に反映しているか",
    ],
    "テスト": [
        "新機能にテストを追加したか",
        "バグ修正に回帰テストを追加したか",
        "テストが他のテストに依存していないか（独立性）",
    ],
    "ドキュメント": [
        "公開 API の変更にドキュメント更新は必要か",
        "README や CHANGELOG の更新は必要か",
        "コメントが最新の実装と一致しているか",
    ],
}

# 研究結果: セルフレビューで指摘事項の30-40%は事前に除去できる
```

### 2.3 PR サイズガイドライン

```
PR サイズと品質の関係

  変更行数    レビュー品質    推奨度    レビュー時間目安
  ────────────────────────────────────────────────────
  < 50行      非常に高い      最適      15分以内
  50-200行    高い           推奨      30分以内
  200-400行   中程度         許容      60分以内
  400-800行   低い           分割推奨  60分以上
  > 800行     非常に低い      分割必須  分割を依頼

  研究結果 (SmartBear, Cisco):
  - 200行以下のレビューで欠陥発見率が最大
  - 400行を超えると「LGTM」と流す傾向が強まる
  - 60分以上のレビューで集中力が低下
  - 1回のレビューは最大60分、休憩を挟んで再開
```

```
大きな PR の分割戦略:

  1. レイヤー別分割
     ├── PR 1: DB マイグレーション + モデル
     ├── PR 2: ビジネスロジック + サービス層
     └── PR 3: API エンドポイント + テスト

  2. 機能別分割（Feature Flag 活用）
     ├── PR 1: ユーザー登録 API
     ├── PR 2: メール認証機能
     └── PR 3: 管理画面 UI

  3. リファクタリング + 機能追加の分離
     ├── PR 1: 既存コードのリファクタリング（機能変更なし）
     └── PR 2: 新機能の追加

  原則: 各 PR が独立してマージ可能な単位にする
```

### 2.4 レスポンスタイム SLA

```python
# レビューの効率的な時間配分
review_time_guide = {
    "small_pr":   {"lines": "< 100",   "time": "15分以内"},
    "medium_pr":  {"lines": "100-300",  "time": "30分以内"},
    "large_pr":   {"lines": "300-500",  "time": "60分以内"},
    "too_large":  {"lines": "> 500",    "time": "分割を依頼"},
}

# レスポンスタイム SLA
response_sla = {
    "initial_review":  "24時間以内",    # 最初のレビュー
    "re_review":       "8時間以内",     # 修正後の再レビュー
    "urgent_hotfix":   "2時間以内",     # 緊急修正
    "documentation":   "48時間以内",    # ドキュメントのみの変更
}

# レビュー効率を上げるための環境設定
review_setup = {
    "通知設定": "Slack/Teams に PR 通知を設定",
    "時間ブロック": "毎日30分のレビュー専用時間を確保",
    "バッチ処理": "小さな PR は2-3件まとめてレビュー",
    "コンテキスト切替最小化": "深い作業の合間ではなく、切りの良いタイミングで",
}
```

---

## 3. コメントの分類と書き方

### 3.1 コメントプレフィックス

```
[MUST]     必ず修正が必要（マージブロッカー）
[SHOULD]   できれば修正してほしい
[NIT]      些細な指摘（修正任意）
[QUESTION] 質問・確認事項
[PRAISE]   良いコードへの称賛
[FYI]      参考情報の共有
[DISCUSS]  議論が必要な設計判断

使用例:
  [MUST] SQL インジェクションの脆弱性があります。
         パラメータバインドを使用してください。

  [SHOULD] この関数が40行あるので、バリデーション部分を
           別メソッドに抽出すると可読性が向上します。

  [NIT] 変数名 `d` → `delivery_date` の方が意図が明確です。

  [PRAISE] このテストケースの境界値の網羅性が素晴らしいです。

  [QUESTION] このタイムアウト値(30秒)の根拠を教えてください。
             外部API のSLAに基づいていますか？

  [FYI] 似た処理が utils/date.ts にあるので、共通化できるかもしれません。

  [DISCUSS] この設計だと将来の拡張が難しそうです。
            Strategy パターンの導入を検討しませんか？
```

### 3.2 提案型コメントの書き方

```python
# BAD: 否定だけのコメント
# 「このコードは読みにくいです」
# 「なんでこう書いたんですか？」

# GOOD: 問題の特定 + 理由 + 具体的な改善案

# ===== パターン1: Before/After で提示 =====

# [SHOULD] ネストが深くなっています。
# 早期リターンパターンに変更すると可読性が向上します：
#
# Before:
def process(order):
    if order:
        if order.is_valid():
            if order.items:
                # 処理...
                pass

# Suggested:
def process(order):
    if not order:
        return
    if not order.is_valid():
        raise ValueError("Invalid order")
    if not order.items:
        raise ValueError("Empty items")
    # 処理...

# ===== パターン2: 理由を添えて提案 =====

# [SHOULD] このループ内で DB アクセスが発生しており、
# N+1 問題になっています。
# items が100件ある場合、100回のクエリが発行されます。
#
# 改善案: バッチクエリに変更
# Before:
for item in items:
    product = db.get_product(item.product_id)  # N回クエリ

# Suggested:
product_ids = [item.product_id for item in items]
products = db.get_products_by_ids(product_ids)  # 1回のクエリ
product_map = {p.id: p for p in products}

# ===== パターン3: トレードオフを示して議論を促す =====

# [DISCUSS] ここでキャッシュを導入すべきか検討が必要です。
# メリット: レスポンスが約10倍高速化（DB round-trip 削減）
# デメリット: キャッシュ無効化の複雑さが増加
# 現在のトラフィック量を考えると、今は不要かもしれません。
# どう思いますか？
```

### 3.3 GitHub Suggestion 機能の活用

```python
# GitHub の Suggestion 構文で直接修正を提案

# レビューコメントで以下のように書く:
#
# [NIT] 定数名は UPPER_SNAKE_CASE にしましょう。
#
# ```suggestion
# MAX_RETRY_COUNT = 3
# TIMEOUT_SECONDS = 30
# ```
#
# → 作者はワンクリックで適用可能

# 複数行の提案も可能:
#
# [SHOULD] 型ヒントを追加しましょう。
#
# ```suggestion
# def calculate_total(
#     items: list[OrderItem],
#     tax_rate: float = 0.10,
# ) -> int:
# ```

# バッチ提案: 複数の suggestion をまとめて適用可能
# → 小さな修正を1コミットにまとめられる
```

### 3.4 褒める文化の実践

```
良いレビューは「指摘」だけでなく「称賛」を含む。

  称賛すべきポイント:
  ├── 読みやすい命名
  ├── 巧みなテストケース設計
  ├── エッジケースの適切な処理
  ├── 既存コードの改善（ボーイスカウトルール）
  ├── 良いドキュメント
  └── パフォーマンスへの配慮

  称賛の例:
  [PRAISE] このエラーハンドリングのパターン、とても参考になります。
           他の箇所にも適用したいです。

  [PRAISE] テストケースの境界値テストが網羅的で素晴らしい。
           特に0件の場合と上限値の場合を両方カバーしている点が良い。

  [PRAISE] この関数の分割方法が的確です。
           各関数が単一責任で、テストも書きやすくなっています。

  研究結果: 称賛を含むレビューは、指摘のみのレビューと比較して
  修正の取り込み率が23%高い（Google Engineering Practices）
```

---

## 4. 自動化との組み合わせ

### 4.1 CI/CD パイプラインとの統合

```
レビューの役割分担

  自動化 (CI) が担当:
  ├── コードスタイル (Ruff, ESLint, Prettier)
  ├── 型チェック (MyPy, TypeScript)
  ├── テスト実行
  ├── カバレッジ計測
  ├── セキュリティスキャン (Bandit, Snyk, Trivy)
  ├── 依存関係の脆弱性チェック
  ├── ライセンス互換性チェック
  └── コード複雑度チェック (cyclomatic complexity)

  人間が担当:
  ├── ビジネスロジックの正確性
  ├── 設計・アーキテクチャの妥当性
  ├── 可読性と命名の適切さ
  ├── テストケースの十分性（カバレッジだけでなく意味的な網羅性）
  ├── コンテキスト依存の判断
  └── 将来の拡張性・保守性の評価

  ★ 自動化できることは自動化し、人間は高次の判断に集中
```

### 4.2 CI 設定の実践例

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on:
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint Check
        run: |
          ruff check .        # Python
          ruff format --check . # フォーマットチェック

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Type Check
        run: mypy src/ --strict

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Tests
        run: pytest --cov=src --cov-report=xml --cov-fail-under=80

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Security Scan
        run: |
          bandit -r src/       # Python セキュリティ
          pip-audit            # 依存関係の脆弱性

  pr-size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Check PR Size
        run: |
          CHANGED_LINES=$(git diff --stat origin/main...HEAD | tail -1 | awk '{print $4}')
          if [ "$CHANGED_LINES" -gt 400 ]; then
            echo "::warning::PR が400行を超えています。分割を検討してください。"
          fi
```

### 4.3 CODEOWNERS の設計

```
# .github/CODEOWNERS

# デフォルトレビュアー
* @team-leads

# フロントエンド
/frontend/             @frontend-team
/frontend/src/auth/    @security-team @frontend-team

# バックエンド
/backend/              @backend-team
/backend/src/billing/  @billing-team @backend-team

# インフラ
/infrastructure/       @sre-team
/docker/              @sre-team
/.github/             @devops-team

# DB マイグレーション（必ず DBA がレビュー）
/backend/migrations/   @dba-team

# セキュリティ関連（セキュリティチームの承認必須）
**/auth/**            @security-team
**/crypto/**          @security-team
```

```
CODEOWNERS 設計のベストプラクティス:

  ├── 過度に細かく設定しない（レビュー待ちのボトルネック）
  ├── チーム単位でアサイン（個人単位だと休暇時に滞留）
  ├── セキュリティ・DB マイグレーションは専門チーム必須
  ├── 定期的に見直し（チーム体制の変化に追従）
  └── Optional reviewers も活用（知識共有目的のレビュー）
```

---

## 5. レビューメトリクスと改善

### 5.1 追跡すべきメトリクス

```
レビュープロセスの健全性指標:

  速度メトリクス:
  ├── Time to First Review: PR 作成からレビュー開始まで
  │   目標: < 24時間、理想: < 4時間
  ├── Review Cycle Time: PR 作成からマージまで
  │   目標: < 48時間
  └── Re-review Time: 修正後の再レビューまで
      目標: < 8時間

  品質メトリクス:
  ├── Defect Escape Rate: レビューを通過したバグの割合
  │   目標: < 5%
  ├── Review Coverage: レビューを受けたPRの割合
  │   目標: 100%（hotfix除く）
  └── Comments per PR: PR あたりのコメント数
      目安: 2-5件（0件は形式的、10+件はPRが大きすぎる）

  チームメトリクス:
  ├── Review Load Balance: レビュアーごとの負荷分散
  ├── Knowledge Distribution: CODEOWNERS の偏り
  └── PR Size Distribution: PRサイズの分布
```

### 5.2 メトリクスの可視化

```python
# GitHub API を使ったレビューメトリクスの収集

import requests
from datetime import datetime, timedelta
from collections import defaultdict

def collect_review_metrics(repo: str, token: str, days: int = 30):
    """過去N日間のレビューメトリクスを収集"""
    headers = {"Authorization": f"Bearer {token}"}
    since = (datetime.now() - timedelta(days=days)).isoformat()

    # PRの取得
    prs = requests.get(
        f"https://api.github.com/repos/{repo}/pulls",
        headers=headers,
        params={"state": "closed", "since": since, "per_page": 100},
    ).json()

    metrics = {
        "total_prs": len(prs),
        "avg_time_to_first_review": [],
        "avg_cycle_time": [],
        "avg_comments_per_pr": [],
        "pr_sizes": [],
        "reviewer_load": defaultdict(int),
    }

    for pr in prs:
        # PR作成日時
        created_at = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
        merged_at = pr.get("merged_at")

        if merged_at:
            merged_at = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            cycle_time = (merged_at - created_at).total_seconds() / 3600
            metrics["avg_cycle_time"].append(cycle_time)

        # レビューの取得
        reviews = requests.get(
            pr["url"] + "/reviews",
            headers=headers,
        ).json()

        if reviews:
            first_review = datetime.fromisoformat(
                reviews[0]["submitted_at"].replace("Z", "+00:00")
            )
            time_to_first = (first_review - created_at).total_seconds() / 3600
            metrics["avg_time_to_first_review"].append(time_to_first)

            for review in reviews:
                reviewer = review["user"]["login"]
                metrics["reviewer_load"][reviewer] += 1

        # コメント数
        comments = requests.get(pr["url"] + "/comments", headers=headers).json()
        metrics["avg_comments_per_pr"].append(len(comments))

        # PR サイズ
        metrics["pr_sizes"].append(pr.get("additions", 0) + pr.get("deletions", 0))

    # 集計
    return {
        "total_prs": metrics["total_prs"],
        "avg_time_to_first_review_hours": (
            sum(metrics["avg_time_to_first_review"])
            / len(metrics["avg_time_to_first_review"])
            if metrics["avg_time_to_first_review"] else 0
        ),
        "avg_cycle_time_hours": (
            sum(metrics["avg_cycle_time"])
            / len(metrics["avg_cycle_time"])
            if metrics["avg_cycle_time"] else 0
        ),
        "avg_comments_per_pr": (
            sum(metrics["avg_comments_per_pr"])
            / len(metrics["avg_comments_per_pr"])
            if metrics["avg_comments_per_pr"] else 0
        ),
        "median_pr_size": sorted(metrics["pr_sizes"])[len(metrics["pr_sizes"]) // 2]
            if metrics["pr_sizes"] else 0,
        "reviewer_load": dict(metrics["reviewer_load"]),
    }
```

---

## 6. 特殊なレビュー対象

### 6.1 DB マイグレーションのレビュー

```
DB マイグレーション専用チェックリスト:

  安全性:
  ├── [x] ロールバック可能か？（down マイグレーション）
  ├── [x] 大テーブルのロック時間は許容範囲か？
  ├── [x] NOT NULL 制約の追加にデフォルト値はあるか？
  ├── [x] インデックスの追加は CONCURRENTLY か？（PostgreSQL）
  └── [x] データ移行のバッチサイズは適切か？

  互換性:
  ├── [x] 旧バージョンのコードと互換性があるか？（ローリングデプロイ）
  ├── [x] カラム削除は2段階（まず読み取り停止 → 次回削除）か？
  └── [x] 外部キー制約の追加はアプリケーション停止を伴わないか？

  テスト:
  ├── [x] ステージング環境で実行済みか？
  ├── [x] 本番と同等のデータ量でテスト済みか？
  └── [x] 実行時間を計測したか？
```

### 6.2 セキュリティクリティカルなコードのレビュー

```
セキュリティレビューの重点チェック:

  認証・認可:
  ├── トークン検証ロジックは正しいか？
  ├── 権限チェックの漏れはないか？（Broken Access Control）
  ├── セッション管理は安全か？
  └── パスワードハッシュアルゴリズムは適切か？（bcrypt, Argon2）

  データ保護:
  ├── PII（個人情報）のマスキングは適切か？
  ├── ログに機密情報が含まれていないか？
  ├── 暗号化キーのローテーション対応はあるか？
  └── データの最小権限の原則に従っているか？

  入力検証:
  ├── 全ての外部入力にバリデーションがあるか？
  ├── ファイルパスのトラバーサル攻撃対策は？
  ├── XML External Entity (XXE) 対策は？
  └── Server-Side Request Forgery (SSRF) 対策は？
```

### 6.3 パフォーマンスクリティカルなコードのレビュー

```
パフォーマンスレビューの重点チェック:

  データベース:
  ├── EXPLAIN ANALYZE でクエリプランを確認したか？
  ├── フルテーブルスキャンになっていないか？
  ├── 不要な JOIN はないか？
  └── バッチ処理のチャンクサイズは適切か？

  メモリ:
  ├── 大量データの一括ロードをしていないか？
  ├── ストリーム処理が可能な箇所をバッチで処理していないか？
  ├── クロージャによるメモリリークはないか？
  └── イベントリスナーの解除忘れはないか？

  ネットワーク:
  ├── 不要な API コールはないか？
  ├── レスポンスの gzip 圧縮は有効か？
  ├── CDN を活用すべき静的リソースはあるか？
  └── WebSocket vs ポーリングの選択は適切か？
```

---

## 7. レビュー手法の比較

| レビュー手法 | 対象 | コスト | 欠陥発見率 | 知識共有効果 |
|------------|------|-------|:--------:|:----------:|
| PR レビュー (非同期) | コード差分 | 低 | 中 | 中 |
| ペアプログラミング | リアルタイム | 高 | 高 | 高 |
| モブプログラミング | チーム全体 | 最高 | 最高 | 最高 |
| 自動レビュー (CI) | 静的解析 | 最低 | 低 (パターン限定) | なし |
| アーキテクチャレビュー | 設計文書 | 中 | 高 (設計レベル) | 高 |

| 観点 | 自動化可能 | 人間が必要 |
|------|:--------:|:--------:|
| コードスタイル | 完全自動化 | -- |
| 型安全性 | 完全自動化 | -- |
| テスト通過 | 完全自動化 | -- |
| ビジネスロジック | -- | 必須 |
| 設計判断 | -- | 必須 |
| 命名の適切さ | 部分自動化 | 必須 |
| テストの十分性 | 部分自動化 | 必須 |
| セキュリティ | 部分自動化 | 必須 |

```
レビュー手法の使い分け:

  コードの複雑度
    │
    ├── 低（バグ修正、小さな機能追加）
    │   → 非同期 PR レビュー（1名）
    │
    ├── 中（中規模機能、APIの追加）
    │   → 非同期 PR レビュー（2名）+ CI 自動チェック
    │
    ├── 高（アーキテクチャ変更、新サービス）
    │   → アーキテクチャレビュー + ペアプログラミング + PR レビュー
    │
    └── 最高（セキュリティ、金融ロジック）
        → 専門チームによるレビュー + モブプログラミング
```

---

## 8. アンチパターン

### 8.1 アンチパターン：人格攻撃になるレビュー

```
BAD:
  「なんでこんな書き方するんですか？普通はこう書きます」
  「このコードは素人レベルです」
  「前にも言ったのに、なぜ直さないんですか？」
  → 心理的安全性の崩壊、レビュー文化の衰退

GOOD:
  「[SHOULD] この部分、早期リターンパターンを使うと
   ネストが減って可読性が上がります。以下のようにいかがでしょう？」
  「[PRAISE] このエラーハンドリングの設計は参考になります」
  「[FYI] この命名パターン、チームのコーディング規約にもある
   ベストプラクティスです: [リンク]」
  → コードに対するフィードバック、人に対する敬意
```

**根本原因**: レビューの目的が「問題の発見」ではなく「批判」になっている。レビュアーは「コードをより良くするために協力する」というマインドセットが必要。

**対策**: (1) チームでレビューガイドラインを策定。(2) コメントプレフィックスの義務化。(3) レビュー研修の実施。(4) 1:1 でのフィードバック方法の見直し。

### 8.2 アンチパターン：LGTM スタンプ

```
BAD:
  「LGTM」(1分でレビュー完了、400行のPR)
  → レビューの意味がない、品質保証にならない

GOOD:
  - 最低1つは具体的なコメントを残す
  - 良い点も指摘する ([PRAISE])
  - PR が大きすぎる場合は分割を依頼する
  - 理解できない箇所は [QUESTION] で質問する
  - 変更の要約を自分の言葉で書く（理解の確認）
```

**根本原因**: レビューの時間が確保されていない、レビューの価値が組織で認識されていない。

**対策**: (1) レビュー時間を業務時間として正式に確保。(2) レビューメトリクスの可視化。(3) レビューの最低基準を CODEOWNERS の approve 条件に設定。

### 8.3 アンチパターン：ゲートキーパー型レビュー

```
BAD:
  特定の個人がボトルネックになるレビュー体制
  ├── シニアエンジニア1名が全PRをレビュー
  ├── 承認まで3-5日待ち
  └── チームの自律性が育たない

GOOD:
  分散型レビュー体制
  ├── レビュアーはチーム内でローテーション
  ├── CODEOWNERS は「チーム」単位で設定
  ├── ジュニアメンバーも積極的にレビュー参加（学習機会）
  └── レビューガイドラインを明文化し、属人化を防ぐ
```

**根本原因**: 「シニアでないとレビューできない」という思い込み。

**対策**: (1) ジュニアのレビューをシニアが「レビューのレビュー」して育成。(2) ドメイン知識はペアレビューで共有。(3) 承認条件を「2名中1名がシニア」に緩和。

### 8.4 アンチパターン：スタイル論争（Bike-shedding）

```
BAD:
  PR コメントの80%が以下のような議論:
  ├── タブ vs スペース
  ├── セミコロンの有無
  ├── 括弧の位置
  └── インポートの順序

  → 本質的な問題（バグ、設計、セキュリティ）が見落とされる

GOOD:
  ├── コードスタイルは Linter/Formatter で自動強制
  │   (Prettier, Black, Ruff, gofmt)
  ├── スタイルガイドをドキュメント化（議論は初回のみ）
  ├── CI が自動でフォーマットチェック
  └── 人間はロジック・設計・セキュリティに集中
```

**根本原因**: 自動化可能な項目を人間がチェックしている。

**対策**: `.editorconfig`, `prettier`, `ruff` 等を CI に組み込み、フォーマット違反は自動で検出・修正。

---

## 9. 演習問題

### 演習1（基礎）: コードレビューの実践

**課題**: 以下の Python コードをレビューし、適切なプレフィックス付きのコメントを5つ以上作成せよ。

```python
# レビュー対象コード
import json
import os

def get_users(db, role, page):
    query = f"SELECT * FROM users WHERE role = '{role}' ORDER BY id LIMIT 20 OFFSET {page * 20}"
    users = db.execute(query)
    result = []
    for u in users:
        if u.active == True:
            data = {}
            data['id'] = u.id
            data['name'] = u.first_name + ' ' + u.last_name
            data['email'] = u.email
            data['role'] = u.role
            data['created'] = str(u.created_at)
            data['password_hash'] = u.password_hash  # フロントエンドが必要としている
            result.append(data)
    return json.dumps(result)
```

**期待される出力**:

```
5つ以上のレビューコメント（プレフィックス付き）
```

**模範解答**:

```
[MUST] SQL インジェクションの脆弱性があります。
  role パラメータがエスケープされずにクエリに直接埋め込まれています。
  パラメータバインドを使用してください:
  query = "SELECT * FROM users WHERE role = %s ORDER BY id LIMIT %s OFFSET %s"
  db.execute(query, (role, 20, page * 20))

[MUST] password_hash がレスポンスに含まれています。
  これはセキュリティ上の重大な問題です。
  フロントエンドにパスワードハッシュを返す必要はありません。
  必要なフィールドのみを含む DTO に変換してください。

[SHOULD] 関数が JSON 文字列を返しています。
  通常、関数はデータ構造（list[dict]）を返し、
  シリアライゼーションは呼び出し元（API層）に任せるべきです。
  これにより、テストが容易になり、再利用性が上がります。

[SHOULD] `u.active == True` は `u.active` と書けます。
  また、フィルタリングは SQL 側で行う方が効率的です:
  WHERE role = %s AND active = TRUE

[SHOULD] ページネーションの定数 20 がハードコードされています。
  PER_PAGE = 20 として定数化するか、引数にしてください。

[NIT] 変数名 `u` は `user` の方が可読性が高いです。
  for user in users:

[NIT] 日時のフォーマットに str() を使うと実装依存の形式になります。
  ISO 8601 形式を使用してください:
  data['created_at'] = u.created_at.isoformat()

[QUESTION] このエンドポイントのページネーションですが、
  オフセットベースで問題ないですか？
  ユーザー数が多い場合、カーソルベースの方が効率的です。
```

---

### 演習2（応用）: レビュープロセスの設計

**課題**: 以下のチーム構成でレビュープロセスを設計せよ。

```
チーム構成:
  - テックリード 1名
  - シニアエンジニア 2名
  - ミドルエンジニア 3名
  - ジュニアエンジニア 2名

課題:
  - テックリードがボトルネックになっている（全PRをレビュー）
  - レビュー待ち時間が平均48時間
  - ジュニアがレビューに参加していない
  - コードスタイルの議論が多い
```

**期待される出力**:

```
1. レビュールール（CODEOWNERS、承認条件）
2. 自動化の提案（CI 設定）
3. レビュー文化の改善策
4. メトリクス目標
```

**模範解答**:

```
1. レビュールール:

  CODEOWNERS:
    /src/           @backend-team  (チーム全員)
    /src/billing/   @senior-team   (シニア以上必須)
    /infrastructure/ @tech-lead    (テックリード必須)
    /migrations/    @tech-lead @senior-team

  承認条件:
    通常PR:  2名の承認（うち1名はシニア以上）
    セキュリティ関連: テックリード + シニア
    ドキュメントのみ: 1名の承認
    hotfix: テックリード or シニアの1名承認

  ジュニアの参加:
    ジュニアは全PRに「任意レビュアー」として追加
    承認権限はないが、コメント・質問は推奨
    週1でシニアがジュニアのレビューコメントをフィードバック

2. 自動化:
    Prettier / Ruff を CI に導入 → スタイル論争を排除
    PR サイズチェック（400行超で警告）
    カバレッジ80%未満で警告
    セキュリティスキャン自動実行

3. レビュー文化:
    毎日30分のレビュータイム確保（全員）
    週1の「Good Review」共有会（5分）
    レビューガイドラインの明文化
    [PRAISE] コメントの奨励

4. メトリクス目標:
    Time to First Review: < 8時間（現在48時間）
    Review Cycle Time: < 24時間
    レビュアーの負荷分散: 偏差20%以内
    ジュニアのレビューコメント: 週3件以上
```

---

### 演習3（発展）: AI コードレビューツールの活用戦略

**課題**: AI コードレビューツール（GitHub Copilot、Coderabbit 等）をチームに導入する戦略を設計せよ。

```
条件:
  - チームは10名
  - 月間200 PR
  - レビュー待ち時間を50%短縮したい
  - AI による誤検出を最小限に抑えたい
```

**期待される出力**:

```
1. AI ツールが担当すべき領域
2. 人間が引き続き担当すべき領域
3. 導入のフェーズ計画
4. 品質検証の方法
```

**模範解答**:

```
1. AI ツールが担当する領域:
   ├── コードスタイル・フォーマットの提案
   ├── 一般的なバグパターンの検出
   ├── 未使用変数・未処理例外の検出
   ├── セキュリティパターンの基本チェック
   ├── ドキュメント・コメントの提案
   └── テストカバレッジの提案

2. 人間が担当する領域:
   ├── ビジネスロジックの正確性（AI は要件を知らない）
   ├── アーキテクチャ・設計の妥当性
   ├── ドメイン固有の慣習・ルール
   ├── パフォーマンスの実測に基づく判断
   ├── AI の提案の最終承認
   └── チームメンバーの育成・メンタリング

3. 導入フェーズ:
   Phase 1 (Month 1): AI をコメントのみモードで導入
     → AI の提案精度を計測（適合率・再現率）
   Phase 2 (Month 2): 高精度な項目のみ自動承認
     → スタイル、未使用コード等の明確な問題
   Phase 3 (Month 3): AI + 人間のハイブリッドフロー確立
     → AI が初回レビュー → 人間がロジック・設計レビュー

4. 品質検証:
   ├── AI の提案を人間が1ヶ月間追跡
   ├── False Positive 率を計測（目標: < 10%）
   ├── AI 導入前後のバグ数を比較
   ├── レビュー待ち時間の変化を計測
   └── チームの満足度アンケート
```

---

## 10. FAQ

### Q1. レビュアーは何人が適切か？

**A.** 1-2名が最適。3名以上になると「誰かが見てくれるだろう」効果（社会的手抜き / Diffusion of Responsibility）が発生する。重要な変更やアーキテクチャに関わる変更は2名、通常の変更は1名で十分。CODEOWNERS ファイルで自動アサインを設定し、ドメイン知識を持つ適切なレビュアーに振り分ける。

### Q2. レビューで意見が対立した場合は？

**A.** エスカレーションのルールを事前に決めておく:

1. **客観的根拠**で議論する（パフォーマンスベンチマーク、公式ドキュメント）
2. **3コメント以上往復したらオフライン**（ビデオ通話）で直接話す
3. チームの**コーディング規約に明記**して今後の基準にする
4. 合意できない場合は**テックリードが最終判断**する
5. **個人の好みの問題**は議論せず、チーム規約に任せる（タブ vs スペース等）

### Q3. セルフレビューのポイントは？

**A.** PR 作成後、レビュー依頼前に自分で差分を確認する。チェックポイント: (1) デバッグ用コード（print, console.log）が残っていないか。(2) コミットメッセージは変更内容を正確に反映しているか。(3) 不要な変更（フォーマットのみの差分）が混ざっていないか。(4) テストを追加したか。セルフレビューで指摘事項の30%は事前に除去できる。

### Q4. レビューの時間が足りないとき、何を優先すべきか？

**A.** 時間が限られている場合の優先順位:

1. **セキュリティ**: 認証・認可、入力検証、機密情報の露出
2. **正確性**: ビジネスロジックのバグ、エッジケース
3. **テスト**: 新機能・バグ修正のテストカバレッジ
4. **保守性**: 設計・アーキテクチャの問題
5. **可読性**: 命名、コメント、コードスタイル

時間がない場合、5は CI に任せ、1-3 に集中する。

### Q5. ジュニアメンバーはどのようにレビューに参加すべきか？

**A.** ジュニアのレビュー参加は学習効果が非常に高い。推奨されるステップ:

1. **観察**: まずシニアのレビューコメントを読んで学ぶ
2. **質問**: `[QUESTION]` プレフィックスで疑問点を聞く（恥ずかしがらない）
3. **簡単な指摘**: 命名、コメント、フォーマットの `[NIT]` から始める
4. **テストの確認**: テストケースの網羅性を確認する
5. **徐々にロジックへ**: 理解できる範囲でロジックの正確性を確認

ジュニアのレビューコメントに対して、シニアが「良い観点」「もっとこう見るとよい」とフィードバックすることで、レビュースキルが向上する。

### Q6. レビューで「Approve」するタイミングは？

**A.** 以下の3条件が全て満たされたとき:

1. **MUST が全て解決**: マージブロッカーの指摘が全て修正された
2. **SHOULD の合意**: 修正するか次回に回すか、作者との合意がある
3. **理解**: 変更内容を自分が理解し、説明できる状態である

「完璧でなくてもよい」が原則。改善の余地があっても、現在のコードより良くなっていれば Approve してよい。完璧を求めるとマージが遅延し、チーム全体の生産性が下がる。

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| 5軸チェック | 正確性、可読性、保守性、セキュリティ、パフォーマンス |
| PR サイズ | 200行以下が最適。400行超は分割必須 |
| レスポンスタイム | 初回24時間以内、再レビュー8時間以内 |
| コメント分類 | MUST / SHOULD / NIT / QUESTION / PRAISE / DISCUSS で明確化 |
| 提案型フィードバック | 否定ではなく具体的な改善案を提示。Before/After で示す |
| 自動化との分担 | スタイル・型・テストは CI、ロジック・設計は人間 |
| CODEOWNERS | チーム単位で設定、ボトルネックを防ぐ |
| 心理的安全性 | コードへのフィードバック、人への敬意。称賛を含める |
| メトリクス | Time to First Review、Cycle Time、Defect Escape Rate |
| セルフレビュー | レビュー依頼前に30%の問題を自分で除去 |

```
レビュー文化の成熟度モデル:

  Level 0: レビューなし（個人作業）
      ↓
  Level 1: 形式的レビュー（LGTM スタンプ）
      ↓
  Level 2: チェックリストベースのレビュー
      ↓
  Level 3: 建設的フィードバック + 自動化
      ↓
  Level 4: メトリクス駆動の継続的改善
      ↓
  Level 5: 知識共有文化としてのレビュー
```

---

## 次に読むべきガイド

- [03-api-design.md](./03-api-design.md) — API設計（レビュー対象となる API の設計原則）
- [../01-practices/04-testing-principles.md](../01-practices/04-testing-principles.md) — テスト原則（テストコードのレビュー観点）
- [../02-refactoring/03-technical-debt.md](../02-refactoring/03-technical-debt.md) — 技術的負債（レビューで負債の蓄積を防ぐ）
- [../00-principles/00-naming-conventions.md](../00-principles/00-naming-conventions.md) — 命名規則（可読性レビューの基準）
- [../00-principles/04-solid-principles.md](../00-principles/04-solid-principles.md) — SOLID原則（設計レビューの基準）
- [00-immutability.md](./00-immutability.md) — イミュータビリティ（コード品質の評価基準）
- [../../design-patterns-guide/docs/04-architectural/](../../design-patterns-guide/docs/04-architectural/) — アーキテクチャパターン（設計レビューの参照）

---

## 参考文献

1. **Software Engineering at Google** — Titus Winters et al. (O'Reilly, 2020) — Google のコードレビュープラクティス
2. **The Art of Readable Code** — Dustin Boswell & Trevor Foucher (O'Reilly, 2011) — 可読性の原則
3. **Google Engineering Practices: Code Review** — https://google.github.io/eng-practices/review/ — Google のレビューガイドライン
4. **SmartBear: Best Practices for Code Review** — https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/ — レビューの定量的研究
5. **Microsoft Research: Code Review Best Practices** — https://www.microsoft.com/en-us/research/publication/code-reviewing-in-the-trenches/ — Microsoft のレビュー研究
6. **Conventional Comments** — https://conventionalcomments.org/ — コメントプレフィックスの標準
7. **GitHub Pull Request Best Practices** — https://docs.github.com/en/pull-requests — PR の公式ガイド
8. **OWASP Secure Code Review Guide** — https://owasp.org/www-project-code-review-guide/ — セキュリティレビューのガイド
9. **Accelerate** — Nicole Forsgren et al. (IT Revolution, 2018) — DevOps メトリクスとレビューの関連
10. **Amy Edmondson, "The Fearless Organization"** (Wiley, 2018) — 心理的安全性とチームパフォーマンス
