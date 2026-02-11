# AIコードレビュー ── 自動レビュー、品質チェック

> AIを活用したコードレビューの自動化手法と品質チェックプロセスを理解し、レビューの速度と精度を大幅に向上させる体制を構築する。

---

## この章で学ぶこと

1. **AIコードレビューツールの活用** ── CodeRabbit、Claude Code等を使った自動レビューの導入方法を学ぶ
2. **レビュー観点の体系化** ── AIが検出すべき問題と人間が判断すべき問題の切り分けを理解する
3. **レビュープロセスの最適化** ── AI+人間のハイブリッドレビューで効率と品質を両立する方法を確立する

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

## FAQ

### Q1: AIレビューで誤検出（False Positive）が多い場合の対処法は？

3つのアプローチがある。(1) ルールファイル(.coderabbit.yaml等)でプロジェクト固有のルールを定義し、誤検出パターンを除外する。(2) path_instructions でディレクトリごとのレビュー指針を設定する。(3) 誤検出をログに記録し、定期的にプロンプトを改善するフィードバックループを回す。

### Q2: 大規模なPR（1000行以上）をAIでレビューする方法は？

大きなPRはまず「分割すべき」と提案するのがベストプラクティス。それが難しい場合は、(1) ファイル単位で個別にレビュー、(2) 変更の種類（リファクタリング、新機能、バグ修正）ごとにグルーピング、(3) 要約を先に生成してからレビューの優先順位を決定する。

### Q3: AIレビューをチームに導入する際の抵抗をどう乗り越えるか？

段階的導入が鍵。(1) まず1つのリポジトリでパイロット導入し、効果を数値で示す（レビュー時間の短縮、検出したバグ数等）。(2) レビュアーの負担軽減という「味方」のポジションで提案。(3) AIレビューは「最終判断」ではなく「ドラフトレビュー」であることを明確にし、人間の権限を脅かさないことを示す。

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
