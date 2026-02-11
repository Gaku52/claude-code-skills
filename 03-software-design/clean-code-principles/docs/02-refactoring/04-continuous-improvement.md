# 継続的改善

> ソフトウェア品質の継続的な向上を、CI/CD パイプライン・自動化された品質ゲート・フィードバックループの構築を通じて実現する方法論を解説する。改善は一度きりのイベントではなく、日常に組み込まれたプロセスである

## この章で学ぶこと

1. **CI/CD パイプラインによる品質自動化** — リント・テスト・カバレッジの自動チェックとゲート設定
2. **メトリクス駆動の改善** — DORA メトリクス、品質トレンドの可視化、改善サイクル
3. **チーム文化としての改善** — 振り返り（レトロスペクティブ）、ボーイスカウトルール、実験的改善

---

## 1. CI/CD パイプラインの品質ゲート

### 1.1 パイプライン全体構成

```
  git push
    |
    v
  [Lint & Format Check] ──失敗──> PR ブロック
    |
    v
  [Unit Tests] ──失敗──> PR ブロック
    |
    v
  [Integration Tests] ──失敗──> PR ブロック
    |
    v
  [Coverage Check] ──80%未満──> PR ブロック
    |
    v
  [Security Scan] ──脆弱性検出──> PR ブロック
    |
    v
  [Build] ──失敗──> PR ブロック
    |
    v
  [Deploy to Staging]
    |
    v
  [E2E Tests on Staging] ──失敗──> デプロイ停止
    |
    v
  [Deploy to Production]
    |
    v
  [Smoke Tests] ──失敗──> 自動ロールバック
```

### 1.2 GitHub Actions 実装

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      # 1. Lint & Format
      - name: Ruff (Lint)
        run: ruff check src/ tests/

      - name: Black (Format)
        run: black --check src/ tests/

      - name: MyPy (Type Check)
        run: mypy src/ --strict

      # 2. Unit Tests + Coverage
      - name: Unit Tests with Coverage
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-fail-under=80 \
            --junitxml=test-results.xml

      # 3. Integration Tests
      - name: Integration Tests
        run: pytest tests/integration/ -v

      # 4. Security Scan
      - name: Bandit (Security)
        run: bandit -r src/ -f json -o bandit-report.json || true

      - name: Safety (Dependency Vulnerabilities)
        run: safety check --json --output safety-report.json

      # 5. Coverage Report
      - name: Upload Coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
          fail_ci_if_error: true
```

### 1.3 pre-commit 設定

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: detect-private-key
```

---

## 2. DORA メトリクス

### 2.1 4つの主要指標

```
DORA (DevOps Research and Assessment) メトリクス

+-------------------+---------------------+
|  デプロイ頻度      |  リードタイム         |
|  (Deployment      |  (Lead Time for     |
|   Frequency)      |   Changes)          |
|                   |                     |
|  Elite: 日次複数回  |  Elite: < 1時間      |
|  High:  日次〜週次  |  High:  < 1日       |
|  Med:   週次〜月次  |  Med:   < 1週間     |
|  Low:   月次以下   |  Low:   > 1ヶ月     |
+-------------------+---------------------+
|  変更失敗率        |  復旧時間            |
|  (Change Failure  |  (Time to Restore)  |
|   Rate)           |                     |
|                   |                     |
|  Elite: < 5%      |  Elite: < 1時間      |
|  High:  < 15%     |  High:  < 1日       |
|  Med:   < 30%     |  Med:   < 1週間     |
|  Low:   > 30%     |  Low:   > 1ヶ月     |
+-------------------+---------------------+
```

### 2.2 メトリクス収集スクリプト

```python
# DORA メトリクス収集
import subprocess
from datetime import datetime, timedelta

def collect_dora_metrics(repo_path: str, days: int = 30) -> dict:
    """過去N日間のDORAメトリクスを収集"""
    since = (datetime.now() - timedelta(days=days)).isoformat()

    # 1. デプロイ頻度
    deploys = subprocess.run(
        ['git', 'log', '--oneline', '--since', since,
         '--grep', 'deploy\\|release', '-i'],
        capture_output=True, text=True, cwd=repo_path
    )
    deploy_count = len(deploys.stdout.strip().split('\n'))
    deploy_frequency = deploy_count / days

    # 2. リードタイム (PR作成からマージまでの平均時間)
    # gh CLI で取得
    prs = subprocess.run(
        ['gh', 'pr', 'list', '--state', 'merged', '--limit', '50', '--json',
         'createdAt,mergedAt'],
        capture_output=True, text=True, cwd=repo_path
    )
    lead_times = calculate_lead_times(prs.stdout)

    # 3. 変更失敗率
    total_deploys = deploy_count
    failed_deploys = count_reverts_and_hotfixes(repo_path, since)
    failure_rate = (failed_deploys / total_deploys * 100) if total_deploys > 0 else 0

    return {
        'deploy_frequency_per_day': round(deploy_frequency, 2),
        'avg_lead_time_hours': round(sum(lead_times) / len(lead_times), 1),
        'change_failure_rate': round(failure_rate, 1),
        'period_days': days,
    }
```

### 2.3 トレンド可視化

```
  デプロイ頻度の推移 (過去6ヶ月)

  回/日
  3.0 |                                          *
  2.5 |                                    *
  2.0 |                              *
  1.5 |                        *
  1.0 |              *   *
  0.5 |  *     *
  0.0 +----+----+----+----+----+----+
      9月   10月  11月  12月  1月   2月

  改善ポイント:
  - 10月: CI/CD パイプライン導入
  - 12月: Feature Flag 導入
  - 2月: Trunk-Based Development 移行
```

---

## 3. 品質改善サイクル

### 3.1 PDCA サイクル

```python
# 改善サイクルの実装例

class ImprovementCycle:
    """品質改善の PDCA サイクル管理"""

    def plan(self):
        """Plan: 現状分析と改善目標設定"""
        current_metrics = collect_dora_metrics()
        targets = {
            'deploy_frequency': max(current_metrics['deploy_frequency'] * 1.5, 1.0),
            'lead_time_hours': current_metrics['avg_lead_time_hours'] * 0.7,
            'test_coverage': min(current_metrics.get('test_coverage', 60) + 10, 95),
        }
        return {'current': current_metrics, 'targets': targets}

    def do(self, actions: list):
        """Do: 改善アクションの実施"""
        for action in actions:
            print(f"実施中: {action['name']}")
            action['execute']()

    def check(self, targets: dict):
        """Check: 効果測定"""
        actual = collect_dora_metrics()
        for metric, target_value in targets.items():
            actual_value = actual.get(metric, 0)
            achieved = actual_value >= target_value
            print(f"  {metric}: 目標={target_value} 実績={actual_value} "
                  f"{'達成' if achieved else '未達'}")

    def act(self, results: dict):
        """Act: 標準化 or 方針修正"""
        for metric, result in results.items():
            if result['achieved']:
                print(f"  {metric}: 改善を標準プロセスに組み込み")
            else:
                print(f"  {metric}: 原因分析 → 次サイクルで再挑戦")
```

### 3.2 振り返り (レトロスペクティブ) テンプレート

```
Sprint N レトロスペクティブ

【Keep (続けること)】
  - PR レビューの24時間以内ルール → リードタイム短縮に効果
  - ペアプログラミングの週1回実施 → 知識共有に効果

【Problem (問題点)】
  - E2E テストが不安定 (Flaky率: 15%)
  - デプロイ後の手動確認に30分かかっている

【Try (次に試すこと)】
  - Flaky テストの quarantine と根本対策 (担当: Alice)
  - Smoke テストの自動化 (担当: Bob)
  - カバレッジ目標を75% → 80%に引き上げ

【メトリクス】
  デプロイ頻度:    1.2/日 → 1.5/日 (目標: 2.0)
  リードタイム:    18時間 → 12時間 (目標: 8時間)
  変更失敗率:      12% → 8% (目標: 5%)
```

---

## 4. 比較表

| 改善手法 | 効果が出るまで | コスト | 持続性 | 適用場面 |
|---------|:----------:|:-----:|:-----:|---------|
| pre-commit hooks | 即座 | 低 | 高 | コードスタイル統一 |
| CI/CD パイプライン | 1-2週間 | 中 | 高 | テスト・ビルド自動化 |
| DORA メトリクス | 1-3ヶ月 | 低 | 高 | チームパフォーマンス可視化 |
| レトロスペクティブ | スプリント単位 | 低 | 中 | プロセス改善 |
| 技術的負債スプリント | 四半期 | 高 | 中 | 大規模改善 |

| 品質ゲート | 検出対象 | 推奨ツール |
|-----------|---------|-----------|
| Lint | コードスタイル・潜在バグ | Ruff, ESLint |
| Type Check | 型安全性 | MyPy, TypeScript |
| Unit Test | ロジックの正しさ | pytest, Jest |
| Coverage | テスト網羅率 | coverage.py, Istanbul |
| Security Scan | 脆弱性 | Bandit, Snyk, Trivy |
| Dependency Audit | 古い依存 | Safety, npm audit |

---

## 5. アンチパターン

### アンチパターン 1: メトリクスの目的化 (Goodhart's Law)

```
BAD: カバレッジ100%を目標にする
  → 意味のないテストが大量に書かれる
  → getter/setter のテスト、assert true のテスト
  → カバレッジは100%だがバグは減らない

GOOD: メトリクスは指標であり目標ではない
  → カバレッジ80%を「最低ライン」として設定
  → クリティカルパスの網羅を重視
  → ミューテーションテストで「テストの品質」も測定
```

### アンチパターン 2: 改善のための改善

```
BAD:
  「最新のツールを導入しよう！」
  → 既存のワークフローを壊す
  → チームの学習コストが高い
  → 実際の品質は改善しない

GOOD: 問題起点の改善
  1. 「本番障害が月3件ある」← 問題
  2. 「テストカバレッジが40%」← 原因
  3. 「CI にカバレッジゲート追加」← 対策
  4. 効果測定: 障害件数の推移を追跡
```

---

## 6. FAQ

### Q1. CI パイプラインが遅い場合の対策は？

**A.** (1) テストの並列実行（`pytest-xdist`, GitHub Actions の `matrix`）。(2) Docker レイヤーキャッシュの活用。(3) 変更されたファイルに関連するテストのみ実行（affected test detection）。(4) ビルドキャッシュ（`actions/cache`）。(5) ユニットテストと統合テストのジョブ分離による並列化。目標は PR のフィードバックまで10分以内。

### Q2. DORA メトリクスの改善が停滞した場合は？

**A.** ボトルネック分析を行う。バリューストリームマッピングで「コード変更からデプロイまで」の各ステップの待ち時間を可視化する。多くの場合、コードレビューの待ち時間、手動テスト、承認プロセスがボトルネック。技術的な改善だけでなく、プロセスやチーム構造の改善（小さな PR、レビュー応答時間の SLA 設定など）が必要になる。

### Q3. 品質ゲートを厳しくしすぎて開発速度が落ちていないか？

**A.** 品質ゲートの導入初期は開発速度が一時的に低下する。しかし2-3ヶ月後には回帰バグの減少・レビューの効率化により、トータルの開発速度は向上する（Accelerate の研究結果）。ゲートが厳しすぎると感じる場合は、Warning レベルと Blocking レベルを分けて段階的に導入する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CI/CD パイプライン | Lint → Test → Coverage → Security → Build → Deploy を自動化 |
| 品質ゲート | PR マージの条件としてカバレッジ80%以上、テスト全通過を設定 |
| DORA メトリクス | デプロイ頻度、リードタイム、変更失敗率、復旧時間の4指標 |
| PDCA サイクル | 計画→実施→確認→定着化の反復的改善 |
| レトロスペクティブ | Keep / Problem / Try で各スプリントを振り返り |
| メトリクスの注意 | Goodhart's Law: メトリクスが目標になると指標としての価値を失う |

---

## 次に読むべきガイド

- [技術的負債](./03-technical-debt.md) — 負債の定量化と返済計画
- [テスト原則](../01-practices/04-testing-principles.md) — 品質ゲートの基盤となるテスト設計
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) — レビューによる品質維持

---

## 参考文献

1. **Accelerate** — Nicole Forsgren, Jez Humble, Gene Kim (IT Revolution, 2018) — DORA メトリクスの研究結果
2. **Continuous Delivery** — Jez Humble & David Farley (Addison-Wesley, 2010) — CI/CD の原典
3. **The Phoenix Project** — Gene Kim, Kevin Behr, George Spafford (IT Revolution, 2013) — DevOps の物語形式の解説
4. **Team Topologies** — Matthew Skelton & Manuel Pais (IT Revolution, 2019) — チーム構造と開発フロー
