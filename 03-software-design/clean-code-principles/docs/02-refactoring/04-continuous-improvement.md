# 継続的改善

> ソフトウェア品質の継続的な向上を、CI/CD パイプライン・自動化された品質ゲート・フィードバックループの構築を通じて実現する方法論を解説する。改善は一度きりのイベントではなく、日常に組み込まれたプロセスである。トヨタ生産方式の「カイゼン」、Lean Software Development の「Build-Measure-Learn」、そして Google の SRE プラクティスを融合し、エンジニアリング組織が持続的に品質を向上させるための体系的フレームワークを提供する

## 前提知識

| トピック | 必要レベル | 参照ガイド |
|---------|----------|-----------|
| テスト原則 | 基礎 | [テスト原則](../01-practices/04-testing-principles.md) |
| 技術的負債 | 基礎 | [技術的負債](./03-technical-debt.md) |
| リファクタリング技法 | 推奨 | [リファクタリング技法](./01-refactoring-techniques.md) |
| コードスメル | 推奨 | [コードスメル](./00-code-smells.md) |
| レガシーコード | 推奨 | [レガシーコード](./02-legacy-code.md) |

## この章で学ぶこと

1. **CI/CD パイプラインの品質ゲート設計** -- リント・テスト・カバレッジ・セキュリティスキャンの自動化と段階的ゲート設定
2. **DORA メトリクスによるチームパフォーマンス測定** -- デプロイ頻度・リードタイム・変更失敗率・復旧時間の4指標とベンチマーク
3. **品質メトリクスの可視化とトレンド分析** -- ダッシュボード構築、劣化検知、改善効果の定量的評価
4. **PDCA/OODA サイクルによる改善プロセス** -- 計画・実行・検証・定着の反復サイクルとレトロスペクティブ
5. **チーム文化としての改善** -- 心理的安全性、実験的改善、ボーイスカウトルール、学習する組織

---

## 1. CI/CD パイプラインの品質ゲート

### 1.1 パイプライン全体アーキテクチャ

```
CI/CD パイプラインの品質ゲートアーキテクチャ

  Developer
     |
     v
  [git push / PR]
     |
     v
  ┌─────────────────────────────────────────────────────┐
  │  Stage 1: Fast Feedback (< 2分)                      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Lint     │  │ Format   │  │ Type     │          │
  │  │ (Ruff)   │  │ (Black)  │  │ Check    │          │
  │  │          │  │          │  │ (MyPy)   │          │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
  │       └──────────┬───┘            │                 │
  │                  v                v                 │
  │            [全通過?] ──No──> PR ブロック              │
  │                  │Yes                               │
  │                  v                                  │
  ├─────────────────────────────────────────────────────┤
  │  Stage 2: Core Verification (< 5分)                  │
  │  ┌──────────┐  ┌──────────┐                         │
  │  │ Unit     │  │ Coverage │                         │
  │  │ Tests    │  │ Check    │                         │
  │  │          │  │ (≥80%)   │                         │
  │  └────┬─────┘  └────┬─────┘                         │
  │       └──────┬───────┘                              │
  │              v                                      │
  │        [全通過?] ──No──> PR ブロック                  │
  │              │Yes                                   │
  │              v                                      │
  ├─────────────────────────────────────────────────────┤
  │  Stage 3: Extended Verification (< 10分)             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Integ.   │  │ Security │  │ Dep.     │          │
  │  │ Tests    │  │ Scan     │  │ Audit    │          │
  │  │          │  │ (Bandit) │  │ (Safety) │          │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
  │       └──────────┬───┘            │                 │
  │                  v                v                 │
  │            [全通過?] ──No──> PR ブロック              │
  │                  │Yes                               │
  │                  v                                  │
  ├─────────────────────────────────────────────────────┤
  │  Stage 4: Build & Deploy                             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Build    │  │ Deploy   │  │ E2E      │          │
  │  │          │  │ Staging  │  │ Tests    │          │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
  │       └──────────┬───┘            │                 │
  │                  v                v                 │
  │            [全通過?] ──No──> デプロイ停止             │
  │                  │Yes                               │
  │                  v                                  │
  ├─────────────────────────────────────────────────────┤
  │  Stage 5: Production                                 │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Deploy   │  │ Smoke    │  │ Monitor  │          │
  │  │ Prod     │  │ Tests    │  │ & Alert  │          │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
  │       │              │              │               │
  │       │        [失敗?]──Yes──> 自動ロールバック       │
  │       │              │No                            │
  │       v              v                              │
  │       ✓ デプロイ完了                                  │
  └─────────────────────────────────────────────────────┘

  設計原則:
  - Fast Feedback First: 軽量チェックを先に（失敗の90%は最初の2分で検出）
  - Fail Fast: 失敗したら即座に後続ステージをスキップ
  - Parallel Execution: 独立したチェックは並列実行
  - Progressive Confidence: ステージが進むほど信頼度が上がる
```

### 1.2 GitHub Actions 実装

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

# 同一PRの実行中ワークフローをキャンセル
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ===== Stage 1: Fast Feedback =====
  lint-and-format:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Ruff (Lint + Format)
        run: |
          ruff check src/ tests/
          ruff format --check src/ tests/

      - name: MyPy (Type Check)
        run: mypy src/ --strict

  # ===== Stage 2: Core Verification =====
  unit-tests:
    needs: lint-and-format
    runs-on: ubuntu-latest
    timeout-minutes: 10
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Unit Tests with Coverage
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=term-missing \
            --cov-fail-under=80 \
            --junitxml=test-results.xml \
            -x -q \
            --timeout=30

      - name: Upload Coverage to Codecov
        if: matrix.python-version == '3.12'
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
          fail_ci_if_error: true
          token: ${{ secrets.CODECOV_TOKEN }}

      - name: Upload Test Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.python-version }}
          path: test-results.xml

  # ===== Stage 3: Extended Verification =====
  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    timeout-minutes: 15
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Integration Tests
        env:
          DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379
        run: pytest tests/integration/ -v --timeout=60

  security-scan:
    needs: unit-tests
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt bandit safety

      - name: Bandit (SAST)
        run: bandit -r src/ -f json -o bandit-report.json -ll
        continue-on-error: true

      - name: Safety (Dependency Vulnerabilities)
        run: safety check --json --output safety-report.json

      - name: Upload Security Reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json

  # ===== Stage 4: Build =====
  build:
    needs: [integration-tests, security-scan]
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker Image
        run: |
          docker build \
            --tag ${{ github.repository }}:${{ github.sha }} \
            --label "org.opencontainers.image.revision=${{ github.sha }}" \
            .

      - name: Trivy Vulnerability Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ github.repository }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
```

### 1.3 pre-commit 設定

```yaml
# .pre-commit-config.yaml
# ローカル開発での品質チェック（CIの前段階）
repos:
  # Ruff: Lint + Format (Python)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # MyPy: Type Check
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-pyyaml]
        args: [--strict]

  # 汎用チェック
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: detect-private-key
      - id: check-merge-conflict
      - id: no-commit-to-branch
        args: [--branch, main, --branch, master]

  # コミットメッセージ規約
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.1.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: [feat, fix, refactor, docs, test, chore, ci, perf]

  # セキュリティ
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### 1.4 品質ゲートの段階的導入

```
品質ゲートの段階的導入ロードマップ

Phase 1 (Week 1-2): 基本ゲート
  ┌────────────────────────────────────────┐
  │  [Warning のみ]                         │
  │  - Lint (Ruff)                         │
  │  - Format (Black/Ruff)                 │
  │  - 基本テスト実行                       │
  │                                        │
  │  目的: チームの慣れ、既存コードの把握    │
  └────────────────────────────────────────┘

Phase 2 (Week 3-4): ブロッキングゲート
  ┌────────────────────────────────────────┐
  │  [Blocking]                             │
  │  - Lint エラー → PR ブロック             │
  │  - テスト失敗 → PR ブロック             │
  │                                        │
  │  [Warning]                              │
  │  - カバレッジ報告                       │
  │  - 型チェック                           │
  │                                        │
  │  目的: 最低限の品質保証                  │
  └────────────────────────────────────────┘

Phase 3 (Month 2): 品質強化
  ┌────────────────────────────────────────┐
  │  [Blocking]                             │
  │  - Lint + Format                       │
  │  - テスト全通過                         │
  │  - カバレッジ ≥ 70%                     │
  │  - 型チェック                           │
  │                                        │
  │  [Warning]                              │
  │  - セキュリティスキャン                  │
  │  - 依存関係監査                         │
  │                                        │
  │  目的: 中程度の品質保証                  │
  └────────────────────────────────────────┘

Phase 4 (Month 3+): フルゲート
  ┌────────────────────────────────────────┐
  │  [Blocking]                             │
  │  - Lint + Format + Type Check          │
  │  - Unit Tests 全通過                    │
  │  - Integration Tests 全通過             │
  │  - カバレッジ ≥ 80%                     │
  │  - セキュリティスキャン (Critical/High)  │
  │  - 依存関係の既知脆弱性 = 0              │
  │                                        │
  │  目的: 高い品質保証                      │
  └────────────────────────────────────────┘
```

---

## 2. DORA メトリクス

### 2.1 4つの主要指標

DORA (DevOps Research and Assessment) メトリクスは、2014年から始まった大規模調査に基づくソフトウェアデリバリーパフォーマンスの4つの主要指標である:

```
DORA メトリクスの4指標とベンチマーク

┌─────────────────────────┬─────────────────────────┐
│  1. デプロイ頻度          │  2. リードタイム           │
│  (Deployment Frequency)  │  (Lead Time for Changes)│
│                          │                         │
│  「どれくらい頻繁に       │  「コミットからデプロイ    │
│   本番にデプロイするか」  │   まで何時間かかるか」    │
│                          │                         │
│  Elite: 日次複数回 (10x) │  Elite: < 1時間          │
│  High:  日次 〜 週次     │  High:  < 1日            │
│  Med:   週次 〜 月次     │  Med:   < 1週間          │
│  Low:   月次以下         │  Low:   > 1ヶ月          │
│                          │                         │
│  [スループット指標]       │  [スループット指標]       │
├─────────────────────────┼─────────────────────────┤
│  3. 変更失敗率            │  4. 復旧時間              │
│  (Change Failure Rate)   │  (Time to Restore)      │
│                          │                         │
│  「デプロイの何%が        │  「障害発生から復旧まで   │
│   障害を引き起こすか」    │   何時間かかるか」       │
│                          │                         │
│  Elite: < 5%             │  Elite: < 1時間          │
│  High:  < 15%            │  High:  < 1日            │
│  Med:   < 30%            │  Med:   < 1週間          │
│  Low:   > 30%            │  Low:   > 1ヶ月          │
│                          │                         │
│  [安定性指標]            │  [安定性指標]            │
└─────────────────────────┴─────────────────────────┘

重要な発見 (Accelerate 研究):
  - Elite パフォーマーは Low と比較して:
    - デプロイ頻度が 973倍 高い
    - リードタイムが 6570倍 短い
    - 復旧時間が 6570倍 短い
    - 変更失敗率が 3倍 低い
  - スループットと安定性はトレードオフではなく、両立する
```

### 2.2 メトリクス収集と自動化

```python
"""DORA メトリクス収集フレームワーク"""
import subprocess
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DORAMetrics:
    """DORA メトリクスのデータモデル"""
    # 計測期間
    period_start: datetime
    period_end: datetime

    # 1. デプロイ頻度
    deploy_count: int = 0
    deploy_frequency_per_day: float = 0.0

    # 2. リードタイム
    lead_times_hours: list[float] = field(default_factory=list)

    # 3. 変更失敗率
    total_changes: int = 0
    failed_changes: int = 0

    # 4. 復旧時間
    restore_times_hours: list[float] = field(default_factory=list)

    @property
    def period_days(self) -> int:
        return (self.period_end - self.period_start).days

    @property
    def avg_lead_time_hours(self) -> float:
        if not self.lead_times_hours:
            return 0.0
        return sum(self.lead_times_hours) / len(self.lead_times_hours)

    @property
    def median_lead_time_hours(self) -> float:
        if not self.lead_times_hours:
            return 0.0
        sorted_times = sorted(self.lead_times_hours)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]

    @property
    def change_failure_rate(self) -> float:
        if self.total_changes == 0:
            return 0.0
        return (self.failed_changes / self.total_changes) * 100

    @property
    def avg_restore_time_hours(self) -> float:
        if not self.restore_times_hours:
            return 0.0
        return sum(self.restore_times_hours) / len(self.restore_times_hours)

    def classify(self, metric: str) -> str:
        """指標のパフォーマンスレベルを判定"""
        classifications = {
            "deploy_frequency": [
                (1.0, "Elite"),    # 日次以上
                (0.14, "High"),    # 週次以上
                (0.03, "Medium"),  # 月次以上
                (0.0, "Low"),
            ],
            "lead_time": [
                (1.0, "Elite"),    # 1時間以内
                (24.0, "High"),    # 1日以内
                (168.0, "Medium"), # 1週間以内
                (float("inf"), "Low"),
            ],
            "change_failure_rate": [
                (5.0, "Elite"),
                (15.0, "High"),
                (30.0, "Medium"),
                (float("inf"), "Low"),
            ],
            "restore_time": [
                (1.0, "Elite"),
                (24.0, "High"),
                (168.0, "Medium"),
                (float("inf"), "Low"),
            ],
        }

        if metric == "deploy_frequency":
            value = self.deploy_frequency_per_day
            for threshold, level in classifications[metric]:
                if value >= threshold:
                    return level
        elif metric == "lead_time":
            value = self.median_lead_time_hours
            for threshold, level in classifications[metric]:
                if value <= threshold:
                    return level
        elif metric == "change_failure_rate":
            value = self.change_failure_rate
            for threshold, level in classifications[metric]:
                if value <= threshold:
                    return level
        elif metric == "restore_time":
            value = self.avg_restore_time_hours
            for threshold, level in classifications[metric]:
                if value <= threshold:
                    return level

        return "Low"

    @property
    def overall_level(self) -> str:
        """総合パフォーマンスレベル"""
        levels = {
            "Elite": 4, "High": 3, "Medium": 2, "Low": 1
        }
        metrics = [
            self.classify("deploy_frequency"),
            self.classify("lead_time"),
            self.classify("change_failure_rate"),
            self.classify("restore_time"),
        ]
        avg_score = sum(levels[m] for m in metrics) / len(metrics)

        if avg_score >= 3.5: return "Elite"
        elif avg_score >= 2.5: return "High"
        elif avg_score >= 1.5: return "Medium"
        else: return "Low"


def collect_dora_metrics(
    repo_path: str,
    days: int = 30,
    deploy_tag_pattern: str = "v*"
) -> DORAMetrics:
    """Git + GitHub CLI からDORAメトリクスを収集"""
    end = datetime.now()
    start = end - timedelta(days=days)
    since = start.isoformat()

    metrics = DORAMetrics(period_start=start, period_end=end)

    # 1. デプロイ頻度 (タグベース)
    result = subprocess.run(
        ["git", "tag", "-l", deploy_tag_pattern, "--sort=-creatordate",
         "--format=%(creatordate:iso)"],
        capture_output=True, text=True, cwd=repo_path
    )
    deploy_dates = []
    for line in result.stdout.strip().split("\n"):
        if line:
            try:
                dt = datetime.fromisoformat(line.strip().split("+")[0].strip())
                if dt >= start:
                    deploy_dates.append(dt)
            except ValueError:
                continue

    metrics.deploy_count = len(deploy_dates)
    metrics.deploy_frequency_per_day = metrics.deploy_count / max(days, 1)

    # 2. リードタイム (PR作成 → マージまでの時間)
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "merged", "--limit", "100",
             "--json", "createdAt,mergedAt"],
            capture_output=True, text=True, cwd=repo_path, timeout=30
        )
        if result.returncode == 0:
            prs = json.loads(result.stdout)
            for pr in prs:
                created = datetime.fromisoformat(pr["createdAt"].replace("Z", "+00:00"))
                merged = datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00"))
                if created >= start.replace(tzinfo=created.tzinfo):
                    lead_time = (merged - created).total_seconds() / 3600
                    metrics.lead_times_hours.append(lead_time)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 3. 変更失敗率 (revert / hotfix コミットの割合)
    result = subprocess.run(
        ["git", "log", "--oneline", "--since", since],
        capture_output=True, text=True, cwd=repo_path
    )
    all_commits = [
        line for line in result.stdout.strip().split("\n") if line
    ]
    metrics.total_changes = len(all_commits)

    result = subprocess.run(
        ["git", "log", "--oneline", "--since", since,
         "--grep=revert\\|hotfix\\|rollback", "-i"],
        capture_output=True, text=True, cwd=repo_path
    )
    failed_commits = [
        line for line in result.stdout.strip().split("\n") if line
    ]
    metrics.failed_changes = len(failed_commits)

    return metrics


def print_dora_dashboard(metrics: DORAMetrics) -> None:
    """DORAメトリクスダッシュボード"""
    width = 64
    print("=" * width)
    print("  DORA メトリクスダッシュボード".center(width))
    print(f"  期間: {metrics.period_start.strftime('%Y-%m-%d')} "
          f"~ {metrics.period_end.strftime('%Y-%m-%d')} "
          f"({metrics.period_days}日間)".center(width))
    print("=" * width)

    # デプロイ頻度
    level = metrics.classify("deploy_frequency")
    print(f"\n  [1] デプロイ頻度")
    print(f"      回数: {metrics.deploy_count} 回")
    print(f"      頻度: {metrics.deploy_frequency_per_day:.2f} 回/日")
    print(f"      レベル: {level}")

    # リードタイム
    level = metrics.classify("lead_time")
    print(f"\n  [2] リードタイム (PR作成→マージ)")
    print(f"      平均: {metrics.avg_lead_time_hours:.1f} 時間")
    print(f"      中央値: {metrics.median_lead_time_hours:.1f} 時間")
    print(f"      レベル: {level}")

    # 変更失敗率
    level = metrics.classify("change_failure_rate")
    print(f"\n  [3] 変更失敗率")
    print(f"      総変更: {metrics.total_changes} 件")
    print(f"      失敗: {metrics.failed_changes} 件")
    print(f"      失敗率: {metrics.change_failure_rate:.1f}%")
    print(f"      レベル: {level}")

    # 復旧時間
    level = metrics.classify("restore_time")
    print(f"\n  [4] 復旧時間")
    print(f"      平均: {metrics.avg_restore_time_hours:.1f} 時間")
    print(f"      レベル: {level}")

    # 総合
    print(f"\n" + "-" * width)
    overall = metrics.overall_level
    print(f"  総合パフォーマンス: {overall}")
    print("=" * width)
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

  改善イベントの対応:
  - 10月: CI/CD パイプライン導入 → 自動デプロイ開始
  - 12月: Feature Flag 導入 → デプロイとリリースの分離
  - 2月:  Trunk-Based Development 移行 → 小さなPR文化


リードタイムの推移 (過去6ヶ月)

  時間
  72  |  *
  48  |     *     *
  24  |              *
  12  |                    *
   8  |                        *   *
   4  |                                 *
   2  |                                      *
   0  +----+----+----+----+----+----+
      9月   10月  11月  12月  1月   2月

  改善イベントの対応:
  - 10月: PR サイズ制限導入 (< 400行)
  - 12月: レビュー応答 SLA 導入 (< 4時間)
  - 2月:  自動レビューツール導入 (CodeRabbit)


変更失敗率の推移 (過去6ヶ月)

  %
  25  |  *
  20  |     *
  15  |           *
  10  |              *
   8  |                    *
   5  |                         *   *   *
   0  +----+----+----+----+----+----+
      9月   10月  11月  12月  1月   2月

  改善イベントの対応:
  - 10月: カバレッジゲート 60% 導入
  - 11月: カバレッジゲート 70% に引き上げ
  - 1月:  カバレッジゲート 80% + E2E テスト追加
```

---

## 3. 品質メトリクスの可視化

### 3.1 品質ダッシュボード

```python
"""品質メトリクスの統合ダッシュボード"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QualitySnapshot:
    """ある時点の品質メトリクスのスナップショット"""
    timestamp: datetime

    # コード品質
    avg_complexity: float
    duplication_percent: float
    type_coverage_percent: float

    # テスト
    test_coverage_percent: float
    test_count: int
    test_pass_rate: float       # テスト成功率 (%)
    test_execution_sec: float

    # セキュリティ
    known_vulnerabilities: int
    security_hotspots: int

    # 保守性
    todo_fixme_count: int
    outdated_deps: int
    avg_file_size_lines: float


@dataclass
class QualityTrend:
    """品質メトリクスのトレンド分析"""
    snapshots: list[QualitySnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: QualitySnapshot) -> None:
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.timestamp)

    def get_trend(self, metric: str, periods: int = 6) -> list[tuple[datetime, float]]:
        """指定メトリクスのトレンドデータを取得"""
        recent = self.snapshots[-periods:]
        return [(s.timestamp, getattr(s, metric, 0.0)) for s in recent]

    def detect_degradation(
        self,
        metric: str,
        threshold_percent: float = 10.0,
        higher_is_better: bool = True
    ) -> bool:
        """品質劣化を検出

        直近2回のスナップショットを比較し、
        threshold_percent 以上の劣化があれば True
        """
        if len(self.snapshots) < 2:
            return False

        current = getattr(self.snapshots[-1], metric, 0.0)
        previous = getattr(self.snapshots[-2], metric, 0.0)

        if previous == 0:
            return False

        change_percent = ((current - previous) / abs(previous)) * 100

        if higher_is_better:
            return change_percent < -threshold_percent
        else:
            return change_percent > threshold_percent

    def generate_report(self) -> str:
        """品質トレンドレポートを生成"""
        if not self.snapshots:
            return "データなし"

        current = self.snapshots[-1]
        lines = [
            "=" * 60,
            "  品質トレンドレポート",
            f"  時点: {current.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
        ]

        # 劣化検出
        degradations = []
        checks = [
            ("test_coverage_percent", "テストカバレッジ", True),
            ("avg_complexity", "平均複雑度", False),
            ("duplication_percent", "コード重複率", False),
            ("known_vulnerabilities", "既知脆弱性", False),
            ("outdated_deps", "古い依存関係", False),
        ]

        for metric, name, higher_is_better in checks:
            if self.detect_degradation(metric, 10.0, higher_is_better):
                degradations.append(name)

        if degradations:
            lines.append(f"\n  [ALERT] 品質劣化検出:")
            for d in degradations:
                lines.append(f"    - {d}")
        else:
            lines.append(f"\n  [OK] 品質劣化なし")

        lines.append("=" * 60)
        return "\n".join(lines)
```

### 3.2 GitHub Actions での品質トレンド収集

```yaml
# .github/workflows/quality-trend.yml
name: Quality Trend Tracking

on:
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜9時
  workflow_dispatch:

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install tools
        run: |
          pip install -r requirements.txt -r requirements-dev.txt
          pip install radon

      - name: Collect quality metrics
        run: |
          python -c "
          import json
          from datetime import datetime

          metrics = {
            'timestamp': datetime.now().isoformat(),
            'test_coverage': $(pytest --cov=src --cov-report=json -q 2>/dev/null; python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered'])" 2>/dev/null || echo 0),
            'avg_complexity': $(radon cc src/ -a -j | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('average',0) if isinstance(d,dict) else 0)" 2>/dev/null || echo 0),
            'todo_count': $(grep -r -c -E 'TODO|FIXME|HACK' src/ 2>/dev/null | awk -F: '{s+=\$2}END{print s+0}'),
          }

          # トレンドファイルに追加
          try:
            with open('quality-trend.json') as f:
              trend = json.load(f)
          except FileNotFoundError:
            trend = []

          trend.append(metrics)

          # 直近52週分を保持
          trend = trend[-52:]

          with open('quality-trend.json', 'w') as f:
            json.dump(trend, f, indent=2)

          print(json.dumps(metrics, indent=2))
          "

      - name: Commit trend data
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add quality-trend.json
          git diff --staged --quiet || git commit -m "chore: update quality trend data"
          git push
```

---

## 4. 改善サイクル

### 4.1 PDCA サイクル

```python
"""改善サイクルの構造化フレームワーク"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CyclePhase(Enum):
    PLAN = "plan"
    DO = "do"
    CHECK = "check"
    ACT = "act"


@dataclass
class ImprovementGoal:
    """改善目標"""
    metric: str
    current_value: float
    target_value: float
    deadline: datetime
    owner: str
    actions: list[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        if self.target_value == self.current_value:
            return 100.0
        return min(100.0, max(0.0,
            abs(self.current_value - self.target_value) /
            abs(self.target_value - self.current_value) * 100
        ))


@dataclass
class ImprovementCycle:
    """品質改善の PDCA サイクル管理"""
    cycle_id: str
    phase: CyclePhase = CyclePhase.PLAN
    goals: list[ImprovementGoal] = field(default_factory=list)
    learnings: list[str] = field(default_factory=list)

    def plan(self, current_metrics: dict, improvement_areas: list[str]) -> dict:
        """Plan: 現状分析と改善目標設定

        ステップ:
        1. 現状メトリクスの分析
        2. ボトルネックの特定
        3. 改善目標の設定 (SMART原則)
        4. アクションプランの策定
        """
        targets = {}
        for area in improvement_areas:
            current = current_metrics.get(area, 0)

            # 改善目標: 現状から10-30%改善
            if area in ("test_coverage", "deploy_frequency"):
                target = min(current * 1.2, 95.0)  # 20%向上、上限95%
            elif area in ("avg_complexity", "lead_time_hours"):
                target = current * 0.8  # 20%削減
            elif area == "change_failure_rate":
                target = max(current * 0.7, 5.0)  # 30%削減、下限5%
            else:
                target = current * 1.1

            targets[area] = {
                "current": current,
                "target": round(target, 1),
                "improvement": f"{abs((target - current) / max(current, 0.1) * 100):.0f}%",
            }

        self.phase = CyclePhase.PLAN
        return {"current_metrics": current_metrics, "targets": targets}

    def do(self, actions: list[dict]) -> list[str]:
        """Do: 改善アクションの実施

        各アクションは以下の形式:
        {"name": str, "owner": str, "deadline": str, "execute": callable}
        """
        results = []
        self.phase = CyclePhase.DO

        for action in actions:
            try:
                action["execute"]()
                results.append(f"[OK] {action['name']} (担当: {action['owner']})")
            except Exception as e:
                results.append(f"[NG] {action['name']}: {e}")

        return results

    def check(self, targets: dict, actual_metrics: dict) -> dict:
        """Check: 効果測定

        各目標に対する達成度を評価
        """
        self.phase = CyclePhase.CHECK
        results = {}

        for metric, target_info in targets.items():
            target_value = target_info["target"]
            actual_value = actual_metrics.get(metric, 0)
            current_value = target_info["current"]

            # 改善方向に応じた達成判定
            if metric in ("test_coverage", "deploy_frequency"):
                achieved = actual_value >= target_value
                improvement = actual_value - current_value
            else:
                achieved = actual_value <= target_value
                improvement = current_value - actual_value

            results[metric] = {
                "target": target_value,
                "actual": actual_value,
                "achieved": achieved,
                "improvement": round(improvement, 1),
            }

        return results

    def act(self, check_results: dict) -> dict:
        """Act: 標準化 or 方針修正

        達成: 改善をプロセスに組み込み（標準化）
        未達: 原因分析 → 次サイクルの Plan に反映
        """
        self.phase = CyclePhase.ACT
        actions = {}

        for metric, result in check_results.items():
            if result["achieved"]:
                actions[metric] = {
                    "action": "standardize",
                    "detail": f"{metric}: 改善をCI/CDパイプラインに組み込み、"
                              f"閾値を {result['actual']} に更新",
                }
            else:
                actions[metric] = {
                    "action": "adjust",
                    "detail": f"{metric}: 原因分析を実施。"
                              f"目標={result['target']}, 実績={result['actual']}。"
                              f"次サイクルで対策を強化",
                }
                self.learnings.append(
                    f"{metric}: 目標未達。Gap={result['target'] - result['actual']:.1f}"
                )

        return actions
```

### 4.2 OODA ループ（高速改善向け）

```
OODA ループ (インシデント対応・緊急改善向け)

  ┌──────────┐      ┌──────────┐
  │ Observe  │ ──→  │ Orient   │
  │ (観察)   │      │ (情勢判断)│
  └──────────┘      └────┬─────┘
       ↑                  │
       │                  v
  ┌──────────┐      ┌──────────┐
  │ Act      │ ←──  │ Decide   │
  │ (行動)   │      │ (決定)   │
  └──────────┘      └──────────┘

  PDCA との違い:
  ┌────────────┬────────────────┬────────────────┐
  │            │ PDCA           │ OODA           │
  ├────────────┼────────────────┼────────────────┤
  │ サイクル速度│ 数週間〜数ヶ月  │ 数分〜数時間    │
  │ 適用場面    │ 計画的改善      │ 緊急対応・実験  │
  │ 重視する点  │ 計画の精度      │ 判断の速度      │
  │ フィードバック│ メトリクス     │ リアルタイム監視 │
  └────────────┴────────────────┴────────────────┘

  例: 本番インシデント対応
  Observe: アラート検知 → エラーログ確認 → 影響範囲特定
  Orient:  根本原因の仮説立て → 過去の類似インシデント参照
  Decide:  ロールバック or ホットフィックス → 対応方針決定
  Act:     対応実施 → 結果確認 → 必要に応じて再ループ
```

### 4.3 振り返り（レトロスペクティブ）

```
スプリント レトロスペクティブ テンプレート

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sprint N レトロスペクティブ (YYYY-MM-DD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Keep (続けること)】
  + PR レビューの24時間以内ルール → リードタイム短縮に効果
  + ペアプログラミングの週1回実施 → 知識共有に効果
  + 毎朝の15分スタンドアップ → ブロッカーの早期発見

【Problem (問題点)】
  - E2E テストが不安定 (Flaky率: 15%)
  - デプロイ後の手動確認に30分かかっている
  - コードレビューの待ち時間が平均8時間

【Try (次に試すこと)】
  - [ ] Flaky テストの quarantine と根本対策 (担当: Alice, 期限: Sprint N+1)
  - [ ] Smoke テストの自動化 (担当: Bob, 期限: Sprint N+2)
  - [ ] カバレッジ目標を 75% → 80% に引き上げ (チーム全体)
  - [ ] レビュー応答時間の SLA: 4時間以内 (トライアル)

【メトリクス (前回 → 今回)】
  デプロイ頻度:    1.2/日 → 1.5/日  [+25%]    (目標: 2.0/日)
  リードタイム:    18h → 12h        [-33%]    (目標: 8h)
  変更失敗率:      12% → 8%         [-33%]    (目標: 5%)
  カバレッジ:      72% → 75%        [+4%]     (目標: 80%)
  ビルド時間:      8min → 6min      [-25%]    (目標: 5min)

【前回の Try の結果】
  [達成] テスト並列化 → ビルド時間 8分→6分
  [未達] API ドキュメント自動生成 → リソース不足で着手できず → 次Sprintに繰越
  [達成] pre-commit hooks 導入 → Lint違反のPRが0件に

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```python
"""レトロスペクティブの構造化"""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class RetroItem:
    """レトロスペクティブのアイテム"""
    category: str   # "keep", "problem", "try"
    description: str
    owner: str = ""
    deadline: str = ""
    status: str = "open"  # open, achieved, not_achieved, carried_over

    def __str__(self) -> str:
        prefix = {"keep": "+", "problem": "-", "try": "[ ]"}
        return f"  {prefix.get(self.category, '?')} {self.description}"


@dataclass
class SprintRetro:
    """スプリントレトロスペクティブ"""
    sprint_name: str
    date: date
    items: list[RetroItem] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    previous_try_results: list[dict] = field(default_factory=list)

    @property
    def keeps(self) -> list[RetroItem]:
        return [i for i in self.items if i.category == "keep"]

    @property
    def problems(self) -> list[RetroItem]:
        return [i for i in self.items if i.category == "problem"]

    @property
    def tries(self) -> list[RetroItem]:
        return [i for i in self.items if i.category == "try"]

    def carry_over_unfinished(self) -> list[RetroItem]:
        """未達のTryを次回に繰り越し"""
        return [
            RetroItem(
                category="try",
                description=f"[繰越] {item.description}",
                owner=item.owner,
                deadline=item.deadline,
                status="carried_over",
            )
            for item in self.tries
            if item.status == "not_achieved"
        ]

    def effectiveness_score(self) -> float:
        """レトロスペクティブの効果スコア

        前回のTryのうち達成されたものの割合
        """
        if not self.previous_try_results:
            return 0.0

        achieved = sum(
            1 for r in self.previous_try_results
            if r.get("status") == "achieved"
        )
        return (achieved / len(self.previous_try_results)) * 100
```

---

## 5. チーム文化としての改善

### 5.1 心理的安全性と改善文化

```
改善文化の構築ピラミッド

                    ┌─────────────┐
                    │ 実験的改善   │
                    │ (Innovation) │
                    ├─────────────┤
                    │ 継続的改善   │
                    │ (Kaizen)    │
                    ├─────────────┤
                    │ 標準化       │
                    │ (Standards) │
                    ├─────────────┤
                    │ 心理的安全性  │
                    │ (Safety)    │
                    └─────────────┘

各レベルの特徴:

  1. 心理的安全性 (基盤)
     - 「このコード、もっと良くできますね」と気軽に言える
     - 「わかりません」と言える
     - 失敗を責めるのではなく、学びとして共有する
     - バグレポートは非難ではなく感謝

  2. 標準化
     - コーディング規約の合意と自動チェック
     - テストの最低基準 (Definition of Done)
     - CI/CD パイプラインの品質ゲート
     - ドキュメントテンプレート

  3. 継続的改善 (カイゼン)
     - ボーイスカウトルールの実践
     - 毎スプリント 20% を改善に
     - レトロスペクティブの定期開催
     - メトリクスによる改善の可視化

  4. 実験的改善
     - 新ツール・プラクティスの試験導入
     - A/B テスト的なプロセス改善
     - ハッカソン・20%タイム
     - 失敗を前提とした小さな実験
```

### 5.2 改善のための組織プラクティス

```python
"""改善プラクティスの実装パターン"""


class ImprovementPractices:
    """チーム改善プラクティス集"""

    @staticmethod
    def blameless_postmortem_template() -> str:
        """ブレームレス・ポストモーテムのテンプレート"""
        return """
        ========================================
        ポストモーテム: [インシデント名]
        日時: [YYYY-MM-DD]
        ========================================

        ## タイムライン
        - HH:MM 検知: [何が起きたか]
        - HH:MM 対応開始: [誰が何をしたか]
        - HH:MM 解決: [どう解決したか]

        ## 影響範囲
        - 影響ユーザー数: [N人]
        - ダウンタイム: [N分]
        - 収益影響: [推定金額]

        ## 根本原因
        [5 Whys 分析]
        Why 1: なぜサービスが停止した？→ OOM Kill
        Why 2: なぜメモリ不足？→ メモリリーク
        Why 3: なぜリークを検知できなかった？→ 監視がなかった
        Why 4: なぜ監視がなかった？→ 設定を忘れていた
        Why 5: なぜ忘れた？→ チェックリストがなかった

        ## アクションアイテム
        - [ ] メモリ監視アラートの追加 (担当: Alice, 期限: MM/DD)
        - [ ] デプロイ前チェックリストに監視確認を追加 (担当: Bob)
        - [ ] メモリリーク検出のテスト追加 (担当: Carol)

        ## 学び
        - [今回の教訓を文書化]

        ## 注意: ポストモーテムは「誰のせいか」ではなく
                「システムとプロセスをどう改善するか」に焦点を当てる
        """

    @staticmethod
    def tech_radar_categories() -> dict:
        """Technology Radar (技術選定の可視化)"""
        return {
            "Adopt (推奨)": [
                "Python 3.12", "pytest", "Ruff", "GitHub Actions",
                "PostgreSQL 16", "Redis 7", "Docker",
            ],
            "Trial (試用中)": [
                "FastAPI", "Pydantic v2", "uv (package manager)",
                "Playwright (E2E)", "OpenTelemetry",
            ],
            "Assess (評価中)": [
                "Rust (パフォーマンスクリティカル部分)",
                "Deno", "Bun", "Effect-TS",
            ],
            "Hold (非推奨)": [
                "Django (新規プロジェクト)", "unittest (pytest推奨)",
                "Travis CI (GitHub Actions推奨)", "Python 3.9以前",
            ],
        }

    @staticmethod
    def definition_of_done() -> list[str]:
        """Definition of Done (完了の定義)"""
        return [
            "[ ] コードがリファクタリングされている（ボーイスカウトルール適用）",
            "[ ] ユニットテストが書かれている（カバレッジ ≥ 80%）",
            "[ ] 型ヒントが追加されている（MyPy strict パス）",
            "[ ] Lint/Format チェックをパスしている",
            "[ ] コードレビューが完了している（1名以上の承認）",
            "[ ] 統合テストが更新されている（必要な場合）",
            "[ ] ドキュメントが更新されている（API変更の場合）",
            "[ ] セキュリティチェックをパスしている",
            "[ ] パフォーマンスへの影響が確認されている",
            "[ ] 新しい技術的負債がバックログに記録されている（発生した場合）",
        ]
```

### 5.3 改善の阻害要因と対策

```
改善の阻害要因と対策マップ

  阻害要因                        対策
  ┌───────────────────────┐     ┌───────────────────────┐
  │ 「改善する時間がない」  │────→│ 20%ルールの制度化       │
  │                       │     │ スプリント計画に組込     │
  └───────────────────────┘     └───────────────────────┘
  ┌───────────────────────┐     ┌───────────────────────┐
  │ 「効果が見えない」     │────→│ メトリクスの可視化       │
  │                       │     │ Before/After の定量比較  │
  └───────────────────────┘     └───────────────────────┘
  ┌───────────────────────┐     ┌───────────────────────┐
  │ 「失敗が怖い」        │────→│ 心理的安全性の構築       │
  │                       │     │ ブレームレス文化         │
  └───────────────────────┘     └───────────────────────┘
  ┌───────────────────────┐     ┌───────────────────────┐
  │ 「何を改善すべきか     │────→│ DORA メトリクス          │
  │  わからない」          │     │ ホットスポート分析       │
  └───────────────────────┘     └───────────────────────┘
  ┌───────────────────────┐     ┌───────────────────────┐
  │ 「改善が続かない」     │────→│ レトロスペクティブ       │
  │                       │     │ PDCA サイクルの制度化    │
  └───────────────────────┘     └───────────────────────┘
  ┌───────────────────────┐     ┌───────────────────────┐
  │ 「経営層の理解がない」  │────→│ コスト試算で説明         │
  │                       │     │ ROI ベースの提案         │
  └───────────────────────┘     └───────────────────────┘
```

---

## 6. 比較表

### 6.1 改善手法比較

| 改善手法 | 効果発現 | コスト | 持続性 | リスク | 適用場面 |
|---------|:-------:|:-----:|:-----:|:-----:|---------|
| pre-commit hooks | 即座 | 低 | 高 | 最小 | コードスタイル統一、基本チェック |
| CI/CD パイプライン | 1-2週間 | 中 | 高 | 低 | テスト・ビルド・デプロイ自動化 |
| DORA メトリクス | 1-3ヶ月 | 低 | 高 | 最小 | チームパフォーマンス可視化 |
| レトロスペクティブ | Sprint単位 | 低 | 中 | 低 | プロセス改善、チーム学習 |
| 20%ルール | 2-4週 | 低 | 高 | 低 | 計画的な品質向上 |
| 技術的負債スプリント | 2-4週 | 高 | 中 | 中 | 蓄積した負債の集中返済 |
| Feature Flag | 1-2週間 | 中 | 高 | 低 | デプロイとリリースの分離 |
| Trunk-Based Dev | 1-3ヶ月 | 中 | 高 | 中 | 開発フロー最適化 |

### 6.2 品質ゲート一覧

| 品質ゲート | 検出対象 | 推奨ツール (Python) | 推奨ツール (TypeScript) | 推奨閾値 |
|-----------|---------|-------------------|----------------------|---------|
| Lint | コードスタイル・潜在バグ | Ruff | ESLint | エラー0件 |
| Format | コードフォーマット | Ruff Format | Prettier | 差分0件 |
| Type Check | 型安全性 | MyPy (strict) | TypeScript (strict) | エラー0件 |
| Unit Test | ロジックの正しさ | pytest | Jest / Vitest | 全通過 |
| Coverage | テスト網羅率 | coverage.py | Istanbul / c8 | >= 80% |
| Integration Test | コンポーネント連携 | pytest | Jest / Playwright | 全通過 |
| Security Scan (SAST) | コードの脆弱性 | Bandit | ESLint Security | Critical/High: 0件 |
| Security Scan (SCA) | 依存の脆弱性 | Safety / pip-audit | npm audit / Snyk | Critical: 0件 |
| Container Scan | コンテナ脆弱性 | Trivy | Trivy | Critical/High: 0件 |
| Complexity | コード複雑度 | radon | ESLint complexity | CC < 10 |

### 6.3 CI パイプライン速度最適化

| 最適化手法 | 効果 | 実装コスト | 適用条件 |
|-----------|------|----------|---------|
| テスト並列化 (xdist) | ビルド時間 50-70% 削減 | 低 | テストが独立している |
| Docker レイヤーキャッシュ | ビルド時間 30-50% 削減 | 低 | Docker ビルドあり |
| 依存キャッシュ (actions/cache) | インストール時間 80% 削減 | 低 | 常に有効 |
| Affected Test Detection | テスト時間 60-80% 削減 | 高 | モノレポ、大規模テスト |
| ジョブ並列化 (matrix) | ビルド時間 40-60% 削減 | 低 | 複数環境テスト |
| Spot/Preemptible Runners | コスト 60-80% 削減 | 中 | 非緊急ジョブ |

---

## 7. 演習問題

### 演習 1: CI/CD パイプラインの設計（基礎）

以下の要件に基づき、GitHub Actions の CI パイプラインを設計せよ。

```
プロジェクト情報:
- 言語: Python 3.12
- フレームワーク: FastAPI
- DB: PostgreSQL 16
- テスト: pytest (unit + integration)
- 現状: CI なし、手動デプロイ

要件:
1. PR 時に自動実行される品質チェック
2. Lint + Format + Type Check (Stage 1)
3. Unit Tests + Coverage >= 75% (Stage 2)
4. Integration Tests with PostgreSQL (Stage 3)
5. セキュリティスキャン (Stage 4)
6. main ブランチへのマージ時に Staging デプロイ

成果物:
- .github/workflows/ci.yml の完全な YAML
- .pre-commit-config.yaml
- 品質ゲートの閾値設定
```

**期待される回答のポイント:**
- ステージ間の依存関係（needs）の正しい設定
- PostgreSQL サービスコンテナの設定
- キャッシュの活用（pip, Docker）
- タイムアウトの設定
- concurrency によるPRの重複実行防止

### 演習 2: DORA メトリクス改善計画（応用）

以下のチームの DORA メトリクスを分析し、改善計画を策定せよ。

```
現状のメトリクス:
- デプロイ頻度: 0.3回/日 (週2回程度)
- リードタイム: 72時間 (PR作成→マージ)
- 変更失敗率: 18%
- 復旧時間: 4時間

ボトルネック（バリューストリームマッピング結果）:
- コーディング: 4時間
- PR作成: 30分
- レビュー待ち: 24時間  ← ボトルネック1
- レビュー: 2時間
- CI パイプライン: 20分
- マージ後の手動テスト: 4時間  ← ボトルネック2
- 手動デプロイ: 2時間  ← ボトルネック3

課題:
1. 各メトリクスのパフォーマンスレベルを判定
2. 3つのボトルネックに対する改善策を提案
3. 3ヶ月後の目標メトリクスを設定
4. 改善の PDCA サイクルを設計
5. レトロスペクティブのテンプレートを作成
```

**期待される回答のポイント:**
- レビュー待ち → レビュー応答 SLA 4時間、PRサイズ制限 400行
- 手動テスト → E2E テスト自動化、Smoke テスト
- 手動デプロイ → CI/CD パイプライン、自動デプロイ
- 3ヶ月目標: デプロイ1.0/日、リードタイム24h、変更失敗率10%

### 演習 3: 改善文化の構築計画（発展）

レガシーなチーム文化を持つ組織で、継続的改善を定着させる計画を策定せよ。

```
現状:
- チーム: 12名 (バックエンド6名、フロントエンド4名、QA2名)
- テスト文化: QA チームが手動テストを実施。開発者はテストを書かない
- CI: Jenkins が動いているが、失敗しても無視される
- レトロスペクティブ: 実施されていない
- コードレビュー: 形骸化（LGTM のみ）
- 技術的負債: 膨大だが定量化されていない

課題:
1. 心理的安全性を高めるための具体的アクション (5つ)
2. 品質ゲートの段階的導入計画 (4フェーズ)
3. 開発者がテストを書く文化への移行計画
4. メトリクス駆動の改善サイクル設計
5. 6ヶ月後の KPI と測定方法
6. 経営層への説得資料 (ROI 試算)

評価基準:
- 段階的アプローチの妥当性
- チーム文化への配慮
- 定量的な目標設定
- 持続可能性
- 経営層への説得力
```

---

## 8. アンチパターン

### アンチパターン 1: メトリクスの目的化 (Goodhart's Law)

```
NG パターン:
  「カバレッジ 100% を目標にする」
  → 意味のないテストが大量に書かれる
  → getter/setter のテスト、assert True のテスト
  → カバレッジは 100% だがバグは減らない
  → 開発速度は低下、チームのモチベーションも低下

  「毎日デプロイすることが目標」
  → 中身のない空デプロイが増える
  → DORA メトリクスは改善するが品質は変わらない

  "When a measure becomes a target,
   it ceases to be a good measure."
  -- Goodhart's Law

OK パターン:
  メトリクスは「指標」であり「目標」ではない
  → カバレッジ 80% を「最低ライン」として設定
  → クリティカルパスの網羅を重視
  → ミューテーションテストで「テストの品質」も測定
  → 「バグの検出率」「回帰バグの発生率」を真の目標にする
```

### アンチパターン 2: 改善のための改善 (Shiny Object Syndrome)

```
NG パターン:
  「最新のツールを導入しよう！」
  → 既存のワークフローを壊す
  → チームの学習コストが高い
  → 実際の品質は改善しない
  → 数ヶ月後に別の新しいツールに移行

OK パターン: 問題起点の改善
  1. 「本番障害が月3件ある」← 問題を特定
  2. 「テストカバレッジが40%」← 原因を分析
  3. 「CI にカバレッジゲート追加」← 対策を実施
  4. 効果測定: 障害件数の推移を追跡
  5. 改善確認: 月3件 → 月1件に減少 → 対策継続

  改善の動機は常に「具体的な問題」であるべき。
  「面白そう」「流行っている」は改善の動機としては不適切。
```

### アンチパターン 3: Big Bang 改善

```
NG パターン:
  「来月から全てのプロジェクトで以下を必須にする:
   - カバレッジ 80%
   - 型チェック strict
   - E2E テスト必須
   - セキュリティスキャン必須」

  → 既存プロジェクトの CI が全部 Red になる
  → 開発者が CI を無視し始める
  → 「品質ゲートは邪魔」という認識が広まる

OK パターン: 段階的導入
  Phase 1 (2週間): Warning のみ、Blocking なし
  Phase 2 (2週間): Lint + テスト失敗のみ Blocking
  Phase 3 (1ヶ月): カバレッジ 60% を追加
  Phase 4 (1ヶ月): カバレッジ 80% + 型チェック

  各フェーズで:
  - チームのフィードバックを収集
  - 痛みが大きければ閾値を調整
  - 「なぜこのゲートが必要か」を丁寧に説明
```

### アンチパターン 4: レトロスペクティブの形骸化

```
NG パターン:
  毎回同じ流れ:
  Keep: 「特にない」
  Problem: 「忙しかった」
  Try: 「もっと頑張る」
  → 具体的なアクションアイテムなし
  → 次回のレトロでも同じ内容
  → 「レトロは時間の無駄」という認識

OK パターン:
  1. 事前にメトリクスを共有（DORA、品質ダッシュボード）
  2. データに基づく議論（「忙しかった」→「リードタイムが72h」）
  3. 具体的なアクションアイテム（担当者、期限、完了条件）
  4. 前回のTryの結果を必ず確認（達成/未達/繰越）
  5. 効果測定（レトロの効果スコアを追跡）
  6. ファシリテーターをローテーション（マンネリ防止）
```

---

## 9. FAQ

### Q1. CI パイプラインが遅い場合の対策は？

**A.** 5つの対策を優先度順に: (1) テストの並列実行（`pytest-xdist`, GitHub Actions の `matrix` 戦略）。最も効果が大きく、導入コストも低い。(2) Docker レイヤーキャッシュと pip キャッシュ（`actions/cache`）の活用。(3) 変更されたファイルに関連するテストのみ実行（affected test detection）。大規模リポジトリで特に有効。(4) ユニットテストと統合テストのジョブ分離による並列化。(5) Fast Feedback First の原則に基づくステージ設計（Lint は2分以内、ユニットテストは5分以内）。目標は PR のフィードバックまで10分以内。

### Q2. DORA メトリクスの改善が停滞した場合は？

**A.** バリューストリームマッピング（VSM）を実施する。「コード変更からデプロイまで」の各ステップの待ち時間を可視化する。多くの場合、ボトルネックは技術的な問題ではなくプロセスにある: コードレビューの待ち時間、手動テスト、承認プロセス。具体的な対策として、(1) レビュー応答時間の SLA 設定（4時間以内）、(2) PR サイズの制限（400行以下）、(3) 手動テストの自動化、(4) 承認プロセスの簡素化。Team Topologies のコンセプトを参考に、チーム構造自体の見直しも検討する。

### Q3. 品質ゲートを厳しくしすぎて開発速度が落ちていないか？

**A.** 品質ゲートの導入初期は開発速度が一時的に10-20%低下する。しかし Accelerate の研究によれば、2-3ヶ月後には回帰バグの減少・レビューの効率化により、トータルの開発速度はゲート導入前を上回る。ゲートが厳しすぎると感じる場合は、(1) Warning レベルと Blocking レベルを分けて段階的に導入する、(2) 新規コードのみにゲートを適用する（既存コードは免除）、(3) チームのフィードバックを定期的に収集する。重要なのは「品質と速度はトレードオフではない」というマインドセットの共有。

### Q4. メトリクスを導入したが、チームが数字を気にしすぎてしまう

**A.** Goodhart's Law（アンチパターン1）に陥っている可能性がある。対策: (1) メトリクスは「健康診断の数値」であり「成績」ではないことを繰り返し伝える。(2) メトリクスを個人評価には絶対に使わない。チーム単位でのトレンドのみ追跡する。(3) 「なぜこのメトリクスを見るのか」の目的を常に明確にする。(4) 定量的メトリクスと定性的フィードバック（開発者満足度、DX Survey）を組み合わせる。

### Q5. 小さなチーム（3-5名）でも DORA メトリクスは有効か？

**A.** 有効だが、収集方法を簡略化する。小規模チームでは自動収集スクリプトの構築に時間をかけるより、(1) デプロイ頻度は「先週何回デプロイしたか」を手動記録、(2) リードタイムは GitHub の PR マージ時間をスプレッドシートに転記、(3) 変更失敗率は「先週 revert した回数」を記録、から始める。月1回のレトロスペクティブでこれらの数値を振り返り、改善ポイントを議論する。自動化は改善文化が定着してから段階的に導入する。

### Q6. Feature Flag はどのように継続的改善に貢献するか？

**A.** Feature Flag により「デプロイ」と「リリース」を分離できる。これが継続的改善に貢献する理由: (1) デプロイ頻度の向上 -- 未完成機能もフラグで無効化してデプロイできるため、小さな変更を頻繁にデプロイできる。(2) 変更失敗率の低下 -- 問題が発生したらフラグを無効化するだけで即座にロールバック。(3) リードタイムの短縮 -- ブランチの長寿命化を防ぎ、コンフリクトを減らす。(4) 実験的改善 -- A/B テストやカナリアリリースで新機能の効果を定量的に検証できる。注意点として、使い終わったフラグの削除を怠ると技術的負債になるため、フラグのライフサイクル管理が重要。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CI/CD パイプライン | Fast Feedback First: Lint → Test → Coverage → Security → Build → Deploy |
| 品質ゲート | PR マージ条件: カバレッジ >= 80%、Lint 全通過、テスト全通過 |
| 段階的導入 | Warning → 基本 Blocking → 品質強化 → フルゲート（3ヶ月かけて） |
| DORA メトリクス | デプロイ頻度、リードタイム、変更失敗率、復旧時間の4指標 |
| パフォーマンスレベル | Elite > High > Medium > Low（スループットと安定性は両立する） |
| PDCA サイクル | Plan → Do → Check → Act の反復。メトリクスで効果を検証 |
| レトロスペクティブ | Keep / Problem / Try + メトリクス + 前回 Try の結果確認 |
| チーム文化 | 心理的安全性 → 標準化 → 継続的改善 → 実験的改善 |
| アンチパターン | メトリクスの目的化、改善のための改善、Big Bang 導入、レトロの形骸化 |

---

## 次に読むべきガイド

- [技術的負債](./03-technical-debt.md) -- 負債の定量化・優先度付け・計画的返済戦略
- [コードスメル](./00-code-smells.md) -- 品質低下の兆候を早期発見するカタログ
- [リファクタリング技法](./01-refactoring-techniques.md) -- 品質改善の具体的なコード変換手法
- [レガシーコード](./02-legacy-code.md) -- 既存システムへの安全な変更技法
- [テスト原則](../01-practices/04-testing-principles.md) -- 品質ゲートの基盤となるテスト設計
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) -- レビューによる品質維持
- [エラーハンドリング](../01-practices/02-error-handling.md) -- 堅牢なエラー処理と障害対応

---

## 参考文献

1. **Accelerate: The Science of Lean Software and DevOps** -- Nicole Forsgren, Jez Humble, Gene Kim (IT Revolution, 2018) -- DORA メトリクスの研究結果。6年間、数万チームの調査に基づくソフトウェアデリバリーパフォーマンスの科学的エビデンス
2. **Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation** -- Jez Humble & David Farley (Addison-Wesley, 2010) -- CI/CD の原典。デプロイメントパイプラインの設計原則と実装パターン
3. **The Phoenix Project** -- Gene Kim, Kevin Behr, George Spafford (IT Revolution, 2013) -- DevOps を物語形式で解説。IT 運用改善のフレームワーク（Three Ways）を提示
4. **Team Topologies: Organizing Business and Technology Teams for Fast Flow** -- Matthew Skelton & Manuel Pais (IT Revolution, 2019) -- チーム構造と開発フロー。Cognitive Load を考慮したチーム設計
5. **The DevOps Handbook, 2nd Edition** -- Gene Kim, Jez Humble, Patrick Debois, John Willis (IT Revolution, 2021) -- DevOps の包括的な実践ガイド。Three Ways（Flow, Feedback, Continual Learning）の詳細な実装方法
6. **Lean Software Development** -- Mary Poppendieck & Tom Poppendieck (Addison-Wesley, 2003) -- トヨタ生産方式のソフトウェア開発への適用。ムダの排除と価値の最大化
7. **Site Reliability Engineering** -- Betsy Beyer et al. (O'Reilly, 2016) -- Google の SRE プラクティス。SLI/SLO/SLA、エラーバジェット、ポストモーテムの実践
