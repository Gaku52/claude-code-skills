# DevOps概要

> ソフトウェアの開発(Dev)と運用(Ops)を統合し、価値提供の速度と信頼性を最大化する文化・プラクティス体系

## この章で学ぶこと

1. DevOpsの文化的背景と5つの原則(CALMS)を理解する
2. DORA メトリクスによるパフォーマンス計測手法を習得する
3. DevOps導入のロードマップとアンチパターンを把握する
4. Three Ways の原則を実務に適用する方法を理解する
5. プラットフォームエンジニアリングとの関連を把握する

---

## 1. DevOpsとは何か

### 1.1 歴史的背景

従来のソフトウェア開発では、開発チームと運用チームが分断された組織構造(サイロ)で働いていた。開発チームは「新機能を早く出す」ことを、運用チームは「システムを安定させる」ことを目標とし、両者の間に構造的な対立が生まれていた。

```
従来のウォーターフォール型リリースサイクル:

企画 → 要件定義 → 設計 → 実装 → テスト → リリース → 運用
 |                                              |
 +--- 開発チームの責任範囲 ---+--- 運用チームの責任範囲 ---+
                              ^
                           "壁越し"
                        (Wall of Confusion)
```

2008年のAgile Conference、2009年のDevOpsdays Ghent で Patrick Debois が「DevOps」という用語を普及させ、開発と運用の壁を壊す運動が始まった。

### 1.2 DevOps の年表

DevOps の進化を年表で整理すると、技術と文化の両面で段階的に成熟してきたことがわかる。

```
年代          出来事                                        意義
────────────────────────────────────────────────────────────────────
2001         Agile Manifesto 発表                         反復型開発の基盤
2006         Amazon "You build it, you run it"            開発者責任の拡大
2008         Agile Conference (Agile Infrastructure)      インフラ自動化の議論開始
2009         DevOpsDays Ghent (Patrick Debois)            DevOps 誕生
2010         "Continuous Delivery" 出版 (Humble & Farley) CI/CD の体系化
2011         "The Phoenix Project" 執筆開始               DevOps 小説による普及
2012         State of DevOps Report 初回                  定量的研究の開始
2013         Docker 1.0 リリース                          コンテナ革命
2014         Kubernetes 発表 (Google)                      コンテナオーケストレーション
2015         SRE 本出版 (Google)                          SRE の体系化
2016         "DevOps Handbook" 出版                       実践ガイドの決定版
2018         "Accelerate" 出版                            DORA メトリクスの学術的根拠
2019         GitOps 概念の普及 (Weaveworks)               宣言的運用の新パラダイム
2020         Platform Engineering の台頭                   内部開発者プラットフォーム
2021         DevOps Handbook 2nd Edition                  最新プラクティスの統合
2022-        AI-Assisted DevOps / AIOps                   機械学習による運用最適化
2024-        Platform as a Product                        プラットフォームの製品化
```

### 1.3 DevOpsの定義

DevOpsは単一のツールや役職ではなく、以下の要素を統合した**文化・運動・プラクティス**である。

```
+-----------------------------------------------------------+
|                     DevOps の全体像                         |
+-----------------------------------------------------------+
|                                                           |
|   文化 (Culture)                                          |
|   +---------------------------------------------------+  |
|   | 自動化 (Automation)                                |  |
|   |   +---------------------------------------------+ |  |
|   |   | 計測 (Measurement)                           | |  |
|   |   |   +---------------------------------------+ | |  |
|   |   |   | 共有 (Sharing)                         | | |  |
|   |   |   |   +-------------------------------+   | | |  |
|   |   |   |   | リーン (Lean)                  |   | | |  |
|   |   |   |   +-------------------------------+   | | |  |
|   |   |   +---------------------------------------+ | |  |
|   |   +---------------------------------------------+ |  |
|   +---------------------------------------------------+  |
+-----------------------------------------------------------+
```

### 1.4 DevOps と関連する概念の関係性

DevOps は単独で存在するのではなく、複数の思想・手法と密接に関連している。

```
                    ┌──────────────────┐
                    │   Lean Thinking  │
                    │  (ムダの排除)     │
                    └────────┬─────────┘
                             │ 影響
                    ┌────────▼─────────┐
                    │   Agile          │
                    │  (反復型開発)     │
                    └────────┬─────────┘
                             │ 拡張
              ┌──────────────▼──────────────┐
              │         DevOps              │
              │  (開発+運用の統合文化)        │
              └──┬──────────┬───────────┬───┘
                 │          │           │
         ┌───────▼──┐ ┌────▼─────┐ ┌───▼──────────┐
         │  SRE     │ │ GitOps   │ │ Platform     │
         │ (信頼性  │ │ (宣言的  │ │ Engineering  │
         │  工学)   │ │  運用)   │ │ (開発者基盤) │
         └──────────┘ └──────────┘ └──────────────┘
```

**Lean Thinking**: トヨタ生産方式に由来。ムダの排除、フロー最適化、継続的改善（カイゼン）の概念を提供。

**Agile**: 反復型開発、顧客フィードバック、変化への適応。DevOps は Agile の原則を運用領域まで拡張したもの。

**SRE (Site Reliability Engineering)**: Google が提唱。DevOps の原則を具体的なプラクティス（エラーバジェット、SLI/SLO/SLA、トイル削減）として実装。

**GitOps**: Git リポジトリを Single Source of Truth として、宣言的にインフラとアプリケーションを管理する手法。

**Platform Engineering**: 内部開発者プラットフォーム（IDP）を構築し、開発者のセルフサービスを実現する組織的アプローチ。

---

## 2. CALMS フレームワーク

DevOpsの成熟度を評価する5つの柱が CALMS である。

### 2.1 Culture（文化）

チーム間の信頼、責任共有、失敗から学ぶ姿勢。

```yaml
# 文化の具体例: Blameless Postmortem テンプレート
postmortem:
  title: "2024-01-15 API サーバーダウン"
  severity: SEV-1
  duration: "45 minutes"
  impact: "全ユーザーの API リクエストが失敗"
  timeline:
    - time: "14:00"
      event: "デプロイ実行"
    - time: "14:05"
      event: "エラーレート急上昇を検知"
    - time: "14:10"
      event: "ロールバック開始"
    - time: "14:45"
      event: "復旧確認"
  root_cause: "未テストの DB マイグレーションスクリプト"
  action_items:
    - "マイグレーションのステージング環境テスト必須化"
    - "カナリーデプロイの導入"
    - "DB マイグレーション専用の CI ジョブを追加"
  blame: "個人を責めない。プロセスを改善する。"
```

#### Blameless Postmortem の実践的テンプレート

```markdown
# インシデント振り返り: [タイトル]

## 基本情報
- **日時**: YYYY-MM-DD HH:MM - HH:MM (JST)
- **影響時間**: XX分
- **重大度**: SEV-1 / SEV-2 / SEV-3
- **対応リーダー**: @担当者名
- **参加者**: @チームメンバー

## 影響範囲
- 影響を受けたサービス:
- 影響を受けたユーザー数:
- SLO への影響:
- 推定損失額:

## タイムライン
| 時刻 | イベント | 担当者 |
|------|---------|--------|
| HH:MM | 最初の異常検知 | Monitoring |
| HH:MM | アラート発報 | PagerDuty |
| HH:MM | 対応開始 | @oncall |
| HH:MM | 根本原因特定 | @engineer |
| HH:MM | 修正適用 | @engineer |
| HH:MM | 復旧確認 | @oncall |

## 根本原因分析（5 Whys）
1. **なぜ** サービスがダウンした？ → DB 接続プールが枯渇した
2. **なぜ** 接続プールが枯渇した？ → 新デプロイでスロークエリが発生
3. **なぜ** スロークエリが発生した？ → N+1 問題のあるコードがデプロイされた
4. **なぜ** N+1 が検出されなかった？ → パフォーマンステストが CI に含まれていない
5. **なぜ** パフォーマンステストがない？ → テスト戦略にパフォーマンス観点が不足

## アクションアイテム
| 優先度 | アクション | 担当 | 期限 |
|--------|----------|------|------|
| P0 | スロークエリの修正 | @eng | 完了 |
| P1 | CI にパフォーマンステスト追加 | @team | 2週間 |
| P2 | DB 接続プール監視アラート追加 | @sre | 1週間 |
| P3 | テスト戦略ドキュメント更新 | @lead | 1ヶ月 |

## 教訓（What went well / What didn't）
### うまくいったこと
- アラートが5分以内に発報された
- ロールバック手順が整備されていた

### 改善が必要なこと
- パフォーマンス回帰の自動検出がない
- ステージング環境のデータ量が本番と乖離
```

#### 心理的安全性の構築

DevOps 文化で最も重要なのは心理的安全性（Psychological Safety）である。Google のProject Aristotle の研究で、チームパフォーマンスの最大の予測因子として特定された。

```yaml
# 心理的安全性チェックリスト
psychological_safety:
  indicators:
    positive:
      - "チームメンバーが失敗を率直に報告できる"
      - "反対意見を安全に表明できる"
      - "わからないことを質問できる"
      - "実験や新しいアプローチを提案できる"
      - "インシデント報告が増加している（隠蔽されていない）"
    negative:
      - "問題を報告した人が責められる"
      - "失敗の隠蔽が発生する"
      - "チーム間の責任の押し付け合い"
      - "「前からそう言っていた」的な後出し批判"
      - "提案が却下されると二度と提案しなくなる"

  practices:
    daily:
      - "スタンドアップで障害・課題を共有"
      - "Slack で #incidents チャンネルに積極投稿"
    weekly:
      - "レトロスペクティブで改善提案"
      - "ペアプログラミング / モブプログラミング"
    monthly:
      - "Blameless Postmortem の実施"
      - "チーム健全性チェック"
    quarterly:
      - "心理的安全性サーベイ"
      - "DevOps 成熟度アセスメント"
```

### 2.2 Automation（自動化）

手作業を排除し、再現性と速度を確保する。

```bash
#!/bin/bash
# 手動デプロイ vs 自動化デプロイの対比

# --- 手動デプロイ（アンチパターン）---
ssh production-server
cd /var/www/app
git pull origin main
npm install
npm run build
pm2 restart app
# 問題: 手順書依存、ヒューマンエラー、再現性なし

# --- 自動化デプロイ ---
# GitHub Actions が PR マージ時に自動実行
# 1. テスト → 2. ビルド → 3. コンテナ作成 → 4. デプロイ → 5. ヘルスチェック
# 全てコードで定義され、バージョン管理される
```

#### 自動化の優先度マトリクス

何から自動化すべきかを判断するフレームワークを示す。

```
自動化の優先度 = (実行頻度 × 手動所要時間 × エラーリスク) / 自動化コスト

          高頻度
            │
    ┌───────┼───────┐
    │ 優先度 │ 最優先 │  ← ここから着手
    │  中   │  高   │
    ├───────┼───────┤
    │ 優先度 │ 優先度 │
    │  低   │  中   │
    └───────┼───────┘
            │
          低頻度
     短時間    長時間
        手動所要時間
```

#### 自動化すべき領域のチェックリスト

```yaml
automation_checklist:
  must_automate:  # 最優先で自動化
    - name: "テスト実行"
      reason: "プッシュ/PR ごとに実行、手動だと1時間/回"
      tools: ["GitHub Actions", "Jest", "pytest"]

    - name: "コードリント・フォーマット"
      reason: "全コミットで実行、一貫性確保"
      tools: ["ESLint", "Prettier", "Black", "pre-commit"]

    - name: "ビルド・アーティファクト生成"
      reason: "リリースごとに実行、再現性必須"
      tools: ["Docker", "GitHub Actions", "Makefile"]

    - name: "デプロイ"
      reason: "日次〜週次で実行、ヒューマンエラー排除"
      tools: ["ArgoCD", "GitHub Actions", "Terraform"]

  should_automate:  # 次に自動化
    - name: "セキュリティスキャン"
      reason: "脆弱性の早期検出"
      tools: ["Trivy", "Snyk", "CodeQL"]

    - name: "依存関係更新"
      reason: "定期的なセキュリティパッチ適用"
      tools: ["Dependabot", "Renovate"]

    - name: "環境プロビジョニング"
      reason: "新環境構築の迅速化"
      tools: ["Terraform", "Pulumi", "CloudFormation"]

    - name: "ドキュメント生成"
      reason: "API ドキュメントの自動更新"
      tools: ["Swagger", "TypeDoc", "Sphinx"]

  consider_automating:  # 余裕があれば自動化
    - name: "リリースノート生成"
      tools: ["release-please", "semantic-release"]

    - name: "パフォーマンスベンチマーク"
      tools: ["k6", "Lighthouse CI"]

    - name: "コスト最適化レポート"
      tools: ["Infracost", "AWS Cost Explorer API"]
```

#### 自動化パイプラインの全体像

```yaml
# .github/workflows/full-pipeline.yml
name: Full CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ステージ1: 静的解析
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run type-check

  # ステージ2: テスト（並列実行）
  test-unit:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:unit -- --coverage
      - uses: actions/upload-artifact@v4
        with:
          name: coverage-unit
          path: coverage/

  test-integration:
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test
          REDIS_URL: redis://localhost:6379

  # ステージ3: セキュリティスキャン
  security:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

  # ステージ4: ビルド
  build:
    runs-on: ubuntu-latest
    needs: [test-unit, test-integration, security]
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ステージ5: デプロイ（main ブランチのみ）
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging
        run: |
          echo "Deploying ${{ github.sha }} to staging..."
          # kubectl set image deployment/app app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to production (canary)
        run: |
          echo "Canary deploying ${{ github.sha }} to production..."
          # 10% のトラフィックを新バージョンに振り分け
```

### 2.3 Lean（リーン）

ムダの排除、小さなバッチ、フロー最適化。

```
バッチサイズとリスクの関係:

リスク
  ^
  |        *
  |       *
  |      *
  |    *
  |  *
  | *
  +*-----------> バッチサイズ
  小              大

小さなバッチ = 小さなリスク = 速いフィードバック
```

#### リーンの7つのムダ（ソフトウェア開発版）

トヨタ生産方式の7つのムダをソフトウェア開発に適用する。

```
製造業のムダ              ソフトウェア開発のムダ           対策
─────────────────────────────────────────────────────────────
1. 作りすぎ         →  不要な機能開発               → MVP、フィーチャーフラグ
2. 手持ち(待ち)     →  承認待ち、レビュー待ち       → 自動承認、非同期レビュー
3. 運搬             →  チーム間のハンドオフ         → クロスファンクショナルチーム
4. 加工のムダ       →  過剰なプロセス・文書化       → 必要十分なドキュメント
5. 在庫             →  未リリースのコード           → 継続的デプロイ、小バッチ
6. 動作のムダ       →  ツール切り替え、環境構築     → 統合開発環境、IDP
7. 不良品           →  バグ、手戻り                 → TDD、CI、自動テスト
```

#### バリューストリームマッピング

開発プロセスのボトルネックを可視化する手法。

```
バリューストリームマップ例: フィーチャーリクエストからリリースまで

                プロセス時間    待ち時間
                ──────────    ──────
要件定義         2日          3日待ち（承認待ち）
    ↓
設計             1日          2日待ち（レビュー待ち）
    ↓
実装             3日          0.5日待ち（PR レビュー待ち）
    ↓
コードレビュー   0.5日        1日待ち（修正待ち）
    ↓
テスト           0.5日        2日待ち（テスト環境待ち）
    ↓
デプロイ         0.1日        5日待ち（リリースウィンドウ待ち）
    ↓
リリース確認     0.2日
────────────────────────────────
合計: プロセス時間 7.3日 / 待ち時間 13.5日
リードタイム: 20.8日
プロセス効率: 7.3 / 20.8 = 35%

改善後の目標:
プロセス時間 5日 / 待ち時間 2日
リードタイム: 7日
プロセス効率: 5 / 7 = 71%
```

### 2.4 Measurement（計測）

改善にはデータが必要。計測しないものは改善できない。

```python
# DORA メトリクス計測の実装例
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class PerformanceLevel(Enum):
    ELITE = "Elite"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class Deployment:
    """デプロイメント情報"""
    id: str
    commit_sha: str
    commit_time: datetime
    deploy_time: datetime
    environment: str
    success: bool
    rollback: bool = False

@dataclass
class Incident:
    """インシデント情報"""
    id: str
    severity: str  # SEV-1, SEV-2, SEV-3
    started_at: datetime
    resolved_at: Optional[datetime]
    caused_by_deployment: Optional[str] = None

class DORAMetrics:
    """DORA メトリクスの計測と評価"""

    def __init__(self, deployments: List[Deployment], incidents: List[Incident]):
        self.deployments = deployments
        self.incidents = incidents

    def deployment_frequency(self, period_days: int = 30) -> Dict:
        """デプロイ頻度: どのくらいの頻度でデプロイするか"""
        prod_deploys = [
            d for d in self.deployments
            if d.environment == "production" and d.success
        ]
        freq = len(prod_deploys) / period_days

        if freq >= 1:
            level = PerformanceLevel.ELITE
            description = "1日複数回"
        elif freq >= 1/7:
            level = PerformanceLevel.HIGH
            description = "週1回〜日1回"
        elif freq >= 1/30:
            level = PerformanceLevel.MEDIUM
            description = "月1回〜週1回"
        else:
            level = PerformanceLevel.LOW
            description = "月1回未満"

        return {
            "metric": "Deployment Frequency",
            "value": f"{len(prod_deploys)} deploys / {period_days} days",
            "frequency_per_day": round(freq, 2),
            "level": level.value,
            "description": description,
        }

    def lead_time_for_changes(self) -> Dict:
        """変更リードタイム: コミットから本番デプロイまでの時間"""
        prod_deploys = [
            d for d in self.deployments
            if d.environment == "production" and d.success
        ]
        if not prod_deploys:
            return {"metric": "Lead Time for Changes", "error": "No deployments"}

        lead_times = [
            (d.deploy_time - d.commit_time).total_seconds()
            for d in prod_deploys
        ]
        median_seconds = sorted(lead_times)[len(lead_times) // 2]
        median_hours = median_seconds / 3600

        if median_hours < 1:
            level = PerformanceLevel.ELITE
            description = "1時間未満"
        elif median_hours < 24:
            level = PerformanceLevel.HIGH
            description = "1日未満"
        elif median_hours < 168:  # 7日
            level = PerformanceLevel.MEDIUM
            description = "1週間未満"
        else:
            level = PerformanceLevel.LOW
            description = "1週間超"

        return {
            "metric": "Lead Time for Changes",
            "median_hours": round(median_hours, 1),
            "level": level.value,
            "description": description,
        }

    def change_failure_rate(self) -> Dict:
        """変更障害率: デプロイの何%が障害を引き起こすか"""
        prod_deploys = [
            d for d in self.deployments
            if d.environment == "production"
        ]
        if not prod_deploys:
            return {"metric": "Change Failure Rate", "error": "No deployments"}

        failed = [d for d in prod_deploys if not d.success or d.rollback]
        rate = len(failed) / len(prod_deploys) * 100

        if rate <= 5:
            level = PerformanceLevel.ELITE
        elif rate <= 15:
            level = PerformanceLevel.HIGH
        elif rate <= 30:
            level = PerformanceLevel.MEDIUM
        else:
            level = PerformanceLevel.LOW

        return {
            "metric": "Change Failure Rate",
            "rate_percent": round(rate, 1),
            "failed_deploys": len(failed),
            "total_deploys": len(prod_deploys),
            "level": level.value,
        }

    def time_to_restore_service(self) -> Dict:
        """サービス復旧時間: 障害発生から復旧までの時間"""
        resolved = [i for i in self.incidents if i.resolved_at]
        if not resolved:
            return {"metric": "Time to Restore Service", "error": "No resolved incidents"}

        restore_times = [
            (i.resolved_at - i.started_at).total_seconds()
            for i in resolved
        ]
        median_seconds = sorted(restore_times)[len(restore_times) // 2]
        median_hours = median_seconds / 3600

        if median_hours < 1:
            level = PerformanceLevel.ELITE
        elif median_hours < 24:
            level = PerformanceLevel.HIGH
        elif median_hours < 168:
            level = PerformanceLevel.MEDIUM
        else:
            level = PerformanceLevel.LOW

        return {
            "metric": "Time to Restore Service",
            "median_hours": round(median_hours, 1),
            "level": level.value,
        }

    def generate_report(self) -> Dict:
        """全メトリクスのレポート生成"""
        return {
            "report_date": datetime.now().isoformat(),
            "metrics": {
                "deployment_frequency": self.deployment_frequency(),
                "lead_time": self.lead_time_for_changes(),
                "change_failure_rate": self.change_failure_rate(),
                "mttr": self.time_to_restore_service(),
            },
            "overall_level": self._calculate_overall_level(),
        }

    def _calculate_overall_level(self) -> str:
        """全体の成熟度レベルを算出"""
        levels = []
        for metric_func in [
            self.deployment_frequency,
            self.lead_time_for_changes,
            self.change_failure_rate,
            self.time_to_restore_service,
        ]:
            result = metric_func()
            if "level" in result:
                levels.append(result["level"])

        level_scores = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
        if not levels:
            return "Unknown"
        avg = sum(level_scores.get(l, 0) for l in levels) / len(levels)
        if avg >= 3.5:
            return "Elite"
        elif avg >= 2.5:
            return "High"
        elif avg >= 1.5:
            return "Medium"
        else:
            return "Low"
```

#### DORA メトリクスダッシュボードの構築例

```python
# Grafana ダッシュボード用の Prometheus メトリクス
from prometheus_client import Counter, Histogram, Gauge

# デプロイ頻度
deployment_counter = Counter(
    'deployments_total',
    'Total number of deployments',
    ['environment', 'status', 'team']
)

# 変更リードタイム
lead_time_histogram = Histogram(
    'deployment_lead_time_seconds',
    'Time from commit to production deployment',
    ['team'],
    buckets=[60, 300, 900, 3600, 14400, 43200, 86400, 604800]
    # 1分, 5分, 15分, 1時間, 4時間, 12時間, 1日, 1週間
)

# 変更障害率
change_failure_counter = Counter(
    'deployment_failures_total',
    'Deployments that caused failures',
    ['environment', 'team', 'failure_type']
)

# サービス復旧時間
restore_time_histogram = Histogram(
    'incident_restore_time_seconds',
    'Time to restore service after incident',
    ['severity', 'team'],
    buckets=[300, 900, 1800, 3600, 14400, 43200, 86400, 604800]
)

# 現在のパフォーマンスレベル（Gauge）
dora_level_gauge = Gauge(
    'dora_performance_level',
    'Current DORA performance level (1=Low, 4=Elite)',
    ['metric', 'team']
)
```

### 2.5 Sharing（共有）

知識、責任、ツールの共有。

```yaml
# 共有のプラクティス例
sharing_practices:
  knowledge:
    - "ADR (Architecture Decision Records)"
    - "Runbook / Playbook"
    - "Tech Radar"
    - "内部テックブログ"
    - "Lunch & Learn セッション"
  responsibility:
    - "You build it, you run it"
    - "SRE のエラーバジェット"
    - "共同オンコール"
    - "ローテーション制度"
  tools:
    - "共通 CI/CD パイプライン"
    - "統一監視ダッシュボード"
    - "ChatOps (Slack + Bot)"
    - "内部開発者ポータル (Backstage)"
```

#### ADR (Architecture Decision Record) テンプレート

```markdown
# ADR-0012: API ゲートウェイに Kong を採用

## ステータス
Accepted (2024-03-15)

## コンテキスト
マイクロサービスアーキテクチャへの移行に伴い、以下の要件を満たす
API ゲートウェイが必要:
- レート制限
- 認証・認可
- ルーティング
- ロードバランシング
- リクエスト/レスポンス変換

## 検討した選択肢
1. **Kong** - Lua ベース、豊富なプラグインエコシステム
2. **AWS API Gateway** - マネージド、AWS ロックイン
3. **Envoy + Istio** - サービスメッシュ統合、学習コスト高
4. **Nginx** - 軽量だがAPI管理機能が限定的

## 決定
Kong を採用する。

## 理由
- オープンソースで、ベンダーロックインを回避
- DB-less モードで GitOps との親和性が高い
- プラグインエコシステムが充実
- チーム内に Lua 経験者がいる
- Kubernetes Ingress Controller として統合可能

## 結果
- Kong を Kubernetes 上にデプロイ
- 設定は Git リポジトリで管理 (DB-less モード)
- カスタムプラグインの開発ガイドラインを策定
```

#### Runbook テンプレート

```markdown
# Runbook: API レスポンスタイム劣化

## アラート条件
- P95 レスポンスタイムが 500ms を超過（5分間継続）
- Grafana アラート: `api_response_time_p95_high`

## 影響
- ユーザー体験の劣化
- タイムアウトによるエラー増加
- SLO 違反のリスク

## 診断手順

### Step 1: 現状確認
```bash
# Grafana ダッシュボードで確認
# URL: https://grafana.example.com/d/api-performance

# メトリクスを直接確認
kubectl top pods -n production
kubectl logs -f deployment/api-server -n production --tail=100
```

### Step 2: 原因の切り分け
```bash
# DB クエリの遅延確認
kubectl exec -it deployment/api-server -- \
  curl -s localhost:9090/metrics | grep db_query_duration

# 外部 API の遅延確認
kubectl exec -it deployment/api-server -- \
  curl -s localhost:9090/metrics | grep external_api_duration

# メモリ・CPU 使用率
kubectl top pods -n production --sort-by=memory
```

### Step 3: 対応
| 原因 | 対応 |
|------|------|
| DB スロークエリ | DB チームにエスカレーション |
| メモリリーク | Pod 再起動 + 調査 Issue 起票 |
| 外部 API 遅延 | サーキットブレーカー有効化 |
| トラフィック急増 | HPA スケールアウト確認 |
| 最近のデプロイ | ロールバック検討 |

### Step 4: エスカレーション
- 30分以内に解決しない場合: チームリーダーに連絡
- SEV-1 に該当する場合: インシデントコマンダーを招集
```

---

## 3. DORA メトリクス詳解

DORA (DevOps Research and Assessment) は Google Cloud が支援する研究プログラムで、ソフトウェアデリバリーのパフォーマンスを4つのメトリクスで定量化する。

### 3.1 四大メトリクス比較表

| メトリクス | Elite | High | Medium | Low |
|---|---|---|---|---|
| デプロイ頻度 | オンデマンド(1日複数回) | 週1〜月1回 | 月1〜半年に1回 | 半年に1回未満 |
| 変更リードタイム | 1時間未満 | 1日〜1週間 | 1週間〜1ヶ月 | 1ヶ月〜半年 |
| 変更障害率 | 0〜5% | 5〜15% | 15〜30% | 30%超 |
| サービス復旧時間 | 1時間未満 | 1日未満 | 1日〜1週間 | 1週間超 |

### 3.2 DORA メトリクスの5つ目: 信頼性

2022年の State of DevOps Report から、5つ目のメトリクスとして「信頼性（Reliability）」が追加された。

```yaml
reliability_metric:
  definition: "ユーザーの期待に対してサービスがどの程度応えているか"
  measurement:
    - "SLO の達成率"
    - "可用性（uptime）"
    - "エラーレート"
    - "レイテンシの一貫性"

  levels:
    elite:
      - "SLO 達成率 99.9% 以上"
      - "エラーバジェットを戦略的に活用"
      - "カオスエンジニアリングを実施"
    high:
      - "SLO 達成率 99.5% 以上"
      - "SLI/SLO が定義・監視されている"
    medium:
      - "SLO 達成率 99% 以上"
      - "基本的な監視は導入済み"
    low:
      - "SLO が未定義"
      - "可用性が不安定"
```

### 3.3 DevOps 成熟度モデル比較

| 段階 | プラクティス | ツール例 | 組織の特徴 |
|---|---|---|---|
| Level 0: 手動 | 手動デプロイ、手動テスト | FTP、手作業SSH | サイロ化した組織 |
| Level 1: 部分自動化 | CI導入、自動テスト一部 | Jenkins、基本的なスクリプト | 開発チーム内で自動化 |
| Level 2: CI/CD | 完全CI/CD、IaC導入 | GitHub Actions、Terraform | クロスファンクショナルチーム |
| Level 3: 継続的改善 | カナリーデプロイ、SLO | ArgoCD、Datadog | プラットフォームチーム |
| Level 4: 最適化 | カオスエンジニアリング、ML-Ops | Chaos Monkey、Feature Flags | 学習する組織 |

### 3.4 メトリクス計測の自動化

DORA メトリクスを GitHub Actions で自動計測する実装例を示す。

```yaml
# .github/workflows/dora-metrics.yml
name: DORA Metrics Collection

on:
  workflow_run:
    workflows: ["Deploy to Production"]
    types: [completed]
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜 9:00 にレポート生成

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全履歴が必要

      - name: Calculate Deployment Frequency
        id: deploy-freq
        run: |
          # 過去30日のデプロイ数をカウント
          DEPLOYS=$(git log --oneline --since="30 days ago" \
            --grep="deploy\|release" --all | wc -l)
          FREQ=$(echo "scale=2; $DEPLOYS / 30" | bc)
          echo "deploys=$DEPLOYS" >> $GITHUB_OUTPUT
          echo "frequency=$FREQ" >> $GITHUB_OUTPUT
          echo "Deployment Frequency: $DEPLOYS deploys in 30 days ($FREQ/day)"

      - name: Calculate Lead Time
        id: lead-time
        run: |
          # 直近のデプロイのリードタイム（コミットからデプロイまで）を計算
          DEPLOY_TIME=$(date +%s)
          LAST_COMMIT=$(git log -1 --format=%ct HEAD)
          LEAD_TIME=$((DEPLOY_TIME - LAST_COMMIT))
          LEAD_TIME_HOURS=$((LEAD_TIME / 3600))
          echo "lead_time_hours=$LEAD_TIME_HOURS" >> $GITHUB_OUTPUT
          echo "Lead Time: ${LEAD_TIME_HOURS} hours"

      - name: Calculate Change Failure Rate
        id: cfr
        run: |
          # 過去30日のデプロイ中、ロールバックが発生した割合
          TOTAL=$(git log --oneline --since="30 days ago" \
            --grep="deploy" --all | wc -l)
          FAILURES=$(git log --oneline --since="30 days ago" \
            --grep="rollback\|hotfix\|revert" --all | wc -l)
          if [ "$TOTAL" -gt 0 ]; then
            CFR=$(echo "scale=1; $FAILURES * 100 / $TOTAL" | bc)
          else
            CFR=0
          fi
          echo "cfr=$CFR" >> $GITHUB_OUTPUT
          echo "Change Failure Rate: ${CFR}%"

      - name: Send to Monitoring
        run: |
          # Prometheus Pushgateway にメトリクスを送信
          cat <<METRICS | curl --data-binary @- \
            http://pushgateway.example.com/metrics/job/dora/team/backend
          # HELP dora_deployment_frequency Deployments per day
          # TYPE dora_deployment_frequency gauge
          dora_deployment_frequency ${{ steps.deploy-freq.outputs.frequency }}
          # HELP dora_lead_time_hours Lead time in hours
          # TYPE dora_lead_time_hours gauge
          dora_lead_time_hours ${{ steps.lead-time.outputs.lead_time_hours }}
          # HELP dora_change_failure_rate Change failure rate percentage
          # TYPE dora_change_failure_rate gauge
          dora_change_failure_rate ${{ steps.cfr.outputs.cfr }}
          METRICS
```

---

## 4. DevOps の無限ループ

```
          Plan → Code → Build → Test
         ↗                           ↘
    Monitor                           Release
         ↖                           ↙
          Operate ← Deploy ← Stage

+--------------------------------------------------+
|              DevOps ∞ ループ                       |
|                                                    |
|   ┌─────────── Dev ──────────┐                    |
|   │  Plan → Code → Build → Test                  |
|   │  ↑                       │                    |
|   │  │    ┌── Feedback ──┐   ↓                    |
|   │  │    │              │  Release                |
|   │  │    │              │   │                    |
|   │  │    ↓              │   ↓                    |
|   │  Monitor ← Operate ← Deploy                  |
|   │  └─────────── Ops ──────────┘                 |
|   └──────────────────────────────┘                |
+--------------------------------------------------+
```

### 4.1 各フェーズの詳細と推奨ツール

```yaml
devops_loop_phases:
  plan:
    description: "要件定義、バックログ管理、スプリント計画"
    tools:
      - name: "Jira"
        use_case: "大規模プロジェクトのバックログ管理"
      - name: "GitHub Issues + Projects"
        use_case: "軽量なタスク管理、開発ワークフロー統合"
      - name: "Linear"
        use_case: "モダンなプロジェクト管理、高速UI"
    practices:
      - "ユーザーストーリーマッピング"
      - "OKR によるゴール設定"
      - "技術的負債のバックログ化"

  code:
    description: "開発、コードレビュー、ブランチ管理"
    tools:
      - name: "GitHub"
        use_case: "ソースコード管理、PR レビュー"
      - name: "VS Code / Cursor"
        use_case: "統合開発環境、AI アシスタント"
      - name: "pre-commit"
        use_case: "コミット前の品質チェック"
    practices:
      - "Trunk-Based Development"
      - "ペアプログラミング / モブプログラミング"
      - "Feature Flags による開発"

  build:
    description: "コンパイル、パッケージング、コンテナイメージ作成"
    tools:
      - name: "Docker"
        use_case: "コンテナイメージのビルド"
      - name: "GitHub Actions"
        use_case: "CI パイプラインの実行"
      - name: "Buildpack"
        use_case: "Dockerfile 不要のビルド"
    practices:
      - "マルチステージビルド"
      - "ビルドキャッシュの活用"
      - "再現可能なビルド"

  test:
    description: "自動テスト、品質ゲート、セキュリティスキャン"
    tools:
      - name: "Jest / Vitest"
        use_case: "ユニットテスト"
      - name: "Playwright / Cypress"
        use_case: "E2E テスト"
      - name: "Trivy / Snyk"
        use_case: "セキュリティスキャン"
    practices:
      - "テストピラミッド（Unit > Integration > E2E）"
      - "品質ゲートの定義"
      - "Shift-Left テスティング"

  release:
    description: "バージョニング、リリースノート、アーティファクト管理"
    tools:
      - name: "semantic-release"
        use_case: "自動バージョニング"
      - name: "release-please"
        use_case: "リリース PR の自動作成"
      - name: "GitHub Releases"
        use_case: "リリースノートの管理"
    practices:
      - "Semantic Versioning (SemVer)"
      - "Conventional Commits"
      - "リリースブランチ戦略"

  deploy:
    description: "本番デプロイ、カナリー、ブルーグリーン"
    tools:
      - name: "ArgoCD"
        use_case: "Kubernetes GitOps デプロイ"
      - name: "Helm"
        use_case: "Kubernetes パッケージ管理"
      - name: "Terraform"
        use_case: "インフラストラクチャプロビジョニング"
    practices:
      - "カナリーデプロイ"
      - "ブルーグリーンデプロイ"
      - "ローリングアップデート"

  operate:
    description: "インシデント管理、スケーリング、設定管理"
    tools:
      - name: "PagerDuty"
        use_case: "オンコール管理、インシデント対応"
      - name: "Kubernetes"
        use_case: "コンテナオーケストレーション"
      - name: "Vault"
        use_case: "シークレット管理"
    practices:
      - "インシデント対応プロセス"
      - "自動スケーリング"
      - "カオスエンジニアリング"

  monitor:
    description: "メトリクス収集、ログ分析、トレーシング"
    tools:
      - name: "Datadog / Grafana"
        use_case: "メトリクスダッシュボード"
      - name: "OpenTelemetry"
        use_case: "分散トレーシング"
      - name: "Loki / Elasticsearch"
        use_case: "ログ集約・検索"
    practices:
      - "SLI / SLO / SLA の定義"
      - "アラートの階層化"
      - "ダッシュボードの標準化"
```

---

## 5. Three Ways（3つの道）

Gene Kim の「The Phoenix Project」で提唱された DevOps の基本原則。

```
第1の道: フロー (Systems Thinking)
  Dev → Ops → Customer
  左から右へのフローを最大化

第2の道: フィードバック
  Dev ← Ops ← Customer
  右から左へのフィードバックを最大化

第3の道: 継続的学習
  +---→ 実験 → 失敗 → 学習 ---+
  |                             |
  +-------- 繰り返し ←---------+
  リスクを取り、反復し、習熟する
```

### 5.1 第1の道: フローの原則

システム全体のパフォーマンスを最適化し、局所最適化を避ける。

```yaml
first_way_practices:
  principles:
    - "WIP（仕掛品）の制限"
    - "バッチサイズの縮小"
    - "ハンドオフの削減"
    - "制約の特定と解消"
    - "ムダの排除"

  implementation:
    wip_limits:
      description: "同時進行タスク数の制限"
      example:
        development: 3  # 開発中の機能は最大3つ
        code_review: 5  # レビュー待ちは最大5つ
        testing: 3      # テスト中は最大3つ
        deployment: 1   # デプロイは一度に1つ

    batch_size_reduction:
      before: "3ヶ月分の機能を一度にリリース"
      after: "毎日小さな変更をリリース"
      benefit: "リスク低減、フィードバック速度向上"

    constraint_identification:
      tool: "バリューストリームマッピング"
      steps:
        - "現状のプロセスを可視化"
        - "待ち時間・ボトルネックを特定"
        - "制約を解消する施策を実行"
        - "改善効果を計測"
```

### 5.2 第2の道: フィードバックの原則

右から左への迅速なフィードバックループを構築する。

```yaml
second_way_practices:
  feedback_loops:
    immediate:  # 秒〜分
      - "IDE のリアルタイムエラー表示"
      - "pre-commit フック"
      - "ユニットテスト"
    short:  # 分〜時間
      - "CI パイプラインの結果"
      - "コードレビューのコメント"
      - "セキュリティスキャン結果"
    medium:  # 時間〜日
      - "ステージング環境でのテスト結果"
      - "パフォーマンステスト結果"
      - "ユーザーフィードバック（ベータ）"
    long:  # 日〜週
      - "本番環境のメトリクス"
      - "ユーザーの行動分析"
      - "A/B テスト結果"

  implementation:
    telemetry:
      - "全レイヤーにメトリクス・ログ・トレースを埋め込む"
      - "異常検知による自動アラート"
    peer_review:
      - "全変更に対するコードレビュー"
      - "ペアプログラミングの導入"
    swarm:
      - "問題発生時にチーム全体で対応"
      - "知識の属人化を防ぐ"
```

### 5.3 第3の道: 継続的学習と実験の原則

組織全体で学習し、改善し続ける文化を構築する。

```yaml
third_way_practices:
  experimentation:
    - "20% タイムの確保（Google 方式）"
    - "ハッカソンの定期開催"
    - "新技術の PoC を推奨"
    - "失敗を許容する文化"

  learning_from_failure:
    - "Blameless Postmortem の徹底"
    - "Game Day（障害訓練）の実施"
    - "カオスエンジニアリング"

  knowledge_sharing:
    - "社内テックトーク"
    - "ドキュメンテーション文化"
    - "メンタリング制度"
    - "カンファレンス参加の奨励"

  mastery:
    - "反復による習熟"
    - "日々の改善（カイゼン）"
    - "技術ブログの執筆"
```

---

## 6. プラットフォームエンジニアリング

DevOps の進化形として、プラットフォームエンジニアリングが注目されている。

### 6.1 背景

DevOps の「You build it, you run it」原則により、開発者の認知負荷が増大した。インフラ、CI/CD、モニタリング、セキュリティなど、開発者が管理すべき領域が広がりすぎた結果、本来の開発業務に集中できなくなった。

```
DevOps の課題（認知負荷の増大）:

開発者が管理すべきもの:
┌──────────────────────────────────────────────┐
│ アプリケーションコード                          │
│ テスト                                         │
│ CI/CD パイプライン                              │
│ コンテナ (Dockerfile, Kubernetes manifests)     │
│ インフラ (Terraform, CloudFormation)            │
│ モニタリング (Datadog, Grafana)                 │
│ セキュリティ (脆弱性スキャン, シークレット管理) │
│ コスト管理                                      │
│ コンプライアンス                                │
└──────────────────────────────────────────────┘
         ↓ 認知負荷が過大
         ↓
プラットフォームエンジニアリングで解決
```

### 6.2 内部開発者プラットフォーム（IDP）

```yaml
internal_developer_platform:
  definition: >
    開発者がセルフサービスでインフラやツールを利用できる
    統合プラットフォーム。開発者体験（DX）の向上が目的。

  components:
    developer_portal:
      tool: "Backstage (Spotify OSS)"
      features:
        - "サービスカタログ"
        - "テンプレートからの新規サービス作成"
        - "API ドキュメント集約"
        - "チーム情報・オーナーシップ管理"

    infrastructure_abstraction:
      tools: ["Crossplane", "Terraform Cloud", "Pulumi"]
      features:
        - "セルフサービスのインフラプロビジョニング"
        - "ガードレール付きの環境構築"
        - "コスト可視化"

    ci_cd_platform:
      tools: ["GitHub Actions", "Dagger", "Tekton"]
      features:
        - "標準化されたパイプラインテンプレート"
        - "自動セキュリティスキャン"
        - "デプロイ承認フロー"

    observability_platform:
      tools: ["Grafana Stack", "Datadog", "OpenTelemetry"]
      features:
        - "自動インストルメンテーション"
        - "標準ダッシュボード"
        - "アラートルーティング"

  golden_paths:
    description: >
      開発者が推奨されるパスに沿って効率的に作業できるテンプレート。
      強制ではなく推奨であることが重要。
    examples:
      - "新規 API サービスのテンプレート"
      - "フロントエンドアプリのテンプレート"
      - "データパイプラインのテンプレート"
```

---

## 7. アンチパターン

### アンチパターン1: ツール先行型 DevOps

```
悪い例:
  "Kubernetes を導入したから DevOps できている"
  "CI/CD ツールを入れたから DevOps 完了"

問題:
  - 文化の変革なしにツールだけ導入
  - チーム間のサイロが残ったまま
  - ツールが複雑になり逆に生産性低下

改善:
  1. まず文化・プロセスの改善に着手
  2. 小さな自動化から始める
  3. ツールは問題解決の手段として選定
```

### アンチパターン2: DevOps チーム症候群

```
悪い例:
  組織図に「DevOps チーム」を新設し、
  開発チームと運用チームの間に第3のサイロを作る

  Dev Team  →  DevOps Team  →  Ops Team
              (新たな壁)      (依然として壁)

改善:
  - DevOps はチーム名ではなく文化
  - プラットフォームエンジニアリングチームとして
    開発者体験(DX)を向上させるツール・基盤を提供
  - "You build it, you run it" の原則
```

### アンチパターン3: 計測なき改善

```
悪い例:
  "感覚的に速くなった気がする"
  "たぶんデプロイ頻度が上がった"

問題:
  - 改善の根拠がない
  - 投資対効果を説明できない
  - 後退しても気づけない

改善:
  1. DORA メトリクスのベースラインを計測
  2. 改善施策ごとに効果を定量評価
  3. ダッシュボードで可視化・共有
```

### アンチパターン4: 自動化の罠

```
悪い例:
  - 全てを一度に自動化しようとする
  - テストなしでデプロイを自動化
  - 自動化スクリプトのメンテナンス不足

問題:
  - 自動化されたプロセスが壊れても気づかない
  - 「自動化されている＝安全」という誤解
  - メンテナンスコストが手動コストを上回る

改善:
  1. 段階的に自動化（まずテスト → ビルド → デプロイ）
  2. 自動化コード自体もテスト・レビュー対象
  3. 定期的な自動化パイプラインの健全性チェック
```

### アンチパターン5: モノリシックなCI/CDパイプライン

```
悪い例:
  全サービスが1つの巨大な CI/CD パイプラインを共有
  1つのサービスの変更で全サービスのテスト・デプロイが実行

問題:
  - パイプライン実行時間が膨大
  - 他チームの変更で自チームのデプロイがブロック
  - 障害の影響範囲が全サービスに波及

改善:
  - サービスごとに独立したパイプライン
  - 共通部分はReusable Workflowsで共有
  - 変更検知による選択的実行
```

---

## 8. DevOps 導入ロードマップ

### 8.1 フェーズ別導入計画

```
Phase 0: 現状把握（2週間）
├── バリューストリームマッピング
├── DORA メトリクスのベースライン計測
├── チームアンケート（文化・課題の把握）
└── ツール棚卸し

Phase 1: Quick Wins（1-3ヶ月）
├── バージョン管理の統一（Git）
├── 基本的な CI の導入（lint + unit test）
├── 自動ビルドの構築
├── チャットツールの統一（Slack/Teams）
└── 目標: 開発者が「便利になった」と実感

Phase 2: CI/CD 確立（3-6ヶ月）
├── テスト自動化の拡充（integration, E2E）
├── CD パイプラインの構築
├── Infrastructure as Code の導入
├── 監視・アラートの基盤構築
└── 目標: 手動作業を50%削減

Phase 3: 継続的改善（6-12ヶ月）
├── カナリーデプロイの導入
├── SLI/SLO の定義
├── Blameless Postmortem の制度化
├── プラットフォームチームの発足
└── 目標: DORA メトリクスの High レベル達成

Phase 4: 最適化（12ヶ月〜）
├── カオスエンジニアリングの導入
├── フィーチャーフラグの活用
├── 内部開発者プラットフォーム構築
├── AIOps の検討
└── 目標: DORA メトリクスの Elite レベル
```

### 8.2 組織変革のための実践的アドバイス

```yaml
organizational_transformation:
  executive_support:
    why: "トップダウンのサポートなしに文化変革は不可能"
    how:
      - "経営層への DORA レポートの定期報告"
      - "DevOps による事業貢献の定量化"
      - "先行事例（Netflix, Amazon）の共有"

  start_small:
    why: "大規模な変革は失敗しやすい"
    how:
      - "パイロットチームを選定（意欲の高いチーム）"
      - "小さな成功を積み重ねる"
      - "成功事例を社内に展開"

  measure_and_share:
    why: "計測なくして改善なし"
    how:
      - "DORA メトリクスの定期計測"
      - "改善施策の Before/After を可視化"
      - "ダッシュボードを全員に公開"

  invest_in_people:
    why: "ツールよりも人材への投資が重要"
    how:
      - "トレーニング・学習時間の確保"
      - "カンファレンス参加の支援"
      - "社内勉強会の開催"
      - "外部コミュニティとの交流"
```

---

## 9. FAQ

### Q1: DevOps エンジニアという職種は正しいのか？

DevOps は本来文化・プラクティスであり職種名ではない。しかし市場では「DevOps エンジニア」という職種が定着している。実態としてはインフラ自動化、CI/CD 構築、クラウド運用を担当するエンジニアを指すことが多い。最近では「Platform Engineer」「SRE (Site Reliability Engineer)」というより正確な名称が普及しつつある。

### Q2: DevOps と SRE の違いは何か？

DevOps は文化・原則のフレームワークであり、SRE は Google が生み出した DevOps の具体的実装の1つ。SRE はエラーバジェット、SLI/SLO/SLA、トイル削減など、より体系化されたプラクティスを持つ。DevOps が「何を(What)」すべきかを示し、SRE が「どのように(How)」実現するかを示すと理解できる。

```
DevOps vs SRE の比較:

観点           DevOps                     SRE
──────────────────────────────────────────────────
起源           コミュニティ               Google (2003年〜)
性質           文化・原則                 具体的プラクティス
焦点           開発と運用の統合           信頼性の工学的アプローチ
失敗への対応   Blameless Culture          エラーバジェット
作業の分類     自動化推進                 トイル削減 (50%ルール)
サービス品質   なんとなく良くする         SLI/SLO/SLA で定量管理
オンコール     共同責任                   エンジニアが担当
スケーリング   チームごとに異なる         50%ルールで持続可能に
```

### Q3: 小さなチームでも DevOps は必要か？

必要である。むしろ小さなチームこそ自動化の恩恵が大きい。5人のチームで週2時間の手動デプロイ作業があれば、年間520人時の損失になる。CI/CD を1日で構築すれば、その投資は1週間で回収できる。小さく始めて段階的に成熟度を上げるアプローチが推奨される。

```yaml
# 小規模チーム向け DevOps スターターキット
small_team_devops:
  day_1:
    - "GitHub リポジトリ作成"
    - "ブランチ保護ルール設定"
    - "基本的な CI ワークフロー（lint + test）"

  week_1:
    - "Docker 化"
    - "自動デプロイパイプライン"
    - "基本的な監視（uptime + エラーレート）"

  month_1:
    - "テストカバレッジの向上"
    - "ステージング環境の構築"
    - "インシデント対応プロセスの策定"

  tools_recommendation:
    vcs: "GitHub (Free tier)"
    ci_cd: "GitHub Actions (2,000分/月 無料)"
    monitoring: "Grafana Cloud (Free tier)"
    alerting: "PagerDuty (Free for up to 5 users)"
    infrastructure: "Terraform Cloud (Free tier)"
```

### Q4: DevOps導入にどのくらいの期間がかかるか？

組織の規模と現状の成熟度による。基本的なCI/CDパイプラインは数日で構築できるが、文化の変革には6ヶ月〜2年を要する。DORA メトリクスの「Elite」レベルに到達するには、継続的な改善の積み重ねが必要。まずは3ヶ月で「Quick Win」を出し、組織の信頼を得ることが重要。

### Q5: DevOps の ROI をどう説明するか？

```python
# DevOps ROI 計算の例
def calculate_devops_roi():
    """DevOps 投資対効果の試算"""

    # 現状コスト（年間）
    current_costs = {
        "manual_deploy_hours": 5 * 52,         # 週5時間 × 52週 = 260時間
        "incident_response_hours": 10 * 52,     # 週10時間 × 52週 = 520時間
        "environment_setup_hours": 8 * 12,      # 月8時間 × 12ヶ月 = 96時間
        "manual_testing_hours": 20 * 52,        # 週20時間 × 52週 = 1040時間
        "hourly_engineer_cost": 5000,           # エンジニア時給(円)
        "downtime_cost_per_hour": 100000,       # ダウンタイムコスト(円/時間)
        "avg_downtime_hours_year": 48,          # 年間ダウンタイム
    }

    total_manual_hours = (
        current_costs["manual_deploy_hours"]
        + current_costs["incident_response_hours"]
        + current_costs["environment_setup_hours"]
        + current_costs["manual_testing_hours"]
    )

    current_labor_cost = total_manual_hours * current_costs["hourly_engineer_cost"]
    current_downtime_cost = (
        current_costs["avg_downtime_hours_year"]
        * current_costs["downtime_cost_per_hour"]
    )
    total_current_cost = current_labor_cost + current_downtime_cost

    # DevOps 導入後の予測
    devops_improvements = {
        "manual_work_reduction": 0.70,     # 手動作業70%削減
        "downtime_reduction": 0.60,        # ダウンタイム60%削減
        "deployment_frequency": 10,        # デプロイ頻度10倍
        "lead_time_reduction": 0.80,       # リードタイム80%短縮
    }

    investment = {
        "tooling_cost": 2000000,           # ツール費用(年間)
        "training_cost": 1000000,          # トレーニング費用
        "initial_setup_hours": 500,        # 初期構築工数
    }

    # ROI 計算
    savings_labor = current_labor_cost * devops_improvements["manual_work_reduction"]
    savings_downtime = current_downtime_cost * devops_improvements["downtime_reduction"]
    total_savings = savings_labor + savings_downtime

    total_investment = (
        investment["tooling_cost"]
        + investment["training_cost"]
        + investment["initial_setup_hours"] * current_costs["hourly_engineer_cost"]
    )

    roi = (total_savings - total_investment) / total_investment * 100

    return {
        "current_annual_cost": f"{total_current_cost:,}円",
        "annual_savings": f"{total_savings:,}円",
        "total_investment": f"{total_investment:,}円",
        "roi_percent": f"{roi:.1f}%",
        "payback_months": round(total_investment / (total_savings / 12), 1),
    }
```

### Q6: DevOps とアジャイルの関係は？

アジャイルは開発プロセスの改善に焦点を当て、DevOps はその原則をデリバリーと運用まで拡張したもの。アジャイルなしに DevOps は成立しないが、アジャイルだけでは不十分。両者は補完関係にある。

```
Agile:  要件 → 設計 → 実装 → テスト  [反復]
                 ↓ 拡張
DevOps: 要件 → 設計 → 実装 → テスト → デプロイ → 運用 → 監視  [反復]
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| DevOps の本質 | 文化・プラクティス・ツールの統合体 |
| CALMS | Culture, Automation, Lean, Measurement, Sharing |
| DORA メトリクス | デプロイ頻度、変更リードタイム、変更障害率、復旧時間（+信頼性） |
| Three Ways | フロー、フィードバック、継続的学習 |
| プラットフォームエンジニアリング | 認知負荷を軽減し開発者体験を向上させる進化形 |
| 最重要原則 | ツールより文化、計測して改善、小さく始める |
| アンチパターン | ツール先行、DevOpsチーム症候群、計測なき改善、自動化の罠 |

---

## 次に読むべきガイド

- [CI/CD概念](./01-ci-cd-concepts.md) -- DevOps の中核プラクティスであるCI/CDを深掘り
- [Infrastructure as Code](./02-infrastructure-as-code.md) -- インフラ自動化の具体的手法
- [GitOps](./03-gitops.md) -- 宣言的なインフラ・アプリケーション管理
- [オブザーバビリティ](../03-monitoring/00-observability.md) -- 計測・監視の実践

---

## 参考文献

1. Gene Kim, Jez Humble, Patrick Debois, John Willis. *The DevOps Handbook*, 2nd Edition. IT Revolution Press, 2021.
2. Nicole Forsgren, Jez Humble, Gene Kim. *Accelerate: The Science of Lean Software and DevOps*. IT Revolution Press, 2018.
3. Google Cloud. "DORA | DevOps Research and Assessment." https://dora.dev/
4. Gene Kim. *The Phoenix Project: A Novel about IT, DevOps, and Helping Your Business Win*. IT Revolution Press, 2013.
5. Atlassian. "DevOps: Breaking the Development-Operations barrier." https://www.atlassian.com/devops
6. Team Topologies. Matthew Skelton, Manuel Pais. IT Revolution Press, 2019.
7. Backstage. "An open platform for building developer portals." https://backstage.io/
8. Google. "Site Reliability Engineering." https://sre.google/
9. CNCF. "Cloud Native Landscape." https://landscape.cncf.io/
10. State of DevOps Report 2023. DORA / Google Cloud.
