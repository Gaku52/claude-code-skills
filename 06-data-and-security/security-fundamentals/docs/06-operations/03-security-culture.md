# セキュリティ文化

> DevSecOps による開発プロセスへのセキュリティ統合、バグバウンティプログラムの運用、組織全体のセキュリティ意識を向上させるための実践ガイド

## この章で学ぶこと

1. **DevSecOps** — 開発・運用・セキュリティを統合した組織体制とプロセス
2. **脅威モデリング** — 設計段階からリスクを体系的に特定する手法
3. **バグバウンティ** — 外部研究者によるセキュリティテストの仕組みと運用
4. **セキュリティ教育と意識向上** — 組織全体でセキュリティを自分事にする文化の構築
5. **セキュリティメトリクスとガバナンス** — 定量的にセキュリティ状態を把握・改善する仕組み

---

## 1. DevSecOps

### DevOps から DevSecOps へ

```
DevOps:
  Plan → Code → Build → Test → Release → Deploy → Operate → Monitor
                                                       |
                                                  セキュリティは
                                                  ここだけ (遅い)

DevSecOps:
  Plan → Code → Build → Test → Release → Deploy → Operate → Monitor
    |      |       |       |       |        |         |         |
   脅威   SAST   SCA    DAST    署名    IaC     ランタイム  SIEM
  モデリング コードレビュー Trivy  ZAP    検証    スキャン  保護    異常検知
```

従来の開発プロセスでは、セキュリティテストはリリース直前のゲートとして実施されていた。この「Shift Left」以前のアプローチには根本的な問題がある。リリース直前に発見された脆弱性は修正コストが高く、スケジュールへの影響も大きい。DevSecOps はセキュリティを開発ライフサイクルの全段階に組み込むことで、脆弱性を早期に発見し、修正コストを劇的に低減する。

### DevSecOps パイプラインの実装

```yaml
# .github/workflows/devsecops-pipeline.yml
name: DevSecOps Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # Phase 1: 静的解析 (SAST)
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Semgrep SAST Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/owasp-top-ten
            p/javascript
            p/typescript
            p/react
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}

      - name: CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          languages: javascript, typescript

      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif

  # Phase 2: ソフトウェア構成分析 (SCA)
  sca:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trivy Vulnerability Scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          scan-ref: '.'
          severity: CRITICAL,HIGH
          exit-code: '1'
          format: sarif
          output: trivy-results.sarif

      - name: npm audit
        run: npm audit --audit-level=high

      - name: License Check
        run: npx license-checker --failOn "GPL-3.0;AGPL-3.0"

  # Phase 3: シークレット検出
  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Gitleaks Secret Detection
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # Phase 4: コンテナスキャン
  container-scan:
    runs-on: ubuntu-latest
    needs: [sast, sca, secrets-scan]
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Image
        run: docker build -t app:${{ github.sha }} .

      - name: Trivy Container Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:${{ github.sha }}
          severity: CRITICAL,HIGH
          exit-code: '1'

      - name: Dockle Lint
        uses: erzz/dockle-action@v1
        with:
          image: app:${{ github.sha }}
          exit-code: '1'

  # Phase 5: DAST (動的解析)
  dast:
    runs-on: ubuntu-latest
    needs: container-scan
    steps:
      - name: Deploy to staging
        run: ./deploy-staging.sh

      - name: OWASP ZAP Full Scan
        uses: zaproxy/action-full-scan@v0.8.0
        with:
          target: https://staging.example.com
          rules_file_name: zap-rules.tsv
          cmd_options: '-a -j'

      - name: Nuclei Scan
        run: |
          nuclei -u https://staging.example.com \
            -t cves/ -t vulnerabilities/ \
            -severity critical,high \
            -o nuclei-results.txt

  # Phase 6: IaC セキュリティ
  iac-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Checkov IaC Scan
        uses: bridgecrewio/checkov-action@master
        with:
          directory: infrastructure/
          framework: terraform,cloudformation
          soft_fail: false

      - name: tfsec Terraform Scan
        uses: aquasecurity/tfsec-action@v1.0.0
        with:
          working_directory: infrastructure/terraform/
```

### DevSecOps の成熟度モデル

```
+----------------------------------------------------------+
|          DevSecOps 成熟度モデル                             |
|----------------------------------------------------------|
|                                                          |
|  Level 1: 初期 (Ad Hoc)                                  |
|  +-- セキュリティは専門チームの責任                         |
|  +-- 手動のペネトレーションテスト (年次)                    |
|  +-- リリース前のゲート型レビュー                          |
|  +-- ツール導入なし、属人的な判断                          |
|                                                          |
|  Level 2: 管理 (Managed)                                  |
|  +-- CI/CD にセキュリティテストを統合                      |
|  +-- SAST/SCA の自動実行                                 |
|  +-- セキュリティチャンピオンの任命                         |
|  +-- 脆弱性管理プロセスの文書化                            |
|                                                          |
|  Level 3: 定義 (Defined)                                  |
|  +-- 脅威モデリングが設計プロセスの一部                     |
|  +-- セキュリティ要件が User Story に含まれる              |
|  +-- 全開発者がセキュアコーディング研修を修了               |
|  +-- セキュリティゲートの基準が明確に定義                   |
|                                                          |
|  Level 4: 測定 (Measured)                                 |
|  +-- セキュリティメトリクスの継続測定                       |
|  +-- 脆弱性の平均修正時間 (MTTR) を追跡                   |
|  +-- セキュリティ債務の可視化と管理                         |
|  +-- ダッシュボードによるリアルタイム可視化                  |
|                                                          |
|  Level 5: 最適化 (Optimized)                              |
|  +-- セキュリティが全員の責任として浸透                     |
|  +-- 自動修復・自動封じ込め                               |
|  +-- 継続的な改善サイクルの確立                            |
|  +-- AI/ML による脅威検知の高度化                          |
+----------------------------------------------------------+
```

### 成熟度評価チェックリスト

```python
# DevSecOps 成熟度自動評価ツール
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class MaturityLevel(IntEnum):
    AD_HOC = 1
    MANAGED = 2
    DEFINED = 3
    MEASURED = 4
    OPTIMIZED = 5


@dataclass
class MaturityAssessment:
    """DevSecOps 成熟度評価"""

    # Level 2 criteria
    ci_cd_sast_enabled: bool = False
    ci_cd_sca_enabled: bool = False
    security_champions_appointed: bool = False
    vulnerability_process_documented: bool = False

    # Level 3 criteria
    threat_modeling_in_design: bool = False
    security_in_user_stories: bool = False
    secure_coding_training_complete: bool = False
    security_gate_criteria_defined: bool = False

    # Level 4 criteria
    metrics_continuously_tracked: bool = False
    mttr_tracked: bool = False
    security_debt_visible: bool = False
    dashboard_exists: bool = False

    # Level 5 criteria
    auto_remediation: bool = False
    security_as_everyones_job: bool = False
    continuous_improvement_cycle: bool = False
    ai_ml_threat_detection: bool = False

    def calculate_level(self) -> MaturityLevel:
        """現在の成熟度レベルを計算"""
        level_2_criteria = [
            self.ci_cd_sast_enabled,
            self.ci_cd_sca_enabled,
            self.security_champions_appointed,
            self.vulnerability_process_documented,
        ]
        level_3_criteria = [
            self.threat_modeling_in_design,
            self.security_in_user_stories,
            self.secure_coding_training_complete,
            self.security_gate_criteria_defined,
        ]
        level_4_criteria = [
            self.metrics_continuously_tracked,
            self.mttr_tracked,
            self.security_debt_visible,
            self.dashboard_exists,
        ]
        level_5_criteria = [
            self.auto_remediation,
            self.security_as_everyones_job,
            self.continuous_improvement_cycle,
            self.ai_ml_threat_detection,
        ]

        def level_met(criteria: list[bool], threshold: float = 0.75) -> bool:
            return sum(criteria) / len(criteria) >= threshold

        if level_met(level_5_criteria):
            return MaturityLevel.OPTIMIZED
        if level_met(level_4_criteria):
            return MaturityLevel.MEASURED
        if level_met(level_3_criteria):
            return MaturityLevel.DEFINED
        if level_met(level_2_criteria):
            return MaturityLevel.MANAGED
        return MaturityLevel.AD_HOC

    def generate_roadmap(self) -> list[str]:
        """次のレベルへのロードマップを生成"""
        current = self.calculate_level()
        recommendations = []

        if current < MaturityLevel.MANAGED:
            if not self.ci_cd_sast_enabled:
                recommendations.append(
                    "CI/CD に Semgrep (SAST) を導入する"
                )
            if not self.ci_cd_sca_enabled:
                recommendations.append(
                    "CI/CD に Trivy (SCA) を導入する"
                )
            if not self.security_champions_appointed:
                recommendations.append(
                    "各チームからセキュリティチャンピオンを1名任命する"
                )
        elif current < MaturityLevel.DEFINED:
            if not self.threat_modeling_in_design:
                recommendations.append(
                    "設計フェーズで STRIDE 脅威モデリングを実施する"
                )
            if not self.secure_coding_training_complete:
                recommendations.append(
                    "全開発者にセキュアコーディング研修を受講させる"
                )
        elif current < MaturityLevel.MEASURED:
            if not self.metrics_continuously_tracked:
                recommendations.append(
                    "セキュリティメトリクスの自動収集を設定する"
                )
            if not self.dashboard_exists:
                recommendations.append(
                    "セキュリティダッシュボードを構築する"
                )

        return recommendations
```

### セキュリティチャンピオン制度

```
+----------------------------------------------------------+
|           セキュリティチャンピオン制度                       |
|----------------------------------------------------------|
|                                                          |
|  セキュリティチーム (中央)                                  |
|  +-- ポリシー策定                                        |
|  +-- ツール選定・運用                                     |
|  +-- 高度なインシデント対応                               |
|  +-- チャンピオンの育成・サポート                          |
|       |                                                  |
|       v                                                  |
|  セキュリティチャンピオン (各チーム 1名)                     |
|  +-- 開発チーム A: チャンピオン A                          |
|  +-- 開発チーム B: チャンピオン B                          |
|  +-- 開発チーム C: チャンピオン C                          |
|  +-- インフラチーム: チャンピオン D                         |
|                                                          |
|  チャンピオンの役割:                                       |
|  +-- チーム内のセキュリティレビュー推進                     |
|  +-- 脅威モデリングのファシリテーション                     |
|  +-- セキュリティツールの導入支援                          |
|  +-- セキュリティチームとの橋渡し                          |
|  +-- チーム内のセキュリティ意識向上                         |
+----------------------------------------------------------+
```

### セキュリティチャンピオン育成プログラム

```python
# セキュリティチャンピオン管理システム
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class SecurityChampion:
    """セキュリティチャンピオンの情報管理"""
    name: str
    team: str
    appointed_date: date
    skill_level: str = "beginner"  # beginner, intermediate, advanced
    certifications: list[str] = field(default_factory=list)
    completed_trainings: list[str] = field(default_factory=list)
    mentored_reviews: int = 0
    threat_models_led: int = 0

    @property
    def tenure_months(self) -> int:
        return (date.today() - self.appointed_date).days // 30

    def should_advance(self) -> bool:
        """昇格条件の判定"""
        if self.skill_level == "beginner":
            return (
                self.tenure_months >= 3
                and len(self.completed_trainings) >= 3
                and self.mentored_reviews >= 5
            )
        elif self.skill_level == "intermediate":
            return (
                self.tenure_months >= 9
                and self.threat_models_led >= 3
                and len(self.certifications) >= 1
            )
        return False


CHAMPION_CURRICULUM = {
    "month_1": {
        "title": "基礎",
        "topics": [
            "OWASP Top 10 概要と実例",
            "セキュリティレビューチェックリストの使い方",
            "SAST/SCA ツールの結果の読み方",
        ],
        "hands_on": "自チームの直近PRから脆弱性パターンを3つ特定",
    },
    "month_2": {
        "title": "実践",
        "topics": [
            "脅威モデリング (STRIDE) の実施方法",
            "セキュアコーディングパターン",
            "認証・認可の設計レビュー",
        ],
        "hands_on": "自チームのサービスで脅威モデリングを実施",
    },
    "month_3": {
        "title": "応用",
        "topics": [
            "インシデント対応手順",
            "脆弱性トリアージの実践",
            "セキュリティメトリクスの分析",
        ],
        "hands_on": "月次セキュリティレポートの作成と発表",
    },
    "month_4_6": {
        "title": "深化",
        "topics": [
            "CTF チャレンジへの参加",
            "ペネトレーションテストの基礎",
            "クラウドセキュリティの設計パターン",
        ],
        "hands_on": "新規機能の設計段階からセキュリティレビューをリード",
    },
}
```

### DevSecOps ツールチェーン選定ガイド

```
+--------------------------------------------------------------------+
|              DevSecOps ツールチェーン                                  |
|--------------------------------------------------------------------|
|                                                                    |
| [Plan]                                                             |
|   脅威モデリング: OWASP Threat Dragon, Microsoft TMT, IriusRisk     |
|   セキュリティ要件: OWASP ASVS チェックリスト                         |
|                                                                    |
| [Code]                                                             |
|   SAST: Semgrep (OSS), SonarQube, Checkmarx                       |
|   Secret Detection: Gitleaks, TruffleHog, git-secrets              |
|   IDE Plugin: Snyk, SonarLint                                      |
|                                                                    |
| [Build]                                                            |
|   SCA: Trivy, Snyk, Dependabot                                    |
|   License: license-checker, FOSSA                                  |
|   SBOM: Syft, CycloneDX                                           |
|                                                                    |
| [Test]                                                             |
|   DAST: OWASP ZAP, Nuclei, Burp Suite                             |
|   API Security: Postman Security, 42Crunch                         |
|   Fuzzing: AFL++, Jazzer                                           |
|                                                                    |
| [Deploy]                                                           |
|   Container: Trivy, Grype, Dockle                                 |
|   IaC: Checkov, tfsec, KICS                                       |
|   Signing: Cosign, Notary                                         |
|                                                                    |
| [Operate]                                                          |
|   Runtime: Falco, Sysdig, Aqua                                    |
|   WAF: AWS WAF, Cloudflare, ModSecurity                           |
|   CSPM: Prowler, ScoutSuite                                       |
|                                                                    |
| [Monitor]                                                          |
|   SIEM: Elastic Security, Splunk, Wazuh                           |
|   Alerting: PagerDuty, Opsgenie                                   |
|   Audit: CloudTrail, Azure Monitor                                |
+--------------------------------------------------------------------+
```

---

## 2. 脅威モデリング

### STRIDE フレームワーク

| 脅威カテゴリ | 説明 | 対策例 |
|------------|------|--------|
| **S**poofing (なりすまし) | 他者のアイデンティティを詐称 | 認証、MFA |
| **T**ampering (改竄) | データや通信の不正変更 | 完全性検証、署名 |
| **R**epudiation (否認) | 行為の否定 | 監査ログ、デジタル署名 |
| **I**nformation Disclosure (情報漏洩) | 機密情報の不正アクセス | 暗号化、アクセス制御 |
| **D**enial of Service (サービス妨害) | サービスの可用性低下 | レートリミット、冗長化 |
| **E**levation of Privilege (権限昇格) | 権限の不正取得 | 最小権限、入力検証 |

### 脅威モデリングの手順

```
Step 1: システムのモデル化 (Data Flow Diagram)

  +--------+     HTTPS      +--------+     SQL      +--------+
  | ユーザ  | ------------> | Web    | -----------> | DB     |
  | (外部)  |              | サーバ  |              |        |
  +--------+     認証       +--------+   内部NW     +--------+
                Cookie           |
                             +--------+
                             | 外部   |
                             | API    |
                             +--------+

Step 2: STRIDE で脅威を列挙
  - Spoofing: セッションハイジャック
  - Tampering: SQLインジェクション
  - Information Disclosure: エラーメッセージからのDB情報漏洩
  - ...

Step 3: リスク評価 (DREAD or 影響度 x 発生確率)

Step 4: 対策の決定と実装
```

### PASTA (Process for Attack Simulation and Threat Analysis)

```
+---------------------------------------------------------------+
|  PASTA 脅威モデリング — 7段階プロセス                             |
|---------------------------------------------------------------|
|                                                               |
|  Stage 1: ビジネス目標の定義                                    |
|    → アプリケーションの目的、ビジネスインパクト、                   |
|      コンプライアンス要件を明確化                                 |
|    → 例: ECサイト — 決済データの保護が最優先                      |
|                                                               |
|  Stage 2: 技術スコープの定義                                    |
|    → システムアーキテクチャ、データフロー、                        |
|      技術スタックの文書化                                        |
|    → DFD (Data Flow Diagram) の作成                            |
|                                                               |
|  Stage 3: アプリケーションの分解                                 |
|    → コンポーネント間の信頼境界の特定                             |
|    → エントリポイント、アセット、                                 |
|      アクセス制御ポイントの列挙                                   |
|                                                               |
|  Stage 4: 脅威分析                                              |
|    → 脅威インテリジェンスの収集                                   |
|    → 攻撃者のプロファイリング（内部/外部、スキルレベル）            |
|    → 業界固有の脅威パターンの特定                                 |
|                                                               |
|  Stage 5: 脆弱性と弱点の分析                                    |
|    → 既存のスキャン結果の分析                                    |
|    → 設計上の弱点の特定                                         |
|    → CVSS スコアリング                                          |
|                                                               |
|  Stage 6: 攻撃モデリングとシミュレーション                        |
|    → 攻撃ツリーの構築                                           |
|    → 攻撃シナリオのシミュレーション                               |
|    → ペネトレーションテストとの連携                               |
|                                                               |
|  Stage 7: リスク分析と対策                                      |
|    → ビジネスインパクトの定量評価                                 |
|    → 対策の優先順位付け                                         |
|    → 残余リスクの受容判断                                        |
+---------------------------------------------------------------+
```

### 攻撃ツリー分析

```
攻撃目標: ユーザアカウントの不正アクセス
│
├── 1. 認証情報の窃取
│   ├── 1.1 フィッシングメール [確率: 高, コスト: 低]
│   │   ├── 1.1.1 偽ログインページへの誘導
│   │   └── 1.1.2 マルウェア添付ファイル
│   ├── 1.2 クレデンシャルスタッフィング [確率: 中, コスト: 低]
│   │   └── 1.2.1 過去の漏洩DBからの認証試行
│   ├── 1.3 キーロガー [確率: 低, コスト: 中]
│   └── 1.4 ショルダーサーフィング [確率: 低, コスト: 低]
│
├── 2. セッションハイジャック
│   ├── 2.1 XSS によるCookie窃取 [確率: 中, コスト: 中]
│   ├── 2.2 中間者攻撃 (MitM) [確率: 低, コスト: 高]
│   └── 2.3 セッション固定攻撃 [確率: 低, コスト: 中]
│
├── 3. 認証バイパス
│   ├── 3.1 パスワードリセット機能の悪用 [確率: 中, コスト: 低]
│   │   ├── 3.1.1 予測可能なリセットトークン
│   │   └── 3.1.2 メール傍受
│   ├── 3.2 OAuth リダイレクト操作 [確率: 低, コスト: 中]
│   └── 3.3 JWT アルゴリズム混同攻撃 [確率: 低, コスト: 高]
│
└── 4. 権限昇格
    ├── 4.1 IDOR (直接オブジェクト参照) [確率: 中, コスト: 低]
    ├── 4.2 ロールの不正変更 [確率: 低, コスト: 中]
    └── 4.3 API パラメータ改竄 [確率: 中, コスト: 低]
```

### 脅威モデリングの実施テンプレート

```yaml
# threat-model.yaml
system: "User Authentication Service"
date: "2025-03-15"
version: "2.0"
participants:
  - "Security Champion: 田中"
  - "Tech Lead: 鈴木"
  - "Backend Developer: 佐藤"
  - "QA Engineer: 高橋"

assets:
  - name: "ユーザ認証情報"
    sensitivity: "HIGH"
    data_classification: "Confidential"
  - name: "セッショントークン"
    sensitivity: "HIGH"
    data_classification: "Confidential"
  - name: "ユーザプロフィール"
    sensitivity: "MEDIUM"
    data_classification: "Internal"

trust_boundaries:
  - name: "インターネット ↔ WAF"
    type: "Network"
  - name: "WAF ↔ アプリケーション"
    type: "Network"
  - name: "アプリケーション ↔ データベース"
    type: "Network"
  - name: "ブラウザ ↔ API"
    type: "Process"

threats:
  - id: T001
    category: "Spoofing"
    description: "盗まれた認証情報による不正ログイン"
    attack_vector: "フィッシング、クレデンシャルスタッフィング"
    risk: "HIGH"
    cvss: 8.1
    mitigation:
      - "MFA の必須化"
      - "異常ログイン検知 (Impossible Travel)"
      - "Have I Been Pwned API によるパスワード漏洩チェック"
    status: "MITIGATED"
    residual_risk: "LOW"

  - id: T002
    category: "Information Disclosure"
    description: "ブルートフォースによるアカウント列挙"
    attack_vector: "ログインフォームのエラーメッセージの差異"
    risk: "MEDIUM"
    cvss: 5.3
    mitigation:
      - "ログイン失敗時の一律エラーメッセージ"
      - "レートリミット (5回/分)"
      - "CAPTCHA (3回失敗後)"
    status: "MITIGATED"
    residual_risk: "LOW"

  - id: T003
    category: "Elevation of Privilege"
    description: "JWT の改竄による権限昇格"
    attack_vector: "alg:none攻撃、鍵混同攻撃"
    risk: "HIGH"
    cvss: 9.1
    mitigation:
      - "RS256 署名の検証"
      - "alg ヘッダのホワイトリスト検証"
      - "JWK のローテーション (90日)"
    status: "MITIGATED"
    residual_risk: "LOW"

  - id: T004
    category: "Tampering"
    description: "セッショントークンの改竄"
    attack_vector: "XSS経由のCookie操作"
    risk: "HIGH"
    cvss: 7.5
    mitigation:
      - "HttpOnly, Secure, SameSite=Strict Cookie属性"
      - "CSP (Content Security Policy) の適用"
      - "サーバサイドでのセッション検証"
    status: "MITIGATED"
    residual_risk: "LOW"

review_schedule:
  next_review: "2025-06-15"
  trigger_events:
    - "新機能の追加"
    - "アーキテクチャの変更"
    - "重大なインシデントの発生"
    - "依存ライブラリの大規模アップデート"
```

### 脅威モデリング自動化ツール

```python
# 脅威モデリングの自動化ヘルパー
import json
from pathlib import Path
from typing import Optional


class ThreatModelGenerator:
    """OpenAPI仕様から脅威モデルの雛形を自動生成"""

    STRIDE_PATTERNS = {
        "authentication": {
            "threats": [
                {
                    "category": "Spoofing",
                    "template": "認証エンドポイント {endpoint} に対するクレデンシャルスタッフィング",
                    "default_risk": "HIGH",
                },
                {
                    "category": "Repudiation",
                    "template": "認証イベントのログ不足による否認",
                    "default_risk": "MEDIUM",
                },
            ],
        },
        "data_retrieval": {
            "threats": [
                {
                    "category": "Information Disclosure",
                    "template": "エンドポイント {endpoint} での過剰なデータ露出",
                    "default_risk": "MEDIUM",
                },
                {
                    "category": "Elevation of Privilege",
                    "template": "エンドポイント {endpoint} での IDOR による他ユーザデータアクセス",
                    "default_risk": "HIGH",
                },
            ],
        },
        "data_modification": {
            "threats": [
                {
                    "category": "Tampering",
                    "template": "エンドポイント {endpoint} でのリクエストボディ改竄",
                    "default_risk": "HIGH",
                },
                {
                    "category": "Denial of Service",
                    "template": "エンドポイント {endpoint} への大量リクエストによるDoS",
                    "default_risk": "MEDIUM",
                },
            ],
        },
    }

    def generate_from_openapi(self, spec_path: str) -> dict:
        """OpenAPI仕様から脅威モデルを生成"""
        with open(spec_path) as f:
            spec = json.load(f)

        threats = []
        threat_id = 1

        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                endpoint = f"{method.upper()} {path}"

                # エンドポイントの特性を判定
                if "auth" in path.lower() or "login" in path.lower():
                    pattern_key = "authentication"
                elif method in ("get", "head"):
                    pattern_key = "data_retrieval"
                else:
                    pattern_key = "data_modification"

                for threat_template in self.STRIDE_PATTERNS[pattern_key]["threats"]:
                    threats.append({
                        "id": f"T{threat_id:03d}",
                        "category": threat_template["category"],
                        "description": threat_template["template"].format(
                            endpoint=endpoint
                        ),
                        "risk": threat_template["default_risk"],
                        "status": "OPEN",
                        "mitigation": [],
                    })
                    threat_id += 1

        return {
            "system": spec.get("info", {}).get("title", "Unknown"),
            "version": spec.get("info", {}).get("version", "1.0"),
            "threats": threats,
            "total_threats": len(threats),
            "by_risk": {
                "HIGH": sum(1 for t in threats if t["risk"] == "HIGH"),
                "MEDIUM": sum(1 for t in threats if t["risk"] == "MEDIUM"),
                "LOW": sum(1 for t in threats if t["risk"] == "LOW"),
            },
        }
```

---

## 3. バグバウンティ

### バグバウンティプログラムの設計

```
+----------------------------------------------------------+
|           バグバウンティプログラム                           |
|----------------------------------------------------------|
|                                                          |
|  [スコープ定義]                                           |
|  +-- 対象: app.example.com, api.example.com              |
|  +-- 除外: staging.example.com, 社内ツール               |
|  +-- 禁止行為: DoS, ソーシャルエンジニアリング              |
|                                                          |
|  [報奨金テーブル]                                         |
|  +-- Critical (RCE, SQLi): $5,000 - $15,000             |
|  +-- High (XSS, IDOR): $1,000 - $5,000                  |
|  +-- Medium (情報漏洩): $500 - $1,000                    |
|  +-- Low (設定ミス): $100 - $500                          |
|                                                          |
|  [対応 SLA]                                               |
|  +-- 初期応答: 1営業日以内                                |
|  +-- トリアージ: 3営業日以内                               |
|  +-- 修正: Critical 7日, High 30日, Medium 90日          |
|  +-- 報奨金支払い: 修正確認後 30日以内                     |
+----------------------------------------------------------+
```

### バグバウンティプラットフォームの比較

| 項目 | HackerOne | Bugcrowd | Intigriti |
|------|-----------|----------|-----------|
| 研究者数 | 100万+ | 50万+ | 7万+ |
| 地域 | グローバル | グローバル | 欧州中心 |
| 管理型 | あり | あり | あり |
| プライベートプログラム | あり | あり | あり |
| トリアージ代行 | あり (有料) | あり (有料) | あり (有料) |
| 最低予算 | $1,000/月程度 | $1,000/月程度 | 要問い合わせ |
| GDPR 対応 | あり | あり | 特に強い |
| 日本語サポート | 限定的 | 限定的 | なし |

### バグバウンティプログラム成熟度ロードマップ

```
Phase 1: 内部準備 (1-2ヶ月)
├── VDP (Vulnerability Disclosure Policy) の策定
├── セキュリティチームの対応体制整備
├── インシデント対応プロセスとの連携
├── 法務チームとの報奨金契約テンプレート準備
└── 内部ペネトレーションテストの実施

Phase 2: プライベートプログラム (3-6ヶ月)
├── 招待制で 10-20 名の研究者を選定
├── スコープは主要アプリケーションに限定
├── 対応フローの検証と改善
├── 平均対応時間 (MTTR) の測定開始
└── 重複報告の管理プロセス確立

Phase 3: プライベート拡大 (6-12ヶ月)
├── 研究者を 50-100 名に拡大
├── スコープをAPI, モバイルアプリに拡大
├── 報奨金テーブルの見直し
├── 四半期レポートの作成開始
└── 自動トリアージツールの導入検討

Phase 4: パブリックプログラム (12ヶ月以降)
├── 全研究者に公開
├── フルスコープ (全公開資産)
├── Hall of Fame ページの設置
├── 年次レポートの公開
└── バグバウンティイベント (Live Hacking) の検討
```

### バグ報告の処理フロー

```python
# バグバウンティ報告の処理自動化
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ReportStatus(Enum):
    NEW = "new"
    TRIAGED = "triaged"
    DUPLICATE = "duplicate"
    INFORMATIVE = "informative"
    NOT_APPLICABLE = "not_applicable"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    BOUNTY_PAID = "bounty_paid"


@dataclass
class BugReport:
    """バグバウンティ報告"""
    id: str
    title: str
    description: str
    reporter_email: str
    severity: str
    submitted_at: datetime
    status: ReportStatus = ReportStatus.NEW
    assigned_to: str | None = None
    bounty_amount: float | None = None
    resolution_notes: str = ""
    sla_breach: bool = False


class BugBountyWorkflow:
    """バグバウンティ報告の処理ワークフロー"""

    SEVERITY_SLA = {
        'critical': {'triage_hours': 4, 'fix_days': 7, 'bounty_range': (5000, 15000)},
        'high': {'triage_hours': 24, 'fix_days': 30, 'bounty_range': (1000, 5000)},
        'medium': {'triage_hours': 72, 'fix_days': 90, 'bounty_range': (500, 1000)},
        'low': {'triage_hours': 168, 'fix_days': 180, 'bounty_range': (100, 500)},
    }

    def __init__(self):
        self.reports: list[BugReport] = []
        self.known_vulnerabilities: list[str] = []

    def receive_report(self, report: dict) -> BugReport:
        """報告の受領と初期対応"""
        bug_report = BugReport(
            id=f"BB-{len(self.reports) + 1:04d}",
            title=report['title'],
            description=report['description'],
            reporter_email=report['reporter_email'],
            severity="pending",
            submitted_at=datetime.now(),
        )
        self.reports.append(bug_report)

        # 1. 自動応答
        self.send_acknowledgment(bug_report)

        # 2. 重複チェック
        if self.is_duplicate(bug_report):
            bug_report.status = ReportStatus.DUPLICATE
            self.notify_reporter(
                "既知の脆弱性として報告済みです。ご報告ありがとうございます。",
                bug_report,
            )
            return bug_report

        # 3. 重大度評価とトリアージ
        severity = self.assess_severity(bug_report)
        bug_report.severity = severity
        sla = self.SEVERITY_SLA[severity]
        bug_report.status = ReportStatus.TRIAGED

        # 4. 内部チケット作成
        ticket = self.create_internal_ticket(bug_report, sla)

        # 5. セキュリティチームに通知
        self.notify_security_team(bug_report, severity)

        # 6. SLA モニタリング開始
        self.start_sla_monitoring(bug_report, sla)

        return bug_report

    def assess_severity(self, report: BugReport) -> str:
        """CVSS ベースの重大度評価"""
        title_lower = report.title.lower()
        desc_lower = report.description.lower()
        combined = f"{title_lower} {desc_lower}"

        critical_indicators = ['rce', 'remote code execution', 'sqli', 'sql injection',
                               'ssrf', 'authentication bypass', 'privilege escalation']
        high_indicators = ['xss', 'cross-site scripting', 'idor', 'insecure direct',
                          'csrf', 'path traversal', 'file upload']
        medium_indicators = ['information disclosure', 'sensitive data', 'misconfiguration',
                            'open redirect', 'clickjacking']

        if any(indicator in combined for indicator in critical_indicators):
            return 'critical'
        elif any(indicator in combined for indicator in high_indicators):
            return 'high'
        elif any(indicator in combined for indicator in medium_indicators):
            return 'medium'
        return 'low'

    def calculate_bounty(self, report: BugReport) -> float:
        """報奨金の算出"""
        sla = self.SEVERITY_SLA[report.severity]
        base_min, base_max = sla['bounty_range']

        # 影響範囲、再現性、報告品質で調整
        quality_multiplier = 1.0

        # 詳細な再現手順が含まれている場合
        if len(report.description) > 500:
            quality_multiplier += 0.1

        # PoC コードが含まれている場合
        if 'poc' in report.description.lower() or 'proof of concept' in report.description.lower():
            quality_multiplier += 0.15

        # 修正提案が含まれている場合
        if 'fix' in report.description.lower() or 'recommendation' in report.description.lower():
            quality_multiplier += 0.1

        base = (base_min + base_max) / 2
        return min(base * quality_multiplier, base_max)

    def generate_monthly_report(self) -> dict:
        """月次バグバウンティレポート"""
        now = datetime.now()
        month_start = now.replace(day=1)
        monthly_reports = [
            r for r in self.reports
            if r.submitted_at >= month_start
        ]

        return {
            'period': now.strftime('%Y-%m'),
            'total_reports': len(monthly_reports),
            'by_severity': {
                sev: sum(1 for r in monthly_reports if r.severity == sev)
                for sev in ['critical', 'high', 'medium', 'low']
            },
            'by_status': {
                status.value: sum(1 for r in monthly_reports if r.status == status)
                for status in ReportStatus
            },
            'total_bounty_paid': sum(
                r.bounty_amount or 0 for r in monthly_reports
                if r.status == ReportStatus.BOUNTY_PAID
            ),
            'avg_triage_hours': self._calc_avg_triage_time(monthly_reports),
            'sla_breach_count': sum(1 for r in monthly_reports if r.sla_breach),
        }

    def _calc_avg_triage_time(self, reports: list[BugReport]) -> float:
        triaged = [r for r in reports if r.status != ReportStatus.NEW]
        if not triaged:
            return 0.0
        return sum(4.0 for _ in triaged) / len(triaged)  # simplified

    def send_acknowledgment(self, report: BugReport):
        """自動応答の送信"""
        pass

    def is_duplicate(self, report: BugReport) -> bool:
        """重複チェック"""
        return report.title in self.known_vulnerabilities

    def notify_reporter(self, message: str, report: BugReport):
        """報告者への通知"""
        pass

    def create_internal_ticket(self, report: BugReport, sla: dict) -> str:
        """内部チケット作成"""
        return f"SEC-{report.id}"

    def notify_security_team(self, report: BugReport, severity: str):
        """セキュリティチームへの通知"""
        pass

    def start_sla_monitoring(self, report: BugReport, sla: dict):
        """SLAモニタリング開始"""
        pass
```

---

## 4. セキュリティ教育

### 教育プログラムの設計

```
+----------------------------------------------------------+
|          セキュリティ教育プログラム                          |
|----------------------------------------------------------|
|                                                          |
|  [全社員向け (年次)]                                       |
|  +-- フィッシング対策研修                                  |
|  +-- パスワード管理とMFAの必要性                           |
|  +-- SNS での情報漏洩防止                                  |
|  +-- インシデント報告の手順                                |
|  +-- 物理セキュリティ（クリーンデスク、尾行防止）            |
|                                                          |
|  [開発者向け (四半期)]                                     |
|  +-- OWASP Top 10 ハンズオン                              |
|  +-- セキュアコーディング演習                              |
|  +-- 脅威モデリングワークショップ                          |
|  +-- CTF (Capture The Flag) イベント                      |
|  +-- セキュリティレビュー実践（PR ベース）                  |
|                                                          |
|  [セキュリティチャンピオン向け (月次)]                      |
|  +-- 最新脆弱性のブリーフィング                            |
|  +-- ツール活用の深掘り                                   |
|  +-- インシデントケーススタディ                             |
|  +-- ペネトレーションテスト基礎                            |
|                                                          |
|  [経営層向け (半期)]                                       |
|  +-- セキュリティリスクレポート                             |
|  +-- 投資対効果の説明                                     |
|  +-- コンプライアンス状況の報告                             |
|  +-- サイバー保険の評価                                   |
+----------------------------------------------------------+
```

### セキュアコーディング研修カリキュラム

```
Day 1: Web アプリケーションセキュリティ基礎 (4時間)
├── 座学 (1.5h)
│   ├── OWASP Top 10 の概要
│   ├── 各脆弱性の実例と影響
│   └── セキュリティの基本原則（多層防御、最小権限）
├── ハンズオン (2h)
│   ├── OWASP WebGoat / Juice Shop を使った攻撃体験
│   ├── SQLインジェクションの発見と修正
│   └── XSS の発見と修正
└── 振り返り (0.5h)
    └── 自社アプリに当てはまるパターンの議論

Day 2: 認証・認可とデータ保護 (4時間)
├── 座学 (1.5h)
│   ├── 認証の安全な実装パターン
│   ├── JWT / セッション管理のベストプラクティス
│   └── 暗号化とハッシュの正しい使い方
├── ハンズオン (2h)
│   ├── 安全なパスワードハッシュの実装
│   ├── IDOR 脆弱性の発見と修正
│   └── CSRF 対策の実装
└── 振り返り (0.5h)

Day 3: セキュリティテストとツール (4時間)
├── 座学 (1h)
│   ├── SAST/DAST/SCA の違いと使い分け
│   └── CI/CD パイプラインへの統合方法
├── ハンズオン (2.5h)
│   ├── Semgrep カスタムルールの作成
│   ├── OWASP ZAP によるスキャン実行
│   └── Trivy による依存関係スキャン
└── 修了テスト (0.5h)
    └── 脆弱性のあるコードの特定と修正（実技）
```

### フィッシングシミュレーション

```python
# フィッシング訓練の管理と分析
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PhishingCampaign:
    """フィッシング訓練キャンペーン"""
    name: str
    template: str
    targets: list[str]
    launched_at: Optional[datetime] = None
    results: dict = field(default_factory=dict)


class PhishingSimulator:
    """フィッシング訓練の管理"""

    TEMPLATES = {
        "password_reset": {
            "subject": "【重要】パスワードの再設定が必要です",
            "difficulty": "easy",
            "indicators": [
                "送信元ドメインが正規と異なる",
                "URLのドメインが正規と異なる",
                "緊急性を煽る文言",
            ],
        },
        "invoice_payment": {
            "subject": "請求書のご確認をお願いいたします",
            "difficulty": "medium",
            "indicators": [
                "添付ファイルの拡張子(.xlsm)",
                "送信者名は正しいがアドレスが異なる",
                "マクロの有効化を促す文言",
            ],
        },
        "ceo_urgent": {
            "subject": "至急対応をお願いします（社長より）",
            "difficulty": "hard",
            "indicators": [
                "社内のメールに似せたフォーマット",
                "役職者の名前を使用",
                "返信先が外部アドレス",
            ],
        },
        "shared_document": {
            "subject": "共有ドキュメントが更新されました",
            "difficulty": "medium",
            "indicators": [
                "Google/Microsoft風のフィッシングページ",
                "OAuth認可画面を模倣",
                "正規に見えるが微妙に異なるURL",
            ],
        },
    }

    def create_campaign(
        self,
        name: str,
        template_key: str,
        department: str,
        targets: list[str],
    ) -> PhishingCampaign:
        """フィッシング訓練キャンペーンの作成"""
        template = self.TEMPLATES[template_key]
        campaign = PhishingCampaign(
            name=name,
            template=template_key,
            targets=targets,
        )
        return campaign

    def analyze_results(self, campaign: PhishingCampaign) -> dict:
        """キャンペーン結果の分析"""
        total = len(campaign.targets)
        results = campaign.results

        stats = {
            'total_targets': total,
            'emails_sent': results.get('sent', total),
            'emails_opened': results.get('opened', 0),
            'links_clicked': results.get('clicked', 0),
            'credentials_submitted': results.get('submitted', 0),
            'reported_phishing': results.get('reported', 0),
        }

        # 各率の計算
        stats['open_rate'] = f"{stats['emails_opened'] / total * 100:.1f}%"
        stats['click_rate'] = f"{stats['links_clicked'] / total * 100:.1f}%"
        stats['submit_rate'] = f"{stats['credentials_submitted'] / total * 100:.1f}%"
        stats['report_rate'] = f"{stats['reported_phishing'] / total * 100:.1f}%"

        # リスクスコア
        click_pct = stats['links_clicked'] / total * 100
        if click_pct > 30:
            stats['risk_level'] = "CRITICAL"
            stats['recommendation'] = "即座に全社向けフィッシング研修を実施"
        elif click_pct > 15:
            stats['risk_level'] = "HIGH"
            stats['recommendation'] = "対象部門の追加研修を実施"
        elif click_pct > 5:
            stats['risk_level'] = "MEDIUM"
            stats['recommendation'] = "クリックした個人への個別フォローアップ"
        else:
            stats['risk_level'] = "LOW"
            stats['recommendation'] = "現在の研修プログラムを継続"

        return stats

    def generate_trend_report(self, campaigns: list[PhishingCampaign]) -> dict:
        """経時トレンドレポート"""
        trend = []
        for campaign in sorted(campaigns, key=lambda c: c.launched_at or datetime.min):
            stats = self.analyze_results(campaign)
            trend.append({
                'campaign': campaign.name,
                'date': campaign.launched_at,
                'click_rate': stats['click_rate'],
                'report_rate': stats['report_rate'],
                'risk_level': stats['risk_level'],
            })

        # 改善率の計算
        if len(trend) >= 2:
            first_click = float(trend[0]['click_rate'].rstrip('%'))
            last_click = float(trend[-1]['click_rate'].rstrip('%'))
            improvement = first_click - last_click
            return {
                'trend': trend,
                'improvement': f"{improvement:.1f}%",
                'direction': 'improving' if improvement > 0 else 'degrading',
            }

        return {'trend': trend, 'improvement': 'N/A', 'direction': 'insufficient_data'}
```

### CTF (Capture The Flag) 社内イベントの運営

```
+----------------------------------------------------------+
|          社内 CTF イベント設計                               |
|----------------------------------------------------------|
|                                                          |
|  [カテゴリと配点]                                         |
|  Web (50-500pt)                                          |
|  +-- SQL Injection (初級: 50pt)                          |
|  +-- XSS + CSP Bypass (中級: 200pt)                      |
|  +-- SSRF to Cloud Metadata (上級: 500pt)                |
|                                                          |
|  Crypto (50-400pt)                                       |
|  +-- Base64 エンコード解読 (初級: 50pt)                   |
|  +-- JWT 改竄 (中級: 200pt)                              |
|  +-- RSA パディングオラクル (上級: 400pt)                  |
|                                                          |
|  Forensics (100-300pt)                                   |
|  +-- アクセスログ分析 (初級: 100pt)                       |
|  +-- メモリダンプ解析 (中級: 200pt)                       |
|  +-- マルウェアの挙動分析 (上級: 300pt)                   |
|                                                          |
|  Miscellaneous (50-200pt)                                |
|  +-- OSINT (50-100pt)                                    |
|  +-- ソーシャルエンジニアリング認識 (100pt)                |
|  +-- セキュリティポリシー理解度 (50pt)                     |
|                                                          |
|  [運営]                                                   |
|  +-- CTFd プラットフォーム使用                             |
|  +-- 4時間のタイムボックス                                |
|  +-- チーム戦 (3-4名/チーム)                              |
|  +-- 上位チームにはセキュリティカンファレンス参加権          |
+----------------------------------------------------------+
```

---

## 5. セキュリティメトリクス

### 測定すべき KPI

```
+----------------------------------------------------------+
|          セキュリティ KPI ダッシュボード                      |
|----------------------------------------------------------|
|                                                          |
|  [検知]                                                   |
|  +-- MTTD (Mean Time to Detect): 平均検知時間              |
|  +-- 検知率: 攻撃に対するアラート発報率                     |
|  +-- False Positive率: 誤検知の割合                        |
|                                                          |
|  [対応]                                                   |
|  +-- MTTR (Mean Time to Respond): 平均対応時間             |
|  +-- MTTC (Mean Time to Contain): 平均封じ込め時間         |
|  +-- MTTF (Mean Time to Fix): 平均修正時間                |
|                                                          |
|  [脆弱性]                                                 |
|  +-- 未修正脆弱性数 (Critical/High/Medium/Low)             |
|  +-- 脆弱性修正の平均日数                                  |
|  +-- SLA内修正率 (%)                                      |
|  +-- 再発率 (同種の脆弱性の再出現率)                        |
|                                                          |
|  [プロセス]                                               |
|  +-- セキュリティレビュー実施率                             |
|  +-- 脅威モデリング実施率                                  |
|  +-- パッチ適用率 (SLA 内)                                 |
|  +-- コードカバレッジ (セキュリティテスト)                   |
|                                                          |
|  [人材]                                                   |
|  +-- セキュリティ研修完了率                                |
|  +-- フィッシング訓練クリック率                             |
|  +-- セキュリティチャンピオンの充足率                       |
|  +-- セキュリティ関連資格保有率                             |
+----------------------------------------------------------+
```

### セキュリティダッシュボード実装

```python
# セキュリティメトリクスの自動収集と可視化
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class SecurityMetrics:
    """月次セキュリティメトリクス"""
    period: str

    # 脆弱性メトリクス
    open_critical: int = 0
    open_high: int = 0
    open_medium: int = 0
    open_low: int = 0
    vulnerabilities_found: int = 0
    vulnerabilities_fixed: int = 0
    avg_fix_days_critical: float = 0.0
    avg_fix_days_high: float = 0.0
    sla_compliance_rate: float = 0.0

    # 検知・対応メトリクス
    mttd_hours: float = 0.0
    mttr_hours: float = 0.0
    incidents_total: int = 0
    incidents_critical: int = 0
    false_positive_rate: float = 0.0

    # プロセスメトリクス
    security_review_rate: float = 0.0
    threat_model_coverage: float = 0.0
    patch_compliance_rate: float = 0.0

    # 人材メトリクス
    training_completion_rate: float = 0.0
    phishing_click_rate: float = 0.0
    champion_coverage: float = 0.0


class SecurityDashboard:
    """セキュリティダッシュボードのデータ収集"""

    def __init__(self):
        self.metrics_history: list[SecurityMetrics] = []

    def collect_vulnerability_metrics(self) -> dict:
        """脆弱性メトリクスの収集"""
        # 各ソースからデータを収集
        sast_findings = self._collect_sast_findings()
        sca_findings = self._collect_sca_findings()
        dast_findings = self._collect_dast_findings()
        pentest_findings = self._collect_pentest_findings()

        total = {
            'sast': sast_findings,
            'sca': sca_findings,
            'dast': dast_findings,
            'pentest': pentest_findings,
        }

        # 重複排除と統合
        return self._deduplicate_and_merge(total)

    def calculate_risk_score(self, metrics: SecurityMetrics) -> dict:
        """組織のセキュリティリスクスコアを計算"""
        # 重み付きスコアリング (0-100, 低いほど良い)
        vuln_score = (
            metrics.open_critical * 40
            + metrics.open_high * 20
            + metrics.open_medium * 5
            + metrics.open_low * 1
        )
        vuln_score = min(vuln_score, 100)

        process_score = 100 - (
            metrics.security_review_rate * 30
            + metrics.threat_model_coverage * 30
            + metrics.patch_compliance_rate * 40
        )

        people_score = 100 - (
            metrics.training_completion_rate * 40
            + (100 - metrics.phishing_click_rate) * 30
            + metrics.champion_coverage * 30
        )

        response_score = min(
            (metrics.mttr_hours / 24) * 20  # 24時間以内が理想
            + metrics.false_positive_rate * 30
            + (100 - metrics.sla_compliance_rate),
            100,
        )

        overall = (
            vuln_score * 0.35
            + process_score * 0.25
            + people_score * 0.20
            + response_score * 0.20
        )

        return {
            'overall_risk_score': round(overall, 1),
            'vulnerability_risk': round(vuln_score, 1),
            'process_risk': round(process_score, 1),
            'people_risk': round(people_score, 1),
            'response_risk': round(response_score, 1),
            'rating': self._score_to_rating(overall),
        }

    def _score_to_rating(self, score: float) -> str:
        if score <= 20:
            return "A (Excellent)"
        elif score <= 40:
            return "B (Good)"
        elif score <= 60:
            return "C (Acceptable)"
        elif score <= 80:
            return "D (Needs Improvement)"
        return "F (Critical)"

    def generate_executive_report(self, metrics: SecurityMetrics) -> dict:
        """経営層向けサマリーレポート"""
        risk = self.calculate_risk_score(metrics)

        # 前月との比較
        trend = {}
        if len(self.metrics_history) >= 2:
            prev = self.metrics_history[-2]
            trend = {
                'open_critical_change': metrics.open_critical - prev.open_critical,
                'mttr_change_hours': metrics.mttr_hours - prev.mttr_hours,
                'sla_compliance_change': (
                    metrics.sla_compliance_rate - prev.sla_compliance_rate
                ),
                'phishing_click_change': (
                    metrics.phishing_click_rate - prev.phishing_click_rate
                ),
            }

        return {
            'period': metrics.period,
            'risk_score': risk,
            'trend': trend,
            'key_highlights': self._generate_highlights(metrics),
            'action_items': self._generate_action_items(metrics),
        }

    def _generate_highlights(self, m: SecurityMetrics) -> list[str]:
        highlights = []
        if m.open_critical == 0:
            highlights.append("Critical脆弱性ゼロを維持")
        if m.sla_compliance_rate >= 95:
            highlights.append(f"SLA遵守率 {m.sla_compliance_rate}% (目標達成)")
        if m.phishing_click_rate < 5:
            highlights.append(f"フィッシングクリック率 {m.phishing_click_rate}% (業界平均以下)")
        return highlights

    def _generate_action_items(self, m: SecurityMetrics) -> list[str]:
        items = []
        if m.open_critical > 0:
            items.append(f"Critical脆弱性 {m.open_critical}件の即時対応")
        if m.training_completion_rate < 90:
            items.append(f"セキュリティ研修未完了者への督促 (完了率: {m.training_completion_rate}%)")
        if m.patch_compliance_rate < 95:
            items.append(f"パッチ適用率の改善 (現在: {m.patch_compliance_rate}%)")
        return items

    def _collect_sast_findings(self) -> list:
        return []

    def _collect_sca_findings(self) -> list:
        return []

    def _collect_dast_findings(self) -> list:
        return []

    def _collect_pentest_findings(self) -> list:
        return []

    def _deduplicate_and_merge(self, findings: dict) -> dict:
        return findings
```

### メトリクス収集の自動化 (AWS)

```python
# AWS 環境でのセキュリティメトリクス自動収集
import boto3
from datetime import datetime, timedelta


def collect_aws_security_metrics() -> dict:
    """月次セキュリティメトリクスの収集 (AWS)"""
    metrics = {}

    # 脆弱性メトリクス (Security Hub)
    securityhub = boto3.client('securityhub')
    findings = securityhub.get_findings(
        Filters={
            'RecordState': [{'Value': 'ACTIVE', 'Comparison': 'EQUALS'}],
            'WorkflowStatus': [{'Value': 'NEW', 'Comparison': 'EQUALS'}],
        },
    )
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for finding in findings['Findings']:
        label = finding['Severity']['Label']
        if label in severity_counts:
            severity_counts[label] += 1

    metrics['open_vulnerabilities'] = severity_counts

    # パッチ適用率 (SSM)
    ssm = boto3.client('ssm')
    compliance = ssm.list_compliance_summaries(
        Filters=[{'Key': 'ComplianceType', 'Values': ['Patch'], 'Type': 'EQUAL'}]
    )
    metrics['patch_compliance'] = compliance

    # GuardDuty 検知数
    guardduty = boto3.client('guardduty')
    metrics['threat_detections'] = {
        'period': 'last_30_days',
        'count': len(guardduty.list_findings(
            DetectorId='detector-id',
            FindingCriteria={
                'Criterion': {
                    'updatedAt': {
                        'GreaterThanOrEqual': int(
                            (datetime.now() - timedelta(days=30)).timestamp() * 1000
                        )
                    }
                }
            },
        ).get('FindingIds', [])),
    }

    # IAM Access Analyzer
    analyzer = boto3.client('accessanalyzer')
    findings_resp = analyzer.list_findings(
        analyzerArn='arn:aws:access-analyzer:ap-northeast-1:123456789012:analyzer/my-analyzer',
        filter={'status': {'eq': ['ACTIVE']}},
    )
    metrics['iam_findings'] = len(findings_resp.get('findings', []))

    # Config ルール準拠率
    config = boto3.client('config')
    compliance_summary = config.get_compliance_summary_by_config_rule()
    metrics['config_compliance'] = {
        'compliant': compliance_summary['ComplianceSummary']['CompliantResourceCount']['CappedCount'],
        'non_compliant': compliance_summary['ComplianceSummary']['NonCompliantResourceCount']['CappedCount'],
    }

    return metrics
```

---

## 6. セキュリティガバナンス

### セキュリティポリシーフレームワーク

```
+--------------------------------------------------------------------+
|              セキュリティポリシー体系                                  |
|--------------------------------------------------------------------|
|                                                                    |
|  Level 1: セキュリティポリシー (経営層承認)                           |
|  +-- 情報セキュリティ基本方針                                       |
|  +-- 適用範囲、責任体制、罰則規定                                    |
|  +-- 年次レビュー                                                  |
|                                                                    |
|  Level 2: セキュリティスタンダード (CISO承認)                        |
|  +-- アクセス制御スタンダード                                       |
|  +-- データ分類スタンダード                                         |
|  +-- 暗号化スタンダード                                             |
|  +-- インシデント対応スタンダード                                    |
|  +-- 四半期レビュー                                                |
|                                                                    |
|  Level 3: プロシージャ (チームリード承認)                             |
|  +-- パスワード管理手順                                             |
|  +-- セキュリティレビュー手順                                       |
|  +-- 脆弱性管理手順                                                |
|  +-- 変更管理手順                                                  |
|  +-- 月次レビュー                                                  |
|                                                                    |
|  Level 4: ガイドライン (推奨事項)                                    |
|  +-- セキュアコーディングガイドライン                                |
|  +-- クラウドセキュリティガイドライン                                |
|  +-- リモートワークセキュリティガイドライン                           |
|  +-- 必要に応じて更新                                              |
+--------------------------------------------------------------------+
```

### セキュリティレビューゲートの設計

```python
# セキュリティレビューゲートの自動化
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewDecision(Enum):
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"


@dataclass
class ChangeRequest:
    """変更リクエスト"""
    id: str
    title: str
    description: str
    changes_auth: bool = False
    changes_data_handling: bool = False
    changes_api_surface: bool = False
    changes_infrastructure: bool = False
    new_dependency_count: int = 0
    sast_findings_critical: int = 0
    sast_findings_high: int = 0
    sca_vulnerabilities_critical: int = 0
    sca_vulnerabilities_high: int = 0
    has_threat_model: bool = False
    security_review_completed: bool = False


class SecurityGate:
    """セキュリティレビューゲートの判定"""

    def assess_risk(self, change: ChangeRequest) -> RiskLevel:
        """変更リクエストのリスクレベルを評価"""
        risk_score = 0

        if change.changes_auth:
            risk_score += 40
        if change.changes_data_handling:
            risk_score += 30
        if change.changes_api_surface:
            risk_score += 20
        if change.changes_infrastructure:
            risk_score += 25
        if change.new_dependency_count > 5:
            risk_score += 15

        if risk_score >= 60:
            return RiskLevel.CRITICAL
        elif risk_score >= 40:
            return RiskLevel.HIGH
        elif risk_score >= 20:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def evaluate(self, change: ChangeRequest) -> tuple[ReviewDecision, list[str]]:
        """ゲート評価の実行"""
        risk = self.assess_risk(change)
        blockers = []
        conditions = []

        # 絶対ブロッカー
        if change.sast_findings_critical > 0:
            blockers.append(
                f"SAST Critical findings: {change.sast_findings_critical}件を修正してください"
            )
        if change.sca_vulnerabilities_critical > 0:
            blockers.append(
                f"SCA Critical vulnerabilities: {change.sca_vulnerabilities_critical}件を修正してください"
            )

        # リスクレベル別の要件
        if risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            if not change.has_threat_model:
                blockers.append("脅威モデリングの実施が必要です")
            if not change.security_review_completed:
                blockers.append("セキュリティチームによるレビューが必要です")

        if risk == RiskLevel.MEDIUM:
            if not change.security_review_completed:
                conditions.append("セキュリティチャンピオンによるレビューを推奨")

        if change.sast_findings_high > 0:
            conditions.append(
                f"SAST High findings: {change.sast_findings_high}件の確認を推奨"
            )

        # 判定
        if blockers:
            return ReviewDecision.BLOCKED, blockers
        elif conditions:
            return ReviewDecision.CONDITIONAL, conditions
        return ReviewDecision.APPROVED, []
```

---

## 7. アンチパターン

### アンチパターン 1: セキュリティはセキュリティチームだけの仕事

```
NG:
  → セキュリティチームが全 PR をレビュー (ボトルネック化)
  → 開発者はセキュリティに無関心
  → 「セキュリティは邪魔」という認識
  → リリース直前のゲート型レビューで手戻り頻発

OK:
  → セキュリティチャンピオンが各チームにいる
  → 自動化ツールで基本チェックを開発者が実行
  → セキュリティチームは設計レビューと高度な分析に集中
  → 「セキュリティは品質の一部」という文化
  → 開発早期からのセキュリティ考慮 (Shift Left)
```

### アンチパターン 2: 恐怖による動機付け

```
NG:
  → 「セキュリティ違反したら罰則」
  → インシデントの犯人探し
  → 失敗を隠す文化の醸成
  → セキュリティ報告の抑制

OK:
  → 脆弱性を見つけた人を称賛
  → Blameless ポストモーテム
  → 学習機会としてのインシデント共有
  → セキュリティ改善への貢献を評価
  → 心理的安全性の確保
```

### アンチパターン 3: ツール偏重

```
NG:
  → 高額なツールを導入して安心
  → アラートの洪水で本当の脅威を見逃す
  → ツールの結果を誰もレビューしない
  → ツール導入 = セキュリティ対策完了という誤解

OK:
  → ツールは人の判断を補助するもの
  → アラートのチューニングと優先順位付け
  → ツールの結果を定期的にレビューし改善
  → プロセスと文化の改善がツールに先行する
```

### アンチパターン 4: コンプライアンス = セキュリティ

```
NG:
  → チェックリストを埋めることが目的化
  → 監査のためだけのセキュリティ施策
  → 実効性のないポリシー文書の山
  → 年次監査の時だけセキュリティを意識

OK:
  → コンプライアンスはセキュリティのベースライン
  → リスクベースのアプローチで優先順位付け
  → 継続的なセキュリティ改善サイクル
  → 自社のリスクプロファイルに合わせた対策
```

---

## 8. FAQ

### Q1. DevSecOps を導入するための最初のステップは?

まず CI/CD パイプラインに SAST ツール (Semgrep) と SCA ツール (Trivy) を追加し、Critical/High のみをビルドブロッカーにする。並行して各開発チームから 1 名のセキュリティチャンピオンを任命し、月次で集まりを開始する。ツールの結果を開発者にとってアクショナブルにすることが普及の鍵である。最初から全てを導入しようとせず、3ヶ月ごとに1つずつツールを追加していくインクリメンタルなアプローチが効果的である。

### Q2. バグバウンティプログラムはいつ始めるべきか?

内部のセキュリティテスト (SAST/DAST/ペネトレーションテスト) が成熟してから始めるべきである。既知の脆弱性が多数残っている段階で始めると、大量の報告に対応しきれず、報奨金の予算も圧迫する。まずプライベートプログラム (招待制) で少数の研究者から始め、対応プロセスが安定したらパブリックに移行する。目安として、内部ペネトレーションテストでCritical/Highの脆弱性がゼロになってからが適切なタイミングである。

### Q3. セキュリティ文化の効果をどう測定するか?

定量的指標としてフィッシング訓練のクリック率の推移、脆弱性の平均修正時間 (MTTR)、セキュリティレビュー実施率を追跡する。定性的にはセキュリティに関する質問が開発チームから自発的に上がるか、インシデント報告が迅速に行われるかを観察する。半年に一度、全社アンケートでセキュリティ意識の変化を測定することも有効である。

### Q4. セキュリティチャンピオンのモチベーションを維持するには?

通常業務の 10-20% をセキュリティ活動に充てることを公式に認める。セキュリティカンファレンスへの参加費用を支援する。チャンピオン間のコミュニティを作り、定期的な知識共有の場を設ける。人事評価にセキュリティへの貢献を明示的に含める。資格取得の支援（費用補助、学習時間の確保）も効果的である。

### Q5. 小規模チームでも DevSecOps は実践できるか?

できる。小規模チームほど全員がセキュリティを意識する文化を作りやすい。まずは GitHub の Dependabot（無料）でSCAを、Semgrep（OSS）でSASTを導入する。専任のセキュリティチームがいなくても、全員がセキュリティチャンピオンとして機能する文化を目指す。重要なのはツールの数ではなく、セキュリティを開発プロセスの一部として組み込む意識である。

### Q6. セキュリティ研修の効果が出ない場合はどうするか?

座学中心の研修は効果が低い。ハンズオン形式（OWASP Juice Shop、TryHackMe、社内CTF）に切り替える。自社のコードベースから実際の脆弱性（修正済み）を題材にすると、開発者の当事者意識が高まる。研修は1回きりではなく、四半期ごとの反復が重要。学んだ内容をすぐにコードレビューで実践する場を設けることで定着率が大幅に向上する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| DevSecOps | セキュリティを開発プロセスの全段階に統合 |
| セキュリティチャンピオン | 各チームに 1 名、セキュリティの橋渡し役 |
| 脅威モデリング | STRIDE/PASTA で設計段階からリスクを特定 |
| バグバウンティ | プライベートから始め、段階的にパブリック化 |
| セキュリティ教育 | 対象別プログラム、フィッシング訓練は定期実施 |
| メトリクス | MTTD/MTTR、脆弱性数、研修完了率を継続測定 |
| ガバナンス | ポリシー→スタンダード→プロシージャの階層構造 |
| 文化 | 恐怖でなく称賛で動機付け、Blameless ポストモーテム |

---

## 次に読むべきガイド

- [インシデント対応](./00-incident-response.md) — セキュリティ文化が試される場面
- [コンプライアンス](./02-compliance.md) — 組織的なセキュリティガバナンス
- [セキュアコーディング](../04-application-security/00-secure-coding.md) — 開発者が身につけるべきスキル

---

## 参考文献

1. **NIST Cybersecurity Framework** — https://www.nist.gov/cyberframework
2. **OWASP DevSecOps Guideline** — https://owasp.org/www-project-devsecops-guideline/
3. **OWASP Threat Modeling** — https://owasp.org/www-community/Threat_Modeling
4. **HackerOne — Bug Bounty Program Guide** — https://www.hackerone.com/resources
5. **Google — Building Security Culture** — https://sre.google/sre-book/culture/
6. **SANS Security Awareness Report** — https://www.sans.org/security-awareness-training/reports/
7. **Microsoft SDL (Security Development Lifecycle)** — https://www.microsoft.com/en-us/securityengineering/sdl
8. **BSIMM (Building Security In Maturity Model)** — https://www.bsimm.com/
