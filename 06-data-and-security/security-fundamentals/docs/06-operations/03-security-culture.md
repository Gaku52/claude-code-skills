# セキュリティ文化

> DevSecOps による開発プロセスへのセキュリティ統合、バグバウンティプログラムの運用、組織全体のセキュリティ意識を向上させるための実践ガイド

## この章で学ぶこと

1. **DevSecOps** — 開発・運用・セキュリティを統合した組織体制とプロセス
2. **バグバウンティ** — 外部研究者によるセキュリティテストの仕組みと運用
3. **セキュリティ教育と意識向上** — 組織全体でセキュリティを自分事にする文化の構築

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
|                                                          |
|  Level 2: 管理 (Managed)                                  |
|  +-- CI/CD にセキュリティテストを統合                      |
|  +-- SAST/SCA の自動実行                                 |
|  +-- セキュリティチャンピオンの任命                         |
|                                                          |
|  Level 3: 定義 (Defined)                                  |
|  +-- 脅威モデリングが設計プロセスの一部                     |
|  +-- セキュリティ要件が User Story に含まれる              |
|  +-- 全開発者がセキュアコーディング研修を修了               |
|                                                          |
|  Level 4: 測定 (Measured)                                 |
|  +-- セキュリティメトリクスの継続測定                       |
|  +-- 脆弱性の平均修正時間 (MTTR) を追跡                   |
|  +-- セキュリティ債務の可視化と管理                         |
|                                                          |
|  Level 5: 最適化 (Optimized)                              |
|  +-- セキュリティが全員の責任として浸透                     |
|  +-- 自動修復・自動封じ込め                               |
|  +-- 継続的な改善サイクルの確立                            |
+----------------------------------------------------------+
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

### 脅威モデリングの実施テンプレート

```yaml
# threat-model.yaml
system: "User Authentication Service"
date: "2025-03-15"
participants:
  - "Security Champion: 田中"
  - "Tech Lead: 鈴木"
  - "Backend Developer: 佐藤"

assets:
  - name: "ユーザ認証情報"
    sensitivity: "HIGH"
  - name: "セッショントークン"
    sensitivity: "HIGH"
  - name: "ユーザプロフィール"
    sensitivity: "MEDIUM"

threats:
  - id: T001
    category: "Spoofing"
    description: "盗まれた認証情報による不正ログイン"
    risk: "HIGH"
    mitigation:
      - "MFA の必須化"
      - "異常ログイン検知 (Impossible Travel)"
    status: "MITIGATED"

  - id: T002
    category: "Information Disclosure"
    description: "ブルートフォースによるアカウント列挙"
    risk: "MEDIUM"
    mitigation:
      - "ログイン失敗時の一律エラーメッセージ"
      - "レートリミット (5回/分)"
    status: "MITIGATED"

  - id: T003
    category: "Elevation of Privilege"
    description: "JWT の改竄による権限昇格"
    risk: "HIGH"
    mitigation:
      - "RS256 署名の検証"
      - "alg ヘッダのホワイトリスト検証"
    status: "MITIGATED"
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

### バグ報告の処理フロー

```python
# バグバウンティ報告の処理自動化

class BugBountyWorkflow:
    """バグバウンティ報告の処理ワークフロー"""

    SEVERITY_SLA = {
        'critical': {'triage_hours': 4, 'fix_days': 7},
        'high': {'triage_hours': 24, 'fix_days': 30},
        'medium': {'triage_hours': 72, 'fix_days': 90},
        'low': {'triage_hours': 168, 'fix_days': 180},
    }

    def receive_report(self, report: dict):
        """報告の受領と初期対応"""
        # 1. 自動応答
        self.send_acknowledgment(report['reporter_email'])

        # 2. 重複チェック
        if self.is_duplicate(report):
            self.notify_reporter("既知の脆弱性です", report)
            return

        # 3. トリアージ
        severity = self.assess_severity(report)
        sla = self.SEVERITY_SLA[severity]

        # 4. 内部チケット作成
        ticket = self.create_jira_ticket(
            title=f"[BB-{severity.upper()}] {report['title']}",
            description=report['description'],
            severity=severity,
            deadline_days=sla['fix_days'],
        )

        # 5. セキュリティチームに通知
        self.notify_security_team(ticket, severity)

        return ticket

    def assess_severity(self, report: dict) -> str:
        """CVSS ベースの重大度評価"""
        # 簡易評価: 実際には CVSS Calculator を使用
        if 'RCE' in report['title'] or 'SQLi' in report['title']:
            return 'critical'
        elif 'XSS' in report['title'] or 'IDOR' in report['title']:
            return 'high'
        elif 'information' in report['title'].lower():
            return 'medium'
        return 'low'
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
|  +-- パスワード管理                                       |
|  +-- SNS での情報漏洩防止                                  |
|  +-- インシデント報告の手順                                |
|                                                          |
|  [開発者向け (四半期)]                                     |
|  +-- OWASP Top 10 ハンズオン                              |
|  +-- セキュアコーディング演習                              |
|  +-- 脅威モデリングワークショップ                          |
|  +-- CTF (Capture The Flag) イベント                      |
|                                                          |
|  [セキュリティチャンピオン向け (月次)]                      |
|  +-- 最新脆弱性のブリーフィング                            |
|  +-- ツール活用の深掘り                                   |
|  +-- インシデントケーススタディ                             |
|                                                          |
|  [経営層向け (半期)]                                       |
|  +-- セキュリティリスクレポート                             |
|  +-- 投資対効果の説明                                     |
|  +-- コンプライアンス状況の報告                             |
+----------------------------------------------------------+
```

### フィッシングシミュレーション

```python
# フィッシング訓練の管理スクリプト (GoPhish API)
import requests

GOPHISH_API = "https://gophish.internal.example.com/api"
API_KEY = "your-api-key"

def create_phishing_campaign(name: str, template: str, targets: list):
    """フィッシング訓練キャンペーンの作成"""
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # ターゲットグループの作成
    group = requests.post(f"{GOPHISH_API}/groups/", headers=headers, json={
        "name": f"{name}-targets",
        "targets": [{"email": t} for t in targets],
    }).json()

    # キャンペーンの作成
    campaign = requests.post(f"{GOPHISH_API}/campaigns/", headers=headers, json={
        "name": name,
        "template": {"name": template},
        "page": {"name": "awareness-training"},  # クリック後の教育ページ
        "smtp": {"name": "training-smtp"},
        "groups": [{"name": group["name"]}],
        "launch_date": "2025-04-01T09:00:00+09:00",
    }).json()

    return campaign

# 結果の集計
def get_campaign_results(campaign_id: int) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    results = requests.get(
        f"{GOPHISH_API}/campaigns/{campaign_id}/results",
        headers=headers,
    ).json()

    stats = {
        'total': len(results),
        'opened': sum(1 for r in results if r['status'] == 'Email Opened'),
        'clicked': sum(1 for r in results if r['status'] == 'Clicked Link'),
        'submitted': sum(1 for r in results if r['status'] == 'Submitted Data'),
        'reported': sum(1 for r in results if r['status'] == 'Email Reported'),
    }
    stats['click_rate'] = f"{stats['clicked'] / stats['total'] * 100:.1f}%"
    return stats
```

---

## 5. セキュリティメトリクス

### 測定すべき KPI

```
+----------------------------------------------------------+
|          セキュリティ KPI                                   |
|----------------------------------------------------------|
|                                                          |
|  [検知]                                                   |
|  +-- MTTD (Mean Time to Detect): 平均検知時間              |
|  +-- 検知率: 攻撃に対するアラート発報率                     |
|                                                          |
|  [対応]                                                   |
|  +-- MTTR (Mean Time to Respond): 平均対応時間             |
|  +-- MTTC (Mean Time to Contain): 平均封じ込め時間         |
|                                                          |
|  [脆弱性]                                                 |
|  +-- 未修正脆弱性数 (Critical/High/Medium/Low)             |
|  +-- 脆弱性修正の平均日数                                  |
|  +-- 再発率 (同種の脆弱性の再出現率)                        |
|                                                          |
|  [プロセス]                                               |
|  +-- セキュリティレビュー実施率                             |
|  +-- 脅威モデリング実施率                                  |
|  +-- パッチ適用率 (SLA 内)                                 |
|                                                          |
|  [人材]                                                   |
|  +-- セキュリティ研修完了率                                |
|  +-- フィッシング訓練クリック率                             |
|  +-- セキュリティチャンピオンの充足率                       |
+----------------------------------------------------------+
```

### メトリクス収集の自動化

```python
# セキュリティメトリクスの自動収集
import boto3
from datetime import datetime, timedelta

def collect_security_metrics() -> dict:
    """月次セキュリティメトリクスの収集"""
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

    return metrics
```

---

## 6. アンチパターン

### アンチパターン 1: セキュリティはセキュリティチームだけの仕事

```
NG:
  → セキュリティチームが全 PR をレビュー (ボトルネック化)
  → 開発者はセキュリティに無関心
  → 「セキュリティは邪魔」という認識

OK:
  → セキュリティチャンピオンが各チームにいる
  → 自動化ツールで基本チェックを開発者が実行
  → セキュリティチームは設計レビューと高度な分析に集中
  → 「セキュリティは品質の一部」という文化
```

### アンチパターン 2: 恐怖による動機付け

```
NG:
  → 「セキュリティ違反したら罰則」
  → インシデントの犯人探し
  → 失敗を隠す文化の醸成

OK:
  → 脆弱性を見つけた人を称賛
  → Blameless ポストモーテム
  → 学習機会としてのインシデント共有
  → セキュリティ改善への貢献を評価
```

---

## 7. FAQ

### Q1. DevSecOps を導入するための最初のステップは?

まず CI/CD パイプラインに SAST ツール (Semgrep) と SCA ツール (Trivy) を追加し、Critical/High のみをビルドブロッカーにする。並行して各開発チームから 1 名のセキュリティチャンピオンを任命し、月次で集まりを開始する。ツールの結果を開発者にとってアクショナブルにすることが普及の鍵である。

### Q2. バグバウンティプログラムはいつ始めるべきか?

内部のセキュリティテスト (SAST/DAST/ペネトレーションテスト) が成熟してから始めるべきである。既知の脆弱性が多数残っている段階で始めると、大量の報告に対応しきれず、報奨金の予算も圧迫する。まずプライベートプログラム (招待制) で少数の研究者から始め、対応プロセスが安定したらパブリックに移行する。

### Q3. セキュリティ文化の効果をどう測定するか?

定量的指標としてフィッシング訓練のクリック率の推移、脆弱性の平均修正時間 (MTTR)、セキュリティレビュー実施率を追跡する。定性的にはセキュリティに関する質問が開発チームから自発的に上がるか、インシデント報告が迅速に行われるかを観察する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| DevSecOps | セキュリティを開発プロセスの全段階に統合 |
| セキュリティチャンピオン | 各チームに 1 名、セキュリティの橋渡し役 |
| 脅威モデリング | STRIDE で設計段階からリスクを特定 |
| バグバウンティ | プライベートから始め、段階的にパブリック化 |
| セキュリティ教育 | 対象別プログラム、フィッシング訓練は定期実施 |
| メトリクス | MTTD/MTTR、脆弱性数、研修完了率を継続測定 |
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
3. **HackerOne — Bug Bounty Program Guide** — https://www.hackerone.com/resources
4. **Google — Building Security Culture** — https://sre.google/sre-book/culture/
5. **SANS Security Awareness Report** — https://www.sans.org/security-awareness-training/reports/
