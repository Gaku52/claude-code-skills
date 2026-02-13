# インシデント対応

> インシデント対応フローの設計、CSIRT の組織体制、フォレンジック調査の手法、自動化による封じ込めまで、セキュリティインシデント発生時に迅速かつ正確に対応するためのガイド。インシデント対応は「技術」だけでなく「プロセス」と「人」の三位一体で成り立つ。

## この章で学ぶこと

1. **インシデント対応フロー** -- 準備から復旧・教訓までの 6 段階プロセスと各フェーズの具体的アクション
2. **CSIRT の組織と役割** -- インシデント対応チームの構成、責任分担、エスカレーションパス
3. **フォレンジック調査** -- 証拠保全、タイムライン分析、根本原因の特定手法と法的要件
4. **自動化と演習** -- SOAR による対応自動化、プレイブック設計、机上演習の実施方法

## 前提知識

- [セキュリティの基本概念](../00-basics/00-cia-triad.md) -- CIA トライアドの理解
- [ネットワークセキュリティ](../03-network-security/00-firewall-ids.md) -- ファイアウォール・IDS の基礎
- [暗号化の基礎](../02-cryptography/00-encryption-basics.md) -- ハッシュ関数・署名の理解
- [監視/ログ](./01-monitoring-logging.md) -- ログ収集・SIEM の基礎

---

## 1. インシデント対応の全体像

### なぜインシデント対応が重要なのか

インシデント対応（Incident Response: IR）が重要な理由は、セキュリティ侵害が「起きるかどうか」ではなく「いつ起きるか」の問題だからである。IBM の Cost of a Data Breach Report 2024 によると、データ侵害の平均コストは 488 万ドルであり、インシデント対応チームと IR 計画を持つ組織はそうでない組織に比べて平均 226 万ドルのコスト削減を実現している。

```
インシデント対応の投資対効果:

  IR 計画なし:
    検知まで平均 277 日
    封じ込めまで平均 70 日
    平均コスト: $4.88M
    ブランド毀損、顧客離反、法的制裁

  IR 計画あり + 訓練済み:
    検知まで平均 200 日以下
    封じ込めまで平均 55 日以下
    平均コスト: $2.62M（46% 削減）
    迅速な復旧、信頼維持
```

### NIST SP 800-61 に基づく対応フロー

```
+------+     +--------+     +---------+     +----------+
| 準備  | --> | 検知/   | --> | 封じ込め | --> | 根絶/    |
|      |     | 分析    |     |         |     | 復旧     |
+------+     +--------+     +---------+     +----------+
   ^                                              |
   |              +----------+                    |
   +--------------| 教訓/    |<-------------------+
                  | 改善     |
                  +----------+

※ このサイクルは一方向ではなく、状況に応じて前のフェーズに戻ることがある
※ 例: 封じ込め中に新たな侵害を検知 → 検知/分析フェーズに戻る
```

### 各フェーズの詳細

```
+----------------------------------------------------------+
|  Phase 1: 準備 (Preparation)                              |
|  +-- インシデント対応計画の策定と経営層の承認               |
|  +-- CSIRT メンバーの任命と訓練                             |
|  +-- 連絡先リスト (エスカレーションパス) の整備              |
|  +-- ツール・環境の整備 (フォレンジックキット)               |
|  +-- 定期的な机上演習 (Tabletop Exercise)                  |
|  +-- コミュニケーション手段の確保 (帯域外通信)              |
|----------------------------------------------------------|
|  Phase 2: 検知と分析 (Detection & Analysis)               |
|  +-- アラートのトリアージ (真偽判定)                        |
|  +-- 影響範囲の特定 (横展開の有無)                         |
|  +-- 重大度の判定 (Severity Level)                         |
|  +-- タイムライン構築の開始                                |
|  +-- IOC (Indicator of Compromise) の収集                 |
|----------------------------------------------------------|
|  Phase 3: 封じ込め (Containment)                          |
|  +-- 短期: 被害拡大の即座の阻止                            |
|  +-- 長期: 恒久対策までの暫定対応                          |
|  +-- 証拠保全 (フォレンジックイメージ取得)                  |
|  +-- 攻撃者への通知を避ける (検知を悟られない)              |
|----------------------------------------------------------|
|  Phase 4: 根絶 (Eradication)                              |
|  +-- マルウェアの除去                                     |
|  +-- 脆弱性の修正                                        |
|  +-- 侵害されたアカウントの無効化と再作成                   |
|  +-- バックドアの検出と除去                                |
|----------------------------------------------------------|
|  Phase 5: 復旧 (Recovery)                                 |
|  +-- システムの段階的な復旧                                |
|  +-- 監視強化期間の設定 (最低 30 日)                       |
|  +-- 正常性の確認 (ベースライン比較)                       |
|  +-- ステークホルダーへの状況報告                          |
|----------------------------------------------------------|
|  Phase 6: 教訓 (Lessons Learned)                          |
|  +-- ポストモーテム (事後レビュー) の実施                    |
|  +-- 対応手順の改善                                       |
|  +-- 再発防止策の実施と追跡                                |
|  +-- メトリクス (MTTD/MTTR) の記録                        |
+----------------------------------------------------------+
```

### 対応フローの比較 -- 成熟度別

| 成熟度 | 検知 | 封じ込め | 復旧 | 教訓 |
|--------|------|---------|------|------|
| Level 1 (初期) | 手動・偶発的 | 場当たり的 | 再構築 | 実施しない |
| Level 2 (管理) | SIEM アラート | 手順書あり | 手順に従い実施 | 実施するが形式的 |
| Level 3 (定義) | 相関分析・脅威インテル | 自動+手動 | 段階的・検証済み | Blameless で実施 |
| Level 4 (最適化) | ML 異常検知・SOAR | 自動封じ込め | 自動復旧・IaC | 継続的改善サイクル |

---

## 2. 重大度レベルの定義

### インシデント重大度

| レベル | 定義 | 対応時間 | 例 | エスカレーション先 |
|--------|------|---------|-----|-------------------|
| SEV-1 (Critical) | サービス停止・大規模データ漏洩 | 15分以内に対応開始 | ランサムウェア、DB 全件漏洩 | 経営層・法務・広報 |
| SEV-2 (High) | サービス劣化・限定的データ漏洩 | 1時間以内に対応開始 | 不正アクセス、DDoS | 部門長・セキュリティチーム |
| SEV-3 (Medium) | 潜在的リスク・小規模影響 | 24時間以内に対応開始 | フィッシング成功、マルウェア検知 | セキュリティチーム |
| SEV-4 (Low) | 軽微な問題・情報収集 | 1週間以内に対応 | ポートスキャン、誤設定 | オンコール担当 |

### 重大度の判定基準

```python
"""
インシデント重大度の自動判定ロジック
"""
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class Severity(IntEnum):
    SEV1 = 1  # Critical
    SEV2 = 2  # High
    SEV3 = 3  # Medium
    SEV4 = 4  # Low


@dataclass
class IncidentFactors:
    """インシデントの影響要因"""
    data_exposed: bool = False           # データ漏洩の有無
    data_count: int = 0                  # 影響を受けたレコード数
    service_impact: str = "none"         # none, degraded, down
    pii_involved: bool = False           # 個人情報の関与
    financial_data: bool = False         # 金融データの関与
    active_threat: bool = False          # 脅威が継続中か
    external_facing: bool = False        # 外部公開サービスか
    regulatory_impact: bool = False      # 法規制への影響


def assess_severity(factors: IncidentFactors) -> Severity:
    """
    影響要因に基づいてインシデントの重大度を判定する。

    判定ロジック:
    - SEV-1: サービス停止 or 大規模データ漏洩 or 規制影響
    - SEV-2: サービス劣化 or 限定的データ漏洩 or アクティブ脅威
    - SEV-3: 潜在的リスク or 小規模影響
    - SEV-4: 情報収集レベル
    """
    # SEV-1 条件
    if factors.service_impact == "down":
        return Severity.SEV1
    if factors.data_exposed and factors.data_count > 10000:
        return Severity.SEV1
    if factors.data_exposed and (factors.pii_involved or factors.financial_data):
        return Severity.SEV1
    if factors.regulatory_impact:
        return Severity.SEV1

    # SEV-2 条件
    if factors.service_impact == "degraded":
        return Severity.SEV2
    if factors.data_exposed and factors.data_count > 100:
        return Severity.SEV2
    if factors.active_threat and factors.external_facing:
        return Severity.SEV2

    # SEV-3 条件
    if factors.active_threat:
        return Severity.SEV3
    if factors.data_exposed:
        return Severity.SEV3

    # デフォルト
    return Severity.SEV4


# 使用例
factors = IncidentFactors(
    data_exposed=True,
    data_count=50000,
    pii_involved=True,
    service_impact="degraded",
)
severity = assess_severity(factors)
print(f"判定結果: SEV-{severity}")  # 判定結果: SEV-1
```

---

## 3. CSIRT の組織体制

### CSIRT の構成

```
+----------------------------------------------------------+
|                  CSIRT 組織構成                              |
|----------------------------------------------------------|
|                                                          |
|  Incident Commander (IC)                                 |
|  +-- インシデント全体の指揮・意思決定                       |
|  +-- 経営層へのエスカレーション判断                          |
|  +-- リソース配分の決定                                    |
|                                                          |
|  Technical Lead                                          |
|  +-- 技術的な調査・分析の統括                               |
|  +-- 封じ込め・根絶の技術判断                               |
|  +-- フォレンジック作業の監督                               |
|                                                          |
|  Communications Lead                                     |
|  +-- 社内外への情報発信                                    |
|  +-- 顧客・規制当局への通知                                |
|  +-- ステータスページの更新                                |
|                                                          |
|  Scribe (記録係)                                          |
|  +-- 全てのアクションとタイムラインの記録                    |
|  +-- 意思決定の理由の文書化                                |
|  +-- ポストモーテム用データの収集                           |
|                                                          |
|  Responders                                              |
|  +-- ログ分析・フォレンジック担当                           |
|  +-- インフラ・ネットワーク担当                             |
|  +-- アプリケーション担当                                  |
|                                                          |
|  Support                                                 |
|  +-- 法務 (法的助言、証拠保全要件)                          |
|  +-- 広報 (外部コミュニケーション)                          |
|  +-- HR (内部脅威の場合)                                   |
+----------------------------------------------------------+
```

### オンコールローテーション設計

```python
"""
CSIRT オンコールローテーション管理
"""
from datetime import datetime, timedelta
from typing import Optional


class OnCallSchedule:
    """オンコールスケジュール管理"""

    def __init__(self):
        self.primary_rotation = []      # プライマリ担当者リスト
        self.secondary_rotation = []    # セカンダリ担当者リスト
        self.escalation_chain = []      # エスカレーション順序

    def get_current_oncall(self) -> dict:
        """現在のオンコール担当者を取得"""
        now = datetime.utcnow()
        week_number = now.isocalendar()[1]

        primary_idx = week_number % len(self.primary_rotation)
        secondary_idx = week_number % len(self.secondary_rotation)

        return {
            "primary": self.primary_rotation[primary_idx],
            "secondary": self.secondary_rotation[secondary_idx],
            "week": week_number,
        }

    def escalate(self, incident_id: str, current_level: int) -> dict:
        """
        エスカレーション処理

        Level 0: オンコール担当者 (5分以内に応答)
        Level 1: セキュリティチームリード (15分以内)
        Level 2: CISO / VP of Engineering (30分以内)
        Level 3: CEO / 経営会議 (1時間以内)
        """
        if current_level >= len(self.escalation_chain):
            raise ValueError("最上位までエスカレーション済み")

        target = self.escalation_chain[current_level]
        response_deadline = {
            0: timedelta(minutes=5),
            1: timedelta(minutes=15),
            2: timedelta(minutes=30),
            3: timedelta(hours=1),
        }

        return {
            "incident_id": incident_id,
            "escalation_level": current_level,
            "target": target,
            "response_deadline": str(response_deadline[current_level]),
            "notification_channels": ["pagerduty", "phone", "slack"],
        }


# 使用例
schedule = OnCallSchedule()
schedule.primary_rotation = ["tanaka", "suzuki", "sato", "yamada"]
schedule.secondary_rotation = ["kimura", "takahashi", "watanabe", "ito"]
schedule.escalation_chain = [
    {"role": "oncall", "name": "Auto-assigned"},
    {"role": "team_lead", "name": "田中太郎"},
    {"role": "ciso", "name": "鈴木花子"},
    {"role": "ceo", "name": "山田一郎"},
]

current = schedule.get_current_oncall()
print(f"今週の担当: Primary={current['primary']}, Secondary={current['secondary']}")
```

---

## 4. インシデント対応プレイブック

### プレイブックの設計原則

```
プレイブックの構成要素:

  ┌─────────────────────────────────────────────────────┐
  │  1. トリガー条件                                      │
  │     → どのアラート/イベントでこのプレイブックを使うか    │
  │                                                     │
  │  2. 初動対応 (最初の 15 分)                            │
  │     → チェックリスト形式で迷わず実行できる内容          │
  │                                                     │
  │  3. 調査手順                                          │
  │     → 何をどの順番で調べるか                           │
  │                                                     │
  │  4. 封じ込め手順                                      │
  │     → 具体的なコマンド・操作手順                       │
  │                                                     │
  │  5. エスカレーション基準                               │
  │     → どの時点で誰に連絡するか                        │
  │                                                     │
  │  6. コミュニケーションテンプレート                       │
  │     → 社内/社外への通知文テンプレート                   │
  └─────────────────────────────────────────────────────┘
```

### ランサムウェア対応プレイブック

```python
"""
ランサムウェアインシデント対応プレイブック

トリガー:
- EDR がランサムウェア活動を検知
- ユーザーからの「ファイルが開けない」報告
- ランサムノートの発見
"""

RANSOMWARE_PLAYBOOK = {
    "name": "Ransomware Response",
    "severity": "SEV-1",
    "owner": "Security Team",
    "last_updated": "2025-03-15",
    "steps": [
        {
            "phase": "Immediate (0-15 min)",
            "actions": [
                "IC を任命し、インシデントチャンネル (Slack/Teams) を開設",
                "ランサムノートの内容をスクリーンショットで記録",
                "影響を受けたシステムのリストを作成開始",
                "暗号化されたファイルの拡張子を記録",
                "マルウェアのハッシュ値を取得 (VirusTotal で確認)",
                "全社への注意喚起: 不審なファイルを開かない",
            ],
        },
        {
            "phase": "Containment (15-60 min)",
            "actions": [
                "感染ホストをネットワークから隔離 (SG 変更 or ケーブル抜去)",
                "Active Directory の特権アカウントを無効化",
                "バックアップシステムへのネットワーク接続を遮断",
                "C2 通信先のドメイン/IP をファイアウォールでブロック",
                "影響を受けていないシステムのスナップショットを取得",
                "EDR で全エンドポイントのスキャンを実行",
            ],
        },
        {
            "phase": "Investigation (1-4 hours)",
            "actions": [
                "マルウェアの初期侵入経路を特定 (メール/RDP/脆弱性)",
                "フォレンジックイメージを取得 (ディスク + メモリ)",
                "横展開 (ラテラルムーブメント) の範囲を特定",
                "侵害の時系列タイムラインを構築",
                "No More Ransom でデクリプタの有無を確認",
            ],
        },
        {
            "phase": "Eradication (4-24 hours)",
            "actions": [
                "全感染ホストを再構築 (クリーンインストール)",
                "侵害された認証情報を全てリセット",
                "パッチ適用・脆弱性修正",
                "バックドアの検出と除去",
                "Kerberos チケット (krbtgt) のリセット (AD 環境の場合)",
            ],
        },
        {
            "phase": "Recovery (24-72 hours)",
            "actions": [
                "クリーンなバックアップからデータ復元",
                "バックアップの整合性を検証",
                "段階的にサービスを復旧 (重要度順)",
                "EDR/IDS の監視を強化 (最低 30 日間)",
                "ユーザーへの全パスワード変更指示",
            ],
        },
    ],
    "communication_templates": {
        "internal": "全社員へ: セキュリティインシデントを検知し対応中です。不審なメールを開かず、IT部門の指示に従ってください。",
        "customer": "お客様へ: 現在セキュリティインシデントの調査を行っています。お客様データへの影響については調査完了後速やかにお知らせします。",
        "regulatory": "監督機関御中: [組織名]においてセキュリティインシデントが発生し、現在対応中です。詳細は追って報告いたします。",
    },
    "do_not": [
        "身代金を支払わない (法務と相談の上判断)",
        "攻撃者と直接コンタクトしない",
        "感染システムを電源オフしない (メモリ証拠が消失)",
        "個人の判断で情報を公開しない",
    ],
}
```

### AWS での自動封じ込め

```python
"""
AWS 環境でのインシデント自動封じ込めスクリプト
GuardDuty 検知をトリガーに EC2 インスタンスを自動隔離する
"""
import boto3
import json
from datetime import datetime
from typing import Optional


class AWSIncidentContainment:
    """AWS インシデント封じ込め自動化クラス"""

    def __init__(self, isolation_sg: str, forensic_bucket: str):
        self.ec2 = boto3.client('ec2')
        self.ssm = boto3.client('ssm')
        self.s3 = boto3.client('s3')
        self.sns = boto3.client('sns')
        self.isolation_sg = isolation_sg
        self.forensic_bucket = forensic_bucket

    def contain_ec2_instance(
        self,
        instance_id: str,
        finding_id: str,
        preserve_memory: bool = True,
    ) -> dict:
        """
        EC2 インスタンスを自動隔離する

        手順:
        1. 現在のセキュリティグループを記録 (復旧用)
        2. 隔離用 SG に変更 (全通信拒否)
        3. EBS スナップショットを取得 (証拠保全)
        4. オプションでメモリダンプを取得
        """
        timestamp = datetime.utcnow().isoformat()
        results = {"instance_id": instance_id, "finding_id": finding_id}

        # 1. 現在の SG を記録
        instance = self.ec2.describe_instances(
            InstanceIds=[instance_id]
        )
        current_sgs = [
            sg['GroupId']
            for sg in instance['Reservations'][0]['Instances'][0]['SecurityGroups']
        ]

        # タグに元の SG を保存
        self.ec2.create_tags(
            Resources=[instance_id],
            Tags=[
                {'Key': 'IncidentId', 'Value': finding_id},
                {'Key': 'OriginalSecurityGroups', 'Value': ','.join(current_sgs)},
                {'Key': 'IsolatedAt', 'Value': timestamp},
                {'Key': 'IsolatedBy', 'Value': 'auto-containment'},
            ],
        )

        # 2. 隔離 SG に変更 (ネットワーク隔離)
        self.ec2.modify_instance_attribute(
            InstanceId=instance_id,
            Groups=[self.isolation_sg],
        )
        results['network_isolated'] = True

        # 3. EBS スナップショット取得 (証拠保全)
        volumes = self.ec2.describe_volumes(
            Filters=[{
                'Name': 'attachment.instance-id',
                'Values': [instance_id],
            }]
        )
        snapshot_ids = []
        for vol in volumes['Volumes']:
            snapshot = self.ec2.create_snapshot(
                VolumeId=vol['VolumeId'],
                Description=f"Forensic snapshot - Incident {finding_id}",
                TagSpecifications=[{
                    'ResourceType': 'snapshot',
                    'Tags': [
                        {'Key': 'Purpose', 'Value': 'forensic'},
                        {'Key': 'IncidentId', 'Value': finding_id},
                        {'Key': 'SourceVolume', 'Value': vol['VolumeId']},
                    ],
                }],
            )
            snapshot_ids.append(snapshot['SnapshotId'])
        results['forensic_snapshots'] = snapshot_ids

        # 4. メモリダンプ (オプション)
        if preserve_memory:
            try:
                self.ssm.send_command(
                    InstanceIds=[instance_id],
                    DocumentName='AWS-RunShellScript',
                    Parameters={
                        'commands': [
                            'apt-get install -y lime-forensics 2>/dev/null || true',
                            'insmod /lib/modules/$(uname -r)/lime.ko '
                            f'"path=/tmp/memdump-{finding_id}.lime format=lime"',
                        ],
                    },
                )
                results['memory_dump_initiated'] = True
            except Exception as e:
                results['memory_dump_error'] = str(e)

        # 5. 通知
        self.sns.publish(
            TopicArn='arn:aws:sns:ap-northeast-1:123456:security-incidents',
            Subject=f'[SEV-1] EC2 Instance Isolated: {instance_id}',
            Message=json.dumps(results, indent=2),
        )

        results['status'] = 'isolated'
        results['original_sgs'] = current_sgs
        return results

    def contain_iam_user(self, username: str, finding_id: str) -> dict:
        """
        侵害された IAM ユーザーを無効化する

        手順:
        1. アクセスキーを無効化
        2. インラインでDenyAllポリシーをアタッチ
        3. アクティブセッションを無効化
        """
        iam = boto3.client('iam')
        results = {"username": username, "finding_id": finding_id}

        # 1. 全アクセスキーを無効化
        keys = iam.list_access_keys(UserName=username)
        for key in keys['AccessKeyMetadata']:
            iam.update_access_key(
                UserName=username,
                AccessKeyId=key['AccessKeyId'],
                Status='Inactive',
            )
        results['keys_disabled'] = len(keys['AccessKeyMetadata'])

        # 2. DenyAll ポリシーをアタッチ
        deny_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateLessThan": {
                        "aws:TokenIssueTime": datetime.utcnow().isoformat() + "Z"
                    }
                },
            }],
        })
        iam.put_user_policy(
            UserName=username,
            PolicyName=f'IncidentContainment-{finding_id}',
            PolicyDocument=deny_policy,
        )
        results['deny_policy_attached'] = True

        # 3. コンソールパスワードを無効化
        try:
            iam.delete_login_profile(UserName=username)
            results['console_access_disabled'] = True
        except iam.exceptions.NoSuchEntityException:
            results['console_access_disabled'] = 'N/A (no console access)'

        return results


# 使用例
containment = AWSIncidentContainment(
    isolation_sg='sg-isolation-xxxxxxxx',
    forensic_bucket='forensic-evidence-bucket',
)

# EC2 隔離
result = containment.contain_ec2_instance(
    instance_id='i-0abc123def456789',
    finding_id='FINDING-2025-001',
)
print(json.dumps(result, indent=2))
```

---

## 5. フォレンジック調査

### フォレンジック手順の全体像

```
+----------------------------------------------------------+
|              デジタルフォレンジック手順                       |
|----------------------------------------------------------|
|                                                          |
|  1. 証拠保全 (Preservation)                               |
|     +-- ディスクイメージの取得 (dd, FTK Imager)            |
|     +-- メモリダンプの取得 (LiME, WinPmem, Volatility)    |
|     +-- ネットワークキャプチャ (tcpdump, Wireshark)        |
|     +-- ハッシュ値の記録 (SHA-256)                        |
|     +-- Chain of Custody (証拠の連鎖) を文書化             |
|                                                          |
|  2. タイムライン分析 (Analysis)                            |
|     +-- ログの時系列整理                                  |
|     +-- ファイルシステムのタイムスタンプ分析                 |
|     +-- 攻撃者の行動を時系列で再構築                       |
|     +-- IOC のクロスリファレンス                            |
|                                                          |
|  3. マルウェア分析 (Malware Analysis)                     |
|     +-- 静的解析: ハッシュ確認、文字列抽出                  |
|     +-- 動的解析: サンドボックス実行                        |
|     +-- IOC (Indicator of Compromise) の抽出              |
|     +-- YARA ルールによるパターンマッチ                     |
|                                                          |
|  4. 報告書作成 (Reporting)                                |
|     +-- 発見事項のまとめ                                  |
|     +-- 根本原因の特定                                    |
|     +-- 再発防止策の提言                                  |
|     +-- 法的手続きに使える形式での文書化                    |
+----------------------------------------------------------+
```

### 証拠保全 -- Chain of Custody

```
Chain of Custody (証拠の連鎖) の重要性:

  法的に有効な証拠とするために必要な記録:
  ┌────────────────────────────────────────────────────┐
  │  1. 誰が証拠を取得したか (取得者名)                   │
  │  2. いつ取得したか (日時・タイムゾーン)                │
  │  3. どこから取得したか (デバイス・場所)                │
  │  4. どのように取得したか (ツール・手順)                │
  │  5. 取得後の保管場所・アクセス者の記録                 │
  │  6. 証拠の完全性の証明 (ハッシュ値)                    │
  └────────────────────────────────────────────────────┘
```

### ログ分析の実践

```bash
# CloudTrail ログから不審な活動を検索

# 権限昇格の試行を検索
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=CreateRole \
  --start-time "2025-01-01T00:00:00Z" \
  --end-time "2025-01-02T00:00:00Z"

# 特定 IP からの全アクティビティ
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceType,AttributeValue=AWS::IAM::User \
  --start-time "2025-01-01" \
  --max-results 50

# Athena で CloudTrail を高速検索
# 大量のログを分析する場合に有効
cat << 'SQL'
SELECT
  eventtime,
  eventname,
  sourceipaddress,
  useridentity.arn,
  errorcode
FROM cloudtrail_logs
WHERE eventtime BETWEEN '2025-01-01' AND '2025-01-02'
  AND sourceipaddress = '203.0.113.50'
ORDER BY eventtime
SQL
```

```python
"""
複数ソースからインシデントタイムラインを構築するスクリプト
"""
import json
from datetime import datetime
from typing import Any


class IncidentTimeline:
    """インシデントタイムラインビルダー"""

    def __init__(self):
        self.events = []

    def add_cloudtrail_events(self, events: list[dict]) -> None:
        """CloudTrail イベントを追加"""
        for event in events:
            self.events.append({
                'timestamp': event['EventTime'],
                'source': 'CloudTrail',
                'action': event['EventName'],
                'actor': event.get('Username', 'unknown'),
                'ip': event.get('SourceIPAddress'),
                'details': event.get('Resources', []),
                'severity': self._classify_event(event['EventName']),
            })

    def add_vpc_flow_logs(self, logs: list[dict]) -> None:
        """VPC Flow Log イベントを追加"""
        for log in logs:
            self.events.append({
                'timestamp': log['timestamp'],
                'source': 'VPC Flow',
                'action': (
                    f"{log['srcaddr']}:{log['srcport']} -> "
                    f"{log['dstaddr']}:{log['dstport']}"
                ),
                'actor': log['srcaddr'],
                'details': {
                    'action': log['action'],
                    'bytes': log['bytes'],
                    'protocol': log.get('protocol'),
                },
                'severity': 'info',
            })

    def add_application_logs(self, logs: list[dict]) -> None:
        """アプリケーションログを追加"""
        for log in logs:
            self.events.append({
                'timestamp': log.get('timestamp'),
                'source': 'Application',
                'action': log.get('event', 'unknown'),
                'actor': log.get('userId', log.get('sourceIp', 'unknown')),
                'details': log.get('details', {}),
                'severity': log.get('level', 'info').lower(),
            })

    def build(self) -> list[dict]:
        """時系列順にソートしたタイムラインを返す"""
        self.events.sort(key=lambda x: x['timestamp'])
        return self.events

    def get_suspicious_events(self) -> list[dict]:
        """不審なイベントのみをフィルタ"""
        return [
            e for e in self.build()
            if e['severity'] in ('high', 'critical')
        ]

    def export_csv(self, filepath: str) -> None:
        """タイムラインを CSV でエクスポート"""
        import csv
        timeline = self.build()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'source', 'action', 'actor', 'ip', 'severity'],
            )
            writer.writeheader()
            for event in timeline:
                writer.writerow({
                    'timestamp': event['timestamp'],
                    'source': event['source'],
                    'action': event['action'],
                    'actor': event['actor'],
                    'ip': event.get('ip', ''),
                    'severity': event['severity'],
                })

    @staticmethod
    def _classify_event(event_name: str) -> str:
        """イベント名から重大度を分類"""
        critical_events = {
            'DeleteTrail', 'StopLogging', 'CreateUser',
            'CreateAccessKey', 'PutBucketPolicy',
            'AuthorizeSecurityGroupIngress',
        }
        high_events = {
            'CreateRole', 'AttachRolePolicy', 'PutUserPolicy',
            'ModifyInstanceAttribute', 'RunInstances',
        }
        if event_name in critical_events:
            return 'critical'
        if event_name in high_events:
            return 'high'
        return 'info'


# 使用例
timeline = IncidentTimeline()
timeline.add_cloudtrail_events([
    {
        'EventTime': '2025-03-15T14:30:00Z',
        'EventName': 'CreateAccessKey',
        'Username': 'compromised-user',
        'SourceIPAddress': '203.0.113.50',
    },
    {
        'EventTime': '2025-03-15T14:35:00Z',
        'EventName': 'PutBucketPolicy',
        'Username': 'compromised-user',
        'SourceIPAddress': '203.0.113.50',
        'Resources': [{'ARN': 'arn:aws:s3:::sensitive-data'}],
    },
])

suspicious = timeline.get_suspicious_events()
for event in suspicious:
    print(f"[{event['severity'].upper()}] {event['timestamp']} "
          f"{event['action']} by {event['actor']}")
```

---

## 6. SOAR (Security Orchestration, Automation and Response)

### SOAR の全体像

```
SOAR の役割:

  ┌─────────────────────────────────────────────────────┐
  │  1. Orchestration (オーケストレーション)               │
  │     → 複数のセキュリティツールを連携                    │
  │     → SIEM + EDR + FW + Ticketing の統合              │
  │                                                     │
  │  2. Automation (自動化)                               │
  │     → 定型的な対応手順の自動実行                       │
  │     → IOC のブロックリスト自動追加                     │
  │     → 封じ込めアクションの自動実行                     │
  │                                                     │
  │  3. Response (対応)                                   │
  │     → プレイブックに基づく標準化された対応              │
  │     → ケース管理とコラボレーション                     │
  │     → メトリクスの自動収集                             │
  └─────────────────────────────────────────────────────┘

主要 SOAR ツール:

  ツール          | 種別       | 特徴
  ──────────────┼──────────┼─────────────────────
  Palo Alto XSOAR| 商用       | 豊富な統合、ML 搭載
  Splunk SOAR    | 商用       | Splunk SIEM と統合
  Shuffle        | OSS        | 無料、Docker ベース
  TheHive        | OSS        | ケース管理に強み
  AWS Step Func. | クラウド    | AWS ネイティブ連携
```

### Step Functions による自動対応フロー

```python
"""
AWS Step Functions + Lambda による自動インシデント対応
GuardDuty → EventBridge → Step Functions → Lambda
"""
import json

# Step Functions のステートマシン定義
STATE_MACHINE = {
    "Comment": "GuardDuty Finding Auto-Response",
    "StartAt": "ClassifyFinding",
    "States": {
        "ClassifyFinding": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:ap-northeast-1:123456:function:classify-finding",
            "Next": "SeverityCheck",
        },
        "SeverityCheck": {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": "$.severity",
                    "NumericGreaterThanEquals": 7,
                    "Next": "AutoContain",
                },
                {
                    "Variable": "$.severity",
                    "NumericGreaterThanEquals": 4,
                    "Next": "CreateTicket",
                },
            ],
            "Default": "LogOnly",
        },
        "AutoContain": {
            "Type": "Parallel",
            "Branches": [
                {
                    "StartAt": "IsolateResource",
                    "States": {
                        "IsolateResource": {
                            "Type": "Task",
                            "Resource": "arn:aws:lambda:ap-northeast-1:123456:function:isolate",
                            "End": True,
                        },
                    },
                },
                {
                    "StartAt": "NotifyTeam",
                    "States": {
                        "NotifyTeam": {
                            "Type": "Task",
                            "Resource": "arn:aws:lambda:ap-northeast-1:123456:function:notify",
                            "End": True,
                        },
                    },
                },
                {
                    "StartAt": "PreserveEvidence",
                    "States": {
                        "PreserveEvidence": {
                            "Type": "Task",
                            "Resource": "arn:aws:lambda:ap-northeast-1:123456:function:preserve",
                            "End": True,
                        },
                    },
                },
            ],
            "Next": "CreateTicket",
        },
        "CreateTicket": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:ap-northeast-1:123456:function:create-ticket",
            "End": True,
        },
        "LogOnly": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:ap-northeast-1:123456:function:log-finding",
            "End": True,
        },
    },
}
```

---

## 7. ポストモーテム

### Blameless ポストモーテムの原則

```
Blameless ポストモーテムが重要な理由:

  NG: 「山田さんが設定を間違えた」
  OK: 「設定変更プロセスにバリデーションチェックがなかった」

  ┌─────────────────────────────────────────────────────┐
  │  Blameless ポストモーテムの原則:                       │
  │                                                     │
  │  1. 人ではなくプロセスに焦点を当てる                   │
  │  2. 「なぜそのアクションが合理的に見えたか」を理解する  │
  │  3. 再発防止策はシステム/プロセスの改善に向ける        │
  │  4. 学習の文化を醸成する                              │
  │  5. 参加者が安全に発言できる環境を確保する            │
  └─────────────────────────────────────────────────────┘
```

### ポストモーテムテンプレート

```markdown
# インシデントポストモーテム

## 基本情報
- インシデントID: INC-2025-042
- 発生日時: 2025-03-15 14:30 JST
- 検知日時: 2025-03-15 14:35 JST (MTTD: 5分)
- 解決日時: 2025-03-15 18:20 JST (MTTR: 3時間50分)
- 重大度: SEV-2
- Incident Commander: 山田太郎

## サマリ
S3 バケットの設定ミスにより、顧客データが4時間にわたり
パブリックアクセス可能な状態になっていた。

## 影響
- 影響を受けたユーザー数: 0 (外部アクセスなし)
- データ漏洩: なし (CloudTrail で確認済み)
- SLA 違反: なし
- 金銭的影響: なし

## タイムライン
- 14:30 - Terraform apply でバケットポリシーが変更
- 14:35 - AWS Config ルール違反を検知、アラート発報
- 14:40 - IC がインシデント宣言、対応開始
- 14:50 - パブリックアクセスブロックを手動で有効化 (封じ込め)
- 15:30 - CloudTrail でアクセスログを分析
- 16:00 - 外部からのアクセスがないことを確認
- 18:00 - Terraform コードの修正とデプロイ
- 18:20 - インシデントクローズ

## 根本原因
Terraform モジュールの更新時に、S3 パブリックアクセスブロック
のリソースが誤って削除された。PR レビューで見落とされた。

## 寄与因子
1. Terraform の plan 出力が長く、削除されたリソースを見落とした
2. IaC のセキュリティスキャン (tfsec) が CI に未導入だった
3. S3 SCP による組織レベルの制御がなかった

## うまくいったこと
- AWS Config による自動検知が 5 分以内に機能した
- IC の迅速な判断で封じ込めが 20 分以内に完了した

## 改善が必要なこと
- IaC のセキュリティスキャンがなかった
- PR レビューのセキュリティ観点チェックリストがなかった

## 再発防止策 (Action Items)
| ID | 対策 | 担当 | 期限 | 状態 |
|----|------|------|------|------|
| 1 | tfsec/Checkov を CI/CD に統合 | 鈴木 | 2025-03-22 | TODO |
| 2 | AWS Config の自動修復ルールを追加 | 佐藤 | 2025-03-29 | TODO |
| 3 | S3 SCP で組織全体のパブリックアクセスを禁止 | 田中 | 2025-04-05 | TODO |
| 4 | PR レビューチェックリストにセキュリティ項目を追加 | 山田 | 2025-03-19 | TODO |
```

### インシデントメトリクスの追跡

```python
"""
インシデント対応メトリクスの収集と可視化
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class IncidentMetrics:
    """インシデントメトリクス"""
    incident_id: str
    severity: int
    detected_at: datetime
    responded_at: datetime
    contained_at: datetime
    resolved_at: datetime
    root_cause: str
    category: str

    @property
    def mttd(self) -> timedelta:
        """Mean Time to Detect (平均検知時間)"""
        return self.detected_at - self.detected_at  # 実際は発生時刻から

    @property
    def mttr(self) -> timedelta:
        """Mean Time to Respond (平均対応時間)"""
        return self.responded_at - self.detected_at

    @property
    def mttc(self) -> timedelta:
        """Mean Time to Contain (平均封じ込め時間)"""
        return self.contained_at - self.detected_at

    @property
    def mttre(self) -> timedelta:
        """Mean Time to Resolve (平均解決時間)"""
        return self.resolved_at - self.detected_at


def calculate_team_metrics(incidents: list[IncidentMetrics]) -> dict:
    """チーム全体のメトリクスを集計"""
    if not incidents:
        return {}

    total = len(incidents)
    avg_mttr = sum(
        (i.mttr.total_seconds() for i in incidents), 0
    ) / total
    avg_mttc = sum(
        (i.mttc.total_seconds() for i in incidents), 0
    ) / total

    severity_counts = {}
    for i in incidents:
        severity_counts[f'SEV-{i.severity}'] = (
            severity_counts.get(f'SEV-{i.severity}', 0) + 1
        )

    category_counts = {}
    for i in incidents:
        category_counts[i.category] = (
            category_counts.get(i.category, 0) + 1
        )

    return {
        'total_incidents': total,
        'avg_mttr_minutes': round(avg_mttr / 60, 1),
        'avg_mttc_minutes': round(avg_mttc / 60, 1),
        'by_severity': severity_counts,
        'by_category': category_counts,
    }
```

---

## 8. 机上演習 (Tabletop Exercise)

### 演習の設計と実施

```
机上演習の進め方:

  ┌─────────────────────────────────────────────────────┐
  │  準備 (演習の 2 週間前)                                │
  │  +-- シナリオの作成 (現実的な脅威に基づく)              │
  │  +-- 参加者の選定 (CSIRT + 関連部門)                   │
  │  +-- ファシリテーターの任命                            │
  │  +-- 注入イベント (Inject) の準備                      │
  │                                                     │
  │  実施 (2-4 時間)                                      │
  │  +-- シナリオの提示                                   │
  │  +-- 各フェーズでの判断と対応を議論                    │
  │  +-- Inject で状況を変化させる                         │
  │  +-- 参加者全員の発言を促す                            │
  │                                                     │
  │  事後 (1 週間以内)                                     │
  │  +-- 発見事項のまとめ                                 │
  │  +-- ギャップの特定                                   │
  │  +-- 改善 Action Items の策定                          │
  │  +-- 次回演習の計画                                   │
  └─────────────────────────────────────────────────────┘
```

```python
"""
机上演習シナリオジェネレータ
"""
import random


class TabletopScenario:
    """机上演習シナリオの生成"""

    SCENARIOS = [
        {
            "title": "ランサムウェア攻撃",
            "description": "月曜朝、複数の社員から「ファイルが開けない」と報告。"
                          "デスクトップにランサムノートが表示されている。",
            "injects": [
                "感染は Active Directory 経由で拡大中。ドメインコントローラーにも到達。",
                "攻撃者から「48時間以内にビットコインで身代金を支払え」とメール。",
                "記者から「御社がサイバー攻撃を受けたという情報がある」と問い合わせ。",
                "バックアップサーバーも暗号化されていることが判明。",
            ],
            "discussion_points": [
                "最初の 15 分で何をすべきか?",
                "ネットワーク全体を遮断するか? 部分的に隔離するか?",
                "身代金を支払うべきか? 法的・倫理的な観点は?",
                "顧客・取引先への通知のタイミングと内容は?",
                "バックアップが使えない場合の復旧戦略は?",
            ],
        },
        {
            "title": "内部犯行によるデータ持ち出し",
            "description": "退職予定の社員が大量のファイルを個人デバイスにコピーしている"
                          "というアラートが DLP ツールから発報。",
            "injects": [
                "該当社員は顧客リストと技術文書 500 件をダウンロード済み。",
                "社員は「業務に必要な作業」と主張。",
                "競合他社に転職予定であることが判明。",
                "過去 3 ヶ月分のアクセスログを確認すると、不自然なパターンが見つかる。",
            ],
            "discussion_points": [
                "HR とどのタイミングで連携するか?",
                "法務部門をいつ巻き込むか?",
                "社員のアクセス権をいつ無効化するか?",
                "証拠保全のために何をすべきか?",
                "法的措置を取る場合の手順は?",
            ],
        },
    ]

    @classmethod
    def generate(cls, scenario_type: str = "random") -> dict:
        """シナリオを生成"""
        if scenario_type == "random":
            return random.choice(cls.SCENARIOS)
        for s in cls.SCENARIOS:
            if scenario_type.lower() in s["title"].lower():
                return s
        return cls.SCENARIOS[0]


# 使用例
scenario = TabletopScenario.generate("ランサムウェア")
print(f"シナリオ: {scenario['title']}")
print(f"状況: {scenario['description']}")
print("\n議論ポイント:")
for i, point in enumerate(scenario['discussion_points'], 1):
    print(f"  {i}. {point}")
```

---

## 9. アンチパターン

### アンチパターン 1: 証拠を破壊してしまう

```
NG:
  → 感染サーバを即座に再起動・再構築
  → ログを消去してクリーンな状態に
  → マルウェアファイルを削除
  → 「とにかく早く復旧させろ」で証拠を上書き

  結果: 根本原因が特定できない、法的証拠が使えない、再発する

OK:
  → まずディスクイメージとメモリダンプを取得
  → ログを読み取り専用で保全 (別の安全な場所にコピー)
  → 隔離した上で調査を実施
  → Chain of Custody を記録
  → フォレンジック担当者の指示に従う
```

### アンチパターン 2: インシデント対応計画の未策定

```
NG:
  → 「インシデントが起きたら考える」
  → 連絡先リストが存在しない
  → 役割分担が決まっていない
  → 計画はあるが一度も訓練していない

  結果: パニック、対応の遅延、意思決定の混乱

OK:
  → 対応計画を文書化し定期的に更新
  → 四半期ごとに机上演習を実施
  → 年次でレッドチーム演習を実施
  → 計画の実効性をメトリクスで検証
```

### アンチパターン 3: 全てを手動で対応する

```
NG:
  → 全ての封じ込めアクションを手動実行
  → スクリプト化されていない繰り返し作業
  → 夜間・休日は対応開始まで数時間
  → 対応者のスキルに品質が依存

  結果: 対応の遅延、ヒューマンエラー、対応品質のばらつき

OK:
  → SOAR による自動封じ込めの導入
  → プレイブックのスクリプト化
  → 低〜中リスクの自動対応フロー構築
  → 高リスクは自動検知+人間の判断で対応
```

---

## 10. 実践演習

### 演習 1: インシデント重大度の判定 (基礎)

以下のインシデントの重大度 (SEV-1 〜 SEV-4) を判定し、その理由と初動対応を述べよ。

1. 社内の開発者が誤って本番 DB のテーブルを DROP した
2. 外部のセキュリティ研究者から XSS 脆弱性の報告を受けた
3. GuardDuty が EC2 インスタンスから暗号通貨マイニング通信を検知した
4. 社員 50 名がフィッシングメールのリンクをクリックした

<details>
<summary>模範解答</summary>

1. **DROP TABLE -- SEV-1 (Critical)**
   - 理由: 本番データの消失はサービス停止に直結。データ復旧にバックアップからのリストアが必要
   - 初動: バックアップの確認、リストア開始、影響範囲の特定、ユーザーへの通知

2. **XSS 報告 -- SEV-3 (Medium)**
   - 理由: 外部研究者による責任ある開示。攻撃が実行された証拠がなければ潜在的リスク
   - 初動: 脆弱性の再現確認、WAF ルール追加 (暫定)、修正パッチ開発の優先度決定

3. **暗号通貨マイニング -- SEV-2 (High)**
   - 理由: EC2 が侵害されている。ラテラルムーブメントや追加の侵害の可能性
   - 初動: EC2 の隔離 (SG 変更)、メモリ/ディスクの証拠保全、侵入経路の調査

4. **フィッシング 50 名クリック -- SEV-2 (High)**
   - 理由: 50 名が潜在的に侵害。クレデンシャル窃取やマルウェア感染の可能性
   - 初動: 対象者全員のパスワード強制リセット、セッション無効化、対象端末のスキャン
</details>

### 演習 2: ポストモーテムの作成 (応用)

以下のシナリオに基づいて、Blameless ポストモーテムを作成せよ。

**シナリオ:** 深夜 2:00 に Web アプリケーションが全面停止。原因は、当日 18:00 にデプロイされたコード変更がメモリリークを引き起こし、6 時間かけて全インスタンスが OOM (Out of Memory) でクラッシュした。アラートは設定されていたが、Slack 通知のみでオンコール担当者は睡眠中で気づかなかった。

<details>
<summary>模範解答</summary>

```markdown
# インシデントポストモーテム

## 基本情報
- インシデントID: INC-2025-047
- 発生日時: 2025-04-10 02:00 JST
- 検知日時: 2025-04-10 02:45 JST (ユーザー報告で検知)
- 解決日時: 2025-04-10 04:30 JST
- 重大度: SEV-1
- MTTD: 45分, MTTR: 2時間30分

## サマリ
18:00 のデプロイに含まれるコード変更がメモリリークを引き起こし、
6時間後に全インスタンスが OOM でクラッシュ、サービス全面停止。

## 根本原因
新機能のコードで HTTP コネクションプールが適切に close されて
おらず、リクエストごとにメモリが蓄積。

## 寄与因子
1. メモリ使用量の閾値アラートはあったが、Slack 通知のみだった
2. 負荷テスト (長時間実行) がデプロイ前に行われなかった
3. カナリアデプロイではなく全インスタンス同時デプロイだった
4. ロールバック手順が文書化されていなかった

## うまくいったこと
- ユーザー報告後の対応は迅速だった
- ロールバック自体は 30 分で完了した

## 再発防止策
| 対策 | 担当 | 期限 |
|------|------|------|
| PagerDuty への通知チャネル追加 | インフラチーム | 1週間 |
| カナリアデプロイの導入 | SRE | 1ヶ月 |
| 長時間負荷テストの CI 追加 | QA | 2週間 |
| メモリリーク検知の自動テスト追加 | 開発チーム | 2週間 |
```
</details>

### 演習 3: 自動封じ込めスクリプトの設計 (発展)

以下の要件を満たすインシデント自動封じ込めシステムを設計せよ。

**要件:**
- GuardDuty が「S3 バケットへの匿名アクセス」を検知した場合、自動で S3 バケットのパブリックアクセスをブロックする
- 元の設定をタグに保存する (復旧用)
- 証跡として CloudWatch Logs にアクション記録を出力する
- SNS で セキュリティチームに通知する
- 誤検知の場合のロールバック手順も設計する

<details>
<summary>模範解答</summary>

```python
"""
S3 パブリックアクセス自動封じ込め
EventBridge -> Lambda でトリガーされる
"""
import boto3
import json
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    """
    GuardDuty S3 パブリックアクセス検知時の自動対応
    """
    s3 = boto3.client('s3')
    sns = boto3.client('sns')

    # GuardDuty Finding からバケット名を取得
    finding = event['detail']
    bucket_name = finding['resource']['s3BucketDetails'][0]['name']
    finding_id = finding['id']
    finding_type = finding['type']

    logger.info(f"Processing finding {finding_id} for bucket {bucket_name}")

    # 1. 現在の設定を保存
    try:
        current_config = s3.get_public_access_block(Bucket=bucket_name)
        original_config = json.dumps(
            current_config['PublicAccessBlockConfiguration']
        )
    except s3.exceptions.NoSuchPublicAccessBlockConfiguration:
        original_config = json.dumps({
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        })

    # 2. タグに元の設定を保存 (復旧用)
    s3.put_bucket_tagging(
        Bucket=bucket_name,
        Tagging={
            'TagSet': [
                {'Key': 'IncidentId', 'Value': finding_id},
                {'Key': 'OriginalPublicAccessConfig', 'Value': original_config},
                {'Key': 'ContainedAt', 'Value': datetime.utcnow().isoformat()},
                {'Key': 'ContainedBy', 'Value': 'auto-containment-lambda'},
            ],
        },
    )

    # 3. パブリックアクセスを全てブロック
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True,
            'IgnorePublicAcls': True,
            'BlockPublicPolicy': True,
            'RestrictPublicBuckets': True,
        },
    )

    # 4. ログ記録
    log_entry = {
        'action': 's3_public_access_blocked',
        'bucket': bucket_name,
        'finding_id': finding_id,
        'finding_type': finding_type,
        'original_config': original_config,
        'timestamp': datetime.utcnow().isoformat(),
    }
    logger.info(json.dumps(log_entry))

    # 5. セキュリティチームに通知
    sns.publish(
        TopicArn='arn:aws:sns:ap-northeast-1:123456:security-incidents',
        Subject=f'[AUTO] S3 Public Access Blocked: {bucket_name}',
        Message=json.dumps(log_entry, indent=2),
    )

    return {'statusCode': 200, 'body': log_entry}


def rollback_containment(bucket_name: str, incident_id: str) -> dict:
    """
    誤検知の場合のロールバック手順
    承認プロセスを経た上で実行する
    """
    s3 = boto3.client('s3')

    # タグから元の設定を取得
    tags = s3.get_bucket_tagging(Bucket=bucket_name)
    original_config = None
    for tag in tags['TagSet']:
        if tag['Key'] == 'OriginalPublicAccessConfig':
            original_config = json.loads(tag['Value'])
            break

    if not original_config:
        raise ValueError("元の設定が見つかりません")

    # 元の設定に復元
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration=original_config,
    )

    # インシデントタグを更新
    s3.put_bucket_tagging(
        Bucket=bucket_name,
        Tagging={
            'TagSet': [
                {'Key': 'IncidentId', 'Value': incident_id},
                {'Key': 'RolledBackAt', 'Value': datetime.utcnow().isoformat()},
                {'Key': 'Reason', 'Value': 'false_positive'},
            ],
        },
    )

    return {'status': 'rolled_back', 'bucket': bucket_name}
```
</details>

---

## 11. FAQ

### Q1. インシデント対応チームの最小構成は?

小規模組織では、IC (兼 Technical Lead) 1名、Responder 1-2名、Communications Lead 1名の 3-4名が最小構成である。重要なのは役割の明確化と、緊急時の連絡手段の確保である。外部の CSIRT サービス (MDR: Managed Detection and Response) を契約し、不足する専門性を補完するのも有効である。チームが小さい場合でも、プレイブックの整備と定期的な訓練を行うことで対応品質を担保できる。

### Q2. フォレンジック調査を外部委託すべきタイミングは?

以下のいずれかに該当する場合は外部委託を検討すべきである:
- 法的手続き (訴訟、法執行機関への報告) が想定される場合 -- 証拠の法的有効性を確保するため
- 組織内にフォレンジックの専門スキルがない場合
- 内部犯行の可能性がある場合 -- 内部メンバーによる証拠隠滅リスクを排除
- インシデントの規模が大きく内部リソースでは対応しきれない場合
- サイバー保険の適用条件として外部調査が求められる場合

### Q3. インシデント後の情報公開はどこまで行うべきか?

個人データの漏洩がある場合は GDPR (72時間以内) や個人情報保護法に基づく通知義務がある。それ以外でも、影響を受けた顧客・パートナーには誠実に開示すべきである。公開する情報は「何が起きたか」「影響範囲」「取った対策」「今後の防止策」を含め、攻撃者に利する技術的詳細 (使用されたエクスプロイト、内部ネットワーク構成等) は含めない。透明性は信頼を維持する上で重要だが、法務と広報と連携して開示範囲を慎重に決定する。

### Q4. 身代金を支払うべきか?

一般的に推奨されない。理由: (1) 支払ってもデータが復元される保証がない、(2) 犯罪組織の資金源となる、(3) 再攻撃のリスクが高まる、(4) 一部の国では制裁対象組織への支払いが法的に問題になる。ただし、人命に関わる場合やバックアップが完全に失われた場合は、法務・経営層と協議の上で判断する。FBI や警察庁サイバー犯罪対策課への相談も検討する。

### Q5. インシデント対応の成熟度を評価するには?

NIST の Cybersecurity Framework (CSF) の「Respond」カテゴリや、SANS の Incident Response Maturity Model を基に評価する。主要な評価指標は: (1) MTTD/MTTR の推移、(2) プレイブックのカバー率、(3) 訓練の実施頻度と結果、(4) 自動化の導入率、(5) ポストモーテムの実施率と Action Items の完了率。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 対応フロー | 準備→検知→封じ込め→根絶→復旧→教訓の6段階 |
| 重大度 | SEV-1 (15分) から SEV-4 (1週間) の対応 SLA |
| CSIRT | IC・Technical Lead・Communications Lead・Scribe の役割明確化 |
| 封じ込め | ネットワーク隔離→証拠保全→影響範囲確認 |
| フォレンジック | ディスク/メモリイメージ取得、Chain of Custody、タイムライン分析 |
| SOAR | 自動封じ込め・プレイブック実行による対応速度向上 |
| ポストモーテム | 非難なし (Blameless)、根本原因と再発防止策に集中 |
| メトリクス | MTTD・MTTR・MTTC を継続的に計測し改善 |
| 訓練 | 四半期の机上演習、年次のレッドチーム演習 |

---

## 次に読むべきガイド

- [監視/ログ](./01-monitoring-logging.md) -- インシデント検知の基盤となる監視体制
- [コンプライアンス](./02-compliance.md) -- インシデント報告の法的義務 (GDPR 72時間ルール等)
- [セキュリティ文化](./03-security-culture.md) -- 組織全体のセキュリティ意識向上と DevSecOps
- [認証と認可の基礎](../../authentication-and-authorization/docs/00-fundamentals/00-authentication-vs-authorization.md) -- 侵害後のクレデンシャル管理
- [クラウドセキュリティ](../05-cloud-security/01-aws-security.md) -- AWS 環境でのセキュリティ対応

---

## 参考文献

1. **NIST SP 800-61 Rev.2 -- Computer Security Incident Handling Guide** -- https://csrc.nist.gov/publications/detail/sp/800-61/rev-2/final
2. **SANS Incident Handler's Handbook** -- https://www.sans.org/white-papers/incident-handlers-handbook/
3. **PagerDuty Incident Response Documentation** -- https://response.pagerduty.com/
4. **Google SRE Book -- Managing Incidents** -- https://sre.google/sre-book/managing-incidents/
5. **MITRE ATT&CK Framework** -- https://attack.mitre.org/ -- 攻撃者の戦術・技術・手順 (TTP) の分類体系
6. **IBM Cost of a Data Breach Report 2024** -- https://www.ibm.com/reports/data-breach
