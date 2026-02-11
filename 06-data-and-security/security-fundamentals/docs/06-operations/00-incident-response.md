# インシデント対応

> インシデント対応フローの設計、CSIRT の組織体制、フォレンジック調査の手法まで、セキュリティインシデント発生時に迅速かつ正確に対応するためのガイド

## この章で学ぶこと

1. **インシデント対応フロー** — 準備から復旧・教訓までの6段階プロセス
2. **CSIRT の組織と役割** — インシデント対応チームの構成と責任分担
3. **フォレンジック調査** — 証拠保全、タイムライン分析、根本原因の特定手法

---

## 1. インシデント対応の全体像

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
```

### 各フェーズの詳細

```
+----------------------------------------------------------+
|  Phase 1: 準備 (Preparation)                              |
|  +-- インシデント対応計画の策定                              |
|  +-- CSIRT メンバーの任命と訓練                             |
|  +-- 連絡先リスト (エスカレーションパス) の整備              |
|  +-- ツール・環境の整備 (フォレンジックキット)               |
|  +-- 定期的な机上演習 (Tabletop Exercise)                  |
|----------------------------------------------------------|
|  Phase 2: 検知と分析 (Detection & Analysis)               |
|  +-- アラートのトリアージ (真偽判定)                        |
|  +-- 影響範囲の特定                                       |
|  +-- 重大度の判定 (Severity Level)                         |
|  +-- タイムライン構築の開始                                |
|----------------------------------------------------------|
|  Phase 3: 封じ込め (Containment)                          |
|  +-- 短期: 被害拡大の即座の阻止                            |
|  +-- 長期: 恒久対策までの暫定対応                          |
|  +-- 証拠保全 (フォレンジックイメージ取得)                  |
|----------------------------------------------------------|
|  Phase 4: 根絶 (Eradication)                              |
|  +-- マルウェアの除去                                     |
|  +-- 脆弱性の修正                                        |
|  +-- 侵害されたアカウントの無効化と再作成                   |
|----------------------------------------------------------|
|  Phase 5: 復旧 (Recovery)                                 |
|  +-- システムの段階的な復旧                                |
|  +-- 監視強化期間の設定                                   |
|  +-- 正常性の確認                                        |
|----------------------------------------------------------|
|  Phase 6: 教訓 (Lessons Learned)                          |
|  +-- ポストモーテム (事後レビュー)                          |
|  +-- 対応手順の改善                                       |
|  +-- 再発防止策の実施                                     |
+----------------------------------------------------------+
```

---

## 2. 重大度レベルの定義

### インシデント重大度

| レベル | 定義 | 対応時間 | 例 |
|--------|------|---------|-----|
| SEV-1 (Critical) | サービス停止・大規模データ漏洩 | 15分以内に対応開始 | ランサムウェア、DB 漏洩 |
| SEV-2 (High) | サービス劣化・限定的データ漏洩 | 1時間以内に対応開始 | 不正アクセス、DDoS |
| SEV-3 (Medium) | 潜在的リスク・小規模影響 | 24時間以内に対応開始 | フィッシング成功、マルウェア検知 |
| SEV-4 (Low) | 軽微な問題・情報収集 | 1週間以内に対応 | ポートスキャン、誤設定 |

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
|                                                          |
|  Technical Lead                                          |
|  +-- 技術的な調査・分析の統括                               |
|  +-- 封じ込め・根絶の技術判断                               |
|                                                          |
|  Communications Lead                                     |
|  +-- 社内外への情報発信                                    |
|  +-- 顧客・規制当局への通知                                |
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

---

## 4. インシデント対応プレイブック

### ランサムウェア対応プレイブック

```python
"""
ランサムウェアインシデント対応プレイブック
"""

PLAYBOOK = {
    "name": "Ransomware Response",
    "severity": "SEV-1",
    "steps": [
        {
            "phase": "Detection",
            "actions": [
                "ランサムノートの内容を記録 (スクリーンショット)",
                "影響を受けたシステムのリストを作成",
                "暗号化されたファイルの拡張子を記録",
                "マルウェアのハッシュ値を取得",
            ],
        },
        {
            "phase": "Containment",
            "actions": [
                "感染ホストをネットワークから隔離 (ケーブル抜去/SG変更)",
                "Active Directory の特権アカウントを無効化",
                "バックアップシステムへのネットワーク接続を遮断",
                "C2 通信先のドメイン/IP をファイアウォールでブロック",
                "影響を受けていないシステムのスナップショットを取得",
            ],
        },
        {
            "phase": "Eradication",
            "actions": [
                "マルウェアの初期侵入経路を特定",
                "フォレンジックイメージを取得",
                "全感染ホストを再構築 (クリーンインストール)",
                "侵害された認証情報を全てリセット",
                "パッチ適用・脆弱性修正",
            ],
        },
        {
            "phase": "Recovery",
            "actions": [
                "クリーンなバックアップからデータ復元",
                "バックアップの整合性を検証",
                "段階的にサービスを復旧",
                "EDR/IDS の監視を強化 (30日間)",
            ],
        },
    ],
}
```

### AWS での自動封じ込め

```python
import boto3
from datetime import datetime

def auto_contain_ec2(instance_id: str, finding_id: str):
    """GuardDuty 検知をトリガーに EC2 を自動隔離"""
    ec2 = boto3.client('ec2')

    # 隔離用セキュリティグループ (全通信拒否)
    ISOLATION_SG = 'sg-isolation-xxxxxxxx'

    # 1. 現在の SG を記録 (復旧用)
    instance = ec2.describe_instances(InstanceIds=[instance_id])
    current_sgs = [
        sg['GroupId']
        for sg in instance['Reservations'][0]['Instances'][0]['SecurityGroups']
    ]

    # タグに元の SG を保存
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            {'Key': 'IncidentId', 'Value': finding_id},
            {'Key': 'OriginalSecurityGroups', 'Value': ','.join(current_sgs)},
            {'Key': 'IsolatedAt', 'Value': datetime.utcnow().isoformat()},
        ],
    )

    # 2. 隔離 SG に変更 (ネットワーク隔離)
    ec2.modify_instance_attribute(
        InstanceId=instance_id,
        Groups=[ISOLATION_SG],
    )

    # 3. EBS スナップショット取得 (証拠保全)
    volumes = ec2.describe_volumes(
        Filters=[{'Name': 'attachment.instance-id', 'Values': [instance_id]}]
    )
    for vol in volumes['Volumes']:
        ec2.create_snapshot(
            VolumeId=vol['VolumeId'],
            Description=f"Forensic snapshot - Incident {finding_id}",
            TagSpecifications=[{
                'ResourceType': 'snapshot',
                'Tags': [
                    {'Key': 'Purpose', 'Value': 'forensic'},
                    {'Key': 'IncidentId', 'Value': finding_id},
                ],
            }],
        )

    # 4. メモリダンプ用に SSM コマンド送信 (隔離前に実行することもある)
    ssm = boto3.client('ssm')
    ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={
            'commands': [
                'dd if=/dev/mem of=/tmp/memdump.raw bs=1M 2>/dev/null || true',
            ]
        },
    )

    return {
        'status': 'isolated',
        'instance_id': instance_id,
        'original_sgs': current_sgs,
    }
```

---

## 5. フォレンジック調査

### フォレンジック手順

```
+----------------------------------------------------------+
|              デジタルフォレンジック手順                       |
|----------------------------------------------------------|
|                                                          |
|  1. 証拠保全 (Preservation)                               |
|     +-- ディスクイメージの取得 (dd, FTK Imager)            |
|     +-- メモリダンプの取得 (LiME, WinPmem)                |
|     +-- ネットワークキャプチャ (tcpdump)                   |
|     +-- ハッシュ値の記録 (SHA-256)                        |
|     +-- Chain of Custody (証拠の連鎖) を文書化             |
|                                                          |
|  2. タイムライン分析 (Analysis)                            |
|     +-- ログの時系列整理                                  |
|     +-- ファイルシステムのタイムスタンプ分析                 |
|     +-- 攻撃者の行動を時系列で再構築                       |
|                                                          |
|  3. マルウェア分析 (Malware Analysis)                     |
|     +-- 静的解析: ハッシュ確認、文字列抽出                  |
|     +-- 動的解析: サンドボックス実行                        |
|     +-- IOC (Indicator of Compromise) の抽出              |
|                                                          |
|  4. 報告書作成 (Reporting)                                |
|     +-- 発見事項のまとめ                                  |
|     +-- 根本原因の特定                                    |
|     +-- 再発防止策の提言                                  |
+----------------------------------------------------------+
```

### ログ分析の実践

```bash
# CloudTrail ログから不審な活動を検索
# 権限昇格の試行
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=CreateRole \
  --start-time "2025-01-01T00:00:00Z" \
  --end-time "2025-01-02T00:00:00Z"

# 特定 IP からの全アクティビティ
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceType,AttributeValue=AWS::IAM::User \
  --start-time "2025-01-01" \
  --max-results 50
```

```python
# タイムライン構築スクリプト
import json
from datetime import datetime

def build_timeline(cloudtrail_events, vpc_flow_logs, app_logs):
    """複数ソースからイベントタイムラインを構築"""
    timeline = []

    for event in cloudtrail_events:
        timeline.append({
            'timestamp': event['EventTime'],
            'source': 'CloudTrail',
            'action': event['EventName'],
            'actor': event.get('Username', 'unknown'),
            'ip': event.get('SourceIPAddress'),
            'details': event.get('Resources', []),
        })

    for log in vpc_flow_logs:
        timeline.append({
            'timestamp': log['timestamp'],
            'source': 'VPC Flow',
            'action': f"{log['srcaddr']}:{log['srcport']} -> {log['dstaddr']}:{log['dstport']}",
            'actor': log['srcaddr'],
            'details': {'action': log['action'], 'bytes': log['bytes']},
        })

    # 時系列でソート
    timeline.sort(key=lambda x: x['timestamp'])
    return timeline
```

---

## 6. ポストモーテム

### ポストモーテムテンプレート

```
# インシデントポストモーテム

## 基本情報
- インシデントID: INC-2025-042
- 発生日時: 2025-03-15 14:30 JST
- 検知日時: 2025-03-15 14:35 JST
- 解決日時: 2025-03-15 18:20 JST
- 重大度: SEV-2
- Incident Commander: 山田太郎

## サマリ
S3 バケットの設定ミスにより、顧客データが4時間にわたり
パブリックアクセス可能な状態になっていた。

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

## 再発防止策
1. tfsec/Checkov を CI/CD に統合 (必須チェック化)
2. AWS Config の自動修復ルールを追加
3. S3 SCP で組織全体のパブリックアクセスを禁止
```

---

## 7. アンチパターン

### アンチパターン 1: 証拠を破壊してしまう

```
NG:
  → 感染サーバを即座に再起動・再構築
  → ログを消去してクリーンな状態に
  → マルウェアファイルを削除

OK:
  → まずディスクイメージとメモリダンプを取得
  → ログを読み取り専用で保全
  → 隔離した上で調査を実施
  → Chain of Custody を記録
```

### アンチパターン 2: インシデント対応計画の未策定

```
NG:
  → 「インシデントが起きたら考える」
  → 連絡先リストが存在しない
  → 役割分担が決まっていない

OK:
  → 対応計画を文書化し定期的に更新
  → 四半期ごとに机上演習を実施
  → 年次でレッドチーム演習を実施
```

---

## 8. FAQ

### Q1. インシデント対応チームの最小構成は?

小規模組織では、IC (兼 Technical Lead) 1名、Responder 1-2名、Communications Lead 1名の 3-4名が最小構成である。重要なのは役割の明確化と、緊急時の連絡手段の確保である。外部の CSIRT サービス (MDR) を契約し、不足する専門性を補完するのも有効である。

### Q2. フォレンジック調査を外部委託すべきタイミングは?

法的手続き (訴訟、法執行機関への報告) が想定される場合は、証拠の法的有効性を確保するために外部のフォレンジック専門企業に委託すべきである。また、組織内に専門スキルがない場合や、内部犯行の可能性がある場合も外部委託が適切である。

### Q3. インシデント後の情報公開はどこまで行うべきか?

個人データの漏洩がある場合は GDPR (72時間以内) や個人情報保護法に基づく通知義務がある。それ以外でも、影響を受けた顧客・パートナーには誠実に開示すべきである。公開する情報は「何が起きたか」「影響範囲」「取った対策」「今後の防止策」を含め、攻撃者に利する技術的詳細は含めない。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 対応フロー | 準備→検知→封じ込め→根絶→復旧→教訓の6段階 |
| 重大度 | SEV-1 (15分) から SEV-4 (1週間) の対応 SLA |
| CSIRT | IC・Technical Lead・Communications Lead の役割明確化 |
| 封じ込め | ネットワーク隔離→証拠保全→影響範囲確認 |
| フォレンジック | ディスク/メモリイメージ取得、タイムライン分析 |
| ポストモーテム | 非難なし (blameless)、根本原因と再発防止策に集中 |
| 訓練 | 四半期の机上演習、年次のレッドチーム演習 |

---

## 次に読むべきガイド

- [監視/ログ](./01-monitoring-logging.md) — インシデント検知の基盤となる監視体制
- [コンプライアンス](./02-compliance.md) — インシデント報告の法的義務
- [セキュリティ文化](./03-security-culture.md) — 組織全体のセキュリティ意識向上

---

## 参考文献

1. **NIST SP 800-61 Rev.2 — Computer Security Incident Handling Guide** — https://csrc.nist.gov/publications/detail/sp/800-61/rev-2/final
2. **SANS Incident Handler's Handbook** — https://www.sans.org/white-papers/incident-handlers-handbook/
3. **PagerDuty Incident Response Documentation** — https://response.pagerduty.com/
4. **Google SRE Book — Managing Incidents** — https://sre.google/sre-book/managing-incidents/
