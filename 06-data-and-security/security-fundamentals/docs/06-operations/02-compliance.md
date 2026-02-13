# コンプライアンス

> GDPR、SOC 2、PCI DSS を中心に、セキュリティに関する法規制・業界標準への準拠方法と実装のポイントを体系的に学ぶ

## 前提知識

この章を理解するために、以下の知識が必要である。

- **暗号化の基礎** — 対称鍵暗号・非対称鍵暗号の仕組み（[暗号化](../03-cryptography/00-encryption.md) を参照）
- **アクセス制御** — RBAC/ABAC の概念（[認証と認可](../02-network-security/02-access-control.md) を参照）
- **ログと監視** — 監査ログの基本的な仕組み（[監視/ログ](./01-monitoring-logging.md) を参照）
- **ネットワークセキュリティ** — ファイアウォール、TLS の基礎

---

## この章で学ぶこと

1. **GDPR (一般データ保護規則)** — EU の個人データ保護規制と技術的対応
2. **SOC 2** — サービス組織の内部統制に関する監査報告書と準拠のポイント
3. **PCI DSS** — クレジットカード情報を扱うシステムのセキュリティ要件
4. **HIPAA** — 米国の医療情報保護規制
5. **日本の個人情報保護法** — 日本国内の法的要件
6. **コンプライアンス自動化** — 継続的コンプライアンスの実装手法
7. **監査対応の実践** — 監査準備から完了までの具体的プロセス

---

## 1. コンプライアンスの全体像

### なぜコンプライアンスが重要か

コンプライアンスは単なる法令遵守ではなく、組織のセキュリティ態勢を体系的に構築・維持するためのフレームワークである。コンプライアンスが重要な理由は以下の 4 つに集約される。

1. **法的リスクの回避** — GDPR 違反で最大 2000 万 EUR または全世界売上高の 4% の制裁金が課される。2023 年には Meta が 12 億 EUR の制裁金を受けた事例がある
2. **顧客の信頼獲得** — SOC 2 Type II レポートは B2B SaaS の商談で事実上の必須要件となっている
3. **セキュリティ態勢の底上げ** — コンプライアンス要件を満たす過程で、組織のセキュリティ基盤が体系的に整備される
4. **事業継続性の確保** — PCI DSS 非準拠のカード加盟店はカード処理が停止され、事業に直接的な打撃を受ける

### 主要な規制・基準の分類

```
+------------------------------------------------------------------+
|              セキュリティコンプライアンスの分類                       |
|------------------------------------------------------------------|
|                                                                  |
|  [法規制 (法的義務)]                                              |
|  +-- GDPR (EU 一般データ保護規則)                                 |
|  |   +-- 適用範囲: EU 域内の個人データ処理                         |
|  |   +-- 施行: 2018年5月25日                                     |
|  +-- 個人情報保護法 (日本)                                        |
|  |   +-- 2022年改正: 漏洩報告義務化、罰則強化                      |
|  +-- CCPA/CPRA (カリフォルニア州)                                 |
|  |   +-- 消費者のオプトアウト権を重視                              |
|  +-- HIPAA (米国医療情報)                                         |
|  |   +-- PHI (Protected Health Information) の保護                |
|  +-- LGPD (ブラジル一般データ保護法)                               |
|  |   +-- GDPR と類似の構造                                       |
|  +-- PIPA (韓国個人情報保護法)                                    |
|                                                                  |
|  [業界標準 (業界義務)]                                            |
|  +-- PCI DSS (カード決済)                                        |
|  |   +-- v4.0: 2024年3月31日から全面適用                          |
|  +-- FISC (金融情報システムセンター)                               |
|  |   +-- 日本の金融機関向けガイドライン                            |
|  +-- SWIFT CSP (国際送金)                                        |
|                                                                  |
|  [監査フレームワーク (信頼性証明)]                                  |
|  +-- SOC 2 Type I/II                                             |
|  |   +-- AICPA Trust Service Criteria に基づく                    |
|  +-- ISO 27001                                                   |
|  |   +-- ISMS (情報セキュリティマネジメントシステム)                 |
|  +-- ISO 27701                                                   |
|  |   +-- プライバシー情報マネジメント (PIMS)                       |
|  +-- ISO 27017 / ISO 27018                                       |
|      +-- クラウドセキュリティ / クラウドプライバシー                 |
|                                                                  |
|  [ベストプラクティス (任意)]                                       |
|  +-- NIST Cybersecurity Framework (CSF) v2.0                     |
|  +-- CIS Controls v8                                             |
|  +-- OWASP Top 10                                                |
|  +-- NIST SP 800-53 Rev.5                                        |
|  +-- COBIT 2019                                                  |
+------------------------------------------------------------------+
```

### 規制・基準の比較

| 項目 | GDPR | SOC 2 | PCI DSS | ISO 27001 | HIPAA | 個人情報保護法 |
|------|------|-------|---------|-----------|-------|------------|
| 対象 | EU 個人データ | SaaS サービス全般 | カード決済 | 情報セキュリティ全般 | 医療情報 (PHI) | 日本の個人情報 |
| 強制力 | 法律 | 顧客要求 | 業界標準 (事実上必須) | 任意 (契約要件化あり) | 法律 | 法律 |
| 罰則 | 最大 2000万EUR or 売上4% | なし (信頼喪失) | 罰金 + カード処理停止 | なし | 最大 $1.5M/年 | 最大 1億円 (法人) |
| 監査 | 規制当局 | 独立監査人 (CPA) | QSA / ISA | 認証機関 | HHS OCR | 個人情報保護委員会 |
| 更新頻度 | 適宜 | 年次 | 3-4年で改訂 | 年次サーベイランス | 適宜 | 3年ごとに見直し |
| 認証取得期間 | N/A | 3-12ヶ月 | 6-18ヶ月 | 6-12ヶ月 | N/A | N/A |

### コンプライアンスフレームワーク間のマッピング

多くの組織は複数のコンプライアンス要件を同時に満たす必要がある。フレームワーク間の統制は 60-80% が重複するため、統合的なアプローチが効率的である。

```
+------------------------------------------------------------------+
|          統制マッピングの概念図                                      |
|------------------------------------------------------------------|
|                                                                  |
|     GDPR        SOC 2       PCI DSS      ISO 27001              |
|      |            |            |             |                   |
|      v            v            v             v                   |
|  +------------------------------------------------------+       |
|  |           統合統制フレームワーク (GRC)                    |       |
|  |------------------------------------------------------|       |
|  |  アクセス制御  → GDPR Art.32 / CC6.1 / Req.7 / A.9    |       |
|  |  暗号化      → GDPR Art.32 / CC6.7 / Req.3,4 / A.10  |       |
|  |  ログ監視    → GDPR Art.30 / CC7.2 / Req.10 / A.12   |       |
|  |  変更管理    →    --      / CC8.1 / Req.6 / A.12     |       |
|  |  インシデント → GDPR Art.33 / CC7.3 / Req.12 / A.16   |       |
|  |  脆弱性管理  →    --      / CC7.1 / Req.5,6,11/ A.12 |       |
|  |  教育・訓練  → GDPR Art.39 / CC1.4 / Req.12 / A.7    |       |
|  +------------------------------------------------------+       |
+------------------------------------------------------------------+
```

---

## 2. GDPR

### GDPR の適用範囲と基本概念

GDPR (General Data Protection Regulation) は 2018 年 5 月 25 日に施行された EU の個人データ保護規則である。GDPR の適用範囲を正しく理解することが、対応の第一歩となる。

**適用条件（地理的範囲 — 第 3 条）**:
- EU 域内に拠点を持つ組織がデータを処理する場合
- EU 域外の組織が EU 域内の個人に商品・サービスを提供する場合
- EU 域外の組織が EU 域内の個人の行動をモニタリングする場合

**基本用語**:
- **データ主体 (Data Subject)** — 個人データの対象となる自然人
- **データ管理者 (Data Controller)** — データ処理の目的と手段を決定する者
- **データ処理者 (Data Processor)** — 管理者の代理でデータを処理する者
- **DPO (Data Protection Officer)** — データ保護責任者。大規模データ処理を行う場合は任命が義務

### GDPR の 7 原則

```
+------------------------------------------------------------------+
|                  GDPR データ保護 7 原則                              |
|------------------------------------------------------------------|
|                                                                  |
|  1. 適法性・公正性・透明性 (Lawfulness, Fairness, Transparency)    |
|     → 6 つの法的根拠のいずれかに基づく処理                         |
|     → プライバシーポリシーの公開、平易な言語での説明               |
|     → 処理の目的・法的根拠をデータ主体に通知                       |
|                                                                  |
|  2. 目的の限定 (Purpose Limitation)                               |
|     → 収集時に明示された目的のみにデータを使用                     |
|     → 後から目的を追加する場合は互換性テストを実施                  |
|     → アーカイブ、科学研究、統計目的は互換性ありとみなされる        |
|                                                                  |
|  3. データの最小化 (Data Minimisation)                             |
|     → 目的に対して必要最小限のデータのみ収集                       |
|     → 「あると便利」なデータは収集しない                           |
|     → 定期的にデータの必要性をレビュー                             |
|                                                                  |
|  4. 正確性 (Accuracy)                                             |
|     → データを最新・正確に保つ合理的な措置                         |
|     → 不正確なデータは遅滞なく消去または訂正                       |
|                                                                  |
|  5. 保存期間の制限 (Storage Limitation)                            |
|     → 必要な期間のみ保持、不要になったら削除                       |
|     → データ保持ポリシーの策定と遵守                               |
|     → 保持期間の定期レビュー                                      |
|                                                                  |
|  6. 完全性・機密性 (Integrity & Confidentiality)                   |
|     → 適切な技術的・組織的措置によるデータの保護                    |
|     → 暗号化、仮名化、アクセス制御の実施                           |
|     → 不正アクセス・データ損失からの保護                           |
|                                                                  |
|  7. アカウンタビリティ (Accountability)                            |
|     → 上記 6 原則の遵守を証明できる記録の保持                      |
|     → DPIA (データ保護影響評価) の実施                             |
|     → データ処理活動の記録 (ROPA) の維持                           |
+------------------------------------------------------------------+
```

### GDPR の 6 つの法的根拠

データ処理には以下 6 つの法的根拠のいずれかが必要である（第 6 条）。

| 法的根拠 | 説明 | 使用例 | 注意点 |
|---------|------|--------|--------|
| 同意 (Consent) | データ主体の明確な同意 | マーケティングメール | 同意は自由に撤回可能 |
| 契約の履行 | 契約遂行に必要な処理 | 注文処理、配送 | 契約に直接必要なもののみ |
| 法的義務 | 法令で義務づけられた処理 | 税務記録の保存 | 根拠法を特定する必要あり |
| 重大な利益 | 人の生命・安全を守る処理 | 医療緊急時 | 他の根拠が使えない場合のみ |
| 公共の利益 | 公的機関の任務遂行 | 行政サービス | 公的機関に限定的 |
| 正当な利益 | 管理者の正当な事業目的 | 不正防止、ネットワークセキュリティ | LIA (Legitimate Interest Assessment) が必要 |

### データ主体の権利と技術的実装

```python
# データ主体の権利への技術的対応

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

class GDPRCompliance:
    """GDPR データ主体の権利を実装するサービス

    GDPR 第 12-23 条に定められたデータ主体の 8 つの権利を
    技術的に実装する。各メソッドは監査ログを自動記録し、
    応答期限（原則 1 ヶ月）の管理機能を含む。
    """

    RESPONSE_DEADLINE_DAYS = 30  # 第12条: 原則1ヶ月以内に応答

    def __init__(self, db, audit_logger, notification_service):
        self.db = db
        self.audit = audit_logger
        self.notify = notification_service

    def right_to_access(self, user_id: str, request_id: str) -> dict:
        """アクセス権 (第15条): 保有データの開示

        データ主体は自身に関するデータの処理の有無、処理目的、
        データのカテゴリ、受領者、保存期間等を知る権利を有する。
        """
        self.audit.log("access_request", user_id=user_id, request_id=request_id)

        data = {
            'request_id': request_id,
            'generated_at': datetime.utcnow().isoformat(),
            'deadline': (datetime.utcnow() + timedelta(
                days=self.RESPONSE_DEADLINE_DAYS
            )).isoformat(),
            'personal_data': self.db.get_user_data(user_id),
            'processing_purposes': [
                {'purpose': 'service_provision', 'legal_basis': 'contract'},
                {'purpose': 'analytics', 'legal_basis': 'legitimate_interest'},
                {'purpose': 'marketing', 'legal_basis': 'consent'},
            ],
            'categories': ['identity', 'contact', 'usage', 'payment'],
            'recipients': [
                {'name': 'Stripe', 'purpose': 'payment_processing', 'country': 'US',
                 'safeguard': 'Standard Contractual Clauses'},
                {'name': 'SendGrid', 'purpose': 'email_delivery', 'country': 'US',
                 'safeguard': 'Standard Contractual Clauses'},
            ],
            'retention_period': '3 years after account deletion',
            'data_source': 'user_registration',
            'automated_decision_making': False,
            'right_to_lodge_complaint': 'You may lodge a complaint with your '
                                        'local supervisory authority.',
        }
        return self.export_as_portable_format(data)  # JSON/CSV

    def right_to_rectification(self, user_id: str, corrections: dict) -> dict:
        """訂正権 (第16条): 不正確なデータの訂正

        データ主体は不正確な個人データの訂正を要求でき、
        不完全なデータの補完を要求できる。
        """
        old_data = self.db.get_user_data(user_id)
        changes = []

        for field, new_value in corrections.items():
            if field in old_data and old_data[field] != new_value:
                changes.append({
                    'field': field,
                    'old_value': old_data[field],
                    'new_value': new_value,
                })

        self.db.update_user_data(user_id, corrections)
        self.audit.log("rectification", user_id=user_id, changes=changes)

        # 第三者への通知義務 (第19条)
        self.notify_recipients_of_change(user_id, changes)

        return {'status': 'rectified', 'changes': changes}

    def right_to_erasure(self, user_id: str, request_id: str) -> dict:
        """削除権 / 忘れられる権利 (第17条)

        以下の場合にデータ主体は削除を要求できる:
        - データが目的に不要になった場合
        - 同意を撤回した場合
        - 処理に異議を申し立てた場合
        - 違法な処理の場合

        ただし、法的義務の履行、公共の利益、法的請求の
        行使・防御に必要な場合は例外として保持できる。
        """
        legal_holds = self.check_legal_retention(user_id)

        deleted = []
        retained = []

        for table in self.get_user_tables():
            if table in legal_holds:
                # 匿名化 (削除の代わりに不可逆的な匿名化)
                self.anonymize_data(user_id, table)
                retained.append({
                    'table': table,
                    'reason': legal_holds[table],
                    'action': 'anonymized',
                })
            else:
                self.delete_data(user_id, table)
                deleted.append(table)

        # バックアップからの削除も予約 (次回ローテーション時に削除)
        self.schedule_backup_deletion(user_id)

        # 第三者への通知 (第19条)
        self.notify_recipients_of_erasure(user_id)

        self.audit.log("erasure", user_id=user_id, request_id=request_id,
                      deleted=deleted, retained=retained)

        return {
            'request_id': request_id,
            'deleted': deleted,
            'retained_anonymized': retained,
            'backup_deletion_scheduled': True,
            'third_parties_notified': True,
        }

    def right_to_portability(self, user_id: str, format: str = 'json') -> bytes:
        """データポータビリティ権 (第20条)

        データ主体が提供したデータを、構造化された一般的に使用される
        機械可読形式で受け取り、別のサービスに移転する権利。
        """
        data = self.db.get_user_provided_data(user_id)

        if format == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        elif format == 'csv':
            return self.convert_to_csv(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def right_to_restriction(self, user_id: str, reason: str) -> dict:
        """処理の制限権 (第18条)

        データの正確性に異議がある場合、処理が違法な場合、
        管理者がデータを不要とするが法的請求に必要な場合等に
        処理の制限を要求できる。
        """
        self.db.set_processing_restriction(user_id, restricted=True)
        self.audit.log("restriction", user_id=user_id, reason=reason)
        return {'status': 'processing_restricted', 'reason': reason}

    def right_to_object(self, user_id: str, processing_purpose: str) -> dict:
        """異議申立権 (第21条)

        データ主体は正当な利益または公共の利益に基づく処理に
        異議を申し立てることができる。ダイレクトマーケティング
        目的の処理に対する異議は無条件で認められる。
        """
        if processing_purpose == 'direct_marketing':
            # ダイレクトマーケティングへの異議は無条件で受理
            self.db.opt_out_marketing(user_id)
            return {'status': 'objection_accepted', 'purpose': processing_purpose}

        # その他の目的は LIA (正当利益評価) を再実施
        return {
            'status': 'objection_received',
            'purpose': processing_purpose,
            'message': 'Your objection will be reviewed within 30 days.',
        }

    def data_breach_notification(self, breach_details: dict):
        """データ侵害通知 (第33条/第34条)

        72 時間以内の監督機関への通知が義務。
        高リスクの場合はデータ主体への通知も必要。
        """
        breach_record = {
            'detected_at': datetime.utcnow().isoformat(),
            'deadline': (datetime.utcnow() + timedelta(hours=72)).isoformat(),
            'details': breach_details,
        }

        # 監督機関に 72 時間以内に通知 (第33条)
        self.notify_supervisory_authority(
            breach_details,
            deadline_hours=72,
        )

        # 高リスクの場合はデータ主体にも通知 (第34条)
        if breach_details['risk_level'] == 'high':
            self.notify_affected_individuals(breach_details)

        self.audit.log("breach_notification", breach=breach_record)
        return breach_record

    def conduct_dpia(self, processing_activity: dict) -> dict:
        """データ保護影響評価 (DPIA) の実施 (第35条)

        以下の場合に DPIA が義務:
        - プロファイリング等の自動処理に基づく体系的評価
        - 特別カテゴリデータの大規模処理
        - 公的にアクセス可能な区域の体系的監視
        """
        dpia = {
            'processing_description': processing_activity,
            'necessity_assessment': self._assess_necessity(processing_activity),
            'risk_assessment': self._assess_risks(processing_activity),
            'mitigation_measures': self._propose_mitigations(processing_activity),
            'dpo_opinion': None,  # DPO による確認が必要
            'conducted_at': datetime.utcnow().isoformat(),
        }
        return dpia
```

### プライバシーバイデザイン (Privacy by Design)

GDPR 第 25 条はデータ保護をシステム設計段階から組み込むことを要求している。以下に 7 つの基本原則と実装パターンを示す。

```
+------------------------------------------------------------------+
|          プライバシーバイデザインの 7 原則                           |
|          (Ann Cavoukian, 2009)                                    |
|------------------------------------------------------------------|
|                                                                  |
|  1. 事後対応ではなく事前防止                                       |
|     → 設計段階からプライバシーリスクを排除                          |
|                                                                  |
|  2. デフォルトでのプライバシー保護                                  |
|     → ユーザが何もしなくても最大限のプライバシーが保護される         |
|                                                                  |
|  3. 設計への組み込み                                              |
|     → プライバシーはアドオンではなく中核機能                        |
|                                                                  |
|  4. 全機能性の確保 (ゼロサムでない)                                 |
|     → プライバシーとセキュリティの両立                              |
|                                                                  |
|  5. ライフサイクル全体の保護                                       |
|     → 収集から削除まで一貫した保護                                 |
|                                                                  |
|  6. 可視性と透明性                                                |
|     → 処理の透明性確保、独立した検証可能性                         |
|                                                                  |
|  7. ユーザのプライバシーの尊重                                      |
|     → ユーザ中心のデザイン                                        |
+------------------------------------------------------------------+
```

```python
# プライバシーバイデザインの実装例

class UserRegistration:
    """データ最小化と同意管理を組み込んだユーザ登録

    GDPR 第 5 条 (データ最小化) と第 7 条 (同意の条件) に準拠。
    """

    REQUIRED_FIELDS = ['email']  # サービス提供に最低限必要
    OPTIONAL_FIELDS = {
        'name': {'purpose': 'personalization', 'legal_basis': 'consent'},
        'phone': {'purpose': 'two_factor_auth', 'legal_basis': 'consent'},
        'birthday': {'purpose': 'age_verification', 'legal_basis': 'consent'},
    }

    def register(self, data: dict, consents: dict) -> dict:
        """プライバシーバイデザインに基づくユーザ登録"""
        user_data = {}

        # 必須フィールドのみ (法的根拠: 契約の履行)
        for field in self.REQUIRED_FIELDS:
            user_data[field] = data[field]

        # オプションフィールド (法的根拠: 同意)
        for field, meta in self.OPTIONAL_FIELDS.items():
            if consents.get(f'collect_{field}') and field in data:
                user_data[field] = data[field]

        # 保持期限の設定 (第 5 条 1(e): 保存期間の制限)
        user_data['retention_until'] = self.calculate_retention_date()

        # 同意の記録 (第 7 条 1: 同意の立証責任)
        consent_record = self.record_consent(
            email=user_data['email'],
            consents=consents,
            ip_address=self.get_hashed_ip(),  # 匿名化した IP
            timestamp=datetime.utcnow(),
            consent_version='v2.1',  # 同意文面のバージョン管理
        )

        return self.db.create_user(user_data)

    def calculate_retention_date(self) -> str:
        """保持期限の算出 (サービス提供目的: アカウント存続中 + 3 年)"""
        return (datetime.utcnow() + timedelta(days=365 * 3)).isoformat()


class DataRetentionManager:
    """データ保持ポリシーの自動実行

    保持期限を過ぎたデータを自動的に削除または匿名化する。
    """

    RETENTION_POLICIES = {
        'user_profiles': {'days': 1095, 'action': 'delete'},
        'access_logs': {'days': 365, 'action': 'anonymize'},
        'payment_records': {'days': 2555, 'action': 'archive'},  # 7年 (税法)
        'consent_records': {'days': 1825, 'action': 'archive'},  # 5年
        'session_data': {'days': 30, 'action': 'delete'},
    }

    def execute_retention_policy(self):
        """日次バッチで保持期限超過データを処理"""
        for table, policy in self.RETENTION_POLICIES.items():
            cutoff = datetime.utcnow() - timedelta(days=policy['days'])
            expired_records = self.db.find_expired(table, cutoff)

            if policy['action'] == 'delete':
                count = self.db.delete_records(table, expired_records)
                logger.info(f"Deleted {count} expired records from {table}")
            elif policy['action'] == 'anonymize':
                count = self.db.anonymize_records(table, expired_records)
                logger.info(f"Anonymized {count} expired records from {table}")
            elif policy['action'] == 'archive':
                count = self.db.archive_records(table, expired_records)
                logger.info(f"Archived {count} expired records from {table}")
```

### GDPR 国際データ移転

EU 域外へのデータ移転は GDPR 第 5 章で規定されている。2020 年の Schrems II 判決により、EU-US 間の Privacy Shield は無効化され、代替メカニズムが必要となった。

```
+------------------------------------------------------------------+
|          GDPR 国際データ移転の仕組み                                |
|------------------------------------------------------------------|
|                                                                  |
|  [十分性認定 (Adequacy Decision)]                                 |
|  +-- 日本 (2019年1月、相互認定)                                   |
|  +-- 英国 (2021年6月)                                            |
|  +-- 韓国 (2022年12月)                                           |
|  +-- EU-US Data Privacy Framework (2023年7月)                    |
|  → 十分性認定国: 追加措置なしでデータ移転可能                      |
|                                                                  |
|  [標準契約条項 (SCC - Standard Contractual Clauses)]              |
|  +-- 2021年6月の新SCC (旧SCCは2022年12月で失効)                   |
|  +-- Module 1: Controller → Controller                           |
|  +-- Module 2: Controller → Processor (最も一般的)                |
|  +-- Module 3: Processor → Processor                             |
|  +-- Module 4: Processor → Controller                            |
|  → Transfer Impact Assessment (TIA) の実施が必要                  |
|                                                                  |
|  [拘束的企業準則 (BCR - Binding Corporate Rules)]                 |
|  +-- グループ企業間のデータ移転                                    |
|  +-- 監督機関の承認が必要 (取得に 1-2 年)                          |
|  +-- 大企業向け                                                  |
|                                                                  |
|  [例外的移転 (Derogations)]                                       |
|  +-- データ主体の明示的同意                                       |
|  +-- 契約の履行に必要                                             |
|  +-- 重大な公共の利益                                             |
|  → 反復的・大量のデータ移転には不適切                              |
+------------------------------------------------------------------+
```

---

## 3. SOC 2

### SOC 2 の位置づけと仕組み

SOC 2 (System and Organization Controls 2) は AICPA (米国公認会計士協会) が定める監査フレームワークである。クラウドサービスプロバイダーが顧客データを適切に管理していることを第三者が検証する仕組みであり、B2B SaaS では事実上の標準となっている。

SOC 2 レポートが求められる典型的なシナリオ:
- エンタープライズ顧客との商談で「SOC 2 レポートを提出してください」と依頼される
- RFP (提案依頼書) の必須要件として記載されている
- サイバー保険の適用条件として要求される

### SOC 2 の Trust Service Criteria (TSC)

```
+------------------------------------------------------------------+
|            SOC 2 Trust Service Criteria (TSC)                     |
|------------------------------------------------------------------|
|                                                                  |
|  CC: Common Criteria (全レポートで必須 — セキュリティ)              |
|  +-- CC1: 統制環境 (Control Environment)                         |
|  |   → 組織のガバナンス、倫理観、人材管理                          |
|  |   → 取締役会・経営陣によるセキュリティへのコミットメント          |
|  +-- CC2: コミュニケーションと情報                                |
|  |   → セキュリティポリシーの文書化と周知                          |
|  +-- CC3: リスク評価                                             |
|  |   → 定期的なリスクアセスメントの実施                            |
|  +-- CC4: 監視活動                                               |
|  |   → 統制の有効性を継続的に監視                                 |
|  +-- CC5: 統制活動                                               |
|  |   → リスクを軽減するための方針と手続                            |
|  +-- CC6: 論理的・物理的アクセス制御                               |
|  |   → 認証、認可、物理セキュリティ                               |
|  +-- CC7: システム運用                                            |
|  |   → 異常検知、インシデント対応                                 |
|  +-- CC8: 変更管理                                               |
|  |   → 変更の承認・テスト・本番適用のプロセス                      |
|  +-- CC9: リスク軽減                                             |
|      → ベンダー管理、ビジネスリスクの軽減策                        |
|                                                                  |
|  追加カテゴリ (必要に応じて選択):                                   |
|  +-- A: 可用性 (Availability)                                     |
|  |   → SLA、バックアップ、DR 計画                                 |
|  +-- PI: 処理の完全性 (Processing Integrity)                      |
|  |   → データ処理の正確性・完全性の保証                            |
|  +-- C: 機密性 (Confidentiality)                                  |
|  |   → 機密情報の保護、暗号化、アクセス制御                        |
|  +-- P: プライバシー (Privacy)                                    |
|      → AICPA Privacy Criteria、GDPR との整合                     |
+------------------------------------------------------------------+
```

### SOC 2 Type I vs Type II

| 項目 | Type I | Type II |
|------|--------|---------|
| 評価対象 | 特定時点の統制設計 | 一定期間の統制運用 |
| 評価期間 | スナップショット (1日) | 通常 6-12 ヶ月 |
| 信頼性 | 低い (設計のみ) | 高い (実際の運用を検証) |
| 用途 | 初回取得、準備段階 | 本格的な信頼性証明 |
| 取得期間 | 1-3 ヶ月 | 6-12 ヶ月 |
| 費用 | $20K-$60K | $30K-$100K+ |
| 顧客の評価 | 「まず第一歩」 | 「信頼できる証明」 |

### SOC 2 対応の技術的統制

```yaml
# SOC 2 統制の実装マッピング (実践的なサンプル)

CC1.4_Security_Training:
  description: "セキュリティ意識向上プログラム"
  controls:
    - name: "新入社員セキュリティ研修"
      implementation: "入社1週間以内に必須研修を受講"
      evidence: "LMS 修了記録、テスト結果"
    - name: "年次セキュリティ研修"
      implementation: "全社員が年1回受講"
      evidence: "LMS 修了記録、受講率レポート"

CC6.1_Logical_Access:
  description: "論理的アクセス制御"
  controls:
    - name: "SSO + MFA の必須化"
      implementation: "Okta SAML + FIDO2 (YubiKey / Passkey)"
      evidence: "Okta ログ、MFA 登録率レポート"
      test_procedure: |
        1. MFA なしでログインを試行し、拒否されることを確認
        2. Okta ダッシュボードで MFA 登録率 100% を確認
        3. 退職者のアカウントが無効化されていることを確認

    - name: "最小権限の IAM ポリシー"
      implementation: "AWS IAM + SCP + Permission Boundary"
      evidence: "IAM Access Analyzer レポート、権限一覧"
      test_procedure: |
        1. IAM Access Analyzer で外部アクセス可能なリソースを確認
        2. 管理者権限を持つユーザの一覧と正当性を確認
        3. 未使用の権限が付与されていないことを確認

    - name: "定期的なアクセスレビュー"
      implementation: "四半期ごとの棚卸し"
      evidence: "アクセスレビュー記録、承認ログ"

CC6.6_External_Threats:
  description: "外部脅威からの保護"
  controls:
    - name: "WAF によるアプリケーション保護"
      implementation: "AWS WAF + CloudFront"
      evidence: "WAF ルール設定、ブロックログ"
    - name: "ペネトレーションテスト"
      implementation: "年次で外部業者が実施"
      evidence: "ペネトレーションテスト報告書、改善対応記録"

CC7.2_Monitoring:
  description: "異常・セキュリティイベントの監視"
  controls:
    - name: "SIEM によるログ監視"
      implementation: "Datadog SIEM + PagerDuty"
      evidence: "アラートログ、インシデント対応記録"
    - name: "脆弱性スキャン"
      implementation: "Trivy (週次)、OWASP ZAP (月次)"
      evidence: "スキャンレポート"

CC7.3_Incident_Response:
  description: "セキュリティインシデントへの対応"
  controls:
    - name: "インシデント対応手順書"
      implementation: "Confluence に文書化、年次で訓練実施"
      evidence: "手順書、訓練記録、ポストモーテムレポート"
    - name: "インシデント通知"
      implementation: "PagerDuty → Slack → 関係者への通知"
      evidence: "通知ログ、通知タイムラインの記録"

CC8.1_Change_Management:
  description: "変更管理プロセス"
  controls:
    - name: "コードレビュー必須"
      implementation: "GitHub PR + 2名承認 + CODEOWNERS"
      evidence: "PR マージログ、レビュー履歴"
    - name: "CI/CD パイプラインテスト"
      implementation: "GitHub Actions (自動テスト + SAST + SCA)"
      evidence: "CI/CD ログ、テストカバレッジレポート"
    - name: "本番デプロイの承認フロー"
      implementation: "GitHub Environments + Required Reviewers"
      evidence: "デプロイログ、承認記録"
```

### SOC 2 監査準備のタイムライン

```
+------------------------------------------------------------------+
|          SOC 2 Type II 取得のロードマップ                           |
|------------------------------------------------------------------|
|                                                                  |
|  Month 1-2: ギャップ分析 (Gap Assessment)                         |
|  +-- 現在の統制状況を TSC にマッピング                             |
|  +-- ギャップを特定し優先順位付け                                  |
|  +-- 監査法人の選定・契約                                         |
|                                                                  |
|  Month 2-4: 統制の実装 (Remediation)                              |
|  +-- ポリシー・手順書の策定                                       |
|  +-- 技術的統制の実装 (MFA、暗号化、監視等)                        |
|  +-- ツール導入 (Drata / Vanta / Secureframe)                    |
|                                                                  |
|  Month 4-5: Type I 監査 (オプション)                               |
|  +-- 特定時点の統制設計を検証                                      |
|  +-- 早期に問題を発見・修正                                       |
|                                                                  |
|  Month 5-11: 観察期間 (Observation Period)                        |
|  +-- 最低 6 ヶ月間の統制運用を実施                                 |
|  +-- 証跡を継続的に収集・保管                                     |
|  +-- 内部監査で問題がないか確認                                   |
|                                                                  |
|  Month 11-12: Type II 監査 (Audit)                                |
|  +-- 監査人がサンプルテストを実施                                  |
|  +-- 統制の運用有効性を検証                                       |
|  +-- レポート発行                                                |
|                                                                  |
|  以降: 年次更新                                                   |
|  +-- 毎年 Type II レポートを更新                                   |
|  +-- 継続的な統制運用と証跡収集                                    |
+------------------------------------------------------------------+
```

---

## 4. PCI DSS

### PCI DSS v4.0 の概要

PCI DSS (Payment Card Industry Data Security Standard) はクレジットカード情報を扱うすべての組織に適用されるセキュリティ基準である。PCI SSC (Payment Card Industry Security Standards Council) が策定し、Visa、Mastercard、American Express、JCB、Discover の 5 大カードブランドが参加している。

PCI DSS v4.0 は 2022 年 3 月に公開され、2024 年 3 月 31 日に v3.2.1 から完全移行した。v4.0 の主な変更点:
- **カスタマイズアプローチ** の導入 — 要件の目的を満たす代替手法を認める柔軟性
- **多要素認証 (MFA)** の要件強化 — カードデータ環境 (CDE) へのすべてのアクセスに MFA
- **リスクベースアプローチ** — ターゲット型リスク分析の義務化
- **認証要件の強化** — パスワード最低 12 文字、フィッシング対策

### PCI DSS v4.0 の要件概要

```
+------------------------------------------------------------------+
|                PCI DSS v4.0 の 12 要件                             |
|------------------------------------------------------------------|
|                                                                  |
|  [ネットワークの構築と維持]                                        |
|  要件 1: ネットワークセキュリティ統制の導入・維持                    |
|      → FW/ACL でカードデータ環境 (CDE) を分離                     |
|      → ネットワークセグメンテーションの実施                         |
|  要件 2: すべてのシステムコンポーネントに安全な設定を適用             |
|      → デフォルトパスワードの変更                                  |
|      → 不要なサービス・機能の無効化                                |
|                                                                  |
|  [アカウントデータの保護]                                          |
|  要件 3: 保存されたアカウントデータの保護                           |
|      → PAN の暗号化、マスキング、トランケーション                   |
|      → CVV/PIN の保存禁止                                        |
|  要件 4: オープンな公共ネットワーク経由の送信データの暗号化           |
|      → TLS 1.2 以上の使用                                        |
|      → 安全でないプロトコル (SSL, 初期 TLS) の排除                 |
|                                                                  |
|  [脆弱性管理プログラムの維持]                                      |
|  要件 5: すべてのシステムとネットワークをマルウェアから保護           |
|      → アンチマルウェアソリューションの導入                         |
|  要件 6: 安全なシステムとソフトウェアの開発・維持                    |
|      → セキュアな開発ライフサイクル (SDLC)                         |
|      → 公開アプリのWAF保護(6.4.2)、スクリプト管理(6.4.3)          |
|                                                                  |
|  [強力なアクセス制御の実施]                                        |
|  要件 7: アカウントデータへのアクセスを業務上の必要性に制限           |
|      → 最小権限の原則 (Need to Know)                              |
|  要件 8: ユーザの識別と認証                                        |
|      → MFA の導入拡大 (v4.0 で要件強化)                           |
|      → パスワード最低 12 文字 (v4.0 新要件)                       |
|  要件 9: カードデータへの物理アクセスの制限                         |
|      → 物理セキュリティ統制 (入退室管理)                           |
|                                                                  |
|  [定期的な監視とテスト]                                            |
|  要件 10: すべてのシステムコンポーネントとカードデータへの              |
|           アクセスのログ記録と監視                                  |
|      → 監査ログの生成・保護・レビュー                              |
|      → NTP によるタイムシンク                                     |
|  要件 11: システムとネットワークのセキュリティを定期的にテスト         |
|      → ASV スキャン (四半期)、ペネトレーションテスト (年次)          |
|      → 内部脆弱性スキャン (四半期)                                 |
|                                                                  |
|  [情報セキュリティポリシーの維持]                                    |
|  要件 12: 組織のポリシーとプログラムで情報セキュリティをサポート       |
|      → セキュリティポリシーの策定・維持・周知                       |
|      → セキュリティ意識向上プログラム                               |
|      → インシデント対応計画                                       |
+------------------------------------------------------------------+
```

### PCI DSS 対応の実装例

```python
# 要件 3: カードデータの保護

import hashlib
import hmac
import secrets
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

audit_logger = logging.getLogger('pci_audit')

class CardDataProtection:
    """PCI DSS 要件 3 に準拠したカードデータ保護

    以下の PCI DSS v4.0 要件に対応:
    - 3.3.2: SAD (Sensitive Authentication Data) の保存禁止
    - 3.4.1: PAN の保存時のマスキングまたは暗号化
    - 3.5.1: PAN の暗号化に使用する鍵の管理
    - 3.5.1.2: ディスク暗号化の使用条件
    """

    def __init__(self, encryption_key: bytes, hmac_key: bytes):
        # 要件 3.5.1: 鍵管理 (HSM または KMS 推奨)
        self.aesgcm = AESGCM(encryption_key)
        self.hmac_key = hmac_key

    def store_card(self, pan: str, expiry: str, cvv: str = None) -> dict:
        """カード情報の安全な保存

        PAN は暗号化して保存し、CVV は決して保存しない。
        """
        # 要件 3.3.2: CVV は認可完了後に保存禁止
        if cvv is not None:
            audit_logger.warning("CVV provided but will NOT be stored (PCI DSS 3.3.2)")
            # CVV は認可処理でのみ使用し、直後に破棄

        # 要件 3.4.1: PAN の暗号化 (AES-256-GCM)
        nonce = secrets.token_bytes(12)
        encrypted_pan = self.aesgcm.encrypt(
            nonce, pan.encode(), associated_data=b"pan"
        )

        # 要件 3.4: PAN の末尾 4 桁のみ表示用に保持
        masked_pan = f"****-****-****-{pan[-4:]}"

        # 検索用のトークン化 (要件 3.5 — 可逆暗号化とは別管理)
        token = hmac.new(
            self.hmac_key,
            pan.encode(),
            hashlib.sha256,
        ).hexdigest()

        audit_logger.info(f"Card stored: masked={masked_pan}, token={token[:8]}...")

        return {
            'token': token,               # 検索・参照用
            'masked_pan': masked_pan,      # 表示用
            'encrypted_pan': nonce + encrypted_pan,  # 暗号化 PAN
            'expiry_encrypted': self.aesgcm.encrypt(
                secrets.token_bytes(12), expiry.encode(), associated_data=b"expiry"
            ),
            # CVV は保存しない (要件 3.3.2)
            # PIN は保存しない (要件 3.3.3)
            # 磁気ストライプの全データは保存しない (要件 3.3.1)
        }

    def retrieve_pan(self, encrypted_data: bytes, reason: str,
                     user: str) -> str:
        """PAN の復号 (要件 3.5 + 要件 10 - アクセスログ記録必須)

        復号にはビジネス上の正当な理由と監査ログの記録が必須。
        """
        # 要件 10.2.1: カードデータへのアクセスをログ記録
        audit_logger.info(
            f"PAN decryption: user={user}, reason={reason}"
        )
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, associated_data=b"pan").decode()

    def validate_pan_format(self, pan: str) -> bool:
        """PAN のフォーマット検証 (Luhn アルゴリズム)"""
        digits = [int(d) for d in pan if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        # Luhn チェック
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0
```

### カード情報の分類と保護要件

```
+------------------------------------------------------------------+
|            カードデータの分類と保護要件 (v4.0)                       |
|------------------------------------------------------------------|
|                                                                  |
|  データ種別        | 保存  | 暗号化 | マスク | 例                  |
|  ----------------------------------------------------------------|
|  PAN (カード番号)   | 可    | 必須   | 必須   | 4111...1111        |
|  カード会員名       | 可    | 推奨   | --    | TARO YAMADA        |
|  有効期限           | 可    | 推奨   | --    | 12/26              |
|  サービスコード     | 可    | 推奨   | --    | 201                |
|  CVV/CVC           | 不可  | --    | --    | 123                |
|  PIN / PIN Block   | 不可  | --    | --    | ****               |
|  磁気ストライプ     | 不可  | --    | --    | --                 |
|  EMV チップデータ   | 不可  | --    | --    | --                 |
|                                                                  |
|  ※「不可」= 認可完了後に一切保存してはならない                      |
|  ※ PAN マスキング: 最初の6桁と最後の4桁以外を表示                   |
|    (BIN + 末尾4桁の表示はビジネス上の必要性が必要)                   |
+------------------------------------------------------------------+
```

### PCI DSS スコープの縮小戦略

```
+------------------------------------------------------------------+
|          PCI DSS スコープ縮小の 4 つの戦略                           |
|------------------------------------------------------------------|
|                                                                  |
|  戦略 1: トークナイゼーション                                      |
|  +-- Stripe / Braintree 等の PSP を使用                           |
|  +-- 自社システムにカードデータが一切触れない構成                    |
|  +-- SAQ-A (最も簡易) で済む可能性                                 |
|  +-- 推奨度: ★★★★★ (最も効果的)                                   |
|                                                                  |
|  戦略 2: ネットワークセグメンテーション                             |
|  +-- CDE (カードデータ環境) を分離された VLAN に隔離                |
|  +-- CDE 以外のシステムはスコープ外                                |
|  +-- セグメンテーションテスト (年2回) が必要                        |
|  +-- 推奨度: ★★★★☆                                               |
|                                                                  |
|  戦略 3: P2PE (Point-to-Point Encryption)                        |
|  +-- PCI SSC 認定の P2PE ソリューションを使用                      |
|  +-- 端末からPSPまでエンドツーエンドで暗号化                        |
|  +-- 対面決済向け                                                 |
|  +-- 推奨度: ★★★☆☆ (対面決済がある場合)                           |
|                                                                  |
|  戦略 4: クラウドプロバイダの責任共有                               |
|  +-- AWS / GCP / Azure の PCI DSS 準拠環境を活用                   |
|  +-- 物理セキュリティ等はクラウドプロバイダの責任                    |
|  +-- 自社の責任範囲を明確化 (Shared Responsibility Matrix)         |
|  +-- 推奨度: ★★★★☆                                               |
+------------------------------------------------------------------+
```

### SAQ (自己問診) の種類

| SAQ 種別 | 対象 | 要件数 | 主な条件 |
|---------|------|--------|---------|
| SAQ-A | カードデータを一切扱わない e-commerce | ~22 | 決済ページが完全に PSP 側 |
| SAQ-A-EP | リダイレクト型だが自社で一部制御 | ~191 | 決済フォームの JavaScript を制御 |
| SAQ-B | インプリンタまたはダイヤルアップ端末 | ~41 | 電子的にカードデータを保存しない |
| SAQ-B-IP | IP 接続の PTS 端末 | ~82 | スタンドアロン IP 端末のみ |
| SAQ-C | POS システム (インターネット接続) | ~160 | PA-DSS 認定アプリを使用 |
| SAQ-C-VT | 仮想端末 (手動入力) | ~79 | ウェブベースの仮想端末のみ |
| SAQ-D | 上記に該当しないすべて | ~329 | フル要件準拠が必要 |

---

## 5. HIPAA (医療情報保護)

### HIPAA の基本構造

HIPAA (Health Insurance Portability and Accountability Act) は米国の医療情報保護法であり、PHI (Protected Health Information) の保護を義務づけている。

```
+------------------------------------------------------------------+
|            HIPAA の主要ルール                                      |
|------------------------------------------------------------------|
|                                                                  |
|  [Privacy Rule (プライバシールール)]                               |
|  +-- PHI の使用・開示に関する制限                                  |
|  +-- 患者の権利 (アクセス権、修正権)                               |
|  +-- NPP (Notice of Privacy Practices) の提供義務                |
|  +-- Minimum Necessary Standard (必要最小限の原則)                |
|                                                                  |
|  [Security Rule (セキュリティルール)]                              |
|  +-- ePHI (電子的 PHI) の保護に関する技術的要件                     |
|  +-- 管理的保護措置 (Administrative Safeguards)                   |
|  +-- 物理的保護措置 (Physical Safeguards)                         |
|  +-- 技術的保護措置 (Technical Safeguards)                        |
|                                                                  |
|  [Breach Notification Rule (侵害通知ルール)]                      |
|  +-- 500 人以上に影響: 60 日以内に HHS と個人に通知                 |
|  +-- 500 人未満: 年次で HHS に報告                                 |
|  +-- メディアへの通知 (500 人以上の場合)                            |
+------------------------------------------------------------------+
```

### HIPAA Security Rule の技術的保護措置

```python
# HIPAA Security Rule 準拠の ePHI 保護実装例

class HIPAACompliance:
    """HIPAA Security Rule の技術的保護措置を実装

    §164.312 Technical Safeguards に準拠
    """

    def access_control(self, user_id: str, resource: str) -> bool:
        """§164.312(a)(1) アクセス制御

        ePHI を含むシステムへのアクセスを許可された者のみに制限
        """
        # Unique User Identification (§164.312(a)(2)(i))
        if not self.verify_user_identity(user_id):
            return False

        # Emergency Access Procedure (§164.312(a)(2)(ii))
        if self.is_emergency_mode():
            return self.emergency_access_check(user_id, resource)

        # Automatic Logoff (§164.312(a)(2)(iii))
        if self.session_inactive(user_id, timeout_minutes=15):
            self.terminate_session(user_id)
            return False

        # Role-based access check
        return self.rbac_check(user_id, resource)

    def audit_controls(self, event: dict):
        """§164.312(b) 監査統制

        ePHI へのアクセス・変更を記録する仕組み
        """
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': event['user_id'],
            'action': event['action'],  # create, read, update, delete
            'resource': event['resource'],
            'phi_accessed': event.get('phi_fields', []),
            'ip_address': event['ip_address'],
            'outcome': event['outcome'],  # success, failure
        }
        # WORM ストレージに書き込み (改竄防止)
        self.audit_store.write_immutable(audit_entry)

    def transmission_security(self, data: bytes, destination: str) -> bytes:
        """§164.312(e)(1) 送信セキュリティ

        ePHI のネットワーク経由の送信時の暗号化
        """
        # Encryption (§164.312(e)(2)(ii))
        # TLS 1.2+ は必須 (NIST SP 800-52 Rev.2 準拠)
        return self.encrypt_for_transmission(data, destination)
```

---

## 6. 日本の個人情報保護法

### 2022 年改正のポイント

```
+------------------------------------------------------------------+
|          個人情報保護法 2022 年改正の主なポイント                     |
|------------------------------------------------------------------|
|                                                                  |
|  [漏洩報告の義務化]                                               |
|  +-- 個人情報保護委員会への報告が義務に (速報: 概ね3-5日以内)       |
|  +-- 本人への通知も義務に                                         |
|  +-- 対象: 要配慮個人情報、財産的被害のおそれ、                     |
|      不正目的のおそれ、1,000件超の漏洩                             |
|                                                                  |
|  [個人の権利拡大]                                                 |
|  +-- 利用停止・消去請求権の拡大                                   |
|  +-- 開示請求のデジタル化 (電磁的記録での開示)                     |
|  +-- 第三者提供記録の開示請求                                     |
|                                                                  |
|  [罰則の強化]                                                    |
|  +-- 法人に対する罰金: 最大 1 億円 (従来 50 万円)                  |
|  +-- データベース等不正提供罪の法定刑引き上げ                       |
|                                                                  |
|  [新たな類型]                                                    |
|  +-- 仮名加工情報の新設 (内部分析用の緩和措置)                     |
|  +-- 個人関連情報の第三者提供規制                                  |
|  +-- Cookie 等を他の情報と組み合わせる場合の同意                   |
+------------------------------------------------------------------+
```

---

## 7. コンプライアンス自動化

### 継続的コンプライアンスの仕組み

```
+------------------------------------------------------------------+
|          継続的コンプライアンス (Continuous Compliance)              |
|------------------------------------------------------------------|
|                                                                  |
|  [自動チェック (Infrastructure as Code)]                          |
|  +-- AWS Config Rules → リソース設定の継続監視                     |
|  |   → S3 バケットの暗号化、パブリックアクセスの禁止               |
|  +-- Prowler → CIS/PCI DSS ベンチマークスキャン                   |
|  +-- ScoutSuite → マルチクラウドセキュリティ監査                   |
|  +-- tfsec / checkov → Terraform 設定のセキュリティスキャン        |
|  +-- OPA (Open Policy Agent) → ポリシーのコード化                 |
|                                                                  |
|  [証跡の自動収集]                                                 |
|  +-- CloudTrail → API 操作ログ (全リージョン有効化)                |
|  +-- VPC Flow Logs → ネットワーク通信ログ                         |
|  +-- GitHub Audit Log → コード変更・レビューの証跡                  |
|  +-- Okta System Log → 認証・アクセスの証跡                       |
|  +-- PagerDuty → インシデント対応の記録                            |
|                                                                  |
|  [レポートの自動生成]                                              |
|  +-- Security Hub → コンプライアンススコアの集約                    |
|  +-- Drata/Vanta/Secureframe → SOC 2 証跡の自動収集              |
|  +-- AWS Audit Manager → 監査フレームワークの自動評価              |
+------------------------------------------------------------------+
```

### GRC プラットフォームの比較

| 項目 | Drata | Vanta | Secureframe | 手動運用 |
|------|-------|-------|-------------|---------|
| 対応フレームワーク | SOC 2, ISO 27001, GDPR, HIPAA, PCI DSS | SOC 2, ISO 27001, HIPAA, GDPR | SOC 2, ISO 27001, HIPAA, PCI DSS | 全て |
| 証跡自動収集 | 75+ 連携 | 50+ 連携 | 40+ 連携 | なし |
| 費用 (年間) | $10K-$50K | $10K-$40K | $8K-$30K | 人件費のみ |
| セットアップ期間 | 1-2 週間 | 1-2 週間 | 1-2 週間 | N/A |
| 監査人連携 | あり | あり | あり | 個別対応 |
| ポリシーテンプレート | あり | あり | あり | 自作 |

### 自動化スクリプト

```bash
# Prowler で PCI DSS チェックを実行
prowler aws --compliance pci_dss_4.0 --output-formats html,json

# CIS Benchmark チェック (AWS)
prowler aws --compliance cis_2.0_aws --severity critical high

# GDPR 関連チェック
prowler aws --compliance gdpr_aws

# SOC 2 関連チェック
prowler aws --compliance soc2

# HIPAA 関連チェック
prowler aws --compliance hipaa

# レポートを S3 にアップロード
aws s3 cp /tmp/prowler-output/ s3://compliance-reports/$(date +%Y-%m-%d)/ --recursive
```

```python
# AWS Config Rules による継続的コンプライアンス監視

import boto3
import json

def setup_compliance_rules():
    """PCI DSS / SOC 2 に必要な AWS Config Rules のセットアップ"""
    config = boto3.client('config')

    # 主要なマネージドルール
    rules = [
        {
            'name': 'encrypted-volumes',
            'source': 'AWS_CONFIG_RULES',
            'identifier': 'ENCRYPTED_VOLUMES',
            'description': 'EBS ボリュームの暗号化チェック (PCI DSS 3.4)',
        },
        {
            'name': 's3-bucket-server-side-encryption',
            'source': 'AWS_CONFIG_RULES',
            'identifier': 'S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED',
            'description': 'S3 バケットのサーバサイド暗号化 (PCI DSS 3.4)',
        },
        {
            'name': 'iam-user-mfa-enabled',
            'source': 'AWS_CONFIG_RULES',
            'identifier': 'IAM_USER_MFA_ENABLED',
            'description': 'IAM ユーザの MFA 有効化 (PCI DSS 8.3)',
        },
        {
            'name': 'cloudtrail-enabled',
            'source': 'AWS_CONFIG_RULES',
            'identifier': 'CLOUD_TRAIL_ENABLED',
            'description': 'CloudTrail の有効化 (PCI DSS 10.1)',
        },
        {
            'name': 'rds-storage-encrypted',
            'source': 'AWS_CONFIG_RULES',
            'identifier': 'RDS_STORAGE_ENCRYPTED',
            'description': 'RDS ストレージの暗号化 (PCI DSS 3.4)',
        },
        {
            'name': 'vpc-flow-logs-enabled',
            'source': 'AWS_CONFIG_RULES',
            'identifier': 'VPC_FLOW_LOGS_ENABLED',
            'description': 'VPC Flow Logs の有効化 (PCI DSS 10.1)',
        },
    ]

    for rule in rules:
        config.put_config_rule(
            ConfigRule={
                'ConfigRuleName': rule['name'],
                'Description': rule['description'],
                'Source': {
                    'Owner': rule['source'],
                    'SourceIdentifier': rule['identifier'],
                },
                'Scope': {
                    'ComplianceResourceTypes': [],
                },
            }
        )
        print(f"Config Rule '{rule['name']}' created.")


def generate_compliance_report():
    """コンプライアンスレポートの自動生成"""
    config = boto3.client('config')

    response = config.describe_compliance_by_config_rule()
    report = {
        'generated_at': datetime.utcnow().isoformat(),
        'summary': {'compliant': 0, 'non_compliant': 0, 'not_applicable': 0},
        'details': [],
    }

    for rule in response['ComplianceByConfigRules']:
        status = rule['Compliance']['ComplianceType']
        report['details'].append({
            'rule': rule['ConfigRuleName'],
            'status': status,
        })
        if status == 'COMPLIANT':
            report['summary']['compliant'] += 1
        elif status == 'NON_COMPLIANT':
            report['summary']['non_compliant'] += 1
        else:
            report['summary']['not_applicable'] += 1

    return report
```

### Terraform による Policy as Code

```hcl
# OPA (Open Policy Agent) を使った PCI DSS ポリシーの例
# policy/pci_dss.rego

package pci_dss

# 要件 3.4: PAN の暗号化
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_rds_cluster"
    not resource.change.after.storage_encrypted
    msg := sprintf(
        "PCI DSS 3.4: RDS cluster '%s' must have encryption enabled",
        [resource.name]
    )
}

# 要件 4.1: TLS 1.2 以上の使用
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_lb_listener"
    resource.change.after.protocol == "HTTPS"
    resource.change.after.ssl_policy == "ELBSecurityPolicy-2016-08"
    msg := sprintf(
        "PCI DSS 4.1: Load balancer '%s' must use TLS 1.2+ policy",
        [resource.name]
    )
}

# 要件 10.1: ログの有効化
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_s3_bucket"
    not has_logging(resource)
    msg := sprintf(
        "PCI DSS 10.1: S3 bucket '%s' must have access logging enabled",
        [resource.name]
    )
}
```

---

## 8. 監査対応の実践

### 監査準備チェックリスト

```
+------------------------------------------------------------------+
|          監査準備チェックリスト (SOC 2 / PCI DSS 共通)               |
|------------------------------------------------------------------|
|                                                                  |
|  [ドキュメント]                                                   |
|  [ ] 情報セキュリティポリシー (年次更新済み)                        |
|  [ ] アクセス管理手順書                                           |
|  [ ] 変更管理手順書                                               |
|  [ ] インシデント対応手順書                                        |
|  [ ] 事業継続計画 (BCP) / 災害復旧計画 (DRP)                       |
|  [ ] リスク評価報告書 (年次)                                       |
|  [ ] ベンダー管理ポリシーと評価記録                                 |
|                                                                  |
|  [技術的証跡]                                                     |
|  [ ] アクセスレビュー記録 (四半期)                                  |
|  [ ] 脆弱性スキャン結果と改善記録                                  |
|  [ ] ペネトレーションテスト報告書 (年次)                            |
|  [ ] パッチ適用記録                                               |
|  [ ] バックアップ・リストアテスト記録                               |
|  [ ] 変更管理チケット (PR ログ)                                    |
|  [ ] インシデント対応記録 (ポストモーテム)                          |
|                                                                  |
|  [人的証跡]                                                       |
|  [ ] セキュリティ研修の受講記録                                    |
|  [ ] 秘密保持契約 (NDA) の締結記録                                 |
|  [ ] バックグラウンドチェック記録                                   |
|  [ ] 退職者のアクセス権削除記録                                    |
+------------------------------------------------------------------+
```

### 監査人への対応の心構え

```python
# 監査対応のベストプラクティス (疑似コードで表現)

class AuditResponseStrategy:
    """監査人への効果的な対応戦略"""

    def prepare_evidence(self, control_id: str) -> dict:
        """証跡の事前準備

        原則: 「聞かれる前に用意する」
        """
        return {
            'control_description': self.get_control_description(control_id),
            'implementation_details': self.get_implementation(control_id),
            'evidence_artifacts': self.collect_evidence(control_id),
            'test_results': self.get_test_results(control_id),
            'exceptions': self.get_known_exceptions(control_id),
        }

    def respond_to_finding(self, finding: dict) -> dict:
        """指摘事項への対応

        原則: 「否定せず、具体的な改善計画を提示する」
        """
        return {
            'finding_id': finding['id'],
            'acknowledgment': True,
            'root_cause': self.analyze_root_cause(finding),
            'remediation_plan': {
                'short_term': self.get_immediate_fix(finding),
                'long_term': self.get_permanent_fix(finding),
                'timeline': self.estimate_timeline(finding),
                'owner': self.assign_owner(finding),
            },
        }

    # DO NOT:
    # - 証跡を後から作成する (監査人は日付の整合性をチェックする)
    # - 質問されていないことまで回答する (スコープ拡大のリスク)
    # - 技術的に不正確な説明をする (信頼の喪失)
```

---

## 9. エッジケース

### エッジケース 1: GDPR と他法規制の矛盾

GDPR の削除権と税法上の帳簿保存義務が矛盾する場合がある。たとえば、ドイツの AO (税法) は商業文書の 10 年保存を義務づけている。この場合、GDPR 第 17 条 3 項 (b) に基づき「法的義務の履行」として保持が認められるが、処理の制限 (第 18 条) を適用し、保持目的以外の処理を制限する必要がある。

### エッジケース 2: バックアップからのデータ削除

GDPR の削除権に基づく削除要求を受けた場合、バックアップに含まれるデータの即時削除は技術的に困難な場合がある。この場合、以下のアプローチが許容される:
- バックアップの次回ローテーション時に削除を予約する
- リストア時に削除対象データを再削除する仕組み (暗号消去) を導入する
- 暗号化鍵の破棄 (Crypto Shredding) により事実上のデータ削除を実現する

### エッジケース 3: マイクロサービス環境でのデータ主体権利の行使

マイクロサービスアーキテクチャでは、データが複数のサービスに分散している。削除権やアクセス権の行使時に、すべてのサービスから確実にデータを取得・削除するためには、以下の設計が必要:

```
+------------------------------------------------------------------+
|          マイクロサービスでの GDPR 対応アーキテクチャ                 |
|------------------------------------------------------------------|
|                                                                  |
|  [GDPR Orchestrator Service]                                     |
|  +-- データ主体の権利行使リクエストを一元管理                       |
|  +-- 各サービスに対して fan-out で要求を配信                       |
|  +-- Saga パターンで全サービスの完了を確認                          |
|                                                                  |
|  User API  ---+                                                  |
|  Order API ---+--> GDPR Orchestrator --> 結果の集約 & 応答        |
|  Analytics ---+     |                                            |
|  Payment API--+     +-- 全サービスの応答を追跡                     |
|                     +-- タイムアウト・失敗時のリトライ              |
|                     +-- 監査ログの記録                             |
+------------------------------------------------------------------+
```

### エッジケース 4: PCI DSS のカスタマイズアプローチ

PCI DSS v4.0 で導入されたカスタマイズアプローチでは、定義されたアプローチ (従来の方法) の代わりに、要件の「目的」を満たす代替手法を提案できる。ただし、ターゲット型リスク分析の実施と QSA による検証が必要であり、小規模組織には負担が大きい場合がある。

---

## 10. アンチパターン

### アンチパターン 1: 年次監査だけのコンプライアンス

```
NG:
  → 監査の直前にだけ対策を実施 (「監査シーズン」の発生)
  → 年間の大半は統制が機能していない
  → 監査のためだけのドキュメント作成 (形骸化)
  → 監査直前の「証跡作り」は監査人に見抜かれる

  根本原因:
  → コンプライアンスを「コスト」として捉えている
  → 経営層のコミットメント不足
  → 日常業務との統合ができていない

OK:
  → 継続的なコンプライアンス監視を自動化
  → AWS Config / Security Hub で日次チェック
  → 証跡を自動収集し、監査時の負荷を軽減
  → コンプライアンスを開発プロセスに組み込む (Compliance as Code)
  → GRC プラットフォーム (Drata/Vanta) で常時可視化
```

### アンチパターン 2: チェックリスト型コンプライアンス

```
NG:
  → 要件を形式的に満たすだけ (letter of the law)
  → 例: 「暗号化必須」→ MD5 ハッシュで「暗号化対応済み」
  → 例: 「ログ取得」→ ログを取るが誰も分析しない
  → 例: 「パスワードポリシー」→ 90日ごとの変更を強制
       (NIST SP 800-63B では定期変更は推奨されない)

  根本原因:
  → 要件の「意図」を理解していない
  → セキュリティの専門知識の不足
  → 形式的な達成を優先する組織文化

OK:
  → 要件の意図を理解し実効性のある対策を実施 (spirit of the law)
  → 暗号化 → AES-256-GCM + AWS KMS 管理鍵
  → ログ → SIEM 連携 + 異常検知ルール + 定期レビュー
  → 認証 → パスワードレス認証 (FIDO2) への移行
  → 定期的なペネトレーションテストで実効性を検証
```

### アンチパターン 3: 全社一律のコンプライアンス適用

```
NG:
  → PCI DSS の要件を全システムに適用 (過剰な統制)
  → コスト増大、開発速度の低下
  → 「セキュリティは面倒」という認識の拡大

OK:
  → リスクベースでスコープを適切に定義
  → PCI DSS は CDE のみ、他システムは CIS Controls ベース
  → セグメンテーションによるスコープ最小化
  → 環境ごとに適切なレベルの統制を適用
```

---

## 11. 演習

### 演習 1: GDPR データ主体アクセス要求 (DSAR) の処理

**課題**: あるユーザから GDPR 第 15 条に基づくアクセス要求が届いた。以下の要件を満たすシステムを設計せよ。

1. 複数のデータベース (PostgreSQL, Redis, S3) に分散したユーザデータを網羅的に収集する
2. 第三者に共有したデータのリストを生成する
3. 応答期限 (30 日) の管理を行う
4. データをポータブルフォーマット (JSON) で提供する

**ヒント**:
- データインベントリ (ROPA — Records of Processing Activities) を事前に整備しておくことが重要
- 各サービスに「GDPR エンドポイント」を実装し、Orchestrator パターンで統合する

### 演習 2: SOC 2 Type II 証跡の自動収集

**課題**: 以下の SOC 2 統制に対する証跡を自動収集するスクリプトを作成せよ。

1. CC6.1: MFA 登録率が 100% であることを確認 (Okta API)
2. CC7.2: 過去 30 日間の脆弱性スキャン結果サマリ
3. CC8.1: すべての本番デプロイが PR レビューを経ていることを確認 (GitHub API)

**期待する出力**: 監査人に提出できるフォーマットの JSON/CSV レポート

### 演習 3: PCI DSS スコープ縮小の設計

**課題**: 自社 EC サイトの PCI DSS スコープを最小化するアーキテクチャを設計せよ。

1. 現在の構成: 自社サーバでカード情報を受け取り、決済ゲートウェイに送信
2. 目標: SAQ-A または SAQ-A-EP レベルまでスコープを縮小
3. Stripe Elements / PaymentIntents を活用した設計を提案

**ヒント**:
- Stripe.js を使用し、カード情報が自社サーバを経由しない構成にする
- iFrame ベースの決済フォームにより、自社ドメインからの JavaScript でカードデータにアクセスできない構成が理想

### 演習 4: コンプライアンスダッシュボードの構築

**課題**: AWS Security Hub と CloudWatch を使用して、以下のコンプライアンスメトリクスをリアルタイムで表示するダッシュボードを設計せよ。

1. PCI DSS ベンチマークのスコア (% 準拠)
2. CIS Benchmark のスコア
3. 未修正の Critical/High 脆弱性の数
4. Config Rules の準拠状況

---

## 12. FAQ

### Q1. SOC 2 と ISO 27001 のどちらを取得すべきか?

北米市場向けの SaaS では SOC 2 が標準的に求められる。グローバル市場では ISO 27001 の認知度が高い。両方を取得する企業も多い。まず顧客の要求を確認し、最も求められるものから取得するのが効率的である。統制の重複は 60-70% 程度あるため、一方を取得すればもう一方は比較的容易に取得できる。

実務的な判断基準:
- **SOC 2 を優先**: 北米の B2B SaaS、スタートアップの初回取得
- **ISO 27001 を優先**: グローバル展開、政府・公共機関との取引、欧州市場
- **両方同時**: 効率的な取得が可能（統合監査アプローチ）

### Q2. GDPR は EU に顧客がいない場合も適用されるか?

EU 域内の個人にサービスを提供している場合、または EU 域内の個人の行動をモニタリングしている場合は、企業の所在地に関係なく GDPR が適用される（第 3 条の域外適用）。日本企業でも EU 向けサービスを提供していれば対象となる。

具体的な判断基準:
- ウェブサイトに EU の言語 (英語以外) や通貨 (EUR) のオプションがある
- EU からのアクセスを意図的にターゲティングしている
- EU 居住者の行動データを収集している (Cookie, トラッキング)

### Q3. PCI DSS の対象範囲 (スコープ) を縮小するには?

トークナイゼーションサービス (Stripe, AWS Payment Cryptography) を使い、自社システムでカード情報を扱わないようにする。これにより PCI DSS のスコープが大幅に縮小され、SAQ-A (最も簡易な自己問診) で済む場合がある。カード情報を自社で保持・処理する必要がないなら、まずスコープ縮小を検討すべきである。

### Q4. コンプライアンス対応の費用はどの程度か?

| 項目 | スタートアップ (50名) | 中堅企業 (200名) | 大企業 (1000名+) |
|------|-------------------|----------------|-----------------|
| SOC 2 Type II (初回) | $50K-$100K | $100K-$200K | $200K-$500K |
| ISO 27001 (初回) | $30K-$80K | $80K-$150K | $150K-$300K |
| PCI DSS (SAQ-D) | $50K-$200K | $200K-$500K | $500K-$1M+ |
| GRC プラットフォーム | $10K-$30K/年 | $30K-$60K/年 | $60K-$150K/年 |

※ 上記は監査費用 + ツール費用 + コンサルティング費用の概算。人件費は含まない。

### Q5. 複数のコンプライアンス要件を効率的に管理するには?

統合コンプライアンスフレームワーク (Unified Compliance Framework) のアプローチが有効:
1. 共通統制を特定し、一度の実装で複数の要件を満たす
2. GRC プラットフォームで統制のマッピングを管理
3. 証跡の収集を自動化し、複数の監査で同じ証跡を再利用
4. 統合内部監査を実施し、監査疲れを防ぐ

---

## まとめ

| 項目 | 要点 |
|------|------|
| GDPR | データ最小化、同意管理、72時間以内の侵害通知、国際データ移転 |
| SOC 2 | Trust Service Criteria に基づく統制の設計と運用、Type II で信頼性証明 |
| PCI DSS | カードデータの暗号化・マスク・アクセス制御、v4.0 で MFA 要件強化 |
| HIPAA | ePHI の保護、Security Rule の管理的・物理的・技術的保護措置 |
| 個人情報保護法 | 2022年改正で漏洩報告義務化、法人罰金最大1億円 |
| 継続的コンプライアンス | AWS Config + Prowler + GRC プラットフォームで自動監視 |
| 証跡自動収集 | CloudTrail + GitHub ログ + Okta ログで監査準備を効率化 |
| スコープ縮小 | トークナイゼーション + セグメンテーションでスコープを最小化 |
| Policy as Code | OPA / tfsec / checkov でコンプライアンスをコードとして管理 |

---

## 次に読むべきガイド

- [セキュリティ文化](./03-security-culture.md) — 組織全体でコンプライアンスを推進する文化
- [インシデント対応](./00-incident-response.md) — GDPR のデータ侵害通知に対応するフロー
- [監視/ログ](./01-monitoring-logging.md) — コンプライアンスに必要なログ収集と保存
- [暗号化](../03-cryptography/00-encryption.md) — PCI DSS / GDPR で求められる暗号化技術
- [アクセス制御](../02-network-security/02-access-control.md) — SOC 2 CC6 で求められるアクセス統制

---

## 参考文献

1. **GDPR 全文 (日本語訳)** — https://www.ppc.go.jp/enforcement/infoprovision/EU/
2. **GDPR 全文 (英語原文)** — https://eur-lex.europa.eu/eli/reg/2016/679/oj
3. **AICPA SOC 2 Trust Service Criteria** — https://www.aicpa.org/resources/landing/system-and-organization-controls-soc-suite-of-services
4. **PCI DSS v4.0** — https://www.pcisecuritystandards.org/document_library/
5. **PCI DSS v4.0 Summary of Changes** — https://www.pcisecuritystandards.org/document_library/
6. **NIST Cybersecurity Framework v2.0** — https://www.nist.gov/cyberframework
7. **NIST SP 800-53 Rev.5** — https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final
8. **HIPAA Security Rule** — https://www.hhs.gov/hipaa/for-professionals/security/index.html
9. **個人情報保護法 (2022 年改正)** — https://www.ppc.go.jp/personalinfo/legal/
10. **Prowler (AWS セキュリティ監査ツール)** — https://github.com/prowler-cloud/prowler
11. **OWASP Top 10** — https://owasp.org/www-project-top-ten/
12. **CIS Controls v8** — https://www.cisecurity.org/controls
