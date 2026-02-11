# コンプライアンス

> GDPR、SOC 2、PCI DSS を中心に、セキュリティに関する法規制・業界標準への準拠方法と実装のポイントを体系的に学ぶ

## この章で学ぶこと

1. **GDPR (一般データ保護規則)** — EU の個人データ保護規制と技術的対応
2. **SOC 2** — サービス組織の内部統制に関する監査報告書と準拠のポイント
3. **PCI DSS** — クレジットカード情報を扱うシステムのセキュリティ要件

---

## 1. コンプライアンスの全体像

### 主要な規制・基準の分類

```
+----------------------------------------------------------+
|            セキュリティコンプライアンスの分類                 |
|----------------------------------------------------------|
|                                                          |
|  [法規制 (法的義務)]                                       |
|  +-- GDPR (EU 一般データ保護規則)                          |
|  +-- 個人情報保護法 (日本)                                 |
|  +-- CCPA/CPRA (カリフォルニア州)                          |
|  +-- HIPAA (米国医療情報)                                  |
|                                                          |
|  [業界標準 (業界義務)]                                     |
|  +-- PCI DSS (カード決済)                                 |
|  +-- FISC (金融情報)                                      |
|                                                          |
|  [監査フレームワーク (信頼性証明)]                          |
|  +-- SOC 2 Type I/II                                     |
|  +-- ISO 27001                                           |
|  +-- ISO 27701                                           |
|                                                          |
|  [ベストプラクティス (任意)]                                |
|  +-- NIST Cybersecurity Framework                        |
|  +-- CIS Controls                                        |
|  +-- OWASP Top 10                                        |
+----------------------------------------------------------+
```

### 規制・基準の比較

| 項目 | GDPR | SOC 2 | PCI DSS | ISO 27001 |
|------|------|-------|---------|-----------|
| 対象 | EU 個人データ | SaaS サービス全般 | カード決済 | 情報セキュリティ全般 |
| 強制力 | 法律 | 顧客要求 | 業界標準 (事実上必須) | 任意 (契約要件化あり) |
| 罰則 | 最大 2000万EUR or 売上4% | なし (信頼喪失) | 罰金 + カード処理停止 | なし |
| 監査 | 規制当局 | 独立監査人 (CPA) | QSA / ISA | 認証機関 |
| 更新頻度 | 適宜 | 年次 | 3-4年で改訂 | 年次サーベイランス |
| 認証取得期間 | N/A | 3-12ヶ月 | 6-18ヶ月 | 6-12ヶ月 |

---

## 2. GDPR

### GDPR の 7 原則

```
+----------------------------------------------------------+
|                GDPR データ保護 7 原則                       |
|----------------------------------------------------------|
|  1. 適法性・公正性・透明性 (Lawfulness)                     |
|     → 法的根拠に基づく処理、プライバシーポリシーの公開       |
|                                                          |
|  2. 目的の限定 (Purpose Limitation)                       |
|     → 明示された目的のみにデータを使用                     |
|                                                          |
|  3. データの最小化 (Data Minimisation)                     |
|     → 必要最小限のデータのみ収集                           |
|                                                          |
|  4. 正確性 (Accuracy)                                     |
|     → データを最新・正確に保つ                             |
|                                                          |
|  5. 保存期間の制限 (Storage Limitation)                    |
|     → 必要な期間のみ保持、不要になったら削除               |
|                                                          |
|  6. 完全性・機密性 (Integrity & Confidentiality)           |
|     → 暗号化・アクセス制御で保護                           |
|                                                          |
|  7. アカウンタビリティ (Accountability)                     |
|     → 上記原則の遵守を証明できること                       |
+----------------------------------------------------------+
```

### GDPR 対応の技術的実装

```python
# データ主体の権利への技術的対応

class GDPRCompliance:
    """GDPR データ主体の権利を実装するサービス"""

    def right_to_access(self, user_id: str) -> dict:
        """アクセス権 (第15条): 保有データの提供"""
        data = {
            'personal_data': self.db.get_user_data(user_id),
            'processing_purposes': ['service_provision', 'analytics'],
            'categories': ['identity', 'contact', 'usage'],
            'recipients': ['payment_processor', 'email_service'],
            'retention_period': '3 years after account deletion',
            'data_source': 'user_registration',
        }
        return self.export_as_portable_format(data)  # JSON/CSV

    def right_to_erasure(self, user_id: str) -> dict:
        """削除権 / 忘れられる権利 (第17条)"""
        # 法的保持義務のあるデータは除外
        legal_holds = self.check_legal_retention(user_id)

        deleted = []
        retained = []

        for table in self.get_user_tables():
            if table in legal_holds:
                # 匿名化 (削除の代わり)
                self.anonymize_data(user_id, table)
                retained.append(table)
            else:
                self.delete_data(user_id, table)
                deleted.append(table)

        # バックアップからの削除も予約
        self.schedule_backup_deletion(user_id)

        return {
            'deleted': deleted,
            'retained_anonymized': retained,
            'reason': 'legal_retention_obligation',
        }

    def right_to_portability(self, user_id: str) -> bytes:
        """データポータビリティ権 (第20条)"""
        data = self.db.get_user_provided_data(user_id)
        # 機械可読形式 (JSON) でエクスポート
        return json.dumps(data, indent=2).encode('utf-8')

    def data_breach_notification(self, breach_details: dict):
        """データ侵害通知 (第33条/第34条)"""
        # 監督機関に72時間以内に通知
        self.notify_supervisory_authority(
            breach_details,
            deadline_hours=72,
        )
        # 高リスクの場合はデータ主体にも通知
        if breach_details['risk_level'] == 'high':
            self.notify_affected_individuals(breach_details)
```

### プライバシーバイデザイン

```python
# データ最小化の実装例
class UserRegistration:
    """収集するデータを必要最小限に"""

    REQUIRED_FIELDS = ['email']
    OPTIONAL_FIELDS = ['name', 'phone']  # 明示的な同意が必要

    def register(self, data: dict, consents: dict) -> dict:
        user_data = {}

        # 必須フィールド
        for field in self.REQUIRED_FIELDS:
            user_data[field] = data[field]

        # オプションフィールド (同意がある場合のみ)
        for field in self.OPTIONAL_FIELDS:
            if consents.get(f'collect_{field}') and field in data:
                user_data[field] = data[field]

        # 保持期限の設定
        user_data['retention_until'] = self.calculate_retention_date()

        # 同意の記録
        self.record_consent(user_data['email'], consents)

        return self.db.create_user(user_data)
```

---

## 3. SOC 2

### SOC 2 の Trust Service Criteria (TSC)

```
+----------------------------------------------------------+
|          SOC 2 Trust Service Criteria                     |
|----------------------------------------------------------|
|                                                          |
|  CC: Common Criteria (全レポートで必須)                    |
|  +-- CC1: 統制環境 (COSO)                                |
|  +-- CC2: コミュニケーションと情報                         |
|  +-- CC3: リスク評価                                     |
|  +-- CC4: 監視活動                                       |
|  +-- CC5: 統制活動                                       |
|  +-- CC6: 論理的・物理的アクセス制御                       |
|  +-- CC7: システム運用                                    |
|  +-- CC8: 変更管理                                       |
|  +-- CC9: リスク軽減                                     |
|                                                          |
|  追加カテゴリ (必要に応じて選択):                           |
|  +-- A: 可用性 (Availability)                             |
|  +-- PI: 処理の完全性 (Processing Integrity)              |
|  +-- C: 機密性 (Confidentiality)                          |
|  +-- P: プライバシー (Privacy)                             |
+----------------------------------------------------------+
```

### SOC 2 対応の技術的統制

```yaml
# SOC 2 統制の実装マッピング

CC6.1_Logical_Access:
  description: "論理的アクセス制御"
  controls:
    - name: "SSO + MFA の必須化"
      implementation: "Okta SAML + YubiKey"
      evidence: "Okta ログ、MFA 登録率レポート"

    - name: "最小権限の IAM ポリシー"
      implementation: "AWS IAM + SCP"
      evidence: "IAM Access Analyzer レポート"

    - name: "定期的なアクセスレビュー"
      implementation: "四半期ごとの棚卸し"
      evidence: "アクセスレビュー記録"

CC7.2_Monitoring:
  description: "異常・セキュリティイベントの監視"
  controls:
    - name: "SIEM によるログ監視"
      implementation: "Datadog SIEM + PagerDuty"
      evidence: "アラートログ、インシデント対応記録"

    - name: "脆弱性スキャン"
      implementation: "Trivy (週次)、OWASP ZAP (月次)"
      evidence: "スキャンレポート"

CC8.1_Change_Management:
  description: "変更管理プロセス"
  controls:
    - name: "コードレビュー必須"
      implementation: "GitHub PR + 2名承認"
      evidence: "PR マージログ"

    - name: "CI/CD パイプラインテスト"
      implementation: "GitHub Actions (自動テスト + セキュリティスキャン)"
      evidence: "CI/CD ログ"
```

### SOC 2 Type I vs Type II

| 項目 | Type I | Type II |
|------|--------|---------|
| 評価対象 | 特定時点の統制設計 | 一定期間の統制運用 |
| 評価期間 | スナップショット (1日) | 通常 6-12 ヶ月 |
| 信頼性 | 低い (設計のみ) | 高い (実際の運用を検証) |
| 用途 | 初回取得、準備段階 | 本格的な信頼性証明 |
| 取得期間 | 1-3 ヶ月 | 6-12 ヶ月 |

---

## 4. PCI DSS

### PCI DSS v4.0 の要件概要

```
+----------------------------------------------------------+
|              PCI DSS v4.0 要件                             |
|----------------------------------------------------------|
|                                                          |
|  要件 1: ネットワークセキュリティ統制                       |
|  要件 2: 安全な設定の適用                                  |
|  要件 3: 保存されたアカウントデータの保護                    |
|  要件 4: オープンネットワーク経由の暗号化                   |
|  要件 5: マルウェア対策                                    |
|  要件 6: セキュアなシステム・ソフトウェアの開発              |
|  要件 7: アクセス制御 (Need to Know)                       |
|  要件 8: ユーザ識別と認証                                  |
|  要件 9: 物理アクセスの制限                                |
|  要件 10: ログと監視                                      |
|  要件 11: セキュリティテスト                                |
|  要件 12: 情報セキュリティポリシー                          |
+----------------------------------------------------------+
```

### PCI DSS 対応の実装例

```python
# 要件 3: カードデータの保護

import hashlib
import hmac
from cryptography.fernet import Fernet

class CardDataProtection:
    """PCI DSS 要件 3 に準拠したカードデータ保護"""

    def __init__(self, encryption_key: bytes, hmac_key: bytes):
        self.cipher = Fernet(encryption_key)
        self.hmac_key = hmac_key

    def store_card(self, pan: str, expiry: str) -> dict:
        """カード情報の安全な保存"""
        # PAN の暗号化 (要件 3.5)
        encrypted_pan = self.cipher.encrypt(pan.encode())

        # PAN の末尾 4 桁のみ表示用に保持 (要件 3.4)
        masked_pan = f"****-****-****-{pan[-4:]}"

        # 検索用のトークン化 (可逆暗号化とは別)
        token = hmac.new(
            self.hmac_key,
            pan.encode(),
            hashlib.sha256,
        ).hexdigest()

        return {
            'token': token,            # 検索・参照用
            'masked_pan': masked_pan,   # 表示用
            'encrypted_pan': encrypted_pan,  # 暗号化 PAN
            'expiry': self.cipher.encrypt(expiry.encode()),
            # CVV は保存禁止 (要件 3.3.2)
        }

    def retrieve_pan(self, encrypted_pan: bytes) -> str:
        """PAN の復号 (要件 3.5 - アクセスログ記録必須)"""
        self.audit_log("pan_decryption", reason="transaction_dispute")
        return self.cipher.decrypt(encrypted_pan).decode()
```

### カード情報の分類と保護

```
+----------------------------------------------------------+
|          カードデータの分類と保護要件                        |
|----------------------------------------------------------|
|  データ種別        | 保存  | 暗号化 | マスク | 例           |
|----------------------------------------------------------|
|  PAN (カード番号)   | 可    | 必須   | 必須   | 4111...1111 |
|  カード会員名       | 可    | 推奨   | --    | TARO YAMADA |
|  有効期限           | 可    | 推奨   | --    | 12/26       |
|  サービスコード     | 可    | 推奨   | --    | 201         |
|  CVV/CVC           | 不可  | --    | --    | 123         |
|  PIN               | 不可  | --    | --    | ****        |
|  磁気ストライプ     | 不可  | --    | --    | --          |
+----------------------------------------------------------+
```

---

## 5. コンプライアンス自動化

### 継続的コンプライアンスの仕組み

```
+----------------------------------------------------------+
|          継続的コンプライアンス (Continuous Compliance)      |
|----------------------------------------------------------|
|                                                          |
|  [自動チェック]                                           |
|  +-- AWS Config Rules → リソース設定の継続監視             |
|  +-- Prowler → CIS/PCI DSS ベンチマークスキャン           |
|  +-- ScoutSuite → マルチクラウドセキュリティ監査           |
|                                                          |
|  [証跡の自動収集]                                         |
|  +-- CloudTrail → API 操作ログ                           |
|  +-- GitHub PR ログ → 変更管理の証跡                      |
|  +-- PagerDuty → インシデント対応の記録                    |
|                                                          |
|  [レポートの自動生成]                                      |
|  +-- Security Hub → コンプライアンススコア                  |
|  +-- Drata/Vanta → SOC 2 証跡の自動収集                   |
+----------------------------------------------------------+
```

```bash
# Prowler で PCI DSS チェックを実行
prowler aws --compliance pci_dss_321

# CIS Benchmark チェック
prowler aws --compliance cis_2.0_aws

# GDPR 関連チェック
prowler aws --compliance gdpr_aws

# HTML レポート生成
prowler aws --compliance pci_dss_321 -M html -o /tmp/prowler-report
```

---

## 6. アンチパターン

### アンチパターン 1: 年次監査だけのコンプライアンス

```
NG:
  → 監査の直前にだけ対策を実施
  → 年間の大半は統制が機能していない
  → 監査のためだけのドキュメント作成

OK:
  → 継続的なコンプライアンス監視を自動化
  → AWS Config / Security Hub で日次チェック
  → 証跡を自動収集し、監査時の負荷を軽減
  → コンプライアンスを開発プロセスに組み込む
```

### アンチパターン 2: チェックリスト型コンプライアンス

```
NG:
  → 要件を形式的に満たすだけ
  → 例: 「暗号化必須」→ 弱い暗号方式で「対応済み」
  → 例: 「ログ取得」→ ログを取るが分析しない

OK:
  → 要件の意図を理解し実効性のある対策を実施
  → 暗号化 → AES-256-GCM + KMS 管理鍵
  → ログ → SIEM 連携 + 異常検知ルール
  → 定期的なペネトレーションテストで実効性を検証
```

---

## 7. FAQ

### Q1. SOC 2 と ISO 27001 のどちらを取得すべきか?

北米市場向けの SaaS では SOC 2 が標準的に求められる。グローバル市場ではISO 27001 の認知度が高い。両方を取得する企業も多い。まず顧客の要求を確認し、最も求められるものから取得するのが効率的である。統制の重複は 60-70% 程度あるため、一方を取得すればもう一方は比較的容易に取得できる。

### Q2. GDPR は EU に顧客がいない場合も適用されるか?

EU 域内の個人にサービスを提供している場合、または EU 域内の個人の行動をモニタリングしている場合は、企業の所在地に関係なく GDPR が適用される。日本企業でも EU 向けサービスを提供していれば対象となる。

### Q3. PCI DSS の対象範囲 (スコープ) を縮小するには?

トークナイゼーションサービス (Stripe, AWS Payment Cryptography) を使い、自社システムでカード情報を扱わないようにする。これにより PCI DSS のスコープが大幅に縮小され、SAQ-A (最も簡易な自己問診) で済む場合がある。カード情報を自社で保持・処理する必要がないなら、まずスコープ縮小を検討すべきである。

---

## まとめ

| 項目 | 要点 |
|------|------|
| GDPR | データ最小化、同意管理、72時間以内の侵害通知 |
| SOC 2 | Trust Service Criteria に基づく統制の設計と運用 |
| PCI DSS | カードデータの暗号化・マスク・アクセス制御 |
| 継続的コンプライアンス | AWS Config + Prowler で自動監視 |
| 証跡自動収集 | CloudTrail + GitHub ログで監査準備を効率化 |
| スコープ縮小 | トークナイゼーションで PCI DSS スコープを最小化 |

---

## 次に読むべきガイド

- [セキュリティ文化](./03-security-culture.md) — 組織全体でコンプライアンスを推進する文化
- [インシデント対応](./00-incident-response.md) — GDPR のデータ侵害通知に対応するフロー
- [監視/ログ](./01-monitoring-logging.md) — コンプライアンスに必要なログ収集と保存

---

## 参考文献

1. **GDPR 全文 (日本語訳)** — https://www.ppc.go.jp/enforcement/infoprovision/EU/
2. **AICPA SOC 2 Trust Service Criteria** — https://www.aicpa.org/resources/landing/system-and-organization-controls-soc-suite-of-services
3. **PCI DSS v4.0** — https://www.pcisecuritystandards.org/document_library/
4. **NIST Cybersecurity Framework** — https://www.nist.gov/cyberframework
