# 鍵管理

> 暗号鍵のライフサイクル管理、HSM/KMS による安全な鍵保管、エンベロープ暗号化の仕組みまで、暗号鍵を正しく運用するための包括的ガイド

## この章で学ぶこと

1. **鍵ライフサイクル** — 生成から廃棄まで、暗号鍵の各段階で求められる管理要件
2. **HSM と KMS の使い分け** — ハードウェアセキュリティモジュールとクラウド鍵管理サービスの特性
3. **エンベロープ暗号化** — データ暗号化鍵 (DEK) と鍵暗号化鍵 (KEK) による多層暗号化パターン

---

## 1. 鍵ライフサイクル

### 鍵の状態遷移

```
  +----------+     +----------+     +----------+
  |  生成     | --> |  有効化   | --> |  運用中   |
  | Generate |     | Activate |     |  Active  |
  +----------+     +----------+     +----------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
                  +----------+                      +----------+
                  | 一時停止  |                      |  期限切れ |
                  | Suspend  |                      | Expired  |
                  +----------+                      +----------+
                        |                                 |
                        v                                 v
                  +----------+                      +----------+
                  | 無効化   |                       |  アーカイブ|
                  | Deactivate|                     | Archive  |
                  +----------+                      +----------+
                        |                                 |
                        +----------------+----------------+
                                         |
                                         v
                                   +----------+
                                   |  廃棄     |
                                   | Destroy  |
                                   +----------+
```

### 鍵ライフサイクルの各段階

| 段階 | 説明 | 期間の目安 |
|------|------|-----------|
| 生成 | 暗号学的に安全な乱数で鍵を生成 | 即時 |
| 有効化 | 鍵をシステムに登録し使用可能にする | 即時 |
| 運用中 | 暗号化・署名に使用する期間 | 1-2年 (対称鍵) |
| 一時停止 | 調査等のため一時的に使用停止 | 数日-数週間 |
| 無効化 | 暗号化には使用不可、復号のみ許可 | 数年 |
| アーカイブ | 過去データの復号のためのみ保持 | 規制に依存 |
| 廃棄 | 安全に削除 (暗号学的消去) | 不可逆 |

---

## 2. 鍵の種類と用途

### 鍵の分類

```
+-----------------------------------------------+
|              鍵の階層構造                        |
|-----------------------------------------------|
|                                               |
|  +-- Master Key (マスター鍵)                   |
|  |   HSM 内に格納、エクスポート不可              |
|  |                                             |
|  +-- Key Encryption Key (KEK / ラッピング鍵)   |
|  |   マスター鍵で暗号化して保管                  |
|  |                                             |
|  +-- Data Encryption Key (DEK / データ鍵)      |
|      KEK で暗号化して保管                       |
|      実際のデータを暗号化                       |
+-----------------------------------------------+
```

### 鍵の生成 (Python)

```python
import os
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives import serialization

# 対称鍵: AES-256 用の 256 ビット鍵
aes_key = os.urandom(32)  # 暗号学的に安全な乱数

# 非対称鍵: RSA 4096 ビット
rsa_private = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
)

# 非対称鍵: ECDSA P-256
ec_private = ec.generate_private_key(ec.SECP256R1())

# 秘密鍵の暗号化保存
pem = ec_private.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(b"passphrase"),
)
```

---

## 3. HSM (Hardware Security Module)

### HSM の役割

```
+---------------------------------------------------+
|                  アプリケーション                     |
|---------------------------------------------------|
|  1. 鍵生成リクエスト      4. 署名リクエスト          |
|  2. 暗号化リクエスト      5. 鍵ラッピング            |
|  3. 復号リクエスト                                  |
+---------------------------------------------------+
            |  PKCS#11 / JCE / CNG API
            v
+---------------------------------------------------+
|                    HSM                             |
|---------------------------------------------------|
|  [耐タンパ性ハードウェア]                            |
|                                                   |
|  +-- 鍵ストレージ (鍵がHSM外に出ない)              |
|  +-- 暗号エンジン (HSM内で演算)                    |
|  +-- 乱数生成器 (TRNG)                            |
|  +-- 監査ログ                                     |
|  +-- FIPS 140-2/3 Level 3 認証                    |
+---------------------------------------------------+
```

### PKCS#11 を使った HSM 操作

```python
import pkcs11
from pkcs11 import KeyType, Mechanism

# HSM ライブラリのロード
lib = pkcs11.lib("/usr/lib/softhsm/libsofthsm2.so")
token = lib.get_token(token_label="my-token")

with token.open(user_pin="1234") as session:
    # HSM 内で AES 鍵を生成
    key = session.generate_key(
        KeyType.AES, 256,
        label="data-encryption-key",
        store=True,        # HSM に永続保存
        extractable=False, # HSM 外への抽出を禁止
    )

    # HSM 内で暗号化
    plaintext = b"Sensitive data"
    iv = session.generate_random(128)
    ciphertext = key.encrypt(plaintext, mechanism=Mechanism.AES_CBC_PAD, mechanism_param=iv)

    # HSM 内で復号
    decrypted = key.decrypt(ciphertext, mechanism=Mechanism.AES_CBC_PAD, mechanism_param=iv)
```

---

## 4. クラウド KMS

### AWS KMS の基本操作

```python
import boto3
import base64

kms = boto3.client("kms", region_name="ap-northeast-1")

# カスタマーマスターキー (CMK) の作成
response = kms.create_key(
    Description="Application data encryption key",
    KeyUsage="ENCRYPT_DECRYPT",
    KeySpec="SYMMETRIC_DEFAULT",  # AES-256-GCM
    Tags=[{"TagKey": "Environment", "TagValue": "production"}],
)
key_id = response["KeyMetadata"]["KeyId"]

# データの暗号化
encrypt_response = kms.encrypt(
    KeyId=key_id,
    Plaintext=b"Secret data",
    EncryptionContext={"purpose": "user-data", "tenant": "acme-corp"},
)
ciphertext = encrypt_response["CiphertextBlob"]

# データの復号
decrypt_response = kms.decrypt(
    CiphertextBlob=ciphertext,
    EncryptionContext={"purpose": "user-data", "tenant": "acme-corp"},
)
plaintext = decrypt_response["Plaintext"]
```

### KMS 比較表

| 項目 | AWS KMS | GCP Cloud KMS | Azure Key Vault |
|------|---------|---------------|-----------------|
| HSM バックエンド | CloudHSM 連携 | Cloud HSM | Managed HSM |
| FIPS 認証 | 140-2 Level 2/3 | 140-2 Level 3 | 140-2 Level 2/3 |
| 自動ローテーション | 年次 | カスタム | カスタム |
| 料金モデル | リクエスト課金 | リクエスト課金 | オペレーション課金 |
| 鍵のインポート | 可能 | 可能 | 可能 |
| マルチリージョン | マルチリージョンキー | グローバル | geo レプリケーション |

---

## 5. エンベロープ暗号化

### エンベロープ暗号化の仕組み

```
暗号化フロー:

  +-----------+     DEK (平文)     +-----------+
  |  KMS      | ---- 生成 -----> |  DEK       |
  | (マスター  |                   | (データ鍵) |
  |   鍵)     |                   +-----------+
  +-----------+                        |
       |                               |
       | DEK を                        | データを
       | マスター鍵で暗号化              | DEK で暗号化
       |                               |
       v                               v
  +-----------+                   +-----------+
  | 暗号化DEK  |                   | 暗号化     |
  | (メタデータ |                   | データ     |
  |  に保存)   |                   +-----------+
  +-----------+                        |
       |                               |
       +------- 一緒に保存 ------------+
```

### エンベロープ暗号化の実装

```python
import boto3
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

kms = boto3.client("kms")

def encrypt_data(key_id: str, plaintext: bytes) -> dict:
    """エンベロープ暗号化でデータを暗号化する"""
    # KMS から DEK (Data Encryption Key) を取得
    response = kms.generate_data_key(
        KeyId=key_id,
        KeySpec="AES_256",
    )
    dek_plaintext = response["Plaintext"]       # 平文 DEK (メモリ上のみ)
    dek_encrypted = response["CiphertextBlob"]  # 暗号化 DEK (保存用)

    # DEK でデータをローカル暗号化 (AES-256-GCM)
    nonce = os.urandom(12)
    aesgcm = AESGCM(dek_plaintext)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    # 平文 DEK をメモリから消去
    dek_plaintext = b"\x00" * len(dek_plaintext)

    return {
        "encrypted_dek": dek_encrypted,
        "nonce": nonce,
        "ciphertext": ciphertext,
    }

def decrypt_data(encrypted_bundle: dict) -> bytes:
    """エンベロープ暗号化で暗号化されたデータを復号する"""
    # KMS で DEK を復号
    response = kms.decrypt(CiphertextBlob=encrypted_bundle["encrypted_dek"])
    dek_plaintext = response["Plaintext"]

    # DEK でデータをローカル復号
    aesgcm = AESGCM(dek_plaintext)
    plaintext = aesgcm.decrypt(
        encrypted_bundle["nonce"],
        encrypted_bundle["ciphertext"],
        None,
    )

    return plaintext
```

---

## 6. 鍵ローテーション

### 自動ローテーション戦略

```python
import boto3
from datetime import datetime

kms = boto3.client("kms")

# AWS KMS で自動ローテーションを有効化 (年次)
kms.enable_key_rotation(KeyId="alias/my-app-key")

# ローテーション状態の確認
status = kms.get_key_rotation_status(KeyId="alias/my-app-key")
print(f"自動ローテーション: {status['KeyRotationEnabled']}")

# カスタムローテーション (手動)
def rotate_key_manual(alias: str):
    """新しい鍵を作成してエイリアスを切り替える"""
    # 新しい鍵を作成
    new_key = kms.create_key(
        Description=f"Rotated key - {datetime.now().isoformat()}",
        KeyUsage="ENCRYPT_DECRYPT",
    )
    new_key_id = new_key["KeyMetadata"]["KeyId"]

    # エイリアスを新しい鍵に向ける
    kms.update_alias(
        AliasName=f"alias/{alias}",
        TargetKeyId=new_key_id,
    )

    # 旧鍵は復号のために残す (無効化しない)
    return new_key_id
```

---

## 7. アンチパターン

### アンチパターン 1: 鍵のハードコーディング

```python
# NG: ソースコードに暗号鍵を直書き
ENCRYPTION_KEY = b"0123456789abcdef0123456789abcdef"

# OK: 環境変数または KMS から取得
import os
key_ref = os.environ["KMS_KEY_ARN"]
# KMS API 経由で暗号化・復号を行う
```

**影響**: Git 履歴に鍵が残り、ローテーションが不可能になる。漏洩時に全データが危殆化する。

### アンチパターン 2: 鍵とデータの同一ストレージ保管

```
NG:
  S3 バケット/
    ├── data/encrypted-data.bin
    └── keys/encryption-key.txt     ← 同じバケットに鍵を保管

OK:
  S3 バケット (データ)/
    └── data/encrypted-data.bin     ← 暗号化 DEK をメタデータに格納
  KMS (鍵)/
    └── マスターキー                  ← 別サービスで管理
```

**影響**: S3 バケットが漏洩した場合、鍵もデータも同時に流出し暗号化の意味がなくなる。

---

## 8. FAQ

### Q1. HSM と KMS のどちらを使うべきか?

コンプライアンス要件 (FIPS 140-2 Level 3) がある場合や、鍵が物理的に HSM 外に出てはならない場合は HSM を選択する。それ以外のほとんどのケースでは、クラウド KMS の方が運用負荷が低く費用対効果が高い。KMS の裏側も HSM で動作しているため、一般的なセキュリティ要件は満たせる。

### Q2. 鍵ローテーションの頻度はどのくらいが適切か?

NIST SP 800-57 では対称鍵の運用期間を最大 2 年と推奨している。PCI DSS では暗号鍵の年次ローテーションを求めている。実運用では自動ローテーションを有効にし、少なくとも年次で行うのが標準的である。

### Q3. エンベロープ暗号化を使う理由は何か?

大量データを KMS で直接暗号化すると、ネットワーク越しにデータを送信する必要がありレイテンシとコストが増大する。エンベロープ暗号化では KMS には小さな DEK のみ送り、データの暗号化はローカルで行うため高速かつ経済的である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 鍵ライフサイクル | 生成→有効化→運用→無効化→廃棄の全段階を管理 |
| HSM | 耐タンパ性ハードウェアで鍵の漏洩を物理的に防止 |
| KMS | クラウドマネージドで鍵管理の運用負荷を軽減 |
| エンベロープ暗号化 | DEK + KEK の二層構造で大量データを効率的に暗号化 |
| 鍵ローテーション | 自動ローテーションを有効にし最低年次で実施 |
| 鍵の分離 | 鍵とデータは必ず別のストレージ・サービスで管理 |
| 監査 | 鍵の使用履歴を CloudTrail 等で記録・監視 |

---

## 次に読むべきガイド

- [TLS/証明書](./01-tls-certificates.md) — TLS で鍵がどのように使われるかを理解する
- [クラウドセキュリティ基礎](../05-cloud-security/00-cloud-security-basics.md) — KMS を含むクラウドセキュリティの全体像
- [AWSセキュリティ](../05-cloud-security/01-aws-security.md) — AWS KMS・CloudHSM の実践的な活用

---

## 参考文献

1. **NIST SP 800-57 Part 1 — Recommendation for Key Management** — https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final
2. **AWS KMS Developer Guide** — https://docs.aws.amazon.com/kms/latest/developerguide/
3. **OWASP Cryptographic Storage Cheat Sheet** — https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
4. **Google Cloud KMS Documentation** — https://cloud.google.com/kms/docs
