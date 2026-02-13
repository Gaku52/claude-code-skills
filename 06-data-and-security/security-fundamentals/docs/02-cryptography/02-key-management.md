# 鍵管理

> 暗号鍵のライフサイクル管理、HSM/KMS による安全な鍵保管、エンベロープ暗号化の仕組みまで、暗号鍵を正しく運用するための包括的ガイド

## この章で学ぶこと

1. **鍵ライフサイクル** — 生成から廃棄まで、暗号鍵の各段階で求められる管理要件
2. **HSM と KMS の使い分け** — ハードウェアセキュリティモジュールとクラウド鍵管理サービスの特性
3. **エンベロープ暗号化** — データ暗号化鍵 (DEK) と鍵暗号化鍵 (KEK) による多層暗号化パターン
4. **鍵ローテーションと運用** — 自動ローテーション戦略と緊急対応手順

## 前提知識

- 暗号の基礎 (対称鍵暗号と公開鍵暗号の違い)
- AES、RSA、ECDSA の基本概念
- クラウドサービス (AWS/GCP/Azure) の基本
- Python の基本的なプログラミング知識

---

## 1. 鍵ライフサイクル

### 鍵管理の全体像

暗号鍵の管理は暗号システム全体の安全性を左右する最も重要な要素である。NIST SP 800-57 では鍵のライフサイクルを厳密に定義しており、各段階で適切な管理を行わないと暗号化の意味が失われる。歴史的に、暗号アルゴリズム自体の脆弱性よりも、鍵管理の不備によるインシデントの方がはるかに多い。

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

### 鍵ライフサイクルの各段階 (詳細)

| 段階 | 説明 | 許可される操作 | 期間の目安 | セキュリティ要件 |
|------|------|--------------|-----------|----------------|
| 生成 | 暗号学的に安全な乱数で鍵を生成 | なし (生成のみ) | 即時 | CSPRNG 必須、HSM内推奨 |
| 有効化 | 鍵をシステムに登録し使用可能にする | 暗号化、署名 | 即時 | アクセス制御設定 |
| 運用中 | 暗号化・署名に使用する期間 | 暗号化、復号、署名、検証 | 1-2年 (対称鍵) | 使用ログ監査 |
| 一時停止 | 調査等のため一時的に使用停止 | なし | 数日-数週間 | 理由のドキュメント化 |
| 無効化 | 暗号化には使用不可、復号のみ許可 | 復号、検証のみ | 数年 | 暗号化操作の禁止 |
| アーカイブ | 過去データの復号のためのみ保持 | 復号のみ (要承認) | 規制に依存 | 厳格なアクセス制御 |
| 廃棄 | 安全に削除 (暗号学的消去) | なし (不可逆) | 不可逆 | 全コピーの確実な消去 |

### 暗号期間 (Cryptoperiod) の目安

```
鍵の種類                 暗号化期間    復号/検証期間    合計
+-----------------------------------------------------------+
| 対称暗号 (AES)         | 1-2年      | +3年          | 5年  |
| 非対称 (RSA署名)       | 1-3年      | +7年          | 10年 |
| 非対称 (ECDSA署名)     | 1-3年      | +7年          | 10年 |
| TLS サーバ証明書       | 1年        | -             | 1年  |
| ルート CA 証明書        | 10-20年    | +10年         | 30年 |
| 中間 CA 証明書         | 3-5年      | +5年          | 10年 |
| SSH ホスト鍵           | 1-2年      | -             | 2年  |
| API 署名鍵             | 90日-1年   | +1年          | 2年  |
+-----------------------------------------------------------+

出典: NIST SP 800-57 Part 1 Rev.5 Table 1
```

### 鍵ライフサイクル管理の実装

```python
import os
import json
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class KeyState(Enum):
    """NIST SP 800-57 に基づく鍵の状態"""
    PRE_ACTIVATION = "pre-activation"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    COMPROMISED = "compromised"
    DESTROYED = "destroyed"

@dataclass
class ManagedKey:
    """鍵ライフサイクルを管理するクラス"""
    key_id: str
    algorithm: str
    key_size: int
    state: KeyState = KeyState.PRE_ACTIVATION
    created_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    destroyed_at: Optional[datetime] = None
    usage_count: int = 0
    max_usage_count: int = 1_000_000
    metadata: dict = field(default_factory=dict)

    def activate(self, cryptoperiod_days: int = 365):
        """鍵を有効化する"""
        if self.state != KeyState.PRE_ACTIVATION:
            raise ValueError(f"Cannot activate key in state: {self.state}")
        self.state = KeyState.ACTIVE
        self.activated_at = datetime.utcnow()
        self.expiry_date = self.activated_at + timedelta(days=cryptoperiod_days)

    def use(self) -> bool:
        """鍵を使用する (使用可否を返す)"""
        if self.state != KeyState.ACTIVE:
            return False
        if self.expiry_date and datetime.utcnow() > self.expiry_date:
            self.state = KeyState.DEACTIVATED
            self.deactivated_at = datetime.utcnow()
            return False
        if self.usage_count >= self.max_usage_count:
            self.state = KeyState.DEACTIVATED
            self.deactivated_at = datetime.utcnow()
            return False
        self.usage_count += 1
        return True

    def suspend(self, reason: str):
        """鍵を一時停止する"""
        if self.state != KeyState.ACTIVE:
            raise ValueError(f"Cannot suspend key in state: {self.state}")
        self.state = KeyState.SUSPENDED
        self.metadata["suspend_reason"] = reason
        self.metadata["suspended_at"] = datetime.utcnow().isoformat()

    def compromise(self, reason: str):
        """鍵の危殆化を記録する"""
        self.state = KeyState.COMPROMISED
        self.metadata["compromise_reason"] = reason
        self.metadata["compromised_at"] = datetime.utcnow().isoformat()
        # 即座に鍵ローテーションを開始すべき

    def destroy(self):
        """鍵を安全に廃棄する"""
        if self.state in (KeyState.ACTIVE, KeyState.PRE_ACTIVATION):
            raise ValueError("Cannot destroy active or pre-activation key directly")
        self.state = KeyState.DESTROYED
        self.destroyed_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """鍵が期限切れか確認"""
        if self.expiry_date is None:
            return False
        return datetime.utcnow() > self.expiry_date

    def days_until_expiry(self) -> Optional[int]:
        """有効期限までの日数"""
        if self.expiry_date is None:
            return None
        delta = self.expiry_date - datetime.utcnow()
        return max(0, delta.days)


class KeyManager:
    """鍵ライフサイクル全体を管理するマネージャ"""

    def __init__(self):
        self.keys: dict[str, ManagedKey] = {}
        self.audit_log: list[dict] = []

    def generate_key(self, algorithm: str, key_size: int,
                     metadata: dict = None) -> ManagedKey:
        """新しい鍵を生成"""
        key_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        key = ManagedKey(
            key_id=key_id,
            algorithm=algorithm,
            key_size=key_size,
            metadata=metadata or {},
        )
        self.keys[key_id] = key
        self._audit("generate", key_id, f"Generated {algorithm}-{key_size} key")
        return key

    def rotate_key(self, old_key_id: str, cryptoperiod_days: int = 365) -> ManagedKey:
        """鍵をローテーション (旧鍵は復号のみ、新鍵を生成)"""
        old_key = self.keys.get(old_key_id)
        if not old_key:
            raise KeyError(f"Key {old_key_id} not found")

        # 旧鍵を無効化 (復号のみ可)
        if old_key.state == KeyState.ACTIVE:
            old_key.state = KeyState.DEACTIVATED
            old_key.deactivated_at = datetime.utcnow()
            self._audit("deactivate", old_key_id, "Deactivated for rotation")

        # 新鍵を生成・有効化
        new_key = self.generate_key(
            algorithm=old_key.algorithm,
            key_size=old_key.key_size,
            metadata={"rotated_from": old_key_id},
        )
        new_key.activate(cryptoperiod_days)
        self._audit("rotate", new_key.key_id,
                     f"Rotated from {old_key_id}")
        return new_key

    def get_expiring_keys(self, days: int = 30) -> list[ManagedKey]:
        """指定日数以内に期限切れになる鍵を取得"""
        expiring = []
        for key in self.keys.values():
            remaining = key.days_until_expiry()
            if remaining is not None and 0 < remaining <= days:
                expiring.append(key)
        return expiring

    def _audit(self, action: str, key_id: str, detail: str):
        """監査ログを記録"""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "key_id": key_id,
            "detail": detail,
        })
```

---

## 2. 鍵の種類と用途

### 鍵の階層構造

```
+-------------------------------------------------------+
|              鍵の階層構造 (Key Hierarchy)                |
|-------------------------------------------------------|
|                                                       |
|  Level 0: Root of Trust                               |
|  +-- Master Key (マスター鍵)                           |
|  |   HSM 内に格納、エクスポート不可                     |
|  |   FIPS 140-2/3 Level 3 で保護                      |
|  |   ライフサイクル: 10-20年                            |
|  |                                                    |
|  Level 1: Key Encryption Keys                         |
|  +-- KEK (Key Encryption Key / ラッピング鍵)          |
|  |   マスター鍵で暗号化して保管                         |
|  |   データ鍵の保護に使用                               |
|  |   ライフサイクル: 1-3年                              |
|  |                                                    |
|  Level 2: Data Keys                                   |
|  +-- DEK (Data Encryption Key / データ鍵)             |
|      KEK で暗号化して保管                              |
|      実際のデータを暗号化                              |
|      ライフサイクル: 数時間-1年                         |
|      (用途に応じて異なる)                               |
+-------------------------------------------------------+

分離の原則:
  - 各レベルの鍵は異なるストレージに保管
  - 上位の鍵が下位の鍵を保護
  - マスター鍵はHSM外に出ない
  - 鍵の危殆化時、影響範囲を限定できる
```

### 鍵の種類と推奨パラメータ

| 用途 | アルゴリズム | 鍵長 | セキュリティ強度 | 推奨/非推奨 |
|------|------------|------|----------------|-----------|
| データ暗号化 | AES-256-GCM | 256 bit | 256 bit | 推奨 |
| データ暗号化 | AES-128-GCM | 128 bit | 128 bit | 許容 |
| データ暗号化 | ChaCha20-Poly1305 | 256 bit | 256 bit | 推奨 |
| デジタル署名 | RSA | 4096 bit | ~140 bit | 推奨 |
| デジタル署名 | ECDSA P-256 | 256 bit | 128 bit | 推奨 |
| デジタル署名 | Ed25519 | 256 bit | ~128 bit | 強く推奨 |
| 鍵交換 | X25519 | 256 bit | ~128 bit | 強く推奨 |
| 鍵交換 | ECDH P-384 | 384 bit | 192 bit | 推奨 |
| パスワードハッシュ | Argon2id | - | - | 強く推奨 |
| パスワードハッシュ | bcrypt | - | - | 許容 |
| パスワードハッシュ | SHA-256 | - | - | 非推奨 (単独使用) |
| 鍵導出 | HKDF-SHA256 | - | 128 bit | 推奨 |

### 鍵の生成 (Python)

```python
import os
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec, rsa, ed25519, x25519
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hashes import SHA256

# =============================================
# 1. 対称鍵の生成
# =============================================

# AES-256 用の 256 ビット鍵 (暗号学的に安全な乱数)
aes_key = os.urandom(32)  # 32 bytes = 256 bits

# os.urandom は以下のソースを使用:
#   Linux: getrandom(2) → /dev/urandom
#   macOS: getentropy(2)
#   Windows: CryptGenRandom
# いずれも CSPRNG (暗号学的疑似乱数生成器)

# NG: random モジュール (予測可能)
# import random
# bad_key = bytes([random.randint(0, 255) for _ in range(32)])

# =============================================
# 2. 非対称鍵の生成
# =============================================

# RSA 4096 ビット (レガシー互換性が必要な場合)
rsa_private = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
)

# ECDSA P-256 (一般的な用途)
ec_private = ec.generate_private_key(ec.SECP256R1())

# Ed25519 (推奨: 高速、安全、定数時間)
ed_private = ed25519.Ed25519PrivateKey.generate()

# X25519 (鍵交換用)
x_private = x25519.X25519PrivateKey.generate()

# =============================================
# 3. 秘密鍵の暗号化保存
# =============================================

# パスフレーズで暗号化して PEM 形式で保存
pem_encrypted = ec_private.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(
        b"strong-passphrase-here"
    ),
)

# 暗号化なしの PEM (HSM内 or メモリ上のみ使用)
pem_unencrypted = ec_private.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

# 公開鍵の導出とエクスポート
public_pem = ec_private.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

# =============================================
# 4. 鍵導出 (HKDF)
# =============================================

# マスターシークレットから複数の用途別鍵を導出
master_secret = os.urandom(32)

def derive_key(master: bytes, purpose: str, length: int = 32) -> bytes:
    """HKDF で用途別の鍵を導出"""
    return HKDF(
        algorithm=SHA256(),
        length=length,
        salt=None,  # ランダムソルト推奨 (本番では固定しない)
        info=purpose.encode(),
    ).derive(master)

encryption_key = derive_key(master_secret, "encryption")
signing_key = derive_key(master_secret, "signing")
auth_key = derive_key(master_secret, "authentication")

# 重要: 1つのマスターから用途別に導出することで、
# 鍵の使い回し (key reuse) を防ぐ
```

---

## 3. HSM (Hardware Security Module)

### HSM の内部アーキテクチャ

```
+---------------------------------------------------+
|                  アプリケーション                     |
|---------------------------------------------------|
|  1. 鍵生成リクエスト      4. 署名リクエスト          |
|  2. 暗号化リクエスト      5. 鍵ラッピング            |
|  3. 復号リクエスト        6. HMAC 演算              |
+---------------------------------------------------+
            |  PKCS#11 / JCE / CNG / KMIP API
            v
+---------------------------------------------------+
|                    HSM                             |
|---------------------------------------------------|
|  [耐タンパ性ハードウェア]                            |
|  物理筐体は耐タンパ加工 (開封→鍵消去)               |
|                                                   |
|  +-- 鍵ストレージ                                  |
|  |   暗号化された不揮発性メモリ                      |
|  |   鍵がHSM外に出ない (non-extractable)           |
|  |                                                |
|  +-- 暗号エンジン                                  |
|  |   専用ASICチップで暗号演算                       |
|  |   AES-NI 相当の高速処理                         |
|  |   RSA 署名: ~1,000 ops/sec                    |
|  |   ECDSA P-256: ~5,000 ops/sec                |
|  |                                                |
|  +-- 乱数生成器 (TRNG)                            |
|  |   物理的エントロピー源 (熱雑音等)                 |
|  |   NIST SP 800-90B 準拠                        |
|  |                                                |
|  +-- 監査ログ                                     |
|  |   全操作を内部ログに記録                          |
|  |   改竄不可能                                    |
|  |                                                |
|  +-- FIPS 140-2/3 Level 3 認証                    |
|      Level 2: 物理的不正開封の検知                  |
|      Level 3: 不正開封時の鍵消去                    |
|      Level 4: 環境攻撃への耐性                     |
+---------------------------------------------------+
```

### HSM の種類と比較

| 種類 | 例 | 特徴 | コスト | 用途 |
|------|-----|------|--------|------|
| オンプレミス HSM | Thales Luna, nCipher | 完全管理、最高セキュリティ | $10K-$100K+ | 金融、政府 |
| クラウド HSM | AWS CloudHSM, GCP Cloud HSM | マネージド、FIPS Level 3 | $1-2/時間 | クラウドワークロード |
| USB HSM | YubiHSM 2 | 小型、低コスト | $650 | 小規模、開発、署名 |
| ソフトウェア HSM | SoftHSM 2 | 開発/テスト用 | 無料 | 開発環境のみ |
| クラウド KMS | AWS KMS, GCP Cloud KMS | フルマネージド | $1/鍵/月 | 汎用暗号化 |

### PKCS#11 を使った HSM 操作

```python
import pkcs11
from pkcs11 import KeyType, Mechanism, ObjectClass

# =============================================
# HSM ライブラリのロード
# =============================================

# SoftHSM 2 (開発用)
lib = pkcs11.lib("/usr/lib/softhsm/libsofthsm2.so")

# Thales Luna (本番用)
# lib = pkcs11.lib("/usr/lib/libCryptoki2_64.so")

# AWS CloudHSM
# lib = pkcs11.lib("/opt/cloudhsm/lib/libcloudhsm_pkcs11.so")

token = lib.get_token(token_label="my-token")

# =============================================
# セッション開始と鍵操作
# =============================================

with token.open(user_pin="1234") as session:
    # --- AES 鍵の生成 (HSM 内) ---
    aes_key = session.generate_key(
        KeyType.AES, 256,
        label="data-encryption-key-001",
        id=b"\x01\x00\x01",
        store=True,         # HSM に永続保存
        extractable=False,  # HSM 外への抽出を禁止
        sensitive=True,     # 平文での読み出し禁止
        encrypt=True,       # 暗号化に使用可
        decrypt=True,       # 復号に使用可
        wrap=False,         # 他の鍵のラッピングには使用不可
    )

    # --- HSM 内で暗号化 (AES-CBC-PAD) ---
    plaintext = b"Sensitive data that must be encrypted"
    iv = session.generate_random(128)  # 128 bit IV
    ciphertext = aes_key.encrypt(
        plaintext,
        mechanism=Mechanism.AES_CBC_PAD,
        mechanism_param=iv,
    )

    # --- HSM 内で復号 ---
    decrypted = aes_key.decrypt(
        ciphertext,
        mechanism=Mechanism.AES_CBC_PAD,
        mechanism_param=iv,
    )
    assert decrypted == plaintext

    # --- RSA 鍵ペアの生成 (HSM 内) ---
    pub_key, priv_key = session.generate_keypair(
        KeyType.RSA, 4096,
        label="signing-key-001",
        store=True,
        private={
            "extractable": False,
            "sensitive": True,
            "sign": True,
        },
        public={
            "verify": True,
        },
    )

    # --- HSM 内でデジタル署名 ---
    message = b"Data to be signed"
    signature = priv_key.sign(
        message,
        mechanism=Mechanism.SHA256_RSA_PKCS,
    )

    # --- 署名検証 ---
    pub_key.verify(
        message,
        signature,
        mechanism=Mechanism.SHA256_RSA_PKCS,
    )

    # --- 鍵のラッピング (KEK で DEK を暗号化) ---
    # ラッピング用 KEK を生成
    kek = session.generate_key(
        KeyType.AES, 256,
        label="key-encryption-key-001",
        store=True,
        extractable=False,
        wrap=True,      # ラッピングに使用可
        unwrap=True,    # アンラッピングに使用可
    )

    # DEK を KEK でラップ (暗号化して取り出し)
    wrapped_dek = kek.wrap_key(
        aes_key,
        mechanism=Mechanism.AES_KEY_WRAP,
    )
    # wrapped_dek は HSM の外に安全に保存可能

    # 必要時にアンラップ (HSM 内で復元)
    restored_dek = kek.unwrap_key(
        ObjectClass.SECRET_KEY,
        KeyType.AES,
        wrapped_dek,
        mechanism=Mechanism.AES_KEY_WRAP,
        template={
            "encrypt": True,
            "decrypt": True,
            "extractable": False,
        },
    )
```

### HSM のアクセス制御モデル

```
+----------------------------------------------------------+
|              HSM アクセス制御                               |
|----------------------------------------------------------|
|                                                          |
|  [認証方式]                                               |
|  +-- PIN/パスワード: 基本的な認証                         |
|  +-- M of N 認証: N人中M人の承認が必要                    |
|  |   例: 3 of 5 (管理者5人中3人が同時に認証)              |
|  +-- スマートカード + PIN: 二要素認証                     |
|                                                          |
|  [ロール]                                                |
|  +-- Security Officer (SO): HSM の管理・設定             |
|  +-- Crypto Officer (CO): 鍵の生成・削除                 |
|  +-- Crypto User (CU): 鍵の使用 (暗号化/署名)           |
|  +-- Auditor: 監査ログの閲覧のみ                        |
|                                                          |
|  [ポリシー]                                              |
|  +-- 鍵属性: extractable, sensitive, modifiable          |
|  +-- 操作制限: encrypt, decrypt, sign, wrap              |
|  +-- 時間制限: 使用可能時間帯の設定                       |
|  +-- 使用回数制限: 最大使用回数の設定                     |
+----------------------------------------------------------+
```

---

## 4. クラウド KMS

### AWS KMS の基本操作

```python
import boto3
import base64
import json

kms = boto3.client("kms", region_name="ap-northeast-1")

# =============================================
# 1. カスタマーマスターキー (CMK) の作成
# =============================================

response = kms.create_key(
    Description="Application data encryption key",
    KeyUsage="ENCRYPT_DECRYPT",
    KeySpec="SYMMETRIC_DEFAULT",  # AES-256-GCM
    Origin="AWS_KMS",             # KMS 生成 (推奨)
    # Origin="EXTERNAL",          # 外部から鍵をインポート
    MultiRegion=False,
    Tags=[
        {"TagKey": "Environment", "TagValue": "production"},
        {"TagKey": "Application", "TagValue": "user-data-service"},
        {"TagKey": "Compliance", "TagValue": "PCI-DSS"},
    ],
)
key_id = response["KeyMetadata"]["KeyId"]
key_arn = response["KeyMetadata"]["Arn"]

# エイリアスの設定 (人間が読める名前)
kms.create_alias(
    AliasName="alias/user-data-key",
    TargetKeyId=key_id,
)

# =============================================
# 2. 鍵ポリシーの設定
# =============================================

key_policy = {
    "Version": "2012-10-17",
    "Id": "user-data-key-policy",
    "Statement": [
        {
            "Sid": "Enable IAM policies",
            "Effect": "Allow",
            "Principal": {"AWS": f"arn:aws:iam::123456789012:root"},
            "Action": "kms:*",
            "Resource": "*",
        },
        {
            "Sid": "Allow application to encrypt/decrypt",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::123456789012:role/app-server-role"
            },
            "Action": [
                "kms:Encrypt",
                "kms:Decrypt",
                "kms:GenerateDataKey",
                "kms:GenerateDataKeyWithoutPlaintext",
                "kms:DescribeKey",
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "kms:EncryptionContext:purpose": "user-data",
                },
            },
        },
        {
            "Sid": "Deny direct data encryption (enforce envelope)",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "kms:Encrypt",
            "Resource": "*",
            "Condition": {
                "NumericGreaterThan": {
                    "kms:DataKeySpec": "4096",
                },
            },
        },
    ],
}

kms.put_key_policy(
    KeyId=key_id,
    PolicyName="default",
    Policy=json.dumps(key_policy),
)

# =============================================
# 3. データの暗号化 (直接)
# =============================================

# 小さなデータ (<4KB) は直接暗号化可能
encrypt_response = kms.encrypt(
    KeyId="alias/user-data-key",
    Plaintext=b"Secret data",
    EncryptionContext={
        "purpose": "user-data",
        "tenant": "acme-corp",
        "user_id": "user-12345",
    },
    # EncryptionContext は AAD (Additional Authenticated Data) として機能
    # 復号時に同じ Context を指定しないと復号できない
)
ciphertext = encrypt_response["CiphertextBlob"]

# =============================================
# 4. データの復号
# =============================================

decrypt_response = kms.decrypt(
    CiphertextBlob=ciphertext,
    EncryptionContext={
        "purpose": "user-data",
        "tenant": "acme-corp",
        "user_id": "user-12345",
    },
    # EncryptionContext が不一致 → InvalidCiphertextException
)
plaintext = decrypt_response["Plaintext"]

# =============================================
# 5. 非対称鍵の操作
# =============================================

# 署名用 RSA 鍵の作成
sign_key = kms.create_key(
    Description="API signing key",
    KeyUsage="SIGN_VERIFY",
    KeySpec="RSA_4096",
    Tags=[{"TagKey": "Purpose", "TagValue": "api-signing"}],
)
sign_key_id = sign_key["KeyMetadata"]["KeyId"]

# データの署名
import hashlib
message = b"Important message to sign"
digest = hashlib.sha256(message).digest()

sign_response = kms.sign(
    KeyId=sign_key_id,
    Message=digest,
    MessageType="DIGEST",
    SigningAlgorithm="RSASSA_PKCS1_V1_5_SHA_256",
)
signature = sign_response["Signature"]

# 署名の検証
verify_response = kms.verify(
    KeyId=sign_key_id,
    Message=digest,
    MessageType="DIGEST",
    Signature=signature,
    SigningAlgorithm="RSASSA_PKCS1_V1_5_SHA_256",
)
assert verify_response["SignatureValid"] is True
```

### KMS 比較表 (詳細)

| 項目 | AWS KMS | GCP Cloud KMS | Azure Key Vault |
|------|---------|---------------|-----------------|
| HSM バックエンド | CloudHSM 連携 | Cloud HSM | Managed HSM |
| FIPS 認証 | 140-2 Level 2 (標準), Level 3 (CloudHSM) | 140-2 Level 3 | 140-2 Level 2 (標準), Level 3 (Premium) |
| 自動ローテーション | 年次 (対称鍵のみ) | カスタム期間 | カスタム期間 |
| 料金 (鍵/月) | $1 (対称), $1 (非対称) | $0.06 (ソフトウェア), $1-2.50 (HSM) | $0.03/10,000操作 |
| 料金 (リクエスト) | $0.03/10,000 | $0.03/10,000 | $0.03/10,000 |
| 鍵のインポート | 可能 (BYOK) | 可能 (BYOK) | 可能 (BYOK) |
| マルチリージョン | マルチリージョンキー | グローバルリソース | geo レプリケーション |
| 暗号化コンテキスト | EncryptionContext (AAD) | Additional AAD | - |
| 鍵ポリシー | KMS Key Policy + IAM | IAM | RBAC + Access Policy |
| CloudTrail 統合 | 全操作をログ | Cloud Audit Logs | Activity Log |
| 最大データサイズ (直接) | 4 KB | 64 KB | - |

### GCP Cloud KMS の使用例

```python
from google.cloud import kms

def encrypt_with_gcp_kms(
    project_id: str,
    location_id: str,
    key_ring_id: str,
    key_id: str,
    plaintext: bytes,
) -> bytes:
    """GCP Cloud KMS でデータを暗号化"""
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(
        project_id, location_id, key_ring_id, key_id
    )

    # 暗号化 (AAD 付き)
    response = client.encrypt(
        request={
            "name": key_name,
            "plaintext": plaintext,
            "additional_authenticated_data": b"context-info",
        }
    )
    return response.ciphertext

def decrypt_with_gcp_kms(
    project_id: str,
    location_id: str,
    key_ring_id: str,
    key_id: str,
    ciphertext: bytes,
) -> bytes:
    """GCP Cloud KMS でデータを復号"""
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(
        project_id, location_id, key_ring_id, key_id
    )

    response = client.decrypt(
        request={
            "name": key_name,
            "ciphertext": ciphertext,
            "additional_authenticated_data": b"context-info",
        }
    )
    return response.plaintext
```

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
       | (KMS API)                     | (ローカル)
       |                               |
       v                               v
  +-----------+                   +-----------+
  | 暗号化DEK  |                   | 暗号化     |
  | (メタデータ |                   | データ     |
  |  に保存)   |                   +-----------+
  +-----------+                        |
       |                               |
       +------- 一緒に保存 ------------+

復号フロー:

  +-----------+                   +-----------+
  | 暗号化DEK  |                   | 暗号化     |
  +-----------+                   | データ     |
       |                          +-----------+
       | KMS に送信                      |
       v                               |
  +-----------+     DEK (平文)          |
  |  KMS      | ---- 復号 ----→        |
  | (マスター  |                   DEK で復号
  |   鍵)     |                   (ローカル)
  +-----------+                        |
                                       v
                                 +-----------+
                                 | 平文データ  |
                                 +-----------+

メリット:
  1. 大量データを KMS に送信する必要がない (レイテンシ削減)
  2. KMS の API 呼び出し回数を最小化 (コスト削減)
  3. マスター鍵が KMS/HSM の外に出ない (セキュリティ)
  4. DEK のローテーションが容易 (新DEKで再暗号化)
```

### エンベロープ暗号化の実装 (本番品質)

```python
import boto3
import os
import json
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dataclasses import dataclass
from typing import Optional

@dataclass
class EncryptedBundle:
    """暗号化結果のバンドル"""
    encrypted_dek: bytes    # KMS で暗号化された DEK
    nonce: bytes            # AES-GCM のナンス (12 bytes)
    ciphertext: bytes       # 暗号化されたデータ
    key_id: str             # 使用した KMS キーのARN
    algorithm: str          # 使用した暗号アルゴリズム
    context: dict           # 暗号化コンテキスト

    def serialize(self) -> bytes:
        """バイナリフォーマットにシリアライズ"""
        # フォーマット: [version][dek_len][dek][nonce][ciphertext]
        version = struct.pack("B", 1)  # v1
        dek_len = struct.pack(">H", len(self.encrypted_dek))
        return version + dek_len + self.encrypted_dek + self.nonce + self.ciphertext

    @classmethod
    def deserialize(cls, data: bytes, key_id: str, context: dict) -> "EncryptedBundle":
        """バイナリフォーマットからデシリアライズ"""
        version = struct.unpack("B", data[0:1])[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        dek_len = struct.unpack(">H", data[1:3])[0]
        encrypted_dek = data[3:3+dek_len]
        nonce = data[3+dek_len:3+dek_len+12]
        ciphertext = data[3+dek_len+12:]
        return cls(
            encrypted_dek=encrypted_dek,
            nonce=nonce,
            ciphertext=ciphertext,
            key_id=key_id,
            algorithm="AES-256-GCM",
            context=context,
        )


class EnvelopeEncryption:
    """エンベロープ暗号化の本番品質実装"""

    def __init__(self, kms_key_id: str, region: str = "ap-northeast-1"):
        self.kms = boto3.client("kms", region_name=region)
        self.kms_key_id = kms_key_id

    def encrypt(self, plaintext: bytes, context: dict) -> EncryptedBundle:
        """エンベロープ暗号化でデータを暗号化"""
        # Step 1: KMS から DEK を取得
        response = self.kms.generate_data_key(
            KeyId=self.kms_key_id,
            KeySpec="AES_256",
            EncryptionContext=context,
        )
        dek_plaintext = response["Plaintext"]       # 平文 DEK
        dek_encrypted = response["CiphertextBlob"]  # 暗号化 DEK

        try:
            # Step 2: DEK でデータをローカル暗号化 (AES-256-GCM)
            nonce = os.urandom(12)  # 96-bit nonce (GCM推奨)
            aesgcm = AESGCM(dek_plaintext)
            # AAD として暗号化コンテキストを含める
            aad = json.dumps(context, sort_keys=True).encode()
            ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

            return EncryptedBundle(
                encrypted_dek=dek_encrypted,
                nonce=nonce,
                ciphertext=ciphertext,
                key_id=self.kms_key_id,
                algorithm="AES-256-GCM",
                context=context,
            )
        finally:
            # Step 3: 平文 DEK をメモリから確実に消去
            # Python ではガベージコレクタに依存するため完全な消去は困難
            # ctypes でメモリを直接ゼロ化する方法もある
            dek_plaintext = b"\x00" * len(dek_plaintext)
            del dek_plaintext

    def decrypt(self, bundle: EncryptedBundle) -> bytes:
        """暗号化されたデータを復号"""
        # Step 1: KMS で DEK を復号
        response = self.kms.decrypt(
            CiphertextBlob=bundle.encrypted_dek,
            EncryptionContext=bundle.context,
        )
        dek_plaintext = response["Plaintext"]

        try:
            # Step 2: DEK でデータをローカル復号
            aesgcm = AESGCM(dek_plaintext)
            aad = json.dumps(bundle.context, sort_keys=True).encode()
            plaintext = aesgcm.decrypt(bundle.nonce, bundle.ciphertext, aad)
            return plaintext
        finally:
            dek_plaintext = b"\x00" * len(dek_plaintext)
            del dek_plaintext

    def reencrypt_with_new_key(self, bundle: EncryptedBundle,
                                new_key_id: str) -> EncryptedBundle:
        """別の KMS キーで再暗号化 (鍵ローテーション用)"""
        # 旧キーで復号 → 新キーで暗号化
        plaintext = self.decrypt(bundle)
        old_key = self.kms_key_id
        self.kms_key_id = new_key_id
        try:
            return self.encrypt(plaintext, bundle.context)
        finally:
            self.kms_key_id = old_key
            plaintext = b"\x00" * len(plaintext)
            del plaintext


# 使用例
if __name__ == "__main__":
    crypto = EnvelopeEncryption(kms_key_id="alias/user-data-key")

    # 暗号化
    context = {"tenant": "acme", "purpose": "pii-data"}
    bundle = crypto.encrypt(b"John Doe, SSN: 123-45-6789", context)

    # 復号
    decrypted = crypto.decrypt(bundle)
    print(f"Decrypted: {decrypted.decode()}")

    # シリアライズ/デシリアライズ (ストレージ保存用)
    serialized = bundle.serialize()
    restored = EncryptedBundle.deserialize(
        serialized, "alias/user-data-key", context
    )
    assert crypto.decrypt(restored) == b"John Doe, SSN: 123-45-6789"
```

### エンベロープ暗号化 vs 直接暗号化

| 項目 | エンベロープ暗号化 | 直接 KMS 暗号化 |
|------|------------------|----------------|
| データサイズ制限 | 無制限 | AWS: 4KB, GCP: 64KB |
| レイテンシ | 低い (ローカル暗号化) | 高い (全データをKMSに送信) |
| KMS API 呼び出し | 1回 (DEK生成) | データ量に比例 |
| コスト | 低い | 高い (大量データ時) |
| マスター鍵の露出 | なし (KMS内) | なし (KMS内) |
| DEK のローテーション | 容易 (新DEKで再暗号化) | 不要 (KMS管理) |
| 適用シーン | 大量データ、ファイル | 小さなシークレット、トークン |

---

## 6. 鍵ローテーション

### 自動ローテーション戦略

```
鍵ローテーションの方式:

方式1: KMS 自動ローテーション (推奨)
+------------------------------------------+
| KMS Key (alias/my-key)                   |
|                                          |
| v1 (2024-01) → 暗号化/復号 (現行)        |
| v2 (2025-01) → 暗号化/復号 (自動生成)     |
| v3 (2026-01) → 暗号化 (最新)              |
|                                          |
| - エイリアスは変更なし                     |
| - 旧バージョンの鍵で暗号化されたデータも    |
|   同じエイリアスで復号可能                  |
| - アプリケーション変更不要                  |
+------------------------------------------+

方式2: 手動ローテーション (エイリアス切り替え)
+------------------------------------------+
| Step 1: 新しい鍵を作成                     |
| Step 2: エイリアスを新しい鍵に向ける        |
| Step 3: 旧鍵は復号のために残す             |
| Step 4: 必要に応じて旧データを再暗号化      |
|                                          |
| alias/my-key → key-id-NEW (暗号化)       |
| key-id-OLD は復号のみ (無効化しない)       |
+------------------------------------------+

方式3: ダブルライト + バッチ移行
+------------------------------------------+
| Phase 1: 新旧両方の鍵で暗号化 (ダブルライト)|
| Phase 2: 読み取り時に旧鍵データを新鍵で    |
|          再暗号化 (Lazy Migration)          |
| Phase 3: バッチで残りの旧データを移行       |
| Phase 4: 旧鍵を無効化 → 廃棄             |
+------------------------------------------+
```

### 鍵ローテーションの実装

```python
import boto3
from datetime import datetime, timedelta
from typing import Optional

kms = boto3.client("kms")

class KeyRotationManager:
    """KMS 鍵ローテーションの管理"""

    def __init__(self, region: str = "ap-northeast-1"):
        self.kms = boto3.client("kms", region_name=region)

    def enable_auto_rotation(self, key_id: str, rotation_period_days: int = 365):
        """自動ローテーションを有効化"""
        self.kms.enable_key_rotation(KeyId=key_id)

        # ローテーション状態の確認
        status = self.kms.get_key_rotation_status(KeyId=key_id)
        print(f"自動ローテーション: {status['KeyRotationEnabled']}")

        # カスタムローテーション期間 (AWS KMS は 90-2560 日)
        if rotation_period_days != 365:
            self.kms.update_key_rotation_period(
                KeyId=key_id,
                RotationPeriodInDays=rotation_period_days,
            )

    def manual_rotate(self, alias: str) -> str:
        """手動ローテーション (新しい鍵を作成してエイリアスを切り替え)"""
        # 旧鍵の情報を取得
        old_key = self.kms.describe_key(KeyId=f"alias/{alias}")
        old_key_id = old_key["KeyMetadata"]["KeyId"]

        # 新しい鍵を作成
        new_key = self.kms.create_key(
            Description=f"Rotated key - {datetime.now().isoformat()}",
            KeyUsage=old_key["KeyMetadata"]["KeyUsage"],
            KeySpec=old_key["KeyMetadata"]["KeySpec"],
            Tags=[
                {"TagKey": "RotatedFrom", "TagValue": old_key_id},
                {"TagKey": "RotatedAt", "TagValue": datetime.now().isoformat()},
            ],
        )
        new_key_id = new_key["KeyMetadata"]["KeyId"]

        # エイリアスを新しい鍵に向ける (アトミック)
        self.kms.update_alias(
            AliasName=f"alias/{alias}",
            TargetKeyId=new_key_id,
        )

        print(f"ローテーション完了: {old_key_id} → {new_key_id}")
        print(f"旧鍵 {old_key_id} は復号のために残存")

        return new_key_id

    def emergency_rotate(self, alias: str, reason: str) -> str:
        """緊急ローテーション (鍵の危殆化時)"""
        print(f"[EMERGENCY] 緊急ローテーション開始: {reason}")

        # 新しい鍵を即座に作成
        new_key_id = self.manual_rotate(alias)

        # 旧鍵をスケジュール無効化 (即座には無効化しない)
        # 理由: 旧鍵で暗号化されたデータの復号が必要なため
        old_key = self.kms.describe_key(KeyId=f"alias/{alias}")

        # インシデント記録
        print(f"[EMERGENCY] 新鍵: {new_key_id}")
        print(f"[EMERGENCY] 旧データの再暗号化が必要")
        print(f"[EMERGENCY] 理由: {reason}")

        return new_key_id

    def audit_key_ages(self) -> list[dict]:
        """全鍵の年齢を監査"""
        aging_keys = []
        paginator = self.kms.get_paginator("list_keys")

        for page in paginator.paginate():
            for key in page["Keys"]:
                try:
                    metadata = self.kms.describe_key(
                        KeyId=key["KeyId"]
                    )["KeyMetadata"]

                    if metadata["KeyState"] != "Enabled":
                        continue

                    age_days = (datetime.now(metadata["CreationDate"].tzinfo)
                               - metadata["CreationDate"]).days

                    if age_days > 365:
                        aging_keys.append({
                            "key_id": key["KeyId"],
                            "description": metadata.get("Description", ""),
                            "age_days": age_days,
                            "rotation_enabled": self.kms.get_key_rotation_status(
                                KeyId=key["KeyId"]
                            ).get("KeyRotationEnabled", False),
                        })
                except Exception:
                    continue

        return aging_keys
```

### 鍵ローテーション Terraform 設定

```hcl
# AWS KMS キーの作成と自動ローテーション
resource "aws_kms_key" "data_key" {
  description             = "Data encryption key for user service"
  deletion_window_in_days = 30    # 削除猶予期間
  enable_key_rotation     = true  # 自動ローテーション
  rotation_period_in_days = 365   # ローテーション間隔

  # 鍵ポリシー
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM policies"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow application access"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.app_role.arn
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey",
        ]
        Resource = "*"
      },
    ]
  })

  tags = {
    Environment = "production"
    Compliance  = "PCI-DSS"
    ManagedBy   = "terraform"
  }
}

resource "aws_kms_alias" "data_key" {
  name          = "alias/user-data-key"
  target_key_id = aws_kms_key.data_key.key_id
}

# CloudWatch アラーム: 鍵の使用異常検知
resource "aws_cloudwatch_metric_alarm" "kms_usage_anomaly" {
  alarm_name          = "kms-unusual-activity"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "NumberOfDecryptRequests"
  namespace           = "AWS/KMS"
  period              = 300
  statistic           = "Sum"
  threshold           = 1000    # 5分間で1000回以上の復号はアラート

  dimensions = {
    KeyId = aws_kms_key.data_key.key_id
  }

  alarm_actions = [aws_sns_topic.security_alerts.arn]
}
```

---

## 7. メモリ上の鍵の安全な取り扱い

### 鍵のメモリ安全性

```
脅威:
  1. メモリダンプ (core dump) に鍵が含まれる
  2. スワップ領域に鍵が書き出される
  3. GC (ガベージコレクタ) が遅延的にメモリを解放
  4. フォーク時に子プロセスにコピーされる
  5. サイドチャネル攻撃 (キャッシュタイミング等)

対策:
  +-- OS レベル ---+
  |  mlock()       |  メモリをスワップ不可にする
  |  madvise()     |  fork時にコピーしない
  |  core dump無効 |  /proc/sys/kernel/core_pattern
  +----------------+

  +-- アプリレベル --+
  |  即時ゼロ化     |  使用後にメモリをゼロで上書き
  |  スコープ制限   |  鍵の生存期間を最小化
  |  HSM利用       |  鍵をメモリに持たない (最善)
  +----------------+
```

### Python での安全な鍵取り扱い

```python
import ctypes
import os
import mmap
from contextlib import contextmanager

class SecureBuffer:
    """安全なメモリバッファ (鍵の一時保存用)"""

    def __init__(self, size: int):
        self.size = size
        # mmap で個別のメモリページを確保
        self._buf = mmap.mmap(-1, size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        # mlock でスワップ防止 (root 権限 or CAP_IPC_LOCK 必要)
        try:
            ctypes.CDLL("libc.so.6").mlock(
                ctypes.c_void_p(id(self._buf)),
                ctypes.c_size_t(size),
            )
        except (OSError, AttributeError):
            pass  # mlock 失敗時は続行 (macOS等)

    def write(self, data: bytes):
        """データをバッファに書き込み"""
        if len(data) > self.size:
            raise ValueError("Data exceeds buffer size")
        self._buf.seek(0)
        self._buf.write(data)

    def read(self) -> bytes:
        """バッファからデータを読み出し"""
        self._buf.seek(0)
        return self._buf.read(self.size)

    def zero(self):
        """バッファをゼロで上書き (安全な消去)"""
        self._buf.seek(0)
        self._buf.write(b"\x00" * self.size)

    def __del__(self):
        """デストラクタでゼロ化を保証"""
        try:
            self.zero()
            self._buf.close()
        except Exception:
            pass

@contextmanager
def secure_key(key_data: bytes):
    """鍵の安全な一時保持 (コンテキストマネージャ)"""
    buf = SecureBuffer(len(key_data))
    buf.write(key_data)
    try:
        yield buf.read()
    finally:
        buf.zero()
        # 元のデータもゼロ化
        ctypes.memset(id(key_data) + 32, 0, len(key_data))

# 使用例
# with secure_key(dek_plaintext) as key:
#     aesgcm = AESGCM(key)
#     ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
# # ← ここで key はゼロ化済み
```

---

## 8. アンチパターン

### アンチパターン 1: 鍵のハードコーディング

```python
# NG: ソースコードに暗号鍵を直書き
ENCRYPTION_KEY = b"0123456789abcdef0123456789abcdef"
SECRET_API_KEY = "sk_live_abcdef123456"

# NG: 設定ファイルに平文で保存
# config.yaml:
#   encryption_key: "0123456789abcdef"

# NG: 環境変数に長期保存 (ローテーション不可)
# export ENCRYPTION_KEY="0123456789abcdef"

# OK: KMS から動的に取得
import os
key_ref = os.environ["KMS_KEY_ARN"]  # ARN のみ保持
# KMS API 経由で暗号化・復号を行う (鍵自体はメモリに持たない)

# OK: Secrets Manager から取得 (自動ローテーション付き)
import boto3
sm = boto3.client("secretsmanager")
response = sm.get_secret_value(SecretId="myapp/api-key")
api_key = response["SecretString"]
```

**影響**: Git 履歴に鍵が残り、ローテーションが不可能になる。漏洩時に全データが危殆化する。GitHub の調査によると、パブリックリポジトリの約 10% に何らかのシークレットがコミットされている。

### アンチパターン 2: 鍵とデータの同一ストレージ保管

```
NG:
  S3 バケット/
    ├── data/encrypted-data.bin
    └── keys/encryption-key.txt     ← 同じバケットに鍵を保管
  (S3バケットが漏洩 → 鍵もデータも同時に流出)

NG:
  データベース/
    ├── users テーブル (暗号化カラム)
    └── encryption_keys テーブル      ← 同じDBに鍵を保管
  (SQLi → 鍵もデータも取得可能)

OK:
  S3 バケット (データ)/
    └── data/encrypted-data.bin     ← 暗号化 DEK をメタデータに格納
  KMS (鍵)/
    └── マスターキー                  ← 別サービスで管理
  (S3が漏洩しても暗号化DEKの復号にはKMSアクセスが必要)

OK:
  データベース/
    └── users テーブル (暗号化データ + 暗号化DEK)
  AWS KMS/
    └── CMK (マスター鍵)              ← IAMで厳密にアクセス制御
```

**影響**: S3 バケットが漏洩した場合、鍵もデータも同時に流出し暗号化の意味がなくなる。暗号化は「鍵がなければ読めない」ことが前提であり、鍵とデータを分離しなければその前提が崩れる。

### アンチパターン 3: 鍵のローテーションなし

```
NG:
  → 5年前に作成した AES 鍵をそのまま使い続ける
  → 「動いているから変えない」
  → ローテーション手順が未整備
  → 退職者がアクセスしていた鍵が残存

OK:
  → KMS の自動ローテーションを有効化
  → ローテーション手順を文書化・自動化
  → 定期的な監査 (鍵の年齢チェック)
  → 退職/異動時に鍵のアクセス見直し

コンプライアンス要件:
  PCI DSS: 年次ローテーション必須
  NIST SP 800-57: 対称鍵は最大2年
  HIPAA: 適切な鍵管理の要求
```

---

## 9. 演習問題

### 演習 1 (基礎): 鍵生成と暗号化

**課題**: 以下を Python で実装せよ。
1. AES-256-GCM 用の鍵を CSPRNG で生成
2. 任意のテキストを暗号化・復号
3. AAD (Additional Authenticated Data) を含める
4. ナンスの一意性を保証する方法を説明

**ヒント**: `os.urandom()` と `cryptography` ライブラリの `AESGCM` を使用。

### 演習 2 (応用): エンベロープ暗号化の設計

**課題**: 以下の要件を満たすエンベロープ暗号化システムを設計せよ。
- マルチテナント環境 (テナントごとに異なるマスター鍵)
- 鍵ローテーション時に既存データの再暗号化が必要
- 暗号化コンテキストにテナントID、データ種別を含める
- 不正アクセス時のアラート機構

**成果物**: クラス設計図 + シーケンス図

### 演習 3 (実践): KMS 鍵管理の自動化

**課題**: Terraform で以下の KMS 構成を構築せよ。
- 3 つの用途別 KMS キー (データ暗号化、署名、シークレット管理)
- 各キーに適切なキーポリシー
- 自動ローテーション有効化
- CloudWatch アラームによる異常検知
- 鍵の年齢を監査する Lambda 関数

**ヒント**: `aws_kms_key`, `aws_kms_alias`, `aws_cloudwatch_metric_alarm` を使用。

---

## 10. トラブルシューティング

### よくある問題と対処

```
問題1: KMS で復号できない (AccessDeniedException)
  → IAM ポリシーに kms:Decrypt が含まれているか確認
  → KMS キーポリシーに呼び出し元の Principal があるか確認
  → EncryptionContext が暗号化時と完全一致しているか確認
  → 鍵がDISABLED状態になっていないか確認

問題2: KMS で復号できない (InvalidCiphertextException)
  → EncryptionContext の不一致 (最も多い)
  → 暗号文の破損 (S3 転送時のエンコーディング問題)
  → 異なるリージョンの鍵で暗号化されたデータ
  → マルチリージョンキーの場合: レプリカキーの確認

問題3: HSM セッションエラー
  → PIN/パスワードの有効期限切れ
  → HSM パーティションの容量不足 (鍵数上限)
  → ネットワーク接続の問題 (CloudHSM)
  → HSM クラスタの同期状態の確認

問題4: 鍵ローテーション後に復号できない
  → 旧鍵が削除されていないか確認
  → エイリアスが正しい鍵を指しているか確認
  → 暗号文に鍵バージョン情報が含まれているか確認
  → KMS 自動ローテーションでは旧バージョンは自動保持される
```

### デバッグコマンド

```bash
# AWS KMS: 鍵の状態確認
aws kms describe-key --key-id alias/my-key \
  --query 'KeyMetadata.{KeyId:KeyId,KeyState:KeyState,CreationDate:CreationDate}'

# AWS KMS: ローテーション状態
aws kms get-key-rotation-status --key-id alias/my-key

# AWS KMS: 最近の鍵使用状況 (CloudTrail)
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceType,AttributeValue=AWS::KMS::Key \
  --max-results 20

# AWS KMS: 鍵ポリシーの確認
aws kms get-key-policy --key-id alias/my-key --policy-name default

# AWS CloudHSM: クラスタ状態
aws cloudhsmv2 describe-clusters \
  --query 'Clusters[].{Id:ClusterId,State:State,HSMs:Hsms[].{Id:HsmId,State:State}}'
```

---

## 11. パフォーマンス考慮事項

### KMS API のレイテンシとスループット

| 操作 | レイテンシ (P50) | レイテンシ (P99) | スループット上限 |
|------|----------------|----------------|----------------|
| Encrypt (対称) | 5-10 ms | 30 ms | 30,000 rps |
| Decrypt (対称) | 5-10 ms | 30 ms | 30,000 rps |
| GenerateDataKey | 5-10 ms | 30 ms | 30,000 rps |
| Sign (RSA-4096) | 50-100 ms | 200 ms | 500 rps |
| Sign (ECDSA) | 10-20 ms | 50 ms | 2,000 rps |

### パフォーマンス最適化

```python
# 1. DEK キャッシュ (エンベロープ暗号化時)
from functools import lru_cache
from datetime import datetime, timedelta

class CachedEnvelopeEncryption:
    """DEK をキャッシュしてKMS API呼び出しを最小化"""

    def __init__(self, kms_key_id: str):
        self.kms = boto3.client("kms")
        self.kms_key_id = kms_key_id
        self._dek_cache = {}
        self._cache_ttl = timedelta(minutes=5)

    def _get_or_generate_dek(self, context_key: str, context: dict):
        """キャッシュからDEKを取得、なければ新規生成"""
        now = datetime.utcnow()
        if context_key in self._dek_cache:
            cached = self._dek_cache[context_key]
            if now - cached["created_at"] < self._cache_ttl:
                return cached["plaintext"], cached["encrypted"]

        response = self.kms.generate_data_key(
            KeyId=self.kms_key_id,
            KeySpec="AES_256",
            EncryptionContext=context,
        )
        self._dek_cache[context_key] = {
            "plaintext": response["Plaintext"],
            "encrypted": response["CiphertextBlob"],
            "created_at": now,
        }
        return response["Plaintext"], response["CiphertextBlob"]

# 2. バッチ処理
# 複数データを同じDEKで暗号化 (ナンスは各データで一意)
# → KMS API呼び出し1回で複数データを処理
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| 鍵ライフサイクル | NIST SP 800-57 に基づき生成→有効化→運用→無効化→廃棄の全段階を管理 |
| 鍵の階層 | Master Key → KEK → DEK の三層構造、各層は分離保管 |
| HSM | 耐タンパ性ハードウェアで鍵の漏洩を物理的に防止、FIPS 140-2/3 認証 |
| KMS | クラウドマネージドで鍵管理の運用負荷を軽減、暗号化コンテキスト必須 |
| エンベロープ暗号化 | DEK + KEK の二層構造で大量データを効率的に暗号化 |
| 鍵ローテーション | 自動ローテーションを有効にし最低年次で実施、緊急時手順も準備 |
| 鍵の分離 | 鍵とデータは必ず別のストレージ・サービスで管理 |
| メモリ安全性 | 平文鍵は即座にゼロ化、mlock でスワップ防止、HSM利用が最善 |
| 監査 | 鍵の全操作を CloudTrail 等で記録、異常使用をアラート |

---

## 次に読むべきガイド

- [TLS/証明書](./01-tls-certificates.md) — TLS で鍵がどのように使われるかを理解する
- [クラウドセキュリティ基礎](../05-cloud-security/00-cloud-security-basics.md) — KMS を含むクラウドセキュリティの全体像
- [AWSセキュリティ](../05-cloud-security/01-aws-security.md) — AWS KMS・CloudHSM の実践的な活用
- [暗号基礎](./00-cryptography-basics.md) — 暗号アルゴリズムの理論的背景

---

## 参考文献

1. **NIST SP 800-57 Part 1 Rev.5 — Recommendation for Key Management** — https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final
2. **NIST SP 800-57 Part 2 Rev.1 — Best Practices for Key Management Organizations** — https://csrc.nist.gov/publications/detail/sp/800-57-part-2/rev-1/final
3. **AWS KMS Developer Guide** — https://docs.aws.amazon.com/kms/latest/developerguide/
4. **OWASP Cryptographic Storage Cheat Sheet** — https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
5. **Google Cloud KMS Documentation** — https://cloud.google.com/kms/docs
6. **PKCS#11 Specification** — https://docs.oasis-open.org/pkcs11/pkcs11-base/v2.40/pkcs11-base-v2.40.html
7. **FIPS 140-3 Standard** — https://csrc.nist.gov/publications/detail/fips/140/3/final
