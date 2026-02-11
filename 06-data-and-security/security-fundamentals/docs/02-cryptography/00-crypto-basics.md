# 暗号基礎

> 対称鍵暗号、非対称鍵暗号、ハッシュ関数、MAC、暗号化モード（AES-GCM）を体系的に解説し、安全な暗号実装の基盤を構築する。

## この章で学ぶこと

1. **対称鍵暗号と非対称鍵暗号**の仕組み、特徴、使い分けを理解する
2. **ハッシュ関数とMAC**の違いと適切な用途を習得する
3. **AES-GCM**による認証付き暗号の正しい実装方法を身につける

---

## 1. 暗号の分類

```
暗号技術の全体像:

                        暗号技術
                          |
            +-------------+-------------+
            |                           |
        暗号化                      ハッシュ/MAC
        (可逆)                      (不可逆)
            |                           |
    +-------+-------+           +-------+-------+
    |               |           |               |
  対称鍵          非対称鍵     ハッシュ関数      MAC
  (AES)          (RSA, ECC)   (SHA-256)      (HMAC)
    |               |
  共通鍵1つ      公開鍵+秘密鍵
  高速           低速だが鍵配送問題なし
```

---

## 2. 対称鍵暗号

同じ鍵で暗号化と復号を行う方式。高速で大量データの暗号化に適する。

```python
# コード例1: AES-GCMによる認証付き暗号
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class AESEncryptor:
    """AES-256-GCMによる認証付き暗号化"""

    KEY_SIZE = 32   # 256ビット
    NONCE_SIZE = 12 # 96ビット（GCM推奨）

    def __init__(self, key: bytes = None):
        if key is None:
            key = AESGCM.generate_key(bit_length=256)
        if len(key) != self.KEY_SIZE:
            raise ValueError(f"鍵長は{self.KEY_SIZE}バイトである必要があります")
        self._aesgcm = AESGCM(key)
        self._key = key

    def encrypt(self, plaintext: bytes,
                associated_data: bytes = None) -> bytes:
        """暗号化（ノンス + 暗号文 + 認証タグを返す）"""
        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext  # ノンスを先頭に付加

    def decrypt(self, data: bytes,
                associated_data: bytes = None) -> bytes:
        """復号（ノンスを分離してから復号）"""
        nonce = data[:self.NONCE_SIZE]
        ciphertext = data[self.NONCE_SIZE:]
        return self._aesgcm.decrypt(nonce, ciphertext, associated_data)

# 使用例
encryptor = AESEncryptor()
plaintext = b"This is a secret message"
aad = b"metadata:user_id=123"  # 追加認証データ（暗号化しないが改ざん検知）

encrypted = encryptor.encrypt(plaintext, aad)
decrypted = encryptor.decrypt(encrypted, aad)
assert decrypted == plaintext
```

### 暗号化モードの比較

| モード | 認証 | 並列処理 | パディング | 推奨度 |
|--------|:----:|:-------:|:---------:|:-----:|
| ECB | なし | 可 | 必要 | 使用禁止 |
| CBC | なし | 復号のみ | 必要 | 条件付き |
| CTR | なし | 可 | 不要 | 条件付き |
| GCM | あり | 可 | 不要 | 推奨 |
| CCM | あり | 不可 | 不要 | 可 |

```
ECBモードの問題（同一平文ブロック -> 同一暗号文ブロック）:

  平文:       [AAAA][BBBB][AAAA][CCCC]
  ECB暗号:    [xxxx][yyyy][xxxx][zzzz]  <- パターンが漏洩!
  CBC/GCM暗号:[abcd][efgh][ijkl][mnop]  <- パターンが隠蔽される
```

---

## 3. 非対称鍵暗号

公開鍵と秘密鍵のペアを使用する方式。鍵配送問題を解決するが、処理速度は遅い。

```python
# コード例2: RSAとECDSAの鍵生成・署名・検証
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization

class AsymmetricCrypto:
    """非対称鍵暗号の操作"""

    @staticmethod
    def generate_rsa_keypair(key_size: int = 4096):
        """RSA鍵ペアの生成"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        return private_key, private_key.public_key()

    @staticmethod
    def generate_ec_keypair():
        """ECC鍵ペアの生成（P-256曲線）"""
        private_key = ec.generate_private_key(ec.SECP256R1())
        return private_key, private_key.public_key()

    @staticmethod
    def rsa_sign(private_key, message: bytes) -> bytes:
        """RSA-PSS署名"""
        return private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    @staticmethod
    def rsa_verify(public_key, message: bytes, signature: bytes) -> bool:
        """RSA-PSS署名の検証"""
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

# 使用例
crypto = AsymmetricCrypto()
priv, pub = crypto.generate_rsa_keypair()
message = b"Important document"
sig = crypto.rsa_sign(priv, message)
print(crypto.rsa_verify(pub, message, sig))  # True
```

### 対称鍵暗号と非対称鍵暗号の比較

| 特性 | 対称鍵暗号 | 非対称鍵暗号 |
|------|-----------|------------|
| 鍵の数 | 1つ（共通鍵） | 2つ（公開鍵+秘密鍵） |
| 速度 | 高速（100-1000倍） | 低速 |
| 鍵配送 | 安全な経路が必要 | 公開鍵は自由に配布可 |
| 用途 | 大量データの暗号化 | 鍵交換、デジタル署名 |
| 代表例 | AES-256-GCM | RSA-4096, ECDSA P-256 |
| 鍵長 | 128/256ビット | 2048/4096ビット（RSA） |

---

## 4. ハッシュ関数

```python
# コード例3: ハッシュ関数の安全な使い方
import hashlib
import hmac
import secrets

class SecureHash:
    """安全なハッシュ操作"""

    @staticmethod
    def sha256(data: bytes) -> str:
        """SHA-256ハッシュ（データの完全性検証用）"""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def file_hash(filepath: str, algorithm: str = "sha256") -> str:
        """大きなファイルのハッシュをストリーミング計算"""
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """タイミング攻撃を防ぐ比較"""
        return hmac.compare_digest(a.encode(), b.encode())
```

```
ハッシュ関数の特性:

  入力（任意長）    --> ハッシュ関数 --> 出力（固定長）

  "hello"           --> SHA-256     --> 2cf24dba5fb0a30e...
  "hello!"          --> SHA-256     --> ce06092fb948d9ff...
  (1ビットの変化で出力が大きく変化 = 雪崩効果)

  必須特性:
  1. 一方向性     : ハッシュ値から元データを復元不可
  2. 衝突耐性     : 同じハッシュ値を持つ異なる入力を見つけるのが困難
  3. 第二原像耐性 : 特定の入力と同じハッシュ値を持つ別の入力を見つけるのが困難
```

---

## 5. MAC（Message Authentication Code）

```python
# コード例4: HMACの実装と使い方
import hmac
import hashlib
import time

class MessageAuthenticator:
    """HMACによるメッセージ認証"""

    def __init__(self, key: bytes):
        self.key = key

    def create_mac(self, message: bytes) -> str:
        """メッセージのMACを計算"""
        return hmac.new(self.key, message, hashlib.sha256).hexdigest()

    def verify_mac(self, message: bytes, mac: str) -> bool:
        """MACを検証（タイミング攻撃対策済み）"""
        expected = self.create_mac(message)
        return hmac.compare_digest(expected, mac)

    def create_signed_message(self, message: bytes) -> bytes:
        """タイムスタンプ付き署名メッセージを生成"""
        timestamp = str(int(time.time())).encode()
        payload = timestamp + b"." + message
        mac = self.create_mac(payload)
        return payload + b"." + mac.encode()

    def verify_signed_message(self, signed: bytes,
                               max_age: int = 300) -> bytes:
        """署名メッセージを検証（有効期限チェック付き）"""
        parts = signed.rsplit(b".", 1)
        if len(parts) != 2:
            raise ValueError("Invalid signed message format")

        payload, mac = parts[0], parts[1].decode()
        if not self.verify_mac(payload, mac):
            raise ValueError("MAC verification failed")

        timestamp_str, message = payload.split(b".", 1)
        timestamp = int(timestamp_str)
        if time.time() - timestamp > max_age:
            raise ValueError("Message expired")

        return message

# 使用例
auth = MessageAuthenticator(b"secret-key-32-bytes-long!!!!!!!!")
signed = auth.create_signed_message(b"payment:100:USD")
original = auth.verify_signed_message(signed, max_age=60)
```

---

## 6. ハイブリッド暗号方式

実際のシステムでは、対称鍵暗号と非対称鍵暗号を組み合わせるハイブリッド方式が一般的である。

```python
# コード例5: ハイブリッド暗号（エンベロープ暗号化）
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class HybridEncryption:
    """ハイブリッド暗号方式（RSA + AES-GCM）"""

    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=4096
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, plaintext: bytes) -> dict:
        """ハイブリッド暗号化"""
        # 1. ランダムなAES鍵（データ鍵）を生成
        data_key = AESGCM.generate_key(bit_length=256)

        # 2. データ鍵でデータを暗号化（対称鍵暗号 = 高速）
        nonce = os.urandom(12)
        aesgcm = AESGCM(data_key)
        encrypted_data = aesgcm.encrypt(nonce, plaintext, None)

        # 3. データ鍵をRSA公開鍵で暗号化（非対称鍵暗号）
        encrypted_key = self.public_key.encrypt(
            data_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return {
            "encrypted_key": encrypted_key,
            "nonce": nonce,
            "encrypted_data": encrypted_data,
        }

    def decrypt(self, package: dict) -> bytes:
        """ハイブリッド復号"""
        # 1. RSA秘密鍵でデータ鍵を復号
        data_key = self.private_key.decrypt(
            package["encrypted_key"],
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        # 2. データ鍵でデータを復号
        aesgcm = AESGCM(data_key)
        return aesgcm.decrypt(
            package["nonce"], package["encrypted_data"], None
        )

```

```
ハイブリッド暗号のフロー:

  送信者                                    受信者
    |                                         |
    |-- 1. AESデータ鍵を生成                   |
    |-- 2. データ鍵でメッセージを暗号化(高速)   |
    |-- 3. 受信者の公開鍵でデータ鍵を暗号化     |
    |                                         |
    |-- [暗号化データ鍵] + [暗号化データ] -->  |
    |                                         |
    |                  4. 秘密鍵でデータ鍵を復号 |
    |                  5. データ鍵でデータを復号  |
```

---

## アンチパターン

### アンチパターン1: 独自暗号アルゴリズムの発明

「XOR暗号」や独自のシャッフルアルゴリズムを使うパターン。暗号の安全性は数十年にわたる学術的な検証によって証明されるものであり、独自実装は未知の脆弱性を含む可能性が極めて高い。AES-GCM等の標準アルゴリズムを使用すべきである。

### アンチパターン2: ECBモードの使用

AES-ECBモードは、同一の平文ブロックが同一の暗号文ブロックに変換されるため、データのパターンが漏洩する。AES-GCM等の認証付き暗号モードを使用すべきである。

---

## FAQ

### Q1: AES-128とAES-256のどちらを使うべきですか?

一般的にはAES-256が推奨される。AES-128も現時点で十分な安全性を持つが、量子コンピュータ時代の到来を考慮すると、AES-256の方がより長期的な安全マージンがある。

### Q2: ハッシュ関数は暗号化の代わりに使えますか?

使えない。ハッシュ関数は一方向関数であり、ハッシュ値から元データを復元できない。データの完全性検証やパスワード保存には適するが、復号が必要なユースケースでは暗号化を使用する。

### Q3: RSAの鍵長はどれくらい必要ですか?

2048ビット以上が最低要件、4096ビットが推奨。ただし、ECDSAのP-256曲線はRSA-3072と同等のセキュリティレベルを持ち、鍵サイズが小さく処理も高速なため、新規システムではECDSAの採用を検討すべきである。

---

## まとめ

| 技術 | 用途 | 推奨アルゴリズム |
|------|------|----------------|
| 対称鍵暗号 | データの暗号化 | AES-256-GCM |
| 非対称鍵暗号 | 鍵交換、デジタル署名 | RSA-4096, ECDSA P-256 |
| ハッシュ関数 | 完全性検証 | SHA-256, SHA-3 |
| MAC | メッセージ認証 | HMAC-SHA256 |
| パスワードハッシュ | パスワード保存 | Argon2id, bcrypt |

---

## 次に読むべきガイド

- [01-tls-certificates.md](./01-tls-certificates.md) -- TLSハンドシェイクと証明書管理
- [02-key-management.md](./02-key-management.md) -- 鍵のライフサイクルと管理手法
- [../01-web-security/04-auth-vulnerabilities.md](../01-web-security/04-auth-vulnerabilities.md) -- パスワードハッシュの実践

---

## 参考文献

1. NIST SP 800-175B: Guideline for Using Cryptographic Standards -- https://csrc.nist.gov/publications/detail/sp/800-175b/rev-1/final
2. Christof Paar & Jan Pelzl, "Understanding Cryptography" -- Springer
3. OWASP Cryptographic Storage Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
