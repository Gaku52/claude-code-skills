# 暗号基礎

> 対称鍵暗号、非対称鍵暗号、ハッシュ関数、MAC、暗号化モード（AES-GCM）、ハイブリッド暗号方式を体系的に解説し、安全な暗号実装の基盤を構築する。暗号技術はセキュリティの土台であり、「なぜそのアルゴリズムを選ぶのか」「なぜその使い方が危険なのか」を理解することが、安全なシステム設計の第一歩である。

## この章で学ぶこと

1. **対称鍵暗号と非対称鍵暗号**の仕組み、特徴、使い分けを理解する
2. **ハッシュ関数と MAC**の違いと適切な用途を習得する
3. **AES-GCM** による認証付き暗号の正しい実装方法を身につける
4. **ハイブリッド暗号方式**（エンベロープ暗号化）の設計パターンを理解する
5. **暗号の選定基準**とアンチパターンを把握し、実務で正しい判断ができるようになる

## 前提知識

- [セキュリティ概要](../00-basics/00-security-overview.md) -- CIA 三要素（機密性・完全性・可用性）
- [セキュリティ原則](../00-basics/02-security-principles.md) -- 最小権限・多層防御等の基本原則
- Python の基本的なプログラミング知識
- ビットとバイトの概念（128ビット = 16バイト等）

---

## 1. 暗号の分類と全体像

### 1.1 暗号技術の体系

暗号技術は大きく「可逆な処理（暗号化）」と「不可逆な処理（ハッシュ/MAC）」に分かれる。それぞれの技術がどのような場面で使われるかを理解することが重要である。

```
暗号技術の全体像:

                        暗号技術
                          |
            +-------------+-------------+
            |                           |
        暗号化                      ハッシュ/MAC
        (可逆)                      (不可逆)
        「元に戻せる」              「元に戻せない」
            |                           |
    +-------+-------+           +-------+-------+
    |               |           |               |
  対称鍵          非対称鍵     ハッシュ関数      MAC
  (AES)          (RSA, ECC)   (SHA-256)      (HMAC)
    |               |           |               |
  共通鍵1つ      公開鍵+秘密鍵  完全性検証      認証+完全性
  高速           低速          改ざん検知      鍵付き改ざん検知

  実際のシステムではこれらを組み合わせて使う:
  - TLS: 非対称鍵で鍵交換 → 対称鍵でデータ暗号化
  - デジタル署名: ハッシュ → 非対称鍵で署名
  - パスワード保存: 専用ハッシュ（Argon2id）
```

### 1.2 暗号技術の用途別選択ガイド

```
用途別の暗号技術選択:

  +------------------------------------------+
  | やりたいこと         → 使う技術           |
  |------------------------------------------|
  | データを秘匿したい   → AES-256-GCM       |
  | 改ざんを検知したい   → SHA-256 / HMAC    |
  | パスワードを保存     → Argon2id / bcrypt |
  | 鍵を安全に交換       → ECDH / RSA-OAEP  |
  | 文書に署名           → ECDSA / RSA-PSS  |
  | 通信を暗号化         → TLS 1.3          |
  | 大量データを暗号化   → エンベロープ暗号化  |
  +------------------------------------------+
```

| 用途 | 推奨アルゴリズム | 鍵長 | 備考 |
|------|----------------|------|------|
| データ暗号化 | AES-256-GCM | 256ビット | 認証付き暗号が必須 |
| ファイル完全性検証 | SHA-256 / SHA-3 | - | 衝突耐性が重要 |
| メッセージ認証 | HMAC-SHA256 | 256ビット | 鍵付きハッシュ |
| パスワード保存 | Argon2id | - | メモリハード必須 |
| デジタル署名 | ECDSA (P-256) | 256ビット | RSA-4096 も可 |
| 鍵交換 | ECDH (X25519) | 256ビット | 前方秘匿性確保 |

---

## 2. 対称鍵暗号

### 2.1 対称鍵暗号の基本概念

対称鍵暗号は同じ鍵で暗号化と復号を行う方式である。高速で大量データの暗号化に適するが、鍵の安全な配送が課題となる。

```
対称鍵暗号の仕組み:

  送信者                              受信者
    |                                   |
    |  平文: "Hello, World!"           |
    |     ↓ 暗号化 (共通鍵K)           |
    |  暗号文: "x8f3k2m9..."           |
    |                                   |
    |--- 暗号文を送信 ----------------> |
    |                                   |  暗号文: "x8f3k2m9..."
    |                                   |     ↓ 復号 (共通鍵K)
    |                                   |  平文: "Hello, World!"
    |                                   |
    +-- 共通鍵K は安全な方法で共有済み --+

  課題: 鍵配送問題
  - N人が互いに通信するには N×(N-1)/2 個の鍵が必要
  - 10人 → 45個、100人 → 4,950個の鍵管理が必要
  - → 非対称鍵暗号やKMS で解決する
```

### 2.2 AES-GCM による認証付き暗号

```python
# コード例1: AES-256-GCM による認証付き暗号
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


class AESEncryptor:
    """AES-256-GCM による認証付き暗号化

    WHY AES-GCM を推奨するか:
    - 暗号化と認証を同時に行う（Authenticated Encryption with Associated Data）
    - 改ざん検知機能内蔵（認証タグによる完全性保証）
    - 高速（AES-NI ハードウェアアクセラレーション対応）
    - NIST 推奨、TLS 1.3 のデフォルト暗号スイート

    WHY ノンス（Nonce）が重要か:
    - 同じ鍵で同じノンスを使うと暗号文から平文の情報が漏洩する
    - GCM のノンスは同一鍵で絶対に再利用してはならない
    - 96ビットのランダムノンスなら 2^32 回まで安全に使用可能
    """

    KEY_SIZE = 32   # 256ビット
    NONCE_SIZE = 12  # 96ビット（GCM推奨）

    def __init__(self, key: bytes = None):
        if key is None:
            key = AESGCM.generate_key(bit_length=256)
        if len(key) != self.KEY_SIZE:
            raise ValueError(f"鍵長は{self.KEY_SIZE}バイトである必要があります")
        self._aesgcm = AESGCM(key)
        self._key = key

    def encrypt(self, plaintext: bytes,
                associated_data: bytes = None) -> bytes:
        """暗号化（ノンス + 暗号文 + 認証タグを返す）

        Args:
            plaintext: 暗号化するデータ
            associated_data: 追加認証データ（AAD）
                暗号化はしないが、改ざんを検知したいデータ。
                例: メタデータ、ヘッダ情報

        Returns:
            nonce(12バイト) + ciphertext + tag(16バイト)
        """
        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext  # ノンスを先頭に付加

    def decrypt(self, data: bytes,
                associated_data: bytes = None) -> bytes:
        """復号（ノンスを分離してから復号）

        認証タグの検証に失敗した場合は例外が発生する。
        これにより改ざんされたデータの復号を防止する。
        """
        nonce = data[:self.NONCE_SIZE]
        ciphertext = data[self.NONCE_SIZE:]
        return self._aesgcm.decrypt(nonce, ciphertext, associated_data)


# 使用例
encryptor = AESEncryptor()

# 基本的な暗号化/復号
plaintext = b"This is a secret message"
encrypted = encryptor.encrypt(plaintext)
decrypted = encryptor.decrypt(encrypted)
assert decrypted == plaintext
print(f"暗号化成功: {len(encrypted)} バイト")

# AAD（追加認証データ）付き暗号化
# メタデータは暗号化しないが、改ざんを検知したい場合
aad = b"metadata:user_id=123"
encrypted_with_aad = encryptor.encrypt(plaintext, aad)
decrypted_with_aad = encryptor.decrypt(encrypted_with_aad, aad)
assert decrypted_with_aad == plaintext

# AADが異なると復号に失敗する（改ざん検知）
try:
    encryptor.decrypt(encrypted_with_aad, b"tampered_metadata")
except Exception as e:
    print(f"改ざん検知: {type(e).__name__}")
```

### 2.3 暗号化モードの比較

| モード | 認証 | 並列処理 | パディング | ノンス再利用の影響 | 推奨度 |
|--------|:----:|:-------:|:---------:|:---------------:|:-----:|
| ECB | なし | 可 | 必要 | - | 使用禁止 |
| CBC | なし | 復号のみ | 必要 | IV漏洩 | 条件付き |
| CTR | なし | 可 | 不要 | 平文漏洩 | 条件付き |
| GCM | あり | 可 | 不要 | 認証鍵漏洩 | 推奨 |
| CCM | あり | 不可 | 不要 | 安全性低下 | 可 |
| ChaCha20-Poly1305 | あり | 不可 | 不要 | 認証鍵漏洩 | 推奨 |

```
ECB モードの問題（同一平文ブロック → 同一暗号文ブロック）:

  平文ブロック:  [AAAA][BBBB][AAAA][CCCC]
  ECB 暗号化:   [xxxx][yyyy][xxxx][zzzz]  ← 同一パターンが漏洩!
  CBC/GCM:      [abcd][efgh][ijkl][mnop]  ← パターンが完全に隠蔽

  有名な例: ECB モードで暗号化された画像
  - 暗号化後も画像の輪郭が見える（ECBペンギン問題）
  - CBC/GCM で暗号化するとランダムノイズになる
```

### 2.4 ChaCha20-Poly1305（AES-GCM の代替）

```python
# コード例2: ChaCha20-Poly1305 による暗号化
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import os


class ChaChaEncryptor:
    """ChaCha20-Poly1305 による認証付き暗号化

    WHY ChaCha20 を検討するか:
    - AES-NI 非対応環境（古いCPU、ARM等）で AES-GCM より高速
    - ソフトウェア実装でもサイドチャネル攻撃に耐性がある
    - TLS 1.3 で AES-GCM と並ぶ標準暗号スイート
    - Google が多くのサービスで採用（特にモバイル向け）
    """

    NONCE_SIZE = 12  # 96ビット

    def __init__(self, key: bytes = None):
        if key is None:
            key = ChaCha20Poly1305.generate_key()
        self._cipher = ChaCha20Poly1305(key)

    def encrypt(self, plaintext: bytes,
                associated_data: bytes = None) -> bytes:
        """暗号化"""
        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext = self._cipher.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    def decrypt(self, data: bytes,
                associated_data: bytes = None) -> bytes:
        """復号"""
        nonce = data[:self.NONCE_SIZE]
        ciphertext = data[self.NONCE_SIZE:]
        return self._cipher.decrypt(nonce, ciphertext, associated_data)


# 使用例
chacha = ChaChaEncryptor()
encrypted = chacha.encrypt(b"Secret data for mobile app")
decrypted = chacha.decrypt(encrypted)
print(f"ChaCha20 復号結果: {decrypted}")
```

---

## 3. 非対称鍵暗号

### 3.1 非対称鍵暗号の基本概念

公開鍵と秘密鍵のペアを使用する方式。鍵配送問題を解決するが、処理速度は対称鍵暗号の100〜1000倍遅い。

```
非対称鍵暗号の仕組み:

  ■ 暗号化（機密性）:
  送信者                              受信者
    |                                   |
    |  受信者の公開鍵で暗号化           |
    |  → 受信者の秘密鍵でのみ復号可能  |
    |                                   |
  ■ デジタル署名（認証・完全性）:
  送信者                              受信者
    |                                   |
    |  送信者の秘密鍵で署名             |
    |  → 送信者の公開鍵で検証可能       |
    |  → 送信者の身元と改ざんがないことを確認 |

  ■ RSA vs ECC の違い:
  +----------+------------------+------------------+
  | 項目     | RSA              | ECC (楕円曲線)    |
  +----------+------------------+------------------+
  | 数学基盤 | 大きな素数の積の  | 楕円曲線上の      |
  |          | 素因数分解困難性  | 離散対数問題      |
  +----------+------------------+------------------+
  | 鍵長     | 2048-4096ビット  | 256-384ビット     |
  | 速度     | 遅い             | 速い              |
  | 鍵サイズ | 大きい           | 小さい            |
  +----------+------------------+------------------+
```

### 3.2 RSA と ECDSA の実装

```python
# コード例3: RSA と ECDSA の鍵生成・署名・検証
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization


class AsymmetricCrypto:
    """非対称鍵暗号の操作

    RSA と ECC の使い分け:
    - RSA-4096: 互換性が最も高い、レガシーシステムとの接続
    - ECDSA P-256: 新規システムの標準、TLS 証明書
    - Ed25519: 最新の署名アルゴリズム、最高速度
    """

    @staticmethod
    def generate_rsa_keypair(key_size: int = 4096):
        """RSA 鍵ペアの生成

        公開指数 65537 (0x10001) は標準的な値で、
        暗号化の効率とセキュリティのバランスが良い。
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        return private_key, private_key.public_key()

    @staticmethod
    def generate_ec_keypair(curve=None):
        """ECC 鍵ペアの生成

        P-256 (secp256r1): NIST 推奨、最も広くサポート
        P-384 (secp384r1): より高セキュリティ
        """
        if curve is None:
            curve = ec.SECP256R1()
        private_key = ec.generate_private_key(curve)
        return private_key, private_key.public_key()

    @staticmethod
    def rsa_sign(private_key, message: bytes) -> bytes:
        """RSA-PSS 署名

        WHY PKCS#1 v1.5 ではなく PSS を使うか:
        - PSS（Probabilistic Signature Scheme）はランダム性を含む
        - 同じメッセージに対しても毎回異なる署名が生成される
        - 数学的に証明されたセキュリティを持つ
        """
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
        """RSA-PSS 署名の検証"""
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

    @staticmethod
    def ec_sign(private_key, message: bytes) -> bytes:
        """ECDSA 署名"""
        return private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256()),
        )

    @staticmethod
    def ec_verify(public_key, message: bytes, signature: bytes) -> bool:
        """ECDSA 署名の検証"""
        try:
            public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256()),
            )
            return True
        except Exception:
            return False

    @staticmethod
    def export_public_key_pem(public_key) -> str:
        """公開鍵を PEM 形式でエクスポート"""
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

    @staticmethod
    def export_private_key_pem(private_key, passphrase: bytes = None) -> str:
        """秘密鍵を PEM 形式でエクスポート（暗号化オプション付き）"""
        encryption = (
            serialization.BestAvailableEncryption(passphrase)
            if passphrase
            else serialization.NoEncryption()
        )
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        ).decode()


# 使用例
crypto = AsymmetricCrypto()

# RSA 署名
rsa_priv, rsa_pub = crypto.generate_rsa_keypair()
message = b"Important document"
rsa_sig = crypto.rsa_sign(rsa_priv, message)
print(f"RSA 署名検証: {crypto.rsa_verify(rsa_pub, message, rsa_sig)}")  # True

# ECDSA 署名
ec_priv, ec_pub = crypto.generate_ec_keypair()
ec_sig = crypto.ec_sign(ec_priv, message)
print(f"ECDSA 署名検証: {crypto.ec_verify(ec_pub, message, ec_sig)}")  # True

# 公開鍵のエクスポート
pem = crypto.export_public_key_pem(ec_pub)
print(f"公開鍵 PEM:\n{pem[:100]}...")
```

### 3.3 対称鍵暗号と非対称鍵暗号の比較

| 特性 | 対称鍵暗号 | 非対称鍵暗号 |
|------|-----------|------------|
| 鍵の数 | 1つ（共通鍵） | 2つ（公開鍵+秘密鍵） |
| 速度 | 高速（100-1000倍） | 低速 |
| 鍵配送 | 安全な経路が必要 | 公開鍵は自由に配布可 |
| 用途 | 大量データの暗号化 | 鍵交換、デジタル署名 |
| 代表例 | AES-256-GCM | RSA-4096, ECDSA P-256 |
| 鍵長 | 128/256ビット | 2048/4096ビット（RSA） |
| 量子コンピュータ耐性 | AES-256 は安全 | RSA/ECC は破られる可能性 |

---

## 4. ハッシュ関数

### 4.1 ハッシュ関数の特性

```
ハッシュ関数の特性:

  入力（任意長）    --> ハッシュ関数 --> 出力（固定長）

  "hello"           --> SHA-256     --> 2cf24dba5fb0a30e...
  "hello!"          --> SHA-256     --> ce06092fb948d9ff...
  (1ビットの変化で出力が大きく変化 = 雪崩効果 / Avalanche Effect)

  必須特性:
  1. 一方向性     : ハッシュ値から元データを復元不可
  2. 衝突耐性     : 同じハッシュ値を持つ異なる入力を見つけるのが困難
  3. 第二原像耐性 : 特定の入力と同じハッシュ値を持つ別の入力を見つけるのが困難
  4. 雪崩効果     : 入力の微小な変化で出力が大きく変化する

  ハッシュ関数の選択:
  +-------------+----------+--------+-------------------+
  | アルゴリズム | 出力長   | 状態   | 用途              |
  +-------------+----------+--------+-------------------+
  | MD5         | 128ビット | 廃止   | 使用禁止          |
  | SHA-1       | 160ビット | 廃止   | 使用禁止(衝突発見) |
  | SHA-256     | 256ビット | 現役   | データ完全性検証   |
  | SHA-3       | 可変     | 現役   | SHA-2の代替       |
  | BLAKE2      | 可変     | 現役   | 高速ハッシュ       |
  +-------------+----------+--------+-------------------+
```

### 4.2 安全なハッシュ関数の使い方

```python
# コード例4: ハッシュ関数の安全な使い方
import hashlib
import hmac


class SecureHash:
    """安全なハッシュ操作

    ハッシュ関数は「元に戻せない」一方向関数であり、
    暗号化（元に戻せる）とは根本的に異なる。
    """

    @staticmethod
    def sha256(data: bytes) -> str:
        """SHA-256 ハッシュ（データの完全性検証用）

        用途: ファイルのチェックサム、データの改ざん検知
        用途外: パスワード保存（Argon2id を使うこと）
        """
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def sha3_256(data: bytes) -> str:
        """SHA-3-256 ハッシュ（SHA-2 の代替）

        SHA-3 は SHA-2 とは異なる数学的構造（Keccak）を持つ。
        SHA-2 に脆弱性が発見された場合のフォールバック。
        """
        return hashlib.sha3_256(data).hexdigest()

    @staticmethod
    def file_hash(filepath: str, algorithm: str = "sha256") -> str:
        """大きなファイルのハッシュをストリーミング計算

        WHY ストリーミングが必要か:
        - 数GBのファイルを一度にメモリに読み込むとOOMが発生する
        - 8KB ずつ読み込むことでメモリ使用量を一定に保つ
        """
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """タイミング攻撃を防ぐ比較

        WHY 通常の == ではダメか:
        - a == b は最初の不一致バイトで比較を打ち切る
        - 攻撃者は応答時間の差から正しいバイトを推測できる
        - hmac.compare_digest は常に全バイトを比較する
        """
        return hmac.compare_digest(a.encode(), b.encode())


# 使用例
sh = SecureHash()

# データの完全性検証
data = b"Important document content"
hash_value = sh.sha256(data)
print(f"SHA-256: {hash_value}")

# 改ざん検知
modified_data = b"Important document contenT"  # 末尾1文字変更
modified_hash = sh.sha256(modified_data)
print(f"改ざん検知: {hash_value != modified_hash}")  # True

# 安全な比較
print(f"一致: {sh.constant_time_compare(hash_value, hash_value)}")  # True
```

---

## 5. MAC（Message Authentication Code）

### 5.1 ハッシュと MAC の違い

```
ハッシュ vs MAC の違い:

  ハッシュ (SHA-256):
    - 入力: メッセージのみ
    - 出力: ハッシュ値
    - 用途: 誰でも計算可能 → 完全性のみ保証
    - 問題: 攻撃者もハッシュを再計算できる

  MAC (HMAC-SHA256):
    - 入力: メッセージ + 秘密鍵
    - 出力: MAC 値（認証タグ）
    - 用途: 鍵を知る者のみ計算可能 → 完全性 + 認証
    - 利点: 攻撃者は鍵を知らないので偽造不可能

  具体例:
    API リクエストの改ざん防止
    - ハッシュだけ: 攻撃者がリクエストを変更してハッシュも再計算
    - HMAC: 秘密鍵がないと正しい MAC を計算できない
```

### 5.2 HMAC の実装

```python
# コード例5: HMAC によるメッセージ認証
import hmac
import hashlib
import time


class MessageAuthenticator:
    """HMAC によるメッセージ認証

    HMAC の内部構造:
    HMAC(K, M) = H((K ^ opad) || H((K ^ ipad) || M))
    - 二重ハッシュ構造で Length Extension 攻撃を防止
    - 鍵をそのままハッシュに含めるとLength Extension攻撃に脆弱
    """

    def __init__(self, key: bytes):
        if len(key) < 32:
            raise ValueError("鍵は最低32バイト（256ビット）必要です")
        self.key = key

    def create_mac(self, message: bytes) -> str:
        """メッセージの MAC を計算"""
        return hmac.new(self.key, message, hashlib.sha256).hexdigest()

    def verify_mac(self, message: bytes, mac: str) -> bool:
        """MAC を検証（タイミング攻撃対策済み）"""
        expected = self.create_mac(message)
        return hmac.compare_digest(expected, mac)

    def create_signed_message(self, message: bytes) -> bytes:
        """タイムスタンプ付き署名メッセージを生成

        用途: 一時的な URL、API トークン、CSRF トークン
        """
        timestamp = str(int(time.time())).encode()
        payload = timestamp + b"." + message
        mac = self.create_mac(payload)
        return payload + b"." + mac.encode()

    def verify_signed_message(self, signed: bytes,
                               max_age: int = 300) -> bytes:
        """署名メッセージを検証（有効期限チェック付き）

        Args:
            signed: 署名付きメッセージ
            max_age: 最大有効期間（秒）
        """
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

# 基本的な MAC の計算と検証
message = b"payment:100:USD"
mac_value = auth.create_mac(message)
print(f"MAC: {mac_value}")
print(f"検証: {auth.verify_mac(message, mac_value)}")  # True
print(f"改ざん検知: {auth.verify_mac(b'payment:999:USD', mac_value)}")  # False

# タイムスタンプ付き署名メッセージ
signed = auth.create_signed_message(b"action=approve&id=123")
original = auth.verify_signed_message(signed, max_age=60)
print(f"検証済みメッセージ: {original}")
```

---

## 6. ハイブリッド暗号方式

### 6.1 なぜハイブリッド方式が必要なのか

実際のシステムでは、対称鍵暗号と非対称鍵暗号を組み合わせるハイブリッド方式が標準的に使われる。これは両者の長所を組み合わせた設計である。

```
ハイブリッド暗号のフロー（エンベロープ暗号化）:

  送信者                                    受信者
    |                                         |
    |-- 1. ランダムなAESデータ鍵を生成         |
    |-- 2. データ鍵でメッセージを暗号化(高速)   |
    |-- 3. 受信者の公開鍵でデータ鍵を暗号化     |
    |                                         |
    |-- [暗号化データ鍵] + [暗号化データ] -->  |
    |                                         |
    |                  4. 秘密鍵でデータ鍵を復号 |
    |                  5. データ鍵でデータを復号  |

  WHY ハイブリッドが必要か:
  - 対称鍵暗号は高速だが鍵配送問題がある
  - 非対称鍵暗号は鍵配送を解決するが遅い
  - → 非対称鍵暗号で「鍵」だけ安全に送り、
      実データは対称鍵暗号で高速に暗号化する
```

### 6.2 ハイブリッド暗号の実装

```python
# コード例6: ハイブリッド暗号（エンベロープ暗号化）
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


class HybridEncryption:
    """ハイブリッド暗号方式（RSA + AES-GCM）

    TLS、PGP、S/MIME 等の実際のプロトコルと同じ設計パターン。
    AWS KMS のエンベロープ暗号化もこの原理に基づく。
    """

    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=4096
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, plaintext: bytes) -> dict:
        """ハイブリッド暗号化

        ステップ:
        1. ランダムな AES 鍵（データ鍵）を生成
        2. データ鍵でデータを暗号化（AES-GCM: 高速）
        3. データ鍵を RSA 公開鍵で暗号化（RSA-OAEP: 安全な鍵配送）
        4. 平文のデータ鍵をメモリから消去
        """
        # 1. ランダムな AES 鍵を生成
        data_key = AESGCM.generate_key(bit_length=256)

        # 2. データ鍵でデータを暗号化
        nonce = os.urandom(12)
        aesgcm = AESGCM(data_key)
        encrypted_data = aesgcm.encrypt(nonce, plaintext, None)

        # 3. データ鍵を RSA 公開鍵で暗号化
        encrypted_key = self.public_key.encrypt(
            data_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # 4. 平文データ鍵をゼロクリア（セキュリティのため）
        data_key = b"\x00" * len(data_key)

        return {
            "encrypted_key": encrypted_key,
            "nonce": nonce,
            "encrypted_data": encrypted_data,
        }

    def decrypt(self, package: dict) -> bytes:
        """ハイブリッド復号"""
        # 1. RSA 秘密鍵でデータ鍵を復号
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
        plaintext = aesgcm.decrypt(
            package["nonce"], package["encrypted_data"], None
        )

        # データ鍵をゼロクリア
        data_key = b"\x00" * len(data_key)

        return plaintext


# 使用例
hybrid = HybridEncryption()
plaintext = b"Sensitive financial data: account=123456, balance=1000000"

# 暗号化
package = hybrid.encrypt(plaintext)
print(f"暗号化データ鍵: {len(package['encrypted_key'])} バイト")
print(f"暗号化データ: {len(package['encrypted_data'])} バイト")

# 復号
decrypted = hybrid.decrypt(package)
assert decrypted == plaintext
print(f"復号成功: {decrypted.decode()}")
```

---

## 7. ポスト量子暗号（PQC）

量子コンピュータの発展により、現在の RSA や ECC が将来的に破られる可能性がある。NIST はポスト量子暗号の標準化を進めている。

```
量子コンピュータの脅威:

  現在の暗号             量子コンピュータの影響
  +------------------+  +----------------------------------+
  | RSA              |  | Shor のアルゴリズムで破られる       |
  | ECDSA / ECDH     |  | Shor のアルゴリズムで破られる       |
  | AES-128          |  | Grover で探索空間が半減 → AES-256 推奨 |
  | AES-256          |  | 安全（128ビット相当のセキュリティ）  |
  | SHA-256          |  | 安全（128ビット相当のセキュリティ）  |
  +------------------+  +----------------------------------+

  NIST PQC 標準化（2024年発表）:
  - ML-KEM (CRYSTALS-Kyber): 鍵カプセル化メカニズム
  - ML-DSA (CRYSTALS-Dilithium): デジタル署名
  - SLH-DSA (SPHINCS+): ハッシュベース署名（バックアップ）

  対応方針:
  1. 「Harvest Now, Decrypt Later」攻撃に備える
     → 今の暗号文を保存し、量子コンピュータで将来復号する攻撃
  2. AES-256 を使用し対称鍵暗号の安全性を確保
  3. ハイブリッドモード（従来暗号 + PQC）への移行を計画
```

---

## アンチパターン

### アンチパターン1: 独自暗号アルゴリズムの発明

```python
# NG: 独自の暗号化アルゴリズム（XOR暗号の例）
def my_encrypt(data: bytes, key: bytes) -> bytes:
    """独自暗号: 単純なXOR"""
    return bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))

# 問題点:
# - 鍵の再利用でパターンが漏洩する
# - 暗号学的な安全性が証明されていない
# - 認証機能がない（改ざん検知不可）
# - 既知平文攻撃で鍵が復元される

# OK: 標準アルゴリズムを使用
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)
nonce = os.urandom(12)
ciphertext = aesgcm.encrypt(nonce, data, None)
```

**なぜ危険か**: 暗号の安全性は数十年にわたる学術的な検証によって証明されるものであり、独自実装は未知の脆弱性を含む可能性が極めて高い。「**Don't roll your own crypto**」は暗号分野の鉄則である。

### アンチパターン2: ECB モードの使用

```python
# NG: AES-ECB モード
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
cipher = Cipher(algorithms.AES(key), modes.ECB())
# → 同一平文ブロック = 同一暗号文ブロック → パターン漏洩

# OK: AES-GCM（認証付き暗号モード）
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
aesgcm = AESGCM(key)
ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
# → パターン隠蔽 + 改ざん検知
```

**なぜ危険か**: ECB モードは各ブロックを独立に暗号化するため、同一の平文ブロックが同一の暗号文ブロックに変換され、データのパターンが漏洩する。

### アンチパターン3: ノンス/IV の再利用

```python
# NG: 固定ノンスの使用
nonce = b"\x00" * 12  # 固定ノンス
for message in messages:
    ciphertext = aesgcm.encrypt(nonce, message, None)  # 同じノンスを再利用

# OK: 毎回ランダムなノンスを生成
import os
for message in messages:
    nonce = os.urandom(12)  # 毎回新しいノンス
    ciphertext = aesgcm.encrypt(nonce, message, None)
```

**なぜ危険か**: AES-GCM で同じ鍵・同じノンスで2つのメッセージを暗号化すると、認証鍵が漏洩し、暗号文の偽造が可能になる（Forbidden Attack）。

### アンチパターン4: 暗号化のみで認証なし

```python
# NG: CBC モードで暗号化のみ（認証なし）
cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
encryptor = cipher.encryptor()
ciphertext = encryptor.update(plaintext) + encryptor.finalize()
# → 暗号文の改ざんを検知できない（Padding Oracle Attack のリスク）

# OK: GCM（認証付き暗号）
aesgcm = AESGCM(key)
ciphertext = aesgcm.encrypt(nonce, plaintext, None)
# → 暗号文が改ざんされると復号時に例外が発生
```

---

## FAQ

### Q1: AES-128 と AES-256 のどちらを使うべきですか?

一般的には AES-256 が推奨される。AES-128 も現時点で十分な安全性を持ち、破られる可能性は実質的にない。しかし、量子コンピュータ時代の到来を考慮すると、AES-256 の方がより長期的な安全マージンがある（Grover のアルゴリズムで探索空間が半減しても 128 ビットの安全性を維持）。パフォーマンス差はAES-NI対応CPUではほぼ無視できるため、特別な理由がなければ AES-256 を選択する。

### Q2: ハッシュ関数は暗号化の代わりに使えますか?

使えない。ハッシュ関数は一方向関数であり、ハッシュ値から元データを復元できない。データの完全性検証やパスワード保存には適するが、復号が必要なユースケースでは暗号化を使用する。逆に、暗号化はハッシュの代わりにはならない（パスワードは暗号化ではなくハッシュで保存する）。

### Q3: RSA の鍵長はどれくらい必要ですか?

2048 ビット以上が最低要件、4096 ビットが推奨。ただし、ECDSA の P-256 曲線は RSA-3072 と同等のセキュリティレベルを持ち、鍵サイズが小さく処理も高速なため、新規システムでは ECDSA の採用を検討すべきである。量子コンピュータ耐性を考慮すると、いずれ PQC への移行が必要になる。

### Q4: なぜ「暗号化」と「ハッシュ化」を使い分ける必要がありますか?

目的が異なるためである。暗号化は「元に戻す（復号する）」ことを前提とした可逆処理であり、通信の秘匿やデータの保護に使う。ハッシュ化は「元に戻さない」ことを前提とした不可逆処理であり、パスワード保存やデータの完全性検証に使う。パスワードを暗号化で保存すると、暗号鍵が漏洩した時点で全パスワードが復元されるため、ハッシュ化が正しい選択である。

### Q5: GCM モードでノンスが重複するとどうなりますか?

同じ鍵で同じノンスを使って2つの異なるメッセージを暗号化すると、以下の問題が発生する:
1. 2つの暗号文の XOR から平文の XOR が求まる
2. 認証鍵（GHASH鍵）が漏洩し、認証タグの偽造が可能になる
これを「Forbidden Attack」と呼ぶ。96ビットのランダムノンスなら、同一鍵で約 2^32 回（約43億回）の暗号化まで安全に使用できる（誕生日問題の閾値）。それ以上使う場合は鍵をローテーションする。

---

## 実践演習

### 演習1（基礎）: ファイル暗号化ツールの実装

AES-256-GCM を使用して、ファイルの暗号化・復号を行うツールを実装せよ。

**要件**:
- パスワードから暗号鍵を導出（PBKDF2 または Argon2 使用）
- ソルトをランダム生成し暗号文に含める
- 暗号化ファイルの形式: salt(16B) + nonce(12B) + ciphertext + tag(16B)

<details>
<summary>模範解答</summary>

```python
import os
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


class FileEncryptor:
    """パスワードベースのファイル暗号化ツール"""

    SALT_SIZE = 16
    NONCE_SIZE = 12
    KEY_SIZE = 32  # AES-256
    ITERATIONS = 600000  # OWASP 推奨値（2024年）

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """パスワードから暗号鍵を導出"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_SIZE,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return kdf.derive(password.encode())

    def encrypt_file(self, input_path: str, output_path: str,
                     password: str) -> None:
        """ファイルを暗号化"""
        # ソルトとノンスを生成
        salt = os.urandom(self.SALT_SIZE)
        nonce = os.urandom(self.NONCE_SIZE)

        # パスワードから鍵を導出
        key = self._derive_key(password, salt)
        aesgcm = AESGCM(key)

        # ファイルを読み込んで暗号化
        with open(input_path, "rb") as f:
            plaintext = f.read()

        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # salt + nonce + ciphertext を書き込み
        with open(output_path, "wb") as f:
            f.write(salt + nonce + ciphertext)

        print(f"暗号化完了: {output_path}")

    def decrypt_file(self, input_path: str, output_path: str,
                     password: str) -> None:
        """ファイルを復号"""
        with open(input_path, "rb") as f:
            data = f.read()

        # salt と nonce を分離
        salt = data[:self.SALT_SIZE]
        nonce = data[self.SALT_SIZE:self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = data[self.SALT_SIZE + self.NONCE_SIZE:]

        # パスワードから鍵を導出
        key = self._derive_key(password, salt)
        aesgcm = AESGCM(key)

        # 復号
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        with open(output_path, "wb") as f:
            f.write(plaintext)

        print(f"復号完了: {output_path}")


# 使用例
encryptor = FileEncryptor()
# encryptor.encrypt_file("secret.txt", "secret.enc", "MyPassword123!")
# encryptor.decrypt_file("secret.enc", "secret_decrypted.txt", "MyPassword123!")
```

</details>

### 演習2（応用）: デジタル署名による文書検証システム

ECDSA を使用して、文書の署名と検証を行うシステムを実装せよ。

**要件**:
- 鍵ペアの生成と PEM 形式での保存/読み込み
- 文書（バイト列）への署名
- 署名の検証
- 署名付き文書パッケージの作成（文書 + 署名 + 公開鍵のバンドル）

<details>
<summary>模範解答</summary>

```python
import json
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization


class DocumentSigner:
    """ECDSA によるデジタル署名システム"""

    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    def sign(self, document: bytes) -> bytes:
        """文書に署名"""
        return self.private_key.sign(document, ec.ECDSA(hashes.SHA256()))

    @staticmethod
    def verify(public_key, document: bytes, signature: bytes) -> bool:
        """署名を検証"""
        try:
            public_key.verify(signature, document, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False

    def create_signed_package(self, document: bytes) -> str:
        """署名付き文書パッケージを JSON 形式で作成"""
        signature = self.sign(document)
        pub_pem = self.public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return json.dumps({
            "document": base64.b64encode(document).decode(),
            "signature": base64.b64encode(signature).decode(),
            "public_key": pub_pem.decode(),
            "algorithm": "ECDSA-P256-SHA256",
        })

    @staticmethod
    def verify_package(package_json: str) -> dict:
        """署名付きパッケージを検証"""
        package = json.loads(package_json)
        document = base64.b64decode(package["document"])
        signature = base64.b64decode(package["signature"])
        public_key = serialization.load_pem_public_key(
            package["public_key"].encode()
        )

        is_valid = DocumentSigner.verify(public_key, document, signature)
        return {
            "valid": is_valid,
            "document": document if is_valid else None,
        }


# テスト
signer = DocumentSigner()
doc = b"Contract: Party A agrees to pay Party B $1000"

# 署名付きパッケージの作成と検証
package = signer.create_signed_package(doc)
result = DocumentSigner.verify_package(package)
print(f"署名検証: {result['valid']}")  # True
print(f"文書: {result['document'].decode()}")
```

</details>

### 演習3（発展）: 鍵交換プロトコルの実装

ECDH（楕円曲線ディフィー・ヘルマン）鍵交換を実装し、2者間で安全に共通鍵を確立するプロトコルを設計せよ。

**要件**:
- ECDH による鍵交換
- 派生した共有秘密から AES 鍵を導出（HKDF 使用）
- 導出した鍵で AES-GCM 暗号化通信
- 前方秘匿性（Perfect Forward Secrecy）の実現

<details>
<summary>模範解答</summary>

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


class SecureChannel:
    """ECDH 鍵交換 + AES-GCM による安全な通信チャネル

    前方秘匿性: 各セッションで新しい一時鍵ペアを生成するため、
    長期秘密鍵が漏洩しても過去の通信は解読されない。
    """

    def __init__(self):
        # 一時鍵ペア（エフェメラル鍵）を生成
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self._private_key.public_key()
        self._shared_key = None

    def derive_shared_key(self, peer_public_key) -> None:
        """相手の公開鍵から共有秘密を導出"""
        # ECDH で共有秘密を計算
        shared_secret = self._private_key.exchange(
            ec.ECDH(), peer_public_key
        )
        # HKDF で AES 鍵を導出
        self._shared_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"secure-channel-v1",
        ).derive(shared_secret)

    def encrypt(self, plaintext: bytes) -> bytes:
        """共有鍵で暗号化"""
        if not self._shared_key:
            raise RuntimeError("鍵交換が完了していません")
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._shared_key)
        return nonce + aesgcm.encrypt(nonce, plaintext, None)

    def decrypt(self, data: bytes) -> bytes:
        """共有鍵で復号"""
        if not self._shared_key:
            raise RuntimeError("鍵交換が完了していません")
        nonce = data[:12]
        ciphertext = data[12:]
        aesgcm = AESGCM(self._shared_key)
        return aesgcm.decrypt(nonce, ciphertext, None)


# 使用例: Alice と Bob の安全な通信
alice = SecureChannel()
bob = SecureChannel()

# 公開鍵の交換（安全でないチャネルでOK）
alice.derive_shared_key(bob.public_key)
bob.derive_shared_key(alice.public_key)

# Alice → Bob
message = b"Hello Bob, this is a secure message!"
encrypted = alice.encrypt(message)
decrypted = bob.decrypt(encrypted)
print(f"Bob received: {decrypted.decode()}")

# Bob → Alice
reply = b"Hi Alice, received your message!"
encrypted_reply = bob.encrypt(reply)
decrypted_reply = alice.decrypt(encrypted_reply)
print(f"Alice received: {decrypted_reply.decode()}")
```

</details>

---

## まとめ

| 技術 | 用途 | 推奨アルゴリズム | 鍵長 |
|------|------|----------------|------|
| 対称鍵暗号 | データの暗号化 | AES-256-GCM / ChaCha20-Poly1305 | 256ビット |
| 非対称鍵暗号 | 鍵交換、デジタル署名 | ECDSA P-256, RSA-4096 | 256ビット / 4096ビット |
| ハッシュ関数 | 完全性検証 | SHA-256, SHA-3 | - |
| MAC | メッセージ認証 | HMAC-SHA256 | 256ビット |
| パスワードハッシュ | パスワード保存 | Argon2id, bcrypt | - |
| 鍵交換 | 共通鍵の確立 | ECDH (X25519, P-256) | 256ビット |
| ハイブリッド暗号 | 大量データの安全な暗号化 | RSA-OAEP + AES-GCM | - |

---

## 次に読むべきガイド

- [TLS/証明書](./01-tls-certificates.md) -- TLS ハンドシェイクと証明書管理
- [鍵管理](./02-key-management.md) -- 鍵のライフサイクルと管理手法
- [認証脆弱性](../01-web-security/04-auth-vulnerabilities.md) -- パスワードハッシュの実践
- [ネットワークセキュリティ基礎](../03-network-security/00-network-security-basics.md) -- 暗号化通信の上位レイヤー
- [パスワードセキュリティ](../../authentication-and-authorization/docs/00-fundamentals/01-password-security.md) -- パスワードハッシュの詳細

---

## 参考文献

1. **NIST SP 800-175B: Guideline for Using Cryptographic Standards** -- https://csrc.nist.gov/publications/detail/sp/800-175b/rev-1/final
2. **Christof Paar & Jan Pelzl, "Understanding Cryptography"** -- Springer（暗号学の教科書として最高水準）
3. **OWASP Cryptographic Storage Cheat Sheet** -- https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
4. **NIST Post-Quantum Cryptography Standards** -- https://csrc.nist.gov/projects/post-quantum-cryptography
5. **Dan Boneh & Victor Shoup, "A Graduate Course in Applied Cryptography"** -- https://toc.cryptobook.us/ （無料で読める暗号学の決定版テキスト）
6. **RFC 5116: An Interface and Algorithms for Authenticated Encryption** -- https://datatracker.ietf.org/doc/html/rfc5116
