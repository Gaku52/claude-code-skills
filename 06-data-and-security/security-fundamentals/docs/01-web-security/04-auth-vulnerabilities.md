# 認証脆弱性

> パスワード管理、セッション管理、ブルートフォース対策を中心に、安全な認証システムの設計と実装方法を解説する。

## この章で学ぶこと

1. **安全なパスワード管理**（ハッシュ化、ポリシー、MFA）の実装方法を理解する
2. **セッション管理**の脆弱性パターンと堅牢な実装手法を習得する
3. **ブルートフォース対策**とアカウントロックアウトの適切な設計を身につける

---

## 1. パスワード管理

### 1.1 安全なパスワードハッシュ

```
パスワードハッシュの進化:

  NG: 平文保存      "password123"
  NG: MD5           "482c811da5d5b4bc..."  (高速すぎる、レインボーテーブル)
  NG: SHA-256       "ef92b778bafe..."      (高速すぎる)
  NG: SHA-256+salt  "a1b2c3..." + salt     (高速すぎる)
  OK: bcrypt        "$2b$12$LJ3..."        (意図的に低速、ソルト内蔵)
  OK: Argon2id      "$argon2id$v=19$..."   (メモリハード、推奨)
  OK: scrypt        (メモリハード)
```

```python
# コード例1: 安全なパスワードハッシュ実装
import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

class PasswordManager:
    """安全なパスワード管理"""

    def __init__(self):
        # Argon2id（推奨）の設定
        self.ph = PasswordHasher(
            time_cost=3,        # 反復回数
            memory_cost=65536,  # メモリ使用量（KB）
            parallelism=4,      # 並列度
        )

    def hash_password(self, password: str) -> str:
        """パスワードをArgon2idでハッシュ化"""
        return self.ph.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワードを検証"""
        try:
            return self.ph.verify(hashed, password)
        except VerifyMismatchError:
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """パラメータが古い場合に再ハッシュが必要か判定"""
        return self.ph.check_needs_rehash(hashed)

    def validate_policy(self, password: str) -> list:
        """パスワードポリシーの検証"""
        errors = []
        if len(password) < 12:
            errors.append("12文字以上必要です")
        if len(password) > 128:
            errors.append("128文字以下にしてください")
        if not any(c.isupper() for c in password):
            errors.append("大文字を1文字以上含めてください")
        if not any(c.islower() for c in password):
            errors.append("小文字を1文字以上含めてください")
        if not any(c.isdigit() for c in password):
            errors.append("数字を1文字以上含めてください")
        # 漏洩パスワードチェック（Have I Been Pwned API）
        if self._is_breached(password):
            errors.append("このパスワードは過去のデータ漏洩に含まれています")
        return errors

    def _is_breached(self, password: str) -> bool:
        """HIBPのk-匿名性APIでパスワード漏洩チェック"""
        import hashlib, requests
        sha1 = hashlib.sha1(password.encode()).hexdigest().upper()
        prefix, suffix = sha1[:5], sha1[5:]
        resp = requests.get(f"https://api.pwnedpasswords.com/range/{prefix}")
        return suffix in resp.text

# 使用例
pm = PasswordManager()
hashed = pm.hash_password("MyS3cur3P@ssw0rd!")
print(pm.verify_password("MyS3cur3P@ssw0rd!", hashed))  # True
print(pm.verify_password("wrong", hashed))               # False
```

---

## 2. セッション管理

```
セッション管理のフロー:

  クライアント                            サーバー
    |                                       |
    |-- POST /login (credentials) -------> |
    |                                       |-- 認証成功
    |                                       |-- セッションID生成
    |                                       |   (暗号学的乱数)
    |<-- Set-Cookie: sid=<random> --------- |
    |    HttpOnly; Secure; SameSite=Lax     |
    |                                       |
    |-- GET /dashboard ------------------>  |
    |   Cookie: sid=<random>                |-- セッション検証
    |                                       |-- ユーザー情報取得
    |<-- 200 OK (ダッシュボード) ----------- |
    |                                       |
    |-- POST /logout -------------------->  |
    |                                       |-- セッション無効化
    |<-- Set-Cookie: sid=; Max-Age=0 -----  |
```

```python
# コード例2: セキュアなセッション管理
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import hashlib

class SecureSessionManager:
    """安全なセッション管理"""

    SESSION_ID_LENGTH = 32  # 256ビット
    SESSION_TIMEOUT = 3600  # 1時間
    IDLE_TIMEOUT = 1800     # 30分

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self, user_id: str, ip: str,
                       user_agent: str) -> str:
        """認証成功後にセッションを作成"""
        session_id = secrets.token_hex(self.SESSION_ID_LENGTH)
        now = time.time()
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": now,
            "last_activity": now,
            "ip": ip,
            "user_agent_hash": hashlib.sha256(
                user_agent.encode()
            ).hexdigest(),
        }
        return session_id

    def validate_session(self, session_id: str, ip: str,
                         user_agent: str) -> Optional[str]:
        """セッションの検証"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        now = time.time()

        # 絶対タイムアウトチェック
        if now - session["created_at"] > self.SESSION_TIMEOUT:
            self.destroy_session(session_id)
            return None

        # アイドルタイムアウトチェック
        if now - session["last_activity"] > self.IDLE_TIMEOUT:
            self.destroy_session(session_id)
            return None

        # セッション固定攻撃対策: IPとUser-Agentの変更を検出
        ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()
        if session["user_agent_hash"] != ua_hash:
            self.destroy_session(session_id)
            return None

        # 最終アクティビティ更新
        session["last_activity"] = now
        return session["user_id"]

    def regenerate_session(self, old_session_id: str) -> Optional[str]:
        """セッションIDの再生成（権限昇格時に必須）"""
        session = self.sessions.get(old_session_id)
        if not session:
            return None
        # 旧セッションを削除
        del self.sessions[old_session_id]
        # 新セッションIDで同じデータを登録
        new_session_id = secrets.token_hex(self.SESSION_ID_LENGTH)
        self.sessions[new_session_id] = session
        return new_session_id

    def destroy_session(self, session_id: str) -> None:
        """セッションの破棄"""
        self.sessions.pop(session_id, None)

    def destroy_all_user_sessions(self, user_id: str) -> int:
        """特定ユーザーの全セッションを破棄（パスワード変更時等）"""
        to_delete = [
            sid for sid, data in self.sessions.items()
            if data["user_id"] == user_id
        ]
        for sid in to_delete:
            del self.sessions[sid]
        return len(to_delete)
```

---

## 3. ブルートフォース対策

```
ブルートフォース対策の層:

  +--------------------------------------------------+
  |  Layer 1: レートリミット                           |
  |  - IP単位: 10回/分                                |
  |  - アカウント単位: 5回/5分                         |
  +--------------------------------------------------+
  |  Layer 2: プログレッシブ遅延                       |
  |  - 1回目失敗: 即応答                              |
  |  - 2回目失敗: 1秒待機                             |
  |  - 3回目失敗: 2秒待機                             |
  |  - 5回目失敗: 15秒待機                            |
  +--------------------------------------------------+
  |  Layer 3: アカウントロックアウト                    |
  |  - 10回失敗: 30分ロック                           |
  |  - 一定期間後に自動解除                            |
  +--------------------------------------------------+
  |  Layer 4: CAPTCHA                                 |
  |  - 3回失敗後にCAPTCHA表示                         |
  +--------------------------------------------------+
```

```python
# コード例3: ブルートフォース対策の実装
import time
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class LoginAttempt:
    count: int = 0
    first_attempt: float = 0
    last_attempt: float = 0
    locked_until: float = 0

class BruteForceProtection:
    """ブルートフォース攻撃対策"""

    MAX_ATTEMPTS = 5
    LOCKOUT_DURATION = 1800  # 30分
    WINDOW = 300             # 5分間のウィンドウ
    PROGRESSIVE_DELAYS = [0, 1, 2, 4, 8, 15]  # 秒

    def __init__(self):
        self.attempts: Dict[str, LoginAttempt] = defaultdict(LoginAttempt)

    def check_and_record(self, identifier: str) -> dict:
        """ログイン試行を確認・記録する"""
        attempt = self.attempts[identifier]
        now = time.time()

        # ロックアウト中かチェック
        if attempt.locked_until > now:
            remaining = int(attempt.locked_until - now)
            return {
                "allowed": False,
                "reason": "account_locked",
                "retry_after": remaining,
            }

        # ウィンドウのリセット
        if now - attempt.first_attempt > self.WINDOW:
            attempt.count = 0
            attempt.first_attempt = now

        # 試行回数の記録
        if attempt.count == 0:
            attempt.first_attempt = now
        attempt.count += 1
        attempt.last_attempt = now

        # ロックアウト判定
        if attempt.count >= self.MAX_ATTEMPTS:
            attempt.locked_until = now + self.LOCKOUT_DURATION
            return {
                "allowed": False,
                "reason": "too_many_attempts",
                "retry_after": self.LOCKOUT_DURATION,
            }

        # プログレッシブ遅延
        delay_idx = min(attempt.count, len(self.PROGRESSIVE_DELAYS) - 1)
        delay = self.PROGRESSIVE_DELAYS[delay_idx]

        return {
            "allowed": True,
            "delay": delay,
            "attempts_remaining": self.MAX_ATTEMPTS - attempt.count,
            "require_captcha": attempt.count >= 3,
        }

    def reset(self, identifier: str) -> None:
        """ログイン成功時にカウンターをリセット"""
        self.attempts.pop(identifier, None)
```

---

## 4. 多要素認証（MFA）

```python
# コード例4: TOTP（Time-based One-Time Password）の実装
import hmac
import hashlib
import struct
import time
import base64
import secrets

class TOTP:
    """RFC 6238準拠のTOTP実装"""

    def __init__(self, secret: bytes, digits: int = 6,
                 period: int = 30, algorithm=hashlib.sha1):
        self.secret = secret
        self.digits = digits
        self.period = period
        self.algorithm = algorithm

    @classmethod
    def generate_secret(cls) -> str:
        """新しいTOTPシークレットを生成"""
        return base64.b32encode(secrets.token_bytes(20)).decode()

    def generate_code(self, timestamp: float = None) -> str:
        """現在のTOTPコードを生成"""
        if timestamp is None:
            timestamp = time.time()
        counter = int(timestamp) // self.period
        counter_bytes = struct.pack(">Q", counter)

        mac = hmac.new(self.secret, counter_bytes, self.algorithm).digest()
        offset = mac[-1] & 0x0F
        truncated = struct.unpack(">I", mac[offset:offset + 4])[0]
        truncated &= 0x7FFFFFFF
        code = truncated % (10 ** self.digits)
        return str(code).zfill(self.digits)

    def verify(self, code: str, window: int = 1) -> bool:
        """TOTPコードを検証（前後windowステップ分を許容）"""
        now = time.time()
        for offset in range(-window, window + 1):
            check_time = now + (offset * self.period)
            if hmac.compare_digest(
                self.generate_code(check_time), code
            ):
                return True
        return False
```

### 認証方式の比較

| 方式 | セキュリティ | ユーザビリティ | コスト | フィッシング耐性 |
|------|:--------:|:--------:|:-----:|:--------:|
| パスワードのみ | 低 | 高 | 低 | なし |
| パスワード + SMS OTP | 中 | 中 | 中 | 低 |
| パスワード + TOTP | 高 | 中 | 低 | 低 |
| パスワード + FIDO2/WebAuthn | 最高 | 高 | 中 | 高 |
| パスキー（Passkeys） | 最高 | 最高 | 低 | 最高 |

---

## 5. JWTの安全な使用

```python
# コード例5: JWTの安全な実装
import jwt
import time
from typing import Optional

class SecureJWT:
    """安全なJWTトークン管理"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        # 安全でないアルゴリズムを拒否
        if algorithm in ("none", "None"):
            raise ValueError("Algorithm 'none' is not allowed")
        self.algorithm = algorithm

    def create_token(self, user_id: str, roles: list,
                     expires_in: int = 900) -> str:
        """アクセストークンを生成（デフォルト15分）"""
        now = int(time.time())
        payload = {
            "sub": user_id,
            "roles": roles,
            "iat": now,
            "exp": now + expires_in,
            "nbf": now,        # Not Before
            "jti": secrets.token_hex(16),  # JWT ID（リプレイ防止）
            "iss": "myapp",    # 発行者
        }
        return jwt.encode(payload, self.secret_key,
                          algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """トークンを検証"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # リスト形式で指定
                options={
                    "require": ["exp", "iat", "sub", "iss"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                },
                issuer="myapp",
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# 使用例
jwt_manager = SecureJWT("my-secret-key-at-least-256-bits-long!!")
token = jwt_manager.create_token("user123", ["viewer"])
payload = jwt_manager.verify_token(token)
```

---

## アンチパターン

### アンチパターン1: パスワードの平文保存

パスワードを暗号化（復号可能な方式）やハッシュ化なしで保存するパターン。データベースが漏洩した時点で全ユーザーのパスワードが露出する。必ずArgon2idやbcrypt等の適切なアルゴリズムでハッシュ化する。

### アンチパターン2: JWTの `algorithm: "none"` 攻撃

JWTライブラリが `none` アルゴリズムを受け入れてしまうパターン。攻撃者が署名なしのJWTを生成し、任意のペイロードで認証をバイパスできる。`algorithms` パラメータを明示的に指定し、`none` を含めないことが重要である。

---

## FAQ

### Q1: パスワードの最大長は制限すべきですか?

128文字程度の上限は設定すべきである。無制限にすると、非常に長いパスワードによるDoS攻撃（ハッシュ計算に時間がかかる）のリスクがある。ただし、8文字等の短すぎる最大長は避ける。

### Q2: セッションIDはどこに保存すべきですか?

HttpOnly+Secure+SameSite属性付きのCookieが推奨される。localStorageはXSSに脆弱であり、URLパラメータはRefererヘッダーやブラウザ履歴から漏洩する危険がある。

### Q3: JWTとセッションベース認証のどちらを使うべきですか?

モノリスアプリケーションではセッションベースが推奨。マイクロサービスやSPAではJWTが適している。ただし、JWTには即座にトークンを無効化できないという課題があるため、短い有効期限とリフレッシュトークンの組み合わせが必要である。

---

## まとめ

| 項目 | 推奨対策 | 重要度 |
|------|---------|--------|
| パスワードハッシュ | Argon2id / bcrypt | 必須 |
| セッションID | 256ビット以上の暗号学的乱数 | 必須 |
| セッション管理 | HttpOnly/Secure Cookie + タイムアウト | 必須 |
| ブルートフォース対策 | レートリミット + プログレッシブ遅延 + ロックアウト | 必須 |
| MFA | FIDO2/WebAuthn またはTOTP | 強く推奨 |
| JWT | 短い有効期限 + アルゴリズム明示指定 | 条件付き推奨 |

---

## 次に読むべきガイド

- [../02-cryptography/00-crypto-basics.md](../02-cryptography/00-crypto-basics.md) -- 暗号化の基礎知識
- [../03-network-security/02-api-security.md](../03-network-security/02-api-security.md) -- API認証（OAuth2/JWT）の詳細
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般

---

## 参考文献

1. OWASP Authentication Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
2. OWASP Session Management Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
3. NIST SP 800-63B: Digital Identity Guidelines -- https://pages.nist.gov/800-63-3/sp800-63b.html
4. RFC 6238: TOTP -- https://datatracker.ietf.org/doc/html/rfc6238
