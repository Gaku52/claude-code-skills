# 認証脆弱性

> パスワード管理、セッション管理、ブルートフォース対策、多要素認証、JWTセキュリティを中心に、安全な認証システムの設計と実装方法を包括的に解説する。認証は「あなたは誰か」を確認するプロセスであり、その脆弱性は直接的な不正アクセスにつながるため、セキュリティにおいて最も重要な領域の一つである。

## この章で学ぶこと

1. **安全なパスワード管理**（ハッシュ化、ポリシー、漏洩チェック、MFA）の実装方法を理解する
2. **セッション管理**の脆弱性パターンと堅牢な実装手法を習得する
3. **ブルートフォース対策**とアカウントロックアウトの適切な設計を身につける
4. **JWT の安全な使用**と攻撃パターンへの防御手法を理解する
5. **多要素認証（MFA）**の各方式の特性と実装上の注意点を把握する

## 前提知識

- [セキュリティ概要](../00-basics/00-security-overview.md) -- CIA三要素と基本的なセキュリティ概念
- [脅威モデリング](../00-basics/01-threat-modeling.md) -- 攻撃者の動機とリスク評価手法
- [認証と認可の違い](../../authentication-and-authorization/docs/00-fundamentals/00-authentication-vs-authorization.md) -- 認証(AuthN)と認可(AuthZ)の基本
- [暗号基礎](../02-cryptography/00-crypto-basics.md) -- ハッシュ関数と暗号化の基礎
- HTTP の基本（Cookie、ヘッダ、ステータスコード）を理解していること

---

## 1. パスワード管理

### 1.1 なぜパスワードハッシュが重要なのか

パスワードを適切にハッシュ化する理由は「**データベースが漏洩した場合の被害を最小化する**」ことである。企業のデータベースが流出する事件は毎年発生しており、平文や不適切なハッシュで保存されたパスワードは即座に悪用される。

適切なハッシュアルゴリズムを使用すれば、たとえデータベース全体が漏洩しても、攻撃者がパスワードを復元するには莫大な計算コストが必要となり、実質的に不可能にできる。

### 1.2 パスワードハッシュの進化

```
パスワード保存方式の進化と問題点:

  NG: 平文保存      "password123"                → 即座に悪用可能
  NG: MD5           "482c811da5d5b4bc..."         → 高速すぎる、レインボーテーブル攻撃
  NG: SHA-256       "ef92b778bafe..."             → 高速すぎる、GPU で毎秒数十億回計算可能
  NG: SHA-256+salt  "a1b2c3..." + salt            → ソルトがあっても高速すぎる
  OK: bcrypt        "$2b$12$LJ3..."               → 意図的に低速、ソルト内蔵、コスト調整可能
  OK: scrypt        (メモリハード)                  → GPU/ASIC 耐性あり
  OK: Argon2id      "$argon2id$v=19$..."           → メモリハード + CPU ハード、現在の最推奨

  ポイント:
  - パスワードハッシュは「遅い」ことが正義
  - 汎用ハッシュ(MD5/SHA)は速度が求められる用途向け
  - パスワード専用ハッシュ(bcrypt/Argon2id)は意図的に低速
```

### 1.3 安全なパスワードハッシュ実装

```python
# コード例1: Argon2id によるパスワードハッシュ管理
import hashlib
import requests
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


class PasswordManager:
    """安全なパスワード管理クラス

    Argon2id を使用し、パスワードのハッシュ化・検証・ポリシーチェックを提供する。
    Argon2id は Argon2i（サイドチャネル耐性）と Argon2d（GPU耐性）を
    組み合わせたハイブリッドモードであり、OWASP が第一推奨とするアルゴリズムである。
    """

    def __init__(self):
        # OWASP 推奨パラメータ（2024年時点）
        # - time_cost: 反復回数（大きいほど低速=安全）
        # - memory_cost: メモリ使用量 KB（大きいほど GPU/ASIC 耐性向上）
        # - parallelism: 並列度（CPU コア数に合わせる）
        self.ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,  # 64 MB
            parallelism=4,
        )

    def hash_password(self, password: str) -> str:
        """パスワードを Argon2id でハッシュ化

        ソルトは自動生成・ハッシュ文字列に埋め込まれる。
        """
        return self.ph.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワードを検証（タイミング攻撃耐性あり）"""
        try:
            return self.ph.verify(hashed, password)
        except VerifyMismatchError:
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """パラメータが古い場合に再ハッシュが必要か判定

        アルゴリズムのパラメータを強化した場合、
        ログイン成功時に自動的に再ハッシュを行うための判定。
        """
        return self.ph.check_needs_rehash(hashed)

    def validate_policy(self, password: str) -> list:
        """NIST SP 800-63B 準拠のパスワードポリシー検証

        NISTガイドラインのポイント:
        - 最小8文字（推奨12文字以上）
        - 最大64文字以上を許容
        - 文字種の強制は不要（ユーザビリティ低下のため）
        - 漏洩パスワードリストとの照合は必須
        """
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
        """HIBPのk-匿名性APIでパスワード漏洩チェック

        パスワードの SHA-1 ハッシュの先頭5文字のみを送信するため、
        パスワード自体がネットワーク上に流れることはない。
        """
        sha1 = hashlib.sha1(password.encode()).hexdigest().upper()
        prefix, suffix = sha1[:5], sha1[5:]
        try:
            resp = requests.get(
                f"https://api.pwnedpasswords.com/range/{prefix}",
                timeout=5,
            )
            return suffix in resp.text
        except requests.RequestException:
            # API障害時は安全側に倒す（ブロックしない）
            return False


# 使用例
pm = PasswordManager()

# ポリシーチェック
errors = pm.validate_policy("MyS3cur3P@ssw0rd!")
if errors:
    print(f"ポリシー違反: {errors}")
else:
    # ハッシュ化と保存
    hashed = pm.hash_password("MyS3cur3P@ssw0rd!")
    print(f"ハッシュ値: {hashed}")
    # 例: $argon2id$v=19$m=65536,t=3,p=4$salt$hash

    # 検証
    print(pm.verify_password("MyS3cur3P@ssw0rd!", hashed))  # True
    print(pm.verify_password("wrong", hashed))               # False

    # ログイン時の再ハッシュチェック
    if pm.needs_rehash(hashed):
        new_hash = pm.hash_password("MyS3cur3P@ssw0rd!")
        print("パラメータが更新されたため再ハッシュしました")
```

### 1.4 bcrypt との比較実装

```python
# コード例2: bcrypt によるパスワードハッシュ（レガシーシステム向け）
import bcrypt


class BcryptPasswordManager:
    """bcrypt によるパスワード管理

    Argon2id が使えない環境（古い Python/ライブラリ制約）向け。
    bcrypt は 1999 年から使用されている実績のあるアルゴリズム。

    WHY bcrypt の cost factor が重要か:
    - cost=10: 約 100ms/hash（2015 年の標準）
    - cost=12: 約 400ms/hash（2024 年の推奨）
    - cost=14: 約 1.6s/hash（高セキュリティ環境）
    ハードウェアの進化に合わせて cost を上げることで安全性を維持する。
    """

    DEFAULT_ROUNDS = 12  # 2024年推奨値

    def hash_password(self, password: str) -> str:
        """bcrypt ハッシュ化（ソルト自動生成）"""
        # bcrypt は内部で 72 バイトに切り詰めるため注意
        if len(password.encode("utf-8")) > 72:
            # 長いパスワードは事前に SHA-256 でハッシュ
            import hashlib
            password = hashlib.sha256(password.encode()).hexdigest()

        salt = bcrypt.gensalt(rounds=self.DEFAULT_ROUNDS)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワードを検証"""
        if len(password.encode("utf-8")) > 72:
            import hashlib
            password = hashlib.sha256(password.encode()).hexdigest()

        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed.encode("utf-8"),
        )


# 使用例
bpm = BcryptPasswordManager()
hashed = bpm.hash_password("SecurePass123!")
print(bpm.verify_password("SecurePass123!", hashed))  # True
```

### 1.5 パスワードハッシュアルゴリズム比較

| アルゴリズム | メモリハード | GPU耐性 | 推奨度 | 計算コスト調整 | 備考 |
|-------------|:---------:|:------:|:-----:|:----------:|------|
| MD5 | - | - | 使用禁止 | 不可 | 高速すぎる、衝突発見済み |
| SHA-256 | - | - | 不適 | 不可 | パスワード用途には不向き |
| PBKDF2 | - | 低 | 条件付き | 反復回数 | FIPS 140-2準拠が必要な場合 |
| bcrypt | - | 中 | 推奨 | rounds | 72バイト制限あり |
| scrypt | あり | 高 | 推奨 | N, r, p | パラメータ調整が複雑 |
| Argon2id | あり | 最高 | 最推奨 | time, memory, parallelism | Password Hashing Competition 優勝 |

---

## 2. セッション管理

### 2.1 セッション管理の全体像

セッション管理はステートレスな HTTP プロトコル上でユーザーの状態を維持する仕組みである。セッション管理の脆弱性は、攻撃者がユーザーになりすますことを可能にするため、認証と同等に重要である。

```
セッション管理のフロー:

  クライアント                            サーバー
    |                                       |
    |-- POST /login (credentials) -------> |
    |                                       |-- 認証成功
    |                                       |-- セッションID生成
    |                                       |   (暗号学的乱数 256bit以上)
    |<-- Set-Cookie: sid=<random> --------- |
    |    HttpOnly; Secure; SameSite=Lax     |
    |    Path=/; Max-Age=3600               |
    |                                       |
    |-- GET /dashboard ------------------>  |
    |   Cookie: sid=<random>                |-- セッション検証
    |                                       |   1. セッションID存在確認
    |                                       |   2. 絶対タイムアウト確認
    |                                       |   3. アイドルタイムアウト確認
    |                                       |   4. UA/IPの一貫性確認
    |                                       |-- ユーザー情報取得
    |<-- 200 OK (ダッシュボード) ----------- |
    |                                       |
    |-- POST /logout -------------------->  |
    |                                       |-- サーバー側セッション破棄
    |                                       |-- 関連データ削除
    |<-- Set-Cookie: sid=; Max-Age=0 -----  |
```

### 2.2 セッション管理の脆弱性パターン

```
セッション管理の主要な脅威:

  +--------------------------------------------------+
  | 1. セッション固定攻撃 (Session Fixation)            |
  |    攻撃者が事前にセッションIDを設定し、               |
  |    ログイン後もそのIDが使い回される                    |
  |    対策: ログイン成功時にセッションIDを再生成          |
  +--------------------------------------------------+
  | 2. セッションハイジャック                            |
  |    ネットワーク盗聴やXSSでセッションIDを窃取          |
  |    対策: Secure属性, HttpOnly属性, HTTPS必須        |
  +--------------------------------------------------+
  | 3. セッション予測                                   |
  |    推測可能なセッションIDを使用                       |
  |    対策: 暗号学的安全乱数 (secrets.token_hex)       |
  +--------------------------------------------------+
  | 4. セッション情報のクライアント側保存                  |
  |    Cookie にユーザーIDや権限を直接格納               |
  |    対策: サーバー側セッションストア                    |
  +--------------------------------------------------+
```

### 2.3 セキュアなセッション管理の実装

```python
# コード例3: セキュアなセッション管理
import secrets
import time
import hashlib
from typing import Optional, Dict


class SecureSessionManager:
    """安全なセッション管理

    WHY 各設計判断が必要か:
    - SESSION_ID_LENGTH=32: 256ビットのエントロピーで推測攻撃を防止
      (2^256 の組み合わせ → ブルートフォース不可能)
    - SESSION_TIMEOUT=3600: セッション固定攻撃の時間窓を制限
    - IDLE_TIMEOUT=1800: 離席時のセッション悪用を防止
    - UA ハッシュ検証: セッションハイジャックの検知
    """

    SESSION_ID_LENGTH = 32   # 32バイト = 256ビットのエントロピー
    SESSION_TIMEOUT = 3600   # 絶対タイムアウト: 1時間
    IDLE_TIMEOUT = 1800      # アイドルタイムアウト: 30分
    MAX_SESSIONS_PER_USER = 5  # 同時セッション数制限

    def __init__(self):
        # 本番環境では Redis 等の外部ストアを使用
        self.sessions: Dict[str, dict] = {}

    def create_session(self, user_id: str, ip: str,
                       user_agent: str) -> str:
        """認証成功後にセッションを作成

        Returns:
            暗号学的に安全なセッションID文字列
        """
        # 同時セッション数の制限
        user_sessions = [
            sid for sid, data in self.sessions.items()
            if data["user_id"] == user_id
        ]
        if len(user_sessions) >= self.MAX_SESSIONS_PER_USER:
            # 最も古いセッションを削除
            oldest = min(user_sessions,
                        key=lambda s: self.sessions[s]["created_at"])
            self.destroy_session(oldest)

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
        """セッションの検証

        検証項目:
        1. セッションの存在確認
        2. 絶対タイムアウト（作成から一定時間経過）
        3. アイドルタイムアウト（最終アクティビティから一定時間経過）
        4. User-Agent の一貫性（セッションハイジャック検知）
        """
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

        # User-Agent の変更を検出（セッションハイジャックの兆候）
        ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()
        if session["user_agent_hash"] != ua_hash:
            self.destroy_session(session_id)
            return None

        # 最終アクティビティ更新
        session["last_activity"] = now
        return session["user_id"]

    def regenerate_session(self, old_session_id: str) -> Optional[str]:
        """セッションIDの再生成（権限昇格時に必須）

        WHY: ログイン前に攻撃者が設定したセッションIDが
        ログイン後も使い回されるとセッション固定攻撃が成立する。
        ログイン成功・権限変更時に必ずセッションIDを再生成する。
        """
        session = self.sessions.get(old_session_id)
        if not session:
            return None
        # 旧セッションを削除
        del self.sessions[old_session_id]
        # 新セッションIDで同じデータを登録
        new_session_id = secrets.token_hex(self.SESSION_ID_LENGTH)
        session["last_activity"] = time.time()
        self.sessions[new_session_id] = session
        return new_session_id

    def destroy_session(self, session_id: str) -> None:
        """セッションの破棄"""
        self.sessions.pop(session_id, None)

    def destroy_all_user_sessions(self, user_id: str) -> int:
        """特定ユーザーの全セッションを破棄

        用途: パスワード変更時、アカウント侵害検出時
        """
        to_delete = [
            sid for sid, data in self.sessions.items()
            if data["user_id"] == user_id
        ]
        for sid in to_delete:
            del self.sessions[sid]
        return len(to_delete)

    def get_active_sessions(self, user_id: str) -> list:
        """ユーザーのアクティブセッション一覧を取得

        ユーザーが自分の全セッションを確認・管理できるようにする。
        """
        now = time.time()
        return [
            {
                "session_id": sid[:8] + "...",  # 先頭のみ表示
                "created_at": data["created_at"],
                "last_activity": data["last_activity"],
                "ip": data["ip"],
            }
            for sid, data in self.sessions.items()
            if data["user_id"] == user_id
            and now - data["created_at"] <= self.SESSION_TIMEOUT
        ]


# 使用例
sm = SecureSessionManager()

# ログイン成功時
session_id = sm.create_session(
    "user123", "192.168.1.100", "Mozilla/5.0..."
)
print(f"セッション作成: {session_id[:16]}...")

# セッションID再生成（権限昇格時）
new_session_id = sm.regenerate_session(session_id)
print(f"セッション再生成: {new_session_id[:16]}...")

# セッション検証
user_id = sm.validate_session(
    new_session_id, "192.168.1.100", "Mozilla/5.0..."
)
print(f"認証ユーザー: {user_id}")  # user123

# ログアウト時
sm.destroy_session(new_session_id)
```

### 2.4 Cookie 属性の設計

| 属性 | 設定値 | 目的 | 設定しない場合のリスク |
|------|--------|------|---------------------|
| `HttpOnly` | `true` | JavaScript からのアクセスを禁止 | XSS でセッションID窃取 |
| `Secure` | `true` | HTTPS のみで送信 | 平文通信で盗聴される |
| `SameSite` | `Lax` or `Strict` | クロスサイトリクエストでの送信を制限 | CSRF 攻撃のリスク |
| `Path` | `/` | Cookie の送信範囲 | 不要な範囲に送信される |
| `Max-Age` | `3600` | Cookie の有効期間 | ブラウザ終了まで永続 |
| `Domain` | 省略推奨 | Cookie の送信先ドメイン | サブドメインに漏洩 |

---

## 3. ブルートフォース対策

### 3.1 多層防御の設計

ブルートフォース攻撃は単純だが効果的な攻撃であり、単一の対策では不十分である。複数の防御層を組み合わせることで、攻撃の成功率を実質的にゼロにする。

```
ブルートフォース対策の層:

  +--------------------------------------------------+
  |  Layer 1: レートリミット                           |
  |  - IP単位: 10回/分                                |
  |  - アカウント単位: 5回/5分                         |
  |  - グローバル: 異常な認証失敗率を検知               |
  +--------------------------------------------------+
  |  Layer 2: プログレッシブ遅延                       |
  |  - 1回目失敗: 即応答                              |
  |  - 2回目失敗: 1秒待機                             |
  |  - 3回目失敗: 2秒待機                             |
  |  - 5回目失敗: 15秒待機                            |
  |  ※一定時間の応答遅延で自動化攻撃を非効率にする      |
  +--------------------------------------------------+
  |  Layer 3: CAPTCHA                                 |
  |  - 3回失敗後にCAPTCHA表示                         |
  |  - reCAPTCHA v3 (スコアベース) が推奨              |
  +--------------------------------------------------+
  |  Layer 4: アカウントロックアウト                    |
  |  - 10回失敗: 30分ロック                           |
  |  - 一定期間後に自動解除                            |
  |  - ※DoS攻撃に悪用されないよう自動解除必須          |
  +--------------------------------------------------+
  |  Layer 5: 異常検知                                 |
  |  - 同一IPから複数アカウントへの試行                 |
  |  - 地理的に不自然なログイン                        |
  |  - Credential Stuffing パターンの検知              |
  +--------------------------------------------------+
```

### 3.2 ブルートフォース対策の実装

```python
# コード例4: ブルートフォース対策の多層実装
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LoginAttempt:
    """ログイン試行の追跡データ"""
    count: int = 0
    first_attempt: float = 0
    last_attempt: float = 0
    locked_until: float = 0


class BruteForceProtection:
    """ブルートフォース攻撃対策

    WHY プログレッシブ遅延とロックアウトを組み合わせるか:
    - 遅延だけ: 攻撃を遅くするが止められない
    - ロックアウトだけ: 正規ユーザーを DoS できてしまう
    - 組み合わせ: 攻撃を遅くし、最終的に停止させつつ、
      自動解除で正規ユーザーへの影響を最小化する
    """

    MAX_ATTEMPTS = 5
    LOCKOUT_DURATION = 1800  # 30分
    WINDOW = 300             # 5分間のウィンドウ
    PROGRESSIVE_DELAYS = [0, 1, 2, 4, 8, 15]  # 秒

    def __init__(self):
        self.account_attempts: Dict[str, LoginAttempt] = defaultdict(
            LoginAttempt
        )
        self.ip_attempts: Dict[str, LoginAttempt] = defaultdict(LoginAttempt)

    def check_and_record(self, username: str, ip: str) -> dict:
        """ログイン試行を確認・記録する

        アカウント単位とIP単位の両方でチェックする。
        Credential Stuffing 攻撃（同一IPから多数のアカウントへの試行）
        を検知するため、IP単位のチェックも重要。
        """
        # アカウント単位のチェック
        account_result = self._check_identifier(
            self.account_attempts, username
        )
        if not account_result["allowed"]:
            return account_result

        # IP単位のチェック（より緩いレート）
        ip_result = self._check_identifier(
            self.ip_attempts, ip, max_attempts=20
        )
        if not ip_result["allowed"]:
            ip_result["reason"] = "ip_rate_limited"
            return ip_result

        return account_result

    def _check_identifier(self, attempts_store: dict,
                          identifier: str,
                          max_attempts: int = None) -> dict:
        """識別子に対するチェックと記録"""
        if max_attempts is None:
            max_attempts = self.MAX_ATTEMPTS

        attempt = attempts_store[identifier]
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
        if attempt.count >= max_attempts:
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
            "attempts_remaining": max_attempts - attempt.count,
            "require_captcha": attempt.count >= 3,
        }

    def reset(self, username: str) -> None:
        """ログイン成功時にカウンターをリセット"""
        self.account_attempts.pop(username, None)

    def is_credential_stuffing(self, ip: str, threshold: int = 10) -> bool:
        """Credential Stuffing 攻撃の検知

        同一IPから短時間に多数の異なるアカウントへログインを
        試行するパターンを検知する。
        """
        ip_attempt = self.ip_attempts.get(ip)
        if ip_attempt and ip_attempt.count >= threshold:
            return True
        return False


# 使用例
protection = BruteForceProtection()

# ログイン試行
result = protection.check_and_record("admin", "192.168.1.100")
print(f"許可: {result['allowed']}")
print(f"残り試行: {result.get('attempts_remaining')}")

# 5回失敗後
for _ in range(5):
    result = protection.check_and_record("admin", "192.168.1.100")

print(f"ロックアウト: {not result['allowed']}")
print(f"解除まで: {result.get('retry_after')}秒")

# ログイン成功時のリセット
protection.reset("admin")
```

---

## 4. 多要素認証（MFA）

### 4.1 MFA の認証要素

```
多要素認証の3つの要素:

  +------------------+     +------------------+     +------------------+
  |  知識要素 (SYK)   |     |  所持要素 (SYH)   |     |  生体要素 (SYA)   |
  |  Something You   |     |  Something You   |     |  Something You   |
  |  Know             |     |  Have             |     |  Are              |
  +------------------+     +------------------+     +------------------+
  |  パスワード       |     |  スマートフォン    |     |  指紋             |
  |  PIN              |     |  ハードウェアトークン|    |  顔認証           |
  |  秘密の質問       |     |  セキュリティキー  |     |  虹彩             |
  +------------------+     +------------------+     +------------------+

  MFA = 異なるカテゴリの要素を2つ以上組み合わせる
  ※同じカテゴリ2つ（パスワード+PIN）は MFA にならない
```

### 4.2 TOTP の実装

```python
# コード例5: TOTP（Time-based One-Time Password）の実装
import hmac
import hashlib
import struct
import time
import base64
import secrets


class TOTP:
    """RFC 6238 準拠の TOTP 実装

    WHY TOTP が広く使われるか:
    - サーバーとクライアントが時刻のみを共有すれば良い
    - 通信不要（オフラインで生成可能）
    - Google Authenticator 等の標準アプリで利用可能
    - SMS OTP と異なり SIM スワップ攻撃に耐性がある
    """

    def __init__(self, secret: bytes, digits: int = 6,
                 period: int = 30, algorithm=hashlib.sha1):
        """
        Args:
            secret: 共有シークレット（最低160ビット推奨）
            digits: OTP の桁数（6桁が標準）
            period: コード更新間隔（30秒が標準）
            algorithm: HMACアルゴリズム（SHA-1が標準、SHA-256も可）
        """
        self.secret = secret
        self.digits = digits
        self.period = period
        self.algorithm = algorithm

    @classmethod
    def generate_secret(cls) -> str:
        """新しい TOTP シークレットを生成

        Base32エンコードで返す（QRコード/手入力用）。
        20バイト = 160ビットのエントロピー。
        """
        return base64.b32encode(secrets.token_bytes(20)).decode()

    def generate_code(self, timestamp: float = None) -> str:
        """現在の TOTP コードを生成

        内部処理:
        1. 現在時刻を period で割ってカウンター値を算出
        2. カウンター値を HMAC でハッシュ
        3. Dynamic Truncation で 6 桁の数値を抽出
        """
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
        """TOTP コードを検証

        WHY window が必要か:
        サーバーとクライアントの時刻にわずかなズレがあっても
        認証が成功するよう、前後 window ステップ分を許容する。
        window=1 の場合、前後30秒（合計90秒間）有効。
        """
        now = time.time()
        for offset in range(-window, window + 1):
            check_time = now + (offset * self.period)
            if hmac.compare_digest(
                self.generate_code(check_time), code
            ):
                return True
        return False

    def get_provisioning_uri(self, account: str,
                             issuer: str) -> str:
        """QRコード用のプロビジョニングURIを生成

        Google Authenticator 等のアプリで読み取るためのURI。
        """
        secret_b32 = base64.b32encode(self.secret).decode()
        return (
            f"otpauth://totp/{issuer}:{account}"
            f"?secret={secret_b32}"
            f"&issuer={issuer}"
            f"&algorithm=SHA1"
            f"&digits={self.digits}"
            f"&period={self.period}"
        )


# 使用例
secret_b32 = TOTP.generate_secret()
print(f"シークレット: {secret_b32}")

secret_bytes = base64.b32decode(secret_b32)
totp = TOTP(secret_bytes)

# QRコード用URI
uri = totp.get_provisioning_uri("user@example.com", "MyApp")
print(f"プロビジョニングURI: {uri}")

# コード生成と検証
code = totp.generate_code()
print(f"現在のコード: {code}")
print(f"検証結果: {totp.verify(code)}")  # True
```

### 4.3 認証方式の比較

| 方式 | セキュリティ | ユーザビリティ | コスト | フィッシング耐性 | SIMスワップ耐性 |
|------|:--------:|:--------:|:-----:|:--------:|:--------:|
| パスワードのみ | 低 | 高 | 低 | なし | - |
| パスワード + SMS OTP | 中 | 中 | 中 | 低 | なし |
| パスワード + TOTP | 高 | 中 | 低 | 低 | あり |
| パスワード + プッシュ通知 | 高 | 高 | 中 | 中 | あり |
| パスワード + FIDO2/WebAuthn | 最高 | 高 | 中 | 高 | あり |
| パスキー（Passkeys） | 最高 | 最高 | 低 | 最高 | あり |

```
認証方式の推奨フロー（2024年〜）:

  新規システム設計
    |
    +-- パスキー（Passkeys）対応可能か？
    |     YES → パスキーを第一選択肢として実装
    |     NO  → FIDO2/WebAuthn を検討
    |
    +-- ハードウェアキー配布可能か？
    |     YES → FIDO2 セキュリティキー
    |     NO  → TOTP (Google Authenticator 等)
    |
    +-- 最終手段（非推奨）
          → SMS OTP（SIMスワップ、SS7攻撃のリスクあり）
```

---

## 5. JWT の安全な使用

### 5.1 JWT の脅威モデル

```
JWT に対する主要な攻撃:

  +--------------------------------------------------+
  | 1. Algorithm None 攻撃                            |
  |    ヘッダの alg を "none" に変更し署名をバイパス      |
  |    対策: algorithms パラメータで許可する             |
  |          アルゴリズムを明示的に指定                  |
  +--------------------------------------------------+
  | 2. Algorithm Confusion 攻撃                       |
  |    RS256 (非対称) を HS256 (対称) に変更し          |
  |    公開鍵を共通鍵として署名を偽造                    |
  |    対策: アルゴリズムをサーバー側で固定               |
  +--------------------------------------------------+
  | 3. 署名未検証                                      |
  |    ペイロードだけデコードして検証なしで使用            |
  |    対策: jwt.decode() 時に必ず verify=True          |
  +--------------------------------------------------+
  | 4. 長すぎる有効期限                                 |
  |    有効期限が数日〜無期限に設定されている              |
  |    対策: アクセストークン 15分、リフレッシュ 7日       |
  +--------------------------------------------------+
  | 5. JWTの即時無効化不能                              |
  |    ステートレスなため、発行済みトークンを              |
  |    即座に無効化できない                              |
  |    対策: 短い有効期限 + ブラックリスト                |
  +--------------------------------------------------+
```

### 5.2 安全な JWT 実装

```python
# コード例6: JWT の安全な実装
import jwt
import time
import secrets
from typing import Optional


class SecureJWT:
    """安全な JWT トークン管理

    WHY HS256 ではなく RS256 を推奨するか:
    - HS256: 署名と検証に同じ秘密鍵を使用
      → 検証側にも秘密鍵を共有する必要がある
      → マイクロサービスでは鍵の配布が問題になる
    - RS256: 秘密鍵で署名、公開鍵で検証
      → 検証側には公開鍵のみ配布すればよい
      → JWKS エンドポイントで公開鍵を自動配布可能
    """

    # トークンのブラックリスト（即時無効化用）
    _blacklist: set = set()

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        # 安全でないアルゴリズムを拒否
        allowed_algorithms = {"HS256", "HS384", "HS512",
                              "RS256", "RS384", "RS512",
                              "ES256", "ES384", "ES512"}
        if algorithm not in allowed_algorithms:
            raise ValueError(
                f"Algorithm '{algorithm}' is not allowed. "
                f"Use one of: {allowed_algorithms}"
            )
        self.algorithm = algorithm

    def create_token(self, user_id: str, roles: list,
                     expires_in: int = 900) -> str:
        """アクセストークンを生成（デフォルト15分）

        最小権限の原則: トークンには必要最小限の情報のみ含める。
        個人情報（メールアドレス等）はトークンに含めない。
        """
        now = int(time.time())
        payload = {
            "sub": user_id,
            "roles": roles,
            "iat": now,
            "exp": now + expires_in,
            "nbf": now,                      # Not Before
            "jti": secrets.token_hex(16),    # JWT ID（リプレイ防止）
            "iss": "myapp",                  # 発行者
            "aud": "myapp-api",              # 対象者
        }
        return jwt.encode(payload, self.secret_key,
                          algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str,
                             expires_in: int = 604800) -> str:
        """リフレッシュトークンを生成（デフォルト7日）

        リフレッシュトークンはアクセストークンの再発行にのみ使用。
        スコープを制限し、DBに保存して管理する。
        """
        now = int(time.time())
        payload = {
            "sub": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + expires_in,
            "jti": secrets.token_hex(16),
            "iss": "myapp",
        }
        return jwt.encode(payload, self.secret_key,
                          algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """トークンを検証

        検証項目:
        1. 署名の正当性（改竄検知）
        2. 有効期限（exp）
        3. 発行時刻（iat）
        4. Not Before（nbf）
        5. 発行者（iss）
        6. 対象者（aud）
        7. ブラックリストチェック
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # リスト形式で明示指定
                options={
                    "require": ["exp", "iat", "sub", "iss"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                },
                issuer="myapp",
                audience="myapp-api",
            )

            # ブラックリストチェック
            if payload.get("jti") in self._blacklist:
                return None

            return payload
        except jwt.ExpiredSignatureError:
            return None  # トークン期限切れ
        except jwt.InvalidTokenError:
            return None  # その他の検証エラー

    def revoke_token(self, token: str) -> bool:
        """トークンを即座に無効化（ブラックリスト方式）

        短い有効期限と組み合わせることで、
        ブラックリストのサイズを制限する。
        """
        try:
            # 署名検証なしでペイロードを取得（jti のみ必要）
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            if jti:
                self._blacklist.add(jti)
                return True
        except jwt.InvalidTokenError:
            pass
        return False


# 使用例
jwt_manager = SecureJWT("my-secret-key-at-least-256-bits-long!!")

# アクセストークン生成
access_token = jwt_manager.create_token("user123", ["viewer"])
print(f"アクセストークン: {access_token[:50]}...")

# リフレッシュトークン生成
refresh_token = jwt_manager.create_refresh_token("user123")

# トークン検証
payload = jwt_manager.verify_token(access_token)
print(f"ペイロード: {payload}")

# トークン無効化
jwt_manager.revoke_token(access_token)
revoked_result = jwt_manager.verify_token(access_token)
print(f"無効化後の検証: {revoked_result}")  # None
```

### 5.3 JWT とセッションの使い分け

| 観点 | セッションベース | JWT |
|------|---------------|-----|
| 状態管理 | サーバー側（ステートフル） | クライアント側（ステートレス） |
| スケーラビリティ | セッションストアが必要 | サーバー間の状態共有不要 |
| 即時無効化 | セッション削除で即座に無効化 | ブラックリスト等の追加実装が必要 |
| トークンサイズ | セッションID のみ（小） | ペイロード含む（大） |
| 適するアーキテクチャ | モノリス、サーバーレンダリング | マイクロサービス、SPA |
| 実装の複雑さ | 低（フレームワーク支援あり） | 中（セキュリティ考慮点多数） |

---

## 6. パスワードリセットのセキュリティ

### 6.1 安全なパスワードリセットフロー

```python
# コード例7: セキュアなパスワードリセットの実装
import secrets
import time
import hashlib
from typing import Optional, Tuple


class PasswordResetManager:
    """安全なパスワードリセット管理

    パスワードリセットの脆弱性パターン:
    1. 推測可能なリセットトークン（連番、タイムスタンプベース）
    2. トークンの有効期限なし
    3. トークンの使い回し（リプレイ攻撃）
    4. ユーザーの存在有無が推測可能なエラーメッセージ
    """

    TOKEN_EXPIRY = 3600  # 1時間
    TOKEN_LENGTH = 32    # 256ビット

    def __init__(self):
        self.reset_tokens: dict = {}

    def request_reset(self, email: str) -> Tuple[str, str]:
        """パスワードリセットをリクエスト

        セキュリティ上の注意:
        - ユーザーの存在有無に関わらず同じレスポンスを返す
        - レートリミットを適用（同一メールへの連続リクエストを制限）
        """
        # 暗号学的に安全なトークンを生成
        token = secrets.token_urlsafe(self.TOKEN_LENGTH)

        # トークンをハッシュ化して保存（DB漏洩時の保護）
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        self.reset_tokens[token_hash] = {
            "email": email,
            "created_at": time.time(),
            "used": False,
        }

        return token, token_hash

    def verify_and_consume(self, token: str) -> Optional[str]:
        """リセットトークンを検証して消費（一回限り使用）"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        data = self.reset_tokens.get(token_hash)

        if not data:
            return None

        # 有効期限チェック
        if time.time() - data["created_at"] > self.TOKEN_EXPIRY:
            del self.reset_tokens[token_hash]
            return None

        # 使用済みチェック
        if data["used"]:
            # トークンの再利用はセキュリティインシデントの可能性
            del self.reset_tokens[token_hash]
            return None

        # トークンを使用済みとしてマーク
        data["used"] = True

        return data["email"]


# 使用例
reset_mgr = PasswordResetManager()
token, _ = reset_mgr.request_reset("user@example.com")
email = reset_mgr.verify_and_consume(token)
print(f"リセット対象: {email}")

# 2回目の使用は失敗
email2 = reset_mgr.verify_and_consume(token)
print(f"再利用: {email2}")  # None
```

---

## 7. Credential Stuffing 対策

```
Credential Stuffing 攻撃の仕組み:

  攻撃者
    |
    |-- 漏洩したメール/パスワードリスト（数百万件）
    |   (他サイトのデータ漏洩から入手)
    |
    |-- 自動化ツールで標的サイトにログイン試行
    |   ┌──────────────────────────────────────┐
    |   │  user1@mail.com / pass123            │
    |   │  user2@mail.com / qwerty             │
    |   │  user3@mail.com / letmein            │
    |   │  ...（数百万回）                      │
    |   └──────────────────────────────────────┘
    |
    +-- パスワード使い回しユーザーのアカウントに不正アクセス

  対策:
  1. HIBP API でパスワード漏洩チェック
  2. IP ベースのレートリミット
  3. CAPTCHA（自動化ツール対策）
  4. MFA（パスワード漏洩しても突破されない）
  5. 異常ログイン検知（新しいデバイス/場所からの通知）
```

---

## アンチパターン

### アンチパターン1: パスワードの平文/可逆暗号化保存

```python
# NG: パスワードを平文で保存
def save_user(username, password):
    db.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, password)  # 平文！
    )

# NG: AES等の可逆暗号化で保存
from cryptography.fernet import Fernet
key = Fernet.generate_key()
def save_user_encrypted(username, password):
    encrypted = Fernet(key).encrypt(password.encode())
    db.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, encrypted)  # 復号可能 = 鍵が漏洩すれば全滅
    )

# OK: Argon2id で一方向ハッシュ化
from argon2 import PasswordHasher
ph = PasswordHasher()
def save_user_secure(username, password):
    hashed = ph.hash(password)  # 復元不可能
    db.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, hashed)
    )
```

**なぜ危険か**: 平文保存はデータベース漏洩時に即座に全ユーザーのパスワードが露出する。可逆暗号化は暗号鍵の漏洩で同じ結果になる。パスワードは復元する必要がないため、必ず一方向ハッシュを使用する。

### アンチパターン2: JWT の `algorithm: "none"` 攻撃

```python
# NG: アルゴリズムを検証しない
import jwt
payload = jwt.decode(token, secret, algorithms=None)  # 全アルゴリズム許可

# NG: デフォルト設定のまま使用（ライブラリによっては none を許可）
payload = jwt.decode(token, secret)

# OK: 許可するアルゴリズムを明示的に指定
payload = jwt.decode(
    token,
    secret,
    algorithms=["HS256"],  # HS256 のみ許可
    options={"require": ["exp", "iss"]},
)
```

**なぜ危険か**: 攻撃者が JWT ヘッダの `alg` を `"none"` に変更すると、署名検証がスキップされ、任意のペイロードで認証をバイパスできる。2015 年に多くの JWT ライブラリでこの脆弱性が発見された（CVE-2015-9235）。

### アンチパターン3: セッション固定攻撃に無防備

```python
# NG: ログイン前後でセッションIDが変わらない
@app.route("/login", methods=["POST"])
def login():
    if authenticate(request.form["username"], request.form["password"]):
        session["user_id"] = get_user_id(request.form["username"])
        return redirect("/dashboard")
        # セッションIDは変わらない！攻撃者が事前に仕込んだIDがそのまま使われる

# OK: ログイン成功時にセッションIDを再生成
@app.route("/login", methods=["POST"])
def login_secure():
    if authenticate(request.form["username"], request.form["password"]):
        session.regenerate()  # セッションIDを再生成
        session["user_id"] = get_user_id(request.form["username"])
        return redirect("/dashboard")
```

**なぜ危険か**: 攻撃者が事前にセッションIDをユーザーのブラウザに設定し、ユーザーがそのIDでログインすると、攻撃者も同じセッションIDでアクセスできてしまう。

### アンチパターン4: エラーメッセージからのユーザー列挙

```python
# NG: ユーザーの存在を推測可能なエラーメッセージ
def login(username, password):
    user = find_user(username)
    if not user:
        return {"error": "ユーザーが見つかりません"}  # ユーザー存在が分かる
    if not verify_password(password, user.password_hash):
        return {"error": "パスワードが間違っています"}  # ユーザーは存在する

# OK: 同一のエラーメッセージ
def login_secure(username, password):
    user = find_user(username)
    if not user or not verify_password(password, user.password_hash if user else "dummy"):
        return {"error": "ユーザー名またはパスワードが間違っています"}
```

---

## FAQ

### Q1: パスワードの最大長は制限すべきですか?

128文字程度の上限は設定すべきである。Argon2id や bcrypt はパスワードをハッシュ化する際に計算コストがかかるが、極端に長いパスワード（数万文字等）を送信されると DoS 攻撃になり得る。ただし、bcrypt 固有の 72 バイト制限には注意が必要で、それ以上のパスワードは事前に SHA-256 等でハッシュしてから bcrypt に渡す方法がある。NIST SP 800-63B では最低でも 64 文字以上を受け入れることを推奨している。

### Q2: セッションIDはどこに保存すべきですか?

`HttpOnly` + `Secure` + `SameSite` 属性付きの Cookie が推奨される。localStorage は XSS に脆弱（JavaScript からアクセス可能）であり、URL パラメータは Referer ヘッダやブラウザ履歴から漏洩する危険がある。sessionStorage はタブを閉じると消えるため、ユーザビリティの観点からも Cookie が最適である。

### Q3: JWT とセッションベース認証のどちらを使うべきですか?

アーキテクチャと要件に依存する。モノリスアプリケーションではセッションベースが推奨される（シンプルで即時無効化が容易）。マイクロサービスや SPA では JWT が適している（サーバー間の状態共有不要）。ただし、JWT には即座にトークンを無効化できないという課題があるため、短い有効期限（15分）とリフレッシュトークン（7日）の組み合わせ + ブラックリスト機能が必要である。

### Q4: SMS OTP はなぜ非推奨になったのですか?

NIST SP 800-63B で SMS OTP は「制限付き（restricted）」の認証手段とされた。理由は以下の通り:
- **SIM スワップ攻撃**: 攻撃者が携帯キャリアを騙して被害者の電話番号を別の SIM に移す
- **SS7 プロトコルの脆弱性**: 通信経路上で SMS を傍受可能
- **マルウェア**: スマートフォン上のマルウェアが SMS を読み取る
代替として TOTP やFIDO2/WebAuthn が推奨される。

### Q5: パスワードレス認証は安全ですか?

パスキー（Passkeys）に代表されるパスワードレス認証は、従来のパスワード認証より安全性が高い。公開鍵暗号ベースでフィッシング耐性があり、生体認証と組み合わせることでユーザビリティも向上する。FIDO Alliance と W3C が標準化を推進しており、Apple・Google・Microsoft が対応済みである。2024年以降の新規システムでは第一候補として検討すべきである。

---

## 実践演習

### 演習1（基礎）: パスワードポリシーチェッカーの実装

以下の要件を満たすパスワードポリシーチェッカーを実装せよ。

**要件**:
- 最小12文字、最大128文字
- 大文字・小文字・数字をそれぞれ1文字以上含む
- 過去に使用したパスワード3件との重複を禁止
- 一般的な弱いパスワード（"password", "123456" 等）を拒否

<details>
<summary>模範解答</summary>

```python
import hashlib
from typing import List, Optional


# よく使われる弱いパスワードリスト（実運用ではより大きなリストを使用）
COMMON_PASSWORDS = {
    "password", "123456", "12345678", "qwerty", "abc123",
    "monkey", "1234567", "letmein", "trustno1", "dragon",
    "baseball", "master", "michael", "shadow", "ashley",
    "password1", "password123", "admin", "welcome", "login",
}


class PasswordPolicyChecker:
    """パスワードポリシーチェッカー"""

    MIN_LENGTH = 12
    MAX_LENGTH = 128
    HISTORY_SIZE = 3

    def __init__(self):
        # ユーザーごとのパスワード履歴（ハッシュで保存）
        self.password_history: dict = {}

    def check(self, password: str, user_id: str = None) -> List[str]:
        """パスワードポリシーをチェック"""
        errors = []

        # 長さチェック
        if len(password) < self.MIN_LENGTH:
            errors.append(f"{self.MIN_LENGTH}文字以上必要です")
        if len(password) > self.MAX_LENGTH:
            errors.append(f"{self.MAX_LENGTH}文字以下にしてください")

        # 文字種チェック
        if not any(c.isupper() for c in password):
            errors.append("大文字を1文字以上含めてください")
        if not any(c.islower() for c in password):
            errors.append("小文字を1文字以上含めてください")
        if not any(c.isdigit() for c in password):
            errors.append("数字を1文字以上含めてください")

        # 弱いパスワードチェック
        if password.lower() in COMMON_PASSWORDS:
            errors.append("一般的すぎるパスワードは使用できません")

        # パスワード履歴チェック
        if user_id and user_id in self.password_history:
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            if pw_hash in self.password_history[user_id]:
                errors.append(
                    f"直近{self.HISTORY_SIZE}件と同じパスワードは使用できません"
                )

        return errors

    def record_password(self, user_id: str, password: str) -> None:
        """パスワード履歴に記録"""
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        if user_id not in self.password_history:
            self.password_history[user_id] = []
        history = self.password_history[user_id]
        history.append(pw_hash)
        # 履歴サイズを制限
        if len(history) > self.HISTORY_SIZE:
            history.pop(0)


# テスト
checker = PasswordPolicyChecker()

# 弱いパスワード
print(checker.check("password"))
# ['12文字以上必要です', '数字を1文字以上含めてください', '一般的すぎるパスワードは使用できません']

# 強いパスワード
print(checker.check("MyStr0ngP@ssword2024"))
# []

# パスワード履歴チェック
checker.record_password("user1", "MyStr0ngP@ssword2024")
print(checker.check("MyStr0ngP@ssword2024", "user1"))
# ['直近3件と同じパスワードは使用できません']
```

</details>

### 演習2（応用）: レートリミット付きログインAPIの実装

Flask を使用して、以下の要件を満たすログイン API を実装せよ。

**要件**:
- Argon2id でパスワードを検証
- IP単位とアカウント単位のレートリミット
- 3回失敗後に遅延を増加
- 5回失敗後にアカウントロック（30分）
- ログイン成功時にセッションIDを発行

<details>
<summary>模範解答</summary>

```python
import time
import secrets
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from flask import Flask, request, jsonify, make_response
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

app = Flask(__name__)
ph = PasswordHasher()

# 模擬ユーザーデータベース
users_db = {
    "admin": ph.hash("AdminP@ss123!"),
    "user1": ph.hash("User1P@ss456!"),
}

# セッションストア
sessions = {}


@dataclass
class AttemptTracker:
    count: int = 0
    first_attempt: float = 0.0
    locked_until: float = 0.0


# レートリミットトラッカー
account_attempts = defaultdict(AttemptTracker)
ip_attempts = defaultdict(AttemptTracker)

DELAYS = [0, 0, 0, 2, 4, 8]
MAX_ATTEMPTS = 5
LOCK_DURATION = 1800
WINDOW = 300


def check_rate_limit(tracker: AttemptTracker) -> dict:
    """レートリミットのチェック"""
    now = time.time()
    if tracker.locked_until > now:
        return {"blocked": True, "retry_after": int(tracker.locked_until - now)}
    if now - tracker.first_attempt > WINDOW:
        tracker.count = 0
        tracker.first_attempt = now
    return {"blocked": False, "count": tracker.count}


def record_failure(tracker: AttemptTracker):
    """失敗を記録"""
    now = time.time()
    if tracker.count == 0:
        tracker.first_attempt = now
    tracker.count += 1
    if tracker.count >= MAX_ATTEMPTS:
        tracker.locked_until = now + LOCK_DURATION


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"error": "username and password required"}), 400

    username = data["username"]
    client_ip = request.remote_addr

    # IPレートリミット
    ip_check = check_rate_limit(ip_attempts[client_ip])
    if ip_check["blocked"]:
        return jsonify({
            "error": "Too many requests",
            "retry_after": ip_check["retry_after"],
        }), 429

    # アカウントレートリミット
    acct_check = check_rate_limit(account_attempts[username])
    if acct_check["blocked"]:
        return jsonify({
            "error": "Account locked",
            "retry_after": acct_check["retry_after"],
        }), 429

    # 遅延の適用
    delay_idx = min(acct_check.get("count", 0), len(DELAYS) - 1)
    if DELAYS[delay_idx] > 0:
        time.sleep(DELAYS[delay_idx])

    # 認証
    password_hash = users_db.get(username)
    if not password_hash:
        record_failure(account_attempts[username])
        record_failure(ip_attempts[client_ip])
        return jsonify({"error": "Invalid credentials"}), 401

    try:
        ph.verify(password_hash, data["password"])
    except VerifyMismatchError:
        record_failure(account_attempts[username])
        record_failure(ip_attempts[client_ip])
        return jsonify({"error": "Invalid credentials"}), 401

    # 成功: カウンターリセット
    account_attempts.pop(username, None)

    # セッション発行
    session_id = secrets.token_hex(32)
    sessions[session_id] = {
        "user_id": username,
        "created_at": time.time(),
    }

    response = make_response(jsonify({"message": "Login successful"}))
    response.set_cookie(
        "session_id",
        session_id,
        httponly=True,
        secure=True,
        samesite="Lax",
        max_age=3600,
    )
    return response


if __name__ == "__main__":
    app.run(debug=True)
```

</details>

### 演習3（発展）: JWT リフレッシュトークンローテーションの実装

以下の要件を満たす JWT トークン管理システムを設計・実装せよ。

**要件**:
- アクセストークン（15分）とリフレッシュトークン（7日）のペア発行
- リフレッシュトークンのローテーション（使用するたびに新しいものと交換）
- リフレッシュトークンの再利用検知（盗難検出）
- トークンファミリー管理（同一ログインセッション由来のトークン追跡）

<details>
<summary>模範解答</summary>

```python
import jwt
import time
import secrets
from typing import Optional, Tuple


class TokenRotationManager:
    """JWT リフレッシュトークンローテーション

    リフレッシュトークン再利用検知の仕組み:
    - リフレッシュトークンは使い捨て（一度使ったら無効化）
    - 同一ログインから派生するトークンを「ファミリー」として追跡
    - 使用済みトークンが再度使われた場合、
      そのファミリー全体を無効化（盗難を検知）
    """

    def __init__(self, secret: str):
        self.secret = secret
        # ファミリーID → {使用済みトークンのjtiリスト, 最新のjti}
        self.token_families: dict = {}
        # jti → ファミリーID のマッピング
        self.jti_to_family: dict = {}
        # 無効化されたファミリー
        self.revoked_families: set = set()

    def create_token_pair(self, user_id: str,
                          family_id: str = None) -> Tuple[str, str]:
        """アクセストークンとリフレッシュトークンのペアを生成"""
        now = int(time.time())

        # 新規ログインの場合はファミリーIDを生成
        if family_id is None:
            family_id = secrets.token_hex(16)
            self.token_families[family_id] = {
                "used_jtis": set(),
                "latest_jti": None,
                "user_id": user_id,
            }

        # アクセストークン（15分）
        access_jti = secrets.token_hex(16)
        access_token = jwt.encode({
            "sub": user_id,
            "type": "access",
            "jti": access_jti,
            "iat": now,
            "exp": now + 900,  # 15分
        }, self.secret, algorithm="HS256")

        # リフレッシュトークン（7日）
        refresh_jti = secrets.token_hex(16)
        refresh_token = jwt.encode({
            "sub": user_id,
            "type": "refresh",
            "jti": refresh_jti,
            "family": family_id,
            "iat": now,
            "exp": now + 604800,  # 7日
        }, self.secret, algorithm="HS256")

        # ファミリーの最新jtiを更新
        self.token_families[family_id]["latest_jti"] = refresh_jti
        self.jti_to_family[refresh_jti] = family_id

        return access_token, refresh_token

    def refresh(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """リフレッシュトークンで新しいトークンペアを取得

        ローテーション: 古いリフレッシュトークンは無効化され、
        新しいペアが返される。
        """
        try:
            payload = jwt.decode(
                refresh_token, self.secret,
                algorithms=["HS256"],
                options={"require": ["sub", "jti", "family"]},
            )
        except jwt.InvalidTokenError:
            return None

        jti = payload["jti"]
        family_id = payload["family"]
        user_id = payload["sub"]

        # ファミリーが無効化されている場合
        if family_id in self.revoked_families:
            return None

        family = self.token_families.get(family_id)
        if not family:
            return None

        # 再利用検知: このjtiが既に使用済みの場合
        if jti in family["used_jtis"]:
            # トークン盗難を検知！ファミリー全体を無効化
            self.revoked_families.add(family_id)
            print(f"[ALERT] トークン再利用検知: family={family_id}")
            return None

        # 現在のjtiを使用済みとしてマーク
        family["used_jtis"].add(jti)

        # 新しいトークンペアを生成
        return self.create_token_pair(user_id, family_id)

    def revoke_family(self, family_id: str) -> None:
        """ファミリー全体を無効化（ログアウト時等）"""
        self.revoked_families.add(family_id)


# 使用例
manager = TokenRotationManager("secret-key-256-bits-long!!!!!!!!")

# 初回ログイン
access, refresh = manager.create_token_pair("user123")
print("=== 初回ログイン ===")
print(f"Access: {access[:50]}...")
print(f"Refresh: {refresh[:50]}...")

# リフレッシュ（正常）
new_access, new_refresh = manager.refresh(refresh)
print("\n=== リフレッシュ（正常） ===")
print(f"New Access: {new_access[:50]}...")

# 古いリフレッシュトークンの再利用（盗難検知）
result = manager.refresh(refresh)  # 使用済みトークン
print(f"\n=== 再利用検知 ===")
print(f"結果: {result}")  # None（ファミリー全体が無効化）

# 新しいリフレッシュトークンも無効化されている
result2 = manager.refresh(new_refresh)
print(f"新トークンも無効: {result2}")  # None
```

</details>

---

## まとめ

| 項目 | 推奨対策 | 重要度 |
|------|---------|--------|
| パスワードハッシュ | Argon2id（第一推奨）/ bcrypt | 必須 |
| セッションID | 256ビット以上の暗号学的乱数 | 必須 |
| セッション管理 | HttpOnly/Secure Cookie + 絶対/アイドルタイムアウト | 必須 |
| セッション固定対策 | ログイン時のセッションID再生成 | 必須 |
| ブルートフォース対策 | レートリミット + プログレッシブ遅延 + ロックアウト | 必須 |
| MFA | FIDO2/WebAuthn またはTOTP（SMS OTPは非推奨） | 強く推奨 |
| JWT | 短い有効期限 + アルゴリズム明示指定 + aud/iss 検証 | 条件付き推奨 |
| パスワードリセット | 暗号学的トークン + ハッシュ保存 + 一回限り使用 | 必須 |
| Credential Stuffing | HIBP チェック + 異常ログイン検知 + MFA | 強く推奨 |

---

## 次に読むべきガイド

- [暗号基礎](../02-cryptography/00-crypto-basics.md) -- ハッシュ関数と暗号化アルゴリズムの詳細
- [APIセキュリティ](../03-network-security/02-api-security.md) -- OAuth 2.0/JWT を使った API 認証の詳細
- [セキュアコーディング](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般
- [パスワードセキュリティ](../../authentication-and-authorization/docs/00-fundamentals/01-password-security.md) -- パスワード管理のさらなる深掘り
- [多要素認証](../../authentication-and-authorization/docs/00-fundamentals/02-multi-factor-authentication.md) -- MFA 実装の詳細ガイド
- [セッション vs トークン](../../authentication-and-authorization/docs/00-fundamentals/03-session-vs-token.md) -- セッションとトークンの比較
- [JWT Deep Dive](../../authentication-and-authorization/docs/02-token-auth/00-jwt-deep-dive.md) -- JWT の詳細な仕組みと攻撃手法

---

## 参考文献

1. **OWASP Authentication Cheat Sheet** -- https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
2. **OWASP Session Management Cheat Sheet** -- https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
3. **NIST SP 800-63B: Digital Identity Guidelines** -- https://pages.nist.gov/800-63-3/sp800-63b.html
4. **RFC 6238: TOTP (Time-Based One-Time Password Algorithm)** -- https://datatracker.ietf.org/doc/html/rfc6238
5. **OWASP Password Storage Cheat Sheet** -- https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
6. **FIDO Alliance: Passkeys** -- https://fidoalliance.org/passkeys/
7. **Auth0 Blog: Refresh Token Rotation** -- https://auth0.com/blog/refresh-tokens-what-are-they-and-when-to-use-them/
