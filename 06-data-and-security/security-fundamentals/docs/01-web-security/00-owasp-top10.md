# OWASP Top 10

> Webアプリケーションにおける最も深刻な10のセキュリティリスクを、攻撃手法・影響・対策コード付きで包括的に解説する。

## この章で学ぶこと

1. **OWASP Top 10 (2021)** の各脆弱性カテゴリの意味と深刻度を理解する
2. **各脆弱性の攻撃手法**と実際のコードレベルでの対策を習得する
3. **脆弱性テストの手法**と予防的セキュリティ設計のアプローチを身につける

---

## 1. OWASP Top 10 概観

OWASP（Open Worldwide Application Security Project）が定期的に発表するWebアプリケーション脆弱性のランキング。

```
OWASP Top 10 (2021):

  順位    カテゴリ                               深刻度
  ----    --------                               ------
  A01     アクセス制御の不備                       ████████████ Critical
  A02     暗号化の失敗                            ███████████  Critical
  A03     インジェクション                        ██████████   High
  A04     安全でない設計                          ██████████   High
  A05     セキュリティの設定ミス                   █████████    High
  A06     脆弱で古いコンポーネント                 ████████     High
  A07     識別と認証の失敗                        ████████     High
  A08     ソフトウェアとデータの整合性の不具合      ███████      Medium
  A09     セキュリティログとモニタリングの不備      ██████       Medium
  A10     SSRF（サーバーサイドリクエストフォージェリ）██████       Medium
```

---

## 2. A01: アクセス制御の不備（Broken Access Control）

認可されていないリソースへのアクセスを許してしまう脆弱性。

```python
# コード例1: 安全なアクセス制御の実装
from functools import wraps
from flask import Flask, request, abort, g

app = Flask(__name__)

# 悪い例: IDORの脆弱性
@app.route("/api/orders/<int:order_id>")
def get_order_bad(order_id):
    # 誰でも任意のorder_idにアクセスできてしまう
    order = db.query("SELECT * FROM orders WHERE id = ?", order_id)
    return jsonify(order)

# 良い例: オーナーシップチェック付き
@app.route("/api/orders/<int:order_id>")
@login_required
def get_order_good(order_id):
    order = db.query(
        "SELECT * FROM orders WHERE id = ? AND user_id = ?",
        order_id, g.current_user.id  # ユーザーIDでフィルタ
    )
    if not order:
        abort(404)  # 403ではなく404（情報漏洩防止）
    return jsonify(order)

def require_role(role):
    """ロールベースアクセス制御デコレータ"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not g.current_user or g.current_user.role != role:
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator

@app.route("/admin/users")
@login_required
@require_role("admin")
def admin_users():
    """管理者のみアクセス可能"""
    return jsonify(db.query("SELECT id, name FROM users"))
```

---

## 3. A02: 暗号化の失敗（Cryptographic Failures）

機密データの暗号化が不十分、または暗号化の設計が不適切な脆弱性。

```python
# コード例2: 適切な暗号化の実装
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import bcrypt
import os
import base64

class SecureCrypto:
    """安全な暗号化ユーティリティ"""

    @staticmethod
    def hash_password(password: str) -> str:
        """パスワードのハッシュ化（bcrypt使用）"""
        # 悪い例: MD5やSHA-256の直接使用
        # hashlib.md5(password.encode()).hexdigest()  # NG!

        # 良い例: bcryptによるソルト付きハッシュ
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed.encode())

    @staticmethod
    def encrypt_sensitive_data(plaintext: str, master_key: bytes) -> str:
        """機密データの暗号化（AES-256）"""
        # PBKDF2でマスターキーから暗号鍵を導出
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key))
        f = Fernet(key)
        encrypted = f.encrypt(plaintext.encode())
        # ソルトと暗号文を結合して返す
        return base64.b64encode(salt + encrypted).decode()
```

---

## 4. A03: インジェクション（Injection）

ユーザー入力がコード・クエリの一部として解釈されてしまう脆弱性。

```python
# コード例3: SQLインジェクション対策
import sqlite3

# 悪い例: 文字列連結によるSQL構築
def search_users_bad(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)  # ' OR '1'='1 で全件取得可能

# 良い例: パラメータ化クエリ
def search_users_good(username):
    query = "SELECT * FROM users WHERE name = ?"
    return db.execute(query, (username,))

# 良い例: ORMの使用（SQLAlchemy）
from sqlalchemy import select
from models import User

def search_users_orm(session, username):
    stmt = select(User).where(User.name == username)
    return session.execute(stmt).scalars().all()
```

---

## 5. A04-A10: 概要一覧

```
A04: 安全でない設計
  +--> 脅威モデリングの欠如、セキュリティ要件の不足
  +--> 対策: 設計段階からセキュリティを組み込む

A05: セキュリティの設定ミス
  +--> デフォルト設定、不要な機能の有効化
  +--> 対策: ハードニング、構成管理の自動化

A06: 脆弱で古いコンポーネント
  +--> 既知の脆弱性を持つライブラリの使用
  +--> 対策: SCA、自動アップデート

A07: 識別と認証の失敗
  +--> 弱いパスワード、セッション管理の不備
  +--> 対策: MFA、セキュアなセッション管理

A08: ソフトウェアとデータの整合性の不具合
  +--> CI/CDパイプラインの侵害、安全でないデシリアライゼーション
  +--> 対策: 署名検証、依存関係の検証

A09: セキュリティログとモニタリングの不備
  +--> 侵入検知の遅延、ログの不足
  +--> 対策: SIEM、アラート設定

A10: SSRF
  +--> サーバーから内部リソースへの不正リクエスト
  +--> 対策: URL検証、ネットワーク分離
```

```python
# コード例4: SSRF対策（A10）
import ipaddress
from urllib.parse import urlparse

class SSRFProtection:
    """SSRF攻撃を防止するURL検証"""

    BLOCKED_NETWORKS = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("169.254.0.0/16"),  # リンクローカル
        ipaddress.ip_network("::1/128"),          # IPv6ループバック
    ]

    ALLOWED_SCHEMES = {"http", "https"}

    @classmethod
    def validate_url(cls, url: str) -> bool:
        """外部アクセスに使用するURLを検証する"""
        parsed = urlparse(url)

        # スキームチェック
        if parsed.scheme not in cls.ALLOWED_SCHEMES:
            return False

        # ホスト解決とプライベートIP検出
        try:
            import socket
            resolved_ip = socket.gethostbyname(parsed.hostname)
            ip = ipaddress.ip_address(resolved_ip)
            for network in cls.BLOCKED_NETWORKS:
                if ip in network:
                    return False  # 内部ネットワークへのアクセスを拒否
        except (socket.gaierror, ValueError):
            return False

        return True
```

```python
# コード例5: セキュリティヘッダーの設定（A05対策）
from flask import Flask, Response

app = Flask(__name__)

@app.after_request
def set_security_headers(response: Response) -> Response:
    """全レスポンスにセキュリティヘッダーを付与する"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"  # CSPに委任
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "frame-ancestors 'none'"
    )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains; preload"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    return response
```

### 各脆弱性の対策比較

| 脆弱性 | 主要対策 | ツール | 検出フェーズ |
|--------|---------|-------|-------------|
| A01 アクセス制御 | RBAC、オーナーシップチェック | Burp Suite | DAST |
| A02 暗号化失敗 | TLS 1.3、AES-GCM、bcrypt | testssl.sh | 設計レビュー |
| A03 インジェクション | パラメータ化クエリ、ORM | SQLMap、SAST | SAST/DAST |
| A04 安全でない設計 | 脅威モデリング、セキュリティ要件 | - | 設計レビュー |
| A05 設定ミス | ハードニング、IaC | ScoutSuite | 構成監査 |
| A06 古いコンポーネント | SCA、自動更新 | Dependabot | SCA |
| A07 認証の失敗 | MFA、レートリミット | Hydra | ペネトレーションテスト |
| A08 整合性不具合 | 署名検証、SRI | Sigstore | CI/CD |
| A09 ログ不備 | SIEM、監査ログ | ELK Stack | 運用監視 |
| A10 SSRF | URL検証、ネットワーク分離 | Burp Suite | DAST |

---

## アンチパターン

### アンチパターン1: セキュリティヘッダーの欠如

セキュリティヘッダーを設定せずにアプリケーションをデプロイするパターン。CSP、HSTS、X-Frame-Options等のヘッダーは、追加コストなしでクライアントサイドの攻撃を大幅に緩和できる。

### アンチパターン2: エラーメッセージでの情報漏洩

スタックトレースやDB接続情報をエラーレスポンスに含めるパターン。本番環境では一般的なエラーメッセージのみを返し、詳細はサーバーサイドのログに記録する。

---

## FAQ

### Q1: OWASP Top 10は全てカバーすれば十分ですか?

十分ではない。OWASP Top 10は最も一般的な脆弱性を示したものであり、網羅的なセキュリティチェックリストではない。OWASP ASVS（Application Security Verification Standard）をより包括的なガイドラインとして活用すべきである。

### Q2: A04「安全でない設計」はコードレベルで対策できますか?

コードレベルだけでは対策できない。設計段階での脅威モデリング、セキュリティ要件の定義、アーキテクチャレビューが必要である。「セキュアコーディング」は「安全な設計」を前提として初めて効果を発揮する。

### Q3: どの脆弱性から優先的に対策すべきですか?

自組織のリスクアセスメントに基づいて判断すべきだが、一般的にはA01（アクセス制御）とA03（インジェクション）が最も被害が大きく、対策の優先度が高い。

---

## まとめ

| 順位 | カテゴリ | 核心的な対策 |
|------|---------|-------------|
| A01 | アクセス制御の不備 | サーバーサイドでの認可チェック、デフォルト拒否 |
| A02 | 暗号化の失敗 | TLS強制、適切なアルゴリズム選択 |
| A03 | インジェクション | パラメータ化クエリ、入力検証 |
| A04 | 安全でない設計 | 脅威モデリング、セキュリティ要件 |
| A05 | 設定ミス | ハードニング、自動構成管理 |

---

## 次に読むべきガイド

- [01-xss-prevention.md](./01-xss-prevention.md) -- XSS攻撃の詳細な対策手法
- [03-injection.md](./03-injection.md) -- インジェクション攻撃の深掘り
- [04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md) -- 認証脆弱性の詳細

---

## 参考文献

1. OWASP Top 10:2021 -- https://owasp.org/Top10/
2. OWASP Application Security Verification Standard (ASVS) -- https://owasp.org/www-project-application-security-verification-standard/
3. OWASP Testing Guide -- https://owasp.org/www-project-web-security-testing-guide/
4. CWE/SANS Top 25 Most Dangerous Software Errors -- https://cwe.mitre.org/top25/
