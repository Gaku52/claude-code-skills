# OWASP Top 10

> Webアプリケーションにおける最も深刻な10のセキュリティリスクを、攻撃手法・影響・対策コード付きで包括的に解説する。

## この章で学ぶこと

1. **OWASP Top 10 (2021)** の各脆弱性カテゴリの意味と深刻度を理解する
2. **各脆弱性の攻撃手法**と実際のコードレベルでの対策を習得する
3. **脆弱性テストの手法**と予防的セキュリティ設計のアプローチを身につける

### 前提知識

- HTTP プロトコルの基本（リクエスト/レスポンス、ステータスコード、ヘッダー）
- Webアプリケーション開発の基礎（サーバサイド/クライアントサイドの区別）
- Python または JavaScript の基本構文

### 関連ガイド

- [XSS防止](./01-xss-prevention.md) — クロスサイトスクリプティングの詳細対策
- [インジェクション](./03-injection.md) — インジェクション攻撃の深掘り
- [認証脆弱性](./04-auth-vulnerabilities.md) — 認証に関する脆弱性の詳細
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) — 通信暗号化の基盤技術

---

## 1. OWASP Top 10 概観

OWASP（Open Worldwide Application Security Project）が定期的に発表するWebアプリケーション脆弱性のランキング。2003年から始まり、2021年版が最新である。データドリブンなアプローチで、数十万のアプリケーションから収集されたインシデントデータに基づいている。

### OWASP Top 10 の歴史と変遷

```
2017年版からの変更点:

  2017                          2021
  ----                          ----
  A1 インジェクション          → A3 に降格
  A2 認証の不備                → A7 に統合
  A3 機密データの露出          → A2 暗号化の失敗
  A4 XML外部実体参照           → A3 インジェクションに統合
  A5 アクセス制御の不備        → A1 に昇格
  A6 セキュリティの設定ミス    → A5
  A7 XSS                      → A3 インジェクションに統合
  A8 安全でないデシリアライゼーション → A8 整合性の不具合
  A9 既知の脆弱性を持つコンポーネント → A6
  A10 不十分なログとモニタリング → A9

  新規追加 (2021):
  A04 安全でない設計 ← NEW
  A08 ソフトウェアとデータの整合性の不具合 ← 再編
  A10 SSRF ← NEW
```

### カテゴリ一覧と深刻度

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

### 各カテゴリの CWE マッピング

```
+------------------------------------------------------------------+
|  OWASP カテゴリと主要 CWE の対応                                    |
|------------------------------------------------------------------|
|  A01 アクセス制御の不備                                             |
|    +-- CWE-200: 情報漏洩                                          |
|    +-- CWE-284: 不適切なアクセス制御                                |
|    +-- CWE-285: 不適切な認可                                       |
|    +-- CWE-639: IDOR (安全でない直接オブジェクト参照)               |
|                                                                    |
|  A02 暗号化の失敗                                                   |
|    +-- CWE-259: ハードコードされたパスワード                        |
|    +-- CWE-327: 壊れた/危険な暗号アルゴリズム                       |
|    +-- CWE-328: 弱いハッシュ                                       |
|    +-- CWE-916: 不十分な計算量のパスワードハッシュ                   |
|                                                                    |
|  A03 インジェクション                                               |
|    +-- CWE-20: 不適切な入力検証                                    |
|    +-- CWE-79: XSS                                                |
|    +-- CWE-89: SQL インジェクション                                |
|    +-- CWE-78: OS コマンドインジェクション                          |
+------------------------------------------------------------------+
```

---

## 2. A01: アクセス制御の不備（Broken Access Control）

認可されていないリソースへのアクセスを許してしまう脆弱性。2021年版で A5 から A1 に昇格し、最も深刻なカテゴリとなった。調査対象アプリの 94% でアクセス制御の問題が検出されている。

### 攻撃手法の分類

```
+------------------------------------------------------------------+
|  アクセス制御の攻撃手法                                              |
|------------------------------------------------------------------|
|                                                                    |
|  [水平権限昇格 (Horizontal)]                                       |
|  +-- IDOR: /api/users/123 → /api/users/456 で他人のデータ参照      |
|  +-- パラメータ改竄: user_id=me → user_id=admin                    |
|                                                                    |
|  [垂直権限昇格 (Vertical)]                                         |
|  +-- URL操作: /user/dashboard → /admin/dashboard                   |
|  +-- APIメソッド: GET (許可) → DELETE (本来は禁止)                  |
|  +-- 強制ブラウジング: 非公開URLの推測アクセス                       |
|                                                                    |
|  [コンテキスト依存の不備]                                           |
|  +-- マルチテナント: テナントAのユーザがテナントBのデータにアクセス   |
|  +-- ステート操作: ワークフローのステップをスキップ                   |
|  +-- メタデータ操作: JWTのroleクレームを改竄                        |
+------------------------------------------------------------------+
```

### IDOR（安全でない直接オブジェクト参照）の詳細

IDOR は最も頻繁に発見されるアクセス制御の脆弱性である。攻撃者がURLパラメータやリクエストボディのIDを改竄するだけで、他ユーザのデータにアクセスできてしまう。

```python
# コード例1: 安全なアクセス制御の実装
from functools import wraps
from flask import Flask, request, abort, g
import uuid

app = Flask(__name__)

# ===== 悪い例: IDORの脆弱性 =====
@app.route("/api/orders/<int:order_id>")
def get_order_bad(order_id):
    # 誰でも任意のorder_idにアクセスできてしまう
    order = db.query("SELECT * FROM orders WHERE id = ?", order_id)
    return jsonify(order)

# ===== 良い例: オーナーシップチェック付き =====
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


# ===== ロールベースアクセス制御 (RBAC) =====
class Permission:
    """権限定義"""
    READ_OWN = "read:own"
    READ_ALL = "read:all"
    WRITE_OWN = "write:own"
    WRITE_ALL = "write:all"
    ADMIN = "admin"

ROLE_PERMISSIONS = {
    "user":    [Permission.READ_OWN, Permission.WRITE_OWN],
    "manager": [Permission.READ_OWN, Permission.WRITE_OWN, Permission.READ_ALL],
    "admin":   [Permission.READ_OWN, Permission.WRITE_OWN,
                Permission.READ_ALL, Permission.WRITE_ALL, Permission.ADMIN],
}

def require_permission(permission):
    """パーミッションベースの認可デコレータ"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not g.current_user:
                abort(401)
            user_permissions = ROLE_PERMISSIONS.get(g.current_user.role, [])
            if permission not in user_permissions:
                # 監査ログを記録
                audit_log.warning(
                    f"Access denied: user={g.current_user.id}, "
                    f"permission={permission}, path={request.path}"
                )
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator


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


# ===== ABAC (属性ベースアクセス制御) の実装 =====
class ABACPolicy:
    """属性ベースアクセス制御"""

    def __init__(self):
        self.policies = []

    def add_policy(self, resource_type, action, condition_fn):
        self.policies.append({
            "resource_type": resource_type,
            "action": action,
            "condition": condition_fn,
        })

    def check(self, user, resource_type, action, resource=None):
        for policy in self.policies:
            if (policy["resource_type"] == resource_type and
                policy["action"] == action):
                if policy["condition"](user, resource):
                    return True
        return False

abac = ABACPolicy()

# ポリシー定義: ドキュメントの所有者のみ編集可能
abac.add_policy(
    "document", "edit",
    lambda user, doc: doc.owner_id == user.id
)

# ポリシー定義: 同じ部署のマネージャーは閲覧可能
abac.add_policy(
    "document", "view",
    lambda user, doc: (
        user.department == doc.department and
        user.role in ["manager", "admin"]
    )
)

# ポリシー定義: 公開ドキュメントは誰でも閲覧可能
abac.add_policy(
    "document", "view",
    lambda user, doc: doc.visibility == "public"
)
```

### UUIDによるIDOR緩和

```python
# コード例2: 推測困難なIDの使用
import uuid

class Order:
    def __init__(self, user_id, items):
        # 連番IDではなくUUIDv4を使用
        self.id = str(uuid.uuid4())  # "a3b8f9c2-1d4e-4a6b-8c3d-9e7f0a1b2c3d"
        self.user_id = user_id
        self.items = items

# 注意: UUIDの使用はIDORの根本対策ではない
# 推測を困難にするが、認可チェックは依然として必須

@app.route("/api/orders/<order_id>")
@login_required
def get_order(order_id):
    # UUIDでもオーナーシップチェックは必須
    order = Order.query.filter_by(
        id=order_id,
        user_id=g.current_user.id
    ).first_or_404()
    return jsonify(order.to_dict())
```

---

## 3. A02: 暗号化の失敗（Cryptographic Failures）

機密データの暗号化が不十分、または暗号化の設計が不適切な脆弱性。2017年版では「機密データの露出」と呼ばれていたが、根本原因である暗号化の失敗に名称が変更された。

### 暗号化の失敗パターン

```
+------------------------------------------------------------------+
|  暗号化の失敗パターン                                                |
|------------------------------------------------------------------|
|                                                                    |
|  [転送中のデータ]                                                   |
|  +-- HTTP（非HTTPS）での機密データ送信                              |
|  +-- TLS 1.0/1.1 の使用（既知の脆弱性あり）                        |
|  +-- 弱い暗号スイートの許可（RC4, DES, 3DES）                      |
|  +-- 証明書検証の無効化                                             |
|                                                                    |
|  [保存データ]                                                       |
|  +-- 平文でのパスワード保存                                         |
|  +-- MD5/SHA-1 でのパスワードハッシュ                               |
|  +-- ソルトなしのハッシュ                                           |
|  +-- データベース暗号化の未実施                                     |
|  +-- バックアップの非暗号化                                         |
|                                                                    |
|  [暗号アルゴリズムの誤用]                                           |
|  +-- ECB モードの使用（パターン漏洩）                               |
|  +-- 固定IV/ノンスの使用                                           |
|  +-- 自作暗号の使用                                                 |
|  +-- 不十分な鍵長（RSA 1024bit, AES 128bit未満）                   |
+------------------------------------------------------------------+
```

### データ分類と暗号化要件

| データ分類 | 例 | 転送時暗号化 | 保存時暗号化 | アクセス制御 | 保持期間 |
|-----------|-----|------------|------------|------------|---------|
| 公開 | プレスリリース | 推奨 | 不要 | 不要 | 無期限 |
| 内部 | 社内文書 | 必須 | 推奨 | ロールベース | 規定に従う |
| 機密 | 顧客情報 | 必須(TLS 1.2+) | 必須(AES-256) | 最小権限 | 法令に従う |
| 極秘 | クレジットカード | 必須(TLS 1.3) | 必須(HSM管理鍵) | Need-to-know | PCI DSS準拠 |

```python
# コード例3: 適切な暗号化の実装
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import bcrypt
import os
import base64
import hmac
import hashlib

class SecureCrypto:
    """安全な暗号化ユーティリティ"""

    @staticmethod
    def hash_password(password: str) -> str:
        """パスワードのハッシュ化（bcrypt使用）

        bcryptが推奨される理由:
        1. ソルトが自動生成・付与される
        2. コストファクター（rounds）で計算時間を調整可能
        3. GPU/ASIC による並列攻撃に対して耐性がある
        4. タイミング攻撃に対して一定時間比較を行う
        """
        # 悪い例: MD5やSHA-256の直接使用
        # hashlib.md5(password.encode()).hexdigest()  # NG!
        # hashlib.sha256(password.encode()).hexdigest()  # NG!

        # 良い例: bcryptによるソルト付きハッシュ
        # rounds=12 は2025年時点で推奨される最小値
        # サーバの性能に応じて13-14まで上げることが望ましい
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """パスワードの検証（定数時間比較）"""
        return bcrypt.checkpw(password.encode(), hashed.encode())

    @staticmethod
    def hash_password_argon2(password: str) -> str:
        """Argon2idによるパスワードハッシュ（OWASP推奨）

        Argon2id は OWASP が最も推奨するパスワードハッシュアルゴリズム。
        メモリハード関数であり、GPU/ASIC攻撃に対してbcryptより強い耐性を持つ。
        """
        from argon2 import PasswordHasher
        # OWASP推奨パラメータ (2024):
        # memory_cost=65536 (64MB), time_cost=3, parallelism=4
        ph = PasswordHasher(
            memory_cost=65536,   # 64MB
            time_cost=3,         # 3回の反復
            parallelism=4,       # 4スレッド
            hash_len=32,         # 256bit出力
            salt_len=16,         # 128bitソルト
        )
        return ph.hash(password)

    @staticmethod
    def encrypt_sensitive_data(plaintext: str, master_key: bytes) -> str:
        """機密データの暗号化（AES-256-GCM）

        GCMモード（Galois/Counter Mode）を使用する理由:
        1. 認証付き暗号化（AEAD）: 暗号化と改竄検知を同時に実現
        2. 並列処理可能: CBCモードより高速
        3. IVの再利用が致命的: 12バイトのランダムノンスを毎回生成
        """
        # ノンスは毎回一意に生成（12バイトが推奨）
        nonce = os.urandom(12)
        key = AESGCM.generate_key(bit_length=256) if not master_key else master_key[:32]

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)

        # ノンス + 暗号文を結合して返す
        return base64.b64encode(nonce + ciphertext).decode()

    @staticmethod
    def encrypt_with_kdf(plaintext: str, master_key: bytes) -> str:
        """KDFを使用した暗号化（鍵導出関数付き）"""
        # PBKDF2でマスターキーから暗号鍵を導出
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP推奨: 最低600,000（PBKDF2-SHA256の場合）
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key))
        f = Fernet(key)
        encrypted = f.encrypt(plaintext.encode())
        # ソルトと暗号文を結合して返す
        return base64.b64encode(salt + encrypted).decode()

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """定数時間文字列比較（タイミング攻撃対策）

        通常の == 比較は最初の不一致文字で即座に返すため、
        応答時間の差から正しい文字列を推測できてしまう。
        """
        return hmac.compare_digest(a.encode(), b.encode())


# パスワードハッシュアルゴリズムの比較表
"""
+------------------------------------------------------------------+
|  パスワードハッシュアルゴリズム比較                                    |
|------------------------------------------------------------------|
|  アルゴリズム | GPU耐性 | メモリ使用 | OWASP推奨 | 備考             |
|  ------------|---------|----------|----------|------------------|
|  MD5          | 最弱    | 極小     | 非推奨   | 衝突攻撃が容易     |
|  SHA-256      | 弱      | 極小     | 非推奨   | 高速すぎる         |
|  bcrypt       | 中      | 4KB固定  | 推奨     | 72バイト制限あり    |
|  scrypt       | 強      | 可変     | 推奨     | パラメータ設定が複雑 |
|  Argon2id     | 最強    | 可変     | 最推奨   | PHC勝者(2015)      |
+------------------------------------------------------------------+
"""
```

---

## 4. A03: インジェクション（Injection）

ユーザー入力がコード・クエリの一部として解釈されてしまう脆弱性。SQLインジェクション、XSS、コマンドインジェクション、LDAPインジェクション等が含まれる。

### インジェクション攻撃の内部メカニズム

```
SQL インジェクションの仕組み:

正常なクエリ:
  入力: "alice"
  クエリ: SELECT * FROM users WHERE name = 'alice'
  結果: aliceのデータのみ返る

攻撃クエリ:
  入力: "' OR '1'='1' --"
  クエリ: SELECT * FROM users WHERE name = '' OR '1'='1' --'
                                        ~~~~~~~~~~~~~~~
                                        常にTRUE → 全件返る

UNION攻撃:
  入力: "' UNION SELECT username, password FROM admin_users --"
  クエリ: SELECT name, email FROM users WHERE name = ''
          UNION SELECT username, password FROM admin_users --'
  結果: 管理者のユーザ名とパスワードが漏洩

二次インジェクション:
  Step 1: ユーザ登録時に "admin'--" という名前を登録（エスケープされて保存）
  Step 2: パスワード変更時にDBから名前を取得しクエリに使用
          UPDATE users SET password='new' WHERE name='admin'--'
          → admin のパスワードが変更されてしまう
```

```python
# コード例4: 包括的なインジェクション対策
import sqlite3
import subprocess
import shlex
import re

# ===== SQLインジェクション対策 =====

# 悪い例: 文字列連結によるSQL構築
def search_users_bad(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)  # ' OR '1'='1 で全件取得可能

# 良い例1: パラメータ化クエリ
def search_users_good(username):
    query = "SELECT * FROM users WHERE name = ?"
    return db.execute(query, (username,))

# 良い例2: ORMの使用（SQLAlchemy）
from sqlalchemy import select
from models import User

def search_users_orm(session, username):
    stmt = select(User).where(User.name == username)
    return session.execute(stmt).scalars().all()

# 良い例3: ストアドプロシージャの使用
def search_users_stored_proc(username):
    return db.execute("CALL search_users(?)", (username,))


# ===== OSコマンドインジェクション対策 =====

# 悪い例: os.system / shell=True
def ping_host_bad(hostname):
    os.system(f"ping -c 1 {hostname}")  # ; rm -rf / が挿入可能

# 良い例: subprocessでシェルを介さない
def ping_host_good(hostname):
    # ホワイトリスト検証
    if not re.match(r'^[a-zA-Z0-9\.\-]+$', hostname):
        raise ValueError("Invalid hostname")
    # shell=False（デフォルト）で実行
    result = subprocess.run(
        ["ping", "-c", "1", hostname],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout


# ===== テンプレートインジェクション (SSTI) 対策 =====
from jinja2 import Environment, select_autoescape, sandbox

# 悪い例: ユーザ入力をテンプレートとして評価
def render_bad(user_input):
    from jinja2 import Template
    return Template(user_input).render()  # {{7*7}} → 49, RCE可能

# 良い例: サンドボックス環境を使用
def render_good(template_name, context):
    env = sandbox.SandboxedEnvironment(
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template(template_name)
    return template.render(**context)


# ===== LDAPインジェクション対策 =====
import ldap3

# 悪い例: 文字列連結
def search_ldap_bad(username):
    filter_str = f"(uid={username})"  # *)(uid=*))(|(uid=* で全件取得
    conn.search("dc=example,dc=com", filter_str)

# 良い例: エスケープ関数の使用
def search_ldap_good(username):
    from ldap3.utils.conv import escape_filter_chars
    safe_username = escape_filter_chars(username)
    filter_str = f"(uid={safe_username})"
    conn.search("dc=example,dc=com", filter_str)
```

### XSS（クロスサイトスクリプティング）の分類

```
+------------------------------------------------------------------+
|  XSS の3分類                                                       |
|------------------------------------------------------------------|
|                                                                    |
|  [Reflected XSS (反射型)]                                          |
|  攻撃者 → 悪意のあるURL → 被害者がクリック                          |
|  → サーバがユーザ入力をレスポンスに含める                             |
|  → ブラウザでスクリプト実行                                         |
|  例: https://site.com/search?q=<script>alert(1)</script>           |
|                                                                    |
|  [Stored XSS (格納型)]                                              |
|  攻撃者 → 掲示板にスクリプトを投稿 → DBに保存                       |
|  → 他のユーザが閲覧時にスクリプト実行                                |
|  例: コメント欄に <img src=x onerror=steal(cookie)>                |
|                                                                    |
|  [DOM-based XSS (DOM型)]                                           |
|  攻撃者 → URLフラグメントにスクリプト                               |
|  → クライアントサイドJSがDOMに挿入                                  |
|  例: https://site.com/page#<img src=x onerror=alert(1)>           |
|  document.getElementById('x').innerHTML = location.hash            |
+------------------------------------------------------------------+
```

```python
# コード例5: XSS対策の実装
from markupsafe import escape
from flask import Flask, Markup

app = Flask(__name__)

# 悪い例: ユーザ入力を直接HTMLに埋め込み
@app.route("/profile")
def profile_bad():
    name = request.args.get("name")
    return f"<h1>Welcome, {name}</h1>"  # XSS脆弱

# 良い例1: エスケープ
@app.route("/profile")
def profile_good():
    name = escape(request.args.get("name", ""))
    return f"<h1>Welcome, {name}</h1>"

# 良い例2: テンプレートエンジンの自動エスケープ
# templates/profile.html: <h1>Welcome, {{ name }}</h1>
# Jinja2はデフォルトで自動エスケープ

# CSP (Content Security Policy) の設定
@app.after_request
def set_csp(response):
    """XSSの影響を最小化するCSPヘッダー"""
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'nonce-{nonce}'; "  # nonceベースの許可
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    return response
```

---

## 5. A04: 安全でない設計（Insecure Design）

設計段階でのセキュリティ考慮の欠如に起因する脆弱性。コーディングレベルの対策では解決できない、アーキテクチャ上の問題を指す。

### 脅威モデリング（STRIDE）

```
+------------------------------------------------------------------+
|  STRIDE 脅威モデリングフレームワーク                                  |
|------------------------------------------------------------------|
|                                                                    |
|  S - Spoofing (なりすまし)                                         |
|    → 対策: 認証、証明書、MFA                                        |
|                                                                    |
|  T - Tampering (改竄)                                              |
|    → 対策: 完全性チェック、MAC、デジタル署名                         |
|                                                                    |
|  R - Repudiation (否認)                                            |
|    → 対策: 監査ログ、タイムスタンプ、デジタル署名                    |
|                                                                    |
|  I - Information Disclosure (情報漏洩)                              |
|    → 対策: 暗号化、アクセス制御、最小権限                            |
|                                                                    |
|  D - Denial of Service (サービス拒否)                              |
|    → 対策: レートリミット、リソース制限、冗長構成                    |
|                                                                    |
|  E - Elevation of Privilege (権限昇格)                             |
|    → 対策: 最小権限、入力検証、サンドボックス                        |
+------------------------------------------------------------------+
```

### セキュアな設計パターン

```python
# コード例6: ビジネスロジックのセキュリティ設計
from datetime import datetime, timedelta
from collections import defaultdict

class SecurePasswordReset:
    """安全なパスワードリセットの設計

    安全でない設計の例:
    - リセットリンクが予測可能（連番ID）
    - リセットトークンに有効期限がない
    - レートリミットがない（列挙攻撃可能）
    - 「ユーザが存在しません」というエラーメッセージ（情報漏洩）
    """

    def __init__(self):
        self.reset_tokens = {}
        self.attempt_counts = defaultdict(list)
        self.TOKEN_EXPIRY = timedelta(minutes=15)  # 短い有効期限
        self.MAX_ATTEMPTS = 3  # 1時間あたり3回まで

    def request_reset(self, email: str) -> dict:
        # レートリミットチェック
        now = datetime.utcnow()
        recent = [t for t in self.attempt_counts[email]
                  if t > now - timedelta(hours=1)]
        self.attempt_counts[email] = recent

        if len(recent) >= self.MAX_ATTEMPTS:
            # 同じレスポンスを返す（情報漏洩防止）
            return {"message": "If the email exists, a reset link has been sent."}

        self.attempt_counts[email].append(now)

        # ユーザの存在有無に関わらず同じレスポンス
        user = find_user_by_email(email)
        if user:
            token = secrets.token_urlsafe(32)  # 256bit のランダムトークン
            self.reset_tokens[token] = {
                "user_id": user.id,
                "expires_at": now + self.TOKEN_EXPIRY,
                "used": False,
            }
            send_reset_email(email, token)

        # 攻撃者にユーザの存在を知らせない
        return {"message": "If the email exists, a reset link has been sent."}

    def verify_reset(self, token: str, new_password: str) -> bool:
        data = self.reset_tokens.get(token)
        if not data:
            return False
        if data["used"]:
            return False  # ワンタイム使用
        if datetime.utcnow() > data["expires_at"]:
            del self.reset_tokens[token]
            return False

        # トークンを使用済みにマーク
        data["used"] = True
        update_password(data["user_id"], new_password)

        # 他のセッションを無効化
        invalidate_all_sessions(data["user_id"])

        return True
```

---

## 6. A05: セキュリティの設定ミス（Security Misconfiguration）

デフォルト設定の変更忘れ、不要な機能の有効化、過剰な権限付与など、設定に起因するセキュリティ問題。

### 典型的な設定ミスと対策

```python
# コード例7: セキュリティヘッダーの包括的な設定
from flask import Flask, Response

app = Flask(__name__)

# 本番環境ではデバッグモードを無効化
app.config["DEBUG"] = False
app.config["TESTING"] = False

# セッションの安全な設定
app.config["SESSION_COOKIE_SECURE"] = True       # HTTPS必須
app.config["SESSION_COOKIE_HTTPONLY"] = True      # JavaScript からアクセス不可
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"    # CSRF対策
app.config["PERMANENT_SESSION_LIFETIME"] = 1800   # 30分でタイムアウト

@app.after_request
def set_security_headers(response: Response) -> Response:
    """全レスポンスにセキュリティヘッダーを付与する"""
    # XSSフィルタ（CSPに委任）
    response.headers["X-XSS-Protection"] = "0"

    # MIMEタイプスニッフィング防止
    response.headers["X-Content-Type-Options"] = "nosniff"

    # クリックジャッキング防止
    response.headers["X-Frame-Options"] = "DENY"

    # Content Security Policy（XSS緩和の最重要ヘッダー）
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' https://api.example.com; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "upgrade-insecure-requests"
    )

    # HTTPS強制（HSTS）
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains; preload"
    )

    # リファラー制御
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # ブラウザ機能の制限
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), "
        "payment=(), usb=(), magnetometer=()"
    )

    # キャッシュ制御（機密データ）
    if "api" in request.path:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

    return response


# エラーハンドラ（情報漏洩防止）
@app.errorhandler(500)
def internal_error(error):
    """本番環境ではスタックトレースを返さない"""
    app.logger.error(f"Internal error: {error}", exc_info=True)
    return {"error": "Internal server error"}, 500

@app.errorhandler(404)
def not_found(error):
    return {"error": "Resource not found"}, 404
```

### ハードニングチェックリスト

```
+------------------------------------------------------------------+
|  サーバハードニングチェックリスト                                     |
|------------------------------------------------------------------|
|  [ ] デフォルトアカウント/パスワードの変更                           |
|  [ ] 不要なポート・サービスの無効化                                  |
|  [ ] ディレクトリリスティングの無効化                                 |
|  [ ] サーババージョン情報の非公開化                                   |
|  [ ] デバッグモードの無効化                                          |
|  [ ] スタックトレースの非公開化                                       |
|  [ ] CORSの適切な設定                                               |
|  [ ] HTTPメソッドの制限（OPTIONS, TRACE等の無効化）                  |
|  [ ] TLS 1.2+ の強制                                               |
|  [ ] セキュリティヘッダーの設定                                       |
|  [ ] Cookie属性の適切な設定 (Secure, HttpOnly, SameSite)            |
|  [ ] ファイルアップロードの制限と検証                                 |
|  [ ] エラーページのカスタマイズ                                       |
+------------------------------------------------------------------+
```

---

## 7. A06: 脆弱で古いコンポーネント（Vulnerable and Outdated Components）

既知の脆弱性を持つライブラリ、フレームワーク、その他のソフトウェアコンポーネントの使用。サプライチェーン攻撃の入り口となる。

### 依存関係の管理

```bash
# コード例8: 依存関係のセキュリティチェック

# Python: pip-audit
pip install pip-audit
pip-audit

# Python: Safety
pip install safety
safety check

# Node.js: npm audit
npm audit
npm audit fix

# Go: govulncheck
go install golang.org/x/vuln/cmd/govulncheck@latest
govulncheck ./...

# Rust: cargo-audit
cargo install cargo-audit
cargo audit

# マルチ言語対応: Trivy
trivy fs --scanners vuln .
```

### SCA（Software Composition Analysis）ツールの比較

| ツール | 対応言語 | コスト | 特徴 |
|--------|---------|--------|------|
| Dependabot | 多数 | 無料 (GitHub) | 自動PR作成 |
| Snyk | 多数 | フリーミアム | 修正提案が詳細 |
| OWASP Dependency-Check | Java, .NET | 無料 | NVDベース |
| pip-audit | Python | 無料 | OSV/PyPIデータベース |
| npm audit | Node.js | 無料 | npm内蔵 |
| Trivy | 多数 | 無料 | コンテナも対応 |

```yaml
# GitHub Actions: Dependabot設定
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "security"
    reviewers:
      - "security-team"
```

---

## 8. A07: 識別と認証の失敗（Identification and Authentication Failures）

### セキュアな認証の実装

```python
# コード例9: セッション管理のベストプラクティス
import secrets
from datetime import datetime, timedelta
from flask import Flask, session, request

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # 256bit

class SecureSessionManager:
    """安全なセッション管理"""

    def __init__(self):
        self.sessions = {}
        self.MAX_SESSIONS_PER_USER = 5

    def create_session(self, user_id: str, ip: str, user_agent: str) -> str:
        """セッション作成"""
        # 既存セッション数チェック
        user_sessions = [
            s for s in self.sessions.values()
            if s["user_id"] == user_id and not s["expired"]
        ]
        if len(user_sessions) >= self.MAX_SESSIONS_PER_USER:
            # 最も古いセッションを無効化
            oldest = min(user_sessions, key=lambda s: s["created_at"])
            oldest["expired"] = True

        session_id = secrets.token_urlsafe(32)  # 256bitのランダムID
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ip": ip,
            "user_agent": user_agent,
            "expired": False,
        }
        return session_id

    def validate_session(self, session_id: str, ip: str) -> bool:
        """セッション検証"""
        data = self.sessions.get(session_id)
        if not data or data["expired"]:
            return False

        # 絶対タイムアウト: 作成から24時間
        if datetime.utcnow() - data["created_at"] > timedelta(hours=24):
            data["expired"] = True
            return False

        # アイドルタイムアウト: 最終操作から30分
        if datetime.utcnow() - data["last_activity"] > timedelta(minutes=30):
            data["expired"] = True
            return False

        # IP変更の検知（セッションハイジャック対策）
        if data["ip"] != ip:
            audit_log.warning(
                f"Session IP mismatch: session={session_id}, "
                f"original={data['ip']}, current={ip}"
            )
            # 厳密モード: セッション無効化
            # data["expired"] = True
            # return False

        data["last_activity"] = datetime.utcnow()
        return True

    def destroy_session(self, session_id: str):
        """セッション破棄（ログアウト時）"""
        if session_id in self.sessions:
            self.sessions[session_id]["expired"] = True

    def destroy_all_user_sessions(self, user_id: str):
        """全セッション破棄（パスワード変更時）"""
        for session_data in self.sessions.values():
            if session_data["user_id"] == user_id:
                session_data["expired"] = True
```

### 認証方式の比較

| 方式 | セキュリティ | UX | 実装難易度 | 適用場面 |
|------|------------|-----|-----------|---------|
| パスワードのみ | 低 | 簡単 | 低 | 非推奨 |
| パスワード+TOTP | 中 | やや面倒 | 中 | 一般的 |
| パスワード+WebAuthn | 高 | 初回のみ面倒 | 中 | 推奨 |
| パスキー (Passkeys) | 高 | 簡単 | 中 | 最推奨 |
| SSO (SAML/OIDC) | 高 | 簡単 | 高 | エンタープライズ |

---

## 9. A08-A10: 詳細解説

### A08: ソフトウェアとデータの整合性の不具合

CI/CDパイプラインの侵害、安全でないデシリアライゼーション、ソフトウェアの更新検証の欠如。

```python
# コード例10: 安全なデシリアライゼーション
import json
import hmac
import hashlib

# 悪い例: pickle の直接使用（任意コード実行の危険）
import pickle
def load_bad(data):
    return pickle.loads(data)  # 絶対にNG! RCE可能

# 良い例: JSONによるシリアライゼーション + 署名検証
class SecureSerializer:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def serialize(self, data: dict) -> str:
        """データをシリアライズして署名"""
        payload = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            self.secret_key, payload.encode(), hashlib.sha256
        ).hexdigest()
        return json.dumps({"payload": payload, "signature": signature})

    def deserialize(self, raw: str) -> dict:
        """署名を検証してデシリアライズ"""
        container = json.loads(raw)
        expected_sig = hmac.new(
            self.secret_key,
            container["payload"].encode(),
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, container["signature"]):
            raise ValueError("Signature verification failed")
        return json.loads(container["payload"])
```

### A09: セキュリティログとモニタリングの不備

```python
# コード例11: 構造化セキュリティログ
import logging
import json
from datetime import datetime

class SecurityLogger:
    """セキュリティイベント専用ロガー"""

    def __init__(self):
        self.logger = logging.getLogger("security")
        handler = logging.FileHandler("/var/log/app/security.json")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: dict):
        """構造化されたセキュリティイベントを記録"""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "details": details,
            "source_ip": request.remote_addr if request else None,
            "user_agent": request.headers.get("User-Agent") if request else None,
        }
        self.logger.info(json.dumps(event))

    def log_auth_failure(self, username: str, reason: str):
        self.log_event("AUTH_FAILURE", {
            "username": username,
            "reason": reason,
        })

    def log_access_denied(self, user_id: str, resource: str):
        self.log_event("ACCESS_DENIED", {
            "user_id": user_id,
            "resource": resource,
        })

    def log_suspicious_activity(self, user_id: str, activity: str):
        self.log_event("SUSPICIOUS", {
            "user_id": user_id,
            "activity": activity,
            "severity": "HIGH",
        })

# ログすべきイベント一覧:
# - 認証の成功/失敗
# - アクセス制御の拒否
# - 入力検証の失敗
# - セッションの作成/破棄
# - 権限の変更
# - 管理操作（ユーザ作成、設定変更）
# - 高額取引/重要操作
```

### A10: SSRF（サーバーサイドリクエストフォージェリ）

```python
# コード例12: SSRF対策の完全な実装
import ipaddress
import socket
from urllib.parse import urlparse
import re

class SSRFProtection:
    """SSRF攻撃を防止するURL検証

    SSRFの攻撃シナリオ:
    1. 内部メタデータAPIへのアクセス
       (AWS: http://169.254.169.254/latest/meta-data/)
    2. 内部サービスへのポートスキャン
    3. クラウドプロバイダの認証情報の窃取
    4. 内部ネットワークのファイル読み取り (file://)
    """

    BLOCKED_NETWORKS = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("169.254.0.0/16"),  # リンクローカル/メタデータ
        ipaddress.ip_network("100.64.0.0/10"),   # CGNATレンジ
        ipaddress.ip_network("::1/128"),          # IPv6ループバック
        ipaddress.ip_network("fc00::/7"),         # IPv6 ULA
        ipaddress.ip_network("fe80::/10"),        # IPv6 リンクローカル
    ]

    ALLOWED_SCHEMES = {"http", "https"}
    ALLOWED_PORTS = {80, 443, 8080, 8443}

    @classmethod
    def validate_url(cls, url: str) -> bool:
        """外部アクセスに使用するURLを検証する"""
        parsed = urlparse(url)

        # スキームチェック
        if parsed.scheme not in cls.ALLOWED_SCHEMES:
            return False

        # ポートチェック
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in cls.ALLOWED_PORTS:
            return False

        # ホスト名の検証
        hostname = parsed.hostname
        if not hostname:
            return False

        # DNS rebinding 対策: ドット区切りの数値IPを直接チェック
        try:
            ip = ipaddress.ip_address(hostname)
            return not cls._is_blocked(ip)
        except ValueError:
            pass  # ホスト名の場合はDNS解決が必要

        # ホスト解決とプライベートIP検出
        try:
            # 全てのアドレスを解決（CNAMEチェーン対策）
            addrinfos = socket.getaddrinfo(hostname, port)
            for family, _, _, _, sockaddr in addrinfos:
                ip = ipaddress.ip_address(sockaddr[0])
                if cls._is_blocked(ip):
                    return False
        except (socket.gaierror, ValueError):
            return False

        return True

    @classmethod
    def _is_blocked(cls, ip: ipaddress.IPv4Address) -> bool:
        for network in cls.BLOCKED_NETWORKS:
            if ip in network:
                return True
        return False

    @classmethod
    def safe_fetch(cls, url: str, timeout: int = 10) -> bytes:
        """安全な外部URLフェッチ"""
        if not cls.validate_url(url):
            raise ValueError(f"Blocked URL: {url}")

        import requests
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=False,  # リダイレクトを手動で処理
            headers={"User-Agent": "MyApp/1.0"},
        )

        # リダイレクト先も検証
        if response.is_redirect:
            redirect_url = response.headers.get("Location")
            if redirect_url and cls.validate_url(redirect_url):
                return cls.safe_fetch(redirect_url, timeout)
            raise ValueError(f"Blocked redirect: {redirect_url}")

        return response.content
```

---

## 10. 各脆弱性の対策比較

| 脆弱性 | 主要対策 | ツール | 検出フェーズ | CWE |
|--------|---------|-------|-------------|-----|
| A01 アクセス制御 | RBAC、オーナーシップチェック | Burp Suite | DAST | CWE-284 |
| A02 暗号化失敗 | TLS 1.3、AES-GCM、bcrypt | testssl.sh | 設計レビュー | CWE-327 |
| A03 インジェクション | パラメータ化クエリ、ORM | SQLMap、SAST | SAST/DAST | CWE-89 |
| A04 安全でない設計 | 脅威モデリング、セキュリティ要件 | - | 設計レビュー | CWE-501 |
| A05 設定ミス | ハードニング、IaC | ScoutSuite | 構成監査 | CWE-16 |
| A06 古いコンポーネント | SCA、自動更新 | Dependabot | SCA | CWE-1104 |
| A07 認証の失敗 | MFA、レートリミット | Hydra | ペネトレーションテスト | CWE-287 |
| A08 整合性不具合 | 署名検証、SRI | Sigstore | CI/CD | CWE-502 |
| A09 ログ不備 | SIEM、監査ログ | ELK Stack | 運用監視 | CWE-778 |
| A10 SSRF | URL検証、ネットワーク分離 | Burp Suite | DAST | CWE-918 |

### 対策の実装優先度マトリクス

```
+------------------------------------------------------------------+
|  影響度 vs 対策コスト マトリクス                                      |
|------------------------------------------------------------------|
|                                                                    |
|  影響: 大 |  A01 アクセス制御  |  A04 安全な設計    |               |
|           |  A03 インジェクション|  (設計段階で対処)  |               |
|           |  (即座に対策すべき)  |                    |               |
|  ---------|--------------------|--------------------|               |
|  影響: 中 |  A02 暗号化        |  A06 コンポーネント |               |
|           |  A05 設定ミス      |  A08 整合性        |               |
|           |  A07 認証          |                    |               |
|  ---------|--------------------|--------------------|               |
|  影響: 小 |  A10 SSRF         |  A09 ログ          |               |
|           |  (ネットワーク分離) |  (運用改善)        |               |
|           |                    |                    |               |
|           |  対策コスト: 低     |  対策コスト: 高     |               |
+------------------------------------------------------------------+
```

---

## 11. セキュリティテスト手法

### SAST/DAST/IAST の比較

| 手法 | 正式名称 | テスト対象 | タイミング | 長所 | 短所 |
|------|---------|-----------|-----------|------|------|
| SAST | 静的アプリケーションセキュリティテスト | ソースコード | 開発時 | 早期発見、高カバレッジ | 偽陽性が多い |
| DAST | 動的アプリケーションセキュリティテスト | 実行中のアプリ | テスト時 | 実際の攻撃をシミュレート | カバレッジが低い |
| IAST | 対話型アプリケーションセキュリティテスト | 実行中のアプリ+コード | テスト時 | 低偽陽性、高精度 | エージェント必要 |
| SCA | ソフトウェア構成分析 | 依存関係 | ビルド時 | 既知脆弱性の検出 | 0dayは検出不可 |

### セキュリティテストの自動化パイプライン

```yaml
# GitHub Actions: 包括的なセキュリティテスト
name: Security Pipeline
on:
  pull_request:
    branches: [main]

jobs:
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Semgrep SAST
        uses: returntocorp/semgrep-action@v1
        with:
          config: "p/owasp-top-ten"

  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: pip-audit
        run: |
          pip install pip-audit
          pip-audit -r requirements.txt

  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified
```

---

## 12. エッジケース分析

### エッジケース1: JWTの`alg`ヘッダー操作

攻撃者がJWTの`alg`フィールドを`none`に変更し、署名検証をバイパスする。または`RS256`（非対称）を`HS256`（対称）に変更し、公開鍵をHMAC鍵として使用する。

```python
# 対策: アルゴリズムを明示的に指定し、ヘッダーの値を信用しない
import jwt

# NG: ヘッダーのalgを信用
payload = jwt.decode(token, key, algorithms=jwt.get_unverified_header(token)["alg"])

# OK: 許可するアルゴリズムを固定
payload = jwt.decode(token, public_key, algorithms=["RS256"])
```

### エッジケース2: Unicode正規化によるアクセス制御バイパス

```
URL: /admin/settings  → 403 Forbidden (ブロック)
URL: /ａdmin/settings → 200 OK (Unicodeの全角'ａ'で回避)
URL: /admin%2fsettings → パスの解釈がサーバによって異なる
```

対策: パス正規化を行った後にアクセス制御チェックを適用する。

### エッジケース3: HTTPメソッドの不整合

```
GET /api/users/123  → 認可チェックあり → 403
HEAD /api/users/123 → 認可チェックなし → 200 (情報漏洩)
OPTIONS /api/users/123 → CORS preflight → 許可メソッド一覧が漏洩
```

対策: 全てのHTTPメソッドに対して一貫した認可チェックを実装する。

### エッジケース4: レースコンディションによる認可バイパス

```python
# TOCTOU (Time of Check to Time of Use) 問題
# Step 1: 残高チェック（残高: 100円）
# Step 2: 引き出し処理（100円引き出し）
# 並行リクエスト: Step 1とStep 2の間に同じリクエストが通ると二重引き出し

# 対策: データベースレベルのロック
def withdraw(user_id, amount):
    with db.transaction():
        balance = db.execute(
            "SELECT balance FROM accounts WHERE user_id = ? FOR UPDATE",
            (user_id,)
        ).fetchone()
        if balance[0] >= amount:
            db.execute(
                "UPDATE accounts SET balance = balance - ? WHERE user_id = ?",
                (amount, user_id)
            )
```

---

## 13. アンチパターン

### アンチパターン1: セキュリティヘッダーの欠如

セキュリティヘッダーを設定せずにアプリケーションをデプロイするパターン。CSP、HSTS、X-Frame-Options等のヘッダーは、追加コストなしでクライアントサイドの攻撃を大幅に緩和できる。

**検出方法**: `securityheaders.com` でスキャン、またはブラウザの開発者ツールでレスポンスヘッダーを確認。

### アンチパターン2: エラーメッセージでの情報漏洩

スタックトレースやDB接続情報をエラーレスポンスに含めるパターン。本番環境では一般的なエラーメッセージのみを返し、詳細はサーバーサイドのログに記録する。

```python
# NG: スタックトレースを返す
@app.errorhandler(Exception)
def handle_error(e):
    return {"error": str(e), "traceback": traceback.format_exc()}, 500

# OK: 一般的なメッセージのみ
@app.errorhandler(Exception)
def handle_error(e):
    error_id = str(uuid.uuid4())
    app.logger.error(f"Error {error_id}: {e}", exc_info=True)
    return {"error": "Internal server error", "reference": error_id}, 500
```

### アンチパターン3: クライアントサイドのみのバリデーション

```javascript
// NG: フロントエンドのみでバリデーション
// → ブラウザの開発者ツールやcurlで簡単にバイパス可能

// OK: フロントエンドは UX 向上のため、サーバサイドが本質的な防御
// フロントエンド: 即時フィードバック用
// バックエンド: セキュリティ用（必須）
```

---

## 14. 演習

### 演習1（基礎）: セキュリティヘッダーの確認

任意のWebサイトの開発者ツール（Network タブ）を開き、以下のセキュリティヘッダーの有無を確認せよ:

1. `Content-Security-Policy`
2. `Strict-Transport-Security`
3. `X-Content-Type-Options`
4. `X-Frame-Options`
5. `Referrer-Policy`

各ヘッダーの目的と、欠如している場合のリスクを説明せよ。

### 演習2（中級）: SQLインジェクションの検出と修正

以下のコードにはSQLインジェクションの脆弱性がある。攻撃ペイロードを特定し、安全なコードに修正せよ。

```python
def login(username, password):
    query = f"""
        SELECT * FROM users
        WHERE username = '{username}'
        AND password = '{hashlib.md5(password.encode()).hexdigest()}'
    """
    result = db.execute(query)
    if result:
        return create_session(result[0])
    return None
```

修正すべき問題:
- SQLインジェクション
- MD5ハッシュの使用
- ソルトなしのハッシュ

### 演習3（上級）: 包括的なセキュリティレビュー

以下のFlaskアプリケーションのセキュリティ問題を全て特定し、修正版を作成せよ:

```python
from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)
app.secret_key = "secret123"

@app.route("/api/user/<user_id>")
def get_user(user_id):
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
    user = cursor.fetchone()
    conn.close()
    return jsonify({"user": user})

@app.route("/api/search")
def search():
    q = request.args.get("q")
    return f"<h1>Results for: {q}</h1>"

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    file.save(f"/uploads/{file.filename}")
    return jsonify({"status": "uploaded"})
```

問題点: シークレットキーのハードコード、SQLインジェクション、XSS、パストラバーサル、アクセス制御の欠如、セキュリティヘッダーの欠如。

---

## 15. パフォーマンスに関する考察

### セキュリティ対策のパフォーマンスへの影響

| 対策 | パフォーマンス影響 | 最適化方法 |
|------|------------------|-----------|
| bcrypt (rounds=12) | ~300ms/ハッシュ | 認証時のみ使用、非同期処理 |
| TLS 1.3 | 初回接続 +1-2ms | 0-RTT再接続、TLSセッション再利用 |
| CSP ヘッダー | ほぼゼロ | - |
| 入力検証 | <1ms | コンパイル済み正規表現 |
| レートリミット (Redis) | ~1ms | ローカルキャッシュ併用 |
| WAF | 5-20ms | ルールの最適化、バイパスルート |

### パスワードハッシュのコストファクターとレスポンス時間

```
bcrypt rounds vs 処理時間（概算）:
  rounds=10:  ~100ms  ← 開発環境向け
  rounds=11:  ~200ms
  rounds=12:  ~400ms  ← 本番最低ライン（OWASP推奨）
  rounds=13:  ~800ms
  rounds=14: ~1600ms  ← 高セキュリティ要件

Argon2id パラメータ vs 処理時間:
  memory=32MB, time=3:   ~100ms  ← 最低ライン
  memory=64MB, time=3:   ~200ms  ← OWASP推奨
  memory=128MB, time=4:  ~500ms  ← 高セキュリティ要件
```

---

## 16. トラブルシューティング

### よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| CSP違反エラーが大量発生 | CSPポリシーが厳しすぎる | report-onlyモードで段階的に導入 |
| HSTSが効かない | preloadリストに未登録 | max-age を十分長くし preload 申請 |
| CORSエラー | Origin が許可リストにない | 正確なオリジンを指定（* は避ける） |
| セッション固定攻撃 | ログイン後にセッションIDが変わらない | 認証成功後にセッションを再生成 |
| bcryptが遅すぎる | rounds が高すぎる | 非同期処理 + 適切なrounds設定 |

---

## FAQ

### Q1: OWASP Top 10は全てカバーすれば十分ですか?

十分ではない。OWASP Top 10は最も一般的な脆弱性を示したものであり、網羅的なセキュリティチェックリストではない。OWASP ASVS（Application Security Verification Standard）をより包括的なガイドラインとして活用すべきである。ASVSはレベル1（基本）、レベル2（標準）、レベル3（高度）の3段階で286の検証項目を提供する。

### Q2: A04「安全でない設計」はコードレベルで対策できますか?

コードレベルだけでは対策できない。設計段階での脅威モデリング、セキュリティ要件の定義、アーキテクチャレビューが必要である。「セキュアコーディング」は「安全な設計」を前提として初めて効果を発揮する。具体的には、STRIDE分析、データフロー図（DFD）の作成、信頼境界の定義を設計段階で実施する。

### Q3: どの脆弱性から優先的に対策すべきですか?

自組織のリスクアセスメントに基づいて判断すべきだが、一般的にはA01（アクセス制御）とA03（インジェクション）が最も被害が大きく、対策の優先度が高い。ただし、A05（設定ミス）は対策コストが最も低く、即座に適用できるため、最初に着手することが推奨される。

### Q4: OWASP Top 10 はどのような頻度で更新されますか?

2-4年ごとに更新される。過去のリリース: 2003, 2004, 2007, 2010, 2013, 2017, 2021。次回の更新は2025年または2026年が見込まれる。各バージョン間の変更は、新たな脅威の出現とデータ分析に基づいている。

### Q5: マイクロサービスアーキテクチャ特有のOWASP対策は?

マイクロサービスでは以下の追加考慮が必要: (1) サービス間認証（mTLS、JWTの伝播）、(2) API Gatewayでの一元的なレートリミットとInput Validation、(3) サービスメッシュによるネットワークポリシー、(4) 分散トレーシングによるセキュリティイベントの可視化。

---

## まとめ

| 順位 | カテゴリ | 核心的な対策 | 検出ツール |
|------|---------|-------------|-----------|
| A01 | アクセス制御の不備 | サーバーサイドでの認可チェック、デフォルト拒否 | Burp Suite |
| A02 | 暗号化の失敗 | TLS強制、適切なアルゴリズム選択 | testssl.sh |
| A03 | インジェクション | パラメータ化クエリ、入力検証 | SQLMap, Semgrep |
| A04 | 安全でない設計 | 脅威モデリング、セキュリティ要件 | 設計レビュー |
| A05 | 設定ミス | ハードニング、自動構成管理 | ScoutSuite |
| A06 | 古いコンポーネント | SCA、自動更新 | Dependabot, Snyk |
| A07 | 認証の失敗 | MFA、セキュアなセッション管理 | Hydra |
| A08 | 整合性不具合 | 署名検証、SRI | Sigstore |
| A09 | ログ不備 | SIEM、構造化ログ | ELK Stack |
| A10 | SSRF | URL検証、ネットワーク分離 | Burp Suite |

### 防御の原則

```
+------------------------------------------------------------------+
|  セキュリティ防御の5原則                                             |
|------------------------------------------------------------------|
|  1. Defense in Depth (多層防御)                                    |
|     単一の対策に依存しない。WAF + 入力検証 + パラメータ化クエリ      |
|                                                                    |
|  2. Least Privilege (最小権限)                                     |
|     必要最小限の権限のみ付与。デフォルト拒否                         |
|                                                                    |
|  3. Fail Securely (安全な失敗)                                     |
|     エラー時はアクセスを拒否。情報を漏洩しない                       |
|                                                                    |
|  4. Don't Trust User Input (入力を信頼しない)                      |
|     全ての入力をサーバーサイドで検証                                 |
|                                                                    |
|  5. Security by Design (設計段階からのセキュリティ)                 |
|     後付けではなく、設計段階からセキュリティを組み込む                |
+------------------------------------------------------------------+
```

---

## 次に読むべきガイド

- [01-xss-prevention.md](./01-xss-prevention.md) -- XSS攻撃の詳細な対策手法
- [03-injection.md](./03-injection.md) -- インジェクション攻撃の深掘り
- [04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md) -- 認証脆弱性の詳細
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) -- 暗号化通信の基盤
- [APIセキュリティ](../03-network-security/02-api-security.md) -- API の認証認可とレートリミット

---

## 参考文献

1. OWASP Top 10:2021 -- https://owasp.org/Top10/
2. OWASP Application Security Verification Standard (ASVS) v4.0 -- https://owasp.org/www-project-application-security-verification-standard/
3. OWASP Testing Guide v4.2 -- https://owasp.org/www-project-web-security-testing-guide/
4. OWASP Cheat Sheet Series -- https://cheatsheetseries.owasp.org/
5. CWE/SANS Top 25 Most Dangerous Software Errors -- https://cwe.mitre.org/top25/
6. NIST SP 800-53 Security and Privacy Controls -- https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final
7. RFC 6749 - The OAuth 2.0 Authorization Framework -- https://datatracker.ietf.org/doc/html/rfc6749
8. Mozilla Web Security Guidelines -- https://infosec.mozilla.org/guidelines/web_security
