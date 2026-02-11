# セキュアコーディング

> 入力検証、出力エンコード、安全なエラーハンドリングを中心に、アプリケーションコードレベルでの脆弱性を防止するための実践的ガイド

## この章で学ぶこと

1. **入力検証の原則** — ホワイトリスト方式による安全な入力処理とインジェクション防止
2. **出力エンコード** — XSS を防ぐコンテキスト別のエスケープ手法
3. **安全なエラーハンドリング** — 情報漏洩を防ぎつつデバッグを支援するエラー処理設計

---

## 1. セキュアコーディングの原則

### 基本原則

```
+----------------------------------------------------------+
|              セキュアコーディング 7 原則                     |
|----------------------------------------------------------|
|  1. 入力を信用しない (Validate All Input)                  |
|  2. 最小権限の原則 (Principle of Least Privilege)          |
|  3. 多層防御 (Defense in Depth)                           |
|  4. 安全なデフォルト (Secure Defaults)                     |
|  5. フェイルセキュア (Fail Securely)                       |
|  6. 攻撃面の最小化 (Minimize Attack Surface)               |
|  7. セキュリティを分離しない (Don't Roll Your Own Crypto)   |
+----------------------------------------------------------+
```

---

## 2. 入力検証

### 入力検証の戦略

```
外部入力 (信頼できない)
    |
    v
+------------------+
| 1. 型チェック     |  文字列? 数値? 配列?
+------------------+
    |
    v
+------------------+
| 2. 長さ制限      |  最大長・最小長
+------------------+
    |
    v
+------------------+
| 3. 範囲チェック   |  最小値・最大値
+------------------+
    |
    v
+------------------+
| 4. パターン検証   |  ホワイトリスト正規表現
+------------------+
    |
    v
+------------------+
| 5. ビジネスロジック|  整合性・妥当性
+------------------+
    |
    v
安全な内部データ
```

### SQL インジェクション防止

```python
# NG: 文字列結合による SQL 構築
def get_user_bad(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    # 入力: ' OR '1'='1' -- で全件取得される
    cursor.execute(query)

# OK: パラメータ化クエリ (プリペアドステートメント)
def get_user_good(username):
    query = "SELECT * FROM users WHERE name = %s"
    cursor.execute(query, (username,))

# OK: ORM を使用
def get_user_orm(username):
    user = User.query.filter_by(name=username).first()
    return user
```

### コマンドインジェクション防止

```python
import subprocess
import shlex

# NG: シェルコマンドの文字列結合
def ping_bad(host):
    os.system(f"ping -c 1 {host}")
    # 入力: "8.8.8.8; rm -rf /" で任意コマンド実行

# OK: subprocess + リスト引数 (shell=False)
def ping_good(host):
    # ホワイトリストで検証
    import re
    if not re.match(r'^[\d.]+$', host):
        raise ValueError("Invalid host")

    result = subprocess.run(
        ["ping", "-c", "1", host],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout
```

### パストラバーサル防止

```python
import os

# NG: ユーザ入力をパスに直接使用
def read_file_bad(filename):
    path = f"/app/uploads/{filename}"
    # 入力: "../../etc/passwd" で任意ファイル読み取り
    return open(path).read()

# OK: パスの正規化と検証
def read_file_good(filename):
    base_dir = os.path.realpath("/app/uploads")
    full_path = os.path.realpath(os.path.join(base_dir, filename))

    # ベースディレクトリ外へのアクセスを拒否
    if not full_path.startswith(base_dir + os.sep):
        raise ValueError("Access denied: path traversal detected")

    if not os.path.isfile(full_path):
        raise FileNotFoundError("File not found")

    return open(full_path).read()
```

---

## 3. 出力エンコード

### コンテキスト別エスケープ

```
+-------------------------------------------------------+
|  コンテキスト      |  エスケープ方法                      |
|-------------------------------------------------------+
|  HTML 本文        |  & → &amp; < → &lt; > → &gt;       |
|  HTML 属性        |  " → &quot; ' → &#x27;             |
|  JavaScript       |  \x エスケープ                       |
|  URL              |  パーセントエンコーディング            |
|  CSS              |  \HH エスケープ                      |
|  SQL              |  パラメータ化クエリ (エスケープ不要)    |
+-------------------------------------------------------+
```

### XSS 防止の実装

```javascript
// テンプレートエンジン (EJS) - 自動エスケープ
// <%= user.name %> → HTML エスケープされる
// <%- user.bio %>  → 生の HTML (危険!)

// React - JSX は自動エスケープ
function UserProfile({ user }) {
  return (
    <div>
      <h1>{user.name}</h1>  {/* 自動エスケープ */}

      {/* NG: dangerouslySetInnerHTML は XSS リスク */}
      <div dangerouslySetInnerHTML={{ __html: user.bio }} />

      {/* OK: サニタイズライブラリを通す */}
      <div dangerouslySetInnerHTML={{
        __html: DOMPurify.sanitize(user.bio, {
          ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a'],
          ALLOWED_ATTR: ['href'],
        })
      }} />
    </div>
  );
}
```

### Content Security Policy (CSP)

```
# 厳格な CSP ヘッダ
Content-Security-Policy:
  default-src 'none';
  script-src 'self' 'nonce-{RANDOM}';
  style-src 'self';
  img-src 'self' https://cdn.example.com;
  font-src 'self';
  connect-src 'self' https://api.example.com;
  frame-ancestors 'none';
  base-uri 'self';
  form-action 'self';
  upgrade-insecure-requests;
```

---

## 4. エラーハンドリング

### 安全なエラーレスポンス設計

```
+----------------------------------------------------------+
|                エラーレスポンスの設計                        |
|----------------------------------------------------------|
|                                                          |
|  開発環境:                                                |
|  {                                                       |
|    "error": "DatabaseError",                             |
|    "message": "relation \"users\" does not exist",       |
|    "stack": "at Query.run (/app/db.js:42:15)...",        |
|    "query": "SELECT * FROM users WHERE..."               |
|  }                                                       |
|                                                          |
|  本番環境:                                                |
|  {                                                       |
|    "error": "Internal Server Error",                     |
|    "requestId": "req-abc123"                             |
|  }                                                       |
|  (詳細はサーバログに記録、requestId で追跡)                 |
+----------------------------------------------------------+
```

### エラーハンドリングの実装

```python
import logging
import uuid
from flask import Flask, jsonify, request

app = Flask(__name__)
logger = logging.getLogger(__name__)

class AppError(Exception):
    """アプリケーションエラー基底クラス"""
    def __init__(self, message, status_code=500, internal_message=None):
        self.message = message              # ユーザに返すメッセージ
        self.status_code = status_code
        self.internal_message = internal_message  # ログ用の詳細

class NotFoundError(AppError):
    def __init__(self, resource="Resource"):
        super().__init__(f"{resource} not found", 404)

class ValidationError(AppError):
    def __init__(self, details):
        super().__init__("Validation failed", 400)
        self.details = details

@app.errorhandler(AppError)
def handle_app_error(error):
    request_id = str(uuid.uuid4())[:8]

    # 内部ログには詳細を記録
    logger.error(
        "AppError: %s | request_id=%s | path=%s | internal=%s",
        error.message, request_id, request.path,
        error.internal_message or "N/A",
    )

    # ユーザには最小限の情報のみ返す
    response = {
        "error": error.message,
        "requestId": request_id,
    }
    if hasattr(error, 'details'):
        response["details"] = error.details

    return jsonify(response), error.status_code

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    request_id = str(uuid.uuid4())[:8]

    # 予期しないエラーはスタックトレース付きでログ
    logger.exception("Unexpected error: request_id=%s", request_id)

    # ユーザには一般的なメッセージのみ
    return jsonify({
        "error": "Internal Server Error",
        "requestId": request_id,
    }), 500
```

---

## 5. その他のセキュアコーディング手法

### CSRF 防止

```python
# Flask-WTF による CSRF トークン
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect(app)

# テンプレート内
# <form method="POST">
#   <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
# </form>

# AJAX リクエストの場合
# <meta name="csrf-token" content="{{ csrf_token() }}">
# fetch('/api/data', {
#   method: 'POST',
#   headers: { 'X-CSRFToken': document.querySelector('meta[name=csrf-token]').content },
# })
```

### 安全なパスワード処理

```python
import bcrypt

# パスワードのハッシュ化
def hash_password(password: str) -> bytes:
    # bcrypt は自動的にソルトを生成
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

# パスワードの検証
def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# NG: MD5/SHA256 を直接使用
# import hashlib
# hashlib.sha256(password.encode()).hexdigest()  # ソルトなし、高速すぎる
```

### セキュアなセッション管理

```python
from flask import Flask, session
import os

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ['SESSION_SECRET'],   # 十分な長さのランダム値
    SESSION_COOKIE_SECURE=True,                # HTTPS のみ
    SESSION_COOKIE_HTTPONLY=True,               # JavaScript からアクセス不可
    SESSION_COOKIE_SAMESITE='Lax',             # CSRF 防止
    PERMANENT_SESSION_LIFETIME=1800,           # 30 分で期限切れ
)
```

---

## 6. アンチパターン

### アンチパターン 1: クライアント側のみの検証

```javascript
// NG: フロントエンドのみでバリデーション
function submitForm() {
  if (document.getElementById('age').value < 0) {
    alert('Invalid age');
    return;
  }
  fetch('/api/users', { method: 'POST', body: formData });
  // → 攻撃者は DevTools や curl で直接 API を叩ける
}

// OK: サーバ側でも必ず検証
// フロントエンドの検証は UX のためであり、セキュリティではない
```

### アンチパターン 2: エラーメッセージでの情報漏洩

```python
# NG: 内部情報を含むエラーメッセージ
@app.route('/login', methods=['POST'])
def login():
    user = User.query.filter_by(email=email).first()
    if not user:
        return {"error": "User with this email does not exist"}, 401
    if not verify_password(password, user.password_hash):
        return {"error": "Incorrect password"}, 401
    # → ユーザの存在を確認できる (列挙攻撃)

# OK: 同一メッセージを返す
@app.route('/login', methods=['POST'])
def login():
    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return {"error": "Invalid email or password"}, 401
```

---

## 7. FAQ

### Q1. 入力検証はフロントエンドとバックエンドの両方に必要か?

はい。フロントエンドの検証は UX 改善のためであり、セキュリティ対策にはならない。攻撃者は curl や Burp Suite で直接バックエンドに任意のリクエストを送信できる。バックエンドでの検証が唯一の信頼できる防御線である。

### Q2. ORM を使えば SQL インジェクションは完全に防げるか?

ORM の標準的な API を使う限りはほぼ安全である。ただし、生 SQL を組み立てる機能 (SQLAlchemy の `text()` や Django の `raw()`) を使う場合は、パラメータ化クエリと同じ注意が必要である。ORM でも `extra()` や `RawSQL` の使用は慎重に行うべきである。

### Q3. Content Security Policy (CSP) はどこまで厳格にすべきか?

`default-src 'none'` から始めて、必要なリソースのみを明示的に許可するのが理想的である。nonce ベースの script-src を採用し、`unsafe-inline` や `unsafe-eval` は避ける。CSP Reporting を有効にして、違反レポートを収集しながら段階的に厳格化するのが現実的なアプローチである。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 入力検証 | ホワイトリスト方式、サーバ側で必ず実施 |
| SQL インジェクション | パラメータ化クエリまたは ORM を使用 |
| XSS 防止 | コンテキスト別エスケープ + CSP |
| CSRF 防止 | トークンベース + SameSite Cookie |
| エラーハンドリング | 本番では詳細を隠し requestId で追跡 |
| パスワード | bcrypt/argon2 でハッシュ化、MD5/SHA 禁止 |
| セッション | Secure + HttpOnly + SameSite 属性必須 |

---

## 次に読むべきガイド

- [依存関係セキュリティ](./01-dependency-security.md) — サードパーティライブラリの脆弱性管理
- [SAST/DAST](./03-sast-dast.md) — 自動化されたコードセキュリティ検査
- [APIセキュリティ](../03-network-security/02-api-security.md) — API レベルの認証・認可

---

## 参考文献

1. **OWASP Secure Coding Practices Quick Reference Guide** — https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/
2. **OWASP Cheat Sheet Series** — https://cheatsheetseries.owasp.org/
3. **CWE/SANS Top 25 Most Dangerous Software Weaknesses** — https://cwe.mitre.org/top25/
4. **Mozilla Web Security Guidelines** — https://infosec.mozilla.org/guidelines/web_security
