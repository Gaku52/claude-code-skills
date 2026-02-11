# インジェクション

> SQL、NoSQL、コマンド、LDAPインジェクションの攻撃手法と、パラメータ化クエリ、ORM、入力検証による体系的な防御策を解説する。

## この章で学ぶこと

1. **各種インジェクション攻撃**（SQL/NoSQL/コマンド/LDAP）の原理と危険性を理解する
2. **パラメータ化クエリとORM**を使った根本的な防御手法を習得する
3. **入力検証と出力エンコード**による多層防御のアプローチを身につける

---

## 1. インジェクション攻撃の原理

インジェクションとは、ユーザー入力がコード・クエリ・コマンドの一部として解釈されることで、攻撃者が意図しない操作を実行する脆弱性である。

```
インジェクションの基本原理:

  正常なリクエスト:
  ユーザー入力: "alice"
  生成SQL: SELECT * FROM users WHERE name = 'alice'
                                        ^^^^^^^^
                                        データとして扱われる

  攻撃リクエスト:
  ユーザー入力: "' OR '1'='1"
  生成SQL: SELECT * FROM users WHERE name = '' OR '1'='1'
                                        ^^^^^^^^^^^^^^^^^^^^^
                                        コードとして解釈される!
```

---

## 2. SQLインジェクション

### 2.1 基本的な攻撃パターン

```python
# コード例1: SQLインジェクションの攻撃パターンと防御

import sqlite3

# === 脆弱なコード ===
def login_vulnerable(username, password):
    """文字列連結によるSQL構築 -> SQLインジェクション脆弱"""
    conn = sqlite3.connect("app.db")
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    # 攻撃例: username = "admin' --"
    # 生成SQL: SELECT * FROM users WHERE username='admin' --' AND password=''
    # -- 以降はコメント -> パスワード検証がスキップされる
    result = conn.execute(query).fetchone()
    return result is not None

# === 安全なコード: パラメータ化クエリ ===
def login_safe(username, password):
    """パラメータ化クエリで安全にSQLを実行"""
    conn = sqlite3.connect("app.db")
    query = "SELECT * FROM users WHERE username=? AND password=?"
    # ? はプレースホルダ -> 入力は常にデータとして扱われる
    result = conn.execute(query, (username, password)).fetchone()
    return result is not None

# === さらに安全: ORM使用 ===
from sqlalchemy.orm import Session
from sqlalchemy import select

def login_orm(session: Session, username: str, password_hash: str):
    """ORMを使用した安全なクエリ"""
    stmt = select(User).where(
        User.username == username,
        User.password_hash == password_hash,
    )
    return session.execute(stmt).scalar_one_or_none()
```

### 2.2 高度なSQLインジェクション

```
SQLインジェクションの種類:

+----------------+-----------------------------+------------------+
| 種類           | 特徴                        | 検出難度         |
+----------------+-----------------------------+------------------+
| Classic        | エラーメッセージから情報取得  | 低               |
| Union-based    | UNIONで他テーブルのデータ取得 | 中               |
| Blind (Boolean)| 真偽値の応答差から情報推測    | 高               |
| Blind (Time)   | レスポンス時間差から情報推測  | 高               |
| Second-order   | 保存後に別の場所で発動       | 非常に高         |
+----------------+-----------------------------+------------------+
```

```python
# コード例2: Second-order SQLインジェクションの例と対策

# Second-order: 入力時ではなく、保存したデータの使用時に発動

# 脆弱なコード
def register_user(username, password):
    """ユーザー登録（パラメータ化されているので安全に見える）"""
    db.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, password)  # ここは安全
    )

def change_password(username, new_password):
    """パスワード変更（ここが脆弱!）"""
    # usernameをDBから取得してSQLに埋め込む
    user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    # user["username"] = "admin'--" (登録時に仕込まれた値)
    db.execute(
        f"UPDATE users SET password='{new_password}' WHERE username='{user['username']}'"
    )
    # 結果: UPDATE users SET password='...' WHERE username='admin'--'
    # admin のパスワードが変更される!

# 安全なコード: すべてのSQL文でパラメータ化を徹底
def change_password_safe(username, new_password):
    db.execute(
        "UPDATE users SET password=? WHERE username=?",
        (new_password, username)  # 常にパラメータ化
    )
```

---

## 3. NoSQLインジェクション

```python
# コード例3: NoSQLインジェクション（MongoDB）の攻撃と対策
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["myapp"]

# === 脆弱なコード ===
def find_user_vulnerable(request_data):
    """JSONデータをそのままクエリに使用 -> NoSQL injection"""
    username = request_data["username"]
    password = request_data["password"]
    # 攻撃: {"username": "admin", "password": {"$ne": ""}}
    # $ne (not equal) で空文字以外 -> 任意のパスワードで認証成功
    user = db.users.find_one({"username": username, "password": password})
    return user

# === 安全なコード ===
def find_user_safe(request_data):
    """入力をバリデーションしてからクエリに使用"""
    username = request_data.get("username", "")
    password = request_data.get("password", "")

    # 型チェック: 文字列のみ許可（オブジェクトを拒否）
    if not isinstance(username, str) or not isinstance(password, str):
        raise ValueError("Invalid input type")

    # 長さ制限
    if len(username) > 100 or len(password) > 200:
        raise ValueError("Input too long")

    # MongoDB演算子の除去
    if any(key.startswith("$") for key in [username, password]
           if isinstance(key, str) and key.startswith("$")):
        raise ValueError("Invalid characters in input")

    user = db.users.find_one({
        "username": str(username),  # 明示的に文字列に変換
        "password_hash": hash_password(str(password)),
    })
    return user
```

---

## 4. コマンドインジェクション

```python
# コード例4: コマンドインジェクションの攻撃と対策
import subprocess
import shlex
import re

# === 脆弱なコード ===
def ping_host_vulnerable(host):
    """os.systemやshell=Trueでのコマンド実行 -> コマンドインジェクション"""
    import os
    os.system(f"ping -c 3 {host}")
    # 攻撃: host = "google.com; cat /etc/passwd"
    # 実行: ping -c 3 google.com; cat /etc/passwd

# === 安全なコード ===
def ping_host_safe(host: str) -> str:
    """安全なコマンド実行"""
    # Step 1: 入力バリデーション（ホワイトリスト方式）
    if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
        raise ValueError(f"Invalid hostname: {host}")

    # Step 2: shell=Falseでリスト形式で引数を渡す
    result = subprocess.run(
        ["ping", "-c", "3", host],  # リスト形式 -> シェル解釈されない
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,  # 明示的にFalse（デフォルトだが明示する）
    )
    return result.stdout

# === より安全: 外部コマンドを使わない ===
import socket

def check_host_reachable(host: str) -> bool:
    """外部コマンドを使わずにホストの到達性を確認"""
    if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
        raise ValueError(f"Invalid hostname: {host}")
    try:
        socket.create_connection((host, 80), timeout=5)
        return True
    except (socket.timeout, socket.error):
        return False
```

---

## 5. LDAPインジェクション

```python
# コード例5: LDAPインジェクションの攻撃と対策

# === 脆弱なコード ===
def search_user_vulnerable(username):
    """文字列連結によるLDAPフィルタ構築"""
    ldap_filter = f"(&(uid={username})(objectClass=person))"
    # 攻撃: username = "*)(uid=*))(|(uid=*"
    # 生成: (&(uid=*)(uid=*))(|(uid=*)(objectClass=person))
    # -> 全ユーザーが返される
    return ldap_conn.search_s(base_dn, ldap.SCOPE_SUBTREE, ldap_filter)

# === 安全なコード ===
def ldap_escape(value: str) -> str:
    """LDAP特殊文字をエスケープする"""
    escape_chars = {
        '\\': r'\5c',
        '*': r'\2a',
        '(': r'\28',
        ')': r'\29',
        '\x00': r'\00',
    }
    result = value
    for char, replacement in escape_chars.items():
        result = result.replace(char, replacement)
    return result

def search_user_safe(username: str):
    """エスケープ済みのLDAPフィルタを構築"""
    safe_username = ldap_escape(username)
    ldap_filter = f"(&(uid={safe_username})(objectClass=person))"
    return ldap_conn.search_s(base_dn, ldap.SCOPE_SUBTREE, ldap_filter)
```

---

## 6. インジェクション防御の体系

```
インジェクション防御の多層構造:

  Layer 1: 入力バリデーション
  +----------------------------------------------+
  | ホワイトリスト、型チェック、長さ制限            |
  +----------------------------------------------+
                      |
  Layer 2: パラメータ化 / ORM
  +----------------------------------------------+
  | データとコードの分離、プリペアドステートメント   |
  +----------------------------------------------+
                      |
  Layer 3: 出力エンコード
  +----------------------------------------------+
  | コンテキスト別エスケープ（HTML/SQL/Shell/LDAP）|
  +----------------------------------------------+
                      |
  Layer 4: 最小権限
  +----------------------------------------------+
  | DB権限の制限、サンドボックス、WAF              |
  +----------------------------------------------+
```

### インジェクション種別の対策比較

| インジェクション種別 | 根本対策 | 補助対策 | テストツール |
|-------------------|---------|---------|------------|
| SQL | パラメータ化クエリ | WAF、最小権限DB | SQLMap |
| NoSQL | 型チェック、演算子フィルタ | スキーマ検証 | NoSQLMap |
| コマンド | shell=False、引数リスト | 入力ホワイトリスト | Commix |
| LDAP | 特殊文字エスケープ | 入力バリデーション | LDAP Injection Tester |
| XPath | パラメータ化XPath | 入力制限 | - |
| テンプレート | サンドボックス | テンプレートエンジン設定 | tplmap |

---

## アンチパターン

### アンチパターン1: ブラックリストによるフィルタリング

`SELECT`、`DROP` 等の危険なキーワードをフィルタするアプローチ。バイパス手法は無数にあり（大文字小文字の混在、エンコーディング、コメント挿入等）、根本的な対策にはならない。パラメータ化クエリが唯一の正解である。

### アンチパターン2: クライアントサイドのみのバリデーション

フロントエンドのJavaScriptでのみ入力検証を行うパターン。攻撃者はブラウザを経由せずAPIに直接リクエストを送信できるため、必ずサーバーサイドでバリデーションを実施する。

---

## FAQ

### Q1: ORMを使っていればSQLインジェクションは完全に防げますか?

ほぼ防げるが、完全ではない。ORMでも`raw()`や`execute()`で直接SQLを書く場合や、文字列連結でクエリを構築する場合にはSQLインジェクションが発生する。また、ORMの特定バージョンに脆弱性が存在する場合もある。

### Q2: WAFだけでインジェクションを防げますか?

WAFだけでは不十分。WAFはシグネチャベースの検出であり、高度なバイパス手法には対応できない場合がある。WAFは補助的な防御層として位置づけ、パラメータ化クエリ等の根本対策と併用すべきである。

### Q3: プリペアドステートメントとパラメータ化クエリは同じものですか?

概念的にはほぼ同じだが、厳密には異なる。プリペアドステートメントはDB側でクエリプランをキャッシュする仕組みを含み、パラメータ化クエリはデータとコードを分離する手法を指す。両方とも、インジェクション防御として有効である。

---

## まとめ

| 防御手法 | 対象 | 効果 | 推奨度 |
|---------|------|------|--------|
| パラメータ化クエリ | SQL/NoSQL | データとコードの完全分離 | 必須 |
| ORM | SQL | 安全なクエリ構築の抽象化 | 推奨 |
| 入力バリデーション | 全般 | 不正入力の早期排除 | 必須 |
| shell=Falseリスト実行 | コマンド | シェル解釈の回避 | 必須 |
| エスケープ | LDAP/XPath | 特殊文字の無害化 | 必須 |
| WAF | 全般 | 既知パターンのブロック | 補助 |

---

## 次に読むべきガイド

- [04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md) -- 認証脆弱性とセッション管理
- [01-xss-prevention.md](./01-xss-prevention.md) -- XSS（HTMLインジェクション）の詳細
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般

---

## 参考文献

1. OWASP Injection Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Injection_Prevention_Cheat_Sheet.html
2. OWASP SQL Injection Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
3. PortSwigger Web Security Academy: SQL Injection -- https://portswigger.net/web-security/sql-injection
4. CWE-89: Improper Neutralization of Special Elements used in an SQL Command -- https://cwe.mitre.org/data/definitions/89.html
