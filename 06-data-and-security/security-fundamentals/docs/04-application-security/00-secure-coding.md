# セキュアコーディング

> 入力検証、出力エンコード、安全なエラーハンドリングを中心に、アプリケーションコードレベルでの脆弱性を防止するための実践的ガイド。OWASP Top 10 に対応したセキュアコーディングの原則、具体的な攻撃手法と防御策、言語別ベストプラクティスを網羅する。

## この章で学ぶこと

1. **入力検証の原則と実装** — ホワイトリスト方式による安全な入力処理、型安全な検証、インジェクション防止の多層防御
2. **出力エンコードとXSS防御** — コンテキスト別のエスケープ手法、CSP による多層防御、DOMベース XSS の防止
3. **安全なエラーハンドリング** — 情報漏洩を防ぎつつデバッグを支援するエラー処理設計、構造化ログとの統合
4. **セキュアなセッション・認証管理** — CSRF 防止、パスワード処理、セッション固定攻撃対策

## 前提知識

| トピック | 参照先 |
|---------|--------|
| Web セキュリティの基本概念 | [Webセキュリティ基礎](../01-web-security/) |
| 暗号化の基礎 | [暗号化基礎](../02-cryptography/) |
| HTTP プロトコルの理解 | [ネットワークセキュリティ](../03-network-security/) |
| 認証・認可の基本 | [認証と認可](../../authentication-and-authorization/00-fundamentals/) |

---

## 1. セキュアコーディングの原則

### WHY: なぜセキュアコーディングが必要か

セキュリティ脆弱性の修正コストは、開発フェーズが進むほど指数関数的に増大する。設計段階での修正コストを1とすると、テスト段階では15倍、本番稼働後では100倍以上になるという研究がある（NIST/IBM Systems Sciences Institute）。セキュアコーディングは「後付けのセキュリティ」ではなく、開発プロセスの最初からセキュリティを組み込む「シフトレフト」の中核をなす。

### 基本原則

```
+-------------------------------------------------------------------+
|                セキュアコーディング 7 原則                             |
|-------------------------------------------------------------------|
|  1. 入力を信用しない (Validate All Input)                            |
|     → 全ての外部入力はホワイトリスト方式で検証                         |
|                                                                   |
|  2. 最小権限の原則 (Principle of Least Privilege)                    |
|     → 処理に必要な最小限のアクセス権のみ付与                           |
|                                                                   |
|  3. 多層防御 (Defense in Depth)                                     |
|     → 単一の防御策に依存せず複数の防御層を構築                         |
|                                                                   |
|  4. 安全なデフォルト (Secure Defaults)                               |
|     → デフォルト設定が最も安全な状態にする                             |
|                                                                   |
|  5. フェイルセキュア (Fail Securely)                                 |
|     → エラー発生時は安全側に倒して処理を中止                           |
|                                                                   |
|  6. 攻撃面の最小化 (Minimize Attack Surface)                         |
|     → 不要な機能・API・ポートを無効化                                 |
|                                                                   |
|  7. セキュリティを自作しない (Don't Roll Your Own Crypto)             |
|     → 暗号化・認証は実績あるライブラリを使用                           |
+-------------------------------------------------------------------+
```

### 原則の適用マトリクス

| 原則 | 入力処理 | 出力処理 | エラー処理 | 認証処理 | データ保存 |
|------|---------|---------|-----------|---------|-----------|
| 入力検証 | ホワイトリスト | - | エラー入力の検証 | 資格情報の形式検証 | データ形式の検証 |
| 最小権限 | 必要なフィールドのみ受理 | 必要な情報のみ出力 | 最小限のエラー情報 | 必要な権限のみ付与 | 必要なカラムのみアクセス |
| 多層防御 | フロント+バック検証 | エスケープ+CSP | ログ+監視+アラート | MFA+レート制限 | 暗号化+アクセス制御 |
| フェイルセキュア | 不正入力は拒否 | エスケープ失敗は非表示 | 500で詳細非公開 | 認証失敗はロック | 復号失敗はアクセス拒否 |

---

## 2. 入力検証

### 入力検証の戦略

```
外部入力 (信頼できないデータ)
    │
    ▼
┌──────────────────────┐
│ 1. 型チェック         │  文字列? 数値? 配列? null?
│    (Type Validation) │  → 型が異なれば即リジェクト
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. 長さ制限           │  最大長・最小長
│    (Length Check)     │  → バッファオーバーフロー防止
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 3. 範囲チェック       │  最小値・最大値
│    (Range Check)     │  → 整数オーバーフロー防止
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 4. パターン検証       │  ホワイトリスト正規表現
│    (Pattern Match)   │  → ブラックリストより安全
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 5. ビジネスロジック   │  整合性・妥当性
│    (Business Rules)  │  → ドメイン固有の制約
└──────────┬───────────┘
           │
           ▼
安全な内部データ (Trusted Data)
```

### 入力検証の WHY: ホワイトリスト vs ブラックリスト

ブラックリスト方式（「危険な文字列を除外する」）は回避手法が無数に存在するため、必ずホワイトリスト方式（「許可する文字・パターンのみ受理する」）を採用する。

```python
import re

# NG: ブラックリスト方式 — 回避が容易
def validate_input_blacklist(user_input: str) -> str:
    """危険な文字を除去する方式は常に不完全"""
    dangerous = ["<script>", "DROP TABLE", "OR 1=1"]
    for d in dangerous:
        user_input = user_input.replace(d, "")
    return user_input
    # 攻撃者は "<scr<script>ipt>" のような入れ子で回避可能
    # エンコーディング変換(%3Cscript%3E)でも回避可能

# OK: ホワイトリスト方式 — 許可パターンのみ受理
def validate_username(username: str) -> str:
    """ユーザ名は英数字とアンダースコアのみ許可"""
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        raise ValueError(
            "ユーザ名は3-20文字の英数字とアンダースコアのみ使用可能です"
        )
    return username
```

### 型安全な入力検証 (Python / Pydantic)

```python
from pydantic import BaseModel, Field, field_validator, EmailStr
from datetime import date
from typing import Optional
import re

class UserRegistration(BaseModel):
    """型安全な入力バリデーション（Pydantic v2）"""
    username: str = Field(
        min_length=3,
        max_length=20,
        pattern=r'^[a-zA-Z0-9_]+$',
        description="英数字とアンダースコアのみ",
    )
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    birth_date: date
    age: int = Field(ge=13, le=150)
    bio: Optional[str] = Field(default=None, max_length=500)

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """パスワード強度の検証"""
        if not re.search(r'[A-Z]', v):
            raise ValueError('大文字を1文字以上含む必要があります')
        if not re.search(r'[a-z]', v):
            raise ValueError('小文字を1文字以上含む必要があります')
        if not re.search(r'\d', v):
            raise ValueError('数字を1文字以上含む必要があります')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('特殊文字を1文字以上含む必要があります')
        return v

    @field_validator('birth_date')
    @classmethod
    def validate_birth_date(cls, v: date) -> date:
        """未来の日付や非現実的な日付を拒否"""
        if v > date.today():
            raise ValueError('生年月日は未来の日付にできません')
        if v.year < 1900:
            raise ValueError('無効な生年月日です')
        return v

# 使用例
try:
    user = UserRegistration(
        username="john_doe",
        email="john@example.com",
        password="SecureP@ss123!",
        birth_date="1990-01-15",
        age=34,
    )
    print(f"検証成功: {user.username}")
except Exception as e:
    print(f"検証エラー: {e}")
```

### SQL インジェクション防止

```python
import sqlite3
from typing import Optional

# NG: 文字列結合による SQL 構築
def get_user_bad(username: str) -> Optional[dict]:
    """脆弱なコード — SQL インジェクションが可能"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE name = '{username}'"
    # 入力: ' OR '1'='1' --
    # 生成SQL: SELECT * FROM users WHERE name = '' OR '1'='1' --'
    # → 全ユーザの情報が取得される
    cursor.execute(query)
    return cursor.fetchone()

# OK: パラメータ化クエリ (プリペアドステートメント)
def get_user_good(username: str) -> Optional[dict]:
    """安全なコード — パラメータがクエリ構造と分離"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    # プレースホルダを使用 (DBドライバが安全にエスケープ)
    query = "SELECT * FROM users WHERE name = ?"
    cursor.execute(query, (username,))
    row = cursor.fetchone()
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None

# OK: ORM を使用 (SQLAlchemy)
from sqlalchemy import select
from sqlalchemy.orm import Session

def get_user_orm(session: Session, username: str) -> Optional["User"]:
    """ORM は内部でパラメータ化クエリを生成"""
    stmt = select(User).where(User.name == username)
    return session.execute(stmt).scalar_one_or_none()

# 注意: ORM でも生SQLを使う場合は注意が必要
from sqlalchemy import text

def search_users_safe(session: Session, search_term: str) -> list:
    """ORM の text() でもバインドパラメータを使用する"""
    # NG: f"SELECT * FROM users WHERE name LIKE '%{search_term}%'"
    # OK: バインドパラメータ使用
    stmt = text("SELECT * FROM users WHERE name LIKE :term")
    result = session.execute(stmt, {"term": f"%{search_term}%"})
    return result.fetchall()
```

### コマンドインジェクション防止

```python
import subprocess
import re
import shlex
from pathlib import Path

# NG: シェルコマンドの文字列結合
def ping_bad(host: str) -> str:
    """脆弱なコード — 任意コマンド実行が可能"""
    import os
    os.system(f"ping -c 1 {host}")
    # 入力: "8.8.8.8; rm -rf /" → ping 実行後に全ファイル削除
    # 入力: "8.8.8.8 && cat /etc/passwd" → パスワードファイル読み取り
    return "done"

# OK: subprocess + リスト引数 (shell=False) + ホワイトリスト
def ping_good(host: str) -> str:
    """安全なコード — 引数が分離されシェル解釈されない"""
    # Step 1: ホワイトリストで入力を検証
    if not re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host):
        raise ValueError(f"無効なIPアドレス形式: {host}")

    # Step 2: 各オクテットの範囲チェック
    octets = host.split('.')
    for octet in octets:
        if not 0 <= int(octet) <= 255:
            raise ValueError(f"無効なIPアドレス: {host}")

    # Step 3: shell=False でリスト引数として実行
    result = subprocess.run(
        ["ping", "-c", "1", "-W", "3", host],  # リスト形式
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout

# OK: 外部コマンドの代わりに Python ライブラリを使用（最も推奨）
import socket

def check_host_reachable(host: str, port: int = 80, timeout: float = 3.0) -> bool:
    """外部コマンドを使わず Python で直接ネットワーク確認"""
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
```

### パストラバーサル防止

```python
import os
from pathlib import Path

# NG: ユーザ入力をパスに直接使用
def read_file_bad(filename: str) -> str:
    """脆弱なコード — 任意のファイルが読める"""
    path = f"/app/uploads/{filename}"
    # 入力: "../../etc/passwd" → /etc/passwd を読み取れる
    # 入力: "....//....//etc/passwd" → 正規化前に結合される
    return open(path).read()

# OK: パスの正規化と検証
def read_file_good(filename: str) -> str:
    """安全なコード — ベースディレクトリ外へのアクセスを防止"""
    base_dir = Path("/app/uploads").resolve()  # 絶対パスに正規化
    # ファイル名からディレクトリトラバーサル文字を除去
    safe_filename = Path(filename).name  # ディレクトリ部分を除去
    full_path = (base_dir / safe_filename).resolve()

    # ベースディレクトリ外へのアクセスを拒否
    if not str(full_path).startswith(str(base_dir)):
        raise PermissionError("アクセス拒否: パストラバーサルが検出されました")

    if not full_path.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {safe_filename}")

    # ファイルサイズの制限チェック
    if full_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
        raise ValueError("ファイルサイズが制限を超えています")

    return full_path.read_text(encoding='utf-8')
```

### SSRF (Server-Side Request Forgery) 防止

```python
import ipaddress
import urllib.parse
from typing import Optional

def validate_url_for_ssrf(url: str) -> str:
    """SSRF を防止するURL検証"""
    parsed = urllib.parse.urlparse(url)

    # スキーマの制限
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"許可されないスキーマ: {parsed.scheme}")

    # ホスト名の解決
    import socket
    try:
        resolved_ip = socket.gethostbyname(parsed.hostname)
    except socket.gaierror:
        raise ValueError(f"ホスト名を解決できません: {parsed.hostname}")

    # プライベートIPアドレスの拒否
    ip = ipaddress.ip_address(resolved_ip)
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        raise ValueError(
            f"内部ネットワークへのアクセスは禁止されています: {resolved_ip}"
        )

    # AWS メタデータエンドポイントの拒否
    if resolved_ip == "169.254.169.254":
        raise ValueError("メタデータサービスへのアクセスは禁止されています")

    return url

# 使用例
import requests

def fetch_external_resource(url: str) -> Optional[str]:
    """安全な外部リソース取得"""
    validated_url = validate_url_for_ssrf(url)
    response = requests.get(
        validated_url,
        timeout=10,
        allow_redirects=False,  # リダイレクトでSSRFを回避されるのを防止
    )
    response.raise_for_status()
    return response.text
```

---

## 3. 出力エンコード

### コンテキスト別エスケープの WHY

XSS 攻撃は、ユーザ入力がそのまま HTML に埋め込まれることで発生する。防御の基本は「出力時に適切なエスケープを行うこと」である。ただし、エスケープ方法はHTMLの「どの位置」に出力するかによって異なる。

```
┌───────────────────────────────────────────────────────────────┐
│  コンテキスト        │  エスケープ方法          │  理由         │
│─────────────────────┼────────────────────────┼──────────────│
│  HTML 本文           │  & → &amp;             │  タグ生成防止  │
│  <p>{{ここ}}</p>    │  < → &lt; > → &gt;     │              │
│─────────────────────┼────────────────────────┼──────────────│
│  HTML 属性           │  " → &quot;            │  属性脱出防止  │
│  <div id="{{ここ}}"> │  ' → &#x27;           │              │
│─────────────────────┼────────────────────────┼──────────────│
│  JavaScript          │  Unicode エスケープ      │  コード注入防止│
│  var x = '{{ここ}}'; │  \uXXXX 形式           │              │
│─────────────────────┼────────────────────────┼──────────────│
│  URL パラメータ       │  パーセントエンコーディング│  URL構造破壊防止│
│  href="?q={{ここ}}"  │  encodeURIComponent    │              │
│─────────────────────┼────────────────────────┼──────────────│
│  CSS                 │  CSS エスケープ          │  スタイル注入防止│
│  style="color:{{}}"; │  \HH 形式              │              │
│─────────────────────┼────────────────────────┼──────────────│
│  SQL                 │  パラメータ化クエリ       │  SQL注入防止   │
│  WHERE x = {{ここ}}  │  (エスケープ不要)        │              │
└───────────────────────────────────────────────────────────────┘
```

### XSS 防止の実装パターン

```javascript
// ========================================
// パターン1: React (JSX の自動エスケープ)
// ========================================
function UserProfile({ user }) {
  return (
    <div>
      {/* OK: JSX は自動的に HTML エスケープする */}
      <h1>{user.name}</h1>
      <p>{user.bio}</p>

      {/* NG: dangerouslySetInnerHTML は XSS リスク */}
      <div dangerouslySetInnerHTML={{ __html: user.bio }} />

      {/* OK: サニタイズライブラリ (DOMPurify) を通す */}
      <div dangerouslySetInnerHTML={{
        __html: DOMPurify.sanitize(user.bio, {
          ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
          ALLOWED_ATTR: ['href', 'target', 'rel'],
          FORBID_ATTR: ['style', 'onerror', 'onclick'],
        })
      }} />

      {/* NG: href に javascript: プロトコルが入る可能性 */}
      <a href={user.website}>Website</a>

      {/* OK: URL スキーマを検証 */}
      <a href={sanitizeUrl(user.website)}>Website</a>
    </div>
  );
}

/**
 * URL のスキーマを検証して javascript: プロトコル等を排除
 */
function sanitizeUrl(url) {
  if (!url) return '#';
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
      return '#';
    }
    return url;
  } catch {
    return '#';
  }
}
```

```python
# ========================================
# パターン2: サーバサイド (Python/Flask)
# ========================================
from markupsafe import escape, Markup
from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/profile/<username>')
def profile(username: str):
    # Jinja2 テンプレートは {{ }} 内を自動エスケープ
    template = """
    <h1>{{ username }}</h1>
    <p>{{ bio }}</p>
    """
    # username に <script>alert(1)</script> が入っても安全
    return render_template_string(template, username=username, bio="Hello")

# 手動エスケープが必要な場合
def escape_for_html(text: str) -> str:
    """手動 HTML エスケープ"""
    return str(escape(text))

# NG: |safe フィルターの誤用
# {{ user_input | safe }}  → エスケープされない！
# OK: サニタイズ済みデータにのみ |safe を使用
# {{ sanitized_html | safe }}
```

```javascript
// ========================================
// パターン3: DOM ベース XSS の防止
// ========================================

// NG: DOM に直接 innerHTML を代入
function displaySearchResults_bad(query) {
  // URL: ?q=<img src=x onerror=alert(1)>
  document.getElementById('results').innerHTML =
    `検索結果: ${query}`;  // XSS!
}

// OK: textContent を使用（HTMLとして解釈されない）
function displaySearchResults_good(query) {
  const resultsEl = document.getElementById('results');
  resultsEl.textContent = `検索結果: ${query}`;
}

// OK: DOM API で安全に要素を構築
function displaySearchResults_best(query) {
  const resultsEl = document.getElementById('results');
  const textNode = document.createTextNode(`検索結果: ${query}`);
  resultsEl.replaceChildren(textNode);
}
```

### Content Security Policy (CSP) の詳細設計

```
# ========================================
# レベル1: 基本的な CSP (開始点)
# ========================================
Content-Security-Policy:
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;

# ========================================
# レベル2: 厳格な CSP (推奨)
# ========================================
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

# ========================================
# レベル3: Strict CSP (最も安全)
# ========================================
Content-Security-Policy:
  script-src 'nonce-{RANDOM}' 'strict-dynamic';
  object-src 'none';
  base-uri 'none';
  require-trusted-types-for 'script';
```

```python
# CSP ヘッダーの実装 (Flask)
import secrets

@app.after_request
def add_security_headers(response):
    """セキュリティヘッダーを全レスポンスに付与"""
    nonce = secrets.token_urlsafe(32)

    csp_directives = [
        "default-src 'none'",
        f"script-src 'self' 'nonce-{nonce}'",
        "style-src 'self'",
        "img-src 'self' https://cdn.example.com",
        "font-src 'self'",
        "connect-src 'self' https://api.example.com",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "upgrade-insecure-requests",
    ]
    response.headers['Content-Security-Policy'] = '; '.join(csp_directives)

    # その他のセキュリティヘッダー
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '0'  # 古いブラウザ対策は無効化
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=()'
    )

    # nonce をテンプレートに渡す
    response.headers['X-Nonce'] = nonce
    return response
```

### CSP 違反レポートの収集

```python
# CSP Report-Only モードで段階的導入
@app.after_request
def add_csp_report_only(response):
    """CSP を Report-Only モードで試験運用"""
    response.headers['Content-Security-Policy-Report-Only'] = (
        "default-src 'self'; "
        "report-uri /api/csp-report; "
        "report-to csp-endpoint"
    )
    return response

@app.route('/api/csp-report', methods=['POST'])
def csp_report():
    """CSP 違反レポートを収集"""
    import json
    import logging
    logger = logging.getLogger('csp')

    report = json.loads(request.data)
    logger.warning(
        "CSP Violation: blocked_uri=%s, violated_directive=%s, document_uri=%s",
        report.get('csp-report', {}).get('blocked-uri'),
        report.get('csp-report', {}).get('violated-directive'),
        report.get('csp-report', {}).get('document-uri'),
    )
    return '', 204
```

---

## 4. エラーハンドリング

### 安全なエラーレスポンス設計の WHY

エラーメッセージには2つの相反する要求がある。開発者にはデバッグのために詳細な情報が必要だが、攻撃者にはその情報が攻撃の手がかりになる。この両立を「環境別エラーレスポンス」と「相関ID」で実現する。

```
┌───────────────────────────────────────────────────────────────┐
│                  エラーレスポンスの設計                          │
│───────────────────────────────────────────────────────────────│
│                                                               │
│  開発環境 (debug=True):                                       │
│  {                                                            │
│    "error": "DatabaseError",                                  │
│    "message": "relation \"users\" does not exist",            │
│    "stack": "at Query.run (/app/db.js:42:15)...",             │
│    "query": "SELECT * FROM users WHERE...",                   │
│    "request": { "method": "POST", "path": "/api/users" }     │
│  }                                                            │
│  → 開発者がすぐに原因を特定できる                                │
│                                                               │
│  本番環境 (debug=False):                                      │
│  {                                                            │
│    "error": "Internal Server Error",                          │
│    "requestId": "req-a1b2c3d4"                                │
│  }                                                            │
│  → 攻撃者に内部情報を漏洩しない                                 │
│  → requestId でサーバログと紐付けてデバッグ                      │
│                                                               │
│  サーバログ (内部):                                            │
│  [ERROR] request_id=req-a1b2c3d4 user_id=123                  │
│          error=DatabaseError message="relation does not exist" │
│          stack_trace="..." query="SELECT * FROM..."            │
│  → 全詳細情報がログに記録される                                  │
└───────────────────────────────────────────────────────────────┘
```

### 階層化されたエラーハンドリング実装

```python
import logging
import uuid
import traceback
from flask import Flask, jsonify, request, g
from functools import wraps
from datetime import datetime

app = Flask(__name__)
logger = logging.getLogger(__name__)

# ========================================
# エラークラスの階層設計
# ========================================
class AppError(Exception):
    """アプリケーションエラー基底クラス"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        internal_message: str = None,
    ):
        self.message = message              # ユーザに返すメッセージ
        self.status_code = status_code
        self.error_code = error_code        # 機械判読可能なエラーコード
        self.internal_message = internal_message  # ログ用の詳細

class NotFoundError(AppError):
    def __init__(self, resource: str = "Resource", resource_id: str = None):
        internal_msg = f"{resource} not found (id={resource_id})" if resource_id else None
        super().__init__(
            message=f"{resource} not found",
            status_code=404,
            error_code="NOT_FOUND",
            internal_message=internal_msg,
        )

class ValidationError(AppError):
    def __init__(self, details: list):
        super().__init__(
            message="Validation failed",
            status_code=400,
            error_code="VALIDATION_ERROR",
        )
        self.details = details

class AuthenticationError(AppError):
    def __init__(self):
        super().__init__(
            message="Authentication required",
            status_code=401,
            error_code="UNAUTHORIZED",
        )

class RateLimitError(AppError):
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Too many requests",
            status_code=429,
            error_code="RATE_LIMITED",
        )
        self.retry_after = retry_after

# ========================================
# リクエスト追跡ミドルウェア
# ========================================
@app.before_request
def assign_request_id():
    """各リクエストに一意の追跡IDを割り当て"""
    g.request_id = request.headers.get(
        'X-Request-ID',
        str(uuid.uuid4())[:8]
    )
    g.request_start = datetime.utcnow()

# ========================================
# エラーハンドラ
# ========================================
@app.errorhandler(AppError)
def handle_app_error(error):
    """アプリケーションエラーのハンドリング"""
    request_id = getattr(g, 'request_id', 'unknown')

    # 内部ログには詳細を記録
    logger.error(
        "AppError: error_code=%s message=%s request_id=%s path=%s "
        "method=%s internal=%s",
        error.error_code, error.message, request_id,
        request.path, request.method,
        error.internal_message or "N/A",
    )

    # ユーザには最小限の情報のみ返す
    response = {
        "error": error.message,
        "errorCode": error.error_code,
        "requestId": request_id,
    }
    if hasattr(error, 'details'):
        response["details"] = error.details

    resp = jsonify(response)
    resp.status_code = error.status_code

    if hasattr(error, 'retry_after'):
        resp.headers['Retry-After'] = str(error.retry_after)

    return resp

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """予期しないエラーのハンドリング"""
    request_id = getattr(g, 'request_id', 'unknown')

    # 予期しないエラーはスタックトレース付きでログ
    logger.exception(
        "Unexpected error: request_id=%s path=%s method=%s",
        request_id, request.path, request.method,
    )

    # ユーザには一般的なメッセージのみ
    return jsonify({
        "error": "Internal Server Error",
        "errorCode": "INTERNAL_ERROR",
        "requestId": request_id,
    }), 500
```

### 構造化ログとの統合

```python
import json
import logging
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """JSON構造化ログフォーマッタ"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # request_id がある場合は追加
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id

        # 例外情報がある場合
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # セキュリティ上の注意:
        # パスワード、トークン、個人情報をログに含めない
        return json.dumps(log_entry, ensure_ascii=False)

# 使用例
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger = logging.getLogger('app')
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

---

## 5. CSRF 防止

### CSRF (Cross-Site Request Forgery) の仕組み

```
攻撃の流れ:

1. ユーザが bank.example.com にログイン済み（Cookieがブラウザに保存）
2. ユーザが evil.example.com にアクセス
3. evil.example.com に以下のHTMLが仕込まれている:
   <form action="https://bank.example.com/transfer" method="POST">
     <input type="hidden" name="to" value="attacker">
     <input type="hidden" name="amount" value="1000000">
   </form>
   <script>document.forms[0].submit();</script>
4. ブラウザは bank.example.com の Cookie を自動送信
5. bank.example.com はログイン済みユーザからの正規リクエストと判断
6. → 攻撃者への送金が実行される！
```

### CSRF 防止の実装

```python
# ========================================
# 方法1: CSRF トークン (Flask-WTF)
# ========================================
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
csrf = CSRFProtect(app)

# HTMLフォーム内にトークンを埋め込む
# <form method="POST">
#   <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
#   <button type="submit">送信</button>
# </form>

# AJAX リクエストの場合
# <meta name="csrf-token" content="{{ csrf_token() }}">
# <script>
# fetch('/api/data', {
#   method: 'POST',
#   headers: {
#     'X-CSRFToken': document.querySelector('meta[name=csrf-token]').content,
#     'Content-Type': 'application/json',
#   },
#   body: JSON.stringify(data),
# })
# </script>

# ========================================
# 方法2: SameSite Cookie (最も簡単)
# ========================================
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',   # クロスサイトからの POST で Cookie を送信しない
    SESSION_COOKIE_SECURE=True,       # HTTPS のみ
    SESSION_COOKIE_HTTPONLY=True,      # JavaScript からアクセス不可
)

# ========================================
# 方法3: Double Submit Cookie パターン
# ========================================
import secrets

@app.before_request
def set_csrf_cookie():
    """CSRF トークンを Cookie とレスポンスの両方に設定"""
    if 'csrf_token' not in request.cookies:
        g.csrf_token = secrets.token_urlsafe(32)
    else:
        g.csrf_token = request.cookies['csrf_token']

@app.after_request
def add_csrf_cookie(response):
    """CSRF Cookie を設定"""
    response.set_cookie(
        'csrf_token',
        g.csrf_token,
        httponly=False,  # JS から読み取り可能にする
        secure=True,
        samesite='Lax',
    )
    return response
```

---

## 6. 安全なパスワード処理

### パスワードハッシュ化の WHY

パスワードは平文やMD5/SHA256で保存してはならない。なぜなら:
- 平文: データベース侵害で全パスワードが即座に漏洩
- MD5/SHA256: レインボーテーブル攻撃で短時間で逆算可能（GPU1台で1秒間に数十億回のハッシュ計算）
- ソルトなし: 同一パスワードが同一ハッシュになり一括解読が可能

bcrypt/argon2 は「意図的に遅い」ハッシュ関数で、ブルートフォース攻撃のコストを飛躍的に高める。

```python
# ========================================
# 推奨: bcrypt によるパスワードハッシュ化
# ========================================
import bcrypt

def hash_password(password: str) -> bytes:
    """パスワードを安全にハッシュ化"""
    # bcrypt は自動的にソルト(16バイト)を生成
    # rounds=12: 約0.3秒/ハッシュ (ブルートフォース対策)
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

def verify_password(password: str, hashed: bytes) -> bool:
    """パスワードの検証（タイミング攻撃耐性あり）"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# ========================================
# より推奨: argon2 (Password Hashing Competition 優勝)
# ========================================
from argon2 import PasswordHasher

ph = PasswordHasher(
    time_cost=3,       # 反復回数
    memory_cost=65536,  # メモリ使用量(KB) = 64MB
    parallelism=4,      # 並列度
)

def hash_password_argon2(password: str) -> str:
    """Argon2id でパスワードをハッシュ化"""
    return ph.hash(password)

def verify_password_argon2(password: str, hashed: str) -> bool:
    """Argon2id でパスワードを検証"""
    try:
        return ph.verify(hashed, password)
    except Exception:
        return False

# ========================================
# NG: 以下は絶対に使用しない
# ========================================
# import hashlib
# hashlib.md5(password.encode()).hexdigest()      # NG: 高速すぎる + ソルトなし
# hashlib.sha256(password.encode()).hexdigest()    # NG: 高速すぎる + ソルトなし
# hashlib.sha256((salt + password).encode()).hexdigest()  # NG: 高速すぎる
```

### パスワードハッシュアルゴリズムの比較

| アルゴリズム | 速度 | メモリ | ソルト | 推奨度 | 備考 |
|-------------|------|--------|-------|--------|------|
| MD5 | 超高速 | 少量 | なし | 使用禁止 | レインボーテーブルで即座に逆算 |
| SHA-256 | 高速 | 少量 | なし | 使用禁止 | GPU で秒間数十億回計算可能 |
| PBKDF2 | 調整可 | 少量 | あり | 許容 | NIST推奨だがGPU耐性が低い |
| bcrypt | 調整可 | 4KB | あり | 推奨 | 実績豊富、最大72バイト制限 |
| scrypt | 調整可 | 調整可 | あり | 推奨 | メモリハードだが設定が複雑 |
| Argon2id | 調整可 | 調整可 | あり | 最推奨 | PHC優勝、GPU/ASIC耐性最高 |

---

## 7. セキュアなセッション管理

```python
from flask import Flask, session
import os

app = Flask(__name__)

# ========================================
# セッション設定のベストプラクティス
# ========================================
app.config.update(
    SECRET_KEY=os.environ['SESSION_SECRET'],   # 十分な長さのランダム値(32バイト以上)
    SESSION_COOKIE_SECURE=True,                # HTTPS でのみ送信
    SESSION_COOKIE_HTTPONLY=True,               # JavaScript からアクセス不可
    SESSION_COOKIE_SAMESITE='Lax',             # CSRF 防止
    PERMANENT_SESSION_LIFETIME=1800,           # 30分で期限切れ
    SESSION_COOKIE_NAME='__Host-session',      # __Host- プレフィックスで保護強化
)

# ========================================
# セッション固定攻撃の防止
# ========================================
@app.route('/login', methods=['POST'])
def login():
    """ログイン成功時にセッションIDを再生成"""
    # 認証処理...
    if authenticate(request.form['email'], request.form['password']):
        # 重要: ログイン成功後にセッションを再生成
        session.clear()  # 古いセッションデータを破棄
        session.regenerate()  # 新しいセッションIDを生成
        session['user_id'] = user.id
        session['login_time'] = datetime.utcnow().isoformat()
        return redirect('/dashboard')
    return render_template('login.html', error="Invalid credentials")

# ========================================
# セッションの妥当性検証
# ========================================
@app.before_request
def validate_session():
    """各リクエストでセッションの妥当性を確認"""
    if 'user_id' in session:
        login_time = datetime.fromisoformat(session.get('login_time', ''))
        # 長時間アイドルの場合はセッションを無効化
        if (datetime.utcnow() - login_time).total_seconds() > 3600:
            session.clear()
            return redirect('/login?reason=timeout')
```

---

## 8. デシリアライゼーション攻撃の防止

```python
# ========================================
# NG: pickle による安全でないデシリアライゼーション
# ========================================
import pickle

# 絶対にやってはいけない: 信頼できないデータを pickle.loads
def load_user_data_bad(serialized_data: bytes):
    """脆弱: pickle は任意のPythonオブジェクトを復元する"""
    return pickle.loads(serialized_data)
    # 攻撃者は __reduce__ メソッドを持つオブジェクトを送信し
    # os.system('rm -rf /') などの任意コマンドを実行できる

# ========================================
# OK: JSON を使用する（安全なシリアライゼーション）
# ========================================
import json
from typing import Any

def load_user_data_good(serialized_data: str) -> dict:
    """安全: JSON はデータのみを復元する（コード実行不可）"""
    data = json.loads(serialized_data)
    # さらに型を検証
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data

# ========================================
# OK: YAML の安全な読み込み
# ========================================
import yaml

def load_config_safe(yaml_content: str) -> dict:
    """安全: safe_load は基本型のみ許可"""
    # NG: yaml.load(yaml_content, Loader=yaml.FullLoader)  # コード実行リスク
    # OK: yaml.safe_load は str, int, list, dict のみ
    return yaml.safe_load(yaml_content)
```

---

## 9. アンチパターン集

### アンチパターン 1: クライアント側のみの検証

```javascript
// NG: フロントエンドのみでバリデーション
async function submitForm() {
  const age = document.getElementById('age').value;
  if (age < 0 || age > 150) {
    alert('無効な年齢です');
    return;
  }
  // 攻撃者は DevTools, curl, Burp Suite で直接 API を叩ける
  await fetch('/api/users', {
    method: 'POST',
    body: JSON.stringify({ age: -999 }),  // バリデーションを完全にバイパス
  });
}
```

```python
# OK: サーバ側でも必ず検証
from pydantic import BaseModel, Field

class UserUpdate(BaseModel):
    age: int = Field(ge=0, le=150)  # サーバ側で型+範囲を強制

@app.route('/api/users', methods=['POST'])
def create_user():
    try:
        data = UserUpdate(**request.json)  # 検証失敗は自動で422
    except ValueError as e:
        return jsonify({"error": "Validation failed", "details": str(e)}), 400
    # フロントエンド検証はUXのため、セキュリティはサーバ側の責任
```

### アンチパターン 2: エラーメッセージでの情報漏洩

```python
# NG: ユーザ列挙攻撃を可能にするエラーメッセージ
@app.route('/login', methods=['POST'])
def login_bad():
    user = User.query.filter_by(email=request.form['email']).first()
    if not user:
        return {"error": "このメールアドレスは登録されていません"}, 401
    if not verify_password(request.form['password'], user.password_hash):
        return {"error": "パスワードが間違っています"}, 401
    # → 攻撃者はエラーメッセージからユーザの存在を確認できる
    # → メールアドレスのリストを作成し、パスワードスプレー攻撃に利用

# OK: 同一メッセージを返す + タイミング攻撃も防止
import time
import secrets

@app.route('/login', methods=['POST'])
def login_good():
    user = User.query.filter_by(email=request.form['email']).first()
    if not user:
        # ユーザが存在しなくてもパスワード検証と同等の時間をかける
        bcrypt.hashpw(b"dummy", bcrypt.gensalt(rounds=12))
        return {"error": "メールアドレスまたはパスワードが無効です"}, 401

    if not verify_password(request.form['password'], user.password_hash):
        return {"error": "メールアドレスまたはパスワードが無効です"}, 401

    # 成功
    return {"token": generate_jwt(user)}
```

### アンチパターン 3: ハードコードされた秘密情報

```python
# NG: ソースコードに秘密情報を直接記述
SECRET_KEY = "my-super-secret-key-12345"
DATABASE_URL = "postgresql://admin:password123@db.example.com/prod"
API_KEY = "sk-live-abcdef1234567890"

# OK: 環境変数やシークレットマネージャから取得
import os

SECRET_KEY = os.environ['SECRET_KEY']
DATABASE_URL = os.environ['DATABASE_URL']

# さらに安全: AWS Secrets Manager から取得
import boto3
import json

def get_secret(secret_name: str) -> dict:
    """AWS Secrets Manager からシークレットを取得"""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

### アンチパターン 4: 不適切な正規表現による ReDoS

```python
import re

# NG: 壊滅的バックトラッキングを引き起こす正規表現
# "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!" のような入力で指数時間かかる
bad_pattern = re.compile(r'^(a+)+$')

# OK: 非バックトラッキングな正規表現、または入力長を制限
def validate_safe(text: str) -> bool:
    if len(text) > 100:  # 入力長を制限
        return False
    # 原子的グループやポゼッシブ量指定子を使えない場合は
    # パターンを単純化する
    return bool(re.match(r'^a+$', text))
```

---

## 10. セキュリティチェックリスト

### 開発時のセキュリティチェックリスト

```
┌──────────────────────────────────────────────────────┐
│  カテゴリ       │  チェック項目                        │
│─────────────────┼───────────────────────────────────│
│  入力検証        │  □ 全入力にサーバ側バリデーション     │
│                 │  □ ホワイトリスト方式を使用           │
│                 │  □ 入力長の上限を設定                │
│                 │  □ ファイルアップロードの型・サイズ制限│
│─────────────────┼───────────────────────────────────│
│  出力エンコード   │  □ コンテキスト別のエスケープ         │
│                 │  □ CSP ヘッダーの設定                │
│                 │  □ dangerouslySetInnerHTML 不使用    │
│─────────────────┼───────────────────────────────────│
│  認証            │  □ bcrypt/argon2 でパスワード保存    │
│                 │  □ セッション固定攻撃対策            │
│                 │  □ レート制限の実装                   │
│─────────────────┼───────────────────────────────────│
│  エラー処理      │  □ 本番でスタックトレース非公開       │
│                 │  □ 統一エラーメッセージ(列挙攻撃防止)│
│                 │  □ requestId による追跡             │
│─────────────────┼───────────────────────────────────│
│  データ保護      │  □ シークレットは環境変数/KMS        │
│                 │  □ 通信は全て TLS                   │
│                 │  □ 安全なデシリアライゼーション        │
│─────────────────┼───────────────────────────────────│
│  ヘッダー        │  □ X-Content-Type-Options: nosniff │
│                 │  □ X-Frame-Options: DENY            │
│                 │  □ Strict-Transport-Security        │
│                 │  □ Referrer-Policy                  │
└──────────────────────────────────────────────────────┘
```

---

## 11. 実践演習

### 演習1: 入力バリデーション関数の実装（基礎）

**課題**: 以下の要件を満たすユーザ登録バリデーション関数を Python で実装せよ。
- ユーザ名: 3-20文字、英数字とアンダースコアのみ
- メールアドレス: 標準的なメール形式
- パスワード: 12文字以上、大文字・小文字・数字・記号をそれぞれ1つ以上含む
- 年齢: 13-150の整数
- 全て検証に失敗した場合はどのフィールドが無効かを返す

<details>
<summary>模範解答</summary>

```python
import re
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ValidationResult:
    is_valid: bool
    errors: dict = field(default_factory=dict)

def validate_user_registration(
    username: str,
    email: str,
    password: str,
    age: int,
) -> ValidationResult:
    """ユーザ登録データのバリデーション"""
    errors = {}

    # ユーザ名の検証
    if not isinstance(username, str):
        errors['username'] = '文字列である必要があります'
    elif not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        errors['username'] = '3-20文字の英数字とアンダースコアのみ使用可能です'

    # メールアドレスの検証
    if not isinstance(email, str):
        errors['email'] = '文字列である必要があります'
    elif not re.match(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email
    ):
        errors['email'] = '有効なメールアドレス形式ではありません'
    elif len(email) > 254:
        errors['email'] = 'メールアドレスが長すぎます'

    # パスワードの検証
    if not isinstance(password, str):
        errors['password'] = '文字列である必要があります'
    elif len(password) < 12:
        errors['password'] = '12文字以上必要です'
    elif len(password) > 128:
        errors['password'] = '128文字以下にしてください'
    else:
        pw_errors = []
        if not re.search(r'[A-Z]', password):
            pw_errors.append('大文字')
        if not re.search(r'[a-z]', password):
            pw_errors.append('小文字')
        if not re.search(r'\d', password):
            pw_errors.append('数字')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            pw_errors.append('記号')
        if pw_errors:
            errors['password'] = f'以下を含む必要があります: {", ".join(pw_errors)}'

    # 年齢の検証
    if not isinstance(age, int):
        errors['age'] = '整数である必要があります'
    elif age < 13:
        errors['age'] = '13歳以上である必要があります'
    elif age > 150:
        errors['age'] = '無効な年齢です'

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)

# テスト
result = validate_user_registration(
    username="john_doe",
    email="john@example.com",
    password="SecureP@ss123!",
    age=25,
)
assert result.is_valid
assert result.errors == {}

result2 = validate_user_registration(
    username="ab",
    email="invalid",
    password="short",
    age=5,
)
assert not result2.is_valid
assert 'username' in result2.errors
assert 'email' in result2.errors
assert 'password' in result2.errors
assert 'age' in result2.errors
print("全テストパス")
```

</details>

### 演習2: セキュアなファイルアップロード API（応用）

**課題**: 以下の要件を満たす安全なファイルアップロード API を Flask で実装せよ。
- 許可する拡張子: .jpg, .png, .gif のみ
- ファイルサイズ上限: 5MB
- ファイル名はランダムUUIDに置換（元のファイル名を使用しない）
- MIMEタイプのマジックバイト検証
- パストラバーサル防止
- アップロード先ディレクトリの存在確認

<details>
<summary>模範解答</summary>

```python
import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_DIR = Path("/app/uploads")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif'}
ALLOWED_MIME_TYPES = {
    'image/jpeg': b'\xff\xd8\xff',
    'image/png': b'\x89PNG',
    'image/gif': b'GIF8',
}

def validate_file_magic_bytes(file_data: bytes, extension: str) -> bool:
    """ファイルのマジックバイトを検証"""
    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
    }
    expected_mime = mime_map.get(extension)
    if not expected_mime:
        return False
    expected_magic = ALLOWED_MIME_TYPES.get(expected_mime)
    if not expected_magic:
        return False
    return file_data[:len(expected_magic)] == expected_magic

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """安全なファイルアップロード"""
    # ファイルの存在確認
    if 'file' not in request.files:
        return jsonify({"error": "ファイルが指定されていません"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイル名が空です"}), 400

    # 拡張子の検証
    original_ext = Path(file.filename).suffix.lower()
    if original_ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"許可されていない拡張子です。許可: {ALLOWED_EXTENSIONS}"
        }), 400

    # ファイルサイズの検証
    file_data = file.read()
    if len(file_data) > MAX_FILE_SIZE:
        return jsonify({
            "error": f"ファイルサイズが制限({MAX_FILE_SIZE // 1024 // 1024}MB)を超えています"
        }), 400

    if len(file_data) == 0:
        return jsonify({"error": "ファイルが空です"}), 400

    # マジックバイトの検証
    if not validate_file_magic_bytes(file_data, original_ext):
        return jsonify({
            "error": "ファイルの内容が拡張子と一致しません"
        }), 400

    # 安全なファイル名を生成（UUID + 元の拡張子）
    safe_filename = f"{uuid.uuid4()}{original_ext}"
    upload_path = (UPLOAD_DIR / safe_filename).resolve()

    # パストラバーサル防止
    if not str(upload_path).startswith(str(UPLOAD_DIR.resolve())):
        return jsonify({"error": "不正なパスです"}), 400

    # アップロードディレクトリの確認
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ファイル保存
    upload_path.write_bytes(file_data)

    return jsonify({
        "message": "アップロード成功",
        "filename": safe_filename,
        "size": len(file_data),
    }), 201
```

</details>

### 演習3: レート制限付きログインAPI の実装（発展）

**課題**: 以下のセキュリティ要件を全て満たすログインAPIを実装せよ。
- パスワードは argon2id でハッシュ化して保存
- ログイン試行は IP アドレスあたり 5回/15分に制限
- ユーザ列挙攻撃を防止する統一エラーメッセージ
- タイミング攻撃を防止する一定時間レスポンス
- ログイン成功時にセッションIDを再生成
- 全試行をログに記録（ただしパスワードはログに含めない）

<details>
<summary>模範解答</summary>

```python
import time
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
logger = logging.getLogger('auth')

ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)

# レート制限のための簡易ストレージ（本番では Redis を使用）
login_attempts = defaultdict(list)

RATE_LIMIT_WINDOW = 15 * 60  # 15分
RATE_LIMIT_MAX = 5            # 最大5回

# ダミーのユーザDB（実際はデータベースを使用）
USERS_DB = {
    "user@example.com": {
        "id": 1,
        "email": "user@example.com",
        "password_hash": ph.hash("SecureP@ss123!"),
    }
}

def check_rate_limit(ip_address: str) -> bool:
    """IPアドレスベースのレート制限チェック"""
    now = time.time()
    # 古い記録を削除
    login_attempts[ip_address] = [
        t for t in login_attempts[ip_address]
        if now - t < RATE_LIMIT_WINDOW
    ]
    return len(login_attempts[ip_address]) < RATE_LIMIT_MAX

def record_attempt(ip_address: str):
    """ログイン試行を記録"""
    login_attempts[ip_address].append(time.time())

def constant_time_response(start_time: float, min_duration: float = 0.3):
    """タイミング攻撃を防止するため一定時間待機"""
    elapsed = time.time() - start_time
    if elapsed < min_duration:
        time.sleep(min_duration - elapsed)

@app.route('/api/login', methods=['POST'])
def login():
    """セキュアなログインAPI"""
    start_time = time.time()
    client_ip = request.remote_addr
    request_id = str(uuid.uuid4())[:8]

    # レート制限チェック
    if not check_rate_limit(client_ip):
        logger.warning(
            "Rate limit exceeded: ip=%s request_id=%s",
            client_ip, request_id,
        )
        remaining_time = int(RATE_LIMIT_WINDOW - (
            time.time() - min(login_attempts[client_ip])
        ))
        constant_time_response(start_time)
        return jsonify({
            "error": "試行回数の上限に達しました。しばらく待ってから再試行してください。",
            "requestId": request_id,
        }), 429, {'Retry-After': str(remaining_time)}

    # 入力の取得と検証
    data = request.get_json(silent=True)
    if not data or 'email' not in data or 'password' not in data:
        constant_time_response(start_time)
        return jsonify({
            "error": "メールアドレスとパスワードを入力してください。",
            "requestId": request_id,
        }), 400

    email = data['email']
    password = data['password']

    # ログ記録（パスワードは含めない）
    logger.info(
        "Login attempt: email=%s ip=%s request_id=%s",
        email, client_ip, request_id,
    )

    # ユーザ検索
    user = USERS_DB.get(email)

    if not user:
        # ユーザが存在しなくてもパスワード検証と同等の時間をかける
        try:
            ph.verify(ph.hash("dummy"), "dummy")
        except VerifyMismatchError:
            pass
        record_attempt(client_ip)
        logger.warning(
            "Login failed (user not found): email=%s ip=%s request_id=%s",
            email, client_ip, request_id,
        )
        constant_time_response(start_time)
        # 統一エラーメッセージ
        return jsonify({
            "error": "メールアドレスまたはパスワードが無効です。",
            "requestId": request_id,
        }), 401

    # パスワード検証
    try:
        ph.verify(user['password_hash'], password)
    except VerifyMismatchError:
        record_attempt(client_ip)
        logger.warning(
            "Login failed (wrong password): email=%s ip=%s request_id=%s",
            email, client_ip, request_id,
        )
        constant_time_response(start_time)
        return jsonify({
            "error": "メールアドレスまたはパスワードが無効です。",
            "requestId": request_id,
        }), 401

    # ログイン成功
    session.clear()  # セッション固定攻撃防止
    session['user_id'] = user['id']
    session['login_time'] = datetime.utcnow().isoformat()
    session['ip'] = client_ip

    # 成功ログ
    logger.info(
        "Login success: user_id=%s email=%s ip=%s request_id=%s",
        user['id'], email, client_ip, request_id,
    )

    # パスワードハッシュのリハッシュ（パラメータ更新時）
    if ph.check_needs_rehash(user['password_hash']):
        user['password_hash'] = ph.hash(password)
        logger.info("Password rehashed: user_id=%s", user['id'])

    constant_time_response(start_time)
    return jsonify({
        "message": "ログイン成功",
        "requestId": request_id,
    }), 200
```

</details>

---

## 12. FAQ

### Q1. 入力検証はフロントエンドとバックエンドの両方に必要か?

はい、両方に必要である。ただし目的が異なる。フロントエンドの検証は**UX改善**のためであり、ユーザにリアルタイムでフィードバックを返す。バックエンドの検証は**セキュリティ**のためであり、攻撃者は curl や Burp Suite で直接バックエンドに任意のリクエストを送信できるため、フロントエンド検証は完全にバイパスされる。したがってバックエンドでの検証が唯一の信頼できる防御線である。

### Q2. ORM を使えば SQL インジェクションは完全に防げるか?

ORM の標準的な API（`filter_by()`, `where()` 等）を使う限りはほぼ安全である。しかし、ORM にも生 SQL を組み立てる機能がある（SQLAlchemy の `text()`, Django の `raw()`, `extra()`）。これらを使用する場合はパラメータ化クエリと同じ注意が必要である。また、LIKE 句のワイルドカード (`%`, `_`) が特殊文字として機能する点も注意が必要で、ユーザ入力をそのまま LIKE パターンに使用するとデータ漏洩のリスクがある。

### Q3. Content Security Policy (CSP) はどこまで厳格にすべきか?

`default-src 'none'` から始めて、必要なリソースのみを明示的に許可するのが理想的である。段階的に導入するには、まず `Content-Security-Policy-Report-Only` ヘッダーで「レポート専用モード」で運用し、違反レポートを収集しながら正しいポリシーを策定する。nonce ベースの `script-src` を採用し、`unsafe-inline` や `unsafe-eval` は避ける。CDN からのスクリプト読み込みが必要な場合は SRI (Subresource Integrity) ハッシュと併用する。

### Q4. タイミング攻撃とは何か? なぜパスワード比較で問題になるか?

タイミング攻撃とは、処理にかかる時間の差から秘密情報を推測する攻撃である。例えば、文字列比較が最初に異なる文字を見つけた時点で `False` を返す実装（短絡評価）の場合、攻撃者は比較に要する時間からパスワードの一致文字数を推測できる。`bcrypt.checkpw()` や `hmac.compare_digest()` は内部で定数時間比較を行うためこの攻撃に耐性がある。自前で文字列比較を行う場合は必ず定数時間比較を使用すること。

### Q5. 安全でないデシリアライゼーションを検出するにはどうすればよいか?

SAST ツール（Semgrep, Bandit）で `pickle.loads()`, `yaml.load()`, `eval()`, `exec()` の使用箇所を自動検出する。コードレビューでは「信頼できないデータソースから復元されるオブジェクト」を重点的に確認する。代替として JSON や Protocol Buffers のような安全なシリアライゼーション形式を使用する。詳細は [SAST/DAST](./03-sast-dast.md) を参照。

---

## まとめ

| 項目 | 要点 | 対策手法 |
|------|------|---------|
| 入力検証 | ホワイトリスト方式、サーバ側で必ず実施 | Pydantic, Joi, Bean Validation |
| SQL インジェクション | パラメータ化クエリまたは ORM を使用 | PreparedStatement, ORM |
| XSS 防止 | コンテキスト別エスケープ + CSP | DOMPurify, テンプレートエンジン |
| CSRF 防止 | トークンベース + SameSite Cookie | Flask-WTF, csurf |
| エラーハンドリング | 本番では詳細を隠し requestId で追跡 | 構造化ログ + 相関ID |
| パスワード | bcrypt/argon2 でハッシュ化 | argon2id 推奨 |
| セッション | Secure + HttpOnly + SameSite 属性必須 | セッション固定攻撃防止 |
| SSRF 防止 | プライベートIP・メタデータAPIへのアクセス拒否 | URL検証 + allowlist |
| デシリアライゼーション | pickle/eval を信頼できないデータに使わない | JSON/Protobuf を使用 |
| ReDoS | 正規表現の複雑さ制限 + 入力長制限 | re2ライブラリの使用 |

---

## 次に読むべきガイド

- [依存関係セキュリティ](./01-dependency-security.md) — サードパーティライブラリの脆弱性管理と SCA
- [SAST/DAST](./03-sast-dast.md) — Semgrep/SonarQube による自動化されたコードセキュリティ検査
- [コンテナセキュリティ](./02-container-security.md) — コンテナ化されたアプリケーションのセキュリティ
- [APIセキュリティ](../03-network-security/02-api-security.md) — API レベルの認証・認可・レート制限
- [認証の基礎](../../authentication-and-authorization/00-fundamentals/) — 認証・認可の体系的理解
- [暗号化基礎](../02-cryptography/) — ハッシュ関数・暗号化の理論的背景

---

## 参考文献

1. **OWASP Secure Coding Practices Quick Reference Guide** — https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/
2. **OWASP Cheat Sheet Series** — https://cheatsheetseries.owasp.org/
3. **CWE/SANS Top 25 Most Dangerous Software Weaknesses** — https://cwe.mitre.org/top25/
4. **Mozilla Web Security Guidelines** — https://infosec.mozilla.org/guidelines/web_security
5. **NIST SP 800-63B — Digital Identity Guidelines: Authentication** — https://pages.nist.gov/800-63-3/sp800-63b.html
6. **Google Application Security** — https://cloud.google.com/security/application-security
