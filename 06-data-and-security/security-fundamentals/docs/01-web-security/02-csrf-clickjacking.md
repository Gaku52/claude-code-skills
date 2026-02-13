# CSRF/クリックジャッキング

> CSRF（クロスサイトリクエストフォージェリ）とクリックジャッキングの攻撃メカニズムを解説し、トークン方式、SameSite Cookie、X-Frame-Optionsによる防御を実装する。

## 前提知識

- HTTP Cookie とセッション管理の基本（[04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md)）
- Same-Origin Policy の概念（[01-xss-prevention.md](./01-xss-prevention.md)）
- HTML フォームと HTTP メソッド（GET/POST/PUT/DELETE）の基礎

## この章で学ぶこと

1. **CSRF攻撃**のメカニズムとトークンベースの防御手法を理解する
2. **SameSite Cookie** 属性による最新のCSRF対策を習得する
3. **クリックジャッキング**の原理とX-Frame-Options/CSP frame-ancestorsによる防御を身につける
4. **Origin/Referer ヘッダー検証**による補助的防御を実装する
5. **CSRF と XSS の組み合わせ攻撃**への対処法を理解する

---

## 1. CSRF（Cross-Site Request Forgery）とは

ユーザーが認証済みのWebサイトに対して、攻撃者が意図しないリクエストを送信させる攻撃。OWASP Top 10 でも継続的にリストされている代表的なWeb脆弱性である。

### 1.1 攻撃の基本メカニズム

CSRFが成立する条件は以下の3つが同時に満たされることである:

1. **Cookie ベースの認証**: ブラウザがリクエスト時に自動的にCookieを送信する
2. **予測可能なリクエスト形式**: 攻撃者がリクエストパラメータを推測できる
3. **副作用のあるアクション**: 状態変更（送金、設定変更、データ削除等）が可能

```
CSRF攻撃のフロー:

  被害者             攻撃者のサイト          銀行サイト
    |                    |                     |
    |-- ログイン済み --->|                     |
    |   (セッション      |                     |
    |    Cookieあり)     |                     |
    |                    |                     |
    |-- 攻撃サイトに --> |                     |
    |   アクセス         |                     |
    |                    |-- 隠しフォームで --> |
    |                    |   送金リクエスト     |
    |                    |   (Cookieが自動送信) |
    |                    |                     |-- 送金実行!
    |                    |                     |   (正規セッション)
```

### 1.2 ブラウザのCookie送信メカニズム（内部動作）

```
ブラウザのCookie送信判定フロー:

  リクエスト発生
       |
       v
  +------------------+
  | Cookie ストアから |
  | ドメイン一致する  |
  | Cookieを検索     |
  +------------------+
       |
       v
  +------------------+     いいえ
  | Secure属性あり?  |---------> HTTPSでない場合は送信しない
  +------------------+
       |はい/なし
       v
  +------------------+     いいえ
  | Path一致?        |---------> 送信しない
  +------------------+
       |はい
       v
  +------------------+
  | SameSite属性     |
  | チェック         |
  +------------------+
       |
  +----+----+----+
  |         |         |
  v         v         v
Strict    Lax      None
  |         |         |
  v         v         v
同一サイト  同一サイト  常に送信
のみ送信   +トップレベル (Secure必須)
           ナビゲーション
           のGETのみ送信
```

### 1.3 CSRF攻撃の具体的な手法

```
攻撃手法のバリエーション:

1. 隠しフォーム自動送信:
   <form action="https://bank.com/transfer" method="POST">
     <input type="hidden" name="to" value="attacker">
     <input type="hidden" name="amount" value="1000000">
   </form>
   <script>document.forms[0].submit();</script>

2. イメージタグ（GETリクエスト）:
   <img src="https://bank.com/transfer?to=attacker&amount=1000000">

3. XMLHttpRequest / Fetch API:
   fetch('https://bank.com/api/transfer', {
     method: 'POST',
     credentials: 'include',  // Cookieを含める
     body: JSON.stringify({to: 'attacker', amount: 1000000})
   });
   ※ CORS が正しく設定されていれば通常はブロックされる

4. iframe + フォーム:
   <iframe name="csrf-frame" style="display:none"></iframe>
   <form target="csrf-frame" action="https://bank.com/transfer"
         method="POST">
     ...
   </form>
```

```python
# コード例1: CSRF攻撃の例と防御
from flask import Flask, request, session, abort
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# 脆弱なコード: CSRFトークンなし
@app.route("/transfer", methods=["POST"])
def transfer_vulnerable():
    # Cookieだけで認証 -> CSRF攻撃可能
    to_account = request.form["to"]
    amount = request.form["amount"]
    execute_transfer(session["user_id"], to_account, amount)
    return "送金完了"

# 安全なコード: CSRFトークン検証付き
def generate_csrf_token() -> str:
    """セッションごとのCSRFトークンを生成"""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]

def validate_csrf_token(token: str) -> bool:
    """CSRFトークンを検証"""
    expected = session.get("csrf_token")
    if not expected or not secrets.compare_digest(token, expected):
        return False
    return True

@app.route("/transfer", methods=["POST"])
def transfer_safe():
    # CSRFトークンの検証
    token = request.form.get("csrf_token", "")
    if not validate_csrf_token(token):
        abort(403, "Invalid CSRF token")
    to_account = request.form["to"]
    amount = request.form["amount"]
    execute_transfer(session["user_id"], to_account, amount)
    return "送金完了"
```

---

## 2. CSRFトークン方式

### 2.1 Synchronizer Token Pattern

最も一般的で信頼性の高いCSRF防御手法。サーバー側でランダムなトークンを生成し、フォームに埋め込む。

```
CSRFトークンの流れ:

  ブラウザ                              サーバー
    |                                     |
    |-- GET /transfer (フォーム取得) -->   |
    |                                     |-- トークン生成
    |                                     |   session["csrf"] = "abc123"
    |<-- フォーム + hidden token ------    |
    |   <input type="hidden"              |
    |    name="csrf_token"                |
    |    value="abc123">                  |
    |                                     |
    |-- POST /transfer ------------>      |
    |   csrf_token=abc123                 |-- トークン検証
    |   to=..., amount=...               |   form値 == session値?
    |                                     |-- OK -> 送金実行
```

```
Synchronizer Token の内部実装フロー:

  +------------------+
  | セッション開始    |
  +------------------+
         |
         v
  +------------------+
  | CSPRNG で         |
  | トークン生成      |
  | (32バイト以上)    |
  +------------------+
         |
    +----+----+
    |         |
    v         v
  セッション   フォームの
  ストアに    hidden input
  保存        に埋め込み
    |         |
    +----+----+
         |
         v (POST時)
  +------------------+
  | フォーム値と      |
  | セッション値を    |
  | 定数時間比較      |
  | (timing attack防止)|
  +------------------+
         |
    +----+----+
    |         |
    v         v
  一致:      不一致:
  処理実行   403 Forbidden
```

### 2.2 Double Submit Cookie Pattern

ステートレスなCSRF防御手法。サーバー側でセッションにトークンを保存する必要がない。

```python
# コード例2: Double Submit Cookie パターン
import secrets
import hmac

class DoubleSubmitCSRF:
    """Double Submit Cookie方式のCSRF対策

    原理:
    - ランダムな値をCookieとフォームの両方に設定
    - 攻撃者はCookieは読み取れない（Same-Origin Policy）
    - したがってフォームに正しい値を設定できない

    注意: サブドメイン攻撃に対する耐性を高めるため
    HMAC署名を併用する
    """

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self) -> tuple:
        """CookieとHTMLフォーム用のトークンペアを生成"""
        random_value = secrets.token_hex(32)
        # HMACで署名してCookie値を生成
        signed = hmac.new(
            self.secret_key.encode(),
            random_value.encode(),
            "sha256"
        ).hexdigest()
        return random_value, signed  # (cookie_value, form_value)

    def validate(self, cookie_value: str, form_value: str) -> bool:
        """Cookie値とフォーム値を照合"""
        expected_signed = hmac.new(
            self.secret_key.encode(),
            cookie_value.encode(),
            "sha256"
        ).hexdigest()
        return hmac.compare_digest(expected_signed, form_value)

csrf = DoubleSubmitCSRF("my-secret-key")
cookie_val, form_val = csrf.generate_token()
# cookie_val -> Set-Cookie: csrf=<random_value>
# form_val   -> <input type="hidden" name="csrf_token" value="<signed>">
```

### 2.3 Signed Double Submit Cookie（改良版）

```python
# コード例3: 署名付きDouble Submit Cookie（タイムスタンプ含む）
import time
import hmac
import hashlib
import secrets
import json
import base64

class SignedDoubleSubmitCSRF:
    """改良版 Double Submit Cookie

    従来のDouble Submitの弱点を補完:
    - タイムスタンプで有効期限を設定
    - セッションIDとの紐付けでセッション固定攻撃を防止
    - HMAC-SHA256 で改ざん検知
    """

    def __init__(self, secret_key: str, max_age: int = 3600):
        self.secret_key = secret_key.encode()
        self.max_age = max_age  # トークンの有効期限（秒）

    def generate_token(self, session_id: str) -> str:
        """タイムスタンプ付きCSRFトークンを生成"""
        payload = {
            "sid": session_id,
            "ts": int(time.time()),
            "nonce": secrets.token_hex(16),
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()

        signature = hmac.new(
            self.secret_key,
            payload_b64.encode(),
            hashlib.sha256,
        ).hexdigest()

        return f"{payload_b64}.{signature}"

    def validate_token(self, token: str, session_id: str) -> bool:
        """CSRFトークンを検証"""
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return False
            payload_b64, signature = parts

            # 署名検証
            expected_sig = hmac.new(
                self.secret_key,
                payload_b64.encode(),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                return False

            # ペイロード検証
            payload = json.loads(
                base64.urlsafe_b64decode(payload_b64)
            )

            # セッションID一致確認
            if payload.get("sid") != session_id:
                return False

            # 有効期限確認
            if int(time.time()) - payload.get("ts", 0) > self.max_age:
                return False

            return True
        except Exception:
            return False

# 使用例
csrf = SignedDoubleSubmitCSRF("super-secret-key", max_age=3600)
token = csrf.generate_token(session_id="sess_abc123")
is_valid = csrf.validate_token(token, session_id="sess_abc123")
```

### 2.4 トークン方式の比較

| 方式 | ステートフル | サブドメイン攻撃耐性 | 実装難度 | SPA対応 |
|------|:----------:|:------------------:|:-------:|:------:|
| Synchronizer Token | はい | 高 | 低 | 中 |
| Double Submit Cookie | いいえ | 低 | 中 | 高 |
| Signed Double Submit | いいえ | 高 | 高 | 高 |
| Encrypted Token | いいえ | 高 | 高 | 高 |
| HMAC-based Token | いいえ | 高 | 中 | 高 |

---

## 3. SameSite Cookie

SameSite属性は、ブラウザが第三者サイトからのリクエストにCookieを送信するかどうかを制御する。

### 3.1 SameSite属性の動作原理

```
「同一サイト」の定義:

  eTLD+1 (Effective Top-Level Domain + 1) が一致するかどうか

  例:
  - www.example.com と api.example.com → 同一サイト (eTLD+1 = example.com)
  - example.com と example.org → 異なるサイト
  - a.github.io と b.github.io → 異なるサイト (github.io は eTLD)

  注意: Same-Site と Same-Origin は異なる概念
  - Same-Origin: スキーム + ホスト + ポートが完全一致
  - Same-Site: eTLD+1 が一致
```

```python
# コード例4: SameSite Cookie の設定（各値の詳細）
from flask import Flask, make_response

app = Flask(__name__)

@app.route("/login", methods=["POST"])
def login():
    response = make_response("ログイン成功")

    # 推奨設定: SameSite=Lax
    response.set_cookie(
        "session_id",
        value=generate_session_id(),
        httponly=True,     # JavaScriptからアクセス不可
        secure=True,       # HTTPS通信でのみ送信
        samesite="Lax",    # クロスサイトのGETは許可、POSTは拒否
        max_age=3600,
        path="/",
    )
    return response

@app.route("/set-strict-cookie")
def set_strict():
    """最も厳格な設定（銀行サイト等向け）"""
    response = make_response("OK")
    response.set_cookie(
        "bank_session",
        value=generate_session_id(),
        httponly=True,
        secure=True,
        samesite="Strict",  # クロスサイトからは一切送信しない
        max_age=1800,
        path="/",
        domain=".bank.example.com",  # サブドメインにも適用
    )
    return response

@app.route("/set-none-cookie")
def set_none():
    """サードパーティCookie（OAuth等で必要な場合）"""
    response = make_response("OK")
    response.set_cookie(
        "third_party",
        value=generate_token(),
        httponly=True,
        secure=True,       # SameSite=NoneにはSecure必須
        samesite="None",   # クロスサイトでも送信
        max_age=3600,
        path="/",
    )
    return response
```

### 3.2 SameSite属性の比較

| 値 | クロスサイトGET | クロスサイトPOST | CSRF防御 | 使いやすさ |
|----|:----------:|:-----------:|:--------:|:--------:|
| Strict | 送信しない | 送信しない | 最強 | リンクからの遷移でログアウト状態 |
| Lax | 送信する | 送信しない | 強 | バランスが良い（推奨） |
| None | 送信する | 送信する | なし | Secure属性が必須 |

```
SameSite=Lax の動作:

  他サイトからのリンククリック (GET):
    example.com -> bank.com/dashboard
    Cookie: session_id=xxx  ✓ 送信される

  他サイトからのフォーム送信 (POST):
    evil.com -> bank.com/transfer
    Cookie: session_id=xxx  ✗ 送信されない（CSRF防御）

  他サイトからのiframe読み込み:
    evil.com に <iframe src="bank.com/dashboard">
    Cookie: session_id=xxx  ✗ 送信されない

  他サイトからの画像読み込み:
    evil.com に <img src="bank.com/profile.jpg">
    Cookie: session_id=xxx  ✗ 送信されない

  他サイトからのfetch/XHR (POST):
    evil.com から fetch("bank.com/api", {method: "POST"})
    Cookie: session_id=xxx  ✗ 送信されない
```

### 3.3 SameSite属性のブラウザサポートと注意点

```
SameSite のブラウザデフォルト値の変遷:

  2020年以前:
    SameSite未指定 → None として扱う（Cookieは常に送信）

  2020年以降 (Chrome 80+):
    SameSite未指定 → Lax として扱う
    SameSite=None には Secure 属性が必須

  現在 (2025+):
    主要ブラウザすべてが Lax をデフォルトに
    サードパーティCookie の段階的廃止が進行中

  エッジケース:
    - 2分間の猶予: Chrome は SameSite=Lax のCookieでも
      設定後2分以内はクロスサイトPOSTで送信する
      (Lax+POST 緩和)
    - iOS Safari の実装差異に注意
```

---

## 4. Origin/Referer ヘッダー検証

### 4.1 Origin ヘッダーによる検証

```python
# コード例5: Origin/Referer ヘッダー検証の実装
from flask import Flask, request, abort
from urllib.parse import urlparse

app = Flask(__name__)

ALLOWED_ORIGINS = {
    "https://myapp.example.com",
    "https://www.myapp.example.com",
}

class OriginVerifier:
    """Origin/Referer ヘッダーによるCSRF防御（補助的対策）

    注意: Origin/Referer ヘッダーは以下の場合に欠落する可能性がある
    - プライバシー拡張機能が削除
    - Referrer-Policy: no-referrer が設定されている
    - 古いブラウザ
    - プロキシが除去
    したがって、CSRFトークンとの併用が推奨される
    """

    def __init__(self, allowed_origins: set):
        self.allowed_origins = allowed_origins

    def verify(self) -> bool:
        """Origin または Referer ヘッダーを検証"""
        origin = request.headers.get("Origin")

        if origin:
            return origin in self.allowed_origins

        # Origin がない場合は Referer をフォールバックとして使用
        referer = request.headers.get("Referer")
        if referer:
            parsed = urlparse(referer)
            referer_origin = f"{parsed.scheme}://{parsed.netloc}"
            return referer_origin in self.allowed_origins

        # 両方ない場合の判断
        # 厳格モード: 拒否（推奨）
        # 緩和モード: 許可（互換性重視だがリスクあり）
        return False  # 厳格モード

verifier = OriginVerifier(ALLOWED_ORIGINS)

@app.before_request
def check_origin():
    """状態変更リクエストのOriginを検証"""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if not verifier.verify():
        abort(403, "Origin verification failed")
```

### 4.2 Referer ヘッダーの制御

```
Referrer-Policy とCSRFの関係:

  Referrer-Policy値         Refererヘッダーの内容        CSRF検証への影響
  +-----------------------+---------------------------+-------------------+
  | no-referrer           | 送信されない               | 検証不可          |
  | no-referrer-when-     | HTTPS→HTTPで送信されない   | HTTPS内は検証可能 |
  |   downgrade           |                           |                   |
  | origin                | オリジンのみ               | 検証可能          |
  | origin-when-          | クロスオリジンはオリジンのみ | 検証可能          |
  |   cross-origin        |                           |                   |
  | same-origin           | 同一オリジンのみフルURL    | クロスオリジンは  |
  |                       |                           | 検証不可          |
  | strict-origin         | オリジンのみ(ダウングレード | 検証可能          |
  |                       | 時は送信しない)            |                   |
  | strict-origin-when-   | 同一オリジン:フルURL       | 推奨設定          |
  |   cross-origin(推奨)  | クロスオリジン:オリジンのみ |                   |
  | unsafe-url            | 常にフルURL                | 検証可能だが      |
  |                       |                           | プライバシーリスク |
  +-----------------------+---------------------------+-------------------+
```

---

## 5. SPA (Single Page Application) でのCSRF対策

### 5.1 SPAにおけるCSRF防御パターン

```
SPA でのCSRFトークンの流れ:

  ブラウザ (React/Vue/Angular)        API サーバー
    |                                     |
    |-- GET /api/csrf-token ----------->  |
    |                                     |-- トークン生成
    |<-- { "token": "abc123" } --------   |
    |   + Set-Cookie: csrf=abc123         |
    |                                     |
    |-- POST /api/transfer ------------>  |
    |   X-CSRF-Token: abc123             |-- ヘッダー値と
    |   Cookie: csrf=abc123              |   Cookie値を照合
    |   Cookie: session=xyz              |
    |                                     |-- OK -> 処理実行
```

```python
# コード例6: SPA向けCSRF対策（FastAPI + React）
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import secrets

app = FastAPI()

# CORS設定（CSRFと密接に関連）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.example.com"],  # 明示的に指定
    allow_credentials=True,   # Cookieを許可
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["X-CSRF-Token"],  # カスタムヘッダーを許可
)

@app.get("/api/csrf-token")
async def get_csrf_token(response: Response):
    """CSRFトークンを発行するエンドポイント"""
    token = secrets.token_hex(32)
    # Double Submit Cookie パターン
    response.set_cookie(
        key="csrf_token",
        value=token,
        httponly=False,  # JSから読み取り可能にする（フォーム送信用）
        secure=True,
        samesite="strict",
        max_age=3600,
    )
    return {"csrf_token": token}

@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """CSRF検証ミドルウェア"""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return await call_next(request)

    # カスタムヘッダーの存在確認
    # Simple Request ではカスタムヘッダーを送信できないため
    # これ自体がCSRF対策になる（CORSプリフライト必須）
    csrf_header = request.headers.get("X-CSRF-Token")
    csrf_cookie = request.cookies.get("csrf_token")

    if not csrf_header or not csrf_cookie:
        raise HTTPException(status_code=403, detail="CSRF token missing")

    if not secrets.compare_digest(csrf_header, csrf_cookie):
        raise HTTPException(status_code=403, detail="CSRF token mismatch")

    return await call_next(request)
```

```javascript
// コード例7: React側のCSRFトークン管理
// csrf.js - CSRFトークンユーティリティ

class CSRFTokenManager {
  constructor() {
    this.token = null;
  }

  async fetchToken() {
    const response = await fetch('/api/csrf-token', {
      credentials: 'include',  // Cookieを含める
    });
    const data = await response.json();
    this.token = data.csrf_token;
    return this.token;
  }

  getToken() {
    if (!this.token) {
      // Cookieからトークンを読み取る（フォールバック）
      const cookies = document.cookie.split(';');
      for (const cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'csrf_token') {
          this.token = value;
          break;
        }
      }
    }
    return this.token;
  }

  /**
   * CSRF保護付きfetchラッパー
   */
  async secureFetch(url, options = {}) {
    if (!this.token) {
      await this.fetchToken();
    }

    const headers = {
      ...options.headers,
      'X-CSRF-Token': this.token,
      'Content-Type': 'application/json',
    };

    return fetch(url, {
      ...options,
      headers,
      credentials: 'include',
    });
  }
}

// 使用例
const csrfManager = new CSRFTokenManager();

async function transferFunds(toAccount, amount) {
  const response = await csrfManager.secureFetch('/api/transfer', {
    method: 'POST',
    body: JSON.stringify({ to: toAccount, amount }),
  });

  if (!response.ok) {
    throw new Error(`Transfer failed: ${response.status}`);
  }
  return response.json();
}
```

### 5.2 カスタムヘッダーによるCSRF防御

```
カスタムヘッダーがCSRF防御になる理由:

  通常のフォーム送信 (Simple Request):
  +--------------------------------------------------+
  | <form action="https://api.example.com/transfer"  |
  |       method="POST">                              |
  |   → Content-Type: application/x-www-form-urlencoded |
  |   → カスタムヘッダーは設定不可                      |
  |   → CORSプリフライトなしで送信される               |
  +--------------------------------------------------+

  Fetch/XHR with カスタムヘッダー:
  +--------------------------------------------------+
  | fetch("https://api.example.com/transfer", {      |
  |   headers: { "X-CSRF-Token": "abc" }             |
  | })                                                |
  |   → CORSプリフライト (OPTIONS) が必須              |
  |   → サーバーが許可しないオリジンからは送信不可      |
  +--------------------------------------------------+

  したがって:
  - CORSが正しく設定されている場合
  - カスタムヘッダーの存在チェックだけでCSRF防御可能
  - ただしブラウザのCORS実装バグの可能性があるため
    トークン値の検証も併用すべき
```

---

## 6. クリックジャッキング

透明なiframeでターゲットサイトを重ね、ユーザーの意図しないクリックを誘発する攻撃。

### 6.1 攻撃のメカニズム

```
クリックジャッキングの仕組み:

  攻撃者のページ（表面）:
  +----------------------------------+
  | "無料iPhoneが当たりました!"      |
  |                                  |
  |        [賞品を受け取る]           |
  |                                  |
  +----------------------------------+

  iframeで重ねた銀行サイト（透明）:
  +----------------------------------+
  |  Bank: 送金確認                   |
  |                                  |
  |        [送金を実行する]  <-- 同じ位置|
  |                                  |
  +----------------------------------+

  ユーザーは「賞品を受け取る」をクリックしたつもりが、
  実際には「送金を実行する」をクリックしている
```

### 6.2 クリックジャッキングの亜種

```
クリックジャッキングのバリエーション:

1. Classic Clickjacking:
   透明iframeでターゲットのボタンを重ねる

2. Likejacking:
   SNSの「いいね」ボタンを隠して不正にクリックさせる
   (Facebook Like ボタンの悪用)

3. Cursorjacking:
   カーソルの表示位置をずらし、意図しない場所をクリックさせる
   CSS: cursor: url('custom.cur'), auto;

4. Filejacking:
   ドラッグ&ドロップ操作を利用して
   ユーザーのファイルを攻撃者のサーバーにアップロードさせる

5. Strokejacking:
   キーストロークを奪取する（keypress イベントのリダイレクト）
   フォーカスを透明iframeに合わせてパスワード入力を窃取

6. Multi-step Clickjacking:
   複数回のクリックを誘導し、複数ステップの操作を完了させる
   (確認ダイアログの「はい」もクリックさせる)
```

### 6.3 防御手法の実装

```python
# コード例8: クリックジャッキング対策
from flask import Flask

app = Flask(__name__)

@app.after_request
def anti_clickjacking(response):
    """クリックジャッキング対策ヘッダーの設定"""
    # 方法1: X-Frame-Options（レガシーだが広くサポート）
    response.headers["X-Frame-Options"] = "DENY"
    # DENY: すべてのiframe埋め込みを拒否
    # SAMEORIGIN: 同一オリジンからのみ許可
    # ALLOW-FROM uri: 特定のオリジンからのみ許可（非推奨）

    # 方法2: CSP frame-ancestors（推奨、より柔軟）
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'none'"
        # 'none': すべてのiframe埋め込みを拒否
        # 'self': 同一オリジンからのみ許可
        # https://trusted.com: 特定のオリジンからのみ許可
    )
    return response
```

### 6.4 X-Frame-Options と CSP frame-ancestors の比較

| 項目 | X-Frame-Options | CSP frame-ancestors |
|------|----------------|-------------------|
| 仕様の状態 | 非推奨（レガシー） | W3C 標準 |
| 複数オリジン指定 | 不可（ALLOW-FROM は1つのみ） | 可能（スペース区切り） |
| ワイルドカード | 不可 | 可能（`*.example.com`） |
| ブラウザサポート | 全ブラウザ | モダンブラウザ |
| 優先度 | CSP frame-ancestors が優先 | X-Frame-Options を上書き |
| 推奨 | 後方互換性のため両方設定 | メインの対策として使用 |

### 6.5 JavaScript による Frame Busting

```javascript
// コード例9: JavaScriptによるフレーム脱出（補助的対策）

// 基本的なframe busting
if (window.top !== window.self) {
  window.top.location = window.self.location;
}

// より堅牢な方法
(function() {
  // sandbox属性による回避を防ぐ
  if (self === top) {
    // iframeに埋め込まれていない -> 正常
    document.documentElement.style.display = "block";
  } else {
    // iframeに埋め込まれている -> 脱出試行
    try {
      top.location = self.location;
    } catch (e) {
      // クロスオリジンでブロックされた場合
      document.body.innerHTML =
        "<h1>このページはiframe内では表示できません</h1>";
    }
  }
})();

// 最も堅牢な方法: CSSで初期非表示 + JSで表示
// HTML: <style>html { display: none; }</style>
(function() {
  if (self === top) {
    document.documentElement.style.display = "block";
  } else {
    // iframeに埋め込まれている場合は何も表示しない
    top.location = self.location;
  }
})();
```

```
Frame Busting の回避手法と対策:

  回避手法                          対策
  +-----------------------------+----------------------------------+
  | sandbox="allow-forms"       | CSP frame-ancestors を使用       |
  | (JSを無効化)                |                                  |
  +-----------------------------+----------------------------------+
  | onbeforeunload で遷移阻止   | HTTPヘッダーでの対策を主体に      |
  | window.onbeforeunload =     |                                  |
  |   () => false;              |                                  |
  +-----------------------------+----------------------------------+
  | Double framing              | top !== self ではなく             |
  | (攻撃者のiframe内に          | top.location !== self.location   |
  |  別のiframeを挿入)           | で判定                           |
  +-----------------------------+----------------------------------+
  | location変更の上書き          | Object.defineProperty は         |
  | Object.defineProperty(       | top からは使えないため安全        |
  |   window, 'top', ...)       |                                  |
  +-----------------------------+----------------------------------+

  結論: Frame Busting はあくまで補助的対策。
        CSP frame-ancestors + X-Frame-Options が主体。
```

---

## 7. 包括的な防御戦略

### 7.1 CSRF + クリックジャッキング統合防御

```python
# コード例10: CSRF + クリックジャッキング統合防御ミドルウェア
from flask import Flask, request, session, abort
from functools import wraps
import secrets
import time
import hashlib
import hmac

app = Flask(__name__)

class SecurityMiddleware:
    """CSRF・クリックジャッキング統合防御ミドルウェア"""

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

    def __init__(self, app, secret_key: str, csrf_header: str = "X-CSRF-Token"):
        self.app = app
        self.secret_key = secret_key
        self.csrf_header = csrf_header
        app.before_request(self._check_csrf)
        app.after_request(self._set_security_headers)

    def _check_csrf(self):
        """状態変更リクエストのCSRFトークンを検証"""
        if request.method in self.SAFE_METHODS:
            return None

        # AJAX リクエストの判定（補助的対策）
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            # AJAX リクエストはCORSプリフライトが必要なため
            # 基本的にCSRF安全だが、追加検証も行う
            pass

        # Originヘッダーの検証（補助的対策）
        origin = request.headers.get("Origin")
        if origin and not self._is_same_origin(origin):
            abort(403, "Cross-origin request blocked")

        # CSRFトークンの検証
        token = (request.form.get("csrf_token") or
                 request.headers.get(self.csrf_header))
        if not token or not self._validate_token(token):
            abort(403, "Invalid CSRF token")

    def _set_security_headers(self, response):
        """セキュリティヘッダーの設定"""
        # クリックジャッキング対策
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "frame-ancestors 'none'"
        )

        # その他のセキュリティヘッダー
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "0"  # モダンブラウザでは無効化推奨
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # CSRF Cookie の設定（Double Submit用）
        if "csrf_token" in session:
            response.set_cookie(
                "csrf_token",
                value=session["csrf_token"],
                httponly=False,  # JSから読み取り可能
                secure=True,
                samesite="Strict",
            )

        return response

    def _is_same_origin(self, origin: str) -> bool:
        return origin in ("https://myapp.example.com",)

    def _validate_token(self, token: str) -> bool:
        expected = session.get("csrf_token", "")
        return secrets.compare_digest(token, expected)

# ミドルウェアの適用
SecurityMiddleware(app, "my-secret-key")
```

### 7.2 Django でのCSRF対策

```python
# コード例11: Django の組み込みCSRF対策の活用

# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',  # CSRF ミドルウェア
    # ...
]

# CSRF設定
CSRF_COOKIE_SECURE = True        # HTTPS のみ
CSRF_COOKIE_HTTPONLY = False      # JSから読み取り可能（AJAX用）
CSRF_COOKIE_SAMESITE = "Lax"     # SameSite=Lax
CSRF_USE_SESSIONS = False        # CookieにCSRFトークンを保存（True でセッション）
CSRF_TRUSTED_ORIGINS = [         # 信頼するオリジン
    "https://myapp.example.com",
]

# セキュリティヘッダー設定
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = False  # X-XSS-Protection は無効化

# テンプレートでの使用
# <form method="post">
#   {% csrf_token %}
#   ...
# </form>

# AJAX での使用（Django公式パターン）
# views.py
from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import JsonResponse

@ensure_csrf_cookie
def get_csrf(request):
    """CSRFクッキーを設定するエンドポイント"""
    return JsonResponse({"status": "ok"})
```

### 7.3 防御手法の比較

| 防御手法 | CSRF対策 | クリックジャッキング対策 | 推奨度 |
|---------|:--------:|:---------:|:------:|
| CSRFトークン（Synchronizer） | 強 | - | 高 |
| Double Submit Cookie | 中 | - | 中 |
| Signed Double Submit | 強 | - | 高 |
| SameSite=Lax | 強 | - | 高 |
| SameSite=Strict | 最強 | - | 用途限定 |
| Origin/Refererヘッダー検証 | 補助 | - | 補助 |
| カスタムヘッダー（CORS連携） | 強 | - | 高 (SPA) |
| X-Frame-Options | - | 強 | 高 |
| CSP frame-ancestors | - | 強 | 最高 |
| Frame Busting (JS) | - | 弱 | 補助のみ |

---

## 8. CORS（Cross-Origin Resource Sharing）との関係

```
CORS と CSRF の関係:

  CORS はブラウザによるクロスオリジンリクエストの制御:
  - レスポンスの読み取りを制限する（送信は制限しない!）
  - したがって CORS だけでは CSRF は防げない

  Simple Request（CORSプリフライトなし）:
  +--------------------------------------------------+
  | 条件:                                              |
  | - メソッド: GET, HEAD, POST                        |
  | - Content-Type: text/plain,                       |
  |   application/x-www-form-urlencoded,              |
  |   multipart/form-data                             |
  | - カスタムヘッダーなし                               |
  |                                                   |
  | → リクエストは送信される（レスポンスは読めない）       |
  | → CSRF攻撃は成立する!                               |
  +--------------------------------------------------+

  Preflight Request（カスタムヘッダーあり）:
  +--------------------------------------------------+
  | 条件:                                              |
  | - Content-Type: application/json                  |
  | - カスタムヘッダー (X-CSRF-Token等)                 |
  | - メソッド: PUT, DELETE等                           |
  |                                                   |
  | → まず OPTIONS リクエストが送信される                |
  | → サーバーが許可しなければ本リクエストは送信されない   |
  | → CSRF攻撃は成立しない（CORSが正しく設定されていれば）|
  +--------------------------------------------------+
```

```python
# コード例12: 正しいCORS設定
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# NG: ワイルドカードオリジン + credentials
# CORS(app, origins="*", supports_credentials=True)  # 危険!

# OK: 明示的なオリジン指定
CORS(app,
    origins=["https://myapp.example.com"],
    supports_credentials=True,
    allow_headers=["Content-Type", "X-CSRF-Token"],
    methods=["GET", "POST", "PUT", "DELETE"],
    max_age=3600,
)
```

---

## 9. エッジケース

### エッジケース1: Login CSRF

```
Login CSRF（ログインCSRF）:

  攻撃者が被害者のブラウザで攻撃者のアカウントにログインさせる攻撃

  攻撃フロー:
  1. 攻撃者が自分のクレデンシャルでログインCSRFを仕込む
  2. 被害者が攻撃サイトを訪問
  3. 被害者のブラウザが攻撃者のアカウントでログイン
  4. 被害者は気づかず攻撃者のアカウントで操作
  5. 攻撃者は後からアカウントを確認し
     被害者の入力データ（クレカ情報等）を窃取

  対策:
  - ログインフォームにもCSRFトークンを設定
  - ログイン後にセッションIDを再生成（Session Fixation対策にもなる）
```

### エッジケース2: JSON APIへのCSRF

```
JSON Content-Type でのCSRF:

  application/json は Simple Request の条件を満たさないため
  CORSプリフライトが発生し、通常はCSRFは成立しない。

  ただし以下の場合は注意:
  1. サーバーが Content-Type を検証しない場合
     → text/plain で JSON を送信可能
  2. Flash を利用した307リダイレクト（現在は修正済み）
  3. サーバーがCORS設定でワイルドカードを使用

  対策:
  - Content-Type: application/json を厳格に検証
  - カスタムヘッダーの必須化
  - CORS設定の厳格化
```

### エッジケース3: サブドメインを利用したCSRF

```
サブドメイン攻撃:

  攻撃者が app.example.com のサブドメイン
  (evil.example.com) を取得できた場合:

  1. evil.example.com から .example.com スコープの
     Cookieを設定可能（Cookie Injection）
  2. SameSite Cookie は Same-Site（eTLD+1が同じ）なので
     サブドメイン間は「同一サイト」として扱われる
  3. Double Submit Cookie のCookie値を上書き可能

  対策:
  - __Host- プレフィックス付きCookieを使用
    (ドメイン属性を設定できない → サブドメインから上書き不可)
  - Signed Double Submit Cookie を使用
  - サブドメインの管理を厳格化
```

---

## 10. テスト手法

```python
# コード例13: CSRF脆弱性のテスト
import requests

class CSRFTester:
    """CSRF脆弱性の自動テストツール"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def test_missing_token(self, endpoint: str, method: str = "POST",
                            data: dict = None) -> dict:
        """CSRFトークンなしでリクエストを送信"""
        response = getattr(self.session, method.lower())(
            f"{self.base_url}{endpoint}",
            data=data or {},
        )
        return {
            "test": "missing_token",
            "status_code": response.status_code,
            "vulnerable": response.status_code != 403,
        }

    def test_invalid_token(self, endpoint: str, method: str = "POST",
                            data: dict = None) -> dict:
        """無効なCSRFトークンでリクエストを送信"""
        test_data = data or {}
        test_data["csrf_token"] = "invalid_token_12345"
        response = getattr(self.session, method.lower())(
            f"{self.base_url}{endpoint}",
            data=test_data,
        )
        return {
            "test": "invalid_token",
            "status_code": response.status_code,
            "vulnerable": response.status_code != 403,
        }

    def test_cross_origin(self, endpoint: str, method: str = "POST",
                           data: dict = None) -> dict:
        """異なるOriginヘッダーでリクエストを送信"""
        headers = {"Origin": "https://evil.example.com"}
        response = getattr(self.session, method.lower())(
            f"{self.base_url}{endpoint}",
            data=data or {},
            headers=headers,
        )
        return {
            "test": "cross_origin",
            "status_code": response.status_code,
            "vulnerable": response.status_code != 403,
        }

    def test_clickjacking(self) -> dict:
        """X-Frame-Options / CSP frame-ancestors の確認"""
        response = self.session.get(self.base_url)
        xfo = response.headers.get("X-Frame-Options", "")
        csp = response.headers.get("Content-Security-Policy", "")

        vulnerable = True
        if xfo.upper() in ("DENY", "SAMEORIGIN"):
            vulnerable = False
        if "frame-ancestors" in csp:
            vulnerable = False

        return {
            "test": "clickjacking",
            "x_frame_options": xfo,
            "csp_frame_ancestors": csp,
            "vulnerable": vulnerable,
        }

# 使用例
tester = CSRFTester("https://myapp.example.com")
results = [
    tester.test_missing_token("/transfer", data={"to": "test", "amount": "1"}),
    tester.test_invalid_token("/transfer", data={"to": "test", "amount": "1"}),
    tester.test_cross_origin("/transfer", data={"to": "test", "amount": "1"}),
    tester.test_clickjacking(),
]
for r in results:
    status = "VULNERABLE" if r["vulnerable"] else "SAFE"
    print(f"[{status}] {r['test']}: {r.get('status_code', 'N/A')}")
```

---

## 11. パフォーマンスに関する考察

```
CSRF対策のパフォーマンス影響:

  +------------------------+------------------+------------------+
  | 対策                   | レイテンシ影響    | メモリ影響        |
  +------------------------+------------------+------------------+
  | Synchronizer Token     | セッション読み書き | セッションストア  |
  |                        | ~1ms              | +64B/セッション  |
  +------------------------+------------------+------------------+
  | Double Submit Cookie   | HMAC計算          | なし（ステートレス）|
  |                        | ~0.1ms            |                  |
  +------------------------+------------------+------------------+
  | SameSite Cookie        | なし              | なし              |
  |                        | (ブラウザ側処理)   |                  |
  +------------------------+------------------+------------------+
  | Origin検証             | 文字列比較        | なし              |
  |                        | ~0.01ms           |                  |
  +------------------------+------------------+------------------+

  結論:
  - パフォーマンスへの影響はほぼ無視できるレベル
  - SameSite Cookie はサーバー側処理不要で最も軽量
  - Synchronizer Token はセッションストア（Redis等）への
    追加書き込みが発生するが、実用上問題にならない
```

---

## 演習

### 演習1: 基礎 — CSRFトークンの実装

**課題**: Flask アプリケーションに Synchronizer Token Pattern のCSRF対策を実装せよ。

```
要件:
1. /form (GET) でCSRFトークン付きのフォームを表示
2. /submit (POST) でCSRFトークンを検証
3. トークンが不正な場合は 403 を返す
4. トークンはセッションごとに一意であること

ヒント:
- secrets.token_hex() でトークンを生成
- secrets.compare_digest() でタイミング安全な比較
- テンプレートに {{ csrf_token }} で埋め込み
```

### 演習2: 応用 — SPA向けCSRF対策

**課題**: React + FastAPI 構成で Double Submit Cookie パターンを実装せよ。

```
要件:
1. GET /api/csrf でCSRFトークンをCookieとJSONレスポンスで返す
2. POST/PUT/DELETE リクエストで X-CSRF-Token ヘッダーと
   csrf Cookie の値を照合する
3. CORS設定を適切に行い、許可するオリジンを限定する
4. React側でAxiosインターセプターを使ってCSRFトークンを自動付与

検証:
- curl で X-CSRF-Token なしのPOSTが拒否されることを確認
- curl で不正な Origin からのリクエストが拒否されることを確認
```

### 演習3: 発展 — 包括的セキュリティヘッダー設定

**課題**: 以下のセキュリティヘッダーを全て設定するミドルウェアを実装せよ。

```
必須ヘッダー:
- X-Frame-Options: DENY
- Content-Security-Policy: frame-ancestors 'none'; ...
- X-Content-Type-Options: nosniff
- Referrer-Policy: strict-origin-when-cross-origin
- Strict-Transport-Security: max-age=31536000; includeSubDomains
- Permissions-Policy: camera=(), microphone=(), geolocation=()

追加要件:
- ヘッダーの設定をJSON設定ファイルから読み込み可能にする
- 環境（開発/ステージング/本番）に応じて設定を切り替える
- ヘッダーの正当性を検証するユニットテストを書く
```

---

## アンチパターン

### アンチパターン1: GETリクエストでの状態変更

GETリクエストで送金や削除などの状態変更を行うパターン。`<img src="/delete?id=123">` のような単純な攻撃でCSRFが成立する。状態変更はPOST/PUT/DELETEメソッドに限定し、GETは安全な（副作用のない）操作のみに使用する。

```python
# NG: GETで状態変更
@app.route("/delete-account")  # GETでアカウント削除
def delete_account():
    db.delete_user(session["user_id"])
    return "削除完了"
# 攻撃: <img src="/delete-account"> をメール等に埋め込むだけ

# OK: POSTで状態変更 + CSRF対策
@app.route("/delete-account", methods=["POST"])
def delete_account():
    validate_csrf_token(request.form["csrf_token"])
    db.delete_user(session["user_id"])
    return "削除完了"
```

### アンチパターン2: CSRFトークンの使い回し

全ユーザーで同じCSRFトークンを共有する、またはトークンの有効期限を設定しないパターン。トークンはセッションごとに一意であり、適切な有効期限を設定すべきである。

```python
# NG: 固定トークン
CSRF_TOKEN = "fixed-token-for-all-users"  # 全ユーザー共通!

# NG: 有効期限なし
session["csrf_token"] = secrets.token_hex(32)
# → セッションが続く限り同じトークンが使われる

# OK: セッションごとにユニーク + 有効期限付き
def generate_csrf_token():
    token_data = {
        "value": secrets.token_hex(32),
        "created_at": time.time(),
    }
    session["csrf_token"] = token_data
    return token_data["value"]
```

### アンチパターン3: CORSの過度な緩和

```python
# NG: ワイルドカードオリジン + credentials
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    # → ブラウザはこの組み合わせを拒否するが、
    #   Originをエコーバックするサーバーは攻撃を許す

# NG: Originヘッダーをそのまま反映
@app.after_request
def add_cors_echo(response):
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin  # 危険!
    response.headers["Access-Control-Allow-Credentials"] = "true"

# OK: ホワイトリストで検証
ALLOWED_ORIGINS = {"https://myapp.example.com", "https://admin.example.com"}

@app.after_request
def add_cors_safe(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response
```

---

## FAQ

### Q1: SameSite=Lax があればCSRFトークンは不要ですか?

推奨はしない。SameSite=Laxは強力な対策だが、以下のリスクが残る:
- 古いブラウザではサポートされていない場合がある
- サブドメインからの攻撃（Same-Siteとみなされる）には効かない
- GETメソッドによる状態変更がある場合は防げない
- ブラウザの実装バグの可能性
CSRFトークンとの併用が推奨される（多層防御の原則）。

### Q2: APIのみのアプリケーションでもCSRF対策は必要ですか?

Cookie認証を使用するAPIではCSRF対策が必要。Bearer Token（Authorization ヘッダー）で認証するAPIでは、CookieがないためCSRFのリスクは低い。ただし、CORS設定は正しく行う必要がある。

### Q3: iframeを正当な目的で使いたい場合は?

CSP frame-ancestorsで特定のオリジンのみを許可する。例えば `frame-ancestors 'self' https://partner.com` とすれば、自サイトと信頼されたパートナーサイトからのみiframe埋め込みが可能になる。

### Q4: CSRFトークンをURLパラメータに含めても良いですか?

推奨しない。URLパラメータは以下の場所に漏洩する可能性がある:
- ブラウザ履歴
- サーバーアクセスログ
- Referer ヘッダー
- プロキシログ
CSRFトークンは常にPOSTボディまたはHTTPヘッダーで送信すべきである。

### Q5: WebSocketにCSRF対策は必要ですか?

WebSocket の初期ハンドシェイク（HTTP Upgrade）時にCookieが送信されるため、CSRF対策が必要。対策として:
- Origin ヘッダーの検証
- CSRFトークンをWebSocket接続URLのクエリパラメータに含める
- WebSocket接続後に最初のメッセージで認証トークンを送信

### Q6: __Host- プレフィックスCookieとは何ですか?

```
__Host- プレフィックスCookie:

  Set-Cookie: __Host-session=abc123; Secure; Path=/

  制約:
  - Secure属性が必須
  - Domain属性を設定不可（設定すると拒否される）
  - Path=/ が必須

  効果:
  - サブドメインからの上書きが不可能
  - Double Submit Cookie のCookie Injection攻撃を防止

  ブラウザサポート: Chrome 49+, Firefox 50+, Safari 12+
```

---

## トラブルシューティング

### CSRFトークン検証が失敗する

```
チェックリスト:
1. セッションが正しく維持されているか
   → Redis/Memcached のセッションストアが稼働しているか確認
2. ロードバランサーでセッションが分散していないか
   → Sticky Session または共有セッションストアを使用
3. フォームに csrf_token の hidden input が含まれているか
   → ブラウザのデベロッパーツールで確認
4. AJAX リクエストにヘッダーが設定されているか
   → Network タブで X-CSRF-Token ヘッダーを確認
5. SameSite Cookie でブロックされていないか
   → Application タブで Cookie の SameSite 属性を確認
```

### iframe埋め込みが意図せずブロックされる

```
チェックリスト:
1. X-Frame-Options と CSP frame-ancestors の設定を確認
2. iframe を許可する場合は frame-ancestors で明示的に指定
3. CSP frame-ancestors が X-Frame-Options より優先される
4. ALLOW-FROM は Chrome/Firefox で非対応 → frame-ancestors を使用
```

---

## まとめ

| 脅威 | 推奨対策 | 優先度 |
|------|---------|--------|
| CSRF | SameSite=Lax + CSRFトークン | 最優先 |
| クリックジャッキング | CSP frame-ancestors + X-Frame-Options | 最優先 |
| クロスオリジンリクエスト | CORS設定 + Origin検証 | 高 |
| セッションハイジャック | HttpOnly + Secure Cookie | 高 |
| Login CSRF | ログインフォームにもCSRFトークン | 中 |
| サブドメイン攻撃 | __Host- プレフィックス + Signed Token | 中 |

---

## 次に読むべきガイド

- [03-injection.md](./03-injection.md) -- インジェクション攻撃の詳細と対策
- [04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md) -- 認証・セッション管理の脆弱性
- [01-xss-prevention.md](./01-xss-prevention.md) -- XSSとCSRFの組み合わせ攻撃への対処
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般

---

## 参考文献

1. OWASP CSRF Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
2. OWASP Clickjacking Defense Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Clickjacking_Defense_Cheat_Sheet.html
3. MDN Web Docs: SameSite Cookies -- https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite
4. RFC 6454: The Web Origin Concept -- https://tools.ietf.org/html/rfc6454
5. Fetch Standard (CORS) -- https://fetch.spec.whatwg.org/
6. CSP Level 3: frame-ancestors -- https://www.w3.org/TR/CSP3/#directive-frame-ancestors
7. Cookie Prefixes (__Host-, __Secure-) -- https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie#cookie_prefixes
8. Robust Defenses for Cross-Site Request Forgery (Stanford) -- https://seclab.stanford.edu/websec/csrf/csrf.pdf
