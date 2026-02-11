# CSRF/クリックジャッキング

> CSRF（クロスサイトリクエストフォージェリ）とクリックジャッキングの攻撃メカニズムを解説し、トークン方式、SameSite Cookie、X-Frame-Optionsによる防御を実装する。

## この章で学ぶこと

1. **CSRF攻撃**のメカニズムとトークンベースの防御手法を理解する
2. **SameSite Cookie** 属性による最新のCSRF対策を習得する
3. **クリックジャッキング**の原理とX-Frame-Options/CSP frame-ancestorsによる防御を身につける

---

## 1. CSRF（Cross-Site Request Forgery）とは

ユーザーが認証済みのWebサイトに対して、攻撃者が意図しないリクエストを送信させる攻撃。

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

### 2.2 Double Submit Cookie Pattern

```python
# コード例2: Double Submit Cookie パターン
import secrets
import hmac

class DoubleSubmitCSRF:
    """Double Submit Cookie方式のCSRF対策"""

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

---

## 3. SameSite Cookie

SameSite属性は、ブラウザが第三者サイトからのリクエストにCookieを送信するかどうかを制御する。

```python
# コード例3: SameSite Cookie の設定
from flask import Flask, make_response

app = Flask(__name__)

@app.route("/login", methods=["POST"])
def login():
    response = make_response("ログイン成功")
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
```

### SameSite属性の比較

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
```

---

## 4. クリックジャッキング

透明なiframeでターゲットサイトを重ね、ユーザーの意図しないクリックを誘発する攻撃。

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

  ユーザーは「賞品を受け取る」を  クリックしたつもりが、
  実際には「送金を実行する」をクリックしている
```

```python
# コード例4: クリックジャッキング対策
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

### JavaScript による Frame Busting

```javascript
// コード例5: JavaScriptによるフレーム脱出（補助的対策）

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
```

---

## 5. 包括的な防御戦略

```python
# コード例6: CSRF + クリックジャッキング統合防御
from flask import Flask, request, session, abort
from functools import wraps
import secrets
import time

app = Flask(__name__)

class SecurityMiddleware:
    """CSRF・クリックジャッキング統合防御ミドルウェア"""

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

    def __init__(self, app, secret_key: str):
        self.app = app
        self.secret_key = secret_key
        app.before_request(self._check_csrf)
        app.after_request(self._set_security_headers)

    def _check_csrf(self):
        """状態変更リクエストのCSRFトークンを検証"""
        if request.method in self.SAFE_METHODS:
            return None

        # Originヘッダーの検証（補助的対策）
        origin = request.headers.get("Origin")
        if origin and not self._is_same_origin(origin):
            abort(403, "Cross-origin request blocked")

        # CSRFトークンの検証
        token = (request.form.get("csrf_token") or
                 request.headers.get("X-CSRF-Token"))
        if not token or not self._validate_token(token):
            abort(403, "Invalid CSRF token")

    def _set_security_headers(self, response):
        """セキュリティヘッダーの設定"""
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def _is_same_origin(self, origin: str) -> bool:
        return origin in ("https://myapp.example.com",)

    def _validate_token(self, token: str) -> bool:
        expected = session.get("csrf_token", "")
        return secrets.compare_digest(token, expected)

# ミドルウェアの適用
SecurityMiddleware(app, "my-secret-key")
```

### 防御手法の比較

| 防御手法 | CSRF対策 | クリックジャッキング対策 | 推奨度 |
|---------|:--------:|:---------:|:------:|
| CSRFトークン（Synchronizer） | 強 | - | 高 |
| Double Submit Cookie | 中 | - | 中 |
| SameSite=Lax | 強 | - | 高 |
| Origin/Refererヘッダー検証 | 補助 | - | 補助 |
| X-Frame-Options | - | 強 | 高 |
| CSP frame-ancestors | - | 強 | 最高 |
| Frame Busting (JS) | - | 弱 | 補助のみ |

---

## アンチパターン

### アンチパターン1: GETリクエストでの状態変更

GETリクエストで送金や削除などの状態変更を行うパターン。`<img src="/delete?id=123">` のような単純な攻撃でCSRFが成立する。状態変更はPOST/PUT/DELETEメソッドに限定し、GETは安全な（副作用のない）操作のみに使用する。

### アンチパターン2: CSRFトークンの使い回し

全ユーザーで同じCSRFトークンを共有する、またはトークンの有効期限を設定しないパターン。トークンはセッションごとに一意であり、適切な有効期限を設定すべきである。

---

## FAQ

### Q1: SameSite=Lax があればCSRFトークンは不要ですか?

推奨はしない。SameSite=Laxは強力な対策だが、古いブラウザではサポートされていない場合がある。また、サブドメインからの攻撃やGETメソッドによる状態変更がある場合は防げない。CSRFトークンとの併用が推奨される。

### Q2: APIのみのアプリケーションでもCSRF対策は必要ですか?

Cookie認証を使用するAPIではCSRF対策が必要。Bearer Token（Authorization ヘッダー）で認証するAPIでは、CookieがないためCSRFのリスクは低い。ただし、CORS設定は正しく行う必要がある。

### Q3: iframeを正当な目的で使いたい場合は?

CSP frame-ancestorsで特定のオリジンのみを許可する。例えば `frame-ancestors 'self' https://partner.com` とすれば、自サイトと信頼されたパートナーサイトからのみiframe埋め込みが可能になる。

---

## まとめ

| 脅威 | 推奨対策 | 優先度 |
|------|---------|--------|
| CSRF | SameSite=Lax + CSRFトークン | 最優先 |
| クリックジャッキング | CSP frame-ancestors + X-Frame-Options | 最優先 |
| クロスオリジンリクエスト | CORS設定 + Origin検証 | 高 |
| セッションハイジャック | HttpOnly + Secure Cookie | 高 |

---

## 次に読むべきガイド

- [03-injection.md](./03-injection.md) -- インジェクション攻撃の詳細と対策
- [04-auth-vulnerabilities.md](./04-auth-vulnerabilities.md) -- 認証・セッション管理の脆弱性
- [01-xss-prevention.md](./01-xss-prevention.md) -- XSSとCSRFの組み合わせ攻撃への対処

---

## 参考文献

1. OWASP CSRF Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
2. OWASP Clickjacking Defense Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Clickjacking_Defense_Cheat_Sheet.html
3. MDN Web Docs: SameSite Cookies -- https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite
