# HTTPキャッシュ

> HTTPキャッシュはWebパフォーマンスの要。Cache-Control、ETag、CDNの仕組みを理解し、「何を」「どこで」「どれくらい」キャッシュするかの戦略を設計する。

## この章で学ぶこと

- [ ] HTTPキャッシュの基本原理と種類を体系的に理解する
- [ ] Cache-Controlディレクティブを状況に応じて正しく設定できる
- [ ] ETagとLast-Modifiedによる条件付きリクエストの仕組みを把握する
- [ ] CDNキャッシュの構成と運用を設計できる
- [ ] キャッシュ無効化とキャッシュバスティング戦略を使い分けられる
- [ ] stale-while-revalidateなど先進的パターンを適用できる

---

## 1. HTTPキャッシュの基本原理

### 1.1 なぜキャッシュが必要か

Webアプリケーションにおいて、すべてのリクエストがオリジンサーバーまで到達する設計は、以下の問題を引き起こす。

1. **レイテンシの増大**: 地理的に離れたサーバーへの往復時間（RTT）が応答速度を支配する
2. **帯域幅の浪費**: 同一リソースを繰り返し転送することでネットワーク帯域を消費する
3. **サーバー負荷**: リクエスト数に比例してCPU・メモリ・I/O負荷が増大する
4. **コスト増**: クラウド環境では転送量とリクエスト数が直接的に課金される
5. **可用性リスク**: オリジンサーバー障害時にサービスが完全停止する

HTTPキャッシュは、一度取得したリソースのコピーを中間地点（ブラウザ、プロキシ、CDN）に保存し、同一リソースへの後続リクエストに対してそのコピーを返すことで、これらの問題を包括的に解決する仕組みである。

### 1.2 キャッシュの階層構造

```
┌─────────────────────────────────────────────────────────────────┐
│                  HTTPキャッシュの階層構造                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ユーザーA    ユーザーB    ユーザーC                              │
│    │            │            │                                  │
│    ▼            ▼            ▼                                  │
│  ┌──────┐   ┌──────┐   ┌──────┐      Layer 1: ブラウザ         │
│  │Browser│   │Browser│   │Browser│      プライベートキャッシュ    │
│  │Cache  │   │Cache  │   │Cache  │      (各ユーザー専用)         │
│  └──┬───┘   └──┬───┘   └──┬───┘                               │
│     │  miss     │  miss     │  miss                             │
│     ▼           ▼           ▼                                   │
│  ┌─────────────────────────────┐       Layer 2: フォワード      │
│  │   Forward Proxy Cache       │       プロキシキャッシュ        │
│  │   (企業内プロキシ等)         │       (組織内共有)              │
│  └──────────┬──────────────────┘                                │
│             │  miss                                             │
│             ▼                                                   │
│  ┌─────────────────────────────┐       Layer 3: CDN Edge        │
│  │   CDN Edge Server           │       エッジキャッシュ          │
│  │   (CloudFront/Cloudflare)   │       (地理的に分散)            │
│  └──────────┬──────────────────┘                                │
│             │  miss                                             │
│             ▼                                                   │
│  ┌─────────────────────────────┐       Layer 4: CDN Shield      │
│  │   CDN Origin Shield         │       オリジンシールド          │
│  │   (中間キャッシュ層)         │       (オリジン保護)            │
│  └──────────┬──────────────────┘                                │
│             │  miss                                             │
│             ▼                                                   │
│  ┌─────────────────────────────┐       Layer 5: リバースプロキシ │
│  │   Reverse Proxy (nginx)     │       サーバー前段              │
│  └──────────┬──────────────────┘                                │
│             │  miss                                             │
│             ▼                                                   │
│  ┌─────────────────────────────┐       Layer 6: アプリケーション │
│  │   Application Cache         │       Redis/Memcached等        │
│  │   (Redis / Memcached)       │                                │
│  └──────────┬──────────────────┘                                │
│             │  miss                                             │
│             ▼                                                   │
│  ┌─────────────────────────────┐       Layer 7: データベース     │
│  │   Database                  │       永続ストレージ            │
│  └─────────────────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 プライベートキャッシュと共有キャッシュ

HTTPキャッシュは大別して「プライベートキャッシュ」と「共有キャッシュ」に分類される。この区別はセキュリティとパフォーマンスの両面で極めて重要である。

| 特性 | プライベートキャッシュ | 共有キャッシュ |
|------|----------------------|---------------|
| 格納場所 | ブラウザ | CDN、プロキシ |
| 利用者 | 単一ユーザー | 複数ユーザー |
| ユーザー固有データ | キャッシュ可能 | キャッシュ不可（情報漏洩リスク） |
| Cache-Control指定 | `private` | `public` |
| 容量 | 数百MB〜数GB | 数TB〜数PB（分散合計） |
| 効果範囲 | 同一ユーザーの再訪問 | 全ユーザーへの初回配信高速化 |
| 無効化の容易さ | ブラウザ操作で即座に可能 | パージAPI等で伝搬に時間がかかる |

**重要な設計原則**: ユーザーのセッション情報、個人データ、認証トークンを含むレスポンスには必ず `Cache-Control: private` または `Cache-Control: no-store` を設定する。`public` を設定するとCDNにキャッシュされ、他のユーザーに配信される可能性がある。

### 1.4 キャッシュの鮮度モデル

HTTPキャッシュは「鮮度（Freshness）」という概念に基づいて動作する。キャッシュされたレスポンスは一定期間「新鮮（fresh）」であり、その期間が過ぎると「古い（stale）」とみなされる。

```
┌────────────────────────────────────────────────────────────────┐
│              キャッシュの鮮度ライフサイクル                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  時刻 0          max-age (60s)                                 │
│  │                    │                                        │
│  ▼                    ▼                                        │
│  ├────── fresh ───────┼──── stale ────────────────────────▶    │
│  │  (キャッシュを即返却) │  (検証が必要)                         │
│  │                    │                                        │
│  │  HTTP 200 受信      │  If-None-Match / If-Modified-Since    │
│  │  キャッシュに保存    │  を送信して検証                        │
│  │                    │                                        │
│  │  age = 0           │  age > max-age                         │
│  │                    │                                        │
│  │  Cache-Control:    │  304 Not Modified → キャッシュ再利用     │
│  │  max-age=60        │  200 OK → 新しいレスポンスで更新         │
│  │                    │                                        │
│  └────────────────────┴───────────────────────────────────────│
│                                                                │
│  鮮度計算:                                                      │
│  response_is_fresh = (age < max-age)                           │
│  age = now - date_header_value                                 │
│                                                                │
│  ヒューリスティックキャッシュ:                                    │
│  Cache-Control / Expires がない場合、ブラウザは                  │
│  Last-Modified から独自に鮮度を推定する                           │
│  heuristic_freshness = (now - last_modified) * 0.1             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**ヒューリスティックキャッシュの注意点**: Cache-Controlヘッダーが設定されていないレスポンスに対し、ブラウザはLast-Modifiedヘッダーの値から独自に鮮度を推定する。これは意図しないキャッシュを引き起こす原因となるため、すべてのレスポンスに明示的なCache-Controlヘッダーを設定することが推奨される。

---

## 2. Cache-Control ヘッダー詳解

### 2.1 ディレクティブ一覧と意味

Cache-Controlヘッダーは、HTTPキャッシュの振る舞いを制御するための最も重要なメカニズムである。RFC 9111で標準化されており、リクエストとレスポンスの両方で使用できる。

**レスポンスディレクティブ一覧**:

| ディレクティブ | 意味 | 用途 |
|--------------|------|------|
| `max-age=N` | N秒間キャッシュが有効（新鮮） | 基本的なTTL設定 |
| `s-maxage=N` | 共有キャッシュでのN秒間の有効期限 | CDN向けTTL設定 |
| `no-cache` | キャッシュは保存するが使用前に必ずサーバーで検証 | HTML等の常に最新を保ちたいリソース |
| `no-store` | 一切キャッシュに保存しない | 機密データ |
| `private` | プライベートキャッシュ（ブラウザ）のみ保存可能 | ユーザー固有データ |
| `public` | 共有キャッシュでも保存可能 | 全ユーザー共通の公開データ |
| `must-revalidate` | キャッシュ期限切れ後は必ずサーバーで検証（stale提供禁止） | 重要なリソース |
| `proxy-revalidate` | 共有キャッシュ限定の`must-revalidate` | CDN/プロキシ向け |
| `immutable` | リソースが変更されないことを宣言 | ハッシュ付きファイル名のアセット |
| `no-transform` | 中間キャッシュによるコンテンツ変換を禁止 | 画像圧縮等の変換を防止 |
| `stale-while-revalidate=N` | 期限切れ後N秒間は古いキャッシュを返しつつバックグラウンドで更新 | UX向上 |
| `stale-if-error=N` | オリジンエラー時にN秒間は古いキャッシュを返す | 可用性向上 |

**リクエストディレクティブ一覧**:

| ディレクティブ | 意味 | 用途 |
|--------------|------|------|
| `no-cache` | キャッシュを使用せずオリジンに問い合わせ | 強制リフレッシュ |
| `no-store` | レスポンスをキャッシュに保存しない | 一時的な秘匿通信 |
| `max-age=0` | キャッシュの鮮度を0とみなす（検証を強制） | 再検証の強制 |
| `max-stale=N` | 期限切れ後N秒以内のキャッシュも受け入れる | オフライン耐性 |
| `min-fresh=N` | 少なくともN秒間は新鮮なキャッシュのみ受け入れる | 厳密な鮮度要求 |
| `only-if-cached` | キャッシュにある場合のみ応答（なければ504） | オフラインモード |

### 2.2 よくある誤解の解消

```
┌──────────────────────────────────────────────────────────────────┐
│             no-cache と no-store の違い（頻出の誤解）              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ■ no-cache の動作:                                              │
│                                                                  │
│  Client          Cache            Origin                         │
│    │── GET ───────▶│                │                            │
│    │               │ キャッシュあり   │                            │
│    │               │── If-None-Match ▶│                          │
│    │               │◀── 304 ─────────│  変更なし                  │
│    │◀── cached ────│                │  → キャッシュを返す          │
│    │               │                │                            │
│    │── GET ───────▶│                │                            │
│    │               │ キャッシュあり   │                            │
│    │               │── If-None-Match ▶│                          │
│    │               │◀── 200 ─────────│  変更あり                  │
│    │◀── new ───────│ 更新           │  → 新レスポンスを返す       │
│                                                                  │
│  → キャッシュに保存する。使用前に毎回サーバーで検証する。            │
│  → 変更がなければ 304 で帯域を節約できる。                         │
│                                                                  │
│  ■ no-store の動作:                                              │
│                                                                  │
│  Client          Cache            Origin                         │
│    │── GET ───────▶│                │                            │
│    │               │── GET ─────────▶│                           │
│    │               │◀── 200 ─────────│                           │
│    │◀── 200 ───────│ 保存しない     │                            │
│    │               │                │                            │
│    │── GET ───────▶│                │                            │
│    │               │── GET ─────────▶│  毎回フルレスポンス        │
│    │               │◀── 200 ─────────│                           │
│    │◀── 200 ───────│ 保存しない     │                            │
│                                                                  │
│  → キャッシュに一切保存しない。毎回フルレスポンスが必要。           │
│  → 帯域の節約効果はない。                                         │
│                                                                  │
│  結論: "キャッシュ禁止" = no-store                                │
│        "検証付きキャッシュ" = no-cache                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 実践的なCache-Control設定パターン

#### パターン1: 静的アセット（ハッシュ付きファイル名）

```nginx
# nginx設定例: ハッシュ付き静的ファイル
# ファイル例: app.a1b2c3d4.js, style.e5f6g7h8.css
location ~* \.[0-9a-f]{8,}\.(js|css|woff2?|png|jpg|webp|avif|svg)$ {
    expires 365d;
    add_header Cache-Control "public, max-age=31536000, immutable";
    add_header X-Cache-Strategy "immutable-asset";

    # gzip/brotli 圧縮
    gzip_static on;
    brotli_static on;
}
```

このパターンでは、ファイル名にコンテンツハッシュが含まれるため、ファイル内容が変更されるとファイル名自体が変わる。したがって、既存のURLに対するキャッシュは永久に有効と宣言できる。`immutable` ディレクティブにより、ブラウザは期限内の再検証リクエスト（条件付きGET）すら送信しない。

#### パターン2: HTMLドキュメント

```nginx
# nginx設定例: HTMLファイル
location ~* \.html$ {
    add_header Cache-Control "no-cache";
    add_header X-Cache-Strategy "always-validate";

    # ETagを有効化（nginx はデフォルトで有効）
    etag on;
}
```

HTMLファイルは静的アセットへの参照を含むため、常に最新版を提供する必要がある。`no-cache` により、ブラウザはキャッシュを保持するが、使用前に必ずサーバーで検証する。ETagが一致すれば304応答となり帯域を節約できる。

#### パターン3: パブリックAPIレスポンス

```python
# FastAPI の例: パブリックAPIレスポンスのキャッシュ設定
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import hashlib
import json

app = FastAPI()

@app.get("/api/products")
async def get_products(response: Response):
    products = await fetch_products_from_db()

    # レスポンスボディのETagを生成
    body = json.dumps(products, sort_keys=True)
    etag = hashlib.md5(body.encode()).hexdigest()

    response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300"
    response.headers["ETag"] = f'"{etag}"'
    response.headers["Vary"] = "Accept-Encoding"

    return products


@app.get("/api/products/{product_id}")
async def get_product(product_id: int, response: Response):
    product = await fetch_product_from_db(product_id)

    body = json.dumps(product, sort_keys=True)
    etag = hashlib.md5(body.encode()).hexdigest()

    # 個別商品は短めのキャッシュ
    response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120"
    response.headers["ETag"] = f'"{etag}"'
    response.headers["Vary"] = "Accept-Encoding"

    return product
```

#### パターン4: ユーザー固有のAPIレスポンス

```python
# FastAPI の例: ユーザー固有データのキャッシュ設定
from fastapi import FastAPI, Response, Depends

@app.get("/api/me/profile")
async def get_my_profile(
    response: Response,
    current_user = Depends(get_current_user)
):
    profile = await fetch_user_profile(current_user.id)

    # private: ブラウザのみキャッシュ可。CDNには保存されない
    response.headers["Cache-Control"] = "private, max-age=0, must-revalidate"
    response.headers["ETag"] = f'"{profile.version}"'

    return profile


@app.get("/api/me/settings")
async def get_my_settings(
    response: Response,
    current_user = Depends(get_current_user)
):
    settings = await fetch_user_settings(current_user.id)

    # 設定変更はリアルタイム反映が必要
    response.headers["Cache-Control"] = "private, no-cache"
    response.headers["ETag"] = f'"{settings.updated_at.isoformat()}"'

    return settings
```

#### パターン5: 機密データ

```python
# 機密データは一切キャッシュしない
@app.get("/api/me/payment-methods")
async def get_payment_methods(
    response: Response,
    current_user = Depends(get_current_user)
):
    methods = await fetch_payment_methods(current_user.id)

    # no-store: メモリにもディスクにも保存しない
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"  # HTTP/1.0 後方互換

    return methods
```

### 2.4 Varyヘッダーとキャッシュキー

`Vary` ヘッダーは、キャッシュのキーにどのリクエストヘッダーを含めるかを指定する。同一URLでもリクエストヘッダーの値が異なれば、別のキャッシュエントリとして扱われる。

```
Vary ヘッダーの動作:

  レスポンス: Vary: Accept-Encoding, Accept-Language

  キャッシュキー = URL + Accept-Encoding + Accept-Language

  リクエスト1: Accept-Encoding: gzip, Accept-Language: ja
    → キャッシュエントリ A に保存

  リクエスト2: Accept-Encoding: br, Accept-Language: ja
    → キャッシュエントリ B に保存（Accept-Encoding が異なる）

  リクエスト3: Accept-Encoding: gzip, Accept-Language: en
    → キャッシュエントリ C に保存（Accept-Language が異なる）

  リクエスト4: Accept-Encoding: gzip, Accept-Language: ja
    → キャッシュエントリ A にヒット

  注意: Vary: * を指定すると、事実上キャッシュ不可になる
  （すべてのリクエストが一意とみなされるため）
```

**Varyの設計ガイドライン**:

| シナリオ | Vary設定 | 理由 |
|---------|---------|------|
| 通常のAPI | `Vary: Accept-Encoding` | 圧縮形式ごとに別キャッシュ |
| 多言語サイト | `Vary: Accept-Language` | 言語ごとに別コンテンツ |
| コンテンツネゴシエーション | `Vary: Accept` | JSON/XML等で別レスポンス |
| 認証付きAPI | `Vary: Authorization` | ユーザーごとに別レスポンス（非推奨、privateを使うべき） |

---

## 3. 条件付きリクエスト（ETag / Last-Modified）

### 3.1 ETag（Entity Tag）の仕組み

ETagはリソースの特定バージョンを識別するための不透明な文字列である。サーバーがレスポンスに付与し、クライアントは後続のリクエストで条件付きヘッダーとして送信する。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ETag 検証フロー詳細                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  === 初回リクエスト ===                                              │
│                                                                     │
│  Client                    Server                                   │
│    │                         │                                      │
│    │── GET /api/users/42 ──▶│                                      │
│    │                         │  リソースを取得                       │
│    │                         │  ETag を計算: "v1-abc123"            │
│    │                         │                                      │
│    │◀── 200 OK ─────────────│                                      │
│    │   ETag: "v1-abc123"     │                                      │
│    │   Cache-Control: no-cache                                      │
│    │   Content-Length: 245   │                                      │
│    │   Body: {"id":42,...}   │                                      │
│    │                         │                                      │
│    │  ブラウザがレスポンスと                                          │
│    │  ETag をキャッシュに保存                                        │
│    │                         │                                      │
│  === 2回目のリクエスト（変更なし） ===                                │
│    │                         │                                      │
│    │── GET /api/users/42 ──▶│                                      │
│    │   If-None-Match:        │                                      │
│    │   "v1-abc123"           │                                      │
│    │                         │  リソースを取得                       │
│    │                         │  ETag を計算: "v1-abc123"            │
│    │                         │  → 一致! 変更なし                     │
│    │                         │                                      │
│    │◀── 304 Not Modified ───│                                      │
│    │   ETag: "v1-abc123"     │                                      │
│    │   (ボディなし)           │  ★ 帯域を節約                       │
│    │                         │                                      │
│    │  キャッシュから                                                  │
│    │  レスポンスボディを復元                                          │
│    │                         │                                      │
│  === 3回目のリクエスト（変更あり） ===                                │
│    │                         │                                      │
│    │── GET /api/users/42 ──▶│                                      │
│    │   If-None-Match:        │                                      │
│    │   "v1-abc123"           │                                      │
│    │                         │  リソースを取得                       │
│    │                         │  ETag を計算: "v2-def456"            │
│    │                         │  → 不一致! 変更あり                   │
│    │                         │                                      │
│    │◀── 200 OK ─────────────│                                      │
│    │   ETag: "v2-def456"     │                                      │
│    │   Content-Length: 260   │                                      │
│    │   Body: {"id":42,...}   │  ★ 新しいレスポンスを返す             │
│    │                         │                                      │
│    │  キャッシュを新しい                                              │
│    │  レスポンスで更新                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 強いETag と 弱いETag

ETagには「強い検証」と「弱い検証」の2種類がある。

```
強いETag:
  ETag: "abc123"
  → バイト単位で完全に一致することを保証
  → Range リクエスト（部分ダウンロード）に使用可能
  → コンテンツのハッシュ値（MD5, SHA-256等）から生成するのが一般的

弱いETag:
  ETag: W/"abc123"
  → 意味的に同等であることを示す（バイト単位の一致は保証しない）
  → マイナーな差異（空白、コメント、日付フォーマット等）は無視
  → Range リクエストには使用不可

使い分け:
  静的ファイル → 強いETag（ファイルハッシュ）
  動的コンテンツ → 弱いETag（セマンティックバージョン）
  HTMLテンプレート → 弱いETag（レンダリング結果の微細な差異を許容）
```

### 3.3 ETagの生成戦略

```python
# ETag生成の実装例
import hashlib
import json
from datetime import datetime

class ETagGenerator:
    @staticmethod
    def from_content(content: bytes) -> str:
        """コンテンツのハッシュからETagを生成（強いETag）"""
        hash_value = hashlib.sha256(content).hexdigest()[:16]
        return f'"{hash_value}"'

    @staticmethod
    def from_version(version: int, updated_at: datetime) -> str:
        """バージョン番号と更新日時からETagを生成（弱いETag）"""
        raw = f"{version}-{updated_at.isoformat()}"
        hash_value = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f'W/"{hash_value}"'

    @staticmethod
    def from_db_row(row: dict, fields: list[str]) -> str:
        """データベース行の特定フィールドからETagを生成"""
        subset = {k: row[k] for k in fields if k in row}
        raw = json.dumps(subset, sort_keys=True, default=str)
        hash_value = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f'"{hash_value}"'

# 使用例
etag1 = ETagGenerator.from_content(b'{"name": "Taro"}')
# → '"a1b2c3d4e5f6g7h8"'

etag2 = ETagGenerator.from_version(3, datetime(2025, 1, 15, 10, 30))
# → 'W/"1a2b3c4d5e6f"'

etag3 = ETagGenerator.from_db_row(
    {"id": 42, "name": "Taro", "email": "taro@example.com", "internal_flag": True},
    ["id", "name", "email"]  # internal_flag は除外
)
# → '"9f8e7d6c5b4a3210"'
```

### 3.4 Last-Modifiedとの比較

| 特性 | ETag | Last-Modified |
|------|------|---------------|
| 精度 | 任意の粒度（バイト単位可） | 秒単位（1秒以内の変更を検出できない） |
| 対応ヘッダー | `If-None-Match` | `If-Modified-Since` |
| 複数バリアント | 可能（カンマ区切り） | 不可（単一の日時のみ） |
| 弱い比較 | `W/"..."` で可能 | 本質的に弱い比較 |
| サーバー負荷 | ハッシュ計算が必要 | ファイルシステムのmtimeを使用可能 |
| 分散環境 | コンテンツベースで安定 | サーバー間でmtimeが異なる可能性あり |
| Range対応 | 強いETagのみ対応 | 非対応 |
| 推奨度 | 優先的に使用 | 補助的に使用 |

**推奨**: ETagを主とし、Last-Modifiedを補助的に併用する。両方が存在する場合、HTTPの仕様ではETagが優先される。

### 3.5 条件付きリクエストによる更新の競合防止

ETagは読み取りキャッシュだけでなく、更新操作の競合防止（楽観的ロック）にも活用できる。

```python
# PUT/PATCH での楽観的ロック実装例
from fastapi import FastAPI, Response, Request, HTTPException

@app.put("/api/users/{user_id}")
async def update_user(
    user_id: int,
    request: Request,
    response: Response,
    body: UserUpdate
):
    # 現在のリソースを取得
    current = await fetch_user(user_id)
    current_etag = ETagGenerator.from_version(
        current.version, current.updated_at
    )

    # If-Match ヘッダーの検証
    if_match = request.headers.get("If-Match")
    if if_match is None:
        raise HTTPException(
            status_code=428,
            detail="If-Match header is required for updates"
        )

    if if_match != current_etag:
        raise HTTPException(
            status_code=412,  # Precondition Failed
            detail="Resource has been modified by another request"
        )

    # 更新を実行
    updated = await update_user_in_db(user_id, body, current.version)
    new_etag = ETagGenerator.from_version(
        updated.version, updated.updated_at
    )

    response.headers["ETag"] = new_etag
    response.headers["Cache-Control"] = "private, no-cache"

    return updated
```

---

## 4. キャッシュ無効化とキャッシュバスティング

### 4.1 キャッシュ無効化の2つの課題

Phil Karltonの名言「コンピュータサイエンスで難しいことは2つだけ。キャッシュの無効化と命名」が示す通り、キャッシュの無効化はソフトウェア工学における根本的な課題の一つである。

**課題1: 伝搬遅延**
CDNのエッジサーバーは世界中に分散しており、パージ命令が全ノードに伝搬するまでにタイムラグがある。

**課題2: ブラウザキャッシュの制御不能性**
一度ブラウザにキャッシュされたリソースは、サーバー側から強制的に無効化する手段がない。`max-age=31536000` で配信されたリソースは、ユーザーがブラウザキャッシュをクリアするか、異なるURLでアクセスしない限り更新されない。

### 4.2 キャッシュバスティング戦略

```
┌─────────────────────────────────────────────────────────────────┐
│             キャッシュバスティング戦略の比較                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ■ 戦略1: コンテンツハッシュ（推奨）                              │
│                                                                 │
│    ビルド時にファイル内容のハッシュをファイル名に埋め込む            │
│                                                                 │
│    app.js  →  app.a1b2c3d4.js                                   │
│    style.css → style.e5f6g7h8.css                               │
│    logo.png → logo.9i0j1k2l.png                                 │
│                                                                 │
│    HTML: <script src="/assets/app.a1b2c3d4.js">                 │
│                                                                 │
│    長所: 内容が変わらない限り同一URL → キャッシュ効率が最大         │
│          内容が変わるとURL自体が変わる → 確実に新版を取得           │
│    短所: ビルドツール（Vite, webpack）の設定が必要                 │
│                                                                 │
│  ■ 戦略2: バージョンクエリパラメータ                              │
│                                                                 │
│    app.js?v=1.2.3                                               │
│    style.css?v=20250115                                         │
│                                                                 │
│    長所: 実装が簡単、ビルドツール不要                              │
│    短所: 一部のCDN/プロキシがクエリパラメータを                    │
│          キャッシュキーに含めない場合がある                        │
│          全ファイルを一括無効化してしまうリスク                     │
│                                                                 │
│  ■ 戦略3: ディレクトリベースのバージョニング                      │
│                                                                 │
│    /v1/app.js → /v2/app.js                                      │
│    /assets/1.2.3/style.css                                      │
│                                                                 │
│    長所: すべてのキャッシュで確実に動作                            │
│    短所: ディレクトリ管理が煩雑、旧版の残留                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 ビルドツールによるキャッシュバスティング設定

```javascript
// vite.config.ts — Viteのキャッシュバスティング設定
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // コンテンツハッシュをファイル名に含める（デフォルトで有効）
    rollupOptions: {
      output: {
        // エントリーポイント: app.[hash].js
        entryFileNames: 'assets/[name].[hash].js',
        // コード分割チャンク: chunk-[name].[hash].js
        chunkFileNames: 'assets/chunk-[name].[hash].js',
        // アセット: [name].[hash].[ext]
        assetFileNames: 'assets/[name].[hash].[ext]',
      },
    },
    // ソースマップの生成（本番環境ではhiddenを推奨）
    sourcemap: 'hidden',
  },
});
```

### 4.4 stale-while-revalidate パターン

stale-while-revalidate（SWR）は、キャッシュの鮮度とユーザー体験を両立させるための強力なパターンである。

```
stale-while-revalidate の動作タイムライン:

  Cache-Control: max-age=60, stale-while-revalidate=300

  時刻(秒)  0         60                    360
            │          │                      │
            ▼          ▼                      ▼
  ┌─────────┼──────────┼──────────────────────┼──────────────▶
  │  Phase  │  FRESH   │  STALE-WHILE-        │  STALE
  │         │          │  REVALIDATE          │  (検証必須)
  ├─────────┼──────────┼──────────────────────┼──────────────
  │  動作   │キャッシュ │キャッシュを即返却     │サーバーに
  │         │を即返却   │+ バックグラウンドで   │問い合わせて
  │         │          │サーバーに問い合わせ    │から返却
  │         │          │→ 次回から新版を返却   │
  ├─────────┼──────────┼──────────────────────┼──────────────
  │  体感   │ 即座     │ 即座                 │ 遅延あり
  │  速度   │ (<5ms)   │ (<5ms)               │ (RTT分)
  └─────────┴──────────┴──────────────────────┴──────────────

  特徴:
  - ユーザーは0-360秒の間、常に即座にレスポンスを得る
  - 60-360秒の間は「少し古い」データが返る可能性がある
  - バックグラウンド更新後、次のリクエストからは最新データが返る
  - 「速度」と「鮮度」のトレードオフを柔軟に調整可能
```

### 4.5 CDNキャッシュパージの実装

```python
# CDNキャッシュパージの実装例

import boto3
import time

class CDNCachePurger:
    """CDNキャッシュのパージを実行するユーティリティ"""

    def __init__(self, distribution_id: str):
        self.client = boto3.client('cloudfront')
        self.distribution_id = distribution_id

    def purge_paths(self, paths: list[str]) -> dict:
        """指定パスのキャッシュをパージする"""
        response = self.client.create_invalidation(
            DistributionId=self.distribution_id,
            InvalidationBatch={
                'Paths': {
                    'Quantity': len(paths),
                    'Items': paths
                },
                'CallerReference': f'purge-{int(time.time())}'
            }
        )
        return {
            'invalidation_id': response['Invalidation']['Id'],
            'status': response['Invalidation']['Status'],
            'paths': paths
        }

    def purge_all(self) -> dict:
        """全キャッシュをパージする（コスト注意）"""
        return self.purge_paths(['/*'])


# 使用例
purger = CDNCachePurger(distribution_id='E1A2B3C4D5E6F7')

# 特定パスのパージ
result = purger.purge_paths([
    '/api/products/*',
    '/images/hero.webp'
])
print(f"Invalidation ID: {result['invalidation_id']}")

# デプロイ時の全パージ
result = purger.purge_all()
```

---

## 5. CDNキャッシュの設計と運用

### 5.1 CDNの基本アーキテクチャ

CDN（Content Delivery Network）は、世界中に分散配置されたエッジサーバー群によって、コンテンツをユーザーに近い地点から配信するインフラストラクチャである。

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CDN アーキテクチャ詳細図                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                     ┌──────────────┐                                 │
│                     │  Origin      │                                 │
│                     │  Server      │                                 │
│                     │  (東京DC)    │                                 │
│                     └──────┬───────┘                                 │
│                            │                                         │
│                     ┌──────┴───────┐                                 │
│                     │  Origin      │  ← 全エッジからのリクエストを集約  │
│                     │  Shield      │    オリジンへの負荷を軽減         │
│                     │  (東京)      │                                 │
│                     └──────┬───────┘                                 │
│                            │                                         │
│            ┌───────────────┼───────────────┐                         │
│            │               │               │                         │
│     ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐                 │
│     │  Edge POP   │ │  Edge POP   │ │  Edge POP   │  ← PoP:        │
│     │  東京       │ │  シンガポール│ │  ロンドン   │    Point of     │
│     │  (10+ nodes)│ │  (10+ nodes)│ │  (10+ nodes)│    Presence     │
│     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                 │
│            │               │               │                         │
│     ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐                 │
│     │  日本の     │ │  ASEANの   │ │  欧州の     │                 │
│     │  ユーザー   │ │  ユーザー   │ │  ユーザー   │                 │
│     └─────────────┘ └─────────────┘ └─────────────┘                 │
│                                                                      │
│  リクエストフロー:                                                    │
│  1. DNS解決 → 最寄りのEdge PoPのIPアドレスを返す                      │
│  2. Edge POP にキャッシュあり → 即座に返却（Cache HIT）              │
│  3. Edge POP にキャッシュなし → Origin Shield に問い合わせ            │
│  4. Origin Shield にキャッシュあり → Edge に返却・キャッシュ          │
│  5. Origin Shield にキャッシュなし → Origin Server に問い合わせ       │
│  6. Origin Server がレスポンス → Shield → Edge → ユーザー           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 CDNキャッシュ制御ヘッダーの使い分け

CDNに対するキャッシュ制御には複数のヘッダーが存在し、それぞれ適用範囲が異なる。

| ヘッダー | 適用対象 | 標準化状況 | 用途 |
|---------|---------|-----------|------|
| `Cache-Control: s-maxage` | 全ての共有キャッシュ（CDN含む） | RFC 9111 | CDNとプロキシ共通のTTL |
| `CDN-Cache-Control` | CDNのみ（プロキシには適用されない） | RFC 9213 | CDN専用のTTL |
| `Surrogate-Control` | 対応CDNのみ（Fastly等） | W3C TR | CDN固有の高度な制御 |
| `Cloudflare-CDN-Cache-Control` | Cloudflareのみ | 独自仕様 | Cloudflare専用制御 |

```
ヘッダーの優先順位（CDN側の解釈）:

  CDN固有ヘッダー (Surrogate-Control等)
    ↓ なければ
  CDN-Cache-Control
    ↓ なければ
  Cache-Control: s-maxage
    ↓ なければ
  Cache-Control: max-age
    ↓ なければ
  Expires ヘッダー
    ↓ なければ
  ヒューリスティックキャッシュ or キャッシュなし

  推奨: Cache-Control の s-maxage を基本とし、
  CDN固有の要件がある場合のみ CDN-Cache-Control を併用する。
```

### 5.3 CloudFront 設定例

```yaml
# AWS CloudFormation による CloudFront 設定例
AWSTemplateFormatVersion: '2010-09-09'
Description: CloudFront distribution with optimized caching

Resources:
  # キャッシュポリシー: 静的アセット用
  StaticAssetsCachePolicy:
    Type: AWS::CloudFront::CachePolicy
    Properties:
      CachePolicyConfig:
        Name: StaticAssets-1Year
        DefaultTTL: 86400        # 1日（Cache-Controlがない場合）
        MaxTTL: 31536000         # 1年（上限）
        MinTTL: 0                # 0秒（最小）
        ParametersInCacheKeyAndForwardedToOrigin:
          CookiesConfig:
            CookieBehavior: none   # Cookie をキャッシュキーに含めない
          HeadersConfig:
            HeaderBehavior: none   # ヘッダーをキャッシュキーに含めない
          QueryStringsConfig:
            QueryStringBehavior: none  # クエリパラメータを含めない
          EnableAcceptEncodingGzip: true
          EnableAcceptEncodingBrotli: true

  # キャッシュポリシー: API用
  APICachePolicy:
    Type: AWS::CloudFront::CachePolicy
    Properties:
      CachePolicyConfig:
        Name: API-ShortTTL
        DefaultTTL: 60           # 1分
        MaxTTL: 300              # 5分
        MinTTL: 0
        ParametersInCacheKeyAndForwardedToOrigin:
          CookiesConfig:
            CookieBehavior: none
          HeadersConfig:
            HeaderBehavior: whitelist
            Headers:
              - Accept
              - Accept-Language
          QueryStringsConfig:
            QueryStringBehavior: all  # 全クエリパラメータをキーに含める
          EnableAcceptEncodingGzip: true
          EnableAcceptEncodingBrotli: true

  # CloudFront ディストリビューション
  Distribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Enabled: true
        HttpVersion: http2and3
        PriceClass: PriceClass_200  # アジア・欧米
        Origins:
          - Id: AppOrigin
            DomainName: app.example.com
            CustomOriginConfig:
              OriginProtocolPolicy: https-only
              OriginSSLProtocols: [TLSv1.2]
        DefaultCacheBehavior:
          # HTML: 常に検証
          TargetOriginId: AppOrigin
          ViewerProtocolPolicy: redirect-to-https
          CachePolicyId: !Ref APICachePolicy
          Compress: true
        CacheBehaviors:
          # 静的アセット
          - PathPattern: '/assets/*'
            TargetOriginId: AppOrigin
            ViewerProtocolPolicy: redirect-to-https
            CachePolicyId: !Ref StaticAssetsCachePolicy
            Compress: true
          # API
          - PathPattern: '/api/*'
            TargetOriginId: AppOrigin
            ViewerProtocolPolicy: redirect-to-https
            CachePolicyId: !Ref APICachePolicy
            Compress: true
```

### 5.4 主要CDNサービスの比較

| 特性 | CloudFront (AWS) | Cloudflare | Fastly | Akamai |
|------|-----------------|------------|--------|--------|
| PoP数 | 600+ | 300+ | 70+ | 4,000+ |
| 無料枠 | 1TB/月 | 無制限（Free plan） | なし | なし |
| パージ速度 | 数分 | 約30秒 | 約150ms | 約5秒 |
| エッジコンピューティング | Lambda@Edge, Functions | Workers | Compute@Edge | EdgeWorkers |
| HTTP/3対応 | あり | あり | あり | あり |
| WebSocket対応 | あり | あり | あり | あり |
| DDoS防御 | AWS Shield | 標準搭載 | 標準搭載 | Kona Site Defender |
| 価格モデル | 従量課金 | プランベース | 従量課金 | 契約ベース |
| 強み | AWSエコシステム統合 | 設定の簡便さ | パージ速度 | 大規模配信 |

---

## 6. 実践的なキャッシュ戦略の設計

### 6.1 リソースタイプ別キャッシュ戦略マトリクス

| リソースタイプ | Cache-Control | ETag | CDN | バスティング | 備考 |
|--------------|---------------|------|-----|-------------|------|
| HTML | `no-cache` | あり | 短TTL (60s) | 不要（常に検証） | 最新のアセット参照を保証 |
| JS/CSS（ハッシュ付き） | `public, max-age=31536000, immutable` | 不要 | 長TTL (1年) | ファイル名ハッシュ | 変更時はURLが変わる |
| 画像（ハッシュ付き） | `public, max-age=31536000, immutable` | 不要 | 長TTL (1年) | ファイル名ハッシュ | WebP/AVIF変換はCDNエッジで |
| フォント | `public, max-age=31536000, immutable` | 不要 | 長TTL (1年) | ファイル名ハッシュ | CORS設定が必要な場合あり |
| パブリックAPI | `public, max-age=60, s-maxage=300` | あり | 中TTL | 不要 | Varyヘッダーに注意 |
| ユーザー固有API | `private, no-cache` | あり | なし | 不要 | CDNにキャッシュしない |
| 機密データ | `no-store` | なし | なし | 不要 | 一切キャッシュ禁止 |
| Service Worker | `no-cache, max-age=0` | あり | 短TTL | 不要 | 24時間上限（ブラウザ仕様） |
| favicon.ico | `public, max-age=86400` | あり | 1日 | クエリパラメータ | 頻繁には変更しない |
| robots.txt | `public, max-age=86400` | あり | 1日 | 不要 | クロール設定 |
| sitemap.xml | `public, max-age=3600` | あり | 1時間 | 不要 | SEO関連 |

### 6.2 nginx による包括的キャッシュ設定

```nginx
# /etc/nginx/conf.d/cache.conf
# 包括的なキャッシュ設定

# ── 共通設定 ──

# プロキシキャッシュの定義
proxy_cache_path /var/cache/nginx levels=1:2
    keys_zone=app_cache:100m    # キャッシュキーのメタデータ領域
    max_size=10g                # ディスク上の最大サイズ
    inactive=60m                # 60分アクセスがなければ削除
    use_temp_path=off;          # 一時ファイルを使わない（性能向上）

# キャッシュキーの定義
proxy_cache_key "$scheme$request_method$host$request_uri";

server {
    listen 443 ssl http2;
    server_name example.com;

    # ── 静的アセット（ハッシュ付き） ──
    location ~* /assets/.*\.[0-9a-f]{8,}\.(js|css|woff2?|png|jpg|webp|avif|svg)$ {
        root /var/www/app/dist;

        # 1年間キャッシュ、変更なしを宣言
        add_header Cache-Control "public, max-age=31536000, immutable";
        add_header X-Cache-Strategy "immutable-hashed-asset";

        # 圧縮済みファイルがあれば使用
        gzip_static on;

        # アクセスログを抑制（大量のアセットリクエスト）
        access_log off;
    }

    # ── 静的アセット（ハッシュなし） ──
    location ~* \.(ico|png|jpg|jpeg|gif|svg|webp)$ {
        root /var/www/app/dist;
        add_header Cache-Control "public, max-age=86400";
        add_header X-Cache-Strategy "static-no-hash";
        etag on;
    }

    # ── HTML ──
    location ~* \.html$ {
        root /var/www/app/dist;
        add_header Cache-Control "no-cache";
        add_header X-Cache-Strategy "html-always-validate";
        etag on;
    }

    # ── SPA のフォールバック ──
    location / {
        root /var/www/app/dist;
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache";
        etag on;
    }

    # ── パブリック API ──
    location /api/public/ {
        proxy_pass http://backend;
        proxy_cache app_cache;
        proxy_cache_valid 200 5m;
        proxy_cache_valid 404 1m;

        # キャッシュヒット状況をレスポンスヘッダーに表示
        add_header X-Cache-Status $upstream_cache_status;

        # stale-while-revalidate の実装
        proxy_cache_use_stale updating error timeout http_500 http_502;
        proxy_cache_background_update on;
        proxy_cache_lock on;

        # Vary ヘッダーの適切な処理
        proxy_ignore_headers Vary;
        proxy_cache_key "$scheme$request_method$host$request_uri$http_accept";
    }

    # ── プライベート API ──
    location /api/me/ {
        proxy_pass http://backend;
        proxy_no_cache 1;         # キャッシュしない
        proxy_cache_bypass 1;     # キャッシュをバイパス
        add_header Cache-Control "private, no-cache";
    }

    # ── 機密 API ──
    location /api/secure/ {
        proxy_pass http://backend;
        proxy_no_cache 1;
        proxy_cache_bypass 1;
        add_header Cache-Control "no-store";
        add_header Pragma "no-cache";
    }
}
```

### 6.3 Service Worker によるキャッシュ戦略

Service Workerを活用すると、ブラウザ側でより細かいキャッシュ制御が可能になる。代表的な戦略は以下の通りである。

```
Service Worker キャッシュ戦略:

  ■ Cache First（キャッシュ優先）
    → キャッシュにあればそれを返す。なければネットワークから取得
    → 用途: 静的アセット、フォント

  ■ Network First（ネットワーク優先）
    → ネットワークから取得を試み、失敗したらキャッシュを返す
    → 用途: API、HTML

  ■ Stale While Revalidate
    → キャッシュを即返しつつ、バックグラウンドでネットワーク更新
    → 用途: ニュースフィード、SNSタイムライン

  ■ Cache Only
    → キャッシュのみ参照（オフライン専用アセット）
    → 用途: プリキャッシュされたアプリシェル

  ■ Network Only
    → ネットワークのみ（キャッシュを一切使わない）
    → 用途: 決済処理、リアルタイムデータ
```

```javascript
// Service Worker のキャッシュ戦略実装例
// sw.js

const CACHE_NAME = 'app-cache-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/assets/app.js',
  '/assets/style.css',
];

// インストール時にアプリシェルをプリキャッシュ
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
});

// フェッチイベントの処理
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // 静的アセット: Cache First
  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(cacheFirst(event.request));
    return;
  }

  // API: Network First
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(event.request));
    return;
  }

  // HTML: Stale While Revalidate
  event.respondWith(staleWhileRevalidate(event.request));
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
  }
  return response;
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response('Offline', { status: 503 });
  }
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      const cache = caches.open(CACHE_NAME).then((c) => {
        c.put(request, response.clone());
      });
    }
    return response;
  });

  return cached || fetchPromise;
}
```

---

## 7. アンチパターンとエッジケース

### 7.1 アンチパターン1: max-age のみで immutable を忘れる

```
■ アンチパターン:
  Cache-Control: public, max-age=31536000

  問題:
  - ブラウザが「条件付きリクエスト」を送信する場合がある
  - ページ遷移やリロード時に 304 往復が発生
  - 特に Safari は積極的に再検証を行う傾向がある

  ┌─────────────────────────────────────────────────────┐
  │  ブラウザ                        サーバー            │
  │    │                               │               │
  │    │── GET /app.a1b2.js ──────────▶│               │
  │    │   If-None-Match: "xyz"        │               │
  │    │                               │               │
  │    │◀── 304 Not Modified ──────────│               │
  │    │                               │               │
  │    │  ★ この往復が無駄!             │               │
  │    │  ファイル名にハッシュがあるので │               │
  │    │  内容は絶対に変わらない         │               │
  └─────────────────────────────────────────────────────┘

■ 正しいパターン:
  Cache-Control: public, max-age=31536000, immutable

  → immutable を付けることで、ブラウザは期限内の
    条件付きリクエストを送信しなくなる
  → ページ遷移時のパフォーマンスが向上
  → 特にモバイル環境（低帯域・高レイテンシ）で効果が大きい
```

**影響の程度**: 高トラフィックサイトでは、不要な304リクエストが毎秒数千〜数万回発生する可能性がある。各リクエストのRTTが50ms程度だとしても、ユーザー体験への累積的な影響は無視できない。

### 7.2 アンチパターン2: Vary: * の使用

```
■ アンチパターン:
  Vary: *

  問題:
  - すべてのリクエストが一意のキャッシュキーを持つことになる
  - 事実上、キャッシュが完全に無効化される
  - CDN/プロキシでのキャッシュヒット率が 0% になる
  - 意図せずこうなるケースが多い（フレームワークのデフォルト設定等）

■ よくある原因:
  1. フレームワークが自動的に Vary: * を付与している
  2. ミドルウェアが過剰な Vary ヘッダーを追加している
  3. CORS ミドルウェアが Vary: Origin を追加し、
     他のミドルウェアが Vary: Accept-Encoding を追加し、
     最終的に結合されて Vary: * に変換されるケースがある

■ 正しいパターン:
  必要最小限のヘッダーのみを Vary に指定する
  Vary: Accept-Encoding
  Vary: Accept-Encoding, Accept-Language

■ デバッグ方法:
  curl -I https://example.com/api/data
  → レスポンスの Vary ヘッダーを確認
  → * が含まれていないか、過剰なヘッダーが含まれていないかチェック
```

### 7.3 アンチパターン3: Set-Cookie と Cache-Control の競合

```
■ アンチパターン:
  HTTP/1.1 200 OK
  Cache-Control: public, max-age=3600
  Set-Cookie: session=abc123; HttpOnly; Secure

  問題:
  - Set-Cookie を含むレスポンスが CDN にキャッシュされる
  - 他のユーザーに対してそのセッションCookieが配信される
  - セッションハイジャック等の深刻なセキュリティ脆弱性

■ 正しいパターン:
  選択肢A: キャッシュ不可にする
    Cache-Control: private, no-cache
    Set-Cookie: session=abc123; HttpOnly; Secure

  選択肢B: Set-Cookie を含まないようにする
    Cache-Control: public, max-age=3600
    (Set-Cookie なし — セッション管理は別のエンドポイントで)

  選択肢C: CDN側で Set-Cookie を除去する設定
    (CloudFront の Response Headers Policy 等)
```

### 7.4 エッジケース1: クロックスキューによるキャッシュ異常

```
■ 状況:
  クライアントとサーバーの時計がずれている場合

  サーバー時刻: 2025-01-15 10:00:00
  クライアント時刻: 2025-01-15 09:55:00 (5分遅れ)

  レスポンス:
    Date: Wed, 15 Jan 2025 10:00:00 GMT
    Cache-Control: max-age=300

  クライアントの計算:
    age = client_now - date = -300秒（負の値!）
    → 実装によって挙動が異なる

  ■ 発生しうる問題:
    - キャッシュが期待より長く/短く有効になる
    - Age ヘッダーが負の値になりパースエラーが発生する
    - CDN間でキャッシュの鮮度判定が不整合になる

  ■ 対策:
    - サーバー側: NTP で時刻を正確に同期する
    - アプリ側: Age ヘッダーも活用し、相対的な鮮度を計算する
      response_age = max(0, age_header_value)
      freshness_lifetime = max_age - response_age
    - CDN側: 多くのCDNはAge ヘッダーを自動付与し、
      クロックスキューの影響を軽減する
```

### 7.5 エッジケース2: POST/PUT/DELETE によるキャッシュの暗黙的無効化

```
■ HTTP仕様の規定（RFC 9111 Section 4.4）:

  安全でないメソッド（POST, PUT, DELETE, PATCH）の
  成功レスポンス（2xx）を受信したとき、キャッシュは
  同一URIの保存済みレスポンスを無効化しなければならない（MUST）。

  また、Content-Location または Location ヘッダーの
  URIのキャッシュも無効化しなければならない。

  ■ 例:
    1. GET /api/users/42 → 200 (キャッシュに保存)
    2. PUT /api/users/42 → 200 (成功)
    3. /api/users/42 のキャッシュが自動的に無効化される
    4. 次の GET /api/users/42 → サーバーに問い合わせ

  ■ 注意点:
    - この無効化はローカルキャッシュのみに適用される
    - CDN のキャッシュは自動的には無効化されない
    - CDN のキャッシュ無効化には明示的なパージが必要

  ■ CDN でのベストプラクティス:
    POST /api/users/42 の成功後に:
    1. アプリケーションが CDN パージ API を呼ぶ
    2. または、s-maxage を短く設定して自然な有効期限切れを待つ
    3. または、CDN の「オリジンリクエスト時に更新」機能を使う
```

### 7.6 エッジケース3: Range リクエストとキャッシュ

```
■ 状況:
  大きなファイル（動画等）の部分ダウンロード時

  リクエスト:
    GET /video/lecture.mp4 HTTP/1.1
    Range: bytes=0-1048575

  レスポンス:
    HTTP/1.1 206 Partial Content
    Content-Range: bytes 0-1048575/104857600
    ETag: "abc123"
    Cache-Control: public, max-age=3600

  ■ キャッシュの課題:
    - 部分レスポンス（206）もキャッシュ可能だが、
      キャッシュの実装が複雑になる
    - 同一URLに対して異なる Range のリクエストが来る
    - 一部のCDN は 206 をキャッシュしない設定がデフォルト

  ■ ベストプラクティス:
    - CDN に全体のファイルをキャッシュさせ、
      エッジで Range リクエストに対応させる
    - 強い ETag を使用する（弱い ETag は Range 非対応）
    - CloudFront: 自動的に Range をサポート
    - Cloudflare: Enterprise プランで Range キャッシュ最適化
```

---

## 8. キャッシュのモニタリングとデバッグ

### 8.1 キャッシュヒット率の計測

キャッシュの効果を定量的に把握するには、キャッシュヒット率の継続的な計測が不可欠である。

```
キャッシュヒット率の計算:

  hit_rate = cache_hits / (cache_hits + cache_misses) * 100

  目安:
  ┌──────────────────┬────────────┬─────────────────────────────┐
  │ ヒット率          │ 評価       │ 対応                        │
  ├──────────────────┼────────────┼─────────────────────────────┤
  │ 95%+             │ 優秀       │ 現状維持                    │
  │ 80-95%           │ 良好       │ 微調整で改善可能            │
  │ 50-80%           │ 改善必要   │ TTL、キャッシュキーを見直す  │
  │ 50%未満          │ 要対応     │ 戦略の根本的な見直し        │
  └──────────────────┴────────────┴─────────────────────────────┘
```

### 8.2 デバッグ用ヘッダーの活用

```bash
# curl でキャッシュヘッダーを確認
curl -I https://example.com/assets/app.a1b2c3.js

# 期待されるレスポンスヘッダー:
# HTTP/2 200
# cache-control: public, max-age=31536000, immutable
# etag: "abc123"
# x-cache: Hit from cloudfront           ← CloudFront のキャッシュ状態
# age: 12345                             ← キャッシュに入ってからの経過秒数
# cf-cache-status: HIT                   ← Cloudflare のキャッシュ状態
# x-cache-status: HIT                    ← nginx のキャッシュ状態

# CDN別のキャッシュ状態ヘッダー:
#
# CloudFront:
#   X-Cache: Hit from cloudfront
#   X-Cache: Miss from cloudfront
#   X-Cache: RefreshHit from cloudfront  ← SWRで返却
#
# Cloudflare:
#   CF-Cache-Status: HIT
#   CF-Cache-Status: MISS
#   CF-Cache-Status: EXPIRED
#   CF-Cache-Status: STALE
#   CF-Cache-Status: DYNAMIC             ← キャッシュ対象外
#   CF-Cache-Status: BYPASS
#
# Fastly:
#   X-Cache: HIT
#   X-Cache: MISS
#   X-Cache-Hits: 5                      ← ヒット回数
#   X-Served-By: cache-tyo...            ← 応答したエッジサーバー
```

### 8.3 ブラウザDevToolsによるキャッシュ確認

```
Chrome DevTools での確認手順:

  1. Network タブを開く
  2. 「Disable cache」のチェックを外す（通常のキャッシュ動作を確認）
  3. ページを読み込む
  4. 各リソースの以下を確認:

     Size 列:
       - (disk cache) → ディスクキャッシュから取得
       - (memory cache) → メモリキャッシュから取得
       - (ServiceWorker) → Service Worker から取得
       - 数値 → ネットワークから取得

     Status 列:
       - 200 → 新規取得 or キャッシュから復元
       - 304 → サーバーで検証済み（変更なし）

     Headers タブ:
       - Response Headers の Cache-Control, ETag, Age を確認
       - Request Headers の If-None-Match, If-Modified-Since を確認

  5. 「Disable cache」にチェックを入れると:
     → Cache-Control: no-cache がリクエストに追加される
     → すべてのリソースがネットワークから取得される
     → デバッグ時に有用
```

### 8.4 キャッシュ関連の主要メトリクス

| メトリクス | 計測方法 | 目標値 | 意味 |
|-----------|---------|--------|------|
| CDN ヒット率 | CDNダッシュボード | 90%+ | CDNでの応答割合 |
| 304 レスポンス率 | アクセスログ解析 | HTML: 60%+ | 帯域節約の効果 |
| TTFB（Time To First Byte） | RUM / Synthetic | <200ms | 最初のバイトまでの時間 |
| バイト節約量 | CDNダッシュボード | - | 転送量削減効果 |
| パージ成功率 | CDN API ログ | 99.9%+ | パージの信頼性 |
| stale 配信率 | カスタムヘッダー | <5% | 古いコンテンツ配信の割合 |

---

## 9. 高度なキャッシュパターン

### 9.1 Surrogate Keys によるタグベースパージ

従来のパスベースのパージでは、関連するすべてのURLを列挙する必要がある。Surrogate Keys（タグベースパージ）を使うと、リソースにタグを付与し、タグ単位でパージできる。

```
Surrogate Keys の仕組み（Fastly の例）:

  ■ レスポンスにタグを付与:
    GET /api/products/42

    HTTP/1.1 200 OK
    Surrogate-Key: product-42 category-electronics all-products
    Cache-Control: public, s-maxage=3600

    GET /api/categories/electronics

    HTTP/1.1 200 OK
    Surrogate-Key: category-electronics all-categories
    Cache-Control: public, s-maxage=3600

  ■ 商品42を更新した場合:
    PURGE tag: product-42

    → /api/products/42 がパージされる
    → /api/products/42 を参照する他のURLもパージ可能

  ■ 全商品を更新した場合:
    PURGE tag: all-products

    → all-products タグを持つ全URLが一括パージされる

  利点:
  - パージ対象のURL列挙が不要
  - コンテンツの論理的な関係に基づいた無効化が可能
  - 数千URLの一括パージも高速（Fastly: 150ms以内）
```

### 9.2 Edge Side Includes (ESI)

ESI は、ページの一部を動的に組み立てるためのマークアップ言語である。CDNエッジで処理され、ページの各部分に異なるキャッシュポリシーを適用できる。

```html
<!-- ESI の例: ページ構成 -->
<!-- ヘッダー: ユーザー固有、キャッシュ短め -->
<esi:include src="/fragments/header"
  onerror="continue"
  maxwait="500" />

<!-- メインコンテンツ: パブリック、キャッシュ長め -->
<esi:include src="/fragments/product/42" />

<!-- サイドバー: パブリック、中程度のキャッシュ -->
<esi:include src="/fragments/sidebar/recommendations" />

<!-- フッター: パブリック、長期キャッシュ -->
<esi:include src="/fragments/footer" />

<!--
  /fragments/header       → Cache-Control: private, max-age=60
  /fragments/product/42   → Cache-Control: public, s-maxage=3600
  /fragments/sidebar/...  → Cache-Control: public, s-maxage=600
  /fragments/footer       → Cache-Control: public, s-maxage=86400

  → ページ全体をキャッシュ不可にする必要がない
  → パブリックな部分は CDN にキャッシュされる
  → ユーザー固有部分だけが毎回取得される
-->
```

### 9.3 Cache Stampede（キャッシュスタンピード）対策

```
■ Cache Stampede とは:
  キャッシュの有効期限が切れた瞬間に、多数のリクエストが
  同時にオリジンサーバーに到達する現象。
  「Thundering Herd」問題とも呼ばれる。

  タイムライン:

  ─────────────────────────────┬───────────────────────────
  ◀── キャッシュ有効 ──────────│──── キャッシュ期限切れ ──▶
                               │
                    Request 1 ─┼──▶ Origin ──▶ 応答
                    Request 2 ─┼──▶ Origin ──▶ 応答
                    Request 3 ─┼──▶ Origin ──▶ 応答
                    ...        │
                    Request N ─┼──▶ Origin ──▶ 応答
                               │
                    ★ N個のリクエストが同時にオリジンに殺到
                    ★ オリジンが過負荷になる可能性

■ 対策1: Request Coalescing（リクエスト結合）
  同一キーのリクエストを1つにまとめ、
  結果を全リクエストに配信する。

  ─────────────────────────────┬───────────────────────────
                               │
                    Request 1 ─┤
                    Request 2 ─┼──▶ 1つだけ Origin へ
                    Request 3 ─┤
                               │
                    全リクエストに同じ結果を返す

  nginx: proxy_cache_lock on;
  Varnish: coalescing はデフォルトで有効

■ 対策2: Probabilistic Early Expiration
  キャッシュの期限切れより少し前に、確率的に更新を開始する。

  計算式:
    should_refresh = (random() < beta * log(random()))
                     && (now > expiry - delta)

  → 期限切れ前に1つのリクエストだけが更新を実行
  → 残りのリクエストは既存キャッシュを使い続ける

■ 対策3: stale-while-revalidate
  Cache-Control: max-age=60, stale-while-revalidate=300
  → 期限切れ後も古いキャッシュを返しつつ、
    バックグラウンドで1つだけ更新リクエストを送信
```

### 9.4 マルチテナント環境でのキャッシュ分離

```
■ 課題:
  SaaS アプリケーションで、テナントごとに異なるコンテンツを
  提供する場合、キャッシュキーにテナント識別子を含める必要がある。

■ 方法1: サブドメインベース
  tenant-a.app.example.com → キャッシュキーにホスト名を含む
  tenant-b.app.example.com → 自然にテナント分離される

■ 方法2: パスベース
  app.example.com/tenant-a/api/data
  app.example.com/tenant-b/api/data
  → URLが異なるため自然に分離される

■ 方法3: ヘッダーベース
  app.example.com/api/data
  X-Tenant-ID: tenant-a

  Vary: X-Tenant-ID
  → ヘッダー値ごとに別キャッシュエントリ

  注意: CDN のキャッシュキーに X-Tenant-ID を含める設定が必要
  CloudFront: Cache Policy の Headers に追加
  Cloudflare: Cache Key の Custom Headers に追加

■ セキュリティ上の注意:
  - テナントAのキャッシュがテナントBに配信されないことを
    厳密にテストする
  - Vary ヘッダーの設定漏れは深刻なデータ漏洩になる
  - CDN の設定とアプリケーションの設定を二重にチェックする
```

---

## 10. HTTP/2 および HTTP/3 におけるキャッシュの考慮事項

### 10.1 HTTP/2 Server Push とキャッシュ

```
■ HTTP/2 Server Push の基本:
  サーバーがHTMLレスポンスと一緒に、必要になるであろう
  リソース（CSS, JS等）を先行して送信する機能。

  GET /index.html HTTP/2

  レスポンス:
    PUSH_PROMISE: /assets/style.a1b2.css
    PUSH_PROMISE: /assets/app.c3d4.js

    DATA: <html>...</html>
    DATA: /* style.a1b2.css の内容 */
    DATA: /* app.c3d4.js の内容 */

■ キャッシュとの問題:
  - ブラウザに既にキャッシュがあっても Push される
  - 帯域の浪費になる
  - Chrome 106 以降、Server Push のサポートが削除された

■ 代替手段: 103 Early Hints
  HTTP/1.1 103 Early Hints
  Link: </assets/style.a1b2.css>; rel=preload; as=style
  Link: </assets/app.c3d4.js>; rel=preload; as=script

  HTTP/1.1 200 OK
  Content-Type: text/html
  ...

  → ブラウザはキャッシュを確認してから取得を開始する
  → 不要な転送を回避できる
  → CloudFront, Cloudflare が対応済み
```

### 10.2 HTTP/3 (QUIC) とキャッシュ

```
■ HTTP/3 固有のキャッシュ考慮事項:

  1. 接続の復元（0-RTT）:
     QUIC の 0-RTT ハンドシェイクでは、前回の接続情報を
     キャッシュして再利用する。
     → 接続確立が高速化されるが、リプレイ攻撃のリスクがある
     → 安全でないメソッド（POST等）は 0-RTT で送信すべきでない

  2. サーバー証明書のキャッシュ:
     QUIC は TLS 1.3 を使用し、セッションチケットを
     キャッシュすることで再接続を高速化する

  3. HTTPヘッダー圧縮（QPACK）:
     HTTP/3 では QPACK によるヘッダー圧縮が行われる
     Cache-Control 等の頻出ヘッダーは効率的に圧縮される
     → キャッシュの動作自体は HTTP/2 と同じ

  4. コネクションマイグレーション:
     ネットワーク切り替え（Wi-Fi → モバイル）時にも
     接続が維持されるため、キャッシュの一貫性が保たれる
```

---

## 11. セキュリティとキャッシュ

### 11.1 キャッシュポイズニング攻撃

```
■ Web Cache Poisoning:
  攻撃者がキャッシュに悪意のあるレスポンスを格納させ、
  他のユーザーにそれを配信させる攻撃。

  攻撃手法:
  1. キャッシュキーに含まれないヘッダー（Unkeyed Input）を発見
  2. そのヘッダーがレスポンスに反映されることを確認
  3. 悪意のある値を含むリクエストを送信
  4. CDN がそのレスポンスをキャッシュ
  5. 他のユーザーに悪意のあるレスポンスが配信される

  例:
    GET /page HTTP/1.1
    Host: example.com
    X-Forwarded-Host: evil.com     ← Unkeyed Input

    レスポンス:
    <link href="https://evil.com/style.css" rel="stylesheet">
    → このレスポンスがキャッシュされると、
      全ユーザーに evil.com の CSS が配信される

■ 対策:
  1. Unkeyed Input を排除する
     → レスポンスに反映するヘッダーはすべて Vary に追加
     → 不要なヘッダーの処理をアプリから除去

  2. CDN のキャッシュキーを適切に設定する
     → 必要なヘッダーをキャッシュキーに含める

  3. レスポンスの入力検証を徹底する
     → ヘッダー値を無条件にレスポンスに埋め込まない

  4. Cache-Control: private をデフォルトにする
     → 明示的に public にするリソースのみ CDN キャッシュ
```

### 11.2 Cache Deception 攻撃

```
■ Web Cache Deception:
  攻撃者が被害者に特殊なURLにアクセスさせ、
  被害者の個人データを CDN にキャッシュさせる攻撃。

  攻撃手法:
  1. 攻撃者が被害者に以下のURLを踏ませる:
     https://example.com/api/me/profile/nonexistent.css

  2. サーバーは /api/me/profile のレスポンスを返す
     (パスの末尾を無視するフレームワークの場合)

  3. CDN は .css 拡張子を見て静的ファイルとしてキャッシュ
     Cache-Control: public, max-age=31536000

  4. 攻撃者が同じURLにアクセスし、被害者のプロフィールを取得

■ 対策:
  1. パスの正規化を厳密に行う
     → /api/me/profile/xxx.css は 404 を返す

  2. コンテンツタイプに基づくキャッシュ制御
     → application/json は CDN でキャッシュしない

  3. 拡張子に基づくキャッシュルールを避ける
     → パスパターンではなく、レスポンスヘッダーに基づいてキャッシュ

  4. ユーザー固有レスポンスには必ず Cache-Control: private
```
