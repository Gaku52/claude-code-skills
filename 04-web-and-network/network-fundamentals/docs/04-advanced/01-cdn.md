# CDN（Content Delivery Network）

> CDNは世界中のエッジサーバーにコンテンツをキャッシュし、ユーザーに最も近い場所から配信する分散型インフラストラクチャである。レイテンシ削減、帯域節約、オリジン負荷軽減、DDoS防御を同時に実現し、現代のWebサービスにとって不可欠な存在となっている。本章ではCDNの基盤技術からCloudFront/Cloudflareの詳細設定、Edge Computingの最前線までを体系的に解説する。

## この章で学ぶこと

- [ ] CDNの基本アーキテクチャとリクエストルーティングの仕組みを理解する
- [ ] キャッシュ制御ヘッダーとキャッシュ戦略の設計手法を習得する
- [ ] CloudFront / Cloudflare の実践的な設定とデプロイを行える
- [ ] キャッシュパージ戦略とバージョニングの使い分けを判断できる
- [ ] Edge Computingの活用パターンとCloudflare Workersの実装ができる
- [ ] CDN起因の障害パターンを理解し、トラブルシューティングできる

---

## 1. CDNの基本アーキテクチャ

### 1.1 CDNが解決する課題

インターネットにおけるコンテンツ配信は、物理的な距離に起因する3つの根本課題を抱えている。

1. **レイテンシ（遅延）**: 光の速度には限界があり、東京からUS西海岸までの往復は約100ms。TLS ハンドシェイクを含めると初回接続だけで300ms以上かかる
2. **帯域幅の制約**: 海底ケーブルや中継ネットワークの容量には限りがあり、大量のトラフィックが集中するとボトルネックになる
3. **オリジンサーバーの負荷**: 全リクエストがオリジンに到達すると、スケーリングコストが線形に増大する

CDNはこれらの課題を、**地理的に分散配置されたエッジサーバー群**によるコンテンツのキャッシュと配信で解決する。

### 1.2 CDNの全体構成

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CDN アーキテクチャ全体図                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                      │
│   │ オリジン  │   │ オリジン  │   │ オリジン  │   ← Origin Tier     │
│   │ Server A │   │  S3/GCS  │   │ Server B │                      │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘                      │
│        │              │              │                              │
│   ─────┴──────────────┴──────────────┴─────── Origin Shield ───    │
│        │                                                            │
│   ┌────▼──────────────────────────────────────┐                    │
│   │           ミッドティア / シールド層           │  ← Mid-Tier      │
│   │  ┌────────┐  ┌────────┐  ┌────────┐      │                    │
│   │  │ 東京   │  │ 北米   │  │ 欧州   │      │                    │
│   │  │Regional│  │Regional│  │Regional│      │                    │
│   │  └───┬────┘  └───┬────┘  └───┬────┘      │                    │
│   └──────┼───────────┼───────────┼────────────┘                    │
│          │           │           │                                  │
│   ┌──────▼───────────▼───────────▼────────────┐                    │
│   │              エッジ層（PoP）                 │  ← Edge Tier     │
│   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐   │                    │
│   │  │東京│ │大阪│ │SFO│ │NYC│ │LON│ │FRA│   │                    │
│   │  │PoP│ │PoP│ │PoP│ │PoP│ │PoP│ │PoP│   │                    │
│   │  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘   │                    │
│   └────┼─────┼─────┼─────┼─────┼─────┼──────┘                    │
│        │     │     │     │     │     │                              │
│   ─────┴─────┴─────┴─────┴─────┴─────┴───── Internet ─────        │
│        │     │     │     │     │     │                              │
│       👤    👤    👤    👤    👤    👤   ← End Users              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

PoP = Point of Presence（接続拠点）
各PoPには複数のエッジサーバーが配置される
```

### 1.3 リクエストルーティングの仕組み

CDNがユーザーを最寄りのエッジサーバーへ誘導する方法は主に3つ存在する。

**DNS ベースルーティング**

最も一般的な方式。ユーザーのDNSリクエストに対して、地理的に最も近いエッジサーバーのIPアドレスを返す。

```
リクエストルーティング（DNSベース）:

  1. ユーザー（東京）がcdn.example.comにアクセス
     │
     ▼
  2. DNSリゾルバがCDNの権威DNSに問い合わせ
     │
     ▼
  3. CDN DNS がリゾルバのIPから地理情報を推定
     │  ┌─────────────────────────────────────┐
     │  │ リゾルバIP: 203.0.113.1              │
     │  │ → GeoIP判定: 日本 / 東京             │
     │  │ → 最寄りPoP: 東京エッジ (198.51.100.5)│
     │  └─────────────────────────────────────┘
     │
     ▼
  4. ユーザーが東京エッジ(198.51.100.5)に接続
     │
     ▼
  5. エッジサーバーがキャッシュを確認
     ├─ HIT  → キャッシュから即座に応答
     └─ MISS → オリジンから取得 → キャッシュ保存 → 応答
```

**Anycast ルーティング**

Cloudflareが採用する方式。複数のPoPに同一IPアドレスを割り当て、BGPルーティングプロトコルにより最短経路のPoPへ自動的にルーティングされる。DNS解決のオーバーヘッドがなく、DDoS耐性にも優れている。

**HTTP リダイレクト**

初回リクエストに対して302/307リダイレクトで最寄りのエッジURLを返す方式。柔軟だがリダイレクト分の遅延が発生するため、補助的に使われることが多い。

### 1.4 CDNが配信するコンテンツ分類

| カテゴリ | 具体例 | キャッシュ適性 | TTL目安 |
|---------|--------|--------------|---------|
| 静的アセット | CSS, JS, 画像, フォント, favicon | 非常に高い | 1年（ハッシュ付きファイル名） |
| HTML | ページHTML | 中程度 | 0秒〜5分（再検証） |
| メディア | 動画(HLS/DASH), 音声, PDF | 高い | 1日〜1ヶ月 |
| APIレスポンス | 公開API, GraphQLクエリ結果 | 低〜中 | 数秒〜数分 |
| 動的コンテンツ | パーソナライズページ, リアルタイムデータ | 原則なし | Edge Computingで生成 |

---

## 2. キャッシュ制御の深層

### 2.1 HTTPキャッシュヘッダー体系

CDNのキャッシュ動作を制御する主要なHTTPヘッダーを体系的に理解する。

```
HTTPキャッシュヘッダーの優先順位:

  ┌──────────────────────────────────────────────────┐
  │               レスポンスヘッダー                    │
  │                                                    │
  │  1. CDN固有ヘッダー（最優先）                       │
  │     CDN-Cache-Control: max-age=3600               │
  │     Surrogate-Control: max-age=86400              │
  │     CloudFront: Cache-Policy                      │
  │                                                    │
  │  2. Cache-Control（標準・推奨）                     │
  │     Cache-Control: public, max-age=31536000       │
  │     Cache-Control: private, no-cache              │
  │     Cache-Control: s-maxage=600, max-age=60       │
  │                                                    │
  │  3. Expires（レガシー、Cache-Controlが優先）        │
  │     Expires: Thu, 01 Dec 2025 16:00:00 GMT        │
  │                                                    │
  │  4. ETag / Last-Modified（条件付きリクエスト用）    │
  │     ETag: "abc123"                                │
  │     Last-Modified: Wed, 15 Nov 2024 12:00:00 GMT  │
  │                                                    │
  └──────────────────────────────────────────────────┘

  s-maxage vs max-age:
    s-maxage  → 共有キャッシュ（CDN, プロキシ）のTTL
    max-age   → すべてのキャッシュ（ブラウザ含む）のTTL
    → s-maxage は max-age より優先される（CDN上で）
```

### 2.2 Cache-Control ディレクティブ詳解

| ディレクティブ | 対象 | 説明 |
|---------------|------|------|
| `public` | CDN + ブラウザ | 共有キャッシュに保存可能 |
| `private` | ブラウザのみ | CDNではキャッシュしない |
| `no-cache` | 両方 | キャッシュ保存するが毎回再検証 |
| `no-store` | 両方 | 一切キャッシュしない |
| `max-age=N` | 両方 | N秒間キャッシュが有効 |
| `s-maxage=N` | CDNのみ | CDN上でN秒間有効 |
| `stale-while-revalidate=N` | 両方 | 期限切れ後N秒間は古いキャッシュを返しつつバックグラウンド更新 |
| `stale-if-error=N` | 両方 | オリジンエラー時にN秒間は古いキャッシュを返す |
| `must-revalidate` | 両方 | 期限切れ後は必ずオリジンに再検証 |
| `immutable` | 両方 | コンテンツは不変であり再検証不要 |

### 2.3 コード例1: Nginx でのキャッシュ制御ヘッダー設定

```nginx
# /etc/nginx/conf.d/cache-headers.conf
# オリジンサーバー側でのキャッシュ制御ヘッダー設定

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # --- 静的アセット（ハッシュ付きファイル名） ---
    # app.a1b2c3d4.js, style.e5f6g7h8.css のようなファイル
    location ~* \.[a-f0-9]{8}\.(js|css|woff2?|ttf|eot|svg|png|jpg|webp|avif)$ {
        # 1年間キャッシュ + 不変宣言
        add_header Cache-Control "public, max-age=31536000, immutable";
        add_header CDN-Cache-Control "max-age=31536000";

        # CORSヘッダー（フォント等で必要）
        add_header Access-Control-Allow-Origin "*";

        # ETag は不要（immutableのため）
        etag off;
    }

    # --- 通常の静的ファイル（ハッシュなし） ---
    location ~* \.(ico|gif|bmp)$ {
        add_header Cache-Control "public, max-age=86400, stale-while-revalidate=604800";
        etag on;
    }

    # --- HTML ファイル ---
    location ~* \.html$ {
        # CDNでは5分キャッシュ、ブラウザでは毎回再検証
        add_header Cache-Control "public, s-maxage=300, max-age=0, must-revalidate";
        add_header CDN-Cache-Control "max-age=300, stale-while-revalidate=60";
        etag on;
    }

    # --- API レスポンス ---
    location /api/ {
        # CDNでは30秒キャッシュ、ブラウザではキャッシュしない
        add_header Cache-Control "public, s-maxage=30, max-age=0, no-cache";
        add_header Vary "Accept, Accept-Encoding, Authorization";

        # パージ用のサロゲートキー
        add_header Surrogate-Key "api-response";

        proxy_pass http://backend;
    }

    # --- ユーザー固有データ ---
    location /api/me/ {
        # CDNキャッシュ禁止
        add_header Cache-Control "private, no-store, max-age=0";
        proxy_pass http://backend;
    }

    # --- エラーページ ---
    location = /error.html {
        # オリジンダウン時にCDNが古いキャッシュを返す
        add_header Cache-Control "public, max-age=60, stale-if-error=86400";
    }
}
```

### 2.4 キャッシュキーの設計

キャッシュキーは「このリクエストに対してどのキャッシュエントリを返すか」を決定する識別子である。設計を誤ると、キャッシュヒット率の低下（キーが細かすぎる場合）や誤配信（キーが粗すぎる場合）を引き起こす。

```
キャッシュキーの構成要素:

  デフォルト: URL全体（ホスト + パス + クエリ文字列）

  ┌──────────────────────────────────────────────────────────┐
  │                    キャッシュキーの例                       │
  │                                                            │
  │  最小キー（推奨）:                                          │
  │    Host + Path のみ                                        │
  │    cdn.example.com/images/logo.png                         │
  │    → クエリ無視でヒット率最大化                              │
  │                                                            │
  │  中間キー:                                                  │
  │    Host + Path + 必要なクエリのみ                           │
  │    cdn.example.com/api/products?category=shoes              │
  │    → ?utm_source=twitter 等のトラッキングパラメータは除外    │
  │                                                            │
  │  最大キー（非推奨）:                                        │
  │    Host + Path + 全クエリ + 全ヘッダー                     │
  │    → キャッシュが細分化されすぎてヒット率が極端に低下         │
  │                                                            │
  └──────────────────────────────────────────────────────────┘

  Vary ヘッダーによるキー拡張:
    Vary: Accept-Encoding
    → gzip 版と Brotli 版を別キャッシュとして保持

    Vary: Accept
    → image/webp 版と image/jpeg 版を別キャッシュとして保持

    注意: Vary: Cookie は事実上キャッシュ無効化と同義
    （ユーザーごとにCookieが異なるため）
```

---

## 3. AWS CloudFront 実践設定

### 3.1 CloudFront の構成要素

CloudFrontは以下のコンポーネントで構成される。

| コンポーネント | 役割 |
|---------------|------|
| Distribution | CDN配信の設定単位。1ドメインに対し1つ作成 |
| Origin | コンテンツの取得元（S3, ALB, カスタムオリジン等） |
| Behavior | URLパスパターンごとのキャッシュ・転送設定 |
| Cache Policy | キャッシュキーとTTLの設定 |
| Origin Request Policy | オリジンに転送するヘッダー・クエリ・Cookieの設定 |
| Response Headers Policy | レスポンスに付与するセキュリティヘッダー等 |
| Function | CloudFront Functions / Lambda@Edge |

### 3.2 コード例2: CloudFront Distribution（Terraform）

```hcl
# cloudfront.tf
# AWS CloudFront Distribution の Terraform 設定

# --- S3 オリジン用のOAC（Origin Access Control） ---
resource "aws_cloudfront_origin_access_control" "s3_oac" {
  name                              = "s3-oac-${var.environment}"
  description                       = "OAC for S3 static assets"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# --- キャッシュポリシー ---
resource "aws_cloudfront_cache_policy" "static_assets" {
  name        = "static-assets-${var.environment}"
  comment     = "Cache policy for immutable static assets"
  default_ttl = 86400      # 1日
  max_ttl     = 31536000   # 1年
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    cookies_config {
      cookie_behavior = "none"  # Cookieをキャッシュキーに含めない
    }
    headers_config {
      header_behavior = "none"  # ヘッダーをキャッシュキーに含めない
    }
    query_strings_config {
      query_string_behavior = "none"  # クエリ文字列をキャッシュキーに含めない
    }
    enable_accept_encoding_gzip  = true   # gzip圧縮版を自動キャッシュ
    enable_accept_encoding_brotli = true  # Brotli圧縮版を自動キャッシュ
  }
}

resource "aws_cloudfront_cache_policy" "dynamic_content" {
  name        = "dynamic-content-${var.environment}"
  comment     = "Cache policy for API and dynamic content"
  default_ttl = 30    # 30秒
  max_ttl     = 300   # 5分
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    cookies_config {
      cookie_behavior = "none"
    }
    headers_config {
      header_behavior = "whitelist"
      headers {
        items = ["Accept", "Accept-Language"]
      }
    }
    query_strings_config {
      query_string_behavior = "whitelist"
      query_strings {
        items = ["page", "limit", "category", "lang"]
      }
    }
    enable_accept_encoding_gzip   = true
    enable_accept_encoding_brotli = true
  }
}

# --- オリジンリクエストポリシー ---
resource "aws_cloudfront_origin_request_policy" "api_forward" {
  name    = "api-forward-${var.environment}"
  comment = "Forward necessary headers to API origin"

  cookies_config {
    cookie_behavior = "all"  # 全Cookieをオリジンに転送
  }
  headers_config {
    header_behavior = "whitelist"
    headers {
      items = [
        "Accept",
        "Accept-Language",
        "Authorization",
        "Content-Type",
        "Origin",
        "Referer",
        "X-Request-ID"
      ]
    }
  }
  query_strings_config {
    query_string_behavior = "all"  # 全クエリをオリジンに転送
  }
}

# --- レスポンスヘッダーポリシー ---
resource "aws_cloudfront_response_headers_policy" "security_headers" {
  name    = "security-headers-${var.environment}"
  comment = "Security headers for all responses"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = true
      preload                    = true
      override                   = true
    }
    content_type_options {
      override = true  # X-Content-Type-Options: nosniff
    }
    frame_options {
      frame_option = "DENY"
      override     = true
    }
    xss_protection {
      mode_block = true
      protection = true
      override   = true
    }
    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
    content_security_policy {
      content_security_policy = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com"
      override                = true
    }
  }

  custom_headers_config {
    items {
      header   = "Permissions-Policy"
      override = true
      value    = "camera=(), microphone=(), geolocation=()"
    }
    items {
      header   = "X-CDN-Pop"
      override = false
      value    = ""  # CloudFrontが自動でPoP情報を付与
    }
  }
}

# --- メインDistribution ---
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  http_version        = "http2and3"  # HTTP/3 (QUIC) を有効化
  price_class         = "PriceClass_200"  # 北米+欧州+アジア+中東+アフリカ
  default_root_object = "index.html"
  comment             = "Main distribution - ${var.environment}"
  aliases             = [var.domain_name]

  # --- S3 オリジン（静的アセット） ---
  origin {
    domain_name              = aws_s3_bucket.static_assets.bucket_regional_domain_name
    origin_id                = "s3-static"
    origin_access_control_id = aws_cloudfront_origin_access_control.s3_oac.id
    origin_shield {
      enabled              = true
      origin_shield_region = "ap-northeast-1"  # 東京リージョン
    }
  }

  # --- ALB オリジン（API） ---
  origin {
    domain_name = aws_lb.api.dns_name
    origin_id   = "alb-api"
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 30
    }
    custom_header {
      name  = "X-Origin-Verify"
      value = var.origin_verify_secret
    }
  }

  # --- デフォルトビヘイビア（S3静的アセット） ---
  default_cache_behavior {
    target_origin_id           = "s3-static"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD", "OPTIONS"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = aws_cloudfront_cache_policy.static_assets.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id

    # CloudFront Functions でURLリライト
    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.url_rewrite.arn
    }
  }

  # --- APIビヘイビア ---
  ordered_cache_behavior {
    path_pattern               = "/api/*"
    target_origin_id           = "alb-api"
    viewer_protocol_policy     = "https-only"
    allowed_methods            = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = aws_cloudfront_cache_policy.dynamic_content.id
    origin_request_policy_id   = aws_cloudfront_origin_request_policy.api_forward.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id
  }

  # --- ハッシュ付きアセット専用ビヘイビア ---
  ordered_cache_behavior {
    path_pattern           = "/assets/*"
    target_origin_id       = "s3-static"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true
    cache_policy_id        = aws_cloudfront_cache_policy.static_assets.id
  }

  # --- SSL証明書 ---
  viewer_certificate {
    acm_certificate_arn      = var.acm_certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  # --- WAF ---
  web_acl_id = var.waf_web_acl_arn

  # --- アクセスログ ---
  logging_config {
    include_cookies = false
    bucket          = aws_s3_bucket.cf_logs.bucket_domain_name
    prefix          = "cloudfront/${var.environment}/"
  }

  # --- カスタムエラーレスポンス（SPA対応） ---
  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# --- CloudFront Functions（URLリライト） ---
resource "aws_cloudfront_function" "url_rewrite" {
  name    = "url-rewrite-${var.environment}"
  runtime = "cloudfront-js-2.0"
  comment = "Rewrite URLs for SPA routing"
  publish = true
  code    = file("${path.module}/functions/url-rewrite.js")
}
```

### 3.3 CloudFront Functions によるURLリライト

```javascript
// functions/url-rewrite.js
// CloudFront Functions: SPA用URLリライトとセキュリティヘッダー付与

function handler(event) {
    var request = event.request;
    var uri = request.uri;
    var headers = request.headers;

    // --- トレーリングスラッシュの正規化 ---
    // /about/ → /about （ただしルートは除外）
    if (uri.length > 1 && uri.endsWith('/')) {
        return {
            statusCode: 301,
            statusDescription: 'Moved Permanently',
            headers: {
                'location': { value: uri.slice(0, -1) },
                'cache-control': { value: 'max-age=3600' }
            }
        };
    }

    // --- 拡張子がないパスをSPAルートとしてindex.htmlに書き換え ---
    // /about, /products/123 → /index.html
    // /style.css, /app.js → そのまま
    if (!uri.includes('.')) {
        request.uri = '/index.html';
    }

    // --- セキュリティ: パストラバーサル防止 ---
    if (uri.includes('..') || uri.includes('//')) {
        return {
            statusCode: 400,
            statusDescription: 'Bad Request',
            headers: {
                'content-type': { value: 'text/plain' }
            },
            body: 'Invalid request path'
        };
    }

    // --- Accept-Language に基づく言語リダイレクト ---
    if (uri === '/index.html' || uri === '/') {
        var acceptLang = headers['accept-language']
            ? headers['accept-language'].value : '';
        if (acceptLang.startsWith('ja')) {
            request.uri = '/ja/index.html';
        }
    }

    return request;
}
```

### 3.4 コード例3: CloudFront キャッシュパージ（AWS CLI / SDK）

```bash
#!/bin/bash
# scripts/cloudfront-invalidate.sh
# CloudFront キャッシュ無効化スクリプト

DISTRIBUTION_ID="${CF_DISTRIBUTION_ID:?'CF_DISTRIBUTION_ID is required'}"

# --- 個別パスの無効化 ---
invalidate_paths() {
    local paths=("$@")
    echo "Invalidating ${#paths[@]} paths on distribution ${DISTRIBUTION_ID}..."

    aws cloudfront create-invalidation \
        --distribution-id "${DISTRIBUTION_ID}" \
        --paths "${paths[@]}" \
        --output json | jq '{
            InvalidationId: .Invalidation.Id,
            Status: .Invalidation.Status,
            Paths: .Invalidation.InvalidationBatch.Paths.Items,
            CreateTime: .Invalidation.CreateTime
        }'
}

# --- デプロイ後の標準無効化パターン ---
deploy_invalidation() {
    echo "=== Post-Deploy Invalidation ==="

    # HTMLファイルとサービスワーカーのみ無効化
    # （ハッシュ付き静的ファイルは無効化不要）
    invalidate_paths \
        "/index.html" \
        "/ja/index.html" \
        "/en/index.html" \
        "/sw.js" \
        "/manifest.json" \
        "/robots.txt" \
        "/sitemap.xml"
}

# --- 無効化の完了待ち ---
wait_for_invalidation() {
    local invalidation_id="$1"
    echo "Waiting for invalidation ${invalidation_id} to complete..."

    aws cloudfront wait invalidation-completed \
        --distribution-id "${DISTRIBUTION_ID}" \
        --id "${invalidation_id}"

    echo "Invalidation ${invalidation_id} completed."
}

# --- 全キャッシュクリア（緊急時のみ） ---
purge_all() {
    echo "WARNING: Purging ALL cache for distribution ${DISTRIBUTION_ID}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        invalidate_paths "/*"
    else
        echo "Aborted."
    fi
}

# --- メイン ---
case "${1}" in
    deploy)  deploy_invalidation ;;
    purge)   purge_all ;;
    paths)   shift; invalidate_paths "$@" ;;
    *)       echo "Usage: $0 {deploy|purge|paths <path1> <path2> ...}" ;;
esac
```

```python
# scripts/cloudfront_invalidate.py
# Python SDK (boto3) を使ったプログラマティックな無効化

import boto3
import time
from datetime import datetime
from typing import List, Optional

class CloudFrontInvalidator:
    """CloudFront キャッシュ無効化ユーティリティ"""

    def __init__(self, distribution_id: str, region: str = "us-east-1"):
        self.distribution_id = distribution_id
        self.client = boto3.client("cloudfront", region_name=region)

    def invalidate(self, paths: List[str]) -> dict:
        """指定パスのキャッシュを無効化"""
        caller_reference = f"inv-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        response = self.client.create_invalidation(
            DistributionId=self.distribution_id,
            InvalidationBatch={
                "Paths": {
                    "Quantity": len(paths),
                    "Items": paths,
                },
                "CallerReference": caller_reference,
            },
        )

        invalidation = response["Invalidation"]
        return {
            "id": invalidation["Id"],
            "status": invalidation["Status"],
            "paths": paths,
            "created": invalidation["CreateTime"].isoformat(),
        }

    def wait_for_completion(
        self, invalidation_id: str, timeout: int = 600
    ) -> bool:
        """無効化の完了を待機（デフォルト10分タイムアウト）"""
        waiter = self.client.get_waiter("invalidation_completed")
        try:
            waiter.wait(
                DistributionId=self.distribution_id,
                Id=invalidation_id,
                WaiterConfig={
                    "Delay": 10,
                    "MaxAttempts": timeout // 10,
                },
            )
            return True
        except Exception as e:
            print(f"Timeout waiting for invalidation: {e}")
            return False

    def deploy_invalidation(self) -> dict:
        """デプロイ後の標準無効化"""
        standard_paths = [
            "/index.html",
            "/sw.js",
            "/manifest.json",
            "/robots.txt",
            "/sitemap.xml",
        ]
        return self.invalidate(standard_paths)


# 使用例
if __name__ == "__main__":
    invalidator = CloudFrontInvalidator("E1234567890ABC")
    result = invalidator.deploy_invalidation()
    print(f"Invalidation created: {result['id']}")
    invalidator.wait_for_completion(result["id"])
```

---

## 4. Cloudflare 実践設定

### 4.1 Cloudflare のアーキテクチャ特性

CloudflareはAnycastネットワークを基盤とし、全PoPで同一の機能セットを提供する「エブリウェアクラウド」アーキテクチャを採用している。

```
Cloudflare Anycast アーキテクチャ:

  従来のCDN（階層型）:
  ┌─────────────────────────────────────┐
  │  ユーザー → エッジ → リージョン → オリジン │
  │  （3段階のホップ、ミス時は高レイテンシ）   │
  └─────────────────────────────────────┘

  Cloudflare（フラット型）:
  ┌─────────────────────────────────────────────────┐
  │                                                   │
  │   全PoP が同一IP: 104.16.x.x                     │
  │   BGPが自動的に最寄りPoPへルーティング              │
  │                                                   │
  │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐        │
  │   │東京 │   │ SFO │   │ LON │   │ SYD │        │
  │   │ PoP │   │ PoP │   │ PoP │   │ PoP │        │
  │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘        │
  │      │         │         │         │              │
  │   各PoPが以下を全て実行:                           │
  │   ・キャッシュ                                     │
  │   ・WAF / DDoS防御                                │
  │   ・Workers実行                                   │
  │   ・DNS解決                                       │
  │   ・SSL終端                                       │
  │   ・画像最適化                                     │
  │                                                   │
  └─────────────────────────────────────────────────┘
```

### 4.2 コード例4: Cloudflare Workers によるエッジ処理

```javascript
// workers/edge-api-cache.js
// Cloudflare Workers: APIレスポンスのインテリジェントキャッシュ

/**
 * エッジでAPIレスポンスをキャッシュし、
 * stale-while-revalidate パターンを実装するWorker
 */

const CACHE_CONFIG = {
  // パスパターンごとのキャッシュ設定
  '/api/products': { ttl: 300, swr: 600, tags: ['products'] },
  '/api/categories': { ttl: 3600, swr: 7200, tags: ['categories'] },
  '/api/search': { ttl: 60, swr: 120, tags: ['search'] },
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // --- POST/PUT/DELETE はキャッシュしない ---
    if (request.method !== 'GET') {
      return handleMutation(request, env);
    }

    // --- キャッシュ設定の取得 ---
    const cacheConfig = getCacheConfig(url.pathname);
    if (!cacheConfig) {
      // キャッシュ対象外のパスはそのままオリジンへ
      return fetch(request);
    }

    // --- Cache APIでキャッシュ確認 ---
    const cache = caches.default;
    const cacheKey = new Request(url.toString(), {
      method: 'GET',
      headers: { 'Accept': request.headers.get('Accept') || 'application/json' },
    });

    let response = await cache.match(cacheKey);

    if (response) {
      // --- キャッシュヒット ---
      const age = getAge(response);
      const isStale = age > cacheConfig.ttl;

      if (isStale && age < cacheConfig.ttl + cacheConfig.swr) {
        // stale-while-revalidate: 古いキャッシュを返しつつバックグラウンド更新
        ctx.waitUntil(revalidateCache(cache, cacheKey, request, cacheConfig));
        response = new Response(response.body, response);
        response.headers.set('X-Cache', 'STALE');
        response.headers.set('X-Cache-Age', String(age));
        return response;
      }

      if (!isStale) {
        response = new Response(response.body, response);
        response.headers.set('X-Cache', 'HIT');
        response.headers.set('X-Cache-Age', String(age));
        return response;
      }
      // TTL + SWR を超過 → フォールスルーして再取得
    }

    // --- キャッシュミス: オリジンから取得 ---
    const originResponse = await fetch(request);

    if (originResponse.ok) {
      const cachedResponse = new Response(originResponse.body, originResponse);
      cachedResponse.headers.set('Cache-Control', `public, max-age=${cacheConfig.ttl + cacheConfig.swr}`);
      cachedResponse.headers.set('X-Cache-Tags', cacheConfig.tags.join(','));
      cachedResponse.headers.set('X-Cache-Timestamp', String(Date.now()));

      // バックグラウンドでキャッシュ保存
      ctx.waitUntil(cache.put(cacheKey, cachedResponse.clone()));

      const finalResponse = new Response(cachedResponse.body, cachedResponse);
      finalResponse.headers.set('X-Cache', 'MISS');
      return finalResponse;
    }

    // --- オリジンエラー時のフォールバック ---
    if (response) {
      // 古いキャッシュがあれば返す (stale-if-error)
      const fallback = new Response(response.body, response);
      fallback.headers.set('X-Cache', 'STALE-ERROR');
      return fallback;
    }

    return originResponse;
  },
};

function getCacheConfig(pathname) {
  for (const [pattern, config] of Object.entries(CACHE_CONFIG)) {
    if (pathname.startsWith(pattern)) {
      return config;
    }
  }
  return null;
}

function getAge(response) {
  const timestamp = response.headers.get('X-Cache-Timestamp');
  if (!timestamp) return Infinity;
  return Math.floor((Date.now() - parseInt(timestamp)) / 1000);
}

async function revalidateCache(cache, cacheKey, request, config) {
  try {
    const freshResponse = await fetch(request);
    if (freshResponse.ok) {
      const cached = new Response(freshResponse.body, freshResponse);
      cached.headers.set('Cache-Control', `public, max-age=${config.ttl + config.swr}`);
      cached.headers.set('X-Cache-Tags', config.tags.join(','));
      cached.headers.set('X-Cache-Timestamp', String(Date.now()));
      await cache.put(cacheKey, cached);
    }
  } catch (e) {
    console.error('Revalidation failed:', e);
  }
}

async function handleMutation(request, env) {
  const response = await fetch(request);
  // 書き込み操作後にキャッシュをパージ
  if (response.ok) {
    const cache = caches.default;
    // 関連するキャッシュエントリを削除
    // 本番環境ではKVやDurable Objectsでキャッシュキーを管理
  }
  return response;
}
```

### 4.3 Cloudflare ページルールとキャッシュ設定

```toml
# wrangler.toml
# Cloudflare Workers の設定ファイル

name = "edge-api-cache"
main = "workers/edge-api-cache.js"
compatibility_date = "2024-09-01"

# --- 環境設定 ---
[env.production]
routes = [
  { pattern = "api.example.com/api/*", zone_name = "example.com" }
]

[env.staging]
routes = [
  { pattern = "api-staging.example.com/api/*", zone_name = "example.com" }
]

# --- KV Namespace（キャッシュメタデータ用） ---
[[kv_namespaces]]
binding = "CACHE_META"
id = "abc123def456"
preview_id = "789ghi012jkl"

# --- 環境変数 ---
[vars]
ENVIRONMENT = "production"
ORIGIN_URL = "https://origin.example.com"

# --- シークレット（wrangler secret putで設定） ---
# ORIGIN_AUTH_TOKEN
# PURGE_API_KEY
```

---

## 5. キャッシュパージ戦略の詳細設計

### 5.1 パージ方式の比較

| パージ方式 | 精度 | 速度 | コスト | 適用場面 |
|-----------|------|------|--------|---------|
| パス指定パージ | 最高 | 数秒〜数分 | 低（CloudFront: $0.005/パス） | 個別ファイル更新時 |
| ワイルドカードパージ | 高 | 数秒〜数分 | 中 | ディレクトリ単位の更新 |
| タグベースパージ | 高 | 即時（Fastly） | 中 | コンテンツ種別ごとの一括更新 |
| 全パージ | 最低 | 数分 | 低 | 大規模変更・緊急時 |
| バージョニング | 不要 | 即時 | なし | ビルドアセット（推奨） |

### 5.2 バージョニング（パージ不要設計）

パージそのものを不要にする設計が、CDN運用における最良のプラクティスである。

```
バージョニング戦略:

  ビルド時にファイル名にコンテンツハッシュを付与:

  src/                         dist/
  ├── app.js           →      ├── app.a1b2c3d4.js
  ├── style.css        →      ├── style.e5f6g7h8.css
  ├── logo.png         →      ├── logo.i9j0k1l2.png
  └── index.html       →      └── index.html (参照先を更新)

  index.html の中身:
  <link rel="stylesheet" href="/style.e5f6g7h8.css">
  <script src="/app.a1b2c3d4.js"></script>

  デプロイフロー:
  1. 新ビルドアセットをS3にアップロード（新ハッシュ名）
  2. index.html を更新（新ハッシュ名を参照）
  3. index.html のみCDN無効化
  4. 旧アセットはTTL満了まで残る（他ユーザーが使用中の可能性）

  利点:
  ・パージ不要 → オペレーションミスのリスクゼロ
  ・ロールバック容易 → 旧index.htmlに戻すだけ
  ・キャッシュ期間を最大化 → max-age=31536000, immutable
```

---

## 6. キャッシュヒット/ミスフロー詳細

### 6.1 リクエストライフサイクル

```
CDN リクエスト処理フロー（詳細）:

  ブラウザ                    CDNエッジ                    オリジン
    │                           │                           │
    │  GET /api/products        │                           │
    ├──────────────────────────►│                           │
    │                           │                           │
    │                    ┌──────┤ キャッシュルックアップ      │
    │                    │      │                           │
    │                    │  ┌───▼───┐                      │
    │                    │  │キャッシュ│                      │
    │                    │  │  HIT?  │                      │
    │                    │  └───┬───┘                      │
    │                    │      │                           │
    │              ┌─────┴──────┼───────────┐              │
    │              │            │           │              │
    │           [HIT]     [STALE]      [MISS]             │
    │              │            │           │              │
    │              │     Background     ┌───▼───┐          │
    │              │     Revalidate     │ オリジン │          │
    │              │       ┌───────────►│  取得  │          │
    │              │       │           └───┬───┘          │
    │              │       │               │              │
    │              │       │         ┌─────▼─────┐        │
    │              │       │         │200: キャッシュ│        │
    │              │       │         │    保存     │        │
    │              │       │         │304: TTL更新 │        │
    │              │       │         │5xx: stale   │        │
    │              │       │         │    使用可   │        │
    │              │       │         └─────┬─────┘        │
    │  ◄───────────┘       │               │              │
    │  X-Cache: HIT        │  ◄────────────┘              │
    │                      │  X-Cache: MISS               │
    │                                                      │
    │  レスポンスヘッダー例:                                 │
    │  X-Cache: HIT                                        │
    │  X-Cache-Hits: 42                                    │
    │  Age: 120                                            │
    │  CF-Cache-Status: HIT (Cloudflare)                   │
    │  X-Served-By: cache-tyo1234 (Fastly)                │
```

### 6.2 キャッシュステータスの判定

| ステータス | 意味 | 対処 |
|-----------|------|------|
| HIT | キャッシュから応答 | 理想的な状態 |
| MISS | オリジンから取得しキャッシュ保存 | 初回アクセスなら正常 |
| EXPIRED | TTL切れでオリジンに再検証 | TTL設定の見直しを検討 |
| STALE | 古いキャッシュで応答（SWR中） | 正常動作 |
| BYPASS | キャッシュをスキップ | 設定ルールを確認 |
| DYNAMIC | キャッシュ不可と判定 | Cache-Controlヘッダーを確認 |
| REVALIDATED | 304でキャッシュ継続使用 | ETag/Last-Modifiedが正しく動作中 |

---

## 7. CDNサービス詳細比較

### 7.1 機能比較表

```
┌──────────────────┬────────────┬─────────────┬──────────┬──────────┬──────────┐
│ 機能              │ CloudFront │ Cloudflare  │ Fastly   │ Akamai   │ Vercel   │
├──────────────────┼────────────┼─────────────┼──────────┼──────────┼──────────┤
│ PoP数            │ 600+       │ 300+        │ 90+      │ 4,000+   │ 自動     │
│ 無料枠           │ 1TB/月     │ 無制限帯域   │ なし     │ なし     │ 100GB/月 │
│ HTTP/3 (QUIC)    │ 対応       │ 対応        │ 対応     │ 対応     │ 対応     │
│ WebSocket        │ 対応       │ 対応        │ 対応     │ 対応     │ 対応     │
│ Edge Computing   │ CF Func +  │ Workers     │ Compute  │ Edge     │ Edge     │
│                  │ Lambda@Edge│ + Pages     │ @Edge    │ Workers  │ Functions│
│ 即時パージ        │ 数分       │ 数秒        │ <150ms   │ 数秒     │ 自動     │
│ ワイルドカード     │ 対応       │ 対応        │ タグ推奨 │ 対応     │ 自動     │
│   パージ          │            │             │          │          │          │
│ リアルタイムログ   │ Kinesis    │ Logpush     │ 対応     │ DataStr. │ 対応     │
│ DDoS防御         │ Shield     │ 標準搭載    │ 対応     │ Kona     │ 基本     │
│ WAF              │ AWS WAF    │ 標準搭載    │ Next-Gen │ Kona     │ Firewall │
│ 画像最適化        │ Lambda     │ Polish/     │ IO       │ Image    │ 対応     │
│                  │ @Edge      │ Image Resiz.│          │ Manager  │          │
│ TLS証明書        │ ACM無料    │ Universal   │ 対応     │ 対応     │ 自動     │
│                  │            │ SSL無料     │          │          │          │
│ gRPC対応         │ 非対応     │ 対応        │ 対応     │ 対応     │ 非対応   │
│ 料金モデル        │ 従量課金   │ 定額+従量   │ 従量課金 │ 要問合   │ 定額+従量│
└──────────────────┴────────────┴─────────────┴──────────┴──────────┴──────────┘
```

### 7.2 コスト比較表

| 項目 | CloudFront | Cloudflare Pro | Fastly |
|------|-----------|---------------|--------|
| 月額基本料 | $0 | $20/ドメイン | $50〜 |
| 帯域（北米/欧州） | $0.085/GB | 定額内 | $0.12/GB |
| 帯域（アジア） | $0.114/GB | 定額内 | $0.19/GB |
| HTTPS リクエスト | $0.01/万件 | 定額内 | $0.009/万件 |
| Invalidation | $0.005/パス(1,000超) | 無料 | 無料（即時） |
| Edge Computing | $0.6/100万req(CF Func) | $0.5/100万req(Workers) | $0.5/100万req |
| SSL証明書 | 無料（ACM） | 無料（Universal） | 有料オプション |
| DDoS防御 | $3,000/月(Shield Adv.) | 無料（標準搭載） | 有料オプション |

**選定ガイドライン:**

- **AWSエコシステムに統合**: CloudFront（S3/ALB/Lambda との親和性が最高）
- **コスト最小化 + セキュリティ**: Cloudflare（無料プランでもDDoS防御が充実）
- **即時パージが必須**: Fastly（150ms以下のリアルタイムパージ）
- **Next.js アプリケーション**: Vercel（ISR/SSR との統合が最も自然）
- **エンタープライズ + グローバル**: Akamai（PoP数最大、SLAが厳格）

---

## 8. Edge Computing 実践パターン

### 8.1 Edge Computing のユースケース分類

Edge Computingは、従来オリジンサーバーで行っていた処理の一部をCDNエッジで実行する技術である。すべての処理がEdge向きというわけではなく、適切なユースケースの見極めが重要である。

```
Edge Computing 適性マトリックス:

  ┌─────────────────────────────────────────────────────────┐
  │               Edge に適するか？                          │
  │                                                         │
  │  高  │  A/Bテスト    │ パーソナライズ  │                  │
  │  い  │  リダイレクト  │ 画像最適化      │                  │
  │  ↑  │  ヘッダー操作  │ 認証・認可      │                  │
  │  レ  ├───────────────┼────────────────┤                  │
  │  イ  │  Bot検出      │ API集約        │                  │
  │  テ  │  地理制限     │ SSR/ISR         │                  │
  │  ン  │  レート制限    │ HTML変換       │                  │
  │  シ  ├───────────────┼────────────────┤                  │
  │  改  │  ログ収集     │ DB操作          │                  │
  │  善  │              │ バッチ処理       │                  │
  │  低  │              │ 長時間演算       │                  │
  │  い  │              │                 │                  │
  │      └──────────────┴─────────────────┘                  │
  │        低い ←── 計算量 ──→ 高い                          │
  │                                                         │
  │  左上: Edge最適  右上: Edge適  左下: Edge可  右下: 不適   │
  └─────────────────────────────────────────────────────────┘
```

### 8.2 CloudFront Functions vs Lambda@Edge

AWS CloudFront には2種類のEdge Computing機能があり、使い分けが重要である。

| 特性 | CloudFront Functions | Lambda@Edge |
|------|---------------------|-------------|
| 実行タイミング | Viewer Request / Response のみ | Viewer/Origin の Request/Response |
| 実行環境 | JavaScript (ES 5.1 互換) | Node.js / Python |
| 最大実行時間 | 1ms | 5秒 (Viewer) / 30秒 (Origin) |
| メモリ | 2MB | 128MB〜10GB |
| ネットワークアクセス | 不可 | 可 |
| ファイルシステム | 不可 | /tmp (512MB) |
| 料金 | $0.10 / 100万リクエスト | $0.60 / 100万リクエスト + 実行時間 |
| デプロイ | 即時（全PoP） | 数分（レプリカ作成） |
| 適用場面 | ヘッダー操作、URL書換、単純判定 | 外部API呼出、認証、画像変換 |

```
CloudFront Functions と Lambda@Edge の実行ポイント:

  ブラウザ              CloudFront エッジ              オリジン
    │                      │                            │
    │   リクエスト          │                            │
    ├─────────────────────►│                            │
    │                      │                            │
    │              ┌───────▼────────┐                   │
    │              │ Viewer Request │ ← CF Functions    │
    │              │  (URLリライト)  │   Lambda@Edge     │
    │              └───────┬────────┘                   │
    │                      │                            │
    │              ┌───────▼────────┐                   │
    │              │  キャッシュ確認  │                   │
    │              └───────┬────────┘                   │
    │                      │ (MISS時)                   │
    │              ┌───────▼────────┐                   │
    │              │ Origin Request │ ← Lambda@Edge     │
    │              │ (ヘッダー追加)  │   のみ             │
    │              └───────┬────────┘                   │
    │                      ├──────────────────────────►│
    │                      │                            │
    │                      │◄──────────────────────────┤
    │              ┌───────▼────────┐                   │
    │              │Origin Response │ ← Lambda@Edge     │
    │              │ (変換・加工)    │   のみ             │
    │              └───────┬────────┘                   │
    │              ┌───────▼────────┐                   │
    │              │Viewer Response │ ← CF Functions    │
    │              │(セキュリティHdr)│   Lambda@Edge     │
    │              └───────┬────────┘                   │
    │   レスポンス         │                            │
    │◄─────────────────────┤                            │
```

### 8.3 コード例5: Lambda@Edge による画像最適化

```javascript
// lambda/image-optimizer.js
// Lambda@Edge: Origin Response トリガーで画像フォーマットを最適化

const AWS = require('aws-sdk');
const sharp = require('sharp');
const S3 = new AWS.S3({ region: 'ap-northeast-1' });

const SUPPORTED_FORMATS = ['webp', 'avif', 'jpeg', 'png'];
const MAX_WIDTH = 2048;
const MAX_HEIGHT = 2048;
const QUALITY_MAP = {
  webp: 80,
  avif: 65,
  jpeg: 85,
  png: 90,
};

exports.handler = async (event) => {
  const response = event.Records[0].cf.response;
  const request = event.Records[0].cf.request;

  // 画像リクエスト以外はそのまま返す
  if (!isImageRequest(request.uri)) {
    return response;
  }

  // オリジンが200以外ならそのまま返す
  if (response.status !== '200') {
    return response;
  }

  try {
    // クエリパラメータからリサイズ指定を取得
    const params = parseQueryString(request.querystring);
    const width = Math.min(parseInt(params.w) || 0, MAX_WIDTH) || undefined;
    const height = Math.min(parseInt(params.h) || 0, MAX_HEIGHT) || undefined;

    // Accept ヘッダーから最適なフォーマットを決定
    const acceptHeader = request.headers['accept']
      ? request.headers['accept'][0].value
      : '';
    const targetFormat = determineFormat(acceptHeader, request.uri);

    // S3から元画像を取得
    const s3Key = decodeURIComponent(request.uri.substring(1));
    const s3Object = await S3.getObject({
      Bucket: process.env.S3_BUCKET || 'my-images-bucket',
      Key: s3Key,
    }).promise();

    // sharp で変換
    let pipeline = sharp(s3Object.Body);

    // リサイズ（指定がある場合）
    if (width || height) {
      pipeline = pipeline.resize(width, height, {
        fit: 'inside',
        withoutEnlargement: true,
      });
    }

    // フォーマット変換
    const quality = QUALITY_MAP[targetFormat] || 80;
    pipeline = pipeline.toFormat(targetFormat, { quality });

    const optimizedBuffer = await pipeline.toBuffer();

    // 変換後のレスポンスを構築
    const optimizedResponse = {
      status: '200',
      statusDescription: 'OK',
      headers: {
        'content-type': [
          { key: 'Content-Type', value: `image/${targetFormat}` },
        ],
        'cache-control': [
          { key: 'Cache-Control', value: 'public, max-age=31536000, immutable' },
        ],
        'x-image-optimized': [
          { key: 'X-Image-Optimized', value: `format=${targetFormat}, size=${optimizedBuffer.length}` },
        ],
        'vary': [
          { key: 'Vary', value: 'Accept' },
        ],
      },
      body: optimizedBuffer.toString('base64'),
      bodyEncoding: 'base64',
    };

    return optimizedResponse;
  } catch (error) {
    console.error('Image optimization failed:', error);
    // エラー時は元のレスポンスをそのまま返す
    return response;
  }
};

function isImageRequest(uri) {
  return /\.(jpe?g|png|gif|webp|avif|svg)$/i.test(uri);
}

function parseQueryString(qs) {
  if (!qs) return {};
  return qs.split('&').reduce((acc, pair) => {
    const [key, value] = pair.split('=');
    acc[decodeURIComponent(key)] = decodeURIComponent(value || '');
    return acc;
  }, {});
}

function determineFormat(acceptHeader, uri) {
  // AVIF 対応ブラウザ
  if (acceptHeader.includes('image/avif')) return 'avif';
  // WebP 対応ブラウザ
  if (acceptHeader.includes('image/webp')) return 'webp';
  // 元のフォーマットを維持
  const ext = uri.split('.').pop().toLowerCase();
  if (ext === 'jpg') return 'jpeg';
  return ext;
}
```

---

## 9. CDN セキュリティ

### 9.1 DDoS 防御

CDNはその分散アーキテクチャにより、DDoS攻撃の吸収と緩和に優れている。

```
DDoS 防御の多層構造:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Layer 7 (Application):                                  │
  │  ┌──────────────────────────────────────────────┐       │
  │  │  WAF ルール                                   │       │
  │  │  ・SQLインジェクション検出                      │       │
  │  │  ・XSS パターンブロック                         │       │
  │  │  ・レートリミッティング                         │       │
  │  │  ・Bot検出 / CAPTCHA チャレンジ                │       │
  │  └──────────────────────────────────────────────┘       │
  │                                                          │
  │  Layer 4 (Transport):                                    │
  │  ┌──────────────────────────────────────────────┐       │
  │  │  SYN Flood 防御                               │       │
  │  │  ・SYN Cookie                                 │       │
  │  │  ・接続レート制限                              │       │
  │  │  ・GeoIP ブロック                              │       │
  │  └──────────────────────────────────────────────┘       │
  │                                                          │
  │  Layer 3 (Network):                                      │
  │  ┌──────────────────────────────────────────────┐       │
  │  │  ボリューム型攻撃の吸収                        │       │
  │  │  ・Anycast による分散                          │       │
  │  │  ・ブラックホールルーティング                    │       │
  │  │  ・帯域幅: Tbps 級の吸収能力                   │       │
  │  └──────────────────────────────────────────────┘       │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

### 9.2 オリジン保護

CDN導入時にオリジンサーバーの直接アクセスを防がなければ、CDNをバイパスした攻撃が可能になる。

**CloudFront + ALB のオリジン保護例:**

```
オリジン保護パターン:

  方法1: カスタムヘッダーによる検証
  ┌────────────────────────────────────────────────────┐
  │  CloudFront → (X-Origin-Verify: secret123) → ALB  │
  │  ALB の WAF ルールで X-Origin-Verify を検証         │
  │  ヘッダーなし or 不一致 → 403 拒否                  │
  └────────────────────────────────────────────────────┘

  方法2: AWS マネージドプレフィックスリスト
  ┌────────────────────────────────────────────────────┐
  │  ALB のセキュリティグループで                        │
  │  CloudFront の IP レンジのみ許可                     │
  │  com.amazonaws.global.cloudfront.origin-facing      │
  │  マネージドプレフィックスリストを参照                  │
  └────────────────────────────────────────────────────┘

  方法3: Cloudflare Authenticated Origin Pulls
  ┌────────────────────────────────────────────────────┐
  │  Cloudflare ⇔ オリジン間で相互TLS認証              │
  │  Cloudflare の TLS クライアント証明書を検証          │
  │  証明書なし → オリジンが接続拒否                     │
  └────────────────────────────────────────────────────┘

  方法4: Cloudflare Tunnel (推奨)
  ┌────────────────────────────────────────────────────┐
  │  オリジンがインバウンドポートを開放しない             │
  │  cloudflared デーモンがアウトバウンド接続で           │
  │  Cloudflare ネットワークにトンネルを張る             │
  │  → オリジンのIPアドレスが完全に隠蔽される            │
  └────────────────────────────────────────────────────┘
```

### 9.3 HTTPS / TLS 設定のベストプラクティス

| 設定項目 | 推奨値 | 理由 |
|---------|--------|------|
| 最小TLSバージョン | TLS 1.2 | TLS 1.0/1.1 は既知の脆弱性あり |
| HSTS | max-age=31536000; includeSubDomains; preload | ダウングレード攻撃防止 |
| OCSP Stapling | 有効 | 証明書検証の高速化 |
| CT ログ | 有効 | 不正証明書の検出 |
| SSL Mode (Cloudflare) | Full (Strict) | オリジンとの通信も暗号化 + 証明書検証 |
| Origin Protocol (CloudFront) | HTTPS Only | オリジンとの通信を暗号化 |

---

## 10. パフォーマンス最適化

### 10.1 圧縮

CDNは自動的にレスポンスを圧縮して帯域を節約できる。

| 圧縮方式 | 圧縮率 | CPU負荷 | ブラウザ対応 | CDN対応 |
|---------|--------|---------|-------------|---------|
| gzip | 60-70% | 低 | ほぼ全て | 全CDN |
| Brotli (br) | 70-80% | 中〜高 | モダンブラウザ | 主要CDN |
| zstd | 70-80% | 低〜中 | 一部 | Cloudflare |

**圧縮対象のContent-Type:**

```
圧縮すべきMIMEタイプ:
  text/html
  text/css
  text/javascript / application/javascript
  application/json
  application/xml / text/xml
  image/svg+xml
  application/wasm
  font/woff  (woff2は既に圧縮済み)

圧縮してはいけないもの:
  image/jpeg, image/png, image/webp  (既に圧縮済み)
  video/*, audio/*  (既に圧縮済み)
  font/woff2  (既に圧縮済み)
  application/zip, application/gzip  (既に圧縮済み)
```

### 10.2 HTTP/2 と HTTP/3 の活用

```
プロトコル進化と CDN の対応:

  HTTP/1.1:
  ┌──────────────────────────────────────────────┐
  │  1接続 = 1リクエスト（同時接続6本制限）        │
  │  Head-of-Line Blocking あり                   │
  │  ヘッダー圧縮なし                              │
  │  → ドメインシャーディングが必要だった           │
  └──────────────────────────────────────────────┘

  HTTP/2:
  ┌──────────────────────────────────────────────┐
  │  1接続で多重リクエスト（ストリーム）            │
  │  HPACKヘッダー圧縮                             │
  │  サーバープッシュ（廃止傾向）                   │
  │  TCP レベルの HoL Blocking は残存              │
  │  → CDNはH2を標準サポート                       │
  └──────────────────────────────────────────────┘

  HTTP/3 (QUIC):
  ┌──────────────────────────────────────────────┐
  │  UDP ベース（TCP HoL Blocking を解消）         │
  │  0-RTT ハンドシェイク（再接続時）               │
  │  接続マイグレーション（Wi-Fi⇔モバイル切替）     │
  │  QPACK ヘッダー圧縮                            │
  │  → モバイルユーザーに特に効果大                 │
  │  → CloudFront/Cloudflare は対応済み            │
  └──────────────────────────────────────────────┘

  CDN での推奨設定:
  ・エッジ ⇔ ブラウザ: HTTP/3 有効化
  ・エッジ ⇔ オリジン: HTTP/2 で十分（QUICのメリット小）
```

### 10.3 Origin Shield（キャッシュ階層化）

Origin Shield は、エッジとオリジンの間に追加のキャッシュ層を配置し、オリジンへのリクエストを集約する機能である。

```
Origin Shield の効果:

  Shield なし:
  ┌──────────────────────────────────────┐
  │  東京PoP ──(MISS)──► オリジン       │
  │  大阪PoP ──(MISS)──► オリジン       │
  │  福岡PoP ──(MISS)──► オリジン       │
  │  ソウルPoP ──(MISS)──► オリジン     │
  │  シンガポールPoP ──(MISS)──► オリジン │
  │                                      │
  │  → 5つのPoPそれぞれがオリジンに問合せ │
  │  → オリジンへのリクエスト = 5         │
  └──────────────────────────────────────┘

  Shield あり（東京リージョン）:
  ┌──────────────────────────────────────────────┐
  │  東京PoP ──(MISS)──► Shield(東京) ──► オリジン│
  │  大阪PoP ──(MISS)──► Shield(東京) ──(HIT)    │
  │  福岡PoP ──(MISS)──► Shield(東京) ──(HIT)    │
  │  ソウルPoP ──(MISS)──► Shield(東京) ──(HIT)  │
  │  シンガポールPoP ──(MISS)──► Shield(東京)(HIT)│
  │                                                │
  │  → 全PoPがShield経由                            │
  │  → オリジンへのリクエスト = 1                    │
  │  → オリジン負荷を最大80%削減                     │
  └──────────────────────────────────────────────┘
```

---

## 11. アンチパターン

### 11.1 アンチパターン1: Cache-Control ヘッダーの矛盾

**問題:**

```
# 悪い例: 矛盾するキャッシュヘッダー
Cache-Control: public, no-cache, max-age=3600
```

`no-cache` と `max-age=3600` は意味的に矛盾する。`no-cache` は「キャッシュしてよいが毎回オリジンに再検証せよ」という意味であり、`max-age=3600` は「3600秒間は再検証不要」という意味である。CDNによってどちらを優先するかの挙動が異なり、予期しないキャッシュ動作の原因になる。

**正しい設定パターン:**

```
# パターンA: 短時間キャッシュ（CDNのみ）
# 意図: CDNで5分キャッシュ、ブラウザは毎回再検証
Cache-Control: public, s-maxage=300, max-age=0, must-revalidate

# パターンB: キャッシュするが毎回再検証
# 意図: キャッシュは保持するが、使用前にETag/Last-Modifiedで304確認
Cache-Control: public, no-cache
ETag: "v1.2.3"

# パターンC: 完全にキャッシュしない
# 意図: ユーザー固有データなどキャッシュ厳禁
Cache-Control: private, no-store, max-age=0

# パターンD: 長期キャッシュ（ハッシュ付きファイル）
# 意図: 不変ファイルを最大限キャッシュ
Cache-Control: public, max-age=31536000, immutable
```

**影響の重大さ:** キャッシュヘッダーの設定ミスは、古いコンテンツの配信（ユーザーに古いJSが残り続ける等）や、個人情報の漏洩（`public` が付いた認証付きレスポンスがCDNにキャッシュされる等）を引き起こす可能性がある。

### 11.2 アンチパターン2: Vary ヘッダーの過剰設定

**問題:**

```
# 悪い例: Vary に Cookie を含める
Vary: Accept-Encoding, Cookie, User-Agent

# 結果:
# ユーザーAの Cookie: session=abc123
# ユーザーBの Cookie: session=def456
# → 全く同じコンテンツなのに異なるキャッシュエントリが作成される
# → キャッシュヒット率が事実上 0% に近づく
# → CDNが存在しないのと同等の状態
```

**正しいアプローチ:**

```
# 良い例1: 最小限の Vary
Vary: Accept-Encoding
# → gzip版とBrotli版のみを区別

# 良い例2: コンテンツネゴシエーションが必要な場合
Vary: Accept-Encoding, Accept-Language
# → 言語別にキャッシュを分離（言語数は有限）

# Cookie による分岐が必要な場合の代替策:
# → Edge Computing で Cookie を解析し、
#    キャッシュキーに国コードや会員種別のみ含める
# → Vary: Cookie の代わりに Vary: X-User-Segment
#    （Edge が Cookie を解析して X-User-Segment に変換）
```

### 11.3 アンチパターン3: TTL なしのキャッシュ設定

**問題:**

Cache-Control ヘッダーを一切設定しないままCDNを導入するケース。CDNごとにデフォルトの挙動が異なり、意図しないキャッシュ（POST レスポンスのキャッシュ等）や、全くキャッシュされない（オリジン負荷が減らない）状態を招く。

```
# 悪い例: ヘッダーなし
HTTP/1.1 200 OK
Content-Type: text/html
# Cache-Control が存在しない

# CDNごとのデフォルト動作:
# CloudFront: Cache-Controlがなければデフォルトで24時間キャッシュ
#             (Cache Policy の Default TTL による)
# Cloudflare: Cache-Controlがなければオリジンの指示に従う
#             (指示なし = キャッシュしない場合が多い)
# → 同じコンテンツでもCDNによって動作が異なる
```

**解決策:** すべてのレスポンスに明示的な `Cache-Control` ヘッダーを設定する。オリジンのアプリケーションフレームワークのミドルウェアとして一元管理することを推奨する。

---

## 12. エッジケース分析

### 12.1 エッジケース1: キャッシュスタンピード（Thunder Herd Problem）

**現象:** 人気コンテンツのキャッシュTTLが同時に満了し、多数のエッジサーバーが一斉にオリジンへリクエストを送信する現象。オリジンが過負荷でダウンする可能性がある。

```
キャッシュスタンピード:

  通常時:
  PoP-A ──(HIT)──► キャッシュ応答
  PoP-B ──(HIT)──► キャッシュ応答
  PoP-C ──(HIT)──► キャッシュ応答
  → オリジンへのリクエスト: 0

  TTL満了の瞬間:
  PoP-A ──(MISS)──► オリジン ←── 100 req/s
  PoP-B ──(MISS)──► オリジン ←── 100 req/s  → 合計300 req/s
  PoP-C ──(MISS)──► オリジン ←── 100 req/s     オリジン過負荷!

  対策:
  1. stale-while-revalidate
     → TTL切れ直後も古いキャッシュを返しつつバックグラウンド更新
     → Cache-Control: s-maxage=300, stale-while-revalidate=60

  2. Request Coalescing (Fastly: Request Collapsing)
     → 同時期の同一リクエストを1つにまとめてオリジンに転送
     → 100リクエストが来ても、オリジンへは1リクエスト

  3. Origin Shield
     → 全PoPのMISSをShield層で集約
     → Shieldがオリジンに1回だけ問い合わせ

  4. TTLジッタリング
     → TTLにランダムな揺らぎを加えて一斉満了を防ぐ
     → 例: TTL = 300 + random(0, 60) 秒
```

### 12.2 エッジケース2: Set-Cookie とキャッシュの干渉

**現象:** オリジンが `Set-Cookie` ヘッダー付きのレスポンスを返すと、それがCDNにキャッシュされ、全ユーザーに同じCookieが配信されるセキュリティインシデントが発生する。

```
Set-Cookie 問題:

  1. ユーザーAがログイン
  2. オリジンが応答:
     HTTP/1.1 200 OK
     Set-Cookie: session=USER_A_SESSION; Path=/
     Cache-Control: public, max-age=300   ← 問題の根源
     Content-Type: text/html

  3. CDNがこのレスポンスをキャッシュ
     （Set-Cookie ヘッダーごとキャッシュ）

  4. ユーザーBが同じページにアクセス
  5. CDNがキャッシュからSet-Cookie付きで応答
     → ユーザーBにユーザーAのセッションCookieが設定される
     → セッションハイジャックの発生

  対策:
  1. Set-Cookie を含むレスポンスは private, no-store にする
  2. CDN設定でSet-Cookie付きレスポンスのキャッシュを禁止
     CloudFront: Cache Policy で Cookie を「なし」に設定
     Cloudflare: Page Rule で "Cache Level: Bypass"
  3. Set-Cookie はAPI応答のみで返し、
     HTML/JSとは分離する（フロントで Cookie 設定）
```

### 12.3 エッジケース3: CORS と CDN のキャッシュ

**現象:** CDNが異なるOriginヘッダーのリクエストに対して同じキャッシュを返すことで、CORSエラーが発生する。

```
CORS + CDN の問題:

  1. https://app-a.example.com から CDN 上の画像をリクエスト
     Origin: https://app-a.example.com
     → オリジンが応答:
       Access-Control-Allow-Origin: https://app-a.example.com
     → CDN がキャッシュ

  2. https://app-b.example.com から同じ画像をリクエスト
     Origin: https://app-b.example.com
     → CDN がキャッシュから応答:
       Access-Control-Allow-Origin: https://app-a.example.com  ← 不一致!
     → ブラウザが CORS エラーを出す

  対策:
  1. Vary: Origin を設定
     → Origin ヘッダーの値ごとに別キャッシュを保持
     → ただしキャッシュ効率は若干低下

  2. ワイルドカードを使用（公開リソースの場合）
     Access-Control-Allow-Origin: *
     → すべての Origin に対応
     → ただし credentials: 'include' と併用不可

  3. Edge Function で Origin を検証して動的にヘッダー付与
     → キャッシュキーに Origin を含めなくてよい
     → Edge で Allow-Origin を書き換え
```
