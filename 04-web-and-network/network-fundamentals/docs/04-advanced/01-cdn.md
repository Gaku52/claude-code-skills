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
