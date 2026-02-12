# CDN (Content Delivery Network)

> 世界中に分散配置されたエッジサーバーを活用し、ユーザーに最も近い拠点からコンテンツを配信することでレイテンシを最小化し、オリジンサーバーの負荷を軽減する技術を、CloudFront・Cloudflare・Fastly の比較を通じて解説する

## この章で学ぶこと

1. **CDN の基本原理** --- エッジキャッシュ、オリジンシールド、POP (Point of Presence) の仕組みと、レイテンシ削減のメカニズム
2. **主要 CDN サービスの比較** --- Amazon CloudFront、Cloudflare、Fastly の特性と選定基準
3. **キャッシュ戦略と無効化** --- Cache-Control ヘッダー、キャッシュキー設計、パージ戦略の実践
4. **エッジコンピューティング** --- CDN エッジでのコード実行によるオリジン負荷の削減手法
5. **セキュリティと可用性** --- DDoS 防御、WAF、TLS 終端、オリジン保護の設計

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| HTTP プロトコル | 基礎 | [Web 基礎](../../04-web-and-network/) |
| キャッシング | 基礎 | [キャッシング](./01-caching.md) |
| ロードバランサー | 基礎 | [ロードバランサー](./00-load-balancer.md) |
| DNS | 基礎 | [Web 基礎](../../04-web-and-network/) |

---

## 0. WHY --- なぜ CDN が必要か

### 0.1 CDN なしの課題

```
ユーザー (東京) --------> オリジン (バージニア)
   RTT: 180ms             処理: 50ms

合計レスポンス時間: 180ms (RTT) + 50ms (処理) + 180ms (RTT) = 410ms
TLS ハンドシェイクを含めると: ~600ms

--- CDN 導入後 ---

ユーザー (東京) --> CDN Edge (東京)
   RTT: 5ms       キャッシュ HIT

合計レスポンス時間: 5ms (RTT) + 0ms (処理) + 5ms (RTT) = 10ms
→ 60倍の高速化
```

### 0.2 CDN がもたらす価値

```
              CDN の 4つの価値

 ┌──────────────────────────────────────────────┐
 │                                              │
 │  1. レイテンシ削減                             │
 │     └─ 地理的に近いエッジから配信              │
 │        例: 180ms → 5ms (97%削減)              │
 │                                              │
 │  2. オリジン負荷軽減                           │
 │     └─ キャッシュヒットによりリクエスト到達を   │
 │        85-99% 削減                            │
 │                                              │
 │  3. 可用性向上                                │
 │     └─ オリジン障害時もキャッシュから配信       │
 │        (stale-if-error)                      │
 │                                              │
 │  4. セキュリティ                              │
 │     └─ DDoS 吸収、WAF、TLS 終端              │
 │        エッジで攻撃を遮断                     │
 │                                              │
 └──────────────────────────────────────────────┘
```

### 0.3 定量的な効果

| 指標 | CDN なし | CDN あり | 改善率 |
|------|---------|---------|--------|
| TTFB (Time to First Byte) | 400-800ms | 10-50ms | 90-98% |
| ページロード時間 (3G) | 8-15s | 2-5s | 60-75% |
| オリジンサーバー負荷 | 100% | 5-20% | 80-95% |
| 帯域コスト | $10,000/月 | $2,000/月 | 80% |
| 可用性 (SLA) | 99.9% | 99.99% | 10x 改善 |
| DDoS 耐性 | 数 Gbps | 数 Tbps | 1000x |

---

## 1. CDN の基本アーキテクチャ

### 1.1 リクエストフロー

```
ユーザー (東京)                    CDN Edge (東京 POP)           オリジンサーバー (us-east-1)
     |                                  |                              |
     |--- DNS 解決 ------------------>  |                              |
     |<-- CDN Edge の IP アドレス -----  |                              |
     |                                  |                              |
     |--- GET /img/hero.jpg ---------->|                              |
     |                                  |-- キャッシュ確認              |
     |                                  |   HIT? --> 即座にレスポンス   |
     |                                  |   MISS? ----GET /img/hero.jpg-->|
     |                                  |<--------- 200 OK + データ ---|
     |                                  |-- キャッシュ保存              |
     |<--------- 200 OK + データ -------|                              |
     |   (X-Cache: Miss from CDN)       |                              |
     |                                  |                              |
     |--- GET /img/hero.jpg ---------->|                              |
     |<--------- 200 OK (Cache HIT) ---|  (オリジンへ問い合わせなし)    |
     |   (X-Cache: Hit from CDN)        |                              |
     |   (Age: 120)                     |                              |
```

### 1.2 グローバル POP 配置と Anycast

```
                        CDN グローバルネットワーク

   北米                   ヨーロッパ                アジア太平洋
  +-------+              +-------+              +-------+
  | POP   |              | POP   |              | POP   |
  | NYC   |              | LDN   |              | TYO   |  <-- ユーザー最寄り
  +-------+              +-------+              +-------+
  | POP   |              | POP   |              | POP   |
  | SFO   |              | FRA   |              | SIN   |
  +-------+              +-------+              +-------+
  | POP   |              | POP   |              | POP   |
  | IAD   |              | AMS   |              | SYD   |
  +-------+              +-------+              +-------+
       \                    |                    /
        +------- Origin Shield (中間キャッシュ) --+
                          |
                   +-------------+
                   |   Origin    |
                   |   Server    |
                   +-------------+

  Anycast ルーティング:
  全 POP が同一 IP アドレスを広告
  → BGP により最短経路の POP に自動ルーティング
  → DNS ベースよりも高速・正確な最寄り POP 選択
```

### 1.3 キャッシュ階層と Origin Shield

```
レイヤー 1: ブラウザキャッシュ     (RTT = 0ms)
    ↓ MISS
レイヤー 2: CDN Edge (POP)       (RTT = 1-20ms)
    ↓ MISS
レイヤー 3: Origin Shield         (RTT = 20-50ms)
    ↓ MISS
レイヤー 4: Origin Server         (RTT = 50-300ms)

  キャッシュヒット率の目標:
  - 静的アセット: 95%+ (L2 で HIT)
  - 動的コンテンツ: 60-80% (短TTL + Stale-While-Revalidate)
  - API レスポンス: 30-60% (キャッシュキー設計が鍵)

  Origin Shield の効果:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  Origin Shield なし:                                │
  │    POP_TYO ─┐                                      │
  │    POP_SIN ─┼─→ Origin  (3 リクエスト到達)          │
  │    POP_SYD ─┘                                      │
  │                                                     │
  │  Origin Shield あり:                                │
  │    POP_TYO ─┐                                      │
  │    POP_SIN ─┼─→ Shield ──→ Origin  (1 リクエスト)   │
  │    POP_SYD ─┘    (集約)                             │
  │                                                     │
  │  → オリジンへのリクエスト数を 60-90% 削減            │
  └─────────────────────────────────────────────────────┘
```

### 1.4 CDN の DNS 解決プロセス

```python
"""CDN の DNS 解決フローの解説"""

# 1. ユーザーが cdn.example.com にアクセス
# 2. DNS 解決の流れ:

#   cdn.example.com
#     → CNAME: d123.cloudfront.net
#       → Anycast IP: 13.224.x.x (最寄り POP の IP)

# GeoDNS を使う場合 (Cloudflare):
#   cdn.example.com
#     → Anycast IP: 104.16.x.x
#     → BGP ルーティングで最寄り POP に到達

# 3. DNS TTL の設計
dns_config = {
    "cdn.example.com": {
        "type": "CNAME",
        "value": "d123456.cloudfront.net",
        "ttl": 300,  # 5分: CDN 切り替えに対応できる短さ
    },
    # A/AAAA レコード (Cloudflare Anycast)
    "api.example.com": {
        "type": "A",
        "value": "104.16.132.229",  # Anycast IP
        "ttl": 300,
        "proxied": True,  # Cloudflare プロキシ有効
    }
}
```

---

## 2. Cache-Control の設計

### 2.1 ヘッダー指示子一覧

| 指示子 | 意味 | 使用例 |
|--------|------|--------|
| `public` | CDN・ブラウザ両方でキャッシュ可 | 静的アセット |
| `private` | ブラウザのみキャッシュ可 | ユーザー固有コンテンツ |
| `no-cache` | 毎回オリジンに検証（ETag/Last-Modified） | 最新性が重要な API |
| `no-store` | 一切キャッシュ禁止 | 個人情報、決済ページ |
| `max-age=N` | N秒間キャッシュ有効 | 一般的な TTL 制御 |
| `s-maxage=N` | CDN 用の max-age（ブラウザは max-age を使う） | CDN とブラウザで TTL を分離 |
| `stale-while-revalidate=N` | 期限切れ後 N 秒間は古いキャッシュを返しつつ裏で更新 | 高可用性 API |
| `stale-if-error=N` | オリジンエラー時に N 秒間は古いキャッシュを返す | 可用性重視 |
| `immutable` | コンテンツは変更されない（再検証不要） | ハッシュ付きアセット |
| `must-revalidate` | 期限切れ後は必ずオリジンに検証 | HTML ページ |
| `no-transform` | CDN/プロキシによる変換を禁止 | 画像最適化を防止 |

### 2.2 Cache-Control 決定フローチャート

```
コンテンツの種類は？
  │
  ├─ 個人情報・決済関連 ──→ private, no-store
  │
  ├─ ユーザー固有の動的コンテンツ ──→ private, max-age=0, must-revalidate
  │
  ├─ 共有動的コンテンツ (API)
  │   │
  │   ├─ リアルタイム性が必要 ──→ public, s-maxage=5, stale-while-revalidate=30
  │   └─ 数分の遅延許容 ──→ public, s-maxage=300, stale-while-revalidate=600
  │
  ├─ HTML ページ ──→ public, s-maxage=60, max-age=0, must-revalidate
  │
  └─ 静的アセット
      │
      ├─ ハッシュ付き (main.a1b2c3.js) ──→ public, max-age=31536000, immutable
      └─ ハッシュなし (logo.png) ──→ public, max-age=86400
```

### 2.3 アセット種別ごとの設定例

```nginx
# Nginx での Cache-Control 設定例

# ハッシュ付き静的アセット（CSS/JS）: 1年キャッシュ + immutable
location ~* \.(?:css|js)$ {
    # ファイル名にハッシュ: main.a1b2c3.js
    add_header Cache-Control "public, max-age=31536000, immutable";
    add_header X-Content-Type-Options "nosniff";
    # Brotli / Gzip 圧縮 (事前圧縮ファイルがある場合)
    gzip_static on;
    brotli_static on;
}

# 画像: 30日キャッシュ + WebP/AVIF 自動変換
location ~* \.(?:jpg|jpeg|png|gif|webp|avif|svg)$ {
    add_header Cache-Control "public, max-age=2592000";
    add_header Vary "Accept";  # Accept ヘッダーでコンテンツネゴシエーション
}

# フォント: 1年キャッシュ + CORS
location ~* \.(?:woff2?|ttf|otf|eot)$ {
    add_header Cache-Control "public, max-age=31536000, immutable";
    add_header Access-Control-Allow-Origin "*";
}

# HTML: CDN 60秒、ブラウザはキャッシュなし
location ~* \.html$ {
    add_header Cache-Control "public, s-maxage=60, max-age=0, must-revalidate";
    # オリジン障害時は5分間古いキャッシュを返す
    add_header Cache-Control "stale-if-error=300" always;
}

# API レスポンス: CDN 10秒 + stale-while-revalidate
location /api/ {
    add_header Cache-Control "public, s-maxage=10, stale-while-revalidate=60, stale-if-error=300";
    add_header Vary "Accept-Encoding, Authorization";
}

# 個人情報: キャッシュ禁止
location /api/user/profile {
    add_header Cache-Control "private, no-store";
    add_header Pragma "no-cache";  # HTTP/1.0 互換
}
```

### 2.4 ETag と条件付きリクエスト

```python
"""ETag を使った条件付きリクエストの実装"""
import hashlib
from fastapi import FastAPI, Request, Response

app = FastAPI()

def generate_etag(content: bytes) -> str:
    """コンテンツから ETag を生成"""
    return f'"{hashlib.sha256(content).hexdigest()[:16]}"'

@app.get("/api/products/{product_id}")
async def get_product(product_id: str, request: Request):
    product = await fetch_product(product_id)
    content = json.dumps(product).encode()
    etag = generate_etag(content)

    # クライアントの If-None-Match をチェック
    if_none_match = request.headers.get("If-None-Match")
    if if_none_match == etag:
        # コンテンツ未変更 → 304 を返す (帯域節約)
        return Response(status_code=304, headers={
            "ETag": etag,
            "Cache-Control": "public, s-maxage=30, stale-while-revalidate=60",
        })

    return Response(
        content=content,
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "public, s-maxage=30, stale-while-revalidate=60",
            "Vary": "Accept-Encoding",
        }
    )

# リクエスト・レスポンスフロー:
#
# 1回目: GET /api/products/123
#   → 200 OK + ETag: "a1b2c3d4e5f6g7h8"
#
# 2回目: GET /api/products/123
#        If-None-Match: "a1b2c3d4e5f6g7h8"
#   → 304 Not Modified (ボディなし、帯域を節約)
#
# CDN の動作:
#   CDN キャッシュ期限切れ → オリジンに条件付きリクエスト
#   → 304 → CDN キャッシュを更新して TTL リセット
#   → オリジンのデータ転送量を大幅削減
```

---

## 3. Amazon CloudFront の設定

### 3.1 ディストリビューション作成

```python
# CloudFront ディストリビューション作成 (boto3)
import boto3

cf = boto3.client('cloudfront')

distribution_config = {
    'CallerReference': 'my-app-2026',
    'Origins': {
        'Quantity': 2,
        'Items': [
            {
                # S3 オリジン (静的アセット)
                'Id': 'S3-static-assets',
                'DomainName': 'my-static-assets.s3.amazonaws.com',
                'S3OriginConfig': {
                    'OriginAccessIdentity':
                        'origin-access-identity/cloudfront/XXXXXXX'
                },
                'OriginShield': {
                    'Enabled': True,
                    'OriginShieldRegion': 'ap-northeast-1',  # 東京
                },
            },
            {
                # ALB オリジン (動的 API)
                'Id': 'ALB-api',
                'DomainName': 'api-internal.example.com',
                'CustomOriginConfig': {
                    'HTTPPort': 80,
                    'HTTPSPort': 443,
                    'OriginProtocolPolicy': 'https-only',
                    'OriginSslProtocols': {
                        'Quantity': 1, 'Items': ['TLSv1.2']
                    },
                    'OriginKeepaliveTimeout': 60,  # Keep-Alive 秒数
                    'OriginReadTimeout': 30,
                },
            },
        ]
    },
    'DefaultCacheBehavior': {
        'TargetOriginId': 'S3-static-assets',
        'ViewerProtocolPolicy': 'redirect-to-https',
        'CachePolicyId': '658327ea-f89d-4fab-a63d-7e88639e58f6',
        'Compress': True,              # Brotli/Gzip 自動圧縮
        'AllowedMethods': {'Quantity': 2, 'Items': ['GET', 'HEAD']},
    },
    'CacheBehaviors': {
        'Quantity': 1,
        'Items': [{
            'PathPattern': '/api/*',
            'TargetOriginId': 'ALB-api',
            'ViewerProtocolPolicy': 'https-only',
            'AllowedMethods': {
                'Quantity': 7,
                'Items': ['GET','HEAD','OPTIONS','PUT','POST','PATCH','DELETE'],
            },
            'CachePolicyId': '4135ea2d-6df8-44a3-9df3-4b5a84be39ad',
            'OriginRequestPolicyId':
                '216adef6-5c7f-47e4-b989-5492eafa07d3',  # AllViewer
            'ForwardedValues': {
                'QueryString': True,
                'Headers': {'Quantity': 3, 'Items': [
                    'Authorization', 'Accept', 'Content-Type'
                ]},
                'Cookies': {'Forward': 'none'},
            },
        }],
    },
    'Enabled': True,
    'PriceClass': 'PriceClass_200',    # 北米・欧州・アジア
    'HttpVersion': 'http2and3',        # HTTP/2 + HTTP/3 (QUIC) 有効化
    'Comment': 'Production distribution with S3 + ALB origins',
}

response = cf.create_distribution(DistributionConfig=distribution_config)
print(f"Distribution ID: {response['Distribution']['Id']}")
print(f"Domain: {response['Distribution']['DomainName']}")
```

### 3.2 CloudFront Functions (軽量エッジ処理)

```javascript
// CloudFront Functions: URL リライト + セキュリティヘッダー追加
// 実行環境: JavaScript (ES 5.1), 最大実行時間: 1ms, 最大メモリ: 2MB

function handler(event) {
    var request = event.request;
    var uri = request.uri;

    // 1. SPA のフォールバック: /app/* → /app/index.html
    if (uri.startsWith('/app/') && !uri.includes('.')) {
        request.uri = '/app/index.html';
    }

    // 2. 拡張子がない場合は .html を追加
    if (!uri.includes('.') && uri !== '/') {
        request.uri = uri + '/index.html';
    }

    // 3. セキュリティヘッダーの追加 (レスポンスイベントの場合)
    if (event.response) {
        var response = event.response;
        response.headers['strict-transport-security'] = {
            value: 'max-age=63072000; includeSubDomains; preload'
        };
        response.headers['x-content-type-options'] = {
            value: 'nosniff'
        };
        response.headers['x-frame-options'] = {
            value: 'DENY'
        };
        response.headers['x-xss-protection'] = {
            value: '1; mode=block'
        };
        response.headers['content-security-policy'] = {
            value: "default-src 'self'; script-src 'self' 'unsafe-inline'; " +
                   "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;"
        };
        return response;
    }

    return request;
}
```

### 3.3 Lambda@Edge (高度なエッジ処理)

```python
# Lambda@Edge: A/B テスト + 画像最適化ルーティング
# 実行環境: Node.js / Python, 最大実行時間: 5s (viewer) / 30s (origin)

import json
import hashlib

def viewer_request_handler(event, context):
    """ビューワーリクエスト: A/B テスト振り分け"""
    request = event['Records'][0]['cf']['request']
    headers = request['headers']

    # Cookie から A/B テストグループを判定
    cookies = headers.get('cookie', [{}])[0].get('value', '')
    ab_group = None

    for cookie in cookies.split(';'):
        cookie = cookie.strip()
        if cookie.startswith('ab_group='):
            ab_group = cookie.split('=')[1]
            break

    # 未割り当てならハッシュベースで振り分け
    if not ab_group:
        client_ip = event['Records'][0]['cf']['request']['clientIp']
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        ab_group = 'A' if hash_val % 100 < 50 else 'B'

    # カスタムヘッダーでオリジンに伝達
    request['headers']['x-ab-group'] = [{'key': 'X-AB-Group', 'value': ab_group}]

    return request

def origin_response_handler(event, context):
    """オリジンレスポンス: A/B グループ Cookie を設定"""
    response = event['Records'][0]['cf']['response']
    request = event['Records'][0]['cf']['request']

    ab_group = request['headers'].get('x-ab-group', [{}])[0].get('value', 'A')

    # Set-Cookie で A/B グループを永続化
    response['headers']['set-cookie'] = [{
        'key': 'Set-Cookie',
        'value': f'ab_group={ab_group}; Path=/; Max-Age=604800; SameSite=Lax'
    }]

    return response
```

### 3.4 キャッシュ無効化

```bash
# CloudFront キャッシュ無効化（パージ）
# 注意: 1,000パス/月まで無料、超過は $0.005/パス

# 特定パスのパージ
aws cloudfront create-invalidation \
  --distribution-id E1234567890 \
  --paths "/index.html" "/api/*"

# 全キャッシュのパージ (緊急時のみ)
aws cloudfront create-invalidation \
  --distribution-id E1234567890 \
  --paths "/*"

# パージ状況の確認
aws cloudfront get-invalidation \
  --distribution-id E1234567890 \
  --id I1234567890
```

```python
# Python でのキャッシュバスティング実装
import boto3
import hashlib
import json
from datetime import datetime

class CDNInvalidator:
    """CloudFront キャッシュ無効化マネージャー"""

    def __init__(self, distribution_id: str):
        self.cf = boto3.client('cloudfront')
        self.distribution_id = distribution_id

    def invalidate_paths(self, paths: list[str]) -> str:
        """指定パスのキャッシュを無効化"""
        caller_ref = f"inv-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        response = self.cf.create_invalidation(
            DistributionId=self.distribution_id,
            InvalidationBatch={
                'Paths': {
                    'Quantity': len(paths),
                    'Items': paths,
                },
                'CallerReference': caller_ref,
            }
        )
        invalidation_id = response['Invalidation']['Id']
        print(f"Invalidation created: {invalidation_id}")
        print(f"Status: {response['Invalidation']['Status']}")
        return invalidation_id

    def wait_for_invalidation(self, invalidation_id: str):
        """無効化完了まで待機"""
        waiter = self.cf.get_waiter('invalidation_completed')
        print(f"Waiting for invalidation {invalidation_id}...")
        waiter.wait(
            DistributionId=self.distribution_id,
            Id=invalidation_id,
            WaiterConfig={'Delay': 10, 'MaxAttempts': 30}
        )
        print("Invalidation completed.")

    def deploy_with_cache_busting(
        self, s3_bucket: str, local_dir: str, html_files: list[str]
    ):
        """
        キャッシュバスティングを含むデプロイ:
        1. ハッシュ付きアセットをアップロード (パージ不要)
        2. HTML ファイルをアップロード (短 TTL)
        3. HTML のみパージ
        """
        s3 = boto3.client('s3')
        html_paths = []

        for html_file in html_files:
            s3.upload_file(
                Filename=f"{local_dir}/{html_file}",
                Bucket=s3_bucket,
                Key=html_file,
                ExtraArgs={
                    'ContentType': 'text/html',
                    'CacheControl': 'public, s-maxage=60, max-age=0, must-revalidate',
                }
            )
            html_paths.append(f"/{html_file}")

        # HTML のみパージ (ハッシュ付きアセットはパージ不要)
        if html_paths:
            inv_id = self.invalidate_paths(html_paths)
            self.wait_for_invalidation(inv_id)

# 使用例
invalidator = CDNInvalidator("E1234567890")

# デプロイ
invalidator.deploy_with_cache_busting(
    s3_bucket="my-static-assets",
    local_dir="./dist",
    html_files=["index.html", "about/index.html"]
)
```

---

## 4. Cloudflare の設定

### 4.1 Cloudflare Workers によるエッジコンピューティング

```javascript
// Cloudflare Workers: エッジでの API レスポンスキャッシュ
// V8 isolate ベース: 起動時間 < 1ms, CPU 50ms/リクエスト

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cache = caches.default;

    // GET リクエストのみキャッシュ
    if (request.method !== 'GET') {
      return fetch(request);
    }

    // 1. キャッシュチェック
    const cacheKey = new Request(url.toString(), request);
    let response = await cache.match(cacheKey);

    if (response) {
      // Cache HIT: ヘッダーを追加して返却
      const newResponse = new Response(response.body, response);
      newResponse.headers.set('X-Cache-Status', 'HIT');
      return newResponse;
    }

    // 2. オリジンからフェッチ
    response = await fetch(request);

    // 3. 成功レスポンスのみキャッシュ
    if (response.ok) {
      const cacheResponse = new Response(response.body, response);
      cacheResponse.headers.set(
        'Cache-Control', 'public, s-maxage=300, stale-while-revalidate=600'
      );
      cacheResponse.headers.set('X-Cache-Status', 'MISS');

      // 非同期でキャッシュ書き込み（レスポンスを遅延させない）
      const ctx = { waitUntil: (p) => p };
      ctx.waitUntil(cache.put(cacheKey, cacheResponse.clone()));

      return cacheResponse;
    }

    return response;
  }
};
```

### 4.2 Cloudflare Workers: 地理情報ベースルーティング

```javascript
// Cloudflare Workers: 地理情報に基づくオリジン選択
export default {
  async fetch(request, env) {
    const cf = request.cf;  // Cloudflare のリクエストメタデータ

    // 地理情報の取得
    const country = cf.country;       // "JP"
    const continent = cf.continent;   // "AS"
    const city = cf.city;             // "Tokyo"
    const latitude = cf.latitude;
    const longitude = cf.longitude;
    const asn = cf.asn;               // ISP の AS 番号

    // 大陸ごとに最寄りのオリジンサーバーを選択
    const origins = {
      'AS': 'https://api-ap.example.com',    // アジア
      'NA': 'https://api-us.example.com',    // 北米
      'EU': 'https://api-eu.example.com',    // ヨーロッパ
      'SA': 'https://api-us.example.com',    // 南米 → 北米にフォールバック
      'AF': 'https://api-eu.example.com',    // アフリカ → 欧州にフォールバック
      'OC': 'https://api-ap.example.com',    // オセアニア → アジアにフォールバック
    };

    const originUrl = origins[continent] || origins['NA'];

    // オリジンにリクエスト転送
    const url = new URL(request.url);
    const originRequest = new Request(
      `${originUrl}${url.pathname}${url.search}`,
      request
    );

    const response = await fetch(originRequest);

    // レスポンスにルーティング情報を追加 (デバッグ用)
    const newResponse = new Response(response.body, response);
    newResponse.headers.set('X-Origin-Region', continent);
    newResponse.headers.set('X-Client-Country', country);

    return newResponse;
  }
};
```

### 4.3 Cloudflare Workers KV: エッジデータストア

```javascript
// Workers KV を使ったエッジでの設定管理
// KV: 結果整合性、読み取り最適化 (60秒以内に伝播)

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 機能フラグの取得 (エッジで即座にレスポンス)
    if (url.pathname === '/api/feature-flags') {
      const flags = await env.FEATURE_FLAGS.get('current', { type: 'json' });
      return new Response(JSON.stringify(flags), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=30',
        }
      });
    }

    // レートリミット (エッジで実装)
    if (url.pathname.startsWith('/api/')) {
      const clientIp = request.headers.get('CF-Connecting-IP');
      const rateLimitKey = `ratelimit:${clientIp}`;

      // 現在のリクエスト数を取得
      const current = parseInt(await env.RATE_LIMITS.get(rateLimitKey) || '0');

      if (current >= 100) {  // 100 req/min
        return new Response('Rate limit exceeded', {
          status: 429,
          headers: {
            'Retry-After': '60',
            'X-RateLimit-Limit': '100',
            'X-RateLimit-Remaining': '0',
          }
        });
      }

      // カウントを更新 (TTL: 60秒)
      await env.RATE_LIMITS.put(rateLimitKey, String(current + 1), {
        expirationTtl: 60
      });
    }

    return fetch(request);
  }
};
```

---

## 5. キャッシュキー設計

### 5.1 キャッシュキーの構成要素

```
キャッシュキー = URL + Vary ヘッダーで指定された要素

デフォルトキー:  scheme + host + path + query string
例: https://example.com/api/products?page=1&sort=price

カスタムキー要素:
  - ヘッダー: Accept-Encoding, Accept-Language
  - Cookie: session_id, ab_group
  - デバイス: Mobile / Desktop
  - 地域: 国コード
```

### 5.2 キャッシュキー最適化の実装

```python
"""キャッシュキー設計のベストプラクティス"""
from urllib.parse import urlparse, parse_qs, urlencode

class CacheKeyBuilder:
    """CDN キャッシュキーの設計と正規化"""

    # キャッシュに影響しないパラメータ (除外対象)
    EXCLUDED_PARAMS = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term',
        'utm_content',    # UTM トラッキングパラメータ
        'fbclid',         # Facebook クリック ID
        'gclid',          # Google クリック ID
        '_ga',            # Google Analytics
        'ref', 'source',  # リファラルパラメータ
        '_t', 'timestamp', 'nocache',  # キャッシュバスティング
    }

    # キャッシュキーに含めるパラメータ (ホワイトリスト方式)
    INCLUDED_PARAMS = {
        '/api/products': {'page', 'sort', 'category', 'limit'},
        '/api/search': {'q', 'page', 'sort', 'filters'},
    }

    @classmethod
    def normalize_cache_key(cls, url: str) -> str:
        """URL を正規化してキャッシュキーを生成"""
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        # 方法1: 除外リスト方式
        filtered_params = {
            k: v for k, v in params.items()
            if k.lower() not in cls.EXCLUDED_PARAMS
        }

        # パラメータをソートして正規化
        sorted_params = urlencode(
            sorted(filtered_params.items()), doseq=True
        )

        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" + \
               (f"?{sorted_params}" if sorted_params else "")

    @classmethod
    def normalize_with_whitelist(cls, url: str, path: str) -> str:
        """ホワイトリスト方式でキャッシュキーを生成"""
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        allowed = cls.INCLUDED_PARAMS.get(path, set())
        filtered_params = {
            k: v for k, v in params.items() if k in allowed
        }

        sorted_params = urlencode(
            sorted(filtered_params.items()), doseq=True
        )
        return f"{parsed.path}?{sorted_params}" if sorted_params else parsed.path

# テスト
urls = [
    "https://example.com/api/products?page=1&sort=price&utm_source=google",
    "https://example.com/api/products?utm_source=twitter&page=1&sort=price",
    "https://example.com/api/products?page=1&sort=price&fbclid=abc123",
]

for url in urls:
    normalized = CacheKeyBuilder.normalize_cache_key(url)
    print(f"Original: {url}")
    print(f"Cache Key: {normalized}")
    print()

# 出力:
# 3つのURLが全て同じキャッシュキーに正規化される:
# https://example.com/api/products?page=1&sort=price
```

### 5.3 Vary ヘッダーの設計

```python
"""Vary ヘッダーによるキャッシュバリエーション管理"""
from fastapi import FastAPI, Request, Response

app = FastAPI()

@app.get("/api/products")
async def get_products(request: Request):
    """デバイス・言語に応じたレスポンスバリエーション"""
    # Accept-Language からロケール判定
    lang = request.headers.get("Accept-Language", "en")
    locale = "ja" if "ja" in lang else "en"

    # User-Agent からデバイス判定
    ua = request.headers.get("User-Agent", "")
    is_mobile = "Mobile" in ua

    products = await fetch_products(locale=locale, mobile=is_mobile)

    return Response(
        content=json.dumps(products),
        headers={
            "Content-Type": "application/json",
            "Cache-Control": "public, s-maxage=300",
            # Vary: CDN に「これらのヘッダーが異なれば別キャッシュ」と指示
            "Vary": "Accept-Language, Accept-Encoding",
            # 注意: Vary: User-Agent は絶対に使わない！
            # → 無数のバリエーションが生成されキャッシュが効かなくなる
            # 代わりにCDN のデバイス検出機能を利用
        }
    )

# Vary ヘッダーの効果:
#
# Vary: Accept-Encoding の場合:
#   GET /api/products (Accept-Encoding: gzip)  → キャッシュ A (gzip版)
#   GET /api/products (Accept-Encoding: br)    → キャッシュ B (brotli版)
#   GET /api/products (Accept-Encoding: なし)   → キャッシュ C (非圧縮版)
#
# Vary: Accept-Language の場合:
#   GET /api/products (Accept-Language: ja)     → キャッシュ D (日本語版)
#   GET /api/products (Accept-Language: en)     → キャッシュ E (英語版)
```

---

## 6. 主要 CDN サービス比較

### 比較表 1: 機能比較

| 特性 | CloudFront | Cloudflare | Fastly |
|-----|-----------|------------|--------|
| **POP 数** | 450+ | 300+ | 90+ |
| **エッジコンピューティング** | Lambda@Edge / CloudFront Functions | Workers (V8 isolate) | Compute@Edge (Wasm) |
| **無料枠** | 1TB/月 (12ヶ月) | 無制限帯域（Free プラン） | なし |
| **キャッシュパージ速度** | 数秒 ~ 数十秒 | < 30ms (Instant Purge) | < 150ms |
| **DDoS 防御** | AWS Shield Standard (無料) | 標準搭載（全プラン） | Shield |
| **WAF** | AWS WAF (別料金) | 標準搭載 (Pro 以上) | Next-Gen WAF |
| **HTTP/3 (QUIC)** | 対応 | 対応 | 対応 |
| **WebSocket** | 対応 | 対応 | 対応 |
| **画像最適化** | 非対応 (Lambda@Edge で実装) | Polish / Image Resizing | Image Optimizer |
| **ログ** | S3 / Kinesis | Logpush | Real-time log streaming |
| **価格モデル** | 従量課金 (リクエスト + 帯域) | プラン + 従量課金 | 従量課金 |
| **最適用途** | AWS エコシステム統合 | 汎用・セキュリティ重視 | 高速パージ・API キャッシュ |

### 比較表 2: ユースケース別選定ガイド

| ユースケース | 推奨 CDN | 理由 |
|------------|---------|------|
| AWS S3/ALB との統合 | CloudFront | IAM ベースのアクセス制御、OAI/OAC |
| 即時キャッシュパージが必要 | Cloudflare / Fastly | < 150ms で全 POP パージ |
| 無料で始めたい | Cloudflare | Free プラン: 無制限帯域 + DDoS 防御 |
| エッジでの JS 実行 | Cloudflare | Workers: < 1ms 起動、50ms CPU/req |
| 動的 API のキャッシュ | Fastly | Surrogate-Key でタグベースパージ |
| グローバル映像配信 | CloudFront | MediaStore + CloudFront の統合 |
| マルチクラウド環境 | Cloudflare / Fastly | クラウド非依存 |
| エッジでの Wasm 実行 | Fastly | Compute@Edge: Rust/Go/JS 対応 |

### 比較表 3: コスト比較 (月間 10TB 配信)

| コスト項目 | CloudFront | Cloudflare Pro | Fastly |
|-----------|-----------|---------------|--------|
| 基本料金 | $0 | $20/月 | $50/月 |
| 帯域 (10TB, 北米) | ~$850 | $0 (無制限) | ~$800 |
| HTTPS リクエスト (1億) | ~$100 | $0 (含む) | ~$75 |
| キャッシュパージ | $0 (1000/月無料) | $0 (含む) | $0 (含む) |
| WAF | ~$5/ルール | $0 (含む) | 別料金 |
| **合計 (概算)** | **~$950** | **~$20** | **~$925** |
| **備考** | AWS 統合の価値 | コスパ最強 | パージ速度の価値 |

---

## 7. 高度なトピック

### 7.1 Surrogate-Key (タグベースパージ)

```python
"""Fastly の Surrogate-Key を使ったタグベースキャッシュパージ"""
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    product = await fetch_product(product_id)

    return Response(
        content=json.dumps(product),
        headers={
            "Content-Type": "application/json",
            "Cache-Control": "public, s-maxage=3600",
            # Surrogate-Key: 複数のタグを設定
            # → タグ単位でパージ可能
            "Surrogate-Key": f"product-{product_id} "
                             f"category-{product['category']} "
                             f"all-products",
        }
    )

@app.get("/api/categories/{category_id}")
async def get_category(category_id: str):
    products = await fetch_products_by_category(category_id)

    return Response(
        content=json.dumps(products),
        headers={
            "Content-Type": "application/json",
            "Cache-Control": "public, s-maxage=3600",
            "Surrogate-Key": f"category-{category_id} all-products",
        }
    )

# パージの例:
# 1. 特定商品を更新した場合:
#    POST /service/{id}/purge/product-123
#    → product-123 のキャッシュのみパージ
#
# 2. カテゴリ全体を更新した場合:
#    POST /service/{id}/purge/category-electronics
#    → electronics カテゴリの全商品 + カテゴリ一覧をパージ
#
# 3. 全商品をパージ:
#    POST /service/{id}/purge/all-products
#
# メリット:
# - URL ベースのパージより柔軟
# - ワイルドカード不要で正確なパージが可能
# - 関連コンテンツを一括パージ
```

### 7.2 画像最適化 CDN パイプライン

```python
"""CDN エッジでの画像最適化パイプライン"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ImageFormat(Enum):
    WEBP = "webp"
    AVIF = "avif"
    JPEG = "jpeg"
    PNG = "png"

@dataclass
class ImageTransform:
    width: Optional[int] = None
    height: Optional[int] = None
    quality: int = 80
    format: ImageFormat = ImageFormat.WEBP
    fit: str = "cover"  # cover, contain, fill

class ImageOptimizationCDN:
    """
    CDN エッジでの画像最適化設計

    URL パターン: /images/{transforms}/{original_path}
    例: /images/w_800,h_600,q_80,f_webp/photos/hero.jpg
    """

    # レスポンシブ画像のプリセット
    PRESETS = {
        'thumbnail': ImageTransform(width=150, height=150, quality=60),
        'card':      ImageTransform(width=400, height=300, quality=75),
        'hero':      ImageTransform(width=1920, height=1080, quality=80),
        'og':        ImageTransform(width=1200, height=630, quality=85),
    }

    @staticmethod
    def generate_srcset(image_path: str, widths: list[int]) -> str:
        """レスポンシブ画像の srcset を生成"""
        srcset_entries = []
        for w in widths:
            url = f"/images/w_{w},f_auto/{image_path}"
            srcset_entries.append(f"{url} {w}w")
        return ",\n  ".join(srcset_entries)

    @staticmethod
    def generate_picture_tag(image_path: str) -> str:
        """<picture> タグを生成 (AVIF > WebP > JPEG フォールバック)"""
        return f"""<picture>
  <!-- AVIF (最高圧縮率) -->
  <source
    type="image/avif"
    srcset="{ImageOptimizationCDN.generate_srcset(image_path, [400, 800, 1200])}"
    sizes="(max-width: 768px) 100vw, 50vw" />
  <!-- WebP (広いブラウザ対応) -->
  <source
    type="image/webp"
    srcset="{ImageOptimizationCDN.generate_srcset(image_path, [400, 800, 1200])}"
    sizes="(max-width: 768px) 100vw, 50vw" />
  <!-- JPEG フォールバック -->
  <img
    src="/images/w_800,f_jpeg/{image_path}"
    loading="lazy"
    decoding="async"
    alt="" />
</picture>"""

# Cloudflare Workers での画像最適化リクエスト例:
#
# export default {
#   async fetch(request) {
#     const url = new URL(request.url);
#     const accept = request.headers.get('Accept') || '';
#
#     // ブラウザの対応フォーマットを判定
#     let format = 'jpeg';
#     if (accept.includes('image/avif')) format = 'avif';
#     else if (accept.includes('image/webp')) format = 'webp';
#
#     // Cloudflare Image Resizing を利用
#     return fetch(url.toString(), {
#       cf: {
#         image: {
#           width: 800,
#           height: 600,
#           quality: 80,
#           format: format,
#           fit: 'cover',
#         }
#       }
#     });
#   }
# };
```

### 7.3 CDN とセキュリティ

```python
"""CDN セキュリティ設計: DDoS 防御 + WAF + Bot 管理"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CDNSecurityConfig:
    """CDN セキュリティ設定の設計"""

    # 1. TLS / SSL 設定
    tls_config: dict = field(default_factory=lambda: {
        'min_version': 'TLSv1.2',
        'preferred_ciphers': [
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256',
        ],
        'hsts': {
            'enabled': True,
            'max_age': 63072000,  # 2年
            'include_subdomains': True,
            'preload': True,
        },
        'ocsp_stapling': True,
    })

    # 2. WAF ルール
    waf_rules: dict = field(default_factory=lambda: {
        'managed_rules': [
            'OWASP Top 10',           # SQL Injection, XSS 等
            'Known Bad Inputs',        # 既知の攻撃パターン
            'Bot Management',          # 悪質ボット検出
        ],
        'custom_rules': [
            {
                'name': 'Block non-standard methods',
                'condition': 'http.request.method not in {"GET" "POST" "PUT" "DELETE" "PATCH" "OPTIONS"}',
                'action': 'block',
            },
            {
                'name': 'Rate limit login',
                'condition': 'http.request.uri.path eq "/api/auth/login"',
                'action': 'rate_limit',
                'rate': '10 per minute per ip',
            },
        ],
    })

    # 3. オリジン保護
    origin_protection: dict = field(default_factory=lambda: {
        # オリジンは CDN からのリクエストのみ受け付ける
        'allowed_ips': 'CDN IP ranges only',
        'origin_secret_header': {
            'name': 'X-Origin-Verify',
            'value': 'shared-secret-value',  # CDN → Origin の認証
        },
        # CloudFront: OAC (Origin Access Control)
        'oac_enabled': True,
    })

    # 4. DDoS 防御レイヤー
    ddos_protection: dict = field(default_factory=lambda: {
        'layer_3_4': {
            'provider': 'CDN built-in',  # TCP/UDP フラッド
            'capacity': '100+ Tbps',
        },
        'layer_7': {
            'rate_limiting': True,        # HTTP フラッド
            'challenge_page': True,       # JS チャレンジ
            'geo_blocking': ['KP', 'IR'], # 特定国のブロック
        },
    })

# オリジン保護の実装例 (Nginx)
NGINX_ORIGIN_PROTECTION = """
# CDN の IP レンジのみ許可
# CloudFront IP ranges: https://d7uri8nf7uskq.cloudfront.net/tools/list-cloudfront-ips
geo $is_cdn {
    default         0;
    13.224.0.0/14   1;  # CloudFront
    52.84.0.0/15    1;  # CloudFront
    99.84.0.0/16    1;  # CloudFront
    # ... (全レンジを記載)
}

server {
    # CDN 以外からのアクセスを拒否
    if ($is_cdn = 0) {
        return 403;
    }

    # CDN からのシークレットヘッダーを検証
    if ($http_x_origin_verify != "shared-secret-value") {
        return 403;
    }
}
"""
```

### 7.4 CDN モニタリングと可観測性

```python
"""CDN パフォーマンスモニタリング"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class CDNMetrics:
    """CDN モニタリングで追跡すべきメトリクス"""

    # キャッシュ効率
    cache_hit_ratio: float       # 目標: 静的 95%+, 動的 60%+
    cache_miss_ratio: float
    cache_expired_ratio: float   # TTL 切れの割合

    # パフォーマンス
    ttfb_p50_ms: float           # 目標: < 50ms
    ttfb_p95_ms: float           # 目標: < 200ms
    ttfb_p99_ms: float           # 目標: < 500ms
    total_latency_ms: float

    # オリジン健全性
    origin_request_count: int    # 目標: 全リクエストの 5-20%
    origin_error_rate: float     # 目標: < 0.1%
    origin_response_time_ms: float

    # 帯域・コスト
    bandwidth_gb: float
    request_count: int
    cost_usd: float

    # エラー
    http_4xx_rate: float         # 目標: < 1%
    http_5xx_rate: float         # 目標: < 0.01%

class CDNMonitor:
    """CDN メトリクスの収集と分析"""

    # アラート閾値
    ALERT_THRESHOLDS = {
        'cache_hit_ratio_low': 0.80,       # 80% 未満で警告
        'origin_error_rate_high': 0.01,     # 1% 以上で警告
        'ttfb_p95_high_ms': 200,            # 200ms 以上で警告
        'http_5xx_rate_high': 0.001,        # 0.1% 以上で緊急
    }

    @staticmethod
    def analyze_cache_efficiency(metrics: CDNMetrics) -> dict:
        """キャッシュ効率の分析とアドバイス"""
        analysis = {'status': 'healthy', 'recommendations': []}

        if metrics.cache_hit_ratio < 0.80:
            analysis['status'] = 'degraded'
            analysis['recommendations'].append(
                "キャッシュヒット率が低い。"
                "以下を確認: (1) TTL が短すぎないか、"
                "(2) Vary ヘッダーが過剰でないか、"
                "(3) クエリパラメータがキャッシュキーに不要に含まれていないか"
            )

        if metrics.cache_expired_ratio > 0.30:
            analysis['recommendations'].append(
                "キャッシュ期限切れが多い。TTL の延長または "
                "stale-while-revalidate の導入を検討"
            )

        if metrics.origin_error_rate > 0.01:
            analysis['status'] = 'critical'
            analysis['recommendations'].append(
                "オリジンエラー率が高い。"
                "stale-if-error ヘッダーの設定でユーザー影響を軽減"
            )

        return analysis

    @staticmethod
    def calculate_cost_savings(
        total_requests: int,
        cache_hit_ratio: float,
        avg_origin_cost_per_request: float = 0.00001,
    ) -> dict:
        """CDN によるコスト削減効果を計算"""
        requests_served_by_cache = int(total_requests * cache_hit_ratio)
        requests_to_origin = total_requests - requests_served_by_cache
        origin_cost = requests_to_origin * avg_origin_cost_per_request
        saved_cost = requests_served_by_cache * avg_origin_cost_per_request

        return {
            'total_requests': total_requests,
            'cache_hits': requests_served_by_cache,
            'origin_requests': requests_to_origin,
            'origin_cost_usd': round(origin_cost, 2),
            'saved_cost_usd': round(saved_cost, 2),
            'saving_ratio': f"{cache_hit_ratio * 100:.1f}%",
        }
```

---

## 8. S3 + CloudFront の CDK 構成

```python
# AWS CDK v2: CloudFront + S3 + WAF の本番構成
from aws_cdk import (
    Stack, Duration, RemovalPolicy,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_certificatemanager as acm,
    aws_route53 as route53,
    aws_route53_targets as targets,
    aws_wafv2 as wafv2,
)
from constructs import Construct

class CDNStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # 1. S3 バケット (静的アセット)
        bucket = s3.Bucket(
            self, "StaticAssets",
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,  # バージョニング有効
        )

        # 2. ACM 証明書 (us-east-1 で作成が必須)
        certificate = acm.Certificate(
            self, "Certificate",
            domain_name="cdn.example.com",
            validation=acm.CertificateValidation.from_dns(),
        )

        # 3. キャッシュポリシー
        cache_policy = cloudfront.CachePolicy(
            self, "CachePolicy",
            cache_policy_name="OptimizedCaching",
            default_ttl=Duration.hours(24),
            max_ttl=Duration.days(365),
            min_ttl=Duration.seconds(0),
            header_behavior=cloudfront.CacheHeaderBehavior.allow_list(
                "Accept-Encoding", "Accept-Language"
            ),
            query_string_behavior=cloudfront.CacheQueryStringBehavior.allow_list(
                "page", "sort", "category"  # 必要なパラメータのみ
            ),
            cookie_behavior=cloudfront.CacheCookieBehavior.none(),
            enable_accept_encoding_gzip=True,
            enable_accept_encoding_brotli=True,
        )

        # 4. CloudFront ディストリビューション
        distribution = cloudfront.Distribution(
            self, "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(
                    bucket,
                    origin_shield_region="ap-northeast-1",
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cache_policy,
                compress=True,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
            ),
            domain_names=["cdn.example.com"],
            certificate=certificate,
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            price_class=cloudfront.PriceClass.PRICE_CLASS_200,
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",  # SPA フォールバック
                    ttl=Duration.seconds(10),
                ),
            ],
        )

        # 5. Route 53 レコード
        zone = route53.HostedZone.from_lookup(
            self, "Zone", domain_name="example.com"
        )
        route53.ARecord(
            self, "CDNRecord",
            zone=zone,
            record_name="cdn",
            target=route53.RecordTarget.from_alias(
                targets.CloudFrontTarget(distribution)
            ),
        )
```

---

## 9. アンチパターン

### アンチパターン 1: 全てのレスポンスに同一 TTL を適用

```python
# NG: 全 URL に一律の Cache-Control
class BadCacheMiddleware:
    """全てのレスポンスに同じ TTL を適用"""
    async def __call__(self, request, call_next):
        response = await call_next(request)
        # 全てに 1日キャッシュ
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

# 問題:
# - index.html を更新しても24時間反映されない
# - API レスポンスが古いまま返される
# - 個人情報が CDN にキャッシュされるリスク
# - ハッシュ付き JS は1日で無駄に再取得

# OK: アセット種別ごとに TTL を最適化
class GoodCacheMiddleware:
    """アセット種別に応じた Cache-Control 設定"""

    CACHE_RULES = [
        # (パスパターン, Cache-Control)
        (r'^/api/user/', 'private, no-store'),
        (r'^/api/', 'public, s-maxage=10, stale-while-revalidate=60, stale-if-error=300'),
        (r'\.[a-f0-9]{8,}\.(js|css)$', 'public, max-age=31536000, immutable'),
        (r'\.(jpg|png|webp|avif)$', 'public, max-age=2592000'),
        (r'\.html$', 'public, s-maxage=60, max-age=0, must-revalidate'),
    ]

    async def __call__(self, request, call_next):
        response = await call_next(request)
        path = request.url.path

        for pattern, cache_control in self.CACHE_RULES:
            if re.match(pattern, path):
                response.headers["Cache-Control"] = cache_control
                break
        else:
            # デフォルト: キャッシュなし (安全側に倒す)
            response.headers["Cache-Control"] = "private, no-cache"

        return response
```

### アンチパターン 2: キャッシュキーに不要な要素を含める

```python
# NG: Vary: User-Agent でキャッシュキーを生成
class BadVaryConfig:
    """Vary: User-Agent を使う"""
    def get_response_headers(self):
        return {
            "Cache-Control": "public, s-maxage=300",
            "Vary": "User-Agent",  # 数千種類の UA → 数千のキャッシュバリエーション
        }
    # 結果: キャッシュヒット率が 5% 以下に低下

# NG: クエリパラメータ全てをキャッシュキーに含める
# /products?utm_source=google&utm_medium=cpc → キャッシュ MISS
# /products?utm_source=twitter              → キャッシュ MISS
# /products                                 → キャッシュ MISS
# → 同じコンテンツなのに3つの別キャッシュエントリ

# OK: 必要なパラメータのみキャッシュキーに含める
class GoodVaryConfig:
    """デバイス判定は CDN の機能を利用"""
    def get_response_headers(self, request):
        return {
            "Cache-Control": "public, s-maxage=300",
            "Vary": "Accept-Encoding, Accept-Language",
            # デバイス判定: CDN のヘッダーを利用
            # CloudFront: CloudFront-Is-Mobile-Viewer
            # Cloudflare: CF-Device-Type
        }

    def configure_cache_policy(self):
        """CloudFront キャッシュポリシーで UTM パラメータを除外"""
        return {
            'QueryStringBehavior': 'whitelist',
            'QueryStrings': ['page', 'sort', 'category', 'q'],
            # utm_*, fbclid, gclid は自動的に除外
        }
```

### アンチパターン 3: CDN パージに依存したデプロイ

```python
# NG: デプロイのたびに全キャッシュをパージ
class BadDeployment:
    """デプロイ時に /* をパージ"""
    def deploy(self):
        # 1. ファイルをアップロード
        upload_files()

        # 2. 全キャッシュをパージ
        invalidate_all("/*")

        # 問題:
        # - パージ完了まで数秒〜数十秒のラグ
        # - パージ中に古いHTMLが新しいJSを参照 → エラー
        # - 全POPで一貫性が保たれない瞬間がある
        # - パージ費用が発生 (CloudFront: 1000パス/月超で課金)

# OK: キャッシュバスティングでパージを不要にする
class GoodDeployment:
    """ハッシュ付きファイル名でパージ不要のデプロイ"""
    def deploy(self):
        # 1. アセットはハッシュ付きファイル名 (パージ不要)
        #    main.js → main.a1b2c3d4.js (新ファイル = 新キャッシュ)
        upload_hashed_assets()

        # 2. HTML だけ短 TTL (s-maxage=60) で自然更新
        upload_html(cache_control="s-maxage=60, must-revalidate")

        # 3. 緊急時のみ HTML をパージ
        invalidate_paths(["/index.html"])

        # メリット:
        # - 一貫性の問題なし (HTMLが新JSを参照する時点でJSは既にCDNにある)
        # - パージコスト最小 (HTMLのみ)
        # - 古いバージョンへのロールバックも即座 (HTMLのパスを戻すだけ)
```

---

## 10. 練習問題

### 演習 1 (基礎): Cache-Control ヘッダーの設計

以下の各コンテンツに最適な `Cache-Control` ヘッダーを設計せよ。

```
1. ハッシュ付き JavaScript: main.a1b2c3.js
2. HTML ページ: /index.html
3. ユーザープロフィール API: /api/user/profile
4. 商品一覧 API: /api/products?page=1
5. ユーザーアバター画像: /images/avatar/user-123.jpg
6. フォントファイル: /fonts/noto-sans.woff2
7. リアルタイム株価 API: /api/stocks/AAPL
```

**期待される出力例:**

```
1. public, max-age=31536000, immutable
   理由: ハッシュが変われば新URL → 1年キャッシュ安全

2. public, s-maxage=60, max-age=0, must-revalidate
   理由: CDN 60秒キャッシュ、ブラウザは毎回確認

3. private, no-store
   理由: 個人情報、CDN にキャッシュ禁止

4. public, s-maxage=300, stale-while-revalidate=600
   理由: 共有コンテンツ、5分キャッシュ + バックグラウンド更新

5. public, max-age=86400, stale-while-revalidate=604800
   理由: アバター変更頻度低、1日キャッシュ + 1週間 SWR

6. public, max-age=31536000, immutable
   理由: フォントは不変、CORS ヘッダーも必要

7. public, s-maxage=5, stale-while-revalidate=10, stale-if-error=60
   理由: 鮮度重要だが、エラー時は古いデータでも可
```

### 演習 2 (応用): CDN キャッシュキー最適化

以下のアクセスログを分析し、キャッシュヒット率を改善するための施策を提案せよ。

```python
access_logs = [
    {"url": "/products?page=1&utm_source=google", "cache": "MISS"},
    {"url": "/products?page=1&utm_source=twitter", "cache": "MISS"},
    {"url": "/products?page=1", "cache": "MISS"},
    {"url": "/products?page=1&fbclid=abc", "cache": "MISS"},
    {"url": "/api/user/profile", "cache": "MISS", "vary": "User-Agent"},
    {"url": "/api/user/profile", "cache": "MISS", "vary": "User-Agent"},
    {"url": "/images/hero.jpg", "cache": "MISS"},
    {"url": "/images/hero.jpg?v=1", "cache": "MISS"},
    {"url": "/images/hero.jpg?v=2", "cache": "MISS"},
]

# 課題:
# 1. 現在のキャッシュヒット率を計算せよ
# 2. 各 MISS の原因を特定せよ
# 3. キャッシュヒット率を改善する施策を3つ以上提案せよ
# 4. 改善後の期待キャッシュヒット率を見積もれ
```

**期待される出力:**

```
1. 現在のキャッシュヒット率: 0% (9 MISS / 9 リクエスト)

2. 原因分析:
   - /products: UTM パラメータがキャッシュキーに含まれている (4 MISS → 1 で済む)
   - /api/user/profile: Vary: User-Agent が設定されている (キャッシュ不可)
   - /images/hero.jpg: クエリパラメータ ?v= が不要なバリエーションを生成

3. 改善施策:
   a. CDN キャッシュポリシーで utm_*, fbclid パラメータを除外
   b. Vary: User-Agent を削除し、CDN のデバイス検出機能を利用
   c. 画像のバージョニングをファイル名ハッシュ方式に変更

4. 改善後の期待キャッシュヒット率:
   - /products: 4リクエスト → 1 MISS + 3 HIT (75%)
   - /api/user/profile: Vary 修正で 2リクエスト → 1 MISS + 1 HIT (50%)
   - /images/hero.jpg: 3リクエスト → 1 MISS + 2 HIT (67%)
   - 全体: 9リクエスト → 3 MISS + 6 HIT (67%)
```

### 演習 3 (上級): マルチオリジン CDN アーキテクチャ設計

以下の要件を満たす CDN 構成を設計せよ。

```
要件:
- SPA フロントエンド (React) を S3 から配信
- REST API を ALB 経由のバックエンドから配信
- 画像を S3 から配信 (エッジでリサイズ・フォーマット変換)
- WebSocket 接続をサポート
- 全世界のユーザーにサービス提供 (主要地域: 日本、北米、欧州)
- 99.99% 可用性目標

設計項目:
1. オリジン構成 (複数オリジンの設計)
2. キャッシュポリシー (パスパターンごと)
3. エッジ処理 (CloudFront Functions / Lambda@Edge の使い分け)
4. セキュリティ (WAF ルール、オリジン保護)
5. モニタリング (主要メトリクス、アラート閾値)
6. 障害時のフェイルオーバー戦略
```

**期待される出力:** 各項目について具体的な設定とその理由を含む設計書 (500 文字以上)

---

## 11. FAQ

### Q1. 動的コンテンツにも CDN は有効か？

**A.** 有効である。CDN は動的コンテンツにも4つの恩恵がある。(1) **TCP/TLS ハンドシェイクの高速化** --- エッジとオリジン間のコネクション再利用 (Keep-Alive) により、ユーザーがオリジンと直接通信するよりも速い。(2) **短 TTL + `stale-while-revalidate`** --- `s-maxage=5, stale-while-revalidate=30` のように設定すれば、5秒間はキャッシュから返し、裏でオリジンを更新。リクエストの 90% 以上がキャッシュヒットになり得る。(3) **エッジコンピューティング** --- CloudFront Functions / Cloudflare Workers でオリジンへのリクエスト自体を削減。認証トークンの検証やレートリミットをエッジで実行。(4) **接続の最適化** --- CDN のバックボーンネットワークは公衆インターネットより最適化されており、オリジンへの通信自体が高速。

### Q2. キャッシュの無効化（パージ）はどう管理すべきか？

**A.** パージに頼る設計は避け、「キャッシュバスティング」を基本とする。静的アセットにはコンテンツハッシュをファイル名に含め（`app.abc123.js`）、HTML から参照するパスを更新する。これにより新しいファイル名 = 新しいキャッシュエントリとなり、パージ不要。HTML 自体は短 TTL (s-maxage=60) で自然に更新する。API レスポンスのキャッシュには Fastly の Surrogate-Key のようなタグベースパージが効果的。緊急時のみワイルドカードパージ（`/api/*`）を使うが、全キャッシュパージ (`/*`) は最終手段とする。

### Q3. CDN とオリジンの通信を最適化するには？

**A.** (1) **Origin Shield** を有効化し、複数 POP からオリジンへの重複リクエストを集約する。Origin Shield をオリジンに最も近いリージョンに配置することで、オリジンへのリクエスト数を 60-90% 削減できる。(2) **Keep-Alive / HTTP/2** でコネクション数を最適化する。CDN とオリジン間の永続的接続により、TLS ハンドシェイクのコストを削減。(3) **Gzip / Brotli 圧縮** をオリジンまたはエッジで有効化する。テキストベースのコンテンツで 60-80% の帯域削減。(4) **ETag / Last-Modified** による条件付きリクエスト（304 Not Modified）でデータ転送を削減する。CDN キャッシュ期限切れ後もオリジンのデータが変わっていなければ 304 が返り、帯域を節約。

### Q4. マルチ CDN 構成はどういう場合に必要か？

**A.** 以下のケースでマルチ CDN を検討する: (1) **可用性要件が 99.99% 以上** --- 単一 CDN の障害リスクを軽減。DNS レベル (Route 53 / NS1) でフェイルオーバー。(2) **地域ごとの最適化** --- アジアでは Cloudflare、北米では CloudFront のように地域ごとに最適な CDN を選択。(3) **コスト最適化** --- トラフィック量が月間 100TB 以上の場合、CDN 間で価格交渉の材料になる。(4) **ベンダーロックイン回避** --- 特定 CDN に依存しない設計。ただし、マルチ CDN は運用複雑性が高く、キャッシュ効率も低下するため、月間 100TB 未満であれば単一 CDN で十分なことが多い。

### Q5. HTTP/3 (QUIC) は CDN でどのような効果があるか？

**A.** HTTP/3 は UDP ベースの QUIC プロトコルを使用し、以下の効果がある: (1) **0-RTT 接続** --- 再訪問時の TLS ハンドシェイクを省略。初回接続でも 1-RTT で完了（HTTP/2 は 2-3 RTT）。(2) **Head-of-Line Blocking の解消** --- HTTP/2 では 1 つのパケットロスが全ストリームをブロックするが、HTTP/3 では影響を受けたストリームのみ。モバイル回線で特に効果的。(3) **コネクションマイグレーション** --- Wi-Fi ↔ セルラー切り替え時にコネクションを維持。現在の主要 CDN (CloudFront, Cloudflare, Fastly) は全て HTTP/3 に対応しており、クライアント側のブラウザも Chrome/Firefox/Safari で対応済み。有効化は CDN 設定で `HttpVersion: http2and3` を指定するだけ。

### Q6. CDN の費用を最適化するには？

**A.** (1) **キャッシュヒット率の最大化** --- ヒット率 1% の改善がオリジンコスト数%削減に直結。キャッシュキーの正規化、適切な TTL 設計、Origin Shield の有効化が鍵。(2) **圧縮の活用** --- Brotli 圧縮で帯域を 60-80% 削減。多くの CDN は自動圧縮機能を提供。(3) **PriceClass の最適化** --- CloudFront では PriceClass_100 (北米+欧州のみ) や PriceClass_200 (+ アジア) で不要な POP を除外。(4) **リザーブドキャパシティ** --- CloudFront Savings Plan (最大 30% 割引) や Cloudflare の年間契約。(5) **画像最適化** --- WebP/AVIF 変換で画像サイズを 30-50% 削減。(6) **不要なリクエストの削減** --- ブラウザキャッシュの max-age を適切に設定し、CDN へのリクエスト自体を減らす。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CDN の役割 | エッジキャッシュによるレイテンシ削減、オリジン負荷軽減、DDoS 防御 |
| キャッシュ戦略 | アセット種別ごとに TTL を最適化。immutable / stale-while-revalidate 活用 |
| キャッシュバスティング | ファイル名にハッシュを含め、パージ依存を回避 |
| キャッシュキー設計 | UTM/トラッキングパラメータを除外、Vary ヘッダーは最小限に |
| CloudFront | AWS エコシステムとのネイティブ統合。Lambda@Edge / Functions でエッジ処理 |
| Cloudflare | 無料帯域、即時パージ、Workers による軽量エッジコンピューティング |
| Fastly | Surrogate-Key によるタグベースパージ、Compute@Edge (Wasm) |
| Origin Shield | 中間キャッシュ層によるオリジンへのリクエスト集約 (60-90% 削減) |
| セキュリティ | HTTPS 強制、WAF、DDoS 防御、オリジン保護を CDN レイヤーで実装 |
| モニタリング | キャッシュヒット率、TTFB p95、オリジンエラー率を継続監視 |

---

## 次に読むべきガイド

- [ロードバランサー](./00-load-balancer.md) --- CDN の背後にあるトラフィック分散
- [キャッシング](./01-caching.md) --- アプリケーション層のキャッシュ戦略との連携
- [DBスケーリング](./04-database-scaling.md) --- データ層のスケーリング戦略
- [メッセージキュー](./02-message-queue.md) --- 非同期メッセージング基盤
- [信頼性](../00-fundamentals/02-reliability.md) --- 可用性 SLA と障害対策の全体設計

---

## 参考文献

1. **Web Performance in Action** --- Jeremy Wagner (Manning, 2017) --- CDN とキャッシュ戦略の実践ガイド
2. **Amazon CloudFront Developer Guide** --- AWS Documentation --- https://docs.aws.amazon.com/cloudfront/
3. **Cloudflare Learning Center** --- https://www.cloudflare.com/learning/ --- CDN の基礎から高度な活用まで
4. **HTTP Caching (MDN Web Docs)** --- https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching
5. **High Performance Browser Networking** --- Ilya Grigorik (O'Reilly, 2013) --- https://hpbn.co/ --- HTTP/2, QUIC, CDN の深い解説
