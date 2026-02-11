# CDN (Content Delivery Network)

> 世界中に分散配置されたエッジサーバーを活用し、ユーザーに最も近い拠点からコンテンツを配信することでレイテンシを最小化し、オリジンサーバーの負荷を軽減する技術を、CloudFront・Cloudflare の比較を通じて解説する

## この章で学ぶこと

1. **CDN の基本原理** — エッジキャッシュ、オリジンシールド、POP (Point of Presence) の仕組み
2. **主要 CDN サービスの比較** — Amazon CloudFront、Cloudflare、Fastly の特性と選定基準
3. **キャッシュ戦略と無効化** — Cache-Control ヘッダー、キャッシュキー設計、パージ戦略の実践

---

## 1. CDN の基本アーキテクチャ

### 1.1 リクエストフロー

```
ユーザー (東京)                    CDN Edge (東京 POP)           オリジンサーバー (us-east-1)
     |                                  |                              |
     |--- GET /img/hero.jpg ---------->|                              |
     |                                  |-- キャッシュ確認              |
     |                                  |   HIT? --> 即座にレスポンス   |
     |                                  |   MISS? ----GET /img/hero.jpg-->|
     |                                  |<--------- 200 OK + データ ---|
     |                                  |-- キャッシュ保存              |
     |<--------- 200 OK + データ -------|                              |
     |                                  |                              |
     |--- GET /img/hero.jpg ---------->|                              |
     |<--------- 200 OK (Cache HIT) ---|  (オリジンへ問い合わせなし)    |
```

### 1.2 グローバル POP 配置

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
       \                    |                    /
        +------- Origin Shield (中間キャッシュ) --+
                          |
                   +-------------+
                   |   Origin    |
                   |   Server    |
                   +-------------+
```

### 1.3 キャッシュ階層

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
| `s-maxage=N` | CDN 用の max-age（ブラウザは max-age を使う） | CDN と ブラウザで TTL を分離 |
| `stale-while-revalidate=N` | 期限切れ後 N 秒間は古いキャッシュを返しつつ裏で更新 | 高可用性 API |
| `immutable` | コンテンツは変更されない（再検証不要） | ハッシュ付きアセット |

### 2.2 アセット種別ごとの設定例

```nginx
# Nginx での Cache-Control 設定例

# ハッシュ付き静的アセット（CSS/JS/画像）: 1年キャッシュ
location ~* \.(css|js)$ {
    # ファイル名にハッシュ: main.a1b2c3.js
    add_header Cache-Control "public, max-age=31536000, immutable";
}

# 画像: 30日キャッシュ
location ~* \.(jpg|jpeg|png|gif|webp|avif|svg)$ {
    add_header Cache-Control "public, max-age=2592000";
}

# HTML: CDN 60秒、ブラウザはキャッシュなし
location ~* \.html$ {
    add_header Cache-Control "public, s-maxage=60, max-age=0, must-revalidate";
}

# API レスポンス: CDN 10秒 + stale-while-revalidate
location /api/ {
    add_header Cache-Control "public, s-maxage=10, stale-while-revalidate=60";
}

# 個人情報: キャッシュ禁止
location /api/user/profile {
    add_header Cache-Control "private, no-store";
}
```

```python
# CloudFront + S3 でのキャッシュヘッダー設定 (boto3)
import boto3

s3 = boto3.client('s3')

# ハッシュ付き JS ファイル: 長期キャッシュ
s3.put_object(
    Bucket='my-static-assets',
    Key='js/app.abc123.js',
    Body=open('dist/app.abc123.js', 'rb'),
    ContentType='application/javascript',
    CacheControl='public, max-age=31536000, immutable',
)

# index.html: 短TTL + 再検証
s3.put_object(
    Bucket='my-static-assets',
    Key='index.html',
    Body=open('dist/index.html', 'rb'),
    ContentType='text/html',
    CacheControl='public, s-maxage=60, max-age=0, must-revalidate',
)
```

---

## 3. Amazon CloudFront の設定

```python
# CloudFront ディストリビューション作成 (boto3)
import boto3

cf = boto3.client('cloudfront')

distribution_config = {
    'CallerReference': 'my-app-2026',
    'Origins': {
        'Quantity': 1,
        'Items': [{
            'Id': 'S3-my-static-assets',
            'DomainName': 'my-static-assets.s3.amazonaws.com',
            'S3OriginConfig': {
                'OriginAccessIdentity': 'origin-access-identity/cloudfront/XXXXXXX'
            },
        }]
    },
    'DefaultCacheBehavior': {
        'TargetOriginId': 'S3-my-static-assets',
        'ViewerProtocolPolicy': 'redirect-to-https',
        'CachePolicyId': '658327ea-f89d-4fab-a63d-7e88639e58f6',  # CachingOptimized
        'Compress': True,              # Brotli/Gzip 自動圧縮
        'AllowedMethods': {'Quantity': 2, 'Items': ['GET', 'HEAD']},
    },
    'Enabled': True,
    'PriceClass': 'PriceClass_200',    # 北米・欧州・アジア
    'HttpVersion': 'http2and3',        # HTTP/2 + HTTP/3 有効化
    'Comment': 'Production static assets',
}

response = cf.create_distribution(DistributionConfig=distribution_config)
print(f"Distribution ID: {response['Distribution']['Id']}")
print(f"Domain: {response['Distribution']['DomainName']}")
```

```bash
# CloudFront キャッシュ無効化（パージ）
aws cloudfront create-invalidation \
  --distribution-id E1234567890 \
  --paths "/index.html" "/api/*"
```

---

## 4. Cloudflare の設定

```javascript
// Cloudflare Workers によるエッジコンピューティング
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cache = caches.default;

    // 1. キャッシュチェック
    let response = await cache.match(request);
    if (response) {
      return response; // Cache HIT
    }

    // 2. オリジンからフェッチ
    response = await fetch(request);

    // 3. レスポンスをクローンしてキャッシュ保存
    const cacheResponse = new Response(response.body, response);
    cacheResponse.headers.set('Cache-Control', 'public, s-maxage=300');
    cacheResponse.headers.set('X-Cache-Status', 'MISS');

    // 非同期でキャッシュ書き込み（レスポンスを遅延させない）
    event.waitUntil(cache.put(request, cacheResponse.clone()));

    return cacheResponse;
  }
};
```

---

## 5. 主要 CDN サービス比較

| 特性 | CloudFront | Cloudflare | Fastly |
|-----|-----------|------------|--------|
| **POP 数** | 450+ | 300+ | 90+ |
| **エッジコンピューティング** | Lambda@Edge / CloudFront Functions | Workers (V8 isolate) | Compute@Edge (Wasm) |
| **無料枠** | 1TB/月 (12ヶ月) | 無制限帯域（Free プラン） | なし |
| **キャッシュパージ速度** | 数秒〜数十秒 | < 30ms (Instant Purge) | < 150ms |
| **DDoS 防御** | AWS Shield | 標準搭載（全プラン） | Shield |
| **WAF** | AWS WAF (別料金) | 標準搭載 (Pro 以上) | Next-Gen WAF |
| **HTTP/3 (QUIC)** | 対応 | 対応 | 対応 |
| **最適用途** | AWS エコシステム統合 | 汎用・セキュリティ重視 | 高速パージ・API キャッシュ |

| 判断基準 | CloudFront | Cloudflare | Fastly |
|---------|-----------|------------|--------|
| AWS S3/ALB との統合 | 最適 | 可能 | 可能 |
| 即時キャッシュパージ | -- | 最適 | 最適 |
| 無料で始めたい | -- | 最適 | -- |
| エッジ JS 実行 | 可能 | 最適 | -- |
| 動的コンテンツ加速 | 可能 | 可能 | 最適 |

---

## 6. アンチパターン

### アンチパターン 1: 全てのレスポンスに同一 TTL を適用

```
BAD:
  Cache-Control: public, max-age=86400  ← 全 URL に一律1日

  問題:
  - index.html を更新しても24時間反映されない
  - API レスポンスが古いまま返される
  - ハッシュ付き JS は1日で無駄に再取得

GOOD: アセット種別ごとに TTL を最適化
  index.html    → s-maxage=60, must-revalidate  (1分)
  app.abc123.js → max-age=31536000, immutable    (1年)
  /api/products → s-maxage=10, stale-while-revalidate=60
  /api/user/*   → private, no-store
```

### アンチパターン 2: キャッシュキーに不要な要素を含める

```
BAD: クエリパラメータ全てをキャッシュキーに含める
  /products?utm_source=google&utm_medium=cpc → キャッシュ MISS
  /products?utm_source=twitter              → キャッシュ MISS
  /products                                 → キャッシュ MISS
  → 同じコンテンツなのに3つの別キャッシュエントリ

GOOD: キャッシュに影響するパラメータのみをキーに含める
  CloudFront Cache Policy で utm_* パラメータを除外
  → /products は全バリエーションで同一キャッシュ HIT
```

---

## 7. FAQ

### Q1. 動的コンテンツにも CDN は有効か？

**A.** 有効である。CDN は動的コンテンツにも3つの恩恵がある。(1) TCP/TLS ハンドシェイクの高速化（エッジとオリジン間のコネクション再利用）、(2) 短 TTL + `stale-while-revalidate` による部分キャッシュ、(3) エッジコンピューティング（CloudFront Functions / Cloudflare Workers）でオリジンへのリクエスト自体を削減。API レスポンスでも `s-maxage=5, stale-while-revalidate=30` のような設定で劇的に改善する。

### Q2. キャッシュの無効化（パージ）はどう管理すべきか？

**A.** パージに頼る設計は避け、「キャッシュバスティング」を基本とする。静的アセットにはコンテンツハッシュをファイル名に含め（`app.abc123.js`）、HTML から参照するパスを更新する。これにより新しいファイル名 = 新しいキャッシュエントリとなり、パージ不要。HTML 自体は短 TTL で自然に更新する。緊急時のみワイルドカードパージ（`/api/*`）を使う。

### Q3. CDN とオリジンの通信を最適化するには？

**A.** (1) **Origin Shield** を有効化し、複数 POP からオリジンへの重複リクエストを集約する。(2) **Keep-Alive / HTTP/2** でコネクション数を最適化する。(3) **Gzip / Brotli 圧縮** をオリジンまたはエッジで有効化する。(4) **ETag / Last-Modified** による条件付きリクエスト（304 Not Modified）でデータ転送を削減する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CDN の役割 | エッジキャッシュによるレイテンシ削減、オリジン負荷軽減、DDoS 防御 |
| キャッシュ戦略 | アセット種別ごとに TTL を最適化。immutable / stale-while-revalidate 活用 |
| キャッシュバスティング | ファイル名にハッシュを含め、パージ依存を回避 |
| CloudFront | AWS エコシステムとのネイティブ統合。Lambda@Edge で高度な処理 |
| Cloudflare | 無料帯域、即時パージ、Workers による軽量エッジコンピューティング |
| Origin Shield | 中間キャッシュ層によるオリジンへのリクエスト集約 |
| セキュリティ | HTTPS 強制、WAF、DDoS 防御を CDN レイヤーで実装 |

---

## 次に読むべきガイド

- [DBスケーリング](./04-database-scaling.md) — データ層のスケーリング戦略
- [メッセージキュー](./02-message-queue.md) — 非同期メッセージング基盤
- [レートリミッター設計](../03-case-studies/03-rate-limiter.md) — CDN エッジでのレート制限

---

## 参考文献

1. **Web Performance in Action** — Jeremy Wagner (Manning, 2017) — CDN とキャッシュ戦略の実践ガイド
2. **Amazon CloudFront Developer Guide** — AWS Documentation — https://docs.aws.amazon.com/cloudfront/
3. **Cloudflare Learning Center** — https://www.cloudflare.com/learning/ — CDN の基礎から高度な活用まで
4. **HTTP Caching (MDN Web Docs)** — https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching
