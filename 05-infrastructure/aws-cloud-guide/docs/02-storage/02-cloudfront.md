# CloudFront

> AWS のグローバル CDN — キャッシュポリシー、オリジン設定、Lambda@Edge、OAC でコンテンツ配信を最適化する

## この章で学ぶこと

1. CloudFront のディストリビューションとオリジンを設計し、効率的なコンテンツ配信を構築できる
2. キャッシュポリシーとキャッシュ無効化を適切に設定し、パフォーマンスと鮮度を両立できる
3. Lambda@Edge / CloudFront Functions と OAC を使って、エッジでの処理とセキュアなオリジンアクセスを実現できる
4. CloudFormation / CDK を使った CloudFront ディストリビューションの Infrastructure as Code を実装できる
5. WAF 連携、署名付き URL、地理的制限などのセキュリティ機能を活用できる

---

## 1. CloudFront とは

### 1.1 CloudFront のアーキテクチャ

```
CloudFront グローバルネットワーク

  ユーザー (東京)                ユーザー (ニューヨーク)
       |                              |
       v                              v
  +----------+                   +----------+
  | Edge     |                   | Edge     |
  | Location |                   | Location |
  | (東京)   |                   | (NYC)    |
  +----+-----+                   +----+-----+
       |                              |
       | キャッシュミス時のみ            |
       v                              v
  +------------------+          +------------------+
  | Regional Edge    |          | Regional Edge    |
  | Cache (大阪)     |          | Cache (バージニア)|
  +--------+---------+          +--------+---------+
           |                             |
           +-------------+---------------+
                         |
                         v
                  +-------------+
                  |   Origin    |
                  | (S3/ALB等)  |
                  +-------------+

  Edge Location: 400+ 拠点（ユーザーに最も近い）
  Regional Edge Cache: 13 拠点（中間キャッシュ層）
```

### 1.2 CloudFront の主要コンポーネント

```
+------------------------------------------------------+
| Distribution (ディストリビューション)                    |
|                                                       |
|  +------------------------------------------------+  |
|  | Origins (オリジン)                                |  |
|  | - S3 バケット                                    |  |
|  | - ALB / EC2                                     |  |
|  | - カスタムオリジン (任意の HTTP サーバー)           |  |
|  | - MediaStore / MediaPackage                     |  |
|  +------------------------------------------------+  |
|                                                       |
|  +------------------------------------------------+  |
|  | Behaviors (ビヘイビア)                            |  |
|  | - パスパターン: /api/*, /static/*, デフォルト(*)  |  |
|  | - キャッシュポリシー                              |  |
|  | - オリジンリクエストポリシー                       |  |
|  | - レスポンスヘッダーポリシー                       |  |
|  +------------------------------------------------+  |
|                                                       |
|  +------------------------------------------------+  |
|  | Settings                                        |  |
|  | - 代替ドメイン (CNAME)                           |  |
|  | - SSL 証明書 (ACM)                              |  |
|  | - 価格クラス                                    |  |
|  | - WAF 連携                                     |  |
|  +------------------------------------------------+  |
+------------------------------------------------------+
```

### 1.3 CloudFront の料金体系

| 課金項目 | 内容 | 東京リージョン参考価格 |
|---------|------|---------------------|
| データ転送量 | Edge → インターネット | $0.114/GB (最初の 10TB) |
| HTTP リクエスト | GET/HEAD | $0.0090/万リクエスト |
| HTTPS リクエスト | GET/HEAD | $0.0120/万リクエスト |
| Invalidation | キャッシュ無効化 | 月 1,000 パスまで無料 |
| Lambda@Edge | 実行回数 + 実行時間 | $0.60/100万リクエスト |
| CloudFront Functions | 実行回数 | $0.10/100万リクエスト |

### 1.4 価格クラスの選択

```bash
# 価格クラスの比較
# PriceClass_All:     全 Edge Location を使用（最高パフォーマンス）
# PriceClass_200:     北米、欧州、アジア、中東、アフリカ
# PriceClass_100:     北米、欧州のみ（最安）

# 日本向けサービスなら PriceClass_200 が推奨
# PriceClass_100 だとアジアの Edge Location が使われない
```

---

## 2. オリジン設定

### 2.1 オリジンタイプ比較

| オリジンタイプ | 用途 | 認証方式 | プロトコル |
|--------------|------|---------|----------|
| S3 バケット | 静的ファイル | OAC (推奨) | HTTPS |
| ALB | 動的コンテンツ | カスタムヘッダー | HTTP/HTTPS |
| EC2 | 直接接続 | セキュリティグループ | HTTP/HTTPS |
| API Gateway | API | IAM / Cognito | HTTPS |
| MediaStore | 動画配信 | OAC | HTTPS |
| カスタムオリジン | 外部サーバー | 共有シークレット | HTTPS |

### 2.2 コード例: CloudFront ディストリビューションの作成

```bash
# S3 + ALB のマルチオリジン構成
aws cloudfront create-distribution --distribution-config '{
  "CallerReference": "my-dist-2024",
  "Comment": "Production distribution",
  "Enabled": true,
  "DefaultRootObject": "index.html",
  "Origins": {
    "Quantity": 2,
    "Items": [
      {
        "Id": "S3-static",
        "DomainName": "my-bucket.s3.ap-northeast-1.amazonaws.com",
        "OriginAccessControlId": "EXXXXXXXX",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      },
      {
        "Id": "ALB-api",
        "DomainName": "my-alb-123456.ap-northeast-1.elb.amazonaws.com",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "https-only",
          "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
          "OriginReadTimeout": 30,
          "OriginKeepaliveTimeout": 5
        },
        "CustomHeaders": {
          "Quantity": 1,
          "Items": [{
            "HeaderName": "X-Origin-Verify",
            "HeaderValue": "my-secret-header-value"
          }]
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-static",
    "ViewerProtocolPolicy": "redirect-to-https",
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
    "Compress": true,
    "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}
  },
  "CacheBehaviors": {
    "Quantity": 1,
    "Items": [{
      "PathPattern": "/api/*",
      "TargetOriginId": "ALB-api",
      "ViewerProtocolPolicy": "https-only",
      "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
      "OriginRequestPolicyId": "216adef6-5c7f-47e4-b989-5492eafa07d3",
      "AllowedMethods": {"Quantity": 7, "Items": ["GET","HEAD","OPTIONS","PUT","POST","PATCH","DELETE"]},
      "Compress": true
    }]
  },
  "ViewerCertificate": {
    "ACMCertificateArn": "arn:aws:acm:us-east-1:123456789012:certificate/xxx",
    "SSLSupportMethod": "sni-only",
    "MinimumProtocolVersion": "TLSv1.2_2021"
  },
  "Aliases": {"Quantity": 1, "Items": ["www.example.com"]}
}'
```

### 2.3 オリジンフェイルオーバー

```
オリジンフェイルオーバー構成

  CloudFront
      |
  Origin Group
  +-----------------------+
  | Primary:   S3-主系    |  ← 通常はこちらに転送
  | Secondary: S3-DR系    |  ← Primary が 5xx/4xx を返した場合に自動切替
  +-----------------------+

  フェイルオーバー条件:
  - HTTP 500, 502, 503, 504
  - HTTP 403, 404（オプション）
  - 接続タイムアウト
```

```bash
# オリジングループの作成（フェイルオーバー構成）
aws cloudfront create-distribution --distribution-config '{
  "CallerReference": "failover-dist-2024",
  "Comment": "Distribution with origin failover",
  "Enabled": true,
  "Origins": {
    "Quantity": 2,
    "Items": [
      {
        "Id": "S3-primary",
        "DomainName": "primary-bucket.s3.ap-northeast-1.amazonaws.com",
        "OriginAccessControlId": "EXXXXXXXX",
        "S3OriginConfig": {"OriginAccessIdentity": ""}
      },
      {
        "Id": "S3-failover",
        "DomainName": "failover-bucket.s3.us-west-2.amazonaws.com",
        "OriginAccessControlId": "EYYYYYYYY",
        "S3OriginConfig": {"OriginAccessIdentity": ""}
      }
    ]
  },
  "OriginGroups": {
    "Quantity": 1,
    "Items": [{
      "Id": "my-origin-group",
      "FailoverCriteria": {
        "StatusCodes": {
          "Quantity": 4,
          "Items": [500, 502, 503, 504]
        }
      },
      "Members": {
        "Quantity": 2,
        "Items": [
          {"OriginId": "S3-primary"},
          {"OriginId": "S3-failover"}
        ]
      }
    }]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "my-origin-group",
    "ViewerProtocolPolicy": "redirect-to-https",
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
    "Compress": true,
    "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}
  }
}'
```

### 2.4 ALB オリジンのセキュリティ強化

```bash
# ALB が CloudFront 経由のリクエストのみ受け付けるようにする

# 方法1: カスタムヘッダーで検証
# CloudFront 側でカスタムヘッダーを付与し、ALB のルールで検証

# ALB リスナールールの設定
aws elbv2 create-rule \
  --listener-arn arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:listener/app/my-alb/xxx/yyy \
  --conditions '[{
    "Field": "http-header",
    "HttpHeaderConfig": {
      "HttpHeaderName": "X-Origin-Verify",
      "Values": ["my-secret-header-value"]
    }
  }]' \
  --actions '[{"Type": "forward", "TargetGroupArn": "arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:targetgroup/my-targets/xxx"}]' \
  --priority 10

# カスタムヘッダーなしのリクエストを拒否するデフォルトルール
aws elbv2 modify-rule \
  --rule-arn arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:listener-rule/app/my-alb/xxx/yyy/zzz \
  --actions '[{"Type": "fixed-response", "FixedResponseConfig": {"StatusCode": "403", "ContentType": "text/plain", "MessageBody": "Direct access not allowed"}}]'

# 方法2: AWS マネージドプレフィックスリストで CloudFront の IP 範囲を許可
# Security Group に CloudFront の IP 範囲を追加
aws ec2 authorize-security-group-ingress \
  --group-id sg-0123456789abcdef0 \
  --ip-permissions '[{
    "IpProtocol": "tcp",
    "FromPort": 443,
    "ToPort": 443,
    "PrefixListIds": [{"PrefixListId": "pl-3b927c52"}]
  }]'
# pl-3b927c52 は CloudFront のマネージドプレフィックスリスト
```

---

## 3. キャッシュポリシー

### 3.1 キャッシュの仕組み

```
キャッシュヒット/ミスのフロー

  リクエスト → Edge Location
                  |
          キャッシュキーで検索
                  |
         +--------+--------+
         |                 |
      ヒット             ミス
         |                 |
    キャッシュから     Regional Edge Cache
    即座にレスポンス        |
                    +------+------+
                    |             |
                 ヒット         ミス
                    |             |
               キャッシュから   オリジンに
               レスポンス     リクエスト転送

  キャッシュキー = URL + ヘッダー + クエリ文字列 + Cookie
  (キャッシュポリシーで構成要素を制御)
```

### 3.2 マネージドキャッシュポリシー

| ポリシー名 | ポリシー ID | TTL | クエリ文字列 | ヘッダー | 用途 |
|-----------|-----------|-----|------------|---------|------|
| CachingOptimized | 658327ea-... | 86400s | なし | なし | 静的コンテンツ |
| CachingOptimizedForUncompressedObjects | b2884449-... | 86400s | なし | なし | 非圧縮 |
| CachingDisabled | 4135ea2d-... | 0s | 全て | 全て | API / 動的ページ |
| Amplify | 2e54312d-... | 2s | 全て | Authorization | Amplify アプリ |
| Elemental-MediaPackage | 08627262-... | 86400s | 一部 | なし | 動画配信 |

### 3.3 マネージドオリジンリクエストポリシー

| ポリシー名 | 用途 | 転送内容 |
|-----------|------|---------|
| AllViewer | 全てのビューワーヘッダーを転送 | ヘッダー全て、クエリ文字列全て、Cookie 全て |
| AllViewerExceptHostHeader | Host 以外を転送 | ALB オリジン向け |
| CORS-S3Origin | CORS ヘッダーを転送 | Origin, Access-Control-Request-* |
| UserAgentRefererHeaders | UA + Referer を転送 | 分析用途 |
| AllViewerAndCloudFrontHeaders-2022-06 | CF ヘッダー含む全て | 地理情報等の CF 独自ヘッダーも転送 |

### 3.4 コード例: カスタムキャッシュポリシー

```bash
# カスタムキャッシュポリシーを作成
aws cloudfront create-cache-policy --cache-policy-config '{
  "Name": "CustomStaticAssets",
  "Comment": "Static assets with long TTL",
  "DefaultTTL": 86400,
  "MaxTTL": 31536000,
  "MinTTL": 0,
  "ParametersInCacheKeyAndForwardedToOrigin": {
    "EnableAcceptEncodingGzip": true,
    "EnableAcceptEncodingBrotli": true,
    "HeadersConfig": {
      "HeaderBehavior": "none"
    },
    "CookiesConfig": {
      "CookieBehavior": "none"
    },
    "QueryStringsConfig": {
      "QueryStringBehavior": "whitelist",
      "QueryStrings": {
        "Quantity": 1,
        "Items": ["v"]
      }
    }
  }
}'

# API 向けカスタムキャッシュポリシー（短 TTL + Authorization ヘッダー）
aws cloudfront create-cache-policy --cache-policy-config '{
  "Name": "ApiShortTTL",
  "Comment": "API with short TTL and Authorization cache key",
  "DefaultTTL": 5,
  "MaxTTL": 60,
  "MinTTL": 0,
  "ParametersInCacheKeyAndForwardedToOrigin": {
    "EnableAcceptEncodingGzip": true,
    "EnableAcceptEncodingBrotli": true,
    "HeadersConfig": {
      "HeaderBehavior": "whitelist",
      "Headers": {
        "Quantity": 1,
        "Items": ["Authorization"]
      }
    },
    "CookiesConfig": {
      "CookieBehavior": "none"
    },
    "QueryStringsConfig": {
      "QueryStringBehavior": "all"
    }
  }
}'
```

### 3.5 コード例: キャッシュ無効化 (Invalidation)

```bash
# 特定パスのキャッシュを無効化
aws cloudfront create-invalidation \
  --distribution-id EXXXXXXXXXX \
  --paths '/index.html' '/css/*' '/js/*'

# 全キャッシュを無効化（コスト注意: 月 1,000 パスまで無料）
aws cloudfront create-invalidation \
  --distribution-id EXXXXXXXXXX \
  --paths '/*'

# 無効化の状態確認
aws cloudfront get-invalidation \
  --distribution-id EXXXXXXXXXX \
  --id IXXXXXXXXX

# デプロイ後のキャッシュ無効化スクリプト
#!/bin/bash
DIST_ID="EXXXXXXXXXX"
INVALIDATION_ID=$(aws cloudfront create-invalidation \
  --distribution-id $DIST_ID \
  --paths '/*' \
  --query 'Invalidation.Id' --output text)

echo "Invalidation ID: $INVALIDATION_ID"

# 完了を待機
aws cloudfront wait invalidation-completed \
  --distribution-id $DIST_ID \
  --id $INVALIDATION_ID

echo "Invalidation completed!"
```

### 3.6 キャッシュヒット率の最適化

```bash
# キャッシュヒット率の確認
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name CacheHitRate \
  --dimensions Name=DistributionId,Value=EXXXXXXXXXX Name=Region,Value=Global \
  --start-time "$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 3600 \
  --statistics Average

# キャッシュヒット率が低い場合のチェックリスト:
# 1. クエリ文字列: 不要なクエリ文字列をキャッシュキーから除外
# 2. Cookie: 不要な Cookie をキャッシュキーから除外
# 3. ヘッダー: Accept-Encoding 以外の不要なヘッダーを除外
# 4. TTL: MinTTL が 0 の場合、オリジンの Cache-Control ヘッダーを確認
# 5. バージョニング: ファイル名にハッシュを含めて長い TTL を設定
```

---

## 4. Lambda@Edge と CloudFront Functions

### 4.1 実行タイミング

```
リクエスト/レスポンスの4つのイベントポイント

  クライアント                               オリジン
     |                                        |
     |  Viewer Request                        |
     |  (CF Functions / Lambda@Edge)          |
     +------->+                               |
              |  Origin Request               |
              |  (Lambda@Edge のみ)            |
              +------------------------------>+
              |                               |
              |  Origin Response              |
              |  (Lambda@Edge のみ)            |
              +<------------------------------+
     |  Viewer Response                       |
     |  (CF Functions / Lambda@Edge)          |
     +<-------+                               |
     |                                        |
```

### 4.2 CloudFront Functions vs Lambda@Edge

| 特性 | CloudFront Functions | Lambda@Edge |
|------|---------------------|-------------|
| 実行場所 | 全 Edge Location (400+) | Regional Edge Cache (13) |
| 実行時間 | 最大 1ms | 最大 5s (Viewer) / 30s (Origin) |
| メモリ | 2MB | 128-3008 MB |
| ネットワークアクセス | 不可 | 可能 |
| ファイルシステム | 不可 | /tmp (512MB) |
| ランタイム | JavaScript (ES 5.1) | Node.js / Python |
| 料金 | $0.10/100万リクエスト | $0.60/100万リクエスト + 実行時間 |
| ログ | CloudWatch Logs (限定的) | CloudWatch Logs (各リージョン) |
| ユースケース | URL 書き換え、ヘッダー操作、単純認証 | A/Bテスト、認証、画像リサイズ、レスポンス生成 |

### 4.3 コード例: CloudFront Functions

```javascript
// === SPA のフォールバック: /about → /index.html ===
function handler(event) {
  var request = event.request;
  var uri = request.uri;

  // 拡張子がないパスは index.html にフォールバック
  if (!uri.includes('.')) {
    request.uri = '/index.html';
  }

  return request;
}
```

```javascript
// === Basic 認証 ===
function handler(event) {
  var request = event.request;
  var headers = request.headers;

  var authString = 'Basic ' + 'dXNlcjpwYXNzd29yZA=='; // user:password

  if (
    typeof headers.authorization === 'undefined' ||
    headers.authorization.value !== authString
  ) {
    return {
      statusCode: 401,
      statusDescription: 'Unauthorized',
      headers: {
        'www-authenticate': { value: 'Basic realm="Restricted"' }
      }
    };
  }

  return request;
}
```

```javascript
// === URL の正規化（トレイリングスラッシュの統一） ===
function handler(event) {
  var request = event.request;
  var uri = request.uri;

  // ファイル拡張子がない場合、末尾にスラッシュを追加
  if (!uri.endsWith('/') && !uri.includes('.')) {
    return {
      statusCode: 301,
      statusDescription: 'Moved Permanently',
      headers: {
        location: { value: uri + '/' }
      }
    };
  }

  // /index.html は / にリダイレクト
  if (uri.endsWith('/index.html')) {
    return {
      statusCode: 301,
      statusDescription: 'Moved Permanently',
      headers: {
        location: { value: uri.replace('/index.html', '/') }
      }
    };
  }

  return request;
}
```

```bash
# CloudFront Function を作成
aws cloudfront create-function \
  --name spa-url-rewrite \
  --function-config '{"Comment":"SPA URL rewrite","Runtime":"cloudfront-js-2.0"}' \
  --function-code fileb://function.js

# テスト
aws cloudfront test-function \
  --name spa-url-rewrite \
  --if-match EXXXXX \
  --event-object fileb://test-event.json

# テストイベントの例
# test-event.json:
# {
#   "version": "1.0",
#   "context": {"eventType": "viewer-request"},
#   "viewer": {"ip": "1.2.3.4"},
#   "request": {
#     "method": "GET",
#     "uri": "/about",
#     "querystring": {},
#     "headers": {}
#   }
# }

# 公開
aws cloudfront publish-function \
  --name spa-url-rewrite \
  --if-match EXXXXX

# ディストリビューションのビヘイビアに関連付け
aws cloudfront update-distribution --id EXXXXXXXXXX \
  --distribution-config '{
    ...
    "DefaultCacheBehavior": {
      ...
      "FunctionAssociations": {
        "Quantity": 1,
        "Items": [{
          "EventType": "viewer-request",
          "FunctionARN": "arn:aws:cloudfront::123456789012:function/spa-url-rewrite"
        }]
      }
    }
  }'
```

### 4.4 コード例: Lambda@Edge (画像リサイズ)

```javascript
// Origin Response で画像をリサイズ
const sharp = require('sharp');
const aws = require('aws-sdk');
const s3 = new aws.S3();

exports.handler = async (event) => {
  const response = event.Records[0].cf.response;
  const request = event.Records[0].cf.request;

  if (response.status === '200') {
    const params = new URLSearchParams(request.querystring);
    const width = parseInt(params.get('w')) || null;
    const height = parseInt(params.get('h')) || null;
    const format = params.get('f') || 'webp';  // webp, avif, jpeg, png

    if (width || height) {
      const s3Response = await s3.getObject({
        Bucket: 'my-images-bucket',
        Key: request.uri.substring(1),
      }).promise();

      let pipeline = sharp(s3Response.Body)
        .resize(width, height, { fit: 'inside', withoutEnlargement: true });

      // フォーマット変換
      switch (format) {
        case 'avif':
          pipeline = pipeline.avif({ quality: 80 });
          break;
        case 'webp':
          pipeline = pipeline.webp({ quality: 85 });
          break;
        case 'jpeg':
          pipeline = pipeline.jpeg({ quality: 85, progressive: true });
          break;
        default:
          pipeline = pipeline.webp({ quality: 85 });
      }

      const resized = await pipeline.toBuffer();

      // Lambda@Edge のレスポンスボディ上限は 1MB
      if (resized.length < 1048576) {
        response.body = resized.toString('base64');
        response.bodyEncoding = 'base64';
        response.headers['content-type'] = [{ value: `image/${format}` }];
        response.headers['cache-control'] = [{ value: 'public, max-age=31536000, immutable' }];
      }
    }
  }

  return response;
};
```

### 4.5 コード例: Lambda@Edge (A/B テスト)

```javascript
// Viewer Request で A/B テストのルーティングを行う
exports.handler = async (event) => {
  const request = event.Records[0].cf.request;
  const headers = request.headers;

  // 既存の Cookie からバリアント判定
  const cookies = headers.cookie || [];
  let variant = null;

  for (const cookie of cookies) {
    const match = cookie.value.match(/ab-variant=([AB])/);
    if (match) {
      variant = match[1];
      break;
    }
  }

  // Cookie がない場合、ランダムに振り分け
  if (!variant) {
    variant = Math.random() < 0.5 ? 'A' : 'B';
  }

  // バリアントに応じてオリジンパスを変更
  if (variant === 'B') {
    request.origin.s3.path = '/variant-b';
  }

  // カスタムヘッダーにバリアント情報を追加
  request.headers['x-ab-variant'] = [{ key: 'X-AB-Variant', value: variant }];

  return request;
};
```

```javascript
// Viewer Response で A/B テストの Cookie を設定
exports.handler = async (event) => {
  const response = event.Records[0].cf.response;
  const request = event.Records[0].cf.request;

  const variant = request.headers['x-ab-variant']
    ? request.headers['x-ab-variant'][0].value
    : 'A';

  // Cookie を設定（30日間有効）
  response.headers['set-cookie'] = response.headers['set-cookie'] || [];
  response.headers['set-cookie'].push({
    value: `ab-variant=${variant}; Path=/; Max-Age=2592000; SameSite=Lax`
  });

  return response;
};
```

---

## 5. OAC (Origin Access Control)

### 5.1 OAC vs OAI

```
OAI (旧方式、非推奨):
  CloudFront --- OAI (特別な IAM) ---> S3
  制限: SSE-KMS 非対応、署名 V2、S3 のみ

OAC (新方式、推奨):
  CloudFront --- OAC (SigV4 署名) ---> S3 / MediaStore / Lambda URL
  利点:
  - SSE-KMS 暗号化バケットに対応
  - 署名 V4（最新の署名方式）
  - S3 以外のオリジンタイプにも対応
  - きめ細かいポリシー制御
  - 全 AWS リージョンをサポート
```

### 5.2 コード例: OAC の設定

```bash
# OAC を作成
OAC_ID=$(aws cloudfront create-origin-access-control \
  --origin-access-control-config '{
    "Name": "my-s3-oac",
    "Description": "OAC for S3 origin",
    "OriginAccessControlOriginType": "s3",
    "SigningBehavior": "always",
    "SigningProtocol": "sigv4"
  }' --query 'OriginAccessControl.Id' --output text)

echo "OAC ID: $OAC_ID"

# S3 バケットポリシー（CloudFront からのアクセスのみ許可）
aws s3api put-bucket-policy --bucket my-static-bucket --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontServicePrincipal",
      "Effect": "Allow",
      "Principal": {
        "Service": "cloudfront.amazonaws.com"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-static-bucket/*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::123456789012:distribution/EXXXXXXXXXX"
        }
      }
    }
  ]
}'

# SSE-KMS 暗号化バケットの場合、KMS キーポリシーも必要
# KMS キーポリシーに CloudFront サービスプリンシパルを追加
aws kms put-key-policy --key-id alias/my-s3-key --policy-name default --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontDecrypt",
      "Effect": "Allow",
      "Principal": {
        "Service": "cloudfront.amazonaws.com"
      },
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::123456789012:distribution/EXXXXXXXXXX"
        }
      }
    }
  ]
}'
```

### 5.3 OAI から OAC への移行

```bash
# 1. OAC を作成
OAC_ID=$(aws cloudfront create-origin-access-control \
  --origin-access-control-config '{
    "Name": "migration-oac",
    "OriginAccessControlOriginType": "s3",
    "SigningBehavior": "always",
    "SigningProtocol": "sigv4"
  }' --query 'OriginAccessControl.Id' --output text)

# 2. ディストリビューションを更新（OAI → OAC）
# ETag を取得
ETAG=$(aws cloudfront get-distribution-config --id EXXXXXXXXXX --query 'ETag' --output text)

# config を取得して OAC に変更
aws cloudfront get-distribution-config --id EXXXXXXXXXX --query 'DistributionConfig' > dist-config.json

# dist-config.json を編集:
# Origins.Items[].S3OriginConfig.OriginAccessIdentity を "" に変更
# Origins.Items[].OriginAccessControlId に OAC_ID を設定

aws cloudfront update-distribution \
  --id EXXXXXXXXXX \
  --if-match $ETAG \
  --distribution-config file://dist-config.json

# 3. S3 バケットポリシーを OAC 用に更新
# (上記 5.2 の AllowCloudFrontServicePrincipal を追加)

# 4. 動作確認後、OAI を削除
aws cloudfront delete-cloud-front-origin-access-identity --id OAIXXXXXXXXX --if-match $OAI_ETAG
```

---

## 6. セキュリティヘッダーとレスポンスポリシー

### 6.1 コード例: レスポンスヘッダーポリシー

```bash
aws cloudfront create-response-headers-policy \
  --response-headers-policy-config '{
  "Name": "SecurityHeaders",
  "SecurityHeadersConfig": {
    "StrictTransportSecurity": {
      "Override": true,
      "AccessControlMaxAgeSec": 31536000,
      "IncludeSubdomains": true,
      "Preload": true
    },
    "ContentTypeOptions": {
      "Override": true
    },
    "FrameOptions": {
      "Override": true,
      "FrameOption": "DENY"
    },
    "XSSProtection": {
      "Override": true,
      "Protection": true,
      "ModeBlock": true
    },
    "ContentSecurityPolicy": {
      "Override": true,
      "ContentSecurityPolicy": "default-src '\''self'\''; img-src '\''self'\'' data: https:; script-src '\''self'\''"
    },
    "ReferrerPolicy": {
      "Override": true,
      "ReferrerPolicy": "strict-origin-when-cross-origin"
    }
  },
  "CorsConfig": {
    "AccessControlAllowOrigins": {
      "Quantity": 1,
      "Items": ["https://example.com"]
    },
    "AccessControlAllowHeaders": {
      "Quantity": 1,
      "Items": ["*"]
    },
    "AccessControlAllowMethods": {
      "Quantity": 3,
      "Items": ["GET", "HEAD", "OPTIONS"]
    },
    "AccessControlAllowCredentials": false,
    "OriginOverride": true
  },
  "CustomHeadersConfig": {
    "Quantity": 1,
    "Items": [{
      "Header": "X-Robots-Tag",
      "Value": "noindex, nofollow",
      "Override": true
    }]
  },
  "ServerTimingHeadersConfig": {
    "Enabled": true,
    "SamplingRate": 50
  }
}'
```

### 6.2 Server-Timing ヘッダー

```
# Server-Timing ヘッダーの出力例
Server-Timing: cdn-cache-hit;desc="Hit from cloudfront", cdn-upstream-layer;desc="EDGE"

# SamplingRate で出力比率を制御（0-100）
# 50 = 50% のリクエストに Server-Timing ヘッダーを付与
# パフォーマンス測定に有用だが、本番では 1-10% 程度に設定
```

---

## 7. 署名付き URL / Cookie

### 7.1 署名付きアクセスの概要

```
署名付き URL vs 署名付き Cookie

署名付き URL:
  - 単一ファイルへのアクセス制御
  - メール等で共有する一時的なリンク
  - 例: https://d111.cloudfront.net/premium/video.mp4?
        Expires=1708099200&Signature=xxxx&Key-Pair-Id=KYYY

署名付き Cookie:
  - 複数ファイルへのアクセス制御
  - ログイン済みユーザーへの限定コンテンツ配信
  - 例: Set-Cookie: CloudFront-Policy=xxx;
        Set-Cookie: CloudFront-Signature=yyy;
        Set-Cookie: CloudFront-Key-Pair-Id=zzz;
```

### 7.2 コード例: 署名付き URL の生成

```bash
# CloudFront キーペアの作成（パブリックキーのアップロード）
# まず RSA キーペアを生成
openssl genrsa -out private_key.pem 2048
openssl rsa -in private_key.pem -pubout -out public_key.pem

# パブリックキーを CloudFront に登録
PUBLIC_KEY_ID=$(aws cloudfront create-public-key \
  --public-key-config '{
    "CallerReference": "my-key-2024",
    "Name": "my-signing-key",
    "EncodedKey": "'"$(cat public_key.pem)"'"
  }' --query 'PublicKey.Id' --output text)

# キーグループを作成
KEY_GROUP_ID=$(aws cloudfront create-key-group \
  --key-group-config '{
    "Name": "my-key-group",
    "Items": ["'"$PUBLIC_KEY_ID"'"]
  }' --query 'KeyGroup.Id' --output text)

echo "Key Group ID: $KEY_GROUP_ID"
# ディストリビューションのビヘイビアに TrustedKeyGroups として設定
```

```python
# Python で署名付き URL を生成
import datetime
from botocore.signers import CloudFrontSigner
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def rsa_signer(message):
    with open('private_key.pem', 'rb') as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(), password=None
        )
    return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())

key_id = 'KXXXXXXXXXX'
cf_signer = CloudFrontSigner(key_id, rsa_signer)

# 署名付き URL を生成（1時間有効）
url = cf_signer.generate_presigned_url(
    url='https://d111111abcdef8.cloudfront.net/premium/video.mp4',
    date_less_than=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
)
print(url)

# カスタムポリシーで IP 制限付き署名 URL を生成
from botocore.signers import CloudFrontSigner
import json

custom_policy = json.dumps({
    "Statement": [{
        "Resource": "https://d111111abcdef8.cloudfront.net/premium/*",
        "Condition": {
            "DateLessThan": {"AWS:EpochTime": int((datetime.datetime.utcnow() + datetime.timedelta(hours=1)).timestamp())},
            "IpAddress": {"AWS:SourceIp": "203.0.113.0/24"}
        }
    }]
})

signed_url = cf_signer.generate_presigned_url(
    url='https://d111111abcdef8.cloudfront.net/premium/video.mp4',
    policy=custom_policy
)
```

---

## 8. 地理的制限とアクセス制御

### 8.1 地理的制限の設定

```bash
# 特定の国からのアクセスをブロック
aws cloudfront update-distribution --id EXXXXXXXXXX \
  --distribution-config '{
    ...
    "Restrictions": {
      "GeoRestriction": {
        "RestrictionType": "blacklist",
        "Quantity": 2,
        "Items": ["CN", "RU"]
      }
    }
  }'

# 特定の国のみ許可（ホワイトリスト）
aws cloudfront update-distribution --id EXXXXXXXXXX \
  --distribution-config '{
    ...
    "Restrictions": {
      "GeoRestriction": {
        "RestrictionType": "whitelist",
        "Quantity": 1,
        "Items": ["JP"]
      }
    }
  }'
```

### 8.2 WAF 連携

```bash
# WAF Web ACL を CloudFront に関連付け
aws wafv2 create-web-acl \
  --name cloudfront-waf \
  --scope CLOUDFRONT \
  --region us-east-1 \
  --default-action '{"Allow":{}}' \
  --rules '[
    {
      "Name": "RateLimit",
      "Priority": 1,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 2000,
          "AggregateKeyType": "IP"
        }
      },
      "Action": {"Block": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "RateLimit"
      }
    },
    {
      "Name": "AWSManagedRulesCommonRuleSet",
      "Priority": 2,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "OverrideAction": {"None": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "CommonRuleSet"
      }
    },
    {
      "Name": "AWSManagedRulesSQLiRuleSet",
      "Priority": 3,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesSQLiRuleSet"
        }
      },
      "OverrideAction": {"None": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "SQLiRuleSet"
      }
    }
  ]' \
  --visibility-config '{
    "SampledRequestsEnabled": true,
    "CloudWatchMetricsEnabled": true,
    "MetricName": "cloudfront-waf"
  }'

# WAF Web ACL をディストリビューションに関連付け
# ディストリビューション作成/更新時に WebACLId を指定
```

---

## 9. CloudFormation / CDK による構築

### 9.1 CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: CloudFront Distribution with S3 + ALB origins, OAC, WAF

Parameters:
  DomainName:
    Type: String
    Description: カスタムドメイン名
  CertificateArn:
    Type: String
    Description: ACM 証明書 ARN (us-east-1)
  AlbDomainName:
    Type: String
    Description: ALB のドメイン名
  HostedZoneId:
    Type: String
    Description: Route 53 ホストゾーン ID

Resources:
  # S3 バケット（静的コンテンツ）
  StaticBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-static'
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

  # S3 バケットポリシー
  StaticBucketPolicy:
    Type: AWS::S3::BucketPolicy
    Properties:
      Bucket: !Ref StaticBucket
      PolicyDocument:
        Statement:
          - Sid: AllowCloudFront
            Effect: Allow
            Principal:
              Service: cloudfront.amazonaws.com
            Action: s3:GetObject
            Resource: !Sub '${StaticBucket.Arn}/*'
            Condition:
              StringEquals:
                AWS:SourceArn: !Sub 'arn:aws:cloudfront::${AWS::AccountId}:distribution/${Distribution}'

  # OAC
  OriginAccessControl:
    Type: AWS::CloudFront::OriginAccessControl
    Properties:
      OriginAccessControlConfig:
        Name: !Sub '${AWS::StackName}-oac'
        OriginAccessControlOriginType: s3
        SigningBehavior: always
        SigningProtocol: sigv4

  # キャッシュポリシー（静的コンテンツ）
  StaticCachePolicy:
    Type: AWS::CloudFront::CachePolicy
    Properties:
      CachePolicyConfig:
        Name: !Sub '${AWS::StackName}-static-cache'
        DefaultTTL: 86400
        MaxTTL: 31536000
        MinTTL: 0
        ParametersInCacheKeyAndForwardedToOrigin:
          EnableAcceptEncodingGzip: true
          EnableAcceptEncodingBrotli: true
          CookiesConfig:
            CookieBehavior: none
          HeadersConfig:
            HeaderBehavior: none
          QueryStringsConfig:
            QueryStringBehavior: whitelist
            QueryStrings:
              - v
              - ver

  # レスポンスヘッダーポリシー
  SecurityHeadersPolicy:
    Type: AWS::CloudFront::ResponseHeadersPolicy
    Properties:
      ResponseHeadersPolicyConfig:
        Name: !Sub '${AWS::StackName}-security-headers'
        SecurityHeadersConfig:
          StrictTransportSecurity:
            AccessControlMaxAgeSec: 31536000
            IncludeSubdomains: true
            Override: true
            Preload: true
          ContentTypeOptions:
            Override: true
          FrameOptions:
            FrameOption: DENY
            Override: true
          ReferrerPolicy:
            ReferrerPolicy: strict-origin-when-cross-origin
            Override: true

  # CloudFront Function (SPA URL 書き換え)
  SpaRewriteFunction:
    Type: AWS::CloudFront::Function
    Properties:
      Name: !Sub '${AWS::StackName}-spa-rewrite'
      AutoPublish: true
      FunctionConfig:
        Comment: SPA URL rewrite for single page application
        Runtime: cloudfront-js-2.0
      FunctionCode: |
        function handler(event) {
          var request = event.request;
          var uri = request.uri;
          if (!uri.includes('.')) {
            request.uri = '/index.html';
          }
          return request;
        }

  # ディストリビューション
  Distribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Enabled: true
        DefaultRootObject: index.html
        PriceClass: PriceClass_200
        HttpVersion: http2and3
        Comment: !Sub '${AWS::StackName} production distribution'
        Aliases:
          - !Ref DomainName
        ViewerCertificate:
          AcmCertificateArn: !Ref CertificateArn
          SslSupportMethod: sni-only
          MinimumProtocolVersion: TLSv1.2_2021
        Origins:
          - Id: S3Origin
            DomainName: !GetAtt StaticBucket.RegionalDomainName
            OriginAccessControlId: !Ref OriginAccessControl
            S3OriginConfig:
              OriginAccessIdentity: ''
          - Id: AlbOrigin
            DomainName: !Ref AlbDomainName
            CustomOriginConfig:
              HTTPSPort: 443
              OriginProtocolPolicy: https-only
              OriginSSLProtocols:
                - TLSv1.2
            OriginCustomHeaders:
              - HeaderName: X-Origin-Verify
                HeaderValue: !Sub '{{resolve:secretsmanager:${AWS::StackName}/origin-verify:SecretString:token}}'
        DefaultCacheBehavior:
          TargetOriginId: S3Origin
          ViewerProtocolPolicy: redirect-to-https
          CachePolicyId: !Ref StaticCachePolicy
          ResponseHeadersPolicyId: !Ref SecurityHeadersPolicy
          Compress: true
          FunctionAssociations:
            - EventType: viewer-request
              FunctionARN: !GetAtt SpaRewriteFunction.FunctionMetadata.FunctionARN
        CacheBehaviors:
          - PathPattern: /api/*
            TargetOriginId: AlbOrigin
            ViewerProtocolPolicy: https-only
            CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad
            OriginRequestPolicyId: 216adef6-5c7f-47e4-b989-5492eafa07d3
            AllowedMethods:
              - GET
              - HEAD
              - OPTIONS
              - PUT
              - POST
              - PATCH
              - DELETE
            Compress: true
          - PathPattern: /static/*
            TargetOriginId: S3Origin
            ViewerProtocolPolicy: redirect-to-https
            CachePolicyId: 658327ea-f89d-4fab-a63d-7e88639e58f6
            Compress: true
        CustomErrorResponses:
          - ErrorCode: 403
            ResponseCode: 200
            ResponsePagePath: /index.html
            ErrorCachingMinTTL: 10
          - ErrorCode: 404
            ResponseCode: 200
            ResponsePagePath: /index.html
            ErrorCachingMinTTL: 10

  # Route 53 レコード
  DnsRecord:
    Type: AWS::Route53::RecordSet
    Properties:
      HostedZoneId: !Ref HostedZoneId
      Name: !Ref DomainName
      Type: A
      AliasTarget:
        DNSName: !GetAtt Distribution.DomainName
        HostedZoneId: Z2FDTNDATAQYW2  # CloudFront のグローバルホストゾーン ID

Outputs:
  DistributionId:
    Value: !Ref Distribution
  DistributionDomainName:
    Value: !GetAtt Distribution.DomainName
  BucketName:
    Value: !Ref StaticBucket
```

### 9.2 CDK (TypeScript) による構築

```typescript
import * as cdk from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as targets from 'aws-cdk-lib/aws-route53-targets';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import { Construct } from 'constructs';

interface CloudFrontStackProps extends cdk.StackProps {
  domainName: string;
  hostedZoneId: string;
  zoneName: string;
  albArn: string;
}

export class CloudFrontStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: CloudFrontStackProps) {
    super(scope, id, props);

    // S3 バケット
    const bucket = new s3.Bucket(this, 'StaticBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // ACM 証明書（us-east-1 が必要）
    const certificate = acm.Certificate.fromCertificateArn(
      this, 'Cert',
      `arn:aws:acm:us-east-1:${this.account}:certificate/xxx`
    );

    // ALB の参照
    const alb = elbv2.ApplicationLoadBalancer.fromLookup(this, 'ALB', {
      loadBalancerArn: props.albArn,
    });

    // CloudFront Function（SPA 書き換え）
    const spaRewrite = new cloudfront.Function(this, 'SpaRewrite', {
      code: cloudfront.FunctionCode.fromInline(`
        function handler(event) {
          var request = event.request;
          if (!request.uri.includes('.')) {
            request.uri = '/index.html';
          }
          return request;
        }
      `),
      runtime: cloudfront.FunctionRuntime.JS_2_0,
    });

    // レスポンスヘッダーポリシー
    const responseHeadersPolicy = new cloudfront.ResponseHeadersPolicy(this, 'SecurityHeaders', {
      securityHeadersBehavior: {
        strictTransportSecurity: {
          accessControlMaxAge: cdk.Duration.days(365),
          includeSubdomains: true,
          preload: true,
          override: true,
        },
        contentTypeOptions: { override: true },
        frameOptions: {
          frameOption: cloudfront.HeadersFrameOption.DENY,
          override: true,
        },
        referrerPolicy: {
          referrerPolicy: cloudfront.HeadersReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
          override: true,
        },
      },
    });

    // ディストリビューション
    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(bucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
        responseHeadersPolicy,
        functionAssociations: [{
          function: spaRewrite,
          eventType: cloudfront.FunctionEventType.VIEWER_REQUEST,
        }],
        compress: true,
      },
      additionalBehaviors: {
        '/api/*': {
          origin: new origins.LoadBalancerV2Origin(alb, {
            protocolPolicy: cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
            customHeaders: {
              'X-Origin-Verify': 'my-secret-value',
            },
          }),
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        },
      },
      domainNames: [props.domainName],
      certificate,
      priceClass: cloudfront.PriceClass.PRICE_CLASS_200,
      httpVersion: cloudfront.HttpVersion.HTTP2_AND_3,
      errorResponses: [
        { httpStatus: 403, responsePagePath: '/index.html', responseHttpStatus: 200, ttl: cdk.Duration.seconds(10) },
        { httpStatus: 404, responsePagePath: '/index.html', responseHttpStatus: 200, ttl: cdk.Duration.seconds(10) },
      ],
    });

    // Route 53 レコード
    const hostedZone = route53.HostedZone.fromHostedZoneAttributes(this, 'Zone', {
      hostedZoneId: props.hostedZoneId,
      zoneName: props.zoneName,
    });

    new route53.ARecord(this, 'AliasRecord', {
      zone: hostedZone,
      recordName: props.domainName,
      target: route53.RecordTarget.fromAlias(new targets.CloudFrontTarget(distribution)),
    });

    // 出力
    new cdk.CfnOutput(this, 'DistributionId', { value: distribution.distributionId });
    new cdk.CfnOutput(this, 'BucketName', { value: bucket.bucketName });
  }
}
```

---

## 10. 監視とトラブルシューティング

### 10.1 CloudWatch メトリクス

```bash
# リクエスト数の取得
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name Requests \
  --dimensions Name=DistributionId,Value=EXXXXXXXXXX Name=Region,Value=Global \
  --start-time "$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 3600 \
  --statistics Sum

# 主要メトリクス一覧
# Requests:         リクエスト総数
# BytesDownloaded:  ダウンロードバイト数
# BytesUploaded:    アップロードバイト数
# TotalErrorRate:   全エラー率
# 4xxErrorRate:     4xx エラー率
# 5xxErrorRate:     5xx エラー率
# CacheHitRate:     キャッシュヒット率
# OriginLatency:    オリジンレイテンシ

# CloudWatch アラームの設定（5xx エラー率）
aws cloudwatch put-metric-alarm \
  --alarm-name "CloudFront-5xx-Error-Rate" \
  --metric-name 5xxErrorRate \
  --namespace AWS/CloudFront \
  --dimensions Name=DistributionId,Value=EXXXXXXXXXX Name=Region,Value=Global \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions "arn:aws:sns:us-east-1:123456789012:alerts"
```

### 10.2 CloudFront ログの分析

```bash
# 標準ログ（S3 配信）の有効化
aws cloudfront update-distribution --id EXXXXXXXXXX \
  --distribution-config '{
    ...
    "Logging": {
      "Enabled": true,
      "IncludeCookies": false,
      "Bucket": "my-cf-logs.s3.amazonaws.com",
      "Prefix": "production/"
    }
  }'

# リアルタイムログの設定（Kinesis Data Streams 連携）
aws cloudfront create-realtime-log-config \
  --name production-realtime-logs \
  --sampling-rate 100 \
  --fields '["timestamp","c-ip","sc-status","cs-method","cs-uri-stem","cs-bytes","time-taken","x-edge-result-type","x-edge-response-result-type"]' \
  --end-points '[{
    "StreamType": "Kinesis",
    "KinesisStreamConfig": {
      "RoleARN": "arn:aws:iam::123456789012:role/cloudfront-realtime-log-role",
      "StreamARN": "arn:aws:kinesis:us-east-1:123456789012:stream/cf-realtime-logs"
    }
  }]'

# Athena で標準ログを分析
# テーブル作成
# CREATE EXTERNAL TABLE cloudfront_logs (
#   `date` date, time string, x_edge_location string,
#   sc_bytes bigint, c_ip string, cs_method string,
#   cs_host string, cs_uri_stem string, sc_status int,
#   cs_referer string, cs_user_agent string, cs_uri_query string,
#   cs_cookie string, x_edge_result_type string,
#   x_edge_request_id string, x_host_header string,
#   cs_protocol string, cs_bytes bigint, time_taken float,
#   x_forwarded_for string, ssl_protocol string,
#   ssl_cipher string, x_edge_response_result_type string,
#   cs_protocol_version string, fle_status string,
#   fle_encrypted_fields int, c_port int,
#   time_to_first_byte float, x_edge_detailed_result_type string,
#   sc_content_type string, sc_content_len bigint,
#   sc_range_start bigint, sc_range_end bigint
# )
# ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
# LOCATION 's3://my-cf-logs/production/'
# TBLPROPERTIES ('skip.header.line.count'='2');

# 上位 404 パスの集計
# SELECT cs_uri_stem, COUNT(*) as cnt
# FROM cloudfront_logs
# WHERE sc_status = 404
# GROUP BY cs_uri_stem
# ORDER BY cnt DESC LIMIT 20;
```

---

## 11. アンチパターン

### アンチパターン 1: API レスポンスを長時間キャッシュする

動的な API レスポンスを長い TTL でキャッシュすると、ユーザーに古いデータが返り続ける。API はキャッシュ無効 (`CachingDisabled`) か短い TTL を設定すべきである。

```
# 悪い例
/api/* → CachingOptimized (TTL: 24時間)
→ ユーザー情報が24時間古いまま

# 良い例
/api/* → CachingDisabled (キャッシュなし)
/static/* → CachingOptimized (TTL: 24時間)
/api/public/* → カスタムポリシー (TTL: 5秒)  ← 公開 API は短い TTL
```

### アンチパターン 2: OAI (旧方式) を新規構成で使い続ける

OAI は SSE-KMS 暗号化バケットに対応しておらず、SigV2 ベースで将来的な廃止が予想される。新規構成では必ず OAC を使用すべきである。

### アンチパターン 3: Invalidation を頻繁に使う

```
# 悪い例
デプロイのたびに /* を Invalidation
→ 月 1,000 パスを超えると課金
→ 全 Edge Location への伝播に数分かかる

# 良い例
ファイル名にコンテンツハッシュを含める
app.abc123.js, styles.def456.css
→ ファイル名が変わるのでキャッシュが自然に更新される
→ index.html のみ短い TTL を設定
```

### アンチパターン 4: キャッシュキーに不要な要素を含める

```
# 悪い例
全てのヘッダーとクエリ文字列をキャッシュキーに含める
→ ヘッダーの微差でキャッシュミス（ヒット率低下）
→ 同じコンテンツが異なるキーで重複キャッシュ

# 良い例
必要最小限のキャッシュキーを設定
- 静的コンテンツ: ヘッダーなし、Cookie なし、クエリ文字列なし
- API: Authorization ヘッダー + 全クエリ文字列
```

### アンチパターン 5: カスタムエラーページを設定しない

```
# 悪い例
S3 オリジンで存在しないパスにアクセス
→ 403 Forbidden の XML エラーが表示される
→ ユーザー体験が悪い

# 良い例
CustomErrorResponses で 403/404 を /index.html (200) にマップ
→ SPA がクライアント側でルーティング処理
→ カスタム 404 ページを表示
```

---

## 12. FAQ

### Q1. CloudFront の料金体系は？

主に (1) データ転送量 (GB あたり)、(2) HTTP/HTTPS リクエスト数、(3) Lambda@Edge / CloudFront Functions 実行数で課金される。価格クラスを制限 (PriceClass_100 など) すると、一部リージョンの Edge を除外しコストを削減できる。Shield Standard（DDoS 防御）は無料で含まれる。

### Q2. キャッシュ無効化 (Invalidation) のコストは？

月間 1,000 パスまで無料、それ以降は 1 パスあたり $0.005。`/*` は 1 パスとしてカウントされる。頻繁な無効化が必要な場合は、ファイル名にバージョン文字列（例: `app.abc123.js`）を含める方が効率的。

### Q3. CloudFront で SPA (React / Vue) を配信する際の注意点は？

CloudFront Functions で URL 書き換えを行い、拡張子のないパス (`/about`, `/users/123`) を `/index.html` にフォールバックさせる。カスタムエラーページで 403/404 を `index.html` にリダイレクトする方法もあるが、CloudFront Functions の方が柔軟。HTTP/2 と Brotli 圧縮を有効にするとパフォーマンスが大幅に向上する。

### Q4. CloudFront と S3 の静的ウェブサイトホスティングの違いは？

S3 静的ウェブサイトホスティングは HTTP のみで HTTPS 非対応、カスタムドメインの HTTPS にはCloudFront が必須。CloudFront は HTTPS、HTTP/2、HTTP/3、Brotli 圧縮、地理的制限、WAF 連携など多くの機能を提供する。コスト面でも、CloudFront 経由の S3 アクセスはデータ転送料金が無料になるため、直接 S3 からの配信よりも安くなる場合がある。

### Q5. CloudFront の HTTP/3 (QUIC) を有効にするには？

ディストリビューションの設定で `HttpVersion` を `http2and3` に設定するだけで有効化できる。HTTP/3 はクライアントが対応していれば自動的に使用され、非対応の場合は HTTP/2 にフォールバックする。

```bash
# HTTP/3 の有効化
# ディストリビューション作成/更新時に HttpVersion を設定
aws cloudfront update-distribution --id EXXXXXXXXXX \
  --distribution-config '{
    ...
    "HttpVersion": "http2and3"
  }'
```

### Q6. CloudFront のキャッシュが反映されているか確認する方法は？

```bash
# curl でレスポンスヘッダーを確認
curl -I https://www.example.com/index.html

# チェックすべきヘッダー:
# X-Cache: Hit from cloudfront  → キャッシュヒット
# X-Cache: Miss from cloudfront → キャッシュミス
# X-Cache: RefreshHit from cloudfront → TTL 切れ後の再取得
# Age: 3600 → キャッシュされてからの秒数
# X-Amz-Cf-Pop: NRT52-C4 → Edge Location の識別子
```

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| Edge Location | 400+ 拠点、ユーザーに最も近い場所からコンテンツ配信 |
| オリジン | S3 (静的) + ALB (動的) のマルチオリジン構成が一般的 |
| フェイルオーバー | Origin Group で Primary/Secondary の自動切替 |
| キャッシュ | 静的=長 TTL、動的=キャッシュ無効、バージョン付きファイル名推奨 |
| Lambda@Edge | 認証、A/B テスト、画像リサイズ等のエッジ処理 |
| CloudFront Functions | URL 書き換え、ヘッダー操作（軽量・安価） |
| OAC | S3 オリジンへのセキュアアクセス（OAI より推奨） |
| セキュリティ | レスポンスヘッダーポリシー + WAF 連携 + 署名付き URL/Cookie |
| 監視 | CloudWatch メトリクス + 標準ログ + リアルタイムログ |
| IaC | CloudFormation / CDK で宣言的に管理 |

---

## 次に読むべきガイド

- [../03-database/00-rds-basics.md](../03-database/00-rds-basics.md) — RDS の基礎
- [../04-networking/01-route53.md](../04-networking/01-route53.md) — Route 53 DNS 設定

---

## 参考文献

1. Amazon CloudFront 開発者ガイド — https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/
2. CloudFront キャッシュポリシー — https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/controlling-the-cache-key.html
3. Lambda@Edge 開発者ガイド — https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/lambda-at-the-edge.html
4. OAC ユーザーガイド — https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html
5. CloudFront Functions 開発者ガイド — https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/cloudfront-functions.html
6. CloudFront 署名付き URL — https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-signed-urls.html
7. AWS WAF と CloudFront の統合 — https://docs.aws.amazon.com/waf/latest/developerguide/cloudfront-features.html
