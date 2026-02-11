# CloudFront

> AWS のグローバル CDN — キャッシュポリシー、オリジン設定、Lambda@Edge、OAC でコンテンツ配信を最適化する

## この章で学ぶこと

1. CloudFront のディストリビューションとオリジンを設計し、効率的なコンテンツ配信を構築できる
2. キャッシュポリシーとキャッシュ無効化を適切に設定し、パフォーマンスと鮮度を両立できる
3. Lambda@Edge / CloudFront Functions と OAC を使って、エッジでの処理とセキュアなオリジンアクセスを実現できる

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

---

## 2. オリジン設定

### 2.1 オリジンタイプ比較

| オリジンタイプ | 用途 | 認証方式 |
|--------------|------|---------|
| S3 バケット | 静的ファイル | OAC (推奨) |
| ALB | 動的コンテンツ | カスタムヘッダー |
| EC2 | 直接接続 | セキュリティグループ |
| API Gateway | API | IAM / Cognito |
| カスタムオリジン | 外部サーバー | 共有シークレット |

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
          "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]}
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

| ポリシー名 | TTL | クエリ文字列 | ヘッダー | 用途 |
|-----------|-----|------------|---------|------|
| CachingOptimized | 86400s | なし | なし | 静的コンテンツ |
| CachingOptimizedForUncompressedObjects | 86400s | なし | なし | 非圧縮 |
| CachingDisabled | 0s | 全て | 全て | API / 動的ページ |
| Amplify | 2s | 全て | Authorization | Amplify アプリ |

### 3.3 コード例: カスタムキャッシュポリシー

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
```

### 3.4 コード例: キャッシュ無効化 (Invalidation)

```bash
# 特定パスのキャッシュを無効化
aws cloudfront create-invalidation \
  --distribution-id EXXXXXXXXXX \
  --paths '/index.html' '/css/*' '/js/*'

# 全キャッシュを無効化（コスト注意）
aws cloudfront create-invalidation \
  --distribution-id EXXXXXXXXXX \
  --paths '/*'

# 無効化の状態確認
aws cloudfront get-invalidation \
  --distribution-id EXXXXXXXXXX \
  --id IXXXXXXXXX
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
| 実行場所 | 全 Edge Location | Regional Edge Cache |
| 実行時間 | 最大 1ms | 最大 5s (Viewer) / 30s (Origin) |
| メモリ | 2MB | 128-3008 MB |
| ネットワークアクセス | 不可 | 可能 |
| ランタイム | JavaScript のみ | Node.js / Python |
| 料金 | 非常に安い | Lambda@Edge 料金 |
| ユースケース | URL 書き換え、ヘッダー操作 | A/Bテスト、認証、画像リサイズ |

### 4.3 コード例: CloudFront Functions (URL 書き換え)

```javascript
// SPA のフォールバック: /about → /index.html
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

# 公開
aws cloudfront publish-function \
  --name spa-url-rewrite \
  --if-match EXXXXX
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

    if (width || height) {
      const s3Response = await s3.getObject({
        Bucket: 'my-images-bucket',
        Key: request.uri.substring(1),
      }).promise();

      const resized = await sharp(s3Response.Body)
        .resize(width, height, { fit: 'inside' })
        .toBuffer();

      response.body = resized.toString('base64');
      response.bodyEncoding = 'base64';
      response.headers['content-type'] = [{ value: 'image/webp' }];
      response.headers['cache-control'] = [{ value: 'public, max-age=31536000' }];
    }
  }

  return response;
};
```

---

## 5. OAC (Origin Access Control)

### 5.1 OAC vs OAI

```
OAI (旧方式、非推奨):
  CloudFront --- OAI (特別な IAM) ---> S3
  制限: SSE-KMS 非対応、署名 V2

OAC (新方式、推奨):
  CloudFront --- OAC (SigV4 署名) ---> S3
  利点: SSE-KMS 対応、署名 V4、きめ細かいポリシー
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
  }
}'
```

---

## 7. アンチパターン

### アンチパターン 1: API レスポンスを長時間キャッシュする

動的な API レスポンスを長い TTL でキャッシュすると、ユーザーに古いデータが返り続ける。API はキャッシュ無効 (`CachingDisabled`) か短い TTL を設定すべきである。

```
# 悪い例
/api/* → CachingOptimized (TTL: 24時間)
→ ユーザー情報が24時間古いまま

# 良い例
/api/* → CachingDisabled (キャッシュなし)
/static/* → CachingOptimized (TTL: 24時間)
```

### アンチパターン 2: OAI (旧方式) を新規構成で使い続ける

OAI は SSE-KMS 暗号化バケットに対応しておらず、SigV2 ベースで将来的な廃止が予想される。新規構成では必ず OAC を使用すべきである。

---

## 8. FAQ

### Q1. CloudFront の料金体系は？

主に (1) データ転送量 (GB あたり)、(2) HTTP/HTTPS リクエスト数、(3) Lambda@Edge / CloudFront Functions 実行数で課金される。価格クラスを制限 (PriceClass_100 など) すると、一部リージョンの Edge を除外しコストを削減できる。

### Q2. キャッシュ無効化 (Invalidation) のコストは？

月間 1,000 パスまで無料、それ以降は 1 パスあたり $0.005。`/*` は 1 パスとしてカウントされる。頻繁な無効化が必要な場合は、ファイル名にバージョン文字列（例: `app.abc123.js`）を含める方が効率的。

### Q3. CloudFront で SPA (React / Vue) を配信する際の注意点は？

CloudFront Functions で URL 書き換えを行い、拡張子のないパス (`/about`, `/users/123`) を `/index.html` にフォールバックさせる。カスタムエラーページで 403/404 を `index.html` にリダイレクトする方法もあるが、CloudFront Functions の方が柔軟。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| Edge Location | 400+ 拠点、ユーザーに最も近い場所からコンテンツ配信 |
| オリジン | S3 (静的) + ALB (動的) のマルチオリジン構成が一般的 |
| キャッシュ | 静的=長 TTL、動的=キャッシュ無効、バージョン付きファイル名推奨 |
| Lambda@Edge | 認証、A/B テスト、画像リサイズ等のエッジ処理 |
| CloudFront Functions | URL 書き換え、ヘッダー操作（軽量・安価） |
| OAC | S3 オリジンへのセキュアアクセス（OAI より推奨） |
| セキュリティ | レスポンスヘッダーポリシー + WAF 連携 |

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
