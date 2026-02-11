# TLS/証明書

> TLS ハンドシェイクから証明書チェーン、Let's Encrypt による自動化、mTLS まで、安全な通信の基盤技術を体系的に学ぶ

## この章で学ぶこと

1. **TLS ハンドシェイクの仕組み** — クライアントとサーバ間で暗号化通信が確立されるまでの全ステップ
2. **証明書チェーンと PKI** — ルート CA から中間 CA、サーバ証明書に至る信頼の連鎖
3. **実運用での証明書管理** — Let's Encrypt による自動化と mTLS による双方向認証

---

## 1. TLS の全体像

### TLS とは何か

TLS (Transport Layer Security) はトランスポート層の上で動作する暗号化プロトコルである。SSL の後継として策定され、現在の推奨バージョンは TLS 1.3 である。

```
+-------------------+
|   Application     |  HTTP, SMTP, etc.
+-------------------+
|       TLS         |  暗号化・認証・完全性
+-------------------+
|       TCP         |  信頼性のある配送
+-------------------+
|       IP          |  ルーティング
+-------------------+
```

### TLS のバージョン比較

| バージョン | 状態 | ハンドシェイク RTT | 主な特徴 |
|-----------|------|-------------------|---------|
| SSL 3.0 | 廃止 (2015) | 2-RTT | POODLE 脆弱性 |
| TLS 1.0 | 廃止 (2020) | 2-RTT | BEAST 脆弱性 |
| TLS 1.1 | 廃止 (2020) | 2-RTT | CBC 改善 |
| TLS 1.2 | 現役 | 2-RTT | AEAD 暗号スイート |
| TLS 1.3 | 推奨 | 1-RTT (0-RTT可) | ハンドシェイク簡素化 |

---

## 2. TLS 1.3 ハンドシェイク

### ハンドシェイクの流れ

```
Client                                    Server
  |                                          |
  |--- ClientHello ----------------------->  |
  |    + supported_versions                  |
  |    + key_share (ECDHE)                   |
  |    + signature_algorithms                |
  |                                          |
  |  <--- ServerHello ---------------------  |
  |       + key_share (ECDHE)                |
  |  <--- EncryptedExtensions -------------  |
  |  <--- Certificate ---------------------  |
  |  <--- CertificateVerify --------------  |
  |  <--- Finished -----------------------  |
  |                                          |
  |--- Finished ------------------------->   |
  |                                          |
  |========= 暗号化通信開始 ================|
```

### OpenSSL でハンドシェイクを確認

```bash
# TLS 1.3 ハンドシェイクの詳細を表示
openssl s_client -connect example.com:443 -tls1_3 -msg

# 証明書チェーンを確認
openssl s_client -connect example.com:443 -showcerts

# 使用される暗号スイートの確認
openssl s_client -connect example.com:443 -cipher 'TLS_AES_256_GCM_SHA384'
```

### TLS 1.2 と 1.3 のハンドシェイク比較

| 項目 | TLS 1.2 | TLS 1.3 |
|------|---------|---------|
| ラウンドトリップ | 2-RTT | 1-RTT |
| 0-RTT 再接続 | 不可 | 可能 |
| 鍵交換 | RSA / ECDHE | ECDHE のみ |
| 暗号スイート数 | 数十個 | 5個 |
| ハンドシェイク暗号化 | なし | ServerHello 以降暗号化 |
| 前方秘匿性 | オプション | 必須 |

---

## 3. 証明書チェーンと PKI

### 証明書チェーンの構造

```
+---------------------------+
|    Root CA 証明書          |  自己署名 / OS・ブラウザに内蔵
|    (例: DigiCert Root)    |
+---------------------------+
          |
          | 署名
          v
+---------------------------+
|    中間 CA 証明書          |  Root CA が署名
|    (例: DigiCert SHA2)    |
+---------------------------+
          |
          | 署名
          v
+---------------------------+
|    サーバ証明書            |  中間 CA が署名
|    (例: *.example.com)    |  サーバが提示
+---------------------------+
```

### 証明書の中身を確認

```bash
# 証明書の内容をデコード
openssl x509 -in server.crt -text -noout

# 出力例:
# Certificate:
#     Data:
#         Version: 3 (0x2)
#         Serial Number: 04:00:00:00:00:01:2f:...
#         Signature Algorithm: sha256WithRSAEncryption
#         Issuer: C=US, O=DigiCert Inc, CN=DigiCert SHA2 ...
#         Validity:
#             Not Before: Jan  1 00:00:00 2025 GMT
#             Not After : Dec 31 23:59:59 2025 GMT
#         Subject: CN=*.example.com
#         Subject Public Key Info:
#             Public Key Algorithm: rsaEncryption
#             RSA Public-Key: (2048 bit)

# 証明書チェーンの検証
openssl verify -CAfile ca-bundle.crt server.crt
```

### X.509 証明書の主要フィールド

```
+-----------------------------------------------------+
|  X.509 Certificate                                   |
|-----------------------------------------------------|
|  Version:             3 (v3)                         |
|  Serial Number:       一意の識別子                     |
|  Signature Algorithm: sha256WithRSAEncryption        |
|  Issuer:              発行者 (CA) の DN               |
|  Validity:            有効期間 (Not Before/After)     |
|  Subject:             所有者の DN                     |
|  Public Key:          公開鍵                          |
|  Extensions:                                         |
|    - Subject Alt Name (SAN): ドメイン一覧             |
|    - Key Usage: digitalSignature, keyEncipherment    |
|    - Basic Constraints: CA:FALSE                     |
|  Signature:           CA の電子署名                   |
+-----------------------------------------------------+
```

---

## 4. Let's Encrypt による自動化

### ACME プロトコルの仕組み

```
クライアント (certbot)              Let's Encrypt CA
     |                                      |
     |--- アカウント登録 ----------------->  |
     |                                      |
     |--- 証明書発行リクエスト ----------->  |
     |    (ドメイン: example.com)            |
     |                                      |
     |  <--- チャレンジ発行 --------------  |
     |       (HTTP-01 or DNS-01)            |
     |                                      |
     |--- チャレンジ応答 ----------------->  |
     |    (.well-known/acme-challenge/...)   |
     |                                      |
     |  <--- 検証完了 --------------------  |
     |  <--- 証明書発行 ------------------  |
     |                                      |
```

### certbot による証明書取得

```bash
# Nginx 用に証明書を取得・設定
sudo certbot --nginx -d example.com -d www.example.com

# DNS-01 チャレンジでワイルドカード証明書を取得
sudo certbot certonly --manual --preferred-challenges dns \
  -d "*.example.com" -d "example.com"

# 自動更新のテスト
sudo certbot renew --dry-run

# crontab による自動更新
# 0 3 * * * certbot renew --quiet --post-hook "systemctl reload nginx"
```

### チャレンジ方式の比較

| 方式 | 用途 | 自動化 | ワイルドカード |
|------|------|--------|--------------|
| HTTP-01 | Web サーバ | 容易 | 不可 |
| DNS-01 | 任意 | DNS API 必要 | 可能 |
| TLS-ALPN-01 | 443 ポートのみ | 中程度 | 不可 |

---

## 5. mTLS (Mutual TLS)

### 通常の TLS と mTLS の違い

```
通常の TLS:
  Client ---- サーバ証明書を検証 ---> Server
  (クライアントは認証されない)

mTLS:
  Client ---- サーバ証明書を検証 ---> Server
  Client <--- クライアント証明書を検証 --- Server
  (双方が認証される)
```

### mTLS の設定例 (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    # サーバ証明書
    ssl_certificate     /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    # クライアント証明書の検証
    ssl_client_certificate /etc/nginx/ssl/ca.crt;
    ssl_verify_client on;
    ssl_verify_depth 2;

    location / {
        # クライアント証明書の情報をバックエンドに転送
        proxy_set_header X-Client-DN $ssl_client_s_dn;
        proxy_set_header X-Client-Verify $ssl_client_verify;
        proxy_pass http://backend;
    }
}
```

### Go でのクライアント証明書生成

```go
package main

import (
    "crypto/ecdsa"
    "crypto/elliptic"
    "crypto/rand"
    "crypto/x509"
    "crypto/x509/pkix"
    "encoding/pem"
    "math/big"
    "os"
    "time"
)

func generateClientCert(caCert *x509.Certificate, caKey *ecdsa.PrivateKey) error {
    // クライアント鍵ペア生成
    clientKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

    // 証明書テンプレート
    template := &x509.Certificate{
        SerialNumber: big.NewInt(2),
        Subject: pkix.Name{
            Organization: []string{"MyOrg"},
            CommonName:   "client-service-a",
        },
        NotBefore:             time.Now(),
        NotAfter:              time.Now().Add(365 * 24 * time.Hour),
        KeyUsage:              x509.KeyUsageDigitalSignature,
        ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
        BasicConstraintsValid: true,
    }

    // CA で署名
    certDER, _ := x509.CreateCertificate(
        rand.Reader, template, caCert, &clientKey.PublicKey, caKey,
    )

    // PEM 書き出し
    certFile, _ := os.Create("client.crt")
    pem.Encode(certFile, &pem.Block{Type: "CERTIFICATE", Bytes: certDER})
    certFile.Close()

    return nil
}
```

---

## 6. アンチパターン

### アンチパターン 1: 証明書検証の無効化

```python
# NG: 本番環境で証明書検証を無効化
import requests
response = requests.get("https://api.example.com", verify=False)

# OK: 正しい CA バンドルを指定
response = requests.get("https://api.example.com", verify="/path/to/ca-bundle.crt")
```

**なぜ危険か**: 中間者攻撃 (MITM) が可能になり、通信内容の盗聴・改竄のリスクがある。開発環境でも自己署名 CA を作成して正しく検証すべきである。

### アンチパターン 2: 古い TLS バージョンの許可

```nginx
# NG: TLS 1.0/1.1 を許可
ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;

# OK: TLS 1.2 以上のみ許可
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
ssl_prefer_server_ciphers on;
```

**なぜ危険か**: TLS 1.0/1.1 には既知の脆弱性 (BEAST, POODLE) があり、PCI DSS でも使用が禁止されている。

### アンチパターン 3: 秘密鍵のハードコーディング

```python
# NG: ソースコードに秘密鍵を埋め込む
PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHQCAQEEIBkg4LVWM...
-----END EC PRIVATE KEY-----"""

# OK: 環境変数またはシークレットマネージャから取得
import os
key_path = os.environ["TLS_KEY_PATH"]
with open(key_path) as f:
    private_key = f.read()
```

---

## 7. FAQ

### Q1. TLS 1.3 の 0-RTT は安全か?

0-RTT (Early Data) は再接続時のレイテンシを削減するが、リプレイ攻撃のリスクがある。べき等でない操作 (POST による状態変更など) には使用すべきでない。Nginx では `ssl_early_data on;` で有効化し、バックエンドで `Early-Data: 1` ヘッダを確認して保護する。

### Q2. 証明書の有効期間はどのくらいが適切か?

CA/Browser Forum の規定により、公開証明書の最大有効期間は 398 日 (約13ヶ月) である。Let's Encrypt は 90 日で発行し、自動更新を前提とする設計になっている。短い有効期間は鍵の危殆化リスクを低減する。

### Q3. 自己署名証明書を使ってよい場面は?

開発環境、内部マイクロサービス間の mTLS、テスト環境に限定すべきである。公開サービスでは必ず信頼された CA の証明書を使用する。自己署名でも CA を作成し、チェーンを正しく構成する運用が望ましい。

---

## まとめ

| 項目 | 要点 |
|------|------|
| TLS バージョン | TLS 1.3 を推奨、最低でも TLS 1.2 |
| ハンドシェイク | TLS 1.3 は 1-RTT で完了し前方秘匿性を必須化 |
| 証明書チェーン | Root CA → 中間 CA → サーバ証明書の信頼の連鎖 |
| Let's Encrypt | ACME プロトコルで証明書の取得・更新を自動化 |
| mTLS | クライアント証明書による双方向認証でゼロトラスト実現 |
| 証明書管理 | 自動更新必須、秘密鍵はシークレットマネージャで保護 |
| 監視 | 証明書の有効期限を常時監視し期限切れを防止 |

---

## 次に読むべきガイド

- [鍵管理](./02-key-management.md) — 暗号鍵のライフサイクルと安全な保管方法
- [ネットワークセキュリティ基礎](../03-network-security/00-network-security-basics.md) — ファイアウォール・IDS/IPS による多層防御
- [APIセキュリティ](../03-network-security/02-api-security.md) — TLS の上に構築する API 保護

---

## 参考文献

1. **RFC 8446 — The Transport Layer Security (TLS) Protocol Version 1.3** (2018) — https://datatracker.ietf.org/doc/html/rfc8446
2. **Mozilla SSL Configuration Generator** — https://ssl-config.mozilla.org/ — サーバソフトウェア別の推奨 TLS 設定
3. **Let's Encrypt Documentation** — https://letsencrypt.org/docs/ — ACME プロトコルと証明書自動化の公式ガイド
4. **Qualys SSL Labs — SSL/TLS Best Practices** — https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices
