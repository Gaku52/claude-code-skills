# TLS/証明書

> TLS ハンドシェイクから証明書チェーン、Let's Encrypt による自動化、mTLS まで、安全な通信の基盤技術を体系的に学ぶ

## この章で学ぶこと

1. **TLS ハンドシェイクの仕組み** — クライアントとサーバ間で暗号化通信が確立されるまでの全ステップ
2. **証明書チェーンと PKI** — ルート CA から中間 CA、サーバ証明書に至る信頼の連鎖
3. **実運用での証明書管理** — Let's Encrypt による自動化と mTLS による双方向認証

### 前提知識

- 公開鍵暗号の基本（RSA、楕円曲線暗号、電子署名）
- TCP/IP ネットワーキング（3ウェイハンドシェイク、ポート）
- HTTP プロトコルの基本動作

### 関連ガイド

- [鍵管理](./02-key-management.md) — 暗号鍵のライフサイクルと安全な保管方法
- [ネットワークセキュリティ基礎](../03-network-security/00-network-security-basics.md) — ファイアウォール・IDS/IPS による多層防御
- [API セキュリティ](../03-network-security/02-api-security.md) — TLS の上に構築する API 保護
- [DNS セキュリティ](../03-network-security/01-dns-security.md) — DANE/TLSA による DNS ベースの証明書検証

---

## 1. TLS の全体像

### TLS とは何か

TLS (Transport Layer Security) はトランスポート層の上で動作する暗号化プロトコルである。SSL の後継として策定され、現在の推奨バージョンは TLS 1.3 (RFC 8446) である。

TLS は以下の 3 つのセキュリティ特性を提供する:
- **機密性** (Confidentiality): 通信内容の暗号化
- **完全性** (Integrity): データ改竄の検知
- **認証** (Authentication): 通信相手の身元確認

```
+-------------------+
|   Application     |  HTTP, SMTP, IMAP, etc.
+-------------------+
|       TLS         |  暗号化・認証・完全性
|  +-------------+  |
|  | Record      |  |  データの分割・暗号化・MAC
|  +-------------+  |
|  | Handshake   |  |  鍵交換・認証・パラメータ交渉
|  +-------------+  |
|  | Alert       |  |  エラー通知・接続終了
|  +-------------+  |
|  | Change Cipher|  |  暗号仕様の切り替え (TLS 1.2)
|  +-------------+  |
+-------------------+
|       TCP         |  信頼性のある配送
+-------------------+
|       IP          |  ルーティング
+-------------------+
```

### TLS のバージョン比較

| バージョン | 状態 | ハンドシェイク RTT | 主な特徴 | 廃止理由/脆弱性 |
|-----------|------|-------------------|---------|---------------|
| SSL 2.0 | 廃止 (2011) | 2-RTT | - | 根本的な設計欠陥 |
| SSL 3.0 | 廃止 (2015) | 2-RTT | - | POODLE 脆弱性 |
| TLS 1.0 | 廃止 (2020) | 2-RTT | SSL 3.0 の改良 | BEAST 脆弱性 |
| TLS 1.1 | 廃止 (2020) | 2-RTT | CBC 改善 | Lucky13、モダン暗号スイート未対応 |
| TLS 1.2 | 現役 | 2-RTT | AEAD 暗号スイート | GCM 推奨、CBC は非推奨 |
| TLS 1.3 | 推奨 | 1-RTT (0-RTT可) | ハンドシェイク簡素化 | 前方秘匿性必須 |

### TLS 1.3 で廃止されたもの

```
+------------------------------------------------------------------+
|  TLS 1.3 で廃止された機能・アルゴリズム                              |
|------------------------------------------------------------------|
|                                                                    |
|  [鍵交換]                                                          |
|  - RSA 鍵交換 (静的RSA) → ECDHE のみに                            |
|    理由: 前方秘匿性なし。サーバ秘密鍵の漏洩で過去の通信全て復号可能   |
|                                                                    |
|  [暗号アルゴリズム]                                                 |
|  - RC4 → 廃止 (統計的バイアス)                                     |
|  - 3DES → 廃止 (64bitブロック、Sweet32攻撃)                        |
|  - CBC モード → 廃止 (パディングオラクル)                           |
|  - DES → 廃止 (鍵長不足)                                          |
|                                                                    |
|  [ハッシュ]                                                        |
|  - MD5 → 廃止 (衝突攻撃)                                          |
|  - SHA-1 → 廃止 (衝突攻撃)                                        |
|                                                                    |
|  [プロトコル機能]                                                   |
|  - 圧縮 → 廃止 (CRIME/BREACH 攻撃)                                |
|  - 再ネゴシエーション → 廃止 (三者間攻撃)                           |
|  - ChangeCipherSpec メッセージ → 廃止                              |
|  - カスタム暗号スイート定義 → 5 スイートのみに限定                   |
+------------------------------------------------------------------+
```

---

## 2. TLS 1.3 ハンドシェイク

### ハンドシェイクの流れ

```
Client                                    Server
  |                                          |
  |--- ClientHello ----------------------->  |
  |    + supported_versions: [TLS 1.3]      |
  |    + key_share: (x25519 公開鍵)          |
  |    + signature_algorithms: [Ed25519,     |
  |      ECDSA-P256, RSA-PSS]               |
  |    + psk_key_exchange_modes             |
  |    + supported_groups: [x25519, P-256]  |
  |                                          |
  |  <--- ServerHello ---------------------  |
  |       + key_share: (x25519 公開鍵)       |
  |       [ここからサーバ→クライアント暗号化]   |
  |  <--- EncryptedExtensions -------------  |
  |  <--- Certificate ---------------------  |
  |       (サーバ証明書チェーン)               |
  |  <--- CertificateVerify --------------  |
  |       (ハンドシェイクメッセージの署名)      |
  |  <--- Finished -----------------------  |
  |       (ハンドシェイクの MAC)              |
  |                                          |
  |--- Finished ------------------------->   |
  |    [ここからクライアント→サーバ暗号化]      |
  |                                          |
  |========= 暗号化通信開始 ================|
```

### TLS 1.3 の暗号スイート

```
TLS 1.3 で使用可能な 5 つの暗号スイート:

  TLS_AES_256_GCM_SHA384        最も推奨
  TLS_AES_128_GCM_SHA256        推奨
  TLS_CHACHA20_POLY1305_SHA256  モバイル/ARM に最適
  TLS_AES_128_CCM_SHA256        IoT 向け
  TLS_AES_128_CCM_8_SHA256      IoT (短い認証タグ)

暗号スイートの構成要素 (TLS 1.3):
  [AEAD暗号] + [ハッシュ]

  鍵交換: ECDHE (x25519 または P-256) — 暗号スイート外で選択
  署名: Ed25519, ECDSA, RSA-PSS — 暗号スイート外で選択

注: TLS 1.2 では暗号スイートが鍵交換+認証+暗号+MACの4要素を
含んでいたが、TLS 1.3 では分離された。
```

### TLS 1.2 と 1.3 のハンドシェイク比較

| 項目 | TLS 1.2 | TLS 1.3 |
|------|---------|---------|
| ラウンドトリップ | 2-RTT | 1-RTT |
| 0-RTT 再接続 | 不可 | 可能 (Early Data) |
| 鍵交換 | RSA / ECDHE | ECDHE のみ |
| 暗号スイート数 | 数十個 | 5個 |
| ハンドシェイク暗号化 | なし (平文) | ServerHello 以降暗号化 |
| 前方秘匿性 | オプション | 必須 |
| 圧縮 | サポート | 廃止 |
| 再ネゴシエーション | あり | なし (KeyUpdate で代替) |
| セッション再開 | セッションID/チケット | PSK ベース |
| ServerHello の内容 | 平文 | 暗号化 |

### OpenSSL でハンドシェイクを確認

```bash
# コード例1: TLS 接続の詳細確認

# TLS 1.3 ハンドシェイクの詳細を表示
openssl s_client -connect example.com:443 -tls1_3 -msg

# 証明書チェーンを確認
openssl s_client -connect example.com:443 -showcerts

# 使用される暗号スイートの確認
openssl s_client -connect example.com:443 -cipher 'TLS_AES_256_GCM_SHA384'

# サーバがサポートする暗号スイート一覧
nmap --script ssl-enum-ciphers -p 443 example.com

# testssl.sh による包括的なTLSテスト
testssl.sh https://example.com

# sslyze による詳細分析
sslyze --regular example.com

# curl でTLSバージョンとプロトコル情報を表示
curl -vI https://example.com 2>&1 | grep -E "SSL|TLS|subject|issuer"
```

### 0-RTT (Early Data) の仕組みと注意点

```
0-RTT の流れ (PSK ベースの再接続):

  Client                                    Server
    |                                          |
    |--- ClientHello ----------------------->  |
    |    + pre_shared_key (前回のセッションから) |
    |    + early_data (アプリケーションデータ)   |
    |                                          |
    |  <--- ServerHello ---------------------  |
    |  <--- Finished -----------------------  |
    |                                          |
    |--- Finished ------------------------->   |
    |                                          |
    | 0-RTT で送ったデータは既にサーバで処理済み |

  利点: 再接続時のレイテンシがゼロ
  リスク: リプレイ攻撃が可能

  対策:
  +-- べき等でない操作 (POST, DELETE) には使用しない
  +-- Anti-Replay 機構をサーバ側で実装
  +-- Nginx: ssl_early_data on; + Early-Data ヘッダの確認
  +-- サーバ側で同一 PSK の 0-RTT を一度だけ受け入れる
```

### 前方秘匿性 (Forward Secrecy) の仕組み

```
前方秘匿性 (PFS) なし (静的RSA鍵交換):

  Client                              Server
    |--- RSA暗号化(premaster secret) --> |
    |    サーバの公開鍵で暗号化           |
    |                                    |
  問題: サーバの秘密鍵が将来漏洩した場合、
  過去に記録した通信全てを復号できる

前方秘匿性あり (ECDHE 鍵交換):

  Client                              Server
    |--- ECDHE公開鍵 ------------------> |
    |<--- ECDHE公開鍵 ------------------|
    |    両者が独立に共有秘密を計算        |
    |    一時的な鍵ペアは破棄             |
    |                                    |
  利点: 各セッションで異なる鍵ペアを使用するため、
  サーバの秘密鍵が漏洩しても過去の通信は復号不可能

  TLS 1.3 では PFS が必須 (ECDHE のみ)
```

---

## 3. 証明書チェーンと PKI

### 証明書チェーンの構造

```
+---------------------------+
|    Root CA 証明書          |  自己署名 / OS・ブラウザに内蔵
|    (例: DigiCert Root)    |  有効期間: 20-30年
|    自分自身の秘密鍵で署名   |
+---------------------------+
          |
          | 署名
          v
+---------------------------+
|    中間 CA 証明書          |  Root CA が署名
|    (例: DigiCert SHA2)    |  有効期間: 5-10年
|    エンドエンティティに署名  |  Root CA のオフライン保護
+---------------------------+
          |
          | 署名
          v
+---------------------------+
|    サーバ証明書            |  中間 CA が署名
|    (例: *.example.com)    |  有効期間: 最大398日
|    サーバが TLS で提示     |  (CA/B Forum 規定)
+---------------------------+

なぜ中間 CA が必要か:
1. Root CA の秘密鍵をオフラインで保護（HSM に格納）
2. Root CA の侵害時の被害範囲を限定
3. 証明書失効時に中間 CA のみ失効すれば済む
4. 異なるポリシー/用途ごとに中間 CA を分離
```

### 証明書の中身を確認

```bash
# コード例2: 証明書の詳細確認コマンド

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
#             Public Key Algorithm: id-ecPublicKey
#             Public-Key: (256 bit)
#         X509v3 extensions:
#             X509v3 Subject Alternative Name:
#                 DNS:*.example.com, DNS:example.com
#             X509v3 Key Usage: critical
#                 Digital Signature
#             X509v3 Extended Key Usage:
#                 TLS Web Server Authentication
#             Authority Information Access:
#                 OCSP - URI:http://ocsp.digicert.com
#                 CA Issuers - URI:http://cacerts.digicert.com/...
#             X509v3 CRL Distribution Points:
#                 URI:http://crl3.digicert.com/...

# 証明書チェーンの検証
openssl verify -CAfile ca-bundle.crt server.crt

# リモートサーバの証明書チェーン全体を表示
openssl s_client -connect example.com:443 -showcerts 2>/dev/null | \
  openssl x509 -noout -subject -issuer -dates

# 証明書のフィンガープリント
openssl x509 -in server.crt -fingerprint -sha256 -noout

# CSR (証明書署名要求) の確認
openssl req -in server.csr -text -noout

# 秘密鍵と証明書の一致確認
openssl x509 -in server.crt -modulus -noout | openssl md5
openssl rsa -in server.key -modulus -noout | openssl md5
# 同じハッシュ値なら一致
```

### X.509 証明書の主要フィールド

```
+-----------------------------------------------------+
|  X.509 v3 Certificate                                |
|-----------------------------------------------------|
|  Version:             3 (v3)                         |
|  Serial Number:       一意の識別子 (CA内で一意)        |
|  Signature Algorithm: sha256WithRSAEncryption        |
|                       or ecdsa-with-SHA256           |
|  Issuer:              発行者 (CA) の Distinguished Name|
|  Validity:                                           |
|    Not Before:        発行日時                        |
|    Not After:         有効期限                        |
|  Subject:             所有者の Distinguished Name     |
|  Public Key:          公開鍵 (RSA 2048+ / ECDSA P-256)|
|  Extensions:                                         |
|    - Subject Alt Name (SAN): ドメイン一覧 (必須)      |
|    - Key Usage: digitalSignature, keyEncipherment    |
|    - Extended Key Usage: serverAuth, clientAuth      |
|    - Basic Constraints: CA:FALSE / CA:TRUE           |
|    - Authority Key Identifier: 発行者鍵の識別子       |
|    - Subject Key Identifier: 証明書鍵の識別子         |
|    - CRL Distribution Points: CRL の取得先           |
|    - Authority Info Access: OCSP レスポンダ URL       |
|    - Certificate Policies: 発行ポリシー OID          |
|    - SCT List: Certificate Transparency ログ          |
|  Signature:           CA の電子署名                   |
+-----------------------------------------------------+
```

### 証明書の種類と検証レベル

| 種類 | 略称 | 検証内容 | 発行時間 | 用途 | ブラウザ表示 |
|------|------|---------|---------|------|------------|
| ドメイン検証 | DV | ドメイン制御の確認のみ | 数分 | 一般的なWebサイト | 鍵マーク |
| 組織検証 | OV | + 組織の実在確認 | 1-3日 | 企業サイト | 鍵マーク + 組織名 |
| 拡張検証 | EV | + 厳格な組織審査 | 1-4週間 | 金融・EC サイト | 鍵マーク + 組織名 |
| ワイルドカード | WC | *.example.com | DV/OV に準ずる | 多数のサブドメイン | 同上 |
| マルチドメイン | SAN | 複数ドメインを1枚に | DV/OV に準ずる | 複数サイトの統合 | 同上 |

---

## 4. 証明書の失効チェック

### OCSP と CRL の比較

```
+------------------------------------------------------------------+
|  証明書失効チェックの仕組み                                          |
|------------------------------------------------------------------|
|                                                                    |
|  [CRL (Certificate Revocation List)]                               |
|  CA が定期的に失効証明書リストを公開                                  |
|  クライアントがリストをダウンロードして検証                            |
|                                                                    |
|  問題点:                                                           |
|  - リストが巨大化 (数MB)                                            |
|  - 更新頻度に依存 (リアルタイム性なし)                               |
|  - ダウンロード失敗時のフォールバック (fail-open)                     |
|                                                                    |
|  [OCSP (Online Certificate Status Protocol)]                       |
|  クライアントが個別の証明書について CA に問い合わせ                    |
|                                                                    |
|  問題点:                                                           |
|  - レイテンシの増加 (CA への往復)                                    |
|  - プライバシー (CA にアクセス先が漏れる)                             |
|  - OCSP レスポンダの障害リスク                                      |
|                                                                    |
|  [OCSP Stapling (推奨)]                                            |
|  サーバが CA から OCSP レスポンスを事前に取得                         |
|  TLS ハンドシェイク時にクライアントに提供                             |
|                                                                    |
|  利点:                                                             |
|  - クライアントが CA に直接問い合わせる必要がない                     |
|  - プライバシーが保護される                                          |
|  - レイテンシの増加なし                                              |
+------------------------------------------------------------------+
```

| 方式 | リアルタイム性 | プライバシー | パフォーマンス | 信頼性 |
|------|------------|------------|-------------|--------|
| CRL | 低 (定期更新) | 高 | 低 (大きなリスト) | 中 |
| OCSP | 高 | 低 (CA に漏れる) | 中 (往復あり) | CA依存 |
| OCSP Stapling | 高 | 高 | 高 | サーバ依存 |
| CRLite (実験的) | 高 | 高 | 高 | ブラウザ内蔵 |

```nginx
# コード例3: Nginx での OCSP Stapling 設定
server {
    listen 443 ssl;
    server_name example.com;

    ssl_certificate     /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    # OCSP Stapling の有効化
    ssl_stapling on;
    ssl_stapling_verify on;

    # OCSP レスポンダへの接続に使用する CA 証明書
    ssl_trusted_certificate /etc/nginx/ssl/ca-chain.crt;

    # OCSP レスポンスの DNS 解決
    resolver 1.1.1.1 8.8.8.8 valid=300s;
    resolver_timeout 5s;
}
```

### Certificate Transparency (CT)

```
Certificate Transparency の仕組み:

  CA が証明書を発行する際:
  1. CA が証明書を CT ログサーバに提出
  2. CT ログサーバが SCT (Signed Certificate Timestamp) を返す
  3. SCT が証明書に埋め込まれる
  4. ブラウザが SCT を検証し、CT ログに記録されていることを確認

  目的:
  - 不正な証明書の発行を検知 (例: 2011年 DigiNotar 事件)
  - ドメイン所有者が自分のドメインの証明書を監視
  - 透明性による CA の信頼性向上

  Chrome の要件:
  - 2018年4月以降、全ての新規証明書は CT ログへの登録が必須
  - 最低 2 つの CT ログからの SCT が必要

  監視ツール:
  - crt.sh: https://crt.sh/?q=example.com
  - Google CT Dashboard
  - certspotter (CLI ツール)
```

```bash
# コード例4: CT ログの監視
# crt.sh で特定ドメインの全証明書を確認
curl -s "https://crt.sh/?q=%25.example.com&output=json" | \
  python3 -m json.tool | head -50

# certspotter でリアルタイム監視
# certspotter watch example.com
```

---

## 5. Let's Encrypt による自動化

### ACME プロトコルの仕組み

```
ACME (Automatic Certificate Management Environment) - RFC 8555:

クライアント (certbot)              Let's Encrypt CA
     |                                      |
     |--- (1) アカウント登録 ------------>  |
     |    JWK公開鍵 + 利用規約同意          |
     |                                      |
     |  <--- アカウント作成完了 ----------  |
     |                                      |
     |--- (2) 証明書発行リクエスト ------> |
     |    (ドメイン: example.com)            |
     |                                      |
     |  <--- (3) チャレンジ発行 ----------  |
     |       (HTTP-01 or DNS-01)            |
     |                                      |
     |--- (4) チャレンジ応答 ------------>  |
     |    HTTP-01: /.well-known/acme-       |
     |    challenge/{token} にトークン配置   |
     |    DNS-01: _acme-challenge TXT       |
     |    レコードにトークン設定              |
     |                                      |
     |  <--- (5) 検証実行 ----------------  |
     |    CA が HTTP/DNS でトークンを確認     |
     |                                      |
     |  <--- (6) 検証完了 ----------------  |
     |                                      |
     |--- (7) CSR 送信 ------------------>  |
     |                                      |
     |  <--- (8) 証明書発行 --------------  |
     |    (チェーン付き証明書)               |
```

### certbot による証明書取得

```bash
# コード例5: certbot の各種使用方法

# Nginx 用に証明書を取得・設定（最も簡単）
sudo certbot --nginx -d example.com -d www.example.com

# Apache 用
sudo certbot --apache -d example.com

# スタンドアロンモード（既存のWebサーバを停止して使用）
sudo certbot certonly --standalone -d example.com

# Webroot モード（既存のWebサーバを停止せずに使用）
sudo certbot certonly --webroot -w /var/www/html -d example.com

# DNS-01 チャレンジでワイルドカード証明書を取得
sudo certbot certonly --manual --preferred-challenges dns \
  -d "*.example.com" -d "example.com"

# DNS-01 + Cloudflare プラグインで自動化
sudo certbot certonly \
  --dns-cloudflare \
  --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini \
  -d "*.example.com" -d "example.com"

# 自動更新のテスト
sudo certbot renew --dry-run

# 自動更新の設定
# systemd timer (推奨)
# /etc/systemd/system/certbot-renewal.timer
# [Timer]
# OnCalendar=*-*-* 03:00:00
# RandomizedDelaySec=3600

# crontab による自動更新（代替）
# 0 3 * * * certbot renew --quiet --post-hook "systemctl reload nginx"

# 証明書の情報確認
sudo certbot certificates

# 証明書の手動失効
sudo certbot revoke --cert-path /etc/letsencrypt/live/example.com/cert.pem
```

### チャレンジ方式の比較

| 方式 | 用途 | 自動化 | ワイルドカード | ポート要件 | DNS 変更 |
|------|------|--------|--------------|-----------|---------|
| HTTP-01 | Web サーバ | 容易 | 不可 | 80 | 不要 |
| DNS-01 | 任意 | DNS API 必要 | 可能 | 不要 | 必要 |
| TLS-ALPN-01 | 443 ポートのみ | 中程度 | 不可 | 443 | 不要 |

### ACME クライアントの比較

| クライアント | 言語 | 特徴 | 用途 |
|------------|------|------|------|
| certbot | Python | 公式、最も普及 | 汎用 |
| acme.sh | Shell | 軽量、依存関係なし | シンプルな環境 |
| lego | Go | シングルバイナリ | Docker/CI |
| cert-manager | Go | K8s ネイティブ | Kubernetes |
| Caddy (内蔵) | Go | 自動TLS | Webサーバ |
| Traefik (内蔵) | Go | 自動TLS | リバースプロキシ |

---

## 6. mTLS (Mutual TLS)

### 通常の TLS と mTLS の違い

```
通常の TLS (一方向認証):
  Client ---- サーバ証明書を検証 ---> Server
  (クライアントは認証されない)
  用途: 一般的な HTTPS

mTLS (双方向認証):
  Client ---- サーバ証明書を検証 ---> Server
  Client <--- クライアント証明書を検証 --- Server
  (双方が認証される)
  用途: マイクロサービス間通信、ゼロトラスト、API 認証
```

### mTLS ハンドシェイクの詳細

```
Client                                    Server
  |                                          |
  |--- ClientHello ----------------------->  |
  |                                          |
  |  <--- ServerHello ---------------------  |
  |  <--- EncryptedExtensions -------------  |
  |  <--- CertificateRequest -------------  |  ← mTLS: クライアント証明書を要求
  |       (許可される CA の一覧)              |
  |  <--- Certificate ---------------------  |
  |  <--- CertificateVerify --------------  |
  |  <--- Finished -----------------------  |
  |                                          |
  |--- Certificate ----------------------->  |  ← mTLS: クライアント証明書を提示
  |--- CertificateVerify ---------------->   |  ← mTLS: クライアントの署名
  |--- Finished ------------------------->   |
  |                                          |
  |========= 双方向認証された暗号化通信 ======|
```

### mTLS の設定例 (Nginx)

```nginx
# コード例6: Nginx での mTLS 設定
server {
    listen 443 ssl;
    server_name api.example.com;

    # サーバ証明書
    ssl_certificate     /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    # TLS プロトコルと暗号スイートの設定
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers on;

    # クライアント証明書の検証
    ssl_client_certificate /etc/nginx/ssl/ca.crt;  # 信頼するCA証明書
    ssl_verify_client on;       # 必須 (optional にすると選択的になる)
    ssl_verify_depth 2;         # チェーンの最大深度

    # CRL (証明書失効リスト) の設定
    ssl_crl /etc/nginx/ssl/ca.crl;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    location / {
        # クライアント証明書の情報をバックエンドに転送
        proxy_set_header X-Client-DN $ssl_client_s_dn;
        proxy_set_header X-Client-Serial $ssl_client_serial;
        proxy_set_header X-Client-Verify $ssl_client_verify;
        proxy_set_header X-Client-Fingerprint $ssl_client_fingerprint;
        proxy_pass http://backend;
    }
}
```

### Go でのクライアント証明書生成と mTLS クライアント

```go
// コード例7: Go での mTLS 実装（サーバ + クライアント）
package main

import (
    "crypto/ecdsa"
    "crypto/elliptic"
    "crypto/rand"
    "crypto/tls"
    "crypto/x509"
    "crypto/x509/pkix"
    "encoding/pem"
    "fmt"
    "math/big"
    "net/http"
    "os"
    "time"
)

// ===== CA と証明書の生成 =====

func generateCA() (*x509.Certificate, *ecdsa.PrivateKey, error) {
    caKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

    template := &x509.Certificate{
        SerialNumber: big.NewInt(1),
        Subject: pkix.Name{
            Organization: []string{"MyOrg Internal CA"},
            CommonName:   "MyOrg Root CA",
        },
        NotBefore:             time.Now(),
        NotAfter:              time.Now().Add(10 * 365 * 24 * time.Hour),
        KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
        BasicConstraintsValid: true,
        IsCA:                  true,
        MaxPathLen:            1,
    }

    certDER, _ := x509.CreateCertificate(
        rand.Reader, template, template, &caKey.PublicKey, caKey,
    )
    cert, _ := x509.ParseCertificate(certDER)
    return cert, caKey, nil
}

func generateClientCert(caCert *x509.Certificate, caKey *ecdsa.PrivateKey,
    commonName string) error {
    // クライアント鍵ペア生成
    clientKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

    // 証明書テンプレート
    template := &x509.Certificate{
        SerialNumber: big.NewInt(2),
        Subject: pkix.Name{
            Organization: []string{"MyOrg"},
            CommonName:   commonName,
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
    certFile, _ := os.Create(commonName + ".crt")
    pem.Encode(certFile, &pem.Block{Type: "CERTIFICATE", Bytes: certDER})
    certFile.Close()

    keyDER, _ := x509.MarshalECPrivateKey(clientKey)
    keyFile, _ := os.Create(commonName + ".key")
    pem.Encode(keyFile, &pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
    keyFile.Close()

    return nil
}

// ===== mTLS サーバ =====

func startMTLSServer(caCertPool *x509.CertPool) {
    tlsConfig := &tls.Config{
        ClientAuth: tls.RequireAndVerifyClientCert,
        ClientCAs:  caCertPool,
        MinVersion: tls.VersionTLS13,
    }

    server := &http.Server{
        Addr:      ":8443",
        TLSConfig: tlsConfig,
    }

    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        // クライアント証明書の情報を取得
        if len(r.TLS.PeerCertificates) > 0 {
            clientCert := r.TLS.PeerCertificates[0]
            fmt.Fprintf(w, "Hello, %s (verified by mTLS)\n",
                clientCert.Subject.CommonName)
        }
    })

    server.ListenAndServeTLS("server.crt", "server.key")
}

// ===== mTLS クライアント =====

func createMTLSClient(clientCert, clientKey, caCert string) (*http.Client, error) {
    cert, err := tls.LoadX509KeyPair(clientCert, clientKey)
    if err != nil {
        return nil, err
    }

    caCertPEM, _ := os.ReadFile(caCert)
    caCertPool := x509.NewCertPool()
    caCertPool.AppendCertsFromPEM(caCertPEM)

    tlsConfig := &tls.Config{
        Certificates: []tls.Certificate{cert},
        RootCAs:      caCertPool,
        MinVersion:   tls.VersionTLS13,
    }

    return &http.Client{
        Transport: &http.Transport{TLSClientConfig: tlsConfig},
        Timeout:   30 * time.Second,
    }, nil
}
```

### Python での mTLS クライアント

```python
# コード例8: Python mTLS クライアント
import httpx
import ssl

def create_mtls_client(
    client_cert: str,
    client_key: str,
    ca_cert: str,
) -> httpx.Client:
    """mTLS 対応の HTTP クライアントを作成"""
    ssl_context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=ca_cert,
    )
    ssl_context.load_cert_chain(
        certfile=client_cert,
        keyfile=client_key,
    )
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

    return httpx.Client(
        verify=ssl_context,
        timeout=30.0,
    )

# 使用例
client = create_mtls_client(
    client_cert="client.crt",
    client_key="client.key",
    ca_cert="ca.crt",
)
response = client.get("https://api.example.com/data")
print(response.json())
```

---

## 7. Nginx / Apache の TLS 設定ベストプラクティス

### Mozilla SSL Configuration Generator 準拠の設定

```nginx
# コード例9: Nginx TLS 設定（Modern プロファイル）
server {
    listen 443 ssl http2;
    server_name example.com;

    # 証明書
    ssl_certificate     /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;

    # プロトコル（TLS 1.3 のみ — Modern）
    ssl_protocols TLSv1.3;
    # TLS 1.2 も含める場合（Intermediate）:
    # ssl_protocols TLSv1.2 TLSv1.3;

    # 暗号スイート
    # TLS 1.3 の場合は ssl_ciphers の設定は不要（TLS 1.3 のスイートは固定）
    # TLS 1.2 を含める場合:
    # ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;  # TLS 1.3 ではクライアント選択を推奨

    # DH パラメータ (TLS 1.2 使用時)
    # openssl dhparam -out /etc/nginx/ssl/dhparam.pem 2048
    # ssl_dhparam /etc/nginx/ssl/dhparam.pem;

    # セッションキャッシュ
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;  # PFS を完全に保証するため

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/nginx/ssl/chain.pem;
    resolver 1.1.1.1 8.8.8.8 valid=300s;

    # 0-RTT (TLS 1.3)
    ssl_early_data on;

    # セキュリティヘッダ
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;

    # HTTP → HTTPS リダイレクト（別の server ブロックで）
}

server {
    listen 80;
    server_name example.com;
    return 301 https://$host$request_uri;
}
```

---

## 8. 証明書の自動管理と監視

### 証明書有効期限の監視

```python
# コード例10: 証明書有効期限監視スクリプト
import ssl
import socket
import datetime
import json
from typing import Optional

class CertificateMonitor:
    """TLS 証明書の有効期限を監視"""

    def __init__(self, warning_days: int = 30, critical_days: int = 7):
        self.warning_days = warning_days
        self.critical_days = critical_days

    def check_certificate(self, hostname: str, port: int = 443) -> dict:
        """証明書情報を取得して有効期限をチェック"""
        context = ssl.create_default_context()
        try:
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()

            # 有効期限を解析
            not_after = datetime.datetime.strptime(
                cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
            )
            days_remaining = (not_after - datetime.datetime.utcnow()).days

            # ステータス判定
            if days_remaining < 0:
                status = "EXPIRED"
            elif days_remaining <= self.critical_days:
                status = "CRITICAL"
            elif days_remaining <= self.warning_days:
                status = "WARNING"
            else:
                status = "OK"

            # SAN (Subject Alternative Name) の取得
            san = []
            for entry_type, value in cert.get("subjectAltName", []):
                if entry_type == "DNS":
                    san.append(value)

            return {
                "hostname": hostname,
                "status": status,
                "days_remaining": days_remaining,
                "not_after": not_after.isoformat(),
                "issuer": dict(x[0] for x in cert["issuer"]),
                "subject": dict(x[0] for x in cert["subject"]),
                "san": san,
                "serial_number": cert.get("serialNumber"),
                "version": cert.get("version"),
            }

        except ssl.SSLCertVerificationError as e:
            return {"hostname": hostname, "status": "VERIFICATION_ERROR",
                    "error": str(e)}
        except (socket.timeout, ConnectionRefusedError) as e:
            return {"hostname": hostname, "status": "CONNECTION_ERROR",
                    "error": str(e)}

    def check_multiple(self, hostnames: list[str]) -> list[dict]:
        """複数ホストの証明書を一括チェック"""
        results = []
        for hostname in hostnames:
            result = self.check_certificate(hostname)
            results.append(result)
            if result["status"] in ("CRITICAL", "EXPIRED"):
                print(f"[{result['status']}] {hostname}: "
                      f"{result.get('days_remaining', 'N/A')} days remaining")
        return results


# 使用例
monitor = CertificateMonitor(warning_days=30, critical_days=7)
domains = [
    "example.com",
    "api.example.com",
    "app.example.com",
]
results = monitor.check_multiple(domains)
print(json.dumps(results, indent=2, default=str))
```

### Kubernetes での cert-manager

```yaml
# コード例11: cert-manager による自動証明書管理
# ClusterIssuer (Let's Encrypt)
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-production
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-production-key
    solvers:
      - http01:
          ingress:
            class: nginx
      - dns01:
          cloudflare:
            email: admin@example.com
            apiTokenSecretRef:
              name: cloudflare-api-token
              key: api-token
        selector:
          dnsZones:
            - "example.com"

---
# Certificate リソース
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: example-com-tls
  namespace: default
spec:
  secretName: example-com-tls-secret
  issuerRef:
    name: letsencrypt-production
    kind: ClusterIssuer
  dnsNames:
    - example.com
    - "*.example.com"
  duration: 2160h      # 90日
  renewBefore: 720h    # 30日前に更新

---
# Ingress での使用
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-production"
spec:
  tls:
    - hosts:
        - example.com
      secretName: example-com-tls-secret
  rules:
    - host: example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-service
                port:
                  number: 80
```

---

## 9. エッジケース

### エッジケース 1: SNI (Server Name Indication) と仮想ホスト

TLS ハンドシェイクの ClientHello に含まれる SNI 拡張により、同一 IP アドレスで複数ドメインの証明書を使い分けられる。ただし、SNI は平文で送信されるため、接続先ドメイン名がネットワーク上で可視。TLS 1.3 の Encrypted Client Hello (ECH) でこれを暗号化する提案がある。

### エッジケース 2: 証明書ピンニングの問題

HPKP (HTTP Public Key Pinning) は証明書の公開鍵ハッシュをブラウザにピン留めする仕組みだったが、運用ミスによるサイト到達不能のリスクが高く、Chrome 72 で廃止された。代替として DANE/TLSA レコード（DNSSEC 前提）や Certificate Transparency の監視が推奨される。

### エッジケース 3: ワイルドカード証明書と SAN の制限

ワイルドカード証明書（*.example.com）は 1 レベルのサブドメインのみカバーする。`sub.sub.example.com` はカバーされない。また、SAN に最大何ドメインまで含められるかは CA ごとに異なり（Let's Encrypt は 100 まで）、大量のドメインを持つ場合は複数の証明書が必要。

### エッジケース 4: 時刻同期の影響

証明書の有効期間チェックはクライアントの時刻に依存する。NTP が正しく設定されていないクライアントでは、有効な証明書でも検証に失敗する場合がある。特に IoT デバイスや組み込みシステムで問題になりやすい。

---

## 10. アンチパターン

### アンチパターン 1: 証明書検証の無効化

```python
# NG: 本番環境で証明書検証を無効化
import requests
response = requests.get("https://api.example.com", verify=False)
# WARNING: InsecureRequestWarning が出るが無視される

# OK: 正しい CA バンドルを指定
response = requests.get("https://api.example.com", verify="/path/to/ca-bundle.crt")

# OK: 環境変数で CA バンドルを指定
# export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.crt
```

**なぜ危険か**: 中間者攻撃 (MITM) が可能になり、通信内容の盗聴・改竄のリスクがある。開発環境でも自己署名 CA を作成して正しく検証すべきである。`verify=False` はセキュリティスキャナで自動検出される重大な問題。

### アンチパターン 2: 古い TLS バージョンの許可

```nginx
# NG: TLS 1.0/1.1 を許可
ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;

# OK: TLS 1.2 以上のみ許可
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
ssl_prefer_server_ciphers on;
```

**なぜ危険か**: TLS 1.0/1.1 には既知の脆弱性 (BEAST, POODLE, Lucky13) があり、PCI DSS でも使用が禁止されている。2020年に主要ブラウザが TLS 1.0/1.1 のサポートを終了。

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

# OK: AWS Secrets Manager / HashiCorp Vault から取得
import boto3
client = boto3.client("secretsmanager")
secret = client.get_secret_value(SecretId="tls/server-key")
```

### アンチパターン 4: 証明書更新の手動運用

```
NG: 証明書の更新を手動で行い、カレンダーで管理
  → 更新忘れでサービス障害が発生
  → 2020年: Microsoft Teams が証明書期限切れで数時間のダウン

OK: cert-manager, certbot, Caddy などで自動更新
  → 証明書の有効期限を Prometheus/Grafana で監視
  → 30日前にアラート、7日前にクリティカルアラート
```

---

## 11. 演習

### 演習 1（基礎）: TLS 接続の確認

以下のコマンドを実行し、TLS 接続の詳細を観察せよ:

```bash
# 1. 任意のサイトの TLS バージョンと暗号スイートを確認
openssl s_client -connect google.com:443 -tls1_3

# 2. 証明書チェーンの発行者を確認
openssl s_client -connect github.com:443 -showcerts 2>/dev/null | \
  grep -E "subject|issuer"

# 質問:
# - TLS 1.3 で使用されている暗号スイートは?
# - 証明書チェーンは何階層か?
# - Root CA は何か?
```

### 演習 2（中級）: 自己署名 CA と証明書の作成

OpenSSL を使用して以下を作成せよ:
1. Root CA の鍵ペアと自己署名証明書
2. 中間 CA の鍵ペアと Root CA で署名した証明書
3. サーバ証明書（SAN 付き）
4. 証明書チェーンの検証

### 演習 3（上級）: mTLS サービス間通信

Go または Python で以下を実装せよ:
1. 内部 CA の構築
2. サーバ証明書とクライアント証明書の自動生成
3. mTLS で保護された HTTP サーバ
4. mTLS クライアントからの API 呼び出し
5. クライアント証明書の失効チェック

---

## 12. パフォーマンスに関する考察

### TLS ハンドシェイクのレイテンシ

| 設定 | レイテンシ (概算) | 最適化方法 |
|------|-----------------|-----------|
| TLS 1.2 フルハンドシェイク | ~100ms | - |
| TLS 1.3 フルハンドシェイク | ~50ms | 1-RTT |
| TLS 1.3 0-RTT | ~0ms | PSK 再利用 |
| TLS セッション再開 (1.2) | ~50ms | セッションチケット |
| ECDSA 署名検証 | ~0.1ms | RSA より高速 |
| RSA 2048 署名検証 | ~0.3ms | - |
| OCSP ステープリング | 節約 ~50ms | CA への問い合わせ不要 |

### 暗号アルゴリズムのスループット比較

```
暗号化スループット (Intel Xeon, single core):

  AES-256-GCM:           ~5 GB/s  (AES-NI ハードウェア)
  AES-128-GCM:           ~6 GB/s
  ChaCha20-Poly1305:     ~2 GB/s  (ソフトウェア)
  ChaCha20-Poly1305:     ~4 GB/s  (AVX2)
  AES-256-CBC:           ~1 GB/s  (非推奨)

  鍵交換:
  ECDHE (x25519):        ~50,000 ops/sec
  ECDHE (P-256):         ~30,000 ops/sec
  RSA-2048 鍵交換:      ~15,000 ops/sec

  → AES-NI 対応の x86 サーバでは AES-GCM が最速
  → ARM (モバイル/Raspberry Pi) では ChaCha20 が高速
```

---

## 13. FAQ

### Q1. TLS 1.3 の 0-RTT は安全か?

0-RTT (Early Data) は再接続時のレイテンシを削減するが、リプレイ攻撃のリスクがある。べき等でない操作 (POST による状態変更など) には使用すべきでない。Nginx では `ssl_early_data on;` で有効化し、バックエンドで `Early-Data: 1` ヘッダを確認して保護する。Google は検索クエリ（べき等な GET）で 0-RTT を使用している。

### Q2. 証明書の有効期間はどのくらいが適切か?

CA/Browser Forum の規定により、公開証明書の最大有効期間は 398 日 (約13ヶ月) である。Let's Encrypt は 90 日で発行し、自動更新を前提とする設計になっている。短い有効期間は鍵の危殆化リスクを低減する。Apple は 2025 年から 45 日への短縮を提案している。

### Q3. 自己署名証明書を使ってよい場面は?

開発環境、内部マイクロサービス間の mTLS、テスト環境に限定すべきである。公開サービスでは必ず信頼された CA の証明書を使用する。自己署名でも CA を作成し、チェーンを正しく構成する運用が望ましい。

### Q4. ECC (楕円曲線暗号) と RSA のどちらを選ぶべきか?

新規構築では ECC (P-256 or Ed25519) を推奨。RSA 2048 と同等の安全性を ECC 256 で実現でき、鍵サイズ・署名サイズが大幅に小さい。RSA は互換性が高いが、将来的には ECC が主流になる。Let's Encrypt は ECC 証明書をデフォルトで発行している。

### Q5. TLS の終端はどこで行うべきか?

ロードバランサ/リバースプロキシで TLS を終端し、バックエンドへは HTTP で通信するパターンが一般的。ただし、ゼロトラスト環境では mTLS でバックエンドまで暗号化を維持すべき。サービスメッシュ (Istio, Linkerd) を使うと mTLS を透過的に導入できる。

---

## まとめ

| 項目 | 要点 |
|------|------|
| TLS バージョン | TLS 1.3 を推奨、最低でも TLS 1.2 |
| ハンドシェイク | TLS 1.3 は 1-RTT で完了し前方秘匿性を必須化 |
| 暗号スイート | AES-GCM / ChaCha20-Poly1305 + ECDHE |
| 証明書チェーン | Root CA → 中間 CA → サーバ証明書の信頼の連鎖 |
| 失効チェック | OCSP Stapling を推奨、CT ログで監視 |
| Let's Encrypt | ACME プロトコルで証明書の取得・更新を自動化 |
| mTLS | クライアント証明書による双方向認証でゼロトラスト実現 |
| 証明書管理 | 自動更新必須、秘密鍵はシークレットマネージャで保護 |
| 監視 | 証明書の有効期限を常時監視し期限切れを防止 |
| 0-RTT | レイテンシ削減だがリプレイリスクあり、べき等操作のみ |

---

## 次に読むべきガイド

- [鍵管理](./02-key-management.md) — 暗号鍵のライフサイクルと安全な保管方法
- [ネットワークセキュリティ基礎](../03-network-security/00-network-security-basics.md) — ファイアウォール・IDS/IPS による多層防御
- [APIセキュリティ](../03-network-security/02-api-security.md) — TLS の上に構築する API 保護
- [DNSセキュリティ](../03-network-security/01-dns-security.md) — DANE/TLSA による DNS ベースの証明書検証

---

## 参考文献

1. **RFC 8446 — The Transport Layer Security (TLS) Protocol Version 1.3** (2018) — https://datatracker.ietf.org/doc/html/rfc8446
2. **RFC 8555 — Automatic Certificate Management Environment (ACME)** — https://datatracker.ietf.org/doc/html/rfc8555
3. **Mozilla SSL Configuration Generator** — https://ssl-config.mozilla.org/ — サーバソフトウェア別の推奨 TLS 設定
4. **Let's Encrypt Documentation** — https://letsencrypt.org/docs/ — ACME プロトコルと証明書自動化の公式ガイド
5. **Qualys SSL Labs — SSL/TLS Best Practices** — https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices
6. **RFC 6960 — X.509 Internet PKI Online Certificate Status Protocol - OCSP** — https://datatracker.ietf.org/doc/html/rfc6960
7. **Certificate Transparency (RFC 6962)** — https://certificate.transparency.dev/
8. **cert-manager Documentation** — https://cert-manager.io/docs/
