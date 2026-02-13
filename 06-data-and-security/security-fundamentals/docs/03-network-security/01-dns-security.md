# DNS セキュリティ

> DNSSEC による応答の完全性保証、DNS over HTTPS によるプライバシー保護、ポイズニング対策まで、DNS に対する脅威と防御手法を体系的に学ぶ

## この章で学ぶこと

1. **DNS の脅威モデル** — キャッシュポイズニング、DNS スプーフィング、DNS トンネリングなど主要な攻撃手法
2. **DNSSEC の仕組み** — 電子署名による DNS 応答の改竄検知メカニズム
3. **暗号化 DNS** — DNS over HTTPS (DoH) / DNS over TLS (DoT) によるクエリの秘匿

### 前提知識

- TCP/IP ネットワーキングの基本（IP アドレス、ポート、UDP/TCP の違い）
- DNS の基本動作（再帰クエリ、権威サーバ、キャッシュリゾルバ）
- 公開鍵暗号の概念（電子署名、ハッシュ関数）

### 関連ガイド

- [ネットワークセキュリティ基礎](./00-network-security-basics.md) — ファイアウォール・IDS/IPS の基本
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) — DoH/DoT の基盤となる TLS の詳細
- [API セキュリティ](./02-api-security.md) — アプリケーション層での通信保護
- [監視/ログ](../06-operations/01-monitoring-logging.md) — DNS ログを含む統合的な監視

---

## 1. DNS の基礎と内部動作

### DNS 名前解決の完全なフロー

```
ユーザが www.example.com にアクセスする場合:

Browser                    Stub Resolver            Recursive Resolver
  |                            |                         |
  |-- www.example.com? ------> |                         |
  |                            |-- www.example.com? ---> |
  |                            |                         |
  |                            |                    Root NS (.):
  |                            |                    「.com は ns1.com に聞け」
  |                            |                         |
  |                            |                    .com NS:
  |                            |                    「example.com は
  |                            |                     ns1.example.com に聞け」
  |                            |                         |
  |                            |                    example.com NS:
  |                            |                    「www.example.com = 93.184.216.34」
  |                            |                         |
  |                            | <-- 93.184.216.34 ----- |
  |                            |     (キャッシュに保存     |
  |                            |      TTL=3600秒)        |
  | <-- 93.184.216.34 ---------|                         |
```

### DNS レコードタイプとセキュリティ上の意味

| レコード | 目的 | セキュリティ上の意味 |
|---------|------|-------------------|
| A / AAAA | ドメイン→IP 解決 | ポイズニングの直接的な標的 |
| NS | 権威サーバの委任 | NS 乗っ取りでゾーン全体を制御 |
| MX | メール配送先 | フィッシングメールの配信先偽装 |
| TXT | 任意テキスト | SPF/DKIM/DMARC、DNS トンネリングの悪用 |
| CNAME | エイリアス | ダングリングCNAME（サブドメインテイクオーバー） |
| SRV | サービス発見 | 内部サービスの列挙に悪用可能 |
| CAA | 証明書発行制御 | 不正な CA による証明書発行を防止 |
| DNSKEY | DNSSEC 公開鍵 | 鍵の漏洩で DNSSEC をバイパス |
| DS | 委任署名者 | 信頼チェーンの接続点 |
| RRSIG | レコード署名 | DNSSEC の完全性保証 |
| NSEC/NSEC3 | 不在証明 | ゾーンウォーキング防止 (NSEC3) |
| TLSA | DANE (TLS 認証) | 証明書ピンニングの DNS 版 |

---

## 2. DNS の脅威

### DNS を狙う攻撃の分類

```
+-----------------------------------------------------------+
|                    DNS への脅威                              |
|-----------------------------------------------------------|
|                                                           |
|  [改竄系]                                                  |
|  +-- キャッシュポイズニング: 偽の応答をキャッシュに注入       |
|  +-- DNS スプーフィング: 偽の DNS サーバに誘導               |
|  +-- BGP ハイジャック経由: 経路を乗っ取り DNS を偽装         |
|  +-- DNS リバインディング: TTL 操作で内部ネットワークにアクセス|
|                                                           |
|  [盗聴系]                                                  |
|  +-- DNS クエリの傍受: 平文クエリからアクセス先を把握        |
|  +-- パッシブ DNS 収集: 組織の通信パターンを分析             |
|  +-- Wi-Fi ハニーポット: 偽APでDNSクエリを収集              |
|                                                           |
|  [悪用系]                                                  |
|  +-- DNS トンネリング: DNS クエリにデータを埋め込んで外部送信 |
|  +-- DDoS (DNS アンプ): 増幅攻撃の踏み台                   |
|  +-- ドメインハイジャック: レジストラアカウントを乗っ取り      |
|  +-- サブドメインテイクオーバー: 未使用CNAMEの悪用           |
|  +-- ファストフラックス: ボットネットのC2をDNSで隠蔽         |
+-----------------------------------------------------------+
```

### キャッシュポイズニングの仕組み（Kaminsky 攻撃）

```
正常な DNS 解決:
  Client --> Resolver --> Authoritative NS
  Client <-- Resolver <-- 正しい応答 (1.2.3.4)

キャッシュポイズニング:
  Client --> Resolver --> Authoritative NS
                 ^
                 |  攻撃者が正規応答より先に偽応答を送信
                 +-- Attacker: "example.com = 6.6.6.6"

  Client <-- Resolver <-- 偽応答 (6.6.6.6) がキャッシュされる
  (以後、TTL期間中は全クライアントが偽IPに誘導される)

Kaminsky 攻撃 (2008) の詳細:
  1. 攻撃者は rand12345.example.com などランダムなサブドメインを問い合わせ
  2. リゾルバは example.com の権威サーバに問い合わせる
  3. 攻撃者は大量の偽応答を送信（Transaction ID のブルートフォース）
     偽応答の Additional Section に以下を含める:
     "example.com の NS は attacker-ns.evil.com である"
  4. Transaction ID が一致すれば、権威サーバごと乗っ取れる

  対策:
  +-- ソースポートのランダム化 (2^16 の追加エントロピー)
  +-- 0x20 エンコーディング (大文字小文字のランダム化)
  +-- DNSSEC (暗号学的な応答の検証)
```

### DNS リバインディング攻撃

```
DNS リバインディングの仕組み:

1. 攻撃者が evil.com を所有し、攻撃者のDNSサーバを設定
2. 被害者が evil.com にアクセス
3. 攻撃者のDNSサーバは最初に正規のIPを返す (TTL=0)
   evil.com → 1.2.3.4 (攻撃者のサーバ)
4. ブラウザが JavaScript をダウンロードして実行
5. JavaScript が evil.com に再度リクエスト
6. 今度は攻撃者のDNSサーバが内部IPを返す
   evil.com → 192.168.1.1 (被害者のルーター)
7. ブラウザの Same-Origin Policy は evil.com として通過
8. JavaScript が内部ネットワークのリソースにアクセス

対策:
+-- DNS リゾルバで RFC1918 アドレスの応答をフィルタ
+-- ブラウザの DNS ピンニング
+-- 内部サービスの Host ヘッダ検証
```

### サブドメインテイクオーバー

```python
# コード例1: サブドメインテイクオーバーの検出スクリプト
import dns.resolver
import requests

KNOWN_FINGERPRINTS = {
    "github.io": "There isn't a GitHub Pages site here",
    "herokuapp.com": "No such app",
    "s3.amazonaws.com": "NoSuchBucket",
    "azurewebsites.net": "404 Web Site not found",
    "cloudfront.net": "Bad Request: ERROR: The request could not be satisfied",
    "netlify.app": "Not Found - Request ID:",
}

def check_subdomain_takeover(domain: str) -> dict:
    """サブドメインテイクオーバーの脆弱性をチェック"""
    result = {"domain": domain, "vulnerable": False, "details": ""}

    try:
        # CNAME レコードを取得
        answers = dns.resolver.resolve(domain, "CNAME")
        for rdata in answers:
            cname_target = str(rdata.target).rstrip(".")
            result["cname"] = cname_target

            # 既知のサービスのフィンガープリントをチェック
            for service, fingerprint in KNOWN_FINGERPRINTS.items():
                if service in cname_target:
                    try:
                        resp = requests.get(
                            f"http://{domain}",
                            timeout=10,
                            allow_redirects=True
                        )
                        if fingerprint in resp.text:
                            result["vulnerable"] = True
                            result["details"] = (
                                f"CNAME points to {cname_target} "
                                f"but the resource does not exist"
                            )
                    except requests.RequestException:
                        result["details"] = f"CNAME target unreachable: {cname_target}"

    except dns.resolver.NXDOMAIN:
        result["details"] = "Domain does not exist"
    except dns.resolver.NoAnswer:
        result["details"] = "No CNAME record"
    except Exception as e:
        result["details"] = str(e)

    return result

# 使用例
subdomains = ["blog.example.com", "staging.example.com", "old.example.com"]
for sub in subdomains:
    result = check_subdomain_takeover(sub)
    if result["vulnerable"]:
        print(f"[VULNERABLE] {result['domain']}: {result['details']}")
```

### DNS アンプ攻撃の仕組みと緩和

```
DNS アンプ/リフレクション攻撃:

  Attacker                    Open Resolver               Victim
     |                            |                         |
     |-- 偽装パケット ---------->  |                         |
     |   Src IP: Victim IP        |                         |
     |   Query: ANY example.com   |                         |
     |   (60 bytes)               |                         |
     |                            |                         |
     |                            |-- 応答 (大量データ) ---> |
     |                            |   ANY レコード全て       |
     |                            |   (~3000 bytes)          |
     |                            |   増幅率: ~50倍         |

  攻撃の特徴:
  - ソースIPの偽装 (BCP38/BCP84 未実装のネットワークから)
  - ANY クエリで最大の応答を引き出す
  - 数千のオープンリゾルバを利用して大規模DDoSを実現

  緩和策:
  +-- リゾルバのアクセス制限 (allow-recursion)
  +-- レスポンスレートリミット (RRL)
  +-- BCP38 (ソースアドレス検証)
  +-- ANY クエリの応答制限 (RFC 8482)
```

---

## 3. DNSSEC

### DNSSEC の概要

DNSSEC (Domain Name System Security Extensions) は DNS 応答の真正性と完全性を暗号学的に保証する拡張仕様である。RFC 4033-4035 で定義され、既存の DNS プロトコルに電子署名を追加する形で動作する。

**重要**: DNSSEC は応答の改竄を検知する仕組みであり、暗号化（秘匿性）は提供しない。DNS クエリの内容は依然として平文である。暗号化には DoH/DoT が必要。

### DNSSEC の信頼チェーン

```
Root Zone (.)
  +-- KSK (Key Signing Key): ZSK に署名
  +-- ZSK (Zone Signing Key): レコードに署名
  +-- DS レコード: 子ゾーンの KSK ハッシュ
       |
       v
TLD (.com)
  +-- KSK / ZSK
  +-- DS レコード (example.com の KSK ハッシュ)
       |
       v
Zone (example.com)
  +-- KSK / ZSK
  +-- RRSIG: 各レコードの電子署名
  +-- NSEC/NSEC3: 不在証明
```

### DNSSEC のレコードタイプ詳解

```
+------------------------------------------------------------------+
|  DNSSEC 関連レコード                                                |
|------------------------------------------------------------------|
|                                                                    |
|  DNSKEY (DNS Public Key)                                           |
|    KSK (flags=257): 鍵署名鍵。DS レコードのハッシュ対象            |
|    ZSK (flags=256): ゾーン署名鍵。実際のレコードに署名             |
|    アルゴリズム: ECDSAP256SHA256 (推奨), RSA/SHA-256               |
|                                                                    |
|  RRSIG (Resource Record Signature)                                 |
|    各 RRset に対する電子署名                                        |
|    含む情報: アルゴリズム、署名有効期間、署名者名、署名値            |
|                                                                    |
|  DS (Delegation Signer)                                            |
|    子ゾーンの KSK のハッシュ値                                      |
|    親ゾーンに配置し、信頼チェーンを構成                              |
|    ハッシュアルゴリズム: SHA-256 (推奨)                              |
|                                                                    |
|  NSEC / NSEC3 (Next Secure)                                        |
|    NSEC: 次のドメイン名を示す → ゾーンウォーキングが可能             |
|    NSEC3: ハッシュ化されたドメイン名 → ゾーンウォーキングを困難化    |
|    NSEC3PARAM: NSEC3 のパラメータ (ハッシュ回数、ソルト)            |
+------------------------------------------------------------------+
```

### DNSSEC の検証プロセス

```
Resolver が example.com の A レコードを検証:

1. example.com の A レコード + RRSIG を取得
2. example.com の DNSKEY (ZSK) で RRSIG を検証
3. example.com の DNSKEY (KSK) で ZSK の RRSIG を検証
4. .com の DS レコードで example.com の KSK を検証
5. .com の DNSKEY で DS の RRSIG を検証
6. Root の DS で .com の KSK を検証
7. Root の KSK はトラストアンカーとして事前に保持

→ チェーン全体が検証できれば「Authenticated Data (AD)」

検証失敗の場合:
  SERVFAIL を返す (検証付きリゾルバの場合)
  → 偽装された応答はクライアントに到達しない
```

### DNSSEC の設定 (BIND)

```bash
# コード例2: BIND での DNSSEC 設定

# ===== 鍵の生成 =====

# ゾーン署名鍵 (ZSK) の生成
# ECDSAP256SHA256 を推奨（RSA より鍵サイズが小さく高速）
dnssec-keygen -a ECDSAP256SHA256 -n ZONE example.com
# 出力: Kexample.com.+013+12345 (.key と .private)

# 鍵署名鍵 (KSK) の生成
dnssec-keygen -a ECDSAP256SHA256 -n ZONE -f KSK example.com
# 出力: Kexample.com.+013+67890 (.key と .private)

# ===== ゾーンファイルの署名 =====

# NSEC3 を使用したゾーン署名
dnssec-signzone -A -3 $(head -c 1000 /dev/urandom | sha1sum | cut -b 1-16) \
  -N INCREMENT -o example.com -t db.example.com

# 出力: db.example.com.signed

# ===== BIND 設定 (named.conf) =====
# DNSSEC 検証を有効化（リゾルバ側）
options {
    dnssec-validation auto;  # トラストアンカーの自動更新 (RFC 5011)
    # dnssec-validation yes; # 手動でトラストアンカーを管理する場合
};

# 権威サーバでの DNSSEC 設定
zone "example.com" {
    type primary;
    file "/etc/bind/zones/db.example.com.signed";
    key-directory "/etc/bind/keys";

    # 自動署名 (inline-signing)
    inline-signing yes;
    auto-dnssec maintain;

    # NSEC3 パラメータの設定
    # rndc signing -nsec3param 1 0 10 auto example.com
};

# ===== 鍵のロールオーバー =====
# ZSK のロールオーバー（推奨: 90日ごと）
# 1. 新しい ZSK を事前公開 (pre-publish)
# 2. TTL 経過後に新 ZSK で署名
# 3. 旧 ZSK を削除

# KSK のロールオーバー（推奨: 1-2年ごと）
# 1. 新 KSK を生成・公開
# 2. 親ゾーンに新 DS レコードを登録
# 3. 旧 KSK で新 KSK を署名
# 4. 旧 DS レコードを削除
# 5. 旧 KSK を削除
```

### dig による DNSSEC 検証

```bash
# コード例3: DNSSEC の検証コマンド集

# DNSSEC 情報付きで問い合わせ
dig +dnssec example.com A

# 検証結果の確認 (flags に ad が含まれれば検証成功)
# ;; flags: qr rd ra ad; QUERY: 1, ANSWER: 2
# ad = Authenticated Data (DNSSEC検証成功)

# DNSKEY レコードの確認
dig example.com DNSKEY +short
# 257 3 13 oJMRESz5E4gYzS/... (KSK, flags=257)
# 256 3 13 2Nwz6FfpJlWey/... (ZSK, flags=256)

# DS レコードの確認
dig example.com DS +short
# 67890 13 2 ABC123DEF456...

# RRSIG の確認（署名の有効期間を含む）
dig example.com A +dnssec +multi
# example.com. 3600 IN RRSIG A 13 2 3600 (
#     20260301000000 20260201000000 12345 example.com.
#     base64signature... )

# NSEC3 による不在証明の確認
dig nonexistent.example.com A +dnssec
# NSEC3 レコードが返り、ドメインが存在しないことを証明

# 信頼チェーンの完全な検証
dig +sigchase +trusted-key=./root.keys example.com A

# delv コマンドによる検証（BIND 9.10+）
delv @8.8.8.8 example.com A +rtrace
# ;; validating example.com/A: verify rdataset (keyid: 12345)
# ;; fully validated

# DNSSEC が有効なドメインの例
dig +dnssec cloudflare.com A   # Cloudflare
dig +dnssec nic.cz A           # CZ NIC
dig +dnssec isc.org A          # ISC (BIND開発元)
```

### DNSSEC のアルゴリズムと鍵長の比較

| アルゴリズム | ID | 鍵長 | 署名長 | 推奨度 | 備考 |
|------------|-----|------|--------|--------|------|
| RSA/SHA-256 | 8 | 2048bit | 256byte | 使用可 | 応答サイズが大きい |
| RSA/SHA-512 | 10 | 2048bit | 256byte | 使用可 | SHA-512の利点は限定的 |
| ECDSAP256SHA256 | 13 | 256bit | 64byte | 推奨 | 鍵・署名が小さく高速 |
| ECDSAP384SHA384 | 14 | 384bit | 96byte | 使用可 | 高セキュリティ用途 |
| Ed25519 | 15 | 256bit | 64byte | 推奨 | 最も高速、段階的導入中 |
| Ed448 | 16 | 456bit | 114byte | 使用可 | 最高セキュリティ |

---

## 4. 暗号化 DNS

### DoH / DoT / DoQ の比較

| 項目 | 平文 DNS | DoT | DoH | DoQ |
|------|---------|-----|-----|-----|
| プロトコル | UDP/53 | TLS/853 | HTTPS/443 | QUIC/853 |
| 暗号化 | なし | TLS | TLS | QUIC |
| 認証 | なし | サーバ証明書 | サーバ証明書 | サーバ証明書 |
| ブロック検知 | 容易 | ポート 853 で判別可 | 通常 HTTPS と区別困難 | ポート 853 で判別可 |
| レイテンシ | 低い | TLS ハンドシェイク分 | HTTP/2 接続分 | 0-RTT 可能 |
| 標準化 | RFC 1035 | RFC 7858 | RFC 8484 | RFC 9250 |
| パディング | なし | RFC 7830 | HTTP/2 フレーム | QUIC フレーム |
| 多重化 | なし | なし | HTTP/2 ストリーム | QUIC ストリーム |
| Head-of-line blocking | N/A | あり | あり | なし |

### DoH の内部プロトコル

```
DoH のリクエスト形式 (RFC 8484):

GET 方式:
  GET /dns-query?dns=AAABAAABAAAAAAAAA3d3dwdleGFtcGxlA2NvbQAAAQAB
  Accept: application/dns-message

  dns パラメータ: DNS ワイヤフォーマットの Base64url エンコード

POST 方式:
  POST /dns-query
  Content-Type: application/dns-message
  Body: [DNS ワイヤフォーマットのバイナリ]

レスポンス:
  HTTP/2 200 OK
  Content-Type: application/dns-message
  Body: [DNS 応答のバイナリ]

JSON API 方式 (Google/Cloudflare 独自拡張):
  GET https://dns.google/resolve?name=example.com&type=A
  レスポンス:
  {
    "Status": 0,
    "TC": false,
    "RD": true,
    "RA": true,
    "AD": true,    // DNSSEC 検証成功
    "CD": false,
    "Question": [{"name": "example.com", "type": 1}],
    "Answer": [
      {"name": "example.com", "type": 1, "TTL": 300,
       "data": "93.184.216.34"}
    ]
  }
```

### DNS over HTTPS の設定

```bash
# コード例4: 各種 DoH/DoT の設定

# ===== dnscrypt-proxy で DoH を利用 =====
# /etc/dnscrypt-proxy/dnscrypt-proxy.toml
listen_addresses = ['127.0.0.1:53']
server_names = ['cloudflare', 'google']
ipv6_servers = false

# セキュリティ設定
require_dnssec = true          # DNSSEC 検証必須
require_nofilter = true        # フィルタリングなし
require_nolog = true           # ログなしサーバのみ

[sources]
  [sources.public-resolvers]
  urls = ['https://raw.githubusercontent.com/DNSCrypt/dnscrypt-resolvers/master/v3/public-resolvers.md']
  cache_file = 'public-resolvers.md'
  minisign_key = 'RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3'

# ===== systemd-resolved で DoT を有効化 =====
# /etc/systemd/resolved.conf
[Resolve]
DNS=1.1.1.1#cloudflare-dns.com 1.0.0.1#cloudflare-dns.com
FallbackDNS=8.8.8.8#dns.google 8.8.4.4#dns.google
DNSOverTLS=yes
DNSSEC=yes

# 設定反映
# sudo systemctl restart systemd-resolved
# resolvectl status で確認

# ===== Unbound で DoT を設定 =====
# /etc/unbound/unbound.conf
server:
    interface: 127.0.0.1
    interface: ::1
    tls-cert-bundle: /etc/ssl/certs/ca-certificates.crt

    # DNSSEC 検証
    auto-trust-anchor-file: "/var/lib/unbound/root.key"
    val-clean-additional: yes

forward-zone:
    name: "."
    forward-tls-upstream: yes
    # Cloudflare DNS over TLS
    forward-addr: 1.1.1.1@853#cloudflare-dns.com
    forward-addr: 1.0.0.1@853#cloudflare-dns.com
    # Google DNS over TLS
    forward-addr: 8.8.8.8@853#dns.google
    forward-addr: 8.8.4.4@853#dns.google
```

### Go で DoH クライアント

```go
// コード例5: 完全な DoH クライアント実装 (Go)
package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/miekg/dns"
)

// DoHClient は DNS over HTTPS クライアント
type DoHClient struct {
	httpClient *http.Client
	serverURL  string
}

// NewDoHClient は新しい DoH クライアントを生成する
func NewDoHClient(serverURL string) *DoHClient {
	return &DoHClient{
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					MinVersion: tls.VersionTLS13,
				},
				MaxIdleConns:        100,
				IdleConnTimeout:     90 * time.Second,
				TLSHandshakeTimeout: 10 * time.Second,
			},
		},
		serverURL: serverURL,
	}
}

// Query は DoH で DNS クエリを実行する
func (c *DoHClient) Query(domain string, qtype uint16) (*dns.Msg, error) {
	// DNS メッセージを構築
	msg := new(dns.Msg)
	msg.SetQuestion(dns.Fqdn(domain), qtype)
	msg.RecursionDesired = true

	// EDNS0 パディング (RFC 7830)
	opt := &dns.OPT{
		Hdr: dns.RR_Header{Name: ".", Rrtype: dns.TypeOPT},
		Option: []dns.EDNS0{
			&dns.EDNS0_PADDING{Padding: make([]byte, 128)},
		},
	}
	opt.SetUDPSize(4096)
	opt.SetDo(true) // DNSSEC OK フラグ
	msg.Extra = append(msg.Extra, opt)

	// Wire format にエンコード
	packed, err := msg.Pack()
	if err != nil {
		return nil, fmt.Errorf("failed to pack DNS message: %w", err)
	}

	// DoH POST リクエスト
	req, err := http.NewRequest(
		"POST",
		c.serverURL,
		bytes.NewReader(packed),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/dns-message")
	req.Header.Set("Accept", "application/dns-message")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("DoH request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DoH server returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	response := new(dns.Msg)
	if err := response.Unpack(body); err != nil {
		return nil, fmt.Errorf("failed to unpack DNS response: %w", err)
	}

	return response, nil
}

// QueryWithDNSSECValidation は DNSSEC 検証付きクエリ
func (c *DoHClient) QueryWithDNSSECValidation(domain string) ([]net.IP, bool, error) {
	resp, err := c.Query(domain, dns.TypeA)
	if err != nil {
		return nil, false, err
	}

	var ips []net.IP
	authenticated := resp.AuthenticatedData // AD フラグ

	for _, answer := range resp.Answer {
		if a, ok := answer.(*dns.A); ok {
			ips = append(ips, a.A)
		}
	}

	return ips, authenticated, nil
}

func main() {
	client := NewDoHClient("https://cloudflare-dns.com/dns-query")

	ips, dnssecValid, err := client.QueryWithDNSSECValidation("cloudflare.com")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("IPs: %v\n", ips)
	fmt.Printf("DNSSEC validated: %v\n", dnssecValid)
}
```

### Python で DoH クライアント

```python
# コード例6: Python DoH クライアント
import dns.message
import dns.query
import dns.rdatatype
import httpx
import base64

class DoHResolver:
    """DNS over HTTPS リゾルバ"""

    def __init__(self, server_url: str = "https://cloudflare-dns.com/dns-query"):
        self.server_url = server_url
        self.client = httpx.Client(
            http2=True,
            timeout=10.0,
            headers={"Accept": "application/dns-message"}
        )

    def resolve(self, domain: str, rdtype: str = "A") -> list:
        """DoH でドメインを解決"""
        # DNS クエリメッセージの構築
        query = dns.message.make_query(domain, rdtype, want_dnssec=True)
        wire = query.to_wire()

        # POST リクエスト
        response = self.client.post(
            self.server_url,
            content=wire,
            headers={"Content-Type": "application/dns-message"}
        )
        response.raise_for_status()

        # レスポンスの解析
        dns_response = dns.message.from_wire(response.content)

        results = []
        for rrset in dns_response.answer:
            for rdata in rrset:
                results.append({
                    "name": str(rrset.name),
                    "type": dns.rdatatype.to_text(rrset.rdtype),
                    "ttl": rrset.ttl,
                    "data": str(rdata),
                })

        return {
            "answers": results,
            "dnssec_validated": bool(dns_response.flags & dns.flags.AD),
            "rcode": dns.rcode.to_text(dns_response.rcode()),
        }

    def close(self):
        self.client.close()


# 使用例
resolver = DoHResolver()
result = resolver.resolve("example.com", "A")
print(f"Answers: {result['answers']}")
print(f"DNSSEC: {result['dnssec_validated']}")
resolver.close()
```

---

## 5. DNS ポイズニング対策

### 多層的な防御策

```
+----------------------------------------------+
|            DNS ポイズニング対策                  |
|----------------------------------------------|
|                                              |
|  [プロトコル層]                                |
|  +-- DNSSEC: 応答の改竄を検知                  |
|  +-- DoH/DoT: クエリの盗聴・改竄を防止         |
|  +-- DANE/TLSA: DNS で TLS 証明書を検証       |
|                                              |
|  [リゾルバ層]                                  |
|  +-- ソースポートランダム化                     |
|  +-- Query ID ランダム化 (16bit)              |
|  +-- 0x20 エンコーディング (大文字小文字ミックス) |
|  +-- TCP へのフォールバック                    |
|  +-- キャッシュの最小TTL設定                   |
|                                              |
|  [ネットワーク層]                               |
|  +-- BCP38 (ソースアドレス検証)                |
|  +-- ACL でリゾルバへのアクセス制限             |
|  +-- 異常なDNSトラフィックの検知               |
|                                              |
|  [運用層]                                     |
|  +-- TTL の適切な設定                          |
|  +-- DNS ログの監視                           |
|  +-- RPZ (Response Policy Zone) でブロック     |
|  +-- CAA レコードの設定                        |
+----------------------------------------------+
```

### 0x20 エンコーディング

```
0x20 エンコーディング (DNS 0x20 bit encoding):

通常のクエリ:
  Query: www.example.com

0x20 エンコーディング適用:
  Query: wWw.ExAmPlE.cOm  (ランダムに大文字小文字を混ぜる)

DNS は大文字小文字を区別しないため、正規のサーバは
リクエストと同じ大文字小文字パターンで応答する。

攻撃者は正しい大文字小文字パターンを推測する必要があるため、
ブルートフォースの難易度が大幅に上がる。

追加エントロピー: ドメイン名の長さ × 1 bit
例: www.example.com (15文字) → 2^15 = 32,768 通り

Transaction ID (16bit) + ソースポート (16bit) + 0x20 (15bit)
= 2^47 ≈ 140 兆通り → ブルートフォースは実質不可能
```

### RPZ (Response Policy Zone) の設定

```bash
# コード例7: BIND での RPZ 設定

# /etc/bind/named.conf
options {
    response-policy {
        zone "rpz.local" policy given;
        zone "rpz.spamhaus" policy given;
    };
};

zone "rpz.local" {
    type primary;
    file "/etc/bind/db.rpz.local";
    allow-query { none; };
};

# RPZ ゾーンファイル (/etc/bind/db.rpz.local)
$TTL 300
@ IN SOA localhost. admin.localhost. ( 1 3600 900 604800 300 )
@ IN NS localhost.

; ===== マルウェアドメインのブロック =====
; NXDOMAIN を返す（最も安全）
malware.example.com   CNAME .
*.malware.example.com CNAME .

; ===== フィッシングサイトをブロックページにリダイレクト =====
phishing.example.com  A     10.0.0.100
*.phishing.example.com A    10.0.0.100

; ===== 特定のIPアドレスへのアクセスをブロック =====
; C2サーバのIPをブロック
32.1.168.192.rpz-ip     CNAME .

; ===== ワイルドカードによる大量ブロック =====
; DGA (Domain Generation Algorithm) 対策
*.dga-pattern.com  CNAME .

; ===== パススルー (ブロック除外) =====
; RPZ ルールの例外
safe.malware.example.com CNAME rpz-passthru.
```

### CAA レコードによる証明書発行制御

```bash
# コード例8: CAA レコードの設定

# CAA (Certificate Authority Authorization) レコード
# 指定した CA のみがドメインの証明書を発行できる

# Let's Encrypt のみ許可
example.com.  CAA 0 issue "letsencrypt.org"

# ワイルドカード証明書は DigiCert のみ
example.com.  CAA 0 issuewild "digicert.com"

# 証明書発行に関する問題を報告するメールアドレス
example.com.  CAA 0 iodef "mailto:security@example.com"

# 設定確認
dig example.com CAA +short
# 0 issue "letsencrypt.org"
# 0 issuewild "digicert.com"
# 0 iodef "mailto:security@example.com"
```

---

## 6. DNS トンネリングの検知

### DNS トンネリングの仕組み

```
DNS トンネリングの原理:

  内部ネットワーク (ファイアウォール内)        攻撃者の DNS サーバ
       Client (マルウェア)                    ns1.evil.com
         |                                       |
         |  Base32/Hex エンコードしたデータを      |
         |  サブドメインに埋め込んで送信            |
         |                                       |
  TXT クエリ: dGhpcyBpcyBzZWNyZXQ.evil.com       |
         |  → "this is secret" の Base64          |
         |                                       |
         |  ファイアウォールは DNS (port 53) を     |
         |  通常許可しているため通過する            |
         |                                       |
         |  <--- TXT レスポンスでコマンドを受信     |
         |  "exec: whoami"                        |

  特徴:
  - 帯域幅: ~500kbps (TXT レコード使用時)
  - レイテンシ: 高い（DNS TTL に依存）
  - ツール: iodine, dnscat2, dns2tcp
```

### 検知手法と実装

```python
# コード例9: DNS トンネリング検知スクリプト (拡張版)
import collections
import math
import re
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class DNSQuery:
    timestamp: datetime
    src_ip: str
    qname: str
    qtype: str

def shannon_entropy(s: str) -> float:
    """文字列のシャノンエントロピーを計算"""
    if not s:
        return 0.0
    probabilities = [
        count / len(s)
        for count in collections.Counter(s).values()
    ]
    return -sum(p * math.log2(p) for p in probabilities)

def extract_base_domain(qname: str) -> str:
    """サブドメインを除去してベースドメインを抽出"""
    parts = qname.rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return qname

def get_subdomain_labels(qname: str) -> str:
    """ベースドメインを除いたサブドメイン部分を取得"""
    parts = qname.rstrip(".").split(".")
    if len(parts) > 2:
        return ".".join(parts[:-2])
    return ""

class DNSTunnelDetector:
    """DNS トンネリングの検知エンジン"""

    # しきい値（チューニング可能）
    ENTROPY_THRESHOLD = 3.5      # ランダム文字列のエントロピー
    LABEL_LENGTH_THRESHOLD = 50  # サブドメインラベルの長さ
    QUERY_RATE_THRESHOLD = 100   # 1分間のクエリ数
    TXT_RATIO_THRESHOLD = 0.3    # TXTクエリの割合
    UNIQUE_SUBDOMAIN_THRESHOLD = 50  # ユニークなサブドメイン数/分

    def __init__(self):
        self.alerts = []

    def analyze(self, queries: list[DNSQuery], window_minutes: int = 5) -> list:
        """クエリログを分析してトンネリングの兆候を検知"""
        # ソースIPごとにグループ化
        by_source = collections.defaultdict(list)
        for q in queries:
            by_source[q.src_ip].append(q)

        for src_ip, src_queries in by_source.items():
            indicators = self._check_indicators(src_queries, window_minutes)
            if indicators["score"] >= 3:  # 3つ以上の指標でアラート
                self.alerts.append({
                    "source_ip": src_ip,
                    "severity": "HIGH" if indicators["score"] >= 4 else "MEDIUM",
                    "indicators": indicators,
                    "sample_queries": [q.qname for q in src_queries[:5]],
                })

        return self.alerts

    def _check_indicators(self, queries: list[DNSQuery], window_min: int) -> dict:
        score = 0
        indicators = {}

        # 指標 1: 異常に長いサブドメインラベル
        long_labels = [
            q for q in queries
            if len(get_subdomain_labels(q.qname)) > self.LABEL_LENGTH_THRESHOLD
        ]
        if long_labels:
            score += 1
            indicators["long_labels"] = len(long_labels)

        # 指標 2: 高エントロピーのサブドメイン
        high_entropy = []
        for q in queries:
            subdomain = get_subdomain_labels(q.qname)
            if subdomain and shannon_entropy(subdomain) > self.ENTROPY_THRESHOLD:
                high_entropy.append(q)
        if high_entropy:
            score += 1
            indicators["high_entropy"] = len(high_entropy)

        # 指標 3: TXT レコードの異常な割合
        txt_queries = [q for q in queries if q.qtype == "TXT"]
        txt_ratio = len(txt_queries) / max(len(queries), 1)
        if txt_ratio > self.TXT_RATIO_THRESHOLD:
            score += 1
            indicators["txt_ratio"] = round(txt_ratio, 3)

        # 指標 4: 同一ベースドメインへの高頻度クエリ
        domain_counts = collections.Counter(
            extract_base_domain(q.qname) for q in queries
        )
        high_freq = {d: c for d, c in domain_counts.items()
                     if c > self.QUERY_RATE_THRESHOLD}
        if high_freq:
            score += 1
            indicators["high_freq_domains"] = high_freq

        # 指標 5: ユニークなサブドメインの数が異常に多い
        unique_subdomains = collections.defaultdict(set)
        for q in queries:
            base = extract_base_domain(q.qname)
            sub = get_subdomain_labels(q.qname)
            if sub:
                unique_subdomains[base].add(sub)

        for base, subs in unique_subdomains.items():
            if len(subs) > self.UNIQUE_SUBDOMAIN_THRESHOLD:
                score += 1
                indicators["unique_subdomains"] = {base: len(subs)}
                break

        # 指標 6: Base32/Base64 パターンの検出
        b64_pattern = re.compile(r'^[A-Za-z0-9+/=]{20,}$')
        b32_pattern = re.compile(r'^[A-Z2-7=]{20,}$')
        encoded_count = sum(
            1 for q in queries
            if b64_pattern.match(get_subdomain_labels(q.qname).replace(".", ""))
            or b32_pattern.match(get_subdomain_labels(q.qname).replace(".", ""))
        )
        if encoded_count > 10:
            score += 1
            indicators["encoded_queries"] = encoded_count

        indicators["score"] = score
        return indicators
```

### ネットワーク機器での DNS モニタリング

```bash
# コード例10: DNS モニタリングの設定

# ===== Suricata での DNS 異常検知ルール =====
# /etc/suricata/rules/dns-tunnel.rules

# 異常に長い DNS クエリの検出
alert dns any any -> any any (msg:"DNS Tunnel - Long query name";
  dns.query; content:"|00|"; offset:50;
  threshold: type threshold, track by_src, count 10, seconds 60;
  classtype:bad-unknown; sid:1000001; rev:1;)

# TXT レコードの大量クエリ
alert dns any any -> any any (msg:"DNS Tunnel - Excessive TXT queries";
  dns.query; content:"|00 10|";  # TXT record type
  threshold: type threshold, track by_src, count 50, seconds 60;
  classtype:bad-unknown; sid:1000002; rev:1;)

# ===== Zeek (Bro) での DNS ログ分析 =====
# dns.zeek スクリプト
@load base/protocols/dns

event dns_request(c: connection, msg: dns_msg, query: string, qtype: count)
{
    if ( |query| > 60 )
    {
        NOTICE([$note=DNS::Tunneling_Indicator,
                $msg=fmt("Long DNS query from %s: %s", c$id$orig_h, query),
                $conn=c]);
    }
}

# ===== tcpdump での DNS トラフィック監視 =====
# DNS クエリをリアルタイム監視
tcpdump -i eth0 -n port 53 -l | awk '/A\?/ {print $NF}'

# 長いクエリ名のみ表示
tcpdump -i eth0 -n port 53 -l | awk 'length($NF) > 60 {print}'
```

---

## 7. DANE (DNS-Based Authentication of Named Entities)

### DANE / TLSA レコード

```
DANE は DNS を使って TLS 証明書を認証する仕組み (RFC 6698)。
DNSSEC が必須の前提条件。

TLSA レコードの構造:
  _port._protocol.domain TLSA usage selector matching data

  Usage (利用法):
    0 = PKIX-TA: CA 証明書を指定
    1 = PKIX-EE: サーバ証明書を指定 (CA信頼ストアも検証)
    2 = DANE-TA: 独自 CA を指定 (CA信頼ストア不要)
    3 = DANE-EE: サーバ証明書を指定 (CA信頼ストア不要)

  Selector:
    0 = 証明書全体
    1 = 公開鍵のみ (SubjectPublicKeyInfo)

  Matching:
    0 = 完全一致
    1 = SHA-256 ハッシュ
    2 = SHA-512 ハッシュ

例:
  _443._tcp.example.com. IN TLSA 3 1 1 \
    2bb183af2b5a15f1168960b45a258a4e180f5... (公開鍵の SHA-256)
```

```bash
# TLSA レコードの生成
openssl x509 -in server.crt -pubkey -noout | \
  openssl pkey -pubin -outform DER | \
  openssl dgst -sha256 -binary | \
  xxd -p -c 256

# TLSA レコードの検証
dig _443._tcp.example.com TLSA +short
```

---

## 8. アンチパターン

### アンチパターン 1: DNSSEC 未導入のまま放置

```
NG: DNSSEC を設定せず平文 DNS のまま運用
  → キャッシュポイズニングで利用者をフィッシングサイトに誘導される

OK: DNSSEC を有効化し DS レコードをレジストラに登録
  → 改竄された応答は検証失敗で破棄される
```

**影響**: 中間者が DNS 応答を改竄できるため、正規ドメインで偽サイトに誘導可能。2024年の Savannah 攻撃ではDNSSEC未導入の金融機関が標的となった。

### アンチパターン 2: 社内 DNS リゾルバの外部公開

```
NG: 社内リゾルバが 0.0.0.0:53 でリッスン
  → DNS アンプ攻撃の踏み台になる
  → 内部ドメイン情報が漏洩する

OK: リゾルバは社内ネットワークのみにバインド
  listen-on { 10.0.0.0/8; 127.0.0.1; };
  allow-recursion { 10.0.0.0/8; };
  allow-transfer { none; };
```

### アンチパターン 3: ワイルドカード DNS レコードの不用意な設定

```
NG: *.example.com → 1.2.3.4
  → 任意のサブドメインでフィッシングサイトを構築可能
  → SSL 証明書の発行が容易になる
  → サブドメインテイクオーバーの検出が困難

OK: 必要なサブドメインのみ個別にレコードを作成
  www.example.com → 1.2.3.4
  api.example.com → 1.2.3.5
  不要なサブドメインは NXDOMAIN を返す
```

### アンチパターン 4: DNS ログの未収集

```
NG: DNS クエリログを保存していない
  → インシデント発生時に C2 通信の痕跡を追跡できない
  → DNS トンネリングを検知できない

OK: DNS クエリログを SIEM に転送し、異常検知ルールを設定
  - 高エントロピーのクエリ名
  - 異常な TXT レコード比率
  - 未知ドメインへの大量クエリ
  - 通常時間帯外の DNS トラフィック
```

---

## 9. エッジケース

### エッジケース 1: DNS rebinding によるファイアウォールバイパス

DNS の TTL を 0 に設定し、最初のクエリでは正規の IP を、再クエリでは内部 IP (192.168.x.x) を返すことで、ブラウザの Same-Origin Policy を維持しながら内部ネットワークにアクセスする。対策には DNS ピンニングと内部サービスの Host ヘッダ検証が必要。

### エッジケース 2: NSEC ゾーンウォーキング

DNSSEC の NSEC レコードは「次のドメイン名」を示すため、NSEC レコードを順に辿ることでゾーン内の全ドメインを列挙できる。NSEC3 はドメイン名をハッシュ化することでこれを困難にするが、完全には防げない（オフラインでのハッシュクラック）。NSEC5 (実験的) はさらなる改善を目指している。

### エッジケース 3: DNS の UDP フラグメンテーション

DNSSEC 応答は通常の DNS 応答より大きく（署名データを含む）、UDP の最大サイズ (512 バイト) を超えることがある。EDNS0 で UDP ペイロードサイズを拡張するが、中間ネットワーク機器がフラグメンテーションされたパケットを破棄する場合がある。DNS Flag Day 2020 以降、応答サイズは 1232 バイト以下が推奨される。

### エッジケース 4: Happy Eyeballs と DNS

デュアルスタック環境での A/AAAA レコードの同時解決と接続レースにおいて、DNS 応答の到着順序によって IPv4/IPv6 の選択が変わる。攻撃者が一方のレコードのみをポイズニングした場合、接続先が不定になるリスクがある。

---

## 10. 演習

### 演習 1（基礎）: DNSSEC の検証

以下のコマンドを実行し、DNSSEC の検証状態を確認せよ。

```bash
# 1. DNSSEC 対応ドメインの確認
dig +dnssec cloudflare.com A

# 2. DNSSEC 非対応ドメインとの違いを確認
# (AD フラグの有無に注目)

# 3. DNSKEY と DS レコードの確認
dig cloudflare.com DNSKEY +short
dig cloudflare.com DS +short

# 質問:
# - AD フラグは何を意味するか?
# - KSK と ZSK のフラグ値の違いは?
# - DS レコードはどのゾーンに配置されるか?
```

### 演習 2（中級）: DoH クライアントの実装

Python の `httpx` ライブラリを使用して、以下の機能を持つ DoH クライアントを実装せよ:

1. RFC 8484 準拠の POST リクエスト
2. DNSSEC 検証結果 (AD フラグ) の表示
3. 複数の DoH サーバへのフェイルオーバー

### 演習 3（上級）: DNS トンネリング検知システム

DNS クエリログ（BIND query log 形式）を入力として、以下の指標でDNSトンネリングを検知するシステムを構築せよ:

1. サブドメインのシャノンエントロピー
2. クエリ名の長さ分布の統計的異常
3. TXT レコードクエリの割合
4. 同一ベースドメインへのユニークサブドメイン数
5. Base32/Base64 エンコードパターンの検出

しきい値は設定可能とし、アラートをJSON形式で出力すること。

---

## 11. パフォーマンスに関する考察

### DNSSEC のパフォーマンス影響

| 項目 | 影響 | 緩和策 |
|------|------|--------|
| 応答サイズの増大 | RRSIG で 200-500 バイト増 | ECDSA (小さい署名) |
| 検証の CPU 負荷 | RSA 検証: ~1ms, ECDSA: ~0.3ms | Ed25519 でさらに高速化 |
| UDP フラグメンテーション | 中間機器で破棄される可能性 | EDNS0 バッファサイズ 1232 |
| 鍵ロールオーバー | 一時的にキャッシュミス増加 | 事前公開期間を十分に |
| 不在証明 (NSEC3) | ハッシュ計算のオーバーヘッド | NSEC3 iterations を 0 に (RFC 9276) |

### DoH/DoT のレイテンシ比較

```
レイテンシ比較 (平均値、同一サーバ):

  平文 DNS (UDP):     ~5ms   (キャッシュヒット: <1ms)
  DoT (初回接続):     ~50ms  (TLS ハンドシェイク含む)
  DoT (再利用):       ~10ms  (TLS セッション再利用)
  DoH (初回接続):     ~80ms  (HTTP/2 + TLS)
  DoH (再利用):       ~15ms  (HTTP/2 多重化)
  DoQ (初回接続):     ~30ms  (QUIC 0-RTT)
  DoQ (再利用):       ~5ms   (0-RTT 再接続)

  → DoQ が次世代の暗号化 DNS として最もバランスが良い
```

---

## 12. FAQ

### Q1. DNSSEC はなぜ普及が遅いのか?

DNSSEC は鍵管理の複雑さ、ゾーン署名の運用負荷、NSEC によるゾーンウォーキング（列挙攻撃）の懸念がある。また、応答サイズが大きくなり UDP フラグメンテーション問題が発生しうる。NSEC3 や自動署名 (BIND の inline-signing) の導入で改善されつつあるが、依然として導入障壁は高い。2025年時点での DNSSEC 署名率は .com で約5%、.nl (オランダ) で約60% と地域差が大きい。

### Q2. DoH を企業ネットワークで使うべきか?

DoH はプライバシーを向上させるが、企業のセキュリティ監視を迂回するリスクがある。企業ネットワークでは内部 DoH リゾルバを運用し、外部 DoH/DoT への通信をブロックするのが一般的である。これによりプライバシーと可視性を両立できる。具体的には、ポート 853 (DoT) のブロック、既知の DoH エンドポイント (cloudflare-dns.com 等) のブロック、内部 CA を使った HTTPS インスペクションの検討が必要。

### Q3. DNS フィルタリングはセキュリティ対策として有効か?

RPZ や Pi-hole などによる DNS フィルタリングは、マルウェアの C2 通信やフィッシングサイトへのアクセスを低コストで防止できる有効な対策である。ただし、IP 直接アクセスや DoH バイパスに対しては無力であり、多層防御の一層として位置付けるべきである。

### Q4. DNSログの保持期間はどのくらいが適切か?

NIST SP 800-92 では最低90日のログ保持を推奨。GDPR環境下では個人データとしてのDNSクエリの取り扱いに注意が必要。インシデントレスポンスの観点からは6ヶ月〜1年が理想的だが、ストレージコストとのバランスを考慮する。圧縮やサマリログの活用で保持期間を延長できる。

### Q5. DNS over QUIC (DoQ) はいつ普及するか?

RFC 9250 で標準化済み。AdGuard DNS など一部のサービスが対応を開始しているが、クライアント側のサポートが限定的。QUIC の 0-RTT により DoH/DoT よりも低レイテンシを実現でき、Head-of-line blocking も解消されるため、将来的には DoH を置き換える可能性がある。

---

## まとめ

| 項目 | 要点 |
|------|------|
| DNS の脅威 | ポイズニング・スプーフィング・トンネリング・リバインディングが主要リスク |
| DNSSEC | 電子署名で応答の完全性を検証、信頼チェーンで root まで辿る |
| DoH/DoT/DoQ | DNS クエリを暗号化しプライバシーと改竄防止を実現 |
| ポイズニング対策 | DNSSEC + ポートランダム化 + 0x20 エンコーディング |
| DNS トンネリング | クエリ長・エントロピー・頻度で異常を検知 |
| RPZ | ポリシーベースで悪意あるドメインをブロック |
| DANE/TLSA | DNS で TLS 証明書を検証 (DNSSEC 必須) |
| CAA | 証明書発行を許可する CA を DNS で制限 |
| サブドメインテイクオーバー | ダングリング CNAME の定期的な監査 |

---

## 次に読むべきガイド

- [ネットワークセキュリティ基礎](./00-network-security-basics.md) — ファイアウォール・IDS/IPS による網羅的な防御
- [APIセキュリティ](./02-api-security.md) — アプリケーション層での通信保護
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) — DoH/DoT の基盤となる TLS の詳細
- [監視/ログ](../06-operations/01-monitoring-logging.md) — DNS ログを含む統合的な監視体制

---

## 参考文献

1. **RFC 4033-4035 — DNS Security Introduction and Requirements (DNSSEC)** — https://datatracker.ietf.org/doc/html/rfc4033
2. **RFC 8484 — DNS Queries over HTTPS (DoH)** — https://datatracker.ietf.org/doc/html/rfc8484
3. **RFC 7858 — Specification for DNS over Transport Layer Security (DoT)** — https://datatracker.ietf.org/doc/html/rfc7858
4. **RFC 9250 — DNS over Dedicated QUIC Connections (DoQ)** — https://datatracker.ietf.org/doc/html/rfc9250
5. **RFC 6698 — DNS-Based Authentication of Named Entities (DANE)** — https://datatracker.ietf.org/doc/html/rfc6698
6. **RFC 9276 — Guidance for NSEC3 Parameter Settings** — https://datatracker.ietf.org/doc/html/rfc9276
7. **NIST SP 800-81-2 — Secure Domain Name System (DNS) Deployment Guide** — https://csrc.nist.gov/publications/detail/sp/800-81/2/final
8. **DNS Flag Day** — https://dnsflagday.net/ — DNS の最新標準準拠に関する業界イニシアチブ
9. **Kaminsky DNS Vulnerability (2008)** — https://www.kb.cert.org/vuls/id/800113
