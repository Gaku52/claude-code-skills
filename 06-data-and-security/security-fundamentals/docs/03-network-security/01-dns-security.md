# DNS セキュリティ

> DNSSEC による応答の完全性保証、DNS over HTTPS によるプライバシー保護、ポイズニング対策まで、DNS に対する脅威と防御手法を体系的に学ぶ

## この章で学ぶこと

1. **DNS の脅威モデル** — キャッシュポイズニング、DNS スプーフィング、DNS トンネリングなど主要な攻撃手法
2. **DNSSEC の仕組み** — 電子署名による DNS 応答の改竄検知メカニズム
3. **暗号化 DNS** — DNS over HTTPS (DoH) / DNS over TLS (DoT) によるクエリの秘匿

---

## 1. DNS の脅威

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
|                                                           |
|  [盗聴系]                                                  |
|  +-- DNS クエリの傍受: 平文クエリからアクセス先を把握        |
|  +-- パッシブ DNS 収集: 組織の通信パターンを分析             |
|                                                           |
|  [悪用系]                                                  |
|  +-- DNS トンネリング: DNS クエリにデータを埋め込んで外部送信 |
|  +-- DDoS (DNS アンプ): 増幅攻撃の踏み台                   |
|  +-- ドメインハイジャック: レジストラアカウントを乗っ取り      |
+-----------------------------------------------------------+
```

### キャッシュポイズニングの仕組み

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
```

---

## 2. DNSSEC

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
```

### DNSSEC の設定 (BIND)

```bash
# ゾーン署名鍵 (ZSK) の生成
dnssec-keygen -a ECDSAP256SHA256 -n ZONE example.com

# 鍵署名鍵 (KSK) の生成
dnssec-keygen -a ECDSAP256SHA256 -n ZONE -f KSK example.com

# ゾーンファイルの署名
dnssec-signzone -A -3 $(head -c 1000 /dev/urandom | sha1sum | cut -b 1-16) \
  -N INCREMENT -o example.com -t db.example.com
```

### dig による DNSSEC 検証

```bash
# DNSSEC 情報付きで問い合わせ
dig +dnssec example.com A

# 検証結果の確認 (flags に ad が含まれれば検証成功)
# ;; flags: qr rd ra ad; QUERY: 1, ANSWER: 2

# DNSKEY レコードの確認
dig example.com DNSKEY +short

# DS レコードの確認
dig example.com DS +short

# NSEC3 による不在証明の確認
dig nonexistent.example.com A +dnssec
```

---

## 3. 暗号化 DNS

### DoH / DoT / DoQ の比較

| 項目 | 平文 DNS | DoT | DoH | DoQ |
|------|---------|-----|-----|-----|
| プロトコル | UDP/53 | TLS/853 | HTTPS/443 | QUIC/853 |
| 暗号化 | なし | TLS | TLS | QUIC |
| 認証 | なし | サーバ証明書 | サーバ証明書 | サーバ証明書 |
| ブロック検知 | 容易 | ポート 853 で判別可 | 通常 HTTPS と区別困難 | ポート 853 で判別可 |
| レイテンシ | 低い | TLS ハンドシェイク分 | HTTP/2 接続分 | 0-RTT 可能 |
| 標準化 | RFC 1035 | RFC 7858 | RFC 8484 | RFC 9250 |

### DNS over HTTPS の設定

```bash
# dnscrypt-proxy で DoH を利用 (/etc/dnscrypt-proxy/dnscrypt-proxy.toml)
listen_addresses = ['127.0.0.1:53']
server_names = ['cloudflare', 'google']

[sources]
  [sources.public-resolvers]
  urls = ['https://raw.githubusercontent.com/DNSCrypt/dnscrypt-resolvers/master/v3/public-resolvers.md']
  cache_file = 'public-resolvers.md'
  minisign_key = 'RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3'

# systemd-resolved で DoT を有効化
# /etc/systemd/resolved.conf
[Resolve]
DNS=1.1.1.1#cloudflare-dns.com
DNSOverTLS=yes
```

### Go で DoH クライアント

```go
package main

import (
    "context"
    "fmt"
    "net"
    "net/http"

    "github.com/miekg/dns"
)

func queryDoH(domain string) (*dns.Msg, error) {
    // DNS メッセージを構築
    msg := new(dns.Msg)
    msg.SetQuestion(dns.Fqdn(domain), dns.TypeA)
    msg.RecursionDesired = true

    // Wire format にエンコード
    packed, err := msg.Pack()
    if err != nil {
        return nil, err
    }

    // DoH POST リクエスト
    resp, err := http.Post(
        "https://cloudflare-dns.com/dns-query",
        "application/dns-message",
        bytes.NewReader(packed),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    response := new(dns.Msg)
    response.Unpack(body)

    return response, nil
}
```

---

## 4. DNS ポイズニング対策

### 多層的な防御策

```
+----------------------------------------------+
|            DNS ポイズニング対策                  |
|----------------------------------------------|
|                                              |
|  [プロトコル層]                                |
|  +-- DNSSEC: 応答の改竄を検知                  |
|  +-- DoH/DoT: クエリの盗聴・改竄を防止         |
|                                              |
|  [リゾルバ層]                                  |
|  +-- ソースポートランダム化                     |
|  +-- Query ID ランダム化                      |
|  +-- 0x20 エンコーディング (大文字小文字ミックス) |
|                                              |
|  [運用層]                                     |
|  +-- TTL の適切な設定                          |
|  +-- DNS ログの監視                           |
|  +-- RPZ (Response Policy Zone) でブロック     |
+----------------------------------------------+
```

### RPZ (Response Policy Zone) の設定

```bash
# BIND での RPZ 設定 (/etc/bind/named.conf)
options {
    response-policy {
        zone "rpz.local" policy given;
    };
};

zone "rpz.local" {
    type master;
    file "/etc/bind/db.rpz.local";
    allow-query { none; };
};

# RPZ ゾーンファイル (/etc/bind/db.rpz.local)
$TTL 300
@ IN SOA localhost. admin.localhost. ( 1 3600 900 604800 300 )
@ IN NS localhost.

; マルウェアドメインのブロック
malware.example.com   CNAME .  ; NXDOMAIN を返す
*.malware.example.com CNAME .

; フィッシングサイトをブロックページにリダイレクト
phishing.example.com  A     10.0.0.100
```

---

## 5. DNS トンネリングの検知

### 検知の観点

```bash
# DNS クエリログの分析スクリプト (Python)
import collections
from datetime import datetime

def detect_dns_tunneling(log_entries):
    """DNS トンネリングの兆候を検知"""
    alerts = []

    for src_ip, queries in group_by_source(log_entries):
        # 指標 1: 異常に長いドメイン名
        long_queries = [q for q in queries if len(q.qname) > 50]

        # 指標 2: 高エントロピーのサブドメイン
        high_entropy = [q for q in queries if shannon_entropy(q.qname) > 3.5]

        # 指標 3: TXT レコードの大量クエリ
        txt_queries = [q for q in queries if q.qtype == "TXT"]

        # 指標 4: 同一ドメインへの高頻度クエリ
        domain_counts = collections.Counter(
            extract_base_domain(q.qname) for q in queries
        )
        high_freq = {d: c for d, c in domain_counts.items() if c > 100}

        if long_queries or high_entropy or len(txt_queries) > 50 or high_freq:
            alerts.append({
                "source": src_ip,
                "indicators": {
                    "long_queries": len(long_queries),
                    "high_entropy": len(high_entropy),
                    "txt_queries": len(txt_queries),
                    "high_freq_domains": high_freq,
                },
            })

    return alerts
```

---

## 6. アンチパターン

### アンチパターン 1: DNSSEC 未導入のまま放置

```
NG: DNSSEC を設定せず平文 DNS のまま運用
  → キャッシュポイズニングで利用者をフィッシングサイトに誘導される

OK: DNSSEC を有効化し DS レコードをレジストラに登録
  → 改竄された応答は検証失敗で破棄される
```

**影響**: 中間者が DNS 応答を改竄できるため、正規ドメインで偽サイトに誘導可能。

### アンチパターン 2: 社内 DNS リゾルバの外部公開

```
NG: 社内リゾルバが 0.0.0.0:53 でリッスン
  → DNS アンプ攻撃の踏み台になる
  → 内部ドメイン情報が漏洩する

OK: リゾルバは社内ネットワークのみにバインド
  listen-on { 10.0.0.0/8; 127.0.0.1; };
  allow-recursion { 10.0.0.0/8; };
```

---

## 7. FAQ

### Q1. DNSSEC はなぜ普及が遅いのか?

DNSSEC は鍵管理の複雑さ、ゾーン署名の運用負荷、NSEC によるゾーンウォーキング (列挙攻撃) の懸念がある。また、応答サイズが大きくなり UDP フラグメンテーション問題が発生しうる。NSEC3 や自動署名 (BIND の inline-signing) の導入で改善されつつあるが、依然として導入障壁は高い。

### Q2. DoH を企業ネットワークで使うべきか?

DoH はプライバシーを向上させるが、企業のセキュリティ監視を迂回するリスクがある。企業ネットワークでは内部 DoH リゾルバを運用し、外部 DoH/DoT への通信をブロックするのが一般的である。これによりプライバシーと可視性を両立できる。

### Q3. DNS フィルタリングはセキュリティ対策として有効か?

RPZ や Pi-hole などによる DNS フィルタリングは、マルウェアの C2 通信やフィッシングサイトへのアクセスを低コストで防止できる有効な対策である。ただし、IP 直接アクセスや DoH バイパスに対しては無力であり、多層防御の一層として位置付けるべきである。

---

## まとめ

| 項目 | 要点 |
|------|------|
| DNS の脅威 | ポイズニング・スプーフィング・トンネリングが主要リスク |
| DNSSEC | 電子署名で応答の完全性を検証、信頼チェーンで root まで辿る |
| DoH/DoT | DNS クエリを暗号化しプライバシーと改竄防止を実現 |
| ポイズニング対策 | DNSSEC + ポートランダム化 + 0x20 エンコーディング |
| DNS トンネリング | クエリ長・エントロピー・頻度で異常を検知 |
| RPZ | ポリシーベースで悪意あるドメインをブロック |

---

## 次に読むべきガイド

- [ネットワークセキュリティ基礎](./00-network-security-basics.md) — ファイアウォール・IDS/IPS による網羅的な防御
- [APIセキュリティ](./02-api-security.md) — アプリケーション層での通信保護
- [監視/ログ](../06-operations/01-monitoring-logging.md) — DNS ログを含む統合的な監視体制

---

## 参考文献

1. **RFC 4033-4035 — DNS Security Introduction and Requirements (DNSSEC)** — https://datatracker.ietf.org/doc/html/rfc4033
2. **RFC 8484 — DNS Queries over HTTPS (DoH)** — https://datatracker.ietf.org/doc/html/rfc8484
3. **NIST SP 800-81-2 — Secure Domain Name System (DNS) Deployment Guide** — https://csrc.nist.gov/publications/detail/sp/800-81/2/final
4. **DNS Flag Day** — https://dnsflagday.net/ — DNS の最新標準準拠に関する業界イニシアチブ
