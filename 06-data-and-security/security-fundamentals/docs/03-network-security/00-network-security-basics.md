# ネットワークセキュリティ基礎

> ファイアウォール、IDS/IPS、VPN を中心に、ネットワーク層でのセキュリティ対策を体系的に学ぶ

## この章で学ぶこと

1. **ファイアウォールの種類と設定** — パケットフィルタリングからアプリケーション層ファイアウォールまでの防御技術
2. **IDS/IPS による侵入検知と防御** — ネットワーク上の不正トラフィックを検知・遮断する仕組み
3. **VPN によるセキュア通信** — IPsec と WireGuard による安全なネットワーク接続

---

## 1. ネットワークセキュリティの多層防御

### 防御の層

```
+----------------------------------------------------------+
|                    インターネット                           |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 1: 境界防御 (Perimeter)                             |
|  +-- ファイアウォール (WAF / NGFW)                         |
|  +-- DDoS 対策 (CloudFlare, AWS Shield)                   |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 2: ネットワークセグメンテーション                     |
|  +-- VLAN / VPC サブネット                                |
|  +-- マイクロセグメンテーション                              |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 3: 侵入検知/防止 (IDS/IPS)                         |
|  +-- シグネチャベース検知                                  |
|  +-- 異常検知 (Anomaly Detection)                         |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 4: ホストベース防御                                  |
|  +-- OS ファイアウォール (iptables / nftables)             |
|  +-- EDR (Endpoint Detection & Response)                 |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 5: アプリケーション層                                |
|  +-- TLS / 認証 / 入力検証                                |
+----------------------------------------------------------+
```

---

## 2. ファイアウォール

### ファイアウォールの種類

| 種類 | OSI 層 | 特徴 | 例 |
|------|--------|------|-----|
| パケットフィルタリング | L3-L4 | IP/ポートで許可・拒否 | iptables, ACL |
| ステートフル | L3-L4 | コネクション状態を追跡 | nftables, pf |
| アプリケーション GW | L7 | プロトコル内容を検査 | Squid, HAProxy |
| NGFW | L3-L7 | IPS + アプリ識別 + SSL 復号 | Palo Alto, FortiGate |
| WAF | L7 | HTTP 特化の攻撃防御 | AWS WAF, ModSecurity |

### iptables/nftables の設定例

```bash
# iptables: 基本的なサーバ設定
# デフォルトポリシー: すべて拒否
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# ループバックを許可
iptables -A INPUT -i lo -j ACCEPT

# 確立済みコネクションを許可
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# SSH (22), HTTP (80), HTTPS (443) を許可
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# ICMP (ping) をレート制限付きで許可
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s -j ACCEPT

# ログ記録後に拒否
iptables -A INPUT -j LOG --log-prefix "IPT_DROP: "
iptables -A INPUT -j DROP
```

### nftables (iptables 後継) の設定

```bash
#!/usr/sbin/nft -f
flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # 確立済みコネクション
        ct state established,related accept

        # ループバック
        iif lo accept

        # SSH (管理ネットワークのみ)
        tcp dport 22 ip saddr 10.0.0.0/8 accept

        # Web サービス
        tcp dport { 80, 443 } accept

        # ICMP
        icmp type echo-request limit rate 1/second accept

        # ログ & ドロップ
        log prefix "nft_drop: " drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

### AWS Security Group と NACL の比較

| 項目 | Security Group | Network ACL |
|------|---------------|-------------|
| 適用レベル | ENI (インスタンス) | サブネット |
| ステートフル | はい | いいえ |
| ルール | 許可のみ | 許可 + 拒否 |
| 評価順序 | 全ルール評価 | 番号順 |
| デフォルト | 全拒否 (インバウンド) | 全許可 |

```
VPC (10.0.0.0/16)
+---------------------------------------------------+
|  Public Subnet (10.0.1.0/24)                      |
|  [NACL: HTTP/HTTPS 許可, SSH 管理IPのみ]           |
|                                                   |
|  +-- ALB ----+                                    |
|  |  SG: 80,443|                                   |
|  +------------+                                   |
+---------------------------------------------------+
|  Private Subnet (10.0.2.0/24)                     |
|  [NACL: ALB からの通信のみ許可]                     |
|                                                   |
|  +-- App Server --+                               |
|  |  SG: ALB-SG:8080|                              |
|  +-----------------+                              |
+---------------------------------------------------+
|  Data Subnet (10.0.3.0/24)                        |
|  [NACL: App Subnet からのみ許可]                    |
|                                                   |
|  +-- RDS ----------+                              |
|  |  SG: App-SG:5432|                              |
|  +------------------+                             |
+---------------------------------------------------+
```

---

## 3. IDS/IPS

### IDS と IPS の違い

```
IDS (Intrusion Detection System):
  トラフィック --[コピー]--> IDS --> アラート
        |                            |
        v                            v
     通過 (遮断しない)           管理者に通知

IPS (Intrusion Prevention System):
  トラフィック ---> IPS ---> 通過 or 遮断
                    |
                    v
              アラート + 自動遮断
```

### Suricata (オープンソース IDS/IPS) の設定

```yaml
# /etc/suricata/suricata.yaml
%YAML 1.1
---
vars:
  address-groups:
    HOME_NET: "[10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16]"
    EXTERNAL_NET: "!$HOME_NET"

  port-groups:
    HTTP_PORTS: "80"
    HTTPS_PORTS: "443"

# ルールファイル
default-rule-path: /etc/suricata/rules
rule-files:
  - suricata.rules
  - local.rules

# IPS モード (AF_PACKET)
af-packet:
  - interface: eth0
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
```

### カスタムルールの作成

```bash
# /etc/suricata/rules/local.rules

# SQL インジェクション検知
alert http $EXTERNAL_NET any -> $HOME_NET $HTTP_PORTS (
    msg:"SQL Injection Attempt";
    flow:to_server,established;
    content:"UNION"; nocase;
    content:"SELECT"; nocase;
    sid:1000001; rev:1;
)

# SSH ブルートフォース検知
alert ssh $EXTERNAL_NET any -> $HOME_NET 22 (
    msg:"SSH Brute Force Attempt";
    flow:to_server;
    threshold: type both, track by_src, count 5, seconds 60;
    sid:1000002; rev:1;
)

# C2 通信の検知 (DNS トンネリング)
alert dns $HOME_NET any -> any any (
    msg:"Suspicious DNS Query Length";
    dns.query; content:"|00|"; offset:50;
    sid:1000003; rev:1;
)
```

---

## 4. VPN

### IPsec vs WireGuard

| 項目 | IPsec (IKEv2) | WireGuard |
|------|--------------|-----------|
| コード行数 | ~400,000 | ~4,000 |
| 暗号方式 | 交渉可能 (多数) | Noise Protocol (固定) |
| パフォーマンス | 中程度 | 高速 (カーネルモジュール) |
| 設定の複雑さ | 高い | 低い |
| プロトコル | ESP (IP プロトコル 50) | UDP |
| 状態管理 | 複雑 (SA, SPD) | ステートレス |
| モバイル対応 | IKEv2 MOBIKE | 組み込み |

### WireGuard の設定

```bash
# サーバ側の設定 (/etc/wireguard/wg0.conf)
[Interface]
PrivateKey = SERVER_PRIVATE_KEY
Address = 10.200.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
# クライアント A
PublicKey = CLIENT_A_PUBLIC_KEY
AllowedIPs = 10.200.0.2/32

[Peer]
# クライアント B
PublicKey = CLIENT_B_PUBLIC_KEY
AllowedIPs = 10.200.0.3/32
```

```bash
# クライアント側の設定
[Interface]
PrivateKey = CLIENT_PRIVATE_KEY
Address = 10.200.0.2/24
DNS = 10.200.0.1

[Peer]
PublicKey = SERVER_PUBLIC_KEY
Endpoint = vpn.example.com:51820
AllowedIPs = 0.0.0.0/0      # 全トラフィックをVPN経由
PersistentKeepalive = 25
```

```bash
# WireGuard の鍵生成と起動
wg genkey | tee privatekey | wg pubkey > publickey
sudo wg-quick up wg0
sudo wg show   # ステータス確認
```

---

## 5. ネットワークセグメンテーション

### ゼロトラストネットワーク

```
従来のモデル (城と堀):
+------------------------------------------------------+
|  信頼されたネットワーク (社内)                          |
|  +-- サーバA --- サーバB --- サーバC                   |
|  |   (自由に通信可能)                                 |
+------------------------------------------------------+
   ^ ファイアウォール (境界のみ)

ゼロトラスト:
+------------------------------------------------------+
|  全通信を検証                                         |
|  +-- サーバA --[mTLS + 認可]--> サーバB               |
|  |                                                   |
|  +-- サーバB --[mTLS + 認可]--> サーバC               |
|  |   (各通信でID検証 + 最小権限)                       |
+------------------------------------------------------+
   ^ マイクロセグメンテーション + ポリシーエンジン
```

---

## 6. アンチパターン

### アンチパターン 1: フラットネットワーク

```
NG: 全サーバが同一サブネット
  +-- Web Server --+
  +-- App Server --+-- 同一セグメント (10.0.1.0/24)
  +-- DB Server  --+
  (DBが直接インターネットから到達可能)

OK: サブネット分離
  Public:  10.0.1.0/24 -- ALB のみ
  Private: 10.0.2.0/24 -- App Server
  Data:    10.0.3.0/24 -- DB (App からのみ)
```

**影響**: 1 台が侵害されると水平移動 (Lateral Movement) で全サーバに到達される。

### アンチパターン 2: ファイアウォールの ANY-ANY ルール

```bash
# NG: 全通信を許可
iptables -A INPUT -j ACCEPT

# NG: 広すぎるセキュリティグループ
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol -1 --port -1 \
  --cidr 0.0.0.0/0

# OK: 必要最小限のポート・送信元のみ許可
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp --port 443 \
  --cidr 0.0.0.0/0
```

---

## 7. FAQ

### Q1. WAF と NGFW の違いは?

WAF は HTTP/HTTPS トラフィックに特化し、SQL インジェクションや XSS などの Web 攻撃を防御する。NGFW はネットワーク全体のトラフィックを L3-L7 で検査し、アプリケーション識別・IPS 機能を統合したものである。両者は補完関係にあり、Web サービスでは両方を配置するのが理想的である。

### Q2. IDS と IPS のどちらを導入すべきか?

本番環境では IPS をインラインに配置し自動遮断するのが推奨される。ただし誤検知による正常通信の遮断リスクがあるため、まず IDS モードで運用しルールをチューニングした後に IPS モードに移行するのが安全である。

### Q3. VPN と ゼロトラストは共存できるか?

できる。VPN はネットワーク層のアクセス制御として機能し、ゼロトラストはその上で各アクセスをリクエスト単位で検証する。ただし、VPN に接続しただけで社内ネットワーク全体にアクセスできる従来の運用はゼロトラストの思想に反する。VPN + マイクロセグメンテーション + ID ベース認可の組み合わせが現実的な解である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 多層防御 | 境界・ネットワーク・ホスト・アプリの各層で防御 |
| ファイアウォール | デフォルト拒否、必要最小限のルールのみ許可 |
| IDS/IPS | シグネチャ + 異常検知で不正トラフィックを識別 |
| VPN | WireGuard が設定の簡潔さとパフォーマンスで優位 |
| セグメンテーション | サブネット分離とマイクロセグメンテーションで水平移動を阻止 |
| ゼロトラスト | すべての通信を検証し、暗黙の信頼を排除 |

---

## 次に読むべきガイド

- [DNSセキュリティ](./01-dns-security.md) — DNS 層での攻撃と対策
- [APIセキュリティ](./02-api-security.md) — アプリケーション層の API 保護
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) — 暗号化通信の基盤技術

---

## 参考文献

1. **NIST SP 800-41 Rev.1 — Guidelines on Firewalls and Firewall Policy** — https://csrc.nist.gov/publications/detail/sp/800-41/rev-1/final
2. **NIST SP 800-207 — Zero Trust Architecture** — https://csrc.nist.gov/publications/detail/sp/800-207/final
3. **Suricata Documentation** — https://docs.suricata.io/en/latest/
4. **WireGuard — Conceptual Overview** — https://www.wireguard.com/papers/wireguard.pdf
