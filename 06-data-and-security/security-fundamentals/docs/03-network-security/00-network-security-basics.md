# ネットワークセキュリティ基礎

> ファイアウォール、IDS/IPS、VPN を中心に、ネットワーク層でのセキュリティ対策を体系的に学ぶ

## この章で学ぶこと

1. **ファイアウォールの種類と設定** — パケットフィルタリングからアプリケーション層ファイアウォールまでの防御技術
2. **IDS/IPS による侵入検知と防御** — ネットワーク上の不正トラフィックを検知・遮断する仕組み
3. **VPN によるセキュア通信** — IPsec と WireGuard による安全なネットワーク接続
4. **ゼロトラストアーキテクチャ** — 境界防御を超えた現代的なネットワークセキュリティモデル

## 前提知識

- TCP/IP の基本 (IP アドレス、ポート番号、3 ウェイハンドシェイク)
- OSI 参照モデルの各層の役割
- Linux の基本操作 (コマンドライン、ファイル編集)
- クラウドサービス (AWS/GCP) の基本概念

---

## 1. ネットワークセキュリティの多層防御

### 防御の層

ネットワークセキュリティは単一の防御メカニズムに依存してはならない。NIST SP 800-207 および MITRE ATT&CK フレームワークが示すように、攻撃者は複数のフェーズ (偵察 → 初期アクセス → 横展開 → 目的達成) を経て侵入するため、各フェーズに対応した多層防御 (Defense in Depth) が必要である。

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
|  +-- BGP フィルタリング / RPKI                             |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 2: ネットワークセグメンテーション                     |
|  +-- VLAN / VPC サブネット                                |
|  +-- マイクロセグメンテーション                              |
|  +-- East-West トラフィック制御                             |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 3: 侵入検知/防止 (IDS/IPS)                         |
|  +-- シグネチャベース検知                                  |
|  +-- 異常検知 (Anomaly Detection)                         |
|  +-- TLS インスペクション                                  |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 4: ホストベース防御                                  |
|  +-- OS ファイアウォール (iptables / nftables)             |
|  +-- EDR (Endpoint Detection & Response)                 |
|  +-- ファイル整合性監視 (AIDE / OSSEC)                     |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|  Layer 5: アプリケーション層                                |
|  +-- TLS / mTLS                                          |
|  +-- 認証・認可 (OAuth 2.0 / OIDC)                       |
|  +-- 入力検証 / サニタイゼーション                          |
+----------------------------------------------------------+
```

### 多層防御が必要な理由

```
攻撃シナリオ: 各層の防御が段階的に攻撃を阻止する

攻撃者 → [L1: WAF がSQLi検知] → ブロック (80%の攻撃はここで止まる)
         ↓ (WAF回避に成功)
        [L2: セグメンテーション] → DBサブネットへの直接アクセス不可
         ↓ (Webサーバを侵害)
        [L3: IPS がC2通信検知] → 外向きの不審通信を遮断
         ↓ (暗号化C2で回避)
        [L4: EDR が不審プロセス検知] → マルウェア実行をブロック
         ↓ (Living off the Land)
        [L5: mTLS が無効な証明書を拒否] → 他サービスへのアクセス不可

→ 1層が突破されても次の層で阻止される
```

### MITRE ATT&CK マッピング

| ATT&CK フェーズ | 防御レイヤー | 防御メカニズム |
|-----------------|------------|--------------|
| Reconnaissance | L1 境界 | レート制限、ポートスキャン検知 |
| Initial Access | L1 境界 | WAF、IPS シグネチャ |
| Execution | L4 ホスト | EDR、アプリケーションホワイトリスト |
| Persistence | L4 ホスト | ファイル整合性監視、auditd |
| Lateral Movement | L2 セグメンテーション | マイクロセグメンテーション、mTLS |
| Exfiltration | L3 IDS/IPS | DLP、DNS トンネリング検知 |
| C2 | L3 IDS/IPS | 異常検知、ドメインレピュテーション |

---

## 2. ファイアウォール

### ファイアウォールの種類と比較

| 種類 | OSI 層 | 特徴 | 処理速度 | 検査深度 | 例 |
|------|--------|------|---------|---------|-----|
| パケットフィルタリング | L3-L4 | IP/ポートで許可・拒否 | 最速 | 浅い | iptables, ACL |
| ステートフル | L3-L4 | コネクション状態を追跡 | 速い | 中程度 | nftables, pf |
| アプリケーション GW | L7 | プロトコル内容を検査 | 中程度 | 深い | Squid, HAProxy |
| NGFW | L3-L7 | IPS + アプリ識別 + SSL 復号 | 遅い | 最深 | Palo Alto, FortiGate |
| WAF | L7 | HTTP 特化の攻撃防御 | 中程度 | HTTP限定 | AWS WAF, ModSecurity |

### パケットフィルタリングの内部動作

```
受信パケット
    |
    v
+-- ヘッダ解析 --+
| 送信元 IP      |
| 宛先 IP        |
| プロトコル      |
| 送信元ポート    |
| 宛先ポート      |
+----------------+
    |
    v
+-- ルールテーブル (上から順に評価) --+
| Rule 1: ALLOW 10.0.0.0/8 → :22   |  ← マッチ → ACCEPT
| Rule 2: ALLOW any → :80           |  ← マッチ → ACCEPT
| Rule 3: ALLOW any → :443          |  ← マッチ → ACCEPT
| Rule N: DROP any → any (デフォルト)|  ← 最終ルール
+------------------------------------+
    |
    v
  ACCEPT / DROP / REJECT / LOG
```

### ステートフルインスペクションの仕組み

```
コネクション追跡テーブル (conntrack):

+------+----------+-----------+--------+-------+--------+--------+
| ID   | Protocol | Src IP    | Src    | Dst IP| Dst    | State  |
|      |          |           | Port   |       | Port   |        |
+------+----------+-----------+--------+-------+--------+--------+
| 1    | TCP      |192.168.1.5| 52341  | Web   | 443    | ESTAB  |
| 2    | TCP      |10.0.1.100 | 38912  | DB    | 5432   | ESTAB  |
| 3    | UDP      |192.168.1.5| 51234  | DNS   | 53     | NEW    |
+------+----------+-----------+--------+-------+--------+--------+

状態遷移:
  NEW → SYN パケット受信
  ESTABLISHED → SYN-ACK 確認後 (双方向通信成立)
  RELATED → 関連コネクション (FTP データ、ICMP エラー)
  INVALID → 不正な状態遷移 → DROP

利点:
  - 戻りパケットを自動許可 (個別ルール不要)
  - SYN flood 等の異常を状態遷移で検知
  - パケットフィルタリングより安全
```

### iptables/nftables の設定例

```bash
# iptables: 本番サーバの実践的設定
# ============================================

# ルールの初期化
iptables -F
iptables -X
iptables -Z

# デフォルトポリシー: すべて拒否
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# ループバックを許可
iptables -A INPUT -i lo -j ACCEPT

# 確立済みコネクションを許可 (ステートフル)
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# 不正なパケットを拒否 (INVALID状態)
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# SSH (22) を管理ネットワークからのみ許可 + レート制限
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 \
  -m conntrack --ctstate NEW \
  -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 \
  -m conntrack --ctstate NEW \
  -m recent --update --seconds 60 --hitcount 4 --name SSH \
  -j DROP
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT

# HTTP (80), HTTPS (443) を許可
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# ICMP (ping) をレート制限付きで許可
iptables -A INPUT -p icmp --icmp-type echo-request \
  -m limit --limit 1/s --limit-burst 4 -j ACCEPT

# SYN flood 対策
iptables -A INPUT -p tcp --syn \
  -m limit --limit 25/s --limit-burst 50 -j ACCEPT
iptables -A INPUT -p tcp --syn -j DROP

# ポートスキャン検知 (NULL スキャン)
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j LOG --log-prefix "NULL_SCAN: "
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP

# Xmas スキャン検知
iptables -A INPUT -p tcp --tcp-flags ALL FIN,PSH,URG -j LOG --log-prefix "XMAS_SCAN: "
iptables -A INPUT -p tcp --tcp-flags ALL FIN,PSH,URG -j DROP

# ログ記録後に拒否 (最終ルール)
iptables -A INPUT -j LOG --log-prefix "IPT_DROP: " --log-level 4
iptables -A INPUT -j DROP
```

### nftables (iptables 後継) の実践的設定

```bash
#!/usr/sbin/nft -f
flush ruleset

# テーブルとチェインの定義
table inet filter {
    # IP セット定義 (動的にブロック対象を管理)
    set blocklist {
        type ipv4_addr
        flags timeout
        timeout 1h
    }

    set ratelimit_ssh {
        type ipv4_addr
        flags dynamic, timeout
        timeout 5m
    }

    # 入力チェイン
    chain input {
        type filter hook input priority 0; policy drop;

        # ブロックリスト
        ip saddr @blocklist counter drop

        # 確立済みコネクション
        ct state established,related accept

        # 不正パケット
        ct state invalid counter drop

        # ループバック
        iif lo accept

        # ICMP (レート制限)
        icmp type echo-request limit rate 2/second burst 5 packets accept
        icmpv6 type { echo-request, nd-neighbor-solicit, nd-router-advert } accept

        # SSH (管理ネットワーク + レート制限)
        tcp dport 22 ip saddr 10.0.0.0/8 ct state new \
            add @ratelimit_ssh { ip saddr limit rate 3/minute } accept

        # Web サービス
        tcp dport { 80, 443 } accept

        # SYN flood 対策
        tcp flags syn limit rate 25/second burst 50 accept
        tcp flags syn counter drop

        # ログ & ドロップ (最終ルール)
        log prefix "nft_drop: " counter drop
    }

    # 転送チェイン
    chain forward {
        type filter hook forward priority 0; policy drop;

        # Docker / コンテナ用の転送ルール
        iifname "docker0" oifname "eth0" accept
        iifname "eth0" oifname "docker0" ct state established,related accept
    }

    # 出力チェイン
    chain output {
        type filter hook output priority 0; policy accept;

        # 外向き DNS の制限 (DNS トンネリング対策)
        udp dport 53 ip daddr != { 10.0.0.2, 8.8.8.8 } counter drop
    }
}
```

### iptables と nftables の比較

| 項目 | iptables | nftables |
|------|----------|---------|
| カーネル統合 | xtables フレームワーク | nf_tables フレームワーク |
| ルール構文 | コマンド引数形式 | 構造化されたDSL |
| テーブル | ip/ip6/arp/bridge 別 | inet (IPv4/IPv6 統合) |
| セット | ipset (別ツール) | 組み込みセット |
| アトミック更新 | iptables-restore | nft -f (デフォルト) |
| パフォーマンス | 線形ルール評価 | 最適化されたルックアップ |
| 動的ルール | 困難 | set + map で容易 |
| 状態 | 現行 (レガシー) | 推奨 (後継) |

### AWS Security Group と NACL の比較

| 項目 | Security Group | Network ACL |
|------|---------------|-------------|
| 適用レベル | ENI (インスタンス) | サブネット |
| ステートフル | はい (戻り通信自動許可) | いいえ (明示的に許可必要) |
| ルール | 許可のみ | 許可 + 拒否 |
| 評価順序 | 全ルール評価 (OR) | 番号順 (最初にマッチ) |
| デフォルト | 全拒否 (インバウンド) | 全許可 |
| ルール数上限 | 60 (インバウンド+アウトバウンド) | 20 (各方向) |
| 適用タイミング | インスタンス起動時 | サブネット作成時 |

```
VPC (10.0.0.0/16)
+---------------------------------------------------+
|  Public Subnet (10.0.1.0/24)                      |
|  [NACL: HTTP/HTTPS 許可, SSH 管理IPのみ]           |
|  [NACL: エフェメラルポート 1024-65535 OUT許可]      |
|                                                   |
|  +-- ALB ----+                                    |
|  |  SG: 80,443 from 0.0.0.0/0                    |
|  |  SG: 443 out to App-SG                        |
|  +------------+                                   |
+---------------------------------------------------+
|  Private Subnet (10.0.2.0/24)                     |
|  [NACL: ALB Subnet からの 8080 のみ許可]           |
|  [NACL: NAT GW への OUT 許可]                      |
|                                                   |
|  +-- App Server --+                               |
|  |  SG: 8080 from ALB-SG only                    |
|  |  SG: 5432 out to DB-SG                        |
|  +-----------------+                              |
+---------------------------------------------------+
|  Data Subnet (10.0.3.0/24)                        |
|  [NACL: App Subnet からの 5432 のみ許可]           |
|  [NACL: OUT は App Subnet のみ]                    |
|                                                   |
|  +-- RDS ----------+                              |
|  |  SG: 5432 from App-SG only                    |
|  |  SG: No outbound rules                        |
|  +------------------+                             |
+---------------------------------------------------+
```

### VPC フローログによるトラフィック監視

```python
import boto3
import gzip
import json
from datetime import datetime, timedelta

# VPC フローログの有効化 (Terraform)
"""
resource "aws_flow_log" "vpc_flow" {
  vpc_id          = aws_vpc.main.id
  traffic_type    = "ALL"
  iam_role_arn    = aws_iam_role.flow_log.arn
  log_destination = aws_cloudwatch_log_group.flow_log.arn

  # カスタムフォーマット (追加フィールド)
  log_format = "$${version} $${account-id} $${interface-id} $${srcaddr} $${dstaddr} $${srcport} $${dstport} $${protocol} $${packets} $${bytes} $${start} $${end} $${action} $${log-status} $${tcp-flags} $${flow-direction}"
}
"""

# Athena でフローログを分析: 不審なトラフィックパターン
SUSPICIOUS_TRAFFIC_QUERY = """
-- 大量の REJECT されたトラフィック (ポートスキャンの兆候)
SELECT
    srcaddr,
    dstport,
    COUNT(*) as reject_count,
    SUM(bytes) as total_bytes
FROM vpc_flow_logs
WHERE action = 'REJECT'
  AND start > to_unixtime(now() - interval '1' hour)
GROUP BY srcaddr, dstport
HAVING COUNT(*) > 100
ORDER BY reject_count DESC
LIMIT 50;
"""

# 異常な外向きトラフィック検知 (データ流出の兆候)
DATA_EXFIL_QUERY = """
SELECT
    srcaddr,
    dstaddr,
    dstport,
    SUM(bytes) as total_bytes,
    COUNT(*) as flow_count
FROM vpc_flow_logs
WHERE flow_direction = 'egress'
  AND dstaddr NOT LIKE '10.%'
  AND dstaddr NOT LIKE '172.16.%'
  AND start > to_unixtime(now() - interval '1' hour)
GROUP BY srcaddr, dstaddr, dstport
HAVING SUM(bytes) > 1073741824  -- 1GB超
ORDER BY total_bytes DESC;
"""
```

---

## 3. IDS/IPS

### IDS と IPS の違い

```
IDS (Intrusion Detection System) - パッシブモード:
  ネットワーク
  トラフィック ----+----> 宛先 (通信はそのまま通過)
                  |
                  | (ミラーリング/TAP)
                  v
                IDS エンジン
                  |
                  v
            アラート発報 → SIEM → SOC チーム

IPS (Intrusion Prevention System) - インラインモード:
  ネットワーク
  トラフィック ----> IPS エンジン ----> 宛先
                      |
                      +-- 正常 → 通過
                      +-- 不正 → 遮断 + アラート
                      +-- 疑わしい → ログ + 通過 (タグ付け)

ハイブリッドモード (推奨):
  トラフィック ----> IPS (インライン) ----> 宛先
                      |
                      | (コピー)
                      v
                    IDS (検知のみ)
                      |
                      v
                高度な分析 (ML/AI)
```

### 検知方式の比較

| 方式 | 仕組み | 長所 | 短所 |
|------|--------|------|------|
| シグネチャベース | 既知の攻撃パターンとマッチング | 精度が高い、誤検知少ない | 未知の攻撃を検知不可 |
| 異常検知 (Anomaly) | 正常トラフィックの統計モデルとの偏差 | 未知の攻撃を検知可能 | 誤検知が多い、学習期間必要 |
| プロトコル分析 | RFC 準拠のプロトコル動作と比較 | プロトコル異常を確実に検知 | カバー範囲がプロトコル限定 |
| ヒューリスティック | 複数の弱いシグナルを組み合わせ | 回避が困難 | チューニングが複雑 |
| ML/AI ベース | 機械学習モデルによる分類 | 適応的、パターン認識 | ブラックボックス、計算コスト |

### Suricata (オープンソース IDS/IPS) の実践設定

```yaml
# /etc/suricata/suricata.yaml (本番環境向け設定)
%YAML 1.1
---

# ネットワーク定義
vars:
  address-groups:
    HOME_NET: "[10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16]"
    EXTERNAL_NET: "!$HOME_NET"
    DNS_SERVERS: "[10.0.0.2/32]"
    HTTP_SERVERS: "[10.0.1.0/24]"
    SQL_SERVERS: "[10.0.3.0/24]"

  port-groups:
    HTTP_PORTS: "[80, 8080, 8443]"
    HTTPS_PORTS: "443"
    DNS_PORTS: "53"
    SSH_PORTS: "22"

# パフォーマンスチューニング
threading:
  set-cpu-affinity: yes
  detect-thread-ratio: 1.5

# 検知エンジン
detect:
  profile: high      # high / medium / low
  custom-values:
    toclient-groups: 50
    toserver-groups: 50
  sgh-mpm-context: auto
  inspection-recursion-limit: 3000

# EVE JSON ログ (SIEM 連携)
outputs:
  - eve-log:
      enabled: yes
      filetype: regular
      filename: /var/log/suricata/eve.json
      types:
        - alert:
            tagged-packets: yes
            metadata: yes
        - http:
            extended: yes
        - dns:
            query: yes
            answer: yes
        - tls:
            extended: yes
            session-resumption: yes
        - flow
        - stats:
            totals: yes
            threads: yes

# ルールファイル
default-rule-path: /etc/suricata/rules
rule-files:
  - suricata.rules          # ET Open ルール
  - local.rules             # カスタムルール
  - emerging-exploit.rules  # エクスプロイト検知
  - emerging-malware.rules  # マルウェア検知

# IPS モード設定 (AF_PACKET インラインモード)
af-packet:
  - interface: eth0
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
    use-mmap: yes
    ring-size: 200000
    block-size: 262144
    copy-mode: ips           # IPS モード
    copy-iface: eth1         # 出力インターフェース
```

### カスタムルールの作成 (実践)

```bash
# /etc/suricata/rules/local.rules
# ============================================

# --- SQL インジェクション検知 (高精度) ---
alert http $EXTERNAL_NET any -> $HTTP_SERVERS $HTTP_PORTS (
    msg:"SQL Injection - UNION SELECT";
    flow:to_server,established;
    http.uri;
    content:"UNION"; nocase;
    content:"SELECT"; nocase; distance:0;
    pcre:"/UNION\s+(ALL\s+)?SELECT/Ui";
    classtype:web-application-attack;
    sid:1000001; rev:2;
    metadata:attack_target web_server, deployment perimeter;
)

# --- SQL インジェクション (Time-based Blind) ---
alert http $EXTERNAL_NET any -> $HTTP_SERVERS $HTTP_PORTS (
    msg:"SQL Injection - Time-based Blind (SLEEP/BENCHMARK)";
    flow:to_server,established;
    http.uri;
    pcre:"/(?:SLEEP|BENCHMARK|WAITFOR\s+DELAY|pg_sleep)\s*\(/Ui";
    classtype:web-application-attack;
    sid:1000010; rev:1;
)

# --- SSH ブルートフォース検知 ---
alert ssh $EXTERNAL_NET any -> $HOME_NET $SSH_PORTS (
    msg:"SSH Brute Force Attempt";
    flow:to_server;
    threshold: type both, track by_src, count 5, seconds 60;
    classtype:attempted-admin;
    sid:1000002; rev:1;
)

# --- C2 通信の検知 (DNS トンネリング) ---
alert dns $HOME_NET any -> any $DNS_PORTS (
    msg:"Possible DNS Tunneling - Long Query";
    dns.query;
    pcre:"/^.{50,}\./";     # 50文字以上のサブドメイン
    threshold: type both, track by_src, count 10, seconds 60;
    classtype:trojan-activity;
    sid:1000003; rev:2;
)

# --- DNS トンネリング (高エントロピー) ---
alert dns $HOME_NET any -> any $DNS_PORTS (
    msg:"Possible DNS Tunneling - High Entropy Subdomain";
    dns.query;
    # Base64/Hex エンコードされた長いサブドメイン
    pcre:"/^[a-zA-Z0-9+\/=]{30,}\./";
    threshold: type both, track by_src, count 5, seconds 120;
    classtype:trojan-activity;
    sid:1000004; rev:1;
)

# --- 暗号通貨マイニング通信検知 ---
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"Cryptomining - Stratum Protocol over TLS";
    flow:to_server,established;
    tls.sni; content:"pool"; nocase;
    pcre:"/(?:mining|pool|stratum|xmr|monero|coinhive)/i";
    classtype:trojan-activity;
    sid:1000005; rev:1;
)

# --- 不審なユーザエージェント ---
alert http $HOME_NET any -> $EXTERNAL_NET any (
    msg:"Suspicious User-Agent - Possible Bot/Scanner";
    flow:to_server,established;
    http.user_agent;
    pcre:"/^(?:python-requests|curl|wget|nikto|sqlmap|nmap)/i";
    classtype:web-application-attack;
    sid:1000006; rev:1;
)

# --- ICMP トンネリング検知 ---
alert icmp $HOME_NET any -> $EXTERNAL_NET any (
    msg:"Possible ICMP Tunneling - Large Payload";
    icode:0;
    itype:8;
    dsize:>800;    # 通常のpingは64バイト
    threshold: type both, track by_src, count 5, seconds 60;
    classtype:trojan-activity;
    sid:1000007; rev:1;
)

# --- Lateral Movement: SMB/RPC ---
alert tcp $HOME_NET any -> $HOME_NET [139,445] (
    msg:"Possible Lateral Movement - Internal SMB";
    flow:to_server,established;
    content:"|FF|SMB"; offset:4; depth:4;
    classtype:attempted-admin;
    sid:1000008; rev:1;
)
```

### Suricata のアラート分析と自動対応

```python
import json
import subprocess
from datetime import datetime
from collections import defaultdict
from pathlib import Path

class SuricataAlertAnalyzer:
    """Suricata の EVE JSON ログを分析し自動対応を行う"""

    def __init__(self, eve_log_path: str = "/var/log/suricata/eve.json"):
        self.eve_log_path = Path(eve_log_path)
        self.alert_counts = defaultdict(int)
        self.blocked_ips = set()

    def parse_alerts(self, minutes: int = 5) -> list[dict]:
        """直近N分のアラートを解析"""
        alerts = []
        cutoff = datetime.utcnow().timestamp() - (minutes * 60)

        with open(self.eve_log_path) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get("event_type") != "alert":
                        continue
                    ts = datetime.fromisoformat(
                        event["timestamp"].replace("Z", "+00:00")
                    ).timestamp()
                    if ts >= cutoff:
                        alerts.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue
        return alerts

    def aggregate_by_source(self, alerts: list[dict]) -> dict:
        """送信元IPごとにアラートを集約"""
        by_src = defaultdict(lambda: {"count": 0, "signatures": set(), "severity": 0})
        for alert in alerts:
            src = alert.get("src_ip", "unknown")
            sig = alert["alert"]["signature"]
            severity = alert["alert"].get("severity", 3)
            by_src[src]["count"] += 1
            by_src[src]["signatures"].add(sig)
            by_src[src]["severity"] = max(by_src[src]["severity"], severity)
        return dict(by_src)

    def auto_block(self, ip: str, duration_hours: int = 1):
        """nftables で IP を動的ブロック"""
        if ip in self.blocked_ips:
            return
        subprocess.run([
            "nft", "add", "element", "inet", "filter",
            "blocklist", f"{{ {ip} timeout {duration_hours}h }}"
        ], check=True)
        self.blocked_ips.add(ip)
        print(f"[BLOCK] {ip} を {duration_hours}時間ブロック")

    def respond_to_alerts(self):
        """アラートに基づく自動対応"""
        alerts = self.parse_alerts(minutes=5)
        by_src = self.aggregate_by_source(alerts)

        for ip, info in by_src.items():
            # 高重大度 or 大量アラート → 自動ブロック
            if info["severity"] <= 1 or info["count"] >= 20:
                self.auto_block(ip, duration_hours=4)
            elif info["count"] >= 10:
                self.auto_block(ip, duration_hours=1)


# 使用例
if __name__ == "__main__":
    analyzer = SuricataAlertAnalyzer()
    analyzer.respond_to_alerts()
```

### Snort vs Suricata の比較

| 項目 | Snort 3 | Suricata |
|------|---------|----------|
| マルチスレッド | 対応 (3.x~) | ネイティブ対応 |
| プロトコル解析 | AppID | 組み込み (HTTP, TLS, DNS等) |
| ルール互換性 | Snort固有 | Snort互換 + 独自拡張 |
| 出力形式 | Unified2, JSON | EVE JSON (構造化) |
| TLS インスペクション | 対応 | 対応 (JA3/JA4 fingerprint) |
| ファイル抽出 | 対応 | 対応 |
| Lua スクリプト | 対応 | 対応 |
| ライセンス | GPL v2 | GPL v2 |
| コミュニティ | Cisco主導 | OISF主導 (オープン) |

---

## 4. VPN

### VPN プロトコルの全体比較

| 項目 | IPsec (IKEv2) | WireGuard | OpenVPN | L2TP/IPsec |
|------|--------------|-----------|---------|-----------|
| コード行数 | ~400,000 | ~4,000 | ~100,000 | ~400,000 |
| 暗号方式 | 交渉可能 (多数) | Noise Protocol (固定) | OpenSSL (多数) | IPsec依存 |
| パフォーマンス | 中程度 | 最高速 | 中程度 | 低い |
| 設定の複雑さ | 高い | 低い | 中程度 | 高い |
| プロトコル | ESP (IP 50) | UDP | UDP/TCP | UDP 1701 |
| 状態管理 | 複雑 (SA, SPD) | ステートレス | TLS セッション | 複雑 |
| モバイル対応 | IKEv2 MOBIKE | 組み込み | アプリ必要 | OS 組み込み |
| NAT 越え | NAT-T (UDP 4500) | ネイティブ (UDP) | ネイティブ | 困難な場合あり |
| 監査性 | 難 (複雑) | 容易 (小さい) | 中程度 | 難 |
| PFS | 設定依存 | 常時有効 | 設定依存 | 設定依存 |

### WireGuard の内部動作

```
WireGuard Noise Protocol IK ハンドシェイク:

イニシエータ (Client)                    レスポンダ (Server)
    |                                        |
    |  1. Handshake Initiation               |
    |  [sender_index, ephemeral_pub,         |
    |   encrypted_static, encrypted_ts]      |
    |  ----→  (Noise_IKpsk2)  ----→          |
    |                                        |
    |  2. Handshake Response                 |
    |  [sender_index, receiver_index,        |
    |   ephemeral_pub, encrypted_nothing]    |
    |  ←----  (Noise_IKpsk2)  ←----         |
    |                                        |
    |  === トランスポートデータ ===            |
    |  3. Transport Data                     |
    |  [receiver_index, counter,             |
    |   encrypted_payload (ChaCha20Poly1305)]|
    |  ←============================→        |
    |                                        |

暗号プリミティブ (固定、交渉なし):
  - Curve25519 (ECDH)
  - ChaCha20-Poly1305 (AEAD)
  - BLAKE2s (ハッシュ)
  - SipHash (ハッシュテーブル鍵)
  - HKDF (鍵導出)

利点: 暗号アジリティを排除 → ダウングレード攻撃が不可能
```

### WireGuard の設定 (サーバ)

```bash
# 鍵生成
wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key
chmod 600 /etc/wireguard/server_private.key

# サーバ側の設定 (/etc/wireguard/wg0.conf)
[Interface]
PrivateKey = SERVER_PRIVATE_KEY
Address = 10.200.0.1/24
ListenPort = 51820

# IP フォワーディングとNAT
PostUp = sysctl -w net.ipv4.ip_forward=1
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostUp = iptables -A INPUT -p udp --dport 51820 -j ACCEPT
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D INPUT -p udp --dport 51820 -j ACCEPT

# DNS (Unbound をローカルDNSとして使用)
# PostUp = systemctl start unbound

[Peer]
# クライアント A (開発者)
PublicKey = CLIENT_A_PUBLIC_KEY
AllowedIPs = 10.200.0.2/32
# PresharedKey = PSK_A  # 量子耐性のための追加層

[Peer]
# クライアント B (運用チーム)
PublicKey = CLIENT_B_PUBLIC_KEY
AllowedIPs = 10.200.0.3/32
```

### WireGuard の設定 (クライアント)

```bash
# クライアント側の設定
[Interface]
PrivateKey = CLIENT_PRIVATE_KEY
Address = 10.200.0.2/24
DNS = 10.200.0.1

# Kill Switch (VPN切断時に全通信を遮断)
PostUp = iptables -I OUTPUT ! -o wg0 -m mark ! --mark $(wg show wg0 fwmark) -m addrtype ! --dst-type LOCAL -j REJECT
PreDown = iptables -D OUTPUT ! -o wg0 -m mark ! --mark $(wg show wg0 fwmark) -m addrtype ! --dst-type LOCAL -j REJECT

[Peer]
PublicKey = SERVER_PUBLIC_KEY
Endpoint = vpn.example.com:51820
AllowedIPs = 0.0.0.0/0, ::/0   # 全トラフィックをVPN経由
PersistentKeepalive = 25
```

### WireGuard ピア管理の自動化

```python
import subprocess
import ipaddress
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WireGuardPeer:
    name: str
    public_key: str
    allowed_ips: str
    preshared_key: str | None = None

class WireGuardManager:
    """WireGuard ピアの追加・削除・一覧を管理"""

    def __init__(self, interface: str = "wg0"):
        self.interface = interface
        self.config_path = Path(f"/etc/wireguard/{interface}.conf")
        self.next_ip = ipaddress.IPv4Address("10.200.0.10")

    def generate_peer_config(self, peer_name: str) -> tuple[str, str]:
        """新しいピアの鍵ペアとクライアント設定を生成"""
        # 鍵生成
        private_key = subprocess.run(
            ["wg", "genkey"], capture_output=True, text=True
        ).stdout.strip()
        public_key = subprocess.run(
            ["wg", "pubkey"], input=private_key,
            capture_output=True, text=True
        ).stdout.strip()

        # PSK 生成 (量子耐性)
        psk = subprocess.run(
            ["wg", "genpsk"], capture_output=True, text=True
        ).stdout.strip()

        peer_ip = str(self.next_ip)
        self.next_ip += 1

        # サーバ設定に追加
        server_peer_config = f"""
[Peer]
# {peer_name}
PublicKey = {public_key}
PresharedKey = {psk}
AllowedIPs = {peer_ip}/32
"""

        # クライアント設定を生成
        server_pub = self._get_server_public_key()
        client_config = f"""[Interface]
PrivateKey = {private_key}
Address = {peer_ip}/24
DNS = 10.200.0.1

[Peer]
PublicKey = {server_pub}
PresharedKey = {psk}
Endpoint = vpn.example.com:51820
AllowedIPs = 10.0.0.0/8
PersistentKeepalive = 25
"""
        # サーバにピアを動的追加
        subprocess.run([
            "wg", "set", self.interface,
            "peer", public_key,
            "preshared-key", "/dev/stdin",
            "allowed-ips", f"{peer_ip}/32",
        ], input=psk, text=True, check=True)

        return server_peer_config, client_config

    def revoke_peer(self, public_key: str):
        """ピアを無効化"""
        subprocess.run([
            "wg", "set", self.interface,
            "peer", public_key, "remove",
        ], check=True)

    def list_peers(self) -> list[dict]:
        """接続中のピア一覧"""
        result = subprocess.run(
            ["wg", "show", self.interface, "dump"],
            capture_output=True, text=True,
        )
        peers = []
        for line in result.stdout.strip().split("\n")[1:]:  # ヘッダスキップ
            parts = line.split("\t")
            if len(parts) >= 7:
                peers.append({
                    "public_key": parts[0],
                    "endpoint": parts[2],
                    "allowed_ips": parts[3],
                    "latest_handshake": parts[4],
                    "transfer_rx": parts[5],
                    "transfer_tx": parts[6],
                })
        return peers

    def _get_server_public_key(self) -> str:
        result = subprocess.run(
            ["wg", "show", self.interface, "public-key"],
            capture_output=True, text=True,
        )
        return result.stdout.strip()
```

### IPsec (strongSwan) の設定例

```bash
# /etc/swanctl/swanctl.conf (strongSwan 5.x+)
connections {
    site-to-site {
        version = 2           # IKEv2
        local_addrs = 203.0.113.1
        remote_addrs = 198.51.100.1

        local {
            auth = pubkey
            certs = server.cert.pem
            id = vpn.example.com
        }
        remote {
            auth = pubkey
            id = remote.example.com
        }

        children {
            net-net {
                local_ts = 10.0.0.0/16     # ローカルネットワーク
                remote_ts = 172.16.0.0/16   # リモートネットワーク
                esp_proposals = aes256gcm128-x25519
                dpd_action = restart
                start_action = trap   # オンデマンド接続
            }
        }

        proposals = aes256-sha256-x25519
        rekey_time = 4h
        dpd_delay = 30s
    }
}
```

---

## 5. ネットワークセグメンテーション

### ゼロトラストネットワーク

```
従来のモデル (城と堀 / Castle-and-Moat):
+------------------------------------------------------+
|  "信頼された" ネットワーク (社内LAN)                    |
|                                                      |
|  +-- サーバA --- サーバB --- サーバC                   |
|  |   (自由に通信可能、認証なし)                        |
|  |                                                   |
|  +-- 開発PC --- 管理PC --- IoT デバイス              |
|  |   (全て同一ネットワーク)                            |
|  |                                                   |
|  問題: 1台が侵害されると全体に到達可能                  |
|  (Lateral Movement / 水平展開が容易)                   |
+------------------------------------------------------+
   ^ ファイアウォール (境界のみ)

ゼロトラスト (BeyondCorp / NIST SP 800-207):
+------------------------------------------------------+
|  全通信を検証 (暗黙の信頼なし)                          |
|                                                      |
|  +-- サーバA --[mTLS + 認可]--> サーバB               |
|  |   (サービスメッシュ: Istio / Linkerd)              |
|  |                                                   |
|  +-- サーバB --[mTLS + 認可]--> サーバC               |
|  |                                                   |
|  各通信で:                                            |
|  1. ID検証 (mTLS / SPIFFE)                           |
|  2. デバイス検証 (パッチレベル、EDR有無)               |
|  3. コンテキスト検証 (時間、場所、リスクスコア)         |
|  4. 最小権限 (リクエスト単位のポリシー)                 |
+------------------------------------------------------+
   ^ マイクロセグメンテーション + ポリシーエンジン (OPA)
```

### マイクロセグメンテーションの実装パターン

```
方式1: ネットワークベース (VLAN / VPC)
+--------+    +--------+    +--------+
| Web    | -> | App    | -> | DB     |
| VLAN10 |    | VLAN20 |    | VLAN30 |
+--------+    +--------+    +--------+
  許可: → App:8080   許可: → DB:5432
  拒否: その他       拒否: その他

方式2: ホストベース (iptables / セキュリティグループ)
+--------+      +--------+      +--------+
| Web    | ---> | App    | ---> | DB     |
| SG-Web |      | SG-App |      | SG-DB  |
+--------+      +--------+      +--------+
各SGで送信元/宛先を厳密に制御

方式3: サービスメッシュ (Istio / Linkerd)
+--------+      +--------+      +--------+
| Web    | ---> | App    | ---> | DB     |
|+Envoy |  mTLS |+Envoy |  mTLS |+Envoy |
+--------+      +--------+      +--------+
   Istio AuthorizationPolicy で
   L7レベルの通信制御 (メソッド、パス単位)

方式4: eBPF ベース (Cilium)
+--------+      +--------+      +--------+
| Web    | ---> | App    | ---> | DB     |
|+Cilium |      |+Cilium |      |+Cilium |
+--------+      +--------+      +--------+
   カーネルレベルで高速に
   L3-L7のポリシー適用
```

### Kubernetes NetworkPolicy の実装

```yaml
# Web → App のみ許可 (App Pod の NetworkPolicy)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: app-server-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: app-server
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Web サーバからの 8080 のみ許可
    - from:
        - podSelector:
            matchLabels:
              app: web-server
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # DB への 5432 のみ許可
    - to:
        - podSelector:
            matchLabels:
              app: database
      ports:
        - protocol: TCP
          port: 5432
    # DNS を許可 (必須)
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
```

### Istio AuthorizationPolicy (L7 制御)

```yaml
# API パス単位の認可制御
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: app-server-authz
  namespace: production
spec:
  selector:
    matchLabels:
      app: app-server
  rules:
    # Web サーバからの GET/POST のみ許可
    - from:
        - source:
            principals: ["cluster.local/ns/production/sa/web-server"]
      to:
        - operation:
            methods: ["GET", "POST"]
            paths: ["/api/v1/*"]
    # 管理APIは管理サービスのみ
    - from:
        - source:
            principals: ["cluster.local/ns/admin/sa/admin-service"]
      to:
        - operation:
            methods: ["GET", "POST", "DELETE"]
            paths: ["/admin/*"]
  # 上記以外は暗黙的に拒否
```

---

## 6. DDoS 対策

### DDoS 攻撃の分類

```
+-----------------------------------------------------------+
|                   DDoS 攻撃の分類                           |
|-----------------------------------------------------------|
|                                                           |
|  [ボリューメトリック攻撃 (L3/L4)]                           |
|  +-- UDP Flood: 大量のUDPパケット                         |
|  +-- ICMP Flood: 大量のpingパケット                       |
|  +-- DNS Amplification: DNS増幅 (x50-70倍)              |
|  +-- NTP Amplification: NTP増幅 (x556倍)               |
|  +-- Memcached Amplification: (x51,000倍)               |
|  → 対策: ISP / CDN での吸収 (AWS Shield, CloudFlare)     |
|                                                           |
|  [プロトコル攻撃 (L3/L4)]                                  |
|  +-- SYN Flood: 大量のSYNパケット                         |
|  +-- Ping of Death: 巨大ICMPパケット                      |
|  +-- Smurf Attack: ブロードキャストICMP                   |
|  → 対策: SYN Cookie、レート制限、ステートフルFW            |
|                                                           |
|  [アプリケーション攻撃 (L7)]                                |
|  +-- HTTP Flood: 大量のHTTPリクエスト                     |
|  +-- Slowloris: 接続を長時間占有                          |
|  +-- RUDY: 低速POST送信                                  |
|  +-- HTTP/2 Rapid Reset (CVE-2023-44487)                |
|  → 対策: WAF、レート制限、CAPTCHA                         |
+-----------------------------------------------------------+
```

### SYN Flood 対策 (SYN Cookie)

```
通常の TCP 3-way ハンドシェイク:
Client              Server
  |--- SYN -------->|  (SYN キューに保存)
  |<-- SYN-ACK -----|  (メモリ確保)
  |--- ACK -------->|  (ESTABLISHED)

SYN Flood 攻撃:
攻撃者   Server
  |--- SYN (偽IP1) -->|  SYN キュー: [1]
  |--- SYN (偽IP2) -->|  SYN キュー: [1,2]
  |--- SYN (偽IP3) -->|  SYN キュー: [1,2,3]
  |--- SYN ...     -->|  SYN キュー: [FULL] → 新規接続拒否

SYN Cookie による対策:
攻撃者   Server (SYN Cookie有効)
  |--- SYN -------->|  キューに保存しない
  |<-- SYN-ACK -----|  ISN = hash(src,dst,port,secret,time)
  |                 |  (ステートレス、メモリ不使用)
  |  (ACK来ない)    |  → リソース消費なし

正規Client  Server
  |--- SYN -------->|  キューに保存しない
  |<-- SYN-ACK -----|  ISN = hash(...)
  |--- ACK -------->|  ISN+1 を検証 → ESTABLISHED
```

```bash
# SYN Cookie の有効化 (Linux)
sysctl -w net.ipv4.tcp_syncookies=1
sysctl -w net.ipv4.tcp_max_syn_backlog=65536
sysctl -w net.ipv4.tcp_synack_retries=2

# 永続化
echo "net.ipv4.tcp_syncookies = 1" >> /etc/sysctl.d/99-security.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.d/99-security.conf
```

---

## 7. DNS セキュリティ (概要)

### DNS 関連の攻撃と対策

```
+--------------------------------------------+
| DNS 攻撃と対策                               |
|--------------------------------------------|
| 攻撃              | 対策                    |
|--------------------------------------------|
| DNS Spoofing      | DNSSEC                 |
| DNS Cache Poison  | ソースポートランダム化   |
| DNS Amplification | レスポンスレート制限     |
| DNS Tunneling     | DNS クエリ監視/IDS      |
| DNS Hijacking     | DoH/DoT (暗号化DNS)    |
| Domain Shadowing  | ドメイン監視            |
+--------------------------------------------+
```

### DNSSEC の仕組み

```
DNS 応答の署名検証チェイン:

ルート (.)
  |  KSK/ZSK で .com のDSレコードに署名
  v
.com
  |  KSK/ZSK で example.com のDSレコードに署名
  v
example.com
  |  ZSK で A/AAAA/MX 等のレコードに署名
  v
www.example.com = 93.184.216.34 + RRSIG

リゾルバは各階層の署名を検証:
1. ルートの KSK (トラストアンカー) を保持
2. .com の DS レコードを検証
3. example.com の DS レコードを検証
4. www.example.com の RRSIG を検証
→ 改ざんがあれば SERVFAIL を返す
```

---

## 8. アンチパターン

### アンチパターン 1: フラットネットワーク

```
NG: 全サーバが同一サブネット
  +-- Web Server --+
  +-- App Server --+-- 同一セグメント (10.0.1.0/24)
  +-- DB Server  --+
  +-- 管理サーバ  --+
  (DBが直接インターネットから到達可能)
  (1台の侵害で全サーバに水平移動可能)

OK: サブネット分離 + マイクロセグメンテーション
  DMZ:     10.0.0.0/24 -- リバースプロキシ/WAF のみ
  Public:  10.0.1.0/24 -- ALB のみ
  Private: 10.0.2.0/24 -- App Server (ALBからのみ)
  Data:    10.0.3.0/24 -- DB (App からの5432のみ)
  Mgmt:    10.0.4.0/24 -- Bastion/踏み台 (VPNからのみ)
```

**影響**: 1 台が侵害されると水平移動 (Lateral Movement) で全サーバに到達される。Capital One (2019年) や SolarWinds (2020年) の事件では、フラットなネットワーク構成が被害拡大の一因となった。

### アンチパターン 2: ファイアウォールの ANY-ANY ルール

```bash
# NG: 全通信を許可 (テスト時に設定して本番に残る)
iptables -A INPUT -j ACCEPT

# NG: 広すぎるセキュリティグループ
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol -1 --port -1 \
  --cidr 0.0.0.0/0

# NG: SSH を全世界に開放
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp --port 22 \
  --cidr 0.0.0.0/0

# OK: 必要最小限のポート・送信元のみ許可
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp --port 443 \
  --cidr 0.0.0.0/0

# OK: SSH は VPN/踏み台経由のみ
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp --port 22 \
  --source-group sg-bastion
```

### アンチパターン 3: ログ監視なし

```
NG:
  → ファイアウォールログを収集していない
  → IDS アラートを誰も見ていない
  → VPN 接続ログを保持していない
  → 侵害の検知まで平均 197 日 (IBM X-Force 2023)

OK:
  → 全ファイアウォールログを SIEM に集約
  → IDS アラートの自動エスカレーション
  → VPN 認証ログの異常検知 (深夜の接続、普段と異なる国)
  → SOC チームによる 24/7 監視 or MSSP の利用
  → インシデントレスポンスプランの策定と定期訓練
```

### アンチパターン 4: VPN に接続しただけで全アクセス許可

```
NG (従来型VPN):
  VPN接続 → 社内ネットワーク全体にフルアクセス
  (侵害されたPCがVPN経由で全システムに到達)

OK (ゼロトラスト + VPN):
  VPN接続 → デバイス検証 → ユーザ認証 → ポリシー評価 → 必要なサービスのみアクセス
  (VPNは暗号化トンネルの役割のみ、認可は別レイヤー)
```

---

## 9. 演習問題

### 演習 1 (基礎): nftables でファイアウォールを構築

**課題**: 以下の要件を満たす nftables 設定ファイルを作成せよ。
- デフォルトポリシー: INPUT DROP, FORWARD DROP, OUTPUT ACCEPT
- ループバック許可
- 確立済みコネクション許可
- SSH は 192.168.1.0/24 からのみ許可 (レート制限: 3/分)
- HTTP/HTTPS を許可
- ICMP echo-request を 1/秒 で許可
- それ以外はログ記録後に DROP

**ヒント**: `set` と `limit rate` を組み合わせる。

### 演習 2 (応用): Suricata カスタムルールの作成

**課題**: 以下の攻撃を検知する Suricata ルールを書け。
1. HTTP リクエストの User-Agent が空 (ボット/スキャナの兆候)
2. 内部ネットワークから外部への大量 DNS クエリ (50回/分以上、DNS トンネリングの兆候)
3. 内部サーバから外部への SSH 接続 (通常は発生しないはずの通信)

**ヒント**: `threshold` と `flow` オプションを活用する。

### 演習 3 (実践): WireGuard VPN の構築

**課題**: 以下の構成で WireGuard VPN を構築せよ。
- サーバ: 10.200.0.1/24, ポート 51820
- クライアント 3 台: 10.200.0.2-4/32
- Split Tunnel: 10.0.0.0/8 のみ VPN 経由 (残りは直接通信)
- PresharedKey を使用 (量子耐性)
- Kill Switch をクライアントに設定

**ヒント**: `AllowedIPs` で経路制御、`PostUp/PostDown` で Kill Switch を実装。

---

## 10. トラブルシューティング

### ファイアウォールのデバッグ

```bash
# iptables: パケットカウンタの確認
iptables -L -v -n --line-numbers

# nftables: ルールごとのカウンタ
nft list ruleset -a

# conntrack: コネクション追跡テーブル
conntrack -L
conntrack -E   # リアルタイムイベント

# tcpdump: パケットキャプチャ
tcpdump -i eth0 -n 'port 443 and host 10.0.1.100'
tcpdump -i eth0 -n 'tcp[tcpflags] & (tcp-syn) != 0'  # SYN パケットのみ

# ss: ソケット統計
ss -tlnp         # TCP リスニングポート
ss -s            # ソケット統計サマリ
```

### VPN 接続のトラブルシューティング

```bash
# WireGuard: ステータス確認
wg show wg0

# WireGuard: ハンドシェイク確認
wg show wg0 latest-handshakes
# 0 = ハンドシェイク未完了 (鍵不一致、FW、ルーティング問題)

# WireGuard: デバッグログ有効化
echo module wireguard +p > /sys/kernel/debug/dynamic_debug/control
dmesg | grep wireguard

# MTU 問題の診断 (VPN 越しのパケット断片化)
ping -M do -s 1400 10.200.0.1
# → "Frag needed" → MTU を下げる (1280-1420)

# IPsec: 接続ステータス
swanctl --list-sas
swanctl --log   # リアルタイムログ
```

### IDS/IPS のパフォーマンス問題

```bash
# Suricata: 処理統計
suricatasc -c "dump-counters" | grep -E "drop|bypass"

# パケットドロップの確認
cat /proc/net/dev | grep eth0
# → RX dropped が増加 → AF_PACKET のリングバッファを拡大

# CPU バインディングの確認
suricatasc -c "iface-stat eth0"

# ルールプロファイリング
suricata --engine-analysis
# → 負荷の高いルールを特定して最適化
```

---

## 11. パフォーマンス考慮事項

### ファイアウォールのパフォーマンス

| 要因 | 影響 | 対策 |
|------|------|------|
| ルール数 | 線形評価で O(n) | セット/マップの活用、ルール順序最適化 |
| conntrack テーブル | メモリ消費 | `nf_conntrack_max` の調整 |
| ログ出力 | I/O ボトルネック | 非同期ログ、サンプリング |
| DPI (深層検査) | CPU 負荷 | ハードウェアオフロード (FPGA/ASIC) |
| TLS インスペクション | CPU 負荷 (暗号処理) | SSL アクセラレータ |

```bash
# conntrack テーブルのチューニング
sysctl -w net.nf_conntrack_max=1048576
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_established=3600
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_time_wait=30

# nftables: セットによる高速ルックアップ
# O(n) の線形評価 → O(1) のハッシュルックアップ
table inet filter {
    set allowed_ips {
        type ipv4_addr
        flags interval
        elements = { 10.0.0.0/8, 172.16.0.0/12 }
    }
    chain input {
        ip saddr @allowed_ips tcp dport 22 accept
    }
}
```

---

## 12. FAQ

### Q1. WAF と NGFW の違いは?

WAF は HTTP/HTTPS トラフィックに特化し、SQL インジェクションや XSS などの Web 攻撃を防御する。NGFW はネットワーク全体のトラフィックを L3-L7 で検査し、アプリケーション識別・IPS 機能を統合したものである。両者は補完関係にあり、Web サービスでは WAF を ALB の前段に、NGFW を VPC の境界に配置するのが理想的である。WAF は OWASP Top 10 の Web 攻撃に強く、NGFW はネットワーク全体の可視性と制御に優れる。

### Q2. IDS と IPS のどちらを導入すべきか?

本番環境では IPS をインラインに配置し自動遮断するのが推奨される。ただし誤検知による正常通信の遮断リスクがあるため、以下の段階的アプローチが安全である:
1. IDS モード (検知のみ) で 2-4 週間運用
2. アラートを分析し誤検知パターンを特定
3. Suppression ルールと Threshold を調整
4. 段階的に IPS モード (自動遮断) に移行
5. 初期は低重大度を通過させ、高重大度のみ遮断
6. チューニング後に全ルールで IPS モードを有効化

### Q3. VPN と ゼロトラストは共存できるか?

できる。VPN はネットワーク層の暗号化トンネルとして機能し、ゼロトラストはその上で各アクセスをリクエスト単位で検証する。ただし、VPN に接続しただけで社内ネットワーク全体にアクセスできる従来の運用はゼロトラストの思想に反する。現実的な構成は:
- VPN で暗号化トンネルを提供 (WireGuard / IPsec)
- Identity-Aware Proxy (Google BeyondCorp, Cloudflare Access) でアプリケーション単位の認証
- マイクロセグメンテーションで East-West トラフィックを制御
- SASE (Secure Access Service Edge) による統合的なセキュリティ

### Q4. Kubernetes 環境でのネットワークセキュリティはどうすべきか?

Kubernetes では以下の多層アプローチが推奨される:
1. **NetworkPolicy**: Pod 間の L3/L4 通信制御 (CNI プラグイン依存)
2. **サービスメッシュ (Istio/Linkerd)**: mTLS + L7 認可ポリシー
3. **Cilium (eBPF)**: カーネルレベルの高速ネットワークポリシー
4. **Ingress Gateway**: 外部からの入口を一元管理
5. **Egress Gateway**: 外向き通信を制御 (データ流出防止)

### Q5. 暗号化された通信 (TLS) を IDS/IPS で検査するには?

TLS インスペクションには 2 つのアプローチがある:
1. **TLS 終端**: ロードバランサ/リバースプロキシで TLS を終端し、平文を IDS/IPS に流す (最も一般的)
2. **TLS インスペクション**: NGFW/IPS が中間者として TLS を復号・再暗号化 (プライバシー懸念あり)

JA3/JA4 フィンガープリンティングにより、TLS を復号せずにクライアントの種類を識別することも可能 (マルウェア C2 の検知等)。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 多層防御 | 境界・ネットワーク・ホスト・アプリの各層で防御、単一障害点を排除 |
| ファイアウォール | デフォルト拒否、必要最小限のルールのみ許可、nftables 推奨 |
| IDS/IPS | シグネチャ + 異常検知、IDS→IPS の段階的移行、SIEM 連携必須 |
| VPN | WireGuard が設定の簡潔さとパフォーマンスで優位、PFS 常時有効 |
| セグメンテーション | サブネット分離とマイクロセグメンテーションで水平移動を阻止 |
| ゼロトラスト | すべての通信を検証し暗黙の信頼を排除、mTLS + ポリシーエンジン |
| DDoS 対策 | L3/L4 はインフラ層 (CDN/Shield)、L7 は WAF + レート制限 |
| DNS セキュリティ | DNSSEC + DoH/DoT + クエリ監視 |
| 監視 | 全ログを SIEM に集約、自動アラート + SOC 運用 |

---

## 次に読むべきガイド

- [DNSセキュリティ](./01-dns-security.md) — DNS 層での攻撃と対策の詳細
- [APIセキュリティ](./02-api-security.md) — アプリケーション層の API 保護
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) — 暗号化通信の基盤技術
- [クラウドセキュリティ](../05-cloud-security/00-cloud-security-basics.md) — AWS/GCP のネットワークセキュリティ

---

## 参考文献

1. **NIST SP 800-41 Rev.1 — Guidelines on Firewalls and Firewall Policy** — https://csrc.nist.gov/publications/detail/sp/800-41/rev-1/final
2. **NIST SP 800-207 — Zero Trust Architecture** — https://csrc.nist.gov/publications/detail/sp/800-207/final
3. **NIST SP 800-94 — Guide to Intrusion Detection and Prevention Systems** — https://csrc.nist.gov/publications/detail/sp/800-94/final
4. **Suricata Documentation** — https://docs.suricata.io/en/latest/
5. **WireGuard — Conceptual Overview** — https://www.wireguard.com/papers/wireguard.pdf
6. **MITRE ATT&CK Framework** — https://attack.mitre.org/
7. **Kubernetes Network Policies** — https://kubernetes.io/docs/concepts/services-networking/network-policies/
8. **nftables Wiki** — https://wiki.nftables.org/
