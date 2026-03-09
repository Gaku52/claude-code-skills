# ネットワークデバッグ

> ネットワーク問題の切り分けと解決に必要なツールと手法を体系的に学ぶ。tcpdump、Wireshark、Chrome DevTools、curl、ss/netstat をはじめとする主要ツールを使いこなし、再現性のある効率的なトラブルシューティング手法を身につける。

---

## この章で学ぶこと

- [ ] 主要なネットワークデバッグツール（curl, dig, tcpdump, Wireshark, Chrome DevTools）の使い方を理解する
- [ ] OSI 参照モデルの各レイヤーに対応した問題の切り分け手法を把握する
- [ ] パケットキャプチャの取得・解析・レポーティング手順を習得する
- [ ] Chrome DevTools の Network タブを活用してフロントエンドの通信問題を特定する
- [ ] 体系的なトラブルシューティング決定木に従って問題を迅速に解決する
- [ ] デバッグにおけるアンチパターンを認識し、回避する

---

## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- [TCP](../01-protocols/00-tcp.md) — 3ウェイハンドシェイク、ウィンドウサイズ、再送制御の仕組み
- [HTTP基礎](../02-http/00-http-basics.md) — HTTPメソッド、ステータスコード、ヘッダーの基本
- [DNS](../00-introduction/03-dns.md) — 名前解決の流れ、レコードタイプ、キャッシュ
- [TLS/SSL](../03-security/00-tls-ssl.md) — TLSハンドシェイク、証明書チェーン、暗号スイート
- [IPアドレッシング](../00-introduction/02-ip-addressing.md) — IPv4/IPv6、サブネット、NATの基礎

---

## 1. ネットワークデバッグの全体像

ネットワークの問題は多層的であり、レイヤーごとに適切なツールを選択することが重要である。以下の ASCII 図はデバッグの全体的な流れを示している。

### 図1: ネットワークデバッグの全体フロー

```
+================================================================+
|              ネットワークデバッグ 全体フロー                        |
+================================================================+
|                                                                |
|  [問題発生]                                                     |
|      |                                                         |
|      v                                                         |
|  +--------------------+                                        |
|  | 1. 症状の整理      |  何が起きているか？                       |
|  |    - エラーメッセージ|  いつから？ 影響範囲は？                  |
|  |    - 発生頻度       |  再現手順は？                           |
|  |    - 影響範囲       |                                        |
|  +--------+-----------+                                        |
|           |                                                    |
|           v                                                    |
|  +--------------------+     +---------------------------+      |
|  | 2. レイヤー特定     |---->| L1-L2: 物理/データリンク   |      |
|  |    OSIモデルに沿って |     | ツール: ip link, ethtool  |      |
|  |    上位から順に確認  |     +---------------------------+      |
|  +--------+-----------+     +---------------------------+      |
|           |            |--->| L3: ネットワーク層         |      |
|           |                 | ツール: ping, traceroute   |      |
|           |                 +---------------------------+      |
|           |                 +---------------------------+      |
|           |            |--->| L4: トランスポート層       |      |
|           |                 | ツール: ss, netstat, nc    |      |
|           |                 +---------------------------+      |
|           |                 +---------------------------+      |
|           |            |--->| L5-L7: アプリケーション層   |      |
|           |                 | ツール: curl, DevTools     |      |
|           |                 +---------------------------+      |
|           v                                                    |
|  +--------------------+                                        |
|  | 3. 仮説の立案      |  収集した情報から原因の仮説を立てる        |
|  +--------+-----------+                                        |
|           |                                                    |
|           v                                                    |
|  +--------------------+                                        |
|  | 4. 検証・再現      |  tcpdump / Wireshark で証拠を集める      |
|  +--------+-----------+                                        |
|           |                                                    |
|           v                                                    |
|  +--------------------+                                        |
|  | 5. 修正・確認      |  修正を適用し、問題が解消されたことを確認   |
|  +--------+-----------+                                        |
|           |                                                    |
|           v                                                    |
|  +--------------------+                                        |
|  | 6. 文書化          |  原因・対処・再発防止策を記録              |
|  +--------------------+                                        |
|                                                                |
+================================================================+
```

### デバッグツールとOSIレイヤーの対応関係

| OSIレイヤー | レイヤー名 | 主要デバッグツール | 確認できる問題 |
|:-----------:|:----------:|:------------------:|:-------------|
| L1 | 物理層 | `ethtool`, `ip link` | ケーブル断線、NIC障害、リンクダウン |
| L2 | データリンク層 | `arp`, `ip neigh`, `bridge` | MACアドレス解決失敗、VLAN設定ミス |
| L3 | ネットワーク層 | `ping`, `traceroute`, `mtr` | ルーティング問題、IP到達不能 |
| L4 | トランスポート層 | `ss`, `netstat`, `nc`, `tcpdump` | ポート未開放、接続タイムアウト、再送多発 |
| L5-L7 | セッション〜アプリケーション層 | `curl`, `openssl`, `Chrome DevTools` | TLSエラー、HTTPエラー、アプリケーションバグ |

---

## 2. curl によるHTTPデバッグ

curl は HTTP/HTTPS の通信をコマンドラインから直接テストできる万能ツールである。サーバーのレスポンス内容、ヘッダー、タイミング、TLS 情報などを詳細に確認できる。

### 2.1 基本的な使い方

```bash
# 基本的なGETリクエスト
$ curl https://api.example.com/users

# レスポンスヘッダーのみを表示（HEADリクエスト）
$ curl -I https://api.example.com/users

# リクエスト/レスポンスの詳細表示（-v: verbose）
$ curl -v https://api.example.com/users

# POSTリクエスト（JSONペイロード）
$ curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token123" \
  -d '{"name": "Taro", "email": "taro@example.com"}'

# PUTリクエスト（既存リソースの更新）
$ curl -X PUT https://api.example.com/users/42 \
  -H "Content-Type: application/json" \
  -d '{"name": "Taro Updated"}'

# DELETEリクエスト
$ curl -X DELETE https://api.example.com/users/42 \
  -H "Authorization: Bearer token123"

# リダイレクトを自動追跡（-L: location）
$ curl -L -v https://example.com

# レスポンスボディをファイルに保存
$ curl -o response.json https://api.example.com/data

# HTTP/2で接続
$ curl --http2 -v https://api.example.com/users

# HTTP/3 (QUIC) で接続（curl 7.66+）
$ curl --http3 -v https://api.example.com/users

# 特定のIPアドレスに対してリクエスト（DNS回避）
$ curl --resolve api.example.com:443:203.0.113.10 \
  https://api.example.com/users

# クライアント証明書を指定
$ curl --cert client.crt --key client.key \
  https://secure.example.com/api
```

### 2.2 タイミング計測（コード例1）

curl の `-w` オプションを使うと、接続の各フェーズにかかった時間を詳細に計測できる。パフォーマンスのボトルネック特定に非常に有用である。

```bash
# タイミング情報の詳細表示
$ curl -o /dev/null -s -w "\
  DNS Lookup:      %{time_namelookup}s\n\
  TCP Connect:     %{time_connect}s\n\
  TLS Handshake:   %{time_appconnect}s\n\
  Start Transfer:  %{time_starttransfer}s\n\
  Redirect:        %{time_redirect}s\n\
  Total:           %{time_total}s\n\
  \n\
  HTTP Status:     %{http_code}\n\
  Download Size:   %{size_download} bytes\n\
  Upload Size:     %{size_upload} bytes\n\
  Speed Download:  %{speed_download} bytes/s\n\
  Speed Upload:    %{speed_upload} bytes/s\n\
  Num Connects:    %{num_connects}\n\
  Num Redirects:   %{num_redirects}\n\
  SSL Verify:      %{ssl_verify_result}\n\
  Remote IP:       %{remote_ip}\n\
  Remote Port:     %{remote_port}\n" \
  https://api.example.com/users
```

**出力例と各フェーズの解説:**

```
  DNS Lookup:      0.012345s    ← 名前解決にかかった時間
  TCP Connect:     0.034567s    ← TCP 3ウェイハンドシェイク完了まで
  TLS Handshake:   0.089012s    ← TLSハンドシェイク完了まで
  Start Transfer:  0.123456s    ← 最初の1バイト受信まで（TTFB）
  Redirect:        0.000000s    ← リダイレクト処理の合計時間
  Total:           0.156789s    ← 全体の所要時間

  HTTP Status:     200
  Download Size:   4523 bytes
  Upload Size:     0 bytes
  Speed Download:  28852 bytes/s
  Speed Upload:    0 bytes/s
  Num Connects:    1
  Num Redirects:   0
  SSL Verify:      0            ← 0 = 検証成功
  Remote IP:       203.0.113.10
  Remote Port:     443
```

**タイミング値の読み方:**

```
0                  time_namelookup
|---DNS解決--------|
                   time_connect
|---DNS+TCP--------|
                   time_appconnect
|---DNS+TCP+TLS----|
                   time_starttransfer
|---DNS+TCP+TLS+サーバー処理---|
                                time_total
|---全体の処理時間--------------|

各フェーズの所要時間の計算:
  DNS解決時間      = time_namelookup
  TCP接続時間      = time_connect - time_namelookup
  TLS時間          = time_appconnect - time_connect
  サーバー処理時間  = time_starttransfer - time_appconnect
  コンテンツ転送時間 = time_total - time_starttransfer
```

### 2.3 curlでの連続テストスクリプト

```bash
#!/bin/bash
# endpoint_health_check.sh
# 複数エンドポイントの応答時間を連続計測するスクリプト

ENDPOINTS=(
  "https://api.example.com/health"
  "https://api.example.com/users"
  "https://api.example.com/products"
  "https://cdn.example.com/assets/main.js"
)

echo "Timestamp,Endpoint,Status,DNS,Connect,TLS,TTFB,Total"

for endpoint in "${ENDPOINTS[@]}"; do
  result=$(curl -o /dev/null -s -w \
    "%{http_code},%{time_namelookup},%{time_connect},%{time_appconnect},%{time_starttransfer},%{time_total}" \
    --max-time 10 \
    "$endpoint")

  timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${timestamp},${endpoint},${result}"
done
```

---

## 3. DNS デバッグ

DNS の問題は「名前が引けない」「間違った IP が返る」「解決に時間がかかる」の 3 パターンに大別できる。

### 3.1 dig による詳細な DNS 調査

```bash
# Aレコード（IPv4アドレス）の問い合わせ
$ dig example.com

# 特定レコードタイプの問い合わせ
$ dig example.com A          # IPv4アドレス
$ dig example.com AAAA       # IPv6アドレス
$ dig example.com MX         # メールサーバー
$ dig example.com NS         # ネームサーバー
$ dig example.com TXT        # テキストレコード（SPF等）
$ dig example.com CNAME      # 別名
$ dig example.com SOA        # 権威情報

# 短縮出力（結果のみ）
$ dig example.com +short

# 特定のDNSサーバーに問い合わせ
$ dig @8.8.8.8 example.com           # Google Public DNS
$ dig @1.1.1.1 example.com           # Cloudflare DNS
$ dig @208.67.222.222 example.com    # OpenDNS

# 名前解決の全過程を追跡
$ dig +trace example.com

# 逆引き（IPアドレスからホスト名）
$ dig -x 203.0.113.10

# DNSSEC検証情報の表示
$ dig +dnssec example.com

# 応答時間の確認（Query time に注目）
$ dig example.com | grep "Query time"
;; Query time: 12 msec
```

### 3.2 nslookup と host

```bash
# nslookup — インタラクティブ/非インタラクティブ
$ nslookup example.com
$ nslookup -type=CNAME www.example.com
$ nslookup -type=MX example.com 8.8.8.8

# host — 最も簡潔な出力
$ host example.com
$ host -t MX example.com
$ host -t AAAA example.com
```

### 3.3 DNS キャッシュ管理

```bash
# macOS: DNSキャッシュのクリア
$ sudo dscacheutil -flushcache
$ sudo killall -HUP mDNSResponder

# Linux (systemd-resolved):
$ sudo systemd-resolve --flush-caches
$ sudo systemd-resolve --statistics   # キャッシュ統計

# Linux (nscd):
$ sudo systemctl restart nscd

# /etc/hosts の確認（ローカルオーバーライド）
$ cat /etc/hosts

# /etc/resolv.conf の確認（使用中のDNSサーバー）
$ cat /etc/resolv.conf
```

### 3.4 DNS 問題の切り分けフロー

```
  [DNSの問題が疑われる]
        |
        v
  dig +short example.com
        |
   +---------+----------+
   |                     |
   v                     v
  IPが返る             IPが返らない
   |                     |
   v                     v
  正しいIPか？         dig @8.8.8.8 example.com
   |                     |
  +---+---+          +---+---+
  |       |          |       |
  v       v          v       v
 正しい  間違い     返る    返らない
  |       |          |       |
  v       v          v       v
 DNS以外  キャッシュ  ローカル  ドメイン
 の問題   またはCDN   DNS設定  自体の問題
          の問題      の問題   (NXDOMAIN)
```

---

## 4. ネットワーク接続デバッグ

### 4.1 ping による疎通確認

```bash
# 基本的な疎通確認
$ ping example.com
$ ping -c 5 example.com        # 5回だけ送信
$ ping -c 10 -i 0.5 example.com  # 0.5秒間隔で10回

# IPv6での疎通確認
$ ping6 example.com

# パケットサイズを指定（MTU問題の調査）
$ ping -s 1472 -M do example.com  # Don't Fragment フラグ付き
# 応答があれば MTU 1500 (1472 + 28 = 1500) で問題なし
# "Frag needed" が返れば MTU が小さい経路がある

# タイムスタンプ付き
$ ping -D example.com            # Linux
```

### 4.2 traceroute / mtr による経路調査

```bash
# traceroute — 経路の各ホップを表示
$ traceroute example.com
$ traceroute -T example.com      # TCP traceroute（ICMP がブロックされる場合）
$ traceroute -p 443 example.com  # ポート443で経路確認
$ traceroute -n example.com      # 逆引きなし（高速）

# mtr — ping + traceroute の統合リアルタイム表示
$ mtr example.com
$ mtr --report -c 100 example.com    # レポートモード（100回計測）
$ mtr --tcp --port 443 example.com   # TCP/443での計測
```

### 4.3 nc (netcat) によるポート接続テスト

```bash
# TCPポートが開いているか確認
$ nc -zv example.com 80       # HTTP
$ nc -zv example.com 443      # HTTPS
$ nc -zv example.com 22       # SSH
$ nc -zv example.com 3306     # MySQL

# UDPポートの確認
$ nc -zuv example.com 53      # DNS

# ポート範囲のスキャン
$ nc -zv example.com 80-100

# 簡易HTTPリクエスト
$ echo -e "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n" | nc example.com 80

# TCPプロキシ/リレー（デバッグ用）
$ nc -l -p 8080 | tee capture.txt | nc target.example.com 80
```

### 4.4 ss / netstat によるソケット状態の確認（コード例2）

ss は netstat の後継コマンドであり、カーネルのソケット情報をより高速かつ詳細に表示できる。

```bash
# ============================================
# ss コマンド（推奨: netstatの後継）
# ============================================

# リッスン中のTCPポート一覧（プロセス名付き）
$ ss -tlnp
# State  Recv-Q Send-Q  Local Address:Port  Peer Address:Port  Process
# LISTEN 0      128     0.0.0.0:80          0.0.0.0:*          users:(("nginx",pid=1234))
# LISTEN 0      128     0.0.0.0:443         0.0.0.0:*          users:(("nginx",pid=1234))
# LISTEN 0      511     127.0.0.1:3000      0.0.0.0:*          users:(("node",pid=5678))

# オプション解説:
#   -t : TCPのみ
#   -l : LISTENINGのみ
#   -n : ポート番号を数値で表示（名前解決しない）
#   -p : プロセス情報を表示

# UDPソケットの一覧
$ ss -ulnp

# 全ソケットの統計サマリ
$ ss -s
# Total: 342
# TCP:   120 (estab 45, closed 20, orphaned 3, timewait 15)
# UDP:   12

# 特定の状態の接続を表示
$ ss -t state established          # 確立済み接続
$ ss -t state time-wait            # TIME_WAIT状態
$ ss -t state close-wait           # CLOSE_WAIT状態
$ ss -t state syn-sent             # SYN送信済み（接続試行中）

# 特定ポートの接続を表示
$ ss -t dst :443                   # 宛先ポート443の接続
$ ss -t src :8080                  # 送信元ポート8080の接続

# 特定ホストへの接続を表示
$ ss -t dst 203.0.113.10

# CLOSE_WAIT が大量にある場合の調査
$ ss -t state close-wait -p | awk '{print $NF}' | sort | uniq -c | sort -rn
#   150 users:(("java",pid=9876))  ← このプロセスが接続を閉じていない
#    23 users:(("python",pid=5432))

# TIME_WAIT の数を監視
$ watch -n 1 'ss -s | grep -i time'

# ============================================
# netstat コマンド（レガシー、一部環境で利用）
# ============================================

$ netstat -tlnp                    # ss -tlnp と同等
$ netstat -an | grep ESTABLISHED   # 確立済み接続
$ netstat -s                       # プロトコル別統計

# ============================================
# lsof によるポート調査
# ============================================

# 特定ポートを使用しているプロセス
$ lsof -i :8080
$ lsof -i :80 -i :443             # 複数ポート

# 全ネットワーク接続をリスト
$ lsof -i -P -n

# 特定プロセスのネットワーク接続
$ lsof -i -a -p 1234
```

### TCP 接続状態とその意味

| 状態 | 意味 | 問題の可能性 |
|:-----|:-----|:------------|
| ESTABLISHED | 接続確立済み、通信中 | 正常（大量の場合はリソース枯渇に注意） |
| TIME_WAIT | 接続終了後の待機中（通常 60秒） | 大量蓄積は短時間に多数の接続を開閉している兆候 |
| CLOSE_WAIT | 相手が FIN を送信済み、こちらが close() していない | アプリケーションのバグ（ソケットリーク）の可能性大 |
| SYN_SENT | SYN を送信済み、応答待ち | 相手が到達不能、またはファイアウォールでブロック |
| SYN_RECV | SYN を受信し SYN+ACK を返した、ACK 待ち | SYN Flood 攻撃の可能性 |
| FIN_WAIT1 | こちらが FIN を送信済み、ACK 待ち | 相手側のレスポンスが遅い |
| FIN_WAIT2 | FIN の ACK を受信済み、相手の FIN 待ち | 相手側アプリケーションが close() していない |
| LAST_ACK | こちらが FIN を送信済み、最後の ACK 待ち | 通常は一時的な状態 |

---

## 5. tcpdump によるパケットキャプチャ（コード例3）

tcpdump はコマンドラインで動作するパケットキャプチャツールであり、ネットワークインタフェースを流れるパケットをリアルタイムで取得・表示できる。サーバー上でのデバッグやスクリプトによる自動化に適している。

### 5.1 基本的な使い方

```bash
# 全トラフィックをキャプチャ（Ctrl+C で停止）
$ sudo tcpdump -i eth0

# 特定ホストのトラフィック
$ sudo tcpdump host 203.0.113.10
$ sudo tcpdump host example.com

# 送信元または宛先を限定
$ sudo tcpdump src host 203.0.113.10
$ sudo tcpdump dst host 203.0.113.10

# 特定ポートのトラフィック
$ sudo tcpdump port 443
$ sudo tcpdump port 80 or port 443

# 特定のネットワーク範囲
$ sudo tcpdump net 192.168.1.0/24

# プロトコル指定
$ sudo tcpdump tcp
$ sudo tcpdump udp
$ sudo tcpdump icmp

# 複合フィルタ
$ sudo tcpdump 'host 203.0.113.10 and port 443 and tcp'
$ sudo tcpdump 'src net 192.168.1.0/24 and dst port 80'
```

### 5.2 詳細表示とファイル保存

```bash
# ASCII表示（HTTPの内容が読める）
$ sudo tcpdump -A port 80

# HEX + ASCII 表示
$ sudo tcpdump -X port 80

# タイムスタンプの形式を指定
$ sudo tcpdump -tttt port 443    # 人間が読みやすい日時形式

# パケット数を制限
$ sudo tcpdump -c 100 port 443   # 100パケットで停止

# pcapファイルに保存（Wiresharkで開ける）
$ sudo tcpdump -w capture.pcap -c 1000 port 443

# pcapファイルの読み込み
$ sudo tcpdump -r capture.pcap

# ファイルに保存しつつ画面にも表示
$ sudo tcpdump -w capture.pcap -c 500 port 443 &
$ sudo tcpdump -r capture.pcap   # 別ターミナルで随時確認

# スナップショットサイズの指定（パケット全体をキャプチャ）
$ sudo tcpdump -s 0 -w full_capture.pcap

# ローテーションキャプチャ（100MBごとにファイル分割、最大10ファイル保持）
$ sudo tcpdump -w capture_%Y%m%d_%H%M%S.pcap -G 3600 -W 10 -C 100
```

### 5.3 TCP フラグを使ったフィルタリング

```bash
# SYNパケットのみ（接続開始）
$ sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0'

# SYN+ACK パケット（接続応答）
$ sudo tcpdump 'tcp[tcpflags] & (tcp-syn|tcp-ack) == (tcp-syn|tcp-ack)'

# FINパケット（接続終了）
$ sudo tcpdump 'tcp[tcpflags] & tcp-fin != 0'

# RSTパケット（接続リセット — 異常終了の兆候）
$ sudo tcpdump 'tcp[tcpflags] & tcp-rst != 0'

# PSHパケット（データ送信）
$ sudo tcpdump 'tcp[tcpflags] & tcp-push != 0'
```

### 5.4 実用的な tcpdump ワンライナー集

```bash
# DNS問い合わせの監視
$ sudo tcpdump -i any port 53 -l | grep -i 'A?'

# HTTP GETリクエストの監視
$ sudo tcpdump -A -s 0 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | grep -i 'GET\|Host'

# TLSハンドシェイクの監視（Client Hello）
$ sudo tcpdump -i any 'tcp port 443 and (tcp[((tcp[12]&0xf0)>>2)]=22)' -c 20

# 再送パケットの検出
$ sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0' -c 1000 -w syn_analysis.pcap

# 特定のHTTPステータスコードを含むレスポンス
$ sudo tcpdump -A -s 0 'tcp port 80' | grep -E 'HTTP/1\.[01] [45][0-9]{2}'
```

---

## 6. Wireshark によるパケット解析

Wireshark は GUI ベースのパケット解析ツールであり、tcpdump で取得した pcap ファイルの詳細解析に適している。

### 6.1 Wireshark 表示フィルタ（コード例4）

Wireshark の表示フィルタ（Display Filter）はキャプチャ済みパケットから条件に合致するものを絞り込む。BPF（tcpdump のフィルタ構文）とは異なる独自の構文を持つ。

```
# ============================================
# IPアドレスによるフィルタ
# ============================================
ip.addr == 192.168.1.100           # 送信元 or 宛先が指定IP
ip.src == 192.168.1.100            # 送信元のみ
ip.dst == 203.0.113.10             # 宛先のみ
ip.addr == 192.168.1.0/24          # サブネットで指定

# ============================================
# ポートによるフィルタ
# ============================================
tcp.port == 443                     # TCPポート443（送信元 or 宛先）
tcp.dstport == 80                   # 宛先ポート80
tcp.srcport >= 1024                 # エフェメラルポート
udp.port == 53                      # DNS

# ============================================
# プロトコルフィルタ
# ============================================
http                                # HTTPトラフィック全体
http.request                        # HTTPリクエストのみ
http.response                       # HTTPレスポンスのみ
http.request.method == "GET"        # GETリクエスト
http.request.method == "POST"       # POSTリクエスト
http.response.code == 200           # ステータス200
http.response.code >= 400           # エラーレスポンス
http.host == "api.example.com"      # 特定ホストへのリクエスト
http.request.uri contains "/api/"   # URIに"/api/"を含む

dns                                 # DNSトラフィック
dns.qry.name == "example.com"       # 特定ドメインのDNSクエリ
dns.flags.rcode != 0                # DNSエラーレスポンス

tls                                 # TLSトラフィック
tls.handshake.type == 1             # Client Hello
tls.handshake.type == 2             # Server Hello
tls.handshake.extensions.supported_versions  # TLSバージョン情報

# ============================================
# TCP解析フィルタ
# ============================================
tcp.analysis.retransmission         # TCP再送パケット
tcp.analysis.duplicate_ack          # 重複ACK
tcp.analysis.zero_window            # ゼロウィンドウ
tcp.analysis.window_full            # ウィンドウフル
tcp.analysis.fast_retransmission    # 高速再送
tcp.analysis.lost_segment           # ロストセグメント
tcp.analysis.out_of_order           # 順序逆転パケット

tcp.flags.syn == 1 && tcp.flags.ack == 0   # SYN（接続開始）
tcp.flags.reset == 1                        # RST（リセット）
tcp.flags.fin == 1                          # FIN（接続終了）

# ============================================
# 複合フィルタ（AND / OR / NOT）
# ============================================
ip.addr == 192.168.1.100 && tcp.port == 443
http.request || http.response
!(arp || dns || icmp)               # ARP, DNS, ICMP を除外
tcp.analysis.retransmission && ip.dst == 203.0.113.10

# ============================================
# 時間ベースのフィルタ
# ============================================
frame.time >= "2024-01-15 10:00:00" && frame.time <= "2024-01-15 10:05:00"

# ============================================
# パケットサイズのフィルタ
# ============================================
frame.len > 1400                    # MTUに近いサイズのパケット
tcp.len == 0                        # データなしのTCPパケット（ACKのみ等）
```

### 6.2 Wireshark の便利な機能

```
Wireshark の解析機能:

1. Follow TCP Stream（TCP ストリーム追跡）
   → 特定の TCP 接続の会話全体をまとめて表示
   → HTTP のリクエスト/レスポンスの内容を確認するのに便利
   → 右クリック → Follow → TCP Stream

2. Follow TLS Stream
   → TLS 復号キーが設定されている場合、暗号化された内容を表示
   → Edit → Preferences → Protocols → TLS → (Pre)-Master-Secret log filename

3. Statistics メニュー
   → Conversations: ホスト間の通信量サマリ
   → Endpoints: 各ホストの通信量
   → Protocol Hierarchy: プロトコル別のトラフィック割合
   → I/O Graphs: トラフィック量の時系列グラフ
   → Flow Graph: TCP の接続フロー図

4. Expert Information（エキスパート情報）
   → Analyze → Expert Information
   → 警告やエラーを自動検出して一覧表示
   → 再送、ゼロウィンドウ、リセットなどを即座に把握

5. Coloring Rules（色分けルール）
   → TCP再送: 赤色
   → HTTPエラー: 赤色
   → TCPリセット: 赤色
   → DNS: 青色
   → カスタムルールの追加も可能
```

---

## 7. Chrome DevTools によるブラウザレベルのデバッグ

Chrome DevTools の Network タブは、ブラウザから発生する全ての HTTP リクエスト/レスポンスをリアルタイムで監視できるツールである。フロントエンド開発者にとって最も身近なネットワークデバッグ手段である。

### 7.1 Network タブの基本操作

```
Chrome DevTools の開き方:
  - F12 キー
  - Cmd + Opt + I (macOS) / Ctrl + Shift + I (Windows/Linux)
  - 右クリック → 検証(Inspect) → Network タブ

Network タブの主要機能:

  [1] リクエスト一覧
      各リクエストについて以下の情報が表示される:
      - Name:       リソース名（URL パス）
      - Status:     HTTP ステータスコード
      - Type:       リソースタイプ（document, script, stylesheet, fetch 等）
      - Initiator:  リクエストの発行元（スクリプト、パーサー等）
      - Size:       転送サイズ / リソースサイズ
      - Time:       総所要時間
      - Waterfall:  タイムラインバー

  [2] フィルタバー
      リソースタイプ別の絞り込み:
      - All:    全リクエスト
      - Fetch/XHR: API呼び出し（Ajax/Fetch）
      - JS:     JavaScriptファイル
      - CSS:    スタイルシート
      - Img:    画像
      - Media:  動画/音声
      - Font:   フォントファイル
      - Doc:    HTMLドキュメント
      - WS:     WebSocket
      - Wasm:   WebAssembly

      テキストフィルタ:
      - "api" と入力 → URLに "api" を含むリクエストのみ表示
      - "-status-code:200" → ステータス200以外を表示
      - "larger-than:100k" → 100KB以上のリソース
      - "method:POST" → POSTリクエストのみ
      - "domain:api.example.com" → 特定ドメインのみ
      - "has-response-header:set-cookie" → Cookie設定あり
```

### 7.2 Timing（タイミング）分析

```
各リクエストをクリック → Timing タブで詳細が見える:

+------------------------------------------------------------------+
| Queueing        |  ブラウザのリクエストキューで待機中                  |
|                  |  （優先度が低い、接続数上限に達している等）           |
+------------------------------------------------------------------+
| Stalled          |  接続プールの空き待ち / プロキシネゴシエーション中    |
|                  |  → 長い場合: 同一オリジンへの同時接続数上限           |
|                  |    (HTTP/1.1は6接続/オリジンが一般的)                |
+------------------------------------------------------------------+
| DNS Lookup       |  DNS名前解決の所要時間                              |
|                  |  → 長い場合: DNSサーバーの応答が遅い                 |
|                  |    dns-prefetch の導入を検討                         |
+------------------------------------------------------------------+
| Initial Conn.    |  TCP 3ウェイハンドシェイク + TLSハンドシェイク        |
|                  |  → 長い場合: ネットワーク遅延が大きい                 |
|                  |    HTTP/2やHTTP/3で接続を多重化                      |
+------------------------------------------------------------------+
| SSL              |  TLSハンドシェイクのみの時間                         |
|                  |  → 長い場合: 証明書チェーンが長い、OCSPが遅い         |
+------------------------------------------------------------------+
| Request Sent     |  リクエスト送信にかかった時間（通常は極めて短い）      |
+------------------------------------------------------------------+
| Waiting (TTFB)   |  最初のレスポンスバイト受信まで                       |
|  Time to First   |  ← サーバーの処理時間を直接反映                       |
|  Byte            |  → 長い場合: サーバー側の最適化が必要                  |
|                  |    DB クエリ、キャッシュ、アプリケーションロジック     |
+------------------------------------------------------------------+
| Content Download |  レスポンスボディの受信時間                           |
|                  |  → 長い場合: レスポンスが大きい or 帯域が狭い          |
|                  |    圧縮(gzip/brotli)、ページネーション等を検討        |
+------------------------------------------------------------------+
```

### 7.3 Waterfall（ウォーターフォール）の読み方

Waterfall はすべてのリクエストの時系列を横棒グラフで表示する機能であり、リソース読み込みのボトルネックを視覚的に把握できる。

```
Waterfall の読み方:

  Time →
  |  0ms        100ms       200ms       300ms       400ms       500ms
  |  |           |           |           |           |           |
  |
  |  index.html
  |  [==DNS==][=Conn=][=TLS=][===TTFB===][==DL==]
  |
  |  style.css                    (パーサーが発見次第ロード)
  |                          [=Conn=][TLS][=TTFB=][DL]
  |
  |  app.js                      (パーサーが発見次第ロード)
  |                          [=Conn=][TLS][==TTFB==][===DL===]
  |
  |  api/users                   (JSの実行後にfetch)
  |                                              [C][T][====TTFB====][DL]
  |
  |  avatar.png                  (APIレスポンス後に描画)
  |                                                              [C][TTFB][DL]
  |
  凡例:
    DNS   = DNS Lookup (緑色)
    Conn  = Initial Connection (オレンジ色)
    TLS   = SSL/TLS (紫色)
    TTFB  = Waiting / Time to First Byte (緑色)
    DL    = Content Download (青色)

  読み取りポイント:
    - 縦に長い空白 → リソース間の依存関係（ウォーターフォール）
    - 横に長いバー → 個別のリソース読み込みが遅い
    - 多数のリクエストが同時開始 → HTTP/2多重化が機能
    - 6本ずつ段階的に開始 → HTTP/1.1の接続数制限
```

### 7.4 Chrome DevTools の高度な使い方（コード例5）

```javascript
// ============================================
// Console タブからのネットワークデバッグ
// ============================================

// Performance API でリソースタイミングを取得
const resources = performance.getEntriesByType('resource');
resources.forEach(r => {
  console.log(`${r.name}: DNS=${r.domainLookupEnd - r.domainLookupStart}ms, ` +
              `Connect=${r.connectEnd - r.connectStart}ms, ` +
              `TTFB=${r.responseStart - r.requestStart}ms, ` +
              `Download=${r.responseEnd - r.responseStart}ms, ` +
              `Total=${r.duration}ms`);
});

// Navigation Timing API でページ全体のタイミング
const nav = performance.getEntriesByType('navigation')[0];
console.table({
  'DNS Lookup':       `${nav.domainLookupEnd - nav.domainLookupStart}ms`,
  'TCP Connection':   `${nav.connectEnd - nav.connectStart}ms`,
  'TLS Handshake':    `${nav.secureConnectionStart > 0 ?
                         nav.connectEnd - nav.secureConnectionStart : 0}ms`,
  'TTFB':             `${nav.responseStart - nav.requestStart}ms`,
  'Content Download': `${nav.responseEnd - nav.responseStart}ms`,
  'DOM Interactive':  `${nav.domInteractive - nav.fetchStart}ms`,
  'DOM Complete':     `${nav.domComplete - nav.fetchStart}ms`,
  'Load Event':       `${nav.loadEventEnd - nav.fetchStart}ms`,
});

// 遅いリクエストのみをフィルタ（500ms以上）
const slowResources = performance.getEntriesByType('resource')
  .filter(r => r.duration > 500)
  .sort((a, b) => b.duration - a.duration);
console.table(slowResources.map(r => ({
  name: r.name.split('/').pop(),
  type: r.initiatorType,
  duration: `${Math.round(r.duration)}ms`,
  size: `${r.transferSize} bytes`,
})));

// Service Worker の状態確認
navigator.serviceWorker.getRegistrations().then(registrations => {
  registrations.forEach(reg => {
    console.log('Scope:', reg.scope);
    console.log('State:', reg.active?.state);
  });
});

// WebSocket 接続のモニタリング
const originalWS = window.WebSocket;
window.WebSocket = function(...args) {
  const ws = new originalWS(...args);
  console.log('[WS] Connecting to:', args[0]);
  ws.addEventListener('open', () => console.log('[WS] Connected'));
  ws.addEventListener('close', (e) =>
    console.log('[WS] Closed:', e.code, e.reason));
  ws.addEventListener('error', (e) => console.log('[WS] Error:', e));
  return ws;
};
```

### 7.5 DevTools のネットワーク設定

```
重要な設定項目:

  [Preserve log]
    チェック → ページ遷移やリロード時にログが消えない
    リダイレクト問題のデバッグに必須

  [Disable cache]
    チェック → ブラウザキャッシュを無効化
    キャッシュの影響を排除した純粋なネットワーク計測が可能

  [Throttling]
    ネットワーク速度のシミュレーション:
    - No throttling:  制限なし（デフォルト）
    - Fast 3G:        1.6 Mbps down, 768 Kbps up, 562ms RTT
    - Slow 3G:        400 Kbps down, 400 Kbps up, 2000ms RTT
    - Offline:        完全オフライン
    - Custom:         任意の帯域幅/遅延を設定可能

  [Request Blocking]
    Cmd+Shift+P → "Show Request Blocking" で有効化
    特定のURL パターンをブロックして動作確認:
    例: *.analytics.com  → アナリティクスをブロック
    例: */api/v2/*       → 特定APIをブロック

  [HAR (HTTP Archive) エクスポート]
    → 全リクエスト/レスポンスを JSON 形式で記録
    → 問題の再現・共有に使用
    → ネットワークログを右クリック → "Save all as HAR with content"
    → https://toolbox.googleapps.com/apps/har_analyzer/ で解析可能
```

---

## 8. TLS/SSL デバッグ

HTTPS 関連の問題は、証明書の期限切れ、チェーンの不備、プロトコルバージョンの不一致など多岐にわたる。

### 8.1 openssl による TLS 接続テスト

```bash
# TLS接続テスト（証明書情報の表示）
$ openssl s_client -connect example.com:443 -servername example.com

# 証明書の有効期限確認
$ openssl s_client -connect example.com:443 -servername example.com 2>/dev/null \
  | openssl x509 -noout -dates
# notBefore=Jan  1 00:00:00 2024 GMT
# notAfter=Dec 31 23:59:59 2024 GMT

# 証明書チェーン全体の表示
$ openssl s_client -connect example.com:443 -servername example.com -showcerts

# 特定のTLSバージョンで接続
$ openssl s_client -connect example.com:443 -tls1_2
$ openssl s_client -connect example.com:443 -tls1_3

# 対応している暗号スイートの確認
$ openssl s_client -connect example.com:443 -cipher 'ECDHE-RSA-AES256-GCM-SHA384'

# 証明書の詳細情報（Subject, Issuer, SAN 等）
$ openssl s_client -connect example.com:443 2>/dev/null \
  | openssl x509 -noout -text | head -30

# OCSP Stapling の確認
$ openssl s_client -connect example.com:443 -status 2>/dev/null \
  | grep -A 5 "OCSP Response"

# ALPN（Application-Layer Protocol Negotiation）の確認
$ openssl s_client -connect example.com:443 -alpn h2,http/1.1 2>/dev/null \
  | grep "ALPN"
```

### 8.2 TLS 問題の判断基準

| 症状 | 考えられる原因 | 確認方法 |
|:-----|:-------------|:---------|
| `certificate has expired` | 証明書の有効期限切れ | `openssl x509 -noout -dates` |
| `unable to verify the first certificate` | 中間証明書の欠落 | `-showcerts` で証明書チェーンを確認 |
| `certificate verify failed` | ルートCA不信任 / 自己署名 | CA証明書の確認 |
| `wrong version number` | 非TLSポートにTLS接続 | ポート番号の確認 |
| `handshake failure` | 暗号スイートの不一致 | `-cipher` で個別にテスト |
| `tlsv1 alert protocol version` | TLSバージョン非対応 | `-tls1_2` `-tls1_3` で個別テスト |
| `sslv3 alert handshake failure` | SNIが必要 | `-servername` を指定 |

---

## 9. トラブルシューティング体系

### 9.1 問題の分類

ネットワーク問題は大きく以下の 4 カテゴリに分類できる。

```
+========================+========================+
|    接続不能            |    間欠的障害           |
|    (Connection Failed) |    (Intermittent)       |
|                        |                         |
|  - DNS解決失敗         |  - 時々タイムアウト      |
|  - ポート未開放        |  - パケットロス          |
|  - ファイアウォール    |  - 負荷依存の障害        |
|  - ルーティング異常    |  - DNS TTL問題          |
|                        |                         |
+========================+========================+
|    パフォーマンス低下   |    アプリケーション      |
|    (Slow Performance)  |    エラー                |
|                        |    (Application Error)   |
|  - 高レイテンシ        |  - HTTP 4xx/5xx         |
|  - 低スループット      |  - TLS/SSL エラー        |
|  - TCP再送多発         |  - CORS問題              |
|  - MTU問題             |  - WebSocket切断         |
|                        |                         |
+========================+========================+
```

### 図2: トラブルシューティング決定木

```
+================================================================+
|            トラブルシューティング決定木                            |
+================================================================+

  [Webページが表示されない / APIが応答しない]
      |
      v
  (1) ping <host> は通るか？
      |
      +--- NO ---> (A) IPアドレスは正しいか？
      |                 |
      |                 +--- dig +short <host> で確認
      |                 |
      |                 +--- IP が返らない
      |                 |     → DNS問題: /etc/resolv.conf, DNSサーバー確認
      |                 |
      |                 +--- IP は返るが ping が通らない
      |                       → traceroute でどのホップで止まるか確認
      |                       → ファイアウォールで ICMP がブロック？
      |                       → ルーティング問題？
      |
      +--- YES --> (2) nc -zv <host> <port> は通るか？
                       |
                       +--- NO ---> ポートが閉じている
                       |            → サービスが起動しているか確認
                       |            → ss -tlnp でリッスン状態を確認
                       |            → ファイアウォールルールを確認
                       |            → Security Group / ACL を確認
                       |
                       +--- YES --> (3) curl -v <URL> の結果は？
                                       |
                                       +--- TLSエラー
                                       |    → openssl s_client で証明書確認
                                       |    → 有効期限、チェーン、SNI
                                       |
                                       +--- HTTP 3xx
                                       |    → リダイレクト先の確認
                                       |    → curl -L で追跡
                                       |    → 無限リダイレクトでないか
                                       |
                                       +--- HTTP 4xx
                                       |    → 401: 認証情報の確認
                                       |    → 403: 権限/IP制限の確認
                                       |    → 404: URLパスの確認
                                       |    → 429: レート制限の確認
                                       |
                                       +--- HTTP 5xx
                                       |    → サーバーログを確認
                                       |    → 502: バックエンドの死活確認
                                       |    → 503: 過負荷/メンテナンス
                                       |    → 504: バックエンドのタイムアウト
                                       |
                                       +--- タイムアウト
                                       |    → TTFB が遅いか？
                                       |    → サーバー処理が遅いか？
                                       |    → DB やキャッシュの確認
                                       |
                                       +--- 正常 (200) だがページが壊れている
                                            → DevToolsでJSエラー確認
                                            → APIレスポンスの内容確認
                                            → CORSエラーの確認
```

### 9.2 レイヤー別デバッグコマンド一覧

```bash
# ============================================
# L1-L2: 物理層 / データリンク層
# ============================================
$ ip link show                     # インタフェースの状態
$ ethtool eth0                     # NICの詳細情報
$ ip neigh show                    # ARPテーブル
$ arp -a                           # ARPテーブル（レガシー）

# ============================================
# L3: ネットワーク層
# ============================================
$ ip addr show                     # IPアドレスの確認
$ ip route show                    # ルーティングテーブル
$ ip route get 203.0.113.10        # 特定IPへのルート
$ ping -c 5 example.com            # 疎通確認
$ traceroute example.com           # 経路確認
$ mtr --report example.com         # 経路+パケットロス

# ============================================
# L4: トランスポート層
# ============================================
$ ss -tlnp                         # TCP LISTENポート
$ ss -t state established          # 確立済みTCP接続
$ nc -zv example.com 443           # ポート疎通確認
$ sudo tcpdump -i eth0 port 443    # パケットキャプチャ

# ============================================
# L5-L7: アプリケーション層
# ============================================
$ dig +short example.com           # DNS解決
$ curl -v https://example.com      # HTTP通信テスト
$ openssl s_client -connect example.com:443   # TLS確認
# Chrome DevTools → Network タブ    # ブラウザレベル
```

### 9.3 よくある問題と対処法

#### 問題1: CLOSE_WAIT の大量蓄積

```bash
# 症状: CLOSE_WAIT 状態の接続が数百〜数千蓄積している
$ ss -t state close-wait | wc -l
523

# 原因の特定: どのプロセスが CLOSE_WAIT を持っているか
$ ss -t state close-wait -p | awk '{print $NF}' | sort | uniq -c | sort -rn
  489 users:(("java",pid=12345,fd=892))
   34 users:(("python3",pid=6789,fd=45))

# 解説:
# CLOSE_WAIT は「相手が FIN を送ったが、こちらが close() していない」状態
# アプリケーションにソケットリーク（接続を閉じ忘れるバグ）がある
#
# 対処:
# 1. アプリケーションのコードを修正（try-with-resources, finally で close）
# 2. HTTP クライアントのコネクションプール設定を見直す
# 3. keepalive タイムアウトの設定を確認
```

#### 問題2: TIME_WAIT の大量蓄積

```bash
# 症状: TIME_WAIT が数万に達し、エフェメラルポートが枯渇
$ ss -t state time-wait | wc -l
28456

# エフェメラルポート範囲の確認
$ cat /proc/sys/net/ipv4/ip_local_port_range
32768   60999
# 利用可能ポート数: 60999 - 32768 = 28231
# → TIME_WAIT が 28231 を超えると新規接続不可

# 対処（根本的）:
# 1. HTTP keep-alive を有効にして接続の再利用を促進
# 2. コネクションプーリングを導入
# 3. HTTP/2 で接続の多重化

# 対処（一時的、副作用に注意）:
$ sudo sysctl -w net.ipv4.tcp_tw_reuse=1        # TIME_WAIT の再利用を許可
$ sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"  # ポート範囲拡大
```

#### 問題3: CORS エラー

```
# Chrome DevTools Console で表示されるエラー:
# Access to fetch at 'https://api.example.com/data'
# from origin 'https://www.example.com' has been blocked by CORS policy:
# No 'Access-Control-Allow-Origin' header is present on the requested resource.

# 確認方法:
$ curl -v -X OPTIONS https://api.example.com/data \
  -H "Origin: https://www.example.com" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: Authorization"

# 期待されるレスポンスヘッダー:
# Access-Control-Allow-Origin: https://www.example.com
# Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
# Access-Control-Allow-Headers: Authorization, Content-Type
# Access-Control-Max-Age: 86400

# よくある原因:
# 1. サーバー側で CORS ヘッダーが設定されていない
# 2. ワイルドカード (*) は認証付きリクエストで使用不可
# 3. プリフライトリクエスト (OPTIONS) が 200 以外を返している
# 4. レスポンスに必要なヘッダーが含まれていない
```

---

## 10. デバッグツール比較表

### 比較表1: パケットキャプチャツールの比較

| 特性 | tcpdump | Wireshark | tshark | ngrep |
|:-----|:--------|:----------|:-------|:------|
| インタフェース | CLI | GUI | CLI | CLI |
| リアルタイム表示 | 可 | 可 | 可 | 可 |
| pcap保存 | 可 | 可 | 可 | 可 |
| pcap読み込み | 可 | 可 | 可 | 可 |
| フィルタ構文 | BPF | Display Filter | Display Filter | 正規表現 |
| プロトコル解析深度 | 基本的 | 非常に詳細 | 非常に詳細 | 基本的 |
| SSH経由の利用 | 容易 | 不可（X転送除く） | 容易 | 容易 |
| サーバーでの利用 | 最適 | 不向き | 最適 | 適 |
| メモリ使用量 | 少 | 多 | 中 | 少 |
| 学習コスト | 中 | 中〜高 | 高 | 低 |
| 推奨用途 | サーバー上でのキャプチャ | GUIでの詳細解析 | スクリプト連携 | テキスト検索 |

### 比較表2: HTTP デバッグツールの比較

| 特性 | curl | HTTPie | wget | Postman | DevTools |
|:-----|:-----|:-------|:-----|:--------|:---------|
| インタフェース | CLI | CLI | CLI | GUI | ブラウザ内蔵 |
| JSON整形出力 | jq併用 | 組み込み | 不可 | 組み込み | 組み込み |
| リクエスト保存 | スクリプト化 | スクリプト化 | 不可 | コレクション | HAR |
| タイミング計測 | `-w` オプション | `--print=h` | 不可 | 組み込み | Timing タブ |
| TLS詳細 | `-v` で表示 | 限定的 | 不可 | 限定的 | Security タブ |
| HTTP/2対応 | `--http2` | 対応 | 非対応 | 対応 | 対応 |
| HTTP/3対応 | `--http3` | 未対応 | 非対応 | 未対応 | 対応 |
| WebSocket | `--ws` (7.86+) | 未対応 | 非対応 | 対応 | WS フレーム表示 |
| 自動化適性 | 高 | 高 | 高 | 中（Newman） | 低 |
| Cookie管理 | `-b/-c` | `--session` | 組み込み | 自動 | 自動 |
| 認証サポート | 多種 | 多種 | Basic | 多種 | ブラウザ依存 |
| 学習コスト | 中 | 低 | 低 | 低 | 低 |

---

## 11. アンチパターン

### アンチパターン1: 「とりあえず再起動」症候群

```
+================================================================+
|  アンチパターン: 原因を調べずにサービスを再起動する               |
+================================================================+

  問題発生
      |
      v
  「とりあえず再起動しよう」          ← ここが問題
      |
      v
  サービス再起動
      |
      +--- 一時的に直る --------> 安心してしまう
      |                                |
      |                                v
      |                          再発する（数時間〜数日後）
      |                                |
      |                                v
      |                          「また再起動すればいいか」
      |                                |
      |                                v
      |                          根本原因が不明のまま繰り返す
      |                          → 信頼性の低いシステムになる
      |
      +--- 直らない ------------> さらに混乱する
                                  → ログが消えて調査困難に

  正しいアプローチ:
  +---------------------------------------------------------+
  | 1. まずログを確認する（再起動前に！）                      |
  |    $ journalctl -u myservice --since "1 hour ago"        |
  |    $ tail -100 /var/log/myservice/error.log              |
  |                                                         |
  | 2. 現在の状態を記録する                                   |
  |    $ ss -tlnp > /tmp/socket_state.txt                    |
  |    $ ps auxf > /tmp/process_state.txt                    |
  |    $ top -b -n 1 > /tmp/resource_state.txt               |
  |    $ sudo tcpdump -w /tmp/before_restart.pcap -c 1000 &  |
  |                                                         |
  | 3. 原因の仮説を立てる                                     |
  |                                                         |
  | 4. 仮説を検証する                                         |
  |                                                         |
  | 5. 根本原因を修正する                                     |
  |                                                         |
  | 6. 修正できない場合のみ、記録を残して再起動する             |
  +---------------------------------------------------------+
```

### アンチパターン2: 本番環境での無差別パケットキャプチャ

```
+================================================================+
|  アンチパターン: 本番環境で無制限にtcpdumpを実行する              |
+================================================================+

  やりがちなミス:
    $ sudo tcpdump -i eth0 -w capture.pcap
    → フィルタなし、パケット数制限なし、ファイルサイズ制限なし

  何が起こるか:
    1. ディスクが急速に消費される（1Gbpsなら約100MB/秒）
    2. CPU負荷が増加する（特にASCII表示 -A オプション時）
    3. 機密データ（パスワード、トークン）がキャプチャされる
    4. pcapファイルが巨大すぎて解析困難になる

  正しいアプローチ:
  +---------------------------------------------------------+
  | 1. フィルタを必ず指定する                                 |
  |    $ sudo tcpdump host 203.0.113.10 and port 443         |
  |                                                         |
  | 2. パケット数を制限する                                   |
  |    $ sudo tcpdump -c 1000 ...                            |
  |                                                         |
  | 3. ファイルサイズを制限する                                |
  |    $ sudo tcpdump -C 100 -W 5 ...                        |
  |    (100MBごとにローテーション、最大5ファイル)               |
  |                                                         |
  | 4. キャプチャ時間を制限する                                |
  |    $ timeout 60 sudo tcpdump ...                         |
  |    (60秒で自動停止)                                       |
  |                                                         |
  | 5. 機密データへの配慮                                     |
  |    - スナップショットサイズを制限: -s 96                   |
  |      (ヘッダーのみキャプチャ、ペイロードは含まない)          |
  |    - キャプチャファイルのアクセス権を制限                    |
  |    - 解析後は速やかに削除                                  |
  +---------------------------------------------------------+
```

---

## FAQ（よくある質問）

### Q1: tcpdump と Wireshark はどのように使い分けるべきか？

**A:** 使用環境と目的に応じて使い分ける。

**tcpdump を使うべきケース:**
- **サーバー上でのキャプチャ**: SSH 経由でリモートサーバーにアクセスし、その場でパケットをキャプチャする場合（GUI 不要）
- **自動化・スクリプト化**: cron や監視スクリプトからパケットキャプチャを定期実行する場合
- **リアルタイム監視**: ログをリアルタイムで流し見する場合（`tcpdump -A port 80 | grep "GET"`）
- **軽量な環境**: メモリやディスク容量が限られたサーバー、組み込み機器
- **フィルタリング重視**: BPF フィルタで高速にパケットを絞り込む場合

**Wireshark を使うべきケース:**
- **詳細なプロトコル解析**: HTTP/2, TLS, DNS, TCP の詳細なフィールドを確認する場合
- **視覚的な分析**: ストリーム追跡、フローグラフ、I/O グラフなどの可視化機能を使う場合
- **エキスパート情報の活用**: 再送、ゼロウィンドウ、重複 ACK などの問題を自動検出する場合
- **pcap ファイルの解析**: tcpdump で取得した pcap ファイルを後からじっくり解析する場合
- **初学者向け**: CLI に慣れていない場合、GUI の方が直感的

**推奨ワークフロー:**
1. **サーバー上で tcpdump でキャプチャ**: `sudo tcpdump -w capture.pcap -c 1000 port 443`
2. **ローカルに pcap ファイルを転送**: `scp server:/tmp/capture.pcap .`
3. **Wireshark で詳細解析**: GUI で Follow TCP Stream, Expert Information を活用

**tshark（Wireshark の CLI 版）という選択肢:**
- Wireshark の表示フィルタを CLI で使える
- サーバー上でも Wireshark の高度な解析機能を利用可能
- 例: `tshark -r capture.pcap -Y "http.response.code == 500"`

### Q2: DNS のトラブルシューティング手順は？

**A:** DNS 問題は以下のフローで体系的に切り分ける。

**Step 1: 名前解決が可能か確認**

```bash
# 基本的な名前解決テスト
$ dig +short example.com
203.0.113.10
```

**結果が返らない場合:**

```bash
# Step 1-1: 別の DNS サーバーで試す
$ dig @8.8.8.8 +short example.com       # Google Public DNS
$ dig @1.1.1.1 +short example.com       # Cloudflare DNS
$ dig @208.67.222.222 +short example.com # OpenDNS

# 返る場合 → ローカル DNS サーバーの問題
#   - /etc/resolv.conf の nameserver 設定を確認
#   - 社内 DNS サーバーの障害を確認
#   - DNS キャッシュをクリア: sudo systemd-resolve --flush-caches

# 返らない場合 → ドメイン自体の問題
#   - ドメインの登録状況を確認: whois example.com
#   - 権威 DNS サーバーを確認: dig +trace example.com
```

**Step 2: 返ってきた IP は正しいか確認**

```bash
# 期待する IP と実際の IP を比較
$ dig +short example.com
192.0.2.1  # これは期待通りか？

# 別の DNS サーバーとも比較
$ dig @8.8.8.8 +short example.com
203.0.113.10  # ローカル DNS と値が異なる！

# → DNS キャッシュポイズニング or 古いキャッシュの可能性
```

**Step 3: DNS 解決時間が遅い場合**

```bash
# DNS 解決時間を計測
$ dig example.com | grep "Query time"
;; Query time: 523 msec  # 500ms 以上は遅い

# 原因の切り分け:
# 1. DNS サーバーが遠い → 近いパブリック DNS に変更
# 2. DNS サーバーが過負荷 → 別の DNS サーバーを試す
# 3. DNS リゾルバの障害 → systemd-resolved / dnsmasq の再起動

# traceroute で DNS サーバーまでの経路確認
$ traceroute 8.8.8.8
```

**Step 4: /etc/hosts による上書き確認**

```bash
# /etc/hosts でローカルオーバーライドされていないか確認
$ grep example.com /etc/hosts
127.0.0.1  example.com  # ← これが原因でローカルホストに接続していた!

# /etc/hosts は DNS より優先されるため、意図しない設定が残っている場合がある
```

**Step 5: DNS の伝播待ち（DNS 変更直後の場合）**

```bash
# 権威 DNS サーバーに直接問い合わせ
$ dig @ns1.example-dns.com example.com

# 権威サーバーでは新しい IP、キャッシュサーバーでは古い IP が返る場合:
# → TTL が経過するまで待つ（通常 300-3600 秒）

# 複数地域の DNS サーバーで確認
# - https://www.whatsmydns.net/ で全世界の DNS 伝播状況を確認可能
```

**よくある DNS 問題と対処:**

| 症状 | 原因 | 対処 |
|------|------|------|
| `NXDOMAIN` | ドメインが存在しない | ドメイン名のタイポ確認、whois で登録状況確認 |
| `SERVFAIL` | DNS サーバーの障害 | 別の DNS サーバーを試す（8.8.8.8 等） |
| `connection timed out` | DNS サーバーに到達不能 | ファイアウォール、ルーティング確認 |
| 古い IP が返る | DNS キャッシュ | キャッシュクリア、TTL 経過待ち |
| 名前解決が遅い | 遠隔 DNS サーバー | ローカル DNS キャッシュサーバー導入 |

### Q3: ネットワークレイテンシの問題はどう特定するか？

**A:** レイテンシ問題は以下の手順で切り分ける。

**Step 1: 全体のレイテンシを計測**

```bash
# curl でフェーズごとの時間を計測
$ curl -o /dev/null -s -w "\
  DNS:      %{time_namelookup}s\n\
  Connect:  %{time_connect}s\n\
  TLS:      %{time_appconnect}s\n\
  TTFB:     %{time_starttransfer}s\n\
  Total:    %{time_total}s\n" \
  https://api.example.com/data

# 出力例:
# DNS:      0.015s  ← DNS 解決は高速
# Connect:  0.045s  ← TCP 接続は正常
# TLS:      0.089s  ← TLS ハンドシェイクは正常
# TTFB:     1.234s  ← サーバー処理が遅い！
# Total:    1.456s
```

**各フェーズの判断基準:**

| フェーズ | 正常範囲 | 遅い場合の原因 | 対処 |
|---------|---------|--------------|------|
| DNS | < 50ms | DNS サーバーが遅い、遠い | 近いパブリック DNS、DNS プリフェッチ |
| Connect | < 100ms | ネットワーク遅延が大きい | CDN 導入、サーバーの地理的分散 |
| TLS | < 150ms | 証明書チェーンが長い、OCSP 遅い | OCSP Stapling、証明書チェーン最適化 |
| TTFB | < 500ms | サーバー処理が遅い | DB 最適化、キャッシュ導入、インデックス追加 |
| Download | 帯域依存 | 大きなレスポンス、狭い帯域 | gzip/Brotli 圧縮、レスポンスサイズ削減 |

**Step 2: ネットワーク経路の遅延を特定**

```bash
# mtr でリアルタイム経路監視
$ mtr --report -c 100 api.example.com

# 出力例:
# HOST: myhost                Loss%   Snt   Last   Avg  Best  Wrst StDev
#   1. gateway                 0.0%   100    1.2   1.3   1.0   2.5   0.2
#   2. isp-router              0.0%   100   15.2  16.1  14.5  25.3   2.1
#   3. isp-core                2.0%   100   45.3  46.8  44.2  78.5   5.3  ← パケットロス!
#   4. peering-point           0.0%   100   48.1  49.2  47.5  55.1   1.8
#   5. api.example.com         0.0%   100   50.2  51.3  49.8  58.2   2.0

# ホップ 3 でパケットロス 2% → この区間に問題がある
# → ISP に問い合わせ or 別経路（VPN 等）を検討
```

**Step 3: サーバー処理時間の内訳を特定**

```bash
# Chrome DevTools の Network タブで確認:
# - Waiting (TTFB) が長い → サーバー側の問題
#   - DB クエリが遅い → EXPLAIN ANALYZE で実行計画確認
#   - 外部 API 呼び出しが遅い → タイムアウト設定、キャッシュ導入
#   - CPU 使用率が高い → プロファイリング（pprof, py-spy 等）

# サーバーログで処理時間を記録
# Nginx の例:
log_format timed_combined '$remote_addr - $remote_user [$time_local] '
                          '"$request" $status $body_bytes_sent '
                          '"$http_referer" "$http_user_agent" '
                          'rt=$request_time uct=$upstream_connect_time '
                          'uht=$upstream_header_time urt=$upstream_response_time';

# request_time が大きい → アプリケーション処理が遅い
# upstream_*_time が大きい → バックエンドサーバーが遅い
```

**Step 4: 間欠的な遅延の場合**

```bash
# 連続計測して統計を取る
$ for i in {1..100}; do
    curl -o /dev/null -s -w "%{time_total}\n" https://api.example.com/data
  done | awk '{sum+=$1; if($1>max) max=$1; if(NR==1 || $1<min) min=$1} END {print "Avg:", sum/NR, "Min:", min, "Max:", max}'

# 出力例:
# Avg: 0.234s  Min: 0.189s  Max: 2.345s
# → Max が異常に大きい = 間欠的な遅延が発生

# 原因:
# - GC（ガベージコレクション）による一時停止
# - DB コネクションプールの枯渇
# - キャッシュミス時のスロークエリ
# - サーバーのスワップ発生
```

**結論: レイテンシ問題は curl の時間計測 → mtr で経路確認 → サーバーログ/APM で内訳特定、の順で切り分ける。**

---

## まとめ

| 概念 | ポイント |
|------|---------|
| デバッグフロー | 症状整理 → レイヤー特定（OSI モデル）→ 仮説立案 → 検証 → 修正 → 文書化 |
| curl | HTTP デバッグの万能ツール、`-w` でタイミング計測、`-v` で詳細表示 |
| DNS | dig が最も詳細、+trace で全経路追跡、@8.8.8.8 で代替 DNS 確認 |
| 疎通確認 | ping → traceroute/mtr → nc でポート確認の順 |
| ソケット | ss（推奨）or netstat、CLOSE_WAIT/TIME_WAIT の蓄積に注意 |
| パケットキャプチャ | tcpdump（サーバー）+ Wireshark（詳細解析）の組み合わせ |
| TLS | openssl s_client で証明書確認、-showcerts でチェーン表示 |
| ブラウザ | Chrome DevTools の Network タブ、Timing/Waterfall/HAR エクスポート |
| トラブルシューティング | レイヤー別に上から順に確認、ログは再起動前に必ず保存 |

---

## 次に読むべきガイド

ネットワークデバッグの手法を習得したら、次は以下のトピックに進むことを推奨する。

- **[パフォーマンス最適化](./03-performance.md)**: デバッグで特定したボトルネックを解消するための総合的なネットワークパフォーマンスチューニング手法を学ぶ
- **[HTTP の詳細](../02-http/)**: HTTP プロトコルの仕様、キャッシュ制御、セキュリティヘッダーを深く理解してデバッグ精度を向上させる
- **[TCP/IP プロトコル](../01-protocols/)**: パケットレベルのデバッグをより深く行うために、TCP の再送制御、フロー制御、輻輳制御を学ぶ

---

## 参考文献

1. Stevens, W. R. "TCP/IP Illustrated, Volume 1: The Protocols." Addison-Wesley, 2011.
2. tcpdump.org. "tcpdump Manual." tcpdump.org, 2024.
3. Wireshark Foundation. "Wireshark User's Guide." wireshark.org, 2024.
4. Mozilla Developer Network. "Chrome DevTools Network Reference." developer.mozilla.org, 2024.
5. RFC 1035. "Domain Names - Implementation and Specification." IETF, 1987.
6. Grigorik, I. "High Performance Browser Networking." O'Reilly, 2013.


