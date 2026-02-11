# ネットワークデバッグ

> ネットワーク問題の切り分けと解決に必要なツールと手法を学ぶ。curl、tcpdump、Wireshark、Chrome DevToolsを使いこなし、効率的にトラブルシューティングする。

## この章で学ぶこと

- [ ] 主要なネットワークデバッグツールの使い方を理解する
- [ ] 問題の切り分け手法を把握する
- [ ] Chrome DevToolsのネットワークタブを活用する

---

## 1. curl

```bash
# 基本的なGETリクエスト
$ curl https://api.example.com/users

# レスポンスヘッダーを表示
$ curl -I https://api.example.com/users

# リクエスト/レスポンスの詳細表示
$ curl -v https://api.example.com/users

# POSTリクエスト（JSON）
$ curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token123" \
  -d '{"name": "Taro", "email": "taro@example.com"}'

# タイミング情報の表示
$ curl -o /dev/null -s -w "\
  DNS:        %{time_namelookup}s\n\
  TCP:        %{time_connect}s\n\
  TLS:        %{time_appconnect}s\n\
  First Byte: %{time_starttransfer}s\n\
  Total:      %{time_total}s\n\
  Status:     %{http_code}\n\
  Size:       %{size_download} bytes\n" \
  https://api.example.com/users

# リダイレクトを追跡
$ curl -L -v https://example.com

# レスポンスボディをファイルに保存
$ curl -o response.json https://api.example.com/data

# HTTP/2で接続
$ curl --http2 -v https://api.example.com/users
```

---

## 2. DNS デバッグ

```bash
# dig — DNS問い合わせ
$ dig example.com              # Aレコード
$ dig example.com MX           # MXレコード
$ dig example.com +short       # 結果のみ
$ dig @8.8.8.8 example.com    # Google DNSに問い合わせ
$ dig +trace example.com       # 解決過程を表示

# nslookup — DNS問い合わせ（簡易版）
$ nslookup example.com
$ nslookup -type=CNAME www.example.com

# host — DNS問い合わせ（最も簡単）
$ host example.com
$ host -t MX example.com

# DNSキャッシュのクリア
$ sudo dscacheutil -flushcache   # macOS
$ sudo systemd-resolve --flush-caches  # Linux (systemd)

DNS問題の切り分け:
  1. dig +short example.com → IPが返るか？
  2. dig @8.8.8.8 example.com → 別DNSで確認
  3. dig +trace example.com → どの段階で問題か
  4. 正しいIPが返るのにアクセスできない → DNSではなくネットワーク
```

---

## 3. ネットワーク接続デバッグ

```bash
# ping — 疎通確認
$ ping example.com
$ ping -c 5 example.com  # 5回だけ

# traceroute — 経路確認
$ traceroute example.com
$ traceroute -T example.com  # TCP traceroute（ファイアウォール通過用）

# mtr — ping + traceroute の統合（リアルタイム）
$ mtr example.com

# nc (netcat) — ポート接続テスト
$ nc -zv example.com 443     # TCPポート443が開いているか
$ nc -zvu example.com 53     # UDPポート53

# ss — ソケット状態の確認（netstatの後継）
$ ss -tlnp                    # リッスン中のTCPポート
$ ss -s                       # ソケット統計
$ ss -t state time-wait       # TIME_WAIT状態の接続

# lsof — ポートを使用しているプロセス
$ lsof -i :8080               # ポート8080を使っているプロセス
$ lsof -i -P -n               # 全ネットワーク接続

接続問題の切り分け:
  1. ping example.com → ICMPが通るか
  2. traceroute → どのホップで止まるか
  3. nc -zv host 443 → ポートが開いているか
  4. curl -v → HTTP レベルの問題か
  5. openssl s_client → TLS の問題か
```

---

## 4. tcpdump / Wireshark

```bash
# tcpdump — パケットキャプチャ（CLI）

# 特定ホストのトラフィック
$ sudo tcpdump host example.com

# 特定ポートのトラフィック
$ sudo tcpdump port 443

# TCP SYNパケットのみ
$ sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0'

# DNS（ポート53）のトラフィック
$ sudo tcpdump port 53

# ファイルに保存（Wiresharkで開ける）
$ sudo tcpdump -w capture.pcap -c 1000

# 読みやすい形式で表示
$ sudo tcpdump -A port 80  # ASCII
$ sudo tcpdump -X port 80  # HEX + ASCII

Wireshark:
  → GUI のパケット分析ツール
  → tcpdumpの.pcapファイルを読み込み可能
  → フィルター例:
     http.request.method == "GET"
     tcp.port == 443
     ip.addr == 192.168.1.100
     dns.qry.name == "example.com"
     tcp.analysis.retransmission  （再送パケット）

  → Follow TCP Stream で会話全体を表示
  → Statistics > I/O Graphs でトラフィック推移
```

---

## 5. Chrome DevTools（Networkタブ）

```
Chrome DevTools のNetworkタブ:
  → F12 または Cmd+Opt+I で開く

主要な機能:
  ① リクエスト一覧:
     → メソッド、URL、ステータス、サイズ、時間
     → フィルター: XHR, JS, CSS, Img, Media, Font, WS

  ② タイミング分析（Timing）:
     Queueing:         キュー待ち
     Stalled:          接続プール待ち
     DNS Lookup:       DNS解決
     Initial Connection: TCP接続
     SSL:              TLSハンドシェイク
     Request Sent:     リクエスト送信
     Waiting (TTFB):   最初のバイト受信まで ← サーバー処理時間
     Content Download: コンテンツ受信

  ③ Waterfall:
     → 全リクエストのタイムライン表示
     → ボトルネックの視覚的特定

  ④ スロットリング:
     → Slow 3G, Fast 3G, Offline のシミュレーション
     → カスタムプロファイルも作成可能

  ⑤ HAR（HTTP Archive）エクスポート:
     → 全リクエスト/レスポンスを記録
     → 問題再現に使用

デバッグのコツ:
  ✓ "Preserve log" を有効化（リダイレクト時にログが消えない）
  ✓ "Disable cache" を有効化（キャッシュの影響を排除）
  ✓ Cmd+Shift+P → "Show Request Blocking" → 特定URLをブロック
```

---

## 6. トラブルシューティングフロー

```
問題: Webページが表示されない

  Step 1: DNS確認
    $ dig +short example.com
    → IPが返らない → DNS問題
    → IPが返る → Step 2へ

  Step 2: 接続確認
    $ nc -zv example.com 443
    → 接続できない → ネットワーク/ファイアウォール問題
    → 接続できる → Step 3へ

  Step 3: TLS確認
    $ openssl s_client -connect example.com:443
    → エラー → 証明書/TLS問題
    → 成功 → Step 4へ

  Step 4: HTTP確認
    $ curl -v https://example.com
    → ステータスコードで判断
    → 3xx → リダイレクトの確認
    → 4xx → クライアント側の問題
    → 5xx → サーバー側の問題

  Step 5: パフォーマンス確認
    $ curl -o /dev/null -s -w "TTFB: %{time_starttransfer}s\n" url
    → TTFB が遅い → サーバー処理の問題
    → ダウンロードが遅い → 帯域/ファイルサイズの問題
```

---

## まとめ

| ツール | 用途 |
|--------|------|
| curl | HTTPリクエストの送信・タイミング計測 |
| dig/nslookup | DNS問い合わせ |
| ping/traceroute | 疎通確認・経路確認 |
| tcpdump/Wireshark | パケットキャプチャ・分析 |
| Chrome DevTools | ブラウザレベルのネットワーク分析 |

---

## 次に読むべきガイド
→ [[03-performance.md]] — ネットワーク最適化

---

## 参考文献
1. Everything curl. "curl Documentation." curl.se, 2024.
2. Wireshark Foundation. "Wireshark User's Guide." 2024.
