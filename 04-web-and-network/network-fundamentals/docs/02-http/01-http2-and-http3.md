# HTTP/2とHTTP/3

> HTTP/2はバイナリフレーミングと多重化でHTTP/1.1の性能問題を解決。HTTP/3はQUICベースでさらに高速化。Webパフォーマンスを左右するプロトコルの進化を理解する。

## この章で学ぶこと

- [ ] HTTP/1.1の問題点とHTTP/2が解決した課題を理解する
- [ ] HTTP/2の主要機能（多重化、HPACK、サーバープッシュ等）を把握する
- [ ] HTTP/3とQUICプロトコルの設計思想と利点を学ぶ
- [ ] 各プロトコルのサーバー設定と検証方法を習得する
- [ ] パフォーマンス計測ツール（h2load、curl）を使いこなす
- [ ] 本番環境へのHTTP/2・HTTP/3導入の判断基準を理解する

---

## 1. HTTP/1.1の問題点 -- なぜHTTP/2が必要だったか

### 1.1 Head-of-Line Blocking (HoL Blocking)

HTTP/1.1の最も深刻な問題は Head-of-Line Blocking である。1つのTCP接続上で
リクエストとレスポンスは厳密に順番に処理される。前のレスポンスが完了するまで
次のリクエストの処理に取りかかれない。

```
HTTP/1.1 の Head-of-Line Blocking:

  Client                          Server
    │                               │
    │──── GET /page.html ──────────→│
    │                               │ (処理中...)
    │←── 200 OK + HTML ────────────│
    │                               │
    │──── GET /style.css ──────────→│  ← HTML完了後にしか送れない
    │                               │ (処理中...)
    │←── 200 OK + CSS ─────────────│
    │                               │
    │──── GET /app.js ─────────────→│  ← CSS完了後にしか送れない
    │                               │ (処理中...)
    │←── 200 OK + JS ──────────────│
    │                               │

  合計時間 = RTT × 3 + 各処理時間の合計
  → リソースが多いページほど深刻
```

HTTP/1.1にはパイプライニング (Pipelining) という仕組みが仕様として存在するが、
実際にはほとんどのサーバーとプロキシで正しく実装されておらず、ブラウザ側でも
デフォルト無効になっている。パイプライニングはレスポンスの順序を保証する必要が
あるため、根本的なHoL Blocking解消にはならなかった。

### 1.2 多数のTCP接続によるリソース消費

ブラウザはHoL Blockingを緩和するために、同一オリジンに対して最大6本の
TCP接続を同時に張る。しかしこれには次のコストが伴う。

```
同一ドメインへの6接続（HTTP/1.1）:

  ┌────────────────────────────────────────────────┐
  │ ブラウザ (example.com へ 6本)                    │
  │                                                  │
  │  接続1: ─── 3-way HS ─── TLS HS ─── req/res ───│
  │  接続2: ─── 3-way HS ─── TLS HS ─── req/res ───│
  │  接続3: ─── 3-way HS ─── TLS HS ─── req/res ───│
  │  接続4: ─── 3-way HS ─── TLS HS ─── req/res ───│
  │  接続5: ─── 3-way HS ─── TLS HS ─── req/res ───│
  │  接続6: ─── 3-way HS ─── TLS HS ─── req/res ───│
  │                                                  │
  │  各接続ごとに:                                    │
  │    TCP 3-way HS: 1 RTT                           │
  │    TLS 1.2 HS:   2 RTT                           │
  │    合計:         3 RTT × 6接続 = 18 RTT の       │
  │                  ハンドシェイクオーバーヘッド       │
  │                                                  │
  │  サーバー側の影響:                                 │
  │    ・ソケットリソース × 6（メモリ、FD消費）         │
  │    ・TCPスロースタート × 6（帯域の非効率利用）     │
  │    ・10,000クライアント → 60,000接続              │
  └────────────────────────────────────────────────┘
```

### 1.3 ヘッダーの冗長性

HTTP/1.1ではヘッダーが圧縮されずにテキストとして送信される。
同一サイト内のページ遷移では、Cookie、User-Agent、Accept-Language など
ほぼ同一のヘッダーが何度も繰り返し送られる。

```
典型的なHTTP/1.1リクエストヘッダー（約700バイト）:

GET /api/users HTTP/1.1
Host: api.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Accept: application/json, text/plain, */*
Accept-Language: ja,en-US;q=0.9,en;q=0.8
Accept-Encoding: gzip, deflate, br
Cookie: session=abc123def456; csrftoken=xyz789; preferences=theme%3Ddark%26lang%3Dja
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx...
Referer: https://example.com/dashboard
Connection: keep-alive

→ 100リクエストで約70KBのヘッダーだけが通信される
→ モバイル回線では無視できないオーバーヘッド
```

### 1.4 HTTP/1.1時代の最適化テクニック（HTTP/2で不要になったもの）

HTTP/1.1の制約を回避するために開発者が用いていた手法と、HTTP/2でそれらが
不要になった理由を整理する。

```
┌─────────────────┬─────────────────────┬──────────────────────┐
│ 手法             │ HTTP/1.1 での目的    │ HTTP/2 での扱い       │
├─────────────────┼─────────────────────┼──────────────────────┤
│ ドメインシャーディ│ 6接続制限の回避      │ 不要（1接続で多重化） │
│ ング             │ (cdn1, cdn2...)     │ 逆にTLSコスト増加     │
├─────────────────┼─────────────────────┼──────────────────────┤
│ スプライト画像    │ リクエスト数削減     │ 不要（多重化で並列取得）│
│                  │ 複数画像→1枚に結合  │ キャッシュ効率が悪化   │
├─────────────────┼─────────────────────┼──────────────────────┤
│ CSSインライン化   │ 外部リクエスト削減   │ 不要（並列取得可能）   │
│                  │ HTML内に<style>埋込  │ キャッシュ不可になる   │
├─────────────────┼─────────────────────┼──────────────────────┤
│ ファイル結合       │ JS/CSSを1ファイルに │ 不要（多重化で並列取得）│
│ (Bundling)       │ リクエスト数を削減   │ 差分キャッシュ不可     │
├─────────────────┼─────────────────────┼──────────────────────┤
│ データURIスキーム  │ 小画像をBase64で埋込 │ 不要（小ファイルも     │
│                  │ リクエスト数削減     │ 効率よく多重化取得）   │
└─────────────────┴─────────────────────┴──────────────────────┘

注意: HTTP/2環境でもバンドルは依然として有用な場合がある。
      ファイル数が非常に多い場合（数百のモジュール）は、
      適度なバンドルで初回ロードを最適化できる。
```

---

## 2. HTTP/2の主要機能

### 2.1 バイナリフレーミング層

HTTP/2はHTTP/1.1のテキストベースプロトコルから、バイナリベースの
フレーミング層を導入した。すべてのHTTP通信はフレーム単位に分割され、
ストリームという論理チャネル上を流れる。

```
HTTP/2 フレーム構造:

  ┌─────────────────────────────────────────────┐
  │  Length (24ビット)                            │  フレームペイロードのバイト数
  ├──────────────────────┬──────────────────────┤
  │  Type (8ビット)       │  Flags (8ビット)      │  フレーム種別とフラグ
  ├──────────────────────┴──────────────────────┤
  │  R │  Stream Identifier (31ビット)           │  所属ストリームID
  ├─────────────────────────────────────────────┤
  │                                               │
  │  Frame Payload (可変長)                       │  実際のデータ
  │                                               │
  └─────────────────────────────────────────────┘

  フレームの最小ヘッダー: 9バイト
  最大ペイロードサイズ: 16,384バイト（デフォルト）
                       最大 16,777,215バイト（約16MB）

主要なフレームタイプ:

  ┌────────────┬──────┬─────────────────────────────────┐
  │ タイプ      │ 値   │ 説明                             │
  ├────────────┼──────┼─────────────────────────────────┤
  │ DATA       │ 0x00 │ HTTPボディデータ                  │
  │ HEADERS    │ 0x01 │ HTTPヘッダー（HPACK圧縮済み）     │
  │ PRIORITY   │ 0x02 │ ストリーム優先度（非推奨）         │
  │ RST_STREAM │ 0x03 │ ストリームの中止                  │
  │ SETTINGS   │ 0x04 │ 接続パラメータの交渉              │
  │ PUSH_PROMISE│0x05 │ サーバープッシュの予告             │
  │ PING       │ 0x06 │ 接続のヘルスチェック              │
  │ GOAWAY     │ 0x07 │ 接続の終了通知                    │
  │ WINDOW_UPDATE│0x08│ フロー制御ウィンドウの更新         │
  │ CONTINUATION│0x09 │ ヘッダーブロックの継続             │
  └────────────┴──────┴─────────────────────────────────┘
```

### 2.2 ストリームとマルチプレキシング（多重化）

HTTP/2の最も重要な機能がマルチプレキシング（多重化）である。
1つのTCP接続上に複数の論理ストリームを確立し、各ストリームが独立して
リクエスト・レスポンスを運ぶ。

```
HTTP/1.1 vs HTTP/2 のリソース読み込みフロー:

  === HTTP/1.1（6接続、12リソース） ===

  時間 ──────────────────────────────────────→

  接続1: [──HTML──]                [──img3──]
  接続2:     [──CSS──]             [──img4──]
  接続3:     [──JS1──]             [──img5──]
  接続4:         [──JS2──]         [──img6──]
  接続5:             [──img1──]
  接続6:             [──img2──]

  → 6接続で最大6並列、接続確立コスト大
  → 1つの接続が遅いと他に影響なし（良い点）
  → 接続管理のオーバーヘッドが大きい

  === HTTP/2（1接続、12リソース） ===

  時間 ──────────────────────────────────→

  Stream 1:  [──HTML──]
  Stream 3:  [───CSS───]
  Stream 5:  [──JS1──]
  Stream 7:    [──JS2──]
  Stream 9:      [──img1──]
  Stream 11:     [──img2──]
  Stream 13:       [──img3──]
  Stream 15:       [──img4──]
  Stream 17:         [──img5──]
  Stream 19:         [──img6──]
  Stream 21:           [font1]
  Stream 23:           [font2]

  → 1接続で全リソースを並列取得
  → 接続確立は1回のみ
  → ストリームIDで識別（クライアント発は奇数）
  → フレーム単位でインターリーブされる
```

マルチプレキシングの内部動作をさらに詳しく見る。

```
1つのTCP接続上でフレームがインターリーブされる様子:

  TCP接続 ─────────────────────────────────────────→

  │H1│D1│H3│D3│D1│H5│D5│D3│D1│D5│D3│D5│

  H = HEADERS フレーム  D = DATA フレーム
  数字 = ストリームID

  展開すると:
    Stream 1 (HTML):  [H1][D1]   [D1]   [D1]
    Stream 3 (CSS):      [H3][D3]   [D3]   [D3]
    Stream 5 (JS):            [H5][D5]   [D5][D5]

  → 各ストリームのフレームが時分割で混在
  → 受信側でストリームIDごとに再構成
  → 大きなレスポンスが小さなレスポンスをブロックしない
```

### 2.3 HPACK（ヘッダー圧縮）

HTTP/2はHPACK (RFC 7541) という専用のヘッダー圧縮方式を採用している。
HPACK は静的テーブル、動的テーブル、ハフマン符号の3つの要素で構成される。

```
HPACK 圧縮の仕組み:

  ┌────────────────────────────────────────────────┐
  │                 HPACK エンコーダ                 │
  │                                                  │
  │  1. 静的テーブル（61エントリ、RFC定義）            │
  │     ┌───────┬────────────────┬──────────────┐   │
  │     │ Index │ Header Name    │ Header Value │   │
  │     ├───────┼────────────────┼──────────────┤   │
  │     │ 1     │ :authority     │              │   │
  │     │ 2     │ :method        │ GET          │   │
  │     │ 3     │ :method        │ POST         │   │
  │     │ 4     │ :path          │ /            │   │
  │     │ 5     │ :path          │ /index.html  │   │
  │     │ 6     │ :scheme        │ http         │   │
  │     │ 7     │ :scheme        │ https        │   │
  │     │ ...   │ ...            │ ...          │   │
  │     │ 61    │ www-authenticate│             │   │
  │     └───────┴────────────────┴──────────────┘   │
  │                                                  │
  │  2. 動的テーブル（FIFO、接続ごとに管理）           │
  │     ┌───────┬────────────────┬──────────────┐   │
  │     │ 62    │ host           │ api.example  │   │
  │     │ 63    │ authorization  │ Bearer xxx   │   │
  │     │ 64    │ custom-header  │ value123     │   │
  │     └───────┴────────────────┴──────────────┘   │
  │                                                  │
  │  3. ハフマン符号化                                │
  │     → 頻出文字に短いビット列を割り当て            │
  │     → 英数字は5-7ビット（ASCIIの8ビットより短い） │
  │     → ヘッダー値をさらに15-20%圧縮               │
  └────────────────────────────────────────────────┘

  効果の例:
    1回目のリクエスト:
      :method: GET             → 静的テーブル Index 2（1バイト）
      :path: /api/users        → 名前は静的 Index 4、値はリテラル
      host: api.example.com    → リテラル（動的テーブルに追加）
      authorization: Bearer... → リテラル（動的テーブルに追加）
      合計: 約120バイト（元800バイトから85%削減）

    2回目のリクエスト:
      :method: GET             → 静的テーブル Index 2（1バイト）
      :path: /api/users/123    → 名前は静的、値のみリテラル
      host: api.example.com    → 動的テーブル Index 62（1バイト）
      authorization: Bearer... → 動的テーブル Index 63（1バイト）
      合計: 約30バイト（元800バイトから96%削減）
```

### 2.4 サーバープッシュ

サーバープッシュは、クライアントがリクエストする前にサーバーがリソースを
先行送信する機能である。HTMLを解析してCSS/JSのリクエストが来ると予測し、
先にPUSH_PROMISEフレームで予告してから送信する。

```
サーバープッシュのフロー:

  Client                              Server
    │                                   │
    │──── HEADERS (GET /index.html) ───→│
    │                                   │
    │←── PUSH_PROMISE (Stream 2)  ─────│  「/style.css を送るよ」
    │←── PUSH_PROMISE (Stream 4)  ─────│  「/app.js を送るよ」
    │                                   │
    │←── HEADERS (Stream 1, 200 OK) ───│  index.html のヘッダー
    │←── DATA (Stream 1, HTML body) ───│  index.html のボディ
    │                                   │
    │←── HEADERS (Stream 2, 200 OK) ───│  style.css のヘッダー
    │←── DATA (Stream 2, CSS body) ────│  style.css のボディ
    │                                   │
    │←── HEADERS (Stream 4, 200 OK) ───│  app.js のヘッダー
    │←── DATA (Stream 4, JS body)  ────│  app.js のボディ
    │                                   │

  クライアントは RST_STREAM で不要なプッシュを拒否可能

  重要な経緯:
    Chrome 106（2022年10月）でサーバープッシュを廃止
    理由:
      - プッシュされたリソースがキャッシュ済みの場合は帯域の浪費
      - 正確なプッシュ対象の判断が難しい
      - 103 Early Hints の方が柔軟で効果的
      - CDN経由ではプッシュがオリジンの意図通りに動作しない

  代替技術:
    103 Early Hints:
      サーバーが最終レスポンス（200）の前にヒントを送る
      → ブラウザは先に preload / preconnect を開始
      → サーバープッシュと違い、ブラウザが取得判断する

    <link rel="preload">:
      HTMLの<head>でリソースの先読みを宣言
      → ブラウザの優先度制御に委ねる
```

### 2.5 フロー制御

HTTP/2はストリームレベルと接続レベルの2段階でフロー制御を行う。
これはTCPのフロー制御とは独立した、アプリケーション層の制御である。

```
HTTP/2 フロー制御の仕組み:

  初期ウィンドウサイズ: 65,535バイト（SETTINGS で変更可能）

  ┌──────────────────────────────────────────┐
  │ 接続レベルのフロー制御                      │
  │   全ストリームの合計送信量を制御             │
  │   WINDOW_UPDATE(Stream 0) で更新           │
  │                                            │
  │  ┌────────────────────────────────────┐   │
  │  │ Stream 1 のフロー制御               │   │
  │  │   このストリームの送信量を制御        │   │
  │  │   WINDOW_UPDATE(Stream 1) で更新    │   │
  │  └────────────────────────────────────┘   │
  │  ┌────────────────────────────────────┐   │
  │  │ Stream 3 のフロー制御               │   │
  │  │   このストリームの送信量を制御        │   │
  │  │   WINDOW_UPDATE(Stream 3) で更新    │   │
  │  └────────────────────────────────────┘   │
  └──────────────────────────────────────────┘

  送信側:
    DATA フレーム送信 → ウィンドウサイズ減少
    ウィンドウ = 0 になったら送信停止

  受信側:
    DATA 受信・処理 → WINDOW_UPDATE 送信
    → 送信側のウィンドウ回復 → 送信再開
```

### 2.6 ストリーム優先度

HTTP/2では各ストリームに重みと依存関係を設定できる。
これにより重要なリソース（CSS、フォント）を画像より先に配信できる。

```
HTTP/2 ストリーム優先度の依存関係ツリー:

          Root (Stream 0)
          ├── Stream 1 (HTML)    重み: 256
          │   ├── Stream 3 (CSS) 重み: 256   ← CSSを最優先
          │   │   └── Stream 5 (JS) 重み: 220
          │   └── Stream 7 (Font) 重み: 183
          └── Stream 9 (Image)   重み: 110   ← 画像は後回し

  帯域配分の例（親ストリーム完了後）:
    CSS:  256/(256+183) = 58% の帯域
    Font: 183/(256+183) = 42% の帯域
    → CSS完了後に JS が 100% の帯域を獲得

  注意:
    RFC 9218 (Extensible Priorities) で HTTP/2・HTTP/3 共通の
    新しい優先度方式が標準化された。
    → Priority ヘッダーフィールドを使用
    → urgency (u=0..7) と incremental (i=true/false) の2パラメータ
    → 依存関係ツリーよりシンプルで実装しやすい
```

---

## 3. HTTP/2の検証と計測

### 3.1 curlでHTTP/2接続を確認する

```bash
# コード例1: curl で HTTP/2 を使用して接続を確認する
# --http2 フラグで HTTP/2 を要求
# -v (verbose) で接続の詳細を表示

$ curl -v --http2 https://www.google.com/ -o /dev/null 2>&1 | head -30

* Connected to www.google.com (142.250.xx.xx) port 443
* ALPN: curl offers h2,http/1.1
* TLSv1.3 (OUT), TLS handshake, Client hello
* TLSv1.3 (IN), TLS handshake, Server hello
* SSL connection using TLSv1.3 / TLS_AES_256_GCM_SHA384
* ALPN: server accepted h2          ← HTTP/2 が選択された
* using HTTP/2
* [HTTP/2] [1] OPENED stream for https://www.google.com/
* [HTTP/2] [1] [:method: GET]
* [HTTP/2] [1] [:scheme: https]
* [HTTP/2] [1] [:authority: www.google.com]
* [HTTP/2] [1] [:path: /]
> GET / HTTP/2                       ← HTTP/2 でリクエスト
> Host: www.google.com
> User-Agent: curl/8.x.x
> Accept: */*
>
< HTTP/2 200                         ← HTTP/2 でレスポンス
< content-type: text/html; charset=UTF-8
< date: Thu, 01 Jan 2026 00:00:00 GMT
```

```bash
# コード例2: HTTP/2 の詳細フレーム情報を表示
# nghttp コマンドを使用（nghttp2 パッケージ）

$ nghttp -nv https://www.example.com/

[  0.023] Connected
[  0.050] recv SETTINGS frame
          (niv=3)
          [SETTINGS_MAX_CONCURRENT_STREAMS(0x03):100]
          [SETTINGS_INITIAL_WINDOW_SIZE(0x04):65535]
          [SETTINGS_MAX_FRAME_SIZE(0x05):16384]
[  0.050] send SETTINGS frame
          (niv=2)
          [SETTINGS_MAX_CONCURRENT_STREAMS(0x03):100]
          [SETTINGS_INITIAL_WINDOW_SIZE(0x04):65535]
[  0.050] send HEADERS frame
          ; END_STREAM | END_HEADERS
          (padlen=0)
          :method: GET
          :path: /
          :scheme: https
          :authority: www.example.com
[  0.075] recv HEADERS frame
          ; END_HEADERS
          :status: 200
          content-type: text/html
[  0.076] recv DATA frame
          ; END_STREAM
```

### 3.2 h2load によるベンチマーク

h2load は nghttp2 に含まれるHTTP/2対応のベンチマークツールである。
HTTP/1.1、HTTP/2、HTTP/3の性能を比較計測できる。

```bash
# コード例3: h2load によるHTTP/2ベンチマーク

# 基本的な使い方
# -n: 総リクエスト数  -c: 同時接続数  -m: 1接続あたりの最大ストリーム数
$ h2load -n 10000 -c 100 -m 10 https://www.example.com/

starting benchmark...
spawning thread #0: 100 total client(s). 10000 total requests
TLS Protocol: TLSv1.3
Cipher: TLS_AES_256_GCM_SHA384
Server Temp Key: X25519 253 bits
Application protocol: h2                 ← HTTP/2 で接続

finished in 2.35s, 4255.32 req/s, 6.23MB/s
requests: 10000 total, 10000 started, 10000 done, 10000 succeeded, 0 failed
status codes: 10000 2xx, 0 3xx, 0 4xx, 0 5xx
traffic: 14.64MB (15347712) total, 390.63KB (400000) headers, 14.17MB (14853600) data
                     min         max         mean         sd        +/- sd
time for request:     2.10ms    120.45ms     23.45ms     15.30ms    78.50%
time for connect:    45.20ms    185.30ms     98.45ms     35.20ms    65.00%
time to 1st byte:    48.30ms    210.50ms    115.20ms     40.10ms    62.00%
req/s           :      42.55       52.30       45.12        3.20    70.00%

# HTTP/1.1 との比較計測
$ h2load -n 10000 -c 100 --h1 https://www.example.com/
# → HTTP/1.1 では 1接続1リクエストのため req/s が大幅に低下

# HTTP/3 での計測（h2load が対応している場合）
$ h2load -n 10000 -c 100 --npn-list h3 https://www.example.com/
```

### 3.3 ブラウザ開発者ツールでの確認

```
Chrome DevTools での HTTP/2 確認方法:

  1. Network タブを開く
  2. 任意のリクエストを選択
  3. Headers タブで以下を確認:
     - "Protocol: h2" と表示されていれば HTTP/2
     - "Protocol: h3" と表示されていれば HTTP/3

  4. Protocol 列の表示（デフォルトでは非表示）:
     - Network タブのヘッダー行を右クリック
     - "Protocol" にチェック
     - 全リクエストのプロトコルを一覧確認

  5. Connection ID 列:
     - 同じ Connection ID = 同じTCP接続を共有
     - HTTP/2 では多くのリクエストが同じ ID
     - HTTP/1.1 では異なる ID が散在

  Waterfall の見方:
    HTTP/1.1: 階段状（順次ロード）
    HTTP/2:   並列（多数のリクエストが同時開始）
```

---

## 4. HTTP/3（QUIC）の設計と仕組み

### 4.1 QUICプロトコルの全体像

HTTP/3はQUICプロトコル上に構築されたHTTPの第3世代である。
QUICはGoogleが2012年に開発を開始し、2021年にRFC 9000として標準化された。

```
プロトコルスタックの比較:

  HTTP/1.1          HTTP/2           HTTP/3
  ┌──────────┐   ┌──────────┐    ┌──────────┐
  │  HTTP    │   │  HTTP/2  │    │  HTTP/3  │
  ├──────────┤   ├──────────┤    ├──────────┤
  │          │   │  TLS 1.2+│    │          │
  │  TCP     │   ├──────────┤    │  QUIC    │
  │          │   │  TCP     │    │(TLS 1.3  │
  ├──────────┤   ├──────────┤    │ 内蔵)    │
  │  IP      │   │  IP      │    ├──────────┤
  └──────────┘   └──────────┘    │  UDP     │
                                  ├──────────┤
                                  │  IP      │
                                  └──────────┘

  QUIC の特徴:
    ・UDP 上に構築（カーネル変更不要）
    ・TLS 1.3 を統合（暗号化が必須）
    ・ストリームがトランスポート層で実装
    ・Connection ID による接続識別
    ・ユーザー空間で実装（OSカーネル非依存）
```

### 4.2 QUICの接続確立（0-RTT / 1-RTT）

QUICの最大の利点の1つが接続確立の高速化である。TCP+TLSでは2-3 RTTかかる
ハンドシェイクを、QUICでは1 RTT、再接続時は0 RTTで完了できる。

```
接続確立の比較:

  === TCP + TLS 1.2 (HTTP/2) === 合計 3 RTT ===

  Client                          Server
    │──── SYN ─────────────────→│        ┐
    │←── SYN-ACK ──────────────│        │ TCP: 1.5 RTT
    │──── ACK ─────────────────→│        ┘
    │──── ClientHello ─────────→│        ┐
    │←── ServerHello + Cert ───│        │ TLS 1.2: 2 RTT
    │──── Key Exchange ────────→│        │
    │←── Finished ─────────────│        ┘
    │──── HTTP Request ────────→│   ← ここでやっとデータ送信
    │←── HTTP Response ────────│

  === TCP + TLS 1.3 (HTTP/2) === 合計 2 RTT ===

  Client                          Server
    │──── SYN ─────────────────→│        ┐
    │←── SYN-ACK ──────────────│        │ TCP: 1.5 RTT
    │──── ACK ─────────────────→│        ┘
    │──── ClientHello ─────────→│        ┐
    │←── ServerHello+Finished ─│        │ TLS 1.3: 1 RTT
    │──── Finished ────────────→│        ┘
    │──── HTTP Request ────────→│
    │←── HTTP Response ────────│

  === QUIC (HTTP/3) === 初回接続 1 RTT ===

  Client                          Server
    │──── Initial(ClientHello) ─→│       ┐
    │←── Initial(ServerHello)  ──│       │ QUIC+TLS: 1 RTT
    │←── Handshake(Finished)   ──│       │
    │──── Handshake(Finished)  ──→│       ┘
    │──── HTTP Request ──────────→│  ← TCPハンドシェイク不要
    │←── HTTP Response ──────────│

  === QUIC 0-RTT 再接続 === 合計 0 RTT ===

  Client                          Server
    │──── Initial + 0-RTT Data ──→│  ← 前回のセッション情報を使い
    │←── Handshake + Response  ──│    即座にデータを送信
    │                              │
    ※ リプレイ攻撃のリスクがあるため
      冪等なリクエスト（GET等）のみ 0-RTT が推奨
```

### 4.3 ストリーム独立性（TCP HoL Blockingの解消）

HTTP/2はアプリケーション層でのHoL Blockingを解消したが、TCP層での
HoL Blockingが依然として残っていた。QUICはこれを根本的に解決する。

```
TCP層 HoL Blocking（HTTP/2の問題）:

  TCP接続上の1つのパケットがロストすると、
  全ストリームが再送待ちになる

  TCP接続: ──[S1][S3][S5][S3][✗ S1 ロスト][S5]──
                                  ↑
                         パケットロスト
                         ↓
  Stream 1: ──[data]─────────────[再送待ち...]──→
  Stream 3: ──[data][data]───────[ブロック]──→  ← 無関係なのに停止
  Stream 5: ──[data]────[data]───[ブロック]──→  ← 無関係なのに停止

  TCPは順序保証があるため、ロストパケットの前のデータまで
  アプリケーションに渡せない。

QUIC のストリーム独立性:

  QUIC接続: ──[S1][S3][S5][S3][✗ S1 ロスト][S5]──
                                    ↑
                           パケットロスト
                           ↓
  Stream 1: ──[data]─────────────[再送待ち...]──→ ← 影響あり
  Stream 3: ──[data][data]───────[data]──→        ← 影響なし!
  Stream 5: ──[data]────[data]───[data]──→        ← 影響なし!

  QUICは各ストリームが独立した順序保証を持つため、
  1つのストリームのロスが他に波及しない。
```

### 4.4 接続移行 (Connection Migration)

TCPは接続を「送信元IP:ポート + 宛先IP:ポート」の4タプルで識別する。
そのためIPアドレスが変わると接続が切れる。QUICはConnection IDで
接続を識別するため、ネットワーク切り替え時も接続を維持できる。

```
接続移行のシナリオ:

  === TCP（HTTP/2）: Wi-Fi → モバイル切り替え ===

  Wi-Fi接続中:
    Client 192.168.1.10:54321 ←→ Server 203.0.113.1:443
    [TCP接続確立済み、データ通信中]

  モバイルに切り替え:
    Client IPが 100.64.0.50 に変わる
    → TCP 4タプルが変わる → 接続断
    → 新たにTCPハンドシェイク (1 RTT)
    → 新たにTLSハンドシェイク (1-2 RTT)
    → 合計 2-3 RTT のダウンタイム + セッション再構築

  === QUIC（HTTP/3）: Wi-Fi → モバイル切り替え ===

  Wi-Fi接続中:
    Client 192.168.1.10 ←→ Server 203.0.113.1
    Connection ID: 0xABCD1234
    [QUIC接続確立済み、データ通信中]

  モバイルに切り替え:
    Client IPが 100.64.0.50 に変わる
    → Connection ID は 0xABCD1234 のまま
    → Path Validation（到達性確認）のみ
    → 0.5 RTT 程度で通信再開
    → ストリームの状態もそのまま維持

  ユースケース:
    ・電車内での Wi-Fi ↔ モバイル切り替え
    ・カフェから屋外への移動
    ・VPN接続時のネットワーク変更
    ・IoTデバイスのネットワーク遷移
```

---

## 5. QPACK（HTTP/3のヘッダー圧縮）

HTTP/3ではHPACKの代わりにQPACKを使用する。
HPACKはストリーム間の順序依存があったが、QPACKはQUICのストリーム独立性に
合わせて設計されている。

```
HPACK vs QPACK の違い:

  HPACK（HTTP/2）:
    ・動的テーブルの更新はストリーム順に処理
    ・ストリームAの参照がストリームBの更新に依存する可能性
    → QUICのストリーム独立性と矛盾

  QPACK（HTTP/3）:
    ・エンコーダストリームとデコーダストリームを分離
    ・動的テーブルの更新は専用の単方向ストリームで管理
    ・参照可能なエントリの範囲を明示的に管理

  ┌──────────────────────────────────────────┐
  │ QPACK のストリーム構成                      │
  │                                            │
  │  エンコーダストリーム（単方向）              │
  │    Client → Server                         │
  │    動的テーブルへのエントリ追加を通知        │
  │                                            │
  │  デコーダストリーム（単方向）                │
  │    Server → Client                         │
  │    エントリの処理完了を通知（ACK）           │
  │                                            │
  │  リクエストストリーム（双方向、複数）        │
  │    ヘッダーブロックを送信                    │
  │    必要なエントリが利用可能になるまで待機    │
  └──────────────────────────────────────────┘

  圧縮効率はHPACKとほぼ同等だが、
  ストリーム間のブロッキングを最小化している。
```

---

## 6. サーバー設定の実践

### 6.1 Nginx でのHTTP/2・HTTP/3設定

```nginx
# コード例4: Nginx で HTTP/2 と HTTP/3 を同時に有効化する完全設定

# /etc/nginx/conf.d/http2-http3.conf

# HTTP → HTTPS リダイレクト
server {
    listen 80;
    listen [::]:80;
    server_name example.com www.example.com;
    return 301 https://$host$request_uri;
}

# HTTPS + HTTP/2 + HTTP/3
server {
    # HTTP/2 over TLS
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;

    # HTTP/3 over QUIC (UDP)
    listen 443 quic reuseport;
    listen [::]:443 quic reuseport;
    http3 on;

    server_name example.com www.example.com;

    # TLS 証明書
    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # TLS設定（HTTP/3にはTLS 1.3が必須）
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # QUIC用の設定
    ssl_early_data on;          # 0-RTT を有効化
    quic_retry on;              # アドレス検証を有効化（DoS対策）
    quic_gso on;                # Generic Segmentation Offload

    # Alt-Svc ヘッダーでHTTP/3の利用可能性を通知
    # ブラウザは次回以降の接続でHTTP/3を使用する
    add_header Alt-Svc 'h3=":443"; ma=86400' always;

    # HTTP/2 サーバープッシュ（非推奨だが参考用）
    # http2_push /style.css;
    # http2_push /app.js;

    # 代わりに 103 Early Hints を使用
    # add_header Link "</style.css>; rel=preload; as=style" always;

    # HTTP/2の同時ストリーム数制限
    http2_max_concurrent_streams 128;

    # コンテンツ配信
    root /var/www/example.com;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # API プロキシ
    location /api/ {
        proxy_pass http://backend:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 6.2 Node.js でのHTTP/2サーバー実装

```javascript
// コード例5: Node.js HTTP/2 サーバーの実装（サーバープッシュ付き）

const http2 = require('node:http2');
const fs = require('node:fs');
const path = require('node:path');

// HTTP/2 セキュアサーバーを作成
const server = http2.createSecureServer({
    key: fs.readFileSync(path.join(__dirname, 'certs/server.key')),
    cert: fs.readFileSync(path.join(__dirname, 'certs/server.crt')),
    // HTTP/2 固有の設定
    settings: {
        maxConcurrentStreams: 100,     // 最大同時ストリーム数
        initialWindowSize: 65535,      // 初期ウィンドウサイズ
        maxHeaderListSize: 65535,      // 最大ヘッダーリストサイズ
        enableConnectProtocol: false,  // WebSocket over HTTP/2
    },
    // TLS 1.3 を優先
    minVersion: 'TLSv1.2',
    maxVersion: 'TLSv1.3',
});

// ストリームハンドラ（HTTP/2特有のイベント）
server.on('stream', (stream, headers) => {
    const reqPath = headers[':path'];
    const method = headers[':method'];

    console.log(`${method} ${reqPath} (Stream ID: ${stream.id})`);

    if (reqPath === '/') {
        // メインHTML返却
        stream.respond({
            ':status': 200,
            'content-type': 'text/html; charset=utf-8',
            'cache-control': 'public, max-age=3600',
        });

        // 103 Early Hints の代替としてリンクヘッダーを使用
        const html = `<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>HTTP/2 Demo</title>
    <link rel="stylesheet" href="/style.css">
    <script src="/app.js" defer></script>
</head>
<body>
    <h1>HTTP/2 サーバー動作中</h1>
    <p>Stream ID: ${stream.id}</p>
    <p>Protocol: HTTP/2</p>
</body>
</html>`;
        stream.end(html);

    } else if (reqPath === '/style.css') {
        stream.respond({
            ':status': 200,
            'content-type': 'text/css',
            'cache-control': 'public, max-age=86400',
        });
        stream.end(`
            body {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                max-width: 800px;
                margin: 2rem auto;
                padding: 0 1rem;
                background: #f5f5f5;
            }
            h1 { color: #2563eb; }
        `);

    } else if (reqPath === '/app.js') {
        stream.respond({
            ':status': 200,
            'content-type': 'application/javascript',
            'cache-control': 'public, max-age=86400',
        });
        stream.end(`
            console.log('HTTP/2 connection established');
            document.addEventListener('DOMContentLoaded', () => {
                const info = document.createElement('p');
                info.textContent = 'JavaScript loaded via HTTP/2 stream';
                document.body.appendChild(info);
            });
        `);

    } else {
        stream.respond({ ':status': 404 });
        stream.end('Not Found');
    }
});

// エラーハンドリング
server.on('error', (err) => {
    console.error('Server error:', err);
});

server.on('sessionError', (err) => {
    console.error('Session error:', err);
});

// セッションイベント（デバッグ用）
server.on('session', (session) => {
    console.log('New HTTP/2 session established');

    session.on('close', () => {
        console.log('HTTP/2 session closed');
    });

    // フロー制御の監視
    session.on('localSettings', (settings) => {
        console.log('Local settings:', settings);
    });

    session.on('remoteSettings', (settings) => {
        console.log('Remote settings:', settings);
    });
});

const PORT = 8443;
server.listen(PORT, () => {
    console.log(`HTTP/2 server listening on https://localhost:${PORT}`);
});
```

### 6.3 HTTP/3（QUIC）サーバー設定

```
HTTP/3 対応のサーバーソフトウェア一覧:

  ┌─────────────────┬──────────────┬────────────────────────┐
  │ サーバー          │ QUIC ライブラリ │ 対応状況             │
  ├─────────────────┼──────────────┼────────────────────────┤
  │ Nginx 1.25+     │ quictls     │ 正式対応               │
  │ Caddy 2.x       │ quic-go     │ デフォルト有効          │
  │ LiteSpeed       │ lsquic      │ 正式対応（最速級）      │
  │ Apache (実験的)  │ mod_http3   │ 実験的サポート          │
  │ H2O             │ quicly      │ 正式対応               │
  │ Cloudflare      │ quiche      │ CDN全体で有効           │
  │ Node.js (実験的) │ ngtcp2      │ --experimental-quic    │
  └─────────────────┴──────────────┴────────────────────────┘
```

```
# Caddy でのHTTP/3自動設定
# Caddyfile（Caddyは自動でHTTP/3を有効化）

example.com {
    root * /var/www/example.com
    file_server

    # TLS は自動取得・自動更新（Let's Encrypt）
    # HTTP/3 はデフォルトで有効

    # HTTPSへのリダイレクトも自動
    encode gzip zstd

    header {
        # セキュリティヘッダー
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
    }

    # API のリバースプロキシ
    handle /api/* {
        reverse_proxy localhost:3000
    }
}
```

---

## 7. プロトコルバージョン間の詳細比較

### 7.1 機能比較表

```
┌─────────────────────┬────────────┬─────────────┬─────────────┐
│ 機能                 │ HTTP/1.1   │ HTTP/2      │ HTTP/3      │
├─────────────────────┼────────────┼─────────────┼─────────────┤
│ リリース年            │ 1997(RFC)  │ 2015(RFC)   │ 2022(RFC)   │
│ RFCドキュメント       │ RFC 9110   │ RFC 9113    │ RFC 9114    │
│ トランスポート層      │ TCP        │ TCP         │ QUIC (UDP)  │
│ メッセージ形式        │ テキスト    │ バイナリ     │ バイナリ     │
│ 多重化               │ なし       │ あり         │ あり         │
│ ヘッダー圧縮          │ なし       │ HPACK       │ QPACK       │
│ サーバープッシュ      │ なし       │ あり(廃止傾向)│ あり(非推奨) │
│ 暗号化               │ 任意(HTTPS)│ 事実上必須   │ 必須         │
│ 接続確立RTT          │ 1-3 RTT   │ 2-3 RTT     │ 1 RTT(0-RTT)│
│ HoL Blocking        │ HTTP層+TCP層│ TCP層のみ   │ なし         │
│ フロー制御           │ TCP依存    │ 2段階        │ 2段階        │
│ 接続移行             │ 不可       │ 不可         │ 可能         │
│ 優先度制御           │ なし       │ 依存関係ツリー│ Priority    │
│                     │            │              │ ヘッダー     │
│ TLSバージョン        │ 1.0+      │ 1.2+         │ 1.3のみ     │
└─────────────────────┴────────────┴─────────────┴─────────────┘
```

### 7.2 パフォーマンス特性比較表

```
┌──────────────────────┬────────────┬────────────┬────────────┐
│ シナリオ              │ HTTP/1.1   │ HTTP/2     │ HTTP/3     │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 低レイテンシ環境       │ 良好       │ 最良       │ 良好       │
│ (LAN, 同一DC)        │            │            │            │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 高レイテンシ環境       │ 悪い       │ 改善       │ 最良       │
│ (海外サーバー等)      │ HoL深刻   │ TCP HoL残  │ HoLなし    │
├──────────────────────┼────────────┼────────────┼────────────┤
│ パケットロスあり       │ 悪い       │ 悪い       │ 最良       │
│ (モバイル回線)        │ 全停止     │ 全停止     │ 部分停止   │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 少数の大ファイル       │ 普通       │ 改善少     │ 改善少     │
│ (動画DL等)           │            │ 多重化不要  │ 多重化不要  │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 多数の小ファイル       │ 非常に悪い  │ 最良       │ 最良       │
│ (SPA, API通信)       │ 接続制限   │ 多重化効果大│ 多重化効果大│
├──────────────────────┼────────────┼────────────┼────────────┤
│ ネットワーク切替       │ 接続断     │ 接続断     │ 継続       │
│ (Wi-Fi↔モバイル)     │ 再接続要   │ 再接続要   │ 移行可能   │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 初回接続速度           │ 1 RTT     │ 2-3 RTT   │ 1 RTT     │
│                      │ (TCP only) │ (TCP+TLS)  │ (QUIC)    │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 再接続速度            │ 1 RTT     │ 1-2 RTT   │ 0 RTT     │
│                      │ (TCP)      │(TLS resume)│(0-RTT)    │
├──────────────────────┼────────────┼────────────┼────────────┤
│ サーバーリソース消費   │ 高い       │ 低い       │ 中程度     │
│                      │ 多接続     │ 1接続      │ UDP処理    │
├──────────────────────┼────────────┼────────────┼────────────┤
│ ファイアウォール通過   │ 問題なし   │ 問題なし   │ UDPブロック│
│                      │            │            │ の場合あり  │
└──────────────────────┴────────────┴────────────┴────────────┘
```

---

## 8. アンチパターン

### 8.1 アンチパターン1: HTTP/2環境でのドメインシャーディング継続

```
誤った構成:

  HTTP/2サイトで以下のように複数ドメインを使い続ける:

  <link rel="stylesheet" href="https://static1.example.com/style.css">
  <script src="https://static2.example.com/app.js"></script>
  <img src="https://static3.example.com/hero.jpg">
  <img src="https://cdn.example.com/logo.png">

  問題点:
  ┌────────────────────────────────────────────────────┐
  │  HTTP/2 + ドメインシャーディング = 逆効果           │
  │                                                    │
  │  ・各ドメインごとにTLS接続が必要                     │
  │    static1: TCP HS (1 RTT) + TLS HS (1-2 RTT)     │
  │    static2: TCP HS (1 RTT) + TLS HS (1-2 RTT)     │
  │    static3: TCP HS (1 RTT) + TLS HS (1-2 RTT)     │
  │    cdn:     TCP HS (1 RTT) + TLS HS (1-2 RTT)     │
  │    → 合計 8-12 RTT の接続オーバーヘッド             │
  │                                                    │
  │  ・各接続で独立したTCPスロースタート                  │
  │    → 帯域利用効率の低下                             │
  │                                                    │
  │  ・HPACKの動的テーブルが接続ごとに独立               │
  │    → ヘッダー圧縮効率が悪化                         │
  │                                                    │
  │  ・HTTP/2のストリーム優先度が接続間で調整不可         │
  │    → 重要リソースの優先配信ができない                │
  └────────────────────────────────────────────────────┘

  正しいアプローチ:
    ・1つのドメイン（または最小限のドメイン）に集約
    ・HTTP/2の多重化で十分な並列性を確保
    ・CDNを使う場合もHTTP/2対応CDN 1つに統一
```

### 8.2 アンチパターン2: HTTP/3への安易な全面移行

```
誤った判断:

  「HTTP/3が最新だからHTTP/2を廃止してHTTP/3だけにしよう」

  問題点:
  ┌────────────────────────────────────────────────────┐
  │  HTTP/3 のみ = 多くのクライアントが接続不可          │
  │                                                    │
  │  1. 企業ファイアウォール                             │
  │     ・多くの企業がUDP 443をブロック                  │
  │     ・HTTP/3（QUIC）はUDP上で動作                   │
  │     ・TCP 443（HTTPS）のみ許可が一般的              │
  │                                                    │
  │  2. 古いクライアント                                │
  │     ・HTTP/3非対応ブラウザ（IE、古いSafari等）       │
  │     ・HTTP/3非対応のHTTPライブラリ                   │
  │     ・社内ツール、スクリプト、bot                    │
  │                                                    │
  │  3. 一部のISP/ネットワーク                          │
  │     ・UDP通信のQoS優先度が低い場合がある            │
  │     ・UDP帯域制限をかけているISPが存在              │
  │                                                    │
  │  4. デバッグの難しさ                                │
  │     ・QUIC通信は暗号化されており中間機器で観察困難   │
  │     ・tcpdumpでの解析がTCPより複雑                  │
  │     ・Wiresharkの対応もTCPほど成熟していない        │
  └────────────────────────────────────────────────────┘

  正しいアプローチ:
    ・HTTP/2 をベースラインとして維持
    ・HTTP/3 を追加の選択肢として提供（Alt-Svc ヘッダー）
    ・ブラウザが自動的に最適なプロトコルを選択
    ・フォールバック: HTTP/3 → HTTP/2 → HTTP/1.1

  Alt-Svc ヘッダーによる段階的アップグレード:
    1. クライアントが HTTP/2 で接続
    2. サーバーが Alt-Svc: h3=":443"; ma=86400 を返す
    3. クライアントは次回からHTTP/3を試行
    4. HTTP/3が使えなければHTTP/2にフォールバック
```

### 8.3 アンチパターン3: HTTP/2のSETTINGSパラメータ無調整

```
誤った設定（デフォルトのまま放置）:

  問題のある状況:
    ・大量のAPIリクエストを処理するマイクロサービス
    ・デフォルトの SETTINGS_MAX_CONCURRENT_STREAMS = 100
    ・初期ウィンドウサイズが小さすぎる

  ┌────────────────────────────────────────────────────┐
  │  調整すべきSETTINGSパラメータ                       │
  │                                                    │
  │  SETTINGS_MAX_CONCURRENT_STREAMS                   │
  │    デフォルト: 100                                  │
  │    API集中型: 256-1000 に増加                       │
  │    リソース制限型: 32-64 に減少                      │
  │                                                    │
  │  SETTINGS_INITIAL_WINDOW_SIZE                       │
  │    デフォルト: 65,535 (64KB)                        │
  │    高帯域環境: 1,048,576 (1MB) 以上                 │
  │    低帯域環境: 16,384 (16KB) に縮小                 │
  │                                                    │
  │  SETTINGS_MAX_FRAME_SIZE                            │
  │    デフォルト: 16,384 (16KB)                        │
  │    大ファイル配信: 65,536 (64KB) に拡大              │
  │    フレームの分割オーバーヘッドを削減                │
  │                                                    │
  │  SETTINGS_HEADER_TABLE_SIZE                         │
  │    デフォルト: 4,096 (4KB)                          │
  │    ヘッダーの種類が多い場合: 8,192-16,384 に拡大    │
  │    メモリ制約環境: 2,048 に縮小                     │
  └────────────────────────────────────────────────────┘
```

---

## 9. エッジケース分析

### 9.1 エッジケース1: QUIC接続とUDPブロック

```
シナリオ:
  企業ネットワークやホテルWi-FiでUDP 443がブロックされている環境

  ┌──────────────────────────────────────────────────┐
  │                                                    │
  │  Client ──── Firewall ──── Internet ──── Server   │
  │                  │                                  │
  │                  ├── TCP 443: 許可 (HTTPS)         │
  │                  ├── TCP 80:  許可 (HTTP)          │
  │                  └── UDP 443: ブロック              │
  │                                                    │
  └──────────────────────────────────────────────────┘

  影響:
    1. ブラウザがHTTP/3で接続試行 → タイムアウト
    2. フォールバック待ち時間が発生（300ms-数秒）
    3. HTTP/2にフォールバック → 正常動作

  ブラウザの動作:
    Chrome の Happy Eyeballs v2 実装:
      ・HTTP/3 と HTTP/2 を同時に試行
      ・先に成功した方を使用
      ・HTTP/3が一定時間応答しなければHTTP/2を選択
      ・次回以降のHTTP/3試行を一定期間スキップ

  サーバー側の対策:
    # Alt-Svc の max-age を適切に設定
    # 長すぎるとUDPブロック環境のクライアントが毎回タイムアウト
    add_header Alt-Svc 'h3=":443"; ma=3600' always;
    # ma=86400（24時間）ではなく ma=3600（1時間）程度に
    # UDPブロック環境のクライアントのフォールバック頻度を下げる

  クライアント側の対策:
    # curlでHTTP/3を無効化
    curl --http2 https://example.com/  # HTTP/2を強制

    # ブラウザでHTTP/3を無効化（デバッグ用）
    # Chrome: chrome://flags/#enable-quic → Disabled
    # Firefox: about:config → network.http.http3.enable → false
```

### 9.2 エッジケース2: HTTP/2のTCP HoL Blockingが顕在化する場面

```
シナリオ:
  パケットロス率が高いモバイルネットワーク（3G/不安定なWi-Fi）で
  HTTP/2のマルチプレキシングが逆効果になるケース

  パケットロス率 2%の環境:

  === HTTP/1.1（6接続）===
    接続1: [Stream A]─────[ロスト]─再送─[完了]
    接続2: [Stream B]────────────────[完了]  ← 影響なし
    接続3: [Stream C]────────────────[完了]  ← 影響なし
    接続4: [Stream D]─[ロスト]─再送──[完了]
    接続5: [Stream E]────────────────[完了]  ← 影響なし
    接続6: [Stream F]────────────────[完了]  ← 影響なし

    → 6接続中2接続が影響、4接続は正常
    → パケットロスの影響が分散される

  === HTTP/2（1接続）===
    TCP接続: [A][B][C][D][ロスト][E][F]
                            ↑
                     このパケットのロスで
                     A-F 全ストリームが停止

    → 1接続なのでロスが全ストリームに波及
    → HTTP/1.1の6接続より遅くなるケースがある

  === HTTP/3（QUIC）===
    QUIC接続: [A][B][C][D][ロスト][E][F]
                            ↑
                     Stream Dのロス
    Stream A: ──────────[完了]  ← 影響なし
    Stream B: ──────────[完了]  ← 影響なし
    Stream C: ──────────[完了]  ← 影響なし
    Stream D: ──[再送待ち]────  ← 影響あり
    Stream E: ──────────[完了]  ← 影響なし
    Stream F: ──────────[完了]  ← 影響なし

    → ストリームDのみ影響、他は正常

  観測データの傾向:
    パケットロス率 0%:   HTTP/2 ≈ HTTP/3 > HTTP/1.1
    パケットロス率 1%:   HTTP/3 > HTTP/2 > HTTP/1.1
    パケットロス率 2%+:  HTTP/3 >> HTTP/1.1 > HTTP/2
    → パケットロスが増えるとHTTP/2はHTTP/1.1より悪化しうる
```

### 9.3 エッジケース3: HTTP/2の大量ストリーム時のメモリ問題

```
シナリオ:
  マイクロサービス間通信でHTTP/2 gRPCを使用。
  1つの接続上に数千のストリームが同時に存在する場合。

  ┌────────────────────────────────────────────────┐
  │  gRPC over HTTP/2 の問題パターン                │
  │                                                │
  │  サービスA → サービスB（gRPC）                   │
  │  同時リクエスト: 5,000ストリーム                 │
  │                                                │
  │  メモリ消費:                                    │
  │    ストリームごとの状態管理:                     │
  │      ・HPACKの動的テーブル: 4KB/接続             │
  │      ・フロー制御ウィンドウ: 管理構造体          │
  │      ・ストリームバッファ: 可変                  │
  │      ・優先度ツリーのノード: 各ストリーム        │
  │                                                │
  │  5,000ストリーム × 管理構造 ≈ 数十MB/接続       │
  │  100接続 × 数十MB = 数GB のメモリ消費           │
  │                                                │
  │  対策:                                          │
  │    1. MAX_CONCURRENT_STREAMS を適切に制限        │
  │       (サーバー側で 100-500 に設定)              │
  │    2. 接続プーリングで接続数を制御               │
  │    3. ストリームのタイムアウトを設定              │
  │    4. GOAWAY で定期的に接続をリフレッシュ        │
  └────────────────────────────────────────────────┘
```
