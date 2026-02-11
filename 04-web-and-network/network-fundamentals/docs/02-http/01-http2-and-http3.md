# HTTP/2とHTTP/3

> HTTP/2はバイナリフレーミングと多重化でHTTP/1.1の性能問題を解決。HTTP/3はQUICベースでさらに高速化。Webパフォーマンスを左右するプロトコルの進化を理解する。

## この章で学ぶこと

- [ ] HTTP/1.1の問題点とHTTP/2が解決した課題を理解する
- [ ] HTTP/2の主要機能（多重化、HPACK等）を把握する
- [ ] HTTP/3（QUIC）の利点と採用状況を学ぶ

---

## 1. HTTP/1.1の問題点

```
HTTP/1.1の限界:

  ① Head-of-Line Blocking:
     → 1つのTCP接続で1リクエストずつ順番に処理
     → 前のレスポンスが完了しないと次が送れない

     リクエスト1 ────→ レスポンス1 ────→
                                         リクエスト2 ────→ レスポンス2 ────→
                                                                              リクエスト3 ...

  ② 多数のTCP接続:
     → ブラウザは同一ドメインに最大6接続
     → 接続確立コスト（3-way ハンドシェイク × 6）
     → サーバーリソースの消費

  ③ ヘッダーの冗長性:
     → 毎回同じヘッダーを送信（Cookie, User-Agent等）
     → 圧縮なし（1リクエスト 500-800バイトのヘッダー）
     → 100リクエスト = 50-80KB のヘッダーだけで

  HTTP/1.1での回避策（アンチパターンになった手法）:
     → ドメインシャーディング: 複数ドメインで接続数を増やす
     → スプライト画像: 複数画像を1枚にまとめる
     → CSSインライン化: 外部CSSをHTML内に埋め込む
     → ファイル結合: JS/CSSを1ファイルに結合
     → HTTP/2 ではこれらは不要（逆効果になることも）
```

---

## 2. HTTP/2の主要機能

```
① バイナリフレーミング:
  HTTP/1.1: テキストベース
    GET / HTTP/1.1\r\nHost: example.com\r\n...

  HTTP/2: バイナリフレーム
    ┌───────────┬──────────┬──────────┐
    │ Length(24) │ Type(8)  │ Flags(8) │
    ├───────────┴──────────┴──────────┤
    │ Stream Identifier (31)          │
    ├─────────────────────────────────┤
    │ Frame Payload                    │
    └─────────────────────────────────┘

  フレームタイプ:
    HEADERS:   HTTPヘッダー
    DATA:      HTTPボディ
    SETTINGS:  接続設定
    PUSH_PROMISE: サーバープッシュ
    RST_STREAM: ストリーム中止
    PING:      接続確認
    GOAWAY:    接続終了

② 多重化（Multiplexing）:
  1つのTCP接続で複数リクエストを並列処理

  HTTP/1.1（6接続使用）:
  接続1: ────リクエスト1────レスポンス1────
  接続2: ────リクエスト2────レスポンス2────
  接続3: ────リクエスト3────レスポンス3────

  HTTP/2（1接続でOK）:
  接続1: ─req1─req2─req3─res2─res1─res3─
         （ストリームIDで識別、順序不同で返却可能）

③ HPACK（ヘッダー圧縮）:
  → 静的テーブル: よく使うヘッダーを番号で参照（61エントリ）
  → 動的テーブル: 過去に送ったヘッダーを番号で参照
  → ハフマン符号: 値の圧縮

  効果: ヘッダーサイズ 85-95% 削減
  1回目: Host: api.example.com（フルサイズ）
  2回目: インデックス62（数バイト）← 動的テーブル参照

④ サーバープッシュ:
  → クライアントのリクエスト前にリソースを送信
  → HTML → CSS/JS を先読み

  クライアント: GET /index.html
  サーバー:     PUSH_PROMISE /style.css
                PUSH_PROMISE /app.js
                DATA index.html
                DATA style.css
                DATA app.js

  注意: HTTP/2 サーバープッシュは Chrome 106 で廃止
  → 代替: 103 Early Hints, preload/prefetch

⑤ ストリームの優先度:
  → リソースの重要度に応じた配信順序
  → CSS/JSは優先度高、画像は低
  → HTTP/2では依存関係ベース → HTTP/3で改善
```

---

## 3. HTTP/3（QUIC）

```
HTTP/3 = QUIC上のHTTP

  HTTP/1.1: HTTP → TCP → IP
  HTTP/2:   HTTP → TLS → TCP → IP
  HTTP/3:   HTTP → QUIC(TLS内蔵) → UDP → IP

HTTP/2の残る問題:
  → TCP レベルの Head-of-Line Blocking
  → パケットロス時に全ストリームが停止
  → TCP + TLS の接続確立に 2-3 RTT

HTTP/3の解決:
  ① ストリーム独立性:
     ストリーム1: ─パケットA─パケットB─
     ストリーム2: ─パケットC─（ロス）─パケットD─
     → ストリーム2のロスはストリーム1に影響しない

  ② 接続確立の高速化:
     HTTP/2 (TCP+TLS): 2-3 RTT
     HTTP/3 (QUIC):    1 RTT（0-RTT再接続可能）

  ③ 接続移行:
     → Connection ID ベース（IP が変わっても維持）
     → Wi-Fi ↔ モバイル切替時に再接続不要

パフォーマンス比較:
  ┌───────────────────┬────────┬────────┬────────┐
  │                   │HTTP/1.1│HTTP/2  │HTTP/3  │
  ├───────────────────┼────────┼────────┼────────┤
  │ 接続確立          │ 1 RTT  │ 2-3 RTT│ 1 RTT  │
  │ 多重化            │ ✗      │ ✓      │ ✓      │
  │ ヘッダー圧縮      │ ✗      │HPACK   │QPACK   │
  │ HoL Blocking     │ ✓      │TCP層で │ ✗      │
  │ 接続移行          │ ✗      │ ✗      │ ✓      │
  │ 暗号化            │任意    │実質必須│ 必須   │
  └───────────────────┴────────┴────────┴────────┘

採用状況（2024年）:
  Google: 全サービスで HTTP/3
  Cloudflare: デフォルトで有効
  Meta: Facebook, Instagram
  → 全Web通信の約30%がHTTP/2、約8%がHTTP/3
```

---

## 4. サーバー設定

```
Nginx でHTTP/2を有効化:
  server {
      listen 443 ssl;
      http2 on;
      ssl_certificate /path/to/cert.pem;
      ssl_certificate_key /path/to/key.pem;
  }

Nginx でHTTP/3を有効化:
  server {
      listen 443 quic reuseport;
      listen 443 ssl;
      http2 on;
      http3 on;
      add_header Alt-Svc 'h3=":443"; ma=86400';
      ssl_certificate /path/to/cert.pem;
      ssl_certificate_key /path/to/key.pem;
  }

Node.js でHTTP/2:
  const http2 = require('http2');
  const fs = require('fs');

  const server = http2.createSecureServer({
    key: fs.readFileSync('key.pem'),
    cert: fs.readFileSync('cert.pem'),
  });

  server.on('stream', (stream, headers) => {
    stream.respond({ ':status': 200, 'content-type': 'text/html' });
    stream.end('<h1>Hello HTTP/2</h1>');
  });

  server.listen(8443);
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| HTTP/1.1 | テキスト、HoL Blocking、6接続制限 |
| HTTP/2 | バイナリ、多重化、HPACK、1接続 |
| HTTP/3 | QUIC(UDP)、ストリーム独立、1RTT接続 |
| 移行 | HTTP/2は必須級、HTTP/3は推奨 |

---

## 次に読むべきガイド
→ [[02-rest-api.md]] — REST API設計

---

## 参考文献
1. RFC 9113. "HTTP/2." IETF, 2022.
2. RFC 9114. "HTTP/3." IETF, 2022.
3. RFC 9000. "QUIC." IETF, 2021.
