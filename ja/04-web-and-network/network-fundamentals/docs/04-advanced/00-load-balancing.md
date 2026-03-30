# ロードバランシング

> ロードバランサーはトラフィックを複数のサーバーに分散し、可用性とスケーラビリティを実現する中核インフラである。L4/L7 の動作原理、分散アルゴリズムの数理的背景、AWS ALB/NLB の実践的構成、HAProxy による高可用性設計までを体系的に解説する。

## この章で学ぶこと

- [ ] L4（トランスポート層）と L7（アプリケーション層）ロードバランシングの動作原理を理解する
- [ ] 主要な分散アルゴリズム 6 種の特性・選定基準・数理的背景を把握する
- [ ] Nginx / HAProxy による実践的なロードバランサー設定を習得する
- [ ] AWS ALB / NLB / GWLB の構成パターンと IaC（Terraform）による構築を学ぶ
- [ ] ヘルスチェックの設計原則と段階的ヘルスチェックの実装を理解する
- [ ] 高可用性（HA）構成における VRRP / Keepalived の仕組みを把握する
- [ ] ロードバランサーに関するアンチパターンとエッジケースを認識する

---

## 1. ロードバランシングの基本概念

### 1.1 なぜロードバランシングが必要か

単一サーバーでサービスを運用する場合、以下の限界に直面する。

```
単一サーバー構成の問題:

  クライアント群
   │ │ │ │ │
   ▼ ▼ ▼ ▼ ▼
  ┌──────────┐
  │ サーバー  │ ← 単一障害点（SPOF）
  │ CPU: 95% │ ← 処理能力の上限
  │ Mem: 90% │ ← メモリ不足
  └──────────┘

  問題:
  ① SPOF: サーバー障害 = サービス全停止
  ② スケール限界: 垂直スケールのみ（CPU/RAM 追加）
  ③ メンテナンス不可: 更新時にダウンタイム発生
  ④ 地理的制約: 単一ロケーションへの依存
```

ロードバランサーを導入することで、これらの問題を根本的に解決する。

```
ロードバランサー導入後:

  クライアント群
   │ │ │ │ │
   ▼ ▼ ▼ ▼ ▼
  ┌────────────┐
  │ ロード      │ ← 仮想 IP（VIP）で受付
  │ バランサー  │ ← トラフィックを振り分け
  └─┬──┬──┬──┬─┘
    │  │  │  │
  ┌─▼┐┌▼─┐┌▼─┐┌─▼┐
  │S1││S2││S3││S4│  ← バックエンドサーバー群
  └──┘└──┘└──┘└──┘

  解決される問題:
  ① 高可用性: S1 障害時 → S2/S3/S4 で継続
  ② 水平スケール: サーバー追加で処理能力向上
  ③ 無停止メンテナンス: ローリングデプロイ可能
  ④ 地理分散: マルチリージョン対応
```

### 1.2 ロードバランサーの基本動作

ロードバランサーはクライアントとサーバー間のプロキシとして機能する。その基本動作は以下の通りである。

```
┌────────────────────────────────────────────────────────────┐
│              ロードバランサーの処理フロー                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. リクエスト受信                                           │
│     Client ──TCP SYN──→ LB (VIP:443)                       │
│                                                            │
│  2. 振り分け先の決定                                         │
│     LB: アルゴリズムに基づきバックエンドを選択                  │
│         ヘルスチェック結果を参照し、正常なサーバーのみ候補       │
│                                                            │
│  3. リクエスト転送                                           │
│     LB ──リクエスト転送──→ Backend Server (S2)              │
│                                                            │
│  4. レスポンス返却                                           │
│     Backend Server (S2) ──レスポンス──→ LB ──→ Client      │
│                                                            │
│  ※ DSR（Direct Server Return）の場合:                       │
│     Backend Server (S2) ──レスポンス──→ Client（LB 経由せず）│
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.3 DSR（Direct Server Return）

DSR はレスポンスをロードバランサー経由せず、バックエンドサーバーから直接クライアントに返す方式である。大容量レスポンス（動画ストリーミングなど）でロードバランサーのボトルネックを回避する。

```
通常モード:
  Client → LB → Server
  Client ← LB ← Server   （LB がボトルネックになりうる）

DSR モード:
  Client → LB → Server
  Client ←────── Server   （レスポンスは直接返却）

  利点: LB の帯域使用量が大幅に削減
  欠点: L7 の機能（レスポンス書き換え等）が使えない
        サーバー側でループバック設定が必要
```

---

## 2. L4 vs L7 ロードバランシング

### 2.1 OSI 参照モデルにおける位置づけ

```
┌──────────────────────────────────────────────────────────┐
│                  OSI 参照モデルとロードバランサー            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 7: アプリケーション層  ←── L7 LB はここで動作       │
│           HTTP, HTTPS, gRPC, WebSocket                   │
│           URL パス、ホスト名、Cookie、ヘッダーで振り分け    │
│                                                          │
│  Layer 6: プレゼンテーション層                              │
│           TLS/SSL 終端をここで実施                         │
│                                                          │
│  Layer 5: セッション層                                     │
│                                                          │
│  Layer 4: トランスポート層    ←── L4 LB はここで動作       │
│           TCP, UDP                                       │
│           IP アドレス + ポート番号で振り分け                │
│                                                          │
│  Layer 3: ネットワーク層      ←── GWLB はここで動作       │
│           IP パケットの転送                                │
│                                                          │
│  Layer 2: データリンク層                                   │
│  Layer 1: 物理層                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 L4 ロードバランシングの動作原理

L4 ロードバランサーは TCP/UDP のヘッダー情報（送信元 IP、宛先 IP、送信元ポート、宛先ポート）のみに基づいて振り分けを行う。パケットのペイロード（中身）は一切検査しない。

```
L4 ロードバランシングの動作:

  クライアント: 192.168.1.100:54321
       │
       │  TCP SYN (dst: 10.0.0.1:443)
       ▼
  ┌─────────────┐
  │  L4 LB      │  判断材料:
  │  10.0.0.1   │   - 送信元 IP: 192.168.1.100
  │             │   - 宛先ポート: 443
  │  NAT 変換:  │   - プロトコル: TCP
  │  dst →      │
  │  10.0.1.x   │  ペイロードは見ない
  └──┬──────┬───┘  → HTTP/HTTPS の区別不可
     │      │      → URL パスの区別不可
     ▼      ▼
  10.0.1.1  10.0.1.2
  Server A  Server B

  特徴:
  ・カーネル空間で動作可能 → 高スループット
  ・コネクション単位の振り分け（パケット単位ではない）
  ・同一コネクションのパケットは同じサーバーへ
  ・TLS 終端は不可（パススルー）
```

### 2.3 L7 ロードバランシングの動作原理

L7 ロードバランサーは HTTP/HTTPS のプロトコルを理解し、リクエストの内容に基づいたインテリジェントなルーティングを行う。

```
L7 ロードバランシングの動作:

  クライアント
       │
       │  HTTPS リクエスト
       │  Host: api.example.com
       │  GET /v2/users/123
       │  Cookie: session=abc123
       │  Authorization: Bearer xyz
       ▼
  ┌──────────────────┐
  │    L7 LB         │  判断材料:
  │                  │   - URL パス: /v2/users/123
  │  TLS 終端 ✓     │   - ホスト名: api.example.com
  │  HTTP 解析 ✓    │   - Cookie: session=abc123
  │  ヘッダー検査 ✓ │   - HTTP メソッド: GET
  │  Cookie 検査 ✓  │   - ヘッダー: Authorization
  │  圧縮/展開 ✓    │   - Content-Type
  │                  │   - クエリパラメータ
  └─┬────┬────┬──┬──┘
    │    │    │  │
    ▼    ▼    ▼  ▼
  ┌──┐┌──┐┌──┐┌──┐
  │API││API││WS││静的│
  │v1 ││v2 ││  ││   │
  └──┘└──┘└──┘└──┘

  ルーティングルール例:
  /v1/*          → API v1 サーバー群
  /v2/*          → API v2 サーバー群
  /ws/*          → WebSocket サーバー群
  /static/*      → 静的ファイルサーバー
  Host: admin.*  → 管理画面サーバー
```

### 2.4 L4 vs L7 詳細比較表

```
┌───────────────────┬──────────────────────┬──────────────────────┐
│ 比較項目          │ L4 ロードバランサー    │ L7 ロードバランサー    │
├───────────────────┼──────────────────────┼──────────────────────┤
│ 動作レイヤー      │ トランスポート層（L4） │ アプリケーション層（L7）│
│ 判断基準          │ IP + ポート番号       │ URL, ヘッダー, Cookie │
│ プロトコル理解    │ TCP/UDP のみ          │ HTTP/HTTPS/gRPC/WS   │
│ 処理速度          │ 非常に高速            │ やや遅い（解析コスト） │
│ スループット      │ 数百万 RPS           │ 数十万 RPS            │
│ レイテンシ追加    │ < 1ms                │ 1-5ms                 │
│ TLS 終端         │ 不可（パススルー）     │ 可能（証明書管理）     │
│ コンテンツ        │ 不可                  │ パスベース             │
│   ルーティング    │                      │ ホストベース           │
│                   │                      │ ヘッダーベース         │
│ ヘルスチェック    │ TCP 接続確認のみ      │ HTTP ステータス確認    │
│                   │                      │ レスポンス本文確認     │
│ WebSocket        │ パススルー            │ プロトコル認識して処理 │
│ HTTP/2           │ パススルー            │ 多重化認識             │
│ gRPC             │ パススルー            │ gRPC ルーティング可    │
│ セッション維持   │ 送信元 IP ベース      │ Cookie ベース          │
│ WAF 連携         │ 不可                  │ 可能                   │
│ レスポンス操作   │ 不可                  │ ヘッダー追加/削除可    │
│ クライアント IP  │ 保持可能              │ X-Forwarded-For 追加  │
│ 代表的な実装     │ LVS, NLB, HAProxy    │ Nginx, ALB, Envoy     │
│ 主な用途         │ DB, メール, ゲーム    │ Web API, マイクロ      │
│                   │ IoT, DNS             │ サービス, SPA          │
│ コスト           │ 低い                  │ やや高い               │
└───────────────────┴──────────────────────┴──────────────────────┘

選定ガイドライン:
  L4 を選ぶ場合:
    - 超低レイテンシが必須（ゲームサーバー、金融取引）
    - TCP/UDP 直接通信（データベースプロキシ）
    - TLS パススルーが必要（クライアント証明書認証）
    - 大量コネクション処理（IoT デバイス接続）

  L7 を選ぶ場合:
    - パスベースルーティングが必要（マイクロサービス）
    - TLS 終端を LB で行いたい（証明書一元管理）
    - A/B テスト、カナリアデプロイ
    - WAF（Web Application Firewall）連携
    - レスポンスの圧縮・キャッシュ
```

---

## 3. 分散アルゴリズム詳解

### 3.1 ラウンドロビン（Round Robin）

最も基本的なアルゴリズムで、バックエンドサーバーに順番にリクエストを振り分ける。

```
ラウンドロビンの動作:

  リクエスト順序:  R1  R2  R3  R4  R5  R6  R7  R8  R9
  振り分け先:      S1  S2  S3  S1  S2  S3  S1  S2  S3

  時間経過:
  ──────────────────────────────────────────────→ t

  S1: ■□□■□□■□□  (3 リクエスト)
  S2: □■□□■□□■□  (3 リクエスト)
  S3: □□■□□■□□■  (3 リクエスト)

  ■ = リクエスト処理中

  利点:
    - 実装が極めてシンプル（カウンタ 1 つ）
    - オーバーヘッドがほぼゼロ
    - サーバーが均質な場合に最適

  欠点:
    - サーバーの処理能力差を考慮しない
    - リクエストの処理時間差を考慮しない
    - セッション維持ができない

  疑似コード:
    counter = 0
    servers = [S1, S2, S3]

    function next_server():
        server = servers[counter % len(servers)]
        counter += 1
        return server
```

### 3.2 加重ラウンドロビン（Weighted Round Robin）

サーバーの性能差を考慮し、処理能力に応じた重みを付与して振り分ける。

```
加重ラウンドロビンの動作:

  重み設定: S1=5, S2=3, S3=2 （合計 10）

  10 リクエストの振り分け:
  R1→S1, R2→S1, R3→S1, R4→S1, R5→S1,
  R6→S2, R7→S2, R8→S2,
  R9→S3, R10→S3

  S1 (w=5): ■■■■■□□□□□  50% のリクエスト (8 core, 32GB)
  S2 (w=3): □□□□□■■■□□  30% のリクエスト (4 core, 16GB)
  S3 (w=2): □□□□□□□□■■  20% のリクエスト (2 core, 8GB)

  スムーズ加重ラウンドロビン（Nginx 方式）:
    → 連続して同じサーバーに振り分けない
    → S1, S2, S1, S3, S1, S2, S1, S1, S2, S3
    → バースト的なリクエスト集中を防ぐ
```

### 3.3 最小接続数（Least Connections）

現在のアクティブ接続数が最も少ないサーバーにリクエストを送る。処理時間がリクエストごとに大きく異なる場合に有効である。

```
最小接続数の動作:

  状態:
  S1: 接続数 = 12  ──→ 選ばれない
  S2: 接続数 = 5   ──→ 次のリクエストはここへ ★
  S3: 接続数 = 8   ──→ 選ばれない

  時間経過による接続数変動:

  S1: ████████████____
  S2: █████___________  ← 最小 → 新規リクエスト投入
  S3: ████████________

  t=0: S1=12, S2=5, S3=8 → S2 に振り分け
  t=1: S1=11, S2=6, S3=7 → S2 に振り分け
  t=2: S1=10, S2=7, S3=6 → S3 に振り分け

  加重最小接続数:
    score = active_connections / weight
    S1: 12/5 = 2.4
    S2: 5/3  = 1.67  ← 最小スコア → 選択
    S3: 8/2  = 4.0

  利点:
    - 処理時間が不均一なワークロードに強い
    - 長時間接続（WebSocket 等）がある場合に有効
    - 自動的に遅いサーバーへのリクエストが減少

  欠点:
    - 接続数の追跡オーバーヘッド
    - 新規サーバー追加直後に一時的にリクエスト集中
```

### 3.4 IP ハッシュ（IP Hash）

クライアントの IP アドレスからハッシュ値を計算し、特定のサーバーに固定的に振り分ける。

```
IP ハッシュの動作:

  hash(client_ip) % server_count = server_index

  例:
  hash("192.168.1.10") % 3 = 0 → S1
  hash("192.168.1.20") % 3 = 2 → S3
  hash("192.168.1.30") % 3 = 1 → S2
  hash("192.168.1.10") % 3 = 0 → S1 （同じ IP は常に同じサーバー）

  ┌──────────────┐     ┌──────┐
  │ 192.168.1.10 │────→│  S1  │  常に S1
  └──────────────┘     └──────┘
  ┌──────────────┐     ┌──────┐
  │ 192.168.1.20 │────→│  S3  │  常に S3
  └──────────────┘     └──────┘
  ┌──────────────┐     ┌──────┐
  │ 192.168.1.30 │────→│  S2  │  常に S2
  └──────────────┘     └──────┘

  問題: サーバー台数変更時にほぼ全クライアントの振り分け先が変わる
  解決: コンシステントハッシュを使用
```

### 3.5 コンシステントハッシュ（Consistent Hashing）

サーバーの追加・削除時に、影響を受けるクライアントを最小限に抑えるアルゴリズムである。

```
コンシステントハッシュの動作:

  ハッシュ空間をリング状に配置:

              0
           ╱     ╲
         ╱    S1   ╲
       ╱    (pos=30) ╲
      │                │
  270 │     ハッシュ     │ 90
      │     リング      │
       ╲              ╱
         ╲   S2     ╱
           ╲(pos=150)╱
             ╲   ╱
              180
                S3 (pos=210)

  クライアント IP のハッシュ値から時計回りで最初のサーバーに割り当て:

  hash("client_A") = 50  → 時計回り → S2 (pos=150)
  hash("client_B") = 100 → 時計回り → S2 (pos=150)
  hash("client_C") = 200 → 時計回り → S3 (pos=210)
  hash("client_D") = 250 → 時計回り → S1 (pos=30 ※リング循環)

  S4 (pos=120) を追加した場合:
  → client_B (hash=100) だけが S2 → S4 に変更
  → 他のクライアントは影響なし

  仮想ノード:
    各物理サーバーを複数の仮想ノードとしてリング上に配置
    → 負荷の偏りを軽減
    → S1 → S1_v1(30), S1_v2(120), S1_v3(250) のように分散配置
```

### 3.6 最小レスポンスタイム（Least Response Time）

ヘルスチェックの応答時間を考慮し、最も応答が速いサーバーにリクエストを送る。

```
最小レスポンスタイムの動作:

  各サーバーの応答時間を継続的に計測:

  S1: 平均応答時間 = 15ms  ← 最速 → 選択
  S2: 平均応答時間 = 45ms
  S3: 平均応答時間 = 30ms

  応答時間は指数移動平均（EMA）で更新:
    new_avg = α × latest_response + (1 - α) × old_avg
    α = 0.3 (新しい値の重み)

  利点:
    - サーバーの実際のパフォーマンスを反映
    - ネットワーク遅延も考慮される

  欠点:
    - ヘルスチェック結果と実際のリクエスト処理時間の乖離
    - 計測オーバーヘッド
    - パフォーマンスが不安定なサーバーで振動する可能性
```

### 3.7 アルゴリズム選定フローチャート

```
アルゴリズム選定:

  セッション維持は必要？
    ├── Yes → Cookie ベースで可能？
    │         ├── Yes → L7 LB + Cookie Sticky
    │         └── No  → IP ハッシュ or コンシステントハッシュ
    └── No  → サーバーのスペックは均一？
              ├── Yes → リクエスト処理時間は均一？
              │         ├── Yes → ラウンドロビン
              │         └── No  → 最小接続数
              └── No  → 加重ラウンドロビン or 加重最小接続数

  ワークロード別推奨:
  ┌────────────────────────┬──────────────────────────┐
  │ ワークロード            │ 推奨アルゴリズム          │
  ├────────────────────────┼──────────────────────────┤
  │ REST API（ステートレス）│ ラウンドロビン            │
  │ WebSocket 長時間接続   │ 最小接続数               │
  │ キャッシュサーバー      │ コンシステントハッシュ    │
  │ 異機種混合環境          │ 加重ラウンドロビン        │
  │ リアルタイム処理        │ 最小レスポンスタイム      │
  │ マイクロサービス        │ ラウンドロビン + ヘルス   │
  │ ゲームサーバー          │ 最小接続数 + 地理考慮    │
  └────────────────────────┴──────────────────────────┘
```

---

## 4. Nginx によるロードバランシング設定

### 4.1 基本設定（ラウンドロビン）

```nginx
# /etc/nginx/nginx.conf
# Nginx ロードバランサー基本設定

# ワーカープロセス数: CPU コア数に合わせる
worker_processes auto;

# イベント処理設定
events {
    worker_connections 65536;   # 1 ワーカーあたりの最大接続数
    use epoll;                  # Linux 環境では epoll を使用
    multi_accept on;            # 複数接続の同時受付
}

http {
    # ログフォーマット: アップストリームの情報を含む
    log_format upstream_log '$remote_addr - $remote_user [$time_local] '
                           '"$request" $status $body_bytes_sent '
                           '"$http_referer" "$http_user_agent" '
                           'upstream: $upstream_addr '
                           'response_time: $upstream_response_time '
                           'status: $upstream_status';

    access_log /var/log/nginx/access.log upstream_log;

    # アップストリーム定義: バックエンドサーバー群
    upstream backend_servers {
        # デフォルトはラウンドロビン
        server 10.0.1.1:8080;
        server 10.0.1.2:8080;
        server 10.0.1.3:8080;
    }

    server {
        listen 80;
        server_name api.example.com;

        location / {
            proxy_pass http://backend_servers;

            # プロキシヘッダーの設定
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # タイムアウト設定
            proxy_connect_timeout 10s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;

            # バッファリング設定
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }
    }
}
```

### 4.2 高度な Nginx 設定（加重・ヘルスチェック・SSL 終端）

```nginx
# /etc/nginx/conf.d/load-balancer.conf
# 高度なロードバランサー設定

# 加重ラウンドロビン + バックアップサーバー
upstream api_servers {
    # 加重ラウンドロビン
    server 10.0.1.1:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 10.0.1.2:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 10.0.1.3:8080 weight=2 max_fails=3 fail_timeout=30s;

    # バックアップサーバー（他が全滅時のみ使用）
    server 10.0.1.99:8080 backup;

    # パッシブヘルスチェック設定:
    #   max_fails=3:    3 回連続失敗で除外
    #   fail_timeout=30s: 30 秒後に再度チェック

    # 接続キープアライブ
    keepalive 32;
    keepalive_timeout 60s;
    keepalive_requests 1000;
}

# 最小接続数アルゴリズム
upstream websocket_servers {
    least_conn;
    server 10.0.2.1:8080;
    server 10.0.2.2:8080;
    server 10.0.2.3:8080;
}

# IP ハッシュ（セッション固定）
upstream session_servers {
    ip_hash;
    server 10.0.3.1:8080;
    server 10.0.3.2:8080;
    server 10.0.3.3:8080;

    # サーバーを一時的に除外（ハッシュ値を維持）
    # server 10.0.3.4:8080 down;
}

# SSL 終端 + パスベースルーティング
server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL 証明書
    ssl_certificate     /etc/ssl/certs/api.example.com.crt;
    ssl_certificate_key /etc/ssl/private/api.example.com.key;

    # SSL セキュリティ設定
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # HSTS ヘッダー
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # パスベースルーティング
    location /api/ {
        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket サーバー
    location /ws/ {
        proxy_pass http://websocket_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;  # WebSocket 用に長めのタイムアウト
    }

    # ヘルスチェックエンドポイント（LB 自体のヘルス）
    location /health {
        access_log off;
        return 200 '{"status": "healthy"}';
        add_header Content-Type application/json;
    }

    # 静的ファイル（直接配信）
    location /static/ {
        alias /var/www/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}

# HTTP → HTTPS リダイレクト
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

---

## 5. HAProxy によるロードバランシング

### 5.1 HAProxy 基本設定

HAProxy は高性能なロードバランサーであり、L4/L7 両方のモードで動作する。Linux カーネルの最適化と組み合わせることで、数百万の同時接続を処理できる。

```haproxy
# /etc/haproxy/haproxy.cfg
# HAProxy 基本設定

#--------------------------------------------------
# グローバル設定
#--------------------------------------------------
global
    log         /dev/log local0
    log         /dev/log local1 notice
    chroot      /var/lib/haproxy
    pidfile     /var/run/haproxy.pid
    maxconn     100000              # 最大同時接続数
    user        haproxy
    group       haproxy
    daemon

    # SSL/TLS 設定
    ssl-default-bind-ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256
    ssl-default-bind-options no-sslv3 no-tlsv10 no-tlsv11

    # パフォーマンスチューニング
    tune.ssl.default-dh-param 2048
    tune.bufsize 32768
    tune.maxrewrite 1024

#--------------------------------------------------
# デフォルト設定
#--------------------------------------------------
defaults
    log     global
    mode    http                    # L7 モード（tcp で L4）
    option  httplog
    option  dontlognull
    option  http-server-close       # サーバー側のコネクション再利用
    option  forwardfor              # X-Forwarded-For ヘッダー追加
    option  redispatch              # サーバー障害時に別サーバーへ再送

    retries                 3       # 再試行回数
    timeout http-request    10s     # リクエスト受信タイムアウト
    timeout queue           1m      # キュー待ちタイムアウト
    timeout connect         10s     # バックエンド接続タイムアウト
    timeout client          1m      # クライアント側タイムアウト
    timeout server          1m      # サーバー側タイムアウト
    timeout http-keep-alive 10s     # Keep-Alive タイムアウト
    timeout check           10s     # ヘルスチェックタイムアウト

    # エラーページ
    errorfile 400 /etc/haproxy/errors/400.http
    errorfile 403 /etc/haproxy/errors/403.http
    errorfile 408 /etc/haproxy/errors/408.http
    errorfile 500 /etc/haproxy/errors/500.http
    errorfile 502 /etc/haproxy/errors/502.http
    errorfile 503 /etc/haproxy/errors/503.http
    errorfile 504 /etc/haproxy/errors/504.http

#--------------------------------------------------
# 統計ダッシュボード
#--------------------------------------------------
listen stats
    bind *:8404
    mode http
    stats enable
    stats uri /stats
    stats refresh 10s
    stats admin if LOCALHOST          # localhost からのみ管理操作可
    stats auth admin:SecureP@ss123    # 認証設定

#--------------------------------------------------
# フロントエンド: HTTPS 受付
#--------------------------------------------------
frontend https_front
    bind *:443 ssl crt /etc/ssl/certs/example.com.pem alpn h2,http/1.1
    bind *:80

    # HTTP → HTTPS リダイレクト
    http-request redirect scheme https unless { ssl_fc }

    # セキュリティヘッダー
    http-response set-header Strict-Transport-Security "max-age=31536000"
    http-response set-header X-Content-Type-Options nosniff
    http-response set-header X-Frame-Options DENY

    # ACL（アクセス制御リスト）によるルーティング
    acl is_api     path_beg /api/
    acl is_ws      path_beg /ws/
    acl is_static  path_beg /static/
    acl is_admin   hdr(host) -i admin.example.com
    acl is_health  path /health

    # レートリミット
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny deny_status 429 if { sc_http_req_rate(0) gt 100 }

    # バックエンド振り分け
    use_backend api_backend     if is_api
    use_backend ws_backend      if is_ws
    use_backend static_backend  if is_static
    use_backend admin_backend   if is_admin
    use_backend health_backend  if is_health
    default_backend web_backend

#--------------------------------------------------
# バックエンド: API サーバー
#--------------------------------------------------
backend api_backend
    balance roundrobin
    option httpchk GET /health HTTP/1.1\r\nHost:\ api.internal
    http-check expect status 200

    # Connection Draining: 除外時に処理中リクエストを待つ
    default-server inter 5s fall 3 rise 2 slowstart 60s

    server api1 10.0.1.1:8080 check weight 5
    server api2 10.0.1.2:8080 check weight 3
    server api3 10.0.1.3:8080 check weight 2

#--------------------------------------------------
# バックエンド: WebSocket サーバー
#--------------------------------------------------
backend ws_backend
    balance leastconn
    option httpchk GET /health

    # WebSocket 用タイムアウト延長
    timeout server 3600s
    timeout tunnel 3600s

    server ws1 10.0.2.1:8080 check
    server ws2 10.0.2.2:8080 check
    server ws3 10.0.2.3:8080 check

#--------------------------------------------------
# バックエンド: 静的ファイルサーバー
#--------------------------------------------------
backend static_backend
    balance roundrobin
    option httpchk GET /health

    http-response set-header Cache-Control "public, max-age=2592000"

    server static1 10.0.3.1:80 check
    server static2 10.0.3.2:80 check

#--------------------------------------------------
# バックエンド: 管理画面
#--------------------------------------------------
backend admin_backend
    balance roundrobin

    # IP 制限
    acl allowed_ip src 10.0.0.0/8 172.16.0.0/12
    http-request deny unless allowed_ip

    server admin1 10.0.4.1:8080 check
    server admin2 10.0.4.2:8080 check

#--------------------------------------------------
# バックエンド: Web サーバー（デフォルト）
#--------------------------------------------------
backend web_backend
    balance roundrobin
    option httpchk GET /health
    cookie SERVERID insert indirect nocache

    server web1 10.0.5.1:8080 check cookie web1
    server web2 10.0.5.2:8080 check cookie web2
    server web3 10.0.5.3:8080 check cookie web3

#--------------------------------------------------
# バックエンド: ヘルスチェック
#--------------------------------------------------
backend health_backend
    mode http
    http-request return status 200 content-type application/json \
        lf-string '{"status":"healthy","timestamp":"%[date]"}'
```

### 5.2 HAProxy L4 モード設定

```haproxy
# /etc/haproxy/haproxy-l4.cfg
# L4（TCP）モード設定例: データベースプロキシ

defaults
    mode tcp
    log global
    option tcplog
    timeout connect 10s
    timeout client  30m
    timeout server  30m

# PostgreSQL プライマリ
frontend pgsql_front
    bind *:5432
    default_backend pgsql_backend

backend pgsql_backend
    balance leastconn
    option pgsql-check user haproxy  # PostgreSQL 固有のヘルスチェック

    server pgsql1 10.0.10.1:5432 check inter 3s fall 3 rise 2
    server pgsql2 10.0.10.2:5432 check inter 3s fall 3 rise 2 backup

# Redis Sentinel 構成
frontend redis_front
    bind *:6379
    default_backend redis_backend

backend redis_backend
    balance first                    # 最初の正常なサーバーへ
    option tcp-check
    tcp-check send PING\r\n
    tcp-check expect string +PONG

    server redis1 10.0.11.1:6379 check inter 1s fall 3 rise 2
    server redis2 10.0.11.2:6379 check inter 1s fall 3 rise 2
    server redis3 10.0.11.3:6379 check inter 1s fall 3 rise 2
```

---

## 6. ヘルスチェックの設計

### 6.1 ヘルスチェックの種類と段階

```
ヘルスチェックの 3 段階:

┌──────────────────────────────────────────────────────────┐
│                   ヘルスチェックの階層                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Level 1: インフラレベル                                   │
│  ┌────────────────────────────────────────────┐          │
│  │ TCP ポート接続確認                          │          │
│  │ → ポート 8080 に接続可能か？                │          │
│  │ → プロセスが起動しているかの最低限の確認    │          │
│  └────────────────────────────────────────────┘          │
│                    ↓                                      │
│  Level 2: アプリケーションレベル                            │
│  ┌────────────────────────────────────────────┐          │
│  │ HTTP エンドポイント応答確認                  │          │
│  │ → GET /health → 200 OK                    │          │
│  │ → アプリケーションが正常に動作しているか     │          │
│  └────────────────────────────────────────────┘          │
│                    ↓                                      │
│  Level 3: 依存サービスレベル                               │
│  ┌────────────────────────────────────────────┐          │
│  │ 深いヘルスチェック（Deep Health Check）      │          │
│  │ → DB 接続、キャッシュ接続、外部 API         │          │
│  │ → 全依存サービスが正常か                    │          │
│  └────────────────────────────────────────────┘          │
│                                                          │
│  ロードバランサーには Level 1-2 を使用                      │
│  Level 3 は Kubernetes の Readiness Probe 等で使用        │
└──────────────────────────────────────────────────────────┘
```

### 6.2 ヘルスチェックエンドポイントの実装例

```python
# Python (FastAPI) でのヘルスチェック実装例
from fastapi import FastAPI, Response, status
from datetime import datetime, timezone
import asyncpg
import aioredis
import time

app = FastAPI()
start_time = time.time()

# === Liveness Check（L1: プロセスは生きているか）===
@app.get("/healthz")
async def liveness():
    """
    Liveness probe: アプリケーションプロセスの生存確認。
    常に 200 を返す（デッドロック検出を除く）。
    Kubernetes の livenessProbe や LB の基本ヘルスチェックに使用。
    """
    return {"status": "alive"}


# === Readiness Check（L2: リクエストを受ける準備ができているか）===
@app.get("/ready")
async def readiness():
    """
    Readiness probe: サービスがリクエストを処理可能か確認。
    起動直後の初期化完了チェックにも使用。
    LB のヘルスチェックに推奨。
    """
    checks = {}
    is_ready = True

    # データベース接続チェック
    try:
        conn = await asyncpg.connect(
            host="db.internal", port=5432,
            user="app", password="secret", database="mydb",
            timeout=3
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        checks["database"] = "connected"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        is_ready = False

    # Redis 接続チェック
    try:
        redis = aioredis.from_url("redis://cache.internal:6379")
        await redis.ping()
        await redis.close()
        checks["cache"] = "connected"
    except Exception as e:
        checks["cache"] = f"error: {str(e)}"
        is_ready = False

    uptime_seconds = int(time.time() - start_time)

    response_data = {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if not is_ready:
        return Response(
            content=json.dumps(response_data),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json"
        )
    return response_data


# === Deep Health Check（L3: 全依存サービスの状態）===
@app.get("/health/deep")
async def deep_health():
    """
    Deep health check: 全依存サービスの詳細な状態を返す。
    監視システムや運用ダッシュボードから呼び出す。
    LB のヘルスチェックには使用しない（カスケード障害の原因になる）。
    """
    checks = {}

    # 各依存サービスのチェック（省略: 上記と同様）
    # ...

    return {
        "status": "healthy",
        "version": "2.3.1",
        "checks": checks,
        "system": {
            "uptime_seconds": int(time.time() - start_time),
            "memory_usage_mb": get_memory_usage(),
            "cpu_percent": get_cpu_usage()
        }
    }
```

### 6.3 ヘルスチェックパラメータの設計指針

```
ヘルスチェックパラメータの相互関係:

  interval (チェック間隔)
  ────┤    ├────┤    ├────┤    ├────
      check    check    check

  fail threshold (不健全しきい値) = 3
  ────X────X────X──→ 除外
       fail   fail  fail

  rise threshold (復帰しきい値) = 2
  ────✓────✓──→ 復帰
      pass  pass

  障害検知にかかる最大時間:
    = interval × fail_threshold
    = 10s × 3 = 30s

  推奨パラメータ:

  ┌────────────────────┬──────────┬─────────┬──────────┐
  │ サービス種別        │ interval │ fall    │ rise     │
  ├────────────────────┼──────────┼─────────┼──────────┤
  │ Web API            │ 10s      │ 3       │ 2        │
  │ リアルタイム通信    │ 3s       │ 2       │ 2        │
  │ バッチ処理         │ 30s      │ 5       │ 3        │
  │ データベース        │ 5s       │ 3       │ 2        │
  │ キャッシュ         │ 3s       │ 2       │ 1        │
  └────────────────────┴──────────┴─────────┴──────────┘

  注意点:
  ① interval が短すぎるとバックエンドに負荷がかかる
  ② fall が少なすぎると一時的なエラーで不必要に除外される（フラッピング）
  ③ rise が多すぎると復帰に時間がかかりすぎる
  ④ タイムアウトは interval より短く設定すること
```

### 6.4 グレースフルシャットダウンとコネクションドレイニング

```
グレースフルシャットダウンのフロー:

  時間 ──────────────────────────────────────────→

  t=0: シャットダウン開始
  │
  ├─ ① ヘルスチェック応答を 503 に変更
  │     GET /health → 503 Service Unavailable
  │
  ├─ ② LB がヘルスチェック失敗を検知
  │     (interval × fall = 10s × 3 = 30s 後)
  │
  ├─ ③ LB がサーバーを除外
  │     → 新規リクエストは他サーバーへ
  │
  ├─ ④ コネクションドレイニング開始
  │     → 処理中のリクエストの完了を待つ
  │     → タイムアウト: 300s (AWS ALB デフォルト)
  │
  └─ ⑤ プロセス停止
        → 全リクエスト処理完了 or タイムアウト

  コネクションドレイニングの動作:

  ┌──────────────────────────────────────────────────┐
  │ ドレイニング開始                                    │
  │                                                    │
  │ 処理中のリクエスト:                                  │
  │   R1: ████████████████████████──→ 完了              │
  │   R2: ██████████████──→ 完了                        │
  │   R3: ████████████████████████████████──→ 完了      │
  │                                                    │
  │ 新規リクエスト:                                      │
  │   R4: ──→ 拒否（他サーバーへリダイレクト）            │
  │   R5: ──→ 拒否                                      │
  │                                                    │
  │ 全リクエスト完了後にプロセス停止                      │
  └──────────────────────────────────────────────────┘
```

---

## 7. AWS のロードバランサー

### 7.1 ALB / NLB / GWLB 比較

```
┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ 比較項目          │ ALB               │ NLB               │ GWLB              │
│                   │ (Application)     │ (Network)         │ (Gateway)         │
├───────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ レイヤー          │ L7                │ L4                │ L3+L4             │
│ プロトコル        │ HTTP, HTTPS,      │ TCP, UDP, TLS     │ IP プロトコル     │
│                   │ gRPC, WebSocket   │                   │                   │
│ パフォーマンス    │ 数十万 RPS        │ 数百万 RPS        │ -                 │
│ レイテンシ        │ ms オーダー       │ μs オーダー       │ -                 │
│ 静的 IP          │ なし（DNS 名）    │ あり（EIP 対応）  │ あり              │
│ TLS 終端         │ あり              │ あり（TLS LB）    │ なし              │
│ パスルーティング  │ あり              │ なし              │ なし              │
│ ホストルーティング│ あり              │ なし              │ なし              │
│ ヘルスチェック    │ HTTP/HTTPS/gRPC   │ TCP/HTTP/HTTPS    │ HTTP/HTTPS/TCP    │
│ Sticky Session   │ Cookie ベース     │ 送信元 IP ベース  │ 5-tuple ベース    │
│ WAF 連携         │ あり              │ なし              │ なし              │
│ Lambda 連携      │ あり              │ なし              │ なし              │
│ クロスゾーン     │ デフォルト有効    │ デフォルト無効    │ デフォルト無効    │
│ 料金モデル        │ LCU ベース        │ NLCU ベース       │ GWLCU ベース      │
│ 主な用途          │ Web アプリ、API   │ ゲーム、IoT、     │ セキュリティ      │
│                   │ マイクロサービス   │ 金融取引           │ アプライアンス    │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

### 7.2 ALB アーキテクチャ

```
ALB のアーキテクチャ:

  ┌─────────────────────────────────────────────────────────┐
  │                       Internet                          │
  └──────────────────────┬──────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────┐
  │                    ALB (L7)                              │
  │  ┌──────────────────────────────────────────────┐       │
  │  │ リスナー (Listener)                           │       │
  │  │   - HTTPS:443 → TLS 終端                     │       │
  │  │   - HTTP:80 → HTTPS リダイレクト              │       │
  │  └──────────────────────────────────────────────┘       │
  │                                                         │
  │  ┌──────────────────────────────────────────────┐       │
  │  │ リスナールール (Rules)                        │       │
  │  │   Priority 1: Host=api.*  → TG-api           │       │
  │  │   Priority 2: Path=/ws/* → TG-websocket      │       │
  │  │   Priority 3: Header=X-Api-Version:v2 → TG-v2│       │
  │  │   Default:    → TG-default                    │       │
  │  └──────────────────────────────────────────────┘       │
  └──┬────────────┬────────────┬────────────┬───────────────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
  ┌──────┐   ┌──────┐   ┌──────────┐  ┌──────────┐
  │TG-api│   │TG-ws │   │TG-v2     │  │TG-default│
  ├──────┤   ├──────┤   ├──────────┤  ├──────────┤
  │ECS   │   │EC2   │   │ECS       │  │EC2       │
  │Fargate│  │i-xxx │   │Fargate   │  │i-yyy     │
  │      │   │i-yyy │   │          │  │i-zzz     │
  └──────┘   └──────┘   └──────────┘  └──────────┘

  Target Group (TG) の設定項目:
    - ターゲットタイプ: instance / ip / lambda
    - プロトコル: HTTP / HTTPS / gRPC
    - ヘルスチェック: パス、間隔、しきい値
    - スティッキーセッション: Cookie 設定
    - 登録解除遅延: コネクションドレイニング
    - スロースタート: 新規ターゲットへの段階的トラフィック増加
```

### 7.3 Terraform による ALB 構築

```hcl
# terraform/modules/alb/main.tf
# AWS ALB + Target Group + Listener の構成

# --- ALB 本体 ---
resource "aws_lb" "main" {
  name               = "app-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = true    # 誤削除防止
  enable_http2              = true     # HTTP/2 有効化
  idle_timeout              = 60       # アイドルタイムアウト（秒）
  drop_invalid_header_fields = true    # 不正ヘッダー破棄

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    prefix  = "alb"
    enabled = true
  }

  tags = {
    Environment = var.environment
    Service     = "web-app"
  }
}

# --- ALB セキュリティグループ ---
resource "aws_security_group" "alb_sg" {
  name_prefix = "alb-sg-"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTPS from Internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP from Internet (redirect to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- ターゲットグループ: API ---
resource "aws_lb_target_group" "api" {
  name                 = "api-tg"
  port                 = 8080
  protocol             = "HTTP"
  vpc_id               = var.vpc_id
  target_type          = "ip"          # Fargate の場合は ip
  deregistration_delay = 30            # ドレイニング待機時間

  health_check {
    enabled             = true
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 10
    matcher             = "200"
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = 3600             # 1 時間
    enabled         = false
  }

  # スロースタート: 新規ターゲットに段階的にトラフィック増加
  slow_start = 60                      # 60 秒かけて 100% に到達

  tags = {
    Service = "api"
  }
}

# --- ターゲットグループ: WebSocket ---
resource "aws_lb_target_group" "websocket" {
  name        = "ws-tg"
  port        = 8081
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "instance"

  health_check {
    enabled             = true
    path                = "/ws/health"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 3
    interval            = 5
    matcher             = "200"
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true             # WebSocket はスティッキー推奨
  }
}

# --- HTTPS リスナー ---
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# --- HTTP → HTTPS リダイレクト ---
resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# --- リスナールール: WebSocket パス ---
resource "aws_lb_listener_rule" "websocket" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.websocket.arn
  }

  condition {
    path_pattern {
      values = ["/ws/*"]
    }
  }
}

# --- リスナールール: API バージョニング ---
resource "aws_lb_listener_rule" "api_v2" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 20

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api_v2.arn
  }

  condition {
    http_header {
      http_header_name = "X-Api-Version"
      values           = ["v2"]
    }
  }
}

# --- WAF 連携 ---
resource "aws_wafv2_web_acl_association" "alb" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = var.waf_web_acl_arn
}
```

### 7.4 Terraform による NLB 構築

```hcl
# terraform/modules/nlb/main.tf
# AWS NLB（L4）の構成

resource "aws_lb" "nlb" {
  name               = "app-nlb"
  internal           = false
  load_balancer_type = "network"
  subnets            = var.public_subnet_ids

  enable_deletion_protection       = true
  enable_cross_zone_load_balancing = true  # クロスゾーン LB

  # 静的 IP（Elastic IP）の割り当て
  dynamic "subnet_mapping" {
    for_each = var.public_subnet_ids
    content {
      subnet_id     = subnet_mapping.value
      allocation_id = var.eip_allocation_ids[subnet_mapping.key]
    }
  }

  tags = {
    Environment = var.environment
    Service     = "game-server"
  }
}

# --- ターゲットグループ: TCP ---
resource "aws_lb_target_group" "tcp" {
  name        = "game-tcp-tg"
  port        = 7777
  protocol    = "TCP"
  vpc_id      = var.vpc_id
  target_type = "instance"

  # TCP ヘルスチェック
  health_check {
    enabled             = true
    port                = "traffic-port"
    protocol            = "TCP"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 10
  }

  # コネクション維持設定
  connection_termination = false
  deregistration_delay   = 300

  # Proxy Protocol v2（クライアント IP 保持）
  proxy_protocol_v2 = true

  stickiness {
    enabled = true
    type    = "source_ip"
  }
}

# --- UDP ターゲットグループ（ゲームサーバー等）---
resource "aws_lb_target_group" "udp" {
  name        = "game-udp-tg"
  port        = 7778
  protocol    = "UDP"
  vpc_id      = var.vpc_id
  target_type = "instance"

  health_check {
    enabled             = true
    port                = 7777         # TCP ポートでヘルスチェック
    protocol            = "TCP"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 10
  }
}

# --- TCP リスナー ---
resource "aws_lb_listener" "tcp" {
  load_balancer_arn = aws_lb.nlb.arn
  port              = 7777
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.tcp.arn
  }
}

# --- TLS リスナー（NLB での TLS 終端）---
resource "aws_lb_listener" "tls" {
  load_balancer_arn = aws_lb.nlb.arn
  port              = 443
  protocol          = "TLS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.tcp.arn
  }
}
```

---

## 8. 高可用性（HA）構成

### 8.1 アクティブ-スタンバイ構成

ロードバランサー自体が単一障害点にならないよう、HA 構成を組む必要がある。

```
アクティブ-スタンバイ構成:

  ┌──────────────────────────────────────────────────┐
  │              仮想 IP (VIP): 10.0.0.100            │
  │              DNS: lb.example.com                  │
  └──────────┬───────────────────┬────────────────────┘
             │                   │
    ┌────────▼────────┐ ┌───────▼─────────┐
    │ LB1 (Active)    │ │ LB2 (Standby)   │
    │ 10.0.0.1        │ │ 10.0.0.2        │
    │                 │ │                 │
    │ VRRP Master     │ │ VRRP Backup     │
    │ Priority: 100   │ │ Priority: 50    │
    │ ★ VIP 保持     │ │                 │
    └─────┬──┬──┬─────┘ └─────┬──┬──┬─────┘
          │  │  │              │  │  │
     ┌────▼──▼──▼──────────────▼──▼──▼────┐
     │        バックエンドサーバー群         │
     │  S1    S2    S3    S4    S5    S6   │
     └────────────────────────────────────┘

  フェイルオーバー:
  1. LB1 が障害発生
  2. VRRP が障害を検知（2-3 秒以内）
  3. LB2 が VIP を引き継ぎ Master に昇格
  4. クライアントは VIP 宛のため切り替えを意識しない
  5. ダウンタイム: 1-3 秒程度
```

### 8.2 アクティブ-アクティブ構成

```
アクティブ-アクティブ構成:

  ┌───────────────────────────────────────────────┐
  │           DNS ラウンドロビン                    │
  │  lb.example.com → 10.0.0.1, 10.0.0.2         │
  └───────┬───────────────────────┬───────────────┘
          │                       │
  ┌───────▼───────┐     ┌────────▼──────┐
  │ LB1 (Active)  │     │ LB2 (Active)  │
  │ 10.0.0.1      │     │ 10.0.0.2      │
  │ 50% トラフィック│     │ 50% トラフィック│
  └──┬──┬──┬──────┘     └──┬──┬──┬──────┘
     │  │  │                │  │  │
     └──┼──┼────────────────┼──┼──┘
        │  │                │  │
   ┌────▼──▼────────────────▼──▼────┐
   │      バックエンドサーバー群      │
   │  S1    S2    S3    S4    S5    │
   └────────────────────────────────┘

  利点:
    - 両方の LB が常にトラフィックを処理 → リソース効率が高い
    - 片方の障害時も残りの LB で処理継続

  欠点:
    - DNS TTL の問題（切り替えに時間がかかる場合がある）
    - セッション共有の仕組みが必要
```

### 8.3 Keepalived による VRRP 設定

```bash
# /etc/keepalived/keepalived.conf
# Keepalived VRRP 設定（アクティブ-スタンバイ）

# --- グローバル設定 ---
global_defs {
    router_id LB1                    # ノード識別子
    script_user root
    enable_script_security

    notification_email {
        ops@example.com              # 障害通知メール
    }
    notification_email_from keepalived@example.com
    smtp_server smtp.example.com
    smtp_connect_timeout 30
}

# --- HAProxy の生存確認スクリプト ---
vrrp_script check_haproxy {
    script "/usr/bin/killall -0 haproxy"   # プロセス存在確認
    interval 2                              # 2 秒間隔でチェック
    weight -20                              # 失敗時に優先度を -20
    fall 3                                  # 3 回連続失敗で異常判定
    rise 2                                  # 2 回連続成功で正常復帰
}

# --- VRRP インスタンス ---
vrrp_instance VI_1 {
    state MASTER                             # 初期状態: MASTER
    interface eth0                           # 監視するインターフェース
    virtual_router_id 51                     # VRRP グループ ID（同一でペア）
    priority 100                             # 優先度（大きい方が MASTER）
    advert_int 1                             # VRRP 広告間隔（秒）

    authentication {
        auth_type PASS
        auth_pass secretpass                 # VRRP 認証パスワード
    }

    # 仮想 IP アドレス
    virtual_ipaddress {
        10.0.0.100/24 dev eth0               # クライアント向け VIP
    }

    # ヘルスチェックスクリプトの監視
    track_script {
        check_haproxy
    }

    # 状態遷移時の通知スクリプト
    notify_master "/etc/keepalived/notify.sh MASTER"
    notify_backup "/etc/keepalived/notify.sh BACKUP"
    notify_fault  "/etc/keepalived/notify.sh FAULT"
}
```

```bash
#!/bin/bash
# /etc/keepalived/notify.sh
# VRRP 状態遷移時の通知スクリプト

STATE=$1
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

case $STATE in
    "MASTER")
        echo "$TIMESTAMP: Transitioned to MASTER" >> /var/log/keepalived-notify.log
        # Slack/PagerDuty 等への通知
        curl -s -X POST "https://hooks.slack.com/services/xxx" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"$(hostname) is now MASTER at $TIMESTAMP\"}"
        ;;
    "BACKUP")
        echo "$TIMESTAMP: Transitioned to BACKUP" >> /var/log/keepalived-notify.log
        ;;
    "FAULT")
        echo "$TIMESTAMP: FAULT detected" >> /var/log/keepalived-notify.log
        curl -s -X POST "https://hooks.slack.com/services/xxx" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"ALERT: $(hostname) entered FAULT state at $TIMESTAMP\"}"
        ;;
esac
```

---

## 前提知識

本ガイドを読む前に、以下の知識があると理解がスムーズになる。

- **TCP の基礎**: 3ウェイハンドシェイク、コネクション管理、TCP フラグ（SYN, ACK, FIN, RST）の意味を理解していること。詳細は [TCP プロトコル](../01-protocols/00-tcp.md) を参照
- **HTTP の基本**: HTTP メソッド（GET/POST/PUT/DELETE）、ステータスコード、リクエスト/レスポンスヘッダーの役割を把握していること。詳細は [HTTP の基礎](../02-http/00-http-basics.md) を参照
- **DNS の仕組み**: 名前解決の流れ、A/AAAA レコード、TTL の概念を理解していること。詳細は [DNS](../00-introduction/03-dns.md) を参照
- **Linux コマンドラインの基本**: `curl`, `ping`, `ss`/`netstat` などの基本的な使い方を知っていること

---

## FAQ（よくある質問）

### Q1: L4 ロードバランシングと L7 ロードバランシングの選択基準は？

**A:** 以下の観点で判断する。

**L4 を選ぶべきケース:**
- 超低レイテンシが必須の場合（ゲームサーバー、金融取引、リアルタイム通信）
- TCP/UDP を直接扱うプロトコルの場合（データベースプロキシ、VPN、DNS、メールサーバー）
- TLS パススルーが必要な場合（クライアント証明書認証をバックエンドで行う）
- 大量コネクション処理（数百万同時接続の IoT デバイス等）
- スループット最大化が最優先（数百万 RPS が必要）

**L7 を選ぶべきケース:**
- パスベースルーティングが必要（マイクロサービスアーキテクチャ、API バージョニング）
- TLS 終端を LB で行いたい（証明書の一元管理、バックエンドの負荷軽減）
- A/B テスト、カナリアデプロイ、ブルーグリーンデプロイを実施したい
- WAF（Web Application Firewall）と連携したい
- レスポンスの圧縮・キャッシュ・ヘッダー操作が必要
- WebSocket や gRPC のルーティングを HTTP パスで制御したい

**両方を組み合わせるパターン:**
- L7 LB（ALB/Nginx）を複数配置し、それらを L4 LB（NLB）で束ねる構成
- L4 で大量のトラフィックを分散し、L7 で詳細なルーティング制御を実現

### Q2: ヘルスチェックはどのように設計すべきか？

**A:** ヘルスチェックは 3 段階に分けて設計する。

**Level 1: インフラレベル（LB の基本ヘルスチェックに使用）**
- TCP ポート接続確認（`nc -zv <host> <port>`）
- 最低限の生存確認のみ
- 失敗 = プロセスが起動していない or ポートが閉じている

**Level 2: アプリケーションレベル（LB の標準ヘルスチェックに推奨）**
- HTTP エンドポイント応答確認（例: `GET /health → 200 OK`）
- アプリケーションが正常に動作しているか確認
- レスポンスタイムが閾値内か確認
- 依存サービスへの軽量な接続確認（DB への `SELECT 1` 程度）

**Level 3: 深いヘルスチェック（監視システムやオペレーター向け）**
- 全依存サービスの詳細チェック（DB, キャッシュ, 外部 API, キュー等）
- `/health/deep` や `/health/detailed` エンドポイントで提供
- LB のヘルスチェックには使わない（カスケード障害の原因になる）

**ヘルスチェックパラメータ設計指針:**

| サービス種別 | interval | fall (unhealthy判定) | rise (healthy復帰) | timeout |
|------------|---------|---------------------|-------------------|---------|
| Web API | 10s | 3 回 | 2 回 | 5s |
| リアルタイム通信 | 3s | 2 回 | 2 回 | 2s |
| バッチ処理 | 30s | 5 回 | 3 回 | 10s |
| データベース | 5s | 3 回 | 2 回 | 3s |

**注意点:**
- `interval` が短すぎるとバックエンドに負荷がかかる
- `fall` が少なすぎると一時的なエラーで不必要に除外される（フラッピング）
- `rise` が多すぎると復帰に時間がかかりすぎる
- `timeout` は必ず `interval` より短く設定すること

### Q3: セッションスティッキー（Session Affinity）はどう実装すべきか？

**A:** アプリケーションの性質に応じて以下の手法を選択する。

**推奨: ステートレス設計 + 外部ストア（最も堅牢）**
- セッション情報を Redis や DynamoDB に保存
- ロードバランサーはラウンドロビンで自由に振り分け
- サーバー障害時も他サーバーでセッション継続可能
- 水平スケール時もセッション維持可能

**方式 1: Cookie ベース（L7 LB で使用）**
```nginx
# Nginx の例
upstream backend {
    server 10.0.1.1:8080;
    server 10.0.1.2:8080;
    server 10.0.1.3:8080;
    sticky cookie srv_id expires=1h domain=.example.com path=/;
}
```
- LB が Cookie を発行し、同じサーバーに振り分け
- AWS ALB の場合: Target Group の Stickiness 設定を有効化
- 利点: アプリケーション側の変更不要
- 欠点: サーバー障害時にセッション喪失、Cookie の改ざんリスク

**方式 2: IP ハッシュ（L4 LB で使用）**
```haproxy
backend app_backend
    balance source  # 送信元IPでハッシュ
    server app1 10.0.1.1:8080
    server app2 10.0.1.2:8080
    server app3 10.0.1.3:8080
```
- クライアント IP で振り分け先を固定
- AWS NLB の場合: デフォルトで送信元 IP ベースのスティッキネス
- 利点: Cookie 不要、シンプル
- 欠点: NAT 配下のクライアントが全て同じサーバーに集中、IP 変更時にセッション喪失

**方式 3: アプリケーションレベル（JWT 等）**
```javascript
// JWT にサーバーIDを含める（非推奨）
const token = jwt.sign({ userId: 123, serverId: 'app-2' }, secret);

// 推奨: JWT を使うが外部ストアでセッション管理
const token = jwt.sign({ userId: 123, sessionId: 'uuid' }, secret);
// sessionId で Redis からセッション情報を取得
```
- JWT トークンにサーバー識別子を含める
- または JWT + 外部ストアの組み合わせ
- 利点: LB に依存しない、マイクロサービス間で共有可能
- 欠点: 実装コストが高い

**結論: 新規システムならステートレス + 外部ストアを強く推奨。既存システムの移行が難しい場合のみ Cookie ベースを使用。**

---

## まとめ

| 概念 | ポイント |
|------|---------|
| L4 vs L7 | L4=高速・低機能、L7=柔軟・高機能、用途で使い分け |
| アルゴリズム | ラウンドロビン（基本）、最小接続数（長時間接続）、コンシステントハッシュ（キャッシュ）|
| ヘルスチェック | 3段階設計（インフラ/アプリ/深層）、LBには Level 1-2 を使用 |
| セッション | ステートレス + 外部ストアが最善、次点で Cookie ベース |
| HA構成 | VRRP/Keepalived でアクティブ-スタンバイ、DNS RR でアクティブ-アクティブ |
| AWS | ALB=L7・マイクロサービス向き、NLB=L4・超高速、GWLB=セキュリティアプライアンス |
| 運用 | グレースフルシャットダウン + コネクションドレイニングでゼロダウンタイムデプロイ |

---

## 次に読むべきガイド

ロードバランシングを理解したら、次は以下のトピックに進むことを推奨する。

- **[CDN（Content Delivery Network）](./01-cdn.md)**: ロードバランシングの延長として、地理分散型のコンテンツ配信ネットワークの仕組みとキャッシュ戦略を学ぶ
- **[ネットワークデバッグ](./02-network-debugging.md)**: ロードバランサーやバックエンドの問題を切り分けるためのデバッグツールとトラブルシューティング手法を習得する
- **[パフォーマンス最適化](./03-performance.md)**: ロードバランシング設計を踏まえた総合的なネットワークパフォーマンスチューニングを実践する

---

## 参考文献

1. Nginx. "HTTP Load Balancing." nginx.org, 2024.
2. HAProxy Technologies. "HAProxy Configuration Manual." haproxy.com, 2024.
3. AWS. "Elastic Load Balancing Documentation." docs.aws.amazon.com, 2024.
4. Karger, D., et al. "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." ACM, 1997.
5. RFC 5798. "Virtual Router Redundancy Protocol (VRRP) Version 3." IETF, 2010.
6. Google. "The Google File System." SOSP, 2003.

