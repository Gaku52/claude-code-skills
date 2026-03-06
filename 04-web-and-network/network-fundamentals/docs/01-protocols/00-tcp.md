# TCP（Transmission Control Protocol）

> TCPは信頼性のある通信を実現するプロトコル。3-wayハンドシェイク、シーケンス番号、フロー制御、輻輳制御の仕組みを理解し、なぜWebの通信基盤なのかを学ぶ。本ガイドでは、RFC 9293 に基づく最新のTCP仕様を網羅的に解説し、tcpdump・Wireshark・ソケットプログラミングを用いた実践的な分析手法を身につける。

## この章で学ぶこと

- [ ] TCPの3-wayハンドシェイクの各ステップと状態遷移を理解する
- [ ] フロー制御（スライディングウィンドウ）の動作原理を把握する
- [ ] 輻輳制御アルゴリズム（Reno, CUBIC, BBR）の違いを説明できる
- [ ] tcpdumpとWiresharkを用いてTCPパケットを解析できる
- [ ] ソケットプログラミングでTCP通信を実装できる
- [ ] TCPヘッダーの各フィールドの役割を把握する
- [ ] TIME_WAIT問題やNagleアルゴリズムなどの実運用上の課題を理解する

---

## 1. TCPの基本特性と位置づけ

### 1.1 OSI参照モデルにおけるTCPの位置

```
┌─────────────────────────────────────────────────────────┐
│  OSI参照モデルにおけるTCPの位置                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Layer 7  アプリケーション層  HTTP, FTP, SMTP, SSH      │
│            ─────────────────────────────────────────    │
│   Layer 6  プレゼンテーション層  TLS/SSL                  │
│            ─────────────────────────────────────────    │
│   Layer 5  セッション層                                   │
│            ─────────────────────────────────────────    │
│   Layer 4  トランスポート層   ★ TCP / UDP ★              │
│            ─────────────────────────────────────────    │
│   Layer 3  ネットワーク層     IP (IPv4, IPv6)             │
│            ─────────────────────────────────────────    │
│   Layer 2  データリンク層     Ethernet, Wi-Fi             │
│            ─────────────────────────────────────────    │
│   Layer 1  物理層             銅線, 光ファイバー            │
│                                                         │
│   TCP/IPモデル（4層）:                                    │
│     アプリケーション層 → HTTP, FTP, SMTP                  │
│     トランスポート層   → TCP, UDP                         │
│     インターネット層   → IP                               │
│     ネットワークIF層   → Ethernet                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

TCPはトランスポート層（Layer 4）に位置し、アプリケーション層のプロトコル（HTTP, FTP, SMTP等）に対して信頼性のあるバイトストリーム通信を提供する。IPプロトコルが「ベストエフォート」で配送するパケットに対して、TCPは到達保証・順序保証・エラー検出の機能を付加する。

### 1.2 TCPの特性一覧

```
TCP = コネクション指向の信頼性あるプロトコル

利点:
  [1] 信頼性（Reliability）
      データの到達を保証する。ACKが返らなければ再送する。
      アプリケーション側でデータの欠損を心配する必要がない。

  [2] 順序保証（Ordering）
      送信順に受信側で再構成する。
      シーケンス番号により、到着順が入れ替わっても正しく並べ替える。

  [3] エラー検出（Error Detection）
      TCPヘッダーとペイロードのチェックサム（16ビット）で破損を検知する。
      ただし暗号学的な完全性保証ではないため、TLSとの併用が一般的。

  [4] フロー制御（Flow Control）
      受信側の処理能力に合わせて送信速度を調整する。
      受信ウィンドウ（rwnd）によりバッファオーバーフローを防止する。

  [5] 輻輳制御（Congestion Control）
      ネットワークの混雑状態を推測し、送信レートを動的に調整する。
      ネットワーク全体の安定性に貢献する協調的なメカニズム。

  [6] 全二重通信（Full-Duplex）
      双方向に同時にデータを送受信できる。
      各方向で独立したシーケンス番号を管理する。

欠点:
  [1] オーバーヘッド
      ヘッダーが最小20バイト（オプション含めると最大60バイト）。
      UDPの8バイトヘッダーと比較して大きい。

  [2] 接続確立の遅延
      3-wayハンドシェイクに1.5 RTT（Round Trip Time）が必要。
      TLSを加えると最大3 RTT（TCP 1.5 + TLS 1.5）。
      TLS 1.3 + TCP Fast Openで改善可能。

  [3] Head-of-Line Blocking（HoL Blocking）
      1つのパケットがロストすると、後続パケットの配信が全てブロックされる。
      HTTP/2の多重化でも、TCP層でのHoL Blockingは回避できない。
      → HTTP/3がQUIC（UDP上の独自プロトコル）を採用した主な理由。

  [4] NAT/ファイアウォールの状態管理コスト
      コネクション指向のため、中間機器が状態テーブルを維持する必要がある。
      大量の短寿命接続はNATテーブルを圧迫する。

主な用途:
  HTTP/HTTPS   → Webブラウジング
  FTP          → ファイル転送
  SMTP/IMAP    → メール送受信
  SSH          → リモート管理
  データベース  → MySQL(3306), PostgreSQL(5432)
  → 「データの欠損が許されない通信」全般で使用する
```

### 1.3 TCP vs UDP 比較表

| 特性 | TCP | UDP |
|------|-----|-----|
| コネクション | コネクション指向（3-way handshake） | コネクションレス |
| 信頼性 | あり（ACK + 再送） | なし（ベストエフォート） |
| 順序保証 | あり（シーケンス番号） | なし |
| フロー制御 | あり（ウィンドウ制御） | なし |
| 輻輳制御 | あり（Reno, CUBIC, BBR等） | なし（アプリケーション側で実装可能） |
| ヘッダーサイズ | 20〜60バイト | 8バイト固定 |
| 通信方式 | 全二重 | 単方向でも双方向でも可能 |
| ブロードキャスト | 不可（1対1のみ） | 可能 |
| 遅延 | 大（接続確立 + 再送待ち） | 小（即座に送信） |
| 適用例 | HTTP, FTP, SSH, DB接続 | DNS, NTP, VoIP, 動画ストリーミング |
| HoL Blocking | あり | なし |
| 状態管理 | サーバー側で接続状態を保持 | ステートレス |

### 1.4 TCP vs QUIC 比較表

| 特性 | TCP | QUIC |
|------|-----|------|
| トランスポート層 | OS カーネル内実装 | ユーザースペース実装（UDP上） |
| 暗号化 | オプション（TLS併用） | 必須（TLS 1.3統合） |
| 接続確立 | 1.5 RTT（+ TLS 1.5 RTT） | 1 RTT（0-RTT再接続可能） |
| HoL Blocking | あり（ストリーム単位で影響） | なし（ストリームが独立） |
| 多重化 | なし（HTTP/2で追加） | ネイティブサポート |
| 接続マイグレーション | IPアドレス変更で切断 | Connection IDで継続可能 |
| 輻輳制御 | カーネル実装に依存 | アプリケーション側で柔軟に選択 |
| パケットロス回復 | 全ストリームがブロック | ロストしたストリームのみ影響 |
| 標準化 | RFC 9293（2022年） | RFC 9000（2021年） |
| 普及状況 | ほぼ全てのインターネット通信 | Google, CloudFlare, Meta等で採用拡大中 |

---

## 2. TCPヘッダー構造の詳細

### 2.1 ヘッダーフォーマット

```
TCP ヘッダー構造（20バイト〜60バイト）:

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  ┌───────────────────────────┬───────────────────────────┐
  │      送信元ポート (16bit)   │      宛先ポート (16bit)    │  0-3 byte
  ├─────────────────────────────────────────────────────────┤
  │                   シーケンス番号 (32bit)                  │  4-7 byte
  ├─────────────────────────────────────────────────────────┤
  │                    ACK番号 (32bit)                       │  8-11 byte
  ├──────┬────────┬─┬─┬─┬─┬─┬─┬───────────────────────────┤
  │データ │  予約   │U│A│P│R│S│F│                           │
  │オフセ │  (4bit) │R│C│S│S│Y│I│  ウィンドウサイズ (16bit)  │ 12-15 byte
  │ット   │        │G│K│H│T│N│N│                           │
  │(4bit) │        │ │ │ │ │ │ │                           │
  ├───────────────────────────┬───────────────────────────┤
  │   チェックサム (16bit)      │   緊急ポインタ (16bit)     │ 16-19 byte
  ├───────────────────────────┴───────────────────────────┤
  │              オプション（0〜40バイト）                    │ 20-59 byte
  ├─────────────────────────────────────────────────────────┤
  │                   ペイロード（データ）                    │
  └─────────────────────────────────────────────────────────┘
```

### 2.2 各フィールドの詳細

```
■ 送信元ポート / 宛先ポート（各16ビット）
  範囲: 0〜65535
  ウェルノウンポート: 0〜1023（HTTP=80, HTTPS=443, SSH=22）
  登録ポート: 1024〜49151
  エフェメラルポート: 49152〜65535（クライアント側で自動割当）

■ シーケンス番号（32ビット）
  送信するデータの先頭バイトの番号。
  初期値（ISN: Initial Sequence Number）はランダムに決定される。
  → 予測可能だとTCPシーケンス番号攻撃に脆弱になるため。
  32ビットで約4.3GB分のバイト番号空間を持つ。
  → 高速回線（10Gbps）では約3.4秒で一巡（ラップアラウンド）する。
  → PAWS（Protection Against Wrapped Sequences）で対処。

■ ACK番号（32ビット）
  次に受信を期待するバイト番号。
  累積的確認応答（Cumulative ACK）：この番号未満は全て受信済みを意味する。

■ データオフセット（4ビット）
  TCPヘッダーの長さを4バイト単位で表す。
  最小値: 5（20バイト = オプションなし）
  最大値: 15（60バイト = オプション40バイト）

■ フラグフィールド（6ビット）
  SYN（Synchronize）: 接続開始。ISNの同期に使用。
  ACK（Acknowledge）: ACK番号フィールドが有効であることを示す。
  FIN（Finish）: 接続終了。これ以上送信するデータがないことを示す。
  RST（Reset）: 接続を即座に切断。異常終了やポートスキャン検出で使用。
  PSH（Push）: バッファリングせず即座にアプリケーションに配信。
  URG（Urgent）: 緊急データの存在を示す（現在はほぼ使用されない）。

  追加フラグ（RFC 3168, RFC 3540）:
  ECE（ECN-Echo）: 輻輳通知（ECN）の応答。
  CWR（Congestion Window Reduced）: 輻輳ウィンドウ縮小の通知。
  NS（ECN-Nonce Sum）: ECNのセキュリティ強化（実験的）。

■ ウィンドウサイズ（16ビット）
  受信可能なバイト数を通知する。
  最大値: 65535バイト（約64KB）。
  → Window Scaleオプションで最大1GBまで拡張可能。

■ チェックサム（16ビット）
  TCPヘッダー + ペイロード + 擬似ヘッダーの整合性を検証。
  擬似ヘッダーには送信元IP・宛先IP・プロトコル番号・TCPセグメント長を含む。
  → IP層の情報もチェック対象に含めることで、誤配送を検出する。

■ 緊急ポインタ（16ビット）
  URGフラグがセットされている場合のみ有効。
  緊急データの末尾位置を示す。実際にはほとんど使用されない。
```

### 2.3 重要なTCPオプション

```
■ MSS（Maximum Segment Size）— Kind=2, Length=4
  1セグメントで送れるペイロードの最大サイズ。
  SYNパケットでのみ交渉される。
  通常の計算: MSS = MTU - IPヘッダ(20) - TCPヘッダ(20) = 1500 - 40 = 1460
  VPN/トンネル環境: MTUが小さくなるためMSSも小さくなる。

■ Window Scale — Kind=3, Length=3
  ウィンドウサイズフィールドを左シフトするビット数を指定。
  シフト値: 0〜14（最大スケールファクター = 2^14 = 16384）
  最大ウィンドウサイズ: 65535 × 16384 = 約1GB
  SYNパケットでのみ交渉される。

■ SACK Permitted — Kind=4, Length=2
  Selective ACK（SACK）の使用可否をSYNで交渉。
  SACKにより、連続していないデータブロックの受信状況を通知できる。
  → 不必要な再送を削減し、回復速度を向上させる。

■ SACK — Kind=5, Length=可変
  受信済みだが連続していないブロックの範囲（左端, 右端）を通知。
  最大4ブロックまで報告可能（オプション領域の制約）。

■ Timestamp — Kind=8, Length=10
  TSval: 送信側のタイムスタンプ
  TSecr: 受信したTSvalのエコー
  用途1: RTTの正確な測定（RTTM: Round-Trip Time Measurement）
  用途2: PAWS（Protection Against Wrapped Sequences）
         → シーケンス番号がラップアラウンドしても区別可能にする。
```

---

## 3. 3-wayハンドシェイクの詳細

### 3.1 接続確立の全体像

```
TCP接続の確立（3-way Handshake）:

  クライアント                              サーバー
  [CLOSED]                                 [LISTEN]
       │                                      │
       │── SYN ─────────────────────────────→ │
       │   seq=1000, win=65535                 │
       │   MSS=1460, WScale=7, SACK_OK        │
       │   [SYN_SENT]                          │
       │                                      │
       │ ←──────────────────── SYN-ACK ──────│
       │   seq=5000, ack=1001, win=65535       │
       │   MSS=1460, WScale=7, SACK_OK        │
       │                             [SYN_RCVD]│
       │                                      │
       │── ACK ─────────────────────────────→ │
       │   seq=1001, ack=5001, win=65535       │
       │   [ESTABLISHED]              [ESTABLISHED]
       │                                      │
       │ ←════════ データ転送開始 ═══════════→ │

  各ステップの詳細:

  [Step 1] SYN（クライアント → サーバー）
    - クライアントが接続を要求する
    - ISN（Initial Sequence Number）をランダムに選択（例: 1000）
    - TCPオプションで MSS, Window Scale, SACK, Timestamp を交渉
    - クライアントの状態: CLOSED → SYN_SENT

  [Step 2] SYN-ACK（サーバー → クライアント）
    - サーバーが接続を許可する
    - サーバー側のISNをランダムに選択（例: 5000）
    - ack = クライアントのISN + 1（1001）で「次に1001を期待する」と通知
    - サーバーの状態: LISTEN → SYN_RCVD

  [Step 3] ACK（クライアント → サーバー）
    - クライアントがサーバーのSYNを確認する
    - ack = サーバーのISN + 1（5001）
    - このACKにデータを含めることも可能（ピギーバック）
    - 双方の状態: ESTABLISHED

  所要時間: 1.5 RTT（Round Trip Time）
    東京 ↔ 大阪:     約 5ms  → 接続確立 約 7.5ms
    東京 ↔ US西海岸:  約100ms → 接続確立 約150ms
    東京 ↔ EU:       約250ms → 接続確立 約375ms
```

### 3.2 なぜ3-wayなのか

```
2-wayハンドシェイクの問題:

  シナリオ: 古いSYNパケットが遅延して到着する場合

  クライアント                         サーバー
       │                                 │
       │── SYN(seq=100) ──→ [遅延]       │  古い接続試行
       │── SYN(seq=200) ─────────────→   │  新しい接続試行
       │                                 │
       │←─── ACK(ack=201) ─────────── │  新しいSYNへの応答
       │   [接続確立]                      │  [接続確立]
       │                                 │
       │            [遅延したSYN到着]       │
       │              SYN(seq=100) ──→   │  古いSYN到着!
       │                                 │
       │←─── ACK(ack=101) ─────────── │  古いSYNへの応答
       │                                 │  [偽の接続確立!!]

  → 2-wayでは古いSYNに対して誤った接続を確立してしまう
  → 3-wayならクライアントが最後のACKで「正しい接続か」を確認できる
  → 古いSYN-ACKを受け取ったクライアントはRSTを返して拒否する

4-wayにしない理由:
  → 3-wayで十分に双方のISNを同期できる
  → ステップ数を増やすと接続確立の遅延が増える
  → セキュリティ上の利点もない
```

### 3.3 TCP Fast Open（TFO）

```
TCP Fast Open（RFC 7413）:

  通常のTCP:  SYN → SYN-ACK → ACK → データ = 2 RTT で最初のデータ到達
  TFO:       SYN+データ → SYN-ACK+データ = 1 RTT で最初のデータ到達

  仕組み:
  [初回接続]
    クライアント                        サーバー
         │── SYN + TFO Cookie要求 ──→  │
         │←── SYN-ACK + TFO Cookie ── │
         │── ACK ──→                   │
         │  (通常の3-way handshake)      │

  [2回目以降]
    クライアント                        サーバー
         │── SYN + Cookie + データ ──→  │  ★ SYNにデータを含める
         │                              │  サーバーはCookieを検証
         │                              │  → 有効ならデータを即処理
         │←── SYN-ACK + データ ──────  │  ★ 応答もすぐ返せる
         │── ACK ──→                   │
         │                              │

  利点:
  - 接続確立とデータ送信を同時に行える
  - Webページの初期表示時間を短縮（特にDNSやAPI呼び出し）

  制限事項:
  - 冪等なリクエスト（GET等）にのみ適用すべき
  - 初回接続ではCookie取得のため通常の3-wayが必要
  - 一部のミドルボックス（ファイアウォール等）で問題が発生する場合がある
```

### 3.4 コード例1: tcpdumpで3-wayハンドシェイクを観察する

```bash
# ターミナル1: tcpdumpでキャプチャ開始
# -i any: 全インターフェースを監視
# -nn: ホスト名・ポート番号を解決しない
# -S: シーケンス番号を絶対値で表示（相対値ではなく）
# port 80: HTTPポートのみフィルタ
sudo tcpdump -i any -nn -S port 80

# ターミナル2: HTTP接続を発生させる
curl http://example.com

# tcpdump の出力例:
# [Step 1] SYN
# 14:23:01.123456 IP 192.168.1.100.54321 > 93.184.216.34.80:
#   Flags [S], seq 2847291038, win 65535,
#   options [mss 1460,nop,wscale 6,nop,nop,TS val 123456 ecr 0,
#   sackOK,eol], length 0

# [Step 2] SYN-ACK
# 14:23:01.223456 IP 93.184.216.34.80 > 192.168.1.100.54321:
#   Flags [S.], seq 1428573920, ack 2847291039, win 65535,
#   options [mss 1460,nop,wscale 7,nop,nop,TS val 789012 ecr 123456,
#   sackOK,eol], length 0

# [Step 3] ACK
# 14:23:01.223567 IP 192.168.1.100.54321 > 93.184.216.34.80:
#   Flags [.], seq 2847291039, ack 1428573921, win 1024,
#   options [nop,nop,TS val 123457 ecr 789012], length 0

# フラグの読み方:
#   [S]  = SYN
#   [S.] = SYN-ACK（SYN + ACK）
#   [.]  = ACK
#   [P.] = PSH + ACK（データ送信）
#   [F.] = FIN + ACK（接続終了）
#   [R]  = RST（リセット）
#   [R.] = RST + ACK

# より詳細な表示:
# -X: パケット内容を16進数 + ASCII で表示
# -v: 詳細表示（TTL, ID, フラグメント情報等を含む）
sudo tcpdump -i any -nn -S -X -v port 80

# 特定ホストとの通信のみ表示:
sudo tcpdump -i any -nn -S host 93.184.216.34 and port 80

# SYNパケットのみフィルタ（ポートスキャン検出に有用）:
sudo tcpdump -i any -nn 'tcp[tcpflags] & (tcp-syn) != 0'

# pcapファイルに保存して後でWiresharkで分析:
sudo tcpdump -i any -nn -w /tmp/tcp_capture.pcap port 80
```

### 3.5 TCP状態遷移図

```
TCP状態遷移の全体像:

                              ┌──────────┐
                              │  CLOSED  │
                              └────┬─────┘
                       ┌──────────┤├──────────────┐
                 パッシブオープン  ││  アクティブオープン  │
                  (listen())     ││   (connect())      │
                       │         ││         │          │
                       ▼         ││         ▼          │
                  ┌────────┐    ││   ┌──────────┐    │
                  │ LISTEN │    ││   │ SYN_SENT │    │
                  └───┬────┘    ││   └────┬─────┘    │
             SYN受信  │         ││        │ SYN-ACK   │
            SYN-ACK送信│        ││        │ 受信       │
                       │         ││        │ ACK送信    │
                       ▼         ││        ▼          │
                  ┌──────────┐  ││  ┌─────────────┐  │
                  │ SYN_RCVD │──┘│  │ ESTABLISHED │  │
                  └────┬─────┘   │  └──────┬──────┘  │
               ACK受信 │          │         │         │
                       ▼          │   close() │        │
                  ┌─────────────┐│    FIN送信 │        │
                  │ ESTABLISHED ││         ▼         │
                  └──────┬──────┘│  ┌──────────┐     │
                   close()│       │  │ FIN_WAIT1│     │
                   FIN送信 │       │  └────┬─────┘     │
                          │       │  ACK受信│  FIN+ACK  │
                          │       │        ▼  受信     │
                          │       │  ┌──────────┐     │
                          │       │  │ FIN_WAIT2│     │
                          │       │  └────┬─────┘     │
                   FIN受信 │       │  FIN受信│          │
                   ACK送信 │       │  ACK送信│          │
                          ▼       │        ▼          │
                  ┌──────────┐   │  ┌──────────┐     │
                  │CLOSE_WAIT│   │  │TIME_WAIT │     │
                  └────┬─────┘   │  └────┬─────┘     │
               close() │         │  2MSL待機│          │
               FIN送信  │         │        ▼          │
                        ▼         │  ┌──────────┐     │
                  ┌──────────┐   │  │  CLOSED  │     │
                  │ LAST_ACK │   │  └──────────┘     │
                  └────┬─────┘   │                    │
               ACK受信 │          │                    │
                        ▼         │                    │
                  ┌──────────┐   │                    │
                  │  CLOSED  │   │                    │
                  └──────────┘   │                    │
                                  └────────────────────┘

  各状態の説明:
    CLOSED:      初期状態 / 最終状態
    LISTEN:      接続待機中（サーバー）
    SYN_SENT:    SYN送信済み、SYN-ACK待ち（クライアント）
    SYN_RCVD:    SYN-ACK送信済み、ACK待ち（サーバー）
    ESTABLISHED: 接続確立済み、データ送受信可能
    FIN_WAIT_1:  FIN送信済み、ACK待ち
    FIN_WAIT_2:  FINのACK受信済み、相手のFIN待ち
    TIME_WAIT:   相手のFIN受信済み、2MSL待機中
    CLOSE_WAIT:  相手のFIN受信済み、自分のclose()待ち
    LAST_ACK:    FIN送信済み、最後のACK待ち
    CLOSING:     同時クローズ時の特殊状態
```

---

## 4. データ転送メカニズム

### 4.1 シーケンス番号とACKの動作

```
データ転送の基本フロー:

  クライアント                           サーバー
  [ESTABLISHED]                         [ESTABLISHED]
       │                                    │
       │── DATA (seq=1001, len=500) ──────→│  500バイトのデータ送信
       │                                    │
       │←──── ACK (ack=1501) ─────────── │  「次は1501を期待する」
       │                                    │
       │── DATA (seq=1501, len=500) ──────→│  次の500バイト
       │                                    │
       │←──── ACK (ack=2001) ─────────── │  「次は2001を期待する」
       │                                    │

  ポイント:
  - seq: このセグメントの先頭バイトの番号
  - len: ペイロードのバイト数
  - ack: 次に受信を期待するバイト番号 = seq + len
  - ACKは累積的: ack=2001 は「2000バイト目までは全て受信済み」を意味する
```

### 4.2 再送メカニズムの詳細

```
■ タイムアウト再送（RTO: Retransmission Timeout）

  RTOの計算（RFC 6298）:
    SRTT = Smoothed RTT（RTTの指数移動平均）
    RTTVAR = RTTの分散
    RTO = SRTT + max(G, 4 × RTTVAR)
      ※ G = クロック粒度（通常1ms）

  初期値:
    RTO = 1秒（RFC 6298推奨）
    最小RTO = 1秒（RFC推奨）、Linux実装では200ms
    最大RTO = 60秒（Linux実装）

  バックオフ:
    タイムアウト発生ごとに RTO = RTO × 2（指数バックオフ）
    最大再送回数: Linuxデフォルトは15回（tcp_retries2）

■ 高速再送（Fast Retransmit）

  クライアント                           サーバー
       │── seq=1, len=100 ──→             │  ✓ 受信
       │── seq=101, len=100 ──→  ✗ ロスト │
       │── seq=201, len=100 ──→           │  受信するが穴がある
       │←─ ack=101 ──────────────────── │  DupACK #1
       │── seq=301, len=100 ──→           │  受信するが穴がある
       │←─ ack=101 ──────────────────── │  DupACK #2
       │── seq=401, len=100 ──→           │  受信するが穴がある
       │←─ ack=101 ──────────────────── │  DupACK #3 → 高速再送!
       │                                    │
       │── seq=101, len=100 ──→（再送）    │  ✓ 再送成功
       │←─ ack=501 ──────────────────── │  穴が埋まり全て受信済み

  条件: 3つの重複ACK（Duplicate ACK）を受信した時点で即座に再送
  利点: RTO を待たずに素早くロストセグメントを回復できる
  前提: パケットの順序入れ替え（reordering）と区別する必要がある

■ SACK（Selective ACK）による効率的な再送

  SACKがない場合:
    受信側は累積ACKしか返せないため、どのセグメントがロストしたか
    送信側は正確に知ることができない → 不必要な再送が発生する

  SACKがある場合:
    受信側が「受信済みブロック」の範囲を明示的に通知する

    例: seq=101 がロスト、seq=201〜500 は受信済み
    ACK: ack=101, SACK=[201-301, 301-401, 401-501]
    → 送信側は seq=101 だけを再送すればよいと判断できる
```

### 4.3 コード例2: PythonによるTCPソケットプログラミング

```python
#!/usr/bin/env python3
"""
TCP エコーサーバーとクライアントの実装例
ソケットプログラミングの基本パターンを示す
"""

import socket
import threading
import time
import struct

# ============================================================
# TCPエコーサーバー
# ============================================================
class TCPEchoServer:
    """
    マルチスレッド対応のTCPエコーサーバー。
    受信したデータをそのまま送り返す。
    """

    def __init__(self, host='127.0.0.1', port=8080, backlog=5):
        self.host = host
        self.port = port
        self.backlog = backlog
        self.server_socket = None
        self.running = False

    def start(self):
        """サーバーを起動する"""
        # AF_INET: IPv4, SOCK_STREAM: TCP
        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        # SO_REUSEADDR: TIME_WAIT 状態のポートを再利用可能にする
        # サーバー再起動時に "Address already in use" エラーを回避
        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        # TCP_NODELAY: Nagle アルゴリズムを無効化
        # 小さなパケットを即座に送信する（低遅延が必要な場合）
        self.server_socket.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )

        # バインドとリッスン
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.backlog)
        self.running = True
        print(f"[SERVER] Listening on {self.host}:{self.port}")
        print(f"[SERVER] Backlog (SYN Queue + Accept Queue): {self.backlog}")

        # 接続ソケットのオプションを表示
        self._print_socket_options()

        while self.running:
            try:
                # accept() は3-wayハンドシェイク完了済みの接続を取り出す
                # ブロッキング呼び出し: 接続が来るまで待機する
                client_socket, client_addr = self.server_socket.accept()
                print(f"[SERVER] Connection from {client_addr}")

                # クライアントごとにスレッドを作成
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_addr),
                    daemon=True
                )
                thread.start()

            except OSError:
                break

    def _handle_client(self, client_socket, client_addr):
        """個々のクライアント接続を処理する"""
        try:
            # TCP Keep-Alive を有効にする
            client_socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_KEEPALIVE,
                1
            )

            while True:
                # recv() はTCPの受信バッファからデータを読み取る
                # 4096: 一度に読み取る最大バイト数
                # TCPはバイトストリームなので、送信側の send() 呼び出しと
                # 受信側の recv() 呼び出しは 1対1 に対応しない
                data = client_socket.recv(4096)

                if not data:
                    # 相手が接続を閉じた（FINを受信）
                    print(f"[SERVER] {client_addr} disconnected")
                    break

                print(f"[SERVER] Received {len(data)} bytes from {client_addr}")
                # エコーバック: 受信データをそのまま返す
                client_socket.sendall(data)

        except ConnectionResetError:
            # 相手がRSTを送信した場合
            print(f"[SERVER] Connection reset by {client_addr}")
        except BrokenPipeError:
            # 切断済みの接続に書き込もうとした場合
            print(f"[SERVER] Broken pipe for {client_addr}")
        finally:
            client_socket.close()

    def _print_socket_options(self):
        """ソケットオプションの現在値を表示する"""
        sock = self.server_socket
        print(f"[SERVER] SO_REUSEADDR: "
              f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)}")
        print(f"[SERVER] SO_RCVBUF (受信バッファ): "
              f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)} bytes")
        print(f"[SERVER] SO_SNDBUF (送信バッファ): "
              f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)} bytes")

    def stop(self):
        """サーバーを停止する"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


# ============================================================
# TCPクライアント
# ============================================================
class TCPEchoClient:
    """TCPエコークライアント"""

    def __init__(self, host='127.0.0.1', port=8080, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_and_receive(self, message: str) -> str:
        """メッセージを送信し、エコーバックを受信する"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # connect() で3-wayハンドシェイクが実行される
            # タイムアウトを設定して無限待機を防止
            sock.settimeout(self.timeout)

            start_time = time.time()
            sock.connect((self.host, self.port))
            connect_time = time.time() - start_time
            print(f"[CLIENT] Connected in {connect_time*1000:.2f}ms "
                  f"(≈ 1.5 RTT)")

            # データ送信
            data = message.encode('utf-8')
            sock.sendall(data)
            print(f"[CLIENT] Sent {len(data)} bytes")

            # エコーバック受信
            response = sock.recv(4096)
            rtt = time.time() - start_time
            print(f"[CLIENT] Received {len(response)} bytes "
                  f"(total RTT: {rtt*1000:.2f}ms)")

            return response.decode('utf-8')
            # with ブロック終了時に close() が呼ばれる
            # → 4-way handshake でTCP接続が切断される


# ============================================================
# 使用例
# ============================================================
if __name__ == '__main__':
    # サーバーをバックグラウンドスレッドで起動
    server = TCPEchoServer()
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)  # サーバー起動待ち

    # クライアントからメッセージを送信
    client = TCPEchoClient()
    response = client.send_and_receive("Hello, TCP!")
    print(f"[RESULT] Echo response: {response}")

    server.stop()
```

---

## 5. フロー制御（Flow Control）の詳細

### 5.1 スライディングウィンドウの動作原理

```
フロー制御 = 受信側のバッファ溢れを防ぐメカニズム

■ スライディングウィンドウの概念

  送信側のバッファ状態:
  ┌─────────────────────────────────────────────────────────────────┐
  │ ACK済み │ 送信済み未ACK │  送信可能  │     送信不可           │
  │ (解放)  │  (in-flight)  │ (ウィンドウ)│  (ウィンドウ外)       │
  └─────────────────────────────────────────────────────────────────┘
             ↑               ↑            ↑
          SND.UNA          SND.NXT     SND.UNA+SND.WND

  SND.UNA:   最古の未確認シーケンス番号（Unacknowledged）
  SND.NXT:   次に送信するシーケンス番号
  SND.WND:   送信ウィンドウサイズ（= min(rwnd, cwnd)）

  送信可能量 = SND.WND - (SND.NXT - SND.UNA)
             = ウィンドウサイズ - インフライト量

  受信側のバッファ状態:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 処理済み  │  受信済み未処理  │     空き領域（rwnd）             │
  │ (解放)   │  (バッファ内)     │     ← ウィンドウサイズ →        │
  └─────────────────────────────────────────────────────────────────┘
              ↑                  ↑                                 ↑
           RCV.NXT            RCV.NXT              RCV.NXT + RCV.WND
          (次に期待する        + 受信済み
           シーケンス番号)

■ ウィンドウの変化（具体例）

  受信バッファサイズ = 10KB、アプリケーション処理速度 = 2KB/s の場合:

  時刻0: rwnd = 10KB  →  「10KBまで送ってOK」
         ┌─────────────────────────────────┐
         │             空き: 10KB           │
         └─────────────────────────────────┘

  時刻1: 送信側が 4KB 送信 → rwnd = 6KB
         ┌──────────┬──────────────────────┐
         │受信済 4KB │     空き: 6KB        │
         └──────────┴──────────────────────┘

  時刻2: さらに 4KB 送信 → rwnd = 2KB
         ┌──────────────────────┬──────────┐
         │  受信済 8KB          │空き: 2KB │
         └──────────────────────┴──────────┘

  時刻3: アプリが 6KB 処理 → rwnd = 8KB（Window Update送信）
         ┌──────────┬──────────────────────────────┐
         │残り 2KB  │          空き: 8KB            │
         └──────────┴──────────────────────────────┘
```

### 5.2 ゼロウィンドウとSilly Window Syndrome

```
■ ゼロウィンドウ（Zero Window）

  受信バッファが一杯になると rwnd = 0 を通知 → 送信停止

  クライアント                           サーバー
       │── DATA ──→                       │
       │←── ACK (rwnd=0) ───────────── │  バッファ満杯!
       │                                    │
       │  [送信停止]                          │
       │                                    │
       │── Zero Window Probe ──→           │  「まだ rwnd=0 ?」
       │←── ACK (rwnd=0) ───────────── │  「まだ一杯」
       │                                    │
       │   ... 待機 ...                      │
       │                                    │
       │── Zero Window Probe ──→           │  再度確認
       │←── ACK (rwnd=4096) ──────────  │  「空きができた!」
       │                                    │
       │── DATA ──→                       │  送信再開
       │                                    │

  Zero Window Probe:
  - 送信側が定期的に1バイトのプローブを送信する
  - 受信側が rwnd > 0 を返すまで繰り返す
  - プローブ間隔はRTOに基づき指数バックオフする
  - Linuxでは tcp_probe_interval（デフォルト:75秒）で設定可能

■ Silly Window Syndrome（SWS）

  問題: 受信側が極小のウィンドウ（数バイト）を通知
       → 送信側が極小セグメントを送信
       → ヘッダーのオーバーヘッド比率が極端に高くなる

  例: 1バイトのペイロード + 40バイトのヘッダー
      → 効率 = 1/41 = 2.4%（大半がオーバーヘッド）

  対策（受信側 - Clark のアルゴリズム）:
  - rwnd が MSS 以上、またはバッファの 50% 以上空くまで
    ウィンドウ更新を通知しない
  - 小さなウィンドウ更新を抑制する

  対策（送信側 - Nagle アルゴリズム）:
  → 次のセクションで詳述
```

### 5.3 Nagleアルゴリズム

```
■ Nagle アルゴリズム（RFC 896）

  目的: 小さなパケットの大量送信（Tinygram Problem）を防ぐ

  ルール:
  if (未確認データが存在する) {
      送信データをバッファに蓄積する
      ACKが返るか、MSS分溜まったら送信する
  } else {
      即座に送信する
  }

  効果:
  - キーストロークを1文字ずつ送信するような場合、
    複数文字を1セグメントにまとめて送信する
  - ネットワーク上の小パケットを削減する

  問題:
  - リアルタイム性が求められるアプリケーションでは遅延が発生する
  - 特にTelnet, SSH, ゲームのマウス入力、APIレスポンスなど
  - Delayed ACK と組み合わさると最大 200ms の遅延が発生する

  無効化:
  - TCP_NODELAY ソケットオプションを設定する
  - 低遅延を優先する場合に使用する
```

