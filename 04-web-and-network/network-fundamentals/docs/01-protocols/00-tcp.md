# TCP（Transmission Control Protocol）

> TCPは信頼性のある通信を実現するプロトコル。3-wayハンドシェイク、シーケンス番号、フロー制御、輻輳制御の仕組みを理解し、なぜWebの通信基盤なのかを学ぶ。本ガイドでは、RFC 9293 に基づく最新のTCP仕様を網羅的に解説し、tcpdump・Wireshark・ソケットプログラミングを用いた実践的な分析手法を身につける。

## 前提知識

このガイドを最大限に活用するには、以下の知識が必要です。

**必須**

**推奨**
- ネットワークインターフェース（Ethernet、Wi-Fi）の基本的な理解
- コマンドライン操作の基礎知識（tcpdump、Wireshark等のツールを使用）

---

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

### 5.4 コード例3: ソケットオプションの確認と設定

```python
#!/usr/bin/env python3
"""
TCP ソケットオプションの確認と設定
フロー制御・バッファサイズ関連のパラメータを操作する
"""

import socket
import sys

def inspect_tcp_socket_options():
    """TCPソケットの主要オプションを確認する"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("=" * 60)
    print("TCP ソケットオプション一覧")
    print("=" * 60)

    # ── 汎用ソケットオプション ──
    print("\n■ 汎用ソケットオプション (SOL_SOCKET)")
    print(f"  SO_RCVBUF   (受信バッファサイズ): "
          f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF):,} bytes")
    print(f"  SO_SNDBUF   (送信バッファサイズ): "
          f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF):,} bytes")
    print(f"  SO_REUSEADDR(アドレス再利用)    : "
          f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)}")
    print(f"  SO_KEEPALIVE(キープアライブ)    : "
          f"{sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)}")

    # ── TCP固有オプション ──
    print("\n■ TCP固有オプション (IPPROTO_TCP)")
    print(f"  TCP_NODELAY (Nagle無効化)       : "
          f"{sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)}")

    # Linux固有オプション（macOSでは一部未対応）
    if sys.platform == 'linux':
        print(f"  TCP_MAXSEG  (MSS)               : "
              f"{sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG)}")
        print(f"  TCP_WINDOW_CLAMP(最大ウィンドウ) : "
              f"{sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_WINDOW_CLAMP)}")

    # ── バッファサイズの変更例 ──
    print("\n■ バッファサイズ変更")
    new_rcvbuf = 256 * 1024  # 256KB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, new_rcvbuf)
    actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"  設定値: {new_rcvbuf:,} bytes → 実際の値: {actual:,} bytes")
    print(f"  ※ カーネルが設定値を2倍にする実装がある (Linux)")

    # ── Keep-Alive の詳細設定（Linux） ──
    if sys.platform == 'linux':
        print("\n■ TCP Keep-Alive 設定 (Linux)")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Keep-Aliveの開始までの時間（秒）
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        # Keep-Aliveプローブの間隔（秒）
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        # Keep-Aliveプローブの最大回数
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        print(f"  TCP_KEEPIDLE  : 60秒（最初のプローブまでの待機時間）")
        print(f"  TCP_KEEPINTVL : 10秒（プローブ間隔）")
        print(f"  TCP_KEEPCNT   :  5回（最大プローブ回数）")
        print(f"  → 60 + 10 × 5 = 110秒で接続断判定")

    sock.close()

if __name__ == '__main__':
    inspect_tcp_socket_options()
```

---

## 6. 輻輳制御（Congestion Control）の詳細

### 6.1 輻輳制御の目的とフロー制御との違い

```
■ フロー制御 vs 輻輳制御

  フロー制御（Flow Control）:
  - 目的: 受信側のバッファオーバーフローを防ぐ
  - 制御対象: エンドツーエンド（送信者 ←→ 受信者）
  - 制御変数: rwnd（受信ウィンドウ）
  - 通知方法: TCPヘッダーのウィンドウサイズフィールド
  - 受信側が明示的に通知する

  輻輳制御（Congestion Control）:
  - 目的: ネットワーク全体の混雑を回避する
  - 制御対象: ネットワーク全体（ルーター、スイッチの負荷）
  - 制御変数: cwnd（輻輳ウィンドウ）
  - 推測方法: パケットロス、RTTの変動から推測する
  - 送信側が暗黙的に推測する

  実際の送信ウィンドウ:
  effective_window = min(rwnd, cwnd)
  → フロー制御と輻輳制御の両方を満たす範囲でのみ送信する
```

### 6.2 輻輳制御の4フェーズ

```
■ 輻輳制御アルゴリズム（TCP Reno）の詳細

[Phase 1] スロースタート（Slow Start）
  目的: ネットワークの利用可能帯域を素早く見つける
  動作:
  - 初期値: cwnd = 1 MSS（または IW = 10 MSS: RFC 6928）
  - ACKを受信するたびに cwnd += 1 MSS
  - 1 RTT あたり cwnd が2倍になる（指数的増加）
  - ssthresh（スロースタート閾値）に達したら Phase 2 へ
  - パケットロスを検出したら Phase 3/4 へ

  cwnd の変化（RTTごと）:
    RTT 0:  cwnd = 1 MSS  → 1セグメント送信
    RTT 1:  cwnd = 2 MSS  → 2セグメント送信
    RTT 2:  cwnd = 4 MSS  → 4セグメント送信
    RTT 3:  cwnd = 8 MSS  → 8セグメント送信
    RTT 4:  cwnd = 16 MSS → 16セグメント送信
    ...
    → 約 log2(N) RTT で N MSS に到達する

[Phase 2] 輻輳回避（Congestion Avoidance）
  目的: 輻輳を起こさないよう慎重にレートを上げる
  動作:
  - cwnd >= ssthresh のとき適用
  - ACKを受信するたびに cwnd += MSS × (MSS / cwnd)
  - 1 RTT あたり cwnd が約1 MSS増える（線形増加: AIMD の AI 部分）

[Phase 3] 高速再送（Fast Retransmit）
  トリガー: 3つの重複ACK（Duplicate ACK）を受信
  動作:
  - タイムアウトを待たずに即座にロストセグメントを再送
  - ssthresh = cwnd / 2 に設定

[Phase 4] 高速回復（Fast Recovery）
  目的: 輻輳後の回復を高速化する
  動作（TCP Reno）:
  - ssthresh = cwnd / 2
  - cwnd = ssthresh + 3 MSS（3つのDupACK分）
  - さらにDupACKを受信するたびに cwnd += 1 MSS
  - 新しいACK（元のデータのACK）を受信したら cwnd = ssthresh
  - 輻輳回避フェーズに移行

  タイムアウト発生時（最も深刻な輻輳シグナル）:
  - ssthresh = cwnd / 2
  - cwnd = 1 MSS（スロースタートに戻る）
  - Phase 1 から再開
```

### 6.3 輻輳制御アルゴリズムの推移図

```
cwnd の変化を時系列で表示:

cwnd (MSS)
  ^
  |
32|                          *
  |                        * | パケットロス（DupACK×3）
  |                      *   |
  |                    *     | ssthresh = 32/2 = 16
  |                  *       |
  |               *          ↓
16|─ ─ ─ ─ ─ ─*─ ─ ─ ─ ─ ─ ─ ─ ─ ─ * ─ ─ ─ ─ ─ ─ ─ ─ *
  |           *  (1)スロースタート     *                   *
  |         *    (指数増加)          *  (2)輻輳回避        *
  |       *                        *    (線形増加)        *
  |     *                        *                      *
  |   *                        *                       *
  |  *                       *                  タイムアウト!
  | *                      *                     ↓ cwnd=1
1 |*─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─*
  |                                              | (1)再度
  +──────────────────────────────────────────────→ 時間 (RTT)

  凡例:
  --- Phase 1: スロースタート（指数増加）
  --- Phase 2: 輻輳回避（線形増加: AIMD）
  ↓   Phase 3+4: 高速再送 + 高速回復（cwnd半減）
  ↓↓  タイムアウト: cwnd = 1（スロースタートに戻る）

  AIMD = Additive Increase, Multiplicative Decrease
    Additive Increase:    1 RTTごとに cwnd += 1 MSS
    Multiplicative Decrease: パケットロス時に cwnd = cwnd / 2
    → ネットワーク公平性を確保するための重要な特性
```

### 6.4 主要な輻輳制御アルゴリズム比較

| 特性 | TCP Reno | TCP CUBIC | BBR |
|------|----------|-----------|-----|
| 年代 | 1990年 | 2006年 | 2016年 |
| RFC | RFC 5681 | RFC 9438 | 実験的 |
| ロス検知方式 | パケットロスベース | パケットロスベース | 帯域×遅延モデル |
| cwnd増加方式 | AIMD（線形） | 3次関数（BIC曲線） | BDP推定に基づく |
| 高帯域×高遅延 | 性能低い | 性能良好 | 非常に良好 |
| 公平性 | 基準 | Renoと概ね公平 | 異種アルゴリズムとの公平性に課題 |
| OS採用 | BSD, 旧Linux | Linux標準（3.x〜） | Google内部, Linux 4.9+ |
| 適用環境 | 低遅延LAN | 一般的なインターネット | 高BDP環境, 動画配信 |
| バッファ溢れ耐性 | 弱い | 中程度 | 浅いバッファに適応 |

### 6.5 CUBIC アルゴリズムの詳細

```
■ TCP CUBIC（RFC 9438）

  Linuxのデフォルト輻輳制御アルゴリズム（カーネル 2.6.19 以降）

  核心アイデア:
  - cwndの増加を時間ベースの3次関数（cubic function）で制御する
  - ロス発生前のcwndに素早く戻り、その付近で慎重に探索する

  3次関数:
    W(t) = C * (t - K)^3 + W_max

    C:     定数（0.4）
    t:     最後のパケットロスからの経過時間
    K:     W(t) = W_max / 2 から W_max に到達する推定時間
           K = (W_max * beta / C) ^ (1/3)
    W_max: ロス発生前のcwnd
    beta:  削減係数（0.7: Renoの0.5より緩やか）

  cwnd の変化パターン:
  cwnd
   ^
   |        W_max
   |  ------*-----------------*---- <- ロス前のcwnd
   |       /|*               *|
   |      / |  *           *  |
   |     /  |    *       *    |
   |    /   |      * * *      |    <- 凸部: 慎重な探索
   |   /    |    凹部: 急速回復|
   |  /     |                 |
   | /      |                 |
   |/       |                 |
   +--------+-----------------+---> 時間
         ロス発生            新たなロス

  CUBICの利点:
  1. 高BDP（Bandwidth-Delay Product）環境に適応
     - Renoは線形増加のため、高帯域リンクの活用に時間がかかる
     - CUBICは3次関数で素早くW_maxに接近する
  2. RTTに対する公平性
     - Renoは短いRTTの接続が有利（ACKが速く返るため）
     - CUBICは時間ベースのため、RTTの影響が小さい
```

### 6.6 BBR（Bottleneck Bandwidth and Round-trip propagation time）

```
■ BBR（Google, 2016年〜）

  従来のアプローチ: パケットロスを輻輳のシグナルとして利用
  BBRのアプローチ:  ネットワークの物理特性（帯域と遅延）を直接推定

  2つの指標:
    BtlBw:  ボトルネック帯域幅（最大スループット）
    RTprop: 最小RTT（伝搬遅延のみ、キューイング遅延を除く）

    最適動作点 = BtlBw * RTprop（BDP: Bandwidth-Delay Product）

  状態マシン:
  +-----------------------------------------------------+
  |                                                     |
  |  [STARTUP]                                          |
  |    cwnd を指数的に増加                                 |
  |    BtlBw が3RTT連続で増加しなくなるまで                |
  |         |                                           |
  |         v                                           |
  |  [DRAIN]                                            |
  |    過剰にバッファに溜めたデータを排出                  |
  |    inflight を BDP まで減少させる                      |
  |         |                                           |
  |         v                                           |
  |  [PROBE_BW]   <- 大部分の時間をここで過ごす            |
  |    BtlBwを定期的に探索する                            |
  |    8RTTサイクル: 1.25 -> 0.75 -> 1.0 x 6             |
  |    （帯域増加を探索 -> 過剰分を排出 -> 安定運転）      |
  |         |                                           |
  |         v（200ms以上RTpropが更新されない場合）         |
  |  [PROBE_RTT]                                        |
  |    cwnd = 4 MSS に一時的に縮小                        |
  |    キューを空にして最小RTTを再測定                      |
  |    200ms後にPROBE_BWに戻る                           |
  |                                                     |
  +-----------------------------------------------------+

  BBRの利点:
  1. バッファ肥大化（Bufferbloat）を回避
     - キューを溜めずに帯域を最大活用する
  2. パケットロスに過剰反応しない
     - ランダムロスでcwndを半減しない
  3. 高BDP環境で優れた性能
     - 大陸間通信やデータセンター間で効果的

  BBRの課題:
  1. 公平性問題
     - CUBICとの共存時にBBRが帯域を過剰に占有する場合がある
  2. 高パケットロス環境での性能
     - ロスを輻輳シグナルと見なさないため、実際の輻輳を見逃す可能性
  3. BBRv2 で改善中
     - ECN（Explicit Congestion Notification）の活用
     - パケットロスにも一定の反応を行う
```

### 6.7 コード例4: Linuxでの輻輳制御アルゴリズムの確認と変更

```bash
# ── 現在の輻輳制御アルゴリズムを確認 ──
sysctl net.ipv4.tcp_congestion_control
# 出力例: net.ipv4.tcp_congestion_control = cubic

# ── 利用可能なアルゴリズム一覧 ──
sysctl net.ipv4.tcp_available_congestion_control
# 出力例: net.ipv4.tcp_available_congestion_control = reno cubic

# ── BBR モジュールのロードと有効化 ──
sudo modprobe tcp_bbr
echo "tcp_bbr" | sudo tee -a /etc/modules-load.d/modules.conf

# BBR を有効化
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
# 確認
sysctl net.ipv4.tcp_congestion_control
# 出力: net.ipv4.tcp_congestion_control = bbr

# ── 永続化（/etc/sysctl.conf に追記） ──
echo "net.ipv4.tcp_congestion_control=bbr" | sudo tee -a /etc/sysctl.conf
echo "net.core.default_qdisc=fq" | sudo tee -a /etc/sysctl.conf
# ※ BBRは fq (Fair Queue) qdisc との組み合わせが推奨される

# ── TCP関連のカーネルパラメータ一覧 ──
sysctl -a | grep "^net.ipv4.tcp"

# 重要なパラメータ:
# net.ipv4.tcp_rmem = 4096 131072 6291456
#   → 受信バッファ: 最小 / デフォルト / 最大
# net.ipv4.tcp_wmem = 4096 16384 4194304
#   → 送信バッファ: 最小 / デフォルト / 最大
# net.ipv4.tcp_window_scaling = 1
#   → Window Scale オプションの有効/無効
# net.ipv4.tcp_sack = 1
#   → SACK の有効/無効
# net.ipv4.tcp_timestamps = 1
#   → Timestamp オプションの有効/無効
# net.ipv4.tcp_max_syn_backlog = 4096
#   → SYN キューの最大サイズ
# net.ipv4.tcp_fin_timeout = 60
#   → FIN_WAIT_2 状態のタイムアウト
# net.ipv4.tcp_tw_reuse = 2
#   → TIME_WAIT ソケットの再利用

# ── 接続ごとに輻輳制御を指定する（Python） ──
# import socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION,
#                 b'bbr')
```

### 6.8 ECN（Explicit Congestion Notification）

```
■ ECN（RFC 3168）

  従来: パケットロスで輻輳を検知 → 遅い、データ損失を伴う
  ECN:  ルーターが「輻輳しているよ」と明示的にマーク → ロス前に対処可能

  仕組み:
  1. 送信側と受信側がTCPハンドシェイクでECN対応を交渉
     （SYNに ECE + CWR フラグを設定）
  2. ルーターがキューが溢れそうになったら、
     IPヘッダーのECNフィールド（2ビット）にCE（Congestion Experienced）をマーク
  3. 受信側がCEマークを検出し、TCPのECEフラグで送信側に通知
  4. 送信側がcwndを縮小し、CWRフラグで受信側に「対処した」と通知

  IPヘッダーのECNフィールド:
    00: Non ECN-Capable Transport (Not-ECT)
    01: ECN Capable Transport (ECT(1))
    10: ECN Capable Transport (ECT(0))
    11: Congestion Experienced (CE)

  利点:
  - パケットを落とさずに輻輳を通知できる
  - 特にショートフロー（Webリクエスト等）で効果的
  - BBRv2 はECNを積極的に活用する

  現状:
  - 多くのOSで対応済み（デフォルト無効の場合が多い）
  - Apple（iOS/macOS）は積極的に有効化
  - 一部のミドルボックスがECNマークを除去する問題あり
```

---

## 7. TCP接続の切断（4-way Handshake）

### 7.1 正常な切断手順

```
TCP切断（4-way Handshake）:

  クライアント                         サーバー
  [ESTABLISHED]                       [ESTABLISHED]
       |                                  |
       |-- FIN (seq=1000) ------------>   |  (1) close() 呼び出し
       |   [FIN_WAIT_1]                    |
       |                                  |
       |<----- ACK (ack=1001) ---------- |  (2) FINのACK
       |   [FIN_WAIT_2]                    |  [CLOSE_WAIT]
       |                                  |
       |      （サーバーが残りのデータを送信）   |
       |                                  |
       |<----- FIN (seq=5000) ---------- |  (3) サーバーもclose()
       |                                  |  [LAST_ACK]
       |                                  |
       |-- ACK (ack=5001) ------------>   |  (4) FINのACK
       |   [TIME_WAIT]                     |  [CLOSED]
       |                                  |
       |   2MSL 待機                        |
       |   (通常 60秒〜120秒)               |
       |                                  |
       |   [CLOSED]                        |
       |                                  |

  なぜ4-wayか（3-wayにしない理由）:
  - 片方がFINを送っても、もう片方にはまだ送るデータがあるかもしれない
  - ハーフクローズ（半二重クローズ）を実現するために、
    各方向で独立にFIN-ACKが必要
  - 接続の確立（SYN + SYN-ACK）と異なり、切断のFINとACKは
    同時に送れない場合がある

  同時クローズ（Simultaneous Close）:
  - 双方が同時にFINを送信した場合の特殊なケース
  - 双方がFIN_WAIT_1 → CLOSING → TIME_WAIT → CLOSED と遷移
```

### 7.2 TIME_WAITの詳細と問題

```
■ TIME_WAIT 状態

  目的1: 最後のACKが失われた場合のリカバリ
    - 相手のFINに対するACKが失われた場合、
      相手はFINを再送する
    - TIME_WAIT中であれば再度ACKを返せる

  目的2: 古いセグメントの消滅を保証
    - 同じ4-tuple（src IP, src port, dst IP, dst port）で
      新しい接続を開始する前に、ネットワーク上の古いセグメントが
      全て消滅するのを待つ
    - MSL（Maximum Segment Lifetime）= パケットがネットワーク上で
      生存できる最大時間

  TIME_WAIT の待機時間:
    2 * MSL = 2 * 60秒 = 120秒（Linux）
    ※ MSLの値はOS実装により異なる

  問題: 大量のTIME_WAIT蓄積
  ─────────────────────────────────────────
  原因:
  - 高頻度の短寿命TCP接続（HTTPリクエスト等）
  - 各接続が120秒間TIME_WAIT状態を維持
  - 利用可能なエフェメラルポート（約16,000個）を圧迫

  確認コマンド:
  $ ss -tan state time-wait | wc -l
  $ netstat -an | grep TIME_WAIT | wc -l

  対策:
  1. SO_REUSEADDR: TIME_WAIT状態のアドレスを再利用
     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

  2. tcp_tw_reuse（Linux）: クライアント側でTIME_WAITを再利用
     sysctl -w net.ipv4.tcp_tw_reuse=1

  3. HTTP Keep-Alive: 接続を再利用して短寿命接続を削減
     Connection: keep-alive

  4. 接続プーリング: データベース接続等で接続を使い回す

  5. エフェメラルポート範囲の拡大:
     sysctl -w net.ipv4.ip_local_port_range="1024 65535"
```

---

## 8. コード例5: Wiresharkによるパケット解析

```
■ Wireshark でTCP通信を解析する手順

[Step 1] キャプチャの開始
  - Wireshark を起動
  - 適切なネットワークインターフェースを選択
  - キャプチャフィルタを設定: "tcp port 443"

[Step 2] フィルタ式の活用

  表示フィルタ（Display Filter）:
  ─────────────────────────────────────────────────────
  目的                    フィルタ式
  ─────────────────────────────────────────────────────
  SYNパケットのみ          tcp.flags.syn == 1 && tcp.flags.ack == 0
  SYN-ACKパケットのみ      tcp.flags.syn == 1 && tcp.flags.ack == 1
  RSTパケットのみ          tcp.flags.reset == 1
  FINパケットのみ          tcp.flags.fin == 1
  再送パケット             tcp.analysis.retransmission
  重複ACK                 tcp.analysis.duplicate_ack
  ゼロウィンドウ           tcp.analysis.zero_window
  ウィンドウ更新           tcp.analysis.window_update
  特定のストリーム         tcp.stream eq 5
  ペイロードあり           tcp.len > 0
  特定ポート              tcp.port == 80
  特定のフラグ組み合わせ    tcp.flags == 0x12  (SYN-ACK)
  RTTが100ms以上          tcp.analysis.ack_rtt > 0.1
  ─────────────────────────────────────────────────────

[Step 3] TCP Stream の追跡
  - パケットを右クリック → 「Follow → TCP Stream」
  - クライアント→サーバーのデータが赤、逆が青で表示される
  - HTTP通信の場合、リクエストとレスポンスが見える

[Step 4] TCP統計情報の確認
  - Statistics → TCP Stream Graphs → Time-Sequence (tcptrace)
    → シーケンス番号の推移をグラフで表示
    → 再送やスループットの変化を視覚的に確認
  - Statistics → TCP Stream Graphs → Window Scaling
    → ウィンドウサイズの変化を確認
  - Statistics → TCP Stream Graphs → Round Trip Time
    → RTTの変動を確認
  - Statistics → Flow Graph
    → シーケンス図（ラダーダイアグラム）を表示

[Step 5] tshark（コマンドライン版Wireshark）でのフィルタリング

  # 3-way handshake のみ抽出
  tshark -r capture.pcap -Y "tcp.flags.syn == 1" \
    -T fields -e frame.time -e ip.src -e ip.dst \
    -e tcp.srcport -e tcp.dstport -e tcp.flags

  # 再送パケットを抽出
  tshark -r capture.pcap -Y "tcp.analysis.retransmission" \
    -T fields -e frame.time -e ip.src -e tcp.seq -e tcp.len

  # RTTの統計を取得
  tshark -r capture.pcap -Y "tcp.analysis.ack_rtt" \
    -T fields -e tcp.analysis.ack_rtt | \
    awk '{ sum+=$1; count++; if($1>max)max=$1 }
    END { print "Avg:", sum/count*1000, "ms",
                "Max:", max*1000, "ms",
                "Count:", count }'

  # ウィンドウサイズの推移を CSV で出力
  tshark -r capture.pcap \
    -Y "tcp.stream eq 0" \
    -T fields -e frame.time_relative -e tcp.window_size_value \
    -E separator=, > window_size.csv
```

---

## 9. FAQ（よくある質問）

### Q1: TCP と UDP、どちらを選ぶべきか？

**判断基準**

| 要件 | 選択 | 理由 |
|------|------|------|
| データの完全性が最重要 | **TCP** | 再送・順序保証により、データ欠損・破損を防ぐ |
| リアルタイム性が最重要 | **UDP** | 接続確立・再送待ちがなく、遅延を最小化できる |
| 小さなリクエスト・レスポンス | **UDP** | DNS、NTPのように1往復で完結する通信に適する |
| 長時間のストリーミング | **TCP** | HTTP/2、WebSocketで安定した配信が可能 |
| パケットロスが多い環境 | **TCP** | 自動再送により、ロス耐性が高い |
| ブロードキャスト・マルチキャスト | **UDP** | TCPは1対1通信のみ対応 |

**ハイブリッドアプローチ**
- QUIC（HTTP/3の基盤）: UDP上に独自の再送・輻輳制御を実装
- WebRTC: 映像（UDP）と制御信号（TCP/WebSocket）を併用

### Q2: TIME_WAIT問題をどう対処すべきか？

**問題の本質**
```bash
# TIME_WAIT 状態のソケット数を確認
$ ss -tan state time-wait | wc -l
12845  # 大量のTIME_WAITソケットが蓄積
```

**対策の優先順位**

**1. 接続の再利用（最も推奨）**
```python
# HTTP Keep-Alive: 同じTCPコネクションで複数リクエストを送信
import requests

session = requests.Session()
session.get('http://example.com/api/1')  # 接続確立
session.get('http://example.com/api/2')  # 同じ接続を再利用
session.get('http://example.com/api/3')  # 同じ接続を再利用
```

**2. SO_REUSEADDR オプションの活用**
```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# TIME_WAIT状態のアドレスを即座に再バインド可能にする
```

**3. tcp_tw_reuse の有効化（Linuxクライアント側のみ）**
```bash
# TIME_WAITソケットを新しい接続に再利用
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
```

**4. エフェメラルポート範囲の拡大**
```bash
# デフォルト: 32768〜60999（約28,000ポート）
# 拡大: 1024〜65535（約64,000ポート）
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"
```

**やってはいけない対策**
- ❌ `tcp_tw_recycle=1`: RFC違反で接続失敗の原因になる（カーネル4.12で削除済み）
- ❌ `tcp_fin_timeout` の過度な短縮: 古いパケットの誤配送リスクが高まる

### Q3: TCPのウィンドウサイズはどう調整すべきか？

**BDP（Bandwidth-Delay Product）の計算**
```
最適バッファサイズ = 帯域幅 × RTT

例1: 東京 ↔ 大阪（100Mbps、5ms RTT）
  BDP = 100Mbps × 5ms = 500Kb ÷ 8 = 62.5KB

例2: 東京 ↔ US西海岸（1Gbps、100ms RTT）
  BDP = 1Gbps × 100ms = 100Mb ÷ 8 = 12.5MB
```

**Linuxでの設定**
```bash
# 受信バッファ: 最小 / デフォルト / 最大
sudo sysctl -w net.ipv4.tcp_rmem="4096 131072 16777216"

# 送信バッファ: 最小 / デフォルト / 最大
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# 自動チューニングを有効化（推奨）
sudo sysctl -w net.ipv4.tcp_window_scaling=1
sudo sysctl -w net.ipv4.tcp_moderate_rcvbuf=1
```

**アプリケーション側での設定**
```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 受信バッファを 1MB に設定
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
```

**注意点**
- カーネルの自動チューニングに任せるのが基本的に最適
- 手動設定は計測結果に基づいて慎重に行う
- バッファを大きくしすぎると Bufferbloat（遅延増大）を引き起こす

---

## FAQ

### Q1: TCPとUDPはどう使い分けるべき?
信頼性が必要な通信（Webブラウジング、ファイル転送、メール、データベース接続等）にはTCPを使います。一方、リアルタイム性が最優先で多少のパケットロスが許容される用途（ゲーム、音声・映像ストリーミング、DNS問い合わせ等）にはUDPが適しています。HTTP/3はQUIC（UDP上に構築）を採用しており、TCPのHead-of-Line Blocking問題を回避しつつ信頼性も確保しています。

### Q2: TIME_WAIT状態のコネクションが大量に発生するのはなぜ?
TIME_WAITはTCPの正常な動作です。接続を能動的に閉じた側が2MSL（通常60秒〜120秒）の間この状態を維持し、遅延パケットが新しいコネクションに混入することを防いでいます。高トラフィック環境ではTIME_WAIT状態のコネクションが蓄積しポート枯渇を起こすことがあります。対策としては `SO_REUSEADDR` の設定、コネクションプーリングの活用、Keep-Aliveによる接続再利用が有効です。

### Q3: BBRとCUBICの違いは何? どちらを選ぶべき?
CUBICはパケットロスを輻輳シグナルとして利用する損失ベースのアルゴリズムで、Linux標準として広く使われています。BBR（Bottleneck Bandwidth and Round-trip propagation time）はGoogleが開発した帯域×遅延モデルに基づくアルゴリズムで、パケットロスに依存せずに最適な送信レートを推定します。BBRはバッファの大きいネットワーク（Bufferbloat環境）で高い性能を発揮しますが、フェアネスの問題も指摘されています。サーバーのカーネルバージョンと用途に応じて選択してください。

---

## まとめ

### TCP の核心理解

| 要素 | 内容 | 重要度 |
|------|------|--------|
| **信頼性保証** | シーケンス番号・ACK・再送により、データの到達・順序を保証する | ★★★ |
| **3-wayハンドシェイク** | SYN → SYN-ACK → ACK で接続を確立（1.5 RTT） | ★★★ |
| **フロー制御** | スライディングウィンドウ（rwnd）で受信側のバッファ溢れを防ぐ | ★★★ |
| **輻輳制御** | cwnd を動的調整してネットワーク全体の安定性に貢献（Reno, CUBIC, BBR） | ★★★ |
| **4-wayハンドシェイク** | FIN → ACK → FIN → ACK で接続を終了、TIME_WAIT で古いパケットを排除 | ★★☆ |
| **HoL Blocking** | 1つのパケットロストが全体をブロックする問題（QUIC/HTTP/3 で解決） | ★★☆ |

### キーポイント

1. **TCP = 信頼性と引き換えに遅延を許容するプロトコル**
   - 接続確立に 1.5 RTT、TLS併用で最大 3 RTT
   - パケットロス時の再送待ちで遅延が増大
   - HTTP/3（QUIC）はこれらの課題を UDP ベースで解決

2. **フロー制御と輻輳制御は別物**
   - フロー制御（rwnd）: 受信側の処理能力に合わせる（エンドツーエンド）
   - 輻輳制御（cwnd）: ネットワークの混雑状況を推測する（送信側が自律的に実施）
   - 実効ウィンドウ = min(rwnd, cwnd)

3. **TCP は進化し続けている**
   - RFC 9293（2022年）: 最新の TCP 仕様
   - CUBIC（2006年〜）: Linux 標準の輻輳制御アルゴリズム
   - BBR（2016年〜）: Google が推進する帯域×遅延モデル
   - ECN（Explicit Congestion Notification）: ルーターからの明示的な輻輳通知

---

## 次に読むべきガイド

**プロトコルの理解を深める**

**実践的なスキル**

---

## 参考文献

### RFC（標準仕様）

1. **RFC 9293 - Transmission Control Protocol (TCP)**
   https://www.rfc-editor.org/rfc/rfc9293.html
   2022年8月発行。TCPの最新の標準仕様。RFC 793（1981年）の後継で、40年分の更新を統合。

2. **RFC 5681 - TCP Congestion Control**
   https://www.rfc-editor.org/rfc/rfc5681.html
   スロースタート、輻輳回避、高速再送・高速回復を規定。TCP Renoの基盤。

3. **RFC 7413 - TCP Fast Open**
   https://www.rfc-editor.org/rfc/rfc7413.html
   SYNパケットにデータを含めることで接続確立を1 RTTに短縮する仕組み。

4. **RFC 6298 - Computing TCP's Retransmission Timer**
   https://www.rfc-editor.org/rfc/rfc6298.html
   RTO（再送タイムアウト）の計算方法を規定。Karn's Algorithm、Jacobson's Algorithm を含む。

5. **RFC 3168 - The Addition of Explicit Congestion Notification (ECN) to IP**
   https://www.rfc-editor.org/rfc/rfc3168.html
   パケットロスなしに輻輳を通知する仕組み。BBRv2で活用される。

6. **RFC 9438 - CUBIC for Fast Long-Distance Networks**
   https://www.rfc-editor.org/rfc/rfc9438.html
   2023年8月発行。Linux標準の輻輳制御アルゴリズムCUBICの仕様。

### 書籍

7. **Stevens, W. Richard. "TCP/IP Illustrated, Volume 1: The Protocols, 2nd Edition." Addison-Wesley, 2011.**
   TCPの動作を詳細に解説した定番書。パケットキャプチャと図解で理解が深まる。

8. **Fall, Kevin R. and Stevens, W. Richard. "TCP/IP Illustrated, Volume 2: The Implementation." Addison-Wesley, 1995.**
   BSD TCPの実装を詳細に解説。カーネルレベルの理解に最適。

9. **Grigorik, Ilya. "High Performance Browser Networking." O'Reilly Media, 2013.**
   Chapter 2: Building Blocks of TCP でTCPの性能最適化を実践的に解説。無料公開版: https://hpbn.co/

### 論文・技術記事

10. **Cardwell, Neal et al. "BBR: Congestion-Based Congestion Control." ACM Queue, Vol. 14 No. 5, 2016.**
    https://queue.acm.org/detail.cfm?id=3022184
    Google開発のBBRアルゴリズムの設計思想と評価。

11. **Ha, Sangtae et al. "CUBIC: A New TCP-Friendly High-Speed TCP Variant." ACM SIGOPS Operating Systems Review, 2008.**
    CUBICの原論文。高BDP環境での性能向上を実証。

12. **Jacobson, Van. "Congestion Avoidance and Control." ACM SIGCOMM, 1988.**
    TCP輻輳制御の基礎を築いた歴史的論文。スロースタート、輻輳回避の原典。

