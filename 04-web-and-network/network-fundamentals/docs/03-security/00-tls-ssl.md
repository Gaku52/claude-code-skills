# TLS/SSL

> TLSはインターネット通信の暗号化を担うプロトコル。ハンドシェイク、証明書、暗号スイートの仕組みを理解し、安全な通信の基盤を学ぶ。

## 前提知識

このガイドを理解するには以下の知識が必要です:
- [[../01-protocols/00-tcp.md]] — TCPの3-wayハンドシェイクとコネクション管理
- 公開鍵暗号の基礎 — RSA/ECDSA等の非対称暗号の仕組み
- [[../02-http/00-http-basics.md]] — HTTPプロトコルの基本とリクエスト/レスポンス構造

---

## この章で学ぶこと

- [ ] TLSハンドシェイクの流れを理解する
- [ ] 証明書の仕組みと検証プロセスを把握する
- [ ] TLS 1.3の改善点を学ぶ
- [ ] 暗号スイートの選定と設定を実践する
- [ ] 実務でのTLS設定とトラブルシューティングを習得する

---

## 1. TLSの基本

```
TLS（Transport Layer Security）:
  → 通信の暗号化、認証、完全性を提供
  → HTTPS = HTTP + TLS

歴史:
  SSL 2.0 (1995) → 脆弱、使用禁止
  SSL 3.0 (1996) → POODLE攻撃、使用禁止
  TLS 1.0 (1999) → 非推奨（2021年にRFC 8996で正式廃止）
  TLS 1.1 (2006) → 非推奨（2021年にRFC 8996で正式廃止）
  TLS 1.2 (2008) → 現在広く使用
  TLS 1.3 (2018) → 最新、推奨

  ※ SSLは全バージョン廃止。「SSL証明書」は慣習的な名称

TLSが提供する3つの機能:
  ① 機密性（Confidentiality）: 通信内容の暗号化
  ② 認証（Authentication）: 通信相手の本人確認
  ③ 完全性（Integrity）: データの改ざん検知

TLSの位置づけ（OSI参照モデル）:
  アプリケーション層: HTTP, SMTP, FTP
  ────────────────────────────────
  TLS（セッション〜プレゼンテーション層）
  ────────────────────────────────
  トランスポート層: TCP
  ────────────────────────────────
  ネットワーク層: IP

  → TLSはTCPの上、アプリケーションの下に位置
  → アプリケーションプロトコルに依存しない
  → HTTP以外にもSMTPS, FTPS, LDAPS等で使用
```

### 1.1 暗号技術の基礎

```
TLSが使用する暗号技術:

① 共通鍵暗号（対称暗号）:
  → 同じ鍵で暗号化と復号
  → 高速（大量データの暗号化に適する）
  → 鍵の配送が課題
  → AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305

  平文 ──[共通鍵]──→ 暗号文 ──[共通鍵]──→ 平文

② 公開鍵暗号（非対称暗号）:
  → 公開鍵で暗号化、秘密鍵で復号
  → 低速（鍵交換・署名に使用）
  → 鍵の配送問題を解決
  → RSA, ECDSA, Ed25519

  平文 ──[公開鍵]──→ 暗号文 ──[秘密鍵]──→ 平文

③ ハッシュ関数:
  → 任意長データから固定長ダイジェストを生成
  → 一方向性（逆算不可能）
  → 衝突耐性（同じハッシュになる入力を見つけるのが困難）
  → SHA-256, SHA-384, SHA-512

④ MAC（Message Authentication Code）:
  → メッセージの完全性と認証を保証
  → HMAC: ハッシュ関数ベースのMAC
  → AEAD: 暗号化と認証を同時に行う（GCM, Poly1305）

⑤ 鍵交換アルゴリズム:
  → 安全でないチャネル上で共通鍵を確立
  → Diffie-Hellman（DH）
  → Elliptic Curve Diffie-Hellman（ECDH）
  → 一時鍵（Ephemeral）: ECDHE → 前方秘匿性を提供

TLSでの使い分け:
  鍵交換:    ECDHE（公開鍵暗号ベース）
  認証:      RSA or ECDSA（公開鍵暗号）
  データ暗号化: AES-GCM or ChaCha20（共通鍵暗号）
  完全性:    AEAD（GCM, Poly1305）が暗号化と同時に提供
```

---

## 2. TLS 1.2 ハンドシェイク

```
TLS 1.2 ハンドシェイク（2 RTT）:

  クライアント                      サーバー
       │                              │
  ①    │── ClientHello ──→            │
       │   対応TLSバージョン           │
       │   対応暗号スイート一覧        │
       │   クライアントランダム        │
       │   セッションID                │
       │   SNI（Server Name Indication）│
       │                              │
  ②    │←── ServerHello ──           │
       │   選択されたTLSバージョン     │
       │   選択された暗号スイート      │
       │   サーバーランダム            │
       │   セッションID                │
       │                              │
  ③    │←── Certificate ──           │
       │   サーバー証明書チェーン      │
       │                              │
  ④    │←── ServerKeyExchange ──     │
       │   鍵交換パラメータ(ECDHE等)   │
       │   署名（秘密鍵による）        │
       │                              │
  ⑤    │←── ServerHelloDone ──       │
       │                              │
       │   [クライアント側で証明書検証]│
       │                              │
  ⑥    │── ClientKeyExchange ──→     │
       │   鍵交換パラメータ            │
       │                              │
       │   [両者がプリマスターシークレットを算出]
       │   [マスターシークレット → セッション鍵を導出]
       │                              │
  ⑦    │── ChangeCipherSpec ──→      │
       │── Finished ──→               │
       │   （ハンドシェイク全体のハッシュを暗号化）
       │                              │
  ⑧    │←── ChangeCipherSpec ──      │
       │←── Finished ──               │
       │                              │
       │   暗号化通信開始              │

  所要時間: 2 RTT（東京-US間: 約200ms）
```

### 2.1 各ステップの詳細解説

```
ClientHello の詳細:
  → TLSバージョン: 最高でTLS 1.2を提示
  → CipherSuites: 優先順位付きの暗号スイートリスト
  → Compression Methods: 通常null（圧縮はCRIME攻撃の原因）
  → Extensions:
     - server_name (SNI): 接続先ドメイン名
     - supported_groups: 対応する楕円曲線（P-256, X25519等）
     - signature_algorithms: 対応する署名アルゴリズム
     - session_ticket: セッション再開用チケット

SNI（Server Name Indication）:
  → 1つのIPで複数ドメインのTLS証明書を使い分けるための拡張
  → ClientHelloに接続先ドメイン名を含める
  → ただしClientHelloは平文 → ドメイン名が見える
  → ESNI（Encrypted SNI）/ ECH（Encrypted Client Hello）で解決

セッション再開（Session Resumption）:
  ① Session ID:
     → サーバー側にセッション状態を保存
     → クライアントがIDを提示して再開
     → サーバーのメモリ消費が課題

  ② Session Ticket:
     → セッション状態を暗号化してクライアントに渡す
     → サーバーはステートレス
     → チケット暗号化鍵の管理が課題

  再開時: 1 RTT（フルハンドシェイク不要）
```

### 2.2 鍵導出プロセス

```
TLS 1.2の鍵導出:

  1. プリマスターシークレット:
     → ECDHE: 共有シークレットを算出
     → 両者のDHパラメータから同じ値を導出

  2. マスターシークレット:
     master_secret = PRF(
       pre_master_secret,
       "master secret",
       ClientHello.random + ServerHello.random
     )

  3. セッション鍵の導出:
     key_block = PRF(
       master_secret,
       "key expansion",
       ServerHello.random + ClientHello.random
     )

     key_blockから以下を切り出し:
     → client_write_MAC_key
     → server_write_MAC_key
     → client_write_key（暗号化鍵）
     → server_write_key（暗号化鍵）
     → client_write_IV
     → server_write_IV

  PRF = 擬似乱数関数（HMAC-SHA256ベース）

  ポイント:
  → クライアントとサーバーで別々の鍵を使用
  → MACと暗号化で別々の鍵を使用
  → ランダム値を含めることでセッションごとにユニークな鍵
```

---

## 3. TLS 1.3 ハンドシェイク

```
TLS 1.3 ハンドシェイク（1 RTT）:

  クライアント                      サーバー
       │                              │
  ①    │── ClientHello ──→            │
       │   対応暗号スイート            │
       │   鍵共有パラメータ（推測）    │  ← 1RTTの秘密
       │   supported_versions拡張      │
       │                              │
  ②    │←── ServerHello ──           │
       │   選択された暗号スイート      │
       │   鍵共有パラメータ            │
       │                              │
       │←── {EncryptedExtensions} ── │  ← ここから暗号化
       │←── {Certificate} ──         │
       │←── {CertificateVerify} ──   │
       │←── {Finished} ──             │
       │                              │
  ③    │── {Finished} ──→             │
       │                              │
       │   暗号化通信開始              │

  改善点:
  ① 1 RTT: 鍵交換を最初から送信（推測ベース）
  ② 0-RTT: 再接続時はデータも同時送信可能
  ③ 脆弱な暗号スイートを廃止
  ④ ハンドシェイクも暗号化（証明書が見えない）
  ⑤ CertificateVerify追加（サーバー認証の強化）
```

### 3.1 TLS 1.3の鍵スケジュール

```
TLS 1.3の鍵導出（HKDF ベース）:

  HKDF = HMAC-based Key Derivation Function
  → TLS 1.2のPRFよりも明確で安全な設計

  鍵導出の流れ:

  PSK（Pre-Shared Key、なければ0）
      │
      ▼
  Early Secret ── Derive-Secret → client_early_traffic_secret
      │                            （0-RTTデータ用）
      │
  ECDHE共有シークレット
      │
      ▼
  Handshake Secret
      │
      ├── Derive-Secret → client_handshake_traffic_secret
      │                    （ハンドシェイク暗号化用）
      ├── Derive-Secret → server_handshake_traffic_secret
      │
      ▼
  Master Secret
      │
      ├── Derive-Secret → client_application_traffic_secret
      │                    （アプリケーションデータ用）
      ├── Derive-Secret → server_application_traffic_secret
      │
      ├── Derive-Secret → exporter_master_secret
      └── Derive-Secret → resumption_master_secret
                           （セッション再開用）

  ポイント:
  → 段階的な鍵導出で用途別の鍵を生成
  → ハンドシェイクとアプリケーションデータで別鍵
  → 鍵のコンプロマイズ範囲を限定
```

### 3.2 0-RTT（Early Data）

```
0-RTT 再接続:
  クライアント ── ClientHello + アプリデータ ──→ サーバー
  → 接続確立とデータ送信を同時に

  前提:
  → 以前の接続でPSK（Pre-Shared Key）を取得済み
  → PSKからearly_traffic_secretを導出
  → 最初のリクエストを暗号化して送信

  利点:
  → 0 RTTでデータ送信開始
  → Webパフォーマンスの大幅改善

  リスク:
  → リプレイ攻撃: 攻撃者がパケットを再送可能
  → 前方秘匿性なし: PSKが漏洩すると0-RTTデータは復号可能

  対策:
  → 冪等なリクエストのみ（GET等）で使用
  → サーバー側でリプレイ検知を実装
  → Single-Use Ticketの使用
  → Anti-Replay Window の設定

  設定例（Nginx）:
  ssl_early_data on;
  proxy_set_header Early-Data $ssl_early_data;
  → バックエンドで Early-Data: 1 を見て冪等性を判断
```

### 3.3 TLS 1.3で廃止された機能

```
TLS 1.3で廃止された暗号・機能:

  暗号アルゴリズム:
  ✗ RSA鍵交換（前方秘匿性なし）
  ✗ CBC モードの暗号（パディングオラクル攻撃）
  ✗ RC4（多数の脆弱性）
  ✗ DES, 3DES（鍵長が短い）
  ✗ MD5, SHA-1（衝突攻撃が実用的）
  ✗ 静的DH（前方秘匿性なし）
  ✗ カスタムDHグループ（弱いパラメータの可能性）
  ✗ 圧縮（CRIME攻撃）
  ✗ 再ネゴシエーション（三重ハンドシェイク攻撃）

  プロトコル機能:
  ✗ ChangeCipherSpec メッセージ（不要に）
  ✗ セッションID による再開（PSKベースに変更）
  ✗ ランダムなセッションIDの生成

  なぜ廃止が重要か:
  → 選択肢が少ない = 設定ミスの可能性が低い
  → 安全でない構成が不可能になる
  → 実装の複雑さが軽減
  → 監査が容易に
```

---

## 4. 証明書

```
X.509 証明書の構造:
  ┌───────────────────────────────────────┐
  │ バージョン: v3                         │
  │ シリアル番号: 01:AB:CD:EF:...         │
  │ 署名アルゴリズム: SHA-256 with RSA    │
  │ 発行者: Let's Encrypt Authority X3     │
  │ 有効期間: 2024-01-01 〜 2024-03-31    │
  │ サブジェクト: example.com              │
  │ 公開鍵: RSA 2048bit / ECDSA P-256     │
  │ 拡張:                                  │
  │   SANs: example.com, www.example.com  │
  │   Key Usage: Digital Signature        │
  │   Extended Key Usage: Server Auth     │
  │   Basic Constraints: CA:FALSE         │
  │   CRL Distribution Points: ...       │
  │   Authority Information Access: ...   │
  │     OCSP Responder: http://ocsp...    │
  │     CA Issuers: http://...            │
  │ 署名値: 3A:4B:5C:...                  │
  └───────────────────────────────────────┘
```

### 4.1 証明書チェーン

```
証明書チェーン:
  ┌────────────────┐
  │ ルートCA証明書   │ ← OSやブラウザに内蔵（信頼の起点）
  │ (自己署名)      │    約150〜200個の信頼済みCA
  └───────┬────────┘
          │ 署名
  ┌───────▼────────┐
  │ 中間CA証明書    │ ← ルートCAが署名
  │               │    ルートCAのオフライン保護のため
  └───────┬────────┘
          │ 署名
  ┌───────▼────────┐
  │ サーバー証明書   │ ← 中間CAが署名
  │ example.com    │
  └────────────────┘

検証プロセス:
  1. サーバー証明書の署名を中間CAの公開鍵で検証
  2. 中間CA証明書の署名をルートCAの公開鍵で検証
  3. ルートCA証明書がOS/ブラウザの信頼ストアに存在するか確認
  4. 証明書の有効期限を確認
  5. ドメイン名（SAN）の一致を確認
  6. 失効確認（OCSP / CRL）
  7. Certificate Transparency（CT）ログの確認

中間CA証明書の重要性:
  → サーバーは中間CA証明書も送信する必要がある
  → 送信漏れ → 「信頼されていない証明書」エラー
  → 特にAndroid等で問題になりやすい（信頼ストアが異なる）
  → ssl-cert-check や SSL Labs でチェーン不備を検出可能

クロスサイン（Cross-signing）:
  → 新しいCAが古いルートCAにも署名してもらう
  → 古いデバイスでも信頼チェーンが成立
  → Let's Encrypt: ISRG Root X1 + DST Root CA X3（クロスサイン）
```

### 4.2 証明書の種類と選定

```
証明書の種類:

  DV（ドメイン検証）:
    → ドメインの所有のみ確認
    → Let's Encrypt等（無料）
    → 発行まで数分
    → 個人サイト、スタートアップ向け

  OV（組織検証）:
    → 組織の実在を確認
    → 企業サイト向け
    → 発行まで数日
    → 年間$50〜$300程度

  EV（拡張検証）:
    → 厳格な組織審査
    → 金融機関等
    → 発行まで数週間
    → 年間$150〜$1,000程度
    → ブラウザ表示の差は縮小傾向（アドレスバーの組織名表示廃止）

ワイルドカード証明書:
  → *.example.com で全サブドメインに対応
  → ただし1階層のみ（*.sub.example.com は不可）
  → DNSチャレンジが必要（HTTP-01では不可）

マルチドメイン証明書（SAN証明書）:
  → 1枚の証明書に複数ドメインを含める
  → example.com + example.org + example.net
  → 管理の簡素化

選定ガイドライン:
  個人サイト → Let's Encrypt（DV、無料）
  一般企業サイト → DV or OV
  金融・医療 → OV or EV
  マイクロサービス → 内部CA（自己署名）
  複数サブドメイン → ワイルドカード
```

### 4.3 証明書の失効

```
証明書の失効確認:

① CRL（Certificate Revocation List）:
  → CAが失効証明書のリストを公開
  → クライアントがダウンロードして照合
  → 欠点: リストが大きくなる、更新頻度が低い

② OCSP（Online Certificate Status Protocol）:
  → リアルタイムで証明書の有効性を確認
  → CAのOCSPレスポンダーに問い合わせ

  クライアント ──→ OCSPレスポンダー
                   │
                   ← "good" / "revoked" / "unknown"

  欠点:
  → レイテンシ増加（追加のネットワークリクエスト）
  → CAがダウンすると確認不可
  → プライバシー問題（CAにアクセス先がわかる）

③ OCSP Stapling:
  → サーバーがOCSPレスポンスを事前取得してTLSハンドシェイクに含める
  → クライアントはCAに問い合わせ不要
  → レイテンシ削減 + プライバシー向上

  設定例（Nginx）:
  ssl_stapling on;
  ssl_stapling_verify on;
  resolver 8.8.8.8 8.8.4.4 valid=300s;
  resolver_timeout 5s;

④ OCSP Must-Staple:
  → 証明書にOCSP Staling必須の拡張を含める
  → Stapled応答がない場合は接続を拒否
  → 失効確認のソフトフェイル問題を解決

⑤ CRLite（Firefox実装）:
  → CRLをBloomフィルターで圧縮してブラウザに配布
  → 全証明書の失効状態をローカルで確認
  → OCSP不要、プライバシー保護
```

---

## 5. 暗号スイート

```
暗号スイートの構成（TLS 1.2）:
  TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
  │     │      │      │    │    │     │
  │     │      │      │    │    │     └── ハッシュ関数
  │     │      │      │    │    └─ モード
  │     │      │      │    └─ 鍵長
  │     │      │      └── 共通鍵暗号
  │     │      └── 認証アルゴリズム
  │     └── 鍵交換アルゴリズム
  └── プロトコル

  各コンポーネントの選択肢:
  鍵交換: ECDHE（推奨）, DHE, RSA（非推奨）
  認証:   RSA, ECDSA, Ed25519
  暗号化: AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305
  ハッシュ: SHA-256, SHA-384

TLS 1.3の暗号スイート（簡素化）:
  TLS_AES_256_GCM_SHA384
  TLS_AES_128_GCM_SHA256
  TLS_CHACHA20_POLY1305_SHA256
  → 鍵交換は常にECDHE（前方秘匿性あり）
  → 認証は証明書で別途処理
  → 選択肢を減らしてセキュリティ向上
  → AEAD暗号のみ（CBC廃止）
```

### 5.1 前方秘匿性（Forward Secrecy）

```
前方秘匿性（Forward Secrecy / Perfect Forward Secrecy）:
  → サーバーの秘密鍵が漏洩しても、過去の通信は復号できない
  → 毎回新しい一時鍵（ephemeral key）を生成
  → ECDHE: Elliptic Curve Diffie-Hellman Ephemeral

RSA鍵交換（前方秘匿性なし）:
  1. クライアント → プリマスターシークレットをRSA公開鍵で暗号化
  2. サーバー → RSA秘密鍵で復号
  → 秘密鍵が漏洩 → 全過去通信を復号可能

  攻撃シナリオ:
  ① 攻撃者が暗号化通信を長期間記録
  ② サーバーの秘密鍵を何らかの方法で入手
  ③ 記録した全通信を復号
  → "Record now, decrypt later" 攻撃

ECDHE鍵交換（前方秘匿性あり）:
  1. 両者が一時的なDHパラメータを交換
  2. 共有シークレットを算出
  3. 一時鍵は使い捨て（メモリから消去）
  → 秘密鍵が漏洩しても一時鍵は復元不可能
  → 過去の通信は安全

  楕円曲線の選択:
  → X25519: 最速、推奨、Daniel J. Bernstein設計
  → P-256: NIST標準、広くサポート
  → P-384: 高セキュリティ要件向け
  → P-521: 最高セキュリティ（パフォーマンスに影響）

ポスト量子暗号（Post-Quantum）:
  → 量子コンピュータでECDHEが破られる可能性
  → NIST PQC標準化（2024年）:
     ML-KEM（旧CRYSTALS-Kyber）
     ML-DSA（旧CRYSTALS-Dilithium）
  → ハイブリッド方式: X25519 + ML-KEM-768
  → Chrome/Firefox: X25519Kyber768 の試験実装
  → TLS 1.3で先行対応可能
```

### 5.2 AES-GCM vs ChaCha20-Poly1305

```
AES-GCM:
  → AES暗号 + Galois/Counter Modeの認証付き暗号
  → ハードウェアアクセラレーション（AES-NI）対応
  → Intel/AMDプロセッサで非常に高速
  → サーバーサイドのデファクトスタンダード

ChaCha20-Poly1305:
  → ChaCha20ストリーム暗号 + Poly1305 MAC
  → ソフトウェア実装で高速（AES-NI非対応環境向け）
  → モバイルデバイス（ARM）で有利
  → Google開発、広く採用

選択の指針:
  デスクトップ/サーバー: AES-256-GCM（AES-NI活用）
  モバイル:              ChaCha20-Poly1305（省電力）
  推奨: 両方サポートし、クライアントに選ばせる

暗号モードの比較:
  GCM（Galois/Counter Mode）:
    → AEAD: 暗号化 + 認証を同時
    → カウンターモードベース → 並列処理可能
    → nonce再利用で致命的な脆弱性
    → 96ビットnonce + 32ビットカウンター

  CBC（Cipher Block Chaining）:
    → TLS 1.3で廃止
    → パディングオラクル攻撃に脆弱
    → MACの処理順序の問題（Encrypt-then-MAC vs MAC-then-Encrypt）
    → 直列処理のみ → 遅い
```

---

## 6. 実務でのTLS設定

### 6.1 Nginx設定

```nginx
# /etc/nginx/conf.d/ssl.conf
# 推奨TLS設定（Mozilla Intermediate Configuration）

server {
    listen 443 ssl http2;
    server_name example.com;

    # 証明書
    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # TLSバージョン
    ssl_protocols TLSv1.2 TLSv1.3;

    # 暗号スイート（TLS 1.2用）
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;  # TLS 1.3ではクライアント優先

    # DH パラメータ（TLS 1.2 DHE用）
    ssl_dhparam /etc/nginx/dhparam.pem;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/example.com/chain.pem;
    resolver 8.8.8.8 8.8.4.4 valid=300s;

    # セッション設定
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;  # 前方秘匿性を厳密に保つなら無効化

    # セキュリティヘッダー
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # TLS 1.3 0-RTT（オプション）
    ssl_early_data on;
    proxy_set_header Early-Data $ssl_early_data;
}

# HTTPからHTTPSへのリダイレクト
server {
    listen 80;
    server_name example.com;
    return 301 https://$host$request_uri;
}
```

### 6.2 Let's Encrypt自動化

```bash
# certbot インストール（Ubuntu）
$ sudo apt install certbot python3-certbot-nginx

# 証明書取得（Nginx自動設定）
$ sudo certbot --nginx -d example.com -d www.example.com

# 証明書取得（DNSチャレンジ、ワイルドカード用）
$ sudo certbot certonly --manual --preferred-challenges dns \
  -d '*.example.com' -d example.com

# 自動更新のテスト
$ sudo certbot renew --dry-run

# 自動更新（cron / systemd timer）
$ sudo crontab -e
0 0,12 * * * certbot renew --quiet --post-hook "systemctl reload nginx"

# systemd timerの確認
$ systemctl list-timers | grep certbot

# 証明書の状態確認
$ sudo certbot certificates
```

```bash
# ACME プロトコルの仕組み
#
# 1. アカウント登録
#    クライアント → Let's Encrypt: 公開鍵を登録
#
# 2. 認証チャレンジ
#    HTTP-01: http://example.com/.well-known/acme-challenge/<TOKEN>
#    DNS-01:  _acme-challenge.example.com TXT <TOKEN>
#    TLS-ALPN-01: TLSハンドシェイク中にトークンを提示
#
# 3. チャレンジ検証
#    Let's Encrypt → example.com: トークンを確認
#
# 4. 証明書発行
#    CSR（Certificate Signing Request）を送信
#    署名済み証明書を受信
#
# HTTP-01 vs DNS-01:
#   HTTP-01: 簡単、ポート80のアクセス必要、ワイルドカード不可
#   DNS-01: DNSレコード操作が必要、ワイルドカード可能、CDN背後でも使用可能
```

### 6.3 内部通信のTLS（mTLS）

```
mTLS（mutual TLS / 相互TLS認証）:
  → クライアントもサーバーも証明書を提示
  → サービスメッシュやマイクロサービス間通信で使用

  通常のTLS:
    クライアント → サーバーの証明書を検証（一方向）

  mTLS:
    クライアント → サーバーの証明書を検証
    サーバー   → クライアントの証明書を検証（双方向）

  用途:
  → Kubernetes のサービス間通信（Istio, Linkerd）
  → ゼロトラストネットワーク
  → APIゲートウェイの認証
  → IoTデバイスの認証

  Nginx設定:
  ssl_client_certificate /etc/nginx/ca.crt;
  ssl_verify_client on;
  ssl_verify_depth 2;

  → クライアント証明書がない場合: 403 Forbidden
  → 無効な証明書の場合: 400 Bad Request
```

```yaml
# Istio での mTLS設定（PeerAuthentication）
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system
spec:
  mtls:
    mode: STRICT  # 全通信でmTLS必須

---
# DestinationRule（クライアント側設定）
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: default
spec:
  host: "*.local"
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
```

---

## 7. TLSの脆弱性と攻撃

### 7.1 歴史的な脆弱性

```
主要なTLS関連の脆弱性:

① BEAST（2011）:
  → TLS 1.0のCBCモードに対する攻撃
  → 選択平文攻撃でCookieを復号
  → 対策: TLS 1.1以上に移行

② CRIME（2012）:
  → TLS圧縮を利用した攻撃
  → 圧縮率の差からシークレットを推測
  → 対策: TLS圧縮の無効化

③ Lucky Thirteen（2013）:
  → CBCモードのパディング処理のタイミング差を悪用
  → 対策: constant-time実装、AEAD暗号へ移行

④ Heartbleed（2014）:
  → OpenSSLのハートビート拡張の実装バグ
  → サーバーメモリの最大64KB漏洩
  → 秘密鍵、セッションデータ等が漏洩
  → CVE-2014-0160
  → 対策: OpenSSLアップデート、証明書再発行

⑤ POODLE（2014）:
  → SSL 3.0のCBCパディング処理の脆弱性
  → SSL 3.0への ダウングレード攻撃と組み合わせ
  → 対策: SSL 3.0の無効化

⑥ FREAK（2015）:
  → 輸出グレード暗号（512ビットRSA）への ダウングレード
  → 対策: 輸出グレード暗号の無効化

⑦ Logjam（2015）:
  → DHE鍵交換の弱いパラメータ（512/1024ビット）
  → 対策: 2048ビット以上のDHパラメータ

⑧ ROBOT（2017）:
  → RSA鍵交換のパディングオラクル
  → Bleichenbacher攻撃の変種
  → 対策: RSA鍵交換の無効化（ECDHE使用）

⑨ RACCOON（2020）:
  → DH鍵交換のタイミングサイドチャネル
  → 対策: constant-time実装

教訓:
  → TLS 1.3は上記の多くの攻撃を根本的に排除
  → 古いプロトコルバージョンを無効化する重要性
  → ソフトウェアの定期的なアップデート
```

### 7.2 ダウングレード攻撃とその対策

```
ダウングレード攻撃:
  → クライアントとサーバーの間に入り、
    弱い暗号スイートやTLSバージョンを強制

  攻撃手法:
  1. ClientHelloを改ざん（対応バージョンをTLS 1.0に書き換え）
  2. サーバーはTLS 1.0で応答
  3. 弱い暗号で通信が確立
  4. 攻撃者が復号

  対策:
  ① TLS_FALLBACK_SCSV:
     → フォールバック時に特殊な暗号スイートを送信
     → サーバーがダウングレードを検知して接続拒否

  ② TLS 1.3のダウングレード防止:
     → ServerHello.random の末尾8バイトに固定値
     → TLS 1.2: 0x44 0x4F 0x57 0x4E 0x47 0x52 0x44 0x01
     → TLS 1.1以下: 0x44 0x4F 0x57 0x4E 0x47 0x52 0x44 0x00
     → クライアントがこれを検出して接続拒否

  ③ 古いバージョンの無効化:
     → サーバー側でTLS 1.0/1.1を無効に
     → ssl_protocols TLSv1.2 TLSv1.3;

  ④ HSTS Preload:
     → ブラウザがHTTP接続自体を拒否
     → https://hstspreload.org/ で登録
```

---

## 8. Certificate Transparency（CT）

```
Certificate Transparency:
  → 証明書の発行を公開ログに記録する仕組み
  → 不正な証明書の発行を検知

  背景:
  → 過去にCAが不正な証明書を発行した事例
  → DigiNotar（2011）: *.google.com の偽証明書
  → Symantec（2017）: 信頼取り消し

  仕組み:
  1. CAが証明書をCTログサーバーに登録
  2. CTログサーバーがSCT（Signed Certificate Timestamp）を返却
  3. SCTが証明書に埋め込まれるか、TLSハンドシェイクで送信
  4. ブラウザがSCTを検証

  ┌────┐     ┌──────────┐     ┌───────┐
  │ CA │ ──→ │ CTログ    │ ──→ │ モニター│
  └──┬─┘     └──────┬───┘     └───────┘
     │              │                 │
     │ SCT          │ 公開ログ        │ 不正検知
     ▼              ▼                 ▼
  ┌────────┐  ┌──────────┐     ┌───────┐
  │ サーバー │  │ ブラウザ  │     │ 管理者 │
  └────────┘  └──────────┘     └───────┘

  CTログの監視:
  → crt.sh: 発行された証明書の検索
  → https://crt.sh/?q=example.com
  → 不正な証明書の発行を早期に検知

  Chrome要件:
  → 2018年以降、CT対応が必須
  → SCTが2つ以上必要
  → CT非対応の証明書は「Not Transparent」警告
```

---

## 9. 実務での確認コマンド

```bash
# TLS接続の詳細確認
$ openssl s_client -connect example.com:443

# TLS 1.3での接続を強制
$ openssl s_client -connect example.com:443 -tls1_3

# 証明書チェーンの表示
$ openssl s_client -connect example.com:443 -showcerts

# 証明書の内容を確認
$ echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -text -noout

# 証明書の有効期限確認
$ echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -noout -dates

# 証明書のSAN（サブジェクト代替名）確認
$ echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -noout -ext subjectAltName

# OCSP Staplingの確認
$ echo | openssl s_client -connect example.com:443 -status 2>/dev/null | \
  grep -A 20 "OCSP Response"

# TLSバージョンとスイートの確認
$ curl -v https://example.com 2>&1 | grep -E '(SSL|TLS)'

# 特定の暗号スイートでの接続テスト
$ openssl s_client -connect example.com:443 \
  -cipher ECDHE-RSA-AES256-GCM-SHA384

# 対応する暗号スイートの一覧取得
$ nmap --script ssl-enum-ciphers -p 443 example.com

# SSL Labs でオンラインチェック（グレードA+を目指す）
# https://www.ssllabs.com/ssltest/

# testssl.sh（包括的なTLSテストツール）
$ git clone https://github.com/drwetter/testssl.sh
$ ./testssl.sh example.com

# certigo（証明書の詳細表示）
$ certigo connect example.com:443

# SSLyze（Python製TLSスキャナー）
$ pip install sslyze
$ sslyze example.com
```

### 9.1 証明書管理の自動化スクリプト

```bash
#!/bin/bash
# check-cert-expiry.sh - 証明書有効期限の一括確認

DOMAINS=(
  "example.com"
  "api.example.com"
  "www.example.com"
)

WARN_DAYS=30

for domain in "${DOMAINS[@]}"; do
  expiry=$(echo | openssl s_client -connect "$domain:443" 2>/dev/null | \
    openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)

  if [ -z "$expiry" ]; then
    echo "[ERROR] $domain: 証明書取得失敗"
    continue
  fi

  expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || \
                 date -j -f "%b %d %T %Y %Z" "$expiry" +%s 2>/dev/null)
  now_epoch=$(date +%s)
  days_left=$(( (expiry_epoch - now_epoch) / 86400 ))

  if [ "$days_left" -lt 0 ]; then
    echo "[EXPIRED] $domain: ${days_left}日前に期限切れ"
  elif [ "$days_left" -lt "$WARN_DAYS" ]; then
    echo "[WARNING] $domain: 残り${days_left}日 (${expiry})"
  else
    echo "[OK]      $domain: 残り${days_left}日 (${expiry})"
  fi
done
```

```python
# Python での証明書有効期限確認
import ssl
import socket
from datetime import datetime

def check_certificate(hostname: str, port: int = 443) -> dict:
    """証明書情報を取得して返す"""
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port), timeout=10) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert = ssock.getpeercert()

    # 有効期限
    not_after = datetime.strptime(
        cert['notAfter'], '%b %d %H:%M:%S %Y %Z'
    )
    days_left = (not_after - datetime.utcnow()).days

    # SAN（Subject Alternative Names）
    sans = []
    for type_name, value in cert.get('subjectAltName', []):
        if type_name == 'DNS':
            sans.append(value)

    # 発行者
    issuer = dict(x[0] for x in cert['issuer'])

    return {
        'hostname': hostname,
        'issuer': issuer.get('organizationName', 'Unknown'),
        'not_before': cert['notBefore'],
        'not_after': cert['notAfter'],
        'days_left': days_left,
        'sans': sans,
        'serial_number': cert.get('serialNumber', ''),
        'version': cert.get('version', ''),
        'tls_version': ssock.version(),
        'cipher': ssock.cipher(),
    }

# 使用例
domains = ['example.com', 'api.example.com', 'www.example.com']
for domain in domains:
    try:
        info = check_certificate(domain)
        status = "OK" if info['days_left'] > 30 else "WARNING"
        print(f"[{status}] {domain}: 残り{info['days_left']}日, "
              f"発行者={info['issuer']}, TLS={info['tls_version']}")
    except Exception as e:
        print(f"[ERROR] {domain}: {e}")
```

---

## 10. HSTS（HTTP Strict Transport Security）

```
HSTS:
  → ブラウザにHTTPS接続を強制する仕組み
  → HTTPへのダウングレードを防止

ヘッダー:
  Strict-Transport-Security: max-age=63072000; includeSubDomains; preload

  max-age: HTTPSを強制する期間（秒）
    → 63072000 = 2年
    → 最低でも31536000（1年）を推奨

  includeSubDomains: サブドメインも対象
    → 全サブドメインがHTTPS対応必要

  preload: HSTS Preload Listに登録
    → ブラウザに事前組み込み
    → 初回アクセスからHTTPS強制

HSTS Preload:
  → https://hstspreload.org/ で登録申請
  → Chrome, Firefox, Safari, Edge に組み込み
  → 一度登録すると解除が困難（数ヶ月〜1年）

  登録要件:
  ① 有効なSSL証明書
  ② HTTPからHTTPSへリダイレクト
  ③ 全サブドメインがHTTPS
  ④ max-age が31536000以上
  ⑤ includeSubDomains ディレクティブ
  ⑥ preload ディレクティブ

注意事項:
  → 段階的にmax-ageを増やす
     300 → 86400 → 604800 → 2592000 → 63072000
  → includeSubDomainsは全サブドメインの確認後に追加
  → preloadは十分テスト後に有効化
  → 設定ミスでサイトにアクセス不能になるリスク
```

---

## 11. TLSのパフォーマンス最適化

```
TLSのパフォーマンス最適化:

① セッション再開:
  → Session Ticket: サーバーステートレス
  → 1-RTTハンドシェイクに短縮

② TLS 1.3 0-RTT:
  → 再接続時の最初のリクエストが0-RTT
  → ただしリプレイリスク

③ OCSP Stapling:
  → クライアントのOCSP問い合わせを削減
  → ハンドシェイク時間の短縮

④ 証明書の最適化:
  → ECDSA証明書（RSAより小さい）
     RSA 2048: 約256バイト
     ECDSA P-256: 約64バイト
  → 証明書チェーンの最小化
  → 不要な中間CA証明書を含めない

⑤ TLS Record Size:
  → 初回は小さなレコードサイズ（1369バイト）
  → TCPスロースタートとの整合
  → 安定後に大きなレコードサイズ（16KB）
  → Dynamic Record Sizingの活用

⑥ TCP Fast Open + TLS 1.3:
  → TCP接続確立とTLSハンドシェイクを並行
  → 合計1-RTTで暗号化通信開始

⑦ ハードウェアアクセラレーション:
  → AES-NI: AES暗号化のハードウェア支援
  → QAT（Quick Assist Technology）: Intelのオフロードデバイス
  → SSL/TLS専用ハードウェア（HSM）

パフォーマンス計測:
  $ curl -o /dev/null -s -w "\
    TCP:  %{time_connect}s\n\
    TLS:  %{time_appconnect}s\n\
    TTFB: %{time_starttransfer}s\n" \
    https://example.com

  TLS handshake目標:
  TLS 1.2: < 100ms（同一リージョン）
  TLS 1.3: < 50ms（同一リージョン）
  TLS 1.3 0-RTT: < 10ms（再接続時）
```

---

## 12. HTTP/2, HTTP/3とTLS

```
HTTP/2 と TLS:
  → HTTP/2はTLS 1.2以上が必須（事実上）
  → ALPN（Application-Layer Protocol Negotiation）で HTTP/2 をネゴシエート
  → ALPNはTLSハンドシェイク中に行われる（追加RTT不要）

  ClientHello:
    ALPN: h2, http/1.1
  ServerHello:
    ALPN: h2
  → HTTP/2で通信開始

HTTP/3 と QUIC:
  → HTTP/3はQUIC上で動作
  → QUIC = UDP上のトランスポートプロトコル
  → TLS 1.3がQUICに組み込み（別プロトコルではない）

  TCP + TLS 1.3:
    TCP 3-way handshake: 1 RTT
    TLS handshake:       1 RTT
    合計:                2 RTT

  QUIC:
    QUIC handshake（TLS 1.3組み込み）: 1 RTT
    合計: 1 RTT

  QUIC 0-RTT:
    再接続時: 0 RTT
    → 接続確立と同時にデータ送信

  QUICのメリット:
  → Head-of-Line Blocking の解消
  → コネクションマイグレーション（IP変更に対応）
  → 組み込みの暗号化（平文通信不可）
  → ユーザー空間での実装（カーネル変更不要）

  Nginx QUIC設定:
  server {
      listen 443 quic reuseport;
      listen 443 ssl;
      http2 on;
      add_header Alt-Svc 'h3=":443"; ma=86400';
      # ...TLS設定は通常と同じ
  }
```

---

## 13. 実務でのトラブルシューティング

```
よくあるTLS問題と解決策:

① 証明書エラー: "NET::ERR_CERT_AUTHORITY_INVALID"
  原因: 中間CA証明書の未送信
  確認: openssl s_client -connect host:443 -showcerts
  解決: fullchain.pemを使用（中間CA含む）

② 証明書エラー: "NET::ERR_CERT_DATE_INVALID"
  原因: 証明書の期限切れ
  確認: openssl x509 -noout -dates
  解決: 証明書の更新、自動更新の設定

③ Mixed Content:
  原因: HTTPSページ内にHTTPリソース
  確認: Chrome DevTools Console
  解決: 全リソースをHTTPS化、CSPのupgrade-insecure-requests

④ HSTS関連エラー:
  原因: 証明書が無効なのにHSTSでHTTPS強制
  確認: chrome://net-internals/#hsts
  解決: 証明書の修正、HSSTSエントリの削除

⑤ SNI関連エラー:
  原因: 古いクライアントがSNI非対応
  確認: openssl s_client -connect host:443 -servername host
  解決: デフォルト証明書の設定、古いクライアントの切り捨て

⑥ Protocol/Cipher Mismatch:
  原因: クライアントとサーバーで共通の暗号スイートなし
  確認: sslyze / testssl.sh で対応暗号確認
  解決: 暗号スイート設定の見直し

⑦ TLS接続タイムアウト:
  原因: ファイアウォール、プロキシ、MTU問題
  確認: tcpdump でパケットキャプチャ
  解決: ファイアウォールルール確認、MTU調整

⑧ クライアント証明書エラー（mTLS）:
  原因: クライアント証明書がない、または無効
  確認: curl --cert client.crt --key client.key https://...
  解決: クライアント証明書の配布、CA証明書の設定
```

---

## 14. mTLS（相互TLS認証）の実践

```
mTLS（mutual TLS）:
  → 通常のTLSに加えて、クライアントも証明書で認証
  → サーバー → クライアント認証 + クライアント → サーバー認証

  通常のTLS:
  クライアント ───────→ サーバー
               ← サーバー証明書
  「サーバーは本物か？」のみ検証

  mTLS:
  クライアント ───────→ サーバー
               ← サーバー証明書
               → クライアント証明書
  「サーバーは本物か？」+ 「クライアントは本物か？」を相互検証

  用途:
  ① マイクロサービス間通信: サービス同士の認証
  ② IoTデバイス認証: デバイス証明書による接続制御
  ③ VPNの代替: BeyondCorp / ゼロトラスト
  ④ 金融系API: 強固な認証が必要なケース
```

### 14.1 mTLS証明書の作成と設定

```bash
# プライベートCA（Root CA）の作成
# 1. ルートCA秘密鍵の生成
$ openssl ecparam -genkey -name prime256v1 -out root-ca.key

# 2. ルートCA証明書の生成（10年間有効）
$ openssl req -new -x509 -sha256 -key root-ca.key \
  -out root-ca.crt -days 3650 \
  -subj "/C=JP/O=MyOrg/CN=MyOrg Root CA"

# サーバー証明書の作成
# 3. サーバー秘密鍵
$ openssl ecparam -genkey -name prime256v1 -out server.key

# 4. CSR（証明書署名要求）の作成
$ openssl req -new -sha256 -key server.key -out server.csr \
  -subj "/C=JP/O=MyOrg/CN=api.example.com"

# 5. サーバー証明書の署名（SANを含む）
$ cat > server-ext.cnf << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment
subjectAltName = @alt_names
[alt_names]
DNS.1 = api.example.com
DNS.2 = *.api.example.com
EOF

$ openssl x509 -req -sha256 -in server.csr -CA root-ca.crt \
  -CAkey root-ca.key -CAcreateserial -out server.crt \
  -days 365 -extfile server-ext.cnf

# クライアント証明書の作成
# 6. クライアント秘密鍵
$ openssl ecparam -genkey -name prime256v1 -out client.key

# 7. クライアントCSR
$ openssl req -new -sha256 -key client.key -out client.csr \
  -subj "/C=JP/O=MyOrg/CN=service-a"

# 8. クライアント証明書の署名
$ cat > client-ext.cnf << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature
extendedKeyUsage = clientAuth
EOF

$ openssl x509 -req -sha256 -in client.csr -CA root-ca.crt \
  -CAkey root-ca.key -CAcreateserial -out client.crt \
  -days 365 -extfile client-ext.cnf
```

### 14.2 Nginx での mTLS 設定

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    # サーバー証明書
    ssl_certificate     /etc/nginx/certs/server.crt;
    ssl_certificate_key /etc/nginx/certs/server.key;

    # mTLS設定
    ssl_client_certificate /etc/nginx/certs/root-ca.crt;  # クライアント証明書のCA
    ssl_verify_client on;           # クライアント証明書を必須化
    ssl_verify_depth 2;             # 証明書チェーンの深さ

    # クライアント証明書情報をバックエンドに転送
    location / {
        proxy_pass http://backend;
        proxy_set_header X-Client-DN $ssl_client_s_dn;
        proxy_set_header X-Client-Serial $ssl_client_serial;
        proxy_set_header X-Client-Verify $ssl_client_verify;
    }

    # 証明書検証に失敗した場合
    # ssl_verify_client on; の場合、Nginx が 400 Bad Request を返す
    # ssl_verify_client optional; にすると検証はするが必須でない
}
```

### 14.3 Node.js での mTLS 接続

```javascript
const https = require('https');
const fs = require('fs');

// mTLSサーバー
const serverOptions = {
  key: fs.readFileSync('/path/to/server.key'),
  cert: fs.readFileSync('/path/to/server.crt'),
  ca: fs.readFileSync('/path/to/root-ca.crt'),  // クライアント証明書のCA
  requestCert: true,       // クライアント証明書を要求
  rejectUnauthorized: true // 検証失敗時に拒否
};

const server = https.createServer(serverOptions, (req, res) => {
  const clientCert = req.socket.getPeerCertificate();
  console.log('Client CN:', clientCert.subject.CN);
  console.log('Client Org:', clientCert.subject.O);
  res.writeHead(200);
  res.end(`Hello, ${clientCert.subject.CN}!`);
});

server.listen(443);

// mTLSクライアント
const clientOptions = {
  hostname: 'api.example.com',
  port: 443,
  path: '/api/data',
  method: 'GET',
  key: fs.readFileSync('/path/to/client.key'),
  cert: fs.readFileSync('/path/to/client.crt'),
  ca: fs.readFileSync('/path/to/root-ca.crt')
};

const clientReq = https.request(clientOptions, (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => console.log('Response:', data));
});

clientReq.end();
```

---

## 15. TLSセキュリティの監査とテスト

### 15.1 自動セキュリティ監査ツール

```bash
# testssl.sh — 包括的なTLS設定テスト
$ git clone --depth 1 https://github.com/drwetter/testssl.sh.git
$ cd testssl.sh
$ ./testssl.sh https://example.com

# 出力例:
#  Testing protocols via sockets
#  SSLv2      not offered (OK)
#  SSLv3      not offered (OK)
#  TLS 1      not offered (OK)
#  TLS 1.1    not offered (OK)
#  TLS 1.2    offered (OK)
#  TLS 1.3    offered (OK)
#
#  Testing vulnerabilities
#  Heartbleed     not vulnerable (OK)
#  CCS            not vulnerable (OK)
#  Ticketbleed    not vulnerable (OK)
#  ROBOT          not vulnerable (OK)
#  CRIME          not vulnerable (OK)
#  BREACH         potentially vulnerable (WARN)
#  POODLE         not vulnerable (OK)
#  DROWN          not vulnerable (OK)
#  LOGJAM         not vulnerable (OK)
#  BEAST          not vulnerable (OK)

# sslyze — Python製のTLSスキャナー
$ pip install sslyze
$ sslyze example.com

# SSL Labs API — Webベースの詳細スキャン
$ curl "https://api.ssllabs.com/api/v3/analyze?host=example.com" | jq .

# nmap — TLS暗号スイートの列挙
$ nmap --script ssl-enum-ciphers -p 443 example.com
```

### 15.2 TLSセキュリティチェックリスト

```
TLS設定チェックリスト（2024年版）:

プロトコル:
  [x] TLS 1.3 を有効化
  [x] TLS 1.2 を有効化（互換性のため）
  [x] TLS 1.1 以下を無効化
  [x] SSLv2, SSLv3 を無効化

暗号スイート:
  [x] AEADのみ許可（AES-GCM, ChaCha20-Poly1305）
  [x] 前方秘匿性を提供する鍵交換のみ（ECDHE）
  [x] RSA鍵交換を無効化
  [x] 3DES, RC4, DES を無効化
  [x] CBC モードを無効化（推奨）

証明書:
  [x] RSA 2048bit以上 or ECDSA P-256以上
  [x] SHA-256以上の署名アルゴリズム
  [x] 中間CA証明書を含むフルチェーン
  [x] SANにワイルドカードでなく具体的なドメインを列挙（可能な場合）
  [x] 自動更新の設定と監視
  [x] Certificate Transparency の有効化

セキュリティヘッダー:
  [x] HSTS: max-age=31536000; includeSubDomains; preload
  [x] HSTS Preload List への登録
  [x] Expect-CT（レポートモード）

パフォーマンス:
  [x] OCSP Stapling の有効化
  [x] TLSセッション再開（Session Ticket / Session ID）
  [x] TLS 1.3の0-RTT（安全なリクエストのみ）
  [x] 適切なTLSレコードサイズ

運用:
  [x] 秘密鍵の安全な保管（ファイルパーミッション 600）
  [x] 証明書期限の監視とアラート
  [x] 定期的なセキュリティスキャン（SSL Labs A+）
  [x] インシデント時の証明書失効手順の整備
```

---

## 16. ポスト量子暗号とTLSの未来

```
量子コンピュータの脅威:

現在の公開鍵暗号:
  RSA: 素因数分解問題に基づく → Shorのアルゴリズムで解読可能
  ECDHE: 楕円曲線離散対数問題 → Shorのアルゴリズムで解読可能
  → 十分な量子ビット数が実現された場合、現在のTLSは破綻

タイムライン（推定）:
  2024-2030: 量子コンピュータの発展期
  2030-2040: 暗号関連の量子コンピュータが実用化の可能性
  現在: "Harvest Now, Decrypt Later" 攻撃への対策が急務
    → 現在暗号化された通信を保存し、将来量子コンピュータで解読

NIST ポスト量子暗号標準（2024年策定）:

① ML-KEM（Module-Lattice-Based Key Encapsulation）:
  → 旧称: CRYSTALS-Kyber
  → 鍵交換用（TLSのECDHEの代替）
  → 格子問題に基づく → 量子コンピュータでも解読困難
  → Chromeが実験的に ML-KEM-768 を実装済み

② ML-DSA（Module-Lattice-Based Digital Signature）:
  → 旧称: CRYSTALS-Dilithium
  → デジタル署名用（証明書の署名の代替）

③ SLH-DSA（Stateless Hash-Based Digital Signature）:
  → 旧称: SPHINCS+
  → ハッシュベースの署名（保守的な選択肢）

ハイブリッド方式（移行期の推奨アプローチ）:
  → 従来の暗号 + ポスト量子暗号を組み合わせ
  → どちらか一方が破られても安全

  TLS 1.3 での実装:
  鍵交換: X25519 + ML-KEM-768（X25519MLKEM768）
  → 両方の共有秘密を結合して最終鍵を導出
  → 従来の安全性は維持しつつ量子耐性を追加

  Chrome/Edge の対応状況（2024年〜）:
  → X25519MLKEM768 をデフォルトで有効化
  → ハンドシェイクサイズが約1KB増加
  → 一部の古いミドルボックスで接続問題の報告あり

実務での推奨対応:
  今すぐ:
  ✓ 暗号アジリティの確保（暗号スイートの変更が容易な設計）
  ✓ TLS 1.3の完全移行
  ✓ 長期保存データの暗号化レベルの見直し

  2025-2030:
  ✓ ハイブリッド方式のテスト導入
  ✓ ライブラリ/フレームワークのPQC対応状況の監視
  ✓ HSMのポスト量子対応の確認

  2030以降:
  ✓ 完全なポスト量子暗号への移行
  ✓ 古い暗号スイートの廃止
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| TLS | 暗号化 + 認証 + 完全性 |
| TLS 1.2 | 2 RTT ハンドシェイク |
| TLS 1.3 | 1 RTT（0-RTT再接続）、脆弱暗号廃止 |
| 証明書 | X.509、CA署名チェーン、DV/OV/EV |
| 前方秘匿性 | ECDHE で過去の通信を保護 |
| 暗号スイート | AES-GCM / ChaCha20-Poly1305、AEAD必須 |
| HSTS | HTTPS強制、Preload Listで初回から保護 |
| mTLS | 相互認証、マイクロサービス間通信 |
| CT | 証明書発行の透明性、不正発行の検知 |
| HTTP/3 | QUIC上でTLS 1.3組み込み、1-RTT |
| ポスト量子 | ML-KEM/ML-DSA、ハイブリッド方式で先行対応 |

---

## FAQ

### Q1: TLS 1.2とTLS 1.3の主な違いは何ですか？

**A1: パフォーマンス、セキュリティ、シンプルさが大幅に改善されています。**

主要な違い:

1. **ハンドシェイクの高速化**:
   - TLS 1.2: 2 RTT（往復通信が2回必要）
   - TLS 1.3: 1 RTT（往復通信が1回）
   - TLS 1.3（再接続）: 0-RTT（データと同時送信可能）
   - 実測: TLS 1.2で200ms → TLS 1.3で100ms（東京-US間）

2. **暗号化範囲の拡大**:
   - TLS 1.2: アプリケーションデータのみ暗号化
   - TLS 1.3: ハンドシェイクの大部分も暗号化（証明書が見えない）
   - プライバシー向上: 接続先サーバーの証明書が傍受者に見えない

3. **脆弱な暗号の廃止**:
   - RSA鍵交換（前方秘匿性なし）→ 廃止
   - CBC モード（パディングオラクル攻撃）→ 廃止
   - 圧縮（CRIME攻撃）→ 廃止
   - 再ネゴシエーション → 廃止
   - → 常にECDHEによる前方秘匿性、AEADによる認証付き暗号

4. **鍵導出の改善**:
   - TLS 1.2: PRF（Pseudo-Random Function）
   - TLS 1.3: HKDF（HMAC-based KDF）— より明確で安全な設計
   - 段階的鍵導出: 0-RTT鍵、ハンドシェイク鍵、アプリケーション鍵が分離

5. **選択肢の削減によるセキュリティ向上**:
   - TLS 1.2: 約40種類の暗号スイート
   - TLS 1.3: 5種類のみ（TLS_AES_128_GCM_SHA256等）
   - → 設定ミスのリスク低減、監査が容易

**実務での推奨**: TLS 1.3をメインに、互換性のためTLS 1.2も許可。TLS 1.0/1.1は無効化。

---

### Q2: 証明書チェーンの仕組みと、なぜ中間CA証明書が必要なのですか？

**A2: 信頼の連鎖を構築しつつ、ルートCAの秘密鍵を安全に保つための設計です。**

証明書チェーンの構造:
```
┌────────────────┐
│ ルートCA証明書  │ ← OS/ブラウザに内蔵（約150-200個）
│ (自己署名)      │    秘密鍵はオフライン保管（HSM内、物理的隔離）
└───────┬────────┘    侵害されると全インターネットが影響
        │ 署名
┌───────▼────────┐
│ 中間CA証明書    │ ← ルートCAが署名
│               │    秘密鍵はオンライン（証明書発行に使用）
└───────┬────────┘    侵害されてもルート失効で対処可能
        │ 署名
┌───────▼────────┐
│ サーバー証明書   │ ← 中間CAが署名（Let's Encrypt等）
│ example.com    │    有効期限: DV証明書は通常90日
└────────────────┘
```

中間CAが必要な理由:

1. **ルートCAの保護**:
   - ルートCAの秘密鍵は厳重に隔離（エアギャップ環境）
   - 証明書発行の度にルートCAにアクセスするのは非現実的
   - 中間CAがオンラインで発行処理を担当

2. **リスクの分散**:
   - 中間CAが侵害されても、その中間CAのみ失効すればよい
   - ルートCAが侵害されると、全ての派生証明書が無効に
   - 実例: 2011年のDigiNotar侵害 → ルートCA失効 → ブラウザから削除

3. **柔軟な運用**:
   - 用途別・地域別に複数の中間CAを使い分け
   - DV専用、EV専用の中間CAで分離
   - 中間CAの有効期限更新が容易

検証プロセス（ブラウザ側）:
```
1. サーバー証明書の署名を中間CAの公開鍵で検証
2. 中間CA証明書の署名をルートCAの公開鍵で検証
3. ルートCA証明書がOS/ブラウザの信頼ストアに存在するか確認
4. 各証明書の有効期限を確認
5. ドメイン名（SAN）の一致を確認
6. 失効確認（OCSP Stapling / CRL / CRLite）
7. Certificate Transparencyログの確認
```

よくある問題と対処法:

**問題1: 中間CA証明書の送信漏れ**
```bash
# 症状: "信頼されていない証明書" エラー
# 原因: サーバーがサーバー証明書のみ送信、中間CAを含まない
# 確認:
$ openssl s_client -connect example.com:443 -showcerts

# 対処: fullchain.pem を使用（サーバー証明書+中間CA証明書）
ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
# cert.pemではなくfullchain.pemを指定
```

**問題2: 証明書の順序が逆**
```
正しい順序: サーバー証明書 → 中間CA証明書 → ルートCA証明書（省略可）
誤った順序: 中間CA → サーバー証明書 → エラー
```

**クロスサイン**:
新しいルートCAの普及期に、古いルートCAにも署名してもらうテクニック。
- Let's Encrypt: ISRG Root X1（新）+ DST Root CA X3（旧）でクロスサイン
- 古いAndroidデバイスでもDST経由で信頼チェーン成立
- 2024年現在はISRG Root X1がほぼ全デバイスで信頼済み

---

### Q3: Let's Encryptの仕組みと実務での使い方を教えてください。

**A3: ACMEプロトコルによる自動化されたドメイン検証で、無料のDV証明書を90日間発行します。**

Let's Encryptの特徴:
- **完全無料**: DV（ドメイン検証）証明書
- **自動化**: certbotで取得・更新を自動化
- **有効期限**: 90日（意図的に短く設定 → 自動更新の促進）
- **信頼**: ISRG（Internet Security Research Group）運営、主要ブラウザで信頼済み
- **制限**: ワイルドカード証明書はDNSチャレンジのみ、レート制限あり

ACMEプロトコルのフロー:

```
1. アカウント登録
   クライアント → Let's Encrypt: 公開鍵を送信
   → アカウントURLを取得

2. 証明書リクエスト
   クライアント → Let's Encrypt: example.comの証明書をリクエスト

3. チャレンジ提示
   Let's Encrypt → クライアント: 以下のいずれかで認証せよ

   ① HTTP-01チャレンジ:
      http://example.com/.well-known/acme-challenge/<TOKEN>
      に特定の内容を配置
      → Let's Encryptがアクセスして確認
      → ポート80が必要、ワイルドカード不可

   ② DNS-01チャレンジ:
      _acme-challenge.example.com TXT "<TOKEN>"
      にDNSレコードを設定
      → Let's EncryptがDNS問い合わせで確認
      → ワイルドカード証明書可能、ポート80不要
      → DNSプロバイダーのAPI必要（自動化のため）

   ③ TLS-ALPN-01チャレンジ:
      TLSハンドシェイク中にトークンを含む証明書を提示
      → HTTP-01の代替、ポート443のみ使用

4. チャレンジ応答
   クライアント: トークンを配置 → Let's Encryptに通知

5. 検証
   Let's Encrypt: トークンを確認 → 成功

6. 証明書発行
   クライアント: CSR（Certificate Signing Request）を送信
   Let's Encrypt: 署名済み証明書を返却
```

実務での設定例:

```bash
# certbotインストール（Ubuntu）
$ sudo apt update
$ sudo apt install certbot python3-certbot-nginx

# Nginx用の証明書取得（自動設定）
$ sudo certbot --nginx -d example.com -d www.example.com
# → Nginxの設定ファイルを自動編集
# → 証明書を/etc/letsencrypt/live/example.com/に配置

# 証明書のみ取得（手動設定）
$ sudo certbot certonly --webroot -w /var/www/html \
  -d example.com -d www.example.com

# ワイルドカード証明書（DNSチャレンジ必須）
$ sudo certbot certonly --manual --preferred-challenges dns \
  -d '*.example.com' -d example.com
# → DNSレコード設定を指示される:
#   _acme-challenge.example.com TXT "abc123..."
# → 設定後にEnterで検証

# 自動更新の設定（systemd timer、Ubuntuでは自動設定済み）
$ systemctl list-timers | grep certbot
# または cron
$ sudo crontab -e
0 0,12 * * * certbot renew --quiet --post-hook "systemctl reload nginx"
# → 1日2回チェック、期限30日以内なら更新

# 更新のテスト（実際には更新しない）
$ sudo certbot renew --dry-run

# 証明書の確認
$ sudo certbot certificates
Found the following certs:
  Certificate Name: example.com
    Domains: example.com www.example.com
    Expiry Date: 2024-03-31 12:00:00+00:00 (VALID: 89 days)
    Certificate Path: /etc/letsencrypt/live/example.com/fullchain.pem
    Private Key Path: /etc/letsencrypt/live/example.com/privkey.pem
```

Nginx設定（手動の場合）:
```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    # Let's Encrypt証明書
    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # ACME チャレンジパス（更新時に必要）
    location /.well-known/acme-challenge/ {
        root /var/www/html;
        allow all;
    }
}

# HTTPからHTTPSへリダイレクト
server {
    listen 80;
    server_name example.com;

    # ACMEチャレンジのみ許可
    location /.well-known/acme-challenge/ {
        root /var/www/html;
        allow all;
    }

    # それ以外はHTTPSへ
    location / {
        return 301 https://$host$request_uri;
    }
}
```

レート制限（2024年時点）:
- 登録ドメイン/週: 50証明書
- 重複証明書/週: 5証明書
- アカウント登録: 10アカウント/3時間/IP
- → 本番前にステージング環境で十分テスト

ステージング環境:
```bash
$ sudo certbot --nginx --staging \
  -d example.com -d www.example.com
# → レート制限が緩い、信頼されない証明書（テスト用）
```

トラブルシューティング:
```bash
# ログ確認
$ sudo tail -f /var/log/letsencrypt/letsencrypt.log

# 証明書の手動削除
$ sudo certbot delete --cert-name example.com

# Nginxの設定確認
$ sudo nginx -t

# ポート80が開いているか確認（HTTP-01チャレンジ）
$ curl http://example.com/.well-known/acme-challenge/test
```

代替クライアント:
- acme.sh: シェルスクリプトのみ、依存なし
- Caddy: Webサーバーに証明書取得機能を内蔵
- Traefik: リバースプロキシに証明書取得機能を内蔵

---

## 次に読むべきガイド
→ [[01-authentication.md]] — 認証方式

---

## 参考文献
1. RFC 8446. "The Transport Layer Security (TLS) Protocol Version 1.3." IETF, 2018.
2. RFC 5246. "The Transport Layer Security (TLS) Protocol Version 1.2." IETF, 2008.
3. RFC 8996. "Deprecating TLS 1.0 and TLS 1.1." IETF, 2021.
4. RFC 6960. "X.509 Internet Public Key Infrastructure Online Certificate Status Protocol - OCSP." IETF, 2013.
5. RFC 6962. "Certificate Transparency." IETF, 2013.
6. RFC 6797. "HTTP Strict Transport Security (HSTS)." IETF, 2012.
7. Mozilla. "Security/Server Side TLS." Mozilla Wiki, 2024.
8. SSL Labs. "SSL/TLS Deployment Best Practices." Qualys, 2024.
9. NIST. "SP 800-52 Rev. 2: Guidelines for the Selection, Configuration, and Use of TLS Implementations." 2019.
10. Cloudflare. "What is TLS?" Cloudflare Learning Center, 2024.
