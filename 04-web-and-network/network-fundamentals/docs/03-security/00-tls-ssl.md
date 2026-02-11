# TLS/SSL

> TLSはインターネット通信の暗号化を担うプロトコル。ハンドシェイク、証明書、暗号スイートの仕組みを理解し、安全な通信の基盤を学ぶ。

## この章で学ぶこと

- [ ] TLSハンドシェイクの流れを理解する
- [ ] 証明書の仕組みと検証プロセスを把握する
- [ ] TLS 1.3の改善点を学ぶ

---

## 1. TLSの基本

```
TLS（Transport Layer Security）:
  → 通信の暗号化、認証、完全性を提供
  → HTTPS = HTTP + TLS

歴史:
  SSL 2.0 (1995) → 脆弱、使用禁止
  SSL 3.0 (1996) → POODLE攻撃、使用禁止
  TLS 1.0 (1999) → 非推奨
  TLS 1.1 (2006) → 非推奨
  TLS 1.2 (2008) → 現在広く使用
  TLS 1.3 (2018) → 最新、推奨

  ※ SSLは全バージョン廃止。「SSL証明書」は慣習的な名称

TLSが提供する3つの機能:
  ① 機密性（Confidentiality）: 通信内容の暗号化
  ② 認証（Authentication）: 通信相手の本人確認
  ③ 完全性（Integrity）: データの改ざん検知
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
       │                              │
  ②    │←── ServerHello ──           │
       │   選択されたTLSバージョン     │
       │   選択された暗号スイート      │
       │   サーバーランダム            │
       │                              │
  ③    │←── Certificate ──           │
       │   サーバー証明書              │
       │                              │
  ④    │←── ServerKeyExchange ──     │
       │   鍵交換パラメータ(DH等)      │
       │                              │
  ⑤    │←── ServerHelloDone ──       │
       │                              │
  ⑥    │── ClientKeyExchange ──→     │
       │   鍵交換パラメータ            │
       │                              │
  ⑦    │── ChangeCipherSpec ──→      │
       │── Finished ──→               │
       │                              │
  ⑧    │←── ChangeCipherSpec ──      │
       │←── Finished ──               │
       │                              │
       │   暗号化通信開始              │

  所要時間: 2 RTT（東京-US間: 約200ms）
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
       │                              │
  ②    │←── ServerHello ──           │
       │   選択された暗号スイート      │
       │   鍵共有パラメータ            │
       │   {Certificate}（暗号化済み） │
       │   {Finished}                  │
       │                              │
  ③    │── {Finished} ──→             │
       │                              │
       │   暗号化通信開始              │

  改善点:
  ① 1 RTT: 鍵交換を最初から送信（推測ベース）
  ② 0-RTT: 再接続時はデータも同時送信可能
  ③ 脆弱な暗号スイートを廃止
  ④ ハンドシェイクも暗号化（証明書が見えない）

  0-RTT 再接続:
  クライアント ── ClientHello + アプリデータ ──→ サーバー
  → 接続確立とデータ送信を同時に
  → ただしリプレイ攻撃のリスクあり（冪等なリクエストのみ推奨）

  TLS 1.3で廃止された暗号:
  RSA鍵交換（前方秘匿性なし）
  CBC モードの暗号
  RC4, DES, 3DES
  MD5, SHA-1
  静的DH
```

---

## 4. 証明書

```
X.509 証明書の構造:
  ┌───────────────────────────────────────┐
  │ サブジェクト: example.com              │
  │ 発行者: Let's Encrypt Authority X3     │
  │ 有効期間: 2024-01-01 〜 2024-03-31    │
  │ 公開鍵: RSA 2048bit / ECDSA P-256     │
  │ SANs: example.com, www.example.com    │
  │ 署名: SHA-256 with RSA                │
  └───────────────────────────────────────┘

証明書チェーン:
  ┌────────────────┐
  │ ルートCA証明書   │ ← OSやブラウザに内蔵（信頼の起点）
  │ (自己署名)      │
  └───────┬────────┘
          │ 署名
  ┌───────▼────────┐
  │ 中間CA証明書    │ ← ルートCAが署名
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

証明書の種類:
  DV（ドメイン検証）: ドメインの所有のみ確認。Let's Encrypt等
  OV（組織検証）: 組織の実在を確認。企業サイト向け
  EV（拡張検証）: 厳格な組織審査。金融機関等（ブラウザ表示の差は縮小傾向）

Let's Encrypt:
  → 無料DV証明書
  → 自動更新（certbot）
  → 有効期間90日（短いほど安全）
  → ACME プロトコルで自動発行
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

TLS 1.3の暗号スイート（簡素化）:
  TLS_AES_256_GCM_SHA384
  TLS_AES_128_GCM_SHA256
  TLS_CHACHA20_POLY1305_SHA256
  → 鍵交換は常にECDHE（前方秘匿性あり）
  → 認証は証明書で別途処理
  → 選択肢を減らしてセキュリティ向上

前方秘匿性（Forward Secrecy）:
  → サーバーの秘密鍵が漏洩しても、過去の通信は復号できない
  → 毎回新しい一時鍵（ephemeral key）を生成
  → ECDHE: Elliptic Curve Diffie-Hellman Ephemeral
```

---

## 6. 実務での確認コマンド

```bash
# TLS接続の詳細確認
$ openssl s_client -connect example.com:443

# 証明書チェーンの表示
$ openssl s_client -connect example.com:443 -showcerts

# 証明書の内容を確認
$ echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -text -noout

# TLSバージョンとスイートの確認
$ curl -v https://example.com 2>&1 | grep -E '(SSL|TLS)'

# nmap でTLS設定をスキャン
$ nmap --script ssl-enum-ciphers -p 443 example.com

# SSL Labs でオンラインチェック
# https://www.ssllabs.com/ssltest/
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

---

## 次に読むべきガイド
→ [[01-authentication.md]] — 認証方式

---

## 参考文献
1. RFC 8446. "The Transport Layer Security (TLS) Protocol Version 1.3." IETF, 2018.
2. RFC 5246. "The Transport Layer Security (TLS) Protocol Version 1.2." IETF, 2008.
