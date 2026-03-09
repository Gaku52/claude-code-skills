# DNS（Domain Name System）

> DNSはインターネットの根幹を支える分散データベースシステムである。人間が記憶しやすいドメイン名を、コンピュータが通信に使用するIPアドレスへ変換する「名前解決」を担う。本ガイドでは、再帰/反復クエリ、DNSレコードの種類、キャッシュ機構、セキュリティ拡張、そして運用上の設計判断までを体系的に解説する。

---

## この章で学ぶこと

- [ ] DNSの階層的な分散構造と名前空間の仕組みを理解する
- [ ] 再帰クエリと反復クエリの違いを正確に説明できるようになる
- [ ] 主要なDNSレコード（A, AAAA, CNAME, MX, TXT, NS, SOA, SRV, PTR, CAA）の用途を把握する
- [ ] DNSキャッシュとTTLの設計判断ができるようになる
- [ ] dig / nslookup / host コマンドで実践的なDNSデバッグができるようになる
- [ ] DNSSEC、DoH、DoTなどのセキュリティ拡張の概要を理解する
- [ ] DNS運用におけるアンチパターンとベストプラクティスを学ぶ

## 前提知識

- IPアドレスの基礎（./02-ip-addressing.md）
- UDPプロトコルの概要（コネクションレス型のトランスポートプロトコル）
- コマンドライン（ターミナル）の基本操作

---

## 1. DNSの基本概念

### 1.1 なぜDNSが必要なのか

インターネット上のすべての通信は、最終的にIPアドレスで相手を特定する。しかし、人間が `93.184.216.34` のような数字列を記憶して使い分けることは非現実的である。DNS（Domain Name System）は、人間にとって意味のある文字列（ドメイン名）と、ネットワーク上の実体（IPアドレス）を結びつけるために設計された。

```
DNSが存在しない世界:
  ブラウザに 93.184.216.34 を入力 → Example社のWebサイトが表示
  ブラウザに 142.250.196.110 を入力 → Googleの検索ページが表示
  ブラウザに 31.13.82.36 を入力    → Facebookが表示

  → 事実上、一般ユーザーにはインターネットは使えない

DNSが存在する世界:
  ブラウザに example.com を入力  → DNSが 93.184.216.34 に変換 → 表示
  ブラウザに google.com を入力  → DNSが 142.250.196.110 に変換 → 表示
  ブラウザに facebook.com を入力 → DNSが 31.13.82.36 に変換 → 表示

  → ドメイン名という「人間向けインターフェース」がインターネットを
    実用的なものにしている
```

DNSは1983年にPaul MockapetrisがRFC 882/883として提案し、その後RFC 1034/1035で標準化された。それ以前は `HOSTS.TXT` という単一のテキストファイルで全インターネットの名前解決を行っていたが、ホスト数の爆発的増加により管理が破綻した。DNSはこの問題を「分散データベース」という設計で解決した。

### 1.2 DNSの階層構造

DNSの名前空間は、ファイルシステムのディレクトリ構造に似た逆ツリー型の階層を持つ。最上位にルート（`.`）があり、そこからTLD（トップレベルドメイン）、セカンドレベルドメイン、サブドメインと枝分かれしていく。

```
DNSの階層構造（名前空間ツリー）:

                          . (ルート)
                          |
          +---------------+---------------+---------------+
          |               |               |               |
         com.            org.            jp.            net.
          |               |               |               |
    +-----+-----+    +---+---+    +------+------+        |
    |     |     |    |       |    |      |      |        |
 example google github  wiki  mozilla  co    ac    go   cloudflare
   .com  .com  .com   .org   .org  .jp   .jp   .jp    .net
                                    |
                              +-----+-----+
                              |           |
                           example     toyota
                            .co.jp      .co.jp

各階層の呼称:
  ルート       → . (ドット)
  TLD         → com, org, jp, net, edu, gov, ...
  SLD         → example, google, github, ...
  サブドメイン → www, mail, api, blog, ...
```

### 1.3 FQDN（完全修飾ドメイン名）

FQDN（Fully Qualified Domain Name）は、ルートからの完全なパスを示すドメイン名である。末尾のドット（`.`）がルートを表す。

```
FQDNの構成要素:

  www.example.com.
  ^^^  ^^^^^^^  ^^^  ^
   |      |      |   |
   |      |      |   +--- ルート（通常は省略される）
   |      |      +------- TLD（トップレベルドメイン）
   |      +-------------- SLD（セカンドレベルドメイン）
   +--------------------- ホスト名（サブドメイン）

具体例:
  FQDN                        ホスト名   ドメイン
  ─────────────────────────────────────────────────
  www.example.com.             www       example.com
  mail.example.co.jp.          mail      example.co.jp
  api.v2.internal.example.com. api       v2.internal.example.com
  example.com.                 (なし)     example.com

注意: 末尾のドットの有無
  "example.com"  → 相対名（/etc/resolv.conf の search ドメインが付与される場合がある）
  "example.com." → 絶対名（FQDN、曖昧さがない）

  DNS設定ファイル（ゾーンファイル等）では末尾ドットの有無が
  致命的な設定ミスの原因になることがある（後述のアンチパターン参照）
```

### 1.4 DNSサーバーの種類

DNS名前解決には複数の種類のサーバーが関与する。それぞれの役割を正確に理解することが重要である。

```
DNSサーバーの分類:

┌─────────────────────────────────────────────────────────────────────┐
│                     DNSサーバーの種類                                │
├─────────────────┬───────────────────────────────────────────────────┤
│ ルートDNSサーバー │ ・DNSツリーの最上位（13クラスタ: a〜m）          │
│                  │ ・TLDサーバーの情報を保持                        │
│                  │ ・Anycastで世界中に分散配置（実体は1000台以上）  │
│                  │ ・例: a.root-servers.net (198.41.0.4)            │
├─────────────────┼───────────────────────────────────────────────────┤
│ TLD DNSサーバー  │ ・各TLD（.com, .org, .jp等）を管理               │
│                  │ ・権威DNSサーバーのNSレコードを返す              │
│                  │ ・例: a.gtld-servers.net（.com担当）             │
├─────────────────┼───────────────────────────────────────────────────┤
│ 権威DNSサーバー  │ ・特定ゾーンの正式なレコードを保持               │
│  (Authoritative) │ ・最終的な回答（Authoritative Answer）を返す     │
│                  │ ・ゾーンの管理者が設定・運用                     │
│                  │ ・例: ns1.example.com                            │
├─────────────────┼───────────────────────────────────────────────────┤
│ フルリゾルバ     │ ・クライアントからの再帰クエリを受け付ける       │
│ (Recursive       │ ・反復クエリで各DNSサーバーに問い合わせ          │
│  Resolver)       │ ・結果をキャッシュして高速化                     │
│                  │ ・ISPが提供、またはパブリックDNS（8.8.8.8等）    │
├─────────────────┼───────────────────────────────────────────────────┤
│ スタブリゾルバ   │ ・クライアントOS内蔵の最小限のDNS機能            │
│ (Stub Resolver)  │ ・設定されたフルリゾルバに再帰クエリを送る       │
│                  │ ・自身では反復クエリを行わない                   │
│                  │ ・/etc/resolv.conf で設定                        │
├─────────────────┼───────────────────────────────────────────────────┤
│ フォワーダー     │ ・受け取ったクエリを別のリゾルバに転送           │
│ (Forwarder)      │ ・企業内ネットワーク等で使用                     │
│                  │ ・内部ゾーンは自身で解決、外部は上位に転送       │
└─────────────────┴───────────────────────────────────────────────────┘
```

---

## 2. 名前解決の詳細フロー

### 2.1 完全な名前解決の過程

ユーザーがブラウザに `www.example.com` を入力してからWebページが表示されるまでに、裏側では複雑なDNS問い合わせが行われている。以下にその全過程を示す。

```
DNS名前解決フロー（完全版）:

  ユーザー        スタブ         フル          ルート       .com TLD      権威DNS
  (ブラウザ)     リゾルバ      リゾルバ        DNS          DNS        (example.com)
     |              |             |              |            |             |
     | ① URL入力    |             |              |            |             |
     |------------->|             |              |            |             |
     |  ブラウザ    |             |              |            |             |
     |  キャッシュ  |             |              |            |             |
     |  確認(miss)  |             |              |            |             |
     |              |             |              |            |             |
     |  ② OS       |             |              |            |             |
     |  キャッシュ  |             |              |            |             |
     |  確認(miss)  |             |              |            |             |
     |              |             |              |            |             |
     |  ③ /etc/hosts|             |              |            |             |
     |  確認(miss)  |             |              |            |             |
     |              |             |              |            |             |
     |              | ④ 再帰クエリ |              |            |             |
     |              |------------>|              |            |             |
     |              |             |              |            |             |
     |              |             | ⑤ 反復クエリ  |            |             |
     |              |             |------------->|            |             |
     |              |             |   ".comはここ" |            |             |
     |              |             |<-------------|            |             |
     |              |             |              |            |             |
     |              |             | ⑥ 反復クエリ               |             |
     |              |             |-------------------------->|             |
     |              |             |   "example.comはここ"      |             |
     |              |             |<--------------------------|             |
     |              |             |              |            |             |
     |              |             | ⑦ 反復クエリ                            |
     |              |             |---------------------------------------->|
     |              |             |   "93.184.216.34"                       |
     |              |             |<----------------------------------------|
     |              |             |              |            |             |
     |              | ⑧ 応答      |              |            |             |
     |              |<------------|              |            |             |
     |              |  (キャッシュ |              |            |             |
     |              |   に保存)   |              |            |             |
     |              |             |              |            |             |
     | ⑨ IPアドレス  |             |              |            |             |
     |<-------------|             |              |            |             |
     |              |             |              |            |             |
     | ⑩ TCP接続開始（93.184.216.34:443）                                   |
     |------------------------------------------------------------------>  |
```

### 2.2 各ステップの詳細

**ステップ1-3: ローカルキャッシュの確認**

名前解決はまずローカルで完結できないかを確認する。ブラウザの内部キャッシュ、OSのDNSキャッシュ、そして `/etc/hosts` ファイルを順に検索する。

```
キャッシュ確認の順序と確認コマンド:

1. ブラウザキャッシュ:
   Chrome:  chrome://net-internals/#dns
   Firefox: about:networking#dns
   → ブラウザごとに独自のキャッシュを保持
   → ブラウザを閉じるとクリアされることが多い

2. OSキャッシュ:
   macOS:   $ sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
   Linux:   $ sudo systemd-resolve --flush-caches    # systemd-resolved使用時
            $ sudo systemctl restart nscd             # nscd使用時
   Windows: > ipconfig /flushdns

3. /etc/hosts ファイル（ローカル名前解決）:
   $ cat /etc/hosts
   127.0.0.1    localhost
   ::1          localhost
   192.168.1.10 myserver.local myserver

   → /etc/hosts はDNSより優先される（nsswitch.conf の設定による）
   → 開発環境でのドメインオーバーライドに使われる
```

**ステップ4: 再帰クエリ（スタブリゾルバ → フルリゾルバ）**

スタブリゾルバはフルリゾルバに対して「最終的な回答」を要求する。これが再帰クエリである。フルリゾルバは回答を得るまで責任を持つ。

**ステップ5-7: 反復クエリ（フルリゾルバ → 各権威サーバー）**

フルリゾルバは、ルートDNSから順に「次に問い合わせるべきサーバー」の情報（リファラル）を受け取りながら、最終的な権威サーバーにたどり着く。

**ステップ8-9: 応答とキャッシュ**

フルリゾルバは得られた回答をTTLに基づいてキャッシュし、クライアントに返す。次回同じ名前の問い合わせがあった場合、キャッシュから直接応答できる。

---

## 3. 再帰クエリと反復クエリの詳細比較

### 3.1 再帰クエリ（Recursive Query）

再帰クエリでは、問い合わせを受けたサーバーが完全な回答を返す責任を負う。クライアントは最終結果を待つだけでよい。

```
再帰クエリの動作:

  クライアント                    フルリゾルバ
     |                               |
     | "www.example.com のIPは？"     |
     |------------------------------>|
     |                               |  ← ここからフルリゾルバが
     |    （クライアントは待機）       |    全ての問い合わせを代行
     |                               |
     |                               |  ルートDNS → .com DNS →
     |                               |  権威DNS と順に問い合わせ
     |                               |
     | "93.184.216.34 です"           |
     |<------------------------------|
     |                               |

特徴:
  ・クライアントの実装がシンプル
  ・フルリゾルバに処理負荷が集中
  ・一般的なクライアント ↔ リゾルバ間で使用
  ・DNSヘッダの RD（Recursion Desired）ビット = 1
```

### 3.2 反復クエリ（Iterative Query）

反復クエリでは、問い合わせを受けたサーバーは自身が知っている範囲の情報を返す。「自分では分からないが、このサーバーなら知っているかもしれない」というリファラル（紹介）を返すことが多い。

```
反復クエリの動作:

  フルリゾルバ            ルートDNS          .com DNS          権威DNS
     |                       |                  |                 |
     | ".comのNSは？"         |                  |                 |
     |---------------------->|                  |                 |
     | "a.gtld-servers.net"  |                  |                 |
     |<----------------------|                  |                 |
     |                       |                  |                 |
     | "example.comのNSは？"                     |                 |
     |----------------------------------------->|                 |
     | "ns1.example.com"                        |                 |
     |<-----------------------------------------|                 |
     |                       |                  |                 |
     | "www.example.comのAは？"                                    |
     |---------------------------------------------------------->|
     | "93.184.216.34"                                            |
     |<----------------------------------------------------------|

特徴:
  ・問い合わせ先サーバーの負荷が低い
  ・フルリゾルバが複数回の問い合わせを行う
  ・リゾルバ ↔ 権威サーバー間で一般的に使用
  ・DNSヘッダの RD ビット = 0
```

### 3.3 再帰クエリと反復クエリの比較表

| 比較項目 | 再帰クエリ | 反復クエリ |
|---------|-----------|-----------|
| 回答責任 | 問い合わせ先が完全な回答を返す義務を負う | 自身が知っている範囲だけ返せばよい |
| クライアント側の処理 | 結果を待つだけ | リファラルを受けて次の問い合わせを行う |
| 主な使用場面 | スタブリゾルバ → フルリゾルバ | フルリゾルバ → 権威サーバー |
| RDビット | 1（再帰を要求） | 0（再帰を要求しない） |
| サーバー負荷 | 問い合わせ先に集中 | 各サーバーに分散 |
| キャッシュ | フルリゾルバがキャッシュ | 各段階でキャッシュ可能 |
| セキュリティリスク | DNS増幅攻撃の踏み台になりうる | リスクが限定的 |
| 応答時間 | クライアントからは単一のRTT | 複数のRTTが発生 |

---

## 4. DNSレコードの種類と詳細

### 4.1 主要レコード一覧

DNSレコードは、ドメイン名に紐づく様々な情報を格納するリソースレコード（RR）である。以下に主要なレコードタイプとその用途を示す。

```
DNSリソースレコードの一般的な書式（ゾーンファイル形式）:

  <名前>    <TTL>   <クラス>  <タイプ>  <データ>

  例:
  www.example.com.  3600  IN  A      93.184.216.34
  example.com.      3600  IN  MX  10 mail.example.com.
  example.com.      3600  IN  TXT    "v=spf1 include:_spf.google.com ~all"

フィールドの説明:
  名前    → レコードが紐づくドメイン名（FQDN）
  TTL     → キャッシュ有効期間（秒）
  クラス  → ほぼ常に IN（Internet）
  タイプ  → レコードの種類（A, AAAA, CNAME, MX, ...）
  データ  → レコード固有の値
```

### 4.2 各レコードタイプの詳細

**Aレコード（Address Record）**

ドメイン名をIPv4アドレスに対応づける最も基本的なレコード。

```
Aレコードの例:

  ゾーンファイル:
  example.com.      300   IN  A  93.184.216.34
  www.example.com.  300   IN  A  93.184.216.34
  api.example.com.  60    IN  A  10.0.1.100
  api.example.com.  60    IN  A  10.0.1.101    # 複数のAレコード（ラウンドロビン）

  digコマンドで確認:
  $ dig example.com A +noall +answer
  example.com.    300  IN  A  93.184.216.34

  複数IPの場合（DNSラウンドロビン）:
  $ dig api.example.com A +noall +answer
  api.example.com.  60  IN  A  10.0.1.100
  api.example.com.  60  IN  A  10.0.1.101
  → クライアントは返されたIPの中から1つを選択して接続
  → 簡易的な負荷分散として機能するが、ヘルスチェックはない
```

**AAAAレコード（IPv6 Address Record）**

ドメイン名をIPv6アドレスに対応づけるレコード。「クアッドA」と読む。

```
AAAAレコードの例:

  ゾーンファイル:
  example.com.  3600  IN  AAAA  2606:2800:0220:0001:0248:1893:25c8:1946

  digコマンドで確認:
  $ dig example.com AAAA +noall +answer
  example.com.  3600  IN  AAAA  2606:2800:220:1:248:1893:25c8:1946

  デュアルスタック環境（IPv4 + IPv6 の両方を設定）:
  example.com.  300  IN  A     93.184.216.34
  example.com.  300  IN  AAAA  2606:2800:220:1:248:1893:25c8:1946
  → Happy Eyeballs アルゴリズムにより、速い方が優先的に使用される
```

**CNAMEレコード（Canonical Name Record）**

ドメイン名の別名（エイリアス）を設定するレコード。

```
CNAMEレコードの例:

  ゾーンファイル:
  www.example.com.   3600  IN  CNAME  example.com.
  blog.example.com.  3600  IN  CNAME  example.github.io.
  shop.example.com.  3600  IN  CNAME  shops.myshopify.com.

  digコマンドで確認:
  $ dig www.example.com +noall +answer
  www.example.com.  3600  IN  CNAME  example.com.
  example.com.      300   IN  A      93.184.216.34

重要な制約:
  ・CNAMEはゾーンの頂点（Zone Apex）には設定できない
    ×  example.com.  IN  CNAME  other.example.com.  ← RFC違反
    ○  www.example.com.  IN  CNAME  other.example.com.  ← OK

  ・CNAMEと他のレコードは同じ名前に共存できない
    ×  www  IN  CNAME  example.com.
       www  IN  A      1.2.3.4            ← RFC違反
    ○  www  IN  CNAME  example.com.       ← CNAMEのみ

  理由: CNAMEは「この名前に対するすべてのクエリを転送先に委譲する」
        という意味を持つため、他のレコードとの共存は論理的に矛盾する

  Zone Apexでの代替手段:
  ・ALIAS / ANAMEレコード（一部DNSプロバイダ独自拡張）
  ・AWS Route 53 のエイリアスレコード
  ・Cloudflare のCNAME Flattening
```

**MXレコード（Mail Exchange Record）**

メールの配送先サーバーを指定するレコード。優先度（preference値）を持つ。

```
MXレコードの例:

  ゾーンファイル:
  example.com.  3600  IN  MX  10 mail1.example.com.
  example.com.  3600  IN  MX  20 mail2.example.com.
  example.com.  3600  IN  MX  30 mail-backup.example.com.

  → 数値が小さいほど優先度が高い
  → mail1 が応答しない場合、mail2 → mail-backup の順にフォールバック

  Google Workspace を使用する場合:
  example.com.  3600  IN  MX  1  ASPMX.L.GOOGLE.COM.
  example.com.  3600  IN  MX  5  ALT1.ASPMX.L.GOOGLE.COM.
  example.com.  3600  IN  MX  5  ALT2.ASPMX.L.GOOGLE.COM.
  example.com.  3600  IN  MX  10 ALT3.ASPMX.L.GOOGLE.COM.
  example.com.  3600  IN  MX  10 ALT4.ASPMX.L.GOOGLE.COM.

  digコマンドで確認:
  $ dig example.com MX +noall +answer
  example.com.  3600  IN  MX  10 mail1.example.com.
  example.com.  3600  IN  MX  20 mail2.example.com.

  重要: MXレコードの値にはIPアドレスではなくFQDNを指定する
        MXレコードの値にCNAMEは使用すべきでない（RFC 2181）
```

**TXTレコード（Text Record）**

任意のテキストデータを格納するレコード。メール認証（SPF, DKIM, DMARC）やドメイン所有権の検証に広く使用される。

```
TXTレコードの用途と例:

  1. SPF（Sender Policy Framework）:
     example.com.  3600  IN  TXT  "v=spf1 ip4:192.0.2.0/24 include:_spf.google.com ~all"
     → このドメインからメールを送信できるサーバーを宣言
     → ~all: 上記以外からの送信はソフトフェイル（疑わしいが拒否はしない）
     → -all: ハードフェイル（上記以外は完全拒否）

  2. DKIM（DomainKeys Identified Mail）:
     selector._domainkey.example.com.  3600  IN  TXT
       "v=DKIM1; k=rsa; p=MIGfMA0GCSqGSIb3DQEBAQUAA4GN..."
     → メールの電子署名を検証するための公開鍵

  3. DMARC（Domain-based Message Authentication）:
     _dmarc.example.com.  3600  IN  TXT
       "v=DMARC1; p=reject; rua=mailto:dmarc@example.com"
     → SPFとDKIMの結果に基づくメール処理ポリシー

  4. ドメイン所有権の検証:
     example.com.  300  IN  TXT  "google-site-verification=abc123..."
     example.com.  300  IN  TXT  "MS=ms12345678"
     → Google, Microsoftなどのサービスがドメイン所有者を確認

  5. セキュリティポリシー:
     _mta-sts.example.com.  3600  IN  TXT  "v=STSv1; id=20240101"
     → MTA-STS（SMTP TLSの強制）のバージョン管理
```

**NSレコード（Name Server Record）**

ゾーンの権威ネームサーバーを指定するレコード。

```
NSレコードの例:

  ゾーンファイル:
  example.com.  86400  IN  NS  ns1.example.com.
  example.com.  86400  IN  NS  ns2.example.com.
  example.com.  86400  IN  NS  ns3.example-dns.net.

  → 通常、最低2つのNSレコードを設定（冗長性確保）
  → 異なるネットワーク上にNSを配置することが推奨される
  → TTLは長めに設定されることが多い（86400秒 = 24時間）

  グルーレコード（Glue Record）:
  NSサーバー自体がそのゾーン内にある場合、
  循環参照を防ぐためにTLDのゾーンにAレコードが追加される

  例: example.com のNSが ns1.example.com の場合
  → ns1.example.com のIPを知るには example.com のNSに聞く必要がある
  → しかし example.com のNSが ns1.example.com である → 循環参照

  解決: 親ゾーン（.com）にグルーレコードを登録
  .com ゾーン内:
    example.com.      IN  NS  ns1.example.com.
    ns1.example.com.  IN  A   198.51.100.1      ← グルーレコード
```

**SOAレコード（Start of Authority Record）**

ゾーンの管理情報を記述するレコード。すべてのゾーンに必ず1つ存在する。

```
SOAレコードの例:

  example.com. 86400 IN SOA ns1.example.com. admin.example.com. (
    2024010101  ; シリアル番号（ゾーンのバージョン）
    3600        ; リフレッシュ間隔（セカンダリがプライマリを確認する間隔）
    900         ; リトライ間隔（リフレッシュ失敗時の再試行間隔）
    604800      ; 有効期限（セカンダリがゾーンデータを有効とみなす最大期間）
    86400       ; ネガティブキャッシュTTL（NXDOMAINをキャッシュする期間）
  )

  各フィールドの解説:
  ・MNAME (ns1.example.com.)   : プライマリネームサーバー
  ・RNAME (admin.example.com.) : 管理者のメールアドレス（@を.に置換）
                                  → 実際は admin@example.com
  ・シリアル番号: YYYYMMDDnn 形式が一般的
    → ゾーン変更時に必ずインクリメントする
    → セカンダリDNSはこの値でゾーン転送の必要性を判断

  digコマンドで確認:
  $ dig example.com SOA +noall +answer
```

**SRVレコード（Service Record）**

特定のサービスが稼働しているホストとポート番号を指定するレコード。

```
SRVレコードの書式:
  _サービス._プロトコル.ドメイン  TTL  IN  SRV  優先度 重み ポート ホスト

  例:
  _sip._tcp.example.com.     3600 IN SRV 10 60 5060 sipserver1.example.com.
  _sip._tcp.example.com.     3600 IN SRV 10 40 5060 sipserver2.example.com.
  _sip._tcp.example.com.     3600 IN SRV 20 0  5060 sipbackup.example.com.

  → 優先度10のサーバー間で重み60:40の比率で負荷分散
  → 優先度10が全滅した場合、優先度20にフォールバック

  Active Directoryでの使用例:
  _ldap._tcp.dc._msdcs.example.com.  600 IN SRV 0 100 389 dc1.example.com.
  _kerberos._tcp.example.com.        600 IN SRV 0 100 88  kdc.example.com.
```

**PTRレコード（Pointer Record）**

IPアドレスからドメイン名への逆引きを行うレコード。メールサーバーの信頼性検証に重要。

```
PTRレコードの例:

  IPv4の逆引き（in-addr.arpa）:
  34.216.184.93.in-addr.arpa.  3600  IN  PTR  example.com.
  → IPアドレスのオクテットを逆順にして .in-addr.arpa を付加

  IPv6の逆引き（ip6.arpa）:
  6.4.9.1.8.c.5.2.3.9.8.1.8.4.2.0.1.0.0.0.0.2.2.0.0.0.8.2.6.0.6.2.ip6.arpa.
    3600  IN  PTR  example.com.
  → 各ニブル（4ビット）を逆順にして .ip6.arpa を付加

  digコマンドで逆引き確認:
  $ dig -x 93.184.216.34 +noall +answer
  34.216.184.93.in-addr.arpa. 3600 IN PTR example.com.

  逆引きが重要な場面:
  ・メール送信時: 受信サーバーがPTRレコードをチェック
    → PTRが設定されていない、または正引きと一致しないとスパム判定される
  ・ログ解析: IPアドレスをホスト名に変換して可読性を向上
  ・セキュリティ監査: 不審なIPアドレスの所有者を特定
```

**CAAレコード（Certification Authority Authorization）**

ドメインに対してSSL/TLS証明書を発行できるCA（認証局）を制限するレコード。

```
CAAレコードの例:

  example.com.  3600  IN  CAA  0 issue "letsencrypt.org"
  example.com.  3600  IN  CAA  0 issue "digicert.com"
  example.com.  3600  IN  CAA  0 issuewild "letsencrypt.org"
  example.com.  3600  IN  CAA  0 iodef "mailto:security@example.com"

  フラグとタグ:
  ・0 issue        → 通常の証明書発行を許可するCA
  ・0 issuewild    → ワイルドカード証明書の発行を許可するCA
  ・0 iodef        → ポリシー違反時の通知先
  ・128 issue      → 128 = critical flag（未知のタグを持つ場合、発行を拒否）

  digコマンドで確認:
  $ dig example.com CAA +noall +answer

  CAAの動作:
  1. CAが証明書発行リクエストを受ける
  2. 対象ドメインのCAAレコードを確認
  3. 自身がissueに含まれていなければ発行を拒否
  4. CAAレコードが存在しなければ制限なし（任意のCAが発行可能）
```

### 4.3 レコードタイプの用途別比較表

| レコード | 用途 | 値の例 | 典型的なTTL | 設定頻度 |
|---------|------|-------|------------|---------|
| A | ドメイン→IPv4 | 93.184.216.34 | 300-3600 | 非常に高い |
| AAAA | ドメイン→IPv6 | 2606:2800:220:1:... | 300-3600 | 高い |
| CNAME | エイリアス | www→example.com | 3600 | 高い |
| MX | メール配送先 | 10 mail.example.com | 3600 | 中程度 |
| TXT | テキスト情報 | "v=spf1 ..." | 3600 | 中程度 |
| NS | 権威DNS指定 | ns1.example.com | 86400 | 低い |
| SOA | ゾーン管理情報 | (複合データ) | 86400 | 低い |
| SRV | サービス位置 | 10 60 5060 sip.ex... | 3600 | 低い |
| PTR | 逆引き | example.com | 3600 | 低い |
| CAA | CA制限 | 0 issue "le..." | 3600 | 低い |

---

## 5. DNSキャッシュとTTL

### 5.1 キャッシュの階層構造

DNS名前解決は多段階のキャッシュによって高速化されている。各階層のキャッシュが連携して動作することで、DNSサーバーへの問い合わせ回数を大幅に削減している。

```
DNSキャッシュの階層:

  ┌──────────────────────────────────────────────────────────────┐
  │                    キャッシュ階層図                           │
  │                                                              │
  │  ┌─────────────────────────────┐    応答速度: < 1ms          │
  │  │  1. アプリケーションキャッシュ │    Chrome内部、curlキャッシュ等│
  │  │     (ブラウザ等)             │    TTL: アプリ依存           │
  │  └─────────────┬───────────────┘                             │
  │        miss    │                                              │
  │                ▼                                              │
  │  ┌─────────────────────────────┐    応答速度: < 1ms          │
  │  │  2. OSキャッシュ             │    systemd-resolved,        │
  │  │     (スタブリゾルバ)         │    mDNSResponder等          │
  │  └─────────────┬───────────────┘    TTL: レコードのTTLに従う  │
  │        miss    │                                              │
  │                ▼                                              │
  │  ┌─────────────────────────────┐    応答速度: 1-5ms          │
  │  │  3. ローカルDNSキャッシュ     │    ルーター、dnsmasq等      │
  │  │     (ホームルーター等)       │    TTL: レコードのTTLに従う  │
  │  └─────────────┬───────────────┘                             │
  │        miss    │                                              │
  │                ▼                                              │
  │  ┌─────────────────────────────┐    応答速度: 5-50ms         │
  │  │  4. ISPリゾルバキャッシュ     │    ISP提供のDNSサーバー     │
  │  │     (フルリゾルバ)           │    大量ユーザーのキャッシュ共有│
  │  └─────────────┬───────────────┘    TTL: レコードのTTLに従う  │
  │        miss    │                                              │
  │                ▼                                              │
  │  ┌─────────────────────────────┐    応答速度: 50-200ms       │
  │  │  5. 権威DNSサーバー          │    ルート→TLD→権威の        │
  │  │     (反復クエリ)             │    反復クエリを実行          │
  │  └─────────────────────────────┘                             │
  └──────────────────────────────────────────────────────────────┘
```

### 5.2 TTL（Time To Live）の設計

TTLの設定はDNS運用における最も重要な設計判断の一つである。

```
TTL設定のガイドライン:

  短いTTL（60〜300秒）:
  ┌────────────────────────────────────────────────────────┐
  │ メリット                                                │
  │ ・DNS切り替えの反映が速い                               │
  │ ・フェイルオーバーの応答が速い                           │
  │ ・Blue-Greenデプロイやカナリアリリースに適する            │
  │                                                        │
  │ デメリット                                              │
  │ ・DNSサーバーへの問い合わせ頻度が増加                    │
  │ ・ネットワーク遅延の影響を受けやすい                     │
  │ ・権威DNSサーバーの負荷が高い                            │
  │                                                        │
  │ 推奨場面                                                │
  │ ・CDN（CloudFront, Fastly等）の設定                     │
  │ ・ロードバランサーのDNS設定                              │
  │ ・インフラ移行の準備期間                                 │
  │ ・障害時の切り替えが重要なサービス                       │
  └────────────────────────────────────────────────────────┘

  長いTTL（3600〜86400秒）:
  ┌────────────────────────────────────────────────────────┐
  │ メリット                                                │
  │ ・DNS問い合わせの回数が大幅に減少                        │
  │ ・名前解決のレイテンシが低い（キャッシュヒット率が高い）  │
  │ ・権威DNSサーバーの負荷が低い                            │
  │                                                        │
  │ デメリット                                              │
  │ ・DNS変更の反映に時間がかかる                            │
  │ ・障害時の切り替えが遅い                                 │
  │                                                        │
  │ 推奨場面                                                │
  │ ・NSレコード（変更頻度が非常に低い）                     │
  │ ・SOAレコード                                           │
  │ ・安定運用中のサービスのAレコード                        │
  │ ・MXレコード                                            │
  └────────────────────────────────────────────────────────┘
```

### 5.3 DNS移行時のTTL運用ベストプラクティス

```
サーバー移行時のTTL運用手順:

  時間軸:
  ──────┬────────────────────┬──────────────────┬──────────────────┬─────
   T-48h │                    │ T-0（移行実施）   │ T+24h            │ T+72h
        │                    │                  │                  │
   ① TTLを短縮              ② レコード変更      ③ 正常性確認      ④ TTL復元
   (3600→300)               (旧IP→新IP)        (全リゾルバ反映)   (300→3600)

  詳細手順:
  ① 移行48時間前:
     旧設定: example.com.  3600  IN  A  198.51.100.1
     変更後: example.com.   300  IN  A  198.51.100.1   ← TTLだけ短縮
     → 元のTTL（3600秒 = 1時間）の2倍以上待つ
     → すべてのキャッシュが新しいTTL（300秒）で更新される

  ② 移行実施:
     変更後: example.com.   300  IN  A  203.0.113.50   ← IPアドレスを変更
     → 最大300秒（5分）で全世界に反映

  ③ 移行24時間後:
     監視項目:
     ・新サーバーへのトラフィック推移
     ・旧サーバーへのトラフィック消失確認
     ・エラーレート、レイテンシの確認

  ④ 移行72時間後:
     変更後: example.com.  3600  IN  A  203.0.113.50   ← TTLを元に戻す
     → 安定運用に移行

  ★ よくある失敗:
     TTLを短縮せずにいきなりIPを変更
     → 旧TTL（例: 86400秒 = 24時間）の間、
       古いIPにアクセスするユーザーが残る
     → その間のリクエストは全て失敗する
```

### 5.4 ネガティブキャッシュ

存在しないドメイン（NXDOMAIN）に対する応答もキャッシュされる。これをネガティブキャッシュという。

```
ネガティブキャッシュの仕組み:

  $ dig nonexistent.example.com
  ;; ->>HEADER<<- opcode: QUERY, status: NXDOMAIN, ...
  ;; AUTHORITY SECTION:
  example.com.  86400  IN  SOA  ns1.example.com. admin.example.com. ...

  → NXDOMAIN のキャッシュ期間は SOA レコードの最後のフィールド
    （ネガティブキャッシュTTL）で決定される

  影響:
  ・新しいサブドメインを作成しても、ネガティブキャッシュの期間中は
    「存在しない」という応答が返され続ける場合がある
  ・SOAのネガティブキャッシュTTLが長すぎると問題になる

  推奨: ネガティブキャッシュTTLは 300〜3600 秒に設定
```

---

## 6. コード例: DNSデバッグの実践

### 6.1 digコマンドの活用

`dig`（Domain Information Groper）はDNS問い合わせのための最も強力なコマンドラインツールである。

```bash
# ============================================================
# コード例1: dig の基本的な使い方
# ============================================================

# 基本的なAレコードの問い合わせ
$ dig example.com

# 出力の読み方:
# ;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 12345
#   → status: NOERROR = 正常応答
#   → status: NXDOMAIN = ドメインが存在しない
#   → status: SERVFAIL = サーバーエラー
#   → status: REFUSED = 問い合わせ拒否
#
# ;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1
#   → qr = Query Response（応答）
#   → rd = Recursion Desired（再帰要求あり）
#   → ra = Recursion Available（再帰利用可能）
#   → aa = Authoritative Answer（権威応答）← 重要
#
# ;; ANSWER SECTION:
# example.com.  300  IN  A  93.184.216.34
#   → ドメイン名  TTL  クラス  タイプ  データ

# 特定のレコードタイプを問い合わせ
$ dig example.com A          # IPv4アドレス
$ dig example.com AAAA       # IPv6アドレス
$ dig example.com MX         # メールサーバー
$ dig example.com TXT        # テキストレコード
$ dig example.com NS         # ネームサーバー
$ dig example.com SOA        # ゾーン管理情報
$ dig example.com ANY        # すべてのレコード（※多くのサーバーで制限あり）

# 簡潔な出力
$ dig +short example.com
93.184.216.34

# 応答セクションだけ表示
$ dig +noall +answer example.com
example.com.  300  IN  A  93.184.216.34

# 特定のDNSサーバーに問い合わせ
$ dig @8.8.8.8 example.com        # Google Public DNS
$ dig @1.1.1.1 example.com        # Cloudflare DNS
$ dig @9.9.9.9 example.com        # Quad9 DNS

# DNSの解決過程を追跡（+trace）
$ dig +trace example.com
# → ルートDNS → .com TLD DNS → 権威DNS の全過程が表示される
# . 518400 IN NS a.root-servers.net.
# ...
# com. 172800 IN NS a.gtld-servers.net.
# ...
# example.com. 300 IN A 93.184.216.34
```

### 6.2 nslookupコマンド

`nslookup` はdigより古いが、Windows環境でも標準で使用できるツールである。

```bash
# ============================================================
# コード例2: nslookup の使い方
# ============================================================

# 基本的な問い合わせ
$ nslookup example.com
Server:    192.168.1.1
Address:   192.168.1.1#53

Non-authoritative answer:
Name:      example.com
Address:   93.184.216.34

# 特定のレコードタイプ
$ nslookup -type=MX example.com
$ nslookup -type=TXT example.com
$ nslookup -type=NS example.com

# 特定のDNSサーバーを指定
$ nslookup example.com 8.8.8.8

# 対話モード
$ nslookup
> server 8.8.8.8
Default server: 8.8.8.8
Address: 8.8.8.8#53
> set type=MX
> example.com
example.com    mail exchanger = 10 mail.example.com.
> exit

# 逆引き
$ nslookup 93.184.216.34
```

### 6.3 hostコマンド

`host` はdigの簡易版で、人間に読みやすい出力を生成する。

```bash
# ============================================================
# コード例3: host コマンドの使い方
# ============================================================

# 基本的な問い合わせ
$ host example.com
example.com has address 93.184.216.34
example.com has IPv6 address 2606:2800:220:1:248:1893:25c8:1946
example.com mail is handled by 0 .

# 特定のレコードタイプ
$ host -t MX example.com
example.com mail is handled by 10 mail.example.com.

$ host -t NS example.com
example.com name server ns1.example.com.
example.com name server ns2.example.com.

# 逆引き
$ host 93.184.216.34
34.216.184.93.in-addr.arpa domain name pointer example.com.

# 詳細出力（digに近い形式）
$ host -v example.com

# 特定のDNSサーバーに問い合わせ
$ host example.com 8.8.8.8
```

### 6.4 ゾーンファイルの設定例

BINDなどの権威DNSサーバーで使用するゾーンファイルの完全な例を示す。

```bash
# ============================================================
# コード例4: BINDゾーンファイル（/etc/bind/zones/example.com.zone）
# ============================================================

$TTL 3600                              ; デフォルトTTL = 1時間
$ORIGIN example.com.                   ; ゾーンの基点

; ── SOAレコード ──
@   IN  SOA  ns1.example.com.  admin.example.com. (
            2024031501    ; シリアル番号（YYYYMMDDNN形式）
            3600          ; リフレッシュ（1時間）
            900           ; リトライ（15分）
            604800        ; 期限切れ（7日）
            86400         ; ネガティブキャッシュTTL（1日）
          )

; ── NSレコード ──
@           IN  NS    ns1.example.com.
@           IN  NS    ns2.example.com.

; ── Aレコード（IPv4） ──
@           IN  A     93.184.216.34
www         IN  A     93.184.216.34
api         IN  A     10.0.1.100
api         IN  A     10.0.1.101         ; ラウンドロビン
staging     IN  A     10.0.2.50

; ── AAAAレコード（IPv6） ──
@           IN  AAAA  2606:2800:220:1:248:1893:25c8:1946

; ── CNAMEレコード ──
blog        IN  CNAME example.github.io.
shop        IN  CNAME shops.myshopify.com.
docs        IN  CNAME example-docs.netlify.app.

; ── MXレコード ──
@           IN  MX  10  mail1.example.com.
@           IN  MX  20  mail2.example.com.

; ── TXTレコード ──
@           IN  TXT   "v=spf1 ip4:93.184.216.0/24 include:_spf.google.com ~all"
_dmarc      IN  TXT   "v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com"

; ── SRVレコード ──
_sip._tcp   IN  SRV   10 60 5060 sipserver.example.com.

; ── CAAレコード ──
@           IN  CAA   0 issue "letsencrypt.org"
@           IN  CAA   0 issuewild "letsencrypt.org"

; ── NSサーバーのAレコード ──
ns1         IN  A     198.51.100.1
ns2         IN  A     198.51.100.2

; ── メールサーバーのAレコード ──
mail1       IN  A     198.51.100.10
mail2       IN  A     198.51.100.11
```

### 6.5 resolv.confとsystemd-resolvedの設定

```bash
# ============================================================
# コード例5: クライアント側のDNS設定
# ============================================================

# --- /etc/resolv.conf ---
# 基本設定
nameserver 8.8.8.8          # プライマリDNS（Google）
nameserver 8.8.4.4          # セカンダリDNS（Google）
nameserver 1.1.1.1          # ターシャリDNS（Cloudflare）

search example.com internal.example.com
# → "myhost" を検索すると以下の順に解決を試みる:
#    1. myhost.example.com
#    2. myhost.internal.example.com
#    3. myhost（FQDNとして）

options timeout:2 attempts:3 rotate
# timeout:2  → 各クエリのタイムアウト2秒
# attempts:3 → 最大3回再試行
# rotate     → ネームサーバーをラウンドロビンで使用

# --- systemd-resolved の設定 ---
# /etc/systemd/resolved.conf
[Resolve]
DNS=8.8.8.8 1.1.1.1
FallbackDNS=8.8.4.4 9.9.9.9
Domains=~.
DNSSEC=allow-downgrade
DNSOverTLS=opportunistic
Cache=yes
DNSStubListener=yes

# systemd-resolved の状態確認
$ resolvectl status
Global
       Protocols: +LLMNR +mDNS -DNSOverTLS DNSSEC=allow-downgrade/supported
resolv.conf mode: stub
     DNS Servers: 8.8.8.8 1.1.1.1
Fallback DNS Servers: 8.8.4.4 9.9.9.9

# キャッシュの統計確認
$ resolvectl statistics
DNSSEC supported: yes
Current Transactions: 0
  Total Transactions: 12345
  Current Cache Size: 234
          Cache Hits: 5678
        Cache Misses: 6789

# キャッシュのフラッシュ
$ resolvectl flush-caches
```

---

## 7. DNSセキュリティ

### 7.1 DNSに対する主な脅威

```
DNSの脅威モデル:

  ┌──────────────────────────────────────────────────────────────┐
  │                    DNS攻撃の分類                              │
  ├──────────────────┬───────────────────────────────────────────┤
  │ DNSスプーフィング │ 偽のDNS応答を注入し、ユーザーを偽サイトへ  │
  │ (DNSポイズニング) │ 誘導。キャッシュに偽レコードを挿入する。   │
  │                  │ 対策: DNSSEC、ソースポートランダム化       │
  ├──────────────────┼───────────────────────────────────────────┤
  │ DNS増幅攻撃      │ 送信元IPを偽装したDNSクエリを大量送信。    │
  │ (DDoS)           │ 応答が被害者に集中し、帯域を圧迫。        │
  │                  │ 対策: レートリミット、オープンリゾルバの    │
  │                  │       排除、BCP38（送信元検証）            │
  ├──────────────────┼───────────────────────────────────────────┤
  │ DNSハイジャック   │ レジストラアカウントの乗っ取り、            │
  │                  │ ゾーンファイルの不正変更。                  │
  │                  │ 対策: レジストラロック、二要素認証          │
  ├──────────────────┼───────────────────────────────────────────┤
  │ DNS盗聴          │ 平文のDNSクエリを盗聴し、ユーザーの        │
  │                  │ 閲覧行動を追跡。                           │
  │                  │ 対策: DoH（DNS over HTTPS）、              │
  │                  │       DoT（DNS over TLS）                  │
  ├──────────────────┼───────────────────────────────────────────┤
  │ NXDOMAINハイジャック│ 存在しないドメインの問い合わせを          │
  │                  │ ISPが横取りし、広告ページへ転送。           │
  │                  │ 対策: DNSSEC検証、パブリックDNSの使用      │
  └──────────────────┴───────────────────────────────────────────┘
```

### 7.2 DNSSEC（DNS Security Extensions）

DNSSECはDNS応答の真正性と完全性を暗号学的に検証する仕組みである。

```
DNSSECの仕組み:

  署名の流れ:
  1. ゾーン管理者がZSK（Zone Signing Key）でレコードに署名
  2. ZSKをKSK（Key Signing Key）で署名
  3. KSKのハッシュ（DSレコード）を親ゾーンに登録
  4. 親ゾーンが自身のKSKでDSレコードに署名
  5. ルートまで信頼の連鎖（Chain of Trust）が形成される

  信頼の連鎖:
  ルートKSK（トラストアンカー: IANAが管理）
    ↓ DSレコードで検証
  .com KSK
    ↓ DSレコードで検証
  example.com KSK
    ↓ ZSKで検証
  www.example.com A  93.184.216.34  ← この応答が改竄されていないことを暗号的に保証

  追加されるレコードタイプ:
  ・RRSIG   : 各レコードセットの電子署名
  ・DNSKEY  : ゾーンの公開鍵（ZSK, KSK）
  ・DS      : 子ゾーンのDNSKEYのハッシュ（親ゾーンに登録）
  ・NSEC/NSEC3 : レコードが存在しないことの証明

  digコマンドでDNSSEC検証:
  $ dig +dnssec example.com A
  → フラグに "ad" が含まれれば DNSSEC 検証成功
  → ad = Authentic Data
```

### 7.3 DoH / DoT（暗号化DNS）

従来のDNS（Do53: ポート53のUDP/TCP平文通信）に対し、暗号化されたDNSプロトコルが登場している。

| 比較項目 | Do53（従来DNS） | DoT（DNS over TLS） | DoH（DNS over HTTPS） |
|---------|----------------|---------------------|----------------------|
| ポート | 53 (UDP/TCP) | 853 (TCP) | 443 (TCP) |
| 暗号化 | なし | TLS | HTTPS (TLS) |
| プライバシー | なし（盗聴可能） | 高い | 非常に高い |
| ファイアウォール通過 | 容易 | ブロック可能 | ブロック困難（HTTPS混在） |
| 遅延 | 最小 | TLSハンドシェイク分 | HTTPSオーバーヘッド分 |
| 対応リゾルバ | すべて | Cloudflare, Google等 | Cloudflare, Google等 |
| 標準規格 | RFC 1035 | RFC 7858 | RFC 8484 |

---

## 8. クラウド環境でのDNS

### 8.1 AWS Route 53

Route 53はAWSが提供するスケーラブルなマネージドDNSサービスである。ドメイン登録、DNSホスティング、ヘルスチェックの3機能を備える。

```
Route 53 の主要機能:

  1. ドメイン登録
     → .com, .net, .org, .jp 等のドメインを直接登録可能
     → WHOISプライバシー保護が無料で付属
     → ドメインロック機能で不正移管を防止

  2. DNSホスティング（権威DNSサーバー）
     → ホストゾーンを作成してレコードを管理
     → パブリックホストゾーン: インターネット向け
     → プライベートホストゾーン: VPC内部向け
     → SLA 100% の可用性保証

  3. ヘルスチェック + ルーティング
     → エンドポイントのヘルスチェック（HTTP/HTTPS/TCP）
     → 異常検知時に自動フェイルオーバー
     → CloudWatch と統合した監視・アラート

Route 53 ルーティングポリシー:
  ┌───────────────┬──────────────────────────────────────────────┐
  │ ポリシー       │ 説明と使用場面                                │
  ├───────────────┼──────────────────────────────────────────────┤
  │ シンプル       │ 1つのリソースにルーティング。最も基本的。      │
  │               │ 例: 単一のWebサーバー                         │
  ├───────────────┼──────────────────────────────────────────────┤
  │ 加重          │ 割合ベースで振り分け。A/Bテスト、段階的移行に。│
  │ (Weighted)    │ 例: 新バージョンに10%、旧バージョンに90%      │
  ├───────────────┼──────────────────────────────────────────────┤
  │ レイテンシー   │ 最も遅延の少ないリージョンへルーティング。     │
  │ (Latency)     │ 例: 日本ユーザー→東京、米国ユーザー→バージニア│
  ├───────────────┼──────────────────────────────────────────────┤
  │ フェイルオーバー│ プライマリ障害時にセカンダリへ自動切り替え。   │
  │ (Failover)    │ 例: EC2障害時にS3静的サイトへフォールバック    │
  ├───────────────┼──────────────────────────────────────────────┤
  │ 位置情報       │ ユーザーの地理的位置に基づくルーティング。     │
  │ (Geolocation) │ 例: 日本からのアクセスは日本語サイトへ        │
  ├───────────────┼──────────────────────────────────────────────┤
  │ 地理的近接性   │ リソースの地理的位置とバイアス値に基づく。     │
  │ (Geoproximity)│ 例: Traffic Flowと組み合わせて精密制御        │
  ├───────────────┼──────────────────────────────────────────────┤
  │ 複数値応答     │ 最大8つの正常なIPをランダムに返す。           │
  │ (Multivalue)  │ 例: 簡易的な負荷分散（ヘルスチェック付き）    │
  ├───────────────┼──────────────────────────────────────────────┤
  │ IPベース       │ クライアントIPの範囲に基づくルーティング。     │
  │ (IP-based)    │ 例: 特定ISPのユーザーを最適なエンドポイントへ │
  └───────────────┴──────────────────────────────────────────────┘

エイリアスレコード（AWS固有の拡張）:
  通常のCNAME:
    ・ゾーンの頂点（example.com）には設定不可
    ・DNSクエリが2回発生（CNAME解決 + A解決）
    ・クエリ料金が発生

  Route 53 エイリアスレコード:
    ・ゾーンの頂点にも設定可能
    ・AWSリソースへの問い合わせは無料
    ・対象: CloudFront, ELB, S3, API Gateway, VPCエンドポイント等
    ・内部的にAレコードとして解決（追加クエリ不要）
```

### 8.2 パブリックDNSサービスの比較

| サービス | プライマリIP | セカンダリIP | DoH | DoT | DNSSEC検証 | 特徴 |
|---------|------------|------------|-----|-----|-----------|------|
| Google Public DNS | 8.8.8.8 | 8.8.4.4 | 対応 | 対応 | 対応 | 最も普及、安定性が高い |
| Cloudflare DNS | 1.1.1.1 | 1.0.0.1 | 対応 | 対応 | 対応 | 低遅延、プライバシー重視 |
| Quad9 | 9.9.9.9 | 149.112.112.112 | 対応 | 対応 | 対応 | マルウェアドメインブロック |
| OpenDNS | 208.67.222.222 | 208.67.220.220 | 対応 | 非対応 | 対応 | フィルタリング機能 |

---

## 9. アンチパターン

### 9.1 アンチパターン1: ゾーンファイルでの末尾ドット忘れ

```
アンチパターン: CNAMEやMXの値で末尾ドットを忘れる

  問題のあるゾーンファイル:
  ──────────────────────────────────────────
  $ORIGIN example.com.
  www   IN  CNAME  example.com       ← 末尾ドットなし（危険）
  @     IN  MX  10 mail.example.com  ← 末尾ドットなし（危険）
  ──────────────────────────────────────────

  BINDの解釈:
  ・$ORIGIN が example.com. の場合、末尾ドットがない名前には
    自動的に $ORIGIN が付加される
  ・上記の場合:
    www IN CNAME example.com         → example.com.example.com. に展開される
    @   IN MX 10 mail.example.com    → mail.example.com.example.com. に展開される

  → 意図しないドメインを参照してしまい、名前解決が失敗する
  → 障害の原因特定が難しく、デバッグに時間がかかる

  正しいゾーンファイル:
  ──────────────────────────────────────────
  $ORIGIN example.com.
  www   IN  CNAME  example.com.       ← 末尾ドットあり（正しい）
  @     IN  MX  10 mail.example.com.  ← 末尾ドットあり（正しい）
  ──────────────────────────────────────────

  予防策:
  ・ゾーンファイルの値には常にFQDN（末尾ドット付き）を使用する
  ・named-checkzone コマンドでゾーンファイルの構文チェックを行う
    $ named-checkzone example.com /etc/bind/zones/example.com.zone
  ・CI/CDパイプラインにゾーンファイルの自動検証を組み込む
```

### 9.2 アンチパターン2: TTLを考慮しないDNS変更

```
アンチパターン: 長いTTLのまま急なDNS変更を行う

  状況:
  ──────────────────────────────────────────
  現在の設定:
  example.com.  86400  IN  A  198.51.100.1    ← TTL = 24時間

  緊急のサーバー移行が必要になった:
  example.com.  86400  IN  A  203.0.113.50    ← IPだけ変更
  ──────────────────────────────────────────

  問題:
  ・変更前のレコード（198.51.100.1）が世界中のリゾルバに最大24時間キャッシュされている
  ・変更後も最大24時間、旧IPにアクセスするユーザーが存在する
  ・旧サーバーが停止している場合、そのユーザーはサービスに接続できない

  発生する事象:
  ────────────────────────────────────────────────
  時間    キャッシュ残時間  アクセス先        結果
  ────────────────────────────────────────────────
  T+0     23時間59分       198.51.100.1      失敗
  T+6h    17時間59分       198.51.100.1      失敗
  T+12h   11時間59分       198.51.100.1      失敗
  T+18h   5時間59分        198.51.100.1      失敗
  T+24h   0                203.0.113.50      成功
  ────────────────────────────────────────────────
  → 最悪の場合、24時間のサービス断が発生

  正しい手順（再掲）:
  1. TTLを短縮（300秒等）して旧TTLの2倍以上待つ
  2. IPアドレスを変更
  3. 安定後にTTLを戻す

  緊急時の代替策:
  ・旧IPから新IPへのリバースプロキシ設定
  ・旧サーバーで301リダイレクトを返す
  ・CDN経由の場合はCDN側のオリジン設定を変更
```

---

## 10. エッジケース分析

### 10.1 エッジケース1: CNAMEチェーンとループ

```
エッジケース: CNAME が別のCNAMEを指す（CNAMEチェーン）

  正常なCNAMEチェーン:
  ──────────────────────────────────────────
  www.example.com.     CNAME  lb.example.com.
  lb.example.com.      CNAME  us-east-1.elb.amazonaws.com.
  us-east-1.elb.amazonaws.com.  A  54.239.28.85
  ──────────────────────────────────────────
  → 3段階のCNAME解決が発生
  → 各段階でDNS問い合わせが必要（キャッシュミス時）
  → 遅延が増加する

  問題になるケース:
  1. 長すぎるCNAMEチェーン
     → 多くのリゾルバはチェーンの深さに制限を設けている（通常8〜16段）
     → 制限を超えるとSERVFAIL（名前解決失敗）が返される

  2. CNAMEループ（循環参照）
     a.example.com.  CNAME  b.example.com.
     b.example.com.  CNAME  c.example.com.
     c.example.com.  CNAME  a.example.com.    ← ループ
     → リゾルバがループを検出してSERVFAILを返す
     → ループ検出までにDNSクエリが無駄に消費される

  3. 外部サービスのCNAME先が変更・削除された場合
     shop.example.com.  CNAME  shops.myshopify.com.
     → Shopifyがドメインを変更した場合、shop.example.com が解決不能に
     → 外部サービスのCNAME先の変更を監視する仕組みが必要

  推奨事項:
  ・CNAMEチェーンは最大2〜3段に抑える
  ・ゾーン内のCNAMEは可能な限りAレコードに置き換える
  ・外部サービスのCNAME先は定期的に監視する
  ・CNAMEループを検出するテストを導入する
```

### 10.2 エッジケース2: ネガティブキャッシュとサービス起動順序

```
エッジケース: サービスデプロイ時のネガティブキャッシュ汚染

  シナリオ:
  ──────────────────────────────────────────
  1. Kubernetesで新しいサービスをデプロイ
  2. DNSレコード（api-v2.example.com）を作成
  3. デプロイ完了前にヘルスチェックがDNSを問い合わせ
  4. まだレコードが反映されていないため NXDOMAIN が返される
  5. NXDOMAINがネガティブキャッシュされる（SOA TTLに従う）
  6. レコードが反映されても、キャッシュ期間中は NXDOMAIN が返される
  7. サービスが利用不能な状態が続く
  ──────────────────────────────────────────

  時系列:
  T+0    DNSレコード作成
  T+10s  ヘルスチェックが問い合わせ → NXDOMAIN（まだ反映されていない）
  T+30s  レコードが全権威サーバーに反映
  T+30s  ヘルスチェックが再問い合わせ → まだ NXDOMAIN（キャッシュ）
  ...
  T+3600s ネガティブキャッシュ期限切れ → ようやく正常応答

  予防策:
  1. DNSレコードを先に作成し、反映を確認してからサービスをデプロイ
  2. SOAのネガティブキャッシュTTLを短く設定する（300秒推奨）
  3. デプロイパイプラインにDNS反映確認ステップを組み込む:
     $ until dig +short api-v2.example.com | grep -q .; do
     >   echo "Waiting for DNS propagation..."
     >   sleep 5
     > done
     $ echo "DNS record is live!"
  4. 複数のパブリックDNSリゾルバで反映を確認:
     $ dig @8.8.8.8 api-v2.example.com +short
     $ dig @1.1.1.1 api-v2.example.com +short
```

---

## 11. 演習問題

### 11.1 基礎演習

```
演習1（基礎）: DNSレコードの調査

目的: digコマンドを使ったDNS情報の取得に慣れる

課題:
  以下のドメインについて、指定されたレコードを dig で取得せよ。

  1. google.com の Aレコードを取得せよ
     $ dig google.com A +noall +answer

  2. google.com の MXレコードを取得せよ
     $ dig google.com MX +noall +answer

  3. google.com の NSレコードを取得せよ
     $ dig google.com NS +noall +answer

  4. google.com の TXTレコード（SPF）を取得せよ
     $ dig google.com TXT +noall +answer

  5. 93.184.216.34 の逆引き（PTRレコード）を取得せよ
     $ dig -x 93.184.216.34 +noall +answer

確認ポイント:
  ・各レコードのTTL値を記録し、その値の意味を考える
  ・MXレコードの優先度値を確認する
  ・NSレコードが複数返される理由を説明できるか

期待される学習成果:
  ・dig コマンドの基本的な使い方を習得
  ・各レコードタイプの出力形式を理解
  ・TTL、優先度などの付加情報の意味を把握
```

### 11.2 応用演習

```
演習2（応用）: DNS解決フローのトレースと分析

目的: DNS名前解決の全過程を可視化し、各ステップを理解する

課題:
  1. dig +trace で名前解決の全過程をトレースせよ
     $ dig +trace www.example.com

     出力を分析し、以下の質問に答えよ:
     a) ルートDNSサーバーはどのサーバーが選ばれたか？
     b) .com のTLDサーバーはどのサーバーが応答したか？
     c) 権威DNSサーバーのFQDNは何か？
     d) 最終的なIPアドレスとTTLは？

  2. 異なるDNSリゾルバでの応答を比較せよ
     $ dig @8.8.8.8 example.com +noall +answer +stats
     $ dig @1.1.1.1 example.com +noall +answer +stats
     $ dig @9.9.9.9 example.com +noall +answer +stats

     分析:
     a) 各リゾルバのQuery timeを比較
     b) TTL値に差異があるか確認
     c) 差異がある場合、その理由は何か

  3. CNAMEチェーンの解決過程を確認せよ
     CNAMEを使用しているドメイン（例: GitHub Pagesのカスタムドメイン）
     を調べ、チェーンの各段階をトレースせよ

確認ポイント:
  ・トレースの各ステップでどのサーバーが応答したかを特定できるか
  ・権威応答（aa フラグ）と非権威応答の違いを見分けられるか
  ・Query timeの差異がキャッシュの有無を反映していることを理解できるか

期待される学習成果:
  ・DNS解決の全過程を実際に観察し、理論と実践を結びつける
  ・異なるリゾルバの応答特性を把握
  ・CNAMEチェーンの実際の動作を理解
```

### 11.3 発展演習

```
演習3（発展）: ゾーンファイルの設計とDNSSEC検証

目的: 権威DNSの設計能力とセキュリティ検証スキルを養う

課題A: ゾーンファイルの設計
  以下の要件を満たすゾーンファイルを作成せよ。

  ドメイン: mycompany.example.com
  要件:
  ・Webサーバー（www）: 203.0.113.10 と 203.0.113.11（ラウンドロビン）
  ・APIサーバー（api）: 203.0.113.20
  ・ステージング（staging）: staging.herokuapp.com へのCNAME
  ・メール: Google Workspace を使用
  ・SPFレコード: Google と自社IP（203.0.113.0/24）を許可
  ・DMARCレコード: quarantine ポリシー、レポート先は admin@mycompany.example.com
  ・CAAレコード: Let's Encrypt のみ許可
  ・TTL: Webサーバー300秒、メール関連3600秒、NS 86400秒

  ヒント:
  ・Google WorkspaceのMXレコード設定はGoogleのドキュメントを参照
  ・SPFの include 構文を使用
  ・DMARCはサブドメイン _dmarc に設定

課題B: DNSSEC検証
  1. DNSSEC に対応しているドメインを見つけ、検証せよ
     $ dig +dnssec +multi example.com A
     → ad フラグの有無を確認

  2. DNSKEYレコードとRRSIGレコードを取得して確認せよ
     $ dig example.com DNSKEY +noall +answer
     $ dig example.com RRSIG +noall +answer

  3. DSレコードを親ゾーンから取得し、信頼の連鎖を確認せよ
     $ dig example.com DS +noall +answer

  4. delv コマンド（BINDの DNSSEC 検証ツール）で検証せよ
     $ delv @8.8.8.8 example.com A +rtrace
     → "fully validated" が表示されれば検証成功

課題C: DNS障害シミュレーション
  ローカル環境（/etc/hosts や dnsmasq）で以下のシナリオを再現し、
  影響と対処法を検証せよ:
  1. 権威DNSサーバーが応答しない場合のタイムアウト動作
  2. 異なるDNSリゾルバにフォールバックする動作
  3. ネガティブキャッシュの影響

確認ポイント:
  ・ゾーンファイルの構文が正しいか（named-checkzoneで検証）
  ・DNSSEC の信頼の連鎖が理解できているか
  ・障害時の動作を予測し、適切な対処法を提案できるか

期待される学習成果:
  ・実際のゾーンファイルを設計・検証できるスキル
  ・DNSSECの実際の検証手順の習得
  ・DNS障害対応の基礎力
```

---

## 12. FAQ（よくある質問）

### FAQ 1: DNSの変更が反映されるまでにどのくらいかかるか？

```
Q: DNSレコードを変更したが、まだ古い値が返される。いつ反映されるのか？

A: DNS変更の反映時間は、以下の要素によって決まる。

  1. 旧レコードのTTL
     → TTL=3600（1時間）なら、最大1時間で全キャッシュから消える
     → TTL=86400（24時間）なら、最大24時間

  2. リゾルバの実装
     → 一部のリゾルバはTTLを厳密に守らない場合がある
     → TTLの最小値を独自に設定していることがある（例: 最低60秒）
     → RFC 8767 では、権威サーバーが応答しない場合に
       期限切れキャッシュの一時利用を許容している

  3. アプリケーションのキャッシュ
     → ブラウザ、OS、ルーターそれぞれが独自にキャッシュ
     → Java等の一部言語ランタイムはDNSをアプリ内にキャッシュ
       （JVMのデフォルトは正引き30秒、負引き10秒）

  確認方法:
  $ dig @8.8.8.8 example.com +short      # Google DNS
  $ dig @1.1.1.1 example.com +short      # Cloudflare DNS
  $ dig @ns1.example.com example.com +short  # 権威DNS直接

  権威DNSに直接問い合わせて新しい値が返されるなら、
  変更自体は完了している。リゾルバのキャッシュ期限切れを待つ必要がある。

  対策:
  ・事前にTTLを短縮しておく（前述のベストプラクティス参照）
  ・dig +trace で権威サーバーからの応答を直接確認する
  ・whatsmydns.net 等のツールで世界各地の伝搬状況を確認する
```

### FAQ 2: CNAMEとAレコードのどちらを使うべきか？

```
Q: WebサーバーのDNS設定で、CNAMEとAレコードのどちらを使うべきか？

A: 以下の判断基準で選択する。

  Aレコードを使うべき場合:
  ──────────────────────────────────────
  ・ゾーンの頂点（example.com）のレコード
    → CNAMEはゾーン頂点に設定できない（RFC制約）
  ・IPアドレスが固定で変更されない場合
  ・最小のDNS問い合わせ回数が求められる場合
  ・他のレコード（MX, TXT等）と同じ名前に共存させる場合

  CNAMEを使うべき場合:
  ──────────────────────────────────────
  ・外部サービス（CDN, PaaS等）を指す場合
    例: blog.example.com → example.github.io
    → 外部サービスのIP変更に自動的に追従
  ・複数のサブドメインが同じ先を指す場合
    → 変更時にCNAME先だけ更新すれば全サブドメインに反映
  ・CloudFront、Heroku等のIPが動的に変わるサービス

  決定フローチャート:
  ゾーンの頂点か？ → Yes → Aレコード（またはALIAS/ANAME）
                    → No  → 外部サービスか？ → Yes → CNAME
                                             → No  → Aレコード
```

### FAQ 3: 「DNS浸透」という表現は正しいか？

```
Q: 「DNSの浸透に時間がかかる」という表現をよく聞くが、これは正確か？

A: 「DNS浸透（propagation）」という表現は厳密には不正確であり、
   DNS業界では誤解を招く用語とされている。

  誤った理解:
  「DNSの変更が、世界中のDNSサーバーに徐々に伝わっていく」
  → まるで水が染み込むように情報が広がるイメージ
  → 実際にはこのようなメカニズムは存在しない

  正しい理解:
  「各キャッシュのTTLが切れ、新しいレコードが取得される」
  → DNSは「プッシュ型」ではなく「プル型」
  → 各リゾルバが独自のタイミングでキャッシュを更新する
  → TTLが切れた時点で新しい問い合わせが発生し、
    最新のレコードが取得される

  なぜ「浸透」に見えるのか:
  ・世界中のリゾルバが異なるタイミングでキャッシュを取得している
  ・あるユーザーには新IP、別のユーザーには旧IPが返される過渡期がある
  ・これが「徐々に浸透している」ように見える

  実態:
  変更前TTL = 3600秒の場合:
  ・変更直後にキャッシュが切れたリゾルバ → 即座に新しい値を取得
  ・変更直前にキャッシュを取得したリゾルバ → 最大3600秒後に新しい値を取得
  ・すべてのリゾルバが新しい値を持つのは最大3600秒後

  より正確な表現:
  × 「DNSの浸透に24時間かかる」
  ○ 「旧レコードのTTLが最大24時間であるため、
    すべてのキャッシュが更新されるまで最大24時間を要する」
```

### FAQ 4: 1つのドメインに複数のAレコードを設定するとどうなるか？

```
Q: 同じドメインに複数のAレコード（異なるIP）を設定した場合の動作は？

A: DNSラウンドロビンとして動作する。

  設定例:
  api.example.com.  300  IN  A  10.0.1.100
  api.example.com.  300  IN  A  10.0.1.101
  api.example.com.  300  IN  A  10.0.1.102

  動作:
  ・リゾルバは全てのAレコードを返す（順序はランダムまたはラウンドロビン）
  ・クライアントは通常、リスト内の最初のIPに接続を試みる
  ・最初のIPに接続できない場合、次のIPを試す（アプリ依存）

  注意点:
  ・ヘルスチェック機能がない → 障害サーバーにもトラフィックが向く
  ・均等な負荷分散は保証されない → クライアント実装に依存
  ・セッションの固定（スティッキー）がない
  ・本格的な負荷分散にはロードバランサー（ALB, NLB等）を使用すべき
```

---

## 13. まとめ

### 主要概念の整理

| 概念 | ポイント |
|------|---------|
| DNS | ドメイン名をIPアドレスに変換する分散データベースシステム |
| FQDN | ルートからの完全なドメイン名（末尾ドット付き） |
| 再帰クエリ | クライアント→リゾルバ間。リゾルバが完全な回答を返す |
| 反復クエリ | リゾルバ→権威サーバー間。リファラルを返す |
| TTL | キャッシュの有効期間。設計判断が運用品質を左右する |
| Aレコード | ドメイン→IPv4アドレスの対応 |
| CNAMEレコード | エイリアス。ゾーン頂点には使用不可 |
| MXレコード | メール配送先。優先度値を持つ |
| TXTレコード | SPF, DKIM, DMARC等のメール認証に使用 |
| NSレコード | ゾーンの権威ネームサーバーを指定 |
| SOAレコード | ゾーンの管理情報。ネガティブキャッシュTTLを含む |
| DNSSEC | DNS応答の真正性を暗号的に検証する拡張 |
| DoH / DoT | DNSクエリの暗号化プロトコル |
| ネガティブキャッシュ | NXDOMAINのキャッシュ。新規レコード作成時に注意 |

### キーポイント

1. **DNSは分散階層データベースである**: ルートサーバーからTLD、権威サーバーへと階層的に名前解決を行い、単一障害点を回避している
2. **再帰クエリと反復クエリの違いを理解する**: クライアント→リゾルバ間は再帰クエリ、リゾルバ→権威サーバー間は反復クエリで、それぞれ異なる責務を持つ
3. **TTLとキャッシュが性能とレジリエンスを支える**: 適切なTTL設定により、DNSクエリ負荷を削減し、応答速度を向上させつつ、変更時の柔軟性も確保できる

---

## まとめ

このガイドでは以下を学びました:

- DNSは分散階層データベースとして設計されており、ルートサーバー、TLDサーバー、権威サーバーの3層構造で名前解決を行うこと
- 再帰クエリ（クライアント→リゾルバ）と反復クエリ（リゾルバ→権威サーバー）の違いと、それぞれの責務分担
- A、AAAA、CNAME、MX、TXT、NS、SOA、SRV、PTR、CAAなど主要DNSレコードの用途と設定方法
- TTLの設計がキャッシュ効率とレコード変更時の反映速度のトレードオフであること
- DNSSEC、DoH（DNS over HTTPS）、DoT（DNS over TLS）によるDNSセキュリティ拡張の仕組み

---

## 次に読むべきガイド

- ../01-protocols/00-tcp.md -- TCP（トランスポート層プロトコル）
- ../01-protocols/01-tls.md -- TLS（トランスポート層セキュリティ）
- ../01-protocols/02-http.md -- HTTP（アプリケーション層プロトコル）

---

## 参考文献

1. Mockapetris, P. "Domain Names - Concepts and Facilities." RFC 1034, IETF, November 1987. https://www.rfc-editor.org/rfc/rfc1034
2. Mockapetris, P. "Domain Names - Implementation and Specification." RFC 1035, IETF, November 1987. https://www.rfc-editor.org/rfc/rfc1035
3. Hoffman, P. and McManus, P. "DNS Queries over HTTPS (DoH)." RFC 8484, IETF, October 2018. https://www.rfc-editor.org/rfc/rfc8484
4. Hu, Z. et al. "Specification for DNS over Transport Layer Security (TLS)." RFC 7858, IETF, May 2016. https://www.rfc-editor.org/rfc/rfc7858
5. Arends, R. et al. "DNS Security Introduction and Requirements." RFC 4033, IETF, March 2005. https://www.rfc-editor.org/rfc/rfc4033
6. Cotton, M. et al. "DNS Terminology." RFC 8499, IETF, January 2019. https://www.rfc-editor.org/rfc/rfc8499
7. Cloudflare Learning Center. "What is DNS?" https://www.cloudflare.com/learning/dns/what-is-dns/
8. Amazon Web Services. "Amazon Route 53 Developer Guide." https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/
