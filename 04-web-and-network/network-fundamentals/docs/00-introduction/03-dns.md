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
