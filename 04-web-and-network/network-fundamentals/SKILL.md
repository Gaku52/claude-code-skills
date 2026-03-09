# ネットワーク基礎 完全ガイド

> インターネットを支えるプロトコル群を体系的に理解する。TCP/IP、HTTP、DNS、TLS から WebSocket、gRPC まで、エンジニアに必要なネットワーク知識を網羅。

## このSkillの対象者

- ネットワークの仕組みを基礎から学びたいエンジニア
- HTTP/HTTPSの深い理解が必要なWeb開発者
- セキュリティやパフォーマンスを意識した開発をしたい人

## 前提知識

- コンピュータの基本（ビット、バイト、2進数）
- 参照: [CS基礎](../../01-cs-fundamentals/computer-science-fundamentals/)

## ガイド一覧

### 00-introduction（導入）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-how-internet-works.md](docs/00-introduction/00-how-internet-works.md) | インターネットの仕組み | パケット通信、ISP、海底ケーブル、ルーティング |
| [01-osi-and-tcpip-model.md](docs/00-introduction/01-osi-and-tcpip-model.md) | OSI/TCP/IPモデル | 7層/4層モデル、各層の役割、プロトコルの対応 |
| [02-ip-addressing.md](docs/00-introduction/02-ip-addressing.md) | IPアドレッシング | IPv4/IPv6、サブネット、CIDR、NAT、DHCP |
| [03-dns.md](docs/00-introduction/03-dns.md) | DNS | 名前解決の仕組み、再帰/反復クエリ、DNSレコード |

### 01-protocols（プロトコル）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-tcp.md](docs/01-protocols/00-tcp.md) | TCP | 3wayハンドシェイク、フロー制御、輻輳制御 |
| [01-udp.md](docs/01-protocols/01-udp.md) | UDP | データグラム、リアルタイム通信、QUIC |
| [02-websocket.md](docs/01-protocols/02-websocket.md) | WebSocket | 双方向通信、ハンドシェイク、実装パターン |
| [03-grpc.md](docs/01-protocols/03-grpc.md) | gRPC | Protocol Buffers、ストリーミング、REST比較 |

### 02-http（HTTP）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-http-basics.md](docs/02-http/00-http-basics.md) | HTTP基礎 | メソッド、ステータスコード、ヘッダー |
| [01-http2-and-http3.md](docs/02-http/01-http2-and-http3.md) | HTTP/2とHTTP/3 | マルチプレキシング、サーバープッシュ、QUIC |
| [02-rest-api.md](docs/02-http/02-rest-api.md) | REST API設計 | リソース設計、バージョニング、HATEOAS |
| [03-caching.md](docs/02-http/03-caching.md) | HTTPキャッシュ | Cache-Control、ETag、CDN |
| [04-cors.md](docs/02-http/04-cors.md) | CORS | 同一オリジンポリシー、プリフライト、設定 |

### 03-security（セキュリティ）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-tls-ssl.md](docs/03-security/00-tls-ssl.md) | TLS/SSL | ハンドシェイク、証明書、暗号スイート |
| [01-authentication.md](docs/03-security/01-authentication.md) | 認証方式 | Basic、Bearer、OAuth 2.0、JWT |
| [02-common-attacks.md](docs/03-security/02-common-attacks.md) | ネットワーク攻撃 | MITM、DNS汚染、DDoS、対策 |

### 04-advanced（高度なトピック）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-load-balancing.md](docs/04-advanced/00-load-balancing.md) | ロードバランシング | L4/L7、アルゴリズム、ヘルスチェック |
| [01-cdn.md](docs/04-advanced/01-cdn.md) | CDN | エッジキャッシュ、配信最適化、設定 |
| [02-network-debugging.md](docs/04-advanced/02-network-debugging.md) | ネットワークデバッグ | curl, tcpdump, Wireshark, Chrome DevTools |
| [03-performance.md](docs/04-advanced/03-performance.md) | ネットワーク最適化 | レイテンシ削減、帯域最適化、接続管理 |

## 学習パス

```
基礎:     00-introduction → 01-protocols/00-01
Web:      02-http → 01-protocols/02-03
安全:     03-security
運用:     04-advanced
```

## FAQ

### Q1: ネットワークの学習はどの順番で進めるべき?
まず00-introduction（インターネットの仕組み、OSI/TCP/IPモデル、IPアドレッシング、DNS）で基礎を固め、次に01-protocols（TCP、UDP）でトランスポート層を理解してください。その上で02-http（HTTP基礎、HTTP/2・3、REST API、キャッシュ、CORS）を学び、03-security（TLS、認証、攻撃手法）で安全性を、04-advanced（ロードバランシング、CDN、デバッグ、パフォーマンス）で運用力を身につけるのが効率的です。

### Q2: Web開発者にネットワークの深い知識は本当に必要?
はい、確実に必要です。パフォーマンス問題の大半はネットワークに起因します。例えばHTTPキャッシュの誤設定、CORSエラー、TLSハンドシェイクの遅延、DNS解決の遅さなど、ネットワークの基礎を理解していないと原因特定すらできません。また、マイクロサービスアーキテクチャではgRPCやWebSocketの選択・設計判断にもネットワーク知識が不可欠です。

### Q3: このSkillと「ブラウザとWebプラットフォーム」Skillの違いは?
本Skillはプロトコルやインフラ側の知識（TCP/IP、HTTP、DNS、TLS、ルーティング等）に焦点を当てています。一方「ブラウザとWebプラットフォーム」はブラウザのレンダリングエンジン、DOM、Web API、ServiceWorker等のクライアント側技術を扱います。両者は補完関係にあり、Webの全体像を理解するには両方の学習が推奨されます。

## まとめ

このSkillでは以下を体系的に学べます:

- インターネットの物理構造からアプリケーション層までの全レイヤーの仕組み（OSIモデル、TCP/IP、ルーティング、DNS）
- TCP、UDP、WebSocket、gRPCなど主要プロトコルの設計思想・内部動作・使い分け
- HTTP/1.1からHTTP/3までの進化、REST API設計、キャッシュ戦略、CORS
- TLS/SSL、認証方式、ネットワーク攻撃と対策などのセキュリティ知識
- ロードバランシング、CDN、ネットワークデバッグ、パフォーマンス最適化の実践スキル

## 関連Skills

- [CS基礎](../../01-cs-fundamentals/computer-science-fundamentals/) — コンピュータサイエンス基礎
- [ブラウザとWebプラットフォーム](../browser-and-web-platform/) — ブラウザとWebプラットフォーム
- [セキュリティ基礎](../../06-data-and-security/security-fundamentals/) — セキュリティ基礎

## 参考文献

- [Computer Networking: A Top-Down Approach](https://gaia.cs.umass.edu/kurose_ross/index.php) - Kurose & Ross著、ネットワーク工学の定番教科書。トップダウンアプローチで各層を体系的に解説
- [MDN Web Docs - HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP) - Mozilla による HTTP の包括的リファレンス。ヘッダー、ステータスコード、キャッシュ等を網羅
- [RFC Editor](https://www.rfc-editor.org/) - インターネット標準（RFC）の公式リポジトリ。TCP、UDP、HTTP、TLS等すべてのプロトコル仕様の原典
- [High Performance Browser Networking](https://hpbn.co/) - Ilya Grigorik著、ブラウザネットワーキングの実践ガイド。TCP、TLS、HTTP/2、WebSocket等を詳解
- [Cloudflare Learning Center](https://www.cloudflare.com/learning/) - DNS、CDN、DDoS、TLS等のネットワーク技術をわかりやすく解説する学習リソース
