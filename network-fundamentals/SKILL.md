# ネットワーク基礎 完全ガイド

> インターネットを支えるプロトコル群を体系的に理解する。TCP/IP、HTTP、DNS、TLS から WebSocket、gRPC まで、エンジニアに必要なネットワーク知識を網羅。

## このSkillの対象者

- ネットワークの仕組みを基礎から学びたいエンジニア
- HTTP/HTTPSの深い理解が必要なWeb開発者
- セキュリティやパフォーマンスを意識した開発をしたい人

## 前提知識

- コンピュータの基本（ビット、バイト、2進数）
- 参照: [[computer-science-fundamentals]]

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

## 関連Skills

- [[computer-science-fundamentals]] — CS基礎
- [[browser-and-web-platform]] — ブラウザとWeb（予定）
- [[security-fundamentals]] — セキュリティ基礎（予定）
