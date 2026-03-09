# API・ライブラリ設計 完全ガイド

> APIとライブラリの設計・実装・運用を体系的に学ぶ。REST/GraphQL設計、SDK開発、バージョニング、セキュリティ、ドキュメンテーション、モニタリングまで、APIに関する全てをカバー。

## このSkillの対象者

- API設計・開発に携わるバックエンドエンジニア
- SDKやライブラリを開発するエンジニア
- APIの品質とセキュリティを向上させたい開発者

## 前提知識

- HTTPの基本 → 参照: [ネットワーク基礎](../network-fundamentals/)
- プログラミングの基本 → 参照: プログラミング言語基礎

## ガイド一覧

### 00-api-design-principles（API設計原則）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-api-first-design.md](docs/00-api-design-principles/00-api-first-design.md) | API First設計 | API設計の哲学、契約先行開発、OpenAPI |
| [01-naming-and-conventions.md](docs/00-api-design-principles/01-naming-and-conventions.md) | 命名規則と慣例 | エンドポイント命名、レスポンス形式、エラー設計 |
| [02-versioning-strategy.md](docs/00-api-design-principles/02-versioning-strategy.md) | バージョニング戦略 | URI/ヘッダー方式、破壊的変更の管理 |
| [03-pagination-and-filtering.md](docs/00-api-design-principles/03-pagination-and-filtering.md) | ページネーションとフィルタリング | Cursor/Offset、ソート、検索 |

### 01-rest-and-graphql（REST と GraphQL）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-rest-best-practices.md](docs/01-rest-and-graphql/00-rest-best-practices.md) | RESTベストプラクティス | HATEOAS、idempotency、content negotiation |
| [01-graphql-fundamentals.md](docs/01-rest-and-graphql/01-graphql-fundamentals.md) | GraphQL基礎 | スキーマ、Query/Mutation、リゾルバー |
| [02-graphql-advanced.md](docs/01-rest-and-graphql/02-graphql-advanced.md) | GraphQL応用 | Subscription、DataLoader、キャッシュ |
| [03-rest-vs-graphql.md](docs/01-rest-and-graphql/03-rest-vs-graphql.md) | REST vs GraphQL | 選定基準、ハイブリッドアプローチ |

### 02-sdk-and-libraries（SDK・ライブラリ）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-sdk-design.md](docs/02-sdk-and-libraries/00-sdk-design.md) | SDK設計 | クライアントライブラリ、DX、型安全性 |
| [01-npm-package-development.md](docs/02-sdk-and-libraries/01-npm-package-development.md) | npmパッケージ開発 | package.json、ビルド、公開 |
| [02-api-documentation.md](docs/02-sdk-and-libraries/02-api-documentation.md) | APIドキュメンテーション | OpenAPI/Swagger、自動生成、Storybook |

### 03-api-security（APIセキュリティ）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-authentication-patterns.md](docs/03-api-security/00-authentication-patterns.md) | 認証パターン | OAuth 2.0、API Key、JWT、mTLS |
| [01-rate-limiting.md](docs/03-api-security/01-rate-limiting.md) | レート制限 | Token Bucket、Sliding Window、分散レート制限 |
| [02-input-validation.md](docs/03-api-security/02-input-validation.md) | 入力バリデーション | Zod、JSON Schema、サニタイゼーション |

### 04-api-operations（API運用）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-api-testing.md](docs/04-api-operations/00-api-testing.md) | APIテスト | 統合テスト、コントラクトテスト、負荷テスト |
| [01-monitoring-and-logging.md](docs/04-api-operations/01-monitoring-and-logging.md) | 監視とロギング | エラー率、レイテンシ、分散トレーシング |
| [02-api-gateway.md](docs/04-api-operations/02-api-gateway.md) | APIゲートウェイ | Kong、AWS API Gateway、認証/レート制限の一元化 |

## 学習パス

```
設計:    00-api-design-principles
実装:    01-rest-and-graphql → 02-sdk-and-libraries
安全:    03-api-security
運用:    04-api-operations
```

## FAQ

### Q1: REST APIとGraphQLのどちらを採用すべきか?
REST APIはリソースベースのCRUD操作に適しており、キャッシュ戦略が確立されている点が強みである。GraphQLは複雑なデータ取得やフロントエンド主導の開発に適しており、オーバーフェッチ・アンダーフェッチの問題を解消する。公開API（外部パートナー向け）にはREST、社内のBFF（Backend for Frontend）にはGraphQLを採用するハイブリッドアプローチが多くのプロジェクトで有効である。

### Q2: APIのバージョニングはいつから設計すべきか?
APIの初回設計段階から計画すべきである。バージョニング戦略を後付けで導入するのは既存クライアントへの影響が大きく困難を伴う。URIパス方式（/api/v1/）が最もシンプルで広く採用されている。バージョンアップ時には最低12ヶ月の並行運用期間を設け、非推奨化（Deprecation）通知を計画的に行うことが重要である。

### Q3: SDK開発で最も重視すべき点は何か?
開発者体験（DX）を最優先に設計すべきである。具体的には、型安全性の確保（TypeScript型定義の提供）、直感的なAPIインターフェース（Resource-basedパターン）、充実したエラーメッセージ（次に何をすべきかを示すactionableな情報）、そして完全なドキュメント（コード例付き）の4点が重要である。StripeやTwilioのSDKが設計の良い参考例となる。

## まとめ

このガイドでは以下を学びました:

- API First設計の哲学と、OpenAPI仕様を活用した契約先行開発の手法
- REST APIのベストプラクティス（HATEOAS、冪等性、エラーハンドリング）とGraphQLの基礎から応用
- SDK・ライブラリの設計原則（DX重視、型安全性、リトライ戦略）とnpmパッケージの公開ワークフロー
- APIセキュリティの実装パターン（OAuth 2.0、レート制限、入力バリデーション）
- API運用のためのテスト戦略、監視・ロギング、APIゲートウェイの活用方法

## 関連Skills

- [ネットワーク基礎](../network-fundamentals/) — ネットワーク基礎
- [セキュリティ基礎](../../06-data-and-security/security-fundamentals/) — セキュリティ基礎
- [Webアプリケーション開発](../web-application-development/) — Webアプリケーション開発

## 参考文献

- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html) - API仕様記述の業界標準。契約先行開発とコード生成の基盤となる仕様書
- [Stripe API Reference](https://docs.stripe.com/api) - REST API設計とSDK設計の業界標準として広く参照される優れた実装例
- [Google API Design Guide](https://cloud.google.com/apis/design) - Googleの大規模APIエコシステムから得られた設計原則とベストプラクティスの集大成
