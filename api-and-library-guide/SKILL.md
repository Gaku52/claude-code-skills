# API・ライブラリ設計 完全ガイド

> APIとライブラリの設計・実装・運用を体系的に学ぶ。REST/GraphQL設計、SDK開発、バージョニング、セキュリティ、ドキュメンテーション、モニタリングまで、APIに関する全てをカバー。

## このSkillの対象者

- API設計・開発に携わるバックエンドエンジニア
- SDKやライブラリを開発するエンジニア
- APIの品質とセキュリティを向上させたい開発者

## 前提知識

- HTTPの基本 → 参照: [[network-fundamentals]]
- プログラミングの基本 → 参照: [[programming-language-fundamentals]]

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

## 関連Skills

- [[network-fundamentals]] — ネットワーク基礎
- [[backend-development]] — バックエンド開発
- [[testing-strategy]] — テスト戦略
- [[security-fundamentals]] — セキュリティ基礎
