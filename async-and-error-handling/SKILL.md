# 非同期処理とエラーハンドリング 完全ガイド

> 非同期処理は現代アプリケーションの基盤。エラーハンドリングはソフトウェアの信頼性の要。この2つの密接に関連する技術を、各言語の実装とベストプラクティスとともに体系的に解説。

## このSkillの対象者

- 非同期処理の仕組みとパターンを深く理解したいエンジニア
- エラーハンドリングのベストプラクティスを学びたい開発者
- Promise, async/await, Result型 の使い分けを整理したい人

## 前提知識

- プログラミングの基本（関数、制御構文）
- 参照: [[programming-language-fundamentals]]

## ガイド一覧

### 00-introduction（導入）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-sync-vs-async.md](docs/00-introduction/00-sync-vs-async.md) | 同期 vs 非同期 | ブロッキング/ノンブロッキング、なぜ非同期が必要か |
| [01-concurrency-models.md](docs/00-introduction/01-concurrency-models.md) | 並行モデル概要 | マルチスレッド、イベントループ、アクターモデル |

### 01-async-patterns（非同期パターン）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-callbacks.md](docs/01-async-patterns/00-callbacks.md) | コールバック | コールバック地獄、Node.jsのerror-firstパターン |
| [01-promises.md](docs/01-async-patterns/01-promises.md) | Promise | Promise チェーン、Promise.all/race/allSettled |
| [02-async-await.md](docs/01-async-patterns/02-async-await.md) | async/await | 各言語の実装、並行実行パターン |
| [03-reactive-streams.md](docs/01-async-patterns/03-reactive-streams.md) | Reactive Streams | RxJS、Observable、バックプレッシャー |

### 02-error-handling（エラーハンドリング）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-exceptions.md](docs/02-error-handling/00-exceptions.md) | 例外処理 | try/catch/finally、例外階層、checked/unchecked |
| [01-result-type.md](docs/02-error-handling/01-result-type.md) | Result型 | Rust Result、TypeScript never-throw、Go error |
| [02-error-boundaries.md](docs/02-error-handling/02-error-boundaries.md) | エラー境界 | React Error Boundary、グローバルハンドラ |
| [03-custom-errors.md](docs/02-error-handling/03-custom-errors.md) | カスタムエラー | エラー設計、エラーコード、ドメインエラー |

### 03-advanced（高度なトピック）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-event-loop.md](docs/03-advanced/00-event-loop.md) | イベントループ | Node.js/ブラウザのイベントループ詳解 |
| [01-cancellation.md](docs/03-advanced/01-cancellation.md) | キャンセル処理 | AbortController、CancellationToken、タイムアウト |
| [02-retry-and-backoff.md](docs/03-advanced/02-retry-and-backoff.md) | リトライ戦略 | 指数バックオフ、サーキットブレーカー |
| [03-structured-concurrency.md](docs/03-advanced/03-structured-concurrency.md) | 構造化並行性 | Kotlin coroutines、Swift structured concurrency |

### 04-practical（実践）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-api-error-design.md](docs/04-practical/00-api-error-design.md) | APIエラー設計 | HTTPステータス、エラーレスポンス設計、RFC 7807 |
| [01-logging-and-monitoring.md](docs/04-practical/01-logging-and-monitoring.md) | ログとモニタリング | 構造化ログ、エラートラッキング、Sentry |
| [02-testing-async.md](docs/04-practical/02-testing-async.md) | 非同期テスト | 非同期コードのテスト手法、モック、タイマー |
| [03-real-world-patterns.md](docs/04-practical/03-real-world-patterns.md) | 実践パターン集 | キュー処理、WebSocket、ファイルアップロード |

## 学習パス

```
基礎:     00-introduction → 01-async-patterns（00→02）
エラー:   02-error-handling（00→03）
応用:     01-async-patterns/03 → 03-advanced → 04-practical
```

## 関連Skills

- [[programming-language-fundamentals]] — 言語の基礎概念
- [[nodejs-development]] — Node.js開発
- [[backend-development]] — バックエンド開発
