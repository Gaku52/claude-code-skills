# Go 実践ガイド

> Go はシンプルさと並行性を重視した言語。goroutine/channel による並行プログラミング、Web 開発、CLI ツール開発、テスト戦略まで、Go の実践的な全てを解説する。

## このSkillの対象者

- Go を実践的に学びたいエンジニア
- 高パフォーマンスなバックエンドを開発したい方
- CLI ツール/マイクロサービスを Go で構築したい方

## 前提知識

- 何らかのプログラミング言語の経験
- Web 開発の基礎知識

## 学習ガイド

### 00-basics — Go の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-basics/00-go-overview.md]] | Go の設計哲学、セットアップ、モジュール、基本構文 |
| 01 | [[docs/00-basics/01-types-and-structs.md]] | 型、構造体、interface、ポインタ、メソッド |
| 02 | [[docs/00-basics/02-error-handling.md]] | エラー処理パターン、errors.Is/As、カスタムエラー、sentinel |
| 03 | [[docs/00-basics/03-packages-and-modules.md]] | パッケージ設計、go.mod、依存関係管理、internal |
| 04 | [[docs/00-basics/04-testing.md]] | testing パッケージ、テーブル駆動テスト、ベンチマーク、fuzzing |

### 01-concurrency — 並行プログラミング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-concurrency/00-goroutines-and-channels.md]] | goroutine、channel、select、sync パッケージ |
| 01 | [[docs/01-concurrency/01-patterns.md]] | Fan-In/Fan-Out、Pipeline、Worker Pool、Context |
| 02 | [[docs/01-concurrency/02-advanced.md]] | データ競合検出、sync.Pool、atomic、singleflight |

### 02-web — Web 開発

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-web/00-net-http.md]] | net/http、ServeMux（1.22+パターンマッチ）、ミドルウェア |
| 01 | [[docs/02-web/01-frameworks.md]] | Echo/Gin/Chi 比較、ルーティング、バリデーション |
| 02 | [[docs/02-web/02-database.md]] | database/sql、sqlc、GORM、マイグレーション |
| 03 | [[docs/02-web/03-api-design.md]] | REST API 設計、gRPC、OpenAPI、認証 |

### 03-tools — ツール開発

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-tools/00-cli-development.md]] | cobra、CLI 設計、引数パース、設定ファイル |
| 01 | [[docs/03-tools/01-build-and-deploy.md]] | クロスコンパイル、Docker マルチステージ、リリース自動化 |
| 02 | [[docs/03-tools/02-profiling.md]] | pprof、trace、ベンチマーク、メモリプロファイリング |

## クイックリファレンス

```
Go 早見表:
  go mod init myapp     — モジュール初期化
  go run main.go        — 実行
  go build -o app       — ビルド
  go test ./...         — テスト
  go vet ./...          — 静的解析
  golangci-lint run     — リント
```

## 参考文献

1. Go. "Documentation." go.dev/doc, 2024.
2. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
3. Go. "Effective Go." go.dev/doc/effective_go, 2024.
