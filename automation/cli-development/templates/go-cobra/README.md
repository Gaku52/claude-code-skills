# Go CLI Template (Cobra)

完全な機能を持つ Go CLI テンプレート（Cobra + Viper）

## 特徴

- ✅ Cobra によるコマンド管理
- ✅ Viper による設定管理
- ✅ カラフルな出力
- ✅ テスト
- ✅ クロスコンパイル対応

## セットアップ

```bash
# 依存関係インストール
go mod download

# ビルド
go build -o mycli

# テスト
go test ./...

# 実行
./mycli --help
```

## 使用例

```bash
# プロジェクト作成
mycli create myapp

# テンプレート指定
mycli create myapp --template react

# プロジェクト一覧
mycli list

# プロジェクト削除
mycli delete myapp --force
```

## ディレクトリ構造

```
.
├── cmd/
│   ├── root.go          # ルートコマンド
│   ├── create.go        # create コマンド
│   ├── list.go          # list コマンド
│   └── delete.go        # delete コマンド
├── pkg/
│   └── project/         # ビジネスロジック
│       └── service.go
├── main.go              # エントリーポイント
├── go.mod
└── README.md
```

## 開発

```bash
# フォーマット
go fmt ./...

# リント
golangci-lint run

# テスト
go test ./... -v

# カバレッジ
go test ./... -cover
```

## ビルド & 配布

```bash
# ローカルビルド
go build -o mycli

# クロスコンパイル（Linux）
GOOS=linux GOARCH=amd64 go build -o mycli-linux-amd64

# クロスコンパイル（macOS）
GOOS=darwin GOARCH=amd64 go build -o mycli-darwin-amd64
GOOS=darwin GOARCH=arm64 go build -o mycli-darwin-arm64

# クロスコンパイル（Windows）
GOOS=windows GOARCH=amd64 go build -o mycli-windows-amd64.exe

# インストール
go install
```
