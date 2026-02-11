# Go CLI開発ガイド

> cobra、flag、promptuiを活用して本格的なコマンドラインツールをGoで構築する

## この章で学ぶこと

1. **cobra** フレームワークを使ったサブコマンド付きCLIアプリケーションの設計と実装
2. **標準flag** パッケージと **pflag** の違い、フラグ管理のベストプラクティス
3. **promptui** による対話型CLI（選択メニュー・入力プロンプト）の構築法

---

## 1. Go CLIの全体像

### CLI フレームワーク選定フロー

```
CLI ツールを作りたい
        |
        +-- シンプル（フラグ数個）
        |       |
        |       v
        |   標準 flag パッケージ
        |
        +-- サブコマンドあり
        |       |
        |       v
        |   cobra（業界標準）
        |
        +-- 対話型 UI が必要
                |
                v
            promptui / survey
```

### CLI アーキテクチャ

```
+------------------------------------------------------+
|                     main.go                          |
|  func main() { cmd.Execute() }                       |
+------------------------------------------------------+
        |
        v
+------------------------------------------------------+
|                   cmd/root.go                        |
|  rootCmd: アプリ名、バージョン、グローバルフラグ        |
+------------------------------------------------------+
        |
        +-------> cmd/serve.go   (serve サブコマンド)
        |
        +-------> cmd/migrate.go (migrate サブコマンド)
        |
        +-------> cmd/config.go  (config サブコマンド)
                      |
                      +-> cmd/config_set.go (config set)
                      +-> cmd/config_get.go (config get)
```

---

## 2. 標準 flag パッケージ

### コード例1: flag パッケージの基本

```go
package main

import (
    "flag"
    "fmt"
    "os"
)

func main() {
    // フラグ定義
    host := flag.String("host", "localhost", "サーバーホスト名")
    port := flag.Int("port", 8080, "サーバーポート番号")
    verbose := flag.Bool("verbose", false, "詳細出力を有効にする")

    // カスタムUsage
    flag.Usage = func() {
        fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\nOptions:\n", os.Args[0])
        flag.PrintDefaults()
    }

    flag.Parse()

    // 残りの引数（非フラグ）
    args := flag.Args()

    if *verbose {
        fmt.Printf("Host: %s, Port: %d\n", *host, *port)
        fmt.Printf("Args: %v\n", args)
    }

    fmt.Printf("サーバー起動: %s:%d\n", *host, *port)
}
```

```bash
$ myapp -host 0.0.0.0 -port 3000 -verbose extra_arg
Host: 0.0.0.0, Port: 3000
Args: [extra_arg]
サーバー起動: 0.0.0.0:3000
```

---

## 3. cobra フレームワーク

### インストール

```bash
go get github.com/spf13/cobra@latest
go install github.com/spf13/cobra-cli@latest
```

### コード例2: cobra のルートコマンド

```go
// cmd/root.go
package cmd

import (
    "fmt"
    "os"

    "github.com/spf13/cobra"
    "github.com/spf13/viper"
)

var (
    cfgFile string
    verbose bool
)

var rootCmd = &cobra.Command{
    Use:     "mytool",
    Short:   "My awesome CLI tool",
    Long:    `mytool はGoで構築された多機能CLIツールです。`,
    Version: "1.0.0",
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}

func init() {
    cobra.OnInitialize(initConfig)

    // Persistent Flags: 全サブコマンドで使える
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "",
        "設定ファイルパス (デフォルト: $HOME/.mytool.yaml)")
    rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false,
        "詳細出力")

    // viper と連携
    viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
}

func initConfig() {
    if cfgFile != "" {
        viper.SetConfigFile(cfgFile)
    } else {
        home, _ := os.UserHomeDir()
        viper.AddConfigPath(home)
        viper.SetConfigName(".mytool")
    }
    viper.AutomaticEnv()
    viper.ReadInConfig()
}
```

### コード例3: サブコマンドの追加

```go
// cmd/serve.go
package cmd

import (
    "fmt"
    "net/http"

    "github.com/spf13/cobra"
)

var (
    servePort int
    serveHost string
)

var serveCmd = &cobra.Command{
    Use:   "serve",
    Short: "HTTPサーバーを起動する",
    Long:  `HTTPサーバーを指定されたホストとポートで起動します。`,
    Example: `  mytool serve
  mytool serve --port 3000
  mytool serve --host 0.0.0.0 --port 8080`,
    RunE: func(cmd *cobra.Command, args []string) error {
        addr := fmt.Sprintf("%s:%d", serveHost, servePort)
        fmt.Printf("サーバー起動: http://%s\n", addr)

        mux := http.NewServeMux()
        mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
            fmt.Fprintln(w, "Hello from mytool!")
        })
        return http.ListenAndServe(addr, mux)
    },
}

func init() {
    rootCmd.AddCommand(serveCmd)

    // Local Flags: このコマンド専用
    serveCmd.Flags().IntVarP(&servePort, "port", "p", 8080, "ポート番号")
    serveCmd.Flags().StringVar(&serveHost, "host", "localhost", "ホスト名")
}
```

### コード例4: ネストしたサブコマンド

```go
// cmd/config.go
package cmd

import "github.com/spf13/cobra"

var configCmd = &cobra.Command{
    Use:   "config",
    Short: "設定を管理する",
}

var configSetCmd = &cobra.Command{
    Use:   "set [key] [value]",
    Short: "設定値を変更する",
    Args:  cobra.ExactArgs(2),
    RunE: func(cmd *cobra.Command, args []string) error {
        key, value := args[0], args[1]
        viper.Set(key, value)
        return viper.WriteConfig()
    },
}

var configGetCmd = &cobra.Command{
    Use:   "get [key]",
    Short: "設定値を表示する",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Println(viper.GetString(args[0]))
    },
}

func init() {
    rootCmd.AddCommand(configCmd)
    configCmd.AddCommand(configSetCmd)
    configCmd.AddCommand(configGetCmd)
}
```

---

## 4. cobra フラグ体系

### Persistent Flags vs Local Flags 比較表

| 項目 | Persistent Flags | Local Flags |
|------|-----------------|-------------|
| スコープ | 定義コマンド＋全子コマンド | 定義コマンドのみ |
| 定義方法 | `PersistentFlags()` | `Flags()` |
| 用途 | `--verbose`, `--config` など共通設定 | `--port`, `--output` などコマンド固有 |
| viper連携 | `BindPFlag` で永続化可能 | 同様に可能 |
| 継承 | 子コマンドが自動的に継承 | 継承されない |

### flag vs pflag vs cobra 比較表

| 機能 | 標準 flag | pflag | cobra |
|------|----------|-------|-------|
| POSIX形式 `--flag` | `-flag` のみ | 対応 | 対応（pflag内蔵） |
| 短縮形 `-v` | 非対応 | 対応 | 対応 |
| サブコマンド | 非対応 | 非対応 | 対応 |
| 自動ヘルプ | 基本的 | 基本的 | リッチ |
| シェル補完 | 非対応 | 非対応 | bash/zsh/fish/powershell |
| 引数バリデーション | 手動 | 手動 | `Args` で宣言的に |
| 設定ファイル連携 | 手動 | 手動 | viper統合 |

---

## 5. promptui による対話型CLI

### コード例5: 選択メニューと入力プロンプト

```go
package main

import (
    "fmt"
    "strings"

    "github.com/manifoldco/promptui"
)

func main() {
    // 選択プロンプト
    envSelect := promptui.Select{
        Label: "デプロイ環境を選択",
        Items: []string{"development", "staging", "production"},
        Templates: &promptui.SelectTemplates{
            Active:   "▸ {{ . | cyan }}",
            Inactive: "  {{ . }}",
            Selected: "✓ {{ . | green }}",
        },
    }
    _, env, err := envSelect.Run()
    if err != nil {
        fmt.Printf("選択キャンセル: %v\n", err)
        return
    }

    // production の場合、確認プロンプト
    if env == "production" {
        confirm := promptui.Prompt{
            Label:     "本番環境へのデプロイを確認 (yes/no)",
            IsConfirm: true,
        }
        _, err := confirm.Run()
        if err != nil {
            fmt.Println("デプロイをキャンセルしました")
            return
        }
    }

    // 入力プロンプト（バリデーション付き）
    tagPrompt := promptui.Prompt{
        Label: "リリースタグ (例: v1.2.3)",
        Validate: func(input string) error {
            if !strings.HasPrefix(input, "v") {
                return fmt.Errorf("タグは 'v' で始まる必要があります")
            }
            return nil
        },
    }
    tag, err := tagPrompt.Run()
    if err != nil {
        return
    }

    fmt.Printf("デプロイ実行: env=%s, tag=%s\n", env, tag)
}
```

### 対話フロー

```
$ mytool deploy

? デプロイ環境を選択:
    development
  ▸ staging
    production

✓ staging

? リリースタグ (例: v1.2.3): v1.5.0

デプロイ実行: env=staging, tag=v1.5.0
```

---

## 6. CLI 設計のベストプラクティス

### コード例6: エラーハンドリングと終了コード

```go
var rootCmd = &cobra.Command{
    // RunE を使い、エラーを返す
    RunE: func(cmd *cobra.Command, args []string) error {
        if err := doSomething(); err != nil {
            // ユーザー向けメッセージはラップして返す
            return fmt.Errorf("処理に失敗しました: %w", err)
        }
        return nil
    },
    // SilenceUsage: エラー時にUsage を表示しない
    SilenceUsage: true,
    // SilenceErrors: cobra のデフォルトエラー表示を抑制
    SilenceErrors: true,
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        // エラーを stderr に出力
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

---

## 7. アンチパターン

### アンチパターン1: main() にロジックを直書き

```go
// NG: テスト不能、再利用不能
func main() {
    flag.Parse()
    db, _ := sql.Open("postgres", *dsn)
    rows, _ := db.Query("SELECT ...")
    for rows.Next() {
        // 全処理がmainに集中
    }
}

// OK: ロジックを分離し、mainはエントリポイントのみ
func main() {
    if err := run(os.Args[1:], os.Stdout); err != nil {
        fmt.Fprintf(os.Stderr, "error: %v\n", err)
        os.Exit(1)
    }
}

func run(args []string, stdout io.Writer) error {
    // テスト可能なロジック
    cfg, err := parseFlags(args)
    if err != nil {
        return err
    }
    return execute(cfg, stdout)
}
```

### アンチパターン2: グローバル変数の乱用

```go
// NG: 全てグローバルで管理
var (
    db      *sql.DB
    logger  *log.Logger
    config  Config
    client  *http.Client
)

// OK: 構造体にまとめて依存注入
type App struct {
    DB     *sql.DB
    Logger *log.Logger
    Config Config
    Client *http.Client
}

func NewApp(cfg Config) (*App, error) {
    db, err := sql.Open("postgres", cfg.DSN)
    if err != nil {
        return nil, err
    }
    return &App{
        DB:     db,
        Logger: log.New(os.Stderr, "", log.LstdFlags),
        Config: cfg,
        Client: &http.Client{Timeout: 30 * time.Second},
    }, nil
}
```

---

## FAQ

### Q1. cobra と urfave/cli、どちらを選ぶべき？

cobraはDocker、Kubernetes、Hugo、GitHub CLIなど大規模プロジェクトで採用されており、エコシステムが充実している。urfave/cliはよりシンプルだが、シェル補完やviper連携などの機能はcobraが優る。新規プロジェクトではcobraを推奨。

### Q2. CLIツールのバイナリ配布はどうする？

GoReleaserを使うと、`git tag` をトリガーにクロスコンパイル・GitHub Releases・Homebrew Tap・Docker Image の自動生成ができる。`.goreleaser.yaml` を設定してGitHub Actionsと連携させるのが標準的。

### Q3. 設定ファイル・環境変数・フラグの優先順位は？

viperの標準優先順位は: 1) 明示的な `Set()` 呼び出し → 2) フラグ → 3) 環境変数 → 4) 設定ファイル → 5) デフォルト値。この順序により、ユーザーは設定ファイルをベースにしつつ、環境変数やフラグで上書きできる。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 標準flag | シンプルなCLIに十分、`-flag` 形式 |
| pflag | POSIX互換 `--flag`、短縮形 `-f` 対応 |
| cobra | サブコマンド・ヘルプ・補完の業界標準 |
| viper | 設定ファイル・環境変数・フラグの統合管理 |
| promptui | 対話型選択メニュー・入力プロンプト |
| RunE | エラーを返すコマンド実行（Run より推奨） |
| SilenceUsage | エラー時のUsage表示抑制 |
| GoReleaser | クロスコンパイル・配布の自動化 |

---

## 次に読むべきガイド

- **03-tools/01-generics.md** — ジェネリクス：型パラメータ、制約
- **03-tools/03-deployment.md** — デプロイ：Docker、クロスコンパイル
- **02-web/04-testing.md** — テスト：table-driven tests、testify、httptest

---

## 参考文献

1. **spf13/cobra GitHub** https://github.com/spf13/cobra
2. **spf13/viper GitHub** https://github.com/spf13/viper
3. **manifoldco/promptui GitHub** https://github.com/manifoldco/promptui
4. **GoReleaser 公式ドキュメント** https://goreleaser.com/
