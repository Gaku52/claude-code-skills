# Go CLI開発ガイド

> cobra、flag、promptuiを活用して本格的なコマンドラインツールをGoで構築する

## この章で学ぶこと

1. **cobra** フレームワークを使ったサブコマンド付きCLIアプリケーションの設計と実装
2. **標準flag** パッケージと **pflag** の違い、フラグ管理のベストプラクティス
3. **promptui** による対話型CLI（選択メニュー・入力プロンプト）の構築法
4. **viper** による設定管理（ファイル・環境変数・フラグの統合）
5. **シェル補完** と **ドキュメント自動生成** の実装

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
        |       |
        |       v
        |   promptui / survey
        |
        +-- 超軽量（依存なし）
                |
                v
            手動パース / 標準 flag + サブコマンド手動実装
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
        |              |
        |              +-> cmd/config_set.go (config set)
        |              +-> cmd/config_get.go (config get)
        |              +-> cmd/config_list.go (config list)
        |
        +-------> cmd/version.go (version サブコマンド)
        |
        +-------> cmd/completion.go (シェル補完)
```

### プロジェクトディレクトリ構成

```
myapp/
├── main.go                    # エントリポイント（最小限）
├── cmd/                       # コマンド定義
│   ├── root.go               # ルートコマンド
│   ├── serve.go              # serve サブコマンド
│   ├── migrate.go            # migrate サブコマンド
│   ├── config.go             # config サブコマンド群
│   ├── version.go            # version サブコマンド
│   └── completion.go         # シェル補完コマンド
├── internal/                  # 内部パッケージ
│   ├── config/               # 設定管理
│   ├── server/               # サーバーロジック
│   └── migration/            # マイグレーションロジック
├── pkg/                       # 外部公開パッケージ（任意）
├── .goreleaser.yaml           # GoReleaserの設定
├── Makefile                   # ビルド・テストコマンド
└── go.mod
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

### コード例2: FlagSet を使ったサブコマンド実装

```go
package main

import (
    "flag"
    "fmt"
    "os"
)

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: myapp <command> [options]")
        fmt.Println("Commands: serve, migrate, version")
        os.Exit(1)
    }

    // サブコマンドごとにFlagSetを定義
    serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
    servePort := serveCmd.Int("port", 8080, "ポート番号")
    serveHost := serveCmd.String("host", "localhost", "ホスト名")

    migrateCmd := flag.NewFlagSet("migrate", flag.ExitOnError)
    migrateDir := migrateCmd.String("dir", "./migrations", "マイグレーションディレクトリ")
    migrateDSN := migrateCmd.String("dsn", "", "データベース接続文字列")

    switch os.Args[1] {
    case "serve":
        serveCmd.Parse(os.Args[2:])
        fmt.Printf("サーバー起動: %s:%d\n", *serveHost, *servePort)

    case "migrate":
        migrateCmd.Parse(os.Args[2:])
        if *migrateDSN == "" {
            fmt.Fprintln(os.Stderr, "Error: -dsn フラグは必須です")
            migrateCmd.Usage()
            os.Exit(1)
        }
        fmt.Printf("マイグレーション実行: dir=%s, dsn=%s\n", *migrateDir, *migrateDSN)

    case "version":
        fmt.Println("myapp v1.0.0")

    default:
        fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
        os.Exit(1)
    }
}
```

### コード例3: カスタムフラグ型

```go
// StringSlice はカンマ区切りまたは複数指定のフラグ
type StringSlice []string

func (s *StringSlice) String() string {
    return fmt.Sprintf("%v", *s)
}

func (s *StringSlice) Set(value string) error {
    *s = append(*s, value)
    return nil
}

// Duration型のカスタムフラグ
type DurationFlag struct {
    value time.Duration
}

func (d *DurationFlag) String() string {
    return d.value.String()
}

func (d *DurationFlag) Set(s string) error {
    dur, err := time.ParseDuration(s)
    if err != nil {
        return fmt.Errorf("無効なDuration: %s", s)
    }
    d.value = dur
    return nil
}

func main() {
    var tags StringSlice
    flag.Var(&tags, "tag", "タグ（複数指定可）")

    var timeout DurationFlag
    timeout.value = 30 * time.Second
    flag.Var(&timeout, "timeout", "タイムアウト（例: 30s, 5m）")

    flag.Parse()

    fmt.Printf("Tags: %v\n", tags)
    fmt.Printf("Timeout: %v\n", timeout.value)
}
```

```bash
$ myapp -tag web -tag api -tag v2 -timeout 5m
Tags: [web api v2]
Timeout: 5m0s
```

---

## 3. cobra フレームワーク

### インストール

```bash
go get github.com/spf13/cobra@latest
go install github.com/spf13/cobra-cli@latest
```

### コード例4: cobra のルートコマンド

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
        viper.AddConfigPath(".")
        viper.SetConfigName(".mytool")
        viper.SetConfigType("yaml")
    }
    viper.SetEnvPrefix("MYTOOL")
    viper.AutomaticEnv()
    viper.ReadInConfig()
}
```

### コード例5: サブコマンドの追加

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

### コード例6: ネストしたサブコマンド

```go
// cmd/config.go
package cmd

import (
    "fmt"

    "github.com/spf13/cobra"
    "github.com/spf13/viper"
)

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
        val := viper.GetString(args[0])
        if val == "" {
            fmt.Printf("キー '%s' は設定されていません\n", args[0])
            return
        }
        fmt.Println(val)
    },
}

var configListCmd = &cobra.Command{
    Use:   "list",
    Short: "全設定を一覧表示する",
    Run: func(cmd *cobra.Command, args []string) {
        for key, val := range viper.AllSettings() {
            fmt.Printf("%s = %v\n", key, val)
        }
    },
}

var configInitCmd = &cobra.Command{
    Use:   "init",
    Short: "設定ファイルを初期化する",
    RunE: func(cmd *cobra.Command, args []string) error {
        // デフォルト値を設定
        viper.SetDefault("server.host", "localhost")
        viper.SetDefault("server.port", 8080)
        viper.SetDefault("database.driver", "postgres")
        viper.SetDefault("log.level", "info")
        viper.SetDefault("log.format", "json")

        if err := viper.SafeWriteConfig(); err != nil {
            return fmt.Errorf("設定ファイルの作成に失敗: %w", err)
        }
        fmt.Println("設定ファイルを作成しました")
        return nil
    },
}

func init() {
    rootCmd.AddCommand(configCmd)
    configCmd.AddCommand(configSetCmd)
    configCmd.AddCommand(configGetCmd)
    configCmd.AddCommand(configListCmd)
    configCmd.AddCommand(configInitCmd)
}
```

### コード例7: 引数バリデーション

```go
// cobra の引数バリデーション関数一覧
var exampleCmds = []*cobra.Command{
    // 引数なし
    {
        Use:  "status",
        Args: cobra.NoArgs,
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // 正確に N 個
    {
        Use:  "rename [old] [new]",
        Args: cobra.ExactArgs(2),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // 最小 N 個
    {
        Use:  "add [file...]",
        Args: cobra.MinimumNArgs(1),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // 最大 N 個
    {
        Use:  "show [name]",
        Args: cobra.MaximumNArgs(1),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // 範囲
    {
        Use:  "between [args...]",
        Args: cobra.RangeArgs(1, 3),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
}

// カスタムバリデーション
var deployCmd = &cobra.Command{
    Use:   "deploy [environment]",
    Short: "指定環境にデプロイする",
    Args: func(cmd *cobra.Command, args []string) error {
        if len(args) != 1 {
            return fmt.Errorf("環境名を1つ指定してください")
        }
        validEnvs := map[string]bool{
            "development": true,
            "staging":     true,
            "production":  true,
        }
        if !validEnvs[args[0]] {
            return fmt.Errorf("無効な環境名: %s（development, staging, production のいずれかを指定）", args[0])
        }
        return nil
    },
    RunE: func(cmd *cobra.Command, args []string) error {
        env := args[0]
        fmt.Printf("%s 環境にデプロイします\n", env)
        return nil
    },
}

// ValidArgsFunction: 動的な補完候補
var connectCmd = &cobra.Command{
    Use:   "connect [server]",
    Short: "サーバーに接続する",
    Args:  cobra.ExactArgs(1),
    ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
        if len(args) != 0 {
            return nil, cobra.ShellCompDirectiveNoFileComp
        }
        // 動的にサーバー一覧を取得
        servers := []string{"web-01", "web-02", "db-01", "cache-01"}
        return servers, cobra.ShellCompDirectiveNoFileComp
    },
    RunE: func(cmd *cobra.Command, args []string) error {
        fmt.Printf("Connecting to %s...\n", args[0])
        return nil
    },
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

### コード例8: フラグの高度な使い方

```go
// cmd/serve.go
func init() {
    rootCmd.AddCommand(serveCmd)

    // 基本的なフラグ定義
    serveCmd.Flags().IntVarP(&port, "port", "p", 8080, "ポート番号")
    serveCmd.Flags().StringVar(&host, "host", "localhost", "ホスト名")

    // 必須フラグ
    serveCmd.Flags().StringVar(&certFile, "cert", "", "TLS証明書ファイル")
    serveCmd.MarkFlagRequired("cert")

    // ファイルパスの補完を有効化
    serveCmd.MarkFlagFilename("cert", "pem", "crt")

    // 相互排他フラグ
    serveCmd.Flags().BoolVar(&useTLS, "tls", false, "TLSを有効にする")
    serveCmd.Flags().BoolVar(&useHTTP2, "h2c", false, "HTTP/2 Cleartext を使う")
    serveCmd.MarkFlagsMutuallyExclusive("tls", "h2c")

    // グループ化（片方を指定したら両方必須）
    serveCmd.Flags().StringVar(&certFile, "cert-file", "", "証明書ファイル")
    serveCmd.Flags().StringVar(&keyFile, "key-file", "", "秘密鍵ファイル")
    serveCmd.MarkFlagsRequiredTogether("cert-file", "key-file")

    // 環境変数との連携
    viper.BindPFlag("server.port", serveCmd.Flags().Lookup("port"))
    viper.BindPFlag("server.host", serveCmd.Flags().Lookup("host"))

    // デフォルト値を環境変数から取得
    viper.BindEnv("server.port", "MYAPP_PORT")
    viper.BindEnv("server.host", "MYAPP_HOST")
}
```

### コード例9: フラグのカスタムバリデーション

```go
// ポート番号のバリデーション
var serveCmd = &cobra.Command{
    Use: "serve",
    PreRunE: func(cmd *cobra.Command, args []string) error {
        port, _ := cmd.Flags().GetInt("port")
        if port < 1 || port > 65535 {
            return fmt.Errorf("ポート番号は 1-65535 の範囲で指定してください: %d", port)
        }
        if port < 1024 {
            fmt.Fprintf(os.Stderr, "警告: ポート %d は特権ポートです（root権限が必要）\n", port)
        }
        return nil
    },
    RunE: func(cmd *cobra.Command, args []string) error {
        // メインロジック
        return nil
    },
}

// 列挙型フラグ
type LogLevel string

const (
    LogDebug LogLevel = "debug"
    LogInfo  LogLevel = "info"
    LogWarn  LogLevel = "warn"
    LogError LogLevel = "error"
)

func (l *LogLevel) String() string { return string(*l) }
func (l *LogLevel) Set(v string) error {
    switch v {
    case "debug", "info", "warn", "error":
        *l = LogLevel(v)
        return nil
    default:
        return fmt.Errorf("無効なログレベル: %s（debug, info, warn, error のいずれか）", v)
    }
}
func (l *LogLevel) Type() string { return "LogLevel" }

var logLevel LogLevel = LogInfo

func init() {
    rootCmd.PersistentFlags().Var(&logLevel, "log-level", "ログレベル (debug|info|warn|error)")
    rootCmd.RegisterFlagCompletionFunc("log-level", func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
        return []string{"debug", "info", "warn", "error"}, cobra.ShellCompDirectiveNoFileComp
    })
}
```

---

## 5. viper による設定管理

### 設定の優先順位

```
+----------------------------------------------------------+
|  viper 設定優先順位（上が高い）                             |
+----------------------------------------------------------+
|                                                          |
|  1. viper.Set() による明示的な設定                        |
|     ↓                                                    |
|  2. コマンドラインフラグ（--port 3000）                   |
|     ↓                                                    |
|  3. 環境変数（MYAPP_PORT=3000）                          |
|     ↓                                                    |
|  4. 設定ファイル（.mytool.yaml）                         |
|     ↓                                                    |
|  5. キー/バリューストア（etcd, Consul）                   |
|     ↓                                                    |
|  6. viper.SetDefault() によるデフォルト値                 |
+----------------------------------------------------------+
```

### コード例10: viperの包括的な設定管理

```go
package config

import (
    "fmt"
    "strings"
    "time"

    "github.com/spf13/viper"
)

// Config はアプリケーションの設定構造体
type Config struct {
    Server   ServerConfig   `mapstructure:"server"`
    Database DatabaseConfig `mapstructure:"database"`
    Log      LogConfig      `mapstructure:"log"`
    Auth     AuthConfig     `mapstructure:"auth"`
}

type ServerConfig struct {
    Host         string        `mapstructure:"host"`
    Port         int           `mapstructure:"port"`
    ReadTimeout  time.Duration `mapstructure:"read_timeout"`
    WriteTimeout time.Duration `mapstructure:"write_timeout"`
    MaxConns     int           `mapstructure:"max_connections"`
}

type DatabaseConfig struct {
    Driver   string `mapstructure:"driver"`
    Host     string `mapstructure:"host"`
    Port     int    `mapstructure:"port"`
    Name     string `mapstructure:"name"`
    User     string `mapstructure:"user"`
    Password string `mapstructure:"password"`
    SSLMode  string `mapstructure:"ssl_mode"`
}

type LogConfig struct {
    Level  string `mapstructure:"level"`
    Format string `mapstructure:"format"`
    Output string `mapstructure:"output"`
}

type AuthConfig struct {
    JWTSecret    string        `mapstructure:"jwt_secret"`
    TokenExpiry  time.Duration `mapstructure:"token_expiry"`
    RefreshToken bool          `mapstructure:"refresh_token"`
}

func Load() (*Config, error) {
    // デフォルト値
    viper.SetDefault("server.host", "0.0.0.0")
    viper.SetDefault("server.port", 8080)
    viper.SetDefault("server.read_timeout", "30s")
    viper.SetDefault("server.write_timeout", "30s")
    viper.SetDefault("server.max_connections", 100)
    viper.SetDefault("database.driver", "postgres")
    viper.SetDefault("database.port", 5432)
    viper.SetDefault("database.ssl_mode", "disable")
    viper.SetDefault("log.level", "info")
    viper.SetDefault("log.format", "json")
    viper.SetDefault("log.output", "stdout")
    viper.SetDefault("auth.token_expiry", "24h")

    // 環境変数のバインド
    viper.SetEnvPrefix("MYAPP")
    viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
    viper.AutomaticEnv()

    // 設定ファイルの読み込み
    if err := viper.ReadInConfig(); err != nil {
        if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
            return nil, fmt.Errorf("設定ファイルの読み込みに失敗: %w", err)
        }
        // 設定ファイルが見つからない場合はデフォルト値で続行
    }

    var cfg Config
    if err := viper.Unmarshal(&cfg); err != nil {
        return nil, fmt.Errorf("設定のデコードに失敗: %w", err)
    }

    return &cfg, nil
}
```

### 設定ファイルの例

```yaml
# .mytool.yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: "30s"
  write_timeout: "30s"
  max_connections: 200

database:
  driver: "postgres"
  host: "localhost"
  port: 5432
  name: "mydb"
  user: "admin"
  password: "${DB_PASSWORD}"  # 環境変数で上書き推奨
  ssl_mode: "require"

log:
  level: "info"
  format: "json"
  output: "stdout"

auth:
  jwt_secret: ""  # 環境変数 MYAPP_AUTH_JWT_SECRET で設定
  token_expiry: "24h"
  refresh_token: true
```

### コード例11: 設定ファイルのホットリロード

```go
package config

import (
    "log"
    "sync"

    "github.com/fsnotify/fsnotify"
    "github.com/spf13/viper"
)

type ConfigWatcher struct {
    mu        sync.RWMutex
    config    *Config
    callbacks []func(*Config)
}

func NewConfigWatcher() *ConfigWatcher {
    return &ConfigWatcher{}
}

func (w *ConfigWatcher) Watch() {
    viper.OnConfigChange(func(e fsnotify.Event) {
        log.Printf("設定ファイルが変更されました: %s", e.Name)

        w.mu.Lock()
        defer w.mu.Unlock()

        var newCfg Config
        if err := viper.Unmarshal(&newCfg); err != nil {
            log.Printf("設定の再読み込みに失敗: %v", err)
            return
        }

        w.config = &newCfg

        // コールバックを実行
        for _, cb := range w.callbacks {
            cb(&newCfg)
        }
    })
    viper.WatchConfig()
}

func (w *ConfigWatcher) OnChange(cb func(*Config)) {
    w.mu.Lock()
    defer w.mu.Unlock()
    w.callbacks = append(w.callbacks, cb)
}

func (w *ConfigWatcher) Get() *Config {
    w.mu.RLock()
    defer w.mu.RUnlock()
    return w.config
}
```

---

## 6. promptui による対話型CLI

### コード例12: 選択メニューと入力プロンプト

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

### コード例13: 構造体を使ったリッチな選択メニュー

```go
type Server struct {
    Name   string
    Host   string
    Region string
    Status string
}

func selectServer() (*Server, error) {
    servers := []Server{
        {Name: "web-01", Host: "10.0.1.10", Region: "ap-northeast-1", Status: "running"},
        {Name: "web-02", Host: "10.0.1.11", Region: "ap-northeast-1", Status: "running"},
        {Name: "web-03", Host: "10.0.2.10", Region: "us-east-1", Status: "stopped"},
        {Name: "db-01", Host: "10.0.1.20", Region: "ap-northeast-1", Status: "running"},
    }

    templates := &promptui.SelectTemplates{
        Label:    "{{ . }}?",
        Active:   "▸ {{ .Name | cyan }} ({{ .Host }}) [{{ .Region }}] {{ if eq .Status \"running\" }}{{ .Status | green }}{{ else }}{{ .Status | red }}{{ end }}",
        Inactive: "  {{ .Name }} ({{ .Host }}) [{{ .Region }}] {{ .Status }}",
        Selected: "✓ {{ .Name | green }} ({{ .Host }})",
        Details: `
--------- Server Details ----------
{{ "Name:" | faint }}     {{ .Name }}
{{ "Host:" | faint }}     {{ .Host }}
{{ "Region:" | faint }}   {{ .Region }}
{{ "Status:" | faint }}   {{ .Status }}`,
    }

    // 検索機能付き
    searcher := func(input string, index int) bool {
        s := servers[index]
        name := strings.Replace(strings.ToLower(s.Name), " ", "", -1)
        input = strings.Replace(strings.ToLower(input), " ", "", -1)
        return strings.Contains(name, input)
    }

    prompt := promptui.Select{
        Label:     "接続先サーバーを選択",
        Items:     servers,
        Templates: templates,
        Size:      10,
        Searcher:  searcher,
    }

    i, _, err := prompt.Run()
    if err != nil {
        return nil, err
    }

    return &servers[i], nil
}
```

### コード例14: パスワード入力

```go
func promptPassword() (string, error) {
    prompt := promptui.Prompt{
        Label: "パスワード",
        Mask:  '*',
        Validate: func(input string) error {
            if len(input) < 8 {
                return fmt.Errorf("パスワードは8文字以上必要です")
            }
            hasUpper := false
            hasDigit := false
            for _, c := range input {
                if c >= 'A' && c <= 'Z' {
                    hasUpper = true
                }
                if c >= '0' && c <= '9' {
                    hasDigit = true
                }
            }
            if !hasUpper {
                return fmt.Errorf("大文字を1文字以上含めてください")
            }
            if !hasDigit {
                return fmt.Errorf("数字を1文字以上含めてください")
            }
            return nil
        },
    }

    return prompt.Run()
}
```

---

## 7. シェル補完とドキュメント生成

### コード例15: シェル補完コマンド

```go
// cmd/completion.go
package cmd

import (
    "os"

    "github.com/spf13/cobra"
)

var completionCmd = &cobra.Command{
    Use:   "completion [bash|zsh|fish|powershell]",
    Short: "シェル補完スクリプトを生成する",
    Long: `指定されたシェル用の補完スクリプトを生成します。

Bash:
  $ source <(mytool completion bash)
  # 永続化するには:
  $ mytool completion bash > /etc/bash_completion.d/mytool

Zsh:
  $ source <(mytool completion zsh)
  # 永続化するには:
  $ mytool completion zsh > "${fpath[1]}/_mytool"

Fish:
  $ mytool completion fish | source
  # 永続化するには:
  $ mytool completion fish > ~/.config/fish/completions/mytool.fish

PowerShell:
  PS> mytool completion powershell | Out-String | Invoke-Expression
`,
    DisableFlagsInUseLine: true,
    ValidArgs:             []string{"bash", "zsh", "fish", "powershell"},
    Args:                  cobra.ExactValidArgs(1),
    RunE: func(cmd *cobra.Command, args []string) error {
        switch args[0] {
        case "bash":
            return cmd.Root().GenBashCompletionV2(os.Stdout, true)
        case "zsh":
            return cmd.Root().GenZshCompletion(os.Stdout)
        case "fish":
            return cmd.Root().GenFishCompletion(os.Stdout, true)
        case "powershell":
            return cmd.Root().GenPowerShellCompletionWithDesc(os.Stdout)
        default:
            return fmt.Errorf("unsupported shell: %s", args[0])
        }
    },
}

func init() {
    rootCmd.AddCommand(completionCmd)
}
```

### コード例16: Markdownドキュメント自動生成

```go
// cmd/docs.go
package cmd

import (
    "github.com/spf13/cobra"
    "github.com/spf13/cobra/doc"
)

var docsCmd = &cobra.Command{
    Use:    "docs",
    Short:  "ドキュメントを生成する",
    Hidden: true, // ユーザーに表示しない
    RunE: func(cmd *cobra.Command, args []string) error {
        outputDir, _ := cmd.Flags().GetString("dir")

        // Markdownドキュメント生成
        if err := doc.GenMarkdownTree(rootCmd, outputDir); err != nil {
            return fmt.Errorf("ドキュメント生成に失敗: %w", err)
        }
        fmt.Printf("ドキュメントを %s に生成しました\n", outputDir)
        return nil
    },
}

func init() {
    rootCmd.AddCommand(docsCmd)
    docsCmd.Flags().String("dir", "./docs", "出力ディレクトリ")
}
```

---

## 8. CLI 設計のベストプラクティス

### コード例17: エラーハンドリングと終了コード

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

        // エラーの種類に応じた終了コード
        var exitErr *ExitError
        if errors.As(err, &exitErr) {
            os.Exit(exitErr.Code)
        }
        os.Exit(1)
    }
}

// カスタム終了コード
type ExitError struct {
    Code    int
    Message string
}

func (e *ExitError) Error() string {
    return e.Message
}
```

### コード例18: テスト可能なCLI設計

```go
// main.go — 最小限のエントリポイント
package main

import (
    "os"

    "myapp/cmd"
)

func main() {
    cmd.Execute()
}

// cmd/root.go — テスト可能な構造
package cmd

import (
    "io"
    "os"
)

// App はCLIアプリケーションの依存関係をまとめる
type App struct {
    Stdout io.Writer
    Stderr io.Writer
    Stdin  io.Reader
    Env    func(string) string
}

func DefaultApp() *App {
    return &App{
        Stdout: os.Stdout,
        Stderr: os.Stderr,
        Stdin:  os.Stdin,
        Env:    os.Getenv,
    }
}

// テスト用
func TestApp(stdout, stderr io.Writer) *App {
    return &App{
        Stdout: stdout,
        Stderr: stderr,
        Stdin:  strings.NewReader(""),
        Env:    func(key string) string { return "" },
    }
}

// cmd/serve_test.go
func TestServeCommand(t *testing.T) {
    var stdout, stderr bytes.Buffer
    app := TestApp(&stdout, &stderr)

    cmd := newServeCmd(app)
    cmd.SetArgs([]string{"--port", "3000", "--host", "localhost"})

    err := cmd.Execute()
    require.NoError(t, err)
    assert.Contains(t, stdout.String(), "サーバー起動")
}
```

### コード例19: プログレスバーとスピナー

```go
package main

import (
    "fmt"
    "time"

    "github.com/schollz/progressbar/v3"
)

func downloadFiles(urls []string) error {
    bar := progressbar.NewOptions(len(urls),
        progressbar.OptionSetDescription("ダウンロード中"),
        progressbar.OptionSetTheme(progressbar.Theme{
            Saucer:        "=",
            SaucerHead:    ">",
            SaucerPadding: " ",
            BarStart:      "[",
            BarEnd:        "]",
        }),
        progressbar.OptionShowCount(),
        progressbar.OptionShowIts(),
        progressbar.OptionSetWidth(40),
    )

    for _, url := range urls {
        // ダウンロード処理
        err := download(url)
        if err != nil {
            return fmt.Errorf("download %s: %w", url, err)
        }
        bar.Add(1)
    }

    fmt.Println("\n完了!")
    return nil
}

// スピナーの実装
func withSpinner(message string, fn func() error) error {
    done := make(chan struct{})
    spinner := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

    go func() {
        i := 0
        for {
            select {
            case <-done:
                return
            default:
                fmt.Printf("\r%s %s", spinner[i%len(spinner)], message)
                i++
                time.Sleep(100 * time.Millisecond)
            }
        }
    }()

    err := fn()
    close(done)

    if err != nil {
        fmt.Printf("\r✗ %s: %v\n", message, err)
    } else {
        fmt.Printf("\r✓ %s\n", message)
    }
    return err
}
```

### コード例20: 出力フォーマットの切り替え

```go
package output

import (
    "encoding/json"
    "fmt"
    "io"
    "text/tabwriter"

    "gopkg.in/yaml.v3"
)

type Format string

const (
    FormatTable Format = "table"
    FormatJSON  Format = "json"
    FormatYAML  Format = "yaml"
    FormatWide  Format = "wide"
)

type Printer struct {
    Format Format
    Writer io.Writer
}

func (p *Printer) PrintUsers(users []User) error {
    switch p.Format {
    case FormatJSON:
        enc := json.NewEncoder(p.Writer)
        enc.SetIndent("", "  ")
        return enc.Encode(users)

    case FormatYAML:
        return yaml.NewEncoder(p.Writer).Encode(users)

    case FormatWide:
        w := tabwriter.NewWriter(p.Writer, 0, 0, 2, ' ', 0)
        fmt.Fprintln(w, "ID\tNAME\tEMAIL\tCREATED\tLAST_LOGIN\tSTATUS")
        for _, u := range users {
            fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n",
                u.ID, u.Name, u.Email, u.CreatedAt, u.LastLogin, u.Status)
        }
        return w.Flush()

    default: // table
        w := tabwriter.NewWriter(p.Writer, 0, 0, 2, ' ', 0)
        fmt.Fprintln(w, "ID\tNAME\tEMAIL")
        for _, u := range users {
            fmt.Fprintf(w, "%s\t%s\t%s\n", u.ID, u.Name, u.Email)
        }
        return w.Flush()
    }
}

// CLI での使用
var outputFormat string

var listCmd = &cobra.Command{
    Use:   "list",
    Short: "ユーザー一覧を表示する",
    RunE: func(cmd *cobra.Command, args []string) error {
        users, err := fetchUsers()
        if err != nil {
            return err
        }

        printer := &output.Printer{
            Format: output.Format(outputFormat),
            Writer: cmd.OutOrStdout(),
        }
        return printer.PrintUsers(users)
    },
}

func init() {
    listCmd.Flags().StringVarP(&outputFormat, "output", "o", "table",
        "出力フォーマット (table|json|yaml|wide)")
    listCmd.RegisterFlagCompletionFunc("output", func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
        return []string{"table", "json", "yaml", "wide"}, cobra.ShellCompDirectiveNoFileComp
    })
}
```

---

## 9. GoReleaser によるCLIバイナリ配布

### コード例21: GoReleaser 設定

```yaml
# .goreleaser.yaml
version: 2
project_name: mytool

before:
  hooks:
    - go mod tidy
    - go test ./...

builds:
  - main: ./main.go
    env:
      - CGO_ENABLED=0
    goos:
      - linux
      - darwin
      - windows
    goarch:
      - amd64
      - arm64
    ldflags:
      - -s -w
      - -X main.version={{.Version}}
      - -X main.commit={{.Commit}}
      - -X main.date={{.Date}}

archives:
  - format: tar.gz
    name_template: "{{ .ProjectName }}_{{ .Os }}_{{ .Arch }}"
    format_overrides:
      - goos: windows
        format: zip
    files:
      - README.md
      - LICENSE
      - completions/*

brews:
  - repository:
      owner: myorg
      name: homebrew-tap
    homepage: "https://github.com/myorg/mytool"
    description: "My awesome CLI tool"
    install: |
      bin.install "mytool"
      bash_completion.install "completions/mytool.bash" => "mytool"
      zsh_completion.install "completions/mytool.zsh" => "_mytool"

nfpms:
  - package_name: mytool
    homepage: "https://github.com/myorg/mytool"
    maintainer: "dev@example.com"
    description: "My awesome CLI tool"
    formats:
      - deb
      - rpm

checksum:
  name_template: 'checksums.txt'

changelog:
  sort: asc
  filters:
    exclude:
      - '^docs:'
      - '^test:'
```

---

## 10. アンチパターン

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

### アンチパターン3: ユーザーに不親切なエラーメッセージ

```go
// NG: 内部エラーをそのまま表示
func RunE(cmd *cobra.Command, args []string) error {
    return db.Query("SELECT ...")  // "pq: relation \"users\" does not exist"
}

// OK: ユーザーが理解できるメッセージ + 詳細は verbose で表示
func RunE(cmd *cobra.Command, args []string) error {
    _, err := db.Query("SELECT ...")
    if err != nil {
        if verbose {
            return fmt.Errorf("データベースクエリに失敗しました\n  詳細: %v\n  ヒント: マイグレーションを実行してください: mytool migrate up", err)
        }
        return fmt.Errorf("データベースクエリに失敗しました（-v で詳細を表示）")
    }
    return nil
}
```

### アンチパターン4: シグナルハンドリングの欠如

```go
// NG: Ctrl+C で即座に終了、リソースリーク
func main() {
    srv := startServer()
    select{} // 永遠にブロック
}

// OK: グレースフルシャットダウン
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // シグナルハンドリング
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

    srv := startServer(ctx)

    // シグナル待ち
    sig := <-sigCh
    fmt.Printf("\nシグナル受信: %v、シャットダウンします...\n", sig)

    // グレースフルシャットダウン
    shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer shutdownCancel()

    if err := srv.Shutdown(shutdownCtx); err != nil {
        fmt.Fprintf(os.Stderr, "シャットダウンエラー: %v\n", err)
        os.Exit(1)
    }
    fmt.Println("正常にシャットダウンしました")
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

### Q4. CLIツールのテストはどう書くべき？

3層に分けてテストする。1) ビジネスロジックのユニットテスト、2) コマンド実行のインテグレーションテスト（`cmd.SetArgs()` + `cmd.Execute()` を使う）、3) バイナリレベルのE2Eテスト（`os/exec` でバイナリを実行）。テスト可能にするには、`io.Writer` を注入し、`os.Stdout` に直接書き込まない設計にする。

### Q5. cobra のPreRun/PostRun はどう使い分ける？

`PersistentPreRun`: 全サブコマンドの前に実行（ロガー初期化、設定読み込みなど）。`PreRun`: 特定コマンドの前に実行（引数バリデーション、前提条件チェック）。`PostRun`: コマンド後に実行（クリーンアップ、ログ出力）。RunE のエラー有無に関わらず PersistentPostRun は実行される。

### Q6. CLIの出力を構造化するベストプラクティスは？

標準出力(stdout)にはプログラムの結果を、標準エラー出力(stderr)にはログ・プログレス・エラーメッセージを出力する。これにより `mytool list | jq .` のようなパイプ処理が正しく動作する。`--output json` フラグでJSON出力をサポートすると、スクリプトからの利用が容易になる。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 標準flag | シンプルなCLIに十分、`-flag` 形式 |
| FlagSet | 標準flagでサブコマンドを実装する方法 |
| pflag | POSIX互換 `--flag`、短縮形 `-f` 対応 |
| cobra | サブコマンド・ヘルプ・補完の業界標準 |
| viper | 設定ファイル・環境変数・フラグの統合管理 |
| promptui | 対話型選択メニュー・入力プロンプト |
| RunE | エラーを返すコマンド実行（Run より推奨） |
| SilenceUsage | エラー時のUsage表示抑制 |
| シェル補完 | bash/zsh/fish/powershell 対応の補完スクリプト |
| GoReleaser | クロスコンパイル・配布の自動化 |
| 出力フォーマット | table/json/yaml の切り替えサポート |
| シグナルハンドリング | グレースフルシャットダウンの実装 |

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
5. **Go公式 — flag パッケージ** https://pkg.go.dev/flag
6. **cobra ドキュメント — User Guide** https://cobra.dev/
