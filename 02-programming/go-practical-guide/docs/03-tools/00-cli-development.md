# Go CLI Development Guide

> Building production-grade command-line tools in Go with cobra, flag, and promptui

## What You Will Learn in This Chapter

1. Designing and implementing CLI applications with subcommands using the **cobra** framework
2. Differences between the standard **flag** package and **pflag**, and best practices for flag management
3. How to build interactive CLIs (selection menus, input prompts) with **promptui**
4. Configuration management with **viper** (integrating files, environment variables, and flags)
5. Implementing **shell completion** and **automatic documentation generation**


## Prerequisites

Before reading this guide, your understanding will be deeper if you have:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Overview of Go CLIs

### CLI Framework Selection Flow

```
Want to build a CLI tool
        |
        +-- Simple (a few flags)
        |       |
        |       v
        |   Standard flag package
        |
        +-- Has subcommands
        |       |
        |       v
        |   cobra (industry standard)
        |
        +-- Requires interactive UI
        |       |
        |       v
        |   promptui / survey
        |
        +-- Ultra-lightweight (no dependencies)
                |
                v
            Manual parsing / standard flag + manually implemented subcommands
```

### CLI Architecture

```
+------------------------------------------------------+
|                     main.go                          |
|  func main() { cmd.Execute() }                       |
+------------------------------------------------------+
        |
        v
+------------------------------------------------------+
|                   cmd/root.go                        |
|  rootCmd: app name, version, global flags            |
+------------------------------------------------------+
        |
        +-------> cmd/serve.go   (serve subcommand)
        |
        +-------> cmd/migrate.go (migrate subcommand)
        |
        +-------> cmd/config.go  (config subcommand)
        |              |
        |              +-> cmd/config_set.go (config set)
        |              +-> cmd/config_get.go (config get)
        |              +-> cmd/config_list.go (config list)
        |
        +-------> cmd/version.go (version subcommand)
        |
        +-------> cmd/completion.go (shell completion)
```

### Project Directory Structure

```
myapp/
├── main.go                    # Entry point (minimal)
├── cmd/                       # Command definitions
│   ├── root.go               # Root command
│   ├── serve.go              # serve subcommand
│   ├── migrate.go            # migrate subcommand
│   ├── config.go             # config subcommand group
│   ├── version.go            # version subcommand
│   └── completion.go         # Shell completion command
├── internal/                  # Internal packages
│   ├── config/               # Configuration management
│   ├── server/               # Server logic
│   └── migration/            # Migration logic
├── pkg/                       # Externally exposed packages (optional)
├── .goreleaser.yaml           # GoReleaser configuration
├── Makefile                   # Build and test commands
└── go.mod
```

---

## 2. Standard flag Package

### Code Example 1: Basics of the flag Package

```go
package main

import (
    "flag"
    "fmt"
    "os"
)

func main() {
    // Flag definitions
    host := flag.String("host", "localhost", "Server host name")
    port := flag.Int("port", 8080, "Server port number")
    verbose := flag.Bool("verbose", false, "Enable verbose output")

    // Custom Usage
    flag.Usage = func() {
        fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\nOptions:\n", os.Args[0])
        flag.PrintDefaults()
    }

    flag.Parse()

    // Remaining arguments (non-flags)
    args := flag.Args()

    if *verbose {
        fmt.Printf("Host: %s, Port: %d\n", *host, *port)
        fmt.Printf("Args: %v\n", args)
    }

    fmt.Printf("Starting server: %s:%d\n", *host, *port)
}
```

```bash
$ myapp -host 0.0.0.0 -port 3000 -verbose extra_arg
Host: 0.0.0.0, Port: 3000
Args: [extra_arg]
Starting server: 0.0.0.0:3000
```

### Code Example 2: Implementing Subcommands with FlagSet

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

    // Define a FlagSet for each subcommand
    serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
    servePort := serveCmd.Int("port", 8080, "Port number")
    serveHost := serveCmd.String("host", "localhost", "Host name")

    migrateCmd := flag.NewFlagSet("migrate", flag.ExitOnError)
    migrateDir := migrateCmd.String("dir", "./migrations", "Migration directory")
    migrateDSN := migrateCmd.String("dsn", "", "Database connection string")

    switch os.Args[1] {
    case "serve":
        serveCmd.Parse(os.Args[2:])
        fmt.Printf("Starting server: %s:%d\n", *serveHost, *servePort)

    case "migrate":
        migrateCmd.Parse(os.Args[2:])
        if *migrateDSN == "" {
            fmt.Fprintln(os.Stderr, "Error: -dsn flag is required")
            migrateCmd.Usage()
            os.Exit(1)
        }
        fmt.Printf("Running migration: dir=%s, dsn=%s\n", *migrateDir, *migrateDSN)

    case "version":
        fmt.Println("myapp v1.0.0")

    default:
        fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
        os.Exit(1)
    }
}
```

### Code Example 3: Custom Flag Types

```go
// StringSlice is a flag that accepts comma-separated values or multiple specifications
type StringSlice []string

func (s *StringSlice) String() string {
    return fmt.Sprintf("%v", *s)
}

func (s *StringSlice) Set(value string) error {
    *s = append(*s, value)
    return nil
}

// Custom flag for Duration type
type DurationFlag struct {
    value time.Duration
}

func (d *DurationFlag) String() string {
    return d.value.String()
}

func (d *DurationFlag) Set(s string) error {
    dur, err := time.ParseDuration(s)
    if err != nil {
        return fmt.Errorf("invalid Duration: %s", s)
    }
    d.value = dur
    return nil
}

func main() {
    var tags StringSlice
    flag.Var(&tags, "tag", "Tags (can be specified multiple times)")

    var timeout DurationFlag
    timeout.value = 30 * time.Second
    flag.Var(&timeout, "timeout", "Timeout (e.g., 30s, 5m)")

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

## 3. The cobra Framework

### Installation

```bash
go get github.com/spf13/cobra@latest
go install github.com/spf13/cobra-cli@latest
```

### Code Example 4: cobra Root Command

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
    Long:    `mytool is a multi-functional CLI tool built with Go.`,
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

    // Persistent Flags: usable across all subcommands
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "",
        "Config file path (default: $HOME/.mytool.yaml)")
    rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false,
        "Verbose output")

    // Integrate with viper
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

### Code Example 5: Adding Subcommands

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
    Short: "Start the HTTP server",
    Long:  `Starts the HTTP server on the specified host and port.`,
    Example: `  mytool serve
  mytool serve --port 3000
  mytool serve --host 0.0.0.0 --port 8080`,
    RunE: func(cmd *cobra.Command, args []string) error {
        addr := fmt.Sprintf("%s:%d", serveHost, servePort)
        fmt.Printf("Starting server: http://%s\n", addr)

        mux := http.NewServeMux()
        mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
            fmt.Fprintln(w, "Hello from mytool!")
        })
        return http.ListenAndServe(addr, mux)
    },
}

func init() {
    rootCmd.AddCommand(serveCmd)

    // Local Flags: specific to this command
    serveCmd.Flags().IntVarP(&servePort, "port", "p", 8080, "Port number")
    serveCmd.Flags().StringVar(&serveHost, "host", "localhost", "Host name")
}
```

### Code Example 6: Nested Subcommands

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
    Short: "Manage configuration",
}

var configSetCmd = &cobra.Command{
    Use:   "set [key] [value]",
    Short: "Change a config value",
    Args:  cobra.ExactArgs(2),
    RunE: func(cmd *cobra.Command, args []string) error {
        key, value := args[0], args[1]
        viper.Set(key, value)
        return viper.WriteConfig()
    },
}

var configGetCmd = &cobra.Command{
    Use:   "get [key]",
    Short: "Display a config value",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        val := viper.GetString(args[0])
        if val == "" {
            fmt.Printf("Key '%s' is not set\n", args[0])
            return
        }
        fmt.Println(val)
    },
}

var configListCmd = &cobra.Command{
    Use:   "list",
    Short: "List all configuration settings",
    Run: func(cmd *cobra.Command, args []string) {
        for key, val := range viper.AllSettings() {
            fmt.Printf("%s = %v\n", key, val)
        }
    },
}

var configInitCmd = &cobra.Command{
    Use:   "init",
    Short: "Initialize the config file",
    RunE: func(cmd *cobra.Command, args []string) error {
        // Set default values
        viper.SetDefault("server.host", "localhost")
        viper.SetDefault("server.port", 8080)
        viper.SetDefault("database.driver", "postgres")
        viper.SetDefault("log.level", "info")
        viper.SetDefault("log.format", "json")

        if err := viper.SafeWriteConfig(); err != nil {
            return fmt.Errorf("failed to create config file: %w", err)
        }
        fmt.Println("Config file created")
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

### Code Example 7: Argument Validation

```go
// List of cobra's argument validation functions
var exampleCmds = []*cobra.Command{
    // No arguments
    {
        Use:  "status",
        Args: cobra.NoArgs,
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // Exactly N
    {
        Use:  "rename [old] [new]",
        Args: cobra.ExactArgs(2),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // At least N
    {
        Use:  "add [file...]",
        Args: cobra.MinimumNArgs(1),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // At most N
    {
        Use:  "show [name]",
        Args: cobra.MaximumNArgs(1),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
    // Range
    {
        Use:  "between [args...]",
        Args: cobra.RangeArgs(1, 3),
        Run:  func(cmd *cobra.Command, args []string) {},
    },
}

// Custom validation
var deployCmd = &cobra.Command{
    Use:   "deploy [environment]",
    Short: "Deploy to the specified environment",
    Args: func(cmd *cobra.Command, args []string) error {
        if len(args) != 1 {
            return fmt.Errorf("please specify exactly one environment name")
        }
        validEnvs := map[string]bool{
            "development": true,
            "staging":     true,
            "production":  true,
        }
        if !validEnvs[args[0]] {
            return fmt.Errorf("invalid environment name: %s (specify one of development, staging, production)", args[0])
        }
        return nil
    },
    RunE: func(cmd *cobra.Command, args []string) error {
        env := args[0]
        fmt.Printf("Deploying to %s environment\n", env)
        return nil
    },
}

// ValidArgsFunction: dynamic completion candidates
var connectCmd = &cobra.Command{
    Use:   "connect [server]",
    Short: "Connect to a server",
    Args:  cobra.ExactArgs(1),
    ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
        if len(args) != 0 {
            return nil, cobra.ShellCompDirectiveNoFileComp
        }
        // Retrieve the server list dynamically
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

## 4. cobra Flag System

### Persistent Flags vs Local Flags Comparison

| Item | Persistent Flags | Local Flags |
|------|-----------------|-------------|
| Scope | Defining command + all child commands | Defining command only |
| Definition method | `PersistentFlags()` | `Flags()` |
| Use case | Common settings such as `--verbose`, `--config` | Command-specific options such as `--port`, `--output` |
| viper integration | Can be persisted via `BindPFlag` | Supported in the same way |
| Inheritance | Automatically inherited by child commands | Not inherited |

### flag vs pflag vs cobra Comparison

| Feature | Standard flag | pflag | cobra |
|---------|--------------|-------|-------|
| POSIX style `--flag` | `-flag` only | Supported | Supported (pflag built-in) |
| Short form `-v` | Not supported | Supported | Supported |
| Subcommands | Not supported | Not supported | Supported |
| Automatic help | Basic | Basic | Rich |
| Shell completion | Not supported | Not supported | bash/zsh/fish/powershell |
| Argument validation | Manual | Manual | Declaratively via `Args` |
| Config file integration | Manual | Manual | Integrated with viper |

### Code Example 8: Advanced Flag Usage

```go
// cmd/serve.go
func init() {
    rootCmd.AddCommand(serveCmd)

    // Basic flag definitions
    serveCmd.Flags().IntVarP(&port, "port", "p", 8080, "Port number")
    serveCmd.Flags().StringVar(&host, "host", "localhost", "Host name")

    // Required flag
    serveCmd.Flags().StringVar(&certFile, "cert", "", "TLS certificate file")
    serveCmd.MarkFlagRequired("cert")

    // Enable file path completion
    serveCmd.MarkFlagFilename("cert", "pem", "crt")

    // Mutually exclusive flags
    serveCmd.Flags().BoolVar(&useTLS, "tls", false, "Enable TLS")
    serveCmd.Flags().BoolVar(&useHTTP2, "h2c", false, "Use HTTP/2 Cleartext")
    serveCmd.MarkFlagsMutuallyExclusive("tls", "h2c")

    // Group (if one is specified, both are required)
    serveCmd.Flags().StringVar(&certFile, "cert-file", "", "Certificate file")
    serveCmd.Flags().StringVar(&keyFile, "key-file", "", "Private key file")
    serveCmd.MarkFlagsRequiredTogether("cert-file", "key-file")

    // Integration with environment variables
    viper.BindPFlag("server.port", serveCmd.Flags().Lookup("port"))
    viper.BindPFlag("server.host", serveCmd.Flags().Lookup("host"))

    // Take default values from environment variables
    viper.BindEnv("server.port", "MYAPP_PORT")
    viper.BindEnv("server.host", "MYAPP_HOST")
}
```

### Code Example 9: Custom Flag Validation

```go
// Port number validation
var serveCmd = &cobra.Command{
    Use: "serve",
    PreRunE: func(cmd *cobra.Command, args []string) error {
        port, _ := cmd.Flags().GetInt("port")
        if port < 1 || port > 65535 {
            return fmt.Errorf("port number must be in the range 1-65535: %d", port)
        }
        if port < 1024 {
            fmt.Fprintf(os.Stderr, "Warning: port %d is a privileged port (root privileges required)\n", port)
        }
        return nil
    },
    RunE: func(cmd *cobra.Command, args []string) error {
        // Main logic
        return nil
    },
}

// Enum-style flag
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
        return fmt.Errorf("invalid log level: %s (must be one of debug, info, warn, error)", v)
    }
}
func (l *LogLevel) Type() string { return "LogLevel" }

var logLevel LogLevel = LogInfo

func init() {
    rootCmd.PersistentFlags().Var(&logLevel, "log-level", "Log level (debug|info|warn|error)")
    rootCmd.RegisterFlagCompletionFunc("log-level", func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
        return []string{"debug", "info", "warn", "error"}, cobra.ShellCompDirectiveNoFileComp
    })
}
```

---

## 5. Configuration Management with viper

### Configuration Priority

```
+----------------------------------------------------------+
|  viper configuration priority (higher is stronger)       |
+----------------------------------------------------------+
|                                                          |
|  1. Explicit settings via viper.Set()                    |
|     ↓                                                    |
|  2. Command-line flags (--port 3000)                     |
|     ↓                                                    |
|  3. Environment variables (MYAPP_PORT=3000)              |
|     ↓                                                    |
|  4. Config file (.mytool.yaml)                           |
|     ↓                                                    |
|  5. Key/value store (etcd, Consul)                       |
|     ↓                                                    |
|  6. Default values via viper.SetDefault()                |
+----------------------------------------------------------+
```

### Code Example 10: Comprehensive Configuration Management with viper

```go
package config

import (
    "fmt"
    "strings"
    "time"

    "github.com/spf13/viper"
)

// Config is the application's configuration struct
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
    // Default values
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

    // Bind environment variables
    viper.SetEnvPrefix("MYAPP")
    viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
    viper.AutomaticEnv()

    // Read the config file
    if err := viper.ReadInConfig(); err != nil {
        if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
            return nil, fmt.Errorf("failed to read config file: %w", err)
        }
        // Continue with default values if the config file is not found
    }

    var cfg Config
    if err := viper.Unmarshal(&cfg); err != nil {
        return nil, fmt.Errorf("failed to decode configuration: %w", err)
    }

    return &cfg, nil
}
```

### Example Config File

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
  password: "${DB_PASSWORD}"  # Recommended to override via environment variable
  ssl_mode: "require"

log:
  level: "info"
  format: "json"
  output: "stdout"

auth:
  jwt_secret: ""  # Set via environment variable MYAPP_AUTH_JWT_SECRET
  token_expiry: "24h"
  refresh_token: true
```

### Code Example 11: Hot Reloading the Config File

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
        log.Printf("Config file changed: %s", e.Name)

        w.mu.Lock()
        defer w.mu.Unlock()

        var newCfg Config
        if err := viper.Unmarshal(&newCfg); err != nil {
            log.Printf("Failed to reload configuration: %v", err)
            return
        }

        w.config = &newCfg

        // Invoke callbacks
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

## 6. Interactive CLIs with promptui

### Code Example 12: Selection Menus and Input Prompts

```go
package main

import (
    "fmt"
    "strings"

    "github.com/manifoldco/promptui"
)

func main() {
    // Selection prompt
    envSelect := promptui.Select{
        Label: "Select deployment environment",
        Items: []string{"development", "staging", "production"},
        Templates: &promptui.SelectTemplates{
            Active:   "▸ {{ . | cyan }}",
            Inactive: "  {{ . }}",
            Selected: "✓ {{ . | green }}",
        },
    }
    _, env, err := envSelect.Run()
    if err != nil {
        fmt.Printf("Selection canceled: %v\n", err)
        return
    }

    // Confirmation prompt for production
    if env == "production" {
        confirm := promptui.Prompt{
            Label:     "Confirm deployment to production (yes/no)",
            IsConfirm: true,
        }
        _, err := confirm.Run()
        if err != nil {
            fmt.Println("Deployment canceled")
            return
        }
    }

    // Input prompt (with validation)
    tagPrompt := promptui.Prompt{
        Label: "Release tag (e.g., v1.2.3)",
        Validate: func(input string) error {
            if !strings.HasPrefix(input, "v") {
                return fmt.Errorf("tag must start with 'v'")
            }
            return nil
        },
    }
    tag, err := tagPrompt.Run()
    if err != nil {
        return
    }

    fmt.Printf("Running deployment: env=%s, tag=%s\n", env, tag)
}
```

### Interactive Flow

```
$ mytool deploy

? Select deployment environment:
    development
  ▸ staging
    production

✓ staging

? Release tag (e.g., v1.2.3): v1.5.0

Running deployment: env=staging, tag=v1.5.0
```

### Code Example 13: Rich Selection Menu Using Structs

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

    // With search feature
    searcher := func(input string, index int) bool {
        s := servers[index]
        name := strings.Replace(strings.ToLower(s.Name), " ", "", -1)
        input = strings.Replace(strings.ToLower(input), " ", "", -1)
        return strings.Contains(name, input)
    }

    prompt := promptui.Select{
        Label:     "Select server to connect to",
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

### Code Example 14: Password Input

```go
func promptPassword() (string, error) {
    prompt := promptui.Prompt{
        Label: "Password",
        Mask:  '*',
        Validate: func(input string) error {
            if len(input) < 8 {
                return fmt.Errorf("password must be at least 8 characters")
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
                return fmt.Errorf("must include at least one uppercase letter")
            }
            if !hasDigit {
                return fmt.Errorf("must include at least one digit")
            }
            return nil
        },
    }

    return prompt.Run()
}
```

---

## 7. Shell Completion and Documentation Generation

### Code Example 15: Shell Completion Command

```go
// cmd/completion.go
package cmd

import (
    "os"

    "github.com/spf13/cobra"
)

var completionCmd = &cobra.Command{
    Use:   "completion [bash|zsh|fish|powershell]",
    Short: "Generate shell completion scripts",
    Long: `Generates a completion script for the specified shell.

Bash:
  $ source <(mytool completion bash)
  # To persist:
  $ mytool completion bash > /etc/bash_completion.d/mytool

Zsh:
  $ source <(mytool completion zsh)
  # To persist:
  $ mytool completion zsh > "${fpath[1]}/_mytool"

Fish:
  $ mytool completion fish | source
  # To persist:
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

### Code Example 16: Automatic Markdown Documentation Generation

```go
// cmd/docs.go
package cmd

import (
    "github.com/spf13/cobra"
    "github.com/spf13/cobra/doc"
)

var docsCmd = &cobra.Command{
    Use:    "docs",
    Short:  "Generate documentation",
    Hidden: true, // Hidden from users
    RunE: func(cmd *cobra.Command, args []string) error {
        outputDir, _ := cmd.Flags().GetString("dir")

        // Generate Markdown documentation
        if err := doc.GenMarkdownTree(rootCmd, outputDir); err != nil {
            return fmt.Errorf("documentation generation failed: %w", err)
        }
        fmt.Printf("Documentation generated in %s\n", outputDir)
        return nil
    },
}

func init() {
    rootCmd.AddCommand(docsCmd)
    docsCmd.Flags().String("dir", "./docs", "Output directory")
}
```

---

## 8. CLI Design Best Practices

### Code Example 17: Error Handling and Exit Codes

```go
var rootCmd = &cobra.Command{
    // Use RunE and return errors
    RunE: func(cmd *cobra.Command, args []string) error {
        if err := doSomething(); err != nil {
            // Wrap user-facing messages before returning
            return fmt.Errorf("operation failed: %w", err)
        }
        return nil
    },
    // SilenceUsage: don't print Usage on error
    SilenceUsage: true,
    // SilenceErrors: suppress cobra's default error display
    SilenceErrors: true,
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        // Print error to stderr
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)

        // Exit code based on error type
        var exitErr *ExitError
        if errors.As(err, &exitErr) {
            os.Exit(exitErr.Code)
        }
        os.Exit(1)
    }
}

// Custom exit code
type ExitError struct {
    Code    int
    Message string
}

func (e *ExitError) Error() string {
    return e.Message
}
```

### Code Example 18: Testable CLI Design

```go
// main.go — minimal entry point
package main

import (
    "os"

    "myapp/cmd"
)

func main() {
    cmd.Execute()
}

// cmd/root.go — testable structure
package cmd

import (
    "io"
    "os"
)

// App groups the CLI application's dependencies
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

// For testing
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
    assert.Contains(t, stdout.String(), "Starting server")
}
```

### Code Example 19: Progress Bars and Spinners

```go
package main

import (
    "fmt"
    "time"

    "github.com/schollz/progressbar/v3"
)

func downloadFiles(urls []string) error {
    bar := progressbar.NewOptions(len(urls),
        progressbar.OptionSetDescription("Downloading"),
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
        // Download operation
        err := download(url)
        if err != nil {
            return fmt.Errorf("download %s: %w", url, err)
        }
        bar.Add(1)
    }

    fmt.Println("\nDone!")
    return nil
}

// Spinner implementation
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

### Code Example 20: Switching Output Formats

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

// Usage in CLI
var outputFormat string

var listCmd = &cobra.Command{
    Use:   "list",
    Short: "Display the user list",
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
        "Output format (table|json|yaml|wide)")
    listCmd.RegisterFlagCompletionFunc("output", func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
        return []string{"table", "json", "yaml", "wide"}, cobra.ShellCompDirectiveNoFileComp
    })
}
```

---

## 9. Distributing CLI Binaries with GoReleaser

### Code Example 21: GoReleaser Configuration

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

## 10. Anti-patterns

### Anti-pattern 1: Writing Logic Directly in main()

```go
// BAD: not testable, not reusable
func main() {
    flag.Parse()
    db, _ := sql.Open("postgres", *dsn)
    rows, _ := db.Query("SELECT ...")
    for rows.Next() {
        // All processing concentrated in main
    }
}

// GOOD: separate logic; main is only an entry point
func main() {
    if err := run(os.Args[1:], os.Stdout); err != nil {
        fmt.Fprintf(os.Stderr, "error: %v\n", err)
        os.Exit(1)
    }
}

func run(args []string, stdout io.Writer) error {
    // Testable logic
    cfg, err := parseFlags(args)
    if err != nil {
        return err
    }
    return execute(cfg, stdout)
}
```

### Anti-pattern 2: Abuse of Global Variables

```go
// BAD: everything managed globally
var (
    db      *sql.DB
    logger  *log.Logger
    config  Config
    client  *http.Client
)

// GOOD: group them in a struct and inject dependencies
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

### Anti-pattern 3: Unfriendly Error Messages for Users

```go
// BAD: display internal errors as-is
func RunE(cmd *cobra.Command, args []string) error {
    return db.Query("SELECT ...")  // "pq: relation \"users\" does not exist"
}

// GOOD: user-understandable message + details shown with verbose
func RunE(cmd *cobra.Command, args []string) error {
    _, err := db.Query("SELECT ...")
    if err != nil {
        if verbose {
            return fmt.Errorf("database query failed\n  details: %v\n  hint: run migrations with: mytool migrate up", err)
        }
        return fmt.Errorf("database query failed (use -v for details)")
    }
    return nil
}
```

### Anti-pattern 4: Missing Signal Handling

```go
// BAD: terminates immediately on Ctrl+C, resource leak
func main() {
    srv := startServer()
    select{} // Blocks forever
}

// GOOD: graceful shutdown
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Signal handling
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

    srv := startServer(ctx)

    // Wait for signal
    sig := <-sigCh
    fmt.Printf("\nReceived signal: %v, shutting down...\n", sig)

    // Graceful shutdown
    shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer shutdownCancel()

    if err := srv.Shutdown(shutdownCtx); err != nil {
        fmt.Fprintf(os.Stderr, "Shutdown error: %v\n", err)
        os.Exit(1)
    }
    fmt.Println("Shut down gracefully")
}
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: template for basic implementation
class Exercise1:
    """Exercise in basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Test
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should be raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise in advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Test
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---

## FAQ

### Q1. Should I choose cobra or urfave/cli?

cobra is adopted by large projects such as Docker, Kubernetes, Hugo, and GitHub CLI, and has a rich ecosystem. urfave/cli is simpler, but cobra is superior for features such as shell completion and viper integration. We recommend cobra for new projects.

### Q2. How should CLI tool binaries be distributed?

Using GoReleaser, you can trigger cross-compilation, GitHub Releases, Homebrew Tap, and Docker Image generation automatically from a `git tag`. Configuring `.goreleaser.yaml` and integrating it with GitHub Actions is the standard approach.

### Q3. What is the priority of config files, environment variables, and flags?

The standard viper priority is: 1) an explicit `Set()` call → 2) flags → 3) environment variables → 4) config file → 5) default values. This ordering allows users to use the config file as a baseline while overriding with environment variables or flags.

### Q4. How should I write tests for CLI tools?

Test in three layers. 1) Unit tests for business logic; 2) integration tests for command execution (using `cmd.SetArgs()` + `cmd.Execute()`); 3) E2E tests at the binary level (running the binary via `os/exec`). To make things testable, design so that you inject an `io.Writer` and don't write directly to `os.Stdout`.

### Q5. How do I use cobra's PreRun/PostRun appropriately?

`PersistentPreRun`: runs before every subcommand (logger initialization, loading configuration, etc.). `PreRun`: runs before a specific command (argument validation, prerequisite checks). `PostRun`: runs after a command (cleanup, logging). `PersistentPostRun` is executed regardless of whether RunE returned an error.

### Q6. What are best practices for structuring CLI output?

Write program results to standard output (stdout) and logs, progress, and error messages to standard error (stderr). This makes pipe processing such as `mytool list | jq .` work correctly. Supporting JSON output via an `--output json` flag makes consumption from scripts easier.

---

## Summary

| Concept | Key points |
|---------|-----------|
| Standard flag | Sufficient for simple CLIs, `-flag` style |
| FlagSet | How to implement subcommands with the standard flag |
| pflag | POSIX-compatible `--flag`, supports short form `-f` |
| cobra | Industry standard for subcommands, help, and completion |
| viper | Integrated management of config files, environment variables, and flags |
| promptui | Interactive selection menus and input prompts |
| RunE | Command execution returning an error (preferred over Run) |
| SilenceUsage | Suppress Usage display on errors |
| Shell completion | Completion scripts supporting bash/zsh/fish/powershell |
| GoReleaser | Automate cross-compilation and distribution |
| Output format | Support switching between table/json/yaml |
| Signal handling | Implement graceful shutdown |

---

## Recommended Next Guides

- **03-tools/01-generics.md** — Generics: type parameters, constraints
- **03-tools/03-deployment.md** — Deployment: Docker, cross-compilation
- **02-web/04-testing.md** — Testing: table-driven tests, testify, httptest

---

## References

1. **spf13/cobra GitHub** https://github.com/spf13/cobra
2. **spf13/viper GitHub** https://github.com/spf13/viper
3. **manifoldco/promptui GitHub** https://github.com/manifoldco/promptui
4. **GoReleaser Official Documentation** https://goreleaser.com/
5. **Go Official — flag package** https://pkg.go.dev/flag
6. **cobra Documentation — User Guide** https://cobra.dev/
