# 高度なシェルスクリプト

> 実務で使えるスクリプトには、エラー処理・ログ・並列処理が不可欠。

## この章で学ぶこと

- [ ] 堅牢なエラーハンドリングができる
- [ ] 配列・連想配列を活用できる
- [ ] 並列処理・テンプレート処理ができる

---

## 1. 堅牢なスクリプトの書き方

```bash
#!/usr/bin/env bash

# 防御的設定（スクリプト冒頭に必ず入れる）
set -euo pipefail

# set -e: コマンドが失敗したら即終了
# set -u: 未定義変数の参照でエラー
# set -o pipefail: パイプ中の失敗を検知

# デバッグ時
set -x                           # 実行コマンドを表示
# set -euxo pipefail            # 全部入り

# エラーハンドリング
trap 'echo "Error at line $LINENO (exit $?)" >&2' ERR

# クリーンアップ
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# エラー関数
die() {
    echo "ERROR: $*" >&2
    exit 1
}

# 使用例
[[ -f "$1" ]] || die "File not found: $1"
```

### ログ出力

```bash
# カラー付きログ関数
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly NC='\033[0m'  # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

log_info "Processing started"
log_warn "File size exceeds 100MB"
log_error "Connection failed"

# ファイルとstdoutの両方にログ
exec > >(tee -a "$LOGFILE") 2>&1
```

---

## 2. 引数処理

```bash
# 位置パラメータ
echo "Script: $0"
echo "First arg: $1"
echo "All args: $@"
echo "Arg count: $#"

# shift で引数をずらす
while [[ $# -gt 0 ]]; do
    echo "Processing: $1"
    shift
done

# getopts（短いオプション）
while getopts "hv:o:" opt; do
    case $opt in
        h) usage; exit 0 ;;
        v) verbose="$OPTARG" ;;
        o) output="$OPTARG" ;;
        ?) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))  # オプション以外の引数にアクセス

# 長いオプション（手動パース）
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage; exit 0
            ;;
        --output|-o)
            output="$2"; shift 2
            ;;
        --verbose|-v)
            verbose=true; shift
            ;;
        --)
            shift; break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            break
            ;;
    esac
done

# usage 関数のテンプレート
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] FILE...

Options:
  -h, --help        Show this help
  -o, --output DIR  Output directory
  -v, --verbose     Verbose output

Examples:
  $(basename "$0") -o /tmp input.txt
  $(basename "$0") --verbose *.csv
EOF
}
```

---

## 3. 配列

```bash
# 通常の配列（インデックス配列）
fruits=("apple" "banana" "cherry")

# アクセス
echo "${fruits[0]}"              # apple
echo "${fruits[1]}"              # banana
echo "${fruits[@]}"              # 全要素
echo "${#fruits[@]}"             # 要素数（3）

# 追加
fruits+=("date")
fruits+=("elderberry" "fig")

# ループ
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

# インデックス付きループ
for i in "${!fruits[@]}"; do
    echo "$i: ${fruits[$i]}"
done

# スライス
echo "${fruits[@]:1:3}"          # index 1から3個

# 削除
unset 'fruits[2]'               # index 2 を削除

# 配列をコマンドの出力から作成
files=($(ls *.txt))
lines=($(cat file.txt))

# 安全な方法（スペース対応）
mapfile -t lines < file.txt
readarray -t lines < file.txt   # 同じ
```

### 連想配列（Bash 4+）

```bash
declare -A colors
colors[red]="#FF0000"
colors[green]="#00FF00"
colors[blue]="#0000FF"

# 一括定義
declare -A config=(
    [host]="localhost"
    [port]="8080"
    [debug]="true"
)

# アクセス
echo "${config[host]}"           # localhost

# キー一覧
echo "${!config[@]}"             # host port debug

# ループ
for key in "${!config[@]}"; do
    echo "$key = ${config[$key]}"
done

# 存在チェック
if [[ -v config[host] ]]; then
    echo "host is set"
fi
```

---

## 4. 並列処理

```bash
# バックグラウンドジョブ + wait
process_file() {
    local file="$1"
    echo "Processing: $file"
    sleep 2  # 模擬処理
    echo "Done: $file"
}

for file in *.csv; do
    process_file "$file" &
done
wait
echo "All files processed"

# 並列数の制限
MAX_JOBS=4
for file in *.csv; do
    process_file "$file" &
    # 同時実行数を制限
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 0.1
    done
done
wait

# xargs による並列処理
find . -name "*.jpg" -print0 | xargs -0 -P 4 -I {} convert {} {}.png
# -P 4: 4並列
# -0:   NUL区切り（ファイル名のスペース対応）
# -I {}: プレースホルダー

# GNU parallel（より高機能）
# brew install parallel
parallel convert {} {.}.png ::: *.jpg
parallel -j 4 process_file ::: *.csv
cat urls.txt | parallel -j 8 curl -sO {}
```

---

## 5. テキスト処理のパターン

```bash
# CSV処理
while IFS=',' read -r name age city; do
    echo "Name: $name, Age: $age, City: $city"
done < data.csv

# INI ファイル風の設定読み込み
parse_config() {
    local file="$1"
    while IFS='=' read -r key value; do
        # コメントと空行をスキップ
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # 前後の空白を削除
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        export "CONFIG_$key=$value"
    done < "$file"
}

# テンプレート処理（envsubst）
export APP_NAME="MyApp"
export APP_PORT="8080"
envsubst < template.conf > output.conf

# テンプレート処理（sed）
sed -e "s/{{APP_NAME}}/$APP_NAME/g" \
    -e "s/{{APP_PORT}}/$APP_PORT/g" \
    template.conf > output.conf
```

---

## 6. 実践的なスクリプト例

```bash
#!/usr/bin/env bash
set -euo pipefail

# デプロイスクリプトの例
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly APP_NAME="myapp"
readonly DEPLOY_DIR="/var/www/$APP_NAME"
readonly BACKUP_DIR="/var/backups/$APP_NAME"
readonly LOG_FILE="/var/log/$APP_NAME/deploy.log"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ログ設定
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# クリーンアップ
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "Deploy FAILED (exit: $exit_code). Rolling back..."
        rollback
    fi
}
trap cleanup EXIT

# ロールバック
rollback() {
    local latest_backup="$BACKUP_DIR/$(ls -t "$BACKUP_DIR" | head -1)"
    if [[ -d "$latest_backup" ]]; then
        log "Restoring from: $latest_backup"
        rsync -a --delete "$latest_backup/" "$DEPLOY_DIR/"
        log "Rollback completed"
    fi
}

# メイン処理
main() {
    log "=== Deploy started ==="

    # バックアップ
    log "Creating backup..."
    mkdir -p "$BACKUP_DIR/$TIMESTAMP"
    cp -a "$DEPLOY_DIR/." "$BACKUP_DIR/$TIMESTAMP/"

    # デプロイ
    log "Deploying..."
    rsync -avz --delete "$SCRIPT_DIR/dist/" "$DEPLOY_DIR/"

    # ヘルスチェック
    log "Health check..."
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/health")
    [[ "$status" -eq 200 ]] || die "Health check failed (status: $status)"

    log "=== Deploy completed ==="
}

main "$@"
```

---

## まとめ

| テクニック | 用途 |
|-----------|------|
| set -euo pipefail | 堅牢なエラー処理 |
| trap '...' EXIT | クリーンアップ保証 |
| getopts / while case | 引数パース |
| declare -A | 連想配列 |
| xargs -P N | 並列処理 |
| mapfile -t | ファイル→配列変換 |
| envsubst | テンプレート展開 |

---

## 次に読むべきガイド
→ [[../06-system-admin/00-systemd.md]] — システム管理

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.24-36, 2019.
2. "Google Shell Style Guide." google.github.io/styleguide.
