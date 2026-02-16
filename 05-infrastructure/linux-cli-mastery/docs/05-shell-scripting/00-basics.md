# シェルスクリプト基礎

> シェルスクリプトは CLI の操作を自動化する最も直接的な方法。日常の繰り返し作業から本格的なシステム管理まで、あらゆるレベルで活用できる。

## この章で学ぶこと

- [ ] シェルスクリプトの基本構文を理解する
- [ ] 変数・条件分岐・ループを使える
- [ ] 関数とスクリプトの引数処理ができる
- [ ] 入出力・リダイレクト・ヒアドキュメントを使いこなす
- [ ] デバッグ手法を身につける
- [ ] 実務で使えるスクリプトテンプレートを持つ

---

## 1. スクリプトの基本

### 1.1 シバン（Shebang）

スクリプトの1行目に記述する `#!` で始まる行をシバン（shebang）と呼ぶ。OS がこの行を読み取り、指定されたインタプリタでスクリプトを実行する。

```bash
#!/bin/bash
# ↑ シバン（shebang）: このスクリプトを実行するインタプリタを指定

# スクリプトの実行方法
chmod +x script.sh               # 実行権限を付与
./script.sh                      # 直接実行
bash script.sh                   # bash で明示的に実行
source script.sh                 # 現在のシェルで実行（. script.sh と同じ）

# シバンの種類
#!/bin/bash                      # bash
#!/bin/zsh                       # zsh
#!/usr/bin/env bash              # PATH から bash を探す（推奨）
#!/usr/bin/env python3           # Python スクリプト
#!/usr/bin/env perl               # Perl スクリプト
#!/bin/sh                        # POSIX sh（移植性最優先の場合）
```

### 1.2 `#!/bin/bash` と `#!/usr/bin/env bash` の違い

```bash
# #!/bin/bash
#   → /bin/bash を直接指定
#   → bash が /bin/ にない環境（一部の BSD、NixOS 等）では動かない
#   → パスが確定しているため起動が微妙に速い

# #!/usr/bin/env bash
#   → $PATH を検索して bash を見つける
#   → 異なる環境間での移植性が高い（推奨）
#   → pyenv, rbenv 等のバージョン管理ツールとも相性が良い

# 確認方法
which bash                       # bash のパスを表示
type bash                        # bash の種類を表示
bash --version                   # バージョン確認
```

### 1.3 スクリプトの実行方法の違い

```bash
# 方法1: 直接実行（サブシェルで実行）
chmod +x script.sh
./script.sh
# → 新しいプロセスが起動する
# → スクリプト内の cd, export は呼び出し元に影響しない

# 方法2: bash コマンドで実行（サブシェル）
bash script.sh
# → 実行権限不要
# → シバン行は無視される

# 方法3: source で実行（現在のシェルで実行）
source script.sh
# または
. script.sh
# → 現在のシェルで直接実行
# → スクリプト内の cd, export, alias が現在のシェルに影響する
# → .bashrc, .zshrc の読み込みに使われる

# 実践例: 環境変数を設定するスクリプト
# --- env.sh ---
#!/bin/bash
export APP_ENV="production"
export APP_PORT="8080"
# --- ここまで ---

# NG: サブシェルで実行しても現在のシェルに反映されない
./env.sh
echo $APP_ENV    # → （空）

# OK: source で実行すれば反映される
source env.sh
echo $APP_ENV    # → production
```

### 1.4 スクリプトファイルの慣例

```bash
# ファイル名の慣例
# - 拡張子 .sh を付ける（必須ではないが推奨）
# - 実行可能スクリプトは拡張子なしも一般的（/usr/local/bin/mycommand）
# - ハイフン区切り: deploy-app.sh, run-tests.sh
# - アンダースコアも可: deploy_app.sh

# ファイルの構成（推奨テンプレート）
#!/usr/bin/env bash
#
# スクリプト名: deploy.sh
# 説明: アプリケーションのデプロイを実行する
# 作成者: Gaku
# 作成日: 2025-01-01
# 使い方: ./deploy.sh [--env production|staging] [--branch main]
#

set -euo pipefail

# 定数
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# メイン処理
main() {
    echo "Starting $SCRIPT_NAME..."
    # ここに処理を書く
}

main "$@"
```

### 1.5 コメントの書き方

```bash
# 行コメント（# から行末まで）
echo "Hello"  # インラインコメント

# 複数行コメント（ヒアドキュメントを利用したテクニック）
: <<'COMMENT'
この部分は実行されない
複数行のコメントとして使える
ただしインデントの制約がある
COMMENT

# ドキュメンテーションコメントの慣例
#######################################
# データベースのバックアップを実行する
# Globals:
#   DB_HOST
#   DB_NAME
# Arguments:
#   $1 - 出力ディレクトリ
# Returns:
#   0 - 成功
#   1 - 失敗
#######################################
backup_database() {
    local output_dir="$1"
    # ...
}
```

---

## 2. 変数

### 2.1 変数の基本

```bash
# 変数の代入（= の前後にスペースを入れない）
name="Gaku"                      # 文字列
count=42                         # 数値
path="/var/log"                  # パス
empty=""                         # 空文字列

# よくあるミス
# name = "Gaku"                  # NG: スペースがあるとコマンドと解釈される
# name ="Gaku"                   # NG: 同上
# name= "Gaku"                   # NG: 同上

# 変数の参照
echo "$name"                     # Gaku
echo "${name}_suffix"            # Gaku_suffix（区切りが必要な場合）
echo "Hello, $name!"             # Hello, Gaku!
echo "${name}san"                # Gakusan（直後に文字が続く場合は {} が必要）

# クォートの違い
echo "Hello, $name"              # → Hello, Gaku（変数展開される）
echo 'Hello, $name'              # → Hello, $name（リテラル）
echo Hello, $name                # → Hello, Gaku（クォートなし: ワード分割が起こる）

# クォートの重要性
file="my file.txt"
# cat $file                      # NG: "my" と "file.txt" の2引数として解釈
cat "$file"                      # OK: "my file.txt" として1引数
# ルール: 変数は常にダブルクォートで囲む

# コマンド置換
today=$(date +%Y-%m-%d)          # 推奨: $() 形式
files=`ls`                       # 旧形式: バッククォート（非推奨）
# $() はネストできる
backup_name="backup_$(date +%Y%m%d)_$(hostname).tar.gz"

# ネストの例
inner_result=$(echo "The date is $(date +%Y-%m-%d)")
# バッククォートではネストが困難: `echo "The date is \`date +%Y-%m-%d\`"`
```

### 2.2 算術演算

```bash
# 算術演算
result=$((3 + 5))                # → 8
count=$((count + 1))             # インクリメント
echo $((10 / 3))                 # → 3（整数除算）
echo $((10 % 3))                 # → 1（剰余）
echo $((2 ** 10))                # → 1024（べき乗）

# 複合代入演算子
(( count += 5 ))                 # count = count + 5
(( count -= 3 ))                 # count = count - 3
(( count *= 2 ))                 # count = count * 2
(( count /= 4 ))                 # count = count / 4
(( count ++ ))                   # count = count + 1
(( count -- ))                   # count = count - 1

# 複雑な計算
width=640
height=480
area=$(( width * height ))
echo "Area: $area pixels"       # → Area: 307200 pixels

# 16進数・8進数
echo $(( 0xFF ))                 # → 255（16進数）
echo $(( 077 ))                  # → 63（8進数）
echo $(( 2#1010 ))               # → 10（2進数）

# 小数点計算（bc を使用）
echo "scale=2; 10 / 3" | bc      # → 3.33
pi=$(echo "scale=10; 4*a(1)" | bc -l)  # → 3.1415926535
result=$(echo "1.5 + 2.3" | bc)  # → 3.8

# awk での計算（より高機能）
awk 'BEGIN { printf "%.2f\n", 10/3 }'  # → 3.33
awk 'BEGIN { printf "%.4f\n", sqrt(2) }'  # → 1.4142
```

### 2.3 特殊変数

```bash
# スクリプト引数関連
echo $0                          # スクリプト名
echo $1                          # 第1引数
echo $2                          # 第2引数
echo ${10}                       # 第10引数（2桁以上は {} が必要）
echo $#                          # 引数の数
echo $@                          # 全引数（個別に展開）
echo $*                          # 全引数（1つの文字列）

# $@ と $* の違い（重要）
# スクリプトを ./test.sh "hello world" foo bar で実行した場合:
# "$@" → "hello world" "foo" "bar"（3つの引数として展開）
# "$*" → "hello world foo bar"（1つの文字列として展開）

# 実践: 引数を他のコマンドに渡す場合は "$@" を使う
wrapper() {
    echo "Calling command with args: $@"
    some_command "$@"  # 引数を正しく渡す
}

# ステータス・プロセス関連
echo $?                          # 直前のコマンドの終了ステータス（0=成功）
echo $$                          # 現在のプロセスID
echo $!                          # 直前のバックグラウンドプロセスのPID
echo $-                          # 現在のシェルオプション
echo $_                          # 直前のコマンドの最後の引数

# 終了ステータスの活用
ls /nonexistent 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "ディレクトリが存在しません"
fi

# より簡潔な書き方
if ls /nonexistent 2>/dev/null; then
    echo "存在する"
else
    echo "存在しない"
fi

# PIDの活用
long_process &
bg_pid=$!
echo "バックグラウンドプロセス: PID=$bg_pid"
wait $bg_pid
echo "プロセス完了: 終了ステータス=$?"
```

### 2.4 環境変数

```bash
# 環境変数の設定
export MY_VAR="value"            # 子プロセスに引き継ぐ
MY_VAR="value"                   # 現在のシェルのみ（子プロセスに渡されない）

# よく使う環境変数
echo "$HOME"                     # ホームディレクトリ（/home/gaku）
echo "$USER"                     # ユーザー名（gaku）
echo "$PATH"                     # コマンド検索パス
echo "$PWD"                      # カレントディレクトリ
echo "$OLDPWD"                   # 前のディレクトリ（cd - で使われる）
echo "$SHELL"                    # ログインシェル（/bin/bash 等）
echo "$HOSTNAME"                 # ホスト名
echo "$LANG"                     # ロケール（ja_JP.UTF-8 等）
echo "$TERM"                     # ターミナルの種類
echo "$EDITOR"                   # デフォルトエディタ
echo "$RANDOM"                   # 0-32767 のランダムな整数
echo "$SECONDS"                  # シェルの起動からの秒数
echo "$LINENO"                   # 現在の行番号
echo "$BASH_VERSION"             # Bash のバージョン

# 環境変数の一覧
env                              # 全環境変数を表示
printenv                         # 同上
printenv PATH                    # 特定の環境変数を表示

# 1回限りの環境変数設定
MY_VAR=value command             # command 実行時のみ MY_VAR を設定
LANG=C sort file.txt             # sort を C ロケールで実行
DEBUG=1 ./myapp.sh               # デバッグモードで実行

# 環境変数の削除
unset MY_VAR                     # 変数を削除
```

### 2.5 文字列操作

```bash
str="Hello, World!"

# 長さ
echo ${#str}                     # → 13

# 部分文字列
echo ${str:0:5}                  # → Hello（位置0から5文字）
echo ${str:7}                    # → World!（位置7から末尾）
echo ${str: -6}                  # → orld!（末尾から6文字。スペース必要）
echo ${str:(-6):3}               # → orl（末尾から6文字目から3文字）

# 置換
echo ${str/World/Bash}           # → Hello, Bash!（最初の1つ）
echo ${str//l/L}                 # → HeLLo, WorLd!（全て置換）
echo ${str/#Hello/Hi}            # → Hi, World!（先頭マッチのみ置換）
echo ${str/%\!/\?}               # → Hello, World?（末尾マッチのみ置換）

# 削除（パターン除去）
filename="archive.tar.gz"
echo ${filename%.gz}             # → archive.tar（末尾から最短一致）
echo ${filename%%.*}             # → archive（末尾から最長一致）
echo ${filename#*.}              # → tar.gz（先頭から最短一致）
echo ${filename##*.}             # → gz（先頭から最長一致）

# 実践: ファイル名とディレクトリの分離
filepath="/home/user/documents/report.pdf"
echo ${filepath##*/}             # → report.pdf（basename 相当）
echo ${filepath%/*}              # → /home/user/documents（dirname 相当）

# 拡張子の取得と変更
file="photo.jpg"
ext="${file##*.}"                 # → jpg
name="${file%.*}"                 # → photo
new_file="${name}.png"            # → photo.png

# 大文字・小文字変換（Bash 4+）
str="Hello World"
echo "${str^^}"                  # → HELLO WORLD（全て大文字）
echo "${str,,}"                  # → hello world（全て小文字）
echo "${str^}"                   # → Hello World（先頭のみ大文字）
echo "${str,}"                   # → hello World（先頭のみ小文字）

# デフォルト値
echo ${undefined:-"default"}     # 未定義なら "default" を表示（代入はしない）
echo ${undefined:="default"}     # 未定義なら "default" を代入して表示
echo ${undefined:+"set"}         # 定義済みなら "set" を表示、未定義なら空
echo ${undefined:?"error msg"}   # 未定義ならエラーメッセージを表示して終了

# 実践: デフォルト値の利用
LOG_DIR="${LOG_DIR:-/var/log/myapp}"
PORT="${PORT:-8080}"
ENV="${ENV:-development}"
echo "Starting on port $PORT in $ENV mode"

# 変数の間接参照
var_name="PATH"
echo "${!var_name}"              # → $PATH の内容を表示
# env | grep "^${var_name}="    # 同等の処理
```

### 2.6 配列の基本（概要）

```bash
# インデックス配列の基本（詳細は 01-advanced-scripting.md）
fruits=("apple" "banana" "cherry")
echo "${fruits[0]}"              # → apple
echo "${fruits[@]}"              # → 全要素
echo "${#fruits[@]}"             # → 3（要素数）

# 追加
fruits+=("date")

# ループ
for fruit in "${fruits[@]}"; do
    echo "$fruit"
done

# コマンドの出力を配列に格納
mapfile -t lines < /etc/hosts    # ファイルの各行を配列に
IFS=$'\n' read -d '' -ra output <<< "$(ls -1)"  # コマンド出力を配列に
```

---

## 3. 条件分岐

### 3.1 if 文の基本

```bash
# if文の基本構文
if [ "$name" = "Gaku" ]; then
    echo "Welcome, Gaku!"
elif [ "$name" = "admin" ]; then
    echo "Welcome, admin!"
else
    echo "Who are you?"
fi

# [[ ]] 形式（bash拡張、推奨）
if [[ "$name" == "Gaku" ]]; then
    echo "Match!"
fi

# [ ] と [[ ]] の違い
# [ ] (test コマンド)
#   - POSIX 準拠（sh でも使える）
#   - 文字列比較は = を使用
#   - 論理演算は -a, -o を使用
#   - ワード分割が起こるため変数はクォート必須
#
# [[ ]] (bash 拡張)
#   - bash/zsh で使える（sh では不可）
#   - 文字列比較は == を使用
#   - 論理演算は &&, || を使用
#   - ワード分割が起こらない（安全）
#   - パターンマッチ・正規表現が使える
#   - 推奨: 特にこだわりがなければ [[ ]] を使う

# 1行の条件分岐（短い場合に便利）
[[ -f "$file" ]] && echo "File exists"
[[ -f "$file" ]] || echo "File not found"
[[ -d "$dir" ]] && cd "$dir" || echo "Directory not found"
```

### 3.2 文字列の比較

```bash
# 文字列比較演算子
[[ "$a" == "$b" ]]               # 等しい
[[ "$a" != "$b" ]]               # 等しくない
[[ -z "$str" ]]                  # 空文字列（zero length）
[[ -n "$str" ]]                  # 空でない（non-zero length）
[[ "$str" =~ ^[0-9]+$ ]]        # 正規表現マッチ
[[ "$str" == pattern* ]]         # グロブパターンマッチ（* はワイルドカード）

# 文字列の比較（辞書順）
[[ "$a" < "$b" ]]                # a が辞書順で先
[[ "$a" > "$b" ]]                # a が辞書順で後

# 正規表現マッチの詳細
input="2025-01-15"
if [[ "$input" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]]; then
    echo "Year:  ${BASH_REMATCH[0]}"   # → 2025-01-15（全体）
    echo "Year:  ${BASH_REMATCH[1]}"   # → 2025
    echo "Month: ${BASH_REMATCH[2]}"   # → 01
    echo "Day:   ${BASH_REMATCH[3]}"   # → 15
fi

# パターンマッチの例
filename="report_2025.csv"
if [[ "$filename" == *.csv ]]; then
    echo "CSV file"
fi
if [[ "$filename" == report_* ]]; then
    echo "Report file"
fi

# 空文字列チェックの実践
validate_input() {
    local input="$1"
    if [[ -z "$input" ]]; then
        echo "Error: Input is empty" >&2
        return 1
    fi
    echo "Input: $input"
}
```

### 3.3 数値の比較

```bash
# 数値比較演算子（[[ ]] 内で使用）
[[ $a -eq $b ]]                  # 等しい（equal）
[[ $a -ne $b ]]                  # 等しくない（not equal）
[[ $a -lt $b ]]                  # より小さい（less than）
[[ $a -gt $b ]]                  # より大きい（greater than）
[[ $a -le $b ]]                  # 以下（less or equal）
[[ $a -ge $b ]]                  # 以上（greater or equal）

# (( )) 形式（数値演算専用。より直感的）
if (( count > 10 )); then
    echo "Too many"
fi
if (( age >= 18 && age <= 65 )); then
    echo "Working age"
fi
if (( score == 100 )); then
    echo "Perfect!"
fi

# (( )) の注意点
# - 文字列比較はできない
# - 変数名の $ は省略可能: (( count > 10 )) と (( $count > 10 )) は同じ
# - 0 は偽、非0 は真: (( 0 )) → false, (( 1 )) → true
# - 未定義変数は 0 として扱われる
```

### 3.4 ファイルの判定

```bash
# ファイルの存在・種類
[[ -f "$file" ]]                 # 通常ファイルが存在
[[ -d "$dir" ]]                  # ディレクトリが存在
[[ -e "$path" ]]                 # 存在する（種類問わず）
[[ -L "$path" ]]                 # シンボリックリンク
[[ -p "$path" ]]                 # 名前付きパイプ（FIFO）
[[ -S "$path" ]]                 # ソケット
[[ -b "$path" ]]                 # ブロックデバイス
[[ -c "$path" ]]                 # キャラクタデバイス

# ファイルの属性
[[ -r "$file" ]]                 # 読み取り可能
[[ -w "$file" ]]                 # 書き込み可能
[[ -x "$file" ]]                 # 実行可能
[[ -s "$file" ]]                 # サイズが0でない
[[ -O "$file" ]]                 # 現在のユーザーが所有者
[[ -G "$file" ]]                 # 現在のグループが所有グループ

# ファイルの比較
[[ "$file1" -nt "$file2" ]]     # file1 が新しい（newer than）
[[ "$file1" -ot "$file2" ]]     # file1 が古い（older than）
[[ "$file1" -ef "$file2" ]]     # 同じinode（同一ファイルかハードリンク）

# 実践: ファイル存在チェック付きの処理
config_file="/etc/myapp/config.yml"
if [[ ! -f "$config_file" ]]; then
    echo "Error: Config file not found: $config_file" >&2
    exit 1
fi
if [[ ! -r "$config_file" ]]; then
    echo "Error: Cannot read config file: $config_file" >&2
    exit 1
fi
echo "Loading config from $config_file..."

# ディレクトリの作成（存在しない場合のみ）
log_dir="/var/log/myapp"
if [[ ! -d "$log_dir" ]]; then
    mkdir -p "$log_dir"
    echo "Created log directory: $log_dir"
fi
```

### 3.5 論理演算

```bash
# [[ ]] 内の論理演算
[[ $a -gt 0 && $a -lt 100 ]]    # AND
[[ $a -eq 0 || $a -eq 1 ]]      # OR
[[ ! -f "$file" ]]               # NOT

# 複数条件の組み合わせ
if [[ -f "$file" && -r "$file" && -s "$file" ]]; then
    echo "File exists, is readable, and is not empty"
fi

# 条件の優先順位（グルーピング）
if [[ ( $a -gt 0 && $a -lt 10 ) || $a -eq 100 ]]; then
    echo "a is 1-9 or 100"
fi

# コマンドの論理演算
command1 && command2             # command1 が成功したら command2 を実行
command1 || command2             # command1 が失敗したら command2 を実行
command1 && command2 || command3 # 成功なら command2、失敗なら command3

# 実践パターン
cd "$dir" && echo "Moved to $dir" || echo "Failed to move to $dir"
grep -q "pattern" "$file" && echo "Found" || echo "Not found"
```

### 3.6 case 文

```bash
# case文の基本
case "$1" in
    start)
        echo "Starting..."
        ;;
    stop)
        echo "Stopping..."
        ;;
    restart)
        echo "Restarting..."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

# パターンマッチの活用
case "$filename" in
    *.tar.gz|*.tgz)
        tar xzf "$filename"
        ;;
    *.tar.bz2)
        tar xjf "$filename"
        ;;
    *.zip)
        unzip "$filename"
        ;;
    *.gz)
        gunzip "$filename"
        ;;
    *.tar)
        tar xf "$filename"
        ;;
    *)
        echo "Unknown archive format: $filename"
        ;;
esac

# case の高度なパターン
case "$input" in
    [0-9])
        echo "Single digit"
        ;;
    [0-9][0-9])
        echo "Two digits"
        ;;
    [a-zA-Z]*)
        echo "Starts with a letter"
        ;;
    "")
        echo "Empty string"
        ;;
    *)
        echo "Other"
        ;;
esac

# yes/no の確認プロンプト
read -p "Continue? [y/N] " response
case "$response" in
    [yY]|[yY][eE][sS])
        echo "Proceeding..."
        ;;
    *)
        echo "Aborted."
        exit 0
        ;;
esac

# Bash 4+ の case: ;& と ;;&
# ;; → マッチしたら case を終了（通常の動作）
# ;& → 次のパターンも無条件で実行（C の fall-through）
# ;;& → 次のパターンもチェックして実行（続行チェック）
case "$level" in
    critical)
        echo "Paging on-call"
        ;&  # fall-through
    error)
        echo "Sending alert email"
        ;&  # fall-through
    warning)
        echo "Logging to file"
        ;;
esac
```

### 3.7 条件分岐の実践パターン

```bash
# パターン1: コマンドの存在チェック
if command -v docker &>/dev/null; then
    echo "Docker is installed"
else
    echo "Docker is not installed"
    exit 1
fi

# パターン2: OS判定
case "$(uname -s)" in
    Linux)
        echo "Running on Linux"
        PACKAGE_MANAGER="apt"
        ;;
    Darwin)
        echo "Running on macOS"
        PACKAGE_MANAGER="brew"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        echo "Running on Windows"
        ;;
    *)
        echo "Unknown OS"
        ;;
esac

# パターン3: 数値の範囲判定
validate_port() {
    local port="$1"
    if ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "Error: Not a number: $port" >&2
        return 1
    fi
    if (( port < 1 || port > 65535 )); then
        echo "Error: Port must be 1-65535: $port" >&2
        return 1
    fi
    if (( port < 1024 )); then
        echo "Warning: Privileged port (requires root): $port" >&2
    fi
    return 0
}

# パターン4: 複数条件のバリデーション
validate_config() {
    local errors=0

    if [[ -z "$DB_HOST" ]]; then
        echo "Error: DB_HOST is not set" >&2
        (( errors++ ))
    fi
    if [[ -z "$DB_NAME" ]]; then
        echo "Error: DB_NAME is not set" >&2
        (( errors++ ))
    fi
    if [[ -z "$DB_USER" ]]; then
        echo "Error: DB_USER is not set" >&2
        (( errors++ ))
    fi

    if (( errors > 0 )); then
        echo "Found $errors configuration error(s)" >&2
        return 1
    fi
    return 0
}
```

---

## 4. ループ

### 4.1 for ループ

```bash
# 基本的な for ループ
for i in 1 2 3 4 5; do
    echo "Number: $i"
done

# C言語風 for
for ((i = 0; i < 10; i++)); do
    echo "Index: $i"
done

# 範囲指定（brace expansion）
for i in {1..10}; do             # 1〜10
    echo "$i"
done
for i in {0..100..5}; do         # 0〜100, 5刻み
    echo "$i"
done
for letter in {a..z}; do         # a〜z
    echo "$letter"
done

# seq コマンド（変数で範囲指定が必要な場合）
max=10
for i in $(seq 1 "$max"); do
    echo "$i"
done
for i in $(seq 0 5 100); do     # 0から100まで5刻み
    echo "$i"
done

# ファイルに対するループ
for file in *.txt; do
    echo "Processing: $file"
done

# 複数のパターン
for file in *.txt *.csv *.json; do
    echo "File: $file"
done

# パターンにマッチするファイルがない場合の対策
shopt -s nullglob               # マッチしない場合は空に展開
for file in *.txt; do
    echo "Processing: $file"
done
shopt -u nullglob               # 元に戻す

# コマンド出力に対するループ
for user in $(cut -d: -f1 /etc/passwd); do
    echo "User: $user"
done

# 配列のループ
servers=("web1" "web2" "web3" "db1")
for server in "${servers[@]}"; do
    echo "Checking: $server"
done

# 引数のループ
for arg in "$@"; do
    echo "Argument: $arg"
done

# インデックス付きのループ
items=("apple" "banana" "cherry")
for i in "${!items[@]}"; do
    echo "$i: ${items[$i]}"
done
```

### 4.2 while ループ

```bash
# 基本的な while ループ
count=0
while [[ $count -lt 5 ]]; do
    echo "Count: $count"
    ((count++))
done

# 無限ループ
while true; do
    echo "Running..."
    sleep 1
    # break で脱出
done

# (( )) を使った while
n=1
while (( n <= 100 )); do
    echo "$n"
    (( n++ ))
done

# ファイルを1行ずつ読む（重要パターン）
while IFS= read -r line; do
    echo "Line: $line"
done < input.txt

# IFS= と -r の意味
# IFS=  → 先頭と末尾の空白を保持する
# -r    → バックスラッシュを特殊文字として扱わない
# この組み合わせで行を正確に読み取る

# パイプからの読み取り（サブシェルに注意）
# NG: パイプの右側はサブシェルなので変数が残らない
count=0
cat file.txt | while read -r line; do
    ((count++))
done
echo "Count: $count"  # → 0（サブシェル内の変数は失われる）

# OK: リダイレクトを使う
count=0
while read -r line; do
    ((count++))
done < file.txt
echo "Count: $count"  # → 正しい値

# OK: プロセス置換を使う
count=0
while read -r line; do
    ((count++))
done < <(cat file.txt)
echo "Count: $count"  # → 正しい値

# CSVファイルの処理
while IFS=',' read -r name age city; do
    echo "Name: $name, Age: $age, City: $city"
done < data.csv

# /etc/passwd の処理
while IFS=':' read -r user _ uid gid _ home shell; do
    echo "User: $user (UID: $uid, Home: $home)"
done < /etc/passwd

# コマンド出力を行単位で処理
while read -r pid user cpu mem command; do
    if (( $(echo "$cpu > 50" | bc -l) )); then
        echo "High CPU: $command ($cpu%)"
    fi
done < <(ps aux --no-headers)
```

### 4.3 until ループ

```bash
# until ループ（条件が真になるまで繰り返す）
until ping -c1 server.com &>/dev/null; do
    echo "Waiting for server..."
    sleep 5
done
echo "Server is up!"

# サービスの起動待ち
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    echo "Waiting for application to start..."
    sleep 2
done
echo "Application is ready!"

# タイムアウト付き待機
timeout=60
elapsed=0
until docker ps | grep -q "my-container"; do
    if (( elapsed >= timeout )); then
        echo "Timeout: Container did not start within ${timeout}s"
        exit 1
    fi
    echo "Waiting for container... (${elapsed}s)"
    sleep 5
    (( elapsed += 5 ))
done
echo "Container is running!"
```

### 4.4 ループ制御

```bash
# break / continue
for i in {1..10}; do
    [[ $i -eq 3 ]] && continue   # 3をスキップ
    [[ $i -eq 8 ]] && break      # 8で終了
    echo "$i"
done
# 出力: 1 2 4 5 6 7

# ネストされたループでの break/continue
for i in {1..3}; do
    for j in {1..3}; do
        if (( i == 2 && j == 2 )); then
            break 2              # 外側のループも break（数字で深さ指定）
        fi
        echo "$i,$j"
    done
done

# continue でもネストの深さを指定可能
for i in {1..3}; do
    for j in {1..3}; do
        if (( j == 2 )); then
            continue 2           # 外側のループの次のイテレーションへ
        fi
        echo "$i,$j"
    done
done
```

### 4.5 ループの実践パターン

```bash
# パターン1: リトライ処理
max_retries=5
retry_count=0
until some_command; do
    ((retry_count++))
    if (( retry_count >= max_retries )); then
        echo "Failed after $max_retries retries"
        exit 1
    fi
    echo "Retry $retry_count/$max_retries..."
    sleep $(( retry_count * 2 ))  # 指数バックオフ
done

# パターン2: ファイルの一括処理（find + while）
find /var/log -name "*.log" -mtime +30 -print0 | while IFS= read -r -d '' file; do
    echo "Compressing: $file"
    gzip "$file"
done
# -print0 と -d '' でファイル名のスペース・改行に対応

# パターン3: プログレスバー
total=100
for ((i = 1; i <= total; i++)); do
    percent=$(( i * 100 / total ))
    bar=$(printf '%*s' $(( percent / 2 )) '' | tr ' ' '#')
    printf "\r[%-50s] %d%%" "$bar" "$percent"
    sleep 0.05
done
echo ""

# パターン4: メニューの表示（select）
PS3="Select an option: "
select opt in "Start" "Stop" "Status" "Quit"; do
    case "$opt" in
        "Start")  echo "Starting..."; ;;
        "Stop")   echo "Stopping..."; ;;
        "Status") echo "Running"; ;;
        "Quit")   break; ;;
        *)        echo "Invalid option"; ;;
    esac
done

# パターン5: ディレクトリの再帰処理
process_dir() {
    local dir="$1"
    for item in "$dir"/*; do
        if [[ -d "$item" ]]; then
            process_dir "$item"     # 再帰呼び出し
        elif [[ -f "$item" ]]; then
            echo "File: $item"
        fi
    done
}
process_dir "/path/to/start"
```

---

## 5. 関数

### 5.1 関数の定義と呼び出し

```bash
# 関数定義（2つの書き方）
greet() {
    local name="$1"              # local: 関数内変数
    echo "Hello, $name!"
}

function greet2 {
    local name="$1"
    echo "Hi, $name!"
}

# 呼び出し
greet "Gaku"                     # → Hello, Gaku!
greet2 "World"                   # → Hi, World!

# 関数の引数
show_args() {
    echo "Function name: $FUNCNAME"
    echo "Argument count: $#"
    echo "First arg: $1"
    echo "Second arg: $2"
    echo "All args: $@"
}
show_args "hello" "world" "foo"
```

### 5.2 戻り値

```bash
# return で終了ステータスを返す（0=成功、1-255=失敗）
is_even() {
    if (( $1 % 2 == 0 )); then
        return 0                 # 成功（真）
    else
        return 1                 # 失敗（偽）
    fi
}

if is_even 4; then
    echo "4 is even"
fi

# 値を返す（stdout 経由）
get_timestamp() {
    date +%Y%m%d_%H%M%S
}
ts=$(get_timestamp)
echo "Timestamp: $ts"

# 複数の値を返す
get_dimensions() {
    local file="$1"
    local width=$(identify -format "%w" "$file" 2>/dev/null)
    local height=$(identify -format "%h" "$file" 2>/dev/null)
    echo "$width $height"        # スペース区切りで出力
}
read -r width height <<< "$(get_dimensions photo.jpg)"
echo "Width: $width, Height: $height"

# 配列を返す（改行区切り）
list_users() {
    cut -d: -f1 /etc/passwd | sort
}
mapfile -t users < <(list_users)
echo "Total users: ${#users[@]}"

# グローバル変数で結果を返す（非推奨だが知っておくべき）
calculate() {
    RESULT=$(( $1 + $2 ))        # グローバル変数
}
calculate 3 5
echo "Result: $RESULT"           # → 8
```

### 5.3 local 変数とスコープ

```bash
# local を使わないと全てグローバル
bad_function() {
    x=100                        # グローバル！
}
bad_function
echo "$x"                        # → 100（関数外でもアクセス可能）

# local で関数内に閉じ込める
good_function() {
    local x=100                  # 関数内のみ
}
good_function
echo "$x"                        # → （空。アクセスできない）

# 実践: 常に local を使う
process_file() {
    local file="$1"
    local content
    local line_count

    content=$(cat "$file")
    line_count=$(wc -l < "$file")

    echo "File: $file ($line_count lines)"
}

# nameref（Bash 4.3+）— 参照渡し
set_value() {
    local -n ref=$1              # nameref: 呼び出し元の変数を参照
    ref="new value"
}
my_var="old value"
set_value my_var
echo "$my_var"                   # → new value
```

### 5.4 エラーハンドリング付き関数

```bash
# エラーハンドリング付き関数
safe_cd() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "Error: Directory '$dir' not found" >&2
        return 1
    fi
    cd "$dir" || return 1
}

# die 関数（スクリプト全体を終了する場合）
die() {
    echo "FATAL: $*" >&2
    exit 1
}

# 使用例
safe_cd "/opt/myapp" || die "Cannot access application directory"

# バリデーション関数
require_command() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        die "Required command not found: $cmd"
    fi
}
require_command "docker"
require_command "git"
require_command "jq"

# ファイル操作の安全なラッパー
safe_rm() {
    local target="$1"
    # 危険なパスを拒否
    case "$target" in
        /|/home|/usr|/var|/etc|/tmp)
            die "Refusing to delete critical directory: $target"
            ;;
    esac
    if [[ -e "$target" ]]; then
        rm -rf "$target"
        echo "Removed: $target"
    else
        echo "Warning: Does not exist: $target" >&2
    fi
}
```

### 5.5 関数の実践パターン

```bash
# パターン1: ログ出力関数群
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" >&2; }
log_debug() { [[ "${DEBUG:-0}" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $(date '+%H:%M:%S') $*"; }

# パターン2: 確認プロンプト
confirm() {
    local message="${1:-Continue?}"
    local default="${2:-n}"
    local prompt

    if [[ "$default" == "y" ]]; then
        prompt="$message [Y/n] "
    else
        prompt="$message [y/N] "
    fi

    read -p "$prompt" response
    response="${response:-$default}"

    [[ "$response" =~ ^[yY] ]]
}

# 使用例
if confirm "Delete all logs?"; then
    rm -rf /var/log/myapp/*
fi

# パターン3: スピナー表示
spinner() {
    local pid=$1
    local spin='|/-\'
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r%s" "${spin:$((i++ % 4)):1}"
        sleep 0.1
    done
    printf "\r"
}

# 使用例
long_process &
spinner $!
wait $!

# パターン4: タイムアウト付きの関数
wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="${3:-30}"
    local elapsed=0

    while ! nc -z "$host" "$port" 2>/dev/null; do
        if (( elapsed >= timeout )); then
            echo "Timeout waiting for $host:$port" >&2
            return 1
        fi
        sleep 1
        (( elapsed++ ))
    done
    echo "$host:$port is available"
    return 0
}

wait_for_port "localhost" 5432 60 || die "Database not available"
```

---

## 6. 入出力

### 6.1 標準入力の読み取り

```bash
# 基本的な読み取り
read -p "Enter your name: " name
read -sp "Enter password: " password    # -s: 非表示
echo ""                                  # 改行（-s は改行を出力しない）
read -t 10 -p "Quick! " answer          # -t: タイムアウト（秒）
read -r line                             # -r: バックスラッシュを特殊扱いしない
read -n 1 -p "Press any key..." key     # -n: 指定文字数で自動確定

# 複数の変数に分割して読む
echo "John 25 Tokyo" | read -r name age city
# ↑ パイプだとサブシェルになるので注意

# 正しいやり方
read -r name age city <<< "John 25 Tokyo"
echo "$name is $age years old from $city"

# 配列として読み取り
read -ra words <<< "hello world foo bar"
echo "${words[0]}"               # → hello
echo "${words[2]}"               # → foo

# デフォルト値付きの入力
read -p "Port [8080]: " port
port="${port:-8080}"

# 入力のバリデーション
while true; do
    read -p "Enter a number (1-100): " num
    if [[ "$num" =~ ^[0-9]+$ ]] && (( num >= 1 && num <= 100 )); then
        break
    fi
    echo "Invalid input. Please try again."
done
echo "You entered: $num"
```

### 6.2 リダイレクト

```bash
# 基本的なリダイレクト
echo "hello" > file.txt          # 上書き（ファイルが存在すれば消去して書き込み）
echo "world" >> file.txt         # 追記
command 2> error.log             # 標準エラーをファイルへ
command > out.log 2>&1           # 標準出力と標準エラーの両方をファイルへ
command &> both.log              # 同上（bash省略形）
command > /dev/null 2>&1         # 全出力を捨てる
command &>/dev/null              # 同上（bash省略形）

# ファイルディスクリプタ
# 0 = stdin（標準入力）
# 1 = stdout（標準出力）
# 2 = stderr（標準エラー）

# 標準エラーのみ表示（標準出力は捨てる）
command > /dev/null              # stdout だけ捨てる
command 2>/dev/null              # stderr だけ捨てる

# 標準出力と標準エラーを別ファイルに
command > stdout.log 2> stderr.log

# 標準出力と標準エラーを入れ替え
command 3>&1 1>&2 2>&3           # stdout ↔ stderr を入れ替え

# 追加のファイルディスクリプタ
exec 3> output.log               # FD3 を output.log に割り当て
echo "Log message" >&3           # FD3 に書き込み
exec 3>&-                        # FD3 を閉じる

# 入力リダイレクト
command < input.txt              # ファイルからの入力
sort < unsorted.txt > sorted.txt # 入力と出力のリダイレクト

# noclobber（上書き防止）
set -o noclobber                 # > での上書きを禁止
echo "test" > existing_file      # → エラー
echo "test" >| existing_file     # >| で強制上書き
set +o noclobber                 # 解除

# tee コマンド（画面とファイルの両方に出力）
command | tee output.log         # stdout に表示しつつファイルにも保存
command | tee -a output.log      # 追記モード
command 2>&1 | tee output.log    # stderr も含めて
```

### 6.3 ヒアドキュメント

```bash
# ヒアドキュメント（変数展開あり）
cat <<EOF
Hello, $name!
Today is $(date).
Your home directory is $HOME
EOF

# ヒアドキュメント（変数展開なし）
cat <<'EOF'
$name は展開されない
$(date) も展開されない
EOF

# ヒアドキュメントでファイルを作成
cat > /tmp/config.ini <<EOF
[database]
host=$DB_HOST
port=$DB_PORT
name=$DB_NAME
EOF

# インデント付きヒアドキュメント（<<- でタブを除去）
if true; then
    cat <<-EOF
	Hello, $name!
	This is indented with tabs.
	EOF
fi
# 注意: <<- はタブのみ除去。スペースは除去されない

# ヒアストリング（1行の入力）
grep "pattern" <<< "$variable"
read -r first rest <<< "hello world foo"
echo "$first"                    # → hello
echo "$rest"                     # → world foo

bc <<< "10 * 20 + 5"            # → 205

# 実践: SSH でリモートコマンド実行
ssh user@server <<'REMOTE'
cd /opt/myapp
git pull
npm install
pm2 restart all
REMOTE

# 実践: SQL の実行
mysql -u root -p"$DB_PASS" <<EOF
USE mydb;
SELECT COUNT(*) FROM users WHERE active = 1;
EOF
```

### 6.4 パイプとプロセス置換

```bash
# パイプ（コマンドの出力を次のコマンドの入力に）
ls -la | sort -k5 -n            # ファイルサイズ順にソート
cat access.log | grep "404" | wc -l  # 404 エラーの数

# 名前付きパイプ（FIFO）
mkfifo /tmp/mypipe
command1 > /tmp/mypipe &
command2 < /tmp/mypipe
rm /tmp/mypipe

# プロセス置換（一時ファイルの代わり）
diff <(ls dir1) <(ls dir2)       # 2つのディレクトリの内容を比較
diff <(sort file1) <(sort file2) # ソートしながら比較

# 実践: 2つのコマンドの出力を比較
diff <(ssh server1 "cat /etc/nginx/nginx.conf") \
     <(ssh server2 "cat /etc/nginx/nginx.conf")

# プロセス置換で複数の出力先
tee >(gzip > output.gz) >(wc -l > count.txt) < input.txt

# パイプのステータス（PIPESTATUS 配列）
false | true | false
echo "${PIPESTATUS[@]}"          # → 1 0 1（各コマンドの終了ステータス）
```

---

## 7. デバッグ

### 7.1 デバッグオプション

```bash
# set -x: 実行コマンドを表示（最も重要なデバッグツール）
set -x
echo "hello"                     # + echo hello が表示される
set +x                          # デバッグ表示を終了

# スクリプト全体をデバッグモードで実行
bash -x script.sh

# 部分的なデバッグ
set -x
# デバッグしたい処理
problematic_function
set +x

# set -v: コマンドを読み取り時に表示（展開前の状態）
set -v
echo "$HOME"                     # echo "$HOME" が表示される（展開前）
set +v

# PS4 でデバッグ出力をカスタマイズ
export PS4='+ ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -x
# 出力例: + script.sh:25: main(): echo hello

# BASH_XTRACEFD でデバッグ出力を別ファイルに
exec 4> /tmp/debug.log
export BASH_XTRACEFD=4
set -x
# デバッグ出力は /tmp/debug.log に書き込まれる
# stdout/stderr は通常通り画面に表示
```

### 7.2 エラーのトラブルシューティング

```bash
# よくあるエラーと対処法

# 1. "unexpected end of file"
#    → if, while, for, case の閉じ忘れ
#    → クォートの閉じ忘れ

# 2. "command not found"
#    → PATH の確認: echo $PATH
#    → 実行権限の確認: ls -la script.sh
#    → シバンの確認

# 3. "unbound variable"
#    → set -u を使用中に未定義変数を参照
#    → デフォルト値を設定: ${var:-default}

# 4. "ambiguous redirect"
#    → 変数が空の場合: > $file → > "" になる
#    → クォート: > "$file"

# 5. "too many arguments"
#    → [ ] 内でクォートなしの変数
#    → [[ ]] を使う

# シンタックスチェック（実行せずに文法確認）
bash -n script.sh                # 文法エラーのみ検出

# ShellCheck（静的解析ツール。強力に推奨）
# brew install shellcheck
shellcheck script.sh             # 潜在的な問題を検出
shellcheck -s bash script.sh     # bash 方言を指定
shellcheck -e SC2086 script.sh   # 特定の警告を除外

# ShellCheck が検出する典型的な問題
# SC2086: Double quote to prevent globbing and word splitting
# SC2034: Variable appears unused
# SC2046: Quote this to prevent word splitting
# SC2016: Expressions don't expand in single quotes
```

### 7.3 デバッグの実践テクニック

```bash
# トラップで実行行を表示
trap 'echo "DEBUG: Line $LINENO: $BASH_COMMAND"' DEBUG

# 関数のトレース
func_trace() {
    echo "TRACE: ${FUNCNAME[1]}() called from line ${BASH_LINENO[0]}"
}

# 変数の状態を確認するヘルパー
dump_vars() {
    echo "=== Variable Dump ==="
    echo "PWD=$PWD"
    echo "count=$count"
    echo "file=$file"
    echo "===================="
}

# タイミング情報付きデバッグ
debug_time() {
    echo "[$(date '+%H:%M:%S.%N')] $*" >&2
}

# 実践: 問題の切り分け
# 1. まず bash -n で文法チェック
# 2. bash -x で実行トレース
# 3. shellcheck で静的解析
# 4. 問題箇所の前後に echo を入れて変数値を確認
# 5. set -x / set +x で部分的にデバッグ
```

---

## 8. 実践的なスクリプトテンプレート

### 8.1 堅牢なスクリプトのテンプレート

```bash
#!/usr/bin/env bash
#
# スクリプト名: template.sh
# 説明: 堅牢なスクリプトのテンプレート
# 使い方: ./template.sh [OPTIONS] ARG
#

set -euo pipefail

# 定数
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly VERSION="1.0.0"

# デフォルト値
VERBOSE=false
DRY_RUN=false
OUTPUT_DIR="."

# 色定義
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly NC='\033[0m'

# ── ログ関数 ──
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()       { log_error "$@"; exit 1; }

# ── クリーンアップ ──
TMPDIR=""
cleanup() {
    local exit_code=$?
    if [[ -n "$TMPDIR" && -d "$TMPDIR" ]]; then
        rm -rf "$TMPDIR"
    fi
    exit $exit_code
}
trap cleanup EXIT

# ── ヘルプ ──
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] FILE...

Description:
  このスクリプトの説明を書く

Options:
  -h, --help         このヘルプを表示
  -V, --version      バージョンを表示
  -v, --verbose      詳細出力
  -n, --dry-run      実際には実行しない
  -o, --output DIR   出力ディレクトリ (default: .)

Examples:
  $SCRIPT_NAME -v input.txt
  $SCRIPT_NAME --output /tmp -n *.csv
EOF
}

# ── 引数パース ──
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)    usage; exit 0 ;;
            -V|--version) echo "$SCRIPT_NAME $VERSION"; exit 0 ;;
            -v|--verbose) VERBOSE=true; shift ;;
            -n|--dry-run) DRY_RUN=true; shift ;;
            -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
            --)           shift; break ;;
            -*)           die "Unknown option: $1" ;;
            *)            break ;;
        esac
    done

    # 残りの引数
    FILES=("$@")

    # バリデーション
    if [[ ${#FILES[@]} -eq 0 ]]; then
        die "No input files specified. Use -h for help."
    fi
}

# ── メイン処理 ──
main() {
    parse_args "$@"

    # 一時ディレクトリ
    TMPDIR=$(mktemp -d)

    log_info "Processing ${#FILES[@]} file(s)..."

    for file in "${FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_warn "File not found: $file"
            continue
        fi

        if [[ "$VERBOSE" == true ]]; then
            log_info "Processing: $file"
        fi

        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY RUN] Would process: $file"
            continue
        fi

        # ここに実際の処理を書く
        process_file "$file"
    done

    log_info "Done!"
}

process_file() {
    local file="$1"
    # 処理の実装
    echo "Processing: $file"
}

main "$@"
```

### 8.2 バックアップスクリプトの例

```bash
#!/usr/bin/env bash
set -euo pipefail

# 設定
readonly BACKUP_SOURCE="/home/user/data"
readonly BACKUP_DEST="/mnt/backup"
readonly MAX_BACKUPS=7
readonly DATE=$(date +%Y%m%d_%H%M%S)
readonly BACKUP_NAME="backup_${DATE}.tar.gz"
readonly LOG_FILE="/var/log/backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# バックアップ先の空き容量チェック
check_disk_space() {
    local available
    available=$(df -BM "$BACKUP_DEST" | awk 'NR==2 {print $4}' | tr -d 'M')
    local source_size
    source_size=$(du -sm "$BACKUP_SOURCE" | awk '{print $1}')

    if (( available < source_size * 2 )); then
        log "WARNING: Low disk space. Available: ${available}MB, Source: ${source_size}MB"
        return 1
    fi
}

# 古いバックアップの削除
cleanup_old_backups() {
    local count
    count=$(ls -1 "$BACKUP_DEST"/backup_*.tar.gz 2>/dev/null | wc -l)

    if (( count > MAX_BACKUPS )); then
        local to_delete=$(( count - MAX_BACKUPS ))
        ls -1t "$BACKUP_DEST"/backup_*.tar.gz | tail -n "$to_delete" | while read -r file; do
            log "Removing old backup: $file"
            rm -f "$file"
        done
    fi
}

# メイン
main() {
    log "=== Backup started ==="

    check_disk_space || exit 1

    log "Creating backup: $BACKUP_NAME"
    tar czf "$BACKUP_DEST/$BACKUP_NAME" -C "$(dirname "$BACKUP_SOURCE")" "$(basename "$BACKUP_SOURCE")"

    local size
    size=$(du -sh "$BACKUP_DEST/$BACKUP_NAME" | awk '{print $1}')
    log "Backup created: $BACKUP_NAME ($size)"

    cleanup_old_backups

    log "=== Backup completed ==="
}

main "$@"
```

### 8.3 ファイル監視スクリプトの例

```bash
#!/usr/bin/env bash
set -euo pipefail

# ファイルの変更を検知して処理を実行する
readonly WATCH_DIR="${1:-.}"
readonly INTERVAL="${2:-5}"

declare -A file_mtimes

get_mtime() {
    stat -c %Y "$1" 2>/dev/null || echo "0"
}

scan_files() {
    local changed=false
    for file in "$WATCH_DIR"/*; do
        [[ -f "$file" ]] || continue
        local current_mtime
        current_mtime=$(get_mtime "$file")
        local basename
        basename=$(basename "$file")

        if [[ -z "${file_mtimes[$basename]:-}" ]]; then
            file_mtimes[$basename]="$current_mtime"
            echo "[NEW] $basename"
            changed=true
        elif [[ "${file_mtimes[$basename]}" != "$current_mtime" ]]; then
            file_mtimes[$basename]="$current_mtime"
            echo "[MOD] $basename"
            changed=true
        fi
    done

    if [[ "$changed" == true ]]; then
        echo "[$(date '+%H:%M:%S')] Changes detected. Running build..."
        # ここにビルドコマンドなどを入れる
    fi
}

echo "Watching $WATCH_DIR every ${INTERVAL}s..."
while true; do
    scan_files
    sleep "$INTERVAL"
done
```

---

## まとめ

| 構文 | 用途 | 例 |
|------|------|----|
| `$var` / `${var}` | 変数参照 | `echo "$name"` |
| `$(command)` | コマンド置換 | `today=$(date)` |
| `$((expr))` | 算術演算 | `$((count + 1))` |
| `[[ cond ]]` | 条件判定 | `[[ -f "$file" ]]` |
| `(( expr ))` | 数値条件 | `(( count > 10 ))` |
| `for / while / until` | ループ | `for i in {1..10}; do` |
| `case` | パターンマッチ分岐 | `case "$1" in ...` |
| `func() { ... }` | 関数定義 | `greet() { echo "Hi"; }` |
| `local var` | 関数内ローカル変数 | `local name="$1"` |
| `set -euo pipefail` | 堅牢性設定 | スクリプト冒頭に |
| `trap '...' SIGNAL` | シグナルハンドラ | `trap cleanup EXIT` |
| `> / >> / 2>` | リダイレクト | `cmd > file 2>&1` |
| `<<EOF` | ヒアドキュメント | 複数行入力 |
| `<<<` | ヒアストリング | 1行入力 |

---

## 次に読むべきガイド
→ [[01-advanced-scripting.md]] -- 高度なシェルスクリプト

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.24-36, 2019.
2. "Bash Reference Manual." GNU, 2024.
3. "Google Shell Style Guide." google.github.io/styleguide.
4. Cooper, M. "Advanced Bash-Scripting Guide." TLDP, 2014.
5. "ShellCheck Wiki." github.com/koalaman/shellcheck/wiki.
6. Robbins, A., Beebe, N. "Classic Shell Scripting." O'Reilly, 2005.
7. Taylor, D. "Wicked Cool Shell Scripts." 2nd Ed, No Starch Press, 2016.
