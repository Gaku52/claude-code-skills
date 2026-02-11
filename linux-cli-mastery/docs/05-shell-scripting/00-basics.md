# シェルスクリプト基礎

> シェルスクリプトは CLI の操作を自動化する最も直接的な方法。

## この章で学ぶこと

- [ ] シェルスクリプトの基本構文を理解する
- [ ] 変数・条件分岐・ループを使える
- [ ] 関数とスクリプトの引数処理ができる

---

## 1. スクリプトの基本

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
```

---

## 2. 変数

```bash
# 変数の代入（= の前後にスペースを入れない）
name="Gaku"                      # 文字列
count=42                         # 数値
path="/var/log"                  # パス

# 変数の参照
echo "$name"                     # Gaku
echo "${name}_suffix"            # Gaku_suffix（区切りが必要な場合）
echo "Hello, $name!"             # Hello, Gaku!

# クォートの違い
echo "Hello, $name"              # → Hello, Gaku（変数展開される）
echo 'Hello, $name'              # → Hello, $name（リテラル）

# コマンド置換
today=$(date +%Y-%m-%d)          # 推奨: $() 形式
files=`ls`                       # 旧形式: バッククォート（非推奨）

# 算術演算
result=$((3 + 5))                # → 8
count=$((count + 1))             # インクリメント
echo $((10 / 3))                 # → 3（整数除算）
echo $((10 % 3))                 # → 1（剰余）

# 特殊変数
echo $0                          # スクリプト名
echo $1                          # 第1引数
echo $2                          # 第2引数
echo $#                          # 引数の数
echo $@                          # 全引数（個別に展開）
echo $*                          # 全引数（1つの文字列）
echo $?                          # 直前のコマンドの終了ステータス
echo $$                          # 現在のプロセスID
echo $!                          # 直前のバックグラウンドPID

# 環境変数
export MY_VAR="value"            # 子プロセスに引き継ぐ
echo "$HOME"                     # ホームディレクトリ
echo "$USER"                     # ユーザー名
echo "$PATH"                     # コマンド検索パス
echo "$PWD"                      # カレントディレクトリ
echo "$SHELL"                    # 現在のシェル
```

### 文字列操作

```bash
str="Hello, World!"

# 長さ
echo ${#str}                     # → 13

# 部分文字列
echo ${str:0:5}                  # → Hello（位置0から5文字）
echo ${str:7}                    # → World!（位置7から末尾）

# 置換
echo ${str/World/Bash}           # → Hello, Bash!（最初の1つ）
echo ${str//l/L}                 # → HeLLo, WorLd!（全て）

# 削除
filename="archive.tar.gz"
echo ${filename%.gz}             # → archive.tar（末尾から最短一致）
echo ${filename%%.*}             # → archive（末尾から最長一致）
echo ${filename#*.}              # → tar.gz（先頭から最短一致）
echo ${filename##*.}             # → gz（先頭から最長一致）

# デフォルト値
echo ${undefined:-"default"}     # 未定義なら "default"
echo ${undefined:="default"}     # 未定義なら代入して展開
```

---

## 3. 条件分岐

```bash
# if文
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

# 文字列の比較
[[ "$a" == "$b" ]]               # 等しい
[[ "$a" != "$b" ]]               # 等しくない
[[ -z "$str" ]]                  # 空文字列
[[ -n "$str" ]]                  # 空でない
[[ "$str" =~ ^[0-9]+$ ]]        # 正規表現マッチ

# 数値の比較
[[ $a -eq $b ]]                  # 等しい（equal）
[[ $a -ne $b ]]                  # 等しくない（not equal）
[[ $a -lt $b ]]                  # より小さい（less than）
[[ $a -gt $b ]]                  # より大きい（greater than）
[[ $a -le $b ]]                  # 以下（less or equal）
[[ $a -ge $b ]]                  # 以上（greater or equal）

# (( )) 形式（数値演算）
if (( count > 10 )); then
    echo "Too many"
fi

# ファイルの判定
[[ -f "$file" ]]                 # 通常ファイルが存在
[[ -d "$dir" ]]                  # ディレクトリが存在
[[ -e "$path" ]]                 # 存在する（種類問わず）
[[ -r "$file" ]]                 # 読み取り可能
[[ -w "$file" ]]                 # 書き込み可能
[[ -x "$file" ]]                 # 実行可能
[[ -s "$file" ]]                 # サイズが0でない
[[ "$file1" -nt "$file2" ]]     # file1 が新しい（newer than）

# 論理演算
[[ $a -gt 0 && $a -lt 100 ]]    # AND
[[ $a -eq 0 || $a -eq 1 ]]      # OR
[[ ! -f "$file" ]]               # NOT

# case文
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
```

---

## 4. ループ

```bash
# for ループ
for i in 1 2 3 4 5; do
    echo "Number: $i"
done

# C言語風 for
for ((i = 0; i < 10; i++)); do
    echo "Index: $i"
done

# 範囲指定
for i in {1..10}; do             # 1〜10
    echo "$i"
done
for i in {0..100..5}; do         # 0〜100, 5刻み
    echo "$i"
done

# ファイルに対するループ
for file in *.txt; do
    echo "Processing: $file"
done

# コマンド出力に対するループ
for user in $(cut -d: -f1 /etc/passwd); do
    echo "User: $user"
done

# while ループ
count=0
while [[ $count -lt 5 ]]; do
    echo "Count: $count"
    ((count++))
done

# ファイルを1行ずつ読む（重要パターン）
while IFS= read -r line; do
    echo "Line: $line"
done < input.txt

# パイプからの読み取り
ps aux | while read -r line; do
    echo "$line" | grep nginx
done

# until ループ（条件が真になるまで）
until ping -c1 server.com &>/dev/null; do
    echo "Waiting for server..."
    sleep 5
done
echo "Server is up!"

# break / continue
for i in {1..10}; do
    [[ $i -eq 3 ]] && continue   # 3をスキップ
    [[ $i -eq 8 ]] && break      # 8で終了
    echo "$i"
done
```

---

## 5. 関数

```bash
# 関数定義
greet() {
    local name="$1"              # local: 関数内変数
    echo "Hello, $name!"
}
greet "Gaku"                     # → Hello, Gaku!

# 戻り値
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

# エラーハンドリング付き関数
safe_cd() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "Error: Directory '$dir' not found" >&2
        return 1
    fi
    cd "$dir" || return 1
}
```

---

## 6. 入出力

```bash
# 標準入力の読み取り
read -p "Enter your name: " name
read -sp "Enter password: " password    # -s: 非表示
read -t 10 -p "Quick! " answer          # -t: タイムアウト
read -r line                             # -r: バックスラッシュを特殊扱いしない

# リダイレクト
echo "hello" > file.txt          # 上書き
echo "world" >> file.txt         # 追記
command 2> error.log             # 標準エラーをファイルへ
command > out.log 2>&1           # 両方をファイルへ
command &> both.log              # 同上（bash省略形）
command > /dev/null 2>&1         # 全出力を捨てる

# ヒアドキュメント
cat <<EOF
Hello, $name!
Today is $(date).
EOF

# ヒアドキュメント（変数展開なし）
cat <<'EOF'
$name は展開されない
EOF

# ヒアストリング
grep "pattern" <<< "$variable"
```

---

## まとめ

| 構文 | 用途 |
|------|------|
| $var / ${var} | 変数参照 |
| $(command) | コマンド置換 |
| $((expr)) | 算術演算 |
| [[ cond ]] | 条件判定 |
| for / while / until | ループ |
| func() { ... } | 関数定義 |
| local var | 関数内ローカル変数 |

---

## 次に読むべきガイド
→ [[01-advanced-scripting.md]] — 高度なシェルスクリプト

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.24-36, 2019.
2. "Bash Reference Manual." GNU, 2024.
