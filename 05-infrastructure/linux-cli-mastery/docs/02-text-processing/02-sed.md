# ストリームエディタ（sed）

> sed は「テキストを行単位で変換する」パイプラインの強力な変換器。

## この章で学ぶこと

- [ ] sed の基本的な置換・削除・挿入ができる
- [ ] 正規表現を活用した高度な置換ができる
- [ ] アドレス指定（行範囲・パターン範囲）を使いこなせる
- [ ] 複数コマンドの組み合わせとスクリプトファイルを活用できる
- [ ] 実務で使える sed パターンを知る
- [ ] GNU sed と BSD sed の違いを理解する

---

## 1. sed の基本

### 1.1 基本構文と動作原理

```bash
# 基本構文: sed [オプション] 'コマンド' [ファイル...]
#
# sed の動作原理:
# 1. 入力から1行を読み込む（パターンスペースに格納）
# 2. コマンドを順番に適用する
# 3. 結果を出力する
# 4. 次の行へ進み、1に戻る
#
# sed はデフォルトで全行を出力する（変更がなくても）
# -n オプションで自動出力を抑制し、明示的に p コマンドで出力する

# 基本的な使い方
sed 's/old/new/' file.txt           # 各行の最初の "old" を "new" に置換
echo "hello world" | sed 's/world/earth/'  # パイプからの入力

# ファイルを直接編集（-i オプション）
sed -i 's/old/new/g' file.txt       # ファイルを直接書き換え（GNU sed）
sed -i '' 's/old/new/g' file.txt    # macOS の BSD sed（空文字列のバックアップ拡張子）
sed -i.bak 's/old/new/g' file.txt   # バックアップを作成して書き換え
```

### 1.2 GNU sed と BSD sed の違い

```bash
# macOS（BSD sed）と Linux（GNU sed）で構文が異なる点がある

# -i オプション（インプレース編集）
sed -i 's/old/new/g' file.txt       # GNU sed: バックアップ拡張子はオプション
sed -i '' 's/old/new/g' file.txt    # BSD sed: バックアップ拡張子が必須（空文字可）
sed -i.bak 's/old/new/g' file.txt   # 両方で動作（バックアップ作成）

# 改行の扱い
sed 's/$/\n/' file.txt              # GNU sed: \n が改行に展開される
sed 's/$/\'$'\n''/' file.txt        # BSD sed: $'\n' で改行を指定

# -E オプション（拡張正規表現）
sed -E 's/(foo|bar)/baz/g' file.txt # 両方で動作（-E は共通）
# GNU sed では -r も使える（-E のエイリアス）
sed -r 's/(foo|bar)/baz/g' file.txt # GNU sed のみ

# ポータブルな書き方:
# - バックアップ拡張子を常に指定する: sed -i.bak
# - -E を使う（-r の代わりに）
# - 改行は $'\n' 構文を使う
# - または gsed (GNU sed) を macOS にインストール: brew install gnu-sed
```

---

## 2. 置換コマンド（s）

### 2.1 基本的な置換

```bash
# 基本構文: s/パターン/置換文字列/フラグ
# デフォルトは各行の最初のマッチのみ置換

sed 's/old/new/' file.txt           # 各行の最初の old を new に
sed 's/old/new/g' file.txt          # 全ての old を new に（g = global）
sed 's/old/new/2' file.txt          # 各行の2番目の old を new に
sed 's/old/new/3g' file.txt         # 各行の3番目以降の old を new に

# 大小文字を無視した置換
sed 's/old/new/gi' file.txt         # 大小文字無視で全置換（GNU sed）
sed 's/old/new/gI' file.txt         # I フラグ（GNU sed の別記法）

# 置換結果の確認（-n + p フラグ）
sed -n 's/old/new/p' file.txt       # 置換が行われた行のみ表示
sed -n 's/old/new/gp' file.txt      # 全置換が行われた行のみ表示
```

### 2.2 区切り文字の変更

```bash
# デフォルトの区切り文字は / だが、任意の文字を使用可能
# URL やファイルパスを扱う際に便利

# | を区切り文字に
sed 's|http://old.com|https://new.com|g' file.txt
sed 's|/usr/local/bin|/opt/bin|g' file.txt

# # を区切り文字に
sed 's#old/path#new/path#g' file.txt
sed 's#/var/log/old#/var/log/new#g' file.txt

# , を区切り文字に
sed 's,foo,bar,g' file.txt

# @ を区切り文字に
sed 's@pattern@replacement@g' file.txt

# 使い分けのコツ:
# パターンや置換文字列に含まれない文字を区切り文字に選ぶ
# URL → | や # を使用
# ファイルパス → | や # を使用
# 正規表現 → @ や , を使用
```

### 2.3 正規表現を使った置換

```bash
# 基本正規表現（BRE）- デフォルト
sed 's/^/  /' file.txt               # 各行の先頭に2スペース追加
sed 's/$/;/' file.txt                # 各行の末尾にセミコロン追加
sed 's/^[ \t]*//' file.txt           # 行頭の空白を削除（左トリム）
sed 's/[ \t]*$//' file.txt           # 行末の空白を削除（右トリム）
sed 's/^[ \t]*//;s/[ \t]*$//' file.txt  # 両端の空白を削除（フルトリム）

# 拡張正規表現（ERE）- -E オプション
sed -E 's/[0-9]+/NUM/g' file.txt     # 数字列を NUM に置換
sed -E 's/(error|warning)/[\1]/g' file.txt  # error/warning を角括弧で囲む
sed -E 's/^(#.*)$//' file.txt        # コメント行を空行に

# 文字クラス
sed 's/[aeiou]/*/g' file.txt         # 母音をアスタリスクに
sed 's/[[:digit:]]/X/g' file.txt     # 数字を X に
sed 's/[[:upper:]]/\L&/g' file.txt   # 大文字を小文字に（GNU sed）
sed 's/[[:space:]]\+/ /g' file.txt   # 連続空白を1つのスペースに

# ワイルドカード
sed 's/error.*/ERROR FOUND/' file.txt  # "error" 以降を全て置換
sed 's/".*//' file.txt                 # 最初の " 以降を全て削除
```

### 2.4 後方参照（キャプチャグループ）

```bash
# \( \) でグループ化し、\1, \2, ... で参照（BRE）
# ( ) でグループ化し、\1, \2, ... で参照（ERE: -E オプション）

# 基本的な後方参照
sed 's/\(.*\)=\(.*\)/\2=\1/' file.txt          # key=value → value=key
sed -E 's/(.*):(.*)=(.*)/\1: \2 -> \3/' file.txt  # 再フォーマット

# 数字の入れ替え
sed -E 's/([0-9]+)-([0-9]+)/\2-\1/' file.txt    # 12-34 → 34-12

# ファイル名の変換
echo "photo_2026.jpg" | sed -E 's/(.*)_([0-9]+)\.(.*)/\1-\2.\3/'
# → photo-2026.jpg

# HTMLタグの変換
sed -E 's/<b>(.*)<\/b>/<strong>\1<\/strong>/g' file.html
# → <b>text</b> を <strong>text</strong> に

# CSV の列操作
sed -E 's/^([^,]*),([^,]*),(.*)$/\2,\1,\3/' data.csv
# → 1列目と2列目を入れ替え

# 日付フォーマットの変換
sed -E 's/([0-9]{2})\/([0-9]{2})\/([0-9]{4})/\3-\1-\2/g' file.txt
# → MM/DD/YYYY → YYYY-MM-DD

# 重複する単語の検出
sed -En 's/\b(\w+)\s+\1\b/[\1 \1]/gp' file.txt
# → 連続する同じ単語を角括弧で囲んで表示

# & は マッチした文字列全体を参照
sed 's/[0-9]\+/(&)/g' file.txt       # 数字を括弧で囲む
sed 's/.*/[ & ]/' file.txt           # 各行を角括弧で囲む
sed 's/[A-Z][a-z]*/(&)/g' file.txt   # 大文字始まりの単語を括弧で囲む
```

### 2.5 大文字・小文字変換（GNU sed）

```bash
# \U: 以降を大文字に変換
# \L: 以降を小文字に変換
# \u: 次の1文字を大文字に
# \l: 次の1文字を小文字に
# \E: 変換を終了

sed 's/[a-z]*/\U&/' file.txt        # 各行の最初の単語を大文字に
sed 's/.*/\U&/' file.txt             # 全行を大文字に
sed 's/.*/\L&/' file.txt             # 全行を小文字に
sed -E 's/\b(\w)/\u\1/g' file.txt   # 各単語の先頭を大文字に（タイトルケース）
sed -E 's/^(\w)/\u\1/' file.txt     # 各行の先頭文字を大文字に

# 変数名のスタイル変換（snake_case → camelCase）
echo "my_variable_name" | sed -E 's/_(.)/\U\1/g'
# → myVariableName

# camelCase → snake_case
echo "myVariableName" | sed -E 's/([A-Z])/_\L\1/g'
# → my_variable_name
```

---

## 3. 行の削除コマンド（d）

### 3.1 基本的な行削除

```bash
# d コマンド: マッチした行を削除（出力しない）

# 行番号による削除
sed '5d' file.txt                   # 5行目を削除
sed '1d' file.txt                   # 1行目（先頭行）を削除
sed '$d' file.txt                   # 最終行を削除
sed '1,3d' file.txt                 # 1〜3行目を削除
sed '1,5d' file.txt                 # 1〜5行目を削除
sed '10,$d' file.txt                # 10行目以降を全て削除

# パターンによる削除
sed '/pattern/d' file.txt           # パターンにマッチする行を削除
sed '/^$/d' file.txt                # 空行を削除
sed '/^#/d' file.txt                # コメント行を削除（# で始まる行）
sed '/^;/d' file.txt                # コメント行を削除（; で始まる行）
sed '/^\/\//d' file.txt             # コメント行を削除（// で始まる行）
sed '/^[[:space:]]*$/d' file.txt    # 空白のみの行を削除

# パターンの否定（!）
sed '/pattern/!d' file.txt          # パターンにマッチしない行を削除（grep 的）
sed '/^#/!d' file.txt               # コメント行以外を削除（コメントのみ表示）

# 範囲削除
sed '/BEGIN/,/END/d' file.txt       # BEGIN〜END の行を削除
sed '1,/^---$/d' file.txt           # 1行目から --- までの行を削除
```

### 3.2 実務での行削除パターン

```bash
# 設定ファイルからコメントと空行を除去（有効な設定のみ表示）
sed '/^#/d; /^$/d; /^;/d' config.ini
sed -E '/^(#|;|$)/d' config.ini     # 同じ意味（拡張正規表現）

# ヘッダーの除去
sed '1d' data.csv                    # CSV のヘッダー行を除去
sed '1,2d' data.csv                  # 最初の2行を除去

# フッターの除去
sed '$d' file.txt                    # 最終行を除去
sed 'N; $!P; $!D; $d' file.txt      # 最後のN行を除去（複雑だが可能）

# HTML タグの除去
sed 's/<[^>]*>//g' file.html         # 全てのHTMLタグを除去
sed -E 's/<(script|style)[^>]*>.*<\/(script|style)>//g' file.html  # script/style除去

# 特定セクションの除去
sed '/<!--.*-->/d' file.html         # HTMLコメントを削除
sed '/^import/d' file.py             # import 行を全て削除
sed '/console\.log/d' file.js        # console.log 行を削除

# 連続する空行を1つにまとめる
sed '/^$/N;/^\n$/d' file.txt
# または
cat -s file.txt                      # cat -s の方が簡単
```

---

## 4. 行の表示コマンド（p）

### 4.1 特定行の表示

```bash
# -n オプションと p コマンドを組み合わせて特定行を表示
# -n: 自動出力を抑制
# p: 明示的にパターンスペースを出力

sed -n '5p' file.txt                # 5行目のみ表示
sed -n '1p' file.txt                # 1行目のみ表示
sed -n '$p' file.txt                # 最終行のみ表示
sed -n '10,20p' file.txt            # 10〜20行目を表示
sed -n '1,5p' file.txt              # 1〜5行目を表示
sed -n '100,$p' file.txt            # 100行目以降を全て表示

# パターンマッチによる表示
sed -n '/error/p' file.txt          # error を含む行を表示（grep 的）
sed -n '/^import/p' file.py         # import で始まる行を表示
sed -n '/BEGIN/,/END/p' file.txt    # BEGIN〜END の範囲を表示

# ステップ指定（GNU sed）
sed -n '1~2p' file.txt              # 奇数行のみ（1行目から2行おき）
sed -n '2~2p' file.txt              # 偶数行のみ（2行目から2行おき）
sed -n '0~3p' file.txt              # 3の倍数行
sed -n '1~5p' file.txt              # 1, 6, 11, 16, ... 行目
```

### 4.2 行番号の表示

```bash
# = コマンド: 行番号を表示
sed -n '/error/=' file.txt           # error を含む行の行番号
sed '=' file.txt                     # 全行の行番号を表示

# 行番号付きでファイルを表示
sed '=' file.txt | sed 'N; s/\n/\t/'   # 行番号とタブで連結
# → cat -n と同等の出力

# 特定パターンの行番号
sed -n '/TODO/{=;p}' file.txt        # TODO がある行番号と内容を表示
```

---

## 5. 挿入・追加・変更コマンド（i / a / c）

### 5.1 行の挿入（i）と追加（a）

```bash
# i コマンド: 指定行の前に挿入
sed '1i\#!/bin/bash' script.sh       # 1行目の前にシバン行を挿入
sed '3i\新しい行' file.txt           # 3行目の前に挿入
sed '/^import/i\# imports:' file.py  # import 行の前にコメントを挿入

# a コマンド: 指定行の後に追加
sed '1a\# This is a comment' file.txt  # 1行目の後にコメント追加
sed '$a\# End of file' file.txt     # 最終行の後に追加
sed '/^import/a\import os' file.py   # import 行の後に追加

# GNU sed での複数行挿入
sed '1i\line1\nline2\nline3' file.txt   # 複数行を挿入
sed '/pattern/a\line1\nline2' file.txt   # パターンの後に複数行追加

# BSD sed（macOS）での複数行挿入
sed '1i\
line1\
line2\
line3' file.txt

# c コマンド: 行の置換（行全体を変更）
sed '5c\この行は置き換えられました' file.txt   # 5行目を置換
sed '/old_line/c\new_line' file.txt              # パターンマッチした行を置換
sed '/^#.*deprecated/c\# This feature is removed' file.txt
```

### 5.2 実務での挿入・追加パターン

```bash
# ファイルのヘッダーを追加
sed '1i\# -*- coding: utf-8 -*-' script.py

# ライセンスヘッダーの追加
sed '1i\/*\n * Copyright 2026\n * MIT License\n */' file.js

# HTML のヘッダー/フッター追加
sed '1i\<html><body>' file.txt
sed '$a\</body></html>' file.txt

# CSV にヘッダー行を追加
sed '1i\name,age,email' data.csv

# 設定ファイルにエントリを追加
sed '/^\[database\]/a\host = localhost' config.ini
# → [database] セクションの直後に host を追加

# 特定の行の後にブロックを追加
sed '/^def main/a\    logger.info("main() started")' script.py
```

---

## 6. アドレス指定（行の選択）

### 6.1 アドレスの種類

```bash
# sed のコマンドは「アドレス」でどの行に適用するかを制御する

# 行番号アドレス
sed '5s/old/new/' file.txt           # 5行目のみ置換
sed '1s/old/new/' file.txt           # 1行目のみ
sed '$s/old/new/' file.txt           # 最終行のみ

# 行範囲アドレス
sed '1,5s/old/new/g' file.txt        # 1〜5行目のみ置換
sed '10,20s/old/new/g' file.txt      # 10〜20行目のみ
sed '10,$s/old/new/g' file.txt       # 10行目以降
sed '1,/^---$/s/old/new/g' file.txt  # 1行目から --- の行まで

# パターンアドレス
sed '/error/s/old/new/' file.txt     # error を含む行で置換
sed '/^#/s/old/new/' file.txt        # コメント行で置換
sed '/^$/!s/old/new/' file.txt       # 空行以外で置換（! = 否定）

# パターン範囲アドレス
sed '/BEGIN/,/END/s/old/new/g' file.txt   # BEGIN〜END の範囲で置換
sed '/^<div/,/^<\/div>/s/old/new/g' file.html  # <div>〜</div> 内で置換

# ステップアドレス（GNU sed）
sed '0~2s/old/new/g' file.txt        # 偶数行で置換
sed '1~2s/old/new/g' file.txt        # 奇数行で置換

# 否定アドレス（!）
sed '1!s/old/new/g' file.txt         # 1行目以外で置換
sed '/^#/!s/old/new/g' file.txt      # コメント行以外で置換
sed '1,5!d' file.txt                 # 1〜5行目以外を削除（= 1〜5行目のみ表示）
```

### 6.2 アドレスの組み合わせ例

```bash
# ヘッダー部分（1〜5行目）だけ大文字に
sed '1,5s/.*/\U&/' file.txt

# コメント行を除いて置換
sed '/^#/!s/foo/bar/g' config.conf

# 特定セクション内でのみ編集
sed '/^\[production\]/,/^\[/s/debug = true/debug = false/' config.ini
# → [production] セクション内の debug を false に変更

# 特定の関数内でのみ編集
sed '/^def process/,/^def /s/print(/logger.info(/g' script.py
# → process 関数内の print を logger.info に変更

# 最初のマッチした行のみ処理（GNU sed の 0, アドレス）
sed '0,/pattern/s/pattern/replacement/' file.txt
# → 最初の pattern のみ置換（他はそのまま）
```

---

## 7. 複数コマンドとスクリプト

### 7.1 複数コマンドの実行

```bash
# -e オプション: 複数コマンドを指定
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt
sed -e '1d' -e 's/old/new/g' file.txt
sed -e '/^#/d' -e '/^$/d' -e 's/  */ /g' file.txt

# セミコロン区切り: 複数コマンドを1つの引数で指定
sed 's/foo/bar/g; s/baz/qux/g' file.txt
sed '/^#/d; /^$/d; s/  */ /g' file.txt

# 中括弧でグループ化: 特定アドレスに複数コマンドを適用
sed '/error/{s/old/new/g; s/foo/bar/g}' file.txt
# → error を含む行に対して2つの置換を実行

sed '1,5{/^#/d; s/old/new/g}' file.txt
# → 1〜5行目のコメント行を削除し、残りの行で置換
```

### 7.2 sed スクリプトファイル

```bash
# -f オプション: スクリプトファイルからコマンドを読み込み
sed -f commands.sed file.txt

# スクリプトファイルの例 (commands.sed)
cat > commands.sed << 'EOF'
# コメント行を削除
/^#/d

# 空行を削除
/^$/d

# 先頭と末尾の空白を削除
s/^[[:space:]]*//
s/[[:space:]]*$//

# foo を bar に置換
s/foo/bar/g

# error を [ERROR] に変換
s/error/[ERROR]/gi
EOF

sed -f commands.sed logfile.txt

# 複雑な処理はスクリプトファイルにまとめると可読性が向上する
```

### 7.3 ホールドスペースの活用

```bash
# sed には2つのバッファがある:
# - パターンスペース: 現在処理中の行
# - ホールドスペース: 一時保存用のバッファ
#
# 関連コマンド:
# h: パターンスペースをホールドスペースにコピー
# H: パターンスペースをホールドスペースに追加（改行で連結）
# g: ホールドスペースをパターンスペースにコピー
# G: ホールドスペースをパターンスペースに追加
# x: パターンスペースとホールドスペースを交換

# ファイルを逆順に表示（tac と同等）
sed -n '1!G;h;$p' file.txt

# 行を2行ずつまとめる
sed 'N;s/\n/ /' file.txt

# 偶数行と奇数行を入れ替える
sed -n 'h;n;p;x;p' file.txt

# 各行を2回表示
sed 'p' file.txt

# ブランク行の後に行番号を挿入
sed '/^$/a\---' file.txt
```

---

## 8. 実務パターン集

### 8.1 ファイル内の一括置換

```bash
# 単一ファイルの置換
sed -i 's/http:/https:/g' file.html       # HTTP を HTTPS に
sed -i 's/localhost/production.server.com/g' config.yml  # ホスト名変更
sed -i 's/v1\.0/v2.0/g' README.md         # バージョン番号更新
sed -i "s/Copyright 2025/Copyright 2026/g" *.py  # 年号更新

# 複数ファイルの一括置換（find + sed）
find . -name "*.html" -exec sed -i 's/http:/https:/g' {} +
find . -name "*.py" -exec sed -i 's/old_module/new_module/g' {} +
find . -name "*.js" -exec sed -i 's/var /const /g' {} +

# grep で対象ファイルを絞り込んでから sed（効率的）
grep -rl "old_function" ./src/ | xargs sed -i 's/old_function/new_function/g'
rg -l "deprecated_api" -t py | xargs sed -i 's/deprecated_api/new_api/g'

# 安全な一括置換（スペース入りファイル名対応）
find . -name "*.txt" -print0 | xargs -0 sed -i 's/old/new/g'
grep -rlZ "pattern" . | xargs -0 sed -i 's/pattern/replacement/g'
```

### 8.2 設定ファイルの編集

```bash
# 特定のキーの値を変更
sed -i 's/^port = .*/port = 8080/' config.ini
sed -i 's/^debug = .*/debug = false/' config.ini
sed -i 's/^log_level = .*/log_level = WARNING/' config.ini

# 特定セクション内のキーを変更
sed -i '/^\[production\]/,/^\[/{s/^host = .*/host = prod-db.example.com/}' config.ini

# 設定の追加（既存のキーがなければ追加）
grep -q "^new_setting" config.ini || sed -i '$a\new_setting = value' config.ini

# 設定のコメントアウト/アンコメント
sed -i 's/^server_name/# server_name/' nginx.conf     # コメントアウト
sed -i 's/^# server_name/server_name/' nginx.conf      # アンコメント
sed -i '/^# *enable_feature/s/^# *//' config.ini       # 先頭の # を除去

# 環境変数の展開（変数を使った置換）
DB_HOST="production-db.example.com"
sed -i "s/DB_HOST=.*/DB_HOST=$DB_HOST/" .env
# 注意: 変数展開するためダブルクォートを使用

# テンプレートファイルの変数展開
sed -e "s/{{APP_NAME}}/$APP_NAME/g" \
    -e "s/{{DB_HOST}}/$DB_HOST/g" \
    -e "s/{{DB_PORT}}/$DB_PORT/g" \
    template.conf > output.conf
```

### 8.3 ログファイルの加工

```bash
# タイムスタンプの変換
sed -E 's/([0-9]{4})-([0-9]{2})-([0-9]{2})/\2\/\3\/\1/g' logfile.txt
# → YYYY-MM-DD を MM/DD/YYYY に

# IPアドレスのマスク
sed -E 's/([0-9]+\.[0-9]+\.[0-9]+\.)[0-9]+/\1XXX/g' access.log
# → 最後のオクテットをマスク

# 機密情報のマスク
sed -E 's/(password[=:])\s*\S+/\1 ********/gi' config.log
sed -E 's/(api_key[=:])\s*\S+/\1 [REDACTED]/gi' app.log
sed -E 's/([0-9]{4})[0-9]{8}([0-9]{4})/\1********\2/g' transaction.log
# → クレジットカード番号のマスク

# ログレベルの強調
sed 's/ERROR/*** ERROR ***/g; s/FATAL/!!! FATAL !!!/g' logfile.txt

# JSON ログの整形（簡易版）
sed 's/,/,\n/g; s/{/{\n/g; s/}/\n}/g' json.log
```

### 8.4 コード変換パターン

```bash
# インデントの変換（タブ → スペース）
sed 's/\t/    /g' file.py             # タブを4スペースに
sed -i 's/\t/  /g' file.yaml          # タブを2スペースに

# インデントの変換（スペース → タブ）
sed 's/    /\t/g' file.txt            # 4スペースをタブに

# 改行コードの変換
sed 's/\r$//' file.txt                # CRLF → LF（Windows → Unix）
sed 's/$/\r/' file.txt                # LF → CRLF（Unix → Windows）

# Python 2 → Python 3 の簡易変換
sed -i 's/print \(.*\)/print(\1)/' *.py         # print 文を print() に
sed -i 's/raw_input/input/g' *.py                # raw_input → input
sed -i 's/xrange/range/g' *.py                   # xrange → range
sed -i "s/except \(.*\), \(.*\):/except \1 as \2:/" *.py  # except 構文

# import 文のソート補助
sed -n '/^import/p; /^from/p' file.py | sort

# 関数名の一括リネーム
sed -i 's/\bold_func\b/new_func/g' *.py
sed -i -E 's/\bold_class\b/NewClass/g' *.py

# コメントスタイルの変換
sed 's|//\(.*\)|/*\1 */|' file.c      # C++ コメントを C コメントに
sed 's|/\*\(.*\)\*/|//\1|' file.c     # C コメントを C++ コメントに
```

### 8.5 テキストの整形

```bash
# 行番号の追加
sed '=' file.txt | sed 'N; s/\n/\t/'   # タブ区切りの行番号

# 各行をクォートで囲む
sed 's/.*/"&"/' file.txt               # ダブルクォートで囲む
sed "s/.*/'&'/" file.txt               # シングルクォートで囲む

# CSV の特定列を加工
sed -E 's/^([^,]*),([^,]*),/\1,"\2",/' data.csv  # 2列目をクォート

# 箇条書きに変換
sed 's/^/- /' file.txt                 # 各行の先頭に "- " を追加
sed 's/^/  * /' file.txt               # 各行の先頭に "  * " を追加

# Markdown のリスト変換
sed 's/^[0-9]*\. /- /' file.md         # 番号リストを箇条書きに
sed 's/^- /1. /' file.md               # 箇条書きを番号リストに

# 重複行の削除（sort | uniq の sed 版）
sed '$!N; /^\(.*\)\n\1$/!P; D' file.txt  # 連続する重複行を削除

# ファイルの連結（区切り付き）
sed -e '$a\---' file1.txt file2.txt file3.txt  # 各ファイルの末尾に --- を追加

# 空行の挿入（各行の後に空行）
sed 'G' file.txt                       # 各行の後に空行を挿入
sed 'G;G' file.txt                     # 各行の後に2つの空行

# N行ごとに空行を挿入
sed '0~5 a\\' file.txt                 # 5行ごとに空行（GNU sed）
```

### 8.6 データ変換パターン

```bash
# JSON から CSV への簡易変換
sed -n 's/.*"name": "\(.*\)".*/\1/p' data.json
# → JSON 内の name フィールドの値を抽出

# key=value 形式の加工
sed 's/\(.*\)=\(.*\)/export \1="\2"/' file.env  # .env をexport文に変換
sed 's/\(.*\)=\(.*\)/\1: \2/' file.env           # YAML 形式に変換

# SQL の生成
sed "s/.*/INSERT INTO users (name) VALUES ('&');/" names.txt
# → 各行から INSERT 文を生成

# シェルコマンドの生成
sed 's|.*|cp & /backup/&|' filelist.txt   # コピーコマンドを生成
sed 's|.*|rm "&"|' filelist.txt            # 削除コマンドを生成（確認用）

# ホスト名からURLを生成
sed 's|.*|https://&/api/health|' hosts.txt
```

---

## 9. 高度なテクニック

### 9.1 複数行の処理

```bash
# N コマンド: 次の行をパターンスペースに追加（\n で連結）
# P コマンド: パターンスペースの最初の行を出力
# D コマンド: パターンスペースの最初の行を削除

# 2行をまとめて1行にする
sed 'N;s/\n/ /' file.txt

# 特定パターンの次の行を変更
sed '/^HEADER/{n;s/.*/MODIFIED/;}' file.txt
# → HEADER の次の行を MODIFIED に変更

# パターン間の行を削除
sed '/START/,/END/{/START/!{/END/!d}}' file.txt
# → START と END の行は残して、間の行を削除

# 複数行のパターンマッチ
sed -n '/BEGIN/{:a;N;/END/!ba;p}' file.txt
# → BEGIN〜END のブロックを表示

# 連続する重複行の削除
sed '$!N; /^\(.*\)\n\1$/!P; D' file.txt
```

### 9.2 ブランチとラベル

```bash
# sed にはブランチ（条件分岐）機能がある
# :label  ラベルの定義
# b label ラベルにジャンプ
# t label 直前の s コマンドが成功したらジャンプ

# 最初のマッチのみ置換（b を使った方法）
sed '/pattern/{s/pattern/replacement/;b};' file.txt

# 置換が成功するまでループ
sed ':loop; s/  / /; t loop' file.txt
# → 連続スペースを1つのスペースに（再帰的に）

# 条件付き処理
sed '/^#/{s/#//;b end}; s/^/> /; :end' file.txt
# → コメント行は # を除去、それ以外は先頭に > を追加

# 全ての連続空白を1つのスペースにする
sed -E ':a;s/  / /;ta' file.txt
```

### 9.3 読み込みと書き込み（r / w）

```bash
# r コマンド: ファイルの内容を読み込んで挿入
sed '/INSERT_HERE/r header.txt' file.txt
# → INSERT_HERE の後に header.txt の内容を挿入

# w コマンド: マッチした行をファイルに書き出す
sed -n '/error/w errors.log' file.txt
# → error を含む行を errors.log に書き出す

sed -n '/WARN/w warnings.log; /ERROR/w errors.log' file.txt
# → レベル別にログを振り分け

# 条件付きファイル分割
sed -n '/^[A-M]/w am.txt; /^[N-Z]/w nz.txt' names.txt
# → A-M で始まる行と N-Z で始まる行を分割
```

---

## 10. ワンライナー集（実務頻出）

### 10.1 テキスト整形

```bash
# 各行の先頭と末尾の空白を除去
sed 's/^[[:space:]]*//;s/[[:space:]]*$//' file.txt

# 連続する空白を1つのスペースに
sed -E 's/[[:space:]]+/ /g' file.txt

# 空行を全て削除
sed '/^$/d' file.txt

# 連続する空行を1つにまとめる
sed '/^$/N;/^\n$/d' file.txt

# DOS 改行を Unix 改行に変換
sed 's/\r$//' file.txt

# 各行の末尾にカンマを追加（最終行を除く）
sed '$!s/$/, /' file.txt

# ファイルの先頭に行を追加
sed '1i\# This file is auto-generated' file.txt

# ファイルの末尾に行を追加
sed '$a\# End of file' file.txt
```

### 10.2 データ加工

```bash
# CSV の特定列を抽出（簡易版 - 引用符内カンマ非対応）
sed -E 's/^([^,]*),([^,]*),(.*)$/\1/' data.csv  # 1列目
sed -E 's/^([^,]*),([^,]*),(.*)$/\2/' data.csv  # 2列目

# key=value から value のみ抽出
sed 's/^[^=]*=//' config.ini

# key=value の key のみ抽出
sed 's/=.*//' config.ini

# 特定の行の前後にテキストを挿入
sed '/MARKER/i\--- Before ---' file.txt
sed '/MARKER/a\--- After ---' file.txt

# 奇数行と偶数行をマージ
sed 'N;s/\n/,/' file.txt              # 改行をカンマに

# XMLタグの値を抽出（簡易版）
sed -n 's/.*<title>\(.*\)<\/title>.*/\1/p' file.xml

# メールアドレスのドメイン部分を抽出
sed -E 's/.*@//' emails.txt

# URLからドメインを抽出
sed -E 's|https?://([^/]+).*|\1|' urls.txt
```

### 10.3 ファイル操作支援

```bash
# ファイルリネーム用コマンドの生成
ls *.jpg | sed 's/\(.*\)\.jpg/mv "\1.jpg" "\1.png"/'
# → mv "file1.jpg" "file1.png" のようなコマンドを生成
# 確認後に | sh で実行

# バッチ処理コマンドの生成
ls *.csv | sed 's/.*/python process.py "&"/'
# → python process.py "file.csv" を生成

# .gitignore の生成支援
find . -name "*.pyc" -printf "%h\n" | sort -u | sed 's|^\./||;s|$|/*.pyc|'
```

---

## 11. トラブルシューティング

### 11.1 よくある問題と対処法

```bash
# 問題: -i オプションが macOS と Linux で異なる
# 対処: バックアップ拡張子を常に指定する
sed -i.bak 's/old/new/g' file.txt    # 両方で動作
rm file.txt.bak                       # バックアップを削除

# 問題: 特殊文字（/, &, \）のエスケープ
# / は区切り文字を変更するか \/ でエスケープ
sed 's|/usr/bin|/opt/bin|g' file.txt  # 区切り文字を | に変更
sed 's/\/usr\/bin/\/opt\/bin/g' file.txt  # エスケープ（見づらい）

# & は置換文字列でマッチ全体を参照するため、リテラルには \& を使う
sed 's/AT/AT\&T/g' file.txt           # AT → AT&T

# \ は \\ でエスケープ
sed 's/\\/\//g' file.txt              # バックスラッシュをスラッシュに

# 問題: 変数を含む sed コマンド
# 対処: ダブルクォートを使用し、特殊文字をエスケープ
NEW_VALUE="production"
sed -i "s/environment=.*/environment=$NEW_VALUE/" config.ini

# 変数に / が含まれる場合
NEW_PATH="/usr/local/bin"
sed -i "s|old_path|$NEW_PATH|g" config.ini    # 区切り文字を | に

# 問題: 改行を含む置換
# GNU sed
sed -i 's/pattern/line1\nline2/' file.txt
# BSD sed（macOS）
sed -i '' $'s/pattern/line1\\\nline2/' file.txt

# 問題: 置換されない（エスケープ忘れ）
# 正規表現のメタ文字をリテラルとして使う場合はエスケープが必要
sed 's/file\.txt/file.log/' file.txt   # . をリテラルとして
sed 's/\[error\]/[warning]/' file.txt  # [] をリテラルとして
sed 's/\$HOME/\/home\/user/' file.txt  # $ をリテラルとして
```

---

## まとめ

| コマンド | 用途 | 例 |
|---------|------|------|
| s/old/new/g | 置換 | sed 's/foo/bar/g' |
| /pattern/d | 行の削除 | sed '/^#/d' |
| -n 'Np' | 特定行の表示 | sed -n '10,20p' |
| -i | ファイル直接書き換え | sed -i 's/old/new/g' |
| Ni\text | N行目の前に挿入 | sed '1i\header' |
| Na\text | N行目の後に追加 | sed '$a\footer' |
| /pat/,/pat/ | パターン範囲 | sed '/BEGIN/,/END/d' |
| -E | 拡張正規表現 | sed -E 's/(a|b)/c/g' |
| -e | 複数コマンド | sed -e 's/a/b/' -e 's/c/d/' |
| -f | スクリプトファイル | sed -f script.sed |
| \1, \2 | 後方参照 | sed -E 's/(.*)/[\1]/' |
| & | マッチ全体の参照 | sed 's/word/[&]/' |
| \U, \L | 大文字/小文字変換 | sed 's/.*/\U&/' |

---

## 13. sed と他ツールの連携パターン

### 13.1 grep + sed のパイプライン

```bash
# grep で対象を絞り、sed で変換する
# ログファイルからエラー行を抽出し、タイムスタンプを整形
grep "ERROR" /var/log/app.log | sed -E 's/^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})/\1\/\2\/\3 \4:\5:\6/'

# 設定ファイルからコメントでない行を抽出し、値を変換
grep -v '^#' config.ini | sed 's/=/ → /'

# 特定パターンの行だけを変換
grep -n "TODO" *.py | sed -E 's/^([^:]+):([0-9]+):/File: \1, Line: \2 → /'

# grep の結果をCSV形式に変換
grep -rn "FIXME\|TODO\|HACK" src/ | sed -E 's/^([^:]+):([0-9]+):(.*)/"\1",\2,"\3"/'

# アクセスログからステータスコード別にカウント用データを生成
grep "HTTP/1.1" access.log | sed -E 's/.*" ([0-9]{3}) .*/\1/' | sort | uniq -c | sort -rn
```

### 13.2 find + sed の組み合わせ

```bash
# 全 Python ファイルのインポートパスを一括変更
find . -name "*.py" -exec sed -i 's/from old_module/from new_module/g' {} +

# HTML ファイルのエンコーディング宣言を一括変更
find . -name "*.html" -exec sed -i 's/charset=EUC-JP/charset=UTF-8/g' {} +

# バックアップファイルを作成しながら設定を変更
find /etc/nginx/sites-available/ -name "*.conf" \
  -exec sed -i.bak 's/listen 80/listen 8080/g' {} \;

# 変更されたファイルのリストを表示
find . -name "*.bak" -newer /tmp/timestamp -exec echo "Modified: {}" \;

# 特定サイズ以上のログファイルからセンシティブ情報を除去
find /var/log -name "*.log" -size +1M \
  -exec sed -i -E 's/[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}/XXXX-XXXX-XXXX-XXXX/g' {} +

# Git 管理下のファイルのみ対象に変更
git ls-files '*.js' | xargs sed -i 's/console\.log/logger.debug/g'
```

### 13.3 sed + awk の役割分担

```bash
# sed で前処理、awk で集計
# ログファイルからタイムスタンプを正規化した後、時間帯別に集計
sed -E 's/^.*\[([0-9]{2})\/[A-Z][a-z]+\/[0-9]{4}:([0-9]{2}):.*/\1 \2/' access.log \
  | awk '{count[$2]++} END {for (h in count) print h, count[h]}' | sort -n

# CSV の前処理（sed）+ 計算（awk）
sed 's/"//g; s/,/ /g' data.csv | awk '{sum += $3} END {print "Total:", sum}'

# sed でデータクレンジング、awk でレポート生成
sed -E '/^$/d; s/[[:space:]]+/ /g; s/^ //; s/ $//' raw_data.txt \
  | awk -F' ' '{
      category[$1] += $2
      count[$1]++
    }
    END {
      for (c in category)
        printf "%-20s avg=%.2f total=%d count=%d\n", c, category[c]/count[c], category[c], count[c]
    }'
```

### 13.4 xargs + sed のバッチ処理

```bash
# ファイルリストを受け取って一括処理
cat file_list.txt | xargs -I{} sed -i 's/old_api/new_api/g' {}

# 並列処理で高速化
cat file_list.txt | xargs -P4 -I{} sed -i 's/old_api/new_api/g' {}

# grep で対象ファイルを特定し、sed で一括変更
grep -rl "deprecated_function" src/ | xargs sed -i 's/deprecated_function/new_function/g'

# 変更前に確認（dry-run 的な使い方）
grep -rl "deprecated_function" src/ | xargs -I{} sh -c 'echo "=== {} ==="; sed -n "s/deprecated_function/new_function/gp" {}'
```

---

## 14. sed のパフォーマンスチューニング

### 14.1 高速化テクニック

```bash
# 不要な処理をスキップする
# アドレスを使って対象行を絞り込む（全行スキャンを避ける）
sed '1,100s/old/new/g' huge_file.txt          # 最初の100行のみ処理
sed '/pattern/s/old/new/g' huge_file.txt       # パターンにマッチする行のみ

# -n + p でマッチ行のみ出力（全行出力を抑制）
sed -n '/ERROR/p' huge.log                     # grep より遅いが変換を同時に行える

# 早期終了（q コマンド）
sed '100q' huge_file.txt                       # 100行目で終了（head -100 相当）
sed '/FOUND/{ p; q; }' huge_file.txt           # 最初のマッチで終了

# 複数の -e より -f（スクリプトファイル）の方が効率的
# 多数の置換ルールがある場合
cat > rules.sed << 'RULES'
s/foo/bar/g
s/baz/qux/g
s/old/new/g
RULES
sed -f rules.sed input.txt

# 正規表現の最適化
# .*? のような欲張りマッチを避ける
sed 's/[^,]*/REPLACED/' file.csv               # [^,]* は .* より高速
sed 's/[[:digit:]]\{3\}/XXX/' file.txt          # \d より [[:digit:]] が明示的

# GNU sed の --unbuffered オプション（リアルタイム出力）
tail -f app.log | sed --unbuffered 's/ERROR/*** ERROR ***/'
```

### 14.2 大量ファイルの処理戦略

```bash
# ファイルを分割して並列処理
split -l 100000 huge_file.txt chunk_
for f in chunk_*; do
  sed -i 's/old/new/g' "$f" &
done
wait
cat chunk_* > result.txt
rm chunk_*

# GNU parallel を活用
parallel --pipe sed 's/old/new/g' < huge_file.txt > result.txt

# メモリ使用量の考慮
# sed は1行ずつ処理するためメモリ効率が良い
# ただし、N コマンドや H コマンドで複数行をバッファする場合は注意
# 巨大ファイルのホールドスペースに全行を蓄積しないこと

# tmpfile を使った安全な置換（-i の代わり）
sed 's/old/new/g' input.txt > tmp_output.txt && mv tmp_output.txt input.txt

# 変更があったかどうかの確認
if sed 's/old/new/g' input.txt | diff -q input.txt - > /dev/null 2>&1; then
  echo "No changes needed"
else
  sed -i 's/old/new/g' input.txt
  echo "File updated"
fi
```

---

## 15. sed のセキュリティとベストプラクティス

### 15.1 セキュリティ上の注意点

```bash
# 入力のサニタイズ（ユーザー入力を sed に渡す場合）
# 危険: ユーザー入力をそのまま sed パターンに使用
user_input="malicious/e touch /tmp/pwned"
sed "s/$user_input/replacement/" file.txt      # 危険！

# 安全: 特殊文字をエスケープ
sanitized=$(printf '%s\n' "$user_input" | sed 's/[&/\]/\\&/g')
sed "s/$sanitized/replacement/" file.txt

# より安全な方法: 変数を sed で使う場合
search="user.input"
replace="safe.output"
sed "s/$(printf '%s' "$search" | sed 's/[.[\*^$/]/\\&/g')/$(printf '%s' "$replace" | sed 's/[&/\]/\\&/g')/g" file.txt

# パスワードやトークンのマスキング
sed -E 's/(password|token|secret|api_key)=.*/\1=***REDACTED***/gi' config.txt
sed -E 's/Bearer [A-Za-z0-9+\/=]+/Bearer ***REDACTED***/g' api.log

# メールアドレスのマスキング
sed -E 's/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/***@***.***/' data.txt

# IP アドレスの匿名化
sed -E 's/([0-9]{1,3}\.)[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/\1xxx.xxx.xxx/g' access.log
```

### 15.2 ベストプラクティス

```bash
# 1. 常にバックアップを取る
sed -i.bak 's/old/new/g' important_file.txt
# 処理後にバックアップを確認
diff important_file.txt important_file.txt.bak

# 2. 変更前にプレビューする
sed 's/old/new/g' file.txt | head -20           # 最初の20行を確認
sed -n 's/old/new/gp' file.txt                   # 変更のあった行のみ表示
diff <(cat file.txt) <(sed 's/old/new/g' file.txt) # 差分を確認

# 3. 段階的に処理する
# 複雑な変換は一度に行わず、パイプで段階的に
cat input.txt \
  | sed 's/step1/result1/g' \
  | sed 's/step2/result2/g' \
  | sed 's/step3/result3/g' \
  > output.txt

# 4. コメント付きスクリプトファイルを使う
cat > transform.sed << 'EOF'
# ヘッダーの正規化
1,5s/OLD_HEADER/NEW_HEADER/

# 空行の除去
/^[[:space:]]*$/d

# コメント行の統一（// → #）
s|//\(.*\)|#\1|

# 末尾の空白を除去
s/[[:space:]]*$//
EOF
sed -f transform.sed input.txt > output.txt

# 5. エラーハンドリング
if ! sed -i 's/old/new/g' file.txt 2>/dev/null; then
  echo "Error: sed command failed" >&2
  exit 1
fi

# 6. ポータブルな書き方を心がける
# macOS / Linux 両対応
if sed --version 2>/dev/null | grep -q 'GNU'; then
  SED_I="sed -i"
else
  SED_I="sed -i ''"
fi
eval "$SED_I 's/old/new/g' file.txt"
```

---

## 16. 実務シナリオ別の総合的な sed レシピ

### 16.1 Webアプリケーション開発での sed 活用

```bash
# HTML テンプレートのプレースホルダー置換
sed -e "s/{{APP_NAME}}/$APP_NAME/g" \
    -e "s/{{VERSION}}/$VERSION/g" \
    -e "s/{{BUILD_DATE}}/$(date +%Y-%m-%d)/g" \
    template.html > index.html

# CSS の minify（簡易版）
sed -E '
  s/\/\*[^*]*\*\///g          # コメント除去
  s/[[:space:]]+/ /g          # 連続空白を1つに
  s/ *([{};:,]) */\1/g       # セレクタ周りの空白除去
  /^$/d                       # 空行除去
' style.css > style.min.css

# JavaScript のデバッグコードを本番用に除去
sed -E '/console\.(log|debug|warn|info|trace)\(/d; /debugger;/d' app.js > app.prod.js

# 環境変数を設定ファイルに展開
sed -E "
  s|\\\$\{DB_HOST\}|${DB_HOST:-localhost}|g
  s|\\\$\{DB_PORT\}|${DB_PORT:-5432}|g
  s|\\\$\{DB_NAME\}|${DB_NAME:-myapp}|g
  s|\\\$\{DB_USER\}|${DB_USER:-admin}|g
" config.template > config.production

# URL のプロトコル一括変更（http → https）
sed -E 's|http://([^"'"'"'[:space:]]+)|https://\1|g' page.html > page_secure.html
```

### 16.2 サーバー運用での sed 活用

```bash
# nginx 設定のポート番号一括変更
sed -E 's/listen[[:space:]]+80;/listen 8080;/g; s/listen[[:space:]]+443/listen 8443/g' \
  /etc/nginx/sites-available/default > /tmp/nginx_new.conf

# Apache の .htaccess 生成
sed -n '
  /^RewriteRule/p
  /^RewriteCond/p
' .htaccess.template | sed '
  s/DOMAIN_NAME/example.com/g
  s/DOC_ROOT/\/var\/www\/html/g
' > .htaccess

# crontab エントリの時刻一括変更
crontab -l | sed -E 's/^([0-9]+) ([0-9]+)/\1 3/' | crontab -

# syslog の整形とフィルタリング
sed -E '
  /^$/d                                  # 空行除去
  s/^([A-Z][a-z]{2} +[0-9]+ [0-9:]+) ([^ ]+) ([^:]+): (.*)/\1 | \2 | \3 | \4/
' /var/log/syslog | tail -50

# SSL 証明書情報の抽出
openssl x509 -in cert.pem -text | sed -n '
  /Subject:/p
  /Issuer:/p
  /Not Before/p
  /Not After/p
  /DNS:/p
'

# /etc/hosts の管理
# 特定のドメインを追加
sed -i '/^# Custom entries/a\192.168.1.100 myapp.local' /etc/hosts

# 一時的にエントリをコメントアウト
sed -i 's/^\(192\.168\.1\.100.*\)/#\1/' /etc/hosts

# コメントアウトを解除
sed -i 's/^#\(192\.168\.1\.100.*\)/\1/' /etc/hosts
```

### 16.3 データ変換とETL処理

```bash
# JSON Lines を CSV に変換（簡易版、jq 不要の場合）
sed -E '
  s/^\{//
  s/\}$//
  s/"[^"]+": *//g
  s/,/\t/g
  s/"//g
' data.jsonl > data.tsv

# 固定長レコードを CSV に変換
# Name(20) Age(3) City(15)
sed -E 's/^(.{20})(.{3})(.{15})$/"\1","\2","\3"/' fixed_width.txt \
  | sed 's/  *"/"/g'  # フィールド内の末尾空白を除去

# XML タグの変換
sed -E '
  s/<([a-z_]+)>([^<]*)<\/\1>/\1=\2/g    # 単純なタグをキー=値に
  s/<[^>]+>//g                            # 残りのタグを除去
  /^[[:space:]]*$/d                       # 空行除去
' data.xml

# 日付フォーマットの変換
# MM/DD/YYYY → YYYY-MM-DD
sed -E 's|([0-9]{2})/([0-9]{2})/([0-9]{4})|\3-\1-\2|g' dates.txt

# YYYY年MM月DD日 → YYYY-MM-DD
sed -E 's/([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日/\1-\2-\3/g' japanese_dates.txt

# 電話番号フォーマットの統一
sed -E 's/0([0-9]{1,4})-([0-9]{1,4})-([0-9]{4})/0\1\2\3/g' phones.txt  # ハイフン除去
sed -E 's/^0([0-9]{2})([0-9]{4})([0-9]{4})$/0\1-\2-\3/' phones.txt      # ハイフン追加

# 文字コードに関する前処理
# BOM（Byte Order Mark）の除去
sed -i '1s/^\xEF\xBB\xBF//' utf8_with_bom.txt
# Windows の改行コード（CRLF → LF）
sed -i 's/\r$//' windows_file.txt
# DOS ファイルの ^M 除去
sed -i 's/\r//g' dos_file.txt
```

### 16.4 CI/CD パイプラインでの sed 活用

```bash
# バージョン番号の自動更新
# package.json のバージョンを更新
sed -i -E 's/"version": "[0-9]+\.[0-9]+\.[0-9]+"/"version": "'"$NEW_VERSION"'"/' package.json

# CHANGELOG にエントリを追加
sed -i "/^## \[Unreleased\]/a\\
\\
## [$NEW_VERSION] - $(date +%Y-%m-%d)\\
### Changed\\
- $CHANGE_DESCRIPTION" CHANGELOG.md

# Dockerfile の FROM イメージタグを更新
sed -i "s|^FROM node:.*|FROM node:${NODE_VERSION}-alpine|" Dockerfile

# Kubernetes マニフェストのイメージタグを更新
sed -i "s|image: myregistry/myapp:.*|image: myregistry/myapp:${GIT_SHA}|" k8s/deployment.yaml

# テスト結果の整形
sed -E '
  s/PASS/✅ PASS/g
  s/FAIL/❌ FAIL/g
  s/SKIP/⏭️  SKIP/g
  s/([0-9]+) passing/\1 tests passing/
  s/([0-9]+) failing/\1 tests FAILING/
' test_results.txt

# ビルド情報の埋め込み
sed -e "s/@GIT_COMMIT@/$(git rev-parse --short HEAD)/" \
    -e "s/@BUILD_TIME@/$(date -u +%Y-%m-%dT%H:%M:%SZ)/" \
    -e "s/@BRANCH@/$(git branch --show-current)/" \
    version.template > version.txt
```

---

## 次に読むべきガイド
→ [[03-awk.md]] — テキスト処理言語

---

## 参考文献
1. Robbins, A. "sed & awk." 2nd Ed, O'Reilly, 1997.
2. Barrett, D. "Efficient Linux at the Command Line." Ch.5, O'Reilly, 2022.
3. GNU sed Manual. https://www.gnu.org/software/sed/manual/
4. Grymoire sed Tutorial. https://www.grymoire.com/Unix/Sed.html
5. sed One-Liners Explained. https://catonmat.net/sed-one-liners-explained-part-one
