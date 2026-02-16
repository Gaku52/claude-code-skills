# ソート・集計（sort, uniq, cut, wc）

> これらのコマンドはパイプラインの「部品」として組み合わせて使う。
> テキスト処理の基盤であり、ログ分析・データ集計・レポート生成に不可欠な道具立てである。

## この章で学ぶこと

- [ ] sort の全オプションを使いこなしてテキストをソートできる
- [ ] uniq で重複排除・カウント・フィルタリングができる
- [ ] cut で列切り出し・フィールド抽出ができる
- [ ] wc で行数・単語数・バイト数・文字数をカウントできる
- [ ] tr で文字変換・削除・圧縮ができる
- [ ] paste, join, comm で複数ファイルの結合・比較ができる
- [ ] これらを組み合わせた実務パイプラインを構築できる

---

## 1. sort — テキストのソート

### 1.1 基本的なソート

```bash
# アルファベット順（辞書順）ソート
sort file.txt

# 数値順ソート（-n）
sort -n numbers.txt
# 例: "1", "2", "10" が正しく 1, 2, 10 の順になる
# -n がないと "1", "10", "2" になる（辞書順）

# 逆順ソート（-r）
sort -r file.txt                 # アルファベット逆順
sort -rn numbers.txt             # 数値の大きい順

# 重複排除付きソート（-u）
sort -u file.txt                 # ソート + 重複行を削除

# 安定ソート（--stable）
sort --stable file.txt
# 同じキーを持つ行の元の順序を保持する

# 大文字小文字を区別しないソート（-f / --ignore-case）
sort -f file.txt

# 先頭の空白を無視してソート（-b）
sort -b file.txt
```

### 1.2 キー指定ソート（-k オプション）

```bash
# 基本構文: -k FIELD[.CHAR][,FIELD[.CHAR]][OPTS]

# 2列目でソート（スペース/タブ区切り）
sort -k2 file.txt

# 2列目を数値ソート
sort -k2,2n file.txt
# -k2,2n の意味: 2列目から2列目の範囲を、数値として比較

# 3列目を逆順で数値ソート
sort -k3,3rn file.txt

# 複数キー指定（第1キー: 2列目の数値順、第2キー: 1列目の辞書順）
sort -k2,2n -k1,1 file.txt

# 2列目の3文字目からソート
sort -k2.3 file.txt

# 実例: /etc/passwd をUID順でソート
sort -t':' -k3,3n /etc/passwd

# 実例: ls -l の出力をファイルサイズ順でソート
ls -l | sort -k5,5n

# 実例: CSV を2列目でソートし、同値なら3列目の降順
sort -t',' -k2,2 -k3,3rn data.csv
```

### 1.3 区切り文字の指定（-t オプション）

```bash
# デフォルト区切り: 空白文字（スペースまたはタブ）

# CSV（カンマ区切り）
sort -t',' -k3 data.csv           # 3列目でソート
sort -t',' -k2,2n data.csv        # 2列目を数値ソート

# TSV（タブ区切り）
sort -t$'\t' -k2 data.tsv

# コロン区切り（/etc/passwd 形式）
sort -t':' -k3,3n /etc/passwd     # UID順

# パイプ区切り
sort -t'|' -k2 data.txt

# セミコロン区切り
sort -t';' -k3 log.csv
```

### 1.4 特殊なソート

```bash
# 人間可読サイズ順（-h）
# 1K, 5M, 2G のような表記をソート
du -sh /* 2>/dev/null | sort -h
du -sh /var/log/* 2>/dev/null | sort -rh | head -10

# バージョン番号順（-V）
echo -e "1.2.3\n1.10.1\n1.2.10\n1.1.0" | sort -V
# 結果: 1.1.0, 1.2.3, 1.2.10, 1.10.1

# 月名でソート（-M）
echo -e "Mar\nJan\nDec\nFeb" | sort -M
# 結果: Jan, Feb, Mar, Dec

# ランダム順（-R）
sort -R file.txt                  # ランダムシャッフル
shuf file.txt                     # こちらも同等（GNU coreutils）

# ゼロ区切り（-z / --zero-terminated）
find . -name "*.txt" -print0 | sort -z | xargs -0 ls -la
```

### 1.5 パフォーマンスと大規模ファイル

```bash
# テンポラリディレクトリの指定（-T）
sort -T /tmp/sort_workspace large_file.txt
# デフォルトの /tmp に十分な空きがない場合に有用

# 並列ソート（--parallel=N）
sort --parallel=4 large_file.txt
# CPUコアを活用して高速化

# メモリバッファサイズの指定（-S）
sort -S 2G large_file.txt
# 2GBまでメモリを使用（大規模ファイルで高速化）

# ソート済みファイルのマージ（-m）
sort -m sorted1.txt sorted2.txt sorted3.txt
# 既にソート済みのファイルを効率的にマージ

# ソート済みか確認（-c / -C）
sort -c file.txt                  # ソート済みでなければエラー出力
sort -C file.txt                  # ソート済みでなければ終了コード1（出力なし）
if sort -C file.txt; then
    echo "ファイルはソート済み"
else
    echo "ファイルはソートされていない"
fi

# 出力ファイル指定（-o）
sort -o sorted.txt file.txt       # 入力と同じファイル名でも安全
# sort file.txt > file.txt は内容が消えるので注意！
```

### 1.6 sort の実践的なオプション組み合わせ

```bash
# du の結果を人間可読サイズの大きい順に並べて上位10件
du -sh /var/log/* 2>/dev/null | sort -rh | head -10

# アクセスログのレスポンスコード別集計（ソート部分）
awk '{print $9}' access.log | sort | uniq -c | sort -rn

# /etc/passwd のシェル別ユーザー数
awk -F: '{print $7}' /etc/passwd | sort | uniq -c | sort -rn

# プロセスのメモリ使用量上位（ps + sort）
ps aux --sort=-%mem | head -20

# CSV の複数キーソート（第1キー: 部門、第2キー: 売上降順）
sort -t',' -k1,1 -k3,3rn sales.csv

# IP アドレスのソート（バージョンソートを活用）
sort -t. -k1,1n -k2,2n -k3,3n -k4,4n ip_list.txt
# または
sort -V ip_list.txt

# 日付でソート（YYYY-MM-DD 形式が先頭の場合）
sort -k1,1 dated_log.txt

# タイムスタンプ付きログのソート
sort -t' ' -k1,1 -k2,2 timestamped.log
```

---

## 2. uniq — 重複行の処理

### 2.1 基本操作

```bash
# 重要: uniq は「連続する」重複行のみ処理する
# → 必ず sort と組み合わせて使う

# 基本的な重複排除
sort file.txt | uniq

# 重複のカウント（-c）
sort file.txt | uniq -c
# 出力例:
#       3 apple
#       1 banana
#       5 cherry

# 重複行のみ表示（-d）
sort file.txt | uniq -d

# 重複のない行（ユニークな行）のみ表示（-u）
sort file.txt | uniq -u

# すべての重複行を表示（-D）
sort file.txt | uniq -D
# -d は重複行の代表1行、-D は重複行すべてを表示
```

### 2.2 比較対象のカスタマイズ

```bash
# 先頭N個のフィールドを無視（-f N）
sort file.txt | uniq -f 1        # 最初の1フィールドを無視して比較
# フィールドはスペース/タブで区切られる

# 先頭N文字を無視（-s N）
sort file.txt | uniq -s 5        # 最初の5文字を無視して比較

# 先頭N文字のみで比較（-w N）
sort file.txt | uniq -w 10       # 最初の10文字のみで比較

# 大文字小文字を区別しない（-i）
sort -f file.txt | uniq -i

# 組み合わせ例: 先頭のタイムスタンプを無視して重複行を検出
sort -k2 log.txt | uniq -f 1 -c | sort -rn
```

### 2.3 定番パターン: 出現頻度ランキング

```bash
# 出現頻度の高い順に並べる（最も使用頻度の高いパターン）
sort file.txt | uniq -c | sort -rn

# 実用例: アクセスログのIPアドレス頻度ランキング
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -20

# 実用例: エラーメッセージの頻度ランキング
grep "ERROR" app.log | awk -F'ERROR' '{print $2}' | sort | uniq -c | sort -rn | head -10

# 実用例: コマンド履歴の使用頻度
history | awk '{print $2}' | sort | uniq -c | sort -rn | head -20

# 実用例: HTTP ステータスコードの集計
awk '{print $9}' access.log | sort | uniq -c | sort -rn
# 出力例:
#   45231 200
#    3421 301
#    1205 404
#     342 500

# 実用例: ファイル拡張子の集計
find . -type f -name "*.*" | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -15

# 実用例: /etc/passwd のシェル種類と利用者数
awk -F: '{print $7}' /etc/passwd | sort | uniq -c | sort -rn

# 実用例: Git コミットの著者別集計
git log --format='%an' | sort | uniq -c | sort -rn

# 実用例: 接続元ポート番号の重複チェック
ss -tn | awk '{print $4}' | rev | cut -d: -f1 | rev | sort | uniq -c | sort -rn | head -10
```

---

## 3. cut — 列の切り出し

### 3.1 フィールド単位の切り出し（-f / -d）

```bash
# 基本構文: cut -d'区切り文字' -f'フィールド番号' ファイル

# CSV の2列目を取得
cut -d',' -f2 data.csv

# コロン区切りの1列目と3列目
cut -d':' -f1,3 /etc/passwd

# タブ区切り（デフォルト）の1-3列目
cut -f1-3 data.tsv

# 3列目以降をすべて取得
cut -d',' -f3- data.csv

# 4列目までを取得
cut -d',' -f-4 data.csv

# 2列目を除外して取得（--complement）
cut -d',' -f2 --complement data.csv

# パイプ区切り
echo "a|b|c|d" | cut -d'|' -f2,4
# 出力: b|d

# スペース区切り
echo "hello world foo bar" | cut -d' ' -f2
# 出力: world

# 出力区切り文字を変更（--output-delimiter）
cut -d':' -f1,3,7 /etc/passwd --output-delimiter=','
# 出力例: root,0,/bin/bash
```

### 3.2 文字単位の切り出し（-c）

```bash
# 1〜10文字目を取得
cut -c1-10 file.txt

# 5文字目のみ
cut -c5 file.txt

# 1文字目と5文字目と10文字目
cut -c1,5,10 file.txt

# 20文字目以降
cut -c20- file.txt

# 最初の50文字（長い行を切り詰め）
cut -c-50 file.txt

# 実用例: 固定長フォーマットの解析
# 銀行の全銀データ形式など
cut -c1-2 fixedwidth.dat       # レコード区分
cut -c3-6 fixedwidth.dat       # 銀行コード
cut -c7-9 fixedwidth.dat       # 支店コード
```

### 3.3 バイト単位の切り出し（-b）

```bash
# バイト単位（マルチバイト文字環境で注意が必要）
cut -b1-10 file.txt

# 文字単位とバイト単位の違い
echo "日本語テスト" | cut -c1-3   # 「日本語」（文字単位）
echo "日本語テスト" | cut -b1-9   # 「日本語」（UTF-8で1文字3バイト）
```

### 3.4 cut の限界と代替手段

```bash
# cut の限界:
# 1. 連続する区切り文字を1つとして扱えない
# 2. 正規表現が使えない
# 3. 複雑なフィールド処理ができない

# 連続スペースの問題
echo "a  b  c" | cut -d' ' -f2
# 出力: ""（空文字）← 2つ目のスペースまでを1フィールドと見なすため

# 解決策1: tr -s で連続スペースを圧縮
echo "a  b  c" | tr -s ' ' | cut -d' ' -f2
# 出力: b

# 解決策2: awk を使う（推奨）
echo "a  b  c" | awk '{print $2}'
# 出力: b

# 解決策3: read を使う
echo "a  b  c" | while read a b c; do echo "$b"; done
# 出力: b

# 複雑なフィールド処理が必要な場合は awk を使う
# 例: 最終フィールドの取得
echo "a,b,c,d" | awk -F',' '{print $NF}'
# 出力: d（cut では事前にフィールド数を知る必要がある）
```

---

## 4. wc — カウント

### 4.1 基本カウント

```bash
# 行数（-l）
wc -l file.txt
# 出力例: 42 file.txt

# 単語数（-w）
wc -w file.txt
# 出力例: 350 file.txt

# バイト数（-c）
wc -c file.txt
# 出力例: 2048 file.txt

# 文字数（-m）
wc -m file.txt
# マルチバイト文字環境では -c と -m の結果が異なる

# 最長行の長さ（-L）
wc -L file.txt
# 出力例: 120 file.txt

# 全部表示
wc file.txt
# 出力例:  42  350 2048 file.txt
#          行   語  バイト ファイル名

# 複数ファイル
wc -l *.txt
# 各ファイルの行数 + 合計行が表示される
```

### 4.2 パイプとの組み合わせ

```bash
# ファイル数のカウント
find . -name "*.py" | wc -l
find . -type f | wc -l

# プロセス数のカウント
ps aux | grep nginx | grep -v grep | wc -l
# pgrep の方がスマート
pgrep -c nginx

# Git の変更ファイル数
git diff --name-only | wc -l
git status --porcelain | wc -l

# ログのエラー行数
grep -c "ERROR" app.log            # grep -c でも同じ
grep "ERROR" app.log | wc -l      # パイプ経由でも同じ結果

# ディレクトリ内のファイル数
ls -1 /var/log/ | wc -l

# 空行のカウント
grep -c "^$" file.txt

# 非空行のカウント
grep -c "." file.txt

# コードの行数（空行とコメントを除く）
grep -v "^$" code.py | grep -v "^#" | wc -l

# 実用例: ソースコードの行数統計
find . -name "*.py" -exec wc -l {} + | tail -1
# 合計行数のみ表示

# 実用例: 各ファイルの行数を大きい順に
find . -name "*.py" -exec wc -l {} + | sort -rn | head -20
```

### 4.3 高度なカウントパターン

```bash
# 特定文字の出現回数
tr -cd ',' < data.csv | wc -c
# CSV のカンマの数を数える

# 各行のフィールド数チェック（CSV の整合性確認）
awk -F',' '{print NF}' data.csv | sort | uniq -c
# 全行のフィールド数が同じなら正常

# 単語出現回数のカウント
tr -s '[:space:]' '\n' < file.txt | grep -cw "error"
# "error" という単語の出現回数

# ユニークな行の数
sort -u file.txt | wc -l

# 重複している行の数
sort file.txt | uniq -d | wc -l

# バイナリファイル内のNULLバイト数
tr -cd '\0' < binary_file | wc -c
```

---

## 5. tr — 文字変換・削除

### 5.1 文字の変換

```bash
# 小文字 → 大文字
echo "hello world" | tr 'a-z' 'A-Z'
# 出力: HELLO WORLD

# 大文字 → 小文字
echo "HELLO WORLD" | tr 'A-Z' 'a-z'
# 出力: hello world

# POSIX 文字クラスを使用
echo "Hello World" | tr '[:upper:]' '[:lower:]'
echo "hello world" | tr '[:lower:]' '[:upper:]'

# 特定文字の置換
echo "2024-01-15" | tr '-' '/'
# 出力: 2024/01/15

echo "a:b:c" | tr ':' '\n'
# 出力:
# a
# b
# c

# タブをスペースに変換
tr '\t' ' ' < file.txt

# 数字をアスタリスクに置換
echo "Password: 12345" | tr '0-9' '*'
# 出力: Password: *****

# ROT13 暗号化（簡易）
echo "Hello World" | tr 'A-Za-z' 'N-ZA-Mn-za-m'
# 出力: Uryyb Jbeyq（もう一度同じ変換で元に戻る）
```

### 5.2 文字の削除（-d）

```bash
# 改行コード（CR）の削除（Windows → Unix 変換）
tr -d '\r' < windows.txt > unix.txt

# 数字の削除
echo "abc123def456" | tr -d '0-9'
# 出力: abcdef

# 空白の削除
echo "  h e l l o  " | tr -d ' '
# 出力: hello

# 特定文字の削除
echo "Hello, World!" | tr -d ',!'
# 出力: Hello World

# 英字以外を削除
echo "abc123!@#def" | tr -d '[:alpha:]'
# 出力: 123!@#

# 補集合で削除（-cd）: 指定文字「以外」を削除
echo "abc123!@#def" | tr -cd '[:alpha:]'
# 出力: abcdef

echo "phone: 03-1234-5678" | tr -cd '0-9'
# 出力: 0312345678

# 印刷可能文字以外を削除
tr -cd '[:print:]' < binary_mixed.txt
```

### 5.3 文字の圧縮（-s / --squeeze-repeats）

```bash
# 連続するスペースを1つに
echo "a   b    c" | tr -s ' '
# 出力: a b c

# 連続する改行を1つに（空行を削除）
tr -s '\n' < file.txt

# 連続するスペースとタブを1つのスペースに
tr -s '[:blank:]' ' ' < file.txt

# 変換 + 圧縮の組み合わせ
echo "  hello    world  " | tr -s ' '
# 出力: " hello world "

# 実用例: 空白で区切られたテーブルを正規化
ps aux | tr -s ' ' | cut -d' ' -f1-5
```

### 5.4 tr の実践パターン

```bash
# CSV を TSV に変換
tr ',' '\t' < data.csv > data.tsv

# TSV を CSV に変換
tr '\t' ',' < data.tsv > data.csv

# パスワード生成（ランダム文字列）
tr -dc 'A-Za-z0-9!@#$%' < /dev/urandom | head -c 20
echo  # 改行追加

# 英数字のみのランダム文字列
tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32
echo

# ファイルの各行をカンマ区切りの1行に変換
tr '\n' ',' < file.txt | sed 's/,$/\n/'

# テキストの正規化（前処理に便利）
cat raw_text.txt | tr 'A-Z' 'a-z' | tr -s '[:space:]' '\n' | tr -d '[:punct:]' | sort | uniq -c | sort -rn

# 制御文字の可視化
cat -v file.txt                   # ^M（CR）などが見える
# 制御文字の除去
tr -d '[:cntrl:]' < file.txt | tr -cd '[:print:]\n'
```

---

## 6. paste — ファイルの水平結合

### 6.1 基本操作

```bash
# 2つのファイルを横に結合（タブ区切り）
paste file1.txt file2.txt
# file1 の各行と file2 の各行がタブで結合される

# 区切り文字を指定
paste -d',' file1.txt file2.txt  # カンマ区切り
paste -d':' file1.txt file2.txt  # コロン区切り

# 3つ以上のファイルを結合
paste names.txt ages.txt cities.txt

# 入力を横に並べる（-s / --serial）
paste -s file.txt
# ファイルの全行をタブ区切りの1行にする

# 区切り文字を指定して1行に
paste -sd',' file.txt
# 出力例: line1,line2,line3,line4
```

### 6.2 paste の活用パターン

```bash
# 数値ファイルの合計（bc と組み合わせ）
paste -sd+ numbers.txt | bc
# 1+2+3+4+5 のように結合してから bc で計算

# CSV の特定列の合計
cut -d',' -f3 data.csv | tail -n +2 | paste -sd+ | bc

# 標準入力を N 列に整形
seq 12 | paste - - -
# 出力:
# 1     2     3
# 4     5     6
# 7     8     9
# 10    11    12

seq 12 | paste -d',' - - - -
# 出力:
# 1,2,3,4
# 5,6,7,8
# 9,10,11,12

# 2つのコマンドの出力を横に結合
paste <(seq 5) <(seq 5 | awk '{print $1*$1}')
# 出力:
# 1     1
# 2     4
# 3     9
# 4     16
# 5     25
```

---

## 7. join — ファイルのリレーショナル結合

### 7.1 基本操作

```bash
# 前提: 両ファイルは結合キーでソート済みであること

# file1.txt:
# 001 Alice
# 002 Bob
# 003 Charlie

# file2.txt:
# 001 Engineering
# 002 Marketing
# 003 Design

# デフォルト結合（第1フィールドで結合）
join file1.txt file2.txt
# 出力:
# 001 Alice Engineering
# 002 Bob Marketing
# 003 Charlie Design

# 結合フィールドを指定
join -1 2 -2 1 file_a.txt file_b.txt
# -1 2: file_a の2列目をキーに
# -2 1: file_b の1列目をキーに

# 区切り文字を指定
join -t',' file1.csv file2.csv

# 出力フォーマットを指定
join -o 1.1,1.2,2.2 file1.txt file2.txt
# file1 の1列目, file1 の2列目, file2 の2列目を出力

# マッチしない行も表示（外部結合）
join -a 1 file1.txt file2.txt    # file1 の非マッチ行も表示（LEFT JOIN）
join -a 2 file1.txt file2.txt    # file2 の非マッチ行も表示（RIGHT JOIN）
join -a 1 -a 2 file1.txt file2.txt  # 両方表示（FULL OUTER JOIN）

# マッチしないフィールドの値を指定
join -a 1 -e "N/A" -o auto file1.txt file2.txt
```

---

## 8. comm — ソート済みファイルの比較

### 8.1 基本操作

```bash
# 2つのソート済みファイルを比較して3列出力
comm file1_sorted.txt file2_sorted.txt
# 列1: file1 にのみ存在する行
# 列2: file2 にのみ存在する行
# 列3: 両方に存在する行

# 特定の列を非表示にする
comm -12 file1.txt file2.txt     # 共通行のみ（列3のみ表示）
comm -23 file1.txt file2.txt     # file1 にのみ存在する行
comm -13 file1.txt file2.txt     # file2 にのみ存在する行

# 実用例: 2つのサーバーのインストール済みパッケージの差分
comm -23 <(ssh server1 "dpkg -l | awk '{print \$2}'" | sort) \
         <(ssh server2 "dpkg -l | awk '{print \$2}'" | sort)
# server1 にだけインストールされているパッケージ

# 実用例: 2つのディレクトリのファイルリスト比較
comm -3 <(ls dir1/ | sort) <(ls dir2/ | sort)

# 実用例: 昨日と今日のプロセス一覧の差分
comm -13 <(sort yesterday_processes.txt) <(sort today_processes.txt)
# 今日新たに現れたプロセス
```

---

## 9. 組み合わせパターン（実務レシピ集）

### 9.1 ログ分析

```bash
# アクセスログのIPアドレスTop10
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -10

# 時間帯別アクセス数
awk '{print $4}' access.log | cut -c2-14 | cut -d: -f1-2 | sort | uniq -c | sort -rn
# [15/Jan/2024:10:30:00 → 15/Jan/2024:10 → 時間帯でグルーピング

# HTTPステータスコード集計
awk '{print $9}' access.log | sort | uniq -c | sort -rn
# 出力例:
#   45231 200
#    3421 301
#    1205 404
#     342 500

# エラーレスポンスのURL Top10
awk '$9 >= 400 {print $7}' access.log | sort | uniq -c | sort -rn | head -10

# User-Agent 集計
awk -F'"' '{print $6}' access.log | sort | uniq -c | sort -rn | head -10

# 特定時間帯のリクエスト数
grep "15/Jan/2024:1[0-2]:" access.log | wc -l

# レスポンスサイズの合計
awk '{sum += $10} END {print sum}' access.log
# awk のみでも可能だが、cut + paste + bc でも:
awk '{print $10}' access.log | paste -sd+ | bc

# 1分あたりのリクエスト数（分単位の集計）
awk '{print $4}' access.log | cut -c2-18 | sort | uniq -c | sort -rn | head -20
```

### 9.2 システム管理

```bash
# ディスク使用量の大きいディレクトリ Top20
du -sh /var/* 2>/dev/null | sort -rh | head -20

# 拡張子別のファイル数・合計サイズ
find . -type f -name "*.*" | sed 's/.*\.//' | sort | uniq -c | sort -rn

# 最近更新されたファイル Top10
find /var/log -type f -mmin -60 -exec ls -lt {} + 2>/dev/null | head -10

# ポート別接続数
ss -tn state established | awk '{print $4}' | rev | cut -d: -f1 | rev | sort | uniq -c | sort -rn

# ユーザー別プロセス数
ps aux | awk 'NR>1 {print $1}' | sort | uniq -c | sort -rn

# メモリ使用量の多いプロセス Top10（RSSベース）
ps aux --sort=-rss | head -11 | awk 'NR==1 || NR>1 {printf "%-10s %8s %8s %s\n", $1, $4, $6, $11}'

# 開いているファイル数（プロセス別）
lsof 2>/dev/null | awk '{print $1}' | sort | uniq -c | sort -rn | head -20

# TCP接続状態の集計
ss -tan | awk 'NR>1 {print $1}' | sort | uniq -c | sort -rn
# 出力例:
#   45 ESTABLISHED
#   12 TIME-WAIT
#    5 CLOSE-WAIT
#    3 LISTEN

# /etc/passwd のシェル種類別ユーザー数
awk -F: '{print $7}' /etc/passwd | sort | uniq -c | sort -rn

# crontab のジョブ数（全ユーザー）
for user in $(cut -d: -f1 /etc/passwd); do
    count=$(sudo crontab -u "$user" -l 2>/dev/null | grep -v "^#" | grep -v "^$" | wc -l)
    [ "$count" -gt 0 ] && echo "$count $user"
done | sort -rn
```

### 9.3 テキスト分析

```bash
# 単語頻度分析
cat file.txt | tr -s '[:space:]' '\n' | tr 'A-Z' 'a-z' | tr -d '[:punct:]' | sort | uniq -c | sort -rn | head -20

# 行の長さの分布
awk '{print length}' file.txt | sort -n | uniq -c
# 出力例:
#    15 0    ← 空行が15行
#    42 20   ← 20文字の行が42行
#   ...

# 特定パターンの行の集計
grep -o '[A-Z][A-Z]*' file.txt | sort | uniq -c | sort -rn | head -10
# 大文字のみの単語の頻度

# CSV のフィールド数チェック（データ品質確認）
awk -F',' '{print NF}' data.csv | sort | uniq -c
# 全行同じ数ならデータは整合的

# 重複行の検出とレポート
sort data.txt | uniq -c | awk '$1 > 1 {print}'

# 2つのファイルの共通行
comm -12 <(sort file1.txt) <(sort file2.txt)

# 2つのファイルの差分行
comm -3 <(sort file1.txt) <(sort file2.txt)

# ファイル内の空行数 vs 非空行数
echo "空行: $(grep -c '^$' file.txt)"
echo "非空行: $(grep -c '.' file.txt)"
echo "総行数: $(wc -l < file.txt)"
```

### 9.4 開発・コード分析

```bash
# ソースコードの行数統計
find . -name "*.py" -exec wc -l {} + | sort -rn | head -20

# 言語別コード行数
for ext in py js ts go rb java; do
    count=$(find . -name "*.${ext}" -exec cat {} + 2>/dev/null | wc -l)
    [ "$count" -gt 0 ] && echo "$count $ext"
done | sort -rn

# TODO/FIXME/HACK コメントの集計
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.py" . | awk -F: '{print $3}' | tr -s ' ' | sort | uniq -c | sort -rn

# import文の頻度（Python）
grep "^import\|^from" *.py | awk '{print $2}' | sort | uniq -c | sort -rn

# 関数定義の数（Python）
grep -c "^def " *.py | sort -t: -k2,2rn

# Git のコミットメッセージでよく使われる単語
git log --oneline | tr -s '[:space:]' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -20

# 各ファイルの変更頻度（Git）
git log --name-only --format="" | sort | uniq -c | sort -rn | head -20

# 著者別コード行数
git log --author="Gaku" --numstat --format="" | awk '{add+=$1; del+=$2} END {print "追加:", add, "削除:", del}'

# Makefile のターゲット一覧
grep "^[a-zA-Z_-]*:" Makefile | cut -d: -f1 | sort

# package.json の依存パッケージ数
cat package.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
deps = len(data.get('dependencies', {}))
dev_deps = len(data.get('devDependencies', {}))
print(f'dependencies: {deps}')
print(f'devDependencies: {dev_deps}')
print(f'total: {deps + dev_deps}')
"
```

### 9.5 CSV/データ処理

```bash
# CSV のヘッダー確認
head -1 data.csv | tr ',' '\n' | nl
# 各列に番号をつけて表示

# CSV の特定列の合計
cut -d',' -f3 data.csv | tail -n +2 | paste -sd+ | bc

# CSV の特定列の平均
cut -d',' -f3 data.csv | tail -n +2 | awk '{sum+=$1; count++} END {print sum/count}'

# CSV の特定列のユニーク値一覧
cut -d',' -f2 data.csv | tail -n +2 | sort -u

# CSV の特定列でグルーピング・集計
cut -d',' -f2,5 data.csv | tail -n +2 | sort -t',' -k1,1 | awk -F',' '{
    if (prev != $1 && prev != "") {
        print prev, sum
        sum = 0
    }
    prev = $1
    sum += $2
} END {print prev, sum}'

# CSV の行数（ヘッダー除外）
tail -n +2 data.csv | wc -l

# CSV のカラム数
head -1 data.csv | tr ',' '\n' | wc -l

# CSV の特定列の最大値・最小値
cut -d',' -f3 data.csv | tail -n +2 | sort -n | head -1  # 最小値
cut -d',' -f3 data.csv | tail -n +2 | sort -rn | head -1 # 最大値

# 2つの CSV をキーで結合
join -t',' <(sort -t',' -k1,1 users.csv) <(sort -t',' -k1,1 orders.csv)

# CSV から SQL INSERT 文を生成
tail -n +2 data.csv | awk -F',' '{
    printf "INSERT INTO table_name VALUES ('\''%s'\'', '\''%s'\'', %s);\n", $1, $2, $3
}'
```

---

## 10. スクリプト実例

### 10.1 アクセスログ分析スクリプト

```bash
#!/bin/bash
# access_log_analyzer.sh - アクセスログの包括的分析
# 使い方: ./access_log_analyzer.sh /var/log/nginx/access.log

LOG_FILE="${1:?使い方: $0 <logfile>}"

if [ ! -f "$LOG_FILE" ]; then
    echo "エラー: ファイルが見つかりません: $LOG_FILE" >&2
    exit 1
fi

TOTAL_REQUESTS=$(wc -l < "$LOG_FILE")

echo "================================================"
echo "アクセスログ分析レポート"
echo "対象ファイル: $LOG_FILE"
echo "総リクエスト数: $TOTAL_REQUESTS"
echo "================================================"

echo ""
echo "--- IPアドレス Top10 ---"
awk '{print $1}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

echo ""
echo "--- HTTPステータスコード集計 ---"
awk '{print $9}' "$LOG_FILE" | sort | uniq -c | sort -rn

echo ""
echo "--- リクエストメソッド集計 ---"
awk '{print $6}' "$LOG_FILE" | tr -d '"' | sort | uniq -c | sort -rn

echo ""
echo "--- リクエストURL Top20 ---"
awk '{print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -20

echo ""
echo "--- 時間帯別アクセス数 ---"
awk '{print $4}' "$LOG_FILE" | cut -c14-15 | sort | uniq -c | sort -k2,2n

echo ""
echo "--- 404 エラー URL Top10 ---"
awk '$9 == 404 {print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

echo ""
echo "--- 500 エラー URL Top10 ---"
awk '$9 >= 500 {print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

echo ""
echo "--- User-Agent Top10 ---"
awk -F'"' '{print $6}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

echo ""
echo "--- レスポンスサイズ統計 ---"
awk '{print $10}' "$LOG_FILE" | grep -E '^[0-9]+$' | sort -n | awk '
    BEGIN { count=0; sum=0 }
    {
        a[count++] = $1
        sum += $1
    }
    END {
        if (count > 0) {
            printf "件数: %d\n", count
            printf "合計: %d bytes (%.2f MB)\n", sum, sum/1048576
            printf "平均: %.0f bytes\n", sum/count
            printf "最小: %d bytes\n", a[0]
            printf "最大: %d bytes\n", a[count-1]
            printf "中央値: %d bytes\n", a[int(count/2)]
        }
    }
'

echo ""
echo "================================================"
echo "分析完了"
echo "================================================"
```

### 10.2 CSV データ品質チェックスクリプト

```bash
#!/bin/bash
# csv_quality_check.sh - CSV ファイルのデータ品質チェック
# 使い方: ./csv_quality_check.sh data.csv

CSV_FILE="${1:?使い方: $0 <csv_file>}"
DELIMITER="${2:-,}"

if [ ! -f "$CSV_FILE" ]; then
    echo "エラー: ファイルが見つかりません: $CSV_FILE" >&2
    exit 1
fi

TOTAL_LINES=$(wc -l < "$CSV_FILE")
DATA_LINES=$((TOTAL_LINES - 1))
HEADER=$(head -1 "$CSV_FILE")
EXPECTED_FIELDS=$(echo "$HEADER" | tr "$DELIMITER" '\n' | wc -l)

echo "================================================"
echo "CSV データ品質チェックレポート"
echo "ファイル: $CSV_FILE"
echo "================================================"

echo ""
echo "--- 基本情報 ---"
echo "総行数: $TOTAL_LINES（ヘッダー含む）"
echo "データ行数: $DATA_LINES"
echo "フィールド数（期待値）: $EXPECTED_FIELDS"
echo "ファイルサイズ: $(wc -c < "$CSV_FILE") bytes"

echo ""
echo "--- ヘッダー ---"
echo "$HEADER" | tr "$DELIMITER" '\n' | nl

echo ""
echo "--- フィールド数チェック ---"
FIELD_CHECK=$(awk -F"$DELIMITER" '{print NF}' "$CSV_FILE" | sort | uniq -c | sort -rn)
echo "$FIELD_CHECK"
INCONSISTENT=$(echo "$FIELD_CHECK" | wc -l)
if [ "$INCONSISTENT" -eq 1 ]; then
    echo "結果: OK（全行のフィールド数が一致）"
else
    echo "警告: フィールド数が一致しない行があります"
    echo "不整合な行（先頭5件）:"
    awk -F"$DELIMITER" -v expected="$EXPECTED_FIELDS" 'NF != expected {print NR": "NF" fields - "$0}' "$CSV_FILE" | head -5
fi

echo ""
echo "--- 空行チェック ---"
EMPTY_LINES=$(grep -c "^$" "$CSV_FILE")
echo "空行数: $EMPTY_LINES"

echo ""
echo "--- 重複行チェック ---"
DUPES=$(tail -n +2 "$CSV_FILE" | sort | uniq -d | wc -l)
echo "重複行数: $DUPES"
if [ "$DUPES" -gt 0 ]; then
    echo "重複行の例（先頭5件）:"
    tail -n +2 "$CSV_FILE" | sort | uniq -d | head -5
fi

echo ""
echo "--- 各フィールドの統計 ---"
col_num=1
echo "$HEADER" | tr "$DELIMITER" '\n' | while read -r col_name; do
    echo ""
    echo "  フィールド $col_num: $col_name"
    VALUES=$(cut -d"$DELIMITER" -f"$col_num" "$CSV_FILE" | tail -n +2)
    TOTAL=$(echo "$VALUES" | wc -l)
    EMPTY=$(echo "$VALUES" | grep -c "^$")
    UNIQUE=$(echo "$VALUES" | sort -u | wc -l)
    echo "    総数: $TOTAL  空欄: $EMPTY  ユニーク: $UNIQUE"
    echo "    上位5値:"
    echo "$VALUES" | sort | uniq -c | sort -rn | head -5 | sed 's/^/      /'
    col_num=$((col_num + 1))
done

echo ""
echo "================================================"
echo "チェック完了"
echo "================================================"
```

### 10.3 テキスト統計レポートスクリプト

```bash
#!/bin/bash
# text_stats.sh - テキストファイルの統計情報レポート
# 使い方: ./text_stats.sh document.txt

FILE="${1:?使い方: $0 <text_file>}"

if [ ! -f "$FILE" ]; then
    echo "エラー: ファイルが見つかりません: $FILE" >&2
    exit 1
fi

echo "================================================"
echo "テキスト統計レポート: $FILE"
echo "================================================"

echo ""
echo "--- 基本カウント ---"
echo "行数:     $(wc -l < "$FILE")"
echo "単語数:   $(wc -w < "$FILE")"
echo "文字数:   $(wc -m < "$FILE")"
echo "バイト数: $(wc -c < "$FILE")"
echo "最長行:   $(wc -L < "$FILE") 文字"

echo ""
echo "--- 行の統計 ---"
echo "空行数:   $(grep -c '^$' "$FILE")"
echo "非空行数: $(grep -c '.' "$FILE")"
echo "行の長さ分布:"
awk '{print length}' "$FILE" | sort -n | awk '
    BEGIN { min=999999; max=0; sum=0; count=0 }
    {
        if ($1 < min) min = $1
        if ($1 > max) max = $1
        sum += $1
        count++
    }
    END {
        if (count > 0) {
            printf "  最短: %d文字\n", min
            printf "  最長: %d文字\n", max
            printf "  平均: %.1f文字\n", sum/count
        }
    }
'

echo ""
echo "--- 単語頻度 Top20 ---"
tr -s '[:space:]' '\n' < "$FILE" | tr 'A-Z' 'a-z' | tr -d '[:punct:]' | sort | uniq -c | sort -rn | head -20

echo ""
echo "--- 文字種別統計 ---"
echo "英大文字: $(tr -cd 'A-Z' < "$FILE" | wc -c)"
echo "英小文字: $(tr -cd 'a-z' < "$FILE" | wc -c)"
echo "数字:     $(tr -cd '0-9' < "$FILE" | wc -c)"
echo "空白:     $(tr -cd ' \t' < "$FILE" | wc -c)"
echo "記号:     $(tr -cd '[:punct:]' < "$FILE" | wc -c)"

echo ""
echo "================================================"
echo "レポート完了"
echo "================================================"
```

---

## 11. パフォーマンスのヒント

```bash
# 大規模ファイルでの注意点

# 1. sort は大量のメモリを消費する可能性がある
# テンポラリ領域とメモリを明示指定
sort -T /tmp/sort_work -S 4G huge_file.txt

# 2. LC_ALL=C でロケール処理をスキップ → ソート高速化
LC_ALL=C sort file.txt
# 日本語の文字コード順は変わるが、圧倒的に高速

# 3. パイプチェーンの最適化
# 悪い例（全行を処理してから head）
sort huge.log | uniq -c | sort -rn | head -10
# 良い例（先に必要な列だけ抽出）
awk '{print $1}' huge.log | sort | uniq -c | sort -rn | head -10

# 4. wc -l は非常に高速（行末文字のカウントのみ）
# 巨大ファイルの行数確認に最適
wc -l huge_file.txt

# 5. cut は awk より高速（単純なフィールド抽出の場合）
cut -d',' -f2 data.csv              # 高速
awk -F',' '{print $2}' data.csv     # やや遅い（awk の起動コスト）

# 6. sort -u は sort | uniq より効率的
sort -u file.txt                    # 高速（1パス）
sort file.txt | uniq                # やや遅い（2プロセス）

# 7. grep -c は grep | wc -l より効率的
grep -c "pattern" file.txt          # 高速（カウントのみ）
grep "pattern" file.txt | wc -l     # やや遅い（パイプのオーバーヘッド）

# 8. 並列処理の活用
# GNU parallel がインストールされている場合
find /var/log -name "*.log" | parallel "grep -c ERROR {} | xargs -I{c} echo {} {c}"
```

---

## 12. トラブルシューティング

```bash
# 問題1: sort の結果が期待通りにならない
# 原因: ロケール設定
# 解決策:
export LC_ALL=C
sort file.txt
# C ロケールでは ASCII コード順でソートされる

# 問題2: uniq が重複を検出しない
# 原因: uniq は「連続する」重複行のみ処理する
# 解決策: 必ず sort を前に置く
sort file.txt | uniq -c

# 問題3: cut が正しく列を抽出しない
# 原因: 連続する区切り文字
# 解決策: tr -s で連続区切り文字を圧縮、または awk を使う
ps aux | tr -s ' ' | cut -d' ' -f4

# 問題4: wc -l が0を返す（ファイルが空でないのに）
# 原因: ファイルの最終行に改行がない
# 確認:
tail -c 1 file.txt | xxd
# 解決策: awk を使う
awk 'END {print NR}' file.txt

# 問題5: sort -n で数値がうまくソートされない
# 原因: 数値の前に空白や不可視文字がある
# 解決策: tr で前処理
tr -d '[:blank:]' < file.txt | sort -n

# 問題6: マルチバイト文字で cut -c が正しく動作しない
# 原因: ロケール設定
# 解決策:
export LC_ALL=ja_JP.UTF-8
cut -c1-5 japanese_text.txt

# 問題7: sort が "write failed: /tmp/sort...: No space left on device"
# 原因: /tmp の空き容量不足
# 解決策: テンポラリディレクトリを変更
sort -T /home/user/tmp large_file.txt
```

---

## まとめ

| コマンド | 用途 | よく使うオプション |
|---------|------|-------------------|
| sort | テキストのソート | -n（数値）, -r（逆順）, -k（キー）, -t（区切り）, -u（重複排除）, -h（人間可読） |
| uniq | 重複行の処理 | -c（カウント）, -d（重複のみ）, -u（ユニークのみ）, -f（フィールド無視） |
| cut | 列の切り出し | -d（区切り）, -f（フィールド）, -c（文字位置）, --complement（補集合） |
| wc | カウント | -l（行）, -w（単語）, -c（バイト）, -m（文字）, -L（最長行） |
| tr | 文字変換・削除 | -d（削除）, -s（圧縮）, -c（補集合） |
| paste | 水平結合 | -d（区切り）, -s（直列化） |
| join | リレーショナル結合 | -t（区切り）, -1/-2（キー列）, -a（外部結合） |
| comm | ソート済み比較 | -1/-2/-3（列非表示） |

### 定番パイプラインパターン

```bash
# 頻度ランキング
... | sort | uniq -c | sort -rn | head -N

# 列の抽出 → ソート → 集計
cut -d',' -fN file.csv | sort | uniq -c | sort -rn

# テキストの正規化 → 単語分析
cat file.txt | tr 'A-Z' 'a-z' | tr -s '[:space:]' '\n' | sort | uniq -c | sort -rn

# CSV のフィールド合計
cut -d',' -fN file.csv | tail -n +2 | paste -sd+ | bc
```

---

## 13. よくある質問（FAQ）

### Q1: sort と sort -u、sort | uniq の違いは？

```bash
# sort -u: ソートと同時に重複排除（1パス、高速）
sort -u file.txt

# sort | uniq: ソート後に別プロセスで重複排除（2パス）
sort file.txt | uniq

# 機能差: uniq -c のようなカウント機能は sort -u にはない
# → カウントが必要な場合は sort | uniq -c を使う
```

### Q2: 大文字小文字を無視してソート・重複排除するには？

```bash
# sort -f と uniq -i を組み合わせる
sort -f file.txt | uniq -i -c | sort -rn

# awk で前処理する方法
awk '{print tolower($0)}' file.txt | sort | uniq -c | sort -rn
```

### Q3: CSV の特定列に空白が含まれる場合の対処法は？

```bash
# cut はクォーテーションを認識しない
# → Python の csv モジュールや csvkit を使う
pip install csvkit

# csvkit の例
csvcut -c 2 data.csv              # 2列目を正しく抽出
csvsort -c 3 data.csv             # 3列目でソート
csvgrep -c 2 -m "Tokyo" data.csv  # 2列目が "Tokyo" の行
csvstat data.csv                   # 全列の統計情報
```

### Q4: sort でロケールの影響を完全に排除するには？

```bash
# 環境変数 LC_ALL を C に設定
LC_ALL=C sort file.txt

# スクリプト内で永続的に設定
export LC_ALL=C
sort file.txt

# 特定のコマンドだけ一時的に変更
LC_ALL=C sort file.txt
# 後続コマンドには影響しない
```

### Q5: 複数ファイルのソート結果をマージするには？

```bash
# 各ファイルが既にソート済みの場合
sort -m file1.txt file2.txt file3.txt > merged.txt

# ソートされていない場合は通常の sort
sort file1.txt file2.txt file3.txt > merged.txt

# 大量のファイルの場合
find /var/log -name "*.log" -exec sort -m {} + > all_sorted.txt
```

---

## 次に読むべきガイド
→ [[../03-process-management/00-ps-top-htop.md]] — プロセス監視

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.5, O'Reilly, 2022.
2. Shotts, W. "The Linux Command Line." 2nd Ed, No Starch Press, 2019.
3. GNU Coreutils Manual. "sort, uniq, cut, wc, tr, paste, join, comm." gnu.org.
4. Kernighan, B. & Pike, R. "The UNIX Programming Environment." Prentice Hall, 1984.
5. Robbins, A. & Beebe, N. "Classic Shell Scripting." O'Reilly, 2005.
