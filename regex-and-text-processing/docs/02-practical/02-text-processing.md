# テキスト処理 -- sed/awk/grep、ログ解析、CSV

> Unixのテキスト処理ツール(grep, sed, awk)は正規表現の最も実践的な活用場面である。ログ解析、CSV処理、データ変換のパイプライン構築を通じて、コマンドラインでの正規表現活用法を体系的に解説する。

## この章で学ぶこと

1. **grep/sed/awk の正規表現構文と使い分け** -- 各ツールの得意分野と適切な選択
2. **ログ解析のパイプライン構築** -- 抽出・集計・整形の実践的ワークフロー
3. **CSV/TSV 処理の正規表現アプローチと限界** -- 構造化データに対する正規表現の適用範囲

---

## 1. grep -- パターン検索

### 1.1 基本的な使い方

```bash
# 基本検索: パターンにマッチする行を表示
grep 'ERROR' /var/log/syslog

# -E: 拡張正規表現 (ERE) を使用
grep -E 'ERROR|WARN' /var/log/syslog

# -i: 大文字小文字を無視
grep -i 'error' /var/log/syslog

# -n: 行番号を表示
grep -n 'ERROR' /var/log/syslog

# -c: マッチした行数をカウント
grep -c 'ERROR' /var/log/syslog

# -v: マッチしない行を表示(反転)
grep -v 'DEBUG' /var/log/syslog

# -o: マッチした部分のみ表示
grep -oE '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}' access.log

# -A/-B/-C: 前後のコンテキスト行を表示
grep -A 3 'ERROR' /var/log/syslog    # 後3行
grep -B 2 'ERROR' /var/log/syslog    # 前2行
grep -C 2 'ERROR' /var/log/syslog    # 前後2行
```

### 1.2 grep の正規表現オプション

```bash
# BRE (Basic Regular Expression) -- デフォルト
# メタ文字 + ? | ( ) { } はエスケープが必要
grep 'hello\(world\)' file.txt
grep 'a\{3\}' file.txt

# ERE (Extended Regular Expression) -- -E オプション
# メタ文字をそのまま使える(推奨)
grep -E 'hello(world)' file.txt
grep -E 'a{3}' file.txt

# PCRE (Perl Compatible) -- -P オプション (GNU grep)
# 先読み、後読み、\d 等が使える
grep -P '(?<=\$)\d+' file.txt
grep -P '\d+(?=円)' file.txt

# 固定文字列検索 -- -F オプション (高速)
# 正規表現を使わない(メタ文字もリテラル)
grep -F '*.txt' file.txt    # "*" をリテラルとして検索
```

### 1.3 実践的な grep パターン

```bash
# IPアドレスを抽出
grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' access.log

# メールアドレスを抽出
grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' contacts.txt

# 特定のHTTPステータスコード
grep -E 'HTTP/[0-9.]+" (4[0-9]{2}|5[0-9]{2})' access.log

# 日付範囲でフィルタ
grep -E '2026-02-(1[0-9]|2[0-9])' logfile.txt

# 複数条件のAND (パイプで連結)
grep 'ERROR' logfile.txt | grep 'database' | grep -v 'timeout'
```

---

## 2. sed -- ストリーム編集

### 2.1 基本的な使い方

```bash
# 置換: s/pattern/replacement/
sed 's/old/new/' file.txt          # 各行の最初のマッチを置換
sed 's/old/new/g' file.txt         # 全マッチを置換(global)
sed 's/old/new/gi' file.txt        # 大文字小文字無視

# 削除: d
sed '/^#/d' file.txt               # コメント行を削除
sed '/^$/d' file.txt               # 空行を削除
sed '1,5d' file.txt                # 1-5行目を削除

# 行指定
sed '3s/old/new/' file.txt         # 3行目のみ置換
sed '1,10s/old/new/g' file.txt     # 1-10行目で置換
sed '/ERROR/s/old/new/g' file.txt  # ERRORを含む行で置換

# インプレース編集(-i)
sed -i 's/old/new/g' file.txt      # ファイルを直接変更(GNU)
sed -i '' 's/old/new/g' file.txt   # macOS (バックアップ拡張子必須)
sed -i.bak 's/old/new/g' file.txt  # バックアップ付き
```

### 2.2 sed の高度なパターン

```bash
# キャプチャグループと後方参照
# 日付形式変換: YYYY-MM-DD → DD/MM/YYYY
sed -E 's/([0-9]{4})-([0-9]{2})-([0-9]{2})/\3\/\2\/\1/g' dates.txt

# HTMLタグの除去
sed 's/<[^>]*>//g' page.html

# 行の先頭/末尾に追加
sed 's/^/PREFIX: /' file.txt       # 行頭に追加
sed 's/$/ SUFFIX/' file.txt        # 行末に追加

# 複数の置換を連続実行
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# パターン間の行を抽出
sed -n '/START/,/END/p' file.txt   # START から END までを出力

# 奇数行/偶数行
sed -n '1~2p' file.txt             # 奇数行のみ(GNU sed)
sed -n '2~2p' file.txt             # 偶数行のみ(GNU sed)

# 空白の正規化
sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' file.txt
```

### 2.3 sed スクリプトの例

```bash
# ログファイルのクレンジング
sed -E '
    /^$/d                          # 空行削除
    s/\t/  /g                      # タブを2スペースに
    s/[[:space:]]+$//              # 末尾空白削除
    s/([0-9]{4})-([0-9]{2})-([0-9]{2})/\1年\2月\3日/g  # 日付変換
' logfile.txt
```

---

## 3. awk -- パターンスキャン・処理

### 3.1 基本的な使い方

```bash
# フィールドの抽出 (デフォルト区切り: 空白)
awk '{print $1}' file.txt          # 第1フィールド
awk '{print $1, $3}' file.txt      # 第1, 第3フィールド
awk '{print $NF}' file.txt         # 最終フィールド
awk '{print NR, $0}' file.txt      # 行番号付き

# 区切り文字の指定
awk -F',' '{print $1, $2}' data.csv      # CSV
awk -F'\t' '{print $1, $2}' data.tsv     # TSV
awk -F':' '{print $1, $3}' /etc/passwd   # コロン区切り

# パターンマッチ
awk '/ERROR/ {print}' logfile.txt           # ERROR を含む行
awk '/^2026-02-11/ {print}' logfile.txt     # 日付で絞り込み
awk '$3 > 100 {print $1, $3}' data.txt     # 第3フィールドが100超

# 正規表現マッチ
awk '$2 ~ /^ERR/ {print}' logfile.txt       # 第2フィールドがERRで始まる
awk '$2 !~ /DEBUG/ {print}' logfile.txt     # 第2フィールドがDEBUGでない
```

### 3.2 awk の集計機能

```bash
# 行数カウント
awk 'END {print NR}' file.txt

# 合計
awk '{sum += $3} END {print "合計:", sum}' data.txt

# 平均
awk '{sum += $3; n++} END {print "平均:", sum/n}' data.txt

# 最大/最小
awk 'NR==1 || $3 > max {max=$3} END {print "最大:", max}' data.txt

# グループ別集計
awk '{count[$1]++} END {for (k in count) print k, count[k]}' access.log

# ユニークカウント
awk '!seen[$0]++' file.txt         # 重複行を除去(順序保持)
```

### 3.3 awk の正規表現活用

```bash
# 正規表現でフィールドを分割
awk -F'[,;|]' '{print $1, $2}' data.txt

# match() 関数で部分文字列を抽出
awk '{
    if (match($0, /[0-9]{4}-[0-9]{2}-[0-9]{2}/)) {
        print substr($0, RSTART, RLENGTH)
    }
}' logfile.txt

# gsub() で置換
awk '{gsub(/ERROR/, "***ERROR***"); print}' logfile.txt

# 複数パターンの処理
awk '
    /ERROR/  {errors++}
    /WARN/   {warns++}
    /INFO/   {infos++}
    END {
        print "ERROR:", errors+0
        print "WARN:",  warns+0
        print "INFO:",  infos+0
    }
' logfile.txt
```

---

## 4. ログ解析パイプライン

### 4.1 Apache アクセスログの解析

```bash
# Apache Combined Log Format:
# 192.168.1.1 - - [11/Feb/2026:10:30:45 +0900] "GET /path HTTP/1.1" 200 1234

# 上位IPアドレス (アクセス数)
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -10

# HTTPステータスコード別カウント
awk '{print $9}' access.log | sort | uniq -c | sort -rn

# 404エラーのURLを抽出
awk '$9 == 404 {print $7}' access.log | sort | uniq -c | sort -rn

# 時間帯別アクセス数
awk -F'[\\[:]' '{print $3}' access.log | sort | uniq -c

# レスポンスサイズの合計
awk '{sum += $10} END {printf "合計: %.2f MB\n", sum/1024/1024}' access.log

# 遅いリクエスト (レスポンス時間が閾値以上)
awk '$NF > 1000 {print $7, $NF "ms"}' access.log | sort -t' ' -k2 -rn | head -10
```

### 4.2 アプリケーションログの解析

```bash
# JSON ログの場合 (jq と組み合わせ)
# {"timestamp":"2026-02-11T10:30:45","level":"ERROR","message":"..."}

# ERROR レベルのログを抽出
grep '"level":"ERROR"' app.log | head -20

# jq でフィールド抽出
grep '"level":"ERROR"' app.log | jq -r '.timestamp + " " + .message'

# テキストログのパターン解析
# [2026-02-11 10:30:45] [ERROR] [module] message

# 時間帯別エラー数
grep -E '\[ERROR\]' app.log | \
    grep -oE '\d{2}:\d{2}' | \
    awk -F: '{print $1":00"}' | \
    sort | uniq -c | sort -rn

# エラーメッセージのトップ10
grep -E '\[ERROR\]' app.log | \
    sed -E 's/.*\[ERROR\] \[[^\]]+\] //' | \
    sort | uniq -c | sort -rn | head -10
```

### 4.3 リアルタイムログ監視

```bash
# tail -f + grep でリアルタイム監視
tail -f /var/log/syslog | grep --color=auto -E 'ERROR|WARN'

# 複数ファイルの同時監視
tail -f /var/log/*.log | grep --color=auto -E 'ERROR|CRITICAL'

# awk でリアルタイム集計
tail -f access.log | awk '
    {
        status[$9]++
        if (NR % 100 == 0) {
            for (s in status) printf "%s: %d  ", s, status[s]
            print ""
        }
    }
'
```

---

## 5. CSV 処理

### 5.1 基本的なCSV処理

```bash
# 単純なCSV (引用符なし)
# name,age,city
# Alice,30,Tokyo
# Bob,25,Osaka

# 特定列の抽出
awk -F',' '{print $1, $3}' data.csv

# 条件フィルタ
awk -F',' '$2 > 25 {print}' data.csv

# 列の順序変更
awk -F',' '{print $3","$1","$2}' data.csv

# ヘッダー付き処理
awk -F',' 'NR==1 {print; next} $2 > 25 {print}' data.csv
```

### 5.2 引用符付きCSVの問題

```
CSV の問題:
┌────────────────────────────────────────────────┐
│ フィールド内にカンマがある場合:                    │
│   "Tokyo, Japan",30,engineer                    │
│                                                │
│ フィールド内に引用符がある場合:                    │
│   "He said ""hello""",30,engineer               │
│                                                │
│ フィールド内に改行がある場合:                      │
│   "Line 1                                       │
│   Line 2",30,engineer                           │
│                                                │
│ → 正規表現だけでは正しく処理できない!             │
│ → 専用パーサーを使うべき                          │
└────────────────────────────────────────────────┘
```

```python
# Python での正しいCSV処理
import csv
import re

# NG: 正規表現でCSVを分割
def parse_csv_bad(line):
    return line.split(',')  # 引用符内のカンマで壊れる

# OK: csv モジュールを使用
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# 正規表現が有用な場面: CSVの各フィールドに対する処理
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # フィールド単位で正規表現を適用
        for field in row:
            if re.match(r'\d{4}-\d{2}-\d{2}', field):
                print(f"  日付フィールド: {field}")
```

---

## 6. ASCII 図解

### 6.1 grep/sed/awk の使い分け

```
ユースケース別ツール選択:

テキスト検索(行のフィルタリング)
  → grep
  grep 'ERROR' log.txt

テキスト置換(行単位の変換)
  → sed
  sed 's/old/new/g' file.txt

フィールド処理(列の抽出・集計)
  → awk
  awk -F',' '{print $1, $3}' data.csv

┌──────────┬────────────┬──────────┬──────────┐
│ 操作      │ grep       │ sed      │ awk      │
├──────────┼────────────┼──────────┼──────────┤
│ 検索      │ ★★★       │ ★        │ ★★       │
│ 置換      │ ─          │ ★★★     │ ★★       │
│ 抽出      │ ★★ (-o)   │ ★        │ ★★★     │
│ 集計      │ ─          │ ─        │ ★★★     │
│ フィルタ  │ ★★★       │ ★★       │ ★★★     │
│ 変換      │ ─          │ ★★★     │ ★★★     │
└──────────┴────────────┴──────────┴──────────┘
```

### 6.2 パイプラインの構造

```
ログ解析パイプラインの典型例:

access.log
    │
    ▼
┌────────┐   ERRORを含む行   ┌────────┐
│  grep  │ ────────────────→ │  awk   │
│ 'ERROR'│                   │ '{$1}' │
└────────┘                   └───┬────┘
                                 │ IPアドレス列
                                 ▼
                            ┌────────┐
                            │  sort  │
                            └───┬────┘
                                │ ソート済み
                                ▼
                            ┌────────┐
                            │ uniq -c│
                            └───┬────┘
                                │ カウント付き
                                ▼
                            ┌──────────┐
                            │sort -rn  │
                            │ head -10 │
                            └──────────┘
                                │
                                ▼
                            上位10 IP
```

### 6.3 BRE vs ERE vs PCRE の構文差異

```
同じパターンの書き方:

「3桁以上の数字」
  BRE:  [0-9]\{3,\}        (波括弧にエスケープ必要)
  ERE:  [0-9]{3,}          (そのまま)
  PCRE: \d{3,}             (\d が使える)

「cat または dog」
  BRE:  cat\|dog           (パイプにエスケープ必要)
  ERE:  cat|dog            (そのまま)
  PCRE: cat|dog            (そのまま)

「グループ + 後方参照」
  BRE:  \(hello\) \1       (括弧にエスケープ必要)
  ERE:  (hello) \1         (括弧はそのまま、参照は \1)
  PCRE: (hello) \1         (括弧はそのまま、参照は \1)

ツール対応:
  grep:       BRE (デフォルト), ERE (-E), PCRE (-P)
  sed:        BRE (デフォルト), ERE (-E)
  awk:        ERE (デフォルト)
```

---

## 7. 比較表

### 7.1 ツール特性比較

| 特性 | grep | sed | awk |
|------|------|-----|-----|
| 主な用途 | パターン検索 | ストリーム編集 | フィールド処理 |
| 正規表現 | BRE/ERE/PCRE | BRE/ERE | ERE |
| 行操作 | フィルタ | 変換 | 変換+集計 |
| フィールド | 不可 | 限定的 | 強力 |
| 計算 | 不可 | 不可 | 可能 |
| 変数 | 不可 | ホールドスペース | 配列・変数 |
| 速度 | 最速 | 速い | やや遅い |
| 学習コスト | 低 | 中 | 高 |

### 7.2 モダンな代替ツール

| 従来ツール | モダン代替 | 特徴 |
|-----------|----------|------|
| grep | ripgrep (rg) | 高速、.gitignore対応、Unicode対応 |
| sed | sd | シンプルな構文、PCRE対応 |
| awk | miller (mlr) | CSV/JSON/TSV のネイティブ対応 |
| find + grep | fd + rg | 高速、直感的なUI |
| cat + grep | bat | シンタックスハイライト付き |
| -- | jq | JSON 処理の専門ツール |
| -- | xsv/qsv | CSV 処理の専門ツール |

---

## 8. アンチパターン

### 8.1 アンチパターン: CSV を正規表現だけで処理する

```bash
# NG: カンマで分割(引用符内のカンマで壊れる)
awk -F',' '{print $2}' data.csv
# 入力: "Tokyo, Japan",30 → $2 = " Japan" (壊れている)

# OK: 専用ツールを使う
# csvkit
csvcut -c 2 data.csv

# Miller
mlr --csv cut -f name data.csv

# Python
python3 -c "
import csv, sys
for row in csv.reader(sys.stdin):
    print(row[1])
" < data.csv
```

### 8.2 アンチパターン: 巨大ファイルに対する非効率なパイプライン

```bash
# NG: 非効率(ファイルを何度も走査)
ERROR_COUNT=$(grep -c 'ERROR' huge.log)
WARN_COUNT=$(grep -c 'WARN' huge.log)
INFO_COUNT=$(grep -c 'INFO' huge.log)
# → ファイルを3回読み込む

# OK: 1回の走査で全てカウント
awk '
    /ERROR/ {e++}
    /WARN/  {w++}
    /INFO/  {i++}
    END {
        print "ERROR:", e+0
        print "WARN:",  w+0
        print "INFO:",  i+0
    }
' huge.log
# → ファイルを1回だけ読み込む
```

---

## 9. FAQ

### Q1: grep -E と egrep の違いは？

**A**: 機能は同一。`egrep` は `grep -E` のエイリアスだが、POSIX.1-2008 で非推奨になった。`grep -E` を使うことを推奨する。同様に `fgrep` も `grep -F` を使うべきである。

### Q2: sed の `-i` オプションは GNU と macOS(BSD) で異なるか？

**A**: **異なる**。GNU sed は `-i` のみで動作するが、BSD sed(macOS)は `-i ''` のように空の拡張子を必須とする:

```bash
# GNU (Linux):
sed -i 's/old/new/g' file.txt

# BSD (macOS):
sed -i '' 's/old/new/g' file.txt

# 両方で動作:
sed -i.bak 's/old/new/g' file.txt && rm file.txt.bak
```

### Q3: awk で正規表現のキャプチャグループを使えるか？

**A**: POSIX awk ではキャプチャグループの後方参照は使えない。`match()` 関数と `substr()` で代替する:

```bash
# POSIX awk: match + substr
awk '{
    if (match($0, /([0-9]+)-([0-9]+)/, arr)) {
        print arr[1], arr[2]
    }
}' file.txt
# 注: 第3引数の配列は GNU awk (gawk) のみ

# gawk: 配列キャプチャ
gawk 'match($0, /([0-9]+)-([0-9]+)/, a) {print a[1], a[2]}' file.txt
```

### Q4: ripgrep (rg) と grep のどちらを使うべきか？

**A**: 新規プロジェクトでは **ripgrep を推奨**。理由:

- デフォルトで `.gitignore` を尊重
- デフォルトで再帰検索
- Unicode 完全対応
- 大規模リポジトリで数倍高速
- PCRE2 対応(先読み等が使える)

```bash
# ripgrep の基本使用法
rg 'ERROR' .                      # 再帰検索(デフォルト)
rg -i 'error' --type py           # Pythonファイルのみ
rg 'pattern' -g '*.log'           # glob でファイル指定
rg -P '(?<=\$)\d+' .              # PCRE2 (先読み・後読み)
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| grep | パターン検索の基本ツール、-E(ERE) -P(PCRE)を活用 |
| sed | ストリーム置換・削除・変換、-E で ERE を使用 |
| awk | フィールド処理・集計、正規表現でパターンマッチ |
| パイプライン | `grep \| awk \| sort \| uniq -c` の組み合わせが強力 |
| CSV | 正規表現だけでは不十分、専用ツール(csvkit, miller)を併用 |
| ログ解析 | 1回の走査で複数指標を集計するのが効率的 |
| モダン代替 | ripgrep, sd, miller が従来ツールを高速化 |

## 次に読むべきガイド

- [03-regex-alternatives.md](./03-regex-alternatives.md) -- 正規表現の代替技術
- [../01-advanced/03-performance.md](../01-advanced/03-performance.md) -- パフォーマンス最適化

## 参考文献

1. **Dale Dougherty & Arnold Robbins** "sed & awk, 2nd Edition" O'Reilly, 1997 -- sed/awk の定番書
2. **GNU Grep Manual** https://www.gnu.org/software/grep/manual/ -- GNU grep の公式リファレンス
3. **The AWK Programming Language** Aho, Kernighan, Weinberger, 2024 (2nd Edition) -- awk 原作者による改訂版
4. **ripgrep** https://github.com/BurntSushi/ripgrep -- モダンな grep 代替ツール
