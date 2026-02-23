# テキスト処理 -- sed/awk/grep、ログ解析、CSV

> Unixのテキスト処理ツール(grep, sed, awk)は正規表現の最も実践的な活用場面である。ログ解析、CSV処理、データ変換のパイプライン構築を通じて、コマンドラインでの正規表現活用法を体系的に解説する。

## この章で学ぶこと

1. **grep/sed/awk の正規表現構文と使い分け** -- 各ツールの得意分野と適切な選択
2. **ログ解析のパイプライン構築** -- 抽出・集計・整形の実践的ワークフロー
3. **CSV/TSV 処理の正規表現アプローチと限界** -- 構造化データに対する正規表現の適用範囲
4. **モダンツールとの連携** -- ripgrep, miller, jq 等との統合的な活用法
5. **実運用におけるベストプラクティス** -- パフォーマンス、安全性、保守性を両立する手法

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

### 1.4 再帰検索とファイル指定

```bash
# -r: 再帰的にディレクトリを検索
grep -r 'TODO' /path/to/project/

# -l: マッチしたファイル名のみ表示
grep -rl 'deprecated' src/

# --include: 検索対象ファイルをパターンで指定
grep -rn 'import' --include='*.py' src/

# --exclude: 特定ファイルを除外
grep -rn 'password' --exclude='*.log' --exclude='*.bak' .

# --exclude-dir: 特定ディレクトリを除外
grep -rn 'API_KEY' --exclude-dir=node_modules --exclude-dir=.git .

# -L: マッチしなかったファイル名を表示
grep -rL 'Copyright' --include='*.py' src/
```

### 1.5 grep の出力制御とフォーマット

```bash
# --color=auto: マッチ部分をハイライト表示
grep --color=auto 'ERROR' logfile.txt

# -H/-h: ファイル名の表示/非表示
grep -H 'ERROR' *.log    # ファイル名付き（デフォルト・複数ファイル時）
grep -h 'ERROR' *.log    # ファイル名なし

# -w: 単語単位でマッチ（部分一致を防止）
grep -w 'error' logfile.txt    # "error" にマッチ、"errors" にはマッチしない

# -x: 行全体がパターンに一致する場合のみマッチ
grep -x 'OK' status.txt

# -m: マッチ数の上限を指定
grep -m 5 'ERROR' huge.log    # 最初の5件でストップ

# -q: 出力なし（スクリプトの条件分岐用）
if grep -q 'ERROR' logfile.txt; then
    echo "エラーが検出されました"
fi

# --count と --files-with-matches の組み合わせ
grep -rl 'TODO' src/ | xargs grep -c 'TODO' | sort -t: -k2 -rn | head -10
# → TODO が多いファイルのランキング
```

### 1.6 grep と正規表現のパフォーマンスチューニング

```bash
# -F（固定文字列）は正規表現より高速
# 正規表現が不要な場合は必ず -F を使う
grep -F 'NullPointerException' huge.log

# LC_ALL=C で処理を高速化（ロケール処理のオーバーヘッド回避）
LC_ALL=C grep 'ERROR' huge.log

# 行バッファリングの制御
grep --line-buffered 'ERROR' /var/log/syslog    # リアルタイム監視時

# バイナリファイルの処理
grep -a 'pattern' binary_file       # バイナリをテキストとして処理
grep -I 'pattern' mixed_files       # バイナリファイルをスキップ

# 大量ファイルの並列検索（xargs + grep）
find /var/log -name '*.log' -print0 | xargs -0 -P 4 grep -l 'ERROR'
# -P 4: 4プロセスで並列実行
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

### 2.4 sed のホールドスペースを使った高度な操作

```bash
# ホールドスペース: sed が持つ2番目のバッファ
# パターンスペース: 現在処理中の行（通常使う）
# ホールドスペース: 行をまたいだ処理用の保存領域

# h: パターンスペース → ホールドスペース（コピー）
# H: パターンスペース → ホールドスペース（追加）
# g: ホールドスペース → パターンスペース（コピー）
# G: ホールドスペース → パターンスペース（追加）
# x: パターンスペースとホールドスペースを交換

# 行を逆順にする
sed -n '1!G;h;$p' file.txt

# 連続する空行を1つにまとめる
sed '/^$/N;/^\n$/d' file.txt

# 2行ずつ結合する
sed 'N;s/\n/ /' file.txt

# パターン間の行を削除（STARTとEND自体は残す）
sed '/START/,/END/{/START/!{/END/!d}}' file.txt

# パターンの前に行を挿入
sed '/TARGET/i\--- ここに挿入 ---' file.txt

# パターンの後に行を追加
sed '/TARGET/a\--- ここに追加 ---' file.txt

# ラベルと分岐を使った複雑な処理
# 連続する行を1行にまとめる（バックスラッシュ継続行の処理）
sed -E ':loop; /\\$/{ N; s/\\\n/ /; b loop }' file.txt
```

### 2.5 sed によるファイル変換の実践例

```bash
# INIファイルからJSON風への変換
# 入力: [section]
#        key=value
sed -E '
    /^\[.*\]$/ {
        s/\[(.*)\]/"\1": {/
    }
    /^[^[#].*=/ {
        s/^([^=]+)=(.*)$/  "\1": "\2",/
    }
    /^$/d
' config.ini

# Markdownの見出しからHTMLへの変換
sed -E '
    s/^### (.*)$/<h3>\1<\/h3>/
    s/^## (.*)$/<h2>\1<\/h2>/
    s/^# (.*)$/<h1>\1<\/h1>/
    s/\*\*([^*]+)\*\*/<strong>\1<\/strong>/g
    s/\*([^*]+)\*/<em>\1<\/em>/g
' document.md

# 設定ファイルの特定セクションの値を変更
sed -E '/^\[database\]$/,/^\[/ {
    s/^(host\s*=\s*).*/\1db.production.example.com/
    s/^(port\s*=\s*).*/\15432/
}' config.ini

# CSV のカラム値をマスキング（3列目をマスク）
sed -E 's/^([^,]+,[^,]+,)[^,]+(.*)/\1****\2/' data.csv

# 複数ファイルに対する一括置換（安全なバックアップ付き）
find src/ -name '*.py' -exec sed -i.bak -E \
    's/from old_module import/from new_module import/g' {} +
# 確認後にバックアップを削除
find src/ -name '*.py.bak' -delete

# ログファイルの個人情報マスキング
sed -E '
    s/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[EMAIL MASKED]/g
    s/\b[0-9]{3}-[0-9]{4}-[0-9]{4}\b/[PHONE MASKED]/g
    s/\b[0-9]{1,3}(\.[0-9]{1,3}){3}\b/[IP MASKED]/g
' sensitive.log > sanitized.log
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

### 3.4 awk の組み込み変数と高度な機能

```bash
# 主要な組み込み変数
# NR:   現在の行番号（全入力通算）
# NF:   現在の行のフィールド数
# FNR:  現在のファイル内での行番号
# FS:   入力フィールドセパレータ
# OFS:  出力フィールドセパレータ
# RS:   レコードセパレータ
# ORS:  出力レコードセパレータ
# FILENAME: 現在処理中のファイル名

# OFS を指定して出力形式を制御
awk -F',' 'BEGIN {OFS="\t"} {print $1, $2, $3}' data.csv
# CSV → TSV に変換

# レコードセパレータの変更（段落単位の処理）
awk 'BEGIN {RS=""; FS="\n"} {print NR": "$1}' paragraphs.txt
# 空行で区切られた段落ごとに処理

# 複数ファイルの処理と FNR の活用
awk 'FNR==1 {print "=== " FILENAME " ==="} {print}' file1.txt file2.txt

# printf でフォーマット出力
awk '{printf "%-20s %10d %8.2f\n", $1, $2, $3}' data.txt

# 連想配列を使った複雑な集計
awk -F',' '{
    category = $1
    amount = $3
    total[category] += amount
    count[category]++
}
END {
    for (cat in total) {
        avg = total[cat] / count[cat]
        printf "%-15s 合計: %10.0f  平均: %8.0f  件数: %d\n", cat, total[cat], avg, count[cat]
    }
}' sales.csv
```

### 3.5 awk プログラミングの実践テクニック

```bash
# awk でヒストグラムを生成
awk '{
    len = length($0)
    bucket = int(len / 10) * 10
    hist[bucket]++
}
END {
    for (b in hist) {
        printf "%3d-%3d: ", b, b+9
        for (i = 0; i < hist[b]; i++) printf "#"
        printf " (%d)\n", hist[b]
    }
}' file.txt

# awk で Top-N 集計（ソートなしで実現）
awk '{
    count[$1]++
}
END {
    # 上位5件を取得
    for (i = 1; i <= 5; i++) {
        max_val = 0; max_key = ""
        for (k in count) {
            if (count[k] > max_val) {
                max_val = count[k]; max_key = k
            }
        }
        if (max_key != "") {
            printf "%5d %s\n", max_val, max_key
            delete count[max_key]
        }
    }
}' access.log

# awk でスライディングウィンドウ（移動平均）
awk '{
    window[NR % 5] = $1
    if (NR >= 5) {
        sum = 0
        for (i in window) sum += window[i]
        printf "%d: %.2f\n", NR, sum / 5
    }
}' numbers.txt

# awk で2つのファイルをJOIN
awk -F',' '
    NR==FNR { lookup[$1] = $2; next }
    { print $0, ($1 in lookup) ? lookup[$1] : "N/A" }
' master.csv detail.csv

# awk でトランザクションログのセッション分析
awk '
    /session_start/ {
        match($0, /session_id=([^ ]+)/, arr)
        sid = arr[1]
        start_time[sid] = $1
    }
    /session_end/ {
        match($0, /session_id=([^ ]+)/, arr)
        sid = arr[1]
        if (sid in start_time) {
            duration = $1 - start_time[sid]
            total_duration += duration
            session_count++
            if (duration > max_duration) max_duration = duration
        }
    }
    END {
        printf "セッション数: %d\n", session_count
        printf "平均時間: %.2f秒\n", total_duration / session_count
        printf "最大時間: %.2f秒\n", max_duration
    }
' transaction.log
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

### 4.4 Nginx ログの高度な解析

```bash
# Nginx のログフォーマット例:
# $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent
# "$http_referer" "$http_user_agent" $request_time

# リクエスト時間の分布をヒストグラム表示
awk '{
    time = $NF
    if (time < 0.1) bucket = "0-0.1s"
    else if (time < 0.5) bucket = "0.1-0.5s"
    else if (time < 1.0) bucket = "0.5-1.0s"
    else if (time < 5.0) bucket = "1.0-5.0s"
    else bucket = "5.0s+"
    count[bucket]++
}
END {
    order[1] = "0-0.1s"; order[2] = "0.1-0.5s"; order[3] = "0.5-1.0s"
    order[4] = "1.0-5.0s"; order[5] = "5.0s+"
    for (i = 1; i <= 5; i++) {
        b = order[i]
        printf "%-12s %6d ", b, count[b]+0
        for (j = 0; j < count[b] / 100; j++) printf "#"
        print ""
    }
}' access.log

# ユーザーエージェント別のアクセス集計
awk -F'"' '{print $6}' access.log | \
    sed -E 's/([^ ]+).*/\1/' | \
    sort | uniq -c | sort -rn | head -10

# リファラー別の流入分析
awk -F'"' '$4 !~ /^-$/ && $4 !~ /^$/ {print $4}' access.log | \
    awk -F'/' '{print $1"//"$3}' | \
    sort | uniq -c | sort -rn | head -10

# 5xx エラーの時系列推移（1分ごと）
awk '$9 >= 500 {
    match($0, /\[([0-9]+\/[A-Za-z]+\/[0-9]+:[0-9]+:[0-9]+)/, arr)
    print arr[1]
}' access.log | \
    sort | uniq -c | \
    awk '{printf "%s %5d ", $2, $1; for(i=0;i<$1;i++) printf "*"; print ""}'

# 特定のエンドポイントのレスポンス時間P50/P90/P99
awk '$7 == "/api/users" {print $NF}' access.log | \
    sort -n | awk '
    {vals[NR] = $1}
    END {
        n = NR
        printf "件数: %d\n", n
        printf "P50: %.3fs\n", vals[int(n*0.50)]
        printf "P90: %.3fs\n", vals[int(n*0.90)]
        printf "P99: %.3fs\n", vals[int(n*0.99)]
        printf "最大: %.3fs\n", vals[n]
    }'
```

### 4.5 複合的なログ解析パイプライン

```bash
# アクセスログからレポートを自動生成
cat > /tmp/log_report.sh << 'SCRIPT'
#!/bin/bash
LOG_FILE=${1:-/var/log/nginx/access.log}
echo "=== ログ解析レポート ==="
echo "対象: $LOG_FILE"
echo "生成日時: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "--- 総リクエスト数 ---"
wc -l < "$LOG_FILE"
echo ""

echo "--- ステータスコード別 ---"
awk '{print $9}' "$LOG_FILE" | sort | uniq -c | sort -rn
echo ""

echo "--- 上位10 IPアドレス ---"
awk '{print $1}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10
echo ""

echo "--- 上位10 URL ---"
awk '{print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10
echo ""

echo "--- 時間帯別アクセス数 ---"
awk -F'[\\[:]' '{print $3":00"}' "$LOG_FILE" | sort | uniq -c | \
    awk '{printf "%s %5d ", $2, $1; for(i=0;i<$1/50;i++) printf "#"; print ""}'
echo ""

echo "--- エラー率 ---"
awk '
    {total++; if ($9 >= 400) errors++}
    END {printf "総リクエスト: %d, エラー: %d, エラー率: %.2f%%\n", total, errors+0, (errors+0)*100/total}
' "$LOG_FILE"
SCRIPT
chmod +x /tmp/log_report.sh

# systemd ジャーナルのログ解析
journalctl -u nginx --since "1 hour ago" --no-pager | \
    grep -E 'error|warn' -i | \
    awk '{print $1, $2, $3}' | \
    sort | uniq -c | sort -rn

# 複数日分のログを結合して分析
zcat /var/log/nginx/access.log.*.gz | \
    cat - /var/log/nginx/access.log | \
    awk '$9 == 500 {print $7}' | \
    sort | uniq -c | sort -rn | head -20
```

### 4.6 セキュリティログの解析

```bash
# SSH ブルートフォース検出
grep 'Failed password' /var/log/auth.log | \
    awk '{print $(NF-3)}' | \
    sort | uniq -c | sort -rn | head -20

# 不正アクセスの兆候（SQLインジェクション試行）
grep -iE "(union\+select|or\+1=1|drop\+table|;--)" access.log | \
    awk '{print $1, $7}' | sort | uniq -c | sort -rn

# WAF ログからブロックされたリクエストの分析
grep 'BLOCKED' waf.log | \
    awk -F'|' '{print $3}' | \    # 攻撃カテゴリ
    sort | uniq -c | sort -rn

# ログイン試行の異常検出（同一IPから短時間に多数のログイン試行）
awk '/login_attempt/ {
    # タイムスタンプとIPを抽出
    match($0, /ip=([0-9.]+)/, ip_arr)
    match($0, /\[([0-9:]+)\]/, time_arr)
    ip = ip_arr[1]
    attempts[ip]++
}
END {
    for (ip in attempts) {
        if (attempts[ip] > 10) {
            printf "疑わしいIP: %-15s  試行回数: %d\n", ip, attempts[ip]
        }
    }
}' auth.log

# ファイルアクセス監査（auditd ログ）
grep 'type=SYSCALL' /var/log/audit/audit.log | \
    grep -E 'syscall=(2|257)' | \
    awk -F' ' '{
        for (i=1; i<=NF; i++) {
            if ($i ~ /^comm=/) comm = $i
            if ($i ~ /^uid=/) uid = $i
        }
        print uid, comm
    }' | sort | uniq -c | sort -rn | head -20
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

### 5.3 csvkit と miller による高度な CSV 処理

```bash
# === csvkit の活用 ===

# csvlook: CSVを見やすいテーブル形式で表示
csvlook data.csv

# csvcut: カラムの抽出（カラム名で指定可能）
csvcut -c name,age data.csv
csvcut -c 1,3 data.csv              # インデックスでも可

# csvgrep: CSVに対する grep（引用符を正しく処理）
csvgrep -c status -m 'active' data.csv
csvgrep -c age -r '^[3-4][0-9]$' data.csv  # 正規表現も使える

# csvsort: CSVのソート
csvsort -c age -r data.csv           # age列で降順ソート

# csvstat: 統計情報の表示
csvstat data.csv

# csvjoin: 2つのCSVをJOIN
csvjoin -c user_id users.csv orders.csv

# csvsql: CSVに対してSQLクエリを実行
csvsql --query "SELECT name, AVG(score) as avg_score \
    FROM data GROUP BY name HAVING avg_score > 80" data.csv

# csvformat: フォーマット変換
csvformat -T data.csv                # CSV → TSV
csvformat -D '|' data.csv           # CSV → パイプ区切り

# === miller (mlr) の活用 ===

# 基本的なフィルタリング
mlr --csv filter '$age > 25' data.csv

# カラムの選択
mlr --csv cut -f name,city data.csv

# カラムのリネーム
mlr --csv rename name,full_name data.csv

# グループ別集計
mlr --csv stats1 -a mean,count -f age -g city data.csv

# ソート
mlr --csv sort-by -nr age data.csv

# フォーマット変換
mlr --icsv --ojson cat data.csv          # CSV → JSON
mlr --icsv --opprint cat data.csv        # CSV → 整形テーブル
mlr --ijson --ocsv cat data.json         # JSON → CSV

# 複雑な変換パイプライン
mlr --csv \
    filter '$status == "active"' \
    then sort-by -nr revenue \
    then head -n 10 \
    then put '$revenue_formatted = format_values($revenue, "%,.0f")' \
    data.csv

# 時系列データの処理
mlr --csv \
    put '$date = strftime(strptime($timestamp, "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d")' \
    then group-by date \
    then stats1 -a sum,mean -f amount \
    transactions.csv
```

### 5.4 TSV/固定長/その他のフォーマット処理

```bash
# TSV 処理
awk -F'\t' '{print $1, $3}' data.tsv
awk -F'\t' 'BEGIN{OFS=","} {print $1,$2,$3}' data.tsv > data.csv

# 固定長レコードの処理
# 例: 名前(20文字) 年齢(3文字) 都市(15文字)
awk '{
    name = substr($0, 1, 20)
    age  = substr($0, 21, 3)
    city = substr($0, 24, 15)
    gsub(/[[:space:]]+$/, "", name)
    gsub(/[[:space:]]+$/, "", city)
    printf "%s,%s,%s\n", name, age+0, city
}' fixed_width.dat

# LTSV (Labeled Tab-Separated Values) の処理
# host:127.0.0.1\tident:-\tuser:-\ttime:[10/Oct/2000:13:55:36 -0700]
awk -F'\t' '{
    for (i=1; i<=NF; i++) {
        split($i, kv, ":")
        fields[kv[1]] = substr($i, length(kv[1])+2)
    }
    print fields["host"], fields["status"], fields["size"]
}' access.ltsv

# Apache ログを CSV に変換
sed -E 's/^([^ ]+) [^ ]+ [^ ]+ \[([^\]]+)\] "([^"]+)" ([0-9]+) ([0-9]+|-)/\1,\2,\3,\4,\5/' access.log

# JSON Lines を CSV に変換
jq -r '[.timestamp, .level, .message] | @csv' app.jsonl > app.csv
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

### 6.4 テキスト処理のデータフロー全体像

```
テキスト処理パイプラインの全体アーキテクチャ:

[入力ソース]                    [処理パイプライン]                [出力先]

 ログファイル ─┐              ┌──→ grep (フィルタ) ──┐         ┌→ ファイル
               │              │                       │         │
 標準入力 ───┤  → 入力選択 ──┤──→ sed  (変換)   ──┤→ 整形 ──┤→ 標準出力
               │              │                       │         │
 ネットワーク ─┤              └──→ awk  (集計)   ──┘         └→ パイプ
               │
 圧縮ファイル ─┘
   (zcat/zgrep)

データ変換のフロー例:

 Raw JSON Log                         構造化データ
 ─────────────                         ──────────
 {"ts":"...",     jq で            CSV/TSV で
  "level":"ERR",  フィールド  ──→   mlr で   ──→  レポート
  "msg":"..."}    抽出              集計・分析

 Apache Log                           統計情報
 ──────────                           ────────
 192.168.1.1 -    awk で           sort |
 [date] "GET      フィールド  ──→  uniq -c  ──→  ランキング
 /path" 200       抽出             sort -rn
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

### 7.3 ユースケース別の推奨ツール選択

| ユースケース | 最適なツール | 理由 |
|------------|------------|------|
| ソースコード検索 | ripgrep (rg) | .gitignore 対応、再帰デフォルト |
| ログのリアルタイム監視 | tail -f + grep | シンプル、軽量 |
| ログの集計・統計 | awk | 組み込みの計算機能 |
| 設定ファイルの一括変更 | sed -i | インプレース編集 |
| CSV の集計 | miller (mlr) | CSV ネイティブ対応 |
| JSON ログの解析 | jq | JSON のネイティブ対応 |
| 大規模データの並列処理 | GNU parallel + grep | 並列化で高速化 |
| 構造化ログの変換 | awk + jq | フィールド抽出 + 構造化変換 |
| バイナリファイルの検索 | grep -a / strings | バイナリ対応モード |
| マルチバイト文字列の処理 | grep -P / rg | Unicode 対応 |

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

### 8.3 アンチパターン: 不要な cat の使用 (Useless Use of Cat)

```bash
# NG: 不要な cat（UUOC）
cat file.txt | grep 'ERROR'
cat file.txt | awk '{print $1}'
cat file.txt | sed 's/old/new/g'

# OK: 直接ファイルを引数に渡す
grep 'ERROR' file.txt
awk '{print $1}' file.txt
sed 's/old/new/g' file.txt

# OK: リダイレクトを使う
grep 'ERROR' < file.txt
```

### 8.4 アンチパターン: sed で複雑なロジックを書く

```bash
# NG: sed で分岐やループを多用（保守困難）
sed -E '
    :start
    /\\$/ {
        N
        s/\\\n/ /
        b start
    }
    s/^[[:space:]]+//
    /^#/d
    /^$/d
    s/([^=]+)=([^;]+);?/\1 = "\2"\n/g
' complex_config.txt

# OK: Python や awk で明確に書く
awk '
    /\\$/ {
        # バックスラッシュ継続行を結合
        line = line substr($0, 1, length($0)-1)
        next
    }
    {
        line = line $0
        # コメントと空行を除外
        gsub(/^[[:space:]]+/, "", line)
        if (line !~ /^#/ && line != "") {
            print line
        }
        line = ""
    }
' complex_config.txt
```

### 8.5 アンチパターン: 正規表現を使ったバリデーションの過信

```bash
# NG: メールアドレスの「完全な」バリデーションを正規表現で試みる
# RFC 5322 準拠のメールアドレス正規表現は数千文字になる
grep -E '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' emails.txt
# → これは簡易チェックにすぎない（国際化ドメイン等に未対応）

# OK: 簡易チェック + 専用ライブラリでの検証
# grep で候補を絞り込み → 専用ライブラリで厳密に検証
grep -E '@.*\.' emails.txt | python3 -c "
import sys
from email_validator import validate_email, EmailNotValidError
for line in sys.stdin:
    email = line.strip()
    try:
        validate_email(email)
        print(f'VALID: {email}')
    except EmailNotValidError as e:
        print(f'INVALID: {email} ({e})')
"
```

### 8.6 アンチパターン: パイプラインの中間結果を確認しない

```bash
# NG: 長いパイプラインをデバッグなしで一気に書く
awk -F',' '{print $3}' data.csv | sed 's/"//g' | sort | uniq -c | sort -rn | head -5

# OK: tee コマンドで中間結果を確認
awk -F',' '{print $3}' data.csv | \
    tee /dev/stderr | \           # 中間結果を標準エラーに出力
    sed 's/"//g' | \
    tee /tmp/debug_step2.txt | \  # ファイルにも保存
    sort | uniq -c | sort -rn | head -5

# OK: 段階的にパイプラインを構築
# Step 1: awk の出力を確認
awk -F',' '{print $3}' data.csv | head -5
# Step 2: sed を追加
awk -F',' '{print $3}' data.csv | sed 's/"//g' | head -5
# Step 3: sort と集計を追加
awk -F',' '{print $3}' data.csv | sed 's/"//g' | sort | uniq -c | sort -rn | head -5
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

### Q5: 大規模なログファイル（数十GB）を処理する場合のコツは？

**A**: 以下のアプローチを組み合わせる:

```bash
# 1. LC_ALL=C でロケール処理を省略（2-3倍高速化）
LC_ALL=C grep 'ERROR' huge.log

# 2. grep -m で早期終了
LC_ALL=C grep -m 1000 'ERROR' huge.log    # 最初の1000件で停止

# 3. GNU parallel で並列処理
parallel --pipepart -a huge.log --block 100M grep 'ERROR'

# 4. split + parallel で分割処理
split -l 1000000 huge.log /tmp/chunk_
ls /tmp/chunk_* | parallel "grep -c 'ERROR' {}" | awk '{sum+=$1} END{print sum}'

# 5. 圧縮ファイルの直接処理
zgrep 'ERROR' huge.log.gz           # gzip のまま検索
bzgrep 'ERROR' huge.log.bz2        # bzip2 のまま検索
xzgrep 'ERROR' huge.log.xz         # xz のまま検索

# 6. awk の方が grep より効率的な場合
# 複数の条件を1パスで処理する場合は awk が有利
awk '/ERROR/{e++} /WARN/{w++} END{print e+0, w+0}' huge.log
```

### Q6: grep/sed/awk でマルチバイト文字（日本語等）を正しく処理するには？

**A**: ロケール設定と文字コードに注意する:

```bash
# ロケールの確認
locale

# UTF-8 環境で日本語を検索
grep '東京' data.txt

# 文字クラスに注意（ロケール依存）
grep '[[:alpha:]]' data.txt     # ロケールに応じた「文字」にマッチ
grep '[a-zA-Z]' data.txt        # ASCII のみ

# sed での日本語置換
sed 's/東京都/東京/g' addresses.txt

# awk での日本語処理
awk '/東京/ {count++} END {print count+0}' data.txt

# 文字コード変換と組み合わせ
# Shift_JIS → UTF-8 変換してから処理
iconv -f SHIFT_JIS -t UTF-8 sjis_file.txt | grep 'パターン'

# nkf を使った文字コード変換
nkf -w sjis_file.txt | grep 'パターン'
```

### Q7: awk のスクリプトが長くなった場合、ファイルに分離すべきか？

**A**: 10行を超える awk スクリプトは **ファイルに分離することを推奨** する:

```bash
# awk スクリプトファイル: analyze.awk
cat > analyze.awk << 'AWK'
BEGIN {
    FS = ","
    OFS = "\t"
    print "カテゴリ", "件数", "合計", "平均"
}
NR > 1 {
    category = $1
    amount = $3
    count[category]++
    total[category] += amount
}
END {
    for (cat in count) {
        avg = total[cat] / count[cat]
        printf "%-15s\t%d\t%.0f\t%.0f\n", cat, count[cat], total[cat], avg
    }
}
AWK

# 実行
awk -f analyze.awk data.csv

# メリット:
# - バージョン管理しやすい
# - エディタのシンタックスハイライトが効く
# - テスト可能
# - 再利用しやすい
```

### Q8: sed と perl -pe はどちらを使うべきか？

**A**: 基本的な置換は sed、複雑なパターンは perl -pe を推奨する:

```bash
# sed で十分な例
sed 's/old/new/g' file.txt
sed '/pattern/d' file.txt

# perl -pe の方が適切な例
# ゼロ幅アサーション（先読み・後読み）
perl -pe 's/(?<=\$)\d+/XXX/g' file.txt

# 非貪欲マッチ
perl -pe 's/<.*?>/[TAG]/g' file.html

# 複数行マッチ
perl -0pe 's/start.*?end/REPLACED/gs' file.txt

# 計算結果での置換
perl -pe 's/(\d+)/sprintf("%05d", $1)/ge' file.txt
```

---

## 10. 実務シナリオ集

### 10.1 障害調査のためのログ解析

```bash
# シナリオ: Webアプリケーションで500エラーが急増、原因を調査

# Step 1: エラーの発生頻度を時系列で確認
awk '$9 == 500 {
    match($0, /\[([0-9]+\/[A-Za-z]+\/[0-9]+:[0-9]+:[0-9]+)/, arr)
    print arr[1]
}' access.log | sort | uniq -c | tail -20

# Step 2: エラーが発生しているエンドポイントを特定
awk '$9 == 500 {print $7}' access.log | sort | uniq -c | sort -rn | head -10

# Step 3: エラーリクエストのIPアドレスを確認（特定IPからの攻撃かどうか）
awk '$9 == 500 {print $1}' access.log | sort | uniq -c | sort -rn | head -10

# Step 4: アプリケーションログとの突合
# access.log のタイムスタンプから app.log の対応するエラーを検索
awk '$9 == 500 {
    match($4, /([0-9]+:[0-9]+:[0-9]+)/, arr)
    print arr[1]
}' access.log | head -5 | while read ts; do
    grep "$ts" app.log | grep -i 'error\|exception\|traceback'
done

# Step 5: エラーの根本原因をパターン分類
grep -A 5 'ERROR' app.log | \
    grep -E 'Exception|Error' | \
    sed -E 's/^.*: //' | \
    sort | uniq -c | sort -rn | head -10
```

### 10.2 デプロイ前のコード品質チェック

```bash
# TODO/FIXME/HACK のリストアップ
rg -n 'TODO|FIXME|HACK|XXX' --type py src/ | \
    awk -F: '{printf "%-40s %s: %s\n", $1, $2, $3}'

# デバッグ用 print 文の検出
rg -n '^\s*(print\(|console\.log|System\.out\.print)' src/

# ハードコードされた認証情報の検出
rg -in '(password|secret|api_key|token)\s*=\s*["\x27][^"\x27]+["\x27]' \
    --type py --type js --type ts src/

# 未使用の import の検出（Python）
for f in $(find src/ -name '*.py'); do
    awk '
        /^import / { modules[$2] = NR }
        /^from .* import / {
            split($0, a, "import ")
            split(a[2], b, ",")
            for (i in b) {
                gsub(/[[:space:]]/, "", b[i])
                modules[b[i]] = NR
            }
        }
        !/^import |^from / {
            for (m in modules) {
                if (index($0, m) > 0) delete modules[m]
            }
        }
        END {
            for (m in modules) print FILENAME":"modules[m]": unused import: "m
        }
    ' "$f"
done

# 長すぎる行の検出
awk 'length > 120 {printf "%s:%d: 行長 %d文字\n", FILENAME, NR, length}' src/*.py
```

### 10.3 データマイグレーションの前処理

```bash
# シナリオ: 旧システムのCSVデータを新システムに移行

# Step 1: データの概要確認
head -1 old_system.csv                    # ヘッダー確認
wc -l old_system.csv                      # 行数
awk -F',' '{print NF}' old_system.csv | sort -u  # フィールド数の確認

# Step 2: データ品質チェック
# 空フィールドの検出
awk -F',' '{
    for (i=1; i<=NF; i++) {
        if ($i == "" || $i == "NULL" || $i == "null") {
            empty[i]++
        }
    }
    total++
}
END {
    for (i in empty) {
        printf "列%d: %d件が空 (%.1f%%)\n", i, empty[i], empty[i]*100/total
    }
}' old_system.csv

# Step 3: 日付フォーマットの統一
sed -E '
    # MM/DD/YYYY → YYYY-MM-DD
    s,([0-9]{2})/([0-9]{2})/([0-9]{4}),\3-\1-\2,g
    # DD-Mon-YYYY → YYYY-MM-DD (簡易版)
    s/Jan/01/g; s/Feb/02/g; s/Mar/03/g; s/Apr/04/g
    s/May/05/g; s/Jun/06/g; s/Jul/07/g; s/Aug/08/g
    s/Sep/09/g; s/Oct/10/g; s/Nov/11/g; s/Dec/12/g
' old_system.csv > normalized_dates.csv

# Step 4: 電話番号の正規化
sed -E '
    s/\+81-?/0/g           # 国際番号を国内番号に
    s/[()-]//g              # 記号を除去
    s/([0-9]{3})([0-9]{4})([0-9]{4})/\1-\2-\3/g  # ハイフン挿入
' normalized_dates.csv > normalized_phones.csv

# Step 5: 重複レコードの検出
awk -F',' 'NR>1 {
    key = $1","$2","$3    # 名前+メール+電話で重複判定
    if (key in seen) {
        print "重複: 行"seen[key]" と 行"NR": "$0
    } else {
        seen[key] = NR
    }
}' old_system.csv
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
| パフォーマンス | LC_ALL=C、-F オプション、parallel で大規模データに対応 |
| 安全性 | バックアップ付き編集、段階的パイプライン構築を徹底 |

## 次に読むべきガイド

- [03-regex-alternatives.md](./03-regex-alternatives.md) -- 正規表現の代替技術
- [../01-advanced/03-performance.md](../01-advanced/03-performance.md) -- パフォーマンス最適化

## 参考文献

1. **Dale Dougherty & Arnold Robbins** "sed & awk, 2nd Edition" O'Reilly, 1997 -- sed/awk の定番書
2. **GNU Grep Manual** https://www.gnu.org/software/grep/manual/ -- GNU grep の公式リファレンス
3. **The AWK Programming Language** Aho, Kernighan, Weinberger, 2024 (2nd Edition) -- awk 原作者による改訂版
4. **ripgrep** https://github.com/BurntSushi/ripgrep -- モダンな grep 代替ツール
5. **Miller (mlr)** https://miller.readthedocs.io/ -- CSV/JSON/TSV 処理のスイスアーミーナイフ
6. **csvkit** https://csvkit.readthedocs.io/ -- CSV 処理のコマンドラインツールキット
7. **jq Manual** https://stedolan.github.io/jq/manual/ -- JSON 処理の定番ツール
8. **GNU Parallel** https://www.gnu.org/software/parallel/ -- コマンドの並列実行ツール
