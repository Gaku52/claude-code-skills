# テキスト処理言語（awk）

> awk は「構造化テキストを列単位で処理する」ミニプログラミング言語。

## この章で学ぶこと

- [ ] awk の基本構文を理解する
- [ ] フィールド操作と集計ができる
- [ ] 組み込み変数と関数を使いこなせる
- [ ] 条件分岐・ループ・配列を活用できる
- [ ] 実務でのログ分析・データ加工パターンを身につける
- [ ] gawk（GNU awk）の拡張機能を理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [ストリームエディタ（sed）](./02-sed.md) の内容を理解していること

---

## 1. awk の基本

### 1.1 基本構文と動作原理

```bash
# 基本構文: awk 'パターン { アクション }' [ファイル...]
#
# awk の動作原理:
# 1. 入力を1行ずつ読み込む（レコード）
# 2. レコードをフィールド（列）に分割する
# 3. パターンに一致するレコードに対してアクションを実行する
# 4. 次の行へ進み、1に戻る
#
# 特殊パターン:
# BEGIN { ... }  入力処理の前に1回実行
# END { ... }    入力処理の後に1回実行
# パターン省略   全行に対してアクションを実行
# アクション省略 マッチした行を表示（print $0 と同等）

# 基本的な使い方
awk '{print}' file.txt                # 全行を表示（cat と同等）
awk '{print $0}' file.txt             # 同上（$0 = 行全体）
awk '{print $1}' file.txt             # 1列目を表示
awk '{print $1, $3}' file.txt         # 1列目と3列目を表示

# パイプからの入力
echo "hello world" | awk '{print $2}' # → world
ps aux | awk '{print $1, $11}'        # プロセスの所有者とコマンド
ls -la | awk '{print $5, $9}'         # ファイルサイズと名前
```

### 1.2 フィールド（列）の基本

```bash
# デフォルトの区切り文字は連続する空白（スペース/タブ）
# $0: 行全体
# $1: 1番目のフィールド（列）
# $2: 2番目のフィールド
# $NF: 最後のフィールド
# $(NF-1): 最後から2番目のフィールド

awk '{print $1}' file.txt              # 1列目を表示
awk '{print $2}' file.txt              # 2列目を表示
awk '{print $NF}' file.txt             # 最終列を表示
awk '{print $(NF-1)}' file.txt         # 最後から2番目の列
awk '{print NR, $0}' file.txt          # 行番号+全体を表示
awk '{print NR": "$0}' file.txt        # 行番号: 行内容

# フィールドの計算
awk '{print $1, $2, $1+$2}' data.txt   # 1列目、2列目、合計
awk '{print $1, $2*100}' data.txt      # 2列目を100倍

# フィールドの連結
awk '{print $1 "-" $2}' file.txt       # ハイフンで連結
awk '{print $1 "," $2 "," $3}' file.txt  # カンマで連結

# フィールド数の確認
awk '{print NF}' file.txt              # 各行のフィールド数
awk 'NF > 0' file.txt                  # 空行以外を表示（フィールド数 > 0）
awk 'NF == 5' file.txt                 # ちょうど5列の行のみ
```

### 1.3 区切り文字の指定（-F / FS）

```bash
# -F オプションで入力区切り文字を指定
awk -F',' '{print $2}' data.csv        # CSV の2列目
awk -F':' '{print $1}' /etc/passwd     # ユーザー名一覧
awk -F'\t' '{print $1}' data.tsv       # TSV の1列目
awk -F'|' '{print $2}' data.txt        # パイプ区切り
awk -F'/' '{print $NF}' paths.txt      # パスの最後の部分

# 正規表現を区切り文字に
awk -F'[,;]' '{print $2}' file.txt     # カンマまたはセミコロン
awk -F'[=:]' '{print $1, $2}' config.txt   # = または : で分割

# BEGIN ブロックで FS を設定
awk 'BEGIN{FS=","} {print $2}' data.csv
awk 'BEGIN{FS=":"} {print $1, $NF}' /etc/passwd

# 出力区切り文字（OFS）の指定
awk -F',' 'BEGIN{OFS="\t"} {print $1, $2, $3}' data.csv
# → CSV をTSVに変換

awk -F':' 'BEGIN{OFS=","} {print $1, $3, $6}' /etc/passwd
# → ユーザー名、UID、ホームディレクトリをCSVで出力

# 複数文字の区切り
awk -F'::' '{print $2}' file.txt       # :: で分割
awk -F' -> ' '{print $2}' file.txt     # " -> " で分割
```

---

## 2. パターンマッチング

### 2.1 条件式によるフィルタリング

```bash
# 比較演算子
awk '$3 > 100' data.txt                # 3列目が100より大きい行
awk '$3 >= 100' data.txt               # 3列目が100以上の行
awk '$3 < 50' data.txt                 # 3列目が50未満の行
awk '$3 == 100' data.txt               # 3列目がちょうど100の行
awk '$3 != 0' data.txt                 # 3列目が0でない行
awk '$1 == "admin"' users.txt          # 1列目が "admin" の行
awk '$1 != "root"' /etc/passwd         # 1列目が "root" でない行

# 文字列比較
awk '$1 > "M"' file.txt                # 1列目がM以降（辞書順）
awk '$2 == ""' file.txt                # 2列目が空の行

# 行番号による選択
awk 'NR >= 10 && NR <= 20' file.txt    # 10〜20行目
awk 'NR == 1' file.txt                 # 1行目のみ
awk 'NR > 1' file.txt                  # 2行目以降（ヘッダースキップ）
awk 'NR % 2 == 0' file.txt             # 偶数行のみ
awk 'NR % 2 == 1' file.txt             # 奇数行のみ

# 論理演算子
awk '$3 > 50 && $3 < 100' data.txt     # 50 < 3列目 < 100
awk '$1 == "error" || $1 == "fatal"' log.txt  # error OR fatal
awk '!($3 > 100)' data.txt             # 3列目が100以下（NOT）
```

### 2.2 正規表現によるフィルタリング

```bash
# ~ : 正規表現マッチ
# !~ : 正規表現非マッチ

awk '/error/' logfile.txt              # error を含む行（grep と同等）
awk '/^#/' config.txt                  # # で始まる行
awk '/error|warning/' logfile.txt      # error または warning を含む行
awk '!/^#/' config.txt                 # # で始まらない行
awk '/^$/' file.txt                    # 空行のみ
awk '!/^$/' file.txt                   # 空行以外

# フィールドに対する正規表現
awk '$1 ~ /^[0-9]+$/' file.txt         # 1列目が数字のみの行
awk '$2 ~ /error/' logfile.txt         # 2列目に error を含む行
awk '$NF !~ /\.log$/' file.txt         # 最終列が .log で終わらない行
awk '$3 ~ /^[A-Z]/' file.txt           # 3列目が大文字で始まる行

# 範囲パターン（sed と同様の開始/終了パターン）
awk '/BEGIN/,/END/' file.txt           # BEGIN〜END の範囲の行
awk '/^<body>/,/^<\/body>/' file.html  # <body>〜</body> の範囲
awk 'NR==5,NR==10' file.txt            # 5〜10行目

# POSIX 文字クラス
```

---

## 3. 組み込み変数

### 3.1 主要な組み込み変数

```bash
# === レコード・フィールド関連 ===
# $0     : 現在の行全体
# $1〜$N : N番目のフィールド
# NR     : 現在の行番号（全ファイル通算）
# NF     : 現在の行のフィールド数
# FNR    : 現在のファイル内の行番号
# FILENAME: 現在処理中のファイル名

# === 区切り文字関連 ===
# FS     : 入力フィールドセパレータ（デフォルト: 空白）
# OFS    : 出力フィールドセパレータ（デフォルト: スペース）
# RS     : 入力レコードセパレータ（デフォルト: 改行）
# ORS    : 出力レコードセパレータ（デフォルト: 改行）

# === その他 ===
# OFMT   : 数値の出力フォーマット（デフォルト: "%.6g"）
# RSTART : match() でマッチした開始位置
# RLENGTH: match() でマッチした長さ
# ARGC   : コマンドライン引数の数
# ARGV   : コマンドライン引数の配列

# 使用例
awk '{print NR, NF, $0}' file.txt      # 行番号、列数、行内容
awk 'END{print NR}' file.txt           # 総行数
awk 'END{print NR " lines"}' file.txt  # 総行数（ラベル付き）

# 複数ファイルでの NR と FNR の違い
awk '{print FILENAME, FNR, NR, $0}' file1.txt file2.txt
# FILENAME: ファイル名
# FNR: ファイル内の行番号（ファイルごとにリセット）
# NR: 通算行番号（リセットされない）

# ファイルの切り替え検知
awk 'FNR==1{print "=== " FILENAME " ==="}; {print}' *.txt
```

### 3.2 区切り文字のカスタマイズ

```bash
# 出力区切り文字の変更
awk 'BEGIN{OFS=","} {print $1, $2, $3}' file.txt
# → 出力をカンマ区切りに

awk 'BEGIN{OFS="\t"} {print $1, $2}' file.txt
# → 出力をタブ区切りに

# フィールドを再割り当てすると OFS が適用される
awk -F',' 'BEGIN{OFS="|"} {$1=$1; print}' data.csv
# → CSV をパイプ区切りに変換

# レコードセパレータの変更（段落モード）
awk 'BEGIN{RS=""} {print NR, $0}' file.txt
# → 空行で区切られた段落単位で処理

# 複数文字のレコードセパレータ（gawk）
awk 'BEGIN{RS="---\n"} {print NR": "$0}' file.txt
# → --- で区切られたブロック単位で処理

# 出力レコードセパレータの変更
awk 'BEGIN{ORS="\n\n"} {print}' file.txt
# → 各行の後に空行を挿入

# CSV → TSV 変換
awk -F',' 'BEGIN{OFS="\t"} {$1=$1; print}' data.csv > data.tsv

# TSV → CSV 変換
awk -F'\t' 'BEGIN{OFS=","} {$1=$1; print}' data.tsv > data.csv
```

---

## 4. 集計と計算

### 4.1 合計・平均・最大・最小

```bash
# 合計
awk '{sum += $2} END {print "合計:", sum}' data.txt
awk -F',' '{sum += $3} END {print sum}' data.csv

# 平均
awk '{sum += $2; n++} END {print "平均:", sum/n}' data.txt
awk '{sum += $2} END {print "平均:", sum/NR}' data.txt  # NR を使う方法

# 最大値
awk 'BEGIN{max=-999999} $2>max{max=$2} END{print "最大:", max}' data.txt
awk 'NR==1{max=$2} $2>max{max=$2} END{print max}' data.txt

# 最小値
awk 'BEGIN{min=999999} $2<min{min=$2} END{print "最小:", min}' data.txt
awk 'NR==1{min=$2} $2<min{min=$2} END{print min}' data.txt

# 合計・平均・最大・最小をまとめて
awk 'NR==1{min=max=$2}
     {sum+=$2; if($2>max)max=$2; if($2<min)min=$2}
     END{printf "合計: %d\n平均: %.2f\n最大: %d\n最小: %d\n", sum, sum/NR, max, min}' data.txt

# 条件付き合計
awk '$1=="sales" {sum += $3} END {print sum}' data.txt
# → 1列目が "sales" の行の3列目を合計

# 累積和
awk '{sum += $1; print sum}' data.txt
# → 各行で累積和を表示

# 移動平均（直近N個）
awk '{a[NR%5]=$1; s=0; for(i in a)s+=a[i]; print s/(NR<5?NR:5)}' data.txt
```

### 4.2 カウントと頻度分析

```bash
# 単純カウント
awk '{count[$1]++} END {for (k in count) print k, count[k]}' access.log
# → 1列目（IPアドレス等）ごとの出現回数

# ソート付きカウント（awk + sort）
awk '{count[$1]++} END {for (k in count) print count[k], k}' access.log | sort -rn
# → 出現回数の降順

# 特定フィールドのユニーク値カウント
awk '{seen[$3]=1} END {print length(seen) " unique values"}' data.txt
# → 3列目のユニークな値の数

# ユニーク値の一覧
awk '!seen[$1]++' file.txt             # 1列目の重複を除去（最初の出現を保持）
awk '!seen[$0]++' file.txt             # 行全体の重複を除去（sort | uniq と同等）

# グループ別の集計
awk '{sum[$1] += $2; count[$1]++}
     END {for (k in sum) printf "%s: total=%d, avg=%.2f\n", k, sum[k], sum[k]/count[k]}' data.txt
# → 1列目のカテゴリごとに合計と平均

# ヒストグラム風の出力
awk '{count[$1]++}
     END {for (k in count) {
         printf "%-20s ", k
         for (i=0; i<count[k]; i++) printf "#"
         printf " (%d)\n", count[k]
     }}' data.txt

# クロス集計（2つのフィールドの組み合わせ）
awk '{count[$1","$2]++}
     END {for (k in count) print k, count[k]}' data.txt | sort
```

### 4.3 統計計算

```bash
# 標準偏差
awk '{sum+=$1; sumsq+=$1*$1}
     END{mean=sum/NR; variance=sumsq/NR-mean*mean;
         printf "Mean: %.2f\nStdDev: %.2f\n", mean, sqrt(variance)}' data.txt

# パーセンタイル（簡易版 - ソート済みデータ）
sort -n data.txt | awk '{a[NR]=$1}
    END{print "25th:", a[int(NR*0.25)];
        print "50th:", a[int(NR*0.50)];
        print "75th:", a[int(NR*0.75)];
        print "90th:", a[int(NR*0.90)]}'

# 度数分布（ビン幅10）
awk '{bin=int($1/10)*10; count[bin]++}
     END{for(b in count) printf "%3d-%3d: %d\n", b, b+9, count[b]}' data.txt | sort -n

# 百分率の計算
awk '{count[$1]++; total++}
     END{for(k in count) printf "%-15s %5d (%5.1f%%)\n", k, count[k], count[k]*100/total}' data.txt
```

---

## 5. 出力フォーマット

### 5.1 print と printf

```bash
# print: 簡単な出力（OFS で区切り、ORS で改行）
awk '{print $1, $2}' file.txt          # フィールドをOFSで区切って出力
awk '{print $1 $2}' file.txt           # フィールドを連結して出力（区切りなし）
awk '{print $1 " - " $2}' file.txt     # 文字列で連結

# printf: C 言語スタイルのフォーマット出力（改行は自動付加されない）
awk '{printf "%s\n", $1}' file.txt                    # 文字列
awk '{printf "%d\n", $1}' file.txt                    # 整数
awk '{printf "%.2f\n", $1}' file.txt                  # 小数2桁
awk '{printf "%10d\n", $1}' file.txt                  # 右寄せ10桁
awk '{printf "%-10s %5d\n", $1, $2}' file.txt         # 左寄せ10桁、右寄せ5桁
awk '{printf "%05d\n", $1}' file.txt                  # ゼロ埋め5桁
awk '{printf "%-20s %10.2f\n", $1, $2}' data.txt      # 表形式

# printf のフォーマット指定子
# %s   : 文字列
# %d   : 10進整数
# %f   : 浮動小数点数
# %e   : 指数表記
# %g   : %f または %e のコンパクトな方
# %x   : 16進数
# %o   : 8進数
# %c   : 文字（ASCII値から）
# %%   : リテラルの %
# %10s : 右寄せ10桁
# %-10s: 左寄せ10桁
# %05d : ゼロ埋め5桁
# %.2f : 小数点以下2桁

# テーブル形式の出力
awk 'BEGIN{printf "%-15s %10s %10s\n", "Name", "Score", "Grade";
           printf "%-15s %10s %10s\n", "----", "-----", "-----"}
     {printf "%-15s %10d %10s\n", $1, $2, $3}' scores.txt

# CSV 形式の出力
awk '{printf "\"%s\",\"%s\",%d\n", $1, $2, $3}' data.txt

# ヘッダー付きレポート
awk 'BEGIN{print "====== Report ======"}
     {printf "  %-20s: %s\n", $1, $2}
     END{print "==== End Report ===="}' data.txt
```

### 5.2 出力リダイレクト

```bash
# awk 内でのファイル出力
awk '{print $0 > "output.txt"}' input.txt          # ファイルに書き出し
awk '{print $0 >> "output.txt"}' input.txt         # ファイルに追記
awk '{print $0 | "sort"}' input.txt                # コマンドにパイプ

# 条件によるファイル分割
awk '{print > $1".txt"}' data.txt
# → 1列目の値をファイル名として各行を振り分け

# ログレベルによるファイル分割
awk '/ERROR/{print > "error.log"} /WARN/{print > "warn.log"} /INFO/{print > "info.log"}' app.log

# ファイルの閉じ方（大量のファイルを開く場合）
awk '{filename = $1".txt"; print >> filename; close(filename)}' data.txt
```

---

## 6. 条件分岐とループ

### 6.1 if-else 文

```bash
# 基本的な if-else
awk '{
    if ($3 >= 90) grade = "A"
    else if ($3 >= 80) grade = "B"
    else if ($3 >= 70) grade = "C"
    else if ($3 >= 60) grade = "D"
    else grade = "F"
    print $1, $3, grade
}' scores.txt

# 三項演算子
awk '{print $1, ($2 > 0 ? "positive" : "non-positive")}' data.txt
awk '{status = ($3 >= 200 && $3 < 300) ? "OK" : "ERROR"; print $0, status}' access.log

# 条件付き出力
awk '{
    if ($1 == "ERROR") {
        printf "\033[31m%s\033[0m\n", $0   # 赤色で表示
    } else if ($1 == "WARN") {
        printf "\033[33m%s\033[0m\n", $0   # 黄色で表示
    } else {
        print $0
    }
}' logfile.txt
```

### 6.2 ループ処理

```bash
# for ループ
awk '{for (i=1; i<=NF; i++) print $i}' file.txt   # 全フィールドを1行ずつ表示
awk '{for (i=NF; i>=1; i--) printf "%s ", $i; printf "\n"}' file.txt  # フィールドを逆順

# while ループ
awk '{i=1; while(i<=NF) {print $i; i++}}' file.txt

# do-while ループ
awk '{i=1; do {print $i; i++} while(i<=NF)}' file.txt

# 配列の for-in ループ
awk '{count[$1]++} END {for (key in count) print key, count[key]}' data.txt

# フィールドの結合（特定のフィールド以降を全て結合）
awk '{result=""; for(i=3;i<=NF;i++) result=result" "$i; print $1, $2, result}' file.txt

# 行の反転（フィールド順を逆にする）
awk '{for(i=NF;i>0;i--) printf "%s%s", $i, (i==1?"\n":OFS)}' file.txt

# 九九表の生成
awk 'BEGIN{for(i=1;i<=9;i++){for(j=1;j<=9;j++)printf "%3d",i*j;print""}}'
```

---

## 7. 連想配列（Associative Arrays）

### 7.1 基本的な配列操作

```bash
# awk の配列は全て連想配列（ハッシュマップ）
# インデックスは文字列（数値も文字列に変換される）

# 基本的なカウント
awk '{count[$1]++}
     END {for (k in count) print k, count[k]}' file.txt

# 配列の存在チェック
awk '{if ($1 in seen) print "duplicate:", $0; seen[$1]=1}' file.txt
# → 重複する1列目を検出

# 配列の削除
awk '{count[$1]++}
     END {delete count["unwanted"]; for(k in count) print k, count[k]}' file.txt

# 多次元配列（疑似的にキーを連結）
awk '{count[$1","$2]++}
     END {for (k in count) print k, count[k]}' data.txt

# gawk の多次元配列
# gawk '{count[$1][$2]++} ...'  # gawk 4.0+ で使用可能

# 配列のソート（gawk の asorti / asort）
awk '{count[$1]++}
     END {
         n = asorti(count, sorted)
         for (i=1; i<=n; i++) print sorted[i], count[sorted[i]]
     }' data.txt
```

### 7.2 配列を使った実務パターン

```bash
# 2つのファイルの結合（JOIN 操作）
awk 'NR==FNR{a[$1]=$2; next} ($1 in a){print $0, a[$1]}' file1.txt file2.txt
# → file1.txt の1列目をキーに、file2.txt と結合

# マスターデータとの照合
awk 'NR==FNR{master[$1]=1; next} !($1 in master){print "Not found:", $0}' master.txt data.txt
# → master.txt に存在しないデータを検出

# 重複チェック
awk 'seen[$0]++ > 0 {print NR": "$0}' file.txt
# → 重複行とその行番号を表示

# フィールド値の集約（GROUP BY 的な操作）
awk -F',' '{
    key = $1
    values[key] = values[key] ? values[key] ", " $2 : $2
}
END {
    for (k in values) print k ": " values[k]
}' data.csv
# → 1列目でグループ化し、2列目の値をカンマ区切りで集約

# ランキング（降順）
awk '{count[$1]++}
     END{
         for(k in count) print count[k], k
     }' data.txt | sort -rn | awk '{printf "%2d. %-20s %d\n", NR, $2, $1}'

# 前の行との比較（差分計算）
awk 'NR>1{print $0, $2-prev} {prev=$2}' data.txt
# → 2列目の前行との差分を表示

# 逆引き（値からキーを検索）
awk '{inv[$2]=$1} END{print inv["target_value"]}' data.txt
```

---

## 8. 組み込み関数

### 8.1 文字列関数

```bash
# length(): 文字列の長さ
awk '{print length($0)}' file.txt          # 各行の文字数
awk 'length($0) > 80' file.txt             # 80文字超の行
awk '{print length($1), $1}' file.txt      # 1列目の長さと値

# substr(): 部分文字列の抽出
awk '{print substr($1, 1, 3)}' file.txt    # 1列目の先頭3文字
awk '{print substr($0, 10)}' file.txt      # 10文字目以降
awk '{print substr($0, 5, 10)}' file.txt   # 5文字目から10文字

# index(): 文字列の検索（位置を返す、見つからなければ0）
awk '{pos = index($0, "error"); if(pos) print NR, pos, $0}' file.txt

# split(): 文字列を分割して配列に格納
awk '{n = split($1, arr, "-"); print arr[1], arr[2]}' file.txt
# → "2026-02-16" → "2026" "02"

awk '{n = split($0, arr, ","); for(i=1;i<=n;i++) print arr[i]}' csv_line.txt

# sub(): 最初のマッチを置換
awk '{sub(/error/, "ERROR"); print}' file.txt

# gsub(): 全てのマッチを置換（global sub）
awk '{gsub(/error/, "ERROR"); print}' file.txt
awk '{gsub(/,/, "\t"); print}' data.csv    # カンマをタブに置換
awk '{n = gsub(/e/, "E"); print n, $0}' file.txt  # 置換回数を取得

# match(): 正規表現マッチ（RSTART, RLENGTH を設定）
awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH)}' file.txt
# → 最初の数字列を抽出

# sprintf(): フォーマット文字列を変数に格納
awk '{result = sprintf("%s: %.2f", $1, $2); print result}' data.txt

# tolower() / toupper(): 大文字/小文字変換
awk '{print tolower($0)}' file.txt         # 全て小文字に
awk '{print toupper($1)}' file.txt         # 1列目を大文字に
awk '{$1=toupper($1); print}' file.txt     # 1列目だけ大文字に変換して出力
```

### 8.2 数値関数

```bash
# int(): 整数部分を取得（切り捨て）
awk '{print int($1)}' file.txt             # 小数を切り捨て
awk '{print int($1 + 0.5)}' file.txt       # 四捨五入

# sqrt(): 平方根
awk '{print sqrt($1)}' file.txt

# sin(), cos(), atan2(): 三角関数
awk 'BEGIN{print sin(3.14159/2)}'          # → 1
awk 'BEGIN{pi=atan2(0,-1); print pi}'      # → 3.14159

# exp(), log(): 指数・対数
awk 'BEGIN{print exp(1)}'                  # → 2.71828（e）
awk 'BEGIN{print log(2.71828)}'            # → 1

# rand(), srand(): 乱数
awk 'BEGIN{srand(); for(i=1;i<=10;i++) print rand()}'
# → 0〜1 の乱数を10個生成

awk 'BEGIN{srand(); for(i=1;i<=10;i++) print int(rand()*100)+1}'
# → 1〜100 の整数乱数を10個生成

# ランダムサンプリング（全行の10%を抽出）
awk 'BEGIN{srand()} rand() < 0.1' large_file.txt
```

### 8.3 時間関数（gawk）

```bash
# systime(): 現在のUNIXタイムスタンプ
gawk 'BEGIN{print systime()}'

# mktime(): 日時文字列からタイムスタンプ
gawk 'BEGIN{print mktime("2026 02 16 12 00 00")}'

# strftime(): タイムスタンプから日時文字列
gawk 'BEGIN{print strftime("%Y-%m-%d %H:%M:%S", systime())}'

# 日付の計算
gawk 'BEGIN{
    now = systime()
    yesterday = now - 86400
    print "Today:", strftime("%Y-%m-%d", now)
    print "Yesterday:", strftime("%Y-%m-%d", yesterday)
}'

# ログのタイムスタンプを変換
gawk '{
    timestamp = mktime(gensub(/[-:]/, " ", "g", $1 " " $2))
    print strftime("%s", timestamp), $0
}' logfile.txt
```

---

## 9. awk スクリプト

### 9.1 awk スクリプトファイル

```bash
#!/usr/bin/awk -f
# analyze_log.awk - ログ分析スクリプト

BEGIN {
    FS = " "
    print "=== ログ分析レポート ==="
    print ""
}

# エラー行のカウント
/ERROR/ {
    error_count++
    error_lines[error_count] = $0
}

# 警告行のカウント
/WARN/ {
    warn_count++
}

# IPアドレスのカウント（1列目がIPの場合）
{
    ip_count[$1]++
    total++
}

END {
    print "総行数: " total
    print "エラー: " error_count+0
    print "警告: " warn_count+0
    print ""

    print "=== エラー詳細 ==="
    for (i = 1; i <= error_count && i <= 10; i++) {
        print "  " error_lines[i]
    }
    print ""

    print "=== IPアドレス別アクセス数（上位10）==="
    # awk 内でソートはできないため、パイプで処理
    n = asorti(ip_count, sorted_ips)
    for (i = 1; i <= n; i++) {
        ip = sorted_ips[i]
        count_arr[ip_count[ip]] = count_arr[ip_count[ip]] ? count_arr[ip_count[ip]] "\n  " ip : ip
    }
}
```

```bash
# 実行方法
awk -f analyze_log.awk access.log
# または
chmod +x analyze_log.awk
./analyze_log.awk access.log
```

### 9.2 複雑な処理の例

```bash
#!/usr/bin/awk -f
# csv_report.awk - CSV レポート生成

BEGIN {
    FS = ","
    OFS = ","

    print "=== CSV データ分析レポート ==="
}

# ヘッダー行の処理
NR == 1 {
    for (i = 1; i <= NF; i++) {
        headers[i] = $i
    }
    num_cols = NF
    next
}

# データ行の処理
{
    for (i = 1; i <= NF; i++) {
        # 数値フィールドの場合は集計
        if ($i ~ /^[0-9.]+$/) {
            sum[i] += $i
            count[i]++
            if (!(i in min) || $i < min[i]) min[i] = $i
            if (!(i in max) || $i > max[i]) max[i] = $i
        }
    }
    data_rows++
}

END {
    printf "\nデータ行数: %d\n", data_rows
    printf "列数: %d\n\n", num_cols

    printf "%-20s %10s %10s %10s %10s\n", "Column", "Sum", "Avg", "Min", "Max"
    printf "%-20s %10s %10s %10s %10s\n", "------", "---", "---", "---", "---"

    for (i = 1; i <= num_cols; i++) {
        if (i in sum) {
            printf "%-20s %10.2f %10.2f %10.2f %10.2f\n",
                headers[i], sum[i], sum[i]/count[i], min[i], max[i]
        }
    }
}
```

---

## 10. 実務パターン集

### 10.1 Apache/Nginx アクセスログ分析

```bash
# ステータスコード別の集計
awk '{count[$9]++} END {for (c in count) print c, count[c]}' access.log | sort -k2 -rn

# IPアドレス別アクセス数トップ20
awk '{count[$1]++} END {for (ip in count) print count[ip], ip}' access.log | sort -rn | head -20

# リクエストURL別アクセス数トップ20
awk '{count[$7]++} END {for (url in count) print count[url], url}' access.log | sort -rn | head -20

# HTTPメソッド別の集計
awk '{gsub(/"/, "", $6); count[$6]++} END {for (m in count) print m, count[m]}' access.log

# 時間帯別のリクエスト数
awk '{split($4, a, ":"); hour=a[2]; count[hour]++}
     END {for (h in count) print h, count[h]}' access.log | sort

# レスポンスタイムの分析（最終フィールドがレスポンスタイムの場合）
awk '{sum += $NF; count++; if($NF > max) max=$NF}
     END {printf "Avg: %.2fms, Max: %.2fms, Count: %d\n", sum/count, max, count}' access.log

# 4xx/5xx エラーのURL一覧
awk '$9 >= 400 {count[$7]++}
     END {for (url in count) print count[url], url}' access.log | sort -rn | head -20

# 帯域使用量の集計（10列目がバイト数の場合）
awk '{bytes[$1] += $10}
     END {for (ip in bytes) printf "%-20s %10.2f MB\n", ip, bytes[ip]/1048576}' access.log | sort -t'M' -k2 -rn | head -10

# 特定のIPからの不正アクセス検出（短時間に大量リクエスト）
awk '{ip_time[$1","$4]++}
     END {for (k in ip_time) if (ip_time[k] > 100) print k, ip_time[k]}' access.log

# スロークエリの検出（1秒以上のレスポンス）
awk '$NF > 1000 {print $4, $7, $NF"ms"}' access.log | head -20
```

### 10.2 システムモニタリング

```bash
# プロセスのメモリ使用量トップ10
ps aux | awk 'NR>1{print $4, $11}' | sort -rn | head -10

# プロセスのCPU使用量トップ10
ps aux | awk 'NR>1{print $3, $11}' | sort -rn | head -10

# ユーザー別プロセス数
ps aux | awk 'NR>1{count[$1]++} END{for(u in count) print count[u], u}' | sort -rn

# ディスク使用率の警告
df -h | awk 'NR>1{gsub(/%/,"",$5); if($5+0 > 80) printf "WARNING: %s is %s%% full\n", $6, $5}'

# メモリ使用状況（/proc/meminfo から）
awk '/^(MemTotal|MemFree|MemAvailable|Buffers|Cached):/{
    gsub(/kB/,""); printf "%-15s %10.2f MB\n", $1, $2/1024
}' /proc/meminfo

# ネットワーク接続の状態別カウント
ss -tan | awk 'NR>1{count[$1]++} END{for(s in count) print s, count[s]}'

# 接続元IPアドレス別カウント
ss -tan | awk 'NR>1{split($5,a,":"); if(a[1]!="*") count[a[1]]++}
               END{for(ip in count) print count[ip], ip}' | sort -rn | head -10

# CPU使用率の時系列モニタリング（1秒おき）
# vmstat 1 | awk '{print strftime("%H:%M:%S"), 100-$15"%"}'
```

### 10.3 CSV/TSV データ処理

```bash
# CSVのヘッダーを除いたデータ行数
awk -F',' 'NR>1{count++} END{print count}' data.csv

# 特定列の値でフィルタリング
awk -F',' '$3 > 1000' data.csv         # 3列目が1000超

# 列の追加（計算列）
awk -F',' 'BEGIN{OFS=","} NR==1{print $0,"total"; next} {print $0, $2+$3+$4}' data.csv

# 列の削除（3列目を除外）
awk -F',' 'BEGIN{OFS=","} {$3=""; gsub(/,,/,","); print}' data.csv
# より堅牢な方法:
awk -F',' 'BEGIN{OFS=","} {
    for(i=1;i<=NF;i++) if(i!=3) printf "%s%s", $i, (i<NF&&i+1!=3?OFS:"");
    print ""
}' data.csv

# 列の並び替え
awk -F',' 'BEGIN{OFS=","} {print $3, $1, $2}' data.csv

# NULL/空値のチェック
awk -F',' '{for(i=1;i<=NF;i++) if($i=="") printf "Row %d, Col %d is empty\n", NR, i}' data.csv

# CSV のダブルクォートを適切に処理（簡易版）
awk -F'","' '{gsub(/^"|"$/, ""); print $2}' data.csv

# データのバリデーション
awk -F',' 'NR>1{
    if ($2 !~ /^[0-9]+$/) print "Invalid number at row " NR ": " $2
    if ($3 == "") print "Empty field at row " NR ", col 3"
    if (NF != expected_cols) print "Wrong column count at row " NR ": " NF " (expected " expected_cols ")"
}' expected_cols=5 data.csv

# ピボットテーブル（行→列変換）
awk -F',' '{
    rows[$1] = 1
    cols[$2] = 1
    data[$1","$2] = $3
}
END {
    printf "%-15s", ""
    for (c in cols) printf "%15s", c
    print ""
    for (r in rows) {
        printf "%-15s", r
        for (c in cols) printf "%15s", data[r","c]+0
        print ""
    }
}' data.csv
```

### 10.4 テキスト変換

```bash
# JSON 風の出力（簡易版）
awk -F',' 'NR==1{split($0,h); next}
    {printf "{\n"
     for(i=1;i<=NF;i++) printf "  \"%s\": \"%s\"%s\n", h[i], $i, (i<NF?",":"")
     printf "}\n"}' data.csv

# key=value から JSON への変換
awk -F'=' 'BEGIN{print "{"} {printf "  \"%s\": \"%s\"", $1, $2; if(NR>1) printf ","
    printf "\n"} END{print "}"}' config.ini

# Markdown テーブルの生成
awk -F',' 'NR==1{
    printf "| "
    for(i=1;i<=NF;i++) printf "%s | ", $i
    printf "\n| "
    for(i=1;i<=NF;i++) printf "--- | "
    printf "\n"
    next
}
{
    printf "| "
    for(i=1;i<=NF;i++) printf "%s | ", $i
    printf "\n"
}' data.csv

# SQL INSERT 文の生成
awk -F',' 'NR>1{
    printf "INSERT INTO table_name VALUES ("
    for(i=1;i<=NF;i++) {
        if($i ~ /^[0-9]+$/) printf "%s", $i
        else printf "'\'%s\''", $i
        if(i<NF) printf ", "
    }
    printf ");\n"
}' data.csv

# HTML テーブルの生成
awk -F',' 'BEGIN{print "<table>"}
    NR==1{print "  <tr>"; for(i=1;i<=NF;i++) print "    <th>"$i"</th>"; print "  </tr>"; next}
    {print "  <tr>"; for(i=1;i<=NF;i++) print "    <td>"$i"</td>"; print "  </tr>"}
    END{print "</table>"}' data.csv

# 複数行レコードの処理（空行区切り）
awk 'BEGIN{RS=""; FS="\n"} {print $1, $2, $3}' records.txt
# → 空行で区切られた複数行のレコードを1行にまとめる
```

### 10.5 ファイル比較と差分

```bash
# 2つのファイルの共通行
awk 'NR==FNR{a[$0]; next} $0 in a' file1.txt file2.txt

# file1 にあって file2 にない行
awk 'NR==FNR{a[$0]; next} !($0 in a)' file2.txt file1.txt

# file2 にあって file1 にない行
awk 'NR==FNR{a[$0]; next} !($0 in a)' file1.txt file2.txt

# フィールド単位での比較
awk -F',' 'NR==FNR{a[$1]=$2; next} ($1 in a) && a[$1]!=$2{
    print $1, "changed:", a[$1], "->", $2
}' old.csv new.csv

# 差分レポートの生成
awk 'NR==FNR{old[$1]=$0; next}
     {
         if ($1 in old) {
             if (old[$1] != $0) print "MODIFIED:", $0
             delete old[$1]
         } else {
             print "ADDED:", $0
         }
     }
     END{for(k in old) print "DELETED:", old[k]}' old.txt new.txt
```

---

## 11. gawk 拡張機能

### 11.1 gawk 固有の機能

```bash
# gawk（GNU awk）は POSIX awk に多くの拡張を追加

# BEGINFILE / ENDFILE: 各ファイルの処理前/後に実行
gawk 'BEGINFILE{print "=== " FILENAME " ==="} {print}' *.txt

# nextfile: 現在のファイルの残りをスキップ
gawk '/ERROR/{print FILENAME; nextfile}' *.log
# → ERROR を含む最初のファイル名を表示して次のファイルへ

# @include: 外部ファイルの読み込み
# gawk スクリプト内で:
# @include "functions.awk"

# ネットワーク通信（gawk のネットワーク機能）
# gawk 'BEGIN{
#     service = "/inet/tcp/0/example.com/80"
#     print "GET / HTTP/1.0\r\nHost: example.com\r\n" |& service
#     service |& getline response
#     print response
#     close(service)
# }'

# 正規表現の強力なマッチ（gensub）
gawk '{print gensub(/([0-9]+)/, "[\\1]", "g")}' file.txt
# → 全ての数字を角括弧で囲む

gawk '{print gensub(/(.)(.)/, "\\2\\1", "g")}' file.txt
# → 2文字ずつ入れ替え

# 多次元配列
gawk '{sales[$1][$2] += $3}
      END{for(dept in sales) for(month in sales[dept])
          print dept, month, sales[dept][month]}' sales.txt

# ソート制御（PROCINFO["sorted_in"]）
gawk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}
      {count[$1]++}
      END{for(k in count) print k, count[k]}' data.txt
# → 値の降順でソートして出力
```

### 11.2 FPAT（フィールドパターン）

```bash
# CSV の正しい解析（ダブルクォート内のカンマを考慮）
gawk 'BEGIN{FPAT="([^,]*)|(\"[^\"]*\")"} {print $2}' data.csv
# → "field1","field with, comma","field3" を正しく分割

# 例: CSV のフィールドをダブルクォート付きで正しく抽出
gawk 'BEGIN{FPAT="([^,]*)|(\"[^\"]*\")"}
      {for(i=1;i<=NF;i++) {
          gsub(/^"|"$/, "", $i)  # クォートを除去
          print i": "$i
      }}' data.csv
```

---

## 12. トラブルシューティング

### 12.1 よくある問題と対処法

```bash
# 問題: フィールドが期待通りに分割されない
# 対処: -F で正しい区切り文字を指定
awk -F',' '{print NF, $0}' data.csv    # 列数を確認
awk -F'\t' '{print NF, $0}' data.tsv   # タブ区切りの確認

# 問題: 数値の比較が文字列比較になる
# 対処: 明示的に数値に変換（+0）
awk '$3+0 > 100' data.txt              # +0 で数値化
awk '{if ($3+0 > 100) print}' data.txt

# 問題: 空のフィールドが正しく処理されない
# 対処: FPAT を使うか、フィールドの存在を確認
awk -F',' '{if ($3 != "") print $3}' data.csv

# 問題: 大きなファイルでメモリ不足
# 対処: 配列の蓄積を避ける、または定期的に削除
awk '{count[$1]++; if(NR%1000000==0){for(k in count){print k,count[k]; delete count[k]}}}
     END{for(k in count) print k, count[k]}' huge_file.txt

# 問題: macOS のデフォルト awk と gawk の違い
# 対処: gawk をインストール
# brew install gawk
# gawk を使用するか、POSIX 互換の構文に限定する

# 問題: 日本語（マルチバイト文字）の処理
# 対処: ロケールを設定
LC_ALL=en_US.UTF-8 awk '{print length($0)}' file.txt
# または gawk を使用（UTF-8 対応が良い）

# デバッグ: 各行の処理を確認
awk '{print "DEBUG:", NR, NF, $0}' file.txt
awk '{for(i=1;i<=NF;i++) print "Field " i ": [" $i "]"}' file.txt
```

### 12.2 パフォーマンスのヒント

```bash
# 1. 不要な出力を避ける
awk '/pattern/' file.txt               # 良い: 条件に一致する行のみ出力
# awk '{if(/pattern/) print}' file.txt # 同じだがやや冗長

# 2. gsub より sub を使う（1回の置換で十分な場合）
awk '{sub(/old/, "new"); print}' file.txt

# 3. 巨大ファイルでの配列使用を最小化
# 全行を配列に格納するのではなく、ストリーム処理を心がける

# 4. 正規表現のキャッシュ
# awk は同じ正規表現を繰り返し使う場合に自動でキャッシュするが、
# 動的な正規表現（変数を使った）はキャッシュされない

# 5. パイプとの組み合わせで役割分担
# 複雑な処理は awk 1つで全てやるのではなく、
# grep でフィルタ → awk で整形 → sort でソート のようにパイプで分担
grep "ERROR" logfile.txt | awk '{print $1, $NF}' | sort | uniq -c | sort -rn
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
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

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない
---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 構文 | 用途 | 例 |
|------|------|------|
| {print $N} | N列目を表示 | awk '{print $1}' |
| -F',' | 区切り文字指定 | awk -F',' '{print $2}' |
| /pattern/ | 行のフィルタ | awk '/error/' |
| $N > val | 条件フィルタ | awk '$3 > 100' |
| BEGIN {} | 前処理 | awk 'BEGIN{FS=","}' |
| END {} | 後処理（集計） | awk 'END{print NR}' |
| count[$1]++ | 連想配列でカウント | グループ別集計 |
| sum += $N | 合計の計算 | awk '{s+=$2}END{print s}' |
| printf | フォーマット出力 | printf "%.2f", val |
| NR | 行番号 | awk 'NR>1' |
| NF | フィールド数 | awk '{print NF}' |
| length() | 文字列長 | awk 'length>80' |
| substr() | 部分文字列 | substr($1,1,3) |
| split() | 文字列分割 | split($1,a,"-") |
| sub/gsub | 置換 | gsub(/old/,"new") |
| tolower/toupper | 大小文字変換 | tolower($0) |
| NR==FNR | ファイル結合 | 2ファイルのJOIN |

### awk vs sed vs grep の使い分け

```
┌────────────────────────────┬──────────────┐
│ やりたいこと               │ 最適なツール │
├────────────────────────────┼──────────────┤
│ パターンに一致する行を抽出 │ grep / rg    │
│ 文字列の置換               │ sed          │
│ 行の削除・挿入             │ sed          │
│ 列（フィールド）の抽出     │ awk          │
│ 数値の計算・集計           │ awk          │
│ グループ別の集計           │ awk          │
│ フォーマット出力           │ awk          │
│ 2つのファイルの結合        │ awk          │
│ 条件付きの複雑な処理       │ awk          │
│ 簡単なフィルタリング       │ grep         │
│ 簡単な置換                 │ sed          │
└────────────────────────────┴──────────────┘
```

---

## 15. awk と他ツールの連携パターン

### 15.1 パイプライン構築の実践

```bash
# grep + awk の効率的な連携
# grep で絞り込み、awk で集計する典型パターン
grep "ERROR" app.log | awk '{count[$5]++} END {for (k in count) print count[k], k}' | sort -rn

# find + awk でファイルシステム分析
find /var/log -type f -name "*.log" -printf "%s %p\n" | \
  awk '{total += $1; count++} END {printf "Files: %d, Total: %.2f MB, Avg: %.2f KB\n", count, total/1024/1024, total/count/1024}'

# ps + awk でプロセスメモリ使用量を集計
ps aux | awk 'NR>1 {mem[$11] += $6} END {for (p in mem) printf "%10d KB  %s\n", mem[p], p}' | sort -rn | head -20

# docker stats + awk でコンテナリソース集計
docker stats --no-stream --format "{{.Name}} {{.MemUsage}} {{.CPUPerc}}" | \
  awk '{gsub(/MiB|%/, ""); print $1, $2, $NF}'

# netstat + awk でコネクション状態の集計
netstat -an 2>/dev/null | awk '/^tcp/ {state[$6]++} END {for (s in state) printf "%-20s %d\n", s, state[s]}'

# sar + awk でCPU使用率のピーク時間帯を特定
sar -u 2>/dev/null | awk 'NR>3 && $NF != "idle" && $NF+0 < 20 {print "High CPU at", $1, "idle:", $NF"%"}'
```

### 15.2 シェルスクリプト内での awk 活用

```bash
# awk の結果をシェル変数に代入
total=$(awk '{sum += $1} END {print sum}' data.txt)
echo "Total: $total"

# awk の結果で条件分岐
if awk '{sum += $1} END {exit (sum > 1000) ? 0 : 1}' data.txt; then
  echo "Sum exceeds 1000"
fi

# awk で生成したコマンドを実行
awk '{printf "mv %s %s.bak\n", $0, $0}' file_list.txt | sh

# awk からシェル変数を参照
threshold=100
awk -v t="$threshold" '$3 > t {print $0}' data.txt

# 複数の変数を同時に取得
read total count avg <<< $(awk '{sum+=$1; n++} END {printf "%d %d %.2f", sum, n, sum/n}' data.txt)
echo "Total: $total, Count: $count, Average: $avg"
```

---

## 次に読むべきガイド

---

## 参考文献
1. Aho, A., Kernighan, B. & Weinberger, P. "The AWK Programming Language." 2nd Ed, 2023.
2. Robbins, A. "sed & awk." 2nd Ed, O'Reilly, 1997.
3. Robbins, A. "Effective awk Programming." 5th Ed, O'Reilly, 2024.
4. GNU Awk User's Guide. https://www.gnu.org/software/gawk/manual/
5. awk One-Liners Explained. https://catonmat.net/awk-one-liners-explained-part-one
