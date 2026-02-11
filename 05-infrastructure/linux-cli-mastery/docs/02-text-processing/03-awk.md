# テキスト処理言語（awk）

> awk は「構造化テキストを列単位で処理する」ミニプログラミング言語。

## この章で学ぶこと

- [ ] awk の基本構文を理解する
- [ ] フィールド操作と集計ができる

---

## 1. awk の基本

```bash
# 基本: awk 'パターン { アクション }' [ファイル]

# フィールド（列）の抽出
awk '{print $1}' file.txt              # 1列目を表示
awk '{print $1, $3}' file.txt          # 1列目と3列目
awk '{print $NF}' file.txt             # 最終列
awk '{print NR, $0}' file.txt          # 行番号+全体

# 区切り文字の指定
awk -F',' '{print $2}' data.csv        # CSV の2列目
awk -F':' '{print $1}' /etc/passwd     # ユーザー名一覧
awk -F'\t' '{print $1}' data.tsv       # TSV の1列目

# パターンマッチ
awk '/error/' logfile.txt              # errorを含む行
awk '$3 > 100' data.txt                # 3列目が100より大きい行
awk '$1 == "admin"' users.txt          # 1列目がadminの行
awk 'NR >= 10 && NR <= 20' file.txt    # 10〜20行目

# 組み込み変数
# $0: 行全体, $1〜$N: 各フィールド
# NR: 行番号, NF: フィールド数
# FS: フィールド区切り, OFS: 出力区切り
```

---

## 2. 集計と変換

```bash
# 合計
awk '{sum += $2} END {print sum}' data.txt    # 2列目の合計
awk '{sum += $2; n++} END {print sum/n}' data.txt  # 平均

# カウント
awk '{count[$1]++} END {for (k in count) print k, count[k]}' access.log
# → IPアドレスごとのアクセス回数

# フォーマット出力
awk '{printf "%-20s %10.2f\n", $1, $2}' data.txt   # 整形出力
awk 'BEGIN {OFS=","} {print $1, $2, $3}' data.txt   # CSV出力

# 条件分岐
awk '{
    if ($3 >= 90) grade = "A"
    else if ($3 >= 80) grade = "B"
    else grade = "C"
    print $1, grade
}' scores.txt

# 実務例: Apache アクセスログ分析
# ステータスコード別集計
awk '{count[$9]++} END {for (c in count) print c, count[c]}' access.log

# レスポンスタイム集計（最終列がレスポンスタイムの場合）
awk '{sum += $NF; n++} END {printf "Avg: %.2fms\n", sum/n}' access.log
```

---

## まとめ

| 構文 | 用途 |
|------|------|
| {print $N} | N列目を表示 |
| -F',' | 区切り文字指定 |
| /pattern/ | 行のフィルタ |
| END {} | 全行処理後の集計 |
| count[$1]++ | 連想配列でカウント |

---

## 次に読むべきガイド
→ [[04-sort-uniq-cut-wc.md]] — ソート・集計

---

## 参考文献
1. Aho, A., Kernighan, B. & Weinberger, P. "The AWK Programming Language." 2nd Ed, 2023.
