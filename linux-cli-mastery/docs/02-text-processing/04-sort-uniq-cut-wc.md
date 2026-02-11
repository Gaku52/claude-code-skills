# ソート・集計（sort, uniq, cut, wc）

> これらのコマンドはパイプラインの「部品」として組み合わせて使う。

## この章で学ぶこと

- [ ] テキストのソート・重複排除・列切り出し・カウントができる

---

## 1. sort

```bash
sort file.txt                     # アルファベット順ソート
sort -n file.txt                  # 数値順ソート
sort -r file.txt                  # 逆順
sort -k2 file.txt                 # 2列目でソート
sort -k2,2n file.txt              # 2列目を数値ソート
sort -t',' -k3 data.csv           # CSV の3列目でソート
sort -u file.txt                  # ソート+重複排除
sort -h file.txt                  # 人間可読サイズ（1K, 2M）順
sort --stable file.txt            # 安定ソート
```

---

## 2. uniq, cut, wc, tr

```bash
# uniq（連続する重複行を処理 → sort と組み合わせ必須）
sort file.txt | uniq              # 重複排除
sort file.txt | uniq -c           # 出現回数付き
sort file.txt | uniq -d           # 重複行のみ表示
sort file.txt | uniq -c | sort -rn  # 出現頻度順（定番パターン）

# cut（列の切り出し）
cut -d',' -f2 data.csv            # CSV の2列目
cut -d':' -f1,3 /etc/passwd       # 1列目と3列目
cut -c1-10 file.txt               # 1〜10文字目
echo "hello world" | cut -d' ' -f2  # → "world"

# wc（カウント）
wc -l file.txt                    # 行数
wc -w file.txt                    # 単語数
wc -c file.txt                    # バイト数
find . -name "*.py" | wc -l       # Pythonファイル数

# tr（文字変換）
echo "Hello" | tr 'A-Z' 'a-z'    # 小文字に変換
echo "hello" | tr 'a-z' 'A-Z'    # 大文字に変換
echo "a:b:c" | tr ':' '\n'       # :を改行に変換
echo "  hello  " | tr -s ' '     # 連続スペースを1つに
cat file.txt | tr -d '\r'         # \r (CR)を削除
```

---

## 3. 組み合わせパターン

```bash
# アクセスログのIPアドレスTop10
cat access.log | awk '{print $1}' | sort | uniq -c | sort -rn | head -10

# ファイル拡張子の集計
find . -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn

# CSV の特定列の合計
cut -d',' -f3 data.csv | tail -n +2 | paste -sd+ | bc

# 単語頻度分析
cat file.txt | tr -s '[:space:]' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -20
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| sort -n | 数値ソート |
| uniq -c | 重複カウント |
| cut -d',' -f2 | 列の切り出し |
| wc -l | 行数カウント |
| tr 'A-Z' 'a-z' | 文字変換 |

---

## 次に読むべきガイド
→ [[../03-process-management/00-ps-top-htop.md]] — プロセス監視

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.5, O'Reilly, 2022.
