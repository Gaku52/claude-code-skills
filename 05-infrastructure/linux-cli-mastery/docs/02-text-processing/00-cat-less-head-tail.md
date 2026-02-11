# ファイル表示

> テキストファイルの内容を確認する方法は目的に応じて使い分ける。

## この章で学ぶこと

- [ ] ファイル表示コマンドを適切に使い分けられる

---

## 1. 表示コマンド

```bash
# cat（全体表示）
cat file.txt                     # 全内容を表示
cat -n file.txt                  # 行番号付き
cat file1.txt file2.txt          # 複数ファイル結合
cat -A file.txt                  # 特殊文字を可視化（タブ、改行）

# bat（catのモダン代替）
# brew install bat
bat file.py                      # シンタックスハイライト+行番号
bat -l json file.txt             # 言語指定

# less（ページャ）
less file.txt                    # ページ単位で表示
# 操作: j/k(行スクロール), Space/b(ページ), /pattern(検索), q(終了)
# less は大きなファイルでも高速（全体を読み込まない）

# head / tail
head file.txt                    # 先頭10行
head -n 20 file.txt              # 先頭20行
head -c 100 file.txt             # 先頭100バイト

tail file.txt                    # 末尾10行
tail -n 20 file.txt              # 末尾20行
tail -f /var/log/syslog          # リアルタイム監視（ログ追跡）
tail -f -n 0 logfile             # 新規行のみ表示

# wc（ワードカウント）
wc file.txt                      # 行数 単語数 バイト数
wc -l file.txt                   # 行数のみ
wc -w file.txt                   # 単語数のみ
wc -c file.txt                   # バイト数のみ
wc -m file.txt                   # 文字数（マルチバイト対応）

# diff（差分表示）
diff file1.txt file2.txt         # 差分表示
diff -u file1.txt file2.txt      # unified形式（gitと同じ）
diff -r dir1/ dir2/              # ディレクトリの再帰比較
```

---

## 2. 使い分けガイド

```
目的に応じた選択:

  ファイル全体を見たい      → cat (小) / less (大) / bat (コード)
  先頭/末尾だけ見たい       → head / tail
  ログをリアルタイム監視    → tail -f
  行数を数えたい            → wc -l
  2つのファイルの差分       → diff -u
  バイナリファイルの確認    → xxd / hexdump
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| cat / bat | 全体表示 |
| less | ページ表示（大ファイル向け） |
| head / tail | 先頭/末尾の表示 |
| tail -f | リアルタイムログ監視 |
| wc -l | 行数カウント |
| diff -u | ファイル差分 |

---

## 次に読むべきガイド
→ [[01-grep-ripgrep.md]] — パターン検索

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.6, 2019.
