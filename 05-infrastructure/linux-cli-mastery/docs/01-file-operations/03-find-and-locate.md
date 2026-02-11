# ファイル検索

> 「あのファイルどこだっけ？」— find と fd があれば必ず見つかる。

## この章で学ぶこと

- [ ] find の主要な使い方をマスターする
- [ ] fd（モダン代替）を使いこなせる

---

## 1. find

```bash
# 基本: find [検索開始パス] [条件] [アクション]

# 名前で検索
find . -name "*.md"               # .md ファイルを再帰検索
find . -iname "readme*"           # 大小文字無視
find /etc -name "*.conf"          # /etc 以下の設定ファイル

# タイプで絞り込み
find . -type f                    # ファイルのみ
find . -type d                    # ディレクトリのみ
find . -type l                    # シンボリックリンクのみ

# サイズで絞り込み
find . -size +100M                # 100MB以上
find . -size -1k                  # 1KB未満
find . -empty                     # 空ファイル/ディレクトリ

# 日時で絞り込み
find . -mtime -7                  # 7日以内に変更
find . -mmin -30                  # 30分以内に変更
find . -newer reference.txt       # reference.txt より新しい

# 組み合わせ
find . -name "*.log" -size +10M   # 10MB以上のログ
find . -name "*.tmp" -mtime +30   # 30日以上前のtmpファイル
find . \( -name "*.js" -o -name "*.ts" \)  # js または ts

# アクション
find . -name "*.tmp" -delete      # 見つけたファイルを削除
find . -name "*.sh" -exec chmod +x {} \;  # 実行権限付与
find . -name "*.log" -exec gzip {} \;     # gzip圧縮

# よく使うパターン
find . -name "node_modules" -type d -prune  # node_modules を除外
find . -name "*.py" -exec grep -l "import os" {} \;  # osをimportしてるpy
```

---

## 2. fd（モダンな代替）

```bash
# brew install fd
fd                                # 全ファイル（.gitignore を尊重）
fd "\.md$"                        # 正規表現で検索
fd -e md                          # 拡張子で検索
fd -t f                           # ファイルのみ
fd -t d                           # ディレクトリのみ
fd -H                             # 隠しファイル含む
fd -s "README"                    # 大小文字区別
fd --changed-within 1h            # 1時間以内に変更
fd -x rm                          # 見つけたファイルを削除

# find との比較
# find: 全ファイルを探索、.gitignore 無視、構文が複雑
# fd:   .gitignore 尊重、カラー表示、シンプル構文、高速
```

---

## 3. locate

```bash
# データベースベースの高速検索
locate filename                   # パス名でデータベース検索
sudo updatedb                     # データベース更新（cron で定期実行）

# locate は古いデータの可能性がある
# リアルタイム検索には find / fd を使う
```

---

## まとめ

| ツール | 特徴 | 用途 |
|--------|------|------|
| find | 標準、高機能 | 複雑な条件検索 |
| fd | 高速、簡潔 | 日常的な検索 |
| locate | DB検索、最速 | ファイル名検索 |

---

## 次に読むべきガイド
→ [[../02-text-processing/00-cat-less-head-tail.md]] — ファイル表示

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.4, O'Reilly, 2022.
