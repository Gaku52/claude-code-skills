# ファイルの作成・コピー・移動・削除

> ファイル操作は CLI の最も頻繁に使うスキル。安全な操作習慣を身につけよう。

## この章で学ぶこと

- [ ] ファイルとディレクトリの基本操作ができる
- [ ] ワイルドカード（グロブ）を使いこなせる

---

## 1. 基本操作

```bash
# ファイル作成
touch file.txt                  # 空ファイル作成（既存ならタイムスタンプ更新）
echo "content" > file.txt       # 内容付きで作成
cat > file.txt <<EOF             # ヒアドキュメントで作成
line 1
line 2
EOF

# ディレクトリ作成
mkdir dirname                   # ディレクトリ作成
mkdir -p path/to/nested/dir     # ネストしたディレクトリを一気に作成

# コピー
cp source.txt dest.txt          # ファイルコピー
cp -r source_dir/ dest_dir/     # ディレクトリを再帰的にコピー
cp -a source/ dest/             # アーカイブモード（パーミッション保持）
cp -i source.txt dest.txt       # 上書き確認

# 移動 / リネーム
mv old_name.txt new_name.txt    # リネーム
mv file.txt /path/to/dir/      # 移動
mv -i source dest               # 上書き確認

# 削除
rm file.txt                     # ファイル削除
rm -r directory/                # ディレクトリを再帰的に削除
rm -i file.txt                  # 確認付き削除
rm -f file.txt                  # 強制削除（確認なし）
rmdir empty_dir                 # 空ディレクトリのみ削除

# ⚠️ 危険なコマンド（絶対に注意）
# rm -rf /        ← システム全体を削除（最悪のコマンド）
# rm -rf ~        ← ホームディレクトリ全削除
# rm -rf *        ← カレントディレクトリの全ファイル削除
```

---

## 2. ワイルドカード（グロブ）

```bash
# ワイルドカードパターン
*              # 任意の文字列（0文字以上）
?              # 任意の1文字
[abc]          # a, b, c のいずれか
[a-z]          # a〜z の範囲
[!abc]         # a, b, c 以外
{foo,bar}      # foo または bar（ブレース展開）

# 使用例
ls *.md                         # 全 .md ファイル
ls image?.png                   # image1.png, image2.png 等
ls file[1-3].txt                # file1.txt, file2.txt, file3.txt
cp {main,test}.py /tmp/         # main.py と test.py をコピー
mkdir -p project/{src,tests,docs}  # 3ディレクトリ一括作成

# 高度なグロブ（zsh / bash globstar）
ls **/*.md                      # 再帰的に全 .md ファイル
ls *.{js,ts}                    # .js と .ts ファイル
```

---

## 3. 安全なファイル操作

```bash
# ゴミ箱を使う（rm の代替）
# brew install trash-cli
trash file.txt                  # ゴミ箱に移動（復元可能）

# バックアップ付きコピー
cp --backup=numbered file.txt dest/

# dry-run で確認（rsync）
rsync -avhn source/ dest/       # -n でドライラン（実行しない）
rsync -avh source/ dest/        # 実行

# rsync の利点:
# - 差分転送（変更されたファイルのみ）
# - リモート対応（SSH経由）
# - 進捗表示、帯域制限
```

---

## まとめ

| 操作 | コマンド |
|------|---------|
| 作成 | touch, mkdir -p |
| コピー | cp -r, rsync -avh |
| 移動 | mv |
| 削除 | rm -r (注意!), trash |
| ワイルドカード | *, ?, [], {} |

---

## 次に読むべきガイド
→ [[02-permissions.md]] — パーミッションと所有者

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.4, 2019.
