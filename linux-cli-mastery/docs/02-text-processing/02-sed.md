# ストリームエディタ（sed）

> sed は「テキストを行単位で変換する」パイプラインの強力な変換器。

## この章で学ぶこと

- [ ] sed の基本的な置換・削除・挿入ができる
- [ ] 実務で使える sed パターンを知る

---

## 1. sed の基本

```bash
# 基本: sed [オプション] 'コマンド' [ファイル]

# 置換（s コマンド）
sed 's/old/new/' file.txt           # 各行の最初のoldをnewに
sed 's/old/new/g' file.txt          # 全てのoldをnewに（gフラグ）
sed 's/old/new/gi' file.txt         # 大小文字無視で全置換
sed -i 's/old/new/g' file.txt       # ファイルを直接書き換え
sed -i.bak 's/old/new/g' file.txt   # バックアップ作成+書き換え

# 行の削除（d コマンド）
sed '5d' file.txt                   # 5行目を削除
sed '1,3d' file.txt                 # 1〜3行目を削除
sed '/pattern/d' file.txt           # パターンにマッチする行を削除
sed '/^$/d' file.txt                # 空行を削除
sed '/^#/d' file.txt                # コメント行を削除

# 行の表示（p コマンド）
sed -n '5p' file.txt                # 5行目のみ表示
sed -n '10,20p' file.txt            # 10〜20行目を表示
sed -n '/error/p' file.txt          # errorを含む行のみ（grep的）

# 挿入・追加
sed '3i\新しい行' file.txt          # 3行目の前に挿入
sed '3a\新しい行' file.txt          # 3行目の後に追加
```

---

## 2. 実務パターン

```bash
# ファイル内の一括置換
sed -i 's/http:/https:/g' *.html      # HTTPをHTTPSに

# 特定行の前後に追加
sed '/^import/a\import os' file.py    # import行の後にimport os追加

# 複数コマンド
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt
sed 's/foo/bar/g; s/baz/qux/g' file.txt

# 区切り文字の変更（URLなどで/が含まれる場合）
sed 's|http://old.com|https://new.com|g' file.txt
sed 's#old/path#new/path#g' file.txt

# 後方参照（グループ化）
sed 's/\(.*\)=\(.*\)/\2=\1/' file.txt  # key=value → value=key
sed -E 's/([0-9]+)-([0-9]+)/\2-\1/' file.txt  # 数字の入れ替え

# 行番号の追加
sed '=' file.txt | sed 'N; s/\n/\t/'  # タブ区切りの行番号
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| s/old/new/g | 置換 |
| /pattern/d | 行の削除 |
| -n 'Np' | 特定行の表示 |
| -i | ファイル直接書き換え |

---

## 次に読むべきガイド
→ [[03-awk.md]] — テキスト処理言語

---

## 参考文献
1. Robbins, A. "sed & awk." 2nd Ed, O'Reilly, 1997.
