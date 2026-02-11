# パターン検索（grep / ripgrep）

> grep は「テキストの中から必要な行を抽出する」最も重要なフィルタリングツール。

## この章で学ぶこと

- [ ] grep の主要オプションを使いこなせる
- [ ] 正規表現を活用した検索ができる
- [ ] ripgrep で高速検索ができる

---

## 1. grep の基本

```bash
# 基本: grep [オプション] パターン [ファイル]
grep "error" logfile.txt            # "error" を含む行
grep -i "error" logfile.txt         # 大小文字無視
grep -n "error" logfile.txt         # 行番号付き
grep -c "error" logfile.txt         # マッチ行数
grep -l "error" *.log               # マッチするファイル名のみ
grep -r "TODO" ./src/               # 再帰検索
grep -v "debug" logfile.txt         # パターンを含まない行
grep -w "error" logfile.txt         # 単語として完全一致
grep -A 3 "error" logfile.txt       # マッチ行+後3行
grep -B 2 "error" logfile.txt       # マッチ行+前2行
grep -C 2 "error" logfile.txt       # マッチ行+前後2行

# 複数パターン
grep -e "error" -e "warn" logfile.txt      # error OR warn
grep "error" logfile.txt | grep "fatal"    # error AND fatal

# 正規表現（-E = 拡張正規表現）
grep -E "^[0-9]{4}-[0-9]{2}" logfile.txt   # 日付パターン
grep -E "(error|warning|fatal)" logfile.txt # OR
grep -E "status:\s+(200|301)" access.log    # HTTPステータス

# パイプとの組み合わせ
ps aux | grep nginx                # nginxプロセス検索
docker ps | grep -i running       # 稼働中コンテナ
history | grep "git push"         # コマンド履歴検索
```

---

## 2. ripgrep（rg）

```bash
# brew install ripgrep
rg "pattern"                       # カレントディレクトリを再帰検索
rg "pattern" -i                    # 大小文字無視
rg "pattern" -t py                 # Pythonファイルのみ
rg "pattern" -g "*.{js,ts}"       # グロブでフィルタ
rg "pattern" --hidden              # 隠しファイル含む
rg "TODO|FIXME" -c                # ファイルごとのマッチ数
rg "pattern" -l                    # マッチするファイル名のみ
rg "pattern" --json                # JSON出力（ツール連携）
rg -F "exact.string"              # 正規表現無効（リテラル検索）

# grep との比較:
# rg: .gitignore尊重, Unicode対応, カラー, 高速(並列)
# grep: 標準搭載, POSIX互換, パイプに最適
```

---

## まとめ

| オプション | 意味 |
|-----------|------|
| -i | 大小文字無視 |
| -r | 再帰検索 |
| -n | 行番号表示 |
| -l | ファイル名のみ |
| -v | 逆マッチ |
| -E | 拡張正規表現 |
| -A/-B/-C | 前後のコンテキスト |

---

## 次に読むべきガイド
→ [[02-sed.md]] — ストリームエディタ

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.5, O'Reilly, 2022.
