# マニュアルとヘルプの活用

> man ページは「Unix界の百科事典」。コマンドの使い方に困ったら、まず man を引く習慣をつけよう。

## この章で学ぶこと

- [ ] man ページの読み方を知る
- [ ] 各種ヘルプの使い分けができる

---

## 1. ヘルプを得る方法

```bash
# 1. --help オプション（最も手軽）
ls --help              # 簡易ヘルプ
git commit --help      # 詳細ヘルプ（manと同等の場合も）

# 2. man ページ（最も包括的）
man ls                 # ls のマニュアル
man 5 passwd           # /etc/passwd のファイル形式（セクション5）
man -k keyword         # キーワードでman検索（apropos）

# manのセクション:
# 1: ユーザーコマンド      (ls, grep, git)
# 2: システムコール         (open, read, fork)
# 3: ライブラリ関数        (printf, malloc)
# 4: デバイスファイル      (/dev/null)
# 5: ファイル形式          (/etc/passwd)
# 7: その他               (ascii, regex)
# 8: システム管理コマンド  (mount, iptables)

# 3. info ページ（GNU系の詳細ドキュメント）
info coreutils

# 4. tldr（Too Long; Didn't Read）
# 実用的な使用例を簡潔に表示
# brew install tldr
tldr tar               # tarの実用例
tldr curl              # curlの実用例

# 5. 組み込みコマンドのヘルプ
help cd                # bash内蔵コマンドのヘルプ
type command           # コマンドの種類確認
```

---

## 2. man ページの読み方

```
man ページの構成:
  NAME:        コマンド名と1行説明
  SYNOPSIS:    使い方の書式
  DESCRIPTION: 詳細な説明
  OPTIONS:     オプション一覧
  EXAMPLES:    使用例（あれば）
  SEE ALSO:    関連コマンド
  BUGS:        既知のバグ

  SYNOPSIS の読み方:
  ls [OPTION]... [FILE]...
  │   │            │
  │   │            └── 省略可能、複数指定可
  │   └── 省略可能、複数指定可
  └── コマンド名

  man 内の操作:
  j/k:        1行スクロール
  Space/b:    1ページスクロール
  /pattern:   前方検索
  ?pattern:   後方検索
  n/N:        次/前の検索結果
  q:          終了
```

---

## まとめ

| 方法 | 用途 |
|------|------|
| --help | 手軽な簡易ヘルプ |
| man | 包括的なマニュアル |
| tldr | 実用的な使用例 |
| type/which | コマンドの場所・種類 |

---

## 次に読むべきガイド
→ [[../01-file-operations/00-navigation.md]] — ディレクトリ移動と一覧

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, No Starch Press, 2019.
