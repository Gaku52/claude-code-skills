# ターミナルとシェルの基礎

> ターミナルは「テキストでコンピュータと対話する窓口」であり、シェルは「コマンドを解釈して実行する通訳者」である。

## この章で学ぶこと

- [ ] ターミナル、シェル、コンソールの違いを説明できる
- [ ] シェルの基本操作を使いこなせる
- [ ] コマンドの基本構造を理解する

---

## 1. 基本概念

```
ターミナル vs シェル vs コンソール:

  ターミナル（端末エミュレータ）:
  → テキストの入出力を行うプログラム
  → iTerm2, Windows Terminal, Alacritty, Warp
  → かつては物理端末（VT100等）

  シェル:
  → コマンドを解釈・実行するプログラム
  → bash, zsh, fish, PowerShell
  → ターミナルの中で動作

  コンソール:
  → 物理的なキーボード+画面（またはその仮想版）
  → Ctrl+Alt+F1〜F6 でLinuxの仮想コンソール切替

  関係:
  ┌──────────────────────────┐
  │ ターミナル               │
  │ ┌──────────────────────┐│
  │ │ シェル (zsh/bash)    ││
  │ │ $ ls -la             ││
  │ │ $ git status         ││
  │ └──────────────────────┘│
  └──────────────────────────┘

主要なシェル:
  sh:   Bourne Shell (1979)。POSIX標準の基盤
  bash: Bourne Again Shell。Linux のデフォルト
  zsh:  Z Shell。bash互換 + 強力な補完。macOS デフォルト
  fish: Friendly Interactive Shell。設定なしで便利
  dash: 軽量sh互換。Ubuntu の /bin/sh

  現在のシェル確認:
  $ echo $SHELL        # デフォルトシェル
  $ echo $0            # 現在実行中のシェル
  $ cat /etc/shells    # 利用可能なシェル一覧
```

---

## 2. コマンドの基本構造

```
コマンドの構造:
  $ command [options] [arguments]

  $ ls -la /home/user
    │   │   └── 引数（対象）
    │   └── オプション（動作を変更）
    └── コマンド

  オプションの形式:
  短い形式: -l, -a, -h（組み合わせ可能: -lah）
  長い形式: --long, --all, --human-readable
  → 長い形式はスクリプトで可読性が高い

基本操作:
  $ pwd                 # 現在のディレクトリ
  $ ls                  # ファイル一覧
  $ cd /path/to/dir     # ディレクトリ移動
  $ cd ~                # ホームに戻る
  $ cd -                # 前のディレクトリに戻る

  $ echo "Hello"        # 文字列出力
  $ date                # 日時表示
  $ whoami              # 現在のユーザー名
  $ hostname            # ホスト名
  $ uname -a            # システム情報

コマンドの種類:
  $ type ls             # コマンドの種類を確認
  → builtin: シェル内蔵（cd, echo, export）
  → file:    外部プログラム（/usr/bin/ls）
  → alias:   エイリアス
  → function: シェル関数

  $ which python        # コマンドのパスを確認
  $ whereis python      # バイナリ+マニュアル+ソースの場所
```

---

## 3. 入出力とリダイレクト

```
3つの標準ストリーム:
  stdin  (0): 標準入力（キーボード）
  stdout (1): 標準出力（画面）
  stderr (2): 標準エラー出力（画面）

リダイレクト:
  $ command > file       # stdoutをファイルに（上書き）
  $ command >> file      # stdoutをファイルに（追記）
  $ command 2> file      # stderrをファイルに
  $ command &> file      # stdout+stderrをファイルに
  $ command < file       # stdinをファイルから
  $ command 2>/dev/null  # エラーを捨てる

パイプ:
  $ command1 | command2  # command1のstdoutをcommand2のstdinへ
  $ ls -la | grep ".md" | wc -l

  パイプラインの例:
  $ cat access.log | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -10
  → アクセスログからIPアドレスのアクセス数トップ10

コマンド置換:
  $ echo "Today is $(date)"
  $ files=$(ls *.md)

ヒアドキュメント:
  $ cat <<EOF
  Hello, $(whoami)!
  Today is $(date).
  EOF
```

---

## 4. キーボードショートカット

```
シェル操作の必須ショートカット:

  移動:
  Ctrl+A    行頭へ
  Ctrl+E    行末へ
  Alt+F     1単語進む
  Alt+B     1単語戻る

  編集:
  Ctrl+U    カーソルから行頭まで削除
  Ctrl+K    カーソルから行末まで削除
  Ctrl+W    直前の1単語を削除
  Ctrl+Y    最後に削除したテキストを貼り付け
  Ctrl+L    画面クリア（clear と同じ）

  履歴:
  Ctrl+R    履歴の検索（逆方向）
  Ctrl+P    前のコマンド（↑と同じ）
  Ctrl+N    次のコマンド（↓と同じ）
  !!        直前のコマンドを再実行
  !$        直前のコマンドの最後の引数

  制御:
  Ctrl+C    実行中のコマンドを中断
  Ctrl+D    EOF送信（シェル終了）
  Ctrl+Z    バックグラウンドに一時停止
```

---

## 実践演習

### 演習1: [基礎] — 基本コマンドの実行

```bash
# 以下のコマンドを実行して出力を確認
pwd && ls -la && whoami && date && uname -a

# リダイレクトとパイプ
echo "Hello, World!" > /tmp/test.txt
cat /tmp/test.txt
echo "Second line" >> /tmp/test.txt
cat /tmp/test.txt | wc -l
```

### 演習2: [応用] — ショートカットの練習

```
以下の操作をキーボードショートカットで実行:
1. 長いコマンドを入力し、Ctrl+A で行頭、Ctrl+E で行末に移動
2. Ctrl+R で過去のコマンドを検索
3. Ctrl+U で行を削除し、Ctrl+Y で復元
4. !! で直前のコマンドを再実行
```

---

## FAQ

### Q1: bash と zsh のどちらを使うべき？

macOS ユーザーは zsh（デフォルト）を推奨。Linux サーバーでは bash が確実。zsh は bash のほぼ上位互換であり、Oh My Zshなどのフレームワークでカスタマイズが容易。スクリプトはPOSIX sh互換で書くのが最も移植性が高い。

### Q2: なぜCLIを学ぶべきか？GUIで十分では？

1. **自動化**: スクリプトで繰り返し作業を自動化
2. **リモート操作**: SSH経由でサーバーを管理
3. **効率性**: マウス操作より速い（慣れれば）
4. **再現性**: コマンド履歴が残る
5. **サーバー**: 多くのサーバーにはGUIがない

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ターミナル | テキストI/Oのウィンドウ |
| シェル | コマンドの解釈・実行。bash/zsh |
| リダイレクト | >, >>, 2>, <, &> |
| パイプ | \| でコマンドを連結 |
| ショートカット | Ctrl+R(検索), Ctrl+A/E(移動), Ctrl+C(中断) |

---

## 次に読むべきガイド
→ [[01-shell-config.md]] — シェル設定

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, No Starch Press, 2019.
