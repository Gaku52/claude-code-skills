# マニュアルとヘルプの活用

> man ページは「Unix界の百科事典」。コマンドの使い方に困ったら、まず man を引く習慣をつけよう。自分で答えを見つける力がつけば、あらゆる未知のコマンドにも対応できる。

## この章で学ぶこと

- [ ] man ページの読み方を完全にマスターする
- [ ] man ページのセクション構造を理解する
- [ ] 各種ヘルプの使い分けができる
- [ ] info ページの操作方法を知る
- [ ] tldr や cheat.sh などのモダンなヘルプツールを活用できる
- [ ] 組み込みコマンドと外部コマンドの違いを理解する
- [ ] オンラインリソースを効果的に活用する方法を知る
- [ ] 自分用の man ページやヘルプドキュメントを作成できる

---

## 1. ヘルプを得る方法の全体像

Linux/Unix環境では、コマンドの使い方を調べる方法が複数存在する。状況に応じて最適な方法を選択することが重要である。

### 1.1 ヘルプ取得方法の一覧と使い分け

```
ヘルプ取得方法の比較:

┌──────────────┬──────────────────┬───────────────────┬──────────────┐
│ 方法         │ 情報量           │ 最適な場面         │ 対象         │
├──────────────┼──────────────────┼───────────────────┼──────────────┤
│ --help       │ 少〜中           │ オプションの確認   │ 外部コマンド │
│ man          │ 多（包括的）     │ 詳細な仕様確認     │ 全般         │
│ info         │ 多（構造的）     │ GNU系の詳細情報    │ GNU系        │
│ help         │ 中               │ シェル組込コマンド │ bash組込     │
│ tldr         │ 少（実例重視）   │ 手早く使い方確認   │ 一般的な CMD │
│ cheat.sh     │ 少〜中（実例）   │ ネット環境で確認   │ 一般的な CMD │
│ type/which   │ 最小             │ コマンドの所在確認 │ 全般         │
│ apropos      │ 検索結果         │ コマンド名が不明時 │ man DB       │
│ whatis       │ 1行説明          │ コマンドの概要確認 │ man DB       │
└──────────────┴──────────────────┴───────────────────┴──────────────┘
```

### 1.2 --help オプション（最も手軽）

```bash
# ほぼ全ての外部コマンドが対応している
ls --help              # ls の簡易ヘルプ
grep --help            # grep の簡易ヘルプ
curl --help            # curl の簡易ヘルプ（カテゴリ別）
curl --help all        # curl の全ヘルプ
git --help             # git のサブコマンド一覧
git commit --help      # git commit の詳細ヘルプ（manページを開く場合も）

# -h が --help の短縮形の場合がある
docker -h
python3 -h
pip3 -h

# コマンドによっては -help（ハイフン1つ）の場合もある
java -help
javac -help

# ヘルプの出力をページャで見る（長い場合）
git --help | less
docker --help | less

# ヘルプの出力から特定のオプションを検索
ls --help | grep -i "sort"        # ソート関連のオプションを探す
curl --help all | grep -i "proxy" # プロキシ関連のオプションを探す

# ヘルプの出力をファイルに保存
git commit --help > /tmp/git-commit-help.txt
```

### 1.3 組み込みコマンドのヘルプ

```bash
# bash/zsh の組み込みコマンドは man ページが無い場合がある
# bash の場合: help コマンドを使う
help                    # 全組み込みコマンドの一覧
help cd                 # cd の詳細ヘルプ
help test               # test の詳細ヘルプ
help [                  # [ (test) の詳細ヘルプ
help for                # for ループの構文
help if                 # if 文の構文
help while              # while ループの構文
help case               # case 文の構文
help read               # read の詳細ヘルプ
help export             # export の詳細ヘルプ
help declare            # declare の詳細ヘルプ
help set                # set の詳細ヘルプ
help shopt              # shopt の詳細ヘルプ
help trap               # trap の詳細ヘルプ
help pushd              # pushd の詳細ヘルプ
help popd               # popd の詳細ヘルプ

# zsh の場合: run-help を使う
# ~/.zshrc に以下を追加:
autoload -Uz run-help
unalias run-help 2>/dev/null
alias help=run-help

# これで help cd や ESC+h でヘルプが表示される
# zshの組み込みコマンドのmanページ
man zshbuiltins         # zsh の全組み込みコマンド
man zshoptions          # zsh のオプション一覧
man zshexpn             # zsh の展開（パラメータ展開等）
man zshmisc             # zsh のその他の機能
man zshzle              # zsh のラインエディタ
man zshcompwid          # zsh の補完ウィジェット
man zshcompsys          # zsh の補完システム
man zshparam            # zsh のパラメータ（変数）

# コマンドの種類を確認する方法
type cd                 # cd is a shell builtin
type ls                 # ls is /bin/ls (外部コマンド)
type ll                 # ll is an alias for ls -lah
type mkcd               # mkcd is a shell function
type if                 # if is a reserved word

# bash での詳細確認
type -a python3         # python3 の全候補を表示
type -t cd              # builtin（種類のみ表示）
type -t ls              # file
type -t ll              # alias

# which コマンド
which python3           # /usr/bin/python3
which -a python3        # 全候補を表示

# where コマンド（zsh限定）
where python3           # PATH上の全候補を表示

# command -v（POSIXスクリプト向け）
command -v cd           # cd（組み込みの場合はコマンド名のみ）
command -v ls           # /bin/ls（外部の場合はパス）
command -v ll           # alias ll='ls -lah'（エイリアスの場合）

# スクリプト内でコマンドの存在確認に使う
if command -v git &>/dev/null; then
    echo "git is installed"
else
    echo "git is not installed"
fi
```

---

## 2. man ページの詳細

### 2.1 man ページのセクション

```bash
# man ページは8つのセクションに分かれている
# 同じ名前で複数のセクションに存在する場合がある

# セクション一覧:
# セクション 1: ユーザーコマンド（一般的なコマンド）
#   例: ls(1), grep(1), git(1), ssh(1), curl(1)
#
# セクション 2: システムコール（カーネルが提供する関数）
#   例: open(2), read(2), write(2), fork(2), exec(2), mmap(2)
#
# セクション 3: ライブラリ関数（Cライブラリ等の関数）
#   例: printf(3), malloc(3), strlen(3), fopen(3), pthread_create(3)
#
# セクション 4: 特殊ファイル（デバイスファイル等）
#   例: null(4), zero(4), random(4), tty(4), console(4)
#
# セクション 5: ファイル形式とプロトコル
#   例: passwd(5), fstab(5), hosts(5), crontab(5), sudoers(5)
#
# セクション 6: ゲームとスクリーンセーバー
#   例: fortune(6), banner(6)
#
# セクション 7: その他の慣例や規約
#   例: ascii(7), utf-8(7), regex(7), signal(7), ip(7), tcp(7)
#
# セクション 8: システム管理コマンド（root用コマンド）
#   例: mount(8), iptables(8), systemctl(8), useradd(8), cron(8)

# セクション指定で man を開く
man ls                  # デフォルト: セクション1の ls
man 5 passwd            # セクション5: /etc/passwd のファイル形式
man 2 open              # セクション2: open() システムコール
man 3 printf            # セクション3: printf() ライブラリ関数
man 7 regex             # セクション7: 正規表現の規約
man 8 mount             # セクション8: mount コマンド

# 同じ名前の複数セクションを確認
man -f passwd           # passwd に関する全セクションを表示
# 出力例:
# passwd (1)           - change user password
# passwd (5)           - the password file

man -f printf
# 出力例:
# printf (1)           - format and print data
# printf (3)           - formatted output conversion

# セクション指定の別の書き方
man passwd.5            # セクション5（一部のシステム）
man -s 5 passwd         # セクション5（-s オプション）
```

### 2.2 man ページの構造

```
典型的な man ページの構成:

┌─────────────────────────────────────────────────┐
│ NAME                                             │
│   コマンド名と1行の簡潔な説明                    │
│                                                  │
│ SYNOPSIS                                         │
│   コマンドの使い方（書式）                        │
│   [ ] は省略可能、... は繰り返し可能             │
│   | は OR（どちらか一方）                         │
│   太字は文字通り入力、下線は置き換え部分          │
│                                                  │
│ DESCRIPTION                                      │
│   コマンドの詳細な説明                            │
│                                                  │
│ OPTIONS                                          │
│   オプションの一覧と説明                          │
│                                                  │
│ ARGUMENTS                                        │
│   引数の説明                                      │
│                                                  │
│ ENVIRONMENT                                      │
│   影響する環境変数                                │
│                                                  │
│ FILES                                            │
│   関連するファイル                                │
│                                                  │
│ EXIT STATUS                                      │
│   終了コード（0=成功、非0=エラー）                │
│                                                  │
│ EXAMPLES                                         │
│   使用例（全てのmanページにあるわけではない）      │
│                                                  │
│ DIAGNOSTICS                                      │
│   エラーメッセージの説明                          │
│                                                  │
│ NOTES / CAVEATS                                  │
│   注意事項                                        │
│                                                  │
│ BUGS                                             │
│   既知のバグ                                      │
│                                                  │
│ SEE ALSO                                         │
│   関連するコマンドやmanページ                     │
│                                                  │
│ AUTHOR                                           │
│   著者情報                                        │
│                                                  │
│ COPYRIGHT                                        │
│   著作権情報                                      │
└─────────────────────────────────────────────────┘
```

### 2.3 SYNOPSIS（書式）の読み方

```bash
# SYNOPSIS の表記規則を理解することは非常に重要

# 例1: ls の SYNOPSIS
# ls [OPTION]... [FILE]...
#
# 分解:
# ls          → コマンド名（必須）
# [OPTION]    → オプション（省略可能 = [ ] で囲まれている）
# ...         → 複数指定可能
# [FILE]      → ファイル名（省略可能）
# ...         → 複数指定可能

# 例2: cp の SYNOPSIS
# cp [OPTION]... [-T] SOURCE DEST
# cp [OPTION]... SOURCE... DIRECTORY
# cp [OPTION]... -t DIRECTORY SOURCE...
#
# 3つの使い方がある:
# 1. cp source dest          （ファイルをコピー）
# 2. cp file1 file2 dir/     （複数ファイルをディレクトリへ）
# 3. cp -t dir/ file1 file2  （-t でターゲットを先に指定）

# 例3: find の SYNOPSIS
# find [-H] [-L] [-P] [-D debugopts] [-Olevel] [path...] [expression]
#
# [-H] [-L] [-P]  → シンボリックリンクの扱い（省略可能、排他的ではない）
# [-D debugopts]   → デバッグオプション（省略可能）
# [-Olevel]        → 最適化レベル（省略可能）
# [path...]        → 検索パス（省略可能、複数可）
# [expression]     → 検索条件（省略可能）

# 例4: git の SYNOPSIS
# git [--version] [--help] [-C <path>] [-c <name>=<value>]
#     [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
#     [-p|--paginate|-P|--no-pager] [--no-replace-objects] [--bare]
#     [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
#     <command> [<args>]
#
# <command>   → サブコマンド（必須 = < > で囲まれている）
# [<args>]    → サブコマンドの引数（省略可能）
# |           → OR（どちらか一方を選択）
# [-p|--paginate]  → -p または --paginate

# 表記規則のまとめ:
# [ ]     → 省略可能（optional）
# < >     → 必須の引数（placeholder）
# ...     → 繰り返し可能（1つ以上）
# |       → OR（選択肢）
# 太字    → リテラル（そのまま入力する文字列）
# 下線    → 変数（ユーザーが指定する値）
```

### 2.4 man ページ内の操作

```bash
# man はデフォルトで less ページャを使用する
# less の全操作が man 内でも使える

# === 基本操作 ===
# j / ↓      → 1行下にスクロール
# k / ↑      → 1行上にスクロール
# Space / f  → 1ページ下にスクロール（forward）
# b          → 1ページ上にスクロール（backward）
# d          → 半ページ下にスクロール
# u          → 半ページ上にスクロール
# g          → 先頭に移動
# G          → 末尾に移動
# q          → 終了

# === 検索 ===
# /pattern   → 前方検索（下方向に検索）
# ?pattern   → 後方検索（上方向に検索）
# n          → 次の検索結果に移動
# N          → 前の検索結果に移動
# &pattern   → pattern にマッチする行のみ表示

# 検索の実用例:
# /EXAMPLES   → 使用例セクションへジャンプ
# /-r         → -r オプションの説明を検索
# /recursive  → "recursive" を含む行を検索
# /-v\b       → -v オプションを正確に検索（\b = 単語境界）

# === マーク ===
# m + 文字   → 現在位置にマークを設定（例: ma）
# ' + 文字   → マーク位置にジャンプ（例: 'a）

# === 画面操作 ===
# h          → ヘルプ画面を表示（less自体のヘルプ）
# =          → ファイル情報（現在位置、行数等）を表示
# v          → $EDITOR でファイルを開く

# === 行番号 ===
# -N         → 行番号を表示/非表示切り替え
# 100g       → 100行目にジャンプ

# 実用的なテクニック:
# 1. man を開いてすぐに /EXAMPLES で使用例を確認
# 2. /OPTIONS でオプション一覧を確認
# 3. man ページを別ウィンドウで開いておき、作業中に参照

# man ページの幅を調整
MANWIDTH=80 man ls        # 80カラム幅で表示
export MANWIDTH=100       # 常に100カラム幅で表示
```

### 2.5 man ページの検索

```bash
# ============================================
# man データベースの検索
# ============================================

# apropos: キーワードで man ページを検索
apropos "copy file"         # "copy file" を含むmanページを検索
apropos -e passwd           # 完全一致で検索
apropos -s 1 network        # セクション1のみで検索
apropos "regular expression" # 正規表現に関するmanページ
apropos -r "^git"           # git で始まるmanページ

# man -k: apropos と同等
man -k "copy file"          # apropos と同じ
man -k "^ls$"               # ls を完全一致で検索
man -k "disk usage"         # ディスク使用量に関するコマンド
man -k "compress"           # 圧縮に関するコマンド

# whatis: コマンドの1行説明を表示
whatis ls                   # ls (1) - list directory contents
whatis passwd               # passwd (1) - change user password
                            # passwd (5) - the password file
whatis mount                # mount (8) - mount a filesystem
whatis printf               # printf (1) - format and print data
                            # printf (3) - formatted output conversion

# man -f: whatis と同等
man -f ls
man -f grep
man -f curl

# manデータベースの更新（検索結果が古い場合）
sudo mandb                  # Linux
sudo /usr/libexec/makewhatis  # macOS（古いバージョン）

# 特定のパスの man ページを開く
man /usr/share/man/man1/ls.1.gz    # パス指定で直接開く
man -l /path/to/custom.man         # ローカルのmanファイルを開く

# man ページをテキストで出力
man ls | col -b > ls.txt           # 整形してテキスト出力
man ls | cat                       # プレーンテキストとして出力

# man ページをPDFで出力（macOS）
man -t ls | open -f -a Preview     # プレビューで表示
man -t ls > ls.ps                  # PostScriptファイルとして出力

# man ページをHTML化
man -H ls                          # ブラウザで表示（groff対応環境）

# man ページの場所を確認
man -w ls                          # /usr/share/man/man1/ls.1.gz
man -wa ls                         # 全候補のパスを表示
```

---

## 3. info ページ

### 3.1 info の基本

```bash
# GNU系のツール（coreutils, bash, gawk等）には
# man よりも詳細な info ページがある

# info ページの表示
info coreutils           # GNU coreutils の info ページ
info bash                # bash の info ページ
info gawk                # gawk の info ページ
info grep                # grep の info ページ（man より詳細）
info sed                 # sed の info ページ
info find                # find の info ページ
info tar                 # tar の info ページ
info make                # make の info ページ
info emacs               # Emacs の info ページ

# 特定のノード（セクション）を直接開く
info coreutils 'ls invocation'
info bash 'Bash Builtins'
info bash 'Shell Expansions'
```

### 3.2 info ページ内の操作

```
info の操作方法:

基本操作:
  Space      → 次のページ
  Backspace  → 前のページ
  n          → 次のノード（セクション）
  p          → 前のノード
  u          → 上位ノード（1つ上の階層）
  t          → トップノード（先頭）
  l          → 直前の位置に戻る（ブラウザの「戻る」と同じ）

ナビゲーション:
  Tab        → 次のハイパーリンクへ移動
  Enter      → リンク先に移動
  [          → 前のノード
  ]          → 次のノード
  m          → メニュー項目を選択（名前入力で移動）
  f          → フォローするリンクを選択
  d          → ディレクトリノード（目次）へ移動

検索:
  s または /  → テキスト検索
  { / }      → 検索結果の前/次

終了:
  q          → 終了

info を使いやすくする:
  - pinfo コマンド（info の改良版、カラー表示対応）
  - Emacs 内の info-mode（M-x info）
  - ブラウザで閲覧: info2html コマンドで HTML 変換
```

### 3.3 man vs info の使い分け

```
man と info の比較:

┌────────────────┬─────────────────────┬──────────────────────┐
│ 特徴           │ man                 │ info                 │
├────────────────┼─────────────────────┼──────────────────────┤
│ 構造           │ フラット（1ファイル）│ ハイパーテキスト      │
│ ナビゲーション │ スクロールのみ       │ ノード間のリンク移動  │
│ 詳細度         │ コマンドの概要       │ チュートリアル含む    │
│ 対象           │ ほぼ全コマンド       │ 主にGNU系ツール      │
│ 学習コスト     │ 低い                │ やや高い             │
│ 使いやすさ     │ less と同じ操作     │ 独自の操作体系       │
│ 更新頻度       │ パッケージに同梱    │ パッケージに同梱     │
└────────────────┴─────────────────────┴──────────────────────┘

実務的な判断:
  - まず man を確認
  - man に記載が少ない場合、または GNU ツールの詳細が必要な場合に info を確認
  - 多くの場合、man で十分
```

---

## 4. tldr（Too Long; Didn't Read）

### 4.1 tldr の概要とインストール

```bash
# tldr は man ページの「使用例だけ」を表示するツール
# コミュニティが管理する実用的な使用例のコレクション

# インストール
brew install tldr            # macOS (Homebrew)
npm install -g tldr          # Node.js
pip3 install tldr            # Python
sudo apt install tldr        # Ubuntu/Debian

# 初回はデータベースの更新が必要
tldr --update

# 基本的な使い方
tldr tar                     # tar の使用例
tldr curl                    # curl の使用例
tldr rsync                   # rsync の使用例
tldr find                    # find の使用例
tldr chmod                   # chmod の使用例
tldr ssh                     # ssh の使用例
tldr git-rebase              # git rebase の使用例
tldr docker-compose          # docker compose の使用例
tldr kubectl-get             # kubectl get の使用例

# プラットフォーム指定
tldr -p linux tar            # Linux版のtar
tldr -p osx pbcopy           # macOS固有のコマンド

# データベースの更新
tldr --update
```

### 4.2 tldr の出力例

```bash
# 例: tldr tar の出力

# tar
# Archiving utility.
# Often combined with a compression method, such as gzip or bzip2.
# More information: https://www.gnu.org/software/tar
#
# Create an archive and write it to a file:
#   tar cf target.tar file1 file2 file3
#
# Create a gzipped archive and write it to a file:
#   tar czf target.tar.gz file1 file2 file3
#
# Create a gzipped archive from a directory using relative paths:
#   tar czf target.tar.gz --directory=path/to/directory .
#
# Extract a (compressed) archive file into the current directory verbosely:
#   tar xvf source.tar[.gz|.bz2|.xz]
#
# Extract a (compressed) archive file into the target directory:
#   tar xf source.tar[.gz|.bz2|.xz] --directory=directory
#
# List the contents of a tar file verbosely:
#   tar tvf source.tar

# man ページと比較して:
# - man tar: 数百行のオプション説明
# - tldr tar: 実用的な6つの使用例のみ
# → 「今すぐこのコマンドを使いたい」場面では tldr が圧倒的に速い
```

### 4.3 tldr の代替・補完ツール

```bash
# ============================================
# cheat.sh（curl で使えるチートシート）
# ============================================

# インストール不要、curlで使える
curl cheat.sh/tar                    # tar のチートシート
curl cheat.sh/curl                   # curl のチートシート
curl cheat.sh/find                   # find のチートシート
curl cheat.sh/git                    # git のチートシート
curl cheat.sh/python/lambda          # Pythonのlambdaの使い方
curl cheat.sh/go/goroutine           # Goのgoroutineの使い方
curl cheat.sh/js/async               # JavaScriptのasync/awaitの使い方

# 検索
curl cheat.sh/~keyword              # キーワードで検索

# シェル関数として定義
cheat() {
    curl -s "cheat.sh/$1"
}
# 使い方: cheat tar, cheat curl, cheat python/list

# ============================================
# navi（インタラクティブなチートシート）
# ============================================

# インストール
brew install navi              # macOS
# cargo install navi           # Rust

# 使い方
navi                           # インタラクティブにチートシートを検索
navi --query "docker"          # docker 関連のコマンドを検索
navi --query "git branch"      # git branch 関連を検索

# Ctrl+G にバインドする（~/.zshrc に追加）
eval "$(navi widget zsh)"

# ============================================
# eg（例を中心としたヘルプ）
# ============================================

# インストール
pip3 install eg

# 使い方
eg tar                         # tar の使用例
eg find                        # find の使用例
eg grep                        # grep の使用例

# ============================================
# bropages（コミュニティ駆動の使用例）
# ============================================

# インストール
# gem install bropages

# 使い方
# bro curl                     # curl の使用例
# bro tar                      # tar の使用例
```

---

## 5. 実践的なヘルプ活用テクニック

### 5.1 効率的な情報収集フロー

```bash
# ============================================
# コマンドの使い方を調べる推奨フロー
# ============================================

# Step 1: tldr で実用例を確認（最速）
tldr rsync

# Step 2: --help でオプション一覧を確認
rsync --help

# Step 3: 詳細が必要なら man を開く
man rsync
# /EXAMPLES で使用例へジャンプ
# /-a で -a オプションの説明を検索

# Step 4: さらに詳しい情報が必要なら info を確認
info rsync

# Step 5: それでも分からない場合はオンラインリソース
# https://explainshell.com/ — コマンドの各部分を分解して説明
# https://www.man7.org/linux/man-pages/ — オンラインmanページ
# https://ss64.com/ — コマンドリファレンス
# Stack Overflow — 具体的な質問の回答
```

### 5.2 explainshell.com の活用

```bash
# explainshell.com は複雑なコマンドを分解して説明するWebサービス

# 例: 以下のコマンドの意味を調べたい場合
# find . -name "*.log" -mtime +30 -exec rm {} \;

# ブラウザで explainshell.com にアクセスし、
# コマンドを入力すると各部分の説明が表示される:
#
# find        → ファイルを検索するコマンド
# .           → 現在のディレクトリから検索
# -name       → ファイル名でマッチング
# "*.log"     → .log で終わるファイル
# -mtime +30  → 30日以上前に変更されたファイル
# -exec       → マッチしたファイルに対してコマンドを実行
# rm {}       → ファイルを削除（{} はマッチしたファイル名に置換）
# \;          → -exec の終了を示す

# コマンドラインから explainshell を開く関数
explain() {
    local url="https://explainshell.com/explain?cmd="
    local encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$*'))")
    open "${url}${encoded}"    # macOS
    # xdg-open "${url}${encoded}"  # Linux
}

# 使い方
# explain find . -name "*.log" -mtime +30 -exec rm {} \;
```

### 5.3 特定の情報を素早く見つけるテクニック

```bash
# ============================================
# man ページから必要な情報だけを素早く取得
# ============================================

# テクニック1: grep でフィルタリング
man ls | grep -A 3 -- "-l"           # -l オプションの説明（3行分）
man curl | grep -A 5 -- "--proxy"    # --proxy の説明
man find | grep -B 2 -A 5 "mtime"   # mtime の説明

# テクニック2: man ページをテキストとして検索
man ls 2>/dev/null | col -b | grep -i "sort"

# テクニック3: 特定セクションだけ表示
man ls | sed -n '/^DESCRIPTION/,/^[A-Z]/p'  # DESCRIPTION セクションのみ

# テクニック4: オプション一覧だけ抽出
man ls | grep "^\s*-"                # オプション行のみ表示

# テクニック5: 終了コードの確認
man ls | sed -n '/^EXIT STATUS/,/^[A-Z]/p'
man grep | sed -n '/^EXIT STATUS/,/^[A-Z]/p'
# grep: 0=マッチあり、1=マッチなし、2=エラー

# テクニック6: 関連コマンドの確認
man ls | sed -n '/^SEE ALSO/,/^[A-Z]/p'

# テクニック7: 環境変数の確認
man ls | sed -n '/^ENVIRONMENT/,/^[A-Z]/p'
man git | sed -n '/^ENVIRONMENT/,/^[A-Z]/p'
```

### 5.4 複数の man ページを比較する

```bash
# ============================================
# 類似コマンドの比較
# ============================================

# cp vs rsync: どちらをコピーに使うべきか?
man cp | head -20                    # cp の概要を確認
man rsync | head -20                 # rsync の概要を確認

# find vs fd: ファイル検索コマンドの比較
man find | wc -l                     # find の man ページの行数（通常 500+）
man fd | wc -l                       # fd の man ページの行数（通常 100-200）

# grep vs rg (ripgrep): テキスト検索の比較
man grep | grep -c "^\s*-"          # grep のオプション数
man rg | grep -c "^\s*-"            # rg のオプション数

# 実用的な比較テクニック
diff <(man cp 2>/dev/null | col -b) <(man rsync 2>/dev/null | col -b)

# 特定のオプションが存在するか確認
man tar | grep -q "\-\-zstd" && echo "zstd対応" || echo "zstd非対応"
```

---

## 6. オンラインリソースの活用

### 6.1 主要なオンラインリソース

```
オンラインで man ページ・ヘルプを確認できるリソース:

■ 公式 man ページ
  - https://www.man7.org/linux/man-pages/
    Linux の公式 man ページ。最新版が常に利用可能
  - https://man.openbsd.org/
    OpenBSD の man ページ。POSIX準拠の参考に

■ 説明・解説系
  - https://explainshell.com/
    コマンドの各部分を視覚的に分解して説明
  - https://www.shellcheck.net/
    シェルスクリプトの文法チェック（オンライン版）

■ チートシート・リファレンス
  - https://cheat.sh/
    curl でアクセス可能なチートシート
  - https://ss64.com/
    各OS（Linux, macOS, Windows）のコマンドリファレンス
  - https://devhints.io/
    各種ツールのチートシート

■ Q&A・コミュニティ
  - https://stackoverflow.com/
    プログラミング全般のQ&A
  - https://unix.stackexchange.com/
    Unix/Linux 特化のQ&A
  - https://askubuntu.com/
    Ubuntu 特化のQ&A
  - https://serverfault.com/
    サーバー管理特化のQ&A

■ チュートリアル
  - https://linuxcommand.org/
    Linux コマンドラインの包括的チュートリアル
  - https://www.gnu.org/software/coreutils/manual/
    GNU coreutils の完全マニュアル
  - https://tldp.org/
    Linux Documentation Project（ガイド集）
```

### 6.2 コマンドラインからオンラインリソースにアクセス

```bash
# ============================================
# ターミナルからオンラインヘルプにアクセス
# ============================================

# cheat.sh をターミナルから使う
curl cheat.sh/tar
curl cheat.sh/find
curl cheat.sh/awk

# シェル関数として定義
cheat() {
    curl -s "cheat.sh/$*" | less -R
}

# howdoi: プログラミングの質問に答える
# pip3 install howdoi
howdoi "extract tar.gz in linux"
howdoi "find files larger than 100MB"
howdoi "python read csv file"

# StackOverflow をターミナルから検索
# pip3 install so
# so "how to find large files in linux"

# Google検索をターミナルから
# Oh My Zsh の web-search プラグインを使う場合:
# google "linux find command examples"
# stackoverflow "bash array loop"

# もしくは関数を定義
google() {
    local query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$*'))")
    open "https://www.google.com/search?q=${query}"    # macOS
    # xdg-open "https://www.google.com/search?q=${query}"  # Linux
}
```

---

## 7. カスタムヘルプの作成

### 7.1 自分用の man ページを作成する

```bash
# ============================================
# カスタム man ページの作成
# ============================================

# man ページは troff/groff マクロで記述する
# 以下は最小限の man ページの例

cat > /tmp/mycommand.1 << 'MANEOF'
.TH MYCOMMAND 1 "2026-02-16" "1.0" "User Commands"
.SH NAME
mycommand \- 自作コマンドの説明
.SH SYNOPSIS
.B mycommand
[\fB\-v\fR]
[\fB\-o\fR \fIoutput\fR]
.IR input
.SH DESCRIPTION
.B mycommand
は入力ファイルを処理して出力するコマンドです。
.PP
詳細な説明をここに書きます。
段落を分けるには .PP を使います。
.SH OPTIONS
.TP
.BR \-v ", " \-\-verbose
詳細な出力を表示します。
.TP
.BR \-o ", " \-\-output " " \fIfile\fR
出力先ファイルを指定します。デフォルトは標準出力です。
.TP
.BR \-h ", " \-\-help
ヘルプメッセージを表示して終了します。
.SH EXAMPLES
.PP
基本的な使い方:
.RS
.nf
mycommand input.txt
.fi
.RE
.PP
出力ファイルを指定:
.RS
.nf
mycommand -o output.txt input.txt
.fi
.RE
.SH EXIT STATUS
.TP
.B 0
成功
.TP
.B 1
一般的なエラー
.TP
.B 2
使い方のエラー（不正な引数）
.SH SEE ALSO
.BR grep (1),
.BR sed (1),
.BR awk (1)
.SH AUTHOR
あなたの名前
.SH BUGS
バグ報告先: https://github.com/user/repo/issues
MANEOF

# 作成した man ページを表示
man /tmp/mycommand.1

# システムにインストール（任意）
sudo cp /tmp/mycommand.1 /usr/local/share/man/man1/
sudo mandb                           # データベース更新（Linux）
```

### 7.2 プロジェクト固有のヘルプシステム

```bash
# ============================================
# プロジェクトに README ベースのヘルプを組み込む
# ============================================

# Makefile に help ターゲットを追加する方法（一般的）
# Makefile:

.PHONY: help
help: ## このヘルプメッセージを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## プロジェクトをビルド
	npm run build

test: ## テストを実行
	npm test

deploy: ## 本番にデプロイ
	./scripts/deploy.sh

lint: ## リンターを実行
	npm run lint

clean: ## ビルド成果物を削除
	rm -rf dist/ node_modules/

# 使い方: make help と実行すると
# build                プロジェクトをビルド
# clean                ビルド成果物を削除
# deploy               本番にデプロイ
# lint                 リンターを実行
# test                 テストを実行
```

```bash
# ============================================
# シェルスクリプトにヘルプを組み込む
# ============================================

#!/bin/bash
# my-deploy.sh — デプロイスクリプト

set -euo pipefail

VERSION="1.0.0"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] <environment>

デプロイスクリプト - アプリケーションを指定環境にデプロイします。

Arguments:
  environment    デプロイ先の環境 (staging|production)

Options:
  -b, --branch BRANCH    デプロイするブランチ (default: main)
  -d, --dry-run          ドライラン（実際にはデプロイしない）
  -f, --force            確認なしでデプロイ
  -v, --verbose          詳細な出力を表示
  -h, --help             このヘルプメッセージを表示
  --version              バージョンを表示

Examples:
  $(basename "$0") staging                      # staging にデプロイ
  $(basename "$0") -b feature/new production    # 特定ブランチを production に
  $(basename "$0") -d production                # ドライラン
  $(basename "$0") -f -v production             # 強制 + 詳細表示

Environment Variables:
  DEPLOY_TOKEN    デプロイ用の認証トークン
  SLACK_WEBHOOK   Slack通知用のWebhook URL

Exit Codes:
  0    成功
  1    一般的なエラー
  2    引数エラー
  3    認証エラー

See Also:
  docs/deployment.md — デプロイの詳細ガイド
  docs/rollback.md   — ロールバック手順

Version: ${VERSION}
EOF
}

# オプション解析
BRANCH="main"
DRY_RUN=false
FORCE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--branch)  BRANCH="$2"; shift 2 ;;
        -d|--dry-run) DRY_RUN=true; shift ;;
        -f|--force)   FORCE=true; shift ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help)    usage; exit 0 ;;
        --version)    echo "$(basename "$0") ${VERSION}"; exit 0 ;;
        -*)           echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)            ENVIRONMENT="$1"; shift ;;
    esac
done

# 引数チェック
if [[ -z "${ENVIRONMENT:-}" ]]; then
    echo "Error: environment is required" >&2
    usage >&2
    exit 2
fi

echo "Deploying branch ${BRANCH} to ${ENVIRONMENT}..."
```

---

## 8. シェルの組み込みドキュメント

### 8.1 bash の特殊変数リファレンス

```bash
# ============================================
# bash/zsh の特殊変数（ヘルプで確認困難なもの）
# ============================================

# プロセス関連
$$                # 現在のシェルのPID
$!                # 最後に実行したバックグラウンドプロセスのPID
$?                # 最後に実行したコマンドの終了コード
$-                # 現在のシェルオプション

# 引数関連（スクリプト内で使用）
$0                # スクリプト名
$1 〜 $9          # 位置パラメータ（引数）
${10}             # 10番目以降は {} が必要
$#                # 引数の数
$@                # 全引数（個別にクォート）
$*                # 全引数（1つの文字列として）
"$@"              # 推奨: 各引数を個別にクォートして展開
"$*"              # 全引数を1つの文字列として展開

# その他
$_                # 直前のコマンドの最後の引数
$PPID             # 親プロセスのPID
$RANDOM           # ランダムな整数（0-32767）
$LINENO           # 現在の行番号
$SECONDS          # シェル起動からの経過秒数
$BASH_VERSION     # bash のバージョン
$ZSH_VERSION      # zsh のバージョン

# 確認方法
man bash           # bash の全ドキュメント
man bash | grep -A 3 "Special Parameters"
help                # bash 組み込みコマンドの一覧
```

### 8.2 bash のテスト演算子リファレンス

```bash
# ============================================
# テスト演算子（test / [ ] / [[ ]] で使用）
# ============================================

# ファイルテスト
[ -e file ]        # file が存在する
[ -f file ]        # 通常のファイル
[ -d file ]        # ディレクトリ
[ -L file ]        # シンボリックリンク
[ -r file ]        # 読み取り可能
[ -w file ]        # 書き込み可能
[ -x file ]        # 実行可能
[ -s file ]        # サイズが0でない
[ -p file ]        # 名前付きパイプ
[ -S file ]        # ソケット
[ -b file ]        # ブロックデバイス
[ -c file ]        # キャラクターデバイス
[ file1 -nt file2 ]  # file1 が file2 より新しい
[ file1 -ot file2 ]  # file1 が file2 より古い
[ file1 -ef file2 ]  # 同じinode（ハードリンク）

# 文字列テスト
[ -z "$str" ]      # 空文字列
[ -n "$str" ]      # 非空文字列
[ "$a" = "$b" ]    # 等しい
[ "$a" != "$b" ]   # 等しくない
[[ "$a" < "$b" ]]  # 辞書順で小さい（[[ ]]内のみ）
[[ "$a" > "$b" ]]  # 辞書順で大きい（[[ ]]内のみ）
[[ "$a" =~ regex ]]  # 正規表現マッチ（[[ ]]内のみ）
[[ "$a" == pattern ]]  # パターンマッチ（[[ ]]内のみ）

# 数値テスト
[ "$a" -eq "$b" ]  # 等しい
[ "$a" -ne "$b" ]  # 等しくない
[ "$a" -lt "$b" ]  # より小さい
[ "$a" -le "$b" ]  # 以下
[ "$a" -gt "$b" ]  # より大きい
[ "$a" -ge "$b" ]  # 以上

# 論理演算
[ condition1 ] && [ condition2 ]   # AND
[ condition1 ] || [ condition2 ]   # OR
[ ! condition ]                     # NOT
[[ condition1 && condition2 ]]     # AND（[[ ]]内）
[[ condition1 || condition2 ]]     # OR（[[ ]]内）

# 確認方法
help test          # bash の test コマンドのヘルプ
man test           # test の man ページ
```

---

## 9. 実践演習

### 演習1: [基礎] ── man ページの基本操作

```bash
# 課題: 以下の情報を man ページから見つけてください

# 1. ls コマンドで「ファイルサイズの大きい順」にソートするオプション
man ls
# → /sort で検索 → -S オプション

# 2. grep コマンドで「マッチしなかった行」を表示するオプション
man grep
# → /invert で検索 → -v オプション

# 3. find コマンドで「7日以内に更新されたファイル」を検索する式
man find
# → /mtime で検索 → -mtime -7

# 4. chmod コマンドで「再帰的にパーミッションを変更」するオプション
man chmod
# → /recursive で検索 → -R オプション

# 5. curl コマンドで「HTTPヘッダーのみ」を表示するオプション
man curl
# → /header.*only で検索 → -I オプション

# 練習: 実際に man ページを開いて各情報を見つけてください
```

### 演習2: [中級] ── 情報収集ワークフロー

```bash
# 課題: 知らないコマンド「rsync」の使い方を5分以内に調べる

# Step 1: whatis で概要確認（10秒）
whatis rsync
# rsync (1) - a fast, versatile, remote (and local) file-copying tool

# Step 2: tldr で使用例確認（30秒）
tldr rsync
# → 基本的なコピー、リモートコピー、同期の例が表示される

# Step 3: --help で主要オプション確認（1分）
rsync --help | head -30
# → 主要なオプションの概要

# Step 4: man ページで特定の情報を深掘り（3分）
man rsync
# /--exclude で除外パターンの使い方を確認
# /--delete で同期時の削除オプションを確認
# /EXAMPLES で実用例を確認

# 結果のまとめ:
# rsync -avh source/ dest/           # ローカルコピー
# rsync -avh source/ user@host:dest/ # リモートコピー
# rsync -avh --delete source/ dest/  # 同期（不要ファイル削除）
# rsync -avhn source/ dest/          # ドライラン（-n）
```

### 演習3: [中級] ── コマンドの違いを理解する

```bash
# 課題: 以下のコマンドの違いを man ページを参考に説明する

# 1. cp vs rsync
# man cp | head -5
# man rsync | head -5
# cp: ファイルのコピー（常に全ファイルをコピー）
# rsync: 差分転送（変更されたファイルのみ転送可能、リモート対応）

# 2. find vs locate
# man find | head -5
# man locate | head -5
# find: リアルタイムでファイルシステムを走査して検索
# locate: 事前に作成したデータベースから検索（高速だが最新でない場合がある）

# 3. grep vs egrep vs fgrep
# man grep で DESCRIPTION を確認
# grep: 基本正規表現（BRE）を使用
# egrep = grep -E: 拡張正規表現（ERE）を使用
# fgrep = grep -F: 固定文字列のみ（正規表現なし、最速）

# 4. kill vs killall vs pkill
# man kill | head -5
# man killall | head -5
# man pkill | head -5
# kill: PID指定でシグナル送信
# killall: プロセス名指定でシグナル送信（全一致）
# pkill: パターンマッチでシグナル送信

# 5. cat vs less vs more
# man cat | head -5
# man less | head -5
# man more | head -5
# cat: ファイルを全て出力（ページング無し）
# less: ページャ（前後スクロール、検索対応）
# more: 古いページャ（前方スクロールのみ）
```

### 演習4: [上級] ── ヘルプシステムを活用したスクリプト作成

```bash
# 課題: ヘルプ機能付きのバックアップスクリプトを作成する
# 要件:
# 1. --help オプションで使い方を表示
# 2. --version オプションでバージョンを表示
# 3. 引数チェックと適切なエラーメッセージ
# 4. 終了コードの適切な設定

cat > /tmp/backup.sh << 'SCRIPT_EOF'
#!/bin/bash
# backup.sh — シンプルなバックアップスクリプト
#
# usage: backup.sh [OPTIONS] <source> <destination>
# See: backup.sh --help for details

set -euo pipefail

readonly VERSION="1.0.0"
readonly SCRIPT_NAME="$(basename "$0")"

# カラー定義
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly NC='\033[0m' # No Color

# ヘルプメッセージ
usage() {
    cat << HELP
Usage: ${SCRIPT_NAME} [OPTIONS] <source> <destination>

指定したソースディレクトリをバックアップします。

Arguments:
  source          バックアップ元のディレクトリ
  destination     バックアップ先のディレクトリ

Options:
  -c, --compress     バックアップを圧縮 (tar.gz)
  -e, --exclude PAT  除外パターン（複数指定可）
  -n, --dry-run      ドライラン（実際にはコピーしない）
  -v, --verbose      詳細な出力
  -q, --quiet        最小限の出力
  -h, --help         このヘルプを表示
  --version          バージョンを表示

Examples:
  ${SCRIPT_NAME} ~/Documents /backup/
  ${SCRIPT_NAME} -c ~/projects /backup/
  ${SCRIPT_NAME} -e "*.tmp" -e "node_modules" ~/src /backup/
  ${SCRIPT_NAME} -nv ~/data /backup/

Exit Codes:
  0  成功
  1  一般エラー
  2  引数エラー（不正なオプション、足りない引数）
  3  ソースが存在しない
  4  バックアップ先に書き込めない

Version: ${VERSION}
Report bugs: https://github.com/example/backup/issues
HELP
}

# エラーメッセージ
error() {
    echo -e "${RED}Error: $*${NC}" >&2
}

# 情報メッセージ
info() {
    if [[ "$QUIET" == false ]]; then
        echo -e "${GREEN}Info: $*${NC}"
    fi
}

# 警告メッセージ
warn() {
    echo -e "${YELLOW}Warning: $*${NC}" >&2
}

# デフォルト値
COMPRESS=false
DRY_RUN=false
VERBOSE=false
QUIET=false
EXCLUDES=()

# オプション解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--compress)  COMPRESS=true; shift ;;
        -e|--exclude)   EXCLUDES+=("$2"); shift 2 ;;
        -n|--dry-run)   DRY_RUN=true; shift ;;
        -v|--verbose)   VERBOSE=true; shift ;;
        -q|--quiet)     QUIET=true; shift ;;
        -h|--help)      usage; exit 0 ;;
        --version)      echo "${SCRIPT_NAME} ${VERSION}"; exit 0 ;;
        -*)             error "Unknown option: $1"; echo; usage >&2; exit 2 ;;
        *)              break ;;
    esac
done

# 引数チェック
if [[ $# -lt 2 ]]; then
    error "source and destination are required"
    echo "Try '${SCRIPT_NAME} --help' for more information." >&2
    exit 2
fi

SOURCE="$1"
DESTINATION="$2"

# ソースの存在チェック
if [[ ! -d "$SOURCE" ]]; then
    error "Source directory not found: $SOURCE"
    exit 3
fi

# バックアップ実行
info "Backing up: $SOURCE -> $DESTINATION"
info "Compress: $COMPRESS, Dry-run: $DRY_RUN"

if [[ "$DRY_RUN" == true ]]; then
    info "DRY RUN - no files will be copied"
fi

echo "Backup complete!"
SCRIPT_EOF

chmod +x /tmp/backup.sh

# テスト
/tmp/backup.sh --help
/tmp/backup.sh --version
/tmp/backup.sh -nv ~/Documents /tmp/backup/
```

### 演習5: [上級] ── カスタムヘルプ関数の作成

```bash
# 課題: 自分がよく使うコマンドのチートシートを
# ターミナルからすぐに参照できる仕組みを作る

# ~/.zsh/cheatsheets/ ディレクトリに各コマンドのチートシートを保存
mkdir -p ~/.zsh/cheatsheets

# git のチートシート例
cat > ~/.zsh/cheatsheets/git << 'CHEAT_EOF'
=== Git Cheatsheet ===

基本操作:
  git init                    リポジトリの初期化
  git clone <url>             リポジトリのクローン
  git add <file>              ステージング
  git commit -m "msg"         コミット
  git push                    リモートにプッシュ
  git pull                    リモートからプル

ブランチ:
  git branch                  ブランチ一覧
  git switch -c <name>        新しいブランチを作成して切り替え
  git switch <name>           ブランチを切り替え
  git merge <branch>          マージ
  git rebase <branch>         リベース

取り消し:
  git reset --soft HEAD~1     直前のコミットを取り消し（変更は保持）
  git reset --hard HEAD~1     直前のコミットを完全に取り消し
  git restore <file>          ファイルの変更を取り消し
  git restore --staged <file> ステージングを取り消し

差分:
  git diff                    ワーキングツリーの差分
  git diff --staged           ステージングの差分
  git log --oneline -10       直近10件のログ
CHEAT_EOF

# docker のチートシート例
cat > ~/.zsh/cheatsheets/docker << 'CHEAT_EOF'
=== Docker Cheatsheet ===

コンテナ操作:
  docker run -it <image> bash     コンテナを起動してbashに入る
  docker ps                       実行中のコンテナ一覧
  docker ps -a                    全コンテナ一覧
  docker stop <container>         コンテナ停止
  docker rm <container>           コンテナ削除
  docker exec -it <c> bash        コンテナ内でbash実行

イメージ操作:
  docker images                   イメージ一覧
  docker build -t <tag> .         イメージビルド
  docker rmi <image>              イメージ削除
  docker pull <image>             イメージ取得

Docker Compose:
  docker compose up -d            バックグラウンドで起動
  docker compose down             停止・削除
  docker compose logs -f          ログをフォロー
  docker compose exec <svc> bash  サービス内でbash実行

クリーンアップ:
  docker system prune -af         未使用リソースを全削除
  docker volume prune             未使用ボリュームを削除
CHEAT_EOF

# チートシートを表示する関数
cs() {
    local sheet="$HOME/.zsh/cheatsheets/$1"
    if [[ -f "$sheet" ]]; then
        bat --plain "$sheet" 2>/dev/null || cat "$sheet"
    elif [[ -z "$1" ]]; then
        echo "Available cheatsheets:"
        ls ~/.zsh/cheatsheets/
    else
        echo "Cheatsheet not found: $1"
        echo "Available: $(ls ~/.zsh/cheatsheets/ | tr '\n' ' ')"
        echo ""
        echo "Falling back to tldr..."
        tldr "$1"
    fi
}

# 使い方:
# cs               → 利用可能なチートシートの一覧
# cs git           → git のチートシート
# cs docker        → docker のチートシート
# cs unknown       → tldr にフォールバック
```

---

## まとめ

| 方法 | 用途 | 情報量 | 速度 |
|------|------|--------|------|
| --help | 手軽な簡易ヘルプ | 少〜中 | 最速 |
| man | 包括的なマニュアル | 多 | 速い |
| info | GNU系の詳細ドキュメント | 多 | 普通 |
| help | bash組み込みコマンド | 中 | 速い |
| tldr | 実用的な使用例 | 少 | 速い |
| cheat.sh | オンラインチートシート | 少〜中 | ネット依存 |
| apropos | コマンド名を知らない時の検索 | 検索結果 | 速い |
| whatis | コマンドの1行説明 | 最小 | 最速 |
| type/which | コマンドの場所・種類 | 最小 | 最速 |
| explainshell | 複雑なコマンドの分解説明 | 中 | ネット依存 |

### 習得すべきスキル

1. **man ページの高速ナビゲーション** -- /検索、g/G移動、セクションジャンプ
2. **SYNOPSIS の読み方** -- [ ]、< >、...、| の意味を即座に理解する
3. **適切なヘルプ方法の選択** -- 状況に応じて tldr / man / info を使い分ける
4. **セクション番号の理解** -- 同名の man ページが複数ある場合に正しいものを開く
5. **組み込み vs 外部コマンドの区別** -- type で確認し、適切なヘルプ方法を選ぶ
6. **オンラインリソースの活用** -- ローカルの man で不十分な場合の代替手段を知る

---

## 次に読むべきガイド
→ [[../01-file-operations/00-navigation.md]] — ディレクトリ移動と一覧

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, No Starch Press, 2019.
2. Powers, S., Peek, J., O'Reilly, T., Loukides, M. "Unix Power Tools." 3rd Ed, O'Reilly, 2002.
3. Kerrisk, M. "The Linux Programming Interface." No Starch Press, 2010.
4. GNU Coreutils マニュアル: https://www.gnu.org/software/coreutils/manual/
5. tldr-pages プロジェクト: https://github.com/tldr-pages/tldr
6. cheat.sh プロジェクト: https://github.com/chubin/cheat.sh
7. explainshell.com: https://explainshell.com/
8. man-pages プロジェクト: https://www.man7.org/linux/man-pages/
