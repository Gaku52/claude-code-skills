# ターミナルとシェルの基礎

> ターミナルは「テキストでコンピュータと対話する窓口」であり、シェルは「コマンドを解釈して実行する通訳者」である。この2つの概念を正確に理解することが、Linux/Unix のコマンドライン操作を効率的に習得するための第一歩となる。

## この章で学ぶこと

- [ ] ターミナル、シェル、コンソールの違いを正確に説明できる
- [ ] シェルの歴史と各種シェルの特徴を理解する
- [ ] コマンドの基本構造と実行の仕組みを理解する
- [ ] 入出力とリダイレクト・パイプを使いこなせる
- [ ] キーボードショートカットで効率的にシェル操作ができる
- [ ] 環境変数とシェルの内部動作を理解する
- [ ] ターミナルエミュレータの選定と設定ができる

---

## 1. 基本概念 — ターミナル・シェル・コンソールの正確な理解

### 1.1 ターミナル（端末エミュレータ）

```
ターミナルとは:
  テキストの入出力を行うプログラム（GUI アプリケーション）
  物理端末（VT100, VT220等）をソフトウェアで再現したもの

  歴史的背景:
  1960s-70s: メインフレームに接続する物理端末（テレタイプ, TTY）
  1980s:     ビデオ端末（VT100等）が登場
  1990s-:    GUI上で動作する「端末エミュレータ」が主流に

  代表的な端末エミュレータ:
  ┌────────────────────┬────────────────────────────────────┐
  │ エミュレータ       │ 特徴                               │
  ├────────────────────┼────────────────────────────────────┤
  │ iTerm2             │ macOS向け。分割・検索・自動補完     │
  │ Alacritty          │ GPU描画。高速。Rust製               │
  │ Warp               │ AI統合。モダンUI。macOS/Linux       │
  │ Windows Terminal   │ Windows公式。タブ・分割対応         │
  │ GNOME Terminal     │ GNOME標準。安定                     │
  │ Konsole            │ KDE標準。高機能                     │
  │ kitty              │ GPU描画。画像表示対応               │
  │ WezTerm            │ Rust製。クロスプラットフォーム      │
  │ Hyper              │ Electron製。Web技術で拡張可能       │
  │ Terminator         │ 画面分割特化。Python製              │
  └────────────────────┴────────────────────────────────────┘

  TTY（Teletypewriter）の名残:
  $ tty                    # 現在の端末デバイスを表示
  /dev/pts/0               # 仮想端末（pseudo-terminal slave）
  /dev/tty1                # 仮想コンソール

  $ who                    # ログイン中のユーザーと端末
  gaku     pts/0        2026-02-15 10:00 (192.168.1.100)
  gaku     tty1         2026-02-15 09:00

  pts = pseudo-terminal slave（SSH, ターミナルエミュレータ経由）
  tty = 仮想コンソール（物理端末相当）
```

### 1.2 シェル

```
シェルとは:
  コマンドを解釈・実行するプログラム（コマンドラインインタプリタ）
  ユーザーとカーネルの間に位置する「殻（shell）」

  シェルの役割:
  1. コマンドラインの解析（パース）
  2. 変数展開・コマンド置換
  3. リダイレクトとパイプの設定
  4. プログラムの実行（fork + exec）
  5. ジョブ制御
  6. スクリプト実行（プログラミング言語としての機能）

  シェルの歴史:
  ┌──────┬───────────────────────┬──────────────────────────────────┐
  │ 年   │ シェル                │ 特徴                             │
  ├──────┼───────────────────────┼──────────────────────────────────┤
  │ 1971 │ Thompson Shell        │ 最初のUnixシェル                 │
  │ 1977 │ Bourne Shell (sh)     │ スクリプト言語機能。POSIX基盤    │
  │ 1978 │ C Shell (csh)         │ C言語風構文。BSD系               │
  │ 1983 │ Korn Shell (ksh)      │ sh互換 + cshの便利機能           │
  │ 1989 │ Bash                  │ GNU版sh。Linux標準               │
  │ 1990 │ Zsh                   │ 最強の対話シェル                 │
  │ 2005 │ Fish                  │ ユーザーフレンドリー設計         │
  │ 2019 │ Nushell               │ 構造化データ処理。Rust製         │
  │ 2021 │ Oil Shell             │ bash互換 + モダン構文            │
  └──────┴───────────────────────┴──────────────────────────────────┘

  現在のシェル確認:
  $ echo $SHELL             # ログインシェル（/etc/passwd で設定）
  /bin/zsh

  $ echo $0                 # 現在実行中のシェル
  -zsh

  $ cat /etc/shells         # 利用可能なシェル一覧
  /bin/sh
  /bin/bash
  /bin/zsh
  /usr/bin/fish

  シェルの変更:
  $ chsh -s /bin/zsh        # ログインシェルをzshに変更
  $ exec bash               # 現在のセッションのみ別シェルに切替
```

### 1.3 コンソール

```
コンソールとは:
  物理的なキーボード＋画面の組み合わせ、またはその仮想版
  サーバールームで直接操作する端末

  仮想コンソール（Linux）:
  Ctrl+Alt+F1   → tty1（多くのディストリでGUIが動作）
  Ctrl+Alt+F2   → tty2（テキストコンソール）
  ...
  Ctrl+Alt+F6   → tty6（テキストコンソール）

  用途:
  - GUIがフリーズした時の緊急操作
  - サーバーの直接操作（データセンター）
  - インストーラ操作
  - カーネルパニック時のデバッグ

  シリアルコンソール:
  - 物理シリアルポート（RS-232）経由の接続
  - ネットワーク不通時のサーバー管理に不可欠
  - BMC/IPMI/iLO/iDRAC のリモートコンソールも同様の概念
```

### 1.4 三者の関係

```
関係図:

  ┌─────────────────────────────────────────────────┐
  │            ターミナルエミュレータ                │
  │  (iTerm2 / Alacritty / Windows Terminal)        │
  │                                                 │
  │  ┌─────────────────────────────────────────┐    │
  │  │              シェル (zsh/bash)           │    │
  │  │                                         │    │
  │  │  $ ls -la /home/user                    │    │
  │  │  total 48                               │    │
  │  │  drwxr-xr-x 12 user user 4096 ...      │    │
  │  │                                         │    │
  │  │  ┌─────────────────────────────────┐    │    │
  │  │  │    外部コマンド / プログラム    │    │    │
  │  │  │    (ls, grep, git, python...)   │    │    │
  │  │  └─────────────────────────────────┘    │    │
  │  │                                         │    │
  │  └─────────────────────────────────────────┘    │
  │                                                 │
  └─────────────────────────────────────────────────┘
                         │
                         ↓
  ┌─────────────────────────────────────────────────┐
  │              カーネル (Linux Kernel)             │
  │     システムコール → ハードウェア制御            │
  └─────────────────────────────────────────────────┘

  処理フロー:
  1. ユーザーがターミナルにキー入力
  2. ターミナルが入力をシェルに渡す
  3. シェルがコマンドを解析
  4. シェルがfork()でプロセスを生成
  5. 子プロセスでexec()によりコマンドを実行
  6. コマンドの出力がシェル経由でターミナルに返る
  7. ターミナルが画面に出力を描画
```

---

## 2. コマンドの基本構造

### 2.1 コマンドの一般的な構文

```bash
# コマンドの基本構造
$ command [options] [arguments]

# 具体例
$ ls -la /home/user
#  │   │   └── 引数（対象ディレクトリ）
#  │   └── オプション（-l: 詳細表示, -a: 隠しファイル含む）
#  └── コマンド

# オプションの形式
# 短い形式（1文字）: ハイフン1つ
$ ls -l
$ ls -a
$ ls -la                    # 短いオプションは結合可能
$ ls -l -a                  # 分けても同じ

# 長い形式（単語）: ハイフン2つ
$ ls --long
$ ls --all
$ ls --all --long           # 長いオプションは結合不可

# 値を取るオプション
$ grep -n "pattern" file    # 短い形式（スペース区切り）
$ grep -n"pattern" file     # 短い形式（スペースなし、一部コマンドで可能）
$ grep --line-number=5 file # 長い形式（=で接続）

# -- の意味（オプション終端マーカー）
$ rm -- -dangerous-file     # ハイフンで始まるファイル名を引数として扱う
$ grep -- "-pattern" file   # ハイフンで始まるパターンを検索
```

### 2.2 コマンドの種類と優先順位

```bash
# type コマンドでコマンドの種類を確認
$ type cd
cd is a shell builtin

$ type ls
ls is aliased to 'ls --color=auto'

$ type grep
grep is /usr/bin/grep

$ type -a echo              # 同名の全てのコマンドを表示
echo is a shell builtin
echo is /usr/bin/echo
echo is /bin/echo

# コマンドの実行優先順位（高い順）:
# 1. エイリアス (alias)
# 2. 関数 (function)
# 3. ビルトインコマンド (builtin)
# 4. ハッシュテーブルのキャッシュ
# 5. PATH上の外部コマンド

# 特定の種類を強制的に実行
$ \ls                       # エイリアスを無視して外部コマンド ls を実行
$ command ls                # エイリアスと関数を無視
$ builtin echo "hello"     # ビルトイン版を強制使用
$ /usr/bin/echo "hello"    # フルパスで外部コマンドを直接指定

# which / whereis でパスを確認
$ which python3             # 最初に見つかるパス
/usr/bin/python3

$ which -a python3          # 全てのパス
/usr/bin/python3
/usr/local/bin/python3

$ whereis python3           # バイナリ + マニュアル + ソース
python3: /usr/bin/python3 /usr/lib/python3 /usr/share/man/man1/python3.1.gz
```

### 2.3 コマンドの結合と制御演算子

```bash
# セミコロン: 順次実行（前のコマンドの成否に関係なく実行）
$ mkdir test; cd test; touch file.txt

# && (AND): 前のコマンドが成功した場合のみ次を実行
$ mkdir project && cd project && git init
# mkdir が失敗したら cd は実行されない

# || (OR): 前のコマンドが失敗した場合のみ次を実行
$ cd /nonexistent || echo "ディレクトリが見つかりません"

# 組み合わせパターン
$ make && make test || echo "ビルドまたはテスト失敗"

# グループ化
$ { command1; command2; }         # 現在のシェルで実行
$ (command1; command2)            # サブシェルで実行

# 実務での使い分け例
# デプロイスクリプト的な連続コマンド
$ git pull && npm install && npm run build && echo "デプロイ準備完了" || echo "エラー発生"

# サブシェルで一時的にディレクトリ変更
$ (cd /tmp && wget https://example.com/file.tar.gz && tar xzf file.tar.gz)
# 元のディレクトリに自動的に戻る

# 複数コマンドの終了ステータス
$ echo $?                    # 直前のコマンドの終了ステータス
0                            # 0 = 成功, 1-255 = 失敗

# PIPESTATUS（bash）/ pipestatus（zsh）でパイプライン各段のステータス
$ false | true | false
$ echo ${PIPESTATUS[@]}      # bash
1 0 1
$ echo ${pipestatus[@]}      # zsh
1 0 1
```

### 2.4 基本コマンド一覧

```bash
# === 情報取得系 ===
$ pwd                         # 現在のディレクトリ（Print Working Directory）
/home/gaku/projects

$ ls                          # ファイル一覧
$ ls -la                      # 詳細 + 隠しファイル
$ ls -lah                     # 人間が読みやすいサイズ表示
$ ls -lt                      # 更新日時順
$ ls -lS                      # サイズ順
$ ls -R                       # 再帰的に表示

$ echo "Hello, World!"        # 文字列出力
$ echo -e "Line1\nLine2"      # エスケープシーケンス解釈
$ echo -n "no newline"        # 改行なし

$ date                        # 日時表示
Sun Feb 15 10:30:00 JST 2026
$ date +"%Y-%m-%d %H:%M:%S"   # フォーマット指定
2026-02-15 10:30:00
$ date -u                     # UTC表示
$ date -d "2 days ago"        # 相対日時（GNU date）

$ whoami                      # 現在のユーザー名
gaku

$ id                          # UID, GID, 所属グループ
uid=1000(gaku) gid=1000(gaku) groups=1000(gaku),27(sudo),998(docker)

$ hostname                    # ホスト名
dev-server

$ uname -a                    # カーネル情報
Linux dev-server 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux

$ uptime                      # 稼働時間とロードアベレージ
10:30:00 up 45 days, 3:22, 2 users, load average: 0.15, 0.10, 0.05

$ cat /etc/os-release         # ディストリビューション情報
NAME="Ubuntu"
VERSION="22.04.3 LTS (Jammy Jellyfish)"

# === ファイル操作系（基本） ===
$ cd /path/to/dir             # ディレクトリ移動
$ cd ~                        # ホームに戻る（cd だけでも同じ）
$ cd -                        # 前のディレクトリに戻る
$ cd ..                       # 親ディレクトリに移動
$ cd ../..                    # 2つ上の親ディレクトリ

$ touch newfile.txt           # 空ファイル作成 / タイムスタンプ更新
$ mkdir newdir                # ディレクトリ作成
$ mkdir -p a/b/c              # 中間ディレクトリも含めて作成

$ cp file1 file2              # ファイルコピー
$ cp -r dir1 dir2             # ディレクトリコピー
$ mv file1 file2              # 移動 / リネーム
$ rm file                     # ファイル削除
$ rm -r dir                   # ディレクトリ削除（再帰的）
$ rm -rf dir                  # 強制的に再帰削除（※慎重に）

# === ヘルプ系 ===
$ command --help              # 簡易ヘルプ
$ man command                 # マニュアルページ
$ info command                # GNU infoドキュメント
$ apropos keyword             # キーワードからコマンドを検索
$ whatis command              # コマンドの1行説明
```

---

## 3. 入出力とリダイレクト

### 3.1 ファイルディスクリプタと標準ストリーム

```
Unix/Linux の I/O モデル:
  全てのI/Oは「ファイルディスクリプタ（fd）」を通じて行われる

  標準ストリーム:
  ┌────────┬──────┬────────────────┬─────────────────────┐
  │ 名前   │ fd   │ デフォルト     │ 用途                │
  ├────────┼──────┼────────────────┼─────────────────────┤
  │ stdin  │ 0    │ キーボード     │ プログラムへの入力  │
  │ stdout │ 1    │ 画面           │ 通常の出力          │
  │ stderr │ 2    │ 画面           │ エラーメッセージ    │
  └────────┴──────┴────────────────┴─────────────────────┘

  fd 3以降はプログラムが任意に使用可能

  ┌─────────┐     stdin(0)     ┌──────────┐     stdout(1)    ┌─────────┐
  │キーボード├────────────────→│  プロセス  ├────────────────→│  画面   │
  └─────────┘                  │  (bash)   ├────────────────→│         │
                               └──────────┘     stderr(2)    └─────────┘
```

### 3.2 出力リダイレクト

```bash
# stdout をファイルに書き込み（上書き）
$ echo "Hello" > output.txt
$ ls -la > filelist.txt

# stdout をファイルに追記
$ echo "World" >> output.txt
$ date >> logfile.txt

# stderr をファイルに書き込み
$ ls /nonexistent 2> error.log

# stdout と stderr を別々のファイルに
$ command > stdout.log 2> stderr.log

# stdout と stderr を同じファイルに
$ command &> combined.log          # bash 4+
$ command > combined.log 2>&1      # POSIX互換（順序重要！）
# 注意: 2>&1 > file は意図通りに動かない

# stderr を stdout にマージ（パイプに流す時に便利）
$ command 2>&1 | grep "Error"

# 出力を捨てる
$ command > /dev/null              # stdout を捨てる
$ command 2> /dev/null             # stderr を捨てる
$ command &> /dev/null             # 両方捨てる

# 実務例: cronジョブでの出力制御
# 成功ログのみファイルに、エラーはメール通知
$ /usr/local/bin/backup.sh > /var/log/backup.log 2> /var/log/backup-error.log

# 実務例: ノイズの多いコマンドからエラーのみ確認
$ find / -name "*.conf" 2>/dev/null          # Permission denied を非表示
$ docker build . 2>&1 | tee build.log        # 画面表示 + ログ保存
```

### 3.3 入力リダイレクト

```bash
# stdin をファイルから読み込み
$ wc -l < /etc/passwd                # ファイルの行数をカウント
$ sort < unsorted.txt > sorted.txt   # ソートして別ファイルに

# ヒアドキュメント（複数行テキストをstdinとして渡す）
$ cat <<EOF
Hello, $(whoami)!
Today is $(date +%Y-%m-%d).
Current directory: $(pwd)
EOF

# ヒアドキュメントで変数展開を抑制
$ cat <<'EOF'
変数は展開されない: $HOME
コマンド置換も無効: $(whoami)
EOF

# ヒアドキュメントでファイル作成（実務で頻出）
$ cat <<'EOF' > /etc/nginx/conf.d/app.conf
server {
    listen 80;
    server_name app.example.com;
    location / {
        proxy_pass http://localhost:3000;
    }
}
EOF

# ヒアストリング（1行のテキストをstdinとして渡す）
$ wc -w <<< "Hello World"           # 単語数カウント → 2
$ grep "pattern" <<< "$variable"     # 変数の内容を検索

# 実務例: SQLクエリの実行
$ mysql -u root -p database <<EOF
SELECT COUNT(*) FROM users WHERE created_at > '2026-01-01';
SELECT name, email FROM users ORDER BY created_at DESC LIMIT 10;
EOF

# 実務例: SSH先でのコマンド実行
$ ssh webserver <<'EOF'
cd /var/log
tail -100 nginx/error.log
df -h
free -h
EOF
```

### 3.4 パイプ

```bash
# パイプの基本: | で左のstdoutを右のstdinに接続
$ ls -la | less                      # 長い出力をページャで表示

# パイプラインの構築（段階的に処理）
$ cat access.log | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -10
#                  │                 │       │           │        │
#                  IPアドレス抽出    ソート  重複カウント 降順ソート 上位10

# 実務例: ログ解析パイプライン
# Apacheアクセスログから404エラーのURLトップ10
$ grep " 404 " access.log | awk '{print $7}' | sort | uniq -c | sort -rn | head -10

# 実務例: プロセス監視
$ ps aux | grep "[n]ginx" | awk '{print $2, $11}'

# 実務例: ディスク使用量の大きいディレクトリ
$ du -sh /var/* 2>/dev/null | sort -rh | head -10

# tee コマンド: 出力を画面とファイルの両方に
$ ls -la | tee filelist.txt              # 画面にも表示、ファイルにも保存
$ make 2>&1 | tee build.log             # ビルドログの保存
$ command | tee -a logfile.txt           # 追記モード

# 名前付きパイプ（FIFO）
$ mkfifo /tmp/mypipe
# ターミナル1:
$ cat /tmp/mypipe
# ターミナル2:
$ echo "Hello via pipe" > /tmp/mypipe

# xargs: パイプからの入力を引数として渡す
$ find . -name "*.log" | xargs rm        # 見つかったファイルを削除
$ find . -name "*.log" -print0 | xargs -0 rm   # スペース入りファイル名対応
$ cat urls.txt | xargs -n1 curl -O       # URLリストからファイルをダウンロード
$ cat servers.txt | xargs -I{} ssh {} "uptime"  # 複数サーバーで実行
```

### 3.5 コマンド置換

```bash
# $() 形式（推奨）
$ echo "Today is $(date +%Y-%m-%d)"
Today is 2026-02-15

$ echo "There are $(ls | wc -l) files here"
There are 42 files here

# バッククォート形式（レガシー、非推奨）
$ echo "Today is `date`"       # ネストが困難なため非推奨

# ネストした例
$ echo "Kernel: $(uname -r), Uptime: $(uptime | awk '{print $3}')"

# 変数に格納
$ current_branch=$(git branch --show-current)
$ file_count=$(find . -name "*.py" | wc -l)
$ latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "no tags")

# 実務例: 日付入りバックアップ
$ tar czf "backup-$(date +%Y%m%d-%H%M%S).tar.gz" /var/www/html

# 実務例: 動的なファイル名
$ mv report.csv "report-$(hostname)-$(date +%Y%m%d).csv"

# プロセス置換（bash, zsh）
# <() で一時ファイルのように扱う
$ diff <(sort file1.txt) <(sort file2.txt)    # ソート済みで比較
$ comm <(sort list1.txt) <(sort list2.txt)     # 共通・差分を表示

# 実務例: 2つのサーバーの設定比較
$ diff <(ssh server1 "cat /etc/nginx/nginx.conf") \
       <(ssh server2 "cat /etc/nginx/nginx.conf")

# 実務例: 複数ソースのマージ
$ sort -m <(sort file1.txt) <(sort file2.txt) <(sort file3.txt) > merged.txt
```

---

## 4. キーボードショートカット

### 4.1 Readline ライブラリ

```
シェルのキーボードショートカットは GNU Readline ライブラリが提供
bash と zsh で共通のものが多い（zsh は ZLE: Zsh Line Editor）

Readline の設定: ~/.inputrc
Zsh のキーバインド確認: bindkey

2つのモード:
  Emacs モード（デフォルト）: Ctrl/Alt キーベース
  Vi モード: vi のモーダル操作

モードの確認・切替:
  $ set -o                   # 現在のオプション一覧
  $ set -o emacs             # Emacs モードに設定
  $ set -o vi                # Vi モードに設定
```

### 4.2 カーソル移動

```
Emacs モード（デフォルト）のカーソル移動:

  ┌───────────────┬────────────────────────────────────┐
  │ ショートカット│ 動作                               │
  ├───────────────┼────────────────────────────────────┤
  │ Ctrl + A      │ 行頭へ移動                         │
  │ Ctrl + E      │ 行末へ移動                         │
  │ Ctrl + F      │ 1文字進む（→キーと同じ）           │
  │ Ctrl + B      │ 1文字戻る（←キーと同じ）           │
  │ Alt + F       │ 1単語進む                          │
  │ Alt + B       │ 1単語戻る                          │
  │ Ctrl + XX     │ 行頭とカーソル位置を交互に移動     │
  └───────────────┴────────────────────────────────────┘

  実用テクニック:
  長いコマンドを入力中に先頭のコマンド名を修正したい時:
  → Ctrl+A で行頭に移動 → 修正 → Ctrl+E で行末に戻る

  パス名の途中を修正したい時:
  → Alt+B で単語単位で戻る方が効率的
```

### 4.3 テキスト編集

```
編集ショートカット:

  ┌───────────────┬────────────────────────────────────────┐
  │ ショートカット│ 動作                                   │
  ├───────────────┼────────────────────────────────────────┤
  │ Ctrl + U      │ カーソルから行頭まで削除（カットリング）│
  │ Ctrl + K      │ カーソルから行末まで削除               │
  │ Ctrl + W      │ カーソル前の1単語を削除                │
  │ Alt + D       │ カーソル後の1単語を削除                │
  │ Ctrl + D      │ カーソル位置の1文字を削除（DELキー）   │
  │ Ctrl + H      │ カーソル前の1文字を削除（BSキー）      │
  │ Ctrl + Y      │ 最後にカットしたテキストを貼り付け     │
  │ Alt + Y       │ カットリングの前のアイテムを貼り付け   │
  │ Ctrl + T      │ カーソル前後の2文字を入れ替え          │
  │ Alt + T       │ カーソル前後の2単語を入れ替え          │
  │ Alt + U       │ カーソル位置から単語末まで大文字に     │
  │ Alt + L       │ カーソル位置から単語末まで小文字に     │
  │ Alt + C       │ カーソル位置の文字を大文字に           │
  │ Ctrl + _      │ Undo（直前の編集を取り消し）           │
  └───────────────┴────────────────────────────────────────┘

  カットリングの概念:
  Ctrl+U, Ctrl+K, Ctrl+W 等で削除したテキストは
  「カットリング」に保存される
  Ctrl+Y で最新のカット内容をペースト
  Alt+Y でカットリング内を順にサイクル

  実用パターン:
  # 長いコマンドの一部を別のコマンドに使い回す
  $ scp user@server:/path/to/long/filename.tar.gz .
  # ↑ Ctrl+W でファイル名を削除、新しいコマンドを打ってCtrl+Y で貼り付け
```

### 4.4 履歴操作

```
履歴ショートカット:

  ┌────────────────┬─────────────────────────────────────────┐
  │ ショートカット │ 動作                                    │
  ├────────────────┼─────────────────────────────────────────┤
  │ Ctrl + R       │ 履歴を逆方向検索（インクリメンタル）    │
  │ Ctrl + S       │ 履歴を順方向検索（stty -ixon が必要）   │
  │ Ctrl + P       │ 前のコマンド（↑キーと同じ）            │
  │ Ctrl + N       │ 次のコマンド（↓キーと同じ）            │
  │ Alt + .        │ 前のコマンドの最後の引数を挿入          │
  │ Ctrl + G       │ 検索をキャンセル                        │
  └────────────────┴─────────────────────────────────────────┘

  履歴展開（イベントデジグネータ）:
  !!               直前のコマンド全体を再実行
  !$               直前のコマンドの最後の引数
  !^               直前のコマンドの最初の引数
  !*               直前のコマンドの全引数
  !n               履歴番号nのコマンド
  !-n              n個前のコマンド
  !string          stringで始まる最新のコマンド
  !?string         stringを含む最新のコマンド
  ^old^new         直前のコマンドのoldをnewに置換して実行

  実務でよく使うパターン:
  $ cat /etc/nginx/nginx.conf       # ファイルを確認
  $ sudo !!                          # 権限不足だったので sudo で再実行
  → sudo cat /etc/nginx/nginx.conf

  $ mkdir /var/www/newsite
  $ cd !$                            # 作成したディレクトリに移動
  → cd /var/www/newsite

  $ vim /etc/nginx/sites-available/default
  $ cp !$ !$:r.bak                   # 同じファイルを .bak でコピー
  → cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.bak

  $ ls file1.txt file2.txt file3.txt
  $ chmod 644 !*                     # 全引数を再利用
  → chmod 644 file1.txt file2.txt file3.txt

  履歴の設定（.bashrc）:
  export HISTSIZE=100000              # メモリ上の履歴数
  export HISTFILESIZE=200000          # ファイルの履歴数
  export HISTCONTROL=ignoreboth       # 重複と空白開始を無視
  export HISTTIMEFORMAT="%F %T "      # タイムスタンプ付き
  shopt -s histappend                 # 上書きでなく追記

  履歴コマンド:
  $ history                           # 全履歴表示
  $ history 20                        # 直近20件
  $ history | grep "docker"           # docker関連のコマンドを検索
  $ fc -l -20                         # 直近20件（番号付き）
  $ fc -e vim 100 110                 # 履歴100-110をvimで編集・実行
```

### 4.5 プロセス制御

```
プロセス制御ショートカット:

  ┌───────────────┬──────────────────────────────────────────┐
  │ ショートカット│ 動作                                     │
  ├───────────────┼──────────────────────────────────────────┤
  │ Ctrl + C      │ SIGINT送信（フォアグラウンドプロセス中断）│
  │ Ctrl + Z      │ SIGTSTP送信（一時停止 → bg/fg で制御）   │
  │ Ctrl + D      │ EOF送信（入力終了 / シェル終了）          │
  │ Ctrl + \      │ SIGQUIT送信（コアダンプ付き強制終了）     │
  │ Ctrl + S      │ 画面出力を一時停止（XOFF）               │
  │ Ctrl + Q      │ 画面出力を再開（XON）                    │
  └───────────────┴──────────────────────────────────────────┘

  使い分けの指針:
  Ctrl+C: 通常の中断。ほとんどの場合これで十分
  Ctrl+Z: 一時停止して後でbg/fgで再開したい時
  Ctrl+\: Ctrl+Cが効かない場合の最終手段
  Ctrl+D: catなどの入力待ちを終了する時

  実務パターン:
  # 長時間コマンドをバックグラウンドに回す
  $ python train_model.py
  # (Ctrl+Z で一時停止)
  [1]+  Stopped     python train_model.py
  $ bg                                # バックグラウンドで再開
  $ disown                            # シェル終了後も継続

  # 画面がフリーズした場合（Ctrl+Sを誤って押した時）
  # → Ctrl+Q で復帰
```

### 4.6 その他の便利なショートカット

```
画面制御:
  Ctrl + L          画面クリア（clear コマンドと同じ）

Tab補完:
  Tab               コマンド/ファイル名の補完
  Tab Tab           候補一覧を表示
  Alt + ?           補完候補の一覧表示

zsh 固有の便利機能:
  Tab               メニュー補完（候補をTabで順に選択）
  Ctrl + X Ctrl + E コマンドラインをエディタで編集
                    （$EDITOR で開き、保存して閉じると実行）

便利なバインド設定例（.zshrc）:
  # Ctrl+X Ctrl+E でコマンドラインをエディタで編集
  autoload -z edit-command-line
  zle -N edit-command-line
  bindkey "^X^E" edit-command-line

  # Alt+. の代替（macOS Terminal で Alt が使えない場合）
  bindkey '\e.' insert-last-word
```

---

## 5. 環境変数とシェル変数

### 5.1 変数の基本

```bash
# シェル変数: 現在のシェルプロセスのみで有効
$ myvar="Hello"
$ echo $myvar
Hello

# 環境変数: 子プロセスにも引き継がれる
$ export MY_ENV_VAR="World"
$ bash -c 'echo $MY_ENV_VAR'   # 子プロセスからアクセス可能
World

# 変数の定義と参照
$ name="gaku"
$ echo "Hello, $name"           # ダブルクォートの中で展開される
Hello, gaku
$ echo 'Hello, $name'           # シングルクォートの中では展開されない
Hello, $name

# 変数名の境界を明示
$ echo "File: ${name}_report.txt"
File: gaku_report.txt

# デフォルト値の設定
$ echo ${UNDEFINED_VAR:-"default value"}   # 未定義時にデフォルト値
default value
$ echo ${UNDEFINED_VAR:="default value"}   # 未定義時にデフォルト値を代入
default value
$ echo ${REQUIRED_VAR:?"エラー: 必須変数が未定義"}  # 未定義時にエラー

# 文字列操作
$ path="/home/gaku/documents/report.txt"
$ echo ${path##*/}              # 最長前方一致削除（ファイル名取得）
report.txt
$ echo ${path%/*}               # 最短後方一致削除（ディレクトリ取得）
/home/gaku/documents
$ echo ${path%.txt}.pdf         # 拡張子変更
/home/gaku/documents/report.pdf
$ echo ${path/gaku/user}        # 最初の一致を置換
/home/user/documents/report.txt
$ echo ${path//\//|}            # 全一致を置換
|home|gaku|documents|report.txt
$ echo ${#path}                 # 文字数
38
```

### 5.2 重要な環境変数

```bash
# PATH: コマンドの検索パス（: 区切り）
$ echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

# PATHに追加（.bashrc / .zshrc に記載）
$ export PATH="$HOME/.local/bin:$PATH"     # 先頭に追加（優先）
$ export PATH="$PATH:/opt/tools/bin"       # 末尾に追加

# HOME: ホームディレクトリ
$ echo $HOME
/home/gaku

# USER: ユーザー名
$ echo $USER
gaku

# SHELL: ログインシェル
$ echo $SHELL
/bin/zsh

# EDITOR / VISUAL: デフォルトエディタ
$ export EDITOR=vim
$ export VISUAL=vim

# LANG / LC_*: ロケール設定
$ echo $LANG
ja_JP.UTF-8
$ locale                        # 全ロケール設定を表示

# TERM: ターミナルの種類
$ echo $TERM
xterm-256color

# PS1: プロンプト文字列
# bash
$ export PS1="\u@\h:\w\$ "     # ユーザー@ホスト:ディレクトリ$
# zsh
$ export PROMPT='%n@%m:%~%# '

# 全環境変数を表示
$ env                           # 環境変数のみ
$ set                           # シェル変数 + 環境変数
$ printenv                      # env と同様
$ export -p                     # export されている変数
$ printenv PATH                 # 特定の環境変数

# 環境変数の削除
$ unset MY_VAR                  # 変数を削除
```

### 5.3 特殊変数

```bash
# シェルの特殊変数（スクリプトで頻出）
$ echo $?                # 直前のコマンドの終了ステータス（0=成功）
$ echo $$                # 現在のシェルのPID
$ echo $!                # 最後にバックグラウンドで実行したプロセスのPID
$ echo $-                # 現在のシェルオプションフラグ
$ echo $0                # シェル名またはスクリプト名

# スクリプト内で使用
$ echo $#                # 引数の数
$ echo $@                # 全引数（個別にクォート）
$ echo $*                # 全引数（一括でクォート）
$ echo $1 $2 $3          # 位置パラメータ（1番目、2番目、3番目の引数）

# $@ と $* の違い（重要）
# "$@" → "$1" "$2" "$3" （個別の文字列として展開）
# "$*" → "$1 $2 $3"     （1つの文字列として展開）

# RANDOM: ランダムな整数（0-32767）
$ echo $RANDOM
16384

# SECONDS: シェル起動からの経過秒数
$ echo $SECONDS
3600

# LINENO: 現在の行番号（スクリプト内）
$ echo $LINENO
42

# BASH_VERSION / ZSH_VERSION
$ echo $BASH_VERSION
5.1.16(1)-release
$ echo $ZSH_VERSION
5.9
```

---

## 6. グロブ（ワイルドカード）パターン

### 6.1 基本パターン

```bash
# * : 任意の文字列（0文字以上）
$ ls *.txt                  # .txt で終わるファイル
$ ls test*                  # test で始まるファイル
$ ls *report*               # report を含むファイル

# ? : 任意の1文字
$ ls file?.txt              # file1.txt, fileA.txt 等
$ ls ???.md                 # 3文字のファイル名.md

# [] : 文字クラス（括弧内のいずれか1文字）
$ ls file[123].txt          # file1.txt, file2.txt, file3.txt
$ ls file[a-z].txt          # filea.txt, fileb.txt, ...
$ ls file[!0-9].txt         # 数字以外の1文字（否定）
$ ls file[^0-9].txt         # 同上（bash）

# {} : ブレース展開（グロブではなくシェルの展開機能）
$ echo file{1,2,3}.txt      # file1.txt file2.txt file3.txt
$ echo {a..z}               # a b c ... z
$ echo {1..10}              # 1 2 3 ... 10
$ echo {01..10}             # 01 02 03 ... 10（ゼロパディング）
$ echo {a..z..2}            # a c e g ... （ステップ2）
$ mkdir -p project/{src,test,docs}/{main,utils}  # ディレクトリツリー作成

# 組み合わせ例
$ cp *.{jpg,png,gif} /backup/images/
$ mv report-{2024,2025,2026}-*.csv archive/
```

### 6.2 拡張グロブ（bash: shopt -s extglob / zsh: デフォルト有効）

```bash
# bash で拡張グロブを有効化
$ shopt -s extglob

# ?(pattern) : 0回または1回の一致
$ ls file?(s).txt            # file.txt, files.txt

# *(pattern) : 0回以上の一致
$ ls file*(s).txt            # file.txt, files.txt, filess.txt, ...

# +(pattern) : 1回以上の一致
$ ls file+(s).txt            # files.txt, filess.txt, ...

# @(pattern1|pattern2) : いずれか1つに一致
$ ls *.@(jpg|png|gif)        # .jpg, .png, .gif のいずれか
$ rm !(important)*.log       # important以外の.logファイルを削除

# !(pattern) : パターンに一致しない
$ ls !(*.txt)                # .txt 以外のファイル
$ rm !(README.md|LICENSE)    # README.md と LICENSE 以外を削除

# zsh の拡張グロブ（setopt EXTENDED_GLOB）
$ ls **/*.py                 # 再帰的にPythonファイルを検索
$ ls **/*(.)                 # 通常ファイルのみ（再帰的）
$ ls **/*(/)                 # ディレクトリのみ（再帰的）
$ ls *(om[1,10])             # 更新日時の新しい順で10件
$ ls *(Lk+100)               # 100KB以上のファイル
```

### 6.3 dotglob とnullglob

```bash
# デフォルトでは * は隠しファイル（.で始まる）にマッチしない
$ ls *                       # .bashrc 等は表示されない

# bash: dotglob を有効化
$ shopt -s dotglob
$ ls *                       # 隠しファイルも含む

# zsh: GLOB_DOTS を有効化
$ setopt GLOB_DOTS

# nullglob: マッチしない場合に空文字列にする
# bash
$ shopt -s nullglob
$ files=(*.xyz)              # マッチなしの場合、配列は空
$ echo ${#files[@]}          # 0

# zsh（デフォルトでエラー。NULL_GLOB で空に）
$ setopt NULL_GLOB

# failglob（bash）: マッチしない場合にエラー
$ shopt -s failglob
$ ls *.xyz                   # bash: no match: *.xyz
```

---

## 7. ターミナルエミュレータの選定と設定

### 7.1 主要ターミナルエミュレータの比較

```
詳細比較表:

┌────────────┬────────┬──────────┬───────────┬────────────────────────┐
│ 名前       │ OS     │ 描画方式 │ 設定形式  │ 主な特徴               │
├────────────┼────────┼──────────┼───────────┼────────────────────────┤
│ iTerm2     │ macOS  │ Metal    │ GUI/JSON  │ 分割/プロファイル/検索 │
│ Alacritty  │ 全OS   │ OpenGL   │ TOML      │ 超高速/軽量/シンプル   │
│ kitty      │ Lin/mac│ OpenGL   │ conf      │ 画像表示/リガチャ      │
│ WezTerm    │ 全OS   │ OpenGL   │ Lua       │ Lua設定/マルチプレクサ │
│ Warp       │ mac/Lin│ GPU      │ YAML      │ AI統合/ブロック編集    │
│ Win Term   │ Win    │ DirectX  │ JSON      │ MS公式/タブ/分割       │
│ GNOME Term │ Linux  │ VTE      │ dconf     │ 安定/GNOME統合         │
│ Konsole    │ Linux  │ Qt       │ rc file   │ KDE統合/高機能         │
│ Terminator │ Linux  │ VTE      │ conf      │ 画面分割特化           │
│ foot       │ Linux  │ Wayland  │ ini       │ Wayland専用/超軽量     │
│ Hyper      │ 全OS   │ Electron │ JS        │ Web技術で拡張可能      │
│ Rio        │ 全OS   │ WGPU     │ TOML      │ Rust製/WebGPU描画      │
└────────────┴────────┴──────────┴───────────┴────────────────────────┘

選定の指針:
  開発者（macOS）     → iTerm2 or Warp or Alacritty
  開発者（Linux）     → Alacritty or kitty or WezTerm
  Windows             → Windows Terminal + WSL2
  軽量・高速重視      → Alacritty or foot
  カスタマイズ重視    → WezTerm（Lua） or kitty
  初心者              → Warp（AI支援付き）
  サーバー管理者      → tmux/screen + 好みのターミナル
```

### 7.2 iTerm2 の設定（macOS）

```
iTerm2 の主要設定:

  プロファイル設定:
  Preferences > Profiles > General
  - Working Directory: Reuse previous session's directory

  Preferences > Profiles > Colors
  - Color Presets: Solarized Dark / Dracula / Tokyo Night 等

  Preferences > Profiles > Text
  - Font: JetBrains Mono Nerd Font / Hack Nerd Font
  - Font Size: 14
  - Use ligatures: ✓

  キーマッピング:
  Preferences > Keys > Key Bindings
  - Alt+← → Send Escape Sequence: b（1単語戻る）
  - Alt+→ → Send Escape Sequence: f（1単語進む）
  - Alt+Delete → Send Hex Codes: 0x17（1単語削除）

  便利な機能:
  Cmd+D              垂直分割
  Cmd+Shift+D        水平分割
  Cmd+T              新しいタブ
  Cmd+Enter          フルスクリーン
  Cmd+Shift+H        ペースト履歴
  Cmd+;              オートコンプリート（過去の入力から）
  Cmd+F              検索（正規表現対応）
  Cmd+Opt+B          タイムスタンプ表示

  Shell Integration:
  iTerm2 > Install Shell Integration
  → コマンドの開始/終了マーカー
  → 前のコマンドの出力にジャンプ（Cmd+Shift+↑/↓）
  → 失敗コマンドのマーカー
```

### 7.3 Alacritty の設定

```toml
# ~/.config/alacritty/alacritty.toml

[window]
padding = { x = 8, y = 8 }
decorations = "Full"
opacity = 0.95
startup_mode = "Windowed"

[scrolling]
history = 10000
multiplier = 3

[font]
normal = { family = "JetBrains Mono Nerd Font", style = "Regular" }
bold = { family = "JetBrains Mono Nerd Font", style = "Bold" }
italic = { family = "JetBrains Mono Nerd Font", style = "Italic" }
size = 14.0

[colors.primary]
background = "#1a1b26"
foreground = "#c0caf5"

[colors.normal]
black   = "#15161e"
red     = "#f7768e"
green   = "#9ece6a"
yellow  = "#e0af68"
blue    = "#7aa2f7"
magenta = "#bb9af7"
cyan    = "#7dcfff"
white   = "#a9b1d6"

[keyboard]
bindings = [
  { key = "V", mods = "Control|Shift", action = "Paste" },
  { key = "C", mods = "Control|Shift", action = "Copy" },
  { key = "N", mods = "Control|Shift", action = "SpawnNewInstance" },
]
```

### 7.4 Windows Terminal の設定

```json
// Windows Terminal settings.json の主要設定
{
    "defaultProfile": "{GUID-of-WSL}",
    "copyOnSelect": true,
    "copyFormatting": "none",
    "profiles": {
        "defaults": {
            "font": {
                "face": "CaskaydiaCove Nerd Font",
                "size": 12
            },
            "colorScheme": "One Half Dark",
            "opacity": 90,
            "useAcrylic": true,
            "padding": "8"
        },
        "list": [
            {
                "name": "Ubuntu",
                "source": "Windows.Terminal.Wsl",
                "startingDirectory": "//wsl$/Ubuntu/home/gaku"
            }
        ]
    },
    "actions": [
        { "command": "splitPane", "keys": "alt+shift+d", "splitMode": "auto" },
        { "command": { "action": "splitPane", "split": "horizontal" }, "keys": "alt+shift+-" },
        { "command": { "action": "splitPane", "split": "vertical" }, "keys": "alt+shift+|" }
    ]
}
```

---

## 8. シェルの内部動作 — コマンド実行の仕組み

### 8.1 コマンド解析のステップ

```
シェルがコマンドラインを処理する順序:

  1. トークン分割（Tokenization）
     入力行をワード（トークン）に分割
     → "ls -la /home" → ["ls", "-la", "/home"]

  2. コマンド識別
     最初のトークンがコマンドかどうかを判定
     → エイリアス展開 → 関数 → ビルトイン → 外部コマンド

  3. 展開（Expansion）
     以下の順序で展開を実行:
     a. ブレース展開:    {a,b,c} → a b c
     b. チルダ展開:      ~ → /home/gaku
     c. パラメータ展開:  $HOME → /home/gaku
     d. コマンド置換:    $(date) → 2026-02-15
     e. 算術展開:        $((1+2)) → 3
     f. プロセス置換:    <(sort file) → /dev/fd/63
     g. 単語分割:        展開結果をIFS（デフォルト: スペース/タブ/改行）で分割
     h. パス名展開:      *.txt → file1.txt file2.txt
     i. クォート除去:    残ったクォートを除去

  4. リダイレクト設定
     >, <, |, 2>&1 等のリダイレクトをセットアップ

  5. コマンド実行
     fork() → exec() で子プロセスとしてコマンドを実行
     ビルトインコマンドの場合は fork() なしで直接実行

展開の確認方法:
  $ set -x                  # デバッグモード（展開結果を表示）
  $ ls *.txt
  + ls file1.txt file2.txt  # 展開後の実際のコマンド
  $ set +x                  # デバッグモード解除
```

### 8.2 fork-exec モデル

```
外部コマンドの実行フロー:

  親プロセス（シェル）          子プロセス
  ┌──────────────┐
  │ $ ls -la     │
  │              │
  │ fork() ──────┼──────→ ┌──────────────┐
  │              │         │ シェルの複製  │
  │ wait()       │         │              │
  │  :           │         │ exec("ls")   │
  │  :           │         │ → ls に変身  │
  │  :           │         │              │
  │  :           │         │ 実行・出力   │
  │  :           │         │              │
  │  :           │←─────── │ exit(0)      │
  │              │  status └──────────────┘
  │ $? = 0       │
  └──────────────┘

  ビルトインコマンド（cd, echo, export等）:
  → fork() せずにシェル自身が直接実行
  → cd がビルトインである理由: 親プロセスのカレントディレクトリを変更する必要がある

  サブシェル:
  $ (cd /tmp && ls)         # サブシェルで実行
  # 元のシェルのカレントディレクトリは変わらない

  $ var=hello
  $ echo "$var" | read response   # パイプの右辺はサブシェル（bash）
  $ echo "$response"              # 空になる（bashの場合）
  # zshではlastpipeが有効で、最後のコマンドが現在のシェルで実行される
```

### 8.3 終了ステータス

```bash
# 終了ステータスの規約
# 0:     成功
# 1:     一般的なエラー
# 2:     シェルビルトインの不正使用
# 126:   コマンドは存在するが実行権限がない
# 127:   コマンドが見つからない
# 128+N: シグナルNで終了（例: 128+9=137 → SIGKILL）
# 130:   Ctrl+C (SIGINT) で終了
# 255:   終了ステータスが範囲外

# 実務での活用
$ grep -q "pattern" file
$ echo $?                    # 0: パターンが見つかった, 1: 見つからない

# 条件分岐
$ if grep -q "error" /var/log/syslog; then
    echo "エラーが見つかりました"
    grep "error" /var/log/syslog | tail -5
  fi

# テストコマンド
$ test -f /etc/passwd && echo "ファイルが存在します"
$ [ -d /var/log ] && echo "ディレクトリが存在します"
$ [[ -n "$variable" ]] && echo "変数は空ではありません"

# 複合条件
$ [[ -f config.yml && -r config.yml ]] && echo "設定ファイルは読み取り可能"
$ [[ "$status" == "running" || "$status" == "started" ]] && echo "動作中"
```

---

## 9. クォートの使い分け

### 9.1 シングルクォート・ダブルクォート・バッククォート

```bash
# シングルクォート: 全ての展開を抑制（リテラル文字列）
$ echo 'Hello, $USER! $(date)'
Hello, $USER! $(date)

# ダブルクォート: 変数展開とコマンド置換は実行、ワード分割とグロブは抑制
$ echo "Hello, $USER! $(date +%Y)"
Hello, gaku! 2026

# ダブルクォート内で抑制されるもの:
# - ワード分割（スペースによるトークン分割）
# - パス名展開（* ? [] のグロブ）
# ダブルクォート内で展開されるもの:
# - 変数展開 $var ${var}
# - コマンド置換 $(command)
# - 算術展開 $((expression))
# - エスケープシーケンス（一部）

# バッククォート（非推奨、$() を使用すべき）
$ echo "Today is `date`"     # ネストが困難

# クォートなし: 全展開 + ワード分割 + グロブ展開
$ echo Hello, $USER! *.txt
Hello, gaku! file1.txt file2.txt

# エスケープ文字
$ echo "She said \"Hello\""   # ダブルクォート内でダブルクォート
$ echo 'It'\''s a test'       # シングルクォート内でシングルクォート
$ echo "Path: \$HOME"         # $ をリテラルとして
$ echo "Backslash: \\"        # バックスラッシュ自体

# $'...' 形式（ANSI-C Quoting）
$ echo $'Line1\nLine2\tTabbed'
Line1
Line2	Tabbed

# 実務でのクォート選択指針:
# リテラル文字列         → シングルクォート
# 変数展開が必要         → ダブルクォート
# エスケープシーケンス   → $'...'
# 何も囲まない           → 原則避ける（意図しない展開の危険）

# よくあるミス
$ file="my document.txt"
$ cat $file                  # 「my」と「document.txt」に分割される！
$ cat "$file"                # 正しい: 1つの引数として渡される

# find での注意点
$ find . -name *.log         # シェルが先に *.log を展開してしまう
$ find . -name "*.log"       # 正しい: find にパターンを渡す
$ find . -name '*.log'       # これも正しい
```

### 9.2 ヒアドキュメントとヒアストリングの詳細

```bash
# ヒアドキュメントのバリエーション

# 1. 通常（変数展開あり）
$ cat <<EOF
User: $USER
Home: $HOME
Date: $(date)
EOF

# 2. クォート付きデリミタ（変数展開なし）
$ cat <<'EOF'
User: $USER    ← そのまま表示
Home: $HOME    ← そのまま表示
EOF

# 3. ハイフン付き（先頭のタブを無視）
$ cat <<-EOF
	This line has a leading tab
	It will be stripped
	EOF

# 4. ヒアストリング（1行のみ）
$ grep "pattern" <<< "search in this string"
$ bc <<< "3.14 * 2"

# 実務パターン: 設定ファイルの動的生成
$ cat <<EOF > /tmp/config.ini
[database]
host=${DB_HOST:-localhost}
port=${DB_PORT:-5432}
name=${DB_NAME:-myapp}
user=${DB_USER:-admin}
EOF

# 実務パターン: 複数行のコマンド実行（リモート）
$ ssh production-server <<'REMOTE'
cd /var/www/app
git pull origin main
docker compose down
docker compose up -d
docker compose logs --tail=20
REMOTE

# 実務パターン: テスト用データの投入
$ psql -U postgres testdb <<SQL
INSERT INTO users (name, email) VALUES
  ('Alice', 'alice@example.com'),
  ('Bob', 'bob@example.com'),
  ('Charlie', 'charlie@example.com');
SQL
```

---

## 10. エイリアスとシェル関数

### 10.1 エイリアス

```bash
# エイリアスの定義
$ alias ll='ls -lah'
$ alias la='ls -A'
$ alias ..='cd ..'
$ alias ...='cd ../..'

# エイリアスの確認
$ alias              # 全エイリアスを表示
$ alias ll           # 特定のエイリアスを確認
$ type ll            # ll is aliased to 'ls -lah'

# エイリアスの削除
$ unalias ll

# 実務で便利なエイリアス集
# --- Git ---
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline --graph --all'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'
alias gpl='git pull'
alias gst='git stash'

# --- Docker ---
alias d='docker'
alias dc='docker compose'
alias dps='docker ps'
alias dpsa='docker ps -a'
alias dimg='docker images'
alias dexec='docker exec -it'
alias dlogs='docker logs -f'
alias dprune='docker system prune -af'

# --- Safety ---
alias rm='rm -i'               # 削除前に確認
alias cp='cp -i'               # 上書き前に確認
alias mv='mv -i'               # 上書き前に確認

# --- Navigation ---
alias projects='cd ~/projects'
alias downloads='cd ~/Downloads'

# --- System ---
alias ports='ss -tlnp'         # 使用中のポート
alias meminfo='free -h'        # メモリ情報
alias diskinfo='df -h'         # ディスク情報
alias myip='curl -s ifconfig.me'  # グローバルIP

# --- Misc ---
alias h='history'
alias c='clear'
alias e='$EDITOR'
alias reload='source ~/.zshrc'
alias path='echo $PATH | tr ":" "\n"'
alias now='date +"%Y-%m-%d %H:%M:%S"'
```

### 10.2 シェル関数

```bash
# 関数の定義（エイリアスより柔軟）
# 引数を取れる、複数行の処理が可能

# ディレクトリ作成と移動を同時に
mkcd() {
    mkdir -p "$1" && cd "$1"
}
$ mkcd new-project   # ディレクトリ作成 + cd

# ファイルのバックアップ
backup() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cp "$file" "${file}.bak.$(date +%Y%m%d-%H%M%S)"
        echo "Backed up: ${file}"
    else
        echo "Error: ${file} not found" >&2
        return 1
    fi
}

# ポートを使用しているプロセスを確認
port() {
    lsof -i :"$1" 2>/dev/null || ss -tlnp | grep ":$1"
}
$ port 8080

# Git ブランチの作成と切替
gcb() {
    git checkout -b "$1" && echo "Created and switched to branch: $1"
}

# Docker コンテナに入る
dsh() {
    docker exec -it "$1" /bin/sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'
}

# 特定のディレクトリのファイルサイズ上位
big() {
    local dir="${1:-.}"
    local count="${2:-10}"
    du -ah "$dir" 2>/dev/null | sort -rh | head -"$count"
}
$ big /var/log 20    # /var/log 以下のサイズ上位20件

# JSONの整形表示
jsonpp() {
    if [[ -f "$1" ]]; then
        python3 -m json.tool "$1"
    else
        echo "$1" | python3 -m json.tool
    fi
}

# 天気情報
weather() {
    curl -s "wttr.in/${1:-Tokyo}?format=3"
}

# extract: 圧縮ファイルの自動展開
extract() {
    if [[ -f "$1" ]]; then
        case "$1" in
            *.tar.bz2) tar xjf "$1"    ;;
            *.tar.gz)  tar xzf "$1"    ;;
            *.tar.xz)  tar xJf "$1"    ;;
            *.bz2)     bunzip2 "$1"    ;;
            *.gz)      gunzip "$1"     ;;
            *.tar)     tar xf "$1"     ;;
            *.tbz2)    tar xjf "$1"    ;;
            *.tgz)     tar xzf "$1"    ;;
            *.zip)     unzip "$1"      ;;
            *.Z)       uncompress "$1" ;;
            *.7z)      7z x "$1"       ;;
            *.rar)     unrar x "$1"    ;;
            *.xz)      xz -d "$1"     ;;
            *.zst)     zstd -d "$1"   ;;
            *)         echo "Unknown format: $1" ;;
        esac
    else
        echo "File not found: $1" >&2
        return 1
    fi
}
```

---

## 11. 実践演習

### 演習1: [基礎] — 基本コマンドとリダイレクト

```bash
# 課題: 以下のコマンドを順に実行し、結果を確認する

# 1. システム情報の収集
$ {
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "Host: $(hostname)"
    echo "Kernel: $(uname -r)"
    echo "Shell: $SHELL ($0)"
    echo "Uptime: $(uptime)"
    echo ""
    echo "=== Disk Usage ==="
    df -h /
    echo ""
    echo "=== Memory ==="
    free -h 2>/dev/null || vm_stat 2>/dev/null
} > /tmp/sysinfo.txt

# 2. 結果の確認
$ cat /tmp/sysinfo.txt

# 3. ファイル操作
$ mkdir -p /tmp/exercise/{src,build,docs}
$ touch /tmp/exercise/src/main.{c,h}
$ touch /tmp/exercise/docs/README.md
$ ls -R /tmp/exercise/

# 4. リダイレクトとパイプの組み合わせ
$ echo "Hello, World!" > /tmp/test.txt
$ echo "Second line" >> /tmp/test.txt
$ echo "Third line" >> /tmp/test.txt
$ cat /tmp/test.txt | wc -l                # 行数
$ cat /tmp/test.txt | wc -w                # 単語数
$ cat /tmp/test.txt | tee /tmp/copy.txt    # 画面にも表示、コピーも作成
```

### 演習2: [応用] — パイプラインの構築

```bash
# 課題: 複数のパイプラインを構築する

# 1. /etc/passwd からユーザー情報を抽出
$ cat /etc/passwd | cut -d: -f1,3,7 | sort -t: -k2 -n | tail -10
# ユーザー名:UID:シェル をUID降順で10件

# 2. プロセス情報の整形
$ ps aux | awk 'NR>1 {printf "%-15s %5s %5s %s\n", $1, $2, $3, $11}' | sort -k3 -rn | head -10
# CPU使用率トップ10のプロセス

# 3. ファイルシステムの分析
$ find /tmp -type f -name "*.txt" 2>/dev/null | while read -r file; do
    size=$(wc -c < "$file")
    echo "$size $file"
  done | sort -rn | head -5
# /tmp 以下の .txt ファイルをサイズ順で5件

# 4. コマンド置換の活用
$ echo "Branch: $(git branch --show-current 2>/dev/null || echo 'not a git repo')"
$ echo "Python: $(python3 --version 2>/dev/null || echo 'not installed')"
$ echo "Node: $(node --version 2>/dev/null || echo 'not installed')"
```

### 演習3: [発展] — ショートカットとヒストリの活用

```
以下の操作をキーボードのみで実行する:

1. カーソル移動
   a. 長いコマンドを入力: echo "This is a very long command with many arguments"
   b. Ctrl+A で行頭に移動
   c. Ctrl+E で行末に移動
   d. Alt+B で1単語ずつ戻る
   e. Alt+F で1単語ずつ進む

2. テキスト編集
   a. 長いコマンドを入力
   b. Ctrl+U で行頭まで削除
   c. 新しいコマンドを入力
   d. Ctrl+Y で削除した内容を復元

3. 履歴操作
   a. Ctrl+R で "ssh" を検索
   b. !! で直前のコマンドを再実行
   c. !$ で直前の最後の引数を利用
   d. ^old^new で直前のコマンドの文字列を置換

4. プロセス制御
   a. sleep 60 を実行
   b. Ctrl+Z で一時停止
   c. bg でバックグラウンドに移動
   d. jobs で確認
   e. fg でフォアグラウンドに戻す
   f. Ctrl+C で中断
```

### 演習4: [実務] — トラブルシューティング演習

```bash
# 課題: 実務的なトラブルシューティングシナリオ

# シナリオ1: コマンドが見つからない
$ mycommand
# bash: mycommand: command not found

# 調査手順:
$ which mycommand                    # パスを確認
$ type mycommand                     # コマンドの種類を確認
$ echo $PATH | tr ':' '\n'           # PATHの確認
$ find / -name "mycommand" 2>/dev/null  # ファイルの場所を検索
$ apt list --installed 2>/dev/null | grep mycommand  # パッケージ確認

# シナリオ2: Permission denied
$ cat /var/log/syslog
# cat: /var/log/syslog: Permission denied

# 調査手順:
$ ls -la /var/log/syslog             # パーミッション確認
$ id                                  # 自分のUID/GIDを確認
$ groups                              # 所属グループを確認
$ sudo cat /var/log/syslog           # sudo で再試行

# シナリオ3: ディスクが満杯
$ df -h                               # ディスク使用率確認
$ du -sh /var/* 2>/dev/null | sort -rh | head -10  # 大きなディレクトリ
$ find /var/log -name "*.log" -size +100M          # 大きなログファイル
$ journalctl --disk-usage             # journald のサイズ確認
$ docker system df                     # Dockerのディスク使用量

# シナリオ4: 高負荷調査
$ uptime                              # ロードアベレージ確認
$ top -bn1 | head -20                # CPU/メモリ使用率
$ ps aux --sort=-%cpu | head -10     # CPU使用率トップ
$ ps aux --sort=-%mem | head -10     # メモリ使用率トップ
$ iostat -x 1 3                       # I/O統計
$ vmstat 1 5                          # システム統計
```

---

## 12. ベストプラクティスとTips

### 12.1 安全なコマンド操作

```bash
# 1. 破壊的コマンドの前に確認
$ rm -rf /path/to/dir               # 危険！
$ rm -ri /path/to/dir               # -i で確認しながら削除
$ ls /path/to/dir                    # まず ls で確認

# 2. 変数を使う時は必ずダブルクォートで囲む
$ rm "$file"                         # 正しい
$ rm $file                           # スペース入りのファイル名で問題発生

# 3. rm -rf でワイルドカードを使う時の注意
$ rm -rf /tmp/test/*                 # OK
$ rm -rf /tmp/test/ *                # 危険！ スペースで「/ *」になる
$ rm -rf "$dir"/*                    # OK（変数はクォート）

# 4. sudo の慎重な使用
$ sudo !!                            # 直前のコマンドを sudo で再実行（確認後）
$ sudo -k                            # sudo のキャッシュをクリア

# 5. 実行前にエコーで確認（dry run パターン）
$ for f in *.log; do echo "rm $f"; done    # まず echo で確認
$ for f in *.log; do rm "$f"; done         # 問題なければ実行

# 6. 重要な操作の前にバックアップ
$ cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak.$(date +%Y%m%d)
$ tar czf backup-before-change.tar.gz /path/to/dir
```

### 12.2 効率的なコマンドライン作業

```bash
# 1. Tab 補完を最大限活用
$ cd /e<Tab>/ng<Tab>/si<Tab>        # /etc/nginx/sites-available

# 2. ブレース展開でファイル操作を効率化
$ cp config.{yml,yml.bak}           # cp config.yml config.yml.bak
$ mv file.{txt,md}                  # mv file.txt file.md
$ touch test_{1..5}.txt             # test_1.txt ... test_5.txt

# 3. Ctrl+R の活用（インクリメンタル検索）
# → Ctrl+R を押して "docker" と入力すると
#   最近の docker コマンドが表示される
# → 複数回 Ctrl+R でさらに古い履歴を検索

# 4. Alt+. で前のコマンドの引数を繰り返し使う
$ ls /var/log/nginx/access.log
$ less <Alt+.>                      # less /var/log/nginx/access.log

# 5. xargs で繰り返し処理
$ find . -name "*.tmp" -print0 | xargs -0 rm
$ cat urls.txt | xargs -P4 -I{} curl -sO {}    # 4並列でダウンロード

# 6. watch でコマンドを定期実行
$ watch -n 2 'docker ps'            # 2秒ごとにコンテナ状態を確認
$ watch -d 'df -h'                  # 差分をハイライト表示

# 7. script でセッションを記録
$ script /tmp/session-$(date +%Y%m%d).log
$ # ... 作業 ...
$ exit                               # 記録終了
```

### 12.3 よくある落とし穴

```bash
# 1. ファイル名のスペース問題
$ file="my document.txt"
$ cat $file              # → "my" と "document.txt" に分割される
$ cat "$file"            # → 正しく1つのファイルとして扱われる

# 2. 変数の未定義
$ echo $UNDEFINED        # 空文字列（エラーにならない）
$ set -u                  # 未定義変数の参照をエラーにする（推奨）
$ echo ${var:?"undefined"}  # 未定義時にエラーメッセージ

# 3. パイプとサブシェル（bashの罠）
$ count=0
$ echo -e "a\nb\nc" | while read line; do
    count=$((count + 1))
  done
$ echo $count            # bash: 0（パイプの右辺はサブシェル）
# 解決策1: プロセス置換
$ while read line; do
    count=$((count + 1))
  done < <(echo -e "a\nb\nc")
# 解決策2: lastpipe（bash 4.2+）
$ shopt -s lastpipe

# 4. for ループとIFS
$ for f in $(ls); do echo "$f"; done   # スペース入りファイル名で問題
$ for f in *; do echo "$f"; done       # 正しいイディオム

# 5. test と [[ ]] の違い
$ [ "$a" == "$b" ]       # POSIX sh では = を使う
$ [[ "$a" == "$b" ]]     # bash/zsh では == が使える
$ [[ "hello" =~ ^he ]]   # 正規表現マッチ（[[ ]] のみ）
$ [[ -f file && -r file ]]  # &&/|| が使える（[ ] では -a/-o）

# 6. 算術評価
$ echo $((10 / 3))       # 3（整数演算のみ）
$ echo "scale=2; 10/3" | bc   # 3.33（浮動小数点が必要な場合）
$ awk 'BEGIN {printf "%.2f\n", 10/3}'  # 3.33（awkでも可能）

# 7. コマンド置換と改行
$ files=$(ls)            # 末尾の改行は削除される
$ echo "$files"          # 改行が保持される（ダブルクォート）
$ echo $files            # 改行がスペースに変換される（クォートなし）
```

---

## 13. FAQ（よくある質問）

### Q1: bash と zsh のどちらを使うべき？

macOS ユーザーは zsh（デフォルト）を推奨。Linux サーバーでは bash が確実。zsh は bash のほぼ上位互換であり、以下の追加機能がある:

```
zsh の bash に対する優位点:
  - 強力な Tab 補完（コマンドオプション、引数の補完）
  - 右プロンプト（RPROMPT）
  - 再帰グロブ（**/*.txt）がデフォルトで使用可能
  - スペル修正提案
  - グロブ修飾子（*(om) で更新日時順等）
  - 配列のインデックスが1始まり（直感的）
  - Oh My Zsh / Prezto 等のフレームワーク
  - 浮動小数点演算のサポート

bash の強み:
  - ほぼ全てのLinuxディストリビューションに標準搭載
  - POSIX sh に近い動作
  - サーバー環境での互換性が高い
  - 情報量（ドキュメント、Stack Overflow等）が多い

推奨:
  対話シェル       → zsh + Oh My Zsh
  シェルスクリプト → #!/usr/bin/env bash（bash 4+）
  移植性重視       → #!/bin/sh（POSIX sh互換）
```

### Q2: なぜCLIを学ぶべきか？GUIで十分では？

```
CLIの優位性:

  1. 自動化
     スクリプトで繰り返し作業を自動化できる
     cron/systemd timer で定期実行
     CI/CDパイプラインの構成要素

  2. リモート操作
     SSH経由でサーバーを管理（GUIは不要）
     帯域が限られた環境でも操作可能
     複数サーバーの一括操作

  3. 効率性
     マウス操作より速い（慣れれば）
     パイプラインで複雑な処理を簡潔に記述
     バッチ処理で大量のファイルを一括操作

  4. 再現性
     コマンド履歴が残る
     手順をスクリプト化して共有可能
     ドキュメント化が容易

  5. サーバー環境
     多くのサーバーにはGUIがない
     コンテナ環境（Docker）はCLIが基本
     クラウドインスタンスはSSH接続が標準

  6. デバッグ
     ログの検索・分析はCLIが圧倒的に高速
     プロセスの監視・制御
     ネットワークの診断

  7. 組み合わせの力
     Unix哲学: 「一つのことをうまくやるプログラム」の組み合わせ
     パイプで既存コマンドを自由に連結
     新しいツールもすぐにパイプラインに組み込める
```

### Q3: ターミナルのフォント設定はどうすべき？

```
プログラミング用フォントの推奨:

  Nerd Font系（アイコン付き）:
  - JetBrains Mono Nerd Font     ← 開発者に最も人気
  - Hack Nerd Font               ← 読みやすさ重視
  - FiraCode Nerd Font           ← リガチャ（合字）対応
  - CaskaydiaCove Nerd Font      ← Microsoft製
  - MesloLGS NF                  ← Oh My Zsh の Powerlevel10k 推奨

  日本語対応:
  - HackGen (白源)               ← Hack + 源ノ角ゴシック
  - PlemolJP                     ← IBM Plex Mono + IBM Plex Sans JP
  - UDEV Gothic                  ← Monaspace + BIZ UDGothic
  - Cica                         ← Hack + Miguフォント

  設定のポイント:
  - フォントサイズ: 12-16pt（画面サイズに応じて）
  - 行間: 1.2-1.5
  - リガチャ: 有効化すると != → ≠, => → ⇒ 等が見やすい
  - Nerd Font: Starship, Powerlevel10k 等のプロンプトに必要
```

### Q4: シェルの起動ファイルの読み込み順序は？

```
bash の起動ファイル:

  ログインシェル:
  1. /etc/profile
  2. ~/.bash_profile (存在しなければ ~/.bash_login → ~/.profile)

  非ログインシェル（ターミナルエミュレータ起動時）:
  1. ~/.bashrc

  推奨: ~/.bash_profile に以下を記述
  if [ -f ~/.bashrc ]; then source ~/.bashrc; fi

zsh の起動ファイル:

  全ユーザー共通 → 個人設定の順で、以下のファイルが読まれる:

  ┌──────────────┬─────────┬────────────┬──────────┐
  │ ファイル     │ ログイン│ 対話的     │ スクリプト│
  ├──────────────┼─────────┼────────────┼──────────┤
  │ .zshenv      │ ✓       │ ✓          │ ✓        │
  │ .zprofile    │ ✓       │            │          │
  │ .zshrc       │ ✓       │ ✓          │          │
  │ .zlogin      │ ✓       │            │          │
  │ .zlogout     │ ✓（終了）│           │          │
  └──────────────┴─────────┴────────────┴──────────┘

  推奨設定場所:
  .zshenv   → PATH 等の環境変数（全場面で必要）
  .zshrc    → エイリアス、関数、プロンプト、補完設定
  .zprofile → ログイン時のみ必要な設定
```

### Q5: WSL2 と ネイティブ Linux の違いは？

```
WSL2 (Windows Subsystem for Linux 2):
  Windows上で本物のLinuxカーネルを仮想マシンで動作

  利点:
  - Windows と Linux を同時に使える
  - ファイルシステムの相互アクセス（/mnt/c/）
  - Docker Desktop との統合
  - VS Code Remote - WSL で透過的な開発

  制限:
  - I/O パフォーマンス（Windows ↔ Linux 間のファイル操作）
  - systemd の制限（WSL2 では設定で有効化可能）
  - ネットワーク設定の複雑さ
  - USB デバイスのアクセス制限

  推奨設定:
  - プロジェクトファイルは Linux ファイルシステム側に置く
  - /mnt/c/ は避ける（遅い）
  - Windows Terminal + WSL2 の組み合わせ
  - wsl.conf で automount とnetwork を適切に設定
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ターミナル | テキストI/Oのウィンドウ。iTerm2, Alacritty, Windows Terminal等 |
| シェル | コマンドの解釈・実行。bash（Linux標準）/ zsh（macOS標準） |
| コマンド構造 | `command [options] [arguments]`。短い/長いオプション |
| リダイレクト | `>` `>>` `2>` `<` `&>` でI/Oの入出力先を変更 |
| パイプ | `\|` でコマンドを連結。Unix哲学の核心 |
| ショートカット | Ctrl+R(検索), Ctrl+A/E(移動), Ctrl+C(中断), Ctrl+Z(停止) |
| 環境変数 | PATH, HOME, SHELL等。`export` で子プロセスに継承 |
| グロブ | `*` `?` `[]` `{}` でファイルパターンマッチ |
| クォート | シングル（リテラル）、ダブル（展開あり）の使い分け |
| fork-exec | シェルのコマンド実行モデル。子プロセスでexec |
| エイリアス/関数 | よく使うコマンドの省略形。関数はより柔軟 |

---

## 次に読むべきガイド
- [[01-shell-config.md]] — シェル設定（.bashrc / .zshrc の詳細設定）
- [[02-man-and-help.md]] — マニュアルとヘルプの活用

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, No Starch Press, 2019.
2. Robbins, A. & Beebe, N. "Classic Shell Scripting." O'Reilly Media, 2005.
3. Ramey, C. & Fox, B. "Bash Reference Manual." GNU Project, 2022.
4. Janssens, J. "Data Science at the Command Line." 2nd Ed, O'Reilly Media, 2021.
5. GNU Coreutils Manual. https://www.gnu.org/software/coreutils/manual/
6. Zsh Documentation. https://zsh.sourceforge.io/Doc/
7. The Open Group. "POSIX.1-2017 Shell & Utilities." IEEE Std 1003.1-2017.
8. Cooper, M. "Advanced Bash-Scripting Guide." The Linux Documentation Project.
