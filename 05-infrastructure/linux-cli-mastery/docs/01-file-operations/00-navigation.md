# ディレクトリ移動と一覧

> ファイルシステムのナビゲーションは CLI の最も基本的なスキル。ここを確実にマスターすることが、全てのコマンド操作の土台となる。

## この章で学ぶこと

- [ ] ディレクトリの移動と一覧表示を使いこなせる
- [ ] 絶対パスと相対パスの違いを理解する
- [ ] ls の主要オプションを完全に把握する
- [ ] ディレクトリスタック（pushd/popd）を活用できる
- [ ] zoxide や fzf を使った高速ナビゲーションを実践できる
- [ ] ファイルシステムの構造を理解する
- [ ] ディスク使用量の確認と分析ができる
- [ ] モダンな代替ツール（eza, tree, ncdu）を使いこなせる

---

## 1. ディレクトリ操作の基本

### 1.1 cd（Change Directory）

```bash
# ============================================
# cd の基本操作
# ============================================

pwd                          # 現在のディレクトリを表示（Print Working Directory）
cd /path/to/dir              # 絶対パスで移動
cd relative/path             # 相対パスで移動
cd ~                         # ホームディレクトリに移動（cd だけでも同じ）
cd                           # ホームディレクトリに移動（引数なし）
cd -                         # 前のディレクトリに戻る（トグル動作）
cd ..                        # 1つ上のディレクトリ（親ディレクトリ）
cd ../..                     # 2つ上のディレクトリ
cd ../../..                  # 3つ上のディレクトリ
cd ~username                 # 指定ユーザーのホームディレクトリ

# 実践的な使い方
cd ~/projects/myapp          # プロジェクトディレクトリに移動
cd /var/log                  # ログディレクトリに移動
cd /etc                      # 設定ファイルディレクトリに移動
cd /tmp                      # 一時ファイルディレクトリに移動
cd /usr/local/bin            # ローカルバイナリディレクトリに移動

# CDPATH: cd のサーチパス
# CDPATH に設定したディレクトリ以下のサブディレクトリに直接移動可能
export CDPATH=".:$HOME:$HOME/projects:$HOME/Documents"
# これにより、どこからでも以下が可能:
cd myapp                     # ~/projects/myapp に移動
cd Documents                 # ~/Documents に移動

# cd の終了コード
cd /existing/dir && echo "移動成功"
cd /nonexistent && echo "これは表示されない"  # エラー: 終了コード1

# cd とコマンドの組み合わせ
cd /var/log && ls -la        # 移動後にls実行
(cd /tmp && make)            # サブシェルで移動（元に戻る）
```

### 1.2 cd のショートカットと効率化

```bash
# ============================================
# zsh の cd 拡張機能
# ============================================

# AUTO_CD: ディレクトリ名だけで cd（setopt AUTO_CD が必要）
setopt AUTO_CD
/tmp                         # cd /tmp と同じ
..                           # cd .. と同じ
~                            # cd ~ と同じ

# AUTO_PUSHD: cd するたびに自動でpushd
setopt AUTO_PUSHD
setopt PUSHD_IGNORE_DUPS     # 重複をスタックに入れない
setopt PUSHD_SILENT           # pushd/popd のメッセージを抑制

# cd のスペル訂正（zsh）
setopt CORRECT
# cd /tpm → zsh: correct '/tpm' to '/tmp' [nyae]?

# ディレクトリの補完強化
setopt COMPLETE_IN_WORD       # カーソル位置で補完
setopt ALWAYS_TO_END          # 補完後にカーソルを末尾に

# ハッシュドディレクトリ（zsh）
hash -d projects=~/projects
hash -d docs=~/Documents
hash -d dl=~/Downloads
# 使い方: cd ~projects, cd ~docs, cd ~dl

# bash のエイリアスによる高速移動
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..'
alias -- -='cd -'

# よく使うディレクトリへのブックマーク
alias cdp='cd ~/projects'
alias cdw='cd ~/work'
alias cdl='cd /var/log'
alias cdc='cd ~/.config'
alias cdt='cd /tmp'
```

### 1.3 ディレクトリスタック（pushd/popd）

```bash
# ============================================
# ディレクトリスタックの活用
# ============================================

# pushd: 現在のディレクトリをスタックに保存して移動
pushd /var/log               # /var/log に移動（元の場所をスタックに保存）
pushd /etc                   # /etc に移動（/var/log をスタックに保存）
pushd /tmp                   # /tmp に移動（/etc をスタックに保存）

# dirs: スタックの内容を表示
dirs                         # スタック全体を表示（左が現在地）
dirs -v                      # 番号付きで表示（一番使う）
# 出力例:
# 0  /tmp
# 1  /etc
# 2  /var/log
# 3  /home/user

# popd: スタックから取り出して戻る
popd                         # スタックの先頭に戻る（/etc に移動）
popd                         # さらに戻る（/var/log に移動）
popd                         # さらに戻る（/home/user に移動）

# スタック番号で移動
dirs -v                      # 番号を確認
pushd +2                     # スタックの2番目に移動
pushd +0                     # スタックのローテーション

# スタックから特定の要素を削除
popd +2                      # 2番目のエントリを削除

# 実践的なワークフロー
# 例: 複数のプロジェクトディレクトリ間を行き来する
pushd ~/projects/frontend
pushd ~/projects/backend
pushd ~/projects/infra
dirs -v
# 0  ~/projects/infra
# 1  ~/projects/backend
# 2  ~/projects/frontend
# pushd +1 で backend に切り替え
# pushd +2 で frontend に切り替え

# スタックのクリア
dirs -c                      # スタックを全クリア
```

### 1.4 zoxide（スマートcd）

```bash
# ============================================
# zoxide の導入と活用
# ============================================

# インストール
brew install zoxide          # macOS
sudo apt install zoxide      # Ubuntu/Debian（新しいバージョン）
# curl -sS https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash

# シェルに設定を追加
eval "$(zoxide init zsh)"    # ~/.zshrc に追加
# eval "$(zoxide init bash)" # bash の場合

# 基本的な使い方
z projects                   # 過去の訪問履歴から最適な "projects" ディレクトリに移動
z proj                       # 部分一致でも移動可能
z my app                     # 複数キーワード（"my" と "app" を含むパス）
z -                          # 前のディレクトリに戻る
zi                           # インタラクティブ選択（fzf連携）

# zoxide のスコアリング
# - 頻繁にアクセスするディレクトリほどスコアが高い
# - 最近アクセスしたディレクトリほどスコアが高い
# - 存在しないディレクトリは自動的にデータベースから削除

# データベース管理
zoxide query                 # データベースの内容を表示
zoxide query --list          # パスのみ表示
zoxide query -s              # スコア付きで表示
zoxide query -s projects     # "projects" を含むエントリのスコア
zoxide add /path/to/dir      # パスを手動追加
zoxide remove /path/to/dir   # パスを手動削除
zoxide edit                  # データベースをエディタで編集

# zoxide + fzf の連携
zi                           # fzf でインタラクティブに選択
# Ctrl+R のように、候補一覧から選んで移動できる

# cd の完全な置き換え
# ~/.zshrc に追加（cdコマンドをzoxideで置き換え）
# alias cd='z'

# 実践シナリオ
# 1日の作業の中で何度もプロジェクトディレクトリを行き来する場合
z frontend                   # ~/projects/mycompany/frontend に一発移動
z api                        # ~/work/services/api に一発移動
z infra terraform            # ~/infra/terraform/environments に移動
# → 手動でフルパスを入力する必要がない
```

### 1.5 fzf を使ったディレクトリナビゲーション

```bash
# ============================================
# fzf によるインタラクティブナビゲーション
# ============================================

# Alt+C: ディレクトリを検索して移動（fzfのデフォルトバインド）
# 設定:
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'
export FZF_ALT_C_OPTS='--preview "eza --tree --level=2 --icons {} 2>/dev/null || tree -L 2 {}"'

# カスタム関数: fzf でディレクトリを検索してcd
fcd() {
    local dir
    dir=$(fd --type d --hidden --follow --exclude .git 2>/dev/null | \
        fzf --height 60% --preview 'ls -la {}' --preview-window=right:50%)
    [ -n "$dir" ] && cd "$dir"
}

# 特定のプロジェクトディレクトリに限定した検索
proj() {
    local dir
    dir=$(fd --type d --max-depth 2 . ~/projects ~/work 2>/dev/null | \
        fzf --height 40% --reverse --preview 'ls -la {}')
    [ -n "$dir" ] && cd "$dir"
}

# 最近アクセスしたディレクトリをfzfで選択（zoxide + fzf）
recent() {
    local dir
    dir=$(zoxide query --list | fzf --height 40% --reverse --preview 'ls -la {}')
    [ -n "$dir" ] && cd "$dir"
}

# Git リポジトリのルートに移動
cdgit() {
    local gitroot
    gitroot=$(git rev-parse --show-toplevel 2>/dev/null)
    if [ -n "$gitroot" ]; then
        cd "$gitroot"
    else
        echo "Not in a git repository"
    fi
}
alias gr='cdgit'

# Git リポジトリ一覧からfzfで選択して移動
repos() {
    local repo
    repo=$(fd -H -t d .git ~/projects ~/work 2>/dev/null | \
        sed 's|/\.git$||' | \
        fzf --height 40% --reverse --preview 'git -C {} log --oneline -5')
    [ -n "$repo" ] && cd "$repo"
}
```

---

## 2. ファイル一覧（ls）

### 2.1 ls の基本オプション

```bash
# ============================================
# ls の主要オプション完全ガイド
# ============================================

# 基本表示
ls                           # ファイル一覧（名前のみ）
ls /path/to/dir              # 指定ディレクトリの一覧
ls file1 file2               # 指定ファイルの情報

# 詳細表示
ls -l                        # 長い形式（パーミッション、サイズ等）
ls -la                       # 隠しファイル含む（. から始まるファイル）
ls -lA                       # . と .. を除く隠しファイル
ls -lh                       # サイズを人間可読形式（KB, MB, GB）
ls -lah                      # 最もよく使う組み合わせ

# 出力例:
# drwxr-xr-x  5 user group  160 Feb 16 10:30 Documents
# -rw-r--r--  1 user group 4.0K Feb 16 09:15 README.md
# lrwxr-xr-x  1 user group   20 Jan  5 14:20 link -> /path/to/target
#
# 各カラムの意味:
# d rwx r-x r-x    パーミッション（タイプ + owner + group + other）
# 5                 ハードリンク数
# user              所有者
# group             グループ
# 160               ファイルサイズ（バイト、-h で人間可読に）
# Feb 16 10:30      最終更新日時
# Documents         ファイル/ディレクトリ名

# ソートオプション
ls -lt                       # 更新日時順（新しい順）
ls -ltr                      # 更新日時順（古い順）
ls -lS                       # ファイルサイズ順（大きい順）
ls -lSr                      # ファイルサイズ順（小さい順）
ls -lX                       # 拡張子順（Linuxのみ）
ls -lU                       # ディレクトリ内の順序（ソートなし）
ls -lv                       # バージョン番号順（file1, file2, file10の正しい順序）

# フィルタリング
ls -d */                     # ディレクトリのみ表示
ls -d .*/                    # 隠しディレクトリのみ表示
ls *.md                      # .md ファイルのみ
ls -d .*                     # 隠しファイルのみ（. と .. 含む）

# 表示形式
ls -1                        # 1行1ファイル（スクリプトで有用）
ls -m                        # カンマ区切り
ls -R                        # 再帰的に表示
ls -F                        # タイプ表示（/ ディレクトリ, * 実行可能, @ リンク）
ls --color=auto              # カラー表示（Linux）
ls -G                        # カラー表示（macOS）

# 再帰表示
ls -R                        # サブディレクトリ内も全て表示
ls -R | head -50             # 最初の50行だけ表示

# inode表示
ls -i                        # inode番号を表示
ls -li                       # 詳細表示 + inode番号

# タイムスタンプの種類
ls -l                        # mtime（最終更新日時、デフォルト）
ls -lc                       # ctime（メタデータ変更日時）
ls -lu                       # atime（最終アクセス日時）
ls -l --time=birth           # 作成日時（Linux、対応FS限定）

# タイムスタンプの表示形式
ls -l --time-style=long-iso  # ISO 8601形式（2026-02-16 10:30）
ls -l --time-style=full-iso  # フルISO形式
ls -l --time-style="+%Y-%m-%d %H:%M"  # カスタム形式
```

### 2.2 ls の出力を理解する

```
ls -la の出力の完全な読み方:

total 48                              ← ブロックの合計サイズ
drwxr-xr-x  12 user group 384 Feb 16 10:30 .
drwxr-xr-x   5 user group 160 Feb 16 09:00 ..
-rw-r--r--   1 user group  35 Feb 16 10:30 .gitignore
drwxr-xr-x   8 user group 256 Feb 16 10:30 .git
-rw-r--r--   1 user group 1.2K Feb 16 10:25 Makefile
-rw-r--r--   1 user group 4.0K Feb 16 10:20 README.md
drwxr-xr-x   5 user group 160 Feb 16 10:15 src
lrwxr-xr-x   1 user group   3 Feb 16 10:10 link -> src

ファイルタイプ（先頭1文字）:
  -  通常のファイル
  d  ディレクトリ
  l  シンボリックリンク
  c  キャラクターデバイス（/dev/null等）
  b  ブロックデバイス（/dev/sda等）
  p  名前付きパイプ（FIFO）
  s  ソケット

カラー表示の意味（一般的な設定）:
  青色     ディレクトリ
  緑色     実行可能ファイル
  シアン   シンボリックリンク
  赤色     壊れたシンボリックリンク / アーカイブファイル
  黄色     デバイスファイル
  マゼンタ 画像・動画ファイル
  白色     通常のファイル
```

### 2.3 ls の実用的な使い方

```bash
# ============================================
# 実務でよく使う ls コマンドパターン
# ============================================

# 大きなファイルを見つける
ls -lhS | head -20                   # サイズの大きい順でトップ20

# 最近更新されたファイルを見つける
ls -lt | head -20                    # 最近更新された順でトップ20
ls -ltr                              # 古い順（最新が一番下）

# 特定の拡張子のファイルだけ表示
ls -la *.{js,ts}                     # JavaScript/TypeScript
ls -la *.{jpg,png,gif,svg}           # 画像ファイル
ls -la *.{yml,yaml}                  # YAML ファイル
ls -la *.{log,err}                   # ログファイル

# ファイル数をカウント
ls -1 | wc -l                        # ファイル数
ls -1A | wc -l                       # 隠しファイル含む
ls -1d */ 2>/dev/null | wc -l        # ディレクトリ数

# ディレクトリのみ表示
ls -d */                             # ディレクトリ名一覧
ls -ld */                            # ディレクトリの詳細

# パーミッションで絞り込み（find との組み合わせ）
ls -la | grep "^d"                   # ディレクトリのみ
ls -la | grep "^-"                   # ファイルのみ
ls -la | grep "^l"                   # シンボリックリンクのみ
ls -la | grep "^-..x"               # 実行可能ファイル

# ファイルを見やすく表示
ls -la --group-directories-first     # ディレクトリを先に表示（Linux）
ls -la | sort -k 5 -n               # サイズ順でソート（5番目のカラム）
ls -la | sort -k 6,7                 # 日付順でソート（6,7番目のカラム）

# 特定のパターンの除外
ls -I "*.bak" -I "*.tmp"            # .bak と .tmp を除外（Linux）
ls | grep -v "\.bak$"               # .bak を除外（パイプ版）

# 再帰的なファイル一覧（フルパス）
ls -R                                # 再帰的に全ファイルを表示
find . -name "*.md" -type f          # .md ファイルをフルパスで表示
```

### 2.4 ls のエイリアス設定

```bash
# ============================================
# 推奨する ls のエイリアス設定
# ============================================

# macOS の場合（BSDのls）
alias ls='ls -G'                     # カラー表示
alias ll='ls -lah'                   # 詳細表示（最もよく使う）
alias la='ls -A'                     # 隠しファイル含む
alias lt='ls -lahtr'                 # 更新日時順（新しいのが下）
alias lS='ls -lahSr'                # サイズ順（大きいのが下）
alias l.='ls -d .*'                  # 隠しファイルのみ
alias ld='ls -d */'                  # ディレクトリのみ

# Linux の場合（GNUのls）
alias ls='ls --color=auto --group-directories-first'
alias ll='ls -lah'
alias la='ls -A'
alias lt='ls -lahtr'
alias lS='ls -lahSr'
alias l.='ls -d .*'
alias ld='ls -d */'
alias lx='ls -lXB'                  # 拡張子順

# eza（モダンなls代替）がある場合
if command -v eza &>/dev/null; then
    alias ls='eza --color=auto --icons'
    alias ll='eza -lah --icons --git'
    alias la='eza -a --icons'
    alias lt='eza -la --sort=modified --icons'
    alias lS='eza -la --sort=size --icons'
    alias l.='eza -d .* --icons'
    alias ld='eza -D --icons'
    alias tree='eza --tree --icons'
    alias ltree='eza --tree --icons --long'
fi
```

---

## 3. eza（モダンな ls 代替ツール）

### 3.1 eza の基本

```bash
# ============================================
# eza（旧 exa）の使い方
# ============================================

# インストール
brew install eza              # macOS
sudo apt install eza          # Ubuntu 24.04+
# cargo install eza           # Rust

# 基本的な使い方
eza                           # ファイル一覧（カラー表示）
eza -l                        # 長い形式
eza -la                       # 隠しファイル含む
eza -lah                      # ヘッダー付き
eza -1                        # 1行1ファイル

# eza の強力なオプション

# Git 連携
eza --git                     # Gitのステータスを表示
eza -la --git                 # 詳細表示 + Git ステータス
# 出力例:
# .M  -rw-r--r--  user  1.2k  Feb 16 10:30  modified-file.txt
# N   -rw-r--r--  user  500   Feb 16 10:20  new-file.txt
# Git ステータス: M=変更, N=新規, I=無視, -=追跡対象外

# アイコン表示（Nerd Font が必要）
eza --icons                   # ファイルタイプに応じたアイコン表示
eza -la --icons               # 詳細 + アイコン

# ツリー表示
eza --tree                    # ツリー形式で表示
eza --tree --level=2          # 深さ2まで
eza --tree --level=3 --icons  # 深さ3 + アイコン
eza --tree -I "node_modules|.git|__pycache__"  # 除外パターン

# ソート
eza --sort=name               # 名前順（デフォルト）
eza --sort=modified           # 更新日時順
eza --sort=size               # サイズ順
eza --sort=extension          # 拡張子順
eza --sort=type               # ファイルタイプ順
eza --sort=created            # 作成日時順
eza --sort=accessed           # アクセス日時順
eza --sort=none               # ソートなし
eza -r --sort=size            # 逆順

# フィルタリング
eza --only-dirs               # ディレクトリのみ
eza --only-files              # ファイルのみ
eza -I "*.bak|*.tmp"          # パターンで除外

# グループ化
eza --group-directories-first # ディレクトリを先に表示

# ヘッダーとカラム
eza -lh                       # ヘッダー行を表示
eza -l --no-user              # ユーザー名を非表示
eza -l --no-permissions       # パーミッションを非表示
eza -l --time-style=long-iso  # ISO形式の日時
eza -l --time-style=relative  # 相対時間（3 hours ago等）

# 組み合わせ例
eza -la --icons --git --group-directories-first --sort=modified
eza --tree --level=3 --icons -I "node_modules|.git|dist|build"
eza -la --icons --git --time-style=relative
```

---

## 4. tree コマンド

### 4.1 tree の基本

```bash
# ============================================
# tree コマンドの使い方
# ============================================

# インストール
brew install tree            # macOS
sudo apt install tree        # Ubuntu/Debian

# 基本的な使い方
tree                         # カレントディレクトリのツリー表示
tree /path/to/dir            # 指定ディレクトリ

# 深さの制限
tree -L 1                    # 1階層のみ
tree -L 2                    # 2階層まで
tree -L 3                    # 3階層まで

# フィルタリング
tree -I "node_modules"                    # node_modules を除外
tree -I "node_modules|.git|dist|build"    # 複数パターン除外
tree -P "*.md"                            # .md ファイルのみ表示
tree -P "*.py" --prune                    # .py があるディレクトリだけ表示

# 表示オプション
tree -a                      # 隠しファイル含む
tree -d                      # ディレクトリのみ
tree -f                      # フルパスを表示
tree -p                      # パーミッションを表示
tree -s                      # サイズを表示
tree -h                      # サイズを人間可読で表示
tree -u                      # ユーザー名を表示
tree -g                      # グループ名を表示
tree -D                      # 最終更新日時を表示
tree --du                    # ディレクトリのサイズ合計を表示
tree -C                      # カラー表示（デフォルトでONの場合が多い）
tree -n                      # カラーなし
tree --dirsfirst             # ディレクトリを先に表示

# ソート
tree --sort=name             # 名前順（デフォルト）
tree --sort=mtime            # 更新日時順
tree --sort=size             # サイズ順
tree -r                      # 逆順

# 出力形式
tree -J                      # JSON形式で出力
tree -X                      # XML形式で出力
tree -H .                    # HTML形式で出力（Webブラウザ用）
tree -H . -o tree.html       # HTMLファイルに出力

# ファイル数・ディレクトリ数の表示
tree                         # 最後の行に合計が表示される
# 3 directories, 12 files

# 実用的な使い方
tree -L 2 --dirsfirst -I "node_modules|.git"
tree -L 3 -P "*.py" --prune --dirsfirst
tree -d -L 2                 # ディレクトリ構造だけ2階層

# プロジェクト構造のドキュメント化
tree -L 3 --dirsfirst -I "node_modules|.git|dist|build|__pycache__" > project-structure.txt
```

### 4.2 プロジェクト構造の表示例

```bash
# プロジェクト構造の表示（一般的なWebアプリ）
tree -L 3 --dirsfirst -I "node_modules|.git|dist|build|.next"

# 出力例:
# .
# ├── src/
# │   ├── components/
# │   │   ├── Header.tsx
# │   │   ├── Footer.tsx
# │   │   └── Sidebar.tsx
# │   ├── pages/
# │   │   ├── index.tsx
# │   │   ├── about.tsx
# │   │   └── api/
# │   ├── styles/
# │   │   ├── globals.css
# │   │   └── Home.module.css
# │   └── utils/
# │       ├── api.ts
# │       └── helpers.ts
# ├── public/
# │   ├── favicon.ico
# │   └── images/
# ├── tests/
# │   ├── unit/
# │   └── e2e/
# ├── package.json
# ├── tsconfig.json
# ├── README.md
# └── .env.example
```

---

## 5. パスの基本と応用

### 5.1 絶対パスと相対パス

```
============================================
パスの種類と表記
============================================

■ 絶対パス（Absolute Path）
  ルート（/）から始まる完全なパス
  例:
    /home/user/documents/file.txt
    /var/log/syslog
    /etc/nginx/nginx.conf
    /usr/local/bin/python3

  特徴:
    - 常に同じファイルを指す（現在地に依存しない）
    - スクリプト内で確実にファイルを指定したい場合に使う
    - 長くなりがち

■ 相対パス（Relative Path）
  現在のディレクトリからの相対位置
  例:
    ./documents/file.txt     # 現在のディレクトリの documents
    ../other/file.txt        # 1つ上の other ディレクトリ
    ../../config/app.yml     # 2つ上の config ディレクトリ
    documents/file.txt       # ./ は省略可能

  特徴:
    - 現在のディレクトリに依存する
    - 短く書ける
    - プロジェクト内のファイル参照に便利

■ 特殊なパス
  ~         → ホームディレクトリ（/home/user）
  ~/        → ホームディレクトリの下
  .         → 現在のディレクトリ
  ..        → 親ディレクトリ（1つ上）
  -         → 前のディレクトリ（cd 限定）
  /         → ルートディレクトリ

■ 環境変数によるパス
  $HOME     → ホームディレクトリ（~ と同等）
  $PWD      → 現在のディレクトリ（pwd の出力と同じ）
  $OLDPWD   → 前のディレクトリ（cd - の移動先）
  $TMPDIR   → テンポラリディレクトリ（macOS: /var/folders/...）
  $PATH     → コマンド検索パス（コロン区切り）
```

### 5.2 パスの操作コマンド

```bash
# ============================================
# パスの操作と変換
# ============================================

# パスの取得
pwd                          # 現在のディレクトリ（物理パス）
pwd -L                       # 論理パス（シンボリックリンクを解決しない）
pwd -P                       # 物理パス（シンボリックリンクを解決する）

# パスの分解
basename /path/to/file.txt   # file.txt（ファイル名のみ）
basename /path/to/file.txt .txt  # file（拡張子を除去）
dirname /path/to/file.txt    # /path/to（ディレクトリ部分のみ）

# パスの正規化
realpath /path/with/../symlinks  # 正規化された絶対パス
realpath --relative-to=. /absolute/path  # 相対パスに変換
readlink -f /path/to/symlink    # シンボリックリンクの解決

# パスの存在確認
test -e /path/to/file        # ファイル/ディレクトリが存在するか
test -f /path/to/file        # 通常のファイルか
test -d /path/to/dir         # ディレクトリか
test -L /path/to/link        # シンボリックリンクか

# パスの結合（bash/zsh）
dir="/var/log"
file="syslog"
full_path="${dir}/${file}"    # /var/log/syslog

# パスの一括変換
# 相対パスから絶対パスに変換
readlink -f ./relative/path

# 全ての .txt ファイルの絶対パスを取得
find . -name "*.txt" -exec realpath {} \;

# パスに含まれるシンボリックリンクを解決
realpath /usr/bin/python3    # /usr/local/Cellar/python@3/3.x/bin/python3

# パスの正規化スクリプト（クロスプラットフォーム）
abspath() {
    if [ -d "$1" ]; then
        (cd "$1" && pwd)
    elif [ -f "$1" ]; then
        local dir=$(dirname "$1")
        local base=$(basename "$1")
        (cd "$dir" && echo "$(pwd)/$base")
    else
        echo "Path not found: $1" >&2
        return 1
    fi
}
```

### 5.3 Linux ファイルシステムの構造

```
============================================
FHS (Filesystem Hierarchy Standard) の概要
============================================

/                   ルートディレクトリ（全てのファイルの起点）
├── bin/            基本コマンド（ls, cp, mv等）
├── sbin/           システム管理コマンド（mount, fsck等）
├── boot/           ブートローダー、カーネルイメージ
├── dev/            デバイスファイル（/dev/null, /dev/sda等）
├── etc/            設定ファイル（システム全体の設定）
│   ├── nginx/      Nginx の設定
│   ├── ssh/        SSH の設定
│   ├── passwd      ユーザーアカウント情報
│   ├── shadow      パスワードハッシュ（root のみ読み取り可）
│   ├── hosts       ホスト名の静的テーブル
│   ├── fstab       ファイルシステムのマウントテーブル
│   └── crontab     cron のスケジュール
├── home/           一般ユーザーのホームディレクトリ
│   └── user/
│       ├── .bashrc
│       ├── .ssh/
│       └── ...
├── lib/            共有ライブラリ
├── media/          リムーバブルメディアのマウントポイント
├── mnt/            一時的なマウントポイント
├── opt/            サードパーティソフトウェア
├── proc/           プロセス情報（仮想ファイルシステム）
│   ├── cpuinfo     CPU情報
│   ├── meminfo     メモリ情報
│   ├── uptime      稼働時間
│   └── [PID]/      各プロセスの情報
├── root/           root ユーザーのホームディレクトリ
├── run/            実行時の変数データ
├── srv/            サービスのデータ
├── sys/            カーネル・デバイス情報（仮想ファイルシステム）
├── tmp/            一時ファイル（再起動で削除される場合がある）
├── usr/            ユーザーアプリケーション
│   ├── bin/        一般的なコマンド
│   ├── sbin/       管理コマンド
│   ├── lib/        ライブラリ
│   ├── local/      ローカルインストールしたソフトウェア
│   │   ├── bin/
│   │   ├── lib/
│   │   └── share/
│   ├── share/      共有データ（manページ、ドキュメント等）
│   └── include/    ヘッダファイル
└── var/            可変データ
    ├── log/        ログファイル
    │   ├── syslog
    │   ├── auth.log
    │   └── nginx/
    ├── cache/      キャッシュデータ
    ├── lib/        アプリケーションの永続データ
    ├── mail/       メールボックス
    ├── run/        実行時データ（PIDファイル等）
    ├── spool/      スプールデータ（印刷キュー等）
    └── tmp/        再起動で消えない一時ファイル

macOS固有:
  /Applications      GUIアプリケーション
  /System            macOSシステムファイル
  /Library           システム全体のライブラリ
  ~/Library          ユーザー固有のライブラリ
  /Volumes           マウントされたボリューム
  /private/etc       /etc のシンボリックリンク先
  /private/var       /var のシンボリックリンク先
  /private/tmp       /tmp のシンボリックリンク先
```

```bash
# ============================================
# ファイルシステム構造の探索コマンド
# ============================================

# 重要なディレクトリを確認
ls /etc/                     # システム設定ファイル
ls /var/log/                 # ログファイル
ls /usr/local/bin/           # ローカルインストールされたコマンド
ls /tmp/                     # 一時ファイル

# システム情報の取得
cat /etc/os-release          # OS情報（Linux）
sw_vers                      # macOS バージョン
uname -a                     # カーネル情報
cat /proc/cpuinfo            # CPU情報（Linux）
cat /proc/meminfo            # メモリ情報（Linux）

# マウント情報
mount                        # マウント済みファイルシステム一覧
df -h                        # パーティションの使用状況
lsblk                        # ブロックデバイス一覧（Linux）
diskutil list                # ディスク一覧（macOS）

# ファイルシステムの種類を確認
df -T                        # ファイルシステムのタイプ表示（Linux）
mount | grep "^/"            # マウントされたデバイスのみ
```

---

## 6. ディスク使用量の確認

### 6.1 du（Disk Usage）

```bash
# ============================================
# du コマンドの使い方
# ============================================

# 基本的な使い方
du                           # カレントディレクトリの使用量（サブディレクトリ含む）
du -h                        # 人間可読形式（KB, MB, GB）
du -s                        # 合計のみ表示（summary）
du -sh                       # 合計を人間可読で表示（最もよく使う）
du -sh *                     # 各ファイル/ディレクトリの使用量

# 深さの指定
du -h --max-depth=1          # 1階層のみ（Linux）
du -h -d 1                   # 1階層のみ（macOS/BSD）
du -h -d 2                   # 2階層まで

# ソートして表示
du -sh * | sort -rh          # サイズの大きい順（最もよく使う）
du -sh * | sort -rh | head -10  # トップ10

# 特定のファイルシステムのみ
du -shx                      # 同一ファイルシステムのみ（マウントポイントを跨がない）

# 除外パターン
du -sh --exclude="*.log" *   # .log ファイルを除外
du -sh --exclude=".git" *    # .git を除外
du -sh --exclude="node_modules" *  # node_modules を除外

# 隠しファイルを含む
du -sh .[^.]* *              # 隠しファイルも含めて表示

# 特定のディレクトリの使用量
du -sh ~/projects/           # プロジェクトディレクトリ全体
du -sh /var/log/             # ログディレクトリ全体
du -sh ~/.cache/             # キャッシュディレクトリ全体

# 大きなディレクトリを見つける
du -h -d 1 / 2>/dev/null | sort -rh | head -20  # システム全体のトップ20

# 特定サイズ以上のファイルを見つける
find . -type f -size +100M -exec ls -lh {} \;  # 100MB以上のファイル
find . -type f -size +1G -exec ls -lh {} \;    # 1GB以上のファイル
```

### 6.2 df（Disk Free）

```bash
# ============================================
# df コマンドの使い方
# ============================================

# 基本的な使い方
df                           # 全パーティションの使用状況
df -h                        # 人間可読形式（最もよく使う）
df -H                        # SI単位（1K=1000）
df -T                        # ファイルシステムのタイプも表示（Linux）

# 特定のパスの情報
df -h .                      # 現在のディレクトリが属するパーティション
df -h /home                  # /home パーティション
df -h /var                   # /var パーティション

# inode の使用状況
df -i                        # inode の使用状況
df -ih                       # inode の使用状況（人間可読）
# inode が枯渇すると、ディスク容量が余っていてもファイルが作成できなくなる

# 出力のカスタマイズ
df -h --output=source,fstype,size,used,avail,pcent,target  # 表示カラムを選択（Linux）

# 出力例:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sda1       50G   25G   23G  52% /
# /dev/sda2      200G  150G   40G  79% /home
# tmpfs           8.0G  1.2G  6.8G  15% /tmp
```

### 6.3 ncdu（インタラクティブなディスク使用量ビューア）

```bash
# ============================================
# ncdu の使い方
# ============================================

# インストール
brew install ncdu            # macOS
sudo apt install ncdu        # Ubuntu/Debian

# 基本的な使い方
ncdu                         # カレントディレクトリを分析
ncdu /                       # システム全体を分析
ncdu /home                   # /home を分析
ncdu ~/projects              # プロジェクトディレクトリを分析

# オプション
ncdu -x                      # 同一ファイルシステムのみ
ncdu --exclude ".git"        # .git を除外
ncdu -e                      # エクスポートモード

# ncdu 内の操作
# ↑/↓ or k/j   → 項目間の移動
# Enter or →    → ディレクトリに入る
# ← or <       → 親ディレクトリに戻る
# d             → 選択した項目を削除（確認あり）
# n             → 名前順でソート
# s             → サイズ順でソート
# C             → アイテム数順でソート
# M             → 最終更新日時順でソート
# g             → グラフ表示を切り替え
# q             → 終了
# ?             → ヘルプ

# 結果をファイルに保存して後で閲覧
ncdu -o /tmp/ncdu-export.json ~/  # エクスポート
ncdu -f /tmp/ncdu-export.json     # インポートして閲覧
```

### 6.4 ディスク容量不足の対処法

```bash
# ============================================
# ディスク容量不足時の対処手順
# ============================================

# Step 1: 現在の使用状況を確認
df -h

# Step 2: 大きなディレクトリを特定
du -h -d 1 / 2>/dev/null | sort -rh | head -20

# Step 3: 一般的な容量消費源をチェック

# ログファイル
du -sh /var/log/
sudo find /var/log -name "*.gz" -delete          # 古い圧縮ログを削除
sudo journalctl --vacuum-size=500M                # systemd ログを500MBに制限

# キャッシュ
du -sh ~/.cache/
rm -rf ~/.cache/pip                               # pip キャッシュ
rm -rf ~/.cache/yarn                              # yarn キャッシュ
rm -rf ~/.npm/_cacache                            # npm キャッシュ

# Docker
docker system df                                   # Docker のディスク使用量
docker system prune -af                            # 未使用の全リソースを削除
docker volume prune                                # 未使用ボリュームを削除

# Homebrew（macOS）
brew cleanup --prune=all                           # 古いバージョンを削除
du -sh $(brew --cache)                             # キャッシュサイズ確認
rm -rf $(brew --cache)                             # キャッシュを削除

# 一時ファイル
du -sh /tmp/
sudo rm -rf /tmp/large-temp-files/

# 大きなファイルの検索
find / -type f -size +500M -exec ls -lh {} \; 2>/dev/null
find /home -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# node_modules の削除（全プロジェクト）
find ~/projects -name "node_modules" -type d -prune -exec du -sh {} \;
# 確認後に削除:
find ~/projects -name "node_modules" -type d -prune -exec rm -rf {} +

# .git ディレクトリのサイズ確認
find ~/projects -name ".git" -type d -exec du -sh {} \;
```

---

## 7. 実践演習

### 演習1: [基礎] ── ディレクトリ移動の基本

```bash
# 課題: 以下の操作を実行してください

# 1. ホームディレクトリに移動
cd ~

# 2. /tmp に移動
cd /tmp

# 3. 前のディレクトリに戻る
cd -

# 4. /var/log に移動して内容を確認
cd /var/log && ls -lt | head -10

# 5. ホームディレクトリに戻る
cd

# 6. ディレクトリを作成して移動
mkdir -p /tmp/exercise/subdir && cd /tmp/exercise/subdir

# 7. 現在のディレクトリを確認
pwd

# 8. 2つ上のディレクトリに移動
cd ../..
pwd
# /tmp になるはず
```

### 演習2: [中級] ── ls の高度な使い方

```bash
# 課題: ls コマンドで以下の情報を取得してください

# 1. ホームディレクトリの隠しファイルを含む詳細一覧
ls -lah ~

# 2. /var/log 内で最近更新されたファイルのトップ5
ls -lt /var/log/ | head -6

# 3. /usr/bin 内のファイル数をカウント
ls -1 /usr/bin/ | wc -l

# 4. カレントディレクトリのディレクトリのみをサイズ付きで表示
ls -ld */ 2>/dev/null

# 5. /etc 内のシンボリックリンクのみ表示
ls -la /etc/ | grep "^l"

# 6. ホームディレクトリで最も大きなファイルトップ10
ls -lhS ~ | head -11
```

### 演習3: [中級] ── ディレクトリスタックの活用

```bash
# 課題: pushd/popd を使って複数ディレクトリ間を効率的に移動する

# 1. ホームディレクトリから開始
cd ~

# 2. 3つのディレクトリをスタックに積む
pushd /var/log
pushd /etc
pushd /tmp

# 3. スタックの内容を確認
dirs -v

# 4. /etc（スタック上の特定エントリ）に移動
pushd +1

# 5. スタックの内容を再確認
dirs -v

# 6. popd で順番に戻る
popd
popd
popd

# 7. 元のディレクトリに戻っていることを確認
pwd
```

### 演習4: [上級] ── ディスク使用量の分析

```bash
# 課題: システムのディスク使用状況を分析する

# 1. 全パーティションの使用状況を確認
df -h

# 2. ホームディレクトリ直下の各ディレクトリのサイズを確認
du -sh ~/* 2>/dev/null | sort -rh | head -15

# 3. 隠しディレクトリのサイズも確認
du -sh ~/.[^.]* 2>/dev/null | sort -rh | head -10

# 4. 100MB以上のファイルを検索
find ~ -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# 5. node_modules ディレクトリの合計サイズ
find ~/projects -name "node_modules" -type d -prune 2>/dev/null | \
    xargs du -sh 2>/dev/null | sort -rh

# 6. .git ディレクトリの合計サイズ
find ~/projects -name ".git" -type d 2>/dev/null | \
    xargs du -sh 2>/dev/null | sort -rh | head -10
```

### 演習5: [上級] ── ナビゲーション効率化のセットアップ

```bash
# 課題: 自分の環境に最適なナビゲーション設定を作成する

# ~/.zshrc に以下を追加:

# 1. zoxide の設定
eval "$(zoxide init zsh)"

# 2. ディレクトリ移動のエイリアス
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# 3. よく使うディレクトリのブックマーク
hash -d p=~/projects
hash -d w=~/work
hash -d d=~/Documents
hash -d dl=~/Downloads

# 4. fzf でのディレクトリ移動関数
fcd() {
    local dir
    dir=$(fd --type d --hidden --follow --exclude .git . "${1:-.}" | \
        fzf --height 40% --reverse --preview 'ls -la {}')
    [ -n "$dir" ] && cd "$dir"
}

# 5. Git リポジトリに高速移動
repos() {
    local repo
    repo=$(fd -H -t d .git ~/projects ~/work 2>/dev/null | \
        sed 's|/\.git$||' | \
        fzf --height 40% --reverse --preview 'git -C {} log --oneline -5')
    [ -n "$repo" ] && cd "$repo"
}

# 6. AUTO_CD と補完の設定
setopt AUTO_CD
setopt AUTO_PUSHD
setopt PUSHD_IGNORE_DUPS
setopt PUSHD_SILENT
```

---

## まとめ

| コマンド | 用途 | 備考 |
|---------|------|------|
| cd | ディレクトリ移動 | 最も基本的な移動手段 |
| pwd | 現在のディレクトリ表示 | -P で物理パス |
| ls -lah | 詳細一覧表示 | 最もよく使う組み合わせ |
| eza | モダンな ls 代替 | Git連携、アイコン表示 |
| tree | ツリー表示 | -L で深さ制限 |
| pushd/popd | ディレクトリスタック | 複数ディレクトリ間の移動 |
| zoxide (z) | スマート移動 | 訪問履歴ベースの高速移動 |
| du -sh | ディスク使用量 | ディレクトリ単位の容量確認 |
| df -h | パーティション情報 | 全体の使用状況確認 |
| ncdu | インタラクティブ分析 | ディスク使用量の可視化 |

### 効率的なナビゲーションのポイント

1. **zoxide を導入する** -- 過去の訪問履歴から最適なディレクトリに一発移動
2. **fzf と組み合わせる** -- ディレクトリ名を覚えていなくてもインタラクティブに検索
3. **エイリアスとブックマークを設定する** -- よく行くディレクトリにはショートカットを用意
4. **AUTO_CD を有効にする** -- zshならディレクトリ名だけで移動可能
5. **pushd/popd を活用する** -- 複数ディレクトリ間を頻繁に行き来する場合に有効
6. **eza + tree でファイル一覧を見やすくする** -- Git連携とアイコン表示で視認性向上
7. **ncdu でディスク使用量を定期的にチェック** -- 不要ファイルの発見と容量管理

---

## 次に読むべきガイド
→ [[01-file-crud.md]] — ファイルの作成・コピー・移動・削除

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.2-3, No Starch Press, 2019.
2. Ward, B. "How Linux Works." 3rd Ed, Ch.2, No Starch Press, 2021.
3. Filesystem Hierarchy Standard: https://refspecs.linuxfoundation.org/FHS_3.0/
4. zoxide 公式リポジトリ: https://github.com/ajeetdsouza/zoxide
5. eza 公式リポジトリ: https://github.com/eza-community/eza
6. fzf 公式リポジトリ: https://github.com/junegunn/fzf
