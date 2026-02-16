# バージョンマネージャー

> プログラミング言語のバージョンをプロジェクト単位で管理し、チーム全体で統一された開発環境を実現するためのガイド。

## この章で学ぶこと

1. nvm / fnm / Volta による Node.js のバージョン管理と使い分け
2. pyenv (Python)、rustup (Rust) の設定と運用方法
3. mise (旧 rtx) を使った統合的なバージョン管理
4. Go、Java、Ruby など他言語のバージョン管理
5. CI/CD パイプラインとバージョンマネージャーの連携
6. トラブルシューティングと移行ガイド

---

## 1. なぜバージョンマネージャーが必要か

### 1.1 バージョン管理なしの問題

```
バージョンマネージャーなしの世界:

  開発者A (Node 18)          開発者B (Node 20)
  ┌──────────────────┐      ┌──────────────────┐
  │ npm install      │      │ npm install      │
  │   → 成功 ✅      │      │   → 失敗 ❌       │
  │                  │      │ (engines不一致)   │
  │ npm run build    │      │ npm run build    │
  │   → 成功 ✅      │      │   → 型エラー ❌   │
  └──────────────────┘      └──────────────────┘

  CI (Node 22)
  ┌──────────────────┐
  │ npm test         │
  │   → 失敗 ❌       │
  │ (API差異)        │
  └──────────────────┘

  全員バラバラ → "僕の環境では動くんだけど..."
```

### 1.2 バージョン管理がもたらす効果

```
バージョンマネージャーで統一された世界:

  .node-version: "20.11.0"
         │
         ├──→ 開発者A: fnm use → Node 20.11.0
         ├──→ 開発者B: fnm use → Node 20.11.0
         ├──→ CI:      setup-node → Node 20.11.0
         └──→ Docker:  FROM node:20.11.0

  全員同じバージョン → 再現性 100%

  追加の効果:
  ┌─────────────────────────────────────────────┐
  │ 1. オンボーディング時間の短縮                  │
  │    新メンバーが cd project && fnm use で即開発 │
  │                                                │
  │ 2. バグ再現の容易さ                            │
  │    同一環境 → 同一結果が保証される              │
  │                                                │
  │ 3. セキュリティパッチの統一適用                 │
  │    バージョンファイル更新 → 全員に自動反映      │
  │                                                │
  │ 4. 複数プロジェクトの並行開発                   │
  │    Project A (Node 18) と                      │
  │    Project B (Node 22) を同時に開発可能        │
  └─────────────────────────────────────────────┘
```

### 1.3 主要ツール比較

| ツール | 対象言語 | 速度 | .nvmrc 互換 | シェル起動影響 | 自動切替 |
|--------|---------|------|------------|-------------|---------|
| nvm | Node.js | 遅い | ネイティブ | 大きい | あり |
| fnm | Node.js | 高速 | あり | 小さい | あり |
| Volta | Node.js (+npm/yarn) | 高速 | 部分的 | なし | あり |
| pyenv | Python | 普通 | - | 中程度 | あり |
| rustup | Rust | 高速 | - | なし | あり |
| mise | 多言語 | 高速 | あり | 小さい | あり |
| asdf | 多言語 | 遅い | プラグイン | 中程度 | あり |
| goenv | Go | 普通 | - | 中程度 | あり |
| sdkman | Java/Kotlin/Scala | 普通 | - | 中程度 | あり |
| rbenv | Ruby | 普通 | - | 中程度 | あり |

### 1.4 バージョンマネージャーの動作原理

```
バージョンマネージャーの共通メカニズム:

  1. PATH シム方式 (pyenv, rbenv)
  ┌────────────────────────────────────────────┐
  │ PATH の先頭にシムディレクトリを挿入          │
  │                                              │
  │ PATH=~/.pyenv/shims:$ORIGINAL_PATH          │
  │                                              │
  │ python コマンド実行時:                       │
  │   ~/.pyenv/shims/python (シムスクリプト)     │
  │     → .python-version を読む                │
  │     → 正しいバージョンの python に転送       │
  │     → ~/.pyenv/versions/3.12.3/bin/python   │
  └────────────────────────────────────────────┘

  2. PATH 動的書換方式 (fnm, mise)
  ┌────────────────────────────────────────────┐
  │ シェルフックで cd 時に PATH を書き換え       │
  │                                              │
  │ cd ~/project-a の時:                        │
  │   PATH=~/.fnm/node-versions/v20/bin:...    │
  │                                              │
  │ cd ~/project-b の時:                        │
  │   PATH=~/.fnm/node-versions/v22/bin:...    │
  │                                              │
  │ メリット: シムのオーバーヘッドなし            │
  └────────────────────────────────────────────┘

  3. プロキシバイナリ方式 (Volta)
  ┌────────────────────────────────────────────┐
  │ Volta がインストールする node バイナリ自体が │
  │ プロキシとして動作                           │
  │                                              │
  │ ~/.volta/bin/node を実行すると:              │
  │   1. カレントディレクトリの package.json 確認│
  │   2. volta.node フィールドからバージョン特定│
  │   3. 正しいバージョンの Node.js で実行       │
  │                                              │
  │ メリット: シェルフック不要・起動影響ゼロ      │
  └────────────────────────────────────────────┘
```

---

## 2. Node.js バージョン管理

### 2.1 fnm (Fast Node Manager) -- 推奨

```bash
# ─── インストール ───
# macOS
brew install fnm
# Linux/macOS (curl)
curl -fsSL https://fnm.vercel.app/install | bash
# Windows
winget install Schniz.fnm
# Cargo (Rust 環境がある場合)
cargo install fnm

# ─── シェル設定 ───
# ~/.zshrc に追加
eval "$(fnm env --use-on-cd --shell zsh)"
# ~/.bashrc に追加
eval "$(fnm env --use-on-cd --shell bash)"
# ~/.config/fish/config.fish に追加
fnm env --use-on-cd --shell fish | source
# PowerShell ($PROFILE に追加)
fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression

# ─── 基本操作 ───
fnm list-remote              # 利用可能なバージョン一覧
fnm install 20               # Node.js 20.x 最新をインストール
fnm install 22               # Node.js 22.x 最新をインストール
fnm install --lts            # 最新 LTS をインストール
fnm use 20                   # 現在のシェルで切替
fnm default 20               # デフォルトバージョン設定
fnm list                     # インストール済み一覧
fnm current                  # 現在のバージョン確認
fnm uninstall 18             # 不要バージョン削除

# ─── プロジェクト設定 ───
echo "20" > .node-version    # プロジェクトルートに配置
# → cd でディレクトリに入ると自動切替 (--use-on-cd)

# ─── 特定のマイナー/パッチバージョンを指定 ───
echo "20.11.0" > .node-version
fnm install                  # .node-version に記載されたバージョンをインストール
fnm use                      # .node-version に記載されたバージョンに切替
```

### 2.1.1 fnm の高度な設定

```bash
# ─── fnm 環境変数によるカスタマイズ ───
# ~/.zshrc に追加

# インストールディレクトリの変更
export FNM_DIR="$HOME/.fnm"

# Corepack を自動有効化
export FNM_COREPACK_ENABLED="true"

# バージョン解決方式（.node-version を上位ディレクトリも探索）
export FNM_RESOLVE_ENGINES="true"

# ログレベル設定
export FNM_LOGLEVEL="info"  # quiet, info, all, error

# 全オプションを含む完全な設定
eval "$(fnm env --use-on-cd --version-file-strategy=recursive --corepack-enabled --shell zsh)"

# ─── バージョンファイル戦略 ───
# recursive: カレントから上位ディレクトリを再帰的に探索
# local: カレントディレクトリのみ
eval "$(fnm env --use-on-cd --version-file-strategy=recursive --shell zsh)"

# ─── fnm completions ───
# zsh 補完の有効化
fnm completions --shell zsh > "${fpath[1]}/_fnm"
# bash 補完
fnm completions --shell bash > /etc/bash_completion.d/fnm

# ─── エイリアス管理 ───
fnm alias 20.11.0 lts-iron   # カスタムエイリアス作成
fnm alias 22.0.0 latest      # 最新版にエイリアス
fnm alias list                # エイリアス一覧
fnm default lts-iron          # エイリアスをデフォルトに設定
```

### 2.1.2 fnm のベンチマーク

```
fnm vs nvm シェル起動時間の比較:

  nvm をロードした場合:
  $ time zsh -i -c exit
  real    0m0.523s    ← 500ms 以上のオーバーヘッド
  user    0m0.312s
  sys     0m0.178s

  fnm をロードした場合:
  $ time zsh -i -c exit
  real    0m0.047s    ← 50ms 以下
  user    0m0.028s
  sys     0m0.015s

  差分: 約 10倍の速度差
  (1日 100回ターミナルを開く場合、年間 4.8時間の差)

  バージョン切替速度の比較:
  ┌──────────┬─────────┬─────────┐
  │ 操作     │   nvm   │   fnm   │
  ├──────────┼─────────┼─────────┤
  │ use      │  280ms  │   12ms  │
  │ install  │ 15.2s   │  14.8s  │ ← ダウンロード依存で差は小さい
  │ list     │  150ms  │    8ms  │
  │ current  │  120ms  │    3ms  │
  └──────────┴─────────┴─────────┘
```

### 2.2 nvm (Node Version Manager)

```bash
# ─── インストール ───
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# ─── ~/.zshrc に自動追加される設定 ───
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# ─── 基本操作 ───
nvm install 20               # インストール
nvm install --lts            # 最新 LTS
nvm use 20                   # 切替
nvm alias default 20         # デフォルト設定
nvm ls                       # インストール済み一覧
nvm ls-remote --lts          # リモートの LTS 一覧
nvm uninstall 18             # 不要バージョン削除

# ─── .nvmrc ───
echo "20" > .nvmrc
nvm use                      # .nvmrc のバージョンを使用

# ─── 自動切替スクリプト (~/.zshrc に追加) ───
autoload -U add-zsh-hook
load-nvmrc() {
  local nvmrc_path="$(nvm_find_nvmrc)"
  if [ -n "$nvmrc_path" ]; then
    local nvmrc_node_version=$(nvm version "$(cat "${nvmrc_path}")")
    if [ "$nvmrc_node_version" = "N/A" ]; then
      nvm install
    elif [ "$nvmrc_node_version" != "$(nvm version)" ]; then
      nvm use
    fi
  fi
}
add-zsh-hook chpwd load-nvmrc
load-nvmrc
```

### 2.2.1 nvm の遅延ロード最適化

```bash
# nvm のシェル起動時間を改善する遅延ロード設定
# ~/.zshrc に追加（nvm 標準設定の代わりに使用）

export NVM_DIR="$HOME/.nvm"

# nvm を遅延ロードする関数群
lazy_load_nvm() {
  unset -f nvm node npm npx yarn pnpm corepack
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
  [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
}

# コマンド初回呼び出し時にロード
nvm() { lazy_load_nvm; nvm "$@"; }
node() { lazy_load_nvm; node "$@"; }
npm() { lazy_load_nvm; npm "$@"; }
npx() { lazy_load_nvm; npx "$@"; }
yarn() { lazy_load_nvm; yarn "$@"; }
pnpm() { lazy_load_nvm; pnpm "$@"; }
corepack() { lazy_load_nvm; corepack "$@"; }

# 効果:
# - シェル起動時間: 500ms → 50ms（nvm 本体のロードを遅延）
# - 初回 node コマンド実行時に一度だけロード
# - デメリット: 初回コマンドが少し遅い（500ms 程度の追加遅延）
```

### 2.2.2 nvm からの移行パッケージ保持

```bash
# nvm でグローバルインストールしたパッケージを新バージョンに引き継ぐ
nvm install 22 --reinstall-packages-from=20

# nvm から fnm への移行手順
# 1. 現在のバージョンを確認
nvm ls
#   v18.19.1
#   v20.11.0
# → default -> 20

# 2. fnm をインストール
brew install fnm

# 3. シェル設定を置き換え
# ~/.zshrc から nvm 関連行を削除し、以下を追加:
eval "$(fnm env --use-on-cd --shell zsh)"

# 4. 同じバージョンをインストール
fnm install 18
fnm install 20
fnm default 20

# 5. .nvmrc はそのまま使える（fnm は .nvmrc を読める）

# 6. nvm を削除
rm -rf ~/.nvm
# ~/.zshrc から NVM_DIR 関連の行を削除
```

### 2.3 Volta

```bash
# ─── インストール ───
# macOS / Linux
curl https://get.volta.sh | bash
# Windows
# https://github.com/volta-cli/volta/releases から .msi をダウンロード

# ─── 基本操作 ───
volta install node@20        # Node.js インストール
volta install node@latest    # 最新版インストール
volta install npm@10         # npm バージョン固定
volta install yarn@4         # yarn バージョン固定
volta install pnpm@9         # pnpm バージョン固定

# ─── プロジェクト固定 (package.json に記録) ───
volta pin node@20
volta pin npm@10
volta pin yarn@4

# package.json に自動追記される:
# {
#   "volta": {
#     "node": "20.11.0",
#     "npm": "10.2.4",
#     "yarn": "4.1.0"
#   }
# }

# ─── グローバルツールのインストール ───
volta install typescript      # tsc コマンドをグローバルに利用可能
volta install @angular/cli    # ng コマンドをグローバルに利用可能
volta install create-react-app

# ─── 情報確認 ───
volta list                   # インストール済みツール一覧
volta list all               # 全バージョン一覧
volta which node             # 現在のプロジェクトで使われる node のパス
```

### 2.3.1 Volta の特殊な機能

```bash
# ─── Volta のプロジェクト自動検出 ───
# package.json の volta フィールドがあるディレクトリに入ると
# 自動的にそのバージョンの Node.js が使われる

# プロジェクトA (Node 18)
$ cd ~/projects/legacy-app
$ node --version
v18.19.1

# プロジェクトB (Node 22)
$ cd ~/projects/new-app
$ node --version
v22.0.0

# シェルフック不要 — Volta の node バイナリ自体がプロキシ

# ─── Volta のツールチェーン管理 ───
# Volta はパッケージマネージャーのバージョンも固定できる
volta pin node@20.11.0
volta pin npm@10.2.4
volta pin yarn@4.1.0

# チームメンバーが Volta を使っていれば、
# clone 後すぐに同じバージョンで開発可能

# ─── package.json の engines との連携 ───
# Volta は engines フィールドも参考にするが、
# volta フィールドが優先される
{
  "engines": {
    "node": ">=20.0.0"
  },
  "volta": {
    "node": "20.11.0"
  }
}

# ─── Volta のフック ───
# ~/.volta/hooks.json で追加設定が可能
{
  "node": {
    "index": {
      "prefix": "https://your-mirror.example.com/node/"
    }
  }
}
```

### 2.4 Node.js バージョンマネージャー選定フロー

```
どの Node.js バージョンマネージャーを使うべきか？

                    START
                      │
                      ▼
              チーム開発？ ──── No ──→ fnm (軽量・高速)
                   │
                  Yes
                   │
                   ▼
          npm/yarn バージョンも
          固定したい？ ──── Yes ──→ Volta
                   │                   (package.json管理)
                  No
                   │
                   ▼
          既存の .nvmrc が
          ある？ ──── Yes ──→ fnm (.nvmrc互換 + 高速)
                   │
                  No
                   │
                   ▼
          多言語プロジェクト？ ──── Yes ──→ mise (統合管理)
                   │
                  No
                   │
                   ▼
              fnm (デフォルト推奨)
```

### 2.5 Node.js の LTS スケジュール理解

```
Node.js リリーススケジュール:

  バージョン  │ ステータス   │ LTS 開始    │ EOL
  ──────────┼────────────┼────────────┼──────────
  18.x      │ Maintenance │ 2022-10    │ 2025-04
  20.x      │ LTS Active  │ 2023-10    │ 2026-04
  22.x      │ LTS Active  │ 2024-10    │ 2027-04
  24.x      │ Current     │ 2025-10    │ 2028-04

  偶数バージョン = LTS 対象
  奇数バージョン = Current のみ（短命）

  推奨戦略:
  ┌─────────────────────────────────────────────┐
  │ 本番環境: 最新の Active LTS を使用            │
  │ 開発環境: Active LTS + 次の Current も検証    │
  │ レガシー: Maintenance LTS（パッチのみ提供）   │
  │                                               │
  │ バージョンアップのタイミング:                   │
  │   新 LTS リリース後 1-2ヶ月で検証開始          │
  │   エコシステムの互換性確認後に移行              │
  └─────────────────────────────────────────────┘
```

---

## 3. Python バージョン管理 (pyenv)

### 3.1 セットアップ

```bash
# ─── インストール ───
# macOS
brew install pyenv pyenv-virtualenv

# Linux (Ubuntu/Debian)
curl https://pyenv.run | bash

# Linux (依存パッケージのインストールが必要)
sudo apt-get update && sudo apt-get install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev

# ─── シェル設定 (~/.zshrc) ───
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# ─── bash の場合 (~/.bashrc) ───
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# ─── ビルド依存のインストール (macOS) ───
brew install openssl readline sqlite3 xz zlib tcl-tk

# ─── 基本操作 ───
pyenv install --list | grep '3.12'   # 利用可能バージョン
pyenv install --list | grep '3.13'   # 最新バージョンの確認
pyenv install 3.12.3                  # インストール
pyenv global 3.12.3                   # グローバルデフォルト
pyenv local 3.12.3                    # プロジェクト固定 (.python-version)
pyenv versions                        # インストール済み一覧
pyenv version                         # 現在のバージョン
pyenv uninstall 3.11.0                # 不要バージョン削除

# ─── 仮想環境 ───
pyenv virtualenv 3.12.3 myproject-env
pyenv activate myproject-env
pyenv deactivate
```

### 3.2 pyenv のビルドトラブルシューティング

```bash
# ─── macOS でよくあるビルドエラーと解決策 ───

# エラー: "zlib not available"
CFLAGS="-I$(brew --prefix zlib)/include" \
LDFLAGS="-L$(brew --prefix zlib)/lib" \
pyenv install 3.12.3

# エラー: "openssl not found"
CONFIGURE_OPTS="--with-openssl=$(brew --prefix openssl@3)" \
pyenv install 3.12.3

# macOS Sonoma 以降の包括的な環境変数設定
export LDFLAGS="-L$(brew --prefix openssl@3)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib"
export CPPFLAGS="-I$(brew --prefix openssl@3)/include -I$(brew --prefix readline)/include -I$(brew --prefix zlib)/include"
export PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig:$(brew --prefix readline)/lib/pkgconfig:$(brew --prefix zlib)/lib/pkgconfig"
pyenv install 3.12.3

# ─── Linux でよくあるビルドエラー ───

# エラー: "No module named '_ctypes'"
sudo apt-get install libffi-dev
pyenv install 3.12.3

# エラー: "ModuleNotFoundError: No module named '_lzma'"
sudo apt-get install liblzma-dev
pyenv install 3.12.3

# エラー: "WARNING: The Python tkinter extension was not compiled"
sudo apt-get install tk-dev
pyenv install 3.12.3

# ─── 最適化ビルド ───
# PROFILE_TASK を使ってプロファイルガイド最適化 (PGO) を有効化
PYTHON_CONFIGURE_OPTS="--enable-optimizations --with-lto" \
PYTHON_CFLAGS="-march=native -mtune=native" \
pyenv install 3.12.3

# ─── デバッグビルド ───
# メモリリークやセグフォルトの調査用
pyenv install --debug 3.12.3
```

### 3.3 pyenv-virtualenv の高度な使い方

```bash
# ─── 仮想環境の作成と管理 ───
pyenv virtualenv 3.12.3 myproject-3.12    # バージョン名付き仮想環境
pyenv virtualenvs                          # 仮想環境の一覧
pyenv virtualenv-delete myproject-3.12     # 仮想環境の削除

# ─── プロジェクトごとの自動有効化 ───
cd ~/projects/myproject
pyenv local myproject-3.12
# → .python-version に "myproject-3.12" が書かれる
# → 以降このディレクトリに入ると自動で仮想環境が有効になる

# ─── 複数 Python バージョンでのテスト ───
# tox や nox と組み合わせて複数バージョンテスト
pyenv install 3.11.8
pyenv install 3.12.3
pyenv install 3.13.0
pyenv local 3.12.3 3.11.8 3.13.0  # 複数バージョンを設定

# tox.ini
# [tox]
# envlist = py311, py312, py313
# [testenv]
# commands = pytest

# ─── pyenv と uv の併用 ───
# pyenv で Python バージョンを管理し、uv でパッケージを管理
pyenv local 3.12.3
uv venv                      # pyenv の Python を使って仮想環境作成
uv pip install -r requirements.txt
```

### 3.4 uv による Python バージョン管理

```bash
# uv は pyenv の代替としても使える（2025年以降の推奨）
# Python 自体のインストール・管理が可能

# ─── Python のインストール ───
uv python install 3.12       # Python 3.12 をインストール
uv python install 3.11 3.12 3.13  # 複数バージョンを一括インストール
uv python list                # 利用可能バージョン一覧
uv python find 3.12           # インストール済み 3.12 のパスを表示

# ─── プロジェクト固定 ───
uv python pin 3.12            # .python-version を生成

# ─── pyenv との違い ───
# pyenv: ソースからビルド（ビルド依存が必要・時間がかかる）
# uv:    プリビルドバイナリをダウンロード（数秒で完了）
#        → python-build-standalone プロジェクトのバイナリを使用

# ─── uv で Python + パッケージを統合管理 ───
uv init my-project            # プロジェクト初期化
cd my-project
uv python pin 3.12            # Python バージョン固定
uv add requests flask         # パッケージ追加
uv run python main.py         # 仮想環境内で実行
```

---

## 4. Rust バージョン管理 (rustup)

### 4.1 セットアップ

```bash
# ─── インストール ───
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# インストール時のオプション選択
# 1) Proceed with standard installation (default)
# 2) Customize installation
# 3) Cancel installation
# → 通常は 1 を選択

# ─── シェル設定（自動で追加されるが確認） ───
# ~/.zshrc または ~/.bashrc
source "$HOME/.cargo/env"

# ─── 基本操作 ───
rustup show                          # 現在の toolchain
rustup update                        # 全 toolchain 更新
rustup default stable                # デフォルト設定
rustup default nightly               # nightly をデフォルトに
rustup toolchain install nightly     # nightly インストール
rustup toolchain install 1.77.0     # 特定バージョンをインストール
rustup toolchain list                # インストール済み toolchain 一覧
rustup toolchain uninstall nightly   # 不要 toolchain 削除

# ─── プロジェクト固定 ───
# rust-toolchain.toml (プロジェクトルート)
cat << 'EOF' > rust-toolchain.toml
[toolchain]
channel = "1.77.0"
components = ["rustfmt", "clippy"]
targets = ["wasm32-unknown-unknown"]
EOF

# ─── コンポーネント管理 ───
rustup component add rustfmt         # フォーマッタ
rustup component add clippy          # リンター
rustup component add rust-analyzer   # LSP
rustup component add rust-src        # ソースコード（IDE 補完用）
rustup component add llvm-tools      # LLVM ツール（カバレッジ等）
rustup component add miri            # 未定義動作検出ツール（nightly のみ）
rustup component list                # 利用可能コンポーネント一覧
```

### 4.2 rustup の高度な使い方

```bash
# ─── クロスコンパイル ───
# ターゲットプラットフォームの追加
rustup target add x86_64-unknown-linux-musl     # 静的リンク Linux
rustup target add aarch64-unknown-linux-gnu     # ARM64 Linux
rustup target add wasm32-unknown-unknown        # WebAssembly
rustup target add aarch64-apple-darwin          # Apple Silicon
rustup target add x86_64-pc-windows-msvc        # Windows

# クロスコンパイルの実行
cargo build --target x86_64-unknown-linux-musl

# ─── nightly 機能の利用 ───
# 特定のファイルだけ nightly を使う
rustup run nightly cargo build
rustup run nightly cargo +nightly fmt

# nightly のみの機能をプロジェクトで使う
cat << 'EOF' > rust-toolchain.toml
[toolchain]
channel = "nightly-2024-03-15"
components = ["rustfmt", "clippy", "miri", "rust-src"]
EOF

# ─── rustup のプロキシ設定 ───
# 企業プロキシ環境での設定
export RUSTUP_DIST_SERVER="https://your-mirror.example.com/rustup"
export RUSTUP_UPDATE_ROOT="https://your-mirror.example.com/rustup/rustup"

# ─── rustup self コマンド ───
rustup self update              # rustup 自体を更新
rustup self uninstall           # rustup と全 toolchain を削除

# ─── オーバーライド ───
# 特定ディレクトリで異なる toolchain を使用
rustup override set nightly     # カレントディレクトリ用
rustup override list            # オーバーライド一覧
rustup override unset           # オーバーライド解除
```

### 4.3 rust-toolchain.toml の詳細設定

```toml
# rust-toolchain.toml - プロジェクトルートに配置

[toolchain]
# チャンネル指定（以下のいずれか）
channel = "1.77.0"          # 特定バージョン
# channel = "stable"        # 最新安定版（推奨しない: 再現性が低い）
# channel = "nightly"       # 最新 nightly
# channel = "nightly-2024-03-15"  # 日付指定 nightly

# 必要なコンポーネント
components = [
  "rustfmt",        # コードフォーマッタ
  "clippy",         # リンター
  "rust-analyzer",  # LSP サーバー
  "rust-src",       # ソースコード（IDE 補完に必要）
]

# クロスコンパイルターゲット
targets = [
  "x86_64-unknown-linux-musl",
  "wasm32-unknown-unknown",
  "aarch64-apple-darwin",
]

# プロファイル（minimal, default, complete）
profile = "default"
```

---

## 5. mise (統合バージョンマネージャー)

### 5.1 セットアップ

```bash
# ─── インストール ───
# macOS
brew install mise
# Linux (推奨)
curl https://mise.run | sh
# npm
npm install -g @jdx/mise
# cargo
cargo install mise

# ─── シェル設定 ───
# ~/.zshrc
eval "$(mise activate zsh)"
# ~/.bashrc
eval "$(mise activate bash)"
# ~/.config/fish/config.fish
mise activate fish | source

# ─── 基本操作 ───
mise use node@20              # Node.js インストール & 設定
mise use python@3.12          # Python インストール & 設定
mise use go@1.22              # Go インストール & 設定
mise use java@21              # Java インストール & 設定
mise use ruby@3.3             # Ruby インストール & 設定

# ─── プロジェクト設定 (.mise.toml) ───
cat << 'EOF' > .mise.toml
[tools]
node = "20"
python = "3.12"
terraform = "1.7"

[env]
NODE_ENV = "development"
DATABASE_URL = "postgresql://localhost:5432/mydb"
EOF

# mise が .nvmrc, .python-version, .tool-versions も読める
mise ls                       # インストール済み一覧
mise outdated                 # 更新可能なツール表示
mise prune                    # 未使用バージョンの削除
```

### 5.2 mise のアーキテクチャ

```
mise の動作原理:

  .mise.toml / .nvmrc / .tool-versions
         │
         ▼
  ┌──────────────────────────────────────┐
  │  mise activate (シェルフック)          │
  │                                        │
  │  cd コマンド時:                        │
  │    1. 設定ファイルを検索               │
  │    2. 必要なバージョンを特定           │
  │    3. PATH を動的に書き換え            │
  │                                        │
  │  ~/.local/share/mise/installs/        │
  │  ├── node/                            │
  │  │   ├── 18.19.0/                     │
  │  │   └── 20.11.0/ ← PATH に追加      │
  │  ├── python/                          │
  │  │   └── 3.12.3/                      │
  │  └── go/                              │
  │      └── 1.22.0/                      │
  └──────────────────────────────────────┘

  設定ファイルの優先順位:
  ┌──────────────────────────────────────┐
  │ 1. .mise.local.toml  (gitignore 推奨) │
  │ 2. .mise.toml                         │
  │ 3. .mise/config.toml                  │
  │ 4. .tool-versions   (asdf 互換)       │
  │ 5. .node-version    (fnm/nvm 互換)    │
  │ 6. .python-version  (pyenv 互換)      │
  │ 7. ~/.config/mise/config.toml (global)│
  └──────────────────────────────────────┘
```

### 5.3 mise の高度な設定

```toml
# .mise.toml - 全機能を活用した設定例

# ツールバージョン指定
[tools]
node = "20"                    # メジャーバージョン指定（最新パッチを自動選択）
python = "3.12.3"              # 完全バージョン指定
go = "latest"                  # 最新版を常に使用
rust = "1.77.0"                # Rust（rustup 連携）
terraform = "1.7"              # HashiCorp Terraform
kubectl = "1.29"               # Kubernetes CLI
awscli = "2"                   # AWS CLI v2
java = "temurin-21"            # Eclipse Temurin JDK 21

# 環境変数設定
[env]
NODE_ENV = "development"
PYTHONDONTWRITEBYTECODE = "1"
RUST_BACKTRACE = "1"
LOG_LEVEL = "debug"

# .env ファイルの読み込み
# mise は .env ファイルも自動で読み込める
_.file = ".env"
_.path = "./node_modules/.bin"  # PATH に追加

# タスク定義（mise tasks）
[tasks.dev]
run = "npm run dev"
description = "開発サーバー起動"

[tasks.test]
run = "npm test"
description = "テスト実行"

[tasks.lint]
run = ["npm run lint", "npm run typecheck"]
description = "リント & 型チェック"

[tasks.setup]
run = """
npm install
cp .env.example .env
npm run db:migrate
"""
description = "プロジェクト初期セットアップ"
```

```bash
# ─── mise tasks の実行 ───
mise run dev                  # 開発サーバー起動
mise run test                 # テスト実行
mise run lint                 # リント & 型チェック
mise tasks                    # 利用可能タスク一覧

# ─── mise のグローバル設定 ───
# ~/.config/mise/config.toml
cat << 'EOF' > ~/.config/mise/config.toml
[tools]
node = "20"          # デフォルトの Node.js バージョン
python = "3.12"      # デフォルトの Python バージョン

[settings]
experimental = true   # 実験的機能を有効化
verbose = false       # 詳細出力を無効化
asdf_compat = true    # asdf 互換モード

[env]
EDITOR = "code --wait"
EOF
```

### 5.4 asdf からの移行

```bash
# mise は asdf のドロップイン代替として使える

# asdf の .tool-versions をそのまま読める
# .tool-versions
# nodejs 20.11.0
# python 3.12.3
# golang 1.22.0

# asdf → mise 移行手順
# 1. mise をインストール
brew install mise

# 2. シェル設定を変更
# ~/.zshrc から asdf 関連行を削除
# 以下を追加:
eval "$(mise activate zsh)"

# 3. .tool-versions はそのまま使える
# (mise は .tool-versions を自動的に読む)

# 4. プラグイン不要
# asdf はプラグインのインストールが必要だったが
# mise は組み込みで多くのツールに対応

# 5. 互換性の確認
mise ls                       # asdf で管理していたツールが表示される

# 6. 段階的に .mise.toml に移行（任意）
mise settings set asdf_compat true
```

---

## 6. Go バージョン管理

### 6.1 goenv によるバージョン管理

```bash
# ─── インストール ───
git clone https://github.com/go-nv/goenv.git ~/.goenv

# ─── シェル設定 (~/.zshrc) ───
export GOENV_ROOT="$HOME/.goenv"
export PATH="$GOENV_ROOT/bin:$PATH"
eval "$(goenv init -)"
export PATH="$GOROOT/bin:$PATH"
export PATH="$GOPATH/bin:$PATH"

# ─── 基本操作 ───
goenv install --list           # 利用可能バージョン
goenv install 1.22.0           # インストール
goenv global 1.22.0            # グローバルデフォルト
goenv local 1.22.0             # プロジェクト固定 (.go-version)
goenv versions                 # インストール済み一覧
```

### 6.2 mise による Go バージョン管理（推奨）

```bash
# goenv の代わりに mise を使う方がシンプル
mise use go@1.22               # Go 1.22 をインストール & 設定

# .mise.toml
# [tools]
# go = "1.22"

# go.mod の go ディレクティブとの整合性を確認
# go.mod:
# module example.com/myproject
# go 1.22
```

---

## 7. Java バージョン管理 (SDKMAN / mise)

### 7.1 SDKMAN

```bash
# ─── インストール ───
curl -s "https://get.sdkman.io" | bash

# ─── 基本操作 ───
sdk list java                   # 利用可能バージョン一覧
sdk install java 21.0.2-tem    # Eclipse Temurin JDK 21
sdk install java 21.0.2-graal  # GraalVM CE 21
sdk install java 17.0.10-tem   # JDK 17 (LTS)
sdk use java 21.0.2-tem        # 現在のシェルで切替
sdk default java 21.0.2-tem    # デフォルト設定
sdk current java                # 現在のバージョン

# ─── .sdkmanrc でプロジェクト固定 ───
cat << 'EOF' > .sdkmanrc
java=21.0.2-tem
gradle=8.5
maven=3.9.6
EOF

sdk env                         # .sdkmanrc の設定を適用
sdk env install                 # .sdkmanrc のツールをインストール

# ─── 自動切替の有効化 ───
sdk config
# sdkman_auto_env=true に設定
```

### 7.2 mise による Java 管理

```bash
# SDKMAN の代替として mise を使う
mise use java@temurin-21       # Temurin JDK 21
mise use java@corretto-21      # Amazon Corretto 21
mise use java@graalvm-21       # GraalVM CE 21

# .mise.toml
# [tools]
# java = "temurin-21"

# 利用可能なディストリビューション
mise ls-remote java | head -20
# temurin-21.0.2
# corretto-21.0.2
# graalvm-21.0.2
# zulu-21.0.2
# liberica-21.0.2
```

---

## 8. Ruby バージョン管理 (rbenv)

### 8.1 セットアップ

```bash
# ─── インストール ───
# macOS
brew install rbenv ruby-build

# Linux
git clone https://github.com/rbenv/rbenv.git ~/.rbenv
git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build

# ─── シェル設定 (~/.zshrc) ───
eval "$(rbenv init - zsh)"

# ─── 基本操作 ───
rbenv install --list           # 利用可能バージョン
rbenv install 3.3.0            # インストール
rbenv global 3.3.0             # グローバルデフォルト
rbenv local 3.3.0              # プロジェクト固定 (.ruby-version)
rbenv versions                 # インストール済み一覧
rbenv rehash                   # shim の再構築

# ─── mise による代替（推奨） ───
mise use ruby@3.3              # Ruby 3.3 をインストール & 設定
```

---

## 9. チーム運用のベストプラクティス

### 9.1 プロジェクトテンプレート

```bash
# プロジェクトルートに配置するファイル群
my-project/
├── .node-version          # Node.js バージョン (fnm/nvm対応)
├── .nvmrc                 # nvm 互換 (= .node-version と同じ値)
├── .python-version        # pyenv 用
├── .mise.toml             # mise 用 (統合)
├── .mise.local.toml       # ローカル設定 (.gitignore に追加)
├── rust-toolchain.toml    # Rust 用
├── .go-version            # Go 用 (goenv)
├── .ruby-version          # Ruby 用 (rbenv)
├── .sdkmanrc              # Java 用 (SDKMAN)
└── package.json           # volta の場合はここに記述
```

### 9.2 バージョンファイルの同期スクリプト

```bash
#!/usr/bin/env bash
# scripts/sync-versions.sh
# 各ツールのバージョンファイルを統一管理

set -euo pipefail

# 定義ファイルから読み込み
NODE_VERSION="20.11.0"
PYTHON_VERSION="3.12.3"

# Node.js
echo "$NODE_VERSION" > .node-version
echo "$NODE_VERSION" > .nvmrc

# Python
echo "$PYTHON_VERSION" > .python-version

# mise
cat > .mise.toml << EOF
[tools]
node = "$NODE_VERSION"
python = "$PYTHON_VERSION"
EOF

# package.json の engines フィールドを更新
# (jq が必要)
if command -v jq &> /dev/null && [ -f package.json ]; then
  jq --arg node "$NODE_VERSION" \
     '.engines.node = ">=" + ($node | split(".") | .[0] + ".0.0")' \
     package.json > package.json.tmp && mv package.json.tmp package.json
fi

echo "バージョンファイルを同期しました"
echo "  Node.js: $NODE_VERSION"
echo "  Python:  $PYTHON_VERSION"
```

### 9.3 新メンバーオンボーディングスクリプト

```bash
#!/usr/bin/env bash
# scripts/setup-dev.sh
# 新メンバーが最初に実行するスクリプト

set -euo pipefail

echo "=== 開発環境セットアップを開始します ==="

# ─── バージョンマネージャーの確認 ───
check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo "❌ $1 が見つかりません"
    echo "   インストール: $2"
    return 1
  else
    echo "✅ $1 $(eval "$1 --version 2>&1 | head -1")"
  fi
}

echo ""
echo "--- ツール確認 ---"
check_command fnm "brew install fnm" || MISSING=true
check_command pyenv "brew install pyenv" || MISSING=true

if [ "${MISSING:-}" = "true" ]; then
  echo ""
  echo "不足しているツールをインストールしてから再実行してください"
  exit 1
fi

# ─── Node.js バージョンのインストール ───
echo ""
echo "--- Node.js セットアップ ---"
if [ -f .node-version ]; then
  NODE_VER=$(cat .node-version)
  echo "  .node-version: $NODE_VER"
  fnm install "$NODE_VER"
  fnm use "$NODE_VER"
  echo "  ✅ Node.js $(node --version) を使用中"
fi

# ─── Python バージョンのインストール ───
echo ""
echo "--- Python セットアップ ---"
if [ -f .python-version ]; then
  PYTHON_VER=$(cat .python-version)
  echo "  .python-version: $PYTHON_VER"
  pyenv install -s "$PYTHON_VER"
  pyenv local "$PYTHON_VER"
  echo "  ✅ Python $(python --version) を使用中"
fi

# ─── 依存のインストール ───
echo ""
echo "--- 依存インストール ---"
if [ -f package.json ]; then
  echo "  npm ci を実行中..."
  npm ci
  echo "  ✅ Node.js 依存インストール完了"
fi

if [ -f requirements.txt ]; then
  echo "  pip install を実行中..."
  pip install -r requirements.txt
  echo "  ✅ Python 依存インストール完了"
fi

echo ""
echo "=== セットアップ完了 ==="
```

### 9.4 CI との統一

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # .node-version / .nvmrc を自動検出
      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'npm'  # npm キャッシュを有効化

      # .python-version を自動検出
      - uses: actions/setup-python@v5
        with:
          python-version-file: '.python-version'
          cache: 'pip'  # pip キャッシュを有効化

      - run: npm ci
      - run: npm test
```

### 9.4.1 CI での Rust セットアップ

```yaml
# .github/workflows/rust-ci.yml
name: Rust CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # rust-toolchain.toml を自動検出
      - uses: dtolnay/rust-toolchain@stable
        # rust-toolchain.toml がある場合は自動的に読み込まれる

      # Rust のビルドキャッシュ
      - uses: Swatinem/rust-cache@v2
        with:
          cache-on-failure: true

      - run: cargo test
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check
```

### 9.4.2 CI での mise セットアップ

```yaml
# .github/workflows/mise-ci.yml
name: CI with mise
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # mise で全ツールを一括セットアップ
      - uses: jdx/mise-action@v2
        with:
          experimental: true

      # mise が .mise.toml を読んで全ツールをインストール
      - run: node --version
      - run: python --version
      - run: npm ci
      - run: npm test
```

### 9.5 Docker でのバージョン統一

```dockerfile
# Dockerfile
# .node-version の値をビルド引数として受け取る

ARG NODE_VERSION=20.11.0
FROM node:${NODE_VERSION}-slim

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --production
COPY . .
CMD ["node", "server.js"]
```

```bash
# docker-compose.yml から .node-version を読む
# docker-compose.yml
# services:
#   app:
#     build:
#       context: .
#       args:
#         NODE_VERSION: ${NODE_VERSION:-20.11.0}

# ビルド時に .node-version を参照
NODE_VERSION=$(cat .node-version) docker compose build
```

### 9.6 Renovate / Dependabot でのバージョン更新自動化

```json
// renovate.json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:base"],
  "regexManagers": [
    {
      "fileMatch": ["^\\.node-version$"],
      "matchStrings": ["(?<currentValue>\\d+\\.\\d+\\.\\d+)"],
      "depNameTemplate": "node",
      "datasourceTemplate": "node-version",
      "versioningTemplate": "node"
    },
    {
      "fileMatch": ["^\\.python-version$"],
      "matchStrings": ["(?<currentValue>\\d+\\.\\d+\\.\\d+)"],
      "depNameTemplate": "python",
      "datasourceTemplate": "docker",
      "packageNameTemplate": "python"
    },
    {
      "fileMatch": ["^rust-toolchain\\.toml$"],
      "matchStrings": ["channel\\s*=\\s*\"(?<currentValue>[^\"]+)\""],
      "depNameTemplate": "rust",
      "datasourceTemplate": "docker",
      "packageNameTemplate": "rust"
    }
  ]
}
```

---

## 10. バージョンマネージャー間の移行

### 10.1 nvm → fnm 移行チェックリスト

```
nvm → fnm 移行チェックリスト:

  ☐ 1. 現在の nvm バージョン一覧を記録
       nvm ls > ~/nvm-versions-backup.txt

  ☐ 2. グローバルパッケージの一覧を記録
       nvm use default
       npm list -g --depth=0 > ~/global-packages-backup.txt

  ☐ 3. fnm をインストール
       brew install fnm

  ☐ 4. シェル設定を更新 (~/.zshrc)
       # 削除:
       # export NVM_DIR="$HOME/.nvm"
       # [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
       # (自動切替スクリプトも削除)
       # 追加:
       eval "$(fnm env --use-on-cd --shell zsh)"

  ☐ 5. 必要なバージョンを fnm でインストール
       fnm install 18
       fnm install 20
       fnm default 20

  ☐ 6. 既存の .nvmrc は変更不要（fnm が読める）

  ☐ 7. グローバルパッケージを再インストール（必要な場合のみ）
       npm install -g typescript @angular/cli

  ☐ 8. 動作確認
       cd ~/projects/project-a && node --version
       cd ~/projects/project-b && node --version

  ☐ 9. nvm をアンインストール
       rm -rf ~/.nvm
       # ~/.zshrc から残りの NVM 設定を削除

  ☐ 10. チームメンバーに移行を通知
```

### 10.2 asdf → mise 移行チェックリスト

```
asdf → mise 移行チェックリスト:

  ☐ 1. 現在の asdf ツール一覧を記録
       asdf list > ~/asdf-tools-backup.txt

  ☐ 2. mise をインストール
       brew install mise

  ☐ 3. シェル設定を更新 (~/.zshrc)
       # 削除:
       # . $(brew --prefix asdf)/libexec/asdf.sh
       # 追加:
       eval "$(mise activate zsh)"

  ☐ 4. asdf 互換モードを有効化
       mise settings set asdf_compat true

  ☐ 5. .tool-versions はそのまま使える
       # mise は .tool-versions を自動的に読み込む
       # プラグインのインストールは不要

  ☐ 6. 動作確認
       mise ls        # 管理中のツール一覧
       node --version
       python --version

  ☐ 7. (オプション) .tool-versions → .mise.toml に変換
       # .tool-versions:
       # nodejs 20.11.0
       # python 3.12.3
       #
       # → .mise.toml:
       # [tools]
       # node = "20.11.0"
       # python = "3.12.3"

  ☐ 8. asdf をアンインストール
       brew uninstall asdf
       rm -rf ~/.asdf
```

### 10.3 個別ツール → mise への統合移行

```bash
# 複数のバージョンマネージャーを mise 1つに統合する

# Before:
#   fnm (Node.js) + pyenv (Python) + goenv (Go) + rbenv (Ruby)
#
# After:
#   mise (全部)

# ─── 現在のバージョンを確認 ───
echo "Node.js: $(node --version)"
echo "Python: $(python --version)"
echo "Go: $(go version)"
echo "Ruby: $(ruby --version)"

# ─── mise をインストール ───
brew install mise

# ─── シェル設定を一本化 ───
# ~/.zshrc から以下を全て削除:
# eval "$(fnm env --use-on-cd --shell zsh)"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
# eval "$(goenv init -)"
# eval "$(rbenv init - zsh)"

# 代わりに:
# eval "$(mise activate zsh)"

# ─── 必要なバージョンをインストール ───
mise use --global node@20
mise use --global python@3.12
mise use --global go@1.22
mise use --global ruby@3.3

# ─── プロジェクトの .mise.toml を作成 ───
cat << 'EOF' > .mise.toml
[tools]
node = "20.11.0"
python = "3.12.3"
go = "1.22.0"
ruby = "3.3.0"
EOF

# ─── 動作確認 ───
mise ls                       # 全ツール一覧
node --version
python --version
go version
ruby --version

# ─── 古いツールを削除 ───
brew uninstall fnm
brew uninstall pyenv pyenv-virtualenv
rm -rf ~/.goenv
brew uninstall rbenv
```

---

## 11. アンチパターン

### 11.1 システムグローバルに直接インストール

```
❌ アンチパターン: brew install node で直接インストール

問題:
  - プロジェクト間でバージョン切替不可
  - バージョンマネージャーとの競合
  - sudo が必要になるケース発生

✅ 正しいアプローチ:
  - バージョンマネージャー経由でインストール
  - Homebrew の node/python は削除
  - brew uninstall node python
```

### 11.2 バージョン指定ファイルをコミットしない

```
❌ アンチパターン: .node-version を .gitignore に入れる

問題:
  - チームメンバーが異なるバージョンを使用
  - CI と開発環境の不一致
  - "僕の環境では動く" 問題の再発

✅ 正しいアプローチ:
  - .node-version, .python-version は必ずコミット
  - package.json の engines フィールドも設定
  - CI で同じバージョン指定ファイルを参照
```

### 11.3 メジャーバージョンのみ指定

```
❌ アンチパターン: .node-version に "20" とだけ書く

問題:
  - 開発者 A は 20.10.0 を使用
  - 開発者 B は 20.11.0 を使用
  - パッチバージョンの違いでの微妙なバグが発生
  (例: TLS 関連の挙動変更、V8 エンジンの最適化差異)

✅ 正しいアプローチ:
  - パッチバージョンまで指定: "20.11.0"
  - engines フィールドは範囲指定: ">=20.11.0 <21"
  - 本番環境と完全一致させる
```

### 11.4 複数のバージョンマネージャーを混在させる

```
❌ アンチパターン: fnm と nvm を同時に使う

問題:
  - PATH の優先順位が不定
  - どちらの node が使われるか予測不能
  - シェル起動時間が倍増
  - デバッグが困難

✅ 正しいアプローチ:
  - チーム全体で1つのツールに統一
  - 移行期間は短く設定（2週間以内）
  - 古いツールは完全に削除

確認方法:
  which -a node
  # 1つの結果のみ表示されるべき
  # 複数表示された場合は競合している
```

### 11.5 バージョンマネージャーを更新しない

```
❌ アンチパターン: インストール後にバージョンマネージャー自体を放置

問題:
  - 新しいランタイムバージョンをインストールできない
  - セキュリティ修正が適用されない
  - 最新 OS との互換性問題

✅ 正しいアプローチ:
  定期的な更新スケジュール:
  - Homebrew: brew upgrade fnm mise（月1回）
  - rustup: rustup self update（自動）
  - nvm: nvm のインストールスクリプトを再実行
  - mise: mise self-update（組み込み機能）
```

---

## 12. トラブルシューティング

### 12.1 "command not found" エラー

```bash
# ─── node が見つからない場合 ───

# 1. バージョンマネージャーがロードされているか確認
which fnm        # fnm 自体のパスが返るか
fnm current      # 現在のバージョンが表示されるか

# 2. シェル設定が正しいか確認
cat ~/.zshrc | grep fnm
# eval "$(fnm env --use-on-cd --shell zsh)" が含まれているか

# 3. fnm にバージョンがインストールされているか
fnm list
# 何も表示されない場合:
fnm install --lts

# 4. 新しいシェルを開いて確認
exec zsh
node --version

# ─── pyenv の python が見つからない場合 ───

# 1. pyenv のシム確認
which python
# ~/.pyenv/shims/python が返るべき

# 2. shim の再構築
pyenv rehash

# 3. バージョンが設定されているか確認
pyenv version
# "system" と表示される場合は未設定
pyenv global 3.12.3
```

### 12.2 バージョンが切り替わらない

```bash
# ─── cd しても自動切替が動かない場合 ───

# 1. fnm の --use-on-cd が有効か確認
fnm env | grep "FNM_VERSION_FILE_STRATEGY"
# "recursive" が推奨

# 2. .node-version ファイルの内容確認
cat .node-version
# 改行コードに注意（LF であるべき、CRLF だと問題になることがある）
file .node-version
# "ASCII text" と表示されるべき

# 3. バージョンが実際にインストールされているか
fnm list
# .node-version に書かれたバージョンが一覧にあるか

# 4. 手動切替は動くか
fnm use
# エラーが出る場合は fnm install を先に実行

# ─── Volta で切り替わらない場合 ───
# package.json の volta フィールドを確認
cat package.json | jq '.volta'
# null の場合は volta pin node@20 を実行
```

### 12.3 PATH の競合解決

```bash
# ─── 複数のバージョンマネージャーが競合する場合 ───

# 1. 全ての node の場所を確認
which -a node
type -a node

# 期待される出力（fnm の場合）:
# /Users/username/.fnm/node-versions/v20.11.0/installation/bin/node

# 問題のある出力例:
# /opt/homebrew/bin/node          ← brew install node の残骸
# /Users/username/.nvm/versions/node/v20.11.0/bin/node  ← nvm の残骸
# /Users/username/.fnm/node-versions/v20.11.0/installation/bin/node

# 2. 不要なインストールを削除
brew uninstall node              # Homebrew の node を削除
rm -rf ~/.nvm                    # nvm を完全削除

# 3. PATH の順序を確認
echo $PATH | tr ':' '\n' | head -20
# バージョンマネージャーのパスが /usr/local/bin より前にあるべき

# 4. シェル設定で不要なパスが追加されていないか確認
grep -n 'PATH' ~/.zshrc
grep -n 'PATH' ~/.zprofile
grep -n 'PATH' ~/.bash_profile
```

### 12.4 pyenv のビルドが失敗する

```bash
# ─── macOS Sonoma/Sequoia でのビルドエラー ───

# "Build failed" の場合のデバッグ手順:

# 1. Xcode Command Line Tools を更新
xcode-select --install
# 既にインストール済みの場合:
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install

# 2. ビルド依存を再インストール
brew reinstall openssl readline sqlite3 xz zlib tcl-tk

# 3. 環境変数を設定してビルド
LDFLAGS="-L$(brew --prefix openssl@3)/lib \
  -L$(brew --prefix readline)/lib \
  -L$(brew --prefix sqlite3)/lib \
  -L$(brew --prefix zlib)/lib \
  -L$(brew --prefix xz)/lib" \
CPPFLAGS="-I$(brew --prefix openssl@3)/include \
  -I$(brew --prefix readline)/include \
  -I$(brew --prefix sqlite3)/include \
  -I$(brew --prefix zlib)/include \
  -I$(brew --prefix xz)/include" \
PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig:\
$(brew --prefix readline)/lib/pkgconfig:\
$(brew --prefix sqlite3)/lib/pkgconfig:\
$(brew --prefix zlib)/lib/pkgconfig:\
$(brew --prefix xz)/lib/pkgconfig" \
pyenv install 3.12.3

# 4. それでもダメなら uv を代替として検討
uv python install 3.12  # プリビルドバイナリなのでビルドエラーなし
```

---

## 13. FAQ

### Q1: fnm と nvm、どちらを選ぶべき？

**A:** 新規プロジェクトなら fnm を推奨。理由は以下の通り。
- Rust 製で起動が 40倍以上高速（シェル起動時間に影響）
- `.nvmrc` 互換なので nvm からの移行が容易
- `--use-on-cd` でディレクトリ移動時の自動切替が標準機能
- クロスプラットフォーム対応（Windows ネイティブ含む）
- Corepack サポートが組み込み

nvm はエコシステムが最も成熟しているが、速度面で fnm に劣る。

### Q2: mise は nvm/pyenv を置き換えられる？

**A:** はい、mise は Node.js、Python、Go、Rust、Terraform など多くのツールを統一管理できる。既存の `.nvmrc`、`.python-version`、`.tool-versions` (asdf形式) を読めるので移行も容易。ただし、pyenv の virtualenv 連携や nvm 固有のスクリプトに依存している場合は段階的に移行するのが安全。

### Q3: .node-version と .nvmrc の違いは？

**A:** 内容は同じ（バージョン番号を1行書くだけ）。fnm と nvm は両方読める。Volta は package.json の volta フィールドを使う。チームで統一するなら `.node-version` が推奨（fnm のデフォルト、GitHub Actions の setup-node も対応）。

### Q4: uv で Python バージョン管理するのと pyenv、どちらが良い？

**A:** 2025年以降の新規プロジェクトでは uv を推奨。理由は以下の通り。
- プリビルドバイナリのダウンロードなのでビルドエラーが発生しない
- pyenv のビルド依存（openssl, readline 等）のインストールが不要
- パッケージ管理と Python バージョン管理を uv 1つで完結できる
- インストールが数秒で完了（pyenv は数分かかることがある）

ただし、pyenv が必要なケースもある。
- 特殊なビルドオプション（デバッグビルド、PGO 最適化等）が必要
- CPython 以外の実装（PyPy 等）を使う必要がある場合

### Q5: Volta と fnm の使い分けは？

**A:** 判断基準は以下の通り。

Volta が適しているケース:
- npm/yarn/pnpm のバージョンも厳密に固定したい
- package.json に全ての設定を集約したい
- シェルフック不要の仕組みを好む（シェル起動影響ゼロ）

fnm が適しているケース:
- Node.js のバージョンだけ管理できれば十分
- .nvmrc との互換性が重要（nvm からの移行）
- 軽量・高速なツールを好む

### Q6: チームで異なるバージョンマネージャーを使っていても問題ない？

**A:** バージョンファイルが統一されていれば実用上は問題ない。例えば `.node-version` ファイルは fnm でも nvm でも mise でも読める。ただし、以下の点に注意が必要。
- バージョンファイルのフォーマットに互換性がない組み合わせがある（例: Volta は .nvmrc を読まない）
- トラブルシューティング時の統一的なサポートが困難
- 可能であればチーム全体で1つのツールに統一することを推奨

### Q7: Node.js のバージョンアップはどのタイミングで行うべき？

**A:** 以下のスケジュールを推奨。
1. **新 LTS リリース直後（10月）**: 検証用ブランチで依存の互換性テスト
2. **リリース後 1-2ヶ月**: 主要ライブラリの対応を確認
3. **リリース後 3ヶ月**: 本番移行を計画・実行
4. **旧 LTS の EOL 3ヶ月前**: 移行を完了

### Q8: バージョンマネージャー自体のバージョンは固定すべき？

**A:** バージョンマネージャー自体は最新版を使うことを推奨。ランタイムのバージョンとは異なり、バージョンマネージャーの更新で破壊的変更が入ることは稀。ただし、CI では `actions/setup-node@v4` のようにメジャーバージョンを固定するのが安全。

---

## 14. まとめ

| 言語 | 推奨ツール | 設定ファイル | 備考 |
|------|-----------|-------------|------|
| Node.js | fnm | `.node-version` | 高速・.nvmrc互換 |
| Node.js (チーム) | Volta | `package.json` | npm/yarn も固定 |
| Python (新規) | uv | `.python-version` | 超高速・ビルド不要 |
| Python (既存) | pyenv | `.python-version` | virtualenv 連携 |
| Rust | rustup | `rust-toolchain.toml` | 公式ツール |
| Go | mise | `.mise.toml` | goenv より統合的 |
| Java | SDKMAN / mise | `.sdkmanrc` / `.mise.toml` | ディストリビューション選択可 |
| Ruby | rbenv / mise | `.ruby-version` / `.mise.toml` | mise 統合推奨 |
| 多言語統合 | mise | `.mise.toml` | 全言語を1ツールで |
| CI | actions/setup-* | 上記ファイルを参照 | 自動検出対応 |

### バージョン管理の5原則

```
1. パッチバージョンまで固定する
   → "20.11.0" であって "20" ではない

2. バージョンファイルは必ずコミットする
   → .node-version, .python-version は .gitignore に入れない

3. CI と開発環境で同じバージョンを使う
   → バージョンファイルを CI でも参照する

4. チーム全体で同じバージョンマネージャーを使う
   → 混在は PATH の競合やサポートの困難さを招く

5. 定期的にバージョンを更新する
   → LTS スケジュールに合わせた計画的な更新
```

---

## 次に読むべきガイド

- [01-package-managers.md](./01-package-managers.md) -- パッケージマネージャーの設定
- [../00-editor-and-tools/01-terminal-setup.md](../00-editor-and-tools/01-terminal-setup.md) -- シェル設定との連携
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) -- チーム標準の設定

---

## 参考文献

1. **fnm (Fast Node Manager)** -- https://github.com/Schniz/fnm -- fnm 公式リポジトリ。ベンチマーク比較あり。
2. **mise documentation** -- https://mise.jdx.dev/ -- mise の公式ドキュメント。全対応ツール一覧。
3. **pyenv** -- https://github.com/pyenv/pyenv -- pyenv 公式。ビルド依存のトラブルシューティングが充実。
4. **Volta** -- https://volta.sh/ -- Volta 公式サイト。package.json 統合の詳細解説。
5. **rustup** -- https://rust-lang.github.io/rustup/ -- rustup 公式ドキュメント。ツールチェーン管理の詳細。
6. **uv** -- https://docs.astral.sh/uv/ -- uv 公式ドキュメント。Python バージョン管理機能の解説。
7. **Node.js Release Schedule** -- https://github.com/nodejs/release -- Node.js のリリーススケジュールと EOL 情報。
8. **SDKMAN** -- https://sdkman.io/ -- SDKMAN 公式サイト。Java ディストリビューションの選択ガイド。
9. **rbenv** -- https://github.com/rbenv/rbenv -- rbenv 公式リポジトリ。Ruby バージョン管理。
10. **goenv** -- https://github.com/go-nv/goenv -- goenv 公式リポジトリ。Go バージョン管理。
