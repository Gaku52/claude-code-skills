# バージョンマネージャー

> プログラミング言語のバージョンをプロジェクト単位で管理し、チーム全体で統一された開発環境を実現するためのガイド。

## この章で学ぶこと

1. nvm / fnm / Volta による Node.js のバージョン管理と使い分け
2. pyenv (Python)、rustup (Rust) の設定と運用方法
3. mise (旧 rtx) を使った統合的なバージョン管理

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

### 1.2 主要ツール比較

| ツール | 対象言語 | 速度 | .nvmrc 互換 | シェル起動影響 | 自動切替 |
|--------|---------|------|------------|-------------|---------|
| nvm | Node.js | 遅い | ネイティブ | 大きい | あり |
| fnm | Node.js | 高速 | あり | 小さい | あり |
| Volta | Node.js (+npm/yarn) | 高速 | 部分的 | なし | あり |
| pyenv | Python | 普通 | - | 中程度 | あり |
| rustup | Rust | 高速 | - | なし | あり |
| mise | 多言語 | 高速 | あり | 小さい | あり |

---

## 2. Node.js バージョン管理

### 2.1 fnm (Fast Node Manager) — 推奨

```bash
# ─── インストール ───
# macOS
brew install fnm
# Linux/macOS (curl)
curl -fsSL https://fnm.vercel.app/install | bash
# Windows
winget install Schniz.fnm

# ─── シェル設定 ───
# ~/.zshrc に追加
eval "$(fnm env --use-on-cd --shell zsh)"
# ~/.config/fish/config.fish に追加
fnm env --use-on-cd --shell fish | source

# ─── 基本操作 ───
fnm list-remote              # 利用可能なバージョン一覧
fnm install 20               # Node.js 20.x 最新をインストール
fnm install --lts             # 最新 LTS をインストール
fnm use 20                   # 現在のシェルで切替
fnm default 20               # デフォルトバージョン設定
fnm list                     # インストール済み一覧
fnm current                  # 現在のバージョン確認

# ─── プロジェクト設定 ───
echo "20" > .node-version    # プロジェクトルートに配置
# → cd でディレクトリに入ると自動切替 (--use-on-cd)
```

### 2.2 nvm (Node Version Manager)

```bash
# ─── インストール ───
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# ─── ~/.zshrc に自動追加される設定 ───
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# ─── 基本操作 ───
nvm install 20               # インストール
nvm use 20                   # 切替
nvm alias default 20         # デフォルト設定
nvm ls                       # 一覧

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

### 2.3 Volta

```bash
# ─── インストール ───
curl https://get.volta.sh | bash

# ─── 基本操作 ───
volta install node@20        # Node.js インストール
volta install npm@10         # npm バージョン固定
volta install yarn@4         # yarn バージョン固定

# ─── プロジェクト固定 (package.json に記録) ───
volta pin node@20
volta pin npm@10

# package.json に自動追記される:
# {
#   "volta": {
#     "node": "20.11.0",
#     "npm": "10.2.4"
#   }
# }
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
              fnm (デフォルト推奨)
```

---

## 3. Python バージョン管理 (pyenv)

### 3.1 セットアップ

```bash
# ─── インストール ───
# macOS
brew install pyenv pyenv-virtualenv

# Linux
curl https://pyenv.run | bash

# ─── シェル設定 (~/.zshrc) ───
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# ─── ビルド依存のインストール (macOS) ───
brew install openssl readline sqlite3 xz zlib tcl-tk

# ─── 基本操作 ───
pyenv install --list | grep '3.12'   # 利用可能バージョン
pyenv install 3.12.3                  # インストール
pyenv global 3.12.3                   # グローバルデフォルト
pyenv local 3.12.3                    # プロジェクト固定 (.python-version)
pyenv versions                        # インストール済み一覧

# ─── 仮想環境 ───
pyenv virtualenv 3.12.3 myproject-env
pyenv activate myproject-env
pyenv deactivate
```

---

## 4. Rust バージョン管理 (rustup)

### 4.1 セットアップ

```bash
# ─── インストール ───
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# ─── 基本操作 ───
rustup show                          # 現在の toolchain
rustup update                        # 全 toolchain 更新
rustup default stable                # デフォルト設定
rustup toolchain install nightly     # nightly インストール

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
```

---

## 5. mise (統合バージョンマネージャー)

### 5.1 セットアップ

```bash
# ─── インストール ───
# macOS
brew install mise
# Linux
curl https://mise.run | sh

# ─── シェル設定 (~/.zshrc) ───
eval "$(mise activate zsh)"

# ─── 基本操作 ───
mise use node@20              # Node.js インストール & 設定
mise use python@3.12          # Python インストール & 設定
mise use go@1.22              # Go インストール & 設定

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
```

---

## 6. チーム運用のベストプラクティス

### 6.1 プロジェクトテンプレート

```bash
# プロジェクトルートに配置するファイル群
my-project/
├── .node-version          # Node.js バージョン (fnm/nvm対応)
├── .nvmrc                 # nvm 互換 (= .node-version と同じ値)
├── .python-version        # pyenv 用
├── .mise.toml             # mise 用 (統合)
├── rust-toolchain.toml    # Rust 用
└── package.json           # volta の場合はここに記述
```

### 6.2 CI との統一

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

      # .python-version を自動検出
      - uses: actions/setup-python@v5
        with:
          python-version-file: '.python-version'

      - run: npm ci
      - run: npm test
```

---

## 7. アンチパターン

### 7.1 システムグローバルに直接インストール

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

### 7.2 バージョン指定ファイルをコミットしない

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

---

## 8. FAQ

### Q1: fnm と nvm、どちらを選ぶべき？

**A:** 新規プロジェクトなら fnm を推奨。理由は以下の通り。
- Rust 製で起動が 40倍以上高速（シェル起動時間に影響）
- `.nvmrc` 互換なので nvm からの移行が容易
- `--use-on-cd` でディレクトリ移動時の自動切替が標準機能
- クロスプラットフォーム対応（Windows ネイティブ含む）

nvm はエコシステムが最も成熟しているが、速度面で fnm に劣る。

### Q2: mise は nvm/pyenv を置き換えられる？

**A:** はい、mise は Node.js、Python、Go、Rust、Terraform など多くのツールを統一管理できる。既存の `.nvmrc`、`.python-version`、`.tool-versions` (asdf形式) を読めるので移行も容易。ただし、pyenv の virtualenv 連携や nvm 固有のスクリプトに依存している場合は段階的に移行するのが安全。

### Q3: .node-version と .nvmrc の違いは？

**A:** 内容は同じ（バージョン番号を1行書くだけ）。fnm と nvm は両方読める。Volta は package.json の volta フィールドを使う。チームで統一するなら `.node-version` が推奨（fnm のデフォルト、GitHub Actions の setup-node も対応）。

---

## 9. まとめ

| 言語 | 推奨ツール | 設定ファイル | 備考 |
|------|-----------|-------------|------|
| Node.js | fnm | `.node-version` | 高速・.nvmrc互換 |
| Node.js (チーム) | Volta | `package.json` | npm/yarn も固定 |
| Python | pyenv | `.python-version` | virtualenv 連携 |
| Rust | rustup | `rust-toolchain.toml` | 公式ツール |
| 多言語統合 | mise | `.mise.toml` | 全言語を1ツールで |
| CI | actions/setup-* | 上記ファイルを参照 | 自動検出対応 |

---

## 次に読むべきガイド

- [01-package-managers.md](./01-package-managers.md) — パッケージマネージャーの設定
- [../00-editor-and-tools/01-terminal-setup.md](../00-editor-and-tools/01-terminal-setup.md) — シェル設定との連携
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) — チーム標準の設定

---

## 参考文献

1. **fnm (Fast Node Manager)** — https://github.com/Schniz/fnm — fnm 公式リポジトリ。ベンチマーク比較あり。
2. **mise documentation** — https://mise.jdx.dev/ — mise の公式ドキュメント。全対応ツール一覧。
3. **pyenv** — https://github.com/pyenv/pyenv — pyenv 公式。ビルド依存のトラブルシューティングが充実。
4. **Volta** — https://volta.sh/ — Volta 公式サイト。package.json 統合の詳細解説。
