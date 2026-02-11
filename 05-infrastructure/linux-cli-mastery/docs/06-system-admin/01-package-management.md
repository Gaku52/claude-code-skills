# パッケージ管理

> パッケージマネージャはソフトウェアのインストール・更新・削除を安全に行う仕組み。

## この章で学ぶこと

- [ ] 主要パッケージマネージャの使い方を理解する
- [ ] パッケージの検索・インストール・更新・削除ができる
- [ ] macOS と Linux のパッケージ管理の違いを理解する

---

## 1. apt（Debian / Ubuntu）

```bash
# パッケージリストの更新
sudo apt update                  # リポジトリ情報を最新化

# インストール
sudo apt install nginx           # インストール
sudo apt install -y nginx        # 確認なしでインストール
sudo apt install nginx=1.24.0-1  # バージョン指定

# 更新
sudo apt upgrade                 # 全パッケージを更新
sudo apt full-upgrade            # 依存関係の変更含む更新
sudo apt update && sudo apt upgrade -y  # 定番パターン

# 削除
sudo apt remove nginx            # パッケージ削除（設定残す）
sudo apt purge nginx             # 設定ファイルごと削除
sudo apt autoremove              # 不要な依存パッケージを削除

# 検索・情報
apt search nginx                 # パッケージ検索
apt show nginx                   # 詳細情報
apt list --installed             # インストール済み一覧
apt list --upgradable            # 更新可能な一覧
dpkg -l | grep nginx             # インストール済みを検索
dpkg -L nginx                    # パッケージのファイル一覧
apt-cache depends nginx          # 依存関係
apt-cache rdepends nginx         # 逆依存関係

# .deb ファイルから直接インストール
sudo dpkg -i package.deb
sudo apt install -f              # 依存関係を解決
```

---

## 2. dnf / yum（RHEL / Fedora / Rocky）

```bash
# dnf（yum の後継）
sudo dnf install nginx           # インストール
sudo dnf remove nginx            # 削除
sudo dnf update                  # 全パッケージ更新
sudo dnf search nginx            # 検索
sudo dnf info nginx              # 詳細情報
sudo dnf list installed          # インストール済み一覧
sudo dnf provides /usr/bin/curl  # ファイルを含むパッケージ

# グループ管理
sudo dnf group list              # グループ一覧
sudo dnf group install "Development Tools"  # 開発ツール一括

# モジュール（RHEL 8+）
sudo dnf module list             # 利用可能なモジュール
sudo dnf module enable nodejs:18 # Node.js 18 を有効化
sudo dnf module install nodejs:18

# .rpm ファイル
sudo dnf install package.rpm
rpm -qa | grep nginx             # インストール済み検索
rpm -ql nginx                    # ファイル一覧
```

---

## 3. Homebrew（macOS / Linux）

```bash
# インストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 基本操作
brew install wget                # CLI ツールのインストール
brew install --cask firefox      # GUI アプリのインストール
brew uninstall wget              # 削除
brew upgrade                     # 全パッケージ更新
brew upgrade wget                # 特定パッケージ更新
brew update                      # Homebrew 自体の更新

# 検索・情報
brew search nginx                # 検索
brew info nginx                  # 詳細情報
brew list                        # インストール済み一覧
brew list --cask                 # GUI アプリ一覧
brew deps nginx                  # 依存関係
brew uses --installed nginx      # 逆依存関係

# メンテナンス
brew cleanup                     # 古いバージョンを削除
brew doctor                      # 問題の診断
brew autoremove                  # 不要な依存を削除

# サービス管理（macOS の launchd と統合）
brew services list               # サービス一覧
brew services start nginx        # 起動
brew services stop nginx         # 停止
brew services restart nginx      # 再起動

# Tap（サードパーティリポジトリ）
brew tap homebrew/cask-fonts     # フォント用Tap追加
brew install --cask font-fira-code

# Bundle（一括管理）
# Brewfile に記述してまとめてインストール
brew bundle dump                 # 現在の状態をBrewfileに出力
brew bundle install              # Brewfileからインストール
```

### Brewfile の例

```ruby
# Brewfile
tap "homebrew/cask"

# CLI tools
brew "git"
brew "gh"
brew "ripgrep"
brew "fd"
brew "bat"
brew "eza"
brew "fzf"
brew "jq"
brew "starship"

# Applications
cask "visual-studio-code"
cask "iterm2"
cask "docker"
cask "firefox"
```

---

## 4. その他のパッケージマネージャ

```bash
# pacman（Arch Linux）
sudo pacman -S nginx             # インストール
sudo pacman -R nginx             # 削除
sudo pacman -Syu                 # 全更新
pacman -Ss nginx                 # 検索
pacman -Qi nginx                 # 情報

# snap（Ubuntu）
sudo snap install code --classic # インストール
snap list                        # 一覧
sudo snap refresh                # 更新

# flatpak
flatpak install flathub org.gimp.GIMP
flatpak list
flatpak update

# nix（宣言的パッケージ管理）
nix-env -iA nixpkgs.nginx       # インストール
nix-env -q                      # 一覧
nix-env -u                      # 更新
nix-env -e nginx                # 削除
```

---

## 5. 言語別パッケージマネージャ

```bash
# Node.js
npm install -g typescript        # グローバル
npm install express              # ローカル（プロジェクト内）
npx create-react-app myapp       # 一時実行

# Python
pip install requests             # インストール
pip install -r requirements.txt  # 一括インストール
pip list                         # 一覧
pip freeze > requirements.txt    # 依存を書き出し
# uvx / pipx: CLIツールのインストールに推奨
pipx install black

# Rust
cargo install ripgrep            # バイナリインストール
rustup update                    # ツールチェーン更新

# Go
go install golang.org/x/tools/gopls@latest

# Ruby
gem install bundler
bundle install                   # Gemfile から一括
```

---

## 6. 実践パターン

```bash
# パターン1: サーバーの初期セットアップ
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    git curl wget \
    build-essential \
    nginx \
    postgresql \
    redis-server

# パターン2: 開発マシンのセットアップ自動化（macOS）
# setup.sh
#!/bin/bash
# Homebrew
command -v brew >/dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Brewfile からインストール
brew bundle install --file=Brewfile

# パターン3: パッケージのセキュリティ確認
# Ubuntu
sudo apt list --upgradable 2>/dev/null | grep -i security
# 自動セキュリティ更新
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# パターン4: どのパッケージがファイルを提供しているか調べる
# Ubuntu
dpkg -S /usr/bin/curl            # curl パッケージ
apt-file search /usr/bin/curl    # apt-file を使用

# macOS
brew which-formula curl          # Homebrew
```

---

## まとめ

| ディストリビューション | マネージャ | インストール | 更新 |
|----------------------|-----------|------------|------|
| Ubuntu/Debian | apt | apt install pkg | apt upgrade |
| RHEL/Fedora | dnf | dnf install pkg | dnf update |
| Arch Linux | pacman | pacman -S pkg | pacman -Syu |
| macOS | brew | brew install pkg | brew upgrade |

---

## 次に読むべきガイド
→ [[../07-advanced/00-tmux-screen.md]] — ターミナルマルチプレクサ

---

## 参考文献
1. "APT User's Guide." Debian Documentation.
2. "Homebrew Documentation." brew.sh.
