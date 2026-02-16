# パッケージ管理

> パッケージマネージャはソフトウェアのインストール・更新・削除を安全に行う仕組み。

## この章で学ぶこと

- [ ] 主要パッケージマネージャの使い方を理解する
- [ ] パッケージの検索・インストール・更新・削除ができる
- [ ] macOS と Linux のパッケージ管理の違いを理解する
- [ ] リポジトリの追加・管理ができる
- [ ] パッケージのセキュリティ対策を理解する
- [ ] 環境の自動セットアップスクリプトが書ける

---

## 1. apt（Debian / Ubuntu）

### 1.1 基本操作

```bash
# パッケージリストの更新
sudo apt update                  # リポジトリ情報を最新化

# インストール
sudo apt install nginx           # インストール
sudo apt install -y nginx        # 確認なしでインストール
sudo apt install nginx=1.24.0-1  # バージョン指定
sudo apt install nginx curl wget # 複数パッケージ

# 更新
sudo apt upgrade                 # 全パッケージを更新
sudo apt full-upgrade            # 依存関係の変更含む更新
sudo apt update && sudo apt upgrade -y  # 定番パターン

# 削除
sudo apt remove nginx            # パッケージ削除（設定残す）
sudo apt purge nginx             # 設定ファイルごと削除
sudo apt autoremove              # 不要な依存パッケージを削除
sudo apt autoremove --purge      # 不要パッケージを設定ごと削除
```

### 1.2 検索・情報表示

```bash
# 検索・情報
apt search nginx                 # パッケージ検索
apt show nginx                   # 詳細情報
apt list --installed             # インストール済み一覧
apt list --upgradable            # 更新可能な一覧
dpkg -l | grep nginx             # インストール済みを検索
dpkg -L nginx                    # パッケージのファイル一覧
dpkg -S /usr/bin/curl            # ファイルを含むパッケージ
apt-cache depends nginx          # 依存関係
apt-cache rdepends nginx         # 逆依存関係

# パッケージの変更履歴
apt changelog nginx

# パッケージポリシー（バージョンと優先度）
apt-cache policy nginx

# 出力例:
# nginx:
#   Installed: 1.24.0-1ubuntu1
#   Candidate: 1.24.0-1ubuntu1
#   Version table:
#  *** 1.24.0-1ubuntu1 500
#         500 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages
#         100 /var/lib/dpkg/status
```

### 1.3 deb パッケージの直接操作

```bash
# .deb ファイルから直接インストール
sudo dpkg -i package.deb
sudo apt install -f              # 依存関係を解決

# より安全な方法（依存関係も自動解決）
sudo apt install ./package.deb

# deb パッケージの中身を確認
dpkg-deb -c package.deb         # ファイル一覧
dpkg-deb -I package.deb         # パッケージ情報
dpkg-deb -x package.deb /tmp/extract  # 展開

# パッケージの再構成
sudo dpkg --configure -a         # 未構成パッケージの構成
sudo dpkg --reconfigure tzdata   # パッケージの再構成
```

### 1.4 リポジトリの管理

```bash
# リポジトリの確認
cat /etc/apt/sources.list
ls /etc/apt/sources.list.d/

# PPA の追加（Ubuntu）
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# PPA の削除
sudo add-apt-repository --remove ppa:deadsnakes/ppa

# サードパーティリポジトリの追加（新しい方法、Ubuntu 22.04+）
# 1. GPGキーのダウンロード
curl -fsSL https://packages.example.com/gpg.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/example-archive-keyring.gpg

# 2. リポジトリの追加
echo "deb [signed-by=/usr/share/keyrings/example-archive-keyring.gpg] \
    https://packages.example.com/apt stable main" | \
    sudo tee /etc/apt/sources.list.d/example.list

# 3. 更新
sudo apt update

# Docker リポジトリの追加例
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# Node.js リポジトリの追加例（NodeSource）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
```

### 1.5 パッケージの固定（ホールド）

```bash
# 特定パッケージのバージョンを固定（更新を防止）
sudo apt-mark hold nginx
sudo apt-mark hold linux-image-generic  # カーネル更新を防止

# 固定を解除
sudo apt-mark unhold nginx

# 固定されたパッケージの確認
apt-mark showhold

# dpkg を使った固定
echo "nginx hold" | sudo dpkg --set-selections

# 固定状態の確認
dpkg --get-selections | grep hold
```

### 1.6 apt の自動更新設定

```bash
# unattended-upgrades: セキュリティ更新の自動適用
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# 設定ファイル: /etc/apt/apt.conf.d/50unattended-upgrades
# Unattended-Upgrade::Allowed-Origins {
#     "${distro_id}:${distro_codename}";
#     "${distro_id}:${distro_codename}-security";
# };
# Unattended-Upgrade::Mail "admin@example.com";
# Unattended-Upgrade::Automatic-Reboot "false";
# Unattended-Upgrade::Automatic-Reboot-Time "02:00";

# 自動更新のテスト
sudo unattended-upgrade --dry-run --debug

# 自動更新ログの確認
cat /var/log/unattended-upgrades/unattended-upgrades.log
```

### 1.7 キャッシュ管理

```bash
# apt キャッシュの管理
sudo apt clean                   # ダウンロード済みパッケージを全削除
sudo apt autoclean               # 古いバージョンのキャッシュのみ削除

# キャッシュの場所
ls /var/cache/apt/archives/

# キャッシュサイズの確認
du -sh /var/cache/apt/archives/

# パッケージリストの再構築
sudo apt update --fix-missing
```

---

## 2. dnf / yum（RHEL / Fedora / Rocky）

### 2.1 基本操作

```bash
# dnf（yum の後継）
sudo dnf install nginx           # インストール
sudo dnf install -y nginx        # 確認なし
sudo dnf remove nginx            # 削除
sudo dnf update                  # 全パッケージ更新
sudo dnf upgrade                 # update と同義（dnf では）
sudo dnf check-update            # 更新可能なパッケージの確認

# 検索・情報
sudo dnf search nginx            # 検索
sudo dnf info nginx              # 詳細情報
sudo dnf list installed          # インストール済み一覧
sudo dnf list available          # 利用可能なパッケージ
sudo dnf provides /usr/bin/curl  # ファイルを含むパッケージ
sudo dnf repoquery --whatrequires nginx  # 逆依存関係
```

### 2.2 グループとモジュール管理

```bash
# グループ管理
sudo dnf group list              # グループ一覧
sudo dnf group info "Development Tools"  # グループの詳細
sudo dnf group install "Development Tools"  # 開発ツール一括
sudo dnf group remove "Development Tools"   # グループ削除

# モジュール（RHEL 8+ / Fedora）
sudo dnf module list             # 利用可能なモジュール
sudo dnf module list nodejs      # 特定モジュールのストリーム一覧
sudo dnf module enable nodejs:20 # Node.js 20 を有効化
sudo dnf module install nodejs:20
sudo dnf module disable nodejs   # モジュール無効化
sudo dnf module reset nodejs     # モジュールのリセット

# モジュールストリームの切り替え
sudo dnf module reset nodejs
sudo dnf module enable nodejs:22
sudo dnf module install nodejs:22
```

### 2.3 リポジトリの管理

```bash
# リポジトリの確認
dnf repolist                     # 有効なリポジトリ
dnf repolist all                 # 全リポジトリ
dnf repoinfo                     # リポジトリ詳細

# リポジトリの有効化/無効化
sudo dnf config-manager --set-enabled powertools
sudo dnf config-manager --set-disabled powertools

# サードパーティリポジトリの追加
sudo dnf config-manager --add-repo https://packages.example.com/repo

# EPEL リポジトリの追加（RHEL/Rocky/AlmaLinux）
sudo dnf install epel-release

# RPM Fusion リポジトリの追加（Fedora）
sudo dnf install \
    https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
    https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
```

### 2.4 rpm の直接操作

```bash
# .rpm ファイル
sudo dnf install package.rpm     # 推奨（依存関係自動解決）
sudo rpm -ivh package.rpm        # rpm 直接（依存関係手動）

# rpm での情報表示
rpm -qa | grep nginx             # インストール済み検索
rpm -ql nginx                    # ファイル一覧
rpm -qi nginx                    # パッケージ情報
rpm -qf /usr/bin/curl            # ファイルの所有パッケージ
rpm -qp package.rpm              # 未インストールパッケージの情報

# GPGキーの管理
rpm --import https://packages.example.com/gpg.key
rpm -qa gpg-pubkey*              # インポート済みキー一覧
```

### 2.5 dnf の履歴管理

```bash
# dnf のトランザクション履歴
dnf history                      # 履歴一覧
dnf history info 15              # 特定トランザクションの詳細
dnf history undo 15              # 特定トランザクションの取り消し
dnf history rollback 15          # 特定時点まで巻き戻し

# 最近の操作のログ
cat /var/log/dnf.log
```

### 2.6 パッケージの固定

```bash
# dnf でのバージョン固定
sudo dnf install dnf-plugin-versionlock
sudo dnf versionlock add nginx
sudo dnf versionlock list
sudo dnf versionlock delete nginx
```

---

## 3. Homebrew（macOS / Linux）

### 3.1 基本操作

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
brew deps --tree nginx           # 依存ツリー
brew uses --installed nginx      # 逆依存関係
brew outdated                    # 更新可能な一覧

# メンテナンス
brew cleanup                     # 古いバージョンを削除
brew cleanup -n                  # 削除対象の確認（dry-run）
brew cleanup --prune=30          # 30日以上前のキャッシュを削除
brew doctor                      # 問題の診断
brew autoremove                  # 不要な依存を削除
brew missing                     # 不足している依存
```

### 3.2 Homebrew の高度な操作

```bash
# バージョン管理
brew list --versions nginx       # インストール済みバージョン
brew pin nginx                   # バージョン固定
brew unpin nginx                 # 固定解除
brew list --pinned               # 固定されたパッケージ

# 特定バージョンのインストール
brew install nginx@1.24          # 特定バージョン（利用可能な場合）
brew install --HEAD nginx        # 開発版（HEAD）

# パッケージの詳細操作
brew link nginx                  # シンボリックリンク作成
brew unlink nginx                # シンボリックリンク削除
brew link --overwrite nginx      # 強制リンク
brew --prefix nginx              # インストール先の確認

# 構成情報
brew config                      # Homebrew の設定情報
brew --prefix                    # Homebrew のインストール先
brew --cellar                    # Cellar の場所
brew --cache                     # キャッシュの場所
```

### 3.3 サービス管理（macOS の launchd と統合）

```bash
# サービス管理
brew services list               # サービス一覧
brew services start nginx        # 起動（自動起動も設定）
brew services stop nginx         # 停止
brew services restart nginx      # 再起動
brew services run nginx          # 起動のみ（自動起動なし）
brew services info nginx         # サービス情報

# plist ファイルの場所
ls ~/Library/LaunchAgents/       # ユーザーサービス
ls /Library/LaunchDaemons/       # システムサービス
```

### 3.4 Tap（サードパーティリポジトリ）

```bash
# Tap の管理
brew tap                         # 追加済みTap一覧
brew tap homebrew/cask-fonts     # フォント用Tap追加
brew tap homebrew/cask-versions  # 旧バージョンのcask
brew tap user/repo               # カスタムTap
brew untap homebrew/cask-fonts   # Tap削除

# 特定のTapからインストール
brew install homebrew/cask-fonts/font-fira-code
```

### 3.5 Bundle（一括管理）

```bash
# Bundle（一括管理）
# Brewfile に記述してまとめてインストール
brew bundle dump                 # 現在の状態をBrewfileに出力
brew bundle dump --force         # 上書き
brew bundle install              # Brewfileからインストール
brew bundle check                # 全てインストール済みか確認
brew bundle cleanup              # Brewfileにないパッケージを削除
brew bundle cleanup --force      # 確認なしで削除
brew bundle list                 # Brewfileの内容を表示
```

### 3.6 Brewfile の例

```ruby
# Brewfile

# Tap
tap "homebrew/cask"
tap "homebrew/cask-fonts"

# CLI tools - 基本
brew "git"
brew "gh"
brew "curl"
brew "wget"

# CLI tools - モダン代替
brew "ripgrep"                  # grep の代替
brew "fd"                       # find の代替
brew "bat"                      # cat の代替
brew "eza"                      # ls の代替
brew "fzf"                      # ファジーファインダー
brew "zoxide"                   # cd の代替
brew "delta"                    # diff の代替
brew "dust"                     # du の代替
brew "duf"                      # df の代替
brew "procs"                    # ps の代替
brew "bottom"                   # top の代替
brew "hyperfine"                # ベンチマーク

# CLI tools - 開発
brew "jq"                       # JSON処理
brew "yq"                       # YAML処理
brew "starship"                 # プロンプト
brew "tmux"                     # ターミナルマルチプレクサ
brew "shellcheck"               # シェルスクリプトリンター

# CLI tools - インフラ
brew "awscli"
brew "terraform"
brew "ansible"
brew "kubectl"
brew "helm"

# 言語・ランタイム
brew "node"
brew "python@3.12"
brew "go"
brew "rust"

# Applications（GUI）
cask "visual-studio-code"
cask "iterm2"
cask "docker"
cask "firefox"
cask "google-chrome"
cask "slack"
cask "1password"
cask "rectangle"                # ウィンドウ管理

# Fonts
cask "font-fira-code-nerd-font"
cask "font-jetbrains-mono-nerd-font"

# Mac App Store（mas コマンドが必要）
# mas "Xcode", id: 497799835
# mas "Keynote", id: 409183694
```

### 3.7 Homebrew on Linux

```bash
# Linux での Homebrew（Linuxbrew）
# インストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# パスの設定（~/.bashrc または ~/.zshrc に追加）
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# Linux での利点:
# - 最新バージョンのツールを root なしでインストール可能
# - ディストリビューションに依存しない
# - macOS と同じ Brewfile を共有可能

# 注意点:
# - /home/linuxbrew/.linuxbrew/ にインストールされる
# - glibc 2.13+ が必要
# - ビルドツール（gcc, make）が事前に必要
sudo apt install build-essential curl file git
```

---

## 4. pacman（Arch Linux）

### 4.1 基本操作

```bash
# pacman（Arch Linux / Manjaro）
sudo pacman -S nginx             # インストール
sudo pacman -R nginx             # 削除
sudo pacman -Rs nginx            # 依存関係ごと削除
sudo pacman -Rns nginx           # 設定ファイルも含めて完全削除
sudo pacman -Syu                 # 全更新（同期 + 更新）
sudo pacman -Syy                 # データベース強制更新
sudo pacman -Ss nginx            # 検索
sudo pacman -Si nginx            # リポジトリのパッケージ情報
sudo pacman -Qi nginx            # インストール済みパッケージ情報
sudo pacman -Ql nginx            # ファイル一覧
sudo pacman -Qo /usr/bin/curl    # ファイルの所有パッケージ
sudo pacman -Qe                  # 明示的にインストールしたパッケージ
sudo pacman -Qd                  # 依存関係でインストールされたパッケージ
sudo pacman -Qdt                 # 孤立パッケージ（不要な依存）

# キャッシュの管理
sudo pacman -Sc                  # 古いキャッシュを削除
sudo pacman -Scc                 # 全キャッシュを削除

# AUR（Arch User Repository）ヘルパー
# yay のインストール
git clone https://aur.archlinux.org/yay.git
cd yay && makepkg -si

# yay の使い方（pacman と同じ構文）
yay -S google-chrome             # AUR からインストール
yay -Syu                         # 全更新（公式 + AUR）
yay -Ss keyword                  # 検索（公式 + AUR）
```

### 4.2 pacman のフラグ早見表

```bash
# pacman のフラグ体系:
# -S: Sync（リポジトリ操作）
#   -S pkg    → インストール
#   -Ss       → 検索
#   -Si       → 情報表示
#   -Sy       → データベース更新
#   -Su       → アップグレード
#   -Syu      → 更新 + アップグレード
#   -Sc       → キャッシュ削除

# -R: Remove（削除操作）
#   -R pkg    → 削除
#   -Rs       → 依存関係も削除
#   -Rn       → 設定ファイルも削除
#   -Rns      → 完全削除

# -Q: Query（クエリ操作）
#   -Q        → インストール済み一覧
#   -Qs       → 検索
#   -Qi       → 情報表示
#   -Ql       → ファイル一覧
#   -Qo file  → 所有パッケージ
#   -Qe       → 明示インストール
#   -Qd       → 依存インストール
#   -Qdt      → 孤立パッケージ

# -F: File（ファイル検索）
#   -Fy       → ファイルデータベース更新
#   -Fs file  → ファイルを含むパッケージ検索
```

---

## 5. その他のパッケージマネージャ

### 5.1 snap（Ubuntu）

```bash
# snap: サンドボックス化されたパッケージ
sudo snap install code --classic # インストール（--classic: サンドボックスなし）
sudo snap install firefox        # サンドボックス付き
snap list                        # 一覧
snap info firefox                # 詳細情報
sudo snap refresh                # 全更新
sudo snap refresh firefox        # 特定パッケージ更新
sudo snap remove firefox         # 削除
sudo snap revert firefox         # 前バージョンに戻す

# チャンネル管理
snap info --verbose firefox      # 利用可能なチャンネル
sudo snap install firefox --channel=esr/stable  # ESR版

# snap のメンテナンス
snap changes                     # 変更履歴
snap connections firefox         # インターフェース接続
sudo snap connect firefox:camera # カメラアクセスを許可

# snap の自動更新を制御
sudo snap set system refresh.timer=sat,04:00  # 土曜4時に更新
```

### 5.2 flatpak

```bash
# flatpak: クロスディストリビューションパッケージ
sudo apt install flatpak         # flatpak のインストール
flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo

# 基本操作
flatpak install flathub org.gimp.GIMP
flatpak list                     # 一覧
flatpak update                   # 更新
flatpak uninstall org.gimp.GIMP  # 削除
flatpak search gimp              # 検索
flatpak info org.gimp.GIMP       # 情報

# ランタイムの管理
flatpak list --runtime           # インストール済みランタイム
flatpak uninstall --unused       # 未使用のランタイムを削除
```

### 5.3 nix（宣言的パッケージ管理）

```bash
# nix: 宣言的・再現可能なパッケージ管理
# インストール
sh <(curl -L https://nixos.org/nix/install) --daemon

# 基本操作（従来のコマンド）
nix-env -iA nixpkgs.nginx       # インストール
nix-env -q                      # 一覧
nix-env -u                      # 更新
nix-env -e nginx                # 削除

# 新しいコマンド（Nix 2.4+、experimental）
nix profile install nixpkgs#nginx
nix profile list
nix profile upgrade
nix profile remove nginx

# 一時的に使用（インストールせずに実行）
nix-shell -p nginx               # 一時的な環境
nix run nixpkgs#cowsay -- "Hello" # 一時実行

# nix の利点:
# - 宣言的な設定で再現可能
# - 複数バージョンの共存が可能
# - ロールバックが容易
# - 全ディストリビューションで同じ
```

### 5.4 AppImage

```bash
# AppImage: ポータブルなLinuxアプリケーション
# ダウンロードして実行権限を付けるだけ
chmod +x MyApp.AppImage
./MyApp.AppImage

# AppImageLauncher でシステム統合
# - デスクトップエントリの自動作成
# - アップデートの管理
sudo apt install appimagelauncher

# 管理のベストプラクティス
mkdir -p ~/Applications
mv MyApp.AppImage ~/Applications/
```

---

## 6. 言語別パッケージマネージャ

### 6.1 Node.js / JavaScript

```bash
# npm（Node.js 同梱）
npm install -g typescript        # グローバル
npm install express              # ローカル（プロジェクト内）
npm install -D jest              # 開発依存
npx create-react-app myapp       # 一時実行

npm list -g --depth=0            # グローバルインストール済み
npm outdated                     # 更新可能なパッケージ
npm update                       # 更新
npm audit                        # セキュリティ監査
npm audit fix                    # 自動修正

# pnpm（高速・ディスク効率的）
npm install -g pnpm
pnpm install                     # package.json からインストール
pnpm add express                 # パッケージ追加
pnpm add -D jest                 # 開発依存
pnpm remove express              # 削除

# Bun（高速なJavaScriptランタイム + パッケージマネージャ）
curl -fsSL https://bun.sh/install | bash
bun install                      # package.json からインストール
bun add express                  # パッケージ追加
bun run dev                      # スクリプト実行

# バージョン管理（Node.js自体）
# fnm（推奨）
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 20                   # Node.js 20 をインストール
fnm use 20                       # バージョン切り替え
fnm list                         # インストール済みバージョン
fnm default 20                   # デフォルトバージョン

# nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20
nvm alias default 20
```

### 6.2 Python

```bash
# pip
pip install requests             # インストール
pip install requests==2.31.0     # バージョン指定
pip install -r requirements.txt  # 一括インストール
pip list                         # 一覧
pip show requests                # 情報表示
pip freeze > requirements.txt    # 依存を書き出し
pip install --upgrade requests   # 更新

# pipx: CLIツールのインストールに推奨（隔離環境）
pip install pipx
pipx install black
pipx install ruff
pipx install poetry
pipx list                        # インストール済み
pipx upgrade-all                 # 全更新

# uv: 高速なPythonパッケージマネージャ
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install requests          # pip互換
uv pip compile requirements.in -o requirements.txt  # ロック
uv venv                          # 仮想環境作成
uv run script.py                 # スクリプト実行

# uvx: pipx の代替（uv内蔵）
uvx black file.py                # 一時実行
uvx ruff check .                 # リンター実行

# 仮想環境
python -m venv .venv             # 仮想環境作成
source .venv/bin/activate        # 有効化
deactivate                       # 無効化

# バージョン管理（Python自体）
# pyenv
curl https://pyenv.run | bash
pyenv install 3.12.0
pyenv global 3.12.0
pyenv local 3.12.0               # ディレクトリ単位で設定
pyenv versions                   # インストール済み
```

### 6.3 Rust

```bash
# rustup（Rust ツールチェーン管理）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update                    # ツールチェーン更新
rustup component add clippy      # コンポーネント追加
rustup target add wasm32-unknown-unknown  # ターゲット追加

# cargo（Rust パッケージマネージャ）
cargo install ripgrep            # バイナリインストール
cargo install --locked bat       # Cargo.lock を尊重
cargo install-update -a          # インストール済みバイナリ更新（要 cargo-update）
```

### 6.4 Go

```bash
# Go ツールのインストール
go install golang.org/x/tools/gopls@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Go バージョン管理
go install golang.org/dl/go1.22.0@latest
go1.22.0 download
```

### 6.5 Ruby

```bash
# gem
gem install bundler
gem install rails
gem list                         # インストール済み
gem update                       # 全更新

# bundle（Bundler）
bundle init                      # Gemfile 作成
bundle install                   # Gemfile から一括
bundle update                    # 依存更新
bundle exec rails server         # 環境内で実行

# rbenv（バージョン管理）
brew install rbenv ruby-build
rbenv install 3.3.0
rbenv global 3.3.0
rbenv versions
```

---

## 7. コンテナ内のパッケージ管理

```bash
# Dockerfile でのベストプラクティス

# Debian/Ubuntu ベース
# --- 推奨パターン ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ポイント:
# - apt ではなく apt-get を使う（非対話的に適している）
# - --no-install-recommends で推奨パッケージを省略
# - 1つの RUN でまとめてキャッシュ削除（レイヤーサイズ削減）
# - /var/lib/apt/lists/* を削除

# Alpine Linux ベース（軽量コンテナ）
RUN apk add --no-cache \
    curl \
    nginx

# RHEL/Fedora ベース
RUN dnf install -y --setopt=install_weak_deps=False \
        curl \
        nginx && \
    dnf clean all

# マルチステージビルドでのパッケージ管理
FROM golang:1.22 AS builder
RUN go build -o /app .

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /app /app
CMD ["/app"]
```

---

## 8. セキュリティとベストプラクティス

### 8.1 パッケージのセキュリティ確認

```bash
# Ubuntu: セキュリティ更新の確認
sudo apt list --upgradable 2>/dev/null | grep -i security

# セキュリティ更新のみ適用
sudo apt-get -s dist-upgrade | grep "^Inst" | grep -i securi
sudo unattended-upgrade --dry-run

# RHEL/Fedora: セキュリティアドバイザリ
sudo dnf updateinfo list security
sudo dnf update --security       # セキュリティ更新のみ
sudo dnf updateinfo info RHSA-2025:0001  # 特定アドバイザリの詳細

# パッケージの整合性チェック
# Debian/Ubuntu
debsums -c                       # 変更されたファイルを検出
sudo debsums -a nginx            # 特定パッケージ

# RHEL/Fedora
rpm -Va                          # 全パッケージの検証
rpm -V nginx                     # 特定パッケージの検証
```

### 8.2 GPGキーの管理

```bash
# apt のGPGキー管理（新しい方法）
# キーリングディレクトリ
ls /usr/share/keyrings/
ls /etc/apt/trusted.gpg.d/

# キーのダウンロードと変換
curl -fsSL https://example.com/gpg.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/example.gpg

# sources.list での指定
# deb [signed-by=/usr/share/keyrings/example.gpg] https://packages.example.com/apt stable main

# rpm のGPGキー管理
sudo rpm --import https://packages.example.com/gpg.key
rpm -qa gpg-pubkey*              # インポート済みキー
```

### 8.3 パッケージ管理のベストプラクティス

```bash
# 1. 常にリポジトリを最新にしてからインストール
sudo apt update && sudo apt install package

# 2. 本番環境ではバージョンを固定
sudo apt install nginx=1.24.0-1ubuntu1
# または
sudo apt-mark hold nginx

# 3. 定期的にセキュリティ更新を適用
# 自動更新の設定
sudo apt install unattended-upgrades

# 4. 不要なパッケージを削除してアタックサーフェスを減らす
sudo apt autoremove --purge
dpkg -l | grep '^rc' | awk '{print $2}' | xargs sudo dpkg --purge

# 5. 信頼できるリポジトリのみ使用
# GPG署名を検証してからインストール

# 6. パッケージの出所を確認
apt-cache policy nginx           # どのリポジトリからか
```

---

## 9. 実践パターン

### 9.1 サーバーの初期セットアップ

```bash
#!/bin/bash
set -euo pipefail

# Ubuntu サーバーの初期セットアップスクリプト
echo "=== System Update ==="
sudo apt update && sudo apt upgrade -y

echo "=== Install Essential Packages ==="
sudo apt install -y \
    git curl wget \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    unzip \
    jq \
    htop \
    tmux \
    vim \
    ufw \
    fail2ban

echo "=== Install Web Server ==="
sudo apt install -y nginx

echo "=== Install Database ==="
sudo apt install -y postgresql postgresql-client

echo "=== Install Redis ==="
sudo apt install -y redis-server

echo "=== Configure Firewall ==="
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

echo "=== Enable Services ==="
sudo systemctl enable --now nginx postgresql redis-server

echo "=== Setup Automatic Security Updates ==="
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

echo "=== Cleanup ==="
sudo apt autoremove -y
sudo apt clean

echo "=== Setup Complete ==="
```

### 9.2 開発マシンのセットアップ自動化（macOS）

```bash
#!/bin/bash
set -euo pipefail

# macOS 開発環境セットアップスクリプト

echo "=== Installing Xcode Command Line Tools ==="
xcode-select --install 2>/dev/null || true

echo "=== Installing Homebrew ==="
if ! command -v brew &>/dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "=== Installing from Brewfile ==="
brew bundle install --file=Brewfile

echo "=== Setting up Shell ==="
# zsh プラグイン
if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

echo "=== Setting up fzf ==="
"$(brew --prefix)/opt/fzf/install" --all --no-bash --no-fish

echo "=== Setting up Starship ==="
mkdir -p ~/.config
[[ -f ~/.config/starship.toml ]] || cat > ~/.config/starship.toml <<'EOF'
[character]
success_symbol = "[>](green)"
error_symbol = "[>](red)"

[directory]
truncation_length = 3
EOF

echo "=== Setting up Git ==="
git config --global init.defaultBranch main
git config --global pull.rebase true
git config --global core.autocrlf input

echo "=== Setting up Node.js (fnm) ==="
if command -v fnm &>/dev/null; then
    fnm install --lts
    fnm default lts-latest
fi

echo "=== Setting up Python (pyenv) ==="
if command -v pyenv &>/dev/null; then
    pyenv install 3.12 --skip-existing
    pyenv global 3.12
fi

echo "=== Setup Complete ==="
echo "Please restart your terminal."
```

### 9.3 パッケージの一括アップデートスクリプト

```bash
#!/bin/bash
set -euo pipefail

# 全パッケージマネージャの一括更新スクリプト
LOG_FILE="/tmp/update-all-$(date +%Y%m%d).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Update All Packages: $(date) ==="

# macOS / Homebrew
if command -v brew &>/dev/null; then
    echo "--- Homebrew ---"
    brew update
    brew upgrade
    brew cleanup
    brew autoremove
fi

# apt (Debian/Ubuntu)
if command -v apt &>/dev/null; then
    echo "--- APT ---"
    sudo apt update
    sudo apt upgrade -y
    sudo apt autoremove -y
fi

# dnf (RHEL/Fedora)
if command -v dnf &>/dev/null; then
    echo "--- DNF ---"
    sudo dnf upgrade -y
    sudo dnf autoremove -y
fi

# snap
if command -v snap &>/dev/null; then
    echo "--- Snap ---"
    sudo snap refresh
fi

# flatpak
if command -v flatpak &>/dev/null; then
    echo "--- Flatpak ---"
    flatpak update -y
fi

# npm (グローバルパッケージ)
if command -v npm &>/dev/null; then
    echo "--- npm (global) ---"
    npm update -g
fi

# pip (pipx 管理のツール)
if command -v pipx &>/dev/null; then
    echo "--- pipx ---"
    pipx upgrade-all
fi

# Rust
if command -v rustup &>/dev/null; then
    echo "--- Rust ---"
    rustup update
fi

echo "=== Update Complete: $(date) ==="
echo "Log saved to: $LOG_FILE"
```

### 9.4 パッケージの比較・移行

```bash
# サーバー間でのパッケージ比較
# サーバーAのパッケージリスト
ssh server-a "dpkg --get-selections" > /tmp/server-a-packages.txt

# サーバーBのパッケージリスト
ssh server-b "dpkg --get-selections" > /tmp/server-b-packages.txt

# 差分の確認
diff /tmp/server-a-packages.txt /tmp/server-b-packages.txt

# パッケージリストの移行
# エクスポート
dpkg --get-selections > packages.txt

# インポート（別サーバー）
sudo dpkg --set-selections < packages.txt
sudo apt-get dselect-upgrade

# macOS のパッケージ移行
# エクスポート
brew bundle dump --force --file=Brewfile

# インポート（別マシン）
brew bundle install --file=Brewfile
```

---

## 10. Alpine Linux のパッケージ管理（apk）

```bash
# apk（Alpine Package Keeper）
# Docker コンテナで最も使われる軽量ディストリビューション

# 基本操作
apk update                       # パッケージリストの更新
apk upgrade                      # 全パッケージ更新
apk add nginx                    # インストール
apk add --no-cache nginx         # キャッシュを残さずインストール
apk del nginx                    # 削除

# 検索・情報
apk search nginx                 # パッケージ検索
apk info nginx                   # パッケージ情報
apk info -L nginx                # ファイル一覧
apk list --installed             # インストール済み一覧

# 仮想パッケージ（ビルド時のみ必要なパッケージ管理）
apk add --virtual .build-deps gcc musl-dev python3-dev
# ビルド後に一括削除
apk del .build-deps

# Dockerfile での典型的パターン
# RUN apk add --no-cache --virtual .build-deps \
#         gcc musl-dev python3-dev && \
#     pip install --no-cache-dir -r requirements.txt && \
#     apk del .build-deps

# リポジトリの管理
cat /etc/apk/repositories
# コミュニティリポジトリの有効化
# echo "http://dl-cdn.alpinelinux.org/alpine/v3.19/community" >> /etc/apk/repositories

# エッジ（テスト）リポジトリからのインストール
apk add --repository=http://dl-cdn.alpinelinux.org/alpine/edge/testing package-name
```

---

## 11. パッケージマネージャの比較と選択指針

### 11.1 システムパッケージマネージャの比較

```bash
# === パッケージフォーマットの比較 ===
# deb (Debian/Ubuntu):
#   - 最も広いエコシステム
#   - PPAによる簡単なサードパーティリポジトリ
#   - aptが依存関係を自動解決
#   - パッケージ数が最も多い

# rpm (RHEL/Fedora):
#   - エンタープライズ向けの安定性
#   - SELinux との統合
#   - モジュールストリームによるバージョン管理
#   - 商用サポートあり（Red Hat）

# PKGBUILD (Arch):
#   - 最新のパッケージが最速で利用可能
#   - AURによる膨大なコミュニティパッケージ
#   - シンプルなパッケージビルドシステム
#   - ローリングリリース

# Homebrew (macOS/Linux):
#   - macOS のデファクトスタンダード
#   - 開発者向けツールが充実
#   - Brewfileによる宣言的管理
#   - root権限不要
```

### 11.2 ユニバーサルパッケージの比較

```bash
# === snap vs flatpak vs AppImage ===

# snap:
#   - Canonical（Ubuntu）が開発
#   - サーバーサイドアプリにも対応
#   - 自動更新
#   - 中央集権的（Snap Store）
#   - 起動がやや遅い

# flatpak:
#   - コミュニティ主導
#   - デスクトップアプリに特化
#   - 複数のリモートリポジトリ
#   - サンドボックスが強力
#   - ランタイム共有でディスク効率的

# AppImage:
#   - パッケージマネージャ不要
#   - 1ファイル = 1アプリ
#   - ポータブル（USBメモリで持ち運び可能）
#   - 自動更新の仕組みが弱い
#   - サンドボックスなし

# 選択指針:
# サーバー → snap（または従来のパッケージ）
# デスクトップ → flatpak（ディストリ非依存）
# ポータブル → AppImage
```

### 11.3 パッケージ管理のトラブルシューティング

```bash
# === apt のトラブルシューティング ===
# ロックファイルの問題
sudo rm /var/lib/dpkg/lock-frontend
sudo rm /var/lib/apt/lists/lock
sudo dpkg --configure -a

# 壊れたパッケージの修復
sudo apt --fix-broken install
sudo dpkg --configure -a
sudo apt update --fix-missing

# sources.list のエラー
sudo apt update 2>&1 | grep "NO_PUBKEY" | awk '{print $NF}' | \
    xargs -I {} sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys {}

# === dnf のトラブルシューティング ===
# キャッシュの破損
sudo dnf clean all
sudo dnf makecache

# 壊れたRPMデータベース
sudo rpm --rebuilddb

# === Homebrew のトラブルシューティング ===
# 問題の診断
brew doctor

# Homebrew の再インストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/uninstall.sh)"
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# リンクの問題
brew link --overwrite package
brew link --overwrite --dry-run package  # 確認のみ

# 権限の問題
sudo chown -R $(whoami) $(brew --prefix)/*
```

---

## まとめ

| ディストリビューション | マネージャ | インストール | 更新 | 検索 |
|----------------------|-----------|------------|------|------|
| Ubuntu/Debian | apt | apt install pkg | apt upgrade | apt search pkg |
| RHEL/Fedora | dnf | dnf install pkg | dnf update | dnf search pkg |
| Arch Linux | pacman | pacman -S pkg | pacman -Syu | pacman -Ss pkg |
| macOS | brew | brew install pkg | brew upgrade | brew search pkg |
| Ubuntu (snap) | snap | snap install pkg | snap refresh | snap find pkg |
| Cross-distro | flatpak | flatpak install pkg | flatpak update | flatpak search pkg |
| Declarative | nix | nix-env -iA pkg | nix-env -u | nix search pkg |

---

## 次に読むべきガイド
→ [[../07-advanced/00-tmux-screen.md]] — ターミナルマルチプレクサ

---

## 参考文献
1. "APT User's Guide." Debian Documentation.
2. "Homebrew Documentation." brew.sh.
3. "DNF Documentation." dnf.readthedocs.io.
4. "Arch Wiki: pacman." wiki.archlinux.org/title/pacman.
5. "Nix Package Manager Guide." nixos.org/manual/nix.
6. Barrett, D. "Efficient Linux at the Command Line." Ch.9, O'Reilly, 2022.
