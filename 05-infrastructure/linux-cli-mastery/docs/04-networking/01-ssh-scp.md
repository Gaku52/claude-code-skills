# リモート接続（SSH, SCP, rsync）

> SSH はリモートサーバー操作の基盤。安全な接続・ファイル転送・トンネリングを可能にする。

## この章で学ぶこと

- [ ] SSH で安全にリモートサーバーに接続できる
- [ ] SSH鍵の生成・管理ができる
- [ ] SCP / rsync でファイル転送ができる
- [ ] SSHトンネリングを活用できる
- [ ] SSH設定ファイルを効率的に構成できる
- [ ] sshd のセキュリティ設定を理解できる
- [ ] 多段SSH・踏み台サーバーを使いこなせる
- [ ] SSH関連のトラブルシューティングができる

---

## 1. SSH の基本

### 1.1 SSH とは

SSH（Secure Shell）は、ネットワーク上の安全な通信チャネルを確立するプロトコル。暗号化された通信により、以下の機能を提供する：

- **リモートシェルアクセス**: 安全にリモートサーバーのコマンドラインを操作
- **ファイル転送**: SCP, SFTP, rsync による暗号化ファイル転送
- **ポートフォワーディング**: SSHトンネルによる安全なネットワーク中継
- **X11フォワーディング**: リモートGUIアプリケーションの表示

```bash
# SSH のバージョン確認
ssh -V
# OpenSSH_9.6p1, LibreSSL 3.3.6

# SSH プロトコルの仕組み（簡略化）
# 1. TCPコネクション確立（デフォルト: ポート22）
# 2. SSHプロトコルバージョン交換
# 3. 鍵交換アルゴリズムのネゴシエーション
# 4. サーバー認証（ホスト鍵の検証）
# 5. ユーザー認証（公開鍵、パスワード等）
# 6. 暗号化通信チャネル確立
```

### 1.2 基本的な接続

```bash
# 基本: ssh [オプション] [ユーザー@]ホスト

# ===== 接続方法 =====

# ユーザー指定で接続
ssh user@server.example.com

# 現在のユーザー名で接続
ssh server.example.com

# ポート指定
ssh -p 2222 user@server.com

# 秘密鍵を明示的に指定
ssh -i ~/.ssh/my_key user@server.com

# IPv6 アドレスで接続
ssh user@"[2001:db8::1]"

# 接続時のホスト鍵チェックをスキップ（初回接続時の自動化用）
ssh -o StrictHostKeyChecking=no user@server.com
# ※ セキュリティリスクあり。自動化スクリプト以外では非推奨

# 接続先のホスト鍵フィンガープリントを事前確認
ssh-keyscan server.example.com 2>/dev/null | ssh-keygen -l -f -
```

### 1.3 リモートコマンド実行

```bash
# リモートでコマンド実行して切断
ssh user@server.com command

# 単一コマンド
ssh user@server "ls -la /var/log"
ssh user@server "df -h && free -m"
ssh user@server 'cat /etc/nginx/nginx.conf'

# 複数コマンドの実行
ssh user@server "uname -a; uptime; who"
ssh user@server "cd /var/log && tail -100 syslog"

# パイプを使ったコマンド
ssh user@server "ps aux | grep nginx | grep -v grep"

# sudo 付きコマンド（-t: 疑似端末を強制割り当て）
ssh -t user@server "sudo systemctl restart nginx"

# ヒアドキュメントで複雑なスクリプトを実行
ssh user@server bash <<'EOF'
echo "=== System Info ==="
uname -a
echo ""
echo "=== Disk Usage ==="
df -h
echo ""
echo "=== Memory ==="
free -m
echo ""
echo "=== Top Processes ==="
ps aux --sort=-%cpu | head -5
EOF

# リモートコマンドの終了コードを受け取る
ssh user@server "test -f /var/run/app.pid"
echo "Exit code: $?"  # 0=ファイルあり, 1=なし

# 複数サーバーで同じコマンド実行
for server in web1 web2 web3; do
    echo "=== $server ==="
    ssh "$server" "uptime"
done

# 並列実行（xargs）
echo -e "web1\nweb2\nweb3" | xargs -P 3 -I {} ssh {} "uptime"

# 並列実行（GNU parallel）
parallel ssh {} "uptime" ::: web1 web2 web3
```

### 1.4 SSH の接続オプション

```bash
# 詳細デバッグ出力（接続問題の調査に必須）
ssh -v user@server    # レベル1（基本的なデバッグ情報）
ssh -vv user@server   # レベル2（より詳細）
ssh -vvv user@server  # レベル3（最大限の詳細）

# 圧縮を有効にする（遅い回線で有効）
ssh -C user@server

# X11 フォワーディング（リモートのGUIアプリを表示）
ssh -X user@server    # X11 フォワーディング
ssh -Y user@server    # 信頼された X11 フォワーディング（セキュリティ緩い）

# 接続のキープアライブ
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 user@server

# 接続タイムアウト
ssh -o ConnectTimeout=10 user@server

# バッチモード（パスワード入力を求めない）
ssh -o BatchMode=yes user@server "uptime"

# エスケープ文字の変更（デフォルトは ~）
ssh -e '%' user@server

# SSH セッション内のエスケープコマンド（デフォルト: ~）
# ~.   → 接続を強制切断
# ~^Z  → SSHをバックグラウンドに移動
# ~~   → ~ 文字を送信
# ~?   → エスケープコマンド一覧表示
# ~#   → フォワードされたコネクション一覧
# ~C   → コマンドラインを開く（動的フォワーディング追加等）
```

---

## 2. SSH鍵の管理

### 2.1 鍵ペアの生成

```bash
# ===== 鍵アルゴリズムの選択 =====

# Ed25519（推奨）: 高速・安全・鍵が短い
ssh-keygen -t ed25519 -C "gaku@example.com"
# 生成される鍵:
# ~/.ssh/id_ed25519      ← 秘密鍵（絶対に共有しない）
# ~/.ssh/id_ed25519.pub  ← 公開鍵（サーバーに登録する）

# Ed25519-SK（FIDO2/U2Fハードウェアキー用）
ssh-keygen -t ed25519-sk -C "gaku@example.com"

# RSA 4096bit: レガシー環境向け（古いシステムとの互換性）
ssh-keygen -t rsa -b 4096 -C "gaku@example.com"

# ECDSA: 楕円曲線暗号（NISTカーブ）
ssh-keygen -t ecdsa -b 521 -C "gaku@example.com"

# ===== 鍵生成のオプション =====

# ファイル名を指定して生成
ssh-keygen -t ed25519 -f ~/.ssh/project_key -C "project@example.com"

# パスフレーズなしで生成（自動化用）
ssh-keygen -t ed25519 -f ~/.ssh/automation_key -N "" -C "automation"

# パスフレーズを変更
ssh-keygen -p -f ~/.ssh/id_ed25519

# 鍵のフィンガープリントを確認
ssh-keygen -l -f ~/.ssh/id_ed25519.pub
# 256 SHA256:AbCdEfGhIj... gaku@example.com (ED25519)

# 鍵のビジュアルフィンガープリント（ランダムアート）
ssh-keygen -lv -f ~/.ssh/id_ed25519.pub

# 公開鍵の内容を確認
cat ~/.ssh/id_ed25519.pub
# ssh-ed25519 AAAA... gaku@example.com

# 秘密鍵から公開鍵を再生成
ssh-keygen -y -f ~/.ssh/id_ed25519 > ~/.ssh/id_ed25519.pub
```

### 2.2 鍵アルゴリズムの比較

```text
┌──────────────┬────────────┬──────────────┬───────────────┬──────────────────┐
│ アルゴリズム │ 鍵長       │ 安全性       │ 速度          │ 備考             │
├──────────────┼────────────┼──────────────┼───────────────┼──────────────────┤
│ Ed25519      │ 256bit     │ ◎ 非常に高い │ ◎ 非常に速い  │ 現在の推奨       │
│ Ed25519-SK   │ 256bit     │ ◎ HW鍵必須   │ ◎             │ FIDO2対応        │
│ ECDSA        │ 256-521bit │ ○ 高い       │ ○ 速い        │ NISTカーブ       │
│ RSA          │ 2048-4096  │ ○ 高い(4096) │ △ やや遅い    │ レガシー互換     │
│ DSA          │ 1024bit    │ × 非推奨     │ ○ 速い        │ OpenSSH 7.0で廃止│
└──────────────┴────────────┴──────────────┴───────────────┴──────────────────┘

推奨: Ed25519 > ECDSA > RSA 4096 > RSA 2048
```

### 2.3 公開鍵の登録

```bash
# 方法1: ssh-copy-id（最も簡単）
ssh-copy-id user@server.com
ssh-copy-id -i ~/.ssh/specific_key.pub user@server.com
ssh-copy-id -p 2222 user@server.com

# 方法2: 手動でコピー
cat ~/.ssh/id_ed25519.pub | ssh user@server "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# 方法3: クリップボードからペースト（macOS）
pbcopy < ~/.ssh/id_ed25519.pub
# サーバー側で ~/.ssh/authorized_keys にペースト

# 方法4: GitHub/GitLab から公開鍵を取得
curl -s https://github.com/username.keys >> ~/.ssh/authorized_keys
curl -s https://gitlab.com/username.keys >> ~/.ssh/authorized_keys

# 登録済み公開鍵の確認（サーバー側）
cat ~/.ssh/authorized_keys

# 特定の鍵に制限を付ける（authorized_keys内）
# コマンド制限: 特定コマンドのみ許可
# command="rsync --server --sender -logDtprze.iLsfxCIvu . /data",no-pty,no-port-forwarding ssh-ed25519 AAAA...
# IPアドレス制限
# from="192.168.1.0/24" ssh-ed25519 AAAA...
# 複数制限の組み合わせ
# from="10.0.0.0/8",command="/usr/local/bin/backup.sh",no-pty,no-port-forwarding ssh-ed25519 AAAA...
```

### 2.4 ssh-agent による鍵管理

```bash
# ssh-agent: パスフレーズ付き鍵のパスフレーズをメモリに保持
# → 毎回パスフレーズを入力する必要がなくなる

# エージェント起動
eval "$(ssh-agent -s)"
# Agent pid 12345

# 鍵をエージェントに登録
ssh-add ~/.ssh/id_ed25519        # デフォルト鍵
ssh-add ~/.ssh/project_key       # 特定の鍵

# 登録済み鍵の一覧
ssh-add -l                       # フィンガープリント表示
ssh-add -L                       # 公開鍵全体を表示

# 全鍵を削除
ssh-add -D

# 特定の鍵を削除
ssh-add -d ~/.ssh/id_ed25519

# 有効期限付きで登録（セキュリティ向上）
ssh-add -t 3600 ~/.ssh/id_ed25519    # 1時間で自動削除
ssh-add -t 28800 ~/.ssh/id_ed25519   # 8時間（就業時間中のみ）

# macOS: Keychainと統合
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
# → ログイン時に自動的に鍵がロードされる

# macOS: ~/.ssh/config に設定（永続化）
# Host *
#     UseKeychain yes
#     AddKeysToAgent yes
#     IdentityFile ~/.ssh/id_ed25519

# エージェントフォワーディング（踏み台経由の認証に使用）
ssh -A user@bastion
# bastion から内部サーバーに接続する際、ローカルの鍵が使える
# ※ セキュリティリスクあり。信頼できるサーバーでのみ使用

# エージェント転送の確認
echo "$SSH_AUTH_SOCK"
ssh-add -l  # 踏み台サーバー上で実行 → ローカルの鍵が見える

# エージェントの停止
ssh-agent -k
```

### 2.5 SSH鍵のセキュリティ

```bash
# ===== パーミッション（重要）=====
chmod 700 ~/.ssh                 # ディレクトリ
chmod 600 ~/.ssh/id_ed25519      # 秘密鍵
chmod 644 ~/.ssh/id_ed25519.pub  # 公開鍵
chmod 600 ~/.ssh/authorized_keys # 認証鍵リスト
chmod 600 ~/.ssh/config          # 設定ファイル
chmod 600 ~/.ssh/known_hosts     # 既知ホスト

# パーミッションが正しくないと SSH は鍵を拒否する
# エラー例: "Permissions 0644 for '/home/user/.ssh/id_ed25519' are too open."

# パーミッション一括修正スクリプト
fix_ssh_permissions() {
    local ssh_dir="$HOME/.ssh"

    chmod 700 "$ssh_dir"
    chmod 600 "$ssh_dir"/id_* 2>/dev/null
    chmod 644 "$ssh_dir"/*.pub 2>/dev/null
    chmod 600 "$ssh_dir"/authorized_keys 2>/dev/null
    chmod 600 "$ssh_dir"/config 2>/dev/null
    chmod 600 "$ssh_dir"/known_hosts 2>/dev/null

    echo "SSH パーミッション修正完了"
    ls -la "$ssh_dir"
}

# ===== known_hosts の管理 =====

# ホスト鍵のハッシュ化（IPアドレスの漏洩防止）
ssh-keygen -H -f ~/.ssh/known_hosts

# 特定ホストの鍵を削除（ホスト鍵が変わった場合）
ssh-keygen -R server.example.com
ssh-keygen -R "[server.example.com]:2222"  # ポート指定の場合

# ホスト鍵を事前に追加
ssh-keyscan -H server.example.com >> ~/.ssh/known_hosts 2>/dev/null
ssh-keyscan -p 2222 -H server.example.com >> ~/.ssh/known_hosts 2>/dev/null

# ===== 鍵のローテーション =====

# 1. 新しい鍵を生成
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_new -C "gaku@example.com (rotated $(date +%Y%m%d))"

# 2. 新しい公開鍵をサーバーに追加
ssh-copy-id -i ~/.ssh/id_ed25519_new.pub user@server

# 3. 新しい鍵で接続テスト
ssh -i ~/.ssh/id_ed25519_new user@server "echo 'New key works'"

# 4. 古い鍵を削除（サーバー側の authorized_keys から）
ssh user@server "sed -i '/OLD_KEY_COMMENT/d' ~/.ssh/authorized_keys"

# 5. ローカルの鍵を入れ替え
mv ~/.ssh/id_ed25519 ~/.ssh/id_ed25519_old
mv ~/.ssh/id_ed25519_new ~/.ssh/id_ed25519
mv ~/.ssh/id_ed25519_new.pub ~/.ssh/id_ed25519.pub
```

---

## 3. SSH設定ファイル（~/.ssh/config）

### 3.1 基本設定

```bash
# ~/.ssh/config で接続設定を管理
# 長いコマンドを短いエイリアスにできる

# 基本的なホスト定義
# ssh production → ssh -p 2222 -i ~/.ssh/prod_key deploy@prod.example.com
Host production
    HostName prod.example.com
    User deploy
    Port 2222
    IdentityFile ~/.ssh/prod_key

Host staging
    HostName staging.example.com
    User deploy
    IdentityFile ~/.ssh/staging_key

# ワイルドカード
Host *.example.com
    User gaku
    IdentityFile ~/.ssh/id_ed25519
```

### 3.2 踏み台サーバー設定

```bash
# ===== ProxyJump（推奨: OpenSSH 7.3+）=====

# 踏み台サーバー経由で内部サーバーに接続
Host bastion
    HostName bastion.example.com
    User admin
    IdentityFile ~/.ssh/bastion_key

Host internal-server
    HostName 192.168.1.100
    User admin
    ProxyJump bastion

# 多段踏み台（bastion1 → bastion2 → target）
Host target
    HostName 10.0.1.50
    User admin
    ProxyJump bastion1,bastion2

# ssh internal-server だけで接続可能

# ===== ProxyCommand（レガシー: 古い OpenSSH 用）=====

Host internal-legacy
    HostName 192.168.1.100
    User admin
    ProxyCommand ssh -W %h:%p bastion

# ===== コマンドラインでの踏み台指定 =====

# -J オプション（ProxyJump のコマンドライン版）
ssh -J bastion.example.com user@10.0.1.100
ssh -J user1@bastion1:22,user2@bastion2:22 user3@target
```

### 3.3 高度な設定

```bash
# ===== 環境別設定 =====

# 開発環境
Host dev-*
    User developer
    IdentityFile ~/.ssh/dev_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR

# 本番環境（厳格設定）
Host prod-*
    User deploy
    IdentityFile ~/.ssh/prod_key
    StrictHostKeyChecking yes
    PasswordAuthentication no
    ForwardAgent no

# AWS EC2 インスタンス
Host ec2-*
    User ec2-user
    IdentityFile ~/.ssh/aws_key.pem
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

# GitHub（鍵の使い分け）
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_ed25519

Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_work_ed25519

# ===== 接続の最適化 =====

# 接続の多重化（ControlMaster）
# 同じホストへの2回目以降の接続が瞬時になる
Host *
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
    # ControlMaster auto: 最初の接続をマスターにする
    # ControlPath: ソケットファイルの場所
    # ControlPersist 600: マスター接続を600秒維持

# ソケットディレクトリの作成が必要
# mkdir -p ~/.ssh/sockets

# 多重化の手動管理
# ssh -O check hostname   → マスター接続の確認
# ssh -O stop hostname    → マスター接続の停止
# ssh -O exit hostname    → マスター接続の終了

# ===== 全ホスト共通設定 =====
Host *
    ServerAliveInterval 60       # 60秒ごとにKeepAlive
    ServerAliveCountMax 3        # 3回失敗で切断
    AddKeysToAgent yes           # 鍵を自動でエージェントに追加
    IdentitiesOnly yes           # 指定した鍵のみ使用
    HashKnownHosts yes           # known_hosts をハッシュ化
    Compression yes              # 圧縮を有効化
    TCPKeepAlive yes             # TCP レベルのキープアライブ
```

### 3.4 設定ファイルの読み込み順序

```bash
# SSH 設定の優先順位（上が高い）
# 1. コマンドラインオプション（ssh -p 2222 ...）
# 2. ユーザー設定（~/.ssh/config）
# 3. システム設定（/etc/ssh/ssh_config）

# 設定ファイル内の優先順位
# - 最初にマッチした Host ブロックの設定が使われる
# - Host * は最後に書く（フォールバック）

# 設定の確認（実際に使われる設定値を表示）
ssh -G hostname
ssh -G hostname | grep -i proxyj

# Include で設定を分割管理
# ~/.ssh/config の先頭に記述:
Include config.d/*

# ~/.ssh/config.d/work
Host work-*
    User developer
    IdentityFile ~/.ssh/work_key

# ~/.ssh/config.d/personal
Host personal-*
    User gaku
    IdentityFile ~/.ssh/personal_key
```

---

## 4. ファイル転送（SCP, rsync）

### 4.1 SCP

```bash
# scp: SSH経由のファイルコピー
# 注意: OpenSSH 9.0 以降、内部的に sftp プロトコルを使用
# シンプルなコピーには使えるが、rsync の方が高機能

# ===== ローカル → リモート =====
scp file.txt user@server:/home/user/
scp -r ./dir user@server:/home/user/       # ディレクトリ（再帰）
scp -P 2222 file.txt user@server:/tmp/     # ポート指定（-P 大文字！）
scp -i ~/.ssh/key file.txt user@server:/tmp/  # 鍵指定

# 複数ファイル
scp file1.txt file2.txt file3.txt user@server:/home/user/
scp *.log user@server:/var/log/backup/

# ===== リモート → ローカル =====
scp user@server:/var/log/app.log ./
scp -r user@server:/home/user/dir ./

# ===== リモート → リモート =====
scp user@server1:/file user@server2:/file

# ===== オプション =====
scp -C file.txt user@server:/tmp/       # 圧縮転送
scp -l 5000 file.txt user@server:/tmp/  # 帯域制限（Kbit/s）
scp -p file.txt user@server:/tmp/       # タイムスタンプ保持
scp -q file.txt user@server:/tmp/       # プログレス非表示
scp -v file.txt user@server:/tmp/       # デバッグ出力

# SCP の制限事項:
# - 差分転送ができない（毎回全データ転送）
# - 中断再開ができない
# - 除外パターンを指定できない
# - シンボリックリンクの扱いが限定的
# → これらが必要な場合は rsync を使用
```

### 4.2 rsync（推奨）

```bash
# rsync: 差分転送（高速・中断再開可能・柔軟）

# ===== 基本構文 =====
# rsync [オプション] 送信元 送信先

# ===== ローカル → リモート =====
rsync -avz ./project/ user@server:/home/user/project/
# -a: アーカイブモード（-rlptgoD と同等）
#   -r: 再帰的
#   -l: シンボリックリンクを保持
#   -p: パーミッション保持
#   -t: タイムスタンプ保持
#   -g: グループ保持
#   -o: オーナー保持
#   -D: デバイスファイル・特殊ファイル保持
# -v: 詳細表示
# -z: 圧縮転送

# ===== リモート → ローカル =====
rsync -avz user@server:/var/log/ ./logs/

# ===== ポート・鍵の指定 =====
rsync -avz -e "ssh -p 2222 -i ~/.ssh/key" ./project/ user@server:/app/

# ===== 重要: 末尾のスラッシュの意味 =====
rsync -avz ./dir  user@server:/dest/   # → /dest/dir/ として転送
rsync -avz ./dir/ user@server:/dest/   # → /dest/ の中身として転送
# 末尾の / は「ディレクトリの中身」を意味する
# / なしは「ディレクトリそのもの」を意味する

# ===== 除外パターン =====
rsync -avz --exclude='.git' --exclude='node_modules' ./project/ user@server:/app/
rsync -avz --exclude='*.log' --exclude='*.tmp' ./data/ user@server:/data/

# パターンファイルで除外
rsync -avz --exclude-from='.rsyncignore' ./project/ user@server:/app/

# .rsyncignore の例:
# .git/
# node_modules/
# *.log
# *.tmp
# .env
# .DS_Store
# __pycache__/
# *.pyc

# 除外と包含の組み合わせ
rsync -avz --include='*.py' --exclude='*' ./src/ user@server:/src/
# Python ファイルのみ転送

# ===== ドライラン（実行せずに確認）=====
rsync -avzn ./project/ user@server:/app/   # -n: ドライラン
rsync -avz --dry-run ./project/ user@server:/app/  # 同上

# ===== 削除同期 =====
# 送信元にないファイルを送信先からも削除
rsync -avz --delete ./project/ user@server:/app/

# 削除前に確認（ドライラン + 削除）
rsync -avzn --delete ./project/ user@server:/app/

# 削除を除外ファイルに限定しない
rsync -avz --delete --delete-excluded ./project/ user@server:/app/

# ===== 帯域制限 =====
rsync -avz --bwlimit=5000 ./large/ user@server:/backup/  # 5MB/s
rsync -avz --bwlimit=1m ./large/ user@server:/backup/    # 1MB/s

# ===== 進捗表示 =====
rsync -avz --progress ./large/ user@server:/backup/
rsync -avz --info=progress2 ./large/ user@server:/backup/  # 全体進捗

# ===== チェックサム検証 =====
rsync -avzc ./project/ user@server:/app/
# -c: タイムスタンプではなくチェックサムで差分判定（遅いが確実）

# ===== 部分転送（中断再開） =====
rsync -avz --partial --partial-dir=.rsync-partial ./large/ user@server:/backup/
# 中断しても部分的に転送されたファイルを保持 → 再開時に続きから

# -P は --partial --progress の短縮形
rsync -avzP ./large/ user@server:/backup/

# ===== バックアップ機能 =====
# 上書きされるファイルのバックアップを作成
rsync -avz --backup --backup-dir=../backup/$(date +%Y%m%d) \
    ./project/ user@server:/app/

# ===== ハードリンクを使った世代バックアップ =====
rsync -avz --link-dest=../backup-prev \
    user@server:/var/www/ ./backup-$(date +%Y%m%d)/
# --link-dest: 前回のバックアップと同じファイルはハードリンクで省容量化
```

### 4.3 rsync の高度な使い方

```bash
# ===== フィルタールール =====
rsync -avz --filter='- .git/' --filter='- node_modules/' ./project/ user@server:/app/

# マージファイル（.rsync-filter）
rsync -avz --filter=': .rsync-filter' ./project/ user@server:/app/

# ===== リモート間のコピー（自分経由） =====
rsync -avz user@server1:/data/ user@server2:/data/
# ※ データはローカルマシン経由で転送される

# ===== ローカル同期（cp の代わり） =====
rsync -avz /source/ /destination/
# cp -a より高速（差分のみ転送）

# ===== inotifywait と組み合わせたリアルタイム同期 =====
# Linux で inotify-tools が必要
while inotifywait -r -e modify,create,delete ./project/; do
    rsync -avz --delete ./project/ user@server:/app/
done

# ===== rsync デーモンモード =====
# サーバー側: /etc/rsyncd.conf
# [data]
#   path = /var/data
#   read only = false
#   auth users = rsyncuser
#   secrets file = /etc/rsyncd.secrets

# クライアント側
rsync -avz ./data/ rsyncuser@server::data/
rsync -avz rsyncuser@server::data/ ./data/

# ===== 統計情報の表示 =====
rsync -avz --stats ./project/ user@server:/app/
# Number of files: 1,234
# Total file size: 123,456,789 bytes
# Total transferred file size: 1,234,567 bytes
# Literal data: 1,200,000 bytes
# Matched data: 34,567 bytes
# Speedup is 100.00
```

### 4.4 sftp

```bash
# sftp: 対話的ファイル転送（FTPライクなインターフェース）

# 接続
sftp user@server
sftp -P 2222 user@server          # ポート指定
sftp -i ~/.ssh/key user@server    # 鍵指定

# sftp コマンド一覧
# ===== ナビゲーション =====
# ls            リモートのファイル一覧
# lls           ローカルのファイル一覧
# cd /var/log   リモートディレクトリ移動
# lcd ~/tmp     ローカルディレクトリ移動
# pwd           リモートの現在ディレクトリ
# lpwd          ローカルの現在ディレクトリ

# ===== ファイル転送 =====
# get remote_file.txt              ダウンロード
# get remote_file.txt local.txt    名前を変えてダウンロード
# get -r remote_dir/               ディレクトリごとダウンロード
# put local_file.txt               アップロード
# put local_file.txt remote.txt    名前を変えてアップロード
# put -r local_dir/                ディレクトリごとアップロード
# mget *.log                       ワイルドカードでダウンロード
# mput *.txt                       ワイルドカードでアップロード

# ===== ファイル操作 =====
# mkdir new_dir    ディレクトリ作成
# rmdir old_dir    ディレクトリ削除
# rm file.txt      ファイル削除
# rename old new   名前変更
# chmod 644 file   パーミッション変更
# chown uid file   オーナー変更

# ===== その他 =====
# !command    ローカルでコマンド実行
# df -h       リモートのディスク使用量
# quit        終了（exit, bye も同じ）

# バッチモード（非対話的）
sftp -b batch.txt user@server
# batch.txt:
# cd /var/log
# get *.log
# quit
```

---

## 5. SSHトンネリング（ポートフォワーディング）

### 5.1 ローカルフォワード

```bash
# ローカルフォワード: リモートのサービスをローカルポート経由でアクセス
# ssh -L [ローカルアドレス:]ローカルポート:リモートホスト:リモートポート user@sshサーバー

# 基本形
ssh -L 8080:localhost:80 user@server
# ローカルの8080 → SSHサーバーの80
# ブラウザで http://localhost:8080 → リモートの80番ポートに接続

# リモートDB への接続
ssh -L 5432:localhost:5432 user@server
# ローカルの5432 → リモートのPostgreSQL
# psql -h localhost -p 5432 でリモートDBに接続

ssh -L 3306:localhost:3306 user@server
# ローカルの3306 → リモートのMySQL

ssh -L 6379:localhost:6379 user@server
# ローカルの6379 → リモートのRedis

# 踏み台経由で内部サーバーに接続
ssh -L 3306:internal-db:3306 user@bastion
# ローカルの3306 → bastion経由 → internal-db:3306
# MySQL Workbench で localhost:3306 に接続 → 内部DBにアクセス

ssh -L 8443:internal-web:443 user@bastion
# ローカルの8443 → bastion経由 → 内部Webサーバーの443

# 複数ポートの同時フォワーディング
ssh -L 5432:db-server:5432 -L 6379:redis-server:6379 -L 9200:es-server:9200 user@bastion

# バインドアドレスの指定
ssh -L 0.0.0.0:8080:localhost:80 user@server
# → ローカルネットワークの他のマシンからもアクセス可能
# ※ セキュリティリスクに注意

ssh -L 127.0.0.1:8080:localhost:80 user@server
# → ローカルホストのみ（デフォルト）
```

### 5.2 リモートフォワード

```bash
# リモートフォワード: ローカルのサービスをリモートポート経由で公開
# ssh -R [リモートアドレス:]リモートポート:ローカルホスト:ローカルポート user@sshサーバー

# 基本形
ssh -R 8080:localhost:3000 user@server
# リモートの8080 → ローカルの3000
# 開発中のアプリをリモートから確認可能
# リモートで curl http://localhost:8080 → ローカルの3000に接続

# ユースケース:
# 1. 開発中のアプリをチームメンバーに見せる
ssh -R 8080:localhost:3000 user@shared-server

# 2. NAT/ファイアウォール内のサービスを外部に公開
ssh -R 0.0.0.0:9090:localhost:8080 user@public-server
# ※ sshd_config で GatewayPorts yes が必要

# 3. Webhook のテスト（ローカル開発サーバーで受信）
ssh -R 8080:localhost:4000 user@server
# Webhook の URL を http://server:8080/ に設定

# リモートフォワードの注意点:
# - デフォルトではリモートの localhost のみバインド
# - 外部からアクセスさせるには GatewayPorts の設定が必要
# - サーバー側の sshd_config: GatewayPorts yes
```

### 5.3 ダイナミックフォワード（SOCKSプロキシ）

```bash
# ダイナミックフォワード: SSH サーバーを SOCKS プロキシとして使用
# ssh -D [ローカルアドレス:]ローカルポート user@sshサーバー

ssh -D 1080 user@server
# ローカルの1080がSOCKSプロキシになる
# すべての通信がSSHサーバー経由になる

# ブラウザの設定:
# SOCKS Host: localhost
# Port: 1080
# SOCKS Version: SOCKS5

# curl でSOCKSプロキシを使用
curl --socks5 localhost:1080 https://example.com
curl --socks5-hostname localhost:1080 https://example.com
# --socks5-hostname: DNS解決もプロキシ経由（推奨）

# Git でSOCKSプロキシ経由
git -c "http.proxy=socks5://localhost:1080" clone https://github.com/user/repo

# ユースケース:
# - 特定の国からしかアクセスできないサービスに接続
# - 会社のネットワークから内部サービスにアクセス
# - 公共Wi-Fiでの通信を暗号化
```

### 5.4 バックグラウンドトンネル

```bash
# バックグラウンドでトンネルを張る
ssh -fNL 5432:localhost:5432 user@server
# -f: バックグラウンドに移行
# -N: コマンド実行しない（トンネルのみ）

# トンネルの確認
ps aux | grep "ssh -fN"
lsof -i :5432

# トンネルの終了
kill $(pgrep -f "ssh -fNL 5432")

# autossh: 自動再接続付きトンネル（推奨）
# brew install autossh / apt install autossh
autossh -M 0 -fNL 5432:localhost:5432 user@server \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3"
# -M 0: 接続監視にSSHのKeepAliveを使用
# 接続が切れると自動的に再接続

# systemd でトンネルを永続化（Linux）
# /etc/systemd/system/ssh-tunnel-db.service
# [Unit]
# Description=SSH Tunnel to Database
# After=network.target
#
# [Service]
# User=deploy
# ExecStart=/usr/bin/autossh -M 0 -NL 5432:localhost:5432 user@server \
#     -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
#     -o "ExitOnForwardFailure yes" -i /home/deploy/.ssh/tunnel_key
# Restart=always
# RestartSec=10
#
# [Install]
# WantedBy=multi-user.target
```

### 5.5 トンネリングの図解

```text
=== ローカルフォワード（-L） ===

[ローカルPC]                    [SSHサーバー]          [ターゲット]
  localhost:8080 ──SSH暗号化──→ sshd ──平文──→ internal-db:3306
  (ブラウザ等)                   (踏み台)

  コマンド: ssh -L 8080:internal-db:3306 user@sshserver

=== リモートフォワード（-R） ===

[ローカルPC]                    [SSHサーバー]
  localhost:3000 ←──SSH暗号化── sshd:8080
  (開発サーバー)                 (外部からアクセス可能)

  コマンド: ssh -R 8080:localhost:3000 user@sshserver

=== ダイナミックフォワード（-D） ===

[ローカルPC]                    [SSHサーバー]          [任意のサーバー]
  SOCKS5:1080 ──SSH暗号化──→ sshd ──平文──→ (どこでも)
  (ブラウザ等)

  コマンド: ssh -D 1080 user@sshserver
```

---

## 6. SSHサーバー設定（sshd_config）

### 6.1 セキュリティ強化設定

```bash
# /etc/ssh/sshd_config の推奨設定

# ===== 基本設定 =====
Port 22                        # ポート番号（変更推奨）
# Port 2222                    # 非標準ポートを使う場合
ListenAddress 0.0.0.0          # IPv4 で待ち受け
ListenAddress ::               # IPv6 で待ち受け

# ===== 認証設定 =====
PermitRootLogin no             # rootログイン禁止（必須）
PasswordAuthentication no      # パスワード認証無効化（鍵認証のみ）
PermitEmptyPasswords no        # 空パスワード禁止
PubkeyAuthentication yes       # 公開鍵認証を有効化
AuthorizedKeysFile .ssh/authorized_keys  # 公開鍵ファイルの場所

# ===== セキュリティ =====
MaxAuthTries 3                 # 認証試行回数の制限
MaxSessions 5                  # 最大セッション数
LoginGraceTime 30              # ログイン猶予時間（秒）
ClientAliveInterval 300        # クライアントの生存確認間隔
ClientAliveCountMax 2          # 生存確認の最大失敗回数

# ===== プロトコル設定 =====
Protocol 2                     # SSH2のみ（SSH1は廃止）
X11Forwarding no               # X11フォワーディング無効化
AllowTcpForwarding yes         # TCPフォワーディング許可
GatewayPorts no                # リモートフォワードのバインド制限
AllowAgentForwarding yes       # エージェントフォワーディング許可

# ===== ユーザー/グループ制限 =====
AllowUsers deploy admin        # 許可ユーザー
# DenyUsers testuser           # 拒否ユーザー
# AllowGroups sshusers         # 許可グループ
# DenyGroups noremote          # 拒否グループ

# ===== ログ =====
SyslogFacility AUTH
LogLevel VERBOSE               # 詳細ログ（監査用）

# ===== その他のセキュリティ =====
UsePAM yes                     # PAM認証を使用
PrintMotd no                   # ログイン時のメッセージ非表示
Banner /etc/ssh/banner.txt     # 接続時のバナー表示
AcceptEnv LANG LC_*            # 環境変数の受け渡し制限

# 設定変更後の反映
sudo sshd -t                   # 設定ファイルのテスト
sudo systemctl restart sshd    # sshd を再起動
# ※ 再起動前に必ず別ターミナルでの接続を確保しておくこと！
```

### 6.2 追加のセキュリティ対策

```bash
# ===== fail2ban による不正アクセス防止 =====
# sudo apt install fail2ban

# /etc/fail2ban/jail.local
# [sshd]
# enabled = true
# port = ssh
# filter = sshd
# logpath = /var/log/auth.log
# maxretry = 3
# bantime = 3600
# findtime = 600

# fail2ban の状態確認
sudo fail2ban-client status sshd

# ===== ファイアウォール（ufw）=====
sudo ufw allow 22/tcp          # SSHポートを許可
sudo ufw allow from 192.168.1.0/24 to any port 22  # 特定ネットワークのみ
sudo ufw enable

# ===== iptables =====
# 特定IPからのみSSH許可
sudo iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j DROP

# ===== SSHログの監視 =====
# 認証失敗の確認
sudo grep "Failed password" /var/log/auth.log | tail -20
sudo grep "Invalid user" /var/log/auth.log | tail -20

# 成功した認証の確認
sudo grep "Accepted" /var/log/auth.log | tail -20

# リアルタイム監視
sudo tail -f /var/log/auth.log | grep sshd

# ===== 二要素認証（Google Authenticator）=====
# sudo apt install libpam-google-authenticator
# google-authenticator  # 初期設定
# /etc/pam.d/sshd に追加:
# auth required pam_google_authenticator.so
# /etc/ssh/sshd_config:
# ChallengeResponseAuthentication yes
# AuthenticationMethods publickey,keyboard-interactive
```

---

## 7. 実践パターン

### 7.1 多段SSH（踏み台サーバー経由）

```bash
# ===== ~/.ssh/config での設定 =====
Host bastion
    HostName bastion.example.com
    User admin
    IdentityFile ~/.ssh/bastion_key

Host target
    HostName 10.0.1.100
    User admin
    ProxyJump bastion

# ssh target だけで接続可能

# ===== 多段ファイル転送 =====
# 踏み台経由の rsync
rsync -avz -e "ssh -J bastion" ./data/ admin@10.0.1.100:/data/

# 踏み台経由の scp
scp -o ProxyJump=bastion file.txt admin@10.0.1.100:/tmp/

# ===== 多段トンネル =====
# bastion → internal-server のDBに接続
ssh -J bastion -L 5432:localhost:5432 admin@10.0.1.100
```

### 7.2 リモートサーバーの監視

```bash
# リモートサーバーのログをリアルタイム監視
ssh user@server "tail -f /var/log/app.log"

# 複数サーバーのログを同時監視（tmux/screen使用推奨）
# tmux:
# Ctrl-b " → 画面分割
# 各ペインで ssh user@web1 "tail -f /var/log/app.log"

# リモートのシステム情報をワンコマンドで取得
ssh user@server bash <<'EOF'
echo "=== $(hostname) ==="
echo ""
echo "--- Uptime ---"
uptime
echo ""
echo "--- CPU ---"
top -bn1 | head -5
echo ""
echo "--- Memory ---"
free -h
echo ""
echo "--- Disk ---"
df -h | grep -v tmpfs
echo ""
echo "--- Network ---"
ss -tlnp
EOF

# 複数サーバーの一括ヘルスチェック
#!/bin/bash
SERVERS=("web1" "web2" "web3" "db1" "db2")

for server in "${SERVERS[@]}"; do
    echo -n "$server: "
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$server" "uptime" 2>/dev/null; then
        :  # OK
    else
        echo "UNREACHABLE"
    fi
done
```

### 7.3 ファイル差分比較

```bash
# リモートとローカルでファイル差分比較
diff <(ssh user@server "cat /etc/nginx/nginx.conf") ./nginx.conf

# カラー付き差分
diff --color <(ssh user@server "cat /etc/nginx/nginx.conf") ./nginx.conf

# vimdiff で比較
vimdiff <(ssh user@server "cat /etc/nginx/nginx.conf") ./nginx.conf

# 複数サーバー間の設定差分
diff <(ssh web1 "cat /etc/nginx/nginx.conf") <(ssh web2 "cat /etc/nginx/nginx.conf")

# ディレクトリ全体の差分（rsync ドライラン）
rsync -avzn user@server:/etc/nginx/ ./nginx-local/ 2>&1 | head -50
```

### 7.4 SSH接続のデバッグ

```bash
# 段階的なデバッグ
ssh -v user@server    # 基本的な接続情報
ssh -vv user@server   # 鍵の試行順序など
ssh -vvv user@server  # 全詳細（パケットレベル）

# よくある問題のデバッグ出力例:

# 問題1: 鍵が見つからない
# debug1: Trying private key: /home/user/.ssh/id_rsa
# debug1: Trying private key: /home/user/.ssh/id_ecdsa
# debug1: Trying private key: /home/user/.ssh/id_ed25519
# debug1: No more authentication methods to try.
# → 解決: 正しい鍵を -i で指定 or ~/.ssh/config で設定

# 問題2: パーミッションエラー
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# → 解決: chmod 600 ~/.ssh/id_ed25519

# 問題3: ホスト鍵の変更
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# → 解決: ssh-keygen -R hostname（正当な変更であることを確認後）

# 問題4: Connection refused
# ssh: connect to host server.com port 22: Connection refused
# → 確認: サーバー側で sshd が起動しているか
#   sudo systemctl status sshd
#   sudo ss -tlnp | grep 22

# 問題5: Connection timed out
# ssh: connect to host server.com port 22: Connection timed out
# → 確認: ファイアウォール、セキュリティグループ、ネットワーク経路

# 接続テスト（タイムアウト付き）
ssh -o ConnectTimeout=5 -o BatchMode=yes user@server "echo OK" 2>/dev/null
echo "Result: $?"
```

### 7.5 定期的なバックアップ

```bash
#!/bin/bash
# remote_backup.sh - rsync を使った定期バックアップスクリプト

# 設定
REMOTE_USER="deploy"
REMOTE_HOST="server.example.com"
REMOTE_DIR="/var/www/app/"
LOCAL_BACKUP_DIR="/backup/app"
RETENTION_DAYS=30
LOG_FILE="/var/log/backup.log"
SSH_KEY="/home/deploy/.ssh/backup_key"

# 日付
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${LOCAL_BACKUP_DIR}/${DATE}"
LATEST_LINK="${LOCAL_BACKUP_DIR}/latest"

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# バックアップ開始
log "バックアップ開始: ${REMOTE_HOST}:${REMOTE_DIR}"

# ディレクトリ作成
mkdir -p "$BACKUP_DIR"

# rsync 実行（ハードリンクで差分バックアップ）
rsync -avz --delete \
    --link-dest="$LATEST_LINK" \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='*.log' \
    --exclude='tmp/' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" \
    "$BACKUP_DIR/"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    # latest シンボリックリンクを更新
    rm -f "$LATEST_LINK"
    ln -s "$BACKUP_DIR" "$LATEST_LINK"
    log "バックアップ成功: $BACKUP_DIR"

    # バックアップサイズ
    SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
    log "バックアップサイズ: $SIZE"
else
    log "バックアップ失敗: exit code $EXIT_CODE"
    rm -rf "$BACKUP_DIR"  # 失敗したバックアップを削除
fi

# 古いバックアップの削除
log "古いバックアップの削除（${RETENTION_DAYS}日以前）"
find "$LOCAL_BACKUP_DIR" -maxdepth 1 -type d -name "20*" \
    -mtime +${RETENTION_DAYS} -exec rm -rf {} \;

# バックアップ一覧
log "現在のバックアップ一覧:"
ls -la "$LOCAL_BACKUP_DIR" | tee -a "$LOG_FILE"

log "バックアップ処理完了"
exit $EXIT_CODE
```

### 7.6 サーバー初期設定の自動化

```bash
#!/bin/bash
# server_setup.sh - 新規サーバーの初期設定スクリプト

set -euo pipefail

SERVER="${1:?使い方: $0 user@server}"

echo "=== サーバー初期設定: $SERVER ==="

# 1. SSH鍵の登録
echo "--- SSH鍵の登録 ---"
ssh-copy-id -i ~/.ssh/id_ed25519.pub "$SERVER"

# 2. 基本パッケージのインストール
echo "--- 基本パッケージのインストール ---"
ssh -t "$SERVER" bash <<'SETUP'
set -e

# パッケージ更新
sudo apt update && sudo apt upgrade -y

# 基本ツール
sudo apt install -y \
    vim htop tmux git curl wget \
    jq tree unzip \
    fail2ban ufw

# fail2ban 設定
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# UFW ファイアウォール
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw --force enable

# SSH セキュリティ強化
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

echo "=== 初期設定完了 ==="
SETUP

echo "=== サーバー初期設定が完了しました ==="
echo "接続テスト: ssh $SERVER"
```

### 7.7 SSH を使ったリモートスクリプト配布・実行

```bash
# ===== 方法1: パイプで直接実行 =====
ssh user@server bash < local_script.sh

# 引数付き
ssh user@server "bash -s arg1 arg2" < local_script.sh

# ===== 方法2: ヒアドキュメント =====
ssh user@server bash <<'EOF'
#!/bin/bash
echo "Running on: $(hostname)"
echo "Date: $(date)"
echo "Uptime: $(uptime)"
EOF

# ===== 方法3: scp + ssh =====
scp script.sh user@server:/tmp/
ssh user@server "chmod +x /tmp/script.sh && /tmp/script.sh && rm /tmp/script.sh"

# ===== 方法4: tar パイプ（複数ファイル転送 + 実行）=====
tar czf - scripts/ | ssh user@server "cd /tmp && tar xzf - && bash scripts/setup.sh"

# ===== 複数サーバーへの一括実行 =====
deploy_to_all() {
    local script=$1
    shift
    local servers=("$@")

    for server in "${servers[@]}"; do
        echo "=== Deploying to $server ==="
        ssh "$server" bash < "$script" &
    done
    wait
    echo "=== All deployments complete ==="
}

deploy_to_all setup.sh web1 web2 web3
```

---

## 8. 高度なSSH活用

### 8.1 SSH over WebSocket / HTTP

```bash
# ファイアウォールでSSHポートがブロックされている場合の対処法

# 方法1: HTTPSポート経由でSSH接続
# サーバー側: sshd を443ポートでも待ち受け
# Port 22
# Port 443

# 方法2: sslh（ポートマルチプレクサ）
# 同じポートでSSHとHTTPSを振り分ける
# sudo apt install sslh

# 方法3: ProxyCommand with corkscrew（HTTPプロキシ経由）
# brew install corkscrew
Host behind-proxy
    HostName server.example.com
    User admin
    ProxyCommand corkscrew proxy.company.com 8080 %h %p

# 方法4: ngrok でSSHトンネル公開
# ngrok tcp 22
# → tcp://0.tcp.ngrok.io:12345 のようなURLが生成される
# ssh -p 12345 user@0.tcp.ngrok.io
```

### 8.2 SSH証明書認証

```bash
# SSH証明書: authorized_keys を使わない集中管理型認証

# 1. CA鍵の生成
ssh-keygen -t ed25519 -f ~/.ssh/ca_key -C "SSH CA Key"

# 2. ユーザー鍵への署名（証明書発行）
ssh-keygen -s ~/.ssh/ca_key \
    -I "gaku@example.com" \
    -n gaku,admin \
    -V +52w \
    ~/.ssh/id_ed25519.pub
# -s: CA秘密鍵
# -I: 証明書の識別名
# -n: 許可されるプリンシパル（ユーザー名）
# -V: 有効期限（+52w = 52週間）
# 結果: ~/.ssh/id_ed25519-cert.pub が生成される

# 3. 証明書の確認
ssh-keygen -L -f ~/.ssh/id_ed25519-cert.pub

# 4. サーバー側の設定（/etc/ssh/sshd_config）
# TrustedUserCAKeys /etc/ssh/ca_key.pub
# → CA が署名した全ての鍵を信頼

# メリット:
# - authorized_keys の管理が不要
# - 有効期限付きで自動失効
# - 証明書の取り消し（revoked_keys）が可能
# - 大規模環境での鍵管理が容易
```

### 8.3 Mosh（Mobile Shell）

```bash
# Mosh: SSH の代替（モバイル/不安定回線向け）
# brew install mosh / apt install mosh

# 接続
mosh user@server
mosh --ssh="ssh -p 2222" user@server

# Mosh の特徴:
# - UDP ベース（ネットワーク切断後も自動再接続）
# - ローミング対応（IPアドレスが変わっても継続）
# - 即時のローカルエコー（入力のレイテンシが低い）
# - SSH で認証後、UDP に切り替え

# 制限:
# - ポートフォワーディング非対応
# - X11 フォワーディング非対応
# - スクロールバック履歴が限定的
# - UDP 60000-61000 ポートの開放が必要
```

### 8.4 tmux / screen との連携

```bash
# SSH切断後もプロセスを維持するための tmux 活用

# 新しい名前付きセッションを作成
ssh user@server -t "tmux new-session -s work"

# 既存セッションに再接続
ssh user@server -t "tmux attach-session -t work || tmux new-session -s work"

# デタッチ: Ctrl-b d（セッションを維持したまま切断）
# SSH が切れても tmux セッション内のプロセスは継続

# ~/.bashrc にエイリアス設定
# alias sshwork='ssh user@server -t "tmux attach-session -t work || tmux new-session -s work"'

# screen の場合
ssh user@server -t "screen -dR work"
```

---

## 9. トラブルシューティング

### 9.1 接続問題のフローチャート

```text
SSH接続失敗
│
├─ "Connection refused"
│  ├─ sshd が起動していない → sudo systemctl start sshd
│  ├─ ポートが違う → ssh -p PORT user@server
│  └─ ファイアウォール → sudo ufw allow 22
│
├─ "Connection timed out"
│  ├─ ネットワーク到達不可 → ping server
│  ├─ ファイアウォール → セキュリティグループ確認
│  └─ ルーティング → traceroute server
│
├─ "Permission denied"
│  ├─ 鍵が登録されていない → ssh-copy-id user@server
│  ├─ パーミッション不正 → chmod 600 ~/.ssh/id_ed25519
│  ├─ ユーザーが存在しない → サーバー側で確認
│  └─ sshd設定で拒否 → AllowUsers/DenyUsers 確認
│
├─ "Host key verification failed"
│  ├─ ホスト鍵が変わった → ssh-keygen -R hostname
│  └─ 中間者攻撃の可能性 → ホスト鍵を管理者に確認
│
├─ "Too many authentication failures"
│  ├─ 鍵が多すぎる → ssh -o IdentitiesOnly=yes -i KEY user@server
│  └─ MaxAuthTries に達した → 管理者に確認
│
└─ 接続後すぐに切断
   ├─ シェルが設定されていない → /etc/passwd 確認
   ├─ /etc/nologin が存在 → ファイルを削除
   └─ ディスク容量不足 → df -h 確認
```

### 9.2 よくあるエラーと解決策

```bash
# ===== エラー1: WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! =====
# 原因: サーバーのホスト鍵が変更された（再インストール、IPアドレス変更等）
# 解決:
ssh-keygen -R server.example.com
ssh-keygen -R 192.168.1.100
# ※ 中間者攻撃の可能性がないことを確認してから実行！

# ===== エラー2: Permission denied (publickey) =====
# 原因: 公開鍵認証に失敗
# 確認手順:
ssh -vvv user@server 2>&1 | grep -A5 "Trying\|Offering\|Authentication"
# 解決:
# 1. 鍵が正しく登録されているか確認
ssh user@server "cat ~/.ssh/authorized_keys"
# 2. パーミッションを確認
ssh user@server "ls -la ~/.ssh/ && ls -la ~/.ssh/authorized_keys"
# 3. ローカルの鍵パーミッション
ls -la ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519

# ===== エラー3: ssh_exchange_identification: Connection closed =====
# 原因: sshd が接続を拒否
# 確認:
# 1. /etc/hosts.allow, /etc/hosts.deny
# 2. MaxStartups 設定（同時接続数制限）
# 3. fail2ban でブロックされている
sudo fail2ban-client status sshd
sudo fail2ban-client set sshd unbanip 192.168.1.100

# ===== エラー4: broken pipe / Write failed =====
# 原因: 接続が切れた
# 解決（~/.ssh/config）:
# Host *
#     ServerAliveInterval 60
#     ServerAliveCountMax 3
#     TCPKeepAlive yes

# ===== エラー5: Agent forwarding でリモートから接続できない =====
# 確認:
echo "$SSH_AUTH_SOCK"   # 空なら agent が動いていない
ssh-add -l              # 鍵がロードされているか確認
# 解決:
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh -A user@bastion     # -A を忘れないこと
```

### 9.3 パフォーマンスチューニング

```bash
# ===== 接続の高速化 =====

# 1. ControlMaster で接続を多重化
Host *
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600

# 2. 暗号アルゴリズムの指定（高速なものを優先）
Host fast-server
    Ciphers aes128-gcm@openssh.com,aes256-gcm@openssh.com,chacha20-poly1305@openssh.com

# 3. 圧縮の有効/無効
Host slow-network
    Compression yes       # 遅い回線では有効化

Host fast-network
    Compression no        # 速い回線では無効化（CPU節約）

# 4. AddressFamily の制限（IPv4のみで高速化）
Host *
    AddressFamily inet    # IPv6解決の待ち時間を省略

# ===== rsync の高速化 =====

# 圧縮レベルの調整
rsync -avz --compress-level=1 ./data/ user@server:/data/
# レベル1: 高速（CPUボトルネック時に有効）

# 暗号化の軽量化（信頼できるネットワーク内のみ）
rsync -avz -e "ssh -c aes128-gcm@openssh.com" ./data/ user@server:/data/

# 並列rsync（大量の小さいファイル向け）
find ./data -maxdepth 1 -type d | xargs -P 4 -I {} \
    rsync -avz {}/ user@server:/data/{}/
```

---

## 10. SSH 関連コマンド一覧

### 10.1 コマンドリファレンス

```text
┌────────────────────┬─────────────────────────────────────────────────────┐
│ コマンド           │ 用途                                              │
├────────────────────┼─────────────────────────────────────────────────────┤
│ ssh                │ リモートサーバーへの接続                          │
│ ssh-keygen         │ SSH鍵の生成・管理                                 │
│ ssh-copy-id        │ 公開鍵をリモートに登録                            │
│ ssh-add            │ ssh-agent に鍵を登録                              │
│ ssh-agent          │ 鍵エージェントの起動・管理                        │
│ ssh-keyscan        │ リモートホストの公開鍵を取得                      │
│ scp                │ SSH経由のファイルコピー                           │
│ sftp               │ 対話的ファイル転送                                │
│ rsync              │ 差分ファイル同期                                  │
│ sshd               │ SSHサーバーデーモン                               │
│ sshd -t            │ sshd設定ファイルのテスト                          │
│ autossh            │ 自動再接続付きSSH接続                             │
│ mosh               │ モバイル向けSSH代替                               │
│ ssh-audit          │ SSHサーバーのセキュリティ監査                     │
└────────────────────┴─────────────────────────────────────────────────────┘
```

### 10.2 主要オプション一覧

```text
===== ssh のオプション =====
-p PORT        ポート指定
-i KEY         秘密鍵ファイル指定
-l USER        ユーザー名指定（user@host の代わり）
-v/-vv/-vvv    デバッグ出力レベル
-C             圧縮有効化
-X/-Y          X11フォワーディング
-A             エージェントフォワーディング
-N             コマンド実行しない（トンネル用）
-f             バックグラウンド実行
-t             疑似端末を強制割り当て
-L             ローカルポートフォワード
-R             リモートポートフォワード
-D             ダイナミックフォワード（SOCKSプロキシ）
-J             ProxyJump（踏み台サーバー）
-o OPTION      設定オプション指定
-W host:port   標準入出力をリモートに転送

===== ssh-keygen のオプション =====
-t TYPE        鍵タイプ（ed25519, rsa, ecdsa）
-b BITS        鍵のビット数（RSA: 4096推奨）
-f FILE        出力ファイル名
-C COMMENT     コメント
-N PASS        パスフレーズ
-p             パスフレーズ変更
-l             フィンガープリント表示
-R HOST        known_hosts からホスト削除
-y             秘密鍵から公開鍵を出力
-s CA_KEY      証明書に署名
-L             証明書情報の表示

===== rsync のオプション =====
-a             アーカイブモード（-rlptgoD）
-v             詳細表示
-z             圧縮転送
-n             ドライラン
-P             --partial --progress
--delete       送信元にないファイルを削除
--exclude      除外パターン
--include      包含パターン
--exclude-from ファイルから除外パターン読み込み
--link-dest    ハードリンクベースの差分バックアップ
--bwlimit      帯域制限
--stats        統計情報表示
-e "ssh ..."   SSH オプション指定
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| ssh user@host | リモート接続 |
| ssh -v user@host | デバッグ接続 |
| ssh-keygen -t ed25519 | 鍵ペア生成 |
| ssh-copy-id user@host | 公開鍵登録 |
| ssh-add key | エージェントに鍵登録 |
| ~/.ssh/config | 接続設定管理 |
| rsync -avz src dst | 差分ファイル転送 |
| rsync -avzn src dst | ドライラン |
| rsync --delete | 完全同期 |
| scp file user@host:/path | ファイルコピー |
| sftp user@host | 対話的ファイル転送 |
| ssh -L local:host:remote | ローカルポートフォワード |
| ssh -R remote:host:local | リモートポートフォワード |
| ssh -D 1080 user@host | SOCKSプロキシ |
| ssh -J bastion target | 踏み台経由接続 |
| autossh -M 0 -fNL ... | 自動再接続トンネル |

---

## よくある質問（FAQ）

### Q1: Ed25519 と RSA、どちらを使うべき？

Ed25519 を推奨する。理由は以下の通り：
- 鍵が短い（RSA 4096bit: 約750文字 vs Ed25519: 約68文字）
- 署名・検証が高速
- セキュリティ強度が高い（128bit相当 vs RSA 4096の約128bit相当）
- 実装が安全（サイドチャネル攻撃に強い）

ただし、古いシステム（OpenSSH 6.5未満）では Ed25519 が使えないため、その場合は RSA 4096bit を使用する。

### Q2: ssh-agent と Keychain の違いは？

ssh-agent はメモリ上にパスフレーズを保持する一時的な仕組み。ログアウトすると消える。macOS の Keychain はシステムレベルのパスワード管理で、再起動後も保持される。`UseKeychain yes` と `AddKeysToAgent yes` を ~/.ssh/config に設定すれば両方を連携できる。

### Q3: SSH接続が頻繁に切れる場合の対処法は？

```bash
# ~/.ssh/config に以下を追加
Host *
    ServerAliveInterval 60   # 60秒ごとにキープアライブ送信
    ServerAliveCountMax 3    # 3回失敗で切断
    TCPKeepAlive yes         # TCP レベルのキープアライブ
```

さらに不安定な場合は、autossh や mosh の使用を検討する。

### Q4: 踏み台サーバー経由で rsync するには？

```bash
# ProxyJump オプションを使用
rsync -avz -e "ssh -J bastion.example.com" ./data/ user@10.0.1.100:/data/

# または ~/.ssh/config に設定
# Host internal
#     HostName 10.0.1.100
#     User admin
#     ProxyJump bastion
# その後:
rsync -avz ./data/ internal:/data/
```

### Q5: パスワード認証を無効にしても大丈夫？

鍵認証が正しく設定されていれば問題ない。ただし、以下を事前に確認すること：

1. 少なくとも1つの鍵ペアで接続テスト済み
2. 別のターミナルで接続を維持したまま sshd を再起動
3. コンソールアクセス（VPS管理画面等）が使える状態であること

### Q6: scp と rsync の使い分けは？

| 項目 | scp | rsync |
|------|-----|-------|
| 差分転送 | 不可 | 可能 |
| 中断再開 | 不可 | 可能（--partial） |
| 除外パターン | 不可 | 可能 |
| 帯域制限 | 限定的 | 可能 |
| シンボリックリンク | 限定的 | 完全対応 |
| 削除同期 | 不可 | 可能（--delete） |
| 推奨度 | 簡単なコピー | 全般的に推奨 |

---

## 次に読むべきガイド
→ [[../05-shell-scripting/00-basics.md]] — シェルスクリプト基礎

---

## 参考文献
1. Barrett, D. "SSH, The Secure Shell: The Definitive Guide." 2nd Ed, O'Reilly, 2005.
2. "OpenSSH Manual Pages." openssh.com.
3. Stahnke, M. "Pro OpenSSH." Apress, 2005.
4. "SSH Hardening Guides." mozilla.github.io/openssh.
5. "rsync man page." linux.die.net/man/1/rsync.
