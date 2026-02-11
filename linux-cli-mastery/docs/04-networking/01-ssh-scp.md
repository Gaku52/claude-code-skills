# リモート接続（SSH, SCP, rsync）

> SSH はリモートサーバー操作の基盤。安全な接続・ファイル転送・トンネリングを可能にする。

## この章で学ぶこと

- [ ] SSH で安全にリモートサーバーに接続できる
- [ ] SSH鍵の生成・管理ができる
- [ ] SCP / rsync でファイル転送ができる
- [ ] SSHトンネリングを活用できる

---

## 1. SSH の基本

```bash
# 基本: ssh [オプション] [ユーザー@]ホスト

# 接続
ssh user@server.example.com       # ユーザー指定で接続
ssh server.example.com            # 現在のユーザー名で接続
ssh -p 2222 user@server.com       # ポート指定
ssh user@server.com command       # リモートでコマンド実行して切断

# リモートコマンド実行
ssh user@server "ls -la /var/log"
ssh user@server "df -h && free -m"
ssh user@server 'cat /etc/nginx/nginx.conf'

# 複数サーバーで同じコマンド実行
for server in web1 web2 web3; do
    echo "=== $server ==="
    ssh "$server" "uptime"
done
```

---

## 2. SSH鍵の管理

```bash
# 鍵ペアの生成
ssh-keygen -t ed25519 -C "gaku@example.com"
# Ed25519: 現在の推奨アルゴリズム（高速・安全・鍵が短い）

ssh-keygen -t rsa -b 4096 -C "gaku@example.com"
# RSA 4096bit: レガシー環境向け

# 鍵の保存場所
# ~/.ssh/id_ed25519      ← 秘密鍵（絶対に共有しない）
# ~/.ssh/id_ed25519.pub  ← 公開鍵（サーバーに登録する）

# 公開鍵をリモートサーバーに登録
ssh-copy-id user@server.com
# または手動で:
cat ~/.ssh/id_ed25519.pub | ssh user@server "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# パスフレーズ付き鍵の場合 → ssh-agent で管理
eval "$(ssh-agent -s)"           # エージェント起動
ssh-add ~/.ssh/id_ed25519        # 鍵をエージェントに登録
ssh-add -l                       # 登録済み鍵の一覧
ssh-add -D                       # 全鍵を削除

# macOS: Keychainと統合
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

### SSH鍵のセキュリティ

```bash
# パーミッション（重要）
chmod 700 ~/.ssh                 # ディレクトリ
chmod 600 ~/.ssh/id_ed25519      # 秘密鍵
chmod 644 ~/.ssh/id_ed25519.pub  # 公開鍵
chmod 600 ~/.ssh/authorized_keys # 認証鍵リスト
chmod 600 ~/.ssh/config          # 設定ファイル

# パーミッションが正しくないと SSH は鍵を拒否する
```

---

## 3. SSH設定ファイル（~/.ssh/config）

```bash
# ~/.ssh/config で接続設定を管理
# 長いコマンドを短いエイリアスにできる

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

# 踏み台サーバー経由（ProxyJump）
Host internal-server
    HostName 192.168.1.100
    User admin
    ProxyJump bastion.example.com

# 全ホスト共通設定
Host *
    ServerAliveInterval 60       # 60秒ごとにKeepAlive
    ServerAliveCountMax 3        # 3回失敗で切断
    AddKeysToAgent yes           # 鍵を自動でエージェントに追加
    IdentitiesOnly yes           # 指定した鍵のみ使用
```

---

## 4. ファイル転送（SCP, rsync）

### SCP

```bash
# scp: SSH経由のファイルコピー（非推奨化の流れ → rsync推奨）

# ローカル → リモート
scp file.txt user@server:/home/user/
scp -r ./dir user@server:/home/user/       # ディレクトリ（再帰）
scp -P 2222 file.txt user@server:/tmp/     # ポート指定

# リモート → ローカル
scp user@server:/var/log/app.log ./
scp -r user@server:/home/user/dir ./

# リモート → リモート
scp user@server1:/file user@server2:/file
```

### rsync（推奨）

```bash
# rsync: 差分転送（高速・中断再開可能）

# 基本構文: rsync [オプション] 送信元 送信先

# ローカル → リモート
rsync -avz ./project/ user@server:/home/user/project/
# -a: アーカイブモード（パーミッション・タイムスタンプ保持）
# -v: 詳細表示
# -z: 圧縮転送

# リモート → ローカル
rsync -avz user@server:/var/log/ ./logs/

# 重要: 末尾のスラッシュの意味
rsync -avz ./dir  user@server:/dest/   # → /dest/dir/ として転送
rsync -avz ./dir/ user@server:/dest/   # → /dest/ の中身として転送

# 除外パターン
rsync -avz --exclude='.git' --exclude='node_modules' ./project/ user@server:/app/
rsync -avz --exclude-from='.rsyncignore' ./project/ user@server:/app/

# ドライラン（実行せずに確認）
rsync -avzn ./project/ user@server:/app/   # -n: ドライラン

# 削除同期（送信元にないファイルを削除先からも削除）
rsync -avz --delete ./project/ user@server:/app/

# 帯域制限
rsync -avz --bwlimit=5000 ./large/ user@server:/backup/  # 5MB/s
```

### sftp

```bash
# sftp: 対話的ファイル転送
sftp user@server
# sftp> ls                       # リモートのファイル一覧
# sftp> lls                      # ローカルのファイル一覧
# sftp> get remote_file.txt      # ダウンロード
# sftp> put local_file.txt       # アップロード
# sftp> cd /var/log              # リモートディレクトリ移動
# sftp> quit                     # 終了
```

---

## 5. SSHトンネリング（ポートフォワーディング）

```bash
# ローカルフォワード（リモートのポートをローカルに転送）
ssh -L 8080:localhost:80 user@server
# ローカルの8080 → リモートの80
# ブラウザで http://localhost:8080 → リモートの80番ポートに接続

# 実用例: リモートDBへの接続
ssh -L 5432:localhost:5432 user@server
# ローカルの5432 → リモートのPostgreSQL

# 踏み台経由で内部サーバーに接続
ssh -L 3306:internal-db:3306 user@bastion
# ローカルの3306 → bastion経由 → internal-db:3306

# リモートフォワード（ローカルのポートをリモートに転送）
ssh -R 8080:localhost:3000 user@server
# リモートの8080 → ローカルの3000
# 開発中のアプリをリモートから確認可能

# ダイナミックフォワード（SOCKSプロキシ）
ssh -D 1080 user@server
# ローカルの1080がSOCKSプロキシになる
# ブラウザの設定で SOCKS5 localhost:1080 を指定

# バックグラウンドでトンネルを張る
ssh -fNL 5432:localhost:5432 user@server
# -f: バックグラウンド  -N: コマンド実行しない
```

---

## 6. 実践パターン

```bash
# パターン1: 多段SSH（踏み台サーバー経由）
# ~/.ssh/config で設定
Host target
    HostName 10.0.1.100
    User admin
    ProxyJump bastion

# ssh target だけで接続可能

# パターン2: リモートサーバーのログをリアルタイム監視
ssh user@server "tail -f /var/log/app.log"

# パターン3: リモートとローカルでファイル差分比較
diff <(ssh user@server "cat /etc/nginx/nginx.conf") ./nginx.conf

# パターン4: SSH接続のデバッグ
ssh -vvv user@server             # 詳細デバッグ出力

# パターン5: 定期的なバックアップ
rsync -avz --delete \
  --exclude='.git' \
  --exclude='node_modules' \
  user@server:/var/www/app/ \
  /backup/app/$(date +%Y%m%d)/
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| ssh user@host | リモート接続 |
| ssh-keygen -t ed25519 | 鍵ペア生成 |
| ssh-copy-id user@host | 公開鍵登録 |
| ~/.ssh/config | 接続設定管理 |
| rsync -avz src dst | 差分ファイル転送 |
| ssh -L local:host:remote | ローカルポートフォワード |
| ssh -D 1080 user@host | SOCKSプロキシ |

---

## 次に読むべきガイド
→ [[../05-shell-scripting/00-basics.md]] — シェルスクリプト基礎

---

## 参考文献
1. Barrett, D. "SSH, The Secure Shell: The Definitive Guide." 2nd Ed, O'Reilly, 2005.
2. "OpenSSH Manual Pages." openssh.com.
