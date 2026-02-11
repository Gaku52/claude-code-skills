# systemd とサービス管理

> systemd は現代の Linux システムの中核。サービスの起動・停止・監視を統一的に管理する。

## この章で学ぶこと

- [ ] systemctl でサービスを管理できる
- [ ] journalctl でログを確認できる
- [ ] カスタムサービスユニットを作成できる

---

## 1. systemctl — サービス管理

```bash
# サービスの操作
sudo systemctl start nginx       # 起動
sudo systemctl stop nginx        # 停止
sudo systemctl restart nginx     # 再起動
sudo systemctl reload nginx      # 設定再読み込み（プロセス維持）
sudo systemctl status nginx      # 状態確認

# 自動起動の管理
sudo systemctl enable nginx      # OS起動時に自動起動
sudo systemctl disable nginx     # 自動起動を無効化
sudo systemctl enable --now nginx  # 有効化 + 即起動
sudo systemctl is-enabled nginx  # 自動起動の確認
sudo systemctl is-active nginx   # 稼働中か確認

# サービス一覧
systemctl list-units --type=service              # 稼働中のサービス
systemctl list-units --type=service --all         # 全サービス
systemctl list-units --type=service --failed      # 失敗したサービス
systemctl list-unit-files --type=service          # 全ユニットファイル

# 依存関係
systemctl list-dependencies nginx
systemctl list-dependencies --reverse nginx       # 逆方向（誰がnginxに依存しているか）
```

### status の読み方

```
● nginx.service - A high performance web server
     Loaded: loaded (/lib/systemd/system/nginx.service; enabled; preset: enabled)
     Active: active (running) since Mon 2025-01-01 00:00:00 JST; 30 days ago
       Docs: man:nginx(8)
   Main PID: 1234 (nginx)
      Tasks: 5 (limit: 4096)
     Memory: 12.5M
        CPU: 1min 23.456s
     CGroup: /system.slice/nginx.service
             ├─1234 "nginx: master process /usr/sbin/nginx"
             ├─1235 "nginx: worker process"

# Active の状態:
# active (running)  → 正常稼働中
# active (exited)   → 実行完了（ワンショット型）
# inactive (dead)   → 停止中
# failed            → 起動失敗
# activating        → 起動中
```

---

## 2. journalctl — ログ管理

```bash
# 全ログ
journalctl                       # 全システムログ
journalctl -f                    # リアルタイム監視（tail -f 相当）

# サービス別
journalctl -u nginx              # nginx のログ
journalctl -u nginx -f           # nginx のリアルタイムログ
journalctl -u nginx --since today  # 今日のログ

# 時間指定
journalctl --since "2025-01-01"
journalctl --since "2025-01-01" --until "2025-01-02"
journalctl --since "1 hour ago"
journalctl --since "30 minutes ago"

# 優先度（重要度）フィルタ
journalctl -p err                # エラー以上
journalctl -p warning            # 警告以上
# 優先度: emerg > alert > crit > err > warning > notice > info > debug

# ブート別
journalctl -b                    # 現在のブート
journalctl -b -1                 # 前回のブート
journalctl --list-boots          # ブート一覧

# 出力形式
journalctl -o json               # JSON形式
journalctl -o json-pretty        # 整形JSON
journalctl -o short-iso          # ISO時刻形式

# ディスク使用量
journalctl --disk-usage          # ログのディスク使用量
sudo journalctl --vacuum-size=500M  # 500MBまで削減
sudo journalctl --vacuum-time=30d   # 30日より古いログを削除

# カーネルログ
journalctl -k                    # カーネルメッセージ（dmesg相当）
```

---

## 3. ユニットファイルの作成

```bash
# ユニットファイルの配置場所
# /etc/systemd/system/          ← カスタムサービス（優先）
# /lib/systemd/system/          ← パッケージインストール（デフォルト）

# 基本的なサービスユニット
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application Server
Documentation=https://example.com/docs
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=myapp
Group=myapp
WorkingDirectory=/opt/myapp
Environment=NODE_ENV=production
EnvironmentFile=/opt/myapp/.env
ExecStart=/usr/bin/node /opt/myapp/server.js
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# セキュリティ設定
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/myapp/data

[Install]
WantedBy=multi-user.target

# Type の種類:
# simple:   ExecStart のプロセスがメインプロセス（デフォルト）
# forking:  デーモン化するプロセス（PIDFileと組み合わせ）
# oneshot:  1回実行して終了（セットアップスクリプト等）
# notify:   sd_notify()で準備完了を通知

# Restart の種類:
# no:          再起動しない
# on-failure:  異常終了時のみ再起動
# on-abnormal: シグナル/タイムアウト時
# always:      常に再起動
```

### ユニットファイルの反映

```bash
# 変更後の反映手順
sudo systemctl daemon-reload     # ユニットファイル再読み込み
sudo systemctl restart myapp     # サービス再起動
sudo systemctl status myapp      # 状態確認
journalctl -u myapp -f           # ログ確認
```

### タイマーユニット（cron の代替）

```bash
# /etc/systemd/system/backup.timer
[Unit]
Description=Daily backup timer

[Timer]
OnCalendar=daily
# OnCalendar=*-*-* 03:00:00     # 毎日3時
# OnCalendar=Mon *-*-* 09:00:00 # 毎週月曜9時
Persistent=true                  # 見逃した実行を起動後に実行

[Install]
WantedBy=timers.target

# /etc/systemd/system/backup.service
[Unit]
Description=Daily backup

[Service]
Type=oneshot
ExecStart=/opt/scripts/backup.sh

# タイマーの管理
sudo systemctl enable --now backup.timer
systemctl list-timers            # タイマー一覧
```

---

## 4. systemd の実践パターン

```bash
# パターン1: サービスの起動失敗を調査
sudo systemctl status myapp      # まず状態確認
journalctl -u myapp --since "10 minutes ago" --no-pager  # 直近ログ
journalctl -u myapp -p err       # エラーのみ

# パターン2: 複数サービスの一括管理
for svc in nginx postgresql redis; do
    echo "=== $svc ==="
    systemctl is-active "$svc"
done

# パターン3: リソース制限
# /etc/systemd/system/myapp.service.d/limits.conf
[Service]
MemoryMax=512M
CPUQuota=50%
TasksMax=100

# パターン4: 起動順序の確認
systemd-analyze                  # 起動時間
systemd-analyze blame            # サービス別起動時間
systemd-analyze critical-chain   # クリティカルパス

# パターン5: ユーザーサービス（root不要）
mkdir -p ~/.config/systemd/user/
# ~/.config/systemd/user/myapp.service を作成
systemctl --user start myapp
systemctl --user enable myapp
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| systemctl start/stop/restart | サービス操作 |
| systemctl enable/disable | 自動起動管理 |
| systemctl status | 状態確認 |
| journalctl -u service | サービスログ |
| journalctl -f | リアルタイムログ |
| systemctl daemon-reload | ユニット再読み込み |

---

## 次に読むべきガイド
→ [[01-package-management.md]] — パッケージ管理

---

## 参考文献
1. "systemd System and Service Manager." systemd.io.
2. Barrett, D. "Efficient Linux at the Command Line." Ch.10, O'Reilly, 2022.
