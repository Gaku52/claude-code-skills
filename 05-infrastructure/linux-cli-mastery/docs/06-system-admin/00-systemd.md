# systemd とサービス管理

> systemd は現代の Linux システムの中核。サービスの起動・停止・監視を統一的に管理する。

## この章で学ぶこと

- [ ] systemctl でサービスを管理できる
- [ ] journalctl でログを確認できる
- [ ] カスタムサービスユニットを作成できる
- [ ] タイマーユニットで定期実行を設定できる
- [ ] systemd のセキュリティ・リソース制限を設定できる
- [ ] トラブルシューティングの手法を理解する

---

## 1. systemctl — サービス管理

### 1.1 基本操作

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

### 1.2 status の読み方

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

### 1.3 status の各フィールドの詳細解説

```bash
# Loaded 行の読み方
# loaded (/lib/systemd/system/nginx.service; enabled; preset: enabled)
#   ↑ ユニットファイルのパス              ↑ 有効/無効  ↑ プリセット

# ユニットファイルのパスからどこに設定があるかわかる:
# /lib/systemd/system/   → パッケージが提供（デフォルト）
# /etc/systemd/system/   → 管理者がカスタマイズ（優先される）
# /run/systemd/system/   → ランタイム生成（再起動で消える）

# Active 行の読み方
# active (running) since Mon 2025-01-01 00:00:00 JST; 30 days ago
# ↑ 状態           ↑ 起動日時                          ↑ 経過時間

# Tasks: プロセス（スレッド）の数
# Memory: 使用メモリ量
# CPU: 累積CPU使用時間
# CGroup: コントロールグループ内のプロセスツリー
```

### 1.4 サービスのマスク・アンマスク

```bash
# マスク: サービスの起動を完全に禁止する（enable も start もできなくなる）
sudo systemctl mask nginx
# /dev/null へのシンボリックリンクが作られる

# マスクの状態確認
systemctl is-enabled nginx       # "masked" と表示される

# アンマスク: マスクを解除する
sudo systemctl unmask nginx

# マスクされたサービスの一覧
systemctl list-unit-files --state=masked

# マスクの用途:
# - 別のサービスと競合する場合（例: iptables と firewalld）
# - 誤って起動されるのを防ぎたい場合
# - 一時的にサービスを完全無効化したい場合
```

### 1.5 サービスの詳細情報

```bash
# ユニットファイルの内容を表示
systemctl cat nginx              # ユニットファイルの内容を表示
systemctl show nginx             # 全プロパティを表示
systemctl show nginx --property=MainPID   # 特定プロパティ
systemctl show nginx --property=ActiveState,SubState

# ユニットファイルの場所を確認
systemctl show nginx --property=FragmentPath
# FragmentPath=/lib/systemd/system/nginx.service

# ユニットファイルの編集
sudo systemctl edit nginx        # オーバーライドファイルを作成
# /etc/systemd/system/nginx.service.d/override.conf が作成される

sudo systemctl edit --full nginx # ユニットファイル全体を編集
# /etc/systemd/system/nginx.service にコピーが作られる

# 変更の確認
systemd-delta                    # オーバーライドされたユニットの一覧
systemd-delta --type=overridden  # オーバーライドのみ表示
```

---

## 2. journalctl — ログ管理

### 2.1 基本的なログ表示

```bash
# 全ログ
journalctl                       # 全システムログ
journalctl -f                    # リアルタイム監視（tail -f 相当）
journalctl -n 50                 # 最新50行
journalctl --no-pager            # ページャーなしで表示

# サービス別
journalctl -u nginx              # nginx のログ
journalctl -u nginx -f           # nginx のリアルタイムログ
journalctl -u nginx --since today  # 今日のログ
journalctl -u nginx -n 100      # nginx の最新100行

# 複数サービス
journalctl -u nginx -u php-fpm   # nginx と php-fpm のログ
```

### 2.2 時間指定によるフィルタ

```bash
# 時間指定
journalctl --since "2025-01-01"
journalctl --since "2025-01-01" --until "2025-01-02"
journalctl --since "1 hour ago"
journalctl --since "30 minutes ago"
journalctl --since "yesterday"
journalctl --since "2025-01-01 09:00:00" --until "2025-01-01 18:00:00"

# 相対時間の書式
journalctl --since "-2h"         # 2時間前から
journalctl --since "-7d"         # 7日前から
journalctl --since "today"       # 今日の0時から
journalctl --since "yesterday" --until "today"  # 昨日のログ
```

### 2.3 優先度（重要度）フィルタ

```bash
# 優先度フィルタ
journalctl -p err                # エラー以上
journalctl -p warning            # 警告以上
journalctl -p crit               # クリティカル以上
journalctl -p info               # info以上

# 優先度の一覧（番号順）:
# 0: emerg   → システムが使用不能
# 1: alert   → 即座に対処が必要
# 2: crit    → 致命的な状態
# 3: err     → エラー状態
# 4: warning → 警告状態
# 5: notice  → 正常だが注意すべき
# 6: info    → 情報メッセージ
# 7: debug   → デバッグメッセージ

# 範囲指定
journalctl -p err..crit          # エラーからクリティカルまで

# 特定サービスのエラーのみ
journalctl -u nginx -p err --since "1 hour ago"
```

### 2.4 ブート別ログ

```bash
# ブート別
journalctl -b                    # 現在のブート
journalctl -b -1                 # 前回のブート
journalctl -b -2                 # 2回前のブート
journalctl --list-boots          # ブート一覧

# 特定のブートIDを指定
journalctl -b abc123def          # ブートID指定

# 前回のブートでのエラーを確認（障害調査に便利）
journalctl -b -1 -p err
journalctl -b -1 -u nginx       # 前回のブートでのnginxログ
```

### 2.5 出力形式

```bash
# 出力形式
journalctl -o json               # JSON形式
journalctl -o json-pretty        # 整形JSON
journalctl -o short-iso          # ISO時刻形式
journalctl -o short-precise      # マイクロ秒精度
journalctl -o verbose            # 全フィールド表示
journalctl -o cat                # メッセージのみ（タイムスタンプなし）
journalctl -o export             # バイナリエクスポート形式

# JSON出力をjqで処理
journalctl -u nginx -o json --since "1 hour ago" | \
    jq -r 'select(.PRIORITY == "3") | .MESSAGE'

# フィールド指定
journalctl -u nginx --output-fields=MESSAGE,PRIORITY

# 特定フィールドの値一覧
journalctl -F _SYSTEMD_UNIT     # ユニット名一覧
journalctl -F _COMM              # コマンド名一覧
journalctl -F PRIORITY           # 使用されている優先度一覧
```

### 2.6 ディスク使用量とログ管理

```bash
# ディスク使用量
journalctl --disk-usage          # ログのディスク使用量

# ログの圧縮・削減
sudo journalctl --vacuum-size=500M  # 500MBまで削減
sudo journalctl --vacuum-time=30d   # 30日より古いログを削除
sudo journalctl --vacuum-files=5    # 5ファイルまで削減

# ログの永続設定（/etc/systemd/journald.conf）
# [Journal]
# Storage=persistent            # 永続化（/var/log/journal/に保存）
# SystemMaxUse=1G               # 最大1GB
# SystemMaxFileSize=100M        # 1ファイル最大100MB
# MaxRetentionSec=1month        # 最大保持期間
# MaxFileSec=1week              # ファイルのローテーション間隔
# Compress=yes                  # 圧縮有効
# RateLimitIntervalSec=30s      # レート制限の間隔
# RateLimitBurst=10000          # レート制限のバースト

# 設定変更の反映
sudo systemctl restart systemd-journald

# ログの転送設定
# ForwardToSyslog=yes           # syslogへ転送
# ForwardToConsole=no           # コンソールへの転送
# ForwardToWall=yes             # wall メッセージの転送
```

### 2.7 カーネルログとその他のフィルタ

```bash
# カーネルログ
journalctl -k                    # カーネルメッセージ（dmesg相当）
journalctl -k -p err             # カーネルのエラーメッセージ
journalctl -k --since "1 hour ago"

# PIDでフィルタ
journalctl _PID=1234

# UIDでフィルタ
journalctl _UID=1000

# 実行ファイルでフィルタ
journalctl _COMM=sshd            # sshdのログ
journalctl _EXE=/usr/sbin/sshd   # パス指定

# ホスト名でフィルタ（ネットワークログ受信時）
journalctl _HOSTNAME=webserver01

# 複合フィルタ
journalctl _SYSTEMD_UNIT=sshd.service _PID=1234

# ログの除外パターン（grepと組み合わせ）
journalctl -u nginx --no-pager | grep -v "GET /health"
```

---

## 3. ユニットファイルの作成

### 3.1 ユニットファイルの配置場所と優先度

```bash
# ユニットファイルの配置場所（優先度順）
# /etc/systemd/system/          ← カスタムサービス（最優先）
# /run/systemd/system/          ← ランタイム生成（再起動で消える）
# /lib/systemd/system/          ← パッケージインストール（デフォルト）
# /usr/lib/systemd/system/      ← ディストリビューション提供

# ユニットファイルの種類
# .service   → サービス（最も一般的）
# .timer     → タイマー（cron代替）
# .socket    → ソケットアクティベーション
# .mount     → マウントポイント
# .target    → ターゲット（グループ化）
# .path      → パス監視
# .device    → デバイス
# .swap      → スワップ
# .slice     → リソース管理グループ
# .scope     → 外部プロセスのグループ化
```

### 3.2 基本的なサービスユニット

```ini
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
```

### 3.3 Type の種類と詳細

```bash
# Type の種類と使い分け:

# simple（デフォルト）:
#   ExecStart のプロセスがメインプロセス
#   プロセスが開始した時点で「起動完了」とみなす
#   用途: Node.js, Python, Go などのサーバー

# exec（systemd 240+）:
#   simple と似ているが、ExecStart のバイナリが実行された時点で起動完了
#   simple より正確なタイミング判定

# forking:
#   デーモン化するプロセス（フォークして親が終了するタイプ）
#   PIDFile と組み合わせて使用
#   用途: Apache httpd, 伝統的なUNIXデーモン

# oneshot:
#   1回実行して終了するタスク
#   RemainAfterExit=yes と組み合わせると、終了後も active 状態を維持
#   用途: セットアップスクリプト、ファイアウォール設定

# notify:
#   sd_notify() で準備完了を通知するプロセス
#   NotifyAccess=main と組み合わせ
#   用途: systemd対応のアプリケーション

# dbus:
#   D-Busバス名を取得した時点で起動完了
#   BusName= と組み合わせ
#   用途: D-Bus対応サービス

# idle:
#   simple と同じだが、全ジョブが完了するまで実行を遅延
#   用途: ログイン後のセットアップ処理
```

### 3.4 Restart の種類と詳細

```bash
# Restart の種類:
# no:          再起動しない（デフォルト）
# on-success:  正常終了時（exit code 0）のみ再起動
# on-failure:  異常終了時のみ再起動（最も一般的）
# on-abnormal: シグナル/タイムアウト/ウォッチドッグ時に再起動
# on-watchdog: ウォッチドッグタイムアウト時のみ再起動
# on-abort:    シグナルによる異常終了時のみ再起動
# always:      常に再起動（停止されても再起動する）

# Restart 関連の設定
# RestartSec=5            → 再起動までの待機時間（秒）
# RestartSteps=5          → 段階的に待機時間を増加（systemd 254+）
# RestartMaxDelaySec=120  → 最大待機時間
# StartLimitIntervalSec=300  → この期間内での起動回数を制限
# StartLimitBurst=5       → 上記期間内の最大起動回数

# 実用的な再起動設定例
# [Service]
# Restart=on-failure
# RestartSec=5
# StartLimitIntervalSec=300
# StartLimitBurst=5
# → 5分間に5回まで再起動を試みる。超えると failed 状態になる
```

### 3.5 各種サービスの実例

```ini
# === Python (Gunicorn) Webアプリケーション ===
# /etc/systemd/system/gunicorn.service
[Unit]
Description=Gunicorn WSGI Server
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/webapp
Environment=PYTHONPATH=/opt/webapp
ExecStart=/opt/webapp/venv/bin/gunicorn \
    --workers 4 \
    --bind unix:/run/gunicorn.sock \
    --access-logfile - \
    --error-logfile - \
    wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=on-failure
RestartSec=5
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target

# === Java (Spring Boot) アプリケーション ===
# /etc/systemd/system/spring-app.service
[Unit]
Description=Spring Boot Application
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=spring
Group=spring
WorkingDirectory=/opt/spring-app
Environment=JAVA_OPTS="-Xmx512m -Xms256m"
Environment=SPRING_PROFILES_ACTIVE=production
ExecStart=/usr/bin/java $JAVA_OPTS -jar /opt/spring-app/app.jar
SuccessExitStatus=143
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target

# === Go バイナリ ===
# /etc/systemd/system/goapp.service
[Unit]
Description=Go Application
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=goapp
Group=goapp
ExecStart=/opt/goapp/server
Restart=always
RestartSec=3
LimitNOFILE=65536
Environment=GOMAXPROCS=4

# セキュリティ強化
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/opt/goapp/data
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target

# === フォーク型デーモン（Apache httpd） ===
# /etc/systemd/system/httpd-custom.service
[Unit]
Description=Custom Apache HTTP Server
After=network.target remote-fs.target nss-lookup.target

[Service]
Type=forking
PIDFile=/run/httpd/httpd.pid
ExecStartPre=/usr/sbin/apachectl configtest
ExecStart=/usr/sbin/apachectl start
ExecReload=/usr/sbin/apachectl graceful
ExecStop=/usr/sbin/apachectl graceful-stop
Restart=on-failure

[Install]
WantedBy=multi-user.target

# === ワンショット型（初期化スクリプト） ===
# /etc/systemd/system/app-init.service
[Unit]
Description=Application Initialization
Before=myapp.service
ConditionPathExists=!/opt/myapp/.initialized

[Service]
Type=oneshot
User=myapp
ExecStart=/opt/myapp/scripts/init.sh
ExecStartPost=/usr/bin/touch /opt/myapp/.initialized
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

### 3.6 ユニットファイルの反映

```bash
# 変更後の反映手順
sudo systemctl daemon-reload     # ユニットファイル再読み込み
sudo systemctl restart myapp     # サービス再起動
sudo systemctl status myapp      # 状態確認
journalctl -u myapp -f           # ログ確認

# ユニットファイルの構文チェック
systemd-analyze verify /etc/systemd/system/myapp.service

# ユニットファイルの依存関係を可視化
systemd-analyze dot nginx.service | dot -Tsvg > nginx-deps.svg
```

---

## 4. タイマーユニット（cron の代替）

### 4.1 基本的なタイマー

```ini
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
```

### 4.2 OnCalendar の書式

```bash
# OnCalendar の書式パターン
# 曜日 年-月-日 時:分:秒

# 具体例
OnCalendar=daily                     # 毎日 0:00
OnCalendar=weekly                    # 毎週月曜 0:00
OnCalendar=monthly                   # 毎月1日 0:00
OnCalendar=yearly                    # 毎年1月1日 0:00
OnCalendar=hourly                    # 毎時 0分

OnCalendar=*-*-* 03:00:00           # 毎日3時
OnCalendar=Mon *-*-* 09:00:00       # 毎週月曜9時
OnCalendar=*-*-01 00:00:00          # 毎月1日
OnCalendar=Mon,Fri *-*-* 08:30:00   # 月曜と金曜の8:30
OnCalendar=*-*-* *:00/15:00         # 15分ごと
OnCalendar=*-*-* 09..17:00:00       # 9時から17時の毎正時
OnCalendar=Sat,Sun *-*-* 10:00:00   # 土日の10時

# 書式の検証
systemd-analyze calendar "Mon *-*-* 09:00:00"
# 次の実行日時が表示される

systemd-analyze calendar "*-*-* *:00/30:00"
# 30分ごとの実行日時が表示される

# 相対時間でのタイマー
# [Timer]
# OnBootSec=5min                    # 起動5分後
# OnUnitActiveSec=1h                # 前回実行の1時間後
# OnActiveSec=30s                   # タイマー有効化の30秒後
# RandomizedDelaySec=5min           # ランダムな遅延（負荷分散用）
# AccuracySec=1min                  # 精度（デフォルト1分）
```

### 4.3 タイマーの管理

```bash
# タイマーの管理
sudo systemctl enable --now backup.timer
systemctl list-timers            # タイマー一覧
systemctl list-timers --all      # 非アクティブ含む

# タイマーの状態確認
systemctl status backup.timer    # タイマーの状態
systemctl status backup.service  # 対応サービスの状態

# 手動実行（テスト用）
sudo systemctl start backup.service  # タイマー経由ではなく直接実行

# 次の実行日時を確認
systemctl list-timers backup.timer
```

### 4.4 タイマーの実践例

```ini
# === ログローテーション（毎日2時） ===
# /etc/systemd/system/log-rotate.timer
[Unit]
Description=Daily log rotation

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
RandomizedDelaySec=5min

[Install]
WantedBy=timers.target

# /etc/systemd/system/log-rotate.service
[Unit]
Description=Rotate application logs

[Service]
Type=oneshot
ExecStart=/opt/scripts/rotate-logs.sh
Nice=19
IOSchedulingClass=idle

# === SSL証明書更新チェック（毎日2回） ===
# /etc/systemd/system/certbot-renew.timer
[Unit]
Description=Certbot renewal timer

[Timer]
OnCalendar=*-*-* 00,12:00:00
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target

# /etc/systemd/system/certbot-renew.service
[Unit]
Description=Certbot renewal
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/certbot renew --quiet
ExecStartPost=/bin/systemctl reload nginx

# === データベースバックアップ（平日の深夜） ===
# /etc/systemd/system/db-backup.timer
[Unit]
Description=Weekday database backup

[Timer]
OnCalendar=Mon..Fri *-*-* 01:30:00
Persistent=true

[Install]
WantedBy=timers.target

# === ディスク使用量チェック（5分ごと） ===
# /etc/systemd/system/disk-check.timer
[Unit]
Description=Periodic disk usage check

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

---

## 5. ソケットアクティベーション

```ini
# ソケットアクティベーション: リクエストが来た時だけサービスを起動
# リソースの節約に有効

# /etc/systemd/system/myapp.socket
[Unit]
Description=My Application Socket

[Socket]
ListenStream=8080
# ListenStream=/run/myapp.sock   # UNIXソケットの場合
Accept=no
# Accept=yes の場合、接続ごとにサービスインスタンスを起動

[Install]
WantedBy=sockets.target

# /etc/systemd/system/myapp.service
[Unit]
Description=My Application
Requires=myapp.socket

[Service]
Type=simple
ExecStart=/opt/myapp/server
# ソケットはファイルディスクリプタ 3 で渡される

[Install]
WantedBy=multi-user.target

# ソケットの管理
sudo systemctl enable --now myapp.socket
systemctl list-sockets           # ソケット一覧
```

---

## 6. パスユニット（ファイル監視）

```ini
# パスユニット: ファイルやディレクトリの変更を監視してサービスを起動

# /etc/systemd/system/deploy-watch.path
[Unit]
Description=Watch for deployment files

[Path]
PathExists=/var/deploy/trigger    # ファイルが存在したら起動
# PathModified=/var/deploy/       # ディレクトリが変更されたら起動
# PathChanged=/var/deploy/        # ファイルが変更されたら起動（閉じた時）
# DirectoryNotEmpty=/var/deploy/  # ディレクトリが空でなくなったら起動
MakeDirectory=yes
Unit=deploy.service

[Install]
WantedBy=multi-user.target

# /etc/systemd/system/deploy.service
[Unit]
Description=Deploy triggered by file watch

[Service]
Type=oneshot
ExecStart=/opt/scripts/deploy.sh
ExecStartPost=/bin/rm -f /var/deploy/trigger

# パスユニットの管理
sudo systemctl enable --now deploy-watch.path
systemctl list-paths              # パスユニット一覧
```

---

## 7. systemd のセキュリティ設定

### 7.1 セキュリティディレクティブ

```ini
# サービスのセキュリティ強化設定

[Service]
# === ファイルシステム保護 ===
ProtectSystem=strict           # / を読み取り専用にマウント
# ProtectSystem=full           # /usr, /boot を読み取り専用
# ProtectSystem=true           # /usr を読み取り専用
ProtectHome=true               # /home, /root, /run/user を空にする
# ProtectHome=read-only        # 読み取り専用にする
# ProtectHome=tmpfs            # tmpfsで覆い隠す
PrivateTmp=true                # 専用の /tmp を使用
ReadWritePaths=/opt/myapp/data  # 書き込み可能なパス
ReadOnlyPaths=/opt/myapp/config # 読み取り専用のパス
InaccessiblePaths=/var/secret   # アクセス不可能にするパス
TemporaryFileSystem=/var:ro     # tmpfsでオーバーレイ
BindPaths=/src:/dest            # バインドマウント

# === ネットワーク保護 ===
PrivateNetwork=true            # 専用ネットワーク名前空間（loのみ）
# RestrictAddressFamilies=AF_INET AF_INET6  # 使用可能なアドレスファミリ
# IPAddressDeny=any             # 全IPアドレスを拒否
# IPAddressAllow=192.168.0.0/16 # 許可するIPアドレス

# === プロセス保護 ===
NoNewPrivileges=true           # 権限昇格を禁止
PrivateUsers=true              # 専用ユーザー名前空間
ProtectKernelTunables=true     # /proc, /sys の書き込み禁止
ProtectKernelModules=true      # カーネルモジュールのロード禁止
ProtectKernelLogs=true         # カーネルログへのアクセス禁止
ProtectControlGroups=true      # cgroupの変更禁止
ProtectClock=true              # システムクロックの変更禁止
ProtectHostname=true           # ホスト名の変更禁止
LockPersonality=true           # 実行ドメインの変更禁止
RestrictRealtime=true          # リアルタイムスケジューリング禁止
RestrictSUIDSGID=true          # SUID/SGIDビットの設定禁止
RestrictNamespaces=true        # 名前空間の作成禁止

# === ケイパビリティ制限 ===
CapabilityBoundingSet=CAP_NET_BIND_SERVICE  # 1024未満のポートにバインド可能
AmbientCapabilities=CAP_NET_BIND_SERVICE    # root以外でも低ポート使用可能
# CapabilityBoundingSet=         # 全ケイパビリティを無効化

# === システムコール制限 ===
SystemCallFilter=@system-service  # システムサービス用のシスコール
# SystemCallFilter=~@debug @mount @privileged  # 危険なシスコールを禁止
SystemCallArchitectures=native    # ネイティブアーキテクチャのみ
SystemCallErrorNumber=EPERM       # 拒否時のエラー番号

# === その他 ===
UMask=0077                     # ファイル作成時のumask
MemoryDenyWriteExecute=true    # W^X（書き込みと実行の排他）
```

### 7.2 セキュリティスコアの確認

```bash
# サービスのセキュリティスコアを確認
systemd-analyze security nginx.service

# 出力例:
# → Overall exposure level for nginx.service: 6.5 MEDIUM
#   NAME                          DESCRIPTION                     EXPOSURE
# ✗ PrivateNetwork=               Service has access to host's network  0.5
# ✗ PrivateUsers=                 Service has access to other users     0.2
# ✓ NoNewPrivileges=              Service process may not gain new privileges 0.0
# ...

# 全サービスのセキュリティスコア
systemd-analyze security

# スコアの目安:
# 0.0-2.0: SAFE（十分なセキュリティ）
# 2.0-4.0: OK（概ね安全）
# 4.0-7.0: MEDIUM（改善余地あり）
# 7.0-10.0: UNSAFE（セキュリティリスクあり）
```

---

## 8. リソース制限

### 8.1 リソース制限ディレクティブ

```ini
# サービスのリソース制限設定

[Service]
# === メモリ制限 ===
MemoryMax=512M                   # メモリ使用量の上限（超えるとOOM kill）
MemoryHigh=384M                  # メモリ使用量のソフトリミット（超えると回収圧力）
MemorySwapMax=0                  # スワップの使用を禁止
MemoryLow=128M                   # メモリ保護（最低保証）

# === CPU制限 ===
CPUQuota=50%                     # CPU使用率の上限（100% = 1コア分）
CPUQuota=200%                    # 2コア分
CPUWeight=100                    # CPU配分の重み（デフォルト100）
CPUAffinity=0 1                  # 使用するCPUコアを指定
AllowedCPUs=0-3                  # 使用可能なCPU範囲

# === IO制限 ===
IOWeight=100                     # IO配分の重み（1-10000）
IOReadBandwidthMax=/dev/sda 50M  # 読み取り帯域制限
IOWriteBandwidthMax=/dev/sda 20M # 書き込み帯域制限
IOReadIOPSMax=/dev/sda 3000      # 読み取りIOPS制限
IOWriteIOPSMax=/dev/sda 1000     # 書き込みIOPS制限

# === プロセス・タスク制限 ===
TasksMax=100                     # 最大タスク（プロセス/スレッド）数
LimitNPROC=100                   # プロセス数制限
LimitNOFILE=65536                # ファイルディスクリプタ数制限
LimitFSIZE=infinity              # ファイルサイズ制限
LimitCORE=0                      # コアダンプ無効
# LimitCORE=infinity             # コアダンプ有効

# === タイムアウト ===
TimeoutStartSec=30               # 起動タイムアウト
TimeoutStopSec=30                # 停止タイムアウト
TimeoutAbortSec=30               # アボートタイムアウト
RuntimeMaxSec=3600               # 最大実行時間（1時間）
WatchdogSec=30                   # ウォッチドッグ間隔
```

### 8.2 ドロップインによるリソース制限の追加

```bash
# ドロップインファイルで既存サービスにリソース制限を追加
# /etc/systemd/system/nginx.service.d/limits.conf
[Service]
MemoryMax=512M
CPUQuota=50%
TasksMax=100

# 作成手順
sudo mkdir -p /etc/systemd/system/nginx.service.d/
sudo tee /etc/systemd/system/nginx.service.d/limits.conf <<'EOF'
[Service]
MemoryMax=512M
CPUQuota=50%
TasksMax=100
LimitNOFILE=65536
EOF

sudo systemctl daemon-reload
sudo systemctl restart nginx

# 適用されたか確認
systemctl show nginx --property=MemoryMax,CPUQuota,TasksMax
```

### 8.3 cgroup によるリソース使用量の監視

```bash
# リソース使用量の確認
systemctl status nginx           # Memory, CPU, Tasks が表示される

# 詳細なリソース情報
systemd-cgtop                    # cgroup別のリソース使用量（topライク）
systemd-cgtop -m                 # メモリ順でソート
systemd-cgtop -c                 # CPU順でソート

# 特定サービスの cgroup 情報
cat /sys/fs/cgroup/system.slice/nginx.service/memory.current
cat /sys/fs/cgroup/system.slice/nginx.service/memory.max
cat /sys/fs/cgroup/system.slice/nginx.service/cpu.stat

# systemctl でのリソース表示
systemctl show nginx.service --property=MemoryCurrent
systemctl show nginx.service --property=CPUUsageNSec
```

---

## 9. systemd の実践パターン

### 9.1 サービスの起動失敗を調査

```bash
# パターン1: サービスの起動失敗を調査
sudo systemctl status myapp      # まず状態確認
journalctl -u myapp --since "10 minutes ago" --no-pager  # 直近ログ
journalctl -u myapp -p err       # エラーのみ

# 起動失敗の一般的な原因と対処:
# 1. 権限エラー → User/Group の確認、ファイル権限の確認
# 2. ポート競合 → ss -tlnp でポート使用状況確認
# 3. 依存サービス未起動 → After/Requires の確認
# 4. 設定ファイルエラー → ExecStartPre でconfigtestを実行
# 5. バイナリが見つからない → ExecStart のパスを確認
# 6. SELinux/AppArmor → ausearch -m AVC -ts recent で確認
```

### 9.2 複数サービスの一括管理

```bash
# パターン2: 複数サービスの一括管理
for svc in nginx postgresql redis; do
    echo "=== $svc ==="
    systemctl is-active "$svc"
done

# 一括再起動
for svc in nginx postgresql redis; do
    sudo systemctl restart "$svc"
    echo "$svc: $(systemctl is-active "$svc")"
done

# ワンライナーでの状態確認
systemctl is-active nginx postgresql redis
```

### 9.3 ターゲットによるサービスグループ化

```ini
# カスタムターゲットでサービスをグループ化
# /etc/systemd/system/webapp.target
[Unit]
Description=Web Application Stack
Requires=nginx.service postgresql.service redis.service myapp.service
After=nginx.service postgresql.service redis.service myapp.service

[Install]
WantedBy=multi-user.target

# 使い方
sudo systemctl enable webapp.target
sudo systemctl start webapp.target   # 全サービスを起動
sudo systemctl stop webapp.target    # 全サービスを停止
```

### 9.4 起動分析

```bash
# パターン4: 起動順序の確認
systemd-analyze                  # 起動時間
systemd-analyze blame            # サービス別起動時間
systemd-analyze critical-chain   # クリティカルパス
systemd-analyze critical-chain nginx.service  # 特定サービスのクリティカルパス

# 起動時間の可視化
systemd-analyze plot > boot.svg  # SVGファイルとして出力

# 起動が遅いサービスの特定
systemd-analyze blame | head -20

# 起動時のデバッグ
# カーネルパラメータに systemd.log_level=debug を追加
```

### 9.5 ユーザーサービス（root不要）

```bash
# パターン5: ユーザーサービス（root不要）
mkdir -p ~/.config/systemd/user/

# ~/.config/systemd/user/myapp.service を作成
cat > ~/.config/systemd/user/myapp.service <<'EOF'
[Unit]
Description=My User Application

[Service]
Type=simple
ExecStart=/home/user/bin/myapp
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# ユーザーサービスの管理
systemctl --user daemon-reload
systemctl --user start myapp
systemctl --user enable myapp
systemctl --user status myapp
journalctl --user -u myapp

# ログアウト後もサービスを継続（重要）
sudo loginctl enable-linger $USER

# ユーザーサービスの一覧
systemctl --user list-units --type=service
```

### 9.6 一時的なサービス実行

```bash
# systemd-run: 一時的なサービスとして実行
# リソース制限やログ管理を即座に適用できる

# 基本的な使い方
sudo systemd-run --unit=temp-backup /opt/scripts/backup.sh

# リソース制限付き
sudo systemd-run --unit=temp-task \
    --property=MemoryMax=256M \
    --property=CPUQuota=50% \
    /opt/scripts/heavy-task.sh

# タイマーとして
sudo systemd-run --on-calendar="*-*-* 03:00:00" \
    --unit=temp-cleanup \
    /opt/scripts/cleanup.sh

# 指定時間後に実行
sudo systemd-run --on-active="5m" \
    --unit=delayed-task \
    /opt/scripts/task.sh

# ユーザースコープで実行
systemd-run --user --scope --unit=my-build make -j4

# 実行中の一時サービスの確認
systemctl list-units --type=service 'run-*'
```

### 9.7 サービスの依存関係の条件設定

```ini
# 条件付き起動の設定
[Unit]
Description=My Conditional Service

# 条件（falseの場合、サービスをスキップ）
ConditionPathExists=/opt/myapp/config.yml     # ファイルが存在すること
ConditionPathExists=!/opt/myapp/.disabled     # ファイルが存在しないこと
ConditionPathIsDirectory=/opt/myapp/data      # ディレクトリが存在すること
ConditionFileIsExecutable=/opt/myapp/bin/app  # 実行可能であること
ConditionDirectoryNotEmpty=/opt/myapp/queue   # ディレクトリが空でないこと

# 環境条件
ConditionVirtualization=!container            # コンテナ内でないこと
ConditionKernelVersion=>=5.10                 # カーネルバージョン条件
ConditionMemory=>=1G                          # メモリ条件
ConditionCPUs=>=2                             # CPU数条件
ConditionEnvironment=ENABLE_MYAPP=true        # 環境変数条件

# アサート（falseの場合、エラー）
AssertPathExists=/opt/myapp/config.yml        # 存在しないとエラー
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と対処法

```bash
# === 問題1: サービスが起動しない ===
# 手順:
sudo systemctl status myapp          # 1. 状態確認
journalctl -u myapp -n 50 --no-pager # 2. ログ確認
systemd-analyze verify /etc/systemd/system/myapp.service  # 3. 構文チェック
sudo -u myapp /opt/myapp/bin/app     # 4. 手動実行テスト

# === 問題2: サービスが頻繁に再起動する ===
journalctl -u myapp --since "1 hour ago" | grep -E "Started|Stopped|Failed"
systemctl show myapp --property=NRestarts  # 再起動回数
# StartLimitBurst/StartLimitIntervalSec を確認

# === 問題3: サービスの停止に時間がかかる ===
# TimeoutStopSec を確認
systemctl show myapp --property=TimeoutStopUSec
# KillMode を確認（control-group, mixed, process, none）
# KillSignal を確認（デフォルト SIGTERM）

# === 問題4: 依存サービスが起動していない ===
systemctl list-dependencies myapp    # 依存関係確認
systemctl list-dependencies --reverse myapp  # 逆依存確認

# === 問題5: ユニットファイルの変更が反映されない ===
sudo systemctl daemon-reload         # 必ず実行
systemctl cat myapp                  # 現在のユニットファイルを確認
systemd-delta                        # オーバーライドの確認

# === 問題6: ログが多すぎてディスクが逼迫 ===
journalctl --disk-usage              # ログのサイズ確認
sudo journalctl --vacuum-size=500M   # 削減
# /etc/systemd/journald.conf で SystemMaxUse を設定
```

### 10.2 デバッグモードでの実行

```bash
# サービスをデバッグモードで実行
sudo systemctl stop myapp

# 環境変数を確認
systemctl show myapp --property=Environment
systemctl show myapp --property=EnvironmentFiles

# ExecStart のコマンドを手動で実行
sudo -u myapp bash -c 'source /opt/myapp/.env && /opt/myapp/bin/app'

# strace でシステムコールをトレース
sudo strace -f -p $(systemctl show myapp --property=MainPID --value)

# デバッグログを有効化
sudo systemctl set-environment SYSTEMD_LOG_LEVEL=debug
sudo systemctl restart myapp
journalctl -u myapp -f
# 終了後:
sudo systemctl unset-environment SYSTEMD_LOG_LEVEL
```

### 10.3 systemd 関連の便利コマンド

```bash
# システム全体の状態確認
systemctl is-system-running      # running, degraded, maintenance 等
systemctl --failed               # 失敗したユニットの一覧

# ブートターゲットの管理
systemctl get-default            # 現在のデフォルトターゲット
sudo systemctl set-default multi-user.target    # CUI起動
sudo systemctl set-default graphical.target     # GUI起動

# 電源管理
sudo systemctl poweroff          # シャットダウン
sudo systemctl reboot            # 再起動
sudo systemctl suspend           # サスペンド
sudo systemctl hibernate         # ハイバネート

# ランレベル互換
# 旧ランレベル → systemd ターゲット
# 0 → poweroff.target
# 1 → rescue.target
# 3 → multi-user.target
# 5 → graphical.target
# 6 → reboot.target

sudo systemctl isolate rescue.target  # レスキューモード
sudo systemctl isolate multi-user.target  # マルチユーザーモード

# ホスト名の管理
hostnamectl                      # ホスト名情報
sudo hostnamectl set-hostname myserver

# 日時の管理
timedatectl                      # 日時情報
sudo timedatectl set-timezone Asia/Tokyo
timedatectl list-timezones       # タイムゾーン一覧
sudo timedatectl set-ntp true    # NTP同期有効化

# ロケールの管理
localectl                        # ロケール情報
sudo localectl set-locale LANG=ja_JP.UTF-8
```

---

## 11. systemd と Docker / コンテナの連携

```ini
# Docker コンテナをsystemdで管理する例
# /etc/systemd/system/docker-myapp.service
[Unit]
Description=MyApp Docker Container
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10

# 既存コンテナを削除してから起動
ExecStartPre=-/usr/bin/docker rm -f myapp
ExecStart=/usr/bin/docker run \
    --name myapp \
    --rm \
    -p 8080:8080 \
    -v /opt/myapp/data:/data \
    --env-file /opt/myapp/.env \
    myapp:latest

ExecStop=/usr/bin/docker stop myapp

[Install]
WantedBy=multi-user.target

# docker compose との連携
# /etc/systemd/system/docker-compose-myapp.service
[Unit]
Description=MyApp Docker Compose Stack
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/myapp
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose up -d --force-recreate

[Install]
WantedBy=multi-user.target
```

---

## 12. systemd ネットワーク管理（networkd）

```bash
# systemd-networkd: ネットワーク設定の管理
# /etc/systemd/network/20-wired.network
# [Match]
# Name=eth0
#
# [Network]
# DHCP=yes
# DNS=8.8.8.8
# DNS=8.8.4.4
#
# [DHCPv4]
# RouteMetric=100

# 静的IPの設定
# /etc/systemd/network/20-static.network
# [Match]
# Name=eth0
#
# [Network]
# Address=192.168.1.100/24
# Gateway=192.168.1.1
# DNS=8.8.8.8

# networkd の管理
sudo systemctl enable --now systemd-networkd
networkctl list                  # ネットワークインターフェース一覧
networkctl status                # 詳細状態
networkctl status eth0           # 特定インターフェースの状態

# systemd-resolved: DNS解決
sudo systemctl enable --now systemd-resolved
resolvectl status                # DNS設定の確認
resolvectl query example.com     # DNS問い合わせ
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| systemctl start/stop/restart | サービス操作 |
| systemctl enable/disable | 自動起動管理 |
| systemctl status | 状態確認 |
| systemctl mask/unmask | サービスの完全無効化 |
| systemctl edit | ドロップインによるカスタマイズ |
| journalctl -u service | サービスログ |
| journalctl -f | リアルタイムログ |
| journalctl -p err | 優先度フィルタ |
| journalctl -b -1 | 前回ブートのログ |
| systemctl daemon-reload | ユニット再読み込み |
| systemd-analyze security | セキュリティ監査 |
| systemd-analyze blame | 起動時間分析 |
| systemd-cgtop | リソース使用量モニタ |
| systemd-run | 一時的なサービス実行 |

---

## 次に読むべきガイド
→ [[01-package-management.md]] — パッケージ管理

---

## 参考文献
1. "systemd System and Service Manager." systemd.io.
2. Barrett, D. "Efficient Linux at the Command Line." Ch.10, O'Reilly, 2022.
3. "systemd.exec — Execution environment configuration." freedesktop.org/software/systemd/man.
4. "systemd.service — Service unit configuration." freedesktop.org/software/systemd/man.
5. "Arch Wiki: systemd." wiki.archlinux.org/title/systemd.
6. Nemeth, E., et al. "UNIX and Linux System Administration Handbook." 5th Ed, Ch.2, Addison-Wesley, 2017.
