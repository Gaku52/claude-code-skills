# Linux CLI Mastery

Linuxコマンドラインを実践的にマスターする包括的ガイド。ファイル操作、テキスト処理、プロセス管理、ネットワーク、シェルスクリプティング、システム管理まで、CLI操作の全てをカバー。

## Skill概要

| 項目 | 内容 |
|------|------|
| カテゴリ | OS・CLI |
| 難易度 | 初級〜中級 |
| 前提知識 | operating-system-guide（基礎レベル） |
| 推定学習時間 | 40〜60時間 |
| ガイド数 | 22ファイル |

## 学習目標

- [ ] シェルの基本操作とカスタマイズができる
- [ ] ファイル操作・パーミッション管理を自在に行える
- [ ] テキスト処理（grep, sed, awk）を使いこなせる
- [ ] プロセスの監視・管理ができる
- [ ] ネットワーク関連コマンドを理解している
- [ ] シェルスクリプトを書いて自動化できる
- [ ] systemd, cron, パッケージ管理を運用できる

## ディレクトリ構成

```
docs/
├── 00-introduction/          # CLI入門
│   ├── 00-terminal-basics.md    # ターミナルとシェルの基礎
│   ├── 01-shell-config.md       # シェル設定（.bashrc, .zshrc）
│   └── 02-man-and-help.md       # マニュアルとヘルプの活用
├── 01-file-operations/       # ファイル操作
│   ├── 00-navigation.md         # ディレクトリ移動と一覧
│   ├── 01-file-crud.md          # ファイルの作成・コピー・移動・削除
│   ├── 02-permissions.md        # パーミッションと所有者
│   └── 03-find-and-locate.md    # ファイル検索
├── 02-text-processing/       # テキスト処理
│   ├── 00-cat-less-head-tail.md # ファイル表示
│   ├── 01-grep-ripgrep.md       # パターン検索
│   ├── 02-sed.md                # ストリームエディタ
│   ├── 03-awk.md                # テキスト処理言語
│   └── 04-sort-uniq-cut-wc.md   # ソート・集計
├── 03-process-management/    # プロセス管理
│   ├── 00-ps-top-htop.md        # プロセス監視
│   └── 01-jobs-signals.md       # ジョブ制御とシグナル
├── 04-networking/            # ネットワーク
│   ├── 00-curl-wget.md          # HTTP通信
│   └── 01-ssh-scp.md            # リモート接続
├── 05-shell-scripting/       # シェルスクリプト
│   ├── 00-basics.md             # 変数・条件分岐・ループ
│   └── 01-advanced-scripting.md # 関数・エラー処理・実践パターン
├── 06-system-admin/          # システム管理
│   ├── 00-systemd.md            # systemdとサービス管理
│   └── 01-package-management.md # パッケージ管理
└── 07-advanced/              # 上級テクニック
    ├── 00-tmux-screen.md        # ターミナルマルチプレクサ
    └── 01-productivity.md       # 生産性向上テクニック
```

## 前提Skill
- [[operating-system-guide]] — OS基礎（プロセス、ファイルシステムの概念）

## 次のステップ
- [[docker-container-guide]] — Docker・コンテナ
- [[devops-and-github-actions]] — DevOps
- [[script-development]] — スクリプト開発

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, No Starch Press, 2019.
2. Barrett, D. "Efficient Linux at the Command Line." O'Reilly, 2022.
3. Robbins, A. & Beebe, N. "Classic Shell Scripting." O'Reilly, 2005.
