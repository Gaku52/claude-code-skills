# ファイル検索

> 「あのファイルどこだっけ？」— find と fd があれば必ず見つかる。

## この章で学ぶこと

- [ ] find の主要な使い方をマスターする
- [ ] fd（モダン代替）を使いこなせる
- [ ] locate / mlocate によるデータベース検索を理解する
- [ ] which / whereis / type によるコマンド検索を使い分ける
- [ ] 実務で頻出するファイル検索パターンを身につける

---

## 1. find — 標準のファイル検索ツール

### 1.1 基本構文

```bash
# 基本構文: find [検索開始パス] [条件式] [アクション]
#
# find は指定されたパスを起点にディレクトリツリーを再帰的に走査し、
# 条件にマッチするファイル/ディレクトリを出力する。
# パスを省略すると . （カレントディレクトリ）が使われる。

# 全ファイル・ディレクトリを表示
find .

# 特定のディレクトリを起点に検索
find /var/log
find /home/user/projects

# 複数の起点ディレクトリを指定
find /etc /usr/local/etc -name "*.conf"
```

### 1.2 名前による検索（-name / -iname / -path / -regex）

```bash
# -name: ファイル名のパターンマッチ（シェルグロブ）
find . -name "*.md"               # .md ファイルを再帰検索
find . -name "*.txt"              # .txt ファイルを再帰検索
find . -name "Makefile"           # Makefile を検索
find . -name "*.log"              # ログファイルを検索
find . -name "config.*"           # config.xxx を検索

# -iname: 大小文字を無視したパターンマッチ
find . -iname "readme*"           # README, readme, Readme 全てマッチ
find . -iname "*.jpg"             # .jpg, .JPG, .Jpg 全てマッチ
find . -iname "license*"          # LICENSE, license, License 全てマッチ

# -path: パス全体に対するパターンマッチ
find . -path "*/src/*.js"         # src ディレクトリ内の .js ファイル
find . -path "*/test/*"           # test ディレクトリ以下の全ファイル
find . -path "*/.git/*" -prune -o -name "*.py" -print  # .git を除外して .py 検索

# -regex: 正規表現によるパス全体のマッチ
find . -regex ".*\.\(js\|ts\|jsx\|tsx\)"   # JS/TS 関連ファイル
find . -regextype posix-extended -regex ".*\.(jpg|jpeg|png|gif)"  # 画像ファイル
find . -regex ".*/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}.*"  # 日付パターンを含むパス

# ワイルドカードの注意点
# シェルが先にグロブ展開しないよう、パターンは必ずクォートで囲む
find . -name "*.log"              # 正しい
# find . -name *.log              # シェルが展開するので危険
```

### 1.3 タイプによる絞り込み（-type）

```bash
# -type で検索対象の種類を指定
find . -type f                    # 通常ファイルのみ
find . -type d                    # ディレクトリのみ
find . -type l                    # シンボリックリンクのみ
find . -type b                    # ブロックデバイス
find . -type c                    # キャラクタデバイス
find . -type p                    # 名前付きパイプ（FIFO）
find . -type s                    # ソケット

# 組み合わせ例
find . -type f -name "*.conf"     # 設定ファイル（ファイルのみ）
find . -type d -name "test*"      # test で始まるディレクトリ
find . -type l -name "*.so"       # .so のシンボリックリンク

# ディレクトリ一覧を取得（プロジェクト構造の確認に便利）
find . -maxdepth 2 -type d | sort
```

### 1.4 サイズによる絞り込み（-size）

```bash
# -size [+-]N[cwbkMG]
# +N: Nより大きい, -N: Nより小さい, N: ちょうどN
# c: バイト, w: ワード(2B), b: ブロック(512B), k: KB, M: MB, G: GB

find . -size +100M                # 100MB以上のファイル
find . -size +1G                  # 1GB以上のファイル
find . -size -1k                  # 1KB未満のファイル
find . -size 0                    # 0バイトのファイル
find . -empty                     # 空ファイル/空ディレクトリ

# サイズ範囲で絞り込み
find . -size +10M -size -100M     # 10MB〜100MB のファイル
find . -size +1k -size -10k       # 1KB〜10KB のファイル

# 実務例: ディスク容量を圧迫している大きなファイルを探す
find / -type f -size +500M 2>/dev/null | head -20
find /var -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# 空ディレクトリを探す（掃除用）
find . -type d -empty
find . -type d -empty -delete     # 空ディレクトリを削除
```

### 1.5 日時による絞り込み（-mtime / -atime / -ctime / -newer）

```bash
# -mtime: 内容の最終更新日（modification time）
# -atime: 最終アクセス日（access time）
# -ctime: メタデータの最終変更日（change time = パーミッション変更等）
# 単位は「日」。+N は N日より前、-N は N日以内

find . -mtime -7                  # 7日以内に変更されたファイル
find . -mtime +30                 # 30日以上前に変更されたファイル
find . -mtime 0                   # 今日変更されたファイル（24時間以内）
find . -atime -1                  # 1日以内にアクセスされたファイル
find . -ctime -3                  # 3日以内にメタデータが変更されたファイル

# -mmin / -amin / -cmin: 分単位での指定
find . -mmin -30                  # 30分以内に変更されたファイル
find . -mmin +60                  # 60分以上前に変更されたファイル
find . -mmin -5                   # 5分以内に変更（デバッグに便利）

# -newer: 基準ファイルより新しいファイルを検索
find . -newer reference.txt       # reference.txt より新しいファイル
find . -newer /tmp/timestamp      # タイムスタンプファイルより新しい

# 実務例: タイムスタンプファイルを活用した差分ファイル検出
touch -t 202601010000 /tmp/since_newyear   # 基準タイムスタンプ作成
find . -newer /tmp/since_newyear -type f    # 基準以降に変更されたファイル

# 日付範囲で検索（-newerXY を使う方法 ※GNU find）
# -newermt: modification time を文字列で指定
find . -newermt "2026-01-01" ! -newermt "2026-02-01" -type f
# → 2026年1月に変更されたファイル

# 古いファイルの掃除
find /tmp -type f -mtime +7 -delete         # 7日以上前の一時ファイルを削除
find /var/log -name "*.gz" -mtime +90 -delete  # 90日以上前の圧縮ログを削除
```

### 1.6 パーミッション・所有者による絞り込み

```bash
# -perm: パーミッションで検索
find . -perm 644                  # 正確に644のファイル
find . -perm -644                 # 644を含む（644以上）ファイル
find . -perm /111                 # 実行権限があるファイル（いずれかのビット）
find . -perm -u+x                # ユーザーに実行権限があるファイル
find . -perm /o+w                 # その他に書き込み権限があるファイル

# セキュリティチェック: 危険なパーミッションのファイルを検出
find / -perm -4000 -type f 2>/dev/null   # SUID ビットが設定されたファイル
find / -perm -2000 -type f 2>/dev/null   # SGID ビットが設定されたファイル
find / -perm /o+w -type f 2>/dev/null    # 全ユーザー書き込み可能ファイル
find /home -perm 777 -type f             # 権限が緩すぎるファイル

# -user / -group: 所有者・グループで検索
find . -user root                 # root が所有するファイル
find . -user nobody               # nobody が所有するファイル
find . -group www-data            # www-data グループのファイル
find . -nouser                    # 所有者が存在しないファイル
find . -nogroup                   # グループが存在しないファイル

# 実務例: 特定ユーザーのファイル一覧
find /home/developer -user developer -type f | wc -l   # ファイル数カウント
find /var/www -not -user www-data -type f              # www-data 以外のファイル
```

### 1.7 論理演算子と条件の組み合わせ

```bash
# AND（暗黙的 / -a）
find . -name "*.log" -size +10M   # 10MB以上のログファイル（暗黙的 AND）
find . -name "*.log" -a -size +10M  # 明示的 AND（同じ意味）

# OR（-o）
find . \( -name "*.js" -o -name "*.ts" \)        # .js または .ts
find . \( -name "*.jpg" -o -name "*.png" -o -name "*.gif" \)  # 画像ファイル
find . -type f \( -name "*.log" -o -name "*.tmp" \) -mtime +30  # 古いログ/tmp

# NOT（! / -not）
find . -type f ! -name "*.md"     # .md 以外のファイル
find . -not -name "*.pyc"         # .pyc 以外のファイル
find . ! -empty                   # 空でないファイル
find . -type f ! -user root       # root 以外が所有するファイル

# 複雑な条件の組み合わせ
# 括弧 \( \) で優先順位を制御（エスケープ必須）
find . -type f \( -name "*.js" -o -name "*.ts" \) ! -path "*/node_modules/*"
# → node_modules を除外した JS/TS ファイル

find . -type f \( -name "*.py" -o -name "*.rb" \) -size +1k -mtime -30
# → 30日以内に変更された 1KB以上の Python/Ruby ファイル

# -prune による除外（効率的なディレクトリスキップ）
find . -path "./.git" -prune -o -type f -print   # .git を除外
find . -name "node_modules" -prune -o -name "*.js" -print  # node_modules除外
find . \( -name ".git" -o -name "node_modules" -o -name "__pycache__" \) -prune -o -type f -print
# → .git, node_modules, __pycache__ を全て除外
```

### 1.8 アクション（-exec / -execdir / -ok / -delete / -print）

```bash
# -print: デフォルトのアクション（パスを表示）
find . -name "*.md" -print        # 明示的に -print（省略可）

# -print0: NULL文字区切りで出力（xargs -0 と組み合わせ）
find . -name "*.txt" -print0 | xargs -0 wc -l   # スペース入りファイル名対応

# -printf: カスタムフォーマット出力（GNU find）
find . -type f -printf "%s %p\n"            # サイズとパス
find . -type f -printf "%T+ %p\n" | sort    # 更新日時でソート
find . -type f -printf "%u %g %m %p\n"      # 所有者 グループ パーミッション パス

# -delete: マッチしたファイルを削除（注意して使用！）
find . -name "*.tmp" -delete      # .tmp ファイルを削除
find . -type d -empty -delete     # 空ディレクトリを削除
find /tmp -mtime +7 -delete       # 7日以上前の一時ファイルを削除

# -exec: 各ファイルに対してコマンドを実行
find . -name "*.sh" -exec chmod +x {} \;        # シェルスクリプトに実行権限付与
find . -name "*.log" -exec gzip {} \;           # ログファイルを圧縮
find . -name "*.bak" -exec rm {} \;             # バックアップファイルを削除
find . -name "*.py" -exec grep -l "import os" {} \;  # osをimportしているPyファイル

# -exec の末尾
# \;  各ファイルごとに1回コマンドを実行（遅い）
# +   まとめてコマンドに渡す（高速、xargs と同等）
find . -name "*.txt" -exec wc -l {} +           # まとめて wc に渡す（高速）
find . -name "*.js" -exec grep -l "console.log" {} +   # まとめて grep

# -execdir: ファイルが存在するディレクトリで実行（セキュリティ向上）
find . -name "*.sh" -execdir chmod +x {} \;

# -ok: 実行前に確認を求める（対話的）
find . -name "*.tmp" -ok rm {} \;   # 1つずつ確認して削除

# xargs との組み合わせ（-exec + と同等だがより柔軟）
find . -name "*.py" | xargs grep "TODO"              # スペースなしファイル名向け
find . -name "*.py" -print0 | xargs -0 grep "TODO"   # スペース対応版
find . -name "*.css" -print0 | xargs -0 -I{} cp {} /backup/   # 個別コピー
find . -name "*.log" -print0 | xargs -0 -P 4 gzip    # 4並列で圧縮
```

### 1.9 深さ制限と検索順序

```bash
# -maxdepth: 検索の最大深度を制限
find . -maxdepth 1 -type f        # カレントディレクトリのファイルのみ（再帰しない）
find . -maxdepth 2 -type d        # 2階層まで のディレクトリ
find . -maxdepth 3 -name "*.md"   # 3階層までの .md ファイル

# -mindepth: 検索の最小深度を指定
find . -mindepth 2 -name "*.py"   # 2階層以降の .py ファイル（直下を除外）
find . -mindepth 1 -maxdepth 1 -type d  # 直下のディレクトリのみ（ls -d */ と同等）

# -depth: 深さ優先（ディレクトリの内容を先に処理）
find . -depth -name "*.tmp" -delete  # 深さ優先で削除（空ディレクトリ対策）

# -mount / -xdev: ファイルシステムをまたがない
find / -mount -name "*.conf" -type f  # ルートファイルシステムのみ検索
```

### 1.10 find の実務パターン集

```bash
# --- プロジェクト管理 ---

# プロジェクト内の全ソースファイルを列挙
find ./src -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" \) \
  ! -path "*/node_modules/*" ! -path "*/.next/*" | sort

# プロジェクトのファイル数をディレクトリごとに集計
find . -type f ! -path "*/.git/*" | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -20

# 最近変更されたファイルトップ20
find . -type f ! -path "*/.git/*" -printf "%T@ %T+ %p\n" | sort -rn | head -20

# 重複ファイルの検出（MD5ハッシュベース）
find . -type f -exec md5sum {} + | sort | uniq -w 32 -d

# --- ディスク管理 ---

# サイズ順に大きいファイルを表示
find . -type f -exec ls -lS {} + | head -20

# ディスク使用量トップ10ディレクトリ
find . -maxdepth 3 -type d -exec du -sh {} + 2>/dev/null | sort -rh | head -10

# 古い大きなファイルの検出（アーカイブ候補）
find /data -type f -size +100M -mtime +365 -ls

# --- ログ管理 ---

# 本日のログファイルのみ表示
find /var/log -type f -mtime 0 -name "*.log"

# ログローテーション: 古い圧縮ログの削除
find /var/log -name "*.gz" -mtime +180 -delete

# エラーを含むログファイルを検索
find /var/log -name "*.log" -mtime -1 -exec grep -l "ERROR" {} +

# --- 開発環境 ---

# 全テストファイルを列挙
find . -type f \( -name "*_test.go" -o -name "*_test.py" -o -name "*.test.js" -o -name "*.spec.ts" \)

# __pycache__ の一括削除
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# .DS_Store の一括削除
find . -name ".DS_Store" -delete

# node_modules のサイズ確認
find . -type d -name "node_modules" -exec du -sh {} + 2>/dev/null

# Go の依存パッケージ一覧
find $GOPATH/pkg -type d -maxdepth 4

# --- セキュリティ ---

# ワールドライタブルなファイルを検出
find /var/www -type f -perm /o+w -ls

# SUID/SGID ファイルの監査
find / -type f \( -perm -4000 -o -perm -2000 \) -exec ls -la {} \; 2>/dev/null

# 所有者不明ファイルの検出
find / -nouser -o -nogroup 2>/dev/null

# 最近変更された設定ファイル（改ざん検知）
find /etc -type f -mmin -60 -ls

# --- バックアップ ---

# rsync 用のファイルリスト生成
find /data -type f -newer /tmp/last_backup -print0 > /tmp/backup_list.txt

# tar アーカイブの作成
find ./project -type f -name "*.py" -print0 | tar czf python_files.tar.gz --null -T -

# 差分バックアップ
find /home -type f -mtime -1 -print0 | cpio -0 -pdm /backup/daily/
```

### 1.11 find のパフォーマンス最適化

```bash
# 1. -type を早い段階で指定（ディレクトリエントリのタイプチェックはコスト低）
find . -type f -name "*.log"      # 良い: type を先に
# find . -name "*.log" -type f    # 動作は同じだが、type 先の方が一般的

# 2. -prune で不要ディレクトリをスキップ
find . -path "./.git" -prune -o -type f -name "*.py" -print
# → .git 以下を完全にスキップするため高速

# 3. -exec + を使う（\; より高速）
find . -name "*.txt" -exec wc -l {} +     # 良い: まとめて実行
# find . -name "*.txt" -exec wc -l {} \;  # 遅い: 1ファイルずつ実行

# 4. -maxdepth で検索範囲を限定
find . -maxdepth 3 -name "*.conf"   # 3階層までに制限

# 5. 標準エラーを抑制（権限エラーの無視）
find / -name "*.conf" 2>/dev/null
find / -name "*.conf" 2>&1 | grep -v "Permission denied"

# 6. 並列処理（xargs -P）
find . -name "*.png" -print0 | xargs -0 -P $(nproc) optipng -o7
# → CPU コア数分の並列で画像最適化

# 7. find の結果をファイルに保存して再利用
find /data -type f -name "*.csv" > /tmp/csv_files.txt
while IFS= read -r file; do
    process_csv "$file"
done < /tmp/csv_files.txt
```

---

## 2. fd — モダンな代替ツール

### 2.1 インストールと概要

```bash
# インストール
brew install fd                   # macOS
sudo apt install fd-find          # Ubuntu/Debian（コマンド名は fdfind）
sudo pacman -S fd                 # Arch Linux
cargo install fd-find             # Rust (Cargo)

# Ubuntu/Debian では fdfind という名前になるため、エイリアスを設定
alias fd='fdfind'
# または ~/.bashrc に追加

# fd の特徴
# - .gitignore を自動で尊重（--no-ignore で無効化可能）
# - カラー出力がデフォルト
# - 正規表現がデフォルト（-g でグロブに切替）
# - Unicode 対応
# - find より簡潔な構文
# - 並列実行による高速化
```

### 2.2 基本的な使い方

```bash
# 基本構文: fd [パターン] [検索パス]

# パターンなし: 全ファイルを表示
fd                                # 全ファイル（.gitignore を尊重）

# 文字列パターン（部分一致、正規表現）
fd readme                         # "readme" を含むファイル/ディレクトリ
fd "\.md$"                        # 正規表現: .md で終わるファイル
fd "^test"                        # test で始まるファイル
fd "[0-9]{4}"                     # 4桁の数字を含むファイル
fd "config\.(json|yaml|toml)"     # 設定ファイル（複数拡張子）

# -g: グロブパターン（find の -name に近い）
fd -g "*.md"                      # グロブで .md ファイル検索
fd -g "Makefile"                  # 完全一致
fd -g "*.{js,ts,jsx,tsx}"         # 複数拡張子

# 検索パスの指定
fd "\.py$" /home/user/projects    # 特定ディレクトリ以下を検索
fd "\.rs$" src/                   # src/ 以下の Rust ファイル
```

### 2.3 主要オプション

```bash
# 拡張子で検索（-e / --extension）
fd -e md                          # .md ファイル
fd -e py                          # .py ファイル
fd -e jpg -e png -e gif           # 画像ファイル（複数拡張子）
fd -e rs -e toml                  # Rust プロジェクト関連

# タイプで絞り込み（-t / --type）
fd -t f                           # ファイルのみ (file)
fd -t d                           # ディレクトリのみ (directory)
fd -t l                           # シンボリックリンクのみ (symlink)
fd -t x                           # 実行可能ファイルのみ (executable)
fd -t e                           # 空ファイル/ディレクトリ (empty)

# 隠しファイル（-H / --hidden）
fd -H                             # 隠しファイル含む（.gitignore は引き続き尊重）
fd -H "\.env"                     # .env ファイル検索

# .gitignore を無視（-I / --no-ignore）
fd -I                             # .gitignore を無視
fd -HI                            # 隠しファイル + .gitignore 無視（find と同等）

# 大小文字の制御
fd -s "README"                    # 大小文字を区別（-s / --case-sensitive）
fd -i "readme"                    # 大小文字を無視（-i / --ignore-case）
# デフォルト: パターンが全て小文字なら case-insensitive（スマートケース）

# 深さ制限
fd -d 1                           # 1階層のみ（--max-depth）
fd -d 3                           # 3階層まで
fd --min-depth 2                  # 2階層以降

# 除外パターン（-E / --exclude）
fd -E node_modules                # node_modules を除外
fd -E "*.min.js"                  # minified JS を除外
fd -E ".git" -E "target"          # 複数ディレクトリを除外

# サイズフィルタ（-S / --size）
fd -S +1m                         # 1MB 以上
fd -S -10k                        # 10KB 以下
fd -S +100k -S -1m               # 100KB〜1MB

# 日時フィルタ
fd --changed-within 1h            # 1時間以内に変更
fd --changed-within 2d            # 2日以内に変更
fd --changed-before 1w            # 1週間以上前に変更
fd --changed-within "2026-01-01"  # 指定日以降に変更

# 所有者フィルタ
fd --owner root                   # root が所有するファイル
fd --owner ":www-data"            # www-data グループのファイル
```

### 2.4 アクション実行（-x / -X）

```bash
# -x / --exec: 各ファイルに対してコマンドを実行
fd -e txt -x wc -l                # 各 .txt ファイルの行数
fd -e sh -x chmod +x              # シェルスクリプトに実行権限
fd -e bak -x rm                   # .bak ファイルを削除
fd -e png -x optipng              # PNG 最適化

# プレースホルダ
# {}   フルパス
# {/}  ファイル名のみ（ディレクトリ部分なし）
# {//} ディレクトリ部分のみ
# {.}  拡張子を除いたパス
# {/.} 拡張子を除いたファイル名
fd -e jpg -x convert {} {.}.png   # JPG → PNG 変換
fd -e md -x echo "File: {/}, Dir: {//}"  # ファイル名とディレクトリ

# -X / --exec-batch: 全結果をまとめて1回のコマンドに渡す（find -exec + 相当）
fd -e py -X wc -l                 # 全 .py ファイルの行数を一括カウント
fd -e js -X eslint                # 全 JS ファイルを一括 lint

# 並列実行（-j / --threads）
fd -e png -x optipng -j 4         # 4スレッドで並列処理
fd -e mp4 -x ffmpeg -i {} {.}.webm -j 2  # 2並列で動画変換
```

### 2.5 fd の実務パターン集

```bash
# プロジェクト内の全ソースファイル行数を集計
fd -e py -X wc -l | tail -1

# 特定パターンのファイルを一括リネーム
fd -e txt -x mv {} {.}.md        # .txt → .md

# Docker 関連ファイルを検索
fd -g "Dockerfile*" -g "docker-compose*" -g ".dockerignore"

# 最近変更されたファイルの一覧（更新日時付き）
fd -t f --changed-within 1d -x ls -la

# テストファイルを除外してソースファイルを列挙
fd -e py -E "*_test.py" -E "test_*" -E "conftest.py"

# 設定ファイルの一括検索
fd -e yaml -e yml -e json -e toml -e ini -e conf

# Git で追跡されていないファイルの中から検索
fd -I -t f "\.log$"              # .gitignore を無視してログファイル検索
```

### 2.6 find と fd の比較表

```
┌─────────────────┬────────────────────────┬────────────────────────┐
│ 機能            │ find                   │ fd                     │
├─────────────────┼────────────────────────┼────────────────────────┤
│ デフォルト動作  │ 全ファイルを表示       │ .gitignore 尊重        │
│ パターン        │ シェルグロブ(-name)    │ 正規表現（デフォルト） │
│ 大小文字        │ 区別（-iname で無視）  │ スマートケース         │
│ 出力            │ モノクロ               │ カラー表示             │
│ 速度            │ 標準的                 │ 高速（並列処理）       │
│ 構文            │ 冗長（-name, -type等） │ 簡潔                   │
│ 環境            │ 標準搭載               │ 別途インストール       │
│ スクリプト      │ POSIX 互換             │ 非標準                 │
│ 隠しファイル    │ 含む                   │ デフォルト除外         │
│ -exec 相当      │ -exec {} \; / +        │ -x / -X                │
│ 深さ制限        │ -maxdepth / -mindepth  │ -d / --min-depth       │
│ サイズ          │ -size +10M             │ -S +10m                │
└─────────────────┴────────────────────────┴────────────────────────┘

使い分けガイド:
- 日常的な検索 → fd（シンプルで高速）
- シェルスクリプト/CI → find（POSIX 互換、標準搭載）
- 複雑な条件 → find（論理演算子が豊富）
- .gitignore 尊重 → fd（デフォルト対応）
```

---

## 3. locate / mlocate / plocate — データベースベースの高速検索

### 3.1 概要とインストール

```bash
# locate はファイルシステムのデータベースを使った高速検索
# find/fd がリアルタイムにディスクを走査するのに対し、
# locate は事前に構築されたデータベースを検索するため非常に高速

# インストール
sudo apt install mlocate          # Ubuntu/Debian（mlocate）
sudo apt install plocate          # Ubuntu 22.04+（plocate、より高速）
sudo yum install mlocate          # CentOS/RHEL
brew install findutils            # macOS（glocate として利用可能）

# データベースの初期構築
sudo updatedb                     # データベースの構築/更新
# cron で毎日自動更新される（通常 /etc/cron.daily/mlocate）
```

### 3.2 基本的な使い方

```bash
# 基本: locate [パターン]
locate filename                   # パス名でデータベース検索
locate "*.conf"                   # ワイルドカード
locate -i "readme"                # 大小文字無視
locate -n 10 "*.log"             # 結果を10件に制限
locate -c "*.py"                 # マッチ数をカウント
locate -b "filename"              # ベースネーム（ファイル名部分）のみで検索
locate -r "\.py$"                 # 正規表現で検索

# データベースの確認
locate -S                         # データベースの統計情報表示
# → ファイル数、ディレクトリ数、データベースサイズ等

# データベースの手動更新
sudo updatedb                     # 全体更新
# updatedb の設定: /etc/updatedb.conf
# PRUNEPATHS: 除外するパス（/tmp, /proc 等）
# PRUNEFS: 除外するファイルシステム（nfs, tmpfs 等）
```

### 3.3 locate の注意点と使い分け

```bash
# locate の制限事項:
# 1. データベースが古い場合、最近のファイルが見つからない
#    → sudo updatedb で手動更新するか、find/fd を使う
# 2. 権限を考慮しない場合がある（設定による）
# 3. 削除済みファイルが表示される可能性がある

# 使い分けガイド:
# - ファイル名だけで素早く探したい → locate
# - 最新の状態を確実に検索したい → find / fd
# - サイズ・日時等の条件が必要 → find / fd
# - サーバー上の設定ファイル検索 → locate（高速で便利）

# 実務例: locate + grep の組み合わせ
locate "nginx.conf" | grep -v backup   # バックアップを除外
locate -b "settings.py" | head -5       # Django の設定ファイルを素早く発見
```

---

## 4. which / whereis / type — コマンドの場所を検索

### 4.1 which

```bash
# which: 実行可能ファイルのパスを表示（PATH から検索）
which python                      # /usr/bin/python
which -a python                   # PATH 上の全 python を表示
which node                        # /usr/local/bin/node
which gcc                         # コンパイラのパス確認

# 実務例
which python3 && python3 --version   # Python3 があればバージョン表示
if which docker > /dev/null 2>&1; then
    echo "Docker is installed"
fi
```

### 4.2 whereis

```bash
# whereis: バイナリ、ソース、マニュアルの場所を検索
whereis python                    # バイナリ、マニュアル等
whereis -b python                 # バイナリのみ
whereis -m python                 # マニュアルのみ
whereis -s python                 # ソースのみ
whereis ls grep awk               # 複数コマンドを一括検索
```

### 4.3 type

```bash
# type: コマンドの種類を表示（bash 組み込み）
type ls                           # ls はエイリアス / 関数 / 外部コマンド のいずれか
type cd                           # cd is a shell builtin
type ll                           # ll is aliased to `ls -la`
type -a python                    # 全ての候補を表示
type -t ls                        # 種類だけ表示（alias, builtin, function, file）

# 実務例: コマンドの実体確認
type -a grep                      # grep がエイリアスか外部コマンドか確認
type -a python python3 pip pip3   # Python 環境の確認
```

---

## 5. 高度なファイル検索テクニック

### 5.1 fzf を活用したインタラクティブ検索

```bash
# fzf: ファジーファインダー（対話的に絞り込み検索）
# brew install fzf

# 基本的な使い方
find . -type f | fzf              # 全ファイルからインタラクティブ選択
fd -t f | fzf                     # fd + fzf の組み合わせ

# fzf + エディタ
vim $(fzf)                        # fzf で選択したファイルを vim で開く
code $(fd -e py | fzf -m)         # 複数選択して VS Code で開く

# fzf + preview
fd -t f | fzf --preview 'bat --color=always {}'   # プレビュー付き
fd -t f | fzf --preview 'head -50 {}'              # head でプレビュー

# fzf + kill（プロセス検索 & kill）
ps aux | fzf | awk '{print $2}' | xargs kill

# fzf + git
git log --oneline | fzf --preview 'git show {1}'   # コミット選択
git branch | fzf | xargs git checkout               # ブランチ切替

# .bashrc / .zshrc に設定するキーバインド
export FZF_DEFAULT_COMMAND='fd -t f --hidden --exclude .git'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
# Ctrl+T: ファイル検索、Ctrl+R: コマンド履歴検索、Alt+C: ディレクトリ移動
```

### 5.2 tree コマンドによるディレクトリ構造の可視化

```bash
# tree: ディレクトリ構造をツリー形式で表示
# brew install tree

tree                              # カレントディレクトリのツリー表示
tree -L 2                         # 2階層まで表示
tree -d                           # ディレクトリのみ
tree -a                           # 隠しファイル含む
tree -I "node_modules|.git"       # 特定ディレクトリを除外
tree -P "*.py"                    # Python ファイルのみ
tree --prune                      # 空ディレクトリを非表示
tree -s                           # ファイルサイズ表示
tree -D                           # 更新日時表示
tree -h                           # 人間が読みやすいサイズ表示
tree -f                           # フルパス表示
tree --du                         # ディレクトリの合計サイズ

# 実務例: プロジェクト構造の文書化
tree -L 3 -I "node_modules|.git|__pycache__|.next" > project_structure.txt
```

### 5.3 ファイル検索の自動化スクリプト

```bash
#!/bin/bash
# find_large_files.sh - 指定サイズ以上のファイルを検索してレポート

set -euo pipefail

SEARCH_DIR="${1:-.}"
MIN_SIZE="${2:-100M}"
OUTPUT_FILE="/tmp/large_files_report_$(date +%Y%m%d_%H%M%S).txt"

echo "=== 大容量ファイルレポート ===" > "$OUTPUT_FILE"
echo "検索ディレクトリ: $SEARCH_DIR" >> "$OUTPUT_FILE"
echo "最小サイズ: $MIN_SIZE" >> "$OUTPUT_FILE"
echo "実行日時: $(date)" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"

find "$SEARCH_DIR" -type f -size +"$MIN_SIZE" -printf "%s\t%p\n" 2>/dev/null \
  | sort -rn \
  | while IFS=$'\t' read -r size path; do
      hr_size=$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")
      echo "$hr_size  $path"
    done >> "$OUTPUT_FILE"

total=$(grep -c "^" "$OUTPUT_FILE" 2>/dev/null || echo "0")
echo "---" >> "$OUTPUT_FILE"
echo "合計: $((total - 5)) ファイル" >> "$OUTPUT_FILE"

echo "レポート出力先: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
```

```bash
#!/bin/bash
# find_duplicates.sh - 重複ファイルの検出

set -euo pipefail

SEARCH_DIR="${1:-.}"
echo "=== 重複ファイル検出 ==="
echo "検索ディレクトリ: $SEARCH_DIR"
echo ""

# 同じサイズのファイルをグループ化し、MD5で比較
find "$SEARCH_DIR" -type f ! -empty -printf "%s %p\n" 2>/dev/null \
  | sort -n \
  | uniq -w 10 -D \
  | awk '{print $2}' \
  | xargs -d '\n' md5sum 2>/dev/null \
  | sort \
  | uniq -w 32 -D \
  | awk '{
      if (prev_hash == substr($0, 1, 32)) {
          print "  DUP: " $2
      } else {
          if (NR > 1) print ""
          print "ORIG: " $2
      }
      prev_hash = substr($0, 1, 32)
    }'
```

```bash
#!/bin/bash
# cleanup_project.sh - プロジェクトの不要ファイル掃除

set -euo pipefail

PROJECT_DIR="${1:-.}"
DRY_RUN="${2:-true}"  # デフォルトはドライラン

echo "=== プロジェクトクリーンアップ ==="
echo "対象: $PROJECT_DIR"
echo "ドライラン: $DRY_RUN"
echo ""

# 削除対象パターン
PATTERNS=(
    "*.pyc"
    "__pycache__"
    ".DS_Store"
    "Thumbs.db"
    "*.swp"
    "*.swo"
    "*~"
    "*.bak"
    "*.orig"
)

total_size=0
total_count=0

for pattern in "${PATTERNS[@]}"; do
    files=$(find "$PROJECT_DIR" -name "$pattern" -not -path "*/.git/*" 2>/dev/null)
    if [ -n "$files" ]; then
        count=$(echo "$files" | wc -l)
        size=$(echo "$files" | xargs -I{} stat -f%z {} 2>/dev/null | awk '{s+=$1}END{print s+0}')
        total_count=$((total_count + count))
        total_size=$((total_size + size))

        echo "[$pattern] $count ファイル ($((size / 1024)) KB)"

        if [ "$DRY_RUN" = "false" ]; then
            echo "$files" | xargs rm -rf
            echo "  → 削除完了"
        fi
    fi
done

echo ""
echo "合計: $total_count ファイル ($((total_size / 1024)) KB)"
if [ "$DRY_RUN" = "true" ]; then
    echo "※ ドライランモードです。実際に削除するには第2引数に false を指定してください。"
fi
```

---

## 6. トラブルシューティング

### 6.1 find でよくあるエラーと対処法

```bash
# エラー: "Permission denied"
# 対処: 標準エラーを抑制
find / -name "*.conf" 2>/dev/null
find / -name "*.conf" 2>&1 | grep -v "Permission denied"

# エラー: "Argument list too long"（ファイルが多すぎる場合）
# 対処: xargs を使う
find . -name "*.log" -print0 | xargs -0 rm          # OK
# find . -name "*.log" -exec rm {} +                 # これも OK
# rm $(find . -name "*.log")                         # NG: 引数が多すぎる

# エラー: ファイル名にスペースや特殊文字が含まれる
# 対処: -print0 と xargs -0 を使う
find . -name "*.txt" -print0 | xargs -0 grep "keyword"

# エラー: -prune が期待通り動作しない
# 対処: -prune は -o (OR) と組み合わせる
find . -name ".git" -prune -o -type f -print        # 正しい
# find . -name ".git" -prune -type f -print          # 誤り

# エラー: macOS の find で GNU find のオプションが使えない
# 対処: GNU find をインストール
# brew install findutils
# gfind を使うか、PATH に追加
# macOS find は BSD find であり、-printf 等が使えない
gfind . -type f -printf "%T+ %p\n"                   # GNU find を使用

# -delete が最初の条件として使われた場合の警告
# 対処: 必ず条件を先に指定
find . -name "*.tmp" -delete                          # 正しい
# find . -delete -name "*.tmp"                        # 危険！全ファイル削除の恐れ
```

### 6.2 検索のデバッグ

```bash
# find の動作を確認するためのテクニック

# 1. まず -print で結果を確認してから -delete / -exec
find . -name "*.tmp" -print            # まず確認
find . -name "*.tmp" -delete           # 確認後に削除

# 2. -ok で対話的に確認
find . -name "*.bak" -ok rm {} \;      # 1つずつ確認

# 3. echo で実行されるコマンドを確認
find . -name "*.sh" -exec echo chmod +x {} \;   # 実際には実行しない

# 4. 件数を確認
find . -name "*.log" | wc -l           # マッチ件数を確認

# 5. fd の --list-file-types でサポートされるファイルタイプ確認
fd --list-file-types                   # fd が認識するファイルタイプ一覧
```

---

## まとめ

| ツール | 特徴 | 用途 | 速度 |
|--------|------|------|------|
| find | 標準搭載、高機能、POSIX互換 | 複雑な条件検索、スクリプト | 中 |
| fd | 高速、簡潔、.gitignore尊重 | 日常的な検索、開発作業 | 高 |
| locate | データベース検索 | ファイル名の素早い検索 | 最速 |
| which | PATH 検索 | コマンドの場所確認 | 即時 |
| whereis | バイナリ+man 検索 | コマンド関連ファイル確認 | 即時 |
| type | bash 組み込み | コマンドの種類確認 | 即時 |
| fzf | ファジー検索 | 対話的なファイル選択 | 即時 |
| tree | ツリー表示 | ディレクトリ構造の可視化 | 中 |

### 選択フローチャート

```
ファイルを探したい
  │
  ├─ ファイル名だけ分かっている → locate（高速）
  │
  ├─ 開発プロジェクト内の検索
  │    ├─ シンプルな検索 → fd（.gitignore 尊重、簡潔）
  │    └─ 複雑な条件    → find（論理演算子、-exec）
  │
  ├─ システム全体の検索
  │    ├─ 条件が名前だけ → locate
  │    └─ サイズ/日時/権限も → find
  │
  ├─ コマンドの場所 → which / type
  │
  └─ 対話的に選びたい → find/fd | fzf
```

---

## 13. find の高度なセキュリティ活用

### 13.1 ファイルシステムセキュリティ監査

```bash
# SUID/SGID ビットが設定されたファイルの検出（権限昇格のリスク）
find / -type f \( -perm -4000 -o -perm -2000 \) -ls 2>/dev/null

# ワールドライタブルなファイルの検出
find / -type f -perm -0002 -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null

# 所有者のないファイル（ユーザーやグループが削除されている）
find / -nouser -o -nogroup 2>/dev/null | head -50

# 最近変更された設定ファイルの確認（侵入の痕跡）
find /etc -type f -mtime -1 -ls 2>/dev/null

# 隠しファイルの検出（ドットファイル、不審なファイル）
find / -name ".*" -type f -not -path "/home/*" -not -path "/root/*" 2>/dev/null | head -50

# 実行可能ファイルで最近作成されたもの
find /tmp /var/tmp /dev/shm -type f -executable -newer /etc/hostname 2>/dev/null

# 不審な cron ジョブの検出
find /etc/cron* /var/spool/cron -type f -ls 2>/dev/null
find / -name "*.cron" -o -name "crontab" 2>/dev/null

# 大きすぎるログファイルの検出（DoS対策）
find /var/log -type f -size +100M -exec ls -lh {} \; 2>/dev/null
```

---

## 次に読むべきガイド
→ [[../02-text-processing/00-cat-less-head-tail.md]] — ファイル表示

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.4, O'Reilly, 2022.
2. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.17-18, 2019.
3. GNU Findutils Manual. https://www.gnu.org/software/findutils/manual/
4. fd GitHub Repository. https://github.com/sharkdp/fd
5. fzf GitHub Repository. https://github.com/junegunn/fzf
6. mlocate / plocate Documentation. https://plocate.sesse.net/
