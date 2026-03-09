# ファイル表示

> テキストファイルの内容を確認する方法は目的に応じて使い分ける。

## この章で学ぶこと

- [ ] ファイル表示コマンドを適切に使い分けられる
- [ ] cat / bat によるファイル全体の表示と結合を使いこなす
- [ ] less / more によるページャの操作を習得する
- [ ] head / tail による部分表示とリアルタイム監視をマスターする
- [ ] wc / diff / xxd 等の補助的な表示ツールを活用できる
- [ ] 実務でのファイル表示パターンを身につける


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. cat — ファイルの連結と表示

### 1.1 基本的な使い方

```bash
# 基本構文: cat [オプション] [ファイル...]
# cat は "concatenate"（連結）の略。元々は複数ファイルの連結が目的

# ファイルの全内容を表示
cat file.txt                     # 全内容を表示
cat /etc/hostname                # システムファイルの確認
cat ~/.bashrc                    # シェル設定の確認
cat /proc/cpuinfo                # CPU情報の表示（Linux）

# 複数ファイルの連結表示
cat file1.txt file2.txt          # 2つのファイルを連結して表示
cat header.txt body.txt footer.txt  # 3つのファイルを連結
cat *.log                        # 全ログファイルを連結表示

# 標準入力からの読み取り
cat                              # 標準入力をそのまま表示（Ctrl+D で終了）
cat -                            # 明示的に標準入力を指定
cat file1.txt - file2.txt        # file1 → 標準入力 → file2 を連結
```

### 1.2 表示オプション

```bash
# -n: 全行に行番号を付与
cat -n file.txt                  # 行番号付きで表示
cat -n /etc/passwd               # 設定ファイルの行番号確認

# -b: 空行以外に行番号を付与（-n より見やすい）
cat -b file.txt                  # 空行には番号を振らない

# -s: 連続する空行を1行にまとめる（squeeze）
cat -s file.txt                  # 連続空行を圧縮

# -A: 全ての特殊文字を可視化（-vET と同等）
cat -A file.txt                  # タブ(^I)、行末($)、制御文字を表示
# → 改行コードの確認、不可視文字のデバッグに非常に有用

# -E: 各行の末尾に $ を表示
cat -E file.txt                  # 行末の空白が可視化される

# -T: タブを ^I として表示
cat -T file.txt                  # タブとスペースの区別が明確に

# -v: 非印刷文字を可視化
cat -v file.txt                  # 制御文字を ^X 形式で表示

# 組み合わせ例
cat -bns file.txt                # 空行圧縮 + 空行以外に行番号 + 連続空行圧縮
cat -An file.txt                 # 行番号 + 全特殊文字表示
```

### 1.3 cat によるファイル操作

```bash
# ファイルの作成（リダイレクト）
cat > newfile.txt                # 標準入力をファイルに書き込み（上書き）
# （入力後 Ctrl+D で終了）

cat >> existingfile.txt          # ファイルに追記

# ヒアドキュメントによるファイル作成
cat > config.conf << 'EOF'
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
}
EOF

# ファイルの連結と保存
cat part1.csv part2.csv part3.csv > combined.csv
cat header.csv data.csv > report.csv

# 複数ファイルの結合（ヘッダー行の重複を避ける）
head -1 file1.csv > combined.csv
tail -n +2 file1.csv >> combined.csv
tail -n +2 file2.csv >> combined.csv
tail -n +2 file3.csv >> combined.csv

# パイプでの活用
cat file.txt | grep "pattern"    # grep に渡す（ただし grep pattern file.txt の方が効率的）
cat file.txt | sort | uniq       # ソートして重複除去
cat file.txt | tr 'a-z' 'A-Z'   # 大文字変換
```

### 1.4 cat の注意点と代替

```bash
# UUOC (Useless Use of Cat) — 不要な cat の使用を避ける
# 多くの場合、cat | command よりも command file の方が効率的

# 悪い例（不要な cat）
cat file.txt | grep "error"      # 不要な cat
cat file.txt | wc -l             # 不要な cat
cat file.txt | sort              # 不要な cat

# 良い例（直接ファイルを指定）
grep "error" file.txt            # 効率的
wc -l file.txt                   # 効率的
sort file.txt                    # 効率的

# ただし、複数ファイルの連結には cat が正当
cat *.log | grep "error"         # これは cat の正当な使用
cat header.txt body.txt | mail   # 連結して渡す

# 巨大ファイルへの注意
# cat は全内容をメモリに読み込むため、巨大ファイルには注意
# 大きなファイルには less や head/tail を使う
# cat large_file.txt             # 数GBのファイルでは問題になりうる
less large_file.txt              # ページ単位で表示（メモリ効率的）
```

---

## 2. bat — cat のモダン代替

### 2.1 インストールと基本

```bash
# インストール
brew install bat                 # macOS
sudo apt install bat             # Ubuntu/Debian（batcat という名前になることがある）
sudo pacman -S bat               # Arch Linux
cargo install bat                # Rust (Cargo)

# Ubuntu/Debian ではコマンド名が batcat になるため、エイリアスを設定
alias bat='batcat'

# bat の特徴
# - シンタックスハイライト（300+ 言語対応）
# - Git 統合（変更行のマーカー表示）
# - 自動ページャ（出力がターミナルに収まらない場合は less を使用）
# - 行番号の自動表示
# - テーマのカスタマイズ
```

### 2.2 基本的な使い方

```bash
# ファイルの表示（シンタックスハイライト付き）
bat file.py                      # Python ファイル（自動言語検出）
bat main.go                      # Go ファイル
bat index.html                   # HTML ファイル
bat config.yaml                  # YAML ファイル
bat Dockerfile                   # Dockerfile

# 言語の明示的指定
bat -l json file.txt             # JSON としてハイライト
bat -l sql query.txt             # SQL としてハイライト
bat -l markdown README           # Markdown としてハイライト

# 複数ファイル
bat file1.py file2.py            # 複数ファイルを表示（ヘッダー付き）
bat src/*.rs                     # Rust ファイルを全表示
```

### 2.3 表示カスタマイズ

```bash
# テーマの一覧と変更
bat --list-themes                # 利用可能なテーマ一覧
bat --theme="Dracula" file.py    # テーマを指定
bat --theme="Monokai Extended" file.py
bat --theme="Nord" file.py

# 環境変数でデフォルトテーマを設定
export BAT_THEME="Dracula"

# 表示スタイル（--style）
bat --style=full file.py         # 全要素表示（デフォルト）
bat --style=numbers file.py      # 行番号のみ
bat --style=plain file.py        # プレーンテキスト（ハイライトのみ）
bat --style=grid file.py         # グリッド線あり
bat --style=header file.py       # ヘッダーのみ
bat --style=changes file.py      # Git 変更マーカーのみ
bat --style=numbers,changes file.py  # 行番号 + Git 変更

# 行範囲の指定（-r / --range）
bat -r 10:20 file.py             # 10〜20行目のみ表示
bat -r :50 file.py               # 先頭から50行
bat -r 100: file.py              # 100行目以降
bat -r 10:20 -r 30:40 file.py    # 複数範囲

# ページャの無効化
bat --paging=never file.py       # ページャなし（cat と同等の動作）
bat -p file.py                   # プレーン出力（パイプ時のデフォルト）

# 行の折り返し
bat --wrap=auto file.py          # 自動折り返し
bat --wrap=never file.py         # 折り返しなし
```

### 2.4 bat の高度な活用

```bash
# パイプでの使用（シンタックスハイライト付き表示）
curl -s https://example.com/api | bat -l json       # API レスポンスをハイライト
docker inspect container_id | bat -l json            # Docker 情報をハイライト
kubectl get pod -o yaml | bat -l yaml                # K8s リソースをハイライト
git diff | bat                                        # Git diff をハイライト

# diff の表示（batdiff / delta との組み合わせ）
bat --diff file1.txt file2.txt   # 差分をハイライト表示

# man ページのハイライト（~/.bashrc に追加）
export MANPAGER="sh -c 'col -bx | bat -l man -p'"
# → man コマンドの出力がシンタックスハイライトされる

# 対応言語の一覧
bat --list-languages             # サポートされている言語一覧

# カスタム言語マッピング
bat --map-syntax "*.conf:INI" file.conf    # .conf を INI としてハイライト
bat --map-syntax ".env:Dotenv" .env        # .env をハイライト

# .bashrc / .zshrc でのエイリアス設定
alias cat='bat --paging=never'   # cat を bat に置き換え
alias catp='bat --style=plain'   # プレーン表示用
alias catl='bat --style=numbers' # 行番号付き表示用
```

---

## 3. less — ページャ（大きなファイルの閲覧）

### 3.1 基本操作

```bash
# 基本: less [オプション] [ファイル]
# less は "less is more" から。more の改良版で、双方向スクロールが可能
# ファイル全体をメモリに読み込まないため、巨大ファイルでも高速

less file.txt                    # ファイルをページ単位で表示
less +F logfile.log              # tail -f モードで開始
less +/pattern file.txt          # pattern を検索した状態で開く
less +100 file.txt               # 100行目から表示
less -N file.txt                 # 行番号付きで表示
less -S file.txt                 # 長い行を折り返さない（水平スクロール可能）
less -R file.txt                 # ANSIカラーコードを解釈
```

### 3.2 less 内のキー操作（完全リファレンス）

```
=== ナビゲーション ===
j / ↓ / Enter      1行下にスクロール
k / ↑              1行上にスクロール
Space / f / PgDn   1ページ下にスクロール
b / PgUp           1ページ上にスクロール
d                  半ページ下にスクロール
u                  半ページ上にスクロール
g / Home           ファイルの先頭に移動
G / End            ファイルの末尾に移動
50g                50行目に移動（行番号指定）
50%                ファイルの50%の位置に移動

=== 検索 ===
/pattern           前方検索（下方向）
?pattern           後方検索（上方向）
n                  次のマッチに移動
N                  前のマッチに移動
&pattern           パターンにマッチする行のみ表示（フィルタリング）
&                  フィルタをクリア

=== 検索の正規表現 ===
/error             "error" を含む行を検索
/^ERROR            行頭が "ERROR" の行
/error|warning     "error" または "warning"
/[0-9]{3}          3桁の数字

=== マーク ===
ma                 現在位置にマーク 'a' を設定
'a                 マーク 'a' の位置に移動
''                 前の位置に戻る

=== 表示制御 ===
-N                 行番号の表示/非表示をトグル
-S                 長い行の折り返しをトグル
-i                 検索の大小文字無視をトグル
-R                 ANSIカラーの解釈をトグル
=                  現在のファイル情報を表示（行数、パーセンテージ等）
v                  $VISUAL / $EDITOR でファイルを編集

=== ファイル操作 ===
:e filename        別のファイルを開く
:n                 次のファイル（複数ファイル指定時）
:p                 前のファイル（複数ファイル指定時）

=== その他 ===
F                  tail -f モード（ファイル末尾をリアルタイム監視、Ctrl+C で解除）
q                  終了
h                  ヘルプ表示
```

### 3.3 less の実務活用パターン

```bash
# ログファイルの閲覧（カラー対応）
less -R /var/log/syslog          # カラー出力を保持
journalctl | less -R             # systemd ジャーナルの閲覧

# 巨大ファイルの効率的な閲覧
less -N large_file.csv           # 行番号付きで CSV を閲覧
less -S wide_file.csv            # 横に長い CSV を折り返さず閲覧

# 複数ファイルの順次閲覧
less file1.txt file2.txt file3.txt   # :n と :p で切り替え

# パイプからの入力
ps aux | less                    # プロセス一覧をページャで閲覧
find / -name "*.conf" 2>/dev/null | less  # 検索結果をページャで閲覧
docker logs container_name | less -R       # Docker ログの閲覧
kubectl logs pod_name | less -R            # K8s ログの閲覧

# 圧縮ファイルの閲覧（自動解凍）
zless file.gz                    # gzip ファイルを直接閲覧
bzless file.bz2                  # bzip2 ファイルを直接閲覧
xzless file.xz                   # xz ファイルを直接閲覧

# LESSOPEN / LESSCLOSE で前処理を設定
# ~/.bashrc に追加すると、様々な形式のファイルを less で閲覧可能に
export LESSOPEN="| /usr/bin/lesspipe %s"
export LESSCLOSE="/usr/bin/lesspipe %s %s"

# 環境変数で less のデフォルトオプションを設定
export LESS="-R -N -S --mouse"
# -R: ANSIカラー解釈
# -N: 行番号表示
# -S: 折り返しなし
# --mouse: マウススクロール対応

# less をデフォルトページャとして設定
export PAGER='less'
export MANPAGER='less -R'
```

### 3.4 more（レガシーページャ）

```bash
# more: less の前身。前方スクロールのみ
more file.txt                    # ページ単位で表示
more -d file.txt                 # ヘルプメッセージ表示
more -10 file.txt                # 10行ずつ表示
more +/pattern file.txt          # パターン検索位置から表示

# more は POSIX 標準だが、実務では less を使うのが一般的
# 多くのシステムで more は less にエイリアスされている
```

---

## 4. head — ファイルの先頭部分を表示

### 4.1 基本的な使い方

```bash
# 基本: head [オプション] [ファイル...]
# デフォルトは先頭10行

head file.txt                    # 先頭10行を表示
head -n 20 file.txt              # 先頭20行を表示
head -n 5 file.txt               # 先頭5行を表示
head -1 file.txt                 # 先頭1行のみ（ヘッダー行の確認に便利）

# バイト数で指定
head -c 100 file.txt             # 先頭100バイトを表示
head -c 1K file.txt              # 先頭1KBを表示
head -c 1M file.txt              # 先頭1MBを表示

# 末尾N行/バイトを除いた表示
head -n -5 file.txt              # 末尾5行を除いた全体を表示
head -c -100 file.txt            # 末尾100バイトを除いた全体を表示

# 複数ファイル
head file1.txt file2.txt         # 各ファイルのヘッダー付きで先頭10行
head -n 5 *.csv                  # 全CSVファイルの先頭5行
head -q -n 3 *.txt               # ヘッダーなしで先頭3行（-q = quiet）
```

### 4.2 head の実務活用

```bash
# CSV/TSVのヘッダー行確認
head -1 data.csv                 # 列名の確認
head -1 data.csv | tr ',' '\n'  # 列名を縦に表示

# ファイルの形式確認
head -c 16 file.bin | xxd        # バイナリファイルのマジックバイト確認
head -3 script.sh                # シバンラインの確認

# ログファイルの冒頭確認
head -20 /var/log/syslog         # ログの最初の20行

# パイプでの活用
ls -la | head -5                 # ディレクトリ一覧の先頭5件
ps aux --sort=-%mem | head -11   # メモリ使用量上位10プロセス（ヘッダー+10行）
du -sh * | sort -rh | head -10   # ディスク使用量トップ10

# ランダムな行の取得（head + shuf / sort -R）
shuf -n 5 file.txt               # ランダムに5行取得
sort -R file.txt | head -5       # ランダムに5行取得（代替方法）

# 大きなファイルの先頭部分だけ処理
head -n 1000 huge_file.csv > sample.csv   # サンプルデータの抽出
head -n 1000000 access.log | awk '{print $1}' | sort | uniq -c | sort -rn  # 先頭100万行を分析
```

---

## 5. tail — ファイルの末尾部分を表示

### 5.1 基本的な使い方

```bash
# 基本: tail [オプション] [ファイル...]
# デフォルトは末尾10行

tail file.txt                    # 末尾10行を表示
tail -n 20 file.txt              # 末尾20行を表示
tail -n 5 file.txt               # 末尾5行を表示
tail -1 file.txt                 # 末尾1行のみ

# バイト数で指定
tail -c 100 file.txt             # 末尾100バイトを表示
tail -c 1K file.txt              # 末尾1KBを表示

# 先頭N行/バイトをスキップした表示
tail -n +2 file.txt              # 2行目以降を表示（ヘッダースキップ）
tail -n +11 file.txt             # 11行目以降を表示
tail -c +100 file.txt            # 100バイト目以降を表示

# 複数ファイル
tail file1.txt file2.txt         # 各ファイルの末尾10行
tail -q -n 5 *.log               # ヘッダーなしで各ファイルの末尾5行
```

### 5.2 tail -f / -F — リアルタイムログ監視

```bash
# -f: ファイル末尾をリアルタイムで追跡（follow）
tail -f /var/log/syslog          # システムログのリアルタイム監視
tail -f /var/log/nginx/access.log   # Nginx アクセスログの監視
tail -f /var/log/nginx/error.log    # Nginx エラーログの監視
tail -f app.log                     # アプリケーションログの監視

# -F: ファイルのローテーションに追従（-f --retry と同等）
tail -F /var/log/syslog          # ログローテーション後も追跡を継続
# -f はファイルディスクリプタを追跡するため、ファイルが置き換わると追跡が切れる
# -F はファイル名を追跡するため、ローテーション後も自動で再接続

# 新規行のみ表示（既存の内容を表示しない）
tail -f -n 0 logfile             # 新規追加行のみ表示
tail -f -n 0 /var/log/syslog    # 監視開始後の新規ログのみ

# 複数ファイルの同時監視
tail -f /var/log/nginx/access.log /var/log/nginx/error.log
# → ファイル名のヘッダー付きで両方を表示

# tail -f + grep でリアルタイムフィルタリング
tail -f /var/log/syslog | grep "ERROR"           # エラーのみ表示
tail -f /var/log/syslog | grep --line-buffered "ERROR"  # バッファリング無効
tail -f /var/log/syslog | grep -i "error\|warning"      # エラーと警告
tail -f access.log | grep "500\|502\|503"                # HTTP 5xx エラー

# tail -f + awk でリアルタイム集計
tail -f access.log | awk '{print $9}' | sort | uniq -c   # ステータスコード集計
tail -f access.log | awk '$9 >= 500 {print}'             # 5xx エラーのみ

# tail -f でタイムスタンプ付き出力
tail -f app.log | while read line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $line"
done

# 監視の終了
# Ctrl+C で tail -f を終了
```

### 5.3 tail の実務パターン

```bash
# ログの最新エントリ確認
tail -100 /var/log/syslog        # 直近100行のログ
tail -n 50 /var/log/auth.log     # 認証ログの直近50行

# CSV のヘッダースキップ
tail -n +2 data.csv              # ヘッダー行を除いたデータ部分
tail -n +2 data.csv | wc -l      # データ行数のカウント（ヘッダー除外）

# ファイルの特定範囲を取得（head + tail の組み合わせ）
head -n 20 file.txt | tail -n 5  # 16〜20行目を表示
sed -n '16,20p' file.txt         # 同じ結果（sed 版）

# 最新のログエントリからエラーを抽出
tail -1000 /var/log/app.log | grep "ERROR" | tail -20   # 直近1000行中のエラー最新20件

# デプロイ後のログ確認パターン
tail -f /var/log/app.log &       # バックグラウンドで監視開始
deploy_command                    # デプロイ実行
# ログを確認後
kill %1                          # バックグラウンドの tail を停止

# ログの差分確認（前回確認時点からの新規ログ）
wc -l < /var/log/app.log > /tmp/log_lines   # 現在の行数を保存
# ... 時間経過後 ...
tail -n +$(cat /tmp/log_lines) /var/log/app.log  # 前回以降の新規ログ
```

---

## 6. multitail — 複数ログの同時監視

```bash
# インストール
brew install multitail            # macOS
sudo apt install multitail        # Ubuntu/Debian

# 基本的な使い方
multitail /var/log/syslog /var/log/auth.log    # 画面分割で2ファイル同時監視
multitail -s 2 /var/log/*.log                  # 2列に分割して複数ログ監視

# カラーリング付き監視
multitail -ci green /var/log/access.log -ci red /var/log/error.log

# フィルタ付き監視
multitail -e "ERROR" /var/log/app.log          # ERROR を含む行のみ表示

# 代替: tmux / screen でのログ監視
# tmux で画面分割して各ペインで tail -f を実行するのも一般的
```

---

## 7. wc — ワードカウント

### 7.1 基本的な使い方

```bash
# 基本: wc [オプション] [ファイル...]
# デフォルトは行数・単語数・バイト数の3つを表示

wc file.txt                      # 行数 単語数 バイト数 ファイル名
wc -l file.txt                   # 行数のみ
wc -w file.txt                   # 単語数のみ
wc -c file.txt                   # バイト数のみ
wc -m file.txt                   # 文字数（マルチバイト対応）
wc -L file.txt                   # 最長行の長さ（文字数）

# 複数ファイル
wc -l *.py                       # 各 .py ファイルの行数 + 合計
wc -l src/*.go                   # Go ソースファイルの行数

# パイプでの活用
cat file.txt | wc -l             # パイプ経由の行数カウント
ls -1 | wc -l                    # ディレクトリ内のファイル数
ps aux | wc -l                   # プロセス数
grep -c "error" file.txt         # パターンにマッチする行数（grep -c の方が効率的）
```

### 7.2 wc の実務パターン

```bash
# ソースコードの行数カウント
find . -name "*.py" -exec wc -l {} + | tail -1   # Python 全体の行数
find . -name "*.go" -exec wc -l {} + | sort -n    # Go ファイルを行数順に
find . \( -name "*.js" -o -name "*.ts" \) -exec wc -l {} + | sort -rn | head -20
# → 行数の多い JS/TS ファイル トップ20

# ファイルサイズの確認
wc -c large_file.bin             # バイト数でサイズ確認
wc -c < file.txt                 # ファイル名なしでバイト数のみ出力

# ディレクトリ内のファイル数
find . -type f | wc -l           # 再帰的にファイル数をカウント
find . -maxdepth 1 -type f | wc -l  # カレントディレクトリのみ

# 空行の数
grep -c "^$" file.txt            # 空行の数
grep -cv "^$" file.txt           # 空行以外の数

# コード行数（空行・コメント除外）
grep -cv "^$\|^#\|^//" file.py   # 空行とコメント行を除外した行数

# 複数言語のプロジェクト行数集計
echo "=== プロジェクト行数レポート ==="
for ext in py js ts go rs; do
    count=$(find . -name "*.$ext" -exec cat {} + 2>/dev/null | wc -l)
    echo "$ext: $count 行"
done
```

---

## 8. diff — ファイル差分の表示

### 8.1 基本的な使い方

```bash
# 基本: diff [オプション] ファイル1 ファイル2

diff file1.txt file2.txt         # デフォルト形式で差分表示
diff -u file1.txt file2.txt      # unified 形式（Git と同じ形式）
diff -c file1.txt file2.txt      # context 形式（前後のコンテキスト付き）
diff -y file1.txt file2.txt      # 横並び表示（side-by-side）
diff --color file1.txt file2.txt # カラー表示

# unified 形式の読み方
# --- file1.txt（変更前のファイル）
# +++ file2.txt（変更後のファイル）
# @@ -1,5 +1,6 @@（変更箇所の行範囲）
# -（削除された行）
# +（追加された行）
#  （変更なしの行）

# オプション
diff -u -B file1.txt file2.txt   # 空行の差異を無視
diff -u -w file1.txt file2.txt   # 全ての空白の差異を無視
diff -u -b file1.txt file2.txt   # 空白の量の変化を無視
diff -u -i file1.txt file2.txt   # 大小文字の差異を無視
diff -u --ignore-blank-lines file1.txt file2.txt  # 空行の追加/削除を無視
```

### 8.2 ディレクトリの比較

```bash
# ディレクトリの再帰比較
diff -r dir1/ dir2/              # 全ファイルの差分を表示
diff -rq dir1/ dir2/             # 異なるファイルのリストのみ表示
diff -r --brief dir1/ dir2/      # -rq と同じ

# 特定ファイルを除外して比較
diff -r --exclude="*.pyc" dir1/ dir2/
diff -r --exclude=".git" dir1/ dir2/
diff -r --exclude="node_modules" dir1/ dir2/

# パッチファイルの生成と適用
diff -u original.py modified.py > changes.patch  # パッチ生成
patch original.py < changes.patch                 # パッチ適用
patch -R original.py < changes.patch              # パッチ取り消し

# ディレクトリ全体のパッチ
diff -ruN dir1/ dir2/ > all_changes.patch
cd dir1/ && patch -p1 < all_changes.patch
```

### 8.3 モダンな diff ツール

```bash
# colordiff: diff にカラーを追加
# brew install colordiff
colordiff -u file1.txt file2.txt

# delta: Git スタイルの美しい差分表示
# brew install git-delta
diff -u file1.txt file2.txt | delta

# icdiff: インライン比較
# pip install icdiff
icdiff file1.txt file2.txt       # 横並びのカラー差分

# vimdiff: Vim 上での差分表示と編集
vimdiff file1.txt file2.txt      # Vim で差分を表示

# Git の設定で delta を使う
# git config --global core.pager delta
# git config --global delta.side-by-side true
```

---

## 9. その他のファイル表示ツール

### 9.1 xxd / hexdump — バイナリファイルの表示

```bash
# xxd: 16進ダンプ
xxd file.bin                     # 16進ダンプ表示
xxd -l 64 file.bin               # 先頭64バイトのみ
xxd -s 0x100 file.bin            # オフセット 0x100 から表示
xxd -c 8 file.bin                # 1行8バイトで表示（デフォルト16）
xxd -p file.bin                  # プレーン16進出力（アドレス・ASCII部分なし）
xxd -b file.bin                  # 2進数で表示
xxd -r hex.txt > file.bin        # 16進テキストからバイナリに逆変換

# hexdump
hexdump -C file.bin              # 16進 + ASCII 表示（最も一般的）
hexdump -n 32 file.bin           # 先頭32バイト
hexdump -s 256 file.bin          # 256バイト目から表示

# od (octal dump)
od -A x -t x1z file.bin         # 16進表示（POSIX 標準）
od -c file.bin                   # 文字として表示

# ファイルタイプの判定
file file.bin                    # ファイルの種類を判定
file -i file.txt                 # MIME タイプを表示
file -b file.bin                 # ファイル名を省略
```

### 9.2 strings — バイナリ内のテキスト抽出

```bash
# strings: バイナリファイルから可読テキストを抽出
strings binary_file              # 印刷可能な文字列を抽出
strings -n 10 binary_file        # 10文字以上の文字列のみ
strings -a binary_file           # ファイル全体を対象
strings binary_file | grep "password"   # パスワード文字列の検索
strings binary_file | grep -i "version" # バージョン情報の検索
strings /usr/bin/python3 | grep -i copyright   # 著作権情報

# 実務例: コアダンプからの情報抽出
strings core.dump | grep "Error"
```

### 9.3 column — 列の整形表示

```bash
# column: テキストを列に整形
column -t file.txt               # 列を揃えて表示（空白区切り）
column -t -s ',' file.csv        # CSV を列揃えで表示
column -t -s ':' /etc/passwd     # /etc/passwd を見やすく表示
column -t -s $'\t' data.tsv      # TSV を列揃えで表示

# mount の出力を整形
mount | column -t

# 実務例: CSV を見やすく表示
head -20 data.csv | column -t -s ','
```

### 9.4 nl — 行番号の付与

```bash
# nl: 行番号を付与（cat -n より柔軟）
nl file.txt                      # 行番号付き（空行は番号なし）
nl -ba file.txt                  # 全行に番号を振る（空行含む）
nl -w 4 file.txt                 # 行番号の幅を4桁に
nl -s ': ' file.txt              # 区切り文字を ': ' に
nl -n rz file.txt                # ゼロ埋め右寄せ（001, 002, ...）
nl -v 0 file.txt                 # 行番号を0から開始
```

### 9.5 rev / tac — 反転表示

```bash
# tac: ファイルを逆順に表示（cat の逆）
tac file.txt                     # 最終行から先頭行の順で表示
tac access.log | head -20        # 最新のログ20行を取得

# rev: 各行の文字列を反転
rev file.txt                     # 各行を右から左に表示
echo "hello" | rev               # "olleh"

# 実務例: 最新のログエントリから検索
tac /var/log/syslog | grep -m 1 "ERROR"   # 最新のERRORを1件取得
```

### 9.6 fold / fmt — テキストの折り返し

```bash
# fold: 指定幅で行を折り返す
fold -w 80 file.txt              # 80文字で折り返し
fold -s -w 80 file.txt           # 単語の途中で折り返さない（-s = space）

# fmt: テキストのフォーマット
fmt -w 72 file.txt               # 72文字幅でフォーマット
fmt -s file.txt                  # 短い行はそのまま（長い行のみ折り返し）
```

---

## 10. 実務パターン集

### 10.1 ログ分析の基本パターン

```bash
# 直近のエラーを確認
tail -100 /var/log/app.log | grep -i "error"

# エラーの発生頻度を確認
grep -c "ERROR" /var/log/app.log

# 時系列でエラーの推移を確認
grep "ERROR" /var/log/app.log | awk '{print $1, $2}' | cut -d: -f1,2 | uniq -c

# 特定時間帯のログを抽出
sed -n '/2026-02-16 14:00/,/2026-02-16 15:00/p' /var/log/app.log

# リアルタイムでエラーを監視しつつファイルにも保存
tail -f /var/log/app.log | tee -a /tmp/monitor.log | grep "ERROR"
```

### 10.2 CSV/TSV データの確認パターン

```bash
# ヘッダーの確認
head -1 data.csv

# データのサンプル表示
head -5 data.csv | column -t -s ','

# 行数の確認（ヘッダー除外）
tail -n +2 data.csv | wc -l

# 列数の確認
head -1 data.csv | tr ',' '\n' | wc -l

# 特定列の値一覧
awk -F',' '{print $3}' data.csv | sort -u | head -20

# データの整合性チェック（全行の列数が一致するか）
awk -F',' '{print NF}' data.csv | sort -u
```

### 10.3 設定ファイルの確認パターン

```bash
# コメントと空行を除いた設定内容の確認
grep -v "^#\|^$\|^;" config.ini

# 設定ファイルの実効行数
grep -cv "^#\|^$\|^;" /etc/nginx/nginx.conf

# 特定の設定値を確認
grep "^listen" /etc/nginx/sites-enabled/*
grep "^port" /etc/redis/redis.conf

# 設定ファイル間の差分確認
diff -u /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak
```

### 10.4 パフォーマンス分析パターン

```bash
# /proc からのシステム情報確認（Linux）
cat /proc/meminfo                # メモリ情報
cat /proc/cpuinfo | head -30     # CPU 情報
cat /proc/loadavg                # ロードアベレージ
cat /proc/uptime                 # 稼働時間

# ディスク使用状況
df -h | head -10                 # ファイルシステム使用状況
du -sh /var/log/*  | sort -rh | head -10  # ログディレクトリのサイズ

# ネットワーク情報
cat /proc/net/tcp | head -5      # TCP 接続情報
ss -tlnp | head -20              # リスニングポート
```

---

## 2. 使い分けガイド

```
目的に応じた選択:

  ファイル全体を見たい      → cat (小ファイル) / bat (コード) / less (大ファイル)
  先頭/末尾だけ見たい       → head / tail
  ログをリアルタイム監視    → tail -f / tail -F / multitail
  行数を数えたい            → wc -l
  2つのファイルの差分       → diff -u / delta / icdiff
  バイナリファイルの確認    → xxd / hexdump / file
  CSV を見やすく表示        → column -t -s ','
  行番号を付けて表示        → cat -n / nl / bat
  逆順に表示                → tac
  テキスト内の文字列抽出    → strings

ファイルサイズ別の推奨:
  〜100行     → cat / bat
  100〜1000行 → bat / less
  1000行〜    → less（必須）
  数GB        → less -N（head/tail で部分表示も検討）
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| コマンド | 用途 | 特徴 |
|---------|------|------|
| cat | ファイル全体表示・連結 | 標準、シンプル |
| bat | シンタックスハイライト表示 | モダン、Git統合 |
| less | ページ表示（大ファイル向け） | 双方向スクロール、検索 |
| more | ページ表示（レガシー） | 前方スクロールのみ |
| head | 先頭部分の表示 | 高速、パイプに最適 |
| tail | 末尾部分の表示 | ログ監視に必須 |
| tail -f/-F | リアルタイムログ監視 | ローテーション追従 |
| wc | 行数・単語数・バイト数 | 集計に必須 |
| diff | ファイル差分 | パッチ生成にも対応 |
| xxd | 16進ダンプ | バイナリ解析 |
| column | 列の整形表示 | CSV/TSV の可読性向上 |
| nl | 行番号付与 | cat -n より柔軟 |
| tac | 逆順表示 | 最新のログから検索 |
| strings | テキスト抽出 | バイナリ内の文字列検索 |

---

## 13. ファイル表示のセキュリティとトラブルシューティング

### 13.1 安全なファイル表示

```bash
# バイナリファイルの誤表示を防ぐ
# file コマンドで事前にファイルタイプを確認
file suspicious_file.txt
# → ASCII text であればテキストファイル
# → data, executable, ELF 等であればバイナリ

# バイナリファイルを誤って cat すると端末が壊れることがある
# その場合の復旧
reset                            # 端末をリセット
stty sane                        # 端末設定を正常に戻す
echo -e "\033c"                  # ESCシーケンスでリセット

# 安全なファイル表示関数
safe_cat() {
  local file="$1"
  if [ ! -f "$file" ]; then
    echo "Error: File not found: $file" >&2
    return 1
  fi
  local filetype=$(file -b "$file")
  case "$filetype" in
    *text*|*ASCII*|*UTF-8*|*empty*)
      cat "$file"
      ;;
    *)
      echo "Warning: Binary file detected ($filetype)" >&2
      echo "Use 'xxd $file | less' or 'strings $file' instead" >&2
      return 1
      ;;
  esac
}

# 大きなファイルの確認（意図しない大量出力を防ぐ）
safe_view() {
  local file="$1"
  local max_lines="${2:-1000}"
  local total_lines=$(wc -l < "$file")
  if [ "$total_lines" -gt "$max_lines" ]; then
    echo "Warning: File has $total_lines lines (showing first $max_lines)" >&2
    head -n "$max_lines" "$file"
    echo "... (truncated, $((total_lines - max_lines)) more lines)" >&2
  else
    cat "$file"
  fi
}

# 機密情報を含むファイルの安全な表示
# パスワードやトークンをマスクして表示

# 環境変数ファイルの安全な表示
cat .env | sed -E 's/^([^=]+=).+$/\1***/'
```

### 13.2 エンコーディングの確認と変換

```bash
# ファイルのエンコーディング確認
file -i document.txt              # MIME タイプとエンコーディング
# → text/plain; charset=utf-8
# → text/plain; charset=iso-8859-1

# nkf を使ったエンコーディング判定（日本語ファイル向け）
nkf --guess document.txt
# → UTF-8 (LF)
# → Shift_JIS (CRLF)
# → EUC-JP (LF)

# 文字化けの対処法
# UTF-8 として表示
cat document.txt | iconv -f SHIFT_JIS -t UTF-8
cat document.txt | nkf -w                      # UTF-8 に変換して表示

# 改行コードの確認
cat -A document.txt | head -3     # ^M が見えれば CRLF（Windows形式）
file document.txt                 # "CRLF line terminators" の有無
xxd document.txt | head -5        # 0d 0a が CRLF

# 改行コードの変換
cat document.txt | tr -d '\r' > unix_file.txt   # CRLF → LF
# または
dos2unix document.txt             # dos2unix コマンド
unix2dos document.txt             # LF → CRLF

# BOM の確認と除去
xxd document.txt | head -1        # ef bb bf が UTF-8 BOM
head -c 3 document.txt | xxd      # 最初の3バイトを確認
sed -i '1s/^\xEF\xBB\xBF//' document.txt  # BOM 除去
```

### 13.3 パフォーマンスの考慮

```bash
# 大量のファイルを効率的に処理する
# cat は小さなファイルの連結に最適
# 大量の小ファイルの連結
cat part_*.txt > combined.txt                    # 高速
find . -name "part_*.txt" -exec cat {} + > combined.txt  # 多すぎる場合

# 巨大ファイルの部分表示（全体を読まない）
head -n 100 huge_file.txt          # 先頭のみ読む（O(1)に近い）
tail -n 100 huge_file.txt          # 末尾のみ読む
sed -n '1000,1100p' huge_file.txt  # 特定範囲のみ読む

# wc の高速化
wc -l huge_file.txt                # 行数だけならメモリ効率が良い
wc -c huge_file.txt                # バイト数はさらに高速（seekで済む場合がある）

# パイプラインの効率化
# 不要なパイプを減らす（UUOC = Useless Use of Cat の応用）
# 悪い例:
cat file.txt | wc -l                           # cat が無駄
cat file.txt | head -10                        # cat が無駄
# 良い例:
wc -l < file.txt                               # リダイレクトを使う
head -10 file.txt                              # 直接ファイルを引数に

# ディスクI/Oの最適化
# 同じファイルを何度も読む場合はキャッシュを活用
lines=$(wc -l < file.txt)
first=$(head -1 file.txt)
last=$(tail -1 file.txt)
# ↓ 代わりにteeで一度のパスで処理
tee >(wc -l > /tmp/count) >(head -1 > /tmp/first) >(tail -1 > /tmp/last) < file.txt > /dev/null
```

### 13.4 less の高度なカスタマイズ

```bash
# less の環境変数設定
export LESS='-R -F -X -S -i -M'
# -R: ANSIカラーを解釈
# -F: 1画面に収まる場合は自動終了
# -X: 終了時に画面をクリアしない
# -S: 長い行を折り返さない
# -i: 検索で大文字小文字を無視
# -M: 詳細なプロンプトを表示

# LESSOPEN / LESSCLOSE でプリプロセッサを設定
# lesspipe が利用可能な場合
eval "$(lesspipe)"
# → tar, gz, zip, pdf, 画像ファイルなどを less で直接閲覧可能

# less のカラー設定
export LESS_TERMCAP_mb=$'\e[1;31m'     # 点滅開始（赤太字）
export LESS_TERMCAP_md=$'\e[1;36m'     # 太字開始（シアン太字）
export LESS_TERMCAP_me=$'\e[0m'         # モード終了
export LESS_TERMCAP_se=$'\e[0m'         # 強調終了
export LESS_TERMCAP_so=$'\e[1;44;33m'  # 強調開始（青背景黄文字）
export LESS_TERMCAP_ue=$'\e[0m'         # 下線終了
export LESS_TERMCAP_us=$'\e[1;32m'     # 下線開始（緑太字）

# less でのファイルタイプ別表示
# Markdown ファイルを整形して表示
mdless() {
  if command -v glow &>/dev/null; then
    glow "$1" | less -R
  elif command -v bat &>/dev/null; then
    bat --style=plain --paging=always "$1"
  else
    less "$1"
  fi
}

# JSON ファイルを整形して表示
jless() {
  if command -v jq &>/dev/null; then
    jq -C '.' "$1" | less -R
  else
    python3 -m json.tool "$1" | less
  fi
}
```

---

## 次に読むべきガイド

---

## 参考文献
1. Shotts, W. "The Linux Command Line." 2nd Ed, Ch.6, 2019.
2. Barrett, D. "Efficient Linux at the Command Line." Ch.3, O'Reilly, 2022.
3. bat GitHub Repository. https://github.com/sharkdp/bat
4. delta GitHub Repository. https://github.com/dandavison/delta
5. GNU Coreutils Manual. https://www.gnu.org/software/coreutils/manual/
