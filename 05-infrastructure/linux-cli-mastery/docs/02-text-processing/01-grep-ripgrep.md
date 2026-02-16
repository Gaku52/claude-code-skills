# パターン検索（grep / ripgrep）

> grep は「テキストの中から必要な行を抽出する」最も重要なフィルタリングツール。

## この章で学ぶこと

- [ ] grep の主要オプションを使いこなせる
- [ ] 正規表現を活用した検索ができる
- [ ] ripgrep（rg）で高速な再帰検索ができる
- [ ] grep 系ツール（egrep, fgrep, ag, ack）の使い分けができる
- [ ] 実務で頻出する検索パターンを身につける

---

## 1. grep の基本

### 1.1 基本構文と動作

```bash
# 基本構文: grep [オプション] パターン [ファイル...]
#
# grep は入力の各行に対してパターンをマッチングし、
# マッチした行を標準出力に出力する。
# パターンはデフォルトで基本正規表現（BRE）として解釈される。

# 基本的な文字列検索
grep "error" logfile.txt            # "error" を含む行を表示
grep "warning" logfile.txt          # "warning" を含む行を表示
grep "fatal" logfile.txt            # "fatal" を含む行を表示

# ファイルを指定しない場合は標準入力から読む
echo "hello world" | grep "world"   # "world" を含む行
ps aux | grep nginx                 # パイプ経由で検索
cat /etc/passwd | grep "root"       # /etc/passwd から root を検索

# 複数ファイルの検索
grep "error" *.log                  # 全 .log ファイルから検索
grep "TODO" src/*.py                # Python ファイルから TODO を検索
grep "import" lib/*.js              # JS ファイルから import を検索
```

### 1.2 主要オプション（出力制御）

```bash
# -i: 大小文字を無視（ignore case）
grep -i "error" logfile.txt         # Error, ERROR, error 全てマッチ
grep -i "warning" logfile.txt       # Warning, WARNING 等もマッチ

# -n: 行番号を表示
grep -n "error" logfile.txt         # "42:error occurred" のように行番号付き
grep -n "TODO" src/*.py             # ファイル名:行番号:マッチ行

# -c: マッチした行数をカウント
grep -c "error" logfile.txt         # マッチ行数を数値で表示
grep -c "error" *.log               # 各ファイルのマッチ行数

# -l: マッチしたファイル名のみ表示（ファイル内容は表示しない）
grep -l "error" *.log               # error を含むログファイルの名前
grep -rl "TODO" src/                # 再帰的に TODO を含むファイル名
grep -rL "test" src/                # test を含まないファイル名（-L は -l の逆）

# -v: 逆マッチ（マッチしない行を表示）
grep -v "debug" logfile.txt         # debug を含まない行
grep -v "^#" config.conf            # コメント行以外
grep -v "^$" file.txt               # 空行以外

# -w: 単語として完全一致（word match）
grep -w "error" logfile.txt         # "error" に完全一致（"errors" はマッチしない）
grep -w "log" file.txt              # "log" に一致（"logging" はマッチしない）
grep -w "main" *.py                 # 単語 "main" を検索

# -x: 行全体が完全一致
grep -x "hello" file.txt            # 行全体が "hello" の行のみ

# -o: マッチした部分のみ表示（行全体ではなく）
grep -o "error[a-z]*" logfile.txt   # "error" で始まる単語だけ抽出
grep -oP "\d+\.\d+\.\d+\.\d+" access.log  # IPアドレスを抽出

# -q: 何も出力しない（終了コードのみ、スクリプト用）
if grep -q "error" logfile.txt; then
    echo "エラーが見つかりました"
fi
```

### 1.3 コンテキスト表示（-A / -B / -C）

```bash
# -A N: マッチ行の後（After）N行も表示
grep -A 3 "error" logfile.txt       # マッチ行 + 後3行
grep -A 5 "Exception" app.log       # 例外発生行 + スタックトレース5行

# -B N: マッチ行の前（Before）N行も表示
grep -B 2 "error" logfile.txt       # 前2行 + マッチ行
grep -B 5 "FATAL" app.log           # エラー前の文脈を確認

# -C N: マッチ行の前後（Context）N行を表示
grep -C 2 "error" logfile.txt       # 前2行 + マッチ行 + 後2行
grep -C 3 "segfault" /var/log/kern.log  # セグフォルト前後3行

# コンテキスト表示の区切り
# 複数のマッチがある場合、"--" で区切られる
grep -C 1 "error" logfile.txt
# → マッチブロック間に "--" が表示される

# グループセパレータの変更
grep --group-separator="===" -C 2 "error" logfile.txt
```

### 1.4 再帰検索（-r / -R）

```bash
# -r: ディレクトリを再帰的に検索
grep -r "TODO" ./src/               # src/ 以下を再帰検索
grep -r "import os" ./              # カレントディレクトリ以下を検索
grep -rn "console.log" ./src/       # 行番号付きで再帰検索

# -R: -r と同等だが、シンボリックリンクもたどる
grep -R "pattern" /etc/             # シンボリックリンク先も検索

# --include: 特定のファイルパターンのみ検索
grep -rn --include="*.py" "import" ./src/     # .py ファイルのみ
grep -rn --include="*.{js,ts}" "fetch" ./src/ # .js と .ts のみ
grep -rn --include="*.go" "func main" ./      # .go ファイルのみ

# --exclude: 特定のファイルパターンを除外
grep -rn --exclude="*.min.js" "function" ./   # minified JS を除外
grep -rn --exclude="*.pyc" "import" ./        # .pyc を除外

# --exclude-dir: 特定のディレクトリを除外
grep -rn --exclude-dir=node_modules "require" ./       # node_modules 除外
grep -rn --exclude-dir=.git "TODO" ./                  # .git 除外
grep -rn --exclude-dir={node_modules,.git,dist} "pattern" ./  # 複数除外

# 実務でよく使う再帰検索パターン
grep -rn --include="*.py" --exclude-dir={__pycache__,.git,venv} "TODO\|FIXME\|HACK" ./
grep -rn --include="*.{js,ts,jsx,tsx}" --exclude-dir={node_modules,.next,dist} "console.log" ./
```

### 1.5 複数パターンの検索

```bash
# -e: 複数パターンの OR 検索
grep -e "error" -e "warning" logfile.txt        # error OR warning
grep -e "fatal" -e "critical" -e "error" app.log  # 3つのパターン

# -f: ファイルからパターンを読み込み
cat > patterns.txt << 'EOF'
error
warning
fatal
EOF
grep -f patterns.txt logfile.txt    # ファイル内のパターンで検索

# パイプで AND 検索
grep "error" logfile.txt | grep "database"    # error AND database
grep "error" logfile.txt | grep -v "timeout"  # error AND NOT timeout

# 正規表現で OR 検索（-E = 拡張正規表現）
grep -E "error|warning|fatal" logfile.txt     # OR 検索
grep -E "(error|warning)" logfile.txt         # グループ化

# AND 検索の別の方法（awk を使用）
awk '/error/ && /database/' logfile.txt       # error AND database を含む行
```

---

## 2. 正規表現の活用

### 2.1 基本正規表現（BRE）と拡張正規表現（ERE）

```bash
# grep はデフォルトで基本正規表現（BRE）を使用
# -E オプション（または egrep）で拡張正規表現（ERE）を使用
# -P オプションで Perl 互換正規表現（PCRE）を使用（GNU grep のみ）

# BRE と ERE の違い:
# BRE: +, ?, |, (), {} はリテラル。エスケープ \+, \?, \|, \(\), \{\} でメタ文字
# ERE: +, ?, |, (), {} がメタ文字。エスケープ不要

# BRE の例
grep "error\|warning" logfile.txt              # OR（BRE）
grep "ab\{2,4\}" file.txt                      # a の後に b が 2〜4個（BRE）
grep "\(abc\)\{2\}" file.txt                   # "abc" が2回繰り返し（BRE）

# ERE の例（-E / egrep）
grep -E "error|warning" logfile.txt            # OR（ERE、エスケープ不要）
grep -E "ab{2,4}" file.txt                     # a の後に b が 2〜4個（ERE）
grep -E "(abc){2}" file.txt                    # "abc" が2回繰り返し（ERE）
```

### 2.2 正規表現メタ文字リファレンス

```bash
# === アンカー ===
grep "^Error" logfile.txt            # 行頭が "Error"
grep "done$" logfile.txt             # 行末が "done"
grep "^$" file.txt                   # 空行
grep -E "^\s*$" file.txt             # 空白のみの行（空行含む）

# === 文字クラス ===
grep "[abc]" file.txt                # a, b, c のいずれか
grep "[a-z]" file.txt                # 小文字アルファベット
grep "[A-Z]" file.txt                # 大文字アルファベット
grep "[0-9]" file.txt                # 数字
grep "[^0-9]" file.txt               # 数字以外
grep "[[:alpha:]]" file.txt          # アルファベット（POSIX クラス）
grep "[[:digit:]]" file.txt          # 数字（POSIX クラス）
grep "[[:space:]]" file.txt          # 空白文字（POSIX クラス）
grep "[[:upper:]]" file.txt          # 大文字（POSIX クラス）
grep "[[:lower:]]" file.txt          # 小文字（POSIX クラス）
grep "[[:alnum:]]" file.txt          # 英数字（POSIX クラス）
grep "[[:punct:]]" file.txt          # 句読点（POSIX クラス）

# === 量指定子 ===
grep -E "ab?" file.txt               # a の後に b が 0 または 1個
grep -E "ab+" file.txt               # a の後に b が 1個以上
grep -E "ab*" file.txt               # a の後に b が 0個以上
grep -E "ab{3}" file.txt             # a の後に b がちょうど3個
grep -E "ab{2,5}" file.txt           # a の後に b が 2〜5個
grep -E "ab{3,}" file.txt            # a の後に b が 3個以上

# === ワイルドカード ===
grep "a.b" file.txt                  # a の後に任意の1文字、その後 b
grep "a.*b" file.txt                 # a と b の間に任意の文字列

# === グループ化と後方参照 ===
grep -E "(abc){2}" file.txt          # "abcabc" にマッチ
grep -E "(error|warn)" logfile.txt   # "error" または "warn"
grep "\(.*\)\1" file.txt             # 同じ文字列の繰り返し（後方参照、BRE）

# === 単語境界 ===
grep "\berror\b" logfile.txt         # 単語 "error"（\b = 単語境界）
grep "\<error\>" logfile.txt         # 同上（\< = 単語先頭、\> = 単語末尾）
grep -w "error" logfile.txt          # -w オプション（同等）
```

### 2.3 Perl 互換正規表現（PCRE）

```bash
# -P オプション（GNU grep のみ。macOS の grep では非対応）
# macOS では brew install grep で ggrep を使うか、ripgrep を使う

# \d: 数字（[0-9] と同等）
grep -P "\d{4}-\d{2}-\d{2}" logfile.txt        # 日付パターン（YYYY-MM-DD）

# \w: 英数字とアンダースコア（[a-zA-Z0-9_] と同等）
grep -P "\w+@\w+\.\w+" file.txt                # メールアドレスの簡易マッチ

# \s: 空白文字
grep -P "status:\s+\d{3}" access.log           # ステータスコード

# 先読み・後読み（lookahead / lookbehind）
grep -P "(?<=price: )\d+" file.txt             # "price: " の後の数字（後読み）
grep -P "\d+(?= yen)" file.txt                 # " yen" の前の数字（先読み）
grep -P "(?<!no )error" logfile.txt            # "no " が前にない "error"（否定後読み）
grep -P "error(?! ignored)" logfile.txt        # " ignored" が後にない "error"（否定先読み）

# 名前付きキャプチャ
grep -oP "(?P<ip>\d+\.\d+\.\d+\.\d+)" access.log  # IPアドレスを抽出

# 非貪欲マッチ
grep -oP '".*?"' file.txt                       # 最短一致の引用符内テキスト
grep -oP '<.*?>' file.html                      # HTMLタグ（最短一致）
```

### 2.4 よく使う正規表現パターン集

```bash
# --- 日付・時刻 ---
grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}" logfile.txt         # YYYY-MM-DD 形式
grep -E "[0-9]{2}:[0-9]{2}:[0-9]{2}" logfile.txt           # HH:MM:SS 形式
grep -P "\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" logfile.txt # ISO 8601 形式

# --- IPアドレス ---
grep -E "\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b" access.log
grep -oP "\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b" access.log  # IP抽出

# --- メールアドレス ---
grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" file.txt

# --- URL ---
grep -E "https?://[a-zA-Z0-9./?&=%-]+" file.txt
grep -oP "https?://[^\s\"'>]+" file.html                     # URL 抽出

# --- HTTP ステータスコード ---
grep -E "HTTP/[0-9.]+ [0-9]{3}" access.log
grep -E "\s(4[0-9]{2}|5[0-9]{2})\s" access.log              # 4xx/5xx エラー

# --- JSON のキー検索 ---
grep -oP '"name"\s*:\s*"[^"]*"' data.json
grep -oP '"version"\s*:\s*"[^"]*"' package.json

# --- プログラミング ---
grep -E "^(def|class) " *.py                                 # Python の関数/クラス定義
grep -E "^(function|const|let|var) " *.js                    # JS の関数/変数定義
grep -E "^func " *.go                                        # Go の関数定義
grep -E "^(pub )?fn " *.rs                                   # Rust の関数定義
```

---

## 3. grep の高度な使い方

### 3.1 grep + パイプの実務パターン

```bash
# プロセス検索（自分自身を除外するテクニック）
ps aux | grep "[n]ginx"             # grep 自身を除外（[n] トリック）
ps aux | grep nginx | grep -v grep  # grep -v で除外（従来の方法）
pgrep -la nginx                     # pgrep を使う（推奨）

# コマンド履歴から検索
history | grep "git push"           # Git push の履歴
history | grep "docker" | tail -20  # Docker 関連の直近20件

# パッケージ検索
dpkg -l | grep "python"             # インストール済み Python パッケージ
pip list | grep "django"            # Django 関連パッケージ
npm list | grep "react"             # React 関連パッケージ

# ネットワーク情報の検索
netstat -tlnp | grep ":80"          # ポート80を使用しているプロセス
ss -tlnp | grep ":443"              # ポート443を使用しているプロセス
ip addr | grep "inet "              # IPアドレスの確認

# Docker 関連
docker ps | grep -i running         # 稼働中のコンテナ
docker images | grep "<none>"       # 名前のないイメージ（dangling）
docker logs container 2>&1 | grep "ERROR"  # コンテナログからエラー検索

# Git 関連
git log --oneline | grep "fix"      # fix を含むコミット
git diff | grep "^+"                # 追加された行のみ
git branch -a | grep "feature"      # feature ブランチの一覧
```

### 3.2 grep の出力のカスタマイズ

```bash
# --color: マッチ部分をカラー表示
grep --color=always "error" logfile.txt   # 常にカラー
grep --color=auto "error" logfile.txt     # 端末出力時のみカラー（デフォルト）
grep --color=never "error" logfile.txt    # カラー無効

# カラーをパイプで保持する
grep --color=always "error" logfile.txt | less -R

# -H / -h: ファイル名の表示制御
grep -H "pattern" file.txt          # ファイル名を常に表示
grep -h "pattern" *.txt             # ファイル名を非表示

# -Z: ファイル名の区切りをNULL文字に（xargs -0 と組み合わせ）
grep -rlZ "pattern" . | xargs -0 sed -i 's/pattern/replacement/g'

# --label: 標準入力に対するラベルを指定
cat file.txt | grep --label="STDIN" -H "pattern"

# -m N: 最初のN件のマッチで停止
grep -m 5 "error" large_logfile.txt      # 最初の5件のみ
grep -m 1 "pattern" file.txt             # 最初の1件のみ（存在確認）

# カウントとファイル名の組み合わせ
grep -rc "TODO" ./src/ | grep -v ":0$" | sort -t: -k2 -rn
# → TODO を含むファイルを、TODO の数の降順で表示
```

### 3.3 grep の特殊なオプション

```bash
# -F: 固定文字列として検索（正規表現を無効化、高速）
grep -F "error.log" file.txt         # ドットをリテラルとして検索
grep -F "[ERROR]" logfile.txt        # 角括弧をリテラルとして検索
grep -F "$HOME" script.sh            # $HOME をリテラルとして検索

# fgrep は grep -F と同等
fgrep "pattern" file.txt             # 固定文字列検索

# -z: NULL文字を行区切りとして扱う（マルチライン検索の一種）
grep -Pzo "function.*?\n.*?return" *.js   # 関数定義から return まで

# --binary-files: バイナリファイルの扱い
grep --binary-files=text "pattern" binary_file   # バイナリをテキストとして検索
grep -a "pattern" binary_file                     # -a と同等

# -T: タブ揃えで出力
grep -Tn "pattern" file.txt          # タブで位置を揃えて表示
```

### 3.4 grep + xargs による一括操作

```bash
# 検索結果のファイルに対して一括操作
grep -rl "old_function" ./src/ | xargs sed -i 's/old_function/new_function/g'
# → old_function を含むファイルを見つけて、一括置換

# 検索結果のファイルをエディタで開く
grep -rl "TODO" ./src/ | xargs code
# → TODO を含むファイルを VS Code で開く

# 検索結果を一覧表示
grep -rl "deprecated" ./src/ | xargs ls -la
# → deprecated を含むファイルの詳細情報

# NULL区切りを使った安全な一括操作（スペース入りファイル名対応）
grep -rlZ "pattern" . | xargs -0 wc -l
grep -rlZ "old" . | xargs -0 sed -i 's/old/new/g'
```

---

## 4. ripgrep（rg）— モダンな高速検索ツール

### 4.1 インストールと概要

```bash
# インストール
brew install ripgrep               # macOS
sudo apt install ripgrep           # Ubuntu/Debian
sudo pacman -S ripgrep             # Arch Linux
cargo install ripgrep              # Rust (Cargo)

# ripgrep の特徴
# - デフォルトで再帰検索
# - .gitignore を自動で尊重
# - Unicode 対応
# - カラー出力がデフォルト
# - 並列処理による高速検索
# - 行番号の自動表示
# - 豊富なファイルタイプフィルタ
```

### 4.2 基本的な使い方

```bash
# 基本構文: rg [オプション] パターン [パス]
rg "pattern"                       # カレントディレクトリを再帰検索
rg "pattern" src/                  # src/ ディレクトリを検索
rg "pattern" file.txt              # 特定ファイルを検索

# 大小文字の制御
rg -i "error"                      # 大小文字無視（ignore case）
rg -s "Error"                      # 大小文字を区別（smart case 無効化）
# デフォルト: スマートケース（パターンが全て小文字なら case-insensitive）

# 行番号と列番号
rg -n "pattern"                    # 行番号表示（デフォルトで有効）
rg --column "pattern"              # 列番号も表示

# コンテキスト表示
rg -A 3 "error"                    # マッチ行の後3行
rg -B 2 "error"                    # マッチ行の前2行
rg -C 2 "error"                    # マッチ行の前後2行
```

### 4.3 ファイルタイプフィルタ

```bash
# -t: ファイルタイプで絞り込み
rg "pattern" -t py                 # Python ファイルのみ
rg "pattern" -t js                 # JavaScript ファイルのみ
rg "pattern" -t go                 # Go ファイルのみ
rg "pattern" -t rust               # Rust ファイルのみ
rg "pattern" -t java               # Java ファイルのみ
rg "pattern" -t cpp                # C++ ファイルのみ
rg "pattern" -t html               # HTML ファイルのみ
rg "pattern" -t css                # CSS ファイルのみ
rg "pattern" -t yaml               # YAML ファイルのみ
rg "pattern" -t json               # JSON ファイルのみ
rg "pattern" -t md                 # Markdown ファイルのみ
rg "pattern" -t sh                 # シェルスクリプトのみ
rg "pattern" -t sql                # SQL ファイルのみ
rg "pattern" -t docker             # Dockerfile のみ
rg "pattern" -t make               # Makefile のみ

# -T: ファイルタイプを除外
rg "pattern" -T js                 # JavaScript 以外
rg "pattern" -T test               # テストファイル以外

# -g: グロブパターンでフィルタ
rg "pattern" -g "*.{js,ts}"        # .js と .ts ファイル
rg "pattern" -g "!*.min.js"        # .min.js を除外
rg "pattern" -g "src/**"           # src/ ディレクトリ以下のみ
rg "pattern" -g "!test/**"         # test/ を除外

# 定義済みタイプの一覧
rg --type-list                     # 全ファイルタイプとその拡張子を表示
rg --type-list | grep "python"     # Python タイプの定義を確認

# カスタムタイプの定義
rg --type-add "web:*.{html,css,js}" -t web "pattern"   # カスタムタイプで検索
```

### 4.4 出力制御

```bash
# -l: マッチしたファイル名のみ表示
rg "pattern" -l                    # マッチするファイル名のみ

# --files-without-match: マッチしないファイル名を表示
rg "pattern" --files-without-match

# -c: ファイルごとのマッチ数をカウント
rg "TODO|FIXME" -c                 # ファイルごとの TODO/FIXME 数

# --count-matches: マッチの総数（行単位ではなく出現数）
rg "pattern" --count-matches

# -o: マッチ部分のみ表示
rg -o "\d+\.\d+\.\d+" file.txt    # バージョン番号を抽出

# --json: JSON 形式で出力（ツール連携用）
rg "pattern" --json                # 検索結果をJSON形式で出力

# --vimgrep: Vim の quickfix 形式で出力
rg "pattern" --vimgrep             # ファイル:行:列:マッチ行

# --no-heading: ファイル名をグループヘッダーではなく各行に表示
rg --no-heading "pattern"

# --heading: ファイル名をグループヘッダーとして表示（デフォルト）
rg --heading "pattern"

# -U: マルチラインマッチ
rg -U "function.*\n.*return" -t js  # 複数行にまたがるパターン

# -M: マッチ行の最大文字数を制限
rg -M 200 "pattern"                # マッチ行が200文字を超えたら省略

# --trim: マッチ行の先頭の空白を除去
rg --trim "pattern"
```

### 4.5 隠しファイルと .gitignore

```bash
# デフォルトの動作
# - .gitignore を尊重
# - 隠しファイル（.xxx）を除外
# - バイナリファイルを除外

# --hidden: 隠しファイルを含める
rg "pattern" --hidden              # .env, .config 等を含めて検索
rg "API_KEY" --hidden              # .env ファイル内の検索

# --no-ignore: .gitignore を無視
rg "pattern" --no-ignore           # .gitignore 対象ファイルも検索
rg "pattern" -u                    # --no-ignore の短縮形

# -uu: 隠しファイル + .gitignore 無視
rg "pattern" -uu                   # --hidden --no-ignore と同等

# -uuu: さらにバイナリファイルも検索
rg "pattern" -uuu                  # 全ファイルを対象（find + grep と同等）

# --no-ignore-vcs: VCS の ignore ファイルのみ無視（.gitignore 等）
rg "pattern" --no-ignore-vcs

# 特定の ignore ファイルを追加
rg "pattern" --ignore-file .customignore
```

### 4.6 rg の置換機能

```bash
# -r / --replace: マッチ部分を置換して表示（ファイルは変更しない）
rg "old" -r "new"                  # "old" を "new" に置換して表示
rg "foo(\d+)" -r "bar$1"          # キャプチャグループを使った置換

# 実際のファイル置換は sed と組み合わせる
rg -l "old_name" -t py | xargs sed -i 's/old_name/new_name/g'

# 置換のプレビュー
rg "old_function" -r "new_function" --passthru
# → マッチしない行も表示し、マッチ行は置換後の状態で表示
```

### 4.7 rg の実務パターン集

```bash
# --- コード検索 ---

# TODO/FIXME/HACK の一括検索
rg "TODO|FIXME|HACK|XXX" -t py -t js -t go

# 未使用の import を検索（Python）
rg "^import|^from .* import" -t py --no-heading

# console.log の残りを検索（JavaScript）
rg "console\.(log|debug|info|warn|error)" -t js -t ts

# デバッグ文の検索（Python）
rg "(print\(|pdb\.set_trace|breakpoint\(\))" -t py

# 関数定義の検索
rg "^(def|class) " -t py                    # Python
rg "^(export )?(function|const|class) " -t js  # JavaScript
rg "^func " -t go                            # Go
rg "^(pub )?fn " -t rust                     # Rust

# --- ログ分析 ---

# HTTPステータスコード別のカウント
rg -o "HTTP/\d\.\d\" (\d{3})" -r '$1' access.log | sort | uniq -c | sort -rn

# 特定時間帯のログ
rg "^2026-02-16 1[4-5]:" app.log

# エラーログの IP アドレス抽出
rg "ERROR" access.log | rg -o "\d+\.\d+\.\d+\.\d+" | sort -u

# --- 設定ファイル ---

# 環境変数の使用箇所を検索
rg "process\.env\." -t js -t ts
rg "os\.environ" -t py
rg "os\.Getenv" -t go

# ハードコードされた認証情報の検出
rg -i "(password|secret|api.?key|token)\s*[:=]\s*[\"']" --hidden

# --- プロジェクト管理 ---

# ファイルの行数統計（rg + wc）
rg --files -t py | xargs wc -l | sort -n

# 使用ライブラリの検索
rg "^import " -t py --no-filename | sort -u    # Python のインポート一覧
rg "require\(" -t js --no-filename -o | sort -u  # Node.js の require 一覧

# 特定の関数の使用箇所
rg "deprecated_function\(" -t py --stats      # --stats でマッチ統計も表示
```

### 4.8 rg の設定ファイル

```bash
# ~/.ripgreprc に設定を記述
# 環境変数 RIPGREP_CONFIG_PATH で設定ファイルのパスを指定

# ~/.ripgreprc の例
cat > ~/.ripgreprc << 'EOF'
# デフォルトでスマートケース
--smart-case

# 最大列数
--max-columns=200

# 隠しファイルを含める
# --hidden

# 特定ディレクトリを常に除外
--glob=!.git/
--glob=!node_modules/
--glob=!target/
--glob=!__pycache__/
--glob=!*.pyc

# カスタムタイプの定義
--type-add=web:*.{html,css,js,ts}
EOF

# 設定ファイルのパスを環境変数で指定（~/.bashrc に追加）
export RIPGREP_CONFIG_PATH="$HOME/.ripgreprc"
```

---

## 5. 他の検索ツール

### 5.1 ag（The Silver Searcher）

```bash
# インストール
brew install the_silver_searcher    # macOS
sudo apt install silversearcher-ag  # Ubuntu/Debian

# 基本的な使い方
ag "pattern"                        # 再帰検索（.gitignore 尊重）
ag -i "pattern"                     # 大小文字無視
ag "pattern" -G "\.py$"             # Python ファイルのみ
ag -l "pattern"                     # マッチファイル名のみ
ag --stats "pattern"                # 統計情報付き

# rg と ag の比較:
# ag: rg 以前のモダン grep。まだ使われているが、rg の方が高速
# rg: ag の後継として開発。ほぼ全ての面で ag を上回る
```

### 5.2 ack

```bash
# インストール
brew install ack                    # macOS
sudo apt install ack               # Ubuntu/Debian

# 基本的な使い方
ack "pattern"                       # 再帰検索
ack "pattern" --python              # Python ファイルのみ
ack "pattern" --type-set=web:ext:html,css,js  # カスタムタイプ
ack -f --python                     # Python ファイルの一覧

# ack は grep の先駆的な代替ツール
# 現在は rg が最も推奨される
```

### 5.3 grep / rg / ag / ack の比較

```
┌──────────┬────────────┬────────────┬────────────┬────────────┐
│ 機能     │ grep       │ rg         │ ag         │ ack        │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│ 速度     │ 標準       │ 最速       │ 高速       │ 中速       │
│ .gitignore│ 非対応    │ 対応       │ 対応       │ 対応       │
│ カラー   │ --color    │ デフォルト │ デフォルト │ デフォルト │
│ Unicode  │ 制限あり   │ 完全対応   │ 部分対応   │ 部分対応   │
│ PCRE     │ -P (GNU)   │ デフォルト │ 部分対応   │ Perl正規表現│
│ マルチライン│ 制限あり│ -U        │ 非対応     │ 非対応     │
│ 環境     │ 標準搭載   │ 別途導入   │ 別途導入   │ 別途導入   │
│ 置換     │ 非対応     │ -r (表示)  │ 非対応     │ 非対応     │
│ 並列処理 │ 非対応     │ 対応       │ 対応       │ 非対応     │
└──────────┴────────────┴────────────┴────────────┴────────────┘

推奨: rg を最優先で使用。スクリプトでは POSIX 互換の grep を使用。
```

---

## 6. 実務パターン集（応用編）

### 6.1 ログ分析ワンライナー

```bash
# アクセスログの IP アドレス別アクセス数トップ20
grep -oP "^\d+\.\d+\.\d+\.\d+" access.log | sort | uniq -c | sort -rn | head -20

# HTTP ステータスコード分布
awk '{print $9}' access.log | sort | uniq -c | sort -rn

# 404 エラーの URL 一覧
grep " 404 " access.log | awk '{print $7}' | sort | uniq -c | sort -rn | head -20

# 特定時間帯のリクエスト数
grep "16/Feb/2026:14:" access.log | wc -l

# レスポンスタイムの分析（最終フィールドがレスポンスタイムの場合）
awk '{print $NF}' access.log | sort -n | tail -20    # 遅いリクエスト

# エラーレートの計算
total=$(wc -l < access.log)
errors=$(grep -c " [45][0-9][0-9] " access.log)
echo "Error rate: $(echo "scale=2; $errors * 100 / $total" | bc)%"

# リアルタイムエラー監視（カラー付き）
tail -f access.log | grep --color=always --line-buffered " [45][0-9][0-9] "
```

### 6.2 セキュリティ監査パターン

```bash
# ハードコードされた認証情報の検出
rg -i "(password|passwd|pwd)\s*[:=]" --hidden -g "!*.{lock,sum}"
rg -i "api[_-]?key\s*[:=]\s*['\"]" --hidden
rg -i "secret\s*[:=]" --hidden -g "!*.{lock,sum}"
rg "-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----" --hidden

# SQL インジェクションの脆弱性検出
rg "execute\(.*\+.*\)" -t py         # 文字列連結による SQL
rg "query\(.*\$" -t js               # テンプレートリテラルによる SQL
rg "f\".*SELECT.*{" -t py            # f-string による SQL

# XSS の脆弱性検出
rg "innerHTML\s*=" -t js -t ts
rg "dangerouslySetInnerHTML" -t js -t ts
rg "v-html=" -t html

# 安全でない関数の使用検出
rg "eval\(" -t py -t js              # eval の使用
rg "exec\(" -t py                    # exec の使用
rg "pickle\.loads?" -t py            # pickle の使用
```

### 6.3 コードレビュー支援パターン

```bash
# デバッグ用コードの残り検出
rg "console\.(log|debug)" -t js -t ts  # JS/TS のデバッグ出力
rg "print\(" -t py -g "!test_*"        # Python の print（テスト除外）
rg "debugger" -t js -t ts              # JS のデバッグブレークポイント
rg "binding\.pry" -t ruby              # Ruby のデバッグ
rg "fmt\.Print" -t go -g "!*_test.go"  # Go の fmt.Print（テスト除外）

# TODO/FIXME のサマリー
rg "TODO|FIXME|HACK|XXX|DEPRECATED" --stats -c | sort -t: -k2 -rn

# 長い関数の検出（行数ベース）
rg -U "^def \w+.*:\n(.*\n){50,}" -t py --no-heading -l
# → 50行以上の Python 関数を含むファイル

# マジックナンバーの検出
rg "(?<![\"'])\b\d{3,}\b(?![\"'])" -t py -g "!test_*" -g "!*constants*"
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と対処法

```bash
# 問題: 特殊文字がパターンとして解釈される
# 対処: -F（固定文字列）を使うか、エスケープする
grep -F "[ERROR]" logfile.txt        # -F で固定文字列として検索
grep "\[ERROR\]" logfile.txt         # エスケープする
rg -F "[ERROR]" logfile.txt          # rg でも -F が使える

# 問題: バイナリファイルがマッチする
# 対処: -I オプションでバイナリを無視
grep -rI "pattern" ./               # バイナリファイルを無視
rg "pattern"                        # rg はデフォルトでバイナリを無視

# 問題: 大量のマッチで出力が溢れる
# 対処: -m, -l, less を使う
grep -m 10 "pattern" logfile.txt    # 最初の10件で停止
grep -l "pattern" *.log             # ファイル名のみ
grep "pattern" logfile.txt | less   # ページャで閲覧

# 問題: macOS の grep で -P（PCRE）が使えない
# 対処: GNU grep をインストールするか、rg を使う
brew install grep                    # ggrep としてインストールされる
ggrep -P "\d+" file.txt             # GNU grep の PCRE
rg "\d+" file.txt                   # rg を使う（推奨）

# 問題: UTF-8 のファイルで日本語が検索できない
# 対処: ロケールを確認/設定
export LANG=en_US.UTF-8
grep "日本語" file.txt              # UTF-8 が正しく設定されていれば動作
rg "日本語" file.txt                # rg はUnicode対応

# 問題: grep -r が遅い
# 対処: rg を使うか、--include/--exclude-dir を指定
rg "pattern"                         # rg は圧倒的に高速
grep -rn --include="*.py" --exclude-dir=venv "pattern" ./  # grep で絞り込み
```

### 7.2 パフォーマンスのヒント

```bash
# 1. -F（固定文字列検索）は正規表現より高速
grep -F "exact string" large_file.txt

# 2. -m で検索を早期終了
grep -m 1 "pattern" large_file.txt   # 最初の1件で停止

# 3. 不要なオプションを避ける
# -i（大小文字無視）は若干遅くなる
# -E（拡張正規表現）は単純パターンでは不要

# 4. 検索範囲を限定する
grep -rn --include="*.py" "pattern" ./  # ファイルタイプを限定
grep "pattern" specific_file.txt         # 特定ファイルのみ

# 5. rg は grep の 2〜10 倍高速
# 大規模コードベースでは rg を第一選択にする

# 6. LC_ALL=C で高速化（ASCII のみの場合）
LC_ALL=C grep "pattern" large_file.txt   # ロケール処理をスキップ
```

---

## まとめ

| オプション / ツール | 意味 |
|-------------------|------|
| grep -i | 大小文字無視 |
| grep -r / -R | 再帰検索 |
| grep -n | 行番号表示 |
| grep -l / -L | マッチ/非マッチファイル名 |
| grep -v | 逆マッチ |
| grep -w | 単語完全一致 |
| grep -E | 拡張正規表現 |
| grep -F | 固定文字列（正規表現無効） |
| grep -P | Perl 互換正規表現 |
| grep -o | マッチ部分のみ表示 |
| grep -c | マッチ行数カウント |
| grep -A/-B/-C N | 前後のコンテキスト |
| grep -q | 終了コードのみ（出力なし） |
| grep --include | 対象ファイルの絞り込み |
| grep --exclude-dir | ディレクトリの除外 |
| rg (ripgrep) | 高速再帰検索（.gitignore 尊重） |
| rg -t TYPE | ファイルタイプフィルタ |
| rg -g GLOB | グロブパターンフィルタ |
| rg --hidden | 隠しファイルを含める |
| rg -U | マルチラインマッチ |

---

## 12. grep / rg のパフォーマンスチューニング

### 12.1 検索速度の最適化

```bash
# grep の高速化テクニック

# 1. 固定文字列検索（-F）は正規表現より高速
grep -F "exact string" file.txt              # 正規表現解析をスキップ
grep -F "error" large_log.txt                # 単純な文字列検索に最適
fgrep "pattern" file.txt                     # -F のエイリアス（非推奨だが利用可）

# 2. ロケール設定で高速化
LC_ALL=C grep "pattern" file.txt             # Cロケール（ASCII前提）で大幅に高速化
# → UTF-8ロケールの場合、文字クラスの処理が重い
# → 英数字のみの検索なら LC_ALL=C が効果的

# 3. 不要なオプションを避ける
grep -c "pattern" file.txt                   # カウントだけなら -c（行全体を出力しない）
grep -q "pattern" file.txt                   # 存在確認だけなら -q（最初のマッチで終了）
grep -l "pattern" *.log                      # ファイル名だけなら -l（全行スキャン不要）
grep -m 1 "pattern" file.txt                 # 最初のマッチだけなら -m 1（早期終了）

# 4. バッファリングの制御
grep --line-buffered "pattern" file.txt      # 行バッファリング（リアルタイム出力）
# → パイプラインで使用する場合に有用

# 5. 並列検索
grep -r "pattern" /path --include="*.py" &   # バックグラウンドで実行
# GNU parallel を使った並列検索
find . -name "*.log" | parallel -j4 grep -l "ERROR" {}

# ripgrep の高速化テクニック
# rg はデフォルトで最適化されているが、さらに調整可能

# スレッド数の制御
rg -j 4 "pattern" /large/directory           # 4スレッドで検索
rg -j 1 "pattern" file.txt                   # シングルスレッド（小ファイル向け）

# メモリマップの制御
rg --mmap "pattern" huge_file.txt            # メモリマップを強制使用
rg --no-mmap "pattern" /nfs/share/           # メモリマップを無効化（NFS等）

# バイナリファイルのスキップ
rg --no-binary "pattern" /mixed/content/     # バイナリファイルを完全スキップ

# エンコーディング指定
rg -E shift-jis "パターン" legacy_file.txt   # 文字コード指定
rg -E euc-jp "検索語" old_data.txt
```

### 12.2 大規模コードベースでの検索戦略

```bash
# プロジェクト全体の検索を効率化

# 1. .gitignore を活用（rg はデフォルトで尊重）
rg "TODO" .                                   # node_modules, .git 等は自動除外

# 2. 検索対象を絞り込む
rg -t py "import" .                           # Python ファイルのみ
rg -g "!tests/" "function" .                  # テストディレクトリを除外
rg -g "*.{ts,tsx}" "useState" .               # TypeScript ファイルのみ

# 3. 事前フィルタリング
# よく検索するパターンをエイリアスに登録
alias rgpy='rg -t py'
alias rgjs='rg -t js -t ts'
alias rgerr='rg -i "error|exception|fail"'
alias rgtodo='rg "TODO|FIXME|HACK|XXX"'

# 4. 検索結果のキャッシュ（頻繁に実行する場合）
rg -l "pattern" > /tmp/matched_files.txt
cat /tmp/matched_files.txt | xargs rg "another_pattern"

# 5. インクリメンタル検索
# fzf と組み合わせた対話的検索
rg --line-number --no-heading "." | fzf --preview 'echo {} | cut -d: -f1-2 | xargs -I{} sh -c "head -n \$(echo {} | cut -d: -f2) \$(echo {} | cut -d: -f1) | tail -5"'

# grep / rg と IDE の検索の使い分け
# - 単純な文字列検索: rg（最速）
# - 構文を考慮した検索: IDE（AST ベース）
# - 正規表現の複雑な検索: rg -P（PCRE2）
# - 対話的な絞り込み: rg | fzf
```

---

## 13. 実務シナリオ別の総合レシピ

### 13.1 インシデント対応での grep/rg 活用

```bash
# 障害発生時の迅速なログ調査

# 1. エラーの初回発生時刻を特定
rg -m 1 "OutOfMemoryError" /var/log/app/*.log
grep -rn -m 1 "Connection refused" /var/log/

# 2. エラー前後のコンテキストを確認
rg -C 10 "FATAL" /var/log/app/app.log | head -50

# 3. エラーの頻度を時系列で確認
rg "ERROR" app.log | rg -o "^\d{4}-\d{2}-\d{2} \d{2}:" | sort | uniq -c

# 4. 特定時間帯のログを抽出
rg "^2024-01-15 1[4-6]:" app.log                # 14:00-16:59 のログ
grep "^2024-01-15 15:3[0-9]:" app.log            # 15:30-15:39 のログ

# 5. 複数サーバーのログを同時検索
for host in web01 web02 web03; do
  echo "=== $host ==="
  ssh "$host" "rg 'ERROR' /var/log/app/app.log | tail -5"
done

# 6. スタックトレースの抽出
rg -U "Exception.*\n(\s+at .*\n)+" app.log       # マルチラインマッチ
grep -A 20 "Exception" app.log | grep -B 1 "^$"  # 空行までのスタックトレース

# 7. 影響を受けたユーザーの特定
rg "ERROR.*user_id=(\w+)" -o -r '$1' app.log | sort -u

# 8. エラー発生パターンの分析
rg -o "ERROR \[([^\]]+)\]" -r '$1' app.log | sort | uniq -c | sort -rn | head -20
```

### 13.2 コードレビューでの grep/rg 活用

```bash
# コードの品質チェック

# 1. ハードコードされた値の検出
rg -n "localhost|127\.0\.0\.1|192\.168\." --type-not test src/
rg -n "password\s*=\s*[\"'][^\"']+[\"']" src/

# 2. デバッグコードの残存チェック
rg -n "console\.(log|debug|warn)|print\(|debugger|binding\.pry" src/
rg -n "TODO|FIXME|HACK|XXX|TEMP" src/ --stats

# 3. 未使用のインポート検出（簡易版）
for import in $(rg -o "^import \{ (\w+)" -r '$1' src/index.ts); do
  count=$(rg -c "\b$import\b" src/index.ts)
  if [ "$count" -le 1 ]; then
    echo "Possibly unused: $import"
  fi
done

# 4. API エンドポイントの一覧抽出
rg -n "(@(Get|Post|Put|Delete|Patch)Mapping|@RequestMapping)" --type java src/
rg -n "router\.(get|post|put|delete|patch)\(" --type js src/
rg -n "@app\.(route|get|post|put|delete)" --type py src/

# 5. SQL インジェクションの可能性チェック
rg -n "execute\(.*\+.*\)|execute\(.*%s.*%|execute\(.*\{.*\}.*\.format" --type py src/
rg -n "query\(.*\+.*\)|query\(.*\$\{" --type js src/

# 6. 例外処理の確認
rg -n "except:|catch\s*\(" --type py --type java --type js src/ | rg -v "except Exception|catch \(Error"

# 7. マジックナンバーの検出
rg -n "(?<!=\s*)\b(0|1)\b(?!\s*[;,\)])" --type java src/ | rg -v "//|/\*|\*/"
```

### 13.3 セキュリティ監査での grep/rg 活用

```bash
# 包括的なセキュリティスキャン

# 1. 機密情報の漏洩チェック
rg -in "(api[_-]?key|secret[_-]?key|access[_-]?token|private[_-]?key)\s*[=:]\s*[\"']?\w+" .
rg -in "BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY" .
rg -in "(password|passwd|pwd)\s*[=:]\s*[\"'][^\"']{4,}" .

# 2. AWS 認証情報の検出
rg "AKIA[0-9A-Z]{16}" .                         # AWS Access Key ID
rg "[0-9a-zA-Z/+]{40}" . | rg -v "test|example"  # AWS Secret Key（要確認）

# 3. 脆弱な暗号化の使用
rg -in "md5|sha1(?!sum)|des[^ign]|rc4" --type py --type java --type js .
rg -in "eval\s*\(|exec\s*\(|system\s*\(" --type py --type rb .

# 4. CORS 設定の確認
rg -in "Access-Control-Allow-Origin.*\*" .
rg -in "cors.*origin.*\*|allowAllOrigins" .

# 5. ファイルパーミッションの確認（設定ファイル内）
rg -in "chmod\s+777|chmod\s+666|umask\s+000" .

# 6. 入力バリデーションの欠如
rg -n "request\.(params|query|body)\.\w+" --type js . | rg -v "validate|sanitize|escape"
```

---

## 次に読むべきガイド
→ [[02-sed.md]] — ストリームエディタ

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.5, O'Reilly, 2022.
2. Friedl, J. "Mastering Regular Expressions." 3rd Ed, O'Reilly, 2006.
3. GNU Grep Manual. https://www.gnu.org/software/grep/manual/
4. ripgrep GitHub Repository. https://github.com/BurntSushi/ripgrep
5. ripgrep User Guide. https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md
