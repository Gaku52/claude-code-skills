# ネットワークツール（curl, wget）

> curl と wget は CLI からの HTTP 通信を可能にする、API開発・デバッグの必須ツール。
> Web API のテスト、ファイルダウンロード、ヘルスチェック、CI/CD 連携まで幅広く活用される。

## この章で学ぶこと

- [ ] curl で各種 HTTP リクエストを送信できる
- [ ] wget でファイルダウンロード・ミラーリングができる
- [ ] API テスト・デバッグに活用できる
- [ ] jq と組み合わせて JSON を処理できる
- [ ] curl を使った自動化スクリプトを書ける
- [ ] セキュリティを意識した通信設定ができる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. curl の基本

### 1.1 GET リクエスト

```bash
# 基本: curl [オプション] URL

# 最も基本的な GET リクエスト
curl https://example.com                     # HTMLをstdoutに出力

# ファイルに保存
curl -o output.html https://example.com      # 指定ファイル名で保存
curl -O https://example.com/file.zip         # 元のファイル名で保存
curl -O -O https://example.com/{a,b}.zip     # 複数ファイル

# サイレントモード
curl -s https://api.example.com/data         # プログレス表示なし
curl -sS https://api.example.com/data        # エラーのみ表示
curl -s --fail https://api.example.com/data  # HTTPエラー時に終了コード非0

# 出力の制御
curl -s https://api.example.com/data | python3 -m json.tool  # JSON整形
curl -s https://api.example.com/data | jq '.'                # jqで整形
```

### 1.2 レスポンスヘッダの確認

```bash
# ヘッダのみ取得（HEAD リクエスト）
curl -I https://example.com
# HTTP/2 200
# content-type: text/html; charset=UTF-8
# date: Mon, 15 Jan 2024 10:30:00 GMT
# content-length: 1234

# ヘッダ + ボディ
curl -i https://example.com

# 詳細な通信情報（リクエスト/レスポンスの全内容）
curl -v https://example.com
# > GET / HTTP/2
# > Host: example.com
# > User-Agent: curl/8.1.0
# > Accept: */*
# >
# < HTTP/2 200
# < content-type: text/html; charset=UTF-8
# ...

# さらに詳細（TLSハンドシェイクも含む）
curl -vvv https://example.com

# トレース（バイナリレベルの詳細）
curl --trace /tmp/curl_trace.log https://example.com
curl --trace-ascii /tmp/curl_trace.txt https://example.com  # ASCII形式
```

### 1.3 リダイレクト

```bash
# リダイレクトに追従（-L / --location）
curl -L https://example.com                  # 301/302 に自動で追従

# 最大リダイレクト数を制限
curl -L --max-redirs 5 https://example.com

# リダイレクト先を確認するだけ
curl -sI -o /dev/null -w "%{url_effective}\n" -L https://short.url/abc

# リダイレクト履歴を表示
curl -sIL https://example.com | grep -i "^location:"
```

### 1.4 User-Agent とヘッダ

```bash
# User-Agent を指定
curl -A "Mozilla/5.0 (compatible; MyBot/1.0)" https://example.com

# カスタムヘッダの追加（-H）
curl -H "Accept: application/json" https://api.example.com
curl -H "Accept-Language: ja" https://example.com
curl -H "X-Custom-Header: value" https://api.example.com

# 複数のカスタムヘッダ
curl -H "Accept: application/json" \
     -H "X-API-Version: 2" \
     -H "X-Request-ID: $(uuidgen)" \
     https://api.example.com

# リファラーの指定
curl -e "https://google.com" https://example.com
curl --referer "https://google.com" https://example.com
```

---

## 2. HTTP メソッド

### 2.1 POST リクエスト

```bash
# JSON データの送信
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Gaku", "email": "gaku@example.com"}'

# JSON データを送る（省略形、curl 7.82+）
curl -X POST https://api.example.com/users \
  --json '{"name": "Gaku"}'
# --json は以下と同等:
# -H "Content-Type: application/json"
# -H "Accept: application/json"
# -d 'データ'

# ファイルからデータを読み込む
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d @data.json

# 標準入力からデータを読み込む
echo '{"name": "Gaku"}' | curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d @-

# URL エンコードされたフォームデータ
curl -X POST https://example.com/login \
  -d "username=gaku&password=secret"
# Content-Type: application/x-www-form-urlencoded が自動設定

# --data-urlencode で自動エンコード
curl -X POST https://example.com/search \
  --data-urlencode "q=hello world" \
  --data-urlencode "lang=日本語"
```

### 2.2 ファイルアップロード（マルチパート）

```bash
# フォームデータ + ファイルアップロード
curl -X POST https://example.com/upload \
  -F "file=@photo.jpg" \
  -F "name=test"
# Content-Type: multipart/form-data が自動設定

# 複数ファイルのアップロード
curl -X POST https://example.com/upload \
  -F "files[]=@file1.pdf" \
  -F "files[]=@file2.pdf" \
  -F "description=Documents"

# ファイルのContent-Typeを明示指定
curl -X POST https://example.com/upload \
  -F "file=@data.csv;type=text/csv"

# ファイル名を変更してアップロード
curl -X POST https://example.com/upload \
  -F "file=@local_name.jpg;filename=uploaded.jpg"

# 大容量ファイルのアップロード（プログレス表示付き）
curl -X POST https://example.com/upload \
  -F "file=@large_file.zip" \
  --progress-bar
```

### 2.3 PUT / PATCH / DELETE

```bash
# PUT（リソースの完全置換）
curl -X PUT https://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Name", "email": "new@example.com"}'

# PATCH（リソースの部分更新）
curl -X PATCH https://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Patched Name"}'

# DELETE
curl -X DELETE https://api.example.com/users/1

# DELETE with body（一部のAPIで使用）
curl -X DELETE https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2, 3]}'

# OPTIONS（CORSプリフライトの確認）
curl -X OPTIONS https://api.example.com/users \
  -H "Origin: https://frontend.example.com" \
  -H "Access-Control-Request-Method: POST" \
  -i
```

---

## 3. 認証

### 3.1 Basic 認証

```bash
# Basic認証
curl -u username:password https://api.example.com
# Authorization: Basic base64(username:password) が自動設定

# パスワードをプロンプトで入力（コマンド履歴に残さない）
curl -u username https://api.example.com
# パスワードの入力を求められる

# .netrc ファイルを使用（認証情報をファイルに保存）
# ~/.netrc:
# machine api.example.com
# login username
# password secret
curl -n https://api.example.com              # .netrc を使用
curl --netrc-file /path/to/netrc https://api.example.com
```

### 3.2 Bearer トークン / API キー

```bash
# Bearer トークン（OAuth2 など）
curl -H "Authorization: Bearer TOKEN_HERE" https://api.example.com

# 変数を使用（推奨）
TOKEN="your_api_token_here"
curl -H "Authorization: Bearer $TOKEN" https://api.example.com

# 環境変数から読み込み
curl -H "Authorization: Bearer ${API_TOKEN}" https://api.example.com

# APIキー（ヘッダ）
curl -H "X-API-Key: KEY_HERE" https://api.example.com

# APIキー（クエリパラメータ）
curl "https://api.example.com/data?api_key=KEY_HERE"

# AWS Signature v4（AWS CLI を使うのが一般的）
curl --aws-sigv4 "aws:amz:us-east-1:s3" \
  --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  https://s3.amazonaws.com/bucket/key
```

### 3.3 Cookie

```bash
# Cookie を送信
curl -b "session=abc123; lang=ja" https://example.com

# Cookie をファイルに保存（レスポンスの Set-Cookie を保存）
curl -c cookies.txt https://example.com/login \
  -d "username=gaku&password=secret"

# 保存した Cookie を送信
curl -b cookies.txt https://example.com/dashboard

# Cookie の保存と送信を同時に（セッション維持）
curl -b cookies.txt -c cookies.txt https://example.com/api/data

# Cookie jar を使ったセッション管理（一連のリクエスト）
COOKIE_JAR=$(mktemp)
trap "rm -f $COOKIE_JAR" EXIT

# ログイン
curl -s -c "$COOKIE_JAR" https://example.com/login \
  -d "username=gaku&password=secret"

# セッションを使ってデータ取得
curl -s -b "$COOKIE_JAR" https://example.com/api/data

# ログアウト
curl -s -b "$COOKIE_JAR" -c "$COOKIE_JAR" https://example.com/logout
```

### 3.4 クライアント証明書

```bash
# クライアント証明書での認証
curl --cert client.pem --key client-key.pem https://secure.example.com

# PKCS#12 形式
curl --cert client.p12:password https://secure.example.com

# CA証明書の指定
curl --cacert ca-bundle.crt https://example.com

# 証明書の検証をスキップ（開発環境のみ）
curl -k https://self-signed.example.com
# 注意: 本番環境では絶対に使わない
```

---

## 4. 高度なオプション

### 4.1 タイムアウトとリトライ

```bash
# 接続タイムアウト（TCP接続確立まで）
curl --connect-timeout 5 https://example.com

# 全体タイムアウト（リクエスト全体）
curl --max-time 30 https://example.com

# 両方指定（推奨）
curl --connect-timeout 5 --max-time 30 https://example.com

# リトライ
curl --retry 3 https://example.com
curl --retry 3 --retry-delay 2 https://example.com          # 2秒間隔
curl --retry 3 --retry-max-time 60 https://example.com      # リトライ全体で60秒まで
curl --retry 3 --retry-all-errors https://example.com        # 全エラーでリトライ
# デフォルトではタイムアウトと一部のHTTPエラーのみリトライ

# DNS解決のタイムアウト
curl --dns-servers 8.8.8.8 https://example.com  # DNSサーバー指定
curl --resolve example.com:443:93.184.216.34 https://example.com  # DNS解決をオーバーライド
```

### 4.2 プロキシ

```bash
# HTTPプロキシ
curl -x http://proxy:8080 https://example.com
curl --proxy http://proxy:8080 https://example.com

# 認証付きプロキシ
curl -x http://user:pass@proxy:8080 https://example.com
curl --proxy-user user:pass -x http://proxy:8080 https://example.com

# SOCKSプロキシ
curl --proxy socks5://proxy:1080 https://example.com
curl --proxy socks5h://proxy:1080 https://example.com  # DNS もプロキシ経由

# プロキシなし
curl --noproxy "*" https://example.com
curl --noproxy "localhost,127.0.0.1,.internal.com" https://example.com

# 環境変数でプロキシ設定
export http_proxy=http://proxy:8080
export https_proxy=http://proxy:8080
export no_proxy=localhost,127.0.0.1
curl https://example.com   # 環境変数のプロキシを使用
```

### 4.3 SSL/TLS

```bash
# 証明書検証スキップ（開発環境のみ）
curl -k https://self-signed.example.com

# CA証明書指定
curl --cacert ca.pem https://example.com

# CA証明書ディレクトリ指定
curl --capath /etc/ssl/certs https://example.com

# TLSバージョンの指定
curl --tlsv1.2 https://example.com           # TLS 1.2以上
curl --tlsv1.3 https://example.com           # TLS 1.3以上

# 暗号スイートの指定
curl --ciphers "ECDHE-RSA-AES256-GCM-SHA384" https://example.com

# 証明書情報の表示
curl -vI https://example.com 2>&1 | grep -A 5 "Server certificate"

# HSTS の確認
curl -sI https://example.com | grep -i "strict-transport-security"
```

### 4.4 レスポンス情報の取得（-w オプション）

```bash
# ステータスコードのみ取得
curl -s -o /dev/null -w "%{http_code}" https://example.com
# 出力: 200

# 応答時間の取得
curl -s -o /dev/null -w "%{time_total}" https://example.com
# 出力: 0.123456

# 詳細なタイミング情報
curl -s -o /dev/null -w "\
DNS解決:    %{time_namelookup}s\n\
TCP接続:    %{time_connect}s\n\
TLSハンドシェイク: %{time_appconnect}s\n\
リダイレクト: %{time_redirect}s\n\
TTFB:       %{time_starttransfer}s\n\
合計:       %{time_total}s\n\
" https://example.com

# -w で使える変数（主要なもの）
# %{http_code}:          ステータスコード
# %{http_version}:       HTTPバージョン
# %{url_effective}:      最終URL（リダイレクト後）
# %{content_type}:       Content-Type
# %{size_download}:      ダウンロードサイズ（バイト）
# %{size_header}:        ヘッダサイズ
# %{speed_download}:     ダウンロード速度（バイト/秒）
# %{time_namelookup}:    DNS解決時間
# %{time_connect}:       TCP接続時間
# %{time_appconnect}:    TLSハンドシェイク完了時間
# %{time_pretransfer}:   転送開始前の時間
# %{time_starttransfer}: 最初のバイト受信時間（TTFB）
# %{time_redirect}:      リダイレクト時間
# %{time_total}:         合計時間
# %{num_redirects}:      リダイレクト回数
# %{ssl_verify_result}:  SSL検証結果（0=成功）
# %{local_ip}:           ローカルIPアドレス
# %{remote_ip}:          リモートIPアドレス
# %{remote_port}:        リモートポート

# JSON 形式で出力
curl -s -o /dev/null -w '{"code":%{http_code},"time":%{time_total},"size":%{size_download}}' \
  https://example.com

# ファイルからフォーマットを読み込む
# format.txt: %{http_code}\t%{time_total}\t%{url_effective}\n
curl -s -o /dev/null -w @format.txt https://example.com
```

### 4.5 ダウンロードの制御

```bash
# ダウンロードの再開
curl -C - -O https://example.com/large.zip
# -C -: 前回の続きから自動的に再開

# 帯域制限
curl --limit-rate 1M -O https://example.com/large.zip  # 1MB/s

# プログレスバー
curl --progress-bar -O https://example.com/large.zip
# #####################################    85%

# 最大ファイルサイズ制限
curl --max-filesize 10485760 -O https://example.com/file.zip
# 10MB以上なら中止

# 条件付きダウンロード（更新されていれば取得）
curl -z "2024-01-01" https://example.com/file.txt  # 指定日以降に更新されていれば
curl -z localfile.txt -O https://example.com/file.txt  # ローカルファイルより新しければ

# 並列ダウンロード（curl 7.66+）
curl --parallel --parallel-max 5 \
  -O https://example.com/file1.zip \
  -O https://example.com/file2.zip \
  -O https://example.com/file3.zip
```

### 4.6 curl の設定ファイル

```bash
# ~/.curlrc に共通設定を記述
# 例:
# --connect-timeout 10
# --max-time 60
# --retry 3
# --silent
# --show-error
# --location
# --user-agent "MyCurlClient/1.0"

# 設定ファイルを無視
curl -q https://example.com

# 別の設定ファイルを指定
curl -K /path/to/config https://example.com
curl --config /path/to/config https://example.com

# 設定ファイルの例（api_config.txt）:
# url = "https://api.example.com/data"
# header = "Authorization: Bearer TOKEN"
# header = "Accept: application/json"
# silent
# show-error

# 使用
curl -K api_config.txt
```

---

## 5. wget

### 5.1 基本的なダウンロード

```bash
# 基本: wget [オプション] URL

# ファイルダウンロード
wget https://example.com/file.zip             # ダウンロード
wget -O output.zip https://example.com/file   # ファイル名指定
wget -O - https://example.com                 # stdoutに出力
wget -c https://example.com/large.zip         # 中断したダウンロードの再開
wget -q https://example.com/file.zip          # 静かにダウンロード
wget -nv https://example.com/file.zip         # 簡易表示

# 複数ファイル
wget -i urls.txt                              # URLリストから一括DL

# 帯域制限
wget --limit-rate=1m https://example.com/large.zip  # 1MB/s制限

# ユーザーエージェント
wget -U "Mozilla/5.0" https://example.com

# バックグラウンドダウンロード
wget -b https://example.com/large.zip
# ログは wget-log に出力
tail -f wget-log                              # 進捗確認
```

### 5.2 再帰的ダウンロード・ミラーリング

```bash
# ミラーリング（Webサイト丸ごとダウンロード）
wget -m https://example.com                   # ミラー
wget -m -p -k https://example.com             # ページ表示に必要なファイル含む
# -m: ミラー（再帰 + タイムスタンプ + 無限深度）
# -p: ページ表示に必要な画像/CSS/JS
# -k: ローカルリンクに変換

# 完全なオフラインコピー
wget -m -p -k -E https://example.com
# -E: .htmlを拡張子に付与

# 再帰ダウンロード
wget -r -l 2 https://example.com              # 深度2まで再帰
wget -r --accept=pdf https://example.com      # PDFのみ
wget -r --reject=jpg,png https://example.com  # 画像除外
wget -r -A "*.pdf,*.doc" https://example.com  # PDFとDOCのみ
wget -r -R "*.exe,*.zip" https://example.com  # exeとzip除外

# ドメイン制限
wget -r -np https://example.com/docs/         # 親ディレクトリに遡らない
wget -r -D example.com https://example.com    # 指定ドメインのみ
wget -r --exclude-domains=ads.example.com https://example.com

# ウェイト（サーバー負荷軽減）
wget -r -w 2 https://example.com              # 2秒間隔
wget -r --random-wait https://example.com     # ランダム間隔（0.5〜1.5倍）
wget -r -w 1 --random-wait https://example.com  # 0.5〜1.5秒のランダム間隔

# ロボット除外を無視（注意して使用）
wget -r -e robots=off https://example.com
```

### 5.3 認証とセキュリティ

```bash
# Basic認証
wget --user=username --password=password https://secure.example.com

# パスワードプロンプト
wget --user=username --ask-password https://secure.example.com

# Cookie
wget --load-cookies cookies.txt https://example.com
wget --save-cookies cookies.txt https://example.com

# SSL 証明書
wget --no-check-certificate https://self-signed.example.com  # 検証スキップ
wget --ca-certificate=ca.pem https://example.com             # CA指定

# ヘッダ追加
wget --header="Authorization: Bearer TOKEN" https://api.example.com
wget --header="Accept: application/json" https://api.example.com

# POST リクエスト
wget --post-data="username=gaku&password=secret" https://example.com/login
wget --post-file=data.json --header="Content-Type: application/json" \
  https://api.example.com/users
```

---

## 6. curl vs wget 比較

```
┌──────────────┬────────────────────┬────────────────────┐
│ 機能         │ curl               │ wget               │
├──────────────┼────────────────────┼────────────────────┤
│ 主な用途     │ API通信・デバッグ  │ ファイルDL         │
│ HTTPメソッド │ 全メソッド対応     │ GET/POST           │
│ プロトコル   │ 多数（FTP,SMTP等） │ HTTP/FTP           │
│ 再帰DL      │ 非対応             │ 対応               │
│ レジューム   │ -C - で対応        │ -c で対応          │
│ 出力先       │ stdout（デフォルト）│ ファイル           │
│ JSON処理     │ --json オプション  │ 非対応             │
│ パイプ       │ 得意               │ 不向き             │
│ Cookie管理   │ -b/-c オプション   │ --cookies          │
│ SSL制御      │ 詳細な制御可能     │ 基本的な制御       │
│ タイミング   │ -w で詳細取得      │ 非対応             │
│ 並列DL      │ --parallel（7.66+）│ 非対応（xargs併用）│
│ ロボット.txt │ 無視               │ 遵守（デフォルト） │
│ ミラーリング │ 非対応             │ -m オプション      │
│ WebSocket    │ 対応（7.86+）      │ 非対応             │
│ HTTP/2       │ 対応               │ 対応（2.0+）       │
│ HTTP/3       │ 対応（実験的）     │ 非対応             │
└──────────────┴────────────────────┴────────────────────┘

使い分けガイド:
  API開発・デバッグ       → curl（メソッド/ヘッダ/認証の柔軟な制御）
  ファイルダウンロード    → wget（curl -O でも可）
  Webサイトミラーリング   → wget -m
  パイプライン処理        → curl -s
  CI/CD スクリプト        → curl（より普及している）
  JSON API テスト         → curl + jq
  ヘルスチェック          → curl -s -o /dev/null -w "%{http_code}"
  レスポンスタイム計測    → curl -w
```

---

## 7. jq との組み合わせ（JSON処理）

### 7.1 基本的な使い方

```bash
# jq: JSON のパース・整形・変換ツール
# インストール:
# macOS: brew install jq
# Ubuntu: sudo apt install jq

# JSON レスポンスの整形
curl -s https://api.github.com/users/octocat | jq '.'

# 特定フィールドの抽出
curl -s https://api.github.com/users/octocat | jq '.name'
# "The Octocat"

# 複数フィールドの抽出
curl -s https://api.github.com/users/octocat | jq '{name: .name, id: .id, location: .location}'

# 文字列として取得（クォートなし）
curl -s https://api.github.com/users/octocat | jq -r '.name'
# The Octocat（クォートなし）
```

### 7.2 配列の処理

```bash
# 配列の全要素の特定フィールド
curl -s https://api.github.com/users/octocat/repos | jq '.[].name'

# 配列の最初の要素
curl -s https://api.github.com/users/octocat/repos | jq '.[0]'

# 配列のスライス
curl -s https://api.github.com/users/octocat/repos | jq '.[:5]'  # 最初の5つ

# 配列の長さ
curl -s https://api.github.com/users/octocat/repos | jq 'length'

# オブジェクトの構築
curl -s https://api.github.com/users/octocat/repos \
  | jq '.[] | {name, stars: .stargazers_count, lang: .language}'

# テーブル形式で出力
curl -s https://api.github.com/users/octocat/repos \
  | jq -r '.[] | "\(.name)\t\(.stargazers_count)\t\(.language)"' \
  | sort -t$'\t' -k2 -rn \
  | head -10
```

### 7.3 フィルタリングと変換

```bash
# 条件でフィルタ
curl -s https://api.github.com/users/octocat/repos \
  | jq '[.[] | select(.stargazers_count > 100)]'

# 言語でフィルタ
curl -s https://api.github.com/users/octocat/repos \
  | jq '[.[] | select(.language == "Ruby")]'

# NULL でないものだけ
curl -s https://api.github.com/users/octocat/repos \
  | jq '[.[] | select(.language != null)]'

# ソート
curl -s https://api.github.com/users/octocat/repos \
  | jq 'sort_by(-.stargazers_count) | .[:5]'

# グループ化（言語別リポジトリ数）
curl -s https://api.github.com/users/octocat/repos \
  | jq 'group_by(.language) | map({language: .[0].language, count: length}) | sort_by(-.count)'

# 集計
curl -s https://api.github.com/users/octocat/repos \
  | jq '[.[].stargazers_count] | add'  # スター数の合計

# map / reduce
curl -s https://api.github.com/users/octocat/repos \
  | jq 'map(.name) | join(", ")'  # 名前をカンマ区切りに

# if-then-else
curl -s https://api.github.com/users/octocat/repos \
  | jq '.[] | {name, popularity: (if .stargazers_count > 1000 then "popular" elif .stargazers_count > 100 then "moderate" else "niche" end)}'
```

### 7.4 jq の高度な使い方

```bash
# JSON の更新（入力を変更して出力）
echo '{"name":"Gaku","age":30}' | jq '.age = 31'

# フィールドの追加
echo '{"name":"Gaku"}' | jq '. + {country: "Japan"}'

# フィールドの削除
echo '{"name":"Gaku","age":30,"email":"a@b.com"}' | jq 'del(.email)'

# 型変換
echo '{"count":"42"}' | jq '.count | tonumber'

# CSV への変換
curl -s https://api.github.com/users/octocat/repos \
  | jq -r '.[] | [.name, .stargazers_count, .language // "N/A"] | @csv'

# TSV への変換
curl -s https://api.github.com/users/octocat/repos \
  | jq -r '.[] | [.name, .stargazers_count, .language // "N/A"] | @tsv'

# HTML への変換
curl -s https://api.github.com/users/octocat/repos \
  | jq -r '.[] | "<li>\(.name) (\(.stargazers_count) stars)</li>"'

# 変数の使用
curl -s https://api.github.com/users/octocat/repos \
  | jq --arg lang "Ruby" '[.[] | select(.language == $lang)]'

# 複数の JSON ファイルの処理
jq -s '.' file1.json file2.json  # 配列にまとめる
jq -s 'add' file1.json file2.json  # マージ
```

---

## 8. 実践パターン

### 8.1 APIのヘルスチェック

```bash
# 基本的なヘルスチェック
check_api() {
    local url=$1
    local timeout=${2:-5}
    local status

    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$timeout" --max-time "$timeout" "$url")

    if [ "$status" -eq 200 ]; then
        echo "OK: $url (${status})"
        return 0
    else
        echo "FAIL: $url (${status})"
        return 1
    fi
}

# 単一エンドポイント
check_api "https://api.example.com/health"

# 複数エンドポイント
for endpoint in /health /ready /metrics; do
    check_api "https://api.example.com${endpoint}"
done

# JSONレスポンスのチェック
check_api_json() {
    local url=$1
    local expected_field=$2
    local expected_value=$3

    local response
    response=$(curl -s --max-time 5 "$url")

    local actual
    actual=$(echo "$response" | jq -r ".$expected_field" 2>/dev/null)

    if [ "$actual" = "$expected_value" ]; then
        echo "OK: $url ($expected_field = $expected_value)"
        return 0
    else
        echo "FAIL: $url ($expected_field = $actual, expected $expected_value)"
        return 1
    fi
}

check_api_json "https://api.example.com/health" "status" "ok"
```

### 8.2 レスポンス時間計測

```bash
# 複数エンドポイントの応答時間計測
for endpoint in /users /posts /comments; do
    time=$(curl -s -o /dev/null -w "%{time_total}" "https://api.example.com$endpoint")
    echo "$endpoint: ${time}s"
done

# 詳細なタイミング計測スクリプト
measure_api() {
    local url=$1
    curl -s -o /dev/null -w "\
URL: %{url_effective}\n\
ステータス: %{http_code}\n\
DNS解決: %{time_namelookup}s\n\
TCP接続: %{time_connect}s\n\
TLS: %{time_appconnect}s\n\
TTFB: %{time_starttransfer}s\n\
合計: %{time_total}s\n\
サイズ: %{size_download} bytes\n\
速度: %{speed_download} bytes/s\n\
---\n\
" "$url"
}

# 複数回計測して平均を出す
measure_avg() {
    local url=$1
    local count=${2:-10}
    local total=0

    for i in $(seq 1 "$count"); do
        time=$(curl -s -o /dev/null -w "%{time_total}" "$url")
        total=$(echo "$total + $time" | bc)
        echo "  試行 $i: ${time}s"
    done

    avg=$(echo "scale=4; $total / $count" | bc)
    echo "平均: ${avg}s ($count 回)"
}

measure_avg "https://api.example.com/health" 5
```

### 8.3 ファイルの並列ダウンロード

```bash
# xargs を使った並列ダウンロード
cat urls.txt | xargs -P 4 -I {} curl -sOL {}
# -P 4: 4並列
# -s: サイレント
# -O: 元のファイル名で保存
# -L: リダイレクト追従

# aria2c（高速ダウンローダー）
# brew install aria2
aria2c -x 16 -s 16 https://example.com/large.zip
# -x 16: 最大16接続
# -s 16: 16分割ダウンロード

# wget + xargs
cat urls.txt | xargs -P 4 -I {} wget -q {}

# curl のビルトイン並列（7.66+）
curl --parallel --parallel-max 4 \
  -O https://example.com/file1.zip \
  -O https://example.com/file2.zip \
  -O https://example.com/file3.zip \
  -O https://example.com/file4.zip
```

### 8.4 Webhook テスト

```bash
# Slack Webhook
curl -X POST https://hooks.slack.com/services/XXX/YYY/ZZZ \
  -H "Content-Type: application/json" \
  -d '{"text": "Deploy completed! :rocket:"}'

# Discord Webhook
curl -X POST "https://discord.com/api/webhooks/ID/TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Deploy completed!"}'

# GitHub API（Issue 作成）
curl -X POST https://api.github.com/repos/owner/repo/issues \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  -d '{"title": "Bug report", "body": "Description here", "labels": ["bug"]}'

# PagerDuty アラート
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "routing_key": "ROUTING_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Server CPU > 90%",
      "severity": "critical",
      "source": "web-server-01"
    }
  }'
```

### 8.5 REST API の CRUD テスト

```bash
#!/bin/bash
# api_crud_test.sh - REST API の CRUD テスト

BASE_URL="${1:-https://jsonplaceholder.typicode.com}"
TOKEN="${API_TOKEN:-}"
AUTH_HEADER=""
[ -n "$TOKEN" ] && AUTH_HEADER="-H \"Authorization: Bearer $TOKEN\""

echo "=== REST API CRUD テスト ==="
echo "Base URL: $BASE_URL"
echo ""

# Create
echo "--- CREATE (POST) ---"
CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/posts" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Post", "body": "Hello, World!", "userId": 1}')
CREATE_BODY=$(echo "$CREATE_RESPONSE" | head -n -1)
CREATE_STATUS=$(echo "$CREATE_RESPONSE" | tail -1)
echo "Status: $CREATE_STATUS"
echo "Response: $(echo "$CREATE_BODY" | jq -c '.')"
CREATED_ID=$(echo "$CREATE_BODY" | jq -r '.id')
echo ""

# Read (single)
echo "--- READ (GET) ---"
READ_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/posts/1")
READ_STATUS=$(echo "$READ_RESPONSE" | tail -1)
echo "Status: $READ_STATUS"
echo "Response: $(echo "$READ_RESPONSE" | head -n -1 | jq -c '{id, title}')"
echo ""

# Read (list)
echo "--- READ LIST (GET) ---"
LIST_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/posts?_limit=3")
LIST_STATUS=$(echo "$LIST_RESPONSE" | tail -1)
echo "Status: $LIST_STATUS"
echo "Count: $(echo "$LIST_RESPONSE" | head -n -1 | jq 'length')"
echo ""

# Update
echo "--- UPDATE (PUT) ---"
UPDATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "$BASE_URL/posts/1" \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "title": "Updated Title", "body": "Updated body", "userId": 1}')
UPDATE_STATUS=$(echo "$UPDATE_RESPONSE" | tail -1)
echo "Status: $UPDATE_STATUS"
echo "Response: $(echo "$UPDATE_RESPONSE" | head -n -1 | jq -c '{id, title}')"
echo ""

# Partial Update
echo "--- PARTIAL UPDATE (PATCH) ---"
PATCH_RESPONSE=$(curl -s -w "\n%{http_code}" -X PATCH "$BASE_URL/posts/1" \
  -H "Content-Type: application/json" \
  -d '{"title": "Patched Title"}')
PATCH_STATUS=$(echo "$PATCH_RESPONSE" | tail -1)
echo "Status: $PATCH_STATUS"
echo "Response: $(echo "$PATCH_RESPONSE" | head -n -1 | jq -c '{id, title}')"
echo ""

# Delete
echo "--- DELETE ---"
DELETE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/posts/1")
echo "Status: $DELETE_STATUS"
echo ""

echo "=== テスト完了 ==="
```

### 8.6 API 監視スクリプト

```bash
#!/bin/bash
# api_monitor.sh - APIの継続監視スクリプト

ENDPOINTS=(
    "https://api.example.com/health"
    "https://api.example.com/v1/status"
    "https://web.example.com/"
)
CHECK_INTERVAL=60
TIMEOUT=10
LOG_FILE="/tmp/api_monitor.log"
ALERT_THRESHOLD=3  # 連続失敗回数

declare -A FAIL_COUNTS

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_endpoint() {
    local url=$1
    local result

    result=$(curl -s -o /dev/null \
        -w "%{http_code},%{time_total},%{size_download}" \
        --connect-timeout "$TIMEOUT" \
        --max-time "$TIMEOUT" \
        "$url" 2>/dev/null)

    local status=$(echo "$result" | cut -d, -f1)
    local time=$(echo "$result" | cut -d, -f2)
    local size=$(echo "$result" | cut -d, -f3)

    if [ "$status" -ge 200 ] && [ "$status" -lt 400 ]; then
        FAIL_COUNTS[$url]=0
        log "OK $url status=$status time=${time}s size=${size}b"
        return 0
    else
        local count=${FAIL_COUNTS[$url]:-0}
        count=$((count + 1))
        FAIL_COUNTS[$url]=$count

        log "FAIL $url status=$status time=${time}s (連続${count}回目)"

        if [ "$count" -ge "$ALERT_THRESHOLD" ]; then
            log "ALERT: $url が ${count} 回連続で失敗"
            # アラート送信（Slack、PagerDuty等）
            # curl -X POST "$SLACK_WEBHOOK" -d "{\"text\":\"ALERT: $url down\"}"
        fi
        return 1
    fi
}

log "監視開始 (${#ENDPOINTS[@]} エンドポイント, ${CHECK_INTERVAL}秒間隔)"

while true; do
    for url in "${ENDPOINTS[@]}"; do
        check_endpoint "$url"
    done
    sleep "$CHECK_INTERVAL"
done
```

### 8.7 curl を使ったスモークテスト

```bash
#!/bin/bash
# smoke_test.sh - デプロイ後のスモークテスト

BASE_URL="${1:?使い方: $0 <base_url>}"
PASSED=0
FAILED=0
TOTAL=0

assert_status() {
    local description=$1
    local expected_status=$2
    local url=$3
    shift 3  # 残りは curl オプション

    TOTAL=$((TOTAL + 1))
    local actual_status
    actual_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$@" "$url")

    if [ "$actual_status" = "$expected_status" ]; then
        echo "  PASS: $description (status=$actual_status)"
        PASSED=$((PASSED + 1))
    else
        echo "  FAIL: $description (expected=$expected_status, actual=$actual_status)"
        FAILED=$((FAILED + 1))
    fi
}

assert_contains() {
    local description=$1
    local expected_text=$2
    local url=$3

    TOTAL=$((TOTAL + 1))
    local body
    body=$(curl -s --max-time 10 "$url")

    if echo "$body" | grep -q "$expected_text"; then
        echo "  PASS: $description"
        PASSED=$((PASSED + 1))
    else
        echo "  FAIL: $description (body does not contain '$expected_text')"
        FAILED=$((FAILED + 1))
    fi
}

echo "=== スモークテスト: $BASE_URL ==="
echo ""

# ヘルスチェック
assert_status "ヘルスチェック" "200" "$BASE_URL/health"

# トップページ
assert_status "トップページ" "200" "$BASE_URL/"
assert_contains "トップページにタイトルが含まれる" "<title>" "$BASE_URL/"

# API エンドポイント
assert_status "API v1 ステータス" "200" "$BASE_URL/api/v1/status"

# 認証なしアクセスの拒否
assert_status "認証なし → 401" "401" "$BASE_URL/api/v1/protected"

# 存在しないページ
assert_status "404 ページ" "404" "$BASE_URL/nonexistent-page"

# リダイレクト
assert_status "HTTP→HTTPS リダイレクト" "301" "http://${BASE_URL#https://}/"

echo ""
echo "=== 結果: $PASSED/$TOTAL 成功, $FAILED/$TOTAL 失敗 ==="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
```

---

## 9. セキュリティベストプラクティス

```bash
# 1. 認証情報をコマンドラインに直接書かない
# 悪い例（psで見える、historyに残る）
curl -u "user:password" https://api.example.com

# 良い例（.netrc ファイルを使用）
curl -n https://api.example.com

# 良い例（環境変数を使用）
curl -H "Authorization: Bearer ${API_TOKEN}" https://api.example.com

# 2. HTTPS を常に使用
# curl はデフォルトで証明書を検証する → -k は開発環境のみ

# 3. レスポンスの検証
response=$(curl -s -w "\n%{http_code}" https://api.example.com/data)
status=$(echo "$response" | tail -1)
body=$(echo "$response" | head -n -1)
if [ "$status" != "200" ]; then
    echo "エラー: ステータス $status" >&2
    exit 1
fi

# 4. タイムアウトを必ず設定
curl --connect-timeout 5 --max-time 30 https://api.example.com

# 5. 出力をサニタイズ（ログに認証情報を残さない）
curl -v https://api.example.com 2>&1 | sed 's/Authorization:.*/Authorization: [REDACTED]/'

# 6. .curlrc でデフォルト設定
# ~/.curlrc:
# --connect-timeout 10
# --max-time 60
# --location
# --fail

# 7. 一時ファイルのセキュアな扱い
TMPFILE=$(mktemp)
chmod 600 "$TMPFILE"
trap "rm -f $TMPFILE" EXIT
curl -s https://api.example.com/data > "$TMPFILE"
```

---

## 10. よくある質問（FAQ）

### Q1: curl と wget、どちらを使うべき？

```bash
# 基本的な使い分け:
# - API操作・RESTリクエスト → curl（メソッド・ヘッダ・ボディの柔軟な制御）
# - ファイルダウンロード → wget（再帰ダウンロード・ミラーリング）
# - スクリプトでの自動化 → curl（戻り値・出力のカスタマイズが豊富）
# - 帯域が不安定な環境 → wget（-c で中断再開が標準サポート）

# curl でも wget 的な使い方は可能
curl -LOC - https://example.com/large.zip
# -L: リダイレクト追従
# -O: 元のファイル名で保存
# -C -: 中断再開
```

### Q2: curl でリクエストボディが大きすぎてコマンドラインに収まらない場合は？

```bash
# 方法1: ファイルから読み込む（@プレフィックス）
curl -X POST https://api.example.com/data \
  -H "Content-Type: application/json" \
  -d @request.json

# 方法2: stdin から読み込む（@-）
cat request.json | curl -X POST https://api.example.com/data \
  -H "Content-Type: application/json" \
  -d @-

# 方法3: ヒアドキュメント
curl -X POST https://api.example.com/data \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "title": "大きなリクエスト",
  "items": [1, 2, 3, 4, 5]
}
EOF
```

### Q3: curl でレスポンスヘッダだけ取得したい場合は？

```bash
# HEAD リクエスト（-I）
curl -I https://example.com
# GET のレスポンスヘッダのみ（-sD - -o /dev/null）
curl -sD - -o /dev/null https://example.com
# 特定のヘッダだけ取得
curl -sI https://example.com | grep -i "content-type"
# -w でも取得可能（curl 7.84+）
curl -s -o /dev/null -w "%header{content-type}" https://example.com
```

### Q4: curl で Cookie を使うには？

```bash
# Cookie の送信
curl -b "session=abc123; lang=ja" https://example.com

# Cookie をファイルに保存
curl -c cookies.txt https://example.com/login -d "user=admin&pass=secret"

# 保存した Cookie を使ってリクエスト
curl -b cookies.txt https://example.com/dashboard

# Cookie の保存と送信を同時に（セッション維持）
curl -b cookies.txt -c cookies.txt https://example.com/page1
curl -b cookies.txt -c cookies.txt https://example.com/page2
```

### Q5: jq がインストールされていない環境で JSON を処理するには？

```bash
# Python を使う（ほとんどの環境で利用可能）
curl -s https://api.example.com/data | python3 -m json.tool

# 特定のフィールド抽出
curl -s https://api.example.com/data | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data['name'])
"

# grep で簡易的に抽出（非推奨だが緊急時に）
curl -s https://api.example.com/data | grep -o '"name":"[^"]*"' | head -1

# sed で簡易的に整形
curl -s https://api.example.com/data | sed 's/,/,\n/g; s/{/{\n/g; s/}/\n}/g'

# Node.js が使える場合
curl -s https://api.example.com/data | node -e "
const chunks = [];
process.stdin.on('data', c => chunks.push(c));
process.stdin.on('end', () => {
  const data = JSON.parse(chunks.join(''));
  console.log(data.name);
});
"
```

### Q6: SSL証明書エラーが出る場合の対処法は？

```bash
# 証明書の詳細を確認
curl -vI https://example.com 2>&1 | grep -A5 "SSL certificate"

# 自己署名証明書を許可（開発環境のみ！）
curl -k https://localhost:8443/api

# CA証明書を明示的に指定
curl --cacert /path/to/ca-bundle.crt https://example.com

# 証明書チェーンの確認
openssl s_client -connect example.com:443 -showcerts </dev/null 2>/dev/null \
  | openssl x509 -noout -dates -subject -issuer

# 本番環境では -k を使わず、証明書の問題を根本的に解決する
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


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| コマンド | 用途 | よく使うオプション |
|---------|------|-------------------|
| curl -s URL | GET リクエスト | -s（サイレント）, -S（エラー表示） |
| curl -X POST --json '{...}' | JSON POST | --json（curl 7.82+） |
| curl -H "Authorization: Bearer TOKEN" | 認証付きリクエスト | -H（ヘッダ追加） |
| curl -w "%{http_code}" | ステータスコード取得 | -o /dev/null と組み合わせ |
| curl -w "%{time_total}" | 応答時間取得 | |
| curl -L | リダイレクト追従 | --max-redirs で制限 |
| curl --connect-timeout 5 --max-time 30 | タイムアウト設定 | 必ず設定すべき |
| curl --retry 3 | リトライ | --retry-delay, --retry-all-errors |
| wget -c URL | 中断可能なダウンロード | -q（静か）, -O（出力先） |
| wget -m URL | Webサイトミラーリング | -p -k で完全コピー |
| jq '.field' | JSON パース | -r（raw出力）, -c（コンパクト） |

---

## 次に読むべきガイド

---

## 参考文献
1. Stenberg, D. "Everything curl." curl.se, 2024.
2. Barrett, D. "Efficient Linux at the Command Line." Ch.11, O'Reilly, 2022.
3. "jq Manual." jqlang.github.io/jq/manual.
4. "GNU Wget Manual." gnu.org/software/wget/manual.
5. "curl man page." curl.se/docs/manpage.html.
