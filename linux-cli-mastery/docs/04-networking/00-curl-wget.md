# ネットワークツール（curl, wget）

> curl と wget は CLI からの HTTP 通信を可能にする、API開発・デバッグの必須ツール。

## この章で学ぶこと

- [ ] curl で各種 HTTP リクエストを送信できる
- [ ] wget でファイルダウンロード・ミラーリングができる
- [ ] API テスト・デバッグに活用できる

---

## 1. curl の基本

```bash
# 基本: curl [オプション] URL

# GETリクエスト
curl https://example.com                     # HTML取得
curl -o output.html https://example.com      # ファイルに保存
curl -O https://example.com/file.zip         # 元のファイル名で保存
curl -s https://api.example.com/data         # サイレントモード
curl -sS https://api.example.com/data        # エラーのみ表示

# レスポンスヘッダ
curl -I https://example.com                  # ヘッダのみ（HEAD）
curl -i https://example.com                  # ヘッダ + ボディ
curl -v https://example.com                  # 詳細（リクエスト含む）

# リダイレクト
curl -L https://example.com                  # リダイレクトに追従
curl -L --max-redirs 5 https://example.com   # 最大リダイレクト数
```

### HTTP メソッド

```bash
# POST
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Gaku", "email": "gaku@example.com"}'

# JSON データを送る（省略形）
curl -X POST https://api.example.com/users \
  --json '{"name": "Gaku"}'                 # curl 7.82+

# ファイルからデータを読み込む
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d @data.json

# PUT
curl -X PUT https://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated"}'

# PATCH
curl -X PATCH https://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Patched"}'

# DELETE
curl -X DELETE https://api.example.com/users/1

# フォームデータ
curl -X POST https://example.com/upload \
  -F "file=@photo.jpg" \
  -F "name=test"
```

### 認証

```bash
# Basic認証
curl -u username:password https://api.example.com

# Bearer トークン
curl -H "Authorization: Bearer TOKEN_HERE" https://api.example.com

# APIキー（ヘッダ）
curl -H "X-API-Key: KEY_HERE" https://api.example.com

# Cookie
curl -b "session=abc123" https://example.com
curl -c cookies.txt https://example.com      # Cookie保存
curl -b cookies.txt https://example.com      # Cookie送信
```

### 高度なオプション

```bash
# タイムアウト
curl --connect-timeout 5 https://example.com   # 接続タイムアウト
curl --max-time 30 https://example.com         # 全体タイムアウト

# リトライ
curl --retry 3 --retry-delay 2 https://example.com

# プロキシ
curl -x http://proxy:8080 https://example.com
curl --proxy socks5://proxy:1080 https://example.com

# SSL/TLS
curl -k https://self-signed.example.com        # 証明書検証スキップ
curl --cacert ca.pem https://example.com       # CA証明書指定

# レスポンス情報の取得
curl -s -o /dev/null -w "%{http_code}" https://example.com  # ステータスコードのみ
curl -s -o /dev/null -w "%{time_total}" https://example.com # 応答時間

# -w で使える変数
# %{http_code}:    ステータスコード
# %{time_total}:   合計時間
# %{time_connect}: 接続時間
# %{size_download}: ダウンロードサイズ
# %{speed_download}: ダウンロード速度
```

---

## 2. wget

```bash
# 基本: wget [オプション] URL

# ファイルダウンロード
wget https://example.com/file.zip             # ダウンロード
wget -O output.zip https://example.com/file   # ファイル名指定
wget -c https://example.com/large.zip         # 中断したダウンロードの再開
wget -q https://example.com/file.zip          # 静かにダウンロード

# 複数ファイル
wget -i urls.txt                              # URLリストから一括DL

# 帯域制限
wget --limit-rate=1m https://example.com/large.zip  # 1MB/s制限

# ミラーリング（Webサイト丸ごとダウンロード）
wget -m https://example.com                   # ミラー
wget -m -p -k https://example.com             # ページ表示に必要なファイル含む
# -m: ミラー（再帰 + タイムスタンプ + 無限深度）
# -p: ページ表示に必要な画像/CSS/JS
# -k: ローカルリンクに変換

# 再帰ダウンロード
wget -r -l 2 https://example.com              # 深度2まで再帰
wget -r --accept=pdf https://example.com      # PDFのみ
wget -r --reject=jpg,png https://example.com  # 画像除外
```

---

## 3. curl vs wget 比較

```
┌─────────────┬──────────────────┬──────────────────┐
│ 機能        │ curl             │ wget             │
├─────────────┼──────────────────┼──────────────────┤
│ 主な用途    │ API通信・デバッグ│ ファイルDL       │
│ HTTPメソッド│ 全メソッド対応   │ GET/POST         │
│ プロトコル  │ 多数（FTP,SMTP等）│ HTTP/FTP        │
│ 再帰DL     │ 非対応           │ 対応             │
│ レジューム  │ -C - で対応      │ -c で対応        │
│ 出力        │ stdout           │ ファイル         │
│ JSON処理    │ --json オプション│ 非対応           │
│ パイプ      │ 得意             │ 不向き           │
└─────────────┴──────────────────┴──────────────────┘

使い分け:
  API開発・デバッグ       → curl
  ファイルダウンロード    → wget（curl -O でも可）
  Webサイトミラーリング   → wget
  パイプライン処理        → curl
  CI/CD スクリプト        → curl
```

---

## 4. jq との組み合わせ（JSON処理）

```bash
# jq: JSON のパース・整形ツール
# brew install jq

# JSON レスポンスの整形
curl -s https://api.github.com/users/octocat | jq '.'

# 特定フィールドの抽出
curl -s https://api.github.com/users/octocat | jq '.name'
curl -s https://api.github.com/users/octocat | jq '.login, .id'

# 配列の処理
curl -s https://api.github.com/users/octocat/repos | jq '.[].name'
curl -s https://api.github.com/users/octocat/repos | jq '.[] | {name, stars: .stargazers_count}'

# フィルタリング
curl -s https://api.github.com/users/octocat/repos \
  | jq '[.[] | select(.stargazers_count > 100)] | length'

# 実践例: GitHub API でリポジトリ情報を取得・整形
curl -s "https://api.github.com/search/repositories?q=language:rust&sort=stars" \
  | jq '.items[:5] | .[] | "\(.full_name) ⭐\(.stargazers_count)"'
```

---

## 5. 実践パターン

```bash
# パターン1: APIのヘルスチェック
check_api() {
    local status=$(curl -s -o /dev/null -w "%{http_code}" "$1")
    if [ "$status" -eq 200 ]; then
        echo "OK: $1"
    else
        echo "FAIL: $1 (status: $status)"
    fi
}
check_api "https://api.example.com/health"

# パターン2: 複数エンドポイントの応答時間計測
for endpoint in /users /posts /comments; do
    time=$(curl -s -o /dev/null -w "%{time_total}" "https://api.example.com$endpoint")
    echo "$endpoint: ${time}s"
done

# パターン3: ファイルの並列ダウンロード
cat urls.txt | xargs -P 4 -I {} curl -sOL {}
# -P 4: 4並列

# パターン4: Webhookのテスト
curl -X POST https://hooks.slack.com/services/XXX \
  -H "Content-Type: application/json" \
  -d '{"text": "Deploy completed!"}'

# パターン5: REST API の CRUD テスト
BASE="https://jsonplaceholder.typicode.com"
# Create
curl -sX POST "$BASE/posts" --json '{"title":"test","body":"hello"}' | jq
# Read
curl -s "$BASE/posts/1" | jq
# Update
curl -sX PUT "$BASE/posts/1" --json '{"title":"updated"}' | jq
# Delete
curl -sX DELETE "$BASE/posts/1"
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| curl -s URL | GET リクエスト |
| curl -X POST --json '{...}' | JSON POST |
| curl -H "Authorization: Bearer TOKEN" | 認証付きリクエスト |
| curl -w "%{http_code}" | ステータスコード取得 |
| wget -c URL | 中断可能なダウンロード |
| wget -m URL | Webサイトミラーリング |
| jq '.field' | JSON パース |

---

## 次に読むべきガイド
→ [[01-ssh-scp.md]] — リモート接続

---

## 参考文献
1. Stenberg, D. "Everything curl." curl.se, 2024.
2. Barrett, D. "Efficient Linux at the Command Line." Ch.11, O'Reilly, 2022.
