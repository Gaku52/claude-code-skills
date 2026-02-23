# HTTP基礎

> HTTPはWebの基盤プロトコル。リクエスト/レスポンスモデル、メソッド、ステータスコード、ヘッダーの仕組みを理解し、Web開発に必須の知識を固める。

## この章で学ぶこと

- [ ] HTTPのリクエスト/レスポンス構造を理解する
- [ ] HTTPメソッドの意味と使い分けを把握する
- [ ] ステータスコードの分類と主要なコードを学ぶ
- [ ] HTTPヘッダーの種類と役割を把握する
- [ ] コネクション管理とパフォーマンスの関係を理解する
- [ ] HTTPSとセキュリティの基礎を学ぶ
- [ ] 実務でのHTTPデバッグ手法を習得する

---

## 1. HTTPの基本

```
HTTP（HyperText Transfer Protocol）:
  → Web上でデータを転送するためのプロトコル
  → ステートレス（各リクエストは独立）
  → テキストベース（HTTP/1.1）→ バイナリ（HTTP/2以降）
  → TCP（HTTP/1.1, HTTP/2）またはUDP（HTTP/3）上で動作

バージョンの歴史:
  HTTP/0.9 (1991): GETのみ、HTML のみ、ヘッダーなし
  HTTP/1.0 (1996): ヘッダー、POST、ステータスコード追加
                    1リクエストごとにTCP接続を切断
  HTTP/1.1 (1997): Keep-Alive、チャンク転送、Host ヘッダー必須
                    パイプライン（ほぼ使われず）
                    RFC 2616 → RFC 7230-7235 → RFC 9110-9112
  HTTP/2   (2015): バイナリ、多重化、サーバープッシュ、HPACK
                    RFC 7540 → RFC 9113
  HTTP/3   (2022): QUIC ベース、UDP 上で動作、QPACK
                    RFC 9114

リクエスト/レスポンスモデル:
  クライアント                    サーバー
  ┌──────────┐                 ┌──────────┐
  │ ブラウザ  │── リクエスト ──→│ Webサーバー│
  │          │←── レスポンス ──│          │
  └──────────┘                 └──────────┘

  通信の流れ（HTTP/1.1 + TLS）:
  ① TCP 3-way ハンドシェイク（SYN → SYN-ACK → ACK）
  ② TLS ハンドシェイク（ClientHello → ServerHello → ...）
  ③ HTTPリクエスト送信
  ④ HTTPレスポンス受信
  ⑤ Keep-Alive: 同じTCP接続で次のリクエストを送信
  ⑥ アイドルタイムアウト後に接続を切断

ステートレスの意味:
  → 各リクエストは前のリクエストの情報を持たない
  → サーバーはクライアントの状態を記憶しない
  → 状態管理が必要な場合:
     ・Cookie（セッションID）
     ・Token（JWT等）
     ・クエリパラメータ
     ・ローカルストレージ

ステートレスのメリット:
  → サーバーの水平スケーリングが容易
  → 任意のサーバーがリクエストを処理可能
  → 障害時のリカバリが簡単
  → キャッシュが効きやすい

ステートレスのデメリット:
  → 毎回認証情報を送信する必要がある
  → リクエストサイズが大きくなりがち
  → セッション管理に別の仕組みが必要
```

---

## 2. HTTPリクエスト

```
リクエスト構造:

  ┌─────────────────────────────────────────┐
  │ GET /api/users?page=1 HTTP/1.1          │ ← リクエストライン
  ├─────────────────────────────────────────┤
  │ Host: api.example.com                   │ ← ヘッダー
  │ Accept: application/json                │
  │ Authorization: Bearer eyJhbG...         │
  │ User-Agent: Mozilla/5.0                 │
  │ Accept-Encoding: gzip, deflate          │
  │ Connection: keep-alive                  │
  ├─────────────────────────────────────────┤
  │                                         │ ← 空行（ヘッダー終了）
  ├─────────────────────────────────────────┤
  │ （ボディ — GET の場合は通常なし）         │
  └─────────────────────────────────────────┘

リクエストラインの構造:
  メソッド SP リクエストターゲット SP HTTPバージョン CRLF

  GET /api/users?page=1&sort=name HTTP/1.1\r\n
  ↑   ↑                          ↑
  メソッド リクエストターゲット    HTTPバージョン

  リクエストターゲットの形式:
  ① origin-form:   /api/users?page=1（最も一般的）
  ② absolute-form: http://example.com/api/users（プロキシ経由）
  ③ authority-form: example.com:443（CONNECT メソッド用）
  ④ asterisk-form: *（OPTIONS メソッド用）

POST リクエストの例:
  POST /api/users HTTP/1.1
  Host: api.example.com
  Content-Type: application/json
  Content-Length: 52
  Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
  Accept: application/json
  X-Request-Id: 550e8400-e29b-41d4-a716-446655440000

  {"name": "Taro", "email": "taro@example.com"}

フォームデータの例:
  POST /login HTTP/1.1
  Host: example.com
  Content-Type: application/x-www-form-urlencoded
  Content-Length: 32

  username=taro&password=secret123

マルチパートフォームデータの例（ファイルアップロード）:
  POST /api/files HTTP/1.1
  Host: api.example.com
  Content-Type: multipart/form-data; boundary=----WebKitFormBoundary
  Content-Length: 1234

  ------WebKitFormBoundary
  Content-Disposition: form-data; name="description"

  プロフィール画像
  ------WebKitFormBoundary
  Content-Disposition: form-data; name="file"; filename="avatar.png"
  Content-Type: image/png

  [バイナリデータ]
  ------WebKitFormBoundary--
```

---

## 3. HTTPメソッド

```
┌────────┬──────────────────────────┬──────┬──────┬──────┐
│ メソッド│ 用途                     │冪等性│安全性│ボディ│
├────────┼──────────────────────────┼──────┼──────┼──────┤
│ GET    │ リソースの取得            │ ✓    │ ✓    │ なし │
│ POST   │ リソースの作成            │ ✗    │ ✗    │ あり │
│ PUT    │ リソースの完全置換        │ ✓    │ ✗    │ あり │
│ PATCH  │ リソースの部分更新        │ ✗    │ ✗    │ あり │
│ DELETE │ リソースの削除            │ ✓    │ ✗    │ 任意 │
│ HEAD   │ レスポンスヘッダーのみ取得│ ✓    │ ✓    │ なし │
│ OPTIONS│ 対応メソッドの確認(CORS)  │ ✓    │ ✓    │ なし │
│ TRACE  │ ループバックテスト        │ ✓    │ ✓    │ なし │
│ CONNECT│ トンネル確立(HTTPS)       │ ✗    │ ✗    │ なし │
└────────┴──────────────────────────┴──────┴──────┴──────┘

冪等性（Idempotent）:
  → 同じリクエストを何度送っても結果が同じ
  → GET, PUT, DELETE は冪等
  → POST は冪等ではない（毎回新しいリソースが作成される）
  → PATCH は冪等ではない（相対的な変更の可能性）

  冪等性の実務的意味:
  → ネットワークエラーでリクエストが届いたか不明な場合
  → 冪等なメソッドは安全にリトライできる
  → 非冪等なメソッドは冪等性キー（Idempotency-Key）で対応

  例: 決済API
  POST /api/payments
  Idempotency-Key: unique-key-12345
  → 同じキーで再送しても二重決済にならない

安全性（Safe）:
  → サーバーの状態を変更しない
  → GET, HEAD, OPTIONS は安全
  → 安全なメソッドはプリフェッチ・キャッシュが可能
  → Webクローラーは安全なメソッドのみ使用すべき

実務での使い分け:
  一覧取得:  GET  /api/users
  詳細取得:  GET  /api/users/123
  作成:      POST /api/users
  完全更新:  PUT  /api/users/123
  部分更新:  PATCH /api/users/123
  削除:      DELETE /api/users/123
  存在確認:  HEAD /api/users/123
  CORS確認:  OPTIONS /api/users

各メソッドの詳細:

  GET:
  → リソースの取得に使用
  → ボディを含めるべきではない（RFC 9110）
  → クエリパラメータでフィルタリング・ページネーション
  → キャッシュの対象
  → ブックマーク可能
  → ブラウザの戻るボタンで再送安全

  POST:
  → リソースの作成、処理の実行に使用
  → ボディにデータを含める
  → 成功時は 201 Created + Location ヘッダー
  → キャッシュの対象外（通常）
  → 同じリクエストで異なる結果が返る可能性

  PUT:
  → リソースの完全置換に使用
  → 存在しない場合は作成（実装による）
  → 送信データがリソースの完全な表現
  → 省略したフィールドはnull/デフォルトになる
  → 部分更新には使わない → PATCHを使用

  PATCH:
  → リソースの部分更新に使用
  → 変更するフィールドのみ送信
  → JSON Merge Patch（RFC 7396）:
    {"name": "Updated Name"}  ← nameだけ更新
  → JSON Patch（RFC 6902）:
    [{"op": "replace", "path": "/name", "value": "Updated Name"}]

  DELETE:
  → リソースの削除に使用
  → 成功時は 204 No Content または 200 OK
  → 既に削除済みでも 204（冪等性）
  → ソフトデリート vs ハードデリート
```

```typescript
// TypeScript での各メソッドの実装例

// GET — リソースの取得
async function getUsers(page: number = 1, perPage: number = 20): Promise<User[]> {
  const response = await fetch(
    `https://api.example.com/api/users?page=${page}&per_page=${perPage}`,
    {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    },
  );

  if (!response.ok) {
    throw new HttpError(response.status, await response.text());
  }

  const data = await response.json();
  return data.users;
}

// POST — リソースの作成
async function createUser(userData: CreateUserInput): Promise<User> {
  const response = await fetch('https://api.example.com/api/users', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
      'Idempotency-Key': generateIdempotencyKey(),
    },
    body: JSON.stringify(userData),
  });

  if (response.status !== 201) {
    const error = await response.json();
    throw new HttpError(response.status, error.message);
  }

  // Location ヘッダーから作成されたリソースのURLを取得
  const location = response.headers.get('Location');
  console.log(`Created at: ${location}`);

  return response.json();
}

// PUT — リソースの完全置換
async function replaceUser(id: string, userData: User): Promise<User> {
  const response = await fetch(`https://api.example.com/api/users/${id}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(userData),
  });

  if (!response.ok) {
    throw new HttpError(response.status, await response.text());
  }

  return response.json();
}

// PATCH — リソースの部分更新
async function updateUser(
  id: string,
  updates: Partial<User>,
): Promise<User> {
  const response = await fetch(`https://api.example.com/api/users/${id}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/merge-patch+json',
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
      'If-Match': currentETag,  // 楽観的ロック
    },
    body: JSON.stringify(updates),
  });

  if (response.status === 409) {
    throw new ConflictError('Resource was modified by another request');
  }

  if (!response.ok) {
    throw new HttpError(response.status, await response.text());
  }

  return response.json();
}

// DELETE — リソースの削除
async function deleteUser(id: string): Promise<void> {
  const response = await fetch(`https://api.example.com/api/users/${id}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.status === 404) {
    // 既に削除済み — 冪等性の観点から成功扱いにする場合もある
    console.warn(`User ${id} already deleted`);
    return;
  }

  if (!response.ok) {
    throw new HttpError(response.status, await response.text());
  }
}

// HEAD — 存在確認・メタデータ取得
async function checkUserExists(id: string): Promise<boolean> {
  const response = await fetch(`https://api.example.com/api/users/${id}`, {
    method: 'HEAD',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  return response.status === 200;
}

// ファイルアップロード（multipart/form-data）
async function uploadFile(file: File, description: string): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('description', description);

  const response = await fetch('https://api.example.com/api/files', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      // Content-Type は FormData の場合、ブラウザが自動設定
    },
    body: formData,
  });

  if (!response.ok) {
    throw new HttpError(response.status, await response.text());
  }

  const data = await response.json();
  return data.fileUrl;
}
```

---

## 4. ステータスコード

```
ステータスコードの分類:
  1xx: 情報（処理継続中）
  2xx: 成功
  3xx: リダイレクト
  4xx: クライアントエラー
  5xx: サーバーエラー

1xx 情報:
  ┌─────┬──────────────────────────────────────────────┐
  │ 100 │ Continue — ボディの送信を続行してよい          │
  │     │ → 大きなボディ送信前にサーバーが受け入れ可能か │
  │     │   確認（Expect: 100-continue ヘッダー使用）    │
  │ 101 │ Switching Protocols — プロトコル変更          │
  │     │ → WebSocket へのアップグレード時に使用         │
  │ 102 │ Processing — 処理中（WebDAV）                 │
  │ 103 │ Early Hints — リソースの先読みヒント           │
  │     │ → Link: </style.css>; rel=preload             │
  │     │ → HTTP/2 サーバープッシュの代替                │
  └─────┴──────────────────────────────────────────────┘

2xx 成功:
  ┌─────┬──────────────────────────────────────────────┐
  │ 200 │ OK — 成功（最も一般的）                       │
  │ 201 │ Created — リソース作成成功                    │
  │     │ → POSTの成功レスポンス                        │
  │     │ → Location ヘッダーで作成先URLを返す          │
  │ 202 │ Accepted — リクエスト受理（処理は未完了）     │
  │     │ → 非同期処理の開始を通知                      │
  │     │ → ポーリング用URLを返すことが多い             │
  │ 204 │ No Content — 成功（ボディなし）               │
  │     │ → DELETE の成功レスポンス                     │
  │     │ → PUT/PATCH でボディを返さない場合            │
  │ 206 │ Partial Content — 部分的なコンテンツ          │
  │     │ → Range リクエストへのレスポンス              │
  │     │ → 動画ストリーミング、大きなファイルの分割DL  │
  └─────┴──────────────────────────────────────────────┘

3xx リダイレクト:
  ┌─────┬──────────────────────────────────────────────┐
  │ 301 │ Moved Permanently — 恒久移転                 │
  │     │ → メソッドがGETに変更される可能性がある       │
  │     │ → SEO: 検索エンジンが新URLをインデックス      │
  │ 302 │ Found — 一時移転                             │
  │     │ → メソッドがGETに変更される可能性がある       │
  │     │ → 実装が曖昧なため、307/308 推奨             │
  │ 303 │ See Other — 別のURIを参照                    │
  │     │ → POST後にGETでリダイレクト（PRGパターン）    │
  │ 304 │ Not Modified — キャッシュ有効                 │
  │     │ → 条件付きリクエスト（If-None-Match等）の結果│
  │     │ → ボディなし（キャッシュを使用）             │
  │ 307 │ Temporary Redirect — 一時移転                │
  │     │ → メソッドを維持（推奨）                     │
  │ 308 │ Permanent Redirect — 恒久移転                │
  │     │ → メソッドを維持（推奨）                     │
  └─────┴──────────────────────────────────────────────┘

  301 vs 302 vs 307 vs 308 の選択:
  ┌────────────┬──────────────┬──────────────┐
  │            │ 恒久（永続）  │ 一時         │
  ├────────────┼──────────────┼──────────────┤
  │ メソッド変更│ 301          │ 302          │
  │ メソッド維持│ 308（推奨）   │ 307（推奨）  │
  └────────────┴──────────────┴──────────────┘

  PRG（Post-Redirect-Get）パターン:
  ① クライアント: POST /order（注文送信）
  ② サーバー: 303 See Other → /order/123
  ③ クライアント: GET /order/123（注文確認ページ）
  → ブラウザの更新ボタンで二重注文を防止

4xx クライアントエラー:
  ┌─────┬──────────────────────────────────────────────┐
  │ 400 │ Bad Request — リクエスト不正                  │
  │     │ → JSON構文エラー、必須パラメータ欠如          │
  │ 401 │ Unauthorized — 未認証                        │
  │     │ → 認証が必要（ログインしていない）            │
  │     │ → WWW-Authenticate ヘッダーを含めるべき      │
  │ 403 │ Forbidden — アクセス禁止                     │
  │     │ → 認証済みだが権限がない                     │
  │     │ → 管理者専用ページへの一般ユーザーアクセス    │
  │ 404 │ Not Found — リソースが存在しない             │
  │     │ → URLが間違っている、またはリソースが削除済み │
  │ 405 │ Method Not Allowed — メソッド非対応          │
  │     │ → Allow ヘッダーで許可メソッドを通知         │
  │     │ → Allow: GET, POST, HEAD                    │
  │ 406 │ Not Acceptable — Accept ヘッダーと不一致    │
  │ 408 │ Request Timeout — リクエストタイムアウト     │
  │ 409 │ Conflict — 競合                              │
  │     │ → 楽観的ロックの失敗                         │
  │     │ → リソースの状態遷移が不正                   │
  │ 410 │ Gone — 恒久的に削除                          │
  │     │ → 404 と異なり「以前は存在した」ことを示す   │
  │ 411 │ Length Required — Content-Length が必要      │
  │ 413 │ Content Too Large — ボディが大きすぎる       │
  │ 414 │ URI Too Long — URIが長すぎる                │
  │ 415 │ Unsupported Media Type — Content-Type非対応 │
  │ 422 │ Unprocessable Content — バリデーションエラー │
  │     │ → JSONの構文は正しいがデータが不正           │
  │ 429 │ Too Many Requests — レート制限               │
  │     │ → Retry-After ヘッダーで再試行時間を通知     │
  │ 451 │ Unavailable For Legal Reasons — 法的理由    │
  └─────┴──────────────────────────────────────────────┘

  401 vs 403:
  → 401: 「誰ですか？」（認証が必要）
  → 403: 「あなたは入れません」（認証済みだが権限なし）
  → 未ログイン → 401
  → ログイン済み＋権限なし → 403
  → セキュリティ上、404 を返す場合もある
     （リソースの存在を隠す）

  400 vs 422:
  → 400: リクエストの構文が不正（JSON parse error 等）
  → 422: 構文は正しいが意味的に不正（バリデーション失敗）
  → 実務では 400 で統一する場合も多い

5xx サーバーエラー:
  ┌─────┬──────────────────────────────────────────────┐
  │ 500 │ Internal Server Error — 内部エラー            │
  │     │ → サーバー側の未処理例外、バグ                │
  │     │ → クライアントに詳細を返さない（セキュリティ）│
  │ 502 │ Bad Gateway — 上流サーバーエラー              │
  │     │ → プロキシ/ゲートウェイが上流サーバーから     │
  │     │   不正なレスポンスを受信                      │
  │ 503 │ Service Unavailable — サービス停止            │
  │     │ → メンテナンス中、過負荷                     │
  │     │ → Retry-After ヘッダーで復旧予定時刻を通知   │
  │ 504 │ Gateway Timeout — 上流タイムアウト            │
  │     │ → プロキシが上流サーバーの応答を待ちきれず   │
  └─────┴──────────────────────────────────────────────┘

  5xx エラーのリトライ戦略:
  → 500: リトライしても同じ結果になる可能性が高い
  → 502: 上流の一時的な問題の可能性 → リトライ推奨
  → 503: 一時的な過負荷 → バックオフ付きリトライ
  → 504: タイムアウト → リトライ推奨
  → リトライ時は指数バックオフ + ジッターを使用
```

---

## 5. HTTPヘッダー

```
ヘッダーの分類:

  ① リクエストヘッダー（クライアント → サーバー）
  ② レスポンスヘッダー（サーバー → クライアント）
  ③ 表現ヘッダー（リソースの表現に関する情報）
  ④ ペイロードヘッダー（ボディの情報）

主要なリクエストヘッダー:
  ┌───────────────────────┬────────────────────────────────┐
  │ ヘッダー              │ 説明                           │
  ├───────────────────────┼────────────────────────────────┤
  │ Host                  │ 接続先ホスト（HTTP/1.1で必須） │
  │ Accept                │ 受け入れ可能なメディアタイプ   │
  │ Accept-Charset        │ 受け入れ可能な文字セット       │
  │ Accept-Encoding       │ 受け入れ可能な圧縮形式         │
  │ Accept-Language       │ 希望言語                       │
  │ Authorization         │ 認証情報                       │
  │ Cookie                │ クッキー                       │
  │ Content-Type          │ ボディのメディアタイプ         │
  │ Content-Length        │ ボディのサイズ（バイト）       │
  │ User-Agent            │ クライアント情報               │
  │ Referer               │ 参照元URL                      │
  │ Origin                │ リクエスト元オリジン（CORS）   │
  │ If-None-Match         │ 条件付きリクエスト（ETag）     │
  │ If-Modified-Since     │ 条件付きリクエスト（日時）     │
  │ Range                 │ 部分的なリソース要求           │
  │ Cache-Control         │ キャッシュ制御（リクエスト側） │
  │ X-Forwarded-For       │ プロキシ経由時の元クライアントIP│
  │ X-Request-Id          │ リクエスト追跡用ID             │
  └───────────────────────┴────────────────────────────────┘

主要なレスポンスヘッダー:
  ┌───────────────────────────┬────────────────────────────────┐
  │ ヘッダー                  │ 説明                           │
  ├───────────────────────────┼────────────────────────────────┤
  │ Content-Type              │ ボディのメディアタイプ         │
  │ Content-Length            │ ボディのサイズ（バイト）       │
  │ Content-Encoding          │ ボディの圧縮形式               │
  │ Content-Disposition       │ ダウンロード時のファイル名     │
  │ Set-Cookie                │ クッキー設定                   │
  │ Cache-Control             │ キャッシュ制御                 │
  │ ETag                      │ リソースのバージョン           │
  │ Last-Modified             │ リソースの最終更新日時         │
  │ Location                  │ リダイレクト先/作成先URI       │
  │ WWW-Authenticate          │ 認証方式の指定（401時）        │
  │ Allow                     │ 対応メソッド一覧（405時）      │
  │ Retry-After               │ リトライ可能時間（429/503時）  │
  │ Access-Control-Allow-*    │ CORSヘッダー群                 │
  │ Strict-Transport-Security │ HSTS（HTTPS強制）              │
  │ X-Content-Type-Options    │ MIMEスニッフィング防止         │
  │ X-Frame-Options           │ クリックジャッキング防止       │
  │ Content-Security-Policy   │ CSP（XSS防止）                 │
  │ X-Request-Id              │ リクエスト追跡用（カスタム）   │
  └───────────────────────────┴────────────────────────────────┘

Content-Type の主要な値:
  ┌─────────────────────────────────────┬──────────────────────┐
  │ Content-Type                        │ 用途                 │
  ├─────────────────────────────────────┼──────────────────────┤
  │ application/json                    │ JSON                 │
  │ application/json; charset=utf-8     │ JSON（文字コード指定）│
  │ application/x-www-form-urlencoded   │ フォームデータ       │
  │ multipart/form-data                 │ ファイルアップロード │
  │ text/html; charset=utf-8            │ HTML                 │
  │ text/plain                          │ プレーンテキスト     │
  │ text/css                            │ CSS                  │
  │ text/javascript                     │ JavaScript           │
  │ application/javascript              │ JavaScript（推奨）   │
  │ application/xml                     │ XML                  │
  │ application/octet-stream            │ バイナリデータ       │
  │ image/png                           │ PNG画像              │
  │ image/jpeg                          │ JPEG画像             │
  │ image/svg+xml                       │ SVG                  │
  │ application/pdf                     │ PDF                  │
  │ application/zip                     │ ZIP                  │
  │ application/graphql+json            │ GraphQL              │
  │ application/problem+json            │ RFC 7807 エラー      │
  │ application/merge-patch+json        │ JSON Merge Patch     │
  │ application/json-patch+json         │ JSON Patch           │
  └─────────────────────────────────────┴──────────────────────┘

Accept ヘッダーのコンテントネゴシエーション:
  Accept: application/json, text/html;q=0.9, */*;q=0.8

  → q値（品質値）で優先度を指定
  → q=1.0 がデフォルト（最高優先度）
  → サーバーは最も適切な形式で返す

  Accept-Language: ja, en-US;q=0.9, en;q=0.8
  → 日本語優先、次に米国英語、最後に英語全般

  Accept-Encoding: gzip, deflate, br
  → gzip, deflate, Brotli の順で圧縮を受け入れ
```

---

## 6. HTTPレスポンス

```
レスポンス構造:

  ┌─────────────────────────────────────────┐
  │ HTTP/1.1 200 OK                         │ ← ステータスライン
  ├─────────────────────────────────────────┤
  │ Content-Type: application/json          │ ← ヘッダー
  │ Content-Length: 85                      │
  │ Cache-Control: no-cache                 │
  │ ETag: "abc123"                          │
  │ X-Request-Id: 550e8400-e29b-...        │
  │ Strict-Transport-Security: max-age=...  │
  ├─────────────────────────────────────────┤
  │                                         │ ← 空行
  ├─────────────────────────────────────────┤
  │ {                                       │ ← ボディ
  │   "id": "123",                          │
  │   "name": "Taro",                       │
  │   "email": "taro@example.com"           │
  │ }                                       │
  └─────────────────────────────────────────┘

ステータスラインの構造:
  HTTPバージョン SP ステータスコード SP 理由フレーズ CRLF

  HTTP/1.1 200 OK\r\n
  ↑        ↑   ↑
  バージョン コード 理由フレーズ

  注意: HTTP/2以降、理由フレーズは送信されない

チャンク転送エンコーディング（HTTP/1.1）:
  HTTP/1.1 200 OK
  Transfer-Encoding: chunked
  Content-Type: text/html

  4\r\n          ← チャンクサイズ（16進数）
  Wiki\r\n       ← チャンクデータ
  6\r\n
  pedia \r\n
  0\r\n          ← 終了チャンク（サイズ0）
  \r\n           ← トレーラー終了

  → Content-Length が不明な場合に使用
  → サーバーが生成中のデータを段階的に送信
  → HTTP/2 以降はフレーム単位でデータを送るため不要

圧縮レスポンス:
  HTTP/1.1 200 OK
  Content-Encoding: gzip
  Content-Type: application/json
  Vary: Accept-Encoding

  [gzip圧縮されたバイナリデータ]

  圧縮形式の比較:
  ┌──────────┬────────────┬────────────────────┐
  │ 形式     │ 圧縮率     │ 対応ブラウザ        │
  ├──────────┼────────────┼────────────────────┤
  │ gzip     │ 良好       │ 全ブラウザ          │
  │ deflate  │ 良好       │ 全ブラウザ          │
  │ br       │ 最も優秀   │ 主要ブラウザ全対応  │
  │ zstd     │ 優秀       │ 対応拡大中          │
  └──────────┴────────────┴────────────────────┘

  Brotli（br）の効果:
  → テキストデータで gzip より 15-25% 小さい
  → 圧縮速度は gzip よりやや遅い
  → HTTPS 必須（HTTP では使用不可）
  → 静的ファイルは事前圧縮で速度問題を回避
```

---

## 7. コネクション管理

```
HTTP/1.0:
  → 1リクエストごとにTCP接続を確立・切断
  → 非効率（毎回3-wayハンドシェイク）

  接続 ── リクエスト ── レスポンス ── 切断
  接続 ── リクエスト ── レスポンス ── 切断
  接続 ── リクエスト ── レスポンス ── 切断

HTTP/1.1 Keep-Alive:
  → 1つのTCP接続で複数リクエストを送信
  → Connection: keep-alive（HTTP/1.1ではデフォルト）
  → Connection: close で明示的に切断

  接続 ── リクエスト1 ── レスポンス1
       ── リクエスト2 ── レスポンス2
       ── リクエスト3 ── レスポンス3 ── 切断

  Keep-Alive の設定（サーバー側）:
  → タイムアウト: アイドル時間の上限
  → 最大リクエスト数: 1接続で処理する最大リクエスト数

  Nginx:
  keepalive_timeout 65;    # 65秒
  keepalive_requests 100;  # 最大100リクエスト

HTTP/1.1 パイプライン:
  → 前のレスポンスを待たずに次のリクエストを送信
  → レスポンスはリクエスト順に返す必要がある
  → Head-of-Line Blocking の問題
  → 実際にはほぼ使われていない（ブラウザも無効化）

  リクエスト1 ── リクエスト2 ── リクエスト3
  レスポンス1 ── レスポンス2 ── レスポンス3
  （レスポンス1が遅いと全て待ち）

HTTP/1.1 での同時接続:
  → ブラウザは同一ドメインに最大6接続（実装依存）
  → つまり最大6リクエストを並列処理
  → ドメインシャーディング: 複数ドメインで接続数を増やす
     → HTTP/2 では逆効果（1接続で多重化できるため）

HTTP/2 多重化:
  → 1つのTCP接続で複数のストリームを並列処理
  → Head-of-Line Blocking 解消（HTTP層では）
  → 接続数の削減 → リソース効率が向上

  ストリーム1: ── リクエスト ── レスポンス ──
  ストリーム2: ── リクエスト ──── レスポンス ──
  ストリーム3: ── リクエスト ── レスポンス ──
  （全て1つのTCP接続上で並列）
```

---

## 8. Cookie

```
Cookie の仕組み:

  サーバー → クライアント:
  Set-Cookie: session_id=abc123; Path=/; HttpOnly; Secure; SameSite=Lax

  クライアント → サーバー（以降のリクエスト）:
  Cookie: session_id=abc123

Set-Cookie の属性:
  ┌───────────────┬───────────────────────────────────────────┐
  │ 属性          │ 説明                                       │
  ├───────────────┼───────────────────────────────────────────┤
  │ Name=Value    │ Cookie名と値                               │
  │ Domain        │ 送信先ドメイン（サブドメイン含む）         │
  │ Path          │ 送信先パス                                 │
  │ Expires       │ 有効期限（日時指定）                       │
  │ Max-Age       │ 有効期限（秒数指定、Expiresより優先）      │
  │ HttpOnly      │ JavaScriptからアクセス不可（XSS対策）      │
  │ Secure        │ HTTPS接続時のみ送信                        │
  │ SameSite      │ クロスサイトリクエストでの送信制御         │
  │               │ Strict: 完全に送信しない                   │
  │               │ Lax: ナビゲーション時のみ送信（デフォルト）│
  │               │ None: 常に送信（Secure必須）               │
  │ Partitioned   │ CHIPS（サードパーティCookie分離）          │
  │ __Host-prefix │ Secure, Path=/, Domain未指定を強制         │
  │ __Secure-prefix│ Secure を強制                             │
  └───────────────┴───────────────────────────────────────────┘

Cookie のセキュリティ設定（推奨）:
  Set-Cookie: __Host-session=abc123;
    Path=/;
    Secure;
    HttpOnly;
    SameSite=Lax;
    Max-Age=3600

  → __Host- プレフィックス: ドメイン固定、Secure必須
  → HttpOnly: XSS攻撃でのCookie窃取を防止
  → Secure: HTTPS以外での送信を防止
  → SameSite=Lax: CSRF攻撃を防止
  → Max-Age: セッションの有効期限を設定

Cookie vs Token（JWT）:
  ┌────────────┬──────────────┬──────────────┐
  │            │ Cookie       │ JWT          │
  ├────────────┼──────────────┼──────────────┤
  │ 保存場所   │ ブラウザ自動 │ JS管理       │
  │ 送信方法   │ 自動         │ 手動         │
  │ CSRF対策   │ 必要         │ 不要         │
  │ XSS対策   │ HttpOnly     │ 困難         │
  │ サイズ     │ 4KB制限      │ 制限緩い     │
  │ サーバー状態│ セッション   │ ステートレス │
  │ ログアウト │ セッション破棄│ トークン無効化│
  │ クロスドメイン│ SameSite制限│ 自由       │
  └────────────┴──────────────┴──────────────┘
```

---

## 9. セキュリティヘッダー

```
推奨するセキュリティヘッダー:

  ① Strict-Transport-Security (HSTS):
     Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
     → HTTPS の強制（HTTP接続を自動でHTTPSにリダイレクト）
     → max-age: ブラウザが記憶する期間（1年）
     → includeSubDomains: サブドメインも対象
     → preload: ブラウザのプリロードリストに登録

  ② Content-Security-Policy (CSP):
     Content-Security-Policy:
       default-src 'self';
       script-src 'self' https://cdn.example.com;
       style-src 'self' 'unsafe-inline';
       img-src 'self' data: https:;
       connect-src 'self' https://api.example.com;
       font-src 'self' https://fonts.googleapis.com;
       frame-ancestors 'none';
     → XSS攻撃の防止
     → リソースの読み込み元を制限

  ③ X-Content-Type-Options:
     X-Content-Type-Options: nosniff
     → MIMEタイプスニッフィングを無効化
     → Content-Type を厳密に解釈

  ④ X-Frame-Options:
     X-Frame-Options: DENY
     → iframe での埋め込みを禁止
     → クリックジャッキング防止
     → CSP の frame-ancestors が後継

  ⑤ Referrer-Policy:
     Referrer-Policy: strict-origin-when-cross-origin
     → リファラー情報の送信を制御
     → 同一オリジン: フルURL
     → クロスオリジン: オリジンのみ

  ⑥ Permissions-Policy:
     Permissions-Policy:
       camera=(),
       microphone=(),
       geolocation=(self),
       payment=(self "https://pay.example.com")
     → ブラウザ機能の使用許可を制御

  ⑦ X-XSS-Protection（非推奨だが互換性のため）:
     X-XSS-Protection: 0
     → ブラウザのXSSフィルタを無効化
     → CSPが推奨される代替策
```

```typescript
// Express.js でのセキュリティヘッダー設定
import helmet from 'helmet';

app.use(helmet());

// カスタム設定
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", 'https://cdn.example.com'],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'", 'https://api.example.com'],
      fontSrc: ["'self'", 'https://fonts.googleapis.com'],
      frameAncestors: ["'none'"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
}));

// Nginx でのセキュリティヘッダー設定
// add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
// add_header X-Content-Type-Options "nosniff" always;
// add_header X-Frame-Options "DENY" always;
// add_header Referrer-Policy "strict-origin-when-cross-origin" always;
// add_header Content-Security-Policy "default-src 'self';" always;
// add_header Permissions-Policy "camera=(), microphone=(), geolocation=(self)" always;
```

---

## 10. HTTPデバッグ

```bash
# === curl でHTTPを確認 ===

# リクエストとレスポンスの詳細表示
curl -v https://api.example.com/users/123

# レスポンスヘッダーのみ
curl -I https://api.example.com/users/123

# POSTリクエスト
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-token" \
  -d '{"name": "Taro", "email": "taro@example.com"}'

# PATCHリクエスト
curl -X PATCH https://api.example.com/users/123 \
  -H "Content-Type: application/merge-patch+json" \
  -H "Authorization: Bearer my-token" \
  -d '{"name": "Updated Taro"}'

# DELETEリクエスト
curl -X DELETE https://api.example.com/users/123 \
  -H "Authorization: Bearer my-token"

# レスポンスボディ + HTTPステータス
curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health

# タイミング情報の表示
curl -s -o /dev/null -w "
DNS Lookup:    %{time_namelookup}s
TCP Connect:   %{time_connect}s
TLS Handshake: %{time_appconnect}s
TTFB:          %{time_starttransfer}s
Total Time:    %{time_total}s
" https://api.example.com/users

# リダイレクトを追跡
curl -L -v https://example.com/old-page

# ファイルアップロード
curl -X POST https://api.example.com/files \
  -H "Authorization: Bearer my-token" \
  -F "file=@/path/to/image.png" \
  -F "description=Profile image"

# Cookie を保存・送信
curl -c cookies.txt https://example.com/login
curl -b cookies.txt https://example.com/dashboard

# HTTP/2 で接続
curl --http2 -v https://api.example.com/users

# プロキシ経由
curl -x http://proxy.example.com:8080 https://api.example.com/users

# 条件付きリクエスト（ETag）
curl -H "If-None-Match: \"abc123\"" https://api.example.com/users/123

# HTTPie（curl の代替、より人間に優しい）
# GET
http GET https://api.example.com/users Authorization:"Bearer my-token"

# POST
http POST https://api.example.com/users \
  name=Taro email=taro@example.com \
  Authorization:"Bearer my-token"

# レスポンスのみ
http --body GET https://api.example.com/users
```

```typescript
// === ブラウザでのHTTPデバッグ ===

// Fetch API でのデバッグ
async function debugRequest(url: string): Promise<void> {
  console.time('Request Duration');

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  console.timeEnd('Request Duration');

  // ステータス情報
  console.log('Status:', response.status, response.statusText);
  console.log('OK:', response.ok);
  console.log('Redirected:', response.redirected);
  console.log('Type:', response.type);
  console.log('URL:', response.url);

  // レスポンスヘッダー
  console.log('Headers:');
  response.headers.forEach((value, key) => {
    console.log(`  ${key}: ${value}`);
  });

  // ボディ
  const data = await response.json();
  console.log('Body:', data);
}

// Performance API での計測
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.entryType === 'resource') {
      const resource = entry as PerformanceResourceTiming;
      console.log({
        name: resource.name,
        duration: `${resource.duration.toFixed(2)}ms`,
        dns: `${(resource.domainLookupEnd - resource.domainLookupStart).toFixed(2)}ms`,
        tcp: `${(resource.connectEnd - resource.connectStart).toFixed(2)}ms`,
        ttfb: `${(resource.responseStart - resource.requestStart).toFixed(2)}ms`,
        download: `${(resource.responseEnd - resource.responseStart).toFixed(2)}ms`,
        size: resource.transferSize,
        protocol: resource.nextHopProtocol,  // "h2", "h3"
      });
    }
  }
});
observer.observe({ type: 'resource', buffered: true });
```

---

## 11. HTTPSとTLS

```
HTTPS = HTTP + TLS（Transport Layer Security）

  HTTP:   http://example.com（ポート80）
  HTTPS:  https://example.com（ポート443）

TLSハンドシェイク（TLS 1.3）:
  クライアント                 サーバー
  │── ClientHello ──→        │
  │   (暗号スイート候補、     │
  │    鍵共有パラメータ)      │
  │                           │
  │←── ServerHello ───       │
  │   (選択した暗号スイート、 │
  │    鍵共有パラメータ、     │
  │    証明書、Finished)      │
  │                           │
  │── Finished ──→           │
  │                           │
  │←→ 暗号化通信開始 ←→      │

  TLS 1.2: 2 RTT で接続確立
  TLS 1.3: 1 RTT で接続確立（0-RTT再接続も可能）

証明書の種類:
  ┌────────┬──────────────────────────────────┐
  │ 種類   │ 説明                              │
  ├────────┼──────────────────────────────────┤
  │ DV     │ ドメイン検証（自動取得可能）       │
  │ OV     │ 組織検証（企業の実在確認）         │
  │ EV     │ 拡張検証（厳格な審査）             │
  └────────┴──────────────────────────────────┘

  Let's Encrypt: 無料のDV証明書（自動更新可能）

HTTPSが必要な理由:
  ① 盗聴防止: 通信内容の暗号化
  ② 改ざん防止: データの整合性保証
  ③ なりすまし防止: サーバーの認証
  ④ SEO: Google は HTTPS を優遇
  ⑤ HTTP/2: 事実上 HTTPS が必須
  ⑥ Brotli圧縮: HTTPS のみで使用可能
  ⑦ Service Worker: HTTPS のみで動作
  ⑧ Geolocation API 等: HTTPS のみで動作

混在コンテンツ（Mixed Content）:
  → HTTPS ページ内の HTTP リソース
  → ブラウザがブロックまたは警告
  → 画像等の「パッシブ」混在: 警告のみ
  → スクリプト等の「アクティブ」混在: ブロック
  → 解決: 全リソースを HTTPS に統一
```

---

## 12. URL/URIの構造

```
URI（Uniform Resource Identifier）の構造:

  https://user:pass@api.example.com:443/v1/users?page=1&sort=name#section1
  ├──┤   ├───────┤ ├───────────────┤├──┤├────────┤├──────────────┤├───────┤
  scheme authority   host           port path      query           fragment

  各部分の説明:
  ┌───────────┬─────────────────────────────────────────┐
  │ 部分      │ 説明                                     │
  ├───────────┼─────────────────────────────────────────┤
  │ scheme    │ プロトコル（http, https, ftp等）          │
  │ userinfo  │ 認証情報（非推奨、セキュリティリスク）   │
  │ host      │ ホスト名またはIPアドレス                 │
  │ port      │ ポート番号（省略時はデフォルト）         │
  │ path      │ リソースのパス                           │
  │ query     │ クエリパラメータ（?key=value&...）       │
  │ fragment  │ ページ内の位置（#section）               │
  └───────────┴─────────────────────────────────────────┘

  URI vs URL vs URN:
  → URI: リソースの識別子（上位概念）
  → URL: リソースの場所を示す（https://example.com/page）
  → URN: リソースの名前を示す（urn:isbn:0451450523）
  → 実務では URI と URL はほぼ同義で使われる

URLエンコーディング:
  → URIで使用できない文字をパーセントエンコード
  → スペース → %20（またはクエリ内では +）
  → 日本語 → %E6%97%A5%E6%9C%AC（UTF-8バイト列を16進数）

  安全な文字（エンコード不要）:
  A-Z, a-z, 0-9, - _ . ~

  予約文字（用途に応じてエンコード）:
  : / ? # [ ] @ ! $ & ' ( ) * + , ; =
```

```typescript
// URL操作（JavaScript/TypeScript）

// URL オブジェクト
const url = new URL('https://api.example.com/v1/users?page=1&sort=name');

console.log(url.protocol);   // "https:"
console.log(url.hostname);   // "api.example.com"
console.log(url.port);       // "" (デフォルトポート)
console.log(url.pathname);   // "/v1/users"
console.log(url.search);     // "?page=1&sort=name"
console.log(url.hash);       // ""
console.log(url.origin);     // "https://api.example.com"

// クエリパラメータ操作
const params = url.searchParams;
console.log(params.get('page'));    // "1"
console.log(params.get('sort'));    // "name"
params.set('page', '2');
params.append('filter', 'active');
console.log(url.toString());
// "https://api.example.com/v1/users?page=2&sort=name&filter=active"

// URLSearchParams の活用
const searchParams = new URLSearchParams({
  page: '1',
  per_page: '20',
  sort: '-created_at',
  status: 'active',
});
const apiUrl = `https://api.example.com/users?${searchParams}`;
// "https://api.example.com/users?page=1&per_page=20&sort=-created_at&status=active"

// エンコーディング
encodeURIComponent('Hello World');    // "Hello%20World"
encodeURIComponent('名前=太郎');      // "%E5%90%8D%E5%89%8D%3D%E5%A4%AA%E9%83%8E"
decodeURIComponent('%E5%90%8D%E5%89%8D');  // "名前"

// encodeURI vs encodeURIComponent
encodeURI('https://example.com/path?q=hello world');
// "https://example.com/path?q=hello%20world"
encodeURIComponent('https://example.com/path?q=hello world');
// "https%3A%2F%2Fexample.com%2Fpath%3Fq%3Dhello%20world"
// → encodeURIComponent は予約文字もエンコード
```

---

## 13. サーバー実装例

```typescript
// Express.js でのHTTPサーバー実装

import express from 'express';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import { v4 as uuidv4 } from 'uuid';

const app = express();

// === ミドルウェア ===

// セキュリティヘッダー
app.use(helmet());

// レスポンス圧縮
app.use(compression({
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  },
  level: 6,  // 圧縮レベル（1-9）
  threshold: 1024,  // 1KB未満は圧縮しない
}));

// リクエストID付与
app.use((req, res, next) => {
  const requestId = req.headers['x-request-id'] as string || uuidv4();
  req.headers['x-request-id'] = requestId;
  res.setHeader('X-Request-Id', requestId);
  next();
});

// アクセスログ
app.use(morgan(':method :url :status :res[content-length] - :response-time ms'));

// JSONボディパース
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// === ルーティング ===

// GET — 一覧取得
app.get('/api/users', async (req, res) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const perPage = Math.min(parseInt(req.query.per_page as string) || 20, 100);
    const sort = req.query.sort as string || 'created_at';

    const { users, total } = await getUsersList(page, perPage, sort);

    res.status(200).json({
      data: users,
      meta: {
        total,
        page,
        per_page: perPage,
        total_pages: Math.ceil(total / perPage),
      },
    });
  } catch (error) {
    handleError(res, error);
  }
});

// GET — 詳細取得（条件付きリクエスト対応）
app.get('/api/users/:id', async (req, res) => {
  try {
    const user = await getUserById(req.params.id);

    if (!user) {
      return res.status(404).json({
        type: 'https://api.example.com/errors/not-found',
        title: 'Not Found',
        status: 404,
        detail: `User ${req.params.id} not found`,
      });
    }

    // ETag生成
    const etag = generateETag(user);
    res.setHeader('ETag', etag);
    res.setHeader('Cache-Control', 'private, no-cache');
    res.setHeader('Last-Modified', user.updated_at.toUTCString());

    // 条件付きリクエストチェック
    if (req.headers['if-none-match'] === etag) {
      return res.status(304).end();
    }

    const ifModifiedSince = req.headers['if-modified-since'];
    if (ifModifiedSince && new Date(ifModifiedSince) >= user.updated_at) {
      return res.status(304).end();
    }

    res.status(200).json({ data: user });
  } catch (error) {
    handleError(res, error);
  }
});

// POST — リソース作成
app.post('/api/users', async (req, res) => {
  try {
    // バリデーション
    const errors = validateCreateUser(req.body);
    if (errors.length > 0) {
      return res.status(422).json({
        type: 'https://api.example.com/errors/validation',
        title: 'Validation Error',
        status: 422,
        detail: 'The request body contains invalid fields.',
        errors,
      });
    }

    const user = await createUser(req.body);

    res
      .status(201)
      .setHeader('Location', `/api/users/${user.id}`)
      .json({ data: user });
  } catch (error) {
    if (error instanceof DuplicateError) {
      return res.status(409).json({
        type: 'https://api.example.com/errors/conflict',
        title: 'Conflict',
        status: 409,
        detail: error.message,
      });
    }
    handleError(res, error);
  }
});

// PATCH — 部分更新（楽観的ロック付き）
app.patch('/api/users/:id', async (req, res) => {
  try {
    const user = await getUserById(req.params.id);

    if (!user) {
      return res.status(404).json({
        type: 'https://api.example.com/errors/not-found',
        title: 'Not Found',
        status: 404,
        detail: `User ${req.params.id} not found`,
      });
    }

    // 楽観的ロック: If-Match ヘッダーでETagを検証
    const ifMatch = req.headers['if-match'];
    if (ifMatch && ifMatch !== generateETag(user)) {
      return res.status(412).json({
        type: 'https://api.example.com/errors/precondition-failed',
        title: 'Precondition Failed',
        status: 412,
        detail: 'Resource has been modified since last retrieval.',
      });
    }

    const updatedUser = await updateUser(req.params.id, req.body);
    const newETag = generateETag(updatedUser);

    res
      .setHeader('ETag', newETag)
      .status(200)
      .json({ data: updatedUser });
  } catch (error) {
    handleError(res, error);
  }
});

// DELETE — リソース削除
app.delete('/api/users/:id', async (req, res) => {
  try {
    const deleted = await deleteUser(req.params.id);

    if (!deleted) {
      // 冪等性: 既に削除済みでも 204 を返す
      return res.status(204).end();
    }

    res.status(204).end();
  } catch (error) {
    handleError(res, error);
  }
});

// HEAD — 存在確認（GETと同じロジック、ボディなし）
app.head('/api/users/:id', async (req, res) => {
  try {
    const user = await getUserById(req.params.id);

    if (!user) {
      return res.status(404).end();
    }

    res.setHeader('ETag', generateETag(user));
    res.setHeader('Content-Type', 'application/json');
    res.status(200).end();
  } catch (error) {
    res.status(500).end();
  }
});

// OPTIONS — CORS対応は cors ミドルウェアが自動処理

// ヘルスチェック
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// エラーハンドリング
function handleError(res: express.Response, error: unknown): void {
  console.error('Internal error:', error);

  res.status(500).json({
    type: 'https://api.example.com/errors/internal',
    title: 'Internal Server Error',
    status: 500,
    detail: 'An unexpected error occurred.',
  });
}

// サーバー起動
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

---

## 14. Fetch APIとHTTPクライアント

```typescript
// === Fetch API の詳細 ===

// 基本的なオプション
const response = await fetch(url, {
  method: 'GET',                    // HTTPメソッド
  headers: new Headers({            // ヘッダー
    'Content-Type': 'application/json',
    'Authorization': 'Bearer token',
  }),
  body: JSON.stringify(data),       // ボディ（GET/HEADでは不可）
  mode: 'cors',                     // CORS モード
  credentials: 'include',           // Cookie送信
  cache: 'no-cache',                // キャッシュ制御
  redirect: 'follow',               // リダイレクト制御
  referrerPolicy: 'strict-origin-when-cross-origin',
  signal: AbortSignal.timeout(5000), // タイムアウト（5秒）
  keepalive: true,                   // ページ離脱後も送信
  priority: 'high',                  // 優先度（high/low/auto）
});

// === レスポンスの読み取り ===
// response.json()   — JSON
// response.text()   — テキスト
// response.blob()   — バイナリ（Blob）
// response.arrayBuffer() — バイナリ（ArrayBuffer）
// response.formData() — FormData
// response.body     — ReadableStream（ストリーミング）

// ストリーミングレスポンスの読み取り
async function streamResponse(url: string): Promise<string> {
  const response = await fetch(url);
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let result = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    result += decoder.decode(value, { stream: true });
    console.log('Received chunk:', decoder.decode(value));
  }

  return result;
}

// タイムアウトとキャンセル
async function fetchWithTimeout(
  url: string,
  timeoutMs: number = 5000,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// リトライ付きfetch
async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries: number = 3,
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      // 5xx エラーはリトライ
      if (response.status >= 500 && attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;  // 指数バックオフ
        const jitter = Math.random() * 1000;          // ジッター
        await new Promise(resolve => setTimeout(resolve, delay + jitter));
        continue;
      }

      return response;
    } catch (error) {
      lastError = error as Error;

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error('All retries failed');
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| HTTP | ステートレスなリクエスト/レスポンスプロトコル |
| メソッド | GET(取得), POST(作成), PUT(更新), PATCH(部分更新), DELETE(削除) |
| 冪等性 | GET, PUT, DELETE は冪等、POST, PATCH は非冪等 |
| ステータス | 2xx(成功), 3xx(リダイレクト), 4xx(クライアントエラー), 5xx(サーバーエラー) |
| ヘッダー | Content-Type, Authorization, Cache-Control, ETag 等 |
| Cookie | HttpOnly, Secure, SameSite でセキュア設定 |
| HTTPS | TLS による暗号化・認証・改ざん防止 |
| セキュリティ | HSTS, CSP, X-Content-Type-Options 等 |
| デバッグ | curl, HTTPie, ブラウザDevTools |

---

## 次に読むべきガイド
→ [[01-http2-and-http3.md]] — HTTP/2とHTTP/3

---

## 参考文献
1. RFC 9110. "HTTP Semantics." IETF, 2022.
2. RFC 9111. "HTTP Caching." IETF, 2022.
3. RFC 9112. "HTTP/1.1." IETF, 2022.
4. MDN Web Docs. "HTTP." Mozilla, 2024.
5. OWASP. "HTTP Security Response Headers." OWASP, 2024.
6. web.dev. "Fetch API." Google, 2024.
7. RFC 6265bis. "Cookies: HTTP State Management Mechanism." IETF, 2024.
8. RFC 8446. "The Transport Layer Security (TLS) Protocol Version 1.3." IETF, 2018.
