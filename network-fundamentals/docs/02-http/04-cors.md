# CORS（Cross-Origin Resource Sharing）

> CORSはブラウザのセキュリティ機構「同一オリジンポリシー」を安全に緩和する仕組み。プリフライトリクエスト、許可ヘッダー、Credentialsの設定を理解し、正しくCORSを構成する。

## この章で学ぶこと

- [ ] 同一オリジンポリシーとCORSの関係を理解する
- [ ] シンプルリクエストとプリフライトの違いを把握する
- [ ] サーバー側のCORS設定方法を学ぶ

---

## 1. 同一オリジンポリシー

```
オリジン = スキーム + ホスト + ポート

  https://example.com:443/path
  ↑        ↑            ↑
  スキーム   ホスト       ポート

  同一オリジンの判定:
  https://example.com/a  と https://example.com/b  → 同一 ✓
  https://example.com    と http://example.com      → 異なる ✗（スキーム）
  https://example.com    と https://api.example.com → 異なる ✗（ホスト）
  https://example.com    と https://example.com:8080→ 異なる ✗（ポート）

同一オリジンポリシー:
  → ブラウザが異なるオリジンへのアクセスを制限
  → XSS等の攻撃を防止
  → fetch/XMLHttpRequest が対象
  → <img>, <script>, <link> は対象外（歴史的理由）

  フロントエンド: https://app.example.com
  バックエンド:  https://api.example.com
  → オリジンが異なる → CORS が必要
```

---

## 2. CORSの仕組み

```
CORSフロー:

  ① シンプルリクエスト（プリフライト不要）:
     条件:
     - メソッド: GET, HEAD, POST のみ
     - ヘッダー: Accept, Content-Type(制限あり) 等の安全なもののみ
     - Content-Type: text/plain, multipart/form-data,
                     application/x-www-form-urlencoded のみ

     ブラウザ                          サーバー
     │── GET /api/data ──→            │
     │   Origin: https://app.example.com│
     │                                  │
     │←── 200 OK ────────             │
     │   Access-Control-Allow-Origin:   │
     │     https://app.example.com      │

  ② プリフライトリクエスト（OPTIONS）:
     条件: シンプルリクエストの条件を満たさない場合
     - PUT, DELETE, PATCH メソッド
     - Content-Type: application/json
     - カスタムヘッダー（Authorization等）

     ブラウザ                          サーバー
     │── OPTIONS /api/data ──→        │  ← プリフライト
     │   Origin: https://app.example.com│
     │   Access-Control-Request-Method: PUT│
     │   Access-Control-Request-Headers: │
     │     Content-Type, Authorization  │
     │                                  │
     │←── 204 No Content ──────       │  ← 許可
     │   Access-Control-Allow-Origin:   │
     │     https://app.example.com      │
     │   Access-Control-Allow-Methods:  │
     │     GET, POST, PUT, DELETE       │
     │   Access-Control-Allow-Headers:  │
     │     Content-Type, Authorization  │
     │   Access-Control-Max-Age: 86400  │
     │                                  │
     │── PUT /api/data ──→             │  ← 実際のリクエスト
     │   Origin: https://app.example.com│
     │   Authorization: Bearer xxx      │
     │                                  │
     │←── 200 OK ────────             │
```

---

## 3. CORSヘッダー

```
レスポンスヘッダー:
  ┌──────────────────────────────┬──────────────────────────┐
  │ ヘッダー                     │ 説明                     │
  ├──────────────────────────────┼──────────────────────────┤
  │ Access-Control-Allow-Origin  │ 許可するオリジン         │
  │                              │ * または 特定オリジン     │
  ├──────────────────────────────┼──────────────────────────┤
  │ Access-Control-Allow-Methods │ 許可するHTTPメソッド     │
  │                              │ GET, POST, PUT, DELETE   │
  ├──────────────────────────────┼──────────────────────────┤
  │ Access-Control-Allow-Headers │ 許可するヘッダー         │
  │                              │ Content-Type, Authorization│
  ├──────────────────────────────┼──────────────────────────┤
  │ Access-Control-Max-Age       │ プリフライト結果の       │
  │                              │ キャッシュ秒数           │
  ├──────────────────────────────┼──────────────────────────┤
  │ Access-Control-Allow-        │ Credentialsの許可        │
  │ Credentials                  │ true または 省略         │
  ├──────────────────────────────┼──────────────────────────┤
  │ Access-Control-Expose-Headers│ JSから読み取り可能な     │
  │                              │ レスポンスヘッダー       │
  └──────────────────────────────┴──────────────────────────┘

  重要な制約:
  Allow-Origin: * と Allow-Credentials: true は併用不可
  → Credentials使用時は特定のオリジンを指定する必要がある
```

---

## 4. サーバー側の設定

```typescript
// Express.js での CORS 設定

// 方法1: cors ミドルウェア（推奨）
import cors from 'cors';

app.use(cors({
  origin: ['https://app.example.com', 'https://admin.example.com'],
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  maxAge: 86400,
}));

// 方法2: 手動設定
app.use((req, res, next) => {
  const allowedOrigins = ['https://app.example.com'];
  const origin = req.headers.origin;

  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }

  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Max-Age', '86400');

  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }

  next();
});
```

```
Nginx での CORS 設定:

  location /api/ {
      if ($request_method = 'OPTIONS') {
          add_header 'Access-Control-Allow-Origin' 'https://app.example.com';
          add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE';
          add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization';
          add_header 'Access-Control-Max-Age' 86400;
          return 204;
      }

      add_header 'Access-Control-Allow-Origin' 'https://app.example.com';
      add_header 'Access-Control-Allow-Credentials' 'true';
      proxy_pass http://backend;
  }
```

---

## 5. よくあるCORSエラーと対処

```
エラー1: No 'Access-Control-Allow-Origin' header
  原因: サーバーがCORSヘッダーを返していない
  対処: サーバーに Access-Control-Allow-Origin を設定

エラー2: Credentials flag が true だが Allow-Origin が *
  原因: credentials: 'include' 使用時に * は不可
  対処: 具体的なオリジンを指定

エラー3: プリフライトが失敗
  原因: OPTIONS リクエストに正しくレスポンスしていない
  対処: OPTIONS ハンドラーを追加

エラー4: 開発環境でのCORS
  → localhost:3000（フロント）→ localhost:8080（API）
  対処:
    ① Vite/Webpack のプロキシ設定（推奨）
    ② 開発用CORS設定（origin: 'http://localhost:3000'）

  Vite のプロキシ設定:
  // vite.config.ts
  export default defineConfig({
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8080',
          changeOrigin: true,
        },
      },
    },
  });
  → /api/users → http://localhost:8080/api/users
  → 同一オリジンになるのでCORS不要

セキュリティ注意:
  ✗ Access-Control-Allow-Origin: *（本番環境で使わない）
  ✗ リクエストの Origin をそのまま返す（検証なし）
  ✓ ホワイトリストで許可するオリジンを明示
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 同一オリジンポリシー | スキーム+ホスト+ポートが一致 |
| シンプルリクエスト | GET/HEAD/POST、制限付きヘッダー |
| プリフライト | OPTIONS で事前確認（PUT/DELETE等） |
| Credentials | Cookie送信時は * 不可、オリジン明示 |
| 開発環境 | プロキシ設定で回避が最もクリーン |

---

## 次に読むべきガイド
→ [[../03-security/00-tls-ssl.md]] — TLS/SSL

---

## 参考文献
1. Fetch Standard. "CORS Protocol." WHATWG, 2024.
2. MDN Web Docs. "Cross-Origin Resource Sharing (CORS)." Mozilla, 2024.
