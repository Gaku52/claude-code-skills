# APIエラー設計

> APIのエラーレスポンスは、クライアント開発者の体験を左右する。HTTPステータスの正しい使い方、RFC 7807 Problem Details、エラーレスポンス設計のベストプラクティスを解説。

## この章で学ぶこと

- [ ] HTTPステータスコードの適切な使い分けを理解する
- [ ] エラーレスポンスの標準フォーマットを把握する
- [ ] 実践的なAPIエラー設計を学ぶ

---

## 1. HTTPステータスコード

```
2xx 成功:
  200 OK               — 汎用的な成功
  201 Created           — リソース作成成功
  204 No Content        — 成功（レスポンスボディなし）

4xx クライアントエラー:
  400 Bad Request       — リクエストが不正
  401 Unauthorized      — 認証が必要
  403 Forbidden         — 認可されていない
  404 Not Found         — リソースが存在しない
  409 Conflict          — 競合（重複登録など）
  422 Unprocessable Entity — バリデーションエラー
  429 Too Many Requests — レート制限

5xx サーバーエラー:
  500 Internal Server Error — サーバー内部エラー
  502 Bad Gateway           — 上流サーバーエラー
  503 Service Unavailable   — サービス一時停止
  504 Gateway Timeout       — 上流サーバータイムアウト

判断基準:
  クライアントのミス → 4xx
  サーバーの問題 → 5xx
  リトライで解決する可能性 → 429, 503, 504
```

---

## 2. エラーレスポンスフォーマット

```json
// RFC 7807 Problem Details（推奨）
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation Error",
  "status": 422,
  "detail": "入力値に問題があります",
  "instance": "/api/users",
  "errors": [
    {
      "field": "email",
      "message": "有効なメールアドレスを入力してください"
    },
    {
      "field": "password",
      "message": "8文字以上で入力してください"
    }
  ],
  "traceId": "abc-123-def"
}
```

```typescript
// エラーレスポンスの型定義
interface ApiError {
  type: string;          // エラーの種類（URL or コード）
  title: string;         // 人間可読なタイトル
  status: number;        // HTTPステータス
  detail: string;        // 詳細メッセージ
  instance?: string;     // リクエストパス
  traceId?: string;      // トレーシングID
  errors?: FieldError[]; // フィールドレベルのエラー
}

interface FieldError {
  field: string;
  message: string;
  code?: string;
}

// Express ミドルウェア
function errorHandler(err: Error, req: Request, res: Response, next: NextFunction) {
  if (err instanceof AppError) {
    const response: ApiError = {
      type: `https://api.example.com/errors/${err.code.toLowerCase()}`,
      title: err.code,
      status: err.statusCode,
      detail: err.message,
      instance: req.path,
      traceId: req.headers['x-trace-id'] as string,
    };

    if (err instanceof ValidationError) {
      response.errors = err.fields;
    }

    res.status(err.statusCode).json(response);
  } else {
    // 予期しないエラー（内部詳細を隠す）
    res.status(500).json({
      type: "https://api.example.com/errors/internal",
      title: "Internal Server Error",
      status: 500,
      detail: "サーバーエラーが発生しました",
      traceId: req.headers['x-trace-id'] as string,
    });
  }
}
```

---

## 3. エラー設計のベストプラクティス

```
1. 一貫性
   → 全エンドポイントで同じエラーフォーマット
   → ステータスコードの使い方を統一

2. セキュリティ
   → 500エラーで内部情報を漏らさない
   → スタックトレースは本番では非表示
   → 「ユーザーが存在しない」vs「パスワードが違う」を区別しない

3. 機械可読性
   → エラーコードは文字列（enum対応）
   → HTTPステータスとエラーコードの組み合わせ

4. 人間可読性
   → detail フィールドで具体的なメッセージ
   → フィールドレベルのバリデーションエラー

5. リトライ可能性の明示
   → 429: Retry-After ヘッダー
   → 503: Retry-After ヘッダー
```

---

## まとめ

| 原則 | ポイント |
|------|---------|
| ステータスコード | 正しいコードを選ぶ |
| フォーマット | RFC 7807 準拠 |
| セキュリティ | 内部情報を漏らさない |
| 一貫性 | 全エンドポイントで統一 |
| リトライ | Retry-After ヘッダー |

---

## 次に読むべきガイド
→ [[01-logging-and-monitoring.md]] — ログとモニタリング

---

## 参考文献
1. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
2. Fielding, R. "REST APIs must be hypertext-driven." 2008.
