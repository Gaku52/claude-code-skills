# エラー境界

> エラー境界は「エラーの影響を局所化する」仕組み。React Error Boundary、グローバルエラーハンドラ、プロセスレベルのエラー処理を理解する。

## この章で学ぶこと

- [ ] エラー境界の概念とレイヤー設計を理解する
- [ ] React Error Boundary の実装を把握する
- [ ] グローバルエラーハンドラの設計を学ぶ

---

## 1. エラー境界のレイヤー

```
エラーは発生源に近い場所で処理するのが原則。
ただし、処理できない場合は上位のレイヤーで捕捉する。

  Layer 4: プロセス/アプリレベル
    → 未捕捉例外のキャッチ
    → エラーレポーティング（Sentry）
    → グレースフルシャットダウン

  Layer 3: ミドルウェア/フレームワーク
    → HTTPエラーレスポンスの統一
    → ログ出力

  Layer 2: サービス/ユースケース
    → ビジネスロジックのエラー処理
    → リトライ、フォールバック

  Layer 1: 関数/メソッド
    → 入力バリデーション
    → 個別のtry/catch
```

---

## 2. React Error Boundary

```tsx
// React: Error Boundary（クラスコンポーネントが必要）
import React, { Component, ReactNode } from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // エラーレポーティングサービスに送信
    console.error('Error Boundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div>
          <h2>エラーが発生しました</h2>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            再試行
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// 使い方: エラーの影響を局所化
function App() {
  return (
    <div>
      <Header /> {/* ヘッダーは常に表示 */}
      <ErrorBoundary fallback={<p>サイドバーの読み込みに失敗</p>}>
        <Sidebar /> {/* サイドバーのエラーは他に影響しない */}
      </ErrorBoundary>
      <ErrorBoundary fallback={<p>メインコンテンツの読み込みに失敗</p>}>
        <MainContent /> {/* メインのエラーも局所化 */}
      </ErrorBoundary>
    </div>
  );
}
```

---

## 3. サーバーサイドのエラー境界

```typescript
// Express: グローバルエラーミドルウェア
import express, { Request, Response, NextFunction } from 'express';

const app = express();

// ルートハンドラ
app.get('/api/users/:id', async (req, res, next) => {
  try {
    const user = await userService.getUser(req.params.id);
    res.json(user);
  } catch (error) {
    next(error); // エラーミドルウェアに委譲
  }
});

// エラー境界: グローバルエラーハンドラ
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
  // エラーの種類に応じてレスポンスを分岐
  if (error instanceof ValidationError) {
    res.status(400).json({
      type: "validation_error",
      message: error.message,
      fields: error.fields,
    });
  } else if (error instanceof NotFoundError) {
    res.status(404).json({
      type: "not_found",
      message: error.message,
    });
  } else if (error instanceof AuthError) {
    res.status(401).json({
      type: "unauthorized",
      message: "認証が必要です",
    });
  } else {
    // 予期しないエラー
    console.error("Unexpected error:", error);
    // Sentry に送信
    res.status(500).json({
      type: "internal_error",
      message: "サーバーエラーが発生しました",
    });
  }
});
```

---

## 4. プロセスレベルのエラー処理

```typescript
// Node.js: 未捕捉例外とunhandled rejection
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // エラーレポーティング
  // グレースフルシャットダウン
  process.exit(1); // 必ず終了
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  // Node.js 15+ では uncaughtException と同様に終了
});

// グレースフルシャットダウン
process.on('SIGTERM', async () => {
  console.log('SIGTERM received. Graceful shutdown...');
  await server.close();
  await db.disconnect();
  process.exit(0);
});
```

---

## 5. エラー境界の設計原則

```
1. 段階的なフォールバック
   → コンポーネントレベル → ページレベル → アプリレベル

2. ユーザーへの情報提供
   → 何が起きたか、何ができるかを伝える
   → 技術的な詳細は隠す（セキュリティ）

3. エラーのログとレポーティング
   → 全てのレイヤーでログを出力
   → 本番環境では Sentry 等に送信

4. リカバリー手段の提供
   → 再試行ボタン
   → 別の操作への誘導
   → 最終手段: ページリロード
```

---

## まとめ

| レイヤー | 手法 | 目的 |
|---------|------|------|
| コンポーネント | Error Boundary | UI の部分的エラー |
| ミドルウェア | エラーハンドラ | HTTPレスポンス統一 |
| プロセス | uncaughtException | 最後の砦 |
| 外部サービス | Sentry | モニタリング |

---

## 次に読むべきガイド
→ [[03-custom-errors.md]] — カスタムエラー

---

## 参考文献
1. React Documentation. "Error Boundaries."
2. Express.js Documentation. "Error Handling."
