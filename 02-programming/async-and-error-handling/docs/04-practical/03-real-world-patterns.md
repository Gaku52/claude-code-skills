# 実践パターン集

> 実際のプロジェクトでよく使われる非同期 + エラーハンドリングのパターンを集約。キュー処理、WebSocket、ファイルアップロード、バッチ処理など。

## この章で学ぶこと

- [ ] 実践的な非同期パターンを習得する
- [ ] エラーハンドリングの実装例を把握する
- [ ] プロダクションレベルのコードパターンを学ぶ

---

## 1. ジョブキュー処理

```typescript
// ジョブキュー: 信頼性の高い非同期処理
interface Job<T = unknown> {
  id: string;
  type: string;
  data: T;
  attempts: number;
  maxAttempts: number;
  createdAt: Date;
}

class JobProcessor {
  private handlers = new Map<string, (data: any) => Promise<void>>();

  register(type: string, handler: (data: any) => Promise<void>): void {
    this.handlers.set(type, handler);
  }

  async process(job: Job): Promise<void> {
    const handler = this.handlers.get(job.type);
    if (!handler) throw new Error(`No handler for job type: ${job.type}`);

    try {
      await handler(job.data);
      await this.markCompleted(job);
    } catch (error) {
      job.attempts++;
      if (job.attempts < job.maxAttempts) {
        // リトライキューに戻す（指数バックオフ）
        const delay = Math.pow(2, job.attempts) * 1000;
        await this.scheduleRetry(job, delay);
      } else {
        // 最大リトライ超過 → デッドレターキューに移動
        await this.moveToDeadLetter(job, error as Error);
      }
    }
  }
}

// 登録
const processor = new JobProcessor();
processor.register('send-email', async (data) => {
  await emailService.send(data.to, data.subject, data.body);
});
processor.register('process-payment', async (data) => {
  await paymentService.charge(data.userId, data.amount);
});
```

---

## 2. WebSocket のエラーハンドリング

```typescript
// WebSocket: 自動再接続パターン
class ResilientWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  constructor(
    private url: string,
    private onMessage: (data: any) => void,
  ) {}

  connect(): void {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0; // リセット
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessage(data);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };

    this.ws.onclose = (event) => {
      if (!event.wasClean) {
        this.reconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private reconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      throw new Error('WebSocket is not connected');
    }
  }

  disconnect(): void {
    this.maxReconnectAttempts = 0; // 再接続を無効化
    this.ws?.close();
  }
}
```

---

## 3. ファイルアップロード

```typescript
// チャンクアップロード: 大きなファイルの信頼性のあるアップロード
async function uploadFileInChunks(
  file: File,
  chunkSize: number = 5 * 1024 * 1024, // 5MB
  onProgress?: (progress: number) => void,
): Promise<string> {
  const totalChunks = Math.ceil(file.size / chunkSize);
  const uploadId = crypto.randomUUID();

  for (let i = 0; i < totalChunks; i++) {
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize, file.size);
    const chunk = file.slice(start, end);

    // リトライ付きでチャンクをアップロード
    await retryWithBackoff(
      async () => {
        const formData = new FormData();
        formData.append('chunk', chunk);
        formData.append('uploadId', uploadId);
        formData.append('chunkIndex', String(i));
        formData.append('totalChunks', String(totalChunks));

        const response = await fetch('/api/upload/chunk', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
      },
      { maxRetries: 3 },
    );

    onProgress?.((i + 1) / totalChunks);
  }

  // 完了通知
  const response = await fetch('/api/upload/complete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uploadId, totalChunks }),
  });

  const result = await response.json();
  return result.fileUrl;
}
```

---

## 4. バッチ処理

```typescript
// バッチ処理: 大量データの段階的処理
async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  options: {
    batchSize?: number;
    concurrency?: number;
    onProgress?: (completed: number, total: number) => void;
    onError?: (item: T, error: Error) => void;
  } = {},
): Promise<{ results: R[]; errors: { item: T; error: Error }[] }> {
  const { batchSize = 100, concurrency = 5, onProgress, onError } = options;
  const results: R[] = [];
  const errors: { item: T; error: Error }[] = [];
  let completed = 0;

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);

    // バッチ内を並行数制限付きで処理
    const batchResults = await promisePool(
      batch.map(item => async () => {
        try {
          return await processor(item);
        } catch (error) {
          const err = error as Error;
          errors.push({ item, error: err });
          onError?.(item, err);
          return null;
        }
      }),
      concurrency,
    );

    results.push(...batchResults.filter(r => r !== null) as R[]);
    completed += batch.length;
    onProgress?.(completed, items.length);
  }

  return { results, errors };
}

// 使用例
const { results, errors } = await processBatch(
  users,
  async (user) => {
    await sendNotification(user);
    return { userId: user.id, sent: true };
  },
  {
    batchSize: 50,
    concurrency: 10,
    onProgress: (done, total) => console.log(`${done}/${total}`),
    onError: (user, err) => console.error(`Failed for ${user.id}: ${err.message}`),
  },
);

console.log(`成功: ${results.length}, 失敗: ${errors.length}`);
```

---

## 5. グレースフルシャットダウン

```typescript
// サーバーのグレースフルシャットダウン
class GracefulServer {
  private server: http.Server;
  private connections = new Set<net.Socket>();

  start(port: number): void {
    this.server = app.listen(port);

    this.server.on('connection', (conn) => {
      this.connections.add(conn);
      conn.on('close', () => this.connections.delete(conn));
    });

    process.on('SIGTERM', () => this.shutdown());
    process.on('SIGINT', () => this.shutdown());
  }

  private async shutdown(): Promise<void> {
    console.log('Graceful shutdown started...');

    // 新しい接続を受け付けない
    this.server.close();

    // 進行中のリクエストの完了を待つ（最大30秒）
    const timeout = setTimeout(() => {
      console.log('Forcing shutdown...');
      this.connections.forEach(conn => conn.destroy());
    }, 30000);

    // リソースのクリーンアップ
    await db.disconnect();
    await cache.disconnect();
    await queue.close();

    clearTimeout(timeout);
    console.log('Shutdown complete');
    process.exit(0);
  }
}
```

---

## まとめ

| パターン | 用途 | キーポイント |
|---------|------|------------|
| ジョブキュー | 信頼性のある非同期処理 | リトライ + デッドレター |
| WebSocket | リアルタイム通信 | 自動再接続 |
| チャンクアップロード | 大ファイル | リトライ + 進捗表示 |
| バッチ処理 | 大量データ | 並行数制限 + エラー分離 |
| グレースフルシャットダウン | サーバー停止 | 進行中処理の完了待ち |

---

## 参考文献
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
