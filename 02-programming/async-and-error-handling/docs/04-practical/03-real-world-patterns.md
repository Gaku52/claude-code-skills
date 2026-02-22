# 実践パターン集

> 実際のプロジェクトでよく使われる非同期 + エラーハンドリングのパターンを集約。キュー処理、WebSocket、ファイルアップロード、バッチ処理など。

## この章で学ぶこと

- [ ] 実践的な非同期パターンを習得する
- [ ] エラーハンドリングの実装例を把握する
- [ ] プロダクションレベルのコードパターンを学ぶ
- [ ] サーキットブレーカー、レート制限の実装を理解する
- [ ] 分散システムでの非同期パターンを把握する

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
  scheduledAt?: Date;
  priority?: number;
}

interface JobResult {
  jobId: string;
  status: 'completed' | 'failed' | 'dead-letter';
  duration: number;
  error?: string;
}

class JobProcessor {
  private handlers = new Map<string, (data: any) => Promise<void>>();
  private metrics = {
    processed: 0,
    failed: 0,
    deadLettered: 0,
  };

  register(type: string, handler: (data: any) => Promise<void>): void {
    this.handlers.set(type, handler);
  }

  async process(job: Job): Promise<JobResult> {
    const handler = this.handlers.get(job.type);
    if (!handler) throw new Error(`No handler for job type: ${job.type}`);

    const startTime = Date.now();

    try {
      await handler(job.data);
      this.metrics.processed++;
      await this.markCompleted(job);
      return {
        jobId: job.id,
        status: 'completed',
        duration: Date.now() - startTime,
      };
    } catch (error) {
      job.attempts++;
      if (job.attempts < job.maxAttempts) {
        // リトライキューに戻す（指数バックオフ）
        const delay = Math.pow(2, job.attempts) * 1000;
        await this.scheduleRetry(job, delay);
        this.metrics.failed++;
        return {
          jobId: job.id,
          status: 'failed',
          duration: Date.now() - startTime,
          error: (error as Error).message,
        };
      } else {
        // 最大リトライ超過 → デッドレターキューに移動
        await this.moveToDeadLetter(job, error as Error);
        this.metrics.deadLettered++;
        return {
          jobId: job.id,
          status: 'dead-letter',
          duration: Date.now() - startTime,
          error: (error as Error).message,
        };
      }
    }
  }

  private async markCompleted(job: Job): Promise<void> {
    // DB更新: ジョブのステータスを完了に
    await db.query(
      'UPDATE jobs SET status = $1, completed_at = NOW() WHERE id = $2',
      ['completed', job.id],
    );
  }

  private async scheduleRetry(job: Job, delayMs: number): Promise<void> {
    const scheduledAt = new Date(Date.now() + delayMs);
    await db.query(
      'UPDATE jobs SET status = $1, attempts = $2, scheduled_at = $3 WHERE id = $4',
      ['pending', job.attempts, scheduledAt, job.id],
    );
  }

  private async moveToDeadLetter(job: Job, error: Error): Promise<void> {
    await db.query(
      `INSERT INTO dead_letter_queue (job_id, job_type, data, error, attempts, created_at)
       VALUES ($1, $2, $3, $4, $5, NOW())`,
      [job.id, job.type, JSON.stringify(job.data), error.message, job.attempts],
    );
    await db.query('UPDATE jobs SET status = $1 WHERE id = $2', ['dead-letter', job.id]);
  }

  getMetrics() {
    return { ...this.metrics };
  }
}

// 登録と使用
const processor = new JobProcessor();
processor.register('send-email', async (data) => {
  await emailService.send(data.to, data.subject, data.body);
});
processor.register('process-payment', async (data) => {
  await paymentService.charge(data.userId, data.amount);
});
processor.register('generate-report', async (data) => {
  const report = await reportService.generate(data.type, data.params);
  await storageService.upload(`reports/${data.id}.pdf`, report);
});
```

### 1.1 優先度付きジョブキュー

```typescript
// 優先度付きキューの実装
class PriorityJobQueue {
  private queues: Map<number, Job[]> = new Map();
  private processing = false;
  private concurrency: number;
  private activeJobs = 0;

  constructor(concurrency: number = 5) {
    this.concurrency = concurrency;
  }

  enqueue(job: Job): void {
    const priority = job.priority ?? 5; // デフォルト優先度5
    if (!this.queues.has(priority)) {
      this.queues.set(priority, []);
    }
    this.queues.get(priority)!.push(job);
    this.processNext();
  }

  private getNextJob(): Job | undefined {
    // 優先度の高い順（小さい数 = 高い優先度）
    const sortedPriorities = [...this.queues.keys()].sort((a, b) => a - b);

    for (const priority of sortedPriorities) {
      const queue = this.queues.get(priority)!;
      if (queue.length > 0) {
        return queue.shift();
      }
    }
    return undefined;
  }

  private async processNext(): Promise<void> {
    if (this.activeJobs >= this.concurrency) return;

    const job = this.getNextJob();
    if (!job) return;

    this.activeJobs++;

    try {
      await processor.process(job);
    } finally {
      this.activeJobs--;
      this.processNext(); // 次のジョブを処理
    }
  }
}

// 使用例
const queue = new PriorityJobQueue(10);

// 高優先度: 決済処理
queue.enqueue({
  id: 'pay-001',
  type: 'process-payment',
  data: { userId: 'u123', amount: 5000 },
  attempts: 0,
  maxAttempts: 5,
  createdAt: new Date(),
  priority: 1, // 最高優先度
});

// 低優先度: レポート生成
queue.enqueue({
  id: 'rep-001',
  type: 'generate-report',
  data: { type: 'monthly', params: { month: '2024-01' } },
  attempts: 0,
  maxAttempts: 3,
  createdAt: new Date(),
  priority: 10, // 低優先度
});
```

---

## 2. WebSocket のエラーハンドリング

```typescript
// WebSocket: 自動再接続パターン
interface WebSocketConfig {
  url: string;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  reconnectBaseDelay?: number;
  onMessage: (data: any) => void;
  onStatusChange?: (status: ConnectionStatus) => void;
}

type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected' | 'failed';

class ResilientWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private heartbeatInterval: number;
  private reconnectBaseDelay: number;
  private heartbeatTimer: NodeJS.Timer | null = null;
  private reconnectTimer: NodeJS.Timer | null = null;
  private messageBuffer: any[] = [];
  private status: ConnectionStatus = 'disconnected';
  private intentionalClose = false;

  constructor(private config: WebSocketConfig) {
    this.maxReconnectAttempts = config.maxReconnectAttempts ?? 10;
    this.heartbeatInterval = config.heartbeatInterval ?? 30000;
    this.reconnectBaseDelay = config.reconnectBaseDelay ?? 1000;
  }

  connect(): void {
    this.intentionalClose = false;
    this.setStatus('connecting');
    this.ws = new WebSocket(this.config.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.setStatus('connected');
      this.startHeartbeat();
      this.flushMessageBuffer();
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Pongメッセージは無視
        if (data.type === 'pong') return;

        this.config.onMessage(data);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };

    this.ws.onclose = (event) => {
      this.stopHeartbeat();

      if (!this.intentionalClose && !event.wasClean) {
        this.reconnect();
      } else {
        this.setStatus('disconnected');
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private reconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      this.setStatus('failed');
      return;
    }

    this.setStatus('reconnecting');

    // 指数バックオフ + ジッター
    const baseDelay = Math.min(
      this.reconnectBaseDelay * Math.pow(2, this.reconnectAttempts),
      30000,
    );
    const jitter = baseDelay * 0.2 * Math.random();
    const delay = baseDelay + jitter;

    this.reconnectAttempts++;
    console.log(`Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, this.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private flushMessageBuffer(): void {
    while (this.messageBuffer.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const data = this.messageBuffer.shift();
      this.ws.send(JSON.stringify(data));
    }
  }

  private setStatus(status: ConnectionStatus): void {
    this.status = status;
    this.config.onStatusChange?.(status);
  }

  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // 接続中はバッファに溜める
      this.messageBuffer.push(data);
      if (this.messageBuffer.length > 100) {
        this.messageBuffer.shift(); // バッファオーバーフロー防止
      }
    }
  }

  disconnect(): void {
    this.intentionalClose = true;
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.ws?.close(1000, 'Client disconnect');
    this.setStatus('disconnected');
  }

  getStatus(): ConnectionStatus {
    return this.status;
  }
}

// 使用例
const ws = new ResilientWebSocket({
  url: 'wss://api.example.com/ws',
  maxReconnectAttempts: 20,
  heartbeatInterval: 15000,
  onMessage: (data) => {
    switch (data.type) {
      case 'notification':
        showNotification(data.payload);
        break;
      case 'data-update':
        updateStore(data.payload);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  },
  onStatusChange: (status) => {
    updateConnectionIndicator(status);
  },
});

ws.connect();
```

---

## 3. ファイルアップロード

```typescript
// チャンクアップロード: 大きなファイルの信頼性のあるアップロード
interface UploadProgress {
  bytesUploaded: number;
  totalBytes: number;
  percentage: number;
  speed: number; // bytes/sec
  estimatedRemaining: number; // seconds
  currentChunk: number;
  totalChunks: number;
}

class ChunkedUploader {
  private chunkSize: number;
  private maxRetries: number;
  private abortController: AbortController | null = null;

  constructor(options: { chunkSize?: number; maxRetries?: number } = {}) {
    this.chunkSize = options.chunkSize ?? 5 * 1024 * 1024; // 5MB
    this.maxRetries = options.maxRetries ?? 3;
  }

  async upload(
    file: File,
    onProgress?: (progress: UploadProgress) => void,
  ): Promise<string> {
    this.abortController = new AbortController();
    const totalChunks = Math.ceil(file.size / this.chunkSize);
    const uploadId = crypto.randomUUID();
    let bytesUploaded = 0;
    const startTime = Date.now();

    // チェックサム計算
    const fileHash = await this.calculateHash(file);

    for (let i = 0; i < totalChunks; i++) {
      const start = i * this.chunkSize;
      const end = Math.min(start + this.chunkSize, file.size);
      const chunk = file.slice(start, end);

      // リトライ付きでチャンクをアップロード
      await this.uploadChunkWithRetry(
        chunk, uploadId, i, totalChunks, fileHash,
      );

      bytesUploaded += chunk.size;

      // 進捗計算
      const elapsed = (Date.now() - startTime) / 1000;
      const speed = bytesUploaded / elapsed;
      const remaining = (file.size - bytesUploaded) / speed;

      onProgress?.({
        bytesUploaded,
        totalBytes: file.size,
        percentage: Math.round((bytesUploaded / file.size) * 100),
        speed,
        estimatedRemaining: remaining,
        currentChunk: i + 1,
        totalChunks,
      });
    }

    // 完了通知
    const response = await fetch('/api/upload/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploadId, totalChunks, fileHash }),
      signal: this.abortController.signal,
    });

    const result = await response.json();
    return result.fileUrl;
  }

  private async uploadChunkWithRetry(
    chunk: Blob,
    uploadId: string,
    chunkIndex: number,
    totalChunks: number,
    fileHash: string,
  ): Promise<void> {
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const formData = new FormData();
        formData.append('chunk', chunk);
        formData.append('uploadId', uploadId);
        formData.append('chunkIndex', String(chunkIndex));
        formData.append('totalChunks', String(totalChunks));
        formData.append('fileHash', fileHash);

        const response = await fetch('/api/upload/chunk', {
          method: 'POST',
          body: formData,
          signal: this.abortController?.signal,
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
        }

        return; // 成功
      } catch (error) {
        if ((error as Error).name === 'AbortError') throw error;
        if (attempt === this.maxRetries) throw error;

        // リトライ前に待機
        await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 1000));
      }
    }
  }

  private async calculateHash(file: File): Promise<string> {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  cancel(): void {
    this.abortController?.abort();
  }
}

// レジュームアップロード（中断再開）
class ResumableUploader extends ChunkedUploader {
  private storageKey: string;

  constructor(file: File) {
    super();
    this.storageKey = `upload-progress-${file.name}-${file.size}-${file.lastModified}`;
  }

  async getResumePoint(): Promise<number> {
    const saved = localStorage.getItem(this.storageKey);
    if (!saved) return 0;

    const { uploadId, lastChunk } = JSON.parse(saved);

    // サーバーに確認
    try {
      const response = await fetch(`/api/upload/status/${uploadId}`);
      const data = await response.json();
      return data.completedChunks;
    } catch {
      return 0;
    }
  }

  saveProgress(uploadId: string, chunkIndex: number): void {
    localStorage.setItem(this.storageKey, JSON.stringify({
      uploadId,
      lastChunk: chunkIndex,
      timestamp: Date.now(),
    }));
  }

  clearProgress(): void {
    localStorage.removeItem(this.storageKey);
  }
}

// 使用例
const uploader = new ChunkedUploader({ chunkSize: 10 * 1024 * 1024 }); // 10MB chunks

const fileUrl = await uploader.upload(selectedFile, (progress) => {
  progressBar.style.width = `${progress.percentage}%`;
  progressText.textContent = `${progress.percentage}% (${formatSpeed(progress.speed)})`;
  etaText.textContent = `残り約${Math.ceil(progress.estimatedRemaining)}秒`;
});
```

---

## 4. バッチ処理

```typescript
// バッチ処理: 大量データの段階的処理
interface BatchOptions<T> {
  batchSize?: number;
  concurrency?: number;
  onProgress?: (completed: number, total: number) => void;
  onError?: (item: T, error: Error) => void;
  onBatchComplete?: (batchIndex: number, results: any[]) => void;
  signal?: AbortSignal;
  retryFailedItems?: boolean;
  maxRetries?: number;
}

async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  options: BatchOptions<T> = {},
): Promise<{ results: R[]; errors: { item: T; error: Error }[] }> {
  const {
    batchSize = 100,
    concurrency = 5,
    onProgress,
    onError,
    onBatchComplete,
    signal,
    retryFailedItems = false,
    maxRetries = 2,
  } = options;

  const results: R[] = [];
  const errors: { item: T; error: Error }[] = [];
  let completed = 0;
  let batchIndex = 0;

  for (let i = 0; i < items.length; i += batchSize) {
    signal?.throwIfAborted();

    const batch = items.slice(i, i + batchSize);

    // バッチ内を並行数制限付きで処理
    const batchResults = await promisePool(
      batch.map(item => async () => {
        try {
          const result = await processWithRetry(item, processor, retryFailedItems ? maxRetries : 0);
          return { success: true as const, result };
        } catch (error) {
          const err = error as Error;
          errors.push({ item, error: err });
          onError?.(item, err);
          return { success: false as const, error: err };
        }
      }),
      concurrency,
    );

    const successfulResults = batchResults
      .filter((r): r is { success: true; result: R } => r.success)
      .map(r => r.result);

    results.push(...successfulResults);
    completed += batch.length;
    onProgress?.(completed, items.length);
    onBatchComplete?.(batchIndex++, successfulResults);
  }

  return { results, errors };
}

async function processWithRetry<T, R>(
  item: T,
  processor: (item: T) => Promise<R>,
  maxRetries: number,
): Promise<R> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await processor(item);
    } catch (error) {
      lastError = error as Error;
      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 500));
      }
    }
  }

  throw lastError!;
}

// 並行数制限付き Promise プール
async function promisePool<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
): Promise<T[]> {
  const results: T[] = [];
  const executing = new Set<Promise<void>>();

  for (const [index, task] of tasks.entries()) {
    const promise = task().then(result => {
      results[index] = result;
    });

    const managed = promise.finally(() => executing.delete(managed));
    executing.add(managed);

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
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
    retryFailedItems: true,
    maxRetries: 3,
    onProgress: (done, total) => console.log(`${done}/${total}`),
    onError: (user, err) => console.error(`Failed for ${user.id}: ${err.message}`),
    onBatchComplete: (idx, results) => console.log(`Batch ${idx}: ${results.length} items`),
  },
);

console.log(`成功: ${results.length}, 失敗: ${errors.length}`);
```

---

## 5. サーキットブレーカー

```typescript
// サーキットブレーカーパターン: 障害の連鎖を防ぐ
type CircuitState = 'closed' | 'open' | 'half-open';

interface CircuitBreakerOptions {
  failureThreshold: number;   // 失敗回数の閾値
  recoveryTimeout: number;    // open → half-open の待ち時間(ms)
  monitoringWindow: number;   // 失敗をカウントする時間窓(ms)
  halfOpenMaxCalls: number;   // half-open時の最大試行回数
  onStateChange?: (from: CircuitState, to: CircuitState) => void;
}

class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures: number[] = []; // 失敗のタイムスタンプ
  private lastOpenTime: number = 0;
  private halfOpenCalls = 0;
  private halfOpenSuccesses = 0;

  constructor(private options: CircuitBreakerOptions) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastOpenTime >= this.options.recoveryTimeout) {
        this.transition('half-open');
      } else {
        throw new CircuitOpenError('Circuit breaker is open');
      }
    }

    if (this.state === 'half-open') {
      if (this.halfOpenCalls >= this.options.halfOpenMaxCalls) {
        throw new CircuitOpenError('Circuit breaker half-open limit reached');
      }
      this.halfOpenCalls++;
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    if (this.state === 'half-open') {
      this.halfOpenSuccesses++;
      if (this.halfOpenSuccesses >= this.options.halfOpenMaxCalls) {
        this.transition('closed');
      }
    }
    // closed状態では失敗カウンタをリセットしない（窓で管理）
  }

  private onFailure(): void {
    const now = Date.now();

    if (this.state === 'half-open') {
      this.transition('open');
      return;
    }

    // 時間窓内の失敗をカウント
    this.failures.push(now);
    this.failures = this.failures.filter(
      t => now - t < this.options.monitoringWindow,
    );

    if (this.failures.length >= this.options.failureThreshold) {
      this.transition('open');
    }
  }

  private transition(newState: CircuitState): void {
    const oldState = this.state;
    this.state = newState;

    if (newState === 'open') {
      this.lastOpenTime = Date.now();
    }

    if (newState === 'closed') {
      this.failures = [];
      this.halfOpenCalls = 0;
      this.halfOpenSuccesses = 0;
    }

    if (newState === 'half-open') {
      this.halfOpenCalls = 0;
      this.halfOpenSuccesses = 0;
    }

    this.options.onStateChange?.(oldState, newState);
    console.log(`Circuit breaker: ${oldState} → ${newState}`);
  }

  getState(): CircuitState {
    return this.state;
  }
}

class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

// 使用例
const breaker = new CircuitBreaker({
  failureThreshold: 5,       // 5回失敗でopen
  recoveryTimeout: 30000,    // 30秒後にhalf-open
  monitoringWindow: 60000,   // 60秒の窓
  halfOpenMaxCalls: 3,       // half-openで3回試行
  onStateChange: (from, to) => {
    if (to === 'open') {
      alertOps(`External service circuit opened (from ${from})`);
    }
  },
});

// API呼び出しをサーキットブレーカーで保護
async function callExternalApi(endpoint: string): Promise<any> {
  try {
    return await breaker.execute(async () => {
      const response = await fetch(`https://external-api.com${endpoint}`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
  } catch (error) {
    if (error instanceof CircuitOpenError) {
      // フォールバック: キャッシュから返す
      return getCachedData(endpoint);
    }
    throw error;
  }
}
```

---

## 6. レート制限

```typescript
// トークンバケット方式のレート制限
class RateLimiter {
  private tokens: number;
  private maxTokens: number;
  private refillRate: number; // tokens/sec
  private lastRefill: number;
  private waitQueue: Array<{
    resolve: () => void;
    reject: (err: Error) => void;
  }> = [];

  constructor(options: {
    maxTokens: number;
    refillRate: number;
  }) {
    this.maxTokens = options.maxTokens;
    this.tokens = options.maxTokens;
    this.refillRate = options.refillRate;
    this.lastRefill = Date.now();
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRate);
    this.lastRefill = now;
  }

  async acquire(count: number = 1): Promise<void> {
    this.refill();

    if (this.tokens >= count) {
      this.tokens -= count;
      return;
    }

    // トークン不足: 待機
    return new Promise((resolve, reject) => {
      this.waitQueue.push({ resolve, reject });

      // 必要なトークンが補充されるまでの時間を計算
      const waitMs = ((count - this.tokens) / this.refillRate) * 1000;

      setTimeout(() => {
        const index = this.waitQueue.findIndex(w => w.resolve === resolve);
        if (index !== -1) {
          this.waitQueue.splice(index, 1);
          this.refill();
          if (this.tokens >= count) {
            this.tokens -= count;
            resolve();
          } else {
            reject(new Error('Rate limit exceeded'));
          }
        }
      }, waitMs);
    });
  }

  // デコレータとして使用
  wrap<T>(fn: () => Promise<T>): () => Promise<T> {
    return async () => {
      await this.acquire();
      return fn();
    };
  }
}

// スライディングウィンドウ方式
class SlidingWindowRateLimiter {
  private timestamps: number[] = [];
  private maxRequests: number;
  private windowMs: number;

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
  }

  canProceed(): boolean {
    const now = Date.now();
    this.timestamps = this.timestamps.filter(t => now - t < this.windowMs);
    return this.timestamps.length < this.maxRequests;
  }

  record(): void {
    this.timestamps.push(Date.now());
  }

  async waitAndProceed(): Promise<void> {
    while (!this.canProceed()) {
      const oldestInWindow = this.timestamps[0];
      const waitMs = this.windowMs - (Date.now() - oldestInWindow) + 1;
      await new Promise(r => setTimeout(r, waitMs));
    }
    this.record();
  }

  getRemaining(): number {
    const now = Date.now();
    this.timestamps = this.timestamps.filter(t => now - t < this.windowMs);
    return Math.max(0, this.maxRequests - this.timestamps.length);
  }
}

// 使用例
const limiter = new RateLimiter({
  maxTokens: 100,    // 最大100トークン
  refillRate: 10,     // 毎秒10トークン回復
});

// APIクライアントにレート制限を適用
class ApiClient {
  private limiter = new SlidingWindowRateLimiter(100, 60000); // 60秒に100リクエスト

  async request<T>(endpoint: string): Promise<T> {
    await this.limiter.waitAndProceed();

    const response = await fetch(`https://api.example.com${endpoint}`);
    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get('Retry-After') || '5');
      await new Promise(r => setTimeout(r, retryAfter * 1000));
      return this.request(endpoint); // リトライ
    }

    return response.json();
  }
}
```

---

## 7. グレースフルシャットダウン

```typescript
// サーバーのグレースフルシャットダウン
import http from 'http';
import net from 'net';

class GracefulServer {
  private server: http.Server;
  private connections = new Set<net.Socket>();
  private isShuttingDown = false;
  private shutdownPromise: Promise<void> | null = null;

  constructor(private app: any) {
    this.server = http.createServer(app);
  }

  start(port: number): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(port, () => {
        console.log(`Server listening on port ${port}`);
        resolve();
      });

      // 接続の追跡
      this.server.on('connection', (conn) => {
        this.connections.add(conn);
        conn.on('close', () => this.connections.delete(conn));
      });

      // シグナルハンドリング
      const shutdownHandler = (signal: string) => {
        console.log(`${signal} received`);
        this.shutdown().then(() => process.exit(0));
      };

      process.on('SIGTERM', () => shutdownHandler('SIGTERM'));
      process.on('SIGINT', () => shutdownHandler('SIGINT'));

      // Unhandled rejection / exception
      process.on('unhandledRejection', (reason) => {
        console.error('Unhandled Rejection:', reason);
        this.shutdown().then(() => process.exit(1));
      });
    });
  }

  async shutdown(): Promise<void> {
    if (this.shutdownPromise) return this.shutdownPromise;

    this.isShuttingDown = true;
    console.log('Graceful shutdown started...');

    this.shutdownPromise = this.performShutdown();
    return this.shutdownPromise;
  }

  private async performShutdown(): Promise<void> {
    // 1. 新しい接続を受け付けない
    this.server.close();

    // 2. Keep-Alive接続にConnection: closeヘッダーを設定
    this.connections.forEach(conn => {
      (conn as any)._httpMessage?.setHeader?.('Connection', 'close');
    });

    // 3. 進行中のリクエストの完了を待つ（最大30秒）
    const forceTimeout = setTimeout(() => {
      console.log('Forcing shutdown: destroying remaining connections');
      this.connections.forEach(conn => conn.destroy());
    }, 30000);

    // 4. リソースのクリーンアップ
    try {
      await Promise.allSettled([
        this.closeDatabase(),
        this.closeCache(),
        this.closeMessageQueue(),
        this.flushLogs(),
      ]);
      console.log('Resources cleaned up successfully');
    } catch (error) {
      console.error('Error during cleanup:', error);
    }

    clearTimeout(forceTimeout);
    console.log('Shutdown complete');
  }

  private async closeDatabase(): Promise<void> {
    console.log('Closing database connections...');
    await db.end();
  }

  private async closeCache(): Promise<void> {
    console.log('Closing cache connections...');
    await redis.quit();
  }

  private async closeMessageQueue(): Promise<void> {
    console.log('Closing message queue...');
    await queue.close();
  }

  private async flushLogs(): Promise<void> {
    console.log('Flushing logs...');
    await logger.flush();
  }

  // ヘルスチェック用
  isHealthy(): boolean {
    return !this.isShuttingDown;
  }
}

// 使用例
const server = new GracefulServer(app);
await server.start(3000);

// ヘルスチェックエンドポイント
app.get('/health', (req, res) => {
  if (server.isHealthy()) {
    res.status(200).json({ status: 'healthy', uptime: process.uptime() });
  } else {
    res.status(503).json({ status: 'shutting-down' });
  }
});
```

---

## 8. 分散ロック

```typescript
// Redis を使った分散ロック（Redlock アルゴリズム簡易版）
class DistributedLock {
  constructor(
    private redis: Redis,
    private lockKey: string,
    private ttlMs: number = 10000,
  ) {}

  async acquire(waitMs: number = 5000): Promise<string | null> {
    const lockId = crypto.randomUUID();
    const startTime = Date.now();

    while (Date.now() - startTime < waitMs) {
      // SET NX（存在しない場合のみ設定）
      const acquired = await this.redis.set(
        `lock:${this.lockKey}`,
        lockId,
        'PX', this.ttlMs,
        'NX',
      );

      if (acquired === 'OK') {
        return lockId; // ロック取得成功
      }

      // 少し待ってリトライ
      await new Promise(r => setTimeout(r, 50 + Math.random() * 50));
    }

    return null; // タイムアウト
  }

  async release(lockId: string): Promise<boolean> {
    // Lua スクリプトでアトミックに解放（自分のロックだけ解放）
    const script = `
      if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
      else
        return 0
      end
    `;

    const result = await this.redis.eval(script, 1, `lock:${this.lockKey}`, lockId);
    return result === 1;
  }

  // ロック付きで処理を実行
  async withLock<T>(fn: () => Promise<T>, waitMs?: number): Promise<T> {
    const lockId = await this.acquire(waitMs);
    if (!lockId) {
      throw new Error(`Failed to acquire lock: ${this.lockKey}`);
    }

    try {
      return await fn();
    } finally {
      await this.release(lockId);
    }
  }
}

// 使用例: 排他的な処理
const lock = new DistributedLock(redis, 'user:123:payment');

try {
  const result = await lock.withLock(async () => {
    const balance = await getBalance('user-123');
    if (balance < amount) throw new Error('Insufficient balance');
    await deductBalance('user-123', amount);
    await createPayment('user-123', amount);
    return { success: true };
  });
} catch (error) {
  if (error.message.includes('Failed to acquire lock')) {
    // 別のプロセスが処理中
    res.status(409).json({ error: 'Payment already in progress' });
  }
}
```

---

## 9. イベント駆動パターン

```typescript
// 型安全なイベントバス
interface EventMap {
  'user:created': { userId: string; email: string };
  'user:updated': { userId: string; changes: Partial<User> };
  'order:placed': { orderId: string; userId: string; amount: number };
  'order:completed': { orderId: string };
  'payment:succeeded': { paymentId: string; orderId: string };
  'payment:failed': { paymentId: string; orderId: string; reason: string };
}

class TypedEventEmitter<TEvents extends Record<string, any>> {
  private handlers = new Map<string, Set<(data: any) => void | Promise<void>>>();

  on<K extends keyof TEvents>(
    event: K,
    handler: (data: TEvents[K]) => void | Promise<void>,
  ): () => void {
    const key = event as string;
    if (!this.handlers.has(key)) {
      this.handlers.set(key, new Set());
    }
    this.handlers.get(key)!.add(handler);

    // unsubscribe関数を返す
    return () => {
      this.handlers.get(key)?.delete(handler);
    };
  }

  async emit<K extends keyof TEvents>(event: K, data: TEvents[K]): Promise<void> {
    const key = event as string;
    const handlers = this.handlers.get(key);
    if (!handlers) return;

    // 全ハンドラを並行実行（エラーは個別にキャッチ）
    const results = await Promise.allSettled(
      [...handlers].map(handler => handler(data)),
    );

    // エラーをログ
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(`Event handler error for ${key}:`, result.reason);
      }
    });
  }
}

// 使用例
const eventBus = new TypedEventEmitter<EventMap>();

// ハンドラ登録
eventBus.on('order:placed', async ({ orderId, userId, amount }) => {
  await paymentService.processPayment(orderId, userId, amount);
});

eventBus.on('order:placed', async ({ orderId, userId }) => {
  await notificationService.sendOrderConfirmation(userId, orderId);
});

eventBus.on('payment:succeeded', async ({ orderId }) => {
  await orderService.markAsPaid(orderId);
  await eventBus.emit('order:completed', { orderId });
});

eventBus.on('payment:failed', async ({ orderId, reason }) => {
  await orderService.markAsFailed(orderId, reason);
  await notificationService.sendPaymentFailure(orderId, reason);
});

// イベント発行
await eventBus.emit('order:placed', {
  orderId: 'ord-123',
  userId: 'usr-456',
  amount: 5000,
});
```

---

## 10. データストリーム処理

```typescript
// Node.js Transform Stream を使ったパイプライン処理
import { Transform, pipeline } from 'stream';
import { promisify } from 'util';
import { createReadStream, createWriteStream } from 'fs';
import { createGzip } from 'zlib';

const pipelineAsync = promisify(pipeline);

// CSVを行ごとに処理するTransform
class CsvLineProcessor extends Transform {
  private buffer = '';
  private headers: string[] = [];
  private lineCount = 0;

  constructor(
    private processor: (row: Record<string, string>) => Record<string, string> | null,
  ) {
    super({ objectMode: true });
  }

  _transform(chunk: Buffer, encoding: string, callback: Function): void {
    this.buffer += chunk.toString();
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop()!; // 最後の不完全な行をバッファに残す

    for (const line of lines) {
      if (line.trim() === '') continue;

      if (this.lineCount === 0) {
        this.headers = line.split(',').map(h => h.trim());
      } else {
        const values = line.split(',');
        const row: Record<string, string> = {};
        this.headers.forEach((h, i) => { row[h] = values[i]?.trim() ?? ''; });

        try {
          const result = this.processor(row);
          if (result) {
            this.push(JSON.stringify(result) + '\n');
          }
        } catch (error) {
          console.error(`Error processing row ${this.lineCount}:`, error);
          // エラー行はスキップして続行
        }
      }

      this.lineCount++;
    }

    callback();
  }

  _flush(callback: Function): void {
    if (this.buffer.trim()) {
      // 最後の行を処理
      const values = this.buffer.split(',');
      const row: Record<string, string> = {};
      this.headers.forEach((h, i) => { row[h] = values[i]?.trim() ?? ''; });

      try {
        const result = this.processor(row);
        if (result) {
          this.push(JSON.stringify(result) + '\n');
        }
      } catch (error) {
        console.error('Error processing last row:', error);
      }
    }
    callback();
  }
}

// 使用例: 大きなCSVファイルの変換
async function transformLargeCsv(inputPath: string, outputPath: string): Promise<void> {
  await pipelineAsync(
    createReadStream(inputPath),
    new CsvLineProcessor((row) => {
      // 変換ロジック
      if (row.status === 'inactive') return null; // フィルタリング
      return {
        ...row,
        name: row.name.toUpperCase(),
        processedAt: new Date().toISOString(),
      };
    }),
    createGzip(), // 圧縮
    createWriteStream(outputPath + '.gz'),
  );

  console.log(`Transformed and compressed: ${inputPath} → ${outputPath}.gz`);
}
```

---

## まとめ

| パターン | 用途 | キーポイント |
|---------|------|------------|
| ジョブキュー | 信頼性のある非同期処理 | リトライ + デッドレター + 優先度 |
| WebSocket | リアルタイム通信 | 自動再接続 + ハートビート + バッファ |
| チャンクアップロード | 大ファイル | リトライ + 進捗表示 + レジューム |
| バッチ処理 | 大量データ | 並行数制限 + エラー分離 + 進捗 |
| サーキットブレーカー | 障害の連鎖防止 | 3状態遷移 + フォールバック |
| レート制限 | API保護 | トークンバケット + スライディングウィンドウ |
| グレースフルシャットダウン | サーバー停止 | 進行中処理の完了待ち + リソースクリーンアップ |
| 分散ロック | 排他制御 | Redis + TTL + Lua Script |
| イベント駆動 | 疎結合な処理連携 | 型安全 + エラー分離 |
| ストリーム処理 | 大データ変換 | パイプライン + バックプレッシャー |

---

## FAQ

### Q1: サーキットブレーカーの閾値はどう決めるか？

失敗率の閾値（例: 50%）とウィンドウサイズ（例: 直近10リクエスト）は、サービスの特性に依存する。一般的に、高頻度のAPI呼び出しでは小さなウィンドウ（10-20件）と低めの閾値（30-50%）を設定し、素早くOpenに遷移させる。低頻度の場合はウィンドウを大きく（50-100件）し、一時的なスパイクでOpenにならないようにする。Half-Openでの試行回数は1-3回が一般的で、成功した場合にClosedに戻す。Hystrix（Netflix）やPolly（.NET）などのライブラリの設定値を参考にするとよい。

### Q2: 分散ロックでRedlockアルゴリズムを使うべきか？

単一のRedisインスタンスでは、そのインスタンスがダウンすると全てのロックが失われる。Redlockアルゴリズムは複数のRedisインスタンス（通常5台）に対してロックを取得し、過半数のインスタンスでロックが取得できた場合にのみ有効とする。ただし、Martin Kleppmann氏による批判（「How to do distributed locking」）もあり、ロックの厳密性が求められる場面ではZooKeeperやetcdのリース機能の方が安全とされる。支払いなどのクリティカルな処理では、べき等性キー（idempotency key）と組み合わせて二重処理を防ぐ設計が推奨される。

### Q3: レート制限をクライアント側で実装する意味はあるか？

サーバー側のレート制限はAPI保護の基本だが、クライアント側でも実装する意味がある。サーバーから429レスポンスを受けてからリトライするより、事前にリクエスト頻度を制御する方が効率的でネットワーク負荷も低い。特にバッチ処理で大量のAPIコールを行う場面では、クライアント側のレート制限により安定したスループットを維持できる。`Retry-After`ヘッダーや`X-RateLimit-Remaining`ヘッダーを活用して動的にレートを調整するアダプティブ方式も有効である。

---

## 参考文献
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
3. Node.js Documentation. "Stream." nodejs.org.
4. Martin Fowler. "Circuit Breaker." martinfowler.com.
5. Redis Documentation. "Distributed Locks." redis.io.
