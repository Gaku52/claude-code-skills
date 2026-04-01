# Real-World Pattern Collection

> A consolidated collection of async + error handling patterns commonly used in real projects. Covers queue processing, WebSocket, file uploads, batch processing, and more.

## What You Will Learn in This Chapter

- [ ] Master practical asynchronous patterns
- [ ] Understand error handling implementation examples
- [ ] Learn production-level code patterns
- [ ] Understand circuit breaker and rate limiting implementations
- [ ] Grasp asynchronous patterns in distributed systems


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Testing Async Code](./02-testing-async.md)

---

## 1. Job Queue Processing

```typescript
// Job queue: Reliable asynchronous processing
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
        // Return to retry queue (exponential backoff)
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
        // Max retries exceeded → Move to dead letter queue
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
    // DB update: Set job status to completed
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

// Registration and usage
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

### 1.1 Priority Job Queue

```typescript
// Priority queue implementation
class PriorityJobQueue {
  private queues: Map<number, Job[]> = new Map();
  private processing = false;
  private concurrency: number;
  private activeJobs = 0;

  constructor(concurrency: number = 5) {
    this.concurrency = concurrency;
  }

  enqueue(job: Job): void {
    const priority = job.priority ?? 5; // Default priority 5
    if (!this.queues.has(priority)) {
      this.queues.set(priority, []);
    }
    this.queues.get(priority)!.push(job);
    this.processNext();
  }

  private getNextJob(): Job | undefined {
    // Process in order of priority (lower number = higher priority)
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
      this.processNext(); // Process the next job
    }
  }
}

// Usage example
const queue = new PriorityJobQueue(10);

// High priority: Payment processing
queue.enqueue({
  id: 'pay-001',
  type: 'process-payment',
  data: { userId: 'u123', amount: 5000 },
  attempts: 0,
  maxAttempts: 5,
  createdAt: new Date(),
  priority: 1, // Highest priority
});

// Low priority: Report generation
queue.enqueue({
  id: 'rep-001',
  type: 'generate-report',
  data: { type: 'monthly', params: { month: '2024-01' } },
  attempts: 0,
  maxAttempts: 3,
  createdAt: new Date(),
  priority: 10, // Low priority
});
```

---

## 2. WebSocket Error Handling

```typescript
// WebSocket: Auto-reconnection pattern
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

        // Ignore pong messages
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

    // Exponential backoff + jitter
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
      // Buffer messages while connecting
      this.messageBuffer.push(data);
      if (this.messageBuffer.length > 100) {
        this.messageBuffer.shift(); // Prevent buffer overflow
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

// Usage example
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

## 3. File Upload

```typescript
// Chunked upload: Reliable upload of large files
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

    // Calculate checksum
    const fileHash = await this.calculateHash(file);

    for (let i = 0; i < totalChunks; i++) {
      const start = i * this.chunkSize;
      const end = Math.min(start + this.chunkSize, file.size);
      const chunk = file.slice(start, end);

      // Upload chunk with retry
      await this.uploadChunkWithRetry(
        chunk, uploadId, i, totalChunks, fileHash,
      );

      bytesUploaded += chunk.size;

      // Progress calculation
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

    // Completion notification
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

        return; // Success
      } catch (error) {
        if ((error as Error).name === 'AbortError') throw error;
        if (attempt === this.maxRetries) throw error;

        // Wait before retry
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

// Resumable upload (interrupt and resume)
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

    // Verify with the server
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

// Usage example
const uploader = new ChunkedUploader({ chunkSize: 10 * 1024 * 1024 }); // 10MB chunks

const fileUrl = await uploader.upload(selectedFile, (progress) => {
  progressBar.style.width = `${progress.percentage}%`;
  progressText.textContent = `${progress.percentage}% (${formatSpeed(progress.speed)})`;
  etaText.textContent = `Approximately ${Math.ceil(progress.estimatedRemaining)} seconds remaining`;
});
```

---

## 4. Batch Processing

```typescript
// Batch processing: Incremental processing of large datasets
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

    // Process within the batch with concurrency limiting
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

// Concurrency-limited Promise pool
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

// Usage example
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

console.log(`Succeeded: ${results.length}, Failed: ${errors.length}`);
```

---

## 5. Circuit Breaker

```typescript
// Circuit breaker pattern: Prevent cascading failures
type CircuitState = 'closed' | 'open' | 'half-open';

interface CircuitBreakerOptions {
  failureThreshold: number;   // Failure count threshold
  recoveryTimeout: number;    // Wait time from open → half-open (ms)
  monitoringWindow: number;   // Time window for counting failures (ms)
  halfOpenMaxCalls: number;   // Max trial calls during half-open
  onStateChange?: (from: CircuitState, to: CircuitState) => void;
}

class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures: number[] = []; // Failure timestamps
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
    // In closed state, failure counter is not reset (managed by window)
  }

  private onFailure(): void {
    const now = Date.now();

    if (this.state === 'half-open') {
      this.transition('open');
      return;
    }

    // Count failures within the time window
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

// Usage example
const breaker = new CircuitBreaker({
  failureThreshold: 5,       // Open after 5 failures
  recoveryTimeout: 30000,    // Half-open after 30 seconds
  monitoringWindow: 60000,   // 60-second window
  halfOpenMaxCalls: 3,       // 3 trial calls during half-open
  onStateChange: (from, to) => {
    if (to === 'open') {
      alertOps(`External service circuit opened (from ${from})`);
    }
  },
});

// Protect API calls with a circuit breaker
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
      // Fallback: Return from cache
      return getCachedData(endpoint);
    }
    throw error;
  }
}
```

---

## 6. Rate Limiting

```typescript
// Token bucket rate limiter
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

    // Insufficient tokens: wait
    return new Promise((resolve, reject) => {
      this.waitQueue.push({ resolve, reject });

      // Calculate time until needed tokens are replenished
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

  // Use as a decorator
  wrap<T>(fn: () => Promise<T>): () => Promise<T> {
    return async () => {
      await this.acquire();
      return fn();
    };
  }
}

// Sliding window rate limiter
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

// Usage example
const limiter = new RateLimiter({
  maxTokens: 100,    // Maximum 100 tokens
  refillRate: 10,     // 10 tokens replenished per second
});

// Apply rate limiting to an API client
class ApiClient {
  private limiter = new SlidingWindowRateLimiter(100, 60000); // 100 requests per 60 seconds

  async request<T>(endpoint: string): Promise<T> {
    await this.limiter.waitAndProceed();

    const response = await fetch(`https://api.example.com${endpoint}`);
    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get('Retry-After') || '5');
      await new Promise(r => setTimeout(r, retryAfter * 1000));
      return this.request(endpoint); // Retry
    }

    return response.json();
  }
}
```

---

## 7. Graceful Shutdown

```typescript
// Server graceful shutdown
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

      // Track connections
      this.server.on('connection', (conn) => {
        this.connections.add(conn);
        conn.on('close', () => this.connections.delete(conn));
      });

      // Signal handling
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
    // 1. Stop accepting new connections
    this.server.close();

    // 2. Set Connection: close header on Keep-Alive connections
    this.connections.forEach(conn => {
      (conn as any)._httpMessage?.setHeader?.('Connection', 'close');
    });

    // 3. Wait for in-progress requests to complete (max 30 seconds)
    const forceTimeout = setTimeout(() => {
      console.log('Forcing shutdown: destroying remaining connections');
      this.connections.forEach(conn => conn.destroy());
    }, 30000);

    // 4. Resource cleanup
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

  // For health checks
  isHealthy(): boolean {
    return !this.isShuttingDown;
  }
}

// Usage example
const server = new GracefulServer(app);
await server.start(3000);

// Health check endpoint
app.get('/health', (req, res) => {
  if (server.isHealthy()) {
    res.status(200).json({ status: 'healthy', uptime: process.uptime() });
  } else {
    res.status(503).json({ status: 'shutting-down' });
  }
});
```

---

## 8. Distributed Locking

```typescript
// Distributed lock using Redis (simplified Redlock algorithm)
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
      // SET NX (set only if not exists)
      const acquired = await this.redis.set(
        `lock:${this.lockKey}`,
        lockId,
        'PX', this.ttlMs,
        'NX',
      );

      if (acquired === 'OK') {
        return lockId; // Lock acquired successfully
      }

      // Wait briefly and retry
      await new Promise(r => setTimeout(r, 50 + Math.random() * 50));
    }

    return null; // Timeout
  }

  async release(lockId: string): Promise<boolean> {
    // Release atomically with Lua script (only release own lock)
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

  // Execute a function with the lock held
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

// Usage example: Exclusive processing
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
    // Another process is handling it
    res.status(409).json({ error: 'Payment already in progress' });
  }
}
```

---

## 9. Event-Driven Pattern

```typescript
// Type-safe event bus
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

    // Return an unsubscribe function
    return () => {
      this.handlers.get(key)?.delete(handler);
    };
  }

  async emit<K extends keyof TEvents>(event: K, data: TEvents[K]): Promise<void> {
    const key = event as string;
    const handlers = this.handlers.get(key);
    if (!handlers) return;

    // Execute all handlers in parallel (catch errors individually)
    const results = await Promise.allSettled(
      [...handlers].map(handler => handler(data)),
    );

    // Log errors
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(`Event handler error for ${key}:`, result.reason);
      }
    });
  }
}

// Usage example
const eventBus = new TypedEventEmitter<EventMap>();

// Register handlers
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

// Emit event
await eventBus.emit('order:placed', {
  orderId: 'ord-123',
  userId: 'usr-456',
  amount: 5000,
});
```

---

## 10. Data Stream Processing

```typescript
// Pipeline processing using Node.js Transform Streams
import { Transform, pipeline } from 'stream';
import { promisify } from 'util';
import { createReadStream, createWriteStream } from 'fs';
import { createGzip } from 'zlib';

const pipelineAsync = promisify(pipeline);

// Transform that processes CSV line by line
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
    this.buffer = lines.pop()!; // Keep the last incomplete line in the buffer

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
          // Skip error rows and continue
        }
      }

      this.lineCount++;
    }

    callback();
  }

  _flush(callback: Function): void {
    if (this.buffer.trim()) {
      // Process the last line
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

// Usage example: Transform a large CSV file
async function transformLargeCsv(inputPath: string, outputPath: string): Promise<void> {
  await pipelineAsync(
    createReadStream(inputPath),
    new CsvLineProcessor((row) => {
      // Transformation logic
      if (row.status === 'inactive') return null; // Filtering
      return {
        ...row,
        name: row.name.toUpperCase(),
        processedAt: new Date().toISOString(),
      };
    }),
    createGzip(), // Compression
    createWriteStream(outputPath + '.gz'),
  );

  console.log(`Transformed and compressed: ${inputPath} → ${outputPath}.gz`);
}
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
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

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be mindful of algorithmic time complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks
---

## Summary

| Pattern | Use Case | Key Points |
|---------|----------|------------|
| Job queue | Reliable async processing | Retry + dead letter + priority |
| WebSocket | Real-time communication | Auto-reconnect + heartbeat + buffer |
| Chunked upload | Large files | Retry + progress display + resume |
| Batch processing | Large datasets | Concurrency limit + error isolation + progress |
| Circuit breaker | Prevent cascading failures | 3-state transitions + fallback |
| Rate limiting | API protection | Token bucket + sliding window |
| Graceful shutdown | Server shutdown | Wait for in-progress requests + resource cleanup |
| Distributed lock | Mutual exclusion | Redis + TTL + Lua script |
| Event-driven | Loosely coupled coordination | Type safety + error isolation |
| Stream processing | Large data transformation | Pipeline + backpressure |

---

## FAQ

### Q1: How should circuit breaker thresholds be determined?

The failure rate threshold (e.g., 50%) and window size (e.g., last 10 requests) depend on the characteristics of the service. Generally, for high-frequency API calls, set a small window (10-20 requests) and a lower threshold (30-50%) to transition to Open quickly. For low-frequency calls, use a larger window (50-100 requests) to avoid opening on temporary spikes. The number of trial calls during Half-Open is typically 1-3, transitioning back to Closed on success. Refer to the default settings of libraries like Hystrix (Netflix) or Polly (.NET) as a starting point.

### Q2: Should the Redlock algorithm be used for distributed locking?

With a single Redis instance, all locks are lost if that instance goes down. The Redlock algorithm acquires locks across multiple Redis instances (typically 5) and considers the lock valid only when a majority of instances grant it. However, there are critiques from Martin Kleppmann ("How to do distributed locking"), and for scenarios requiring strict lock semantics, lease functionality in ZooKeeper or etcd is considered safer. For critical operations like payments, combining idempotency keys with locks to prevent duplicate processing is recommended.

### Q3: Is there value in implementing rate limiting on the client side?

Server-side rate limiting is fundamental for API protection, but client-side implementation also provides value. Proactively controlling request frequency is more efficient and reduces network load compared to waiting for 429 responses and then retrying. Especially in batch processing scenarios with many API calls, client-side rate limiting maintains stable throughput. An adaptive approach that dynamically adjusts rates using `Retry-After` and `X-RateLimit-Remaining` headers is also effective.

---


## Recommended Next Guides

- Refer to other guides in the same category

---

## References
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
3. Node.js Documentation. "Stream." nodejs.org.
4. Martin Fowler. "Circuit Breaker." martinfowler.com.
5. Redis Documentation. "Distributed Locks." redis.io.
