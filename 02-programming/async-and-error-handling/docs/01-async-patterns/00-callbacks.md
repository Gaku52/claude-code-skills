# コールバック

> コールバックは非同期処理の最も原始的なパターン。Node.jsのerror-firstコールバック、コールバック地獄の問題、そしてPromiseへの進化を理解する。

## この章で学ぶこと

- [ ] コールバックの仕組みと使い方を理解する
- [ ] コールバック地獄の問題と原因を把握する
- [ ] error-first パターンの意味を学ぶ
- [ ] 各言語におけるコールバックの実装を比較する
- [ ] コールバックから Promise への移行パターンを習得する

---

## 1. コールバックの基本

### 1.1 コールバックとは

```
コールバック = 「処理が完了したら呼んでね」と渡す関数

  同期:
    const result = readFile("data.txt");
    console.log(result);

  非同期（コールバック）:
    readFile("data.txt", (error, result) => {
      console.log(result);
    });
    // readFile は即座に戻る。結果は後でコールバックに届く

コールバックの分類:
  1. 同期コールバック: map, filter, sort など（即座に実行される）
  2. 非同期コールバック: I/O完了後に呼ばれる（setTimeout, fs.readFileなど）
```

### 1.2 同期コールバック vs 非同期コールバック

```javascript
// === 同期コールバック ===
// 関数の引数として渡され、その場で即座に実行される

// Array.map: 各要素に対してコールバックを適用
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map((n) => n * 2);  // [2, 4, 6, 8, 10]

// Array.filter: コールバックがtrueを返す要素だけ残す
const evens = numbers.filter((n) => n % 2 === 0);  // [2, 4]

// Array.reduce: 累積値を計算
const sum = numbers.reduce((acc, n) => acc + n, 0);  // 15

// Array.sort: コールバックで比較ロジックを注入
const users = [
  { name: "田中", age: 30 },
  { name: "山田", age: 25 },
  { name: "鈴木", age: 35 },
];
users.sort((a, b) => a.age - b.age);
// [{ name: "山田", age: 25 }, { name: "田中", age: 30 }, { name: "鈴木", age: 35 }]

// Array.forEach: 各要素に副作用を実行
numbers.forEach((n) => {
  console.log(n);
});

// Array.find: 条件に合う最初の要素を返す
const firstEven = numbers.find((n) => n % 2 === 0);  // 2

// これらは全て「同期コールバック」
// → 関数が返る時点で全ての処理が完了している
```

```javascript
// === 非同期コールバック ===
// 関数に渡され、I/O完了後や一定時間後に実行される

const fs = require('fs');

// Node.js ファイル読み込み
console.log('1. 読み込み開始');

fs.readFile('/path/to/file', 'utf8', (err, data) => {
  // この関数は「後で」呼ばれる（ファイル読み込み完了時）
  console.log('3. ファイル読み込み完了:', data);
});

console.log('2. 読み込み命令発行後（まだ完了していない）');

// 出力順序:
// 1. 読み込み開始
// 2. 読み込み命令発行後（まだ完了していない）
// 3. ファイル読み込み完了: （ファイルの内容）
```

### 1.3 イベントリスナーとしてのコールバック

```javascript
// ブラウザ: イベントリスナー
document.getElementById('btn').addEventListener('click', (event) => {
  console.log('Clicked!', event.target);
});

// 複数のイベントを登録
const button = document.getElementById('submit');

button.addEventListener('click', handleClick);
button.addEventListener('mouseenter', handleHover);
button.addEventListener('mouseleave', handleLeave);

function handleClick(event) {
  event.preventDefault();
  console.log('Button clicked');
}

function handleHover(event) {
  event.target.style.backgroundColor = '#f0f0f0';
}

function handleLeave(event) {
  event.target.style.backgroundColor = '';
}

// イベントリスナーの解除
button.removeEventListener('click', handleClick);
```

### 1.4 タイマーコールバック

```javascript
// setTimeout: 指定時間後に1回実行
setTimeout(() => {
  console.log('3秒後に実行');
}, 3000);

// setInterval: 指定間隔で繰り返し実行
const intervalId = setInterval(() => {
  console.log('1秒ごとに実行');
}, 1000);

// 停止
setTimeout(() => {
  clearInterval(intervalId);
  console.log('タイマー停止');
}, 5000);

// requestAnimationFrame: 描画フレームごとに実行（ブラウザ）
function animate(timestamp) {
  // アニメーション処理
  updatePosition(timestamp);
  render();

  // 次のフレームを要求
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
```

---

## 2. Node.js の error-first コールバック

### 2.1 基本パターン

```
Node.js の規約（error-first callback）:
  callback(error, result)

  → 第1引数: エラー（成功時は null）
  → 第2引数: 結果（エラー時は undefined）

  利点:
  - エラーチェックが統一的
  - エラーを無視しにくい（第1引数を見る習慣）

  問題:
  - 毎回 if (err) のチェックが必要
  - 型安全性がない（any）
  - ネストが深くなりやすい
```

```javascript
const fs = require('fs');

// error-first コールバックの基本
fs.readFile('/path/to/file', 'utf8', (err, data) => {
  if (err) {
    console.error('Error:', err.message);
    return;
  }
  console.log('Data:', data);
});

// 書き込み
fs.writeFile('/path/to/output', 'Hello, World!', 'utf8', (err) => {
  if (err) {
    console.error('Write failed:', err.message);
    return;
  }
  console.log('File written successfully');
});

// ディレクトリ読み取り
fs.readdir('/path/to/dir', (err, files) => {
  if (err) {
    console.error('Failed to read directory:', err.message);
    return;
  }
  console.log('Files:', files);
});
```

### 2.2 error-first パターンの実装

```javascript
// error-first の自作関数
function readJsonFile(path, callback) {
  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      callback(err, null);
      return;
    }
    try {
      const parsed = JSON.parse(data);
      callback(null, parsed);
    } catch (parseError) {
      callback(parseError, null);
    }
  });
}

// 使用
readJsonFile('config.json', (err, config) => {
  if (err) {
    console.error('Failed to read config:', err.message);
    return;
  }
  console.log('Config loaded:', config);
});
```

```javascript
// 非同期データベース操作（コールバックスタイル）
function getUser(userId, callback) {
  db.query('SELECT * FROM users WHERE id = ?', [userId], (err, rows) => {
    if (err) {
      callback(err, null);
      return;
    }
    if (rows.length === 0) {
      callback(new Error(`User ${userId} not found`), null);
      return;
    }
    callback(null, rows[0]);
  });
}

function getUserOrders(userId, callback) {
  db.query('SELECT * FROM orders WHERE user_id = ?', [userId], (err, rows) => {
    if (err) {
      callback(err, null);
      return;
    }
    callback(null, rows);
  });
}

// HTTPリクエスト（コールバックスタイル）
const http = require('http');

function fetchJSON(url, callback) {
  http.get(url, (res) => {
    let data = '';

    res.on('data', (chunk) => {
      data += chunk;
    });

    res.on('error', (err) => {
      callback(err, null);
    });

    res.on('end', () => {
      try {
        const parsed = JSON.parse(data);
        callback(null, parsed);
      } catch (parseErr) {
        callback(parseErr, null);
      }
    });
  }).on('error', (err) => {
    callback(err, null);
  });
}

// 使用
fetchJSON('http://api.example.com/users/1', (err, user) => {
  if (err) {
    console.error('Failed to fetch user:', err.message);
    return;
  }
  console.log('User:', user);
});
```

### 2.3 コールバックの設計パターン

```javascript
// パターン1: 設定オブジェクトとコールバック
function connectToDatabase(options, callback) {
  const { host, port, database, user, password } = options;

  const connection = new DatabaseConnection({
    host, port, database, user, password
  });

  connection.connect((err) => {
    if (err) {
      callback(err, null);
      return;
    }

    // 接続成功: マイグレーションチェック
    connection.checkMigrations((err, needsMigration) => {
      if (err) {
        connection.close();
        callback(err, null);
        return;
      }

      if (needsMigration) {
        connection.runMigrations((err) => {
          if (err) {
            connection.close();
            callback(err, null);
            return;
          }
          callback(null, connection);
        });
      } else {
        callback(null, connection);
      }
    });
  });
}

// パターン2: EventEmitter スタイル
const EventEmitter = require('events');

class FileProcessor extends EventEmitter {
  process(filePath) {
    fs.readFile(filePath, 'utf8', (err, data) => {
      if (err) {
        this.emit('error', err);
        return;
      }

      this.emit('data', data);

      const lines = data.split('\n');
      this.emit('line-count', lines.length);

      for (const line of lines) {
        this.emit('line', line);
      }

      this.emit('complete', { totalLines: lines.length });
    });
  }
}

// 使用
const processor = new FileProcessor();

processor.on('data', (data) => {
  console.log(`Loaded ${data.length} bytes`);
});

processor.on('line', (line) => {
  // 各行を処理
});

processor.on('complete', ({ totalLines }) => {
  console.log(`Processed ${totalLines} lines`);
});

processor.on('error', (err) => {
  console.error('Error:', err.message);
});

processor.process('large-file.txt');
```

---

## 3. コールバック地獄（Callback Hell）

### 3.1 問題の本質

```javascript
// ❌ コールバック地獄: ネストが深くなり可読性が崩壊
getUser(userId, (err, user) => {
  if (err) { handleError(err); return; }
  getOrders(user.id, (err, orders) => {
    if (err) { handleError(err); return; }
    getOrderDetails(orders[0].id, (err, details) => {
      if (err) { handleError(err); return; }
      getShippingInfo(details.shippingId, (err, shipping) => {
        if (err) { handleError(err); return; }
        getTrackingInfo(shipping.trackingId, (err, tracking) => {
          if (err) { handleError(err); return; }
          // ここまで5段階のネスト
          console.log(tracking);
        });
      });
    });
  });
});

// 問題:
// 1. 横に広がる「ピラミッド型」コード
// 2. エラーハンドリングの重複
// 3. 変数スコープの管理困難
// 4. 処理の流れが追いにくい
// 5. テストが書きにくい
// 6. 制御フロー（条件分岐、ループ）の実装が複雑
```

### 3.2 実務で遭遇する典型的なコールバック地獄

```javascript
// ECサイトの注文処理（コールバック地獄版）
function processOrder(userId, cartId, paymentInfo, callback) {
  // 1. ユーザー認証
  authenticateUser(userId, (err, user) => {
    if (err) { callback(err); return; }

    // 2. カート取得
    getCart(cartId, (err, cart) => {
      if (err) { callback(err); return; }

      // 3. 在庫チェック
      checkInventory(cart.items, (err, availability) => {
        if (err) { callback(err); return; }

        if (!availability.allAvailable) {
          callback(new Error('Some items are out of stock'));
          return;
        }

        // 4. 金額計算
        calculateTotal(cart, user, (err, total) => {
          if (err) { callback(err); return; }

          // 5. 支払い処理
          processPayment(paymentInfo, total, (err, paymentResult) => {
            if (err) {
              // 支払い失敗時の在庫ロールバック
              releaseInventory(cart.items, (rollbackErr) => {
                if (rollbackErr) {
                  console.error('Rollback failed:', rollbackErr);
                }
                callback(err);
              });
              return;
            }

            // 6. 注文作成
            createOrder(user, cart, paymentResult, (err, order) => {
              if (err) {
                // 注文作成失敗時の支払い取り消し
                refundPayment(paymentResult.id, (refundErr) => {
                  if (refundErr) {
                    console.error('Refund failed:', refundErr);
                  }
                  callback(err);
                });
                return;
              }

              // 7. 通知送信
              sendOrderConfirmation(user.email, order, (err) => {
                if (err) {
                  console.error('Email failed:', err);
                  // メール失敗は無視して成功扱い
                }
                callback(null, order);
              });
            });
          });
        });
      });
    });
  });
}
```

### 3.3 改善テクニック1: 名前付き関数で分離

```javascript
// やや改善: 名前付き関数で分離
function handleTracking(err, tracking) {
  if (err) { handleError(err); return; }
  console.log(tracking);
}

function handleShipping(err, shipping) {
  if (err) { handleError(err); return; }
  getTrackingInfo(shipping.trackingId, handleTracking);
}

function handleDetails(err, details) {
  if (err) { handleError(err); return; }
  getShippingInfo(details.shippingId, handleShipping);
}

function handleOrders(err, orders) {
  if (err) { handleError(err); return; }
  getOrderDetails(orders[0].id, handleDetails);
}

function handleUser(err, user) {
  if (err) { handleError(err); return; }
  getOrders(user.id, handleOrders);
}

// エントリポイント
getUser(userId, handleUser);

// 改善点: ネストが浅い
// 残る問題: 関数が逆順で定義され、流れが追いにくい
```

### 3.4 改善テクニック2: async ライブラリ

```javascript
// async.js ライブラリを使ったフロー制御
const async = require('async');

// async.waterfall: 直列実行（前の結果を次に渡す）
async.waterfall([
  // Step 1: ユーザー取得
  (cb) => getUser(userId, cb),

  // Step 2: 注文取得（userは前のステップの結果）
  (user, cb) => getOrders(user.id, (err, orders) => {
    cb(err, user, orders);
  }),

  // Step 3: 注文詳細取得
  (user, orders, cb) => getOrderDetails(orders[0].id, (err, details) => {
    cb(err, user, orders, details);
  }),

  // Step 4: 配送情報取得
  (user, orders, details, cb) => {
    getShippingInfo(details.shippingId, cb);
  },
], (err, shippingInfo) => {
  if (err) {
    handleError(err);
    return;
  }
  console.log('Shipping:', shippingInfo);
});

// async.parallel: 並行実行
async.parallel({
  users: (cb) => fetchUsers(cb),
  orders: (cb) => fetchOrders(cb),
  products: (cb) => fetchProducts(cb),
}, (err, results) => {
  if (err) {
    handleError(err);
    return;
  }
  console.log(results.users, results.orders, results.products);
});

// async.series: 直列実行（結果は別々に）
async.series([
  (cb) => createBackup(cb),
  (cb) => runMigrations(cb),
  (cb) => verifyData(cb),
], (err, results) => {
  if (err) {
    console.error('Pipeline failed:', err);
    return;
  }
  console.log('All steps completed');
});

// async.eachLimit: 並行数制限付き反復
const urls = ['url1', 'url2', 'url3', /* ... */];

async.eachLimit(urls, 5, (url, cb) => {
  fetchAndProcess(url, cb);
}, (err) => {
  if (err) {
    console.error('Processing failed:', err);
    return;
  }
  console.log('All URLs processed');
});
```

### 3.5 改善テクニック3: 制御フロー抽象化

```javascript
// 独自のフロー制御関数
function waterfall(tasks, finalCallback) {
  let index = 0;

  function next(err, ...args) {
    if (err) {
      finalCallback(err);
      return;
    }

    if (index >= tasks.length) {
      finalCallback(null, ...args);
      return;
    }

    const task = tasks[index++];
    try {
      task(...args, next);
    } catch (e) {
      finalCallback(e);
    }
  }

  next(null);
}

// 使用
waterfall([
  (cb) => getUser(userId, cb),
  (user, cb) => getOrders(user.id, cb),
  (orders, cb) => getOrderDetails(orders[0].id, cb),
], (err, details) => {
  if (err) {
    console.error(err);
    return;
  }
  console.log('Details:', details);
});

// 並行実行関数
function parallel(tasks, finalCallback) {
  const results = {};
  let completed = 0;
  let hasError = false;
  const keys = Object.keys(tasks);

  keys.forEach((key) => {
    tasks[key]((err, result) => {
      if (hasError) return;

      if (err) {
        hasError = true;
        finalCallback(err);
        return;
      }

      results[key] = result;
      completed++;

      if (completed === keys.length) {
        finalCallback(null, results);
      }
    });
  });
}
```

---

## 4. 各言語のコールバックパターン

### 4.1 Python のコールバック

```python
import threading
import time
from typing import Callable, Optional, Any

# Python でのコールバックパターン

# 基本的なコールバック
def fetch_data(url: str, on_success: Callable, on_error: Callable) -> None:
    """非同期的にデータを取得（スレッドベース）"""
    def worker():
        try:
            import urllib.request
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            on_success(data)
        except Exception as e:
            on_error(e)

    thread = threading.Thread(target=worker)
    thread.start()

# 使用
def handle_success(data):
    print(f"Received: {data[:100]}...")

def handle_error(error):
    print(f"Error: {error}")

fetch_data("https://api.example.com/data", handle_success, handle_error)

# デコレータとしてのコールバック
def retry(max_retries: int = 3, delay: float = 1.0):
    """リトライデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator

@retry(max_retries=3, delay=0.5)
def unreliable_api_call():
    """不安定なAPI呼び出し"""
    import random
    if random.random() < 0.5:
        raise ConnectionError("Connection failed")
    return {"status": "ok"}

# コンテキストマネージャー + コールバック
class TimedOperation:
    """操作の時間を計測し、コールバックで報告"""
    def __init__(self, name: str, on_complete: Callable[[str, float], None]):
        self.name = name
        self.on_complete = on_complete
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.on_complete(self.name, elapsed)
        return False

# 使用
def log_timing(name: str, elapsed: float):
    print(f"[TIMING] {name}: {elapsed:.3f}s")

with TimedOperation("data_processing", log_timing):
    time.sleep(0.5)  # 何らかの処理
# 出力: [TIMING] data_processing: 0.501s
```

### 4.2 Rust のコールバック

```rust
use std::thread;
use std::sync::mpsc;

// Rust: クロージャをコールバックとして使用
// 所有権と寿命の制約がある

// 基本的なコールバック
fn process_async<F>(data: Vec<i32>, callback: F)
where
    F: FnOnce(Vec<i32>) + Send + 'static,
{
    thread::spawn(move || {
        let result: Vec<i32> = data.iter().map(|x| x * 2).collect();
        callback(result);
    });
}

// 使用
fn main() {
    process_async(vec![1, 2, 3, 4, 5], |result| {
        println!("Result: {:?}", result);
    });

    thread::sleep(std::time::Duration::from_secs(1));
}

// Result型でエラーハンドリング
fn fetch_data<F>(url: &str, callback: F)
where
    F: FnOnce(Result<String, Box<dyn std::error::Error>>) + Send + 'static,
{
    let url = url.to_string();
    thread::spawn(move || {
        let result = reqwest::blocking::get(&url)
            .and_then(|resp| resp.text());
        match result {
            Ok(body) => callback(Ok(body)),
            Err(e) => callback(Err(Box::new(e))),
        }
    });
}

// トレイトオブジェクトを使ったコールバック
trait EventHandler: Send {
    fn on_data(&self, data: &[u8]);
    fn on_error(&self, error: &str);
    fn on_complete(&self);
}

struct DataProcessor {
    handler: Box<dyn EventHandler>,
}

impl DataProcessor {
    fn new(handler: Box<dyn EventHandler>) -> Self {
        DataProcessor { handler }
    }

    fn process(&self, data: &[u8]) {
        if data.is_empty() {
            self.handler.on_error("Empty data");
            return;
        }
        self.handler.on_data(data);
        self.handler.on_complete();
    }
}
```

### 4.3 Go のコールバック

```go
package main

import (
    "fmt"
    "io/ioutil"
    "net/http"
    "time"
)

// Go: 関数を第一級市民として使用
// ただし Go では goroutine + channel の方がイディオマティック

// コールバック型の定義
type ResultCallback func(data []byte, err error)
type ProgressCallback func(current, total int)

// コールバック付きHTTPリクエスト
func fetchWithCallback(url string, callback ResultCallback) {
    go func() {
        resp, err := http.Get(url)
        if err != nil {
            callback(nil, err)
            return
        }
        defer resp.Body.Close()

        body, err := ioutil.ReadAll(resp.Body)
        if err != nil {
            callback(nil, err)
            return
        }
        callback(body, nil)
    }()
}

// プログレス付きダウンロード
func downloadWithProgress(url string, progress ProgressCallback, done ResultCallback) {
    go func() {
        resp, err := http.Get(url)
        if err != nil {
            done(nil, err)
            return
        }
        defer resp.Body.Close()

        contentLength := int(resp.ContentLength)
        data := make([]byte, 0, contentLength)
        buf := make([]byte, 4096)
        received := 0

        for {
            n, err := resp.Body.Read(buf)
            if n > 0 {
                data = append(data, buf[:n]...)
                received += n
                progress(received, contentLength)
            }
            if err != nil {
                break
            }
        }

        done(data, nil)
    }()
}

func main() {
    // 使用例
    fetchWithCallback("https://api.example.com/data", func(data []byte, err error) {
        if err != nil {
            fmt.Println("Error:", err)
            return
        }
        fmt.Println("Received:", len(data), "bytes")
    })

    // Go のイディオム: channel の方が好ましい
    ch := make(chan []byte, 1)
    errCh := make(chan error, 1)

    go func() {
        resp, err := http.Get("https://api.example.com/data")
        if err != nil {
            errCh <- err
            return
        }
        defer resp.Body.Close()
        body, _ := ioutil.ReadAll(resp.Body)
        ch <- body
    }()

    select {
    case data := <-ch:
        fmt.Println("Received:", len(data), "bytes")
    case err := <-errCh:
        fmt.Println("Error:", err)
    case <-time.After(5 * time.Second):
        fmt.Println("Timeout")
    }
}
```

### 4.4 C# のコールバック

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

// C#: デリゲートとイベントによるコールバック

// デリゲート定義
public delegate void DataCallback(string data);
public delegate void ErrorCallback(Exception error);

public class AsyncFetcher
{
    // イベントベースのコールバック
    public event EventHandler<DataEventArgs> DataReceived;
    public event EventHandler<ErrorEventArgs> ErrorOccurred;
    public event EventHandler Completed;

    // コールバックを受け取るメソッド
    public void FetchData(string url, Action<string> onSuccess, Action<Exception> onError)
    {
        Task.Run(async () =>
        {
            try
            {
                using var client = new HttpClient();
                var data = await client.GetStringAsync(url);
                onSuccess(data);
            }
            catch (Exception ex)
            {
                onError(ex);
            }
        });
    }

    // イベントを発火するメソッド
    public async void FetchDataEvent(string url)
    {
        try
        {
            using var client = new HttpClient();
            var data = await client.GetStringAsync(url);
            DataReceived?.Invoke(this, new DataEventArgs(data));
            Completed?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new ErrorEventArgs(ex));
        }
    }
}

// 使用
var fetcher = new AsyncFetcher();

// ラムダ式でコールバック
fetcher.FetchData(
    "https://api.example.com/data",
    data => Console.WriteLine($"Success: {data.Length} chars"),
    error => Console.WriteLine($"Error: {error.Message}")
);

// イベントでコールバック
fetcher.DataReceived += (sender, args) =>
{
    Console.WriteLine($"Data received: {args.Data.Length} chars");
};
fetcher.ErrorOccurred += (sender, args) =>
{
    Console.WriteLine($"Error: {args.Error.Message}");
};
fetcher.FetchDataEvent("https://api.example.com/data");
```

---

## 5. 高階関数としてのコールバック

### 5.1 関数合成とコールバック

```javascript
// コールバックは「高階関数」の一種
// 「何をするか」を引数として渡す

// 戦略パターン: コールバックでアルゴリズムを注入
function sortUsers(users, comparator) {
  return [...users].sort(comparator);
}

const users = [
  { name: "田中", age: 30, score: 85 },
  { name: "山田", age: 25, score: 92 },
  { name: "鈴木", age: 35, score: 78 },
];

// 年齢順
const byAge = sortUsers(users, (a, b) => a.age - b.age);

// スコア順（降順）
const byScore = sortUsers(users, (a, b) => b.score - a.score);

// 名前順
const byName = sortUsers(users, (a, b) => a.name.localeCompare(b.name, 'ja'));
```

```typescript
// ミドルウェアパターン（Express スタイル）
type Middleware = (req: Request, res: Response, next: () => void) => void;

class Router {
  private middlewares: Middleware[] = [];

  use(middleware: Middleware): void {
    this.middlewares.push(middleware);
  }

  handle(req: Request, res: Response): void {
    let index = 0;

    const next = () => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        middleware(req, res, next);
      }
    };

    next();
  }
}

// 使用
const router = new Router();

// ログミドルウェア
router.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

// 認証ミドルウェア
router.use((req, res, next) => {
  if (!req.headers.authorization) {
    res.status(401).send('Unauthorized');
    return;
  }
  next();
});

// ハンドラー
router.use((req, res, next) => {
  res.json({ message: 'Hello, World!' });
});
```

### 5.2 コールバックのカリー化

```typescript
// カリー化されたコールバック
function createLogger(prefix: string) {
  return function(message: string) {
    console.log(`[${prefix}] ${new Date().toISOString()} ${message}`);
  };
}

const infoLog = createLogger('INFO');
const errorLog = createLogger('ERROR');
const debugLog = createLogger('DEBUG');

infoLog('Server started');    // [INFO] 2024-01-01T00:00:00.000Z Server started
errorLog('Connection lost');  // [ERROR] 2024-01-01T00:00:00.000Z Connection lost

// コールバックファクトリ
function createRetryCallback<T>(
  fn: (callback: (err: Error | null, result?: T) => void) => void,
  maxRetries: number,
  delay: number,
): Promise<T> {
  return new Promise((resolve, reject) => {
    let attempts = 0;

    function attempt() {
      fn((err, result) => {
        if (!err) {
          resolve(result!);
          return;
        }

        attempts++;
        if (attempts >= maxRetries) {
          reject(err);
          return;
        }

        setTimeout(attempt, delay * Math.pow(2, attempts));
      });
    }

    attempt();
  });
}

// 使用
const result = await createRetryCallback(
  (cb) => fetchData('https://api.example.com/data', cb),
  3,
  1000,
);
```

---

## 6. コールバックの落とし穴

### 6.1 Zalgo問題（同期・非同期の混在）

```javascript
// ❌ Zalgo: 条件によって同期・非同期が変わる
function getData(cache, key, callback) {
  if (cache[key]) {
    // ⚠️ 同期的にコールバックを呼んでいる
    callback(null, cache[key]);
  } else {
    // 非同期的にコールバックを呼んでいる
    db.query(key, (err, data) => {
      if (!err) cache[key] = data;
      callback(err, data);
    });
  }
}

// 問題: 呼び出し側のコードの実行順序が予測不能
let result;
getData(cache, 'key', (err, data) => {
  result = data;
});
// result が設定されているかどうかは cache の状態に依存
// → 非常にバグを生みやすい

// ✅ 修正: 常に非同期にする
function getDataFixed(cache, key, callback) {
  if (cache[key]) {
    // process.nextTick で非同期化
    process.nextTick(() => callback(null, cache[key]));
  } else {
    db.query(key, (err, data) => {
      if (!err) cache[key] = data;
      callback(err, data);
    });
  }
}

// ✅ より良い修正: queueMicrotask（ブラウザ/Node.js共通）
function getDataBetter(cache, key, callback) {
  if (cache[key]) {
    queueMicrotask(() => callback(null, cache[key]));
  } else {
    db.query(key, (err, data) => {
      if (!err) cache[key] = data;
      callback(err, data);
    });
  }
}
```

### 6.2 コールバックの二重呼び出し

```javascript
// ❌ コールバックが2回呼ばれる可能性
function processFile(path, callback) {
  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      callback(err);
      // ⚠️ return を忘れている！
    }
    // エラー時もここが実行される
    const processed = transform(data); // data は undefined → エラー
    callback(null, processed);
  });
}

// ✅ 修正: return で早期脱出
function processFileFixed(path, callback) {
  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      callback(err);
      return; // ← 重要
    }
    try {
      const processed = transform(data);
      callback(null, processed);
    } catch (transformErr) {
      callback(transformErr);
    }
  });
}

// ✅ より安全: once ラッパー
function once(fn) {
  let called = false;
  return function(...args) {
    if (called) {
      console.warn('Callback called more than once');
      return;
    }
    called = true;
    fn(...args);
  };
}

function processFileSafe(path, callback) {
  const safeCallback = once(callback);

  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      safeCallback(err);
      return;
    }
    try {
      const processed = transform(data);
      safeCallback(null, processed);
    } catch (transformErr) {
      safeCallback(transformErr);
    }
  });
}
```

### 6.3 エラーの飲み込み

```javascript
// ❌ コールバック内のエラーが外に伝播しない
try {
  getUser(userId, (err, user) => {
    if (err) throw err; // ← これはキャッチされない！
    // コールバックは別のコールスタックで実行されるため
    // try-catch は効かない
  });
} catch (err) {
  // ここには到達しない
  console.error(err);
}

// ✅ コールバック内でエラーハンドリング
getUser(userId, (err, user) => {
  if (err) {
    console.error('Error:', err.message);
    // エラーリカバリーやアラート
    return;
  }
  // 正常処理
});

// ✅ ドメインでエラーをキャッチ（非推奨だが参考として）
const domain = require('domain');
const d = domain.create();

d.on('error', (err) => {
  console.error('Domain caught:', err);
});

d.run(() => {
  getUser(userId, (err, user) => {
    if (err) throw err; // ドメインがキャッチ
  });
});
```

### 6.4 メモリリーク

```javascript
// ❌ クロージャによるメモリリーク
function createConnection(config) {
  const connection = new DatabaseConnection(config);
  const largeBuffer = Buffer.alloc(100 * 1024 * 1024); // 100MB

  return {
    query(sql, callback) {
      // largeBuffer はクロージャで保持される（使っていなくても）
      connection.execute(sql, (err, rows) => {
        callback(err, rows);
      });
    },
    close() {
      connection.close();
    }
  };
}

// ✅ 修正: 不要な参照を持たない
function createConnectionFixed(config) {
  const connection = new DatabaseConnection(config);

  // largeBuffer は関数スコープ外
  function processLargeData() {
    const largeBuffer = Buffer.alloc(100 * 1024 * 1024);
    // 使用後に参照が消える
    return transform(largeBuffer);
  }

  return {
    query(sql, callback) {
      connection.execute(sql, callback);
    },
    close() {
      connection.close();
    }
  };
}

// ❌ イベントリスナーの累積
function setupHandler(element) {
  // 呼ばれるたびにリスナーが追加される
  element.addEventListener('click', () => {
    doSomething();
  });
}

// ✅ 修正: 既存のリスナーを解除
function setupHandlerFixed(element) {
  // 名前付き関数で参照を保持
  if (element._clickHandler) {
    element.removeEventListener('click', element._clickHandler);
  }

  element._clickHandler = () => {
    doSomething();
  };
  element.addEventListener('click', element._clickHandler);
}

// ✅ さらに良い: AbortController を使用
function setupHandlerModern(element) {
  const controller = new AbortController();

  element.addEventListener('click', () => {
    doSomething();
  }, { signal: controller.signal });

  // クリーンアップ
  return () => controller.abort();
}
```

---

## 7. コールバックから Promise への移行

### 7.1 手動 Promise 化

```javascript
// 手動でPromise化
function readFilePromise(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) reject(err);
      else resolve(data);
    });
  });
}

// 使用
readFilePromise('file.txt')
  .then(data => console.log(data))
  .catch(err => console.error(err));

// async/await版
async function main() {
  try {
    const data = await readFilePromise('file.txt');
    console.log(data);
  } catch (err) {
    console.error(err);
  }
}
```

### 7.2 util.promisify

```javascript
// Node.js: util.promisify でコールバックを Promise に変換
const { promisify } = require('util');
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);
const readdir = promisify(fs.readdir);

// コールバック版
fs.readFile('file.txt', 'utf8', (err, data) => {
  if (err) throw err;
  console.log(data);
});

// Promise版
readFile('file.txt', 'utf8')
  .then(data => console.log(data))
  .catch(err => console.error(err));

// async/await版
async function main() {
  try {
    const data = await readFile('file.txt', 'utf8');
    console.log(data);
  } catch (err) {
    console.error(err);
  }
}

// fs/promises（Node.js 14+）
const fsPromises = require('fs/promises');

async function modernFileOps() {
  const data = await fsPromises.readFile('file.txt', 'utf8');
  await fsPromises.writeFile('output.txt', data.toUpperCase());
  const files = await fsPromises.readdir('.');
  console.log(files);
}
```

### 7.3 汎用 promisify 関数

```typescript
// 汎用的な promisify 実装
function promisify<T>(
  fn: (...args: [...any[], (err: Error | null, result: T) => void]) => void
): (...args: any[]) => Promise<T> {
  return function (...args: any[]): Promise<T> {
    return new Promise((resolve, reject) => {
      fn(...args, (err: Error | null, result: T) => {
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      });
    });
  };
}

// 複数の戻り値を持つコールバックの promisify
function promisifyMultiResult(fn) {
  return function (...args) {
    return new Promise((resolve, reject) => {
      fn(...args, (err, ...results) => {
        if (err) {
          reject(err);
        } else {
          resolve(results);
        }
      });
    });
  };
}

// EventEmitter を Promise に変換
function waitForEvent(emitter, eventName, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout waiting for event: ${eventName}`));
    }, timeout);

    emitter.once(eventName, (...args) => {
      clearTimeout(timer);
      resolve(args.length === 1 ? args[0] : args);
    });

    emitter.once('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

// 使用
const server = createServer();
const connection = await waitForEvent(server, 'connection', 10000);
```

### 7.4 コールバック API のラッパークラス

```typescript
// レガシーなコールバック API をモダンにラップ
class DatabaseWrapper {
  private db: LegacyDatabase;

  constructor(connectionString: string) {
    this.db = new LegacyDatabase(connectionString);
  }

  // コールバック API を Promise でラップ
  query<T>(sql: string, params?: any[]): Promise<T[]> {
    return new Promise((resolve, reject) => {
      this.db.query(sql, params || [], (err: Error | null, rows: T[]) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  // トランザクション
  async transaction<T>(fn: (tx: TransactionContext) => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.db.beginTransaction((err: Error | null, tx: any) => {
        if (err) {
          reject(err);
          return;
        }

        const context = new TransactionContext(tx);

        fn(context)
          .then((result) => {
            tx.commit((err: Error | null) => {
              if (err) reject(err);
              else resolve(result);
            });
          })
          .catch((error) => {
            tx.rollback((rollbackErr: Error | null) => {
              if (rollbackErr) {
                console.error('Rollback failed:', rollbackErr);
              }
              reject(error);
            });
          });
      });
    });
  }

  // 接続プール
  getConnection(): Promise<Connection> {
    return new Promise((resolve, reject) => {
      this.db.getConnection((err: Error | null, conn: any) => {
        if (err) reject(err);
        else resolve(new Connection(conn));
      });
    });
  }
}

// 使用（クリーンなasync/await）
const db = new DatabaseWrapper('postgres://localhost/mydb');

async function getUserOrders(userId: string) {
  const [user] = await db.query<User>('SELECT * FROM users WHERE id = $1', [userId]);
  if (!user) throw new Error('User not found');

  const orders = await db.query<Order>(
    'SELECT * FROM orders WHERE user_id = $1',
    [userId]
  );

  return { user, orders };
}
```

---

## 8. 実務でのベストプラクティス

### 8.1 コールバック設計のルール

```
1. 常に error-first パターンを使う
   callback(err, result) の形式を守る

2. コールバックは常に非同期で呼ぶ
   Zalgo問題を避けるために process.nextTick / queueMicrotask を使う

3. コールバックは1回だけ呼ぶ
   once() ラッパーで二重呼び出しを防止

4. エラーは必ずコールバック経由で伝える
   throw ではなく callback(err) を使う

5. ネストは3段階以下に抑える
   名前付き関数に分離、または async ライブラリを使う

6. 可能であれば Promise / async-await に移行する
   新規コードではコールバックを避ける
```

### 8.2 移行戦略

```typescript
// 段階的な移行戦略

// Step 1: 既存のコールバック API をラップ
const readFileAsync = promisify(fs.readFile);

// Step 2: 新しい関数は async/await で書く
async function loadConfig(): Promise<Config> {
  const data = await readFileAsync('config.json', 'utf8');
  return JSON.parse(data);
}

// Step 3: コールバックを受け取る関数をデュアルインターフェースにする
function getData(
  key: string,
  callback?: (err: Error | null, data?: Data) => void,
): Promise<Data> | void {
  const promise = getDataInternal(key);

  if (callback) {
    promise
      .then(data => callback(null, data))
      .catch(err => callback(err));
    return;
  }

  return promise;
}

// コールバックスタイルで使用
getData('key', (err, data) => {
  if (err) handleError(err);
  else console.log(data);
});

// Promise スタイルで使用
const data = await getData('key');
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| コールバック | 完了時に呼ばれる関数 |
| error-first | (err, result) の規約 |
| コールバック地獄 | ネスト深化 → Promise で解決 |
| 同期コールバック | map, filter, sort, reduce |
| 非同期コールバック | I/O, タイマー, イベント |
| Zalgo問題 | 同期/非同期の混在を避ける |
| 二重呼び出し | once() ラッパーで防止 |
| メモリリーク | クロージャの参照に注意 |

### コールバックの進化

```
コールバック（1990年代〜）
  ↓ 問題: コールバック地獄
Promise（ES2015 / 2015年〜）
  ↓ 改善: チェーン可能、エラー伝播
async/await（ES2017 / 2017年〜）
  ↓ 改善: 同期的な記述
Reactive Streams（RxJS等）
  ↓ 拡張: ストリーム処理
AsyncIterator / for-await-of（ES2018）
  → 非同期イテレーション
```

---

## 次に読むべきガイド
→ [[01-promises.md]] — Promise

---

## 参考文献
1. Node.js Documentation. "Asynchronous Programming."
2. Ogden, M. "Callback Hell." callbackhell.com.
3. Havoc Pennington. "Don't Release Zalgo!" blog.izs.me.
4. Casciaro, M. & Mammoliti, L. "Node.js Design Patterns." Packt Publishing, 2020.
5. Mozilla Developer Network. "Callback function." MDN Web Docs.
6. Caolan McMahon. "async.js." github.com/caolan/async.
7. Node.js API. "util.promisify." nodejs.org.
8. Rust Documentation. "Closures." doc.rust-lang.org.
