# コールバック

> コールバックは非同期処理の最も原始的なパターン。Node.jsのerror-firstコールバック、コールバック地獄の問題、そしてPromiseへの進化を理解する。

## この章で学ぶこと

- [ ] コールバックの仕組みと使い方を理解する
- [ ] コールバック地獄の問題と原因を把握する
- [ ] error-first パターンの意味を学ぶ

---

## 1. コールバックの基本

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
```

```javascript
// Node.js: error-first コールバック
const fs = require('fs');

// error-first: 第1引数がエラー、第2引数が結果
fs.readFile('/path/to/file', 'utf8', (err, data) => {
  if (err) {
    console.error('Error:', err.message);
    return;
  }
  console.log('Data:', data);
});

// イベントリスナー（ブラウザ）
document.getElementById('btn').addEventListener('click', (event) => {
  console.log('Clicked!', event.target);
});

// setTimeout（タイマー）
setTimeout(() => {
  console.log('3秒後に実行');
}, 3000);
```

---

## 2. コールバック地獄（Callback Hell）

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
```

### 改善: 名前付き関数で分離

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

// 改善されるがまだ冗長 → Promise/async-await で根本解決
```

---

## 3. 高階関数としてのコールバック

```javascript
// コールバックは「高階関数」の一種
// 「何をするか」を引数として渡す

// map: 各要素に対してコールバックを適用
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map((n) => n * 2);  // [2, 4, 6, 8, 10]

// filter: コールバックがtrueを返す要素だけ残す
const evens = numbers.filter((n) => n % 2 === 0);  // [2, 4]

// sort: コールバックで比較ロジックを注入
const users = [
  { name: "田中", age: 30 },
  { name: "山田", age: 25 },
];
users.sort((a, b) => a.age - b.age);

// これらは「同期コールバック」
// 非同期コールバックとは区別する
```

---

## 4. error-first パターン

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
```

```javascript
// error-first の実装例
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

---

## 5. コールバックから Promise への移行

```javascript
// Node.js: util.promisify でコールバックを Promise に変換
const { promisify } = require('util');
const readFile = promisify(fs.readFile);

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

// 手動でPromise化
function readFilePromise(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) reject(err);
      else resolve(data);
    });
  });
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| コールバック | 完了時に呼ばれる関数 |
| error-first | (err, result) の規約 |
| コールバック地獄 | ネスト深化 → Promise で解決 |
| 同期コールバック | map, filter, sort |
| 非同期コールバック | I/O, タイマー, イベント |

---

## 次に読むべきガイド
→ [[01-promises.md]] — Promise

---

## 参考文献
1. Node.js Documentation. "Asynchronous Programming."
2. Ogden, M. "Callback Hell." callbackhell.com.
