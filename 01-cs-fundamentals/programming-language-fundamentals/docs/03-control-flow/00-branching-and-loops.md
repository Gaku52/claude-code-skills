# 分岐とループ

> 制御フローはプログラムの「実行順序を変える」仕組み。分岐とループの設計は言語の哲学を反映する。

## この章で学ぶこと

- [ ] 各言語の分岐構文の違いと設計思想を理解する
- [ ] ループの種類と使い分けを把握する
- [ ] 式ベースの制御フローを理解する

---

## 1. 分岐（Branching）

### if 文 vs if 式

```python
# Python: if は文（statement）
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# 三項演算子（式）
grade = "A" if score >= 90 else "B" if score >= 80 else "C"
```

```rust
// Rust: if は式（expression）→ 値を返す
let grade = if score >= 90 {
    "A"
} else if score >= 80 {
    "B"
} else {
    "C"
};  // セミコロンで束縛

// 1行で
let abs_val = if x >= 0 { x } else { -x };
```

```kotlin
// Kotlin: if も when も式
val grade = if (score >= 90) "A" else if (score >= 80) "B" else "C"

val result = when {
    score >= 90 -> "A"
    score >= 80 -> "B"
    else -> "C"
}
```

### switch / match

```javascript
// JavaScript: switch文（fall-through に注意）
switch (day) {
    case "Mon": case "Tue": case "Wed":
    case "Thu": case "Fri":
        type = "Weekday";
        break;      // break 忘れ → fall-through（バグの温床）
    case "Sat": case "Sun":
        type = "Weekend";
        break;
    default:
        type = "Unknown";
}
```

```rust
// Rust: match式（網羅性チェック + パターンマッチ）
let type_str = match day {
    "Mon" | "Tue" | "Wed" | "Thu" | "Fri" => "Weekday",
    "Sat" | "Sun" => "Weekend",
    _ => "Unknown",
};
// 全パターンを網羅しないとコンパイルエラー
// fall-through なし（安全）
```

```go
// Go: switch（break不要、fall-through は明示的）
switch day {
case "Mon", "Tue", "Wed", "Thu", "Fri":
    typeStr = "Weekday"
case "Sat", "Sun":
    typeStr = "Weekend"
default:
    typeStr = "Unknown"
}
// break 不要（自動的に抜ける）
// fallthrough キーワードで明示的に fall-through
```

---

## 2. ループ

### for ループの進化

```c
// C: 古典的な for ループ
for (int i = 0; i < 10; i++) {
    printf("%d\n", i);
}
```

```python
# Python: for-in（イテレータベース）
for i in range(10):
    print(i)

for item in collection:
    print(item)

# enumerate（インデックス付き）
for i, item in enumerate(collection):
    print(f"{i}: {item}")

# zip（並行イテレーション）
for name, age in zip(names, ages):
    print(f"{name}: {age}")
```

```rust
// Rust: for-in（所有権を意識）
for item in &collection {     // 不変借用（コレクション保持）
    println!("{}", item);
}

for item in &mut collection { // 可変借用（要素を変更）
    *item += 1;
}

for item in collection {      // ムーブ（コレクション消費）
    println!("{}", item);
}
// collection はもう使えない

// レンジ
for i in 0..10 {        // 0〜9
    println!("{}", i);
}
for i in 0..=10 {       // 0〜10（inclusive）
    println!("{}", i);
}
```

### while と loop

```rust
// Rust: loop（無限ループ、値を返せる）
let result = loop {
    let input = get_input();
    if input.is_valid() {
        break input.value();  // break で値を返す
    }
};

// while let（パターンマッチ付き）
while let Some(item) = iterator.next() {
    println!("{}", item);
}
```

```go
// Go: for だけ（while も loop も for で表現）
for i := 0; i < 10; i++ {  // C風 for
    fmt.Println(i)
}

for condition {              // while 相当
    // ...
}

for {                        // 無限ループ
    // ...
}

for i, v := range slice {   // for-range
    fmt.Println(i, v)
}
```

### イテレータメソッド（関数型スタイル）

```typescript
// TypeScript: メソッドチェーン
const result = numbers
    .filter(n => n > 0)
    .map(n => n * 2)
    .reduce((sum, n) => sum + n, 0);

// vs 命令型
let result = 0;
for (const n of numbers) {
    if (n > 0) {
        result += n * 2;
    }
}
```

```rust
// Rust: イテレータ（ゼロコスト抽象化）
let result: i32 = numbers.iter()
    .filter(|&&n| n > 0)
    .map(|&n| n * 2)
    .sum();
// コンパイル後は手書きのループと同等の性能
```

---

## 3. 早期リターンとガード節

```rust
// ガード節パターン（ネストを減らす）
// ❌ ネストが深い
fn process(input: Option<&str>) -> Result<String, Error> {
    if let Some(s) = input {
        if !s.is_empty() {
            if s.len() < 100 {
                Ok(s.to_uppercase())
            } else {
                Err(Error::TooLong)
            }
        } else {
            Err(Error::Empty)
        }
    } else {
        Err(Error::Missing)
    }
}

// ✅ ガード節で早期リターン
fn process(input: Option<&str>) -> Result<String, Error> {
    let s = input.ok_or(Error::Missing)?;
    if s.is_empty() { return Err(Error::Empty); }
    if s.len() >= 100 { return Err(Error::TooLong); }
    Ok(s.to_uppercase())
}
```

---

## まとめ

| 構文 | 文 vs 式 | 特徴 |
|------|---------|------|
| if (Python, JS) | 文 | 古典的、三項演算子は式 |
| if (Rust, Kotlin) | 式 | 値を返せる |
| switch (JS, C) | 文 | fall-through に注意 |
| match (Rust) | 式 | 網羅性チェック、安全 |
| for-in | - | イテレータベース（現代の主流） |
| .filter().map() | 式 | 関数型スタイル |

---

## 次に読むべきガイド
→ [[01-pattern-matching.md]] — パターンマッチ

---

## 参考文献
1. Scott, M. "Programming Language Pragmatics." 4th Ed, Ch.6, Morgan Kaufmann, 2015.
