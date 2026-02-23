# 整数表現と2の補数

> コンピュータが負の数を表現する方法は「2の補数」という天才的な仕組みであり、加算器1つで加算も減算もこなせる。

## この章で学ぶこと

- [ ] 符号なし整数と符号付き整数の違いを説明できる
- [ ] 2の補数の仕組みを手計算で確認できる
- [ ] オーバーフローの原因と対策を説明できる
- [ ] エンディアン（バイトオーダー）の違いを理解する
- [ ] 固定小数点数の仕組みと金融計算での応用を理解する
- [ ] 各言語の整数型の特性と制限を把握する

## 前提知識

- 2進数と16進数 → 参照: [[00-binary-and-number-systems.md]]

---

## 1. 符号なし整数（Unsigned Integer）

### 1.1 基本

```
符号なし整数: 全ビットを値の表現に使用

  Nビットで表現できる範囲: 0 〜 2^N - 1

  8ビット (uint8):   0 〜 255
  16ビット (uint16):  0 〜 65,535
  32ビット (uint32):  0 〜 4,294,967,295 (約43億)
  64ビット (uint64):  0 〜 18,446,744,073,709,551,615 (約1844京)

  例: 8ビットでの表現
  0000 0000 =   0
  0000 0001 =   1
  0111 1111 = 127
  1000 0000 = 128
  1111 1111 = 255
```

### 1.2 符号なし整数の演算

```
符号なし整数の加算（8ビット）:

  基本加算:
    0000 0011 (3)
  + 0000 0101 (5)
  ──────────────
    0000 1000 (8) ✓

  キャリー（繰り上がり）付き加算:
    0110 1100 (108)
  + 0011 0101 (53)
  ──────────────
    1010 0001 (161) ✓

  ラップアラウンド（オーバーフロー）:
    1111 1111 (255)
  + 0000 0001 (1)
  ──────────────
  1 0000 0000 → 8ビットに切り詰め → 0000 0000 (0)
  キャリーフラグ = 1（キャリーアウト）

  符号なし整数の減算:
    実際には「2の補数を加算」として実行される
    5 - 3 → 5 + (-3) → 5 + (256 - 3) → 5 + 253 = 258 → 8ビット: 2

    0000 0101 (5)
  + 1111 1101 (253 = -3の2の補数表現)
  ──────────────
  1 0000 0010 → キャリーを捨てて → 0000 0010 (2) ✓


符号なし整数の乗算:

  8ビット × 8ビット → 最大 255 × 255 = 65,025 → 16ビット必要
  → 乗算結果は元のビット幅の2倍のビット数が必要

  実務的な注意:
  - C言語: unsigned char の乗算は int に昇格してから実行
  - 結果を元の型に格納するとオーバーフローの可能性
  - 中間計算はより広い型で行うのが安全
```

### 1.3 各言語での符号なし整数

```python
# Python: 整数に上限なし（任意精度整数）
x = 2**64  # 18446744073709551616 — 問題なく扱える
x = 2**1000  # これも問題なし

# ただしctypesやstructで固定幅を扱う場合は制限あり
import struct
struct.pack('B', 255)   # uint8: OK
# struct.pack('B', 256)   # struct.error: ubyte format requires 0 <= number <= 255

# struct フォーマット文字
# 'B' = uint8,  'b' = int8
# 'H' = uint16, 'h' = int16
# 'I' = uint32, 'i' = int32
# 'Q' = uint64, 'q' = int64

# ctypes での固定幅整数
import ctypes
val = ctypes.c_uint8(255)
print(val.value)  # 255
val = ctypes.c_uint8(256)
print(val.value)  # 0 (ラップアラウンド)

# numpy での固定幅整数
import numpy as np
a = np.uint8(255)
print(a + np.uint8(1))  # 0 (ラップアラウンド、警告あり)
```

```rust
// Rust: 明示的な型指定が必須
let a: u8 = 255;    // OK
// let b: u8 = 256;    // コンパイルエラー！
let c: u32 = 4_294_967_295;  // OK (アンダースコアで視認性向上)
let d: u64 = 18_446_744_073_709_551_615;  // OK

// 型変換
let small: u8 = 200;
let large: u32 = small as u32;   // 安全な拡張（200のまま）
let back: u8 = large as u8;     // 切り捨て（200が戻る）

// u16 → u8 の切り捨て
let big: u16 = 300;
let truncated: u8 = big as u8;  // 300 - 256 = 44

// usize: プラットフォーム依存のサイズ（32ビットOS=32bit, 64ビットOS=64bit）
let index: usize = 42;  // 配列インデックスに使用
```

```go
// Go: 明確な型システム
var a uint8 = 255
var b uint16 = 65535
var c uint32 = 4294967295
var d uint64 = 18446744073709551615

// uint: プラットフォーム依存（32 or 64ビット）
var e uint = 42

// byte は uint8 のエイリアス
var f byte = 0xFF

// 型変換は明示的
var g uint32 = uint32(a)  // uint8 → uint32
var h uint8 = uint8(b)    // uint16 → uint8（切り捨て）

// オーバーフローチェックはない（ラップアラウンド）
var i uint8 = 255
i++  // i = 0 (ラップアラウンド、エラーなし)

// math パッケージの定数
import "math"
fmt.Println(math.MaxUint8)   // 255
fmt.Println(math.MaxUint16)  // 65535
fmt.Println(math.MaxUint32)  // 4294967295
```

```javascript
// JavaScript: Number型は64ビット浮動小数点
// → 安全に扱える整数の範囲は限定的
Number.MAX_SAFE_INTEGER  // 9007199254740991 (2^53 - 1)
Number.MIN_SAFE_INTEGER  // -9007199254740991

// 安全な範囲を超えると精度が失われる
console.log(9007199254740992 === 9007199254740993);  // true!（区別不可）

// BigInt で任意精度
const big = 18446744073709551615n;  // OK
const sum = big + 1n;  // 18446744073709551616n

// TypedArray で固定幅の符号なし整数
const u8 = new Uint8Array([255]);
const u16 = new Uint16Array([65535]);
const u32 = new Uint32Array([4294967295]);

// DataView でバイナリデータの読み書き
const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
view.setUint32(0, 0xDEADBEEF, true);  // true = リトルエンディアン
console.log(view.getUint8(0).toString(16));  // 'ef'
```

```c
// C言語: 固定幅整数型（stdint.h 推奨）
#include <stdint.h>
#include <limits.h>

uint8_t  a = 255;           // 0 〜 255
uint16_t b = 65535;          // 0 〜 65535
uint32_t c = 4294967295U;   // 0 〜 4,294,967,295
uint64_t d = 18446744073709551615ULL;  // 0 〜 2^64-1

// size_t: メモリサイズ表現用（常に符号なし）
size_t len = sizeof(int);  // 4 or 8

// 伝統的な型（サイズはプラットフォーム依存で非推奨）
unsigned char      uc;    // 少なくとも8ビット
unsigned short     us;    // 少なくとも16ビット
unsigned int       ui;    // 少なくとも16ビット（通常32ビット）
unsigned long      ul;    // 少なくとも32ビット
unsigned long long ull;   // 少なくとも64ビット

// リテラルサフィックス
uint32_t x = 42U;      // unsigned
uint64_t y = 42ULL;    // unsigned long long
```

```java
// Java: 符号なし整数型がない（Java 8以降で部分サポート）

// Java は全ての整数型が符号付き
byte  b = 127;     // -128 〜 127
short s = 32767;   // -32768 〜 32767
int   i = 2147483647;  // -2^31 〜 2^31-1
long  l = 9223372036854775807L;  // -2^63 〜 2^63-1

// Java 8以降: Integer/Long の符号なし演算メソッド
int unsigned = Integer.parseUnsignedInt("4294967295");  // 0xFFFFFFFF
String str = Integer.toUnsignedString(unsigned);  // "4294967295"
int result = Integer.divideUnsigned(unsigned, 2);  // 2147483647
int cmp = Integer.compareUnsigned(-1, 1);  // 正の値 (0xFFFFFFFF > 1)

// byte を符号なしとして扱う
byte byteVal = (byte) 0xFF;  // -1 として格納
int unsignedByte = byteVal & 0xFF;  // 255 として取得
```

---

## 2. 符号付き整数（Signed Integer）— 2の補数

### 2.1 負の数の表現方法の比較

```
負の数を表現する3つの方法（8ビットの場合）:

  方法1: 符号ビット（Sign-Magnitude）
  ─────────────────────────────
    最上位ビット = 符号（0:正, 1:負）
    残り7ビット = 絶対値

    +5 = 0_0000101
    -5 = 1_0000101

    問題点:
    - +0 (0000 0000) と -0 (1000 0000) の2つのゼロが存在
    - 加算に特別な回路が必要
    - 範囲: -127 〜 +127

  方法2: 1の補数（One's Complement）
  ─────────────────────────────
    負の数 = 全ビット反転

    +5 = 0000 0101
    -5 = 1111 1010

    問題点:
    - +0 (0000 0000) と -0 (1111 1111) の2つのゼロ
    - 桁上がりの処理が必要（end-around carry）
    - 範囲: -127 〜 +127

  方法3: 2の補数（Two's Complement）★現代の標準
  ─────────────────────────────
    負の数 = 全ビット反転 + 1

    +5 = 0000 0101
    -5 = 1111 1011  (0000 0101 → 反転 → 1111 1010 → +1 → 1111 1011)

    利点:
    - ゼロが1つだけ（0000 0000）
    - 加算器1つで加算も減算もできる！
    - 範囲: -128 〜 +127（非対称だが合理的）
```

### 2.2 2の補数の数学的理解

```
2の補数の本質:
  -x = 2^N - x  （N = ビット数）

  8ビットの場合: -x = 256 - x

  例: -5 = 256 - 5 = 251 = 1111 1011

  なぜこれで加算が統一できるのか:
  5 + (-5) = 5 + 251 = 256 = 1_0000_0000 (9ビット)
  → 8ビットに収まらない最上位ビット(キャリー)を捨てると 0000 0000 = 0 ✓

  3 + (-5) = 3 + 251 = 254 = 1111 1110
  → 2の補数として解釈すると -2 ✓

  → ハードウェアは符号を意識せず、ただ加算するだけでよい！


数学的な背景（剰余演算/合同算術）:

  2の補数は mod 2^N の世界での演算と等価

  例（mod 256）:
  -5 ≡ 251 (mod 256)   ← 256 - 5 = 251
  -1 ≡ 255 (mod 256)   ← 256 - 1 = 255

  加算: 3 + (-5) ≡ 3 + 251 ≡ 254 ≡ -2 (mod 256) ✓

  → 2の補数は、整数の剰余群 Z/2^N Z そのもの
  → 加算の群演算が自然に成立するため、ハードウェアで効率的


2の補数の重要な性質:

  1. 符号判定: 最上位ビット(MSB)が1なら負
     0xxx xxxx → 正 (0 〜 127)
     1xxx xxxx → 負 (-128 〜 -1)

  2. 符号拡張: ビット幅を広げる際、MSBを複製
     int8 → int16: -5 (1111 1011) → (1111 1111 1111 1011) = -5
     int8 → int16:  5 (0000 0101) → (0000 0000 0000 0101) = 5

  3. 否定: ~x + 1 = -x
     ~0000 0101 = 1111 1010
     1111 1010 + 1 = 1111 1011 = -5

  4. 絶対値: |x| = x が正なら x, 負なら ~x + 1
```

### 2.3 2の補数の全数表（8ビット）

```
8ビット2の補数の全数表（主要な値）:

  2進数        10進数   16進数   説明
  ──────────────────────────────────────
  0111 1111    +127     0x7F    INT8_MAX
  0111 1110    +126     0x7E
  ...
  0000 0010      +2     0x02
  0000 0001      +1     0x01
  0000 0000       0     0x00    ゼロ
  1111 1111      -1     0xFF
  1111 1110      -2     0xFE
  1111 1101      -3     0xFD
  ...
  1000 0010    -126     0x82
  1000 0001    -127     0x81
  1000 0000    -128     0x80    INT8_MIN

  パターンの観察:
  - 正の数: 0x00-0x7F (0-127)
  - 負の数: 0x80-0xFF (-128 〜 -1)
  - -1 = 全ビット1 (0xFF)
  - INT8_MIN = MSBのみ1 (0x80)
  - 0を挟んで: ... -3, -2, -1, 0, +1, +2, +3 ...
  - ビットパターンとしては連続: ... FD, FE, FF, 00, 01, 02, 03 ...
```

### 2.4 2の補数の変換手順

```
正 → 負 の変換:

  方法1: 全ビット反転 + 1
    +42 = 0010 1010
    反転 → 1101 0101
    +1  → 1101 0110 = -42

  方法2: 2^N - x
    -42 = 256 - 42 = 214 = 1101 0110 ✓

  方法3: 右端の1を見つけ、それより左のビットを全て反転
    +42 = 0010 1010
              ↑ 右端の1
    反転 → 1101 0110 = -42

  逆変換（負 → 正）: 同じ操作をもう一度行う
    -42 = 1101 0110
    反転 → 0010 1001
    +1  → 0010 1010 = +42 ✓


具体例をいくつか:

  +1 → -1:
    0000 0001 → 反転 → 1111 1110 → +1 → 1111 1111 = 0xFF = -1

  +100 → -100:
    0110 0100 → 反転 → 1001 1011 → +1 → 1001 1100 = 0x9C = -100

  +127 → -127:
    0111 1111 → 反転 → 1000 0000 → +1 → 1000 0001 = 0x81 = -127

  -128 → ???:
    1000 0000 → 反転 → 0111 1111 → +1 → 1000 0000 = -128（自分自身！）
    → -128は8ビットで対称な正の値を持たない特殊な値
```

### 2.5 2の補数の加減算

```
2の補数での加減算の実例:

  例1: 50 + 30 = 80
    0011 0010 (50)
  + 0001 1110 (30)
  ──────────────
    0101 0000 (80) ✓  符号ビット=0, オーバーフローなし


  例2: 50 + (-30) = 20
    0011 0010 (50)
  + 1110 0010 (-30)
  ──────────────
  1 0001 0100 → キャリーを捨てて → 0001 0100 (20) ✓


  例3: -50 + (-30) = -80
    1100 1110 (-50)
  + 1110 0010 (-30)
  ──────────────
  1 1011 0000 → キャリーを捨てて → 1011 0000
  1011 0000 = -(~1011 0000 + 1) = -(0100 1111 + 1) = -(0101 0000) = -80 ✓


  例4: 100 + 50 = 150 → オーバーフロー！
    0110 0100 (100)
  + 0011 0010 (50)
  ──────────────
    1001 0110 → 2の補数として: -106（正しくない！）
  正+正=負 → オーバーフロー発生！（8ビット符号付きの最大値は127）


  例5: -100 + (-50) = -150 → オーバーフロー！
    1001 1100 (-100)
  + 1100 1110 (-50)
  ──────────────
  1 0110 1010 → キャリーを捨てて → 0110 1010 = 106（正しくない！）
  負+負=正 → オーバーフロー発生！（8ビット符号付きの最小値は-128）


  減算は「2の補数を加算」に変換:
  A - B = A + (-B) = A + (~B + 1)

  例: 30 - 50 = -20
    0001 1110 (30)
  + 1100 1110 (-50)  ← 50の2の補数
  ──────────────
    1110 1100 → 2の補数として: -20 ✓
```

### 2.6 符号付き整数の範囲

```
Nビット2の補数の範囲: -2^(N-1) 〜 2^(N-1) - 1

  型       ビット数  最小値                     最大値
  ──────────────────────────────────────────────────────
  int8     8        -128                       127
  int16    16       -32,768                    32,767
  int32    32       -2,147,483,648             2,147,483,647 (約±21億)
  int64    64       -9,223,372,036,854,775,808  9,223,372,036,854,775,807

  なぜ負の方が1つ多い？
  ────────────────────
  8ビットの場合:
  正の最大: 0111 1111 = +127
  負の最小: 1000 0000 = -128

  -128を反転+1すると:
  1000 0000 → 0111 1111 → 1000 0000 = -128 (自分自身に戻る！)
  → -128は反転操作で対になる正の数が存在しない特殊な値

  対称性の問題:
  - abs(INT_MIN) はオーバーフローする！
  - abs(-128) は128だが、int8で128は表現不可
  - C言語: abs(INT_MIN) は未定義動作
  - Java: Math.abs(Integer.MIN_VALUE) は Integer.MIN_VALUE を返す

  安全な絶対値計算:
  long safe_abs(int x) {
      return (long)x < 0 ? -(long)x : (long)x;  // より広い型に変換
  }
```

### 2.7 符号拡張とゼロ拡張

```
符号拡張（Sign Extension）: 符号付き整数のビット幅を広げる

  int8 → int16:
  +5:  0000 0101 → 0000 0000 0000 0101  (MSBの0を左に拡張)
  -5:  1111 1011 → 1111 1111 1111 1011  (MSBの1を左に拡張)

  int16 → int32:
  -100: 1111 1111 1001 1100
      → 1111 1111 1111 1111 1111 1111 1001 1100

  規則: MSB（符号ビット）を新しいビットにコピーする
  → 値は変わらない


ゼロ拡張（Zero Extension）: 符号なし整数のビット幅を広げる

  uint8 → uint16:
  200: 1100 1000 → 0000 0000 1100 1000  (左に0を詰める)
  255: 1111 1111 → 0000 0000 1111 1111

  規則: 常に0を左に詰める
  → 値は変わらない


C言語での注意:
  int8_t x = -5;     // 1111 1011
  uint16_t y = x;    // 符号拡張 → 1111 1111 1111 1011 → uint16として 65531!
  // 意図: -5 → 65531 になってしまう
  // 正しくは: int16_t y = x; で符号拡張

  uint8_t a = 200;   // 1100 1000
  int16_t b = a;     // ゼロ拡張 → 0000 0000 1100 1000 → 200
  // OK: 符号なし→符号付きへの変換で値が保持される（範囲内なら）
```

```python
# Pythonでの符号拡張シミュレーション

def sign_extend(value, from_bits, to_bits):
    """符号拡張: from_bits幅の符号付き整数をto_bits幅に拡張"""
    # 符号ビットを確認
    if value & (1 << (from_bits - 1)):
        # 負の数: 上位ビットを1で埋める
        mask = ((1 << to_bits) - 1) ^ ((1 << from_bits) - 1)
        return value | mask
    return value

# int8 → int32
print(sign_extend(0xFB, 8, 32))   # 0xFFFFFFFB = -5（32ビット）
print(sign_extend(0x05, 8, 32))   # 0x00000005 = +5（32ビット）

# 8ビット符号付き → Python整数
def int8_to_python(byte_val):
    """uint8値を符号付きint8として解釈"""
    if byte_val & 0x80:
        return byte_val - 256
    return byte_val

print(int8_to_python(0xFB))  # -5
print(int8_to_python(0x05))  # 5
print(int8_to_python(0x80))  # -128
```

---

## 3. オーバーフロー

### 3.1 オーバーフローとは

```
オーバーフロー: 演算結果が表現可能な範囲を超えること

  符号なし8ビット:
  255 + 1 = 256 → 0 (ラップアラウンド)

  1111 1111
+ 0000 0001
──────────
1 0000 0000 → 8ビットに切り詰め → 0000 0000 = 0

  符号付き8ビット (2の補数):
  127 + 1 = 128? → -128 (オーバーフロー!)

  0111 1111  (+127)
+ 0000 0001  (+1)
──────────
  1000 0000  (-128)  ← 正+正=負 は明らかにおかしい

  オーバーフロー検出:
  - 正 + 正 = 負 → オーバーフロー
  - 負 + 負 = 正 → オーバーフロー
  - 正 + 負 は絶対にオーバーフローしない
```

### 3.2 符号なし vs 符号付きオーバーフロー

```
符号なしオーバーフロー（ラップアラウンド）:
  C言語では「well-defined behavior」（定義済み動作）
  結果は mod 2^N

  uint8: 255 + 1 = 0
  uint8: 0 - 1 = 255
  uint16: 65535 + 1 = 0
  uint32: 4294967295 + 1 = 0

  用途:
  - ハッシュ計算（ラップアラウンドを利用）
  - カウンタ（一周して0に戻ることを前提）
  - CRC計算


符号付きオーバーフロー:
  C/C++では「undefined behavior」（未定義動作）！
  → コンパイラは「起きない」と仮定して最適化する

  例:
  int x = INT_MAX;
  if (x + 1 > x) {  // コンパイラはこの条件を常にtrueと仮定
      // ...          // オーバーフローチェックが削除される可能性
  }

  GCC -O2 での最適化例:
  // 元のコード
  int check_overflow(int x) {
      return x + 1 > x;
  }
  // 最適化後: 常に 1 を返す（オーバーフローは起きないと仮定）

  → -fwrapv オプションで符号付きラップアラウンドを保証可能
  → -ftrapv オプションでオーバーフロー時にトラップ
```

### 3.3 実際のバグ・事故

```python
# 有名なオーバーフロー事故

# 1. Ariane 5 ロケット爆発（1996年）
#    64ビット浮動小数点 → 16ビット符号付き整数への変換
#    水平速度が32,767を超え、オーバーフロー → 制御不能 → 爆発
#    損害: 5億ドル

# 2. パックマン 256面バグ（1980年）
#    面数を8ビット符号なし整数で管理
#    255面クリア → 256面 = 0x100 → 8ビットでは0x00
#    右半分が文字化けした「キルスクリーン」が出現

# 3. Boeing 787 電源喪失（2015年）
#    32ビットカウンタが248日で2^31に到達
#    int32オーバーフロー → 電源制御システムがシャットダウン
#    対策: 248日以内に再起動する暫定措置（！）

# 4. 2038年問題（Y2K38）
#    Unix時間: 1970年1月1日からの秒数（int32）
#    2^31 - 1 = 2,147,483,647秒 = 2038年1月19日 03:14:07 UTC
#    → int64への移行が必要
import time
# 2038年問題のタイムスタンプ
print(2**31 - 1)  # 2147483647
# ほとんどの現代システムは64ビットに移行済み

# 5. 文明シリーズの核ガンジー（都市伝説的だが有名）
#    ガンジーの攻撃性パラメータ(uint8)が1で、民主主義で-2されると
#    1 - 2 = -1 → uint8で255（最大値）になり超攻撃的に
#    ※ 実際にはバグではなく仕様だった可能性も指摘されている

# 6. ヒープバッファオーバーフロー（セキュリティ）
#    整数オーバーフローはバッファオーバーフロー攻撃の入り口になる
#    例: size_t size = user_input_width * user_input_height * 4;
#    巨大な width と height で乗算がオーバーフロー
#    → 小さなバッファが確保される
#    → 書き込み時にヒープ破壊 → 任意コード実行
```

### 3.4 バイナリサーチのオーバーフローバグ

```python
# 有名なバイナリサーチのバグ（JDK 6で発見）

# ❌ 古典的な中間点計算（オーバーフローの可能性）
def binary_search_buggy(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2  # ← low + high がオーバーフローする可能性！
        # 例: low=2^30, high=2^30+100 → low+high > INT_MAX
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# ✅ 安全な中間点計算
def binary_search_safe(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2  # オーバーフロー安全
        # または: mid = (low + high) >>> 1  (Java/C#の符号なし右シフト)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# ビット演算での安全な平均値計算
def safe_average(a, b):
    """オーバーフローなしの平均値"""
    return (a & b) + ((a ^ b) >> 1)
    # 共通ビット + 異なるビットの半分
```

### 3.5 各言語のオーバーフロー対策

```python
# Python: 任意精度整数 → オーバーフローなし！
x = 2**100 + 1  # 問題なし
# Pythonは唯一、整数オーバーフローを心配しなくてよい言語

# ただし、ctypes/struct/numpy では固定幅なのでオーバーフローあり
import numpy as np
a = np.int8(127)
print(a + np.int8(1))  # -128 (ラップアラウンド)

# 安全な固定幅演算
def safe_add_int32(a, b):
    """Pythonでint32のオーバーフローをシミュレート"""
    result = a + b
    if result > 2**31 - 1 or result < -(2**31):
        raise OverflowError(f"int32 overflow: {a} + {b} = {result}")
    return result
```

```rust
// Rust: コンパイル時・実行時の検出
let x: u8 = 255;

// デバッグビルド: パニック（プログラム停止）
// let y = x + 1;  // thread 'main' panicked at 'attempt to add with overflow'

// リリースビルド: ラップアラウンド（デフォルト）
// 明示的なオーバーフロー制御メソッド:
let a = x.checked_add(1);    // Option<u8> → None
let b = x.saturating_add(1); // 255 (上限で飽和)
let c = x.wrapping_add(1);   // 0 (明示的ラップ)
let d = x.overflowing_add(1); // (0, true) — 値とオーバーフローフラグ

// 実務での使い分け:
// checked_add: 正確性が最重要（金融、科学計算）
// saturating_add: 上限/下限で止まるのが望ましい（音量、明度）
// wrapping_add: ラップアラウンドが仕様（ハッシュ、カウンタ）
// overflowing_add: オーバーフローの発生を知りたい（低レベルCPU模倣）

// saturating の実用例
fn adjust_volume(current: u8, delta: i8) -> u8 {
    if delta >= 0 {
        current.saturating_add(delta as u8)
    } else {
        current.saturating_sub((-delta) as u8)
    }
}
// adjust_volume(250, 10) → 255（255で飽和）
// adjust_volume(5, -10) → 0（0で飽和）
```

```java
// Java: サイレントラップアラウンド（危険！）
int x = Integer.MAX_VALUE;  // 2147483647
int y = x + 1;              // -2147483648 (警告なし！)

// Java 8以降: Math.addExact()
try {
    int z = Math.addExact(x, 1);  // ArithmeticException
} catch (ArithmeticException e) {
    System.out.println("Overflow detected!");
}

// Math クラスの安全な演算メソッド
Math.addExact(a, b);       // 加算（オーバーフロー時例外）
Math.subtractExact(a, b);  // 減算
Math.multiplyExact(a, b);  // 乗算
Math.negateExact(a);       // 否定
Math.incrementExact(a);    // +1
Math.decrementExact(a);    // -1
Math.toIntExact(longVal);  // long → int（範囲外で例外）
```

```c
// C: 符号付きオーバーフローは未定義動作（最も危険）
int x = INT_MAX;
int y = x + 1;  // 未定義動作！コンパイラが何をしても「正しい」
// GCCの最適化により、オーバーフローチェックが削除される場合もある

// 安全な加算チェック（符号付き）:
#include <limits.h>
#include <stdbool.h>

bool safe_add_int(int a, int b, int *result) {
    if (b > 0 && a > INT_MAX - b) return false;  // オーバーフロー
    if (b < 0 && a < INT_MIN - b) return false;  // アンダーフロー
    *result = a + b;
    return true;
}

// 安全な乗算チェック（符号付き）:
bool safe_mul_int(int a, int b, int *result) {
    if (a == 0 || b == 0) {
        *result = 0;
        return true;
    }
    if (a > 0 && b > 0 && a > INT_MAX / b) return false;
    if (a > 0 && b < 0 && b < INT_MIN / a) return false;
    if (a < 0 && b > 0 && a < INT_MIN / b) return false;
    if (a < 0 && b < 0 && a < INT_MAX / b) return false;
    *result = a * b;
    return true;
}

// GCC/Clang ビルトイン:
int result;
if (__builtin_add_overflow(a, b, &result)) {
    // オーバーフロー発生
}
if (__builtin_mul_overflow(a, b, &result)) {
    // オーバーフロー発生
}
```

```go
// Go: サイレントラップアラウンド（Javaと同様）
package main

import (
    "fmt"
    "math"
)

func main() {
    var x int32 = math.MaxInt32  // 2147483647
    x++  // -2147483648 (ラップアラウンド、エラーなし)
    fmt.Println(x)

    // 安全な加算
    a, b := int32(2000000000), int32(1000000000)
    if safeAddInt32(a, b) {
        fmt.Println("OK:", a+b)
    } else {
        fmt.Println("Overflow!")
    }
}

func safeAddInt32(a, b int32) bool {
    if b > 0 && a > math.MaxInt32-b {
        return false
    }
    if b < 0 && a < math.MinInt32-b {
        return false
    }
    return true
}
```

---

## 4. エンディアン（Byte Order）

### 4.1 ビッグエンディアンとリトルエンディアン

```
エンディアン: マルチバイト値をメモリに格納する際のバイト順序

  値: 0x12345678（32ビット整数）

  ビッグエンディアン（Big-Endian）:
  アドレス:  0x00  0x01  0x02  0x03
  値:        0x12  0x34  0x56  0x78
  → 最上位バイト(MSB)が最小アドレスに格納
  → 人間の読み方と同じ順序
  → ネットワーク通信の標準（ネットワークバイトオーダー）

  リトルエンディアン（Little-Endian）:
  アドレス:  0x00  0x01  0x02  0x03
  値:        0x78  0x56  0x34  0x12
  → 最下位バイト(LSB)が最小アドレスに格納
  → Intel/AMD x86/x64, ARM（デフォルト）
  → 加算時に下位バイトから処理でき、回路が単純

  バイエンディアン（Bi-Endian）:
  → 切り替え可能。ARM, MIPS, PowerPC
  → ARM は実質リトルエンディアンで使用されることが多い
```

### 4.2 エンディアンの実務的影響

```python
import struct

value = 0x12345678

# ビッグエンディアンでパック
big = struct.pack('>I', value)
print(big.hex())  # '12345678'

# リトルエンディアンでパック
little = struct.pack('<I', value)
print(little.hex())  # '78563412'

# ネットワーク通信での注意:
# ネットワーク = ビッグエンディアン
# x86 PC = リトルエンディアン
# → 送受信時にバイトオーダー変換が必要

import socket
# ホストバイトオーダー → ネットワークバイトオーダー
port = 8080
network_port = socket.htons(port)  # host to network short

ip = 0xC0A80001  # 192.168.0.1
network_ip = socket.htonl(ip)  # host to network long

# バイトスワップのビット演算実装
def bswap32(n):
    return ((n & 0xFF000000) >> 24) | \
           ((n & 0x00FF0000) >> 8) | \
           ((n & 0x0000FF00) << 8) | \
           ((n & 0x000000FF) << 24)

def bswap16(n):
    return ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8)

print(hex(bswap32(0x12345678)))  # 0x78563412
print(hex(bswap16(0x1234)))      # 0x3412
```

### 4.3 エンディアンの確認方法

```python
import sys
print(sys.byteorder)  # 'little' (x86/ARM) or 'big'

# バイナリファイルの先頭を見て判断する例:
# BMP画像: 先頭が 'BM' (0x42 0x4D) → リトルエンディアン
# JPEG: 先頭が 0xFF 0xD8 → エンディアン非依存
# ELF: offset 5 に 1(LE) or 2(BE) が格納
# UTF-16 BOM: 0xFE 0xFF(BE) or 0xFF 0xFE(LE)

# エンディアン判定の実用コード
def detect_endianness():
    """実行環境のエンディアンを判定"""
    import struct
    if struct.pack('@I', 1) == struct.pack('<I', 1):
        return 'little'
    else:
        return 'big'
```

### 4.4 バイナリプロトコルとエンディアン

```python
# 実務: バイナリプロトコルの設計と実装

import struct

# パケットヘッダの例（ネットワークバイトオーダー = ビッグエンディアン）
class PacketHeader:
    FORMAT = '>HHI'  # ビッグエンディアン: uint16 type, uint16 length, uint32 sequence
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, msg_type, length, sequence):
        self.msg_type = msg_type
        self.length = length
        self.sequence = sequence

    def pack(self):
        return struct.pack(self.FORMAT, self.msg_type, self.length, self.sequence)

    @classmethod
    def unpack(cls, data):
        msg_type, length, sequence = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        return cls(msg_type, length, sequence)

# 使用例
header = PacketHeader(msg_type=1, length=100, sequence=42)
packed = header.pack()
print(packed.hex())  # '0001006400000002a'

# 受信側
received = PacketHeader.unpack(packed)
print(f"Type: {received.msg_type}, Len: {received.length}, Seq: {received.sequence}")


# バイナリファイルフォーマットの例
class BMPHeader:
    """BMP画像ヘッダ（リトルエンディアン）"""
    FORMAT = '<2sIHHI'  # リトルエンディアン: signature, filesize, reserved1, reserved2, data_offset

    @classmethod
    def read(cls, filepath):
        with open(filepath, 'rb') as f:
            data = f.read(struct.calcsize(cls.FORMAT))
            sig, size, r1, r2, offset = struct.unpack(cls.FORMAT, data)
            return {
                'signature': sig,  # b'BM'
                'filesize': size,
                'data_offset': offset
            }
```

---

## 5. 固定小数点数

### 5.1 固定小数点の仕組み

```
固定小数点数: 小数点の位置を固定して整数演算で小数を扱う

  Q8.8 形式（16ビット: 整数部8ビット + 小数部8ビット）:

  ビット: IIIIIIII.FFFFFFFF

  例: 3.75 を Q8.8 で表現
  整数部: 3 = 0000 0011
  小数部: 0.75 = 0.5 + 0.25 = 2^(-1) + 2^(-2) = 1100 0000
  結果: 0000 0011.1100 0000 = 0x03C0

  格納値 = 実数値 × 2^小数部ビット数
  3.75 × 256 = 960 = 0x03C0 ✓

  逆変換: 実数値 = 格納値 / 2^小数部ビット数
  960 / 256 = 3.75 ✓


固定小数点の演算:

  加算/減算: そのまま整数加算（小数点位置が同じなら）
    3.75 + 1.25 → 960 + 320 = 1280 → 1280/256 = 5.0 ✓

  乗算: 結果を右シフト（小数部ビット数分）
    3.75 × 2.0 → 960 × 512 = 491520 → 491520 >> 8 = 1920 → 1920/256 = 7.5 ✓

  除算: 被除数を左シフトしてから除算
    3.75 / 2.0 → (960 << 8) / 512 = 245760 / 512 = 480 → 480/256 = 1.875 ✓


よく使われる固定小数点フォーマット:

  Q1.15 (16ビット): 信号処理（-1.0 〜 +0.999969）
  Q8.8  (16ビット): 汎用（-128.0 〜 +127.996）
  Q16.16 (32ビット): ゲーム/グラフィックス
  Q1.31 (32ビット): 高精度信号処理
  Q32.32 (64ビット): 高精度計算


用途:
  - 金融計算（通貨は小数2桁固定）
  - 組み込みシステム（FPU非搭載のマイコン）
  - ゲーム（DSP時代の3Dグラフィックス）
  - 音声処理（DSP）
  - GPS座標（マイクロ度単位の整数）
```

### 5.2 固定小数点の実装

```python
# 固定小数点数ライブラリの実装

class FixedPoint:
    """Q16.16 固定小数点数"""
    FRAC_BITS = 16
    SCALE = 1 << FRAC_BITS  # 65536
    MASK = (1 << 32) - 1     # 32ビットマスク

    def __init__(self, value=0):
        if isinstance(value, float):
            self._raw = int(value * self.SCALE)
        elif isinstance(value, int):
            self._raw = value * self.SCALE
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    @classmethod
    def from_raw(cls, raw):
        """内部値から直接生成"""
        obj = cls.__new__(cls)
        obj._raw = raw
        return obj

    def to_float(self):
        return self._raw / self.SCALE

    def __add__(self, other):
        return FixedPoint.from_raw(self._raw + other._raw)

    def __sub__(self, other):
        return FixedPoint.from_raw(self._raw - other._raw)

    def __mul__(self, other):
        # 乗算後に小数部ビット数分右シフト
        return FixedPoint.from_raw((self._raw * other._raw) >> self.FRAC_BITS)

    def __truediv__(self, other):
        # 被除数を左シフトしてから除算
        return FixedPoint.from_raw((self._raw << self.FRAC_BITS) // other._raw)

    def __repr__(self):
        return f"FixedPoint({self.to_float():.6f})"

    def __eq__(self, other):
        return self._raw == other._raw

# 使用例
a = FixedPoint(3.75)
b = FixedPoint(2.0)
print(a + b)      # FixedPoint(5.750000)
print(a - b)      # FixedPoint(1.750000)
print(a * b)      # FixedPoint(7.500000)
print(a / b)      # FixedPoint(1.875000)
```

### 5.3 金融計算での整数活用

```python
# ❌ 浮動小数点で金額計算（危険！）
price = 0.1 + 0.2
print(price)  # 0.30000000000000004
print(price == 0.3)  # False!

# ✅ 整数（セント単位）で金額計算
price_cents = 10 + 20  # 30セント
print(price_cents / 100)  # 0.3

# ✅ Decimal型を使用
from decimal import Decimal, ROUND_HALF_UP
price = Decimal('0.1') + Decimal('0.2')
print(price)  # 0.3
print(price == Decimal('0.3'))  # True

# 通貨計算のベストプラクティス
class Money:
    """整数ベースの金額表現"""

    def __init__(self, amount_cents):
        self._cents = int(amount_cents)

    @classmethod
    def from_string(cls, s):
        """'1234.56' → Money(123456)"""
        d = Decimal(s) * 100
        return cls(int(d))

    @classmethod
    def from_float(cls, f):
        """浮動小数点から（非推奨だが必要な場合）"""
        return cls(round(f * 100))

    def __add__(self, other):
        return Money(self._cents + other._cents)

    def __sub__(self, other):
        return Money(self._cents - other._cents)

    def __mul__(self, factor):
        """金額 × 数量"""
        result = Decimal(self._cents) * Decimal(str(factor))
        return Money(int(result.quantize(Decimal('1'), rounding=ROUND_HALF_UP)))

    def __repr__(self):
        sign = '-' if self._cents < 0 else ''
        abs_cents = abs(self._cents)
        return f"¥{sign}{abs_cents // 100}.{abs_cents % 100:02d}"

# 使用例
item = Money.from_string('1980')     # ¥1980.00
tax = item * Decimal('0.1')          # ¥198.00
total = item + tax                    # ¥2178.00
print(total)                          # ¥2178.00


# ✅ 実務でのベストプラクティス
# データベース: DECIMAL(10, 2) — 整数部10桁、小数部2桁
# JavaScript: 金額は全てセント(整数)で扱い、表示時のみ変換
# Java: BigDecimal を使用
# Python: decimal.Decimal を使用
```

---

## 6. 整数型の選択ガイド

### 6.1 型選択の指針

```
整数型選択のフローチャート:

  1. 負の数が必要？
     YES → 符号付き整数
     NO  → 符号なし整数

  2. 必要な範囲は？
     ┌─────────────────────────────────────┐
     │ 範囲                    推奨型       │
     ├─────────────────────────────────────┤
     │ 0-255                   uint8/byte  │
     │ 0-65535                 uint16      │
     │ 0-約43億                uint32      │
     │ それ以上                 uint64      │
     │ -128〜127               int8        │
     │ -32768〜32767           int16       │
     │ -21億〜21億              int32       │
     │ それ以上                 int64       │
     │ 任意の大きさ            BigInteger  │
     └─────────────────────────────────────┘

  3. 特殊な用途:
     - 配列インデックス: size_t / usize (C/Rust)
     - タイムスタンプ: int64 (2038年問題回避)
     - ID/ハッシュ: uint64
     - 金額: Decimal / BigDecimal
     - フラグ: uint8 / uint16 / uint32
     - ループカウンタ: int (言語のデフォルト整数型)

  4. パフォーマンス考慮:
     - CPUのネイティブ幅（32/64ビット）が最速
     - uint8/uint16 は拡張/切り詰めのコストがかかる場合あり
     - ただしメモリ帯域がボトルネックなら小さい型が有利
     - SIMD: 小さい型 → 同時処理数が増加
```

### 6.2 各言語のデフォルト整数型

```
各言語のデフォルト整数型と推奨:

  Python:  int（任意精度、オーバーフローなし）
           → 型選択を気にする必要なし

  Go:      int（プラットフォーム依存: 32 or 64ビット）
           → 明確なサイズが必要なら int32, int64 を使用
           → 配列インデックスは int を使用

  Rust:    i32（デフォルト推論型）
           → 必ず明示的に型を指定すべき
           → 配列インデックスは usize

  Java:    int（32ビット符号付き）
           → long が必要な場面は多い（タイムスタンプ等）
           → unsigned は Integer.toUnsignedXxx() メソッドで

  C/C++:   int（少なくとも16ビット、通常32ビット）
           → stdint.h の固定幅型を使用すべき

  JavaScript: Number（64ビット浮動小数点）
           → 整数精度は53ビット（MAX_SAFE_INTEGER）
           → BigInt で任意精度

  Swift:   Int（プラットフォーム依存: 32 or 64ビット）
           → 通常は Int を使用
           → 特殊な場面で Int8, UInt32 等

  C#:      int（32ビット符号付き）
           → long, uint, ulong も利用可能
           → BigInteger（System.Numerics）で任意精度
```

### 6.3 暗黙の型変換の罠

```c
// C言語の暗黙の型変換（整数昇格）

// 1. 整数昇格: int より小さい型は int に変換される
uint8_t a = 200;
uint8_t b = 100;
uint8_t c = a + b;  // 200 + 100 = 300 → int(300) → uint8(44)
// a, b は int に昇格されて加算、結果が uint8 に切り捨て

// 2. 符号付きと符号なしの混合演算
int x = -1;
unsigned int y = 1;
if (x < y) {
    printf("x < y\n");  // 期待する出力
} else {
    printf("x >= y\n");  // 実際にはこちら！
}
// -1 は unsigned int に変換 → 0xFFFFFFFF = 4294967295 > 1

// 3. 比較時の暗黙変換
int len = -1;
if (len < sizeof(int)) {
    // sizeof は size_t（符号なし）を返す
    // len(-1) が size_t に変換 → 巨大な正の値
    // → この条件は false になる！
}

// 安全なパターン
if (len >= 0 && (size_t)len < sizeof(int)) {
    // 先に負の値チェック
}
```

```python
# Pythonでの型変換の注意点

# Python 3 の // 演算子（切り捨て除算）
print(7 // 2)     # 3 (正の数は切り捨て)
print(-7 // 2)    # -4 (負の無限大方向への切り捨て)
# C言語の -7 / 2 = -3 (ゼロ方向への切り捨て) とは異なる!

# Python 3 の % 演算子（剰余）
print(7 % 2)      # 1
print(-7 % 2)     # 1 (Pythonの剰余は常に除数と同符号)
# C言語の -7 % 2 = -1 とは異なる!

# int から bool
bool(0)    # False
bool(1)    # True
bool(-1)   # True (0以外は全てTrue)

# bool から int
int(True)  # 1
int(False) # 0
True + True  # 2 (boolはintのサブクラス)
```

---

## 7. 実践演習

### 演習1: 2の補数（基礎）
以下の計算を8ビット2の補数で手計算せよ:
1. -42 のビット表現
2. 50 + (-30) の加算
3. -100 + (-50) の加算（オーバーフローするか？）

### 演習2: オーバーフロー検出（応用）
好きな言語で、2つの32ビット符号付き整数の加算がオーバーフローするかどうかを判定する関数を実装せよ。ただし、64ビット整数への拡張を使わずに判定すること。

### 演習3: エンディアン変換（発展）
バイナリファイルから4バイトの整数を読み取り、リトルエンディアン/ビッグエンディアン両方で解釈した値を表示するプログラムを実装せよ。

### 演習4: 固定小数点演算（応用）
Q8.8形式の固定小数点数で以下を計算し、結果をfloatと比較せよ:
1. 3.14 + 2.71
2. 3.14 × 2.71
3. 10.0 / 3.0

### 演習5: 型変換の罠（実務）
以下のC言語コードの出力を予測し、なぜその結果になるか説明せよ:
```c
unsigned int a = 1;
int b = -1;
printf("%d\n", a > b);  // ???
printf("%u\n", b);       // ???
```

### 演習解答例

```python
# 演習1 解答

# 1. -42 のビット表現（8ビット2の補数）
# +42 = 0010 1010
# 反転 = 1101 0101
# +1  = 1101 0110 = 0xD6 = -42
print(f"-42 = {(-42) & 0xFF:08b} = 0x{(-42) & 0xFF:02X}")
# -42 = 11010110 = 0xD6

# 2. 50 + (-30) の加算
#   0011 0010 (50)
# + 1110 0010 (-30)
# = 1 0001 0100 → キャリーを捨てて → 0001 0100 = 20 ✓
print(f"50 + (-30) = {(50 + (-30)) & 0xFF}")
# 50 + (-30) = 20

# 3. -100 + (-50) の加算
# -100 = 1001 1100 (0x9C)
# -50  = 1100 1110 (0xCE)
#   1001 1100
# + 1100 1110
# = 1 0110 1010 → キャリーを捨てて → 0110 1010 = 106
# 負 + 負 = 正 → オーバーフロー！（-150は8ビット符号付きの範囲外）
result = ((-100) & 0xFF) + ((-50) & 0xFF)
print(f"-100 + (-50) = {result & 0xFF} (unsigned), interpreted as {result & 0xFF if result & 0xFF < 128 else (result & 0xFF) - 256}")
# → 106 (正の値) = オーバーフロー（正しい結果は-150）


# 演習2 解答
def will_overflow_int32(a, b):
    """32ビット符号付き整数の加算がオーバーフローするか判定"""
    INT32_MAX = 2**31 - 1   # 2147483647
    INT32_MIN = -(2**31)    # -2147483648

    if b > 0 and a > INT32_MAX - b:
        return True  # 正のオーバーフロー
    if b < 0 and a < INT32_MIN - b:
        return True  # 負のオーバーフロー
    return False

# テスト
print(will_overflow_int32(2**31 - 1, 1))      # True
print(will_overflow_int32(2**31 - 1, 0))      # False
print(will_overflow_int32(-(2**31), -1))       # True
print(will_overflow_int32(100, -50))           # False


# 演習3 解答
import struct

def read_as_both_endian(data):
    """4バイトを両エンディアンで解釈"""
    le_value = struct.unpack('<I', data)[0]
    be_value = struct.unpack('>I', data)[0]
    le_signed = struct.unpack('<i', data)[0]
    be_signed = struct.unpack('>i', data)[0]

    print(f"Bytes: {data.hex()}")
    print(f"Little-Endian unsigned: {le_value} (0x{le_value:08X})")
    print(f"Big-Endian unsigned:    {be_value} (0x{be_value:08X})")
    print(f"Little-Endian signed:   {le_signed}")
    print(f"Big-Endian signed:      {be_signed}")

# テスト
read_as_both_endian(b'\x12\x34\x56\x78')
# Bytes: 12345678
# Little-Endian unsigned: 2018915346 (0x78563412)
# Big-Endian unsigned:    305419896 (0x12345678)
# Little-Endian signed:   2018915346
# Big-Endian signed:      305419896


# 演習4 解答
class Q8_8:
    """Q8.8固定小数点数"""
    FRAC = 8
    SCALE = 256

    def __init__(self, value):
        if isinstance(value, float):
            self._raw = int(value * self.SCALE)
        else:
            self._raw = value * self.SCALE

    @classmethod
    def _from_raw(cls, raw):
        obj = cls.__new__(cls)
        obj._raw = raw
        return obj

    def to_float(self):
        return self._raw / self.SCALE

    def __add__(self, other):
        return Q8_8._from_raw(self._raw + other._raw)

    def __mul__(self, other):
        return Q8_8._from_raw((self._raw * other._raw) >> self.FRAC)

    def __truediv__(self, other):
        return Q8_8._from_raw((self._raw << self.FRAC) // other._raw)

    def __repr__(self):
        return f"Q8.8({self.to_float():.4f})"

a = Q8_8(3.14)
b = Q8_8(2.71)
print(f"3.14 + 2.71 = {(a + b)} (float: {3.14 + 2.71})")
print(f"3.14 * 2.71 = {(a * b)} (float: {3.14 * 2.71})")
c = Q8_8(10.0)
d = Q8_8(3.0)
print(f"10.0 / 3.0 = {(c / d)} (float: {10.0 / 3.0})")
# 固定小数点は精度が限られるため、float との微小な差が生じる


# 演習5 解答
# unsigned int a = 1;
# int b = -1;
# printf("%d\n", a > b);
# → 出力: 0 (false)
# → bがunsigned intに変換され、-1は4294967295になる
# → 1 > 4294967295 は false

# printf("%u\n", b);
# → 出力: 4294967295
# → -1のビットパターン(0xFFFFFFFF)をunsignedとして解釈
```

---

## FAQ

### Q1: なぜ2の補数が採用されたのですか？
**A**: 加算器1つで加減算が統一できるため。ハードウェアコストが劇的に削減される。符号ビット方式では加算と減算に別の回路が必要で、かつ +0/-0 の2つのゼロの処理が複雑。2の補数は数学的にも美しく（mod 2^N の環）、実装も効率的。

### Q2: 2038年問題は本当に起きますか？
**A**: 32ビットのtime_tを使い続けるシステムでは起きうる。ほとんどのデスクトップOS/サーバーは64ビットに移行済み。問題は組み込みシステム（IoTデバイス、産業制御装置）で、ファームウェア更新が困難な機器が多数残存している。

### Q3: Pythonの整数に上限がないのはなぜですか？
**A**: Pythonは内部的に可変長の整数表現を使用（ob_digit配列）。必要に応じてメモリを動的確保するため、メモリが許す限り任意の大きさの整数を扱える。代償として、固定幅整数に比べて演算速度は遅い。

### Q4: 符号付き整数のオーバーフローがC言語で未定義動作なのはなぜですか？
**A**: 2の補数以外の表現（符号ビット方式、1の補数）を使うプラットフォームも想定していたため。また、未定義とすることでコンパイラが「オーバーフローは起きない」と仮定した最適化が可能になる（ループ展開、インダクション変数の最適化等）。C23では2の補数が必須になった。

### Q5: なぜJavaには符号なし整数型がないのですか？
**A**: 設計者のJames Goslingは「符号なし型は混乱の元」と判断した。C言語での符号付き/符号なし混合演算のバグが多発していたためである。Java 8以降、Integerクラスに符号なし演算メソッド（compareUnsigned, divideUnsigned等）が追加された。

### Q6: 整数除算の丸め方向はなぜ言語ごとに異なるのですか？
**A**: 数学的には複数の合理的な定義があるため。C99/Java/Goは「ゼロ方向への切り捨て」（-7/2=-3）、Pythonは「負の無限大方向への切り捨て」（-7//2=-4）。どちらも一長一短で、剰余の符号が変わる。Pythonの方が数学的に一貫しているが、Cの方がハードウェアの除算命令に合致する。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 符号なし整数 | 全ビットを値に使用。0〜2^N-1 |
| 2の補数 | 負の数 = ビット反転+1。加算器で統一処理 |
| 符号拡張 | ビット幅拡大時にMSBを複製。符号なしは0埋め |
| オーバーフロー | 表現範囲を超える演算。言語ごとに挙動が異なる |
| C言語の未定義動作 | 符号付きオーバーフローは未定義。コンパイラ最適化の罠 |
| エンディアン | バイト格納順序。ネットワーク=BE、x86=LE |
| 固定小数点 | 小数点位置固定。金融・組み込みで使用 |
| 型変換の罠 | 符号付き/なし混合、整数昇格に注意 |

---

## 次に読むべきガイド
→ [[03-floating-point.md]] — 浮動小数点数とIEEE 754

---

## 参考文献
1. Bryant, R. E. & O'Hallaron, D. R. "Computer Systems: A Programmer's Perspective." Chapter 2.
2. Warren, H. S. "Hacker's Delight." 2nd Edition, Chapters 2-4.
3. Goldberg, D. "What Every Computer Scientist Should Know About Floating-Point Arithmetic." 1991.
4. IEEE. "IEEE 754-2019 Standard for Floating-Point Arithmetic."
5. ISO/IEC 9899:2024 (C23). "Programming Languages — C."
6. Seacord, R. C. "Secure Coding in C and C++." 2nd Edition, Addison-Wesley, 2013.
7. Bloch, J. "Nearly All Binary Searches and Mergesorts are Broken." Google Research Blog, 2006.
