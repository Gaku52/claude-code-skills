# 組み込み/WASM — no_std、wasm-bindgen

> Rust の no_std 環境と WebAssembly ターゲットを通じて、リソース制約環境での開発手法を習得する

## この章で学ぶこと

1. **no_std プログラミング** — 標準ライブラリなしの Rust、alloc クレート、組み込みターゲット
2. **WebAssembly (WASM)** — wasm-bindgen による JS 連携、wasm-pack ワークフロー
3. **WASI** — サーバーサイド WASM とサンドボックス実行

---

## 1. no_std と std の関係

```
┌─────────────── Rust ライブラリ階層 ────────────────┐
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  std (標準ライブラリ)                         │   │
│  │  - ファイル I/O, ネットワーク, スレッド        │   │
│  │  - OS 依存機能                                │   │
│  │                                               │   │
│  │  ┌─────────────────────────────────────────┐ │   │
│  │  │  alloc (ヒープ割当)                      │ │   │
│  │  │  - Box, Vec, String, Arc, Rc            │ │   │
│  │  │  - ヒープアロケータが必要                 │ │   │
│  │  │                                         │ │   │
│  │  │  ┌─────────────────────────────────┐   │ │   │
│  │  │  │  core (コアライブラリ)            │   │ │   │
│  │  │  │  - Option, Result, Iterator     │   │ │   │
│  │  │  │  - 数値型, スライス, 参照        │   │ │   │
│  │  │  │  - OS依存なし、割当なし          │   │ │   │
│  │  │  └─────────────────────────────────┘   │ │   │
│  │  └─────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  #![no_std]       → core のみ使用可能              │
│  #![no_std] + alloc → core + alloc 使用可能        │
│  (デフォルト)      → core + alloc + std             │
└─────────────────────────────────────────────────────┘
```

---

## 2. no_std プログラミング

### コード例1: no_std ライブラリ

```rust
// src/lib.rs
#![no_std]

// core からのインポート (std と同じ API が多い)
use core::fmt;

/// no_std 対応のリングバッファ
pub struct RingBuffer<const N: usize> {
    data: [u8; N],
    head: usize,
    tail: usize,
    len: usize,
}

impl<const N: usize> RingBuffer<N> {
    pub const fn new() -> Self {
        RingBuffer {
            data: [0; N],
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, byte: u8) -> Result<(), u8> {
        if self.len == N {
            return Err(byte); // バッファ満杯
        }
        self.data[self.tail] = byte;
        self.tail = (self.tail + 1) % N;
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<u8> {
        if self.len == 0 {
            return None;
        }
        let byte = self.data[self.head];
        self.head = (self.head + 1) % N;
        self.len -= 1;
        Some(byte)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<const N: usize> fmt::Debug for RingBuffer<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer<{}>(len={})", N, self.len)
    }
}
```

### コード例2: ベアメタル組み込み (ARM Cortex-M)

```rust
// src/main.rs
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _; // パニック時に halt

// グローバルアロケータ (alloc を使う場合)
// use embedded_alloc::Heap;
// #[global_allocator]
// static HEAP: Heap = Heap::empty();

#[entry]
fn main() -> ! {
    // ペリフェラルの取得
    let peripherals = stm32f4xx_hal::pac::Peripherals::take().unwrap();
    let gpioa = peripherals.GPIOA.split();

    // LED ピン設定 (PA5)
    let mut led = gpioa.pa5.into_push_pull_output();

    loop {
        led.set_high();
        cortex_m::asm::delay(8_000_000); // 約1秒 (8MHz)
        led.set_low();
        cortex_m::asm::delay(8_000_000);
    }
}

// Cargo.toml:
// [dependencies]
// cortex-m = "0.7"
// cortex-m-rt = "0.7"
// panic-halt = "0.2"
// stm32f4xx-hal = { version = "0.21", features = ["stm32f401"] }
//
// .cargo/config.toml:
// [build]
// target = "thumbv7em-none-eabihf"
```

---

## 3. WebAssembly

### WASM コンパイルターゲット

```
┌────────────── WASM ターゲット ──────────────┐
│                                              │
│  wasm32-unknown-unknown                      │
│  └─ ブラウザ / JS ランタイム向け             │
│  └─ wasm-bindgen で JS 連携                  │
│                                              │
│  wasm32-wasip1 (旧 wasm32-wasi)             │
│  └─ WASI 対応ランタイム向け                  │
│  └─ ファイル I/O、ネットワーク等              │
│  └─ wasmtime, wasmer で実行                  │
│                                              │
│  ビルドツール:                                │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  wasm-pack      │  │  trunk           │  │
│  │  npm パッケージ  │  │  Yew/Leptos SPA │  │
│  └─────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────┘
```

### コード例3: wasm-bindgen 基本

```rust
// Cargo.toml:
// [lib]
// crate-type = ["cdylib"]
//
// [dependencies]
// wasm-bindgen = "0.2"
// js-sys = "0.3"
// web-sys = { version = "0.3", features = ["console", "Document", "Element", "HtmlElement"] }

use wasm_bindgen::prelude::*;

/// JS から呼べる関数
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}! (from Rust/WASM)", name)
}

/// JS のコンソールに出力
#[wasm_bindgen]
pub fn log_to_console(message: &str) {
    web_sys::console::log_1(&message.into());
}

/// JS オブジェクトの操作
#[wasm_bindgen]
pub fn fibonacci(n: u32) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let temp = b;
        b = a.wrapping_add(b);
        a = temp;
    }
    a
}

/// 構造体を JS に公開
#[wasm_bindgen]
pub struct ImageProcessor {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[wasm_bindgen]
impl ImageProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        ImageProcessor {
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize], // RGBA
        }
    }

    pub fn pixels_ptr(&self) -> *const u8 {
        self.pixels.as_ptr()
    }

    pub fn pixels_len(&self) -> usize {
        self.pixels.len()
    }

    /// グレースケール変換
    pub fn grayscale(&mut self) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            let gray = (0.299 * chunk[0] as f64
                + 0.587 * chunk[1] as f64
                + 0.114 * chunk[2] as f64) as u8;
            chunk[0] = gray;
            chunk[1] = gray;
            chunk[2] = gray;
            // chunk[3] (alpha) はそのまま
        }
    }
}
```

### コード例4: JS 側の使用

```javascript
// wasm-pack build --target web 後に:
import init, { greet, fibonacci, ImageProcessor } from './pkg/my_wasm.js';

async function main() {
    await init();

    // 基本関数
    console.log(greet("World"));        // "Hello, World! (from Rust/WASM)"
    console.log(fibonacci(50));          // 12586269025n

    // 画像処理
    const processor = new ImageProcessor(800, 600);
    processor.grayscale();

    // Rust のメモリに直接アクセス
    const ptr = processor.pixels_ptr();
    const len = processor.pixels_len();
    const pixels = new Uint8Array(
        wasm.__wbindgen_export_0.buffer,
        ptr,
        len
    );
}

main();
```

### コード例5: WASI — サーバーサイド WASM

```rust
// WASI 対応プログラム (通常の Rust コードとほぼ同じ)
use std::fs;
use std::io::{self, Read};

fn main() -> io::Result<()> {
    // ファイル読み取り (WASI のサンドボックス内)
    let content = fs::read_to_string("/input/data.txt")?;
    println!("ファイル内容: {}", content);

    // 標準入力からの読み取り
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    println!("入力: {}", buffer);

    // 環境変数
    for (key, value) in std::env::vars() {
        println!("{}={}", key, value);
    }

    Ok(())
}

// ビルドと実行:
// $ rustup target add wasm32-wasip1
// $ cargo build --target wasm32-wasip1 --release
// $ wasmtime --dir /input::/path/to/input target/wasm32-wasip1/release/my_app.wasm
```

---

## 4. wasm-pack ワークフロー

```
┌──────────── wasm-pack ビルドフロー ──────────────┐
│                                                   │
│  src/lib.rs                                       │
│    │                                              │
│    ▼                                              │
│  cargo build --target wasm32-unknown-unknown      │
│    │                                              │
│    ▼                                              │
│  wasm-bindgen (JS バインディング生成)              │
│    │                                              │
│    ▼                                              │
│  wasm-opt (オプション: サイズ最適化)               │
│    │                                              │
│    ▼                                              │
│  pkg/                                             │
│  ├── my_wasm_bg.wasm     (WASMバイナリ)          │
│  ├── my_wasm.js          (JSグルーコード)         │
│  ├── my_wasm.d.ts        (TypeScript型定義)      │
│  └── package.json        (npm パッケージ)        │
│                                                   │
│  コマンド:                                        │
│  $ wasm-pack build --target web                   │
│  $ wasm-pack build --target bundler  (webpack用) │
│  $ wasm-pack build --target nodejs   (Node.js用) │
└───────────────────────────────────────────────────┘
```

---

## 5. 比較表

### WASM ターゲット比較

| 項目 | wasm32-unknown-unknown | wasm32-wasip1 |
|---|---|---|
| 実行環境 | ブラウザ / JS ランタイム | wasmtime / wasmer |
| ファイル I/O | 不可 (web-sys 経由) | 可能 (サンドボックス内) |
| ネットワーク | fetch API 経由 | WASI Socket (実験的) |
| std サポート | 限定的 | ほぼ完全 |
| JS 連携 | wasm-bindgen | 不要 |
| ユースケース | フロントエンド高速化 | サーバーサイド、プラグイン |
| バイナリサイズ | 小 (数十KB〜) | 中 (数百KB〜) |

### 組み込みフレームワーク比較

| フレームワーク | 対象 | 特徴 | HAL |
|---|---|---|---|
| embassy | ARM Cortex-M | async/await ベース | embassy-stm32 等 |
| RTIC | ARM Cortex-M | 割り込み駆動、静的解析 | 独自 |
| esp-hal | ESP32 | ESP32 シリーズ全対応 | 独自 |
| Arduino (avr-hal) | AVR | Arduino UNO 等 | avr-device |

---

## 6. アンチパターン

### アンチパターン1: WASM バイナリの肥大化

```rust
// NG: 不要な機能を全て含める
// Cargo.toml:
// [dependencies]
// serde = { version = "1", features = ["derive"] }
// chrono = "0.4"        ← タイムゾーンDB でサイズ増大
// regex = "1"           ← コンパイル済み正規表現で増大

// OK: WASM 向けに軽量な代替を選択
// Cargo.toml:
// [dependencies]
// serde = { version = "1", features = ["derive"] }
// serde-wasm-bindgen = "0.6"
// time = "0.3"          ← chrono より軽量
//
// [profile.release]
// opt-level = "s"       ← サイズ最適化
// lto = true            ← リンク時最適化
// codegen-units = 1     ← 最大最適化
// strip = true          ← デバッグ情報除去

// wasm-opt でさらに縮小
// $ wasm-opt -Oz -o output.wasm input.wasm
```

### アンチパターン2: no_std でのパニック未処理

```rust
// NG: パニックハンドラなしで #![no_std] → リンクエラー
#![no_std]
// error: `#[panic_handler]` function required

// OK: パニックハンドラを定義
#![no_std]

// 方法1: panic-halt (無限ループで停止)
use panic_halt as _;

// 方法2: カスタムパニックハンドラ
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    // UART にエラーを出力するなど
    // unsafe { write_uart(format_args!("{}", info)); }
    loop {
        core::hint::spin_loop();
    }
}
```

---

## FAQ

### Q1: WASM のパフォーマンスはネイティブと比べてどう?

**A:** 一般的にネイティブの 60-90% 程度の性能です。数値計算は特に良好で、JIT コンパイルされた JS より 2-10 倍高速な場合もあります。ただし DOM 操作は JS-WASM 境界のオーバーヘッドがあるため、大量の細かいDOM操作には不向きです。

### Q2: no_std と alloc を同時に使うには?

**A:** `#![no_std]` を宣言した上で `extern crate alloc;` を追加し、グローバルアロケータを設定します。

```rust
#![no_std]
extern crate alloc;
use alloc::{vec, vec::Vec, string::String};

// アロケータ設定 (組み込み向け)
use embedded_alloc::LlffHeap as Heap;
#[global_allocator]
static HEAP: Heap = Heap::empty();
```

### Q3: WASM のデバッグ方法は?

**A:** (1) `console_error_panic_hook` でパニック時にブラウザコンソールにスタックトレースを表示 (2) Chrome DevTools のWASMデバッガでソースマップを使う (3) `wasm2wat` でテキスト形式に変換して解析。

```rust
// デバッグ用パニックフック
#[wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| no_std | OS なし環境向け。core のみ使用 |
| alloc | ヒープ割当が使えれば Vec/String を追加可能 |
| #![no_main] | エントリポイントを自分で定義 (ベアメタル) |
| wasm-bindgen | Rust ←→ JS の型安全なブリッジ |
| wasm-pack | WASM + JS + d.ts + package.json を一括生成 |
| WASI | サーバーサイド WASM。ファイル I/O 対応 |
| サイズ最適化 | opt-level="s", lto=true, wasm-opt |
| embassy | 組み込みasync ランタイム。現代的な開発体験 |

## 次に読むべきガイド

- [CLIツール](./04-cli-tools.md) — クロスコンパイルの実践
- [FFI](./02-ffi-interop.md) — WASM 以外の言語間連携
- [メモリレイアウト](./00-memory-layout.md) — no_std 環境でのメモリ管理

## 参考文献

1. **Rust and WebAssembly Book**: https://rustwasm.github.io/docs/book/
2. **The Embedded Rust Book**: https://docs.rust-embedded.org/book/
3. **wasm-bindgen Guide**: https://rustwasm.github.io/docs/wasm-bindgen/
