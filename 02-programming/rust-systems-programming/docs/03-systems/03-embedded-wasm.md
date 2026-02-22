# 組み込み/WASM — no_std、wasm-bindgen

> Rust の no_std 環境と WebAssembly ターゲットを通じて、リソース制約環境での開発手法を習得する

## この章で学ぶこと

1. **no_std プログラミング** — 標準ライブラリなしの Rust、alloc クレート、組み込みターゲット
2. **WebAssembly (WASM)** — wasm-bindgen による JS 連携、wasm-pack ワークフロー
3. **WASI** — サーバーサイド WASM とサンドボックス実行
4. **embassy** — 組み込み向け async/await ランタイム
5. **WASM の最適化** — バイナリサイズ削減、パフォーマンスチューニング

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

### no_std で使える・使えない機能

| 機能 | core | alloc | std |
|---|---|---|---|
| Option, Result | o | o | o |
| Iterator | o | o | o |
| 数値演算 | o | o | o |
| スライス操作 | o | o | o |
| fmt (フォーマット) | o | o | o |
| Vec, String | - | o | o |
| Box, Arc, Rc | - | o | o |
| HashMap | - | - | o |
| ファイル I/O | - | - | o |
| ネットワーク | - | - | o |
| スレッド | - | - | o |
| println! | - | - | o |

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

    pub fn is_full(&self) -> bool {
        self.len == N
    }

    pub fn capacity(&self) -> usize {
        N
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// イテレータを返す
    pub fn iter(&self) -> RingBufferIter<'_, N> {
        RingBufferIter {
            buffer: self,
            index: 0,
        }
    }
}

pub struct RingBufferIter<'a, const N: usize> {
    buffer: &'a RingBuffer<N>,
    index: usize,
}

impl<'a, const N: usize> Iterator for RingBufferIter<'a, N> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len {
            return None;
        }
        let pos = (self.buffer.head + self.index) % N;
        self.index += 1;
        Some(self.buffer.data[pos])
    }
}

impl<const N: usize> fmt::Debug for RingBuffer<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer<{}>(len={})", N, self.len)
    }
}

/// no_std 対応の固定サイズスタック
pub struct FixedStack<T: Copy + Default, const N: usize> {
    data: [T; N],
    top: usize,
}

impl<T: Copy + Default, const N: usize> FixedStack<T, N> {
    pub fn new() -> Self {
        FixedStack {
            data: [T::default(); N],
            top: 0,
        }
    }

    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.top >= N {
            return Err(item);
        }
        self.data[self.top] = item;
        self.top += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.top == 0 {
            return None;
        }
        self.top -= 1;
        Some(self.data[self.top])
    }

    pub fn peek(&self) -> Option<&T> {
        if self.top == 0 {
            None
        } else {
            Some(&self.data[self.top - 1])
        }
    }

    pub fn len(&self) -> usize {
        self.top
    }

    pub fn is_empty(&self) -> bool {
        self.top == 0
    }
}

/// no_std 対応の固定小数点数演算
#[derive(Clone, Copy, Debug)]
pub struct FixedPoint {
    raw: i32, // 16.16 固定小数点
}

impl FixedPoint {
    pub const SCALE: i32 = 65536; // 2^16

    pub const fn from_int(n: i32) -> Self {
        FixedPoint { raw: n << 16 }
    }

    pub const fn from_raw(raw: i32) -> Self {
        FixedPoint { raw }
    }

    pub fn from_f32(f: f32) -> Self {
        FixedPoint {
            raw: (f * Self::SCALE as f32) as i32,
        }
    }

    pub fn to_i32(self) -> i32 {
        self.raw >> 16
    }

    pub fn to_f32(self) -> f32 {
        self.raw as f32 / Self::SCALE as f32
    }

    pub fn add(self, other: Self) -> Self {
        FixedPoint {
            raw: self.raw + other.raw,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        FixedPoint {
            raw: self.raw - other.raw,
        }
    }

    pub fn mul(self, other: Self) -> Self {
        FixedPoint {
            raw: ((self.raw as i64 * other.raw as i64) >> 16) as i32,
        }
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

### コード例: embassy による async 組み込み

```rust
#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_time::{Duration, Timer};
use panic_halt as _;

// embassy: 組み込み向け async/await ランタイム
// スレッドなしで複数タスクを協調的に実行

#[embassy_executor::task]
async fn blink_task(mut led: Output<'static>) {
    loop {
        led.set_high();
        Timer::after(Duration::from_millis(500)).await;
        led.set_low();
        Timer::after(Duration::from_millis(500)).await;
    }
}

#[embassy_executor::task]
async fn sensor_task() {
    loop {
        // センサー読み取り (I2C/SPI)
        // let value = i2c.read_register(0x48, 0x00).await;
        Timer::after(Duration::from_secs(1)).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_stm32::init(Default::default());

    let led = Output::new(p.PA5, Level::Low, Speed::Low);

    // 非同期タスクを生成
    spawner.spawn(blink_task(led)).unwrap();
    spawner.spawn(sensor_task()).unwrap();

    // メインタスクも async で動作
    loop {
        Timer::after(Duration::from_secs(10)).await;
        // ウォッチドッグのリセット等
    }
}

// Cargo.toml:
// [dependencies]
// embassy-executor = { version = "0.6", features = ["arch-cortex-m"] }
// embassy-stm32 = { version = "0.2", features = ["stm32f401ce", "time-driver-any"] }
// embassy-time = "0.4"
// panic-halt = "0.2"
```

### コード例: HAL (Hardware Abstraction Layer) の活用

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f4xx_hal::{
    pac,
    prelude::*,
    serial::{config::Config, Serial},
    timer::Timer,
    adc::{Adc, config::AdcConfig},
};
use core::fmt::Write;

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    // クロック設定
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr
        .use_hse(8.MHz())
        .sysclk(84.MHz())
        .pclk1(42.MHz())
        .freeze();

    // GPIO 設定
    let gpioa = dp.GPIOA.split();

    // UART 設定 (PA2: TX, PA3: RX)
    let tx_pin = gpioa.pa2.into_alternate();
    let rx_pin = gpioa.pa3.into_alternate();

    let mut serial = Serial::new(
        dp.USART2,
        (tx_pin, rx_pin),
        Config::default().baudrate(115200.bps()),
        &clocks,
    ).unwrap();

    // ADC 設定
    let adc_pin = gpioa.pa0.into_analog();
    let mut adc = Adc::adc1(dp.ADC1, true, AdcConfig::default());

    // タイマー設定 (1秒周期)
    let mut timer = dp.TIM2.counter_ms(&clocks);
    timer.start(1000.millis()).unwrap();

    writeln!(serial, "System initialized at {}MHz\r", clocks.sysclk().to_MHz()).unwrap();

    loop {
        // ADC 読み取り
        let value: u16 = adc.read(&mut adc_pin).unwrap();
        let voltage = value as f32 * 3.3 / 4096.0;

        writeln!(serial, "ADC: {} ({}V)\r", value, voltage).unwrap();

        // タイマー待機
        nb::block!(timer.wait()).unwrap();
    }
}
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

    /// ブラー (ボックスフィルタ)
    pub fn blur(&mut self, radius: u32) {
        let w = self.width as usize;
        let h = self.height as usize;
        let r = radius as usize;
        let mut output = self.pixels.clone();

        for y in 0..h {
            for x in 0..w {
                let mut sum_r = 0u32;
                let mut sum_g = 0u32;
                let mut sum_b = 0u32;
                let mut count = 0u32;

                for dy in -(r as i32)..=(r as i32) {
                    for dx in -(r as i32)..=(r as i32) {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            let idx = ((ny as usize) * w + nx as usize) * 4;
                            sum_r += self.pixels[idx] as u32;
                            sum_g += self.pixels[idx + 1] as u32;
                            sum_b += self.pixels[idx + 2] as u32;
                            count += 1;
                        }
                    }
                }

                let idx = (y * w + x) * 4;
                output[idx] = (sum_r / count) as u8;
                output[idx + 1] = (sum_g / count) as u8;
                output[idx + 2] = (sum_b / count) as u8;
            }
        }

        self.pixels = output;
    }

    /// 明るさ調整
    pub fn adjust_brightness(&mut self, factor: f64) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            chunk[0] = ((chunk[0] as f64 * factor).min(255.0).max(0.0)) as u8;
            chunk[1] = ((chunk[1] as f64 * factor).min(255.0).max(0.0)) as u8;
            chunk[2] = ((chunk[2] as f64 * factor).min(255.0).max(0.0)) as u8;
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

### コード例: DOM 操作 (web-sys)

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, Window};

#[wasm_bindgen]
pub fn create_todo_app() -> Result<(), JsValue> {
    let window: Window = web_sys::window().ok_or("No window")?;
    let document: Document = window.document().ok_or("No document")?;
    let body: HtmlElement = document.body().ok_or("No body")?;

    // コンテナ作成
    let container = document.create_element("div")?;
    container.set_id("todo-app");
    container.set_class_name("container");

    // ヘッダー
    let header = document.create_element("h1")?;
    header.set_text_content(Some("Rust WASM TODO"));
    container.append_child(&header)?;

    // 入力フォーム
    let input = document.create_element("input")?;
    input.set_attribute("type", "text")?;
    input.set_attribute("placeholder", "新しいタスク...")?;
    input.set_id("todo-input");
    container.append_child(&input)?;

    // 追加ボタン
    let button = document.create_element("button")?;
    button.set_text_content(Some("追加"));

    // ボタンのクリックハンドラ
    let document_clone = document.clone();
    let closure = Closure::wrap(Box::new(move || {
        let input = document_clone
            .get_element_by_id("todo-input")
            .unwrap();
        let input: web_sys::HtmlInputElement = input.dyn_into().unwrap();
        let value = input.value();

        if !value.is_empty() {
            let list = document_clone.get_element_by_id("todo-list").unwrap();
            let item = document_clone.create_element("li").unwrap();
            item.set_text_content(Some(&value));
            list.append_child(&item).unwrap();
            input.set_value("");
        }
    }) as Box<dyn FnMut()>);

    button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    closure.forget(); // メモリリーク注意! 実用ではライフタイム管理が必要
    container.append_child(&button)?;

    // リスト
    let list = document.create_element("ul")?;
    list.set_id("todo-list");
    container.append_child(&list)?;

    body.append_child(&container)?;
    Ok(())
}
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

### コード例: WASI プラグインシステム

```rust
// プラグインのインターフェース定義
use std::io::{self, BufRead, Write};

/// WASI ベースのプラグイン — stdin/stdout でホストと通信
fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;

        // JSON-RPC 風のプロトコル
        if line.starts_with("TRANSFORM:") {
            let input = &line["TRANSFORM:".len()..];
            let output = transform(input);
            writeln!(stdout, "RESULT:{}", output)?;
        } else if line == "QUIT" {
            break;
        }
    }

    Ok(())
}

fn transform(input: &str) -> String {
    // プラグインの処理ロジック
    input.to_uppercase()
}

// ホスト側 (wasmtime API を使用)
// use wasmtime::*;
// use wasmtime_wasi::WasiCtxBuilder;
//
// let engine = Engine::default();
// let module = Module::from_file(&engine, "plugin.wasm")?;
// let wasi = WasiCtxBuilder::new()
//     .stdin(Box::new(input_pipe))
//     .stdout(Box::new(output_pipe))
//     .build();
// let instance = Instance::new(&mut store, &module, &[])?;
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

### コード例: Cargo.toml の WASM 最適化設定

```toml
[package]
name = "my-wasm-app"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = [
    "console",
    "Document",
    "Element",
    "HtmlElement",
    "Window",
    "Performance",
    "CanvasRenderingContext2d",
    "HtmlCanvasElement",
    "ImageData",
] }
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
console_error_panic_hook = "0.1"

[profile.release]
opt-level = "s"       # サイズ最適化 ("z" でさらに小さく)
lto = true            # Link-Time Optimization
codegen-units = 1     # 最大最適化 (ビルドは遅い)
strip = true          # デバッグ情報除去
panic = "abort"       # パニック時の巻き戻しなし (サイズ削減)

[profile.release.package."*"]
opt-level = "s"       # 依存クレートもサイズ最適化
```

---

## 5. WASM の高度なパターン

### コード例: Web Worker との連携

```rust
use wasm_bindgen::prelude::*;
use js_sys::{Promise, Uint8Array};

/// Web Worker 内で重い計算を実行
#[wasm_bindgen]
pub fn heavy_computation(data: &[u8]) -> Vec<u8> {
    // ソート、フィルタ、変換等の重い処理
    let mut result = data.to_vec();
    result.sort_unstable();

    // SHA-256 風のハッシュ計算 (簡易版)
    let mut hash = [0u8; 32];
    for (i, byte) in data.iter().enumerate() {
        hash[i % 32] ^= byte;
        hash[(i + 1) % 32] = hash[(i + 1) % 32].wrapping_add(*byte);
    }

    hash.to_vec()
}

/// SharedArrayBuffer を使ったゼロコピーデータ共有
#[wasm_bindgen]
pub fn process_shared_buffer(buffer: &Uint8Array, offset: usize, length: usize) -> u32 {
    let mut sum: u32 = 0;
    for i in offset..offset + length {
        sum += buffer.get_index(i as u32) as u32;
    }
    sum
}
```

```javascript
// worker.js
importScripts('./pkg/my_wasm.js');

self.onmessage = async function(e) {
    const { init, heavy_computation } = await wasm_bindgen('./pkg/my_wasm_bg.wasm');
    const result = heavy_computation(e.data);
    self.postMessage(result);
};

// main.js
const worker = new Worker('worker.js');
worker.onmessage = (e) => {
    console.log('Worker result:', e.data);
};
worker.postMessage(new Uint8Array([1, 2, 3, 4, 5]));
```

### コード例: Canvas 描画

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[wasm_bindgen]
pub struct GameRenderer {
    ctx: CanvasRenderingContext2d,
    width: f64,
    height: f64,
    particles: Vec<Particle>,
}

struct Particle {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    radius: f64,
    color: String,
}

#[wasm_bindgen]
impl GameRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str, num_particles: usize) -> Result<GameRenderer, JsValue> {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document.get_element_by_id(canvas_id).unwrap();
        let canvas: HtmlCanvasElement = canvas.dyn_into()?;
        let ctx = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;

        let width = canvas.width() as f64;
        let height = canvas.height() as f64;

        // パーティクル初期化
        let mut particles = Vec::with_capacity(num_particles);
        for i in 0..num_particles {
            let angle = (i as f64 / num_particles as f64) * std::f64::consts::TAU;
            particles.push(Particle {
                x: width / 2.0,
                y: height / 2.0,
                vx: angle.cos() * 2.0,
                vy: angle.sin() * 2.0,
                radius: 3.0,
                color: format!("hsl({}, 80%, 60%)", (i * 360 / num_particles)),
            });
        }

        Ok(GameRenderer {
            ctx,
            width,
            height,
            particles,
        })
    }

    pub fn update(&mut self) {
        for p in &mut self.particles {
            p.x += p.vx;
            p.y += p.vy;

            // 壁との反射
            if p.x <= p.radius || p.x >= self.width - p.radius {
                p.vx = -p.vx;
            }
            if p.y <= p.radius || p.y >= self.height - p.radius {
                p.vy = -p.vy;
            }

            // 範囲内にクランプ
            p.x = p.x.clamp(p.radius, self.width - p.radius);
            p.y = p.y.clamp(p.radius, self.height - p.radius);
        }
    }

    pub fn render(&self) {
        // 背景クリア
        self.ctx.set_fill_style_str("rgba(0, 0, 0, 0.1)");
        self.ctx.fill_rect(0.0, 0.0, self.width, self.height);

        // パーティクル描画
        for p in &self.particles {
            self.ctx.begin_path();
            self.ctx.arc(p.x, p.y, p.radius, 0.0, std::f64::consts::TAU).unwrap();
            self.ctx.set_fill_style_str(&p.color);
            self.ctx.fill();
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}
```

---

## 6. 比較表

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

### WASM バイナリサイズの最適化レベル

| 設定 | opt-level | lto | サイズ目安 | ビルド時間 |
|---|---|---|---|---|
| デバッグ | 0 | false | 2-10 MB | 速い |
| バランス | 2 | false | 200-500 KB | 中程度 |
| サイズ優先 | "s" | true | 50-200 KB | 遅い |
| 最小サイズ | "z" | true | 30-150 KB | 最も遅い |

---

## 7. アンチパターン

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

### アンチパターン3: WASM での大量 DOM 操作

```rust
// NG: 個別に DOM 操作 (JS-WASM 境界のオーバーヘッド)
#[wasm_bindgen]
pub fn bad_create_list(items: Vec<String>) {
    let document = web_sys::window().unwrap().document().unwrap();
    let list = document.create_element("ul").unwrap();

    for item in &items {
        let li = document.create_element("li").unwrap(); // 毎回 FFI 呼び出し
        li.set_text_content(Some(item));                  // 毎回 FFI 呼び出し
        list.append_child(&li).unwrap();                  // 毎回 FFI 呼び出し
    }

    document.body().unwrap().append_child(&list).unwrap();
}

// OK: innerHTML で一括挿入 (FFI 呼び出し回数を最小化)
#[wasm_bindgen]
pub fn good_create_list(items: Vec<String>) {
    let document = web_sys::window().unwrap().document().unwrap();
    let list = document.create_element("ul").unwrap();

    // Rust 側で HTML を構築してから一括挿入
    let html: String = items
        .iter()
        .map(|item| format!("<li>{}</li>", html_escape(item)))
        .collect();

    list.set_inner_html(&html); // 1回の FFI 呼び出し

    document.body().unwrap().append_child(&list).unwrap();
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
```

### アンチパターン4: 組み込みでの動的割当の過剰使用

```rust
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;

// NG: 限られたメモリで動的割当を多用
fn bad_process() {
    let mut data = Vec::new(); // ヒープ割当
    for i in 0..1000 {
        data.push(format!("item_{}", i)); // String もヒープ割当
    }
    // メモリ不足でパニックの可能性
}

// OK: 固定サイズのバッファを使用
fn good_process() {
    let mut buffer = [0u8; 256]; // スタック上に固定サイズ
    let mut count = 0;
    for i in 0..256 {
        buffer[i] = (i % 256) as u8;
        count += 1;
    }
    // メモリ使用量が予測可能
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

### Q4: embassy と RTIC の違いは?

**A:** embassy は async/await ベースで、タスクを `.await` で中断・再開できます。RTIC は割り込み駆動で、ハードウェア割り込みに基づくタスク実行モデルです。新しいプロジェクトには embassy が推奨されます。async/await の方が Rust の標準的なプログラミングモデルに近く、学習コストが低いです。

### Q5: WASM で Web フレームワークを使うには?

**A:** Yew, Leptos, Dioxus が主要な選択肢です。

| フレームワーク | 特徴 | アーキテクチャ |
|---|---|---|
| **Yew** | React 風コンポーネントモデル | クライアント SPA |
| **Leptos** | 細粒度リアクティビティ、SSR 対応 | フルスタック |
| **Dioxus** | React 風、マルチプラットフォーム | Web + Desktop + Mobile |

```rust
// Leptos の例
use leptos::*;

#[component]
fn Counter() -> impl IntoView {
    let (count, set_count) = create_signal(0);

    view! {
        <button on:click=move |_| set_count.update(|n| *n += 1)>
            "Click me: " {count}
        </button>
    }
}
```

### Q6: WASM のメモリ制限は?

**A:** デフォルトでは 1 ページ (64KB) から開始し、最大で約 4GB (32-bit アドレス空間の制限) まで成長します。`memory.grow` 命令でページ単位 (64KB) で拡張されます。ブラウザごとに実際の上限は異なります。

---

## 8. テストとデバッグ

### WASM ユニットテスト (wasm-bindgen-test)

```rust
// tests/web.rs
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_add() {
    assert_eq!(my_lib::add(2, 3), 5);
}

#[wasm_bindgen_test]
async fn test_async_fetch() {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, Response};

    let mut opts = RequestInit::new();
    opts.method("GET");

    let request = Request::new_with_str_and_init(
        "https://httpbin.org/get",
        &opts,
    ).unwrap();

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await.unwrap();
    let resp: Response = resp_value.dyn_into().unwrap();
    assert_eq!(resp.status(), 200);
}

#[wasm_bindgen_test]
fn test_dom_manipulation() {
    let document = web_sys::window().unwrap().document().unwrap();
    let div = document.create_element("div").unwrap();
    div.set_id("test-div");
    div.set_text_content(Some("Hello WASM"));
    document.body().unwrap().append_child(&div).unwrap();

    let found = document.get_element_by_id("test-div").unwrap();
    assert_eq!(found.text_content().unwrap(), "Hello WASM");
}
```

```bash
# テスト実行
wasm-pack test --headless --chrome
wasm-pack test --headless --firefox
wasm-pack test --node
```

### 組み込みテスト戦略

```rust
// 組み込みロジックのテストはホスト環境で実行可能にする
// HAL に依存しないロジックを分離

/// ハードウェア非依存のフィルタロジック
pub struct MovingAverage<const N: usize> {
    buffer: [f32; N],
    index: usize,
    count: usize,
}

impl<const N: usize> MovingAverage<N> {
    pub const fn new() -> Self {
        Self {
            buffer: [0.0; N],
            index: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, value: f32) -> f32 {
        self.buffer[self.index] = value;
        self.index = (self.index + 1) % N;
        if self.count < N {
            self.count += 1;
        }
        self.average()
    }

    pub fn average(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f32 = self.buffer[..self.count].iter().sum();
        sum / self.count as f32
    }
}

// ホスト環境で通常の cargo test として実行
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average() {
        let mut avg = MovingAverage::<4>::new();
        assert_eq!(avg.push(10.0), 10.0);
        assert_eq!(avg.push(20.0), 15.0);
        assert_eq!(avg.push(30.0), 20.0);
        assert_eq!(avg.push(40.0), 25.0);
        // バッファが一周
        assert_eq!(avg.push(50.0), 35.0); // (20+30+40+50)/4
    }

    #[test]
    fn test_pid_controller() {
        let mut pid = PidController::new(1.0, 0.1, 0.05);
        let output = pid.update(100.0, 0.0, 0.01);
        assert!(output > 0.0, "正の方向に制御されること");
    }
}

/// PID 制御器（組み込みでもホストでもテスト可能）
pub struct PidController {
    kp: f32,
    ki: f32,
    kd: f32,
    integral: f32,
    prev_error: f32,
}

impl PidController {
    pub fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self { kp, ki, kd, integral: 0.0, prev_error: 0.0 }
    }

    pub fn update(&mut self, setpoint: f32, measurement: f32, dt: f32) -> f32 {
        let error = setpoint - measurement;
        self.integral += error * dt;
        let derivative = (error - self.prev_error) / dt;
        self.prev_error = error;

        self.kp * error + self.ki * self.integral + self.kd * derivative
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}
```

### WASM パフォーマンスプロファイリング

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

/// パフォーマンス計測ユーティリティ
#[wasm_bindgen]
pub struct PerfTimer {
    label: String,
    start: f64,
}

#[wasm_bindgen]
impl PerfTimer {
    #[wasm_bindgen(constructor)]
    pub fn new(label: &str) -> Self {
        let start = now();
        web_sys::console::time_with_label(label);
        Self {
            label: label.to_string(),
            start,
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        now() - self.start
    }

    pub fn end(self) -> f64 {
        let elapsed = self.elapsed_ms();
        web_sys::console::time_end_with_label(&self.label);
        elapsed
    }
}

/// メモリ使用量の監視
#[wasm_bindgen]
pub fn wasm_memory_usage() -> usize {
    // WASM のリニアメモリサイズを取得
    core::arch::wasm32::memory_size(0) * 65536 // ページ数 × 64KB
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| no_std | OS なし環境向け。core のみ使用 |
| alloc | ヒープ割当が使えれば Vec/String を追加可能 |
| #![no_main] | エントリポイントを自分で定義 (ベアメタル) |
| wasm-bindgen | Rust <-> JS の型安全なブリッジ |
| wasm-pack | WASM + JS + d.ts + package.json を一括生成 |
| WASI | サーバーサイド WASM。ファイル I/O 対応 |
| サイズ最適化 | opt-level="s", lto=true, wasm-opt |
| embassy | 組み込みasync ランタイム。現代的な開発体験 |
| web-sys | DOM 操作、Canvas、WebGL 等のブラウザ API |
| 固定サイズバッファ | 組み込みでは動的割当より固定サイズを優先 |
| wasm-bindgen-test | ブラウザ内で WASM のユニットテストを実行 |
| PID 制御 | ハードウェア非依存ロジックはホストでテスト可能 |

## 次に読むべきガイド

- [CLIツール](./04-cli-tools.md) — クロスコンパイルの実践
- [FFI](./02-ffi-interop.md) — WASM 以外の言語間連携
- [メモリレイアウト](./00-memory-layout.md) — no_std 環境でのメモリ管理

## 参考文献

1. **Rust and WebAssembly Book**: https://rustwasm.github.io/docs/book/
2. **The Embedded Rust Book**: https://docs.rust-embedded.org/book/
3. **wasm-bindgen Guide**: https://rustwasm.github.io/docs/wasm-bindgen/
4. **Embassy documentation**: https://embassy.dev/
5. **Leptos documentation**: https://leptos.dev/
