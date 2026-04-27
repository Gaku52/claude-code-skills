# Embedded / WASM -- no_std, wasm-bindgen

> Master development techniques for resource-constrained environments through Rust's no_std environment and WebAssembly targets

## What You Will Learn in This Chapter

1. **no_std Programming** -- Rust without the standard library, the alloc crate, embedded targets
2. **WebAssembly (WASM)** -- JS interop with wasm-bindgen, wasm-pack workflow
3. **WASI** -- Server-side WASM and sandboxed execution
4. **embassy** -- async/await runtime for embedded systems
5. **WASM Optimization** -- Binary size reduction, performance tuning


## Prerequisites

Reading the following before this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the contents of [FFI (Foreign Function Interface)](./02-ffi-interop.md)

---

## 1. The Relationship Between no_std and std

```
┌─────────────── Rust Library Hierarchy ────────────────┐
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  std (standard library)                       │   │
│  │  - File I/O, network, threads                 │   │
│  │  - OS-dependent features                      │   │
│  │                                               │   │
│  │  ┌─────────────────────────────────────────┐ │   │
│  │  │  alloc (heap allocation)                 │ │   │
│  │  │  - Box, Vec, String, Arc, Rc            │ │   │
│  │  │  - Requires a heap allocator             │ │   │
│  │  │                                         │ │   │
│  │  │  ┌─────────────────────────────────┐   │ │   │
│  │  │  │  core (core library)             │   │ │   │
│  │  │  │  - Option, Result, Iterator     │   │ │   │
│  │  │  │  - Numeric types, slices, refs   │   │ │   │
│  │  │  │  - No OS dependency, no alloc    │   │ │   │
│  │  │  └─────────────────────────────────┘   │ │   │
│  │  └─────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  #![no_std]       → core only is usable             │
│  #![no_std] + alloc → core + alloc usable           │
│  (default)         → core + alloc + std             │
└─────────────────────────────────────────────────────┘
```

### Features Available / Unavailable in no_std

| Feature | core | alloc | std |
|---|---|---|---|
| Option, Result | o | o | o |
| Iterator | o | o | o |
| Numeric operations | o | o | o |
| Slice operations | o | o | o |
| fmt (formatting) | o | o | o |
| Vec, String | - | o | o |
| Box, Arc, Rc | - | o | o |
| HashMap | - | - | o |
| File I/O | - | - | o |
| Network | - | - | o |
| Threads | - | - | o |
| println! | - | - | o |

---

## 2. no_std Programming

### Code Example 1: no_std Library

```rust
// src/lib.rs
#![no_std]

// Imports from core (many of the same APIs as std)
use core::fmt;

/// no_std-compatible ring buffer
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
            return Err(byte); // Buffer full
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

    /// Returns an iterator
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

/// no_std-compatible fixed-size stack
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

/// no_std-compatible fixed-point arithmetic
#[derive(Clone, Copy, Debug)]
pub struct FixedPoint {
    raw: i32, // 16.16 fixed point
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

### Code Example 2: Bare-Metal Embedded (ARM Cortex-M)

```rust
// src/main.rs
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _; // Halt on panic

// Global allocator (when using alloc)
// use embedded_alloc::Heap;
// #[global_allocator]
// static HEAP: Heap = Heap::empty();

#[entry]
fn main() -> ! {
    // Acquire peripherals
    let peripherals = stm32f4xx_hal::pac::Peripherals::take().unwrap();
    let gpioa = peripherals.GPIOA.split();

    // LED pin configuration (PA5)
    let mut led = gpioa.pa5.into_push_pull_output();

    loop {
        led.set_high();
        cortex_m::asm::delay(8_000_000); // ~1 second (8MHz)
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

### Code Example: Async Embedded with embassy

```rust
#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_time::{Duration, Timer};
use panic_halt as _;

// embassy: async/await runtime for embedded systems
// Cooperatively executes multiple tasks without threads

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
        // Read sensor (I2C/SPI)
        // let value = i2c.read_register(0x48, 0x00).await;
        Timer::after(Duration::from_secs(1)).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_stm32::init(Default::default());

    let led = Output::new(p.PA5, Level::Low, Speed::Low);

    // Spawn async tasks
    spawner.spawn(blink_task(led)).unwrap();
    spawner.spawn(sensor_task()).unwrap();

    // The main task also runs as async
    loop {
        Timer::after(Duration::from_secs(10)).await;
        // Watchdog reset, etc.
    }
}

// Cargo.toml:
// [dependencies]
// embassy-executor = { version = "0.6", features = ["arch-cortex-m"] }
// embassy-stm32 = { version = "0.2", features = ["stm32f401ce", "time-driver-any"] }
// embassy-time = "0.4"
// panic-halt = "0.2"
```

### Code Example: Leveraging the HAL (Hardware Abstraction Layer)

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

    // Clock configuration
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr
        .use_hse(8.MHz())
        .sysclk(84.MHz())
        .pclk1(42.MHz())
        .freeze();

    // GPIO configuration
    let gpioa = dp.GPIOA.split();

    // UART configuration (PA2: TX, PA3: RX)
    let tx_pin = gpioa.pa2.into_alternate();
    let rx_pin = gpioa.pa3.into_alternate();

    let mut serial = Serial::new(
        dp.USART2,
        (tx_pin, rx_pin),
        Config::default().baudrate(115200.bps()),
        &clocks,
    ).unwrap();

    // ADC configuration
    let adc_pin = gpioa.pa0.into_analog();
    let mut adc = Adc::adc1(dp.ADC1, true, AdcConfig::default());

    // Timer configuration (1 second period)
    let mut timer = dp.TIM2.counter_ms(&clocks);
    timer.start(1000.millis()).unwrap();

    writeln!(serial, "System initialized at {}MHz\r", clocks.sysclk().to_MHz()).unwrap();

    loop {
        // ADC read
        let value: u16 = adc.read(&mut adc_pin).unwrap();
        let voltage = value as f32 * 3.3 / 4096.0;

        writeln!(serial, "ADC: {} ({}V)\r", value, voltage).unwrap();

        // Wait on timer
        nb::block!(timer.wait()).unwrap();
    }
}
```

---

## 3. WebAssembly

### WASM Compilation Targets

```
┌────────────── WASM Targets ──────────────┐
│                                              │
│  wasm32-unknown-unknown                      │
│  └─ For browsers / JS runtimes               │
│  └─ JS interop via wasm-bindgen              │
│                                              │
│  wasm32-wasip1 (formerly wasm32-wasi)        │
│  └─ For WASI-compatible runtimes             │
│  └─ File I/O, networking, etc.               │
│  └─ Run with wasmtime, wasmer                │
│                                              │
│  Build tools:                                │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  wasm-pack      │  │  trunk           │  │
│  │  npm packages   │  │  Yew/Leptos SPA  │  │
│  └─────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────┘
```

### Code Example 3: wasm-bindgen Basics

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

/// A function callable from JS
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}! (from Rust/WASM)", name)
}

/// Outputs to the JS console
#[wasm_bindgen]
pub fn log_to_console(message: &str) {
    web_sys::console::log_1(&message.into());
}

/// Manipulating JS objects
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

/// Exposing a struct to JS
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

    /// Grayscale conversion
    pub fn grayscale(&mut self) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            let gray = (0.299 * chunk[0] as f64
                + 0.587 * chunk[1] as f64
                + 0.114 * chunk[2] as f64) as u8;
            chunk[0] = gray;
            chunk[1] = gray;
            chunk[2] = gray;
            // chunk[3] (alpha) is left unchanged
        }
    }

    /// Blur (box filter)
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

    /// Brightness adjustment
    pub fn adjust_brightness(&mut self, factor: f64) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            chunk[0] = ((chunk[0] as f64 * factor).min(255.0).max(0.0)) as u8;
            chunk[1] = ((chunk[1] as f64 * factor).min(255.0).max(0.0)) as u8;
            chunk[2] = ((chunk[2] as f64 * factor).min(255.0).max(0.0)) as u8;
        }
    }
}
```

### Code Example 4: Usage on the JS Side

```javascript
// After wasm-pack build --target web:
import init, { greet, fibonacci, ImageProcessor } from './pkg/my_wasm.js';

async function main() {
    await init();

    // Basic functions
    console.log(greet("World"));        // "Hello, World! (from Rust/WASM)"
    console.log(fibonacci(50));          // 12586269025n

    // Image processing
    const processor = new ImageProcessor(800, 600);
    processor.grayscale();

    // Direct access to Rust memory
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

### Code Example: DOM Manipulation (web-sys)

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, Window};

#[wasm_bindgen]
pub fn create_todo_app() -> Result<(), JsValue> {
    let window: Window = web_sys::window().ok_or("No window")?;
    let document: Document = window.document().ok_or("No document")?;
    let body: HtmlElement = document.body().ok_or("No body")?;

    // Create container
    let container = document.create_element("div")?;
    container.set_id("todo-app");
    container.set_class_name("container");

    // Header
    let header = document.create_element("h1")?;
    header.set_text_content(Some("Rust WASM TODO"));
    container.append_child(&header)?;

    // Input form
    let input = document.create_element("input")?;
    input.set_attribute("type", "text")?;
    input.set_attribute("placeholder", "New task...")?;
    input.set_id("todo-input");
    container.append_child(&input)?;

    // Add button
    let button = document.create_element("button")?;
    button.set_text_content(Some("Add"));

    // Button click handler
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
    closure.forget(); // Watch out for memory leaks! Lifetime management is required in production
    container.append_child(&button)?;

    // List
    let list = document.create_element("ul")?;
    list.set_id("todo-list");
    container.append_child(&list)?;

    body.append_child(&container)?;
    Ok(())
}
```

### Code Example 5: WASI -- Server-Side WASM

```rust
// WASI-compatible program (almost the same as ordinary Rust code)
use std::fs;
use std::io::{self, Read};

fn main() -> io::Result<()> {
    // File read (within the WASI sandbox)
    let content = fs::read_to_string("/input/data.txt")?;
    println!("File contents: {}", content);

    // Read from standard input
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    println!("Input: {}", buffer);

    // Environment variables
    for (key, value) in std::env::vars() {
        println!("{}={}", key, value);
    }

    Ok(())
}

// Build and run:
// $ rustup target add wasm32-wasip1
// $ cargo build --target wasm32-wasip1 --release
// $ wasmtime --dir /input::/path/to/input target/wasm32-wasip1/release/my_app.wasm
```

### Code Example: WASI Plugin System

```rust
// Plugin interface definition
use std::io::{self, BufRead, Write};

/// WASI-based plugin -- communicates with the host via stdin/stdout
fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;

        // JSON-RPC-style protocol
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
    // Plugin processing logic
    input.to_uppercase()
}

// Host side (using the wasmtime API)
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

## 4. wasm-pack Workflow

```
┌──────────── wasm-pack Build Flow ──────────────┐
│                                                   │
│  src/lib.rs                                       │
│    │                                              │
│    ▼                                              │
│  cargo build --target wasm32-unknown-unknown      │
│    │                                              │
│    ▼                                              │
│  wasm-bindgen (generate JS bindings)              │
│    │                                              │
│    ▼                                              │
│  wasm-opt (optional: size optimization)           │
│    │                                              │
│    ▼                                              │
│  pkg/                                             │
│  ├── my_wasm_bg.wasm     (WASM binary)           │
│  ├── my_wasm.js          (JS glue code)          │
│  ├── my_wasm.d.ts        (TypeScript typedefs)   │
│  └── package.json        (npm package)           │
│                                                   │
│  Commands:                                        │
│  $ wasm-pack build --target web                   │
│  $ wasm-pack build --target bundler  (for webpack) │
│  $ wasm-pack build --target nodejs   (for Node.js) │
└───────────────────────────────────────────────────┘
```

### Code Example: WASM Optimization Settings in Cargo.toml

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
opt-level = "s"       # Size optimization ("z" for even smaller)
lto = true            # Link-Time Optimization
codegen-units = 1     # Maximum optimization (slow build)
strip = true          # Strip debug info
panic = "abort"       # No unwinding on panic (size reduction)

[profile.release.package."*"]
opt-level = "s"       # Size-optimize dependent crates as well
```

---

## 5. Advanced WASM Patterns

### Code Example: Integration with Web Workers

```rust
use wasm_bindgen::prelude::*;
use js_sys::{Promise, Uint8Array};

/// Run heavy computation inside a Web Worker
#[wasm_bindgen]
pub fn heavy_computation(data: &[u8]) -> Vec<u8> {
    // Heavy operations like sort, filter, transform, etc.
    let mut result = data.to_vec();
    result.sort_unstable();

    // SHA-256-style hash computation (simplified)
    let mut hash = [0u8; 32];
    for (i, byte) in data.iter().enumerate() {
        hash[i % 32] ^= byte;
        hash[(i + 1) % 32] = hash[(i + 1) % 32].wrapping_add(*byte);
    }

    hash.to_vec()
}

/// Zero-copy data sharing using SharedArrayBuffer
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

### Code Example: Canvas Rendering

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

        // Initialize particles
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

            // Wall reflection
            if p.x <= p.radius || p.x >= self.width - p.radius {
                p.vx = -p.vx;
            }
            if p.y <= p.radius || p.y >= self.height - p.radius {
                p.vy = -p.vy;
            }

            // Clamp within bounds
            p.x = p.x.clamp(p.radius, self.width - p.radius);
            p.y = p.y.clamp(p.radius, self.height - p.radius);
        }
    }

    pub fn render(&self) {
        // Clear background
        self.ctx.set_fill_style_str("rgba(0, 0, 0, 0.1)");
        self.ctx.fill_rect(0.0, 0.0, self.width, self.height);

        // Render particles
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

## 6. Comparison Tables

### WASM Target Comparison

| Item | wasm32-unknown-unknown | wasm32-wasip1 |
|---|---|---|
| Execution environment | Browser / JS runtime | wasmtime / wasmer |
| File I/O | Not available (via web-sys) | Available (within sandbox) |
| Networking | Via fetch API | WASI Socket (experimental) |
| std support | Limited | Nearly complete |
| JS interop | wasm-bindgen | Not required |
| Use cases | Front-end speedup | Server-side, plugins |
| Binary size | Small (tens of KB+) | Medium (hundreds of KB+) |

### Embedded Framework Comparison

| Framework | Target | Features | HAL |
|---|---|---|---|
| embassy | ARM Cortex-M | async/await based | embassy-stm32, etc. |
| RTIC | ARM Cortex-M | Interrupt-driven, static analysis | Custom |
| esp-hal | ESP32 | Full ESP32 series support | Custom |
| Arduino (avr-hal) | AVR | Arduino UNO, etc. | avr-device |

### WASM Binary Size Optimization Levels

| Setting | opt-level | lto | Size estimate | Build time |
|---|---|---|---|---|
| Debug | 0 | false | 2-10 MB | Fast |
| Balanced | 2 | false | 200-500 KB | Moderate |
| Size-priority | "s" | true | 50-200 KB | Slow |
| Minimum size | "z" | true | 30-150 KB | Slowest |

---

## 7. Anti-Patterns

### Anti-Pattern 1: Bloated WASM Binaries

```rust
// NG: Including all unnecessary features
// Cargo.toml:
// [dependencies]
// serde = { version = "1", features = ["derive"] }
// chrono = "0.4"        ← Increases size with the timezone DB
// regex = "1"           ← Increases size with compiled regex

// OK: Choose lightweight alternatives for WASM
// Cargo.toml:
// [dependencies]
// serde = { version = "1", features = ["derive"] }
// serde-wasm-bindgen = "0.6"
// time = "0.3"          ← Lighter than chrono
//
// [profile.release]
// opt-level = "s"       ← Size optimization
// lto = true            ← Link-time optimization
// codegen-units = 1     ← Maximum optimization
// strip = true          ← Strip debug info

// Shrink further with wasm-opt
// $ wasm-opt -Oz -o output.wasm input.wasm
```

### Anti-Pattern 2: Missing Panic Handler in no_std

```rust
// NG: #![no_std] without a panic handler → link error
#![no_std]
// error: `#[panic_handler]` function required

// OK: Define a panic handler
#![no_std]

// Method 1: panic-halt (stops in an infinite loop)
use panic_halt as _;

// Method 2: Custom panic handler
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    // Output the error to UART, etc.
    // unsafe { write_uart(format_args!("{}", info)); }
    loop {
        core::hint::spin_loop();
    }
}
```

### Anti-Pattern 3: Bulk DOM Operations in WASM

```rust
// NG: Individual DOM operations (JS-WASM boundary overhead)
#[wasm_bindgen]
pub fn bad_create_list(items: Vec<String>) {
    let document = web_sys::window().unwrap().document().unwrap();
    let list = document.create_element("ul").unwrap();

    for item in &items {
        let li = document.create_element("li").unwrap(); // FFI call each time
        li.set_text_content(Some(item));                  // FFI call each time
        list.append_child(&li).unwrap();                  // FFI call each time
    }

    document.body().unwrap().append_child(&list).unwrap();
}

// OK: Bulk insert via innerHTML (minimize FFI call count)
#[wasm_bindgen]
pub fn good_create_list(items: Vec<String>) {
    let document = web_sys::window().unwrap().document().unwrap();
    let list = document.create_element("ul").unwrap();

    // Build HTML on the Rust side, then insert in bulk
    let html: String = items
        .iter()
        .map(|item| format!("<li>{}</li>", html_escape(item)))
        .collect();

    list.set_inner_html(&html); // Single FFI call

    document.body().unwrap().append_child(&list).unwrap();
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
```

### Anti-Pattern 4: Excessive Dynamic Allocation in Embedded

```rust
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;

// NG: Heavy use of dynamic allocation in limited memory
fn bad_process() {
    let mut data = Vec::new(); // Heap allocation
    for i in 0..1000 {
        data.push(format!("item_{}", i)); // String is also heap-allocated
    }
    // May panic due to memory exhaustion
}

// OK: Use a fixed-size buffer
fn good_process() {
    let mut buffer = [0u8; 256]; // Fixed size on the stack
    let mut count = 0;
    for i in 0..256 {
        buffer[i] = (i % 256) as u8;
        count += 1;
    }
    // Memory usage is predictable
}
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Basic implementation pattern exercise"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Input value validation"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
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

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Advanced pattern exercise"""

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
        """Delete by key"""
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

**Key Points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Defective configuration file | Verify the path and format of the configuration file |
| Timeout | Network latency / resource shortage | Adjust the timeout value, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access privileges | Verify the executing user's permissions, review settings |
| Data inconsistency | Race conditions in concurrent processing | Introduce locking, manage transactions |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the source
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output or a debugger to verify hypotheses
5. **Fix and run regression tests**: After fixing, also run tests for related areas

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs the inputs and outputs of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Called: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception raised: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Diagnostic procedure when performance issues arise:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Verify whether there are any memory leaks
3. **Check I/O waits**: Verify the state of disk and network I/O
4. **Check connection counts**: Verify the connection pool state

| Issue type | Diagnostic tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## FAQ

### Q1: How does WASM performance compare to native?

**A:** Generally around 60-90% of native performance. Numerical computation is particularly good and may run 2-10x faster than JIT-compiled JS. However, DOM operations have JS-WASM boundary overhead, so they are unsuitable for large numbers of fine-grained DOM operations.

### Q2: How do I use no_std and alloc together?

**A:** After declaring `#![no_std]`, add `extern crate alloc;` and configure a global allocator.

```rust
#![no_std]
extern crate alloc;
use alloc::{vec, vec::Vec, string::String};

// Allocator configuration (for embedded)
use embedded_alloc::LlffHeap as Heap;
#[global_allocator]
static HEAP: Heap = Heap::empty();
```

### Q3: How do I debug WASM?

**A:** (1) Use `console_error_panic_hook` to display stack traces in the browser console on panic. (2) Use the WASM debugger in Chrome DevTools with source maps. (3) Use `wasm2wat` to convert to text format for analysis.

```rust
// Debug panic hook
#[wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}
```

### Q4: What is the difference between embassy and RTIC?

**A:** embassy is async/await based and can suspend/resume tasks at `.await`. RTIC is interrupt-driven and uses an execution model based on hardware interrupts. embassy is recommended for new projects: async/await is closer to Rust's standard programming model and has a lower learning curve.

### Q5: How do I use a web framework with WASM?

**A:** Yew, Leptos, and Dioxus are the main options.

| Framework | Features | Architecture |
|---|---|---|
| **Yew** | React-style component model | Client-side SPA |
| **Leptos** | Fine-grained reactivity, SSR support | Full-stack |
| **Dioxus** | React-style, multi-platform | Web + Desktop + Mobile |

```rust
// Leptos example
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

### Q6: What are the memory limits in WASM?

**A:** By default it starts at 1 page (64KB) and can grow up to about 4GB (the limit of 32-bit address space). It is expanded with the `memory.grow` instruction in page units (64KB). The actual upper limit varies by browser.

---

## 8. Testing and Debugging

### WASM Unit Tests (wasm-bindgen-test)

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
# Run tests
wasm-pack test --headless --chrome
wasm-pack test --headless --firefox
wasm-pack test --node
```

### Embedded Testing Strategy

```rust
// Make embedded logic tests runnable in the host environment
// Separate logic that does not depend on the HAL

/// Hardware-independent filter logic
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

// Run as ordinary cargo test in the host environment
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
        // Buffer wraps around
        assert_eq!(avg.push(50.0), 35.0); // (20+30+40+50)/4
    }

    #[test]
    fn test_pid_controller() {
        let mut pid = PidController::new(1.0, 0.1, 0.05);
        let output = pid.update(100.0, 0.0, 0.01);
        assert!(output > 0.0, "Should be controlled in the positive direction");
    }
}

/// PID controller (testable on both embedded and host)
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

### WASM Performance Profiling

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

/// Performance measurement utility
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

/// Memory usage monitoring
#[wasm_bindgen]
pub fn wasm_memory_usage() -> usize {
    // Get the size of the WASM linear memory
    core::arch::wasm32::memory_size(0) * 65536 // pages × 64KB
}
```

---

## Summary

| Item | Key points |
|---|---|
| no_std | For OS-less environments. Uses core only |
| alloc | Adds Vec/String when heap allocation is available |
| #![no_main] | Define your own entry point (bare metal) |
| wasm-bindgen | Type-safe bridge between Rust and JS |
| wasm-pack | Generates WASM + JS + d.ts + package.json all at once |
| WASI | Server-side WASM. Supports file I/O |
| Size optimization | opt-level="s", lto=true, wasm-opt |
| embassy | Embedded async runtime. Modern development experience |
| web-sys | Browser APIs such as DOM manipulation, Canvas, WebGL |
| Fixed-size buffers | In embedded, prefer fixed-size over dynamic allocation |
| wasm-bindgen-test | Run WASM unit tests in the browser |
| PID control | Hardware-independent logic can be tested on the host |

## Next Guides to Read

- [CLI Tools](./04-cli-tools.md) -- Cross-compilation in practice
- [FFI](./02-ffi-interop.md) -- Cross-language interop beyond WASM
- [Memory Layout](./00-memory-layout.md) -- Memory management in no_std environments

## References

1. **Rust and WebAssembly Book**: https://rustwasm.github.io/docs/book/
2. **The Embedded Rust Book**: https://docs.rust-embedded.org/book/
3. **wasm-bindgen Guide**: https://rustwasm.github.io/docs/wasm-bindgen/
4. **Embassy documentation**: https://embassy.dev/
5. **Leptos documentation**: https://leptos.dev/
