# ブラウザアーキテクチャ

> モダンブラウザはマルチプロセスアーキテクチャで動作する。ブラウザプロセス、レンダラープロセス、GPUプロセス等の役割分担と、タブごとの分離がセキュリティとパフォーマンスにもたらす利点を理解する。Chromium のソースコード構造から IPC メカニズム、Site Isolation、さらにはレンダリングパイプラインの最適化手法まで、ブラウザ内部を体系的に解説する。

## この章で学ぶこと

- [ ] ブラウザのマルチプロセスアーキテクチャを理解する
- [ ] 各プロセスの役割と連携を把握する
- [ ] ブラウザエンジンの主要コンポーネントを学ぶ
- [ ] Chromium のソースコード構造とビルドシステムを理解する
- [ ] IPC（プロセス間通信）の仕組みと Mojo を把握する
- [ ] Site Isolation のセキュリティモデルを理解する
- [ ] レンダリングパイプラインの各段階を詳細に理解する
- [ ] パフォーマンス最適化のための設計原則を習得する

## 前提知識

- HTTPプロトコルの基礎 → 参照: [HTTPの基礎](../../network-fundamentals/docs/02-http/00-http-basics.md)
- HTML/CSSの基本的な構造の理解
- プロセスとスレッドの概念 → 参照: [OS基礎](../../../01-cs-fundamentals/operating-system-guide/docs/)

---

## 1. マルチプロセスアーキテクチャの全体像

### 1.1 なぜマルチプロセスなのか

1990年代のブラウザは単一プロセスで動作していた。Internet Explorer 6 では、1つのタブがクラッシュするとブラウザ全体が落ちるという致命的な問題があった。2008年に Google Chrome がリリースされた際、最大の革新はマルチプロセスアーキテクチャの採用であった。

マルチプロセスにすることで得られる3つの利点:

1. **安定性（Stability）**: 1つのタブがクラッシュしても他のタブには影響しない
2. **セキュリティ（Security）**: サンドボックスにより各タブのアクセスを制限できる
3. **パフォーマンス（Performance）**: マルチコアCPUを活用してタスクを並列処理できる

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  シングルプロセスモデル（旧来のブラウザ）                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  プロセス                                                  │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────────────────┐│  │
│  │  │ タブ1  │ │ タブ2  │ │ タブ3  │ │ ブラウザUI         ││  │
│  │  │        │ │        │ │        │ │                     ││  │
│  │  │ HTML   │ │ HTML   │ │ HTML   │ │ アドレスバー        ││  │
│  │  │ CSS    │ │ CSS    │ │ CSS    │ │ ブックマーク        ││  │
│  │  │ JS     │ │ JS     │ │ JS     │ │ メニュー            ││  │
│  │  └────────┘ └────────┘ └────────┘ └─────────────────────┘│  │
│  │                                                            │  │
│  │  問題: タブ2 でクラッシュ → プロセス全体が終了             │  │
│  │        タブ1, タブ3, ブラウザUI も巻き添えで消失           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  マルチプロセスモデル（Chrome / Chromium）                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ブラウザプロセス（Browser Process）                      │    │
│  │  UI / ネットワーク / ストレージ / デバイス管理            │    │
│  └──────┬──────────────┬──────────────┬─────────────────────┘    │
│         │              │              │                          │
│  ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐                 │
│  │ レンダラー  │ │ レンダラー │ │ レンダラー │                 │
│  │ プロセス    │ │ プロセス   │ │ プロセス   │                 │
│  │ (タブ1)    │ │ (タブ2)   │ │ (タブ3)   │                 │
│  │ サンドボックス│ │ サンドボックス│ │ サンドボックス│              │
│  └─────────────┘ └────────────┘ └────────────┘                 │
│         │              │              │                          │
│  ┌──────▼──────────────▼──────────────▼─────────────────────┐   │
│  │  GPU プロセス（GPU Process）                              │   │
│  │  画面描画 / ハードウェアアクセラレーション                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  利点: タブ2 がクラッシュ → タブ2 のプロセスだけ終了           │
│        タブ1, タブ3 は影響なし                                 │
│        ブラウザUI も正常に動作                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 プロセスの種類と役割

Chromium では以下のプロセスが協調して動作する。

```
Chromium プロセス構成の詳細:

  ┌─────────────────────────────────────────────────────────────┐
  │              ブラウザプロセス（Browser Process）             │
  │                                                             │
  │  ┌─────────────┐ ┌──────────────┐ ┌───────────────────┐   │
  │  │ UI スレッド  │ │ IO スレッド  │ │ Storage スレッド  │   │
  │  │             │ │              │ │                   │   │
  │  │ ・タブ管理  │ │ ・IPC処理    │ │ ・ファイルI/O     │   │
  │  │ ・ナビゲーション│ │ ・ネットワーク│ │ ・DB操作         │   │
  │  │ ・ウィンドウ │ │  ディスパッチ│ │ ・キャッシュ      │   │
  │  └─────────────┘ └──────────────┘ └───────────────────┘   │
  └────────────────────────┬────────────────────────────────────┘
                           │ Mojo IPC
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
  │ レンダラー  │  │ GPU         │  │ ユーティリティ│
  │ プロセス    │  │ プロセス    │  │ プロセス     │
  │             │  │             │  │              │
  │ ・Blink     │  │ ・Skia      │  │ ・ネットワーク│
  │ ・V8        │  │ ・GL/Vulkan │  │  サービス    │
  │ ・CC (合成) │  │ ・ビデオ    │  │ ・オーディオ  │
  │             │  │  デコード   │  │  サービス    │
  │ サンドボックス│  │             │  │ ・データデコーダ│
  │ 内で実行    │  │             │  │              │
  └─────────────┘  └─────────────┘  └──────────────┘

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ 拡張機能     │  │ プラグイン   │  │ Crashpad     │
  │ プロセス     │  │ プロセス     │  │ ハンドラ     │
  │              │  │ (レガシー)   │  │              │
  │ 各拡張機能ごと│  │ PPAPI等     │  │ クラッシュ   │
  │ に独立       │  │              │  │ レポート     │
  └──────────────┘  └──────────────┘  └──────────────┘
```

各プロセスの詳細な役割は以下のとおりである。

| プロセス | 役割 | サンドボックス | メモリ使用量の目安 |
|---------|------|---------------|-------------------|
| ブラウザプロセス | UI制御、ナビゲーション、全体管理 | なし（特権プロセス） | 100-200MB |
| レンダラープロセス | HTML/CSS/JS の処理、DOM構築 | あり（最も厳格） | 50-300MB/タブ |
| GPUプロセス | 画面描画、ビデオデコード | あり | 100-500MB |
| ネットワークサービス | HTTP/HTTPS通信 | あり | 20-50MB |
| ストレージサービス | IndexedDB、Cache API | あり | 10-30MB |
| オーディオサービス | 音声の入出力 | あり | 10-20MB |
| 拡張機能プロセス | Chrome拡張の実行 | 部分的 | 20-100MB/拡張 |

### 1.3 プロセスモデルの選択戦略

Chromium はメモリ状況に応じてプロセスモデルを動的に切り替える。

```
プロセスモデルの戦略:

  ① Process-per-Site-Instance（デフォルト）
     → 同一サイトの同一インスタンスを1プロセスにまとめる
     → example.com のタブA と example.com のタブB → 同一プロセス
     → example.com と other.com → 別プロセス

  ② Process-per-Site
     → 同一サイトの全タブを1プロセスにまとめる
     → メモリ節約モード（低メモリデバイス向け）

  ③ Process-per-Tab
     → タブごとに1プロセス（分離度が最も高い）
     → --process-per-tab フラグで有効化

  ④ Single Process
     → 全てを1プロセスで実行（デバッグ用途のみ）
     → --single-process フラグで有効化

  メモリ制約時の動作:
  ┌────────────────────────────────────────────────────────┐
  │ メモリ残量  │ 動作                                     │
  ├────────────┼──────────────────────────────────────────┤
  │ 十分       │ Process-per-Site-Instance（通常）         │
  │ やや不足   │ 既存プロセスの再利用を積極化              │
  │ 不足       │ バックグラウンドタブのプロセスを解放       │
  │ 深刻       │ タブの破棄（Tab Discarding）              │
  └────────────────────────────────────────────────────────┘
```

### 1.4 コード例: Chrome プロセスの確認

**コード例 1: chrome.processes API によるプロセス情報取得**

```javascript
// Chrome 拡張機能（manifest V3）でプロセス情報を取得する例
// manifest.json に "permissions": ["processes"] が必要

// プロセス一覧の取得
chrome.processes.getProcessInfo(
  [], // 空配列 = 全プロセス
  true, // メモリ情報を含める
  (processes) => {
    for (const [pid, info] of Object.entries(processes)) {
      console.log(`PID: ${pid}`);
      console.log(`  Type: ${info.type}`);
      // type: "browser", "renderer", "gpu", "utility", "extension" 等
      console.log(`  CPU Usage: ${info.cpu.toFixed(2)}%`);
      console.log(`  Private Memory: ${(info.privateMemory / 1024 / 1024).toFixed(1)}MB`);

      // タブに関連付けられたプロセスの場合
      if (info.tasks) {
        info.tasks.forEach(task => {
          console.log(`  Task: ${task.title} (Tab ID: ${task.tabId})`);
        });
      }
    }
  }
);

// 特定のプロセスを監視（メモリリークの検出など）
function monitorRendererProcesses(intervalMs = 5000) {
  const history = new Map();

  setInterval(() => {
    chrome.processes.getProcessInfo([], true, (processes) => {
      for (const [pid, info] of Object.entries(processes)) {
        if (info.type !== 'renderer') continue;

        if (!history.has(pid)) {
          history.set(pid, []);
        }
        const memoryMB = info.privateMemory / 1024 / 1024;
        history.get(pid).push({
          timestamp: Date.now(),
          memory: memoryMB
        });

        // 直近10回分で50MB以上増加していたら警告
        const records = history.get(pid);
        if (records.length >= 10) {
          const oldest = records[records.length - 10].memory;
          const newest = records[records.length - 1].memory;
          if (newest - oldest > 50) {
            console.warn(
              `[Memory Leak?] PID ${pid}: ` +
              `${oldest.toFixed(1)}MB → ${newest.toFixed(1)}MB ` +
              `(+${(newest - oldest).toFixed(1)}MB)`
            );
          }
        }
      }
    });
  }, intervalMs);
}
```

**コード例 2: Performance API によるメインスレッド監視**

```javascript
// レンダラープロセス内のメインスレッドのパフォーマンスを監視する

// Long Task の検出 (50ms以上のタスクを検出)
const longTaskObserver = new PerformanceObserver((entryList) => {
  for (const entry of entryList.getEntries()) {
    console.log(`[Long Task Detected]`);
    console.log(`  Duration: ${entry.duration.toFixed(2)}ms`);
    console.log(`  Start: ${entry.startTime.toFixed(2)}ms`);
    console.log(`  Name: ${entry.name}`);

    // 100ms以上のタスクは深刻なUI遅延の原因
    if (entry.duration > 100) {
      console.error(
        `CRITICAL: Task took ${entry.duration.toFixed(0)}ms. ` +
        `This blocks rendering and causes jank.`
      );
    }

    // 50-100ms のタスクは改善の余地がある
    if (entry.duration > 50 && entry.duration <= 100) {
      console.warn(
        `WARNING: Task took ${entry.duration.toFixed(0)}ms. ` +
        `Consider breaking this into smaller chunks.`
      );
    }
  }
});

longTaskObserver.observe({ type: 'longtask', buffered: true });

// フレームレートの監視
function monitorFrameRate() {
  let lastTime = performance.now();
  let frameCount = 0;
  const fpsHistory = [];

  function tick(currentTime) {
    frameCount++;
    const elapsed = currentTime - lastTime;

    if (elapsed >= 1000) {
      const fps = Math.round((frameCount * 1000) / elapsed);
      fpsHistory.push(fps);

      if (fps < 30) {
        console.error(`[Frame Drop] FPS: ${fps} - Severe jank detected`);
      } else if (fps < 55) {
        console.warn(`[Frame Drop] FPS: ${fps} - Minor jank`);
      }

      frameCount = 0;
      lastTime = currentTime;
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  return {
    getAverageFPS: () => {
      if (fpsHistory.length === 0) return 0;
      return Math.round(
        fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length
      );
    },
    getHistory: () => [...fpsHistory]
  };
}

const fpsMonitor = monitorFrameRate();
```

---

## 2. レンダラープロセスの内部構造

### 2.1 レンダリングパイプラインの全段階

レンダラープロセスの内部では、HTML がピクセルに変換されるまでに複数の段階を経る。この一連の流れをレンダリングパイプラインと呼ぶ。

```
レンダリングパイプライン（Rendering Pipeline）:

  HTML / CSS / JS
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Parse（パース）                                      │
  │    HTML → DOM ツリー                                    │
  │    CSS  → CSSOM（CSS Object Model）                     │
  │                                                         │
  │    ・HTMLパーサーはインクリメンタル（逐次的）に動作       │
  │    ・<script> に遭遇するとパースを中断してJS実行         │
  │    ・defer / async 属性でブロッキングを回避可能          │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. Style（スタイル計算）                                 │
  │    DOM + CSSOM → Computed Style                          │
  │                                                         │
  │    ・カスケーディングルールの適用                         │
  │    ・継承プロパティの解決                                 │
  │    ・相対値（em, %, vh）の絶対値への変換                 │
  │    ・各DOMノードに最終的なスタイルが割り当てられる        │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. Layout（レイアウト）                                  │
  │    Computed Style → Layout Tree（位置とサイズ）          │
  │                                                         │
  │    ・display:none の要素は Layout Tree に含まれない      │
  │    ・::before, ::after 疑似要素は Layout Tree に追加     │
  │    ・フレキシブルボックス、グリッドの計算                 │
  │    ・テキストの改行位置の決定                             │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. Pre-Paint / Paint（ペイント命令の生成）               │
  │    Layout Tree → Paint Records（描画命令のリスト）       │
  │                                                         │
  │    ・描画順序の決定（z-index、スタッキングコンテキスト） │
  │    ・背景色 → ボーダー → テキスト → 子要素の順           │
  │    ・各レイヤーごとにPaint Recordsを生成                 │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 5. Layerize（レイヤー化）                                │
  │    Paint Records → Compositing Layers                    │
  │                                                         │
  │    ・will-change, transform, opacity で昇格             │
  │    ・overflow:scroll の要素は専用レイヤー                │
  │    ・<video>, <canvas> は専用レイヤー                    │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 6. Commit → Compositor Thread                            │
  │    メインスレッドからコンポジタースレッドへ引き渡し       │
  │                                                         │
  │    ※ ここから先はメインスレッドをブロックしない          │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 7. Tiling & Raster（タイリングとラスタライズ）           │
  │    レイヤーをタイルに分割 → ピクセルに変換               │
  │                                                         │
  │    ・ラスタースレッド（複数）で並列処理                   │
  │    ・GPU ラスタライゼーション（OOP-R）の活用             │
  │    ・ビューポート付近のタイルを優先的に処理              │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 8. Draw / Display（表示）                                │
  │    コンポジターフレームを GPU プロセスへ送信              │
  │    → 最終的な画面への合成と表示                          │
  └─────────────────────────────────────────────────────────┘
```

### 2.2 メインスレッドとコンポジタースレッドの分離

レンダラープロセスの中でも、メインスレッドとコンポジタースレッドの分離は非常に重要な設計判断である。

```
レンダラープロセス内のスレッド構成:

  ┌───────────────────────────────────────────────────────────────┐
  │                   レンダラープロセス                          │
  │                                                               │
  │  ┌───────────────────────────────────────────────────────┐   │
  │  │ メインスレッド（Main Thread）                          │   │
  │  │                                                       │   │
  │  │  ┌─────┐ ┌──────┐ ┌────┐ ┌────────┐ ┌──────┐       │   │
  │  │  │Parse│→│Style │→│Layout│→│Pre-Paint│→│Paint │       │   │
  │  │  └─────┘ └──────┘ └────┘ └────────┘ └──┬───┘       │   │
  │  │                                         │            │   │
  │  │  ┌──────────────────────────────────┐   │            │   │
  │  │  │ JavaScript Engine (V8)           │   │            │   │
  │  │  │ ・スクリプト実行                 │   │            │   │
  │  │  │ ・イベントハンドラ               │   │            │   │
  │  │  │ ・requestAnimationFrame          │   │            │   │
  │  │  │ ・GC（ガベージコレクション）     │   │            │   │
  │  │  └──────────────────────────────────┘   │            │   │
  │  └─────────────────────────────────────────┼────────────┘   │
  │                                     Commit │                 │
  │  ┌─────────────────────────────────────────▼────────────┐   │
  │  │ コンポジタースレッド（Compositor Thread）             │   │
  │  │                                                       │   │
  │  │  ・入力イベントの初期処理                             │   │
  │  │  ・スクロール処理（JS ハンドラなしの場合）            │   │
  │  │  ・CSS アニメーション / transition の処理             │   │
  │  │  ・レイヤーの合成                                     │   │
  │  │  ・タイリングの管理                                   │   │
  │  └──────────────────────────────┬────────────────────────┘   │
  │                                 │                             │
  │  ┌──────────────────────────────▼────────────────────────┐   │
  │  │ ラスタースレッド（Raster Threads）× 複数              │   │
  │  │                                                       │   │
  │  │  ・タイルのピクセル化                                 │   │
  │  │  ・GPU ラスタライゼーション                           │   │
  │  └───────────────────────────────────────────────────────┘   │
  │                                                               │
  │  ┌───────────────────────────────────────────────────────┐   │
  │  │ Worker スレッド（オプショナル）                        │   │
  │  │                                                       │   │
  │  │  ・Web Worker                                         │   │
  │  │  ・Service Worker                                     │   │
  │  │  ・Worklet (Paint Worklet, Audio Worklet等)           │   │
  │  └───────────────────────────────────────────────────────┘   │
  └───────────────────────────────────────────────────────────────┘

  コンポジタースレッドの利点:
  ・メインスレッドが JS でビジー → スクロールは滑らか
  ・transform / opacity アニメーション → メインスレッド不要
  ・60fps の維持が容易
```

### 2.3 コンポジターのみで処理できるプロパティ

パフォーマンス最適化において、コンポジターのみで処理できるCSSプロパティを使うことは極めて重要である。

| CSSプロパティ | レイアウト再計算 | ペイント再実行 | コンポジットのみ |
|-------------|---------------|--------------|----------------|
| `width`, `height` | 必要 | 必要 | --- |
| `margin`, `padding` | 必要 | 必要 | --- |
| `top`, `left` (position) | 必要 | 必要 | --- |
| `color`, `background-color` | --- | 必要 | --- |
| `box-shadow` | --- | 必要 | --- |
| `border-radius` | --- | 必要 | --- |
| `transform` | --- | --- | コンポジットのみ |
| `opacity` | --- | --- | コンポジットのみ |
| `filter` | --- | --- | コンポジットのみ |
| `will-change` | --- | --- | レイヤー昇格のヒント |

**コード例 3: コンポジターフレンドリーなアニメーション**

```css
/* アンチパターン: left を使ったアニメーション
   → 毎フレームでレイアウト再計算が発生する */
.slide-bad {
  position: absolute;
  left: 0;
  transition: left 0.3s ease;
}
.slide-bad.active {
  left: 200px;  /* レイアウト → ペイント → コンポジット の全段階が実行される */
}

/* 推奨パターン: transform を使ったアニメーション
   → コンポジタースレッドのみで処理可能 */
.slide-good {
  transform: translateX(0);
  transition: transform 0.3s ease;
  will-change: transform;  /* レイヤー昇格のヒント */
}
.slide-good.active {
  transform: translateX(200px);  /* コンポジットのみ → 高速 */
}

/* アンチパターン: background-color のアニメーション
   → 毎フレームでペイントが発生する */
.fade-bad {
  background-color: #ffffff;
  transition: background-color 0.3s ease;
}
.fade-bad:hover {
  background-color: #f0f0f0;  /* ペイント → コンポジット が毎フレーム実行 */
}

/* 推奨パターン: opacity を使ったフェード
   → コンポジタースレッドのみで処理可能 */
.fade-good {
  opacity: 1;
  transition: opacity 0.3s ease;
}
.fade-good:hover {
  opacity: 0.8;  /* コンポジットのみ → 高速 */
}
```

```javascript
// JavaScript でもコンポジターフレンドリーなアニメーションを書く

// アンチパターン: style.left による直接操作
function animateBad(element, targetX) {
  let current = 0;
  const step = 2;

  function frame() {
    current += step;
    element.style.left = current + 'px'; // レイアウトスラッシング
    if (current < targetX) {
      requestAnimationFrame(frame);
    }
  }
  requestAnimationFrame(frame);
}

// 推奨パターン: Web Animations API + transform
function animateGood(element, targetX) {
  element.animate(
    [
      { transform: 'translateX(0)' },
      { transform: `translateX(${targetX}px)` }
    ],
    {
      duration: 300,
      easing: 'ease-out',
      fill: 'forwards',
      // composite: 'accumulate' // 既存のtransformと合成
    }
  );
}

// 推奨パターン: CSS Custom Properties + transition
function animateWithCustomProps(element, targetX) {
  element.style.setProperty('--translate-x', `${targetX}px`);
  // CSS で: transform: translateX(var(--translate-x));
  //         transition: transform 0.3s ease;
}
```

---

## 3. ブラウザエンジンの比較と歴史

### 3.1 主要エンジンの系譜

```
ブラウザエンジンの系譜（1998-2025）:

  1998  KHTML (KDE Project)
        │
        ├──── 2001  KHTML → WebKit にフォーク (Apple)
        │            │
        │            ├──── 2003  Safari 1.0 (WebKit)
        │            │
        │            ├──── 2008  Chrome 1.0 (WebKit + V8)
        │            │     │
        │            │     └──── 2013  Blink にフォーク (Google)
        │            │            │
        │            │            ├── Chrome (2013~)
        │            │            ├── Opera (2013~)
        │            │            ├── Edge (2020~)
        │            │            ├── Brave (2016~)
        │            │            ├── Vivaldi (2016~)
        │            │            └── Samsung Internet
        │            │
        │            └──── WebKit (Apple が継続開発)
        │                  ├── Safari (macOS / iOS)
        │                  ├── GNOME Web (Epiphany)
        │                  └── iOS上の全ブラウザ
        │
  1998  Gecko (Netscape → Mozilla)
        ├── Firefox (2004~)
        ├── Thunderbird
        └── Servo (実験的並列エンジン, 2012~)

  1997  Trident (Microsoft)
        ├── Internet Explorer (1997-2022)
        └── EdgeHTML (2015-2020)
             └── 廃止 → Chromium ベースへ移行
```

### 3.2 エンジン比較詳細表

| 特性 | Blink (Chromium) | WebKit (Safari) | Gecko (Firefox) |
|------|-----------------|----------------|-----------------|
| 開発元 | Google主導 | Apple主導 | Mozilla Foundation |
| 初リリース | 2013年 | 2003年 | 1998年 |
| レンダリング言語 | C++ | C++ | C++ / Rust (Stylo) |
| JSエンジン | V8 (C++) | JavaScriptCore (C++) | SpiderMonkey (C++ / Rust) |
| プロセスモデル | マルチプロセス | マルチプロセス (限定的) | マルチプロセス (Fission) |
| CSS Grid | 完全対応 | 完全対応 | 完全対応 |
| Web Components | 完全対応 | 完全対応 | 完全対応 |
| WASM | 完全対応 | 完全対応 | 完全対応 |
| 市場シェア (2024) | 約65-70% | 約18-20% | 約3-4% |
| モバイルシェア | 約65% | 約25% (iOS) | 約1% |
| 特徴的技術 | OilPan GC, LayoutNG | Intelligent Tracking Prevention | Stylo (Rust CSS), Fission |

### 3.3 JavaScriptエンジンの比較

| 特性 | V8 (Chrome) | JavaScriptCore (Safari) | SpiderMonkey (Firefox) |
|------|------------|------------------------|----------------------|
| JIT階層 | Sparkplug → Maglev → Turbofan | LLInt → Baseline → DFG → FTL | Baseline → IC → Warp |
| GC方式 | Generational + Incremental + Concurrent | Generational + Concurrent | Generational + Incremental + Concurrent |
| WASM実装 | Liftoff (baseline) + TurboFan (optimizing) | BBQ (baseline) + OMG (optimizing) | Baseline + Ion (optimizing) |
| 組み込み用途 | Node.js, Deno, Bun | React Native (Hermes) | --- |
| 最適化手法 | Hidden Classes, Inline Caches | Structure Chain | Shape + IC |

---

## 4. Chromium のソースコード構造

### 4.1 ディレクトリ構成

Chromium のソースコードは約3,500万行を超える巨大なコードベースである。主要なディレクトリ構成を理解することは、ブラウザアーキテクチャの理解を深めるために重要である。

```
chromium/src/
├── chrome/              # Chrome ブラウザ固有のコード
│   ├── browser/         #   ブラウザプロセスのUI/ロジック
│   ├── renderer/        #   レンダラープロセスのChrome固有部分
│   ├── common/          #   プロセス間で共有するコード
│   └── test/            #   Chrome固有のテスト
│
├── content/             # ブラウザのコアコンテンツ処理
│   ├── browser/         #   コンテンツ層のブラウザプロセス側
│   ├── renderer/        #   コンテンツ層のレンダラープロセス側
│   ├── gpu/             #   GPUプロセスの実装
│   ├── common/          #   プロセス間で共有
│   └── public/          #   公開API（embedder向け）
│
├── third_party/
│   └── blink/           # Blink レンダリングエンジン
│       ├── renderer/
│       │   ├── core/    #     DOM, CSS, Layout, Paint
│       │   ├── modules/ #     Web API (Fetch, WebGL, etc.)
│       │   ├── platform/#     プラットフォーム抽象化層
│       │   └── bindings/#     V8 バインディング
│       └── web/         #   Blink の公開インターフェース
│
├── v8/                  # V8 JavaScript エンジン
│   ├── src/
│   │   ├── compiler/    #   JIT コンパイラ
│   │   ├── heap/        #   ガベージコレクタ
│   │   ├── interpreter/ #   Ignition インタプリタ
│   │   └── wasm/        #   WebAssembly 実装
│   └── test/
│
├── gpu/                 # GPU コマンドバッファ
├── cc/                  # Chromium Compositor
├── viz/                 # Visual (表示サービス)
├── ui/                  # UI フレームワーク
├── net/                 # ネットワークスタック
├── mojo/                # Mojo IPC フレームワーク
├── ipc/                 # レガシー IPC
├── base/                # 基礎ライブラリ（スレッド、ファイル等）
├── services/            # サービス化コンポーネント
│   ├── network/         #   ネットワークサービス
│   ├── device/          #   デバイスサービス
│   └── data_decoder/    #   データデコーダサービス
├── components/          # 再利用可能なコンポーネント
└── build/               # ビルドシステム（GN + Ninja）
```

### 4.2 Blink の内部構造

Blink はレンダリングエンジンの心臓部であり、DOM からピクセルへの変換を担当する。

**コード例 4: Blink の DOM ノード実装（簡略化）**

```cpp
// third_party/blink/renderer/core/dom/node.h (簡略化)
// Blink の DOM ノードの基本構造

namespace blink {

class Node : public EventTarget {
 public:
  enum NodeType {
    kElementNode = 1,
    kAttributeNode = 2,
    kTextNode = 3,
    kCommentNode = 8,
    kDocumentNode = 9,
    kDocumentFragmentNode = 11,
  };

  // ツリー構造の走査
  Node* parentNode() const { return parent_; }
  Node* firstChild() const { return first_child_; }
  Node* lastChild() const { return last_child_; }
  Node* nextSibling() const { return next_; }
  Node* previousSibling() const { return previous_; }

  // DOM 操作
  Node* appendChild(Node* new_child);
  Node* removeChild(Node* old_child);
  Node* insertBefore(Node* new_child, Node* ref_child);
  Node* replaceChild(Node* new_child, Node* old_child);

  // レイアウト関連
  LayoutObject* GetLayoutObject() const { return layout_object_; }
  void SetLayoutObject(LayoutObject*);

  // スタイル関連
  const ComputedStyle* GetComputedStyle() const;
  void SetNeedsStyleRecalc(StyleChangeType);

  // ガベージコレクション（Oilpan）
  void Trace(Visitor*) const override;

 private:
  Member<Node> parent_;
  Member<Node> first_child_;
  Member<Node> last_child_;
  Member<Node> next_;
  Member<Node> previous_;
  Member<LayoutObject> layout_object_;
  NodeFlags node_flags_;
};

// Oilpan GC によるメモリ管理
// Blink は独自の GC（Oilpan）を使用する
// V8 の GC とは別に動作し、DOM オブジェクトのライフサイクルを管理
// Member<T> はマネージドポインタで、GC がトレースに使用する

}  // namespace blink
```

---

## 5. プロセス間通信（IPC）と Mojo

### 5.1 Mojo IPC フレームワーク

Chromium のプロセス間通信は Mojo というフレームワークで実装されている。Mojo は型安全なメッセージパッシングシステムで、プロセス間の通信を抽象化する。

```
Mojo IPC の構成:

  ┌─────────────────────────────────────────────────────────────┐
  │                    Mojom IDL ファイル                        │
  │                                                             │
  │  // example.mojom                                           │
  │  interface PageHandler {                                    │
  │    GetTitle() => (string title);                            │
  │    SetTitle(string new_title);                              │
  │  };                                                         │
  └────────────────────────┬────────────────────────────────────┘
                           │ コード生成
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ C++ Bindings │ │ Java Bindings│ │ JS Bindings  │
  │              │ │ (Android)    │ │              │
  │ Remote<T>   │ │              │ │              │
  │ Receiver<T> │ │              │ │              │
  └──────┬───────┘ └──────────────┘ └──────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              Mojo メッセージパイプ                           │
  │                                                             │
  │  プロセスA                     プロセスB                    │
  │  ┌──────────┐    パイプ       ┌──────────┐                │
  │  │ Remote   │ ═══════════════ │ Receiver │                │
  │  │ (送信側) │  メッセージ →  │ (受信側) │                │
  │  └──────────┘                 └──────────┘                │
  │                                                             │
  │  特性:                                                      │
  │  ・非同期メッセージパッシング                                │
  │  ・型安全（Mojom IDL から自動生成）                         │
  │  ・プロセス内・プロセス間の両方で動作                       │
  │  ・ハンドルの受け渡しが可能                                 │
  └─────────────────────────────────────────────────────────────┘
```

### 5.2 IPC の具体例: URL ナビゲーション

ユーザーがアドレスバーにURLを入力してからページが表示されるまでの、プロセス間の通信フローを詳細に見てみよう。

```
URL ナビゲーションの IPC フロー:

  ユーザー操作  ブラウザプロセス    ネットワーク     レンダラー     GPU
  (入力)       (Browser)          サービス         プロセス       プロセス
     │              │                │               │              │
     │ URL入力      │                │               │              │
     ├─────────────→│                │               │              │
     │              │                │               │              │
     │              │ BeginNavigation │               │              │
     │              │───────────────→│               │              │
     │              │                │               │              │
     │              │                │ DNS解決        │              │
     │              │                │ TCP接続        │              │
     │              │                │ TLSハンドシェイク│             │
     │              │                │ HTTP要求       │              │
     │              │                │               │              │
     │              │                │ レスポンス     │              │
     │              │ ヘッダー受信   │ ヘッダー       │              │
     │              │←───────────────│               │              │
     │              │                │               │              │
     │              │ Content-Type 判定               │              │
     │              │ (text/html → レンダラー起動)    │              │
     │              │                │               │              │
     │              │ CommitNavigation│               │              │
     │              │────────────────────────────────→│              │
     │              │                │               │              │
     │              │                │  ボディ転送    │              │
     │              │                │──────────────→│              │
     │              │                │               │              │
     │              │                │               │ HTML Parse   │
     │              │                │               │ DOM構築      │
     │              │                │               │ Style計算    │
     │              │                │               │ Layout       │
     │              │                │               │ Paint        │
     │              │                │               │              │
     │              │                │               │ 描画コマンド │
     │              │                │               │─────────────→│
     │              │                │               │              │
     │              │                │               │          画面表示
     │              │                │               │              │
     │              │ DidFinishLoad  │               │              │
     │              │←───────────────────────────────│              │
     │              │                │               │              │
     │  ページ表示  │                │               │              │
     │←─────────────│                │               │              │
```

**コード例 5: Mojo インターフェース定義と使用例**

```cpp
// --- Mojom IDL 定義 ---
// services/network/public/mojom/url_loader.mojom (簡略化)

module network.mojom;

// URL ローダーのインターフェース
interface URLLoader {
  // リダイレクトの追跡
  FollowRedirect(
    array<string> removed_headers,
    map<string, string> modified_headers
  );

  // 優先度の変更
  SetPriority(RequestPriority priority, int32 intra_priority_value);
};

// URL ローダークライアントのインターフェース
interface URLLoaderClient {
  // レスポンスの受信
  OnReceiveResponse(URLResponseHead head,
                    handle<data_pipe_consumer>? body);

  // リダイレクトの通知
  OnReceiveRedirect(URLRequestRedirectInfo redirect_info,
                    URLResponseHead head);

  // 完了の通知
  OnComplete(URLLoaderCompletionStatus status);
};

// --- C++ での使用例 ---
// content/browser/loader/navigation_url_loader.cc (簡略化)

#include "services/network/public/mojom/url_loader.mojom.h"

class NavigationURLLoader {
 public:
  void Start(const GURL& url) {
    // ネットワークサービスへの接続
    mojo::Remote<network::mojom::URLLoaderFactory> factory;
    GetNetworkService()->CreateURLLoaderFactory(
        factory.BindNewPipeAndPassReceiver());

    // URLLoader の作成とリクエスト送信
    mojo::Remote<network::mojom::URLLoader> loader;
    mojo::PendingRemote<network::mojom::URLLoaderClient> client;
    auto client_receiver = client.InitWithNewPipeAndPassReceiver();

    auto request = network::ResourceRequest::New();
    request->url = url;
    request->method = "GET";

    factory->CreateLoaderAndStart(
        loader.BindNewPipeAndPassReceiver(),
        /*request_id=*/0,
        /*options=*/0,
        std::move(request),
        std::move(client),
        /*traffic_annotation=*/net::MutableNetworkTrafficAnnotationTag()
    );
  }
};
```

---

## 6. Site Isolation（サイト分離）

### 6.1 Site Isolation の背景と目的

2018年に発見された Spectre / Meltdown 脆弱性は、プロセスのメモリ空間を超えてデータを読み取ることが理論上可能であることを示した。これを受け、Chromium は Site Isolation を全面的に導入した。

Site Isolation とは、異なるサイト（origin ではなく site 単位）のコンテンツを必ず別のプロセスで実行する仕組みである。これにより、たとえ Spectre 攻撃が成功しても、攻撃者のプロセスには自サイトのデータしか存在しないため、他サイトのデータは読み取れない。

```
Site Isolation の動作:

  ┌──────────────────────────────────────────────────────────────┐
  │ Site Isolation なし（旧来のモデル）                          │
  │                                                              │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │ レンダラープロセス（1つのプロセス内）                 │   │
  │  │                                                      │   │
  │  │  ┌──────────────┐  ┌──────────────────────────────┐ │   │
  │  │  │ example.com  │  │ <iframe src="evil.com">     │ │   │
  │  │  │              │  │                              │ │   │
  │  │  │ ユーザーの   │  │ Spectre 攻撃で              │ │   │
  │  │  │ 個人情報     │  │ example.com のメモリを       │ │   │
  │  │  │ Cookie等    │  │ 読み取り可能!               │ │   │
  │  │  └──────────────┘  └──────────────────────────────┘ │   │
  │  └──────────────────────────────────────────────────────┘   │
  │                                                              │
  │ Site Isolation あり（現在のモデル）                          │
  │                                                              │
  │  ┌────────────────────┐  ┌────────────────────────────┐    │
  │  │ レンダラープロセス A│  │ レンダラープロセス B       │    │
  │  │                    │  │                            │    │
  │  │  ┌──────────────┐ │  │  ┌──────────────────────┐ │    │
  │  │  │ example.com  │ │  │  │ evil.com (iframe)    │ │    │
  │  │  │              │ │  │  │                      │ │    │
  │  │  │ ユーザーの   │ │  │  │ 別プロセスなので     │ │    │
  │  │  │ 個人情報     │ │  │  │ Spectre でも         │ │    │
  │  │  │ Cookie等    │ │  │  │ 読み取り不可能       │ │    │
  │  │  └──────────────┘ │  │  └──────────────────────┘ │    │
  │  └────────────────────┘  └────────────────────────────┘    │
  │                                                              │
  │  プロセス境界 = セキュリティ境界                             │
  └──────────────────────────────────────────────────────────────┘

  Site と Origin の違い:
  ┌───────────────────────────────────────────────────────┐
  │ URL                         │ Site         │ Origin  │
  ├────────────────────────────┼──────────────┼─────────┤
  │ https://a.example.com:443  │ example.com  │ a.example.com:443 │
  │ https://b.example.com:443  │ example.com  │ b.example.com:443 │
  │ https://example.com:8080   │ example.com  │ example.com:8080  │
  │ https://other.com          │ other.com    │ other.com:443     │
  └───────────────────────────────────────────────────────┘

  → a.example.com と b.example.com は同じ Site → 同一プロセス可
  → example.com と other.com は異なる Site → 必ず別プロセス
```

### 6.2 Site Isolation のメモリコスト

Site Isolation はセキュリティを大幅に向上させるが、プロセス数の増加によりメモリ消費が増大する。

| シナリオ | Site Isolation なし | Site Isolation あり | 増加量 |
|---------|-------------------|-------------------|--------|
| タブ5個（全て同一サイト） | プロセス1個 | プロセス1個 | 増加なし |
| タブ5個（全て異なるサイト） | プロセス1-5個 | プロセス5個 | 0-400% |
| 1ページ内に異なるサイトのiframe 3個 | プロセス1個 | プロセス4個 | 300% |
| 一般的なWeb閲覧 | --- | --- | 約10-15%増 |

### 6.3 Cross-Origin Read Blocking (CORB) と CORP

Site Isolation を補完するセキュリティ機構として、CORB と CORP がある。

```javascript
// CORB (Cross-Origin Read Blocking)
// ブラウザが自動的にクロスオリジンの機密データを保護

// 例: 攻撃者が <img> タグで JSON データを読み取ろうとする
// <img src="https://bank.example/api/account"> ← CORB がブロック

// CORB がブロックする Content-Type:
// - text/html
// - application/json
// - text/xml / application/xml

// CORP (Cross-Origin-Resource-Policy)
// サーバー側でリソースの読み込みを制限するヘッダー

// サーバー側の設定例
// 同一オリジンからのみ読み込み可能
// Cross-Origin-Resource-Policy: same-origin

// 同一サイトからのみ読み込み可能
// Cross-Origin-Resource-Policy: same-site

// どのオリジンからも読み込み可能
// Cross-Origin-Resource-Policy: cross-origin

// --- 関連: COOP と COEP ---

// COOP (Cross-Origin-Opener-Policy)
// window.opener の参照をクロスオリジン間で遮断
// Cross-Origin-Opener-Policy: same-origin

// COEP (Cross-Origin-Embedder-Policy)
// クロスオリジンリソースの読み込みに明示的な許可を要求
// Cross-Origin-Embedder-Policy: require-corp

// COOP + COEP の設定で SharedArrayBuffer が利用可能に
// （Spectre 対策として、デフォルトでは無効化されている）

// 確認方法
if (crossOriginIsolated) {
  // SharedArrayBuffer が利用可能
  const sab = new SharedArrayBuffer(1024);
  console.log('Cross-origin isolated:', crossOriginIsolated);
} else {
  console.log('SharedArrayBuffer は使用不可');
  console.log('COOP と COEP ヘッダーを設定してください');
}
```

---

## 7. サンドボックスとセキュリティモデル

### 7.1 レンダラーサンドボックス

レンダラープロセスのサンドボックスは、Chromium のセキュリティの基盤である。たとえレンダラープロセスが悪意あるコードに侵害されても、サンドボックスにより OS レベルの操作が制限される。

```
サンドボックスの制限（OS 別）:

  ┌───────────────────────────────────────────────────────────┐
  │ レンダラーサンドボックスの制限                             │
  │                                                           │
  │ ┌─────────────────────────────────────────────────┐      │
  │ │ 禁止される操作                                   │      │
  │ │                                                   │      │
  │ │ - ファイルシステムへの直接アクセス               │      │
  │ │ - ネットワークソケットの直接作成                 │      │
  │ │ - 他プロセスへの直接アクセス                     │      │
  │ │ - デバイス（カメラ、マイク）への直接アクセス     │      │
  │ │ - クリップボードへの直接アクセス                 │      │
  │ │ - ディスプレイサーバーへの直接接続               │      │
  │ └─────────────────────────────────────────────────┘      │
  │                                                           │
  │ ┌─────────────────────────────────────────────────┐      │
  │ │ 許可される操作（IPC 経由で間接的に）             │      │
  │ │                                                   │      │
  │ │ + ブラウザプロセスへの IPC メッセージ送信        │      │
  │ │ + GPU プロセスへの描画コマンド送信              │      │
  │ │ + 共有メモリの読み書き（限定的）                │      │
  │ │ + CPU 演算（V8 JIT コンパイル含む）             │      │
  │ └─────────────────────────────────────────────────┘      │
  │                                                           │
  │ OS 固有の実装:                                            │
  │                                                           │
  │ Windows: Restricted Token + Job Object + Desktop Isolation│
  │ macOS:   Seatbelt (sandbox-exec) プロファイル             │
  │ Linux:   seccomp-bpf + Namespaces + AppArmor              │
  │ Android: SELinux + seccomp-bpf (isolatedProcess)          │
  │ ChromeOS: Minijail + seccomp-bpf + Namespaces            │
  └───────────────────────────────────────────────────────────┘
```

### 7.2 ブラウザプロセスのセキュリティチェック

ブラウザプロセスは「信頼されたプロセス」として、レンダラープロセスからのリクエストを検証する。

```javascript
// レンダラープロセスからの IPC メッセージの検証（概念的なコード）

// ブラウザプロセス側での検証ロジック
class SecurityChecker {

  // ファイルアクセス要求の検証
  validateFileAccess(rendererProcessId, filePath) {
    // 1. レンダラーが要求したファイルパスの正規化
    const normalizedPath = this.normalizePath(filePath);

    // 2. パストラバーサル攻撃の検出
    if (normalizedPath.includes('..') || normalizedPath.includes('~')) {
      this.killRenderer(rendererProcessId, 'PATH_TRAVERSAL_ATTEMPT');
      return false;
    }

    // 3. ダウンロードディレクトリ外へのアクセスを拒否
    if (!normalizedPath.startsWith(this.allowedBasePath)) {
      this.killRenderer(rendererProcessId, 'UNAUTHORIZED_FILE_ACCESS');
      return false;
    }

    // 4. 機密ファイルへのアクセスを拒否
    const sensitivePatterns = ['/etc/passwd', '/etc/shadow', '.ssh/'];
    for (const pattern of sensitivePatterns) {
      if (normalizedPath.includes(pattern)) {
        this.killRenderer(rendererProcessId, 'SENSITIVE_FILE_ACCESS');
        return false;
      }
    }

    return true;
  }

  // ナビゲーション要求の検証
  validateNavigation(rendererProcessId, sourceOrigin, targetURL) {
    // レンダラーが自身のオリジンを詐称していないか確認
    const expectedOrigin = this.getOriginForProcess(rendererProcessId);
    if (sourceOrigin !== expectedOrigin) {
      this.killRenderer(rendererProcessId, 'ORIGIN_SPOOFING');
      return false;
    }

    // chrome:// や file:// への不正なナビゲーションを拒否
    const scheme = new URL(targetURL).protocol;
    if (['chrome:', 'file:', 'chrome-extension:'].includes(scheme)) {
      if (!this.isSchemeAllowed(rendererProcessId, scheme)) {
        return false;
      }
    }

    return true;
  }

  // 不正なレンダラーを強制終了
  killRenderer(processId, reason) {
    console.error(`Killing renderer ${processId}: ${reason}`);
    // chrome://kills で確認可能
    process.kill(processId);
    this.reportBadMessage(processId, reason);
  }
}
```

---

## 8. GPU プロセスとハードウェアアクセラレーション

### 8.1 GPU プロセスの役割

GPUプロセスは全てのレンダラープロセスからの描画コマンドを受け取り、GPUハードウェアを使って画面を描画する。

```
GPU プロセスの構成:

  レンダラープロセス群                  GPU プロセス
  ┌──────────────┐                    ┌──────────────────────────┐
  │ Renderer A   │                    │                          │
  │ ┌──────────┐ │   コマンドバッファ   │  ┌────────────────────┐│
  │ │Compositor│─│────────────────────│──│ コマンドデコーダ   ││
  │ └──────────┘ │                    │  └────────┬───────────┘│
  └──────────────┘                    │           │            │
                                      │           ▼            │
  ┌──────────────┐                    │  ┌────────────────────┐│
  │ Renderer B   │                    │  │ Skia (GPU Backend) ││
  │ ┌──────────┐ │   コマンドバッファ   │  │                    ││
  │ │Compositor│─│────────────────────│──│ ┌───────┐ ┌──────┐││
  │ └──────────┘ │                    │  │ │OpenGL │ │Vulkan│││
  └──────────────┘                    │  │ └───────┘ └──────┘││
                                      │  │ ┌───────┐ ┌──────┐││
  ┌──────────────┐                    │  │ │Metal  │ │D3D12 │││
  │ Renderer C   │                    │  │ │(macOS)│ │(Win) │││
  │ ┌──────────┐ │   コマンドバッファ   │  │ └───────┘ └──────┘││
  │ │Compositor│─│────────────────────│──│                    ││
  │ └──────────┘ │                    │  └────────┬───────────┘│
  └──────────────┘                    │           │            │
                                      │           ▼            │
                                      │  ┌────────────────────┐│
                                      │  │ ディスプレイ出力   ││
                                      │  │ (VSync 同期)       ││
                                      │  └────────────────────┘│
                                      └──────────────────────────┘

  GPU プロセスを分離する理由:
  (1) GPU ドライバのクラッシュがブラウザ全体に影響しない
  (2) GPU リソースの一元管理（VRAM の効率的利用）
  (3) サンドボックスの境界として機能
  (4) GPUドライバは OS カーネルに近い特権的な操作が必要
```

### 8.2 ハードウェアアクセラレーション対象

```
ハードウェアアクセラレーションの対象と確認方法:

  ┌───────────────────────────────────────────────────────────┐
  │ 機能                     │ GPU 利用 │ 確認場所            │
  ├──────────────────────────┼─────────┼────────────────────┤
  │ ページ合成               │ Yes     │ chrome://gpu        │
  │ CSS 3D Transform         │ Yes     │ chrome://gpu        │
  │ CSS Animation            │ Yes     │ DevTools > Layers   │
  │ WebGL / WebGL2           │ Yes     │ chrome://gpu        │
  │ WebGPU                   │ Yes     │ chrome://flags      │
  │ ビデオデコード           │ Yes     │ chrome://media-internals │
  │ ビデオエンコード         │ Yes     │ chrome://gpu        │
  │ Canvas 2D                │ 部分的  │ chrome://flags      │
  │ SVG レンダリング         │ 部分的  │ ---                 │
  │ テキストレンダリング     │ No      │ CPU で処理          │
  │ JavaScript 実行          │ No      │ CPU で処理          │
  │ DOM 操作                 │ No      │ CPU で処理          │
  └───────────────────────────────────────────────────────────┘

  chrome://gpu で確認できる情報:
  ・Graphics Feature Status（各機能の有効/無効状態）
  ・Driver Information（GPU ドライバ情報）
  ・Compositor Information（コンポジター設定）
  ・GpuMemoryBuffers Status（GPU メモリバッファ状態）
```

---

## 9. DevTools によるプロセス・パフォーマンス分析

### 9.1 Chrome タスクマネージャの活用

```
Chrome タスクマネージャの起動と読み方:

  起動方法:
  ・Windows / Linux: Shift + Esc
  ・macOS: Window メニュー → Task Manager
  ・全OS共通: More tools → Task Manager

  ┌────────────────────────────────────────────────────────────┐
  │ Chrome Task Manager                                        │
  ├─────────────────────┬────────┬───────┬────────┬───────────┤
  │ Task                │ Memory │ CPU   │ Network│ Process ID│
  ├─────────────────────┼────────┼───────┼────────┼───────────┤
  │ Browser             │ 180MB  │ 3.2%  │ 0      │ 12345     │
  │ GPU Process         │ 250MB  │ 8.5%  │ 0      │ 12346     │
  │ Network Service     │ 35MB   │ 0.5%  │ 45KB/s │ 12347     │
  │ Audio Service       │ 15MB   │ 0.1%  │ 0      │ 12348     │
  │ Tab: google.com     │ 95MB   │ 1.2%  │ 2KB/s  │ 12350     │
  │ Tab: youtube.com    │ 320MB  │ 22.3% │ 500KB/s│ 12351     │
  │ Tab: docs.google.com│ 150MB  │ 5.1%  │ 1KB/s  │ 12352     │
  │ Subframe: ads.com   │ 45MB   │ 3.0%  │ 10KB/s │ 12353     │
  │ Extension: uBlock   │ 28MB   │ 0.3%  │ 0      │ 12354     │
  │ Service Worker: PWA │ 22MB   │ 0.0%  │ 0      │ 12355     │
  └─────────────────────┴────────┴───────┴────────┴───────────┘

  注目ポイント:
  ・「Subframe: ads.com」→ Site Isolation により別プロセス化された iframe
  ・YouTube の高いCPU使用率 → 動画デコード + JS処理
  ・Memory が異常に高いタブ → メモリリークの可能性
  ・右クリックで列を追加可能: JavaScript Memory, Image Cache 等
```

### 9.2 Performance パネルの活用

**コード例 6: Performance API を使ったボトルネック特定**

```javascript
// パフォーマンス計測のユーティリティクラス
class BrowserPerformanceAnalyzer {

  constructor() {
    this.marks = new Map();
    this.measures = [];
  }

  // レンダリングパイプラインの各段階を計測
  measureRenderingPipeline() {
    // スタイル再計算のコスト測定
    performance.mark('style-start');
    // ... DOM操作やクラス変更 ...
    requestAnimationFrame(() => {
      performance.mark('style-end');
      performance.measure('Style Recalculation', 'style-start', 'style-end');
    });
  }

  // Layout Thrashing の検出
  detectLayoutThrashing() {
    const originalGetComputedStyle = window.getComputedStyle;
    let readCount = 0;
    let writeCount = 0;
    let thrashingDetected = false;

    // getComputedStyle の呼び出しを監視
    window.getComputedStyle = function(...args) {
      readCount++;
      if (writeCount > 0 && readCount > 1) {
        thrashingDetected = true;
        console.warn(
          `[Layout Thrashing] Read-Write-Read pattern detected. ` +
          `Reads: ${readCount}, Writes: ${writeCount}`
        );
      }
      return originalGetComputedStyle.apply(this, args);
    };

    // 一定時間後にリセット
    requestAnimationFrame(() => {
      window.getComputedStyle = originalGetComputedStyle;
      readCount = 0;
      writeCount = 0;
    });

    return { isThrashing: () => thrashingDetected };
  }

  // Navigation Timing の詳細分析
  analyzeNavigationTiming() {
    const timing = performance.getEntriesByType('navigation')[0];
    if (!timing) return null;

    return {
      // DNS ルックアップ
      dns: {
        duration: timing.domainLookupEnd - timing.domainLookupStart,
        label: 'DNS Lookup'
      },
      // TCP 接続（TLS 含む）
      connection: {
        duration: timing.connectEnd - timing.connectStart,
        label: 'TCP + TLS'
      },
      // TTFB (Time to First Byte)
      ttfb: {
        duration: timing.responseStart - timing.requestStart,
        label: 'TTFB'
      },
      // レスポンスダウンロード
      download: {
        duration: timing.responseEnd - timing.responseStart,
        label: 'Download'
      },
      // DOM パース
      domParse: {
        duration: timing.domInteractive - timing.responseEnd,
        label: 'DOM Parse'
      },
      // DOMContentLoaded
      domContentLoaded: {
        duration: timing.domContentLoadedEventEnd
          - timing.domContentLoadedEventStart,
        label: 'DOMContentLoaded handlers'
      },
      // 全体のロード時間
      totalLoad: {
        duration: timing.loadEventEnd - timing.navigationStart,
        label: 'Total Load'
      }
    };
  }

  // Resource Timing の分析
  analyzeResourceTiming() {
    const resources = performance.getEntriesByType('resource');

    const byType = {};
    for (const resource of resources) {
      const type = resource.initiatorType || 'other';
      if (!byType[type]) {
        byType[type] = { count: 0, totalSize: 0, totalDuration: 0 };
      }
      byType[type].count++;
      byType[type].totalSize += resource.transferSize || 0;
      byType[type].totalDuration += resource.duration;
    }

    return {
      totalResources: resources.length,
      byType,
      slowest: resources
        .sort((a, b) => b.duration - a.duration)
        .slice(0, 5)
        .map(r => ({
          name: r.name.split('/').pop(),
          duration: Math.round(r.duration),
          size: r.transferSize
        }))
    };
  }
}

// 使用例
const analyzer = new BrowserPerformanceAnalyzer();

// ページロード後に分析を実行
window.addEventListener('load', () => {
  setTimeout(() => {
    const navTiming = analyzer.analyzeNavigationTiming();
    const resTiming = analyzer.analyzeResourceTiming();

    console.table(
      Object.entries(navTiming).map(([key, val]) => ({
        Phase: val.label,
        Duration: `${val.duration.toFixed(1)}ms`
      }))
    );

    console.log('Resource Summary:', resTiming);
  }, 100);
});
```

### 9.3 chrome://tracing の活用

```
chrome://tracing（Perfetto UI）の使い方:

  1. chrome://tracing にアクセス
  2. 「Record」ボタンをクリック
  3. カテゴリを選択:
     ・blink    → レンダリングエンジンの内部
     ・cc       → コンポジター
     ・gpu      → GPU コマンド
     ・v8       → JavaScript エンジン
     ・netlog   → ネットワーク
     ・loading  → リソースローディング

  4. 操作を行い「Stop」で記録終了
  5. タイムラインで各プロセス/スレッドの動作を確認

  代替: Perfetto UI（https://ui.perfetto.dev/）
  → より高機能な分析ツール
  → SQL クエリでのデータ分析が可能
  → chrome://tracing のデータをインポート可能

  主要なトレースイベント:
  ┌──────────────────────┬──────────────────────────────────┐
  │ イベント名            │ 意味                             │
  ├──────────────────────┼──────────────────────────────────┤
  │ ParseHTML            │ HTML パース                       │
  │ UpdateLayoutTree     │ スタイル計算                      │
  │ Layout               │ レイアウト計算                    │
  │ PrePaint             │ ペイント準備                      │
  │ Paint                │ ペイント命令生成                  │
  │ CompositeLayers      │ レイヤー合成                      │
  │ V8.Execute           │ JavaScript 実行                   │
  │ V8.GCScavenge        │ マイナー GC                       │
  │ V8.GCMarkCompact     │ メジャー GC                       │
  │ ResourceReceivedData │ ネットワークデータ受信            │
  │ DecodeImage          │ 画像デコード                      │
  │ Rasterize            │ ラスタライズ                      │
  └──────────────────────┴──────────────────────────────────┘
```

---

## 10. アンチパターンと回避策

### 10.1 アンチパターン 1: Layout Thrashing（レイアウトスラッシング）

Layout Thrashing とは、JavaScript でスタイルの読み取りと書き込みを交互に行うことで、ブラウザが毎回レイアウトを強制的に再計算する現象である。これはパフォーマンスを著しく低下させる。

```javascript
// ===== アンチパターン: Layout Thrashing =====

// 悪い例: 読み取りと書き込みの交互実行
function resizeAllBoxesBad(boxes) {
  for (const box of boxes) {
    // 読み取り → 強制レイアウト発生
    const width = box.offsetWidth;

    // 書き込み → レイアウトが無効化される
    box.style.width = (width * 1.1) + 'px';

    // 次のループで再び読み取り → 再度強制レイアウト!
    // N個のボックスに対して N回のレイアウト計算 → O(N) 回のレイアウト
  }
  // 100個のボックスで約 100回のレイアウト再計算
  // → 数十ms～数百ms のブロッキング
}

// ===== 推奨パターン: バッチ読み取り + バッチ書き込み =====

// 良い例: 読み取りを先にまとめ、その後書き込みをまとめる
function resizeAllBoxesGood(boxes) {
  // Phase 1: 全ての読み取りをバッチ処理（レイアウト計算は1回だけ）
  const widths = boxes.map(box => box.offsetWidth);

  // Phase 2: 全ての書き込みをバッチ処理
  boxes.forEach((box, i) => {
    box.style.width = (widths[i] * 1.1) + 'px';
  });
  // レイアウト計算は最初の1回 + 書き込み後の1回 = 合計2回のみ
}

// さらに良い例: requestAnimationFrame を使用
function resizeAllBoxesBest(boxes) {
  // 読み取りは現在のフレームで実行
  const widths = boxes.map(box => box.offsetWidth);

  // 書き込みは次のフレームで実行
  requestAnimationFrame(() => {
    boxes.forEach((box, i) => {
      box.style.width = (widths[i] * 1.1) + 'px';
    });
  });
}

// 強制レイアウト（Forced Synchronous Layout）を引き起こすプロパティ:
// offsetTop, offsetLeft, offsetWidth, offsetHeight
// scrollTop, scrollLeft, scrollWidth, scrollHeight
// clientTop, clientLeft, clientWidth, clientHeight
// getComputedStyle()
// getBoundingClientRect()
// innerText
```

### 10.2 アンチパターン 2: 過剰なレイヤー昇格

```css
/* ===== アンチパターン: 全要素に will-change を設定 ===== */

/* 悪い例: 全要素をレイヤー昇格させる */
* {
  will-change: transform;
  /* 全要素が独立レイヤーになる
     → GPU メモリを大量消費
     → レイヤー管理のオーバーヘッド増大
     → 逆にパフォーマンス低下 */
}

/* 悪い例: 多数のリストアイテムに will-change */
.list-item {
  will-change: transform, opacity;
  /* 1000個のリストアイテムがあれば1000レイヤー
     → GPU メモリ枯渇 → ソフトウェアフォールバック */
}
```

```css
/* ===== 推奨パターン: 必要な要素にだけ、必要なタイミングで ===== */

/* 良い例: hover 時のみ will-change を有効化 */
.card {
  transition: transform 0.3s ease;
}
.card:hover {
  will-change: transform;
}
.card.animating {
  transform: scale(1.05);
}

/* 良い例: JavaScript で動的に管理 */
/*
  element.addEventListener('mouseenter', () => {
    element.style.willChange = 'transform';
  });
  element.addEventListener('transitionend', () => {
    element.style.willChange = 'auto';
  });
*/

/* 良い例: アニメーション対象の少数要素にだけ適用 */
.modal-overlay {
  will-change: opacity;
}
.slide-panel {
  will-change: transform;
}
/* その他の要素には will-change を設定しない */
```

### 10.3 アンチパターン 3: メインスレッドの過負荷

```javascript
// ===== アンチパターン: メインスレッドで重い計算 =====

// 悪い例: 大量データのソートをメインスレッドで実行
function sortLargeDatasetBad(data) {
  // 100万件のデータソート → メインスレッドが数秒間ブロック
  // → スクロール不可、クリック不応答、アニメーション停止
  return data.sort((a, b) => {
    // 複雑な比較ロジック
    return complexComparison(a, b);
  });
}

// ===== 推奨パターン: Web Worker にオフロード =====

// worker.js
// self.addEventListener('message', (e) => {
//   const { data, sortKey } = e.data;
//   const sorted = data.sort((a, b) => a[sortKey] - b[sortKey]);
//   self.postMessage({ sorted });
// });

// メインスレッド側
function sortLargeDatasetGood(data, sortKey) {
  return new Promise((resolve) => {
    const worker = new Worker('worker.js');
    worker.postMessage({ data, sortKey });
    worker.addEventListener('message', (e) => {
      resolve(e.data.sorted);
      worker.terminate();
    });
  });
}

// ===== 推奨パターン: タスク分割 (Time Slicing) =====

// チャンクに分割して処理し、メインスレッドに呼吸させる
async function processInChunks(items, processFn, chunkSize = 100) {
  const results = [];

  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize);
    const chunkResults = chunk.map(processFn);
    results.push(...chunkResults);

    // 各チャンク後にメインスレッドに制御を戻す
    // → レンダリングやイベント処理が割り込み可能
    if (i + chunkSize < items.length) {
      await new Promise(resolve => {
        // scheduler.yield() が利用可能なら使用
        if ('scheduler' in globalThis && 'yield' in scheduler) {
          scheduler.yield().then(resolve);
        } else {
          setTimeout(resolve, 0);
        }
      });
    }
  }

  return results;
}

// 使用例
// const processed = await processInChunks(largeArray, item => {
//   return expensiveTransform(item);
// }, 50);
```

---

## FAQ

### Q1: マルチプロセスアーキテクチャのメリットは何ですか？

**A:** マルチプロセスアーキテクチャには3つの主要なメリットがあります。

1. **安定性（Stability）**: 1つのタブやプラグインがクラッシュしても、他のタブやブラウザ本体には影響しません。レンダラープロセスが異常終了しても、ブラウザプロセスが生きているため「タブがクラッシュしました」と表示してリロードを促すことができます。

2. **セキュリティ（Security）**: 各レンダラープロセスはサンドボックス内で実行されるため、悪意のあるWebサイトがファイルシステムやネットワークに直接アクセスすることを防げます。特権操作はブラウザプロセスを経由する必要があり、権限チェックが行われます。

3. **パフォーマンス（Performance）**: マルチコアCPUを活用して複数のタブを並列処理できます。また、タブごとにプロセスを分離することで、メモリリークが発生してもタブを閉じればプロセスごとメモリが解放されます。

ただし、プロセス数が増えるとメモリオーバーヘッドも増大するため、Chromeは適宜プロセスを統合する最適化も行っています（プロセスモデルの項を参照）。

### Q2: ChromeとFirefoxのアーキテクチャの違いは何ですか？

**A:** 主な違いは以下の通りです。

**Chrome（Chromium）**:
- **タブごとにレンダラープロセス**を分離するマルチプロセスモデル（ただし、同一サイトは統合される場合もある）
- **Site Isolation**: セキュリティを強化するため、クロスサイトiframeも別プロセスで実行（Spectre攻撃対策）
- **GPU プロセス**: 全タブで共有する単一のGPUプロセス
- **Blink レンダリングエンジン** + **V8 JavaScript エンジン**

**Firefox**:
- **Quantum（Electrolysis/e10s）**: タブごとにコンテンツプロセスを分離（Chrome類似）
- **Fission**: Site Isolation相当の機能（iframe分離）を段階的に導入中
- **GPUプロセス**: Chromeと同様に単一のGPUプロセス
- **Gecko レンダリングエンジン** + **SpiderMonkey JavaScript エンジン**

アーキテクチャの基本思想は収束していますが、エンジンの実装や最適化戦略には違いがあります。例えば、Firefoxは「WebRender」という新しいGPU駆動のレンダリングエンジンを採用しており、Chromeとは異なるアプローチでパフォーマンスを追求しています。

### Q3: Site Isolationの仕組みを教えてください

**A:** Site Isolation は、クロスサイト攻撃（特にSpectre攻撃）からユーザーを守るためのセキュリティ機能です。

**基本原理**:
- 異なるオリジン（スキーム + ドメイン + ポート）のコンテンツは**別のレンダラープロセス**で実行される
- 例: `https://example.com` のメインフレームと `https://ad.example.net` のiframeは別プロセス

**なぜ必要か**:
- Spectre攻撃は、同一プロセス内のメモリを読み取る脆弱性です
- Site Isolationにより、悪意のあるiframeが親フレームのメモリ（パスワードやトークンなど）を読み取ることを防ぎます

**実装の詳細**:
1. **OOPIF（Out-of-Process iframes）**: クロスサイトiframeは別プロセスのレンダラーで描画され、メインフレームとはIPCで通信
2. **CORB（Cross-Origin Read Blocking）**: レンダラープロセスが不正なクロスオリジンリソース（HTML/JSON/XML）を読み込むのをブロック
3. **メモリオーバーヘッド**: プロセス数が増えるためメモリ使用量は10-20%増加しますが、セキュリティ上の利点が上回ると判断されています

**有効化状況**:
- Chrome 67以降、デスクトップ版ではデフォルトで有効
- Androidでは一部のハイエンドデバイスのみ有効（メモリ制約のため）

詳細は [Site Isolation Design Document](https://www.chromium.org/Home/chromium-security/site-isolation/) を参照してください。

---

## まとめ

| 項目 | 内容 |
|------|------|
| **マルチプロセスアーキテクチャ** | ブラウザプロセス、レンダラープロセス、GPUプロセス等に分離。安定性・セキュリティ・パフォーマンスを向上 |
| **主要プロセス** | ブラウザ（UI・ネットワーク・ストレージ管理）、レンダラー（HTML/CSS/JSレンダリング、サンドボックス化）、GPU（描画アクセラレーション） |
| **IPC（プロセス間通信）** | Mojoフレームワークでプロセス間メッセージング。型安全・非同期通信を実現 |
| **Site Isolation** | クロスサイトiframeを別プロセスで実行し、Spectre攻撃から保護。OOPIF・CORBと組み合わせてセキュリティ強化 |
| **レンダリングパイプライン** | HTML → DOM、CSS → CSSOM → Render Tree → Layout → Paint → Composite（GPU駆動）の7段階 |
| **最適化戦略** | Layerの最小化、will-change の適切な使用、Web Workerへのオフロード、Time Slicingによるメインスレッド負荷軽減 |

**キーポイント**:

1. **プロセス分離がセキュリティの鍵**: サンドボックスとSite Isolationにより、悪意のあるコンテンツがシステムや他のタブに影響を与えることを防ぐ
2. **レンダリングパイプラインの理解が最適化の第一歩**: Layout・Paint・Compositeの各段階を意識し、不要な再計算を避ける設計が重要
3. **モダンブラウザはGPU駆動**: Compositingレイヤーを活用し、transform/opacityのアニメーションをGPUで処理することで60fpsを実現

---

## 次に読むべきガイド

ブラウザアーキテクチャの全体像を理解したら、次は実際のWebページの読み込みプロセスを深掘りしましょう。

- **[ナビゲーションとローディング](./01-navigation-and-loading.md)**: URLを入力してからページが表示されるまでの詳細なフローを解説
  - DNS解決、TCP/TLS接続、HTTPリクエスト/レスポンス
  - ナビゲーションタイミングAPI
  - Critical Rendering Pathの最適化手法

その他の関連ガイド:
- **[レンダリングエンジン詳説](./02-rendering-engine.md)**: Blink/Geckoの内部実装とレンダリング最適化
- **[JavaScriptエンジン](./03-javascript-engine.md)**: V8/SpiderMonkeyの仕組みとパフォーマンスチューニング

---

## 参考文献

1. **[Inside look at modern web browser (Google Developers)](https://developers.google.com/web/updates/2018/09/inside-browser-part1)**
   Google Chrome チームによるブラウザアーキテクチャの公式解説。4部構成で、マルチプロセスモデルからレンダリングパイプラインまで詳細に説明。

2. **[The Chromium Projects - Multi-process Architecture](https://www.chromium.org/developers/design-documents/multi-process-architecture/)**
   Chromiumの設計ドキュメント。プロセスモデルの設計思想と実装の詳細を記載。

3. **[Life of a Pixel (Chromium)](https://docs.google.com/presentation/d/1boPxbgNrTU0ddsc144rcXayGA_WF53k96imRH8Mp34Y/edit)**
   Chromiumチームの内部プレゼンテーション。ピクセルがどのように画面に描画されるかを詳細に解説。

4. **[Site Isolation Design Document](https://www.chromium.org/Home/chromium-security/site-isolation/)**
   Site Isolationの設計文書。OOPIF、CORB、セキュリティモデルの詳細。

5. **[Mojo Documentation (Chromium)](https://chromium.googlesource.com/chromium/src/+/master/mojo/README.md)**
   MojoフレームワークのREADME。IPC（プロセス間通信）の実装とAPI。

6. **[MDN Web Docs - How browsers work](https://developer.mozilla.org/en-US/docs/Web/Performance/How_browsers_work)**
   Mozilla Developer Networkによるブラウザの仕組み解説。初学者にも分かりやすい。

7. **[Rendering Performance (Web Fundamentals)](https://developers.google.com/web/fundamentals/performance/rendering/)**
   Google Developersのパフォーマンスガイド。60fpsを達成するための実践的なテクニック。

