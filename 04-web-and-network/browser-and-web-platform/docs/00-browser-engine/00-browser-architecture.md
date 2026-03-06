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

