# ブラウザとWebプラットフォーム 完全ガイド

> ブラウザの内部動作を深く理解する。レンダリングエンジン、JavaScript実行環境、Web API、ストレージ、セキュリティモデルまで、Web開発者に必須のブラウザ知識を網羅。

## このSkillの対象者

- ブラウザの仕組みを基礎から学びたいWeb開発者
- パフォーマンス最適化に取り組むフロントエンドエンジニア
- Web APIを深く理解したい開発者

## 前提知識

- HTML/CSS/JavaScriptの基本
- HTTPの基本 → 参照: [[network-fundamentals]]

## ガイド一覧

### 00-browser-engine（ブラウザエンジン）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-browser-architecture.md](docs/00-browser-engine/00-browser-architecture.md) | ブラウザアーキテクチャ | マルチプロセス構造、主要コンポーネント |
| [01-navigation-and-loading.md](docs/00-browser-engine/01-navigation-and-loading.md) | ナビゲーションとローディング | URL入力からページ表示までの流れ |
| [02-parsing-html-css.md](docs/00-browser-engine/02-parsing-html-css.md) | HTML/CSSパーシング | DOM/CSSOM構築、パーサーの動作 |
| [03-browser-security-model.md](docs/00-browser-engine/03-browser-security-model.md) | ブラウザセキュリティモデル | サンドボックス、CSP、サイト分離 |

### 01-rendering（レンダリング）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-rendering-pipeline.md](docs/01-rendering/00-rendering-pipeline.md) | レンダリングパイプライン | Layout, Paint, Composite の流れ |
| [01-css-layout-engine.md](docs/01-rendering/01-css-layout-engine.md) | CSSレイアウトエンジン | Box Model, Flexbox, Grid の内部動作 |
| [02-paint-and-compositing.md](docs/01-rendering/02-paint-and-compositing.md) | Paint と Compositing | レイヤー、GPU合成、will-change |
| [03-animation-performance.md](docs/01-rendering/03-animation-performance.md) | アニメーションパフォーマンス | 60fps、requestAnimationFrame、CSS vs JS |

### 02-javascript-runtime（JavaScript実行環境）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-v8-engine.md](docs/02-javascript-runtime/00-v8-engine.md) | V8エンジン | JIT、Hidden Class、ガベージコレクション |
| [01-event-loop-browser.md](docs/02-javascript-runtime/01-event-loop-browser.md) | ブラウザのイベントループ | タスクキュー、マイクロタスク、rAF |
| [02-web-workers.md](docs/02-javascript-runtime/02-web-workers.md) | Web Workers | Worker, SharedWorker, ServiceWorker |
| [03-memory-management.md](docs/02-javascript-runtime/03-memory-management.md) | メモリ管理 | メモリリーク検出、プロファイリング |

### 03-web-apis（Web API）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-dom-api.md](docs/03-web-apis/00-dom-api.md) | DOM API | DOM操作、MutationObserver、Shadow DOM |
| [01-fetch-and-streams.md](docs/03-web-apis/01-fetch-and-streams.md) | Fetch と Streams | Fetch API, ReadableStream, AbortController |
| [02-intersection-resize-observer.md](docs/03-web-apis/02-intersection-resize-observer.md) | Observer API | IntersectionObserver, ResizeObserver |

### 04-storage-and-caching（ストレージとキャッシュ）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-web-storage.md](docs/04-storage-and-caching/00-web-storage.md) | Webストレージ | localStorage, sessionStorage, IndexedDB, Cookie |
| [01-service-worker-cache.md](docs/04-storage-and-caching/01-service-worker-cache.md) | Service Worker | キャッシュ戦略、オフライン対応、PWA |
| [02-performance-api.md](docs/04-storage-and-caching/02-performance-api.md) | Performance API | Navigation Timing, Resource Timing, PerformanceObserver |

## 学習パス

```
基礎:     00-browser-engine → 01-rendering
実行環境:  02-javascript-runtime
API活用:   03-web-apis → 04-storage-and-caching
```

## 関連Skills

- [[network-fundamentals]] — ネットワーク基礎
- [[frontend-performance]] — フロントエンドパフォーマンス
- [[web-development]] — Web開発
- [[react-development]] — React開発
