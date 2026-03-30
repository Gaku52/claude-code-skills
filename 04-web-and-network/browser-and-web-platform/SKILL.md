[日本語版](../../ja/04-web-and-network/browser-and-web-platform/SKILL.md)

# Browser and Web Platform — Complete Guide

> A deep understanding of browser internals. Covering the rendering engine, JavaScript runtime, Web APIs, storage, and the security model — all the browser knowledge essential for web developers.

## Target Audience

- Web developers who want to learn how browsers work from the ground up
- Frontend engineers working on performance optimization
- Developers seeking a deep understanding of Web APIs

## Prerequisites

- HTML/CSS/JavaScript basics
- HTTP basics — Reference: [Network Fundamentals](../network-fundamentals/)

## Guide Index

### 00-browser-engine (Browser Engine)
| File | Topic | Overview |
|------|-------|----------|
| [00-browser-architecture.md](docs/00-browser-engine/00-browser-architecture.md) | Browser Architecture | Multi-process structure, key components |
| [01-navigation-and-loading.md](docs/00-browser-engine/01-navigation-and-loading.md) | Navigation and Loading | The journey from URL input to page display |
| [02-parsing-html-css.md](docs/00-browser-engine/02-parsing-html-css.md) | HTML/CSS Parsing | DOM/CSSOM construction, parser behavior |
| [03-browser-security-model.md](docs/00-browser-engine/03-browser-security-model.md) | Browser Security Model | Sandboxing, CSP, site isolation |

### 01-rendering (Rendering)
| File | Topic | Overview |
|------|-------|----------|
| [00-rendering-pipeline.md](docs/01-rendering/00-rendering-pipeline.md) | Rendering Pipeline | Layout, Paint, Composite flow |
| [01-css-layout-engine.md](docs/01-rendering/01-css-layout-engine.md) | CSS Layout Engine | Box Model, Flexbox, Grid internals |
| [02-paint-and-compositing.md](docs/01-rendering/02-paint-and-compositing.md) | Paint and Compositing | Layers, GPU compositing, will-change |
| [03-animation-performance.md](docs/01-rendering/03-animation-performance.md) | Animation Performance | 60fps, requestAnimationFrame, CSS vs JS |

### 02-javascript-runtime (JavaScript Runtime)
| File | Topic | Overview |
|------|-------|----------|
| [00-v8-engine.md](docs/02-javascript-runtime/00-v8-engine.md) | V8 Engine | JIT, Hidden Classes, garbage collection |
| [01-event-loop-browser.md](docs/02-javascript-runtime/01-event-loop-browser.md) | Browser Event Loop | Task queue, microtasks, rAF |
| [02-web-workers.md](docs/02-javascript-runtime/02-web-workers.md) | Web Workers | Worker, SharedWorker, ServiceWorker |
| [03-memory-management.md](docs/02-javascript-runtime/03-memory-management.md) | Memory Management | Memory leak detection, profiling |

### 03-web-apis (Web APIs)
| File | Topic | Overview |
|------|-------|----------|
| [00-dom-api.md](docs/03-web-apis/00-dom-api.md) | DOM API | DOM manipulation, MutationObserver, Shadow DOM |
| [01-fetch-and-streams.md](docs/03-web-apis/01-fetch-and-streams.md) | Fetch and Streams | Fetch API, ReadableStream, AbortController |
| [02-intersection-resize-observer.md](docs/03-web-apis/02-intersection-resize-observer.md) | Observer APIs | IntersectionObserver, ResizeObserver |

### 04-storage-and-caching (Storage and Caching)
| File | Topic | Overview |
|------|-------|----------|
| [00-web-storage.md](docs/04-storage-and-caching/00-web-storage.md) | Web Storage | localStorage, sessionStorage, IndexedDB, Cookie |
| [01-service-worker-cache.md](docs/04-storage-and-caching/01-service-worker-cache.md) | Service Worker | Caching strategies, offline support, PWA |
| [02-performance-api.md](docs/04-storage-and-caching/02-performance-api.md) | Performance API | Navigation Timing, Resource Timing, PerformanceObserver |

## Learning Path

```
Basics:      00-browser-engine → 01-rendering
Runtime:     02-javascript-runtime
API usage:   03-web-apis → 04-storage-and-caching
```

## Related Skills

- [Network Fundamentals](../network-fundamentals/) — Network Fundamentals
- [Web Application Development](../web-application-development/) — Web Application Development
- [API and Library Guide](../api-and-library-guide/) — API and Library Design

---

## FAQ

### Q1: Can I study this Skill without knowledge of frontend frameworks (React, Vue, etc.)?
Yes. This Skill focuses on the browser's native mechanisms. Framework knowledge is not a prerequisite. In fact, understanding the rendering pipeline, event loop, and DOM API covered here is extremely valuable for grasping how frameworks work under the hood. Building a solid understanding of browser fundamentals before using frameworks will dramatically improve your ability to debug performance issues and make sound architectural decisions.

### Q2: Do I need to follow the learning order strictly?
Following the recommended path (00-browser-engine, 01-rendering, 02-javascript-runtime, 03-web-apis, 04-storage-and-caching) is the most efficient approach, but you can start with any section that interests you if you already have knowledge in certain areas. That said, understanding the rendering pipeline is a prerequisite for nearly every other section, so reading 01-rendering early on is strongly recommended.

### Q3: How does the knowledge from this Skill apply in real-world work?
It directly applies to a wide range of frontend development scenarios: performance optimization (improving Core Web Vitals, identifying rendering bottlenecks), investigating and fixing memory leaks, implementing security measures (CSP configuration, XSS prevention), and building offline support (Service Worker, Cache API). Profiling and debugging skills using DevTools, in particular, will significantly boost your day-to-day development productivity.

---

## Summary

This Skill covers the following topics:

- The multi-process architecture of browsers and the defense-in-depth approach of the security model
- The six stages of the rendering pipeline (from DOM construction to compositing) and performance optimization techniques
- The JavaScript event loop, V8 engine internals, memory management, and GC algorithms
- Efficient use of Web APIs including the DOM API, Fetch API, and Observer APIs
- Storage and caching strategies using Cookie, localStorage, IndexedDB, Cache API, and Service Worker

---

## References

- [Chrome Developers](https://developer.chrome.com/) - Official documentation for web developers by the Google Chrome team
- [MDN Web Docs](https://developer.mozilla.org/) - Mozilla's comprehensive reference for web technologies
- [web.dev](https://web.dev/) - Google's guide to web performance and best practices
- [WHATWG HTML Living Standard](https://html.spec.whatwg.org/) - The official HTML specification (including the event loop, parser, etc.)
- [W3C CSS Specifications](https://www.w3.org/Style/CSS/) - The official collection of CSS-related specifications
