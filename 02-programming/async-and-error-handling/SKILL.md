[日本語版](../../ja/02-programming/async-and-error-handling/SKILL.md)

# Async Programming and Error Handling Complete Guide

> Async programming is the foundation of modern applications. Error handling is the cornerstone of software reliability. This guide systematically covers these two closely related topics, including implementations and best practices across multiple languages.

## Target Audience

- Engineers who want to deeply understand async programming mechanisms and patterns
- Developers seeking best practices for error handling
- Anyone looking to organize their understanding of Promise, async/await, and Result types

## Prerequisites

- Programming basics (functions, control flow)

## Guide Index

### 00-introduction (Introduction)
| File | Topic | Summary |
|------|-------|---------|
| [00-sync-vs-async.md](docs/00-introduction/00-sync-vs-async.md) | Sync vs Async | Blocking/non-blocking, why async is necessary |
| [01-concurrency-models.md](docs/00-introduction/01-concurrency-models.md) | Concurrency Models Overview | Multithreading, event loops, the actor model |

### 01-async-patterns (Async Patterns)
| File | Topic | Summary |
|------|-------|---------|
| [00-callbacks.md](docs/01-async-patterns/00-callbacks.md) | Callbacks | Callback hell, Node.js error-first pattern |
| [01-promises.md](docs/01-async-patterns/01-promises.md) | Promise | Promise chaining, Promise.all/race/allSettled |
| [02-async-await.md](docs/01-async-patterns/02-async-await.md) | async/await | Implementations across languages, concurrent execution patterns |
| [03-reactive-streams.md](docs/01-async-patterns/03-reactive-streams.md) | Reactive Streams | RxJS, Observable, backpressure |

### 02-error-handling (Error Handling)
| File | Topic | Summary |
|------|-------|---------|
| [00-exceptions.md](docs/02-error-handling/00-exceptions.md) | Exception Handling | try/catch/finally, exception hierarchies, checked/unchecked |
| [01-result-type.md](docs/02-error-handling/01-result-type.md) | Result Types | Rust Result, TypeScript never-throw, Go error |
| [02-error-boundaries.md](docs/02-error-handling/02-error-boundaries.md) | Error Boundaries | React Error Boundary, global handlers |
| [03-custom-errors.md](docs/02-error-handling/03-custom-errors.md) | Custom Errors | Error design, error codes, domain errors |

### 03-advanced (Advanced Topics)
| File | Topic | Summary |
|------|-------|---------|
| [00-event-loop.md](docs/03-advanced/00-event-loop.md) | Event Loop | Deep dive into the Node.js/browser event loop |
| [01-cancellation.md](docs/03-advanced/01-cancellation.md) | Cancellation | AbortController, CancellationToken, timeouts |
| [02-retry-and-backoff.md](docs/03-advanced/02-retry-and-backoff.md) | Retry Strategies | Exponential backoff, circuit breakers |
| [03-structured-concurrency.md](docs/03-advanced/03-structured-concurrency.md) | Structured Concurrency | Kotlin coroutines, Swift structured concurrency |

### 04-practical (Practical)
| File | Topic | Summary |
|------|-------|---------|
| [00-api-error-design.md](docs/04-practical/00-api-error-design.md) | API Error Design | HTTP status codes, error response design, RFC 7807 |
| [01-logging-and-monitoring.md](docs/04-practical/01-logging-and-monitoring.md) | Logging and Monitoring | Structured logging, error tracking, Sentry |
| [02-testing-async.md](docs/04-practical/02-testing-async.md) | Testing Async Code | Testing techniques for async code, mocks, timers |
| [03-real-world-patterns.md](docs/04-practical/03-real-world-patterns.md) | Real-World Patterns | Queue processing, WebSocket, file uploads |

## Learning Path

```
Fundamentals:  00-introduction -> 01-async-patterns (00 -> 02)
Error:         02-error-handling (00 -> 03)
Applied:       01-async-patterns/03 -> 03-advanced -> 04-practical
```

## Related Skills

