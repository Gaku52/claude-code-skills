# デザインパターンガイド

> デザインパターンは再利用可能な設計の知恵。GoF パターン、アーキテクチャパターン、関数型パターン、モダンな JavaScript/TypeScript での実装まで、設計パターンの全てを体系的に解説する。

## このSkillの対象者

- デザインパターンを体系的に学びたいエンジニア
- コードの品質と保守性を向上させたい方
- 設計判断の根拠を明確にしたい方

## 前提知識

- オブジェクト指向プログラミングの基礎
- TypeScript の基礎知識

## 学習ガイド

### 00-creational — 生成パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-creational/00-factory-patterns.md]] | Factory Method、Abstract Factory、Static Factory |
| 01 | [[docs/00-creational/01-singleton-and-builder.md]] | Singleton（モダン実装）、Builder パターン |
| 02 | [[docs/00-creational/02-prototype-and-pool.md]] | Prototype、Object Pool、Dependency Injection |

### 01-structural — 構造パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-structural/00-adapter-and-facade.md]] | Adapter、Facade、デコレータ |
| 01 | [[docs/01-structural/01-proxy-and-decorator.md]] | Proxy（ES Proxy）、Decorator（TC39）、Composite |
| 02 | [[docs/01-structural/02-bridge-and-flyweight.md]] | Bridge、Flyweight、Module パターン |

### 02-behavioral — 振る舞いパターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-behavioral/00-observer-and-mediator.md]] | Observer（EventEmitter）、Mediator、Pub/Sub |
| 01 | [[docs/02-behavioral/01-strategy-and-state.md]] | Strategy、State、Template Method |
| 02 | [[docs/02-behavioral/02-command-and-chain.md]] | Command、Chain of Responsibility、Visitor |
| 03 | [[docs/02-behavioral/03-iterator-and-memento.md]] | Iterator（Symbol.iterator）、Memento、Interpreter |

### 03-architectural — アーキテクチャパターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-architectural/00-mvc-mvvm-mvp.md]] | MVC、MVVM、MVP、Flux/Redux 比較 |
| 01 | [[docs/03-architectural/01-clean-architecture.md]] | Clean Architecture、ヘキサゴナル、オニオン |
| 02 | [[docs/03-architectural/02-microservices-patterns.md]] | Saga、CQRS、Event Sourcing、Circuit Breaker |
| 03 | [[docs/03-architectural/03-ddd-patterns.md]] | Entity、Value Object、Aggregate、Repository、Domain Event |

### 04-functional — 関数型パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-functional/00-fp-patterns.md]] | 純粋関数、不変性、カリー化、合成、モナド |
| 01 | [[docs/04-functional/01-reactive-patterns.md]] | RxJS、Observable、Operator パターン |
| 02 | [[docs/04-functional/02-modern-js-patterns.md]] | モダン JS/TS パターン、Effect-TS、fp-ts |

## クイックリファレンス

```
パターン選定ガイド:
  オブジェクト生成 → Factory / Builder / DI
  インターフェース適合 → Adapter / Facade
  状態管理 → Observer / State / Redux
  アルゴリズム切替 → Strategy
  非同期処理 → Promise / Observable
  エラー処理 → Result 型 / Chain of Responsibility
```

## 参考文献

1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
2. Freeman, E. "Head First Design Patterns." O'Reilly, 2020.
3. Addy Osmani. "Learning JavaScript Design Patterns." O'Reilly, 2023.
