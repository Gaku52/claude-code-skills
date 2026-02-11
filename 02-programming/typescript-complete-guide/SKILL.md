# TypeScript 完全ガイド

> TypeScript は JavaScript に型安全性を加え、大規模開発を可能にする。型システムの深層、ジェネリクス、条件付き型、Template Literal Types、型レベルプログラミングまで、TypeScript の全てを体系的に解説する。

## このSkillの対象者

- TypeScript を基礎から体系的に学びたいエンジニア
- 型システムを深く理解し活用したい方
- ライブラリ・フレームワークの型定義を書きたい方

## 前提知識

- JavaScript（ES2022+）の基礎知識
- Node.js の基本操作

## 学習ガイド

### 00-basics — TypeScript の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-basics/00-typescript-overview.md]] | TypeScript の歴史、セットアップ、tsconfig.json、strict モード |
| 01 | [[docs/00-basics/01-basic-types.md]] | プリミティブ型、配列、タプル、enum、any/unknown/never/void |
| 02 | [[docs/00-basics/02-functions-and-objects.md]] | 関数型、オーバーロード、interface vs type、readonly、optional |
| 03 | [[docs/00-basics/03-classes-and-modules.md]] | クラス、アクセス修飾子、abstract、decorators、モジュール |
| 04 | [[docs/00-basics/04-type-narrowing.md]] | 型ガード、typeof/instanceof/in、Discriminated Unions、assertion functions |

### 01-type-system — 型システム深掘り

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-type-system/00-generics.md]] | ジェネリクス、制約、デフォルト型、推論、共変・反変 |
| 01 | [[docs/01-type-system/01-conditional-types.md]] | 条件付き型、infer、Distributive、Extract/Exclude/ReturnType |
| 02 | [[docs/01-type-system/02-mapped-types.md]] | Mapped Types、キーリマッピング、Partial/Required/Pick/Omit/Record |
| 03 | [[docs/01-type-system/03-template-literal-types.md]] | Template Literal Types、パターンマッチング、型レベル文字列操作 |
| 04 | [[docs/01-type-system/04-type-level-programming.md]] | 型レベルプログラミング、再帰型、型パズル、ts-toolbelt |

### 02-advanced-types — 高度な型パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-advanced-types/00-branded-types.md]] | Branded Types、Opaque Types、型安全な ID・通貨・単位 |
| 01 | [[docs/02-advanced-types/01-builder-pattern-types.md]] | Builder パターン、メソッドチェーン型、Fluent API の型付け |
| 02 | [[docs/02-advanced-types/02-variance-and-soundness.md]] | 共変・反変・不変、型健全性、構造的部分型、strict オプション |
| 03 | [[docs/02-advanced-types/03-declaration-files.md]] | .d.ts、DefinitelyTyped、declare、module augmentation、global |

### 03-patterns — 実践パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-patterns/00-react-typescript.md]] | React + TypeScript、Props 型、Hooks 型、イベント型 |
| 01 | [[docs/03-patterns/01-api-type-safety.md]] | tRPC、Zod + 推論、OpenAPI 型生成、型安全な API |
| 02 | [[docs/03-patterns/02-error-handling-types.md]] | Result 型、Either、neverthrow、型安全なエラーハンドリング |
| 03 | [[docs/03-patterns/03-state-machine-types.md]] | 状態マシン型、XState + TypeScript、有限オートマトン |

### 04-tooling — ツールチェーン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-tooling/00-tsconfig-deep-dive.md]] | tsconfig.json 全オプション、Project References、paths |
| 01 | [[docs/04-tooling/01-build-tools.md]] | tsc、esbuild、swc、tsup、tsx、ts-node、バンドラー統合 |
| 02 | [[docs/04-tooling/02-testing-typescript.md]] | Vitest + TypeScript、型テスト（expectTypeOf）、tsd |

## クイックリファレンス

```
TypeScript 型チートシート:

  ユーティリティ型:
    Partial<T>       — 全プロパティをオプショナルに
    Required<T>      — 全プロパティを必須に
    Readonly<T>      — 全プロパティを readonly に
    Pick<T, K>       — 指定プロパティのみ抽出
    Omit<T, K>       — 指定プロパティを除外
    Record<K, V>     — キーと値の型を指定
    Extract<T, U>    — T から U に代入可能な型を抽出
    Exclude<T, U>    — T から U に代入可能な型を除外
    ReturnType<F>    — 関数の戻り値型
    Parameters<F>    — 関数の引数型
    Awaited<T>       — Promise の解決型
    NonNullable<T>   — null/undefined を除外

  tsconfig 推奨設定:
    "strict": true
    "noUncheckedIndexedAccess": true
    "exactOptionalPropertyTypes": true
```

## 参考文献

1. TypeScript. "Handbook." typescriptlang.org/docs, 2024.
2. Vanderkam, D. "Effective TypeScript." O'Reilly, 2024.
3. TypeScript. "Release Notes." typescriptlang.org/docs/handbook/release-notes, 2024.
