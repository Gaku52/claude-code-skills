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

### 01-type-system — 型システム深掘り

| # | ファイル | 内容 |
|---|---------|------|

### 02-advanced-types — 高度な型パターン

| # | ファイル | 内容 |
|---|---------|------|

### 03-patterns — 実践パターン

| # | ファイル | 内容 |
|---|---------|------|

### 04-tooling — ツールチェーン

| # | ファイル | 内容 |
|---|---------|------|

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
