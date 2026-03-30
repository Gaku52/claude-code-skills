[日本語版](../../ja/02-programming/typescript-complete-guide/SKILL.md)

# TypeScript Complete Guide

> TypeScript adds type safety to JavaScript, enabling large-scale development. This guide systematically covers everything in TypeScript -- from the depths of the type system, generics, and conditional types to Template Literal Types and type-level programming.

## Target Audience

- Engineers who want to systematically learn TypeScript from the ground up
- Developers seeking a deep understanding of the type system
- Those who want to write type definitions for libraries and frameworks

## Prerequisites

- Foundational knowledge of JavaScript (ES2022+)
- Basic familiarity with Node.js

## Study Guide

### 00-basics -- TypeScript Basics

| # | File | Content |
|---|------|---------|

### 01-type-system -- Deep Dive into the Type System

| # | File | Content |
|---|------|---------|

### 02-advanced-types -- Advanced Type Patterns

| # | File | Content |
|---|------|---------|

### 03-patterns -- Practical Patterns

| # | File | Content |
|---|------|---------|

### 04-tooling -- Toolchain

| # | File | Content |
|---|------|---------|

## Quick Reference

```
TypeScript Type Cheat Sheet:

  Utility Types:
    Partial<T>       -- Makes all properties optional
    Required<T>      -- Makes all properties required
    Readonly<T>      -- Makes all properties readonly
    Pick<T, K>       -- Extracts specified properties
    Omit<T, K>       -- Excludes specified properties
    Record<K, V>     -- Defines key and value types
    Extract<T, U>    -- Extracts types from T assignable to U
    Exclude<T, U>    -- Excludes types from T assignable to U
    ReturnType<F>    -- Return type of a function
    Parameters<F>    -- Parameter types of a function
    Awaited<T>       -- Resolved type of a Promise
    NonNullable<T>   -- Excludes null/undefined

  Recommended tsconfig settings:
    "strict": true
    "noUncheckedIndexedAccess": true
    "exactOptionalPropertyTypes": true
```

## References

1. TypeScript. "Handbook." typescriptlang.org/docs, 2024.
2. Vanderkam, D. "Effective TypeScript." O'Reilly, 2024.
3. TypeScript. "Release Notes." typescriptlang.org/docs/handbook/release-notes, 2024.
