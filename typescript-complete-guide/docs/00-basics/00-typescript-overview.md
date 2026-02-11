# TypeScript概要

> JavaScriptの完全な上位互換（スーパーセット）として設計された静的型付き言語。型システムによりコードの安全性・保守性・開発体験を劇的に向上させる。

## この章で学ぶこと

1. **TypeScriptとは何か** -- JavaScriptとの関係、スーパーセットの意味、コンパイルの仕組み
2. **型システムがもたらす価値** -- バグの早期発見、リファクタリング安全性、IDE支援
3. **歴史とエコシステム** -- 誕生の背景、バージョンの変遷、主要ツールチェーン

---

## 1. TypeScriptとは何か

### JavaScriptのスーパーセット

TypeScriptはMicrosoftが2012年に公開したオープンソース言語である。全ての正しいJavaScriptコードはそのままTypeScriptとしても有効である。TypeScriptはそこに**静的型付け**を追加する。

```
+------------------------------------------+
|            TypeScript                     |
|  +------------------------------------+  |
|  |          JavaScript                |  |
|  |  +------------------------------+  |  |
|  |  |       ECMAScript仕様         |  |  |
|  |  +------------------------------+  |  |
|  +------------------------------------+  |
|  + 型アノテーション                     |  |
|  + インターフェース                     |  |
|  + ジェネリクス                         |  |
|  + 列挙型                               |  |
|  + その他の型機能                       |  |
+------------------------------------------+
```

### コード例1: JavaScriptがそのままTypeScript

```typescript
// これは有効なJavaScriptであり、同時に有効なTypeScriptでもある
const greet = (name) => `Hello, ${name}!`;
console.log(greet("World"));
```

### コード例2: 型アノテーションの追加

```typescript
// 型アノテーションを追加すると、TypeScriptの力を活用できる
const greet = (name: string): string => `Hello, ${name}!`;

// コンパイルエラー: Argument of type 'number' is not assignable to parameter of type 'string'
// greet(42);

console.log(greet("World")); // OK
```

### コンパイルフロー

```
  TypeScript ソースコード (.ts / .tsx)
         |
         v
  +-------------------+
  | TypeScript        |
  | コンパイラ (tsc)   |
  +-------------------+
         |
    +----+----+
    |         |
    v         v
 JavaScript  型エラー
 (.js)       レポート
```

### コード例3: コンパイル実行

```bash
# TypeScriptコンパイラのインストール
npm install -g typescript

# コンパイル
tsc hello.ts        # -> hello.js が生成される

# コンパイル（型チェックのみ、出力なし）
tsc --noEmit hello.ts
```

---

## 2. 型システムがもたらす価値

### コード例4: 型がバグを防ぐ

```typescript
// JavaScript（実行時まで気づかない）
function calculateArea(width, height) {
  return width * height;
}
calculateArea("10", 20); // "1020" -- 意図しない文字列連結

// TypeScript（コンパイル時に検出）
function calculateArea(width: number, height: number): number {
  return width * height;
}
// calculateArea("10", 20); // コンパイルエラー！
calculateArea(10, 20); // 200 -- 正しい結果
```

### コード例5: IDEの自動補完

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  createdAt: Date;
}

function displayUser(user: User) {
  // user. と入力した時点で id, name, email, createdAt が候補に表示される
  console.log(`${user.name} <${user.email}>`);
}
```

### 型システムの利点比較

| 観点 | JavaScript (型なし) | TypeScript (型あり) |
|------|---------------------|---------------------|
| バグ検出タイミング | 実行時（本番含む） | コンパイル時（開発中） |
| リファクタリング | 手動で全箇所確認 | コンパイラが影響範囲を自動検出 |
| IDE補完 | 推測ベース（不正確） | 型情報ベース（正確） |
| ドキュメント | コメントで記述（陳腐化しやすい） | 型が生きたドキュメントになる |
| チーム開発 | 口頭・ドキュメント依存 | 型がコントラクトとして機能 |
| 学習コスト | 低い | やや高い（投資価値あり） |

### コード例6: リファクタリング安全性

```typescript
interface Product {
  id: number;
  name: string;
  price: number;
}

// price を priceInCents にリネームした場合、
// TypeScriptは全ての参照箇所でエラーを出してくれる
interface Product {
  id: number;
  name: string;
  priceInCents: number; // リネーム
}

// コンパイラが全ての `product.price` を検出してエラーにする
// 修正漏れが起きない
```

---

## 3. 歴史とエコシステム

### TypeScriptの歴史年表

```
2012  v0.8   初回リリース（Microsoft）
  |
2014  v1.0   安定版リリース
  |
2016  v2.0   Non-nullable types, Tagged Unions
  |
2018  v3.0   Project References, unknown型
  |
2019  v3.7   Optional Chaining, Nullish Coalescing
  |
2020  v4.0   Variadic Tuple Types, Labeled Tuples
  |
2021  v4.5   Awaited型, ESM対応強化
  |
2023  v5.0   Decorators (Stage 3), const型パラメータ
  |
2024  v5.4   NoInfer, Object.groupBy型
  |
2025  v5.7   --erasableSyntaxOnly, 最新機能
```

### エコシステム全体像

| カテゴリ | 主要ツール | 役割 |
|----------|-----------|------|
| コンパイラ | tsc | 型チェック + トランスパイル |
| バンドラ | esbuild, SWC, Vite | 高速ビルド |
| リンター | typescript-eslint | コード品質チェック |
| テスト | Vitest, Jest | 型安全なテスト |
| スキーマ | Zod, io-ts | ランタイムバリデーション |
| ORM | Prisma, Drizzle | 型安全なDB操作 |
| API | tRPC, GraphQL Code Generator | 型安全なAPI通信 |
| フレームワーク | Next.js, Remix, Astro | フルスタック開発 |

### コード例7: 最小限のTypeScriptプロジェクト構成

```bash
# プロジェクト初期化
mkdir my-ts-project && cd my-ts-project
npm init -y
npm install typescript --save-dev
npx tsc --init

# ディレクトリ構成
# my-ts-project/
# ├── src/
# │   └── index.ts
# ├── dist/           # コンパイル出力
# ├── tsconfig.json
# └── package.json
```

---

## アンチパターン

### アンチパターン1: any の濫用

```typescript
// BAD: anyを使うと型システムの恩恵がゼロになる
function processData(data: any): any {
  return data.map((item: any) => item.value);
}

// GOOD: 適切な型を定義する
interface DataItem {
  value: string;
}
function processData(data: DataItem[]): string[] {
  return data.map((item) => item.value);
}
```

### アンチパターン2: TypeScriptを「ただのJavaScript + 拡張子変更」として使う

```typescript
// BAD: .ts にしただけで型を一切書かない
// tsconfig.json で strict: false にする
// → JavaScriptと変わらず、移行コストだけ発生

// GOOD: strict: true を有効にし、段階的に型をつける
// tsconfig.json
{
  "compilerOptions": {
    "strict": true  // 全てのstrictチェックを有効化
  }
}
```

---

## FAQ

### Q1: TypeScriptは実行時に型チェックを行いますか？

**A:** いいえ。TypeScriptの型情報はコンパイル時に全て消去（erasure）されます。実行時はただのJavaScriptです。実行時バリデーションが必要な場合は Zod や io-ts などのライブラリを併用します。

### Q2: TypeScriptを学ぶにはJavaScriptを先に覚えるべきですか？

**A:** はい、推奨します。TypeScriptはJavaScriptの上に構築されているため、JavaScriptの基礎（関数、オブジェクト、プロトタイプ、非同期処理）を理解していると学習がスムーズです。ただし、最初からTypeScriptで学ぶアプローチも増えています。

### Q3: TypeScriptのデメリットは何ですか？

**A:** 主なデメリットは以下の通りです:
- **学習コスト**: 型システムの概念を学ぶ必要がある
- **ビルドステップ**: コンパイルが必要（ただしesbuild等で高速化可能）
- **型定義の保守**: 複雑な型は保守コストが発生する
- **サードパーティ型**: 一部のライブラリは型定義が不完全
とはいえ、中〜大規模プロジェクトではこれらのコストを大きく上回るメリットがあります。

---

## まとめ

| 項目 | 内容 |
|------|------|
| TypeScriptとは | JavaScriptのスーパーセットで、静的型付けを追加する言語 |
| 開発元 | Microsoft（2012年公開、オープンソース） |
| コンパイル | .ts → .js に変換。型情報は実行時に消去される |
| 主な利点 | バグ早期発見、IDE支援、リファクタリング安全性 |
| 主なコスト | 学習曲線、ビルドステップ、型定義保守 |
| エコシステム | tsc, esbuild, Vitest, Zod, Prisma, tRPC など充実 |
| strict モード | 推奨。型チェックの恩恵を最大化する |

---

## 次に読むべきガイド

- [01-type-basics.md](./01-type-basics.md) -- 型の基礎（プリミティブ型、リテラル型、配列、タプル）
- [02-functions-and-objects.md](./02-functions-and-objects.md) -- 関数とオブジェクト型

---

## 参考文献

1. **TypeScript公式ドキュメント** -- https://www.typescriptlang.org/docs/
2. **TypeScript Deep Dive (日本語版)** -- https://typescript-jp.gitbook.io/deep-dive/
3. **Programming TypeScript (Boris Cherny著, O'Reilly)** -- 型システムの理論と実践を網羅した書籍
4. **TypeScript GitHub リポジトリ** -- https://github.com/microsoft/TypeScript
