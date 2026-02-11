# Node.js CLI Template (Commander)

完全な機能を持つ Node.js CLI テンプレート（Commander + Inquirer + chalk）

## 特徴

- ✅ TypeScript サポート
- ✅ Commander による引数パース
- ✅ Inquirer によるインタラクティブプロンプト
- ✅ chalk によるカラフルな出力
- ✅ ora によるスピナー
- ✅ Jest によるテスト
- ✅ ESLint + Prettier

## セットアップ

```bash
# 依存関係インストール
npm install
# or
pnpm install
# or
yarn install

# ビルド
npm run build

# 開発モード
npm run dev

# テスト
npm test

# CLI 実行
npm start -- --help
# or
node dist/index.js --help
```

## 使用例

```bash
# プロジェクト作成
mycli create myapp

# テンプレート指定
mycli create myapp --template react

# プロジェクト一覧
mycli list

# プロジェクト削除
mycli delete myapp --force
```

## ディレクトリ構造

```
.
├── src/
│   ├── index.ts             # エントリーポイント
│   ├── commands/            # コマンド定義
│   │   ├── create.ts
│   │   ├── list.ts
│   │   └── delete.ts
│   ├── core/                # ビジネスロジック
│   │   └── ProjectService.ts
│   ├── utils/               # ユーティリティ
│   │   ├── logger.ts
│   │   └── config.ts
│   └── types/               # 型定義
│       └── index.ts
├── tests/                   # テスト
├── dist/                    # ビルド出力
├── package.json
├── tsconfig.json
└── README.md
```

## 開発

```bash
# 開発モード（ウォッチ）
npm run dev

# リント
npm run lint

# フォーマット
npm run format

# テスト（ウォッチ）
npm run test:watch

# カバレッジ
npm run test:coverage
```

## ビルド & 公開

```bash
# ビルド
npm run build

# npm へ公開
npm publish

# ローカルインストール
npm link
mycli --help
```
