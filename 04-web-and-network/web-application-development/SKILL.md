# Webアプリケーション開発 完全ガイド

> Webアプリケーションの設計から本番デプロイまで。アーキテクチャ選定、状態管理、ルーティング、フォーム設計、認証統合、デプロイ戦略まで、モダンなWebアプリ開発の全体像を体系的に学ぶ。

## このSkillの対象者

- Webアプリケーションの設計・開発に携わるフルスタックエンジニア
- フロントエンドフレームワークの選定を検討している開発者
- 本番運用を見据えたWebアプリの構築を目指す開発者

## 前提知識

- HTML/CSS/JavaScript の基本
- Reactの基本 → 参照: [プログラミング言語基礎](../../02-programming/programming-language-fundamentals/)

## ガイド一覧

### 00-architecture（アーキテクチャ）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-spa-mpa-ssr.md](docs/00-architecture/00-spa-mpa-ssr.md) | SPA/MPA/SSR | レンダリング方式の比較と選定基準 |
| [01-project-structure.md](docs/00-architecture/01-project-structure.md) | プロジェクト構成 | ディレクトリ設計、モジュール分割 |
| [02-component-architecture.md](docs/00-architecture/02-component-architecture.md) | コンポーネント設計 | Atomic Design、Container/Presentational |
| [03-data-fetching-patterns.md](docs/00-architecture/03-data-fetching-patterns.md) | データフェッチング | SWR、TanStack Query、Server Components |

### 01-state-management（状態管理）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-state-management-overview.md](docs/01-state-management/00-state-management-overview.md) | 状態管理概論 | ローカル/グローバル/サーバー状態の分類 |
| [01-zustand-and-jotai.md](docs/01-state-management/01-zustand-and-jotai.md) | Zustand / Jotai | 軽量状態管理ライブラリの使い分け |
| [02-server-state.md](docs/01-state-management/02-server-state.md) | サーバー状態 | TanStack Query、SWR のキャッシュ戦略 |
| [03-url-state.md](docs/01-state-management/03-url-state.md) | URL状態 | 検索パラメータ、ディープリンク |

### 02-routing-and-navigation（ルーティング）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-client-side-routing.md](docs/02-routing-and-navigation/00-client-side-routing.md) | クライアントルーティング | React Router、TanStack Router |
| [01-file-based-routing.md](docs/02-routing-and-navigation/01-file-based-routing.md) | ファイルベースルーティング | Next.js App Router、Remix |
| [02-navigation-patterns.md](docs/02-routing-and-navigation/02-navigation-patterns.md) | ナビゲーション設計 | ブレッドクラム、タブ、サイドバー |
| [03-auth-and-guards.md](docs/02-routing-and-navigation/03-auth-and-guards.md) | 認証ガード | ルート保護、リダイレクト |

### 03-forms-and-validation（フォーム）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-form-design.md](docs/03-forms-and-validation/00-form-design.md) | フォーム設計 | React Hook Form、制御/非制御コンポーネント |
| [01-validation-patterns.md](docs/03-forms-and-validation/01-validation-patterns.md) | バリデーション | Zod統合、リアルタイム検証 |
| [02-file-upload.md](docs/03-forms-and-validation/02-file-upload.md) | ファイルアップロード | ドラッグ&ドロップ、プログレス、S3直接アップロード |
| [03-complex-forms.md](docs/03-forms-and-validation/03-complex-forms.md) | 複雑なフォーム | マルチステップ、動的フォーム、条件分岐 |

### 04-deployment（デプロイ）
| ファイル | テーマ | 概要 |
|---------|--------|------|
| [00-deployment-platforms.md](docs/04-deployment/00-deployment-platforms.md) | デプロイ先 | Vercel、Cloudflare、AWS、Docker |
| [01-environment-and-config.md](docs/04-deployment/01-environment-and-config.md) | 環境設定 | 環境変数、Feature Flags |
| [02-performance-optimization.md](docs/04-deployment/02-performance-optimization.md) | パフォーマンス | バンドル最適化、画像最適化、CDN |
| [03-monitoring-and-error-tracking.md](docs/04-deployment/03-monitoring-and-error-tracking.md) | 監視 | Sentry、Web Vitals、ログ |

## 学習パス

```
アーキテクチャ:  00-architecture
状態管理:        01-state-management
ルーティング:    02-routing-and-navigation
フォーム:        03-forms-and-validation
デプロイ:        04-deployment
```

## FAQ

### Q1: フレームワークはNext.js一択か？他の選択肢はあるか？
Next.jsは最も人気のあるReactフレームワークだが、唯一の選択肢ではない。Remixはweb標準に近いアプローチを採り、progressive enhancementを重視するプロジェクトに向いている。Astroはコンテンツ中心のサイトに最適で、Islands Architectureにより最小限のJavaScriptで高速なページを構築できる。Vue.jsエコシステムではNuxt.js、SvelteではSvelteKitが対応するフレームワークである。プロジェクトの要件（SSR/SSGの必要性、SEO要件、チームのスキルセット）に応じて選定すべきである。

### Q2: 小規模プロジェクトでもFeature-based構成を採用すべきか？
ファイル数が50未満の小規模プロジェクトでは、型ベース（Technical-based）構成でも問題ない。しかし、プロジェクトの成長が見込まれる場合は、初期段階からFeature-based構成を採用することを推奨する。後からの移行はimportパスの変更やテストの修正が必要となり、コストが大きくなるためである。最初は`features/`と`shared/`の2階層から始め、必要に応じてfeatureを追加していくアプローチが現実的である。

### Q3: 状態管理ライブラリはどう選定すべきか？
まず「本当にグローバル状態管理が必要か」を問うべきである。多くの場合、サーバー状態はTanStack QueryやSWRで管理し、URL状態はuseSearchParamsで管理すれば、グローバル状態管理ライブラリが不要になる。それでもグローバル状態が必要な場合、シンプルさを重視するならZustand、細かい再レンダリング制御が必要ならJotaiを選ぶ。Reduxは大規模チームでの実績があるが、ボイラープレートが多いため新規プロジェクトでの採用は減少傾向にある。

## まとめ

このガイドでは以下を学びました:

- SPA/MPA/SSRなどレンダリング方式の特徴と選定基準
- Feature-basedなプロジェクト構成とコンポーネント設計パターン
- Zustand/Jotaiによるクライアント状態管理とTanStack Queryによるサーバー状態管理
- Next.js App Routerを中心としたファイルベースルーティングとナビゲーション設計
- React Hook FormとZodを活用したフォーム設計・バリデーションパターン
- Vercel/Cloudflare/AWSへのデプロイ戦略とパフォーマンス最適化

## 参考文献

1. Next.js. "Documentation." nextjs.org/docs, 2024.
2. React. "React Documentation." react.dev, 2024.
3. TanStack. "TanStack Query Documentation." tanstack.com, 2024.
4. Zustand. "Bear necessities for state management." github.com/pmndrs/zustand, 2024.
5. Vercel. "Deployment Documentation." vercel.com/docs, 2024.

## 関連Skills

- [ブラウザとWebプラットフォーム](../browser-and-web-platform/) — ブラウザとWebプラットフォーム
- [ネットワーク基礎](../network-fundamentals/) — ネットワーク基礎
- [APIガイド](../api-and-library-guide/) — API・ライブラリ設計
