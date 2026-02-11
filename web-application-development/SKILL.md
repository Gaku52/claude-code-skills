# Webアプリケーション開発 完全ガイド

> Webアプリケーションの設計から本番デプロイまで。アーキテクチャ選定、状態管理、ルーティング、フォーム設計、認証統合、デプロイ戦略まで、モダンなWebアプリ開発の全体像を体系的に学ぶ。

## このSkillの対象者

- Webアプリケーションの設計・開発に携わるフルスタックエンジニア
- フロントエンドフレームワークの選定を検討している開発者
- 本番運用を見据えたWebアプリの構築を目指す開発者

## 前提知識

- HTML/CSS/JavaScript の基本 → 参照: [[web-development]]
- React の基本 → 参照: [[react-development]]

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

## 関連Skills

- [[web-development]] — モダンWeb開発基礎
- [[react-development]] — React開発
- [[nextjs-development]] — Next.js開発
- [[frontend-performance]] — フロントエンドパフォーマンス
- [[api-and-library-guide]] — API設計
