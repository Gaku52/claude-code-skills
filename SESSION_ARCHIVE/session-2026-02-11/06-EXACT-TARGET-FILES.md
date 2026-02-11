# 正確な目標ファイル一覧（Agentプロンプト準拠）

> **重要**: SKILL.mdのファイル名とAgentが実際に作成したファイル名は一致しない場合がある。
> このファイルがAgentプロンプトに基づく「正」の目標一覧。
> 再開時はこのファイルを参照し、SKILL.mdではなくこの一覧に基づいてAgentを起動すること。

---

## windows-application-development（目標14ファイル）
### docs/00-fundamentals/（4ファイル — 全完了）
- [x] 00-desktop-app-overview.md
- [x] 01-architecture-patterns.md
- [x] 02-native-features.md
- [x] 03-cross-platform.md

### docs/01-wpf-and-winui/（3ファイル — 全完了）
- [x] 00-windows-ui-frameworks.md
- [x] 01-winui3-basics.md
- [x] 02-webview2.md

### docs/02-electron-and-tauri/（4ファイル — 全完了）
- [x] 00-electron-setup.md
- [x] 01-electron-advanced.md
- [x] 02-tauri-setup.md
- [x] 03-tauri-advanced.md

### docs/03-distribution/（3ファイル — 1完了/2未完了）
- [x] 00-packaging-and-signing.md
- [ ] 01-auto-update.md — 自動更新。electron-updater、Tauri updater、更新サーバー、差分更新、ロールバック
- [ ] 02-store-distribution.md — ストア配布。Microsoft Store(MSIX)、Mac App Store、GitHub Releases、CI/CD

---

## development-environment-setup（目標14ファイル）
### docs/00-editor-and-tools/（4ファイル — 全完了）
- [x] 00-vscode-setup.md
- [x] 01-terminal-setup.md
- [x] 02-git-config.md
- [x] 03-ai-tools.md

### docs/01-runtime-and-package/（4ファイル — 全完了）
- [x] 00-version-managers.md
- [x] 01-package-managers.md
- [x] 02-monorepo-setup.md
- [x] 03-linter-formatter.md

### docs/02-docker-dev/（3ファイル — 1完了/2未完了）
- [x] 00-docker-for-dev.md
- [ ] 01-devcontainer.md — Dev Container。.devcontainer設定、VS Code統合、GitHub Codespaces
- [ ] 02-local-services.md — ローカルサービス。DB(PostgreSQL/MySQL/Redis)、MailHog、MinIOのDocker化

### docs/03-team-setup/（3ファイル — 全未完了）
- [ ] 00-project-standards.md — プロジェクト標準。EditorConfig、.npmrc、.nvmrc、共通設定
- [ ] 01-onboarding-automation.md — オンボーディング自動化。セットアップスクリプト、Makefile
- [ ] 02-documentation-setup.md — ドキュメント環境。VitePress/Docusaurus、ADR

---

## docker-container-guide（目標22ファイル）
### docs/00-fundamentals/（4ファイル — 全完了）
- [x] 00-container-overview.md
- [x] 01-docker-install.md
- [x] 02-docker-basics.md
- [x] 03-image-management.md

### docs/01-dockerfile/（4ファイル — 全完了）
- [x] 00-dockerfile-basics.md
- [x] 01-multi-stage-build.md
- [x] 02-optimization.md
- [x] 03-language-specific.md

### docs/02-compose/（3ファイル — 全未完了）
- [ ] 00-compose-basics.md — Docker Compose基礎。docker-compose.yml構文、services/volumes/networks
- [ ] 01-compose-advanced.md — Compose応用。プロファイル、depends_on、healthcheck、環境変数
- [ ] 02-development-workflow.md — Compose開発ワークフロー。ホットリロード、デバッグ、CI統合

### docs/03-networking/（3ファイル — 全完了）
- [x] 00-docker-networking.md
- [x] 01-volume-and-storage.md
- [x] 02-reverse-proxy.md

### docs/04-production/（3ファイル — 全完了）
- [x] 00-production-best-practices.md
- [x] 01-monitoring.md
- [x] 02-ci-cd-docker.md

### docs/05-orchestration/（3ファイル — 2完了/1未完了）
- [x] 00-orchestration-overview.md
- [x] 01-kubernetes-basics.md
- [ ] 02-kubernetes-advanced.md — Kubernetes応用。Helm、Ingress、ConfigMap/Secret、HPA

### docs/06-security/（2ファイル — 全未完了）
- [ ] 00-container-security.md — コンテナセキュリティ。イメージスキャン(Trivy)、最小権限、シークレット管理
- [ ] 01-supply-chain-security.md — サプライチェーンセキュリティ。イメージ署名(cosign)、SBOM

---

## aws-cloud-guide（目標29ファイル）
### docs/00-fundamentals/（3ファイル — 全完了）
- [x] 00-cloud-overview.md
- [x] 01-aws-account-setup.md
- [x] 02-aws-cli-sdk.md

### docs/01-compute/（3ファイル — 全完了）
- [x] 00-ec2-basics.md
- [x] 01-ec2-advanced.md
- [x] 02-elastic-beanstalk.md

### docs/02-storage/（3ファイル — 全完了）
- [x] 00-s3-basics.md
- [x] 01-s3-advanced.md
- [x] 02-cloudfront.md

### docs/03-database/（3ファイル — 全未完了）
- [ ] 00-rds-basics.md — RDS基礎。MySQL/PostgreSQL、マルチAZ、リードレプリカ、バックアップ
- [ ] 01-dynamodb.md — DynamoDB。テーブル設計、GSI/LSI、キャパシティモード、DAX
- [ ] 02-elasticache.md — ElastiCache。Redis/Memcached、キャッシュ戦略、セッション管理

### docs/04-networking/（3ファイル — 全未完了）
- [ ] 00-vpc-basics.md — VPC基礎。サブネット、ルートテーブル、IGW/NAT GW、NACL
- [ ] 01-route53.md — Route 53。DNS設定、ルーティングポリシー、ヘルスチェック
- [ ] 02-api-gateway.md — API Gateway。REST/HTTP API、Lambda統合、認証

### docs/05-serverless/（3ファイル — 全完了）
- [x] 00-lambda-basics.md
- [x] 01-lambda-advanced.md
- [x] 02-serverless-patterns.md

### docs/06-containers/（3ファイル — 全完了）
- [x] 00-ecs-basics.md
- [x] 01-ecr.md
- [x] 02-eks-overview.md

### docs/07-devops/（3ファイル — 2完了/1未完了）
- [x] 00-cloudformation.md
- [x] 01-cdk.md
- [ ] 02-codepipeline.md — CodePipeline。CodeCommit/Build/Deploy、GitHub統合

### docs/08-security/（3ファイル — 全未完了）
- [ ] 00-iam-deep-dive.md — IAM詳解。ポリシー構文、STS、クロスアカウント、最小権限
- [ ] 01-secrets-management.md — シークレット管理。Secrets Manager、Parameter Store、KMS
- [ ] 02-waf-shield.md — WAF/Shield。WAFルール、DDoS対策、マネージドルール

### docs/09-cost/（2ファイル — 全未完了）
- [ ] 00-cost-optimization.md — コスト最適化。Cost Explorer、Budgets、Savings Plans
- [ ] 01-well-architected.md — Well-Architected。6つの柱、レビュープロセス

---

## security-fundamentals（目標25ファイル）
### docs/00-basics/（3ファイル — 全完了）
- [x] 00-security-overview.md
- [x] 01-threat-modeling.md
- [x] 02-security-principles.md

### docs/01-web-security/（5ファイル — 全完了）
- [x] 00-owasp-top10.md
- [x] 01-xss-prevention.md
- [x] 02-csrf-clickjacking.md
- [x] 03-injection.md
- [x] 04-auth-vulnerabilities.md

### docs/02-cryptography/（3ファイル — 1完了/2未完了）
- [x] 00-crypto-basics.md
- [ ] 01-tls-certificates.md — TLS/証明書。TLSハンドシェイク、証明書チェーン、Let's Encrypt、mTLS
- [ ] 02-key-management.md — 鍵管理。鍵ライフサイクル、HSM、KMS、エンベロープ暗号化

### docs/03-network-security/（3ファイル — 全未完了）
- [ ] 00-network-security-basics.md — ネットワークセキュリティ基礎。ファイアウォール、IDS/IPS、VPN
- [ ] 01-dns-security.md — DNSセキュリティ。DNSSEC、DNS over HTTPS、ポイズニング対策
- [ ] 02-api-security.md — APIセキュリティ。OAuth2/JWT、レートリミット、入力検証

### docs/04-application-security/（4ファイル — 全未完了）
- [ ] 00-secure-coding.md — セキュアコーディング。入力検証、出力エンコード、エラーハンドリング
- [ ] 01-dependency-security.md — 依存関係セキュリティ。SCA、Dependabot、SBOM
- [ ] 02-container-security.md — コンテナセキュリティ。イメージスキャン、ランタイム保護
- [ ] 03-sast-dast.md — SAST/DAST。静的/動的解析、SonarQube、OWASP ZAP

### docs/05-cloud-security/（3ファイル — 全未完了）
- [ ] 00-cloud-security-basics.md — クラウドセキュリティ基礎。責任共有モデル、IAM、暗号化
- [ ] 01-aws-security.md — AWSセキュリティ。GuardDuty、Security Hub、CloudTrail
- [ ] 02-infrastructure-as-code-security.md — IaCセキュリティ。tfsec、Checkov、ポリシーasコード

### docs/06-operations/（4ファイル — 全未完了）
- [ ] 00-incident-response.md — インシデント対応。対応フロー、CSIRT、フォレンジック
- [ ] 01-monitoring-logging.md — 監視/ログ。SIEM、ログ集約、異常検知
- [ ] 02-compliance.md — コンプライアンス。GDPR、SOC2、PCI DSS
- [ ] 03-security-culture.md — セキュリティ文化。DevSecOps、バグバウンティ

---

## devops-and-github-actions（目標17ファイル）
### docs/00-devops-basics/（4ファイル — 全完了）
- [x] 00-devops-overview.md
- [x] 01-ci-cd-concepts.md
- [x] 02-infrastructure-as-code.md
- [x] 03-gitops.md

### docs/01-github-actions/（5ファイル — 全完了）
- [x] 00-actions-basics.md
- [x] 01-actions-advanced.md
- [x] 02-reusable-workflows.md
- [x] 03-ci-recipes.md
- [x] 04-security-actions.md

### docs/02-deployment/（4ファイル — 全未完了）
- [ ] 00-deployment-strategies.md — デプロイ戦略。Blue-Green、Canary、Rolling、Feature Flag
- [ ] 01-cloud-deployment.md — クラウドデプロイ。AWS、Vercel、Cloudflare Workers
- [ ] 02-container-deployment.md — コンテナデプロイ。ECS/K8s、ArgoCD
- [ ] 03-release-management.md — リリース管理。セマンティックバージョニング、CHANGELOG

### docs/03-monitoring/（4ファイル — 全未完了）
- [ ] 00-observability.md — オブザーバビリティ。ログ/メトリクス/トレース、OpenTelemetry
- [ ] 01-monitoring-tools.md — 監視ツール。Datadog、Grafana、CloudWatch
- [ ] 02-alerting.md — アラート戦略。アラート設計、エスカレーション、ポストモーテム
- [ ] 03-performance-monitoring.md — パフォーマンス監視。APM、RUM、Core Web Vitals

---

## version-control-and-jujutsu（目標12ファイル）
### docs/00-git-internals/（4ファイル — 全未完了）
- [ ] 00-git-object-model.md — Gitオブジェクトモデル。blob/tree/commit/tag、SHA-1
- [ ] 01-refs-and-branches.md — Ref・ブランチ。HEAD、reflog、detached HEAD
- [ ] 02-merge-algorithms.md — マージアルゴリズム。3-way merge、ort、rebase
- [ ] 03-packfile-gc.md — Packfile/GC。delta圧縮、リポジトリ最適化

### docs/01-advanced-git/（4ファイル — 全未完了）
- [ ] 00-interactive-rebase.md — インタラクティブRebase。squash、fixup、reword
- [ ] 01-worktree-submodule.md — Worktree/Submodule。複数作業ディレクトリ
- [ ] 02-bisect-blame.md — bisect/blame。バグ特定、二分探索
- [ ] 03-hooks-automation.md — Git Hooks。pre-commit、husky、lint-staged

### docs/02-jujutsu/（4ファイル — 全未完了）
- [ ] 00-jujutsu-introduction.md — Jujutsu入門。Gitとの違い、基本操作
- [ ] 01-jujutsu-workflow.md — Jujutsuワークフロー。変更セット、自動リベース
- [ ] 02-jujutsu-advanced.md — Jujutsu応用。revset、テンプレート、Git連携
- [ ] 03-git-to-jujutsu.md — Git→Jujutsu移行。co-located repo、操作対応表

---

## typescript-complete-guide（目標25ファイル）
### docs/00-basics/（5ファイル — 全完了）
- [x] 00-typescript-overview.md
- [x] 01-type-basics.md
- [x] 02-functions-and-objects.md
- [x] 03-union-intersection.md
- [x] 04-generics.md

### docs/01-advanced-types/（5ファイル — 全完了）
- [x] 00-conditional-types.md
- [x] 01-mapped-types.md
- [x] 02-template-literal-types.md
- [x] 03-type-challenges.md
- [x] 04-declaration-files.md

### docs/02-patterns/（5ファイル — 全未完了）
- [ ] 00-error-handling.md — Result型、カスタムエラー、zod
- [ ] 01-builder-pattern.md — 型安全なビルダー、Fluent API
- [ ] 02-discriminated-unions.md — 判別共用体、Redux、網羅性チェック
- [ ] 03-branded-types.md — ブランド型、公称型、opaque型
- [ ] 04-dependency-injection.md — DI。inversify、tsyringe

### docs/03-tooling/（5ファイル — 全未完了）
- [ ] 00-tsconfig.md — tsconfig.json。コンパイラオプション全解説
- [ ] 01-build-tools.md — ビルドツール。tsc、esbuild、SWC、Vite
- [ ] 02-testing-typescript.md — テスト。Vitest、Jest、型テスト
- [ ] 03-migration-guide.md — JS→TS移行。段階的移行
- [ ] 04-eslint-typescript.md — ESLint + TypeScript

### docs/04-ecosystem/（5ファイル — 全未完了）
- [ ] 00-zod-validation.md — Zod。スキーマ定義、バリデーション
- [ ] 01-prisma-typescript.md — Prisma + TypeScript
- [ ] 02-trpc.md — tRPC。型安全なAPI
- [ ] 03-effect-ts.md — Effect-ts。エフェクトシステム
- [ ] 04-typescript-5x.md — TypeScript 5.x新機能

---

## rust-systems-programming（目標25ファイル）
### docs/00-basics/（5ファイル — 全完了）
- [x] 00-rust-overview.md
- [x] 01-ownership-borrowing.md
- [x] 02-types-and-traits.md
- [x] 03-error-handling.md
- [x] 04-collections-iterators.md

### docs/01-advanced/（5ファイル — 4完了/1未完了）
- [x] 00-lifetimes.md
- [x] 01-smart-pointers.md
- [x] 02-closures-fn-traits.md
- [x] 03-unsafe-rust.md
- [ ] 04-macros.md — マクロ。宣言的/手続き的マクロ、derive、attribute

### docs/02-async/（5ファイル — 全未完了）
- [ ] 00-async-basics.md — async/await基礎。Future trait、tokio
- [ ] 01-tokio-runtime.md — Tokioランタイム。タスク管理、チャネル
- [ ] 02-async-patterns.md — 非同期パターン。Stream、並行制限
- [ ] 03-networking.md — ネットワーク。reqwest/hyper、WebSocket、tonic
- [ ] 04-axum-web.md — Axum。ルーティング、ミドルウェア、状態管理

### docs/03-systems/（5ファイル — 全未完了）
- [ ] 00-memory-layout.md — メモリレイアウト。スタック/ヒープ、repr
- [ ] 01-concurrency.md — 並行性。スレッド、Mutex/RwLock、rayon
- [ ] 02-ffi-interop.md — FFI。bindgen、PyO3、napi-rs
- [ ] 03-embedded-wasm.md — 組み込み/WASM。no_std、wasm-bindgen
- [ ] 04-cli-tools.md — CLIツール。clap、クロスコンパイル

### docs/04-ecosystem/（5ファイル — 全未完了）
- [ ] 00-cargo-workspace.md — Cargo/ワークスペース。features、publish
- [ ] 01-testing.md — テスト。proptest、criterion
- [ ] 02-serde.md — Serde。JSON/TOML/YAML
- [ ] 03-database.md — データベース。sqlx、diesel、SeaORM
- [ ] 04-best-practices.md — ベストプラクティス。clippy、API設計

---

## go-practical-guide（目標18ファイル）
### docs/00-basics/（4ファイル — 全完了）
- [x] 00-go-overview.md
- [x] 01-types-and-structs.md
- [x] 02-error-handling.md
- [x] 03-packages-modules.md

### docs/01-concurrency/（4ファイル — 全完了）
- [x] 00-goroutines-channels.md
- [x] 01-sync-primitives.md
- [x] 02-concurrency-patterns.md
- [x] 03-context.md

### docs/02-web/（5ファイル — 4完了/1未完了）
- [x] 00-net-http.md
- [x] 01-gin-echo.md
- [x] 02-database.md
- [x] 03-grpc.md
- [ ] 04-testing.md — テスト。table-driven tests、testify、httptest

### docs/03-tools/（5ファイル — 全未完了）
- [ ] 00-cli-development.md — CLI開発。cobra、flag、promptui
- [ ] 01-generics.md — ジェネリクス。型パラメータ、制約
- [ ] 02-profiling.md — プロファイリング。pprof、trace
- [ ] 03-deployment.md — デプロイ。Docker、クロスコンパイル
- [ ] 04-best-practices.md — ベストプラクティス。Effective Go

---

## sql-and-query-mastery（目標19ファイル — 全未完了）
### docs/00-basics/（5ファイル）
- [ ] 00-sql-overview.md, 01-crud-operations.md, 02-joins.md, 03-aggregation.md, 04-subqueries.md
### docs/01-advanced/（5ファイル）
- [ ] 00-window-functions.md, 01-cte-recursive.md, 02-transactions.md, 03-indexing.md, 04-query-optimization.md
### docs/02-design/（4ファイル）
- [ ] 00-normalization.md, 01-schema-design.md, 02-migration.md, 03-data-modeling.md
### docs/03-practical/（5ファイル）
- [ ] 00-postgresql-features.md, 01-security.md, 02-performance-tuning.md, 03-orm-comparison.md, 04-nosql-comparison.md

---

## design-patterns-guide（目標20ファイル）
### docs/00-creational/（4ファイル — 全完了）
- [x] 00-singleton.md, 01-factory.md, 02-builder.md, 03-prototype.md
### docs/01-structural/（5ファイル — 全完了）
- [x] 00-adapter.md, 01-decorator.md, 02-facade.md, 03-proxy.md, 04-composite.md
### docs/02-behavioral/（5ファイル — 2完了/3未完了）
- [x] 00-observer.md, [x] 01-strategy.md
- [ ] 02-command.md, 03-state.md, 04-iterator.md
### docs/03-functional/（3ファイル — 全未完了）
- [ ] 00-monad.md, 01-functor-applicative.md, 02-fp-patterns.md
### docs/04-architectural/（3ファイル — 全未完了）
- [ ] 00-mvc-mvvm.md, 01-repository-pattern.md, 02-event-sourcing-cqrs.md

---

## system-design-guide（目標18ファイル — 全未完了）
### docs/00-fundamentals/（4）
- [ ] 00-system-design-overview.md, 01-scalability.md, 02-reliability.md, 03-cap-theorem.md
### docs/01-components/（5）
- [ ] 00-load-balancer.md, 01-caching.md, 02-message-queue.md, 03-cdn.md, 04-database-scaling.md
### docs/02-architecture/（4）
- [ ] 00-monolith-vs-microservices.md, 01-clean-architecture.md, 02-ddd.md, 03-event-driven.md
### docs/03-case-studies/（5）
- [ ] 00-url-shortener.md, 01-chat-system.md, 02-notification-system.md, 03-rate-limiter.md, 04-search-engine.md

---

## clean-code-principles（目標20ファイル — 全未完了）
### docs/00-principles/（5）
- [ ] 00-clean-code-overview.md, 01-solid.md, 02-dry-kiss-yagni.md, 03-coupling-cohesion.md, 04-law-of-demeter.md
### docs/01-practices/（5）
- [ ] 00-naming.md, 01-functions.md, 02-error-handling.md, 03-comments.md, 04-testing-principles.md
### docs/02-refactoring/（5）
- [ ] 00-code-smells.md, 01-refactoring-techniques.md, 02-legacy-code.md, 03-technical-debt.md, 04-continuous-improvement.md
### docs/03-practices-advanced/（5）
- [ ] 00-immutability.md, 01-composition-over-inheritance.md, 02-functional-principles.md, 03-api-design.md, 04-code-review-checklist.md

---

## algorithm-and-data-structures（目標24ファイル）
### docs/00-complexity/（3ファイル — 全完了）
- [x] 00-big-o-notation.md, 01-complexity-analysis.md, 02-space-time-tradeoff.md
### docs/01-data-structures/（7ファイル — 全完了）
- [x] 00-arrays-strings.md, 01-linked-lists.md, 02-stacks-queues.md, 03-hash-tables.md, 04-trees.md, 05-heaps.md, 06-graphs.md
### docs/02-algorithms/（8ファイル — 全未完了）
- [ ] 00-sorting.md, 01-searching.md, 02-graph-traversal.md, 03-shortest-path.md, 04-dynamic-programming.md, 05-greedy.md, 06-divide-conquer.md, 07-backtracking.md
### docs/03-advanced/（4ファイル — 全未完了）
- [ ] 00-union-find.md, 01-segment-tree.md, 02-string-algorithms.md, 03-network-flow.md
### docs/04-practice/（2ファイル — 全未完了）
- [ ] 00-problem-solving.md, 01-competitive-programming.md

---

## regex-and-text-processing（目標12ファイル — 全未完了）
### docs/00-basics/（4）
- [ ] 00-regex-overview.md, 01-basic-syntax.md, 02-character-classes.md, 03-quantifiers-anchors.md
### docs/01-advanced/（4）
- [ ] 00-groups-backreferences.md, 01-lookaround.md, 02-unicode-regex.md, 03-performance.md
### docs/02-practical/（4）
- [ ] 00-language-specific.md, 01-common-patterns.md, 02-text-processing.md, 03-regex-alternatives.md

---

## ai-era-gadgets（目標12ファイル）
### docs/00-smartphones/（4ファイル — 全完了）
- [x] 00-ai-smartphones.md, 01-ai-cameras.md, 02-ai-assistants.md, 03-wearables.md
### docs/01-computing/（4ファイル — 1完了/3未完了）
- [x] 00-ai-pcs.md
- [ ] 01-gpu-computing.md, 02-edge-ai.md, 03-cloud-ai-hardware.md
### docs/02-emerging/（4ファイル — 全未完了）
- [ ] 00-ar-vr-ai.md, 01-robotics.md, 02-smart-home.md, 03-future-hardware.md

---

## ai-analysis-guide（目標16ファイル — 全未完了）
### docs/00-fundamentals/（4）
- [ ] 00-ai-analysis-overview.md, 01-data-preprocessing.md, 02-ml-basics.md, 03-python-ml-stack.md
### docs/01-classical-ml/（4）
- [ ] 00-regression.md, 01-classification.md, 02-clustering.md, 03-dimensionality-reduction.md
### docs/02-deep-learning/（4）
- [ ] 00-neural-networks.md, 01-cnn.md, 02-rnn-transformer.md, 03-frameworks.md
### docs/03-applied/（4）
- [ ] 00-nlp.md, 01-computer-vision.md, 02-mlops.md, 03-responsible-ai.md

---

## ai-audio-generation（目標14ファイル — 全未完了）
### docs/00-fundamentals/（4）
- [ ] 00-audio-ai-overview.md, 01-audio-basics.md, 02-tts-technologies.md, 03-stt-technologies.md
### docs/01-music/（4）
- [ ] 00-music-generation.md, 01-stem-separation.md, 02-audio-effects.md, 03-midi-ai.md
### docs/02-voice/（3）
- [ ] 00-voice-cloning.md, 01-voice-assistants.md, 02-podcast-tools.md
### docs/03-development/（3）
- [ ] 00-audio-apis.md, 01-audio-processing.md, 02-real-time-audio.md

---

## ai-visual-generation（目標14ファイル — 全未完了）
### docs/00-fundamentals/（3）
- [ ] 00-visual-ai-overview.md, 01-diffusion-models.md, 02-prompt-engineering-visual.md
### docs/01-image/（4）
- [ ] 00-image-generation.md, 01-image-editing.md, 02-upscaling.md, 03-design-tools.md
### docs/02-video/（3）
- [ ] 00-video-generation.md, 01-video-editing.md, 02-animation.md
### docs/03-3d/（4）
- [ ] 00-3d-generation.md, 01-game-assets.md, 02-virtual-try-on.md, 03-ethical-considerations.md

---

## llm-and-ai-comparison（目標20ファイル）
### docs/00-fundamentals/（4ファイル — 全完了）
- [x] 00-llm-overview.md, 01-tokenization.md, 02-inference.md, 03-fine-tuning.md
### docs/01-models/（5ファイル — 2完了/3未完了）
- [x] 00-claude.md, [x] 01-gpt.md
- [ ] 02-gemini.md, 03-open-source.md, 04-model-comparison.md
### docs/02-applications/（5ファイル — 全未完了）
- [ ] 00-prompt-engineering.md, 01-rag.md, 02-function-calling.md, 03-embeddings.md, 04-multimodal.md
### docs/03-infrastructure/（4ファイル — 全未完了）
- [ ] 00-api-integration.md, 01-vector-databases.md, 02-local-llm.md, 03-evaluation.md
### docs/04-ethics/（2ファイル — 全未完了）
- [ ] 00-ai-safety.md, 01-ai-governance.md

---

## custom-ai-agents（目標19ファイル — 全未完了）
### docs/00-fundamentals/（4）
- [ ] 00-agent-overview.md, 01-agent-frameworks.md, 02-tool-use.md, 03-memory-systems.md
### docs/01-patterns/（4）
- [ ] 00-single-agent.md, 01-multi-agent.md, 02-workflow-agents.md, 03-autonomous-agents.md
### docs/02-implementation/（5）
- [ ] 00-langchain-agent.md, 01-langgraph.md, 02-mcp-agents.md, 03-claude-agent-sdk.md, 04-evaluation.md
### docs/03-applications/（4）
- [ ] 00-coding-agents.md, 01-research-agents.md, 02-customer-support.md, 03-data-agents.md
### docs/04-production/（2）
- [ ] 00-deployment.md, 01-safety.md

---

## ai-automation-and-monetization（目標15ファイル — 全未完了）
### docs/00-automation/（4）
- [ ] 00-automation-overview.md, 01-workflow-automation.md, 02-document-processing.md, 03-email-communication.md
### docs/01-business/（4）
- [ ] 00-ai-saas.md, 01-ai-consulting.md, 02-content-creation.md, 03-ai-marketplace.md
### docs/02-monetization/（3）
- [ ] 00-pricing-models.md, 01-cost-management.md, 02-scaling-strategy.md
### docs/03-case-studies/（4）
- [ ] 00-successful-ai-products.md, 01-solo-developer.md, 02-startup-guide.md, 03-future-opportunities.md

---

## ai-era-development-workflow（目標15ファイル — 全未完了）
### docs/00-fundamentals/（3）
- [ ] 00-ai-dev-landscape.md, 01-ai-dev-mindset.md, 02-prompt-driven-development.md
### docs/01-ai-coding/（4）
- [ ] 00-github-copilot.md, 01-claude-code.md, 02-cursor-and-windsurf.md, 03-ai-coding-best-practices.md
### docs/02-workflow/（4）
- [ ] 00-ai-testing.md, 01-ai-code-review.md, 02-ai-documentation.md, 03-ai-debugging.md
### docs/03-team/（4）
- [ ] 00-ai-team-practices.md, 01-ai-onboarding.md, 02-future-of-development.md, 03-ai-ethics-development.md

---

## 統計サマリー（2026-02-11 セッション2 最終スナップショット）
- 全目標ファイル数: 439（24 Skill合計）
- 作成済み: 345（ディスク確認。Agentがまだ稼働中のため更に増加の可能性あり）
- 未作成: 94（再開時にfindで再確認必須）

### Skill別 作成済み/目標 内訳（セッション2 最終）
| Skill | 作成済み | 目標 | 未作成 | 状態 |
|-------|---------|------|--------|------|
| windows-application-development | 14 | 14 | 0 | 完了 |
| go-practical-guide | 18 | 18 | 0 | 完了 |
| development-environment-setup | 14 | 14 | 0 | 完了 |
| devops-and-github-actions | 17 | 17 | 0 | 完了 |
| aws-cloud-guide | 24 | 29 | 5 | 残少 |
| typescript-complete-guide | 24 | 25 | 1 | 残少 |
| algorithm-and-data-structures | 22 | 24 | 2 | 残少 |
| regex-and-text-processing | 10 | 12 | 2 | 残少 |
| version-control-and-jujutsu | 11 | 12 | 1 | 残少 |
| llm-and-ai-comparison | 17 | 20 | 3 | 進行中 |
| security-fundamentals | 21 | 25 | 4 | 進行中 |
| rust-systems-programming | 21 | 25 | 4 | 進行中 |
| docker-container-guide | 19 | 22 | 3 | 進行中 |
| design-patterns-guide | 15 | 20 | 5 | 進行中 |
| ai-era-gadgets | 10 | 12 | 2 | 残少 |
| sql-and-query-mastery | 12 | 19 | 7 | 進行中 |
| clean-code-principles | 12 | 20 | 8 | 進行中 |
| custom-ai-agents | 11 | 19 | 8 | 進行中 |
| ai-analysis-guide | 10 | 16 | 6 | 進行中 |
| ai-audio-generation | 9 | 14 | 5 | 進行中 |
| ai-visual-generation | 8 | 14 | 6 | 進行中 |
| ai-automation-and-monetization | 9 | 15 | 6 | 進行中 |
| ai-era-development-workflow | 9 | 15 | 6 | 進行中 |
| system-design-guide | 8 | 18 | 10 | 進行中 |
| **合計** | **345** | **439** | **94** | **Agentまだ稼働中** |

### 完了済み11+4 Skill
全体の総docs .mdファイル数: 807+
完了Skill追加: windows-app, go, dev-env, devops（4 Skill新規完了）

### 再開時の注意
- 上記数値はAgent途中停止の可能性があるため、再開時に find で再確認必須
- Agentが制限後も数ファイル書いている可能性あり → 実際は94未満かも
- 再開プロンプト: /Users/gaku/.claude/skills/RESUME_PROMPT.md
