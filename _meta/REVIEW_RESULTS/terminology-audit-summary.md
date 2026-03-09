# 用語一貫性監査レポート

> 実行日時: 2026-03-09 00:38:46

## 全体サマリー

| 項目 | 数値 |
|------|------|
| チェックファイル数 | 952 |
| 問題ありファイル数 | 597 |
| WARNING数 | 1050 |
| INFO数 | 292 |

---

## 頻出用語問題 (上位30件)

| 推奨表記 | 検出表記 | カテゴリ | ファイル数 | 出現回数 | 文脈依存 |
|---------|---------|---------|----------|---------|---------|
| データベース | DB | 日英混在 | 220 | 873 | ○ |
| すべて | 全て | 漢字ひらがな | 319 | 607 |  |
| レイヤ | レイヤー | カタカナ | 167 | 480 |  |
| インターフェース | インタフェース | カタカナ | 22 | 111 |  |
| インターフェース | インターフェイス | カタカナ | 33 | 99 |  |
| パラメータ | パラメーター | カタカナ | 30 | 94 |  |
| ユーザー | ユーザ(?!ー) | カタカナ | 20 | 90 |  |
| プロバイダ | プロバイダー | カタカナ | 24 | 60 |  |
| マネージャ | マネージャー | カタカナ | 24 | 58 |  |
| さまざま | 様々 | 漢字ひらがな | 40 | 51 | ○ |
| サーバー | サーバ(?!ー) | カタカナ | 16 | 48 |  |
| デプロイ | デプロイメント | カタカナ | 20 | 32 |  |
| ため | 為 | 漢字ひらがな | 14 | 19 | ○ |
| ハンドラ | ハンドラー | カタカナ | 13 | 19 |  |
| マイクロサービス | Microservices | 技術用語 | 11 | 14 | ○ |
| カテゴリ | カテゴリー | カタカナ | 7 | 12 |  |
| メモリ | メモリー | カタカナ | 7 | 10 |  |
| パラメータ | パラメタ | カタカナ | 3 | 9 |  |
| マイクロサービス | microservices | 技術用語 | 7 | 9 | ○ |

---

## カテゴリ別問題分布

| カテゴリ | ファイル数 | 問題数 | WARNING数 |
|---------|----------|-------|----------|
| .github | 3 | 0 | 0 |
| 01-cs-fundamentals | 135 | 182 | 145 |
| 02-programming | 124 | 114 | 88 |
| 03-software-design | 61 | 150 | 99 |
| 04-web-and-network | 79 | 116 | 90 |
| 05-infrastructure | 137 | 222 | 171 |
| 06-data-and-security | 66 | 135 | 98 |
| 07-ai | 133 | 136 | 104 |
| 08-hobby | 207 | 286 | 255 |
| CHANGELOG.md | 1 | 0 | 0 |
| CODE_OF_CONDUCT.md | 1 | 0 | 0 |
| CONTRIBUTING.md | 1 | 0 | 0 |
| MAINTENANCE.md | 1 | 1 | 0 |
| PROGRESS.md | 1 | 0 | 0 |
| README.md | 1 | 0 | 0 |
| SECURITY.md | 1 | 0 | 0 |

---

## WARNING数の多いファイル (上位20件)

| ファイル | WARNING | INFO |
|---------|---------|------|
| 01-cs-fundamentals/computer-science-fundamentals/docs/07-software-engineering/00-development-process.md | 7 | 0 |
| 08-hobby/dj-skills-guide/PROGRESS.md | 6 | 0 |
| 08-hobby/dj-skills-guide/docs/integration/README.md | 6 | 1 |
| 01-cs-fundamentals/operating-system-guide/docs/02-memory-management/02-memory-allocation.md | 5 | 1 |
| 04-web-and-network/api-and-library-guide/docs/03-api-security/02-input-validation.md | 5 | 1 |
| 06-data-and-security/security-fundamentals/docs/06-operations/02-compliance.md | 5 | 0 |
| 08-hobby/dj-skills-guide/docs/resources/README.md | 5 | 0 |
| 08-hobby/dj-skills-guide/docs/resources/plugins-tools.md | 5 | 1 |
| 01-cs-fundamentals/computer-science-fundamentals/docs/01-hardware-basics/05-io-systems.md | 4 | 1 |
| 01-cs-fundamentals/computer-science-fundamentals/docs/07-software-engineering/01-testing.md | 4 | 1 |
| 01-cs-fundamentals/operating-system-guide/docs/01-process-management/01-threads.md | 4 | 1 |
| 01-cs-fundamentals/operating-system-guide/docs/04-io-and-devices/01-interrupts-dma.md | 4 | 0 |
| 03-software-design/clean-code-principles/docs/01-practices/04-testing-principles.md | 4 | 1 |
| 03-software-design/system-design-guide/docs/00-fundamentals/00-system-design-overview.md | 4 | 2 |
| 03-software-design/system-design-guide/docs/02-architecture/03-event-driven.md | 4 | 2 |
| 04-web-and-network/api-and-library-guide/docs/00-api-design-principles/02-versioning-strategy.md | 4 | 0 |
| 04-web-and-network/api-and-library-guide/docs/02-sdk-and-libraries/00-sdk-design.md | 4 | 0 |
| 04-web-and-network/browser-and-web-platform/docs/03-web-apis/00-dom-api.md | 4 | 0 |
| 04-web-and-network/network-fundamentals/docs/04-advanced/02-network-debugging.md | 4 | 0 |
| 05-infrastructure/aws-cloud-guide/docs/04-networking/02-api-gateway.md | 4 | 1 |

---

## 正規化マッピング表（修正時参照）

| 推奨表記 | 非推奨表記 | カテゴリ |
|---------|----------|---------|
| インターフェース | インタフェース, インタフェイス, インターフェイス | カタカナ |
| メソッド | メッソド, メゾッド | カタカナ |
| パラメータ | パラメタ, パラメーター | カタカナ |
| プロパティ | プロパティー | カタカナ |
| セキュリティ | セキュリティー | カタカナ |
| カテゴリ | カテゴリー | カタカナ |
| メモリ | メモリー | カタカナ |
| ディレクトリ | ディレクトリー | カタカナ |
| ライブラリ | ライブラリー | カタカナ |
| リポジトリ | リポジトリー, レポジトリ, レポジトリー | カタカナ |
| サーバー | サーバ(?!ー) | カタカナ |
| コンテナ | コンテナー | カタカナ |
| ユーザー | ユーザ(?!ー) | カタカナ |
| ブラウザ | ブラウザー | カタカナ |
| ハンドラ | ハンドラー | カタカナ |
| コンパイラ | コンパイラー | カタカナ |
| レイヤ | レイヤー | カタカナ |
| マネージャ | マネージャー, マネジャ(?!ー), マネジャー | カタカナ |
| プロバイダ | プロバイダー | カタカナ |
| API | ＡＰＩ | 全角英数 |
| HTTP | ＨＴＴＰ, ｈｔｔｐ | 全角英数 |
| URL | ＵＲＬ, ｕｒｌ | 全角英数 |
| CSS | ＣＳＳ | 全角英数 |
| HTML | ＨＴＭＬ | 全角英数 |
| JSON | ＪＳＯＮ | 全角英数 |
| できる | 出来る | 漢字ひらがな |
| すべて | 全て, 総て | 漢字ひらがな |
| ミドルウェア | ミドルウエア | カタカナ |
| フレームワーク | フレイムワーク | カタカナ |
| デプロイ | デプロイメント, ディプロイ | カタカナ |
| アーキテクチャ | アーキテクチャー | カタカナ |
