# Python Development Best Practices Checklist

> **目的**: 高品質な Python プロジェクトを構築するためのチェックリスト

## 📋 プロジェクト初期設定

### プロジェクト構造
- [ ] 適切なディレクトリ構造を作成（src レイアウト推奨）
- [ ] `pyproject.toml` でプロジェクト設定を定義
- [ ] `README.md` を作成（プロジェクト概要、セットアップ手順）
- [ ] `.gitignore` を設定
- [ ] `LICENSE` ファイルを追加（オープンソースの場合）

### 依存関係管理
- [ ] Poetry または pip-tools で依存関係を管理
- [ ] `pyproject.toml` に依存関係を記載
- [ ] 開発用依存関係を分離（`[project.optional-dependencies]`）
- [ ] Python バージョンを明示（`requires-python`）
- [ ] 定期的に依存関係を更新

### 環境設定
- [ ] 仮想環境を使用（venv, Poetry, conda）
- [ ] `.env.example` を作成（環境変数のテンプレート）
- [ ] `.env` を `.gitignore` に追加
- [ ] `python-dotenv` で環境変数を読み込み

---

## 🔍 コード品質

### 型ヒント
- [ ] すべての関数に型ヒントを追加
- [ ] `from __future__ import annotations` を使用（Python 3.10+）
- [ ] Pydantic で複雑なデータをバリデーション
- [ ] mypy で型チェックを実行
- [ ] `pyproject.toml` で mypy を strict モードに設定

### コードスタイル
- [ ] Ruff または Black でコードをフォーマット
- [ ] 行の長さを 100 文字に設定
- [ ] isort でインポートをソート
- [ ] docstring を追加（Google スタイル推奨）
- [ ] 命名規則に従う（PEP 8）

### Linting
- [ ] Ruff で Lint を実行
- [ ] エラーを自動修正（`--fix` オプション）
- [ ] pycodestyle, pyflakes, pep8-naming を有効化
- [ ] flake8-bugbear で潜在的なバグを検出
- [ ] bandit でセキュリティチェック

---

## 🧪 テスト

### ユニットテスト
- [ ] pytest でテストを作成
- [ ] すべての関数・クラスをテスト
- [ ] テストカバレッジ 80% 以上を目標
- [ ] `tests/` ディレクトリにテストを配置
- [ ] テストファイル名は `test_*.py`

### テスト戦略
- [ ] Fixture で共通のセットアップを定義
- [ ] `@pytest.mark.parametrize` で複数のケースをテスト
- [ ] モック/スタブを使用（`pytest-mock`）
- [ ] 非同期テストは `pytest-asyncio` を使用
- [ ] 統合テストを作成（`@pytest.mark.integration`）

### カバレッジ
- [ ] `pytest-cov` でカバレッジを測定
- [ ] HTML レポートを生成（`--cov-report=html`）
- [ ] カバレッジ閾値を設定（`--cov-fail-under=80`）
- [ ] カバレッジバッジを README に追加（CI/CD）

---

## 🔐 セキュリティ

### 認証・認可
- [ ] パスワードをハッシュ化（bcrypt, argon2）
- [ ] JWT トークン認証を実装
- [ ] トークンの有効期限を設定
- [ ] HTTPS のみで通信
- [ ] CORS を適切に設定

### データ保護
- [ ] 機密情報を環境変数に保存
- [ ] シークレットを `.env` に記載（Git に含めない）
- [ ] SQL インジェクション対策（ORM 使用）
- [ ] XSS 対策（入力のサニタイズ）
- [ ] CSRF トークンを使用

### セキュリティチェック
- [ ] bandit でセキュリティスキャン
- [ ] 依存関係の脆弱性をチェック（`pip-audit`）
- [ ] Secrets をハードコードしない
- [ ] ログに機密情報を出力しない

---

## 🚀 パフォーマンス

### 計測・プロファイリング
- [ ] ボトルネックを特定（cProfile, line_profiler）
- [ ] メモリ使用量を測定（memory_profiler）
- [ ] ベンチマークを作成（timeit）
- [ ] パフォーマンステストを自動化

### 最適化
- [ ] 適切なデータ構造を選択（list vs set vs dict）
- [ ] ジェネレータで遅延評価
- [ ] `functools.lru_cache` でメモ化
- [ ] NumPy/Pandas でベクトル化
- [ ] 並列処理・非同期処理を活用

### データベース
- [ ] N+1 問題を解決（`joinedload`, `prefetch_related`）
- [ ] インデックスを追加
- [ ] バルク挿入を使用
- [ ] クエリを最適化
- [ ] Redis でキャッシュ

---

## 📦 パッケージング・デプロイ

### ビルド
- [ ] `pyproject.toml` でビルド設定を定義
- [ ] `python -m build` でパッケージをビルド
- [ ] `twine check` で検証
- [ ] バージョン番号を適切に管理（SemVer）

### Docker
- [ ] `Dockerfile` を作成
- [ ] マルチステージビルドで最適化
- [ ] 非 root ユーザーで実行
- [ ] `.dockerignore` を設定
- [ ] `docker-compose.yml` で開発環境を構築

### CI/CD
- [ ] GitHub Actions/GitLab CI でテストを自動化
- [ ] Lint, 型チェック, テストを実行
- [ ] カバレッジレポートを生成
- [ ] 自動デプロイを設定
- [ ] バージョンタグで自動リリース

---

## 📚 ドキュメント

### コードドキュメント
- [ ] すべての関数に docstring を追加
- [ ] Google スタイル or NumPy スタイルを統一
- [ ] 型ヒントで引数・戻り値を明示
- [ ] 複雑なロジックにコメントを追加
- [ ] 例外を docstring に記載

### プロジェクトドキュメント
- [ ] `README.md` を充実させる
  - [ ] プロジェクト概要
  - [ ] インストール手順
  - [ ] クイックスタート
  - [ ] API ドキュメントへのリンク
  - [ ] コントリビューションガイド
- [ ] `CHANGELOG.md` を作成
- [ ] API ドキュメントを生成（Sphinx, MkDocs）
- [ ] 使用例を追加（examples/）

---

## 🔧 自動化

### Pre-commit フック
- [ ] `pre-commit` をインストール
- [ ] `.pre-commit-config.yaml` を作成
- [ ] Ruff, mypy を実行
- [ ] ファイル末尾の空行を修正
- [ ] 行末の空白を削除

### タスク自動化
- [ ] `Makefile` を作成
- [ ] `make install` で依存関係をインストール
- [ ] `make test` でテストを実行
- [ ] `make lint` で Lint を実行
- [ ] `make format` でフォーマット

### 定期タスク
- [ ] 依存関係を定期的に更新
- [ ] セキュリティスキャンを定期実行
- [ ] テストを定期実行（CI/CD）
- [ ] ドキュメントを最新に保つ

---

## 🌐 FastAPI 開発（Web API の場合）

### プロジェクト構成
- [ ] レイヤードアーキテクチャを採用
  - [ ] `api/`: エンドポイント
  - [ ] `models/`: データベースモデル
  - [ ] `schemas/`: Pydantic スキーマ
  - [ ] `services/`: ビジネスロジック
- [ ] ルーターで API をモジュール化
- [ ] 依存性注入を活用

### API 設計
- [ ] RESTful な URL 設計
- [ ] 適切な HTTP ステータスコードを返す
- [ ] ページネーションを実装
- [ ] フィルタリング・ソートを実装
- [ ] エラーハンドリングを統一

### ドキュメント
- [ ] OpenAPI ドキュメントを自動生成
- [ ] レスポンスモデルを定義
- [ ] 例を追加（`example` フィールド）
- [ ] タグでエンドポイントをグループ化

---

## 📊 データ処理（Pandas の場合）

### パフォーマンス
- [ ] `iterrows()` を避ける（ベクトル化）
- [ ] カテゴリ型でメモリ削減
- [ ] チャンクで大きなファイルを処理
- [ ] NumPy 配列を直接操作

### データクレンジング
- [ ] 欠損値を処理
- [ ] 重複を削除
- [ ] 型変換を実施
- [ ] 異常値を検出・処理

---

## ✅ リリース前チェック

### コード品質
- [ ] すべてのテストがパス
- [ ] カバレッジが閾値以上
- [ ] Lint エラーなし
- [ ] 型チェックエラーなし
- [ ] セキュリティスキャンでエラーなし

### ドキュメント
- [ ] README が最新
- [ ] CHANGELOG を更新
- [ ] API ドキュメントが最新
- [ ] 移行ガイドを作成（破壊的変更がある場合）

### デプロイ
- [ ] 環境変数を設定
- [ ] データベースマイグレーションを実行
- [ ] ヘルスチェックエンドポイントを実装
- [ ] ログ設定を確認
- [ ] モニタリングを設定

---

## 🎯 継続的改善

### 定期的なレビュー
- [ ] コードレビューを実施
- [ ] リファクタリングを計画
- [ ] 技術的負債を管理
- [ ] パフォーマンスを定期的に測定

### アップデート
- [ ] Python バージョンを最新に保つ
- [ ] 依存関係を定期的に更新
- [ ] セキュリティパッチを適用
- [ ] 新しいベストプラクティスを導入

---

## 📖 参考リソース

### 公式ドキュメント
- [Python 公式ドキュメント](https://docs.python.org/ja/)
- [FastAPI ドキュメント](https://fastapi.tiangolo.com/)
- [pytest ドキュメント](https://docs.pytest.org/)
- [Pydantic ドキュメント](https://docs.pydantic.dev/)

### ツール
- [Ruff](https://github.com/astral-sh/ruff) - 高速 Linter & Formatter
- [mypy](https://mypy.readthedocs.io/) - 型チェッカー
- [Poetry](https://python-poetry.org/) - 依存関係管理
- [pre-commit](https://pre-commit.com/) - Git フック

### ガイド
- [PEP 8](https://peps.python.org/pep-0008/) - Python スタイルガイド
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Real Python](https://realpython.com/) - Python チュートリアル

---

*このチェックリストを活用して、高品質な Python プロジェクトを構築しましょう！*
