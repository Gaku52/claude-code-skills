# Python CLI Template (Typer)

完全な機能を持つ Python CLI テンプレート（Typer + Rich）

## 特徴

- ✅ Typer による型安全な CLI
- ✅ Rich による美しい出力
- ✅ pytest によるテスト
- ✅ TOML 設定ファイルサポート
- ✅ プラグインシステム
- ✅ ロギング
- ✅ エラーハンドリング

## セットアップ

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -e ".[dev]"

# テスト実行
pytest

# CLI 実行
mycli --help
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
mycli delete myapp
```

## ディレクトリ構造

```
.
├── src/
│   └── cli/
│       ├── __init__.py
│       ├── main.py           # エントリーポイント
│       ├── commands/         # コマンド定義
│       │   ├── create.py
│       │   ├── list.py
│       │   └── delete.py
│       ├── core/             # ビジネスロジック
│       │   └── project.py
│       └── utils/            # ユーティリティ
│           ├── logger.py
│           └── config.py
├── tests/                    # テスト
├── pyproject.toml           # プロジェクト設定
└── README.md
```

## 開発

```bash
# コードフォーマット
black src/ tests/

# リント
ruff check src/ tests/

# 型チェック
mypy src/

# テスト（カバレッジ付き）
pytest --cov=src tests/
```

## ビルド & 公開

```bash
# ビルド
python -m build

# PyPI へ公開
twine upload dist/*
```
