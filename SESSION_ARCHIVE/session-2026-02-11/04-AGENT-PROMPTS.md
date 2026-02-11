# 動作確認済みAgentプロンプトテンプレート

## 基本テンプレート（10-15ファイル用）

```
あなたはMIT級品質の技術ガイドを日本語で執筆するライターです。以下の{N}ファイルを全て作成してください。

各ファイル構造: タイトル+1行概要(>引用)、「この章で学ぶこと」3個、コード例5個以上、ASCII図解3個以上、比較表2個以上、アンチパターン2個以上、FAQ3個以上、まとめ表、次に読むべきガイド、参考文献3個以上。全て日本語。

ディレクトリが存在しない場合は先にBashでmkdir -pしてください。

## ファイル一覧

### docs/{section}/
1. /Users/gaku/.claude/skills/{skill-name}/docs/{section}/{filename}.md — {テーマの簡潔な説明}
2. ...

必ずWriteツールで全{N}ファイルを書き込んでください。
```

## Agent起動コード

```python
# Task呼び出しパターン
Task(
    description="Write {skill} {N} guide files",
    subagent_type="general-purpose",
    run_in_background=True,
    prompt="..."  # 上記テンプレートを埋める
)
```

## 重要なポイント
- `subagent_type` は `"general-purpose"` を使用
- `run_in_background: true` で非同期実行
- プロンプトに「Writeツールで書き込め」と明示的に指示
- 「Bashでmkdir -p」の指示も入れる（ディレクトリ未作成の場合）
- ファイルパスは絶対パスで指定
- テーマは1行で簡潔に（長すぎるとコンテキスト消費）
