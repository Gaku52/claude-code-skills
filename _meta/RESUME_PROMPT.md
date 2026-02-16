# Skills超大作 再開プロンプト（40KB+拡充フェーズ）

以下をそのままコピペして新しいClaude Codeセッションで貼り付けてください:

---

```
Skills超大作の40KB+拡充を続きから再開してください。

## 現在の状態（2026-02-15時点）
- Phase 1 = 全901ファイルが40KB+で完成
- 現在: 252/901（27%）コミット&プッシュ済み

### カテゴリ別進捗
- 03-software-design: 58/58 完了
- 06-data-and-security: 63/63 完了
- 07-ai: 120/125（残5ファイル）
  - analysis-guide: 残1（03-python-ml-stack.md）
  - audio-generation: 残1（03-midi-ai.md）
  - automation: 残3（00-ai-saas, 01-ai-consulting, 02-content-creation）
- 05-infrastructure: 8/130（残122ファイル）
- 01-cs-fundamentals: 0/131
- 02-programming: 0/118
- 04-web-and-network: 0/75
- 08-hobby: 0/201（一部4ファイル完了）

### 作業順序
07-ai残り5 → 05-infra残り122 → 02-prog → 01-cs → 04-web → 08-hobby

## 作業手順

### Step 1: 現状確認
for cat in /Users/gaku/.claude/skills/0*/; do
  name=$(basename "$cat")
  total=$(find "$cat" -path "*/docs/*.md" 2>/dev/null | wc -l | tr -d ' ')
  done=$(find "$cat" -path "*/docs/*.md" -size +40000c 2>/dev/null | wc -l | tr -d ' ')
  echo "$name: $done/$total"
done

### Step 2: 残りファイル特定
# 現在作業中のカテゴリの残りファイルを確認
find /Users/gaku/.claude/skills/07-ai/ -path "*/docs/*.md" -not -size +40000c | sort
find /Users/gaku/.claude/skills/05-infrastructure/ -path "*/docs/*.md" -not -size +40000c | sort

### Step 3: 並列エージェント起動
- 1エージェントあたり5-6ファイル（10+はタイムアウトリスク高）
- 同時8-10エージェント
- subagent_type="general-purpose", run_in_background=true
- 1ファイルずつRead→拡充→Write（全部読んでから書くのではなく）

### 各ファイルの拡充基準
- 40,000バイト以上（厳守）
- 日本語で記述
- Markdown形式、コードブロックには言語指定
- 実務で即座に使える具体例を豊富に含める
- 既存の構成・文体を維持しつつ大幅に拡充

### 検証コマンド
find <category> -path "*/docs/*.md" -not -size +40000c -exec sh -c \
  'echo "FAIL: $1 ($(wc -c < "$1") bytes)"' _ {} \;

### コミット&プッシュ
- エージェント停止後にコミット
- cd /Users/gaku/.claude/skills && git add -A && git commit && git push

## 運用ルール
- 90%到達で区切り停止 → クリーン再開が効率的
- コミット前にエージェント全停止
- レート制限到達前に自主停止 → トークン無駄を防ぐ
```

---

## 補足（人間向けメモ — 貼り付け不要）

- 上記の ``` 内をコピペするだけで新セッションから完全再開可能
- MEMORY.md（自動読み込み）にも進捗を記録済み
- 前回コミット: 0dec9db → 397c443（2026-02-15）
