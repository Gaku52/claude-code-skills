# Skills超大作 再開プロンプト

以下をそのままコピペして新しいClaude Codeセッションで貼り付けてください:

---

```
/write-batch を使ってSkills超大作の執筆を完全に続きから再開してください。

## Step 0: 全情報の読み込み（必須・省略禁止）

以下のファイルを全て読み込んでからStep 1に進んでください:

### 正の情報源（最優先）
1. /Users/gaku/.claude/skills/SESSION_ARCHIVE/session-2026-02-11/06-EXACT-TARGET-FILES.md
   → 全24 Skillの正確なファイル一覧（[x]完了/[ ]未作成）。唯一の正。SKILL.mdではなくこれを使う
2. /Users/gaku/.claude/skills/SESSION_ARCHIVE/session-2026-02-11/04-AGENT-PROMPTS.md
   → 動作確認済みAgentプロンプトテンプレート
3. /Users/gaku/.claude/skills/SESSION_ARCHIVE/session-2026-02-11/03-LESSONS.md
   → Agent運用の教訓（必読。Agent数・ファイル数の制限等）

### 設定・状態
4. /Users/gaku/.claude/skills/SESSION_ARCHIVE/session-2026-02-11/02-SETTINGS.md
   → 権限設定の正確な内容
5. /Users/gaku/.claude/skills/SESSION_ARCHIVE/session-2026-02-11/01-STATUS.md
   → 各Skillの状態概要

### 品質基準・テンプレート
6. /Users/gaku/.claude/plans/distributed-gliding-sprout.md
   → マスター計画書（品質基準・ガイド構成テンプレート含む）
7. /Users/gaku/.claude/skills/QUALITY_STANDARDS.md
   → MIT級品質基準書（存在すれば読む）
8. /Users/gaku/.claude/skills/TEMPLATES/GUIDE_TEMPLATE.md
   → ガイドファイル雛形（存在すれば読む）

### 万が一の深掘り用（通常は不要）
9. 前セッション完全ログ: /Users/gaku/.claude/projects/-Users-gaku/593e2fb0-c8cb-4d0c-a622-6d12761351b8.jsonl
   → 判断に迷った場合のみ参照

## Step 1: ディスク状態の検証

find /Users/gaku/.claude/skills/*/docs -name "*.md" -type f | wc -l
→ 597前後であれば正常（完了11 Skill ≈462 + 作業中24 Skill = 135）

## Step 2: 未作成ファイルの特定

06-EXACT-TARGET-FILES.md の [ ] マークのファイルが未作成対象。
ディスク照合済みの正確な数字:
- 全目標: 439ファイル（24 Skill合計）
- 作成済み: 135ファイル
- 未作成: 304ファイル

### 優先順位（残り少ない順）
1. windows-application-development: 残り2
2. development-environment-setup: 残り5
3. docker-container-guide: 残り6
4. go-practical-guide: 残り6
5. ai-era-gadgets: 残り7
6. devops-and-github-actions: 残り8
7. design-patterns-guide: 残り9
8. aws-cloud-guide: 残り12
9. regex-and-text-processing: 残り12
10. version-control-and-jujutsu: 残り12
11. algorithm-and-data-structures: 残り14
12. llm-and-ai-comparison: 残り14
13. ai-audio-generation: 残り14
14. ai-visual-generation: 残り14
15. typescript-complete-guide: 残り15
16. ai-automation-and-monetization: 残り15
17. ai-era-development-workflow: 残り15
18. rust-systems-programming: 残り16
19. security-fundamentals: 残り16
20. ai-analysis-guide: 残り16
21. system-design-guide: 残り18
22. sql-and-query-mastery: 残り19
23. custom-ai-agents: 残り19
24. clean-code-principles: 残り20

## Step 3: バックグラウンドTask Agent並列起動

### Agent設計ルール（厳守）
- 1 Agentあたり10-15ファイル（50+は禁止 → コンテキスト上限で途中停止する）
- 同時Agent数は8-10（15以上は禁止 → リソース消費が激しすぎる）
- subagent_type="general-purpose", run_in_background=true
- 確認なし・即実行・YES/NO質問禁止
- プロンプトにWriteツール使用を明示指示
- プロンプトにBashでmkdir -pを明示指示
- ファイルパスは絶対パスで指定

### Agentプロンプト構造
04-AGENT-PROMPTS.md のテンプレートに従い、06-EXACT-TARGET-FILES.md から
[ ] マークのファイルのパスとテーマ説明を抽出してプロンプトに埋め込む。

### 各ガイドファイルの必須要素
- タイトル + 1行概要（> 引用形式）
- 「この章で学ぶこと」3個以上
- コード例5個以上（完全動作するもの）
- ASCII図解/ダイアグラム3個以上
- 比較表2個以上
- アンチパターン2個以上（NG/OKコード付き）
- FAQ 3個以上
- まとめ表
- 次に読むべきガイド
- 参考文献3個以上
- 全て日本語で執筆

## 完了済み（11 Skill — 絶対に触らないでください）
computer-science-fundamentals, operating-system-guide, linux-cli-mastery,
programming-language-fundamentals, object-oriented-programming, async-and-error-handling,
network-fundamentals, browser-and-web-platform, api-and-library-guide,
web-application-development, authentication-and-authorization

## 権限設定（解決済み）
settings.json, settings.local.json, プロジェクト設定の3箇所全てに
Bash(*), Write, Edit, Read を許可済み。Agentの権限エラーは発生しない。
```

---

## 補足（人間向けメモ — 貼り付け不要）

- 上記の ``` 内をコピペするだけで新セッションから完全再開可能
- SESSION_ARCHIVE の6ファイルが全情報を保持している
- MEMORY.md（自動読み込み）にも要点を記録済み
