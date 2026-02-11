#!/bin/bash
# Skills超大作 検証・再開用スクリプト
# 使い方: bash ~/.claude/skills/SESSION_ARCHIVE/session-2026-02-11/05-verify-and-resume.sh

echo "=========================================="
echo "Skills超大作 状態検証レポート"
echo "実行日時: $(date)"
echo "=========================================="
echo ""

# 完了済み11 Skill
COMPLETED_SKILLS=(
  computer-science-fundamentals
  operating-system-guide
  linux-cli-mastery
  programming-language-fundamentals
  object-oriented-programming
  async-and-error-handling
  network-fundamentals
  browser-and-web-platform
  api-and-library-guide
  web-application-development
  authentication-and-authorization
)

# 作業中24 Skill
WORKING_SKILLS=(
  windows-application-development
  development-environment-setup
  docker-container-guide
  aws-cloud-guide
  security-fundamentals
  devops-and-github-actions
  version-control-and-jujutsu
  typescript-complete-guide
  rust-systems-programming
  go-practical-guide
  sql-and-query-mastery
  design-patterns-guide
  system-design-guide
  clean-code-principles
  algorithm-and-data-structures
  regex-and-text-processing
  ai-era-gadgets
  ai-analysis-guide
  ai-audio-generation
  ai-visual-generation
  llm-and-ai-comparison
  custom-ai-agents
  ai-automation-and-monetization
  ai-era-development-workflow
)

BASE="/Users/gaku/.claude/skills"
TOTAL_EXISTING=0
TOTAL_MISSING=0

echo "## 完了済みSkill（11）"
for skill in "${COMPLETED_SKILLS[@]}"; do
  count=$(find "$BASE/$skill/docs" -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "  ✅ $skill: ${count}ファイル"
  TOTAL_EXISTING=$((TOTAL_EXISTING + count))
done

echo ""
echo "## 作業中Skill（24）— 詳細"
echo ""

for skill in "${WORKING_SKILLS[@]}"; do
  skill_md="$BASE/$skill/SKILL.md"
  docs_dir="$BASE/$skill/docs"

  # SKILL.mdから目標ファイルパスを抽出（[[docs/...]] パターン）
  if [ -f "$skill_md" ]; then
    # SKILL.mdから docs/ で始まるリンクを抽出
    target_files=$(grep -oP '\[\[docs/[^\]]+\.md\]\]' "$skill_md" 2>/dev/null | sed 's/\[\[//;s/\]\]//' | sort)
    target_count=$(echo "$target_files" | grep -c "docs/" 2>/dev/null || echo 0)
  else
    target_files=""
    target_count=0
  fi

  # 実際に存在するファイル
  existing_files=$(find "$docs_dir" -name "*.md" -type f 2>/dev/null | sed "s|$BASE/$skill/||" | sort)
  existing_count=$(echo "$existing_files" | grep -c "docs/" 2>/dev/null || echo 0)

  # 未作成ファイルを計算
  if [ "$target_count" -gt 0 ]; then
    missing_files=$(comm -23 <(echo "$target_files") <(echo "$existing_files") 2>/dev/null)
    missing_count=$(echo "$missing_files" | grep -c "docs/" 2>/dev/null || echo 0)
  else
    missing_files="(SKILL.mdからの目標抽出に失敗 — 手動確認必要)"
    missing_count="?"
  fi

  TOTAL_EXISTING=$((TOTAL_EXISTING + existing_count))

  echo "### $skill"
  echo "  SKILL.md目標: ${target_count}ファイル"
  echo "  作成済み: ${existing_count}ファイル"
  echo "  未作成: ${missing_count}ファイル"

  if [ "$missing_count" != "?" ] && [ "$missing_count" -gt 0 ]; then
    echo "  未作成一覧:"
    echo "$missing_files" | while read -r f; do
      [ -n "$f" ] && echo "    - $f"
    done
  fi
  echo ""
done

echo "=========================================="
echo "## サマリー"
echo "  総作成済みファイル: $TOTAL_EXISTING"
echo "  推定未作成ファイル: 約307（正確な数は上記詳細を参照）"
echo "=========================================="
echo ""
echo "## 権限設定チェック"
for f in "$HOME/.claude/settings.json" "$HOME/.claude/settings.local.json" "$HOME/.claude/projects/-Users-gaku/settings.json"; do
  if [ -f "$f" ]; then
    has_write=$(grep -c '"Write"' "$f" 2>/dev/null || echo 0)
    has_bash=$(grep -c '"Bash(\*)"' "$f" 2>/dev/null || echo 0)
    if [ "$has_write" -gt 0 ] && [ "$has_bash" -gt 0 ]; then
      echo "  ✅ $f — Write+Bash(*) 許可あり"
    else
      echo "  ⚠️  $f — 権限不足（Write=$has_write, Bash=$has_bash）"
    fi
  else
    echo "  ❌ $f — ファイルなし"
  fi
done

echo ""
echo "## ディレクトリ構造チェック"
for skill in "${WORKING_SKILLS[@]}"; do
  dirs=$(find "$BASE/$skill/docs" -type d 2>/dev/null | wc -l | tr -d ' ')
  echo "  $skill: ${dirs}ディレクトリ"
done
