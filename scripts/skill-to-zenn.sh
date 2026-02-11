#!/bin/bash
# skill-to-zenn.sh - SkillをZenn Book形式に変換
#
# 使い方:
#   ./skill-to-zenn.sh <skill-name>           # 特定Skillを変換
#   ./skill-to-zenn.sh <skill-name> <出力先>   # 出力先を指定
#   ./skill-to-zenn.sh --all                  # 全Skillを変換

SKILLS_DIR="${SKILLS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_BASE="${2:-$HOME/zenn-books}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

convert_skill() {
    local skill_name="$1"
    local skill_dir="$SKILLS_DIR/$skill_name"
    local output_dir="$OUTPUT_BASE/$skill_name"

    if [ ! -d "$skill_dir" ]; then
        echo -e "${RED}Error: Skill '$skill_name' not found${NC}"
        return 1
    fi

    echo -e "${YELLOW}Converting: $skill_name${NC}"

    # 出力ディレクトリ作成
    mkdir -p "$output_dir/chapters"

    # SKILL.mdまたはskill.mdから説明を取得
    local skill_md=""
    if [ -f "$skill_dir/SKILL.md" ]; then
        skill_md="$skill_dir/SKILL.md"
    elif [ -f "$skill_dir/skill.md" ]; then
        skill_md="$skill_dir/skill.md"
    fi

    local description=""
    if [ -n "$skill_md" ]; then
        description=$(grep -A1 "^description:" "$skill_md" 2>/dev/null | head -1 | sed 's/^description: //')
    fi

    # config.yaml 生成
    cat > "$output_dir/config.yaml" << YAML
title: "$skill_name"
summary: "$description"
topics: ["programming", "engineering"]
published: true
price: 0
chapters:
YAML

    # docs/配下のファイルをフラット化して変換
    local chapter_num=0
    local docs_dir="$skill_dir/docs"
    local guides_dir="$skill_dir/guides"

    # docs/があればdocsから、なければguides/から
    local source_dir=""
    if [ -d "$docs_dir" ]; then
        source_dir="$docs_dir"
    elif [ -d "$guides_dir" ]; then
        source_dir="$guides_dir"
    fi

    if [ -n "$source_dir" ] && [ -d "$source_dir" ]; then
        # ファイルをソートして処理
        find "$source_dir" -name "*.md" -type f | sort | while read -r file; do
            chapter_num=$((chapter_num + 1))
            local chapter_id
            chapter_id=$(printf "%03d" "$chapter_num")

            # ファイル名からタイトルを取得（最初のH1）
            local title
            title=$(grep -m1 "^# " "$file" | sed 's/^# //')
            if [ -z "$title" ]; then
                title=$(basename "$file" .md)
            fi

            # slug生成（ファイル名ベース）
            local slug
            slug=$(basename "$file" .md | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g')

            # Zenn chapter形式で出力
            local chapter_file="$output_dir/chapters/$slug.md"
            {
                echo "---"
                echo "title: \"$title\""
                echo "---"
                echo ""
                # 内容をコピー（最初のH1は除去）
                tail -n +2 "$file" | sed 's/^# .*//'
            } > "$chapter_file"

            # config.yamlにチャプター追加
            echo "  - $slug" >> "$output_dir/config.yaml"
        done
    fi

    echo -e "${GREEN}Done: $output_dir${NC}"
    local file_count
    file_count=$(find "$output_dir/chapters" -name "*.md" 2>/dev/null | wc -l)
    echo "  Chapters: $file_count"
}

# メイン処理
case "${1:-}" in
    --all)
        echo -e "${YELLOW}Converting all Skills to Zenn format...${NC}"
        echo ""
        for skill_dir in "$SKILLS_DIR"/*/; do
            [ -d "$skill_dir" ] || continue
            skill_name=$(basename "$skill_dir")
            case "$skill_name" in
                TEMPLATES|scripts|docs|.git|.github|node_modules) continue ;;
            esac
            convert_skill "$skill_name"
            echo ""
        done
        echo -e "${GREEN}All conversions complete.${NC}"
        echo "Output: $OUTPUT_BASE"
        ;;
    "")
        echo "Usage: $0 <skill-name> [output-dir]"
        echo "       $0 --all"
        exit 1
        ;;
    *)
        convert_skill "$1"
        ;;
esac
