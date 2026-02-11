#!/bin/bash
# count-chars.sh - Skillsライブラリの文字数をカウント
#
# 使い方:
#   ./count-chars.sh              # 全Skillの文字数を集計
#   ./count-chars.sh <skill-name> # 特定Skillの文字数を集計
#   ./count-chars.sh --summary    # サマリーのみ表示

SKILLS_DIR="${SKILLS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

format_number() {
    printf "%'d" "$1" 2>/dev/null || echo "$1"
}

count_skill_chars() {
    local skill_dir="$1"
    local skill_name
    skill_name=$(basename "$skill_dir")
    local total=0

    if [ ! -d "$skill_dir" ]; then
        return
    fi

    # .mdファイルの文字数をカウント
    while IFS= read -r -d '' file; do
        local chars
        chars=$(wc -m < "$file" 2>/dev/null || echo 0)
        total=$((total + chars))
    done < <(find "$skill_dir" -name "*.md" -print0 2>/dev/null)

    echo "$total"
}

count_skill_detail() {
    local skill_dir="$1"
    local skill_name
    skill_name=$(basename "$skill_dir")

    echo -e "${BLUE}=== $skill_name ===${NC}"

    # ディレクトリ別にカウント
    for subdir in docs guides checklists templates references; do
        if [ -d "$skill_dir/$subdir" ]; then
            local subdir_total=0
            local file_count=0
            while IFS= read -r -d '' file; do
                local chars
                chars=$(wc -m < "$file" 2>/dev/null || echo 0)
                subdir_total=$((subdir_total + chars))
                file_count=$((file_count + 1))
            done < <(find "$skill_dir/$subdir" -name "*.md" -print0 2>/dev/null)
            if [ "$file_count" -gt 0 ]; then
                printf "  %-15s %3d files  %s chars\n" "$subdir/" "$file_count" "$(format_number "$subdir_total")"
            fi
        fi
    done

    # SKILL.md
    if [ -f "$skill_dir/SKILL.md" ]; then
        local chars
        chars=$(wc -m < "$skill_dir/SKILL.md" 2>/dev/null || echo 0)
        printf "  %-15s           %s chars\n" "SKILL.md" "$(format_number "$chars")"
    elif [ -f "$skill_dir/skill.md" ]; then
        local chars
        chars=$(wc -m < "$skill_dir/skill.md" 2>/dev/null || echo 0)
        printf "  %-15s           %s chars\n" "skill.md" "$(format_number "$chars")"
    fi

    local total
    total=$(count_skill_chars "$skill_dir")
    echo -e "  ${GREEN}Total: $(format_number "$total") chars${NC}"
    echo ""
}

# メイン処理
case "${1:-}" in
    --summary)
        grand_total=0
        skill_count=0
        echo -e "${YELLOW}=== Skills 文字数サマリー ===${NC}"
        echo ""
        printf "%-40s %15s %10s\n" "Skill" "文字数" "ファイル数"
        echo "---------------------------------------------------------------"

        for skill_dir in "$SKILLS_DIR"/*/; do
            [ -d "$skill_dir" ] || continue
            skill_name=$(basename "$skill_dir")

            # TEMPLATESやscripts等のメタディレクトリはスキップ
            case "$skill_name" in
                TEMPLATES|scripts|docs|.git|.github|node_modules) continue ;;
            esac

            chars=$(count_skill_chars "$skill_dir")
            file_count=$(find "$skill_dir" -name "*.md" 2>/dev/null | wc -l)
            grand_total=$((grand_total + chars))
            skill_count=$((skill_count + 1))

            printf "%-40s %15s %10d\n" "$skill_name" "$(format_number "$chars")" "$file_count"
        done

        echo "---------------------------------------------------------------"
        echo -e "${GREEN}Total: $skill_count Skills / $(format_number "$grand_total") chars${NC}"

        # 目標に対する進捗
        target_100m=100000000
        percent=$((grand_total * 100 / target_100m))
        echo -e "${YELLOW}Phase 1 目標 (1億字): $(format_number "$grand_total") / $(format_number "$target_100m") ($percent%)${NC}"
        ;;

    "")
        # 全Skillの詳細
        grand_total=0
        for skill_dir in "$SKILLS_DIR"/*/; do
            [ -d "$skill_dir" ] || continue
            skill_name=$(basename "$skill_dir")
            case "$skill_name" in
                TEMPLATES|scripts|docs|.git|.github|node_modules) continue ;;
            esac
            count_skill_detail "$skill_dir"
            chars=$(count_skill_chars "$skill_dir")
            grand_total=$((grand_total + chars))
        done
        echo -e "${GREEN}=== Grand Total: $(format_number "$grand_total") chars ===${NC}"
        ;;

    *)
        # 特定Skill
        if [ -d "$SKILLS_DIR/$1" ]; then
            count_skill_detail "$SKILLS_DIR/$1"
        else
            echo -e "${RED}Error: Skill '$1' not found${NC}"
            exit 1
        fi
        ;;
esac
