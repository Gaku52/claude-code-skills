#!/bin/bash
# check-references.sh - 相互参照リンクの有効性をチェック
#
# 使い方:
#   ./check-references.sh              # 全Skillのリンクチェック
#   ./check-references.sh <skill-name> # 特定Skillのリンクチェック

SKILLS_DIR="${SKILLS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

broken_count=0
total_count=0
checked_files=0

check_file() {
    local file="$1"
    local file_dir
    file_dir=$(dirname "$file")

    # Markdown内部リンクを抽出: [text](path) 形式
    while IFS= read -r line; do
        # URL（http/https）はスキップ
        echo "$line" | grep -oP '\[([^\]]*)\]\((?!https?://)([^)]+)\)' | while read -r match; do
            local link_path
            link_path=$(echo "$match" | grep -oP '\(([^)]+)\)' | tr -d '()')

            # アンカーリンク（#で始まる）はスキップ
            if [[ "$link_path" == \#* ]]; then
                continue
            fi

            # フラグメントを除去
            local clean_path="${link_path%%#*}"

            total_count=$((total_count + 1))

            # 相対パスを絶対パスに変換
            local abs_path
            if [[ "$clean_path" == /* ]]; then
                abs_path="$clean_path"
            else
                abs_path="$file_dir/$clean_path"
            fi

            # ファイルの存在チェック
            if [ ! -f "$abs_path" ] && [ ! -d "$abs_path" ]; then
                echo -e "  ${RED}BROKEN${NC}: $file"
                echo "    → $link_path"
                broken_count=$((broken_count + 1))
            fi
        done
    done < "$file"

    # Skill間参照を抽出: [[skill-name/path]] 形式
    grep -oP '\[\[([^\]|]+)(\|[^\]]+)?\]\]' "$file" 2>/dev/null | while read -r match; do
        local ref_path
        ref_path=$(echo "$match" | tr -d '[]' | cut -d'|' -f1)

        total_count=$((total_count + 1))

        # Skills ディレクトリからの相対パス
        local abs_path="$SKILLS_DIR/$ref_path"

        if [ ! -f "$abs_path" ] && [ ! -d "$abs_path" ]; then
            # SKILL.mdまたはskill.md があるかチェック
            if [ -f "$abs_path/SKILL.md" ] || [ -f "$abs_path/skill.md" ]; then
                continue
            fi
            echo -e "  ${RED}BROKEN${NC}: $file"
            echo "    → [[$ref_path]]"
            broken_count=$((broken_count + 1))
        fi
    done

    checked_files=$((checked_files + 1))
}

# メイン処理
target_dir="$SKILLS_DIR"
if [ -n "${1:-}" ] && [ -d "$SKILLS_DIR/$1" ]; then
    target_dir="$SKILLS_DIR/$1"
    echo -e "${YELLOW}Checking references in: $1${NC}"
else
    echo -e "${YELLOW}Checking all references...${NC}"
fi

echo ""

while IFS= read -r -d '' file; do
    check_file "$file"
done < <(find "$target_dir" -name "*.md" -not -path "*/.git/*" -not -path "*/node_modules/*" -print0 2>/dev/null)

echo ""
echo "---------------------------------------"
echo -e "Checked: $checked_files files"
echo -e "Links found: $total_count"
if [ "$broken_count" -eq 0 ]; then
    echo -e "${GREEN}Broken links: 0${NC}"
else
    echo -e "${RED}Broken links: $broken_count${NC}"
fi
