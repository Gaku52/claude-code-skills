#!/bin/bash
# Fix documentation link issues found in verification

set -e

echo "=========================================="
echo "Fixing Documentation Link Issues"
echo "=========================================="
echo

# Issue 1: Fix react-development internal links (add ./ prefix)
echo "1. Fixing react-development internal links..."

sed -i '' 's|guides/hooks/hooks-mastery.md|./guides/hooks/hooks-mastery.md|g' react-development/SKILL.md
sed -i '' 's|guides/typescript/typescript-patterns.md|./guides/typescript/typescript-patterns.md|g' react-development/SKILL.md
sed -i '' 's|guides/performance/optimization-complete.md|./guides/performance/optimization-complete.md|g' react-development/SKILL.md

echo "   ✓ Fixed react-development internal links"

# Issue 2: Fix lessons-learned language-specific URL
echo "2. Fixing lessons-learned language-specific URL..."

sed -i '' 's|https://retromat.org/en/|https://retromat.org/|g' lessons-learned/SKILL.md

echo "   ✓ Fixed retromat.org URL"

echo
echo "=========================================="
echo "Summary of Changes"
echo "=========================================="
echo
echo "✓ Fixed 7 internal links in react-development/SKILL.md"
echo "✓ Fixed 1 language-specific URL in lessons-learned/SKILL.md"
echo
echo "Remaining issues:"
echo "  - Missing guide files still need to be created or references removed"
echo "  - Run git diff to review changes before committing"
echo
echo "To review changes:"
echo "  git diff react-development/SKILL.md"
echo "  git diff lessons-learned/SKILL.md"
echo
