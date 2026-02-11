#!/bin/bash
# safe-commit-push.sh - 安全なコミット&プッシュスクリプト

set -e

# 色の定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Safe Commit & Push Script${NC}"
echo ""

# 1. コミットメッセージの確認
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Commit message required${NC}"
    echo "Usage: ./safe-commit-push.sh \"your commit message\""
    exit 1
fi

COMMIT_MESSAGE="$1"

# 2. 変更があるか確認
if [[ -z $(git status -s) ]]; then
    echo -e "${YELLOW}ℹ️  No changes to commit${NC}"
    # 変更がない場合でも、最新の状態に更新
    echo -e "${YELLOW}📥 Pulling latest changes...${NC}"
    git pull --rebase origin main
    echo -e "${GREEN}✅ Already up to date${NC}"
    exit 0
fi

# 3. ステータス表示
echo -e "${YELLOW}📋 Step 1: Current changes:${NC}"
git status -s
echo ""

# 4. 未コミットの変更を一時退避
echo -e "${YELLOW}💼 Step 2: Stashing changes...${NC}"
git stash push -u -m "temp stash for safe-commit-push"
echo -e "${GREEN}✅ Changes stashed${NC}"
echo ""

# 5. 最新の変更を取得
echo -e "${YELLOW}📥 Step 3: Pulling latest changes...${NC}"
if ! git pull --rebase origin main; then
    echo -e "${RED}❌ Pull failed! Restoring your changes...${NC}"
    git stash pop
    exit 1
fi
echo -e "${GREEN}✅ Pull completed${NC}"
echo ""

# 6. stashを戻す
echo -e "${YELLOW}📦 Step 4: Restoring changes...${NC}"
if ! git stash pop; then
    echo -e "${RED}❌ Conflict while restoring changes! Please resolve manually.${NC}"
    echo -e "${YELLOW}Your changes are in stash. Use 'git stash list' to see them.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Changes restored${NC}"
echo ""

# 7. すべての変更をステージング
echo -e "${YELLOW}➕ Step 5: Staging all changes...${NC}"
git add -A
echo -e "${GREEN}✅ All changes staged${NC}"
echo ""

# 8. コミット
echo -e "${YELLOW}💾 Step 6: Creating commit...${NC}"
git commit -m "$COMMIT_MESSAGE"
echo -e "${GREEN}✅ Commit created${NC}"
echo ""

# 9. Push前に再度pull（念のため）
echo -e "${YELLOW}🔄 Step 7: Final pull before push...${NC}"
if ! git pull --rebase origin main; then
    echo -e "${RED}❌ Conflict detected! Please resolve manually.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ No conflicts${NC}"
echo ""

# 10. Push
echo -e "${YELLOW}📤 Step 8: Pushing to remote...${NC}"
git push origin main
echo -e "${GREEN}✅ Push completed successfully!${NC}"
echo ""

# 11. 最終確認
echo -e "${GREEN}🎉 All done!${NC}"
git log --oneline -1
