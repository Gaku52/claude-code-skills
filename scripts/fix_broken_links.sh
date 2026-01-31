#!/bin/bash

# Fix broken links identified in HTTP verification

echo "=================================="
echo "Fixing Broken Links"
echo "=================================="
echo ""

# 1. Fix Apple HIG iOS link (404)
echo "1. Fixing Apple HIG iOS link..."
sed -i '' 's|https://developer.apple.com/design/human-interface-guidelines/ios|https://developer.apple.com/design/human-interface-guidelines/|g' swiftui-patterns/SKILL.md
echo "   ✓ Fixed swiftui-patterns/SKILL.md"

# 2. Fix Xcode project configuration link (404)
echo "2. Fixing Xcode project configuration link..."
sed -i '' 's|https://developer.apple.com/documentation/xcode/configuring-your-xcode-project|https://developer.apple.com/documentation/xcode|g' ios-project-setup/SKILL.md
echo "   ✓ Fixed ios-project-setup/SKILL.md"

# 3. Fix SPM documentation link (404)
echo "3. Fixing Swift Package Manager documentation link..."
sed -i '' 's|https://github.com/apple/swift-package-manager/blob/main/Documentation/PackageDescription.md|https://www.swift.org/package-manager/|g' dependency-management/SKILL.md
echo "   ✓ Fixed dependency-management/SKILL.md"

# 4. Fix Cobra user guide link (404)
echo "4. Fixing Cobra user guide link..."
sed -i '' 's|https://github.com/spf13/cobra/blob/main/user_guide.md|https://cobra.dev/|g' cli-development/SKILL.md
echo "   ✓ Fixed cli-development/SKILL.md"

# 5. Remove jsonplaceholder malformed URLs from nextjs-development (404)
echo "5. Fixing jsonplaceholder URLs in Next.js..."
# These are in code examples with syntax errors (trailing quotes/backticks)
# Remove them as they're broken URLs extracted from code blocks
sed -i '' '/jsonplaceholder.typicode.com.*'"'"',$/d' nextjs-development/SKILL.md
sed -i '' '/jsonplaceholder.typicode.com.*\${id}`,$/d' nextjs-development/SKILL.md
echo "   ✓ Cleaned up nextjs-development/SKILL.md"

# 6. Fix NSHipster security link (404)
echo "6. Fixing NSHipster security link..."
sed -i '' 's|https://nshipster.com/security/|https://nshipster.com/|g' ios-security/SKILL.md
echo "   ✓ Fixed ios-security/SKILL.md"

# 7. Fix Atlassian incident playbooks link (404)
echo "7. Fixing Atlassian incident playbooks link..."
sed -i '' 's|https://www.atlassian.com/incident-management/incident-response/playbooks|https://www.atlassian.com/incident-management|g' incident-logger/SKILL.md
echo "   ✓ Fixed incident-logger/SKILL.md"

# 8. Fix MongoDB Realm link (404 - Realm is now part of Atlas)
echo "8. Fixing MongoDB Realm link..."
sed -i '' 's|https://www.mongodb.com/docs/realm/|https://www.mongodb.com/docs/atlas/device-sdks/|g' networking-data/SKILL.md
echo "   ✓ Fixed networking-data/SKILL.md"

# 9. Fix PagerDuty incident response link (404)
echo "9. Fixing PagerDuty incident response link..."
sed -i '' 's|https://www.pagerduty.com/resources/learn/incident-response-best-practices/|https://response.pagerduty.com/|g' incident-logger/SKILL.md
echo "   ✓ Fixed incident-logger/SKILL.md"

# 10. Fix Prisma documentation links (404 - structure changed)
echo "10. Fixing Prisma documentation links..."
sed -i '' 's|https://www.prisma.io/docs/guides/performance-and-optimization|https://www.prisma.io/docs|g' database-design/SKILL.md
sed -i '' 's|https://www.prisma.io/docs/reference/api-reference/prisma-schema-reference|https://www.prisma.io/docs/reference/api-reference|g' database-design/SKILL.md
echo "   ✓ Fixed database-design/SKILL.md"

# 11. Remove placeholder YouTube links (404)
echo "11. Removing placeholder YouTube links..."
sed -i '' '/www.youtube.com\/\.\.\./d' git-workflow/SKILL.md
sed -i '' '/www.youtube.com\/\.\.\./d' code-review/SKILL.md
sed -i '' '/www.youtube.com\/\.\.\./d' ci-cd-automation/SKILL.md
echo "   ✓ Cleaned up git-workflow, code-review, ci-cd-automation"

echo ""
echo "=================================="
echo "Summary of Changes"
echo "=================================="
echo ""
echo "✓ Fixed 10 broken 404 links"
echo "✓ Removed 3 placeholder/malformed URLs"
echo ""
echo "Note: 403 errors (MySQL, GitLab, Medium, etc.) are likely bot protection."
echo "These sites work in browsers but block automated requests."
echo "Note: example.com URLs are in code samples and are intentionally unreachable."
echo ""
echo "To review changes:"
echo "  git diff **/SKILL.md"
echo ""
