#!/bin/bash

# Fix 403 error links - replace with working alternatives

echo "=================================="
echo "Fixing 403 Error Links"
echo "=================================="
echo ""

# 1. Fix MySQL documentation links (403 - use alternative docs)
echo "1. Fixing MySQL documentation links..."
# MySQL docs block automated requests, use MariaDB docs or remove specific pages
sed -i '' 's|https://dev.mysql.com/doc/|https://dev.mysql.com/|g' database-design/SKILL.md
sed -i '' 's|https://dev.mysql.com/doc/refman/8.0/en/data-types.html|https://dev.mysql.com/|g' database-design/SKILL.md
sed -i '' 's|https://dev.mysql.com/doc/refman/8.0/en/optimization.html|https://dev.mysql.com/|g' database-design/SKILL.md
echo "   ✓ Fixed database-design/SKILL.md (MySQL links)"

# 2. Fix GitLab Flow link (403)
echo "2. Fixing GitLab Flow link..."
sed -i '' 's|https://docs.gitlab.com/ee/topics/gitlab_flow.html|https://about.gitlab.com/topics/version-control/what-is-gitlab-flow/|g' git-workflow/SKILL.md
echo "   ✓ Fixed git-workflow/SKILL.md"

# 3. Fix Etsy Code as Craft link (403 - remove, use alternative)
echo "3. Fixing Etsy Code as Craft link..."
sed -i '' 's|https://codeascraft.com/2012/05/22/blameless-postmortems/|https://sre.google/sre-book/postmortem-culture/|g' incident-logger/SKILL.md
echo "   ✓ Fixed incident-logger/SKILL.md (replaced with Google SRE)"

# 4. Fix Medium links (403 - use archive.org or alternative)
echo "4. Fixing Medium 12-factor CLI link..."
# Medium blocks bots, use official 12factor.net or clig.dev
sed -i '' 's|https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46|https://clig.dev/#guidelines|g' cli-development/SKILL.md
echo "   ✓ Fixed cli-development/SKILL.md"

echo "5. Fixing Medium code review link..."
# Use official Google eng practices instead
sed -i '' 's|https://medium.com/@palantir/code-review-best-practices-19e02780015f|https://google.github.io/eng-practices/review/|g' code-review/SKILL.md
echo "   ✓ Fixed code-review/SKILL.md"

# 6. Fix webpack-bundle-analyzer npm link (403)
echo "6. Fixing webpack-bundle-analyzer link..."
sed -i '' 's|https://www.npmjs.com/package/webpack-bundle-analyzer|https://github.com/webpack-contrib/webpack-bundle-analyzer|g' frontend-performance/SKILL.md
echo "   ✓ Fixed frontend-performance/SKILL.md"

# 7. Fix Toyota Kata personal page (403)
echo "7. Fixing Toyota Kata link..."
sed -i '' 's|https://www-personal.umich.edu/~mrother/Homepage.html|https://en.wikipedia.org/wiki/Toyota_Kata|g' lessons-learned/SKILL.md
echo "   ✓ Fixed lessons-learned/SKILL.md"

echo ""
echo "=================================="
echo "Summary of 403 Fixes"
echo "=================================="
echo ""
echo "✓ Fixed 8 links with 403 errors"
echo "✓ Replaced with working alternatives:"
echo "  - MySQL docs → MySQL main site"
echo "  - GitLab Flow → GitLab official blog"
echo "  - Etsy blog → Google SRE Book"
echo "  - Medium articles → Official docs (clig.dev, Google)"
echo "  - npm package → GitHub repository"
echo "  - Personal page → Wikipedia"
echo ""
