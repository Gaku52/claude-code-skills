# Dependency Optimization & Maintenance - Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Bundle Size Optimization](#bundle-size-optimization)
3. [Tree Shaking and Dead Code Elimination](#tree-shaking-and-dead-code-elimination)
4. [Dependency Update Strategies](#dependency-update-strategies)
5. [Breaking Change Management](#breaking-change-management)
6. [Automated Dependency Updates](#automated-dependency-updates)
7. [Deprecation Handling](#deprecation-handling)
8. [Technical Debt Management](#technical-debt-management)
9. [Migration Guides](#migration-guides)
10. [Performance Monitoring](#performance-monitoring)

## Introduction

### The Cost of Dependencies

Every dependency adds:
- **Bundle Size**: JavaScript to download/parse
- **Build Time**: Compilation and processing
- **Maintenance**: Updates, security patches
- **Risk**: Potential vulnerabilities
- **Complexity**: More code paths to understand

**Real-World Statistics:**
- Average npm project: 1,200 dependencies (85MB node_modules)
- Typical React app bundle: 200-500KB (before optimization)
- Build time increase: 20-30% per 100 dependencies
- Security vulnerability risk increases exponentially with dependency count

### Optimization Goals

1. **Minimize Bundle Size**
   - Target: < 200KB initial bundle (gzipped)
   - Lazy load additional features

2. **Reduce Dependency Count**
   - Audit: Can we build this ourselves?
   - Replace: Smaller alternatives?

3. **Fast Builds**
   - Target: < 30 seconds development builds
   - Target: < 5 minutes production builds

4. **Stay Current**
   - No dependencies > 2 major versions behind
   - Security patches within 1 week

## Bundle Size Optimization

### Analyzing Bundle Size

**webpack-bundle-analyzer:**

```bash
# Install
npm install --save-dev webpack-bundle-analyzer

# Add to webpack.config.js
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

module.exports = {
  plugins: [
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      reportFilename: 'bundle-report.html',
      openAnalyzer: false
    })
  ]
};

# Run build
npm run build

# Open report
open dist/bundle-report.html
```

**Output Analysis:**

```
Total Size: 2.5 MB

Top Offenders:
1. moment.js - 530KB (21%)
   └── Includes ALL locales (not needed)

2. lodash - 280KB (11%)
   └── Importing entire library

3. core-js - 240KB (9%)
   └── Full polyfills (not needed for modern browsers)

4. chart.js - 180KB (7%)
   └── Used in one admin page only
```

### Optimization Techniques

**1. Replace Heavy Dependencies**

```javascript
// ❌ Before: moment.js (530KB)
import moment from 'moment';
const formatted = moment(date).format('YYYY-MM-DD');

// ✅ After: date-fns (13KB for used functions)
import { format } from 'date-fns';
const formatted = format(date, 'yyyy-MM-dd');

// Even better: Native JavaScript
const formatted = new Intl.DateTimeFormat('en-US').format(date);
```

**Comparison:**

| Library | Size | Tree-shakeable | Modern |
|---------|------|----------------|--------|
| moment.js | 530KB | ❌ No | Deprecated |
| date-fns | 13KB (per function) | ✅ Yes | Active |
| dayjs | 7KB | ⚠️ Partial | Active |
| luxon | 65KB | ⚠️ Partial | Active |
| Native Intl | 0KB | ✅ Yes | Built-in |

**2. Selective Imports**

```javascript
// ❌ Bad: Import entire library (280KB)
import _ from 'lodash';
const result = _.debounce(fn, 300);

// ⚠️ Better: Import specific function (24KB)
import debounce from 'lodash/debounce';
const result = debounce(fn, 300);

// ✅ Best: Use lodash-es (tree-shakeable, 2KB)
import { debounce } from 'lodash-es';
const result = debounce(fn, 300);

// 🏆 Optimal: Implement simple utilities yourself
function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
}
```

**3. Code Splitting**

```javascript
// ❌ Load everything upfront
import Chart from 'chart.js';
import AdminPanel from './AdminPanel';

// ✅ Dynamic imports (lazy loading)
const Chart = React.lazy(() => import('chart.js'));
const AdminPanel = React.lazy(() => import('./AdminPanel'));

// Usage
<Suspense fallback={<Loading />}>
  <AdminPanel />
</Suspense>
```

**Next.js Dynamic Imports:**

```javascript
// pages/admin.js
import dynamic from 'next/dynamic';

const AdminDashboard = dynamic(
  () => import('../components/AdminDashboard'),
  {
    loading: () => <p>Loading...</p>,
    ssr: false  // Client-side only
  }
);

export default function AdminPage() {
  return <AdminDashboard />;
}
```

**4. Remove Unused Code**

```bash
# Find unused dependencies
npx depcheck

# Output:
# Unused dependencies
# * lodash
# * moment
# * react-spring

# Remove them
npm uninstall lodash moment react-spring
```

**5. Optimize Images and Assets**

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.(png|jpg|jpeg|gif)$/i,
        type: 'asset',
        parser: {
          dataUrlCondition: {
            maxSize: 10 * 1024  // 10KB inline as base64
          }
        }
      }
    ]
  },
  plugins: [
    new ImageMinimizerPlugin({
      minimizer: {
        implementation: ImageMinimizerPlugin.imageminMinify,
        options: {
          plugins: [
            ['imagemin-mozjpeg', { quality: 80 }],
            ['imagemin-pngquant', { quality: [0.7, 0.8] }]
          ]
        }
      }
    })
  ]
};
```

**6. Use CDN for Common Libraries**

```html
<!-- Instead of bundling React -->
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>

<!-- webpack externals -->
module.exports = {
  externals: {
    react: 'React',
    'react-dom': 'ReactDOM'
  }
};
```

### Bundle Size Budget

**Create bundle-size-budget.json:**

```json
{
  "budgets": [
    {
      "path": "dist/main.*.js",
      "maxSize": "200kb",
      "warning": "180kb"
    },
    {
      "path": "dist/vendor.*.js",
      "maxSize": "500kb",
      "warning": "450kb"
    },
    {
      "path": "dist/*.css",
      "maxSize": "50kb"
    }
  ]
}
```

**Enforce in CI:**

```yaml
# .github/workflows/bundle-size.yml
name: Bundle Size Check

on: [pull_request]

jobs:
  check-size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Check bundle size
        uses: andresz1/size-limit-action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          build_script: build
          skip_step: install

      - name: Report bundle size
        run: |
          SIZE=$(du -sh dist/ | cut -f1)
          echo "Bundle size: $SIZE" >> $GITHUB_STEP_SUMMARY
```

**size-limit configuration:**

```json
{
  "size-limit": [
    {
      "name": "Main bundle",
      "path": "dist/main.*.js",
      "limit": "200 KB"
    },
    {
      "name": "Vendor bundle",
      "path": "dist/vendor.*.js",
      "limit": "500 KB"
    }
  ]
}
```

## Tree Shaking and Dead Code Elimination

### How Tree Shaking Works

**Tree shaking** removes unused code from final bundle.

**Requirements:**
1. ES6 modules (import/export)
2. Production mode
3. No side effects in modules

**Example:**

```javascript
// utils.js
export function add(a, b) { return a + b; }
export function subtract(a, b) { return a - b; }
export function multiply(a, b) { return a * b; }

// app.js
import { add } from './utils';  // Only imports 'add'

console.log(add(1, 2));

// Final bundle: Only includes add(), subtract() and multiply() removed!
```

### Enabling Tree Shaking

**webpack.config.js:**

```javascript
module.exports = {
  mode: 'production',  // Enables optimizations

  optimization: {
    usedExports: true,      // Mark unused exports
    minimize: true,          // Minify
    sideEffects: false,      // Enable aggressive tree shaking

    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,     // Remove console.log
            dead_code: true,        // Remove unreachable code
            pure_funcs: ['console.log', 'console.info']
          }
        }
      })
    ]
  }
};
```

**package.json:**

```json
{
  "sideEffects": false  // All files are pure (no side effects)
}

// Or specify files with side effects
{
  "sideEffects": [
    "*.css",
    "*.scss",
    "./src/polyfills.js"
  ]
}
```

### Common Tree Shaking Issues

**Problem 1: CommonJS Modules**

```javascript
// ❌ CommonJS (not tree-shakeable)
const lodash = require('lodash');
module.exports = { myFunction };

// ✅ ES6 Modules (tree-shakeable)
import { debounce } from 'lodash-es';
export { myFunction };
```

**Problem 2: Side Effects**

```javascript
// ❌ Side effect (prevents tree shaking)
import './polyfill';  // Executes code on import

// ✅ Explicit side effect declaration
// package.json
{
  "sideEffects": ["./src/polyfill.js"]
}
```

**Problem 3: Default Exports**

```javascript
// ❌ Less optimal
export default { add, subtract, multiply };
// Harder to tree shake individual functions

// ✅ Better
export { add, subtract, multiply };
// Can tree shake unused functions
```

### Measuring Tree Shaking Effectiveness

```bash
# Build with stats
npm run build -- --stats

# Analyze
npx webpack-bundle-analyzer dist/stats.json

# Check module concatenation
grep "ModuleConcatenation" dist/stats.json
```

## Dependency Update Strategies

### Update Frequency Matrix

| Dependency Type | Update Frequency | Auto-merge |
|----------------|------------------|------------|
| Security patches | Immediate | ✅ Yes |
| Patch versions | Weekly | ✅ Yes (with tests) |
| Minor versions | Monthly | ⚠️ With review |
| Major versions | Quarterly | ❌ Manual only |
| Dev dependencies | Bi-weekly | ✅ Yes |

### Semantic Versioning Update Rules

```bash
# Patch updates (1.0.x)
npm update                  # Updates to latest patch
npm update lodash          # Update specific package

# Minor updates (1.x.0)
npm update --save           # Updates to latest minor

# Major updates (x.0.0)
npm install lodash@latest   # Manual major updates
```

### Safe Update Process

**Step 1: Check Outdated**

```bash
npm outdated

# Output:
# Package      Current  Wanted  Latest  Location
# react        18.2.0   18.2.0  18.3.1  project
# lodash        4.17.20  4.17.21  4.17.21  project
# typescript    5.0.4    5.0.4   5.3.3   project
```

**Step 2: Review Changelog**

```bash
# Get package info
npm info react versions
npm info react time

# View changelog
npm repo react  # Opens GitHub
# Read CHANGELOG.md or release notes
```

**Step 3: Update in Test Environment**

```bash
# Create update branch
git checkout -b chore/update-dependencies

# Update package.json
npm update --save

# Or use ncu
npx npm-check-updates -u
npm install

# Run tests
npm test
npm run e2e
npm run build
```

**Step 4: Test in Staging**

```bash
# Deploy to staging
./deploy.sh staging

# Run smoke tests
./test-staging.sh

# Monitor for errors
# - Check error tracking (Sentry, etc.)
# - Check performance metrics
# - Check user feedback
```

**Step 5: Deploy to Production**

```bash
# If staging looks good
./deploy.sh production

# Monitor closely for 24-48 hours
# - Error rates
# - Performance
# - User complaints

# Have rollback plan ready
git revert <commit>
./deploy.sh production
```

### Batch Updates vs. Individual Updates

**Batch Updates (Recommended):**

```bash
# Update all dependencies at once
npx npm-check-updates -u
npm install

# Pros:
# - Test interactions between updates
# - Single PR to review
# - Faster overall

# Cons:
# - Harder to isolate issues
# - Bigger change set
```

**Individual Updates:**

```bash
# Update one at a time
npm update react
git commit -am "Update React to 18.3.1"

npm update lodash
git commit -am "Update lodash to 4.17.21"

# Pros:
# - Easy to identify breaking changes
# - Incremental risk

# Cons:
# - Time consuming
# - May miss interaction issues
```

**Recommended Approach:**

```bash
# Batch by risk level

# Batch 1: Low-risk (dev dependencies)
npm update --save-dev

# Batch 2: Medium-risk (patch/minor)
npm update --save

# Batch 3: High-risk (major versions)
npm install react@latest  # One at a time
```

## Breaking Change Management

### Identifying Breaking Changes

**1. Read Release Notes**

```bash
# GitHub releases
npm repo package-name

# npm registry
npm info package-name

# changelog.md
gh repo view owner/repo --web
```

**2. Compare Versions**

```bash
# Git diff between versions
git clone https://github.com/facebook/react
cd react
git diff v18.2.0..v19.0.0 -- packages/react/src

# Or use online tools
# https://diff.intrinsic.com/
```

**3. Check TypeScript Types**

```typescript
// Types often reveal API changes
import { Component } from 'react';

// Before React 18
class MyComponent extends Component {
  render() {
    return <div>{this.props.children}</div>;
  }
}

// After React 18 (children not implicit)
interface Props {
  children: React.ReactNode;  // Must be explicit
}
class MyComponent extends Component<Props> {
  render() {
    return <div>{this.props.children}</div>;
  }
}
```

### Migration Strategies

**Strategy 1: Gradual Migration (Feature Flags)**

```javascript
// config.js
export const USE_NEW_ROUTER = process.env.NEW_ROUTER === 'true';

// app.js
import { USE_NEW_ROUTER } from './config';

if (USE_NEW_ROUTER) {
  const { BrowserRouter } = await import('react-router-dom@6');
  // Use new router
} else {
  const { BrowserRouter } = await import('react-router-dom@5');
  // Use old router
}

// Rollout plan:
// Week 1: 10% of users
// Week 2: 25% of users
// Week 3: 50% of users
// Week 4: 100% of users
```

**Strategy 2: Parallel Run**

```javascript
// Run both old and new implementations
async function fetchData(id) {
  const [oldResult, newResult] = await Promise.all([
    legacyFetch(id),
    newFetch(id)
  ]);

  // Compare results
  if (!deepEqual(oldResult, newResult)) {
    logger.warn('Results differ', { oldResult, newResult });
  }

  // Use old result (safe)
  return oldResult;
}

// Once confident, switch to new
return newResult;
```

**Strategy 3: Adapter Pattern**

```javascript
// Create adapter for backward compatibility
// adapter.js
import { useRouter as useNextRouter } from 'next/router@14';

export function useRouter() {
  const router = useNextRouter();

  // Adapt new API to old API
  return {
    pathname: router.pathname,
    query: router.query,
    push: (url) => router.push(url),
    // Map old methods to new ones
  };
}

// app.js
import { useRouter } from './adapter';  // Use adapter
const router = useRouter();  // Same API as before
```

### Breaking Change Checklist

```markdown
## Major Update Checklist

### Pre-Update
- [ ] Read CHANGELOG.md and migration guide
- [ ] Check GitHub issues for known problems
- [ ] Review breaking changes list
- [ ] Identify affected code (grep/search)
- [ ] Create update branch
- [ ] Notify team of upcoming changes

### Update
- [ ] Update package.json
- [ ] Install dependencies
- [ ] Fix TypeScript errors
- [ ] Fix linting errors
- [ ] Update tests
- [ ] Update documentation

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing in dev
- [ ] Performance testing
- [ ] Accessibility testing

### Deployment
- [ ] Deploy to staging
- [ ] Smoke tests on staging
- [ ] Monitor staging for 24h
- [ ] Deploy to production (canary/blue-green)
- [ ] Monitor production for 48h
- [ ] Update internal documentation

### Rollback Plan
- [ ] Documented rollback steps
- [ ] Tested rollback procedure
- [ ] Team knows how to rollback
```

## Automated Dependency Updates

### Dependabot Configuration

**Advanced .github/dependabot.yml:**

```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "daily"
      time: "03:00"
      timezone: "America/New_York"

    # Separate PRs by dependency type
    groups:
      react-ecosystem:
        patterns:
          - "react*"
          - "@types/react*"
        update-types:
          - "minor"
          - "patch"

      testing-libraries:
        patterns:
          - "@testing-library/*"
          - "jest"
          - "vitest"
        update-types:
          - "minor"
          - "patch"

      build-tools:
        patterns:
          - "webpack"
          - "vite"
          - "@vitejs/*"
        update-types:
          - "minor"

    # Version update strategy
    versioning-strategy: increase

    # PR limits
    open-pull-requests-limit: 10

    # Auto-merge rules (via GitHub Actions)
    labels:
      - "dependencies"
      - "automated"

    # Commit message
    commit-message:
      prefix: "chore(deps)"
      prefix-development: "chore(dev-deps)"
      include: "scope"

    # Ignore specific updates
    ignore:
      - dependency-name: "lodash"
        update-types: ["version-update:semver-major"]

      - dependency-name: "typescript"
        versions: ["5.x"]

    # Allow specific versions
    allow:
      - dependency-type: "direct"
      - dependency-type: "production"

    # Milestone
    milestone: 12
```

### Renovate Advanced Configuration

**renovate.json:**

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",

  "extends": [
    "config:base",
    ":dependencyDashboard",
    ":semanticCommits",
    ":automergePatch",
    ":automergeMinor",
    "group:monorepos"
  ],

  "timezone": "America/New_York",
  "schedule": ["after 10pm every weekday", "every weekend"],

  "labels": ["dependencies", "renovate"],
  "assignees": ["@tech-lead"],

  "packageRules": [
    {
      "description": "Auto-merge patch updates after tests pass",
      "matchUpdateTypes": ["patch"],
      "automerge": true,
      "automergeType": "pr",
      "automergeStrategy": "squash",
      "minimumReleaseAge": "3 days",
      "internalChecksFilter": "strict"
    },

    {
      "description": "Group React ecosystem updates",
      "matchPackagePatterns": ["^react", "^@types/react"],
      "groupName": "React ecosystem",
      "automerge": false,
      "schedule": ["before 9am on monday"]
    },

    {
      "description": "Separate major updates",
      "matchUpdateTypes": ["major"],
      "labels": ["major-update"],
      "automerge": false,
      "dependencyDashboardApproval": true,
      "schedule": ["on the 1st day of the month"]
    },

    {
      "description": "Dev dependencies can be more aggressive",
      "matchDepTypes": ["devDependencies"],
      "automerge": true,
      "automergeType": "branch",
      "schedule": ["before 6am every weekday"]
    },

    {
      "description": "Security updates immediately",
      "matchDatasources": ["npm"],
      "matchUpdateTypes": ["patch"],
      "matchCurrentVersion": "!/^0/",
      "labels": ["security"],
      "prPriority": 10,
      "automerge": true,
      "minimumReleaseAge": "0",
      "schedule": ["at any time"]
    },

    {
      "description": "Pin GitHub Actions to commit SHA",
      "matchManagers": ["github-actions"],
      "pinDigests": true
    },

    {
      "description": "Limit concurrent PRs",
      "matchPackagePatterns": ["*"],
      "prConcurrentLimit": 5,
      "prHourlyLimit": 2
    }
  ],

  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security", "vulnerability"],
    "assignees": ["@security-team"],
    "prPriority": 20,
    "automerge": true
  },

  "lockFileMaintenance": {
    "enabled": true,
    "automerge": true,
    "schedule": ["before 3am on the first day of the month"]
  },

  "postUpdateOptions": [
    "npmDedupe",
    "yarnDedupeHighest"
  ],

  "semanticCommits": "enabled",
  "commitMessagePrefix": "chore(deps):",

  "prBodyTemplate": "{{{header}}}{{{table}}}{{{notes}}}{{{changelogs}}}{{{controls}}}",

  "prCreation": "not-pending",

  "rebaseWhen": "behind-base-branch",

  "platformAutomerge": true,

  "rangeStrategy": "bump",

  "separateMajorMinor": true,
  "separateMultipleMajor": true,

  "stabilityDays": 3,

  "dependencyDashboard": true,
  "dependencyDashboardTitle": "Dependency Dashboard",

  "enabledManagers": [
    "npm",
    "dockerfile",
    "github-actions"
  ]
}
```

### Auto-Merge Workflow

```yaml
# .github/workflows/auto-merge-dependencies.yml
name: Auto-Merge Dependencies

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: write
  pull-requests: write
  checks: read

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: |
      github.actor == 'dependabot[bot]' ||
      github.actor == 'renovate[bot]'

    steps:
      - name: Get PR metadata
        id: metadata
        uses: dependabot/fetch-metadata@v1
        with:
          github-token: "${{ secrets.GITHUB_TOKEN }}"

      - name: Wait for CI checks
        uses: lewagon/wait-on-check-action@v1
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          check-regexp: '(build|test|lint).*'
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          wait-interval: 10
          allowed-conclusions: success

      - name: Auto-merge patch and minor updates
        if: |
          (steps.metadata.outputs.update-type == 'version-update:semver-patch' ||
           steps.metadata.outputs.update-type == 'version-update:semver-minor')
        run: |
          gh pr review --approve "$PR_URL"
          gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Comment on major updates
        if: steps.metadata.outputs.update-type == 'version-update:semver-major'
        run: |
          gh pr comment "$PR_URL" --body "⚠️ Major version update requires manual review. Please check breaking changes."
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Deprecation Handling

### Detecting Deprecations

**1. npm deprecation warnings:**

```bash
npm install

# Warning output:
# npm WARN deprecated request@2.88.2: request has been deprecated
# npm WARN deprecated mkdirp@0.5.5: Legacy versions - use mkdirp 1.x
```

**2. Find all deprecated packages:**

```bash
npm list --depth=0 | grep DEPRECATED
```

**3. Check programmatically:**

```javascript
// scripts/check-deprecated.js
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

async function checkDeprecated() {
  const { stdout } = await execAsync('npm list --json');
  const deps = JSON.parse(stdout);

  function findDeprecated(obj, path = []) {
    const deprecated = [];

    for (const [name, data] of Object.entries(obj.dependencies || {})) {
      if (data.invalid === 'deprecated') {
        deprecated.push({
          name,
          version: data.version,
          path: [...path, name].join(' > '),
          reason: data.problems?.[0] || 'Unknown'
        });
      }

      if (data.dependencies) {
        deprecated.push(...findDeprecated(data, [...path, name]));
      }
    }

    return deprecated;
  }

  const deprecated = findDeprecated(deps);

  if (deprecated.length > 0) {
    console.log('❌ Deprecated packages found:\n');
    deprecated.forEach(({ name, version, path, reason }) => {
      console.log(`  ${name}@${version}`);
      console.log(`    Path: ${path}`);
      console.log(`    Reason: ${reason}\n`);
    });
    process.exit(1);
  }

  console.log('✅ No deprecated packages');
}

checkDeprecated();
```

### Migration from Deprecated Packages

**Common Deprecations and Alternatives:**

```javascript
// 1. request → axios / node-fetch
// ❌ request (deprecated)
const request = require('request');
request('https://api.example.com', (error, response, body) => {
  console.log(body);
});

// ✅ axios
const axios = require('axios');
const { data } = await axios.get('https://api.example.com');

// ✅ native fetch (Node.js 18+)
const response = await fetch('https://api.example.com');
const data = await response.json();


// 2. moment → date-fns / dayjs
// ❌ moment (deprecated)
const moment = require('moment');
const formatted = moment().format('YYYY-MM-DD');

// ✅ date-fns
const { format } = require('date-fns');
const formatted = format(new Date(), 'yyyy-MM-dd');


// 3. mkdirp → fs.promises.mkdir
// ❌ mkdirp (old version)
const mkdirp = require('mkdirp');
await mkdirp('/path/to/dir');

// ✅ Node.js built-in
const fs = require('fs').promises;
await fs.mkdir('/path/to/dir', { recursive: true });


// 4. uuid → crypto.randomUUID()
// ❌ uuid (for simple UUIDs)
const { v4: uuidv4 } = require('uuid');
const id = uuidv4();

// ✅ Node.js built-in (Node.js 14.17+)
const { randomUUID } = require('crypto');
const id = randomUUID();
```

### Codemod Tools

**jscodeshift for automated migrations:**

```javascript
// transforms/migrate-request-to-axios.js
module.exports = function(fileInfo, api) {
  const j = api.jscodeshift;
  const root = j(fileInfo.source);

  // Find request usage
  root.find(j.CallExpression, {
    callee: { name: 'request' }
  }).forEach(path => {
    const [url, callback] = path.value.arguments;

    // Transform to axios
    const axiosCall = j.callExpression(
      j.memberExpression(
        j.identifier('axios'),
        j.identifier('get')
      ),
      [url]
    );

    // Wrap in async/await
    path.replace(
      j.awaitExpression(axiosCall)
    );
  });

  return root.toSource();
};
```

```bash
# Run codemod
npx jscodeshift -t transforms/migrate-request-to-axios.js src/
```

## Technical Debt Management

### Dependency Debt Metrics

**1. Dependency Age:**

```javascript
// scripts/dependency-age.js
const { execSync } = require('child_process');
const deps = require('../package.json').dependencies;

async function analyzeAge() {
  const results = [];

  for (const [name, version] of Object.entries(deps)) {
    const info = JSON.parse(
      execSync(`npm info ${name} --json`).toString()
    );

    const installedVersion = version.replace(/^[^0-9]/, '');
    const latest = info['dist-tags'].latest;
    const publishDate = info.time[installedVersion];

    const ageInDays = Math.floor(
      (Date.now() - new Date(publishDate)) / (1000 * 60 * 60 * 24)
    );

    results.push({
      name,
      installed: installedVersion,
      latest,
      ageInDays,
      outdated: installedVersion !== latest
    });
  }

  // Sort by age
  results.sort((a, b) => b.ageInDays - a.ageInDays);

  console.log('📊 Dependency Age Report\n');
  results.forEach(({ name, installed, latest, ageInDays, outdated }) => {
    const emoji = ageInDays > 365 ? '🔴' : ageInDays > 180 ? '🟡' : '🟢';
    console.log(`${emoji} ${name}`);
    console.log(`   Installed: ${installed} (${ageInDays} days old)`);
    if (outdated) {
      console.log(`   Latest: ${latest}`);
    }
    console.log('');
  });
}

analyzeAge();
```

**2. Dependency Depth:**

```bash
# Maximum dependency depth
npm ls --all --depth=10 | grep "└─" | wc -l
```

**3. Duplicate Dependencies:**

```bash
# Find duplicates
npm dedupe --dry-run

# Or use
npx npm-check-duplicates
```

### Debt Reduction Strategies

**Strategy 1: Consolidation**

```json
{
  "dependencies": {
    // ❌ Before: Multiple similar libraries
    "moment": "^2.29.4",
    "date-fns": "^2.30.0",
    "dayjs": "^1.11.10"

    // ✅ After: One library
    "date-fns": "^2.30.0"
  }
}
```

**Strategy 2: Native Replacement**

```javascript
// ❌ Before: Install library for simple task
import isEqual from 'lodash/isEqual';

// ✅ After: Use built-in (for simple cases)
function isEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}
// Or use built-in structuredClone for deep comparison
```

**Strategy 3: Monorepo for Internal Packages**

```bash
# Instead of separate npm packages
my-app/
  node_modules/
    @company/ui/
    @company/utils/
    @company/api-client/

# Use monorepo
my-monorepo/
  packages/
    ui/
    utils/
    api-client/
  apps/
    web/
    mobile/
```

### Technical Debt Dashboard

```javascript
// scripts/tech-debt-report.js
const fs = require('fs');
const { dependencies, devDependencies } = require('../package.json');

async function generateReport() {
  const report = {
    totalDependencies: Object.keys(dependencies).length,
    totalDevDependencies: Object.keys(devDependencies).length,
    outdated: [],
    deprecated: [],
    duplicates: [],
    largePackages: [],
    metrics: {
      avgAge: 0,
      securityIssues: 0,
      licenseIssues: 0
    }
  };

  // ... analyze dependencies ...

  // Generate HTML report
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Dependency Health Report</title>
      <style>
        body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .metric { display: inline-block; margin: 10px; padding: 20px; background: #f0f0f0; border-radius: 8px; }
        .good { color: green; }
        .warning { color: orange; }
        .danger { color: red; }
      </style>
    </head>
    <body>
      <h1>Dependency Health Report</h1>
      <div class="metrics">
        <div class="metric">
          <h3>Total Dependencies</h3>
          <p class="${report.totalDependencies > 100 ? 'warning' : 'good'}">
            ${report.totalDependencies}
          </p>
        </div>
        <div class="metric">
          <h3>Outdated</h3>
          <p class="${report.outdated.length > 10 ? 'danger' : 'good'}">
            ${report.outdated.length}
          </p>
        </div>
        <div class="metric">
          <h3>Security Issues</h3>
          <p class="${report.metrics.securityIssues > 0 ? 'danger' : 'good'}">
            ${report.metrics.securityIssues}
          </p>
        </div>
      </div>

      <h2>Outdated Dependencies</h2>
      <table>
        ${report.outdated.map(d => `
          <tr>
            <td>${d.name}</td>
            <td>${d.current}</td>
            <td>→</td>
            <td>${d.latest}</td>
          </tr>
        `).join('')}
      </table>
    </body>
    </html>
  `;

  fs.writeFileSync('reports/dependency-health.html', html);
  console.log('✅ Report generated: reports/dependency-health.html');
}

generateReport();
```

## Migration Guides

### React 17 → 18 Migration

```markdown
## React 18 Migration Guide

### Breaking Changes

1. **Automatic Batching**
   - Before: Only batched in event handlers
   - After: Batched everywhere (including promises, setTimeout)

2. **Stricter TypeScript Types**
   - `children` prop no longer implicit
   - Must explicitly declare in interface

3. **Suspense Behavior**
   - `fallback` required
   - Stricter error boundaries

### Migration Steps

#### 1. Update Package
\`\`\`bash
npm install react@18 react-dom@18
npm install -D @types/react@18 @types/react-dom@18
\`\`\`

#### 2. Update Root Rendering
\`\`\`javascript
// Before (React 17)
import ReactDOM from 'react-dom';
ReactDOM.render(<App />, document.getElementById('root'));

// After (React 18)
import { createRoot } from 'react-dom/client';
const root = createRoot(document.getElementById('root'));
root.render(<App />);
\`\`\`

#### 3. Fix TypeScript Errors
\`\`\`typescript
// Before
interface Props {}
const MyComponent: React.FC<Props> = ({ children }) => {
  return <div>{children}</div>;
};

// After
interface Props {
  children: React.ReactNode;
}
const MyComponent: React.FC<Props> = ({ children }) => {
  return <div>{children}</div>;
};
\`\`\`

#### 4. Update Tests
\`\`\`javascript
// Before
import { render } from '@testing-library/react';

// After (React 18 compatible)
import { render } from '@testing-library/react';
// Testing Library updated automatically
\`\`\`

#### 5. Enable Concurrent Features (Optional)
\`\`\`javascript
import { Suspense, lazy } from 'react';
import { useTransition } from 'react';

const LazyComponent = lazy(() => import('./Component'));

function App() {
  const [isPending, startTransition] = useTransition();

  return (
    <Suspense fallback={<Loading />}>
      <LazyComponent />
    </Suspense>
  );
}
\`\`\`

### Testing Checklist
- [ ] All TypeScript errors resolved
- [ ] All tests passing
- [ ] Manual testing in dev
- [ ] No console warnings
- [ ] Performance unchanged or improved
\`\`\`
```

### Next.js 13 → 14 Migration

```markdown
## Next.js 14 Migration Guide

### Key Changes
- Turbopack (beta)
- Server Actions (stable)
- Partial Prerendering (preview)

### Migration Steps

#### 1. Update Next.js
\`\`\`bash
npm install next@14 react@18 react-dom@18
\`\`\`

#### 2. Update next.config.js
\`\`\`javascript
// next.config.js
module.exports = {
  experimental: {
    serverActions: true,  // Now stable, can remove
    turbopack: true,      // Optional: Try Turbopack
  }
};
\`\`\`

#### 3. Migrate Image Component
\`\`\`jsx
// Before
import Image from 'next/image';
<Image src="/pic.png" width={500} height={300} />

// After (same, but optimizations improved)
<Image src="/pic.png" width={500} height={300} alt="Description" />
\`\`\`

#### 4. Update Metadata API
\`\`\`javascript
// app/page.js
export const metadata = {
  title: 'Home',
  description: 'Home page'
};
\`\`\`

#### 5. Server Actions
\`\`\`javascript
// app/actions.js
'use server';

export async function createPost(formData) {
  const title = formData.get('title');
  // Database operation
  await db.post.create({ title });
}

// app/page.js
import { createPost } from './actions';

export default function Page() {
  return (
    <form action={createPost}>
      <input name="title" />
      <button>Submit</button>
    </form>
  );
}
\`\`\`
\`\`\`
```

### Webpack 4 → 5 Migration

```markdown
## Webpack 5 Migration Guide

### Breaking Changes
- Node.js polyfills removed
- New asset modules
- Persistent caching

### Migration

#### 1. Update Webpack
\`\`\`bash
npm install webpack@5 webpack-cli@4
\`\`\`

#### 2. Add Node.js Polyfills (if needed)
\`\`\`bash
npm install node-polyfill-webpack-plugin
\`\`\`

\`\`\`javascript
// webpack.config.js
const NodePolyfillPlugin = require('node-polyfill-webpack-plugin');

module.exports = {
  plugins: [
    new NodePolyfillPlugin()
  ],
  resolve: {
    fallback: {
      fs: false,
      path: require.resolve('path-browserify')
    }
  }
};
\`\`\`

#### 3. Update Asset Handling
\`\`\`javascript
// Before (Webpack 4)
module.exports = {
  module: {
    rules: [
      {
        test: /\.(png|jpg|gif)$/,
        use: ['file-loader']
      }
    ]
  }
};

// After (Webpack 5)
module.exports = {
  module: {
    rules: [
      {
        test: /\.(png|jpg|gif)$/,
        type: 'asset/resource'
      }
    ]
  }
};
\`\`\`

#### 4. Enable Persistent Caching
\`\`\`javascript
module.exports = {
  cache: {
    type: 'filesystem',
    buildDependencies: {
      config: [__filename]
    }
  }
};
\`\`\`
\`\`\`
```

## Performance Monitoring

### Build Time Monitoring

```javascript
// scripts/build-metrics.js
const { performance } = require('perf_hooks');
const fs = require('fs');

const startTime = performance.now();

// Run build
require('child_process').execSync('npm run build', { stdio: 'inherit' });

const endTime = performance.now();
const duration = Math.round(endTime - startTime);

const metrics = {
  timestamp: new Date().toISOString(),
  buildTime: duration,
  nodeVersion: process.version,
  dependencies: Object.keys(require('../package.json').dependencies).length
};

// Append to log
fs.appendFileSync(
  'metrics/build-times.json',
  JSON.stringify(metrics) + '\n'
);

console.log(`✅ Build completed in ${duration}ms`);

// Alert if slow
if (duration > 120000) {  // 2 minutes
  console.warn('⚠️  Build time exceeded threshold!');
}
```

### Bundle Size Tracking

```yaml
# .github/workflows/bundle-size-tracking.yml
name: Track Bundle Size

on:
  push:
    branches: [main]

jobs:
  track:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install and build
        run: |
          npm ci
          npm run build

      - name: Get bundle sizes
        id: sizes
        run: |
          MAIN_SIZE=$(stat -f%z dist/main.*.js)
          VENDOR_SIZE=$(stat -f%z dist/vendor.*.js)
          echo "main=$MAIN_SIZE" >> $GITHUB_OUTPUT
          echo "vendor=$VENDOR_SIZE" >> $GITHUB_OUTPUT

      - name: Post to Slack
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d "{\"text\":\"Bundle Size: Main ${{ steps.sizes.outputs.main }}B, Vendor ${{ steps.sizes.outputs.vendor }}B\"}"

      - name: Save metrics
        run: |
          echo "${{ github.sha }},${{ steps.sizes.outputs.main }},${{ steps.sizes.outputs.vendor }}" >> metrics/bundle-sizes.csv
          git add metrics/bundle-sizes.csv
          git commit -m "Track bundle size"
          git push
```

### Runtime Performance Monitoring

```javascript
// Performance monitoring in production
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

function sendToAnalytics(metric) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    id: metric.id,
    dependencies: DEPENDENCY_COUNT  // Track correlation
  });

  if (navigator.sendBeacon) {
    navigator.sendBeacon('/analytics', body);
  } else {
    fetch('/analytics', { body, method: 'POST', keepalive: true });
  }
}

getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getFCP(sendToAnalytics);
getLCP(sendToAnalytics);
getTTFB(sendToAnalytics);
```

## Conclusion

### Optimization & Maintenance Summary

**Key Takeaways:**

1. **Minimize Bundle Size**
   - Aim for < 200KB initial bundle
   - Use code splitting and lazy loading
   - Replace heavy dependencies with lighter alternatives

2. **Automate Updates**
   - Use Dependabot or Renovate
   - Auto-merge low-risk updates
   - Manual review for major versions

3. **Stay Current**
   - Weekly: Patch updates
   - Monthly: Minor updates
   - Quarterly: Major updates

4. **Monitor Performance**
   - Track build times
   - Monitor bundle sizes
   - Measure runtime performance

5. **Manage Technical Debt**
   - Regular dependency audits
   - Remove deprecated packages
   - Consolidate similar libraries

### Essential Tools Checklist

- [ ] webpack-bundle-analyzer (bundle size)
- [ ] Dependabot or Renovate (auto-updates)
- [ ] npm-check-updates (manual updates)
- [ ] depcheck (unused dependencies)
- [ ] size-limit (bundle budget enforcement)
- [ ] lighthouse (performance auditing)

### Maintenance Schedule

```markdown
## Dependency Maintenance Schedule

### Daily (Automated)
- [ ] Security vulnerability scans
- [ ] Dependabot PR creation
- [ ] CI/CD bundle size checks

### Weekly (Team)
- [ ] Review and merge patch updates
- [ ] Review Dependabot PRs
- [ ] Check deprecated warnings

### Monthly (Team)
- [ ] Minor version updates
- [ ] Dependency health report
- [ ] Remove unused dependencies
- [ ] Bundle size optimization review

### Quarterly (Team)
- [ ] Major version update planning
- [ ] Technical debt assessment
- [ ] Migration guide reviews
- [ ] Tooling updates (webpack, build tools)
- [ ] License compliance audit

### Annually (Leadership)
- [ ] Dependency strategy review
- [ ] Monorepo evaluation
- [ ] Build tool modernization
- [ ] Team training on new practices
```

By following these comprehensive optimization and maintenance practices, your projects will remain performant, secure, and maintainable for years to come!
