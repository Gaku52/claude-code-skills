# 🎯 良質なリポジトリ選定基準

> GitHub大規模分析における品質保証フレームワーク
> 目標: 1000件 → 厳選50-100件の高品質リポジトリ

---

## 📊 選定プロセス

```
Phase 1: 自動フィルタリング (1000 → 200)
  ↓
Phase 2: 品質スコアリング (200 → 100)
  ↓
Phase 3: 手動検証 (100 → 50-80)
  ↓
最終データセット: 50-80の高品質プロジェクト
```

---

## 🔍 Phase 1: 自動フィルタリング基準

### 必須条件 (AND条件)

#### 1. **プロダクション品質の証拠**

```javascript
const productionQualityCriteria = {
  // 1.1 活発な開発
  recentActivity: {
    lastCommitWithin: '3 months',  // 3ヶ月以内のコミット
    minimumCommits: 100,            // 総コミット数 100+
    activeContributors: 3           // アクティブ貢献者 3人以上
  },

  // 1.2 コミュニティの信頼
  community: {
    stars: 500,                     // Star 500+ (1000は緩すぎる)
    forks: 50,                      // Fork 50+
    watchers: 20,                   // Watcher 20+
    openIssues: { min: 5, max: 500 } // 5-500件 (多すぎず少なすぎず)
  },

  // 1.3 メンテナンス状態
  maintenance: {
    hasActiveIssueResponse: true,   // Issue返信平均 < 7日
    hasRecentRelease: true,         // 6ヶ月以内のリリース
    dependenciesUpToDate: true      // 依存関係が最新
  },

  // 1.4 プロジェクト規模
  codebase: {
    linesOfCode: { min: 1000, max: 500000 }, // 1K-500K LOC
    numberOfFiles: { min: 10, max: 5000 },   // 10-5000ファイル
    hasMultipleDirectories: true             // src/, tests/ など構造化
  }
};
```

#### 2. **本番利用の証拠**

```javascript
const productionUsageCriteria = {
  // 2.1 公開デプロイ
  deployment: {
    hasLiveURL: true,               // 稼働中のURL
    hasDocsWebsite: true,           // ドキュメントサイト
    hasCDNUsage: false              // CDN利用 (optional)
  },

  // 2.2 企業/組織の使用
  organization: {
    isOrgRepo: true,                // 組織リポジトリ
    hasSponsors: false,             // スポンサーあり (optional)
    inAwesomeList: false            // Awesome系リスト掲載 (optional)
  },

  // 2.3 npm/その他パッケージ公開
  packagePublished: {
    onNPM: true,                    // npm公開
    weeklyDownloads: 1000,          // 週1000DL以上
    hasMultipleVersions: true       // 複数バージョンリリース
  }
};
```

#### 3. **品質管理の証拠**

```javascript
const qualityAssuranceCriteria = {
  // 3.1 テスト
  testing: {
    hasTests: true,                 // テストディレクトリ存在
    testCoverage: 50,               // カバレッジ 50%以上
    ciConfigured: true,             // CI設定あり (.github/workflows)
    ciPassRate: 90                  // CI成功率 90%以上
  },

  // 3.2 ドキュメント
  documentation: {
    hasREADME: true,                // README.md (必須)
    readmeLength: 500,              // README 500文字以上
    hasCONTRIBUTING: true,         // CONTRIBUTING.md
    hasChangelog: true,             // CHANGELOG.md
    hasLicense: true                // LICENSE
  },

  // 3.3 コード品質
  codeQuality: {
    hasLinter: true,                // ESLint/Prettier設定
    hasTypeScript: true,            // TypeScript使用
    tsConfigStrict: true,           // tsconfig.json strict mode
    hasPreCommitHooks: false        // pre-commit hooks (optional)
  }
};
```

#### 4. **フレームワーク固有の基準**

```javascript
const frameworkSpecificCriteria = {
  react: {
    version: '>=18.0.0',            // React 18以降
    hasComponents: true,            // src/components/ 存在
    notTutorial: true,              // "tutorial", "example" を含まない
    notBoilerplate: true            // "boilerplate", "starter" を含まない
  },

  nextjs: {
    version: '>=13.0.0',            // Next.js 13以降 (App Router)
    hasAppDirectory: true,          // app/ ディレクトリ
    hasServerComponents: true,      // Server Components使用
    notTemplate: true               // テンプレートではない
  }
};
```

---

## 📈 Phase 2: 品質スコアリング

### 総合品質スコア (100点満点)

```javascript
function calculateQualityScore(repo) {
  const scores = {
    // 1. 開発活動 (25点)
    development: {
      commitFrequency: scoreCommitFrequency(repo),      // 10点
      contributorDiversity: scoreContributors(repo),    // 10点
      issueResponseTime: scoreIssueResponse(repo)       // 5点
    },

    // 2. コード品質 (25点)
    codeQuality: {
      testCoverage: scoreTestCoverage(repo),            // 10点
      codeComplexity: scoreComplexity(repo),            // 5点
      typeScriptUsage: scoreTypeScript(repo),           // 5点
      linterConfig: scoreLinter(repo)                   // 5点
    },

    // 3. ドキュメント (20点)
    documentation: {
      readmeQuality: scoreREADME(repo),                 // 10点
      apiDocs: scoreAPIDocumentation(repo),             // 5点
      examples: scoreExamples(repo)                     // 5点
    },

    // 4. コミュニティ (15点)
    community: {
      stars: scoreStars(repo),                          // 5点
      forks: scoreForks(repo),                          // 5点
      discussions: scoreDiscussions(repo)               // 5点
    },

    // 5. 本番利用 (15点)
    production: {
      hasLiveDeployment: scoreLiveURL(repo),            // 5点
      npmDownloads: scoreNPMDownloads(repo),            // 5点
      dependentsCount: scoreDependents(repo)            // 5点
    }
  };

  return Object.values(scores).reduce((total, category) => {
    return total + Object.values(category).reduce((sum, score) => sum + score, 0);
  }, 0);
}

// 詳細スコアリング関数
function scoreCommitFrequency(repo) {
  const commitsLastMonth = repo.commits.filter(
    c => new Date(c.date) > new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
  ).length;

  if (commitsLastMonth >= 50) return 10;
  if (commitsLastMonth >= 20) return 8;
  if (commitsLastMonth >= 10) return 6;
  if (commitsLastMonth >= 5) return 4;
  return 2;
}

function scoreTestCoverage(repo) {
  if (!repo.coverage) return 0;

  if (repo.coverage >= 80) return 10;
  if (repo.coverage >= 70) return 8;
  if (repo.coverage >= 60) return 6;
  if (repo.coverage >= 50) return 4;
  return 2;
}

function scoreComplexity(repo) {
  const avgComplexity = repo.cyclomaticComplexity / repo.functionsCount;

  if (avgComplexity <= 5) return 5;  // 優秀
  if (avgComplexity <= 10) return 4; // 良好
  if (avgComplexity <= 15) return 3; // 許容範囲
  if (avgComplexity <= 20) return 2; // 要改善
  return 1;                          // 複雑すぎ
}

// ... 他のスコアリング関数
```

### スコアカットオフ

```javascript
const qualityThresholds = {
  excellent: 80,  // 80点以上: 優秀 → 必ず含める
  good: 65,       // 65-79点: 良好 → 含める
  acceptable: 50, // 50-64点: 許容 → 慎重に検討
  poor: 50        // 50点未満: 除外
};

// Phase 2の結果
// 200リポジトリ → 100リポジトリ (50点以上)
```

---

## 🔬 Phase 3: 手動検証

### 人間によるレビュー (100 → 50-80)

#### チェック項目

```markdown
## 各リポジトリの手動レビューチェックリスト

### A. コードレビュー (サンプル確認)
- [ ] src/ のコード品質を目視確認 (10-20ファイル)
- [ ] 適切な設計パターンの使用
- [ ] コメント・ドキュメントの質
- [ ] テストコードの存在と品質

### B. プロジェクト目的の確認
- [ ] チュートリアル・学習用ではないか?
- [ ] 実際のプロダクトか?
- [ ] メンテナンスが継続されているか?

### C. 技術スタックの妥当性
- [ ] 対象フレームワーク (React/Next.js) の正しい使用
- [ ] 依存関係が適切か (過度に多くないか)
- [ ] ベストプラクティスに準拠しているか

### D. 除外基準の最終確認
- [ ] **Template/Boilerplate**: テンプレートではない
- [ ] **Abandoned**: 放置プロジェクトではない
- [ ] **Tutorial**: 学習用ではない
- [ ] **Fork**: 単なるForkではない
- [ ] **Monorepo subset**: モノレポの一部ではない
```

#### 除外パターン

```javascript
const excludePatterns = {
  namePatterns: [
    /template/i,
    /boilerplate/i,
    /starter/i,
    /example/i,
    /tutorial/i,
    /demo/i,
    /playground/i,
    /learning/i,
    /practice/i,
    /sample/i
  ],

  descriptionPatterns: [
    /getting started/i,
    /learn react/i,
    /教材/,
    /サンプル/,
    /練習/
  ],

  readmeIndicators: [
    /this is a template/i,
    /fork this repository/i,
    /学習用/,
    /初心者向け/
  ]
};

function isHighQualityProduction(repo) {
  // 名前チェック
  if (excludePatterns.namePatterns.some(p => p.test(repo.name))) {
    return false;
  }

  // 説明チェック
  if (excludePatterns.descriptionPatterns.some(p => p.test(repo.description))) {
    return false;
  }

  // README内容チェック
  if (excludePatterns.readmeIndicators.some(p => p.test(repo.readme))) {
    return false;
  }

  return true;
}
```

---

## 🎯 最終データセット構成

### 目標: 50-80の厳選リポジトリ

```javascript
const finalDataset = {
  // カテゴリ分け
  categories: {
    'E-Commerce': 10,           // EC サイト
    'SaaS Products': 10,        // SaaS製品
    'Content Platforms': 10,    // ブログ、メディア
    'Developer Tools': 10,      // 開発ツール
    'Data Visualization': 5,    // データ可視化
    'Social/Community': 5,      // SNS、コミュニティ
    'Corporate Websites': 5,    // 企業サイト
    'Open Source Projects': 10  // OSSプロジェクト
  },

  // 規模の分散
  sizeDistribution: {
    'Small (1K-10K LOC)': 15,
    'Medium (10K-50K LOC)': 25,
    'Large (50K-500K LOC)': 20
  },

  // フレームワークの分散
  frameworkDistribution: {
    'React (CRA)': 15,
    'React (Vite)': 10,
    'Next.js Pages Router': 10,
    'Next.js App Router': 15,
    'Remix': 5,
    'Gatsby': 5
  },

  // 地域の分散
  regionDistribution: {
    'US': 20,
    'Europe': 15,
    'Asia': 10,
    'Other': 5
  }
};

// 合計: 50-65リポジトリ (多様性を確保)
```

---

## 💻 実装例

### 自動選定スクリプト

```typescript
// src/selectors/quality-filter.ts

interface Repo {
  name: string;
  description: string;
  stars: number;
  forks: number;
  // ... その他のメトリクス
}

class QualityRepoSelector {
  async selectHighQualityRepos(): Promise<Repo[]> {
    // Phase 1: 自動フィルタリング
    const phase1 = await this.phase1AutoFilter();
    console.log(`Phase 1: ${phase1.length} repositories`);

    // Phase 2: 品質スコアリング
    const phase2 = await this.phase2QualityScoring(phase1);
    console.log(`Phase 2: ${phase2.length} repositories`);

    // Phase 3: 手動検証用リスト生成
    await this.generateManualReviewList(phase2);

    return phase2;
  }

  private async phase1AutoFilter(): Promise<Repo[]> {
    const allRepos = await this.fetchFromGitHub();

    return allRepos.filter(repo => {
      // 必須条件チェック
      return (
        this.checkProductionQuality(repo) &&
        this.checkProductionUsage(repo) &&
        this.checkQualityAssurance(repo) &&
        this.checkFrameworkSpecific(repo)
      );
    });
  }

  private async phase2QualityScoring(repos: Repo[]): Promise<Repo[]> {
    const scoredRepos = repos.map(repo => ({
      ...repo,
      qualityScore: this.calculateQualityScore(repo)
    }));

    // 50点以上のみ
    return scoredRepos
      .filter(r => r.qualityScore >= 50)
      .sort((a, b) => b.qualityScore - a.qualityScore)
      .slice(0, 100);  // Top 100
  }

  private async generateManualReviewList(repos: Repo[]): Promise<void> {
    const reviewList = repos.map(repo => ({
      name: repo.name,
      url: repo.html_url,
      score: repo.qualityScore,
      stars: repo.stars,
      lastCommit: repo.pushed_at,
      checkboxes: {
        codeQuality: '[ ]',
        isProduction: '[ ]',
        notTemplate: '[ ]',
        approved: '[ ]'
      }
    }));

    // Markdown出力
    const markdown = this.generateMarkdownChecklist(reviewList);
    await fs.writeFile('data/manual-review-checklist.md', markdown);
  }

  private checkProductionQuality(repo: Repo): boolean {
    // 実装: 上記の productionQualityCriteria
    return (
      this.hasRecentActivity(repo) &&
      this.hasActiveCommunity(repo) &&
      this.isWellMaintained(repo) &&
      this.hasAppropriateSize(repo)
    );
  }

  // ... 他のチェック関数
}
```

### 手動レビュー用チェックリスト自動生成

```markdown
# 手動レビューチェックリスト

生成日: 2026-01-03
対象: 100リポジトリ (Phase 2通過)

---

## 1. vercel/next.js

**基本情報:**
- URL: https://github.com/vercel/next.js
- Stars: 120,000
- Score: 95/100
- Last Commit: 2 days ago

**自動評価:**
- ✅ Production Quality: PASS
- ✅ Code Quality: PASS (Coverage: 78%)
- ✅ Documentation: PASS
- ✅ Active Maintenance: PASS

**手動チェック:**
- [ ] コード品質確認 (src/サンプル確認)
- [ ] プロダクション利用確認
- [ ] テンプレートではないことを確認
- [ ] **最終承認**

**メモ:**
_____________________________________________

---

## 2. facebook/react

...

---

[100リポジトリ分のチェックリスト]
```

---

## 📊 期待される成果の質

### Before (質の低いデータセット)

```
問題:
- Star数だけで選定 → 学習用リポジトリが混入
- チュートリアルが多数含まれる
- メンテナンスされていないコードが対象
- 結果の信頼性が低い

統計的問題:
- 外れ値が多い
- 一般化できない
- 査読者から批判される
```

### After (厳選された高品質データセット)

```
強み:
- ✅ 全てプロダクション環境で使用されているコード
- ✅ アクティブにメンテナンスされている
- ✅ 多様な業界・用途をカバー
- ✅ 再現可能性が高い

統計的優位性:
- 外れ値が少ない
- 一般化可能性が高い
- 査読者が納得する品質
- 業界への示唆が明確
```

---

## 🎓 学術的正当性

### 研究手法としての妥当性

```markdown
## Methods セクションでの記述例

### 3.2 Repository Selection

We employed a three-phase selection process to ensure high-quality,
production-grade repositories:

**Phase 1: Automated Filtering (n = 1,000 → 200)**
- Inclusion criteria:
  - Minimum 500 stars
  - Active maintenance (commit within 3 months)
  - Test coverage > 50%
  - Production deployment evidence
  - Not a template or tutorial

**Phase 2: Quality Scoring (n = 200 → 100)**
- Multi-dimensional quality score (0-100):
  - Development activity (25 points)
  - Code quality (25 points)
  - Documentation (20 points)
  - Community engagement (15 points)
  - Production usage (15 points)
- Threshold: Score ≥ 50

**Phase 3: Manual Validation (n = 100 → 65)**
- Two independent reviewers
- Code quality inspection
- Production verification
- Inter-rater reliability: κ = 0.82 (substantial agreement)

**Final Dataset:**
- n = 65 repositories
- Total: 3.2M lines of code
- Categories: E-commerce (n=10), SaaS (n=10), ...
- Frameworks: Next.js (n=25), React (n=25), ...

This rigorous selection ensures that our findings generalize to
real-world, production-grade React applications.
```

---

## ✅ まとめ

### 質の確保により得られるもの

1. **学術的信頼性**
   - 査読者が納得する選定プロセス
   - Systematic selection の証拠

2. **一般化可能性**
   - プロダクションコードからの知見
   - 業界への実践的示唆

3. **再現性**
   - 明確な選定基準
   - 他の研究者が追試可能

4. **論文の受理確率向上**
   - Methods セクションが堅牢
   - Threats to Validity への対処

### 次のステップ

1. **選定スクリプトの実装** (5-10時間)
2. **Phase 1-2の自動実行** (3-5時間)
3. **Phase 3の手動レビュー** (10-15時間)
4. **最終データセット確定** (50-80リポジトリ)

**総工数: 18-30時間** (データ収集全体の中で)

---

**作成日**: 2026年1月3日
**目的**: GitHub大規模分析の品質保証
**対象**: Phase 4 - オリジナル研究
