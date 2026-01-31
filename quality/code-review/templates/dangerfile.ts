// Dangerfile.ts - Comprehensive PR Automation
import { danger, warn, fail, message, markdown } from 'danger';
import * as fs from 'fs';

// ========================================
// Configuration
// ========================================
const CONFIG = {
  prSize: {
    small: 100,
    medium: 500,
    large: 1000,
  },
  coverage: {
    threshold: 80,
  },
  complexity: {
    maxPerFunction: 10,
  },
};

// ========================================
// 1. PR Size Check
// ========================================
function checkPRSize() {
  const additions = danger.github.pr.additions;
  const deletions = danger.github.pr.deletions;
  const changes = additions + deletions;

  if (changes > CONFIG.prSize.large) {
    fail(
      `⚠️  このPRは非常に大きいです（${changes}行）。小さなPRに分割することを強く推奨します。\n\n` +
      `**理由**:\n` +
      `- レビュー時間の短縮\n` +
      `- バグ発見率の向上\n` +
      `- マージリスクの低減`
    );
  } else if (changes > CONFIG.prSize.medium) {
    warn(
      `⚠️  このPRは大きめです（${changes}行）。分割を検討してください。\n` +
      `目安: ${CONFIG.prSize.medium}行以下`
    );
  } else if (changes < CONFIG.prSize.small) {
    message(`✅ PRサイズが適切です（${changes}行）`);
  }
}

// ========================================
// 2. PR Title Check (Conventional Commits)
// ========================================
function checkPRTitle() {
  const title = danger.github.pr.title;
  const conventionalCommitRegex = /^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .+/;

  if (!conventionalCommitRegex.test(title)) {
    fail(
      '❌ PRタイトルはConventional Commits形式に従ってください。\n\n' +
      '**形式**: `type(scope): description`\n\n' +
      '**例**:\n' +
      '- `feat(auth): add login functionality`\n' +
      '- `fix(api): resolve user data fetch issue`\n' +
      '- `docs(readme): update installation guide`\n\n' +
      '**Types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert'
    );
  }
}

// ========================================
// 3. PR Description Check
// ========================================
function checkPRDescription() {
  const description = danger.github.pr.body;

  if (!description || description.length < 50) {
    warn(
      '⚠️  PR説明が短すぎます。以下の情報を含めてください:\n' +
      '- 変更の概要\n' +
      '- 変更の理由\n' +
      '- テスト方法\n' +
      '- スクリーンショット（UI変更の場合）'
    );
  }

  // Check for required sections
  const requiredSections = ['## 概要', '## 変更内容', '## テスト'];
  const missingSections = requiredSections.filter(
    section => !description?.includes(section)
  );

  if (missingSections.length > 0) {
    warn(
      `⚠️  以下のセクションがPR説明に含まれていません:\n${missingSections.join('\n')}`
    );
  }
}

// ========================================
// 4. Label Check
// ========================================
function checkLabels() {
  const labels = danger.github.issue.labels.map(l => l.name);

  if (labels.length === 0) {
    warn('⚠️  ラベルを追加してください（例: feature, bugfix, documentation）');
  }

  // Check for priority label
  const hasPriority = labels.some(l => l.startsWith('priority/'));
  if (!hasPriority) {
    message('💡 優先度ラベル（priority/high, priority/medium, priority/low）の追加を検討してください');
  }
}

// ========================================
// 5. File Change Analysis
// ========================================
function analyzeFileChanges() {
  const modifiedFiles = danger.git.modified_files;
  const createdFiles = danger.git.created_files;
  const deletedFiles = danger.git.deleted_files;

  // Package.json changes
  if (modifiedFiles.includes('package.json')) {
    if (!modifiedFiles.includes('package-lock.json') && !modifiedFiles.includes('yarn.lock')) {
      fail('❌ package.jsonが変更されましたが、ロックファイルが更新されていません。');
    }
    message('📦 依存関係が変更されました。npm auditを実行してください。');
  }

  // Database migrations
  const migrationFiles = [...createdFiles, ...modifiedFiles].filter(f =>
    f.includes('migrations/')
  );
  if (migrationFiles.length > 0) {
    warn(
      '🗄️  データベースマイグレーションが含まれています。\n' +
      '- ロールバック手順を確認してください\n' +
      '- 本番環境での実行計画を立ててください'
    );
  }

  // Environment files
  const envFiles = [...createdFiles, ...modifiedFiles].filter(f =>
    f.endsWith('.env') || f.endsWith('.env.example')
  );
  if (envFiles.length > 0) {
    warn('⚙️  環境変数ファイルが変更されています。ドキュメントを更新してください。');
  }
}

// ========================================
// 6. Test Coverage Check
// ========================================
function checkTestCoverage() {
  const modifiedFiles = danger.git.modified_files;
  const createdFiles = danger.git.created_files;

  // Check if source files changed
  const sourceFiles = [...modifiedFiles, ...createdFiles].filter(
    f => f.startsWith('src/') && !f.includes('.test.') && !f.includes('.spec.')
  );

  // Check if test files changed
  const testFiles = [...modifiedFiles, ...createdFiles].filter(
    f => f.includes('.test.') || f.includes('.spec.')
  );

  if (sourceFiles.length > 0 && testFiles.length === 0) {
    warn('⚠️  ソースコードが変更されていますが、テストが追加されていません。');
  }

  // Read coverage report
  try {
    const coverageSummary = JSON.parse(
      fs.readFileSync('coverage/coverage-summary.json', 'utf-8')
    );

    const coverage = coverageSummary.total.lines.pct;

    if (coverage < CONFIG.coverage.threshold) {
      fail(
        `❌ カバレッジ${coverage.toFixed(2)}%が閾値${CONFIG.coverage.threshold}%未満です。\n` +
        'テストを追加してください。'
      );
    } else {
      message(`✅ カバレッジ: ${coverage.toFixed(2)}%`);
    }

    // Create coverage table
    markdown(`
## 📊 コードカバレッジ

| メトリック | カバレッジ | 閾値 | 状態 |
|-----------|----------|------|------|
| Lines | ${coverageSummary.total.lines.pct.toFixed(2)}% | ${CONFIG.coverage.threshold}% | ${coverageSummary.total.lines.pct >= CONFIG.coverage.threshold ? '✅' : '❌'} |
| Statements | ${coverageSummary.total.statements.pct.toFixed(2)}% | ${CONFIG.coverage.threshold}% | ${coverageSummary.total.statements.pct >= CONFIG.coverage.threshold ? '✅' : '❌'} |
| Functions | ${coverageSummary.total.functions.pct.toFixed(2)}% | ${CONFIG.coverage.threshold}% | ${coverageSummary.total.functions.pct >= CONFIG.coverage.threshold ? '✅' : '❌'} |
| Branches | ${coverageSummary.total.branches.pct.toFixed(2)}% | ${CONFIG.coverage.threshold}% | ${coverageSummary.total.branches.pct >= CONFIG.coverage.threshold ? '✅' : '❌'} |
    `);
  } catch (error) {
    warn('⚠️  カバレッジレポートを読み込めませんでした。テストを実行してください。');
  }
}

// ========================================
// 7. Debug Code Check
// ========================================
function checkDebugCode() {
  const modifiedFiles = danger.git.modified_files;
  const createdFiles = danger.git.created_files;

  const codeFiles = [...modifiedFiles, ...createdFiles].filter(f =>
    f.match(/\.(ts|tsx|js|jsx)$/)
  );

  let debugIssues = 0;

  for (const file of codeFiles) {
    const content = fs.readFileSync(file, 'utf-8');

    // console.log/debug/info
    const consoleStatements = content.match(/console\.(log|debug|info)/g);
    if (consoleStatements && consoleStatements.length > 0) {
      warn(
        `⚠️  ${file}に${consoleStatements.length}個のconsole文があります。\n` +
        'ロギングライブラリを使用するか、削除してください。'
      );
      debugIssues++;
    }

    // debugger
    if (content.includes('debugger')) {
      fail(`❌ ${file}にdebuggerステートメントがあります。削除してください。`);
      debugIssues++;
    }
  }

  if (debugIssues === 0) {
    message('✅ デバッグコードは見つかりませんでした');
  }
}

// ========================================
// 8. TODO/FIXME Comments
// ========================================
function checkTodoComments() {
  const modifiedFiles = danger.git.modified_files;
  const createdFiles = danger.git.created_files;

  const codeFiles = [...modifiedFiles, ...createdFiles].filter(f =>
    f.match(/\.(ts|tsx|js|jsx)$/)
  );

  const todos: { file: string; count: number }[] = [];

  for (const file of codeFiles) {
    const content = fs.readFileSync(file, 'utf-8');
    const todoMatches = content.match(/\/\/ TODO:|\/\/ FIXME:/g);

    if (todoMatches && todoMatches.length > 0) {
      todos.push({ file, count: todoMatches.length });
    }
  }

  if (todos.length > 0) {
    const totalTodos = todos.reduce((sum, t) => sum + t.count, 0);
    warn(
      `⚠️  ${totalTodos}個のTODO/FIXMEコメントが見つかりました:\n` +
      todos.map(t => `- ${t.file}: ${t.count}個`).join('\n') +
      '\n\nIssueを作成することを検討してください。'
    );
  }
}

// ========================================
// 9. Impact Analysis
// ========================================
function analyzeImpact() {
  const modifiedFiles = danger.git.modified_files;

  const impactAreas = {
    database: modifiedFiles.some(f => f.includes('migrations/') || f.includes('models/')),
    api: modifiedFiles.some(f => f.includes('api/') || f.includes('routes/')),
    ui: modifiedFiles.some(f => f.includes('components/') || f.includes('pages/')),
    auth: modifiedFiles.some(f => f.includes('auth/')),
    config: modifiedFiles.some(f => f.includes('config/') || f.endsWith('.config.ts')),
  };

  const impacts = Object.entries(impactAreas)
    .filter(([_, changed]) => changed)
    .map(([area]) => area);

  if (impacts.length > 0) {
    markdown(`
## 🎯 影響範囲

このPRは以下の領域に影響します:

${impacts.map(area => `- **${area}**`).join('\n')}

該当チームのレビューを依頼してください。
    `);
  }
}

// ========================================
// 10. Breaking Changes Check
// ========================================
function checkBreakingChanges() {
  const description = danger.github.pr.body || '';
  const title = danger.github.pr.title;

  const hasBreakingChange =
    title.includes('BREAKING') ||
    title.includes('!:') ||
    description.includes('BREAKING CHANGE') ||
    description.includes('Breaking Changes');

  if (hasBreakingChange) {
    warn(
      '⚠️  **破壊的変更が含まれています**\n\n' +
      '以下を確認してください:\n' +
      '- [ ] マイグレーションガイドを作成\n' +
      '- [ ] CHANGELOGに記載\n' +
      '- [ ] メジャーバージョンアップを検討\n' +
      '- [ ] ユーザーに通知'
    );
  }
}

// ========================================
// 11. Security Check
// ========================================
function checkSecurity() {
  const modifiedFiles = danger.git.modified_files;
  const createdFiles = danger.git.created_files;

  const allFiles = [...modifiedFiles, ...createdFiles];

  // Check for hardcoded secrets
  const secretPatterns = [
    /password\s*=\s*['"][^'"]+['"]/i,
    /api[_-]?key\s*=\s*['"][^'"]+['"]/i,
    /secret\s*=\s*['"][^'"]+['"]/i,
    /token\s*=\s*['"][^'"]+['"]/i,
  ];

  for (const file of allFiles) {
    if (!file.match(/\.(ts|tsx|js|jsx|py|rb|go)$/)) continue;

    const content = fs.readFileSync(file, 'utf-8');

    for (const pattern of secretPatterns) {
      if (pattern.test(content)) {
        fail(
          `🔒 ${file}にハードコードされた秘密情報の可能性があります。\n` +
          '環境変数を使用してください。'
        );
        break;
      }
    }
  }

  // Check for .env files
  const envFiles = allFiles.filter(f => f.endsWith('.env') && !f.endsWith('.env.example'));
  if (envFiles.length > 0) {
    fail(
      '❌ .envファイルをコミットしないでください:\n' +
      envFiles.join('\n')
    );
  }
}

// ========================================
// 12. Documentation Check
// ========================================
function checkDocumentation() {
  const modifiedFiles = danger.git.modified_files;
  const createdFiles = danger.git.created_files;

  // Check if README needs update
  const hasApiChanges = [...modifiedFiles, ...createdFiles].some(f =>
    f.includes('api/') || f.includes('routes/')
  );

  const readmeUpdated = modifiedFiles.includes('README.md');

  if (hasApiChanges && !readmeUpdated) {
    warn('⚠️  API変更がありますが、READMEが更新されていません。ドキュメントの更新を検討してください。');
  }

  // Check for CHANGELOG update
  const changelogUpdated = modifiedFiles.includes('CHANGELOG.md');
  if (!changelogUpdated) {
    message('💡 CHANGELOGの更新を忘れずに！');
  }
}

// ========================================
// 13. Performance Check
// ========================================
function checkPerformance() {
  // This would typically read from benchmark results
  message('💡 パフォーマンスへの影響を確認してください。必要に応じてベンチマークを実行してください。');
}

// ========================================
// 14. Summary Report
// ========================================
function generateSummary() {
  const pr = danger.github.pr;

  markdown(`
# 📋 PR レビューサマリー

## 基本情報
- **作成者**: @${pr.user.login}
- **ブランチ**: \`${pr.head.ref}\` → \`${pr.base.ref}\`
- **変更行数**: +${pr.additions} -${pr.deletions}
- **変更ファイル数**: ${pr.changed_files}

## 次のステップ
1. ✅ 自動チェックの結果を確認
2. 👀 レビュワーによるコードレビュー
3. ✏️  フィードバックへの対応
4. ✅ 承認後にマージ
  `);
}

// ========================================
// Main Execution
// ========================================
async function main() {
  console.log('🤖 Danger.js running...\n');

  // Run all checks
  checkPRSize();
  checkPRTitle();
  checkPRDescription();
  checkLabels();
  analyzeFileChanges();
  checkTestCoverage();
  checkDebugCode();
  checkTodoComments();
  analyzeImpact();
  checkBreakingChanges();
  checkSecurity();
  checkDocumentation();
  checkPerformance();
  generateSummary();

  console.log('✅ Danger.js completed');
}

main();
