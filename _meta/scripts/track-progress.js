#!/usr/bin/env node

/**
 * スキル進捗トラッキングスクリプト
 *
 * 36 Skills / 8カテゴリの進捗を自動測定し、README.mdを更新します。
 *
 * 使用方法:
 *   node _meta/scripts/track-progress.js
 */

const fs = require('fs');
const path = require('path');

// 設定
const SKILLS_ROOT = path.join(__dirname, '..', '..');
const README_FILE = path.join(SKILLS_ROOT, 'README.md');

// カテゴリ定義（新構造）
const CATEGORIES = {
  '01-cs-fundamentals': {
    label: 'CS基礎',
    skills: [
      'computer-science-fundamentals',
      'algorithm-and-data-structures',
      'operating-system-guide',
      'programming-language-fundamentals',
    ],
  },
  '02-programming': {
    label: 'プログラミング',
    skills: [
      'object-oriented-programming',
      'async-and-error-handling',
      'typescript-complete-guide',
      'go-practical-guide',
      'rust-systems-programming',
      'regex-and-text-processing',
    ],
  },
  '03-software-design': {
    label: '設計・品質',
    skills: [
      'clean-code-principles',
      'design-patterns-guide',
      'system-design-guide',
    ],
  },
  '04-web-and-network': {
    label: 'Web・ネットワーク',
    skills: [
      'network-fundamentals',
      'browser-and-web-platform',
      'web-application-development',
      'api-and-library-guide',
    ],
  },
  '05-infrastructure': {
    label: 'インフラ・DevOps',
    skills: [
      'linux-cli-mastery',
      'docker-container-guide',
      'aws-cloud-guide',
      'devops-and-github-actions',
      'development-environment-setup',
      'windows-application-development',
      'version-control-and-jujutsu',
    ],
  },
  '06-data-and-security': {
    label: 'データ・セキュリティ',
    skills: [
      'sql-and-query-mastery',
      'security-fundamentals',
      'authentication-and-authorization',
    ],
  },
  '07-ai': {
    label: 'AI・LLM',
    skills: [
      'llm-and-ai-comparison',
      'ai-analysis-guide',
      'ai-audio-generation',
      'ai-visual-generation',
      'ai-automation-and-monetization',
      'ai-era-development-workflow',
      'ai-era-gadgets',
      'custom-ai-agents',
    ],
  },
  '08-hobby': {
    label: '趣味',
    skills: [
      'dj-skills-guide',
    ],
  },
};

/**
 * ディレクトリ内の全マークダウンファイル数と文字数を集計
 */
function countDocsDir(dirPath) {
  let fileCount = 0;
  let totalChars = 0;

  function traverse(dir) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        traverse(fullPath);
      } else if (entry.name.endsWith('.md')) {
        fileCount++;
        totalChars += fs.readFileSync(fullPath, 'utf-8').length;
      }
    }
  }

  traverse(dirPath);
  return { fileCount, totalChars };
}

/**
 * スキルを分析
 */
function analyzeSkill(categoryDir, skillName) {
  const skillDir = path.join(SKILLS_ROOT, categoryDir, skillName);
  const docsDir = path.join(skillDir, 'docs');
  const { fileCount, totalChars } = countDocsDir(docsDir);

  return {
    name: skillName,
    fileCount,
    totalChars,
  };
}

/**
 * 全カテゴリの進捗を分析
 */
function analyzeAll() {
  const results = {
    categories: {},
    totalSkills: 0,
    totalFiles: 0,
    totalChars: 0,
  };

  for (const [catDir, catDef] of Object.entries(CATEGORIES)) {
    const catResult = {
      label: catDef.label,
      skills: [],
      totalFiles: 0,
      totalChars: 0,
    };

    for (const skillName of catDef.skills) {
      const analysis = analyzeSkill(catDir, skillName);
      catResult.skills.push(analysis);
      catResult.totalFiles += analysis.fileCount;
      catResult.totalChars += analysis.totalChars;
    }

    results.categories[catDir] = catResult;
    results.totalSkills += catDef.skills.length;
    results.totalFiles += catResult.totalFiles;
    results.totalChars += catResult.totalChars;
  }

  return results;
}

/**
 * README.mdのバッジセクションを更新
 */
function updateReadme(results) {
  if (!fs.existsSync(README_FILE)) {
    console.log('⚠️  README.md not found');
    return;
  }

  let content = fs.readFileSync(README_FILE, 'utf-8');

  const charsDisplay = results.totalChars >= 10000000
    ? `${(results.totalChars / 10000).toFixed(0)}万`
    : `${(results.totalChars / 10000).toFixed(0)}万`;

  const badge = [
    '<!-- PROGRESS_BADGES_START -->',
    `![Skills](https://img.shields.io/badge/Skills-${results.totalSkills}-blue)`,
    `![Guides](https://img.shields.io/badge/Guides-${results.totalFiles}-success)`,
    `![Characters](https://img.shields.io/badge/Characters-${Math.round(results.totalChars / 1000)}K-informational)`,
    '<!-- PROGRESS_BADGES_END -->',
  ].join('\n');

  if (content.includes('<!-- PROGRESS_BADGES_START -->')) {
    content = content.replace(
      /<!-- PROGRESS_BADGES_START -->[\s\S]*?<!-- PROGRESS_BADGES_END -->/,
      badge
    );
  } else {
    // バッジセクションをタイトル行の直後に挿入
    content = content.replace(
      /^(# Claude Code Skills\n)/,
      `$1\n${badge}\n`
    );
  }

  fs.writeFileSync(README_FILE, content);
  console.log('✅ README.md updated');
}

/**
 * メイン処理
 */
function main() {
  console.log('📊 Analyzing skills progress...\n');

  const results = analyzeAll();

  // README.md更新
  updateReadme(results);

  // サマリー表示
  console.log('\n📈 Summary:');
  console.log(`   Skills: ${results.totalSkills}`);
  console.log(`   Guide files: ${results.totalFiles}`);
  console.log(`   Total characters: ${results.totalChars.toLocaleString()}`);

  for (const [catDir, catData] of Object.entries(results.categories)) {
    console.log(`   ${catDir} (${catData.label}): ${catData.totalFiles} files / ${catData.totalChars.toLocaleString()} chars`);
  }

  console.log('\n✨ Done!');
}

main();
