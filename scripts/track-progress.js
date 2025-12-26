#!/usr/bin/env node

/**
 * スキル進捗トラッキングスクリプト
 *
 * 全26スキルの進捗を自動測定し、PROGRESS.mdとREADME.mdを更新します。
 *
 * 使用方法:
 *   node scripts/track-progress.js
 */

const fs = require('fs');
const path = require('path');

// 設定
const SKILLS_ROOT = path.join(__dirname, '..');
const PROGRESS_FILE = path.join(SKILLS_ROOT, 'PROGRESS.md');
const README_FILE = path.join(SKILLS_ROOT, 'README.md');

// スキル定義（領域別）
const SKILL_GROUPS = {
  'WEB開発': [
    'react-development',
    'nextjs-development',
    'frontend-performance',
    'web-development',
    'web-accessibility',
  ],
  'iOS開発': [
    'ios-development',
    'swiftui-patterns',
    'ios-security',
    'ios-project-setup',
    'networking-data',
  ],
  'Backend開発': [
    'backend-development',
    'nodejs-development',
    'database-design',
  ],
  'DevOps・品質': [
    'testing-strategy',
    'ci-cd-automation',
    'git-workflow',
    'code-review',
    'quality-assurance',
    'incident-logger',
    'lessons-learned',
  ],
  'その他': [
    'python-development',
    'cli-development',
    'script-development',
    'mcp-development',
    'documentation',
    'dependency-management',
  ],
};

/**
 * ファイルの文字数をカウント
 */
function countCharacters(filePath) {
  try {
    if (!fs.existsSync(filePath)) return 0;
    const content = fs.readFileSync(filePath, 'utf-8');
    return content.length;
  } catch (error) {
    return 0;
  }
}

/**
 * ディレクトリ内の全マークダウンファイルの文字数を集計
 */
function countDirectoryCharacters(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) return 0;

    let total = 0;
    const files = fs.readdirSync(dirPath, { withFileTypes: true });

    for (const file of files) {
      const filePath = path.join(dirPath, file.name);
      if (file.isDirectory()) {
        total += countDirectoryCharacters(filePath);
      } else if (file.name.endsWith('.md')) {
        total += countCharacters(filePath);
      }
    }

    return total;
  } catch (error) {
    return 0;
  }
}

/**
 * 詳細ガイドの数をカウント
 * guidesディレクトリ内の20,000文字以上のマークダウンファイルをカウント
 */
function countGuides(skillDir) {
  const guidesDir = path.join(skillDir, 'guides');
  if (!fs.existsSync(guidesDir)) return 0;

  const MIN_GUIDE_CHARS = 20000; // 詳細ガイドの最小文字数

  let count = 0;

  function traverse(dir) {
    const files = fs.readdirSync(dir, { withFileTypes: true });
    for (const file of files) {
      const filePath = path.join(dir, file.name);

      if (file.isDirectory()) {
        traverse(filePath);
      } else if (file.name.endsWith('.md')) {
        const chars = countCharacters(filePath);
        if (chars >= MIN_GUIDE_CHARS) {
          count++;
        }
      }
    }
  }

  traverse(guidesDir);
  return count;
}

/**
 * スキルの進捗を分析
 */
function analyzeSkill(skillName) {
  const skillDir = path.join(SKILLS_ROOT, skillName);
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  const guidesDir = path.join(skillDir, 'guides');

  // SKILL.mdの文字数
  const skillMdChars = countCharacters(skillMdPath);

  // 詳細ガイドの文字数
  const guidesChars = countDirectoryCharacters(guidesDir);

  // 詳細ガイド数
  const guideCount = countGuides(skillDir);

  // 合計文字数
  const totalChars = skillMdChars + guidesChars;

  // 完成度判定
  let status = 'Not Started';
  let completionRate = 0;

  if (guideCount >= 3) {
    status = 'Complete';
    completionRate = 100;
  } else if (guideCount >= 1) {
    status = 'In Progress';
    completionRate = 33 + (guideCount * 22); // 1本: 55%, 2本: 77%
  } else if (skillMdChars > 5000) {
    status = 'Basic';
    completionRate = 20;
  } else if (skillMdChars > 0) {
    status = 'Started';
    completionRate = 5;
  }

  return {
    name: skillName,
    skillMdChars,
    guidesChars,
    totalChars,
    guideCount,
    status,
    completionRate,
  };
}

/**
 * 全スキルの進捗を分析
 */
function analyzeAllSkills() {
  const results = {
    groups: {},
    totalSkills: 0,
    completedSkills: 0,
    totalChars: 0,
    totalGuides: 0,
  };

  for (const [groupName, skills] of Object.entries(SKILL_GROUPS)) {
    const groupResults = {
      skills: [],
      totalChars: 0,
      completedSkills: 0,
      totalSkills: skills.length,
    };

    for (const skillName of skills) {
      const analysis = analyzeSkill(skillName);
      groupResults.skills.push(analysis);
      groupResults.totalChars += analysis.totalChars;

      if (analysis.status === 'Complete') {
        groupResults.completedSkills++;
        results.completedSkills++;
      }

      results.totalChars += analysis.totalChars;
      results.totalGuides += analysis.guideCount;
    }

    groupResults.completionRate = Math.round(
      (groupResults.completedSkills / groupResults.totalSkills) * 100
    );

    results.groups[groupName] = groupResults;
    results.totalSkills += skills.length;
  }

  results.overallCompletionRate = Math.round(
    (results.completedSkills / results.totalSkills) * 100
  );

  return results;
}

/**
 * プログレスバーを生成
 */
function generateProgressBar(percentage, width = 20) {
  const filled = Math.round((percentage / 100) * width);
  const empty = width - filled;
  return '█'.repeat(filled) + '░'.repeat(empty);
}

/**
 * PROGRESS.mdを生成
 */
function generateProgressMd(results) {
  const lines = [];

  lines.push('# 📊 Skills Progress Tracker');
  lines.push('');
  lines.push('> 自動生成されたファイルです。手動編集しないでください。');
  lines.push('> Last updated: ' + new Date().toLocaleString('ja-JP'));
  lines.push('');

  // 全体サマリー
  lines.push('## 全体進捗');
  lines.push('');
  lines.push(`**進捗率**: ${results.overallCompletionRate}% (${results.completedSkills}/${results.totalSkills} スキル完成)`);
  lines.push('');
  lines.push('```');
  lines.push(generateProgressBar(results.overallCompletionRate, 40) + ` ${results.overallCompletionRate}%`);
  lines.push('```');
  lines.push('');

  // 統計
  lines.push('### 📈 統計');
  lines.push('');
  lines.push('| 項目 | 数値 |');
  lines.push('|------|------|');
  lines.push(`| **完了スキル数** | ${results.completedSkills} / ${results.totalSkills} |`);
  lines.push(`| **総文字数** | ${results.totalChars.toLocaleString()} 文字 |`);
  lines.push(`| **詳細ガイド数** | ${results.totalGuides} 本 |`);
  lines.push(`| **平均文字数/スキル** | ${Math.round(results.totalChars / results.completedSkills || 0).toLocaleString()} 文字 |`);
  lines.push('');

  // 領域別進捗
  lines.push('## 領域別進捗');
  lines.push('');

  for (const [groupName, groupData] of Object.entries(results.groups)) {
    lines.push(`### ${groupName}`);
    lines.push('');
    lines.push(`**進捗**: ${groupData.completionRate}% (${groupData.completedSkills}/${groupData.totalSkills} 完成) | **文字数**: ${groupData.totalChars.toLocaleString()}`);
    lines.push('');
    lines.push('```');
    lines.push(generateProgressBar(groupData.completionRate, 30) + ` ${groupData.completionRate}%`);
    lines.push('```');
    lines.push('');

    // スキル一覧
    lines.push('| スキル | 状態 | 文字数 | ガイド数 | 完成度 |');
    lines.push('|--------|------|--------|---------|--------|');

    for (const skill of groupData.skills) {
      const statusEmoji = {
        'Complete': '✅',
        'In Progress': '🔄',
        'Basic': '📝',
        'Started': '🌱',
        'Not Started': '⬜',
      }[skill.status];

      lines.push(
        `| ${skill.name} | ${statusEmoji} ${skill.status} | ${skill.totalChars.toLocaleString()} | ${skill.guideCount}/3 | ${skill.completionRate}% |`
      );
    }

    lines.push('');
  }

  // フッター
  lines.push('---');
  lines.push('');
  lines.push('_このレポートは `npm run track` で更新できます。_');
  lines.push('');

  return lines.join('\n');
}

/**
 * README.mdのバッジセクションを生成
 */
function generateReadmeBadgeSection(results) {
  const lines = [];

  lines.push('<!-- PROGRESS_BADGES_START -->');
  lines.push(`![Progress](https://img.shields.io/badge/Progress-${results.overallCompletionRate}%25-${results.overallCompletionRate > 50 ? 'green' : 'yellow'})`);
  lines.push(`![Skills](https://img.shields.io/badge/Skills-${results.completedSkills}%2F${results.totalSkills}-blue)`);
  lines.push(`![Characters](https://img.shields.io/badge/Characters-${Math.round(results.totalChars / 1000)}K-informational)`);
  lines.push(`![Guides](https://img.shields.io/badge/Guides-${results.totalGuides}-success)`);
  lines.push('<!-- PROGRESS_BADGES_END -->');

  return lines.join('\n');
}

/**
 * README.mdを更新
 */
function updateReadme(results) {
  if (!fs.existsSync(README_FILE)) {
    console.log('⚠️  README.md not found. Creating new one...');
    const initialContent = `# Claude Code Skills

${generateReadmeBadgeSection(results)}

## 📚 Overview

This repository contains comprehensive guides for software development skills.

**Progress**: ${results.completedSkills}/${results.totalSkills} skills completed (${results.overallCompletionRate}%)

See [PROGRESS.md](./PROGRESS.md) for detailed progress tracking.

## 🎯 Skills

`;

    for (const [groupName, groupData] of Object.entries(results.groups)) {
      const content = initialContent + `\n### ${groupName}\n\n`;
      for (const skill of groupData.skills) {
        const status = skill.status === 'Complete' ? '✅' : '⏳';
        content += `- ${status} [${skill.name}](./${skill.name}/)\n`;
      }
    }

    fs.writeFileSync(README_FILE, initialContent);
    console.log('✅ README.md created');
    return;
  }

  // 既存のREADME.mdのバッジセクションのみ更新
  let content = fs.readFileSync(README_FILE, 'utf-8');

  const badgeSection = generateReadmeBadgeSection(results);

  if (content.includes('<!-- PROGRESS_BADGES_START -->')) {
    content = content.replace(
      /<!-- PROGRESS_BADGES_START -->[\s\S]*?<!-- PROGRESS_BADGES_END -->/,
      badgeSection
    );
  } else {
    // バッジセクションが存在しない場合、先頭に追加
    content = `# Claude Code Skills\n\n${badgeSection}\n\n` + content.replace(/^#.*\n\n/, '');
  }

  fs.writeFileSync(README_FILE, content);
  console.log('✅ README.md updated');
}

/**
 * メイン処理
 */
function main() {
  console.log('📊 Analyzing skills progress...\n');

  const results = analyzeAllSkills();

  // PROGRESS.md生成
  const progressContent = generateProgressMd(results);
  fs.writeFileSync(PROGRESS_FILE, progressContent);
  console.log('✅ PROGRESS.md generated');

  // README.md更新
  updateReadme(results);

  // サマリー表示
  console.log('\n📈 Summary:');
  console.log(`   Progress: ${results.overallCompletionRate}% (${results.completedSkills}/${results.totalSkills} skills)`);
  console.log(`   Total Characters: ${results.totalChars.toLocaleString()}`);
  console.log(`   Total Guides: ${results.totalGuides}`);
  console.log('\n✨ Done! Check PROGRESS.md for details.');
}

main();
