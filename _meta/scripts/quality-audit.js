#!/usr/bin/env node

/**
 * quality-audit.js — 構造・コンテンツ品質チェック
 *
 * QUALITY_STANDARDS.md の7項目基準に基づき、全ガイドファイルを自動チェック。
 * 出力: JSON (_meta/REVIEW_RESULTS/quality-audit.json)
 *       Markdown (_meta/REVIEW_RESULTS/quality-audit-summary.md)
 *
 * 使用方法:
 *   node _meta/scripts/quality-audit.js                    # 全件チェック
 *   node _meta/scripts/quality-audit.js 04-web-and-network # カテゴリ指定
 *   node _meta/scripts/quality-audit.js --json-only        # JSON出力のみ
 */

const fs = require('fs');
const path = require('path');

const SKILLS_ROOT = path.join(__dirname, '..', '..');
const OUTPUT_DIR = path.join(__dirname, '..', 'REVIEW_RESULTS');

// ─── 設定 ─────────────────────────────────────────
const MIN_CHARS = 40000;
const MIN_CODE_BLOCKS = 5;
const MIN_DIAGRAMS = 3;
const MIN_TABLES = 2;
const MIN_FAQ = 3;
const MIN_EXERCISES = 3;
const MIN_REFERENCES = 3;

// 必須セクション（GUIDE_TEMPLATE.md 準拠）
// NOTE: 番号付き見出し（## 6. まとめ 等）にも対応するため (?:\d+[\.\s]+)? を含む
const REQUIRED_SECTIONS = [
  { pattern: /^#\s+.+/m, label: 'H1タイトル' },
  { pattern: /##\s*(?:\d+[\.\s]+)?(?:この章で学ぶこと|学ぶこと|学習目標|概要と学習目標)/m, label: 'この章で学ぶこと' },
  { pattern: /##\s*(?:\d+[\.\s]+)?前提知識/m, label: '前提知識' },
  { pattern: /##\s*(?:\d+[\.\s]+)?(?:FAQ|よくある質問)/mi, label: 'FAQ' },
  { pattern: /##\s*(?:\d+[\.\s]+)?(?:まとめ|総まとめ|おわりに|結論|Summary|Conclusion)/mi, label: 'まとめ' },
  { pattern: /##\s*(?:\d+[\.\s]+)?(?:次に読むべきガイド|関連ガイド|関連リソース|次のステップ|次章|Next\s*Steps?)/mi, label: '次に読むべきガイド' },
  { pattern: /##\s*(?:\d+[\.\s]+)?(?:参考文献|参考資料|参考リンク|References)/mi, label: '参考文献' },
];

// テンプレート構造セクション（番号付き・番号なし両対応）
const STRUCTURAL_SECTIONS = [
  { pattern: /##\s*(?:\d+[\.\s]+)?.*(なぜ|必要|背景|歴史|導入|はじめに|Introduction)/mi, label: '導入セクション' },
  { pattern: /##\s*(?:\d+[\.\s]+)?.*(概念|核心|基本|コンセプト|基礎|Fundamentals)/mi, label: '概念セクション' },
  { pattern: /##\s*(?:\d+[\.\s]+)?.*(実装|実践|コード|プログラ|使い方|活用|Implementation)/mi, label: '実装セクション' },
  { pattern: /##\s*(?:\d+[\.\s]+)?.*(アンチパターン|トレードオフ|注意|落とし穴|間違い|よくある問題)/m, label: 'アンチパターンセクション' },
  { pattern: /##\s*(?:\d+[\.\s]+)?.*(演習|練習|ハンズオン|Exercise|実践課題)/mi, label: '演習セクション' },
  { pattern: /##\s*(?:\d+[\.\s]+)?.*(深掘り|発展|上級|応用|Advanced|詳細)/mi, label: '深掘りセクション' },
];

// フィラー表現（冗長・繰り返しの検出）
const FILLER_PATTERNS = [
  { pattern: /それでは[、,]?早速/g, label: 'それでは早速' },
  { pattern: /いかがでしたでしょうか/g, label: 'いかがでしたでしょうか' },
  { pattern: /ご存[知じ]の通り/g, label: 'ご存知の通り' },
  { pattern: /言うまでもなく/g, label: '言うまでもなく' },
  { pattern: /ここまで.*見てきました/g, label: 'ここまで見てきました' },
  { pattern: /しっかり[と]?理解/g, label: 'しっかり理解' },
  { pattern: /非常に重要/g, label: '非常に重要（多用注意）' },
  { pattern: /まず最初に/g, label: 'まず最初に（冗長）' },
  { pattern: /基本的[にな]/g, label: '基本的に（多用注意）', threshold: 10 },
  { pattern: /重要(です|な[のこ])/g, label: '重要（多用注意）', threshold: 15 },
];

// 古い年号パターン
const OUTDATED_YEAR_PATTERNS = [
  { pattern: /20(?:1[0-9]|2[0-2])年(?:現在|時点|版)/g, label: '古い年号参照' },
  { pattern: /(?:最新|現行).*20(?:1[0-9]|2[0-2])/g, label: '古い「最新」参照' },
];

// 古いバージョン検出（主要ツール・フレームワーク）
const OUTDATED_VERSION_PATTERNS = [
  { pattern: /Node\.?js\s*(?:v?1[0-6]|v?[0-9])[\.\s]/gi, label: 'Node.js古いバージョン' },
  { pattern: /React\s*(?:v?1[0-6]|v?[0-9])[\.\s]/gi, label: 'React古いバージョン' },
  { pattern: /TypeScript\s*(?:v?[1-4])\.[0-9]/gi, label: 'TypeScript古いバージョン' },
  { pattern: /Python\s*(?:2\.|3\.[0-8][\.\s])/gi, label: 'Python古いバージョン' },
  { pattern: /Docker\s*(?:1[0-9]|[0-9])\./gi, label: 'Docker古いバージョン' },
];

// ─── ユーティリティ ──────────────────────────────
function getAllMarkdownFiles(dir) {
  const results = [];
  function walk(d) {
    if (!fs.existsSync(d)) return;
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) {
        if (!['node_modules', '.git', '_meta', '_legacy', '_original-skills'].includes(entry.name)) {
          walk(full);
        }
      } else if (entry.name.endsWith('.md')) {
        results.push(full);
      }
    }
  }
  walk(dir);
  return results;
}

function categorizeFile(filePath) {
  const rel = path.relative(SKILLS_ROOT, filePath);
  const parts = rel.split(path.sep);
  if (parts.length < 2) return { category: '_root', skill: '', type: 'other' };

  const category = parts[0];
  const skill = parts[1] || '';

  let type = 'other';
  if (parts.includes('docs')) type = 'guide';
  else if (path.basename(filePath).match(/^(SKILL|README)\.md$/i)) type = 'readme';
  else if (path.basename(filePath).match(/checklist/i)) type = 'checklist';

  return { category, skill, type };
}

// ─── チェック関数群 ──────────────────────────────
function countCodeBlocks(content) {
  const matches = content.match(/```[\s\S]*?```/g);
  return matches ? matches.length : 0;
}

function countTables(content) {
  // Markdown表: |で始まる行 + セパレータ行（|---|）を含むブロック
  const tablePattern = /(?:^|\n)\s*\|[^\n]+\|\s*\n\s*\|[\s:|-]+\|\s*\n(?:\s*\|[^\n]+\|\s*\n?)+/g;
  const matches = content.match(tablePattern);
  return matches ? matches.length : 0;
}

function countDiagrams(content) {
  let count = 0;
  // ASCIIアート系（矢印、ボックス等を含むコードブロック）
  const codeBlocks = content.match(/```[^`]*```/gs) || [];
  for (const block of codeBlocks) {
    // 言語指定なし or plaintext/text/ascii のコードブロック
    if (block.match(/```\s*(?:text|plaintext|ascii|mermaid)?\s*\n/i)) {
      // 矢印、ボックス描画文字、フローチャート要素を含む
      if (block.match(/[─│┌┐└┘├┤┬┴┼→←↑↓▶◀▲▼\+\-\|>]{3,}|[-=]{3,}>|<[-=]{3,}|[\|+][-=]+[\|+]/)) {
        count++;
      }
    }
  }
  // インラインASCIIダイアグラム
  const inlineDiagrams = content.match(/(?:^|\n)[\s]*[┌┐└┘│─├┤┬┴┼][^\n]*\n/g);
  if (inlineDiagrams && inlineDiagrams.length >= 3) count++;

  return count;
}

function countFAQ(content) {
  const faqSection = content.match(/##\s*(?:\d+[\.\s]+)?(?:FAQ|よくある質問)[\s\S]*?(?=\n## (?!\s*#)|$)/i);
  if (!faqSection) return 0;
  // ### Q1, ### 質問1, ### ...?
  const h3Questions = faqSection[0].match(/###\s*(Q\d*[:：]?\s|質問\d*[:：]?\s|.+[？?])/g) || [];
  // **Q:**, **Q1:**, **Q. **
  const boldQuestions = faqSection[0].match(/\*\*Q\d*[:：．.]\s*\*\*/g) || [];
  return Math.max(h3Questions.length, boldQuestions.length);
}

function countExercises(content) {
  // ### 演習 1, ### 練習問題, ### Exercise, #### 演習, **演習1** etc.
  const h3Exercises = content.match(/#{3,4}\s*(?:演習|練習問題?|Exercise|ハンズオン|実践課題|Practice)\s*\d*/gi) || [];
  const boldExercises = content.match(/\*\*(?:演習|練習問題?|Exercise|実践課題)\s*\d*/gi) || [];
  // 重複排除のためユニークカウント
  return h3Exercises.length + boldExercises.length;
}

function countReferences(content) {
  const refSection = content.match(/##\s*(?:\d+[\.\s]+)?(?:参考文献|参考資料|参考リンク|References)[\s\S]*$/im);
  if (!refSection) return 0;
  const items = refSection[0].match(/^\s*\d+\.\s+/gm);
  const links = refSection[0].match(/^\s*[-*]\s+/gm);
  return (items ? items.length : 0) + (links ? links.length : 0);
}

function countLearningObjectives(content) {
  const section = content.match(/##\s*(?:\d+[\.\s]+)?(?:この章で学ぶこと|学ぶこと|学習目標|概要と学習目標)[\s\S]*?(?=\n## |\n---)/m);
  if (!section) return 0;
  // チェックボックス / バレットリスト / 番号付きリスト すべてカウント
  const checkboxes = section[0].match(/^\s*-\s*\[[ x]\]/gm);
  const bullets = section[0].match(/^\s*[-*]\s+\S/gm);
  const numbered = section[0].match(/^\s*\d+\.\s+/gm);
  const max = Math.max(
    checkboxes ? checkboxes.length : 0,
    bullets ? bullets.length : 0,
    numbered ? numbered.length : 0
  );
  return max;
}

function checkHeadingHierarchy(content) {
  const headings = content.match(/^#{1,6}\s+.+/gm) || [];
  const issues = [];
  let prevLevel = 0;

  for (const h of headings) {
    const level = h.match(/^(#+)/)[1].length;
    if (prevLevel > 0 && level > prevLevel + 1) {
      issues.push(`見出しレベルのスキップ: H${prevLevel} → H${level} ("${h.trim().substring(0, 40)}")`);
    }
    prevLevel = level;
  }
  return issues;
}

function detectFillers(content) {
  const found = [];
  for (const { pattern, label, threshold } of FILLER_PATTERNS) {
    const matches = content.match(pattern);
    const count = matches ? matches.length : 0;
    const limit = threshold || 3;
    if (count >= limit) {
      found.push({ label, count, severity: count >= limit * 2 ? 'WARNING' : 'INFO' });
    }
  }
  return found;
}

function detectOutdatedYears(content) {
  const found = [];
  for (const { pattern, label } of OUTDATED_YEAR_PATTERNS) {
    const matches = content.match(pattern);
    if (matches) {
      found.push({ label, matches: [...new Set(matches)], count: matches.length });
    }
  }
  return found;
}

function detectOutdatedVersions(content) {
  const found = [];
  for (const { pattern, label } of OUTDATED_VERSION_PATTERNS) {
    const matches = content.match(pattern);
    if (matches) {
      found.push({ label, matches: [...new Set(matches)], count: matches.length });
    }
  }
  return found;
}

function detectRepeatedPhrases(content) {
  // 3回以上繰り返される長いフレーズ（10文字以上）を検出
  const sentences = content.split(/[。\n]/);
  const phraseCount = {};
  for (const s of sentences) {
    const trimmed = s.trim();
    if (trimmed.length >= 10 && trimmed.length <= 80) {
      phraseCount[trimmed] = (phraseCount[trimmed] || 0) + 1;
    }
  }
  return Object.entries(phraseCount)
    .filter(([, count]) => count >= 3)
    .map(([phrase, count]) => ({ phrase: phrase.substring(0, 50), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
}

// ─── メインチェック ──────────────────────────────
function auditFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const relPath = path.relative(SKILLS_ROOT, filePath);
  const { category, skill, type } = categorizeFile(filePath);
  const charCount = content.length;

  const result = {
    file: relPath,
    category,
    skill,
    type,
    charCount,
    errors: [],   // P0/P1
    warnings: [], // P2
    info: [],     // P3
    metrics: {},
  };

  // ── サイズチェック ──
  if (type === 'guide') {
    if (charCount < MIN_CHARS) {
      result.errors.push(`サイズ不足: ${charCount.toLocaleString()}字 (最低${MIN_CHARS.toLocaleString()}字)`);
    }
    if (charCount < 50000) {
      result.warnings.push(`目標未達: ${charCount.toLocaleString()}字 (目標50,000字)`);
    }
  }

  // ── 必須セクション（guideファイルのみチェック） ──
  const missingSections = [];
  if (type === 'guide') {
    for (const { pattern, label } of REQUIRED_SECTIONS) {
      if (!pattern.test(content)) {
        missingSections.push(label);
      }
    }
    if (missingSections.length > 0) {
      result.errors.push(`必須セクション欠落: ${missingSections.join(', ')}`);
    }
  }
  result.metrics.missingSections = missingSections;

  // ── 構造セクション ──
  const missingStructural = [];
  for (const { pattern, label } of STRUCTURAL_SECTIONS) {
    if (!pattern.test(content)) {
      missingStructural.push(label);
    }
  }
  if (missingStructural.length > 0 && type === 'guide') {
    result.warnings.push(`推奨構造セクション欠落: ${missingStructural.join(', ')}`);
  }
  result.metrics.missingStructural = missingStructural;

  // ── コードブロック（プログラミング関連カテゴリのみ） ──
  const codeBlockCount = countCodeBlocks(content);
  result.metrics.codeBlocks = codeBlockCount;
  const codeRequiredCategories = [
    '01-cs-fundamentals', '02-programming', '03-software-design',
    '04-web-and-network', '05-infrastructure', '06-data-and-security', '07-ai'
  ];
  if (type === 'guide' && codeRequiredCategories.includes(category) && codeBlockCount < MIN_CODE_BLOCKS) {
    result.errors.push(`コードブロック不足: ${codeBlockCount}個 (最低${MIN_CODE_BLOCKS}個)`);
  }

  // ── 表 ──
  const tableCount = countTables(content);
  result.metrics.tables = tableCount;
  if (type === 'guide' && tableCount < MIN_TABLES) {
    result.warnings.push(`比較表不足: ${tableCount}個 (推奨${MIN_TABLES}個以上)`);
  }

  // ── 図解 ──
  const diagramCount = countDiagrams(content);
  result.metrics.diagrams = diagramCount;
  if (type === 'guide' && diagramCount < MIN_DIAGRAMS) {
    result.info.push(`図解/ダイアグラム: ${diagramCount}個 (推奨${MIN_DIAGRAMS}個以上)`);
  }

  // ── FAQ ──
  const faqCount = countFAQ(content);
  result.metrics.faq = faqCount;
  if (type === 'guide' && faqCount < MIN_FAQ) {
    result.warnings.push(`FAQ不足: ${faqCount}個 (最低${MIN_FAQ}個)`);
  }

  // ── 演習 ──
  const exerciseCount = countExercises(content);
  result.metrics.exercises = exerciseCount;
  if (type === 'guide' && exerciseCount < MIN_EXERCISES) {
    result.warnings.push(`演習不足: ${exerciseCount}個 (最低${MIN_EXERCISES}個)`);
  }

  // ── 参考文献 ──
  const refCount = countReferences(content);
  result.metrics.references = refCount;
  if (type === 'guide' && refCount < MIN_REFERENCES) {
    result.warnings.push(`参考文献不足: ${refCount}個 (最低${MIN_REFERENCES}個)`);
  }

  // ── 学習目標 ──
  const objectiveCount = countLearningObjectives(content);
  result.metrics.learningObjectives = objectiveCount;
  if (type === 'guide' && objectiveCount < 3) {
    result.warnings.push(`学習目標不足: ${objectiveCount}個 (最低3個)`);
  }

  // ── 見出し階層 ──
  const headingIssues = checkHeadingHierarchy(content);
  result.metrics.headingIssues = headingIssues.length;
  if (headingIssues.length > 0) {
    result.info.push(...headingIssues.map(i => `見出し: ${i}`));
  }

  // ── フィラー表現 ──
  const fillers = detectFillers(content);
  result.metrics.fillers = fillers;
  for (const f of fillers) {
    if (f.severity === 'WARNING') {
      result.warnings.push(`フィラー多用: "${f.label}" (${f.count}回)`);
    } else {
      result.info.push(`フィラー: "${f.label}" (${f.count}回)`);
    }
  }

  // ── 古い年号 ──
  const outdatedYears = detectOutdatedYears(content);
  result.metrics.outdatedYears = outdatedYears;
  for (const y of outdatedYears) {
    result.warnings.push(`${y.label}: ${y.matches.join(', ')}`);
  }

  // ── 古いバージョン ──
  const outdatedVersions = detectOutdatedVersions(content);
  result.metrics.outdatedVersions = outdatedVersions;
  for (const v of outdatedVersions) {
    result.warnings.push(`${v.label}: ${v.matches.join(', ')}`);
  }

  // ── 繰り返しフレーズ ──
  const repeats = detectRepeatedPhrases(content);
  result.metrics.repeatedPhrases = repeats.length;
  if (repeats.length > 0) {
    result.info.push(`繰り返しフレーズ ${repeats.length}件: ${repeats.slice(0, 3).map(r => `"${r.phrase}"(${r.count}回)`).join(', ')}`);
  }

  // ── 重要度分類 ──
  result.severity = result.errors.length > 0 ? 'ERROR' : result.warnings.length > 0 ? 'WARNING' : 'OK';

  // ── スコア計算（100点満点） ──
  if (type === 'guide') {
    let score = 100;
    score -= result.errors.length * 15;    // ERROR: -15点/件
    score -= result.warnings.length * 3;   // WARNING: -3点/件
    score -= Math.min(result.info.length, 5) * 1; // INFO: -1点/件 (最大5点)
    result.score = Math.max(0, Math.min(100, score));
  }

  return result;
}

// ─── 集計 & 出力 ──────────────────────────────────
function generateSummary(results) {
  const categories = {};
  let totalErrors = 0;
  let totalWarnings = 0;
  let totalInfo = 0;

  for (const r of results) {
    if (!categories[r.category]) {
      categories[r.category] = { files: 0, errors: 0, warnings: 0, info: 0, errorFiles: [], warningFiles: [] };
    }
    const cat = categories[r.category];
    cat.files++;
    cat.errors += r.errors.length;
    cat.warnings += r.warnings.length;
    cat.info += r.info.length;
    totalErrors += r.errors.length;
    totalWarnings += r.warnings.length;
    totalInfo += r.info.length;

    if (r.errors.length > 0) cat.errorFiles.push({ file: r.file, errors: r.errors });
    if (r.warnings.length > 0) cat.warningFiles.push({ file: r.file, warnings: r.warnings });
  }

  return { categories, totalFiles: results.length, totalErrors, totalWarnings, totalInfo };
}

function writeMarkdownSummary(summary, results, outputPath) {
  const lines = [
    '# 品質監査レポート',
    '',
    `> 実行日時: ${new Date().toISOString().replace('T', ' ').substring(0, 19)}`,
    '',
    '## 全体サマリー',
    '',
    `| 項目 | 数値 |`,
    `|------|------|`,
    `| チェックファイル数 | ${summary.totalFiles} |`,
    `| ERROR (P0/P1) | ${summary.totalErrors} |`,
    `| WARNING (P2) | ${summary.totalWarnings} |`,
    `| INFO (P3) | ${summary.totalInfo} |`,
    '',
    '---',
    '',
    '## カテゴリ別サマリー',
    '',
    '| カテゴリ | ファイル数 | ERROR | WARNING | INFO |',
    '|---------|----------|-------|---------|------|',
  ];

  for (const [cat, data] of Object.entries(summary.categories).sort()) {
    lines.push(`| ${cat} | ${data.files} | ${data.errors} | ${data.warnings} | ${data.info} |`);
  }

  lines.push('', '---', '', '## ERROR詳細 (P0/P1)', '');

  const errorResults = results.filter(r => r.errors.length > 0).sort((a, b) => b.errors.length - a.errors.length);
  if (errorResults.length === 0) {
    lines.push('ERROR なし');
  } else {
    for (const r of errorResults) {
      lines.push(`### ${r.file}`);
      lines.push('');
      for (const e of r.errors) {
        lines.push(`- **ERROR**: ${e}`);
      }
      lines.push('');
    }
  }

  lines.push('---', '', '## WARNING詳細 (P2)', '');

  const warningResults = results.filter(r => r.warnings.length > 0).sort((a, b) => b.warnings.length - a.warnings.length);
  if (warningResults.length === 0) {
    lines.push('WARNING なし');
  } else {
    // カテゴリ別にグループ化
    const grouped = {};
    for (const r of warningResults) {
      if (!grouped[r.category]) grouped[r.category] = [];
      grouped[r.category].push(r);
    }
    for (const [cat, catResults] of Object.entries(grouped).sort()) {
      lines.push(`### ${cat}`, '');
      for (const r of catResults) {
        lines.push(`**${r.file}**`);
        for (const w of r.warnings) {
          lines.push(`- ${w}`);
        }
        lines.push('');
      }
    }
  }

  // メトリクス集計
  const guideResults = results.filter(r => r.type === 'guide');
  if (guideResults.length > 0) {
    lines.push('---', '', '## メトリクス集計 (ガイドファイルのみ)', '');

    const avgChars = Math.round(guideResults.reduce((s, r) => s + r.charCount, 0) / guideResults.length);
    const avgCodeBlocks = (guideResults.reduce((s, r) => s + r.metrics.codeBlocks, 0) / guideResults.length).toFixed(1);
    const avgTables = (guideResults.reduce((s, r) => s + r.metrics.tables, 0) / guideResults.length).toFixed(1);

    lines.push(
      '| メトリクス | 平均 | 最小 | 最大 |',
      '|----------|------|------|------|',
      `| 文字数 | ${avgChars.toLocaleString()} | ${Math.min(...guideResults.map(r => r.charCount)).toLocaleString()} | ${Math.max(...guideResults.map(r => r.charCount)).toLocaleString()} |`,
      `| コードブロック | ${avgCodeBlocks} | ${Math.min(...guideResults.map(r => r.metrics.codeBlocks))} | ${Math.max(...guideResults.map(r => r.metrics.codeBlocks))} |`,
      `| 表 | ${avgTables} | ${Math.min(...guideResults.map(r => r.metrics.tables))} | ${Math.max(...guideResults.map(r => r.metrics.tables))} |`,
      ''
    );
  }

  fs.writeFileSync(outputPath, lines.join('\n'), 'utf-8');
}

// ─── メイン ──────────────────────────────────────
function main() {
  const args = process.argv.slice(2);
  const jsonOnly = args.includes('--json-only');
  const targetCategory = args.find(a => !a.startsWith('--'));

  let targetDir = SKILLS_ROOT;
  if (targetCategory) {
    targetDir = path.join(SKILLS_ROOT, targetCategory);
    if (!fs.existsSync(targetDir)) {
      console.error(`カテゴリが見つかりません: ${targetCategory}`);
      process.exit(1);
    }
  }

  console.log(`📋 品質監査を開始します...`);
  console.log(`   対象: ${targetCategory || '全カテゴリ'}`);

  const files = getAllMarkdownFiles(targetDir);
  console.log(`   ファイル数: ${files.length}`);

  const results = [];
  let processed = 0;
  for (const file of files) {
    results.push(auditFile(file));
    processed++;
    if (processed % 100 === 0) {
      console.log(`   処理中... ${processed}/${files.length}`);
    }
  }

  const summary = generateSummary(results);

  // JSON出力
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const jsonPath = path.join(OUTPUT_DIR, 'quality-audit.json');
  fs.writeFileSync(jsonPath, JSON.stringify({ summary, results }, null, 2), 'utf-8');
  console.log(`\n✅ JSON出力: ${path.relative(SKILLS_ROOT, jsonPath)}`);

  // Markdown出力
  if (!jsonOnly) {
    const mdPath = path.join(OUTPUT_DIR, 'quality-audit-summary.md');
    writeMarkdownSummary(summary, results, mdPath);
    console.log(`✅ Markdown出力: ${path.relative(SKILLS_ROOT, mdPath)}`);
  }

  // スコア計算
  const guideResults = results.filter(r => r.type === 'guide' && r.score !== undefined);
  const avgScore = guideResults.length > 0
    ? (guideResults.reduce((s, r) => s + r.score, 0) / guideResults.length).toFixed(1)
    : 'N/A';
  const minScore = guideResults.length > 0
    ? Math.min(...guideResults.map(r => r.score))
    : 'N/A';
  const scoreBelow90 = guideResults.filter(r => r.score < 90).length;

  // コンソールサマリー
  console.log(`\n📊 結果サマリー:`);
  console.log(`   ファイル数: ${summary.totalFiles}`);
  console.log(`   ERROR (P0/P1): ${summary.totalErrors}`);
  console.log(`   WARNING (P2):  ${summary.totalWarnings}`);
  console.log(`   INFO (P3):     ${summary.totalInfo}`);
  console.log(`   📈 平均スコア: ${avgScore}/100`);
  console.log(`   📉 最低スコア: ${minScore}/100`);
  console.log(`   ⚠️  90点未満: ${scoreBelow90}件`);

  for (const [cat, data] of Object.entries(summary.categories).sort()) {
    console.log(`   ${cat}: ${data.files} files, ${data.errors} errors, ${data.warnings} warnings`);
  }

  console.log('\n✨ 監査完了');

  // エラーがあれば非ゼロで終了
  if (summary.totalErrors > 0) process.exit(1);
}

main();
