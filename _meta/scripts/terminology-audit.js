#!/usr/bin/env node

/**
 * terminology-audit.js — 用語一貫性チェック
 *
 * 同一概念の表記ゆれ、日英混在の不統一を検出。
 *
 * 出力: JSON (_meta/REVIEW_RESULTS/terminology-audit.json)
 *       Markdown (_meta/REVIEW_RESULTS/terminology-audit-summary.md)
 *
 * 使用方法:
 *   node _meta/scripts/terminology-audit.js                    # 全件
 *   node _meta/scripts/terminology-audit.js 04-web-and-network # カテゴリ指定
 */

const fs = require('fs');
const path = require('path');

const SKILLS_ROOT = path.join(__dirname, '..', '..');
const OUTPUT_DIR = path.join(__dirname, '..', 'REVIEW_RESULTS');

// ─── 表記ゆれルール ─────────────────────────────
// [preferred, ...alternatives] 形式
const TERMINOLOGY_RULES = [
  // プログラミング一般
  { preferred: 'インターフェース', alternatives: ['インタフェース', 'インタフェイス', 'インターフェイス'], category: 'カタカナ' },
  { preferred: 'メソッド', alternatives: ['メッソド', 'メゾッド'], category: 'カタカナ' },
  { preferred: 'パラメータ', alternatives: ['パラメタ', 'パラメーター'], category: 'カタカナ' },
  { preferred: 'プロパティ', alternatives: ['プロパティー'], category: 'カタカナ' },
  { preferred: 'セキュリティ', alternatives: ['セキュリティー'], category: 'カタカナ' },
  { preferred: 'カテゴリ', alternatives: ['カテゴリー'], category: 'カタカナ' },
  { preferred: 'メモリ', alternatives: ['メモリー'], category: 'カタカナ' },
  { preferred: 'ディレクトリ', alternatives: ['ディレクトリー'], category: 'カタカナ' },
  { preferred: 'ライブラリ', alternatives: ['ライブラリー'], category: 'カタカナ' },
  { preferred: 'リポジトリ', alternatives: ['リポジトリー', 'レポジトリ', 'レポジトリー'], category: 'カタカナ' },
  { preferred: 'サーバー', alternatives: ['サーバ(?!ー)'], category: 'カタカナ', isRegex: true },
  { preferred: 'コンテナ', alternatives: ['コンテナー'], category: 'カタカナ' },
  { preferred: 'ユーザー', alternatives: ['ユーザ(?!ー)'], category: 'カタカナ', isRegex: true },
  { preferred: 'ブラウザ', alternatives: ['ブラウザー'], category: 'カタカナ' },
  { preferred: 'ハンドラ', alternatives: ['ハンドラー'], category: 'カタカナ' },
  { preferred: 'コンパイラ', alternatives: ['コンパイラー'], category: 'カタカナ' },
  { preferred: 'レイヤ', alternatives: ['レイヤー'], category: 'カタカナ' },
  { preferred: 'マネージャ', alternatives: ['マネージャー', 'マネジャ(?!ー)', 'マネジャー'], category: 'カタカナ', isRegex: true },
  { preferred: 'プロバイダ', alternatives: ['プロバイダー'], category: 'カタカナ' },

  // 技術用語 日英
  { preferred: 'データベース', alternatives: ['DB', 'ＤＢ'], category: '日英混在', contextDependent: true },
  { preferred: 'API', alternatives: ['ＡＰＩ'], category: '全角英数' },
  { preferred: 'HTTP', alternatives: ['ＨＴＴＰ', 'ｈｔｔｐ'], category: '全角英数' },
  { preferred: 'URL', alternatives: ['ＵＲＬ', 'ｕｒｌ'], category: '全角英数' },
  { preferred: 'CSS', alternatives: ['ＣＳＳ'], category: '全角英数' },
  { preferred: 'HTML', alternatives: ['ＨＴＭＬ'], category: '全角英数' },
  { preferred: 'JSON', alternatives: ['ＪＳＯＮ'], category: '全角英数' },

  // 動詞・表現
  { preferred: 'できる', alternatives: ['出来る'], category: '漢字ひらがな' },
  { preferred: 'ため', alternatives: ['為'], category: '漢字ひらがな', contextDependent: true },
  { preferred: 'すべて', alternatives: ['全て', '総て'], category: '漢字ひらがな' },
  { preferred: 'さまざま', alternatives: ['様々'], category: '漢字ひらがな', contextDependent: true },

  // アーキテクチャ
  { preferred: 'マイクロサービス', alternatives: ['マイクロ・サービス', 'Microservices', 'microservices'], category: '技術用語', contextDependent: true },
  { preferred: 'ミドルウェア', alternatives: ['ミドルウエア'], category: 'カタカナ' },
  { preferred: 'フレームワーク', alternatives: ['フレイムワーク'], category: 'カタカナ' },
  { preferred: 'デプロイ', alternatives: ['デプロイメント', 'ディプロイ'], category: 'カタカナ' },
  { preferred: 'アーキテクチャ', alternatives: ['アーキテクチャー'], category: 'カタカナ' },

  // 認証・セキュリティ
  { preferred: '認証', alternatives: ['オーセンティケーション'], category: '日英混在', contextDependent: true },
  { preferred: '認可', alternatives: ['オーソライゼーション'], category: '日英混在', contextDependent: true },
  { preferred: '暗号化', alternatives: ['エンクリプション'], category: '日英混在', contextDependent: true },
  { preferred: '脆弱性', alternatives: ['バルネラビリティ'], category: '日英混在', contextDependent: true },
];

// 全角英数字の検出パターン
const FULLWIDTH_PATTERN = /[Ａ-Ｚａ-ｚ０-９]/g;

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

function isInsideCodeBlock(content, position) {
  // position以前のコードブロック開閉を数えて、コードブロック内かどうか判定
  const before = content.substring(0, position);
  const ticks = before.match(/```/g);
  return ticks ? ticks.length % 2 !== 0 : false;
}

// ─── メインチェック ──────────────────────────────
function auditFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const relPath = path.relative(SKILLS_ROOT, filePath);
  const issues = [];

  // コードブロック外のテキストのみ対象
  const textContent = content.replace(/```[\s\S]*?```/g, '');

  // 1. 表記ゆれチェック
  for (const rule of TERMINOLOGY_RULES) {
    for (const alt of rule.alternatives) {
      const regex = rule.isRegex
        ? new RegExp(alt, 'g')
        : new RegExp(alt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
      const matches = textContent.match(regex);
      if (matches && matches.length > 0) {
        // preferredも使われているか確認
        const prefRegex = new RegExp(rule.preferred.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
        const prefMatches = textContent.match(prefRegex);
        const prefCount = prefMatches ? prefMatches.length : 0;

        issues.push({
          type: 'terminology',
          category: rule.category,
          found: alt,
          preferred: rule.preferred,
          count: matches.length,
          preferredCount: prefCount,
          contextDependent: rule.contextDependent || false,
          severity: rule.contextDependent ? 'INFO' : 'WARNING',
        });
      }
    }
  }

  // 2. 全角英数字チェック
  const fullwidthMatches = textContent.match(FULLWIDTH_PATTERN);
  if (fullwidthMatches && fullwidthMatches.length > 0) {
    const unique = [...new Set(fullwidthMatches)];
    issues.push({
      type: 'fullwidth',
      category: '全角英数',
      found: unique.join(''),
      count: fullwidthMatches.length,
      severity: 'WARNING',
    });
  }

  // 3. 同一ファイル内の表記ゆれ検出（preferred と alternative が混在）
  const mixedUsage = [];
  for (const rule of TERMINOLOGY_RULES) {
    const prefRegex = new RegExp(rule.preferred.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
    const prefCount = (textContent.match(prefRegex) || []).length;

    for (const alt of rule.alternatives) {
      const altRegex = rule.isRegex
        ? new RegExp(alt, 'g')
        : new RegExp(alt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
      const altCount = (textContent.match(altRegex) || []).length;

      if (prefCount > 0 && altCount > 0) {
        mixedUsage.push({
          preferred: rule.preferred,
          alternative: alt,
          preferredCount: prefCount,
          alternativeCount: altCount,
        });
      }
    }
  }

  if (mixedUsage.length > 0) {
    issues.push({
      type: 'mixed-usage',
      category: '表記混在',
      details: mixedUsage,
      count: mixedUsage.length,
      severity: 'WARNING',
    });
  }

  return {
    file: relPath,
    issues,
    issueCount: issues.length,
    warningCount: issues.filter(i => i.severity === 'WARNING').length,
    infoCount: issues.filter(i => i.severity === 'INFO').length,
  };
}

// ─── 全体集計 ────────────────────────────────────
function aggregateResults(results) {
  const termFrequency = {}; // 用語 -> {preferred, alternative, files, totalCount}

  for (const r of results) {
    for (const issue of r.issues) {
      if (issue.type === 'terminology') {
        const key = `${issue.preferred}←${issue.found}`;
        if (!termFrequency[key]) {
          termFrequency[key] = {
            preferred: issue.preferred,
            alternative: issue.found,
            category: issue.category,
            files: 0,
            totalCount: 0,
            contextDependent: issue.contextDependent,
          };
        }
        termFrequency[key].files++;
        termFrequency[key].totalCount += issue.count;
      }
    }
  }

  return {
    totalFiles: results.length,
    filesWithIssues: results.filter(r => r.issueCount > 0).length,
    totalWarnings: results.reduce((s, r) => s + r.warningCount, 0),
    totalInfo: results.reduce((s, r) => s + r.infoCount, 0),
    topTermIssues: Object.values(termFrequency)
      .sort((a, b) => b.totalCount - a.totalCount)
      .slice(0, 30),
  };
}

// ─── Markdown出力 ────────────────────────────────
function writeMarkdownSummary(results, aggregate, outputPath) {
  const lines = [
    '# 用語一貫性監査レポート',
    '',
    `> 実行日時: ${new Date().toISOString().replace('T', ' ').substring(0, 19)}`,
    '',
    '## 全体サマリー',
    '',
    '| 項目 | 数値 |',
    '|------|------|',
    `| チェックファイル数 | ${aggregate.totalFiles} |`,
    `| 問題ありファイル数 | ${aggregate.filesWithIssues} |`,
    `| WARNING数 | ${aggregate.totalWarnings} |`,
    `| INFO数 | ${aggregate.totalInfo} |`,
    '',
    '---',
    '',
    '## 頻出用語問題 (上位30件)',
    '',
    '| 推奨表記 | 検出表記 | カテゴリ | ファイル数 | 出現回数 | 文脈依存 |',
    '|---------|---------|---------|----------|---------|---------|',
  ];

  for (const t of aggregate.topTermIssues) {
    lines.push(
      `| ${t.preferred} | ${t.alternative} | ${t.category} | ${t.files} | ${t.totalCount} | ${t.contextDependent ? '○' : ''} |`
    );
  }

  lines.push('', '---', '', '## カテゴリ別問題分布', '');

  // カテゴリ別集計
  const catStats = {};
  for (const r of results) {
    const cat = r.file.split('/')[0];
    if (!catStats[cat]) catStats[cat] = { files: 0, issues: 0, warnings: 0 };
    catStats[cat].files++;
    catStats[cat].issues += r.issueCount;
    catStats[cat].warnings += r.warningCount;
  }

  lines.push(
    '| カテゴリ | ファイル数 | 問題数 | WARNING数 |',
    '|---------|----------|-------|----------|'
  );
  for (const [cat, stats] of Object.entries(catStats).sort()) {
    lines.push(`| ${cat} | ${stats.files} | ${stats.issues} | ${stats.warnings} |`);
  }

  // WARNINGの多いファイルTop20
  lines.push('', '---', '', '## WARNING数の多いファイル (上位20件)', '');

  const topWarning = results
    .filter(r => r.warningCount > 0)
    .sort((a, b) => b.warningCount - a.warningCount)
    .slice(0, 20);

  if (topWarning.length === 0) {
    lines.push('WARNINGなし', '');
  } else {
    lines.push('| ファイル | WARNING | INFO |', '|---------|---------|------|');
    for (const r of topWarning) {
      lines.push(`| ${r.file} | ${r.warningCount} | ${r.infoCount} |`);
    }
    lines.push('');
  }

  // 表記ゆれ正規化マッピング（修正用参照テーブル）
  lines.push('---', '', '## 正規化マッピング表（修正時参照）', '');
  lines.push('| 推奨表記 | 非推奨表記 | カテゴリ |', '|---------|----------|---------|');
  for (const rule of TERMINOLOGY_RULES.filter(r => !r.contextDependent)) {
    lines.push(`| ${rule.preferred} | ${rule.alternatives.join(', ')} | ${rule.category} |`);
  }
  lines.push('');

  fs.writeFileSync(outputPath, lines.join('\n'), 'utf-8');
}

// ─── メイン ──────────────────────────────────────
function main() {
  const args = process.argv.slice(2);
  const targetCategory = args.find(a => !a.startsWith('--'));

  let targetDir = SKILLS_ROOT;
  if (targetCategory) {
    targetDir = path.join(SKILLS_ROOT, targetCategory);
    if (!fs.existsSync(targetDir)) {
      console.error(`カテゴリが見つかりません: ${targetCategory}`);
      process.exit(1);
    }
  }

  console.log(`📝 用語一貫性監査を開始します...`);
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

  const aggregate = aggregateResults(results);

  // 出力
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // JSON
  const jsonPath = path.join(OUTPUT_DIR, 'terminology-audit.json');
  fs.writeFileSync(jsonPath, JSON.stringify({ aggregate, results }, null, 2), 'utf-8');
  console.log(`\n✅ JSON出力: ${path.relative(SKILLS_ROOT, jsonPath)}`);

  // Markdown
  const mdPath = path.join(OUTPUT_DIR, 'terminology-audit-summary.md');
  writeMarkdownSummary(results, aggregate, mdPath);
  console.log(`✅ Markdown出力: ${path.relative(SKILLS_ROOT, mdPath)}`);

  // コンソールサマリー
  console.log(`\n📊 結果サマリー:`);
  console.log(`   ファイル数: ${aggregate.totalFiles}`);
  console.log(`   問題ありファイル: ${aggregate.filesWithIssues}`);
  console.log(`   WARNING: ${aggregate.totalWarnings}`);
  console.log(`   INFO: ${aggregate.totalInfo}`);

  if (aggregate.topTermIssues.length > 0) {
    console.log(`\n   頻出用語問題 Top5:`);
    for (const t of aggregate.topTermIssues.slice(0, 5)) {
      console.log(`   - "${t.alternative}" → "${t.preferred}" (${t.totalCount}件 / ${t.files}ファイル)`);
    }
  }

  console.log('\n✨ 監査完了');
}

main();
