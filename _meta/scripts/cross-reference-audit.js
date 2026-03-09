#!/usr/bin/env node

/**
 * cross-reference-audit.js — 相互参照チェック
 *
 * 壊れたリンク、孤立ファイル、双方向参照の欠如を検出。
 * check-references.sh のロジックを Node.js で再実装 + 拡張。
 *
 * 出力: JSON (_meta/REVIEW_RESULTS/cross-reference-audit.json)
 *       Markdown (_meta/REVIEW_RESULTS/cross-reference-audit-summary.md)
 *
 * 使用方法:
 *   node _meta/scripts/cross-reference-audit.js                    # 全件
 *   node _meta/scripts/cross-reference-audit.js 04-web-and-network # カテゴリ指定
 */

const fs = require('fs');
const path = require('path');

const SKILLS_ROOT = path.join(__dirname, '..', '..');
const OUTPUT_DIR = path.join(__dirname, '..', 'REVIEW_RESULTS');

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

// ─── リンク抽出 ──────────────────────────────────
function extractLinks(content, filePath) {
  const links = [];
  const fileDir = path.dirname(filePath);

  // 1. Markdownリンク: [text](path)
  const mdLinkPattern = /\[([^\]]*)\]\(([^)]+)\)/g;
  let match;
  while ((match = mdLinkPattern.exec(content)) !== null) {
    const [, text, href] = match;
    // HTTP/HTTPS URLはスキップ
    if (href.match(/^https?:\/\//)) continue;
    // アンカーのみ (#xxx) はスキップ
    if (href.startsWith('#')) continue;
    // メールリンクはスキップ
    if (href.startsWith('mailto:')) continue;

    const cleanPath = href.split('#')[0]; // フラグメント除去
    if (!cleanPath) continue;

    let absPath;
    if (cleanPath.startsWith('/')) {
      absPath = cleanPath;
    } else {
      absPath = path.resolve(fileDir, cleanPath);
    }

    links.push({
      type: 'markdown',
      text: text.substring(0, 50),
      href,
      absPath,
      line: content.substring(0, match.index).split('\n').length,
    });
  }

  // 2. Skill間参照: [[skill-name/path]] or [[skill-name/path|display]]
  const skillRefPattern = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g;
  while ((match = skillRefPattern.exec(content)) !== null) {
    const refPath = match[1].trim();
    const absPath = path.join(SKILLS_ROOT, refPath);

    links.push({
      type: 'skill-ref',
      text: refPath,
      href: `[[${refPath}]]`,
      absPath,
      line: content.substring(0, match.index).split('\n').length,
    });
  }

  return links;
}

// ─── リンク検証 ──────────────────────────────────
function validateLink(link) {
  const p = link.absPath;

  // ファイルまたはディレクトリが存在するか
  if (fs.existsSync(p)) return { valid: true };

  // Skill参照の場合: ディレクトリ + SKILL.md を検索
  if (link.type === 'skill-ref') {
    if (fs.existsSync(path.join(p, 'SKILL.md')) || fs.existsSync(path.join(p, 'skill.md'))) {
      return { valid: true };
    }
    // .md 拡張子を試行
    if (fs.existsSync(p + '.md')) return { valid: true };
  }

  // Markdownリンクの場合: .md 拡張子補完
  if (link.type === 'markdown') {
    if (!p.endsWith('.md') && fs.existsSync(p + '.md')) return { valid: true };
    // ディレクトリ + README.md
    if (fs.existsSync(path.join(p, 'README.md'))) return { valid: true };
  }

  return { valid: false };
}

// ─── 双方向参照チェック ──────────────────────────
function buildReferenceGraph(allFiles) {
  const graph = {}; // file -> Set of referenced files
  const incoming = {}; // file -> Set of files that reference it

  for (const filePath of allFiles) {
    const relPath = path.relative(SKILLS_ROOT, filePath);
    graph[relPath] = new Set();
    if (!incoming[relPath]) incoming[relPath] = new Set();

    const content = fs.readFileSync(filePath, 'utf-8');
    const links = extractLinks(content, filePath);

    for (const link of links) {
      let targetRel = path.relative(SKILLS_ROOT, link.absPath);
      // .md 拡張子の正規化
      if (!targetRel.endsWith('.md')) {
        if (fs.existsSync(link.absPath + '.md')) {
          targetRel += '.md';
        } else if (fs.existsSync(path.join(link.absPath, 'README.md'))) {
          targetRel = path.join(targetRel, 'README.md');
        }
      }
      // 自己参照はスキップ
      if (targetRel === relPath) continue;

      graph[relPath].add(targetRel);
      if (!incoming[targetRel]) incoming[targetRel] = new Set();
      incoming[targetRel].add(relPath);
    }
  }

  return { graph, incoming };
}

// ─── メインチェック ──────────────────────────────
function auditFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const relPath = path.relative(SKILLS_ROOT, filePath);
  const links = extractLinks(content, filePath);

  const brokenLinks = [];
  const validLinks = [];

  for (const link of links) {
    const result = validateLink(link);
    if (result.valid) {
      validLinks.push(link);
    } else {
      brokenLinks.push({
        ...link,
        absPath: path.relative(SKILLS_ROOT, link.absPath),
      });
    }
  }

  return {
    file: relPath,
    totalLinks: links.length,
    validLinks: validLinks.length,
    brokenLinks,
    hasBrokenLinks: brokenLinks.length > 0,
  };
}

// ─── Markdown出力 ────────────────────────────────
function writeMarkdownSummary(auditResults, orphans, missingBidirectional, outputPath) {
  const brokenCount = auditResults.reduce((s, r) => s + r.brokenLinks.length, 0);
  const filesWithBroken = auditResults.filter(r => r.hasBrokenLinks).length;

  const lines = [
    '# 相互参照監査レポート',
    '',
    `> 実行日時: ${new Date().toISOString().replace('T', ' ').substring(0, 19)}`,
    '',
    '## 全体サマリー',
    '',
    '| 項目 | 数値 |',
    '|------|------|',
    `| チェックファイル数 | ${auditResults.length} |`,
    `| 総リンク数 | ${auditResults.reduce((s, r) => s + r.totalLinks, 0)} |`,
    `| 壊れたリンク数 | ${brokenCount} |`,
    `| 壊れたリンクを含むファイル数 | ${filesWithBroken} |`,
    `| 孤立ファイル数 | ${orphans.length} |`,
    `| 双方向参照欠如ペア数 | ${missingBidirectional.length} |`,
    '',
    '---',
    '',
  ];

  // 壊れたリンク詳細
  lines.push('## 壊れたリンク詳細', '');
  if (brokenCount === 0) {
    lines.push('壊れたリンクはありません。', '');
  } else {
    // カテゴリ別にグループ化
    const grouped = {};
    for (const r of auditResults.filter(r => r.hasBrokenLinks)) {
      const cat = r.file.split('/')[0];
      if (!grouped[cat]) grouped[cat] = [];
      grouped[cat].push(r);
    }
    for (const [cat, catResults] of Object.entries(grouped).sort()) {
      lines.push(`### ${cat}`, '');
      for (const r of catResults) {
        lines.push(`**${r.file}** (${r.brokenLinks.length}件):`);
        for (const bl of r.brokenLinks) {
          lines.push(`- L${bl.line}: \`${bl.href}\` → ファイル未存在`);
        }
        lines.push('');
      }
    }
  }

  // 孤立ファイル
  lines.push('---', '', '## 孤立ファイル（どこからも参照されない）', '');
  if (orphans.length === 0) {
    lines.push('孤立ファイルはありません。', '');
  } else {
    const groupedOrphans = {};
    for (const o of orphans) {
      const cat = o.split('/')[0];
      if (!groupedOrphans[cat]) groupedOrphans[cat] = [];
      groupedOrphans[cat].push(o);
    }
    for (const [cat, files] of Object.entries(groupedOrphans).sort()) {
      lines.push(`### ${cat} (${files.length}件)`, '');
      for (const f of files) {
        lines.push(`- ${f}`);
      }
      lines.push('');
    }
  }

  // 双方向参照の欠如
  lines.push('---', '', '## 双方向参照の欠如', '');
  if (missingBidirectional.length === 0) {
    lines.push('双方向参照の欠如はありません。', '');
  } else {
    lines.push(`${missingBidirectional.length}ペアで片方向のみの参照を検出:`, '');
    for (const { from, to } of missingBidirectional.slice(0, 100)) {
      lines.push(`- ${from} → ${to} （逆方向なし）`);
    }
    if (missingBidirectional.length > 100) {
      lines.push(``, `... 他${missingBidirectional.length - 100}ペア`);
    }
    lines.push('');
  }

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

  console.log(`🔗 相互参照監査を開始します...`);
  console.log(`   対象: ${targetCategory || '全カテゴリ'}`);

  const allFiles = getAllMarkdownFiles(targetDir);
  console.log(`   ファイル数: ${allFiles.length}`);

  // 1. 各ファイルのリンク検証
  console.log(`   リンク検証中...`);
  const auditResults = allFiles.map(f => auditFile(f));

  // 2. 参照グラフ構築 & 孤立ファイル検出
  console.log(`   参照グラフ構築中...`);
  const { graph, incoming } = buildReferenceGraph(allFiles);

  const orphans = [];
  for (const filePath of allFiles) {
    const relPath = path.relative(SKILLS_ROOT, filePath);
    // README.md / SKILL.md はエントリポイントなので孤立対象外
    const baseName = path.basename(filePath);
    if (['README.md', 'SKILL.md'].includes(baseName)) continue;
    // docs/ 配下のガイドファイルのみ対象
    if (!relPath.includes('/docs/')) continue;

    const refs = incoming[relPath];
    if (!refs || refs.size === 0) {
      orphans.push(relPath);
    }
  }

  // 3. 双方向参照チェック（同一Skill内のdocs間のみ）
  console.log(`   双方向参照チェック中...`);
  const missingBidirectional = [];
  for (const [from, targets] of Object.entries(graph)) {
    for (const to of targets) {
      // 同一Skill内のdocsファイル間のみチェック
      const fromParts = from.split('/');
      const toParts = to.split('/');
      if (fromParts.length >= 3 && toParts.length >= 3 &&
          fromParts[0] === toParts[0] && fromParts[1] === toParts[1] &&
          from.includes('/docs/') && to.includes('/docs/')) {
        const reverse = graph[to];
        if (reverse && !reverse.has(from)) {
          missingBidirectional.push({ from, to });
        }
      }
    }
  }

  // 出力
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // JSON
  const jsonPath = path.join(OUTPUT_DIR, 'cross-reference-audit.json');
  const jsonData = {
    summary: {
      totalFiles: auditResults.length,
      totalLinks: auditResults.reduce((s, r) => s + r.totalLinks, 0),
      brokenLinks: auditResults.reduce((s, r) => s + r.brokenLinks.length, 0),
      filesWithBrokenLinks: auditResults.filter(r => r.hasBrokenLinks).length,
      orphanFiles: orphans.length,
      missingBidirectional: missingBidirectional.length,
    },
    brokenLinks: auditResults.filter(r => r.hasBrokenLinks),
    orphans,
    missingBidirectional: missingBidirectional.slice(0, 200),
  };
  fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2), 'utf-8');
  console.log(`\n✅ JSON出力: ${path.relative(SKILLS_ROOT, jsonPath)}`);

  // Markdown
  const mdPath = path.join(OUTPUT_DIR, 'cross-reference-audit-summary.md');
  writeMarkdownSummary(auditResults, orphans, missingBidirectional, mdPath);
  console.log(`✅ Markdown出力: ${path.relative(SKILLS_ROOT, mdPath)}`);

  // コンソールサマリー
  const totalBroken = auditResults.reduce((s, r) => s + r.brokenLinks.length, 0);
  console.log(`\n📊 結果サマリー:`);
  console.log(`   総リンク数: ${auditResults.reduce((s, r) => s + r.totalLinks, 0)}`);
  console.log(`   壊れたリンク: ${totalBroken}`);
  console.log(`   孤立ファイル: ${orphans.length}`);
  console.log(`   双方向参照欠如: ${missingBidirectional.length}ペア`);

  console.log('\n✨ 監査完了');

  if (totalBroken > 0) process.exit(1);
}

main();
