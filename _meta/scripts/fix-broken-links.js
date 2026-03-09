#!/usr/bin/env node

/**
 * fix-broken-links.js
 *
 * 壊れた相互参照リンクを修正する。
 * 1. SKILL.md の [[...]] 参照: 実ファイル構造に基づいて修正
 * 2. guide内のマークダウンリンク: 壊れたリンクを削除
 *
 * Usage:
 *   node fix-broken-links.js          # dry-run
 *   node fix-broken-links.js --apply  # 実行
 */

const fs = require('fs');
const path = require('path');

const SKILLS_ROOT = path.resolve(__dirname, '..', '..');
const AUDIT_JSON = path.join(__dirname, '..', 'REVIEW_RESULTS', 'cross-reference-audit.json');
const applyMode = process.argv.includes('--apply');

function main() {
  if (!fs.existsSync(AUDIT_JSON)) {
    console.error('cross-reference-audit.json が見つかりません。');
    process.exit(1);
  }

  const auditData = JSON.parse(fs.readFileSync(AUDIT_JSON, 'utf-8'));
  let totalFixed = 0;
  let totalRemoved = 0;

  for (const entry of auditData.brokenLinks) {
    const filePath = path.join(SKILLS_ROOT, entry.file);
    if (!fs.existsSync(filePath)) continue;

    let content = fs.readFileSync(filePath, 'utf-8');
    let modified = false;
    let fixCount = 0;
    let removeCount = 0;

    for (const bl of entry.brokenLinks) {
      if (bl.type === 'skill-ref') {
        // [[path]] リンクを削除（存在しないファイルへの参照）
        const wikiLink = bl.href; // [[docs/...]]
        if (content.includes(wikiLink)) {
          // 行全体を削除（リスト項目の場合）
          const lines = content.split('\n');
          const newLines = lines.filter(line => !line.includes(wikiLink));
          if (newLines.length < lines.length) {
            content = newLines.join('\n');
            modified = true;
            removeCount++;
          }
        }
      } else if (bl.type === 'markdown') {
        // マークダウンリンク [text](href) → text（リンクを解除）
        const escapedHref = bl.href.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const escapedText = bl.text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const linkPattern = new RegExp(
          `\\[${escapedText}\\]\\(${escapedHref}\\)`,
          'g'
        );
        const before = content;
        content = content.replace(linkPattern, bl.text);
        if (content !== before) {
          modified = true;
          fixCount++;
        }
      }
    }

    if (modified) {
      // 連続する空行を2行までに圧縮
      content = content.replace(/\n{4,}/g, '\n\n\n');

      if (applyMode) {
        fs.writeFileSync(filePath, content, 'utf-8');
      }
      totalFixed += fixCount;
      totalRemoved += removeCount;
      const action = applyMode ? '[修正]' : '[予定]';
      console.log(`${action} ${entry.file}: リンク解除${fixCount}件, 行削除${removeCount}件`);
    }
  }

  console.log(`\n合計: リンク解除${totalFixed}件, 行削除${totalRemoved}件`);
  if (!applyMode) console.log('※ --apply で実行');
}

main();
