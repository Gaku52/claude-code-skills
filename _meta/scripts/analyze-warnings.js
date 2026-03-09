#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const SKILLS_ROOT = path.resolve(__dirname, '..', '..');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'REVIEW_RESULTS', 'quality-audit.json'), 'utf-8'));
const guides = data.results.filter(r => r.type === 'guide');

// 学習目標不足
const objIssues = guides.filter(r => r.warnings.some(w => w.includes('学習目標不足')));
let hasSection = 0, noSection = 0, hasBullets = 0;
for (const r of objIssues.slice(0, 30)) {
  const fp = path.join(SKILLS_ROOT, r.file);
  if (!fs.existsSync(fp)) continue;
  const content = fs.readFileSync(fp, 'utf-8');
  const section = content.match(/##\s*(?:\d+[\.\s]+)?(?:この章で学ぶこと|学ぶこと|学習目標|概要と学習目標)[\s\S]*?(?=\n## |\n---)/m);
  if (section) {
    const bullets = section[0].match(/^\s*[-*]\s+\S/gm);
    hasSection++;
    if (bullets && bullets.length >= 3) hasBullets++;
  } else {
    noSection++;
  }
}
console.log('=== 学習目標不足 (' + objIssues.length + '件) ===');
console.log('Section exists: ' + hasSection + ' (>=3 bullets: ' + hasBullets + '), No section: ' + noSection);

// 演習不足
const exIssues = guides.filter(r => r.warnings.some(w => w.includes('演習不足')));
let exHasContent = 0;
for (const r of exIssues.slice(0, 20)) {
  const fp = path.join(SKILLS_ROOT, r.file);
  if (!fs.existsSync(fp)) continue;
  const content = fs.readFileSync(fp, 'utf-8');
  // 実際にどんな演習パターンがあるか
  const patterns = content.match(/###\s*(?:演習|練習|Exercise|ハンズオン|実践|Practice|課題|問題)/gi);
  if (patterns && patterns.length > 0) exHasContent++;
}
console.log('\n=== 演習不足 (' + exIssues.length + '件) ===');
console.log('Exercise patterns found (no number): ' + exHasContent + '/20 sampled');

// FAQ不足
const faqIssues = guides.filter(r => r.warnings.some(w => w.includes('FAQ不足')));
let faqHasContent = 0;
for (const r of faqIssues.slice(0, 20)) {
  const fp = path.join(SKILLS_ROOT, r.file);
  if (!fs.existsSync(fp)) continue;
  const content = fs.readFileSync(fp, 'utf-8');
  const faqSection = content.match(/##\s*(?:\d+[\.\s]+)?(?:FAQ|よくある質問)[\s\S]*?(?=\n## [^#]|$)/i);
  if (faqSection) {
    const allH3 = faqSection[0].match(/###\s+.+/g);
    const boldQ = faqSection[0].match(/\*\*Q\d*[:：]/g);
    console.log(r.file.split('/').pop() + ': h3=' + (allH3?allH3.length:0) + ', boldQ=' + (boldQ?boldQ.length:0) + ', counted=' + r.metrics.faq);
    if (allH3 && allH3.length >= 3) faqHasContent++;
  }
}
console.log('FAQ with >=3 items (h3): ' + faqHasContent + '/20 sampled');

// 参考文献不足
const refIssues = guides.filter(r => r.warnings.some(w => w.includes('参考文献不足')));
let refHasContent = 0;
for (const r of refIssues.slice(0, 20)) {
  const fp = path.join(SKILLS_ROOT, r.file);
  if (!fs.existsSync(fp)) continue;
  const content = fs.readFileSync(fp, 'utf-8');
  const refSection = content.match(/##\s*(?:\d+[\.\s]+)?(?:参考文献|参考資料|参考リンク|References)[\s\S]*$/im);
  if (refSection) {
    const allBullets = refSection[0].match(/^\s*[-*\d]+[\.\s]+/gm);
    console.log(r.file.split('/').pop() + ': items=' + (allBullets?allBullets.length:0) + ', counted=' + r.metrics.references);
    if (allBullets && allBullets.length >= 3) refHasContent++;
  }
}
console.log('\nRef with >=3 items: ' + refHasContent + '/20 sampled');

// 比較表不足
const tblIssues = guides.filter(r => r.warnings.some(w => w.includes('比較表不足')));
let tblHasContent = 0;
for (const r of tblIssues.slice(0, 20)) {
  const fp = path.join(SKILLS_ROOT, r.file);
  if (!fs.existsSync(fp)) continue;
  const content = fs.readFileSync(fp, 'utf-8');
  const tables = content.match(/\|[^\n]+\|\n\|[-:\s|]+\|/g);
  console.log(r.file.split('/').pop() + ': tablePatterns=' + (tables?tables.length:0) + ', counted=' + r.metrics.tables);
  if (tables && tables.length >= 2) tblHasContent++;
}
console.log('\nTable with >=2: ' + tblHasContent + '/20 sampled');
