#!/usr/bin/env node
// Phase 2 進捗レポートスクリプト
const fs = require('fs');
const path = require('path');

const auditPath = path.join(__dirname, '..', 'REVIEW_RESULTS', 'quality-audit.json');
const data = JSON.parse(fs.readFileSync(auditPath, 'utf8'));

const cats = {};
data.results.forEach(r => {
  const cat = r.file.split('/')[0];
  if (cat.startsWith('.') || cat.startsWith('_') || !cat.includes('-')) return;
  if (!cats[cat]) cats[cat] = { total: 0, errorFree: 0, errors: 0, warnings: 0 };
  cats[cat].total++;
  cats[cat].errors += r.errors.length;
  cats[cat].warnings += r.warnings.length;
  if (r.errors.length === 0) cats[cat].errorFree++;
});

const order = [
  '03-software-design',
  '06-data-and-security',
  '04-web-and-network',
  '02-programming',
  '07-ai',
  '05-infrastructure',
  '01-cs-fundamentals',
  '08-hobby'
];

const statusMap = {
  '03-software-design': '✅ 完了',
  '06-data-and-security': '✅ 完了(98/100)',
  '04-web-and-network': '🔧 修正中',
};

console.log('# Phase 2 進捗レポート');
console.log('');
console.log('| # | カテゴリ | ファイル数 | ERROR | ERRORなし | エラーフリー率 | ステータス |');
console.log('|---|---------|----------|-------|----------|------------|---------|');

let totalFiles = 0, totalErrorFree = 0, totalErrors = 0, totalWarnings = 0;
let completedCats = 0;

order.forEach((cat, i) => {
  if (!cats[cat]) return;
  const c = cats[cat];
  totalFiles += c.total;
  totalErrorFree += c.errorFree;
  totalErrors += c.errors;
  totalWarnings += c.warnings;
  const rate = ((c.errorFree / c.total) * 100).toFixed(0);
  const st = statusMap[cat] || '⬜ 未着手';
  if (st.startsWith('✅')) completedCats++;
  console.log(`| ${i + 1} | ${cat} | ${c.total} | ${c.errors} | ${c.errorFree}/${c.total} | ${rate}% | ${st} |`);
});

const overallRate = ((totalErrorFree / totalFiles) * 100).toFixed(1);
const catProgress = ((completedCats / order.length) * 100).toFixed(0);

console.log('');
console.log('## サマリー');
console.log('');
console.log(`- **カテゴリ進捗**: ${completedCats}/8 完了 (${catProgress}%)`);
console.log(`- **全ERROR**: ${totalErrors}`);
console.log(`- **全WARNING**: ${totalWarnings}`);
console.log(`- **ERRORフリーファイル**: ${totalErrorFree}/${totalFiles} (${overallRate}%)`);
console.log(`- **Phase 2 全体進捗率**: 約${Math.round((completedCats * 12.5) + (statusMap['04-web-and-network'] ? 6 : 0))}%`);
console.log('');
console.log('## 完了条件');
console.log('- [ ] 全カテゴリのERROR数 = 0');
console.log('- [ ] 全カテゴリの平均スコア ≥ 90/100');
console.log('- [ ] 壊れた相互参照リンク = 0');
console.log('- [ ] P0/P1問題が全て解決済み');
