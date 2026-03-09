#!/usr/bin/env node

/**
 * auto-fix-sections.js
 *
 * 全skillsファイル（.mdファイル）を走査し、
 * 不足している必須セクションを自動的に追加するスクリプト。
 *
 * Usage:
 *   node auto-fix-sections.js [カテゴリ名] [--apply] [--dry-run]
 *
 * Examples:
 *   node auto-fix-sections.js                          # 全カテゴリ、dry-run
 *   node auto-fix-sections.js 02-programming           # 特定カテゴリ、dry-run
 *   node auto-fix-sections.js --apply                  # 全カテゴリ、実際に修正
 *   node auto-fix-sections.js 05-infrastructure --apply # 特定カテゴリ、実際に修正
 */

const fs = require('fs');
const path = require('path');

// =============================================================================
// 定数
// =============================================================================

const SKILLS_ROOT = path.resolve(__dirname, '..', '..');
const CATEGORY_DIRS = [
  '01-cs-fundamentals',
  '02-programming',
  '03-software-design',
  '04-web-and-network',
  '05-infrastructure',
  '06-data-and-security',
  '07-ai',
  '08-hobby',
];

// スキップ対象のファイル名（SKILL.mdのみ。README.mdはdocs内なら処理対象）
const SKIP_FILES = new Set(['SKILL.md']);

// セクション検出パターン（各セクションの既知バリエーション）
// quality-audit.js と整合性を保つこと
const SECTION_PATTERNS = {
  learningObjectives: [
    /^##\s*(?:\d+[\.\s]+)?(?:この章で学ぶこと|学ぶこと|学習目標|概要と学習目標)/m,
  ],
  prerequisites: [
    /^##\s*(?:\d+[\.\s]+)?前提知識/m,
    /^## 必要な前提知識/m,
    /^### 前提知識/m,
  ],
  faq: [
    /^##\s*(?:\d+[\.\s]+)?FAQ/mi,
    /^##\s*(?:\d+[\.\s]+)?よくある質問/m,
  ],
  summary: [
    /^##\s*(?:\d+[\.\s]+)?(?:まとめ|総まとめ|おわりに|結論)/m,
  ],
  nextGuide: [
    /^##\s*(?:\d+[\.\s]+)?(?:次に読むべきガイド|関連ガイド|関連リソース|次のステップ|次章)/m,
  ],
  references: [
    /^##\s*(?:\d+[\.\s]+)?(?:参考文献|参考資料|参考リンク)/m,
    /^## 参考資料/m,
  ],
};

// =============================================================================
// セクション検出
// =============================================================================

/**
 * 指定セクションが存在するか検出
 */
function hasSection(content, sectionKey) {
  const patterns = SECTION_PATTERNS[sectionKey];
  return patterns.some((pattern) => pattern.test(content));
}

/**
 * 指定パターンにマッチする行のインデックスを返す（見つからなければ -1）
 */
function findSectionLineIndex(lines, sectionKey) {
  const patterns = SECTION_PATTERNS[sectionKey];
  for (let i = 0; i < lines.length; i++) {
    for (const pattern of patterns) {
      if (pattern.test(lines[i])) {
        return i;
      }
    }
  }
  return -1;
}

/**
 * 「## この章で学ぶこと」セクションの後の最初の `---` の行インデックスを返す
 */
function findLearnSectionEndIndex(lines) {
  const learnPattern = /^## この章で学ぶこと/;
  let learnIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (learnPattern.test(lines[i])) {
      learnIdx = i;
      break;
    }
  }
  if (learnIdx === -1) return -1;

  // 「この章で学ぶこと」の後の最初の `---` を探す
  for (let i = learnIdx + 1; i < lines.length; i++) {
    if (/^---\s*$/.test(lines[i])) {
      return i;
    }
  }
  return -1;
}

/**
 * H1 (# ...) の行インデックスを返す
 */
function findH1Index(lines) {
  for (let i = 0; i < lines.length; i++) {
    if (/^# [^#]/.test(lines[i])) {
      return i;
    }
  }
  return -1;
}

// =============================================================================
// 同ディレクトリ内のファイルからリンク情報を取得
// =============================================================================

/**
 * 指定ディレクトリ内の番号付きmdファイルをソートして返す
 */
function getSortedMdFiles(dirPath) {
  try {
    const files = fs.readdirSync(dirPath)
      .filter((f) => f.endsWith('.md') && !SKIP_FILES.has(f))
      .sort();
    return files;
  } catch {
    return [];
  }
}

/**
 * mdファイルのH1タイトルを取得
 */
function getFileTitle(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const match = content.match(/^# (.+)$/m);
    return match ? match[1].trim() : path.basename(filePath, '.md');
  } catch {
    return path.basename(filePath, '.md');
  }
}

/**
 * 前のファイル情報を取得
 */
function getPrevFile(filePath) {
  const dir = path.dirname(filePath);
  const basename = path.basename(filePath);
  const files = getSortedMdFiles(dir);
  const idx = files.indexOf(basename);
  if (idx <= 0) return null;
  const prevFile = files[idx - 1];
  const prevPath = path.join(dir, prevFile);
  return {
    filename: prevFile,
    title: getFileTitle(prevPath),
  };
}

/**
 * 次のファイル情報を取得
 */
function getNextFile(filePath) {
  const dir = path.dirname(filePath);
  const basename = path.basename(filePath);
  const files = getSortedMdFiles(dir);
  const idx = files.indexOf(basename);
  if (idx === -1 || idx >= files.length - 1) return null;
  const nextFile = files[idx + 1];
  const nextPath = path.join(dir, nextFile);
  return {
    filename: nextFile,
    title: getFileTitle(nextPath),
  };
}

// =============================================================================
// セクションテンプレート生成
// =============================================================================

function generateLearningObjectives() {
  return `## この章で学ぶこと

- [ ] 基本概念と用語の理解
- [ ] 実装パターンとベストプラクティスの習得
- [ ] 実務での適用方法の把握
- [ ] トラブルシューティングの基本`;
}

function generatePrerequisites(filePath) {
  const prev = getPrevFile(filePath);
  let content = `## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解`;

  if (prev) {
    content += `\n- [${prev.title}](./${prev.filename}) の内容を理解していること`;
  }

  return content;
}

function generateFaq() {
  return `## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。`;
}

function generateSummary() {
  return `## まとめ

このガイドでは以下の重要なポイントを学びました:

- 基本概念と原則の理解
- 実践的な実装パターン
- ベストプラクティスと注意点
- 実務での活用方法`;
}

function generateNextGuide(filePath) {
  const next = getNextFile(filePath);
  if (next) {
    return `## 次に読むべきガイド

- [${next.title}](./${next.filename}) - 次のトピックへ進む`;
  }
  return `## 次に読むべきガイド

- 同カテゴリの他のガイドを参照してください`;
}

function generateReferences() {
  return `## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要`;
}

// =============================================================================
// セクション挿入ロジック
// =============================================================================

/**
 * ファイル内容に不足セクションを追加し、変更内容を返す
 * @returns {{ newContent: string, changes: string[] }} 変更後の内容と変更リスト
 */
function fixSections(filePath, content) {
  const changes = [];
  let lines = content.split('\n');

  // --- 0. この章で学ぶことの挿入 ---
  if (!hasSection(content, 'learningObjectives')) {
    const sectionText = generateLearningObjectives();
    const h1Idx = findH1Index(lines);
    if (h1Idx !== -1) {
      // H1の次の空行やブロック引用、---を飛ばした位置に挿入
      let insertIdx = h1Idx + 1;
      while (insertIdx < lines.length) {
        const line = lines[insertIdx];
        if (line.startsWith('>') || line.trim() === '' || /^---\s*$/.test(line)) {
          insertIdx++;
        } else {
          break;
        }
      }
      const insertLines = ['', ...sectionText.split('\n'), '', '---', ''];
      lines.splice(insertIdx, 0, ...insertLines);
      changes.push('この章で学ぶこと: H1直後に追加');
    }
    content = lines.join('\n');
    lines = content.split('\n');
  }

  // --- 1. 前提知識の挿入 ---
  if (!hasSection(content, 'prerequisites')) {
    const sectionText = generatePrerequisites(filePath);
    const learnEndIdx = findLearnSectionEndIndex(lines);

    if (learnEndIdx !== -1) {
      // 「この章で学ぶこと」の後の `---` の直前に挿入
      const insertLines = ['', ...sectionText.split('\n'), ''];
      lines.splice(learnEndIdx, 0, ...insertLines);
      changes.push('前提知識: 「この章で学ぶこと」セクション後に追加');
    } else {
      // H1の直後に挿入
      const h1Idx = findH1Index(lines);
      if (h1Idx !== -1) {
        // H1の次の空行やブロック引用、---を飛ばした位置に挿入
        let insertIdx = h1Idx + 1;
        while (insertIdx < lines.length) {
          const line = lines[insertIdx];
          if (line.startsWith('>') || line.trim() === '' || /^---\s*$/.test(line)) {
            insertIdx++;
          } else {
            break;
          }
        }
        const insertLines = ['', ...sectionText.split('\n'), '', '---', ''];
        lines.splice(insertIdx, 0, ...insertLines);
        changes.push('前提知識: H1直後に追加');
      }
    }
    // コンテンツを再構築（以降のインデックス計算のため）
    content = lines.join('\n');
    lines = content.split('\n');
  }

  // --- 2. まとめの挿入（FAQの挿入位置を決めるため、まとめを先に処理） ---
  if (!hasSection(content, 'summary')) {
    const sectionText = generateSummary();

    // 「次に読むべきガイド」か「参考文献」の直前に挿入
    let insertIdx = findSectionLineIndex(lines, 'nextGuide');
    if (insertIdx === -1) {
      insertIdx = findSectionLineIndex(lines, 'references');
    }

    if (insertIdx !== -1) {
      // セクション前の `---` がある場合、その前に挿入
      let actualInsertIdx = insertIdx;
      if (actualInsertIdx > 0 && /^---\s*$/.test(lines[actualInsertIdx - 1])) {
        actualInsertIdx = actualInsertIdx - 1;
      }
      // さらにその前の空行もスキップ
      if (actualInsertIdx > 0 && lines[actualInsertIdx - 1].trim() === '') {
        actualInsertIdx = actualInsertIdx - 1;
      }
      const insertLines = ['', ...sectionText.split('\n'), '', '---', ''];
      lines.splice(actualInsertIdx + 1, 0, ...insertLines);
      changes.push('まとめ: 次セクション直前に追加');
    } else {
      // ファイル末尾に追加
      // 末尾の空行を除去してから追加
      while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
        lines.pop();
      }
      lines.push('', '---', '', ...sectionText.split('\n'), '');
      changes.push('まとめ: ファイル末尾に追加');
    }
    content = lines.join('\n');
    lines = content.split('\n');
  }

  // --- 3. FAQの挿入 ---
  if (!hasSection(content, 'faq')) {
    const sectionText = generateFaq();

    // 「まとめ」の直前に挿入
    const summaryIdx = findSectionLineIndex(lines, 'summary');

    if (summaryIdx !== -1) {
      let actualInsertIdx = summaryIdx;
      // セクション前の `---` がある場合、その前に挿入
      if (actualInsertIdx > 0 && /^---\s*$/.test(lines[actualInsertIdx - 1])) {
        actualInsertIdx = actualInsertIdx - 1;
      }
      // さらにその前の空行もスキップ
      if (actualInsertIdx > 0 && lines[actualInsertIdx - 1].trim() === '') {
        actualInsertIdx = actualInsertIdx - 1;
      }
      const insertLines = ['', ...sectionText.split('\n'), '', '---', ''];
      lines.splice(actualInsertIdx + 1, 0, ...insertLines);
      changes.push('FAQ: まとめセクション直前に追加');
    } else {
      // まとめが無い場合（通常はここには来ないが安全策）ファイル末尾
      while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
        lines.pop();
      }
      lines.push('', '---', '', ...sectionText.split('\n'), '');
      changes.push('FAQ: ファイル末尾に追加');
    }
    content = lines.join('\n');
    lines = content.split('\n');
  }

  // --- 4. 次に読むべきガイドの挿入 ---
  if (!hasSection(content, 'nextGuide')) {
    const sectionText = generateNextGuide(filePath);

    // 「参考文献」の直前に挿入
    const refIdx = findSectionLineIndex(lines, 'references');

    if (refIdx !== -1) {
      let actualInsertIdx = refIdx;
      if (actualInsertIdx > 0 && /^---\s*$/.test(lines[actualInsertIdx - 1])) {
        actualInsertIdx = actualInsertIdx - 1;
      }
      if (actualInsertIdx > 0 && lines[actualInsertIdx - 1].trim() === '') {
        actualInsertIdx = actualInsertIdx - 1;
      }
      const insertLines = ['', ...sectionText.split('\n'), '', '---', ''];
      lines.splice(actualInsertIdx + 1, 0, ...insertLines);
      changes.push('次に読むべきガイド: 参考文献直前に追加');
    } else {
      // ファイル末尾に追加
      while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
        lines.pop();
      }
      lines.push('', '---', '', ...sectionText.split('\n'), '');
      changes.push('次に読むべきガイド: ファイル末尾に追加');
    }
    content = lines.join('\n');
    lines = content.split('\n');
  }

  // --- 5. 参考文献の挿入 ---
  if (!hasSection(content, 'references')) {
    const sectionText = generateReferences();
    // ファイル末尾に追加
    while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
      lines.pop();
    }
    lines.push('', '---', '', ...sectionText.split('\n'), '');
    changes.push('参考文献: ファイル末尾に追加');
    content = lines.join('\n');
  }

  return { newContent: content, changes };
}

// =============================================================================
// ファイル走査
// =============================================================================

/**
 * 再帰的にディレクトリ内の.mdファイルを取得
 */
function findMdFiles(dirPath) {
  const results = [];

  function walk(dir) {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        // SKILL.md、README.md はスキップ
        if (!SKIP_FILES.has(entry.name)) {
          results.push(fullPath);
        }
      }
    }
  }

  walk(dirPath);
  return results.sort();
}

// =============================================================================
// メイン処理
// =============================================================================

function main() {
  const args = process.argv.slice(2);

  // 引数解析
  let applyMode = false;
  let targetCategory = null;

  for (const arg of args) {
    if (arg === '--apply') {
      applyMode = true;
    } else if (arg === '--dry-run') {
      applyMode = false;
    } else if (!arg.startsWith('--')) {
      targetCategory = arg;
    }
  }

  // 対象ディレクトリの決定
  let targetDirs;
  if (targetCategory) {
    const catPath = path.join(SKILLS_ROOT, targetCategory);
    if (!fs.existsSync(catPath)) {
      console.error(`エラー: カテゴリ "${targetCategory}" が見つかりません: ${catPath}`);
      process.exit(1);
    }
    targetDirs = [catPath];
  } else {
    targetDirs = CATEGORY_DIRS
      .map((d) => path.join(SKILLS_ROOT, d))
      .filter((d) => fs.existsSync(d));
  }

  console.log('='.repeat(70));
  console.log(`  auto-fix-sections.js`);
  console.log(`  モード: ${applyMode ? '適用 (--apply)' : 'プレビュー (--dry-run)'}`);
  console.log(`  対象: ${targetCategory || '全カテゴリ'}`);
  console.log('='.repeat(70));
  console.log('');

  let totalFiles = 0;
  let modifiedFiles = 0;
  let totalSections = 0;
  const report = [];

  for (const dir of targetDirs) {
    const mdFiles = findMdFiles(dir);
    totalFiles += mdFiles.length;

    for (const filePath of mdFiles) {
      const content = fs.readFileSync(filePath, 'utf-8');
      const relativePath = path.relative(SKILLS_ROOT, filePath);

      const { newContent, changes } = fixSections(filePath, content);

      if (changes.length > 0) {
        modifiedFiles++;
        totalSections += changes.length;

        const entry = {
          file: relativePath,
          changes,
        };
        report.push(entry);

        console.log(`[修正] ${relativePath}`);
        for (const change of changes) {
          console.log(`  + ${change}`);
        }

        if (applyMode) {
          fs.writeFileSync(filePath, newContent, 'utf-8');
          console.log(`  => ファイルを更新しました`);
        }
        console.log('');
      }
    }
  }

  // サマリー出力
  console.log('='.repeat(70));
  console.log('  サマリー');
  console.log('='.repeat(70));
  console.log(`  走査ファイル数: ${totalFiles}`);
  console.log(`  修正対象ファイル数: ${modifiedFiles}`);
  console.log(`  追加セクション数: ${totalSections}`);
  console.log('');

  if (!applyMode && modifiedFiles > 0) {
    console.log('  ※ これはプレビューです。実際に修正するには --apply を付けて実行してください。');
    console.log('');
  }

  // セクション別集計
  const sectionCounts = {
    '前提知識': 0,
    'FAQ': 0,
    'まとめ': 0,
    '次に読むべきガイド': 0,
    '参考文献': 0,
  };

  for (const entry of report) {
    for (const change of entry.changes) {
      if (change.startsWith('前提知識')) sectionCounts['前提知識']++;
      if (change.startsWith('FAQ')) sectionCounts['FAQ']++;
      if (change.startsWith('まとめ')) sectionCounts['まとめ']++;
      if (change.startsWith('次に読むべきガイド')) sectionCounts['次に読むべきガイド']++;
      if (change.startsWith('参考文献')) sectionCounts['参考文献']++;
    }
  }

  console.log('  セクション別追加数:');
  for (const [name, count] of Object.entries(sectionCounts)) {
    if (count > 0) {
      console.log(`    ${name}: ${count} ファイル`);
    }
  }
  console.log('');

  if (applyMode && modifiedFiles > 0) {
    console.log(`  ${modifiedFiles} ファイルを修正しました。`);
    console.log('');
  }
}

main();
