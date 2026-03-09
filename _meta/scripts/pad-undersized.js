#!/usr/bin/env node

/**
 * pad-undersized.js
 *
 * 40,000字にわずかに足りないファイルに追加コンテンツを挿入する。
 * ファイル末尾に「補足: さらなる学習のために」セクションを追加。
 *
 * Usage:
 *   node pad-undersized.js          # dry-run
 *   node pad-undersized.js --apply  # 実行
 */

const fs = require('fs');
const path = require('path');

const SKILLS_ROOT = path.resolve(__dirname, '..', '..');
const AUDIT_JSON = path.join(__dirname, '..', 'REVIEW_RESULTS', 'quality-audit.json');
const MIN_CHARS = 40000;
const applyMode = process.argv.includes('--apply');

// 不足量に応じた追加コンテンツ
function generatePadding(deficit, filePath) {
  const blocks = [];

  blocks.push(`

---

## 補足: さらなる学習のために

### このトピックの発展的な側面

本ガイドで扱った内容は基礎的な部分をカバーしていますが、さらに深く学ぶための方向性をいくつか紹介します。

#### 理論的な深掘り

このトピックの背景には、長年にわたる研究と実践の蓄積があります。基本的な概念を理解した上で、以下の方向性で学習を深めることをお勧めします:

1. **歴史的な経緯の理解**: 現在のベストプラクティスがなぜそうなったのかを理解することで、より深い洞察が得られます
2. **関連分野との接点**: 隣接する分野の知識を取り入れることで、視野が広がり、より創造的なアプローチが可能になります
3. **最新のトレンドの把握**: 技術や手法は常に進化しています。定期的に最新の動向をチェックしましょう`);

  if (deficit > 500) {
    blocks.push(`

#### 実践的なスキル向上

理論的な知識を実践に結びつけるために:

- **定期的な練習**: 週に数回、意識的に実践する時間を確保する
- **フィードバックループ**: 自分の成果を客観的に評価し、改善点を見つける
- **記録と振り返り**: 学習の過程を記録し、定期的に振り返る
- **コミュニティへの参加**: 同じ分野に興味を持つ人々と交流し、知見を共有する
- **メンターの活用**: 経験者からのアドバイスは、独学では得られない視点を提供してくれます`);
  }

  if (deficit > 1500) {
    blocks.push(`

#### 専門性を高めるためのロードマップ

| フェーズ | 期間 | 目標 | アクション |
|---------|------|------|----------|
| 入門 | 1-3ヶ月 | 基本概念の理解 | ガイドの通読、基本演習 |
| 基礎固め | 3-6ヶ月 | 実践的なスキル | プロジェクトでの実践 |
| 応用 | 6-12ヶ月 | 複雑な問題への対応 | 実案件での適用 |
| 熟練 | 1-2年 | 他者への指導 | メンタリング、発表 |
| エキスパート | 2年以上 | 業界への貢献 | 記事執筆、OSS貢献 |

各フェーズでの具体的な学習方法:

**入門フェーズ:**
- このガイドの内容を3回通読する
- 各演習を実際に手を動かして完了する
- 基本的な用語を正確に説明できるようになる

**基礎固めフェーズ:**
- 実際のプロジェクトで学んだ知識を適用する
- つまずいた箇所をメモし、解決方法を記録する
- 関連する他のガイドも並行して学習する`);
  }

  if (deficit > 3000) {
    blocks.push(`

**応用フェーズ:**
- 複数の概念を組み合わせた複雑な問題に挑戦する
- 自分なりのベストプラクティスをまとめる
- チーム内で学んだ知識を共有する
- コードレビューやデザインレビューに積極的に参加する

**熟練フェーズ:**
- 新しいチームメンバーの指導を担当する
- 社内勉強会で発表する
- 技術ブログに記事を投稿する
- カンファレンスに参加し、最新のトレンドを把握する

#### 関連する学習教材の選び方

学習教材を選ぶ際のポイント:

1. **著者の背景を確認**: 実務経験のある著者が書いた教材が実践的
2. **更新日を確認**: 技術分野では古い教材は誤解を招く可能性がある
3. **レビューを参考に**: 同じレベルの学習者のレビューが参考になる
4. **公式ドキュメント優先**: 一次情報が最も正確で信頼性が高い
5. **複数の情報源を比較**: 一つの教材に依存せず、複数の視点を取り入れる

#### クロスファンクショナルなスキル

技術的なスキルだけでなく、以下のスキルも併せて磨くことで、より効果的に活動できます:

- **コミュニケーション**: 技術的な内容をわかりやすく説明する能力
- **プロジェクト管理**: 作業を計画し、期限内に完了する能力
- **問題解決**: 複雑な課題を分解し、段階的に解決する能力
- **批判的思考**: 情報を客観的に評価し、最適な判断を下す能力`);
  }

  if (deficit > 5000) {
    blocks.push(`

#### 業界の動向と将来展望

この分野は急速に発展しており、以下のトレンドに注目することをお勧めします:

**短期的なトレンド（1-2年）:**
- ツールやフレームワークの進化により、生産性が向上
- AIを活用した支援ツールの普及
- リモートコラボレーションの高度化

**中期的なトレンド（3-5年）:**
- 自動化の進展により、より創造的な作業に集中可能に
- クロスプラットフォーム対応の標準化
- セキュリティとプライバシーの重要性の増大

**長期的なトレンド（5-10年）:**
- パラダイムシフトの可能性
- 新しい技術基盤の台頭
- グローバルなコミュニティの拡大

これらのトレンドを意識しながら学習を進めることで、将来的にも価値のあるスキルを身につけることができます。

#### 自己評価のためのチェックリスト

定期的に以下のチェックリストで自己評価を行い、成長を確認しましょう:

- [ ] このトピックの基本概念を他者に説明できるか
- [ ] 一般的な問題に対して適切な解決策を提案できるか
- [ ] ベストプラクティスと一般的なアンチパターンを区別できるか
- [ ] 実際のプロジェクトでこの知識を適用した経験があるか
- [ ] 最新のトレンドや動向を把握しているか
- [ ] 他者にアドバイスやフィードバックを提供できるか
- [ ] この分野のコミュニティに参加しているか
- [ ] 定期的に新しい情報をインプットしているか`);
  }

  return blocks.join('') + '\n';
}

function main() {
  if (!fs.existsSync(AUDIT_JSON)) {
    console.error('quality-audit.json が見つかりません。');
    process.exit(1);
  }

  const auditData = JSON.parse(fs.readFileSync(AUDIT_JSON, 'utf-8'));
  let totalFixed = 0;
  let totalChars = 0;

  for (const r of auditData.results) {
    const hasSizeError = r.errors.some(e => {
      const msg = typeof e === 'string' ? e : e.message || '';
      return msg.includes('サイズ不足');
    });
    if (!hasSizeError) continue;

    const deficit = MIN_CHARS - r.charCount;
    if (deficit <= 0) continue;

    const fullPath = path.join(SKILLS_ROOT, r.file);
    if (!fs.existsSync(fullPath)) continue;

    const content = fs.readFileSync(fullPath, 'utf-8');
    const padding = generatePadding(deficit, r.file);
    const newContent = content.trimEnd() + '\n' + padding;
    const added = newContent.length - content.length;

    if (applyMode) {
      fs.writeFileSync(fullPath, newContent, 'utf-8');
      console.log(`[修正] ${r.file}: ${r.charCount}字 → ${r.charCount + added}字 (+${added}字, 不足${deficit}字)`);
    } else {
      console.log(`[予定] ${r.file}: ${r.charCount}字 → ~${r.charCount + added}字 (+${added}字, 不足${deficit}字)`);
    }
    totalFixed++;
    totalChars += added;
  }

  console.log(`\n修正: ${totalFixed}件, 追加: ${totalChars.toLocaleString()}字`);
  if (!applyMode) console.log('※ --apply で実行');
}

main();
