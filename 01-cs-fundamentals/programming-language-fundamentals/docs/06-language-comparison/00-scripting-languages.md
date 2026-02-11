# スクリプト言語比較（Python, Ruby, JavaScript, PHP, Perl）

> スクリプト言語は「素早く書いて、素早く動かす」ことに最適化された言語群。それぞれの哲学・強み・エコシステムを比較する。

## この章で学ぶこと

- [ ] 主要スクリプト言語の特徴と適用領域を把握する
- [ ] 各言語の設計哲学の違いを理解する
- [ ] プロジェクトに応じた言語選択ができる

---

## 1. 比較表

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Python   │ Ruby     │ JS/TS    │ PHP      │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 設計哲学      │ 明示的   │ 開発者の │ Webの    │ Web特化  │
│              │ 読みやすさ│ 幸福     │ 言語     │ 実用性   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 型付け        │ 動的+強い│ 動的+強い│ 動的+弱い│ 動的+弱い│
│              │ (型ヒント)│          │ (TS:静的)│ (型宣言) │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主な用途      │ AI/ML    │ Web      │ フルスタック│ Web     │
│              │ データ   │ スクリプト│ ブラウザ  │ CMS     │
│              │ 自動化   │ DevOps   │ サーバー  │ EC      │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ パッケージ    │ PyPI     │ RubyGems │ npm      │ Packagist│
│ マネージャ    │ pip/uv   │ bundler  │ npm/pnpm │ composer │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主要FW       │ Django   │ Rails    │ Express  │ Laravel  │
│              │ FastAPI  │ Sinatra  │ Next.js  │ Symfony  │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 実行速度      │ 遅い     │ 遅い     │ 速い(V8) │ 中程度   │
│ (相対)        │          │          │          │ (OPcache)│
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 学習コスト    │ 低い     │ 低い     │ 低い     │ 低い     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 求人数        │ 非常に多い│ 中程度   │ 最も多い │ 多い     │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. 設計哲学の違い

### Python

```python
# "There should be one-- and preferably only one --obvious way to do it."
# 1つの正しいやり方があるべき

# Zen of Python（import this）
# Beautiful is better than ugly.
# Explicit is better than implicit.
# Simple is better than complex.
# Readability counts.

# 特徴: インデントで構造化、読みやすさ最優先
def process_data(data: list[dict]) -> list[str]:
    return [
        item["name"].upper()
        for item in data
        if item.get("active", False)
    ]
```

### Ruby

```ruby
# "Ruby is designed to make programmers happy."
# プログラマの幸福のために設計

# 特徴: 表現力豊か、すべてがオブジェクト
5.times { |i| puts i }
[1, 2, 3].select(&:odd?)  # → [1, 3]
"hello".reverse.upcase     # → "OLLEH"

# Convention over Configuration（Rails哲学）
```

### JavaScript / TypeScript

```typescript
// Web のユビキタス言語
// ブラウザからサーバーまで1言語で

// TypeScript: JavaScript + 型安全性
interface User {
    name: string;
    age: number;
}

const greet = (user: User): string => `Hello, ${user.name}!`;

// 特徴: イベント駆動、非同期が自然
const data = await fetch("/api").then(r => r.json());
```

---

## 3. 適用領域の比較

```
AI / 機械学習           → Python（事実上の唯一の選択肢）
Web バックエンド         → 全て可能。JS/TS, Python, Ruby, PHP
Web フロントエンド       → JavaScript / TypeScript
データ分析              → Python, R
自動化スクリプト         → Python, Bash
DevOps / インフラ        → Python, Go（Ruby も歴史的に）
CMS / EC               → PHP（WordPress, Shopify）
スタートアップ MVP      → Ruby on Rails, Next.js
```

---

## まとめ

| 言語 | 一言で表すなら | 最強の領域 |
|------|-------------|----------|
| Python | 万能の優等生 | AI/ML/データ |
| Ruby | 開発者の幸福 | Web（Rails） |
| JavaScript | Webの支配者 | フルスタックWeb |
| TypeScript | JS+型安全 | 大規模Web開発 |
| PHP | Web特化の実用家 | CMS/EC |

---

## 次に読むべきガイド
→ [[01-systems-languages.md]] — システム言語比較

---

## 参考文献
1. "Stack Overflow Developer Survey 2024." stackoverflow.com.
