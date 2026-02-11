# 正規表現とテキスト処理

> 正規表現はテキスト処理の強力な武器。基本構文からルックアヘッド/ビハインド、名前付きキャプチャ、Unicode 対応、各言語での実装まで、正規表現の全てを解説する。

## このSkillの対象者

- 正規表現を体系的に学びたいエンジニア
- テキスト処理・データクレンジングを効率化したい方
- バリデーション実装を改善したい方

## 前提知識

- 基本的なプログラミング経験
- 文字列操作の基礎知識

## 学習ガイド

### 00-basics — 正規表現の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-basics/00-regex-fundamentals.md]] | メタ文字、量指定子、文字クラス、アンカー |
| 01 | [[docs/00-basics/01-groups-and-captures.md]] | グループ、キャプチャ、バックリファレンス、名前付きグループ |
| 02 | [[docs/00-basics/02-common-patterns.md]] | メール、URL、電話番号、日付のパターン集 |
| 03 | [[docs/00-basics/03-regex-engines.md]] | NFA vs DFA、バックトラック、ReDoS 対策 |

### 01-advanced — 高度な機能

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-advanced/00-lookaround.md]] | Lookahead、Lookbehind、条件パターン |
| 01 | [[docs/01-advanced/01-unicode-and-encoding.md]] | Unicode プロパティ、\p{Script}、国際化対応 |
| 02 | [[docs/01-advanced/02-performance.md]] | パフォーマンス最適化、原子グループ、possessive 量指定子 |
| 03 | [[docs/01-advanced/03-text-processing-tools.md]] | sed、awk、grep、jq、テキスト処理パイプライン |

### 02-languages — 言語別実装

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-languages/00-javascript-regex.md]] | JS RegExp、フラグ、String メソッド、matchAll |
| 01 | [[docs/02-languages/01-python-regex.md]] | re モジュール、パターンコンパイル、置換、分割 |
| 02 | [[docs/02-languages/02-rust-and-go-regex.md]] | Rust regex crate、Go regexp、RE2 |

## クイックリファレンス

```
正規表現チートシート:
  .       — 任意の1文字
  \d \w \s — 数字/英数字/空白
  [abc]   — 文字クラス
  ^  $    — 行頭/行末
  *  +  ? — 0回以上/1回以上/0-1回
  {n,m}   — n〜m回
  (...)   — キャプチャグループ
  (?:...) — 非キャプチャグループ
  (?=...) — 先読み
  (?<=..) — 後読み
  \1      — 後方参照
```

## 参考文献

1. Friedl, J. "Mastering Regular Expressions." O'Reilly, 2006.
2. regular-expressions.info — 総合リファレンス
3. regex101.com — オンラインテスター
