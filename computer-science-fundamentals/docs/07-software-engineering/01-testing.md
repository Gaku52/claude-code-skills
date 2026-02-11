# テスト

> テストのないコードはレガシーコードである。——Michael Feathers

## この章で学ぶこと

- [ ] テストピラミッドの概念を理解する
- [ ] ユニットテストの書き方を知る
- [ ] TDDの流れを説明できる

---

## 1. テストピラミッド

```
テストピラミッド:

         /\
        /  \  E2Eテスト（少数、高コスト、遅い）
       /────\
      /      \  統合テスト（中程度）
     /────────\
    /          \  ユニットテスト（大量、低コスト、高速）
   /────────────\

  ユニットテスト (70%):
  - 1つの関数/クラスを単独でテスト
  - 高速（ミリ秒）、大量に実行
  - 外部依存はモックで隔離

  統合テスト (20%):
  - 複数のコンポーネントの連携をテスト
  - DB、API、ファイルシステムとの統合
  - やや遅い（秒〜分）

  E2Eテスト (10%):
  - ユーザーの操作をシミュレート
  - ブラウザ操作（Playwright, Cypress）
  - 遅い（分〜十分）、壊れやすい
```

---

## 2. ユニットテスト

```python
# pytest の例
def calculate_tax(price, rate=0.1):
    if price < 0:
        raise ValueError("Price must be non-negative")
    return round(price * rate, 2)

# テスト
def test_calculate_tax_normal():
    assert calculate_tax(100) == 10.0

def test_calculate_tax_custom_rate():
    assert calculate_tax(100, 0.08) == 8.0

def test_calculate_tax_zero():
    assert calculate_tax(0) == 0.0

def test_calculate_tax_negative():
    import pytest
    with pytest.raises(ValueError):
        calculate_tax(-100)

# テストの原則:
# 1. AAA パターン: Arrange（準備）→ Act（実行）→ Assert（検証）
# 2. 1テスト1アサーション（理想的には）
# 3. テスト名は「何を」テストするか明確に
# 4. 外部依存はモックで隔離
# 5. テストは独立（順序に依存しない）
```

---

## 3. TDD（テスト駆動開発）

```
TDD のサイクル:

  Red → Green → Refactor

  1. Red:    失敗するテストを書く
  2. Green:  テストを通す最小限のコードを書く
  3. Refactor: コードを整理する（テストは変更しない）
  → 繰り返し

  利点:
  - 設計が使いやすいAPIに自然と導かれる
  - 回帰テストが自動的に蓄積
  - 過剰な実装を防ぐ（YAGNI原則）
  - 変更への自信（テストが安全網）

  注意:
  - 全てにTDDを適用する必要はない
  - UIテスト、探索的プログラミングには不向き
  - テストの保守コストも考慮
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| テストピラミッド | UT多数 > 統合中程度 > E2E少数 |
| ユニットテスト | 関数単位。高速。モックで隔離 |
| TDD | Red→Green→Refactor。設計を駆動 |
| テスト原則 | AAA, 独立性, 明確な命名 |

---

## 次に読むべきガイド
→ [[02-design-patterns.md]] — デザインパターン

---

## 参考文献
1. Beck, K. "Test Driven Development: By Example." Addison-Wesley, 2002.
2. Feathers, M. "Working Effectively with Legacy Code." Prentice Hall, 2004.
