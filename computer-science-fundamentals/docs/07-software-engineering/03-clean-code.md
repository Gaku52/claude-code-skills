# クリーンコード

> コードは書く時間より読む時間の方が10倍長い。読みやすいコードは正しいコードへの最短距離である。

## この章で学ぶこと

- [ ] 良い命名の原則を身につける
- [ ] 関数設計の原則を理解する
- [ ] コードの臭い（Code Smell）を認識できる

---

## 1. 命名

```python
# ❌ 悪い命名
d = 30          # 何の日数？
lst = []        # 何のリスト？
def proc(x):    # 何の処理？

# ✅ 良い命名
trial_period_days = 30
active_users = []
def calculate_monthly_revenue(transactions):

# 命名の原則:
# 1. 意図を明確に: is_active, has_permission, should_retry
# 2. 発音可能に: ❌ genymdhms → ✅ generation_timestamp
# 3. 検索可能に: ❌ 7 → ✅ MAX_RETRY_COUNT = 7
# 4. スコープに比例: ループ変数は短く(i)、グローバルは長く
# 5. 一貫性: get/fetch/retrieve を混在させない
```

---

## 2. 関数設計

```python
# 原則: 1関数1責任、短く、引数は少なく

# ❌ 長い関数（複数の責任）
def process_order(order):
    # バリデーション(20行)...
    # 在庫確認(15行)...
    # 決済処理(25行)...
    # メール送信(10行)...
    # ログ記録(5行)...
    pass  # 75行の巨大関数

# ✅ 分割された関数
def process_order(order):
    validate_order(order)
    check_inventory(order.items)
    charge_payment(order.payment)
    send_confirmation_email(order.customer)
    log_order(order)

# 引数の原則:
# 0個（ニラディック）: 理想的
# 1個（モナディック）: 良い
# 2個（ダイアディック）: 許容
# 3個以上: オブジェクトにまとめることを検討
```

---

## 3. コードの臭い

```
Code Smell（リファクタリングの兆候）:

  ┌──────────────────┬──────────────────────────────┐
  │ 臭い             │ 対策                         │
  ├──────────────────┼──────────────────────────────┤
  │ 長いメソッド      │ メソッド抽出                  │
  │ 大きなクラス      │ クラス分割                    │
  │ 重複コード        │ 共通関数に抽出                │
  │ 長い引数リスト    │ パラメータオブジェクト         │
  │ フラグ引数        │ 関数を分割                    │
  │ コメントが必要    │ コードを自己説明的に           │
  │ 深いネスト       │ 早期リターン、ガード節         │
  │ マジックナンバー  │ 名前付き定数                  │
  └──────────────────┴──────────────────────────────┘

  早期リターン:
  # ❌ 深いネスト
  def process(user):
      if user:
          if user.is_active:
              if user.has_permission:
                  # 処理...

  # ✅ ガード節
  def process(user):
      if not user: return
      if not user.is_active: return
      if not user.has_permission: return
      # 処理...
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 命名 | 意図を明確に。検索可能に。一貫性を保つ |
| 関数 | 小さく。1つの責任。引数は少なく |
| コードの臭い | 長い関数、重複、深いネスト → リファクタリング |
| 原則 | DRY, KISS, YAGNI |

---

## 次に読むべきガイド
→ [[04-system-design-basics.md]] — システム設計入門

---

## 参考文献
1. Martin, R. C. "Clean Code." Prentice Hall, 2008.
2. Fowler, M. "Refactoring." 2nd Edition, Addison-Wesley, 2018.
