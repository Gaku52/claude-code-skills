# 責任ある AI — 公平性・説明可能性・プライバシー

> AI システムが社会に対して公正かつ透明に機能し、個人の権利を守るための技術的手法と組織的フレームワークを学ぶ。

---

## この章で学ぶこと

1. **公平性 (Fairness)** — バイアスの種類を理解し、学習データ・モデル・出力の各段階で公平性を測定・改善する手法
2. **説明可能性 (Explainability)** — ブラックボックスモデルの判断根拠を人間が理解できる形で提示する技術
3. **プライバシー** — 差分プライバシー・連合学習など、個人情報を保護しながら AI を活用する仕組み

---

## 1. 責任ある AI の全体フレームワーク

### 1.1 5 つの柱

```
+------------------------------------------------------------------+
|                     責任ある AI (Responsible AI)                   |
+------------------------------------------------------------------+
|                                                                    |
|  +----------+  +----------+  +----------+  +--------+  +--------+ |
|  | 公平性   |  | 説明     |  | プライ   |  | 安全性 |  | 説明   | |
|  | Fairness |  | 可能性   |  | バシー   |  | Safety |  | 責任   | |
|  |          |  | Explain- |  | Privacy  |  |        |  | Account| |
|  |          |  | ability  |  |          |  |        |  | ability| |
|  +----------+  +----------+  +----------+  +--------+  +--------+ |
|                                                                    |
+------------------------------------------------------------------+
```

### 1.2 バイアスの発生箇所

```
データ収集       前処理        モデル学習      推論/意思決定
+--------+    +--------+    +--------+    +--------+
| 歴史的  |    | サンプ  |    | 目的   |    | 自動化  |
| バイアス | -> | リング  | -> | 関数の  | -> | バイアス |
| 社会的  |    | バイアス |    | バイアス |    | フィード |
| 偏り    |    |         |    |         |    | バック  |
+--------+    +--------+    +--------+    +--------+
   |               |              |              |
   v               v              v              v
 収集段階の      表現の偏り     学習過程での    デプロイ後の
 介入手法       の是正         緩和手法       監視・修正
```

---

## 2. 公平性 (Fairness)

### 2.1 公平性指標の定義と計算

```python
# コード例 1: 主要な公平性指標を計算する
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    二値分類における公平性指標を計算する。

    Parameters:
        y_true: 正解ラベル (0 or 1)
        y_pred: 予測ラベル (0 or 1)
        sensitive_attr: センシティブ属性 (0: 非特権グループ, 1: 特権グループ)
    """
    metrics = {}

    for group_name, group_val in [("privileged", 1), ("unprivileged", 0)]:
        mask = sensitive_attr == group_val
        tn, fp, fn, tp = confusion_matrix(
            y_true[mask], y_pred[mask]
        ).ravel()

        # 各グループの指標
        metrics[f"{group_name}_TPR"] = tp / (tp + fn)  # 真陽性率
        metrics[f"{group_name}_FPR"] = fp / (fp + tn)  # 偽陽性率
        metrics[f"{group_name}_selection_rate"] = (tp + fp) / len(y_true[mask])

    # --- 公平性指標 ---
    # Statistical Parity Difference (SPD): 選択率の差
    metrics["SPD"] = (
        metrics["unprivileged_selection_rate"]
        - metrics["privileged_selection_rate"]
    )

    # Equal Opportunity Difference (EOD): 真陽性率の差
    metrics["EOD"] = (
        metrics["unprivileged_TPR"]
        - metrics["privileged_TPR"]
    )

    # Disparate Impact (DI): 選択率の比
    metrics["DI"] = (
        metrics["unprivileged_selection_rate"]
        / metrics["privileged_selection_rate"]
    )

    return metrics

# 使用例
metrics = calculate_fairness_metrics(y_true, y_pred, gender)
print(f"Statistical Parity Difference: {metrics['SPD']:.3f}")
print(f"Disparate Impact: {metrics['DI']:.3f}")
# DI が 0.8〜1.25 の範囲内なら「4/5ルール」を満たす
```

### 2.2 バイアス緩和手法の比較

| 手法 | 介入段階 | アプローチ | ライブラリ |
|------|----------|------------|------------|
| リサンプリング | 前処理 | 不均衡データの調整 | imbalanced-learn |
| リウェイティング | 前処理 | サンプル重みの調整 | AIF360 |
| 敵対的デバイアス | 学習中 | 敵対的ネットワークで属性予測を困難にする | AIF360 |
| 閾値最適化 | 後処理 | グループ別に判定閾値を調整 | Fairlearn |
| Calibrated EqOdds | 後処理 | 較正された等確率化 | AIF360 |

### 2.3 Fairlearn によるバイアス緩和

```python
# コード例 2: Fairlearn で公平性制約付き学習を行う
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# 制約なしモデル（ベースライン）
unconstrained = LogisticRegression()
unconstrained.fit(X_train, y_train)

# 公平性制約付きモデル
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=constraint
)
mitigator.fit(
    X_train, y_train,
    sensitive_features=sensitive_train
)

# 公平性ダッシュボードで比較
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

metric_frame = MetricFrame(
    metrics=accuracy_score,
    y_true=y_test,
    y_pred=mitigator.predict(X_test),
    sensitive_features=sensitive_test
)
print(metric_frame.by_group)
print(f"グループ間の精度差: {metric_frame.difference():.3f}")
```

---

## 3. 説明可能性 (Explainability)

### 3.1 説明手法の分類

```
+-------------------------------------------------------------------+
|                    説明可能性の手法                                  |
+-------------------------------------------------------------------+
|                                                                     |
|  モデル非依存 (Model-Agnostic)     モデル固有 (Model-Specific)       |
|  +---------------------------+    +---------------------------+     |
|  | SHAP (Shapley値)          |    | 決定木の分岐ルール         |     |
|  | LIME (局所的近似)          |    | 線形回帰の係数             |     |
|  | Permutation Importance    |    | Attention重み             |     |
|  | Partial Dependence Plot   |    | Grad-CAM (CNN)            |     |
|  +---------------------------+    +---------------------------+     |
|                                                                     |
|  大域的説明 (Global)               局所的説明 (Local)               |
|  「モデル全体がどう動くか」         「この1件をなぜこう判断したか」   |
+-------------------------------------------------------------------+
```

### 3.2 SHAP による説明

```python
# コード例 3: SHAP で個別予測の説明を生成する
import shap

# モデルの説明器を作成
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 個別の予測を説明 (ウォーターフォールプロット)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns.tolist()
    )
)

# 大域的な特徴量重要度 (サマリープロット)
shap.summary_plot(shap_values, X_test)

# 特定の特徴量の依存関係
shap.dependence_plot("age", shap_values, X_test)
```

### 3.3 LIME による局所的説明

```python
# コード例 4: LIME でテキスト分類の判断根拠を可視化する
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=["negative", "positive"])

def predict_proba(texts):
    """モデルのpredict_probaラッパー"""
    features = vectorizer.transform(texts)
    return model.predict_proba(features)

# 個別テキストの説明
text = "この映画は素晴らしい演技と美しい映像で感動的だった"
explanation = explainer.explain_instance(
    text,
    predict_proba,
    num_features=10,
    num_samples=1000
)

# 判断に寄与した単語とその重み
for feature, weight in explanation.as_list():
    direction = "ポジティブ" if weight > 0 else "ネガティブ"
    print(f"  '{feature}': {weight:.3f} ({direction}寄与)")
```

---

## 4. プライバシー

### 4.1 差分プライバシー

```python
# コード例 5: 差分プライバシーを適用した学習
# pip install opacus
import torch
from opacus import PrivacyEngine
from torch import nn, optim

# モデル定義
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = optim.SGD(model.parameters(), lr=0.1)

# PrivacyEngine をアタッチ
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=3.0,       # プライバシー予算 epsilon
    target_delta=1e-5,        # delta パラメータ
    epochs=10,
    max_grad_norm=1.0,        # 勾配クリッピングのノルム上限
)

# 通常通り学習（内部でDP-SGDが適用される）
for epoch in range(10):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()

    # 消費されたプライバシー予算を確認
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch}: epsilon = {epsilon:.2f}")
```

### 4.2 プライバシー保護手法の比較

| 手法 | 保護対象 | 精度への影響 | 導入コスト | ユースケース |
|------|----------|-------------|------------|-------------|
| 差分プライバシー | 個人データの推定防止 | 中〜大 | 中 | 統計クエリ、モデル学習 |
| 連合学習 | 生データの非共有 | 小〜中 | 大 | 複数組織間の協調学習 |
| 準同型暗号 | 暗号化状態での演算 | なし | 極大 | 医療・金融データ |
| k-匿名化 | 再識別の防止 | 小 | 小 | データ公開 |
| 合成データ | 実データ不使用 | 中 | 中 | テスト・開発環境 |

---

## 5. アンチパターン

### アンチパターン 1: 「公平性の後付け」

```
[誤り] モデルを完成させた後に「公平性チェック」だけ追加する

問題点:
- バイアスがデータ収集段階で混入している場合、後処理では根本解決しない
- 後処理での閾値調整は精度を大幅に犠牲にする場合がある
- チーム全体の意識が公平性に向かない

[正解] パイプライン全体で公平性を考慮する
  1. データ収集: 代表性のあるサンプリング設計
  2. 前処理: バイアス検出と緩和
  3. 学習: 公平性制約付き最適化
  4. 評価: グループ別メトリクスの確認
  5. 監視: 本番環境でのバイアス監視
```

### アンチパターン 2: 「説明可能性 = 特徴量重要度」

```
[誤り] feature_importances_ を見せれば「説明可能」と考える

問題点:
- グローバルな重要度は個別の判断根拠を説明しない
- 相関する特徴量があると重要度が分散する
- 非技術者には依然として理解困難

[正解] 対象者・目的に応じた説明を提供する
  - データサイエンティスト向け: SHAP値、PDP
  - ビジネス担当者向け: 「この顧客が解約予測された主因は
    過去30日間のログイン回数減少(寄与度40%)です」
  - 規制当局向け: モデルカード、監査ログ
  - エンドユーザー向け: 自然言語での判断理由提示
```

---

## 6. FAQ

### Q1: 公平性と精度はトレードオフですか？

**A:** 完全なトレードオフではありません。多くの場合、公平性制約を導入すると精度がわずかに低下しますが、その程度は問題設定に依存します。実務的には以下の戦略が有効です。

- バイアスのあるデータを修正すると、精度が*向上*する場合もある
- 公平性制約の強さを調整して、許容可能な精度・公平性のバランスを探す
- Pareto フロンティアを可視化し、ステークホルダーと合意形成する

### Q2: モデルカードとは何ですか？

**A:** モデルカードは、モデルの詳細情報を標準フォーマットで文書化したものです（Google が 2019 年に提唱）。以下を含みます。

- モデルの概要と意図された用途
- 学習データの説明とバイアス分析
- 性能指標（全体 + サブグループ別）
- 倫理的考慮事項と制限事項
- テスト結果と推奨される使用条件

### Q3: GDPR の「説明を受ける権利」にどう対応すればよいですか？

**A:** GDPR 第22条は、自動化された意思決定に対する説明の権利を定めています。技術的対応として以下を推奨します。

1. **意思決定ログの保存**: 各予測の入力・出力・説明を記録
2. **局所的説明の生成**: SHAP/LIME で個別判断の根拠を生成可能にする
3. **人間による介入経路**: 自動判断に異議申し立てできるプロセスを整備
4. **平易な言語での説明**: 技術的でない表現で判断根拠を伝える仕組み

---

## 7. まとめ

| 領域 | 目的 | 主要手法 | ツール例 |
|------|------|----------|----------|
| 公平性 | バイアスの検出・緩和 | SPD, DI, 敵対的デバイアス | Fairlearn, AIF360 |
| 説明可能性 | 判断根拠の可視化 | SHAP, LIME, PDP | shap, lime |
| プライバシー | 個人情報の保護 | 差分プライバシー, 連合学習 | Opacus, PySyft |
| 透明性 | モデル情報の文書化 | モデルカード | model-card-toolkit |
| 安全性 | 有害出力の防止 | ガードレール, レッドチーム | guardrails-ai |
| 説明責任 | 監査可能性の確保 | 監査ログ, リネージ | MLflow |

---

## 次に読むべきガイド

- [AI セーフティ](../../../llm-and-ai-comparison/docs/04-ethics/00-ai-safety.md) — アライメント、レッドチームの実践
- [AI ガバナンス](../../../llm-and-ai-comparison/docs/04-ethics/01-ai-governance.md) — 規制・ポリシーの動向
- [MLOps](./02-mlops.md) — 責任ある AI を本番パイプラインに組み込む方法

---

## 参考文献

1. Mitchell, M. et al. (2019). "Model Cards for Model Reporting." *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*. ACM. https://doi.org/10.1145/3287560.3287596
2. Dwork, C. & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science, 9*(3-4), 211-407. https://doi.org/10.1561/0400000042
3. Mehrabi, N. et al. (2021). "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys, 54*(6), 1-35. https://doi.org/10.1145/3457607
