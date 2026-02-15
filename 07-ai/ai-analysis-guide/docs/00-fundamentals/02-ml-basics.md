# 機械学習基礎 — 教師あり/なし、評価指標

> 機械学習の基本概念を理論と実装の両面から体系的に理解する

## この章で学ぶこと

1. **学習パラダイム** — 教師あり、教師なし、半教師あり、強化学習の原理と使い分け
2. **バイアス-バリアンストレードオフ** — 過学習・過少適合のメカニズムと対策
3. **評価指標** — 分類・回帰それぞれの評価指標と適切な選択基準
4. **交差検証** — 汎化性能を正しく測定するための分割戦略
5. **ハイパーパラメータ最適化** — Grid/Random/Bayesian 手法の実装と比較
6. **モデル選択** — ビジネス要件に応じたアルゴリズム選択の指針

---

## 1. 教師あり学習の基礎

### 学習の仕組み

```
教師あり学習のフロー:

  訓練データ                    予測
  ┌─────────────┐           ┌─────────┐
  │ 特徴量 (X)  │           │ 新データ │
  │ ラベル (y)  │           │ (X_new) │
  └──────┬──────┘           └────┬────┘
         │                       │
         v                       v
  ┌──────┴──────┐         ┌─────┴─────┐
  │  学習       │         │  推論     │
  │  f(X) ≈ y  │────────>│  ŷ = f(X) │
  │  パラメータ │  モデル │           │
  │  最適化     │         │           │
  └─────────────┘         └───────────┘

  損失関数: L(y, ŷ) を最小化するパラメータ θ を探す
  θ* = argmin_θ Σ L(y_i, f(x_i; θ)) + λR(θ)
                                        ↑ 正則化項
```

### 教師あり学習の主要カテゴリ

教師あり学習は、目的変数の性質によって大きく2つに分かれる。

```
分類（Classification）:
  - 目的変数が離散値（カテゴリ）
  - 例: スパム/非スパム、画像のクラス、顧客の離脱予測
  - 代表的アルゴリズム:
    - ロジスティック回帰
    - サポートベクターマシン (SVM)
    - 決定木 / ランダムフォレスト
    - 勾配ブースティング (XGBoost, LightGBM)
    - ニューラルネットワーク

回帰（Regression）:
  - 目的変数が連続値（数値）
  - 例: 住宅価格予測、売上予測、気温予測
  - 代表的アルゴリズム:
    - 線形回帰 / Ridge / Lasso
    - サポートベクター回帰 (SVR)
    - 決定木回帰 / ランダムフォレスト回帰
    - 勾配ブースティング回帰
    - ニューラルネットワーク回帰
```

### 損失関数の詳細

損失関数はモデルの学習を導く指標であり、適切な選択がモデル性能に直結する。

```python
import numpy as np

# === 回帰の損失関数 ===

def mse_loss(y_true, y_pred):
    """Mean Squared Error: 外れ値に敏感、微分が容易"""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """Mean Absolute Error: 外れ値に頑健、微分不連続点あり"""
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss: MSEとMAEのハイブリッド
    |error| <= delta: MSE（滑らか）
    |error| > delta:  MAE（外れ値に頑健）
    """
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small, squared_loss, linear_loss))

def quantile_loss(y_true, y_pred, quantile=0.5):
    """Quantile Loss: 特定の分位点を予測する場合に使用
    quantile=0.5 → MAEと同等
    quantile=0.9 → 90パーセンタイルの予測
    """
    error = y_true - y_pred
    return np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error))


# === 分類の損失関数 ===

def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    """Binary Cross-Entropy: 二値分類の標準的な損失関数"""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def categorical_cross_entropy(y_true_onehot, y_prob, eps=1e-15):
    """Categorical Cross-Entropy: 多クラス分類の標準的な損失関数"""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_prob), axis=1))

def hinge_loss(y_true, y_score):
    """Hinge Loss: SVMで使用。y_true は {-1, +1}"""
    return np.mean(np.maximum(0, 1 - y_true * y_score))

def focal_loss(y_true, y_prob, gamma=2.0, alpha=0.25, eps=1e-15):
    """Focal Loss: クラス不均衡に強い損失関数（RetinaNetで提案）
    容易なサンプルの損失を下げ、困難なサンプルに集中
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    pt = np.where(y_true == 1, y_prob, 1 - y_prob)
    at = np.where(y_true == 1, alpha, 1 - alpha)
    loss = -at * (1 - pt) ** gamma * np.log(pt)
    return np.mean(loss)


# 損失関数の比較実験
y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
y_pred = np.array([2.8, 5.2, 2.0, 10.0, 4.3])  # 7.0→10.0 は外れ値的な予測

print("回帰損失関数の比較（外れ値あり）:")
print(f"  MSE:   {mse_loss(y_true, y_pred):.4f}")
print(f"  MAE:   {mae_loss(y_true, y_pred):.4f}")
print(f"  Huber: {huber_loss(y_true, y_pred, delta=1.0):.4f}")
```

### コード例1: 教師あり学習のワークフロー全体

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. データ準備
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

# 2. 訓練/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 前処理
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. モデル学習
models = {
    "ロジスティック回帰": LogisticRegression(max_iter=1000, random_state=42),
    "ランダムフォレスト": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    # 交差検証
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="f1")
    print(f"\n{name}")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # テスト評価
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### コード例1b: パイプラインによる前処理とモデルの統合

実務では前処理とモデルをパイプラインにまとめることで、データリークを防止し、再現性を確保する。

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# 現実的なデータセットのシミュレーション
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "age": np.random.normal(40, 15, n),
    "income": np.random.lognormal(10, 1, n),
    "education": np.random.choice(["high_school", "bachelor", "master", "phd"], n),
    "city": np.random.choice(["tokyo", "osaka", "nagoya", "fukuoka"], n),
    "satisfaction": np.random.normal(3.5, 1.0, n),
})
# 欠損値の挿入
for col in ["age", "income", "satisfaction"]:
    mask = np.random.random(n) < 0.05
    df.loc[mask, col] = np.nan

y = (df["income"].fillna(df["income"].median()) > df["income"].median()).astype(int)
X = df.drop(columns=[])

# カラム種別の定義
numeric_features = ["age", "income", "satisfaction"]
categorical_features = ["education", "city"]

# 数値特徴量の前処理パイプライン
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# カテゴリ特徴量の前処理パイプライン
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
])

# 前処理の統合
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# パイプライン全体
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )),
])

# 交差検証（データリークなし）
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# パイプラインの保存
import joblib
pipeline.fit(X, y)
joblib.dump(pipeline, "models/full_pipeline.joblib")

# 本番環境での推論
loaded_pipeline = joblib.load("models/full_pipeline.joblib")
new_data = pd.DataFrame({
    "age": [35], "income": [50000], "education": ["master"],
    "city": ["tokyo"], "satisfaction": [4.0]
})
prediction = loaded_pipeline.predict(new_data)
probability = loaded_pipeline.predict_proba(new_data)[:, 1]
print(f"予測: {prediction[0]}, 確率: {probability[0]:.4f}")
```

---

## 2. 教師なし学習の基礎

### 教師なし学習の体系

教師なし学習はラベル（正解）なしでデータの構造を発見する手法である。

```
教師なし学習のカテゴリ:

  ┌───────────────────────────────────────────────────┐
  │             教師なし学習                           │
  │                                                   │
  │  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
  │  │ クラスタリング│  │ 次元削減     │  │ 異常検知 │ │
  │  ├─────────────┤  ├──────────────┤  ├──────────┤ │
  │  │ K-means     │  │ PCA          │  │ LOF      │ │
  │  │ DBSCAN      │  │ t-SNE        │  │ IF       │ │
  │  │ 階層的      │  │ UMAP         │  │ One-Class│ │
  │  │ GMM         │  │ Autoencoders │  │ SVM      │ │
  │  └─────────────┘  └──────────────┘  └──────────┘ │
  │                                                   │
  │  ┌─────────────┐  ┌──────────────┐               │
  │  │ 関連ルール  │  │ 密度推定     │               │
  │  ├─────────────┤  ├──────────────┤               │
  │  │ Apriori     │  │ KDE          │               │
  │  │ FP-Growth   │  │ GMM          │               │
  │  └─────────────┘  └──────────────┘               │
  └───────────────────────────────────────────────────┘
```

### コード例: クラスタリングと評価

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs

# データ生成
X, y_true = make_blobs(n_samples=500, n_features=2, centers=4,
                        cluster_std=1.0, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 複数のクラスタリングアルゴリズムの比較
algorithms = {
    "K-Means": KMeans(n_clusters=4, random_state=42, n_init=10),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=4),
    "GMM": GaussianMixture(n_components=4, random_state=42),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (name, algo) in enumerate(algorithms.items()):
    if name == "GMM":
        labels = algo.fit_predict(X_scaled)
    else:
        labels = algo.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # ノイズ以外のサンプルで評価
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
        db = davies_bouldin_score(X_scaled[mask], labels[mask])
    else:
        sil, ch, db = 0, 0, 0

    axes[idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", s=10)
    axes[idx].set_title(f"{name}\nSil={sil:.3f}, CH={ch:.0f}, DB={db:.3f}")
    print(f"{name}: clusters={n_clusters}, Silhouette={sil:.4f}, "
          f"CH={ch:.1f}, DB={db:.4f}")

plt.tight_layout()
plt.savefig("reports/clustering_comparison.png", dpi=150)
plt.close()
```

### エルボー法とシルエット分析によるK選択

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def optimal_k_analysis(X, k_range=range(2, 11)):
    """エルボー法とシルエット分析でKを決定"""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # エルボー法
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_xlabel("クラスタ数 K")
    ax1.set_ylabel("慣性 (Inertia)")
    ax1.set_title("エルボー法")
    ax1.grid(True, alpha=0.3)

    # シルエットスコア
    ax2.plot(list(k_range), silhouettes, "ro-")
    ax2.set_xlabel("クラスタ数 K")
    ax2.set_ylabel("シルエットスコア")
    ax2.set_title("シルエット分析")
    ax2.grid(True, alpha=0.3)

    best_k = list(k_range)[np.argmax(silhouettes)]
    ax2.axvline(x=best_k, color="green", linestyle="--", label=f"Best K={best_k}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("reports/optimal_k_analysis.png", dpi=150)
    plt.close()

    return best_k

best_k = optimal_k_analysis(X_scaled)
print(f"最適なクラスタ数: {best_k}")
```

---

## 3. 半教師あり学習と自己教師あり学習

### 半教師あり学習

少量のラベル付きデータと大量のラベルなしデータを組み合わせて学習する手法である。ラベル付けコストが高い実務シナリオで特に有効。

```
半教師あり学習のアプローチ:

  1. Self-Training（自己学習）
     ┌──────────┐    ┌──────────┐    ┌──────────┐
     │ ラベル付き│───>│ モデル   │───>│ ラベル   │
     │ データ   │    │ 学習     │    │ なしを   │
     └──────────┘    └──────────┘    │ 予測     │
                           ↑          └────┬─────┘
                           │               │
                           └───────────────┘
                           高信頼度の予測をラベル付きに追加

  2. Label Propagation（ラベル伝播）
     グラフ上で隣接するデータにラベルを伝播

  3. Co-Training
     異なるビュー（特徴量セット）で2つのモデルを
     相互に学習させる
```

```python
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# データ生成
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ラベルの一部を欠損させる（5%だけラベルあり）
rng = np.random.RandomState(42)
mask = rng.random(len(y_train)) > 0.05  # 95%のラベルを隠す
y_train_semi = y_train.copy()
y_train_semi[mask] = -1  # -1 はラベルなしを表す

n_labeled = (y_train_semi != -1).sum()
n_unlabeled = (y_train_semi == -1).sum()
print(f"ラベル付き: {n_labeled}, ラベルなし: {n_unlabeled}")

# Self-Training
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
self_training = SelfTrainingClassifier(
    base_estimator=base_model,
    threshold=0.8,  # 信頼度80%以上のサンプルをラベル付きに追加
    max_iter=20,
)
self_training.fit(X_train, y_train_semi)
score_semi = self_training.score(X_test, y_test)

# 比較: ラベル付きデータのみで学習
labeled_mask = y_train_semi != -1
base_model_only = RandomForestClassifier(n_estimators=100, random_state=42)
base_model_only.fit(X_train[labeled_mask], y_train[labeled_mask])
score_labeled = base_model_only.score(X_test, y_test)

# 比較: 全ラベルで学習（上限）
full_model = RandomForestClassifier(n_estimators=100, random_state=42)
full_model.fit(X_train, y_train)
score_full = full_model.score(X_test, y_test)

print(f"\n精度比較:")
print(f"  ラベル付きのみ ({n_labeled}サンプル): {score_labeled:.4f}")
print(f"  半教師あり学習:                       {score_semi:.4f}")
print(f"  全ラベル学習 (上限):                  {score_full:.4f}")
```

### 自己教師あり学習

自己教師あり学習はラベルを使わず、データ自体から学習信号を生成する手法である。近年の大規模言語モデルや画像モデルの基盤技術となっている。

```
自己教師あり学習のプレテキストタスク:

  テキスト:
    - Masked Language Model（BERT）: 文中のトークンをマスクして予測
    - Next Token Prediction（GPT）: 次のトークンを予測
    - Sentence Order Prediction: 文の順序を予測

  画像:
    - Contrastive Learning（SimCLR, MoCo）: 同一画像の異なる変換を近づける
    - Masked Image Modeling（MAE）: 画像パッチをマスクして再構成
    - DINO: 自己蒸留によるビジョントランスフォーマーの学習

  表形式:
    - VIME: 値の置換を検出
    - SCARF: 特徴量のサブセットをマスクして対比学習
```

---

## 4. 強化学習の基礎

### 強化学習のフレームワーク

```
マルコフ決定過程 (MDP):

  エージェント                 環境
  ┌──────────┐               ┌──────────┐
  │          │ ── Action ──> │          │
  │  方策    │               │  状態遷移│
  │  π(a|s)  │ <── State ── │          │
  │          │ <── Reward ── │          │
  └──────────┘               └──────────┘

  目標: 累積報酬 G_t = Σ γ^k × r_{t+k} の最大化
         γ: 割引率（0 < γ ≤ 1）

  主要概念:
    - 状態価値関数 V(s): 状態 s から得られる期待累積報酬
    - 行動価値関数 Q(s, a): 状態 s で行動 a を取った場合の期待累積報酬
    - ベルマン方程式: V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
```

### 強化学習アルゴリズムの分類

```
強化学習手法の分類:

  ┌──────────────────────────────────────────┐
  │           強化学習                        │
  │                                          │
  │  ┌──────────────┐  ┌──────────────────┐  │
  │  │ Model-Free   │  │ Model-Based      │  │
  │  ├──────────────┤  ├──────────────────┤  │
  │  │              │  │ 環境モデルを学習  │  │
  │  │ ┌──────────┐ │  │ して計画を行う    │  │
  │  │ │Value-Based│ │  │                  │  │
  │  │ │ Q-Learning│ │  │ 例:              │  │
  │  │ │ DQN      │ │  │  - Dyna-Q        │  │
  │  │ │ SARSA    │ │  │  - World Models  │  │
  │  │ └──────────┘ │  │  - MuZero        │  │
  │  │              │  └──────────────────┘  │
  │  │ ┌──────────┐ │                        │
  │  │ │Policy-   │ │                        │
  │  │ │Based     │ │                        │
  │  │ │ REINFORCE│ │                        │
  │  │ │ PPO      │ │                        │
  │  │ │ A3C      │ │                        │
  │  │ └──────────┘ │                        │
  │  │              │                        │
  │  │ ┌──────────┐ │                        │
  │  │ │Actor-    │ │                        │
  │  │ │Critic    │ │                        │
  │  │ │ A2C      │ │                        │
  │  │ │ SAC      │ │                        │
  │  │ │ TD3      │ │                        │
  │  │ └──────────┘ │                        │
  │  └──────────────┘                        │
  └──────────────────────────────────────────┘
```

### コード例: Q-Learningの基本実装

```python
import numpy as np

class QLearningAgent:
    """Q-Learning エージェントの基本実装"""

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-greedy 方策で行動を選択"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Q値の更新"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Q-Learning 更新則
        self.q_table[state, action] += self.lr * (
            target - self.q_table[state, action]
        )

    def decay_epsilon(self):
        """探索率を減衰"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, n_episodes=1000):
        """学習ループ"""
        rewards_history = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            self.decay_epsilon()
            rewards_history.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode+1}: "
                      f"Avg Reward={avg_reward:.2f}, "
                      f"Epsilon={self.epsilon:.4f}")

        return rewards_history
```

---

## 5. バイアス-バリアンストレードオフ

### 概念図

```
予測誤差の分解:

  総誤差 = バイアス² + バリアンス + ノイズ（削減不可）

  モデル複雑度 →  低い                            高い
                   │                                │
  バイアス:        │████████████████░░░░░░░░░░░░░░░│  高→低
  バリアンス:      │░░░░░░░░░░░░░░░████████████████│  低→高
                   │                                │
  訓練誤差:        │████████░░░░░░░░░░░░░░░░░░░░░░░│  高→極低
  テスト誤差:      │████████░░░░░░░░░░░░░████████░░│  U字型
                   │         ↑                      │
                   │     最適点                     │
                   └────────────────────────────────┘

  過少適合                最適             過学習
  (Underfitting)         (Best)         (Overfitting)
```

### バイアスとバリアンスの数学的定義

```
あるデータ点 x に対する予測の誤差分解:

  E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²

  バイアス（Bias）:
    Bias[f̂(x)] = E[f̂(x)] - f(x)
    = モデルの予測の期待値と真の関数の差
    → モデルが単純すぎると大きくなる

  バリアンス（Variance）:
    Var[f̂(x)] = E[(f̂(x) - E[f̂(x)])²]
    = 異なる訓練データで学習した場合の予測のばらつき
    → モデルが複雑すぎると大きくなる

  ノイズ（σ²）:
    データ自体の不可避なランダム性
    → どんなモデルでも削減不可能
```

### コード例2: 過学習の可視化と診断

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def plot_learning_curve(estimator, X, y, title="学習曲線"):
    """学習曲線をプロットして過学習/過少適合を診断"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="訓練スコア")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="検証スコア")
    ax.set_xlabel("訓練サンプル数")
    ax.set_ylabel("精度")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 診断メッセージ
    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.1:
        ax.text(0.5, 0.05, f"過学習の兆候（Gap={gap:.3f}）",
                transform=ax.transAxes, color="red", fontsize=12)
    elif test_mean[-1] < 0.7:
        ax.text(0.5, 0.05, "過少適合の兆候",
                transform=ax.transAxes, color="red", fontsize=12)

    plt.tight_layout()
    plt.savefig("reports/learning_curve.png", dpi=150)
    plt.close()

    return train_mean, test_mean

# 過学習する深い決定木
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
plot_learning_curve(deep_tree, X_train_s, y_train, "深い決定木（過学習傾向）")

# 正則化された浅い決定木
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
plot_learning_curve(shallow_tree, X_train_s, y_train, "浅い決定木（適切な複雑度）")
```

### コード例2b: Validation Curveによるハイパーパラメータ診断

```python
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_validation_curve(estimator, X, y, param_name, param_range, title="検証曲線"):
    """ハイパーパラメータを変化させたときの訓練/検証スコアを可視化"""
    train_scores, test_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(param_range, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(param_range, test_mean - test_std,
                    test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(param_range, train_mean, "o-", color="blue", label="訓練スコア")
    ax.plot(param_range, test_mean, "o-", color="orange", label="検証スコア")
    ax.set_xlabel(param_name)
    ax.set_ylabel("精度")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_idx = np.argmax(test_mean)
    ax.axvline(x=param_range[best_idx], color="green", linestyle="--",
               label=f"最適値={param_range[best_idx]}")
    ax.legend()

    plt.tight_layout()
    plt.savefig("reports/validation_curve.png", dpi=150)
    plt.close()

    return param_range[best_idx]

# 決定木の深さに対する検証曲線
best_depth = plot_validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train_s, y_train,
    param_name="max_depth",
    param_range=[1, 2, 3, 5, 7, 10, 15, 20, 30, None],
    title="決定木: max_depth の検証曲線"
)
print(f"最適な max_depth: {best_depth}")
```

### 過学習対策の体系的アプローチ

```
過学習対策の分類:

  1. データ量の増加
     ├── データ収集の強化
     ├── データ拡張（画像回転、テキスト同義語置換等）
     └── 合成データ生成（SMOTE、GANs等）

  2. モデルの複雑度制限
     ├── パラメータ数の削減
     ├── 浅い構造の選択
     └── 特徴量選択によるノイズ特徴量除去

  3. 正則化
     ├── L1 正則化（Lasso）: スパース化、特徴量選択効果
     ├── L2 正則化（Ridge）: パラメータを小さく
     ├── Elastic Net: L1 + L2
     ├── Dropout（ニューラルネットワーク）
     └── Batch Normalization

  4. アンサンブル学習
     ├── Bagging: 分散を下げる（RandomForest）
     ├── Boosting: バイアスを下げる（XGBoost, LightGBM）
     └── Stacking: 複数モデルのメタ学習

  5. 早期停止（Early Stopping）
     └── 検証スコアが改善しなくなったら学習を打ち切る
```

```python
# 正則化の実装例
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# L1, L2, ElasticNet の比較
regularizations = {
    "L1 (Lasso)": LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000),
    "L2 (Ridge)": LogisticRegression(penalty="l2", solver="saga", C=1.0, max_iter=5000),
    "ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga",
                                      C=1.0, l1_ratio=0.5, max_iter=5000),
    "正則化なし": LogisticRegression(penalty=None, solver="saga", max_iter=5000),
}

print("正則化手法の比較:")
print("-" * 60)
for name, model in regularizations.items():
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
    model.fit(X_train_s, y_train)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    print(f"  {name:15s}: Accuracy={scores.mean():.4f} ± {scores.std():.4f}, "
          f"非ゼロ係数={n_nonzero}")
```

---

## 6. 交差検証

### 交差検証の種類

```
K-Fold 交差検証 (K=5):

  Fold 1: [TEST] [Train] [Train] [Train] [Train]
  Fold 2: [Train] [TEST] [Train] [Train] [Train]
  Fold 3: [Train] [Train] [TEST] [Train] [Train]
  Fold 4: [Train] [Train] [Train] [TEST] [Train]
  Fold 5: [Train] [Train] [Train] [Train] [TEST]

  最終スコア = 5つのFoldのスコアの平均 ± 標準偏差

Stratified K-Fold（クラス不均衡時）:
  各Foldでクラスの比率を維持

Time Series Split（時系列データ）:
  Fold 1: [Train] → [TEST]
  Fold 2: [Train][Train] → [TEST]
  Fold 3: [Train][Train][Train] → [TEST]
  ※ 未来のデータで訓練しない

Group K-Fold（グループデータ）:
  同じグループが訓練と検証に分かれないようにする
  例: 患者ごとの医療データで患者をグループとして扱う

Leave-One-Out (LOO):
  K = サンプル数。小データに有効だが計算コスト大
```

### コード例3: 交差検証戦略の使い分け

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_validate, RepeatedStratifiedKFold, LeaveOneOut
)
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def comprehensive_cv(model, X, y, cv_strategy="stratified", groups=None):
    """包括的な交差検証を実行"""

    strategies = {
        "kfold": KFold(n_splits=5, shuffle=True, random_state=42),
        "stratified": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "repeated": RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
        "timeseries": TimeSeriesSplit(n_splits=5),
    }

    if cv_strategy == "group" and groups is not None:
        cv = GroupKFold(n_splits=5)
        scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        results = cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=-1, groups=groups
        )
    else:
        cv = strategies[cv_strategy]
        scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        results = cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=-1
        )

    print(f"交差検証戦略: {cv_strategy}")
    print("-" * 70)
    for metric in scoring:
        train_key = f"train_{metric}"
        test_key = f"test_{metric}"
        print(f"  {metric:12s}: "
              f"Train={results[train_key].mean():.4f} ± {results[train_key].std():.4f}  "
              f"Test={results[test_key].mean():.4f} ± {results[test_key].std():.4f}")

    # 過学習度の判定
    train_acc = results["train_accuracy"].mean()
    test_acc = results["test_accuracy"].mean()
    gap = train_acc - test_acc
    if gap > 0.1:
        print(f"\n  ⚠ 過学習の兆候あり (Train-Test Gap = {gap:.4f})")
    elif test_acc < 0.6:
        print(f"\n  ⚠ 過少適合の兆候あり (Test Accuracy = {test_acc:.4f})")
    else:
        print(f"\n  ✓ 良好なフィット (Gap = {gap:.4f})")

    return results

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
results = comprehensive_cv(model, X_train_s, y_train, "stratified")
```

### 交差検証でのデータリークの防止

```python
# ★ 重要: 交差検証でのデータリークを防ぐ正しい実装

# BAD: 交差検証の「前に」全データで前処理
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← テストデータの情報がリーク！
scores = cross_val_score(SVC(), X_scaled, y, cv=5)
print(f"リークあり: {scores.mean():.4f}")  # 楽観的な結果

# GOOD: パイプラインで前処理をCV内に含める
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])
scores = cross_val_score(pipe, X, y, cv=5)  # 各Fold内で個別にfit_transform
print(f"リークなし: {scores.mean():.4f}")  # 正確な結果
```

---

## 7. 評価指標

### コード例4: 分類指標の詳細計算

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt

class ClassificationEvaluator:
    """分類モデルの包括的評価"""

    def __init__(self, y_true, y_pred, y_prob=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def print_metrics(self):
        """主要指標を一覧表示"""
        print("=" * 60)
        print("分類評価指標")
        print("=" * 60)
        print(f"  Accuracy:          {accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"  Balanced Accuracy: {balanced_accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"  Precision:         {precision_score(self.y_true, self.y_pred):.4f}")
        print(f"  Recall:            {recall_score(self.y_true, self.y_pred):.4f}")
        print(f"  F1-Score:          {f1_score(self.y_true, self.y_pred):.4f}")
        print(f"  MCC:               {matthews_corrcoef(self.y_true, self.y_pred):.4f}")
        print(f"  Cohen's Kappa:     {cohen_kappa_score(self.y_true, self.y_pred):.4f}")
        if self.y_prob is not None:
            print(f"  AUC-ROC:           {roc_auc_score(self.y_true, self.y_prob):.4f}")
            print(f"  AP:                {average_precision_score(self.y_true, self.y_prob):.4f}")
            print(f"  Log Loss:          {log_loss(self.y_true, self.y_prob):.4f}")

        cm = confusion_matrix(self.y_true, self.y_pred)
        print(f"\n  混同行列:")
        print(f"           予測=0  予測=1")
        print(f"  実際=0   {cm[0,0]:5d}   {cm[0,1]:5d}  (TN, FP)")
        print(f"  実際=1   {cm[1,0]:5d}   {cm[1,1]:5d}  (FN, TP)")

        # 追加メトリクス
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"\n  Specificity (TNR): {specificity:.4f}")
        print(f"  NPV:               {npv:.4f}")
        print(f"  FPR:               {fp / (fp + tn):.4f}")
        print(f"  FNR:               {fn / (fn + tp):.4f}")

    def plot_roc_pr(self):
        """ROC曲線とPR曲線を並べて描画"""
        if self.y_prob is None:
            print("確率予測が必要です")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ROC曲線
        fpr, tpr, thresholds_roc = roc_curve(self.y_true, self.y_prob)
        auc = roc_auc_score(self.y_true, self.y_prob)
        ax1.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("偽陽性率 (FPR)")
        ax1.set_ylabel("真陽性率 (TPR)")
        ax1.set_title("ROC曲線")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PR曲線
        prec, rec, thresholds_pr = precision_recall_curve(self.y_true, self.y_prob)
        ap = average_precision_score(self.y_true, self.y_prob)
        ax2.plot(rec, prec, label=f"AP = {ap:.3f}")
        ax2.set_xlabel("再現率 (Recall)")
        ax2.set_ylabel("適合率 (Precision)")
        ax2.set_title("PR曲線")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("reports/roc_pr_curves.png", dpi=150)
        plt.close()

    def find_optimal_threshold(self, metric="f1"):
        """最適な閾値を探索"""
        if self.y_prob is None:
            raise ValueError("確率予測が必要です")

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5

        print(f"\n閾値最適化（指標: {metric}）:")
        print("-" * 50)

        for t in thresholds:
            y_pred_t = (self.y_prob >= t).astype(int)
            if metric == "f1":
                score = f1_score(self.y_true, y_pred_t, zero_division=0)
            elif metric == "precision":
                score = precision_score(self.y_true, y_pred_t, zero_division=0)
            elif metric == "recall":
                score = recall_score(self.y_true, y_pred_t, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = t

        print(f"  最適閾値: {best_threshold:.2f}")
        print(f"  最適{metric}: {best_score:.4f}")
        print(f"  デフォルト閾値(0.5)の{metric}: "
              f"{f1_score(self.y_true, (self.y_prob >= 0.5).astype(int)):.4f}")

        return best_threshold

# 使用例
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

evaluator = ClassificationEvaluator(y_test, y_pred, y_prob)
evaluator.print_metrics()
evaluator.plot_roc_pr()
best_t = evaluator.find_optimal_threshold("f1")
```

### コード例4b: 多クラス分類の評価

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def multiclass_evaluation(y_true, y_pred, y_prob, class_names):
    """多クラス分類の包括的評価"""

    print("=" * 60)
    print("多クラス分類評価")
    print("=" * 60)

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=class_names))

    # マクロ / ミクロ / 加重平均
    for avg in ["micro", "macro", "weighted"]:
        f1 = f1_score(y_true, y_pred, average=avg)
        print(f"  F1 ({avg:8s}): {f1:.4f}")

    # 多クラス AUC-ROC
    if y_prob is not None:
        try:
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            auc_ovo = roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")
            print(f"\n  AUC-ROC (OVR macro): {auc_ovr:.4f}")
            print(f"  AUC-ROC (OVO macro): {auc_ovo:.4f}")
        except ValueError as e:
            print(f"  AUC計算エラー: {e}")

    # 混同行列のヒートマップ
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("予測ラベル")
    ax.set_ylabel("実際のラベル")
    ax.set_title("多クラス混同行列")
    plt.tight_layout()
    plt.savefig("reports/multiclass_confusion.png", dpi=150)
    plt.close()

    return cm
```

### コード例5: 回帰指標の計算

```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
    median_absolute_error, max_error,
    explained_variance_score
)
import matplotlib.pyplot as plt

class RegressionEvaluator:
    """回帰モデルの包括的評価"""

    def __init__(self, y_true, y_pred, n_features=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.n_features = n_features

    def all_metrics(self) -> dict:
        """全指標を計算"""
        mse = mean_squared_error(self.y_true, self.y_pred)
        metrics = {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(self.y_true, self.y_pred),
            "Median AE": median_absolute_error(self.y_true, self.y_pred),
            "Max Error": max_error(self.y_true, self.y_pred),
            "MAPE(%)": mean_absolute_percentage_error(self.y_true, self.y_pred) * 100,
            "R²": r2_score(self.y_true, self.y_pred),
            "Explained Var": explained_variance_score(self.y_true, self.y_pred),
        }
        if self.n_features is not None:
            metrics["Adjusted R²"] = self._adjusted_r2(self.n_features)
        return metrics

    def _adjusted_r2(self, n_features: int) -> float:
        """自由度調整済みR²"""
        n = len(self.y_true)
        r2 = r2_score(self.y_true, self.y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def print_metrics(self):
        metrics = self.all_metrics()
        print("=" * 50)
        print("回帰評価指標")
        print("=" * 50)
        for name, value in metrics.items():
            print(f"  {name:15s}: {value:.4f}")

    def plot_diagnostics(self):
        """回帰診断プロットの作成"""
        residuals = self.y_true - self.y_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 予測 vs 実測
        axes[0, 0].scatter(self.y_true, self.y_pred, alpha=0.5, s=20)
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--")
        axes[0, 0].set_xlabel("実測値")
        axes[0, 0].set_ylabel("予測値")
        axes[0, 0].set_title("予測 vs 実測")

        # 2. 残差プロット
        axes[0, 1].scatter(self.y_pred, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("予測値")
        axes[0, 1].set_ylabel("残差")
        axes[0, 1].set_title("残差プロット")

        # 3. 残差のヒストグラム
        axes[1, 0].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(x=0, color="r", linestyle="--")
        axes[1, 0].set_xlabel("残差")
        axes[1, 0].set_ylabel("頻度")
        axes[1, 0].set_title("残差の分布")

        # 4. Q-Qプロット
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        axes[1, 1].scatter(osm, osr, alpha=0.5, s=20)
        axes[1, 1].plot(osm, slope * np.array(osm) + intercept, "r--")
        axes[1, 1].set_xlabel("理論分位数")
        axes[1, 1].set_ylabel("サンプル分位数")
        axes[1, 1].set_title(f"Q-Qプロット (R={r:.4f})")

        plt.tight_layout()
        plt.savefig("reports/regression_diagnostics.png", dpi=150)
        plt.close()

# 使用例
# evaluator = RegressionEvaluator(y_test, y_pred, n_features=10)
# evaluator.print_metrics()
# evaluator.plot_diagnostics()
```

---

## 8. ハイパーパラメータ最適化

### 最適化手法の比較

```
ハイパーパラメータ最適化手法:

  手法                探索方法              計算コスト    発見効率
  ─────────────────────────────────────────────────────────────
  Grid Search         全組合せ              高い          低い
  Random Search       ランダムサンプリング  中程度        中程度
  Bayesian (Optuna)   ベイズ最適化          低い          高い
  Hyperband           Early Stopping付き    低い          高い

  一般的な推奨:
    - パラメータ数が少ない(2-3個): Grid Search
    - パラメータ数が中程度(4-6個): Random Search
    - パラメータ数が多い/探索空間が広い: Bayesian (Optuna)
```

### コード例6: Grid Search と Random Search

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
import time

# Grid Search
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

print("Grid Search")
start = time.time()
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
)
grid.fit(X_train_s, y_train)
elapsed_grid = time.time() - start
print(f"  組合せ数: {len(grid.cv_results_['params'])}")
print(f"  最良スコア: {grid.best_score_:.4f}")
print(f"  最良パラメータ: {grid.best_params_}")
print(f"  計算時間: {elapsed_grid:.1f}秒")

# Random Search
param_distributions = {
    "n_estimators": randint(50, 500),
    "max_depth": [3, 5, 10, 15, 20, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.1, 0.9),
}

print("\nRandom Search")
start = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions, n_iter=50, cv=5, scoring="f1",
    n_jobs=-1, random_state=42, verbose=0
)
random_search.fit(X_train_s, y_train)
elapsed_random = time.time() - start
print(f"  試行数: 50")
print(f"  最良スコア: {random_search.best_score_:.4f}")
print(f"  最良パラメータ: {random_search.best_params_}")
print(f"  計算時間: {elapsed_random:.1f}秒")
```

### コード例7: Optunaによるベイズ最適化

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

def objective(trial):
    """Optunaの目的関数"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }

    model = GradientBoostingClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="f1", n_jobs=-1)
    return scores.mean()

# 最適化実行
study = optuna.create_study(direction="maximize", study_name="gbc_optimization")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n最良スコア: {study.best_value:.4f}")
print(f"最良パラメータ:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# 最適パラメータでモデル学習
best_model = GradientBoostingClassifier(random_state=42, **study.best_params)
best_model.fit(X_train_s, y_train)
test_score = best_model.score(X_test_s, y_test)
print(f"\nテストスコア: {test_score:.4f}")

# パラメータの重要度
importance = optuna.importance.get_param_importances(study)
print("\nパラメータ重要度:")
for param, imp in importance.items():
    print(f"  {param}: {imp:.4f}")
```

### コード例8: LightGBMの最適化（実務的なパターン）

```python
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np

def lgbm_objective(trial, X, y, cv=5):
    """LightGBMのOptuna目的関数（実務向け）"""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        )

        from sklearn.metrics import f1_score
        y_pred = model.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_pred))

        # Optuna pruning
        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

# study = optuna.create_study(direction="maximize",
#                              pruner=optuna.pruners.MedianPruner())
# study.optimize(lambda trial: lgbm_objective(trial, X_train, y_train),
#                n_trials=200, show_progress_bar=True)
```

---

## 9. 特徴量重要度と解釈性

### 特徴量重要度の計算方法

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

def feature_importance_analysis(model, X_train, X_test, y_test, feature_names):
    """複数の方法で特徴量重要度を分析"""

    # 1. モデル組み込みの重要度（不純度ベース）
    if hasattr(model, "feature_importances_"):
        imp_builtin = model.feature_importances_
    else:
        imp_builtin = None

    # 2. Permutation Importance（推奨）
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    # 結果の比較
    df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std": perm_result.importances_std,
    })
    if imp_builtin is not None:
        df["builtin_importance"] = imp_builtin

    df = df.sort_values("perm_importance_mean", ascending=False)

    print("特徴量重要度 (Permutation Importance):")
    print("-" * 60)
    for _, row in df.head(15).iterrows():
        print(f"  {row['feature']:30s}: "
              f"{row['perm_importance_mean']:.4f} ± {row['perm_importance_std']:.4f}")

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(df))
    top_df = df.head(top_n)
    ax.barh(range(top_n), top_df["perm_importance_mean"].values, align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df["feature"].values)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("特徴量重要度 (Top 20)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()

    return df

# SHAP値による解釈（より詳細）
def shap_analysis(model, X_test, feature_names):
    """SHAP値による特徴量の影響分析"""
    import shap

    # TreeExplainer（木ベースモデル用、高速）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary Plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 個別サンプルの説明
    shap.plots.waterfall(explainer(X_test)[0], show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    return shap_values
```

---

## 比較表

### 分類評価指標の使い分け

| 指標 | 数式 | 重視する場面 | クラス不均衡への耐性 |
|---|---|---|---|
| Accuracy | (TP+TN)/N | クラス均衡時の全体評価 | 弱い |
| Balanced Accuracy | (TPR+TNR)/2 | クラス不均衡時の全体評価 | 強い |
| Precision | TP/(TP+FP) | 偽陽性のコストが高い（スパム検出） | 中程度 |
| Recall | TP/(TP+FN) | 偽陰性のコストが高い（がん検診） | 中程度 |
| F1-Score | 2×P×R/(P+R) | PrecisionとRecallのバランス | 中程度 |
| F-beta | (1+β²)PR/(β²P+R) | βでP/Rの重みを調整 | 中程度 |
| MCC | (TP×TN-FP×FN)/√... | 全てのセルを考慮した総合指標 | 強い |
| AUC-ROC | ROC曲線下面積 | 閾値に依存しない総合評価 | 中程度 |
| Average Precision | PR曲線下面積 | クラス不均衡時の陽性検出能力 | 強い |
| Log Loss | -Σ y log(p) | 確率予測の精度 | 中程度 |
| Cohen's Kappa | (accuracy-expected)/(1-expected) | 偶然の一致を除いた評価 | 強い |

### 回帰評価指標の比較

| 指標 | 数式 | スケール依存 | 外れ値耐性 | 解釈性 |
|---|---|---|---|---|
| MSE | Σ(y-y_hat)²/n | あり | 弱い（二乗） | 低い |
| RMSE | √MSE | あり（元の単位） | 弱い | 高い |
| MAE | Σ|y-y_hat|/n | あり（元の単位） | 中程度 | 高い |
| Median AE | median(|y-y_hat|) | あり（元の単位） | 強い | 高い |
| MAPE | Σ|y-y_hat|/y /n | なし（%） | 弱い | 高い |
| R² | 1-SS_res/SS_tot | なし（0〜1） | 弱い | 高い |
| Adjusted R² | 調整済みR² | なし | 弱い | 高い |
| Explained Var | 1-Var(res)/Var(y) | なし（0〜1） | 弱い | 中程度 |

### 学習パラダイムの比較

| パラダイム | データの種類 | 主なタスク | 代表例 |
|---|---|---|---|
| 教師あり学習 | ラベル付き | 分類・回帰 | SVM, RF, XGBoost |
| 教師なし学習 | ラベルなし | クラスタリング・次元削減 | K-means, PCA |
| 半教師あり | 一部ラベル付き | 少量ラベルでの分類 | Self-Training, Label Propagation |
| 自己教師あり | ラベルなし（pretext task） | 表現学習 | BERT, SimCLR |
| 強化学習 | 報酬信号 | 逐次意思決定 | DQN, PPO |
| 転移学習 | 事前学習済みモデル | ドメイン適応 | Fine-tuning |

---

## アンチパターン

### アンチパターン1: 不均衡データでのAccuracy信仰

```python
# BAD: 99%が正常、1%が不正のデータで Accuracy を使う
# "全て正常と予測" → Accuracy 99% だが不正検出能力ゼロ

# GOOD: 不均衡データでは適切な指標を選択
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 方法1: class_weight でクラス不均衡に対処
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# F1, Recall, PR-AUC で評価
print(classification_report(y_test, y_pred))

# 方法2: SMOTEでオーバーサンプリング
smote_pipe = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])
smote_pipe.fit(X_train, y_train)
y_pred_smote = smote_pipe.predict(X_test)
print(f"SMOTE + RF F1: {f1_score(y_test, y_pred_smote):.4f}")

# 方法3: コスト敏感学習
from sklearn.ensemble import GradientBoostingClassifier
# sample_weight で誤分類コストを反映
sample_weights = np.where(y_train == 1, 10.0, 1.0)  # 少数クラスに10倍の重み
model_weighted = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_weighted.fit(X_train, y_train, sample_weight=sample_weights)
```

### アンチパターン2: テストデータでのハイパーパラメータ調整

```python
# BAD: テストデータでスコアを見ながら調整 → 情報リーク
for max_depth in [3, 5, 10, 20]:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # テストで評価 → NG
    print(f"depth={max_depth}: {score:.4f}")

# GOOD: 検証データ or 交差検証で調整、テストは最後に1回だけ
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 5, 10, 20]}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring="f1"
)
grid.fit(X_train, y_train)  # 訓練+検証のみ
print(f"最良パラメータ: {grid.best_params_}")
print(f"テストスコア: {grid.score(X_test, y_test):.4f}")  # 最後に1回
```

### アンチパターン3: 前処理でのデータリーク

```python
# BAD: テストデータを含めて fit する
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(np.vstack([X_train, X_test]))  # リーク！
X_train_scaled = X_all_scaled[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]

# GOOD: 訓練データのみで fit し、テストには transform のみ
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)  # transform のみ
```

### アンチパターン4: 特徴量選択でのデータリーク

```python
# BAD: 全データで特徴量選択してからCV
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)  # 全データで選択 → リーク
scores = cross_val_score(SVC(), X_selected, y, cv=5)

# GOOD: パイプラインに含めてCV内で選択
pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=10)),
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])
scores = cross_val_score(pipe, X, y, cv=5)  # 各Fold内で選択
```

### アンチパターン5: 時系列データでのランダム分割

```python
# BAD: 時系列データに通常のランダム分割を使う
# 未来の情報で過去を予測する「先読み」が発生
X_train, X_test, y_train, y_test = train_test_split(
    X_timeseries, y_timeseries, test_size=0.2, random_state=42  # NG
)

# GOOD: 時系列の順序を維持する分割
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X_timeseries):
    X_train = X_timeseries[train_idx]
    X_test = X_timeseries[test_idx]
    # 常に過去→未来の方向で学習→評価
    assert train_idx.max() < test_idx.min(), "時系列の順序が崩れている"
```

---

## 10. モデル選択のフローチャート

```
データの特性に基づくモデル選択:

                     ┌──────────────────┐
                     │ タスクの種類は？  │
                     └────────┬─────────┘
              ┌───────────────┼───────────────┐
              v               v               v
        ┌──────────┐   ┌──────────┐   ┌──────────────┐
        │  分類    │   │  回帰    │   │ クラスタリング│
        └────┬─────┘   └────┬─────┘   └──────┬───────┘
             │              │                 │
        ┌────┴────┐    ┌────┴────┐      ┌─────┴─────┐
        │ 線形分離│    │ 線形関係│      │ K既知?    │
        │ 可能?   │    │ ?       │      └─────┬─────┘
        └────┬────┘    └────┬────┘        Y/   \N
         Y/   \N        Y/   \N          /       \
        /       \      /       \     K-means   DBSCAN
 ロジスティック  │  線形回帰   │     GMM      階層的
 SVM(線形)    │  Ridge/Lasso │
              │              │
        ┌─────┴─────┐  ┌────┴─────┐
        │ データ量  │  │ データ量 │
        │ 多い?     │  │ 多い?    │
        └─────┬─────┘  └────┬─────┘
         Y/    \N       Y/    \N
        /        \     /        \
  アンサンブル  SVM  アンサンブル  SVR
  (RF/XGB/   (RBF)  (RF/XGB)    KNN回帰
   LightGBM)
```

### 主要アルゴリズムの特性比較

```
アルゴリズム選択の目安:

  アルゴリズム     訓練速度  推論速度  解釈性  非線形  大規模データ
  ──────────────────────────────────────────────────────────────
  線形回帰/LR     ★★★★★   ★★★★★   ★★★★★  ✗       ★★★★★
  SVM             ★★★☆☆   ★★★★☆   ★★☆☆☆  ✓       ★★☆☆☆
  決定木          ★★★★★   ★★★★★   ★★★★☆  ✓       ★★★★☆
  Random Forest   ★★★★☆   ★★★★☆   ★★★☆☆  ✓       ★★★★☆
  XGBoost/LGBM    ★★★★☆   ★★★★★   ★★☆☆☆  ✓       ★★★★★
  KNN             ★★★★★   ★☆☆☆☆   ★★★☆☆  ✓       ★☆☆☆☆
  Neural Network  ★☆☆☆☆   ★★★★☆   ★☆☆☆☆  ✓       ★★★★★
```

---

## トラブルシューティング

### よくある問題と対処法

| 問題 | 症状 | 対処法 |
|---|---|---|
| 過学習 | Train高/Test低 | 正則化強化、データ増加、モデル簡素化 |
| 過少適合 | Train低/Test低 | モデル複雑化、特徴量追加、学習率調整 |
| クラス不均衡 | 少数クラスの指標が低い | SMOTE、class_weight、コスト敏感学習 |
| データリーク | CVスコアが高すぎる | パイプライン化、前処理をCV内に含める |
| 特徴量多すぎ | 訓練遅い・過学習 | PCA、特徴量選択、正則化 |
| メモリ不足 | MemoryError | ミニバッチ学習、特徴量削減、sparse行列 |
| 収束しない | ConvergenceWarning | max_iter増加、学習率調整、正規化 |
| NaN予測 | 予測にNaN含む | 欠損値処理、特徴量のスケーリング確認 |

### デバッグチェックリスト

```python
def ml_debug_checklist(X_train, X_test, y_train, y_test, model):
    """機械学習パイプラインのデバッグチェックリスト"""
    print("=" * 60)
    print("ML デバッグチェックリスト")
    print("=" * 60)

    # 1. データの基本統計
    print("\n1. データ形状:")
    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}, y_test: {y_test.shape}")

    # 2. 欠損値チェック
    nan_train = np.isnan(X_train).sum()
    nan_test = np.isnan(X_test).sum()
    print(f"\n2. 欠損値: Train={nan_train}, Test={nan_test}")
    if nan_train > 0 or nan_test > 0:
        print("   ⚠ 欠損値あり → SimpleImputer等で処理が必要")

    # 3. 無限値チェック
    inf_train = np.isinf(X_train).sum()
    inf_test = np.isinf(X_test).sum()
    print(f"\n3. 無限値: Train={inf_train}, Test={inf_test}")

    # 4. クラス分布
    if len(np.unique(y_train)) < 20:
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n4. クラス分布 (Train):")
        for u, c in zip(unique, counts):
            print(f"   Class {u}: {c} ({c/len(y_train)*100:.1f}%)")
        imbalance_ratio = counts.max() / counts.min()
        if imbalance_ratio > 5:
            print(f"   ⚠ 不均衡比: {imbalance_ratio:.1f}x → SMOTE/class_weight検討")

    # 5. スケールチェック
    print(f"\n5. 特徴量のスケール:")
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    print(f"   平均の範囲: [{means.min():.2f}, {means.max():.2f}]")
    print(f"   標準偏差の範囲: [{stds.min():.2f}, {stds.max():.2f}]")
    if means.max() - means.min() > 100 or stds.max() / (stds.min() + 1e-10) > 100:
        print("   ⚠ スケールのばらつき大 → StandardScaler推奨")

    # 6. 定数特徴量チェック
    const_features = np.where(stds == 0)[0]
    if len(const_features) > 0:
        print(f"\n6. 定数特徴量: {len(const_features)}個 → 除去推奨")

    # 7. 高相関特徴量チェック
    corr = np.corrcoef(X_train.T)
    np.fill_diagonal(corr, 0)
    high_corr = np.where(np.abs(corr) > 0.95)
    n_high_corr = len(high_corr[0]) // 2
    if n_high_corr > 0:
        print(f"\n7. 高相関ペア (|r|>0.95): {n_high_corr}ペア → 一方を除去検討")

    print("\n" + "=" * 60)

# 使用例
# ml_debug_checklist(X_train_s, X_test_s, y_train, y_test, model)
```

---

## FAQ

### Q1: 教師なし学習の「正解」はどう評価するのか？

**A:** 外部評価（正解ラベルがある場合）と内部評価（ラベルなし）の2種類がある。内部評価ではシルエットスコア（クラスタの分離度）、エルボー法（慣性の減少率）、Davies-Bouldin指数等を使う。ただし、最終的にはドメイン知識による定性的評価が不可欠である。

### Q2: 交差検証の K はいくつが良い？

**A:** 一般的には K=5 または K=10 が標準。データが少ない場合は K を大きく（Leave-One-Outまで）、計算コストが高い場合は K=3〜5 で十分。RepeatedKFold（繰り返し交差検証）で分散を安定させる手法もある。

### Q3: 過学習を防ぐ方法にはどんなものがある？

**A:** 主な対策: (1) 正則化（L1/L2、Dropout）、(2) 早期停止（Early Stopping）、(3) データ拡張、(4) モデルの複雑度制限（max_depth等）、(5) アンサンブル学習（Bagging、Boosting）、(6) 交差検証による適切な評価。最も効果的なのは「訓練データを増やす」こと。

### Q4: F1スコアとMCC、どちらを使うべきか？

**A:** MCCは4セルすべて（TP, TN, FP, FN）を考慮するため、クラス不均衡時にF1より信頼性が高い。ただしF1の方が直感的で広く使われている。論文や厳密な評価ではMCCを推奨する研究が増えている。実務では両方を報告し、用途に応じて判断するのが望ましい。

### Q5: Optunaと Grid Search はどう使い分けるか？

**A:** パラメータ空間が小さい（2-3次元、各10値以内）ならGrid Searchで十分。パラメータ空間が広い場合や、連続値パラメータが多い場合はOptunaが効率的。特にLightGBMやXGBoostなど多パラメータのモデルではOptunaのベイズ最適化が圧倒的に効率的である。

### Q6: 特徴量重要度の方法はどう選ぶか？

**A:** 不純度ベースの重要度（Random Forestのfeature_importances_）はカーディナリティの高い特徴量を過大評価しがちなので、Permutation Importanceを優先する。さらに詳細な分析にはSHAP値を使い、個別サンプルレベルでの特徴量の影響を可視化できる。

### Q7: テストデータとバリデーションデータの違いは？

**A:** バリデーションデータはハイパーパラメータ調整やモデル選択に使用する。テストデータは最終評価にのみ使用し、モデル構築のどの段階でも情報を利用してはならない。交差検証ではバリデーションの役割をCV内で行うため、テストデータは一切触れない。3分割（Train/Validation/Test）の原則を厳守すること。

---

## 実践的なワークフロー: エンドツーエンドの例

```python
"""
完全なMLワークフロー: 顧客離脱予測
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")


class ChurnPredictionPipeline:
    """顧客離脱予測の実践パイプライン"""

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline = None
        self.best_params = None

    def _create_preprocessor(self):
        """前処理パイプラインの作成"""
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False,
                                       handle_unknown="ignore")),
        ])
        return ColumnTransformer([
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features),
        ])

    def optimize(self, X, y, n_trials=50):
        """Optunaでハイパーパラメータを最適化"""
        preprocessor = self._create_preprocessor()

        def objective(trial):
            model_name = trial.suggest_categorical("model",
                ["rf", "gb", "lr"])

            if model_name == "rf":
                classifier = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 15),
                    random_state=42,
                )
            elif model_name == "gb":
                classifier = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    learning_rate=trial.suggest_float("lr", 0.01, 0.3, log=True),
                    random_state=42,
                )
            else:
                classifier = LogisticRegression(
                    C=trial.suggest_float("C", 0.01, 100, log=True),
                    max_iter=1000, random_state=42,
                )

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ])

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
                scores.append(roc_auc_score(y.iloc[val_idx], y_prob))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        print(f"最良AUC-ROC: {study.best_value:.4f}")
        print(f"最良パラメータ: {self.best_params}")
        return study

    def train(self, X_train, y_train):
        """最適パラメータでモデルを学習"""
        preprocessor = self._create_preprocessor()

        model_name = self.best_params.get("model", "gb")
        params = {k: v for k, v in self.best_params.items() if k != "model"}

        if model_name == "rf":
            classifier = RandomForestClassifier(random_state=42, **params)
        elif model_name == "gb":
            params_gb = {k.replace("lr", "learning_rate"): v
                         for k, v in params.items()}
            classifier = GradientBoostingClassifier(random_state=42, **params_gb)
        else:
            classifier = LogisticRegression(max_iter=1000, random_state=42, **params)

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """テストデータで評価"""
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 50)
        print("テスト評価結果")
        print("=" * 50)
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    def save(self, path):
        """モデルの保存"""
        joblib.dump(self.pipeline, path)
        print(f"モデルを保存: {path}")

    def load(self, path):
        """モデルの読込"""
        self.pipeline = joblib.load(path)
        print(f"モデルを読込: {path}")
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| 教師あり学習 | 入力Xとラベルyからf(X)≈yを学習。回帰と分類に大別 |
| 教師なし学習 | ラベルなしでデータの構造を発見。クラスタリング・次元削減・異常検知 |
| 半教師あり/自己教師あり | 少量ラベルまたはpretext taskで学習。大規模モデルの基盤技術 |
| 強化学習 | 報酬信号から逐次意思決定の方策を学習 |
| バイアス-バリアンス | 総誤差=バイアス²+バリアンス+ノイズ。モデル複雑度で制御 |
| 交差検証 | K-Fold/Stratified/TimeSeries。テストデータは最後に1回だけ |
| 分類指標 | 不均衡ではF1/PR-AUC/MCC。閾値に依存しない評価はROC-AUC |
| 回帰指標 | RMSE（元の単位）、R²（説明率）、MAPE（%で比較可能） |
| ハイパーパラメータ | 少パラメータはGrid、多パラメータはOptuna（ベイズ最適化） |
| 特徴量重要度 | Permutation Importance推奨、詳細分析はSHAP |
| データリーク防止 | 全前処理をパイプライン内に含め、CV内で個別にfit |

---

## 次に読むべきガイド

- [03-python-ml-stack.md](./03-python-ml-stack.md) — Python ML開発環境の詳細
- [../01-classical-ml/00-regression.md](../01-classical-ml/00-regression.md) — 回帰モデルの実装
- [../01-classical-ml/01-classification.md](../01-classical-ml/01-classification.md) — 分類モデルの実装

---

## 参考文献

1. **Trevor Hastie, Robert Tibshirani, Jerome Friedman** "The Elements of Statistical Learning" 2nd Edition, Springer, 2009
2. **scikit-learn** "Model evaluation: quantifying the quality of predictions" — https://scikit-learn.org/stable/modules/model_evaluation.html
3. **Google Developers** "Machine Learning Crash Course" — https://developers.google.com/machine-learning/crash-course
4. **Optuna** "A hyperparameter optimization framework" — https://optuna.readthedocs.io/
5. **SHAP** "SHapley Additive exPlanations" — https://shap.readthedocs.io/
6. **Chicco, D., Jurman, G.** "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation" BMC Genomics, 2020
