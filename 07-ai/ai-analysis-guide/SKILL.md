# AI 解析ガイド

> AI/ML はデータから価値を引き出す技術。機械学習の基礎、ディープラーニング、自然言語処理、コンピュータビジョン、実践的なモデル開発フローまで、AI 解析の全てを体系的に解説する。

## このSkillの対象者

- AI/ML の基礎を体系的に学びたいエンジニア
- データ分析・予測モデル構築に取り組みたい方
- 業務で AI を活用したい方

## 前提知識

- Python の基礎知識
- 数学の基礎（線形代数、確率統計の概念）

## 学習ガイド

### 00-fundamentals — AI/ML の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-ai-ml-overview.md]] | AI/ML/DL の関係、歴史、現在のトレンド |
| 01 | [[docs/00-fundamentals/01-math-foundations.md]] | 線形代数、微積分、確率統計の ML に必要な範囲 |
| 02 | [[docs/00-fundamentals/02-ml-workflow.md]] | データ収集→前処理→特徴量→学習→評価→デプロイ |
| 03 | [[docs/00-fundamentals/03-tools-and-ecosystem.md]] | Python ML スタック（NumPy/Pandas/Scikit-learn/PyTorch） |

### 01-ml-basics — 機械学習の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-ml-basics/00-supervised-learning.md]] | 線形回帰、ロジスティック回帰、SVM、決定木、ランダムフォレスト |
| 01 | [[docs/01-ml-basics/01-unsupervised-learning.md]] | k-means、階層クラスタリング、PCA、t-SNE |
| 02 | [[docs/01-ml-basics/02-evaluation-and-tuning.md]] | 交差検証、精度指標、ハイパーパラメータチューニング |
| 03 | [[docs/01-ml-basics/03-feature-engineering.md]] | 特徴量設計、エンコーディング、特徴量選択、AutoML |

### 02-deep-learning — ディープラーニング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-deep-learning/00-neural-network-basics.md]] | ニューラルネットワーク、活性化関数、バックプロパゲーション |
| 01 | [[docs/02-deep-learning/01-cnn.md]] | CNN、畳み込み、プーリング、画像分類、物体検出 |
| 02 | [[docs/02-deep-learning/02-rnn-and-transformer.md]] | RNN/LSTM、Attention、Transformer アーキテクチャ |
| 03 | [[docs/02-deep-learning/03-training-techniques.md]] | 最適化、正則化、データ拡張、転移学習 |

### 03-practical — 実践応用

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-practical/00-nlp.md]] | 自然言語処理、テキスト分類、感情分析、NER |
| 01 | [[docs/03-practical/01-computer-vision.md]] | 画像分類、物体検出（YOLO）、セグメンテーション |
| 02 | [[docs/03-practical/02-mlops.md]] | MLOps、実験管理（MLflow）、モデルデプロイ、監視 |
| 03 | [[docs/03-practical/03-ai-ethics.md]] | AI 倫理、バイアス、公平性、説明可能性（XAI） |

## クイックリファレンス

```
ML アルゴリズム選定:
  分類 → ロジスティック回帰 → ランダムフォレスト → XGBoost → NN
  回帰 → 線形回帰 → ランダムフォレスト → XGBoost → NN
  クラスタリング → k-means → DBSCAN → 階層
  次元削減 → PCA → t-SNE → UMAP
  テキスト → Transformer → BERT → GPT
  画像 → CNN → ResNet → Vision Transformer
```

## 参考文献

1. Goodfellow, I. et al. "Deep Learning." MIT Press, 2016.
2. Géron, A. "Hands-On Machine Learning." O'Reilly, 2022.
3. Vaswani, A. et al. "Attention Is All You Need." NeurIPS, 2017.
