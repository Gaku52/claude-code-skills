# 倫理的考慮 -- 著作権、ディープフェイク、AI生成コンテンツの責任

> AI画像・映像生成技術がもたらす倫理的課題を、著作権・肖像権・ディープフェイク・バイアス・環境負荷の観点から体系的に分析し、責任あるAI活用のためのガイドラインと実務上の判断基準を提示する

## この章で学ぶこと

1. **著作権と知的財産の論点** -- AI生成物の著作権帰属、学習データの権利処理、フェアユースの範囲と各国法制度の動向
2. **ディープフェイクと肖像権** -- 顔画像合成の技術的検出手法、法規制、同意なき生成への対策フレームワーク
3. **責任あるAI活用の実践** -- コンテンツ認証（C2PA）、透明性の確保、バイアス対策、組織ガイドライン策定

---

## 1. AI 生成コンテンツの倫理的課題マップ

### 1.1 課題の全体構成

```
AI 生成コンテンツの倫理的課題

  法的課題                    社会的課題                技術的課題
  +-----------+             +-----------+            +-----------+
  | 著作権    |             | ディープ   |            | バイアス   |
  | 帰属問題  |             | フェイク   |            | 再生産    |
  +-----------+             +-----------+            +-----------+
  | 肖像権    |             | 誤情報     |            | 環境負荷   |
  | パブリシティ|            | 拡散      |            | (計算資源) |
  +-----------+             +-----------+            +-----------+
  | 商標権    |             | 同意なき   |            | 透明性    |
  | 侵害リスク |            | 生成      |            | 説明可能性 |
  +-----------+             +-----------+            +-----------+
  | 学習データ |             | 文化的    |            | 検出困難性 |
  | の権利    |             | 盗用      |            | (真偽判定) |
  +-----------+             +-----------+            +-----------+
```

### 1.2 ステークホルダー関係図

```
ステークホルダーと影響範囲

  [AI 開発企業]
       |
       | 学習データ収集
       v
  [アーティスト / クリエイター] ←── 作品が無断学習に使われる懸念
       |
       | AI ツール使用
       v
  [コンテンツ制作者] ──→ AI 生成物を公開
       |
       v
  [消費者 / 一般市民] ←── 真偽の判断が困難
       |
       v
  [プラットフォーム] ←── モデレーション責任
       |
       v
  [規制当局] ←── 法整備・ガイドライン策定
```

### 1.3 リスクレベルマトリクス

```
  影響度
  High |  肖像権侵害    ディープフェイク    児童搾取画像
       |  (個人被害)     (社会的混乱)       (違法)
       |
  Mid  |  著作権侵害    バイアス再生産      誤情報拡散
       |  (権利者被害)   (差別助長)         (信頼低下)
       |
  Low  |  スタイル模倣   環境負荷           透明性不足
       |  (グレーゾーン) (間接的影響)       (信頼低下)
       +-----------------------------------------------
       Low              Mid               High
                     発生頻度
```

---

## 2. 著作権と知的財産

### 2.1 AI 生成物の著作権帰属

```python
# AI 生成物の著作権判断フレームワーク (擬似コード)

class CopyrightAnalyzer:
    """AI 生成コンテンツの著作権リスク分析"""

    def analyze_copyright_status(self, content_metadata):
        """生成物の著作権状態を分析"""
        result = {
            "human_authorship": self._assess_human_contribution(content_metadata),
            "training_data_risk": self._assess_training_data_risk(content_metadata),
            "similarity_risk": self._assess_similarity(content_metadata),
            "jurisdiction_rules": self._get_jurisdiction_rules(content_metadata),
        }
        return result

    def _assess_human_contribution(self, metadata):
        """人間の創作的寄与の評価"""
        # 米国著作権局の基準 (2023年ガイダンス):
        # AI 生成部分は著作権保護の対象外
        # 人間の創作的選択・配置が著作権の根拠
        contribution_factors = {
            "prompt_complexity": metadata.get("prompt_length", 0) > 100,
            "human_editing": metadata.get("post_editing", False),
            "creative_selection": metadata.get("selection_from_variants", False),
            "artistic_arrangement": metadata.get("composition_design", False),
        }
        score = sum(contribution_factors.values()) / len(contribution_factors)
        return {
            "score": score,
            "likely_copyrightable": score >= 0.5,
            "note": "人間の創作的寄与が多いほど著作権保護の可能性が高い"
        }

    def _assess_training_data_risk(self, metadata):
        """学習データに関するリスク評価"""
        model_risks = {
            "adobe_firefly": "低 (ライセンス済みデータのみ)",
            "stable_diffusion": "中 (LAION-5B: 権利未処理あり)",
            "midjourney": "中 (Web スクレイピング含む)",
            "dall_e_3": "中〜低 (フィルタリング済み)",
        }
        model_name = metadata.get("model", "unknown")
        return model_risks.get(model_name, "不明 (要調査)")


# 各国の法的立場
copyright_by_jurisdiction = {
    "日本": {
        "ai_output_copyright": "人間の創作的寄与が認められれば著作物として保護",
        "training_data": "著作権法30条の4: 機械学習目的の利用は原則適法",
        "key_cases": "2024年文化審議会AI著作権ガイドライン",
        "note": "学習段階と生成・利用段階を区別して判断",
    },
    "米国": {
        "ai_output_copyright": "純粋なAI生成物は著作権保護の対象外",
        "training_data": "フェアユースの範囲で議論中",
        "key_cases": "Thaler v. Perlmutter (2023), "
                     "NYT v. OpenAI (2023)",
        "note": "人間の creative authorship が必要",
    },
    "EU": {
        "ai_output_copyright": "AI Act (2024) で透明性義務を規定",
        "training_data": "オプトアウト権を保障 (DSM Directive Art.4)",
        "key_cases": "EU AI Act (2024年施行)",
        "note": "生成AI にはサマリーの公開義務あり",
    },
}
```

### 2.2 学習データの権利問題

```python
# 学習データの権利チェックフロー

def check_training_data_compliance(model_info):
    """モデルの学習データコンプライアンスを確認"""

    checklist = {
        "licensed_data": {
            "question": "学習データはライセンス済みか？",
            "adobe_firefly": True,   # Adobe Stock のみ
            "stable_diffusion": False,  # LAION-5B (Webスクレイピング)
            "dall_e_3": "部分的",      # フィルタリング済み
        },
        "opt_out_respected": {
            "question": "クリエイターのオプトアウトは尊重されているか？",
            "robots_txt": "一部のモデルは robots.txt を無視",
            "do_not_train": "meta タグ対応が進行中",
            "spawning_ai": "Have I Been Trained? でオプトアウト可能",
        },
        "attribution": {
            "question": "学習元への帰属表示はあるか？",
            "current_state": "ほぼ全てのモデルで帰属表示なし",
            "ideal": "学習に使用したデータソースの開示",
        },
    }
    return checklist


# HTML meta タグによるオプトアウト
opt_out_html = """
<!-- AI 学習からの除外を要求 -->
<meta name="robots" content="noai, noimageai">

<!-- robots.txt での除外 -->
# robots.txt
User-agent: GPTBot
Disallow: /

User-agent: CCBot
Disallow: /

User-agent: Google-Extended
Disallow: /
"""
```

### 2.3 商用利用の判断基準

```
商用利用可否の判断フロー

  [AI 生成画像を商用利用したい]
           |
           v
  [モデルのライセンスを確認]
           |
     +-----+------+
     |             |
  商用可         商用不可/要確認
  (Firefly,       (一部のOSSモデル)
   Midjourney      |
   有料プラン,     v
   DALL-E)      [ライセンス条件を
                  詳細確認]
     |
     v
  [生成物が既存作品に酷似していないか確認]
     |
     +-----+------+
     |             |
  酷似なし       酷似あり
     |             |
     v             v
  [クライアントへ   [使用を避ける
   AI生成の旨を     別の画像を生成]
   開示するか検討]
     |
     v
  [コンテンツ認証情報 (C2PA) を付与]
     |
     v
  [商用利用OK]
```

---

## 3. ディープフェイクと肖像権

### 3.1 ディープフェイク検出技術

```python
# ディープフェイク検出パイプライン (擬似コード)
import torch
from deepfake_detection import FaceForensicsDetector

class DeepfakeDetectionPipeline:
    """ディープフェイク検出の多層アプローチ"""

    def __init__(self):
        # 複数の検出手法を組み合わせる
        self.detectors = {
            "frequency_analysis": FrequencyAnalysisDetector(),
            "face_forensics": FaceForensicsDetector(),
            "lip_sync": LipSyncConsistencyChecker(),
            "metadata": MetadataAnalyzer(),
            "c2pa": C2PAVerifier(),
        }

    def analyze(self, media_path):
        """メディアファイルの真正性を分析"""
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.detect(media_path)

        # 総合判定
        fake_scores = [r["fake_probability"] for r in results.values()
                       if "fake_probability" in r]
        avg_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0

        return {
            "overall_fake_probability": avg_score,
            "verdict": "LIKELY_FAKE" if avg_score > 0.7 else
                       "SUSPICIOUS" if avg_score > 0.4 else "LIKELY_REAL",
            "detailed_results": results,
            "confidence": "high" if len(fake_scores) >= 3 else "low",
        }


class FrequencyAnalysisDetector:
    """周波数領域解析によるディープフェイク検出"""

    def detect(self, media_path):
        # GAN 生成画像は特定の周波数パターンを持つ
        # DCT (離散コサイン変換) でスペクトル異常を検出
        image = load_image(media_path)
        dct_spectrum = compute_dct(image)
        anomaly_score = detect_spectral_anomaly(dct_spectrum)
        return {
            "fake_probability": anomaly_score,
            "method": "DCT Spectral Analysis",
            "note": "GAN固有の周波数パターンを検出",
        }
```

### 3.2 コンテンツ認証 (C2PA)

```python
# C2PA (Coalition for Content Provenance and Authenticity)
# コンテンツの来歴を証明する技術標準

class C2PAManager:
    """C2PA 準拠のコンテンツ認証管理"""

    def sign_content(self, content_path, metadata):
        """コンテンツに C2PA マニフェストを付与"""
        manifest = {
            "claim_generator": "MyApp/1.0",
            "title": metadata.get("title", "Untitled"),
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "softwareAgent": metadata.get("tool", "Unknown"),
                                "digitalSourceType": self._get_source_type(metadata),
                            }
                        ]
                    }
                },
                {
                    "label": "c2pa.ai_training",
                    "data": {
                        "use": metadata.get("ai_training_allowed", "notAllowed"),
                    }
                }
            ],
        }
        # 電子署名でマニフェストの改ざんを防止
        signed_manifest = self._sign_with_certificate(manifest)
        return self._embed_manifest(content_path, signed_manifest)

    def _get_source_type(self, metadata):
        """コンテンツの生成方法を分類"""
        source_types = {
            "human_created": "http://cv.iptc.org/newscodes/digitalsourcetype/humanCreated",
            "ai_generated": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            "ai_assisted": "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
            "captured": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
        }
        return source_types.get(metadata.get("source_type", ""), "unknown")


# C2PA 対応状況 (2025年時点)
c2pa_adoption = {
    "Adobe": "Photoshop, Lightroom で Content Credentials 付与",
    "Google": "SynthID で AI 生成物に透かしを埋め込み",
    "Microsoft": "Bing Image Creator で C2PA メタデータ付与",
    "OpenAI": "DALL-E 3 で C2PA メタデータ付与",
    "Meta": "Stable Signatures で生成物にマーキング",
    "カメラメーカー": "Nikon, Sony, Leica が撮影時の C2PA 対応",
}
```

### 3.3 肖像権と同意管理

```python
# 肖像権に配慮した生成フロー

class ConsentManager:
    """肖像権・パブリシティ権の同意管理"""

    CONSENT_LEVELS = {
        "explicit_written": 4,   # 書面での明示的同意
        "explicit_verbal": 3,    # 口頭での明示的同意
        "implied": 2,            # 黙示の同意 (公開の場での撮影等)
        "none": 0,               # 同意なし
    }

    def check_generation_allowed(self, request):
        """生成リクエストの肖像権チェック"""
        checks = {
            "real_person_detected": self._contains_real_person(request),
            "consent_level": self._get_consent_level(request),
            "purpose": request.get("purpose", "unknown"),
            "commercial_use": request.get("commercial", False),
        }

        # 判定ロジック
        if checks["real_person_detected"]:
            if checks["consent_level"] == "none":
                return {
                    "allowed": False,
                    "reason": "実在人物の画像生成には本人の同意が必要",
                    "recommendation": "架空のキャラクターを使用するか、同意を取得してください",
                }
            if checks["commercial_use"] and checks["consent_level"] != "explicit_written":
                return {
                    "allowed": False,
                    "reason": "商用利用には書面での明示的同意が必要",
                    "recommendation": "パブリシティ権のライセンス契約を締結してください",
                }

        return {"allowed": True, "conditions": checks}
```

---

## 4. バイアスと公平性

```python
# AI 画像生成におけるバイアス検出と緩和

class BiasAuditor:
    """生成画像のバイアス監査"""

    def audit_generation_results(self, prompt, generated_images):
        """プロンプトに対する生成結果のバイアスを監査"""

        audit_results = {
            "gender_distribution": self._check_gender_representation(generated_images),
            "racial_distribution": self._check_racial_representation(generated_images),
            "age_distribution": self._check_age_representation(generated_images),
            "body_type_diversity": self._check_body_diversity(generated_images),
            "stereotyping": self._check_stereotypes(prompt, generated_images),
        }

        # バイアスの具体例
        known_biases = [
            {
                "prompt": "CEO",
                "bias": "男性・白人の画像が過剰に生成される",
                "mitigation": "多様な属性を明示的にプロンプトに含める",
            },
            {
                "prompt": "nurse",
                "bias": "女性の画像が圧倒的に多い",
                "mitigation": "性別を指定しない、または多様な性別を生成",
            },
            {
                "prompt": "beautiful person",
                "bias": "特定の美の基準（痩身・若年・白人寄り）に偏る",
                "mitigation": "多様な美の基準を学習データに含める",
            },
            {
                "prompt": "家族",
                "bias": "核家族・異性カップル中心",
                "mitigation": "多様な家族構成を意識的に含める",
            },
        ]

        return {
            "audit": audit_results,
            "known_biases": known_biases,
            "recommendations": self._generate_recommendations(audit_results),
        }

    def _generate_recommendations(self, audit):
        """監査結果に基づく改善提案"""
        recommendations = []
        for dimension, result in audit.items():
            if result.get("bias_detected"):
                recommendations.append({
                    "dimension": dimension,
                    "action": f"{dimension} の多様性を改善してください",
                    "method": "プロンプトの明示的な多様性指定、"
                              "学習データの再バランシング、"
                              "生成後のフィルタリング",
                })
        return recommendations
```

---

## 5. 環境負荷

```
AI 画像生成の環境負荷

  モデル            1枚あたりの推定消費電力    CO2換算 (g)
  ─────────────────────────────────────────────────
  Stable Diffusion  0.01-0.05 kWh              5-25
  DALL-E 3          0.02-0.08 kWh              10-40
  Midjourney        0.01-0.04 kWh              5-20
  Sora (動画)       0.5-2.0 kWh (推定)         250-1000

  比較:
  - スマートフォン充電1回: 0.01 kWh ≈ 5g CO2
  - Google 検索1回: 0.0003 kWh ≈ 0.2g CO2
  - 画像生成1枚 ≈ スマホ充電 1-5回分

  学習フェーズの負荷:
  - Stable Diffusion 学習: ≈ 150,000 kWh
  - GPT-4 学習: ≈ 50,000,000 kWh (推定)
  - 日本の一般家庭の年間消費電力: ≈ 4,000 kWh

  緩和策:
  1. 必要最小限の生成に留める（不要な大量生成を避ける）
  2. 軽量モデル（SDXL Turbo 等）の活用
  3. 再生可能エネルギーで運用されるデータセンターの選択
  4. キャッシュの活用（同一プロンプトの再生成を避ける）
```

---

## 6. 組織ガイドライン策定

```python
# 組織向け AI 生成コンテンツ利用ガイドライン テンプレート

ai_content_policy = {
    "scope": "AI 画像・動画生成ツールの業務利用全般",

    "permitted_uses": [
        "コンセプトアートの初期段階での参考画像生成",
        "社内資料（プレゼン、企画書）への素材使用",
        "マーケティング素材の下書き・ラフ案作成",
        "学習データがライセンス済みのツール使用 (Adobe Firefly 等)",
    ],

    "restricted_uses": [
        "実在人物の顔を含む画像の生成（本人の書面同意が必要）",
        "競合他社のブランド要素を含む画像の生成",
        "著作権が明確な作品のスタイル模倣（特定アーティスト名の指定）",
        "最終納品物としての無編集 AI 生成画像の使用",
    ],

    "prohibited_uses": [
        "ディープフェイク（虚偽の映像・音声）の作成",
        "児童の性的コンテンツの生成",
        "ヘイトスピーチ・差別を助長するコンテンツの生成",
        "選挙・政治目的での偽情報の生成",
        "同意なき個人の画像生成（リベンジポルノ等）",
    ],

    "disclosure_requirements": [
        "クライアント納品物に AI 生成素材を含む場合は事前告知",
        "SNS 投稿の AI 生成画像には #AIGenerated タグ付与",
        "ニュース・報道での使用は明確な AI 生成表示が必須",
        "可能な限り C2PA メタデータを付与",
    ],

    "review_process": [
        "AI 生成コンテンツは公開前に法務チームの確認を経ること",
        "実在人物が含まれる場合は肖像権チェックを実施",
        "商標・ブランド要素の類似性チェックを実施",
        "バイアス・差別表現の確認を実施",
    ],
}
```

---

## 7. 比較表

| 観点 | リスクレベル | 法的枠組み | 技術的対策 | 組織的対策 |
|------|:----------:|:--------:|:--------:|:--------:|
| 著作権侵害 | 中 | 著作権法、AI Act | 類似度検出、学習データ管理 | ライセンス確認プロセス |
| 肖像権侵害 | 高 | 民法、肖像権判例 | 顔検出フィルタ | 同意管理フロー |
| ディープフェイク | 最高 | 各国規制法 | C2PA、SynthID、検出AI | 使用禁止ポリシー |
| バイアス | 中 | 差別禁止法 | バイアス監査 | 多様性チェックリスト |
| 環境負荷 | 低〜中 | ESG 規制 | 軽量モデル活用 | 生成量の管理 |
| 誤情報拡散 | 高 | プラットフォーム規制 | 透かし、メタデータ | 開示ポリシー |

| ツール | 学習データの透明性 | 商用利用 | IP補償 | C2PA対応 |
|--------|:----------------:|:------:|:-----:|:-------:|
| Adobe Firefly | 高 (ライセンス済み) | 有料プラン可 | あり | あり |
| Midjourney | 低 | 有料プラン可 | なし | なし |
| DALL-E 3 | 中 | API利用可 | なし | あり |
| Stable Diffusion | 中 (LAION-5B) | ライセンス依存 | なし | なし |
| Google Imagen | 中 | 限定的 | なし | SynthID |

---

## 8. アンチパターン

### アンチパターン 1: 「AI だから著作権フリー」という誤解

```
BAD:
  「AI が生成したから著作権は存在しない」
  「誰の著作物でもないから自由に使える」
  → 学習データに含まれる既存作品との類似性リスクを無視
  → 特定のアーティストのスタイルを意図的に模倣
  → 法的トラブルに発展するケース増加

GOOD:
  - AI 生成物でも既存作品との類似性チェックを実施
  - 学習データがライセンス済みのツールを優先使用
  - 特定アーティスト名をプロンプトに使用しない
  - 商用利用前にリバース画像検索で類似作品を確認
  - 法務チームへの相談プロセスを確立
```

### アンチパターン 2: 開示義務を怠る

```
BAD:
  AI 生成画像をニュース記事の報道写真として使用
  AI 生成のモデル画像を「実際の着用写真」として EC に掲載
  → 消費者の信頼を損ない、法的責任を問われる可能性

GOOD:
  - AI 生成コンテンツには必ず明示的なラベルを付与
  - 「この画像は AI によって生成されました」の表記
  - C2PA メタデータでコンテンツの来歴を記録
  - プラットフォームの AI 生成コンテンツポリシーに従う
  - 報道・ジャーナリズムでは AI 画像を使用しない
```

### アンチパターン 3: 同意なき肖像利用

```
BAD:
  有名人の顔を使った AI 生成画像を広告に使用
  元パートナーの写真を AI で加工して拡散
  → パブリシティ権侵害、名誉毀損、刑事罰の対象

GOOD:
  - 実在人物の画像生成には必ず書面での同意を取得
  - パブリシティ権のライセンス契約を締結
  - 架空のキャラクターを使用する選択肢を優先
  - 顔検出フィルタで実在人物の生成をブロック
  - 社内ポリシーで実在人物の AI 生成を原則禁止
```

---

## 9. FAQ

### Q1. AI 生成画像を商用利用する場合、何を確認すべきか？

**A.** (1) **ツールのライセンス**: 商用利用可能なプランか確認する（Midjourney: 有料プラン、Adobe Firefly: Creative Cloud、DALL-E: API利用規約）。(2) **類似性チェック**: Google リバース画像検索やTinEye で既存作品との類似度を確認。(3) **肖像権**: 実在人物に似ている場合はリスクあり。(4) **開示義務**: クライアントやプラットフォームの規約に基づき AI 生成であることを開示。(5) **IP補償**: Adobe Firefly は知的財産権侵害に対する補償（IP Indemnification）を提供しており、商用利用での安心感がある。

### Q2. ディープフェイク被害に遭った場合の対処法は？

**A.** (1) **証拠保全**: スクリーンショット、URL、投稿日時を記録する。(2) **プラットフォーム報告**: 各プラットフォームのディープフェイク通報機能を利用（Meta、Google、X は専用の報告フォームあり）。(3) **法的措置**: 弁護士に相談し、名誉毀損・肖像権侵害で損害賠償請求を検討。日本では令和5年（2023年）の刑法改正で性的ディープフェイクの作成・頒布が処罰対象に。(4) **検出ツール**: Sensity AI、Microsoft Video Authenticator 等の検出ツールで証拠を補強。

### Q3. AI 生成コンテンツのバイアスを軽減するには？

**A.** (1) **プロンプトの工夫**: 「diverse group of people」「various ethnicities and ages」等、多様性を明示的に指定する。(2) **生成結果の監査**: 100枚以上生成して性別・人種・年齢の分布を確認する。(3) **ネガティブプロンプト**: ステレオタイプな表現を除外指定する。(4) **複数モデルの比較**: 異なるモデルの結果を比較してバイアスの傾向を把握する。(5) **人間によるレビュー**: 最終的には多様な背景を持つチームメンバーがレビューする。完全なバイアス除去は現状困難だが、意識的な取り組みで軽減は可能。

### Q4. 日本の著作権法でのAI学習はどこまで許容されるか？

**A.** 日本の著作権法30条の4は、「著作物に表現された思想又は感情を自ら享受し又は他人に享受させることを目的としない場合」の利用を許容している。AI の機械学習はこれに該当するとされ、原則として権利者の許諾なく学習に利用できる。ただし (1) 著作権者の利益を不当に害する場合は例外、(2) 生成段階で特定の著作物に類似する出力を行う場合は侵害となりうる、(3) 2024年の文化審議会ガイドラインでは享受目的の学習は30条の4の対象外と整理されている。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 著作権 | AI 生成物の権利帰属は国・地域で異なる。人間の創作的寄与が鍵 |
| 学習データ | ライセンス済みデータのモデル（Adobe Firefly）が最も低リスク |
| ディープフェイク | C2PA、SynthID 等の技術的対策 + 法規制が進行中 |
| 肖像権 | 実在人物の画像生成は書面同意が必須。パブリシティ権に注意 |
| バイアス | プロンプト設計と生成結果の監査で軽減。完全除去は困難 |
| 組織ガイドライン | 許可/制限/禁止の3段階で明確なポリシーを策定 |
| 開示義務 | AI 生成コンテンツには明示的なラベルと C2PA メタデータを付与 |

---

## 次に読むべきガイド

- [バーチャル試着](./02-virtual-try-on.md) -- 3D + AI の応用と肖像権の交差点
- [デザインツール](../01-image/03-design-tools.md) -- 各ツールのライセンスと商用利用条件
- [動画編集](../02-video/01-video-editing.md) -- 動画 AI の倫理的利用

---

## 参考文献

1. **C2PA Technical Specification** -- https://c2pa.org/specifications/ -- コンテンツ認証の技術標準
2. **文化審議会 AI と著作権に関する考え方** -- 文化庁 (2024) -- 日本の AI 著作権ガイドライン
3. **EU AI Act** -- European Parliament (2024) -- EU の AI 規制法
4. **The Ethics of Artificial Intelligence** -- Jobin et al. (Nature Machine Intelligence, 2019) -- AI 倫理の国際的調査
