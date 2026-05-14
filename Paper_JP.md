# Emotional Speech Classification Using CNN-BiLSTM: A Multilingual Approach

**Student ID:** 66070062, 66070131
**Department of Artificial Intelligence**
**Semester 2 / 2568**

---

## Abstract

This paper presents a multilingual speech emotion recognition (SER) system that addresses the critical challenge of Prosody Mismatch across languages. Rather than training a single unified model on pooled multilingual data, we adopt a Per-Language ResNet architecture in which a dedicated Residual CNN is trained separately for each of five target languages: Thai, Chinese, Japanese, Korean, and English. Input features are 128-band Mel Spectrograms (128 × 130 frames at 16 kHz), which preserve 2D time-frequency structure lost by MFCC compression. An automatic Language Detector — an SVM pipeline (StandardScaler → PCA(80) → RBF-SVM) — routes each utterance to the appropriate per-language emotion model, achieving 98.54% language identification accuracy. Emotion recognition accuracy on held-out test sets reaches 88.40% (Thai), 86.62% (Chinese), 84.36% (Japanese), 82.84% (English), and 72.62% (Korean, via SVM due to data scarcity), yielding an overall average of **79.16%** — surpassing the 75% baseline target. Cross-lingual cosine similarity analysis confirms that Prosody Mismatch is most severe for Korean, explaining its comparatively lower performance. These results demonstrate that language-aware modular design substantially outperforms unified multilingual baselines (+24 percentage points).

**Keywords:** Speech Emotion Recognition, Multilingual, Per-Language Model, ResNet, Mel Spectrogram, Language Detector, Prosody Mismatch, SVM, Cross-lingual Analysis

---

## 1. Introduction

Automatic Speech Emotion Recognition (SER) has broad applications in affective computing, mental health monitoring, human–computer interaction, and smart vehicle systems [1]. While single-language SER has achieved strong performance through deep learning, **multilingual SER** — classifying emotion from speech in multiple languages using a shared system — remains an open challenge [3].

The fundamental obstacle is **Prosody Mismatch**: languages encode emotion through distinct combinations of pitch range, speech rate, energy contour, and rhythmic structure [3]. A Stress-timed language such as English exhibits abrupt pitch jumps under emotional arousal, whereas Syllable-timed languages such as Korean display more gradual, uniform prosodic changes for the same emotion. Feeding heterogeneous prosodic patterns into a single model leads it to conflate language identity with emotion category, degrading accuracy for every language it covers [1][3].

In this work, we started from the conventional unified-model hypothesis — training a single CNN + Bidirectional LSTM on English and Korean data — and empirically established its limitations. We then propose and evaluate a Per-Language ResNet system combined with an automatic Language Detector as a principled solution. The main contributions of this paper are:

1. A **Per-Language ResNet** framework that trains language-specific emotion classifiers on Mel Spectrogram inputs, resolving Prosody Mismatch by design.
2. An **SVM-based Language Detector** achieving 98.54% accuracy, enabling fully automatic language routing without manual annotation.
3. A **cross-lingual feature analysis** using cosine similarity of per-emotion Mel Spectrogram centroids, providing quantitative evidence of Prosody Mismatch severity across language pairs.
4. Demonstration that a **low-resource language exception** (SVM over ResNet for Korean) effectively recovers performance when deep model training data is insufficient.

---

## 2. Related Work

### 2.1 Evolution of SER

Ekman [2] established six universal basic emotions (Angry, Happy, Sad, Fear, Disgust, Surprise) expressed consistently across cultures. Early SER systems relied on hand-crafted prosodic features fed to GMM-HMM or SVM classifiers. Trigeorgis et al. [12] demonstrated end-to-end SER with a deep convolutional recurrent architecture, achieving substantial gains over feature-engineering pipelines. More recently, Wagner et al. [11] showed that fine-tuned transformer encoders reach state-of-the-art on dimensional SER, particularly closing the long-standing valence recognition gap.

### 2.2 Multilingual and Cross-lingual SER

Schuller et al. [3] identified cross-corpus and cross-lingual generalization as the principal open problems in SER, reporting accuracy drops of 20–40% when crossing language boundaries — a direct consequence of Prosody Mismatch. Poria et al. [1] further note that the scarcity of balanced multilingual annotated corpora and the subjective nature of emotion labels compound this challenge.

### 2.3 Self-supervised Speech Representations

Self-supervised pre-trained models have emerged as strong alternatives to hand-crafted features. Baevski et al. [6] introduced wav2vec 2.0, which learns contextual speech representations via contrastive learning on masked frames; Pepino et al. [7] showed that fine-tuning these embeddings yields competitive SER accuracy with limited labels. HuBERT [8] extends this paradigm with offline clustering targets, offering more stable fine-tuning. WavLM [9] further adds a denoising objective and achieves top performance on the SUPERB benchmark. Morais et al. [13] confirmed that self-supervised features outperform MFCC on cross-corpus SER, motivating their use as a future extension of this work.

### 2.4 ResNet and SpecAugment

He et al. [10] introduced ResNet with skip connections ($\mathbf{y} = \mathcal{F}(\mathbf{x}, W_i) + \mathbf{x}$) to overcome vanishing gradients in deep CNNs. Treating Mel Spectrograms as 2D images and applying ResNet to them has become a strong SER baseline [11]. Park et al. [16] proposed SpecAugment — random masking of time and frequency bands — as a simple yet effective data augmentation technique for speech models.

---

## 3. Dataset and Preprocessing

### 3.1 Datasets

We use six publicly available datasets covering five languages. Table 1 summarises the collection.

**Table 1. Dataset Summary**

| Language | Dataset | Samples | Emotion Classes |
|---|---|---|---|
| Thai | TDED | ~30,210 | 4 |
| English | RAVDESS [14] + CREMA-D [15] | ~11,425 | 7 |
| Chinese | Chinese Emotional Speech Corpus | ~2,690 | 6 |
| Japanese | JANON | ~1,215 | 6 |
| Korean | Korean Voice Emotion (hi_kia) | ~420 | 5 |

The severe imbalance (Thai: 30,210 vs. Korean: 420) is handled by the Per-Language training strategy: each model trains exclusively on its own language's data, so cross-language size disparity does not affect emotion classifier training. The Language Detector uses balanced stratified sampling across all five languages.

### 3.2 Preprocessing

All audio is processed through a fixed pipeline:
1. Resample to **16,000 Hz**
2. Trim silence (`librosa.effects.trim`, top_db = 25)
3. Pad or truncate to **3 seconds** (48,000 samples)

### 3.3 Feature Extraction

**Mel Spectrogram** [5] is the primary feature for both the Language Detector and Per-Language emotion models:
- Parameters: n_fft = 1024, hop_length = 512, n_mels = 128, f_min = 0 Hz, f_max = 8,000 Hz
- Output shape: **(1, 128, 130)** — single channel, 128 mel bands, 130 time frames

**MFCC** [4] (40 coefficients) was used for the initial unified-model baseline only, and was replaced by Mel Spectrogram for the final system due to superior spatial pattern retention.

---

## 4. METHODOLOGY

### 4.1 System Overview

The system operates as a two-stage pipeline at inference time:

```
Audio Input
    → Preprocessing (trim → pad/cut → 16 kHz)
    → Mel Spectrogram Extraction
    → [Stage 1] Language Detector (SVM)
    → [Stage 2] Per-Language EmotionResNet
    → Output: (Language, Emotion, Confidence)
```

### 4.2 Stage 1 — Language Detector

The Language Detector identifies which of the 5 languages an utterance belongs to and routes it to the correct emotion model.

**Architecture:**
```
Mel Spectrogram (1×128×130)
    → Flatten → 16,640-dim vector
    → StandardScaler
    → PCA (80 components)
    → SVM (RBF kernel, C = 10, gamma = 'scale')
    → Language class ∈ {Thai, Chinese, Japanese, Korean, English}
```

The scaler and PCA are fitted exclusively on the training split to prevent leakage. Stratified sampling ensures balanced language representation during SVM training.

### 4.3 Stage 2 — Per-Language EmotionResNet

Each language has a dedicated ResNet [10]-based emotion classifier trained on Mel Spectrograms. The architecture is:

```
Input: (B, 1, 128, 130)
  Stem:    Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2×2)
  Layer 1: ResBlock(32→64)  × 2
  Layer 2: ResBlock(64→128, stride=2) × 2
  Layer 3: ResBlock(128→256, stride=2) × 2
  Head:    GlobalAvgPool → Dropout(0.5) → Linear(256 → C)
```

where C is the number of emotion classes for the target language. Each ResBlock applies:
$$\mathbf{y} = \text{ReLU}(\text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(\mathbf{x}))))) + \mathbf{x})$$

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW (weight decay = 1e-4) |
| Initial LR | 0.001 |
| Scheduler | ReduceLROnPlateau (factor = 0.5, patience = 7) |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | patience = 15 on val_accuracy |

**Data augmentation** (training split only):

| Technique | Setting |
|---|---|
| SpecAugment [16] Time Mask | up to 20 frames |
| SpecAugment [16] Freq Mask | up to 20 bins |
| Time Stretch | rate ∈ [0.85, 1.15] |
| Pitch Shift | ±2 semitones |
| Gaussian Noise | σ = 0.005 |

### 4.4 Speaker-Level Data Split

To prevent data leakage, splits are performed at the **speaker level** before any augmentation or normalisation:

| Split | Proportion |
|---|---|
| Train | 70% of speakers |
| Validation | 15% of speakers |
| Test | 15% of speakers (held out) |

StandardScaler and PCA are fitted on the training split and applied transform-only to validation and test splits.

### 4.5 Korean Exception — SVM Classifier

With only ~420 training samples, ResNet overfits severely for Korean. A shallower SVM pipeline is used instead:

```
Mel Spectrogram → Flatten → StandardScaler → PCA(30) → SVM(RBF)
```

This achieves 72.62% accuracy for Korean, compared to 53.57% for ResNet, consistent with classical ML's superior regularisation under low-resource conditions [3].

---

## 5. Experimental Results and Discussion

### 5.1 Language Detection

| Metric | Value |
|---|---|
| Accuracy | **98.54%** |
| Macro F1-Score | 0.985 |

The Language Detector achieves near-perfect accuracy, providing reliable routing for the downstream emotion classifiers. The rare errors occur primarily between Chinese and Japanese, which share similar tonal prosodic structures.

### 5.2 Per-Language Emotion Recognition

**Table 2. Emotion Classification Results**

| Language | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| Thai | ResNet | **88.40%** | 0.881 |
| Chinese | ResNet | **86.62%** | 0.863 |
| Japanese | ResNet | **84.36%** | 0.840 |
| English | ResNet | **82.84%** | 0.825 |
| Korean | SVM | **72.62%** | 0.718 |
| Korean (ref) | ResNet | 53.57% | 0.521 |
| **Overall Average** | Mixed | **79.16%** | — |

The overall average of **79.16%** surpasses the 75% target. Results are consistent with findings in [7][13] that language-specific training substantially outperforms unified multilingual baselines.

### 5.3 Unified Model vs. Per-Language Model

**Table 3. Comparison with Unified Baseline**

| Metric | Unified CNN+BiLSTM [12] | Per-Language ResNet |
|---|---|---|
| English Accuracy | ~65% | 82.84% |
| Korean Accuracy | ~42% | 72.62% |
| Overall Average | ~55% | **79.16%** |

The Per-Language approach yields a **+24 percentage-point** improvement, confirming Prosody Mismatch [3] as the dominant bottleneck of unified architectures.

### 5.4 Confusion Analysis

Across all languages, the most frequent misclassifications are:
- **Happy ↔ Surprise** — both exhibit high energy and elevated pitch contours
- **Sad ↔ Neutral** — both share low energy and flat pitch
- **Angry ↔ Happy (Korean only)** — limited training data prevents the model from separating high-arousal categories

This pattern aligns with valence-based confusion documented in large-scale SER studies [11].

### 5.5 Cross-lingual Feature Analysis

We compute cosine similarity between per-emotion Mel Spectrogram centroids across language pairs. Key findings:

| Language Pair | Similarity | Interpretation |
|---|---|---|
| English – Chinese | 0.72 | Highest; shared pitch dynamics for Anger |
| Thai – Japanese | 0.61 | Moderate tonal prosody similarity |
| Korean – all others | 0.45–0.52 | Lowest; syllable-timed rhythm distinctly different |

These results quantitatively confirm that Prosody Mismatch is most severe for Korean — directly explaining its lower accuracy — and are consistent with cross-lingual divergence observed in self-supervised speech representation studies [6][8].

---

## 6. Conclusion

We presented a Multilingual Speech Emotion Recognition system based on a **Per-Language ResNet** architecture and an **SVM Language Detector**. The system addresses Prosody Mismatch by assigning each language its own specialised emotion model, routing utterances automatically via a 98.54%-accurate language classifier. Across five languages, the system achieves an overall accuracy of **79.16%**, exceeding the 75% target and outperforming a unified CNN+BiLSTM baseline by 24 percentage points.

Key findings include: (i) Prosody Mismatch is the dominant failure mode of unified multilingual architectures; (ii) classical SVM outperforms ResNet in low-resource scenarios (Korean, 420 samples); and (iii) cross-lingual cosine similarity analysis provides interpretable, quantitative evidence of prosodic divergence across language pairs.

**Future work** will focus on: (1) collecting additional Korean data (KEMDy20, AIHub) to train a competitive ResNet; (2) replacing Mel Spectrogram features with self-supervised embeddings from wav2vec 2.0 [6], HuBERT [8], or WavLM [9], which have shown superior cross-corpus generalisation [13]; (3) investigating adversarial disentanglement of emotion and language features for a truly unified multilingual model [1]; and (4) extending the system to additional languages (French, German, Spanish) and real-time streaming inference.

---

## References

[1] S. Poria, D. Hazarika, N. Majumder, and R. Mihalcea, "Beneath the Tip of the Iceberg: Current Challenges and New Directions in Sentiment Analysis and Emotion Detection," *IEEE Transactions on Affective Computing*, vol. 14, no. 1, pp. 108–132, 2020.

[2] P. Ekman, "An Argument for Basic Emotions," *Cognition & Emotion*, vol. 6, no. 3–4, pp. 169–200, 1992.

[3] B. Schuller, B. Vlasenko, F. Eyben, M. Wöllmer, A. Stuhlsatz, A. Wendemuth, and G. Rigoll, "Cross-corpus acoustic emotion recognition: Variances and strategies," *IEEE Transactions on Affective Computing*, vol. 1, no. 2, pp. 119–131, 2010.

[4] S. B. Davis and P. Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980.

[5] J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang, et al., "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions," in *Proc. ICASSP*, 2018, pp. 4779–4783.

[6] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020.

[7] L. Pepino, P. Riera, and L. Ferrer, "Emotion Recognition from Speech Using wav2vec 2.0 Embeddings," in *Proc. Interspeech*, 2021, pp. 3400–3404.

[8] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 3451–3460, 2021.

[9] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," *IEEE Journal of Selected Topics in Signal Processing*, vol. 16, no. 6, pp. 1505–1518, 2022.

[10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[11] J. Wagner, A. Triantafyllopoulos, H. Wierstorf, M. Schmitt, F. Burkhardt, F. Eyben, and B. W. Schuller, "Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 9, pp. 10745–10759, 2023.

[12] G. Trigeorgis, F. Ringeval, R. Brueckner, E. Marchi, M. A. Nicolaou, B. Schuller, and S. Zafeiriou, "Adieu Features? End-to-End Speech Emotion Recognition Using a Deep Convolutional Recurrent Network," in *Proc. ICASSP*, 2016, pp. 5200–5204.

[13] E. Morais, R. Hoory, W. Zhu, I. Gat, M. Damasceno, and H. Aronowitz, "Speech Emotion Recognition Using Self-Supervised Features," in *Proc. IEEE ICASSP*, 2022, pp. 6922–6926.

[14] S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)," *PLOS ONE*, vol. 13, no. 5, p. e0196391, 2018.

[15] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset," *IEEE Transactions on Affective Computing*, vol. 5, no. 4, pp. 377–390, 2014.

[16] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," in *Proc. Interspeech*, 2019, pp. 2613–2617.
