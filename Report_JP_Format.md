# Multilingual Speech Emotion Recognition System Using Deep Learning Techniques

**Student ID:** 66070131 | **Department:** Computer Science | **Semester:** 2 / 2024

---

## ABSTRACT

This project presents a **Multilingual Speech Emotion Recognition (Multilingual SER)** system using a **Per-Language ResNet** architecture — a Residual Convolutional Neural Network trained separately for each language. The system supports **5 languages**: Thai, Chinese, Japanese, Korean, and English, using **Mel Spectrogram** features (128 mel bands × 130 time frames) extracted at 16,000 Hz sample rate, replacing the earlier MFCC-based approach to better capture spatial energy distribution patterns.

A key insight from prior research was that a single Unified Multilingual Model suffers from **Prosody Mismatch** — each language has distinctly different pitch contours, speech rates, and energy patterns for the same emotion, causing the model to confuse language characteristics with emotion characteristics. The Per-Language approach resolves this by allowing each model to learn emotion patterns within the prosodic context of its own language.

The system employs an automatic **Language Detector** (SVM pipeline: StandardScaler → PCA(80) → SVM RBF) that identifies the spoken language from audio features before routing to the appropriate emotion model. This detector achieves **98.54% accuracy** on a 5-class language classification task.

Experimental results on held-out test sets demonstrate: Thai 88.40%, Chinese 86.62%, Japanese 84.36%, English 82.84%, and Korean 53.57% (SVM outperforms ResNet at 72.62% due to limited data). The **overall average accuracy is 79.16%**, surpassing the 75% target. The project also includes a Gradio-based Web Demo and a cross-lingual feature analysis module for studying emotion pattern similarities across languages.

---

## KEYWORDS

Speech Emotion Recognition, Multilingual, Per-Language Model, ResNet, Mel Spectrogram, Language Detector, SVM, Cross-lingual Analysis, Deep Learning, Prosody Mismatch

---

## 1 INTRODUCTION

Speech Emotion Recognition (SER) is a research field that has attracted significant attention over the past decade [1]. SER systems are applied in diverse domains: automatic call centers that detect customer dissatisfaction, mental health monitoring systems that detect depression, smart IoT devices that respond to users' emotional states, and in-car emotion detection in modern vehicles [1].

In today's world of increasing cross-linguistic and cross-cultural communication, developing a SER model that supports multiple languages in a single system (**Multilingual SER**) is highly desirable [3]. Such a system reduces development and maintenance costs, and allows a single system to serve users of diverse linguistic backgrounds without building separate models per language.

**Initial approach:** This project began by experimenting with a Multilingual Unified Model — combining English and Korean speech data, then training a single CNN + Bidirectional LSTM model to classify emotions — under the hypothesis that the model would learn language-independent emotion features from diverse data.

However, during development it was discovered that this hypothesis has an important limitation: speech in different languages has significantly different Prosody structures (pitch contour, rhythm, speech rate), which prevented the project from meeting its original target. This observation led to the analysis of **Prosody Mismatch** as the root cause and motivated the Per-Language ResNet architecture as the solution.

### 1.1 Objectives

1. Develop a Multilingual SER system using a **Per-Language ResNet** architecture, training separate models for each language.
2. Build an automatic **Language Detector** that accurately identifies the spoken language to route audio to the appropriate emotion model.
3. Study and analyze **Prosody Mismatch** across languages through Cross-lingual Feature Analysis.
4. Evaluate model performance using Accuracy, F1-Score, Confusion Matrix, and Classification Report.
5. Develop a **Web Demo** with Gradio for real-time system testing.

### 1.2 Scope

**Supported Languages (5 languages):**

| Language | Dataset | Total Samples | Emotion Classes |
|---|---|---|---|
| Thai | Thai Speech Emotion Dataset (TDED) | ~30,210 | 4 |
| Chinese | Chinese Emotional Speech Corpus | ~2,690 | 6 |
| Japanese | JANON Japanese Emotion Dataset | ~1,215 | 6 |
| Korean | Korean Voice Emotion Dataset (hi_kia) | ~420 | 5 |
| English | RAVDESS + CREMA-D | ~11,425 | 7 |

**Emotion classes supported per language:**

| Emotion | Thai | Chinese | Japanese | Korean | English |
|---|---|---|---|---|---|
| Angry | ✓ | ✓ | ✓ | ✓ | ✓ |
| Happy | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sad | ✓ | ✓ | ✓ | ✓ | ✓ |
| Neutral | ✓ | ✓ | — | ✓ | ✓ |
| Surprise | — | ✓ | ✓ | ✓ | ✓ |
| Fear | — | ✓ | ✓ | — | ✓ |
| Disgust | — | — | ✓ | — | ✓ |

**Hardware & Framework:**
- GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- Framework: PyTorch + torchaudio 2.6.0
- Audio Processing: torchaudio, librosa
- Sample Rate: 16,000 Hz | Duration: 3 seconds per file

---

## 2 RELATED WORK

### 2.1 Speech Emotion Recognition Background

Ekman [2] proposed that humans share six universal basic emotions across cultures: Angry, Happy, Sad, Fear, Disgust, and Surprise. Subsequent research found that vocal expression of emotion varies more across cultures than facial expression, particularly in terms of Prosody.

The evolution of SER approaches over time:

| Era | Approach | Strengths | Limitations |
|---|---|---|---|
| 1990s–2000s | Hand-crafted Features + HMM/SVM | Interpretable, lightweight | Manual feature design, low accuracy |
| 2010–2015 | Deep Neural Networks (CNN, RNN) [12] | Higher accuracy | Requires large data |
| 2015–present | CNN + LSTM / Transformer [11] | State-of-the-art | Resource-intensive, Multilingual challenge |

### 2.2 Challenges in Multilingual SER

Schuller et al. [3] identified cross-corpus and cross-lingual generalization as the two primary challenges in SER:
- **Cross-corpus:** Accuracy typically drops 15–30% when testing on a different dataset than training.
- **Cross-lingual:** Accuracy typically drops 20–40% when testing across languages due to Prosody Mismatch.

More recent work [1] further identifies that the lack of large-scale multilingual annotated corpora and the subjective nature of emotion labeling remain open problems even as model architectures improve.

### 2.3 Feature Extraction Methods

**MFCC (Mel-Frequency Cepstral Coefficients)** — the standard feature since the 1980s [4]. It captures the spectral envelope of speech by applying Mel-scale filterbanks followed by a DCT. While effective for single-language SER, MFCC conflates language-specific and emotion-specific spectral patterns, making it less suitable for cross-lingual settings.

**Mel Spectrogram** — represents the distribution of energy across time and frequency on the Mel scale [5], retaining 2D spatial structure that MFCC discards. This allows CNN architectures to directly exploit time-frequency patterns without temporal pooling losses. Mel Spectrogram has become the dominant input representation for deep SER models [11].

**Wav2Vec 2.0** [6] — a self-supervised pre-trained model from Facebook AI that learns contextual speech representations via contrastive learning over masked frames. Its deep transformer architecture captures language-neutral phonetic features, making it a strong candidate for cross-lingual transfer. Pepino et al. [7] demonstrated that fine-tuning wav2vec 2.0 embeddings for SER achieves competitive results even with limited labeled data. Morais et al. [13] further confirmed that self-supervised features significantly outperform MFCC on cross-corpus SER benchmarks.

**HuBERT** [8] — extends the self-supervised learning paradigm by predicting discrete hidden unit targets via offline clustering, achieving state-of-the-art on a range of speech tasks. Its offline cluster-based target generation makes it more stable than contrastive approaches during fine-tuning.

**WavLM** [9] — combines masked speech prediction with a denoising objective, producing representations that generalize across multiple speech processing tasks including emotion recognition, speaker verification, and ASR. WavLM Large achieves the top rank on the SUPERB benchmark as of 2022.

### 2.4 Deep Learning Architectures

**ResNet (Residual Network)** [10] uses skip connections to overcome vanishing gradients in deep CNNs. Applied to 2D Mel Spectrograms, ResNet treats emotion recognition as an image classification problem over time-frequency maps. The skip connection is defined as:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

where $\mathcal{F}$ denotes the residual mapping to be learned.

**CNN + Bidirectional LSTM** [12] captures both local spectral patterns (CNN) and long-range temporal dependencies (Bi-LSTM). The bidirectional design reads sequences in both forward and backward directions:

$$h_t = [\overrightarrow{\text{LSTM}}(x_t) \; ; \; \overleftarrow{\text{LSTM}}(x_t)]$$

**Transformer-based architectures** [11] have recently surpassed CNN+LSTM models on standard SER benchmarks, particularly for valence prediction. Wagner et al. [11] demonstrated that fine-tuned transformer encoders close the long-standing valence gap in dimensional SER, achieving state-of-the-art across multiple corpora.

**SVM (Support Vector Machine)** with an RBF kernel is a strong baseline for lower-resource settings. When training data is limited (e.g., Korean), SVM trained on Mel Spectrogram statistics often outperforms deep models due to its better regularization on small datasets [3].

### 2.5 Datasets

| Dataset | Language | Reference |
|---|---|---|
| RAVDESS | English | Livingstone & Russo [14] |
| CREMA-D | English | Cao et al. [15] |
| Korean Voice Emotion (hi_kia) | Korean | Hugging Face Datasets |
| JANON | Japanese | — |
| Chinese Emotional Speech Corpus | Chinese | — |
| TDED | Thai | — |

---

## 3 DATASET DESCRIPTION

### 3.1 Dataset Overview

This project uses six public datasets covering five languages. Each dataset was collected under controlled recording conditions with professional or crowdsourced actors and validated emotion labels.

**English — RAVDESS + CREMA-D:**
- RAVDESS [14]: 24 professional actors (12 male, 12 female), recorded in an anechoic chamber, WAV mono 48 kHz. Emotion labels are encoded in the filename (e.g., `03-01-05-01-01-01-12.wav` → Angry).
- CREMA-D [15]: 7,442 clips from 91 actors, diverse age/ethnicity, covering 6 emotion classes. Labels were validated through crowdsourced majority voting.
- Combined English samples: ~11,425.

**Thai — TDED (Thai Speech Emotion Dataset):**
- ~30,210 samples covering 4 emotion classes: Angry, Happy, Sad, Neutral.
- Largest dataset in the collection; provides the most robust training signal.

**Chinese — Chinese Emotional Speech Corpus:**
- ~2,690 samples, 6 emotion classes.
- Balanced distribution across classes.

**Japanese — JANON:**
- ~1,215 samples, 6 emotion classes including Disgust.
- Smallest balanced dataset.

**Korean — Korean Voice Emotion Dataset (hi_kia):**
- ~420 samples streamed from Hugging Face, 5 emotion classes.
- Most limited dataset; drives the decision to use SVM for Korean.

### 3.2 Data Distribution and Imbalance

The severe size imbalance across languages (Thai: 30,210 vs. Korean: 420) is a defining challenge of this project. To address this:
- Per-Language Models train each language on its own data, so cross-language imbalance does not affect emotion classification.
- The Language Detector uses balanced sampling across languages during training.
- Data augmentation is applied during training of each Per-Language model.

### 3.3 Preprocessing Pipeline

All audio files go through a uniform preprocessing pipeline before feature extraction:

1. Load at 16,000 Hz (resample if needed)
2. Trim leading/trailing silence (`librosa.effects.trim`, top_db=25)
3. Pad or cut to exactly 3 seconds (48,000 samples)

### 3.4 Feature Extraction

**Mel Spectrogram** [5] (used for Per-Language ResNet and Language Detector):
- n_fft = 1024, hop_length = 512, n_mels = 128, fmin = 0, fmax = 8000
- Output tensor shape: (1, 128, 130) — 1 channel, 128 mel bands, 130 time frames

**MFCC** [4] (used for the initial Unified Model baseline):
- n_mfcc = 40, output shape: (T, 40)

---

## 4 METHODOLOGY

### 4.1 System Architecture Overview

The system operates in two phases:

**Training Phase:** For each language, a separate EmotionResNet model is trained on that language's dataset using Mel Spectrogram features [5]. In parallel, a Language Detector SVM is trained on Mel Spectrogram features from all five languages combined.

**Inference Phase:**
1. Input audio → preprocessing (trim, pad/cut to 3 s) → Mel Spectrogram extraction
2. Language Detector (SVM) → identify language (Thai / Chinese / Japanese / Korean / English)
3. Route to corresponding EmotionResNet → predict emotion class
4. Return: (predicted language, predicted emotion, confidence scores)

### 4.2 Language Detector

The Language Detector is an SVM pipeline trained to classify which of the 5 languages a given audio clip belongs to.

**Architecture:**
```
Mel Spectrogram (1×128×130)
    → Flatten (16,640-dim vector)
    → StandardScaler (zero mean, unit variance)
    → PCA (reduce to 80 principal components)
    → SVM (RBF kernel, C=10, gamma='scale')
    → Language Label (5 classes)
```

**Training:** Stratified sampling ensures balanced representation of all 5 languages. The scaler and PCA are fit on training data only to prevent data leakage.

**Result:** 98.54% accuracy on the held-out language test set (5-class classification).

### 4.3 Per-Language EmotionResNet

Each language uses a ResNet-inspired [10] CNN architecture operating on 2D Mel Spectrograms [5].

**Architecture:**

```
Input: (Batch, 1, 128, 130)
├── Stem: Conv2d(1→32, 3×3) → BN → ReLU → MaxPool
├── Layer 1: ResBlock(32→64) × 2
├── Layer 2: ResBlock(64→128, stride=2) × 2
├── Layer 3: ResBlock(128→256, stride=2) × 2
├── Global Average Pooling → (Batch, 256)
├── Dropout(0.5)
└── Linear(256 → num_classes)
```

Each ResBlock follows the standard residual connection pattern [10]:
```
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
```

**Training Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Better generalization than Adam via weight decay |
| Learning Rate | 0.001 | Standard starting point for ResNet |
| Weight Decay | 1e-4 | L2 regularization |
| Batch Size | 32 | GPU memory vs. gradient stability trade-off |
| Max Epochs | 100 | With early stopping |
| Early Stopping | patience=15 (val_accuracy) | Prevent overfitting |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) | Adaptive learning rate |

**Data Augmentation** (applied to training set only):

| Technique | Parameter | Purpose |
|---|---|---|
| SpecAugment [16] — Time Mask | mask up to 20 time frames | Simulate missing temporal segments |
| SpecAugment [16] — Frequency Mask | mask up to 20 frequency bins | Simulate missing frequency bands |
| Audio Time Stretch | rate ∈ [0.85, 1.15] | Simulate speech rate variation |
| Audio Pitch Shift | ±2 semitones | Simulate pitch variation |
| Gaussian Noise | σ = 0.005 | Improve noise robustness |

### 4.4 Anti-Data-Leakage Split

To prevent data leakage (a critical concern when the same speaker appears across multiple samples), the dataset is split at the **speaker level** before any preprocessing or augmentation:

- Train: 70% of speakers
- Validation: 15% of speakers
- Test: 15% of speakers (held out until final evaluation)

StandardScaler and PCA are fit exclusively on the training fold and then applied to validation and test folds.

### 4.5 Korean Language Exception — SVM Classifier

Due to the Korean dataset's small size (~420 samples), ResNet overfits severely. An SVM classifier is used instead, consistent with findings in [3] that classical models retain competitive performance in low-resource regimes:

```
Mel Spectrogram → Flatten → StandardScaler → PCA(30) → SVM(RBF)
```

Korean SVM achieves 72.62% accuracy, outperforming the Korean ResNet (53.57%).

### 4.6 Cross-lingual Analysis Module

After training all models, a cross-lingual analysis extracts per-emotion feature centroids (mean Mel Spectrogram vectors) for each language and computes cosine similarity across language pairs. This produces a language × language similarity matrix per emotion class, enabling quantitative analysis of Prosody Mismatch — a phenomenon documented in cross-corpus SER literature [3][1].

### 4.7 Web Demo

A Gradio-based web interface allows real-time inference:
1. User uploads an audio file (WAV/MP3/FLAC).
2. System auto-detects language and predicts emotion.
3. Output displays: detected language, predicted emotion, and per-class confidence bar chart.

---

## 5 EXPERIMENTAL RESULTS AND DISCUSSION

### 5.1 Language Detector Performance

| Metric | Value |
|---|---|
| Accuracy | **98.54%** |
| Macro F1-Score | 0.985 |

The Language Detector performs nearly perfectly, enabling reliable routing to Per-Language emotion models. Occasional misclassification between Chinese and Japanese accounts for most errors, which is expected given acoustic similarity between the two languages.

### 5.2 Per-Language Emotion Recognition Results

| Language | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| Thai | ResNet [10] | **88.40%** | 0.881 |
| Chinese | ResNet [10] | **86.62%** | 0.863 |
| Japanese | ResNet [10] | **84.36%** | 0.840 |
| English | ResNet [10] | **82.84%** | 0.825 |
| Korean | SVM | **72.62%** | 0.718 |
| Korean | ResNet [10] | 53.57% | 0.521 |
| **Overall Average** | Mixed | **79.16%** | — |

The overall average of **79.16%** surpasses the 75% target. These results are consistent with recent findings [7][13] showing that language-specific or self-supervised feature approaches significantly outperform unified multilingual baselines.

### 5.3 Comparison: Unified Model vs. Per-Language Model

| Metric | Unified CNN+BiLSTM [12] | Per-Language ResNet [10] |
|---|---|---|
| English Accuracy | ~65% | 82.84% |
| Korean Accuracy | ~42% | 72.62% (SVM) |
| Overall Average | ~55% | **79.16%** |

The Per-Language approach provides a +24 percentage-point improvement on average over the Unified Model, confirming that Prosody Mismatch [3] is the primary bottleneck for unified multilingual architectures.

### 5.4 Analysis of Korean Performance

Korean achieves the lowest accuracy (72.62% with SVM vs. 53.57% with ResNet) due to:
1. **Dataset size:** 420 samples — an order of magnitude smaller than other languages.
2. **Deep model data requirement:** ResNet [10] requires substantially more data to generalize.
3. **SVM advantage on small data:** SVM with RBF kernel achieves better regularization on limited samples, consistent with observations in [3].

### 5.5 Confusion Matrix Analysis

The most frequent misclassifications across all languages follow the same pattern:
- **Happy ↔ Surprise:** Both emotions share high energy and elevated pitch contours.
- **Sad ↔ Neutral:** Both emotions feature low energy and monotone pitch.
- **Angry ↔ Happy (Korean only):** Due to limited training data, the model struggles with high-energy positive vs. high-energy negative emotions.

This emotion confusion pattern is consistent with valence-based confusion reported in large-scale SER studies [11].

### 5.6 Cross-lingual Feature Analysis

Cosine similarity scores between language emotion centroids reveal:
- **English–Chinese similarity:** 0.72 (highest cross-lingual pair) — both share similar pitch dynamics for Anger.
- **Thai–Japanese similarity:** 0.61 — moderate similarity in tonal prosody patterns.
- **Korean–all others:** 0.45–0.52 (lowest) — Korean's syllable-timed rhythm is distinctly different.

This quantitatively confirms that Prosody Mismatch [3] is most severe for Korean, explaining its lower emotion recognition accuracy. Similar cross-lingual feature divergence has been reported in self-supervised speech representation studies [6][8].

---

## 6 CONCLUSION

This project developed a **Multilingual Speech Emotion Recognition** system that successfully supports 5 languages using a **Per-Language ResNet** [10] architecture combined with an automatic **Language Detector**. The key contributions are:

1. **Per-Language ResNet architecture** that resolves Prosody Mismatch [3] by training separate models per language, achieving an overall average accuracy of **79.16%** — exceeding the 75% project target.
2. **Language Detector** (SVM pipeline) with **98.54% accuracy** on 5-class language identification, enabling fully automatic routing without manual language specification.
3. **Empirical validation** that Prosody Mismatch is the root cause of Unified Multilingual Model failure, supported by cross-lingual cosine similarity analysis, consistent with challenges documented in [1][3].
4. **Korean SVM exception** demonstrating that for low-resource languages, classical ML outperforms deep learning, and that architecture selection must consider dataset size.
5. **Gradio Web Demo** enabling real-time end-to-end inference for demonstration and user testing.

The system demonstrates that language-aware modular design is a practical and effective strategy for multilingual SER, yielding substantially better results (+24 percentage points) than naive data pooling.

---

## 7 FUTURE WORK

### 7.1 Korean Data Expansion

The most critical improvement is collecting more Korean speech emotion data. With 2,000+ samples, a ResNet model [10] is expected to match or exceed the accuracy of other languages.

**Recommended sources:**
- KEMDy19 / KEMDy20 (Korean Emotion Dataset, released by ETRI)
- AIHub Korean emotional speech corpus (ai-hub.or.kr)

### 7.2 Self-Supervised Speech Foundation Models

Replace or augment Mel Spectrogram features with embeddings from large pre-trained speech models. Three strong candidates are:
- **wav2vec 2.0** [6] (`facebook/wav2vec2-large-xlsr-53`) — cross-lingual pre-training on 53 languages. Pepino et al. [7] showed competitive SER results via fine-tuning.
- **HuBERT** [8] — masked prediction of cluster assignments; stable fine-tuning and strong on limited labeled data.
- **WavLM** [9] — denoising-augmented pre-training; ranked first on SUPERB benchmark across multiple speech tasks including emotion.

Self-supervised features are expected to capture more language-neutral phonetic representations, potentially reducing Prosody Mismatch at the feature level even within the Per-Language framework [13].

### 7.3 Language-Agnostic Emotion Features

Investigate adversarial training or domain adaptation techniques to disentangle emotion features from language-specific features within a shared encoder [1]. This could enable a true Unified Multilingual Model without Prosody Mismatch degradation. Morais et al. [13] provide evidence that self-supervised representations already reduce cross-domain feature shift, suggesting this direction is viable.

### 7.4 Additional Languages

Extend the system to support additional languages such as:
- **French** (EMOFILM, GEMEP corpus)
- **German** (EmoDB)
- **Mandarin / Cantonese** expansion
- **Spanish** (IEMOCAP-ES)

### 7.5 Real-time Streaming Inference

The current system processes fixed 3-second clips. Future work should implement sliding-window inference to enable continuous real-time emotion monitoring from live audio streams.

### 7.6 Multimodal Fusion

Combine speech-based SER with facial expression recognition and text sentiment analysis [1] to build a more robust multimodal emotion understanding system, particularly for edge cases where a single modality is ambiguous. Transformer-based multimodal fusion [11] has shown strong results on audio-visual emotion benchmarks.

---

## REFERENCES

[1] Poria, S., Hazarika, D., Majumder, N., & Mihalcea, R. (2020). Beneath the Tip of the Iceberg: Current Challenges and New Directions in Sentiment Analysis and Emotion Detection. *IEEE Transactions on Affective Computing*, 14(1), 108–132.

[2] Ekman, P. (1992). An Argument for Basic Emotions. *Cognition & Emotion*, 6(3–4), 169–200.

[3] Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., & Rigoll, G. (2010). Cross-corpus acoustic emotion recognition: Variances and strategies. *IEEE Transactions on Affective Computing*, 1(2), 119–131.

[4] Davis, S. B., & Mermelstein, P. (1980). Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366.

[5] Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., ... & Saurous, R. A. (2018). Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions. *Proceedings of ICASSP*, 4779–4783.

[6] Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.

[7] Pepino, L., Riera, P., & Ferrer, L. (2021). Emotion Recognition from Speech Using wav2vec 2.0 Embeddings. *Proceedings of Interspeech 2021*, 3400–3404.

[8] Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29, 3451–3460.

[9] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Wu, M., Zeng, M., Yu, X., & Wei, F. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 1505–1518.

[10] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

[11] Wagner, J., Triantafyllopoulos, A., Wierstorf, H., Schmitt, M., Burkhardt, F., Eyben, F., & Schuller, B. W. (2023). Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(9), 10745–10759.

[12] Trigeorgis, G., Ringeval, F., Brueckner, R., Marchi, E., Nicolaou, M. A., Schuller, B., & Zafeiriou, S. (2016). Adieu Features? End-to-End Speech Emotion Recognition Using a Deep Convolutional Recurrent Network. *Proceedings of ICASSP*, 5200–5204.

[13] Morais, E., Hoory, R., Zhu, W., Gat, I., Damasceno, M., & Aronowitz, H. (2022). Speech Emotion Recognition Using Self-Supervised Features. *Proceedings of IEEE ICASSP 2022*, 6922–6926.

[14] Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391.

[15] Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390.

[16] Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Proceedings of Interspeech*, 2613–2617.
