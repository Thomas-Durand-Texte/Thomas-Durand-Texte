<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner_dark.png">
  <img alt="Thomas Durand-Texte — Senior Applied AI Scientist · Acoustics · Signal Processing · Computer Vision" src="assets/banner_light.png" width="100%">
</picture>

🎓 PhD in Acoustics · MSc Electro-Acoustics · Engineer (ENSIM) · ML Engineer (CentraleSupélec/OC)
🔬 10+ years in R&D: vibrations, 3D vision, signal processing, deep learning
🏢 Currently: R&D Engineer @ iAudiogram · Previously: LAUM, ACOEM, Valéo, Harman

## 🔥 Highlights

| Project                                                                                            | Impact                                          |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------|
| **[DL for Digital Image Correlation](https://github.com/Thomas-Durand-Texte/DIC-Neural-Networks)** | ×420 speedup, -82% error, -83% resolution limit |
| **[Efficient audio tagging](https://github.com/Thomas-Durand-Texte/audio-tagging-portfolio)**      | 200-label tagger at 5.06M params, no AudioSet   |
| **Speech-in-noise audiometry (ML)**                                                                | ~50% exam time reduction                        |
| **Audiometric calibration automation**                                                             | -65% total time, -95% human time                |
| **[Thermo-mechanical modelling of an ABH](https://github.com/Thomas-Durand-Texte/thermo-mechanical-model)** | Where damping happens, not just how much |
| **Vibration modeling**                                                                             | FEM-equivalent model at -99.5% compute          |

## 🔊 Featured: Efficient Multi-Label Audio Tagging

A learned, **physically-grounded DSP front-end** (ERB SuperGaussian filter bank + adaptive PCEN) feeding a **shared-encoder multi-head model** (peak-pool / modulation-spectrum / transformer-SED, per-class gated fusion). Trained **from scratch on FSD50K — no AudioSet pretraining**.

- **[audio-tagging-portfolio](https://github.com/Thomas-Durand-Texte/audio-tagging-portfolio)** — the write-up: the front-end story, an STFT-vs-analytical-filter-bank stem comparison, and an honest per-class analysis of two model lines
- **[apcen-multihead-tagger](https://github.com/Thomas-Durand-Texte/apcen-multihead-tagger)** — the code + pretrained weights: infer / evaluate / fine-tune / predict, verified **bit-identical** to the research model

## 🌡️ Featured: Where Vibration Energy Is Lost, from a Thermal Video

A structure vibrates, some energy is dissipated, that energy becomes heat, and an infrared camera sees it. **[thermo-mechanical-model](https://github.com/Thomas-Durand-Texte/thermo-mechanical-model)** models every step of that chain — and then runs it backwards, recovering how much power a structure dissipates *and where* from thermal imaging alone.

Damping is normally characterised globally, as a modal loss factor, which says nothing about location. Four coupled models close that gap: a three-layer beam of varying thickness solved by impedance transport, the dissipated power from closed-form through-thickness integrals, one-dimensional heat transfer on control volumes, and the heat equation read as a residual to invert the chain.

- Validated against an **independent finite-element solution**, **laboratory vibrometry and infrared measurements**, and a dozen **closed-form results** that have truth values
- Applied to **Acoustic Black Holes**, and reproducing from geometry alone what the published thermal-imaging work identifies from measurement
- Includes the **negative results** — a hypothesis that predicted the right sign and still failed, a per-cell estimate that does not work, and a measurement that cannot identify what it was hoped to

## 📚 Training Portfolio

**[Machine Learning Engineer Projects](https://github.com/Thomas-Durand-Texte/OC-formation-Ingenieur-Machine-Learning)** — Portfolio of ML projects from OpenClassrooms/CentraleSupélec training program

## 🛠️ Tech Stack

**Languages & Frameworks:** `Python` `PyTorch` `scikit-learn` `NumPy` `pandas` `SQL` `C/C++` `React` `LaTeX` `Matlab` `Simulink`

**Tools & Platforms:** `Docker` `Git` `Linux` `macOS` `Blender`
💻 10+ years Linux & macOS experience

## 📄 Selected Publications

- *Single-camera vision method for vibration measurement* — J. Sound Vib. (2020)
- *Thermal imaging of acoustic black hole damping* — J. Appl. Phys. (2020)
- *Thermal imaging of vibrational energy dissipated in a 2D acoustic black hole pit* — Appl. Phys. Lett. (2021)
- *Comparison of full-field optical techniques* — Nature Sci. Rep. (2023)

## 📫 Contact

✉️ thomas.durandtexte@protonmail.com · 📍 France · 🌐 Open to remote & hybrid roles
