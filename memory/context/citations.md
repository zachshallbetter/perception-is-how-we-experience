# Validated Citations

All 20 citations verified and ready to add to the paper.
Generated: February 19, 2026.

---

## Core ML / Architecture (must-cite)

| # | Citation | Used for |
|---|----------|---------|
| 1 | Vaswani, A., et al. (2017). **Attention Is All You Need.** NeurIPS 30. arXiv:1706.03762 | CDP formula fix; Transformer baseline |
| 2 | Jaegle, A., et al. (2021). **Perceiver: General Perception with Iterative Attention.** ICML. arXiv:2103.03206 | C1 naming conflict; related work |
| 3 | Jaegle, A., et al. (2022). **Perceiver IO: A General Architecture for Structured Inputs & Outputs.** ICLR. arXiv:2107.14795 | C1 naming conflict; related work |
| 4 | Bahdanau, D., Cho, K., & Bengio, Y. (2015). **Neural Machine Translation by Jointly Learning to Align and Translate.** ICLR. arXiv:1409.0473 | Attention mechanisms intro |
| 5 | Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory.** Neural Computation 9(8):1735-1780. | LSTM / vanishing gradient |
| 6 | LeCun, Y., et al. (1998). **Gradient-Based Learning Applied to Document Recognition.** Proc. IEEE 86(11):2278-2324. | CNN background section (C5) |
| 7 | Bengio, Y., Simard, P., & Frasconi, P. (1994). **Learning Long-Term Dependencies with Gradient Descent is Difficult.** IEEE Trans. NN 5(2):157-166. | Vanishing gradient claim |
| 8 | Rosenblatt, F. (1958). **The Perceptron: A Probabilistic Model...** Psychological Review 65(6):386-408. | Perceptron background section (C5) |
| 9 | Sabour, S., Frosst, N., & Hinton, G. E. (2017). **Dynamic Routing Between Capsules.** NeurIPS 30. | Capsule Networks / holistic perception |

## Neuroscience / Biological Motivation

| # | Citation | Used for |
|---|----------|---------|
| 10 | Hebb, D. O. (1949). **The Organization of Behavior.** Wiley. | Hebbian learning (Reference.md) |
| 11 | Wilson, M. A., & McNaughton, B. L. (1994). **Reactivation of Hippocampal Ensemble Memories During Sleep.** Science 265(5172):676-679. | Replay mechanism / REM sleep |
| 12 | Stickgold, R. (2005). **Sleep-Dependent Memory Consolidation.** Nature 437(7063):1272-1278. | Replay mechanism / REM sleep |
| 13 | O'Neill, J., et al. (2010). **Play it Again: Reactivation of Waking Experience and Memory.** Nat. Rev. Neurosci. 11(3):209-219. | Neural replay |
| 14 | Bliss, T. V. P., & Lømo, T. (1973). **Long-lasting Potentiation of Synaptic Transmission...** J. Physiology 232(2):331-356. | LTP / Hebbian mechanism |
| 19 | DeFelipe, J., & Fariñas, I. (1992). **The Pyramidal Neuron of the Cerebral Cortex...** Prog. Neurobiol. 39(6):563-607. | Interneurons 20-30% neocortex claim |
| 20 | McCulloch, W. S., & Pitts, W. (1943). **A Logical Calculus of the Ideas Immanent in Nervous Activity.** Bull. Math. Biophys. 5:115-133. | Perceptron background section |

## Learning Paradigms

| # | Citation | Used for |
|---|----------|---------|
| 15 | Chen, T., et al. (2020). **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).** ICML. arXiv:2002.05709 | Contrastive loss in replay (M4) |
| 16 | Hadsell, R., Chopra, S., & LeCun, Y. (2006). **Dimensionality Reduction by Learning an Invariant Mapping.** CVPR 2006:1735-1742. | Original contrastive loss |

## Cognitive / Perceptual Theory

| # | Citation | Used for |
|---|----------|---------|
| 17 | Wertheimer, M. (1923). **Laws of Organization in Perceptual Forms.** Psychol. Forschung 4:301-350. | Gestalt psychology claim |
| 18 | Dehaene, S., Kerszberg, M., & Changeux, J. P. (1998). **A Neuronal Model of a Global Workspace...** PNAS 95(24):14529-14534. | Global workspace / systems neuroscience |

---

## Already in Literature/ Folder (PDFs)

| File | Citation |
|------|----------|
| `Attention Is All You Need (Vaswani et al., 2017).pdf` | Citation #1 above |
| `The Annotated Transformer.pdf` | Supplementary reading |
| `The Ultimate Guide to Transformer Deep Learning.pdf` | Supplementary reading |

---

## Suggested Experiment Baselines (for Phase 1 / C2 fix)

| Task | Dataset | Baseline to beat |
|------|---------|-----------------|
| Multimodal classification | AudioSet, MSCOCO, AV-MNIST | Perceiver IO (Jaegle et al., 2022) |
| 3D / point cloud | ModelNet40, ShapeNet | PointNet, PointNet++ |
| Continual learning / replay | Split-CIFAR, Permuted MNIST | EWC, GEM, PackNet |
| Vision + language (NLP focus) | VQA v2, Flickr30K | ViLBERT, CLIP, FLAVA |
