# Sensory Perceiver Baselines

To rigorously evaluate the Sensory Perceiver architecture, we provide a suite of implementations for state-of-the-art vision and multimodal baselines. All baselines are located in `baselines/` and leverage the Hugging Face `transformers` library for pre-trained weights.

## Integrated Baselines

### 1. Perceiver (Jaegle et al., 2021)
- **Path:** `baselines/perceiver.py`
- **Description:** The original DeepMind Perceiver architecture, defining the core concept of cross-attending inputs to a smaller set of latent vectors.
- **Comparison Focus:** Validates if Sensory Perceiver's MIME-Type routing is superior to generic positional embeddings.

### 2. Perceiver IO (Jaegle et al., 2022)
- **Path:** `baselines/hf_perceiver_io.py` (HuggingFace Wrap) / `baselines/perceiver_io/` (Custom)
- **Description:** Extending the original Perceiver with an output query array that enables structured output mappings (e.g., dense vision tasks, language modeling).
- **Comparison Focus:** Serves as the primary compute and efficiency benchmark against Sensory Perceiver.

### 3. ViLT (Kim et al., 2021)
- **Path:** `baselines/vilt.py`
- **Description:** Vision-and-Language Transformer. A highly efficient architecture that drops heavy visual embedders (like ResNets) and feeds image patches directly into an attention mechanism alongside text tokens.
- **Comparison Focus:** Benchmarks the Sensory Perceiver against a highly specialized and lightweight dual-modality (Vision+Text) network.

### 4. CLIP (Radford et al., 2021)
- **Path:** `baselines/clip.py`
- **Description:** Contrastive Language-Image Pretraining from OpenAI. The industry standard dual-encoder architecture for multimodal text-image zero-shot capability.
- **Comparison Focus:** Tests if the Sensory Perceiver's fused representation outperforms CLIP's late-fusion dot-product similarity metrics.

### 5. FLAVA (Singh et al., 2022)
- **Path:** `baselines/flava.py`
- **Description:** Foundational Language And Vision Alignment from Meta. A single model trained on unimodal vision data, unimodal language data, and multimodal paired data.
- **Comparison Focus:** Evaluates whether Sensory Perceiver's modular `PerceiverTower` components scale better natively across >2 modalities compared to FLAVA's heavy early/late fusion Transformer stack.

---

## Instructions for Use

To train or evaluate any of these baselines on the core benchmarks, ensure the `transformers` library is installed:

```bash
pip install transformers torch
```

To run a baseline continuously across the orchestration test suite:

```bash
python scripts/run_baselines_cl.py --model vilt --output_dir results/baselines_cl
```

*(Note: Ensure your `experiment_config.yaml` points `--model` to the appropriate baseline identifier).*
