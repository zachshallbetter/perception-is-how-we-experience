# Frequently Asked Questions

## About the Project

**Q: What is Sensory Perceiver?**
A: A neural architecture for multimodal learning that uses modality-specific feature extraction, cross-modal attention, and experience replay for continual learning.

**Q: How is it different from Perceiver IO?**
A: We add explicit modality-specific towers, modality tokens, and continual learning capabilities.

**Q: What makes MIME-Type novel?**
A: Explicit one-hot modality tokens combined with modality-specific towers enable learned routing of information across modalities.

## Setup & Installation

**Q: Do I need a GPU?**
A: Yes, for experiments. For development/testing, CPU works but much slower.

**Q: What GPU do I need?**
A: 8GB minimum (GTX 1080), 16GB+ recommended (TitanX, RTX 3080).

**Q: How do I install?**
A: See `QUICK_START.md` - just run `pip install -r config/requirements.txt`.

## Running Experiments

**Q: How long do experiments take?**
A: About 18 hours on 3 GPUs in parallel. See `planning/EXECUTION_READY.md` for timeline.

**Q: Can I run on fewer GPUs?**
A: Yes, but experiments will take longer. Modify `--num_gpus` in orchestrator.

**Q: How do I just run one experiment?**
A: `python src/train.py --dataset cifar100 --num_epochs 50`

## Results & Interpretation

**Q: Where are results saved?**
A: In `results/` directory, organized by experiment date/time.

**Q: What metrics are computed?**
A: Accuracy, task-specific forgetting, forward/backward transfer, etc.

**Q: How do I visualize results?**
A: Run `python scripts/plot_results.py` after experiments complete.

## Paper & Publication

**Q: When will the paper be submitted?**
A: Target submission end of Week 4 (after M1 experiments + figure generation).

**Q: Can I see the paper?**
A: Yes, in `main-paper/Perception Is How We Experience.md`.

**Q: What conferences?**
A: Target NeurIPS 2026 or ICLR 2026 (depending on timeline).

## Development

**Q: Can I modify the architecture?**
A: Yes! The code is modular. See `src/models.py` for architecture definition.

**Q: How do I add a new modality?**
A: Add a PerceiverTower in `models.py` and data loader in `data/loaders.py`.

**Q: How do I run tests?**
A: Tests are in development. See `scripts/validate_setup.py` for validation.

## Support

**Q: Where can I get help?**
A: Check `docs/TROUBLESHOOTING.md` or the project README.

**Q: How do I report bugs?**
A: Open an issue or contact the project owner.

---

More questions? See the main project documentation or contact zachshallbetter@icloud.com.
