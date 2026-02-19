# Troubleshooting Guide

## Common Issues & Solutions

### Setup Issues

**Issue**: CUDA not found
```
RuntimeError: No CUDA-capable device detected
```
**Solution**:
- Verify GPU: `nvidia-smi`
- Check CUDA version: `nvcc --version`
- Reinstall PyTorch with correct CUDA version from pytorch.org

**Issue**: Out of memory during training
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch size: `--batch_size 16`
- Reduce model size: `--hidden_dim 256`
- Enable gradient checkpointing (see `src/models.py`)

### Experiment Issues

**Issue**: Import errors after directory restructure
```
ModuleNotFoundError: No module named 'src.models'
```
**Solution**:
- Ensure `src/` is in Python path
- Add to your script: `import sys; sys.path.insert(0, '.')`
- Or run from repo root: `python -m src.train`

**Issue**: Experiments not finding data
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/cifar100'
```
**Solution**:
- Run from repo root directory
- Data will auto-download on first run
- Check internet connection for downloads

### Results Issues

**Issue**: Results directory not created
```
FileNotFoundError: results/ does not exist
```
**Solution**:
```bash
mkdir -p results
python src/train.py --results_dir results/
```

**Issue**: Plots not generating
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution**:
```bash
pip install matplotlib seaborn
python scripts/plot_results.py
```

## Performance Tuning

### GPU Utilization

If GPU utilization is low (<50%):
- Increase batch size (if memory allows)
- Reduce number of workers in data loader: `--num_workers 4`
- Reduce model complexity temporarily for profiling

### Training Speed

If training is slow:
- Profile with: `python -m torch.utils.bottleneck src/train.py ...`
- Common bottleneck: Data loading → increase workers
- Consider mixed precision: `--mixed_precision`

## Getting Help

1. **Check logs**: `tail -f results/training.log`
2. **Review error message**: Usually points to exact issue
3. **See FAQ**: `docs/FAQ.md`
4. **Check planning docs**: `planning/EXECUTION_READY.md`

---

For more help, see the main README or contact the project owner.
