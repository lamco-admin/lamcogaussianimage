# Quick Start After VM Restart

**VM Requirements**: 70GB RAM, 8 CPUs

## Step 1: Verify VM Configuration

```bash
cd /home/greg/gaussian-image-projects/lgi-project

# Check RAM (should show ~70G total)
free -h

# Check CPUs (should show 8)
nproc

# Check disk space (need ~5GB free)
df -h .
```

**Expected Output**:
```
              total        used        free
Mem:           70Gi        6Gi        64Gi
```

## Step 2: Read Master Plan

```bash
cat QUANTUM_RESEARCH_MASTER_PLAN.md | less
```

Key sections:
- Phase 1: Instrument encoder (30-60 min implementation)
- Phase 2: Collect Kodak data (1-2 hours automated)
- Phase 3: Prepare dataset (5 min automated)
- Phase 4: Quantum clustering (22-37 min automated)

## Step 3: Resume AI Session

Tell the AI:
> "I've restarted the VM with 70GB RAM. I'm ready to implement Phase 1 of the quantum research plan. The master plan is in QUANTUM_RESEARCH_MASTER_PLAN.md."

The AI will guide you through:
1. Implementing Gaussian data logging in the encoder
2. Running Kodak data collection
3. Processing dataset for quantum
4. Running quantum clustering analysis

## Key Files Created

✓ **QUANTUM_RESEARCH_MASTER_PLAN.md** - Complete implementation plan
✓ **RESOURCE_REQUIREMENTS.md** - Memory analysis and VM sizing

To create:
- `collect_kodak_gaussian_data.py` - Data collection script
- `prepare_quantum_dataset.py` - Dataset preparation
- `Q1_production_real_data.py` - Quantum analysis

## Expected Timeline

- **Implementation**: 30-60 minutes (human work)
- **Data collection**: 1-2 hours (automated)
- **Quantum analysis**: 22-37 minutes (automated)
- **Total**: ~3-4 hours

## Success Indicators

✓ 24 CSV files in `quantum_research/kodak_gaussian_data/`
✓ ~480,000 Gaussian configurations collected
✓ Peak memory stays under 70GB during quantum computation
✓ 4-6 Gaussian channels discovered
✓ Silhouette score > 0.3

## Troubleshooting

**Out of memory?**
- Check `free -h` shows 70GB total
- Reduce quantum samples to 1,200 (edit script)

**Encoder crashes?**
- Test on single image first
- Check Rust compilation: `cd packages/lgi-rs && cargo build --release`

**Quantum too slow?**
- Normal: 22-37 minutes for 1,500 samples
- Acceptable: up to 60 minutes
- If >90 minutes: Something wrong

---

**Ready to proceed? Resume AI session and begin Phase 1 implementation.**
