================================================================================
QUANTUM RESEARCH PROJECT - VM RESTART INSTRUCTIONS
================================================================================

You are about to restart this VM with increased resources for quantum computing.

CURRENT VM: 22GB RAM, 8 CPUs
TARGET VM:  70GB RAM, 8 CPUs

================================================================================
DOCUMENTATION CREATED (Read These After Restart)
================================================================================

1. QUANTUM_RESEARCH_MASTER_PLAN.md (37KB)
   Location: /home/greg/gaussian-image-projects/lgi-project/
   
   Complete implementation plan covering:
   - Phase 1: Instrument encoder for data collection
   - Phase 2: Collect Gaussian data from Kodak images (1-2 hours)
   - Phase 3: Prepare dataset for quantum analysis
   - Phase 4: Run quantum clustering (22-37 minutes on 70GB RAM)
   - Phase 5: Analyze and validate results
   
   THIS IS THE MAIN DOCUMENT - Read first!

2. RESOURCE_REQUIREMENTS.md (5.7KB)
   Location: quantum_research/
   
   Technical analysis of quantum memory requirements:
   - Why 70GB RAM is needed
   - Memory calculation formulas
   - Alternative configurations
   - Optimization strategies

3. QUICK_START_AFTER_RESTART.md (2.3KB)
   Location: quantum_research/
   
   Fast reference guide for resuming work:
   - VM verification commands
   - Key file locations
   - Expected timeline
   - Troubleshooting tips

================================================================================
QUICK SUMMARY
================================================================================

Goal: Discover fundamental "Gaussian channels" using quantum computing

Data Source: Real Gaussian configurations from encoding 24 Kodak images
  - NOT synthetic random data
  - ACTUAL optimizer trajectories
  - ~480,000 configurations collected

Quantum Analysis: Kernel-based clustering in Hilbert space
  - 1,500 diverse samples
  - 64.4GB peak memory (safe with 70GB VM)
  - 22-37 minute runtime
  - Discovers 4-6 natural Gaussian channels

Why This Matters: Classical edge primitives FAILED (1.56 dB PSNR)
  - Quantum might reveal better primitive structure
  - Based on what actually works in real encoding
  - Could fundamentally improve codec quality

================================================================================
AFTER VM RESTART - DO THIS
================================================================================

1. Verify VM configuration:
   cd /home/greg/gaussian-image-projects/lgi-project
   free -h       # Should show ~70G total
   nproc         # Should show 8 CPUs

2. Read master plan:
   cat QUANTUM_RESEARCH_MASTER_PLAN.md | less

3. Resume AI session and say:
   "I've restarted the VM with 70GB RAM. Ready to implement Phase 1."

4. AI will guide you through:
   - Implementing data logging in encoder
   - Running Kodak data collection
   - Processing dataset
   - Running quantum analysis

================================================================================
EXPECTED TIMELINE
================================================================================

Implementation:      30-60 minutes (human work)
Data Collection:     1-2 hours (automated)
Dataset Prep:        5 minutes (automated)
Quantum Clustering:  22-37 minutes (automated)
Analysis:            30-60 minutes (human work)

Total: ~3-4 hours (mostly automated)

================================================================================
FILES TO BE CREATED (After Restart)
================================================================================

quantum_research/collect_kodak_gaussian_data.py
quantum_research/prepare_quantum_dataset.py
quantum_research/Q1_production_real_data.py

quantum_research/kodak_gaussian_data/*.csv (24 files)
quantum_research/kodak_gaussians_quantum_ready.pkl
quantum_research/gaussian_channels_kodak_quantum.json

================================================================================
IMPORTANT NOTES
================================================================================

✓ All plans thoroughly documented
✓ Resource requirements calculated
✓ Implementation steps defined
✓ Troubleshooting guide included
✓ Expected outputs specified

! Do NOT run quantum scripts before VM restart (will crash again)
! Kodak dataset preferred over 4K images (21× faster)
! Peak memory 64.4GB requires 70GB VM (5.6GB headroom)

================================================================================

Ready to restart VM and continue with quantum research!

