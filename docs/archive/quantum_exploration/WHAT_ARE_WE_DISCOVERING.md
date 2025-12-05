# What Are We Discovering? - Quantum Primitive Research Explained

**Status:** Active quantum experiments running
**Key Question:** What image primitives does quantum mechanics reveal?

---

## The Core Concept: Patches

### What Is a Patch?

**A patch = small square extracted from an image (typically 16×16 pixels = 256 pixels)**

**Visual example:**

```
Full Kodak image (768×512):
┌────────────────────────────────┐
│                                │
│    [16×16 patch] ←extracted    │
│         │                      │
│         ↓                      │
│    ▓▓▓▓▓▓▓▓                    │
│    ▓▓▓▓▓▓▓▓  (256 pixels)      │
│                                │
└────────────────────────────────┘
```

### Why Patches?

**Problem:** Full image (393K pixels) has mixed content:
- Sky (smooth)
- Building edges (sharp boundaries)
- Texture (brick, grass)
- Shadows (gradients)

**Solution:** Patch is small enough to have ONE dominant characteristic.

**Example patches from Kodak kodim01 (woman):

1. **Patch from sky:** Mostly smooth blue gradient
2. **Patch from hair edge:** Sharp boundary (hair vs skin)
3. **Patch from fabric:** Texture pattern
4. **Patch from face:** Smooth with gentle gradients

**Each patch ≈ one "feature type" we want to understand.**

---

## What Makes a "Representative" Patch?

**Representative = clearly exhibits a specific characteristic**

### Non-representative (ambiguous):
```
Patch at corner of image: half edge, half smooth
→ Mixed content, unclear what it "is"
```

### Representative edge patch:
```
Clear boundary cutting through patch
High gradient in one direction
Characteristic we want to model
```

### For Quantum Validation (50 patches from free tier):

**Representative set = covers diversity:**
- ~10 patches from each quantum cluster
- Select from cluster centers (typical examples)
- Include 1-2 boundary cases per cluster
- **Total: 50-60 patches testing all groups quantum found**

---

## The Current Question (What's Running Now)

###  "What are the NATURAL groupings of image content?"**

**We're NOT asking:** "Classify as edge/region/texture" (pre-defined categories)

**We ARE asking:** "Let quantum clustering tell us what groups exist naturally"

### The Process:

**Step 1: Feature Extraction (Done)**
```
For each of 1,000-5,000 patches:
  Compute 10 features:
    1. Gradient magnitude (intensity change)
    2. Gradient std dev (variation in change)
    3-4. Structure tensor eigenvalues (directional analysis)
    5. Anisotropy ratio (directional vs isotropic)
    6. Mean intensity (brightness)
    7. Std dev (local variation)
    8. Variance (spread)
    9. Intensity range (min to max)
    10. Edge strength (Laplacian)

→ Each patch = point in 10-dimensional feature space
```

**Step 2: Quantum Encoding**
```
10 features → PCA → 8 features
8 features → amplitude encoding → 8 qubits

Quantum state: |ψ⟩ = ∑_i α_i |i⟩
where α_i are complex amplitudes encoding features

This creates 2^8 = 256-dimensional complex Hilbert space
(NOT Euclidean - has quantum interference, phase structure)
```

**Step 3: Quantum Kernel Computation**
```
For all pairs of patches (i, j):
  Compute quantum similarity: K(i,j) = |⟨ψ_i|ψ_j⟩|²

This measures overlap in quantum Hilbert space
Quantum interference affects similarity
→ Different from Euclidean distance!

Result: Quantum kernel matrix (1000×1000)
```

**Step 4: Spectral Clustering**
```
Find groups in quantum kernel space
Uses eigenvectors of kernel matrix
Groups emerge from quantum mechanical structure

→ Quantum-discovered clusters
```

**Step 5: Analyze Clusters**
```
For each cluster quantum found:
  What feature values characterize it?
  - Mean gradient, anisotropy, variance, etc.

Name cluster based on characteristics:
  - Not "edge" or "region" (pre-conceived)
  - But "High anisotropy, low variance" (data-driven)
```

---

## Why This Matters (Connection to Gaussian Problem)

### Classical Approach Failed:

**Phase 0-1 results:**
- "Edge primitive" = elongated Gaussians for edges
- **Result:** 1.56 dB PSNR (catastrophic failure)
- Compositional approach failed (11 dB vs 25 dB target)

**Why it failed:**
- "Edge" might not be the right primitive category
- Elongated Gaussians might not be the right representation
- **Human-designed primitives might be wrong**

### Quantum Might Reveal:

**Instead of "edges":** Maybe quantum finds:
- Cluster A: "Low variance, high anisotropy" (smooth directional)
- Cluster B: "High variance, high anisotropy" (textured boundaries)
- **These might need DIFFERENT Gaussian representations!**

**Instead of M/E/J/R/B/T:** Maybe quantum finds:
- 4 clusters based on anisotropy levels (continuous, not discrete types)
- OR 8 clusters based on variance×gradient combinations
- **Data says what primitives ARE, not our intuition**

---

## Other Questions We Could Ask

### Question 2: "What Gaussian parameters for quantum-discovered primitives?"

**After quantum reveals clusters:**

For Cluster 0 (whatever it is):
- Run Phase 0-style experiments
- Find optimal Gaussian config for THIS specific cluster
- Might be isotropic, might be elongated, might be something else

**For all clusters:**
- Each gets its own empirical rules
- Based on what works, not what we designed

---

### Question 3: "Can quantum learn placement better?"

**Quantum annealing (QUBO):**
- Given image and Gaussian budget
- Quantum finds optimal discrete placement
- Compare vs gradient-based placement

**Free tier compatible:**
- Small image (64×64 grid = 4,096 locations, select 50)
- ~5-8 minutes
- **Could test this with free tier!**

---

### Question 4: "Does quantum find better formulas?"

**Quantum regression:**
- Learn f(features) → Gaussian_parameters
- QNN might find non-obvious relationships
- Classical regression for comparison

**Cost:**
- Expensive to train on real quantum ($2,000+)
- **Better on simulator** (free), then extract formula

---

### Question 5: "What if we let quantum design the REPRESENTATION?"

**Deep question:**
- Not just "what primitives?"
- But "should we use Gaussians at all?"

**Quantum might reveal:**
- Image structure naturally lives in quantum space
- Optimal representation might be quantum states (deploy classically as lookup)
- **Completely different approach than Gaussians**

**This is speculative but your Phase 1 failure (1.56 dB) suggests Gaussians might be wrong primitive entirely.**

---

## What Results Will Tell Us

### When 1,000-patch quantum finishes (~10-20 min):

**Scenario A: Quantum finds 4-8 meaningful clusters**
→ These are the natural primitives!
→ Analyze characteristics
→ Define primitives from quantum
→ Test if they work better than M/E/J/R/B/T

**Scenario B: Quantum finds same groups as classical**
→ Classical clustering is adequate
→ Problem structure is Euclidean (not quantum)
→ Stick with classical methods

**Scenario C: Quantum finds weird/uninterpretable groups**
→ Need more data OR different feature encoding
→ Quantum seeing structure but we don't understand it yet

---

## Strategic Use of 10 Free Minutes/Month

### Best Use (My Recommendation):

**Month 1 (December):**
- Run 50-patch quantum clustering validation
- Test: does real quantum agree with simulator?
- Cost: ~6 minutes, FREE
- Leaves 4 minutes for other experiments

**Month 2 (January):**
- Test quantum placement (QAOA on small problem)
- ~8 minutes
- See if quantum finds better Gaussian locations

**Month 3 (February):**
- Final validation of quantum-discovered primitives
- Or: test quantum-learned formula on 50 test cases
- ~10 minutes

**Each month: $0, learn something critical**

---

## Bottom Line

**Current experiment (running):**
- Quantum clustering 1,000 patches (simulator, free)
- Will reveal natural primitive groupings
- 10-20 minutes to results

**After results:**
- If quantum found different primitives → this changes everything
- Define new primitives from quantum
- Test if they work (classical experiments)
- Use quantum knowledge, deploy classically

**Cost: $0 for discovery phase**

**Your 10 min/month free tier:** Save for critical validations (not discovery)

---

**Quantum is revealing what image structure ACTUALLY is, not what we think it should be.**

Your classical experiments showed M/E/J/R/B/T is wrong (1.56 dB border failure).
Quantum will show what's RIGHT.

Results in ~15 minutes!
