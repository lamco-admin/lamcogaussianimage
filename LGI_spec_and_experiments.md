# LGI: A Layered Gaussian Image Representation  
_A working concept + experimental plan_

**Author:** (Your Name Here)  
**Status:** Draft / Working Notes  

---

## 1. Executive Summary

This document defines a working specification and experimental roadmap for **LGI** (Gaussian / Large Gaussian Image representation): a way of representing images as **structured layers of 2D Gaussians**, analogous to how **RGB 0–255** provides a standardized representation for color.

Instead of treating an image as a uniform grid of pixels or generic patches, LGI:

- Decomposes the image into **semantic feature types**:
  - **Macro (M):** global low-frequency field  
  - **Edges (E):** curves / boundaries  
  - **Junctions (J):** corners / intersections  
  - **Regions (R):** homogeneous or smoothly shaded interiors  
  - **Blobs (B):** small spots / keypoints  
  - **Texture (T):** fine microstructure / noise-like detail
- Uses a **standard Gaussian atom** to represent all these features.
- Organizes Gaussians in **layers**, each targeting a residual left by previous layers:
  - `Image ≈ M + E + J + R + B + T + (optional residual cleanup)`

The **core hypothesis** is that:

> A layered, feature-driven Gaussian representation (**LGI**) can represent images more efficiently and more interpretably than a monolithic “bag of Gaussians”, **if** Gaussian placement and density are adapted to the underlying feature types (edges, regions, junctions, etc.) derived from a principled image analysis.

This document:

1. Defines a **canonical feature toolkit** (“the RGB of Gaussian splatting”).
2. States the **hypotheses and research questions** for LGI.
3. Describes an **initial experimental program** to test Gaussian placement rules for each feature type.
4. Outlines how to **expand testing** if initial results are promising.

---

## 2. Hypothesis and Research Questions

### 2.1 Main Hypothesis

**H1 – Layered Feature-Based Gaussian Representation**  
For a fixed Gaussian budget (total number of Gaussians, or equivalent bit-rate), a **layered LGI representation** that:

- separates the image into feature types (M/E/J/R/B/T),
- places Gaussians according to feature-specific rules derived from image analysis,

will achieve **better rate–distortion performance** and **fewer artifacts** than:

- a monolithic, feature-agnostic Gaussian fit,
- or simple heuristic approaches (uniform or random Gaussian placement).

### 2.2 Sub-Hypotheses

1. **H2 – Edge-Specific Placement Rules**  
   Optimal Gaussian density and shape along edges is a function of **edge curvature, blur, and contrast**, and can be described by simple rules:
   - Gaussians per unit length = f(|κ|, σ_edge, ΔI)
   - Gaussian aspect ratio and spacing can be adapted to curvature and blur.

2. **H3 – Region-Specific Rules**  
   The number and placement of Gaussians needed to approximate interior regions are determined primarily by:
   - region area,
   - boundary complexity,
   - interior variance / gradient energy,
   - high-frequency (HF) content.

3. **H4 – Junction Specialization**  
   Junctions (L, T, X) require a small, **specialized Gaussian configuration** (central blob + arm-aligned ellipses) that substantially improves local reconstruction compared to edge-only modeling.

4. **H5 – Layering vs Monolithic**  
   Given the same total Gaussian budget K_total, a **layered residual model**:
   - M → E → J → R → B → T
   will outperform a **single-layer Gaussian model** in terms of PSNR/SSIM and visual artifact profile.

5. **H6 – Texture Mode**  
   For complex textures, a **separate texture mode** (parametric or stochastic) may outperform dense micro-Gaussian fields in a rate–distortion sense.

### 2.3 Key Questions

- What is the minimal, **standard feature vocabulary** (like RGB) needed for Gaussified images?
- How do we map from **feature descriptors** (curvature, blur, variance, etc.) to:
  - Gaussian density,
  - placement patterns,
  - and shapes?
- At what point does **layering** produce diminishing returns vs complexity?
- Can these rules be learned experimentally and then expressed as compact **analytic formulas**, not just as neural networks?

---

## 3. LGI Feature Toolkit Specification v0.1

This section defines the **“standard toolkit”** for LGI: the feature types, the Gaussian atom, descriptors, layers, and encoder/decoder behaviors.

### 3.1 Terminology

- **LGI** – “Gaussian Image” / “Large Gaussian Image”: image represented as layers of Gaussians.
- **Gaussian atom** – 2D elliptical Gaussian with color and weight.
- **Feature** – structured entity in the image (edge, region, junction, blob, texture zone).
- **Layer** – group of Gaussians associated with a feature type and residual stage.
- **Residual** – difference between original image and reconstruction from earlier layers.

---

### 3.2 Gaussian Atom

Common parametrization for all Gaussians:

- Position:
  - `x, y` – center in image coordinates (normalized or pixel).
- Shape:
  - `sigma_parallel` (σ‖) – std. dev. along major axis.
  - `sigma_perp` (σ⊥) – std. dev. along minor axis.
  - `theta` – orientation of major axis (radians).
- Color:
  - `r, g, b` in [0,1] or [0,255], or `Y, Cb, Cr`.
- Weight / opacity:
  - `alpha` in [0,1].

Canonical Gaussian atom:

```text
G = {
  x, y,
  sigma_parallel, sigma_perp, theta,
  color: (r, g, b) or (Y, Cb, Cr),
  alpha
}
```

Rendered contribution at pixel (X, Y):

G(X, Y) = alpha * Color * exp( -0.5 * qᵀ Σ⁻¹ q )

where:

- q = [X - x, Y - y]ᵀ
- Σ = R(theta) * diag(sigma_parallel², sigma_perp²) * R(theta)ᵀ
- R(theta) is the 2D rotation matrix.

---

### 3.3 Canonical Feature Types

LGI defines six canonical feature types. Every Gaussian belongs to exactly one of these layers:

1. **M – Macro / Base Field**  
   - Very low-frequency content: lighting, broad gradients, sky, large soft shapes.  
   - Few, very large Gaussians covering the entire image.

2. **E – Edges / Curves**  
   - Object boundaries, contours, lines, silhouettes.  
   - Chains of elongated Gaussians aligned along curves.

3. **J – Junctions / Corners**  
   - L-corners, T-junctions, X-crossings, high-curvature points.  
   - Clusters of overlapping Gaussians at feature points, multiple orientations.

4. **R – Smooth Regions**  
   - Interiors of objects, uniform walls, skies, gently varying patches.  
   - Medium-to-large Gaussians filling region interiors.

5. **B – Blobs / Keypoints**  
   - Stars, small highlights, spots, LoG/DoG keypoints.  
   - Typically one (often isotropic) Gaussian per blob.

6. **T – Texture / Microstructure**  
   - Grass, fabric, fine repetitive patterns, noise-like detail.  
   - Either dense fields of small Gaussians or a separate parametric texture model.

Each Gaussian is tagged:

```text
layer_type ∈ { M, E, J, R, B, T }
```

---

### 3.4 Feature Descriptors

Encoders SHOULD extract feature descriptors as the analysis vocabulary for LGI. These descriptors are independent of any specific Gaussian placement algorithm.

#### 3.4.1 Edge Features (E)

Edges are modeled as curves parameterized by arc length s ∈ [0, L].

Per point on the curve:

- `x(s), y(s)` – position
- `theta(s)` – orientation (tangent angle)
- `kappa(s)` – curvature (signed or magnitude)
- `sigma_edge(s)` – edge blur / thickness (from intensity profile along normal)
- `delta_I(s)` – local contrast (difference between two sides of edge)

Canonical edge descriptor:

```text
E(s) = {
  x(s), y(s),
  theta(s),
  kappa(s),
  sigma_edge(s),
  delta_I(s)
}
```

The full edge feature is a set `{E(s_i)}` along the curve.

---

#### 3.4.2 Junction Features (J)

Junctions occur where multiple edges meet.

- `x, y` – junction location
- `N` – number of arms (N ≥ 2)
- For each arm i:
  - `theta_i` – edge orientation near the junction
  - `sigma_edge_i` – blur/thickness for that arm
  - `delta_I_i` – contrast along that arm

Optional type label:

- `type ∈ { "L", "T", "X", "star", "other" }`

Canonical junction descriptor:

```text
J = {
  x, y,
  N,
  arms: [
    { theta: theta_1, sigma_edge: sigma_edge_1, delta_I: delta_I_1 },
    ...
    { theta: theta_N, sigma_edge: sigma_edge_N, delta_I: delta_I_N }
  ],
  type: "L" | "T" | "X" | "star" | "other"  // OPTIONAL
}
```

---

#### 3.4.3 Region Features (R)

Regions are coherent areas separated by edges.

- `mask` – region mask or polygon / superpixel id
- `A` – area
- `P` – perimeter
- `boundary_complexity` – e.g. P² / A
- Interior stats:
  - `mu_color` – mean color
  - `sigma2` – variance
  - `E_HF` – high-frequency energy (e.g. from a high-pass filter)

Canonical region descriptor:

```text
R = {
  mask_id or polygon,
  A,
  P,
  boundary_complexity,
  mu_color,
  sigma2,
  E_HF
}
```

---

#### 3.4.4 Blob Features (B)

Blobs are local maxima/minima (e.g. from LoG/DoG).

- `x, y` – location
- `sigma_blob` – scale (from detector)
- `amplitude` – intensity
- Optional `eccentricity` – 0 = isotropic, →1 = elongated

Canonical blob descriptor:

```text
B = {
  x, y,
  sigma_blob,
  amplitude,
  eccentricity  // OPTIONAL
}
```

---

#### 3.4.5 Texture Features (T)

Texture zones are regions with substantial high-frequency or structured micro-pattern.

- `mask` – zone mask or polygon
- `E_HF` – high-frequency energy
- `orientation_spectrum` – distribution over orientations
- `frequency_spectrum` – distribution over spatial frequencies
- `periodicity_params` – fundamental periods if a repeating pattern is detected (optional)

Canonical texture descriptor:

```text
T = {
  mask_id or polygon,
  E_HF,
  orientation_spectrum,
  frequency_spectrum,
  periodicity_params  // OPTIONAL
}
```

---

### 3.5 Layer Model

LGI uses a layered residual model:

```text
Image ≈ M + E + J + R + B + T + (optional residual cleanup)
```

Where each term is the rendered contribution of all Gaussians in that layer.

Recommended encoder order:

1. **Layer M (Macro / Base)**  
   - Fit a small set of large Gaussians for low-frequency content.  
   - Residual: R1 = Image − M.

2. **Layer E (Edges)**  
   - Detect edges on original image, fit edge Gaussians to residual R1 in edge strips.  
   - Residual: R2 = R1 − E.

3. **Layer J (Junctions)**  
   - Detect junctions, fit junction Gaussians to R2 around junctions.  
   - Residual: R3 = R2 − J.

4. **Layer R (Regions)**  
   - Fit region Gaussians to R3 within region interiors.  
   - Residual: R4 = R3 − R.

5. **Layer B (Blobs)**  
   - Fit blob Gaussians to R4 at blob locations.  
   - Residual: R5 = R4 − B.

6. **Layer T (Texture / Microstructure)**  
   - Fit dense micro-Gaussians to R5 in texture zones, or use a separate texture codec.  
   - Residual: R6 = R5 − T.

7. Optional final residual cleanup layer (if needed).

Implementations MAY deviate from this ordering but SHOULD document their choices.

---

### 3.6 Encoder Operations

Encoders SHOULD implement:

1. **Feature Extraction**
   - Compute descriptors:
     - E(s), J, R, B, T from the original image (or current residual if desired).

2. **Complexity Mapping**
   - Map descriptors → target Gaussian densities:
     - Edges: Gaussians per unit length = f(κ, σ_edge, ΔI, global stats)
     - Regions: Gaussians per area = f(σ², E_HF, boundary_complexity)
     - Junctions: Gaussians per junction = f(N, arm angles, blur)
     - Blobs: usually 1 per blob, more if elongated/complex
     - Texture: Gaussians per area or separate param model

3. **Placement Initialization**
   - M: large Gaussians on a coarse grid / fitted to downsampled image.
   - E: Gaussians aligned with θ(s), σ⊥ ≈ σ_edge(s), placed along curves.
   - J: small clusters at junction centers, oriented along arms.
   - R: Gaussians within region interiors, size tied to region scale and gradients.
   - B: one Gaussian per blob center, σ ≈ σ_blob.
   - T: dense micro-Gaussians or texture model parameters.

4. **Layered Residual Fitting**
   - For each layer, fit Gaussian parameters against the current residual using an optimizer (e.g. gradient descent / Adam).
   - Update residual after each layer.

5. **Optional Joint Refinement**
   - Final joint optimization over all Gaussians with regularization:
     - e.g., encourage color consistency, reduce artifacts at boundaries.

---

### 3.7 Decoder Operations

Given Gaussians grouped by layer_type:

1. **Full Reconstruction**

I(X,Y) = sum_{G ∈ M,E,J,R,B,T} G(X,Y)

2. **Layer-wise Reconstruction**
   - Ability to reconstruct:
     - M only,
     - M+E, M+E+J, etc., for debugging/visualization.

3. **Level-of-Detail Control**
   - For downscaling or abstraction:
     - optionally drop small-scale Gaussians (small σ),
     - or drop specific layers (e.g. T) to get lower-detail views.

---

## 4. Initial Experimental Steps

The goal of initial experiments is to test individual aspects of the LGI hypothesis, starting with simpler feature types and gradually integrating layers.

### 4.1 Shared Experimental Infrastructure

Before feature-specific experiments, implement:

1. **Gaussian Renderer**
   - Input: list of Gaussians → output image.
   - Support layered rendering: render one layer or a cumulative sum.

2. **Optimizer**
   - Given:
     - a target image or residual,
     - initial Gaussian parameters (some fixed, some free),
   - minimize a loss (MSE, possibly SSIM later).
   - Use Adam or L-BFGS for small sets of Gaussians.

3. **Metrics**
   - PSNR, SSIM, MSE.
   - Optional: edge-aware metrics (error near edges vs flat regions).

4. **Logging**
   - Log for each experiment:
     - feature descriptors (E, J, R, etc.),
     - placement strategy used,
     - number of Gaussians per layer,
     - final error metrics,
     - any qualitative notes (artifacts, halos, etc.).

---

### 4.2 Step 1 – Macro Layer Experiments (M-only)

Objective: Learn basic rules for how many large Gaussians are needed for the global low-frequency field and how to place them.

- Data:
  - Synthetic gradients, vignettes, simple blobs.
  - Downsampled natural images (e.g. 64×64).

- Strategies:
  - M1: fixed grid of large isotropic Gaussians.
  - M2: K-means clustering of pixel positions, one Gaussian per cluster.
  - M3: gradient-aware placement (more Gaussians in areas of higher low-frequency gradient).

- Procedure:
  - For each image, vary K (e.g., 4, 8, 16, 32).
  - Fit only macro layer (no E/J/R/B/T).
  - Evaluate PSNR/SSIM; inspect residuals.

- What to learn:
  - Rough function: K_M ≈ f(image size, global gradient energy).
  - How much structure is already captured by M, and where residual energy tends to live (edges, spots, etc.).

---

### 4.3 Step 2 – Edge Layer Experiments (E on Residual after M)

Objective: Understand Gaussian placement along edges when they operate on a residual after the macro layer.

- Data:
  - Synthetic scenes with known base + edges:
    - e.g., smooth gradient background + single step edge, curved edges, blurred edges.
  - Optionally, simple real images dominated by boundaries.

- Pipeline:
  1. Fit M to the full image → obtain M.
  2. Residual R1 = Image − M.
  3. Extract edge descriptors E(s) from Image (curves, κ(s), σ_edge(s), ΔI(s)).
  4. Place edge Gaussians using different strategies.

- Placement strategies:
  - E1: uniform spacing along arc length; fixed σ⊥ from σ_edge, fixed σ‖.
  - E2: curvature-adaptive spacing (denser in high |κ| regions).
  - E3: curvature-adaptive σ‖ (shorter Gaussians where curvature is high).
  - E4: combined spacing + σ‖ adaptation.

- Variables:
  - Edge curvature (straight, gently curved, highly curved).
  - Edge blur (sharp vs soft).
  - Edge contrast (ΔI high vs low).
  - Gaussian density: Gaussians per unit length.

- What to measure:
  - Error reduction (R1 → R2) vs Gaussian density.
  - Which strategy performs best as a function of κ, σ_edge, ΔI.
  - Derived rule: Gaussian_density_E(s) = f(|kappa|, sigma_edge, delta_I, local residual energy).

- Decision to expand:
  - If feature-specific rules clearly emerge (e.g., curvature-adaptive strategies dominate), formalize them into simple analytic formulas for LGI encoder.

---

### 4.4 Step 3 – Junction Layer Experiments (J after M+E)

Objective: Validate that specialized Gaussian clusters at junctions significantly improve local reconstruction over edge-only modeling.

- Data:
  - Synthetic junctions (L, T, X) with different angles, blur, and contrasts.
  - Simple multi-edge scenes.

- Pipeline:
  1. Fit M.
  2. Fit E along edges (using best strategy from Step 2).
  3. Residual R2 = Image − (M + E).
  4. Extract junction descriptors J from edge graph.

- Strategies:
  - J1: single isotropic Gaussian at junction center.
  - J2: central Gaussian + one elongated Gaussian per arm.
  - J3: only arm Gaussians, no central.
  - J4: small cluster of overlapping Gaussians at multiple orientations.

- What to measure:
  - Local residual reduction near junctions.
  - Best strategy as function of:
    - N arms, arm angles,
    - blur, contrast.
  - Whether a central blob is consistently beneficial.

- Decision to expand:
  - If junction-specific configurations demonstrably reduce artifacts, define a standard junction placement rule in LGI (e.g., N+1 Gaussians per junction with given angle layout).

---

### 4.5 Step 4 – Region Layer Experiments (R after M+E+J)

Objective: Determine how to fill region interiors with Gaussians to handle residual structure not captured by M/E/J.

- Data:
  - Synthetic shapes: rectangles, ellipses, polygons, irregular shapes.
  - Shading types: constant, linear gradient, quadratic, mild texture.

- Pipeline:
  1. Fit M to whole image.
  2. Fit E and J to boundaries and junctions.
  3. Residual R3 = Image − (M + E + J).
  4. Extract region descriptors R.

- Strategies:
  - R1: one Gaussian per region (centered).
  - R2: K Gaussians per region (e.g. 2, 4, 8) at centroid + along principal axes.
  - R3: area-proportional Gaussians, more near complex boundaries.
  - R4: gradient-aware placement inside region (more Gaussians where gradient is higher).

- What to measure:
  - Error reduction vs number of region Gaussians.
  - Relationship between:
    - region area A,
    - boundary_complexity,
    - interior variance / E_HF,
  and required K.

- Decision to expand:
  - If stable relationships appear, define analytic rule:
    - K_region ≈ f(A, boundary_complexity, sigma2, E_HF)
  - Integrate this into LGI encoder.

---

### 4.6 Step 5 – Blob Layer Experiments (B after M+E+J+R)

Objective: Understand whether blobs can be modeled simply and cheaply.

- Data:
  - Synthetic blobs of varying size, shape, and brightness.
  - Real patches with stars, specular highlights.

- Pipeline:
  1. Fit M, E, J, R.
  2. Residual R4 = Image − (M + E + J + R).
  3. Detect blobs and extract B descriptors.

- Strategies:
  - B1: one isotropic Gaussian per blob.
  - B2: elliptical Gaussians for elongated blobs.
  - B3: mini-mixtures of 2–3 Gaussians per blob.

- What to measure:
  - Error vs K per blob.
  - Whether B1 is sufficient in most cases.

- Decision to expand:
  - If 1 Gaussian per blob is nearly always sufficient, B can remain simple in LGI spec.

---

### 4.7 Step 6 – Texture Layer Experiments (T after all others)

Objective: Decide whether texture is best handled as dense Gaussians or via a separate parametric texture mode.

- Data:
  - Synthetic textures: periodic (checkerboard, grids), stochastic (noise, grass-like).
  - Real texture patches (fabric, foliage).

- Pipeline:
  1. Fit M, E, J, R, B.
  2. Residual R5 = Image − (M + E + J + R + B).
  3. Identify texture zones with T descriptors.

- Strategies:
  - T1: dense micro-Gaussian fields (grid or jittered).
  - T2: parametric texture model with limited parameters.
  - T3: hybrid of a few Gaussians + param residual model.

- What to measure:
  - Rate–distortion for each strategy.
  - Perceptual visual quality (textures are tolerant to error).

- Decision to expand:
  - If T2/T3 performs better, formally treat T as a separate codec mode in LGI, rather than just more Gaussians.

---

### 4.8 Step 7 – Layered vs Monolithic Comparison

Objective: Empirically test the value of layering vs a single Gaussian layer.

For a given image or synthetic scene:

1. **Monolithic Fit**
   - Fit a single layer of K Gaussians (no feature types, free placement).
2. **Layered Fit**
   - Use LGI pipeline: M → E → J → R → B → T with total Gaussians K_total = K.

Comparison:

- PSNR/SSIM and perceptual quality.
- Artifact types (halos, seams, over-smoothing).
- Interpretability (feature breakdown vs opaque cloud of Gaussians).

If layered consistently outperforms monolithic for the same K_total, this strongly supports H1 & H5 and justifies keeping the LGI layers as a core design principle.

---

## 5. Expansion Path if Results are Promising

If the above initial experiments produce promising and consistent patterns:

1. **Formalize Analytic Rules**
   - Convert empirical relationships into simple formulas:
     - edge density and aspect ratio rules,
     - region Gaussian counts,
     - junction Gaussian configurations, etc.

2. **Build a Prototype LGI Encoder**
   - Implement the full LGI pipeline with:
     - feature extraction,
     - analytic placement rules,
     - limited optimization steps (for practical speed).

3. **Evaluate on Diverse Datasets**
   - Natural images (landscapes, portraits, urban scenes).
   - Diagrams and synthetic graphics (text, line art).
   - Compare against:
     - monolithic Gaussian fits,
     - existing Gaussian-image codecs (GaussianImage, LIG, etc.),
     - conventional image codecs at approximate bit-rates.

4. **Refine the Toolkit**
   - Decide whether the 6 feature types (M, E, J, R, B, T) are sufficient.
   - Consider adding semantic layers (e.g., object-aware Gaussians) if warranted.

5. **Investigate Learned Guidance**
   - Once analytic rules are stable, optionally train small models to:
     - predict Gaussian densities or placements from feature descriptors,
     - guide or initialize the analytic rules,
     - not replace the interpretable LGI representation but enhance it.

6. **Bitstream & Codec Design**
   - Define a compact binary format:
     - header (image size, color space, global parameters),
     - per-layer Gaussian parameters (quantized),
     - optional texture model parameters.
   - Explore rate–distortion performance vs standard codecs.

---

## 6. Summary

This document proposes:

- A standardized feature toolkit for Gaussian image representation (LGI),
- A layered residual model with six canonical layers (M/E/J/R/B/T),
- A Gaussian atom spec that applies uniformly across features,
- And a stepwise experimental program to:
  - learn where and how to place Gaussians for each feature type,
  - validate the value of layering vs monolithic models,
  - and refine LGI into a potentially practical, interpretable Gaussian image codec.

The immediate next steps are:

1. Implement the shared infrastructure:
   - Gaussian renderer,
   - optimizer,
   - metrics and logging.
2. Run macro-only (M) and edge-on-residual (E) experiments as described.
3. Iterate on rules and expand to J, R, B, T as results justify.

This provides a clear, testable path from concept to a concrete LGI prototype.
