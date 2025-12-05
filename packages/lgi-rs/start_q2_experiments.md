# Q2 Experiments Starting Now

## Optimizers Available for Testing

1. **Adam** - First-order adaptive (current, proven to work)
2. **OptimizerV2** - Gradient descent, can use MS-SSIM or edge-weighted loss  
3. **OptimizerV3** - Perceptual optimizer (MS-SSIM + edge-weighted)

L-BFGS exists but broken (needs argmin dependency + bug fixes). Can add later if needed.

## Immediate Test (Next 30 min)

**Question**: Do Channels 3-4-7 (high quality, small isotropic) perform better with OptimizerV2/V3 than Adam?

**Test**: Single image (kodim03), 3 channels, 3 optimizers = 9 quick experiments

**If promising**: Build full harness and run overnight

## Proceeding with implementation...
