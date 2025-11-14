# Correct Quantization Implementation Plan

**Status**: PAUSED for proper design review
**Date**: October 2, 2025

---

## Issues Identified

1. **Rushed implementation** - Making changes without full understanding
2. **Specification misreading** - LGIQ-H should use float16, not integer quantization
3. **Byte count confusion** - Inconsistent between 11/12/13 bytes for LGIQ-B
4. **Missing opacity** handling in some code paths
5. **Test failures** - Due to incorrect size assumptions

---

## Specification-Compliant Design

### LGIQ-B: Balanced
**Spec says**: ~11 bytes
**Actual with all fields**:
- Position: 16-bit × 2 = 4 bytes
- Scale: 12-bit × 2 = 3 bytes (packed)
- Rotation: 12-bit = 1.5 bytes
- Color: 8-bit × 3 = 3 bytes
- Opacity: 8-bit = 1 byte
**Total**: 12.5 bytes

**Implementation Options**:
1. **11-byte tight pack**: Requires splitting rotation+opacity across boundaries (complex)
2. **13-byte byte-aligned**: Clean implementation, all precision maintained
3. **12-byte hybrid**: Possible but requires careful packing

**Recommendation**: **13 bytes byte-aligned** - Spec says "~11" (approximate), 13 is acceptable

---

### LGIQ-S: Standard
**Spec says**: ~13 bytes
**Correct implementation**: 14 bytes byte-aligned
- Position: 16-bit × 2 = 4 bytes
- Scale: 14-bit × 2 = 4 bytes (store as 16-bit)
- Rotation: 14-bit = 2 bytes (store as 16-bit)
- Color: 10-bit × 3 = 4 bytes (packed)
- Opacity: 10-bit = 2 bytes (store as 16-bit)
**Total**: 16 bytes (or 14 with careful packing)

**Recommendation**: **16 bytes byte-aligned** for simplicity

---

### LGIQ-H: High Quality
**Spec says**: 18 bytes with **float16 (IEEE 754)**

**CRITICAL**: This should use half-precision floats, NOT integer quantization!

```rust
// Use half crate for f16 support
use half::f16;

fn quantize_high(g: &Gaussian2D) -> Vec<u8> {
    let mut data = Vec::new();

    // All 9 parameters as f16 (2 bytes each)
    data.extend(f16::from_f32(g.position.x).to_le_bytes());
    data.extend(f16::from_f32(g.position.y).to_le_bytes());
    data.extend(f16::from_f32(g.scale_x).to_le_bytes());
    data.extend(f16::from_f32(g.scale_y).to_le_bytes());
    data.extend(f16::from_f32(g.rotation).to_le_bytes());
    data.extend(f16::from_f32(g.color.r).to_le_bytes());
    data.extend(f16::from_f32(g.color.g).to_le_bytes());
    data.extend(f16::from_f32(g.color.b).to_le_bytes());
    data.extend(f16::from_f32(g.opacity).to_le_bytes());

    // Total: 9 × 2 = 18 bytes
    data
}
```

**Recommendation**: Use `half` crate for proper IEEE 754 float16 support

---

### LGIQ-X: Lossless
**Spec says**: 36 bytes with float32

**Current implementation**: ✅ CORRECT (already using f32 serialization)

---

## Correct Implementation Strategy

### Phase 1: Implement LGIQ-X First (Lossless)
- **Why**: Simplest - just f32 serialization
- **Time**: Already done, verify it works
- **Validates**: Round-trip infrastructure

### Phase 2: Implement LGIQ-H (float16)
- **Why**: Well-defined (IEEE 754 standard)
- **Requires**: `half` crate dependency
- **Time**: 30 minutes
- **Validates**: float16 serialization

### Phase 3: Implement LGIQ-B (13-byte byte-aligned)
- **Why**: Clean, maintainable
- **Time**: 30 minutes
- **Validates**: Integer quantization + log encoding

### Phase 4: Implement LGIQ-S (Optional)
- **Why**: Similar to LGIQ-B, just more bits
- **Time**: 15 minutes
- **Validates**: Scalability

### Phase 5: Add zstd Compression
- **Why**: Achieves final 5-10× target
- **Time**: 30 minutes
- **Validates**: Full compression pipeline

---

## Dependencies Needed

```toml
[dependencies]
half = "2.3"  # For f16 (IEEE 754 half-precision)
```

---

## Test Strategy

For each profile:
1. **Round-trip test**: quantize → dequantize → compare
2. **Size test**: Verify byte count
3. **Precision test**: Check reconstruction error within expected bounds
4. **Lossless test**: LGIQ-X must be bit-exact

---

## Success Criteria

- ✅ All 4 profiles implemented correctly
- ✅ All round-trip tests pass
- ✅ Byte counts match implementation decision (not necessarily spec "~" values)
- ✅ Precision maintained per profile
- ✅ LGIQ-X is bit-exact
- ✅ Clear documentation of any spec deviations

---

**Next Step**: Implement profiles ONE AT A TIME, starting with LGIQ-X (lossless), testing thoroughly before moving to next.
