# Code Updates Required After Reorganization
**Date:** 2025-12-07
**Status:** Reorganization complete, code updates needed

---

## ✅ WHAT WAS REORGANIZED

### Test Infrastructure Moved

**From:** `packages/lgi-rs/lgi-encoder-v2/examples/` (212 files)
**To:** `test-suite/scripts/` (organized by category)

**From:** `packages/lgi-rs/lgi-encoder-v2/test-results/` (678 JSON + 193 PNG)
**To:** `test-suite/data/` (flat, queryable)

**Findings docs:** `test-suite/findings/` (23 markdown files)

---

## 🔧 REQUIRED CODE UPDATES

### 1. UPDATE TestResult Framework ⚠️ CRITICAL

**File:** `packages/lgi-rs/lgi-encoder-v2/src/test_results.rs`

**Current code (line ~100-150):**
```rust
pub fn save_with_image(&self, rendered: &ImageBuffer, base_dir: &str) -> Result<()> {
    // Current: Creates date folder
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let dir = PathBuf::from(base_dir).join(&date);
    std::fs::create_dir_all(&dir)?;

    // Saves to: base_dir/YYYY-MM-DD/test_name_HH-MM-SS.json
    let path = dir.join(format!("{}_{}.json", self.test_name, timestamp));
}
```

**Required change:**
```rust
pub fn save_with_image(&self, rendered: &ImageBuffer, data_dir: &str) -> Result<()> {
    // NEW: No date folder, flat structure
    let dir = PathBuf::from(data_dir);
    std::fs::create_dir_all(&dir)?;

    // Saves to: data_dir/test_name_HH-MM-SS.json (flat)
    let path = dir.join(format!("{}_{}.json", self.test_name, timestamp));
}
```

**Impact:** All future tests will save to flat directory

**Backward compat:** Could add parameter `use_date_folders: bool = false`

---

### 2. UPDATE Test Script Constants ⚠️ AFFECTS ALL 212 TESTS

**Files:** All 212 test scripts in `test-suite/scripts/`

**Current (in every test):**
```rust
const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";
```

**Required change:**
```rust
const RESULTS_DIR: &str = "test-suite/data";
```

**Migration approaches:**

**Option A:** Global search-replace (fast but brittle)
```bash
find test-suite/scripts -name "*.rs" -exec sed -i 's|lgi-encoder-v2/test-results|test-suite/data|g' {} \;
```

**Option B:** Add config file (better)
Create `test-suite/config.toml`:
```toml
[paths]
data_dir = "test-suite/data"
findings_dir = "test-suite/findings"
```

Then in tests:
```rust
const CONFIG: &str = include_str!("../../config.toml");
// Parse and use
```

**Option C:** Environment variable
```rust
const RESULTS_DIR: &str = env!("TEST_RESULTS_DIR", "test-suite/data");
```

Set when running:
```bash
TEST_RESULTS_DIR=test-suite/data cargo run --example test_name
```

**Recommendation:** Option A for now (simple), migrate to Option B later

---

### 3. UPDATE Cargo.toml Example Configuration

**File:** `packages/lgi-rs/lgi-encoder-v2/Cargo.toml`

Cargo expects examples in `examples/` by default. Since we moved them:

**Check current [[example]] declarations:**
```toml
# If examples are declared explicitly, may need path updates
[[example]]
name = "isolated_structure_tensor_test"
path = "examples/isolated_structure_tensor_test.rs"  # OLD PATH
```

**May need:**
```toml
[[example]]
name = "isolated_structure_tensor_test"
path = "../../../test-suite/scripts/toolkit/isolated_structure_tensor_test.rs"  # NEW PATH
```

**OR:** Create symlink:
```bash
ln -s ../../../../test-suite/scripts examples
```

**Check:** `grep "\[\[example\]\]" packages/lgi-rs/lgi-encoder-v2/Cargo.toml`

---

### 4. UPDATE Image Path References

**Many tests reference:**
```rust
let image_path = format!("test-data/kodak-dataset/{}", img_name);
let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);
```

**Issue:** Relative paths may break with new script locations

**Solution:** Use absolute or workspace-relative paths:
```rust
// Workspace root
let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
    .ancestors()
    .nth(3)  // lgi-encoder-v2 -> lgi-rs -> packages -> lgi-project
    .unwrap();
let full_path = workspace_root.join("test-data/kodak-dataset").join(img_name);
```

**Or:** Add to config.toml:
```toml
[paths]
test_data = "test-data"
```

---

### 5. UPDATE Documentation References

**Files affected:**
- `CLAUDE.md` - Test running examples
- `docs/README.md` - Test suite references
- All catalog files mentioning test locations

**Example in CLAUDE.md:**
```markdown
# OLD
Results saved to: lgi-encoder-v2/test-results/YYYY-MM-DD/

# NEW
Results saved to: test-suite/data/
```

---

## 📋 UPDATE CHECKLIST

### Phase 1: Critical Path (Required for tests to run)

- [ ] Update TestResult framework (test_results.rs)
  - Remove date folder creation
  - Save flat to data_dir

- [ ] Update all 212 test script constants
  - Change RESULTS_DIR paths
  - Verify relative paths still work

- [ ] Test that examples compile from new locations
  - Check Cargo.toml example paths
  - Create symlink if needed

### Phase 2: Paths & References

- [ ] Update image path resolution in tests
  - Ensure test-data/ accessible from new locations

- [ ] Update CLAUDE.md documentation
  - New test running commands
  - New result locations

- [ ] Update catalog references
  - Point to new test-suite/ structure

### Phase 3: Migration & Verification

- [ ] Run one toolkit test to verify
  - Check JSON saves to test-suite/data/
  - Check PNG saves with same name
  - Verify no date folder created

- [ ] Run query_results.py on existing data
  - Verify 678 JSON files queryable

- [ ] Update findings documents
  - Reference new data/ location

---

## 🔍 VERIFICATION COMMANDS

### Check Cargo can find examples

```bash
cargo build --example isolated_structure_tensor_test -p lgi-encoder-v2
```

Should succeed if:
- Cargo.toml has correct paths
- Or examples/ symlink exists
- Or default path assumption works

### Check Test Execution

```bash
cargo run --release --example isolated_structure_tensor_test -p lgi-encoder-v2
```

Should:
- Run successfully
- Save to test-suite/data/test_name_timestamp.json
- Save to test-suite/data/test_name_timestamp.png
- NOT create date folders

### Check Data Queryability

```bash
cd test-suite/data
jq 'keys' isolated_structure_tensor_*.json | head -1
# Should show all JSON fields

python ../analysis/query_results.py --optimizer Adam
# Should list all Adam results
```

---

## 🚨 POTENTIAL ISSUES

### Issue 1: Cargo Example Path Resolution

**Symptom:** `cargo: no example target named 'test_name'`

**Cause:** Cargo can't find examples in new location

**Fix:** Add to `packages/lgi-rs/lgi-encoder-v2/Cargo.toml`:
```toml
[[example]]
name = "isolated_structure_tensor_test"
path = "../../../test-suite/scripts/toolkit/isolated_structure_tensor_test.rs"
```

Or symlink:
```bash
cd packages/lgi-rs/lgi-encoder-v2
ln -s ../../../test-suite/scripts examples
```

### Issue 2: Test Scripts Can't Find test-data/

**Symptom:** "Failed to load image" errors

**Cause:** Relative path `../../../test-data/` incorrect from new location

**Fix:** Update path calculation in tests or use workspace-root-relative paths

### Issue 3: Old Code Still References Old Paths

**Symptom:** New test results save to old location

**Cause:** Forgot to update RESULTS_DIR constant

**Fix:** Search and replace all instances

---

## 📝 RECOMMENDED ORDER

1. **Verify Cargo can still find examples**
   - Try building one test
   - Add symlink or Cargo.toml paths if needed

2. **Update TestResult framework**
   - Remove date folder logic
   - Test with one manual run

3. **Bulk update test script constants**
   - sed command on all 212 files
   - Or add config file approach

4. **Run verification tests**
   - One test from each category
   - Verify data saves correctly

5. **Update documentation**
   - CLAUDE.md paths
   - Catalog references

---

*This reorganization separates test infrastructure from implementation code. All updates are mechanical, no logic changes required.*
