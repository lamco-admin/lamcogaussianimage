# Test Suite Analysis Tools

Scripts for querying and analyzing test results.

## Available Scripts

### query_results.py
Query JSON files by metadata fields.

Usage:
```bash
python query_results.py --optimizer Adam --min-psnr 30.0
python query_results.py --module toolkit
python query_results.py --image kodim03
```

### Future Scripts

**compare_optimizers.py** - Statistical comparison of optimizers
**analyze_toolkit.py** - Toolkit module contribution analysis
**aggregate_findings.py** - Generate summary reports from data
**visualize_comparisons.py** - Plot comparison charts

---

## JSON Query Patterns

### Using jq directly

**Group by optimizer:**
```bash
cd ../data
jq -s 'group_by(.optimizer) | map({opt: .[0].optimizer, avg: (map(.final_psnr)|add/length)})' *.json
```

**Find best configuration:**
```bash
jq -s 'sort_by(.final_psnr) | reverse | .[0]' *.json
```

**Count tests by module:**
```bash
jq -s 'group_by(.module_under_test) | map({module: .[0].module_under_test, count: length})' *.json
```

---

*Analysis tools enable data-driven insights without rigid folder hierarchies.*
