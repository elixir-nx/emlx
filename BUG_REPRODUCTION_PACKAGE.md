# MLX Scalar Layout Bug - Complete Reproduction Package

## 📦 Package Contents

This directory contains everything needed to understand, reproduce, and report the MLX scalar layout bug discovered in EMLX.

### 📄 Documentation

1. **`BUG_REPRODUCTION_README.md`** (START HERE)
   - Complete overview of the bug
   - How to run reproduction scripts
   - Explanation of the bug mechanics
   - Next steps

2. **`MLX_BUG_REPORT.md`**
   - Formal technical bug report
   - Root cause analysis
   - Suggested fixes

3. **`MLX_GITHUB_ISSUE.md`**
   - Ready-to-post GitHub issue template
   - Concise reproduction case
   - Test case for MLX maintainers

4. **`BUG_FIX_SUMMARY.md`**
   - How the bug was discovered
   - Debug methodology
   - Workaround implementation in EMLX

### 🧪 Reproduction Scripts

**Main Reproductions:**
- `test_mlx_scalar_bug_raw.exs` - **Demonstrates the bug** (bypasses workaround)
- `test_mlx_scalar_bug.exs` - Shows the bug is fixed with workaround
- `test_scheduler_proper_state.exs` - Real-world impact test

**Debug Tools:**
- `debug_scheduler_divergence.exs` - Traces operations with Nx.Defn.Evaluator
- `find_divergence.exs` - Finds first divergence point
- `compare_debug_traces.exs` - Compares operation traces

### 🎯 Quick Start

```bash
# 1. See the bug in action
elixir test_mlx_scalar_bug_raw.exs

# 2. Verify the fix works
elixir test_mlx_scalar_bug.exs

# 3. Test real-world impact
elixir test_scheduler_proper_state.exs
```

## 🐛 Bug Summary

**What**: MLX creates scalar tensors with invalid memory layout after `slice` → `squeeze`

**Why**: Squeeze doesn't materialize scalars, leaving them as views into source array

**Impact**: `item()` reads 8 bytes across two consecutive values instead of 4 bytes

**Example**:
- Create array `[0, 1, ..., 951, 952, ...]`
- Extract scalar at index 951
- `item<int64>()` reads `[951, 952]` as single value
- Returns `4,088,808,866,743` instead of `951`

## 📊 Test Results

### Raw Bug (test_mlx_scalar_bug_raw.exs)
```
Index 951: Expected 951, Got 4,088,808,866,743 ❌
Index 998: Expected 998, Got 4,290,672,329,702 ❌
8 out of 9 test cases fail
```

### With Workaround (test_mlx_scalar_bug.exs)
```
All tests pass ✅
Mean difference: 1.0e-8
Std difference: 2.4e-7
```

## 🔧 The Fix

### Current Workaround (in EMLX)

```elixir
defp to_number(%T{} = t) do
  device_tuple = from_nx(t)

  # Force materialization by adding 0
  scalar_zero = EMLX.scalar_tensor(0, EMLX.scalar_type(device_tuple), elem(device_tuple, 0))
  ref_fixed = EMLX.add(device_tuple, scalar_zero)

  EMLX.item(ref_fixed)
end
```

**Location**: `lib/emlx/backend.ex:445-462`

### Needed in MLX

Either:
1. Fix `squeeze()` to materialize scalar results
2. Fix `item()` to handle views correctly
3. Both (recommended)

## 📝 How It Was Found

1. **Observed**: Bumblebee Stable Diffusion had 0.5 std deviation error on EMLX
2. **Traced**: Used `Nx.Defn.Evaluator` debug mode to log all 57 operations
3. **Compared**: Found first divergence at operation #22 (slice)
4. **Debugged**: Added instrumentation to `mlx_slice` function
5. **Discovered**: `to_number()` returned garbage: `3,869,765,534,647` instead of `951`
6. **Analyzed**: Examined memory layout, found repeating pattern
7. **Fixed**: Implemented workaround in `to_number()`
8. **Verified**: All tests pass, numerical differences eliminated

## 🚀 Next Steps

### For Reporting to MLX

1. Use `MLX_GITHUB_ISSUE.md` as the issue template
2. Link to this reproduction package
3. Include output from `test_mlx_scalar_bug_raw.exs`

### For EMLX Development

- ✅ Workaround implemented and working
- ⏳ Monitor MLX for upstream fix
- 🔄 Remove workaround once MLX is fixed
- 📝 Add regression tests

## 📫 Files You Need

**To report the bug to MLX:**
- `MLX_GITHUB_ISSUE.md` (copy/paste to GitHub)
- Output from `test_mlx_scalar_bug_raw.exs`

**To understand the bug:**
- `BUG_REPRODUCTION_README.md`
- `MLX_BUG_REPORT.md`

**To verify in your environment:**
- `test_mlx_scalar_bug_raw.exs`
- `test_mlx_scalar_bug.exs`

## 🎓 Key Learnings

1. **Use debug tracing**: `Nx.Defn.Evaluator` with `debug_options` is invaluable
2. **Compare operation-by-operation**: Find the exact divergence point
3. **Examine memory**: Sometimes the issue is in the tensor layout, not the operation
4. **Test at multiple levels**: From unit tests to integration tests
5. **Document thoroughly**: Makes bug reports actionable

## ⚠️ Important Notes

- The bug is **deterministic** and **reproducible**
- Affects **any integer type** (int8, int16, int32, int64)
- Only manifests when extracting scalars from views/slices
- Direct scalar creation works fine
- Workaround has minimal performance impact

---

**Created**: October 23, 2025
**Status**: Bug confirmed, workaround implemented, ready to report upstream
**Impact**: Critical for numerical correctness

