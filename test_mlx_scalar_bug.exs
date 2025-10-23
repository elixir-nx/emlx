Mix.install([
  {:emlx, path: "/Users/valente/coding/emlx"},
  {:nx, override: true}
])

# Minimal reproduction of MLX scalar layout bug
# Bug: item() returns garbage after slice → squeeze operations

IO.puts("=" |> String.duplicate(70))
IO.puts("MLX Scalar Layout Bug - Minimal Reproduction")
IO.puts("=" |> String.duplicate(70))

Nx.global_default_backend(EMLX.Backend)

# Test 1: Direct scalar creation (should work)
IO.puts("\n📝 Test 1: Direct scalar creation")
direct_scalar = Nx.tensor(951, type: :s32, backend: EMLX.Backend)
direct_value = Nx.to_number(direct_scalar)
IO.puts("Value: #{direct_value}")
IO.puts("Expected: 951")
IO.puts("Status: #{if direct_value == 951, do: "✅ PASS", else: "❌ FAIL"}")

# Test 2: Scalar from slice + squeeze (triggers bug)
IO.puts("\n📝 Test 2: Scalar from slice → squeeze")
array = Nx.iota({1000}, type: :s32, backend: EMLX.Backend)
sliced = Nx.slice_along_axis(array, 951, 1, axis: 0)  # Creates [951] with shape {1}
scalar = Nx.squeeze(sliced, axes: [0])                 # Creates 951 with shape {}

IO.puts("Array shape: #{inspect(array.shape)}")
IO.puts("Sliced shape: #{inspect(sliced.shape)}")
IO.puts("Scalar shape: #{inspect(scalar.shape)}")
IO.puts("Scalar: #{inspect(scalar)}")

# This is where the bug manifests
value = Nx.to_number(scalar)
IO.puts("\nValue: #{value}")
IO.puts("Expected: 951")
IO.puts("Status: #{if value == 951, do: "✅ PASS", else: "❌ FAIL - Got #{value} instead!"}")

# Test 3: Multiple indices to show the pattern
IO.puts("\n📝 Test 3: Testing multiple indices")

test_indices = [0, 1, 100, 500, 900, 950, 951, 998, 999]
failures = []

for idx <- test_indices do
  sliced = Nx.slice_along_axis(array, idx, 1, axis: 0)
  scalar = Nx.squeeze(sliced, axes: [0])
  value = Nx.to_number(scalar)

  status = if value == idx do
    "✓"
  else
    failures = [{idx, value} | failures]
    "✗"
  end

  IO.puts("  Index #{String.pad_leading(to_string(idx), 3)}: #{status} (got #{value})")
end

# Summary
IO.puts("\n" <> String.duplicate("=", 70))
if length(failures) > 0 do
  IO.puts("❌ BUG REPRODUCED!")
  IO.puts("\nFailed cases:")
  for {expected, got} <- Enum.reverse(failures) do
    IO.puts("  Expected: #{expected}, Got: #{got}")
  end

  IO.puts("\nThis is the MLX scalar layout bug.")
  IO.puts("Workaround: Add 0 to force materialization before calling item()")
else
  IO.puts("✅ All tests passed - bug may be fixed!")
end

# Test 4: Demonstrate the workaround
IO.puts("\n📝 Test 4: Workaround (add 0 to fix layout)")
sliced = Nx.slice_along_axis(array, 951, 1, axis: 0)
scalar_buggy = Nx.squeeze(sliced, axes: [0])
scalar_fixed = Nx.add(scalar_buggy, 0)  # Forces materialization

value_buggy = Nx.to_number(scalar_buggy)
value_fixed = Nx.to_number(scalar_fixed)

IO.puts("Without workaround: #{value_buggy}")
IO.puts("With workaround:    #{value_fixed}")
IO.puts("Status: #{if value_fixed == 951, do: "✅ Workaround works!", else: "❌ Workaround failed"}")

# Test 5: Show memory pattern
IO.puts("\n📝 Test 5: Memory pattern analysis")
sliced = Nx.slice_along_axis(array, 951, 1, axis: 0)
scalar = Nx.squeeze(sliced, axes: [0])

# Extract the raw tensor reference
{_device, ref} = EMLX.Backend.from_nx(scalar)

# Get the blob (first 32 bytes)
blob = EMLX.to_blob({:cpu, ref}, 32)

IO.puts("Shape: #{inspect(EMLX.shape({:cpu, ref}))}")
IO.puts("Type: #{inspect(EMLX.scalar_type({:cpu, ref}))}")
IO.puts("First 32 bytes of memory:")
IO.puts("  #{inspect(blob)}")

# Parse as int32 values
int32_values = for <<val::little-signed-32 <- blob>>, do: val
IO.puts("Interpreted as int32 values: #{inspect(int32_values)}")

if length(Enum.uniq(int32_values)) == 1 and hd(int32_values) == 951 do
  IO.puts("❌ BUG CONFIRMED: Memory contains repeating copies of the value!")
  IO.puts("   item<int64>() reads across two copies, producing garbage.")
end

IO.puts("\n" <> String.duplicate("=", 70))
