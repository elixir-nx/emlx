Mix.install([
  {:emlx, path: "/Users/valente/coding/emlx"},
  {:nx, override: true}
])

# Minimal reproduction of MLX scalar layout bug - RAW VERSION
# This version calls EMLX.item() directly to bypass the workaround in to_number()

IO.puts("=" |> String.duplicate(70))
IO.puts("MLX Scalar Layout Bug - RAW (bypassing workaround)")
IO.puts("=" |> String.duplicate(70))

Nx.global_default_backend(EMLX.Backend)

# Create test array
array = Nx.iota({1000}, type: :s32, backend: EMLX.Backend)

# Test different indices
test_indices = [0, 1, 100, 500, 900, 950, 951, 998, 999]

IO.puts("\nTesting slice → squeeze → item() WITHOUT workaround:\n")

results =
  for idx <- test_indices do
    # Perform slice and squeeze
    sliced = Nx.slice_along_axis(array, idx, 1, axis: 0)
    scalar = Nx.squeeze(sliced, axes: [0])

    # Call EMLX.item() directly (bypassing the workaround in to_number)
    {_device, ref} = EMLX.Backend.from_nx(scalar)
    value = EMLX.item({:cpu, ref})

    {idx, value}
  end

for {idx, value} <- results do
  status = if value == idx, do: "✓", else: "✗ FAIL"
  IO.puts("  Index #{String.pad_leading(to_string(idx), 3)}: #{String.pad_trailing(status, 8)} Expected: #{idx}, Got: #{value}")
end

failures = Enum.filter(results, fn {expected, got} -> expected != got end)

# Summary
IO.puts("\n" <> String.duplicate("=", 70))

if length(failures) > 0 do
  IO.puts("❌ BUG REPRODUCED! Found #{length(failures)} failures.\n")

  # Analyze the pattern
  IO.puts("Failed cases:")
  for {expected, got} <- Enum.reverse(failures) do
    # Try to explain the value
    as_bytes = <<got::little-signed-64>>
    <<low::little-signed-32, high::little-signed-32>> = as_bytes

    IO.puts("")
    IO.puts("  Index #{expected}:")
    IO.puts("    Expected: #{expected}")
    IO.puts("    Got:      #{got}")
    IO.puts("    As bytes: #{inspect(as_bytes)}")
    IO.puts("    Low 32b:  #{low}")
    IO.puts("    High 32b: #{high}")

    if low == expected and high == expected do
      IO.puts("    ❌ PATTERN: item<int64>() read TWO copies of the int32 value!")
    end
  end

  IO.puts("\nRoot cause:")
  IO.puts("  - MLX creates scalar tensors with repeating memory pattern")
  IO.puts("  - item<int64>() reads 8 bytes from int32 scalar buffer")
  IO.puts("  - Reads two consecutive int32 values: [951, 951] → int64")
  IO.puts("  - Result: 0x00000003B7000003B7 = 3,869,765,534,647")

else
  IO.puts("✅ All tests passed!")
  IO.puts("The bug may have been fixed in MLX.")
end

# Show memory pattern for a specific case
IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("Memory Layout Analysis for index 951:\n")

sliced = Nx.slice_along_axis(array, 951, 1, axis: 0)
scalar = Nx.squeeze(sliced, axes: [0])
{_device, ref} = EMLX.Backend.from_nx(scalar)

shape = EMLX.shape({:cpu, ref})
type = EMLX.scalar_type({:cpu, ref})
blob = EMLX.to_blob({:cpu, ref}, 64)

IO.puts("Shape: #{inspect(shape)}")
IO.puts("Type:  #{inspect(type)}")
IO.puts("Blob (64 bytes):")

# Show as hex
hex_string = Base.encode16(blob, case: :lower)
Regex.scan(~r/.{1,32}/, hex_string)
|> Enum.each(fn [chunk] ->
  IO.puts("  #{chunk}")
end)

# Parse as int32
int32_values = for <<val::little-signed-32 <- blob>>, do: val
IO.puts("\nAs int32 array: #{inspect(int32_values)}")

unique_values = Enum.uniq(int32_values)
if length(unique_values) == 1 do
  IO.puts("\n❌ All #{length(int32_values)} int32 values are identical: #{hd(unique_values)}")
  IO.puts("   This confirms the memory layout bug!")
  IO.puts("   A scalar should have exactly 4 bytes, not a repeating pattern.")
end

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("\nTo see the bug fixed, check test_mlx_scalar_bug.exs which uses")
IO.puts("the workaround implemented in EMLX.Backend.to_number/1")
IO.puts("=" |> String.duplicate(70))
