# emlx_axon/bench/validate_llama_dense.exs
#
# Dense Llama benchmark for standard Hugging Face safetensors checkpoints.
# Measures stock Bumblebee + EMLX, the EMLXAxon rewrite path, and the native
# EMLXAxon Llama dense path when available.
#
# TTFT is measured as the duration of a warmed one-token request. This keeps
# the measurement comparable across native and non-streaming serving paths.
#
# Run from emlx_axon:
#
#   EMLX_DENSE_LLAMA_MODEL=meta-llama/Llama-3.2-1B \
#   EMLX_DENSE_LLAMA_DEVICE=gpu \
#   EMLX_DENSE_LLAMA_TYPE=f16 \
#   EMLX_DENSE_LLAMA_SEQUENCE_LENGTH=128 \
#   EMLX_DENSE_LLAMA_MAX_NEW=32 \
#   EMLX_DENSE_LLAMA_RUNS=3 \
#   EMLX_DENSE_LLAMA_WARMUP_RUNS=1 \
#   EMLX_DENSE_LLAMA_STRICT_LENGTH=false \
#   EMLX_DENSE_LLAMA_JSON=/tmp/llama_dense_baseline.json \
#     mix run bench/validate_llama_dense.exs

defmodule LlamaDenseBenchmark do
  @moduledoc false

  def run do
    config = config()
    Nx.default_backend({EMLX.Backend, device: config.device})

    IO.puts("""
    === emlx_axon/bench/validate_llama_dense.exs ===
    purpose:          validate dense Llama paths
    model:            #{config.model_ref}
    model_source:     #{inspect(model_source_label(config.model_ref))}
    branch:           #{git(["branch", "--show-current"])}
    commit:           #{git(["rev-parse", "--short", "HEAD"])}
    machine:          #{machine()}
    device:           #{config.device}
    type:             #{config.type}
    sequence_length:  #{config.sequence_length}
    max_new_tokens:   #{config.max_new_tokens}
    warmup_runs:      #{config.warmup_runs}
    measured_runs:    #{config.runs}
    strict_length:    #{config.strict_length?}
    auth_token:       #{if config.auth_token, do: "present", else: "absent"}
    native_llama:     #{if llama_native_exports() == [], do: "unavailable", else: "available"}
    """)

    load_result = timed(fn -> load_model(config) end)

    {:ok, model_info, tokenizer, generation_config} =
      case load_result.value do
        {:ok, model_info, tokenizer, generation_config} ->
          {:ok, model_info, tokenizer, generation_config}

        {:error, reason} ->
          report_load_error(config, load_result.ms, reason)
      end

    spec = model_info.spec
    generation_config = configure_generation(generation_config, model_info, config)

    IO.puts(
      "loaded model in #{format_ms(load_result.ms)} ms " <>
        "spec=#{inspect(spec.__struct__)} arch=#{inspect(spec.architecture)}"
    )

    rewrite_result = timed(fn -> EMLXAxon.rewrite(model_info.model) end)
    rewrite_model_info = %{model_info | model: rewrite_result.value}
    IO.puts("rewrote model in #{format_ms(rewrite_result.ms)} ms")

    stock =
      benchmark_path(
        "stock",
        build_serving(model_info, tokenizer, generation_config, config),
        build_ttft_serving(model_info, tokenizer, generation_config, config),
        config
      )

    rewrite =
      benchmark_path(
        "rewrite",
        build_serving(rewrite_model_info, tokenizer, generation_config, config),
        build_ttft_serving(rewrite_model_info, tokenizer, generation_config, config),
        config
      )

    native_llama =
      benchmark_native_llama(model_info, tokenizer, generation_config, config)

    report = %{
      benchmark: "llama_dense_hf_safetensors",
      purpose: "validate dense Llama paths",
      branch: git(["branch", "--show-current"]),
      commit: git(["rev-parse", "--short", "HEAD"]),
      machine: machine(),
      model: config.model_ref,
      model_source: inspect(model_source_label(config.model_ref)),
      dense_safetensors?: dense_safetensors?(config.model_ref),
      device: config.device,
      type: config.type,
      precision: config.type,
      quantization_bits: nil,
      sequence_length: config.sequence_length,
      max_new_tokens: config.max_new_tokens,
      strict_length?: config.strict_length?,
      warmup_runs: config.warmup_runs,
      measured_runs: config.runs,
      prompt: config.prompt,
      ttft_definition: "warmed one-token request duration",
      load_ms: round_float(load_result.ms),
      rewrite_ms: round_float(rewrite_result.ms),
      paths: %{
        stock: stock,
        rewrite: rewrite,
        native_llama: native_llama
      },
      speedups: speedups(stock, rewrite, native_llama),
      verification:
        verification(%{stock: stock, rewrite: rewrite, native_llama: native_llama}, config)
    }

    print_summary(report)
    maybe_write_json(config.json_path, report)
    enforce_strict_length!(report, config)
    enforce_text_equality!(report)
  end

  defp config do
    %{
      model_ref: System.get_env("EMLX_DENSE_LLAMA_MODEL", "meta-llama/Llama-3.2-1B"),
      device: parse_device(System.get_env("EMLX_DENSE_LLAMA_DEVICE", "gpu")),
      type: parse_type(System.get_env("EMLX_DENSE_LLAMA_TYPE", "f16")),
      sequence_length: parse_pos_int("EMLX_DENSE_LLAMA_SEQUENCE_LENGTH", 128),
      max_new_tokens: parse_pos_int("EMLX_DENSE_LLAMA_MAX_NEW", 32),
      runs: parse_pos_int("EMLX_DENSE_LLAMA_RUNS", 3),
      warmup_runs: parse_non_neg_int("EMLX_DENSE_LLAMA_WARMUP_RUNS", 1),
      strict_length?: parse_bool("EMLX_DENSE_LLAMA_STRICT_LENGTH", false),
      auth_token:
        System.get_env("EMLX_DENSE_LLAMA_AUTH_TOKEN") ||
          System.get_env("HF_TOKEN") ||
          System.get_env("HUGGINGFACE_TOKEN"),
      prompt:
        System.get_env(
          "EMLX_DENSE_LLAMA_PROMPT",
          "Write one short sentence about native Elixir inference."
        ),
      json_path: System.get_env("EMLX_DENSE_LLAMA_JSON")
    }
  end

  defp load_model(config) do
    source = model_source(config)

    with {:ok, model_info} <-
           Bumblebee.load_model(source,
             type: config.type,
             backend: {EMLX.Backend, device: config.device}
           ),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(source),
         {:ok, generation_config} <- Bumblebee.load_generation_config(source) do
      generation_config =
        Bumblebee.configure(generation_config,
          max_new_tokens: config.max_new_tokens,
          strategy: %{type: :greedy_search}
        )

      {:ok, model_info, tokenizer, generation_config}
    end
  end

  defp configure_generation(generation_config, _model_info, %{strict_length?: false}) do
    generation_config
  end

  defp configure_generation(generation_config, model_info, %{strict_length?: true}) do
    impossible_eos = strict_eos_token_id(model_info)

    Bumblebee.configure(generation_config,
      eos_token_id: impossible_eos,
      forced_eos_token_id: nil
    )
  end

  defp strict_eos_token_id(%{spec: %{vocab_size: vocab_size}}) when is_integer(vocab_size) do
    vocab_size + 1_000
  end

  defp report_load_error(config, load_ms, reason) do
    message = Exception.message(RuntimeError.exception(inspect(reason)))

    IO.puts("""

    === Dense Llama Baseline Unavailable ===
    load_ms: #{format_ms(load_ms)}
    reason:  #{message}

    The requested model could not be loaded. If this is a gated Meta Llama
    checkpoint, authenticate with Hugging Face or pass EMLX_DENSE_LLAMA_MODEL
    as a local cached Llama 3.2 1B snapshot path.
    """)

    report = %{
      benchmark: "llama_dense_hf_safetensors_baseline",
      purpose: "baseline before native Llama accelerator work",
      status: "model_unavailable",
      branch: git(["branch", "--show-current"]),
      commit: git(["rev-parse", "--short", "HEAD"]),
      machine: machine(),
      model: config.model_ref,
      model_source: inspect(model_source_label(config.model_ref)),
      device: config.device,
      type: config.type,
      precision: config.type,
      sequence_length: config.sequence_length,
      max_new_tokens: config.max_new_tokens,
      strict_length?: config.strict_length?,
      warmup_runs: config.warmup_runs,
      measured_runs: config.runs,
      prompt: config.prompt,
      load_ms: round_float(load_ms),
      error: message,
      verification: %{
        native_llama_exports: llama_native_exports(),
        native_llama_available?: llama_native_exports() != []
      }
    }

    maybe_write_json(config.json_path, report)
    System.halt(1)
  end

  defp build_serving(model_info, tokenizer, generation_config, config) do
    generation_config =
      Bumblebee.configure(generation_config, max_new_tokens: config.max_new_tokens)

    Bumblebee.Text.generation(model_info, tokenizer, generation_config,
      compile: [batch_size: 1, sequence_length: config.sequence_length],
      defn_options: [compiler: EMLX, device: config.device],
      preallocate_params: true
    )
  end

  defp build_ttft_serving(model_info, tokenizer, generation_config, config) do
    build_serving(model_info, tokenizer, generation_config, %{config | max_new_tokens: 1})
  end

  defp benchmark_path(label, serving, ttft_serving, config) do
    IO.puts("\n==> #{label}: warmup")

    warmups =
      for i <- run_indices(config.warmup_runs) do
        result = run_once(serving, config.prompt, config.max_new_tokens)

        IO.puts(
          "  warmup #{i}: #{result.tokens} tokens / #{format_ms(result.ms)} ms\n" <>
            "    text: #{inspect(result.text)}"
        )

        result
      end

    IO.puts("==> #{label}: measured")

    runs =
      for i <- run_indices(config.runs) do
        result = run_once(serving, config.prompt, config.max_new_tokens)

        IO.puts(
          "  run #{i}: #{result.tokens} tokens / #{format_ms(result.ms)} ms = " <>
            "#{format_rate(result.tok_s)} tok/s finish=#{result.finish_reason}\n" <>
            "    text: #{inspect(result.text)}"
        )

        result
      end

    assert_equal_texts!(label, Enum.map(warmups ++ runs, & &1.text))

    path_report(warmups, runs)
    |> Map.put(:ttft_ms, benchmark_ttft(label, ttft_serving, config))
  end

  defp benchmark_native_llama(model_info, tokenizer, generation_config, config) do
    IO.puts("\n==> native_llama: build state")

    if config.device != :gpu do
      skipped_path("native Llama benchmark requires EMLX_DENSE_LLAMA_DEVICE=gpu")
      |> Map.put(:native_used?, false)
    else
      case EMLXAxon.Llama.DenseLoader.from_model_info(model_info,
             generation_config: generation_config
           ) do
        {:ok, state} ->
          try do
            serving =
              EMLXAxon.TextGeneration.serving(tokenizer, state,
                max_new_tokens: config.max_new_tokens,
                max_len: config.sequence_length + config.max_new_tokens,
                sampler: :greedy,
                host_sync: native_host_sync(config)
              )

            ttft_serving =
              EMLXAxon.TextGeneration.serving(tokenizer, state,
                max_new_tokens: 1,
                max_len: config.sequence_length + 1,
                sampler: :greedy,
                host_sync: :end
              )

            benchmark_path("native_llama", serving, ttft_serving, config)
            |> Map.put(:native_used?, true)
          rescue
            exception ->
              skipped_path(Exception.message(exception))
              |> Map.put(:native_used?, false)
          end

        {:error, reason} ->
          skipped_path(inspect(reason))
          |> Map.put(:native_used?, false)
      end
    end
  end

  defp native_host_sync(%{strict_length?: true}), do: :end
  defp native_host_sync(config), do: {:chunk, min(config.max_new_tokens, 31)}

  defp run_once(serving, prompt, max_new_tokens) do
    result = timed(fn -> Nx.Serving.run(serving, prompt) end)
    {text, tokens, finish_reason} = extract_text_tokens_finish(result.value, max_new_tokens)
    public_run(result.ms, tokens, finish_reason, text)
  end

  defp benchmark_ttft(label, serving, config) do
    warmup = run_once(serving, config.prompt, 1)
    IO.puts("  #{label} TTFT warmup: #{format_ms(warmup.ms)} ms\n    text: #{inspect(warmup.text)}")

    {values, texts} =
      Enum.map(run_indices(config.runs), fn i ->
        result = run_once(serving, config.prompt, 1)

        IO.puts(
          "  #{label} TTFT #{i}: #{format_ms(result.ms)} ms\n" <>
            "    text: #{inspect(result.text)}"
        )

        {result.ms, result.text}
      end)
      |> Enum.unzip()

    assert_equal_texts!("#{label} TTFT", [warmup.text | texts])
    stats(values)
  end

  defp extract_text_tokens_finish(
         %{results: [%{text: text, token_summary: summary}]},
         max_new_tokens
       ) do
    {text, summary.output, inferred_finish_reason(summary.output, max_new_tokens)}
  end

  defp extract_text_tokens_finish(
         %{
           results: [%{generated_text: text, num_tokens: tokens}],
           finish_reason: finish_reason
         },
         _max_new_tokens
       ) do
    {text, tokens, finish_reason}
  end

  defp extract_text_tokens_finish(
         %{results: [%{generated_text: text, num_tokens: tokens}]},
         max_new_tokens
       ) do
    {text, tokens, inferred_finish_reason(tokens, max_new_tokens)}
  end

  defp inferred_finish_reason(tokens, max_new_tokens) when tokens >= max_new_tokens, do: :length
  defp inferred_finish_reason(_tokens, _max_new_tokens), do: :stop

  defp public_run(ms, tokens, finish_reason, text) do
    %{
      ms: ms,
      tokens: tokens,
      tok_s: rate(tokens, ms),
      finish_reason: finish_reason,
      text: text,
      preview: text |> String.replace("\n", " ") |> String.slice(0, 80)
    }
  end

  defp path_report(warmups, runs) do
    %{
      warmup: summarize(warmups),
      measured: summarize(runs),
      runs: Enum.map(runs, &json_run/1)
    }
  end

  defp summarize([]), do: stats([])

  defp summarize(runs) do
    %{
      duration_ms: stats(Enum.map(runs, & &1.ms)),
      tokens: stats(Enum.map(runs, & &1.tokens)),
      tokens_per_sec: stats(Enum.map(runs, & &1.tok_s)),
      finish_reasons: Enum.frequencies(Enum.map(runs, &to_string(&1.finish_reason)))
    }
  end

  defp stats([]), do: %{count: 0}

  defp stats(values) do
    sorted = Enum.sort(values)
    count = length(sorted)

    %{
      count: count,
      min: round_float(List.first(sorted)),
      p50: round_float(percentile(sorted, 0.50)),
      p95: round_float(percentile(sorted, 0.95)),
      max: round_float(List.last(sorted)),
      mean: round_float(Enum.sum(values) / count)
    }
  end

  defp skipped_path(reason) do
    %{
      skipped: true,
      reason: reason,
      ttft_ms: stats([]),
      warmup: stats([]),
      measured: %{
        duration_ms: stats([]),
        tokens: stats([]),
        tokens_per_sec: stats([]),
        finish_reasons: %{"skipped" => 1}
      },
      runs: []
    }
  end

  defp speedups(stock, rewrite, native_llama) do
    stock_tps = get_in(stock, [:measured, :tokens_per_sec, :p50]) || 0.0
    rewrite_tps = get_in(rewrite, [:measured, :tokens_per_sec, :p50]) || 0.0
    native_tps = get_in(native_llama, [:measured, :tokens_per_sec, :p50]) || 0.0

    %{
      rewrite_vs_stock: ratio(rewrite_tps, stock_tps),
      native_llama_vs_stock: ratio(native_tps, stock_tps),
      native_llama_vs_rewrite: ratio(native_tps, rewrite_tps)
    }
  end

  defp verification(paths, config) do
    measured_token_counts =
      Map.new(paths, fn {path, path_report} ->
        {path, Enum.map(path_report[:runs] || [], & &1.tokens)}
      end)

    measured_texts =
      Map.new(paths, fn {path, path_report} ->
        {path, Enum.map(path_report[:runs] || [], & &1.text)}
      end)

    finish_reasons =
      Map.new(paths, fn {path, path_report} ->
        {path, get_in(path_report, [:measured, :finish_reasons]) || %{}}
      end)

    %{
      native_llama_exports: llama_native_exports(),
      native_llama_available?: llama_native_exports() != [],
      strict_length?: config.strict_length?,
      expected_tokens_per_run: config.max_new_tokens,
      measured_token_counts: measured_token_counts,
      equal_measured_token_count?: equal_measured_token_count?(measured_token_counts, config),
      measured_texts: measured_texts,
      equal_measured_text?: equal_measured_text?(measured_texts),
      finish_reasons: finish_reasons,
      comparable_finish_reasons?: comparable_finish_reasons?(finish_reasons, config)
    }
  end

  defp equal_measured_text?(measured_texts) do
    texts =
      measured_texts
      |> Enum.reject(fn {_path, texts} -> texts == [] end)
      |> Enum.flat_map(fn {_path, texts} -> texts end)

    case texts do
      [] -> true
      _ -> MapSet.size(MapSet.new(texts)) == 1
    end
  end

  defp assert_equal_texts!(_label, []), do: :ok

  defp assert_equal_texts!(label, texts) do
    unique = Enum.uniq(texts)

    unless length(unique) == 1 do
      raise """
      #{label} generated unequal texts across calls:
      #{Enum.map_join(Enum.with_index(texts, 1), "\n", fn {text, i} -> "  #{i}: #{inspect(text)}" end)}
      """
    end
  end

  defp enforce_text_equality!(report) do
    verification = report.verification

    unless verification.equal_measured_text? do
      raise """
      Llama dense benchmark generated unequal measured texts across paths:
      #{Enum.map_join(verification.measured_texts, "\n", fn {path, texts} ->
        "  #{path}: #{inspect(texts)}"
      end)}
      """
    end
  end

  defp equal_measured_token_count?(measured_token_counts, config) do
    measured_token_counts
    |> Map.values()
    |> Enum.all?(&(&1 == List.duplicate(config.max_new_tokens, config.runs)))
  end

  defp comparable_finish_reasons?(finish_reasons, %{strict_length?: true, runs: runs}) do
    Enum.all?(finish_reasons, fn {_path, reasons} -> reasons == %{"length" => runs} end)
  end

  defp comparable_finish_reasons?(finish_reasons, _config) do
    finish_reasons
    |> Map.values()
    |> Enum.uniq()
    |> length()
    |> Kernel.==(1)
  end

  defp enforce_strict_length!(_report, %{strict_length?: false}), do: :ok

  defp enforce_strict_length!(report, %{strict_length?: true}) do
    verification = report.verification

    unless verification.equal_measured_token_count? and verification.comparable_finish_reasons? do
      raise """
      strict Llama dense benchmark failed validation:
        measured_token_counts=#{inspect(verification.measured_token_counts)}
        finish_reasons=#{inspect(verification.finish_reasons)}
      """
    end
  end

  defp print_summary(report) do
    IO.puts("\n=== Dense Llama Baseline Summary ===")

    Enum.each([:stock, :rewrite, :native_llama], fn path ->
      path_report = report.paths[path]

      if path_report[:skipped] do
        IO.puts("#{path}: skipped (#{path_report.reason})")
      else
        measured = path_report.measured

        IO.puts(
          "#{path}: duration_p50=#{measured.duration_ms.p50} ms " <>
            "ttft_p50=#{path_report.ttft_ms.p50} ms " <>
            "tok/s_p50=#{measured.tokens_per_sec.p50} " <>
            "tokens_p50=#{measured.tokens.p50} " <>
            "finish=#{inspect(measured.finish_reasons)}"
        )
      end
    end)

    IO.puts("speedups: #{inspect(report.speedups)}")
  end

  defp maybe_write_json(nil, _report), do: :ok

  defp maybe_write_json(path, report) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(report, pretty: true))
    IO.puts("json_output: #{path}")
  end

  defp model_source(%{model_ref: path_or_repo, auth_token: auth_token}) do
    expanded = Path.expand(path_or_repo)

    cond do
      File.dir?(expanded) -> {:local, expanded}
      auth_token -> {:hf, path_or_repo, auth_token: auth_token}
      true -> {:hf, path_or_repo}
    end
  end

  defp model_source_label(path_or_repo) do
    expanded = Path.expand(path_or_repo)

    if File.dir?(expanded), do: {:local, expanded}, else: {:hf, path_or_repo}
  end

  defp dense_safetensors?(model_ref) do
    expanded = Path.expand(model_ref)

    File.dir?(expanded) and
      expanded
      |> File.ls!()
      |> Enum.any?(&String.ends_with?(&1, ".safetensors"))
  rescue
    _ -> false
  end

  defp llama_native_exports do
    EMLXAxon.Llama.Native.module_info(:exports)
    |> Enum.map(fn {name, arity} -> "#{name}/#{arity}" end)
    |> Enum.reject(&(String.starts_with?(&1, "__") or String.starts_with?(&1, "module_info/")))
    |> Enum.sort()
  end

  defp parse_device("gpu"), do: :gpu
  defp parse_device("cpu"), do: :cpu

  defp parse_device(other) do
    raise ArgumentError,
          "expected EMLX_DENSE_LLAMA_DEVICE to be gpu or cpu, got: #{inspect(other)}"
  end

  defp parse_type("f16"), do: :f16
  defp parse_type("bf16"), do: :bf16
  defp parse_type("f32"), do: :f32

  defp parse_type(other) do
    raise ArgumentError,
          "expected EMLX_DENSE_LLAMA_TYPE to be f16, bf16, or f32, got: #{inspect(other)}"
  end

  defp parse_pos_int(env, default) do
    value = System.get_env(env, Integer.to_string(default)) |> String.to_integer()
    if value > 0, do: value, else: raise(ArgumentError, "#{env} must be positive")
  end

  defp parse_non_neg_int(env, default) do
    value = System.get_env(env, Integer.to_string(default)) |> String.to_integer()
    if value >= 0, do: value, else: raise(ArgumentError, "#{env} must be non-negative")
  end

  defp parse_bool(env, default) do
    case System.get_env(env) do
      nil -> default
      "true" -> true
      "false" -> false
      other -> raise ArgumentError, "#{env} must be true or false, got: #{inspect(other)}"
    end
  end

  defp run_indices(0), do: []
  defp run_indices(n), do: 1..n

  defp timed(fun) do
    start = System.monotonic_time(:native)
    value = fun.()
    stop = System.monotonic_time(:native)
    %{value: value, ms: System.convert_time_unit(stop - start, :native, :microsecond) / 1000.0}
  end

  defp rate(tokens, ms) when ms > 0, do: tokens / ms * 1000.0
  defp rate(_tokens, _ms), do: 0.0

  defp ratio(_num, den) when den == 0.0, do: nil
  defp ratio(num, den), do: round_float(num / den)

  defp percentile(sorted, pct) do
    idx = max(0, min(length(sorted) - 1, ceil(length(sorted) * pct) - 1))
    Enum.at(sorted, idx)
  end

  defp round_float(value) when is_integer(value), do: value
  defp round_float(value) when is_float(value), do: Float.round(value, 3)

  defp format_ms(ms), do: :erlang.float_to_binary(ms / 1.0, decimals: 1)
  defp format_rate(rate), do: :erlang.float_to_binary(rate / 1.0, decimals: 1)

  defp json_run(run) do
    %{
      duration_ms: round_float(run.ms),
      tokens: run.tokens,
      tokens_per_sec: round_float(run.tok_s),
      finish_reason: to_string(run.finish_reason),
      text: run.text,
      preview: run.preview
    }
  end

  defp git(args) do
    case System.cmd("git", args, stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      {_output, _status} -> nil
    end
  end

  defp machine do
    cpu = sysctl("machdep.cpu.brand_string")
    mem = sysctl("hw.memsize") |> parse_mem()

    [cpu, mem]
    |> Enum.reject(&(&1 in [nil, ""]))
    |> Enum.join(", ")
  end

  defp sysctl(key) do
    case System.cmd("sysctl", ["-n", key], stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      {_output, _status} -> nil
    end
  end

  defp parse_mem(nil), do: nil

  defp parse_mem(bytes) do
    gb =
      bytes
      |> String.to_integer()
      |> Kernel./(1024 ** 3)

    "#{:erlang.float_to_binary(gb, decimals: 1)} GiB RAM"
  rescue
    _ -> nil
  end
end

LlamaDenseBenchmark.run()
