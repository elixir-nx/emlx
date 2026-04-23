defmodule EMLX.GPUPool do
  @moduledoc """
  Single-slot semaphore for GPU access during the validation test suite.

  MLX streams are thread-local; only one test should hold an active GPU
  context at a time. Call `checkout/0` at the start of a test and
  `checkin/0` when done (or use the `on_exit` hook in `ValidationCase`).

  Waiting callers block on `checkout/0` with no timeout — the test runner's
  own module timeout is the effective deadline.
  """

  use GenServer

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, :queue.new(), name: __MODULE__)
  end

  @doc "Acquire the GPU slot, blocking until available."
  def checkout do
    GenServer.call(__MODULE__, :checkout, :infinity)
  end

  @doc "Release the GPU slot."
  def checkin do
    GenServer.cast(__MODULE__, :checkin)
  end

  # -- GenServer callbacks ---------------------------------------------------

  # State: `{locked?, pending_callers_queue}`

  @impl true
  def init(_queue) do
    {:ok, {false, :queue.new()}}
  end

  @impl true
  def handle_call(:checkout, _from, {false, queue}) do
    {:reply, :ok, {true, queue}}
  end

  def handle_call(:checkout, from, {true, queue}) do
    {:noreply, {true, :queue.in(from, queue)}}
  end

  @impl true
  def handle_cast(:checkin, {_, queue}) do
    case :queue.out(queue) do
      {{:value, next}, rest} ->
        GenServer.reply(next, :ok)
        {:noreply, {true, rest}}

      {:empty, _} ->
        {:noreply, {false, :queue.new()}}
    end
  end
end
