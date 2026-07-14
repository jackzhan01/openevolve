# Six Analysis Dimensions

Every kernel profile report is ambiguous until you look at it through specific lenses. These six dimensions are the ones that consistently matter. Walk through all six; don't stop at the first finding. This Triton version keeps the diagnosis at the NCU level, but only recommends changes that can be expressed through Triton code, launch geometry, or Triton compiler configuration.

For each dimension this doc describes:

- **What you're answering**
- **Which metrics to read** (B200 / sm_100 names)
- **How to read them** (what's "normal", what's "bad")
- **Which `helpers/` to run**

---

## Dimension 1 — SM occupancy & launch geometry

**What:** is the grid large enough to fill the GPU? Is occupancy being limited by registers, shared memory, or block-size constraints?

**Metrics:**

```
launch__grid_size
launch__block_size
launch__grid_dim_x / _y / _z
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count         (148 on B200)
sm__maximum_warps_per_active_cycle_pct          (theoretical occupancy %)
sm__warps_active.avg.pct_of_peak_sustained_active  (achieved occupancy %)
```

**Reading:**

- **Waves / SM < 1**: grid is too small to fill the chip. On B200 with 148 SMs, if `launch__grid_size < 148 × blocks_per_SM`, some SMs sit idle the entire time. `Est. Speedup` from NCU often hits 50-90% here.
- **Waves / SM in [1, 2)**: you have a tail wave (partial last wave). Tail effect magnitude is roughly `(last_wave_blocks / wave_size) × (block_exec_time / total_kernel_time)`.
- **Waves / SM > 4**: grid is plenty big, scheduling averages out.
- **Theoretical occupancy 100% but achieved << 100%**: stalls are the bottleneck, not launch config. Move to Dimension 3.
- **Theoretical occupancy < 100% and `launch__occupancy_limit_registers` is the tightest**: reduce the per-program live state. In Triton, try smaller `BLOCK_*` tiles or accumulator tiles, fewer pipeline stages, and alternative `num_warps` values. Where supported by the installed Triton version, `maxnreg` can also be tested as a compiler configuration rather than assumed to help.
- **`launch__occupancy_limit_shared_mem`** the tightest: compiler-generated shared memory per program is too large. Reduce tile size or `num_stages`, or change the blocking/layout so less data is staged at once.

**Triton controls to inspect:** the launch-grid lambda, `BLOCK_SIZE` / `BLOCK_M` / `BLOCK_N` / `BLOCK_K`, `num_warps`, `num_stages`, and, when used, `num_ctas` or warp specialization. Treat these as a coupled configuration and benchmark several valid combinations; changing one can move the bottleneck from registers to shared memory or vice versa.

**Derived: wave math**

```python
blocks_per_sm = min(occ_limit_blocks, occ_limit_registers, occ_limit_shared_mem, occ_limit_warps)
wave_size = blocks_per_sm * num_sms
num_waves = (total_blocks + wave_size - 1) // wave_size
last_wave_blocks = total_blocks - (num_waves - 1) * wave_size
last_wave_utilization_pct = last_wave_blocks / wave_size * 100
```

**Helper:** `analyze_reports.py` prints all the key launch metrics under "Launch geometry" in the output txt.

---

## Dimension 2 — Thread-block balance (tail effect)

**What:** are blocks finishing at roughly the same time, or do a few outliers drag out the kernel?

**Metrics:**

There is no single "imbalance" metric — use these signals together:

```
# Per-SM active-cycle distribution (from MemoryWorkloadDistribution section)
# These show as "max XX% above average, min YY% below average" in details page
sm__cycles_active.{avg,max,min,sum}       # via action.source_info or the details page

# PM sampling (time series) — the shape matters, not just the mean
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_wait.avg
```

**Reading:**

- NCU's `details` page already says it: `"One or more SMs have a much lower number of active cycles than the average. Maximum instance value is X% above, while the minimum is Y% below."` X=51% / Y=95% (seen in practice) means severe imbalance.
- Render the PM-sampling time series (use `plot_timeline.py`). Possible shapes:
  - **Flat high → clean drop**: ideal. Well-balanced, good SM fill.
  - **Flat high → gradual tail**: tail effect. The tail's length is how much time a few slow blocks waste. Usually caused by variable-length inputs (e.g. seq_len varies per batch element).
  - **Flat low**: grid is too small (Dimension 1).
  - **Periodic waves / sawtooth**: pipeline bubbles — compute and memory alternate, nothing overlaps.

**Where imbalance typically comes from:**

1. **Variable-length per-CTA work**: when each CTA's iteration count depends on an input axis (e.g., per-element lengths driven by a prefix-sum / cumulative-length array), CTAs take very different times.
2. **Runtime-dependent loops or program-uniform branches**: some Triton program instances execute more loop iterations or take a longer scalar control-flow path than others. Per-element masks alone usually do not make the masked arithmetic free.
3. **Persistent scheduling without proper load balancing**: a Triton program loops over multiple logical tiles, but heavy tiles are assigned unevenly.

**Fix direction:** expose more parallel work in the launch grid, split variable-length work into fixed-size chunks, or dispatch separate Triton configurations for short and long cases. A persistent loop or an atomic work queue with `tl.atomic_add` is possible, but should be used only when static chunking cannot balance the work.

**Helper:** `plot_timeline.py` produces ASCII timeline plots. If you see a gradual slope on the right side, that's your tail effect.

Additionally, **always inspect the input distribution**. If per-CTA work is driven by an array like `per_element_lengths`:

```python
# Example: given a per-CTA work-count array, compute imbalance ratios
work_per_cta = [...]  # derive this from whatever drives the inner loop count
avg = sum(work_per_cta) / len(work_per_cta)
print(f"max/avg = {max(work_per_cta)/avg:.2f}x, max/min = {max(work_per_cta)/min(work_per_cta):.2f}x")
```

Ratios > 5x indicate significant potential for tail effect.

---

## Dimension 3 — Stall reason breakdown + per-line hotspots

**What:** when warps aren't issuing, what are they waiting for? Which source lines generate the most stalls?

**Aggregate stall metrics (SOL-adjacent, aggregated over the kernel):**

```
# Ratio per issued warp — how many of 16 active warps are in each stall state
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
```

**Per-line stall metrics (from `--set source`):**

```
smsp__pcsamp_sample_count                                  # total samples
smsp__pcsamp_warps_issue_stalled_long_scoreboard           # per-PC counts
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_selected                  # productive cycles
...
```

**Stall reasons you need to know:**

| Reason                                          | Meaning                               | Typical cause                                                      | Fix direction                                               |
| ----------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------- |
| `long_scoreboard`                               | waiting on long-latency dep           | global memory load hasn't returned                                 | coalesce, reuse, add ILP                                    |
| `short_scoreboard`                              | waiting on short-latency dep          | shared/local memory, or compute chain                              | add ILP, shorten dep chains                                 |
| `wait`                                          | waiting on fixed-latency pipe         | SFU / tensor-core output                                           | more independent ops in flight                              |
| `barrier`                                       | barrier / pipeline wait               | compiler-generated shared-memory or async-pipeline synchronization | reduce cross-warp communication; retune tile shape / stages |
| `membar`                                        | memory fence                          | atomics or compiler-generated ordering                             | reduce unnecessary atomics / synchronization                |
| `math_pipe_throttle`                            | FMA pipe saturated                    | legit compute-bound                                                | you're doing well, find other wins                          |
| `mio_throttle` / `lg_throttle` / `tex_throttle` | LSU/LD-ST/TEX pipe saturated          | too many load/store instructions                                   | improve block layout, reuse, and load/store granularity     |
| `not_selected`                                  | eligible but scheduler picked another | **good sign** — plenty of parallelism                              | ignore                                                      |
| `selected`                                      | actually issuing this cycle           | **productive**                                                     | ignore                                                      |
| `dispatch_stall`                                | dispatch unit busy                    | rare                                                               | usually minor                                               |
| `no_instruction`                                | warp has nothing to issue             | kernel prologue/epilogue                                           | usually minor                                               |
| `drain`                                         | warp finishing last few instructions  | end of kernel                                                      | ignore                                                      |
| `branch_resolving`                              | branch target calc in progress        | tight branches                                                     | usually minor                                               |

**Reading the ratio metric:** a value of e.g. `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio = R` means: for every cycle in which a warp issues, R other warps (on that SM sub-partition, per cycle-with-an-issue) were stalled on `long_scoreboard`. Higher values = more stalled time. On a kernel with 16 active warps per scheduler, R is bounded above by ~15 (all other warps stalled on this reason).

**Reading the `pcsamp` percentages:** normalize by `smsp__pcsamp_sample_count` to get "% of samples stalled on X". Rules of thumb:

- **`long_scoreboard` > 40% of samples**: kernel is memory-latency-bound. Check Dimension 6 (access patterns) next.
- **`short_scoreboard` > 30%**: check for long dep chains or heavy shared-memory use.
- **`barrier` > 20%**: too much synchronization, or warp divergence before a barrier.
- **`selected` < 10%**: very little actual issue — the whole kernel is stall-bound.

**Helper:** `extract_stall_hotspots.py` produces `stall_hotspots_<tag>.txt` which ranks source lines by total stall samples. This points at the offending load, barrier, or compute instruction in generated code; use source correlation to map it back to the Triton load/store, reduction, `tl.dot`, or loop that produced it.

---

## Dimension 4 — Tensor Core utilization

**What:** is the kernel using tensor cores at all? If yes, how well?

**Metrics:**

```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed      # overall TC activity
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active       # per active SM cycle
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed  # BF16/FP16 MMA
sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed  # INT MMA
sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed  # FP64 MMA
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg       # BF16×BF16→FP32 ops
```

**Reading:**

- **`sm__pipe_tensor_cycles_active = 0%`**: no tensor core usage at all. For matmul-ish kernels (attention, GEMM, conv), this is almost always a missed optimization.
- **`... = X%` but X << 50%**: tensor cores are being used but underutilized. Usually means data isn't arriving fast enough (Dimension 6) or tile sizes are wrong.
- **`... > 50%`** on B200: kernel is doing well on the Tensor-Core front. Focus elsewhere.

**Triton-specific note:** express matrix work through `tl.dot` (or `tl.dot_scaled` where appropriate), use supported input/accumulator dtypes, and choose block shapes that the backend can lower to tensor-core instructions. On newer backends, tensor descriptors, TMA-backed loads, or warp specialization may be available through Triton APIs, but they should be treated as version- and architecture-dependent configuration options.

**Fix direction:** if you see 0% and the workload is matrix-multiplication-shaped, redesign the inner computation around `tl.dot`, then tune `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages`. If tensor cores are active but low, first determine whether the limiting factor is tile shape, insufficient parallel work, or operand delivery.

---

## Dimension 5 — SM utilization timeline

**What:** how does SM utilization vary over the kernel's lifetime?

**Metrics (PM sampling, time-series):**

```
pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:sm__warps_active.avg.pct_of_peak_sustained_active
pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
```

**Reading (timeline shapes):**

- **Flat high, clean drop**: ideal.
- **Flat high, long tail**: tail effect (Dimension 2).
- **Flat low**: grid too small (Dimension 1) or severely stall-bound (Dimension 3).
- **Periodic sawtooth (compute ↕ memory)**: no compute-memory overlap. In Triton, inspect loop structure and tune compiler software pipelining through `num_stages` or staged `tl.range`; do not prescribe manual CUDA double-buffering unless the Triton API actually exposes the required mechanism.
- **Slow ramp up, flat middle, clean drop**: kernel has warmup work (prologue), then steady state. Usually fine.

**Helper:** `plot_timeline.py` — renders ASCII plots. Look at multiple series side-by-side (SM throughput + DRAM throughput + long_scoreboard stalls) to distinguish the shapes.

**Note:** PM sampling has ~2µs interval on B200. Very short kernels (< 20 µs) produce few samples — interpret with care.

---

## Dimension 6 — Memory access pattern & cache efficiency

**What:** are global loads coalesced? Are caches hit? Is DRAM actually busy?

**Metrics:**

```
# DRAM
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second                    # achieved BW

# L1 / L2 hit rates
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct

# Sectors per request (coalescing quality)
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum              # total sectors
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum             # total requests
# Compute sectors/request yourself — ideal is 4 (128B aligned load)

# Store efficiency
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio  # useful bytes/sector, max 32

# Register spill (local memory)
smsp__sass_inst_executed_op_local_ld.sum
smsp__sass_inst_executed_op_local_st.sum

# Global instruction counts
smsp__sass_inst_executed_op_global_ld.sum
smsp__sass_inst_executed_op_global_st.sum
smsp__sass_inst_executed_op_shared.sum                     # 0 if no shared memory
```

**Reading:**

- **`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` ≈ 80-100%**: genuinely DRAM-bandwidth-bound. Reduce bytes / amortize reads with shared memory.
- **`... << 10%` but kernel is slow**: _not_ bandwidth-bound. It's latency-bound (Dimension 3) or compute-bound (check `sm__throughput`).
- **`l1tex__t_sector_hit_rate.pct > 90%`**: good data locality, L1 is absorbing the reuse.
- **`lts__t_sector_hit_rate.pct < 50%`**: L2 is being blown through, reads fall to DRAM.
- **Sectors/request = 4.0 (ideal 128B fully-coalesced) to 5.0**: small coalescing issue but acceptable.
- **Sectors/request > 8.0**: serious non-coalesced access — big optimization opportunity.
- **`smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16`**: stores are using less than half of each 32B sector. In Triton, this often means a `tl.store` mask has only a few valid lanes or the output offsets are widely strided.
- **`smsp__sass_inst_executed_op_local_ld.sum > 0`**: **register spill** — very bad, local memory is DRAM-backed. In Triton, shrink tiles or accumulator state, lower `num_stages`, test different `num_warps`, recompute cheap values, or split the kernel.
- **`smsp__sass_inst_executed_op_shared.sum == 0`**: the generated kernel uses no shared-memory instructions. Fine for element-wise kernels. For reuse-heavy kernels, check whether a tiled `tl.dot`, transpose/block-layout change, tensor descriptor, or staged loop gives the compiler a reusable block structure; Triton does not provide the same direct user-managed shared-memory array interface as CUDA C++.

NCU's rule engine often reports the coalescing issue directly:

```
OPT   Est. Speedup: 10.8%
      The memory access pattern for global loads from L1TEX might not be optimal.
      On average, only 7.6 of the 32 bytes transmitted per sector are utilized...
```

**Fix directions (by pattern):**

- Strided access: change the `tl.arange` axes, pointer arithmetic, `tl.make_block_ptr(..., order=...)`, or program-ID mapping so the fastest-varying Triton offsets address contiguous elements.
- Poor L2 reuse despite coalesced loads: change program-ID ordering (for example, grouped `M` ordering with a `GROUP_SIZE_M`-style mapping) so nearby programs reuse the same operand tiles.
- AoS → SoA: restructure the data only when the Triton operator controls or can select the input/output layout.
- Sparse writes: remap output offsets so valid lanes write contiguous elements; for very small outputs, let one program produce several adjacent logical results or use a separate compact store kernel.
- Register spill: shrink block/accumulator shapes, lower `num_stages`, tune `num_warps`, reduce live intermediates, or split the kernel. Test `maxnreg` only where the installed Triton version supports it.
- Compiler cannot prove alignment/contiguity: use `tl.multiple_of`, `tl.max_contiguous`, or `tl.assume` only when the runtime contract truly guarantees the stated property.

---

## Cross-dimension synthesis

After walking through all six, write a one-line diagnosis combining them. Structure: name the top 3–4 signals, each tied to a specific dimension. For example:

> "The kernel runs at X% of peak SM throughput and Y% of peak DRAM (Dim 1, 6). Stall time is dominated by `<stall_reason>` (Z% of samples, Dim 3), concentrated on <N> source lines whose access pattern is <coalesced/uncoalesced/...> (Dim 6). The PM timeline shows <flat / tail / sawtooth> shape (Dim 2/5). Tensor cores <used / unused at W%> (Dim 4)."

Fill in the X/Y/Z/W/N values and <classifications> from your own report. That sentence is the deliverable. Everything else in the report is evidence backing it.
