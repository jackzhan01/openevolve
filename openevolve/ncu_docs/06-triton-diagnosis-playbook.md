# Diagnosis Playbook — Pattern → Cause → Fix

For each observed NCU signal, what does it typically mean, and what's the first Triton-level fix to try? This version keeps the NCU signals, but removes CUDA-C++-only prescriptions and translates each actionable diagnosis into Triton code, launch-grid, or compiler-configuration changes.

Read this after you've gathered the metrics (via [`05-analysis-dimensions-triton.md`](05-analysis-dimensions-triton.md)) — here you translate metrics into diagnoses and fix directions.

---

## How to use this doc

For each _observation_ below, read:

- **Signals** — what specific metric values flag this pattern.
- **Why** — the underlying cause.
- **First-line fix** — the cheapest change to try.
- **Deeper fixes** — when first-line isn't enough.
- **Exceptions** — kernel types where this pattern is actually _expected_ and should be left alone.

Most kernels will match 2-4 patterns simultaneously. **Rank them by magnitude** using NCU's `Est. Speedup: X%` fields (from `--page details`) and the stall-percentage breakdown. Fix the biggest one first. For Triton, turn each proposed fix into a small set of legal configurations and benchmark them; do not assume that a lower register count or larger tile is automatically faster.

---

## Pattern A — Small grid / SM idle

**Signals:**

- `launch__waves_per_multiprocessor < 0.5`
- `launch__grid_size < device__attribute_multiprocessor_count` (e.g., 64 blocks on a 148-SM B200)
- NCU rule: _"The grid for this launch is configured to execute only N blocks, which is less than the M multiprocessors used."_ with `Est. Speedup: 50-90%`

**Why:** each Triton program instance becomes one CTA; with fewer program instances than SMs, some SMs are completely idle throughout the kernel.

**First-line fix:** increase the Triton launch grid by exposing another independent program dimension:

- Add a split dimension for reductions or attention (for example, split-K or split over the reduction axis), followed by a small combine step.
- Split across heads, channels, rows, or output tiles when those units are independent.
- Reduce the amount of work assigned to one `tl.program_id` by using smaller tiles, provided the extra programs do not create excessive overhead or redundant reads.

**Deeper fixes:**

- **Persistent Triton kernel**: launch a fixed number of programs and let each program loop over multiple logical tile IDs. Use static striding first; use an atomic work queue with `tl.atomic_add` only when tile costs are genuinely irregular.
- **Fuse adjacent pointwise or reduction work** when fusion removes launch overhead or an otherwise tiny final stage, while checking that the larger fused kernel does not lose occupancy through register pressure.

**Exceptions:**

- LLM decode (batch=1, query_len=1) is fundamentally small. Splitting over the KV/reduction dimension is the usual Triton-level mitigation.
- Final reduction stages are naturally small; fuse them into the producing Triton kernel or keep them separate if fusion causes excessive register pressure.

**Triton levers:** launch-grid lambda, `BLOCK_*`, `num_warps`, and whether the algorithm uses one or multiple reduction stages.

---

## Pattern B — Tail effect (variable-length inputs)

**Signals:**

- Multi-workload: `max_seq_len / avg_seq_len > 3` in input distribution.
- Per-SM active cycles span 5-100× between slowest and fastest SM (from `--page details` distribution).
- PM timeline shape: long gradual tail at the end (visible via `plot_timeline.py`).
- `launch__waves_per_multiprocessor > 1.05` with partial last wave.
- NCU rule: `"partial wave may account for up to X% of the total runtime"`.

**Why:** each Triton program instance executes a variable-size loop or handles a variable amount of valid work. A few heavy programs keep running after the others finish. A mask does not necessarily remove the cost of arithmetic executed for masked lanes.

**First-line fix (cheap):**

- **Split long items across more program instances**: add a `split_factor` grid dimension; each program handles a fixed-size chunk, and a small Triton combine kernel merges partials when needed.
- **Use shape/length buckets in the Python dispatch wrapper** so short and long inputs select different `BLOCK_*`, `num_warps`, or kernel variants rather than sharing one badly imbalanced configuration.

**Deeper fixes:**

- **Chunkwise Triton kernel**: break each sequence into fixed-size chunks, process chunks in parallel, then stitch them with a small recurrence or reduction.
- **Classify-and-dispatch**: short sequences use one program per sequence; long sequences use a chunked multi-program path.
- **Persistent dynamic scheduling**: use a program loop or `tl.atomic_add` work queue only when static chunking cannot predict the work distribution.

**Exceptions:**

- Short kernels (< 10 µs) where partial-wave cost is absolute-small.
- Workloads already bucketed into nearly equal-length groups.

**Triton levers:** extra grid dimensions, fixed-size chunking, shape-specialized dispatch, and persistent program loops.

---

## Pattern C — Uncoalesced global loads

**Signals:**

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum > 5` (ideal is 4).
- NCU rule: _"uncoalesced global accesses resulting in N excessive sectors (X% of the total)"_.
- NCU rule: _"On average, only Y of the 32 bytes transmitted per sector are utilized"_.
- Primary stall reason on the offending load line is `long_scoreboard`.

**Why:** adjacent lanes generated by a Triton block access non-contiguous addresses, so hardware fetches extra sectors that only a few lanes use.

**First-line fix:** rework the Triton offset and block mapping:

- Make the fastest-varying `tl.arange` dimension correspond to the contiguous tensor dimension.
- For 2-D tiles, change pointer arithmetic or `tl.make_block_ptr(..., order=...)` so adjacent lanes load adjacent addresses.
- Change the `tl.program_id` mapping when the current tile orientation forces strided per-lane access.

**Deeper fixes:**

- Load a contiguous block and use `tl.reshape`, `tl.trans`, or a different block layout for the computation instead of issuing strided global loads directly.
- Increase contiguous load/store granularity by using a larger valid contiguous block and by providing truthful `tl.multiple_of` / `tl.max_contiguous` facts when the compiler cannot infer them.
- Change AoS to SoA only when the Triton operator controls the data layout.

**Exceptions:**

- Gather/scatter by random index (sparse matmul, embedding lookup) is fundamentally irregular. Sorting or bucketing indices for locality requires an operator-level preprocessing or dispatch change.
- Graph / tree traversal.

**Triton levers:** offset construction, block-pointer order, block shape, program-ID mapping, and valid alignment/contiguity hints.

---

## Pattern D — Sparse writes (low store efficiency)

**Signals:**

- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16` (ideal is 32).
- `l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct` lower than expected.
- A `tl.store` has a mask for which only a few lanes are valid, or output offsets are widely strided.

**Why:** only a subset of lanes write useful values, so sectors are only partially utilized.

**First-line fix:** reorganize output ownership so each program writes a contiguous output tile:

- Produce reduced values with Triton block operations such as `tl.sum`, `tl.max`, or `tl.argmax`, then map the resulting values to contiguous output offsets.
- If each logical item emits very few values, let one Triton program process multiple adjacent items so their stores combine into fuller sectors.

**Deeper fixes:**

- Separate computation from compaction: write dense partial results, then use a small second Triton kernel to compact or scatter them.
- Change the output layout when the operator contract permits it.

**Exceptions:**

- Histogram / scatter outputs are inherently sparse; optimize contention and aggregation instead (Pattern G).

---

## Pattern E — Latency-bound (long-scoreboard-dominated)

**Signals:**

- `smsp__pcsamp_warps_issue_stalled_long_scoreboard / smsp__pcsamp_sample_count > 0.40`.
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio > 3`.
- `dram__bytes_read.sum.pct_of_peak_sustained_elapsed < 10%` (→ not DRAM-bandwidth-bound).
- Hotspot lines are global loads (check `stall_hotspots_<tag>.txt`).

**Why:** a Triton program issues a load, then reaches dependent work before enough other independent loads or warps can hide the latency. This is often combined with low occupancy, narrow tiles, or a serial loop.

**First-line fix:** increase independent memory work without exploding register pressure:

- Use `tl.static_range` or a suitably structured `tl.range` loop so several independent loads can be in flight before their values are consumed.
- Test alternative `num_warps` and block shapes; more resident warps or more independent elements per program can hide latency, but larger tiles can also reduce occupancy.
- Tune `num_stages` for pipelined loops. Where supported, use Triton tensor descriptors or TMA-backed loads through the Triton API.

**Deeper fixes:**

- Reorder the loop so tile `N+1` is loaded while tile `N` is being computed, using Triton's compiler-managed staging (`num_stages` or staged `tl.range`).
- Increase reuse inside a program so repeatedly used operands stay in registers/cache or are compiler-staged once.
- Split a serial dependency chain into multiple independent partial accumulations and combine them at the end.

**Exceptions:**

- Pointer chasing / graph traversal — the dependency chain is fundamental.

**Triton levers:** `BLOCK_*`, `num_warps`, `num_stages`, loop form, independent accumulators, and supported descriptor APIs.

---

## Pattern F — Compute-bound but not on tensor cores

**Signals:**

- `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active > 50%`.
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed = 0%`.
- Workload is matmul-ish (GEMM, attention, conv).

**Why:** the Triton expression lowers to scalar/vector ALU operations instead of MMA instructions, often because the computation is written as explicit multiply-adds, the dtype is unsupported for tensor cores, or the block shapes do not match a viable MMA lowering.

**First-line fix:** express the matrix operation with `tl.dot` (or `tl.dot_scaled` where appropriate), use supported input/accumulator dtypes, and tune `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and `num_warps`.

**Deeper fixes:**

- Restructure the data/block layout so the reduction dimension and operand tiles satisfy the backend's MMA requirements.
- Tune `num_stages` to feed the tensor cores; where supported, test tensor descriptors, TMA-backed loads, or warp specialization through Triton APIs.
- Split non-MMA epilogue work if it inflates register pressure enough to suppress tensor-core throughput.

**Exceptions:**

- Non-matrix workloads (reduction, sort, element-wise) — tensor cores do not help.
- Small matrices where MMA tile overhead and padding dominate.

**Triton levers:** `tl.dot`, dtype, block shape, `num_warps`, `num_stages`, and optional descriptor/warp-specialized configurations.

---

## Pattern G — Atomics contention

**Signals:**

- `long_scoreboard` samples concentrate on `ATOM` / `RED` SASS instructions.
- `lts__t_sectors_op_atom.sum` or `lts__t_sectors_op_red.sum` is large.
- L2 throughput is high but compute throughput is low.

**Why:** many Triton program instances or lanes atomically update a small set of locations, causing serialization.

**First-line fix:** aggregate before the atomic:

- Use `tl.sum`, `tl.max`, or another per-program reduction so each program performs one atomic update per output value instead of one atomic per element.
- Increase the amount of independent reduction work owned by one program when that reduces atomics without creating excessive register pressure.

**Deeper fixes:**

- Use a two-kernel reduction: first write one partial per program/tile, then reduce the partials in a second Triton kernel.
- Bucket updates into multiple temporary outputs and merge them later.

**Exceptions:**

- Communication or truly online-update algorithms where atomics are fundamental.

**Triton levers:** per-program reduction shape, number of partial buffers, and `tl.atomic_*` granularity.

---

## Pattern H — Shared-memory bank conflicts

**Signals:**

- `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed` is high for shared-memory operations.
- `short_scoreboard` stalls concentrate on generated shared-memory load/store lines.
- The issue changes strongly with tile shape, transpose orientation, `num_warps`, or `num_stages`.

**Why:** compiler-generated shared-memory staging uses a layout that maps many simultaneous lane accesses to the same banks.

**First-line fix:** change the Triton configuration that determines the generated layout:

- Test different `BLOCK_*` shapes, especially the dimension being transposed or reduced.
- Change `tl.make_block_ptr(..., order=...)`, transpose orientation, or program layout so shared staging is generated with a different access pattern.
- Test different `num_warps`; lane-to-element mapping changes can remove a conflict.

**Deeper fixes:**

- Use a different algorithmic tiling that avoids the problematic transpose/staging path.
- On Triton versions exposing advanced layout or swizzle controls, test them explicitly; otherwise stay within source-level tile and layout changes exposed by the Triton API.

**Exceptions:**

- Broadcast reads are conflict-free.
- Low shared-memory volume where the measured conflict contributes little runtime.
- If all legal Triton layouts generate the same conflict, this may require a compiler/backend change rather than another source-level variant.

**Triton levers:** tile shape, block-pointer order, transpose strategy, `num_warps`, and version-specific layout APIs.

---

## Pattern I — Synchronization overhead

**Signals:**

- `smsp__pcsamp_warps_issue_stalled_barrier` > 20% of samples.
- Source/SASS hotspot is a barrier or async-pipeline wait generated from a reduction, staged loop, tensor-core path, or explicit `tl.debug_barrier`.

**Why:** warps in one Triton program must rendezvous around compiler-generated shared-memory or pipeline phases. Imbalance between warps or too many stages amplifies the wait.

**First-line fix:** reduce the amount of cross-warp coordination:

- Use a smaller tile or fewer warps when the operation does not need wide cross-warp participation.
- Reduce or reorganize multi-phase reductions; split independent phases into separate kernels when the synchronization cost exceeds the fusion benefit.
- Remove unnecessary `tl.debug_barrier` calls and test lower `num_stages` when pipeline barriers dominate.

**Deeper fixes:**

- Use a warp-local or hierarchical algorithm where each program computes independent partials and a second kernel combines them.
- Where supported, test Triton warp specialization so producer/consumer roles overlap rather than rendezvous globally.

**Triton levers:** tile size, `num_warps`, reduction structure, `num_stages`, fusion boundary, and optional warp specialization.

---

## Pattern J — Low achieved vs theoretical occupancy

**Signals:**

- `sm__maximum_warps_per_active_cycle_pct > 50` but `sm__warps_active.avg.pct_of_peak_sustained_active << 50`.
- NCU rule: _"The difference between calculated theoretical (X%) and measured achieved occupancy (Y%) ..."_.

**Why:** theoretical occupancy is the maximum number of warps that could be resident; achieved occupancy reflects what actually ran. The gap can come from stalls, tail imbalance, a short kernel, or a Triton configuration that produces only a few runnable programs at a time.

**Reading:** if the gap is large and Pattern B is present, fix imbalance first. Otherwise inspect stalls and whether `BLOCK_*`, `num_warps`, or `num_stages` create excessive per-program resource use.

**First-line fix:** address the dominant stall pattern and benchmark a small occupancy-oriented config sweep: smaller tiles, lower `num_stages`, and alternative `num_warps`. Do not optimize occupancy as an isolated target; reject variants that raise occupancy but lower throughput.

---

## Pattern K — Register spill

**Signals:**

- `smsp__sass_inst_executed_op_local_ld.sum > 0` or `smsp__sass_inst_executed_op_local_st.sum > 0`.
- NCU rule: _"N bytes spilled to local memory"_ in Instruction Statistics.
- `launch__registers_per_thread > 128`.

**Why:** the generated Triton kernel has too much per-thread live state: large block tensors, accumulators, unrolled values, or too many simultaneously staged tiles.

**First-line fix:** reduce live state:

- Shrink `BLOCK_*` or the accumulator tile.
- Lower `num_stages`.
- Test a different `num_warps`, because it changes how block elements are distributed across threads.
- Avoid keeping large temporary tensors alive across long sections of the kernel.

**Deeper fixes:**

- Recompute cheap values instead of retaining them.
- Split a heavily fused kernel or use a two-pass reduction.
- Where supported by the installed Triton version, test `maxnreg` as an autotuned configuration.

**Exceptions:**

- A small amount of spill can be acceptable if fusion removes much larger memory traffic or launch overhead. Compare end-to-end runtime, not spill count alone.

**Triton levers:** block/accumulator size, `num_stages`, `num_warps`, liveness, fusion boundary, and optional `maxnreg`.

---

## Pattern L — Pipeline bubbles (no compute/memory overlap)

**Signals:**

- PM timeline of `sm__throughput` and `dram__throughput` shows a sawtooth (high compute ↔ high DRAM alternating).
- `long_scoreboard` stalls are high while DRAM throughput also reaches high phases.

**Why:** a Triton loop loads a tile, consumes it completely, and only then starts the next load; the compiler has insufficient independent work or staging information to overlap the phases.

**First-line fix:** enable and tune Triton's compiler-managed software pipeline:

- Test `num_stages` values appropriate for the loop and tile size.
- Use a staged `tl.range`/`tl.static_range` structure in which the next iteration's addresses and loads are independent of the current iteration's compute.
- Reduce tile size if extra stages cause register/shared-memory pressure that removes occupancy.

**Deeper fixes:**

- Use tensor descriptors or TMA-backed paths where supported by the installed Triton/backend.
- Test warp specialization where supported so producer and consumer work can overlap.
- Restructure serial dependencies into multiple partial accumulators.

**Triton levers:** loop structure, `num_stages`, block size, descriptors, and optional warp specialization.

---

## Pattern M — Warp divergence and irregular masking

**Signals:**

- `smsp__thread_inst_executed_per_inst_executed.ratio < 32` (far from the 32 ideal).
- Branch efficiency is low in `--page details`.
- Divergent branches or heavily masked operations correlate with specific Triton source lines.

**Why:** lanes mapped from one Triton block follow different runtime paths, or most lanes are masked while the program still executes the surrounding arithmetic. `tl.where` is not a free branch: both value expressions may be evaluated.

**First-line fix:**

- Map similar work to the same program by bucketing shapes/lengths or dispatching separate kernels for different regimes.
- Use contiguous boundary masks and choose tiles that keep most lanes valid.
- Avoid `tl.where` around expensive alternatives; split into separate kernels or program-uniform branches when the two paths are large and inputs can be classified cheaply.

**Exceptions:**

- Boundary masks affecting only a small fraction of programs.
- Reduction trees where inactive lanes are limited to the final steps.

**Triton levers:** data/program mapping, shape-specialized dispatch, mask shape, and kernel splitting.

---

## Pattern N — Excessive masked work / oversized static tile

**Signals:**

- A reduction or pointwise kernel uses `BLOCK_SIZE = triton.next_power_of_2(N)` and `BLOCK_SIZE` is much larger than typical `N`.
- `smsp__thread_inst_executed_per_inst_executed.ratio` is low without a correspondingly branch-heavy algorithm.
- Register count or instruction count jumps sharply at a power-of-two shape boundary.
- Many `tl.load`, arithmetic, and `tl.store` operations carry masks for which most lanes are false.

**Why:** Triton block shapes are compile-time static. Masked lanes do not access invalid memory, but the program can still reserve registers and execute vector arithmetic for the full block shape.

**First-line fix:**

- Choose a smaller block and loop over the dimension instead of padding one program to a very large power of two.
- Use 2-D tiling or split the reduction across programs when one huge block creates mostly inactive lanes.
- Autotune separate configurations for small, medium, and large shape buckets.

**Deeper fixes:**

- Use multiple specialized kernels selected by the Python wrapper.
- Replace one oversized fused reduction with partial reductions plus a small combine kernel.

**Exceptions:**

- A small amount of masking at tensor boundaries is normal and usually cheaper than another kernel.

**Triton levers:** `BLOCK_SIZE`, multi-dimensional tiling, reduction decomposition, and shape-specialized dispatch.

---

## Pattern O — Poor program-ID ordering / weak L2 reuse

**Signals:**

- Global loads are coalesced, but `lts__t_sector_hit_rate.pct` is low and DRAM traffic is high.
- Neighboring output tiles repeatedly load the same input tile, as in GEMM, attention, or batched reductions.
- Performance changes significantly when the launch order or grouping factor changes.

**Why:** the default linear `tl.program_id` order schedules tiles in an order that evicts reusable operands before nearby programs consume them.

**First-line fix:** use grouped or swizzled program ordering. For matrix-shaped work, a `GROUP_SIZE_M`-style mapping often lets a group of programs reuse the same `B` tiles (or the symmetric mapping for `A`).

**Deeper fixes:**

- Choose the grouping dimension based on which operand has more reuse and which stride is contiguous.
- Combine grouped ordering with a persistent tile loop when the grid is large and tile costs are uniform.

**Exceptions:**

- Pure streaming kernels with no inter-program reuse.
- Working sets that already fit comfortably in cache.

**Triton levers:** `tl.program_id` remapping, grouping factor, tile traversal order, and persistent scheduling.

---

## Pattern P — Compiler lacks alignment or contiguity information

**Signals:**

- The logical access is contiguous/aligned, but NCU still reports excess sectors or generated code uses narrower/scalarized memory operations than expected.
- Strides or offsets arrive as runtime values, so the compiler cannot prove divisibility or contiguous ranges.
- A configuration becomes faster after specializing a stride or shape as `tl.constexpr`.

**Why:** Triton cannot optimize based on a property it cannot prove. Dynamic strides and indirect offset expressions can hide alignment and contiguity from the compiler.

**First-line fix:** provide only truthful compiler facts:

- `tl.multiple_of(offset_or_stride, alignment)` for guaranteed divisibility.
- `tl.max_contiguous(offsets, width)` for a guaranteed contiguous run.
- `tl.assume(condition)` for valid runtime invariants.
- `tl.make_block_ptr(..., order=...)` or `tl.constexpr` specialization when the layout contract is static.

**Deeper fixes:**

- Dispatch separate contiguous and non-contiguous kernels.
- Specialize common stride/layout cases and keep a general fallback.

**Exceptions:**

- Never add these hints speculatively. An invalid assumption can produce incorrect code, not merely slower code.

**Triton levers:** compiler hints, constexpr specialization, block pointers, and layout-specific dispatch.

---

## Ranking template for the final report

When you hand back an optimization plan, rank by `(expected speedup) × (effort ratio)`. NCU's `Est. Speedup` is your best estimator.

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <metric value(s)>
  NCU Est. Speedup: X%
  Effort: <low / medium / high>
  Why now: <reason this is the highest-leverage fix>

Priority 2: ...
```

A good rule of thumb: at most 3-5 priorities in the plan. More than that dilutes the signal, and priorities > 5 usually contribute < 5% speedup each. Each priority should name the exact Triton variant to generate: grid mapping, `BLOCK_*`, `num_warps`, `num_stages`, algorithm split/fusion, or a justified compiler hint.
