# Autoresearch: Workflow Engine Optimization

You are an autonomous optimization agent. Your goal is to continuously improve
the performance of the `workflow_engine` Python package by modifying its source
code and measuring the impact.

## Setup (do this once at the start)

1. Create a dated branch: `git checkout -b autoresearch/YYYY-MM-DD`
2. Read all files in `src/workflow_engine/` to understand the codebase
3. Run the baseline benchmark: `uv run python autoresearch/evaluate.py`
4. Record the baseline in `autoresearch/results.tsv`
5. Read this entire file before starting the experiment loop

## Metrics

| Priority | Metric | How to extract | Target |
|----------|--------|---------------|--------|
| Primary | Total execution time (seconds) | `total_time_s: X.XXXX` | Minimize |
| Secondary | Peak memory (MB) | `peak_memory_mb: X.XX` | Minimize |
| Tertiary | Correctness | `correctness: X/X` | Must be 16/16 (100%) |

**Correctness is non-negotiable.** If correctness drops below 16/16, immediately
revert the change.

## Experiment Loop

Repeat forever:

### 1. Hypothesize

Pick one optimization to try. Ideas (not exhaustive):

**Scheduling & execution:**
- Reduce overhead in topological sort / ready-node computation
- Cache `get_ready_nodes` results for unchanged subgraphs
- Optimize `workflow.expand_node()` for fewer copies
- Reduce `asyncio.create_task` overhead in parallel executor

**Data flow:**
- Reduce Pydantic validation overhead on intermediate data
- Pool or reuse `DataMapping` dicts
- Avoid unnecessary `model_copy()` calls
- Batch edge resolution

**Value system:**
- Cache `cast_to` / `can_cast_to` results for identical type pairs
- Reduce `Value` wrapper allocation (e.g., flyweight for small integers)
- Optimize `SequenceValue` operations

**Node execution:**
- Reduce `Node.__call__` overhead (context hooks, input assembly)
- Batch compatible nodes when possible
- Optimize `cached_property` usage patterns

### 2. Implement

Modify files in `src/workflow_engine/` only. Do NOT modify:
- `autoresearch/evaluate.py` (benchmark harness — read-only)
- `autoresearch/engine.py` (wrapper — read-only)
- `autoresearch/program.md` (this file — read-only)
- `tests/` (existing tests must keep passing)

### 3. Validate

```bash
# Run existing tests — must all pass
uv run pytest

# Run benchmark
uv run python autoresearch/evaluate.py
```

### 4. Record

Append a row to `autoresearch/results.tsv`:

```
experiment_id	timestamp	description	total_time_s	peak_memory_mb	correctness	kept
```

- `experiment_id`: Sequential number (001, 002, ...)
- `timestamp`: ISO 8601
- `description`: One-line description of the change
- `total_time_s`: From benchmark output
- `peak_memory_mb`: From benchmark output
- `correctness`: From benchmark output (must be 16/16)
- `kept`: `yes` if improvement, `no` if reverted

### 5. Decide

- If correctness < 16/16: **revert immediately** (`git checkout -- src/`)
- If total_time_s improved AND correctness = 16/16: **keep** (commit the change)
- If total_time_s same or worse: **revert** unless memory improved significantly
- Commit kept changes: `git add -A && git commit -m "autoresearch: <description>"`

### 6. Repeat

Go back to step 1. **NEVER STOP.**

## Constraints

- Only modify files under `src/workflow_engine/`
- Do NOT install new packages (only use existing dependencies)
- Do NOT modify `pyproject.toml`
- All existing tests must pass (`uv run pytest`)
- Correctness must remain 16/16 at all times
- Each experiment should be a single, focused change
- Keep changes small and reversible

## Search Space Priorities

Start with low-hanging fruit:
1. Reduce per-node overhead in the execution loop
2. Optimize `get_ready_nodes()` — called once per node execution
3. Reduce Value/Data allocation overhead
4. Cache frequently computed results

Then move to deeper optimizations:
5. Alternative data structures for edge lookup
6. Batch node execution where dependencies allow
7. Memory pooling for intermediate results
8. Profile-guided optimization of hot paths
