# Distributed Node Sources

**Status:** Planning

This document describes a system for installing node implementations from arbitrary
locations — published packages, git repositories, and local directories — and
referencing them from workflows through an operator-controlled name map.

The mechanism is deliberately boring: a node source is a normal Python package, it
is installed with `uv`, and it advertises the nodes it exposes through standard
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
(the same plugin pattern pytest, flake8, and spaCy use). `wengine` adds a thin
layer on top: a name map (`engine.yaml`) the operator controls, and the commands
that maintain it. The "Worked example" just below shows the whole thing end to end;
the sections after that specify each piece.

---

## Goals

- Install nodes from a published package, a git repo, or a local directory with a
  single command: `wengine install <target>`.
- Pin sources to exact versions/commits for reproducible execution — via the same
  `uv.lock` the rest of the project already uses.
- Reference third-party nodes from a workflow by a short, backend-agnostic name.
- Let an operator swap one node implementation for another (e.g. a newer `ForEach`)
  without editing any workflow.
- Keep the boundary between _what code runs_ (operator's decision) and _what a
  workflow asks for_ (author's declaration) explicit and enforceable.

## Non-goals

- Sandboxing node execution. Installing a node is arbitrary code execution by
  design — exactly as much as `pip install` or `uv add` is; this document does not
  change that.
- A second dependency manager. `wengine` drives `uv`; it does not resolve, lock, or
  install Python packages itself.
- Fixing the many out-of-date docs in `docs/`. Out of scope.
- A package index / discovery service. Sources are addressed by package name, URL,
  or path, not looked up in a registry.

---

## Worked example

The whole system in one go. `acme-scrapers` is a published package that exposes four
nodes — `HttpFetch`, `HtmlExtract`, and `Screenshot` / `CrawlSite`, the last two of
which drive a headless browser and so declare a `playwright` extra each (its full
`pyproject.toml` is shown later, under "Node-source package"). There's also a git
repo `acme/iteration` exposing a node `ForEachV2`, a local directory
`./vendor/internal-nodes` holding an in-house package `internal-nodes` (a couple of
nodes, no extra deps), and a PyPI grab-bag `legacy-nodes` whose node names happen to
clash with builtins. An operator wires them into an engine — standalone mode, so
`wengine` owns the `uv` project (the commands and `engine.yaml` syntax are specified
in the sections below):

```sh
wengine init                                                                    # creates engine.yaml + a pyproject.toml to install into
wengine install acme-scrapers --only HttpFetch --only HtmlExtract --only Screenshot
wengine install github:acme/iteration@v2.0.0 --only ForEachV2 --as ForEach --force  # override the builtin ForEach (an explicit entry → needs --force)
wengine install ./vendor/internal-nodes                                          # mount every node it exposes, bare
wengine install legacy-nodes --prefix vendor/legacy                              # its node names collide with builtins → namespace the whole bundle
```

Resulting `engine.yaml` (the operator's name map):

```yaml
schema_version: 1

nodes:
  # one explicit entry per builtin, seeded by `wengine init` (elided here):
  #   Add: aceteam-workflow-engine:Add
  #   ForEach: aceteam-workflow-engine:ForEach   # (overridden below)
  #   ... etc
  "*":
    - internal-nodes # `wengine install ./vendor/internal-nodes` → bulk install appends to the catch-all
  HttpFetch: acme-scrapers:HttpFetch # the three `--only` installs → explicit entries
  HtmlExtract: acme-scrapers:HtmlExtract
  Screenshot: acme-scrapers:Screenshot
  ForEach: acme-iteration:ForEachV2 # `--as ForEach` → shadows the builtin `ForEach`
  "vendor/legacy:*": legacy-nodes # `--prefix vendor/legacy` → workflows say `"type": "vendor/legacy:<Name>"`
```

Resulting `pyproject.toml` (the one `wengine init` created; in _embedded_ mode these
same `dependencies` / `[tool.uv.sources]` lines land in the host project's existing
`pyproject.toml` instead):

```toml
[project]
name = "my-engine"        # standalone: a name `wengine init` picked (e.g. the directory name)
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "aceteam-workflow-engine",     # from `wengine init`
  "acme-scrapers[screenshot]",   # HttpFetch & HtmlExtract need no extra; Screenshot pulls `screenshot` (→ playwright). CrawlSite wasn't installed, so no `crawl-site`.
  "acme-iteration",              # the git source — resolved spec below
  "internal-nodes",              # the local editable source
  "legacy-nodes",                # plain PyPI package; mounted under `vendor/legacy:` in engine.yaml
]

[tool.uv.sources]
acme-iteration = { url = "https://codeload.github.com/acme/iteration/tar.gz/3f2a9c1d8e0b…" }  # `wengine` resolved `@v2.0.0` → this commit; `uv.lock` pins it exactly
internal-nodes = { path = "vendor/internal-nodes", editable = true }
```

`uv.lock` (not shown) pins the full resolved closure — `playwright`, `httpx`,
`selectolax`, the exact commit of `acme-iteration`, everything `internal-nodes`
pulls in — to exact versions. `engine.yaml`, `pyproject.toml`, and `uv.lock` are all
committed; a fresh checkout runs `uv sync` (or `wengine install` with no target) to
rebuild the environment.

Now uninstall the Playwright-using node:

```sh
wengine uninstall Screenshot
```

- `engine.yaml`: the `Screenshot:` line is removed. `acme-scrapers` itself stays —
  `HttpFetch` and `HtmlExtract` are still mapped to it.
- `pyproject.toml`: `acme-scrapers[screenshot]` becomes `acme-scrapers`, because no
  remaining mapped node from that distribution declares an extra. On the next resolve
  `uv` drops `playwright` (and anything only it needed) from `uv.lock`.

Had the operator instead done `wengine install acme-scrapers` (bulk — all four nodes
on the `"*"` list) and the package used a _shared_ `browser` extra for both
`Screenshot` and `CrawlSite`, `wengine uninstall Screenshot` would have to either
drop `acme-scrapers` from the `"*"` list and add explicit entries for the three
nodes you keep, or leave the bundle whole — and it would keep `[browser]` because
`CrawlSite` still needs it. Either way `wengine` works it out from the entry-point
tables; the explicit-`--only` setup above just keeps each step a one-line change.

---

## Trust model (the reason for the structure)

There are two parties, and they do not necessarily trust each other:

| Party               | Writes                                    | Controls                                    |
| ------------------- | ----------------------------------------- | ------------------------------------------- |
| **Engine operator** | `engine.yaml`, `pyproject.toml`/`uv.lock` | Which code actually executes on this engine |
| **Workflow author** | `workflow.json`                           | Which node _names_ a workflow references    |

A workflow author writes `"type": "ForEach"` — a name, nothing more. They cannot
name a package, a version, or a module. Only the operator's `engine.yaml` binds
names to code, and only the operator's `uv` project decides which package those
names resolve into. Consequences that follow directly:

- An unmapped node name is a **hard error at load time**, never a fallback that
  fetches or installs code. The author has zero authority to introduce a source.
- `wengine install` is an **operator command**. It runs `uv add` and updates the
  operator's `engine.yaml`. Nothing a workflow does can trigger an install.
- A workflow JSON never carries resolution data (no embedded `name → package@version`
  snapshot). Such a snapshot would be the author handing the operator code to run.
  Bare-names-only in `workflow.json` _is_ the enforcement mechanism. (A workflow's
  set of required node names is therefore exactly the set of `type` values in it —
  statically readable, no execution required; validate it against `wengine nodes
list` on the target engine.)
- **Builtins are not special.** Every node except `InputNode` / `OutputNode` is
  opt-in. `aceteam-workflow-engine`'s own nodes are advertised through the same entry-point
  mechanism as third-party ones, and a fresh `engine.yaml` produced by `wengine
init` simply contains the glob that mounts them (see `engine.yaml` below). An
  operator may curate that down to the subset they want.

Validation therefore has two phases at different trust levels:

- **Author-time** (untrusted input): the workflow references only names; every
  name is in the operator's map; edges typecheck. No provider code is imported at
  workflow-load time — important, because the input is potentially hostile.
- **Operator-time** (trusted): `wengine install` runs `uv add` (which resolves,
  downloads, and installs), reads the package's entry-point table, and writes the
  name map. The first time the operator imports a node, that package's code runs —
  same as any other dependency.

Node signatures (input/output/param types) are **not** declared anywhere out of
band — an entry point is just `name → module:Class`. Typing is always inspected
through `wengine` commands that operate on the installed node classes, the same way
builtin node types are inspected today; there is no separate static declaration to
keep in sync.

**One caveat on the two-party framing.** It is a separation of _authority_, not of
_machines_. If `engine.yaml` and `workflow.json` live in the same repository — a
common setup — then whoever can commit to that repo controls both, and the
separation is organizational discipline rather than a technical boundary. The
structure earns its keep when the two genuinely come from different places (a
hosted engine running workflows submitted by others).

---

## Artifacts

| Artifact                                                 | Location                                                               | Written by                                            | Purpose                                                              |
| -------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| `engine.yaml`                                            | engine project dir                                                     | operator (via `wengine init` / `install`, or by hand) | Maps recognized node names → entry-point refs                        |
| `pyproject.toml` + `uv.lock`                             | the host project (embedded) **or** the engine project dir (standalone) | operator; `wengine install` mutates them via `uv add` | Which packages are installed, and pinned; **committed to VCS**       |
| `[project.entry-points."aceteam_workflow_engine.nodes"]` | inside each node-source package's `pyproject.toml`                     | node publisher                                        | Declares `nodeName → module:Class` for each node the package exposes |
| `workflow.json`                                          | wherever workflows live                                                | workflow author                                       | Graph of nodes referenced by bare `type` name                        |

`engine.yaml` and `workflow.json` are both required to interpret a workflow; this
is intentional (see trust model). They are not assumed to come from the same party.

The Python environment (`.venv/`) is **not** committed — like `node_modules`, it is
reconstructed with `uv sync` from the committed `uv.lock`. There is no separate
`engine.lock`: pinning is whatever `uv.lock` records (an exact version for a PyPI
package, an exact commit for a git source). Local path sources are not pinned at
all — like an editable (`uv pip install -e`) install, the path _is_ the pin;
whatever is on disk is what runs.

Note where extras land: `wengine install acme-scrapers --only Screenshot` runs
`uv add acme-scrapers[screenshot]`, which writes `acme-scrapers[screenshot]` into
`pyproject.toml` and resolves the full closure (including `playwright`) into
`uv.lock`. A flat `pip freeze` / `uv pip freeze` shows the installed packages
(`acme-scrapers==1.4.0`, `playwright==1.40.0`, …) but not the `[screenshot]`
grouping — `pyproject.toml` is the readable record of "we installed that extra," the
same way `engine.yaml` is the readable record of "we offer that name."

### Project layout and discovery

`wengine` discovers the engine project by walking up from the current directory
until it finds an `engine.yaml` (the standard package-manager search). The `uv`
project it operates on is found the same way `uv` finds it — the nearest
`pyproject.toml`. In standalone mode (below) `wengine` places both the `engine.yaml`
and the `pyproject.toml` it owns in the same directory.

### The environment: embedded vs. standalone

"The environment" is the single Python interpreter that the engine and every
installed node source import from. There are two ways `wengine` is used, and they
differ only in _who owns that environment_:

- **Embedded.** `wengine` is a dependency of an existing Python project — e.g. an
  operator building a Flask app around their engine. That project's `pyproject.toml`
  / `uv.lock` / `.venv` _is_ the engine environment; `wengine install` does `uv add`
  against it, so node sources land alongside Flask and everything else, in one
  resolved set. The operator already owns and provisions this environment (including
  its Python version); `wengine` just adds to it. The "engine project dir" is
  wherever the operator put `engine.yaml` within their project.

- **Standalone.** `wengine` is installed as self-contained software — e.g. the
  Claude skill / `wengine.sh` shim — and the user only ever touches the CLI. Here
  `wengine init` creates a `pyproject.toml` (and `.venv`/`uv.lock` follow) as a
  sibling of `engine.yaml`, and `wengine install` does `uv add` against that. Its
  Python is whatever interpreter `wengine` itself runs under.

`wengine` does not pick or pin a Python version — there is no `python:` field
anywhere. A node-source package declares its compatible interpreter range with the
standard `requires-python` in its own `pyproject.toml`, and `uv` enforces the
intersection of every installed package's range at resolve time, refusing the
install (loudly) if the environment's interpreter falls outside it. The operator
(embedded) or `wengine` (standalone) owns _which_ Python; the packages only
constrain it. The same goes for `aceteam-workflow-engine` itself: a node source depends on
`aceteam-workflow-engine>=2.1,<3` like any other dependency, and `uv` either finds a
consistent resolution or fails the `uv add` with a conflict — there is no bespoke
compatibility check.

Nothing else in this document depends on the mode. "The `uv` project" below means
"the host project in embedded mode, or the `wengine`-owned one in standalone mode";
once a source is `uv add`-ed into it, its node classes are treated exactly like
builtins.

### `engine.yaml`

```yaml
schema_version: 1

nodes: # recognized name → entry-point ref
  "*": # glob: mount every node these distributions expose, under its bare name
    - aceteam-workflow-engine # builtins, written here as a hand-authored catch-all (`wengine init` instead emits one explicit entry per builtin)
    - acme-scrapers # stacked on; fine as long as their node names don't collide
  ForEach: acme-iteration:ForEachV2 # explicit: shadow whatever `ForEach` a glob would supply, with the node `acme-iteration` exposes under entry-point name `ForEachV2`
  "vendor/legacy:*": legacy-nodes # prefixed glob: only needed to keep a colliding bundle around — mounts its nodes under vendor/legacy:<Name>
```

The value of an **explicit** entry is `<distributionName>:<entryPointName>`. The
value of a **glob** entry (key `"*"` or `"prefix:*"`) is a distribution name, or a
list of them. The distribution name is the name of the installed package — the same
string you'd `uv remove` — **not** a repo slug (the engine's own distribution is
`aceteam-workflow-engine`; `aceteam-ai/workflow-engine` is just its GitHub repo,
which only matters as an argument to the `github:` install shorthand). The
entry-point names come from each package's `[project.entry-points."aceteam_workflow_engine.nodes"]`
table; `wengine` reads that table from installed package metadata (it does **not**
import the package to discover it).

Glob keys come in two forms, and the plain one is the norm:

- **`"*"`** — mount each listed distribution's nodes under their bare `<Name>`. The
  operator stacks the bundles whose node names don't clash, which is the common case
  — `foo/bar` defines a dozen nodes that have no overlap with the base package, so
  it just goes on the `"*"` list. If two `"*"`-listed distributions expose the same
  name, that is a **hard error**, resolved either by an explicit entry for that name
  (explicit beats every glob — see precedence below) or by moving one of them to the
  prefixed form.
- **`"prefix:*"`** — mount each node as `prefix:<Name>`, a private namespace (its
  own keyspace, so it never collides with the bare names), and workflows reference
  those nodes as `"type": "vendor/legacy:Foo"`. This exists for exactly one
  situation: you want to keep a bundle wholesale despite a name clash on the `"*"`
  list and don't want to enumerate explicit entries. It is not the default.

`wengine init` writes one explicit entry per builtin (`Name: aceteam-workflow-engine:Name`),
not a `"*"` glob — the seeded map states exactly what is mounted, and curating a
subset is then deleting lines. There is no `disable:` list (an explicit entry can
only _add_ or _shadow_, not subtract; you subtract by deleting the entry). The
`"*"` glob form stays valid for anyone who prefers to hand-write a catch-all over
a distribution, but it is not what `init` emits.

### Node-source package: `[project.entry-points."aceteam_workflow_engine.nodes"]`

A node-source package is an ordinary Python distribution. It declares the nodes it
exposes via [entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
in its own `pyproject.toml`. The group is `aceteam_workflow_engine.nodes`; each
entry's name is the node name, and its value is `dotted.module.path:NodeClass` with
an optional `[extra1, extra2]` suffix naming the package extras that node needs
(the standard entry-point [extras syntax](https://packaging.python.org/en/latest/specifications/entry-points/),
which `importlib.metadata` parses — readable without importing the module).

Running example for the rest of the doc: `acme-scrapers` exposes four nodes, two of
which (`Screenshot`, `CrawlSite`) drive a headless browser and so need `playwright`:

```toml
[project]
name = "acme-scrapers"
version = "1.4.0"
requires-python = ">=3.11"
dependencies = [
  "aceteam-workflow-engine>=2.1,<3",
  "httpx>=0.27", # used by every node
  "selectolax",
]

[project.optional-dependencies]
# one extra per Playwright-needing node, named after the node (the recommended pattern)
screenshot = ["playwright>=1.40"]
crawl-site = ["playwright>=1.40"]

[project.entry-points."aceteam_workflow_engine.nodes"]
HttpFetch = "acme_scrapers.http:HttpFetchNode"
HtmlExtract = "acme_scrapers.html:HtmlExtractNode"
Screenshot = "acme_scrapers.browser:ScreenshotNode [screenshot]" # needs Playwright, via the `screenshot` extra
CrawlSite = "acme_scrapers.crawl:CrawlSiteNode [crawl-site]" # also needs Playwright
```

That's the entire publisher-side contract: declare dependencies the normal way,
and list `nodeName = "module:NodeClass [extras]"` for each node. Notes:

- **The entry-point table is the curated surface.** A package may register more node
  classes than it lists (e.g. helper nodes); only the listed ones are
  `wengine install`-able and visible to `wengine nodes list`. The table is readable
  from package metadata without importing the package, so discovery never executes
  provider code.
- **Common dependencies are package dependencies; heavy/optional ones are tied to
  the node that needs them.** A requirement every node needs goes in the package's
  `dependencies` (`httpx`, `selectolax` above). A requirement only some nodes need —
  `playwright`, a big ML stack — goes in `[project.optional-dependencies]` as an
  extra, and each node's entry point carries that extra in its `[…]` suffix (the
  [standard entry-point extras
  syntax](https://packaging.python.org/en/latest/specifications/entry-points/)).
  `wengine install acme-scrapers --only Screenshot` then `uv add`s `acme-scrapers[screenshot]`;
  `--only HttpFetch` adds no extra; a bulk install (mounting the whole package) adds
  the union of every mounted node's extras (here `screenshot,crawl-site`). There is
  no `setup:` script and no node-level install hooks; `uv` resolves everything in
  one pass, which is what makes cross-source conflicts surface as a resolution error
  at install time instead of an `ImportError` at runtime.
  - _Recommended (and what the example does):_ **one extra per node, named after the
    node** — then an operator's `pyproject.toml` reads transparently (`acme-scrapers[screenshot]`
    ⇒ "we installed `Screenshot`"), and removing a node is deleting that line plus
    the `engine.yaml` line. A _shared_ extra works too (e.g. one `browser` extra used
    by both `Screenshot` and `CrawlSite` instead of one each): `wengine` reads the
    entry-point tables and tracks which still-mapped nodes need it (see "Uninstalling"),
    so it never drops an extra another installed node depends on — but then the
    `pyproject.toml` line alone doesn't say who needs it, and you have to go through
    `wengine` rather than hand-editing.
- **Monorepos** are the standard pip "package living in a subdirectory" case — the
  install target is `git+https://github.com/acme/big-monorepo@<ref>#subdirectory=tools/workflow-nodes`;
  nothing `wengine`-specific.
- **A loose `.py` is not a node source.** If you want to publish a node, it has to
  live in a package with a `pyproject.toml` — `uv init` writes a minimal one. This
  is the one real cost of the model: it buys declarable, resolvable dependencies and
  a lockfile, but it does mean "any subdirectory of any repo" is no longer enough on
  its own.

---

## The `wengine install` target

```sh
wengine install <target> [--only <NodeName> ...] [--as <name>] [--prefix <p>] [--force]
wengine install            # no target: sync the environment to uv.lock + re-apply the name map
wengine init               # create engine.yaml (and, standalone, pyproject.toml)
```

`<target>` is anything `uv add` accepts, plus two shorthands:

| Form                                                                   | Expands to / means                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `acme-scrapers` / `acme-scrapers==1.4.0` / `acme-scrapers[screenshot]` | A PyPI (or configured index) requirement, passed straight to `uv add` (extras you name here are added on top of any a node implies).                                                                                                         |
| `git+https://host/owner/repo@<ref>` (optionally `#subdirectory=...`)   | A git requirement, passed straight to `uv add`.                                                                                                                                                                                              |
| `./path` or `path:./path`                                              | An editable local install (`uv add --editable ./path`).                                                                                                                                                                                      |
| `github:owner/repo@<ref>` (and `gitlab:`, `…`)                         | `wengine` resolves `<ref>` → commit over HTTPS, then hands `uv` a pinned **tarball URL** (`https://codeload.github.com/owner/repo/tar.gz/<sha>` or equivalent) — a URL dependency, not a git dependency, so this path needs no system `git`. |
| `owner/repo` (bare, looks like owner/repo)                             | `github:owner/repo` (default branch).                                                                                                                                                                                                        |

| Option              | Meaning                                                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _(none)_            | Append the distribution to the `"*"` list — mount all its nodes bare, and request the union of their declared extras. If a bare name would collide, the install errors and points you at `--only` / `--prefix`. |
| `--only <NodeName>` | Write an explicit entry for just that node instead of touching the `"*"` list, and request only that node's declared extras (repeatable).                                                                       |
| `--as <name>`       | The recognized name an `--only` node maps to (only valid with a single `--only`). Defaults to the entry-point name. Use it to rename-on-install (e.g. `--only ForEachV2 --as ForEach --force` to override the builtin — since `init` seeds builtins as explicit entries, this is an explicit-vs-explicit clash and needs `--force`). |
| `--prefix <p>`      | Write a `"<p>:*"` glob instead of touching the `"*"` list — mount the whole bundle under `<p>:<Name>` (with the union of its nodes' extras). The collision escape hatch.                                        |
| `--force`           | Overwrite an existing explicit `nodes:` entry that points at a different distribution. Without it, such a clash is an error.                                                                                    |

Examples:

```sh
wengine install acme-scrapers                          # mount all 4 nodes; pulls the `screenshot,crawl-site` extras (Screenshot & CrawlSite need Playwright)
wengine install acme-scrapers==1.4.0
wengine install acme-scrapers --only HttpFetch         # just that node; no extras pulled
wengine install acme-scrapers --only Screenshot        # uv add acme-scrapers[screenshot]
wengine install github:acme/scrapers@v1.4.0
wengine install git+https://gitlab.com/acme/iteration.git@main --only ForEachV2 --as ForEach
wengine install github:acme/big-monorepo@main#subdirectory=tools/workflow-nodes
wengine install legacy-nodes --prefix vendor/legacy   # mount under vendor/legacy:<Name>
wengine install ./local/path
```

In short, `wengine install <target>` ≈ `uv add <target>` plus an `engine.yaml`
edit — `wengine` is deliberately thin here. The non-trivial parts it adds: it
expands the `github:` / `owner/repo` shorthands `uv add` doesn't understand; it
reads the package's `aceteam_workflow_engine.nodes` entry-point table _before_ the `uv add`
(to know what to write, and to run the merged-map collision check); and `--only` /
`--as` / `--prefix` / `--force` affect only the `engine.yaml` side, never the `uv add`. In full,
what `wengine install <target>` does (operator-time, trusted), in the discovered
engine project:

1. Parse the target; expand the `github:` / `owner/repo` shorthands (resolving a ref
   to a commit and a tarball URL over HTTPS if needed).
2. **Collect metadata first.** Determine the distribution name and read its
   `[project.entry-points."aceteam_workflow_engine.nodes"]` table (from the sdist/wheel
   metadata `uv` is about to install, or — for a git/path target — by building the
   package's metadata, which `uv` does without running it). Compute the proposed
   `engine.yaml` change: append to the `"*"` list (default), or the explicit /
   `prefix:*` entries implied by `--only` / `--as` / `--prefix`. Compute the extras
   to request: the union of the `[extras]` suffixes on the entry points being
   installed (every node, for a bulk / `--prefix` install; just the `--only` ones
   otherwise).
3. **Check the merged map.** Combine the proposed mappings with the existing
   `engine.yaml`. A new explicit entry that shadows a glob is fine. A new explicit
   entry that collides with an existing _explicit_ entry for a different distribution
   is an error unless `--force` — this includes overriding a builtin, since `init`
   seeds builtins as explicit entries (so `--as ForEach` over the seeded `ForEach`
   needs `--force`). A bare name that
   two glob-mounted distributions would both supply, with no explicit entry to
   disambiguate, is a hard error — abort before touching anything. (This is why
   step 2 reads all the relevant entry-point tables before installing.)
4. `uv add <expanded target>[<extras from step 2>]` — `uv` resolves it together with
   everything already in the project, enforces `requires-python` and all version
   constraints, installs, and updates `pyproject.toml` + `uv.lock`. A conflict here
   aborts the install with `uv`'s error.
5. Apply the `engine.yaml` change from step 2 (append to the `"*"` list, or write
   the explicit / `prefix:*` entries).

`wengine init` creates `engine.yaml` with one explicit entry per builtin (and, in
standalone mode, a minimal `pyproject.toml` declaring `aceteam-workflow-engine`
as a dependency, then runs `uv add aceteam-workflow-engine` to resolve and lock
it). `wengine install` with no target runs `uv sync`
and re-derives the registry from `engine.yaml` — the "reconstruct the environment
from the lockfile" operation, for a fresh checkout or CI.

There is no separate trust prompt: installing a node source is `uv add`, and
`wengine` does not layer a confirmation on top of `uv`/`pip` that they do not have
themselves. Operators who want a review gate already have one — code review on the
`pyproject.toml` / `uv.lock` change.

### Uninstalling

`wengine uninstall <NodeName>` removes the `engine.yaml` entry, then reconciles
`pyproject.toml`:

- If other nodes from the same distribution are still mapped, it recomputes that
  distribution's extra set — the union of the `[extras]` of the _still-mapped_ nodes
  — and, if it shrank, runs `uv add <distribution>[<new set>]` (which lets `uv` drop
  any now-unneeded transitive deps). So uninstalling `Screenshot` rewrites
  `acme-scrapers[screenshot]` → `acme-scrapers` only if no remaining mapped node
  needs the `screenshot` extra (or, with a shared `browser` extra, no remaining node
  needs `browser` — e.g. `CrawlSite` is gone too); `wengine` reads the entry-point
  tables, so it always knows the answer.
- If no entry references the distribution at all anymore, it runs `uv remove <distribution>`.

A distribution reachable through several `engine.yaml` entries (the `"*"` list plus
an explicit override, say) needs all of them gone before that last step fires;
`wengine uninstall` handles that, and `--dist <name>` removes every entry for a
distribution at once. (Hand-editing `pyproject.toml` / `engine.yaml` instead of
using `wengine uninstall` is fine if the package uses one-extra-per-node — see the
publisher recommendation above — and otherwise risks dropping an extra something
still needs; `wengine install` with no target re-reconciles either way.)

---

## Resolution and loading

Node-name lookup, in order:

```text
engine.yaml.nodes[<name>]   →   error
```

There is no implicit builtin fallback — if `aceteam-workflow-engine`'s nodes are
wanted, they are in `engine.yaml` (the explicit entries `wengine init` seeds, or
a hand-authored `"*"` list). When more than one `nodes:` entry could supply a name,
precedence is **explicit entry > glob entry**; two glob-mounted distributions
supplying the same bare name is the install-time error described above (and
`prefix:*` mounts live in their own `prefix:`-keyed space, so they never collide
with bare names), so at load time the map is already unambiguous.

Loading a mapped node (assumes the distribution is already installed — `wengine
install <target>`, or `wengine install` with no target / `uv sync`, put it in the
environment):

1. Resolve the recognized name through `engine.yaml` to `<distribution>:<entryPointName>`.
2. Look up that entry point in `<distribution>`'s metadata to get `module:Class`
   (and any required extras). If the distribution is **not installed**, or the entry
   point declares an extra that isn't installed, this is a fatal error that names the
   distribution (and extra) and tells the operator to run `wengine install
<distribution>[<extra>]` — load/exec never fetches, installs, or builds anything
   implicitly, exactly like a missing Python import, but the message is sharper than
   a raw `ModuleNotFoundError`. (`uv sync` / `wengine install` is the only thing that
   touches the network.)
3. Import `module` — triggers `__init_subclass__` registration as today.
4. Pull `Class` out of the node registry.
5. Register it in the engine's node registry under the recognized name.

The `type` discriminator in `workflow.json` stays a bare name (`"ForEach"`).
Deserialization does **not** use a Pydantic discriminated union — those cannot be
extended at runtime — it goes through the engine's mutable node registry (see
`src/workflow_engine/core/node.py`), which already maps a `type` string to a node
class. Distributed sources just extend that registry. No new "scoped registry"
machinery is needed: a `wengine` process serves one engine project (one
`engine.yaml`), so populating the process's node registry from that project's map
_is_ the scoping.

---

## Compatibility burden on the operator

When an operator edits `engine.yaml` to point `ForEach` at a different distribution
or entry point — or bumps that distribution's version in `pyproject.toml` — every
workflow referencing `ForEach` is silently affected: same name, new code. This is
intended (it is how you patch a node fleet-wide), but the new `ForEach` must still
satisfy the old type signature or those workflows break at load. Implied tooling: a
`wengine verify` that re-typechecks known workflows against the current map after
an operator change. Not core, but the trust model puts this burden on the operator,
so the operator needs the tool.

Note also that nothing in a `workflow.json` declares the signature it relies on
(only the `type` names — see trust model). If workflows are ever meant to be
portable across engines, a forward-declared per-workflow node contract would be the
thing to add; for now a workflow is interpreted relative to one engine, and
`wengine verify` / `wengine nodes list` are how you check the fit.

---

## Build order

1. **`engine.yaml` schema + project discovery** — Pydantic model for `engine.yaml`
   (with the `nodes:` value grammar and glob keys), and the walk-up search for it.
2. **Install-target parser** — `<pip target>` plus the `github:` / `gitlab:` /
   `owner/repo` / `path:` shorthands → a concrete `uv add` argument (resolving a ref
   to a commit + tarball URL over HTTPS for the forge shorthands). Pure, testable in
   isolation.
3. **`uv` integration** — locate the host `uv` project (embedded) or create/own one
   beside `engine.yaml` (standalone); shell out to `uv add` / `uv remove` / `uv sync`
   and surface their output.
4. **Entry-point discovery** — read `[project.entry-points."aceteam_workflow_engine.nodes"]`
   (each entry's `module:Class` and its `[extras]`) from installed (or
   about-to-be-installed) distribution metadata via `importlib.metadata`, without
   importing the package.
5. **`wengine init` / `wengine install` / `wengine uninstall`** — wire 1–4 together;
   two-phase install (collect entry-point tables of all affected distributions →
   merge with `engine.yaml` → reject unresolved bare-name collisions among
   glob-mounted distributions → `uv add` → write `nodes:` entries). `wengine init`
   seeds one explicit entry per builtin.
6. **Loader integration** — populate the process node registry from the project's
   `nodes:` map (precedence: explicit entry > glob; bare-name collisions among globs
   are errors), import-and-pull the class lazily on first use, fatal error if the
   distribution is missing. Builds on the existing registry in
   `src/workflow_engine/core/node.py`.
7. **`wengine nodes list`, `wengine verify`** — follow-ups (`nodes list` is also how
   the operator publishes the available-node set to workflow authors).

---

## Settled decisions

Chosen over alternatives; the body above has the reasoning.

- **A node source is a Python package, installed with `uv`.** One resolver, one
  lockfile (`uv.lock`) — not a second dependency manager; no `setup:` scripts. Cross-source
  conflicts therefore fail at `uv add` time, not at runtime.
- **Nodes are advertised via standard entry points** (`aceteam_workflow_engine.nodes`),
  readable from package metadata without importing; a node's optional deps ride in
  the entry's `[extras]` suffix. `wengine` keeps each distribution's `pkg[extras]`
  line equal to the union of its still-mapped nodes' extras. Recommended: one extra
  per node, named after it.
- **Builtins are not special** — `aceteam-workflow-engine`'s nodes are entry points
  like any others; `wengine init` seeds one explicit entry per builtin that mounts them.
- **Two modes** — embedded (operator's existing `uv` project _is_ the engine env) and
  standalone (`wengine init` creates one beside `engine.yaml`).
- **`uv.lock` committed, environment not** — `uv sync` (or `wengine install` with no
  target) rebuilds it. No separate `engine.lock`.
- **No Python pinning, no bespoke version check** — packages declare `requires-python`
  / `aceteam-workflow-engine>=…`; `uv`'s resolver enforces it.
- **Forge shorthands need no system `git`** — `github:` / `gitlab:` resolve a ref to a
  commit over HTTPS and install a pinned tarball; raw `git+…` URLs are handed to `uv`
  as-is and follow its rules.
- **Two-phase install** — validate the merged name map (rejecting bare-name collisions
  among glob-mounted distributions) before `uv add` or any file write.
- **No lazy install at load/exec time** — a missing distribution is a fatal error
  pointing at `wengine install`, like a missing Python import; only `wengine install`
  / `uv sync` touch the network.
- **No "scoped registry"** — one `wengine` process = one engine project, so the
  project's `nodes:` map _is_ the scoping. Precedence: explicit entry > glob; a
  glob/glob clash on a bare name is an error. Deserialization goes through the
  existing mutable node registry (no Pydantic discriminated union — those can't be
  extended at runtime).
- **`path:` sources are never pinned** — editable semantics; the path is the pin.
- **`nodes:` entry kinds** — explicit (`Name: dist:entryPoint`; enables `--as` /
  overriding a builtin), `"*"` glob (the normal case; stacks), `"prefix:*"` glob (the
  collision escape hatch).

## Open questions

1. **Glob collision UX.** When two `"*"`-listed distributions both expose a node, the
   resolutions are "add an explicit entry for that name" or "move one to a
   `prefix:*` mount" — but the error needs to spell that out, and ideally offer to
   apply one. Mechanics are settled; the message and the assist are not.
2. **`disable:` for builtins.** Curating builtins currently means dropping the
   distribution from the `"*"` list and adding explicit entries for the nodes you
   keep. A `disable: [SomeNode]` list would be more ergonomic but adds a subtractive
   operation to the map's semantics; deferred until there's demand.
