# Distributed Node Sources

**Status:** Planning

This document describes a system for installing node implementations from arbitrary
locations — git repositories (GitHub or otherwise) and local directories — and
referencing them from workflows through an operator-controlled name map.

---

## Goals

- Install nodes from a git repo or local directory with a single command:
  `wengine install <string>`.
- Pin sources to exact commits for reproducible execution.
- Reference third-party nodes from a workflow by a short, backend-agnostic name.
- Let an operator swap one node implementation for another (e.g. a newer `ForEach`)
  without editing any workflow.
- Keep the boundary between _what code runs_ (operator's decision) and _what a
  workflow asks for_ (author's declaration) explicit and enforceable.

## Non-goals

- Sandboxing node execution. Installing a node is arbitrary code execution by
  design; this document does not change that.
- Fixing the many out-of-date docs in `docs/`. Out of scope.
- A package index / discovery service. Sources are addressed by URL or path, not
  looked up in a registry.

---

## Trust model (the reason for the structure)

There are two parties, and they do not necessarily trust each other:

| Party               | Writes                       | Controls                                    |
| ------------------- | ---------------------------- | ------------------------------------------- |
| **Engine operator** | `engine.yaml`, `engine.lock` | Which code actually executes on this engine |
| **Workflow author** | `workflow.json`              | Which node _names_ a workflow references    |

A workflow author writes `"type": "ForEach"` — a name, nothing more. They cannot
name a URL, a commit, or a module. Only the operator's `engine.yaml` binds names
to code. Consequences that follow directly:

- An unmapped node name is a **hard error at load time**, never a fallback that
  fetches code. The author has zero authority to introduce a source.
- `wengine install` is an **operator command**. It mutates the operator's
  `engine.yaml` / `engine.lock`. Nothing a workflow does can trigger a fetch.
- A workflow JSON never carries resolution data (no embedded `name → source@commit`
  snapshot). Such a snapshot would be the author handing the operator code to run.
  Bare-names-only in `workflow.json` _is_ the enforcement mechanism.
- Builtin nodes are simply the operator's default allowlist — the implicit prefix
  of the name map that the engine ships with. An operator may ship an engine that
  disables builtins it does not want.

Validation therefore has two phases at different trust levels:

- **Author-time** (untrusted input): the workflow references only names; every
  name is in the operator's map; edges typecheck. No provider code is executed at
  workflow-load time — important, because the input is potentially hostile.
- **Operator-time** (trusted): `wengine install` fetches, checks compatibility,
  runs the source's setup commands, imports modules. All code execution is gated
  here, by the operator.

Node signatures (input/output/param types) are **not** declared in the manifest —
the manifest only lists which nodes a repo exposes and where to import them from.
Typing is always inspected through `wengine` commands that operate on the installed
node classes, the same way builtin node types are inspected today; there is no
separate static declaration to keep in sync.

---

## Artifacts

| Artifact                          | Location                      | Written by                                   | Purpose                                                                                                     |
| --------------------------------- | ----------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `engine.yaml`                     | engine project dir            | operator (via `wengine install`, or by hand) | Maps recognized node names → fully-qualified source refs                                                    |
| `engine.lock`                     | **sibling of `engine.yaml`**  | `wengine install`                            | Resolves each source ref → exact commit + manifest hash; **committed to VCS**                               |
| `<subdir>/.wengine/manifest.yaml` | **inside each provider repo** | node author                                  | Declares which nodes the repo exposes, where to import them, setup commands, and compatibility requirements |
| `workflow.json`                   | wherever workflows live       | workflow author                              | Graph of nodes referenced by bare name                                                                      |

`engine.yaml` and `workflow.json` are both required to interpret a workflow; this
is intentional (see trust model). They are not assumed to come from the same party.
`engine.yaml` and `engine.lock` are checked into VCS; the Python environment they
describe (see below) is **not** — like `node_modules`, it is reconstructed from the
lockfile (`engine.lock` resolves every source; re-running each source's setup
rebuilds the installed packages).

### Project layout and discovery

`engine.yaml` and `engine.lock` are siblings in one directory — the "engine project
dir". `wengine` discovers the project by walking up from the current directory until
it finds an `engine.yaml` (the standard package-manager search), and `engine.lock`
is written next to the `engine.yaml` it locks.

Raw git checkouts of sources live in a shared, content-addressed cache
(`~/.cache/wengine/checkouts/<commit>/`, immutable) — a checkout is just bytes, no
reason to duplicate it per project.

### Fetching sources

The "`uv` / `curl` / POSIX builtins, nothing else" rule constrains what a
provider's `setup` script may assume — it is _not_ a limit on `wengine` itself.
`wengine` does the fetching, and it carries its own machinery for it: a
**pure-Python git client** (`dulwich`, a small flat dependency — pure Python with an
optional C speedup, only `urllib3` underneath) added to `workflow-engine`. This is
what lets a minimal standalone install with no system `git` still clone
`https`-hosted repos (GitHub, GitLab, Gitea, Bitbucket, …). `git+ssh://` sources are
the one case that may still need a system `ssh`; document that as a known
limitation. `path:` sources are just copied/symlinked, no git involved.

Because we always resolve a ref to a commit SHA _before_ fetching, `wengine` may
also use a provider's HTTPS archive endpoint (`codeload.github.com/.../<sha>.tar.gz`
and equivalents) for the `github:` / `gitlab:` shorthands as a cheaper fast path — a
pinned tarball is exactly as reproducible as a clone, and avoids pulling history for
a large repo. This is an optimization, not a separate code path the user sees.

### The environment: embedded vs. standalone

"The environment" is the single Python interpreter that the engine and every
installed source import from, and the only place setup commands install into. There
are two ways `wengine` is used, and they differ only in _who owns that
environment_:

- **Embedded.** `wengine` is `pip install`-ed into an existing Python project's
  environment — e.g. an operator building a Flask app around their engine. That
  project's environment _is_ the engine environment; `wengine install` installs
  node sources into it, alongside Flask and everything else. There is no separate
  `.wenv`; the operator already owns and provisions the venv (and its Python
  version), `wengine` just adds to it. The "engine project dir" is wherever the
  operator put `engine.yaml` within their project.

- **Standalone.** `wengine` is installed as self-contained software — e.g. the
  Claude skill / `wengine.sh` shim — and the user only ever touches the CLI. Here
  `wengine` provisions and owns its own environment, a `.wenv/` sibling of
  `engine.yaml`, created on the first `wengine install`. Its Python is whatever
  interpreter `wengine` itself runs under.

In neither mode does `wengine` pick or pin a Python version — there is no `python:`
field. Instead each provider manifest declares a _compatible range_ in
`requires.python` (e.g. `">=3.11,<3.14"`), and `wengine install` checks the
environment's interpreter against the intersection of every installed source's
range, refusing the install (loudly) if it falls outside. The operator (embedded)
or `wengine` (standalone) owns _which_ Python; the manifests only constrain it.

In both modes the rest of this document is unchanged — setup runs with that
environment active, `uv pip install ...` targets it, node classes become importable
from it. "`.wenv`" below means "the standalone-mode environment, or the host
project's environment in embedded mode."

### `engine.yaml`

```yaml
schema_version: 1

sources: # optional: named source blocks
  scrapers:
    github: acme/scrapers # sugar for git: https://github.com/acme/scrapers
    ref: v1.4.0 # tag, branch, or commit; resolved in engine.lock
  iteration:
    git: https://gitlab.com/acme/iteration.git
    ref: 3f2a9c1
  monorepo-nodes:
    github: acme/big-monorepo
    subdir: tools/workflow-nodes # the .wengine/ folder lives here, not at repo root
    ref: main
  shared:
    path: ../shared-nodes # local dir; no version concept

nodes: # recognized name → fully-qualified ref
  ForEach: iteration#ForEach # override the builtin ForEach
  HttpFetch: scrapers#HttpFetch
  "acme/scrapers:*": scrapers # glob: mount every node in the source under acme/scrapers:<Name>
```

A `nodes:` entry's value is either `<sourceAlias>#<NodeName>` (reference a source
block) or an inline fully-qualified string (see grammar below) for a one-off.

### `engine.lock`

```yaml
schema_version: 1
sources:
  scrapers:
    git: https://github.com/acme/scrapers
    resolved: 3f2a9c1d8e0b... # exact commit
    manifest_hash: sha256:ab12... # hash of .wengine/manifest.yaml at that commit
  iteration:
    git: https://gitlab.com/acme/iteration.git
    resolved: 9c1f0a3...
    manifest_hash: sha256:cd34...
```

`engine.yaml` records human intent ("that tag"); `engine.lock` records the machine
fact ("that commit") and sits next to the `engine.yaml` it locks. CI and production
read `engine.lock` and never resolve refs over the network beyond fetching the
locked commit. Local `path:` sources are not locked at all — like an editable
(`pip install -e`) install, the path _is_ the pin; whatever is on disk is what runs.

### `<subdir>/.wengine/manifest.yaml` (provider repo)

By default the manifest lives at `.wengine/manifest.yaml` at the repo root. A source
may set `subdir:` (in `engine.yaml`) / `--subdir` (on the install string) to point
at a subdirectory that contains the `.wengine/` folder — for a monorepo where only
one subtree is workflow-related. `module:` import paths in the manifest are resolved
relative to whatever ends up importable in `.wenv/` after setup, not relative to the
subdir; how a subdir's package gets onto the path is the manifest author's job (their
`setup` does the `uv pip install -e .` or equivalent).

```yaml
schema_version: 1
package: acme-scrapers

requires:
  workflow-engine: ">=2.1,<3" # checked at install time; loud error on mismatch
python: ">=3.11"

setup: # repo-level: runs once when the source is installed
  - uv pip install httpx>=0.27 selectolax

nodes:
  HttpFetch:
    module: acme_scrapers.http # imported to trigger registration
    class: HttpFetchNode
    version: 1.2.0
    display_name: HTTP Fetch
    description: Fetches a URL and returns the response body.
  HtmlExtract:
    module: acme_scrapers.html
    class: HtmlExtractNode
    version: 0.9.1
    setup: # per-node: runs when this specific node is installed
      - uv pip install lxml>=5
```

The manifest is the source of truth for what is _installable_ from the repo, even
if the imported module registers more node classes than the manifest lists. This
lets a repo expose a curated subset and provides metadata without executing code.

**Setup commands.** `setup` is an ordered list of shell commands. The repo-level
`setup` runs when the source is installed; a node's `setup` runs when that node is
installed (and only then — installing `HttpFetch` does not trigger `HtmlExtract`'s
setup). Constraints:

- **Idempotent.** Setup commands must be safe to run more than once. `wengine`
  assumes this and will re-run them when in doubt (after a `.wenv` rebuild, a moved
  tag, etc.) rather than tracking fine-grained "already done" state.
- **Run in `.wenv`.** Setup executes with the project's `.wenv/` virtualenv
  active, so `uv pip install ...` installs into `.wenv` — never the operator's
  ambient Python. This is the intended and only mechanism for a node to pull in
  dependencies. `wengine` does not import a source until its setup has run.
- **`uv`, `curl`, POSIX builtins.** `uv` is a dependency of `workflow-engine`
  itself (the wengine CLI is already run under `uv` — see the `wengine.sh` skill
  shim), and `curl` is ubiquitous enough to count on. Setup commands may assume
  `uv`, `curl`, and POSIX shell builtins are on `PATH`, and nothing else (`git`,
  `make`, language toolchains, …).
- **Working directory = the source's subdir.** Setup runs with cwd set to the
  source's `<subdir>/` inside its (read-only-ish) checkout, so `uv pip install -e .`
  / `uv pip install -r requirements.txt` resolve against the repo's own files.
- **Fail-fast.** The list runs in order; the first command to exit non-zero aborts
  the install (repo-level `setup` before any per-node `setup`).
- **Trusted, unsandboxed.** Setup runs at operator-time (see trust model); the
  operator chose to install this source. No sandbox; the host environment (env
  vars, `PATH`) is passed through as-is.

Once setup has run, the source's node classes are importable into `.wenv`, and from
there everything downstream — type resolution, validation, execution — treats them
exactly like builtin node classes.

---

## The `wengine install` string

```sh
wengine install [scheme:]location[//subdir][@ref][#NodeName] [--subdir <path>] [--as <name>]
```

| Part          | Meaning                                                                                                                                                                                             |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scheme`      | `github` \| `gitlab` \| `git+https` \| `git+ssh` \| `path` \| `file`. Default: `github` if `location` looks like `owner/repo`; `path` if it is an existing filesystem path.                         |
| `location`    | `owner/repo`, a full git URL, or a filesystem path.                                                                                                                                                 |
| `//subdir`    | Directory within the repo that contains the `.wengine/` folder. Defaults to the repo root. Also expressible as `--subdir <path>`.                                                                   |
| `@ref`        | Tag, branch, or commit. Omitted → default branch, then pinned in `engine.lock`.                                                                                                                     |
| `#NodeName`   | Install only that node. Omitted → install all nodes in the manifest.                                                                                                                                |
| `--as <name>` | Recognized name to map to (enables rename-on-install, e.g. `--as ForEach` to override the builtin). Defaults to the manifest node name, or for `#`-less installs, the node names from the manifest. |

Examples:

```sh
wengine install github:acme/scrapers
wengine install github:acme/scrapers@v1.4.0
wengine install github:acme/scrapers@3f2a9c1#HttpFetch
wengine install github:acme/big-monorepo//tools/workflow-nodes@main
wengine install git+https://gitlab.com/acme/iteration.git@main --as ForEach
wengine install ./local/path
```

What the command does (operator-time, trusted), in the discovered engine project dir:

1. Parse the string; determine source kind, location, subdir, ref, optional node filter.
2. Resolve `ref` → commit (for git sources). For paths, the path is the identity.
3. Fetch the commit into the content-addressed checkout cache (`~/.cache/wengine/checkouts/<commit>/`); read `<subdir>/.wengine/manifest.yaml` and compute its hash.
4. Check `requires.workflow-engine` against the running version; abort loudly on mismatch.
5. Ensure `.wenv/` exists (create it if this is the first install in the project).
6. Run the repo-level `setup` commands, then the `setup` commands for each node being installed, with `.wenv/` active. (Setup is idempotent, so re-running on a repeat install is harmless.)
7. Write/update the `sources:` block in `engine.yaml` (including `subdir:` if non-default) and the entry in `engine.lock` (commit + manifest hash).
8. Write/update `nodes:` entries for the installed node(s), honoring `--as`.

On first install of a previously-unseen source, prompt the operator to trust it
(à la editor workspace trust). Skipped in non-interactive mode only when the source
is already pinned in `engine.lock` with a matching `manifest_hash`.

---

## Resolution and loading

Node-name lookup, in order:

```text
engine.yaml.nodes[name]   →   builtin registry[name]   →   error
```

A mapped name is redirected before the builtin is consulted, so overriding a
builtin is just an `engine.yaml` entry that shadows it. An unmapped name is fatal.

Loading a mapped node (assumes the source was already installed — `wengine install`
ran its setup into `.wenv/`):

1. Look up the fully-qualified ref via `engine.yaml` → `engine.lock`.
2. Ensure the checkout is present in the cache at the locked commit (fetch if not;
   verify `manifest_hash`), and that its setup has run into `.wenv/` (re-run if not
   — setup is idempotent).
3. With `.wenv/` active, import the declared `module` — triggers `__init_subclass__`
   registration as today.
4. Pull the declared `class` out of the node registry.
5. Register it in the engine's node registry under the recognized name. Builtins
   keep their bare names; third-party nodes are reachable only through the map.

The `type` discriminator in `workflow.json` stays a bare name (`"ForEach"`).
Deserialization does **not** use a Pydantic discriminated union — those cannot be
extended at runtime — it goes through the engine's mutable node registry (see
`src/workflow_engine/core/node.py`), which already maps a `type` string to a node
class. Distributed sources just extend that registry. No new "scoped registry"
machinery is needed: a `wengine` process serves one engine project (one
`engine.yaml`), so populating the process's node registry from that project's map
_is_ the scoping. The only rule to fix is precedence on a name clash — explicit
`nodes:` entries beat glob entries, and any mapped entry beats the builtin it
shadows — which is just the lookup order above made total.

---

## Compatibility burden on the operator

When an operator edits `engine.yaml` to point `ForEach` at a new source or commit,
every workflow referencing `ForEach` is silently affected — same name, new code.
This is intended (it is how you patch a node fleet-wide), but the new `ForEach`
must still satisfy the old type signature or those workflows break at load. Implied
tooling: a `wengine verify` that re-typechecks known workflows against the current
map after an operator edit. Not core, but the trust model puts this burden on the
operator, so the operator needs the tool.

---

## Build order

1. **Reference-string parser** — `[scheme:]location[//subdir][@ref][#NodeName]` →
   structured form. Pure, testable in isolation.
2. **Project discovery + schemas** — walk-up search for `engine.yaml`; Pydantic
   models for `engine.yaml`, `engine.lock`, and `.wengine/manifest.yaml`, with
   validation.
3. **Source resolver + checkout cache** — resolve a ref to a SHA and fetch (via
   `dulwich` over `https`, or the provider tarball fast path) into a commit-keyed
   cache dir, honoring `subdir`; resolve a local `path:`; read and validate the
   manifest; compute its hash; check `requires.python` / `requires.workflow-engine`.
4. **Environment management + setup runner** — locate the host env (embedded) or
   create/locate `.wenv` (standalone); execute the ordered `setup` command lists
   (repo-level, then per-node) with that environment active.
5. **`wengine install`** — wire 1–4 together; write `engine.yaml` / `engine.lock`.
6. **Loader integration** — populate the process node registry from the project's
   map (precedence: explicit > glob > builtin), import-and-pull-class. Builds on
   the existing registry in `src/workflow_engine/core/node.py`.
7. **`wengine verify`**, trust prompt, glob entries — follow-ups.

---

## Settled decisions

- **One shared environment, no per-source isolation.** Engine, builtins, and every
  installed source live in one Python interpreter (host project's in embedded mode;
  a provisioned `.wenv/` in standalone mode). Setups `uv pip install ...` into it.
  Conflicting requirements between two sources are the operator's problem.
- **Two usage modes** — embedded (`wengine` installed into an existing project's
  env; that env is the engine env; operator owns it) and standalone (`wengine`
  owns a `.wenv/` sibling of `engine.yaml`, Python taken from the interpreter
  `wengine` runs under). The CLI walks up from cwd to find `engine.yaml`. Raw
  checkouts go in a shared content-addressed cache.
- **`engine.lock` is committed; the environment is not.** A fresh checkout / CI run
  reconstructs the environment from `engine.lock` (every source resolved to a
  commit) plus re-running each source's setup. `wengine install` with no args does
  this "sync from lock" (build/repair the env to match `engine.lock`).
- **Setup is idempotent and order-independent**, with one ordering rule: a repo's
  repo-level `setup` runs before any of that repo's node-level `setup`s. Across
  repos there is no order, so reconstructing the environment from scratch is safe —
  re-run every repo-level setup, then every node-level setup.
- **Setup execution environment is fixed:** target environment active; cwd = the
  source's `<subdir>/` in its checkout; `uv` / `curl` / POSIX builtins assumed on
  `PATH`, nothing else; host env passed through unmodified; fail-fast on the first
  non-zero exit.
- **`wengine` fetches with a pure-Python git client** (`dulwich`, a small flat
  dependency of `workflow-engine`) — so no system `git` is required for `https`
  sources. `git+ssh://` may still need a system `ssh` (known limitation). The
  `github:` / `gitlab:` shorthands may use the HTTPS archive endpoint as a fast path
  since the ref is already resolved to a SHA.
- **No Python pinning.** There is no `python:` field. The environment's interpreter
  is owned by the operator (embedded) or `wengine` itself (standalone); provider
  manifests declare a _compatible range_ in `requires.python`, and `wengine install`
  refuses (loudly) if the interpreter falls outside the intersection of all
  installed sources' ranges.
- **Embedded mode does not police the host env.** Since the operator's environment
  existed first, an inconsistency between it and what a source's setup installs
  surfaces as the _newly installed node_ breaking — not the host application.
  `wengine` reports what `uv` did; it does not try to protect an env it does not own.
- **No "scoped registry" machinery.** One `wengine` process serves one engine
  project; populating the process node registry from that project's `nodes:` map
  is the scoping. Precedence on a name clash: explicit entry > glob entry > builtin.
- **No Pydantic discriminated union.** Deserialization goes through the engine's
  mutable node registry, which distributed sources extend.
- **No strict `--frozen` install mode** (overkill for now); plain `wengine install`
  with no args syncs the environment to `engine.lock`.
- **Local `path:` sources are never locked** — editable-install semantics; the
  path is the pin.
- **Both explicit and glob `nodes:` entries are supported** — explicit entries
  enable `--as` / rename-on-install; globs (`"acme/scrapers:*"`) keep large
  bundles from bloating `engine.yaml`.

## Open questions

1. **Uninstalling a node / source.** Removing entries from `engine.yaml` /
   `engine.lock` is trivial; cleaning up what a source's `setup` installed into a
   shared environment is not (especially embedded mode — you can't just delete the
   env). Not needed yet — flagged so it is not forgotten.
