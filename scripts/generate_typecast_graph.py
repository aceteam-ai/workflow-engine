"""Generate a visualization of the Value typecasting graph."""

import argparse
import shutil
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

import networkx as nx
import pydot

# Import all value types to trigger registration
import workflow_engine.core.values  # noqa: F401
from workflow_engine.core.values.value import ValueRegistry, ValueType


def build_typecast_graph(all_classes: Mapping[str, ValueType]) -> nx.DiGraph:
    """
    Build the directed typecast graph: one node per registered Value class,
    one edge per pair that can actually be cast.

    A registered caster *key* isn't enough on its own: a caster (e.g.
    Result's casters to StringValue/JSONValue, see result.py) can be
    registered purely to shadow a blanket "cast anything" caster and always
    decline by returning None. can_cast_to() actually invokes the caster
    lookup and confirms it produces a real cast, so a declining caster
    correctly produces no edge.
    """
    G = nx.DiGraph()
    for source_name, source_cls in sorted(all_classes.items()):
        G.add_node(source_name)
        for target_name in sorted(source_cls._get_casters()):
            target_cls = all_classes.get(target_name)
            if target_cls is not None and source_cls.can_cast_to(target_cls):
                G.add_edge(source_name, target_name)
    return G


CAPTION = (
    "Registered concrete Value types only. Edges indicate possible casts; values may fail validation.\n"
    "Generics omitted: Result, SequenceValue, StringMapValue, DataValue, ModelValue.\n"
    "See docs/values.md#generic-cast-rules for parameter-dependent assignability."
)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "typecast_graph.svg"


def render_graph(graph: nx.DiGraph) -> bytes:
    """Render the public diagram, failing explicitly when Graphviz is absent."""
    if shutil.which("dot") is None:
        raise RuntimeError(
            "Graphviz 'dot' is required. Install Graphviz and rerun this script."
        )
    pydot_graph: pydot.Dot = nx.drawing.nx_pydot.to_pydot(graph)
    pydot_graph.set("rankdir", "LR")
    pydot_graph.set("label", CAPTION)
    pydot_graph.set("labelloc", "b")
    pydot_graph.set("labeljust", "l")
    return pydot_graph.create_svg()  # pyright: ignore[reportAttributeAccessIssue]


def svg_content(svg: bytes) -> tuple[set[str], set[str], tuple[str, ...]]:
    """Extract meaning, ignoring Graphviz-version-dependent layout and IDs."""
    root = ElementTree.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    def titles(kind: str) -> set[str]:
        return {
            element.text or ""
            for element in root.findall(f".//svg:g[@class='{kind}']/svg:title", ns)
        }

    caption = tuple(
        element.text or ""
        for element in root.findall(".//svg:g[@class='graph']/svg:text", ns)
    )
    return titles("node"), titles("edge"), caption


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed diagram's nodes, edges or caption are stale.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    all_classes = dict(ValueRegistry.DEFAULT.all_value_classes())
    rendered = render_graph(build_typecast_graph(all_classes))
    if args.check:
        if not args.output.exists() or svg_content(
            args.output.read_bytes()
        ) != svg_content(rendered):
            parser.exit(
                1,
                "Typecast diagram is stale. Run uv run python scripts/generate_typecast_graph.py.\n",
            )
        print("Typecast diagram is current.")
    else:
        args.output.parent.mkdir(exist_ok=True)
        args.output.write_bytes(rendered)
        print(f"Generated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
