"""Generate a visualization of the Value typecasting graph."""

from collections.abc import Mapping
from pathlib import Path

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
    for source_name, source_cls in all_classes.items():
        G.add_node(source_name)
        for target_name in source_cls._get_casters().keys():
            target_cls = all_classes.get(target_name)
            if target_cls is not None and source_cls.can_cast_to(target_cls):
                G.add_edge(source_name, target_name)
    return G


def main():
    all_classes = dict(ValueRegistry.DEFAULT.all_value_classes())
    G = build_typecast_graph(all_classes)

    # Export to SVG via pydot
    output_path = Path(__file__).parent.parent / "docs" / "typecast_graph.svg"
    output_path.parent.mkdir(exist_ok=True)

    pydot_graph: pydot.Dot = nx.drawing.nx_pydot.to_pydot(G)
    # NOTE: pydot is horrible with types because it generates methods at runtime
    pydot_graph.set_rankdir(  # pyright: ignore[reportAttributeAccessIssue]
        "LR"
    )  # Left to right layout
    pydot_graph.write_svg(str(output_path))  # pyright: ignore[reportAttributeAccessIssue]
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
