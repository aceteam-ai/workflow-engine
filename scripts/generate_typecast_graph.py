"""Generate a visualization of the Value typecasting graph."""

from pathlib import Path

import networkx as nx
import pydot

# Import all value types to trigger registration
import workflow_engine.core.values  # noqa: F401
from workflow_engine.core.values.value import value_type_registry


def main():
    # Build directed graph
    G = nx.DiGraph()

    for source_name, source_cls in value_type_registry.types.items():
        G.add_node(source_name)
        for target_name in source_cls._get_casters().keys():
            if target_name in value_type_registry.types:
                G.add_edge(source_name, target_name)

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
