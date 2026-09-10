# Generic cast visibility (#246)

Decision recorded before implementation: choose option 4. The concrete SVG will carry a visible caption naming Result, SequenceValue, StringMapValue, DataValue and ModelValue and linking to their parameter-dependent rules in docs/values.md. Update both the README caption and the values guide so the graph's scope is apparent wherever it appears.

One unbound generic node cannot honestly encode item or model compatibility. Representative parameterizations suggest coverage they do not provide, while a separate diagram duplicates prose without clarifying the type predicates. Keep the graph focused on registered concrete types and explain generic families in a dedicated table and examples.

Check the committed SVG's semantic content (node titles, edge titles and caption) against a newly built graph in CI. Graphviz version changes alter layout and SVG bytes, so a byte comparison across contributor/CI versions would fail for irrelevant rendering differences. A fresh generator run verifies Graphviz is installed and renders successfully; a semantic comparison catches missing types, stale edges and stale captions. Generation must fail with an actionable error when dot is absent. The check runs in a subprocess to avoid test-only Value registrations contaminating the public graph.
