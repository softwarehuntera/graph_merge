"""Functions to generate visual representations of graphs."""

from pathlib import Path

import hvplot.networkx as hvnx
import matplotlib.pyplot as plt
import networkx as nx
from bokeh.models import HoverTool
from bs4 import BeautifulSoup


def _get_graph_attributes(G: nx.Graph) -> dict:
    """Get network plotting attributes from current graph.

    Returns:
        dict: Network plotting attributes.
    """
    attributes = {}
    attributes["node_color"] = [color[1]["color"] for color in G.nodes(data=True)]
    attributes["node_size"] = [node[1]["size"] for node in G.nodes(data=True)]
    attributes["edge_color"] = [edge[2]["color"] for edge in G.edges(data=True)]
    attributes["edge_alpha"] = [edge[2]["alpha"] for edge in G.edges(data=True)]
    return attributes


def save_hvnx_graph(G: nx.Graph, save_path: Path) -> None:
    """Save network as .html.

    Path should include the file name ending in .html.

    Args:
        G: Graph to save.
        save_path (Path): Path to save the plot.
    """
    attributes = _get_graph_attributes(G)
    positions = nx.drawing.nx_agraph.graphviz_layout(G=G, prog="sfdp")

    # Assumes all metdata is present in the first node
    node_attributes = next(iter(data for _, data in G.nodes(data=True))).copy()

    # Suppress plotting attributes from hover text
    node_attributes.pop("color", None)
    node_attributes.pop("alpha", None)
    node_attributes.pop("size", None)
    # hover_metadata = [("uuid", "@index")]
    hover_metadata = [(str(k), f"@{k}") for k in node_attributes]
    # hover_metadata.extend([(str(k), f"@{k}") for k in node_attributes.keys()])

    # Plot nodes
    node_graph = hvnx.draw_networkx_nodes(
        G=G,
        pos=positions,
        node_color=attributes["node_color"],
        node_size=attributes["node_size"],
    ).opts(width=1200, height=1200, tools=[])

    # Plot edges
    edge_graph = hvnx.draw_networkx_edges(
        G=G,
        pos=positions,
        edge_color=attributes["edge_color"],
        edge_alpha=attributes["edge_alpha"],
        # hover_edge_line_alpha=selection_edge_alphas,
        # selection_edge_line_alpha=selection_edge_alphas,
        edge_width=2,
    ).opts(
        width=1200,
        height=1200,
        selection_policy="nodes",
        tools=[HoverTool(tooltips=hover_metadata), "tap"],
    )
    hvnx_graph = node_graph * edge_graph
    hvnx.save(hvnx_graph, filename=save_path)

    # append custom styles
    css = ".bk-root .bk-tooltip{ position: fixed !important; top: 0% !important; left: 0% !important; width: 400px !important; }"  # noqa: E501
    with open(save_path, "r+") as f:
        html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        head = soup.find("head")
        head.style.append(css)

        pretty_html = soup.prettify()

        f.seek(0)
        f.write(pretty_html)
        f.truncate()


def save_nx_graph(G: nx.Graph, save_path: Path) -> None:
    """Save network as .png.

    Path should include the file name ending in .png.

    Args:
        G: Graph to save.
        save_path (Path): Path to save the plot.
    """
    attributes = _get_graph_attributes(G)
    positions = nx.drawing.nx_agraph.graphviz_layout(G=G, prog="sfdp")

    nx.draw(
        G=G,
        pos=positions,
        node_color=attributes["node_color"],
        node_size=attributes["node_size"],
        edge_color=attributes["edge_color"],
    )
    plt.savefig(save_path, format="png")
