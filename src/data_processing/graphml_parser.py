"""
GraphML Parsing Utilities

Shared functions for parsing Code Property Graph (CPG) GraphML files.
Used by both heterogeneous and homogeneous graph builders.
"""
import xml.etree.ElementTree as ET


def _clean_val(val):
    """Clean and escape attribute values."""
    if val is None:
        return ""
    return str(val).replace('\n', '\\n').replace("'", "\\'")


def parse_graphml_to_dict(graphml_content):
    """
    Parses GraphML XML string into a Python Dictionary.

    Returns:
        dict with 'nodes' and 'edges' keys, organized by type:
        {
            'nodes': {node_type: [attr_str, ...]},
            'edges': {(src_type, edge_label, dst_type): {'src': [], 'dst': []}}
        }

    Note:
        Workers return pure Python lists/dicts to avoid mmap errors in multiprocessing.
    """
    if not graphml_content:
        return None

    try:
        root = ET.fromstring(graphml_content)
    except ET.ParseError:
        return None

    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    graph = root.find('g:graph', ns)
    if graph is None:
        return None

    # --- A. Parse Nodes ---
    node_storage = {}
    id_map = {}

    for node in graph.findall('g:node', ns):
        node_id = node.get('id')
        label_elem = node.find("g:data[@key='labelV']", ns)
        node_type = label_elem.text if label_elem is not None else "UNKNOWN"

        if node_type not in node_storage:
            node_storage[node_type] = []

        attrs = []
        for data in node.findall('g:data', ns):
            key = data.get('key')
            val = data.text or ""
            if key == 'labelV' or not val:
                continue
            attrs.append((key, _clean_val(val)))

        attrs.sort(key=lambda x: x[0])
        attr_str = ", ".join([f"{k}='{v}'" for k, v in attrs])

        current_idx = len(node_storage[node_type])
        node_storage[node_type].append(attr_str)
        id_map[node_id] = (node_type, current_idx)

    if not node_storage:
        return None

    # --- B. Parse Edges ---
    edge_storage = {}

    for edge in graph.findall('g:edge', ns):
        s_xml, t_xml = edge.get('source'), edge.get('target')
        if s_xml not in id_map or t_xml not in id_map:
            continue

        src_type, src_idx = id_map[s_xml]
        dst_type, dst_idx = id_map[t_xml]

        label_elem = edge.find("g:data[@key='labelE']", ns)
        edge_label = label_elem.text if label_elem is not None else "edge"

        edge_key = (src_type, edge_label, dst_type)
        if edge_key not in edge_storage:
            edge_storage[edge_key] = {"src": [], "dst": []}

        edge_storage[edge_key]["src"].append(src_idx)
        edge_storage[edge_key]["dst"].append(dst_idx)

    return {
        "nodes": node_storage,
        "edges": edge_storage
    }
