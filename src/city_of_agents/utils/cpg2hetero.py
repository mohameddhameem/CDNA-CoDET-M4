import json
import torch
import os
import shutil
import xml.etree.ElementTree as ET
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==========================================
# 1. Parser: XML -> Python Dict (No Tensors)
# ==========================================
def parse_graphml_to_dict(graphml_content):
    """
    Parses GraphML XML string into a standard Python Dictionary.
    Workers return pure Python lists/dicts to avoid mmap errors.
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

    def _clean_val(val):
        if val is None: return ""
        return str(val).replace('\n', '\\n').replace("'", "\\'")

    for node in graph.findall('g:node', ns):
        node_id = node.get('id')
        label_elem = node.find("g:data[@key='labelV']", ns)
        node_type = label_elem.text if label_elem is not None else "UNKNOWN"
        
        if node_type not in node_storage:
            node_storage[node_type] = []

        attrs = []
        for data in node.findall('g:data', ns):
            key = data.get('key')
            val = data.text
            if key == 'labelV' or not val: continue
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

# ==========================================
# 2. Converter: Dict -> HeteroData (Main Process)
# ==========================================
def dict_to_hetero(raw_dict):
    """
    Converts raw dict to HeteroData. Runs in Main Process.
    """
    data = HeteroData()
    
    # Nodes
    for n_type, texts in raw_dict['nodes'].items():
        data[n_type].num_nodes = len(texts)
        data[n_type].texts = texts

    # Edges
    for (src_t, rel, dst_t), store in raw_dict['edges'].items():
        src = torch.tensor(store["src"], dtype=torch.long)
        dst = torch.tensor(store["dst"], dtype=torch.long)
        data[src_t, rel, dst_t].edge_index = torch.stack([src, dst], dim=0)
        
    return data

# ==========================================
# 3. Worker Function
# ==========================================
def _process_single_record(line_data):
    """
    Worker function. Parses text only.
    """
    line, _ = line_data
    try:
        record = json.loads(line)
        idx = record.get('idx', -1)
        graphml = record.get('graphml', "")
        
        if not graphml:
            return None, 'empty'

        raw_graph = parse_graphml_to_dict(graphml)
        
        if raw_graph is None:
            return None, 'parse_failed'
        
        # Extract metadata (INCLUDING RAW CODE)
        metadata = {
            'target': record.get('target') or 'nan',
            'model': record.get('model') or 'nan',
            'raw_code': record.get('code') or '',
            'dataset_idx': idx,
        }
        
        return (idx, raw_graph, metadata), 'success'
        
    except Exception as e:
        return None, f'error: {str(e)}'


# ==========================================
# 4. Dataset Class
# ==========================================
class CPGHeteroDataset(InMemoryDataset):
    def __init__(self, root='./CPG', transform=None, pre_transform=None, 
                 force_reload=False, num_workers=1):
        
        self.force_reload = force_reload
        self.num_workers = num_workers if num_workers is not None else 1
        
        if force_reload:
            processed_dir = os.path.join(root, 'processed_hetero')
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)
                print(f"Removed processed directory: {processed_dir}")
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cpg_dataset.jsonl']
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_hetero')

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def process(self):
        raw_path = os.path.join(self.root, 'raw', self.raw_file_names[0])
        if not os.path.exists(raw_path):
            raw_path = os.path.join(self.root, self.raw_file_names[0])
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"JSONL file not found at: {raw_path}")
            
        print(f"Processing file: {raw_path}")

        print("Counting lines...")
        with open(raw_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        def line_generator(path):
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    yield (line, i)

        temp_storage = {}
        skipped_count = 0
        all_node_types = set()
        all_edge_types = set()

        # Config
        active_workers = self.num_workers if self.num_workers > 0 else 1

        print(f"High-Performance Mode: Parsing {total_lines} graphs with {active_workers} workers...")

        if active_workers > 1:
            with Pool(processes=active_workers) as pool:
                results_iter = pool.imap_unordered(_process_single_record, line_generator(raw_path), chunksize=128)
                
                for result, status in tqdm(results_iter, total=total_lines, desc="Processing"):
                    if status == 'success' and result is not None:
                        idx, raw_graph, metadata = result
                        
                        # 1. Convert to Tensor (Main Process)
                        data = dict_to_hetero(raw_graph)
                        
                        # 2. Restore Metadata (INCLUDING RAW CODE)
                        data.target = metadata['target']
                        data.model = metadata['model']
                        data.dataset_idx = metadata['dataset_idx']
                        data.raw_code = metadata['raw_code']
                        
                        if self.pre_transform:
                            data = self.pre_transform(data)

                        temp_storage[idx] = data
                        
                        all_node_types.update(data.node_types)
                        all_edge_types.update(data.edge_types)
                    else:
                        skipped_count += 1
        else:
            for line_data in tqdm(line_generator(raw_path), total=total_lines):
                result, status = _process_single_record(line_data)
                if status == 'success':
                    idx, raw_graph, metadata = result
                    data = dict_to_hetero(raw_graph)
                    data.target = metadata['target']
                    data.model = metadata['model']
                    data.dataset_idx = metadata['dataset_idx']
                    data.raw_code = metadata['raw_code']
                    
                    if self.pre_transform: data = self.pre_transform(data)
                    temp_storage[idx] = data
                    all_node_types.update(data.node_types)
                    all_edge_types.update(data.edge_types)
                else:
                    skipped_count += 1

        if skipped_count > 0:
            print(f"Skipped {skipped_count} invalid graphs.")

        # Sorting
        print("Sorting data by index...")
        sorted_indices = sorted(temp_storage.keys())
        data_list = [temp_storage[i] for i in sorted_indices]

        if not data_list:
            print("No valid data found!")
            return

        # Schema Unification
        print(f"Unifying schema ({len(all_node_types)} nodes, {len(all_edge_types)} edges)...")
        for data in tqdm(data_list, desc="Padding graphs"):
            # Padding missing node and edge types
            for n_type in all_node_types:
                if n_type not in data.node_types:
                    data[n_type].num_nodes = 0
                    data[n_type].texts = [] 
            for e_type in all_edge_types:
                if e_type not in data.edge_types:
                    data[e_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        # Collate and Save
        print("Collating data (this may take a moment due to raw code strings)...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Success! Saved to {self.processed_paths[0]}")

    def get(self, idx):
        data = super().get(idx)
        # Slicing for node-level 'texts' list
        for node_type in data.node_types:
            if hasattr(self.data[node_type], 'texts'):
                if node_type in self.slices and 'num_nodes' in self.slices[node_type]:
                    slices = self.slices[node_type]['num_nodes']
                    start, end = slices[idx], slices[idx + 1]
                    data[node_type].texts = self.data[node_type].texts[start:end]
        
        return data

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=64, help='Number of workers')
    args = parser.parse_args()

    dataset = CPGHeteroDataset(
        root='./CPG',
        force_reload=True,
        num_workers=args.workers
    )
