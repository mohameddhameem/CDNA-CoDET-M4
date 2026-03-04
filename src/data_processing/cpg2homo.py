import json
import torch
import os
import shutil
import xml.etree.ElementTree as ET
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from multiprocessing import Pool
from utils.tokenizer import encode_texts, load_codebert

# ==========================================
# 1. Helper Function: Parse GraphML XML
# ==========================================
def parse_graphml_str(graphml_content):
    """
    Parses a GraphML XML string into node and edge lists.
    Returns standard Python lists (no Tensors).
    """
    if not graphml_content:
        return None, None, None

    try:
        root = ET.fromstring(graphml_content)
    except ET.ParseError:
        return None, None, None

    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    graph = root.find('g:graph', ns)
    if graph is None:
        return None, None, None

    # --- Part A: Parse Nodes ---
    node_map = {}
    node_texts = []
    
    def _clean_val(val):
        if val is None: return ""
        return str(val).replace('\n', '\\n').replace("'", "\\'")

    for i, node in enumerate(graph.findall('g:node', ns)):
        node_id = node.get('id')
        node_map[node_id] = i
        
        label_elem = node.find("g:data[@key='labelV']", ns)
        label = label_elem.text if label_elem is not None else "UNKNOWN"
        
        attrs = []
        for data in node.findall('g:data', ns):
            key = data.get('key')
            val = data.text or ""
            if key == 'labelV' or not val: continue
            attrs.append((key, _clean_val(val)))
            
        attrs.sort(key=lambda x: x[0])
        attr_str = ", ".join([f"{k}='{v}'" for k, v in attrs])
        
        if attr_str:
            node_texts.append(f"[{label}] {attr_str}")
        else:
            node_texts.append(f"[{label}]")

    if not node_texts: return None, None, None

    # --- Part B: Parse Edges ---
    src, dst = [], []
    edge_texts = [] 
    
    for edge in graph.findall('g:edge', ns):
        s, t = edge.get('source'), edge.get('target')
        
        if s in node_map and t in node_map:
            src.append(node_map[s])
            dst.append(node_map[t])
            
            label_elem = edge.find("g:data[@key='labelE']", ns)
            edge_label = label_elem.text if label_elem is not None else "edge"
            
            edge_attrs = []
            for data in edge.findall('g:data', ns):
                key = data.get('key')
                val = data.text or ""
                if key == 'labelE' or not val: continue
                edge_attrs.append((key, _clean_val(val)))
            
            edge_attrs.sort(key=lambda x: x[0])
            attr_str = ", ".join([f"{k}='{v}'" for k, v in edge_attrs])
            
            if attr_str:
                edge_texts.append(f"[{edge_label}] {attr_str}")
            else:
                edge_texts.append(f"[{edge_label}]")

    # Return pure lists/tuples
    return node_texts, (src, dst), edge_texts


# ==========================================
# 2. Worker Function (Pure Python - No Tensors)
# ==========================================
def _process_single_record(line_data):
    """
    Worker function to process one line.
    Crucial Change: Returns pure Python Dictionary. No PyTorch Tensors here.
    This avoids the 'RuntimeError: unable to mmap' when using many workers.
    """
    line, _ = line_data 
    try:
        record = json.loads(line)
        idx = record.get('idx', -1)
        graphml = record.get('graphml', '')
        
        if not graphml:
            return None, 'empty'

        # Parse XML -> Pure Python Lists
        n_txt, edges, e_txt = parse_graphml_str(graphml)
        
        if not n_txt:
            return None, 'parse_failed'
        
        # Pack raw graph data into a dictionary
        # We split edges tuple (src, dst) for easier handling
        raw_graph = {
            # 'num_nodes': len(n_txt),
            'node_texts': n_txt,
            'edge_src': edges[0],
            'edge_dst': edges[1],
            'edge_texts': e_txt,
        }
        
        # Pack metadata and enforce required fields
        metadata = {
            'target': record.get('target'),
            'model': record.get('model'),
            'language': record.get('language'),
            'source': record.get('source'),
            'hash': record.get('hash'),
            'split': record.get('split'),
            'code': record.get('code'),
            'dataset_idx': idx
        }

        required_fields = ['target', 'model', 'language', 'source', 'hash', 'split', 'code']
        for field in required_fields:
            if metadata[field] is None:
                raise ValueError(f"Missing required field '{field}' for record idx={idx}")
        
        return (idx, raw_graph, metadata), 'success'
        
    except Exception as e:
        return None, f'error: {str(e)}'


# ==========================================
# 3. Dataset Class
# ==========================================
class CPGHomoDataset(InMemoryDataset):
    def __init__(self, root='./CPG', transform=None, pre_transform=None, 
                 force_reload=False, num_workers=1):
        
        self.force_reload = force_reload
        self.num_workers = num_workers if num_workers is not None else 1
        
        if force_reload:
            processed_dir = os.path.join(root, 'processed_homo')
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
        return os.path.join(self.root, 'processed_homo')

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def process(self):
        # Locate the raw file
        raw_path = os.path.join(self.root, 'raw', self.raw_file_names[0])
        if not os.path.exists(raw_path):
            raw_path = os.path.join(self.root, self.raw_file_names[0])
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"File not found: {raw_path}")

        print(f"Processing file: {raw_path}")

        # 1. Count total lines
        print("Counting lines...")
        with open(raw_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        # 2. Generator
        def line_generator(path):
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    yield (line, i)

        temp_storage = {}
        skipped_count = 0

        # 3. Setup Multiprocessing
        # High-Performance Config for 128 cores
        active_workers = min(120, self.num_workers) if self.num_workers > 0 else 1
        chunk_size = 64 
        tokenizer, model, device = load_codebert()
        
        print(f"High-Performance Mode: Parsing {total_lines} graphs with {active_workers} workers...")
        
        # We use the pure python worker function
        worker_fn = _process_single_record
        
        if active_workers > 1:
            with Pool(processes=active_workers) as pool:
                # Use imap_unordered for max speed
                results_iter = pool.imap_unordered(worker_fn, line_generator(raw_path), chunksize=chunk_size)
                
                for result, status in tqdm(results_iter, total=total_lines, desc="Processing"):
                    if status == 'success' and result is not None:
                        idx, raw_graph, metadata = result
                        
                        edge_index = torch.tensor(
                            [raw_graph['edge_src'], raw_graph['edge_dst']], 
                            dtype=torch.long
                        )

                        node_emb = encode_texts(
                            raw_graph['node_texts'],
                            tokenizer,
                            model,
                            device=device,
                            batch_size=128,
                            max_length=512,
                            pool=True,
                        ) if raw_graph['node_texts'] else torch.empty(0)

                        edge_emb = encode_texts(
                            raw_graph['edge_texts'],
                            tokenizer,
                            model,
                            device=device,
                            batch_size=128,
                            max_length=512,
                            pool=True,
                        ) if raw_graph['edge_texts'] else torch.empty(0)

                        data = Data(
                            edge_index=edge_index,
                            # num_nodes=raw_graph['num_nodes'],
                            # node_texts=raw_graph['node_texts'],
                            # edge_texts=raw_graph['edge_texts'],
                            x=node_emb,
                            edge_attr=edge_emb,
                            target=metadata['target'],
                            model=metadata['model'],
                            language=metadata['language'],
                            source=metadata['source'],
                            hash=metadata['hash'],
                            split=metadata['split'],
                            code=metadata['code'],
                            dataset_idx=metadata['dataset_idx']
                        )

                        # split = str(metadata['split']).lower()
                        # data.train_mask = torch.tensor([split == 'train'], dtype=torch.bool)
                        # data.val_mask = torch.tensor([split in ('val', 'validation')], dtype=torch.bool)
                        # data.test_mask = torch.tensor([split == 'test'], dtype=torch.bool)
                        
                        # Apply pre_transform in Main Process (if any)
                        if self.pre_transform:
                            data = self.pre_transform(data)
                            
                        temp_storage[idx] = data
                    else:
                        skipped_count += 1
        else:
            # Single-process fallback
            for line_data in tqdm(line_generator(raw_path), total=total_lines, desc="Processing"):
                result, status = worker_fn(line_data)
                if status == 'success':
                    idx, raw_graph, metadata = result
                    
                    edge_index = torch.tensor(
                        [raw_graph['edge_src'], raw_graph['edge_dst']], 
                        dtype=torch.long
                    )

                    node_emb = encode_texts(
                        raw_graph['node_texts'],
                        tokenizer,
                        model,
                        device=device,
                        batch_size=128,
                        max_length=256,
                        pool=True,
                    ) if raw_graph['node_texts'] else torch.empty(0)

                    edge_emb = encode_texts(
                        raw_graph['edge_texts'],
                        tokenizer,
                        model,
                        device=device,
                        batch_size=128,
                        max_length=256,
                        pool=True,
                    ) if raw_graph['edge_texts'] else torch.empty(0)

                    data = Data(
                        edge_index=edge_index,
                        # num_nodes=raw_graph['num_nodes'],
                        # node_texts=raw_graph['node_texts'],
                        # edge_texts=raw_graph['edge_texts'],
                        x=node_emb,
                        edge_attr=edge_emb,
                        target=metadata['target'],
                        model=metadata['model'],
                        language=metadata['language'],
                        source=metadata['source'],
                        hash=metadata['hash'],
                        split=metadata['split'],
                        code=metadata['code'],
                        dataset_idx=metadata['dataset_idx']
                    )

                    # split = str(metadata['split']).lower()
                    # data.train_mask = torch.tensor([split == 'train'], dtype=torch.bool)
                    # data.val_mask = torch.tensor([split in ('val', 'validation')], dtype=torch.bool)
                    # data.test_mask = torch.tensor([split == 'test'], dtype=torch.bool)
                    
                    if self.pre_transform:
                        data = self.pre_transform(data)
                        
                    temp_storage[idx] = data
                else:
                    skipped_count += 1

        if skipped_count > 0:
            print(f"Warning: Skipped {skipped_count} invalid graphs.")

        # 4. Sort and Save
        print("Sorting and Collating data...")
        sorted_indices = sorted(temp_storage.keys())
        data_list = [temp_storage[i] for i in sorted_indices]

        if not data_list:
            print("Error: No valid data found!")
            return

        # Collate (Standard PyG collate for Homo graphs works fine usually)
        # But we still check for consistency implicitly by sorted keys
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Success! Saved to {self.processed_paths[0]}")


# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=64, help='Number of CPU workers')
    parser.add_argument('--path', type=str, default="CPG", help='Path to CPG directory')
    args = parser.parse_args()
    
    dataset = CPGHomoDataset(
        root=f'./{args.path}',
        force_reload=True,
        num_workers=args.workers
    )