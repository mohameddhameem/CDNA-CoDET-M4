"""Training and inference CLI.

Notes
-----
Parses arguments and dispatches pretrain/downstream workflows.
"""

import argparse
import gc
import os
import torch
import torch.distributed as dist
from exp.exp_pretrain import ExpPretrain
from exp.exp_downstream import ExpDownstream
from root import ROOT_DIR


def main():
    """Parse CLI arguments and run the selected experiment workflow."""
    parser = argparse.ArgumentParser(description='Graph Foundation Model')

    # Basic parameters
    parser.add_argument('--model', type=str, default='full', help='model name')
    parser.add_argument('--train_language', type=str, default='python', choices=['python', 'java', 'cpp', 'all'])
    parser.add_argument('--test_languages', type=str, default='',
                        help='test language(s), e.g. python,java or all; default uses --train_language')
    parser.add_argument('--task_name', type=str, default='pretrain', choices=['pretrain', 'downstream','infer'])
    parser.add_argument('--pattern', type=str, default='pretrain', choices=['pretrain','none'])
    parser.add_argument('--path', type=str, default='CPG')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Device parameters
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='cuda/mps')
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for sampling')
    parser.add_argument('--infer_batch_size', type=int, default=1024, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=50, help='input dimension after dimension alignment')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--edge_dim', type=int, default=0, help='edge embedding dimension for Hetero Methods')
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension for downstream task')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha parameter')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter')
    parser.add_argument('--num_heads', type=int, default=0, help='number of attention heads:gat only')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for contrastive loss')

    parser.add_argument('--checkpoints', type=str, default=str(ROOT_DIR / 'checkpoints'), help='checkpoint path')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training from last checkpoint')
    parser.add_argument('--is_logging', type=bool, default=False, help='whether to log training progress')

    args = parser.parse_args()

    if not args.test_languages:
        args.test_languages = args.train_language

    dataset_root = os.path.join('.', args.path)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if dist.is_initialized():
        dist.destroy_process_group()

    from utils.cpg2hetero import CPGHeteroDataset
    dataset = CPGHeteroDataset(
        root=dataset_root,
        force_reload=False)

    if args.task_name == 'pretrain':
        exp = ExpPretrain(args, dataset=dataset)
        
        # Train model
        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print('Start training...')
        exp.train()
        

    if args.task_name == 'downstream' or args.task_name == 'infer':
        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print(f'Starting {args.task_name.capitalize()} Classification Task...')
        exp = ExpDownstream(args, dataset=dataset)
        if args.task_name == 'downstream':
            exp.train()
        if args.task_name == 'infer':
            exp.test()

        del exp
        torch.cuda.empty_cache()
        gc.collect()

        if not args.use_multi_gpu or (dist.is_initialized() and dist.get_rank() == 0):
            print(f'All Tasks Completed.')

    if args.use_multi_gpu and dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


if __name__ == '__main__':
    main()
