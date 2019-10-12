import argparse
import os
import torch
from utils.wrn import WRN40_4
from utils.data_loader import dataloader_cifar100
from helpers import load_checkpoint
from utils.utilities import get_constraint, eval_performance
from utils.add_lsqmodule import add_lsqmodule
from micronet_score import get_micronet_score

def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--data_root', default=None, type=str)

    parser.add_argument('--weight_bits', required=True,  type=int)
    parser.add_argument('--activation_bits', default=0, type=int)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--cem', default=False, action='store_true', help='use cem-based bit-widths')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    constr_activation = get_constraint(args.activation_bits, 'activation')

    net = WRN40_4(quan_first=False,
                  quan_last=False,
                  constr_activation=constr_activation,
                  preactivation=False,
                  bw_act=args.activation_bits)
    test_loader = dataloader_cifar100(args.data_root, split='test', batch_size=args.batch_size)
    add_lsqmodule(net, bit_width=args.weight_bits)

    if args.cem:
        strategy = {"block3.layer.0.conv2": 3,
                    "block3.layer.2.conv1": 3,
                    "block3.layer.3.conv1": 3,
                    "block3.layer.4.conv1": 3,
                    "block3.layer.2.conv2": 3,
                    "block3.layer.1.conv2": 3,
                    "block3.layer.3.conv2": 3,
                    "block3.layer.1.conv1": 3,
                    "block3.layer.5.conv1": 2,
                    "block1.layer.1.conv2": 1}
        act_strategy = {"block3.layer.0.relu2": 3,
                        "block3.layer.2.relu1": 3,
                        "block3.layer.3.relu1": 3,
                        "block3.layer.4.relu1": 3,
                        "block3.layer.2.relu2": 3,
                        "block3.layer.1.relu2": 3,
                        "block3.layer.3.relu2": 3,
                        "block3.layer.1.relu1": 3,
                        "block3.layer.5.relu1": 2,
                        "block1.layer.1.relu2": 1}

        add_lsqmodule(net, bit_width=args.weight_bits, strategy=strategy)

        for name, module in net.named_modules():
            if name in act_strategy:
                if "_in_act_quant" in name or "first_act" in name or "_head_act_quant0" in name or "_head_act_quant1" in name:
                    temp_constr_act = get_constraint(act_strategy[name], 'weight') #symmetric
                else:
                    temp_constr_act = get_constraint(act_strategy[name], 'activation') #asymmetric
                module.constraint = temp_constr_act

    name_weights_old = torch.load(args.model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    load_checkpoint(net, name_weights_new)

    criterion = torch.nn.CrossEntropyLoss()

    score = get_micronet_score(net, args.weight_bits, args.activation_bits, weight_strategy=strategy, activation_strategy=act_strategy,
                           input_res=(3,32,32), baseline_params=36500000, baseline_MAC=10490000000)
    
    # Calculate accuracy
    net = net.cuda()

    quan_perf_epoch = eval_performance(net, test_loader, criterion)
    accuracy = quan_perf_epoch[1]

    print("Accuracy:", accuracy)

main()
