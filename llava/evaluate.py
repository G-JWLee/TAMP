import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.pruners.wanda_pruner import LLaVALayerWandaPruner
from llava.pruners.sparsegpt_pruner import LLaVALayerSparseGPTPruner
from llava.pruners.magnitude_pruner import LLaVALayerMagnitudePruner
from llava.pruners.data_loader import create_data_loader
from llava import conversation as conversation_lib

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='LVLM model')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--task_split_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--is_multimodal", action="store_true")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")  # version
    
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')  # batch size
    parser.add_argument('--llm_sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--vit_sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt", "magnitude"])
    parser.add_argument("--sample_select", default="random", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    # parser.add_argument('--approach_for_sparsity', type=str, default=None)
    # parser.add_argument('--num_samples_for_first_stage', type=int, default=128)
    # parser.add_argument('--aggregate_method', type=str, default="avg")
    parser.add_argument('--sparsity_ratio_granularity', type=str, default=None)
    parser.add_argument('--max_sparsity_per_layer', type=float, default=0.8) # Future part
    parser.add_argument('--score_method', type=str, default="GradOnly_sum")
    parser.add_argument('--num_noise', type=int, default=1)
    parser.add_argument('--sparsity_dict', type=str, default=None)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--dataset_type', type=str, default=None)
    parser.add_argument('--token_selection', type=str, default='naive', 
                        choices=["naive",
                                 "amia"])
    
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.llm_sparsity_ratio == 0.5 or args.vit_sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )

    args.image_grid_pinpoints = model.config.image_grid_pinpoints
    args.image_aspect_ratio = model.config.image_aspect_ratio
    args.mm_use_im_start_end = model.config.mm_use_im_start_end
    args.is_multimodal = True

    # Dataloader
    # System prompt for LVLM
    if args.conv_mode in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    dataloader = create_data_loader(tokenizer=tokenizer, image_processor=image_processor,
                                    data_args=args)
    
    if args.llm_sparsity_ratio != 0 or args.vit_sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            pruner = LLaVALayerWandaPruner(
                model=model,
                data_loader=dataloader,
                llm_sparsity_ratio=args.llm_sparsity_ratio,
                vit_sparsity_ratio=args.vit_sparsity_ratio,
                llm_pruning_method=args.prune_method,
                vit_pruning_method=args.prune_method,
                num_samples=args.nsamples,
                prune_n=prune_n,
                prune_m=prune_m,
                vit_model_prefix='model.vision_tower.vision_tower.vision_model.encoder',
                llm_model_prefix='model',
                score_method=args.score_method,
                sparsity_ratio_granularity=args.sparsity_ratio_granularity,
                max_sparsity_per_layer=args.max_sparsity_per_layer,
                num_noise=args.num_noise,
                sparsity_dict=args.sparsity_dict,
                iteration=args.iteration,
                token_selection=args.token_selection,
                )
        elif args.prune_method == "sparsegpt":
            pruner = LLaVALayerSparseGPTPruner(
                model=model,
                data_loader=dataloader,
                llm_sparsity_ratio=args.llm_sparsity_ratio,
                vit_sparsity_ratio=args.vit_sparsity_ratio,
                llm_pruning_method=args.prune_method,
                vit_pruning_method=args.prune_method,
                num_samples=args.nsamples,
                prune_n=prune_n,
                prune_m=prune_m,
                vit_model_prefix='model.vision_tower.vision_tower.vision_model.encoder',
                llm_model_prefix='model',
                score_method=args.score_method,
                sparsity_ratio_granularity=args.sparsity_ratio_granularity,
                max_sparsity_per_layer=args.max_sparsity_per_layer,
                num_noise=args.num_noise,
                sparsity_dict=args.sparsity_dict,
                iteration=args.iteration,
                token_selection=args.token_selection,
            )
        elif args.prune_method == "magnitude":
            print('magnitude based pruning start')
            pruner = LLaVALayerMagnitudePruner(
                model=model,
                data_loader=dataloader,
                llm_sparsity_ratio=args.llm_sparsity_ratio,
                vit_sparsity_ratio=args.vit_sparsity_ratio,
                llm_pruning_method=args.prune_method,
                vit_pruning_method=args.prune_method,
                num_samples=args.nsamples,
                prune_n=prune_n,
                prune_m=prune_m,
                vit_model_prefix='model.vision_tower.vision_tower.vision_model.encoder',
                llm_model_prefix='model',
                score_method=args.score_method,
                sparsity_ratio_granularity=args.sparsity_ratio_granularity,
                max_sparsity_per_layer=args.max_sparsity_per_layer,
                num_noise=args.num_noise,
                sparsity_dict=args.sparsity_dict,
                iteration=args.iteration,
                token_selection=args.token_selection,
            )
        
        model, sparsity_dict = pruner.prune()
    
    distilled_total_size = sum(
        (param != 0).float().sum() for param in model.parameters()
    )
    print(f"Remaining Proportion: {distilled_total_size / orig_total_size * 100}%")

    # Save pruned model
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

        if sparsity_dict is not None and isinstance(sparsity_dict, dict):
            import yaml
            with open(os.path.join(args.save_model,"sparity_dict.yaml"),"w") as f:
                yaml.dump(sparsity_dict, f)

if __name__ == '__main__':
    main()