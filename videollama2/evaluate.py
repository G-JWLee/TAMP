import argparse
import os 
import numpy as np
import torch
from importlib.metadata import version

from videollama2 import model_init
from videollama2.utils import disable_torch_init
from videollama2.pruners.data_loader import create_data_loader
 
from videollama2.pruners.wanda_pruner import VideoLLaMA2LayerWandaPruner
from videollama2.pruners.sparsegpt_pruner import VideoLLaMA2LayerSparseGPTPruner
from videollama2.pruners.magnitude_pruner import VideoLLaMA2LayerMagnitudePruner

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
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--is_multimodal", action="store_true")
    
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')  # batch size
    parser.add_argument('--llm_sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--vit_sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--aud_sparsity_ratio', type=float, default=0, help='Sparsity level')
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
                                 "amia",
                                 ])
    
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.llm_sparsity_ratio == 0.5 or args.vit_sparsity_ratio == 0.5 or args.aud_sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Initialize the model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    model, processor, tokenizer = model_init(model_path)
    
    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )

    args.image_size = 384
    args.is_pretraining = False
    args.va = True
    args.is_multimodal = True

    # Dataloader
    dataloader = create_data_loader(tokenizer=tokenizer, processor=processor,
                                    data_args=args)
    
    if args.llm_sparsity_ratio != 0 or args.vit_sparsity_ratio != 0 or args.aud_sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            pruner = VideoLLaMA2LayerWandaPruner(
                model=model,
                data_loader=dataloader,
                llm_sparsity_ratio=args.llm_sparsity_ratio,
                vit_sparsity_ratio=args.vit_sparsity_ratio,
                aud_sparsity_ratio=args.aud_sparsity_ratio,
                llm_pruning_method=args.prune_method,
                vit_pruning_method=args.prune_method,
                aud_pruning_method=args.prune_method,
                num_samples=args.nsamples,
                prune_n=prune_n,
                prune_m=prune_m,
                aud_model_prefix='model.audio_tower.encoder',
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
            pruner = VideoLLaMA2LayerSparseGPTPruner(
                model=model,
                data_loader=dataloader,
                llm_sparsity_ratio=args.llm_sparsity_ratio,
                vit_sparsity_ratio=args.vit_sparsity_ratio,
                aud_sparsity_ratio=args.aud_sparsity_ratio,
                llm_pruning_method=args.prune_method,
                vit_pruning_method=args.prune_method,
                aud_pruning_method=args.prune_method,
                num_samples=args.nsamples,
                prune_n=prune_n,
                prune_m=prune_m,
                aud_model_prefix='model.audio_tower.encoder',
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
            pruner = VideoLLaMA2LayerMagnitudePruner(
                model=model,
                data_loader=dataloader,
                llm_sparsity_ratio=args.llm_sparsity_ratio,
                vit_sparsity_ratio=args.vit_sparsity_ratio,
                aud_sparsity_ratio=args.aud_sparsity_ratio,
                llm_pruning_method=args.prune_method,
                vit_pruning_method=args.prune_method,
                aud_pruning_method=args.prune_method,
                num_samples=args.nsamples,
                prune_n=prune_n,
                prune_m=prune_m,
                aud_model_prefix='model.audio_tower.encoder',
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
        model.save_pretrained(args.save_model, safe_serialization=False)
        tokenizer.save_pretrained(args.save_model, safe_serialization=False)

        if sparsity_dict is not None and isinstance(sparsity_dict, dict):
            import yaml
            with open(os.path.join(args.save_model,"sparity_dict.yaml"),"w") as f:
                yaml.dump(sparsity_dict, f)

if __name__ == '__main__':
    main()