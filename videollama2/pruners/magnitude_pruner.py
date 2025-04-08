import torch
import torch.nn as nn
import gc
import math
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from time import time
from copy import deepcopy

from videollama2.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity
from videollama2.pruners.utils import print_time

def loss_vision_language(model, samples, cuda_enabled):
    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

    loss_dict = model(samples)
    loss = loss_dict["loss"]

    batch_len = len(samples["text_input"])
    return loss, batch_len


def loss_language(model, samples, cuda_enabled):
    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

    loss_dict = model(samples)
    loss = loss_dict["loss"]

    batch_len = len(samples["text_input"])
    return loss, batch_len


def loss_vision(model, samples, cuda_enabled):
    # cross entropy loss
    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    outputs = model.predict(samples)

    logits = outputs["predictions"] / 100 
    targets = outputs["targets"]

    probs = torch.nn.functional.softmax(logits, -1)
    batch_index = torch.arange(len(targets)).to(targets.device)
    probs = probs[batch_index, targets]
    
    log_probs = probs.log()
    loss = - log_probs.mean()
    batch_len = len(targets)

    return loss, batch_len

def get_module_recursive(base, module_to_process):
    
    if module_to_process == "":
        return base
    
    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    
    base = getattr(base, now)

    return get_module_recursive(base, rest)


def find_layers(module, layers=[nn.Linear], name=''):

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class VideoLLaMA2LayerMagnitudePruner(LayerWiseBasePruner):
    pruner_name = "videollama2_magnitude_pruner"
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        llm_sparsity_ratio=None,
        vit_sparsity_ratio=None,
        aud_sparsity_ratio=None,
        num_samples=64,
        is_global=False,
        llm_model_prefix="t5_model",
        vit_model_prefix="visual_encoder",
        aud_model_prefix="audio_encoder",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        prune_per_model=False,
        prune_n=0,
        prune_m=0,
        token_selection='naive',
        iteration=1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=f"{aud_model_prefix}+{vit_model_prefix}+{llm_model_prefix}",
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            prune_per_model=prune_per_model,
            prune_n=prune_n,
            prune_m=prune_m, 
        )
        
        self.llm_sparsity_ratio = llm_sparsity_ratio
        self.vit_sparsity_ratio = vit_sparsity_ratio
        self.aud_sparsity_ratio = aud_sparsity_ratio

        self.llm_model_prefix = llm_model_prefix
        self.vit_model_prefix = vit_model_prefix
        self.aud_model_prefix = aud_model_prefix

        self.prune_per_model = prune_per_model
        self.iteration = iteration
        
    def compute_importance_scores(self, model, data_loader=None, dict_layers_to_prune={}, loss_func=None):
        return {k: v.data.float().cpu() for k, v in model.named_parameters()}
    
    def get_mask(self, importance_scores, p, max_sparsity_per_layer):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        for k, v in importance_scores.items():
            num_to_set = int(importance_scores[k].numel() * (1 - max_sparsity_per_layer))
            
            if num_to_set > 0:
                threshold, _ = torch.topk(importance_scores[k].flatten(), num_to_set, largest=True)
                threshold = threshold[-1] 

                importance_scores[k][torch.where(v >= threshold)] = torch.finfo(v.dtype).max
        
        # Flatten all tensors and concatenate them
        all_scores = torch.cat([t.flatten() for t in importance_scores.values()])
        
        # Sort and find the threshold
        num_to_zero_out = int(p * all_scores.numel())
        threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
        threshold = threshold[-1]
        
        # Create mask based on threshold
        masks = {}
        for k, v in importance_scores.items():
            masks[k] = (v > threshold).type(v.dtype)
        
        return masks
    
    def get_layerwise_mask(self, importance_scores, p):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        masks = {}
        for k, v in importance_scores.items():
            all_scores = importance_scores[k].flatten()
            num_to_zero_out = int(p * all_scores.numel())
            threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
            threshold = threshold[-1]

            masks[k] = (v > threshold).type(v.dtype)

        return masks
    
    def forward_to_cache(self, model, batch, device):
        return model(batch)
    
    def global_iterative_pruning(self, target_sparsity, dict_layers_to_prune, iteratation=1, max_sparsity_per_layer=1.0):
        
        masks = None
        for i in range(1, iteratation+1):
            p_i = target_sparsity ** (iteratation / i) # Compute modified sparsity for the i^th iteration
            
            importance_measure = self.compute_importance_scores(
                self.model, self.data_loader, dict_layers_to_prune, loss_vision_language
            )
            importance_measure = {k: v for k, v in importance_measure.items() if k in dict_layers_to_prune}
            
            if masks is not None:
                # Apply mask to importance scores (this step is to simulate pruning in iterations)
                for k in importance_measure:
                    importance_measure[k] *= masks[k]

            if self.is_global and not self.prune_per_model:
                # total global
                print("global")
                masks = self.get_mask(importance_measure, p_i, max_sparsity_per_layer)
            elif self.is_global and self.prune_per_model:
                print("model-level global")
                vision_scores = {k: v for k, v in importance_measure.items() if k.startswith(self.vit_model_prefix)}
                language_scores = {k: v for k, v in importance_measure.items() if k.startswith(self.llm_model_prefix)}
                vision_masks = self.get_mask(vision_scores, p_i, max_sparsity_per_layer)
                language_masks = self.get_mask(language_scores, p_i, max_sparsity_per_layer)
                
                vision_masks.update(language_masks)
                masks = vision_masks
            else:
                print("layer-wise")
                masks = self.get_layerwise_mask(importance_measure, p_i)
            
            # prune the model
            for k, v in self.model.named_parameters():
                if k in masks:
                    v.data *= masks[k].type(v.dtype).to(v.device)
                    
            print(f"Step {i}, target sparsity: {p_i:.4f}")
            
        for k, v in self.model.named_parameters():
            print(k, " sparsity: ", (v == 0).float().sum() / v.numel())
        
        return self.model

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)
    
        def check(name, v):
            if len(v.shape) == 2 and ".layers" in name and "relative_attention_bias.weight" not in name:
                if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)) and (not name.startswith(self.aud_model_prefix)) and self.llm_sparsity_ratio != 0:
                    return True
                elif name.startswith(self.vit_model_prefix) and self.vit_sparsity_ratio != 0:
                    return True
                elif name.startswith(self.aud_model_prefix) and self.aud_sparsity_ratio != 0:
                    return True
                else:
                    return False
            return False

        parameters_to_prune = {
            k: v for k, v in self.model.named_parameters() if check(k, v)
        }
        
        sparsity_ratio = self.llm_sparsity_ratio
            
        self.model = self.global_iterative_pruning(
            target_sparsity=sparsity_ratio,  
            dict_layers_to_prune=parameters_to_prune, 
            iteratation=self.iteration, 
            max_sparsity_per_layer=self.max_sparsity_per_layer,  # Ensure the maximum sparsity per layer constraint is met
        )

        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, None