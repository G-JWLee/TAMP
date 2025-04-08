import torch
import torch.nn as nn
import numpy as np
import sys
import os
import math
import copy
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from llava.pruners.utils import print_time
from llava.pruners.base_pruner import BasePruner


def cos_pairwise_density(embeddings, image_mask):
    v_embeddings = embeddings[image_mask]
    v_embeddings = v_embeddings / v_embeddings.norm(dim=1, keepdim=True)
    l_embeddings = embeddings[~image_mask]
    l_embeddings = l_embeddings / l_embeddings.norm(dim=1, keepdim=True)

    v_distances = torch.mm(v_embeddings, v_embeddings.T)
    v_upper_triangular = v_distances.triu(diagonal=1) # Only take upper triangular distances (no duplicates)
    v_mean_dist = v_upper_triangular[v_upper_triangular > 0].mean().item()

    l_distances = torch.mm(l_embeddings, l_embeddings.T)
    l_upper_triangular = l_distances.triu(diagonal=1) # Only take upper triangular distances (no duplicates)
    l_mean_dist = l_upper_triangular[l_upper_triangular > 0].mean().item()

    vl_dist = torch.mm(v_embeddings, l_embeddings.T)
    vl_mean_dist = vl_dist.mean().item()

    return v_mean_dist, l_mean_dist, vl_mean_dist

def cos_pairwise_total_density(embeddings):
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    distances = torch.mm(embeddings, embeddings.T)
    upper_triangular = distances.triu(diagonal=1) # Only take upper triangular distances (no duplicates)
    mean_dist = upper_triangular[upper_triangular > 0].mean().item()

    return mean_dist

def get_module_recursive(base, module_to_process):
    
    if module_to_process == "":
        return base
    
    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    base = getattr(base, now)

    return get_module_recursive(base, rest)


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)

        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out, image_mask=None, score=None, previous_score=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        inp_l = inp[~image_mask]
        inp_v = inp[image_mask]

        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        

class ActivationDensity:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device

        self.v_density = 0
        self.l_density = 0
        self.vl_dist = 0

        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out, image_mask=None, score=None):
        assert len(out) == 1
        v_density, l_density, vl_dist = cos_pairwise_density(out[0], image_mask=image_mask[0])

        self.v_density *= self.nsamples / (self.nsamples+1)
        self.l_density *= self.nsamples / (self.nsamples+1)
        self.vl_dist *= self.nsamples / (self.nsamples+1)

        self.nsamples += 1

        self.v_density += v_density / self.nsamples
        self.l_density += l_density / self.nsamples
        self.vl_dist += vl_dist / self.nsamples


def prepare_calibration_input_encoder(model, dataloader, model_prefix, n_samples, module_to_process="encoder.block"):
    use_cache = getattr(model, model_prefix).config.use_cache
    getattr(model, model_prefix).config.use_cache = False
    # layers = model.encoder.block
    layers = get_module_recursive(model, module_to_process)

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
    inps = []
    # TODO: make this max_leng to make zero-padding
    
    caches = []
    keys_to_cache = [
        "attention_mask", "position_ids", "cache_position",
    ]
    image_masks = []
        
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps.append(inp)                
            cache = {}
            for k in keys_to_cache:
                cache[k] = kwargs[k]
            caches.append(cache)
            raise ValueError

    layers[0] = Catcher(layers[0])
    total_samples = 0
    for i, batch in enumerate(dataloader):
        batch['images'] = [item.to(dtype=torch.float16, device='cuda', non_blocking=True) for item in batch['images']]
        batch['input_ids'] = batch['input_ids'].to(device='cuda', non_blocking=True)
        batch['labels'] = batch['labels'].to(device='cuda', non_blocking=True)
        batch['attention_mask'] = batch['attention_mask'].to(device='cuda', non_blocking=True)

        if total_samples >= n_samples:
            break
        total_samples += len(batch['input_ids'])

        try:
            model(**batch)
        except ValueError:
            image_masks.append(model.temp_label)
            pass
    layers[0] = layers[0].module
    outs = [None] * len(inps)

    getattr(model, model_prefix).config.use_cache = use_cache

    return inps, outs, caches, image_masks


class LayerWiseBasePruner(BasePruner):
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        model_prefix="t5_model",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="GradMagSquare_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        prune_n=0, 
        prune_m=0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
        )

        self.sparsity_ratio_granularity = sparsity_ratio_granularity
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.score_method = score_method
        self.num_data_first_stage = num_data_first_stage
        self.num_noise = num_noise
        self.sparsity_dict = sparsity_dict
        self.noise_eps = noise_eps
        self.prune_per_model=prune_per_model

        self.prune_spec = prune_spec
        self.model_prefix = model_prefix
        self.prune_n, self.prune_m = prune_n, prune_m
        
        self.model_stem = getattr(self.model, model_prefix, None) # self.model.t5_model, self.model.visual, etc
        
    def compute_importance_scores(self, model, data_loader, loss_func):
        raise NotImplementedError

    def get_params(self, model):
        params = []
        names = []

        for name, param in model.named_parameters():
            names.append(name)
            params.append(param)

        return names, params

    def model_setup_and_record_attributes(self, model):
        dtype_record = {}
        requires_grad_record = {}
        # for n, p in model.state_dict().items():
        for n, p in model.named_parameters():
            dtype_record[n] = p.data.dtype
            # p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in model.named_parameters():
            requires_grad_record[n] = p.requires_grad
            p.requires_grad = True

        device = next(iter(model.parameters())).device
        # self.model.to("cpu")

        return dtype_record, requires_grad_record, device

    def model_reset(self, model, dtype_record, requires_grad_record, device):
        # set to original requires grad
        for n, p in model.named_parameters():
            p.requires_grad = requires_grad_record[n]

        # for n, p in model.state_dict().items():
        for n, p in model.named_parameters():
            p.data = p.data.type(dtype_record[n])
            
        model.to(device)
            
    def convert_spec_to_list(self, spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = spec.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        return num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio
    
    def create_pruned_arch(self, *args, **kwargs):
        return NotImplementedError


class LayerSparsity:
    def __init__(
            self, 
            model, 
            data_loader, 
            loss_func, 
            num_samples, 
            original_sparsity, 
            max_sparsity_per_layer=0.8, 
            score_method="GradMagSquare_avg", 
            num_noise=1, 
            noise_eps=1e-3, 
            layer_to_group_mapping={}, 
            prune_per_model=False,
            per_model_group=[],
        ):
        self.importance_measure = {}
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.num_samples = num_samples
        self.original_sparsity = original_sparsity
        self.layer_to_group_mapping = layer_to_group_mapping
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.num_noise = num_noise
        self.noise_eps = noise_eps
        self.prune_per_model = prune_per_model
        
        self.score_method = score_method
        self.per_model_group = per_model_group
        
        if score_method is not None:
            self.score_compute, self.score_aggregate = score_method.split("_")
        
        assert self.max_sparsity_per_layer >= self.original_sparsity
        
    def get_mask(self, importance_scores, p, max_sparsity_per_layer):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        for k, v in importance_scores.items():
            num_to_set = int(importance_scores[k].numel() * (1 - max_sparsity_per_layer))
            
            if num_to_set > 0:
                threshold, _ = torch.topk(importance_scores[k].flatten(), num_to_set, largest=True)
                threshold = threshold[-1] # take the last value

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
            all_scores = importance_scores[k].flatten().cuda()
            num_to_zero_out = int(p * all_scores.numel())
            threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
            threshold = threshold[-1].cpu()

            masks[k] = (v > threshold).type(v.dtype)

        return masks
        
    def global_iterative_pruning(self, target_sparsity, dict_layers_to_prune, iteratation=1, max_sparsity_per_layer=1.0):
        
        weight_copy = {}
        total_parameters = 0
        names = []
        params = []
        for k, v in self.model.named_parameters():  
            if k in dict_layers_to_prune:
                names.append(k)
                params.append(v)
                weight_copy[k] = torch.clone(v).cpu()

        masks = None
        for i in range(1, iteratation+1):
            p_i = target_sparsity ** (iteratation / i) # Compute modified sparsity for the i^th iteration
            
            importance_measure = self.compute_importance_scores(
                dict_layers_to_prune
            )
            
            importance_measure = {k: v for k, v in importance_measure.items() if k in dict_layers_to_prune}
            
            if masks is not None:
                # Apply mask to importance scores (this step is to simulate pruning in iterations)
                for k in importance_measure:
                    importance_measure[k] *= masks[k]

            print("global")
            masks = self.get_mask(importance_measure, p_i, max_sparsity_per_layer)
            # prune the model
            for k, v in self.model.named_parameters():
                if k in masks:
                    v.data *= masks[k].type(v.dtype).to(v.device)
                    
            print(f"Step {i}, target sparsity: {p_i:.4f}")
        
        sparsity_dict = {}
        for k, v in self.model.named_parameters():
            sparsity_dict[k] = ((v == 0).float().sum() / v.numel()).item()
            
        for k, p in zip(names, params):
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            p.data = weight_copy[k].to(p.device)
        
        return sparsity_dict
    
    @print_time
    def return_sparsity(self):
        original_sparsity = self.original_sparsity
        layer_to_group_mapping = self.layer_to_group_mapping
        # print(f"layer_to_group_mapping: {layer_to_group_mapping}")

        if self.score_compute.startswith("real"):
            return self.global_iterative_pruning(
                original_sparsity, layer_to_group_mapping, iteratation=3, max_sparsity_per_layer=1.0
            )

        if layer_to_group_mapping is None or len(layer_to_group_mapping) == 0:
            class uniform_sparsity_module:
                def __getitem__(self, key):
                    return original_sparsity
            return uniform_sparsity_module()

        # compute the global information
        if len(self.importance_measure) == 0:
            if self.score_compute.startswith("mezo"):
                self.importance_measure = self.compute_importance_scores_mezo_diff(layer_to_group_mapping) # update
            elif self.score_compute.startswith("lmezo"):
                self.importance_measure = self.compute_importance_scores_mezo_layer(layer_to_group_mapping) # zeroth-order, fixed samples and noises
            elif self.score_compute.startswith("olmezo"):
                self.importance_measure = self.compute_importance_scores_mezo_layer_one(layer_to_group_mapping) # zeroth-order
            elif self.score_compute.startswith("outlier"):
                self.importance_measure = self.compute_outlier(layer_to_group_mapping)
                layer_sparsity = self.importance_measure
                print(f"layer_sparsity: {layer_sparsity}")
                return layer_sparsity
            elif self.score_compute.startswith("density"):
                self.importance_measure = self.compute_density(layer_to_group_mapping)
            else:
                self.importance_measure = self.compute_importance_scores(layer_to_group_mapping) # first-order

        # create the layer list that for each group
        group_to_layer_mapping = {}
        for k, v in layer_to_group_mapping.items():
            if v not in group_to_layer_mapping:
                group_to_layer_mapping[v] = []

            group_to_layer_mapping[v].append(k)
        
        # store the num of parameters for each group and the total paramters
        num_parameters_dict = {}
        total_parameters = 0
        for k, v in self.model.named_parameters():
            if k in layer_to_group_mapping:
                num_parameters_dict[k] = v.numel()
                total_parameters += v.numel()
        
        # total params to keep
        total_parameters_to_keep = int(total_parameters * (1 - original_sparsity))
        
        # store the importance per parameter for each group
        group_scores = {}
        group_num_parameters = {}
        for group_name, layers in group_to_layer_mapping.items():
            if group_name not in group_scores:
                group_scores[group_name] = 0
            
            num_params = 0
            for l in layers:
                group_scores[group_name] += self.importance_measure[l].sum()
                
                num_params += num_parameters_dict[l]
            
            if self.score_aggregate == "avg":
                group_scores[group_name] /= num_params # normalization
            
            group_num_parameters[group_name] = num_params
            
        def compute_the_sparsity_per_group(total_parameters_to_keep, group_scores, group_num_parameters, max_sparsity_per_layer=0.8):
            scores = torch.FloatTensor(list(group_scores.values()))
            num_parameters = torch.LongTensor(list(group_num_parameters.values()))
            
            parameters_to_keep_per_group = torch.zeros_like(scores, dtype=int)
            
            parameters_to_keep_per_group += torch.ceil(num_parameters * (1 - max_sparsity_per_layer)).int() # to gaurantee the max_sparsity
            
            while parameters_to_keep_per_group.sum() < total_parameters_to_keep:
                total_ratio = torch.sum(scores)
                
                rest_total_parameters_to_keep = total_parameters_to_keep - parameters_to_keep_per_group.sum()
                
                parameters_to_add = torch.ceil((scores / total_ratio) * rest_total_parameters_to_keep)
                
                parameters_to_keep_per_group = parameters_to_keep_per_group + parameters_to_add
                
                scores[parameters_to_keep_per_group >= num_parameters] = 0 # make sure they are not going to add more parameters
                
                parameters_to_keep_per_group = torch.clamp(parameters_to_keep_per_group, max=num_parameters) # remove the extra parameters

                # the following codes are optional
                # they are to make sure the sum of parameters_to_keep_per_group is EXACTLY the same as total_parameters_to_keep
                if parameters_to_add.sum() == 0: # for some reason the algo cannot add more parameters
                    # the algo stuck
                    current_sum = parameters_to_keep_per_group.sum()
                    if current_sum < total_parameters_to_keep:
                        num_need_to_add = total_parameters_to_keep - current_sum
                        
                        while num_need_to_add > 0:
                            # distributed the parameters to the rest of groups
                            for index in torch.where(scores > 0)[0]:
                                parameters_can_add = min(
                                    num_need_to_add, num_parameters[index] - parameters_to_keep_per_group[index]
                                )
                                parameters_to_keep_per_group[index] += parameters_can_add
                                
                                num_need_to_add -= parameters_can_add
                                
                                if num_need_to_add == 0:
                                    break
                                
                if parameters_to_keep_per_group.sum() > total_parameters_to_keep: # for some reason the algo cannot add more parameters
                    # the algo stuck
                    current_sum = parameters_to_keep_per_group.sum()

                    num_need_to_remove = current_sum - total_parameters_to_keep
                    
                    while num_need_to_remove > 0:
                        # remove the parameters from full groups
                        for index in torch.argsort(parameters_to_keep_per_group, descending=True, stable=True):
                            parameters_can_remove = min(
                                num_need_to_remove, 
                                parameters_to_keep_per_group[index] - (num_parameters[index] * (1 - max_sparsity_per_layer)).int() # extra parameters
                            )
                            parameters_to_keep_per_group[index] += parameters_can_remove
                            
                            num_need_to_remove -= parameters_can_remove
                            
                            if num_need_to_remove == 0:
                                break
                            
                ############################### Optional codes end here
            
            # convert the group parameters to keep to sparsity    
            group_sparsity = {}
            
            for k, param_to_keep, group_max_param in zip(group_num_parameters.keys(), parameters_to_keep_per_group, num_parameters):
                group_sparsity[k] = torch.clamp(1 - param_to_keep / group_max_param, min=0, max=1).item()
                
            return group_sparsity
        
        if self.prune_per_model:
            group_sparsity = {}
            for i in range(len(self.per_model_group)):
                submodel_prefix = self.per_model_group[i]
                sparsity = self.per_model_sparsity[i]
                print(submodel_prefix)
                submodel_group_scores = {k: v for k, v in group_scores.items() if k.startswith(submodel_prefix)}
                submodel_group_num_parameters = {k: v for k, v in group_num_parameters.items() if k.startswith(submodel_prefix)}
                
                submodel_total_parameters_to_keep = int(sum(list(submodel_group_num_parameters.values())) * (1 - sparsity))
                submodel_group_sparsity = compute_the_sparsity_per_group(
                    submodel_total_parameters_to_keep, 
                    submodel_group_scores, 
                    submodel_group_num_parameters, 
                    max_sparsity_per_layer=self.max_sparsity_per_layer,
                )
                group_sparsity.update(submodel_group_sparsity)
        else:
            group_sparsity = compute_the_sparsity_per_group(
                total_parameters_to_keep, 
                group_scores, 
                group_num_parameters, 
                max_sparsity_per_layer=self.max_sparsity_per_layer,
            )
        
        compute_total_keep_parameters = 0
        for k in group_num_parameters:
            compute_total_keep_parameters += (1 - group_sparsity[k]) * group_num_parameters[k]
        
        # for checking
        print(f"compute_total_keep_parameters: {compute_total_keep_parameters}, total_parameters_to_keep: {total_parameters_to_keep}")
        
        # import pdb; pdb.set_trace()
        
        # print(group_scores)
        # print(group_num_parameters)
        # print(group_sparsity)
        
        layer_sparsity = {
            k: group_sparsity[v]
            for k, v in layer_to_group_mapping.items()
        }
        print(f"layer_sparsity: {layer_sparsity}")
        return layer_sparsity

    @print_time
    def compute_importance_scores(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        names = []
        params = []
        for k, v in model.named_parameters():
            if k in layer_to_group_mapping:
                names.append(k)
                params.append(v)
            
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        for d in data_loader:
            d['images'][0] = d['images'][0].to(dtype=torch.float16, device='cuda', non_blocking=True)
            d['input_ids'] = d['input_ids'].to(device='cuda', non_blocking=True)
            d['labels'] = d['labels'].to(device='cuda', non_blocking=True)
            d['attention_mask'] = d['attention_mask'].to(device='cuda', non_blocking=True)
            batch_len = len(d['input_ids'])
            # print(accum_samples)
            if accum_samples >= self.num_samples:
                break
            
            loss = loss_func(model, d)['loss']

            accum_samples += batch_len
            current_batch_index += 1

            grads = torch.autograd.grad(loss, params)
            
            assert len(grads) == len(names) == len(params)

            for k, v in zip(names, grads):
                
                if self.score_compute == "obd":
                    gradients_dict[k] += v.cpu().data.float() ** 2
                else:
                    gradients_dict[k] += v.cpu().data.float().abs()

        for k in names:
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] /= current_batch_index
        
        if "obd" in self.score_compute:
            # using square of magnitude multiplied by diagonal fisher as importance scores
            importance_measure = {k: (v.cpu().data.float() ** 2) * gradients_dict[k] for k, v in zip(names, params)}        # fisher information. 
        elif "aobd" in self.score_compute:
            importance_measure = {k: (v.cpu().data.float().abs()) * gradients_dict[k].abs() for k, v in zip(names, params)} # first order. 
            print(f"importance_measure: {importance_measure}")
        elif "gradient" in self.score_compute:
            importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
        
        return importance_measure
    
    def zo_perturb_parameters(self, params, random_seed=1, scaling_factor=1, zo_eps=1e-3):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed)
        
        for param in params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    def compute_importance_scores_mezo_diff(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        model.eval()

        names = []
        params = []
        weight_copy = {}
        total_parameters = 0
        for k, v in model.named_parameters():  
            if k in layer_to_group_mapping:
                names.append(k)
                params.append(v)
                weight_copy[k] = torch.clone(v).cpu()
                total_parameters += v.numel()
                
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        zo_eps = self.noise_eps
        
        learning_rate = 1 / total_parameters * 1e-3
        
        for d in data_loader:
            d['images'][0] = d['images'][0].to(dtype=torch.float16, device='cuda', non_blocking=True)
            d['input_ids'] = d['input_ids'].to(device='cuda', non_blocking=True)
            d['labels'] = d['labels'].to(device='cuda', non_blocking=True)
            d['attention_mask'] = d['attention_mask'].to(device='cuda', non_blocking=True)
            batch_len = len(d['input_ids'])
            if accum_samples >= self.num_samples:
                break
            
            print(accum_samples)
            if accum_samples >= self.num_samples:
                break
            
            zo_random_seed = np.random.randint(1000000000)
            
            self.zo_perturb_parameters(params, random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)
            with torch.no_grad():
                loss1 = loss_func(model, d)['loss']
            
            self.zo_perturb_parameters(params, random_seed=zo_random_seed, scaling_factor=-2, zo_eps=zo_eps)
            with torch.no_grad():
                loss2 = loss_func(model, d)['loss']
        
            # recover the weight
            self.zo_perturb_parameters(params, random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)

            accum_samples += batch_len
            current_batch_index += 1
            
            projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

            torch.manual_seed(zo_random_seed)
            for k, param in zip(names, params):
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data - projected_grad * z * learning_rate

        for k, p in zip(names, params):
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] = (p.data.cpu() - weight_copy[k]).float().abs() / current_batch_index
            
            p.data = weight_copy[k].to(p.device)
            
            del weight_copy[k]
            
        # using square of magnitude multiplied by diagonal fisher as importance scores

        if self.score_compute == "mezo-gradient":
            importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "mezo-aobd":
            importance_measure = {k: v.cpu().data.float().abs() * gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "mezo-obd":
            importance_measure = {k: v.cpu().data.float() ** 2 * gradients_dict[k] ** 2 for k, v in zip(names, params)}
            
        return importance_measure
    
    def compute_importance_scores_mezo_layer(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        names = []
        params = []
        model.eval()
        for k, v in model.named_parameters():  
            if k in layer_to_group_mapping:
                names.append(k)
                params.append(v)
        
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        zo_eps = self.noise_eps
        
        n_mezo = 4
        
        self.num_samples = 8
        
        for i, (name, param) in enumerate(zip(names, params)):
            print(i, name)
            accum_samples = 0
            current_batch_index = 0
            
            for d in data_loader:
                if accum_samples >= self.num_samples:
                    break
                    
                d['images'][0] = d['images'][0].to(dtype=torch.float16, device='cuda', non_blocking=True)
                d['input_ids'] = d['input_ids'].to(device='cuda', non_blocking=True)
                d['labels'] = d['labels'].to(device='cuda', non_blocking=True)
                d['attention_mask'] = d['attention_mask'].to(device='cuda', non_blocking=True)
                batch_len = len(d['input_ids'])
                
                per_gradients_dict = {name: 0}
                
                for _ in range(n_mezo):
                    
                    if accum_samples >= self.num_samples:
                        break
                    
                    zo_random_seed = np.random.randint(1000000000) # TODO fix the seed
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss1 = loss_func(model, d)["loss"]
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=-2, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss2 = loss_func(model, d)["loss"]
                
                    # recover the weight
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)

                    accum_samples += batch_len
                    current_batch_index += 1
                    
                    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()
                    
                    # print(zo_random_seed, loss1, loss2, projected_grad)

                    # gradients_dict = self.zo_gradients(gradients_dict, names, params, projected_grad, random_seed=zo_random_seed)
                    
                    torch.manual_seed(zo_random_seed)
                    per_gradients_dict[name] += projected_grad
                        
                gradients_dict[name] += torch.FloatTensor([per_gradients_dict[name]]).abs()
                
        print(gradients_dict)
    
        if self.score_compute == "lmezo-gradient":
            importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "lmezo-aobd":
            importance_measure = {k: v.cpu().data.float().abs() * gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "lmezo-obd":
            importance_measure = {k: v.cpu().data.float() ** 2 * gradients_dict[k] ** 2 for k, v in zip(names, params)}
            
        return importance_measure
    
    def compute_importance_scores_mezo_layer_one(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        names = []
        params = []
        model.eval()
        for k, v in model.named_parameters():  
            if k in layer_to_group_mapping:
                names.append(k)
                params.append(v)
        
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        zo_eps = self.noise_eps
        
        n_mezo = self.num_noise
        from tqdm import tqdm
        
        for i, (name, param) in tqdm(enumerate(zip(names, params)), total=len(names)):
            print(i, name)
            accum_samples = 0
            current_batch_index = 0
            
            for d in data_loader:
                if accum_samples >= self.num_samples:
                    break
                    
                d['images'][0] = d['images'][0].to(dtype=torch.float16, device='cuda', non_blocking=True)
                d['images'] = [item.to(dtype=torch.float16, device='cuda', non_blocking=True) for item in d['images']]
                d['input_ids'] = d['input_ids'].to(device='cuda', non_blocking=True)
                d['labels'] = d['labels'].to(device='cuda', non_blocking=True)
                d['attention_mask'] = d['attention_mask'].to(device='cuda', non_blocking=True)
                batch_len = len(d['input_ids'])

                per_gradients_dict = {name: 0}
                
                for _ in range(n_mezo):
                    
                    if accum_samples >= self.num_samples:
                        break
                    
                    zo_random_seed = np.random.randint(1000000000)
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss1 = loss_func(model, d)['loss']
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=-2, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss2 = loss_func(model, d)['loss']
                
                    # recover the weight
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)

                    accum_samples += batch_len
                    current_batch_index += 1
                    
                    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item() # TODO Perturbed - Perturbed, a little strange.  
                    
                    torch.manual_seed(zo_random_seed)
                    per_gradients_dict[name] += abs(projected_grad)
                        
                gradients_dict[name] += torch.FloatTensor([per_gradients_dict[name]]).abs()
                
        print(gradients_dict)
    
        if self.score_compute == "olmezo-gradient":
            importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "olmezo-aobd":
            importance_measure = {k: v.cpu().data.float().abs() * gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "olmezo-obd":
            importance_measure = {k: v.cpu().data.float() ** 2 * gradients_dict[k] ** 2 for k, v in zip(names, params)}
            
        return importance_measure


    def compute_density(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader

        names = []
        model.eval()
        for k, v in model.named_parameters():
            if k in layer_to_group_mapping:
                names.append(k)
        
        density_dict = {k: 0 for k in names}

        with torch.no_grad():
            inps, outs, caches, image_masks = prepare_calibration_input_encoder(model, data_loader, 'model', 128, 'model.layers')
        
        n_samples = len(inps)
        layers = get_module_recursive(model, 'model.layers')

        for i in range(len(layers)):

            layer = layers[i]
            subset = find_layers(layer)

            wrapped_layers = {}

            for name in subset:
                full_name = f'model.layers.{i}.{name}.weight'
                if full_name in density_dict:
                    wrapped_layers[name] = ActivationDensity(subset[name], i, name)
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data, image_masks[j])
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j], **caches[j])[0]
            
            for h in handles:
                h.remove()
            
            for name in subset:
                full_name = f'model.layers.{i}.{name}.weight'
                if full_name in density_dict:
                    v_density, l_density, vl_dist = wrapped_layers[name].v_density, wrapped_layers[name].l_density, wrapped_layers[name].vl_dist
                    density_dict[full_name] += sum([1 - v_density, 1 - l_density, 1 - vl_dist])
            
            inps, outs = outs, inps
        
        importance_measure = {k: torch.FloatTensor([density_dict[k]]).abs() for k in names}

        return importance_measure

    def compute_outlier(self, layer_to_group_mapping):

        model = self.model
        data_loader = self.data_loader

        names = []
        model.eval()
        for k, v in model.named_parameters():
            if k in layer_to_group_mapping:
                names.append(k)
        
        all_layer_ratio=[]

        with torch.no_grad():
            inps, outs, caches, image_masks = prepare_calibration_input_encoder(model, data_loader, 'model', 128, 'model.layers')
        
        n_samples = len(inps)
        layers = get_module_recursive(model, 'model.layers')

        for i in range(len(layers)):

            layer = layers[i]
            subset = find_layers(layer)

            wrapped_layers = {}

            for name in subset:
                full_name = f'model.layers.{i}.{name}.weight'
                if full_name in names:
                    wrapped_layers[name] = WrappedGPT(subset[name], i, name)
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data, image_masks[j])
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j], **caches[j])[0]
            
            for h in handles:
                h.remove()
            
            layer_wmetric = []
            for name in subset:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                layer_wmetric.append(W_metric)

            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

            out_ratio_layer = check_outlier_mean(layer_wmetric, threshold=5)
            all_layer_ratio.append(out_ratio_layer)

            inps, outs = outs, inps

        all_layer_ratio = np.array(all_layer_ratio)

        all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * 0.08*2))
    
        all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + self.original_sparsity
        importance_dict = {}

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            for name in subset:
                full_name = f'model.layers.{i}.{name}.weight'
                importance_dict[full_name] = all_layer_ratio[i].item()

        return importance_dict



def net_esd_estimator(
            net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5, 
            filter_zeros=False):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'alphahat': []
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone().cpu()
            # i have checked that the multiplication won't affect the weights value
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            matrix = matrix.to(torch.float32)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            
            if filter_zeros:
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                nz_eigs = eigs
                N = len(nz_eigs)
            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n)
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n)
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()
            final_alphahat=final_alpha*math.log10(spectral_norm)

            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())

    return results