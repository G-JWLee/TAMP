import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import math
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from time import time
from copy import deepcopy
from functools import partial
from tqdm import tqdm

from videollama2.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity
from videollama2.pruners.utils import print_time
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

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

    def add_batch(self, inp, out, video_mask=None, audio_mask=None, score=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


class AdaptiveMultimodalInputActivation:
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
    
    def gaussian_rbf(self, X, Y, sigma=1.0):
        X_norm = (X ** 2).sum(dim=1).view(-1,1)
        Y_norm = (Y ** 2).sum(dim=1).view(1,-1)
        pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, Y.T)

        K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
        return K

    def cos_pairwise_density(self, embeddings, video_mask, audio_mask):
        v_embeddings = embeddings[video_mask]
        a_embeddings = embeddings[audio_mask]
        l_embeddings = embeddings[~torch.logical_or(video_mask,audio_mask)]

        v_distances = torch.mm(v_embeddings, v_embeddings.T)
        v_upper_triangular = v_distances.triu(diagonal=1) # Only take upper triangular distances (no duplicates)
        v_mean_dist = v_upper_triangular[v_upper_triangular > 0].mean().item()

        a_distances = torch.mm(a_embeddings, a_embeddings.T)
        a_upper_triangular = a_distances.triu(diagonal=1) # Only take upper triangular distances (no duplicates)
        a_mean_dist = a_upper_triangular[a_upper_triangular > 0].mean().item()

        l_distances = torch.mm(l_embeddings, l_embeddings.T)
        l_upper_triangular = l_distances.triu(diagonal=1) # Only take upper triangular distances (no duplicates)
        l_mean_dist = l_upper_triangular[l_upper_triangular > 0].mean().item()

        vl_dist = torch.mm(v_embeddings, l_embeddings.T)
        vl_mean_dist = vl_dist.mean().item()

        al_dist = torch.mm(a_embeddings, l_embeddings.T)
        al_mean_dist = al_dist.mean().item()

        av_dist = torch.mm(a_embeddings, v_embeddings.T)
        av_mean_dist = av_dist.mean().item()

        return (v_mean_dist + a_mean_dist + l_mean_dist + vl_mean_dist + al_mean_dist + av_mean_dist) / 6

    def add_batch(self, inp, out, video_mask=None, audio_mask=None, score=None):
        
        if len(out.shape) == 3:
            out = out.reshape((-1, out.shape[-1]))
        # out = out.type(torch.float32)
        out = F.normalize(out, dim=-1)
        # Compute density
        density = self.cos_pairwise_density(out, video_mask=video_mask[0], audio_mask=audio_mask[0])

        # Initialize graph
        distances = 1 - torch.mm(out, out.T)

        num_neigh = 3
        knn_indices = torch.topk(distances, k=num_neigh+1, largest=False).indices[:,1:]
        # forward message pass
        neigh_dist = torch.exp(-torch.gather(distances, dim=1, index=knn_indices) * 1.0) * score[knn_indices]
        graph_score = score + neigh_dist.sum(dim=-1)

        K = self.gaussian_rbf(out, out)
        # Iterative selection
        selected_indices = set()
        while True:
            selected = torch.argmax(graph_score)
            neighbors = knn_indices[selected]
            graph_score[neighbors] = graph_score[neighbors] - torch.exp(-distances[selected,neighbors] * 0.2) * torch.maximum(graph_score[selected], torch.zeros_like(graph_score[selected]))
            selected_indices.add(selected.item())

            graph_score[torch.tensor(list(selected_indices))] = torch.min(graph_score) - 1

            K_XX = K.mean()
            temp_select = torch.tensor(list(selected_indices))
            K_XY = K[:, temp_select].mean()
            K_YY = K[temp_select,:][:,temp_select].mean()
            MMD2 = K_XX + K_YY - 2 * K_XY

            if MMD2 < torch.sqrt(torch.tensor(1-density)) * 0.1:
                break

        # score mask
        score_mask = torch.zeros_like(score, dtype=torch.bool)
        score_mask[torch.tensor(list(selected_indices))] = True

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        inp = inp[score_mask[None,:]]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        # Since we don't zero-pad the endding tokens
        tmp = inp.shape[1]
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples




class LLaMALayerWandaPruner(LayerWiseBasePruner):
    pruner_name = "llama_wanda_pruner"
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
        score_method="obd_avg",
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
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
            prune_n=prune_n,
            prune_m=prune_m,
        )
                
        self.ignore_layers = [] # not used but may be used in the future
        for k in self.model_stem.state_dict():
            # don't prune embedding layers and lm_head
            if any(sub_n in k for sub_n in ["shared", "embed_tokens", "lm_head", "layer_norm"]):
                self.ignore_layers.append(k)

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    @print_time
    def create_pruned_arch(self, transformer, prune_spec):
        side_config = deepcopy(transformer.config)

        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(prune_spec)
        
        side_config.num_decoder_layers = num_layers
        side_config.num_layers = num_layers

        # unstructural
        side_config.d_model = side_config.d_model
        side_config.d_ff = side_config.d_ff
        side_config.d_kv = side_config.d_kv
            
        pruned_transformer = transformer.__class__(side_config)

        return pruned_transformer
    
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith("t5_model"):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}. Use shared.weight for it.")
                    scores[k] = scores["t5_model.shared.weight"]

        return scores
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        use_cache = getattr(model, self.model_prefix).config.use_cache 
        getattr(model, self.model_prefix).config.use_cache = False 

        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

        getattr(model, self.model_prefix).config.use_cache = use_cache
        return float(count)/total_params 
    
    # def forward_to_cache(self, model, batch):
    #     return model(batch)
    
    def prepare_calibration_input_encoder(self, model, dataloader, model_prefix, n_samples, module_to_process="encoder.block"):
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
        video_masks = []
        audio_masks = []
            
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
            batch['images'][0][0]['video'] = batch['images'][0][0]['video'].to(dtype=torch.float16, device='cuda', non_blocking=True)
            batch['images'][0][0]['audio'] = batch['images'][0][0]['audio'].to(dtype=torch.float16, device='cuda', non_blocking=True)
            batch['input_ids'] = batch['input_ids'].to(device='cuda', non_blocking=True)
            batch['labels'] = batch['labels'].to(device='cuda', non_blocking=True)
            batch['attention_mask'] = batch['attention_mask'].to(device='cuda', non_blocking=True)
            
            if total_samples >= n_samples:
                break
            total_samples += len(batch['input_ids'])
            
            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                video_masks.append(model.video_token_masks)
                audio_masks.append(model.audio_token_masks)
                pass 

        layers[0] = layers[0].module
        outs = [None] * len(inps)

        getattr(model, model_prefix).config.use_cache = use_cache
        return inps, outs, caches, video_masks, audio_masks

    @print_time
    def _prune(self, model, dataloader, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5, token_selection='naive'):
        use_cache = getattr(model, model_prefix).config.use_cache 
        getattr(model, model_prefix).config.use_cache = False 

        with torch.no_grad():
            inps, outs, caches, video_masks, audio_masks = self.prepare_calibration_input_encoder(model, dataloader, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))
        layers = get_module_recursive(model, module_to_process)

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            scores = [None] * len(inps)

            # Casual Attention map calculation
            if 'attention' in token_selection:
                for j in range(n_samples):
                    with torch.no_grad():
                        hid_state = layer.input_layernorm(inps[j])
                        bsz, q_len, _ = hid_state.size()

                        q_state = layer.self_attn.q_proj(hid_state)
                        k_state = layer.self_attn.k_proj(hid_state)
                        v_state = layer.self_attn.v_proj(hid_state)

                        q_state = q_state.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
                        k_state = k_state.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
                        v_state = v_state.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)

                        kv_seq_len = k_state.shape[-2]
                        cos, sin = layer.self_attn.rotary_emb(v_state, seq_len=kv_seq_len)
                        q_state, k_state = apply_rotary_pos_emb(q_state, k_state, cos, sin, caches[j]['position_ids'])
                        
                        k_state = repeat_kv(k_state, layer.self_attn.num_key_value_groups)
                        v_state = repeat_kv(v_state, layer.self_attn.num_key_value_groups)

                        attn_weights = torch.matmul(q_state, k_state.transpose(2, 3)) / math.sqrt(layer.self_attn.head_dim)
                        
                        def update_causal_mask(attention_mask, input_tensor, cache_position):
                            dtype, device = input_tensor.dtype, input_tensor.device

                            sequence_length = input_tensor.shape[1]
                            target_length = attention_mask.shape[-1]
                            
                            min_dtype = torch.finfo(dtype).min

                            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                            causal_mask = torch.triu(causal_mask, diagonal=1)
                            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1,1)
                            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
                            
                            causal_mask = causal_mask.clone()
                            mask_length = attention_mask.shape[-1]
                            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                            padding_mask = padding_mask == 0
                            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                                padding_mask, min_dtype
                            )

                            return causal_mask

                        # Causal mask
                        if model.config._attn_implementation in ['flash_attention_2', 'sdpa']:
                            temp = model.config._attn_implementation
                            model.config._attn_implementation = 'eager'
                            attention_mask = torch.ones((hid_state.shape[:2]), dtype=torch.bool, device=hid_state.device)
                            causal_mask = update_causal_mask(attention_mask, hid_state, caches[j]['cache_position'])
                            model.config._attn_implementation = temp
                        else:
                            causal_mask = caches[j]['attention_mask']

                        if causal_mask is not None:
                            causal_mask = causal_mask[:, :, :, : k_state.shape[-2]]
                            attn_weights = attn_weights + causal_mask
                        
                        # upcast attention to fp32 when applying softmax activation
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_state.dtype)
                        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                        scores[j] = attn_weights.mean(dim=1).squeeze()[-1,:]

            wrapped_layers = {}
            for name in subset:
                if token_selection == 'naive':
                    wrapped_layers[name] = WrappedGPT(subset[name])
                elif token_selection == 'amia':
                    wrapped_layers[name] = AdaptiveMultimodalInputActivation(subset[name])
                else:
                    raise ValueError(f"token_selection {token_selection} not defined")

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data, video_masks[j], audio_masks[j], scores[j])
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in tqdm(range(n_samples)):
                with torch.no_grad():
                    outs[j] = layer(inps[j], **caches[j])[0]

            for h in handles:
                h.remove()

            for name in subset:
                # assert wrapped_layers[name].nsamples == len(inps) * inps[0].shape[0]
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

                setattr(subset[name].weight, "importance_score", W_metric.cpu().abs().mean().item())
                
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if self.prune_n != 0:
                    # structured n:m sparsity
                    print(f"pruning {model_prefix} layer {i} {name} at structured {self.prune_n}:{self.prune_m} sparsity")
                    for ii in range(W_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = W_metric[:,ii:(ii+self.prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                else:
                    # unstructured pruning
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                    print(f"pruning {model_prefix} layer {i} {name} at unstructured {sparsity_ratio[sparsity_key]} sparsity")

                    indices = sort_res[1][:,:int(W_metric.shape[1] * sparsity_ratio[sparsity_key])]
                    W_mask.scatter_(1, indices, True)

                setattr(subset[name], "mask", ~W_mask.bool())
                subset[name].weight.data[W_mask] = 0  ## set weights to zero `

            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j], **caches[j])[0]
            inps, outs = outs, inps
                
        getattr(model, model_prefix).config.use_cache = use_cache 
        # del inps, outs, caches 
        torch.cuda.empty_cache()
        gc.collect()
        
        return model
        
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):

        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)
        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.encoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )
        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


class VITLayerWandaPruner(LayerWiseBasePruner):
    pruner_name = "vit_wanda_pruner"
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
        model_prefix="visual",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd_avg",
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
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
            prune_n=prune_n,
            prune_m=prune_m,
        )
                
        self.ignore_layers = []
        
        for k in self.model_stem.state_dict():
            # don't prune embedding layers and output layers
            if any(sub_n in k for sub_n in ["cls_token", "pos_embed", "patch_embed", "norm"]):
                self.ignore_layers.append(k)

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    @print_time
    def create_pruned_arch(self, vit, vit_prune_spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(vit_prune_spec)
        
        if self.is_strct_pruning:
            pruned_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=int(vit.embed_dim * res_keep_ratio),
                attn_dim=int(vit.attn_dim * attn_keep_ratio),
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio * ffn_keep_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        else:
            pruned_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=vit.embed_dim,
                attn_dim=vit.attn_dim,
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        
        return pruned_vit
    
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith(self.model_prefix):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}")

        return scores
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

        return float(count)/total_params 
    
    # def forward_to_cache(self, model, batch, ):
    #     return model.encode_image(batch["image"])
    
    def prepare_calibration_input_encoder(self, model, dataloader, model_prefix, n_samples, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        inps = []        
        caches = []
        
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                
            def forward(self, hidden_states, attention_mask, causal_attention_mask, output_attentions):
                inps.append(hidden_states)
                inps[-1].requires_grad = False
                
                cache = {}
                cache['attention_mask'] = attention_mask
                cache['causal_attention_mask'] = causal_attention_mask
                cache['output_attentions'] = output_attentions
                
                caches.append(cache)
                raise ValueError

        layers[0] = Catcher(layers[0])
        
        total_samples = 0
        for i, batch in enumerate(dataloader):
            batch['images'][0] = batch['images'][0].to(dtype=torch.float16, device='cuda', non_blocking=True)
            if total_samples >= n_samples:
                break
            total_samples += len(batch["images"])
            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                pass 
        layers[0] = layers[0].module

        outs = [None] * len(inps)

        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5):
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))

        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            #     dev = model.hf_device_map[f"model.layers.{i}"]
            #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            # print(f"caches: {caches}")
            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j], **caches[j])

            for h in handles:
                h.remove()

            for name in subset:
                # assert wrapped_layers[name].nsamples == len(inps) * inps[0].shape[0]
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

                setattr(subset[name].weight, "importance_score", W_metric.cpu().abs().mean().item())
                
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if self.prune_n != 0:
                    # structured n:m sparsity
                    print(f"pruning {model_prefix} layer {i} {name} at structured {self.prune_n}:{self.prune_m} sparsity")
                    for ii in range(W_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = W_metric[:,ii:(ii+self.prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                else:
                    # # unstructured pruning
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                    print(f"pruning {model_prefix} layer {i} {name} at unstructured {sparsity_ratio[sparsity_key]} sparsity")
                    thres = torch.sort(W_metric.flatten())[0][int(W_metric.numel() * sparsity_ratio[sparsity_key])]
                    W_mask = (W_metric < thres)
                    
                setattr(subset[name], "mask", ~W_mask.bool())
                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j], **caches[j])[0]
            inps, outs = outs, inps
            
        # del inps, outs, caches 
        torch.cuda.empty_cache()
        gc.collect()

        return model

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):

        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.blocks",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


class VideoLLaMA2LayerWandaPruner(LayerWiseBasePruner):
    pruner_name = "videollama2_wanda_pruner"
    def __init__(
        self,
        model,
        data_loader,
        llm_sparsity_ratio=None,
        vit_sparsity_ratio=None,
        aud_sparsity_ratio=None,
        llm_pruning_method=None,
        vit_pruning_method=None,
        aud_pruning_method=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
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
        noise_eps=1e-3,
        prune_per_model=False,
        peft_postfix="",
        prune_n=0,
        prune_m=0,
        token_selection="naive",
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=None,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=f"{aud_model_prefix}+{vit_model_prefix}+{llm_model_prefix}",
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
            prune_n=prune_n,
            prune_m=prune_m, 
        )
        
        self.llm_sparsity_ratio = llm_sparsity_ratio
        self.vit_sparsity_ratio = vit_sparsity_ratio
        self.aud_sparsity_ratio = aud_sparsity_ratio
        
        self.peft_postfix = peft_postfix
        self.aud_dense = True
        self.vit_dense = True
        self.llm_dense = True
        
        assert llm_pruning_method is not None
        assert vit_pruning_method is not None
        assert aud_pruning_method is not None
        
        self.llm_model_prefix = llm_model_prefix
        self.vit_model_prefix = vit_model_prefix
        self.aud_model_prefix = aud_model_prefix

        # Strategy of token selection when calculating Wanda score
        self.token_selection = token_selection
        
    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        if sparsity_ratio_granularity == None or sparsity_ratio_granularity == "none":
            layer_to_group_mapping = {}
        
        else:
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
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            def use_modality(name):
                if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)) and (not name.startswith(self.aud_model_prefix)):
                    return self.llm_sparsity_ratio != 0
                elif name.startswith(self.vit_model_prefix):
                    return self.vit_sparsity_ratio != 0
                elif name.startswith(self.aud_model_prefix):
                    return self.aud_sparsity_ratio != 0
                else:
                    return False

            if sparsity_ratio_granularity == "model":
                
                def return_group(name):
                    if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)) and (not name.startswith(self.aud_model_prefix)):
                        return self.llm_model_prefix
                    elif name.startswith(self.vit_model_prefix):
                        return self.vit_model_prefix
                    elif name.startswith(self.aud_model_prefix):
                        return self.aud_model_prefix
                    else:
                        return "other"
                
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune if use_modality(k)
                }
                
            elif sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune if use_modality(k)
                }
                
            elif sparsity_ratio_granularity == "block":
                def return_group(name):
                    if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)):
                        return ".".join(name.split(".")[:4])
                    elif name.startswith(self.vit_model_prefix):
                        return ".".join(name.split(".")[:3])
                    elif name.startswith(self.aud_model_prefix):
                        return ".".join(name.split(".")[:3])  # TODO: check here.
                    else:
                        return "other"

                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune if use_modality(k)
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            self.forward_to_cache, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            # prune_per_model=self.prune_per_model,
            # per_model_group=[self.llm_model_prefix, self.vit_model_prefix],
            # per_model_sparsity=[llm_sparsity_ratio, vit_sparsity_ratio],
        )
        
        return sparsity_module.return_sparsity()
        
    def forward_to_cache(self, model, batch):
        return model(**batch)

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):

        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)
        global_sparsity_dict = None

        if self.sparsity_ratio_granularity not in [None, "none"]: 

            global_sparsity_dict = self.get_sparsity(
                self.llm_sparsity_ratio, 
                sparsity_ratio_granularity=self.sparsity_ratio_granularity
            )

        self.aud_dense = True if float(1 - self.aud_sparsity_ratio) < 1. else False
        self.vit_dense = True if float(1 - self.vit_sparsity_ratio) < 1. else False
        self.llm_dense = True if float(1 - self.llm_sparsity_ratio) < 1. else False
        

        if float(1 - self.aud_sparsity_ratio) < 1.:
            
            sparsity_ratio = self.aud_sparsity_ratio
            
            if global_sparsity_dict not in [None, "none"]:
                sparsity_dict = global_sparsity_dict
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    # sparsity_ratio, 
                    sparsity_ratio_granularity=None
                )
            
            # print(f"vit sparsity dict: {sparsity_dict}")
            _aud_prune = partial(VITLayerWandaPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                VITLayerWandaPruner.prepare_calibration_input_encoder,
                self,
            )
            
            self.model = _aud_prune(
                self.model, self.data_loader, 
                model_prefix=self.aud_model_prefix,
                module_to_process=f"{self.aud_model_prefix}.layers",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            )


        if float(1 - self.vit_sparsity_ratio) < 1.:
            
            sparsity_ratio = self.vit_sparsity_ratio
            
            if global_sparsity_dict not in [None, "none"]:
                sparsity_dict = global_sparsity_dict
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    # sparsity_ratio, 
                    sparsity_ratio_granularity=None
                )
            
            # print(f"vit sparsity dict: {sparsity_dict}")
            _vit_prune = partial(VITLayerWandaPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                VITLayerWandaPruner.prepare_calibration_input_encoder,
                self,
            )
            
            self.model = _vit_prune(
                self.model, self.data_loader, 
                model_prefix=self.vit_model_prefix,
                module_to_process=f"{self.vit_model_prefix}.layers",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            )
            
        if float(1 - self.llm_sparsity_ratio) < 1.:
            sparsity_ratio = self.llm_sparsity_ratio
            # print(f"sparsity_ratio: {sparsity_ratio}")
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    # sparsity_ratio, 
                    sparsity_ratio_granularity=None
                )
                
            # print(f"global_sparsity_dict: {global_sparsity_dict}")
            # print(f"sparsity_dict: {sparsity_dict}")
            _llm_prune = partial(LLaMALayerWandaPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                LLaMALayerWandaPruner.prepare_calibration_input_encoder,
                self,
            )
            self.model = _llm_prune(
                self.model, self.data_loader, 
                model_prefix=self.llm_model_prefix,
                module_to_process=f"{self.llm_model_prefix}{self.peft_postfix}.layers", 
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                token_selection=self.token_selection
            )
            
        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, global_sparsity_dict
    
    def check(self, name, v, model_prefix):
        if len(v.shape) == 2 and \
                ".block" in name and \
                    "relative_attention_bias.weight" not in name and \
                        name.startswith(model_prefix):
            return True
        return False