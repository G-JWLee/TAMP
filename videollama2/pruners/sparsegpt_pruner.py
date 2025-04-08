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
from functools import partial

from videollama2.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity
from videollama2.pruners.utils import print_time
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import transformers


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


class SparseGPT:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.nsamples_v = 0
        self.nsamples_l = 0
        
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def fastprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        
        if (torch.isinf(H) * (H > 0)).float().sum() > 0:
            # positive inf value
            pos = torch.isinf(H) * (H > 0)
            H[pos] = torch.quantile(H, 0.999)
            
        if (torch.isinf(H) * (H < 0)).float().sum() > 0:
            # negative inf value
            pos = torch.isinf(H) * (H < 0)
            H[pos] = torch.quantile(H, 0.001)
            
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        
        while True:
            try:
                decompose_H = torch.linalg.cholesky(H)
                
                if not torch.isnan(decompose_H).any():
                    H = decompose_H
                    break
                
                if torch.isinf(damp).any():
                    import pdb; pdb.set_trace()
                # not a positive semi-definite matrix
                H[diag, diag] += damp
            except:
                # not a positive semi-definite matrix
                H[diag, diag] += damp
        # H[diag, diag] += damp
        # H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        
        if (torch.isinf(H) * (H > 0)).float().sum() > 0:
            # positive inf value
            pos = torch.isinf(H) * (H > 0)
            H[pos] = torch.quantile(H, 0.999)
            
        if (torch.isinf(H) * (H < 0)).float().sum() > 0:
            # negative inf value
            pos = torch.isinf(H) * (H < 0)
            H[pos] = torch.quantile(H, 0.001)
            
        damp = percdamp * torch.mean(torch.diag(H).abs())
        diag = torch.arange(self.columns, device=self.dev)
        
        while True:
            try:
                decompose_H = torch.linalg.cholesky(H, upper=True)
                
                if not torch.isnan(decompose_H).any():
                    H = decompose_H
                    break
                # not a positive semi-definite matrix
                H[diag, diag] += damp
            except:
                # not a positive semi-definite matrix
                H[diag, diag] += damp

        # H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        s = W ** 2 / (torch.diag(Hinv).reshape((1, -1))) ** 2
        
        setattr(self.layer.weight, "importance_score", s.cpu().abs().mean().item())

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


class LLaMALayerSparseGPTPruner(LayerWiseBasePruner):
    pruner_name = "llama_sparsegpt_pruner"
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
    def _prune(self, model, dataloader, model_prefix, module_to_process="encoder.block", n_samples=128, sparsity_ratio=0.5, token_selection='naive'):
        use_cache = getattr(model, model_prefix).config.use_cache 
        getattr(model, model_prefix).config.use_cache = False 

        with torch.no_grad():
            inps, outs, caches, video_masks, audio_masks = self.prepare_calibration_input_encoder(model, dataloader, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))
        layers = get_module_recursive(model, module_to_process)
   

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            wrapped_layers = {}
            for name in subset:
                if token_selection == 'naive':
                    wrapped_layers[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
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
                sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                wrapped_layers[name].fastprune(sparsity_ratio[sparsity_key], prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01, blocksize=128)
                wrapped_layers[name].free()

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


class VITLayerSparseGPTPruner(LayerWiseBasePruner):
    pruner_name = "vit_sparsegpt_pruner"
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
                wrapped_layers[name] = SparseGPT(subset[name])

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
                sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                wrapped_layers[name].fastprune(sparsity_ratio[sparsity_key], prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01, blocksize=128)
                wrapped_layers[name].free()

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


class VideoLLaMA2LayerSparseGPTPruner(LayerWiseBasePruner):
    pruner_name = "videollama2_sparsegpt_pruner"
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
            model_prefix=f"{vit_model_prefix}+{llm_model_prefix}",
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

        # Strategy of token selection when calculating SparseGPT score
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
                    if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)) and self.llm_sparsity_ratio != 0:
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
                if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)):
                    return self.llm_sparsity_ratio != 0
                elif name.startswith(self.vit_model_prefix):
                    return self.vit_sparsity_ratio != 0
                elif name.startswith(self.aud_model_prefix):
                    return self.aud_sparsity_ratio != 0
                else:
                    return False

            if sparsity_ratio_granularity == "model":
                
                def return_group(name):
                    if name.startswith(self.llm_model_prefix) and (not name.startswith(self.vit_model_prefix)):
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
                        return ".".join(name.split(".")[:3])  
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
            _vit_prune = partial(VITLayerSparseGPTPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                VITLayerSparseGPTPruner.prepare_calibration_input_encoder,
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
            _llm_prune = partial(LLaMALayerSparseGPTPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                LLaMALayerSparseGPTPruner.prepare_calibration_input_encoder,
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