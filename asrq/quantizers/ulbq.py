# Based on Code obtained from the authors of the paper
# "Ultra-Low Bit Post-Training Quantization of Large Speech Models via K-Means Clustering and Mixed Precision Allocation"

# pyright: reportMissingImports=false
from typing import Any, Tuple

import torch
from sklearn.cluster import KMeans
from omegaconf import DictConfig
from asrq.core.registry import QuantizerNames, register_quantizer, register_quantizer_config
from asrq.core.utils import cuda_empty_cache, cuda_synchronize
from asrq.quantizers.base import HessianAddBatchMixin, QuantConfig, Quantizer



def kmclustering(x,bit):   #K-Means clustering

    # non_zero_elements = x[x != 0].reshape(-1, 1)
    x = x.reshape(-1, 1)

    # if len(non_zero_elements) == 0:
    #     return torch.tensor([0.0])

    # effective_k = min(2**bit, len(non_zero_elements))
    effective_k = min(2**bit, len(x))

    k_means = KMeans(effective_k, random_state=10,n_init='auto')

    # k_means.fit(non_zero_elements.cpu())
    k_means.fit(x.cpu())
    
    q_value = torch.tensor(k_means.cluster_centers_).reshape(-1)
    return q_value


def power_quant(x, value_s):  #quantize weight to nearest centroid
    # prune_mask = x != 0
    shape = x.shape
    xhard = x.view(-1)
    idxs = (xhard.unsqueeze(0) - value_s.reshape(-1,1)).abs().min(dim=0)[1]
    xhard = value_s[idxs].view(shape)
    # return xhard * prune_mask
    return xhard


def index_quant(x, value_s):  #store the index of the codebook
    prune_mask = x != 0
    shape = x.shape
    xhard = x.view(-1)
    idxs = (xhard.unsqueeze(0) - value_s.reshape(-1,1)).abs().min(dim=0)[1]
    return idxs


def quantize_outlier(x, bit, low, up, mean, q_value): # keep outlier K-Means style fake_quantize
    q = power_quant(x,q_value.to(x.device))
    if low>0 and low<1:
        W1 = x.reshape(1,-1)
        upper =  W1.sort(descending=True)[0][0][round((W1.size(1)*low))]
        lower =  W1.sort(descending=True)[0][0][round((W1.size(1)*up))]
        zero = torch.zeros_like(x) 
        one = torch.ones_like(x)
        outlier = torch.where(torch.gt(x, lower) & torch.lt(x, upper), zero, x)
        musk = torch.where(torch.gt(x, lower) & torch.lt(x, upper), one, zero)
        q = q*musk+outlier

    elif low>1:
        W1 = x.reshape(1,-1)
        cur = torch.mean(x.view(-1).float().abs(),dim=0)*low  #colomn level  
        zero = torch.zeros_like(x) 
        one = torch.ones_like(x)
        outlier = torch.where(torch.gt(x, -cur) & torch.lt(x, cur), zero, x)
        musk = torch.where(torch.gt(x, -cur) & torch.lt(x, cur), one, zero)
        q = q*musk+outlier        
    return q.float()


def quantize(x, bit, q_value): # standard K-Means style fake_quantize
    q = power_quant(x,q_value.to(x.device))
    return q.float()


def index_quantize(x, bit, q_value): # standard K-Means style
    q = power_quant(x,q_value.to(x.device))
    center = index_quant(x,q_value.to(x.device))
    return q.float(), center



@register_quantizer_config(QuantizerNames.ULBQ)
class ULBQConfig(QuantConfig):
    def __init__(self, cfg: DictConfig)->None:
        super().__init__(cfg)
        self.percdamp = cfg.percdamp
        self.block_size = cfg.block_size
        self.outlier_col_dynamic = None
        self.outlierorder = None
        if self.bits == 2:
            self.outlier_col_dynamic = True
            self.outlierorder = 2.1



@register_quantizer(QuantizerNames.ULBQ)
class ULBQQuantizer(HessianAddBatchMixin, Quantizer):
    """Quantizer for Ultra-Low Bit Quantization (ULBQ) using K-Means clustering and mixed precision allocation.
    
    This method is a combination of Kmeans and GPTQ updates, which iteratively quantizes weights while minimizing
    the quantization error based on the Hessian of the loss landscape. It also includes dynamic outlier handling 
    and mixed precision allocation based on the distribution of weight values.
    """
    quant_config: ULBQConfig
    def __init__(self, module, name:str, cfg: ULBQConfig):
        HessianAddBatchMixin.__init__(self, module)
        Quantizer.__init__(self, module, name, cfg)


    def __call__(self) -> Tuple[Any, Any]:
        W = self.module.weight.data.clone() # type: ignore
        bits = self.quant_config.bits
        W = W.float()
        KM = []
        # prune_mask = W != 0

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = self.quant_config.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(W.size(1), device=W.device)
        H[diag, diag] += damp

        success = False
        attempts = 0
        Hinv = None
        while not success:
            try:
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
                success = True
            except RuntimeError as e:
                print(f"Attempt {attempts}: Matrix not positive definite, modifying diagonal elements.")
            H[diag, diag] += damp
            attempts += 1
        assert Hinv is not None, "Failed to compute stable Cholesky decomposition after multiple attempts."

        mean = torch.mean(W.view(-1).float().abs(), dim=0)
        t = 13

        if self.quant_config.outlier_col_dynamic:
            zero = torch.zeros_like(W)
            one = torch.ones_like(W)
            sens = torch.sum(torch.where(torch.gt(W, -mean*t) & torch.lt(W, mean*t), zero, one), dim=0)
            par = 0.1
            cur = sens.sort(descending=True)[1]
            sens[cur[:round(len(sens)*par)]] = 0.014
            sens[cur[round(len(sens)*par):round(len(sens)*(1-par))]] = 0.004
            sens[cur[round(len(sens)*(1-par)):]] = 0.004
            out_per = sens

        if self.quant_config.outlierorder:
            outlierorder = self.quant_config.outlierorder
            zero = torch.zeros_like(W)
            one = torch.ones_like(W)
            sens = torch.sum(torch.where(torch.gt(W, -mean*t) & torch.lt(W, mean*t), zero, one), dim=0)
            cur = sens.sort(descending=True)[1]
            if outlierorder < 2 or outlierorder > 4:
                raise ValueError("mixed-precision bit out of range, try 2~4")
            outlierorder_ = (outlierorder-2)/2 if outlierorder < 3 else outlierorder-3
            if outlierorder < 3: 
                sens[cur[:round(len(sens)*outlierorder_)]]=4
                sens[cur[round(len(sens)*outlierorder_):]]=2
            else:
                sens[cur[:round(len(sens)*outlierorder_)]]=4
                sens[cur[round(len(sens)*outlierorder_):]]=3  
            out = sens
        
        columns = W.size(1)
        blocksize = self.quant_config.block_size # type: ignore
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            if self.quant_config.outlierorder:
                out1 = out[i1:i2].clone() # type: ignore
            if self.quant_config.outlier_col_dynamic:
                out_per1 = out_per[i1:i2].clone() # type: ignore
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            # Prune_mask1 = prune_mask[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if self.quant_config.outlierorder:
                    bits = int(out1[i]) # type: ignore
                if self.quant_config.outlier_col_dynamic:
                    outlier = out_per1[i].item() # type: ignore
                else:
                    outlier = 0
                
                km = kmclustering(w, bits)
                KM.append(km)
                if outlier != 0:
                    q = quantize_outlier(w.unsqueeze(1), bits, outlier, 1-outlier, mean, km)
                else:
                    q = quantize(w.unsqueeze(1), bits, km)
                q = q.squeeze()
                # q = q * Prune_mask1[:, i]
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w-q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                # W1[:, i:] *= Prune_mask1[:, i:]
                Err1[:, i] = err1
            
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1/2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            # W[:, i2:] *= prune_mask[:, i2:]
        
        cuda_synchronize()

        inv_perm = torch.argsort(perm)
        Q = Q[:, inv_perm]
        self.module.weight.data = Q.reshape(self.module.weight.shape).to(self.module.weight.data.dtype) # type: ignore

        cuda_empty_cache()
        return (None, None)