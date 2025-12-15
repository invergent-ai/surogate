import torch
from torchao.quantization import Float8Tensor

from surogate.core.model.kernels.fp8 import fp8_linear
from surogate.core.model.kernels.utils import fast_dequantize, _maybe_fake_quantize_activations, get_lora_parameters
from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type = "cuda")
    def forward(
        ctx,
        X: torch.Tensor,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace = True,
    ):
        dtype = X.dtype

        Q = matmul_lora(X, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(X, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(X, VW, VW_quant, VA, VB, VS)

        ctx.custom_saved_tensors = (
            QW,
            QW_quant,
            QS,
            KW,
            KW_quant,
            KS,
            VW,
            VW_quant,
            VS,
        )
        ctx.save_for_backward(
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
        )
        ctx.inplace = inplace
        return Q, K, V

    @staticmethod
    @torch.amp.custom_bwd(device_type = "cuda")
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = ctx.custom_saved_tensors
        (
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
        ) = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])  # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = (
            QA.to(dtype),
            QB.to(dtype),
            KA.to(dtype),
            KB.to(dtype),
            VA.to(dtype),
            VB.to(dtype),
        )

        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        ### Weight projection LoRA weights
        # See our blogpost for more details.
        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)

        # Q Projection
        # d_QA = X.t() @ (dQ @ QB.t())
        # d_QB = (QA.t() @ X.t()) @ dQ
        # d_QA *= QS
        # d_QB *= QS
        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha = QS, beta = 0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha = QS, beta = 0)

        # K Projection
        # d_KA = X.t() @ (dK @ KB.t())
        # d_KB = (KA.t() @ X.t()) @ dK
        # d_KA *= KS
        # d_KB *= KS
        d_KA.addmm_(X.t(), dK @ KB.t(), alpha = KS, beta = 0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha = KS, beta = 0)

        # V Projection
        # d_VA = X.t() @ (dV @ VB.t())
        # d_VB = (VA.t() @ X.t()) @ dV
        # d_VA *= VS
        # d_VB *= VS
        d_VA.addmm_(X.t(), dV @ VB.t(), alpha = VS, beta = 0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha = VS, beta = 0)

        # Combine derivatives to find dX
        # dQ
        QW = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW.t(), out = X if ctx.inplace else None)
        del QW
        # dX += (dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t()))
        dX.addmm_(dQ @ QB.t(), QA.t(), alpha = QS)

        # dK
        KW = fast_dequantize(KW.t(), KW_quant)
        # dX += dK @ KW.t()
        dX.addmm_(dK, KW.t())
        del KW
        # dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())
        dX.addmm_(dK @ KB.t(), KA.t(), alpha = KS)

        # dV
        VW = fast_dequantize(VW.t(), VW_quant)
        # dX += dV @ VW.t()
        dX.addmm_(dV, VW.t())
        del VW
        # dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())
        dX.addmm_(dV @ VB.t(), VA.t(), alpha = VS)

        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_QA.t(),
            d_QB.t(),
            None,
            None,
            None,
            d_KA.t(),
            d_KB.t(),
            None,
            None,
            None,
            d_VA.t(),
            d_VB.t(),
            None,
            None,
        )


def apply_lora_qkv(self, X, inplace = True):
    X = _maybe_fake_quantize_activations(X, self.q_proj)
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace,
    )
    return Q, K, V

class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type = "cuda")
    def forward(
        ctx,
        X: torch.Tensor,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        _forward_function,
        _backward_function,
        inplace = True,
    ):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return i

    @staticmethod
    @torch.amp.custom_bwd(device_type = "cuda")
    def backward(ctx, dY: torch.Tensor):
        (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _backward_function,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = (
            gateA.to(dtype),
            gateB.to(dtype),
            upA.to(dtype),
            upB.to(dtype),
            downA.to(dtype),
            downB.to(dtype),
        )

        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(),
            gateB.t(),
            upA.t(),
            upB.t(),
            downA.t(),
            downB.t(),
        )

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)

        # Down projection LoRA weights
        # d_downA = h.t() @ (dY @ downB.t())
        # d_downB = (downA.t() @ h.t()) @ dY
        # d_downA *= downS
        # d_downB *= downS
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

        # Up projection LoRA weights
        # d_upA   = X.t() @ (df @ upB.t())
        # d_upB   = (upA.t() @ X.t()) @ df
        # d_upA  *= upS
        # d_upB  *= upS
        d_upA.addmm_(X.t(), df @ upB.t(), alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha = upS, beta = 0)

        # Gate projection LoRA weights
        # d_gateA = X.t() @ (de @ gateB.t())
        # d_gateB = (gateA.t() @ X.t()) @ de
        # d_gateA *= gateS
        # d_gateB *= gateS
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
        del upW
        # dX += df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())
        dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        # dX += de @ gateW.t()
        dX.addmm_(de, gateW.t())
        del gateW
        # dX += de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_gateA.t(),
            d_gateB.t(),
            None,
            None,
            None,
            d_upA.t(),
            d_upB.t(),
            None,
            None,
            None,
            d_downA.t(),
            d_downB.t(),
            None,
            None,
            None,
            None,
        )  # _backward and _forward and inplace

class LoRA_W(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type = "cuda")
    def forward(ctx, X: torch.Tensor, W, W_quant, A, B, S):
        dtype = X.dtype
        XW = matmul_lora(X, W, W_quant, A, B, S)
        ctx.custom_saved_tensors = (
            W,
            W_quant,
            S,
        )
        ctx.save_for_backward(A, B, X)
        return XW

    @staticmethod
    @torch.amp.custom_bwd(device_type = "cuda")
    def backward(ctx, dY: torch.Tensor):
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])  # Must be reshape
        X = X.reshape(-1, X.shape[-1])  # Must be reshape
        dtype = X.dtype

        A, B = A.to(dtype), B.to(dtype)

        A, B = A.t(), B.t()

        d_A = torch.empty_like(A)
        d_B = torch.empty_like(B)

        ### Weight projection LoRA weights
        # Weight projection
        # d_A = X.t() @ (dY @ B.t())
        # d_B = (A.t() @ X.t()) @ dY
        # d_A *= S
        # d_B *= S
        d_A.addmm_(X.t(), dY @ B.t(), alpha = S, beta = 0)
        d_B.addmm_(A.t() @ X.t(), dY, alpha = S, beta = 0)

        # Get derivative for dX
        W = fast_dequantize(W.t(), W_quant)
        dX = dY @ W.t()
        del W
        # dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())
        dX.addmm_(dY @ B.t(), A.t(), alpha = S)

        # W, W_quant, A, B, S
        return dX.view(batch, seq_len, hd), None, None, d_A.t(), d_B.t(), None


def apply_lora_o(self, X):
    X = _maybe_fake_quantize_activations(X, self.o_proj)
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    O = LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)
    return O


def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    if isinstance(W, Float8Tensor):
        assert W.ndim == 2
        if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
            # In the backward pass, rowwise scaled becomes colwise scaled after we
            # transpose the weight tensor. Use this case to detect backward.
            # TODO: would be simpler if we simply don't call `matmul_lora` in backward
            W = W.dequantize()
        else:
            W = W.contiguous()
        out = torch.matmul(X, W.t(), out = out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    elif W.dtype in (torch.bfloat16, torch.float16, torch.float32):
        # No quantization needed (e.g., DeepSpeed ZeRO-2 wraps weights as regular bf16/fp16/fp32)
        out = torch.matmul(X, W.t(), out = out)
    else:
        W = fast_dequantize(W, W_quant, use_global_buffer = True)
        out = torch.matmul(X, W.t(), out = out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch.matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)
        # out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out

def apply_lora_mlp_swiglu(self, X, inplace = True):
    X = _maybe_fake_quantize_activations(X, self.gate_proj)
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_fg_kernel,
        swiglu_DWf_DW_dfg_kernel,
        inplace,
    )
    return out

