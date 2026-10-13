import { py } from '@/content/helpers'
export const blockPython=py`import numpy as np
def _norm(x, weight, eps):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps) * weight

def _rope(x):
    t, h, dh = x.shape
    angle = np.arange(t)[:,None,None] * (10000.**(-np.arange(0,dh,2)/dh))[None,None,:]
    c,s = np.cos(angle),np.sin(angle)
    y=x.copy();a,b=x[...,0::2],x[...,1::2]
    y[...,0::2]=a*c-b*s;y[...,1::2]=a*s+b*c
    return y

def decoder_block(x: np.ndarray, w: dict, heads: int, kv_heads: int, eps: float=1e-5) -> np.ndarray:
    t,d=x.shape;dh=d//heads
    z=_norm(x,w['n1'],eps)
    q=_rope((z@w['q']).reshape(t,heads,dh))
    k=_rope((z@w['k']).reshape(t,kv_heads,dh))
    v=(z@w['v']).reshape(t,kv_heads,dh)
    group=heads//kv_heads
    k=np.repeat(k,group,axis=1);v=np.repeat(v,group,axis=1)
    scores=np.einsum('thd,shd->hts',q,k)/np.sqrt(dh)
    scores=np.where(np.arange(t)[None,:]<=np.arange(t)[:,None],scores,-np.inf)
    p=np.exp(scores-scores.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
    attn=np.einsum('hts,shd->thd',p,v).reshape(t,d)
    y=x+attn@w['o']
    z=_norm(y,w['n2'],eps)
    gate=z@w['gate'];silu=gate/(1+np.exp(-gate))
    return y+(silu*(z@w['up']))@w['down']`
export const blockTorch=py`import torch
def decoder_block(x,w,heads,kv_heads,eps=1e-5):
    t,d=x.shape;dh=d//heads
    norm=lambda a,g:a*torch.rsqrt((a*a).mean(-1,keepdim=True)+eps)*g
    def rope(a):
        inv=10000.**(-torch.arange(0,dh,2,device=x.device,dtype=x.dtype)/dh)
        angles=torch.arange(t,device=x.device)[:,None,None]*inv[None,None,:]
        c,s=angles.cos(),angles.sin();b=a.clone()
        b[...,0::2]=a[...,0::2]*c-a[...,1::2]*s
        b[...,1::2]=a[...,0::2]*s+a[...,1::2]*c
        return b
    z=norm(x,w['n1'])
    q=rope((z@w['q']).reshape(t,heads,dh)).transpose(0,1)
    k=rope((z@w['k']).reshape(t,kv_heads,dh)).repeat_interleave(heads//kv_heads,1).transpose(0,1)
    v=(z@w['v']).reshape(t,kv_heads,dh).repeat_interleave(heads//kv_heads,1).transpose(0,1)
    a=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    y=x+a.transpose(0,1).reshape(t,d)@w['o']
    z=norm(y,w['n2'])
    return y+(torch.nn.functional.silu(z@w['gate'])*(z@w['up']))@w['down']`
export const weightSetup=py`import numpy as np
def _weights(d=8,heads=4,kv=2,f=12,seed=21):
    rng=np.random.default_rng(seed);dh=d//heads
    shapes={'q':(d,d),'k':(d,kv*dh),'v':(d,kv*dh),'o':(d,d),'gate':(d,f),'up':(d,f),'down':(f,d)}
    w={name:rng.normal(size=shape)*.1 for name,shape in shapes.items()}
    w['n1']=rng.uniform(.8,1.2,d);w['n2']=rng.uniform(.8,1.2,d)
    return w

def _oracle(x,w,h,hkv,eps=1e-5):
    # 独立逐头、逐位置计算，不使用 einsum 或 repeat。
    t,d=x.shape;dh=d//h
    def norm(a,g):return a/np.sqrt((a*a).mean(-1,keepdims=True)+eps)*g
    z=norm(x,w['n1']);q=(z@w['q']).reshape(t,h,dh);k=(z@w['k']).reshape(t,hkv,dh);v=(z@w['v']).reshape(t,hkv,dh)
    def rotate(a,pos):
        b=a.copy()
        for i in range(0,dh,2):
            theta=pos*10000**(-i/dh);c,s=np.cos(theta),np.sin(theta)
            b[i],b[i+1]=a[i]*c-a[i+1]*s,a[i]*s+a[i+1]*c
        return b
    out=np.zeros((t,h,dh))
    for i in range(t):
        for j in range(h):
            kj=j//(h//hkv)
            scores=np.array([rotate(q[i,j],i)@rotate(k[s,kj],s)/np.sqrt(dh) for s in range(i+1)])
            p=np.exp(scores-scores.max());p/=p.sum()
            for s in range(i+1):out[i,j]+=p[s]*v[s,kj]
    y=x+out.reshape(t,d)@w['o'];z=norm(y,w['n2']);g=z@w['gate']
    return y+(g/(1+np.exp(-g))*(z@w['up']))@w['down']`
