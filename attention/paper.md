
```bash
\usepackage{amsthm}
\usepackage[ruled,vlined]{algorithm2e}
\begin{algorithm}[H]
\caption{Multiview Image Albumentations Pipeline}
\KwIn{Batch $\textbf{B}$ of images  $ I \in R^{B\times C \times H \times W}$}
\KwOut{Batch $\textbf{B}$ of images $I_{out}=\left\{ I_{far},I_{near},I_{original},I_{shear},I_{affine},I_{color}\right\}$}

\For{image $i$ in batch $\textbf{B}$ }{
    \tcp{Far-view transformation:}
    $I_{far} \leftarrow Resize(320,320,cv2.INTER\_LANCZOS4) $\;

    \tcp{Near-view transformation:}
     $T_{s}\sim (0.8, 1.0)$ for scale, $T_{r}\sim (0.75,1.33)$ for  ratio; \\
    {
        $I_{near} \leftarrow RandomResizedCrop(640,640,T_{s},T_{r}) $\;
    }
    
    \tcp{Shear transformation:}
    $D_{s}:\left\{ x\sim U\left ( -15^{\circ},15^{\circ} \right ),y\sim \left ( -10^{\circ},10^{\circ} \right )\right\}$ for shear;\\
    {
        $I_{shear} \leftarrow Affine(D_{s},p) $\;
    }

    \tcp{Affine transformation:}
    $D_{s}:\left\{ x\sim U\left ( 0.9 ,1.1 \right ),y\sim \left ( 0.9,1.1 \right )\right\}$ for scale, $\alpha_{r} \sim \left ( -15^{\circ},15^{\circ} \right )$ for rotate, $D_{t}:\left\{ x\sim U\left ( 0.9 ,1.1 \right ),y\sim \left ( 0.9,1.1 \right )\right\}$ for translate;\\
    {
        $I_{affine} \leftarrow Affine(D_{s},\alpha_{r},D_{t},p) $\;
    }

    \tcp{Color perturbation:}
    $\alpha _{b}\sim U\left ( 0,1 \right )$ for brightness, $\alpha _{c}\sim U\left ( 0,1 \right )$ for contrast,
    $\alpha _{s}\sim U\left ( 0,1 \right )$ for saturation, 
    $\alpha _{h}\sim U\left ( 0,0.5 \right )$ for hue ; \\
    {
         $I_{color} \leftarrow ColorJitter(\alpha _{b},\alpha _{c},\alpha _{s},\alpha _{h},p) $\;
    }
}
\Return $I_{out}$\;
\end{algorithm}
```

$L_{g}$

$L_{l}$

$F^{l}\in R^{\left ( BV\right )\times C \times H \times W}$

$F_{b,v}^{l}\in R^{C \times H \times W}$

$g_{b,v}^{l}= GAP\left ( F_{b,v}^{l} \right )\in R^{C},l\in L_{g}=\left\{ 10\right\}$

$\hat{g}_{b,v}^{l}$

$s_{b}^{l}\left ( v,v^{'} \right )=\left< \hat{g}_{b,v}^{l},\hat{g}_{b,v^{'}}^{l}\right>=\frac{\left ( g_{b,v}^{l} \right )^{T}g_{b,v^{'}}^{l}}{\left\| g_{b,v}^{l}\right\|_{2} \left\|g_{b,v^{'}}^{l} \right\|_{2}},v,v^{'}\in \left\{ 1,...,V\right\}$

$s_{global}^{l}=\frac{1}{BV\left ( V-1 \right )}\sum_{b=1}^{B}\sum_{v,v^{'}=1,v\neq v^{'}}^{V}s_{b}^{l}\left ( v,v^{'} \right )$

$L_{global}=\sum_{l\in L_{g}}^{}\left ( 1-s_{global}^{l} \right )$


$p_{b,v}^{l}\in R^{C \times H \times W}$

$K \times K$
 
$AP_{k}$

$p_{b,v}^{l}= AP_{k}\left ( F_{b,v}^{l} \right )\in R^{C \times K \times K },l\in L_{l}=\left\{ 2,4,6,8,9\right\}$

$P=K^{2}$

$p_{b,v,p}^{l}$

$\hat{p}_{b,v,p}^{l}$

$s_{b}^{l,p}\left ( v,v^{'} \right )=\left< \hat{p}_{b,v,p}^{l},\hat{p}_{b,v^{'},p}^{l}\right>,v,v^{'}\in \left\{ 1,...,V\right\} $

$s_{local}^{l}=\frac{1}{P}\sum_{p=1}^{P}\left [\frac{1}{BV\left ( V-1 \right )}\sum_{b=1}^{B}\sum_{v,v^{'}=1,v\neq v^{'}}^{V}s_{b}^{l,p}\left ( v,v^{'} \right )\right ]$

$L_{local}=\sum_{l\in L_{l}}^{}s_{local}^{l,p} $

$Loss=L_{det}+L_{global}+L_{local}$