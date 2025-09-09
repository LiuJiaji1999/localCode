
\uparrow 表示上升箭头
\downarrow 表示下降箭头

\Uparrow 表示上升箭头（带边框）
\Downarrow 表示下降箭头（带边框）


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

\label{eq1}
$g_{b,v}^{l}= GAP\left ( F_{b,v}^{l} \right )\in R^{C},l\in L_{g}=\left\{ 10\right\}$

$\hat{g}_{b,v}^{l}$

\label{eq2}
$s_{b}^{l}\left ( v,v^{'} \right )=\left< \hat{g}_{b,v}^{l},\hat{g}_{b,v^{'}}^{l}\right>=\frac{\left ( g_{b,v}^{l} \right )^{T}g_{b,v^{'}}^{l}}{\left\| g_{b,v}^{l}\right\|_{2} \left\|g_{b,v^{'}}^{l} \right\|_{2}},v,v^{'}\in \left\{ 1,...,V\right\}$

\label{eq3}
$s_{global}^{l}=\frac{1}{BV\left ( V-1 \right )}\sum_{b=1}^{B}\sum_{v,v^{'}=1,v\neq v^{'}}^{V}s_{b}^{l}\left ( v,v^{'} \right )$

\label{eq4}
$L_{global}=\sum_{l\in L_{g}}^{}\left ( 1-s_{global}^{l} \right )$


$p_{b,v}^{l}\in R^{C \times H \times W}$

$K \times K$
 
$AP_{k}$

\label{eq5}
$p_{b,v}^{l}= AP_{k}\left ( F_{b,v}^{l} \right )\in R^{C \times K \times K },l\in L_{l}=\left\{ 2,4,6,8,9\right\}$

$P=K^{2}$

$p_{b,v,p}^{l}$

$\hat{p}_{b,v,p}^{l}$

\label{eq6}
$s_{b}^{l,p}\left ( v,v^{'} \right )=\left< \hat{p}_{b,v,p}^{l},\hat{p}_{b,v^{'},p}^{l}\right>,v,v^{'}\in \left\{ 1,...,V\right\} $

\label{eq7}
$s_{local}^{l}=\frac{1}{P}\sum_{p=1}^{P}\left [\frac{1}{BV\left ( V-1 \right )}\sum_{b=1}^{B}\sum_{v,v^{'}=1,v\neq v^{'}}^{V}s_{b}^{l,p}\left ( v,v^{'} \right )\right ]$

\label{eq8}
$L_{local}=\sum_{l\in L_{l}}^{}s_{local}^{l,p} $

\label{eq9}
$Loss=L_{det}+L_{global}+L_{local}$

$I_{o}$
$I_{o},I_{f}$
$I_{o},I_{f},I_{n}$
$I_{o},I_{f},I_{n},I_{s}$
$I_{o},I_{f},I_{n},I_{s},I_{a}$
$I_{o},I_{f},I_{n},I_{s},I_{a},I_{c}$

```bash
\usepackage{booktabs}
\usepackage{multirow}

# table1
\begin{table}[]
\centering
\begin{tabular}{@{}cccc@{}}
\toprule
View & PSNR(↑) & SSIM(↑) & LPIPS(↓) \\ \midrule
Far & 25.02 & 0.75 & 0.27 \\
Near & 24.32 & 0.38 & 0.39 \\
Shear & 12.64 & 0.28 & 0.48 \\
Affine & 8.65 & 0.19 & 0.63 \\
ColorJitter & 20.90 & 0.93 & 0.14 \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl1}
\end{table}

# table2
\begin{table}[]
\centering
\begin{tabular}{@{}ccccccccc@{}}
\toprule
View & Method & pin-un & pin-ru & pin-de & Insulator-bu & Insulator-de & Insulator-di & mAP(\%) \\ \midrule
\multirow{12}{*}{Singleview} & Faster R-CNN & 22.5 & 58.5 & 43.8 & 72.4 & 75.2 & 31.5 & 50.7 \\
 & Retina-Net & 11.7 & 54.2 & 37.8 & 70.7 & 65.9 & 22.9 & 43.9 \\
 & GFL & 17.5 & 59.6 & 45.5 & \textbf{75.6} & 64.6 & 24.5 & 47.9 \\
 & TOOD & 22.1 & 62.2 & 44.8 & 70.0 & 69.1 & 23.9 & 48.7 \\
 & DETR &  &  &  &  &  &  &  \\
 & Deformable-DETR &  &  &  &  &  &  &  \\
 & RT-DETR & 24.1 & 69.5 & 51.8 & 68.4 & 65.9 & 35.1 & 52.2 \\
 & YOLOv8 & 10.4 & 55.9 & 34.9 & 65.5 & 64.8 & 22.4 & 42.5 \\
 & YOLO-DTAD & 17.4 & 63.7 & 41.1 & 72.5 & 72.4 & 22.3 & 48.2 \\
 & YOLOv9 & 15.3 & 53.9 & 37.4 & 56.2 & 71.7 & \textbf{41.3} & 46.0 \\
 & YOLOv10 & 19.0 & 55.6 & 39.1 & 62.9 & 66.2 & 21.2 & 44.0 \\
 & YOLOv11 & 20.0 & 57.5 & 44.7 & 71.2 & 74.4 & 39.7 & 51.2 \\
Multiview & MFENet & \textbf{26.8} & \textbf{64.7} & \textbf{47.2} & 73.6 & \textbf{80.0} & 35.8 & \textbf{55.9} \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl2}
\end{table}

# table3
\begin{table}[]
\centering
\begin{tabular}{@{}cccccc@{}}
\toprule
Method & CPLID-defect & VPMBGI-defect & IDID-flashover & IDID-broken & mAP(\%) \\ \midrule
YOLOv8 & 99.5 & 87.0 & 97.9 & 98.1 & 95.6 \\
YOLO-DTAD & 99.5 & 89.0 & 97.7 & 98.8 & 96.2 \\
YOLOv11 & 99.5 & 89.5 & 97.0 & 98.2 & 96.0 \\
MFENet & \textbf{99.5} & \textbf{90.3} & \textbf{97.9} & \textbf{99.4} & \textbf{96.7} \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl3}
\end{table}

# table4
\begin{table}[]
\centering
\begin{tabular}{@{}cccccccccccc@{}}
\toprule
Method & Ign-reg & ped & peo & bic & car & van & tru & tri & aw-tri & bus & mAP(\%) \\ \midrule
YOLOv8 & 33.3 & 19.0 & 13.3 & 77.5 & 42.8 & 44.9 & 24.7 & 23.7 & 60.9 & 36.6 & 37.7 \\
YOLO-DTAD & 33.5 & 20.1 & 13.8 & 77.6 & 43.3 & 44.8 & 24.4 & 25.6 & 62.3 & 36.8 & 38.2 \\
YOLOv11 & 39.7 & 21.3 & 16.5 & 80.7 & 46.7 & 53.9 & \textbf{29.9} & 25.6 & 65.5 & 42.3 & 42.2 \\
MFENet & \textbf{40.7} & \textbf{22.0} & \textbf{19.3} & \textbf{81.5} & \textbf{47.8} & \textbf{55.0} & 29.1 & \textbf{26.3} & \textbf{67.7} & \textbf{44.0} & \textbf{43.3} \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl4}
\end{table}

# table5
\begin{table}[]
\centering
\begin{tabular}{@{}cccccccc@{}}
\toprule
View & pin-un & pin-ru & pin-de & Insulator-bu & Insulator-de & Insulator-di & mAP(\%) \\ \midrule
\multicolumn{1}{l}{eq1} & 20.0 & 57.5 & 44.7 & 71.2 & 74.4 & 39.7 & 51.2 \\
eq2 & 23.8 & 60.1 & 44.0 & 65.9 & 73.3 & 40.4 & 51.2 \\
er & 23.6 & 60.7 & 45.7 & 68.4 & 72.3 & 40.3 & 51.8 \\
eq & 20.5 & 61.6 & 48.5 & \textbf{71.5} & 76.0 & \textbf{40.8} & 53.1 \\
eq & 26.1 & 62.8 & 49.3 & 69.9 & \textbf{76.9} & 37.1 & 53.7 \\
eq & \textbf{28.6} & \textbf{67.0} & \textbf{54.5} & 68.2 & 76.0 & 30.8 & \textbf{54.2} \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl5}
\end{table}

# table6
\begin{table}[]
\centering
\begin{tabular}{@{}cccccccc@{}}
\toprule
Method & pin-un & pin-ru & pin-de & Insulator-bu & Insulator-de & Insulator-di & mAP(\%) \\ \midrule
Base & 20.0 & 57.5 & 44.7 & 71.2 & 74.4 & 39.7 & 51.2 \\
Base+MV & 28.6 & 67.0 & 54.5 & 68.2 & 76.0 & 30.8 & 54.2 \\
Base+MV+GL & 26.8 & 64.7 & 54.3 & \textbf{73.6} & \textbf{80.0} & \textbf{35.8} & \textbf{55.9} \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl6}
\end{table}

# table7
\begin{table}[]
\centering
\begin{tabular}{@{}cccc@{}}
\toprule
Method & FPS(f*s-1) & Params(M) & FLOPs(G) \\ \midrule
Faster R-CNN & 33.2 & 41.4 & 199.9 \\
RetinaNet & 34.3 & 36.4 & 198.0 \\
GFL & 24.1 & 32.3 & 197.0 \\
TOOD & 42.0 & 19.0 & 142.0 \\
DETR &  &  &  \\
Deformable-DETR &  &  &  \\
RT-DETR & 146.7 & 38.6 & 57.0 \\
YOLOv8 & 217.8 & 49.6 & 79.1 \\
YOLO-DTAD & 150.0 & 44.5 & 101.9 \\
YOLOv9 & 102.2 & 251.6 & 130.7 \\
YOLOv10 & 203.3 & 32 & 101.9 \\
MFENet & 213.1 & 38.6 & 67.7 \\ \bottomrule
\end{tabular}
\caption{}
\label{tbl7}
\end{table}
```