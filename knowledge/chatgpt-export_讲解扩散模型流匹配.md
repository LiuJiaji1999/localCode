# 讲解扩散模型流匹配

> 请你作为老师，举例并详细的解释，扩散模型、流匹配等相关概念，要通俗易懂哈！

我会按“先讲故事和直觉，再拆解训练、生成和公式”的顺序来讲，并用同一个例子比较扩散模型与流匹配。最后把 DDPM、DDIM、Score、Flow Matching、Rectified Flow 等容易混淆的概念串起来。

好呀！😄 今天我就以老师的身份，带你从零开始理解 **扩散模型（Diffusion Model）、流匹配（Flow Matching）**，以及它们背后的一系列重要概念。

我会尽量做到三件事：

- **讲人话**：先用生活中的例子理解，而不是一上来就堆公式。
- **讲透彻**：不仅告诉你是什么，还会解释为什么这样设计、训练和推理到底在干什么。
- **串知识**：把 DDPM、DDIM、Score Matching、SDE、ODE、Rectified Flow、DiT 等概念联系起来。

我们把这堂课分成 8 个部分。你完全可以把它当作一份生成式 AI 入门课程。📚

# 第一课：生成模型到底在做什么？

先忘掉扩散、流匹配这些复杂名字。

假设我们有一个任务：

**让 AI 画出一只它从来没有见过的猫。**

<grid columns={2} gap={3}>
  <grid-item>
    [图片：cute orange tabby kitten portrait natural light photography]
    <text color="secondary" size="xs" textAlign="center">训练数据：真实的猫</text>
  </grid-item>
  <grid-item>
    [图片：beautiful photorealistic white fluffy kitten portrait studio background]
    <text color="secondary" size="xs" textAlign="center">目标：生成新的猫</text>
  </grid-item>
</grid>

我们给 AI 看了 100 万张猫的图片。

它要学到的并不是简单地记住这些照片，而是猫的整体规律：

- 猫通常有两只耳朵。
- 猫的眼睛、鼻子和嘴巴存在一定的位置关系。
- 猫的毛色、姿势、体型千变万化。
- 某些特征经常一起出现。

机器学习里，我们把这些规律称为 **数据分布（Data Distribution）**。

## 1.1 什么是数据分布？

用一个简单例子理解。

假设有一所学校，学生的身高大多在 150～190 cm 之间。

<box gap={1}>
  [图表或交互组件，未导出]
  <caption>示意数据：中间身高的人更多，两端更少。</caption>
</box>

如果 AI 学会了这个分布，那么你让它随机生成一个学生身高：

它可能生成 168 cm、175 cm、181 cm。

但是不应该经常生成 300 cm。

这就是生成模型要做的事：

**学习真实数据的概率分布，然后从这个分布里创造新的样本。**

可问题来了：

一张图片有几十万甚至几百万个数值，这个分布太复杂了！我们怎么学？

于是研究者想到一个非常聪明的办法：

**不直接学习如何画猫，而是学习如何把随机噪声变成猫。**

这就引出了扩散模型。

---

# 第二课：扩散模型（Diffusion Model）

## 2.1 核心思想：先破坏，再修复

想象你正在画一只猫。

然后有一个调皮的小朋友，不断往你的画上撒白色和黑色的小点。

<box gap={2}>
  <grid columns={3} gap={2}>
    <grid-item>
      [图片：orange tabby cat closeup portrait clear sharp face plain background]
      <text size="xs" color="secondary" textAlign="center">Step 0 · Clean</text>
    </grid-item>
    <grid-item>
      [图片：orange cat photograph obscured by heavy digital gaussian noise grain static television interference cat face barely recognizable]
      <text size="xs" color="secondary" textAlign="center">Step 1 · Noisy</text>
    </grid-item>
    <grid-item>
      [图片：pure gaussian white noise monochrome random gray static texture square]
      <text size="xs" color="secondary" textAlign="center">Step 2 · Pure noise</text>
    </grid-item>
  </grid>
  <text color="secondary" size="xs" textAlign="center">示意图：正向加噪，逐渐丢失图像信息</text>
</box>

原本清晰的猫，慢慢变得模糊，最后变成一团随机噪声。

这叫：

**正向扩散（Forward Diffusion）**

用符号表示：

$$
x_0\rightarrow x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_T
$$

其中：

- $x_0$：原始清晰图片
- $x_1$：加入少量噪声的图片
- $x_2$：噪声更多的图片
- $x_T$：几乎是纯随机噪声

有意思的地方来了：

既然我们知道怎么把猫变成噪声，能不能让 AI 学会反过来操作？

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0
$$

从噪声开始，一点点恢复出合理的猫。

这个过程叫：

**反向扩散（Reverse Diffusion）**

它就是生成图片的核心。

## 2.2 AI 到底在学习什么？

假设我给你一张被噪声污染的猫图，然后问：

“你觉得这里面哪些部分是噪声？”

你可能说：

“猫的脸应该是平滑的，周围那些乱七八糟的小点可能是噪声。”

扩散模型中的神经网络，就在做类似的事情。

以经典 DDPM 的常见训练方式为例：

<box border radius="xl" padding={3} gap={2}>
  <row align="center" justify="between">
    **训练扩散模型**
    <badge color="secondary">Noise Prediction</badge>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    <text weight="medium">一张真实的猫图 $x_0$</text>
    <icon name="arrow-down" color="tertiary"/>
    <text weight="medium">随机选择一个噪声程度 $t$</text>
    <icon name="arrow-down" color="tertiary"/>
    <text weight="medium">加入随机噪声，得到 $x_t$</text>
    <icon name="arrow-down" color="tertiary"/>
    <box background="rgba(20,125,160,0.09)" radius="md" padding={3} width="100%" align="center" gap={1}>
      **神经网络**
      <text size="sm" textAlign="center">观察图片 + 噪声程度</text>
      <text size="sm" textAlign="center">预测加入了哪些噪声</text>
    </box>
    <icon name="arrow-down" color="tertiary"/>
    <text weight="medium">与真实噪声比较，计算误差</text>
    <icon name="arrow-down" color="tertiary"/>
    <text weight="medium" color="success">调整网络参数，让预测越来越准</text>
  </box>
</box>

这里你要注意一个非常重要的区别：

**训练时，通常不需要让模型完整地从纯噪声恢复一张图片。**

我们可以直接随机选择任意一个噪声程度，构造训练样本，让网络完成一次预测任务。

这样训练会方便很多。

## 2.3 扩散模型的第一个重要公式

来看一下这个公式：

$$
\boxed{x_t=\sqrt{\bar{\alpha}_t}x_0+
\sqrt{1-\bar{\alpha}_t}\epsilon}
$$

别怕！我来翻译成中文。

这其实就是：

**加噪后的图片 = 一部分原图 + 一部分随机噪声**

其中：

| 符号 | 含义 |
|---|---|
| $x_0$ | 原始图片 |
| $x_t$ | 第 $t$ 个噪声程度对应的图片 |
| $\epsilon$ | 从标准高斯分布中采样的噪声 |
| $\bar{\alpha}_t$ | 控制保留多少原图信息 |

例如，我们做一个简化演示：

$$
x_t=0.8x_0+0.6\epsilon
$$

可以理解为保留了一部分真实图像信号，同时混入了一部分噪声。

这里的两个系数满足平方和为 1，是常见的方差保持形式。

当原图的权重越来越小时，图像就越来越像噪声。

训练的目标通常是：

$$
\boxed{
L=\mathbb{E}\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
}
$$

翻译：

“模型预测出来的噪声，和真正添加的噪声，差多少？”

差得越小，说明网络越会识别噪声。

这个经典思路来自 DDPM。([papers.neurips.cc](https://papers.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html))

## 2.4 训练完以后怎么生成？

假设用户说：

“请帮我画一只戴墨镜的橘猫。”

扩散模型生成时：

1. 先产生一张随机噪声图。
2. 神经网络结合提示词，预测当前图片中的噪声或其他等价的去噪信息。
3. 采样算法据此更新图像，让它更接近合理的猫图。
4. 重复多步，直到形成一幅清晰的图片。

要注意，实际采样并不总是简单地“直接减去预测噪声”，还会涉及时间步系数和采样器公式。

你可以把它想象成一个雕塑家：

**扩散模型就像雕塑家从一团混乱的材料中，反复修改，逐渐雕刻出一只猫。**

---

# 第三课：流匹配（Flow Matching）

现在我们换一种思路。

刚刚的扩散模型着重学习如何去噪。

那么，有没有一种方法，可以让 AI 直接学会：

**“如果我现在处于这个状态，下一步应该往哪个方向移动？”**

有的，这就是流匹配的重要思路。

## 3.1 一个非常形象的例子：导航 🚗

假设你要从 A 点开车到 B 点。

有两种教学方式：

第一种是不断让你纠正位置：

“你偏左了，往右一点。”

“你又偏了，再调整一点。”

第二种是给你一套导航规则：

“在当前位置，应该向东北方向移动，速度保持在每小时 40 公里。”

第二种思路很接近 Flow Matching。

神经网络学习一个 **速度场（Velocity Field）**。

什么意思呢？

在空间中的每一个位置、每一个时刻，网络都告诉你：

“从这里出发，应该朝哪里走、走多快。”

<box border radius="xl" padding={3} gap={1}>
  <svg viewBox="0 0 340 185" width="100%" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="#3688a3"/>
      </marker>
    </defs>
    <rect x="5" y="5" width="330" height="175" rx="12" fill="#eaf5f5"/>
    {#each [42,82,122,162] as y}
      {#each [42,82,122,162,202,242,282] as x}
        <line x1={x} y1={y} x2={x+16+(y-80)*0.035} y2={y-8-(x-120)*0.045} stroke="#82afb5" strokeWidth="1.8" markerEnd="url(#arr)" opacity={0.7}/>
      {/each}
    {/each}
    <path d="M45 141 C103 145 121 94 163 90 S239 55 288 42" stroke="#226f8c" strokeWidth="3.5" fill="none" strokeLinecap="round" strokeDasharray="0" markerEnd="url(#arr)"/>
    <circle cx="45" cy="141" r="10" fill="#ec9774" stroke="#fff" strokeWidth="2.5"/>
    <circle cx="288" cy="42" r="10" fill="#287f70" stroke="#fff" strokeWidth="2.5"/>
    <text x="25" y="164" fontSize="12" fontWeight="600" fill="#225066">START</text>
    <text x="273" y="25" fontSize="12" fontWeight="600" fill="#225066">END</text>
  </svg>
  <caption textAlign="center">速度场示意：小箭头表示各个位置的移动方向；曲线是一条可能的运动轨迹。</caption>
</box>

这里有一个特别关键的思想：

**模型不是简单地记住某条路线，而是在学习整个空间里的运动规律。**

假设所有汽车都从不同地方出发，但最终需要到达一些指定区域。

Flow Matching 希望学到一套规则，让大量初始位置按照这些规则运动后，最终形成目标分布。

用于图像生成时：

- 起点分布：随机噪声
- 终点分布：真实图片的分布
- 神经网络：预测运动速度
- 生成过程：沿着学到的速度场，把噪声逐渐变成图片

## 3.2 流匹配最重要的一个公式

我们先选择一个最简单的路径：

**直线。**

假设：

- $z$：随机噪声
- $x_1$：真实图片
- $t$：从 0 到 1 的时间

那么可以定义：

$$
\boxed{x_t=(1-t)z+tx_1}
$$

什么意思？

当 $t=0$：

$$
x_t=z
$$

就是纯噪声。

当 $t=1$：

$$
x_t=x_1
$$

就是真实图片。

当 $t=0.5$：

$$
x_t=0.5z+0.5x_1
$$

就是噪声和图片各占一部分的中间状态。

<box border radius="xl" padding={3} gap={3}>
  <row justify="between" align="center">
    **动手体验：从噪声走向目标**
    <badge>Interactive</badge>
  </row>
  {@body const [t,setT] = DIL.useState(0.5)}
  {@body const z=-2}
  {@body const target=4}
  {@body const value=z*(1-t)+target*t}
  <svg viewBox="0 0 320 103" width="100%">
    <rect x="15" y="31" width="290" height="8" rx="4" fill="#cddce1"/>
    <rect x="15" y="31" width={290*t} height="8" rx="4" fill="#489cb8"/>
    <circle cx={15+290*t} cy="35" r="11" fill="#247f9d" stroke="#fff" strokeWidth="3"/>
    <circle cx="15" cy="35" r="5" fill="#e2946a"/>
    <circle cx="305" cy="35" r="5" fill="#369879"/>
    <text x="15" y="64" fontSize="12" fill="currentColor" textAnchor="start">Noise (-2)</text>
    <text x="305" y="64" fontSize="12" fill="currentColor" textAnchor="end">Data (4)</text>
    <text x={Math.max(27,Math.min(293,15+290*t))} y="16" fontSize="12" fill="#247f9d" fontWeight="bold" textAnchor="middle">{value.toFixed(2)}</text>
    <text x="160" y="91" fontSize="11" fill="currentColor" opacity=".7" textAnchor="middle">Time t = {t.toFixed(2)}</text>
  </svg>
  <slider min={0} max={1} step={0.05} value={t} onChange={setT}/>
  <row justify="between" align="center">
    <text color="secondary" size="sm">$x_t=(1-t)(-2)+t(4)$</text>
    <button size="sm" color="secondary" variant="outline" onClick={()=>setT(0)}>重新开始 <icon name="rotate-ccw" inline/></button>
  </row>
  <text size="sm" color="secondary">拖动滑块观察位置如何随时间变化。这是一维的教学示例，不是真实图像模型的完整采样过程。</text>
</box>

现在我们对刚才的公式求导：

$$
\frac{dx_t}{dt}=x_1-z
$$

结果非常简单！

**这条训练直线路径上的目标速度，就是终点减去起点。**

举例来说：

假设噪声位置是 $-2$，目标位置是 $4$。

那么：

$$
v=4-(-2)=6
$$

我们就可以用这个速度当作训练标签，教网络预测运动方向和快慢。

## 3.3 Flow Matching 如何训练？

它的训练思路如下：

<box background="surface-secondary" padding={3} radius="xl" gap={2} align="center">
  **准备两个样本**

  <grid columns={2} gap={2}>
    <grid-item>
      <box background="surface" radius="lg" padding={3} align="center" gap={1}>
        <icon name="shuffle" size="lg" color="secondary"/>
        **随机噪声 $z$**
        <text size="xs" color="secondary">起点</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface" radius="lg" padding={3} align="center" gap={1}>
        <icon name="image" size="lg" color="success"/>
        **真实图片 $x_1$**
        <text size="xs" color="secondary">终点</text>
      </box>
    </grid-item>
  </grid>
  <icon name="arrow-down" color="tertiary"/>
  <box background="surface" padding={3} radius="lg" width="100%" align="center" gap={1}>
    **随机选择时间 $t$**

    计算中间状态 $x_t=(1-t)z+tx_1$
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="rgba(30,135,171,0.10)" padding={3} radius="lg" width="100%" align="center" gap={1}>
    **神经网络**

    输入 $x_t,t$

    预测当前运动速度 $v_\theta(x_t,t)$
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="surface" padding={3} radius="lg" width="100%" align="center" gap={1}>
    **计算预测误差**

    目标速度 $x_1-z$
  </box>
</box>

它的一个常见训练损失是：

$$
\boxed{
L_{\mathrm{FM}}=
\mathbb E\left[
\left\|v_\theta(x_t,t)-(x_1-z)\right\|^2
\right]
}
$$

是不是和扩散模型的噪声预测损失有点像？

区别在于：

- 经典扩散训练经常预测 **噪声**。
- 这种流匹配训练预测 **速度**。

但请注意：流匹配是更一般的框架，并不要求所有路径必须是直线。这里介绍的是最容易理解的一种形式。原始 Flow Matching 论文也支持包括扩散路径在内的其他概率路径。([arxiv.org](https://arxiv.org/abs/2210.02747))

## 3.4 流匹配怎么生成图片？

训练完成后，不再需要给模型提供真实图片终点。

我们只需要：

1. 随机生成一份初始噪声。
2. 让网络预测此时应该往哪个方向走。
3. 沿着这个方向移动一小段。
4. 在新位置重新预测速度。
5. 不断重复，直到到达终点时间。

数学上就是求解一个微分方程：

$$
\boxed{\frac{dx}{dt}=v_\theta(x,t)}
$$

这种方程叫 **常微分方程（ODE）**。

如果用最简单的欧拉法来求解：

$$
x_{\mathrm{next}}
=x_{\mathrm{current}}+\Delta t\cdot v_\theta(x_{\mathrm{current}},t)
$$

翻译：

**新位置 = 旧位置 + 时间间隔 × 当前预测速度**

想象一下开车：

你现在的位置是 A。

导航说以某个方向和速度行驶 10 秒。

10 秒后，你到了新的位置，再查询一次导航。

就这样慢慢到达目的地。

这里还有一个细节：训练时，我们可以利用已知的噪声和真实图片构造速度标签；但生成时，模型并不知道自己最终要得到哪张具体图片，只能依靠学到的整体速度场前进。

---

# 第四课：扩散模型和流匹配到底有什么区别？

这是今天的重点！⭐

先用一张表建立整体认识。

| 对比维度 | 经典 DDPM 扩散 | 直线路径 Flow Matching |
|---|---|---|
| 最直观的想法 | 学习如何去噪 | 学习如何移动 |
| 常见训练目标 | 预测噪声 | 预测速度 |
| 训练样本 | 清晰数据加入高斯噪声 | 噪声与数据之间插值 |
| 生成起点 | 随机噪声 | 随机噪声 |
| 生成终点 | 目标数据分布 | 目标数据分布 |
| 常见采样形式 | 反向随机过程或确定性采样 | ODE 积分 |
| 生成步数 | 取决于采样器和模型 | 取决于速度场和求解器 |

这里千万不要得出一个过于简单的结论：

“扩散模型一定走弯路，而流匹配一定走直线。”

这不准确。

事实上：

**扩散模型也可以采用确定性的 ODE 采样；Flow Matching 即使使用直线训练路径，实际学到的生成轨迹也未必完全笔直。**

为什么？

因为神经网络在某个中间位置可能会遇到许多不同训练样本的路径，它学的是这些目标速度的条件平均，而不是记住某一条指定路线。

所以更准确的区别是：

**两者通常采用不同的训练目标和建模方式，但在连续时间生成模型的数学框架中存在很深的联系。**

---

# 第五课：Score、SDE 和 ODE 是什么？

这三个词经常出现在扩散论文里。

我们一个一个讲清楚。

## 5.1 Score：告诉你哪里更像真实数据

Score 的中文常叫“得分函数”。

它并不是考试分数。

数学定义是：

$$
\boxed{s(x,t)=\nabla_x\log p_t(x)}
$$

其中 $p_t(x)$ 是在时间 $t$ 的数据概率密度。

这个公式的直觉是：

**在当前位置，朝哪个方向移动，数据出现的概率会增长得最快？**

比如，在一个二维平面里，真实猫图片对应的数据集中在某些区域，随机噪声可能落在其他区域。

Score 就像一支箭头，指出局部概率密度上升最快的方向。

要注意，单独沿 Score 方向走，并不等于完整的扩散采样算法；实际生成还需要考虑时间变化、噪声强度等因素。

## 5.2 SDE：运动过程中存在随机扰动

SDE 全称：

**Stochastic Differential Equation，随机微分方程。**

想象你在湖里划船。

你知道自己应该往东划，但湖水里还有随机波浪，会不断影响你的运动。

可以写成：

$$
dx=f(x,t)dt+g(t)dW_t
$$

两部分：

- $f(x,t)dt$：按照某种规则运动
- $g(t)dW_t$：随机扰动

扩散模型的连续时间理论中，经常用 SDE 描述加噪和对应的反向生成过程。

## 5.3 ODE：没有额外随机扰动的运动规则

ODE 全称：

**Ordinary Differential Equation，常微分方程。**

它可以写成：

$$
dx=v(x,t)dt
$$

或者：

$$
\frac{dx}{dt}=v(x,t)
$$

和 SDE 相比，它没有额外的布朗运动噪声项。

给定起点、模型和求解条件后，这个连续过程通常是确定的。

但这不意味着每次生成的图片都一样，因为你可以每次随机选择不同的起点噪声。

## 5.4 它们之间究竟是什么关系？

这是扩散理论一个非常漂亮的发现：

**某些扩散 SDE 存在对应的 Probability Flow ODE（概率流 ODE），两者可以具有相同的时间边缘分布。**

也就是说：

从整体概率分布的角度看，随机运动和确定性运动可以在每个时刻得到相同的分布。

但它们的单个样本运动轨迹不一定相同。

这个理论是 Score-based Generative Modeling through SDEs 的重要内容。([research.google](https://research.google/pubs/score-based-generative-modeling-through-stochastic-differential-equations/))

因此，不应该把“扩散模型”和“SDE”、“Flow Matching”和“ODE”机械地划上完全互斥的等号。

---

# 第六课：DDPM、DDIM、Rectified Flow 又是什么？

这几个名字特别容易混淆。

## 6.1 DDPM

DDPM，全称 Denoising Diffusion Probabilistic Models。

它是经典的扩散模型框架。

你可以这样记：

**DDPM = 一个经典的扩散模型训练与概率采样框架。**

其典型反向采样步骤带有随机性，通过多步反向转移逐渐得到数据样本。

## 6.2 DDIM

DDIM，全称 Denoising Diffusion Implicit Models。

它和 DDPM 关系很近，尤其体现在采样过程。

你可以把它理解为：

**在与 DDPM 密切相关的训练框架下，使用另一种采样过程。**

DDIM 允许确定性采样（常见设置为 $\eta=0$），也支持非确定性形式。

它的一个重要价值是可以使用较少的采样步骤。

所以请记住：

DDPM 和 DDIM 不一定意味着两套完全不同的神经网络。

很多情况下，差别主要体现在如何从模型的预测结果一步步生成最终样本。

## 6.3 Rectified Flow（修正流 / 整流）

这个名字可以理解为：

**希望把生成轨迹尽量变直。**

想象我们有两点 A 和 B。

一种路线不断绕弯，另一种路线接近直线。

如果生成轨迹更加平直，那么用较少的数值积分步数也可能得到较好的近似结果。

Rectified Flow 的研究就是围绕这种思路建立速度场，并可以通过 reflow 等方法进一步改善轨迹的直线程度。

不过这里要强调：

“希望轨迹更直”不代表每一次生成都保证只需一步，也不意味着它天然优于所有扩散采样器。

([www.cs.utexas.edu](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html))

## 6.4 OT（Optimal Transport，最优传输）

这是 Flow Matching 里经常出现的另一个词。

举个例子。

假设操场上有 100 名学生，教室里有 100 个座位。

你希望每个人都找到一个座位。

最笨的办法是随机给每个人安排座位，可能有很多人需要横穿整个操场。

更聪明的办法是根据位置安排，让大家移动的总成本尽量低。

这就是最优传输的基本直觉。

在生成模型里，我们同样关心：

**如何合理地把噪声分布里的点，运输到真实数据分布？**

选用合适的起终点配对和传输路径，有机会让速度场更容易学习、生成更高效。

---

# 第七课：Latent Diffusion、U-Net、DiT、CFG 是什么？

前面讨论的主要是 **生成过程和学习目标**。

但实际的图像生成系统，还要考虑：

- 神经网络使用什么结构？
- 在什么空间进行生成？
- 如何听懂文字指令？

这些是另外几个维度。

## 7.1 Latent Diffusion：不要直接在原始图片上操作

想象你要处理一张 1024×1024 的图片。

每个像素都有颜色信息，数据量很大。

如果每一步都直接处理所有像素，计算开销就可能非常高。

所以研究者想到：

先把图片压缩成一个更紧凑的表示，再在这个表示空间里运行生成模型。

这个空间叫 **潜空间（Latent Space）**。

<box border radius="xl" padding={3} gap={2} align="center">
  <box border radius="lg" width="100%" padding={2} align="center">
    **原始图片 Pixel Space**
    <text color="secondary" size="sm">高维像素数据</text>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="surface-secondary" radius="lg" width="100%" padding={3} align="center" gap={1}>
    **Encoder · 编码器**
    <text size="sm">将图片压缩到潜空间</text>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="rgba(30,135,171,0.10)" radius="lg" width="100%" padding={3} align="center" gap={1}>
    **Latent Space · 潜空间**
    <text size="sm">在紧凑的表示中进行扩散或流式生成</text>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="surface-secondary" radius="lg" width="100%" padding={3} align="center" gap={1}>
    **Decoder · 解码器**
    <text size="sm">把潜变量还原为图片</text>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  **最终生成的图片**
</box>

你可以把它想象成：

不直接在超大的画布上工作，而是先在一个更紧凑的表示空间中完成创作，最后再展开成高清图像。

Latent Diffusion 和 Flow Matching 并不冲突。

一个讲的是**在哪个空间生成**，另一个讲的是**如何学习生成过程**。

## 7.2 U-Net 和 DiT：神经网络的结构

如果扩散和 Flow Matching 规定的是“学什么”，那么 U-Net 和 DiT 更多是在回答“用什么网络来学”。

| 概念 | 作用 | 通俗理解 |
|---|---|---|
| U-Net | 常见的图像去噪网络架构 | 擅长多尺度观察图像 |
| DiT | Diffusion Transformer 架构 | 利用 Transformer 处理图像表示 |
| Diffusion | 一类生成建模框架 | 学习从噪声恢复数据 |
| Flow Matching | 一类速度场训练方法 | 学习如何把分布运输到目标分布 |

一个 DiT 架构的模型可以采用扩散相关的训练方式，也可以用于流匹配。

因此 **DiT 不等于 Flow Matching，也不等于 DDPM**。

它们处在不同的概念层级。

## 7.3 CFG：为什么提示词能够控制图片？

CFG，全称 Classifier-Free Guidance。

假设我们输入：

“画一只戴红色帽子的猫。”

模型可以在不同条件下预测生成方向：

- 有条件预测：考虑“戴红色帽子的猫”
- 无条件预测：不使用这条具体文本条件

然后把两种预测组合起来，加强文本条件对生成的影响。

在常见的噪声预测表述下，可以写成：

$$
\hat\epsilon
=
\epsilon_{\mathrm{uncond}}
+w(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}})
$$

这里 $w$ 就控制条件引导的强度。

一般来说：

- 引导太弱，可能不够听提示词。
- 引导过强，可能损害自然度或多样性。
- 合适的引导，可以平衡文本一致性和生成质量。

类似的引导思路，也可以用于速度预测形式的模型。

---

# 第八课：把所有概念真正串起来

你可以用下面这张知识地图来记忆。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" align="center" padding={3}>
    **Generative Model · 生成模型**

    <text size="xs" color="secondary">目标：学习真实数据分布，并生成新样本</text>
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="rgba(34,120,185,0.09)" radius="lg" padding={3} gap={2} align="center">
        <text weight="semibold" textAlign="center">Diffusion</text>
        <text size="xs" textAlign="center">典型思路：学习去噪或 Score</text>
        <divider color="subtle"/>
        <text size="sm" textAlign="center">DDPM</text>
        <text size="sm" textAlign="center">DDIM Sampling</text>
        <text size="sm" textAlign="center">SDE / ODE</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(38,145,105,0.09)" radius="lg" padding={3} gap={2} align="center">
        <text weight="semibold" textAlign="center">Flow Matching</text>
        <text size="xs" textAlign="center">典型思路：学习速度场</text>
        <divider color="subtle"/>
        <text size="sm" textAlign="center">Velocity Field</text>
        <text size="sm" textAlign="center">Rectified Flow</text>
        <text size="sm" textAlign="center">ODE / OT Paths</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center" align="center" gap={2}>
    <icon name="arrow-down" color="tertiary"/>
    <text size="xs" color="secondary">这些框架存在数学联系，也可以共享部分设计</text>
    <icon name="arrow-down" color="tertiary"/>
  </row>
  <box border radius="lg" padding={3} gap={2} align="center">
    **实际生成系统还需要其他组件**

    <grid columns={3} gap={2}>
      {#each [{a:"U-Net / DiT",b:"网络结构"},{a:"Latent Space",b:"生成空间"},{a:"CFG",b:"条件引导"}] as item}
        <grid-item>
          <box background="surface-secondary" radius="md" padding={2} gap={1} align="center">
            <text weight="medium" size="xs" textAlign="center">{item.a}</text>
            <text color="secondary" size="3xs" textAlign="center">{item.b}</text>
          </box>
        </grid-item>
      {/each}
    </grid>
  </box>
</box>

上面这张图是一张概念导航图，不是严格的分类树。例如扩散路径也可以通过 Flow Matching 的方法训练。

## 最后，做一个小小的课堂测验 ✍️

试着不看上文，回答这三个问题。

{@body const questions = [
{q:"1. 经典 DDPM 中，神经网络常见的训练目标是什么？",a:["直接输出一张完整图片","预测加入图像的噪声","预测图像的类别"],correct:1,exp:"正确！经典 DDPM 的常见训练目标是预测噪声，再通过采样器把这种预测用于逐步生成。"},
{q:"2. 在直线路径 Flow Matching 中，网络主要学习什么？",a:["当前状态的运动速度","图片的像素数量","只学习最终图片的标签"],correct:0,exp:"正确！网络学习随时间和位置变化的速度场，以便把初始噪声分布运输到数据分布。"},
{q:"3. Flow Matching 与扩散模型的关系是什么？",a:["两者完全没有关系","Flow Matching 一定只能走直线","它们有不同的典型训练表述，但数学上存在联系"],correct:2,exp:"正确！Flow Matching 可以使用扩散概率路径，扩散模型也存在对应的概率流 ODE。"}
]}
{@body const [answers,setAnswers] = DIL.useState({})}
<box border radius="xl" padding={3} gap={3}>
  {#each questions as q,i}
    <box gap={2} key={i}>
      <text weight="medium">{q.q}</text>
      <radio-group direction="col" value={answers[i]===undefined?"":String(answers[i])} onChange={v=>setAnswers(prev=>({...prev,[i]:Number(v)}))}>
        {#each q.a as a,j}
          <radio value={String(j)}>{a}</radio>
        {/each}
      </radio-group>
      {#if answers[i]!==undefined}
        <row gap={2} align="start">
          <icon name={answers[i]===q.correct?"check-circle-2":"info"} color={answers[i]===q.correct?"success":"secondary"}/>
          <text size="sm" color="secondary">{answers[i]===q.correct?q.exp:"再想想：正确答案是「"+q.a[q.correct]+"」。"+q.exp}</text>
        </row>
      {/if}
    </box>
    {#if i<questions.length-1}
      <divider color="subtle"/>
    {/if}
  {/each}
  {#if Object.keys(answers).length===questions.length}
    <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
      <title size="xl" tabularNums>{questions.filter((q,i)=>answers[i]===q.correct).length} / 3</title>
      <text size="sm" color="secondary">{questions.every((q,i)=>answers[i]===q.correct)?"太棒啦，核心概念已经掌握！🎉":"回顾错题，再挑战一次。继续加油！"}</text>
      <button color="secondary" variant="outline" onClick={()=>setAnswers({})}>重新测试 <icon name="rotate-ccw" inline/></button>
    </box>
  {/if}
</box>

## 🎓 今天最值得记住的 5 句话

1. **生成模型**：学习真实数据的分布，从中生成新的样本。
2. **扩散模型**：典型思路是学会识别噪声，再从噪声逐步生成数据。
3. **流匹配**：通过学习速度场，把简单的初始分布运输到复杂的数据分布。
4. **SDE 和 ODE**：分别描述包含随机扰动和不包含额外随机扰动的连续运动过程。
5. **DDPM、DDIM、Rectified Flow、DiT**：并不是同一层级的概念，需要分别理解训练方法、采样过程和网络架构。

如果你想继续深入，我建议下一堂课专门讲：

**“为什么预测噪声就能生成图片？为什么预测速度也能生成图片？从数学上看，$\epsilon$-prediction、$x_0$-prediction、$v$-prediction 之间究竟有什么关系？”**

把这几个问题搞懂之后，你对扩散模型和 Flow Matching 的理解就能真正从“知道大概原理”迈进到“能够看懂论文与模型代码”的层次了。