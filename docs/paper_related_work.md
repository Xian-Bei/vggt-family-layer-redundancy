# Related Work 调研草稿

本文将相关工作分为三类：静态层压缩、动态层压缩，以及面向 3D foundation model 的压缩。整体来看，近两年的主流研究脉络已经比较清晰：在通用 Transformer/LLM 场景中，静态深度压缩仍以层剪枝、层替换、层合并与重参数化为主；动态压缩则更多依赖早退、按 token 自适应深度、以及自推测解码等机制；而在 3D foundation model 场景中，公开工作相对更少，且大多优先处理全局注意力、token 数量和量化部署等瓶颈，而不是直接做 block-level depth compression。

除了“静态/动态”这一按执行方式划分的视角外，层冗余相关工作还可以沿两个彼此正交的维度重新组织：其一是**如何识别冗余**，即使用什么统计量、相似性度量或性能代理来判断哪些层/区间可以压缩；其二是**如何消除或补偿冗余**，即在识别出冗余后，是直接删除、恒等跳过、轻量替换、蒸馏恢复，还是采用层合并与重参数化。因此，如果后续希望让 related work 更紧密地服务于你的方法设计，那么按“识别策略/指标”和“消除/补偿策略”来组织，可能比单纯按静态/动态划分更贴近问题本身。

## 1. 静态层压缩（所有 token 一视同仁）

静态层压缩的基本思想是在推理前固定压缩后的网络结构，使所有 token、所有样本都走同一条计算路径。这一路线的优势在于部署简单、硬件友好、加速收益稳定，因此依然是大模型深度压缩中最常见、最成熟的一类方法。近两年的工作已经从“直接删层”扩展到“删层后补偿”“层合并”和“结构重写”等更细分的技术族谱。

第一类是**整层剪枝（layer pruning / block removal）**。这类方法首先估计各层的重要性，再直接删除冗余层。较早的代表作 **SLEB**（ICML 2024）通过冗余验证和迭代式 block elimination 压缩 LLM，展示了“以整层为粒度直接删除冗余块”本身就能获得可观收益。随后，**ShortGPT**（Findings of ACL 2025）提出 `Block Influence (BI)`，用层输入与输出表征的相似度衡量该层贡献，并据此进行直接删层。相关结论说明，在大模型中，相当一部分 block 对最终功能的边际贡献较小，因而“直接移除冗余层”本身就是一个非常强的基线。

第二类是**补偿式静态压缩（pruning with compensation）**。这类工作认为，层删除后的精度下降并不完全来自“知识丢失”，还来自隐藏状态尺度、残差流或特征分布被破坏。**LLM-Streamline**（ICLR 2025）就体现了这种思路：它在层剪枝之后引入一个轻量替换模块，以 MSE 对齐的方式近似被删层的功能。**Prune\&Comp** 则进一步强调训练自由的补偿思想，通过修正删层造成的 hidden-state magnitude gap 来恢复性能。此类方法的重要意义在于，它表明静态层压缩并不一定要完全依赖昂贵蒸馏或长时间微调，适当的离线补偿也能显著改善压缩后的可用性。

第三类是**层合并 / 层吸收（merging / collapse）**。相较于“简单删除一层”，这一路线尝试把相邻层的信息融合进更少的 block 中。其背后的核心假设是：相邻层之间往往存在功能重叠，若能显式保留两层的主要变换而非硬删除，就可能得到更好的精度-速度权衡。**MKA**（EMNLP 2024）从 manifold alignment 和 information bottleneck 的角度寻找可合并层；**LaCo**（Findings of EMNLP 2024）则以 layer collapse 为核心，将后部层折叠到前层中；近期的 **CoMe / Layer as Puzzle Pieces** 继续沿着“层拼接 + 分层蒸馏”的方向推进。就思想上看，这一类方法与“显式合并”或“residual 吸收”高度接近，因为它们都不是把某一层当作纯冗余噪声直接抹去，而是把它看成可以被相邻结构吸收、折叠或近似的变换。

第四类是**结构重参数化 / Flattening**。这是近两年比较值得关注的新趋势。相比常规删层，**ReplaceMe**（NeurIPS 2025）尝试用线性化后的 block 近似原始 Transformer block，并将近似映射吸收到邻近结构中，体现出一种 training-free 的“结构简化而非纯删除”思路。更进一步，**FlattenGPT** 将相邻两个 Transformer block “压平”为一个 block，从而在减少深度的同时保留原始各层学到的知识。它的关键点并不只是“减少层数”，而是把 depth compression 重写为一种结构重参数化问题：新的 block 仍保持与原始网络兼容的 Transformer 形式，但其参数承载了多个原始 block 的组合功能。与简单 merge 相比，这类方法更系统，也更贴近你所说的 “Flattening / Re-parameterization” 范式。

总体而言，静态层压缩的研究主线已经从“识别冗余层并删除”逐步发展到“删除后如何补偿”和“如何把多个 block 重写为一个更强的 block”。如果要把现有工作映射到你的 Stage 3 设计空间，那么单层 / 区间 skip 对应的是最基础的 block removal baseline，轻量替换与蒸馏对应 replacement/distillation，显式合并与 flattening 则对应更高阶的 interval compression baseline。与此同时，SLEB、ShortGPT、LLM-Streamline、LaCo、MKA、ReplaceMe、FlattenGPT 这条时间线也说明，近两年的静态深度压缩正在逐渐从“删层 heuristics”走向“结构化重写”。

## 2. 动态层压缩（不同 token 走不同跳层或早退路径）

动态层压缩不再假设所有 token 都需要相同的计算深度，而是让模型在推理时根据 token、样本或上下文难度，决定哪些表示需要更深计算、哪些可以提前退出或跳过部分层。相较于静态压缩，这一方向通常有更高的潜在效率上限，但也会引入路由器训练、批处理失衡、系统实现复杂度和延迟不稳定等问题。近两年的动态路线大致沿三条线演进：早退式动态深度、自推测式动态验证、以及更细粒度的 token-adaptive routing / recursive depth。

最经典的一类是**早退（early exit）**。这类方法在中间层直接产生可用输出，当模型对当前预测足够自信时，就不再继续执行后续层。**EE-LLM**（ICML 2024）从系统角度推动了 early-exit LLM 的大规模训练与推理，说明早退不仅是算法问题，也是训练和并行化问题。随后，**LayerSkip**（ACL 2024）将这一思路进一步系统化：训练时使用分层 dropout 和 early-exit loss，让较浅层也具备可直接输出的能力；推理时则可在中间层退出，或将早层结果作为草稿，再由深层进行校验。它的重要启发是，动态深度不一定需要额外分支头或独立小模型，也可以通过训练策略让原模型自身支持“浅层可用”。

在此基础上，一些工作将动态压缩与**自推测解码（self-speculative decoding）**结合。**Kangaroo**（NeurIPS 2024）通过“双重早退”策略，将浅层子网络和原模型输出头组合成自草稿模型，并在置信度不足时动态终止草稿生成，以降低无效计算。后续的 **CAS-Spec**（NeurIPS 2025）则继续发展这一思路，在 on-the-fly lossless acceleration 的框架下做更细的 cascade adaptive drafting。此类方法的价值主要体现在：它们不仅减少深层前向次数，还试图把“浅层近似 + 深层验证”整合到同一个模型里，从而避免额外草稿模型的存储与同步成本。

进一步地，近期研究开始从“样本级动态深度”走向**token 级动态深度（token-adaptive depth）**。例如 **ADEPT** 关注 token 级早退，使不同 token 在预填充和生成阶段走不同深度；**ASTER** 将动态跳层建模为序列决策问题，通过学习式策略选择推理路径；**Mixture-of-Recursions (MoR)**（NeurIPS 2025）与 **Inner Thinking Transformer (ITT)**（ACL 2025）则更进一步，让不同 token 采用不同递归深度或额外“思考步数”，本质上都是将统一深度的 Transformer 改写成按 token 分配计算预算的模型。对于你的问题设定来说，这一类方法最接近“不同 token 走不同的跳层、早退路径”。

除此之外，还有一类与动态层压缩相邻、但不完全等价的方向是**动态特征复用 / 缓存式跳算**。例如 **ProfilingDiT**（ICCV 2025）并不显式让不同 token 走不同深度，而是通过 profiling 识别 foreground-focused 和 background-focused block，并在不同 diffusion steps 上对一部分 block 进行 selective reuse。它更适合被视为动态 computation reuse 的边界案例，而不是严格意义上的 token-adaptive depth；但从“按输入状态改变实际计算量”的角度看，这类工作同样是动态压缩谱系的重要补充。

不过，动态层压缩也有两个常见局限。第一，多数方法需要重新训练路由器、exit head 或深浅层对齐机制，因此工程成本明显高于静态删层。第二，动态路径对系统实现并不总是友好，尤其在需要保持 batch 并行、高吞吐和确定性时，动态路径可能会部分抵消理论 FLOPs 优势。也正因如此，动态方法尽管在研究上活跃，却未必总是优于一个强的静态压缩基线。

对于 3D 几何模型而言，动态层压缩还面临额外挑战。多视图几何推理通常依赖跨视角的一致性，若不同 token 或不同视图在不同深度上过早退出，可能破坏几何对齐、相机估计和全局匹配的稳定性。因此，虽然动态跳层在 LLM 中很自然，但在 VGGT 或 $\pi^3$ 这类多视图几何模型中，其可行性仍有待更系统的验证。

## 3. 冗余性识别策略与指标

如果把相关工作按“如何发现冗余”来整理，现有方法大致可以分为四类。

第一类是**基于表征变化或层间相似性的识别指标**。这类方法的直觉最直接：如果一个 block 的输入和输出几乎不变，或者相邻层/相隔若干层的输出高度相似，那么这些层很可能存在功能重叠。**ShortGPT** 的 `Block Influence (BI)` 就属于这种思路，它利用层输入输出的相似性评估单层贡献。类似地，**LLM-Streamline** 也使用 hidden-state cosine similarity 来定位连续冗余层。更广义地说，cosine similarity、CKA、PWCCA、MSE、Procrustes-style similarity 或 residual change 等表征一致性指标，都可以放在这一类。它们的优点是计算直观、可解释性强，也天然适合生成上三角冗余热力图；缺点则是它们衡量的是“表征接近”，不一定等价于“删除后任务性能不受影响”。

第二类是**基于任务性能退化的识别策略**。这类方法不再把隐藏状态相似性当作最终目标，而是直接测试删掉某层或某段层后，任务指标下降了多少。**SLEB** 通过冗余验证反复测试 block elimination 后的行为变化；**BlockPruner**（Findings of ACL 2025）则将层内 MHA/MLP 分解到更细粒度，并使用 perplexity 作为 pruning criterion；不少 layer pruning 工作还会使用验证集精度、困惑度或 stability 指标来给层打分。相较于纯表征相似性，这类方法更贴近最终目标，但代价也更高，因为通常需要对每个候选层或候选区间重复运行推理或评测。对于你的问题设定，这一类策略与 Stage 2 中“skip 某层/某区间后观察 GT 指标下降”的思路最接近。

第三类是**基于梯度、敏感度或重要性估计的识别方法**。这类方法尝试从优化视角判断某层是否关键，例如通过梯度、Hessian 近似、扰动敏感度或参数重要性来估计删除某层的代价。相关思路在更一般的结构化剪枝文献中非常常见，也常被作为更理论化的 layer selection 工具。近年的一些工作会进一步把 attention / MLP 子模块拆开，以 normalized impact、Taylor sensitivity 或 block benefit 等更细粒度指标进行打分。相比前两类，这类指标往往不如相似性热力图那样直观，但有时能更早反映“这一层对损失曲面有多敏感”。

第四类是**基于结构先验或行为 profiling 的识别方式**。这类方法不显式训练路由器，而是先观察模型在推理中的行为规律，再据此识别哪些层、哪些模块或哪些计算阶段更适合被复用或简化。视频 diffusion 中的 **ProfilingDiT** 就是一个相对边界化但很有启发的例子：它通过离线分析 block 对前景/背景的偏好以及不同 diffusion steps 的稳定性，决定哪些 block 更适合做 feature reuse。类似地，在 3D 与视觉 Transformer 场景中，不少 token pruning / sparse attention 方法也依赖 attention sparsity、token saliency 或跨视角匹配强度等结构先验来识别冗余。虽然它们不一定属于标准的 layer pruning 指标，但它们说明“冗余识别”并不一定只来自精度回退或层间相似性，也可以来自对模型内部行为模式的 profiling。

从你的 pipeline 视角看，Stage 2 实际上已经同时覆盖了前两类识别策略：一方面用 CKA/cosine/-MSE 等度量刻画层间表征冗余，另一方面用 skip-range 的 GT 指标下降来刻画“删掉之后到底损失多少”。这恰好对应文献中最常见、也最互补的两条路线：**representation redundancy** 与 **performance redundancy**。

## 4. 冗余性消除与补偿策略

如果改按“识别出冗余后如何处理”来组织，相关工作则可归纳为以下几类。

第一类是**直接删除或恒等跳过（skip / remove / identity bypass）**。这是最简单也最常见的 baseline：一旦某层被判定为冗余，就直接删除该层，或者在实现上用 `Identity` 替换，使输入直接穿过该 block。**SLEB**、**ShortGPT** 以及许多 reverse-order 或 importance-based layer pruning 方法都属于这一路线。这类方法的最大优点是实现简单、推理路径清晰、加速收益最稳定，因此常被当作一切更复杂方法的起点。缺点同样明显，即它默认被删层的功能可以完全丢弃，因此在压缩率较高时往往会造成显著精度下降。

第二类是**轻量替换（replacement）**。这类方法承认被删层中仍然包含一定有效变换，因此不把它完全置为恒等映射，而是用更便宜的模块近似原 block，例如线性层、简化 MLP、低秩模块或小型 adapter。**LLM-Streamline** 中的 layer replacement 和 **ReplaceMe** 中的 transformer block linearization 都属于这一思路。对你的问题来说，这一类方法正对应“用简单层近似冗余层”的 baseline family。

第三类是**蒸馏与恢复（distillation / recovery）**。这类方法通常和前两类结合出现：先删层或替换层，再通过知识蒸馏、短程 finetune、输出对齐或特征对齐，把压缩后模型拉回教师模型的行为分布。**LLM-Streamline**、**CoMe** 以及不少自推测解码工作都在不同层面使用了对齐或蒸馏。其核心作用不是定义压缩结构，而是在结构变化之后补偿精度损失。也因此，在不少工作中，蒸馏更适合被视为一种“后处理恢复机制”，而不是独立的结构操作。

第四类是**层合并、吸收与区间压缩（merge / collapse / absorb）**。这类方法介于“直接删除”和“重新训练一个替代块”之间。其核心思想是：如果若干相邻层功能高度重叠，那么与其简单删掉其中某些层，不如把它们的主要变换吸收到更少的 block 中。**MKA**、**LaCo**、**CoMe**、一些 residual absorption 思路以及区间级压缩方法都可放入这一类。相较于单层 skip，这一类方法更适合处理“连续若干层共同冗余”的情形。

第五类是**重参数化与 flattening**。这类方法不只是删除或吸收，而是显式重写 block 结构与参数，使多个原始 block 被重新表示成一个新的 block。**FlattenGPT** 就是这一范式的代表，而 **ReplaceMe** 也可被视作更偏 training-free 的重参数化近邻。它的重要性在于，这类方法把“冗余消除”从局部 patch 式修补提升为系统性的结构重写，因此在高压缩率下往往比简单删层更有潜力。

第六类是**动态复用与缓存式补偿**。这类方法并不永久改变网络结构，而是在运行时对部分层或模块跳过重算，改为复用历史特征、缓存中间状态或采用条件计算。严格说它们更接近动态计算优化，而不是静态 depth compression；但从“如何减少冗余计算”这一更宽泛视角看，它们同样属于冗余消除策略的一部分。像 **ProfilingDiT** 这类方法就可以作为边界案例放在这一小类：它不是永久删层，而是在特定 step 上对一部分 block 进行 selective reuse。

如果用这套视角来组织你的 Stage 3，那么最自然的 baseline 谱系其实非常清楚：`skip/identity` 是最小基线，`replacement` 是轻量近似基线，`merge/collapse` 是区间压缩基线，`flattening/re-parameterization` 是更强的结构重写基线，而 `KD/短程恢复` 则可作为上述若干基线共享的补偿模块。

## 5. 3D Foundation Model 压缩（不限于层压缩）

与 LLM 压缩相比，针对 3D foundation model 的压缩研究目前明显更少，且问题结构也不完全相同。以 **VGGT**（CVPR 2025）和 **$\pi^3$**（ICLR 2026）这类 feed-forward multi-view geometry model 为例，其核心瓶颈通常不是词表输出，而是多视图 token 数量巨大、全局注意力复杂度高、长序列输入下显存和时延迅速增长。因此，现有工作更多从**注意力稀疏化、token 压缩、序列级缓存设计和量化部署**切入，而非直接沿用 LLM 中成熟的层剪枝套路。也正因为该方向还很新，当前不少与 VGGT 直接对应的压缩工作仍以最新顶会论文或预印本的形式出现。

最直接的一类工作是**围绕 VGGT / $\pi^3$ 全局注意力瓶颈的结构加速**。**FastVGGT** 指出，VGGT 在长序列下的主要瓶颈来自 global attention，并观察到注意力图中存在 token collapse 现象，因此提出一种面向 3D 场景的、training-free 的 token merging 机制。该方法通过保留参考 token、显著 token 以及具有空间均衡性的采样 token 来减少冗余计算，在 1000 张图像输入上可实现约 `4x` 加速。这条路线说明，对于 3D foundation model，token 冗余往往先于层冗余暴露出来。

与 token merging 不同，**Faster VGGT** 进一步从注意力算子本身入手。该工作分析了 VGGT 与 $\pi^3$ 的全局注意力矩阵，发现大量注意力概率质量集中在少量 patch-patch 交互上，因此用高度优化的 **block-sparse global attention** 替代稠密注意力，在无需重新训练 backbone 的前提下获得最高约 `4x` 的推理加速。相比 FastVGGT，Faster VGGT 更像是一种算子级 retrofit：它不显式改变网络深度，而是直接把几何匹配中的结构稀疏性转化为高效 kernel。

第三类代表性工作是 **FlashVGGT**。它认为，在多视图几何中并不一定需要对所有图像 token 做全局两两注意，而可以先把每帧空间信息压缩成少量 descriptor token，再让原始 token 与 descriptor 做 cross-attention。这样，全局注意力复杂度可从 $\mathcal{O}(N^2)$ 降到 $\mathcal{O}(N^2/r^2)$，并支持基于 descriptor cache 的 chunk-recursive inference，在 1000 图输入下报告了超过 `90%` 的推理加速，同时可扩展到数千视图。与前两者相比，FlashVGGT 更像是对 global reasoning 机制本身的重构。

除结构加速外，**量化**也开始进入 3D foundation model 压缩视野。**TAPTQ (Tail-Aware Post-Training Quantization for 3D Geometry Models)** 明确指出，传统为 2D ViT 设计的 PTQ 方法难以直接迁移到 3D geometry model，因为多视图数据分布更加异质、激活尾部更重、校准代价更高。为此，它提出了渐进式校准集构建、基于三分搜索的量化区间优化，以及 TRE-guided module-wise compensation。值得注意的是，这类工作并不针对 depth compression，却说明 3D foundation model 的压缩问题已经开始从“结构加速”扩展到“部署友好压缩”。

如果把视野进一步放宽到更广义的 3D Transformer / point cloud Transformer 领域，也能看到相似趋势。若干近期工作表明，3D token 数量往往明显冗余，token merging 或 token pruning 可以在较小精度损失下显著降低计算量，例如 **GitMerge3D**（NeurIPS 2025）显示 3D point cloud transformer 往往存在严重 over-tokenization；**Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding**（CVPR 2025）则从任务驱动的 sparse voxel pruning 角度给出了另一条高效 3D 推理路线。这些工作虽然不直接作用于 VGGT，但从侧面支持了一个重要判断：在 3D 场景中，冗余既可能体现在 depth 上，也可能更强地体现在 token 和 attention pattern 上。

因此，现阶段 3D foundation model 压缩的公开文献可概括为三个特点。其一，研究数量仍然有限，远不如 LLM 压缩成熟。其二，已有工作更偏向 token / attention / quantization，而不是 block-level 静态层压缩或动态跳层。其三，许多方法强依赖 3D 几何任务的结构先验，例如跨视角匹配稀疏性、descriptor 可压缩性和多视图校准特性。这也意味着，若要在 VGGT 上开展基于 Stage 2 冗余热力图的层压缩研究，你的工作在当前文献中仍然具有较强的新颖性：它不是泛泛地做 3D 模型加速，而是把“可解释的 block 冗余分析”与“几何基础模型的深度压缩 baseline”直接连起来。

## 6. 可用于论文承接的研究空缺总结

如果需要把上述文献自然引到你的方法，一个比较顺的逻辑是：

1. 在通用 Transformer/LLM 中，静态深度压缩已经从删层发展到补偿、替换和 flattening，但现有工作大多围绕语言模型展开，缺少针对 3D geometry foundation model 的系统研究。
2. 动态层压缩虽然能实现更细粒度的按 token 计算分配，但其训练和部署复杂度较高，而且在多视图几何场景中还存在跨视角一致性被破坏的潜在风险。
3. 当前 3D foundation model 的压缩研究主要集中在 token merging、稀疏注意力、descriptor attention 与 PTQ 等方向，对“层级冗余如何被识别、验证并进一步转化为结构压缩 baseline”讨论较少。
4. 因而，一个自然且有价值的研究问题是：能否先通过统一的冗余分析定位 VGGT 中的冗余 block / 区间，再系统比较 skip、merge、replacement、flattening 等 depth compression baseline，并评估它们在 3D 几何指标上的真实收益与代价。

## 7. 参考文献与链接

### 静态层压缩

- **LLM-Streamline: Streamlining Redundant Layers to Compress Large Language Models**  
  [https://arxiv.org/abs/2403.19135](https://arxiv.org/abs/2403.19135)

- **SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks**  
  [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

- **ShortGPT: Layers in Large Language Models are More Redundant Than You Expect**  
  [https://aclanthology.org/2025.findings-acl.1035/](https://aclanthology.org/2025.findings-acl.1035/)

- **LaCo: Large Language Model Pruning via Layer Collapse**  
  [https://aclanthology.org/2024.findings-emnlp.372.pdf](https://aclanthology.org/2024.findings-emnlp.372.pdf)

- **Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging (MKA)**  
  [https://aclanthology.org/2024.emnlp-main.987/](https://aclanthology.org/2024.emnlp-main.987/)

- **FlattenGPT: Depth Compression for Transformer with Layer Flattening**  
  [https://arxiv.org/abs/2602.08858](https://arxiv.org/abs/2602.08858)

- **CoMe / Layer as Puzzle Pieces: Compressing Large Language Models through Layer Concatenation**  
  [https://arxiv.org/html/2510.15304v1](https://arxiv.org/html/2510.15304v1)

- **ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization**  
  [https://neurips.cc/virtual/2025/poster/115070](https://neurips.cc/virtual/2025/poster/115070)

- **Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation**  
  [https://arxiv.org/html/2507.18212v1](https://arxiv.org/html/2507.18212v1)

### 动态层压缩

- **EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism**  
  [https://proceedings.mlr.press/v235/chen24ae.html](https://proceedings.mlr.press/v235/chen24ae.html)

- **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding**  
  [https://arxiv.org/abs/2404.16710](https://arxiv.org/abs/2404.16710)

- **Kangaroo: Lossless Self-Speculative Decoding for Accelerating LLMs via Double Early Exiting**  
  [https://proceedings.neurips.cc/paper_files/paper/2024/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html)

- **CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs**  
  [https://neurips.cc/virtual/2025/poster/116243](https://neurips.cc/virtual/2025/poster/116243)

- **ADEPT: Adaptive Dynamic Early-Exit Process for Transformers**  
  [https://arxiv.org/abs/2601.03700](https://arxiv.org/abs/2601.03700)

- **ASTER: Adaptive Dynamic Layer-Skipping for Efficient Transformer Inference via Markov Decision Process**  
  [https://dl.acm.org/doi/10.1145/3746027.3755526](https://dl.acm.org/doi/10.1145/3746027.3755526)

- **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation**  
  [https://neurips.cc/virtual/2025/poster/118085](https://neurips.cc/virtual/2025/poster/118085)

- **Inner Thinking Transformer: Leveraging Dynamic Depth for Adaptive Token Computation**  
  [https://aclanthology.org/2025.acl-long.1369.pdf](https://aclanthology.org/2025.acl-long.1369.pdf)

- **Profiling-Based Feature Reuse for Video Diffusion Models (ProfilingDiT)**  
  [https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.pdf)

### 冗余性识别策略与指标

- **ShortGPT: Layers in Large Language Models are More Redundant Than You Expect**  
  [https://aclanthology.org/2025.findings-acl.1035/](https://aclanthology.org/2025.findings-acl.1035/)

- **LLM-Streamline: Streamlining Redundant Layers to Compress Large Language Models**  
  [https://arxiv.org/abs/2403.19135](https://arxiv.org/abs/2403.19135)

- **SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks**  
  [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

- **BlockPruner: Fine-grained Pruning for Large Language Models**  
  [https://aclanthology.org/2025.findings-acl.262/](https://aclanthology.org/2025.findings-acl.262/)

- **What Matters in Transformers? Not All Attention is Needed**  
  [https://openreview.net/forum?id=YLTWwEjkdx](https://openreview.net/forum?id=YLTWwEjkdx)

- **Profiling-Based Feature Reuse for Video Diffusion Models (ProfilingDiT)**  
  [https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.pdf)

### 冗余性消除与补偿策略

- **SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks**  
  [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

- **LLM-Streamline: Streamlining Redundant Layers to Compress Large Language Models**  
  [https://arxiv.org/abs/2403.19135](https://arxiv.org/abs/2403.19135)

- **ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization**  
  [https://neurips.cc/virtual/2025/poster/115070](https://neurips.cc/virtual/2025/poster/115070)

- **Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging (MKA)**  
  [https://aclanthology.org/2024.emnlp-main.987/](https://aclanthology.org/2024.emnlp-main.987/)

- **LaCo: Large Language Model Pruning via Layer Collapse**  
  [https://aclanthology.org/2024.findings-emnlp.372.pdf](https://aclanthology.org/2024.findings-emnlp.372.pdf)

- **FlattenGPT: Depth Compression for Transformer with Layer Flattening**  
  [https://arxiv.org/abs/2602.08858](https://arxiv.org/abs/2602.08858)

- **Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation**  
  [https://arxiv.org/html/2507.18212v1](https://arxiv.org/html/2507.18212v1)

- **Profiling-Based Feature Reuse for Video Diffusion Models (ProfilingDiT)**  
  [https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.pdf)

### 3D foundation model 压缩

- **VGGT: Visual Geometry Grounded Transformer**  
  [https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf)

- **$\pi^3$: Permutation-Equivariant Visual Geometry Learning**  
  [https://openreview.net/forum?id=DTQIjngDta](https://openreview.net/forum?id=DTQIjngDta)

- **FastVGGT: Training-Free Acceleration of Visual Geometry Transformer**  
  [https://arxiv.org/abs/2509.02560](https://arxiv.org/abs/2509.02560)

- **Faster VGGT with Block-Sparse Global Attention**  
  [https://arxiv.org/abs/2509.07120](https://arxiv.org/abs/2509.07120)

- **FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention**  
  [https://arxiv.org/html/2512.01540v1](https://arxiv.org/html/2512.01540v1)

- **Tail-Aware Post-Training Quantization for 3D Geometry Models (TAPTQ)**  
  [https://arxiv.org/html/2602.01741v1](https://arxiv.org/html/2602.01741v1)

- **How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need? / GitMerge3D**  
  [https://openreview.net/forum?id=cFVQJepi4e](https://openreview.net/forum?id=cFVQJepi4e)

- **Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding**  
  [https://cvpr.thecvf.com/virtual/2025/poster/32964](https://cvpr.thecvf.com/virtual/2025/poster/32964)

---

## 0. 先给一个总判断

如果你的目标是“先做几个**靠谱且工程量可控**的 baseline”，我会优先考虑这几个：

1. **Span-MLP Replacement**  
   来自 `LLM-Streamline` 思路，最容易改到你现有 Stage 3。
2. **ReplaceMe-style Linear Replacement**  
   training-free，很适合先做一个“不训练/少训练”的 baseline。
3. **LaCo / RDSC Collapse**  
   很适合做“区间压缩”baseline，和你现在的 skip-range heatmap 对得最紧。
4. **Prune&Comp-style magnitude compensation**  
   作为任何 skip baseline 的“恢复版”很值得做，成本低。
5. **FlattenGPT-style flattening**  
   最有新意，但实现复杂度明显高一档。

如果只做 2 到 3 个，我建议：
- `ReplaceMe-style linear replacement`
- `LaCo-style interval collapse`
- `skip + compensation` 或 `skip + hidden-state KD`

---

# 一、第二类：轻量替换（Replacement）

这类方法的核心不是“直接删掉冗余层”，而是：

> 用一个更便宜的模块 `h` 去近似原来一段 block 的累计作用。

也就是把

$$
x^{(l+n)} \approx F_{l:l+n-1}(x^{(l)})
$$

改成

$$
x^{(l+n)} \approx h(x^{(l)})
$$

其中 `h` 可以是：
- 一个小 `MLP`
- 一个 `SwiGLU`
- 一个单层 transformer block
- 一个线性映射

---

## 1.1 LLM-Streamline：Span Replacement

### 核心思想
先找一段连续冗余层 `[l, l+n-1]`，删掉以后，不让它变成纯 `Identity`，而是训练一个轻量模块 `h` 近似这段跨度映射。

### 层选择公式
他们先找连续区间，基于 hidden state cosine similarity：

$$
l^*(n)=\arg\max_l \cos(x^{(l)}, x^{(l+n)})
$$

意思是：如果第 `l` 层输入和第 `l+n` 层输出已经很像，那么中间这 `n` 层大概率冗余。

### 替换训练公式
收集训练对：

$$
(x_i^{(l^*)}, x_i^{(l^*+n)})
$$

然后训练轻量模块 `h`：

$$
\min_h \mathbb{E}_{(x_i^{(l^*)},x_i^{(l^*+n)})\in \mathcal D}
\operatorname{MSE}(h(x_i^{(l^*)}), x_i^{(l^*+n)})
$$

### 可选替换器
论文里明确说可以用：
- `FFN`
- `SwiGLU-based FFN`
- `Transformer layer`

### 伪代码
```python
# teacher: original model
# span: [l, ..., l+n-1]

pairs = []
for batch in calib_data:
    x_l = teacher.hidden_at(layer=l, batch=batch)
    x_r = teacher.hidden_at(layer=l+n, batch=batch)
    pairs.append((x_l, x_r))

h = SmallMLP(dim=d)  # or SwiGLU / single Transformer block
for step in train_steps:
    x_in, x_tgt = sample(pairs)
    loss = mse(h(x_in), x_tgt)
    loss.backward()
    optimizer.step()

# deployment:
# replace blocks [l:l+n] with h
```

### 迁移到 VGGT 怎么做
对你最自然的版本是：

- teacher = 原始 `global_blocks`
- 选一个区间 `[i..j]`
- 用 Stage1/Stage2 的 calibration scenes
- 记录：
  - `z_i`: 进入 `global_blocks[i]` 之前的 token 表示
  - `z_{j+1}`: 经过 `global_blocks[j]` 后的 token 表示
- 训练一个 `ReplacementBlock(i,j)`，让它拟合 `z_i -> z_{j+1}`

你可以先做 3 个版本：
1. `Linear`
2. `2-layer MLP`
3. `TinyTransformerBlock`（单层 attention+mlp）

### 非常适合你的原因
- 和你现有 skip-range 区间定义完全一致
- 可以直接拿 Stage2 选出的高冗余区间做 replacement
- 训练目标可以先只做 hidden-state MSE，不碰最终几何 loss

### 缺点
- 需要额外训练
- 如果 token 分布跨场景差异大，`h` 容易欠拟合

---

## 1.2 ReplaceMe：Linear Transformation Replacement

这是我觉得你**非常值得实现**的 baseline。

### 核心思想
假设一段连续 block 可以被一个线性映射近似：

$$
F_{i:i+n-1}(x) \approx x + T(\Delta_i(x))
$$

论文里的视角更具体：  
它在 cut point 附近，对 block 输出的一部分中间特征做线性补偿。

### 层选择
对于预设要删的长度 `n`，选 cut index：

$$
i^* = \arg\min_i h(L_i, L_{i+n})
$$

其中 `h` 是 hidden-state distance，论文发现 cosine distance 最有效。

### 线性变换估计
设：
- `M_i` = cut 点处某个中间线性子空间特征（论文里用的是 block 内 MLP output 那一路）
- `Y_i` = 同位置 residual 主干
- `L_{i+n}` = 被删区间后的目标 hidden state

要估计一个线性映射 `T`：

$$
T^* = \arg\min_T h(M_i T + Y_i,\; L_{i+n})
$$

#### L2 解析解
当 `h` 取 L2 时：

$$
T^* = \arg\min_T \| (M_i T + Y_i) - L_{i+n} \|_2^2
$$

闭式解为：

$$
T^* = (M_i^\top M_i)^{-1} M_i^\top (L_{i+n} - Y_i)
$$

#### Cosine 优化形式
他们还用了 cosine 目标：

$$
T^*=
\arg\min_T
\sum_{k=1}^{N}
\left(
1-
\frac{(M_{i,k}T+Y_{i,k})^\top L_{i+n,k}}
{\|M_{i,k}T+Y_{i,k}\|_2 \|L_{i+n,k}\|_2}
\right)
$$

并给了更省内存的形式：

$$
T^* = \arg\min_T \cos(M_iT,\; L_{i+n}-Y_i)
$$

### 融合方式
最关键的一点：  
`T` 不是作为一个新层单独挂进去，而是**融合进相邻线性层权重**，因此不增加新参数。

### 伪代码
```python
# choose span [i+1, ..., i+n]
# collect calibration hidden states

Mi, Yi, target = [], [], []
for batch in calib_data:
    Mi.append(collect_mlp_branch_output(layer=i, batch=batch))
    Yi.append(collect_residual_stream(layer=i, batch=batch))
    target.append(hidden_after_span(layer=i+n, batch=batch))

# solve T
# least squares:
T = solve((M.T @ M), M.T @ (target - Y))

# fuse T into previous/neighbor linear projection
fused_weight = old_weight @ T   # schematic
remove_blocks(i+1, ..., i+n)
```

### 迁移到 VGGT 怎么做
你可以把它简化成一个更适合 VGGT 的版本：

#### 版本 A：最容易
直接学一个线性 projector：

$$
z_{j+1} \approx W z_i + b
$$

这是最朴素的 span-linear replacement。

#### 版本 B：更接近 ReplaceMe
如果你能从 `global_block` 内部拿到某一路线性输出（比如 attention output 或 ffn output），就拟合：

$$
z_{j+1} \approx z_i + T(\Delta z_i)
$$

其中 `\Delta z_i`` 是 block 内某个分支输出。

### 为什么它适合你
- training-free 或 near-training-free
- 只需要 calibration set
- 很适合作为 “比 identity 更强，但比 MLP replacement 更轻”的 baseline

### 风险
- VGGT block 未必像 LLM block 那样“好线性化”
- 区间太长时线性近似可能明显失效

---

## 1.3 这一类的实际建议

### 如果你要做一个最稳的 replacement baseline
我建议直接做这两个：

#### Baseline-R1: `SpanMLP`
$$
z_{j+1} \approx \mathrm{MLP}(z_i)
$$

#### Baseline-R2: `SpanLinear`
$$
z_{j+1} \approx W z_i + b
$$

前者代表 `LLM-Streamline`，后者代表 `ReplaceMe`。

---

# 二、第三类：蒸馏与恢复（Distillation / Recovery）

这一类不是定义新结构，而是在压缩之后把模型“救回来”。

可以分成两种：

1. **显式蒸馏**：teacher-student 对齐
2. **非蒸馏恢复**：比如 `Prune&Comp` 这种补偿

---

## 2.1 最标准的 Hidden-State Distillation

### 核心思想
压缩后的 student 保持你设计的 skip / replacement / merge 结构，然后最小化 teacher 和 student 的中间表示差异。

### 常见损失
#### 隐状态对齐
$$
\mathcal L_{\text{feat}}
=
\sum_{m\in \mathcal M}
\| H_s^{(m)} - H_t^{(\phi(m))} \|_2^2
$$

- `m` 是 student 中保留的层或关键节点
- `\phi(m)` 是对应 teacher 层

#### 输出对齐
如果你更关心最终几何头输出，也可以：
$$
\mathcal L_{\text{out}}
=
\| y_s - y_t \|_2^2
$$

#### Logit / soft target
分类模型里常用：
$$
\mathcal L_{\text{KD}}
=
\tau^2 \mathrm{KL}
\left(
\mathrm{softmax}(z_t/\tau)\;\|\;\mathrm{softmax}(z_s/\tau)
\right)
$$

对 VGGT 来说更实用的是 `feature + output`，而不是 logit KD。

### 总损失
$$
\mathcal L = \lambda_{\text{feat}}\mathcal L_{\text{feat}}
+\lambda_{\text{out}}\mathcal L_{\text{out}}
+\lambda_{\text{task}}\mathcal L_{\text{task}}
$$

### 伪代码
```python
teacher.eval()
student.train()

for batch in data:
    with torch.no_grad():
        t_feats, t_out = teacher(batch, return_intermediate=True)

    s_feats, s_out = student(batch, return_intermediate=True)

    loss_feat = sum(mse(s_feats[m], t_feats[phi(m)]) for m in match_layers)
    loss_out = mse(s_out, t_out)
    loss = lam_feat * loss_feat + lam_out * loss_out

    loss.backward()
    optimizer.step()
```

### 迁移到 VGGT
你非常适合做这两层蒸馏：

1. **block output distillation**  
   对齐压缩区间出口 token features
2. **head output distillation**  
   对齐 pose / depth / track / point map 预测

我会建议先做：
- `span exit feature MSE`
- `final prediction MSE`

而不是一开始就上 GT loss，因为 GT loss 更贵。

---

## 2.2 PPCL：Contiguous Layer Distillation

这个方法虽然在 DiT 上，但对你**非常有启发**，因为它直接处理“连续冗余区间”。

### 核心思想
它的关键不是普通 sequential KD，而是：

> student 的某个压缩区间，直接拿 teacher 对应前一深度的表示作为输入，独立学习该区间该怎么近似 teacher。

这样可以避免“前面压坏了，后面蒸馏目标也歪了”的问题。

### 识别冗余区间
它先对每个 block 训练线性 probe，结合 CKA 轨迹的一阶差分找 contiguous redundant interval。

搜索结果里给出的判定式可以写成：

$$
v = \min \{ k-1 \in [u+2, M] \mid \Delta(u,k) > \Delta(u,k-1) \}
$$

其中 `Δ` 是 CKA 一阶变化量。  
直觉上就是：如果区间还在“连续冗余段”里，差分趋势平滑下降；一旦突然上升，说明到边界了。

### Distillation 思想
不是顺着 student 自己的错误输入往后传，而是：

$$
x_s^{(u)} \leftarrow x_t^{(u)}
$$

让 student 压缩单元直接吃 teacher 的上游表示，然后去拟合 teacher 的区间输出。

你可以把它理解为：

$$
g_{u:v}(x_t^{(u)}) \approx x_t^{(v+1)}
$$

### 更适合 VGGT 的改写
对于你来说，完全可以做成：

- teacher 区间：`global_blocks[i:j]`
- student 区间：一个 replacement / merge / flatten block
- 训练目标：

$$
\mathcal L_{\text{span}} =
\| g_{i:j}(z_t^{(i)}) - z_t^{(j+1)} \|_2^2
$$

可选再加最终预测蒸馏：

$$
\mathcal L = \lambda_1 \mathcal L_{\text{span}} + \lambda_2 \mathcal L_{\text{pred}}
$$

### 伪代码
```python
for batch in calib_data:
    with torch.no_grad():
        z_in  = teacher.hidden_before(i, batch)
        z_out = teacher.hidden_after(j, batch)

    z_hat = compressed_span(z_in)
    loss = mse(z_hat, z_out)

    loss.backward()
    optimizer.step()
```

### 它比普通 KD 好在哪
如果 student 结构改动很大，普通逐层蒸馏容易被前面误差污染；  
PPCL 这种“teacher-input anchored”方式更稳定。

### 对你最有用的点
这是做 **interval baseline** 的非常自然的蒸馏方法。

---

## 2.3 Prune&Comp：非蒸馏恢复 / Magnitude Compensation

这是我觉得你也很值得做的。

### 核心发现
论文认为删层后一个关键问题不是“知识全没了”，而是**隐藏状态幅值变了**。

它定义某层的 magnitude gain ratio：

$$
\delta^{(\ell)}
=
\left(
\mathbb E_{(X^{(\ell)},X^{(\ell+1)})\in\mathcal D}
\frac{1}{C}
\sum_{k=1}^{C}
\frac{\|X_{:,k}^{(\ell+1)}\|_1}{\|X_{:,k}^{(\ell)}\|_1}
-1
\right)\times 100\%
$$

如果某层输出整体比输入“放大”很多，那删掉它以后，后面层接收到的 hidden states 会整体偏小，这就叫 **magnitude gap**。

### 方法本质
不是训练 student，而是：

1. 在 calibration set 上估计删层造成的幅值差
2. 用一个离线缩放因子，把剩余层某些投影权重 rescale
3. 让 hidden state 统计量回到原模型附近

论文里图示表明，补偿作用在 residual-producing projection 上，比如：
- attention output projection
- FFN projection

### 可抽象成的实现
如果删掉区间 `[i..j]` 后，压缩模型在出口处的 token L1/L2 norm 明显偏小，则估计一个标量或向量缩放：

#### 标量版
$$
\gamma = \frac{\mathbb E \|z_t^{(j+1)}\|}{\mathbb E \|z_s^{(j+1)}\|}
$$

然后把 replacement block 或后继 block 的输出乘上 `γ`。

#### 通道版
$$
\gamma_c
=
\frac{\mathbb E |z_{t,c}^{(j+1)}|}{\mathbb E |z_{s,c}^{(j+1)}|}
$$

做逐通道 rescale。

### 伪代码
```python
# after pruning span [i..j]
teacher_norm = 0
student_norm = 0
for batch in calib_data:
    with torch.no_grad():
        z_t = teacher.hidden_after(j, batch)
        z_s = student.hidden_after(j, batch)
    teacher_norm += mean_abs(z_t, dim=(batch, token))
    student_norm += mean_abs(z_s, dim=(batch, token))

gamma = teacher_norm / (student_norm + 1e-6)

# simplest compensation:
student.compensation_scale.data = gamma
# or fuse gamma into output projection weights
```

### 对你这里特别实用的版本
你甚至不必完全照论文做，可以先做个简化版 baseline：

#### Baseline-C1：Skip + Scalar Compensation
$$
z' = \gamma \cdot z_{\text{skip}}
$$

#### Baseline-C2：Skip + Channel Compensation
$$
z'_c = \gamma_c \cdot z_{\text{skip},c}
$$

#### Baseline-C3：Replacement + Compensation
先用 `MLP replacement`，再做 norm matching。

### 它的优点
- 非常便宜
- 可 training-free
- 非常适合给你的 `skip` baseline 加一个恢复版

### 缺点
- 它只能修“尺度错了”，修不了“语义变了”
- 区间太长时单纯 scale 不够

---

# 三、第四类：层合并 / 吸收 / 区间压缩（Merge / Collapse）

这类最适合你的地方在于：

> 你现在 Stage 2 已经有 skip-range heatmap，本来就在找“哪一段连续层一起跳影响最小”。

所以 merge/collapse 几乎天然是你最该做的一类 baseline。

---

## 3.1 LaCo：Layer Collapse / RDSC

这是我觉得最值得你直接仿的一个。

### 核心思想
不是删掉后面几层，而是把它们“折叠”进前面某层。

论文给出的 `RDSC (Reserving-Differences-while-Seeking-Common)` 公式是：

$$
\theta_l^*
=
\theta_l
+
(\theta_{l+1}-\theta_l)
+\cdots+
(\theta_{l+m}-\theta_l)
$$

也就是把后续 `m` 层相对第 `l` 层的“参数差分”吸收到第 `l` 层。

### 直觉
- `\theta_l` 代表共性
- `\theta_{l+k}-\theta_l` 代表每层相对共性的增量
- 把这些增量折回 `\theta_l`，得到一个更强的 collapsed layer

### 选择和接受机制
LaCo 不是无脑 merge。它会：

1. 从高层往低层扫
2. 尝试把 `l+1...l+K` merge 到 `l`
3. 在 few-shot calibration set 上比较 pruned model 和原模型输出表示的 cosine similarity
4. 只有大于阈值 `T` 才接受 merge

### 它的伪代码（论文算法的核心）
```python
M_star = M
l = H - C
while l >= L:
    K = min(C - 1, layer_count(M_star) - l)
    M_tmp = RDSC_merge(M_star, l, K)
    s = avg_cos_sim(M_tmp, M, calib_data)
    if s > T:
        M_star = M_tmp
        l = l - I   # keep interval between collapses
        if l > layer_count(M_star):
            l = layer_count(M_star) - C
    else:
        l = l - 1
return M_star
```

### 迁移到 VGGT 怎么做
这方法其实特别适合你：

#### 版本 A：最直接参数折叠
如果 `global_blocks` 结构相同，可以对 block 内对应参数做：

$$
\theta_i^{\text{new}}
=
\theta_i + \beta_1(\theta_{i+1}-\theta_i) + \cdots + \beta_m(\theta_{i+m}-\theta_i)
$$

先把所有 `\beta_k=1`，就是 LaCo 风格。  
更稳一点可以做衰减：

$$
\beta_k = \rho^{k-1},\quad 0<\rho\le 1
$$

#### 版本 B：只折叠 FFN
因为 FFN 通常更平滑，也更容易合并。

#### 版本 C：只折叠 attention output / FFN down projection
先做部分参数 merge，降低风险。

### 为什么适合你
- 完全支持“区间级” baseline
- 可以直接用你 Stage2 的高冗余区间做候选
- 可以先 training-free，再看要不要短程 finetune

### 风险
- 参数空间线性折叠未必在 VGGT 上稳定
- 区间越长，参数差分叠加越容易爆

---

## 3.2 MKA：Similarity-aware Layer Merging

### 核心思想
它做的事情比 LaCo 更“统计化”：

1. 提取每层激活
2. 做低维 manifold 表示
3. 用 `NPIB` 相似度找最该 merge 的层对
4. 用加权平均融合参数

### 相似度
论文写的是：

$$
S_{ij} = \mathrm{NPIB}(E_i, E_j)
$$

其中 `E_i, E_j` 是 layer manifold embedding。

### 融合权重
$$
\lambda_m = \Psi(\bar s_i,\bar s_j)
=
\frac{e^{S_{ij}}}{e^{S_{ij}} + e^{S_{ji}}}
$$

### 参数融合
$$
\theta_m = \lambda_m \theta_i + (1-\lambda_m)\theta_j
$$

### 伪代码
```python
E = manifold_embed(layer_activations)
S = compute_similarity_matrix(E)   # NPIB or simplified CKA/cosine

while need_more_merge:
    i, j = argmax(S)
    lam = exp(S[i,j]) / (exp(S[i,j]) + exp(S[j,i]))
    theta_m = lam * theta_i + (1 - lam) * theta_j
    merge layer i, j into new layer m
    update S
```

### 迁移到 VGGT
MKA 最复杂的部分是 manifold/NPIB；但你完全可以做一个**简化版**：

#### Simplified MKA for VGGT
把 `NPIB/manifold` 替换成你已经有的：
- `CKA`
- `cosine`
- `-MSE`
- `GT drop score`

然后：
1. 在候选区间中选最相似的一对相邻层/相邻区间
2. 用加权平均合并参数
3. 用 calibration scenes 检查出口表示是否还能接受

即：

$$
\lambda = \frac{\exp(\alpha s_i)}{\exp(\alpha s_i)+\exp(\alpha s_j)}
$$
$$
\theta_{\text{merge}} = \lambda \theta_i + (1-\lambda)\theta_j
$$

### 这个 baseline 的意义
它代表一种“不是直接吸收差分，而是按相似度软融合”的 merge baseline。

### 风险
- 在 block 参数空间里做 convex merge，不一定语义对齐
- 对 attention 的 `W_Q/W_K/W_V` 直接平均有时很危险

所以我建议：
- 先只 merge `MLP`
- attention 先不动，或最后再动

---

## 3.3 你这里最值得做的 merge baseline

我建议你至少可以定义这两个：

### Baseline-M1：LaCo-style Collapse
$$
\theta_i^{new} = \theta_i + \sum_{k=1}^{m}(\theta_{i+k}-\theta_i)
$$

### Baseline-M2：Similarity-weighted Merge
$$
\theta_i^{new} = \lambda \theta_i + (1-\lambda)\theta_j
$$

其中 `λ` 用你 Stage2 的 heatmap 分数定。

---

# 四、第五类：重参数化 / Flattening

这类方法最强，但也最复杂。  
它和 merge 的区别是：

- merge/collapse：还是“几个原 block 合成一个差不多的 block”
- flattening/reparam：**显式重写 block 内部结构**

---

## 4.1 FlattenGPT：先 Flatten，再 Prune

这是目前最值得你认真读实现的一类。

### 核心思想
对相邻两个高度相似的 block，不是删一个保留一个，而是把它们“摊平”成一个**更宽**的单层 block：

$$
B_{\ell-1}, B_\ell \quad \Rightarrow \quad \hat B_{\ell-1,\ell}
$$

然后再把这个 widened block 里的冗余 heads / channels 剪回去。

### Stage 1：选相邻高相似层
用 calibration set 计算相似度矩阵 `S`，贪心选相邻最相似层。

### Stage 1 算法
```python
S = similarity_matrix(calib_data)

while need_flatten:
    (l-1, l) = argmax_adjacent_similarity(S)
    flatten_layers(l-1, l)
    update_similarity_matrix(S)
```

---

## 4.2 Flattening 的具体公式

### 原始 pre-LN transformer
论文写法可抽象成：

$$
\tilde H^{\ell-1} = H^{\ell-1} + \mathrm{MHA}^{\ell}(LN_a^\ell(H^{\ell-1}))
$$
$$
H^\ell = \tilde H^{\ell-1} + \mathrm{MLP}^{\ell}(LN_p^\ell(\tilde H^{\ell-1}))
$$

### MHA flattening
如果 `\ell-1` 和 `\ell` 输入很像，就把两层 attention 输出直接相加：

$$
\mathrm{MHA}^{\ell-1,\ell}(X)
=
\mathrm{MHA}^{\ell-1}(X)+\mathrm{MHA}^{\ell}(X)
$$

这等价于把两层的 heads 拼起来，形成 `2H` heads 的一个更宽 MHA。

论文实现里：
- 先把 norm affine 融入 Q/K/V 投影
- 再拼接参数

例如：

$$
\tilde W_{Q,i}^{\ell-1} = \mathrm{diag}(\alpha_a^{\ell-1}) W_{Q,i}^{\ell-1}
$$

然后拼接：

$$
W_Q^{\ell-1,\ell}
=
[
\tilde W_{Q,1}^{\ell-1},\ldots,\tilde W_{Q,H}^{\ell-1},
\tilde W_{Q,1}^{\ell},\ldots,\tilde W_{Q,H}^{\ell}
]
$$

`K,V` 同理，`W_O` 做纵向拼接。

### MLP flattening
同样，把两个 MLP 输出加起来：

$$
\mathrm{MLP}^{\ell-1,\ell}(X)
=
\mathrm{MLP}^{\ell-1}(X)+\mathrm{MLP}^{\ell}(X)
$$

对于 gated MLP，相当于把 `up/gate` 通道拼起来，把 `down` 做纵向拼接。

---

## 4.3 Flatten 后怎么剪回去

Flatten 后 block 变宽了，但参数还没少。  
所以第二阶段再做 channel/head pruning。

### MHA 头重要性
论文在算法里用每个 head 的响应和输出投影范数定义分数 `f_i`，然后保留 top-k heads。

伪代码可以写成：

```python
for head i:
    f_i = expected_attention_output_i * output_proj_norm_i
keep top-k heads
WQ, WK, WV = WQ @ S_k, WK @ S_k, WV @ S_k
WO = S_k.T @ WO
```

### MLP 通道剪枝
它用 activation correlation 和 ridge leverage score 选 top-k channels。

给定：

$$
C_\psi = \sum_{i=1}^{N}
\left(X_i W_u \psi_g(X_i W_g)\right)^\top
\left(X_i W_u \psi_g(X_i W_g)\right)
$$

通道分数：

$$
s_i = \left[C_\psi (C_\psi + \lambda I)^{-1}\right]_{ii}
$$

选 top-k 通道后，用 Nyström 近似更新 down projection。

### 伪代码
```python
# flatten two adjacent blocks into one wider block
flat_block = flatten(block_lm1, block_l)

# prune MHA heads
head_scores = compute_head_scores(flat_block, calib_data)
keep_heads = topk(head_scores, H)
flat_block.prune_heads(keep_heads)

# prune MLP channels
Cpsi = activation_corr(flat_block, calib_data)
scores = diag(Cpsi @ inv(Cpsi + lam * I))
keep_channels = topk(scores, d_int)
flat_block.prune_channels_with_nystrom(keep_channels)
```

---

## 4.4 迁移到 VGGT 怎么做

严格照搬 FlattenGPT 很难，因为 VGGT 的 block 内部实现和 LLM 不一定完全同构。  
但你可以做一个**弱化版 flattening baseline**：

### Baseline-F1：Output-additive Flattening
对相邻两层 `B_i, B_{i+1}`，定义：

$$
\hat B(x) = B_i(x) + B_{i+1}(x)
$$

然后加一个线性投影把维度/尺度拉回原空间：

$$
\hat B'(x) = P(\hat B(x))
$$

如果两层输入相似，这会比较接近 FlattenGPT 的精神。

### Baseline-F2：Parallel-branch Flattening
保持一个 block 外壳，内部两路并行：

$$
\hat B(x)=x + f_i(x)+f_{i+1}(x)
$$

然后对两路 hidden channels 做 top-k 选择或 1x1 压缩。

### Baseline-F3：Attention/MLP 分开 flatten
只对 `MLP` flatten，不碰 attention。  
这是最稳的版本。

---

# 五、给你一个“可实现性排序”

下面按你现在代码框架的**落地难度**排序：

## 最容易
### 1. `SpanLinear`（ReplaceMe-lite）
- 不训练或少训练
- 直接拟合 `z_i -> z_{j+1}`
- 最适合当第一版 baseline

### 2. `SpanMLP`（LLM-Streamline-lite）
- 训练一个小 MLP
- hidden-state MSE
- 稳

### 3. `Skip + Compensation`（Prune&Comp-lite）
- 在现有 `Identity skip` 上加一个 scale
- 改动最小
- 见效可能很快

## 中等
### 4. `LaCo-style interval collapse`
- 直接参数折叠
- 和你区间热力图最匹配
- 需要小心稳定性

### 5. `Teacher-anchored interval distillation`（PPCL-lite）
- 很适合你
- 但需要蒸馏训练框架

## 最难
### 6. `FlattenGPT-style flatten + prune`
- 最强
- 但需要改 block 内部实现
- 工作量最大

---

# 六、我建议你真正做的几个 baseline

如果你想要一个**论文里既有层次又不至于太难实现**的组合，我建议：

## 方案 A：最实用
1. `Identity Skip`
2. `SpanLinear Replacement`
3. `SpanMLP Replacement`
4. `Skip + Magnitude Compensation`

## 方案 B：更有研究味道
1. `Identity Skip`
2. `LaCo-style Interval Collapse`
3. `Teacher-anchored Interval Distillation`
4. `Flatten-lite`

## 方案 C：最贴合你当前 Stage2
1. 用 heatmap 选 `[i..j]`
2. 在同一候选区间上比较：
   - `skip`
   - `linear replace`
   - `MLP replace`
   - `collapse`
   - `skip + KD`
3. 最后统一报：
   - GT 指标
   - latency
   - params
   - FLOPs
