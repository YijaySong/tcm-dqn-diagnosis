% 第III、IV节中文初稿。正文可直接复制到支持中文的 LaTeX/IEEEtran 主文件中。
% 建议导言区包含：
% \usepackage{amsmath,amssymb,bm}
% \usepackage{algorithm}
% \usepackage{algpseudocode}
% 如需中文算法标题，可在导言区添加：\floatname{algorithm}{算法}

\section{问题描述}
\label{sec:problem_formulation}

\subsection{任务定义}

证候要素辨证旨在根据患者当前的刻下症信息，识别能够刻画其病机状态的一组证候要素。
该任务具有三个区别于普通单标签分类的基本特征。第一，同一患者可同时具有气虚、血瘀、
痰浊等多个证候要素，因而输出是一个集合而非单个类别；第二，不同患者对应的证候要素
数量并不固定，模型需要自主确定输出集合的基数；第三，证候要素之间并非相互独立，已经
选出的要素会影响后续要素的选择及停止时机。因此，本文不将该任务处理为一次性独立多
标签判别，而将其表述为一个带有自动终止动作的序贯集合构造问题。

记数据集为
\begin{equation}
    \mathcal{D}=\left\{\left(X_i,Y_i\right)\right\}_{i=1}^{N},
    \label{eq:dataset}
\end{equation}
其中，$X_i\subseteq\mathcal{V}$ 表示第 $i$ 个病例的刻下症集合，
$Y_i\subseteq\mathcal{L}$ 表示其真实证候要素集合。症状词表与候选证候要素集合分别为
\begin{equation}
    \mathcal{V}=\{v_1,v_2,\ldots,v_M\},\qquad
    \mathcal{L}=\{\ell_1,\ell_2,\ldots,\ell_K\},
    \label{eq:vocabularies}
\end{equation}
其中，$M$ 和 $K$ 分别表示症状特征数和候选证候要素数。

\subsection{集合选择形式化}

从组合优化角度看，模型需要从候选集合 $\mathcal{L}$ 中选出一个规模适当的子集，使其
尽可能覆盖真实证候要素，同时避免引入不属于真实集合的冗余要素。推理阶段的目标集合
$Y_i$ 不可见，模型只能根据症状集合 $X_i$ 估计哪些证候要素应被纳入最终集合。因此，
该任务可以看作目标集合未知条件下的学习式集合覆盖：训练阶段使用 $Y_i$ 评价集合质量，
推理阶段则仅依据 $X_i$ 和当前已选集合进行决策。

对于任意预测集合 $A\subseteq\mathcal{L}$ 和真实集合 $Y$，为每个候选证候要素赋予
非负权重 $w_k$，并定义
\begin{align}
    \operatorname{TP}_{w}(A,Y)&=\sum_{\ell_k\in A\cap Y}w_k,\\
    \operatorname{FP}_{w}(A,Y)&=\sum_{\ell_k\in A\setminus Y}w_k,\\
    \operatorname{FN}_{w}(A,Y)&=\sum_{\ell_k\in Y\setminus A}w_k.
    \label{eq:weighted_counts}
\end{align}
其中，$\operatorname{TP}_{w}$ 表示已覆盖的目标质量，$\operatorname{FN}_{w}$ 表示尚未
覆盖的目标质量，$\operatorname{FP}_{w}$ 表示错误选择产生的代价。集合一致性函数定义为
\begin{equation}
    \Phi_{\beta}(A,Y)=
    \frac{(1+\beta^2)\operatorname{TP}_{w}(A,Y)}
    {(1+\beta^2)\operatorname{TP}_{w}(A,Y)
    +\beta^2\operatorname{FN}_{w}(A,Y)
    +\operatorname{FP}_{w}(A,Y)}.
    \label{eq:set_utility}
\end{equation}
当分母为零时令 $\Phi_{\beta}(A,Y)=0$。参数 $\beta$ 用于调节漏选与误选的相对影响。
式~\eqref{eq:set_utility} 一方面抑制通过无约束增加标签数量获得的虚假覆盖，另一方面直接
评价预测集合与真实集合的整体一致性。

模型通过动作序列逐步构造预测集合。令
\begin{equation}
    A_t=\{a_0,a_1,\ldots,a_{t-1}\}\subseteq\mathcal{L}
    \label{eq:selected_set}
\end{equation}
表示第 $t$ 步决策前已经选择的证候要素集合。每一步允许加入一个尚未选择的证候要素，
或者执行终止动作 $\mathtt{STOP}$。若策略在第 $T$ 步终止，则最终预测为
\begin{equation}
    \widehat{Y}=A_T,\qquad d_{\min}\leq |A_T|\leq K,
    \label{eq:final_prediction_set}
\end{equation}
其中，$d_{\min}$ 为允许终止前的最小输出数。该约束用于避免模型在尚未形成有效证候集合
时过早停止，其具体取值在实验部分确定。

给定策略 $\pi$，记 $A_T^{\pi}(X)$ 为策略根据症状集合 $X$ 完成序贯选择并自主停止后
得到的集合。本文的总体优化目标为
\begin{equation}
    \pi^{*}=\arg\max_{\pi}
    \mathbb{E}_{(X,Y)\sim\mathcal{D}}
    \left[\Phi_{\beta}\!\left(A_T^{\pi}(X),Y\right)\right].
    \label{eq:coverage_objective}
\end{equation}
直接枚举 $2^K$ 个候选子集的计算量随 $K$ 指数增长，而且无法自然表达证候要素之间的
条件依赖和停止行为。为此，本文将集合构造过程展开为马尔可夫决策过程，并通过学习动作
价值函数近似求解式~\eqref{eq:coverage_objective}。

\section{基于 Dueling DQN 的序贯辨证模型}
\label{sec:method}

\subsection{马尔可夫决策过程}

本文将证候要素集合构造定义为有限时域马尔可夫决策过程
\begin{equation}
    \mathcal{M}=\langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma\rangle,
    \label{eq:mdp}
\end{equation}
其中，$\mathcal{S}$、$\mathcal{A}$、$\mathcal{T}$、$\mathcal{R}$ 和 $\gamma$ 分别表示
状态空间、动作空间、状态转移函数、奖励函数和回报折扣因子。

\subsubsection{状态空间}

病例症状集合 $X_i$ 被编码为多热向量 $\bm{x}_i\in\{0,1\}^{M}$，当前已选集合
$A_t$ 被编码为多热向量 $\bm{b}_t\in\{0,1\}^{K}$。状态定义为二者的拼接：
\begin{equation}
    \bm{s}_t=[\bm{x}_i;\bm{b}_t]\in\{0,1\}^{M+K}.
    \label{eq:state}
\end{equation}
初始状态满足 $\bm{b}_0=\bm{0}$。症状块在整个回合中保持不变；当证候要素
$\ell_k$ 被选中时，$\bm{b}_t$ 的第 $k$ 维由0更新为1。由于 $\bm{b}_t$ 只表示集合
成员关系，不记录选择顺序，任何产生相同已选集合的动作排列都映射到相同状态，从而保持
证候集合表示的置换不变性。

\subsubsection{动作空间与合法动作掩码}

动作空间由全部候选证候要素和一个终止动作构成：
\begin{equation}
    \mathcal{A}=\mathcal{L}\cup\{\mathtt{STOP}\},
    \qquad |\mathcal{A}|=K+1.
    \label{eq:action_space}
\end{equation}
为保证集合中不出现重复元素，在时刻 $t$ 的合法动作集合定义为
\begin{equation}
    \mathcal{A}_{t}^{\mathrm{legal}}=
    \left(\mathcal{L}\setminus A_t\right)
    \cup
    \left\{\mathtt{STOP}:|A_t|\geq d_{\min}\right\}.
    \label{eq:legal_actions}
\end{equation}
策略网络、训练中的下一动作选择以及推理阶段均使用同一合法动作掩码。已选择标签以及
不满足最小输出约束时的 $\mathtt{STOP}$ 被赋予负无穷 Q 值，因而不会被选中。

\subsubsection{状态转移}

环境转移是确定性的。若 $a_t=\ell_k$，则
\begin{equation}
    A_{t+1}=A_t\cup\{\ell_k\},\qquad
    \bm{b}_{t+1}^{(k)}=1;
    \label{eq:label_transition}
\end{equation}
其余状态分量保持不变。若 $a_t=\mathtt{STOP}$，则回合终止并输出 $A_t$。因此，给定
当前症状、已选集合和当前动作后，下一状态与过去的动作顺序无关，满足马尔可夫性。

\subsubsection{奖励函数}

证候要素权重完全由训练集计算。令 $n_k$ 表示 $\ell_k$ 在训练集中的支持度，首先计算
\begin{equation}
    \widetilde{w}_k=
    \left(\frac{\sum_{j=1}^{K}n_j}{K\max(n_k,1)}\right)^{\rho},
    \label{eq:raw_label_weight}
\end{equation}
再进行截断和均值归一化：
\begin{equation}
    w_k=
    \frac{\operatorname{clip}(\widetilde{w}_k,w_{\min},w_{\max})}
    {\frac{1}{K}\sum_{j=1}^{K}
    \operatorname{clip}(\widetilde{w}_j,w_{\min},w_{\max})}.
    \label{eq:normalized_label_weight}
\end{equation}
该权重只在此处定义，后续均视为固定常数。具体参数设置在实验部分给出。

为使逐步决策与最终集合目标一致，本文采用式~\eqref{eq:set_utility} 作为势函数。当智能体
选择一个新的证候要素时，即时奖励为
\begin{equation}
    r_t=\eta\Phi_{\beta}(A_{t+1},Y_i)
    -\Phi_{\beta}(A_t,Y_i)-c_{\mathrm{step}},
    \qquad a_t\in\mathcal{L},
    \label{eq:step_reward}
\end{equation}
其中，$\eta$ 为势函数折扣系数，$c_{\mathrm{step}}$ 为每步代价。当执行终止动作时，
终止奖励为
\begin{equation}
    r_T=\lambda_{\mathrm{stop}}\Phi_{\beta}(A_T,Y_i),
    \qquad a_T=\mathtt{STOP}.
    \label{eq:stop_reward}
\end{equation}
该设计使用同一个集合效用连接逐步选择和终止决策，避免多个含义重叠的奖惩项产生相互
竞争。真实标签 $Y_i$ 仅在训练环境中用于计算奖励；推理阶段策略网络无法访问 $Y_i$。

智能体通过最大化期望累计折扣回报学习策略：
\begin{equation}
    J(\pi)=\mathbb{E}_{\pi}
    \left[\sum_{t=0}^{T}\gamma^{t}r_t\right].
    \label{eq:return}
\end{equation}

\begin{algorithm}[t]
\caption{证候要素集合的序贯构造与自主停止}
\label{alg:inference}
\begin{algorithmic}[1]
\Require 症状向量 $\bm{x}$，在线网络 $Q(\cdot;\theta)$，最小输出数 $d_{\min}$
\Ensure 预测证候要素集合 $\widehat{Y}$
\State $A\gets\varnothing$
\Repeat
    \State $\bm{s}\gets[\bm{x};\operatorname{MultiHot}(A)]$
    \State $\bm{q}\gets Q(\bm{s};\theta)$
    \ForAll{$a\in A$}
        \State $q(a)\gets-\infty$ \Comment{禁止重复选择}
    \EndFor
    \If{$|A|<d_{\min}$}
        \State $q(\mathtt{STOP})\gets-\infty$
    \EndIf
    \State $a^{*}\gets\arg\max_{a\in\mathcal{A}}q(a)$
    \If{$a^{*}=\mathtt{STOP}$}
        \State \Return $\widehat{Y}\gets A$
    \Else
        \State $A\gets A\cup\{a^{*}\}$
    \EndIf
\Until{$|A|=K$}
\State \Return $\widehat{Y}\gets A$
\end{algorithmic}
\end{algorithm}

\subsection{Dueling Q 网络}

为同时建模症状证据、已选证候集合及二者交互，网络首先分别编码症状块与已选标签块：
\begin{align}
    \bm{h}_{x}&=f_x(\bm{x}_i),\\
    \bm{h}_{b}&=f_b(\bm{b}_t),
    \label{eq:dual_encoder}
\end{align}
其中，$f_x$ 和 $f_b$ 均由全连接层、层归一化、ReLU 和 Dropout 组成。随后通过逐元素
乘积表示症状与当前证候集合之间的条件交互，并进行联合编码：
\begin{equation}
    \bm{h}_t=f_{\mathrm{joint}}
    \left([\bm{h}_{x};\bm{h}_{b};\bm{h}_{x}\odot\bm{h}_{b}]\right).
    \label{eq:joint_representation}
\end{equation}

Dueling 结构将状态价值与动作优势分解为
\begin{equation}
    Q(\bm{s}_t,a;\theta)=V(\bm{h}_t;\theta_V)
    +G(\bm{h}_t,a;\theta_G)
    -\frac{1}{|\mathcal{A}|}\sum_{a'\in\mathcal{A}}
    G(\bm{h}_t,a';\theta_G),
    \label{eq:dueling_q}
\end{equation}
其中，$V$ 衡量当前部分集合的整体价值，$G$ 衡量不同候选动作相对于当前状态的优势。
给定合法动作集合后，策略选择
\begin{equation}
    a_t=\arg\max_{a\in\mathcal{A}_{t}^{\mathrm{legal}}}
    Q(\bm{s}_t,a;\theta).
    \label{eq:greedy_policy}
\end{equation}
网络层数、隐藏维度、Dropout率及探索方式等实现参数在实验部分统一说明。

\subsection{标签示范与优先经验回放}

随机初始化的 Q 网络难以在训练初期同时学习证候选择和终止语义。本文不执行独立的监督
预训练，而是在 DQN 更新前利用训练标签构造示范轨迹。对每个训练样本，按照一个固定
规范顺序依次加入 $Y_i$ 中的证候要素，并在全部真实要素选完后加入 $\mathtt{STOP}$，得到
\begin{equation}
    \tau_i^{\mathrm{demo}}=
    \left\{(\bm{s}_t,a_t,r_t,\bm{s}_{t+1})\right\}_{t=0}^{|Y_i|}.
    \label{eq:demonstration_trajectory}
\end{equation}
这些轨迹由标签集合派生，并不表示医生提供了唯一正确的证候输出顺序；其作用是向经验池
提供“选择真实标签并在集合完成后停止”的初始转移。经验池容量在训练前按照示范转移总数
确定，避免示范在正式训练开始前被覆盖。

本文采用优先经验回放。对转移 $j$，其采样概率为
\begin{equation}
    P(j)=\frac{p_j^{\alpha}}
    {\sum_{q}p_q^{\alpha}},
    \label{eq:per_probability}
\end{equation}
其中，$p_j$ 根据 TD 误差更新，$\alpha$ 控制优先采样强度。训练时使用重要性采样权重
$\omega_j$ 修正非均匀采样引入的偏差。

\subsection{$n$步 Double DQN 优化}

设在线网络参数为 $\theta$，目标网络参数为 $\theta^{-}$。对于从经验池采样的转移，
$n$步 Double DQN 目标为
\begin{equation}
    y_t^{(n)}=\sum_{j=0}^{n-1}\gamma^{j}r_{t+j}
    +\gamma^{n}Q\!\left(
    \bm{s}_{t+n},
    \arg\max_{a\in\mathcal{A}_{t+n}^{\mathrm{legal}}}
    Q(\bm{s}_{t+n},a;\theta);
    \theta^{-}\right).
    \label{eq:double_dqn_target}
\end{equation}
若 $n$ 步内回合终止，则移除自举项。在线网络负责选择下一动作，目标网络负责评价该动作，
从而减小传统 DQN 的价值过估计。TD 损失采用带重要性权重的 Huber 损失：
\begin{equation}
    \mathcal{L}_{\mathrm{TD}}=
    \frac{1}{B}\sum_{j=1}^{B}\omega_j
    \operatorname{Huber}\!\left(
    y_j^{(n)}-Q(\bm{s}_j,a_j;\theta)\right).
    \label{eq:td_loss}
\end{equation}

为在价值学习期间保留明确的标签方向，本文设置低权重辅助监督目标。给定当前已选集合
$A_t$，若仍有未选择的真实标签，则辅助目标为 $Y_i\setminus A_t$；若真实标签均已选择，
则目标为 $\mathtt{STOP}$。令 $\bm{z}_t=Q(\bm{s}_t,\cdot;\theta)$，
$\bm{u}_t\in\{0,1\}^{K+1}$ 为上述辅助目标，辅助 Focal 损失为
\begin{equation}
    \mathcal{L}_{\mathrm{aux}}=
    \frac{1}{K+1}\sum_{k=1}^{K+1}
    \left(1-p_{t,k}\right)^{\gamma_f}
    \operatorname{BCEWithLogits}(z_{t,k},u_{t,k};\xi_k),
    \label{eq:aux_focal_loss}
\end{equation}
其中，$p_{t,k}=u_{t,k}\sigma(z_{t,k})+
(1-u_{t,k})(1-\sigma(z_{t,k}))$，$\xi_k$ 为正样本权重，$\gamma_f$ 为聚焦参数。
最终优化目标为
\begin{equation}
    \mathcal{L}=\mathcal{L}_{\mathrm{TD}}
    +\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}.
    \label{eq:total_loss}
\end{equation}
因此，本文方法在优化形式上包含基于 Bellman 目标的强化学习，同时使用标签派生示范、
标签奖励和辅助监督，属于示范增强的监督--强化学习混合框架。

每次梯度更新后，目标网络采用软更新：
\begin{equation}
    \theta^{-}\leftarrow(1-\tau)\theta^{-}+\tau\theta.
    \label{eq:soft_target_update}
\end{equation}

% 该算法较长，在 IEEE 双栏模板中使用跨双栏浮动体，避免单栏高度不足时被无限延后。
\begin{algorithm*}[!t]
\caption{标签示范增强的 DQN 训练过程}
\label{alg:training}
\begin{algorithmic}[1]
\Require 训练集 $\mathcal{D}_{\mathrm{tr}}$，验证集 $\mathcal{D}_{\mathrm{val}}$，训练轮数 $E$
\Ensure 验证集最优在线网络参数 $\theta^{*}$
\State 初始化在线网络 $Q(\cdot;\theta)$ 与目标网络 $Q(\cdot;\theta^{-})$
\State 令 $\theta^{-}\gets\theta$，初始化优先经验池 $\mathcal{B}$
\ForAll{$(X_i,Y_i)\in\mathcal{D}_{\mathrm{tr}}$}
    \State 构造标签示范轨迹 $\tau_i^{\mathrm{demo}}$ 并以 $n$ 步转移写入 $\mathcal{B}$
\EndFor
\State $m^{*}\gets-\infty$
\For{$e=1$ to $E$}
    \State 随机打乱 $\mathcal{D}_{\mathrm{tr}}$
    \ForAll{$(X_i,Y_i)\in\mathcal{D}_{\mathrm{tr}}$}
        \State 重置环境：$A\gets\varnothing$，$\bm{s}_0\gets[\operatorname{MultiHot}(X_i);\bm{0}]$
        \Repeat
            \State 根据合法动作掩码选择 $a_t$，执行环境转移并获得 $(r_t,\bm{s}_{t+1})$
            \State 将 $(\bm{s}_t,a_t,r_t,\bm{s}_{t+1})$ 追加到当前轨迹
            \If{$a_t\neq\mathtt{STOP}$}
                \State $A\gets A\cup\{a_t\}$
            \EndIf
        \Until{回合终止}
        \State 将当前轨迹转换为 $n$ 步转移并写入 $\mathcal{B}$
        \For{$u=1$ to 当前样本的更新次数}
            \State 从 $\mathcal{B}$ 按优先级采样小批量转移
            \State 根据式~\eqref{eq:double_dqn_target} 计算 Double DQN 目标
            \State 根据式~\eqref{eq:total_loss} 计算总损失并更新 $\theta$
            \State 梯度裁剪，并软更新目标网络 $\theta^{-}$
            \State 根据最新 TD 误差更新经验优先级
        \EndFor
    \EndFor
    \State 在 $\mathcal{D}_{\mathrm{val}}$ 上执行算法~\ref{alg:inference}，得到选模指标 $m_e$
    \If{$m_e>m^{*}$}
        \State $m^{*}\gets m_e$，保存 $\theta^{*}\gets\theta$
    \EndIf
\EndFor
\State \Return $\theta^{*}$
\end{algorithmic}
\end{algorithm*}

\subsection{训练与推理流程}

训练阶段首先使用标签派生轨迹初始化经验池，随后对训练样本执行序贯交互、$n$步转移存储、
优先经验采样和 Double DQN 更新。每个训练轮次结束后，模型在验证集上按照算法
~\ref{alg:inference} 完成自主停止推理，并根据预先指定的验证指标保存最优检查点。
测试集不参与参数更新或模型选择，仅在恢复验证集最优模型后评价一次。

推理阶段只输入症状向量和模型已经输出的证候要素集合。模型重复执行合法动作筛选与
Q值最大化：选择证候要素时更新状态，选择 $\mathtt{STOP}$ 时输出最终集合。真实标签、
奖励函数和辅助监督目标在推理阶段均不可见。该过程同时保留逐步选择轨迹和终止位置，
便于解释模型的辨证路径并开展错误分析。

% 建议后续配图：
% 1. 总体框架图：症状集合 -> Dueling DQN -> 证候动作/STOP -> 环境与经验回放。
% 2. 状态更新图：固定症状块与逐步增长的已选标签块。
% 3. 训练流程图：标签示范、在线轨迹、PER、Double DQN与辅助监督损失。
