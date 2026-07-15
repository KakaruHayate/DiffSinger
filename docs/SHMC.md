# SHMC（Shift Mouth-opening Curve）参数说明

本文档说明 `feat/shcm-mouth-opening-merge` 分支引入的口型开度（mouth opening）
相关功能与全部配置参数，覆盖三条使用路径：

1. **变化模型预测口型曲线**（`predict_mouth_opening`）
2. **声学模型以口型曲线为条件**（`use_mouth_opening_embed`，SHMC 的 teacher）
3. **SHMC 自蒸馏**（`use_shift_mouth_opening_embed`，SHMC 的 student）——
   让声学模型学会响应一个帧级相对偏移量 `alpha ∈ [-1, 1]`，
   使用户在推理时可以**无需真实口型曲线**，直接用一个滑杆控制口型开合趋势。

---

## 0. 背景：OPEC 与 R3MOE 口型提取器

### 0.1 什么是 OPEC

**OPEC（mouth OPEning Curve，开口度曲线）** 是一条帧级连续曲线，
描述歌唱/说话过程中嘴部张开的程度。本分支代码中 `opec` 即指这条曲线
（如 `calculate_shifted_opec`、`opec_min` / `opec_max`）。

OPEC 的定义源自面部捕捉的 blendshape 参数（ARKit 规范）。
数据采集端（[LipsSync](https://github.com/KCKT0112/LipsSync)）记录
`jawOpen` 与 `mouthClose` 两路 blendshape，训练提取器时支持多种目标曲线定义，
其中推荐的 **修正开口度（corrected jawOpen）** 为：

```
OPEC = jawOpen × (1 − mouthClose)
```

即下颌张开程度经"抿嘴"修正后的有效开口度——下颌打开但双唇闭合
（如哼鸣 /m/）时 OPEC 仍接近 0。理论值域 `[0, 1]`（0 = 闭口，1 = 全开），
实际歌唱数据的常见有效范围约为 `[0.06, 0.8]`
（这也是 `shift_mouth_opening_args.opec_min/opec_max` 默认值的来源）。

### 0.2 R3MOE：从音频估计 OPEC

歌声数据集通常没有面捕数据，因此需要从**纯音频**反推 OPEC。
[R3MOE](https://github.com/KakaruHayate/R3MOE)
（**R**ecurrentNN × **R**egression × **R**egularized based
**M**outh **O**pening **E**stimation）就是这个提取器：

- **输入**：波形的 mel 特征；**输出**：帧级 OPEC 曲线
- **架构**：BiLSTM 回归网络（本分支 `modules/estimators/nets.py` 中的
  `BiLSTMCurveEstimator` 即其推理端实现）
- **训练方式**：半监督（SSL）——少量 LipsSync 面捕标注数据
  （建议 10h+）+ 大量无标注纯人声（建议 50h+），
  结合 R-Drop、Temporal Ensembling、Mean Teacher 等一致性正则方法，
  提升对未见歌手（out-of-distribution）的泛化
- **数据来源**：面捕标注数据来自
  [DiffSinger 口型数据共建计划](https://github.com/openvpi/DiffSinger/discussions/235)，
  无标注数据来自 M4Singer / OpenSinger / PopCS / GTSinger 等公开歌声数据集

用 R3MOE 训练出的 checkpoint（及其同目录 `config.yaml`）即为本分支
`mouth_opening_estimator_ckpt` 所需的文件。DiffSinger 侧只做推理，
不涉及提取器的训练。

### 0.3 为什么要把 OPEC 转化为 SHMC —— 解耦用户操作面

直接把 OPEC 作为声学模型的绝对条件（`use_mouth_opening_embed`）虽然可行，
但把它暴露给最终用户会带来**多参数耦合**的操作问题：

1. **OPEC 与音素/音高/响度天然强相关**。开口度大部分由"唱的是什么音素、
   多高、多响"决定——用户在调音软件里若直接手绘一条绝对 OPEC 曲线，
   必须先"脑补"出这条曲线在当前乐句下的合理基线，画错了
   （如在 /m/ 上画大开口）会与其他条件矛盾，产生脏音质；
2. **改词、改音高之后绝对曲线全部作废**。绝对 OPEC 依附于具体乐句，
   上游任何改动都要求用户重画整条曲线；
3. **用户真正想表达的是"更开一点/更闭一点"的相对意图**，
   而不是开口度的绝对物理值。

SHMC 的做法是把操作面从**绝对曲线**换成**相对偏移量 alpha ∈ [-1, 1]**：
模型内部通过自蒸馏学到"当前语境下合理的 OPEC 基线"（由 teacher 隐式提供），
用户的 alpha 只表达相对这条基线的偏移方向和幅度。效果上：

- `alpha = 0` 即"不干预"，模型自动给出符合音素/音高/响度的自然口型——
  **默认无操作成本**；
- 正/负偏移始终相对合理基线插值（见第 3.2 节公式），
  不会画出与语境矛盾的曲线；
- 改词改音高后 alpha 语义不变，无需重画；
- 单一标量/包络即可驱动，天然适合做成 UI 滑杆或简单包络线，
  与 DiffSinger 现有的 `pitch_expr`（表现力滑杆）操作范式一致。

一句话：**OPEC 是物理量，SHMC 是操作量**——转化的目的是把
"与多个上游参数耦合的绝对物理曲线"折叠成
"单一、语境无关、默认为零的相对控制量"，降低用户操作负担。

---

## 1. 数据准备（binarizer 共用参数）

口型开度曲线由 R3MOE 的 BiLSTM 曲线估计器（见第 0.2 节）从波形中提取，
再经正弦平滑后写入二值化数据。以下参数在
`use_mouth_opening_embed` / `use_shift_mouth_opening_embed` /
`predict_mouth_opening` 任一开启时生效。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mouth_opening_estimator_ckpt` | `checkpoints/mouth_opening/model.ckpt` | R3MOE 曲线估计器 checkpoint 路径。**其同目录下必须存在配套的 `config.yaml`**（记录估计器的 mel 参数与值域）。二值化时若文件不存在会直接报错。 |
| `mouth_opening_smooth_width` | `0.06` | 正弦平滑窗宽（秒），与 energy/breathiness 等其他 variance 的平滑方式一致。 |

估计器输出的 OPEC 曲线理论值域为 `[0, 1]`（0 = 闭口，1 = 全开；
定义见第 0.1 节）。注意估计器可以配置越界余量（如 R3MOE 实验配置中
`vmin: -0.15`）以缓解回归边界效应，DiffSinger 侧的 min/max 配置
应与实际使用的估计器 `config.yaml` 一致。

---

## 2. 变化模型侧（variance）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `predict_mouth_opening` | `false` | 让 variance 模型将口型开度作为一个预测目标（与 energy/breathiness/voicing/tension 并列，共享 MultiVariance diffusion）。 |
| `mouth_opening_min` | `0.0` | 归一化下界。 |
| `mouth_opening_max` | `1.0` | 归一化上界。 |

开启后，binarizer 会提取真实曲线作为训练目标；推理时 variance 模型输出的
`mouth_opening` 曲线可交给下游以 `use_mouth_opening_embed` 训练的声学模型使用。

---

## 3. 声学模型侧 —— 两种互斥的条件方式

### 3.1 `use_mouth_opening_embed`（绝对曲线条件，teacher 模式）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_mouth_opening_embed` | `false` | 将真实口型曲线作为帧级条件注入声学模型（与其他 `use_*_embed` 同机制：`Linear(1, hidden)` 后加进 condition）。 |

以这种方式训练出的声学模型即可作为 SHMC 蒸馏的 **teacher**。
注意：作为 teacher 时要求**除 `use_mouth_opening_embed` 外的所有
`use_energy/breathiness/voicing/tension_embed` 均为 `false`**（训练时会校验）。

### 3.2 `use_shift_mouth_opening_embed`（相对偏移条件，SHMC student 模式）

与 `use_mouth_opening_embed` **互斥**（代码中有断言）。开启后，声学模型接受一个
帧级偏移量 `shift_mouth_opening = alpha ∈ [-1, 1]` 作为条件，
通过对 teacher 的自蒸馏学会「alpha 偏移 → 口型更开/更闭的 mel」的映射。

偏移的数学定义（`utils/shift_mouth_opening_utils.py`）：

```
alpha >= 0:  shifted = opec + alpha * (opec_max - opec)    # 向全开插值
alpha <  0:  shifted = opec + alpha * (opec - opec_min)    # 向闭合插值
结果 clamp 到 [opec_min, opec_max]
```

即 `alpha = +1` 把曲线推到 `opec_max`，`alpha = -1` 推到 `opec_min`，
`alpha = 0` 保持原样；中间值为线性插值，天然适合做成 UI 滑杆。

#### `shift_mouth_opening_args` 子参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `teacher_ckpt_path` | `''`（必填） | 冻结 teacher 的声学 checkpoint 路径。同目录必须有 `config.yaml`。训练启动时会做严格校验（见下）。 |
| `alpha_sigma` | `0.5` | 训练时 alpha 的采样分布：截断正态 `N(0, sigma²)` 截断到 `[-1, 1]`（逆 CDF 采样，O(1) 无拒绝）。sigma 越小，训练越集中在小偏移。 |
| `replacement_prob` | `0.25` | 蒸馏触发概率。训练 batch 先以此概率整体进入蒸馏流程，进入后每个样本再以此概率被替换为 teacher 输出（详见第 4 节）。 |
| `opec_min` | `0.06` | 偏移公式的曲线下界（不是数据裁剪，是 alpha 插值的目标端点）。 |
| `opec_max` | `0.8` | 偏移公式的曲线上界。 |
| `teacher_use_amp` | `false` | teacher 前向包 fp16 autocast，可加速蒸馏采样；对 teacher 输出精度略有影响。 |

#### teacher 校验规则（启动时断言，失败即报错）

- mel 特征完全一致：`audio_sample_rate` / `hop_size` / `win_size` / `fft_size` /
  `audio_num_mel_bins` / `fmin` / `fmax` / `mel_base`
- 数据空间一致：`use_lang_id` / `num_lang` / `use_spk_id` / `num_spk`
- 词典逐语言 **内容哈希** 一致（不仅是路径），`extra_phonemes` 与
  `merged_phoneme_groups` 一致
- teacher 必须 `use_mouth_opening_embed: true` 且其余 `use_*_embed` 全为 false
- teacher 的**网络结构可以不同**（hidden size、层数、diffusion_type 等不要求一致），
  因为蒸馏只消费其推理输出的 mel

---

## 4. SHMC 训练流程（`training/acoustic_task.py`）

每个训练 step（`infer=False`）：

1. **batch 级门控**：以 `replacement_prob` 的概率决定本 batch 是否进入蒸馏
   流程（未命中时零开销，走普通训练）。
2. 命中后为每个样本采样 `alpha ~ TruncNormal(0, alpha_sigma², [-1,1])`
   （每样本一个标量，广播到全部帧）。
3. **样本级掩码**：每个样本再以 `replacement_prob` 决定是否「被替换」。
4. 被替换的样本：用 `calculate_shifted_opec` 得到偏移后的口型曲线，
   送入冻结 teacher 推理，**teacher 生成的 mel 替换该样本的 GT mel 作为训练目标**。
5. 全 batch（含未被替换样本）以 `shift_mouth_opening = alpha` 为条件正常训练。

teacher 常驻显存（`requires_grad=False`、`eval()`），
`teacher_use_amp: true` 可以降低其推理开销。

---

## 5. 推理（`inference/ds_acoustic.py`）

student 模型推理时，`.ds` 参数中新增：

| 参数 | 说明 |
|---|---|
| `shift_mouth_opening` | 空格分隔的帧级曲线（值域 `[-1, 1]`）。**缺省时自动填 0**（等价于不偏移），因此旧的 .ds 文件无需修改即可在 SHMC 模型上推理。 |
| `shift_mouth_opening_timestep` | 上述曲线的时间步长，推理时会重采样对齐。 |

以 `use_mouth_opening_embed` 训练的模型则照常在 `.ds` 中提供
`mouth_opening` / `mouth_opening_timestep`（绝对曲线，值域 `[0, 1]`）。

---

## 6. ONNX 导出（`scripts/export.py`）

| 参数 | 说明 |
|---|---|
| `--freeze_shift_mouth_opening <float>` | 可选，`[-1, 1]`。给定时将 alpha 固化为常量烘焙进 ONNX 图（模型不暴露该输入）；不给定时 `shift_mouth_opening` 作为动态输入暴露，由宿主（如 OpenUtau）逐帧传入。 |

---

## 7. 典型工作流

```text
① 获取 R3MOE 曲线估计器 ckpt（自训练见 https://github.com/KakaruHayate/R3MOE：
   LipsSync 面捕数据 + 无标注人声 SSL 训练；或使用社区发布的现成 ckpt）
   → 配置 mouth_opening_estimator_ckpt（ckpt 同目录放 config.yaml）
② teacher：use_mouth_opening_embed: true（其余 use_*_embed 全 false）
   → binarize（R3MOE 提取 OPEC GT）→ 训练 → 得到 teacher ckpt
③ student：use_shift_mouth_opening_embed: true
   → shift_mouth_opening_args.teacher_ckpt_path 指向 ②
   → binarize（需要 OPEC GT，用于计算偏移曲线）→ 训练
④ 导出 ONNX：暴露 shift_mouth_opening 输入，宿主提供 [-1,1] 滑杆
```

## 8. 注意事项

- **两级 `replacement_prob` 的实际替换率约为 p²**（batch 门控 × 样本掩码，
  默认 0.25 → 实际约 6.25% 的样本以 teacher 输出为目标）。调参时请以
  实际替换率为准考虑训练时长。
- 进入蒸馏流程的 batch 中，**未被替换的样本也会以非零 alpha 为条件**、
  但目标仍是 GT mel。这会在小 alpha 区域引入「忽略 alpha」的正则倾向；
  若希望控制响应更锐利，可提高 `replacement_prob` 或将未替换样本的
  alpha 置零。
- teacher 与 student 的 mel 空间必须严格一致（启动校验只覆盖列出的 key，
  自定义 STFT 修改不在校验范围内）。
- `opec_min` / `opec_max` 决定 alpha=±1 的语义端点，应与估计器实际输出的
  有效值域匹配；设得过宽会让边缘 alpha 落入曲线不可达区域（被 clamp 吸收）。
- **R3MOE 估计器与 DiffSinger 的 mel 参数无需一致**：估计器内部自带独立的
  mel 变换（如 16kHz / 80 bins），推理时按自己的 `config.yaml` 处理波形，
  与声学模型的 mel 空间（44.1kHz / 128 bins）互不干扰。
  需要一致的是 teacher 与 student 之间的 mel 空间（见上）。

## 9. 相关资源

- OPEC 提取器：[R3MOE](https://github.com/KakaruHayate/R3MOE)
- 面捕数据采集：[LipsSync](https://github.com/KCKT0112/LipsSync)
- 曲线可视化：[lips-sync-visualizer](https://github.com/yqzhishen/lips-sync-visualizer)
- 口型数据共建计划：[DiffSinger Discussion #235](https://github.com/openvpi/DiffSinger/discussions/235)
