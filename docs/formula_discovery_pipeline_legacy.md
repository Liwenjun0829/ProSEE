# 零指数和基元公式发现与两阶段拟合

运行入口 `scripts/run_formula_discovery.py` 复用 `data_io.py` 中的 CSV 读取与路径工具，
并从 `generate_zero_sum_bases.py` 获取严格满足指数和为零的不同元数基元。

主流程配置位于 `configs/formula_discovery/pipeline.json`：

- 默认按每个输入数据源随机划分训练集、验证集和测试集，比例为 `50% / 25% / 25%`。训练集用于 LM 拟合；若提供独立 PySR 配置，训练集用于 PySR 搜索，验证集用于树模型与 PySR 的选解及验证集 HV，测试集只用于冻结公式的最终报告、测试集 HV 和气泡图。可通过 `data.split.train_fraction`、`validation_fraction` 和 `test_fraction` 修改比例。

- `bases.arities`、`max_abs_exponent` 控制基元元数和指数范围；
  `bases.allow_binary_reciprocals` 控制是否生成二元纯倒数基元 `1/x1`、`1/x2`，默认
  为 `true`；设为 `false` 时仍保留 `x1/x2`、`x2/x1` 等比值；
- `candidates.max_count`、`two_base_ratio` 控制单项候选规模及双基元比例；
- `combinations.max_count`、`types` 控制加法/乘法组合及抽样上限；
  `combinations.max_terms` 控制每个组合最多包含多少个候选公式，默认 `2`，设置为 `3`、`4` 等即可生成多项组合；
- `fitting.single_sample_count`、`combination_sample_count` 控制拟合抽样；
- `fitting.optimizer` 可设为 `LM` 或 `BFGS`，默认 `LM`。选择 `BFGS` 时，重启次数、每次迭代上限、函数调用上限、梯度容差和初值范围由 `fitting.bfgs` 控制；LM 仍由 `fitting.lm_initializations`、`max_nfev` 等参数控制。
- `fitting.fit_mode` 可设为 `refit`（重新 LM 拟合）或 `reuse_history`（复用输出目录中
  上一次的 `formula_discovery_results.json` 拟合结果）。复用时要求当前 seed 和预计算公式库
  不变，且仍会继续执行结构排序和 Top-K 复拟合；复用模式不会在启动时清理输出目录；
  LM 拟合、排序和搜索阶段会保存检查点；只要历史文件中已有可用拟合结果，下一次即可复用，
  不要求上一次运行完整结束。若可用拟合结果不足以训练排序模型，则跳过排序和搜索并保留当前部分结果；
- 当前多种子实验使用树模型代理辅助迭代搜索，按轮次拟合候选并更新 `gp_pareto_front`；它不会调用独立的轻量 GP 种群演化器。常数显著性筛选与 Log 项合并仍在搜索的既定阶段运行。
- `postprocessing.merge_log_max_r2_drop` 控制常数显著性筛选及 Log 合并检查允许的最大绝对 R² 下降，默认值为 `0.01`；`postprocessing.log_merge_initial_perturbation` 控制 Log 合并重拟合的初始参数扰动幅度。
  非线性算子前的乘法系数不会参与置零检验，但算子参数内部的常数仍会参与检验；初始解筛选统计记录在
  `metadata.gp_initial_seed_selection` 和搜索 metadata 中。
  若同一公式包含至少两个此类受保护系数对应的顶层非线性加法项，程序还会将每一项单独重新拟合为候选；
  拆项必须仍包含全部输入变量（双变量任务即同时含 `x1`、`x2`），原公式不会被替换。这些步骤属于代理辅助搜索的常数检验流程，不产生遗传编程种群。
- `pipeline_mode=3` 时，初始 Top-K 同样执行四状态常数显著性检验，并用检验后的初始 Pareto 前沿建立复杂度区间；多种子实验还会按配置每 5 轮对新增公式执行同样检验。
- 多种子主实验完成后会默认调用 `canonicalize_repeated_structures.py`，只读取主模型结果并写入 `cross_seed_formula_summary/main/`。独立 PySR 实验完成后只 canonicalize PySR 结果，并写入其运行目录下的 `cross_seed_formula_summary/pysr/`。目录包含所有跨 seed 汇总结果（包括只出现在一个 seed 中的公式）；其中 `*_repeated_structures.csv` 与重复结构气泡图只统计出现在多个 seed 中的结构。
- `lm_initializations`、`initialization_range`、`max_nfev` 控制 LM 多随机初始化；当前 `max_nfev` 为 `5000`；
  `log_merge_lm_initializations` 单独控制 Log 合并检查的 LM 重启次数；
  上游初步拟合使用固定的 `single_sample_count`/`combination_sample_count` 和原有拟合顺序。
- `ranking.model_type` 可选 `extra_trees` 或 `xgboost`。
- Top-K 固定采用混合规则：最多由 `ranking.direct_fit_top_k` 个直接 LM 拟合公式按实际拟合 R²排序选出（不使用树模型排序），其余名额只从未参与直接拟合的公式中按树模型排名补足。当前配置为最多 5 个；这保证剩余公式不来自训练集。
- 主实验入口固定使用 `operator_primitive_matrix`；特征编码参数由公式/特征预计算配置管理。
  设置为 `tree_coordinates` 后，所有模型（包括树模型）统一使用与岭回归相同的树坐标特征；特征相关性诊断当前统一关闭。
  设置为 `tree_coordinates_prime_experimental`（别名 `tree_coordinates_prime`）可启用实验性素数编码：节点
  `symbol_onehot` 被单一的固定素数编码替代；节点与第 k 级祖先之间的该字段保存
  `log(节点到第 k 级祖先路径上所有符号素数的乘积)`。为避免溢出，内部以对数求和计算；坐标、子树算子比例、变量比例及它们的乘积特征不变。
  `ranking.tree_coordinate_operator_count_mode` 控制显式树坐标模式中的算子特征，取值为 `ratio` 或 `count`，默认 `ratio`。
  `ratio` 将每个节点子树内的算子计数归一化为算子构成比例；`count` 保留原始计数方式。
  设置为 `value_matrix_svd`（别名 `response_svd`）可改用数值响应特征：每个公式的所有拟合参数先固定为 1，随后在
  `x1,x2 ∈ [0,100]` 的规则网格上计算二维输出矩阵，做有限值 min-max 归一化后提取 SVD 奇异值。网格中由 `log`、除零或其他
  无效运算造成的值会安全填补，不会中断排序。`ranking.response_svd_grid_size` 默认为 `100`，
  `ranking.response_svd_feature_count` 默认为 `20`；该模式只使用这些奇异值输入 surrogate model，不使用公式树特征。
  设置为 `operator_primitive_matrix`（别名 `operator_primitive`）可启用独立的“非线性算子 × 无量纲基元”二值矩阵模式：行按生成器的
  `identity`、`log_abs`、`sin`、`cos`、`log_abs_sin`、`log_abs_cos` 顺序且每类预留两行，列严格复用本次 `bases` 的顺序。矩阵按行优先
  使用原始算子矩阵（`identity`、`log_abs`、`sin`、`cos`、`log_abs_sin`、`log_abs_cos`，每类两行），其中 `sin/cos` 及其 log 复合算子保持区分，再拼接公式响应矩阵的 SVD 特征、人工结构数值特征和树坐标的 RBF 核 PCA 特征。该模式的树模型包含四类特征：算子—基元矩阵、响应矩阵 SVD、人工结构数值特征、树坐标 kPCA 特征。SVD 由 `response_svd_grid_size` 和 `response_svd_feature_count` 控制；树 kPCA 由 `kernel_pca_components` 控制。该模式的树排序训练样本与其他模式一样采用等权处理。
  特征相关性诊断（候选公式采样相关矩阵和 Top-K 热力图）目前对所有 feature mode 统一关闭；不会影响 Top-K 相关性约束本身。
  `model_default` 下的兼容性树坐标仍使用原始计数方式。
  运行时选择模式2时，树坐标取消节点级算子/变量比例，仅保留整棵树的数值特征，并且每个节点只追加与最近祖先的乘积；模式1和模式3保持原有逻辑。
  模式2的后续迭代固定按结构复杂度数值分为5层，候选复杂度上限为25；每轮从5层各选1个公式（基础树排序与随机排列消融使用相同分层策略）。
  运行时选择模式4时沿用模式3的树坐标/数值特征组合，不进行训练前的特征筛选；每个分裂节点从全部树坐标特征中随机抽取 50%，并强制加入全部数值特征。该模式使用分组特征树模型，配置中的全局 `max_features` 不参与此模式。
  在 `tree_coordinates_prime_experimental` 模式下默认使用 `both`（原素数树坐标与归一化
  数值响应矩阵的 SVD 奇异值拼接），运行时不再通过 `input()` 询问。配置中的
  `ranking.tree_coordinates_prime_experimental_feature_mode` 仍可设置为 `tree`、`numeric`
  或 `both`，其他调用接口保持不变。该选择只影响实验性素数模式，`tree_coordinates` 与
  `model_default` 保持原有行为。
  启动脚本选择树特征处理方式 `3`（原始逻辑）后，会进一步询问这三个选项，并仅覆盖本次运行的
  `tree_coordinates_prime_experimental_feature_mode`，不会修改配置文件。
  每次执行初始 LM/初始公式拟合阶段结束后，终端会打印“初始 LM 拟合实际总时长”，并将秒数写入
  `metadata.initial_lm_elapsed_seconds`；复用历史结果时会明确标记为未执行初始 LM 拟合。
  树坐标编码按公式语法树前序遍历，为每个节点编码 `(x, y, symbol, path)` 四个数值特征，
  再 padding 到 `ranking.max_nodes`。其中 `x` 为归一化中序横坐标、`y` 为归一化深度、
  `symbol` 为节点符号编码、`path` 为二进制路径编码。
  显式 `tree_coordinates` 模式将 `symbol` 改为 One-Hot 向量，并在计算乘积前对
  `(x, y, symbol_onehot)` 做逐特征标准化；随后不再复制父节点特征，而是对每个 ancestor
  分别计算逐元素乘积并追加。不存在的 ancestor 槽位使用 0 填充。默认模式保留原有父节点特征和整数编码。
  所有模式都使用每个频段内已拟合公式的 R² 百分位排名监督训练，
  并按预测分数为全部生成公式排序；每个频段内 R² 最高的公式标签为 100，最低为 0；
- `seed` 和 `output.directory` 控制复现种子及输出目录。
- `pysr_baseline.reproducibility_mode` 一键选择 PySR 执行模式：设为 `"multithreaded"` 使用多线程模式
  （`reproducible=false`、`deterministic=false`、`parallelism="multithreading"`、
  `timeout_in_seconds=3600`）；设为 `"reproducible"` 启用严格的同环境复现配置
  （`reproducible=true`、`deterministic=true`、`parallelism="serial"`、
  `timeout_in_seconds=null`）。搜索以固定的 `niterations` 结束，避免机器负载和 Julia 首次编译
  时间改变搜索工作量。串行会牺牲部分吞吐量，但保留 populations、population_size 和迭代预算，
  不降低搜索规模。每个 seed 的配置会保存解析后实际使用的参数。
- `pysr_iteration_timeout_seconds` 限制可复现串行模式下单个 Julia 搜索步的时长，默认 100 秒。超时后 Julia
  尝试中断该步，恢复该步开始前的种群并继续搜索；若 Julia 无法安全中断，`process_watchdog_iteration_timeout_seconds`
  （默认 160 秒）会作为整次 PySR 搜索的兜底无进度时限。`process_watchdog_startup_timeout_seconds` 为 Julia 启动和
  首次进度留出 1800 秒。跳过超时步会改变该次搜索路径；没有超时时，原有固定随机种子和搜索预算不变。
- 每次 PySR 结果记录实际 Python/PySR/NumPy/Julia/SymbolicRegression 版本、Julia 线程数，以及
  PySR 配置、训练/验证/测试数据和 pipeline 源码的 SHA-256 指纹。
- `pysr_baseline.forbid_division_by_nonlinear_structure` 可禁止分母包含非线性结构的除法；
  默认配置启用。由配置的非线性一元算子（如 `log`、`sin`、`cos`）判定；乘法、幂和除法本身不算非线性，
  但分母子树中只要包含这些函数算子就会被拒绝。
  该约束通过 `structural_loss_during_search` 在搜索时施加，并在候选输出阶段再次过滤。
- 当输入包含多个 CSV/formula 数据文件时，Mode 3/4 的 PySR 基线会把训练样本合并进行一次
  `TemplateExpressionSpec` 搜索：公式子树只接收原始特征，各文件 ID 只用于索引类别专属参数向量，
  不会作为普通公式变量参与搜索。验证集和测试集仍按文件分别计算指标，测试集只评估验证集选出的前沿。
  搜索损失对每个文件等权，避免样本数较大的文件单独主导联合拟合。
  `pysr_baseline.shared_template.enabled` 控制此路径，`parameter_count` 控制每个文件可优化的参数槽数
  （默认 4）；公式子树可以将这些类别参数作为输入来构造共享结构。模板路径的 Pareto 候选由 PySR
  原生 `predict(index=...)` 逐个评估，因此不会走普通方程的 SymPy 解析流程。启用
  `structural_loss_during_search` 时，单文件结构约束也会作用于共享的 `f` 子树；联合损失仍通过完整模板
  表达式计算并保留按文件等权权重。特征覆盖约束只检查原始物理特征，不把类别参数计作特征。

## 消融实验

`ablation_experiment.py` 在不修改主 pipeline 的前提下运行
`no_tree`、`no_gp` 和 `no_tree_no_gp` 三种实验，并将每种模式的原有输出格式写入独立子目录。
实验参数集中定义在脚本顶部，不使用命令行参数：

```bash
/home/lwj/miniconda3/envs/py39/bin/python ablation_experiment.py
```

各模式使用相同的数据、随机种子、候选生成和 LM 拟合预算；`no_tree` 使用真实 LM
R² 排序，但复用完整 pipeline 的算子分层 Top-K 选择逻辑。`no_gp` 只关闭常规 GP
进化。脚本会在同一张图中绘制各模式的最终帕累托前沿（复杂度最小化、R² 最大化）。每个结果文件的
`metadata.ablation_experiment` 记录模式、候选池规模、配置哈希、随机种子和选择依据。
根输出目录还会生成 `ablation_summary.csv`、`ablation_summary.json`、
`tree_proxy_pareto_front.csv` 和 `tree_proxy_pareto_front.png`，可直接作为论文消融表格
及帕累托前沿图的来源。默认输出目录为脚本顶部的 `OUTPUT_ROOT`。

## Maritime 实测数据与自动分段

当前 Maritime 数据先由 `preprocessing/Maritime/build_plot_data_dataset.py`
从 `data/raw/Maritime/plot_data2.mat` 生成
`data/processed/Maritime/plot_data2.csv`，主流程以 `data.mode=csv` 读取。
Maritime 配置使用历史模式标识 `data.mode=realdata_mat`，但该模式现在直接读取
`csv_path` 指向的 CSV，不读取 MAT 文件。`data.profile=measured_pathloss` 保留原先的
pooled R²、单变量公式和物理断点分段语义。
转换会过滤非有限值、按 distance 稳定排序，并以 17 位有效数字保存，从而与 MAT
读取器得到的数组及固定 seed 切分完全一致。
启用 `coarse_gp_fine` 分段策略后，
ExtraTrees 与结构排序先使用全局单公式拟合；树模型 Top-K 公式随后用少量均匀切点
粗搜各自的最优分段。GP 在这些粗分段点上以同一公式结构、两段独立常数进行拟合；
GP 最终帕累托公式再穷举合法切点精搜。结果含 `split_index`、`split_value`、
`segment_sample_counts`、`parameters_by_dataset` 与 `segmentation_search_stage`。
Mode 5 的 Maritime 共享结构分段拟合以 pooled R² 最大化、复杂度和断点预测差
最小化作为帕累托目标。断点预测差 `segment_prediction_difference` 在 `split_value`
处计算：左右段公式使用各自拟合参数，对同一个断点输入求值后取绝对差。相邻实测
目标值之差不再计算或作为目标。

```json
"data": {
  "mode": "realdata_mat",
  "profile": "measured_pathloss",
  "name": "measured_pathloss",
  "csv_path": "../../../../data/processed/Maritime/plot_data2.csv",
  "features": ["distance"],
  "target": "path_loss",
  "realdata": {
    "segmentation": {
      "enabled": true,
      "min_segment_samples": 8,
      "breakpoint_prior": {
        "enabled": true,
        "h_t_m": 25.0,
        "h_r_m": 4.0,
        "carrier_frequency_hz": 5800000000.0,
        "speed_of_light_m_per_s": 299792458.0,
        "meters_to_data_units": 0.001
      }
    }
  }
}
```

启用 `breakpoint_prior` 后，代码按 `d_break = 4*h_t*h_r/(c/f)` 精确计算先验点。
本示例得到 `d_break = 7738.6870086 m = 7.7386870086 km`；因此距离数据以 km
表示时，`meters_to_data_units` 设为 `0.001`。由于实测数据只能在相邻样本之间
分段，代码选取分段边界中点距离先验点最近的唯一分段索引，并固定使用该点，
不再进行粗搜或精搜。将 `enabled` 设为 `false`
即恢复原来的全范围搜索。

启用先验时，流程使用 `fixed_breakpoint`：跳过“Top公式粗分段拟合”，
直接对排名公式在固定切点上拟合。

设为 `"enabled": false` 时退回普通的一元
全局拟合。`strategy: "legacy"` 仍可使用旧的直接分段流程，但不推荐用于新实验。
Maritime 实测数据使用统一配置 `pipeline.json`，直接以 CSV 模式运行：


```bash
/home/lwj/miniconda3/envs/py39/bin/python scripts/run_formula_discovery.py \
  --config configs/formula_discovery/pipeline.json
```

运行完整流程：

```bash
cd /home/lwj/另一种NN模型
/home/lwj/miniconda3/envs/py39/bin/python scripts/run_formula_discovery.py \
  --config configs/formula_discovery/pipeline.json
```

只生成候选而不拟合时，把配置中的 `stage` 改为 `generate`。单独生成 1～4 元基元文本：

```bash
python src/prosee/symbolic/core/generate_zero_sum_bases.py --max-variables 4 --max-abs 2 --output-directory artifacts/bases
```

临时禁止二元纯倒数基元时追加 `--disable-binary-reciprocals`。

输出目录包含完整 `formula_discovery_results.json`，以及基元、单项候选、双项组合、
单项全部初始化、单项最佳局部结果、双项最佳全局结果及结构排序 CSV。JSON 和 CSV 均保留
SymPy `srepr`、可读公式、结构/算子/基元元数据；拟合结果含参数、MSE、RMSE、R²、
有效样本数、初始化编号、收敛状态和失败原因。启用实测分段公式对搜索后，
`piecewise_formula_segment_1.csv` 和 `piecewise_formula_segment_2.csv` 分别保存左右段的
独立拟合结果，并保留 `pair_rank`、切点和整对指标以便关联；组合公式对不再单独导出。启用 PySR 后，
同一切点两侧会分别运行搜索，结果分别保存在 `pysr_baseline/segment_1` 和 `segment_2`。

启用 `ranking.enabled` 后还会输出：

- `structure_r2_ranking.csv`：全部单项和双项公式的预测名次、排序分数、节点序列及真实
  频段百分位排名标签（如该公式参与过 LM 拟合）；
- `single_formula_r2_ranking.csv`、`combination_formula_r2_ranking.csv`：分别按树模型分数
  排序的单公式和双公式；`top_ranked_formulas.csv` 保存全局前 100；
- `top_ranked_refit_results.csv` 和 `top_ranked_dataset_fits.csv`：树模型 Top-K 公式在每份
  CSV 上重新独立拟合得到的常数、真实 R²、有效样本数、初始化编号和收敛状态；
- GP 完成后会在 `gp_pareto_dataset_fits.csv` 输出每个最终
  帕累托公式的残差自相关诊断（`lag_1`、`durbin_watson`）。
- `structure_r2_ranker.joblib`：训练后的 ExtraTrees/XGBoost 模型和 One-Hot 编码器；
- `structure_r2_encoder.json`：节点位置类别表；
- `structure_r2_features.json`：固定顺序的最终特征列表、原始/模型特征维度及 CatBoost
  类别特征索引；
- `structure_r2_feature_importance.csv`：按重要性降序保存的模型特征重要性；
- JSON 中的 `ranking_metadata`：训练样本量、验证集 Spearman、Top-K 重合率等排序指标。

这里的 `ranking_score` 只用于相对排序，不作为精确 R² 或百分位数使用。运行排序功能需使用项目
`py39` 环境，其中已经安装 scikit-learn 和 XGBoost：

```bash
/home/lwj/miniconda3/envs/py39/bin/python scripts/run_formula_discovery.py \
  --config configs/formula_discovery/pipeline.json
```

