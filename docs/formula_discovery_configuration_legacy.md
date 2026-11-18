# `formula_discovery_pipeline` 配置参数说明

本文逐项对应当前的 [`pipeline.json`](../configs/formula_discovery/pipeline.json)。路径采用点号表示 JSON 层级，例如 `data.split.seed`。表中的“当前值”以该配置文件为准；运行时选择的 `pipeline_mode`、命令行参数及程序预设可能覆盖配置值。`formula_generation.json` 是公式库预计算脚本的独立配置，**不是**主流程的拟合配置。

## 运行入口与全局设置

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `_comment` | 中文说明 | 给人看的配置备注，程序不读取。 |
| `stage` | `all` | 执行阶段：`generate` 仅生成候选，`fit` 使用已有候选拟合，`all` 全部执行。 |
| `minimal_output_artifacts` | `true` | 精简保存的中间文件和诊断产物。 |
| `require_integer_exponents` | `true` | 限制最终公式中的幂指数为整数。 |
| `seed` | `1518` | 主流程的随机种子。 |
| `log_level` | `INFO` | 日志详细程度。 |
| `mode5_fit_batch_rounds` | `5` | 记录到 Mode 5 搜索元数据的批轮数；当前主流程未用它控制实际拟合循环。 |
| `mode5_constant_test_timeout_seconds` | `210` | Mode 5 常数显著性检验的单次超时秒数。 |
| `periodic_constant_significance_enabled` | `true` | 周期常数显著性检验的开关；还需运行模式及间隔参数满足条件。 |

**运行时覆盖：** `load_config(..., pipeline_mode=3)` 会对 Mode 3/4/5/6 设置 GP 分层、Top-K 和 40 轮搜索上限。搜索会因候选耗尽提前结束。`periodic_constant_significance_enabled` 单独为 `true` 不会启动周期检验；代码还要求运行配置中有 `periodic_constant_significance_experiment_mode` 为 `5` 或 `6`，且 `periodic_constant_significance_interval > 0`。这两个键当前不在 JSON 中，通常由实验入口注入。

`minimal_output_artifacts=false` 时，`save_outputs()` 会走完整 CSV 输出分支，额外保存基元、候选、初始拟合、排序、Top-K 重拟合及检查结果；是否保存总结果 `formula_discovery_results.json` 仍由 `suppress_results_json` 控制：Mode 3/4 运行入口默认将其设为 `true`，因此仅改 `minimal_output_artifacts` 不会得到该 JSON。若另有 `suppress_large_artifacts=true`，完整输出分支会直接跳过 `save_outputs()`；多种子实验和消融子运行也会单独覆盖这些开关。

## 可选 PySR 基线

PySR 参数已从主流程配置中拆出。通用设置位于
[`pysr_baseline.json`](../configs/formula_discovery/pysr_baseline.json)，历史实验的设置位于
[`configs/experiments/pysr/`](../configs/experiments/pysr/)。主配置不带 PySR 配置时不会运行 PySR。
主实验入口不加载这些 sidecar；`scripts/pysr/` 下的独立入口才会读取对应场景配置并运行 PySR。
Sidecar 中的 BFGS 常数重拟合参数只供 PySR 基线使用，不注入主搜索配置。

| 参数（省略 `pysr_baseline.` 前缀） | 当前值 | 作用 |
| --- | --- | --- |
| `enabled` | `false` | 是否运行 PySR 基线。 |
| `reproducibility_mode` | `reproducible` | PySR 运行配置：可复现串行模式或多线程模式；加载时会派生并覆盖并行、确定性和超时设置。 |
| `niterations` | `500` | 符号搜索迭代预算。 |
| `final_validation_r2_window` | `0.1` | 最终候选相对最佳验证集 R² 的允许差距。 |
| `populations` | `20` | 搜索种群数。 |
| `population_size` | `20` | 每个种群的个体数。 |
| `maxsize` | `20` | PySR 表达式最大复杂度。 |
| `binary_operators` | `+,-,*,/` | 搜索允许的二元算子。 |
| `unary_operators` | `sin,cos,log_abs,log_abs_sin,log_abs_cos` | 搜索允许的一元算子及自定义定义。 |
| `model_selection` | `best` | 从 PySR 候选中选模型的策略。 |
| `verbosity` | `1` | PySR 输出详细程度。 |
| `precision` | `64` | PySR 数值精度位数。 |
| `optimizer_algorithm` | `BFGS` | PySR 内部常数优化算法。 |
| `optimizer_nrestarts` | `20` | 常数优化的随机重启次数；主拟合选 BFGS 时也复用。 |
| `optimizer_iterations` | `8` | 每次常数优化的迭代上限；主拟合选 BFGS 时也复用。 |
| `optimizer_f_calls_limit` | `10000` | 常数优化的函数调用上限；主拟合选 BFGS 时也复用。 |
| `process_watchdog_enabled` | `true` | 启用 PySR 子进程无进度监控。 |
| `process_watchdog_grace_seconds` | `60` | 看门狗判定超时前的宽限秒数。 |
| `pysr_iteration_timeout_seconds` | `100` | 可复现模式中单个 Julia 搜索步的超时秒数。 |
| `process_watchdog_iteration_timeout_seconds` | `160` | Julia 搜索步无进度时的进程级兜底超时秒数。 |
| `process_watchdog_startup_timeout_seconds` | `1800` | Julia 启动至首次进度的超时秒数。 |
| `optimize_probability` | `0.14` | 搜索中触发常数优化的概率。 |
| `should_optimize_constants` | `true` | 是否优化表达式中的常数。 |
| `constant_refit_gtol` | `1e-5` | 常数重拟合的梯度收敛容差；主拟合选 BFGS 时也复用。 |
| `constant_refit_initialization_range` | `[-10,10]` | 常数重拟合初值范围；主拟合选 BFGS 时也复用。 |
| `require_all_features` | `true` | 候选表达式必须使用所有输入特征。 |
| `forbid_nested_nonlinear_operators` | `true` | 禁止非线性一元算子嵌套。 |
| `forbid_division_by_nonlinear_structure` | `true` | 禁止分母子树含有非线性算子。 |
| `max_nonconstant_additive_terms` | `2` | 非常数加法项的最大数量。 |
| `structural_loss_during_search` | `true` | 在搜索损失中施加结构约束。 |
| `structural_loss_penalty` | `1e300` | 结构违规时加入的惩罚值。 |
| `power_exponent_constraint_note` | 中文说明 | 对幂指数限制的文字备注；程序不读取。 |
| `forbid_bfgs_constants_in_power_exponents` | `true` | 幂的指数子树中禁止含可由 BFGS 调整的常数。 |
| `require_log_ancestor_for_x1` | `true` | 要求特征 `x1` 处于对数算子的后代结构中。 |
| `require_optimizer_success` | `false` | 是否只接受常数优化成功的候选。 |
| `shared_template.enabled` | `true` | 多数据集时启用 PySR 模板表达式，共享结构并保留各数据集参数。 |
| `shared_template.parameter_count` | `4` | 模板中每个数据集的可优化参数槽数。 |

## 搜索后处理 `postprocessing` 与迭代搜索配置 `mode2`

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `postprocessing.merge_log_max_r2_drop` | `0.01` | 常数显著性筛选及 Log 合并检查允许的最大绝对 R² 下降。 |
| `postprocessing.log_merge_initial_perturbation` | `0.05` | Log 合并重拟合时初值的扰动幅度。 |

**说明：** 当前多种子实验执行树模型代理辅助迭代搜索，不运行独立的轻量 GP 种群演化器。Top-K、批大小和迭代上限由运行时的 `mode2` 设置控制；上述两项仅用于常数及 Log 项处理。

## 数据 `data`

Maritime 使用历史模式标识 `data.mode=realdata_mat`，但数据由 `data.csv_path` 指向的预处理 CSV 提供，不读取 MAT 文件。`data.profile=measured_pathloss` 标记 pooled R²、物理断点和分段拟合语义。其他 CSV 数据继续使用 `data.mode=csv`。

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `data.mode` | `csv` | CSV 文件读取方式。 |
| `data.profile` | `measured_pathloss` | Maritime 实测数据的搜索/指标语义配置。 |
| `data.csv_path` | `data/processed/Maritime/plot_data2.csv` | 由 `data/raw/Maritime/plot_data2.mat` 确定性转换的单 CSV。 |
| `data.features` | `distance` | CSV 输入特征列名及顺序。 |
| `data.target` | `path_loss` | CSV 目标列名。 |
| `data.split.seed` | `1558` | 数据划分随机种子。 |
| `data.split.train_fraction` | `0.5` | 训练集比例，用于拟合与搜索。 |
| `data.split.validation_fraction` | `0.25` | 验证集比例，用于选解与模型评估。 |
| `data.split.test_fraction` | `0.25` | 测试集比例，用于最终报告。 |

### Maritime 实测数据 `data.realdata`

| 参数（省略 `data.realdata.` 前缀） | 当前值 | 作用 |
| --- | --- | --- |
| `name` | `measured_pathloss` | 实测数据集名称。 |
| `segmentation.enabled` | `true` | 是否启用一元数据分段拟合。 |
| `segmentation.min_segment_samples` | `10` | 切点两侧各自需要的最少样本数。 |
| `segmentation.warm_start_perturbation` | `0` | 分段拟合沿用已有参数初值时的扰动幅度。 |
| `segmentation.independent_segment_fitting` | `true` | 切点两侧是否各自独立拟合公式参数。 |
| `segmentation.breakpoint_prior.enabled` | `true` | 启用物理断点先验；当前实现会固定到最接近先验距离的合法切点。 |
| `segmentation.breakpoint_prior.h_t_m` | `25` | 断点计算中的发射端高度，单位米。 |
| `segmentation.breakpoint_prior.h_r_m` | `4` | 断点计算中的接收端高度，单位米。 |
| `segmentation.breakpoint_prior.carrier_frequency_hz` | `5.8e9` | 断点计算中的载波频率，单位赫兹。 |
| `segmentation.breakpoint_prior.speed_of_light_m_per_s` | `299792458` | 断点计算采用的光速，单位米/秒。 |
| `segmentation.breakpoint_prior.meters_to_data_units` | `0.001` | 将米换算为数据距离单位的乘数，当前对应千米。 |

## 基元与公式库

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `bases.arities` | `[1, 2]` | 显式允许生成的基元元数；一元、二元脚本分别使用 `1`、`2`，不得使用 `auto`。 |
| `bases.max_abs_exponent` | `1` | 基元中各变量指数绝对值上限。 |
| `candidates.max_count` | `100000` | 单项候选公式最多保留的数量。 |
| `candidates.two_base_ratio` | `0.5` | 候选中双基元结构的目标占比。 |
| `candidates.wrappers` | `identity, log_abs, sin, cos, log_abs_sin, log_abs_cos` | 作用于基元的算子类型。 |
| `candidates.max_sampling_attempts` | `100000` | 生成单项候选时的最大抽样尝试次数。 |
| `combinations.max_count` | `100000` | 多项组合公式最多保留的数量。 |
| `combinations.max_terms` | `2` | 一个组合最多含多少个候选项。 |
| `combinations.types` | `add` | 组合运算类型；当前只生成加法组合。 |
| `combinations.max_sampling_attempts` | `200000` | 生成组合时的最大抽样尝试次数。 |

## 公式拟合 `fitting`

| 参数（省略 `fitting.` 前缀） | 当前值 | 作用 |
| --- | --- | --- |
| `single_sample_count` | `40` | 初始直接拟合抽样的单项公式数量。 |
| `combination_sample_count` | `160` | 初始直接拟合抽样的组合公式数量。 |
| `fit_mode` | `refit` | `refit` 重新拟合；`reuse_history` 复用兼容的历史拟合结果。 |
| `optimizer` | `LM` | 常数拟合算法，可选 `LM`、`BFGS`；BFGS 参数取自 `pysr_baseline`。 |
| `lm_initializations` | `20` | 每个公式 LM 拟合的随机初值次数。 |
| `max_nfev` | `5000` | 一次 LM 优化最多函数求值次数。 |
| `log_merge_lm_initializations` | `20` | GP 后 Log 合并检查的 LM 初值次数。 |
| `zero_constant_lm_initializations` | `20` | 常数置零候选重拟合的 LM 初值次数。 |
| `zero_constant_initial_perturbation` | `0.05` | 常数置零后沿用原拟合参数时的初值扰动。 |
| `patience` | `20` | 拟合或筛选过程中无改进的容忍次数。 |
| `initialization_range` | `[-10,10]` | LM 随机初值的取值区间。 |
| `minimum_valid_fraction` | `0.9` | 有效预测样本占比下限。 |
| `complex_tolerance` | `1e-12` | 判断复数结果虚部可忽略的容差。 |
| `max_abs_value` | `1e150` | 预测或中间数值绝对值上限，超过视为异常。 |
| `share_constants_across_datasets` | `false` | 旧版跨数据集共享常数选项；当前主流程未读取此键。 |
| `invalid_residual_penalty` | `1000000` | 无效预测在拟合残差中使用的惩罚值。 |

结果中会分别保存 `joint_r2_minimum`、`joint_r2_mean` 和 `joint_r2_pooled`；旧的 `r2` / `joint_r2` 字段继续按既有流程规则提供主指标。

## 结构排序 `ranking`

主实验固定使用 `operator_primitive_matrix`；特征编码参数见 `configs/formula_discovery/formula_generation.json`。

| 参数（省略 `ranking.` 前缀） | 当前值 | 作用 |
| --- | --- | --- |
| `enabled` | `true` | 训练结构到质量的排序模型，并为未直接拟合公式排序。 |
| `ablation_enabled` | `false` | 启用排序模块内部的消融分支；与顶层 Mode 3 自动消融开关不同。 |
| `model_type` | `extra_trees` | 排序回归模型；支持 `extra_trees`、`xgboost` 等实现分支。 |
| `minimum_training_samples` | `20` | 训练排序模型要求的最少已拟合样本数。 |
| `validation_fraction` | `0.2` | 排序模型训练数据内部的验证比例。 |
| `validation_top_k` | `10` | 计算验证集 Top-K 排序指标时的 K。 |
| `feature_correlation.enabled` | `false` | 候选公式特征相关性诊断开关；主流程加载配置时也会强制关闭。 |
| `feature_correlation.sample_size` | `300` | 相关性诊断抽取的候选公式数；诊断关闭时不生效。 |
| `refit_top_k` | `15` | 排名后选出进行精确重拟合的公式数。 |
| `refit_lm_initializations` | `20` | Top-K 精确重拟合的 LM 初值次数。 |
| `n_estimators` | `500` | ExtraTrees 等集成模型中的树数量。 |
| `max_depth` | `null` | 排序树最大深度；`null` 表示不额外限制。 |
| `min_samples_leaf` | `2` | 排序树叶节点最少样本数。 |
| `max_features` | `1.0` | 排序树分裂时考虑的特征比例或数量；部分运行模式使用专门策略。 |
| `n_jobs` | `24` | 排序模型并行工作数。 |
| `learning_rate` | `0.05` | XGBoost 模型学习率；ExtraTrees 下无效。 |
| `subsample` | `0.9` | XGBoost 行采样比例；ExtraTrees 下无效。 |
| `colsample_bytree` | `0.9` | XGBoost 每棵树的列采样比例；ExtraTrees 下无效。 |
| `reg_lambda` | `1.0` | XGBoost L2 正则系数；ExtraTrees 下无效。 |

## 输出与预计算产物

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `output.directory` | `artifacts/expri2` | 主流程结果目录。 |
| `precomputed_formula_feature_artifact` | `precomputed_formula_features.joblib` | CSV 模式使用的预计算公式与特征文件。 |
| `precomputed_formula_feature_artifact_realdata_mat` | `precomputed_formula_features_realdata_mat.joblib` | Maritime `measured_pathloss` profile 使用的冻结公式/特征文件；旧字段名为兼容而保留。 |

二元与一元预计算文件分别由 `precompute_formula_features_bivariate.py` 和 `precompute_formula_features_univariate.py` 生成，并由独立的 `formula_generation.json` 管理。若更改基元、候选或组合生成参数，应重新生成匹配的预计算文件；主流程会检查其配置指纹。


