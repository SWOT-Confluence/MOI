# OSC 与 Unity SOS validation 结果对比

推荐使用“两端提取、本地比较”的流程，不需要下载两个完整的 `sos` 目录。

## 运行顺序

1. 在 OSC 上运行 `01_extract_OSC_validation.ipynb` 的全部 cells。
2. 下载 notebook 生成的整个 `osc_validation_export` 目录。
3. 在 Unity 上运行 `02_extract_Unity_validation.ipynb` 的全部 cells。
4. 下载 notebook 生成的整个 `unity_validation_export` 目录。
5. 将上述两个目录放在 `03_compare_OSC_Unity_validation.ipynb` 旁边，在本地运行该 notebook。

提取 notebook 需要 `numpy`、`pandas` 和 `netCDF4`；本地比较 notebook 还使用
`matplotlib` 绘图。若目录位置不同，只需要修改每个 notebook 顶部的配置 cell。

提取过程不会再把全球所有 reach/algorithm/metric 记录累积在内存中。它先用
`has_validation == 1` 选择 validation rows，再按最多 20,000 行一批读取并直接写入
`.partial` CSV；全部成功后才发布正式 CSV。默认只导出本次比较需要的 `nbias`。
如需额外 validation 指标，可在配置 cell 中将 `METRICS_TO_EXPORT` 改成 `None`，
但建议先用默认设置完成本次对比。

## 数量定义

Notebook 同时保留以下数量，避免“validation 数量”含义不清：

- `validation_flag_count`：`has_validation == 1` 的 reach-algorithm cells 数量；
- `finite_nbias_count`：validation rows 中 `nbias` 为有限值的 cells 数量；
- `result_count`：`has_validation == 1` 且 `nbias` 为有限值的结果数量。

主报告使用 `result_count`。FLPE 算法不写死，读取文件中实际出现的算法并分别统计。

## 本地比较输出

`03_compare_OSC_Unity_validation.ipynb` 会建立 `osc_unity_validation_comparison`：

- `count_comparison.csv`：MOI 总数及各 FLPE/MOI 算法的数量对比；
- `record_coverage.csv`：OSC-only、Unity-only 和两边共有的主键记录；
- `moi_precision_summary.csv`：各 MOI 数值指标的逐 float64 精确相等和容差比较；
- `moi_value_differences.csv`：每条 MOI 指标值的详细差值；
- `moi_accuracy_summary.csv`：共同样本上两次运行的绝对 nBias 汇总；
- `moi_nbias_pairwise.csv`：共同 MOI nBias 样本的逐条精度/准确度对比。

“数值复现精度”和“validation 准确度”分开报告：前者检查两次运行是否得到相同数值，
后者用共同样本的 `|nBias|` 判断哪次运行更接近 validation observation（越低越好）。
