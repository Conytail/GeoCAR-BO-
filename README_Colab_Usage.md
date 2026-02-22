# BioFoundry Active Learning - Colab Notebook 使用指南

## 📌 快速开始

### 1. 准备工作（在本地 Windows 上完成）

#### 1.1 上传数据到 Google Drive
```bash
# 在本地创建文件夹结构
Google Drive/
└── BioFoundry/
    └── data/
        ├── train.lmdb
        ├── val.lmdb
        └── lock.mdb
```

**重要**：确保你的 LMDB 文件包含以下字段：
- `pos`: 原子坐标 (N_atoms, 3)
- `atomic_numbers`: 原子序数 (N_atoms,)
- `y`: 目标分数（Log10 转换后）
- `sid`: 样本 ID（可选，用于追踪）

#### 1.2 上传配置文件（可选）
如果你有自定义的 `main_oc20.py` 或 `gpu_config.yml`，也上传到 Google Drive。

---

### 2. 在 Colab 中运行

#### 2.1 打开 Colab
1. 访问：https://colab.research.google.com/
2. 选择 GPU 运行时：
   - `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`
   - 推荐选择 **A100** 或 **V100**（T4 也可以，但需要降低 batch_size）

#### 2.2 上传 Notebook
1. 点击 `File` → `Upload notebook`
2. 选择 `BioFoundry_ActiveLearning_Colab.py`（或者复制粘贴代码到新的 Notebook）

#### 2.3 逐个运行 Cell
**按照以下顺序运行**：

| Cell | 步骤 | 预计耗时 | 关键点 |
|------|------|----------|--------|
| 1 | GPU 检查 | 10秒 | 确认 GPU 类型，自动推荐 batch_size |
| 2 | 安装依赖 | 3-5分钟 | **必须按顺序**！先卸载再安装 |
| 3 | 挂载 Drive + 数据复制 | 2-5分钟 | ⚠️ **修改路径**：`DRIVE_DATA_PATH` |
| 4 | 克隆代码库 | 1分钟 | 自动下载 OCP + EquiformerV2 |
| 5 | 生成配置 | 5秒 | GPU 自适应生成 `colab_config.yml` |
| 6 | **训练模型** | **2-6小时** | T4: ~6h, V100: ~3h, A100: ~2h |
| 7 | TensorBoard | - | 在新标签页运行，实时监控训练 |
| 8 | **提取嵌入** | 5-10分钟 | 最关键！使用 Hook 提取特征 |
| 9 | 主动学习初始化 | 10秒 | 创建 Batch Diversity Optimizer |
| 10 | 选择第一批候选 | 5秒 | 输出推荐的 10 个实验 |
| 11 | 更新模型（人工验证后） | 5秒 | 输入新的实验结果 |
| 12 | 可视化 | 30秒 | 生成嵌入空间图和采集函数图 |
| 13 | 保存结果 | 1分钟 | 备份到 Google Drive |

---

## ⚠️ 关键修改点（必须执行）

### 修改 1: 更新数据路径（Cell 3）
```python
# 找到这一行：
DRIVE_DATA_PATH = "/content/drive/My Drive/BioFoundry/data"  # ⚠️ Modify this path

# 改成你实际的 Google Drive 路径，例如：
DRIVE_DATA_PATH = "/content/drive/My Drive/我的项目/BioFoundry/data"
```

### 修改 2: 检查 Hook 层名称（Cell 8）
```python
# 在 Cell 8 中，找到这一行：
hook_layer_name = 'energy_block'  # ⚠️ Verify this matches your model

# 如果运行时报错 "Layer 'energy_block' not found"，执行以下步骤：
# 1. 查看 Cell 8 打印的 "Model structure (first 20 layers)"
# 2. 找到最后一个 LayerNorm 或 Linear 层的名字
# 3. 替换 hook_layer_name，例如：
#    - 如果看到 'norm_final' -> hook_layer_name = 'norm_final'
#    - 如果看到 'blocks.3.norm' -> hook_layer_name = 'blocks.3.norm'
```

### 修改 3: 调整初始数据集（Cell 10）
```python
# 找到这一段：
initial_sample_ids = np.random.choice(all_sample_ids, size=20, replace=False).tolist()
initial_scores = {sid: np.random.randn() for sid in initial_sample_ids}

# 替换为你的真实初始实验数据：
initial_sample_ids = ['CAR_001', 'CAR_023', 'CAR_045', ...]  # 你已测试的样本
initial_scores = {
    'CAR_001': 0.85,  # 实际测得的分数（已 Log10 转换）
    'CAR_023': 1.23,
    'CAR_045': -0.54,
    ...
}
```

---

## 🔧 常见问题排查

### 问题 1: CUDA Out of Memory
**症状**：训练时崩溃，提示 `RuntimeError: CUDA out of memory`

**解决方案**：
1. 降低 `batch_size`（Cell 5）：
   ```python
   # 如果是 T4，改为：
   RECOMMENDED_BATCH_SIZE = 2  # 从 4 降到 2
   ```

2. 降低模型复杂度（Cell 5）：
   ```python
   # 修改这些参数：
   "lmax_list": [2],  # 从 [4] 降到 [2]
   "mmax_list": [1],  # 从 [2] 降到 [1]
   "num_layers": 2,   # 从 4 降到 2
   "sphere_channels": 32,  # 从 64 降到 32
   ```

---

### 问题 2: Hook 无法提取嵌入
**症状**：Cell 8 运行后，`features_cache['embedding']` 为空或形状不对

**解决方案**：
1. 打印模型结构：
   ```python
   for name, module in model.named_modules():
       print(f"{name}: {type(module)}")
   ```

2. 手动找到正确的层：
   - 通常是最后一个 `LayerNorm` 或 `Linear` 层
   - 在 `energy_block` 或 `head` 之前

3. 使用 `register_forward_pre_hook` 而不是 `register_forward_hook`：
   ```python
   # 如果输出为空，改用 pre_hook：
   hook_handle = model.energy_block.register_forward_pre_hook(
       lambda m, inp: features_cache.update({'embedding': inp[0].detach()})
   )
   ```

---

### 问题 3: LMDB 读取超慢
**症状**：数据加载速度 < 10 samples/sec

**解决方案**：
- **检查是否复制到本地**：
  ```python
  # 确保 Cell 3 中执行了这一步：
  shutil.copytree(DRIVE_DATA_PATH, LOCAL_DATA_PATH, dirs_exist_ok=True)
  
  # 验证本地存在数据：
  !ls -lh /content/data/
  ```

- **不要直接读 Drive 上的 LMDB**：
  ```python
  # ❌ 错误示例：
  src: "/content/drive/My Drive/data/train.lmdb"
  
  # ✅ 正确示例：
  src: "/content/data/train.lmdb"
  ```

---

### 问题 4: 依赖冲突
**症状**：`ImportError: cannot import name 'sph_harm' from 'scipy.special'`

**解决方案**：
```bash
# 在 Cell 2 后面单独运行：
!pip install scipy==1.13.1 --force-reinstall
```

---

## 📊 预期输出

### Cell 8 成功输出示例：
```
✅ Extracted 1849 embeddings
✅ Embeddings saved to /content/embeddings.npy

Sample embedding shape: (256,)
Sample embedding (first 5 dims): [ 0.234 -1.023  0.567 -0.891  0.123]
```

### Cell 10 成功输出示例：
```
🎯 Recommended candidates for next experiments:
===========================================================
 1. Sample: CAR_T_1234
 2. Sample: CAR_T_0567
 3. Sample: CAR_T_2389
 4. Sample: CAR_T_1890
 5. Sample: CAR_T_0123
 6. Sample: CAR_T_3456
 7. Sample: CAR_T_2901
 8. Sample: CAR_T_1678
 9. Sample: CAR_T_0934
10. Sample: CAR_T_2567
```

---

## 🔄 完整的 DBTL 循环流程

```
第一轮（Round 1）:
1. 运行 Cell 1-10 → 得到第一批候选 (10个)
2. 手动验证这 10 个设计（湿实验或计算验证）
3. 记录实验结果 → new_experimental_results = {...}
4. 运行 Cell 11 → 更新模型 → 得到第二批候选

第二轮（Round 2）:
5. 手动验证第二批
6. 再次运行 Cell 11
7. 重复...

收敛标准:
- Pareto 前沿不再显著改善
- 所有候选的 UCB 上界 < 阈值
- 达到预算上限（例如 100 次实验）
```

---

## 💾 结果备份

运行 Cell 13 后，所有结果会自动保存到：
```
Google Drive/My Drive/BioFoundry/results_20260129_114500/
├── embeddings.npy           # 所有样本的嵌入向量
├── optimizer_state.pkl       # 优化器状态（可恢复）
├── selected_batches.txt      # 所有选中的批次
└── best_model.pt            # 训练好的模型权重
```

---

## 📚 参考资料

- **EquiformerV2 论文**: https://arxiv.org/abs/2306.12059
- **OCP (Open Catalyst Project)**: https://github.com/Open-Catalyst-Project/ocp
- **Bayesian Optimization**: https://github.com/fmfn/BayesianOptimization
- **MOBO-OSD 原始论文**: Ginsbourger et al. (2010) - "Kriging is Well-Suited to Parallelize Optimization"

---

## ✅ Checklist

运行前请确认：
- [ ] Google Drive 中已上传 `train.lmdb` 和 `val.lmdb`
- [ ] Colab 已选择 GPU 运行时（A100/V100/T4）
- [ ] Cell 3 中的 `DRIVE_DATA_PATH` 已修改为实际路径
- [ ] 至少有 20 个初始实验数据（Cell 10）
- [ ] 预留 6-8 小时（如果是 T4 GPU）

运行后请检查：
- [ ] Cell 1 显示正确的 GPU 型号
- [ ] Cell 3 成功复制数据到 `/content/data/`
- [ ] Cell 6 训练完成，无 OOM 错误
- [ ] Cell 8 成功提取 >1000 个嵌入向量
- [ ] Cell 10 输出 10 个推荐候选
- [ ] Cell 13 结果已备份到 Google Drive

---

## 🎯 成功标准

如果你看到以下输出，说明一切正常：

```
🎉 BioFoundry Active Learning Pipeline Complete!
===========================================================

Results saved to: /content/drive/My Drive/BioFoundry/results_...
Embeddings: 1849 samples
Current labeled pool: 20 samples
Remaining unlabeled: 1829 samples

📈 Good luck with your experiments!
```

---

**如有问题，请检查上述"常见问题排查"部分。祝你实验顺利！🚀**
