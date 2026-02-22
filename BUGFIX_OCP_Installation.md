# 🎯 最终修复方案

## 问题根源

你遇到的错误：
```
ERROR: file:///content/ocp does not appear to be a Python project
```

根本原因是：**OCP 仓库已迁移**！

---

## ✅ 正确的做法

### 1. OCP 仓库已迁移

**旧的（错误）**：
```bash
git clone https://github.com/Open-Catalyst-Project/ocp.git
```

**新的（正确）**：
```bash
git clone https://github.com/FAIR-Chem/fairchem.git ocp
cd ocp
git checkout f83d150  # 特定版本
```

### 2. 必须修改代码

在安装前，**必须**修改 `ocpmodels/common/utils.py`：

在第 329 行 `finally:` 后面添加：
```python
finally:
    import nets           # ← 添加这行
    import oc20.trainer   # ← 添加这行
    registry.register("imports_setup", True)
```

### 3. 然后才能安装

```bash
pip install -e .
```

---

## 📝 完整安装流程（Cell 4）

```python
# 1. 克隆正确的仓库
!git clone https://github.com/FAIR-Chem/fairchem.git ocp
!cd ocp && git checkout f83d150

# 2. 修改代码（Python 自动化）
utils_path = "/content/ocp/ocpmodels/common/utils.py"
with open(utils_path, 'r') as f:
    content = f.read()

# 替换代码
modified = content.replace(
    "finally:\\n        registry.register",
    "finally:\\n        import nets\\n        import oc20.trainer\\n        registry.register"
)
with open(utils_path, 'w') as f:
    f.write(modified)

# 3. 安装
!cd ocp && pip install -e .

# 4. 验证
from ocpmodels.common import distutils  # 应该成功
```

---

## 🚀 使用新 Notebook

**文件名**：`BioFoundry_ActiveLearning_Final.ipynb`

这是最终正确版本，包含：
- ✅ Cell 1: GPU 检查
- ✅ Cell 2: 依赖安装（含 submitit）
- ✅ Cell 3: Drive 数据复制
- ✅ **Cell 4: 正确的 OCP 安装**（FAIR-Chem + 代码修改）
- ✅ Cell 5: 配置生成
- ✅ Cell 6: 训练启动

---

## 预期输出

运行 Cell 4 后应该看到：

```
============================================================
Installing OCP (FAIR-Chem)...
============================================================

📥 Cloning FAIR-Chem repository...
✅ Cloned

📌 Checking out version f83d150...
✅ Checked out

🔧 Modifying ocpmodels/common/utils.py...
✅ utils.py modified

📦 Installing ocpmodels package...
[pip install logs...]
✅ OCP installed

📥 Cloning EquiformerV2...
✅ EquiformerV2 cloned

============================================================
Verifying Installation...
============================================================
✅ ocpmodels imports successful

✅ Setup complete
```

---

## 参考文档

EquiformerV2 官方安装指南：
https://github.com/atomicarchitects/equiformer_v2/blob/main/docs/env_setup.md

关键信息：
- OCP 仓库：`FAIR-Chem/fairchem`（不是旧的 Open-Catalyst-Project/ocp）
- 特定版本：`f83d150`
- 代码修改：在 `utils.py` 第 329 行后添加两个 import

---

**现在应该能正常工作了！**
