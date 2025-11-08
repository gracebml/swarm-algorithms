# 📦 Installation Guide

## 🎯 Quick Install

```bash
# 1. Navigate to project directory
cd D:\us\ai_fundamental\code\lab1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python test_quick.py

# 4. Run the program
python main.py
```

---

## ✅ System Requirements

### Minimum Requirements
- **Python:** 3.8 or higher
- **RAM:** 2 GB (4 GB recommended)
- **Disk Space:** 500 MB (for code + results)
- **OS:** Windows, macOS, or Linux

### Recommended
- **Python:** 3.9 - 3.11
- **RAM:** 8 GB (for large sensitivity analyses)
- **CPU:** Multi-core (for better performance)

---

## 📚 Dependencies

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Array operations, math functions |
| pandas | ≥1.3.0 | Data analysis, CSV handling |
| matplotlib | ≥3.4.0 | Plotting and visualization |
| seaborn | ≥0.11.0 | Statistical plots |
| scipy | ≥1.7.0 | Scientific computing (Cuckoo Search) |
| PyYAML | ≥6.0 | Configuration file parsing |

### Where Each Package is Used

#### numpy
- All optimization algorithms (PSO, FA, CS, ABC, GA, SA, ACO, A*)
- Mathematical operations
- Array manipulations

#### pandas
- Data visualization scripts
- Performance metrics tables
- CSV data processing

#### matplotlib
- All visualization modules
- Convergence plots
- Performance charts
- 3D landscape plots

#### seaborn
- Statistical visualizations
- Heatmaps
- Enhanced plot styling

#### scipy
- Cuckoo Search optimizer (`scipy.special` for Lévy flights)

#### PyYAML
- Loading algorithm configurations from `configs/algorithms/`
- Loading problem definitions from `configs/problems/`
- Loading sensitivity analysis configs from `configs/sensitivity/`

---

## 🚀 Installation Methods

### Method 1: pip (Recommended)

```bash
pip install -r requirements.txt
```

**Advantages:**
- ✅ Fast installation
- ✅ Automatically resolves dependencies
- ✅ Works on all platforms

---

### Method 2: conda (Alternative)

```bash
# Create new environment
conda create -n ai_lab1 python=3.9

# Activate environment
conda activate ai_lab1

# Install packages
conda install numpy pandas matplotlib seaborn scipy
pip install PyYAML

# Or install all at once
conda install --file requirements.txt -c conda-forge
```

**Advantages:**
- ✅ Better for scientific computing
- ✅ Isolated environment
- ✅ Easy to manage multiple Python versions

---

### Method 3: Virtual Environment (Best Practice)

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Advantages:**
- ✅ Project isolation
- ✅ No conflicts with other projects
- ✅ Easy to recreate environment

---

## 🔍 Verification

### Test 1: Check Python Version
```bash
python --version
# Should show: Python 3.8.x or higher
```

### Test 2: Check Installed Packages
```bash
pip list | findstr "numpy pandas matplotlib seaborn scipy PyYAML"
# Should show all 6 packages
```

### Test 3: Test Imports
```bash
python test_quick.py
```

**Expected output:**
```
✅ ALL IMPORTS SUCCESSFUL!
✅ BASIC FUNCTIONALITY TEST PASSED!
```

### Test 4: Run Main Program
```bash
python main.py
```

**Expected:** Interactive menu appears

---

## ⚠️ Common Issues

### Issue 1: "pip: command not found"

**Solution:**
```bash
# Try python -m pip instead
python -m pip install -r requirements.txt
```

---

### Issue 2: "Permission denied"

**Solution (Windows):**
```bash
# Run as administrator
# OR install for user only:
pip install --user -r requirements.txt
```

**Solution (macOS/Linux):**
```bash
# Use pip3
pip3 install -r requirements.txt

# OR use --user flag
pip install --user -r requirements.txt
```

---

### Issue 3: Package version conflicts

**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then install
pip install -r requirements.txt

# Or force reinstall
pip install --force-reinstall -r requirements.txt
```

---

### Issue 4: Slow installation in China

**Solution:** Use Tsinghua mirror
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Or set permanently:**
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Issue 5: "No module named 'src'"

This is NOT an installation issue - it's a path issue.

**Solution:** Always run from project root
```bash
# Make sure you're in the right directory
cd D:\us\ai_fundamental\code\lab1

# Then run
python main.py
```

---

## 🧪 Testing Installation

After installation, run these tests:

### Quick Test (30 seconds)
```bash
python main.py
> 1  # Continuous
> 8  # Landscape Visualization
> 4  # All functions
```

Should create 3 PNG files in `visualizations/continuous/landscapes/`

### Full Test
See `COMPREHENSIVE_TEST_GUIDE.md` for detailed testing procedures.

---

## 📊 Package Sizes (Approximate)

| Package | Size | Dependencies |
|---------|------|--------------|
| numpy | ~50 MB | None |
| pandas | ~30 MB | numpy, python-dateutil, pytz |
| matplotlib | ~40 MB | numpy, pillow, kiwisolver |
| seaborn | ~2 MB | matplotlib, pandas |
| scipy | ~60 MB | numpy |
| PyYAML | ~1 MB | None |
| **Total** | **~200 MB** | - |

---

## 🔄 Updating Dependencies

To update all packages to latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

To update specific package:

```bash
pip install --upgrade numpy
```

---

## 🗑️ Uninstallation

To remove all dependencies:

```bash
# If using virtual environment
deactivate
rm -rf venv/  # or rmdir /s venv (Windows)

# If installed globally
pip uninstall -r requirements.txt -y
```

---

## 💾 Exporting Current Environment

To save your exact package versions:

```bash
pip freeze > requirements-freeze.txt
```

This creates a file with exact versions for reproducibility.

---

## 🌐 Alternative Installation Sources

### If PyPI is blocked or slow:

#### Tsinghua Mirror (China)
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Alibaba Mirror
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

#### USTC Mirror
```bash
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

---

## 📝 Development Setup

If you plan to modify the code:

```bash
# Install dev dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/

# Check code style
flake8 src/
```

---

## ✅ Installation Checklist

After installation, verify:

- [ ] Python version ≥ 3.8
- [ ] All 6 packages installed
- [ ] `test_quick.py` passes
- [ ] `main.py` runs without errors
- [ ] Can create visualizations
- [ ] No import errors

---

## 🆘 Still Having Issues?

1. **Check Python version:**
   ```bash
   python --version
   ```

2. **Check pip version:**
   ```bash
   pip --version
   ```

3. **Try clean install:**
   ```bash
   pip uninstall -y numpy pandas matplotlib seaborn scipy PyYAML
   pip install -r requirements.txt
   ```

4. **Check for conflicts:**
   ```bash
   pip check
   ```

5. **Create fresh virtual environment:**
   ```bash
   python -m venv fresh_env
   fresh_env\Scripts\activate
   pip install -r requirements.txt
   ```

---

**Last Updated:** 2025-11-08  
**Tested on:** Windows 10, Python 3.9-3.11  
**Status:** ✅ Working

