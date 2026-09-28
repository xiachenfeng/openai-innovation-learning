# In-context Learning（E3）

比较 zero-shot、one-shot、few-shot，不进行梯度更新。

- `icl_probe.py`：GPT-2 四个尺寸 × K ∈ {0,1,4,8} × 5 组示例，SST-2 打分式读法；可选两位数加法（生成式）
- `run_probe.sh`：冒烟 + 正式跑
- `E3-icl-probe.md`：实验记录
- `outputs/e3_icl.csv`：结果（由云主机推送）
