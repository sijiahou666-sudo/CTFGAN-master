# CTFGAN 验证记录

日期：2026-10-29。

执行：`python -m pytest tests/test_ctfgan.py -q`，最终结果 **11 passed**。

已验证：

- 256×256 输出；G 的七个融合尺度与 D 的六次下采样。
- 修改轮廓会改变生成图与判别器输出。
- 四种 TF/CF 消融组合的前向与反向。
- 平坦图 Sobel 响应为零；非平坦图轮廓损失对图像与生成器参数产生有限、非零梯度。
- 平方 L2 与像素 MSE 的缩放关系。
- DAMSM 排序恢复，以及 batch size 为 1 的推理。
- 含 MA-GP 的完整 D/G 训练更新。
- 合成 COCO 目录的离线 Canny、训练、checkpoint 保存与恢复、真实文件推理。
- 固定种子重复推理输出字节一致。
- 评估保留末尾不满 batch 的样本，记录正确的图片配对与种子。
- CLI 配置仅由显式参数覆盖。

额外检查：新入口 `--help`、新增 Python 文件编译、两个新增 shell 脚本语法检查；架构 PNG 已查看。

测试环境：macOS ARM，Python 3.12、PyTorch 2.15.0、torchvision 0.29.2，CPU。测试采用 nf=2 的小模型和随机初始化的小型 DAMSM 编码器，避免冒充真实预训练模型。评估集成测试用轻量替身替代 Inception，验证采样与统计流程，不验证真实 Inception 权重、下载或论文 FID 数值。

存在三个来自原 DAMSM 单层 LSTM 的 dropout 提示，不影响测试通过；保留编码器定义以兼容原权重。

当前仓库没有真实 CUB/COCO 图片、DAMSM 权重及对应参考统计。未执行完整 GPU 训练，未计算真实数据的 IS/FID/LPIPS，未验证论文报告数值。完整复现还需按 README 准备数据、权重并完成正式训练。
