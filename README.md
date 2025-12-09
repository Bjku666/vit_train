# ViT 训练 / 评测 / 推理 / 伪标签 使用指南

## 环境说明
- 主配置位于 `config.py`（路径、超参、训练目录开关 `CONFIG_INIT_DIRS`）。
- 数据目录：`data/2-MedImage-TrainSet`（有标签训练集），`data/MedImage-TestSet`（无标签测试集），`data/pseudo_labeled_set`（伪标签输出）。
- 设备：脚本自动检测多卡并使用 `DataParallel`（训练/推理/伪标签均支持）。

## 1) 训练（5 折，带 EMA/Albumentations/Warmup）
后台训练示例：
```bash
nohup python -u train_vit.py > logs/train.log 2>&1 &
```
TensorBoard 可视化：
```bash
tensorboard --logdir logs --port 6006
```
说明：
- 训练数据由 `config.TRAIN_DIRS` 聚合（默认包含原始集 + 伪标签集）。
- 学习率策略：线性预热（WARMUP_EPOCHS）后 Cosine；EMA 保存最佳权重到 `models/run_*/vit_foldX.pth`。

## 2) 评测（Benchmark，多模型 8x TTA + 阈值搜索）
```bash
CONFIG_INIT_DIRS=0 python benchmark_vit.py --model_paths "models/run_xxx/vit_fold*.pth"
```
输出：`output/benchmark_result_YYYYMMDD_HHMMSS.json`，包含最佳阈值、ACC、F1；不创建训练日志目录。

## 3) 伪标签生成（Pseudo-Labeling，8x TTA 融合）
```bash
CONFIG_INIT_DIRS=0 python generate_pseudo_labels.py \
    --model_paths "models/run_xxx/vit_fold*.pth" \
    --threshold 0.9 \
    --batch_size 64
```
输出：高置信样本复制到 `data/pseudo_labeled_set/disease|normal`，并打印统计信息。阈值可调以控制伪标签数量与质量。

## 4) 推理提交（Inference，多模型 8x TTA）
```bash
CONFIG_INIT_DIRS=0 python inference.py \
    --model_paths "models/run_xxx/vit_fold*.pth" \
    --benchmark_json output/benchmark_result_YYYYMMDD_HHMMSS.json
```
输出：`output/submission_YYYYMMDD_HHMMSS.csv`，使用 benchmark JSON 的最佳阈值对无标签测试集预测。

## 5) 快速检查
- 训练日志：`logs/<run_id>/fold_k/` (TensorBoard)；文本日志 `logs/train_vit_<run_id>.log`。
- 模型权重：`models/run_<run_id>/vit_fold*.pth`。
- 评测结果：`output/benchmark_result_*.json`。
- 提交文件：`output/submission_*.csv`。
- 伪标签：`data/pseudo_labeled_set/`。
