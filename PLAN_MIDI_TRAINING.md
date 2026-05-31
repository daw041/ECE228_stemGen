# MIDI 分支递增数据规模训练计划

## Context

Audio Token 分支已暂停，全部精力转向 MIDI 符号分支。archive.zip 中有 **1956/2100 tracks 包含 bass MIDI**（每个 MIDI 文件 ~5KB），需要设计一套递增数据规模的实验流程，在训练时间和磁盘空间之间取得平衡。

## 磁盘空间估算

| 阶段 | Tracks | 音频存储 | MIDI 存储 | 特征缓存 |
|------|--------|----------|-----------|----------|
| P1   | 50     | ~700MB   | ~250KB    | ~30MB    |
| P2   | 200    | ~2.8GB   | ~1MB      | ~120MB   |
| P3   | 550    | ~7.7GB   | ~2.5MB    | ~350MB   |
| P4   | 1000   | ~14GB    | ~5MB      | ~630MB   |
| P5   | 1956   | ~27GB    | ~9MB      | ~1.2GB   |

## 实验矩阵

### Phase 1: 50 tracks — 管线验证
- **目标**: 确认 MIDI pipeline 完整跑通，快速迭代
- **训练**: 过拟合 8 clips + 50 epochs full
- **配置**: 4s clips, batch 16, lr=3e-4
- **预计时间**: ~30 分钟
- **里程碑**: 模型能生成有节奏结构的 bass MIDI（非全静音）

### Phase 2: 200 tracks — 基线结果
- **目标**: 获得第一个有意义的指标
- **训练**: 过拟合 8 clips + 100 epochs full
- **预计时间**: ~1.5 小时
- **里程碑**: activity accuracy > 70%, pitch accuracy (active frames) > 50%

### Phase 3: 550 tracks — 与 Audio E2 对齐
- **目标**: 与音频分支 E2 (val acc 51.8%) 同数据量对比
- **训练**: 过拟合 8 clips + 200 epochs full
- **预计时间**: ~3-4 小时
- **里程碑**: activity F1 > 0.80, piano-roll 对比图有清晰 bass 线条

### Phase 4: 1000 tracks — 扩展验证（可选）
- **目标**: 验证更多数据是否持续提升
- **训练**: 200 epochs
- **预计时间**: ~5-6 小时

### Phase 5: 1956 tracks — 全量（可选）
- 最终大规模实验

## 关键代码改进

### 1. 数据提取脚本（新文件）
**文件**: `scripts/extract_midi_dataset.py`
- 从 archive.zip 提取指定数量 tracks
- 过滤条件: `inst_class == bass` AND `midi_saved == true` AND MIDI file exists
- 按 split 优先顺序: train → validation → test → omitted
- 输出结构: `dataset/midi_subset/TrackXXXXX/{metadata.yaml, mix.flac, {Sxx}.flac, MIDI/{Sxx}.mid}`
- 支持增量提取（已有 track 跳过）

### 2. 特征缓存（修改 train_midi.py）
- 在训练前预计算所有 clips 的 mel+chroma features + MIDI labels
- 缓存为 `outputs/midi/features_cache.pt`
- 每个 clip ~105KB，避免每 epoch 重算特征
- 如果缓存存在则直接加载

### 3. 训练脚本增强
**文件**: `scripts/train_midi.py`
- 独立可配置参数（CLIP_SEC, MAX_TRACKS, EPOCHS）
- 按 track 做 train/val split（避免同 track 不同 clip 泄漏到 val）
- 增加指标: activity precision/recall/F1, pitch accuracy, active ratio
- 自动保存 piano-roll 对比图 + loss 曲线

### 4. 评估指标
- `activity_accuracy`: 帧级活动预测准确率
- `activity_f1`: 活动帧 F1（处理正负样本不平衡）
- `pitch_accuracy`: 仅活动帧上的 pitch 准确率
- `active_ratio`: 预测活动帧占比（>10% 且 <50% 是健康范围）
- `note_density`: 每秒平均 note 数

### 5. 验证数据不泄漏
- 当前按 clip 随机 split → 同一 track 的不同 clips 可能同时出现在 train/val
- 改为按 track split: 前 85% tracks 做 train, 后 15% 做 val

## 执行顺序

1. 写 `scripts/extract_midi_dataset.py` 提取脚本
2. 提取 Phase 1 数据（50 tracks）
3. 修改 `scripts/train_midi.py`：特征缓存 + track-level split + 增强指标
4. 运行 Phase 1 训练 + 验证
5. 分析结果 → Phase 2 (200 tracks)
6. 分析结果 → Phase 3 (550 tracks)
7. 对比 Audio E2 结论

## 验证方式

每阶段完成后检查:
1. `pianoroll_comparison.png` — 生成 vs 目标有可见的 bass 音符
2. `generated_bass.mid` > 100 bytes（非空）
3. `generated_bass.wav` > 1KB（有音频输出，非静音）
4. 活动预测 F1 > 0.5
5. loss 曲线下降且未过拟合
