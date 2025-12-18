# vit_train (clean)

This repo supports **training + inference for unlabeled test submission**.
- No pseudo-labeling
- No local benchmark on labeled test
- Model selection and threshold should be decided on your **validation split only**

## 1) Data layout (default)
- `data/2-MedImage-TrainSet/normal/*.jpg`
- `data/2-MedImage-TrainSet/disease/*.jpg`
- `data/MedImage-TestSet/*.jpg`  (unlabeled, for submission)

## 2) Train
Stage1 (224):
```bash
GPU_ID=0 RUN_ID=run_swin_01 STAGE=1 MODEL_NAME="swin_base_patch4_window7_224" ./train.sh
```

Stage2 (448 fine-tune):
```bash
GPU_ID=0 RUN_ID=run_swin_01 STAGE=2 MODEL_NAME="swin_base_patch4_window7_224" ./train.sh
```

Notes:
- Every epoch checkpoint is saved under `models/run_<RUN_ID>/`
- Training writes `*_meta.json` with `best_thr` (computed on validation)

## 3) TensorBoard
```bash
./monitor.sh
```

## 4) Submit (inference on unlabeled test only)
Pick a specific epoch:
```bash
RUN_ID=run_swin_01 STAGE=2 SELECT="epoch:006" ./submit.sh
```

Or average the last K epochs:
```bash
RUN_ID=run_swin_01 STAGE=2 AVG_LAST_K=3 ./submit.sh
```

Override threshold (recommended if you choose it manually from validation):
```bash
RUN_ID=run_swin_01 STAGE=2 THRESHOLD=0.62 ./submit.sh
```
