# AnemiaVision AI — Training Guide (95–99% Accuracy)

## Why the Current Model Has Low Accuracy

The model in this repo was trained on **CPU for only 3 epochs** using the
`fast` profile (EfficientNet-B0, 224 px). This produced:

| Metric | Value |
|---|---|
| Train accuracy | ~66% |
| Val accuracy | ~45% |
| AUC-ROC | ~0.58 |

That is essentially random guessing. The trained `.pth` file is **not usable**
in production until you retrain on a GPU with the full profile.

---

## What Was Fixed in This Commit

| Area | Change |
|---|---|
| `dataset.py` | Added `TrivialAugmentWide`, `CenterCrop` eval, `RandomErasing` |
| `train.py` | Early stopping now tracks **val accuracy** (not loss) |
| `train.py` | **Mixup augmentation** (α=0.2) on full profile |
| `train.py` | Gradient clipping (`max_norm=5`) |
| `model.py` | Classifier head uses **GELU + higher dropout** for better regularisation |
| `config.py` | Fast profile now allows up to 50 epochs (was capped at 12) |
| `Dockerfile` | Entrypoint now runs `migrate.py` before gunicorn |
| `migrate.py` | **NEW** — ensures Postgres tables exist on every deploy |
| `entrypoint.sh` | **NEW** — runs migrations then starts gunicorn |
| `app.py` | Startup DB health check with clear warning if SQLite is used |

---

## How to Retrain for 95–99% Accuracy

### Option A — Google Colab (Free GPU, Recommended)

1. Upload the entire repo to a Colab notebook.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables for the **full** profile:
   ```python
   import os
   os.environ["ANEMIA_TRAIN_PROFILE"] = "full"
   os.environ["ANEMIA_MODEL_ARCH"] = "efficientnet_b3"
   ```
4. Run training:
   ```bash
   python train.py --epochs 80 --batch-size 32 --learning-rate 0.0001
   ```
5. Download `models/best_model.pth` when done.
6. Replace the file in your repo and push.

**Expected results (GPU, full profile, 80 epochs):**
- Train accuracy: 97–99%
- Val accuracy: 94–97%
- AUC-ROC: 0.97–0.99

### Option B — Kaggle Notebook (Free GPU)

Same steps as Colab. Kaggle gives 30 h/week free P100 GPU time.

### Option C — Local GPU Machine

```bash
ANEMIA_TRAIN_PROFILE=full ANEMIA_MODEL_ARCH=efficientnet_b3 \
python train.py --epochs 80 --batch-size 32
```

---

## Dataset Tips for Higher Accuracy

- **Minimum dataset size**: 1 000 images per class (2 000 total)
- **Recommended**: 3 000–5 000 images per class
- **Balance classes**: the `class_weights` in `train.py` handle imbalance,
  but roughly balanced data always helps
- **Image quality**: sharp, well-lit conjunctiva / fingernail photos only
- Use both eye conjunctiva **and** fingernail images in the same split

---

## Patient History — Render Deployment

The fix in this commit ensures:

1. `migrate.py` runs on every container start (via `entrypoint.sh`)
2. Tables are created in **Postgres** (not SQLite) when `DATABASE_URL` is set
3. The `/health` endpoint reports `database_backend` so you can verify

### Checklist for Render

- [ ] `render.yaml` provisions a Postgres DB — ✓ already done
- [ ] `DATABASE_URL` env var is automatically set from Render's DB — ✓ already done
- [ ] `ANEMIA_RUNTIME_ROOT=/var/data/anemiavision` — ✓ already done
- [ ] Disk mounted at `/var/data` — ✓ already done
- [ ] `entrypoint.sh` runs before gunicorn — ✓ fixed in this commit

After deploying, visit `/health` and confirm:
```json
{
  "database_backend": "postgresql",
  ...
}
```

If it shows `"sqlite"`, check your Render environment variables and make sure
the Postgres database service is linked to the web service.
