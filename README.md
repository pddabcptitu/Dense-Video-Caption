# Dense Video Caption (DenseVidCap)

## Giới thiệu

Dự án này xây dựng hệ thống **Dense Video Caption** nhằm sinh chú thích động cho video theo từng đoạn thời gian (temporal localization + captioning). Hệ thống tự động tạo các mô tả ngắn gọn cho các sự kiện trong video cùng với thời gian xảy ra.

### Thành phần chính
- **model/dataset/dataset.py**: Bộ xử lý dataset, augmentation dữ liệu
- **model/vid2seq.py**: Kiến trúc mô hình video-to-sequence
- **trainer.py**: Script huấn luyện mô hình
- **infer.py**: Script test và inference
- **extract/**: Module trích đặc trưng từ video

---

## Yêu cầu hệ thống

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **transformers**: HuggingFace library
- **Thư viện hỗ trợ**: numpy, tqdm, Pillow, json, torch

---

## Cài đặt môi trường

### Bước 1: Tạo virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Bước 2: Cập nhật pip và cài đặt dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Cài đặt thêm các thư viện quan trọng (nếu cần)

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install numpy tqdm Pillow
```

---

## Chuẩn bị dữ liệu

### 1. Định dạng JSON

Cần chuẩn bị **2 file JSON**: `train.json` và `test.json` với cấu trúc:

```json
[
  {
    "video_id": "video_001",
    "feature": "video_001.pt",
    "duration": 120.5,
    "sentences": [
      "Người ăn bánh mỳ",
      "Cô gái uống nước"
    ],
    "timestamps": [
      [5.2, 15.3],
      [20.1, 35.8]
    ]
  },
  ...
]
```

**Giải thích các trường:**
- `video_id`: Mã định danh video (duy nhất)
- `feature`: Tên file đặc trưng (`*.pt`)
- `duration`: Độ dài video (giây)
- `sentences`: Danh sách các mô tả (caption)
- `timestamps`: Danh sách thời gian [t_start, t_end] ứng với mỗi caption

### 2. Đặc trưng Video (Feature Files)

Các file đặc trưng phải được lưu dưới dạng PyTorch tensor (`.pt`) trong `--feature_dir`:

```python
import torch

# Feature shape: (T, D) hoặc (1, T, D)
# T: số frames, D: số chiều embedding
video_feature = torch.randn(100, 768)  # 100 frames, 768-dim embedding
torch.save(video_feature, "video_001.pt")
```

**Lưu ý quan trọng:**
- **Trainer default**: `max_feats=100` (interpolate frames → 100)
- **Infer default**: `max_feats=225` (interpolate frames → 225) ⚠️
- Khi inference, phải match `max_feats` với checkpoint training!
- Nếu training dùng 100 frames, infer cũng phải dùng `--max_feats 100`

---

## Cấu hình và training

### Tham số cấu hình (trainer.py)

| Tham số | Loại | Mặc định | Mô tả |
|---------|------|----------|-------|
| `--train_data` | str | **Required** | Đường dẫn file train.json |
| `--test_data` | str | **Required** | Đường dẫn file test.json |
| `--feature_dir` | str | **Required** | Thư mục chứa features (.pt) |
| `--checkpoint` | str | **Required** | Checkpoint ban đầu |
| `--output_dir` | str | `checkpoints` | Thư mục lưu checkpoint |
| `--t5_path` | str | `t5-base` | Mô hình T5 base |
| `--epochs` | int | 30 | Số epoch training |
| `--batch_size` | int | 4 | Kích thước batch |
| `--lr` | float | 3e-5 | Learning rate |
| `--warmup_ratio` | float | 0.1 | Warmup ratio (10% tổng steps) |
| `--patience` | int | 5 | Early stopping patience |
| `--max_feats` | int | 100 | Số frames interpolate |
| `--num_bins` | int | 100 | Số time bins |
| `--max_output_tokens` | int | 256 | Độ dài tối đa caption |
| `--seed` | int | 42 | Random seed |
| `--num_workers` | int | 2 | Số worker DataLoader |
| `--contrastive_weight` | float | 0.1 | Trọng số contrastive loss |

### Chạy training

**Ví dụ đầy đủ:**
```bash
python trainer.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --feature_dir data/features \
    --checkpoint checkpoints/initial.pth \
    --output_dir checkpoints \
    --epochs 30 \
    --batch_size 4 \
    --lr 3e-5
```

**Ví dụ tối thiểu (dùng default):**
```bash
python trainer.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --feature_dir data/features \
    --checkpoint checkpoints/initial.pth
```

### Augmentation flags

Tất cả augmentation được bật mặc định. Dùng flag `--no_*` để tắt:

```bash
# Tắt tất cả augmentation
python trainer.py \
    --train_data ... \
    --test_data ... \
    --feature_dir ... \
    --checkpoint ... \
    --no_augment

# Tắt riêng một số kỹ thuật
python trainer.py \
    ... \
    --no_speed_jitter \
    --no_temporal_crop \
    --no_gaussian_noise \
    --no_feature_dropout \
    --no_boundary_emphasis
```

**Các kỹ thuật augmentation:**
- `--no_speed_jitter`: Tắt thay đổi tốc độ phát
- `--no_temporal_crop`: Tắt cắt đoạn video ngẫu nhiên
- `--no_gaussian_noise`: Tắt thêm nhiễu Gaussian
- `--no_feature_dropout`: Tắt dropout theo thời gian
- `--no_boundary_emphasis`: Tắt nhấn mạnh biên giới event

---

## Chạy Inference (Test)

### Tham số cấu hình (infer.py)

| Tham số | Loại | Mặc định | Mô tả |
|---------|------|----------|-------|
| `--feature_path` | str | **Required** | Đường dẫn file feature (.pt) |
| `--duration` | float | **Required** | Độ dài video (giây) |
| `--checkpoint` | str | **Required** | Checkpoint model |
| `--t5_path` | str | `t5-base` | Mô hình T5 base |
| `--max_feats` | int | 225 | Số frames tối đa ⚠️ |
| `--num_bins` | int | 100 | Số time bins |

⚠️ **Chú ý**: `max_feats` mặc định trong infer là **225**, khác với trainer (100). Phải match với checkpoint!

### Chạy inference trên một video

```bash
python infer.py \
    --feature_path data/features/video_001.pt \
    --duration 120.5 \
    --checkpoint checkpoints/best.pth
```

### Chạy với tham số tuỳ chỉnh

```bash
python infer.py \
    --feature_path data/features/video_001.pt \
    --duration 120.5 \
    --checkpoint checkpoints/best.pth \
    --t5_path t5-base \
    --max_feats 100 \
    --num_bins 100
```

### Đầu ra

```
Device: cuda

[1] Loading model...
[2] Loading video feature...
[3] Generating caption...

Raw prediction:
<time=5><time=15> Người ăn bánh mỳ <time=20><time=35> Cô gái uống nước

[4] Decoding timestamps...

Final results:
{'caption': 'Người ăn bánh mỳ', 'starts': 5.2, 'ends': 15.3}
{'caption': 'Cô gái uống nước', 'starts': 20.1, 'ends': 35.8}
```

---

## Mô tả module chính

### model/dataset/dataset.py
- Class `Vid2SeqDataset`: Xử lý dữ liệu, tạo batch, augmentation
- Method `_build_target()`: Xây dựng target string với timeline tokens
- Method `_load_video()`: Load features và interpolate số frames

**Timeline tokens:**
- Format: `<time=X>` (X từ 0 đến num_bins-1)
- Mã hóa thời gian thành discrete bins (mặc định 100 bins)

### model/vid2seq.py
- Encoder: Xử lý video features
- Decoder: Sinh caption token-by-token
- Tokenizer: BertTokenizer hoặc tương tự

### extract/
- `extract.py`: Trích đặc trưng từ video gốc
- `model_video_extract.py`: Mô hình pre-trained dùng để extract
- `video_loader.py`: Load và xử lý video

---

## Troubleshooting

### Lỗi: "Missing .pt files"
- **Nguyên nhân**: File feature không tồn tại hoặc tên không khớp
- **Giải pháp**: Kiểm tra `--feature_dir`, đảm bảo tên file trùng với `item["feature"]` hoặc `item["video_id"] + ".pt"`

### Lỗi: "timestamps không khớp duration"
- **Nguyên nhân**: t_end > duration hoặc format không đúng
- **Giải pháp**: Kiểm tra JSON, t_end phải ≤ duration, timestamps phải là `[[t0, t1], ...]`

### Lỗi: Feature shape không khớp
- **Nguyên nhân**: `max_feats` training khác infer, hoặc feature embedding dim sai
- **Giải pháp**: 
  ```bash
  # Nếu training dùng max_feats=100, infer cũng phải dùng:
  python infer.py ... --max_feats 100
  ```

### Lỗi: CUDA out of memory
- **Giải pháp**: Giảm `--batch_size` (thử 2 hoặc 1) hoặc giảm `--max_feats`

### Lỗi: Model expects checkpoint nhưng không match
- **Nguyên nhân**: Checkpoint được train với config khác (num_bins, model size, v.v.)
- **Giải pháp**: Đảm bảo `--t5_path`, `--num_bins`, `--max_feats` giống lúc training

### Encoding file không đúng
- **Giải pháp**: Dùng UTF-8 encoding khi lưu JSON/README
  ```python
  import json
  with open("data.json", "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
  ```

---