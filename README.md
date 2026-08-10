# IoVFD — tái hiện trên CICIoV

> Cao, Di, Jin, *IoVFD*, Future Generation Computer Systems 183 (2026)

Tái hiện trên **CICIoV** (31 đặc trưng, 13 lớp) để so sánh được với ba bài còn
lại và với AFSIC-IoV / FedLiTeCAN trên cùng một backbone.

Ba repo anh em: [VANFED-IDS](https://github.com/TongXuanVu/VANFED-IDS) ·
[FEDIOV](https://github.com/TongXuanVu/FEDIOV) · [IOVFD](https://github.com/TongXuanVu/IOVFD) ·
[SDNFL-IDS](https://github.com/TongXuanVu/SDNFL-IDS)

---

## Ý tưởng được tái hiện

Chưng cất tri thức hai chiều. Tại client: model global làm giáo viên cho model local (`L = Focal + λ·KL`), model local không bị ghi đè nên giữ tính cá nhân hoá. Tại server: ensemble các model local làm giáo viên cho global, chưng cất trên pseudo-data do generator sinh — không dùng dữ liệu thật. Nhiễu lấy từ hệ hỗn loạn thay vì `torch.randn`.

## Cài đặt

```bash
git clone https://github.com/TongXuanVu/IOVFD.git
cd IOVFD
pip install -r requirements.txt
```

## Chạy

Cần **1 server + N client**. Muốn chạy tay thì mở N+1 terminal, server trước:

```bash
python server_iov.py --rounds 30 --num-clients 10 --data-dir <DATA>
python client_iov.py --client-id 0 --data-dir <DATA>
python client_iov.py --client-id 1 --data-dir <DATA>
```

Trên Kaggle/Colab không mở được nhiều terminal — dùng `run_fl.py`, nó tự sinh
server + N client và chạy nối tiếp task 0→4 (class-incremental), resume giữa
các task nên số round liên tục:

```bash
python run_fl.py --data-dir <DATA> --clients 10 --rounds 20
python run_fl.py --data-dir <DATA> --tasks none      # FL thường, gộp cả 5 task
```

Tắt KD để có đường đối chứng FedAvg thuần:

```bash
python run_fl.py --data-dir <DATA> --server-extra="--kd-steps 0"
```


> `--server-extra` và `--client-extra` **bắt buộc viết dạng có dấu `=`**.
> Viết cách ra sẽ lỗi `expected one argument` vì argparse tưởng là option mới.

### Ba chế độ

```bash
python server_iov.py --mode train  --rounds 30
python server_iov.py --mode resume --rounds 50           # chạy tiếp từ latest.pth
python server_iov.py --mode test   --ckpt out/checkpoints/latest.pth
```

## Dữ liệu

Định dạng khớp AFSIC-IoV:

```
<DATA>/federated_data/client_<id>_task_<t>.pt    # t = 1..5, dict {'x','y'}
<DATA>/global_test_data.pt
<DATA>/class_mapping.json
```

Chia lớp theo task: `TASK_INCREMENTS = [3, 3, 3, 2, 2]` (13 lớp / 5 task).
`run_fl.py` tự bỏ qua client thiếu file của task đang chạy thay vì để server
treo chờ mãi.

## Kết quả

Đổ vào `--out-dir` (mặc định `out/`):

| File | Nội dung |
|---|---|
| `metrics_task*.csv` | 1 dòng/round, 12 cột: loss, accuracy, micro/macro/weighted P-R-F1 |
| `confusion_matrix_task*.csv` / `_normalized.csv` / `.png` | cuối mỗi task |
| `classification_report_task*.txt` | P/R/F1 từng lớp |
| `checkpoints/round_NNN.pth`, `latest.pth` | resume được |

Gộp nhiều lần chạy + đo mức độ quên:

```bash
python collect_results.py --runs A=out_a B=out_b --out-dir ket_qua
```

Sinh `comparison.csv`, ma trận quên từng lần chạy (`forgetting_*.csv`), và
`accuracy_curve.png`. Mức độ quên tính theo định nghĩa chuẩn class-incremental:
`forgetting(j) = max_{t<T} acc(j,t) − acc(j,T)`.

## Kiểm thử

```bash
python smoke_test.py
```

Tự sinh dữ liệu giả đúng định dạng, chạy 2 round, kiểm CSV đủ 12 cột, checkpoint
nạp lại được, confusion matrix có sinh ra, và cả `--mode test` lẫn `--mode resume`.

**Trạng thái:** **Đã chạy thật và đạt** trên dữ liệu giả: vòng generator↔student chạy đủ, checkpoint lưu kèm trọng số generator nên resume được. Chưa dò `beta_div`/`beta_adv`, chưa chạy trên CICIoV thật.

## Khác gì so với bài báo

"Nhiễu Gauss từ hệ hỗn loạn" cài bằng ánh xạ logistic (r=3.99) + Box–Muller; bài không nói rõ dùng hệ hỗn loạn nào. Cấu trúc generator, learning rate, số bước chưng cất là **suy đoán** — bài chỉ mô tả bằng lời và công thức loss.

Bài gốc **không công bố source code**. Mọi con số phải tự đo lại, không kỳ vọng
khớp bảng kết quả trong bài.

## Ghi chú về code dùng chung

`common.py` và `model_cnn1d.py` giống hệt ở cả 4 repo (cố ý — bốn bài phải dùng
chung backbone thì so sánh mới công bằng). Sửa một nơi thì phải đồng bộ cả 4;
`check_shared.py` đối chiếu hash giúp phát hiện lệch bản:

```bash
python check_shared.py --against ../VANFED-IDS
```
