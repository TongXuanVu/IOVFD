# IoVFD — tái hiện trên CICIoV

> Cao, Di, Jin, *IoVFD*, Future Generation Computer Systems 183 (2026)

Tái hiện trên **CICIoV** (31 đặc trưng, 13 lớp) để so sánh được với ba bài còn
lại và với AFSIC-IoV / FedLiTeCAN trên cùng một backbone.

Ba repo anh em: [VANFED-IDS](https://github.com/TongXuanVu/VANFED-IDS) ·
[FEDIOV](https://github.com/TongXuanVu/FEDIOV) · [IOVFD](https://github.com/TongXuanVu/IOVFD) ·
[SDNFL-IDS](https://github.com/TongXuanVu/SDNFL-IDS)

---

## Ý tưởng được tái hiện

| Thành phần của bài | Ở đâu | Trạng thái |
|---|---|---|
| Eq. (9) `p(y)` từ tần suất nhãn client | `server_iov.py: aggregate_fit` | ✅ |
| Eq. (10)+(12) nhãn mềm `y*` từ hệ hỗn loạn | `generator.py: noisy_onehot` | ✅ |
| Eq. (14)-(16) generator Linear → 3×(MultiHead+Conv1D) → Linear | `generator.py: Generator` | ✅ |
| Eq. (18) `L_kl_same` khớp nhãn **mềm** | `generator.py: kl_to_soft_label` | ✅ |
| Eq. (19) `L_dis` đa dạng theo khoảng cách nhãn | `generator.py: diversity_loss` | ✅ |
| Eq. (20) `L_G = L_kl_same + L_dis` | `server_iov.py: _distill` | ✅ |
| Eq. (26) `L_GM = CE + λ₁·KL` | `server_iov.py: _distill` | ✅ |
| Eq. (29) `L_LM = CE + λ₂·KL` | `client_iov.py: fit` | ✅ |
| Algorithm 1 dòng 2 — pretrain local | `client_iov.py: _pretrain` | ✅ |
| Algorithm 1 dòng 10 — model local bền vững qua round | `client_iov.py` + `--state-dir` | ✅ |
| λ₁ = λ₂ = 0.2, lr 1e-4, batch 32 | mặc định | ✅ |
| ECDSA, kênh mã hoá | — | ❌ không có |

## Chạy — một lệnh

```bash
python main.py --data_dir <DATA> --num_users 100
```

Mặc định đã là chế độ của bài. Đối chứng từng thành phần:

```bash
python main.py --data_dir <DATA> --no_personalized   # bỏ chưng cất global→local
python main.py --data_dir <DATA> --kd_steps 0        # bỏ chưng cất local→global
python main.py --data_dir <DATA> --fedavg_init       # thêm FedAvg (bài không có)
python main.py --data_dir <DATA> --noise_mode boxmuller   # Gauss thật
```

## Cài đặt

```bash
git clone https://github.com/TongXuanVu/IOVFD.git
cd IOVFD
pip install -r requirements.txt
```

### Trên Kaggle — clone đúng repo này là chạy được

```python
!git clone -q https://github.com/TongXuanVu/IOVFD.git /kaggle/working/IOVFD
!pip install -q flwr
CODE = "/kaggle/working/IOVFD"
DATA = "/kaggle/input/iov-100client"      # Kaggle Dataset chứa federated_data/

!cd {CODE} && python run_fl.py --data-dir {DATA} --clients 10 --rounds 20
```

Kaggle đã có sẵn torch / numpy / scikit-learn / matplotlib, chỉ thiếu `flwr`.
Repo này không phụ thuộc ba repo kia — không cần clone thêm gì.

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

### 1. Eq. (12) không sinh ra nhiễu Gauss

Bài viết `z = (1/√(2π))·exp(−u²/2)` và gọi đó là *"standard Gaussian noise
sampling"*. Nhưng đó là **hàm mật độ** Gauss, không phải phép lấy mẫu. Với
`u ∈ (0,1)` từ ánh xạ logistic, kết quả **luôn nằm trong [0.2420, 0.3989]** —
không có trung bình 0, không có phương sai 1, không bao giờ âm. Đo được đúng
khoảng đó.

Đặt vào Eq. (10) thì `y* = (1−z, z)` với `z ∈ [0.24, 0.40]`, tức là **nhãn mềm
kiểu label smoothing**. Xét như vậy thì hợp lý — chỉ là tên gọi sai. Mặc định
chạy đúng chữ; `--noise_mode boxmuller` cho Gauss thật để đối chứng.

### 2. Algorithm 1 không có FedAvg

Không dòng nào trong Algorithm 1 trung bình tham số. Global model được tạo
**hoàn toàn** bằng chưng cất từ pseudo-data. Đầu ra của thuật toán cũng là các
model **local**, không phải global. Nên mặc định ở đây bỏ hẳn bước FedAvg —
`--fedavg_init` bật lại để đối chứng.

**Rủi ro cần biết trước khi chạy thật:** vì global chỉ học từ pseudo-data, chất
lượng của nó phụ thuộc hoàn toàn vào generator. Trên dữ liệu giả 8 client / 6
round, cả hai cấu hình đều mắc ở mức lớp đa số (acc 0.335 vs 0.336) — tức thí
nghiệm đồ chơi **không phân biệt được**. Cần `--kd_steps` đủ lớn và dữ liệu thật
mới kết luận được.

### 3. Eq. (20) chỉ có hai số hạng

Bản trước trừ thêm `β_adv · KL(global ‖ ensemble)` — đó là loss của họ
DFAD/DENSE, **không có trong bài này**. Đã bỏ, `--beta_adv` mặc định 0.

### 4. Generator: đầu vào là nhãn, không phải vector nhiễu

Bài **không** đưa vector nhiễu `z` riêng vào generator. Nhiễu được gấp vào chính
cái nhãn (Eq. 10), rồi `x̃ = G(y*)` (Eq. 13). Bản MLP cũ nhận `(z, y)` đã **bỏ
hẳn** — giữ lại chỉ tạo ra một cờ chắc chắn sập.

### 5. Bài là bài toán nhị phân

Bảng 3 của bài: normal / abnormal. CICIoV có 13 lớp. Eq. (10) được mở rộng: lớp
đúng giữ khối lượng `(1−z)`, phần `z` chia đều cho 12 lớp còn lại. Với 2 lớp thì
trùng y nguyên Eq. (10).

### 6. Trạng thái local phải ghi ra đĩa

Trong chế độ simulation, đối tượng client bị **tạo lại mỗi round** nên model
"cá nhân hoá" sẽ bị dựng lại từ đầu — cơ chế lõi của bài hỏng **âm thầm**.
Trước đây `run_sim.py` chặn hẳn P3 vì lý do đó. Nay trạng thái lưu vào
`<out_dir>/client_state/`, và server ghi metric `resumed_local` mỗi round để
kiểm chứng: **từ round 2 phải bằng số client**, nếu bằng 0 là cơ chế đã hỏng.

Đo được: có `--state-dir` → `[0, 1, 1]`; không có → `[0, 0, 0]`.

---

## Ghi chú cũ

"Nhiễu Gauss từ hệ hỗn loạn" cài bằng ánh xạ logistic (r=3.99) + Box–Muller; bài không nói rõ dùng hệ hỗn loạn nào. Cấu trúc generator, learning rate, số bước chưng cất là **suy đoán** — bài chỉ mô tả bằng lời và công thức loss.

Bài gốc **không công bố source code**. Mọi con số phải tự đo lại, không kỳ vọng
khớp bảng kết quả trong bài.

## Sửa code

Repo này là **nguồn gốc của chính nó**. Sửa thẳng ở đây, không có bước build
trung gian nào cả. Sửa repo này không đụng gì tới ba repo kia.

```bash
# sửa file...
push.bat "sua gi do"        # Windows
./push.sh "sua gi do"       # Linux/Mac
```

### Về `common.py` và `model_cnn1d.py`

Hai file này ban đầu giống hệt ở cả 4 repo — bốn bài dùng chung backbone thì so
sánh mới công bằng. Khi bạn sửa riêng ở đây, chúng sẽ lệch dần so với ba repo
kia. **Đó là đánh đổi có chủ đích** để bốn repo độc lập thật sự.

Nhưng nếu đang so sánh kết quả giữa bốn bài thì backbone lệch nhau sẽ làm phép
so sánh mất giá trị. Kiểm tra trước khi kết luận:

```bash
python check_shared.py --against ../VANFED-IDS
```
