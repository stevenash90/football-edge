# football-edge

Bộ công cụ Python cho **Premier League (EPL)**:

| Mảng | Module | Nội dung |
|---|---|---|
| Thống kê EPL | `football_edge/stats.py` | Bảng xếp hạng (tổng / sân nhà / sân khách), phong độ N trận, hồ sơ đội (bàn thắng/thua mỗi trận, % Over 2.5, % BTTS, % giữ sạch lưới, sút, tỉ lệ chuyển hoá), đối đầu |
| Phân tích & dự đoán | `football_edge/model.py` | Mô hình **Dixon-Coles** (Poisson + hiệu chỉnh tỉ số thấp + suy giảm theo thời gian): xG, xác suất 1X2, Over/Under 1.5/2.5/3.5, BTTS, tỉ số khả năng cao nhất, kèo châu Á, odds "công bằng"; đánh giá walk-forward (log loss, Brier) |
| Kèo / odds | `football_edge/odds.py` | Xác suất ngầm định, khử margin nhà cái (proportional / power), edge (EV), Kelly phân đoạn, lọc **value bet**, backtest flat & Kelly |
| Fantasy PL | `football_edge/fpl.py` | Dự phóng điểm (xP) theo GW có tính phong độ, thời gian thi đấu, chấn thương, độ khó lịch (FDR), GW trống/kép; **tối ưu đội hình 15 người** bằng quy hoạch nguyên (đúng luật FPL); đề xuất chuyển nhượng có tính -4; fixture ticker; gợi ý chip qua GW trống/kép |

## Nguồn dữ liệu

* **football-data.co.uk**: kết quả, thống kê trận và odds (Bet365, Pinnacle, trung bình/max thị trường, odds đóng cửa) cho từng mùa EPL.
* **Fantasy Premier League API** (công khai): cầu thủ, giá, phong độ, lịch thi đấu, FDR, đội hình của bạn theo team id.

Dữ liệu tải về được cache trong `data/`, nên các lần chạy sau dùng được offline.

## Cài đặt

```bash
pip install -e ".[dev]"
```

## Sử dụng

```bash
# Chạy thử toàn bộ trên dữ liệu giả lập (không cần mạng)
football-edge demo

# Thống kê mùa 2025/26
football-edge stats --seasons 2025

# Dự đoán trận + so sánh với odds nhà cái (H D A)
football-edge predict --seasons 2023 2024 2025 --ratings \
    --match Arsenal "Man City" --odds 2.40 3.40 3.00

# Backtest: mô hình có thắng thị trường không?
football-edge backtest --seasons 2022 2023 2024 2025 --start 2024-08-01 --book avg --blend 0.5 --show 20

# FPL: đội hình tối ưu từ đầu (5 GW tới)
football-edge fpl-draft --horizon 5

# FPL: kế hoạch chuyển nhượng cho đội của bạn (team id trên URL FPL)
football-edge fpl-transfers --entry 1234567 --free-transfers 2 --max-transfers 3
```

Dùng như thư viện:

```python
from football_edge import data, model, odds, fpl

m = data.load_seasons([2023, 2024, 2025])
dc = model.DixonColes().fit(m)
print(dc.predict("Liverpool", "Chelsea"))

d = data.load_fpl()
proj = fpl.project_points(d, horizon=6)
plan = fpl.optimise(proj, budget=100)
print(plan.summary())
```

## Lưu ý thực tế

* Thị trường odds EPL **rất hiệu quả**: luôn so sánh log loss của mô hình với log loss của odds đã khử margin (lệnh `backtest` in cả hai). Thường nên `--blend` mô hình với xác suất thị trường và chỉ cược khi edge rõ ràng. Odds đóng cửa Pinnacle (`--book psc`) là thước đo khắt khe nhất.
* Đây là công cụ phân tích, không đảm bảo lợi nhuận. Hãy cược có trách nhiệm.
* Mô hình xP của FPL là heuristic minh bạch, dễ chỉnh (`FDR_MULT`, `form_weight`, `bench_weight`), có thể thay bằng mô hình dựa trên xG/xA sau.

## Kiểm thử

```bash
pytest
```

Test dùng dữ liệu giả lập có "đáp án" (`football_edge/sample.py`). Chúng kiểm tra rằng mô hình khôi phục được sức mạnh đội, xác suất hợp lệ, và đội hình FPL tuân thủ luật (15 người, 2/5/5/3, tối đa 3 người/CLB, ngân sách, đội hình ra sân hợp lệ, số lần chuyển nhượng).
