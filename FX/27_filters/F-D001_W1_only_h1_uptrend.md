---
id: F-D001_W1_only_h1_uptrend
type: filter
status: draft
---

# F-D001 W1_only + H1_uptrend（事故抑制フィルタ）

## 目的

- family_D_momentum / D001 baseline における「事故抑制フィルタ」を、後出しが効かない形で固定する。
- ここは“regime定義”ではなく、**取引を弾くためのフィルタ**として扱う。

## フィルタ内容

- W1 only（時間帯制約）
  - 定義：`session_label(ts, cfg) == "W1"` のときだけ通す
  - 備考：W1 の具体時間は `backtest_core.Config`（`w1_start`〜`w1_end`）を正典とする
- H1 uptrend（上位足フィルタ）
  - 定義：`compute_h1_uptrend(df10, cfg)` が True のときだけ通す
  - 備考：H1 uptrend の定義（EMA/傾き等）は core 実装を参照するだけ（再発明しない）

## Yes / No 定義（擬似コード）

```text
def pass_F_D001(ts, sess, h1_up) -> bool:
  # W1 only
  if sess != "W1": return False

  # H1 filter ON (existing definition)
  if h1_up is not True: return False

  return True
```

