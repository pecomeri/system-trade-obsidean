背景：上位足で方向性が出ていると、下位のシグナルが“順行”になりやすい

参加者：中期勢＋短期追随

価格の癖：押し戻り後の再加速が出やすい

誤認：横ばいでも無理にトレンド認定する


ロジック
  # 10秒closeからH1の終値系列を作る
  h1_close = resample_1H_last(close_bid)

  # H1終値のEMAを計算
  h1_ema = EMA(h1_close, period=h1_ema_period)

  # EMAの過去値（傾き判定用）
  ema_past = shift(h1_ema, h1_ema_slope_bars)

  # uptrend条件（上昇のみ）
  ok = (h1_close > h1_ema) and (h1_ema > ema_past)

  # “確定足のみ”にするため1本遅らせてから、10秒足へ前方埋め
  ok_closed = shift(ok, 1)
  h1_uptrend_10s = ffill_to_10s(ok_closed)