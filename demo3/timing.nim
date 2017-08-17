include system/timers

template timex*(rep:var int, nn:int, s:untyped): float =
  let n = nn
  let t = getTicks()
  for i in 0..<n: s             # repeats the expression, `s`, `n` times
  threadSingle: rep += n        # also serves as a barrier
  1e-9 * float(getTicks()-t)    # return seconds elapsed
