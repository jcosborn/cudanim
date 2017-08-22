p(v,m,t) = sprintf("< awk '$2==%d && $3==%d && $4~/%s/{print}' ../out/kingly.ex2",v,m,t)
set log x
set key outside width 4
set xlabel 'Memory footprint (KB)'
set ylabel 'Effective bandwidth (GB/s)'
set xrange [50:15000000]
plot \
  p(16,1, 'CPU') u 8:14 w l ls  2 t 'V=16, M=1, CPU', \
  p(16,2, 'CPU') u 8:14 w l ls 22 t 'V=16, M=2, CPU', \
  p(32,1, 'CPU') u 8:14 w l ls  3 t 'V=32, M=1, CPU', \
  p(32,2, 'CPU') u 8:14 w l ls 23 t 'V=32, M=2, CPU', \
  p(64,1, 'CPU') u 8:14 w l ls  4 t 'V=64, M=1, CPU', \
  p(64,2, 'CPU') u 8:14 w l ls 24 t 'V=64, M=2, CPU'
