p(v,m,t) = sprintf("< awk '$2==%d && $3==%d && $4~/%s/{print}' ../out/neddy.ftm.alcf.anl.gov.ex2",v,m,t)
set key outside width 4
set log x
set xlabel 'Memory footprint (KB)'
set ylabel 'Effective bandwidth (GB/s)'
set xrange [50:15000000]
plot \
  p( 8,1, 'CPU') u 8:14 w l ls  1 t  'V=8, M=1, CPU', \
  p( 8,2, 'CPU') u 8:14 w l ls 21 t  'V=8, M=2, CPU', \
  p(16,1, 'CPU') u 8:14 w l ls  2 t 'V=16, M=1, CPU', \
  p(16,2, 'CPU') u 8:14 w l ls 22 t 'V=16, M=2, CPU', \
  p(32,1, 'CPU') u 8:14 w l ls  3 t 'V=32, M=1, CPU', \
  p(32,2, 'CPU') u 8:14 w l ls 23 t 'V=32, M=2, CPU', \
  p(64,1, 'CPU') u 8:14 w l ls  4 t 'V=64, M=1, CPU', \
  p(64,2, 'CPU') u 8:14 w l ls 24 t 'V=64, M=2, CPU', \
  p( 8,1,'GPU5') u 8:14 w l ls  5 t  'V=8, M=1, GPU T/B=32', \
  p( 8,2,'GPU5') u 8:14 w l ls 25 t  'V=8, M=2, GPU T/B=32', \
  p(16,1,'GPU5') u 8:14 w l ls  6 t 'V=16, M=1, GPU T/B=32', \
  p(16,2,'GPU5') u 8:14 w l ls 26 t 'V=16, M=2, GPU T/B=32', \
  p(32,1,'GPU5') u 8:14 w l ls  7 t 'V=32, M=1, GPU T/B=32', \
  p(32,2,'GPU5') u 8:14 w l ls 27 t 'V=32, M=2, GPU T/B=32', \
  p(64,1,'GPU5') u 8:14 w l ls  8 t 'V=64, M=1, GPU T/B=32', \
  p(64,2,'GPU5') u 8:14 w l ls 28 t 'V=64, M=2, GPU T/B=32'
