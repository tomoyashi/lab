#!/bin/bash                                                                                                                                             
#PJM -L "node=10"               # 1ノード                                                                                                                
#PJM -L "rscgrp=small"         # リソースグループの指定                                                                                                 
#PJM -L "elapse=10:00"         # ジョブの経過時間制限値                                                                                                 
#PJM -g ra000006            # グループ指定                                                                                                              
#PJM -x PJM_LLIO_GFSCACHE=/vol0003 # ジョブで使用するデータ領域のvolume                                                                                 
#PJM --mpi "max-proc-per-node=4" # 1ノードあたりに生成するMPIプロセス数の上限値                                                                        
#PJM -s                                                                                                                                                 

export PLE_MPI_STD_EMPTYFILE=off # 標準出力/標準エラー出力への出力がない場合はファイルを作成しない                                                     \
                                                                                                                                                        
export OMP_NUM_THREADS=12

mpiexec -stdout-proc node10/DE -n 40 ./DE_th12
