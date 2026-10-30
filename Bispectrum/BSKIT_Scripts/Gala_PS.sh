for i in {0..24}; do
    export FILE_ROOT=$(printf "/mnt/nvme1/kboone/Data/BigFiles/6E11_Gala_5/Gala_ph0%02d/Gala_256" "$i")
    
    export IN_FILE=${FILE_ROOT}.bigfile
    export NGRID=256
    export NGRIDCIC=256
    export BOX_SIZE=2000.
    
    export DK_IN_KF=1.
    export KMIN_IN_KF=0.5
    
    export OUT_JSON_FILE=${FILE_ROOT}_ps.json
    export OUT_DAT_FILE=${FILE_ROOT}_ps.dat
    
    mpirun -np 8 python ../../../../bskit/scripts/measure/measure_ps_from_bigfile.py $IN_FILE $OUT_JSON_FILE $OUT_DAT_FILE $NGRID $NGRIDCIC $BOX_SIZE $DK_IN_KF $KMIN_IN_KF

done