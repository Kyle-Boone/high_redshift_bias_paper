export OUTFILE_GRID="/mnt/nvme1/kboone/Sims/BigFiles/Matter_grid_auto.dat"

export NGRID=256
export NGRIDCIC=256
export NTSC=2048

export LBOX=2000.
export STARTI=0
export ENDI=4000

export DK=0.006283185307179587
export KMIN=0.0015707963267948967
export KMAX=0.20263272615654168

for i in {0..24}; do

    export POS_FILE=$(printf "../../Data/6E11_Gala_z5_Positions/Gala_ph0%02d.h5" "$i")
    export FILE_ROOT=$(printf "/mnt/nvme1/kboone/Data/BigFiles/6E11_Gala_5/Gala_ph0%02d/Gala_256" "$i")
    export SNAP=${FILE_ROOT}.bigfile
    
    python De_Pos_To_Mesh.py $POS_FILE $SNAP $LBOX $NGRID $NTSC
    
    export OUTFILE_UNNORM=${FILE_ROOT}_unnormbs.dat
    export OUTFILE_FINAL=${FILE_ROOT}_comb.dat
    
    mpirun -np 8 python ../../../../bskit/scripts/measure/measure_bs_fast.py bigfile_grid $SNAP $NGRID $NGRIDCIC $LBOX $STARTI $ENDI $OUTFILE_UNNORM --k_bin_info $KMIN $KMAX $DK  --triangle_type all
    
    python ../../../../bskit/scripts/process/process_fast_bs_measurement.py $OUTFILE_GRID $OUTFILE_UNNORM $KMAX $OUTFILE_FINAL

done
