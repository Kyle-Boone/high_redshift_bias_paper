export FILE_ROOT="/mnt/nvme1/kboone/Amplitude_Sims/"

export NGRID=256
export NGRIDCIC=256

export LBOX=2000.
export STARTI=0
export ENDI=20000

export DK=0.006283185307179587        # 2*kF # 0.025, divide by factor of 20/3 due to larger box
export KMIN=0.0015707963267948967     # 0.5*kF # 0.025
export KMAX=0.20263272615654168       # 64.5*kF # 2.025

export OUTFILE_GRID="${FILE_ROOT}BigFiles/Matter_grid.dat"

for v in $(seq 1 25); do
    echo "Running version ${v}"

    export SNAP1="${FILE_ROOT}BigFiles/d2_v${v}.bigfile"
    export SNAP2="${FILE_ROOT}BigFiles/d2_v${v}.bigfile"
    export SNAP3="${FILE_ROOT}BigFiles/d2_v${v}.bigfile"

    export OUTFILE_UNNORM="${FILE_ROOT}Bispectra/b2b2b2_unnormbs_v${v}.dat"
    export OUTFILE_FINAL="${FILE_ROOT}Bispectra/b2b2b2_comb_v${v}.dat"

    mpirun -np 8 python measure_bs_fast_copy.py \
        bigfile_grid \
        "$SNAP1" \
        "$NGRID" \
        "$NGRIDCIC" \
        "$LBOX" \
        "$STARTI" \
        "$ENDI" \
        "$OUTFILE_UNNORM" \
        --k_bin_info "$KMIN" "$KMAX" "$DK" \
        --triangle_type all \
        --snap_prefix2 "$SNAP2" \
        --snap_prefix3 "$SNAP3"

    python ../../../../bskit/scripts/process/process_fast_bs_measurement.py \
        "$OUTFILE_GRID" \
        "$OUTFILE_UNNORM" \
        "$KMAX" \
        "$OUTFILE_FINAL"

done

