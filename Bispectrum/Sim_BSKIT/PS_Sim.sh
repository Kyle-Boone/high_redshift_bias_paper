export FILE_ROOT="/mnt/nvme1/kboone/Amplitude_Sims/"

export NGRID=256
export NGRIDCIC=256

export DK_IN_KF=1.
export KMIN_IN_KF=0.5

export LBOX=2000.

for v in $(seq 1 25); do
    echo "Running power spectrum version ${v}"

    export SNAP1="${FILE_ROOT}BigFiles/K2_v${v}.bigfile"
    export SNAP2="${FILE_ROOT}BigFiles/K2_v${v}.bigfile"

    export OUT_JSON_FILE="${FILE_ROOT}PowerSpectra/K2K2_ps_v${v}.json"
    export OUT_DAT_FILE="${FILE_ROOT}PowerSpectra/K2K2_ps_v${v}.dat"

    mpirun -np 8 python ../../../../bskit/scripts/measure/measure_ps_from_bigfile.py \
        "$SNAP1" \
        "$OUT_JSON_FILE" \
        "$OUT_DAT_FILE" \
        "$NGRID" \
        "$NGRIDCIC" \
        "$LBOX" \
        "$DK_IN_KF" \
        "$KMIN_IN_KF" \
        --cross "$SNAP2"

done

