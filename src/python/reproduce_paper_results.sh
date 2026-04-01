#!/bin/bash
SECONDS=0
# Local dataset root directories
surrogate_root="../../data/SIComp_Benchmark/SetA_Surrogate"
real_root="../../data/SIComp_Benchmark/SetA_Real_Compensation" #Evaluate using different sets of real-world compensation data
data_names=("color_1" "color_2" "love_1" "love_2" "paint_1" "paint_2" "wood_1" "wood_2")
export DATASET_ROOT="$surrogate_root"

#cd SIComp/src/python/ || exit
for data_name in "${data_names[@]}"; do
    echo "Starting compensation generation"
    start_time=$SECONDS
    export DATA_NAME="$data_name"
    echo -e " [SIComp(#surf=1)] Running model compensation..."
    python SIComp\(#surf=1\).py --mode "valid"
    echo -e "✅ [SIComp(#surf=1)] Compensation completed\n"
#
    echo " [SIComp(#surf=3)] Running model compensation..."
    python SIComp\(#surf=3\).py --mode "valid"
    echo -e "✅ [SIComp(#surf=3)] Compensation completed\n"

    echo " [SIComp(#surf=5)] Running model compensation..."
    python SIComp\(#surf=5_L1+SSIM+DeltaE+LPIPS\).py --mode "valid"
    echo -e "✅ [SIComp(#surf=5)] Compensation completed\n"

    echo " [SIComp(#surf=5 L1+SSIM)] Running model compensation..."
    python SIComp\(#surf=5_L1+SSIM\).py --mode "valid"
    echo -e "✅ [SIComp(#surf=5 L1+SSIM)] Compensation completed\n"

    echo " [SIComp(#surf=5 L1+SSIM+DeltaE)] Running model compensation..."
    python SIComp\(#surf=5_L1+SSIM+DeltaE\).py --mode "valid"
    echo -e "✅ [SIComp(#surf=5 L1+SSIM+DeltaE)] Compensation completed\n"

    echo " [SIComp(#surf=5 L1+SSIM+LPIPS)] Running model compensation..."
    python SIComp\(#surf=5_L1+SSIM+LPIPS\).py --mode "valid"
    echo -e "✅ [SIComp(#surf=5 L1+SSIM+LPIPS)] Compensation completed\n"

    echo " [FF+CompenNeSt] Running model compensation..."
    python FF+CompenNeSt.py --mode "valid"
    echo -e "✅ [FF+CompenNeSt] Compensation completed\n"

    echo -e " [FF+CompenNeSt(Origin)] Running model compensation..."
    python FF+CompenNeSt\(Origin\).py --mode "valid"
    echo -e "✅ [FF+CompenNeSt(origin)] Compensation completed\n"

    echo " [FF+PANet] Running model compensation..."
    python FF+PANet.py --mode "valid"
    echo -e "✅ [FF+PANet] Compensation completed\n"

    echo -e " [FF+PANet(Origin)] Running model compensation..."
    python FF+PANet\(Origin\).py --mode "valid"
    echo -e "✅ [FF+PANet(origin)] Compensation completed\n"

    current_total_seconds=$SECONDS
    item_duration=$((current_total_seconds - start_time))

    echo "------------------------------------------------"
    echo "✅ Dataset [$data_name] finished processing!"
    echo " Current run time: $((item_duration / 60)) min $((item_duration % 60)) sec"
    echo " Total elapsed time: $((current_total_seconds / 60)) min $((current_total_seconds % 60)) sec"
    echo "------------------------------------------------"
done

# --- Evaluate surrogate metrics ---
echo "================================================"
echo " Evaluating surrogate metrics..."
export DATASET_ROOT="$surrogate_root"
python calculate_surrogate_metrics.py

# --- Evaluate actual metrics ---
echo "================================================"
echo " Evaluating actual compensation metrics..."
export DATASET_ROOT="$real_root"
python calculate_real_compensation_metrics.py

total_duration=$SECONDS
echo "================================================"
echo " Done!"
echo " Total runtime: $((total_duration / 3600)) h $(((total_duration % 3600) / 60)) min $((total_duration % 60)) sec"
