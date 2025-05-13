import pandas as pd
import editdistance

# Load predictions
df = pd.read_csv(r"D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\SecondApproach\CRNN\Models\BeamSearch+WeightedSampler\test_predictions.csv")

# CER: average normalized edit distance per sample
cers = [
    editdistance.eval(str(gt), str(pr)) / max(len(str(gt)), 1)
    for gt, pr in zip(df["Ground Truth"], df["Prediction"])
]
mean_cer = sum(cers) / len(cers)

# Full accuracy: exact match percentage
df["match"] = df["Ground Truth"].str.upper().str.strip() == df["Prediction"].str.upper().str.strip()
full_accuracy = df["match"].mean()

# Results
print(f"CER: {mean_cer:.4f}")
print(f"Full Text Accuracy: {full_accuracy * 100:.2f}%")
