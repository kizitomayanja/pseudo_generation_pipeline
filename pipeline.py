
from translators import translate_google, translate_sunbird
from evaluate import evaluate_translation
from generate_data import generate_medical_sentences
import time
import pandas as pd
from tqdm import tqdm
# ============================================
# 1. CONFIGURATION
# ============================================

OUTPUT_FILE = "best_medical_translations.csv"
NUM_SENTENCES = 10000
TARGET_LANG = "lg"  # Luganda (example)
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 2  # to avoid API rate limits
# ============================================
# 5. MAIN PIPELINE
# ============================================

def main():
    print("Generating pseudo-medical sentences...")
    output = "medical_sentences.txt"
    sentences = generate_medical_sentences(output_file=output)

    results = []

    for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
        batch = sentences[i:i + BATCH_SIZE]
        for original in batch:
            # Run translations
            translations = {
                "google": translate_google(original, TARGET_LANG),
                "sunbird": translate_sunbird(original),
                # "nllb": translate_local_nllb(original, TARGET_LANG)
            }

            # Evaluate
            best_model, scores = evaluate_translation(original, translations)

            results.append({
                "original": original,
                "best_model": best_model,
                "best_translation": translations[best_model],
                "scores": scores
            })

        # Periodically save progress
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"Saved progress at {len(results)} sentences...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print("✅ Done! Results saved to:", OUTPUT_FILE)

# ============================================
# 6. ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
