from deep_translator import GoogleTranslator
from transformers import NllbTokenizer, M2M100ForConditionalGeneration
import torch
import time

# GLOBAL CACHED MODEL OBJECTS
_sunbird_model = None
_sunbird_tokenizer = None
_sunbird_lang_tokens = None
_DEVICE = None

# GOOGLE TRANSLATE FUNCTION
def translate_google(sentence, target_lang="lg"):
    try:
        result = GoogleTranslator(source='auto', target=target_lang).translate(sentence)
        return result
    except Exception as e:
        print(f"[Google Translate Error] {e}")
        return None


# SUNBIRD TRANSLATION FUNCTION (Lightweight Model)
def translate_sunbird(sentence: str, source_lang: str = "eng", target_lang: str = "lug", max_new_tokens: int = 100):
    """
    Translate text from source_lang → target_lang using the lightweight Sunbird NLLB model.
    Automatically runs on CUDA, MPS, or CPU. Model is cached after first load.
    """
    global _sunbird_model, _sunbird_tokenizer, _sunbird_lang_tokens, _DEVICE

    try:
        # === Step 1: Load model/tokenizer only once ===
        if _sunbird_model is None or _sunbird_tokenizer is None:
            print("[Sunbird] Loading translation model...")
            _DEVICE = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"[Sunbird] Using device: {_DEVICE}")

            _sunbird_tokenizer = NllbTokenizer.from_pretrained("Sunbird/translate-nllb-1.3b-salt")
            _sunbird_model = M2M100ForConditionalGeneration.from_pretrained(
                "Sunbird/translate-nllb-1.3b-salt"
            ).to(_DEVICE)

            # Mapping of Sunbird’s language tokens
            _sunbird_lang_tokens = {
                'eng': 256047,
                'ach': 256111,
                'lgg': 256008,
                'lug': 256110,
                'nyn': 256002,
                'teo': 256006,
            }

        # === Step 2: Prepare input ===
        inputs = _sunbird_tokenizer(sentence, return_tensors="pt").to(_DEVICE)
        inputs["input_ids"][0][0] = _sunbird_lang_tokens.get(source_lang, 256047)

        # === Step 3: Generate translation ===
        translated_tokens = _sunbird_model.generate(
            **inputs,
            forced_bos_token_id=_sunbird_lang_tokens.get(target_lang, 256110),
            max_length=max_new_tokens,
            num_beams=5,
        )

        result = _sunbird_tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        return result.strip()

    except Exception as e:
        print(f"[Sunbird Translation Error] {e}")
        return None


# BACK-TRANSLATION FUNCTION
def back_translate(sentence, source_lang="lug", model="google"):
    """
    Back-translate a sentence from source_lang → English.
    """
    if model.lower() == "google":
        try:
            return GoogleTranslator(source=source_lang, target="en").translate(sentence)
        except Exception as e:
            print(f"[Google Back-translate Error] {e}")
            return None
    elif model.lower() == "sunbird":
        try:
            return translate_sunbird(sentence, source_lang=source_lang, target_lang="eng")
        except Exception as e:
            print(f"[Sunbird Back-translate Error] {e}")
            return None
    else:
        raise ValueError("Unknown model. Choose 'google' or 'sunbird'.")


# === Example usage ===
if __name__ == "__main__":
    text = "The patient has a high fever and persistent cough."

    print("\nGoogle Translation:")
    print(translate_google(text, target_lang="lg"))

    print("\nSunbird Translation:")
    print(translate_sunbird(text, source_lang="eng", target_lang="lug"))
