from deep_translator import GoogleTranslator
from load_model import sunbird_model, tokenizer, DEVICE, SYSTEM_MESSAGE, RETRY_COUNT, RETRY_DELAY_SECONDS
import time

# GOOGLE TRANSLATE FUNCTION
def translate_google(sentence, target_lang="lg"):
    try:
        result = GoogleTranslator(source='auto', target=target_lang).translate(sentence)
        return result
    except Exception as e:
        print(f"[Google Translate Error] {e}")
        return None

# SUNBIRD SUNFLOWER FUNCTION
def translate_sunbird(sentence: str, source_lang: str = "en", target_lang: str = "lg", max_new_tokens: int = 500):
    """
    Translate text from source_lang → target_lang using the Sunbird Sunflower chat-style prompt.
    Automatically runs on CUDA, MPS, or CPU.
    """
    lang_map_human = {
        "en": "English",
        "lg": "Luganda",
    }
    src_human = lang_map_human.get(source_lang, source_lang)
    tgt_human = lang_map_human.get(target_lang, target_lang)
    instruction = f"Translate from {src_human} to {tgt_human}: {sentence}"

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": instruction},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        print(f"[Sunbird] Chat template not available ({e}); using fallback prompt.")
        prompt = f"{SYSTEM_MESSAGE}\n\nUser: {instruction}\nAssistant:"

    # Tokenize input and move to device
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Try generation with retries
    for attempt in range(RETRY_COUNT + 1):
        try:
            outputs = sunbird_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                do_sample=True,
                temperature=0.5,
            )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()
        except RuntimeError as e:
            print(f"[Sunbird] RuntimeError (attempt {attempt+1}/{RETRY_COUNT+1}): {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                return None
        except Exception as e:
            print(f"[Sunbird] Error (attempt {attempt+1}/{RETRY_COUNT+1}): {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                return None

# BACK-TRANSLATION FUNCTION
def back_translate(sentence, source_lang="lg", model="google"):
    """
    Back-translate a sentence from source_lang → English.
    
    Parameters:
    - sentence: str, sentence in source language
    - source_lang: 'lg' or other code
    - model: 'google' or 'sunbird'
    
    Returns:
    - str: back-translated sentence in English
    """
    if model.lower() == "google":
        try:
            return GoogleTranslator(source=source_lang, target="en").translate(sentence)
        except Exception as e:
            print(f"[Google Back-translate Error] {e}")
            return None
    elif model.lower() == "sunbird":
        try:
            # Use the translate_sunbird function we defined earlier
            return translate_sunbird(sentence, source_lang=source_lang, target_lang="en")
        except Exception as e:
            print(f"[Sunbird Back-translate Error] {e}")
            return None
    else:
        raise ValueError("Unknown model. Choose 'google' or 'sunbird'.")

# Example usage
if __name__ == "__main__":
    text = "The patient has a high fever and persistent cough."
    translation = translate_google(text, target_lang="lg")
    print(f"Google Translation: {translation}\n")
    print(f"Sunbird Translation: {translate_sunbird(text, source_lang='en', target_lang='lg')}\n")
