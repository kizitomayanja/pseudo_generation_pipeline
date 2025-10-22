import os
import time
import requests
from tqdm import tqdm

def generate_medical_sentences(
    output_file="medical_sentences.txt",
    target_count=10000,
    batch_size=10,
    sleep_time=2,
    model="llama3-70b-8192",   # Groq model
    temperature=0.8
):
    """
    Generates realistic medical sentences using the Groq API and saves them to a text file.

    Parameters:
        output_file (str): File to save generated sentences.
        target_count (int): Total number of sentences to generate.
        batch_size (int): Number of sentences to request per API call.
        sleep_time (int): Delay between API calls (seconds).
        model (str): Groq model to use.
        temperature (float): Sampling temperature for variation.

    Notes:
        - The function resumes automatically if the file already contains sentences.
        - Duplicate sentences are avoided.
    """

    # --- Setup Groq API credentials ---
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # --- Resume from existing file ---
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_sentences = [line.strip() for line in f if line.strip()]
    else:
        existing_sentences = []

    start_index = len(existing_sentences)
    print(f"[INFO] Found {start_index} existing sentences. Resuming generation...")

    # --- Prompt for generation ---
    PROMPT = """Generate 10 short, unique, realistic medical sentences.
Each sentence should describe a diagnosis, symptom, procedure, or observation.
Examples:
- The CT scan revealed a small tumor in the left lung.
- The patient reported persistent chest pain and shortness of breath.
- The X-ray showed multiple rib fractures.
Now write 10 new sentences like these in simple, clear medical language.
"""

    # --- Generation loop ---
    with open(output_file, "a", encoding="utf-8") as f:
        for i in tqdm(range(start_index, target_count, batch_size), desc="Generating"):
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": temperature,
                }

                response = requests.post(api_url, headers=headers, json=payload, timeout=60)

                if response.status_code != 200:
                    print(f"[ERROR] Error code {response.status_code} - {response.text}")
                    print("[INFO] Retrying in 10 seconds...")
                    time.sleep(10)
                    continue

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                sentences = content.strip().split("\n")
                sentences = [s.strip("-•1234567890. ").strip() for s in sentences if s.strip()]

                new_sentences = [s for s in sentences if s and s not in existing_sentences]

                for s in new_sentences:
                    f.write(s + "\n")
                    existing_sentences.append(s)

                f.flush()
                time.sleep(sleep_time)

            except Exception as e:
                print(f"[ERROR] {e}")
                print("[INFO] Retrying in 10 seconds...")
                time.sleep(10)
                continue

    print(f"[DONE] Saved {len(existing_sentences)} sentences to {output_file}")
    return existing_sentences