"""
Medical Translation Evaluation Pipeline
----------------------------------------
This script:
1. Generates pseudo-medical English sentences using an LLM.
2. Translates each sentence using multiple translation models.
3. Back-translates to English.
4. Scores translations via BLEU + semantic similarity.
5. Saves the best translations to a CSV file.

Replace the translation stubs with actual model/API calls.
"""

import numpy as np
import openai
import re

# Make sure your API key is set:
# export OPENAI_API_KEY="sk-xxxx"
# or in Windows: setx OPENAI_API_KEY "sk-xxxx"

def generate_medical_sentences(
    num_sentences=10000,
    batch_size=50,
    model="gpt-3.5-turbo",
    max_tokens=100,
    temperature=0.7
):
    """
    Efficiently generate a large number of medical sentences using OpenAI API.
    
    Parameters:
    - num_sentences: total number of sentences to generate
    - batch_size: how many sentences per API call
    - model: OpenAI model
    - max_tokens: max tokens per batch
    - temperature: randomness for variety
    
    Returns:
    - list of unique sentences
    """
    sentences = set()
    while len(sentences) < num_sentences:
        # Prompt instructs the model to generate multiple numbered sentences
        prompt = (
            f"Generate {batch_size} concise, realistic medical sentences in English. "
            "Focus on General Medicine and diagnosis, covering different symptoms, tests, "
            "medications, or procedures. Output as a numbered list."
        )
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            text = response.choices[0].message['content'].strip()

            # Extract sentences from numbered list
            batch_sentences = re.split(r'\d+\.\s+', text)
            batch_sentences = [s.strip() for s in batch_sentences if s.strip()]

            sentences.update(batch_sentences)
        except Exception as e:
            print(f"[Error generating batch]: {e}")
            continue

        print(f"[INFO] Collected {len(sentences)}/{num_sentences} sentences...")

    return list(sentences)[:num_sentences]



if __name__ == "__main__":
    # Test sentence generation
    generated_sentences = generate_medical_sentences(10)
    for sent in generated_sentences:
        print(sent)