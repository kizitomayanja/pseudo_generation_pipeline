from translators import back_translate
from sentence_transformers import SentenceTransformer, util
from sacrebleu import corpus_bleu
# 4. EVALUATION FUNCTION

# Initialize similarity model
semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def evaluate_translation(original, translations):
    """
    Evaluates translations by BLEU + semantic similarity of back-translation.
    Returns the best model name and its score.
    """
    scores = {}
    for name, trans in translations.items():
        back = back_translate(trans)
        bleu = corpus_bleu([back], [[original]]).score
        sim = util.cos_sim(
            semantic_model.encode(original),
            semantic_model.encode(back)
        ).item()
        # Weighted score
        final_score = 0.7 * bleu + 0.3 * sim * 100
        scores[name] = final_score
    best_model = max(scores, key=scores.get)
    return best_model, scores
