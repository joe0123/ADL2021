from tw_rouge import get_rouge

def compute_rouge(predictions, references, avg=True):
    predictions = [pred.strip() + '\n' for pred in predictions]
    references = [ref.strip() + '\n' for ref in references]
    return get_rouge(predictions, references, avg)



