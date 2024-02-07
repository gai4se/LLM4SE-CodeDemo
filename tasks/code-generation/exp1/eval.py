# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse
from nltk.translate.bleu_score import corpus_bleu
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def calculate_bleu(references, hypotheses):
    ref_list = [[ref.split()] for ref in references]
    hyp_list = [hyp.split() for hyp in hypotheses]
    bleu_score = corpus_bleu(ref_list, hyp_list)
    return bleu_score

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in json format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    with open(args.answers, "r") as f:
        answers_data = json.load(f)
    gts = [ans.strip() for ans in answers_data["outputs"]]

    with open(args.predictions, "r") as f:
        predictions_data = json.load(f)
    preds = [pred.strip() for pred in predictions_data["outputs"]]

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = sum([1 for pred, gt in zip(preds, gts) if pred == gt])

    bleu_score = round(calculate_bleu(gts, preds), 2)
    logger.info(f"BLEU: {bleu_score}, EM: {round(EM/total*100, 2)}")

if __name__ == "__main__":
    main()

#BLEU: 0.39, EM: 33.33 for CodeT5 model