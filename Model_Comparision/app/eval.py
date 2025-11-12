import math
from typing import List, Tuple

# --- BLEU-4 (very small, self-contained) ---
def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]

def _precision(candidate: List[str], reference: List[str], n: int) -> float:
    cand_ngrams = _ngrams(candidate, n)
    ref_ngrams = _ngrams(reference, n)
    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1
    match = 0
    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1
    for ng, c in cand_counts.items():
        match += min(c, ref_counts.get(ng, 0))
    return (match / max(1, len(cand_ngrams))) if cand_ngrams else 0.0

def bleu_4(candidate: str, reference: str) -> float:
    cand = candidate.split()
    ref = reference.split()
    precisions = [_precision(cand, ref, n) for n in (1, 2, 3, 4)]
    geo = 1.0
    for p in precisions:
        geo *= max(p, 1e-9)
    geo **= 0.25
    c = len(cand); r = len(ref)
    bp = 1.0 if c > r else math.exp(1 - r / max(1, c))
    return bp * geo

# --- ROUGE-L ---
def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l(candidate: str, reference: str) -> float:
    cand = candidate.split(); ref = reference.split()
    l = _lcs(cand, ref)
    prec = l / max(1, len(cand)); rec = l / max(1, len(ref))
    beta2 = 1.2**2
    denom = prec + beta2*rec
    return ((1+beta2)*prec*rec/denom) if denom > 0 else 0.0