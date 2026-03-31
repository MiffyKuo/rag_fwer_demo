from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 正確文件沒撈到就失敗
from rouge_score import rouge_scorer
import re
import math

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 漏太多就失敗
def retriever_fail(retrieved_docs, gold_doc_ids, tau_1=0.0):
    """
    loss_1 = 1 - (# retrieved relevant docs / # all relevant docs)
    """
    retrieved_ids = {d.metadata["doc_id"] for d in retrieved_docs}
    gold_set = set(gold_doc_ids)

    if len(gold_set) == 0:
        return 1.0, 1

    covered = len(retrieved_ids & gold_set)
    loss_1 = 1.0 - covered / len(gold_set)
    A_i = int(loss_1 > tau_1)
    return loss_1, A_i

# 分數太低就失敗(代表relevent沒排在前面)
def reranker_fail(reranked_docs, gold_doc_ids, tau_2=0.0):
    """
    簡化版 nDCGmod：
    relevant docs 越排前面越好
    """
    doc_ids = [d.metadata["doc_id"] for d in reranked_docs]
    gold_set = set(gold_doc_ids)

    if len(gold_set) == 0:
        return 1.0, 1

    dcg = 0.0
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(rank + 1.0)

    ideal_hits = min(len(gold_set), len(doc_ids))
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))

    if idcg == 0:
        loss_2 = 1.0
    else:
        ndcg_mod = dcg / idcg
        loss_2 = 1.0 - ndcg_mod

    B_i = int(loss_2 > tau_2)
    return loss_2, B_i

# 答案跟標準答案差太多就失敗

# 標準答案有沒有被回答出來
import re

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def generator_fail(generation_set, gold_answer, tau_3=0.0):
    gold = normalize_text(gold_answer)

    if len(generation_set) == 0:
        return 1.0, 1

    hit = 0
    for ans in generation_set:
        pred = normalize_text(ans)
        if gold in pred:
            hit = 1
            break

    risk = 0.0 if hit else 1.0
    fail = int(risk > tau_3)
    return risk, fail

# # ROUGE -> 句子像不像
# def generator_fail(pred_answer, gold_answer, tau_3=0.4):
#     rougeL = scorer.score(gold_answer, pred_answer)["rougeL"].fmeasure
#     risk = 1 - rougeL
#     C_i = int(risk > tau_3)
#     return risk, C_i