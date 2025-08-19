# modules/nli_multilabel_final.py
import re
import json
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm


# NLTK 준비
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# 텍스트 전처리
def clean_text(text: str) -> str:
    text = re.sub(r"\n", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 라벨 팩 정의 (기존과 동일)
LABEL_CORE = [
    # Service/Operations
    "wait time and queue management (waitlist, minutes, seated, reservation)",
    "host and seating process (hostess, greeting, table assignment)",
    "server friendliness and politeness (friendly, rude, courteous)",
    "server attentiveness and follow-ups (refills, check on us, ignored)",
    "order accuracy and missing items (order accuracy, missing, correct)",
    "kitchen speed and ticket time (slow kitchen, quick turnaround)",
    "bill handling and split checks (check, split bill, receipt)",
    "payment methods and checkout (contactless, tap to pay, cash only)",
    "manager response and recovery (manager, apology, comped)",
    # Cleanliness/Safety
    "tableware and utensils cleanliness (fork, plate, glass, stains)",
    "dining area cleanliness (floor, wiped, bussing, trash)",
    "restroom cleanliness and supplies (bathroom, soap, paper)",
    # Environment/Ambience
    "noise level and crowding (loud, noisy, packed)",
    "music volume and selection (music loud, playlist)",
    "lighting and visibility (dim, bright, ambiance)",
    "temperature and ventilation (AC, draft, hot, cold)",
    "smell and odors (grease smell, odor)",
    "decor and interior design (decor, vibe, cozy, dated)",
    "seating comfort and space (cramped, comfortable, booth, high chair)",
    "outdoor seating and patio (patio, terrace, heaters, shade)",
    # Accessibility/Family
    "parking convenience and options (parking lot, street parking, valet)",
    "location and transit accessibility (walkable, subway, bus)",
    "ada accessibility and ramps (wheelchair, ramp, accessible restroom)",
    "family friendly and kids options (kids menu, stroller, highchair)",
    # Menu/Value/Food
    "menu variety and seasonal specials (options, choices, seasonal)",
    "menu clarity and descriptions (pictures, translations)",
    "value for money and price fairness (worth, expensive)",
    "portion size and fullness (small portion, generous, filling)",
    "allergen handling and cross contamination (allergy, celiac, nut free)",
    "overall taste and seasoning balance (flavor, bland, salty, sweet)",
    "ingredient freshness and quality (fresh, stale, frozen)",
    "texture and doneness accuracy (overcooked, undercooked, tender, crispy)",
    "temperature of dishes at serving (served cold, piping hot, lukewarm)",
    # Delivery/Takeout
    "online ordering usability (app, website, checkout)",
    "delivery time and temperature (late delivery, cold on arrival)",
    "takeout packaging and spill protection (container, leak, soggy)",
]

PACK_MENU_GENERIC = [
    "breakfast and brunch dishes (pancakes, benedict, omelet)",
    "sandwiches and burgers (burger, bun, BLT, club)",
    "pizza and flatbreads (pizza, crust, slice)",
    "pasta and noodles (pasta, spaghetti, noodles)",
    "steaks and grilled meats (steak, medium rare, ribeye)",
    "barbecue and smoked meats (bbq, brisket, ribs)",
    "seafood dishes general (shrimp, crab, oyster, salmon)",
    "sushi and raw items (sushi, sashimi, nigiri)",
    "tacos and mexican dishes (taco, burrito, salsa)",
    "asian stir-fry and rice dishes (fried rice, stir fry, curry)",
    "salads and healthy bowls (salad, bowl, greens)",
    "soups and stews (soup, broth, stew)",
    "appetizers and sides (starter, fries, wings)",
    "desserts and sweets (dessert, cake, pie, ice cream)",
    "pastries and baked goods (croissant, pastry, bread)",
]

PACK_BEVERAGE_BAR = [
    "coffee and espresso drinks (latte, cappuccino, espresso)",
    "tea and non coffee beverages (tea, herbal, iced tea)",
    "cocktails quality and consistency (cocktail, watered down, balanced)",
    "beer selection and craft options (tap list, ipa, lager)",
    "wine list and pairing (wine list, pairing, by the glass)",
    "happy hour value and timing (happy hour, specials, discount)",
]

PACK_DELIVERY_DETAIL = [
    "third party delivery issues (Doordash, Uber Eats, driver, handoff)",
    "order status communication (tracking, ready time, text)",
    "curbside pickup and handoff (curbside, pickup window, parking spot)",
]

# 트리거 키워드
TRIGGERS_MENU = ["pizza","burger","sushi","sashimi","taco","ramen","pho","bbq",
                 "steak","pasta","noodle","salad","soup","fries","wings","dessert","croissant"]
TRIGGERS_BEVERAGE = ["latte","espresso","coffee","tea","cocktail","ipa","lager","wine","happy hour"]
TRIGGERS_DELIVERY = ["delivery","doordash","ubereats","grubhub","pickup","curbside","driver","tracking"]


def _share_with_triggers(sentences: List[str], triggers: List[str]) -> float:
    """문장들에서 특정 트리거 키워드가 나타나는 비율 계산"""
    trigs = [t.lower() for t in triggers]
    def hit(s: str) -> bool:
        s = s.lower()
        return any(t in s for t in trigs)
    return float(np.mean([hit(s) for s in sentences])) if sentences else 0.0


def build_candidate_labels(sentences: List[str],
                          mode: str = "auto",
                          menu_thresh: float = 0.006,
                          bev_thresh: float = 0.004,
                          deliv_thresh: float = 0.004) -> List[str]:
    """
    라벨 후보 목록 생성
    mode:
      - "core": 코어 라벨만
      - "full": 코어 + 모든 선택 팩
      - "auto": 코어 + 트리거 비율 임계 통과한 팩만 자동 추가
    """
    labels = LABEL_CORE.copy()
    if mode == "core":
        return labels
    if mode == "full":
        return labels + PACK_MENU_GENERIC + PACK_BEVERAGE_BAR + PACK_DELIVERY_DETAIL

    # mode == "auto"
    if _share_with_triggers(sentences, TRIGGERS_MENU) >= menu_thresh:
        labels += PACK_MENU_GENERIC
    if _share_with_triggers(sentences, TRIGGERS_BEVERAGE) >= bev_thresh:
        labels += PACK_BEVERAGE_BAR
    if _share_with_triggers(sentences, TRIGGERS_DELIVERY) >= deliv_thresh:
        labels += PACK_DELIVERY_DETAIL
    return labels


class NLIMultiLabelClassifier:
    """NLI 기반 멀티라벨 분류기"""
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-mnli",
                 device: str = "auto",
                 batch_size: int = 16,
                 local_files_only: bool = True):
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.model_name = model_name
        
        # NLI 파이프라인 초기화
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                local_files_only=local_files_only
            )
        except Exception as e:
            print(f"Warning: Failed to load model {model_name} with local_files_only={local_files_only}")
            if local_files_only:
                print("Retrying with local_files_only=False...")
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    local_files_only=False
                )
            else:
                raise e
    
    def classify_sentences(self, 
                          sentences: List[str], 
                          labels: List[str],
                          min_prob: float = 0.5,
                          hypothesis_template: str = "This review sentence is about {}.") -> List[Dict]:
        """
        문장들을 멀티라벨 분류
        
        Returns:
            List of dicts with keys: 'sentence', 'labels', 'scores', 'assigned_labels'
        """
        results = []
        
        # 배치 처리
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="NLI Classification"):
            batch_sentences = sentences[i:i + self.batch_size]
            
            for sentence in batch_sentences:
                try:
                    # Zero-shot classification 수행
                    result = self.classifier(
                        sentence, 
                        labels,
                        hypothesis_template=hypothesis_template,
                        multi_label=True
                    )
                    
                    # 결과 처리
                    assigned_labels = []
                    assigned_scores = []
                    
                    for label, score in zip(result['labels'], result['scores']):
                        if score >= min_prob:
                            assigned_labels.append(label)
                            assigned_scores.append(score)
                    
                    # 라벨이 하나도 할당되지 않은 경우 가장 높은 스코어의 라벨을 할당
                    if not assigned_labels and result['labels'] and result['scores']:
                        assigned_labels = [result['labels'][0]]
                        assigned_scores = [result['scores'][0]]
                    
                    results.append({
                        'sentence': sentence,
                        'all_labels': result['labels'],
                        'all_scores': result['scores'],
                        'assigned_labels': assigned_labels,
                        'assigned_scores': assigned_scores,
                        'max_score': max(result['scores']) if result['scores'] else 0.0
                    })
                    
                except Exception as e:
                    print(f"Error processing sentence: {str(e)[:100]}...")
                    results.append({
                        'sentence': sentence,
                        'all_labels': [],
                        'all_scores': [],
                        'assigned_labels': [],
                        'assigned_scores': [],
                        'max_score': 0.0
                    })
        
        return results


def create_pseudo_topics_from_labels(classification_results: List[Dict], 
                                   sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    멀티라벨 결과를 기반으로 pseudo-topic을 생성
    각 고유한 라벨 조합을 하나의 토픽으로 취급
    """
    
    # 각 문장의 라벨 조합을 문자열로 변환
    topic_assignments = []
    topic_labels = []
    
    for i, result in enumerate(classification_results):
        if result['assigned_labels']:
            # 라벨들을 정렬해서 일관성 유지
            sorted_labels = sorted(result['assigned_labels'])
            topic_label = "_".join(sorted_labels)
            topic_assignments.append(topic_label)
            topic_labels.append(topic_label)
        else:
            # 라벨이 없는 경우 noise로 처리
            topic_assignments.append("NOISE")
            topic_labels.append("NOISE")
    
    # 고유한 topic 라벨들을 숫자 ID로 변환
    unique_topics = list(set(topic_assignments))
    topic_to_id = {topic: i for i, topic in enumerate(unique_topics)}
    
    # NOISE는 -1로 설정
    if "NOISE" in topic_to_id:
        topic_to_id["NOISE"] = -1
    
    # 문장들에 topic_id와 label 할당
    sentences_df = sentences_df.copy()
    sentences_df['topic_id'] = [topic_to_id[topic] for topic in topic_assignments]
    sentences_df['topic_label'] = topic_assignments
    
    # 각 문장에 개별 라벨들과 스코어도 저장
    sentences_df['assigned_labels'] = [result['assigned_labels'] for result in classification_results]
    sentences_df['assigned_scores'] = [result['assigned_scores'] for result in classification_results]
    sentences_df['max_label_score'] = [result['max_score'] for result in classification_results]
    
    return sentences_df


def apply_nli_multilabel_for_business(
    sentences_csv_path: str,
    out_csv_path: str,
    business_id: Optional[str] = None,
    nli_model: str = "facebook/bart-large-mnli",
    min_prob: float = 0.5,
    label_mode: str = "auto",
    batch_size: int = 16,
    local_files_only: bool = True,
    business_meta: Optional[Dict] = None
):
    """
    NLI 기반 멀티라벨 분류를 적용하는 메인 함수
    """
    
    print("[Stage] Load & filter...")
    # 데이터 로드 및 필터링
    df = pd.read_csv(sentences_csv_path)
    
    # per-store vs pooled
    if business_id in (None, "__ALL__", "*ALL*"):
        d = df.copy()
        print("[Mode] POOLED across all selected businesses")
    else:
        d = df[df["business_id"] == business_id].copy()
    
    if d.empty:
        raise ValueError(f"No rows for business_id={business_id}")
    
    # 비즈니스 메타데이터 추가
    if business_meta and "business_id" in d.columns:
        d["business_address"] = d["business_id"].map(
            lambda x: (business_meta.get(x, {}) or {}).get("address", "")
        )
        if "business_name" not in d.columns or d["business_name"].eq("").all():
            d["business_name"] = d["business_id"].map(
                lambda x: (business_meta.get(x, {}) or {}).get("name", "")
            )
    
    # 문장 전처리
    docs_raw = d["sentence"].fillna("").astype(str).tolist()
    clean_docs = [clean_text(t) for t in docs_raw]
    
    # 최소 토큰 길이 필터
    token_min = 4
    token_counts = [len(x.split()) for x in clean_docs]
    keep_mask = [tc >= token_min for tc in token_counts]
    d = d.loc[keep_mask].reset_index(drop=True)
    clean_docs = [x for x, k in zip(clean_docs, keep_mask) if k]
    
    if not clean_docs:
        raise ValueError("All sentences removed after cleaning/length filter.")
    
    n_docs = len(clean_docs)
    print(f"[Info] Processing {n_docs} sentences")
    
    # 라벨 후보 생성
    print("[Stage] Building candidate labels...")
    candidate_labels = build_candidate_labels(clean_docs, mode=label_mode)
    print(f"[Info] Using {len(candidate_labels)} candidate labels")
    
    # NLI 분류기 초기화
    print("[Stage] Initializing NLI classifier...")
    classifier = NLIMultiLabelClassifier(
        model_name=nli_model,
        batch_size=batch_size,
        local_files_only=local_files_only
    )
    
    # 멀티라벨 분류 수행
    print("[Stage] Performing multi-label classification...")
    classification_results = classifier.classify_sentences(
        sentences=clean_docs,
        labels=candidate_labels,
        min_prob=min_prob
    )
    
    # Pseudo-topic 생성
    print("[Stage] Creating pseudo-topics from labels...")
    d_with_topics = create_pseudo_topics_from_labels(classification_results, d)
    
    # 토픽 키워드 생성 (라벨이 키워드 역할)
    def get_topic_keywords(topic_label: str) -> str:
        if topic_label == "NOISE" or pd.isna(topic_label):
            return ""
        # 언더스코어로 연결된 라벨들을 쉼표로 구분
        labels = topic_label.split("_")
        # 각 라벨에서 주요 키워드만 추출 (괄호 앞 부분)
        keywords = []
        for label in labels:
            main_part = label.split("(")[0].strip()
            keywords.append(main_part)
        return ", ".join(keywords[:5])  # 최대 5개만
    
    d_with_topics["topic_keywords"] = d_with_topics["topic_label"].apply(get_topic_keywords)
    
    # 결과 저장
    print("[Stage] Save outputs...")
    d_with_topics.to_csv(out_csv_path, index=False)
    
    # 토픽 요약 생성
    summary_df = d_with_topics.groupby("topic_id", as_index=False).agg(
        n=("sentence", "count"),
        pos=("sentiment", lambda x: float(np.mean(x == 1))),
        stars_mean=("stars", "mean"),
        conf_mean=("sentiment_conf", "mean")
    )
    
    # 라벨 정보 추가
    topic_to_label = d_with_topics.groupby("topic_id")["topic_label"].first().to_dict()
    summary_df["label"] = summary_df["topic_id"].map(topic_to_label)
    summary_df["keywords"] = summary_df["topic_id"].map(
        lambda tid: get_topic_keywords(topic_to_label.get(tid, ""))
    )
    
    summary_df["share"] = summary_df["n"] / summary_df["n"].sum()
    summary_df = summary_df.sort_values("n", ascending=False)
    
    # 요약 파일 저장
    summary_df.to_csv(out_csv_path.replace(".csv", "_topic_summary_with_labels.csv"), index=False)
    
    # 월별 트렌드 (날짜가 있는 경우)
    if "date" in d_with_topics.columns:
        d_with_topics["_ym"] = pd.to_datetime(d_with_topics["date"], errors="coerce").dt.to_period("M").astype(str)
        trend = d_with_topics[d_with_topics["_ym"].notna()].groupby(["_ym", "topic_id"], as_index=False).size()
        trend.rename(columns={"size": "count"}, inplace=True)
        trend.to_csv(out_csv_path.replace(".csv", "_topic_trend_by_month.csv"), index=False)
    
    # 예시 문장들
    examples = (
        d_with_topics.sort_values("max_label_score", ascending=False)
        .groupby("topic_id").head(5)[["topic_id", "topic_label", "sentence", "stars", "sentiment", "sentiment_conf", "date"]]
    )
    examples.to_csv(out_csv_path.replace(".csv", "_topic_examples.csv"), index=False)
    
    print("[Stage] Done.")
    print(f"[Summary] Generated {len(summary_df[summary_df['topic_id'] != -1])} topics from {n_docs} sentences")
    
    return d_with_topics, summary_df