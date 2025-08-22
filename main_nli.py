# main_nli.py (NLI 기반, 로컬 테스트 최적화 - CORE 라벨만 사용)
import argparse
from pathlib import Path
import pandas as pd
import re
import time

from config import (
    PATH_BUSINESS, PATH_REVIEW,
    PATH_REVIEWS_FILTERED, PATH_SENT_WITH_SENT, PATH_WITH_TOPICS,
    SENTIMENT_MODEL, EMBEDDING_MODEL
)
from modules.find_business_ids import find_business_ids
from modules.filter_reviews import filter_reviews_by_business_ids
from modules.sentence_sentiment import run_sentence_sentiment
from modules.business_meta import load_business_meta
from modules.nli_multilabel_final import apply_nli_multilabel_for_business  # 수정된 모듈로 변경


def parse_args():
    p = argparse.ArgumentParser(
        description="Mini Yelp DeBERTa NLI pipeline: N businesses → sentence-level sentiment → DeBERTa Multi-Label (CORE labels only, 10+ words)"
    )
    # 대상 비즈니스 선택
    p.add_argument("--name-substr", type=str, default=None)
    p.add_argument("--category", type=str, default="Restaurants")
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--state", type=str, default=None)
    p.add_argument("--biz-id", action="append", help="Explicit business_id (can repeat)")
    p.add_argument("--limit", type=int, default=2)

    # 파이프라인 단계 선택
    p.add_argument("--skip-filter", action="store_true", help="Skip review filtering (use existing filtered file)")
    p.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis (use existing sentiment file)")
    p.add_argument("--do-nli", action="store_true", help="Run NLI multi-label analysis")
    p.add_argument("--nli-only", action="store_true", help="Only run NLI labeling (skip all preprocessing)")
    
    # 🚀 로컬 테스트 최적화 옵션
    p.add_argument("--quick-test", action="store_true", help="Quick test mode: sample 100 sentences per business")
    p.add_argument("--sample-size", type=int, default=100, help="Number of sentences to sample per business in quick test")
    p.add_argument("--test-single-biz", action="store_true", help="Test with only the first business found")
    
    # NLI 스코프
    p.add_argument("--nli-scope", type=str, default="per-store",
                   choices=["per-store", "pooled", "both"],
                   help="Run NLI per store, pooled across stores, or both")

    # 임계치(업체별 최소 문장/리뷰 수) - 로컬 테스트용으로 낮춤
    p.add_argument("--per-store-min-sentences", type=int, default=20,  # 80 → 20으로 낮춤
                   help="Run per-store NLI only if sentence count ≥ this (default: 20)")
    p.add_argument("--per-store-min-reviews", type=int, default=0,
                   help="Optional: pre-filter stores by raw review count (0=ignore)")

    # 🎯 NLI 파라미터
    p.add_argument("--nli-model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli",
                   choices=[
                       "MoritzLaurer/DeBERTa-v3-base-mnli",
                       "cross-encoder/nli-deberta-v3-base", 
                       "microsoft/deberta-v3-large-mnli",       
                       "microsoft/deberta-base-mnli",           # 클래식 버전
                       "microsoft/deberta-large-mnli",          # 클래식 대형
                       "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",  # 멀티 데이터셋 훈련
                       "facebook/bart-large-mnli",              # BART (비교용)
                       "roberta-large-mnli"                     # RoBERTa (비교용)
                   ],
                   help="NLI model for classification (DeBERTa recommended for best performance)")
    p.add_argument("--min-prob", type=float, default=0.35,  # DeBERTa 최적값
                   help="Minimum probability threshold (0.35 recommended for DeBERTa)")
    p.add_argument("--batch-size", type=int, default=8,  # 16 → 8로 낮춤 (로컬 최적화)
                   help="Batch size for NLI processing")
    # --label-mode 제거됨 (CORE 라벨만 사용)
    p.add_argument("--local-only", action="store_true", default=True,  # DeBERTa는 일반적으로 캐시됨
                   help="Use only cached models (no online download)")

    # 출력 경로/파일명 - app.py와 호환성 보장
    p.add_argument("--out-dir", type=str, default="./outputs_nli",  # app.py 기본 디렉토리와 일치
                   help="Directory for NLI outputs; default = ./outputs_nli (app.py compatible)")
    p.add_argument("--filename-template", type=str,
                   default="with_topics__{scope}__{addr_slug}.csv",  # app.py 호환 패턴
                   help="Placeholders: {scope}, {addr_slug} (app.py compatible format)")

    # 실행 요약 인덱스 파일  
    p.add_argument("--write-index", action="store_true", default=True,
                   help="Write an index CSV summarizing per-store run status (nli_run_index.csv)")
    
    # 🔍 디버깅 옵션
    p.add_argument("--verbose", action="store_true", help="Verbose output for debugging")
    p.add_argument("--dry-run", action="store_true", help="Dry run: show what would be done without executing")

    return p.parse_args()


def slugify(s: str, max_len: int = 60) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "store"


def sample_sentences_for_quick_test(sent_df: pd.DataFrame, sample_size: int, verbose: bool = False) -> pd.DataFrame:
    """빠른 테스트를 위한 문장 샘플링"""
    if len(sent_df) <= sample_size:
        if verbose:
            print(f"[Quick Test] Dataset size ({len(sent_df)}) ≤ sample size ({sample_size}), using all data")
        return sent_df
    
    # 각 비즈니스에서 균등하게 샘플링
    sampled_dfs = []
    for biz_id in sent_df['business_id'].unique():
        biz_data = sent_df[sent_df['business_id'] == biz_id]
        biz_sample_size = min(sample_size // len(sent_df['business_id'].unique()), len(biz_data))
        biz_sample = biz_data.sample(n=biz_sample_size, random_state=42)
        sampled_dfs.append(biz_sample)
        if verbose:
            print(f"[Quick Test] Business {biz_id}: sampled {len(biz_sample)}/{len(biz_data)} sentences")
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    if verbose:
        print(f"[Quick Test] Total sampled: {len(result)}/{len(sent_df)} sentences")
    return result


def main():
    args = parse_args()
    start_time = time.time()
    
    if args.verbose:
        print("🚀 Starting DeBERTa NLI Multi-Label Pipeline (CORE labels only, 10+ word sentences)")
        print(f"⚙️ Config: min_prob={args.min_prob}, batch_size={args.batch_size}, model={args.nli_model}")
        print("🎯 Using CORE labels only (33 essential restaurant aspects)")
        print("📏 Filtering: sentences with 10+ words only")
        if args.quick_test:
            print(f"⚡ Quick test mode: {args.sample_size} sentences per business")
    
    # 출력 디렉토리 설정
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        print(f"[DRY RUN] Would create output directory: {out_dir}")
    
    # 🎯 NLI-only 모드: 기존 sentiment 파일만 사용
    if args.nli_only:
        if not Path(PATH_SENT_WITH_SENT).exists():
            raise SystemExit(f"❌ NLI-only mode requires existing sentiment file: {PATH_SENT_WITH_SENT}")
        
        print(f"[NLI-Only Mode] Using existing sentiment file: {PATH_SENT_WITH_SENT}")
        sent_df = pd.read_csv(PATH_SENT_WITH_SENT)
        
        if args.quick_test:
            sent_df = sample_sentences_for_quick_test(sent_df, args.sample_size, args.verbose)
        
        if args.test_single_biz:
            first_biz = sent_df['business_id'].iloc[0]
            sent_df = sent_df[sent_df['business_id'] == first_biz]
            print(f"[Test Single Biz] Using only business: {first_biz}")
        
        # 비즈니스 메타데이터 로드
        biz_ids = sent_df['business_id'].unique().tolist()
        id2meta = load_business_meta(str(PATH_BUSINESS), biz_ids)
        
        # 바로 NLI 분석으로 점프
        jump_to_nli_analysis(args, sent_df, id2meta, out_dir)
        return

    # 1) 대상 business_id 결정
    if args.biz_id:
        biz_ids = args.biz_id
    else:
        biz_ids = find_business_ids(
            business_json_path=str(PATH_BUSINESS),
            name_substring=args.name_substr,
            category_keyword=args.category,
            city=args.city,
            state=args.state,
            limit=args.limit
        )
    
    if not biz_ids:
        raise SystemExit("❌ No business matched. Try different filters or supply --biz-id.")
    
    if args.test_single_biz:
        biz_ids = biz_ids[:1]
        print(f"[Test Single Biz] Using only: {biz_ids[0]}")
    
    print(f"[Picked business_id(s)] {biz_ids}")

    # 1.5) business.jsonl에서 주소/이름 메타 로드
    id2meta = load_business_meta(str(PATH_BUSINESS), biz_ids)
    
    if args.verbose:
        for bid in biz_ids:
            meta = id2meta.get(bid, {}) or {}
            print(f"  • {bid}: {meta.get('name', 'Unknown')} - {meta.get('address', 'Unknown')}")

    # 2) 리뷰 필터 → 통합 JSONL
    if not args.skip_filter and not args.dry_run:
        print("[Step 1/3] Filtering reviews...")
        kept = filter_reviews_by_business_ids(
            review_json_path=str(PATH_REVIEW),
            target_business_ids=biz_ids,
            out_jsonl_path=str(PATH_REVIEWS_FILTERED),
        )
        print(f"✅ Filtered reviews: kept={kept} → {PATH_REVIEWS_FILTERED}")
    elif args.dry_run:
        print(f"[DRY RUN] Would filter reviews for {len(biz_ids)} businesses")
    else:
        print("[Skip] Review filtering")

    # 3) 문장 단위 감성
    if not args.skip_sentiment and not args.dry_run:
        print("[Step 2/3] Sentence sentiment analysis...")
        run_sentence_sentiment(
            filtered_jsonl_path=str(PATH_REVIEWS_FILTERED),
            out_csv_path=str(PATH_SENT_WITH_SENT),
            sentiment_model=SENTIMENT_MODEL,
            max_chars=256,
            batch_size=64
        )
        print(f"✅ Sentence sentiment → {PATH_SENT_WITH_SENT}")
    elif args.dry_run:
        print(f"[DRY RUN] Would run sentiment analysis")
    else:
        print("[Skip] Sentiment analysis")

    # 4) NLI 분석
    if args.do_nli and not args.dry_run:
        sent_df = pd.read_csv(PATH_SENT_WITH_SENT)
        
        if args.quick_test:
            sent_df = sample_sentences_for_quick_test(sent_df, args.sample_size, args.verbose)
        
        jump_to_nli_analysis(args, sent_df, id2meta, out_dir)
    elif args.dry_run:
        print(f"[DRY RUN] Would run NLI analysis with {args.nli_scope} scope")
    else:
        print("[Skip] NLI analysis (use --do-nli to enable)")
    
    elapsed = time.time() - start_time
    print(f"🎉 DeBERTa NLI Pipeline completed in {elapsed:.1f} seconds")
    


def jump_to_nli_analysis(args, sent_df: pd.DataFrame, id2meta: dict, out_dir: Path):
    """NLI 분석 실행"""
    print("[Step 3/3] NLI Multi-Label Analysis (CORE labels, 10+ words)...")
    
    # 3.5) 업체별 문장/리뷰 수 집계
    sent_counts = sent_df.groupby("business_id").size().to_dict()
    
    review_counts = None
    if args.per_store_min_reviews > 0:
        if "review_id" in sent_df.columns:
            review_counts = (
                sent_df.drop_duplicates(["business_id", "review_id"])
                      .groupby("business_id").size().to_dict()
            )
        else:
            print("[Warn] per-store-min-reviews requested but review_id not found; skipping this filter.")

    # 주소 → 슬러그 매핑(+충돌 회피)
    seen_addr = {}
    def addr_slug_of(bid: str) -> str:
        meta = id2meta.get(bid, {}) or {}
        base = meta.get("address") or meta.get("name") or "store"
        s = slugify(base, max_len=60)
        cnt = seen_addr.get(s, 0) + 1
        seen_addr[s] = cnt
        return s if cnt == 1 else f"{s}-{cnt}"

    # 공통 러너
    def run_nli_analysis(scope: str, business_id, out_base_path: Path):
        if args.verbose:
            print(f"🔍 Running NLI analysis: scope={scope}, business_id={business_id}")
            
        start_time = time.time()
        
        result_df, summary_df = apply_nli_multilabel_for_business(
            sentences_csv_path=str(PATH_SENT_WITH_SENT) if not args.quick_test else None,  # None이면 sent_df 직접 사용
            out_csv_path=str(out_base_path),
            business_id=business_id,
            nli_model=args.nli_model,
            min_prob=args.min_prob,
            # label_mode 파라미터 제거됨 (CORE만 사용)
            batch_size=args.batch_size,
            local_files_only=args.local_only,
            business_meta=id2meta
        )
        
        elapsed = time.time() - start_time
        base = str(out_base_path)
        
        print(f"✅ NLI analysis complete: scope={scope}, business_id={business_id} ({elapsed:.1f}s)")
        print(f"   📊 Results: {len(summary_df)} aspects, {len(result_df)} sentences (10+ words)")
        print("   📁 Artifacts:")
        print(f"      • Sentence results: {base}")
        print(f"      • Aspect summary : {base.replace('.csv', '_topic_summary_with_labels.csv')}")
        print(f"      • Monthly trends : {base.replace('.csv', '_topic_trend_by_month.csv')}")
        print(f"      • Examples      : {base.replace('.csv', '_topic_examples.csv')}")
        
        if args.verbose:
            # 간단한 결과 미리보기
            noise_count = len(summary_df[summary_df['topic_id'] == -1])
            valid_aspects = len(summary_df[summary_df['topic_id'] != -1])
            print(f"   📈 Analysis preview: {valid_aspects} valid aspects, {noise_count} noise")
            
            if valid_aspects > 0:
                top_aspect = summary_df[summary_df['topic_id'] != -1].nlargest(1, 'n').iloc[0]
                print(f"   🏆 Top aspect: {top_aspect['label'][:50]}... ({top_aspect['n']} reviews)")
        
        return base, summary_df

    run_index = []
    biz_ids = sent_df['business_id'].unique().tolist()

    # (a) 통합(POOLED)
    if args.nli_scope in ("pooled", "both"):
        print(f"🌐 Running pooled analysis across {len(biz_ids)} businesses...")
        pooled_name = args.filename_template.format(scope="pooled", addr_slug="all-stores")
        pooled_path = out_dir / pooled_name
        
        base, summary_df = run_nli_analysis("pooled", None, pooled_path)
        
        run_index.append({
            "scope": "pooled",
            "business_id": "ALL",
            "business_address": "ALL",
            "n_sentences": int(len(sent_df)),
            "n_reviews": int(sent_df["review_id"].nunique()) if "review_id" in sent_df.columns else None,
            "n_aspects": len(summary_df[summary_df['topic_id'] != -1]),
            "status": "analyzed",
            "output_base": base
        })

    # (b) 업체별(임계치 필터 적용)
    if args.nli_scope in ("per-store", "both"):
        MIN_S = int(args.per_store_min_sentences)
        MIN_R = int(args.per_store_min_reviews)
        
        print(f"🏪 Running per-store analysis (min {MIN_S} sentences, {MIN_R} reviews)...")

        for bid in biz_ids:
            n_s = int(sent_counts.get(bid, 0))
            n_r = int(review_counts.get(bid, 0)) if (review_counts is not None) else None

            if n_s < MIN_S:
                print(f"⏭ Skip {bid}: sentences={n_s} < {MIN_S}")
                run_index.append({
                    "scope":"per-store","business_id":bid,
                    "business_address": (id2meta.get(bid, {}) or {}).get("address",""),
                    "n_sentences":n_s,"n_reviews":n_r,"n_aspects":0,
                    "status":"skipped_low_sentences","output_base":None
                })
                continue
                
            if (review_counts is not None) and (n_r < MIN_R):
                print(f"⏭ Skip {bid}: reviews={n_r} < {MIN_R}")
                run_index.append({
                    "scope":"per-store","business_id":bid,
                    "business_address": (id2meta.get(bid, {}) or {}).get("address",""),
                    "n_sentences":n_s,"n_reviews":n_r,"n_aspects":0,
                    "status":"skipped_low_reviews","output_base":None
                })
                continue

            addr_slug = addr_slug_of(bid)
            fname = args.filename_template.format(scope="perstore", addr_slug=addr_slug)
            out_path = out_dir / fname
            
            base, summary_df = run_nli_analysis("per-store", bid, out_path)
            
            run_index.append({
                "scope":"per-store","business_id":bid,
                "business_address": (id2meta.get(bid, {}) or {}).get("address",""),
                "n_sentences":n_s,"n_reviews":n_r,
                "n_aspects": len(summary_df[summary_df['topic_id'] != -1]),
                "status":"analyzed","output_base":base
            })

    # 5) 실행 인덱스 저장
    if args.write_index and run_index:
        idx_path = out_dir / "nli_run_index.csv"
        index_df = pd.DataFrame(run_index)
        index_df.to_csv(idx_path, index=False, encoding="utf-8")
        print(f"📋 Run index saved → {idx_path}")
        
        if args.verbose:
            print("📊 Run Summary:")
            print(index_df.groupby(['scope', 'status']).size().to_string())


if __name__ == "__main__":
    main()

