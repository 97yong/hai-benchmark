# -*- coding: utf-8 -*-
"""Issue 본문의 제출 JSON 을 검증해 docs/data.json 에 병합한다."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "docs", "data.json")
CONTENT_ID = "744e1d3612d7c8ed"
SPLITS = {"hold_A", "hold_B", "hold_C", "time_82"}
NEED = ("이름", "분할", "F1", "FNR", "FPR", "합")
KEEP = ("이름", "모델", "입력", "파라미터", "분할", "F1", "FNR", "FPR", "합",
        "n_gid", "n_정상", "n_고장", "content_id", "note", "상세", "배지")
DKEY = ("사전학습", "전처리", "모델 구조", "손실 함수", "Optimization", "Epoch", "Batch", "LR")
BADGE = ("사전학습", "Unsupervised")


def fail(msg):
    print(f"::error::{msg}")
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as f:
        f.write(f"ok=false\nmsg={msg}\n")
    sys.exit(0)                      # 실패해도 워크플로는 성공 처리 — 안내 댓글만 단다


def main():
    body = os.environ.get("ISSUE_BODY", "")
    who = os.environ.get("ISSUE_USER", "")
    m = re.search(r"```json\s*(.+?)```", body, re.S)
    if not m:
        fail("제출 내용을 찾지 못했습니다. submit.py 로 다시 올려주세요.")
    try:
        rows = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        fail(f"JSON 을 읽지 못했습니다: {e}")
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        fail("제출 항목이 비어 있습니다.")

    clean = []
    for r in rows:
        if not isinstance(r, dict):
            fail("제출 형식이 올바르지 않습니다.")
        for k in NEED:
            if k not in r:
                fail(f"필수 항목이 없습니다: {k}")
        if r["분할"] not in SPLITS:
            fail(f"알 수 없는 시나리오: {r['분할']}")
        if r.get("content_id") and r["content_id"] != CONTENT_ID:
            fail(f"데이터 캐시가 다릅니다 ({r['content_id']} ≠ {CONTENT_ID}).")
        for k in ("F1", "FNR", "FPR", "합"):
            v = r.get(k)
            if not isinstance(v, (int, float)) or not (0 <= v <= 2):
                fail(f"{k} 값이 올바르지 않습니다: {v}")
        e = {k: r[k] for k in KEEP if k in r}
        dt = e.get("상세")
        e["상세"] = {k: str(dt[k])[:200] for k in DKEY
                     if isinstance(dt, dict) and dt.get(k)} if dt else {}
        e["배지"] = [b for b in BADGE if isinstance(e.get("배지"), list) and b in e["배지"]]
        e["계정"] = who
        clean.append(e)

    old = []
    if os.path.exists(DATA):
        try:
            old = json.load(open(DATA, encoding="utf-8"))
        except Exception:
            old = []
    # 한 사람이 모델을 여러 개 올릴 수 있다 — 같은 (이름, 모델) 이면 덮어쓴다
    ident = lambda x: (x.get("이름"), x.get("모델") or "", x.get("입력") or "", x.get("분할"))
    owner = {x.get("이름"): x.get("계정") for x in old
             if isinstance(x, dict) and x.get("계정")}
    keep = {ident(x): x for x in old if isinstance(x, dict)}
    for e in clean:
        if owner.get(e["이름"], who) != who:
            fail(f"'{e['이름']}' 은 이미 다른 계정({owner[e['이름']]})이 쓰고 있습니다. 다른 이름을 써주세요.")
        keep[ident(e)] = e
    merged = sorted(keep.values(), key=lambda x: (x.get("분할", ""), x.get("합", 9)))
    if len({ident(x)[:3] for x in merged}) > 60:
        fail("등록 한도를 넘었습니다.")
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    json.dump(merged, open(DATA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    names = ", ".join(sorted({e["분할"] for e in clean}))
    msg = f"{clean[0]['이름']} · {names} · {len(clean)}건 반영했습니다."
    print(msg)
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as f:
        f.write(f"ok=true\nmsg={msg}\n")


if __name__ == "__main__":
    main()
