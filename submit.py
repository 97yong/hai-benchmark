#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OHT 진단 벤치마크 — 채점 + 리더보드 등록 (이 파일 하나면 됩니다)

    python3 submit.py 홍길동 "CNN2D" pred_hold_C.csv --input STFT --ckpt model.pt
    python3 submit.py 홍길동 "ResNet18" pred_hold_C.csv --dry      # 채점만, 등록 안 함

예측 CSV 형식 — 열 두 개면 됩니다.
    row,pred          row = oht_loader 행 번호 · pred = 0 정상 / 1 축계 / 2 기어계
    12,1
    13,1
분할은 행 번호로 자동 판별합니다(파일명 규칙 없음).

필요한 것: oht_loader(v2) 패키지 + 진동 캐시 폴더. 둘 다 각자 로컬에 있으면 됩니다.
    --loader /path/to/oht_loader_v2      (기본: OHT_LOADER 환경변수 또는 자동 탐색)
    --cache  /path/to/_cache             (기본: OHT_CACHE 환경변수 또는 자동 탐색)

등록: 본인 GitHub 계정이면 됩니다. 별도 권한을 받을 필요가 없습니다.
    1) gh CLI 로그인 상태        — `gh auth login` 한 번이면 끝(권장)
    2) OHT_TOKEN 환경변수        — public_repo 권한 토큰
    3) --token <토큰>
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, urllib.request, urllib.error

REPO = "97yong/hai-benchmark"
BOARD_URL = "https://97yong.github.io/hai-benchmark/"
SPLITS = ["hold_C", "hold_A", "hold_B", "time_82"]
CLASSES = ["정상", "축계", "기어계"]
MAP3 = {"정상": 0, "축하중 증가": 1, "클램프 풀림": 1, "윤활 불량": 2, "기어 치 파손": 2}
NORMAL, N_BOOT, W_FNR = 0, 1000, 2.0


# ----------------------------------------------------------------- 환경
def find_dir(cli, env, names):
    for c in [cli, os.environ.get(env)]:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    here = os.path.abspath(os.path.dirname(__file__))
    for base in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here)), os.getcwd()]:
        for n in names:
            p = os.path.join(base, n)
            if os.path.isdir(p):
                return p
    return None


def setup(args):
    ld = find_dir(args.loader, "OHT_LOADER", ["oht_loader_v2", "oht_loader"])
    if not ld:
        sys.exit("oht_loader 를 찾지 못했습니다. --loader <경로> 또는 OHT_LOADER 환경변수로 알려주세요.")
    sys.path.insert(0, ld)
    try:
        from oht_loader import OHTData, load_manifest
    except ImportError as e:
        sys.exit(f"oht_loader 를 불러오지 못했습니다 ({ld}): {e}")
    ch = find_dir(args.cache, "OHT_CACHE", ["data", "_cache"])
    if not ch:
        sys.exit("진동 캐시 폴더를 찾지 못했습니다. --cache <경로> 또는 OHT_CACHE 환경변수로 알려주세요.")
    try:
        d = OHTData(ch)
    except Exception as e:
        sys.exit(f"캐시를 열지 못했습니다 ({ch}): {e}")
    sp_dir = os.path.join(ld, "splits")
    if not os.path.isdir(sp_dir):
        sys.exit(f"분할 manifest 폴더가 없습니다: {sp_dir}")
    return d, {s: load_manifest(d, os.path.join(sp_dir, f"{s}.json")) for s in SPLITS}


# ----------------------------------------------------------------- 채점
def metrics(yt, yp):
    import numpy as np
    from sklearn.metrics import f1_score
    yt, yp = np.asarray(yt), np.asarray(yp)
    f = yt != NORMAL
    fnr = float((yp[f] == NORMAL).mean()) if f.any() else float("nan")
    fpr = float((yp[~f] != NORMAL).mean()) if (~f).any() else float("nan")
    return dict(F1=float(f1_score(yt, yp, average="macro")), FNR=fnr, FPR=fpr,
                **{"합": fnr + fpr})


def score(d, sp, pred_by_row):
    """창 예측 -> gid 다수결 -> 지표 + 주행 부트스트랩 95% 구간."""
    import numpy as np, pandas as pd
    rows = sp.test
    miss = [int(r) for r in rows if int(r) not in pred_by_row]
    if miss:
        raise ValueError(f"예측 누락 {len(miss)}행 (예: {miss[:5]})")
    p = np.array([pred_by_row[int(r)] for r in rows], dtype=int)
    key = (pd.Series(d.session[rows].astype(str)) + "#" + pd.Series(d.gid[rows]).astype(str)).values
    y = np.array([MAP3[v] for v in d.label[rows].astype(str)])
    g = pd.DataFrame(dict(k=key, t=y, p=p)).groupby("k").agg(
        t=("t", "first"), p=("p", lambda s: s.value_counts().idxmax()))
    yt, yp = g.t.values, g.p.values
    m = metrics(yt, yp)
    rng = np.random.RandomState(0)
    acc = {k: [] for k in m}
    for _ in range(N_BOOT):
        i = rng.randint(0, len(yt), len(yt))
        if len(np.unique(yt[i])) < 2:
            continue
        b = metrics(yt[i], yp[i])
        for k in acc:
            acc[k].append(b[k])
    for k, v in acc.items():
        m[f"{k}_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    f = yt != NORMAL
    m.update(n_gid=int(len(yt)), n_정상=int((~f).sum()), n_고장=int(f.sum()),
             주지표=W_FNR * m["FNR"] + m["FPR"])
    return m


def read_pred(path):
    import pandas as pd
    df = pd.read_csv(path)
    if not {"row", "pred"} <= set(df.columns):
        raise ValueError(f"{path}: 'row','pred' 열이 필요합니다 (현재 {list(df.columns)})")
    bad = sorted(set(df.pred.unique()) - {0, 1, 2})
    if bad:
        raise ValueError(f"{path}: pred 는 0/1/2 만 허용 (발견: {bad})")
    return dict(zip(df.row.astype(int), df.pred.astype(int)))


def which_split(pred_rows, mans):
    """행 번호 집합으로 분할을 판별한다 — 파일명 규칙 불필요."""
    for name, sp in mans.items():
        need = set(int(r) for r in sp.test)
        if need <= pred_rows:
            return name
    hits = {n: len(set(int(r) for r in sp.test) & pred_rows) / len(sp.test)
            for n, sp in mans.items()}
    best = max(hits, key=hits.get)
    raise ValueError("어느 분할인지 판별하지 못했습니다. 행이 덜 들어있습니다 — "
                     + " · ".join(f"{n} {v:.0%}" for n, v in sorted(hits.items(), key=lambda x: -x[1]))
                     + f"\n  (가장 가까운 {best} 기준으로 {len(mans[best].test)}행이 모두 필요합니다)")


# ----------------------------------------------------------------- 파라미터 세기
SKIP = ("running_mean", "running_var", "num_batches_tracked")


def count_params(path):
    """체크포인트에서 학습 파라미터 수를 센다. BatchNorm 통계는 빼고 센다.
    PyTorch .pt/.pth 만 지원 — 다른 프레임워크면 --params 로 직접 적어주세요."""
    try:
        import torch
    except ImportError:
        print("  ※ torch 가 없어 파라미터를 세지 못했습니다. --params 로 적어주세요.")
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ※ {path} 를 열지 못했습니다 ({e}). --params 로 적어주세요.")
        return None
    if hasattr(obj, "parameters"):                       # 모듈을 통째로 저장한 경우
        return sum(p.numel() for p in obj.parameters())
    sd = obj
    if isinstance(obj, dict):
        for k in ("state", "state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]; break
    if not isinstance(sd, dict):
        print(f"  ※ {path} 에서 가중치를 찾지 못했습니다. --params 로 적어주세요.")
        return None
    n = 0
    for k, v in sd.items():
        if any(k.endswith(x) for x in SKIP):
            continue
        if hasattr(v, "numel"):
            n += v.numel()
    return n or None


def human(n):
    if n is None: return None
    return (f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.2f}M" if n >= 1e6
            else f"{n/1e3:.0f}K" if n >= 1e3 else str(n))


# ----------------------------------------------------------------- 등록
def api(url, token, method="GET", body=None):
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body else None)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if body:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get_token(cli):
    if cli:
        return cli
    if os.environ.get("OHT_TOKEN"):
        return os.environ["OHT_TOKEN"]
    try:
        t = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        if t.returncode == 0 and t.stdout.strip():
            return t.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def register(rows, token):
    """Issue 를 열면 Actions 가 검증 후 리더보드에 반영한다.
    공개 저장소라 GitHub 계정만 있으면 되고, 따로 권한을 받을 필요가 없다."""
    who = rows[0]["이름"]
    names = ", ".join(sorted({r["분할"] for r in rows}))
    body = ("아래 내용은 `submit.py` 가 만든 것입니다. 직접 고치지 마세요.\n\n"
            "```json\n" + json.dumps(rows, ensure_ascii=False, indent=1) + "\n```")
    r = api(f"https://api.github.com/repos/{REPO}/issues", token, "POST",
            {"title": f"[submit] {who} · {names}", "body": body})
    return r.get("html_url", "")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="OHT 진단 벤치마크 채점 · 등록")
    ap.add_argument("name", help="제출자 이름 (리더보드 표시명)")
    ap.add_argument("model", help='모델 정보. 예: "CNN2D w48 · multires+resf"')
    ap.add_argument("pred", nargs="+", help="예측 CSV (row,pred). 여러 개 가능")
    ap.add_argument("--input", default=None,
                    help='입력 표현. 예: STFT, 웨이블릿, 원파형')
    ap.add_argument("--ckpt", default=None,
                    help='체크포인트 경로. 주면 파라미터 수를 세어 넣는다 (PyTorch .pt/.pth)')
    ap.add_argument("--params", default=None,
                    help='파라미터 수를 직접 적을 때. 예: 3.6M (--ckpt 보다 우선)')
    ap.add_argument("--note", default="", help="추가 메모 (선택)")

    g = ap.add_argument_group("상세 정보 (선택) — 리더보드에서 행을 누르면 펼쳐집니다")
    g.add_argument("--pretrain", default=None,
                   help='사전학습. 없으면 생략, 있으면 내용을 적습니다. 예: "시뮬레이션 200k창"')
    g.add_argument("--preprocess", default=None,
                   help='전처리. 예: "창별 RMS 정규화 · STFT n256/h128"')
    g.add_argument("--arch", default=None, help='모델 구조. 예: "CNN2D 5블록 · width 48"')
    g.add_argument("--loss", default=None, help='손실 함수. 예: "CrossEntropy (label smoothing 0.1)"')
    g.add_argument("--optim", default=None, help='Optimization. 예: "AdamW wd 1e-4 · OneCycle"')
    g.add_argument("--epochs", default=None, help="Epoch. 예: 8")
    g.add_argument("--batch", default=None, help="Batch. 예: 128")
    g.add_argument("--lr", default=None, help="LR. 예: 3e-4")
    g.add_argument("--unsup", action="store_true",
                   help="타깃 라벨을 쓰지 않았다면 켭니다 (Unsupervised 배지)")
    ap.add_argument("--loader", default=None), ap.add_argument("--cache", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--dry", action="store_true", help="채점만 하고 등록하지 않음")
    a = ap.parse_args()

    detail = {k: str(v).strip() for k, v in (
        ("사전학습", a.pretrain), ("전처리", a.preprocess), ("모델 구조", a.arch),
        ("손실 함수", a.loss), ("Optimization", a.optim),
        ("Epoch", a.epochs), ("Batch", a.batch), ("LR", a.lr)) if v not in (None, "")}
    badge = []
    pre = detail.get("사전학습", "")
    if pre and pre.strip() not in ("-", "—") and not re.match(
            r"\s*(무|없음|없다|안\s*함|x|n/?a|none|no)\b", pre, re.I):
        badge.append("사전학습")
    if a.unsup:
        badge.append("Unsupervised")

    d, mans = setup(a)
    print(f"캐시 {d.content_id[:16]} · {d.n:,}창\n")
    params = a.params
    if params is None and a.ckpt:
        n = count_params(a.ckpt)
        if n:
            params = human(n)
            print(f"  파라미터 {n:,} ({params}) — {os.path.basename(a.ckpt)}\n")
    files = [p for g in a.pred for p in sorted(glob.glob(g))] or a.pred
    out = []
    for path in files:
        try:
            pb = read_pred(path)
            name = which_split(set(pb), mans)
            m = score(d, mans[name], pb)
        except Exception as e:
            print(f"  ✗ {os.path.basename(path)}: {e}\n")
            continue
        m.update({"이름": a.name, "모델": a.model, "입력": a.input, "파라미터": params, "분할": name,
                  "note": a.note, "content_id": d.content_id[:16],
                  "상세": detail, "배지": badge})
        out.append(m)
        ci = m["합_ci"]
        print(f"  {name:8s}  F1 {m['F1']:.3f}   FNR {m['FNR']:.3f}   FPR {m['FPR']:.3f}   "
              f"합 {m['합']:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]   ({m['n_gid']:,} gid)")
    if not out:
        sys.exit("\n등록할 결과가 없습니다.")
    if a.dry:
        print("\n--dry — 등록하지 않았습니다.")
        return
    token = get_token(a.token)
    if not token:
        sys.exit("\nGitHub 로그인이 필요합니다. `gh auth login` 을 한 번 실행하거나 "
                 "OHT_TOKEN 환경변수 또는 --token 으로 토큰을 주세요.")
    try:
        url = register(out, token)
    except urllib.error.HTTPError as e:
        sys.exit(f"\n등록 실패 (HTTP {e.code}): {e.read().decode()[:200]}")
    print(f"\n{len(out)}건 올렸습니다. 1분쯤 뒤 리더보드에 반영됩니다.\n{BOARD_URL}")
    if url:
        print(f"진행 상황: {url}")


if __name__ == "__main__":
    main()
