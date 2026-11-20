#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous 5m factor research engine. Research only; never places orders."""
from __future__ import annotations
import argparse, glob, hashlib, json, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"open_time","open","high","low","close","volume"}
FEATURES = ["ret1","ret3","ret6","body_pct","range_pct","upper_wick","lower_wick",
            "volume_z","quote_volume_z","trade_count_z","taker_buy_ratio",
            "buy_imbalance","atr_pct","range_atr","close_location",
            "dist_high20","dist_low20","volatility20","hour_sin","hour_cos"]

def load_data(paths):
    frames=[]
    for path in paths:
        d=pd.read_csv(path)
        if not REQUIRED.issubset(d.columns):
            print(f"[SKIP] {path}: missing columns")
            continue
        d=d.copy()
        d["source_file"]=Path(path).name
        frames.append(d)
    if not frames: raise RuntimeError("No valid OHLCV CSV files found.")
    x=pd.concat(frames,ignore_index=True)
    x=x.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["open","high","low","close","volume"])
    return x.reset_index(drop=True)

def build_features(d):
    x=d.copy()
    c=x.close; h=x.high; l=x.low; o=x.open
    prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14,min_periods=14).mean()
    x["ret1"]=c.pct_change(1)
    x["ret3"]=c.pct_change(3)
    x["ret6"]=c.pct_change(6)
    x["body_pct"]=(c-o)/o
    x["range_pct"]=(h-l)/c
    span=(h-l).replace(0,np.nan)
    x["upper_wick"]=(h-np.maximum(o,c))/span
    x["lower_wick"]=(np.minimum(o,c)-l)/span
    def z(v,n=48):
        m=v.rolling(n,min_periods=n).mean()
        s=v.rolling(n,min_periods=n).std().replace(0,np.nan)
        return (v-m)/s
    x["volume_z"]=z(x.volume)
    if "quote_volume" in x: x["quote_volume_z"]=z(pd.to_numeric(x.quote_volume,errors="coerce"))
    else: x["quote_volume_z"]=np.nan
    if "count" in x: x["trade_count_z"]=z(pd.to_numeric(x["count"],errors="coerce"))
    else: x["trade_count_z"]=np.nan
    if "taker_buy_volume" in x:
        x["taker_buy_ratio"]=pd.to_numeric(x.taker_buy_volume,errors="coerce")/x.volume.replace(0,np.nan)
        x["buy_imbalance"]=2*x.taker_buy_ratio-1
    else:
        x["taker_buy_ratio"]=np.nan; x["buy_imbalance"]=np.nan
    x["atr_pct"]=atr/c
    x["range_atr"]=(h-l)/atr
    x["close_location"]=(c-l)/span
    hh=h.rolling(20,min_periods=20).max()
    ll=l.rolling(20,min_periods=20).min()
    x["dist_high20"]=c/hh-1
    x["dist_low20"]=c/ll-1
    x["volatility20"]=x.ret1.rolling(20,min_periods=20).std()
    dt=pd.to_datetime(x.open_time,unit="ms",utc=True)
    hour=dt.dt.hour+dt.dt.minute/60
    x["hour_sin"]=np.sin(2*np.pi*hour/24)
    x["hour_cos"]=np.cos(2*np.pi*hour/24)
    x[FEATURES]=x[FEATURES].replace([np.inf,-np.inf],np.nan)
    return x

def make_outcomes(x, horizon=48, rr=2.0, fee_bps=0.0):
    """Signal at bar i close; enter at i+1 open. SL wins same-bar TP/SL ties."""
    n=len(x); out={}
    op=x.open.to_numpy(float); hi=x.high.to_numpy(float); lo=x.low.to_numpy(float)
    cl=x.close.to_numpy(float); atr=(x.atr_pct.to_numpy(float)*cl)
    for side in (1,-1):
        r=np.full(n,np.nan); exit_reason=np.full(n,"",dtype=object)
        for i in range(n-1):
            entry=op[i+1]; risk=atr[i]
            if not np.isfinite(entry) or not np.isfinite(risk) or risk<=0: continue
            stop=entry-side*risk; target=entry+side*rr*risk
            end=min(n,i+1+horizon)
            done=False
            for j in range(i+1,end):
                sl=(lo[j]<=stop) if side==1 else (hi[j]>=stop)
                tp=(hi[j]>=target) if side==1 else (lo[j]<=target)
                if sl:
                    r[i]=-1.0; exit_reason[i]="SL"; done=True; break
                if tp:
                    r[i]=rr; exit_reason[i]="TP"; done=True; break
            if not done and end>i+1:
                raw=side*(cl[end-1]-entry)/risk
                r[i]=raw; exit_reason[i]="TIME"
            if np.isfinite(r[i]): r[i]-=fee_bps/10000*2*entry/risk
        out[side]=(r,exit_reason)
    return out

def evaluate(mask, outcomes, train_end, min_train=30, min_test=15):
    results={}
    for side,label in [(1,"LONG"),(-1,"SHORT")]:
        r,reason=outcomes[side]
        valid=mask & np.isfinite(r)
        # Purge forward-label horizon at the boundary to prevent train/test leakage.
        idx=np.arange(len(mask))
        tr=valid & (idx < max(0, train_end - 48))
        te=valid & (idx >= train_end)
        a=r[tr]; b=r[te]
        if len(a)<min_train or len(b)<min_test: continue
        results[label]={"train_n":int(len(a)),"test_n":int(len(b)),
          "train_win":float(np.mean(a>0)),"test_win":float(np.mean(b>0)),
          "train_exp_R":float(np.mean(a)),"test_exp_R":float(np.mean(b)),
          "train_profit_factor":float(np.sum(a[a>0])/max(1e-12,-np.sum(a[a<0]))),
          "test_profit_factor":float(np.sum(b[b>0])/max(1e-12,-np.sum(b[b<0]))),
          "test_avg_R":float(np.mean(b))}
    return results

def build_candidates(x, train_end, top_k=24):
    candidates=[]
    for f in FEATURES:
        a=x[f].iloc[:train_end].dropna()
        if len(a)<100: continue
        qs=np.unique(a.quantile([.10,.20,.30,.40,.50,.60,.70,.80,.90]).to_numpy())
        for q in qs:
            for op in ("<=",">="):
                candidates.append({"conditions":[{"feature":f,"op":op,"value":float(q)}]})
    return candidates

def mask_for(x, candidate):
    m=np.ones(len(x),dtype=bool)
    for c in candidate["conditions"]:
        v=x[c["feature"]].to_numpy(float)
        m &= np.isfinite(v)
        m &= (v<=c["value"]) if c["op"]=="<=" else (v>=c["value"])
    return m

def score_candidate(x,cand,outcomes,train_end):
    m=mask_for(x,cand)
    ev=evaluate(m,outcomes,train_end)
    rows=[]
    for side,v in ev.items():
        # rank on train expectancy; test remains an untouched report card
        rows.append((v["train_exp_R"],v["test_exp_R"],v["train_n"],side,v))
    if not rows: return None
    rows.sort(reverse=True)
    train_exp,test_exp,n,side,v=rows[0]
    return {"candidate":cand,"side":side,"metrics":v,
            "rank_score":train_exp,"test_score":test_exp}

def make_id(obj):
    return hashlib.sha1(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:12]

def failure_report(x,cand,side,outcomes,train_end):
    m=mask_for(x,cand) & (np.arange(len(x))>=train_end)
    r,reason=outcomes[1 if side=="LONG" else -1]
    idx=np.flatnonzero(m & np.isfinite(r))
    if len(idx)<15: return {"note":"樣本外交易數不足，無法可靠分析虧損特徵"}
    losses=idx[r[idx]<=0]; wins=idx[r[idx]>0]
    diffs=[]
    for f in FEATURES:
        a=x[f].to_numpy(float)[losses]; b=x[f].to_numpy(float)[wins]
        a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
        if len(a)<5 or len(b)<5: continue
        pooled=np.nanstd(np.r_[a,b])
        if pooled==0: continue
        effect=(np.nanmedian(a)-np.nanmedian(b))/pooled
        diffs.append({"feature":f,"loss_median":float(np.nanmedian(a)),
                      "win_median":float(np.nanmedian(b)),"standardized_gap":float(effect)})
    diffs.sort(key=lambda z:abs(z["standardized_gap"]),reverse=True)
    return {"oos_trades":int(len(idx)),"oos_losses":int(len(losses)),
            "oos_wins":int(len(wins)),"loss_features":diffs[:8],
            "warning":"虧損特徵是診斷線索，不代表可直接用來過濾；新過濾條件必須重新做時間切分驗證。"}

def persist(db_path, rows, data_info):
    con=sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS experiments(
      id TEXT PRIMARY KEY, created_at TEXT, data_info TEXT, candidate TEXT,
      side TEXT, metrics TEXT, failure_report TEXT)""")
    now=pd.Timestamp.utcnow().isoformat()
    for row in rows:
        con.execute("INSERT OR REPLACE INTO experiments VALUES(?,?,?,?,?,?,?)",
          (row["id"],now,json.dumps(data_info),json.dumps(row["candidate"]),
           row["side"],json.dumps(row["metrics"]),json.dumps(row["failure_report"],ensure_ascii=False)))
    con.commit(); con.close()

def main():
    p=argparse.ArgumentParser(description="Autonomous market factor research (no live trading)")
    p.add_argument("--data",default=".",help="Folder containing OHLCV CSV files")
    p.add_argument("--pattern",default="*-5m-*.csv")
    p.add_argument("--db",default="research_memory.sqlite")
    p.add_argument("--rr",type=float,default=2.0)
    p.add_argument("--horizon",type=int,default=48)
    p.add_argument("--fee-bps",type=float,default=0.0,help="Round-trip fee in basis points; 0 means gross results")
    p.add_argument("--top",type=int,default=24)
    p.add_argument("--rounds",type=int,default=3)
    args=p.parse_args()
    paths=sorted(glob.glob(str(Path(args.data)/"**"/args.pattern),recursive=True))
    if not paths: raise SystemExit(f"No files matched: {Path(args.data)/args.pattern}")
    print(f"[1/6] Loading {len(paths)} CSV files")
    raw=load_data(paths); x=build_features(raw)
    valid=x[FEATURES].notna().any(axis=1)
    first=int(np.argmax(valid.to_numpy()))
    x=x.iloc[first:].reset_index(drop=True)
    split=int(len(x)*0.70)
    if split<300 or len(x)-split<100: raise SystemExit("Not enough usable bars; need a longer history.")
    print(f"[2/6] Bars={len(x):,}; train={split:,}; OOS={len(x)-split:,}")
    outcomes=make_outcomes(x,args.horizon,args.rr,args.fee_bps)
    print("[3/6] Generating factor hypotheses")
    base=build_candidates(x,split)
    scored=[]
    for i,c in enumerate(base):
        s=score_candidate(x,c,outcomes,split)
        if s: scored.append(s)
    scored.sort(key=lambda z:z["rank_score"],reverse=True)
    selected=scored[:args.top]
    print(f"  single-factor candidates={len(scored):,}; retained={len(selected)}")
    all_rows=[]
    for round_no in range(1,args.rounds+1):
        print(f"[4/6] Evolution round {round_no}/{args.rounds}")
        pool=selected if round_no==1 else selected[:min(len(selected),args.top)]
        new=[]
        if round_no==1: new=pool
        else:
            # Cross high-ranking, distinct features; each condition uses train-only thresholds.
            for i,a in enumerate(pool):
                fa=a["candidate"]["conditions"][0]["feature"]
                for b in pool[i+1:]:
                    fb=b["candidate"]["conditions"][0]["feature"]
                    if fa==fb: continue
                    cand={"conditions":a["candidate"]["conditions"]+b["candidate"]["conditions"]}
                    new.append(score_candidate(x,cand,outcomes,split))
        new=[z for z in new if z]
        new.sort(key=lambda z:z["rank_score"],reverse=True)
        # retain best distinct hypotheses for the next evolutionary round
        seen=set()
        for z in new:
            key=make_id(z["candidate"])+z["side"]
            if key in seen: continue
            seen.add(key)
            z["id"]=key
            z["failure_report"]=failure_report(x,z["candidate"],z["side"],outcomes,split)
            all_rows.append(z)
        selected=[z for z in new if len(z["candidate"]["conditions"])<4][:args.top]
        print(f"  evaluated={len(new):,}; saved={len(seen):,}")
    persist(args.db,all_rows,{"files":[Path(v).name for v in paths],
      "bars":len(x),"split_index":split,"rr":args.rr,"horizon":args.horizon,
      "fee_bps_round_trip":args.fee_bps,"signal":"bar close; entry next bar open",
      "overlapping_signals":"each signal is an independent trade"})
    flat=[]
    for z in all_rows:
        m=z["metrics"]
        flat.append({"id":z["id"],"side":z["side"],"conditions":json.dumps(z["candidate"]["conditions"],ensure_ascii=False),
          "train_n":m["train_n"],"test_n":m["test_n"],"train_win":m["train_win"],
          "test_win":m["test_win"],"train_exp_R":m["train_exp_R"],
          "test_exp_R":m["test_exp_R"],"test_pf":m["test_profit_factor"]})
    out=pd.DataFrame(flat).sort_values("test_exp_R",ascending=False)
    out.to_csv("research_leaderboard.csv",index=False,encoding="utf-8-sig")
    print("[5/6] Top out-of-sample candidates")
    print(out.head(20).to_string(index=False) if len(out) else "No candidates met minimum sample counts.")
    print("[6/6] Complete. Results saved to",args.db,"and research_leaderboard.csv")
    print("WARNING: this is a research prototype, not evidence of a profitable strategy. Validate on a later untouched period and include realistic fees/slippage before any live use.")

if __name__=="__main__":
    main()
