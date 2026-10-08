#!/usr/bin/env python3
"""Bounded, resumable Blender corpus generation with disjoint index ranges."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GENERATOR = Path(__file__).with_name("generate.py")
VALIDATE = Path(__file__).with_name("validate_corpus.py")
QA = Path(__file__).with_name("qa.py")
BLENDER = REPO / ".local/recognition/blender_synthetic_tooling/blender-4.2.3-linux-x64/blender"
BYPYTHON = REPO / ".local/recognition/blender_synthetic_tooling/blender-4.2.3-linux-x64/4.2/python/bin/python3.11"

def missing_indices(out: Path, start: int, end: int) -> list[int]:
    return [i for i in range(start, end)
            if not (out/"records"/f"synthetic_{i:06d}.json").is_file()
            or not (out/"images"/f"synthetic_{i:06d}.png").is_file()]

def run_chunk(out: Path, cfg: Path, seed: int, start: int, end: int,
              logdir: Path, threads: int, attempts: int) -> dict:
    logpath = logdir / f"chunk_{start:06d}_{end:06d}.log"
    command = [str(BLENDER), "-b", "-t", str(threads),
               "--python", str(GENERATOR), "--", "--config", str(cfg),
               "--output", str(out), "--seed", str(seed),
               "--start-index", str(start), "--count", str(end-start),
               "--resume", "--skip-assemble"]
    errors = []
    for retry in range(attempts):
        with logpath.open("a", encoding="utf-8") as log:
            log.write(f"START retry={retry} time={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            log.flush()
            rc = subprocess.run(command, cwd=REPO, stdout=log,
                                stderr=subprocess.STDOUT, check=False).returncode
        remaining = missing_indices(out, start, end)
        if not remaining:
            return {"start": start, "end": end, "ok": True, "tries": retry+1}
        errors.append({"try": retry+1, "exit_code": rc, "missing": remaining[:20],
                       "missing_count": len(remaining)})
        time.sleep(2)
    return {"start": start, "end": end, "ok": False, "errors": errors}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=GENERATOR.with_name("config.production.json"))
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=760601)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=125)
    ap.add_argument("--render-threads", type=int, default=2)
    ap.add_argument("--chunk-attempts", type=int, default=3)
    ap.add_argument("--no-qa", action="store_true")
    args = ap.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.count < 1:
        ap.error("workers, chunk-size and count must all be positive")
    out = args.output.resolve()
    cfg = args.config.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out/"parallel_logs"
    logs.mkdir(exist_ok=True)
    lock = (out/"run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another corpus job holds the run.lock; refusing duplicate work", flush=True)
        return 2

    end = args.start_index + args.count
    tasks = [(s, min(s+args.chunk_size, end))
             for s in range(args.start_index, end, args.chunk_size)]
    pending = [(s, e) for s, e in tasks if missing_indices(out,s,e)]
    print(json.dumps({"event":"begin","workers":args.workers,"chunk_size":args.chunk_size,
                      "total":args.count,"pending_chunks":len(pending),
                      "reused_chunks":len(tasks)-len(pending)}), flush=True)
    failed = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_chunk, out, cfg, args.seed, s, e, logs,
                            args.render_threads, args.chunk_attempts):(s,e)
                for s,e in pending}
        for number, future in enumerate(futures.as_completed(jobs),1):
            result = future.result()
            if not result["ok"]:
                failed.append(result)
            if number % 4 == 0 or not result["ok"] or number==len(pending):
                missing = len(missing_indices(out,args.start_index,end))
                status = {"event":"progress","chunks_finished":number,
                          "pending_chunks":len(pending),"images_ready":args.count-missing,
                          "total":args.count,"failed_chunks":len(failed)}
                (out/"parallel_status.json").write_text(json.dumps(status,indent=2)+"\n")
                print(json.dumps(status),flush=True)
    remaining = missing_indices(out,args.start_index,end)
    if remaining:
        (out/"parallel_failed.json").write_text(json.dumps({
            "missing_indices":remaining,"chunk_failures":failed},indent=2)+"\n")
        print(json.dumps({"event":"incomplete","remaining":len(remaining),
                          "failed_report":str(out/"parallel_failed.json")}),flush=True)
        return 1
    (out/"parallel_failed.json").unlink(missing_ok=True)
    print(json.dumps({"event":"generated","images":args.count}),flush=True)
    if args.no_qa:
        return 0
    if args.start_index or args.count!=10000:
        print("QA requested on incomplete corpus range; refusing",flush=True)
        return 2
    commands = [
      ("assemble",[str(BLENDER),"-b","--python",str(GENERATOR),"--",
                   "--config",str(cfg),"--output",str(out),"--assemble-only"]),
      ("validate",[str(BYPYTHON),str(VALIDATE),str(out),"--minimum-images",str(args.count)]),
      ("qa",[str(BYPYTHON),str(QA),"--root",str(out),"--min-images",str(args.count)])
    ]
    for name, cmd in commands:
        print(json.dumps({"event":"stage_start","stage":name}),flush=True)
        with (out/f"parallel_{name}.log").open("w") as log:
            result = subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print(json.dumps({"event":"stage_failed","stage":name,
                              "exit_code":result.returncode}),flush=True)
            return result.returncode
        print(json.dumps({"event":"stage_pass","stage":name}),flush=True)
    print(json.dumps({"event":"QA_COMPLETE","images":args.count}),flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
