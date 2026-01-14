import os
import re
import csv
import sys
import time
import json
import hashlib
import mimetypes
import random
import threading
from dataclasses import dataclass
from urllib.parse import urlparse, unquote

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# Config
# =========================
@dataclass
class Config:
    input_csv: str = "data.csv"
    out_dir: str = "images"

    # concurrency
    num_workers: int = 16

    # network
    timeout: int = 25
    retries: int = 3
    backoff_base: float = 0.8  # exponential backoff base

    # behavior
    skip_existing_files: bool = True
    resume_from_meta: bool = True         # 断点续传（读 meta.csv / done.jsonl）
    only_accept_image_content_type: bool = True  # 过滤非图片内容
    min_bytes: int = 256                  # 太小可能是错误页/占位图

    # logging
    log_format: str = "jsonl"             # "jsonl" or "csv"
    # jsonl: images/done.jsonl（实时追加，最稳）
    # csv  : images/meta.csv（实时写入）

    # split dataset (optional)
    make_splits: bool = False
    split_seed: int = 42
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05


CFG = Config()


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]+", "_", name)  # Windows illegal chars
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ._")
    if not name:
        name = "no_caption"
    return name[:max_len]


def stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def guess_ext_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext in [".jpg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"]:
        return ext
    return ""


def guess_ext_from_content_type(content_type: str) -> str:
    if not content_type:
        return ""
    content_type = content_type.split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(content_type) or ""
    if ext == ".jpe":
        ext = ".jpg"
    if ext == ".jpeg":
        ext = ".jpg"
    if ext in [".jpg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"]:
        return ext
    return ""


def is_image_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    content_type = content_type.split(";")[0].strip().lower()
    return content_type.startswith("image/")


def safe_print(*args, **kwargs):
    # avoid messy output from multiple threads
    with PRINT_LOCK:
        print(*args, **kwargs)


PRINT_LOCK = threading.Lock()


# =========================
# Reading input CSV robustly
# =========================
def read_csv_rows(csv_path: str):
    """
    Expect headers: image_url, caption
    but will try to be tolerant with variants:
    url/image/img/link and caption/text/description/title
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [x.strip() for x in (reader.fieldnames or [])]

        def pick(row, candidates):
            for cand in candidates:
                for real_k in fieldnames:
                    if real_k.lower() == cand.lower():
                        return row.get(real_k, "")
            return ""

        for i, row in enumerate(reader, start=1):
            url = pick(row, ["image_url", "url", "image", "img", "link"]).strip()
            caption = pick(row, ["caption", "text", "description", "title"]).strip()
            yield i, url, caption


# =========================
# Resume index (done set)
# =========================
def load_done_set(out_dir: str, log_format: str):
    """
    Returns a set of URL hashes or URLs that are done.
    Use URL string as key to avoid collision.
    """
    done = set()

    if log_format == "jsonl":
        path = os.path.join(out_dir, "done.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # treat ok & exists as done
                        if rec.get("status") in ("ok", "exists", "skip_non_image"):
                            done.add(rec.get("url", ""))
                    except Exception:
                        continue

    elif log_format == "csv":
        path = os.path.join(out_dir, "meta.csv")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") in ("ok", "exists", "skip_non_image"):
                        done.add(row.get("url", ""))

    # remove empty
    done.discard("")
    return done


# =========================
# Logging (thread-safe)
# =========================
class Logger:
    def __init__(self, out_dir: str, log_format: str):
        self.out_dir = out_dir
        self.log_format = log_format
        self.lock = threading.Lock()
        ensure_dir(out_dir)

        self.meta_path = None
        self.meta_file = None
        self.csv_writer = None

        if log_format == "csv":
            self.meta_path = os.path.join(out_dir, "meta.csv")
            new_file = not os.path.exists(self.meta_path)
            self.meta_file = open(self.meta_path, "a", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(
                self.meta_file,
                fieldnames=["idx", "url", "path", "status", "error", "caption"]
            )
            if new_file:
                self.csv_writer.writeheader()
                self.meta_file.flush()

        elif log_format == "jsonl":
            self.meta_path = os.path.join(out_dir, "done.jsonl")
            self.meta_file = open(self.meta_path, "a", encoding="utf-8")

        else:
            raise ValueError("log_format must be 'jsonl' or 'csv'")

    def write(self, rec: dict):
        with self.lock:
            if self.log_format == "csv":
                self.csv_writer.writerow({
                    "idx": rec.get("idx", ""),
                    "url": rec.get("url", ""),
                    "path": rec.get("path", ""),
                    "status": rec.get("status", ""),
                    "error": rec.get("error", ""),
                    "caption": rec.get("caption", ""),
                })
                self.meta_file.flush()
            else:
                self.meta_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.meta_file.flush()

    def close(self):
        with self.lock:
            if self.meta_file:
                self.meta_file.close()


# =========================
# Downloader
# =========================
def build_filename(idx: int, url: str, caption: str, ext: str) -> str:
    cap_part = sanitize_filename(caption)[:60]
    hid = stable_id(url)
    return f"{idx:06d}_{cap_part}_{hid}{ext}"


def download_one(idx: int, url: str, caption: str, cfg: Config, session: requests.Session):
    url = (url or "").strip()
    caption = (caption or "").strip()

    if not url:
        return {
            "idx": idx, "url": url, "caption": caption,
            "status": "skip_empty_url", "path": "", "error": ""
        }

    # base ext from url if possible
    ext = guess_ext_from_url(url) or ".jpg"
    filename = build_filename(idx, url, caption, ext)
    filepath = os.path.join(cfg.out_dir, filename)

    if cfg.skip_existing_files and os.path.exists(filepath) and os.path.getsize(filepath) >= cfg.min_bytes:
        return {
            "idx": idx, "url": url, "caption": caption,
            "status": "exists", "path": filepath, "error": ""
        }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DatasetImageCrawler/1.0)",
        "Accept": "image/*,*/*;q=0.8",
        "Referer": url,
    }

    last_err = ""
    for attempt in range(1, cfg.retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=cfg.timeout, allow_redirects=True) as r:
                r.raise_for_status()

                ct = r.headers.get("Content-Type", "")
                if cfg.only_accept_image_content_type and (not is_image_content_type(ct)):
                    return {
                        "idx": idx, "url": url, "caption": caption,
                        "status": "skip_non_image", "path": "", "error": f"Content-Type={ct}"
                    }

                # if url had no good ext, try content-type
                if not guess_ext_from_url(url):
                    cext = guess_ext_from_content_type(ct)
                    if cext and cext != ext:
                        ext = cext
                        filename = build_filename(idx, url, caption, ext)
                        filepath = os.path.join(cfg.out_dir, filename)

                ensure_dir(cfg.out_dir)
                tmp_path = filepath + ".part"

                total = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)

                os.replace(tmp_path, filepath)

            if os.path.getsize(filepath) < cfg.min_bytes:
                # likely an error page / tiny placeholder
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                raise RuntimeError(f"download_too_small (<{cfg.min_bytes} bytes)")

            return {
                "idx": idx, "url": url, "caption": caption,
                "status": "ok", "path": filepath, "error": ""
            }

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # cleanup tmp file if exists
            try:
                if os.path.exists(filepath + ".part"):
                    os.remove(filepath + ".part")
            except Exception:
                pass

            time.sleep(cfg.backoff_base * (2 ** (attempt - 1)))

    return {
        "idx": idx, "url": url, "caption": caption,
        "status": "failed", "path": "", "error": last_err
    }


# =========================
# Split maker (optional)
# =========================
def make_splits_from_log(out_dir: str, log_format: str, seed: int, ratios):
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    records = []
    if log_format == "jsonl":
        path = os.path.join(out_dir, "done.jsonl")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("status") in ("ok", "exists") and rec.get("path"):
                    records.append(rec)
    else:
        path = os.path.join(out_dir, "meta.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") in ("ok", "exists") and row.get("path"):
                    records.append(row)

    if not records:
        return

    random.Random(seed).shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    def dump_csv(name, lst):
        p = os.path.join(out_dir, f"{name}.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "url", "caption"])
            w.writeheader()
            for r in lst:
                w.writerow({
                    "path": r.get("path", ""),
                    "url": r.get("url", ""),
                    "caption": r.get("caption", ""),
                })

    dump_csv("train", train)
    dump_csv("val", val)
    dump_csv("test", test)


# =========================
# Main
# =========================
def main(cfg: Config):
    if not os.path.exists(cfg.input_csv):
        print(f"ERROR: {cfg.input_csv} not found.")
        sys.exit(1)

    ensure_dir(cfg.out_dir)

    logger = Logger(cfg.out_dir, cfg.log_format)

    done_set = set()
    if cfg.resume_from_meta:
        done_set = load_done_set(cfg.out_dir, cfg.log_format)
        safe_print(f"[resume] loaded done={len(done_set)}")

    total_rows = 0
    submitted = 0
    ok_cnt = 0
    exists_cnt = 0
    failed_cnt = 0
    skipped_cnt = 0

    start_time = time.time()

    with requests.Session() as session:
        futures = []
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            for idx, url, caption in read_csv_rows(cfg.input_csv):
                total_rows += 1
                if cfg.resume_from_meta and url in done_set:
                    skipped_cnt += 1
                    continue
                futures.append(ex.submit(download_one, idx, url, caption, cfg, session))
                submitted += 1

            safe_print(f"[start] rows={total_rows} submitted={submitted} skipped_by_resume={skipped_cnt}")

            for i, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                logger.write(res)

                st = res.get("status")
                if st == "ok":
                    ok_cnt += 1
                elif st == "exists":
                    exists_cnt += 1
                elif st in ("skip_empty_url", "skip_non_image"):
                    skipped_cnt += 1
                elif st == "failed":
                    failed_cnt += 1
                else:
                    skipped_cnt += 1

                if i % 50 == 0 or i == submitted:
                    elapsed = time.time() - start_time
                    safe_print(
                        f"[progress] done={i}/{submitted} ok={ok_cnt} exists={exists_cnt} "
                        f"failed={failed_cnt} skipped={skipped_cnt} elapsed={elapsed:.1f}s"
                    )

    logger.close()

    # Optional: dataset splits
    if cfg.make_splits:
        make_splits_from_log(
            cfg.out_dir,
            cfg.log_format,
            cfg.split_seed,
            (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
        )
        safe_print("[split] generated train.csv/val.csv/test.csv")

    elapsed = time.time() - start_time
    safe_print(
        f"Done. ok={ok_cnt} exists={exists_cnt} failed={failed_cnt} skipped={skipped_cnt} "
        f"elapsed={elapsed:.1f}s"
    )
    safe_print(f"Logs at: {logger.meta_path}")


if __name__ == "__main__":
    main(CFG)
