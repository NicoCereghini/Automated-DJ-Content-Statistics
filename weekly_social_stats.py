#!/usr/bin/env python3
"""
Weekly IG + TikTok Stats Job

Install
1) Python 3.11+
2) pip install -r requirements.txt

Environment setup
1) Copy .env.example to .env
2) Fill required variables for your accounts/apps:
   - IG_ACCESS_TOKEN
   - IG_USER_ID
   - TIKTOK_ACCESS_TOKEN
   - TIKTOK_USER_ID or TIKTOK_ADVERTISER_ID (depends on TikTok API product setup)

Run once
python weekly_social_stats.py --max-videos 7 --since-days 7

Cron setup (Linux, weekly Sunday 9pm America/New_York)
TZ=America/New_York
0 21 * * 0 /usr/bin/env python3 /absolute/path/to/weekly_social_stats.py --max-videos 7 --since-days 7 --outdir /absolute/path/to/out

Outputs
./out/YYYY-MM-DD/
- raw_instagram.json
- raw_tiktok.json
- normalized_videos.csv
- weekly_report.txt
- weekly_report.md
- weekly_summary.csv

Labels and matching via labels.yml or labels.json
- Optional file at ./labels.yml or ./labels.json
- Supported structure (either list or {"entries": [...]}) where each entry includes:
  - video_index (1..N)
  - label (e.g. "Track X")
  - instagram_id (optional)
  - tiktok_id (optional)
- Behavior:
  - label replaces default "Track <index>"
  - if instagram_id/tiktok_id are provided, the script picks those videos for that index when found
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


IG_API_BASE = "https://graph.facebook.com/v21.0"
TT_API_BASE = "https://business-api.tiktok.com/open_api/v1.3"

# Keep endpoint paths and field names centralized to simplify account/app-specific adjustments.
IG_ENDPOINTS = {
    "media_list": "/{ig_user_id}/media",
    "insights": "/{media_id}/insights",
}

TT_ENDPOINTS = {
    # TODO: Verify endpoint versions/paths for your exact TikTok product and scopes.
    "video_list": "/business/video/list/",
    "video_insights": "/business/video/get/",
}


@dataclass
class NormalizedVideo:
    platform: str
    video_id: str
    created_time_utc: Optional[datetime]
    duration_seconds: Optional[float]
    permalink: Optional[str]
    plays_or_views: Optional[int]
    avg_watch_time_seconds: Optional[float]
    percent_completion: Optional[float]
    likes: Optional[int]
    comments: Optional[int]
    shares: Optional[int]
    saves: Optional[int]
    follows_from_video: Optional[int]
    fyp_pct: Optional[float]


@dataclass
class LabelEntry:
    video_index: int
    label: str
    instagram_id: Optional[str] = None
    tiktok_id: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly IG + TikTok stats report")
    parser.add_argument("--since-days", type=int, default=7)
    parser.add_argument("--max-videos", type=int, default=7)
    parser.add_argument("--outdir", default="./out")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_outdir(base_outdir: str) -> Path:
    folder = Path(base_outdir) / datetime.now().date().isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(txt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def fmt_int(value: Optional[int], na_label: str = "N/A") -> str:
    if value is None:
        return na_label
    return f"{value:,}"


def fmt_seconds_1(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def fmt_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return str(int(round(value)))


def plural(word: str, value: Optional[int]) -> str:
    if value == 1:
        return word
    return f"{word}s"


def coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_get(source: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def approximate_percent(avg_watch: Optional[float], duration: Optional[float]) -> Optional[float]:
    if avg_watch is None or duration is None or duration <= 0:
        return None
    return max(0.0, min(100.0, (avg_watch / duration) * 100.0))


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    max_retries: int = 5,
) -> requests.Response:
    attempt = 0
    while True:
        attempt += 1
        response = session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=timeout,
        )
        if response.status_code < 400:
            return response

        retriable = response.status_code == 429 or 500 <= response.status_code < 600
        if not retriable or attempt > max_retries:
            return response

        wait_seconds = None
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                wait_seconds = max(0.0, float(retry_after))
            except ValueError:
                wait_seconds = None

        if wait_seconds is None:
            reset_header = response.headers.get("X-RateLimit-Reset")
            if reset_header:
                try:
                    reset_epoch = float(reset_header)
                    wait_seconds = max(0.0, reset_epoch - time.time())
                except ValueError:
                    wait_seconds = None

        if wait_seconds is None:
            wait_seconds = min(60.0, (2 ** (attempt - 1)) + 0.25)

        time.sleep(wait_seconds)


def load_fixture(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_labels(max_videos: int) -> Dict[int, LabelEntry]:
    labels_path_yaml = Path("./labels.yml")
    labels_path_json = Path("./labels.json")

    payload: Optional[Any] = None
    if labels_path_yaml.exists():
        if yaml is None:
            print("Warning: labels.yml exists but PyYAML is not installed; skipping labels.yml", file=sys.stderr)
        else:
            payload = yaml.safe_load(labels_path_yaml.read_text(encoding="utf-8"))
    elif labels_path_json.exists():
        payload = json.loads(labels_path_json.read_text(encoding="utf-8"))

    if payload is None:
        return {}

    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return {}

    labels: Dict[int, LabelEntry] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        idx = coerce_int(raw.get("video_index"))
        if idx is None or idx < 1 or idx > max_videos:
            continue
        label = str(raw.get("label") or f"Track {idx}").strip() or f"Track {idx}"
        labels[idx] = LabelEntry(
            video_index=idx,
            label=label,
            instagram_id=str(raw.get("instagram_id")).strip() if raw.get("instagram_id") else None,
            tiktok_id=str(raw.get("tiktok_id")).strip() if raw.get("tiktok_id") else None,
        )
    return labels


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_instagram_raw(
    session: requests.Session,
    access_token: str,
    ig_user_id: str,
    max_videos: int,
    since_dt: datetime,
) -> Dict[str, Any]:
    fields = [
        "id",
        "caption",
        "media_type",
        "media_product_type",
        "timestamp",
        "permalink",
        "like_count",
        "comments_count",
        "video_title",
        "video_duration",
        "video_length",
        "insights.metric(plays,video_views,avg_watch_time,average_watch_time,total_watch_time,likes,saved,shares,follows,accounts_engaged,accounts_reached,video_view_time,video_avg_time_watched)",
    ]
    url = IG_API_BASE + IG_ENDPOINTS["media_list"].format(ig_user_id=ig_user_id)

    params = {
        "access_token": access_token,
        "fields": ",".join(fields),
        "limit": max(25, max_videos * 4),
    }

    resp = request_with_retry(session, "GET", url, params=params)
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise RuntimeError(f"Instagram API error ({resp.status_code}): {data}")

    media_items = data.get("data") or []
    if not isinstance(media_items, list):
        media_items = []

    filtered: List[Dict[str, Any]] = []
    for item in media_items:
        if not isinstance(item, dict):
            continue
        ts = parse_datetime(item.get("timestamp"))
        if ts and ts < since_dt:
            continue
        filtered.append(item)

    filtered.sort(key=lambda m: parse_datetime(m.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return {"data": filtered[:max_videos], "paging": data.get("paging")}


def fetch_tiktok_raw(
    session: requests.Session,
    access_token: str,
    user_id: Optional[str],
    advertiser_id: Optional[str],
    max_videos: int,
    since_dt: datetime,
) -> Dict[str, Any]:
    url = TT_API_BASE + TT_ENDPOINTS["video_list"]
    headers = {"Access-Token": access_token, "Content-Type": "application/json"}

    # TODO: Confirm exact payload keys for your TikTok app permissions and account type.
    body: Dict[str, Any] = {
        "page": 1,
        "page_size": max(20, max_videos * 3),
    }
    if user_id:
        body["user_id"] = user_id
    if advertiser_id:
        body["advertiser_id"] = advertiser_id

    resp = request_with_retry(session, "POST", url, headers=headers, json_body=body)
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise RuntimeError(f"TikTok API HTTP error ({resp.status_code}): {data}")

    # Many TikTok APIs encode errors as JSON fields with HTTP 200.
    code = safe_get(data, ["code", "status_code"])
    if code not in (None, 0, "0", "OK"):
        msg = safe_get(data, ["message", "msg", "status_msg"]) or "Unknown API error"
        raise RuntimeError(f"TikTok API error (code={code}): {msg}")

    rows = safe_get(data, ["data", "result", "list", "videos"]) or {}
    if isinstance(rows, dict):
        videos = safe_get(rows, ["list", "videos", "items", "data"]) or []
    elif isinstance(rows, list):
        videos = rows
    else:
        videos = []

    out: List[Dict[str, Any]] = []
    for item in videos:
        if not isinstance(item, dict):
            continue
        dt = parse_datetime(safe_get(item, ["create_time", "created_time", "publish_time", "create_timestamp"]))
        if dt and dt < since_dt:
            continue
        out.append(item)

    out.sort(
        key=lambda m: parse_datetime(safe_get(m, ["create_time", "created_time", "publish_time", "create_timestamp"]))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return {"data": out[:max_videos], "raw": data}


def normalize_instagram(raw: Dict[str, Any]) -> List[NormalizedVideo]:
    data = raw.get("data") or []
    normalized: List[NormalizedVideo] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("id") or "")
        created = parse_datetime(item.get("timestamp"))
        duration = coerce_float(safe_get(item, ["video_duration", "video_length", "duration"]))

        insights_map: Dict[str, Any] = {}
        insights = safe_get(item, ["insights", "video_insights", "metrics"]) or {}
        rows = safe_get(insights, ["data", "metrics", "values"]) if isinstance(insights, dict) else None
        if isinstance(rows, list):
            for metric in rows:
                if not isinstance(metric, dict):
                    continue
                name = metric.get("name") or metric.get("metric")
                value = metric.get("value")
                if isinstance(value, list) and value:
                    first = value[0]
                    if isinstance(first, dict) and "value" in first:
                        value = first.get("value")
                    else:
                        value = first
                if name:
                    insights_map[str(name)] = value

        plays = coerce_int(safe_get(insights_map, ["plays", "video_views", "impressions"]))
        avg_watch = coerce_float(safe_get(insights_map, ["avg_watch_time", "average_watch_time", "video_avg_time_watched"]))
        percent = coerce_float(safe_get(insights_map, ["video_view_time", "percent_watched", "completion_rate"]))
        if percent is None:
            percent = approximate_percent(avg_watch, duration)

        likes = coerce_int(safe_get(insights_map, ["likes"]))
        if likes is None:
            likes = coerce_int(item.get("like_count"))

        comments = coerce_int(safe_get(insights_map, ["comments"]))
        if comments is None:
            comments = coerce_int(item.get("comments_count"))

        saves = coerce_int(safe_get(insights_map, ["saved", "saves"]))
        shares = coerce_int(safe_get(insights_map, ["shares"]))
        follows = coerce_int(safe_get(insights_map, ["follows", "follows_from_content"]))

        normalized.append(
            NormalizedVideo(
                platform="instagram",
                video_id=video_id,
                created_time_utc=created,
                duration_seconds=duration,
                permalink=item.get("permalink"),
                plays_or_views=plays,
                avg_watch_time_seconds=avg_watch,
                percent_completion=percent,
                likes=likes,
                comments=comments,
                shares=shares,
                saves=saves,
                follows_from_video=follows,
                fyp_pct=None,
            )
        )

    normalized.sort(key=lambda x: x.created_time_utc or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return normalized


def normalize_tiktok(raw: Dict[str, Any]) -> List[NormalizedVideo]:
    data = raw.get("data") or []
    normalized: List[NormalizedVideo] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        video_id = str(safe_get(item, ["video_id", "item_id", "id", "aweme_id"]) or "")
        created = parse_datetime(safe_get(item, ["create_time", "created_time", "publish_time", "create_timestamp"]))
        duration = coerce_float(safe_get(item, ["duration", "video_duration", "video_length"]))

        stats = safe_get(item, ["stats", "video_metrics", "metrics", "insights"]) or {}
        if not isinstance(stats, dict):
            stats = {}

        views = coerce_int(safe_get(stats, ["view_count", "views", "play_count", "video_views"]))
        if views is None:
            views = coerce_int(safe_get(item, ["view_count", "views", "play_count", "video_views"]))

        avg_watch = coerce_float(safe_get(stats, ["avg_watch_time", "average_watch_time", "avg_watch_duration"]))
        if avg_watch is None:
            avg_watch = coerce_float(safe_get(item, ["avg_watch_time", "average_watch_time", "avg_watch_duration"]))

        percent_full = coerce_float(safe_get(stats, ["completion_rate", "percent_full", "video_completion_rate"]))
        if percent_full is None:
            percent_full = coerce_float(safe_get(item, ["completion_rate", "percent_full", "video_completion_rate"]))
        if percent_full is None:
            percent_full = approximate_percent(avg_watch, duration)

        fyp_pct = coerce_float(safe_get(stats, ["fyp_pct", "for_you_pct", "fyp_traffic_pct"]))
        if fyp_pct is None:
            fyp_pct = coerce_float(safe_get(item, ["fyp_pct", "for_you_pct", "fyp_traffic_pct"]))

        likes = coerce_int(safe_get(stats, ["like_count", "likes"]))
        if likes is None:
            likes = coerce_int(safe_get(item, ["like_count", "likes"]))

        comments = coerce_int(safe_get(stats, ["comment_count", "comments"]))
        if comments is None:
            comments = coerce_int(safe_get(item, ["comment_count", "comments"]))

        shares = coerce_int(safe_get(stats, ["share_count", "shares"]))
        if shares is None:
            shares = coerce_int(safe_get(item, ["share_count", "shares"]))

        follows = coerce_int(safe_get(stats, ["follows", "follows_from_video", "profile_visits_to_follow"]))
        saves = coerce_int(safe_get(stats, ["save_count", "saves", "favorites"]))

        permalink = safe_get(item, ["share_url", "permalink", "video_url"])

        normalized.append(
            NormalizedVideo(
                platform="tiktok",
                video_id=video_id,
                created_time_utc=created,
                duration_seconds=duration,
                permalink=permalink,
                plays_or_views=views,
                avg_watch_time_seconds=avg_watch,
                percent_completion=percent_full,
                likes=likes,
                comments=comments,
                shares=shares,
                saves=saves,
                follows_from_video=follows,
                fyp_pct=fyp_pct,
            )
        )

    normalized.sort(key=lambda x: x.created_time_utc or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return normalized


def write_normalized_csv(path: Path, rows: List[NormalizedVideo]) -> None:
    fieldnames = [
        "platform",
        "video_id",
        "created_time_utc",
        "duration_seconds",
        "permalink",
        "plays_or_views",
        "avg_watch_time_seconds",
        "percent_completion",
        "likes",
        "comments",
        "shares",
        "saves",
        "follows_from_video",
        "fyp_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "platform": row.platform,
                    "video_id": row.video_id,
                    "created_time_utc": row.created_time_utc.isoformat() if row.created_time_utc else "",
                    "duration_seconds": "" if row.duration_seconds is None else f"{row.duration_seconds:.3f}",
                    "permalink": row.permalink or "",
                    "plays_or_views": "" if row.plays_or_views is None else row.plays_or_views,
                    "avg_watch_time_seconds": "" if row.avg_watch_time_seconds is None else f"{row.avg_watch_time_seconds:.3f}",
                    "percent_completion": "" if row.percent_completion is None else f"{row.percent_completion:.3f}",
                    "likes": "" if row.likes is None else row.likes,
                    "comments": "" if row.comments is None else row.comments,
                    "shares": "" if row.shares is None else row.shares,
                    "saves": "" if row.saves is None else row.saves,
                    "follows_from_video": "" if row.follows_from_video is None else row.follows_from_video,
                    "fyp_pct": "" if row.fyp_pct is None else f"{row.fyp_pct:.3f}",
                }
            )


def build_slots(
    ig_videos: List[NormalizedVideo],
    tt_videos: List[NormalizedVideo],
    labels: Dict[int, LabelEntry],
    max_videos: int,
) -> List[Tuple[int, str, Optional[NormalizedVideo], Optional[NormalizedVideo]]]:
    ig_by_id = {v.video_id: v for v in ig_videos if v.video_id}
    tt_by_id = {v.video_id: v for v in tt_videos if v.video_id}

    combined: List[Tuple[str, NormalizedVideo]] = [("instagram", v) for v in ig_videos] + [("tiktok", v) for v in tt_videos]
    combined.sort(key=lambda pair: pair[1].created_time_utc or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    slots: List[Tuple[int, str, Optional[NormalizedVideo], Optional[NormalizedVideo]]] = []
    for idx in range(1, max_videos + 1):
        default_label = f"Track {idx}"
        label_entry = labels.get(idx)
        label = label_entry.label if label_entry else default_label

        ig_row: Optional[NormalizedVideo] = None
        tt_row: Optional[NormalizedVideo] = None

        if idx - 1 < len(combined):
            platform, video = combined[idx - 1]
            if platform == "instagram":
                ig_row = video
            else:
                tt_row = video

        # Optional explicit mapping overrides slot selection.
        if label_entry:
            if label_entry.instagram_id:
                ig_row = ig_by_id.get(label_entry.instagram_id)
            if label_entry.tiktok_id:
                tt_row = tt_by_id.get(label_entry.tiktok_id)

        slots.append((idx, label, ig_row, tt_row))

    return slots


def render_ig_line(video: Optional[NormalizedVideo]) -> str:
    if video is None:
        return "IG: N/A plays | N/A avg | N/A watched | N/A likes | N/A saves | N/A shares | N/A follows"

    follows_val = video.follows_from_video
    follows_text = plural("follow", follows_val)

    return (
        f"IG: {fmt_int(video.plays_or_views)} plays | "
        f"{fmt_seconds_1(video.avg_watch_time_seconds)}s avg | "
        f"{fmt_percent(video.percent_completion)}% watched | "
        f"{fmt_int(video.likes)} likes | "
        f"{fmt_int(video.saves)} saves | "
        f"{fmt_int(video.shares)} shares | "
        f"{fmt_int(follows_val)} {follows_text}"
    )


def render_tt_line(video: Optional[NormalizedVideo]) -> str:
    if video is None:
        return "TT: N/A views | N/A avg | N/A full | N/A FYP | N/A likes | N/A comments | N/A follows"

    comments_val = video.comments
    follows_val = video.follows_from_video
    comments_text = plural("comment", comments_val)
    follows_text = plural("follow", follows_val)

    return (
        f"TT: {fmt_int(video.plays_or_views)} views | "
        f"{fmt_seconds_1(video.avg_watch_time_seconds)}s avg | "
        f"{fmt_percent(video.percent_completion)}% full | "
        f"{fmt_percent(video.fyp_pct)}% FYP | "
        f"{fmt_int(video.likes)} likes | "
        f"{fmt_int(comments_val)} {comments_text} | "
        f"{fmt_int(follows_val)} {follows_text}"
    )


def write_report(
    txt_path: Path,
    md_path: Path,
    slots: List[Tuple[int, str, Optional[NormalizedVideo], Optional[NormalizedVideo]]],
) -> str:
    lines: List[str] = []
    for idx, label, ig_row, tt_row in slots:
        lines.append(f"Video {idx} – {label}")
        lines.append(render_ig_line(ig_row))
        lines.append(render_tt_line(tt_row))
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    content = "\n".join(lines) + "\n"
    txt_path.write_text(content, encoding="utf-8")
    md_path.write_text(content, encoding="utf-8")
    return content


def summarize_rows(rows: List[NormalizedVideo], week_start: datetime, week_end: datetime) -> List[Dict[str, Any]]:
    def summarize(platform_name: str, platform_rows: List[NormalizedVideo]) -> Dict[str, Any]:
        numeric_avg_watch = [r.avg_watch_time_seconds for r in platform_rows if r.avg_watch_time_seconds is not None]
        avg_avg_watch = (sum(numeric_avg_watch) / len(numeric_avg_watch)) if numeric_avg_watch else None

        def sum_int(attr: str) -> int:
            return int(sum(getattr(r, attr) or 0 for r in platform_rows))

        return {
            "platform": platform_name,
            "week_start_date": week_start.date().isoformat(),
            "week_end_date": week_end.date().isoformat(),
            "num_videos": len(platform_rows),
            "total_plays_or_views": sum_int("plays_or_views"),
            "total_likes": sum_int("likes"),
            "total_comments": sum_int("comments"),
            "total_shares": sum_int("shares"),
            "total_saves": sum_int("saves"),
            "avg_avg_watch_time_seconds": "" if avg_avg_watch is None else f"{avg_avg_watch:.3f}",
            "follower_change_week": "N/A",
        }

    ig_rows = [r for r in rows if r.platform == "instagram"]
    tt_rows = [r for r in rows if r.platform == "tiktok"]
    combined_rows = list(rows)

    return [
        summarize("instagram", ig_rows),
        summarize("tiktok", tt_rows),
        summarize("combined", combined_rows),
    ]


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "platform",
        "week_start_date",
        "week_end_date",
        "num_videos",
        "total_plays_or_views",
        "total_likes",
        "total_comments",
        "total_shares",
        "total_saves",
        "avg_avg_watch_time_seconds",
        "follower_change_week",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    load_dotenv()
    args = parse_args()

    outdir = ensure_outdir(args.outdir)
    fixtures_dir = Path("./fixtures")

    now_utc = datetime.now(timezone.utc)
    since_dt = now_utc - timedelta(days=args.since_days)

    labels = load_labels(args.max_videos)

    session = requests.Session()
    raw_ig: Dict[str, Any] = {"data": []}
    raw_tt: Dict[str, Any] = {"data": []}

    ig_error: Optional[str] = None
    tt_error: Optional[str] = None

    if args.dry_run:
        ig_fixture = load_fixture(fixtures_dir / "instagram.json")
        tt_fixture = load_fixture(fixtures_dir / "tiktok.json")

        if ig_fixture is None:
            ig_error = "Dry-run fixture missing: ./fixtures/instagram.json"
        else:
            raw_ig = ig_fixture

        if tt_fixture is None:
            tt_error = "Dry-run fixture missing: ./fixtures/tiktok.json"
        else:
            raw_tt = tt_fixture
    else:
        ig_token = os.getenv("IG_ACCESS_TOKEN")
        ig_user_id = os.getenv("IG_USER_ID")

        tt_token = os.getenv("TIKTOK_ACCESS_TOKEN")
        tt_user_id = os.getenv("TIKTOK_USER_ID")
        tt_advertiser_id = os.getenv("TIKTOK_ADVERTISER_ID")

        if not ig_token or not ig_user_id:
            missing = []
            if not ig_token:
                missing.append("IG_ACCESS_TOKEN")
            if not ig_user_id:
                missing.append("IG_USER_ID")
            ig_error = f"Missing required Instagram env vars: {', '.join(missing)}"

        if (not tt_token) or (not tt_user_id and not tt_advertiser_id):
            missing = []
            if not tt_token:
                missing.append("TIKTOK_ACCESS_TOKEN")
            if not tt_user_id and not tt_advertiser_id:
                missing.append("TIKTOK_USER_ID or TIKTOK_ADVERTISER_ID")
            tt_error = f"Missing required TikTok env vars: {', '.join(missing)}"

        if ig_error is None:
            try:
                raw_ig = fetch_instagram_raw(
                    session=session,
                    access_token=ig_token or "",
                    ig_user_id=ig_user_id or "",
                    max_videos=args.max_videos,
                    since_dt=since_dt,
                )
            except Exception as exc:
                ig_error = f"Instagram fetch failed: {exc}"

        if tt_error is None:
            try:
                raw_tt = fetch_tiktok_raw(
                    session=session,
                    access_token=tt_token or "",
                    user_id=tt_user_id,
                    advertiser_id=tt_advertiser_id,
                    max_videos=args.max_videos,
                    since_dt=since_dt,
                )
            except Exception as exc:
                tt_error = f"TikTok fetch failed: {exc}"

    save_json(outdir / "raw_instagram.json", raw_ig)
    save_json(outdir / "raw_tiktok.json", raw_tt)

    ig_videos = normalize_instagram(raw_ig) if ig_error is None else []
    tt_videos = normalize_tiktok(raw_tt) if tt_error is None else []

    ig_videos = ig_videos[: args.max_videos]
    tt_videos = tt_videos[: args.max_videos]

    all_rows = sorted(
        ig_videos + tt_videos,
        key=lambda v: v.created_time_utc or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    write_normalized_csv(outdir / "normalized_videos.csv", all_rows)

    slots = build_slots(ig_videos, tt_videos, labels, args.max_videos)
    write_report(outdir / "weekly_report.txt", outdir / "weekly_report.md", slots)

    week_start = since_dt
    week_end = now_utc
    summary_rows = summarize_rows(all_rows, week_start, week_end)
    write_summary_csv(outdir / "weekly_summary.csv", summary_rows)

    if ig_error:
        print(ig_error, file=sys.stderr)
    if tt_error:
        print(tt_error, file=sys.stderr)

    if ig_error and tt_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
