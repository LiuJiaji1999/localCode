
# -*- coding: utf-8 -*-
"""
机巢-杆塔智能调度：基于当前已有数据的可运行参考实现
====================================================

当前数据可以真实支持：
1) 读取机巢/杆塔坐标；
2) 已安装机巢筛选；
3) 机巢-杆塔距离矩阵；
4) 每基杆塔 Top-K 候选机巢；
5) 最近机巢分配、机巢空间负载；
6) 按线路杆塔号恢复线路顺序；
7) 在给定“仿真运行参数”后，构造任务、按线路连续切分架次、计算预计时间、
   做优先级+进度公平性调度。

当前数据不包含：
- 无人机型号/航速/最大航时/电池；
- 单塔不同任务真实作业时间；
- 巡视历史；
- 耐张塔标识；
- 天气窗口/禁飞区。
因此这些内容全部做成参数或可选输入，代码不会把它们冒充成真实数据。

依赖：
    pandas, numpy, openpyxl
无需 OR-Tools，保证当前环境可直接运行。


# 只做真实坐标空间分析，不加入任何飞行假设：
python uav_dispatch.py spatial --nest-xlsx 机巢ZB.xlsx --tower-xlsx 线路ZB.xlsx --outdir output_spatial

# 仅做“通道巡视”的初始化仿真任务（所有杆塔视为待执行）
python uav_dispatch.py plan --nest-xlsx 机巢ZB.xlsx --tower-xlsx 线路ZB.xlsx --task-types 通道巡视 --outdir output_plan
# 同时模拟3类任务
python uav_dispatch.py plan --nest-xlsx 机巢ZB.xlsx --tower-xlsx 线路ZB.xlsx --task-types 通道巡视 精细化巡视 仿线飞行 --start-date 2026-08-26 --days 30 --outdir output_plan

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import math
import re
import pandas as pd
import numpy as np


# =========================
# 1. 配置


TASK_TEMP = "临时任务"
TASK_IR = "红外测温"
TASK_DETAIL = "精细化巡视"
TASK_LINE = "仿线飞行"
TASK_CHANNEL = "通道巡视"

PERIODIC_TASKS = [TASK_DETAIL, TASK_CHANNEL, TASK_LINE]
# 一个“层级编码”，基础优先级上形成层级关系
BASE_PRIORITY = {
    TASK_TEMP: 400.0,
    TASK_IR: 300.0,
    TASK_DETAIL: 200.0,
    TASK_LINE: 200.0,
    TASK_CHANNEL: 200.0,
}

@dataclass
class DispatchConfig:
    # --- 空间参数 ---
    candidate_top_k: int = 3 #每基杆塔最多保留3个候选机巢。
    active_nest_status: Tuple[str, ...] = ("已安装",)

    # 可选硬半径；None 表示只计算候选，不因半径判不可达。
    # 真实值必须来自机型/规程。
    max_one_way_km: Optional[float] = None

    # --- 以下均为仿真参数，必须用真实参数替换 ---
    cruise_speed_kmh: float = 36.0 #假定无人机巡航速度为36km/h；
    max_sortie_minutes: float = 45.0 # 单架次最大时间45分钟；
    max_sortie_distance_km: float = 18.0 #最大航程18km；
    reserve_minutes: float = 8.0 #至少预留8分钟的安全余量。

    # 无人机到达杆塔之后，不同任务需要不同的作业时间 
    service_minutes: Dict[str, float] = field(default_factory=lambda: {
        TASK_CHANNEL: 0.7,
        TASK_DETAIL: 3.0,
        TASK_LINE: 0.8,
        TASK_IR: 2.0,
        TASK_TEMP: 3.0,
    })

    # 每个机巢默认 1 架无人机，每日可调度窗口
    day_start_hour: int = 8
    day_end_hour: int = 17
    turnaround_minutes: float = 15.0

    # 任务调度权重
    overdue_weight: float = 60.0
    urgency_weight: float = 30.0
    fairness_weight: float = 80.0
    route_efficiency_weight: float = 10.0

    # 三类同级周期任务的目标进度份额
    fair_task_types: Tuple[str, ...] = (TASK_DETAIL, TASK_LINE, TASK_CHANNEL)


# =========================
# 2. 数据读取

# 机巢数据：当前机巢数据共计约60个点位，其中：
# 已安装机巢：12个；拟安装机巢：48个。
# 每个机巢当前主要包括： 机巢名称、经度、 纬度、建设状态。
# 当前调度程序只把状态为：**“已安装”**的机巢作为真正可以执行任务的机巢。
def load_nests(path: str) -> pd.DataFrame:
    """
    机巢文件当前结构：
    第1行标题，第2行真实表头。
    返回：
        N001 石滩  经度  纬度  已安装
        N002 杜市  经度  纬度  已安装   
    """
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    # 当前文件第 2 行(索引1)是列名
    df = raw.iloc[2:, [1, 2, 3, 4]].copy()
    df.columns = ["nest_name", "longitude", "latitude", "status"]
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["nest_name", "longitude", "latitude"]).reset_index(drop=True)
    df["nest_id"] = [f"N{i+1:03d}" for i in range(len(df))]
    return df[["nest_id", "nest_name", "longitude", "latitude", "status"]]

# 从杆塔名称中提取杆塔号，
def _parse_tower_no(name: str) -> Optional[int]:
    if pd.isna(name):
        return None
    s = str(name)
    m = re.search(r"#\s*(\d+)", s)
    if not m:
        m = re.search(r"(\d+)\s*$", s)
    return int(m.group(1)) if m else None

# 2.2 杆塔数据
# 当前杆塔数据包括约：17,845基杆塔、312条线路。
# 每基杆塔包括：线路名称、电压等级、杆塔名称 、经度、纬度。
# 万马线#31 中提取 #31 作为tower_no，构建线路拓扑的依据
def load_towers(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = df.rename(columns={
        "线路名称": "line_name",
        "电压等级": "voltage_level",
        "杆塔名称": "tower_name",
        "J度": "longitude",
        "W度": "latitude",
    })
    required = ["line_name", "voltage_level", "tower_name", "longitude", "latitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"杆塔文件缺少字段: {missing}")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["tower_name", "longitude", "latitude"]).reset_index(drop=True)
    df["tower_id"] = [f"T{i+1:06d}" for i in range(len(df))]
    df["tower_no"] = df["tower_name"].map(_parse_tower_no)
    return df[[
        "tower_id", "line_name", "voltage_level", "tower_name",
        "tower_no", "longitude", "latitude"
    ]]


# =========================
# 3. 空间计算，输出结果为两个地理点之间的球面距离
EARTH_RADIUS_KM = 6371.0088 #地球半径

def haversine_matrix_km(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    返回 shape=(len(lat1), len(lat2)) 的球面距离矩阵。
    对当前 17845 x 60 数据规模完全可行。
    
    """
    lat1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(lon2, dtype=float))[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

# Top-3候选机巢
def build_candidate_nests(
    towers: pd.DataFrame,
    nests: pd.DataFrame,
    config: DispatchConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    active = nests[nests["status"].isin(config.active_nest_status)].reset_index(drop=True)
    if active.empty:
        raise ValueError(f"没有状态属于 {config.active_nest_status} 的可用机巢")

    dist = haversine_matrix_km(
        towers["latitude"].to_numpy(), towers["longitude"].to_numpy(),
        active["latitude"].to_numpy(), active["longitude"].to_numpy()
    )

    k = min(config.candidate_top_k, len(active))
    idx = np.argpartition(dist, kth=k-1, axis=1)[:, :k]

    # 每行 Top-K 再排序
    row = np.arange(len(towers))[:, None]
    top_dist = dist[row, idx]
    order = np.argsort(top_dist, axis=1)
    idx = idx[row, order]
    top_dist = dist[row, idx]

    rows = []
    for rank in range(k):
        nest_idx = idx[:, rank]
        tmp = pd.DataFrame({
            "tower_id": towers["tower_id"].values,
            "tower_name": towers["tower_name"].values,
            "line_name": towers["line_name"].values,
            "rank": rank + 1,
            "nest_id": active.iloc[nest_idx]["nest_id"].values,
            "nest_name": active.iloc[nest_idx]["nest_name"].values,
            "distance_km": top_dist[:, rank],
        })
        if config.max_one_way_km is None:
            tmp["spatial_feasible"] = True
        else:
            tmp["spatial_feasible"] = tmp["distance_km"] <= config.max_one_way_km
        rows.append(tmp)

    candidates = pd.concat(rows, ignore_index=True)
    nearest = candidates[candidates["rank"] == 1].copy()
    return candidates, nearest

# 形成17845*12 距离矩阵，得到每基杆塔到所有当前可用机巢的距离。
def build_spatial_summary(
    towers: pd.DataFrame,
    nests: pd.DataFrame,
    nearest: pd.DataFrame,
) -> pd.DataFrame:
    active = nests[nests["status"].eq("已安装")].copy()
     #  使用预计工作分钟数或预计无人机占用小时数作为负载
    load = nearest.groupby(["nest_id", "nest_name"], as_index=False).agg(
        assigned_towers=("tower_id", "count"),
        mean_distance_km=("distance_km", "mean"),
        median_distance_km=("distance_km", "median"),
        max_distance_km=("distance_km", "max"),
    )
    out = active[["nest_id", "nest_name", "status"]].merge(
        load, on=["nest_id", "nest_name"], how="left"
    )
    for c in ["assigned_towers", "mean_distance_km", "median_distance_km", "max_distance_km"]:
        out[c] = out[c].fillna(0)
    return out.sort_values("assigned_towers", ascending=False).reset_index(drop=True)


# =========================
# 4. 巡视任务生成 ，只能生成初始化任务
def build_initial_tasks(
    towers: pd.DataFrame,
    task_types: List[str],
    as_of: pd.Timestamp,
    task_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    当前数据没有巡视历史。
    - 若 task_history=None：将选中的周期任务视为“初始化待执行任务”；
    - 红外测温不能基于当前数据自动生成，因为缺少耐张塔标识。
    - 临时任务应从外部事件单独输入。
    """
    rows = []
    task_id = 1

    for task_type in task_types:
        if task_type == TASK_IR:
            raise ValueError(
                "当前杆塔数据缺少“耐张塔/塔型”字段，不能可靠生成红外测温任务。"
                "请补充目标塔清单或 tower_type/is_tension_tower 字段。"
            )
        if task_type == TASK_TEMP:
            raise ValueError("临时任务应由外部事件/工单输入，不应对全部杆塔批量生成。")

        if task_type not in PERIODIC_TASKS:
            raise ValueError(f"未知任务类型: {task_type}")

        for r in towers.itertuples(index=False):
            rows.append({
                "task_id": f"JOB{task_id:08d}",
                "tower_id": r.tower_id,
                "tower_name": r.tower_name,
                "line_name": r.line_name,
                "tower_no": r.tower_no,
                "task_type": task_type,
                "create_time": as_of,
                # 没有历史，因此不伪造真实 deadline；初始化任务用 NaT
                "deadline": pd.NaT,
                "is_bootstrap": True,
            })
            task_id += 1

    return pd.DataFrame(rows)


# =========================
# 5. 任务到机巢的 候选机巢分配（带简单负载惩罚）
#   距离合理的前提下兼顾机巢任务/资源均衡
def assign_tasks_to_nests(
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
    load_penalty_km: float = 1.5,
) -> pd.DataFrame:
    """
    不做永久归属。每个任务从 Top-K 候选机巢中选择：
        cost = distance_km + load_penalty_km * 当前相对负载
            杆塔i到机巢j的距离 + 负载惩罚系数 * 当前机巢已经承担的任务量；
    是一个可解释、在线可更新的贪心基线算法。
    任务分配流程：
        读取某个任务
            ↓
        找到该杆塔的Top-K候选机巢
            ↓
        剔除空间不可达机巢
            ↓
     计算：
        距离成本+负载成本
            ↓
        选择综合成本最低机巢
    """
    cand_map = {
        tid: g.sort_values(["rank", "distance_km"]).to_dict("records")
        for tid, g in candidates.groupby("tower_id")
    }

    loads = {}
    assigned = []
    for task in tasks.itertuples(index=False):
        opts = cand_map.get(task.tower_id, [])
        opts = [o for o in opts if o["spatial_feasible"]]
        if not opts:
            assigned.append({
                **task._asdict(),
                "nest_id": None,
                "nest_name": None,
                "nest_distance_km": np.nan,
                "assignment_status": "NO_FEASIBLE_NEST",
            })
            continue
       
        mean_load = (sum(loads.values()) / max(len(loads), 1)) if loads else 0.0
        best = None
        best_cost = float("inf")
        for o in opts:
            load = loads.get(o["nest_id"], 0)
            relative = load / max(mean_load, 1.0)
            cost = float(o["distance_km"]) + load_penalty_km * relative
            if cost < best_cost:
                best_cost = cost
                best = o

        loads[best["nest_id"]] = loads.get(best["nest_id"], 0) + 1
        assigned.append({
            **task._asdict(),
            "nest_id": best["nest_id"],
            "nest_name": best["nest_name"],
            "nest_distance_km": best["distance_km"],
            "assignment_status": "ASSIGNED",
        })

    return pd.DataFrame(assigned)


# =========================
# 6.无人机飞行 架次构造 任务-》架次、
#  一个架次通常应该包含多个任务：  08:00从 A机巢起飞→ #31→ #32→ #33→ #34→ A机巢 这是一次完整无人机飞行。
def haversine_pair_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def estimate_route(
    nest_row,
    task_rows: pd.DataFrame,
    tower_lookup: pd.DataFrame,
    config: DispatchConfig,
) -> Tuple[float, float]:
    """
    线路连续任务的估算：
    机巢 -> 第1塔 -> 第2塔 -> ... -> 最后塔 -> 机巢
    """
    coords = []
    service_min = 0.0

    for r in task_rows.itertuples(index=False):
        t = tower_lookup.loc[r.tower_id]
        coords.append((float(t.latitude), float(t.longitude)))
        service_min += config.service_minutes.get(r.task_type, 1.0)

    if not coords:
        return 0.0, 0.0

    total_km = haversine_pair_km(
        float(nest_row.latitude), float(nest_row.longitude),
        coords[0][0], coords[0][1]
    )
    for a, b in zip(coords[:-1], coords[1:]):
        total_km += haversine_pair_km(a[0], a[1], b[0], b[1])
    total_km += haversine_pair_km(
        coords[-1][0], coords[-1][1],
        float(nest_row.latitude), float(nest_row.longitude)
    )
    # 飞行时间 + 作业时间
    flight_min = total_km / config.cruise_speed_kmh * 60.0
    total_min = flight_min + service_min
    return total_km, total_min


def build_sorties(
    assigned_tasks: pd.DataFrame,
    towers: pd.DataFrame,
    nests: pd.DataFrame,
    config: DispatchConfig,
) -> pd.DataFrame:
    """
    按 nest + line + task_type 分组，按杆塔号连续切分。
    每次尝试增加一基塔；若加入后超过最大航时/距离，则关闭当前架次。
    比如 ：当前架次为空
        加入#31
        ↓
        计算航程和时间
        ↓
        没有超限
        ↓
        继续加入#32
        ↓
        重新计算
        ↓
        没有超限
        ↓
        加入#33
        ↓
        ……
        当加入某一个杆塔以后超过限制：
    新杆塔放到下一个架次。    
    """
    tasks = assigned_tasks[assigned_tasks["assignment_status"].eq("ASSIGNED")].copy()

    tower_lookup = towers.set_index("tower_id")
    nest_lookup = nests.set_index("nest_id")

    sorties = []
    sortie_id = 1

    group_cols = ["nest_id", "nest_name", "line_name", "task_type"]

    for key, g in tasks.groupby(group_cols, sort=False):
        nest_id, nest_name, line_name, task_type = key
        g = g.sort_values(["tower_no", "tower_name"], na_position="last").reset_index(drop=True)
        nest_row = nest_lookup.loc[nest_id]

        current_idx = []

        def flush(indices):
            nonlocal sortie_id
            if not indices:
                return
            chunk = g.iloc[indices]
            # ，时间
            km, mins = estimate_route(nest_row, chunk, tower_lookup, config)
            # 预留 reserve_minutes，安全边界显式体现 现场环境影响
            usable_limit = config.max_sortie_minutes - config.reserve_minutes
            feasible = (mins <= usable_limit) and (km <= config.max_sortie_distance_km)

            sorties.append({
                "sortie_id": f"S{sortie_id:07d}",
                "nest_id": nest_id,
                "nest_name": nest_name,
                "line_name": line_name,
                "task_type": task_type,
                "task_count": len(chunk),
                "tower_ids": ",".join(chunk["tower_id"].astype(str)),
                "tower_names": ",".join(chunk["tower_name"].astype(str)),
                "route_distance_km": km,
                "estimated_minutes": mins,
                "reserve_minutes": config.reserve_minutes,
                "feasible_under_demo_params": feasible,
            })
            sortie_id += 1

        for i in range(len(g)):
            trial = current_idx + [i]
            chunk = g.iloc[trial]
            km, mins = estimate_route(nest_row, chunk, tower_lookup, config)

            usable_limit = config.max_sortie_minutes - config.reserve_minutes
            if current_idx and (
                mins > usable_limit or km > config.max_sortie_distance_km
            ):
                flush(current_idx)
                current_idx = [i]
            else:
                current_idx = trial

        flush(current_idx)

    return pd.DataFrame(sorties)


# =========================
# 7. 任务优先级 + 公平进度调度
def schedule_sorties(
    sorties: pd.DataFrame,
    start_date: pd.Timestamp,
    days: int,
    config: DispatchConfig,
) -> pd.DataFrame:
    """
    每个机巢独立排程。
    基础优先级：
        临时 > 红外 > 精细=仿线=通道
    对精细/仿线/通道加入“进度赤字”奖励，防止某一类长期没有进展。
        假设当前：
            精细化完成率 = 60%
            通道巡视完成率 = 70% 最高进度 = 70%
            仿线飞行完成率 = 20%

            精细赤字 = 70%-60%=10%
            通道赤字 = 0%
            仿线赤字 = 70%-20%=50%

    当前初始化任务无真实 deadline，因此不添加虚构的逾期/紧迫度。
    若以后接入 deadline，可在 score 中直接加入。
    """
    x = sorties[sorties["feasible_under_demo_params"]].copy()
    if x.empty:
        return pd.DataFrame()

    all_rows = []

    for nest_id, ng in x.groupby("nest_id"):
        pending = ng.to_dict("records")

        initial_count = {}
        completed_count = {}
        for t in config.fair_task_types:
            initial_count[t] = sum(1 for r in pending if r["task_type"] == t)
            completed_count[t] = 0

        current_day = 0
        minute_cursor = config.day_start_hour * 60.0
        day_end = config.day_end_hour * 60.0

        while pending and current_day < days:
            # 过滤当天还能塞下的任务
            remaining_day = day_end - minute_cursor
            candidates = [
                r for r in pending
                if r["estimated_minutes"] + config.turnaround_minutes <= remaining_day
            ]

            if not candidates:
                current_day += 1
                minute_cursor = config.day_start_hour * 60.0
                continue

            # 动态计算各任务类型完成进度
            progress = {}
            for t in config.fair_task_types:
                total = initial_count.get(t, 0)
                progress[t] = completed_count.get(t, 0) / total if total else 1.0

            fair_present = [
                t for t in config.fair_task_types if initial_count.get(t, 0) > 0
            ]
            target_progress = max(
                [progress[t] for t in fair_present], default=0.0
            )

            def score(r):
                # 1基础业务优先级
                s = BASE_PRIORITY.get(r["task_type"], 0.0)

                # 2三类同级任务：落后越多，奖励越高.
                if r["task_type"] in fair_present:
                    deficit = target_progress - progress[r["task_type"]] # 进度赤字
                    s += config.fairness_weight * max(deficit, 0.0)

                # 3同分时更偏好单位时间任务数多的架次
                eff = r["task_count"] / max(r["estimated_minutes"], 1.0)
                s += config.route_efficiency_weight * eff
                return s

            best = max(candidates, key=score)
            pending.remove(best)

            start_min = minute_cursor
            end_min = start_min + best["estimated_minutes"]

            d = start_date.normalize() + pd.Timedelta(days=current_day)
            planned_start = d + pd.Timedelta(minutes=start_min)
            planned_end = d + pd.Timedelta(minutes=end_min)

            row = dict(best)
            row["planned_start"] = planned_start
            row["planned_end"] = planned_end
            row["dispatch_score"] = score(best)
            all_rows.append(row)

            if best["task_type"] in completed_count:
                completed_count[best["task_type"]] += 1

            minute_cursor = end_min + config.turnaround_minutes

    return pd.DataFrame(all_rows)


# =========================
# 8. 输入和输出


def run_spatial(args):
    cfg = DispatchConfig(
        candidate_top_k=args.top_k,
        max_one_way_km=args.max_one_way_km,
    )
    nests = load_nests(args.nest_xlsx)
    towers = load_towers(args.tower_xlsx)

    candidates, nearest = build_candidate_nests(towers, nests, cfg)
    summary = build_spatial_summary(towers, nests, nearest)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nests.to_csv(outdir / "nests_clean.csv", index=False, encoding="utf-8-sig")
    towers.to_csv(outdir / "towers_clean.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(outdir / "tower_nest_topk.csv", index=False, encoding="utf-8-sig")
    nearest.to_csv(outdir / "tower_nearest_active_nest.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "active_nest_load_summary.csv", index=False, encoding="utf-8-sig")

    print(f"机巢总数: {len(nests)}")
    print(f"已安装机巢: {(nests['status']=='已安装').sum()}")
    print(f"杆塔总数: {len(towers)}")
    print(f"线路数: {towers['line_name'].nunique()}")
    print(f"结果目录: {outdir.resolve()}")
    print(summary.head(20).to_string(index=False))


def run_plan(args):
    cfg = DispatchConfig(
        candidate_top_k=args.top_k,
        max_one_way_km=args.max_one_way_km,
        cruise_speed_kmh=args.cruise_speed_kmh,
        max_sortie_minutes=args.max_sortie_minutes,
        max_sortie_distance_km=args.max_sortie_distance_km,
        reserve_minutes=args.reserve_minutes,
    )
    nests = load_nests(args.nest_xlsx)
    towers = load_towers(args.tower_xlsx)
    candidates, nearest = build_candidate_nests(towers, nests, cfg)

    as_of = pd.Timestamp(args.start_date)
    tasks = build_initial_tasks(
        towers=towers,
        task_types=args.task_types,
        as_of=as_of,
        task_history=None,
    )
    assigned = assign_tasks_to_nests(tasks, candidates)
    sorties = build_sorties(assigned, towers, nests, cfg)
    schedule = schedule_sorties(
        sorties=sorties,
        start_date=as_of,
        days=args.days,
        config=cfg,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks.to_csv(outdir / "bootstrap_tasks.csv", index=False, encoding="utf-8-sig")
    assigned.to_csv(outdir / "task_nest_assignment.csv", index=False, encoding="utf-8-sig")
    sorties.to_csv(outdir / "candidate_sorties.csv", index=False, encoding="utf-8-sig")
    schedule.to_csv(outdir / "demo_schedule.csv", index=False, encoding="utf-8-sig")

    print("WARNING: plan 模式使用 DEMO 飞行/服务参数，不代表真实无人机能力。")
    print(f"任务数: {len(tasks)}")
    print(f"候选架次: {len(sorties)}")
    print(f"{args.days}天内排入架次: {len(schedule)}")
    print(f"输出: {outdir.resolve()}")


def build_parser():
    p = argparse.ArgumentParser(description="机巢-杆塔空间分析与智能调度基线实现")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--nest-xlsx", required=True)
        sp.add_argument("--tower-xlsx", required=True)
        sp.add_argument("--outdir", required=True)
        sp.add_argument("--top-k", type=int, default=3)
        sp.add_argument(
            "--max-one-way-km", type=float, default=None,
            help="可选：单程空间硬半径；当前数据无真实机型参数，默认不设"
        )

    s = sub.add_parser("spatial", help="只做基于真实坐标的空间计算")
    common(s)
    s.set_defaults(func=run_spatial)

    q = sub.add_parser("plan", help="使用显式 DEMO 参数进行任务/架次/日计划仿真")
    common(q)
    q.add_argument(
        "--task-types", nargs="+", default=[TASK_CHANNEL],
        choices=[TASK_CHANNEL, TASK_DETAIL, TASK_LINE, TASK_IR, TASK_TEMP]
    )
    q.add_argument("--start-date", default=str(pd.Timestamp.today().date()))
    q.add_argument("--days", type=int, default=7)

    # DEMO 参数
    q.add_argument("--cruise-speed-kmh", type=float, default=36.0)
    q.add_argument("--max-sortie-minutes", type=float, default=45.0)
    q.add_argument("--max-sortie-distance-km", type=float, default=18.0)
    q.add_argument("--reserve-minutes", type=float, default=8.0)

    q.set_defaults(func=run_plan)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

"""
目前重点在于  一个静态任务池在当前资源条件下应该怎么分配和排序
> 首先基于机巢和杆塔经纬度建立空间关系，为每基杆塔计算多个候选执行机巢，而不是简单固定到最近机巢；
> 随后结合任务类型、线路杆塔顺序和机巢负载完成任务资源分配。对于同一线路连续杆塔，系统进一步按照无人机最大航程、最大飞行时间和安全余量自动切分为若干飞行架次，并计算每个架次的预计航程和执行时间。
> 在任务编排阶段，首先严格满足“临时任务优先于红外，红外优先于其他周期任务”的业务规则；对于精细化巡视、仿线飞行和通道巡视三类同级任务，则引入任务完成进度差异，根据落后程度动态提高调度权重，从而避免系统长期只执行某一种任务，实现多种巡视任务同步推进。
> 后续接入天气、巡视历史和实时飞行状态后，将进一步采用滚动重调度机制，在天气变化、临时任务或设备故障发生时，仅重新优化尚未执行的剩余任务，形成“自动生成任务—自动编排—自动执行—状态反馈—动态重排”的闭环调度体系。

# 缺少
    1.天气影响因子
    2.临时任务编排
    3.无人机真正调度：系统层、飞控层
 """