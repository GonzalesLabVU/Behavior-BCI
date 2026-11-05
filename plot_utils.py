import os
import random
import math
import json
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass

import pygame as pg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

import gspread
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

# ---------------------------
# BASIC CONFIG
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CREDS_PATH = SCRIPT_DIR / "credentials.json"
ANIMAL_MAP_PATH = SCRIPT_DIR / "animal_map.json"

load_dotenv(SCRIPT_DIR / ".env")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly"
    ]


# ---------------------------
# HELPERS (INTERNAL)
# ---------------------------
def _transpose(rows):
    if not rows:
        return []
    
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols == 0:
        return []
    
    cols = []
    for c in range(n_cols):
        col = []
        for r in range(n_rows):
            row = rows[r]
            col.append(row[c] if c < len(row) else "")
        
        cols.append(col)
    
    return cols


def _norm_t(ts):
    hms, ms = ts.split(".", 1)
    h, m, s = hms.split(":")

    return 3600*int(h) + 60*int(m) + int(s) + int(ms[:3])/1000.0


def _norm_date(d_str):
    m, d, y = d_str.split('/')
    return f"{int(m)}/{int(d)}/{y}"


def _date_to_day(d_str):
    d_str = d_str.strip().lstrip('0')
    dt = datetime.strptime(d_str, '%m/%d/%Y')
    
    return f"{dt.strftime('%b')}-{dt.day}"


def _parse_date(d_str):
        return datetime.strptime(d_str, '%m/%d/%Y').date()


def _require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f'{name} not found in .env')

    return value


def _cohort_tokens(map_key):
    return [token.strip() for token in str(map_key).split('_') if token.strip()]


def load_animal_map(path=ANIMAL_MAP_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError('animal_map.json must be a dict')

    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError('animal_map.json keys and values must be strings')

    return data


def get_workbook_id(animal_id, animal_map):
    animal_id = str(animal_id).strip()

    try:
        map_key = next(key for key in animal_map.keys()
                       if animal_id in _cohort_tokens(key))
    except StopIteration:
        raise ValueError(f'No cohort mapping found for animal {animal_id!r}')

    cohort_name = animal_map[map_key]
    return _require_env(f'{cohort_name}_ID')


def _show_struct(dct, depth=0, spaces=4, key_label="root"):
    pad = " " * (depth * spaces)

    if not isinstance(dct, Mapping):
        raise TypeError(f'_show_struct expects a dict-like object, got {type(dct).__name__})')
    
    if depth == 0 and key_label == "root":
        print(f'{pad}{{')
    else:
        print(f'{pad}{key_label}: {{')
    
    def _type_str(obj, _seen=None, _max_sample=100):
        if _seen is None:
            _seen = set()
        
        obj_id = id(obj)
        if obj_id in _seen:
            return "object"
        
        _seen.add(obj_id)

        if obj is None:
            return "None"
        if isinstance(obj, bool):
            return "bool"
        if isinstance(obj, int):
            return "int"
        if isinstance(obj, float):
            return "float"
        if isinstance(obj, complex):
            return "complex"
        if isinstance(obj, str):
            return "str"
        if isinstance(obj, (bytes, bytearray)):
            return type(obj).__name__
        
        if isinstance(obj, Mapping):
            items = list(obj.items())
            if not items:
                return "dict[object: object]"
            
            sample = items[:_max_sample]

            key_types = {_type_str(k, _seen, _max_sample) for k, _ in sample}
            val_types = {_type_str(v, _seen, _max_sample) for _, v in sample}

            k_t = key_types.pop() if len(key_types) == 1 else "object"
            v_t = val_types.pop() if len(val_types) == 1 else "object"

            return f'dict[{k_t}: {v_t}]'
        
        if isinstance(obj, (list, tuple, set, frozenset)):
            name_map = {
                list: "list",
                tuple: "tuple",
                set: "set",
                frozenset: "frozenset"
                }
            base = name_map[type(obj)]

            if not obj:
                return f'{base}[object]'
            
            it = list(obj)
            sample = it[:_max_sample]
            elem_types = {_type_str(e, _seen, _max_sample) for e in sample}
            elem_t = elem_types.pop() if len(elem_types) == 1 else "object"

            return f'{base}[{elem_t}]'

        if isinstance(obj, range):
            return "range[int]"
        
        return type(obj).__name__
    
    for k, v in dct.items():
        k_repr = repr(k)
        line_pad = " " * ((depth + 1) * spaces)

        if isinstance(v, Mapping):
            _show_struct(v, depth=depth+1, spaces=spaces, key_label=k_repr)
        elif v is None:
            print(f'{line_pad}{k_repr}: None')
        elif isinstance(v, (str, bytes, bytearray)):
            print(f'{line_pad}{k_repr}: {_type_str(v)}')
        else:
            print(f'{line_pad}{k_repr}: {_type_str(v)}')
    
    print(f'{pad}}}')


def _parse_header(col1, col2):
    d_tok = str(col1[0]).split('/')
    d_tok = [tok.lstrip('0') for tok in d_tok]
    m, d, y = d_tok
    
    d_str = f"{int(m)}/{int(d)}/{y}"
    a_str = str(col1[1].split(" ")[1].strip().upper())
    p_str = int(col2[1].split(" ")[1].strip())

    return d_str, a_str, p_str


def _build_col_map(raw):
    cols_by_sheet = {name: _transpose(rows) for name, rows in raw.items()}

    sheets = ("metadata", "event", "encoder")
    col_map = dict()

    def _ensure_key(key):
        if key not in col_map:
            col_map[key] = {s: None for s in sheets}
    
    for sheet in sheets:
        if sheet not in cols_by_sheet:
            continue

        cols = cols_by_sheet[sheet]
        for c in range(0, len(cols) - 1, 2):
            key = _parse_header(cols[c], cols[c+1])
            _ensure_key(key)
            col_map[key][sheet] = (c, c+1)
    
    return col_map, cols_by_sheet


def _total_disp(trial):
    if trial is None or not hasattr(trial, 'enc'):
        return 0.0
    
    vals = trial.enc.get('values', [])
    if not vals:
        return 0.0
    
    total = 0.0
    prev = None

    for v in vals:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue

        if prev is not None:
            total += abs(x - prev)
        
        prev = x
    
    return float(total)


# ---------------------------
# PLOTTING
# ---------------------------
def plot_hit_rates(trial_map):
    animals = list(trial_map.keys())
    dates = sorted(
        {d for tm in trial_map.values() for d in tm.keys()},
        key=_parse_date
        )
    
    rate_map = {a: dict() for a in animals}

    for animal, tm in trial_map.items():
        for date, trials in tm.items():
            results = [tr.result for tr in trials]
            n_hits = sum(1 for r in results if r == "hit")
            rate = (n_hits / len(results)) * 100.0

            rate_map[animal].update({date: rate})
    
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    cmap = {a: plt.get_cmap('tab10')(i % 10) for i, a in enumerate(animals)}

    for a in animals:
        rm = rate_map[a]
        last_xy = None

        for i, date in enumerate(dates):
            if date not in rm:
                continue

            ax.scatter(i - 0.5, rm[date],
                       marker='s',
                       s=105,
                       facecolors='none',
                       edgecolors=cmap[a],
                       linewidths=3.0)
            
            if last_xy is None:
                last_xy = (date, i - 0.5, rm[date])
                continue

            delta_days = (_parse_date(date) - _parse_date(last_xy[0])).days
            ls = "-" if delta_days <= 1 else (0, (3, 2))

            xs = (last_xy[1], i - 0.5)
            ys = (last_xy[2], rm[date])

            ax.plot(xs, ys,
                    linewidth=3.0,
                    linestyle=ls,
                    color=cmap[a],
                    zorder=1)
            
            last_xy = (date, i - 0.5, rm[date])
    
    avg_xy = {"date": [], "x": [], "y": []}
    
    for i, date in enumerate(dates):
        all_rates = [rm[date] for rm in rate_map.values() if date in rm]
        if not all_rates:
            continue

        avg_rate = math.exp(sum(math.log(r) for r in all_rates) / len(all_rates))

        avg_xy['date'].append(date)
        avg_xy['x'].append(i - 0.5)
        avg_xy['y'].append(avg_rate)
    
    last_xy = None
    valid_dates = {d for d in dates if any(d in rm for rm in rate_map.values())}

    for date, x, y in zip(avg_xy['date'], avg_xy['x'], avg_xy['y']):
        if date not in valid_dates:
            continue

        ax.scatter(x, y,
                   marker='s',
                   s=105,
                   facecolors='none',
                   edgecolors='black',
                   linewidths=3.0)
        
        if last_xy is None:
            last_xy = (date, x, y)
            continue

        delta_days = (_parse_date(date) - _parse_date(last_xy[0])).days
        ls = "-" if delta_days <= 1 else (0, (3, 2))

        xs = (last_xy[1], x)
        ys = (last_xy[2], y)

        ax.plot(xs, ys,
                linewidth=3.0,
                linestyle=ls,
                color='black',
                zorder=10)
        
        last_xy = (date, x, y)
    
    ax.set_xlim(-1, len(dates) - 1)
    ax.set_ylim(0, 105)

    proxy_h = []
    for a in animals:
        h = Line2D([], [],
                   linestyle='-',
                   linewidth=2.0,
                   marker='s',
                   markersize=7,
                   markerfacecolor='none',
                   markeredgecolor=cmap[a],
                   markeredgewidth=2.0,
                   color=cmap[a],
                   label=f'Animal {a}')
        proxy_h.append(h)
    
    avg_h = Line2D([], [],
                   linestyle='-',
                   linewidth=2.0,
                   marker='s',
                   markersize=7,
                   markerfacecolor='none',
                   markeredgecolor='black',
                   markeredgewidth=2.0,
                   color='black',
                   label='Geometric Mean')
    proxy_h.append(avg_h)

    lgd = ax.legend(handles=proxy_h,
                    alignment='left',
                    loc='best',
                    borderaxespad=2.0,
                    borderpad=0.5,
                    handletextpad=1.0)
    frame = lgd.get_frame()

    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    frame.set_alpha(1.0)
    for txt in lgd.get_texts():
        txt.set_fontsize(10)
    
    x_pos = [(k - 0.5) for k in range(len(dates))]
    x_labels = [_date_to_day(d) for d in dates]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels,
                       rotation=45,
                       ha='center')
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=11)
    
    ax.grid(True,
            which='major',
            axis='both',
            linewidth=0.7,
            alpha=0.3)
    
    ax.set_ylabel('Net Success Rate [%]',
                  fontsize=13)
    fig.suptitle('Success Rates Across Sessions')

    # plt.show()

    setattr(fig, 'save', lambda f: fig.savefig(f, dpi=600))
    return fig


def plot_trial_counts(trial_map):
    animals = list(trial_map.keys())
    dates = {d for tm in trial_map.values() for d in tm.keys()}

    count_map = {a: dict() for a in animals}

    for animal, tm in trial_map.items():
        for date, trials in tm.items():
            if not trials:
                count_map[animal].update({date: {"x": [], "y": []}})
                continue

            t0 = trials[0].tstart

            xs = []
            ys = []

            for n, tr in enumerate(trials, start=1):
                elapsed = (tr.tstop - t0) / 60.0

                xs.append(elapsed)
                ys.append(n)
            
            count_map[animal].update({date: {"x": xs, "y": ys}})
    
    dates_sorted = sorted((_norm_date(d) for d in dates), key=_parse_date)
    cmap = {d: plt.get_cmap('tab10')(i % 10) for i, d in enumerate(dates_sorted)}

    # max_trials = max(cm_series['y'][-1]
    #            for cm in count_map.values()
    #            for cm_series in cm.values()
    #            if cm_series['y']
    #            )
    # max_trials = ((max_trials // 50) + 1) * 50
    max_trials = 350

    fig, axes = plt.subplots(2, 2,
                             figsize=(8.5, 6.5),
                             sharex=False,
                             sharey=True)
    axes = axes.ravel()

    for ax, animal in zip(axes, animals):
        cm = count_map.get(animal, dict())

        for date, cm_series in cm.items():
            d = _norm_date(date)
            if d not in cmap:
                continue

            ax.plot(cm_series['x'], cm_series['y'],
                    linewidth=2.0,
                    color=cmap[d])
            
        proxy_h = [
            Line2D([], [],
                   linewidth=2.0,
                   color=cmap[d],
                   label=_date_to_day(d)
                   )
            for d in dates_sorted
            ]
        
        ax.legend(handles=proxy_h,
                  loc='upper left',
                  fontsize=7,
                  framealpha=1.0,
                  borderaxespad=2.0)
        
        ax.set_xlim(0, 45)
        ax.set_xticks(range(0, 46, 5))
        ax.set_xticklabels([str(xt) for xt in range(0, 46, 5)])
        ax.set_ylim(0, max_trials)
        ax.set_yticks(range(0, max_trials + 1, 50))
        ax.set_yticklabels([str(yt) for yt in range(0, max_trials + 1, 50)])
        
        ax.tick_params(axis='y',
                       labelleft=True)
        ax.grid(True,
                which='major',
                axis='both',
                linewidth=0.7,
                alpha=0.3)
            
        ax.set_xlabel('Time Elapsed [min]')
        ax.set_ylabel('Trials Completed')
        ax.set_title(f'Animal {animal}')
    
    for ax in axes[len(animals):]:
        ax.set_visible(False)
    
    fig.suptitle('Running Trial Counts Across Sessions')
    fig.subplots_adjust(hspace=0.4)

    setattr(fig, 'show', fig.show)
    setattr(fig, 'save', lambda f: fig.savefig(f, dpi=600))

    return fig


def plot_trial_rates(trial_map):
    animals = list(trial_map.keys())
    dates = {d for tm in trial_map.values() for d in tm.keys()}

    count_map = {a: dict() for a in animals}

    for animal, tm in trial_map.items():
        for date, trials in tm.items():
            if not trials:
                count_map[animal].update({date: {"x": [], "y": []}})
                continue

            t0 = trials[0].tstart

            xs = [((tr.tstop - t0) / 60.0) for tr in trials]
            ys = [tr.index for tr in trials]

            count_map[animal].update({date: {"x": xs, "y": ys}})
    
    rate_map = {a: dict() for a in animals}

    for animal, cm in count_map.items():
        for date, cm_series in cm.items():
            xs = cm_series.get('x', [])
            ys = cm_series.get('y', [])

            if len(xs) < 2 or len(ys) < 2:
                rates = [0.0] * len(xs)
            else:
                dx = [(x2 - x1) for x1, x2 in zip(xs[:-1], xs[1:])]
                dy = [(y2 - y1) for y1, y2 in zip(ys[:-1], ys[1:])]

                rates = [0.0] + [(dn / dt) if dt > 0 else float('nan') for dt, dn in zip(dx, dy)]
            
            rate_map[animal].update({date: {"x": xs, "y": rates}})
    
    dates_sorted = sorted((_norm_date(d) for d in dates), key=_parse_date)
    cmap = {d: plt.get_cmap('tab10')(i % 10) for i, d in enumerate(dates_sorted)}

    fig, axes = plt.subplots(2, 2,
                             figsize=(8.5, 6.5),
                             sharex=False,
                             sharey=True)
    axes = axes.ravel()

    for ax, animal in zip(axes, animals):
        for date, rm_series in rate_map.get(animal, dict()).items():
            d = _norm_date(date)
            if d not in cmap:
                continue

            ax.plot(rm_series['x'], rm_series['y'],
                    linewidth=1.0,
                    color=cmap[d])
        
        proxy_h = [
            Line2D([], [],
                   linewidth=2.0,
                   color=cmap[d],
                   label=_date_to_day(d)
                   )
            for d in dates_sorted
            ]
        
        ax.legend(handles=proxy_h,
                  loc='best',
                  fontsize=9,
                  framealpha=1.0)

        ax.set_xlabel('Elapsed Time [min]')
        ax.set_ylabel(r'Trial Rate [min$^{-1}$]')
        ax.set_title(f'Animal {animal}')

        ax.tick_params(axis='y',
                       labelleft=True)
        ax.grid(True,
                which='major',
                axis='both',
                linewidth=0.7,
                alpha=0.3)
    
    for ax in axes[len(animals):]:
        ax.set_visible(False)
    
    fig.subplots_adjust(hspace=0.4)
    # plt.show()

    setattr(fig, 'save', lambda f: fig.savefig(f, dpi=600))
    return fig


def plot_trial_times(trial_map):
    animals = list(trial_map.keys())
    dates = sorted(
        {d for tm in trial_map.values() for d in tm.keys()},
        key=_parse_date
        )

    duration_map = {a: dict() for a in animals}

    for animal, tm in trial_map.items():
        for date, trials in tm.items():
            xs = [tr.tstop for tr in trials]
            ys = [tr.duration for tr in trials]

            duration_map[animal].update({date: list(zip(xs, ys))})
    
    fig, axes = plt.subplots(2, 2,
                             figsize=(8.5, 6.5),
                             sharex=True,
                             sharey=True)
    axes = axes.ravel()

    cmap = {d: plt.get_cmap('tab10')(i % 10) for i, d in enumerate(dates)}

    for ax, animal in zip(axes, animals):
        dm = duration_map[animal]
        
        for date, xy_series in dm.items():
            t0 = xy_series[0][0]

            xs = [((x - t0) / 60.0) for x, _ in xy_series]
            ys = [y for _, y in xy_series]

            ax.plot(xs, ys,
                    linewidth=1.5,
                    color=cmap[date],
                    zorder=1)
            
            ax.set_xlabel('Session Time [min]',
                          fontsize=11)
            ax.set_ylabel('Trial Duration [sec]',
                          fontsize=11)
            ax.set_title(f'Animal {animal}',
                         fontsize=13,
                         fontweight='bold')
    
    proxy_h = [
        Line2D([], [],
               linewidth=1.5,
               color=cmap[d],
               label=_date_to_day(d)
               )
        for d in dates
        ]
    
    lgd = fig.legend(handles=proxy_h,
                     loc='center left',
                     bbox_to_anchor=(0.875, 0.5),
                     borderaxespad=0.0,
                     framealpha=1.0,
                     fontsize=9)
    frame = lgd.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')

    for ax in axes[len(animals):]:
        ax.set_visible(False)

    fig.suptitle('Trial Durations Across Sessions',
                 fontsize=14,
                 fontweight='normal')
    
    fig.subplots_adjust(hspace=0.3,
                        right=0.85)
    # plt.show()

    setattr(fig, 'save', lambda f: fig.savefig(f, dpi=600))
    return fig


def plot_outcomes(trial_map):
    out_map = {a: dict() for a in trial_map.keys()}

    for animal, tm in trial_map.items():
        s_out = {date: [] for date in tm.keys()}

        for date, trials in tm.items():
            s_out[date] = [tr.result for tr in trials]
        
        out_map[animal].update(s_out)
    
    animals = list(trial_map.keys())

    max_trials = max((
        len(outcomes)
        for a_map in out_map.values()
        for outcomes in a_map.values()
        ), default=0
        )
    max_trials = ((max_trials // 20) + 1) * 20

    fig, axes = plt.subplots(4, 1,
                             figsize=(8.5, 8.0),
                             sharex=False,
                             sharey=True)
    axes = axes.ravel()

    cmap = ListedColormap(['#000000', "#00dd00", "#ff0f0f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for ax, animal in zip(axes, animals):
        dates = sorted(out_map[animal].keys(), key=_parse_date)

        if not dates or max_trials == 0:
            ax.set_title(f'Animal {animal}')
            ax.text(0.5, 0.5, 'No Data',
                    ha='center',
                    va='center',
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        grid = [[0 for _ in range(len(dates))] for _ in range(max_trials)]

        for c, d in enumerate(dates):
            outcomes = out_map[animal][d]

            for r, outcome in enumerate(outcomes):
                if outcome == "hit":
                    grid[r][c] = 1
                elif outcome == "miss":
                    grid[r][c] = 2
                else:
                    grid[r][c] = 0
        
        ax.imshow(_transpose(grid),
                  cmap=cmap,
                  norm=norm,
                  origin='upper',
                  aspect='auto',
                  interpolation='nearest')
        
        ax.set_xticks(range(0, max_trials + 1, 40))
        ax.set_xticklabels([str(xt) for xt in range(0, max_trials + 1, 40)],
                           fontsize=8)
        ax.set_yticks(range(len(dates)))
        ax.set_yticklabels([_date_to_day(d) for d in dates],
                           rotation=45,
                           va='center',
                           fontsize=7)
        
        ax.set_xticks([(x - 0.5) for x in range(1, max_trials)],
                      minor=True)
        ax.set_yticks([(y - 0.5) for y in range(1, len(dates))],
                     minor=True)

        ax.tick_params(which='minor',
                       bottom=False,
                       left=False)
        ax.tick_params(axis='y',
                       labelleft=True,
                       pad=3)

        ax.set_ylabel('Trial Index',
                      fontsize=9)
        ax.set_title(f'Animal {animal}',
                     fontsize=11)
    
    for ax in axes[len(animals):]:
        ax.set_visible(False)

    setattr(fig, 'save', lambda f: fig.savefig(f, dpi=600))
    
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Outcomes by Trial Across Sessions',
                 fontsize=13)
    # plt.show()

    return fig


# ---------------------------
# TOP LEVEL
# ---------------------------
class Trial:
    def __init__(self, date, animal, phase, index):
        self.date = str(date).lstrip('0')
        self.animal = str(animal).upper()
        self.phase = int(phase)
        self.index = int(index)

        self.tstart = None
        self.tstop = None

        self._evt = {"timestamps": [], "values": []}
        self._enc = {"timestamps": [], "values": []}

    @property
    def evt(self):
        return self._evt
    
    @evt.setter
    def evt(self, vals):
        if not isinstance(vals, Mapping):
            raise TypeError('Trial.evt type must be Mapping')
        
        if not all(isinstance(k, str) for k in list(vals.keys())):
            raise TypeError('Trial.evt keys must be strings')
        if not all(isinstance(v, (list, tuple)) for v in list(vals.values())):
            raise TypeError('Trial.evt values must be lists or tuples')
        
        self.tstart = _norm_t(vals['timestamps'][0])
        self.tstop = _norm_t(vals['timestamps'][-1])

        self._evt = vals
    
    @property
    def enc(self):
        return self._enc
    
    @enc.setter
    def enc(self, vals):
        if not isinstance(vals, Mapping):
            raise TypeError('Trial.enc type must be Mapping')
        
        if not all(isinstance(k, str) for k in list(vals.keys())):
            raise TypeError('Trial.enc keys must be strings')
        if not all(isinstance(v, (list, tuple)) for v in list(vals.values())):
            raise TypeError('Trial.enc values must be lists or tuples')

        ts_list = vals.get('timestamps', [])
        if ts_list:
            self.tstart = _norm_t(ts_list[0])

        self._enc = vals
    
    @property
    def result(self):
        return self.evt['values'][-1]

    @property
    def duration(self):
        return float(self.tstop - self.tstart)


def load_workbook(workbook_id):
    creds = Credentials.from_service_account_file(str(CREDS_PATH), scopes=SCOPES)
    client = gspread.authorize(creds)

    wb = client.open_by_key(workbook_id)

    data = dict()
    for ws in wb.worksheets():
        data[ws.title.lower()] = ws.get_all_values()
    
    return data


def extract_sessions(raw, target_dates=None):
    col_map, cols_by_sheet = _build_col_map(raw)

    col_map = {
        (_norm_date(k[0]), k[1], k[2]): v
        for k, v in col_map.items()
        }
    cols_by_sheet = {
        sheet: [c[2:] for c in cols]
        for sheet, cols in cols_by_sheet.items()
        }
    
    all_keys = list(col_map.keys())

    if target_dates is not None:
        target_keys = [k for k in all_keys if k[0] in target_dates]
    else:
        target_keys = all_keys

    target_sessions = {
        key: {
            "metadata": {"keys": [], "values": []},
            "event": {"timestamps": [], "values": []},
            "encoder": {"timestamps": [], "values": []}
            }
        for key in target_keys
        }
    
    def _parse_kv(col1, col2):
        out = dict()
        current_key = None
        buf = []

        def _flush():
            nonlocal current_key, buf
            if current_key is None:
                return
            
            vals = [v for v in buf if v not in {"", None}]
            if not vals:
                out[current_key] = []
            elif len(vals) == 1:
                out[current_key] = vals[0]
            else:
                out[current_key] = vals
            
            buf = []
        
        for k, v in zip(col1, col2):
            k, v = ("" if x is None else str(x).strip() for x in (k, v))
            try:
                v = int(v)
            except Exception:
                pass

            if k:
                _flush()
                current_key = k
            
            if current_key is not None and v not in {"", None}:
                buf.append(v)
        
        _flush()

        return list(out.keys()), list(out.values())
    
    def _parse_tseries(col1, col2):
        ts = []
        vals = []

        for t, v in zip(col1, col2):
            if t in {"", None} and v in {"", None}:
                continue

            if t not in {"", None}:
                ts.append(t)
            if v not in {"", None}:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(v)
        
        return ts, vals

    def _set_default(key, sheet):
        if sheet == "metadata":
            target_sessions[key][sheet]['keys'] = []
            target_sessions[key][sheet]['values'] = []
        else:
            target_sessions[key][sheet]['timestamps'] = []
            target_sessions[key][sheet]['values'] = []

    for key, map in col_map.items():
        if key not in target_sessions.keys():
            continue

        if map['metadata']:
            c1, c2 = map['metadata']

            col1 = cols_by_sheet['metadata'][c1]
            col2 = cols_by_sheet['metadata'][c2]

            k, v = _parse_kv(col1, col2)

            target_sessions[key]['metadata']['keys'] = k
            target_sessions[key]['metadata']['values'] = v
        else:
            _set_default(key, 'metadata')
        
        for sheet in ("event", "encoder"):
            if map[sheet]:
                c1, c2 = map[sheet]

                col1 = cols_by_sheet[sheet][c1]
                col2 = cols_by_sheet[sheet][c2]

                k, v = _parse_tseries(col1, col2)

                target_sessions[key][sheet]['timestamps'] = k
                target_sessions[key][sheet]['values'] = v
            else:
                _set_default(key, sheet)
    
    return target_sessions


def parse_trials(key, data):
    date, animal, phase = key

    evt = data['event']
    enc = data['encoder']
    meta = data['metadata']

    start_idx = [i for i, e in enumerate(evt['values']) if e == "cue"]
    stop_idx = [i + 1 for i, e in enumerate(evt['values']) if e in {"hit", "miss"}]

    trials = []

    for n, (k1, k2) in enumerate(zip(start_idx, stop_idx), start=1):
        tr = Trial(date, animal, phase, index=n)

        ts = evt['timestamps'][k1:k2]
        vals = evt['values'][k1:k2]

        tr.evt = {"timestamps": ts, "values": vals}
        
        trials.append(tr)
    
    from bisect import bisect_left, bisect_right

    enc_ts = enc['timestamps']
    enc_vals = enc['values']
    enc_t = [_norm_t(t) for t in enc_ts]

    for tr in trials:
        k1 = bisect_right(enc_t, tr.tstart)
        k2 = bisect_left(enc_t, tr.tstop)

        ts = enc_ts[k1:k2]
        vals = enc_vals[k1:k2]

        tr.enc = {"timestamps": ts, "values": vals}

    return trials


def print_session_summary(sessions, animals):
    sessions_by_animal = {animal: [] for animal in animals}

    for date, animal, phase in sessions.keys():
        if animal in sessions_by_animal:
            sessions_by_animal[animal].append((date, int(phase)))

    for animal in animals:
        print()
        print(f'Animal {animal}')

        ordered_sessions = sorted(
            sessions_by_animal[animal],
            key=lambda session: (_parse_date(session[0]), session[1])
            )
        for session in ordered_sessions:
            print(f'    {session}')

        print()


def process_sessions(sessions, animals):
    trial_map = {a: dict() for a in animals}
    session_map = {a: dict() for a in animals}

    for key, data in sessions.items():
        date, animal, phase = key
        if phase < 3:
            continue

        trials = parse_trials(key, data)
        
        trial_map[animal].update({date: trials})
        session_map[animal].update({date: data})
    
    return trial_map, session_map


class Text:
    name = "calibri"
    size = 32
    color = "#000000"
    valid_anchors = {"center",
                     "topleft",
                     "bottomleft",
                     "topright",
                     "bottomright"}

    def __init__(self, pos=(0, 0), anchor="center", value="0", bold=False):
        self.font = pg.font.SysFont(self.name, self.size, bold=bold)
        self.visible = True

        self.text = None
        self.bbox = None
        self.rect = None
        self.blit_pos = (0, 0)
        
        self._pos = None
        self._anchor = None
        self._value = None

        self.anchor = anchor
        self.pos = pos
        self.value = value

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos
        self._update_layout()

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, new_anchor):
        if new_anchor not in self.valid_anchors:
            raise ValueError(f'anchor must be one of {sorted(self.valid_anchors)}')
        
        self._anchor = new_anchor
        self._update_layout()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = str(new_value)
        self.text = self.font.render(self._value, True, pg.Color(self.color))
        self._update_layout()

    def _update_layout(self):
        if self.text is None or self._pos is None or self._anchor is None:
            return
        
        self.bbox = self.text.get_bounding_rect()

        self.rect = self.bbox.copy()
        setattr(self.rect, self.anchor, self.pos)

        self.blit_pos = (self.rect.x - self.bbox.x,
                         self.rect.y - self.bbox.y)

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def draw(self, surface=None):
        if not self.visible or self.text is None:
            return
        
        if surface is None:
            surface = pg.display.get_surface()

        surface.blit(self.text, self.blit_pos)


class Panel:
    fill_map = {True: pg.Color("#9ef4ff"),
                False: pg.Color("#ffffff")}

    def __init__(self, position):
        x, y, w, h = position

        self.rect = pg.Rect(x, y, w, h)
        self.visible = True

        self.border_color = pg.Color("#000000")
        self.border_width = 3

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def draw(self, selected=False):
        if not self.visible:
            return

        screen = pg.display.get_surface()
        bg_color = self.fill_map[selected]

        pg.draw.rect(screen, bg_color, self.rect)
        pg.draw.rect(screen, self.border_color, self.rect, self.border_width)


class DateCell:
    def __init__(self, parent, position, value=0):
        
        x, y, w, _ = position
        pad = 10

        label_x = x + w - pad
        label_y = y + pad

        self.parent = parent
        self.position = position
        self.visible = True
        self.selected = False

        self.panel = Panel(position)
        self.rect = self.panel.rect
        self.label = Text(pos=(label_x, label_y),
                          anchor="topright",
                          value=str(value),
                          bold=True)

    def show(self):
        self.visible = True
        self.panel.show()

    def hide(self):
        self.visible = False
        self.panel.hide()

    def draw(self):
        self.panel.draw(self.selected)
        self.label.draw()


class Calendar:
    def __init__(self, trial_map):
        pg.init()
        pg.font.init()

        WIN_WIDTH, WIN_HEIGHT = 700, 600
        self.window = pg.display.set_mode((WIDTH, HEIGHT))

        CELL_SZ = 100
        grid = []
        for x in range(0, WIN_HEIGHT + 1, CELL_SZ):
            row = []

            for y in range(0, WIN_WIDTH + 1, CELL_SZ):
                cell = DateCell(parent=self.window,
                                position=(x, y, CELL_SZ, CELL_SZ))
                row.append(cell)
            
            grid.append(row)
        
        self.grid = grid

        # self.clock = 

    def _make_grid(self, first_day="sun"):
        # first = first date cell object / first date's weekday
        pass


if __name__ == "__main__":
    os.system('cls')

    pg.init()
    pg.font.init()

    WIDTH, HEIGHT = 1200, 600
    window = pg.display.set_mode((WIDTH, HEIGHT))

    cells = []
    colors = ("#ffffff", "#80e1ff", "#bdbdbd")
    i = 1

    for y in range(0, HEIGHT, 200):
        for x in range(0, WIDTH, 200):
            cell = DateCell(parent=window,
                            position=(x, y, 200, 200),
                            value=int(i))
            cells.append(cell)
            i += 1

    clock = pg.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(60) / 1000.0

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            
            if event.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                all_clicked = [cell.panel.rect.collidepoint(x, y)
                               for cell in cells]
                
                for cell, clicked in zip(cells, all_clicked):
                    if clicked:
                        cell.selected = not cell.selected

        window.fill('#ffffff')

        for cell in cells:
            cell.draw()

        pg.display.update()
