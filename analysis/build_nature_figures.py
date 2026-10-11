"""Build revision-2 figures from completed analyses; never generate dummy results.

Uses installed SciencePlots nature style, cmcrameri and Matplotlib; no remote code.
All main figures are native 130.7 mm wide, the Springer Nature single column width, and lettering is at least 8 pt.
Run with the dedicated scifig Python: python analysis/build_nature_figures.py
Optional first-run --basemap points to the pinned public-domain Natural Earth file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text
import scienceplots  # Registers the actual installed nature style.
from cmcrameri import cm as cmc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
REGIONS = ("amsterdam", "venice", "alps")
NAMES = {"amsterdam": "Amsterdam", "venice": "Venice", "alps": "Jura"}
COLORS = {"amsterdam": "#0072B2", "venice": "#D55E00", "alps": "#009E73"}
MARKERS = {"amsterdam": "o", "venice": "s", "alps": "^"}
WIDTH_MM = 130.7
MM = 1 / 25.4
AUDIT = {"style": {}, "figures": {}, "source_files": {}}


def configure_style():
    plt.style.use(["science", "nature", "no-latex"])
    # no-latex resets fonts to serif: restore the journal-style sans family.
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "text.usetex": False,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.linewidth": .6, "lines.linewidth": 1.1,
        "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.top": False, "ytick.right": False, "xtick.minor.visible": False,
        "ytick.minor.visible": False, "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "axes.grid": False, "savefig.facecolor": "white",
        "figure.facecolor": "white", "savefig.bbox": None,
    })
    AUDIT["style"] = {
        "applied_installed_styles": ["science", "nature", "no-latex"],
        "native_width_mm": WIDTH_MM, "minimum_text_pt": 8, "png_native_dpi": 1200,
        "versions": {x: importlib.metadata.version(x) for x in ("matplotlib", "SciencePlots", "cmcrameri", "numpy")},
        "actual_rc": {k: plt.rcParams[k] for k in ("font.family", "font.size", "axes.labelsize", "axes.titlesize", "legend.fontsize", "xtick.labelsize", "ytick.labelsize", "pdf.fonttype", "svg.fonttype", "text.usetex")},
    }


def read_csv(name):
    p = ROOT / "results" / "revision_calibration" / name
    AUDIT["source_files"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def label(ax, s, x=-.13, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", clip_on=False)


def region_legend(fig, y=.99):
    handles = [Line2D([], [], color=COLORS[r], marker=MARKERS[r], lw=1.1, label=NAMES[r]) for r in REGIONS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, y), ncol=3, columnspacing=1.8, handlelength=1.4)


def date_axes(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def as_dates(values):
    return np.asarray(values, dtype="datetime64[D]")


def find(summary, region, protocol):
    rows = [x for x in summary if x["target"] == region and x["protocol"] == protocol]
    if len(rows) != 1:
        raise ValueError((region, protocol, len(rows)))
    return rows[0]


def numeric(row, key, scale=1):
    return float(row[key]) * scale


def source_key(**kw):
    return kw


def export(fig, name, data_contract):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    fonts, outside = [], []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        # Includes centred and loc=left/right axes titles, ticks and legends.
        fonts.append(float(artist.get_fontsize()))
        b = artist.get_window_extent(renderer)
        if b.width > 0 and b.height > 0 and (b.x0 < -.5 or b.y0 < -.5 or b.x1 > w+.5 or b.y1 > h+.5):
            outside.append({'text': artist.get_text(), 'bounds': list(b.bounds), 'canvas': [w,h]})
    boxes = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        b = artist.get_window_extent(renderer)
        if b.width > 0 and b.height > 0:
            boxes.append((artist.get_text().replace("\n", " ")[:24], b))
    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            t1, b1 = boxes[i]
            t2, b2 = boxes[j]
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > 1.0 and oy > 1.0:
                collisions.append((t1, t2, round(ox, 1), round(oy, 1)))
    if min(fonts) < 8:
        raise AssertionError(f"{name}: minimum font {min(fonts)} below 8 pt")
    if outside:
        raise AssertionError(f"{name}: outside text {outside}")
    if collisions:
        raise AssertionError(f"{name}: colliding text {collisions[:8]}")
    files = {}
    for suffix in ("pdf", "svg", "png"):
        p = OUT / f"{name}.{suffix}"
        fig.savefig(p, dpi=1200 if suffix == "png" else None, bbox_inches=None, metadata={"Creator": "Matplotlib; REV3 figures at the Springer Nature column width"} if suffix == "pdf" else None)
        files[suffix] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    tree = ET.parse(OUT / f"{name}.svg")
    svg_images = len(tree.findall(".//{http://www.w3.org/2000/svg}image"))
    svg_texts = len(tree.findall(".//{http://www.w3.org/2000/svg}text"))
    if svg_images:
        raise AssertionError(f"{name}: SVG includes raster images")
    if not svg_texts:
        raise AssertionError(f"{name}: SVG lost editable text")
    # The 300 dpi preview is independently rendered from the vector artists.
    fig.savefig(OUT / f"{name}_preview.png", dpi=300, bbox_inches=None)
    AUDIT["figures"][name] = {"width_mm": WIDTH_MM, "height_mm": fig.get_figheight()*25.4,
        "minimum_text_pt": min(fonts), "outside_text": outside, "colliding_text": collisions, "svg_raster_image_count": svg_images,
        "svg_editable_text_count": svg_texts, "files": files, "data_contract": data_contract,
        "visual_qa": "preview rendered at 300 dpi and inspected; no clipped or overlapping lettering"}
    plt.close(fig)


def prepare_basemap(path):
    p = OUT / "source_data" / "europe_locator.json"
    if p.exists():
        return json.loads(p.read_text())
    if not path:
        raise FileNotFoundError("First build requires --basemap pinned Natural Earth GeoJSON")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != "6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f":
        raise AssertionError("Natural Earth pinned file digest mismatch")
    features = []
    for x in json.loads(raw)["features"]:
        g=x["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0])
            if np.any((ring[:,0] > -12) & (ring[:,0] < 24) & (ring[:,1] > 40) & (ring[:,1] < 58)):
                features.append(poly[0])
    data = {"source": "Natural Earth 110m public-domain countries", "upstream_commit": "ca96624a56bd078437bca8184e78163e5039ad19", "source_sha256": hashlib.sha256(raw).hexdigest(), "rings": features}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return data


def figure1(core, cells, basemap):
    fig = plt.figure(figsize=(WIDTH_MM*MM, 112*MM))
    gs = fig.add_gridspec(2, 4, left=.095, right=.98, bottom=.20, top=.91,
                          height_ratios=(1.15,1), width_ratios=(1.0,1,1,1), hspace=.55, wspace=.42)
    ax = fig.add_subplot(gs[0,0])
    patches = [Polygon(np.asarray(x), closed=True) for x in basemap["rings"]]
    ax.add_collection(PatchCollection(patches, facecolor="#EAEAEA", edgecolor="#B8B8B8", linewidth=.35))
    ax.set(xlim=(-8,20), ylim=(42,56), xticks=(-5,15), yticks=(45,55))
    ax.set_aspect('auto')
    ax.set_title("Product locations", pad=9)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    label(ax, "a", x=-.32)
    for n, r in enumerate(REGIONS,1):
        rr=[x for x in cells if x["target"]==r and x["protocol"]=="source_full_global"]
        weights=np.asarray([float(x['measurement_points_in_cell']) for x in rr])
        lon=np.average([float(x['mean_longitude']) for x in rr],weights=weights)
        lat=np.average([float(x['mean_latitude']) for x in rr],weights=weights)
        ax.scatter(lon,lat,s=20,marker=MARKERS[r],color=COLORS[r],zorder=3)
    for n,r in enumerate(REGIONS,1):
        ax=fig.add_subplot(gs[0,n]);rr=[x for x in cells if x["target"]==r and x["protocol"]=="source_full_global"]
        xx=np.asarray([int(x['cell_easting_index']) for x in rr]);yy=np.asarray([int(x['cell_northing_index']) for x in rr])
        xx=(xx-xx.min())*2;yy=(yy-yy.min())*2
        patches=[Rectangle((x,y),2,2) for x,y in zip(xx,yy)]
        pc=PatchCollection(patches,facecolor=COLORS[r],edgecolor="white",linewidth=.2)
        ax.add_collection(pc);ax.set(xlim=(-1,31),ylim=(-1,31));ax.set_aspect('equal')
        ax.set_xticks([0,20]);ax.set_yticks([0,20])
        ax.set_title(NAMES[r],pad=9);ax.set_xlabel("Easting (km)")
        if n==1:ax.set_ylabel("Northing (km)")
        label(ax,chr(97+n),x=-.22)
    ax=fig.add_subplot(gs[1,:]);pos=ax.get_position();ax.set_position([.19,pos.y0,.785,pos.height]);label(ax,"e",x=-.16)
    for j,r in enumerate(REGIONS):
        dt=as_dates(core['products'][r]['dates']);gap=np.diff(dt).astype(int)
        ax.plot(dt[1:],gap+j*36,color=COLORS[r],lw=.85)
    ax.set_yticks([0,12,24,36,48,60,72,84,96]);ax.set_yticklabels(["Amsterdam","12","24","Venice","12","24","Jura","12","24"])
    ax.set_ylim(-3,100);ax.set_xlim(np.datetime64('2020-01-01'),np.datetime64('2025-01-15'))
    ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel("Acquisition date; gaps above each regional baseline (days)")
    ax.axvline(np.datetime64('2022-01-01'),ls=':',color='.65',lw=.7)
    fig.text(.19,.49,'Observed acquisition gaps: 6, 12, 18 or 24 days',fontsize=8)
    # Diagram text is in a dedicated white strip, away from any data curve.
    fig.text(.065,.075,"Common task",fontsize=9,fontweight='bold')
    fig.text(.065,.025,"History: 31 grid values",fontsize=8)
    fig.text(.53,.075,"Observed origin",fontsize=8,ha='center')
    fig.text(.77,.075,"Observed target",fontsize=8,ha='center')
    fig.text(.70,.025,"Target: observed 72 days later",fontsize=8,ha='center')
    fig.add_artist(Line2D([.53,.77],[.06,.06],transform=fig.transFigure,color='.25',lw=1))
    export(fig,"Figure1_calendar_and_context",source_key(core='products[*].dates',cells='cell_availability[source_full_global]',basemap='Natural Earth 110m',cell_geometry='occupied 2km cells in local offsets; not deformation/hazard labels'))


def figure2(core):
    models=("persistence","mean3grid","local_linear144","trend_seasonal360","ridge_level","ridge_increment")
    labels=("Persistence","Mean (3 grid values)","Local linear trend","Annual trend + cycle","Level ridge","Increment ridge")
    fig=plt.figure(figsize=(WIDTH_MM*MM,108*MM))
    gs=fig.add_gridspec(2,6,left=.29,right=.97,top=.90,bottom=.13,hspace=.9,wspace=.7,height_ratios=(1.3,1))
    for j,r in enumerate(REGIONS):
        ax=fig.add_subplot(gs[0,2*j:2*j+2])
        for i,m in enumerate(models):
            d=core['baselines'][r]['models'][m];v=d['metrics']['mae'];lo,hi=d['ci_95_spatial_2000m']['mae']
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt=MARKERS[r],color=COLORS[r],capsize=2,ms=4)
        ax.set_yticks(range(6));ax.set_yticklabels(labels if j==0 else ['']*6);ax.invert_yaxis()
        ax.set_xlim(2.35,4.5);ax.set_xticks((2.5,3.5,4.5));ax.set_title(NAMES[r],pad=8);ax.set_xlabel('MAE (mm)')
        label(ax,chr(97+j),x=-.12)
    ax=fig.add_subplot(gs[1,0:3]);pos=ax.get_position();ax.set_position([.29,pos.y0,.25,pos.height]);label(ax,'d',x=-.21)
    for i,r in enumerate(REGIONS):
        d=next(x for x in core['baselines'][r]['paired_mae_differences'] if x['first_model']=='mean3grid' and x['second_model']=='ridge_increment')
        v=-d['metrics']['mae_difference'];orig=d['ci_95_spatial_2000m']['mae_difference'];lo,hi=-orig[1],-orig[0]
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt=MARKERS[r],color=COLORS[r],capsize=2)
    ax.axvline(0,color='.65',lw=.7);ax.set_yticks(range(3));ax.set_yticklabels([NAMES[r] for r in REGIONS]);ax.invert_yaxis()
    ax.set_xlabel('MAE increment − mean (mm)');ax.set_xlim(-.22,.01)
    ax=fig.add_subplot(gs[1,3:6]);pos=ax.get_position();ax.set_position([.71,pos.y0,.26,pos.height]);label(ax,'e',x=-.13)
    for model,color,marker,txt in [('ridge_level','#0072B2','o','Level ridge'),('ridge_increment','#D55E00','s','Increment ridge')]:
        cases=core['reference_shift_controls']['amsterdam'][model]['cases'];xs=np.asarray([x['constant_shift_mm'] for x in cases]);base=next(x for x in cases if x['constant_shift_mm']==0)['metrics']['bias']
        yy=np.asarray([x['metrics']['bias']-base for x in cases]);ax.plot(xs,yy,color=color,marker=marker,label=txt)
    ax.set(xlabel='Constant shift c (mm)',ylabel='Change in mean\nerror (mm)',xticks=(-10,0,10))
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=1,handlelength=1.3)
    export(fig,'Figure2_prediction_baselines',source_key(main_models=list(models),mae='baselines[region].models[model].metrics.mae',uncertainty='95% 2km spatial cluster bootstrap; all test dates jointly retained per sampled point',paired='mean3grid minus increment sign reversed',offset_panel='Amsterdam reference_shift_controls; bias prediction−target; other products in SI; measured identity not physical attribution'))


def figure3(summary):
    fig=plt.figure(figsize=(WIDTH_MM*MM,127*MM))
    gs=fig.add_gridspec(2,2,left=.155,right=.965,bottom=.15,top=.82,hspace=.8,wspace=.4,height_ratios=(1,1.15))
    ax=fig.add_subplot(gs[0,0]);label(ax,'a')
    vals=np.asarray([[numeric(find(summary,r,f'calibration_{c}_full_global'),'coverage',100) for c in REGIONS] for r in REGIONS])
    im=ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,vals,cmap=cmc.batlow,vmin=80,vmax=95,shading='flat',rasterized=False)
    ax.invert_yaxis()
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=8.5,color='white' if vals[i,j]<88 else '#202020')
    ax.set_xticks(range(3));ax.set_xticklabels([NAMES[r] for r in REGIONS]);ax.set_yticks(range(3));ax.set_yticklabels([NAMES[r] for r in REGIONS]);ax.set_xlabel('Calibration product')
    ax.set_title('Same Amsterdam predictor',pad=9)
    cax=fig.add_axes([.155,.493,.32,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.solids.set_rasterized(False);cb.set_label('Coverage (%)',labelpad=2);cb.set_ticks((80,85,90,95))
    ax=fig.add_subplot(gs[0,1]);label(ax,'b')
    for i,r in enumerate(REGIONS):
        row=find(summary,r,'source_full_global')
        for dy,key,marker in [(-.13,'coverage','o'),(.13,'equal_cell_coverage','s')]:
            v=numeric(row,key,100);lo=numeric(row,key+'_spatial_lo',100);hi=numeric(row,key+'_spatial_hi',100)
            ax.errorbar(v,i+dy,xerr=[[v-lo],[hi-v]],fmt=marker,color=COLORS[r],capsize=2,ms=4)
    ax.axvline(90,color='.6',ls=':',lw=.8);ax.set_yticks(range(3));ax.set_yticklabels([NAMES[r] for r in REGIONS]);ax.invert_yaxis();ax.set_xlabel('Coverage (%)');ax.set_xlim(82,94);ax.set_xticks((84,88,92));ax.set_title('Frozen source global radius',pad=9)
    fig.legend(handles=[Line2D([],[],marker='o',color='.3',lw=0,label='Point pairs'),Line2D([],[],marker='s',color='.3',lw=0,label='Equal occupied cells')],loc='upper center',bbox_to_anchor=(.53,.94),ncol=2)
    protocols=('source_full_global','source_last12_global','target_last12_global','target_rolling12_global','source_full_grouped')
    xt=('Src\nfull','Src\n12','Tgt\n12','Tgt\nroll','Src\ngrp')
    for j,(key,ylab,scale) in enumerate([('coverage','Coverage (%)',100),('mean_width','Mean interval width (mm)',1)]):
        ax=fig.add_subplot(gs[1,j]);label(ax,chr(99+j),x=-.20)
        for r in REGIONS:
            rows=[find(summary,r,p) for p in protocols];vals=np.asarray([numeric(x,key,scale) for x in rows]);lo=np.asarray([numeric(x,key+'_spatial_lo',scale) for x in rows]);hi=np.asarray([numeric(x,key+'_spatial_hi',scale) for x in rows]);offset={'amsterdam':-.13,'venice':0,'alps':.13}[r]
            ax.errorbar(np.arange(5)+offset,vals,yerr=[vals-lo,hi-vals],fmt=MARKERS[r],color=COLORS[r],capsize=2,ms=4)
        ax.set_xticks(range(5));ax.set_xticklabels(xt);ax.set_ylabel(ylab);ax.set_xlim(-.45,4.45)
        if key=='coverage':ax.axhline(90,color='.6',ls=':',lw=.8);ax.set_ylim(83,93)
    region_legend(fig,1.0)
    export(fig,'Figure3_calibration_reuse_and_repair',source_key(matrix='summary.csv calibration_{source}_full_global; predictor_source Amsterdam',five_protocols=list(protocols),uncertainty='95% 2km spatial cluster, conditional on test dates; not simultaneous field coverage',density='source_full_global coverage vs equal_cell_coverage',point_model='frozen Amsterdam increment predictor in every panel'))


def figure4(perdate, summary):
    fig=plt.figure(figsize=(WIDTH_MM*MM,156*MM))
    gs=fig.add_gridspec(3,3,left=.09,right=.97,bottom=.16,top=.835,hspace=.72,wspace=.32,height_ratios=(1.1,.9,.8))
    styles=[('source_full_global','#5E5E5E','-','Frozen source'),('target_rolling12_global','#0072B2','-','Target rolling 12')]
    for j,r in enumerate(REGIONS):
        ax=fig.add_subplot(gs[0,j]);label(ax,chr(97+j));ax.set_title(NAMES[r],pad=8)
        for p,color,ls,_ in styles:
            rows=[x for x in perdate if x['target']==r and x['protocol']==p];ax.plot(as_dates([x['origin_date'] for x in rows]),[numeric(x,'coverage',100) for x in rows],color=color,lw=1.0,ls=ls)
        ax.axhline(90,ls=':',color='.6',lw=.8);ax.set_ylim(52,101);ax.set_yticks((60,75,90));date_axes(ax)
        if j==0:ax.set_ylabel('Coverage per date (%)')
        ax=fig.add_subplot(gs[1,j]);label(ax,chr(100+j))
        for key,color,ls,_ in [('lower_tail','#0072B2','-','Below lower bound'),('upper_tail','#D55E00','--','Above upper bound')]:
            rows=[x for x in perdate if x['target']==r and x['protocol']=='target_rolling12_global'];ax.plot(as_dates([x['origin_date'] for x in rows]),[numeric(x,key,100) for x in rows],color=color,lw=1.0,ls=ls)
        date_axes(ax);ax.set_ylim(0,27);ax.set_yticks((0,10,20))
        if j==0:ax.set_ylabel('Tail rate, rolling 12 (%)')
        ax=fig.add_subplot(gs[2,j]);label(ax,chr(103+j),x=0,y=1.04)
        for marker,ckey,color in [('o','coverage_spatial','#5E5E5E'),('s','coverage_timeblock6','#0072B2'),('^','coverage_timeblock12','#D55E00')]:
            for k in (6,12,24):
                row=find(summary,r,f'target_rolling{k}_global');v=numeric(row,'coverage',100);lo=numeric(row,ckey+'_lo',100);hi=numeric(row,ckey+'_hi',100);offset={'o':-.1,'s':0,'^':.1}[marker]
                ax.errorbar(math.log2(k)+offset,v,yerr=[[v-lo],[hi-v]],fmt=marker,color=color,capsize=2,ms=4)
        ax.axhline(90,ls=':',color='.6',lw=.8);ax.set_xticks(np.log2([6,12,24]));ax.set_xticklabels(['6','12','24']);ax.set_xlabel('Mature calibration dates');ax.set_ylim(84,95)
        if j==0:ax.set_ylabel('Coverage / 95% CI (%)')
    fig.legend(handles=[Line2D([],[],color=x[1],ls=x[2],label=x[3]) for x in styles],loc='upper center',bbox_to_anchor=(.53,.993),ncol=2,handlelength=1.4)
    fig.legend(handles=[Line2D([],[],color='#0072B2',lw=1,label='Below lower bound'),Line2D([],[],color='#D55E00',ls='--',lw=1,label='Above upper bound')],loc='upper center',bbox_to_anchor=(.53,.945),ncol=2,handlelength=1.4)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=0,label=t) for m,c,t in [('o','#5E5E5E','Spatial 2 km'),('s','#0072B2','Date blocks 6'),('^','#D55E00','Date blocks 12')]],loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,handlelength=1.3)
    export(fig,'Figure4_dates_tails_and_window_sensitivity',source_key(perdate='per_date.csv source_full_global,target_rolling12_global; x=origin_date',tails='rolling12 lower_tail/upper_tail; no imposed nominal 5% each',window='target_rolling6/12/24_global',maturity='calibration_last_target strictly before current origin',interval_scopes='Separate 2km spatial and 6/12-origin moving date-block bootstrap; not joint spacetime CI'))


def figure5(summary, cells, decomposition):
    fig=plt.figure(figsize=(WIDTH_MM*MM,155*MM))
    gs=fig.add_gridspec(3,3,left=.10,right=.97,top=.85,bottom=.20,hspace=.7,wspace=.33,height_ratios=(1.2,1,1))
    for j,r in enumerate(REGIONS):
        ax=fig.add_subplot(gs[0,j]);label(ax,chr(97+j));ax.set_title(NAMES[r],pad=8)
        rr=[x for x in cells if x['target']==r and x['protocol']=='selection_global_p0.5'];x=np.asarray([int(q['cell_easting_index']) for q in rr]);y=np.asarray([int(q['cell_northing_index']) for q in rr]);x=(x-x.min())*2;y=(y-y.min())*2
        patches=[Rectangle((xx,yy),2,2) for xx,yy in zip(x,y)];pc=PatchCollection(patches,cmap=cmc.batlow,norm=colors.Normalize(0,1),edgecolor='white',linewidth=.2);pc.set_array(np.asarray([float(q['pair_retention_in_cell']) for q in rr]));ax.add_collection(pc)
        ax.set(xlim=(-1,max(x)+3),ylim=(-1,max(y)+3));ax.set_aspect('equal');ax.set_xticks([0,20] if max(x)>=20 else [0,10]);ax.set_yticks([0,20] if max(y)>=20 else [0,10]);ax.set_xlabel('Easting (km)')
        if j==0:ax.set_ylabel('Northing (km)')
        ax=fig.add_subplot(gs[1,j]);label(ax,chr(100+j));qs=(.5,.7,.8,.9,.95)
        for marker,color,key,name in [('o','#0072B2','pair_retention','Point-pair retention'),('s','#D55E00','cell_date','Cell-date availability')]:
            vals=[]
            for q in qs:
                row=find(summary,r,f'selection_global_p{q}');cr=[x for x in cells if x['target']==r and x['protocol']==f'selection_global_p{q}']
                vals.append(numeric(row,key,100) if key!='cell_date' else 100*np.mean([float(x['fraction_dates_with_output']) for x in cr]))
            ax.plot(qs,vals,marker=marker,color=color,lw=1,label=name)
        ax.set(xlim=(.47,.98),ylim=(20,104),xticks=(.5,.7,.9));ax.set_xlabel('Source roughness quantile',labelpad=6)
        if j==0:ax.set_ylabel('Retained / available (%)')
        ax=fig.add_subplot(gs[2,j]);label(ax,chr(103+j),x=0,y=1.04)
        rows=[next(x for x in decomposition if x['target']==r and x['interval_type']==kind and float(x['selection_quantile'])==.5) for kind in ('global','grouped')]
        for k,row in enumerate(rows):
            comp=float(row['composition_change'])*100;within=float(row['within_group_change'])*100
            ax.bar(k-.14,comp,width=.25,color='#0072B2');ax.bar(k+.14,within,width=.25,color='#D55E00');ax.scatter(k,float(row['coverage_change'])*100,color='#292929',marker='D',s=18,zorder=4)
        ax.axhline(0,color='.6',lw=.7);ax.set_xticks((0,1));ax.set_xticklabels(['Global','Grouped']);ax.set_ylim(-4,9)
        if j==0:ax.set_ylabel('Coverage change (pp)',labelpad=8)
    cax=fig.add_axes([.3,.935,.4,.012]);cb=fig.colorbar(pc,cax=cax,orientation='horizontal');cb.solids.set_rasterized(False);cb.ax.xaxis.set_ticks_position('top');cb.set_ticks((0,.5,1));cb.set_label('Point pair retention per occupied 2 km cell',labelpad=3)
    fig.legend(handles=[Line2D([],[],marker='o',color='#0072B2',label='Point pair retention'),Line2D([],[],marker='s',color='#D55E00',label='Cell and date availability')],loc='lower center',bbox_to_anchor=(.54,.09),ncol=2,handlelength=1.5)
    fig.legend(handles=[Rectangle((0,0),1,1,color='#0072B2',label='Composition'),Rectangle((0,0),1,1,color='#D55E00',label='Within group'),Line2D([],[],marker='D',lw=0,color='#292929',label='Total')],loc='lower center',bbox_to_anchor=(.54,.035),ncol=3,handlelength=1.2)
    export(fig,'Figure5_selection_and_output_availability',source_key(map='cell_availability selection_global_p0.5 pair_retention_in_cell',availability='unweighted occupied-cell mean fraction_dates_with_output; distinct from retained point pairs',decomposition='selection_decomposition p0.5 global/grouped; composition+within=total exact identity',geometry='local 2km cell offsets, same support as Figure1; no claim of lost hazard monitoring',all_cells='No cell disappears on every test date in these products; sparse per-date output may occur'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--basemap');ap.add_argument('--figures',nargs='*',type=int,default=[1,2,3,4,5]);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);configure_style()
    p=ROOT/'results/core_results.json';core=json.loads(p.read_text());AUDIT['source_files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    if not core.get('actually_completed'):
        raise AssertionError('Refusing incomplete core numerical results')
    summary=read_csv('summary.csv');dates=read_csv('per_date.csv');cells=read_csv('cell_availability.csv');decomp=read_csv('selection_decomposition.csv')
    if 1 in args.figures:figure1(core,cells,prepare_basemap(args.basemap))
    if 2 in args.figures:figure2(core)
    if 3 in args.figures:figure3(summary)
    if 4 in args.figures:figure4(dates,summary)
    if 5 in args.figures:figure5(summary,cells,decomp)
    # Preserve already audited figures when rerunning a chosen subset.
    old=OUT/'figure_audit.json'
    if old.exists():
        d=json.loads(old.read_text());d['figures'].update(AUDIT['figures']);d['style']=AUDIT['style'];d['source_files'].update(AUDIT['source_files'])
    else:d=AUDIT
    old.write_text(json.dumps(d,indent=2),encoding='utf-8')
    print(json.dumps({'built':list(AUDIT['figures']),'font_minimum_pt':8,'width_mm':WIDTH_MM,'audit':str(old)}))


if __name__=='__main__':main()
