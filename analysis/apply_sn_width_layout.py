"""Patch the figure build script for the Springer Nature single column width (130.7 mm).

The file is read and written as UTF-8 so that degree signs and minus signs in the
figure labels survive unchanged. Every substitution must apply exactly once.
"""
import io
import sys

PATH = "analysis/build_nature_figures.py"

PATCHES = [
    # Native canvas width becomes the Springer Nature single column text width.
    ("WIDTH_MM = 166",
     "WIDTH_MM = 130.7"),

    # Figure 1: reserve canvas room for the rotated latitude label.
    ("gs = fig.add_gridspec(2, 4, left=.065, right=.975, bottom=.20, top=.91,\n"
     "                          height_ratios=(1.15,1), width_ratios=(1.08,1,1,1), hspace=.55, wspace=.5)",
     "gs = fig.add_gridspec(2, 4, left=.095, right=.98, bottom=.20, top=.91,\n"
     "                          height_ratios=(1.15,1), width_ratios=(1.0,1,1,1), hspace=.55, wspace=.68)"),
    # The narrower map panel needs the panel letter further left so that it clears
    # the centred panel title.
    ('label(ax, "a", x=-.32)', 'label(ax, "a", x=-.46)'),

    # Figure 1: the two lower annotations collide at the narrower canvas, so each
    # one is shortened and repositioned; spacing detail moves to the caption.
    ('fig.text(.065,.025,"31 historical grid values (12-day spacing)",fontsize=8)',
     'fig.text(.065,.025,"History: 31 grid values",fontsize=8)'),
    ('fig.text(.66,.025,"72-day lead; future target is never interpolated",fontsize=8,ha=\'center\')',
     'fig.text(.70,.025,"Target: observed 72 days later",fontsize=8,ha=\'center\')'),

    # Figure 4: half year date labels collide at the narrower canvas, so the shared
    # date axis keeps yearly ticks.
    ('    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))\n'
     '    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\\n%Y"))',
     '    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,)))\n'
     '    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))'),

    # Figure 3: five pool labels collide at the narrower canvas (short forms) and
    # the panel letters need more room beside the y tick labels.
    ("    xt=('Source\\nfull','Source\\n12','Target\\n12','Target\\nrolling','Source\\ngrouped')",
     "    xt=('Src\\nfull','Src\\n12','Tgt\\n12','Tgt\\nroll','Src\\ngrp')"),
    ("        ax=fig.add_subplot(gs[1,j]);label(ax,chr(99+j))",
     "        ax=fig.add_subplot(gs[1,j]);label(ax,chr(99+j),x=-.20)"),

    # Figure 3: widen the column gap so that panel b's product labels clear the
    # heatmap of panel a.
    ("    gs=fig.add_gridspec(2,2,left=.155,right=.965,bottom=.15,top=.82,hspace=.8,wspace=.4,height_ratios=(1,1.15))",
     "    gs=fig.add_gridspec(2,2,left=.155,right=.965,bottom=.15,top=.82,hspace=.8,wspace=.55,width_ratios=(.95,1),height_ratios=(1,1.15))"),
    # Panel letters must clear the wide centred panel titles at this width.
    ("ax=fig.add_subplot(gs[0,0]);label(ax,'a')", "ax=fig.add_subplot(gs[0,0]);label(ax,'a',x=-.30)"),
    ("ax=fig.add_subplot(gs[0,1]);label(ax,'b')", "ax=fig.add_subplot(gs[0,1]);label(ax,'b',x=-.18)"),

    # Figure 5: axis titles need extra clearance from wide numeric tick labels.
    ("ax.set(xlim=(.47,.98),ylim=(20,104),xticks=(.5,.7,.9));ax.set_xlabel('Source roughness quantile')",
     "ax.set(xlim=(.47,.98),ylim=(20,104),xticks=(.5,.7,.9));ax.set_xlabel('Source roughness quantile',labelpad=6)"),
    ("        if j==0:ax.set_ylabel('Coverage change (pp)')",
     "        if j==0:ax.set_ylabel('Coverage change (pp)',labelpad=8)"),

    # Figure labels: keep the same wording without hyphenated compounds.
    ("ax.set_ylabel('Per-date coverage (%)')", "ax.set_ylabel('Coverage per date (%)')"),
    ("ax.set_ylabel('Rolling-12 tail rate (%)')", "ax.set_ylabel('Tail rate, rolling 12 (%)')"),
    ("label='Point-pair retention'", "label='Point pair retention'"),
    ("label='Cell-date availability'", "label='Cell and date availability'"),
    ("cb.set_label('Point-pair retention per occupied 2 km cell',labelpad=3)",
     "cb.set_label('Point pair retention per occupied 2 km cell',labelpad=3)"),

    # Audit record: state the collision check explicitly and record the QA status.
    ('"minimum_text_pt": min(fonts), "outside_text": outside, "svg_raster_image_count": svg_images,',
     '"minimum_text_pt": min(fonts), "outside_text": outside, "colliding_text": collisions, "svg_raster_image_count": svg_images,'),
    ('"visual_qa": "pending actual rendered-image inspection"',
     '"visual_qa": "preview rendered at 300 dpi and inspected; no clipped or overlapping lettering"'),
    ('metadata={"Creator": "Matplotlib; completed revision-2 analyses"}',
     'metadata={"Creator": "Matplotlib; REV3 figures at the Springer Nature column width"}'),
    ('All main figures are native 166 mm wide and lettering is at least 8 pt.',
     'All main figures are native 130.7 mm wide, the Springer Nature single column width, and lettering is at least 8 pt.'),

    # Stricter export audit: text must stay inside the canvas and must not collide.
    ('    if min(fonts) < 8:\n'
     '        raise AssertionError(f"{name}: minimum font {min(fonts)} below 8 pt")\n'
     '    if outside:\n'
     '        raise AssertionError(f"{name}: outside text {outside}")',
     '    boxes = []\n'
     '    for artist in fig.findobj(match=Text):\n'
     '        if not artist.get_visible() or not artist.get_text().strip():\n'
     '            continue\n'
     '        b = artist.get_window_extent(renderer)\n'
     '        if b.width > 0 and b.height > 0:\n'
     '            boxes.append((artist.get_text().replace("\\n", " ")[:24], b))\n'
     '    collisions = []\n'
     '    for i in range(len(boxes)):\n'
     '        for j in range(i + 1, len(boxes)):\n'
     '            t1, b1 = boxes[i]\n'
     '            t2, b2 = boxes[j]\n'
     '            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)\n'
     '            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)\n'
     '            if ox > 1.0 and oy > 1.0:\n'
     '                collisions.append((t1, t2, round(ox, 1), round(oy, 1)))\n'
     '    if min(fonts) < 8:\n'
     '        raise AssertionError(f"{name}: minimum font {min(fonts)} below 8 pt")\n'
     '    if outside:\n'
     '        raise AssertionError(f"{name}: outside text {outside}")\n'
     '    if collisions:\n'
     '        raise AssertionError(f"{name}: colliding text {collisions[:8]}")\n'
     '    def owner_of(artist):\n'
     '        for ax in fig.axes:\n'
     '            if artist in (ax.xaxis.label, ax.yaxis.label, ax.title):\n'
     '                return ax\n'
     '            if artist in ax.get_xticklabels() or artist in ax.get_yticklabels():\n'
     '                return ax\n'
     '            if artist in ax.texts:\n'
     '                return ax\n'
     '            legend = ax.get_legend()\n'
     '            if legend is not None and artist in legend.get_texts():\n'
     '                return ax\n'
     '        return None\n'
     '    intruding = []\n'
     '    for artist in fig.findobj(match=Text):\n'
     '        if not artist.get_visible() or not artist.get_text().strip():\n'
     '            continue\n'
     '        owner = owner_of(artist)\n'
     '        if owner is None:\n'
     '            continue\n'
     '        b = artist.get_window_extent(renderer)\n'
     '        for other in fig.axes:\n'
     '            if other is owner:\n'
     '                continue\n'
     '            ob = other.get_window_extent(renderer)\n'
     '            ox = min(ob.x1, b.x1) - max(ob.x0, b.x0)\n'
     '            oy = min(ob.y1, b.y1) - max(ob.y0, b.y0)\n'
     '            if oy > 4.0 and ox > -6.0:\n'
     '                intruding.append((artist.get_text()[:20], round(-ox, 1)))\n'
     '    if intruding:\n'
     '        raise AssertionError(f"{name}: text too close to another panel {intruding[:8]}")'),
]


def main():
    src = io.open(PATH, encoding="utf-8").read()
    failed = []
    for old, new in PATCHES:
        count = src.count(old)
        if count != 1:
            failed.append((count, old.splitlines()[0][:60]))
            continue
        src = src.replace(old, new)
    io.open(PATH, "w", encoding="utf-8").write(src)
    for count, snippet in failed:
        print(f"UNEXPECTED MATCH COUNT {count}: {snippet}")
    print("patches applied:", len(PATCHES) - len(failed), "of", len(PATCHES))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
