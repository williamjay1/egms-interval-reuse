"""Generate manuscript tables from executed REV2 outputs, with a numeric ledger."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'manuscript'
NAMES={'amsterdam':'Amsterdam','venice':'Venice','alps':'Jura'}
MODEL_NAMES={'persistence':'Persistence','mean3grid':'Three-value mean',
 'local_linear144':'Local linear (144 d)','trend_seasonal360':'Trend + annual (360 d)',
 'ridge_level':'Level ridge','ridge_increment':'Increment ridge',
 'ridge_increment_duplicate':'Increment + duplicate'}
SHORT={'source_full_global':'Source full, global','source_full_grouped':'Source full, grouped',
 'source_last12_global':'Source last 12','target_last12_global':'Target last 12',
 'target_rolling12_global':'Target rolling 12'}
LEDGER=[]
CONTEXT='results/core_results.json or results/revision_calibration/summary.csv; table inputs are explicitly loaded below'

def rows(path):
 with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def f(value, digits=2, scale=1, source=''):
 value=float(value)*scale
 rendered=f'{value:.{digits}f}'
 if float(rendered)==0:rendered=rendered.lstrip('-')
 LEDGER.append({'source':source or CONTEXT,'scaled_value':value,'digits':digits,'rendered':rendered})
 return rendered

def ci(row,key='coverage',scale=100):
 return f(row[key],2,scale)+r' ['+f(row[key+'_spatial_lo'],2,scale)+', '+f(row[key+'_spatial_hi'],2,scale)+']'

def texrow(values):return ' & '.join(map(str,values))+r' \\'+'\n'

def tabular(header,body,columns):
 return '\\begin{tabular}{'+columns+'}\n\\toprule\n'+texrow(header)+'\\midrule\n'+body+'\\bottomrule\n\\end{tabular}\n'

def main_table(name,caption,label,header,body,columns,note):
 text='\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\\setlength{\\tabcolsep}{4.5pt}\n'
 if name=='table2_rev2.tex':text+='\\fontsize{9}{10.5}\\selectfont\\renewcommand{\\arraystretch}{1.02}\n'
 text+=tabular(header,body,columns)+'\\par\\vspace{3pt}\\begin{minipage}{\\linewidth}\\footnotesize '+note+'\\end{minipage}\n\\end{table}\n'
 (MAN/name).write_text(text,encoding='utf-8')

def longtable(title,header,body,columns,note=''):
 text='\\Needspace{8\\baselineskip}\n\\subsection*{'+title+'}\n'+note+'\n'
 text+='\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{'+columns+'}\n\\toprule\n'
 head=texrow(header)+'\\midrule\n'
 text+=head+'\\endfirsthead\n\\toprule\n'+head+'\\endhead\n'
 text+='\\midrule\\multicolumn{'+str(len(header))+'}{r}{Continued on next page}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n'
 text+=body+'\\end{longtable}\n\\endgroup\n'
 return text

def main():
 MAN.mkdir(parents=True,exist_ok=True)
 (ROOT/'qa').mkdir(parents=True,exist_ok=True)
 core=json.loads((ROOT/'results/core_results.json').read_text(encoding='utf-8'))
 assert core['actually_completed']
 caldir=ROOT/'results/revision_calibration'
 cal=rows(caldir/'summary.csv')
 lookup={(r['target'],r['protocol']):r for r in cal}
 body=''
 for key,name in NAMES.items():
  p=core['products'][key]
  body+=texrow([name,f"{p['points']:,}",len(p['dates']),
   '/'.join(str(p['phases'][s]['origin_count']) for s in ['fit','calibration','test']),
   p['block_count'],f(p['interpolation']['mean_interpolated_grid_fraction'],2,100),
   p['interpolation']['maximum_bracket_days']])
 main_table('table1_rev2.tex','Analysed products and calendar-aligned samples.','tab:products',
  ['Product','Points','Dates','Fit/cal/test','Cells','Interp. (\\%)','Gap (d)'],body,'lrrrrrr',
  'All products are ascending VV Calibrated measurements from release \\texttt{2020\\_2024\\_1}. '
  'Amsterdam: \\texttt{088\\_0312\\_IW1}; Venice: \\texttt{117\\_0266\\_IW3}; Jura: \\texttt{088\\_0278\\_IW2}. '
  'Phase counts are eligible origin dates, not independent observations. Cells are occupied 2 km projected-coordinate cells. '
  'Interp. is the fraction of historical grid entries interpolated over all eligible origins; gap is the largest interpolation bracket. Future targets are observed at exactly 72 days.')
 body=''
 protocols=list(SHORT)
 for key,name in NAMES.items():
  for j,protocol in enumerate(protocols):
   r=lookup[key,protocol]
   body+=texrow([name if j==0 else '',SHORT[protocol],ci(r),f(r['mean_width']),f(r['interval_score']),f(r['outside_per_1000_pairs'],1)])
  if key!='alps':body+='\\addlinespace\n'
 main_table('table2_rev2.tex','Interval reuse and mature-target calibration with a frozen predictor.','tab:calibration',
  ['Product','Residual pool','Coverage (\\%) [95\\% CI]','Width (mm)','Score (mm)','Outside/1,000'],body,'llrrrr',
  'Frozen Amsterdam increment predictor; identical target pairs. Nominal coverage: 90\\%; outside count: 100/1,000. '
  'Width is the mean full width; score is defined in Methods. CI: conditional 2 km spatial cluster bootstrap, 1,000 draws. '
  'Last 12 uses static calibration origins; rolling uses only targets strictly before each origin. Amsterdam source-last and target-last coincide. '
  'Tail and separate date-block diagnostics: Supplementary Tables S4 and S6.')
 si=[]
 body=''
 for key,name in NAMES.items():
  p=core['products'][key]
  for s in ['fit','calibration','test']:
   z=p['phases'][s]
   body+=texrow([name,s,z['origin_count'],f"{z['point_origin_pairs']:,}",z['first_origin']+' to '+z['last_origin'],z['first_target']+' to '+z['last_target'],f(p['interpolation'][s]['grid_fraction_interpolated'],2,100)])
 si.append(longtable('Table S1. Calendar splits and input support',
  ['Product','Phase','Origins','Pairs','Origin dates','Target dates','Interp. (\\%)'],body,'llrrllr',
  'Date boundaries are common across products; every target is an observed acquisition exactly 72 days after its origin. Interpolation concerns history only.'))
 body=''
 for key,name in NAMES.items():
  p=core['products'][key];dates=np.array(p['dates'],dtype='datetime64[D]');o=np.array(p['origins'])
  for phase in ['fit','calibration','test']:
   z=p['phases'][phase]
   mask=(np.array(p['origin_dates'])>=z['first_origin'])&(np.array(p['origin_dates'])<=z['last_origin'])
   idx=o[mask];idx=idx[idx+6<len(dates)]
   span=(dates[idx+6]-dates[idx]).astype(int)
   body+=texrow([name,phase,int(span.min()),f(np.median(span),1),int(span.max())])
  d=p['dropout']
  si.append('\\noindent '+name+': '+str(d['eligible_origins'])+' eligible origins; '+str(d['missing_exact_72_day_target_after_history_check'])+' origins without an exact future acquisition (including terminal origins beyond product end); '+str(d['eligible_origins_outside_evaluation_phases'])+' eligible origins outside formal phases.\\par\n')
 si.append(longtable('Six-acquisition lead diagnostic', ['Product','Calendar phase','Minimum (d)','Median (d)','Maximum (d)'],body,'llrrr',
  'For the actual origins retained by the new calendar task, this table describes how far six raw acquisitions would extend. It changes the target definition and is not a controlled estimate of a pure cadence effect. The main experiment always uses 72 days.'))
 body=''
 for key,name in NAMES.items():
  for model,m in core['baselines'][key]['models'].items():
   z=m['metrics'];lo,hi=m['ci_95_spatial_2000m']['mae']
   body+=texrow([name,MODEL_NAMES[model],f(z['mae'])+' ['+f(lo)+', '+f(hi)+']',f(z['rmse']),f(z['bias'])])
 si.append(longtable('Table S2. Prediction baselines on identical calendar targets',
  ['Product','Rule','MAE (mm) [95\\% CI]','RMSE (mm)','Bias (mm)'],body,'llrrr',
  'Bias is prediction minus target. Fixed baselines have no training source; ridge fits use Amsterdam. The duplicate model is an ablation. All 21 paired model differences per product are provided in the machine-readable core results.'))
 body=''
 for source,models in core['fitted_models'].items():
  for model,m in models.items():
   for term,value in m['raw_unit_coefficients'].items():
    body+=texrow([NAMES[source],MODEL_NAMES[model],term.replace('_',r'\_'),f(value,6)])
 si.append(longtable('Table S3. Raw-unit fitted coefficients', ['Source','Rule','Term','Coefficient'],body,'lllr',
  'Feature centring/scaling and the unpenalized target intercept are stored in core\\_results.json. Coefficients refer to the unique calendar features, with the marked duplicate diagnostic separate.'))
 body=''
 for key,name in NAMES.items():
  for r in cal:
   if r['target']==key and (r['protocol'] in ['source_full_global','source_full_grouped'] or r['protocol'].startswith(('source_last','target_last','target_rolling'))):
    body+=texrow([name,r['protocol'].replace('_',r'\_'),f(r['coverage'],2,100),f(r['mean_width']),f(r['interval_score']),f(r['lower_tail'],2,100),f(r['upper_tail'],2,100),f(r['equal_cell_coverage'],2,100)])
 si.append(longtable('Table S4. Residual-window, tails and density-weighting comparison',
  ['Product','Protocol','Cover. (\\%)','Width (mm)','Score (mm)','Lower (\\%)','Upper (\\%)','Cell cover. (\\%)'],body,'llrrrrrr',
  'All prediction centres are fixed. K counts mature origin dates, each contributing all points. Lower/upper mean target below/above the interval; they sum with coverage to 100\\%. Equal-cell coverage weights occupied cells equally.'))
 body=''
 for r in cal:
  if r['protocol'].startswith(('calibration_','predictor_')):
   body+=texrow([NAMES[r['target']],r['protocol'].replace('_',r'\_'),f(r['mae']),f(r['coverage'],2,100),f(r['mean_width']),f(r['interval_score'])])
 si.append(longtable('Table S5. Predictor and calibration-source component controls',
  ['Target','Control','MAE (mm)','Cover. (\\%)','Width (mm)','Score (mm)'],body,'llrrrr',
  'Calibration controls retain Amsterdam centres and change residual source/window. Predictor controls retain the identical numerical Amsterdam full-global radius and change training source. These isolate algorithm components, not physical causes.'))
 comp=rows(caldir/'paired_comparisons.csv');body=''
 for r in comp:
  if r['protocol']=='target_rolling12_global':
   cell=lambda s:f(r['delta_coverage'+s+'_lo'],2,100)+', '+f(r['delta_coverage'+s+'_hi'],2,100)
   body+=texrow([NAMES[r['target']],f(r['delta_coverage'],2,100),cell('_spatial'),cell('_timeblock6'),cell('_timeblock12')])
 si.append(longtable('Table S6. Paired coverage changes for rolling 12 versus source full',
  ['Product','Change (pp)','Spatial 95\\% bounds','6-origin block bounds','12-origin block bounds'],body,'lrrrr',
  'Separate observed-date block calculations and spatial calculations describe different uncertainty scopes. Date resampling conditions on the realized rolling update path; it does not rerun adaptation on a resampled fictitious history. They are not combined into a joint confidence interval.'))
 body='';decomp=rows(caldir/'selection_decomposition.csv')
 for d in decomp:
  key=d['target'];p=float(d['selection_quantile']);r=lookup[key,f"selection_{d['interval_type']}_p{p:g}"]
  body+=texrow([NAMES[key],d['interval_type'],f(p,2),f(d['pair_retention'],2,100),f(r['mae']),f(r['coverage'],2,100),f(r['mean_width']),f(d['composition_change'],3,100),f(d['within_group_change'],3,100),r['n_cells_never_retained'],r['n_cells_at_most_25pct_dates_with_output']])
 si.append(longtable('Table S7. Selection, exact decomposition and output availability',
  ['Product','Interval','Quantile','Kept (\\%)','MAE (mm)','Cover. (\\%)','Width (mm)','Comp. (pp)','Within (pp)','Never','Rare'],body,'llrrrrrrrrr',
  'Comp. and Within add to the coverage change versus that interval rule before filtering. Never counts occupied cells without retained output at any test date; Rare counts cells with output on at most 25\\% of dates. These are cell availability, not land-area fractions.'))
 body=''
 for key,name in NAMES.items():
  for model,control in core['reference_shift_controls'][key].items():
   for c in control['cases']:
    z=c['metrics']
    body+=texrow([name,MODEL_NAMES[model],f(c['constant_shift_mm'],0),f(z['mae']),f(z['bias']),f(z['coverage'],2,100),f(c['analytic_error_change_mm'],3)])
 si.append(longtable('Table S8. Frozen-source reference-shift control',
  ['Product','Rule','Shift (mm)','MAE (mm)','Bias (mm)','Cover. (\\%)','Analytic error change (mm)'],body,'llrrrrr',
  'Every target history and target observation receives the same shift; source transformations, coefficients and model-specific source radii remain fixed. This controlled perturbation is not a real ground-motion observation.'))
 body=''
 for r in rows(caldir/'factorial_summary.csv'):
  body+=texrow([NAMES[r['target']],NAMES[r['predictor_source']],NAMES[r['calibration_source']],f(r['mae']),ci(r),f(r['mean_width']),f(r['interval_score'])])
 si.append(longtable('Table S9. Complete predictor-source by residual-source comparison',
  ['Target','Predictor source','Residual source','MAE (mm)','Cover. (\\%) [95\\% CI]','Width (mm)','Score (mm)'],body,'lllrrrr',
  'Each predictor is fitted on its own source fitting phase; every radius uses residuals from the stated calibration product under that same predictor. All 27 rows use the common target test pairs. The spatial intervals condition on these fitted components. Interaction contrasts and paired comparisons are also supplied as CSV files.'))
 body=''
 for r in rows(caldir/'release_timeline.csv'):
  body+=texrow([NAMES[r['target']],r['last_observation_date_in_product'],r['last_observation_plus72_days'],r['release_announcement_date'],r['announcement_lag_after_last_observation_days'],'Not verified'])
 si.append(longtable('Table S10. Acquisition dates and product announcement',
  ['Product','Last observation','Last observation +72 d','Official announcement','Lag (d)','First file access'],body,'llllrl',
  'The Copernicus Land Monitoring Service announced the 2020--2024 update on 29 April 2026. This announcement is not evidence of the first public access date of each exact CSV. A nominal 72-day target from the last observation precedes the announcement. These dates illustrate why acquisition time cannot be treated as public product availability. Source: \\url{https://land.copernicus.eu/en/news/egms-annual-update-brings-new-data-improvements-to-egms-explorer}.'))
 body=''
 for key,name in NAMES.items():
  b=core['products'][key]['bbox_lon_lat']
  body+=texrow([name]+[f(b[i],5) for i in [0,2,1,3]])
 si.append(longtable('Table S11. Measurement-coordinate bounds',
  ['Product','Minimum longitude','Minimum latitude','Maximum longitude','Maximum latitude'],body,'lrrrr',
  'Bounds describe the retained measurement points in geographical degrees. They are not complete land coverage or a boundary of a ground-motion process. The full exact CSV identifiers and SHA256 checksums are in the Supplementary Software data manifest.'))
 (MAN/'supplement_tables_rev2.tex').write_text('\n'.join(si),encoding='utf-8')
 (ROOT/'qa/table_numeric_ledger.json').write_text(json.dumps(LEDGER,indent=2),encoding='utf-8')
 print(json.dumps({'main_tables':2,'supplementary_tables':11,'ledger_cells':len(LEDGER)},indent=2))

if __name__=='__main__':main()
