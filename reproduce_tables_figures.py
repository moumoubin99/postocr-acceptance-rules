"""Regenerate tables/numbers and the eight numeric plots from supplied aggregate JSON."""
from pathlib import Path
import subprocess, sys, json, hashlib
ROOT=Path(__file__).resolve().parent
for script in ['make_tables.py','make_figs.py']:
    subprocess.run([sys.executable,str(ROOT/'code/paper'/script)],cwd=ROOT,check=True)
tables=[]
for f in (ROOT/'reference/tables').glob('*.tex'):
    got=ROOT/'generated/tables'/f.name
    tables.append({'file':f.name,'same':got.exists() and f.read_bytes()==got.read_bytes()})
reference=json.loads((ROOT/'reference/numbers.json').read_text())
generated=json.loads((ROOT/'generated/numbers.json').read_text())
value_changes=[k for k in set(reference)|set(generated) if reference.get(k,{}).get('value')!=generated.get(k,{}).get('value')]
report={'tables':tables,'number_keys':len(generated),'number_value_changes':value_changes,'figure_files':[f.name for f in sorted((ROOT/'generated/figures').glob('*.pdf'))],'scope':'Aggregate-to-table and aggregate-to-plot regeneration; no experiment rerun. Pipeline diagram supplied as reference only.'}
(ROOT/'generated/regeneration_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if value_changes or any(not x['same'] for x in tables):raise SystemExit(1)
