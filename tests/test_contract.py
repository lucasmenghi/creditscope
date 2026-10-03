from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from src.core import features,FEATURES
def test_sensitive_and_target_not_inputs():
    f=pd.DataFrame([{'SK_ID_CURR':123,'TARGET':1,'CODE_GENDER':'F','DAYS_EMPLOYED':365243,'AMT_INCOME_TOTAL':0,'AMT_CREDIT':10}]);x=features(f)
    assert list(x)==FEATURES and not {'SK_ID_CURR','TARGET','CODE_GENDER'}.intersection(x)
    assert pd.isna(x.DAYS_EMPLOYED.iloc[0]) and x.employment_unknown.iloc[0]==1
    assert pd.isna(x.credit_income_ratio.iloc[0])
def test_split_manifest_and_labels():
    d=json.loads((ROOT/'reports/data_manifest.json').read_text());assert sum(d['splits'].values())==d['rows']==307511
    if d['etl_bridge']['available']:assert d['etl_bridge']['same_customer_ids'] and d['etl_bridge']['same_targets']
    assert d['splits']['test']==46128
def test_capacity_diagnostic_and_metrics():
    m=json.loads((ROOT/'reports/metrics.json').read_text());s=json.loads((ROOT/'reports/capacity_scenarios.json').read_text());r=next(x for x in s if x['capacity']==20)['selected']
    assert abs(r['recall']-m['selected']['recall_at20'])<1e-10
    assert sum(sum(row) for row in m['selected']['confusion_matrix'])==m['test_n']
    assert all(0<=m['selected'][k]<=1 for k in ['roc_auc','average_precision','brier','ks'])
