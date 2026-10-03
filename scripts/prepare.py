"""Rebuild application-time aggregates from the user's original ingestion source."""
from pathlib import Path
import sys,json,zipfile,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from src.core import CAT,RAW_NUM
def main():
    for directory in ['data/processed','models','reports/figures','notebooks','docs']: (ROOT/directory).mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT.parent/'Case Default Risk/etl/csv_home-credit-default-risk.zip');ap.add_argument('--etl-export',type=Path,default=ROOT.parent/'Case Default Risk/data/hc_model_features.parquet');a=ap.parse_args()
    with zipfile.ZipFile(a.source) as z:
        def read(name,cols):
            member=next(x for x in z.namelist() if x.endswith(name));return pd.read_csv(z.open(member),usecols=cols)
        appcols=['SK_ID_CURR','TARGET','CODE_GENDER']+CAT+RAW_NUM[:11]
        frame=read('application_train.csv',appcols)
        desc=pd.read_csv(z.open(next(x for x in z.namelist() if x.endswith('HomeCredit_columns_description.csv'))),encoding='cp1252')
        target=desc[desc['Row']=='TARGET']['Description'].iloc[0]
        b=read('bureau.csv',['SK_ID_CURR','DAYS_CREDIT','DAYS_CREDIT_UPDATE','AMT_CREDIT_SUM_DEBT','CREDIT_DAY_OVERDUE'])
        p=read('previous_application.csv',['SK_ID_CURR','DAYS_DECISION','NAME_CONTRACT_STATUS'])
        audit={'bureau_rows':len(b),'bureau_excluded_future_or_unknown':int((~((b.DAYS_CREDIT<=0)&(b.DAYS_CREDIT_UPDATE<=0))).sum()),'previous_rows':len(p),'previous_excluded':int((~(p.DAYS_DECISION<=0)).sum())}
        b=b[(b.DAYS_CREDIT<=0)&(b.DAYS_CREDIT_UPDATE<=0)]
        ba=b.groupby('SK_ID_CURR').agg(QTD_CREDITOS_BUREAU=('SK_ID_CURR','size'),TOTAL_DIVIDA_BUREAU=('AMT_CREDIT_SUM_DEBT',lambda x:x.sum(min_count=1)),MEDIA_DIAS_ATRASO_BUREAU=('CREDIT_DAY_OVERDUE','mean'))
        p=p[p.DAYS_DECISION<=0];p['refused']=(p.NAME_CONTRACT_STATUS=='Refused').astype(int)
        pa=p.groupby('SK_ID_CURR').agg(QTD_APLICACOES_ANTERIORES=('SK_ID_CURR','size'),QTD_RECUSAS_ANTERIORES=('refused','sum'))
        member=next(x for x in z.namelist() if x.endswith('installments_payments.csv'));parts=[];total=excluded=0
        for chunk in pd.read_csv(z.open(member),usecols=['SK_ID_CURR','DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT'],chunksize=500000):
            total+=len(chunk);keep=(chunk.DAYS_INSTALMENT<=0)&(chunk.DAYS_ENTRY_PAYMENT<=0);excluded+=int((~keep).sum());c=chunk[keep].copy();c['delta']=c.DAYS_ENTRY_PAYMENT-c.DAYS_INSTALMENT;c['late']=(c.delta>0).astype(int)
            parts.append(c.groupby('SK_ID_CURR').agg(n=('delta','size'),total=('delta','sum'),late=('late','sum')))
        ia=pd.concat(parts).groupby(level=0).sum();ia['ATRASO_MEDIO_PAGAMENTO']=ia.total/ia.n;ia['QTD_PARCELAS_ATRASADAS']=ia.late
        audit.update(installment_rows=total,installment_excluded_future_or_missing=excluded)
        for agg in [ba,pa,ia[['ATRASO_MEDIO_PAGAMENTO','QTD_PARCELAS_ATRASADAS']]]:frame=frame.merge(agg,on='SK_ID_CURR',how='left',validate='one_to_one')
        bridge={'available':a.etl_export.exists()}
        if a.etl_export.exists():
            old=pd.read_parquet(a.etl_export);bridge.update(rows=len(old),same_customer_ids=set(old.SK_ID_CURR)==set(frame.SK_ID_CURR),same_targets=bool(old.set_index('SK_ID_CURR').TARGET.sort_index().equals(frame.set_index('SK_ID_CURR').TARGET.sort_index())))
            if not bridge['same_customer_ids'] or not bridge['same_targets']:raise ValueError('ETL export and raw labels differ')
        assert frame.SK_ID_CURR.is_unique and frame.TARGET.notna().all()
        indices=frame.index.to_numpy();train,rest=train_test_split(indices,test_size=.4,stratify=frame.TARGET,random_state=42)
        dev,rest=train_test_split(rest,test_size=.75,stratify=frame.loc[rest,'TARGET'],random_state=43)
        cal,rest=train_test_split(rest,test_size=2/3,stratify=frame.loc[rest,'TARGET'],random_state=44)
        policy,test=train_test_split(rest,test_size=.75,stratify=frame.loc[rest,'TARGET'],random_state=45)
        for name,ix in [('train',train),('development',dev),('calibration',cal),('policy',policy),('test',test)]:frame.loc[ix,'split']=name
        frame.to_parquet(ROOT/'data/processed/applications.parquet',index=False)
    manifest={'source':'https://www.kaggle.com/competitions/home-credit-default-risk/data','source_sha256':hashlib.file_digest(a.source.open('rb'),'sha256').hexdigest(),'rows':len(frame),'positives':int(frame.TARGET.sum()),'target_description':target,'splits':frame.groupby('split').size().to_dict(),'target_rates':frame.groupby('split').TARGET.mean().to_dict(),'date_audit':audit,'etl_bridge':bridge,'excluded_model_fields':['SK_ID_CURR','TARGET','CODE_GENDER','split'],'cutoff':'Relative day 0 (current application); bureau credit and update <=0; previous decision <=0; installment due and payment <=0. Missing dates excluded. Historical tables remain retrospective snapshots; exact field availability is not proven.','validation':'Stratified independent customers, not temporal: no global application calendar available.'}
    (ROOT/'reports/data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
