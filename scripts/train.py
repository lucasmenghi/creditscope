"""Candidate selection, calibration, policy and untouched customer holdout."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,confusion_matrix,roc_curve
from lightgbm import LGBMClassifier
from src.core import CAT,NUM,features,logit,predict,RAW_INPUTS
CANDIDATES={'logistic_c01':{'kind':'logistic','C':.1},'logistic_c1':{'kind':'logistic','C':1},'lightgbm_15':{'kind':'lightgbm','leaves':15,'trees':300},'lightgbm_31':{'kind':'lightgbm','leaves':31,'trees':450}}
def fit(config,frame):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]) if config['kind']=='logistic' else 'passthrough'
    categorical=OneHotEncoder(handle_unknown='ignore',min_frequency=20) if config['kind']=='logistic' else OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    prep=ColumnTransformer([('categorical',categorical,CAT),('numeric',numeric,NUM)])
    model=LogisticRegression(C=config.get('C',1),max_iter=1500,random_state=42) if config['kind']=='logistic' else LGBMClassifier(n_estimators=config['trees'],num_leaves=config['leaves'],learning_rate=.04,min_child_samples=150,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=42,deterministic=True,force_col_wise=True)
    pipe=Pipeline([('prep',prep),('model',model)]);kwargs={'model__categorical_feature':list(range(len(CAT)))} if config['kind']=='lightgbm' else {}
    pipe.fit(features(frame),frame.TARGET,**kwargs);return pipe
def metrics(y,p,threshold):
    y=np.asarray(y);pred=p>=threshold;ix=np.argsort(-p,kind='stable')[:int(np.ceil(.2*len(y)))];prec=float(y[ix].mean());fpr,tpr,_=roc_curve(y,p)
    return {'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'ks':float(np.max(tpr-fpr)),'threshold':threshold,'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'queue_rate':float(pred.mean()),'precision_at20':prec,'recall_at20':float(y[ix].sum()/y.sum()),'lift_at20':prec/float(y.mean()),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
def main():
    start=time.time();df=pd.read_parquet(ROOT/'data/processed/applications.parquet');parts={s:g for s,g in df.groupby('split')};rows=[]
    for name,config in CANDIDATES.items():
        model=fit(config,parts['train']);p=model.predict_proba(features(parts['development']))[:,1]
        row={'model':name,'average_precision':float(average_precision_score(parts['development'].TARGET,p)),'roc_auc':float(roc_auc_score(parts['development'].TARGET,p))};rows.append(row);print(row,flush=True)
    chosen=max(rows,key=lambda x:x['average_precision'])['model'];bundles={};policy_results={}
    for name,candidate in [('selected',chosen),('logistic_baseline','logistic_c1')]:
        model=fit(CANDIDATES[candidate],parts['train']);raw=model.predict_proba(features(parts['calibration']))[:,1];calibrator=LogisticRegression(C=1e6,max_iter=1000,random_state=42).fit(logit(raw),parts['calibration'].TARGET)
        bundle={'model':model,'calibrator':calibrator,'candidate':candidate,'raw_inputs':RAW_INPUTS};pp=predict(bundle,parts['policy']);bundle['threshold']=float(np.quantile(pp,.8));bundles[name]=bundle;policy_results[name]=metrics(parts['policy'].TARGET,pp,bundle['threshold'])
    frozen={'seed':42,'candidates':CANDIDATES,'selected':chosen,'selection':'Highest average precision in separate stratified development customers; single split, no cross-validation claim','split_sizes':{s:len(g) for s,g in parts.items()},'threshold_rule':'80th percentile of calibrated policy scores, frozen before final test','thresholds':{s:b['threshold'] for s,b in bundles.items()},'raw_inputs':RAW_INPUTS,'target':'TARGET=1, payment difficulties under competition label definition','validation':'Independent customer holdout, not temporal validation'}
    (ROOT/'reports/frozen_config.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');pd.DataFrame(rows).to_csv(ROOT/'reports/development.csv',index=False)
    test=parts['test'];scores={s:predict(b,test) for s,b in bundles.items()};result={s:metrics(test.TARGET,p,bundles[s]['threshold']) for s,p in scores.items()}
    result.update(test_n=len(test),test_positive_rate=float(test.TARGET.mean()),policy_metrics=policy_results,runtime_seconds=round(time.time()-start,1))
    pred=test[['SK_ID_CURR','TARGET','CODE_GENDER']].copy()
    for s,p in scores.items():pred[s+'_score']=p;joblib.dump(bundles[s],ROOT/f'models/{s}.joblib')
    pred.to_parquet(ROOT/'reports/test_predictions.parquet',index=False)
    scenarios=[]
    for capacity in [5,10,15,20,30,40,50]:
        row={'capacity':capacity}
        for s,p in scores.items():
            ix=np.argsort(-p,kind='stable')[:int(np.ceil(capacity/100*len(test)))];yy=test.TARGET.to_numpy()[ix];row[s]={'queue':len(ix),'identified':int(yy.sum()),'precision':float(yy.mean()),'recall':float(yy.sum()/test.TARGET.sum()),'lift':float(yy.mean()/test.TARGET.mean())}
        scenarios.append(row)
    groups=[]
    for name,g in pred.groupby('CODE_GENDER'):
        if len(g)>=100 and g.TARGET.nunique()==2:groups.append({'group':name,'n':len(g),'prevalence':float(g.TARGET.mean()),**metrics(g.TARGET,g.selected_score.to_numpy(),bundles['selected']['threshold'])})
    (ROOT/'reports/subgroups.json').write_text(json.dumps(groups,indent=2),encoding='utf-8');(ROOT/'reports/capacity_scenarios.json').write_text(json.dumps(scenarios,indent=2),encoding='utf-8');(ROOT/'reports/metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
