"""Application-time feature contract and calibrated inference."""
import numpy as np
import pandas as pd
CAT=['NAME_CONTRACT_TYPE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE']
RAW_NUM=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_CHILDREN','REGION_RATING_CLIENT','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','QTD_CREDITOS_BUREAU','TOTAL_DIVIDA_BUREAU','MEDIA_DIAS_ATRASO_BUREAU','QTD_APLICACOES_ANTERIORES','QTD_RECUSAS_ANTERIORES','ATRASO_MEDIO_PAGAMENTO','QTD_PARCELAS_ATRASADAS']
DERIVED=['credit_income_ratio','annuity_income_ratio','refusal_rate','employment_unknown']
NUM=RAW_NUM+DERIVED
FEATURES=CAT+NUM
RAW_INPUTS=CAT+RAW_NUM
def features(frame):
    out=frame.reindex(columns=RAW_INPUTS).copy()
    for c in CAT: out[c]=out[c].fillna('Unknown').astype(str)
    for c in RAW_NUM: out[c]=pd.to_numeric(out[c],errors='coerce')
    out['employment_unknown']=(out.DAYS_EMPLOYED==365243).astype(float)
    out.loc[out.DAYS_EMPLOYED==365243,'DAYS_EMPLOYED']=np.nan
    income=out.AMT_INCOME_TOTAL.where(out.AMT_INCOME_TOTAL>0)
    out['credit_income_ratio']=out.AMT_CREDIT/income
    out['annuity_income_ratio']=out.AMT_ANNUITY/income
    out['refusal_rate']=out.QTD_RECUSAS_ANTERIORES/out.QTD_APLICACOES_ANTERIORES.where(out.QTD_APLICACOES_ANTERIORES>0)
    out[NUM]=out[NUM].replace([np.inf,-np.inf],np.nan)
    return out[FEATURES]
def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def predict(bundle,frame):
    raw=bundle['model'].predict_proba(features(frame))[:,1]
    return bundle['calibrator'].predict_proba(logit(raw))[:,1]
