"""Local model demonstration with a synthetic profile; no real customer data collected."""
from pathlib import Path
import json,joblib,pandas as pd,streamlit as st
from src.core import predict
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='CreditScope — Home Credit Case',layout='wide')
st.image(str(ROOT/'assets/creditscope-cover.webp'),width='stretch')
st.title('Home Credit Case · CreditScope')
st.write('Demonstração local de classificação de dificuldade de pagamento. Perfil sintético; não representa uma decisão de aprovação de crédito.')
metrics=json.loads((ROOT/'reports/metrics.json').read_text())['selected']
cols=st.columns(3)
for c,label,value in zip(cols,['ROC AUC no teste','AP no teste','Lift · top 20%'],[f"{metrics['roc_auc']:.3f}",f"{metrics['average_precision']:.3f}",f"{metrics['lift_at20']:.2f}×"]):c.metric(label,value)
with st.form('profile'):
    a,b=st.columns(2)
    income=a.number_input('Renda anual (unidade monetária da fonte)',min_value=1000.,value=180000.,step=1000.)
    credit=b.number_input('Valor do crédito',min_value=1000.,value=500000.,step=1000.)
    annuity=a.number_input('Anuidade',min_value=1.,value=25000.,step=1000.)
    age=b.slider('Idade (anos)',20,75,40)
    contract=a.selectbox('Contrato',['Cash loans','Revolving loans'])
    e1=a.slider('EXT_SOURCE_1 (score anonimizado)',0.,1.,.5)
    e2=b.slider('EXT_SOURCE_2 (score anonimizado)',0.,1.,.5)
    e3=a.slider('EXT_SOURCE_3 (score anonimizado)',0.,1.,.5)
    submitted=st.form_submit_button('Estimar risco do perfil sintético')
st.caption('Demais atributos fixados: Working, Secondary / secondary special, House / apartment; emprego de 5 anos, nenhum filho, região 2, preço do bem igual ao crédito. Históricos sem informação são ausentes, não zero. Os scores externos são entradas demonstrativas, não produzidos por este aplicativo.')
if submitted:
    bundle=joblib.load(ROOT/'models/selected.joblib')
    frame=pd.DataFrame([dict(NAME_CONTRACT_TYPE=contract,NAME_INCOME_TYPE='Working',NAME_EDUCATION_TYPE='Secondary / secondary special',NAME_HOUSING_TYPE='House / apartment',AMT_INCOME_TOTAL=income,AMT_CREDIT=credit,AMT_ANNUITY=annuity,AMT_GOODS_PRICE=credit,DAYS_BIRTH=-age*365.25,DAYS_EMPLOYED=-5*365.25,CNT_CHILDREN=0,REGION_RATING_CLIENT=2,EXT_SOURCE_1=e1,EXT_SOURCE_2=e2,EXT_SOURCE_3=e3)])
    probability=float(predict(bundle,frame)[0]);st.metric('Probabilidade estimada de dificuldade de pagamento',f'{probability:.1%}')
    st.write('Acima do limiar histórico de acompanhamento.' if probability>=bundle['threshold'] else 'Abaixo do limiar histórico de acompanhamento.');st.caption(f"Limiar congelado: {bundle['threshold']:.3f}. Não garante fila de 20% em outro contexto. Avaliação histórica por cliente, sem validação temporal.")
