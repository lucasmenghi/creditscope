from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    m=json.loads((ROOT/'reports/metrics.json').read_text());d=json.loads((ROOT/'reports/data_manifest.json').read_text());f=json.loads((ROOT/'reports/frozen_config.json').read_text());n=m['selected']
    (ROOT/'README.md').write_text(f'''# CreditScope — Home Credit Case

**Lucas Menghi · Credit Risk Intelligence**

![Executive cover](assets/creditscope-cover.webp)

Classificação de dificuldade de pagamento que completa meu projeto de [ingestão e modelagem Oracle da Home Credit](https://github.com/lucasmenghi/Case_Default_Risk). O case transforma as tabelas ingeridas em uma camada de ranking, calibração e diagnóstico para análise humana de risco.

[Estudo executivo](https://lucasmenghi.github.io/portifolio/projects/creditscope.html) · [English study](https://lucasmenghi.github.io/portifolio/projects/creditscope-en.html) · [Notebook executado](notebooks/01_home_credit_case.ipynb) · [Model card](docs/model_card.md)

## Problema e dados

Fonte original: [Home Credit Default Risk, Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data), competição de 2018. São {d['rows']:,} aplicações rotuladas e {d['positives']:,} positivas. O TARGET representa dificuldade de pagamento: atraso superior a X dias em pelo menos uma das primeiras Y parcelas; X e Y não são publicados. Não equivale a uma definição universal de inadimplência. `application_test.csv` não tem rótulos e não é usado nas métricas.

O export `hc_model_features.parquet` do ETL original foi conferido: mesmos clientes e alvos. Campos adicionais, incluindo EXT_SOURCE_1/2/3, já constam do DDL de HC_APPLICATION. Os históricos são reconstruídos dos CSVs originais; o ETL anterior permanece como projeto de origem. Veja [manifesto](reports/data_manifest.json), [contrato](src/core.py) e [consulta analítica Oracle](sql/analytical_layer.sql).

## Preparação e método

- Bureau: DAYS_CREDIT e DAYS_CREDIT_UPDATE ≤ 0; decisões anteriores: DAYS_DECISION ≤ 0; parcelas: vencimento e pagamento ≤ 0. Ausentes nas datas excluídos. A auditoria excluiu 17 registros de bureau e 2.905 de parcelas. A contagem de parcelas acompanha as linhas de eventos da fonte, que podem conter pagamentos parciais; não deve ser interpretada como número de parcelas únicas.
- 26 features: quatro categorias, 18 numéricas originais e quatro derivadas. Gênero, ID e TARGET excluídos das entradas. A sentinela DAYS_EMPLOYED=365243 vira ausente e indicador. Ratios usam denominadores positivos. Imputação e escala da logística aprendidas apenas no treino.
- Separação estratificada por cliente, seed 42 e sementes documentadas nos cortes seguintes: 60% treino, 10% desenvolvimento, 10% calibração, 5% política e 15% teste. Não existe calendário global de aplicações: **não é validação temporal**.
- Quatro candidatos; seleção pela AP em desenvolvimento: **{f['selected']}**. Uma única partição de desenvolvimento, sem alegação de cross-validation. Calibração sigmoid em grupo independente; limiar no percentil 80 de outro grupo; configuração congelada antes de pontuar o teste.

## Resultado real no teste

{m['test_n']:,} clientes reservados; prevalência {m['test_positive_rate']:.2%}.

| Modelo | AP | ROC AUC | KS | Brier ↓ | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| {f['selected']} | {n['average_precision']:.3f} | {n['roc_auc']:.3f} | {n['ks']:.3f} | {n['brier']:.3f} | {n['precision_at20']:.1%} | {n['recall_at20']:.1%} | {n['lift_at20']:.2f}× |
| Logistic baseline | {m['logistic_baseline']['average_precision']:.3f} | {m['logistic_baseline']['roc_auc']:.3f} | {m['logistic_baseline']['ks']:.3f} | {m['logistic_baseline']['brier']:.3f} | {m['logistic_baseline']['precision_at20']:.1%} | {m['logistic_baseline']['recall_at20']:.1%} | {m['logistic_baseline']['lift_at20']:.2f}× |

Limiar congelado {n['threshold']:.4f}: fila {n['queue_rate']:.1%}, precision {n['precision']:.1%}, recall {n['recall']:.1%}. O cenário top 20% por ranking é distinto desse limiar e não garante capacidade futura. Resultados são históricos, sem intervalos de confiança, sem redução de perdas, lucro ou causalidade demonstrados.

## Reprodução

Python 3.11. Baixe o ZIP na competição original respeitando seus termos. Os CSVs e os IDs/previsões individuais não são redistribuídos. Os modelos e diagnósticos agregados ficam versionados. Em checkout novo:

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare.py --source /path/home-credit-default-risk.zip
python scripts/train.py
python scripts/build_report.py
python scripts/build_docs.py
pytest -q
streamlit run app.py
```

`--etl-export /path/hc_model_features.parquet` permite auditar seu export local opcional. Diretórios necessários são criados por prepare.py. O build_site.py integra o portfólio local do autor e não é necessário para reproduzir o modelo. O notebook exibe relatórios reais e foi executado; os scripts fazem o treinamento. Nunca carregue joblib de fontes não confiáveis.

## Limites

Uma semente e snapshots retrospectivos. Filtros de datas relativas reduzem risco de usar eventos futuros, mas não comprovam disponibilidade operacional de todos os atributos. Scores externos são anonimizados, com pouca explicabilidade. A idade permanece no modelo; excluir gênero não garante equidade devido a proxies. A auditoria por gênero é descritiva, não um certificado de equidade. Não usar para decisão automática de crédito. Antes de piloto: validar em outro período, disponibilidade, drift, calibração, impactos e política com especialistas.

Case independente, sem vínculo ou endosso da Home Credit. Logo: [fonte oficial](https://www.homecredit.cz/dokumenty-ke-stazeni/obecne-informace). Capa conceitual gerada com ferramenta de imagem; [prompt](assets/cover-prompt.txt). Gráficos calculados a partir dos dados reais.
''',encoding='utf-8')
    (ROOT/'docs/model_card.md').write_text(f'''# Model card — CreditScope

Author: Lucas Menghi. Version: 1.0. Selected candidate: {f['selected']}.

## Intended use
Educational credit-risk ranking and historical diagnostics for human review. Not credit approval, pricing, limits or automated denial.

## Target and data
Competition TARGET=1: payment difficulties, more than X days late in at least one of first Y installments. X/Y undisclosed. 307,511 labeled applications from the 2018 Home Credit Default Risk release. Full provenance/hash in data_manifest.json. Unlabeled application_test excluded from evaluation. No raw datasets or individual predictions published.

## Availability and features
26 features; borrower identifier, target and gender excluded. Age and anonymized external scores remain. Relative-day historical cutoffs documented in prepare.py. Installment counts refer to payment-event rows, not guaranteed unique installments. Retrospective snapshots do not prove operational point-in-time availability. Train-only imputation/scaling; employment sentinel handled explicitly.

## Evaluation design
Independent stratified customers, 60/10/10/5/15 train/development/calibration/policy/test. No global calendar, no temporal claim. Four fixed candidates, best development AP. Separate sigmoid calibration and frozen policy threshold, no test-driven retraining. One split, no confidence intervals. Test n={m['test_n']}; prevalence {m['test_positive_rate']:.4f}; ROC AUC {n['roc_auc']:.4f}; AP {n['average_precision']:.4f}; Brier {n['brier']:.4f}; KS {n['ks']:.4f}.

## Limitations and impact
Gender exclusion does not demonstrate fairness: proxies and age require review. Descriptive subgroup metrics in reports/subgroups.json. Anonymized external scores limit explanation. No demonstrated temporal generalization, ROI, loss reduction, legal compliance or causal effect. Validate availability, drift, calibration, cohort impacts and credit policy before a human-supervised pilot. This repository contains no production deployment or individual credit decisions.
''',encoding='utf-8');print('README and model card built.')
if __name__=='__main__':main()
