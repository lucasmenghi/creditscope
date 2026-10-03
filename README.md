# CreditScope — Home Credit Case

**Lucas Menghi · Credit Risk Intelligence**

![Executive cover](assets/creditscope-cover.webp)

Classificação de dificuldade de pagamento que completa meu projeto de [ingestão e modelagem Oracle da Home Credit](https://github.com/lucasmenghi/Case_Default_Risk). O case transforma as tabelas ingeridas em uma camada de ranking, calibração e diagnóstico para análise humana de risco.

[Estudo executivo](https://lucasmenghi.github.io/portifolio/projects/creditscope.html) · [English study](https://lucasmenghi.github.io/portifolio/projects/creditscope-en.html) · [Notebook executado](notebooks/01_home_credit_case.ipynb) · [Model card](docs/model_card.md)

## Problema e dados

Fonte original: [Home Credit Default Risk, Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data), competição de 2018. São 307,511 aplicações rotuladas e 24,825 positivas. O TARGET representa dificuldade de pagamento: atraso superior a X dias em pelo menos uma das primeiras Y parcelas; X e Y não são publicados. Não equivale a uma definição universal de inadimplência. `application_test.csv` não tem rótulos e não é usado nas métricas.

O export `hc_model_features.parquet` do ETL original foi conferido: mesmos clientes e alvos. Campos adicionais, incluindo EXT_SOURCE_1/2/3, já constam do DDL de HC_APPLICATION. Os históricos são reconstruídos dos CSVs originais; o ETL anterior permanece como projeto de origem. Veja [manifesto](reports/data_manifest.json), [contrato](src/core.py) e [consulta analítica Oracle](sql/analytical_layer.sql).

## Preparação e método

- Bureau: DAYS_CREDIT e DAYS_CREDIT_UPDATE ≤ 0; decisões anteriores: DAYS_DECISION ≤ 0; parcelas: vencimento e pagamento ≤ 0. Ausentes nas datas excluídos. A auditoria excluiu 17 registros de bureau e 2.905 de parcelas. A contagem de parcelas acompanha as linhas de eventos da fonte, que podem conter pagamentos parciais; não deve ser interpretada como número de parcelas únicas.
- 26 features: quatro categorias, 18 numéricas originais e quatro derivadas. Gênero, ID e TARGET excluídos das entradas. A sentinela DAYS_EMPLOYED=365243 vira ausente e indicador. Ratios usam denominadores positivos. Imputação e escala da logística aprendidas apenas no treino.
- Separação estratificada por cliente, seed 42 e sementes documentadas nos cortes seguintes: 60% treino, 10% desenvolvimento, 10% calibração, 5% política e 15% teste. Não existe calendário global de aplicações: **não é validação temporal**.
- Quatro candidatos; seleção pela AP em desenvolvimento: **lightgbm_31**. Uma única partição de desenvolvimento, sem alegação de cross-validation. Calibração sigmoid em grupo independente; limiar no percentil 80 de outro grupo; configuração congelada antes de pontuar o teste.

## Resultado real no teste

46,128 clientes reservados; prevalência 8.07%.

| Modelo | AP | ROC AUC | KS | Brier ↓ | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| lightgbm_31 | 0.255 | 0.764 | 0.396 | 0.067 | 21.2% | 52.5% | 2.62× |
| Logistic baseline | 0.235 | 0.749 | 0.376 | 0.068 | 20.5% | 50.7% | 2.53× |

Limiar congelado 0.1168: fila 20.4%, precision 21.0%, recall 53.1%. O cenário top 20% por ranking é distinto desse limiar e não garante capacidade futura. Resultados são históricos, sem intervalos de confiança, sem redução de perdas, lucro ou causalidade demonstrados.

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
