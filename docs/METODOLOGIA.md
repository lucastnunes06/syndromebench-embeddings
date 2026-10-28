# Metodologia e escolhas

O pipeline recebe embeddings de 320 dimensões já extraídos de um modelo anterior. O objetivo é verificar a capacidade de métodos de vizinhos próximos diferenciarem classes nos vetores. Os dados originais não são redistribuídos neste repositório.

## 1. Ingestão

A estrutura `classe → indivíduo → imagem → vetor` é convertida em registros. Vetores com tamanho diferente de 320 ou valores não finitos são descartados; a saída processada também preserva IDs para validação do agrupamento, mas estes não devem ser publicados se forem identificadores reais.

## 2. Evitar vazamento

Uma pessoa pode fornecer múltiplas imagens. Por isso a separação deve ser feita por pessoa, não por imagem. O projeto tem 10 folds externos e seleção de hiperparâmetros nos folds internos, com transformações ajustadas apenas nos conjuntos de treino.

## 3. Comparação

São avaliados KNN cosseno e euclidiano, macro-F1, AUC one-vs-rest, Top-k, confusões e baselines. Configurações de pré-processamento diferentes impedem atribuir qualquer variação exclusivamente à métrica de distância.

## 4. Visualização

t-SNE é uma ferramenta exploratória com variação decorrente da semente e dos hiperparâmetros. A projeção não comprova utilidade médica nem desempenho fora do conjunto analisado.

## 5. Reprodutibilidade

Os testes usam dados artificiais. Para reproduzir uma avaliação quantitativa é necessário obter um conjunto de dados cujo uso seja permitido, executar `src/main.py` e gerar novas métricas e tabelas. As métricas usadas em relatórios privados anteriores não foram publicadas neste pacote.
