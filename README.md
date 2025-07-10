## APLICAÇÃO DE CNNs NA CLASSIFICAÇÃO DO CARCINOMA PULMONAR DE CÉLULAS NÃO PEQUENAS

requisitos

* python (version 3.13.0 or later)
* [dataset balanceado]()
* [dataset dividido em treino e teste](https://drive.google.com/drive/folders/1bAshdY1GdLHfQswjQnLCSHxjEIaQXX7Q?usp=drive_link)
* [dataset divido por categoria + cvs com dados do dataset](https://drive.google.com/file/d/1Tryg4cN30dSADB6r-yJrz-2TpBZJvpFD/view?usp=sharing)
* ```
  pip install -r requirements.txt
  ```

## OBJETIVO GERAL

Aplicar e avaliar modelos de CNNs na classificação do carcinoma pulmonar de não pequenas células em imagens de tomografia computadorizada, a fim de investigar seu desempenho como ferramenta de apoio ao diagnóstico.

Objetivos Especificos

1. Realizar o pré-processamento do conjunto de dados;
2. Utilizar técnicas de aumento de dados:
3. Treinar e validar CNNs, utilizando técnicas de ajuste de hiperparâmetros;
4. Avaliar o desempenho dos modelos aplicados e compará-los com a literatura;
5. Analisar e discutir os resultados, destacando os benefícios e limitações do uso de CNNs.

## MOTIVAÇÃO

Câncer pulmonar de células não pequenas (CPCNP), responsável por cerca de 85% dos casos de cancer.

- Neste segundo grupo estão incluídos os subtipos adenocarcinoma, carcinoma de células escamosas e carcinoma de grandes células.
- O principal fator de risco para o desenvolvimento desse tipo de tumor é o tabagismo, associado a cerca de 85% dos casos diagnosticados.

LIMITAÇÕES E TRABALHOS FUTUROS:
a falta de no dataset carcinoma de grandes células.
implementar carcinoma de grandes células na classificação

# Dataset Info

desbalanceamento de classes:
total: 95 pacientes e 330 nódulos anotados, 308 fatias

0  -   Benignos (103)

1  -   Adenocarcinoma (172)

2   -   Carcinoma de Células Escamosas (33)

técnicas sugeridas pela literatura para desbalanciamento:  GANs & Autoencoders

o dataset 2D ja veio particionado assim,:

TRAIN : 80%

Pasta: dataset/train/0 - 83 imagens

Pasta: dataset/train/1 - 137 imagens

Pasta: dataset/train/2 - 26 imagens

TEST : 20%

Pasta: dataset/test/0 - 20 imagens

Pasta: dataset/test/1 - 35 imagens

Pasta: dataset/test/2 - 7 imagens
