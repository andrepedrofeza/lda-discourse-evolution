# lda-discourse-evolution
Neste projeto, consta a implementação em *Python* do cálculo de quatro métricas de evolução de discursos a partir das distribuições de tópicos calculadas através do modelo de LDA.

Estas métricas foram propostas na dissertação de mestrado *"Measuring discourse evolution: An lda-based discourse analysis
framework."* [Menuzzo, V. A. 2023]. São elas:

- **Diversidade**: considera-se diverso um discurso que trata de vários temas com frequência semelhante, enquanto um discurso que trata com mais frequência de um ou dois tópicos é não diverso;
- **Coesão**: discursos coesos são aqueles que apresentam um alto nível de concordância com o discurso de um conjunto de entidades;
- **Dissonância**: determina até que ponto as entidades que são parte dos discursos analisados tendem a divergir dos temas principais;
- **Coerência**: captura a estabilidade do discurso ao longo do tempo.

O cálculo dessas métricas está implementado no arquivo `lib/lda_discourse_evolution.py` através da classe `MetricsCalculator`.

# Exemplo de uso

O exemplo de cálculo das métricas foi dividido em três etapas, cada uma em um arquivo do *Jupyter Notebook*, contidos na pasta `exemplo`:


1. `1-preprocessamento.ipynb`;
   
Aqui, é realizado a seleção dos dados a partir de palavras-chave, criação de colunas adicionais que são necessárias e pré-processamento nos contéudos dos *tweets*.

2. `2-distribuicao-de-topicos.ipynb`;

Neste arquivo, é realizado a aplicação do modelo de LDA e gerada a distribuição de tópicos para cada uma das postagens.

3. `3-calculo-das-metricas.ipynb`.

Nesta etapa, é realizado o cálculo das métricas, partindo das distribuições de tópico calculadas na etapa 2.

# Origem dos dados

O exemplo de uso faz o uso de um *dataset* contendo *tweets* de candidados a prefeitos de todo o Brasil, durante o ano de 2020.

Estes dados foram disponibilizados pela professora Dra. Lorena G. Barberia do Departamento de Ciências Políticas da Universidade de São Paulo (USP).

Mais detalhes sobre este conjunto de dados, incluindo a forma que ele foi obtido e os critérios de seleção, estão disponíveis no arquivo `codebook.pdf`.
