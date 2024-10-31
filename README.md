# APS 6: Predição do Preço do Bitcoin usando Regressão Linear

Este projeto realiza a predição do preço do Bitcoin com base em dados históricos, utilizando um modelo de regressão linear e otimização por descida de gradiente.

## Integrantes:
Carlos Hernani e Mateus Porto

## Estrutura do Projeto

1. **Coleta e Pré-processamento de Dados**:
   - O projeto usa a biblioteca `yfinance` para coletar dados históricos do Bitcoin.
   - Os dados são divididos em treino (70%) e teste (30%) e normalizados para garantir a estabilidade numérica durante a otimização.
   
2. **Modelo de Regressão Linear**:
   - O modelo de regressão linear prevê o preço do Bitcoin no dia seguinte com base nos preços dos três dias anteriores.
   - A função de erro é o erro quadrático médio (MSE), e os pesos são otimizados usando descida de gradiente com a biblioteca `autograd` para derivação automática.

3. **Avaliação e Visualização dos Resultados**:
   - Após o treinamento, os valores previstos e reais são comparados em um gráfico, permitindo visualizar o desempenho do modelo.

## Requisitos

As bibliotecas utilizadas no projeto estão listadas no arquivo `requirements.txt`. Para instalar todas as dependências, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

## Explicação do Código

### Estrutura Principal do Código

- **Coleta de Dados**:
   ```python
   btc_data = yf.download("BTC-USD", start="2014-09-17", end="2024-10-30", interval="1d")
   ```

- **Função para Preparar o Conjunto de Dados**:
   ```python
   def create_dataset(data, k=3):
       X, y = [], []
       for i in range(k, len(data)):
           X.append(data[i-k:i])
           y.append(data[i])
       return np.array(X), np.array(y)
   ```

- **Função de Erro (MSE)**:
   ```python
   def mse_loss(w, X, y):
       y_pred = np.dot(X, w)
       return np.mean((y - y_pred) ** 2)
   ```

### Explicação Matemática

#### 1. Regressão Linear

O modelo de regressão linear utilizado neste projeto segue a equação matemática:

\[
\hat{x} = X \cdot w^T
\]

onde:
- \( X \) é a matriz de entrada com os valores dos últimos dias,
- \( w \) é o vetor de pesos,
- \( \hat{x} \) é o valor previsto (preço do Bitcoin no próximo dia).

#### 2. Função de Erro (Erro Quadrático Médio - MSE)

A função de erro escolhida é o Erro Quadrático Médio (MSE), que mede a média das diferenças ao quadrado entre os valores reais (\( y \)) e os valores preditos (\( \hat{y} \)):

\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

onde:
- \( N \) é o número total de pontos no conjunto de treino,
- \( y \) são os valores reais,
- \( \hat{y} \) são os valores previstos pelo modelo.

#### 3. Descida de Gradiente

Para minimizar a função de erro, utilizamos a técnica de descida de gradiente. Em cada iteração, os pesos \( w \) são atualizados na direção oposta ao gradiente da função de erro com relação a \( w \):

\[
w = w - \text{learning\_rate} \cdot \nabla \text{MSE}
\]

onde:
- \( \nabla \text{MSE} \) é o gradiente do MSE em relação a \( w \),
- `learning_rate` é um fator que controla a velocidade de atualização dos pesos.

#### 4. Normalização dos Dados

Para garantir a estabilidade numérica, normalizamos os dados antes de aplicá-los ao modelo. Isso envolve subtrair a média e dividir pelo desvio padrão:

\[
x_{\text{normalizado}} = \frac{x - \text{média}(x)}{\text{desvio\_padrão}(x)}
\]

Esta normalização permite que o modelo aprenda de maneira mais eficaz, reduzindo problemas com valores extremos.

---

## Execução do Projeto

Para rodar o projeto, basta executar o script principal após instalar as dependências. Os gráficos serão gerados automaticamente para visualização da comparação entre valores reais e preditos.

Arquivo predicao.ipynb

