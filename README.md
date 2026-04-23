# 🧠 Classificação de Imagens com CNN — PyTorch + CIFAR-10

Projeto completo de classificação de imagens desenvolvido em Python com PyTorch e Jupyter Notebook.  
Implementa uma Rede Neural Convolucional (CNN) do zero para classificar imagens do dataset CIFAR-10 em 10 categorias, cobrindo todo o ciclo: preparação dos dados, construção da arquitetura, treinamento, avaliação e uso do modelo em produção.

---

## 📁 Estrutura do Projeto

```
Classificacao-Imagens/
├── Classificacao_Imagens.ipynb   
├── modelo.pth                    
├── dados/                        
├── imagem1.jpg                   
├── imagem2.jpg                   
├── imagem3.png                   
├── imagem4.jpg                   
├── imagem5.jpg                   
└── requirements.txt              
```

---

## 🗂️ Classes do CIFAR-10

O modelo classifica imagens em 10 categorias:

`plane` • `car` • `bird` • `cat` • `deer` • `dog` • `frog` • `horse` • `ship` • `truck`

---

## ⚙️ O que o projeto faz

### 1. Configuração do Dispositivo
Detecta automaticamente o melhor hardware disponível para treinamento: GPU NVIDIA (CUDA), GPU Apple (MPS) ou CPU.

### 2. Hiperparâmetros
- **Épocas:** 50
- **Batch size:** 64
- **Learning rate:** 0.001

### 3. Preparação dos Dados
- Download automático do CIFAR-10 (50.000 imagens de treino, 10.000 de teste)
- Data augmentation no treino: `RandomHorizontalFlip`, `RandomCrop`, `ColorJitter`
- Normalização com média e desvio padrão reais do CIFAR-10

### 4. Arquitetura da CNN
Dois blocos convolucionais com BatchNormalization, seguidos de um classificador com Dropout:
- **Bloco 1:** Conv(3→32) + BN + Conv(32→64) + BN + MaxPool → 16x16
- **Bloco 2:** Conv(64→128) + BN + Conv(128→128) + BN + MaxPool → 8x8
- **Classificador:** Flatten → Dropout(0.5) → FC(8192→512) → FC(512→10)

### 5. Treinamento
- Função de perda: `CrossEntropyLoss`
- Otimizador: `Adam`
- Scheduler: `ReduceLROnPlateau` — reduz o learning rate automaticamente quando a acurácia para de melhorar

### 6. Avaliação
- Acurácia geral no conjunto de teste
- Acurácia individual por classe

### 7. Deploy e Uso
- Salvamento do modelo treinado em `modelo.pth`
- Função `classifica_imagem()` para classificar novas imagens locais com exibição da classe prevista e confiança

---

## 🚀 Como executar

### Pré-requisitos

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Rodando o notebook

```bash
jupyter notebook Classificacao_Imagens.ipynb
```

> O dataset CIFAR-10 será baixado automaticamente na primeira execução na pasta `./dados`.

---

## 🛠️ Tecnologias utilizadas

- Python 3
- PyTorch
- Torchvision
- Torchsummary
- NumPy
- Matplotlib
- Pillow
- Jupyter Notebook
