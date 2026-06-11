# 📖 Bible Studies Bot (RAG Teológico)

Este projeto realiza o processamento e a indexação de documentos teológicos e do texto bíblico em um banco de dados de vetores local utilizando a técnica de **RAG (Retrieval-Augmented Generation)**.

O sistema é construído de forma inteiramente local e offline utilizando **LangChain**, **ChromaDB** e embeddings locais (**all-MiniLM-L6-v2** da HuggingFace), sem a necessidade de chaves de API ou comunicação externa com a OpenAI.

---

## 🚀 Funcionalidades

- **Busca Semântica Avançada:** Processa trechos da Bíblia e confissões de fé históricas, permitindo buscas e recuperações baseadas no significado dos textos.
- **Processamento Offline:** Gera os vetores e o banco de dados localmente no seu computador.
- **Banco de Dados Local:** Salva a base indexada na pasta `chroma_db/` para consumo futuro por agentes de IA ou sistemas de RAG.

---

## 📂 Estrutura do Projeto

*   [cria_RAG.py](cria_RAG.py): Script principal que processa os documentos JSON, gera os fragmentos (chunks) e popula a base vetorial ChromaDB.
*   [organizer.py](organizer.py): Script utilitário para formatação e indentação do arquivo `ARA.json`.
*   [requirements.txt](requirements.txt): Lista de dependências e bibliotecas Python necessárias (incluindo `sentence-transformers`).
*   **Base de Conhecimento (Arquivos JSON):**
    *   [ARA.json](ARA.json): Bíblia na tradução Almeida Revista e Atualizada.
    *   [Institutas_JoãoCalvino.json](Institutas_JoãoCalvino.json): As Institutas da Religião Cristã, de João Calvino.
    *   [Catecismo_Maior.json](Catecismo_Maior.json): Catecismo Maior de Westminster.
    *   [Confissão_de_Westminster.json](Confissão_de_Westminster.json): Confissão de Fé de Westminster.
    *   [Cânones_de_dort.json](Cânones_de_dort.json): Cânones de Dort.

---

## 🛠️ Instalação e Configuração

### 1. Requisitos Prévios

Certifique-se de ter o **Python 3.10 ou superior** instalado em sua máquina.

### 2. Configurar o Ambiente Virtual

Crie e ative um ambiente virtual para isolar as dependências do projeto:

```bash
# Criar o ambiente virtual
python3 -m venv .venv

# Ativar no Linux/macOS
source .venv/bin/activate

# Ativar no Windows (Prompt de Comando)
.venv\Scripts\activate.bat

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instalar Dependências

Com o ambiente virtual ativo, instale as dependências executando:

```bash
pip install -r requirements.txt
```

---

## 🎮 Como Usar

Para ler os documentos e gerar o banco de vetores na pasta local `chroma_db/`, basta executar:

```bash
python cria_RAG.py
```

O script limpará qualquer resíduo antigo na pasta `./chroma_db` para evitar dados duplicados, carregará os arquivos `.json`, fará a quebra em fragmentos (chunks) e populará a base vetorial usando o modelo de embeddings local.

