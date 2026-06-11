# 📖 Bible Studies Bot — RAG Teológico Local

Este projeto realiza a ingestão, o processamento lógico e a indexação semântica de textos bíblicos e documentos confessionais históricos em um banco de dados vetorial local utilizando a técnica de **RAG (Retrieval-Augmented Generation)**.

Toda a arquitetura de indexação e recuperação é projetada para rodar **100% offline e localmente**, utilizando a biblioteca **LangChain**, o banco vetorial **ChromaDB** e embeddings locais (**`all-MiniLM-L6-v2`** da Hugging Face), garantindo privacidade e velocidade no processamento. Para a geração pastoral dos devocionais, o projeto integra-se à API do **Google Gemini** de forma segura.

---

## 🚀 Funcionalidades Principais

- **Engenharia Avançada de Metadados:** Classificação granular de cada fragmento textual com base em sua origem teológica (Livro, Capítulo, Versículo, Pergunta, Resposta, Artigo, Erro Refutado, etc.).
- **Acoplamento Semântico Integrado:** Divisão lógica dos textos projetada para manter heresias e suas refutações ou perguntas e respostas no mesmo bloco de texto (*chunk*), evitando perda de contexto.
- **Armazenamento Persistente Local:** Banco de dados vetorial salvo na pasta `./chroma_db`, pronto para ser consumido por agentes de IA, chatbots ou sistemas de busca semântica.
- **Processamento 100% Offline:** Embeddings gerados via CPU/GPU local sem tráfego externo de dados.

---

## 🛠️ Arquitetura e Estrutura dos Dados

O script [cria_RAG.py](file:///home/jpcob/github/Bible%20Studies%20Bot/cria_RAG.py) processa dinamicamente arquivos JSON na raiz do projeto e mapeia cada um de acordo com sua estrutura interna. Veja os mapeamentos de metadados implementados:

| Origem (Arquivo JSON) | Tipo de Registro (`tipo_registro`) | Status Doutrinário (`status_doutrina`) | Metadados Adicionais Gerados | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| **[ARA.json](file:///home/jpcob/github/Bible%20Studies%20Bot/ARA.json)** | `versiculo_biblico` | `REGRA_CORRETA` | `livro`, `capitulo`, `versiculo` | Tradução bíblica Almeida Revista e Atualizada. |
| **[Confissão_de_Westminster.json](file:///home/jpcob/github/Bible%20Studies%20Bot/Confissão_de_Westminster.json)** | `confissao_secao` | `REGRA_CORRETA` | `capitulo`, `secao` | Capítulos e seções da Confissão de Westminster. |
| **[Catecismo_Maior.json](file:///home/jpcob/github/Bible%20Studies%20Bot/Catecismo_Maior.json)** | `catecismo_pr` | `REGRA_CORRETA` | `numero` | Perguntas, respostas e referências de passagens. |
| **[Cânones_de_dort.json](file:///home/jpcob/github/Bible%20Studies%20Bot/Cânones_de_dort.json)** | `artigo_ortodoxo` / `erro_refutado` | `REGRA_CORRETA` / `HERESIA_REJEITADA` | `capitulo`, `numero` | Distinção entre artigos de fé e refutações detalhadas de heresias. |
| **[Institutas_JoãoCalvino.json](file:///home/jpcob/github/Bible%20Studies%20Bot/Institutas_JoãoCalvino.json)** | `biografia` / `institutas_amostra` / `institutas_cap17` | `REGRA_CORRETA` | `secao` / `capitulo` | Biografia, sumário de capítulos e tradução do clássico de Calvino. |

### Engenharia de Divisão (*Chunking*)
Para evitar que blocos lógicos estruturados (como perguntas e respostas ou refutações doutrinárias) fossem fatiados pela metade, o sistema utiliza o `RecursiveCharacterTextSplitter` configurado com:
- **Tamanho do Chunk (`chunk_size`):** `2500` caracteres (tamanho ideal para manter a coesão semântica de artigos confessionais inteiros).
- **Sobreposição (`chunk_overlap`):** `300` caracteres (garante a continuidade de leitura em transições).

---

## 📂 Estrutura do Projeto

```text
├── chroma_db/               # Banco de dados vetorial persistido localmente (gerado pós-execução)
├── ARA.json                 # Bíblia Almeida Revista e Atualizada
├── Catecismo_Maior.json     # Catecismo Maior de Westminster
├── Confissão_de_Westminster.json # Confissão de Fé de Westminster
├── Cânones_de_dort.json     # Cânones de Dort (Artigos e Refutações)
├── Institutas_JoãoCalvino.json # Institutas da Religião Cristã (Calvino)
├── cria_RAG.py              # Script principal de extração, chunking e indexação
├── gera_devocional.py       # Script de geração de devocionais usando RAG e Google Gemini
├── organizer.py             # Script utilitário para formatar a indentação de arquivos JSON
├── .env                     # Arquivo de configuração local (chave da API do Gemini)
├── requirements.txt         # Arquivo de dependências Python
└── README.md                # Documentação do projeto
```

---

## 💻 Instalação e Configuração

### 1. Pré-requisitos
- **Python 3.10 ou superior** instalado.
- Gerenciador de pacotes `pip`.

### 2. Configurar o Ambiente Virtual

Crie e ative um ambiente virtual isolado:

```bash
# Criar o ambiente virtual (.venv)
python3 -m venv .venv

# Ativar no Linux / macOS
source .venv/bin/activate

# Ativar no Windows (Prompt de Comando)
.venv\Scripts\activate.bat

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instalar Dependências

Com o ambiente virtual ativo, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### 4. Configurar Chave de API (Google Gemini)

Crie um arquivo chamado `.env` na raiz do projeto (ou edite o arquivo criado automaticamente) e insira sua chave da API do Gemini:

```env
GOOGLE_API_KEY=sua_chave_gemini_aqui
```

---

## 🎮 Como Executar e Alimentar o RAG

Para processar todos os documentos teológicos locais e popular o banco vetorial **ChromaDB**, execute o script de ingestão:

```bash
python cria_RAG.py
```

### O que acontece durante a execução:
1. O script escaneia a pasta em busca de arquivos `.json`.
2. Lê os conteúdos e realiza o parsing inteligente baseado no formato do arquivo.
3. Se houver uma pasta `./chroma_db` antiga, ela é removida para evitar dados duplicados.
4. Gera os embeddings de cada *chunk* através do modelo `sentence-transformers/all-MiniLM-L6-v2` (baixado localmente via cache se for a primeira execução).
5. Armazena os textos, metadados e vetores no diretório `./chroma_db`.

---

## ✍️ Como Gerar Devocionais Teológicos

Após alimentar o banco vetorial executando o `cria_RAG.py`, você pode utilizar o gerador de devocionais para produzir meditações profundas e teologicamente embasadas nas escrituras e confissões:

```bash
python gera_devocional.py
```

O script irá:
1. Carregar o banco vetorial persistido localmente (`./chroma_db`).
2. Solicitar um tema ou versículo chave.
3. Buscar referências bíblicas e confessionais correspondentes no ChromaDB (RAG).
4. Gerar um devocional estruturado e fundamentado utilizando o modelo `gemini-1.5-flash`.

---

## 🔍 Exemplo de Uso: Como consultar o Banco Vetorial

Depois de gerar o banco vetorial, você pode realizar buscas semânticas nos textos teológicos locais com o seguinte trecho de código Python:

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Carregar o modelo de embeddings (o mesmo usado na criação)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Carregar o banco vetorial persistido
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Realizar uma busca por similaridade semântica
query = "O que a Bíblia e as confissões ensinam sobre a providência divina?"
resultados = vector_store.similarity_search(query, k=3)

# 4. Exibir resultados
for idx, doc in enumerate(resultados, 1):
    print(f"\n--- Resultado {idx} ---")
    print(f"Fonte: {doc.metadata.get('fonte')}")
    print(f"Tipo de Registro: {doc.metadata.get('tipo_registro')}")
    print(f"Status Doutrinário: {doc.metadata.get('status_doutrina')}")
    if "capitulo" in doc.metadata:
        print(f"Capítulo/Seção: {doc.metadata.get('capitulo')}")
    print(f"Conteúdo:\n{doc.page_content}\n")
```
