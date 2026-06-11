import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"

# 1. Carregar o banco vetorial existente
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# Buscar mais blocos (k=7) para o devocional ter bastante estofo teológico
retriever = vector_store.as_retriever(search_kwargs={"k": 7})

# Usando uma temperatura ligeiramente maior (0.5) para a escrita ser mais pastoral e fluida
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# 2. O Prompt de Engenharia Devocional
system_prompt = (
    "Você é um pastor e teólogo reformado focado na edificação espiritual e na precisão bíblica.\n"
    "Sua tarefa é escrever um DEVOCIONAL DIÁRIO PROFUNDO, EXTENSO E RELEVANTE com base no contexto fornecido.\n\n"
    "O devocional deve ser rigorosamente estruturado nas seguintes seções (use os títulos abaixo):\n\n"
    "### 📖 1. O TEXTO BÍBLICO\n"
    "(Cite textualmente o versículo bíblico principal encontrado no contexto que se relaciona com o tema do usuário).\n\n"
    "### 🔍 2. EXPOSIÇÃO E CONTEXTO TEOLÓGICO\n"
    "(Explique o significado profundo do texto. Faça conexões explícitas com a teologia dos Símbolos de Fé "
    "encontrados no contexto [Confissão de Westminster, Catecismo ou Cânones de Dort]. Mostre como a doutrina "
    "molda a nossa compreensão desse texto).\n\n"
    "### 💡 3. MEDITAÇÃO E REFLEXÃO PASTORAL\n"
    "(Crie um texto reflexivo longo, confrontando o coração do homem moderno. Faça perguntas penetrantes sobre "
    "a vida prática, ídolos do coração, a tendência humana de se desviar e a autossuficiência. Use metáforas ricas).\n\n"
    "### 🎯 4. APLICAÇÃO PRÁTICA\n"
    "(Apresente 3 atitudes ou passos práticos e objetivos que o cristão deve tomar hoje em sua rotina baseando-se no texto).\n\n"
    "### 🙏 5. ORAÇÃO DIRIGIDA\n"
    "(Escreva uma oração fervorosa, humilde e teologicamente centrada na graça de Deus, respondendo ao que foi meditado).\n\n"
    "Regra de Ouro: Baseie-se apenas nas verdades e textos contidos no contexto. Não invente dados fora do banco.\n\n"
    "Contexto Teológico Recuperado:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Crie um devocional profundo baseado no seguinte tema ou versículo: {input}"),
])

# 3. Criar a corrente do RAG
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. Execução
print("\n" + "="*50)
print("🕊️  GERADOR DE DEVOCIONAIS REFORMADOS")
print("="*50 + "\n")

tema_usuario = input("Digite o versículo ou tema chave para o devocional (ex: Salmos 143:8 ou 'Ansiedade'): ")

if tema_usuario.strip():
    print("\n🔍 Consultando as Escrituras e Padrões de Westminster...")
    print("✍️  Escrevendo devocional estendido...\n")
    
    resposta = rag_chain.invoke({"input": tema_usuario})
    
    print(resposta["answer"])
    print("\n" + "="*50)