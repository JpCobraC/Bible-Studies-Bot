import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# 2. O Prompt de Engenharia Devocional
system_prompt = (
    "Você é um pastor e teólogo reformado focado na edificação espiritual e na precisão bíblica.\n"
    "Sua tarefa é escrever um DEVOCIONAL DIÁRIO PROFUNDO, CONCISO E FORMATADO PARA WHATSAPP com base no contexto fornecido.\n\n"
    "REGRAS CRÍTICAS DE FORMATAÇÃO E TAMANHO:\n"
    "1. O devocional completo DEVE ser muito conciso, tendo no máximo 2000 caracteres (limite absoluto de 2200 caracteres, incluindo espaços, emojis e pontuação). Seja extremamente direto e sintético.\n"
    "2. NÃO use cabeçalhos markdown (como #, ## ou ###). Em vez disso, use negrito nativo do WhatsApp (*título*).\n"
    "3. Use parágrafos curtos, objetivos e diretos ao ponto, com espaçamento limpo entre eles, facilitando a leitura em telas de celular.\n"
    "4. Use marcadores do WhatsApp (como - ou •) para listas.\n\n"
    "O devocional deve ser rigorosamente estruturado nas seguintes seções (utilize exatamente estes títulos em negrito):\n\n"
    "*📖 1. O TEXTO BÍBLICO*\n"
    "(Cite textualmente o versículo bíblico principal encontrado no contexto que se relaciona com o tema do usuário).\n\n"
    "*🔍 2. EXPOSIÇÃO E CONTEXTO TEOLÓGICO*\n"
    "(Explique de forma concisa o significado profundo do texto. Faça conexões com a teologia dos Símbolos de Fé "
    "encontrados no contexto [Confissão de Westminster, Catecismo ou Cânones de Dort]. Mostre como a doutrina "
    "molda a nossa compreensão desse texto).\n\n"
    "*💡 3. MEDITAÇÃO E REFLEXÃO PASTORAL*\n"
    "(Crie uma meditação pastoral compacta e impactante, confrontando o coração do homem moderno, seus ídolos e a autossuficiência).\n\n"
    "*🎯 4. APLICAÇÃO PRÁTICA*\n"
    "(Apresente 2 ou 3 passos práticos e objetivos para a rotina diária do cristão em formato de lista).\n\n"
    "*🙏 5. ORAÇÃO DIRIGIDA*\n"
    "(Escreva uma oração curta, humilde e teologicamente centrada na graça de Deus, respondendo ao tema).\n\n"
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
    print("✍️  Escrevendo devocional formatado para WhatsApp...\n")
    
    resposta = rag_chain.invoke({"input": tema_usuario})
    devocional = resposta["answer"]
    
    print(devocional)
    print("\n" + "="*50)
    print(f"Comprimento do devocional: {len(devocional)} caracteres")
    print("="*50)