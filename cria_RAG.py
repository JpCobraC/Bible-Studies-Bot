import os
import json
import glob
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db"

def carrega_documentos_json():
    documentos_langchain = []
    arquivos_json = glob.glob("*.json")
    
    for arquivo in arquivos_json:
        nome_base = os.path.basename(arquivo)
        print(f"Processando arquivo: {nome_base}...")
        
        with open(arquivo, 'r', encoding='utf-8') as f:
            try:
                conteudo = json.load(f)
                
                # --- 1. GLOSSÁRIOS ---
                # Se o arquivo contiver a chave "glossario", mapeamos termo e definição
                if isinstance(conteudo, dict) and "glossario" in conteudo:
                    glossario = conteudo["glossario"]
                    for termo, definicao in glossario.items():
                        page_content = f"Glossário - Termo: {termo}\nDefinição: {definicao}"
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                "fonte": nome_base,
                                "tipo_registro": "glossario",
                                "status_doutrina": "REGRA_CORRETA",
                                "termo": termo
                            }
                        )
                        documentos_langchain.append(doc)

                # --- 2. CATECISMOS (Perguntas e Respostas) ---
                # Mapeia chaves de perguntas e respostas, garantindo que fiquem no mesmo bloco
                if isinstance(conteudo, dict) and ("perguntas_respostas" in conteudo or "perguntas_e_respostas" in conteudo):
                    chave_pr = "perguntas_respostas" if "perguntas_respostas" in conteudo else "perguntas_e_respostas"
                    for item in conteudo[chave_pr]:
                        numero = item.get("numero", "")
                        pergunta = item.get("pergunta", "")
                        resposta = item.get("resposta", "")
                        referencias = item.get("referencias", [])
                        
                        ref_str = f"\nReferências Bíblicas: {', '.join(referencias)}" if referencias else ""
                        page_content = f"Catecismo - Pergunta {numero}: {pergunta}\nResposta: {resposta}{ref_str}"
                        
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                "fonte": nome_base,
                                "tipo_registro": "catecismo_pr",
                                "status_doutrina": "REGRA_CORRETA",
                                "numero": numero
                            }
                        )
                        documentos_langchain.append(doc)

                # --- 3. CÂNONES DE DORT (Artigos Ortodoxos vs. Erros Rejeitados) ---
                if isinstance(conteudo, dict) and "conteudo" in conteudo and nome_base == "Cânones_de_dort.json":
                    for cap in conteudo["conteudo"]:
                        capitulo_nome = cap.get("capitulo", "")
                        capitulo_titulo = cap.get("titulo", "")
                        
                        # Processamento de Artigos Ortodoxos
                        if "artigos" in cap:
                            for art in cap["artigos"]:
                                num_art = art.get("numero", "")
                                tit_art = art.get("titulo", "")
                                texto_art = art.get("texto", "")
                                
                                page_content = f"Cânones de Dort - {capitulo_nome} ({capitulo_titulo})\nArtigo {num_art}: {tit_art}\nTexto: {texto_art}"
                                doc = Document(
                                    page_content=page_content,
                                    metadata={
                                        "fonte": nome_base,
                                        "tipo_registro": "artigo_ortodoxo",
                                        "status_doutrina": "REGRA_CORRETA",
                                        "capitulo": capitulo_nome,
                                        "numero": num_art
                                    }
                                )
                                documentos_langchain.append(doc)
                        
                        # Processamento de Erros / Heresias (Erro e Refutação colados no mesmo Document)
                        if "rejeicao_erros" in cap:
                            for err in cap["rejeicao_erros"]:
                                num_err = err.get("erro", "")
                                descricao_err = err.get("descricao", "")
                                refutacao_err = err.get("refutacao", "")
                                
                                # Acoplamento forte no conteúdo da página para evitar fragmentação
                                page_content = (
                                    f"Cânones de Dort - {capitulo_nome} ({capitulo_titulo})\n"
                                    f"Erro/Tese Rejeitada {num_err}\n"
                                    f"VISÃO REJEITADA (HERESIA/ERRO): {descricao_err}\n"
                                    f"REFUTAÇÃO ORTODOXA (CORRETO): {refutacao_err}"
                                )
                                
                                doc = Document(
                                    page_content=page_content,
                                    metadata={
                                        "fonte": nome_base,
                                        "tipo_registro": "erro_refutado",
                                        "status_doutrina": "HERESIA_REJEITADA",
                                        "capitulo": capitulo_nome,
                                        "numero": num_err
                                    }
                                )
                                documentos_langchain.append(doc)

                # --- 4. CONFISSÃO DE FÉ DE WESTMINSTER (Capítulos e Seções) ---
                if isinstance(conteudo, dict) and "capitulos" in conteudo and nome_base == "Confissão_de_Westminster.json":
                    for cap in conteudo["capitulos"]:
                        num_cap = cap.get("numero", "")
                        tit_cap = cap.get("titulo", "")
                        secoes = cap.get("secoes", [])
                        
                        for idx, secao_texto in enumerate(secoes, start=1):
                            page_content = f"Confissão de Fé de Westminster - Capítulo {num_cap} ({tit_cap})\nSeção {idx}: {secao_texto}"
                            doc = Document(
                                page_content=page_content,
                                metadata={
                                    "fonte": nome_base,
                                    "tipo_registro": "confissao_secao",
                                    "status_doutrina": "REGRA_CORRETA",
                                    "capitulo": num_cap,
                                    "secao": idx
                                }
                            )
                            documentos_langchain.append(doc)

                # --- 5. BÍBLIA ARA (Livros, Capítulos e Versículos) ---
                if isinstance(conteudo, list) and len(conteudo) > 0 and isinstance(conteudo[0], dict) and "chapters" in conteudo[0]:
                    for livro in conteudo:
                        nome_livro = livro.get("name", "")
                        for num_cap, capitulo in enumerate(livro.get("chapters", []), start=1):
                            for num_ver, texto_ver in enumerate(capitulo, start=1):
                                page_content = f"Bíblia ARA - {nome_livro} {num_cap}:{num_ver}\nTexto: {texto_ver}"
                                doc = Document(
                                    page_content=page_content,
                                    metadata={
                                        "fonte": nome_base,
                                        "tipo_registro": "versiculo_biblico",
                                        "status_doutrina": "REGRA_CORRETA",
                                        "livro": nome_livro,
                                        "capitulo": num_cap,
                                        "versiculo": num_ver
                                    }
                                )
                                documentos_langchain.append(doc)

                # --- 6. INSTITUTAS DE JOÃO CALVINO (Mapeamento Customizado) ---
                if nome_base == "Institutas_JoãoCalvino.json":
                    # Biografia
                    if "biografia_joao_calvino" in conteudo:
                        bio = conteudo["biografia_joao_calvino"]
                        cronologia = bio.get("cronologia", [])
                        crono_str = "\n".join([f"- {c.get('ano', c.get('periodo', ''))}: {c.get('evento', '')}" for c in cronologia])
                        page_content = f"Biografia de João Calvino ({bio.get('nome_original', '')})\nNascimento: {bio.get('local_nascimento', '')} em {bio.get('data_nascimento', '')}\nFalecimento: {bio.get('data_falecimento', '')}\nCronologia:\n{crono_str}"
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                "fonte": nome_base,
                                "tipo_registro": "biografia",
                                "status_doutrina": "REGRA_CORRETA"
                            }
                        )
                        documentos_langchain.append(doc)

                    # Capítulos de Amostra
                    if "conteudo_capitulos_amostra" in conteudo:
                        amostra = conteudo["conteudo_capitulos_amostra"]
                        for cap_key, cap_val in amostra.items():
                            titulo_cap = cap_key.replace("capitulo_", "").replace("_", " ").title()
                            tese = cap_val.get("tese_principal", "")
                            pontos = cap_val.get("pontos_chave", [])
                            pontos_str = "\n".join([f"- {p}" for p in pontos])
                            page_content = f"Institutas de João Calvino - {titulo_cap}\nTese Principal: {tese}\nPontos Chave:\n{pontos_str}"
                            doc = Document(
                                page_content=page_content,
                                metadata={
                                    "fonte": nome_base,
                                    "tipo_registro": "institutas_amostra",
                                    "status_doutrina": "REGRA_CORRETA",
                                    "capitulo": titulo_cap
                                }
                            )
                            documentos_langchain.append(doc)

                    # Capítulo XVII Vida Cristã Completo
                    if "capitulo_XVII_vida_crista_completo" in conteudo:
                        cap17 = conteudo["capitulo_XVII_vida_crista_completo"]
                        for sec_key, sec_val in cap17.items():
                            titulo_sec = sec_key.replace("_", " ").title()
                            if isinstance(sec_val, dict):
                                sec_str = "\n".join([f"{k.replace('_', ' ').title()}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in sec_val.items()])
                            elif isinstance(sec_val, list):
                                sec_str = "\n".join([f"- {item}" for item in sec_val])
                            else:
                                sec_str = str(sec_val)
                            
                            page_content = f"Institutas de João Calvino - Capítulo XVII (Vida Cristã) - {titulo_sec}\n{sec_str}"
                            doc = Document(
                                page_content=page_content,
                                metadata={
                                    "fonte": nome_base,
                                    "tipo_registro": "institutas_cap17",
                                    "status_doutrina": "REGRA_CORRETA",
                                    "secao": titulo_sec
                                }
                            )
                            documentos_langchain.append(doc)

                    # Considerações editoriais e depoimentos
                    if "consideracoes_historicas_editoriais" in conteudo:
                        cons = conteudo["consideracoes_historicas_editoriais"]
                        for t in cons.get("testemunhos_historicos", []):
                            page_content = f"Testemunho Histórico sobre as Institutas\nAutor: {t.get('autor', '')}\nDepoimento: {t.get('depoimento', '')}"
                            doc = Document(
                                page_content=page_content,
                                metadata={"fonte": nome_base, "tipo_registro": "testemunho_historico", "status_doutrina": "REGRA_CORRETA"}
                            )
                            documentos_langchain.append(doc)
                        
                        hist = cons.get("breve_historia_das_institutas", {})
                        if hist:
                            page_content = f"História das Institutas\nEdição Definitiva (1559): {hist.get('edicao_definitiva_1559', '')}\nSaúde do Autor: {hist.get('saude_do_autor', '')}"
                            doc = Document(
                                page_content=page_content,
                                metadata={"fonte": nome_base, "tipo_registro": "historia_institutas", "status_doutrina": "REGRA_CORRETA"}
                            )
                            documentos_langchain.append(doc)

                # --- 7. FALLBACK GERAL (Prevenção contra perda de dados de JSON genéricos) ---
                docs_gerados = [d for d in documentos_langchain if d.metadata.get("fonte") == nome_base]
                if len(docs_gerados) == 0:
                    print(f"Aviso: Usando fallback genérico para {nome_base}")
                    texto_completo = json.dumps(conteudo, ensure_ascii=False, indent=2)
                    doc = Document(
                        page_content=texto_completo,
                        metadata={
                            "fonte": nome_base,
                            "tipo_registro": "generico",
                            "status_doutrina": "REGRA_CORRETA"
                        }
                    )
                    documentos_langchain.append(doc)

            except json.JSONDecodeError:
                print(f"Erro ao decodificar JSON: {nome_base}")
                
    return documentos_langchain

def inicializa_rag():
    docs_brutos = carrega_documentos_json()
    if not docs_brutos:
        print("Nenhum documento encontrado.")
        return None

    # Engenharia de Chunking: Definimos um tamanho de chunk suficiente (2500 caracteres)
    # para garantir que os blocos de erros/refutações, Q&A de catecismos e capítulos inteiros 
    # de confissões não sejam fatiados no meio, o que quebraria a amarração do contexto.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, 
        chunk_overlap=300,
        length_function=len
    )
    documentos_fragmentados = text_splitter.split_documents(docs_brutos)
    print(f"Total de fragmentos gerados após divisão lógica: {len(documentos_fragmentados)}")

    if os.path.exists(PERSIST_DIRECTORY): 
        print(f"Removendo banco vetorial antigo em '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    print("Inicializando gerador de embeddings locais e populando banco Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=documentos_fragmentados,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Base de dados vetorial criada e salva em: {PERSIST_DIRECTORY}")
    return vector_store

if __name__ == "__main__":
    print("Iniciando pipeline de criação RAG com Engenharia de Metadados...")
    db = inicializa_rag()
    if db is not None:
        print("Pipeline finalizado com sucesso!")
    else:
        print("Erro na execução do pipeline.")