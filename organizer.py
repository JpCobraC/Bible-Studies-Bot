import json

with open("ARA.json", "r", encoding="utf-8") as f:
    dados = json.load(f)

with open("ARA_formatado.json", "w", encoding="utf-8") as f:
    json.dump(dados, f, indent=4, ensure_ascii=False)

print("Arquivo formatado com sucesso!")