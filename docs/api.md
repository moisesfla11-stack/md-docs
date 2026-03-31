import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Configuração de dispositivo (GPU se disponível)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "vikhyatk/moondream2"

# 1. Carregar Modelo e Tokenizer
print("Carregando o Moondream2...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    revision="2024-03-06" # Versão estável
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-03-06")

def identificar_alteracoes(caminho_imagem, prompt):
    image = Image.open(caminho_imagem)
    enc_image = model.encode_image(image)
    
    # O Moondream responde bem a perguntas diretas sobre mudanças
    # Exemplo: "What has changed in this image compared to a standard room?"
    resposta = model.answer_question(enc_image, prompt, tokenizer)
    
    return resposta

def localizar_objetos(caminho_imagem, objeto):
    image = Image.open(caminho_imagem)
    enc_image = model.encode_image(image)
    
    # Retorna coordenadas [y_min, x_min, y_max, x_max]
    detalhes = model.detect_object(enc_image, objeto, tokenizer)
    
    return detalhes

# --- Exemplo de Uso ---
img_path = "sua_imagem.jpg"

# Pergunta descritiva
print("Análise:", identificar_alteracoes(img_path, "Describe any anomalies or changes in this scene."))

# Localização de um ponto específico de alteração (ex: um buraco ou mancha)
pontos = localizar_objetos(img_path, "area with alteration")
print(f"Coordenadas da alteração: {pontos}")
