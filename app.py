import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Configuración de la página
st.set_page_config(page_title="Sistema de Clasificación Facial", page_icon="👤", layout="centered")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 2)
    # Buscamos el archivo en la carpeta local
    if os.path.exists('modelo_mateo.pth'):
        m.load_state_dict(torch.load('modelo_mateo.pth', map_location='cpu'))
        m.eval()
        return m
    else:
        st.error("⚠️ Archivo 'modelo_mateo.pth' no encontrado en el directorio.")
        return None

model = load_model()
labels = ["Fondo / Otra Persona", "Mateo"]

# --- LÓGICA DE PREDICCIÓN ---
def predict(img):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = t(img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        out = model(img_t)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    
    res_label = labels[pred.item()]
    confianza = conf.item()

    # Umbral de seguridad del 90% para evitar errores con fotos de otros
    if res_label == "Mateo" and confianza < 0.90:
        return "Fondo / Otra Persona", confianza
    
    return res_label, confianza

# --- INTERFAZ DE USUARIO ---
st.title("🛡️ Panel de Validación Facial")
st.markdown(f"**Estudiante:** Mateo Oñate | **Lab 03:** Transfer Learning with PyTorch and Model Improvement")
st.write("---")

st.subheader("📁 Cargar Imágenes para Validación")
st.info("Sube las fotos correspondientes a los Tests 1, 2 y 3 (Mateo, Fondo y Celebridad).")

archivo = st.file_uploader("Selecciona una imagen (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'])

if archivo:
    # Mostrar la imagen seleccionada
    img_mostrar = Image.open(archivo)
    st.image(img_mostrar, caption="Vista previa de la imagen", width=350)
    
    # Ejecutar predicción
    with st.spinner('Analizando rasgos biométricos...'):
        res, c = predict(img_mostrar)
        porcentaje = c * 100

    # Mostrar resultados
    st.write("### Resultado del Análisis:")
    if res == "Mateo":
        st.success(f"✅ **IDENTIFICADO: {res}**")
        st.metric("Nivel de Confianza", f"{porcentaje:.2f}%")
        st.balloons()
    else:
        st.warning(f"⚠️ **CLASIFICACIÓN: {res}**")
        st.metric("Nivel de Confianza", f"{porcentaje:.2f}%")

st.write("---")
st.caption("Laboratorio de Visión Computacional")
