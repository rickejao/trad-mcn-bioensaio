import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Trad-MCN Bioensaio",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Sistema de Bioensaio Trad-MCN (Streamlit)")
st.markdown(
    "Avaliação citogenética automatizada em lâminas de *Tradescantia pallida* "
    "usando pipeline hierárquico (Detector de Campo + Classificador Celular)."
)


@st.cache_resource
def carregar_modelos():
  det = YOLO("detector.pt")
  cls = YOLO("classificador.pt")
  return det, cls


detector, classificador = carregar_modelos()

# Localiza dinamicamente o índice de 'com_mcn'
idx_mcn = 0
for idx, nome in classificador.names.items():
  if "mcn" in nome.lower():
    idx_mcn = idx
    break

# Painel lateral de sensibilidade
st.sidebar.header("⚙️ Ajustes de Calibração")
conf_det = st.sidebar.slider(
    "Sensibilidade do Detector (Tétrades)", 0.20, 0.70, 0.35, 0.05
)
limiar_mcn = st.sidebar.slider(
    "Rigor Diagnóstico de MCN (Probabilidade mínima)", 0.50, 0.95, 0.75, 0.05
)

arquivo_img = st.file_uploader(
    "Envie a micrografia da lâmina:", type=["jpg", "jpeg", "png", "tif"]
)

if arquivo_img is not None:
  imagem_pil = Image.open(arquivo_img).convert("RGB")

  col_esq, col_dir = st.columns(2)
  with col_esq:
    st.subheader("Micrografia Original")
    st.image(imagem_pil, use_container_width=True)

  if st.button(
      "🚀 Executar Análise Citogenética",
      type="primary",
      use_container_width=True,
  ):
    with st.spinner("Processando campo e diagnosticando micronúcleos..."):
      img = np.array(imagem_pil)
      h_orig, w_orig, _ = img.shape

      # Estágio 1: Localização das tétrades
      results_det = detector.predict(
          source=img, conf=conf_det, imgsz=1280, rect=True, verbose=False
      )[0]

      contagem_normal = 0
      contagem_mcn = 0
      img_resultado = img.copy()

      if results_det.boxes is not None and len(results_det.boxes) > 0:
        for box in results_det.boxes:
          cls_id = int(box.cls.cpu().numpy()[0])
          nome_classe = detector.names[cls_id]

          if "tetrade" not in nome_classe.lower():
            continue

          x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)

          pw = int((x2 - x1) * 0.10)
          ph = int((y2 - y1) * 0.10)
          cx1 = max(0, x1 - pw)
          cy1 = max(0, y1 - ph)
          cx2 = min(w_orig, x2 + pw)
          cy2 = min(h_orig, y2 + ph)

          crop = img[cy1:cy2, cx1:cx2]
          if crop.size == 0:
            continue

          # Estágio 2: Diagnóstico no classificador
          res_cls = classificador.predict(
              source=crop, imgsz=224, verbose=False
          )[0]
          probs = res_cls.probs.data.cpu().numpy()
          prob_mcn = float(probs[idx_mcn])

          if prob_mcn >= limiar_mcn:
            contagem_mcn += 1
            cor = (255, 0, 0)
            rotulo = f"MCN ({prob_mcn * 100:.1f}%)"
          else:
            contagem_normal += 1
            cor = (0, 255, 0)
            rotulo = f"Normal ({(1.0 - prob_mcn) * 100:.1f}%)"

          cv2.rectangle(img_resultado, (x1, y1), (x2, y2), cor, 4)
          cv2.putText(
              img_resultado,
              rotulo,
              (x1, max(35, y1 - 10)),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
              cor,
              2,
          )

      total_avaliado = contagem_normal + contagem_mcn
      freq_mcn = (
          (contagem_mcn / total_avaliado * 100) if total_avaliado > 0 else 0.0
      )

      with col_dir:
        st.subheader("Varredura Concluída")
        st.image(
            img_resultado,
            caption="Verde: Normal | Vermelho: Com MCN",
            use_container_width=True,
        )

      st.markdown("---")
      st.subheader("📊 Relatório Citogenético")

      m1, m2, m3, m4 = st.columns(4)
      m1.metric("Tétrades Normais", contagem_normal)
      m2.metric("Tétrades com MCN", contagem_mcn)
      m3.metric("Total Avaliado", total_avaliado)
      m4.metric("Frequência de MCN", f"{freq_mcn:.2f}%")

      dados_csv = pd.DataFrame([{
          "Arquivo": arquivo_img.name,
          "Tetrades_Normais": contagem_normal,
          "Tetrades_Com_MCN": contagem_mcn,
          "Total_Tetrades": total_avaliado,
          "Frequencia_MCN_Percentual": round(freq_mcn, 2),
          "Sensibilidade_Detector": conf_det,
          "Limiar_Rigor_MCN": limiar_mcn,
      }])

      st.download_button(
          label="📥 Baixar Laudo da Amostra (CSV)",
          data=dados_csv.to_csv(index=False),
          file_name=f"laudo_{arquivo_img.name.split('.')[0]}.csv",
          mime="text/csv",
      )