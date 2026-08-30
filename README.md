# Explainable Alzheimer's Disease Detection — CNN + Grad-CAM + RAG

An explainable AI system that classifies structural MRI scans for signs of
Alzheimer's disease, then grounds its explanation in cited literature using
a retrieval-augmented LLM pipeline — built to make a research-grade model's
reasoning inspectable, not just its output.

**Note:** this repository contains a project overview and demo only. The
full implementation is private, as this work extends a co-authored research
paper currently under review — happy to walk through the code and
methodology in more detail on request or in an interview.

---

## Demo

[![Watch the demo](https://img.youtube.com/vi/tHHdiRNwX8c/0.jpg)](https://youtu.be/tHHdiRNwX8c)
*(click the thumbnail to watch — unlisted video)*

---

## Architecture

![Pipeline architecture](assets/architecture-diagram.png)

## Screenshots

| Prediction output | Grad-CAM overlay |
|---|---|
| ![Prediction output](assets/prediction-output.png) | ![Grad-CAM overlay](assets/gradcam-overlay.png) |

| Cited report | Full UI |
|---|---|
| ![Cited report](assets/cited-report.png) | ![Full UI](assets/full-ui.png) |

---

## What it does

1. **CNN prediction** — classifies an MRI slice as AD (Alzheimer's Disease),
   CN (cognitively normal), or MCI (mild cognitive impairment), built on an
   EfficientNetB0 backbone with soft attention and genetic-algorithm-optimized
   architecture search, trained on ADNI structural MRI data.
2. **Grad-CAM explainability** — generates a visual attention heatmap
   showing which regions of the scan most influenced the model's
   prediction, using a custom two-stage split workaround to run Grad-CAM
   on a nested EfficientNet submodel in Keras.
3. **Retrieval** — the prediction and attended region are used to search a
   local corpus (the NIA-AA diagnostic criteria, this project's own paper,
   and six baseline architecture papers — VGG16, ResNet50, EfficientNet-B4,
   ViT, Vision Mamba, MedicalNet) for the most relevant supporting passages.
4. **LLM synthesis** — an LLM drafts a plain-English report, required to
   cite a specific retrieved passage for every clinical claim it makes.
5. **Guardrail verification** — every claim is programmatically checked for
   a valid citation before the report is shown to the user; uncited or
   unsupported claims are rejected rather than displayed.

## Results

- **95.14% classification accuracy** (AD / CN / MCI, 3-class)
- EfficientNetB0 backbone with soft attention and GA-optimized architecture
- Trained and evaluated on ADNI structural MRI data

## Tech stack

- **Modeling:** TensorFlow/Keras, EfficientNetB0, Grad-CAM
- **Orchestration:** LangGraph (StateGraph pipeline with retry logic),
  LangChain
- **LLM:** Groq (Llama 3.3)
- **Retrieval:** local vector store over an 8-document corpus, cosine
  similarity search
- **UI:** Gradio

## Research context

This project extends the CNN and methodology from a co-authored Alzheimer's
disease detection paper into a full explainable, LLM-integrated system. The
underlying paper is currently under review; this repo will be updated with
a link once it's published.
