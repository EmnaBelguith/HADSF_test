# HADSF: Aspect Aware Semantic Control for Explainable Recommendation

The official repo for our paper **HADSF: Aspect Aware Semantic Control for Explainable Recommendation**.

## Initial Setup

Clone this repo:

```shell
git clone https://github.com/niez233/HADSF.git
```

## HADSF Framework

The code implements our **two-stage semantic framework** for LLM-enhanced explainable recommendation:

- **Stage I — Controlled Semantic Aspect Extraction:**  
  Builds a compact and domain-specific aspect vocabulary (A*) via multi-sampling consensus and embedding-based clustering.

- **Stage II — Dynamic Aspect-Aware Review Processing:**  
  Uses personalized user–item history H_ui(tau) to guide fine-grained aspect–opinion extraction and continuously update user preferences.

It also includes two **hallucination quantification metrics**:
- **ADR (Aspect Drift Rate)** – measures deviation from the aspect vocabulary A*.  
- **OFR (Opinion Fidelity Rate)** – measures semantic consistency between extracted opinions and original reviews.

## Data

A processed example dataset is available for reference:  
👉 Google Drive: [https://drive.google.com/file/d/1fK1WOzvBEmgnTt9dWWb2KtYusG9VBgeG/view?usp=sharing](https://drive.google.com/drive/folders/1tt2K8fEDlk85ToFYq8MEERN9ldyE5Oht?usp=drive_link)

---

*(More training and evaluation scripts will be released soon.)*
