# Portafolio de Inteligencia Artificial & LLMOps 🚀

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-121011?style=flat)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.6_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![Cohere](https://img.shields.io/badge/Cohere-Rerank_v3-395144?style=flat)](https://cohere.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00599C?style=flat)](https://github.com/facebookresearch/faiss)

Este repositorio reúne proyectos prácticos orientados al diseño, orquestación, optimización y despliegue de soluciones basadas en **Inteligencia Artificial Generativa** y **Sistemas de Agentes**.

---

## 📂 Contenido del Repositorio

Actualmente, el portafolio cuenta con el siguiente módulo implementado:

| Proyecto | Descripción | Tecnologías |
| :--- | :--- | :--- |
| **[01-langchain-basics](./01-langchain-basics)** | Pipeline RAG Avanzado con Búsqueda Híbrida (Léxica + Vectorial), Re-Ranking multilingüe y orquestación declarativa LCEL. | LangChain, FAISS, BM25, Cohere Rerank, Gemini 3.6 Flash |

---

## 💡 Proyectos Destacados

### [01. RAG Avanzado con Búsqueda Híbrida y Re-Ranking](./01-langchain-basics)

Sistema de Recuperación Aumentada por Generación (RAG) diseñado para mitigar alucinaciones y mejorar la precisión en consultas técnicas complejas.

* **Búsqueda Híbrida:** Combina coincidencia exacta de términos (**BM25**) con búsqueda semántica vectorial (**FAISS** + `models/gemini-embedding-001`).
* **Re-Ranking de Precisión:** Filtra y reordena los candidatos recuperados usando un modelo *cross-encoder* (**Cohere Rerank v3.0**).
* **Orquestación LCEL:** Flujo modular escrito mediante LangChain Expression Language y resuelto con el LLM **Gemini 3.6 Flash**.
* **Gestión Empresarial de Logs:** Captura automática de advertencias del sistema y `stderr` redirigidos a registros locales.

👉 *Para ver la arquitectura completa y los ejemplos de salida, consulta el [`README.md` del módulo](./01-langchain-basics/README.md).*

---

## 👨‍💻 Autor

* **GitHub:** [@sistemasfernandolopez](https://github.com/sistemasfernandolopez)
