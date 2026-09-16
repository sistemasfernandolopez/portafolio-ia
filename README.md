# Portafolio de Inteligencia Artificial & LLMOps 🚀

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Package_Manager-DE5B8B?style=flat&logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-121011?style=flat)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-FF6F61?style=flat)](https://www.langchain.com/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.5_Flash_Lite-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![Cohere](https://img.shields.io/badge/Cohere-Rerank_v3-395144?style=flat)](https://cohere.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00599C?style=flat)](https://github.com/facebookresearch/faiss)

Este repositorio reúne proyectos prácticos orientados al diseño, orquestación, optimización y despliegue de soluciones basadas en **Inteligencia Artificial Generativa**, **Arquitecturas RAG Avanzadas** y **Sistemas Agénticos**.

---

## 📂 Contenido del Repositorio

Actualmente, el portafolio cuenta con los siguientes módulos implementados:

| Proyecto | Descripción | Tecnologías |
| :--- | :--- | :--- |
| **[01-langchain-basics](./01-langchain-basics)** | Pipeline RAG Avanzado con Búsqueda Híbrida (Léxica + Vectorial), Re-Ranking multilingüe y orquestación declarativa LCEL. | LangChain, FAISS, BM25, Cohere Rerank, Gemini |
| **[02-langgraph-workflows](./02-langgraph-workflows)** | Agente RAG Auto-Correctivo (CRAG) con Búsqueda Híbrida, Re-Ranking, evaluación estricta de contexto y Human-In-The-Loop (HITL). | LangGraph, LangChain, FAISS, BM25, Cohere Rerank, Gemini, uv |

---

## 💡 Proyectos Destacados

### [01. RAG Avanzado con Búsqueda Híbrida y Re-Ranking](./01-langchain-basics)

Sistema de Recuperación Aumentada por Generación (RAG) diseñado para mitigar alucinaciones y mejorar la precisión en consultas técnicas complejas.

* **Búsqueda Híbrida:** Combina coincidencia exacta de términos (**BM25**) con búsqueda semántica vectorial (**FAISS** + `models/gemini-embedding-001`).
* **Re-Ranking de Precisión:** Filtra y reordena los candidatos recuperados usando un modelo *cross-encoder* (**Cohere Rerank v3.0**).
* **Orquestación LCEL:** Flujo modular escrito mediante LangChain Expression Language y resuelto con el LLM **Gemini**.
* **Gestión Empresarial de Logs:** Captura automática de advertencias del sistema y `stderr` redirigidos a registros locales.

👉 *Para ver la arquitectura completa y los ejemplos de salida, consulta el [`README.md` del módulo](./01-langchain-basics/README.md).*

---

### [02. Agente RAG Auto-Correctivo con Human-In-The-Loop](./02-langgraph-workflows)

Evolución del RAG tradicional hacia un agente guiado por estados deterministas capaz de corregir de forma autónoma su flujo de búsqueda e interactuar con operadores humanos.

* **Grafos de Estado Deterministas:** Orquestación basada en **LangGraph** con evaluación dinámica de relevancia (*Grader*) para prevenir alucinaciones.
* **Reescritura Autónoma de Consultas:** Optimización automática del prompt de búsqueda (*Query Rewriter*) cuando la información recuperada inicialmente es insuficiente.
* **Intervención Humana en Tiempo de Ejecución (HITL):** Pausa del flujo mediante puntos de interrupción (`interrupt_before`) e inyección de estado manual (`update_state`) tras agotar los reintentos automáticos.
* **Recuperación Multicapa:** Integración de Búsqueda Híbrida (**BM25** + **FAISS**) combinada con **Cohere Rerank v3.0**.

👉 *Para ver el diagrama Mermaid, las trazas y capturas de ejecución, consulta el [`README.md` del módulo](./02-langgraph-workflows/README.md).*

---

## 👨‍💻 Autor

* **GitHub:** [@sistemasfernandolopez](https://github.com/sistemasfernandolopez)