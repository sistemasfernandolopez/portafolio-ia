# 03. Observabilidad y Monitoreo de Agentes RAG (Langfuse)

Este proyecto integra **Langfuse Cloud** como capa de observabilidad y LLMOps sobre un **Agente RAG Auto-Correctivo (CRAG)** desarrollado en **LangGraph**. El sistema permite auditar en tiempo real la ejecución de grafos cíclicos complejos, medir la latencia paso a paso, controlar los costes de API, inspeccionar prompts/outputs por nodo y rastrear sesiones de interacción con intervención humana (**Human-In-The-Loop**).

---

## 🛠️ Stack Tecnológico

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package_Manager-DE5B8B?style=flat&logo=python&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-000000?style=flat&logo=langfuse&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-FF6F61?style=flat)
![LangChain](https://img.shields.io/badge/LangChain-Framework-121011?style=flat)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4?style=flat&logo=google&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-Re--rank_v3.0-D14671?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=flat)

---

## 📐 Arquitectura de Trazabilidad

El flujo agéntico propaga automáticamente los callbacks de seguimiento a través del contexto de ejecución (`RunnableConfig`), enviando eventos asíncronos al backend de Langfuse Cloud:

```mermaid
graph TD
    subgraph LangGraph Local Execution
        START([Inicio]) --> Retrieve[1. Retrieve: Búsqueda Híbrida]
        Retrieve --> Grade[2. Grade: Evaluación de Relevancia]
        Grade --> Decision{¿Contexto Válido?}
        
        Decision -- Sí --> Generate[5. Generate: Respuesta Final]
        Decision -- No & Reintentos < 1 --> Rewrite[3. Rewrite: Reescritura]
        Decision -- No & Reintentos >= 1 --> HITL[4. Human Intervention: Pausa]
        
        Rewrite --> Retrieve
        HITL -- Resumen de Sesión --> Retrieve
        Generate --> END([Fin])
    end

    subgraph LLMOps Backend
        CallbackHandler[Langfuse Callback Handler]
        Cloud[Langfuse Cloud Dashboard]
    end

    Retrieve -.-> CallbackHandler
    Grade -.-> CallbackHandler
    Rewrite -.-> CallbackHandler
    Generate -.-> CallbackHandler
    CallbackHandler ==>|HTTP Async / Flush| Cloud
```

---

## 💡 Aspectos Clave de Arquitectura

1. **Propagación Jerárquica de Callbacks (`RunnableConfig`):** Inyección nativa del `CallbackHandler` de Langfuse en cada nodo del grafo para mapear las relaciones padre-hijo entre cadenas, retrievers y llamadas a LLMs.
2. **Métricas de Rendimiento y Costes:** Cálculo automático de consumo de tokens (Input/Output), desglose de latencia individual por paso y costes estimados de API (Google Gemini y Cohere).
3. **Indexación por Sesión y Etiquetado (`Metadata`):** Agrupación de la trazabilidad mediante `session_id`, `user_id` y etiquetas personalizadas (`Corrective-RAG`, `HITL`) para realizar seguimiento multiturno tras la reanudación del grafo por un operador humano.
4. **Auditoría de Prompts y Detección de Alucinaciones:** Inspección detallada de las entradas, textos de sistema y salidas generadas en cada paso para depurar fallos en las decisiones condicionales.

---

## 🚀 Instalación y Ejecución

### 1. Requisitos Previos

Este proyecto utiliza **[uv](https://github.com/astral-sh/uv)** para la gestión rápida del entorno virtual y dependencias:

```bash
# Crear el entorno virtual con uv
uv venv --python 3.12

# Activar el entorno virtual
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows

# Instalar dependencias exactas
uv pip install -r requirements.txt
```

*(Alternativa rápida sin activación manual: `uv run python workflow.py`)*

### 2. Variables de Entorno

Configura tu archivo `.env` en la raíz de la carpeta con las credenciales necesarias:

```env
GOOGLE_API_KEY=tu_api_key_de_gemini
COHERE_API_KEY=tu_api_key_de_cohere

# Credenciales de Langfuse Cloud
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
```

### 3. Ejecución del Script

```bash
python3 workflow.py
```

---

## 📊 Monitoreo y Observabilidad en Langfuse

### 1. Grafo de Estado y Árbol de Ejecución (Trace Tree)
Visualización interactive del flujo del agente, mostrando el ciclo de reintentos, el estado del grafo y las métricas acumuladas de latencia y tokens.
![Árbol de Traza Langfuse](assets/langfuse-trace-tree.png)

### 2. Auditoría Detallada por Nodo (Prompts & IO)
Inspección profunda de las variables de entrada, prompts del sistema y respuestas generadas en el nodo evaluador y generador.
![Detalle de Nodo Langfuse](assets/langfuse-trace-node.png)

### 3. Historial de Sesiones y Métricas de Usuario
Rastreo agrupado por `session_id` que refleja la interacción continua del usuario y las pausas por intervención humana (HITL).
![Sesiones en Langfuse](assets/langfuse-trace-sessions.png)