import gradio as gr
from app.core.graph_workflow import run_copilot_workflow, resume_hitl_workflow
from app.core.crew_agents import run_tech_news_crew
from app.core.rag_engine import add_documents_to_knowledge_base

def build_ui():
    with gr.Blocks(title="Enterprise IT Copilot OpsCenter", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Enterprise IT Copilot & Agentic Knowledge Operations Center")
        
        # =========================================================================
        # PESTAÑA 1: Copiloto de Soporte (con Intervención Humana / HITL Integrado)
        # =========================================================================
        with gr.Tab("💬 Copiloto de Soporte"):
            gr.Markdown("### Asistente Inteligente con RAG Híbrido, Re-ranking e Interacción en Vivo")
            
            with gr.Row():
                # Columna Izquierda: Entradas de usuario y reformulación
                with gr.Column(scale=2):
                    user_q = gr.Textbox(
                        label="Consulta Técnica", 
                        placeholder="Ej: ¿Cómo solucionar el error ERR-902?", 
                        lines=2
                    )
                    thread_input = gr.Textbox(
                        label="ID de Sesión / Thread", 
                        value="sesion_demo_01"
                    )
                    send_btn = gr.Button("Enviar Consulta", variant="primary")
                    
                    # 🔴 Bloque dinámico de Reformulación (Oculto por defecto)
                    with gr.Column(visible=False) as hitl_box:
                        gr.Markdown("---")
                        gr.Markdown("⚠️ **No se encontró información suficiente en la base de datos.**")
                        gr.Markdown("Por favor, reformula la pregunta o aporta más contexto para reanudar el flujo de LangGraph:")
                        
                        reformulated_q = gr.Textbox(
                            label="Nueva Consulta Reformulada / Contexto Adicional", 
                            placeholder="Ej: Solución al fallo de timeout de conexión en PostgreSQL", 
                            lines=2
                        )
                        resume_btn = gr.Button("🔄 Reanudar Flujo con Nueva Pregunta", variant="stop")

                # Columna Derecha: Salidas del sistema
                with gr.Column(scale=3):
                    status_out = gr.Textbox(label="Estado del Flujo")
                    answer_out = gr.Textbox(label="Respuesta Generada", lines=5)
                    docs_out = gr.JSON(label="Fragmentos Recuperados (FAISS + BM25)")

            # Lógica al enviar la consulta inicial
            def handle_chat(q, tid):
                res = run_copilot_workflow(q, tid)
                
                # Si el grafo se detuvo en el nodo human_intervention
                if res["status"] == "PAUSED_HITL":
                    return (
                        res["status"],
                        "⚠️ No se encontró contexto suficiente. Por favor, reformula la pregunta en el campo de la izquierda y haz clic en 'Reanudar Flujo'.",
                        [],
                        gr.update(visible=True)  # Muestra el bloque de reformulación
                    )
                
                # Si se completó correctamente
                return (
                    res["status"],
                    res["generation"],
                    res.get("documents", []),
                    gr.update(visible=False) # Mantiene oculto el bloque de reformulación
                )

            # Lógica al reanudar la consulta tras la reformulación
            def handle_resume(tid, new_q):
                if not new_q.strip():
                    return (
                        "PAUSED_HITL",
                        "⚠️ Debes ingresar una nueva consulta o instrucción para reanudar el flujo.",
                        [],
                        gr.update(visible=True)
                    )
                
                # Reanuda la ejecución en LangGraph con la nueva pregunta
                res = resume_hitl_workflow(tid, new_q)
                
                return (
                    res["status"],
                    res["generation"],
                    res.get("documents", []),
                    gr.update(visible=False)  # Oculta el bloque HITL tras completar la respuesta
                )

            # Eventos de botones
            send_btn.click(
                handle_chat, 
                inputs=[user_q, thread_input], 
                outputs=[status_out, answer_out, docs_out, hitl_box]
            )
            
            resume_btn.click(
                handle_resume, 
                inputs=[thread_input, reformulated_q], 
                outputs=[status_out, answer_out, docs_out, hitl_box]
            )

        # =========================================================================
        # PESTAÑA 2: Investigación CrewAI & Ingesta de Conocimiento
        # =========================================================================
        with gr.Tab("🕵️ Investigación CrewAI & Ingesta"):
            gr.Markdown("### Escuadrón Multi-Agente de Búsqueda Web y Auto-Indexación")
            topic_in = gr.Textbox(
                label="Tema o Incidencia a Investigar en Internet", 
                placeholder="Ej: Parches de seguridad Valkey cache 2026"
            )
            research_btn = gr.Button("Lanzar Investigación con CrewAI", variant="primary")
            
            report_out = gr.Markdown(label="Reporte de Investigación")
            ingest_btn = gr.Button("📥 Indexar Reporte en Base de Conocimientos (FAISS)", variant="secondary")
            ingest_status = gr.Textbox(label="Estado de Ingesta")

            def handle_research(topic):
                return run_tech_news_crew(topic)

            def handle_ingest(report_text):
                if not report_text.strip():
                    return "Sin contenido para indexar."
                count = add_documents_to_knowledge_base([report_text])
                return f"✅ Indexados {count} nuevos fragmentos en FAISS y BM25 exitosamente."

            research_btn.click(handle_research, inputs=[topic_in], outputs=[report_out])
            ingest_btn.click(handle_ingest, inputs=[report_out], outputs=[ingest_status])

    return demo