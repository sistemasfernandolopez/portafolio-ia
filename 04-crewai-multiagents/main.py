import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from ddgs import DDGS

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY no está definida en el archivo .env")

gemini_llm = LLM(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY")
)

@tool("Herramienta de Búsqueda Web")
def search_tool(query: str) -> str:
    """Busca noticias o información reciente en la web sobre un tema dado."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
        return str(results) if results else "No se obtuvieron resultados para esta consulta."
    except Exception as e:
        return f"Error al ejecutar la búsqueda: {str(e)}"

def run_tech_news_crew(topic: str):
    researcher = Agent(
        role="Investigador Senior de Tecnología",
        goal="Encontrar las noticias de última hora más relevantes y recientes el 18 de septiembre de 2026 sobre {topic}",
        backstory=(
            "Eres un periodista de investigación especializado en tecnología. "
            "Tu fortaleza radica en filtrar el ruido de internet, encontrar noticias "
            "recientes, verificar datos clave y extraer contexto técnico preciso."
        ),
        tools=[search_tool],
        llm=gemini_llm,
        verbose=True,
    )

    writer = Agent(
        role="Redactor Periodístico de Tecnología",
        goal="Transformar datos de investigación en un artículo periodístico atractivo y riguroso sobre {topic}",
        backstory=(
            "Eres un editor senior en un medio tecnológico de prestigio. "
            "Destacas por sintetizar información compleja en artículos claros y estructurados, "
            "manteniendo estricta fidelidad a los hechos investigados."
        ),
        llm=gemini_llm,
        verbose=True,
    )

    research_task = Task(
        description=(
            "1. Busca en la web las noticias más recientes el 18 de septiembre de 2026 sobre: '{topic}'.\n"
            "2. Selecciona el acontecimiento o avance más importante de última hora.\n"
            "3. Recopila los hechos clave: qué ocurrió, empresas/personas involucradas "
            "y datos técnicos relevantes."
        ),
        expected_output="Informe estructurado en viñetas con los hechos clave y fuentes encontradas.",
        tools=[search_tool],
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Con base EXCLUSIVA en el informe del investigador, redacta un artículo en Markdown.\n"
            "Estructura obligatoria:\n"
            "- Titular e impacto.\n"
            "- Introducción (qué, quién, cuándo).\n"
            "- Análisis central del avance/noticia.\n"
            "- Conclusión a corto plazo.\n"
            "No inventes información que no esté en el informe de investigación."
        ),
        expected_output="Un artículo periodístico completo en formato Markdown.",
        agent=writer,
        context=[research_task],
        output_file="articulo_tecnologia.md",
    )

    tech_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    return tech_crew.kickoff(inputs={"topic": topic})

if __name__ == "__main__":
    tema = "Modelos de Lenguaje de Código Abierto"
    resultado = run_tech_news_crew(topic=tema)
    
    print("\n================ RESULTADO FINAL ================\n")
    print(resultado)