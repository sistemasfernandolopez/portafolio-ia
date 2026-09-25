import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No se encontró GOOGLE_API_KEY ni GEMINI_API_KEY en el entorno .env")

gemini_llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    api_key=api_key
)

@tool("Herramienta de Búsqueda Web")
def search_tool(query: str) -> str:
    """Busca noticias o información reciente en la web sobre un tema dado."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return str(results) if results else "No se obtuvieron resultados."
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"

def run_tech_news_crew(topic: str) -> str:
    researcher = Agent(
        role="Investigador Senior de IT",
        goal="Encontrar parches, soluciones técnicas e información reciente sobre {topic}",
        backstory="Periodista e investigador técnico enfocado en validar datos y documentación oficial.",
        tools=[search_tool],
        llm=gemini_llm,
        verbose=True,
    )

    writer = Agent(
        role="Redactor de Documentación Técnica",
        goal="Sintetizar hallazgos en sentencias técnicas claras en Markdown para soporte",
        backstory="Especialista en redactar sentencias en base a conocimiento para ingenieros de soporte.",
        llm=gemini_llm,
        verbose=True,
    )

    research_task = Task(
        description="Busca en la web información reciente sobre: '{topic}'. Extrae hechos clave y soluciones técnicas.",
        expected_output="Informe en viñetas con datos clave y fuentes.",
        tools=[search_tool],
        agent=researcher,
    )

    write_task = Task(
        description="Con base en la investigación, redacta sentencias en Markdown con un resumen del error y su solución.",
        expected_output="Sentencias técnicas en formato Markdown.",
        agent=writer,
        context=[research_task],
    )

    tech_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    result = tech_crew.kickoff(inputs={"topic": topic})
    return str(result)