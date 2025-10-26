## HackPSU — Team Powerful

Gallery / Assets
----------------

Below are the screenshots and visual assets used in our demo. The files are included in the repository under `Lumina/assets/`.

![Initial step](Lumina/assets/initial_step.png)

![Pydantic Model Diagram](Lumina/assets/pydantic_model.png)

![Results Dashboard](Lumina/assets/results.png)

![RAG Example](Lumina/assets/RAG_example.png)


Dates: Oct 25 — Oct 26, 2025

Team: Team Powerful

## Elevator pitch

For researchers and students drowning in academic papers, the literature review process is a manual, months-long bottleneck. Our project, Lumina, is an AI research assistant that transforms this ordeal. Simply upload your PDFs and ask a question in plain English. Lumina instantly extracts and structures the key data, then allows you to have an interactive conversation with your entire library. We turn a mountain of static documents into a live, queryable knowledge base, saving hundreds of hours and accelerating discovery.

### What we built
**Problem**: Academic literature review is painfully slow, manually intensive, and prone to high error rates, hindering research progress.

**Solution**: An AI-powered, multi-agent platform that automates literature review by structuring data from PDFs and enabling interactive, conversational querying of the entire document set.

### Key Features
**Agentic AI Framework**: A custom multi-agent system orchestrates the workflow, from understanding user intent to generating validated, structured data.

**Dynamic Schema Generation**: Users define extraction goals in natural language; the AI generates the required Pydantic validation schema on the fly. No templates needed.

**Hybrid Model Architecture**: Combines fast, local open-source models (Qwen) for parsing with powerful cloud models (Gemini 1.5 Flash) for advanced reasoning and synthesis.

**Interactive RAG Querying**: A Retrieval-Augmented Generation pipeline allows users to ask complex, conversational questions about their newly structured knowledge base.

## Demo (90-second script)
- Upload & Ingest: Drag-and-drop 10+ research PDFs.

- Define Goal: In the text box, type: "Extract the methodology, sample size, and key findings."

- Extract & Structure: Click "Extract Data." Show the structured data table generated in seconds from all PDFs.

- Query & Synthesize: Switch to the "Query" tab. Ask: "Which papers used a sample size over 500?"

- The Win: Display the AI's direct, synthesized answer and the supporting data. "We just turned a week of work into 90 seconds."

## Install dependencies:

pip install -r requirements.txt
Set up environment: Create a .env file with your API keys.

Start the demo:

python app.py
Open the UI: Navigate to http://localhost:7860 in a browser.


Acknowledgements
----------------

Thanks to HackPSU organizers, mentors, and fellow participants. We used open-source tools and datasets; where applicable we've listed licenses in the repo.

Contact
-------

For questions or follow-ups, contact: Team Powerful — Penn State Graduate Students

Email: [your-email@example.com]  ← replace with a real contact

Assets & Submission notes
------------------------


Other generated visuals (from PDF extraction) can be found under `Lumina/output/`.

If you add or reorganize assets, update these paths accordingly. If you'd like, I can also generate a small `assets/README.md` gallery page or resize thumbnails for the README.

How judges should run the demo (concise checklist)
-------------------------------------------------

1. Clone the repo
2. Install requirements
3. Run the start command and follow the 90-second demo script above

----

Notes for customization
-----------------------

This README is structured to be presentation-ready and persuasive. Replace the placeholders (commands, metrics, dataset, contact) with concrete project details so the README tells a complete, judge-friendly story.

Good luck at HackPSU — Team Powerful!
