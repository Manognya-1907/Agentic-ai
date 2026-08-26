import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. INNOVATECORP KT GUIDE
# ============================================================

KT_GUIDE = """
Welcome to InnovateCorp!

This Knowledge Transfer (KT) guide helps new employees
understand the company, Project Alpha team, onboarding
process, tools, resources, culture and expectations.

COMPANY VALUES:
- Innovation
- Collaboration
- Customer Focus

TEAM STRUCTURE:

The employee will join the Project Alpha team.

The employee reports to:
Sarah Chen - Senior Project Manager

Direct teammates:
- David Lee - Lead Developer
- Maria Rodriguez - UI/UX Designer
- Tom Jackson - QA Engineer

MEETINGS:

Team meetings:
Every Monday at 10 AM in Conference Room 3.

Daily stand-ups:
9:30 AM via Google Meet.

KEY TOOLS AND SOFTWARE:

Jira:
Used for task tracking.

Confluence:
Used for documentation.

Slack:
Used for instant messaging.

Google Workspace:
Used for email and calendars.

Python and JavaScript:
Used primarily for development.

GitHub:
Used for hosting code.

Access to these tools will be granted within
the employee's first three days.

ONBOARDING PROCESS:

First week:
Setup and introductions.

Day one:
Employees receive their laptop and login credentials.

Tuesday:
HR conducts an orientation session covering:
- Company policies
- Benefits
- Payroll

During the first week:
Employees have one-on-one meetings with team members.

By the end of the second week:
Employees should:
- Have access to all necessary systems.
- Complete mandatory compliance training modules.

IMPORTANT RESOURCES:

Internal knowledge base:
internal.innovatecorp.com/kb

The knowledge base contains:
- FAQs
- Best practices
- Troubleshooting guides

IT support:
support.innovatecorp.com

Employees can:
- Submit an IT support ticket.
- Call extension 5555.

Health and wellness benefits:
Available on the HR portal.

CULTURE AND EXPECTATIONS:

InnovateCorp encourages:
- A proactive environment.
- A collaborative environment.
- Open communication.
- Continuous learning.

Employees are encouraged to ask questions.
The team supports employee growth.

PERFORMANCE REVIEWS:

Performance reviews are conducted quarterly.

PROFESSIONAL DEVELOPMENT:

Professional development courses are available
through the InnovateLearn platform.
"""


# ============================================================
# 2. LOCAL KT RELEVANCE GUIDE
# ============================================================
#
# This is NOT an LLM call.
#
# It runs before Gemini.
#
# It contains concepts that belong to the KT Guide.
# ============================================================

KT_TOPICS = {

    "company": [
        "innovatecorp",
        "company",
        "employee",
        "employees",
        "new employee",
        "new employees",
        "company values",
        "values"
    ],

    "team": [
        "project alpha",
        "team",
        "team structure",
        "sarah chen",
        "david lee",
        "maria rodriguez",
        "tom jackson",
        "manager",
        "manager name",
        "report to",
        "reporting",
        "lead developer",
        "ui ux",
        "ui/ux",
        "designer",
        "qa",
        "quality assurance"
    ],

    "meetings": [
        "meeting",
        "meetings",
        "team meeting",
        "daily meeting",
        "standup",
        "stand-up",
        "stand up",
        "daily standup",
        "daily stand-up",
        "google meet",
        "conference room"
    ],

    "tools": [
        "jira",
        "confluence",
        "slack",
        "google workspace",
        "github",
        "python",
        "javascript",
        "task tracking",
        "task management",
        "documentation",
        "messaging",
        "email",
        "calendar",
        "code hosting",
        "development tools",
        "software",
        "tools"
    ],

    "onboarding": [
        "onboarding",
        "orientation",
        "first week",
        "first day",
        "day one",
        "day 1",
        "second week",
        "day two",
        "laptop",
        "login",
        "credentials",
        "hr",
        "human resources",
        "payroll",
        "benefits",
        "compliance",
        "training",
        "systems",
        "access"
    ],

    "resources": [
        "knowledge base",
        "internal knowledge base",
        "faq",
        "faqs",
        "best practices",
        "troubleshooting",
        "it support",
        "support",
        "support ticket",
        "ticket",
        "extension 5555",
        "hr portal",
        "health benefits",
        "wellness",
        "resources"
    ],

    "culture": [
        "culture",
        "company culture",
        "expectations",
        "innovation",
        "collaboration",
        "customer focus",
        "communication",
        "continuous learning",
        "growth",
        "proactive"
    ],

    "career": [
        "performance review",
        "performance reviews",
        "professional development",
        "development courses",
        "innovatelearn",
        "quarterly review",
        "career development"
    ]
}


# ============================================================
# 3. OBVIOUSLY IRRELEVANT TOPICS
# ============================================================
#
# These help reject clearly unrelated questions quickly.
# ============================================================

IRRELEVANT_TOPICS = [
    "cricket",
    "football",
    "soccer",
    "world cup",
    "movie",
    "movies",
    "actor",
    "actress",
    "songs",
    "music",
    "celebrity",

    "recipe",
    "cooking",
    "biryani",

    "weather",
    "temperature",
    "rain",
    "forecast",

    "politics",
    "politician",
    "election",

    "stock market",
    "stocks",
    "bitcoin",
    "crypto",
    "cryptocurrency",

    "shopping",
    "fashion",

    "relationship",
    "dating",

    "travel",
    "tourism",
    "vacation",

    "joke",
    "gaming",
    "games"
]


# ============================================================
# 4. LOCAL RELEVANCE CHECKER
# ============================================================
#
# IMPORTANT:
# NO GEMINI CALL HAPPENS HERE.
# ============================================================

def check_kt_relevance(question: str) -> bool:

    question = question.lower().strip()

    if not question:
        return False

    # --------------------------------------------------------
    # Check clearly irrelevant questions first
    # --------------------------------------------------------

    for topic in IRRELEVANT_TOPICS:

        if topic in question:
            return False

    # --------------------------------------------------------
    # Count KT-related concepts
    # --------------------------------------------------------

    matched_topics = 0

    for category, keywords in KT_TOPICS.items():

        for keyword in keywords:

            if keyword in question:
                matched_topics += 1
                break

    # --------------------------------------------------------
    # At least one strong KT concept = relevant
    # --------------------------------------------------------

    if matched_topics >= 1:
        return True

    return False


# ============================================================
# 5. KT GUIDE TOOL
# ============================================================

@tool
def retrieve_kt_context(query: str) -> str:
    """
    Retrieve information from the InnovateCorp KT Guide.
    """

    return KT_GUIDE


# ============================================================
# 6. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ============================================================
# 7. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 8. CREATE KT AGENT
# ============================================================

kt_agent = create_agent(

    model=llm,

    tools=[
        retrieve_kt_context
    ],

    system_prompt="""
You are the InnovateCorp Knowledge Transfer assistant.

Your ONLY source of information is the InnovateCorp KT Guide.

IMPORTANT RULES:

1. Use the KT Guide to answer the user's question.

2. Do not use outside knowledge.

3. Do not guess.

4. Do not invent information.

5. If the requested information is present in the
   KT Guide, answer clearly.

6. If the question is related to the KT Guide but
   the exact answer is not available, say:

   "This information is not available in the
   InnovateCorp KT Guide."

7. Keep answers simple and easy to understand.

8. The KT Guide is the only source of truth.
"""
)


# ============================================================
# 9. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Question for the InnovateCorp KT Guide"
    )


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def run_kt_agent(x):

    # --------------------------------------------------------
    # Get user question
    # --------------------------------------------------------

    if isinstance(x, dict):
        question = x.get("input", "")
    else:
        question = x.input

    # --------------------------------------------------------
    # LOCAL RELEVANCE CHECK
    # --------------------------------------------------------
    #
    # NO GEMINI CALL HERE.
    # --------------------------------------------------------

    relevant = check_kt_relevance(question)

    # --------------------------------------------------------
    # IRRELEVANT QUERY
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The function returns here.
    #
    # Therefore:
    # - Gemini is NOT called
    # - Agent is NOT called
    # - KT tool is NOT called
    # --------------------------------------------------------

    if not relevant:

        return (
            "This question is irrelevant to the "
            "InnovateCorp KT Guide. "
            "Please ask a question about InnovateCorp, "
            "Project Alpha, onboarding, team members, "
            "meetings, tools, company resources, "
            "company culture, or employee development."
        )

    # --------------------------------------------------------
    # RELEVANT QUERY
    # --------------------------------------------------------
    #
    # ONLY NOW DO WE CALL THE AGENT.
    # --------------------------------------------------------

    result = kt_agent.invoke({

        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]

    })

    # --------------------------------------------------------
    # Extract final message
    # --------------------------------------------------------

    messages = result.get("messages", [])

    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            str(last_message)
        )

        # Gemini can sometimes return a list
        if isinstance(content, list):

            text = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        text.append(
                            item.get("text", "")
                        )

                else:
                    text.append(str(item))

            return "".join(text)

        return str(content)

    return "Unable to generate an answer."


# ============================================================
# 11. CREATE LANGCHAIN RUNNABLE
# ============================================================

formatted_agent_chain = (

    RunnableLambda(run_kt_agent)

).with_types(

    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 12. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="InnovateCorp KT Guide Agent",

    description=(
        "KT Guide assistant that calls the LLM "
        "only for relevant questions."
    )
)


# ============================================================
# 13. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 14. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port
    )
