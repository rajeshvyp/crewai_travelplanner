
# Step 1: Import the required libraries
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Step 2: Load environment variables from .env
load_dotenv()

# Step 3: Get the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Please check your .env file."
    )

# Step 4: Ask the user for a city
city = input("Which city would you like to visit? ").strip()

if not city:
    raise ValueError("Please enter a city name.")

# Step 5: Configure the language model
llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=api_key
)

# Step 6: Create the Travel Planner Agent
planner = Agent(
    role="Travel Planner",
    goal=(
        "Suggest exactly three interesting places "
        "to visit in the city selected by the user."
    ),
    backstory=(
        "You are a friendly and knowledgeable travel planner. "
        "You help first-time visitors discover interesting places "
        "and explain recommendations in simple English."
    ),
    llm=llm,
    verbose=True
)

# Step 7: Create the Task
travel_task = Task(
    description=(
        "Suggest exactly 3 places to visit in {city}. "
        "For each place, provide its name and one short sentence "
        "explaining why it is worth visiting. "
        "Keep all recommendations relevant to {city}."
    ),
    expected_output=(
        "A numbered list containing exactly 3 places. "
        "Each item must include the place name and one short "
        "explanation. Use clear, simple English."
    ),
    agent=planner
)

# Step 8: Create the Crew
travel_crew = Crew(
    agents=[planner],
    tasks=[travel_task],
    verbose=True
)

# Step 9: Execute the Crew
result = travel_crew.kickoff(
    inputs={"city": city}
)

# Step 10: Display the final result
print("\n" + "=" * 40)
print("TRAVEL PLANNER RESULT")
print("=" * 40)
print(result)
