"""The lab: ask the four indexed documents five questions (run from backend/):  uv run python try_questions.py"""
from rag.retrieval import ask_llm
QUESTIONS = [
    "What was the total revenue in FY2024?",
    "What payment terms do servo motor suppliers get?",
    "What is the customs duty on programmable controllers imported from Japan?",
    "Over how many years is filling line equipment depreciated?",
    "Who won the football World Cup in 2022?",
]

for number, question in enumerate(QUESTIONS, 1):
    print(f"\n{number}. {question}")
    print("   " + ask_llm(question).replace("\n", "\n   "))
