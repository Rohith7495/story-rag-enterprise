import os
from enterprise_rag import EnterpriseRAG
import inspect

rag = EnterpriseRAG()
sig = inspect.signature(rag.answer_question)
print(f"DEBUG: answer_question signature: {sig}")

try:
    rag.answer_question("test", chat_history=[])
    print("DEBUG: Success calling with chat_history")
except TypeError as e:
    print(f"DEBUG: Failed calling with chat_history: {e}")
