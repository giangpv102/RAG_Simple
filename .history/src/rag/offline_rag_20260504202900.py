from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class Str_OutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self, text_response: str) -> str:
        import re
        match = re.search(r"Answer:\s*(.*)", text_response, re.DOTALL)
        return match.group(1).strip() if match else text_response


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm

        # ✅ Prompt LOCAL (thay vì pull từ hub)
        self.prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")

        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        rag_chain = (
            {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | self.str_parser
        )

        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)