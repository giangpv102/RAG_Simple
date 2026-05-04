
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
        try:
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
        except Exception:
            class _SimpleRAGChain:
                def __init__(self, llm, retriever, formatter, parser):
                    self.llm = llm
                    self.retriever = retriever
                    self.formatter = formatter
                    self.parser = parser

                def invoke(self, question: str):
                    # Retrieve docs using common LangChain retriever API
                    if hasattr(self.retriever, "get_relevant_documents"):
                        docs = self.retriever.get_relevant_documents(question)
                    else:
                        # Some retrievers implement __call__
                        docs = self.retriever(question)

                    prompt_context = self.formatter(docs)
                    prompt = (
                        "You are a helpful assistant.\n\n"
                        "Use the following context to answer the question.\n"
                        "If you don't know the answer, say you don't know.\n\n"
                        f"Context:\n{prompt_context}\n\n"
                        f"Question:\n{question}\n\n"
                        "Answer:"
                    )

                    output = self.llm(prompt)

                    # Normalize output: transformers pipeline returns list[dict]
                    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                        text = output[0].get("generated_text") or str(output[0])
                    else:
                        text = str(output)

                    try:
                        return self.parser.parse(text)
                    except Exception:
                        return text

            return _SimpleRAGChain(self.llm, retriever, self.format_docs, self.str_parser)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)