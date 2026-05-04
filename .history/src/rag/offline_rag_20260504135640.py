import re
from typing import Callable

# Keep compatibility with langchain_core parser if available; otherwise provide
# a tiny local parser with the same interface.
try:
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    class StrOutputParser:
        def parse(self, text: str) -> str:  # pragma: no cover - fallback
            return text


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(
        self,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:

        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Offline_RAG:
    """A lightweight RAG chain compatible with this repo's call sites.

    This avoids depending on langchain.hub and returns an object with
    .invoke(question: str) -> str which the existing `app.py` expects.
    """

    def __init__(self, llm: Callable[[str], str]) -> None:
        self.llm = llm
        # Keep a local parser to extract answer text from model outputs
        self.str_parser = Str_OutputParser()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_chain(self, retriever):
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
                prompt = f"Context:\n{prompt_context}\n\nQuestion: {question}\n\nAnswer:"

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
import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(
        self,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:

        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response
        
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough(),
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )

        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    