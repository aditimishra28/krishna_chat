import os
import re
import fitz  # PyMuPDF
import logging
from typing import Dict

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KrishnaSLMChatbot")

# Basic validation: chapters/verses should match actual Gita structure (18 ch., < 80 verses each)
MAX_CHAPTER = 18
MAX_VERSES = [47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 42, 20, 24, 28, 78]

class KrishnaSLMRAGModel:
    """
    KrishnaChat: Wise, compassionate AI assistant, deeply grounded in the Gita.
    Focuses on authentic, practical, and uplifting guidance.
    """
    
    def __init__(
        self,
        pdf_folder="krishna_pdfs",
        model_id="Qwen/Qwen1.5-1.8B-Chat",  # Fast, strong, CPU-friendly
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size=128,      # Larger chunks, more content per vector
        chunk_overlap=32,
        device="cpu",
        max_new_tokens=128,  # Concise, relevant answers
        trust_remote_code=True
    ):
        logger.info("KrishnaChat: Building a wise, compassionate assistant...")
        self.documents = []
        self._load_documents_from_pdfs(pdf_folder)
        self._load_embedding_model(embedding_model_name)
        self._load_llm(model_id, device, max_new_tokens, trust_remote_code)
        self._create_text_splitter(chunk_size, chunk_overlap)
        self._create_index_and_query_engine()
        logger.info("KrishnaChat is ready to empower your journey!")

    def _load_documents_from_pdfs(self, folder_path: str):
        logger.info(f"Digesting the Gita from: {folder_path}...")
        if not os.path.isdir(folder_path):
            raise ValueError(f"PDF folder '{folder_path}' does not exist.")

        chapter_pattern = re.compile(r'chapter\s+(\d+)', re.IGNORECASE)
        verse_pattern = re.compile(r'(\d+)\.(\d+)')

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".pdf"):
                continue

            path = os.path.join(folder_path, filename)
            text = self._extract_text_from_pdf(path)
            cleaned_text = self._clean_content(text)

            if not cleaned_text:
                continue

            lines = cleaned_text.split('\n')
            current_chapter = None
            current_verse = None
            sanskrit_lines = []
            translation_lines = []
            purport_lines = []
            reading_sanskrit = False
            reading_translation = False
            reading_purport = False

            def save_verse():
                if current_chapter and current_verse:
                    try:
                        chapter_int = int(current_chapter)
                        verse_int = int(current_verse)
                        if 1 <= chapter_int <= MAX_CHAPTER and 1 <= verse_int <= MAX_VERSES[chapter_int - 1]:
                            if sanskrit_lines:
                                self.documents.append(Document(
                                    text="\n".join(sanskrit_lines).strip(),
                                    metadata={"c": chapter_int, "v": verse_int, "t": "skrt", "s": os.path.splitext(filename)[0][:4]}
                                ))
                            if translation_lines:
                                self.documents.append(Document(
                                    text="\n".join(translation_lines).strip(),
                                    metadata={"c": chapter_int, "v": verse_int, "t": "trs", "s": os.path.splitext(filename)[0][:4]}
                                ))
                            if purport_lines:
                                self.documents.append(Document(
                                    text="\n".join(purport_lines).strip(),
                                    metadata={"c": chapter_int, "v": verse_int, "t": "pur", "s": os.path.splitext(filename)[0][:4]}
                                ))
                    except ValueError:
                        logger.warning(f"Invalid verse: chapter={current_chapter}, verse={current_verse}")
                sanskrit_lines.clear()
                translation_lines.clear()
                purport_lines.clear()

            for line in lines:
                line = line.strip()
                chap_match = chapter_pattern.search(line)
                verse_match = verse_pattern.search(line)

                if chap_match:
                    save_verse()
                    current_chapter = chap_match.group(1)
                    current_verse = None
                    reading_sanskrit = False
                    reading_translation = False
                    reading_purport = False
                    continue

                if verse_match:
                    save_verse()
                    current_chapter = verse_match.group(1)
                    current_verse = verse_match.group(2)
                    reading_sanskrit = True
                    reading_translation = False
                    reading_purport = False
                    continue

                if re.match(r'^(PURPORT|PURPORT:)', line, re.IGNORECASE):
                    reading_sanskrit = False
                    reading_translation = False
                    reading_purport = True
                    continue

                if reading_sanskrit and re.match(r'^[A-Za-z]', line):
                    reading_sanskrit = False
                    reading_translation = True

                if reading_sanskrit:
                    sanskrit_lines.append(line)
                elif reading_translation:
                    translation_lines.append(line)
                elif reading_purport:
                    purport_lines.append(line)

            save_verse()
        logger.info(f"Loaded {len(self.documents)} sacred passages for reflection.")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            return ""

    def _clean_content(self, text: str) -> str:
        patterns = [
            r"copyright", r"library of congress", r"all rights reserved",
            r"no part of this publication may be reproduced", r"publisher.?s note",
            r"printed in the united states", r"purchase only authorized electronic editions",
            r"digitized by the internet archive", r"^isbn", r"^\d{4,}", r"^\s*$"
        ]
        regex = re.compile("|".join(patterns), re.IGNORECASE)
        lines = [line for line in text.split('\n') if not regex.search(line) and line.strip()]
        return "\n".join(lines).strip()

    def _load_embedding_model(self, embedding_model_name: str):
        logger.info(f"Building deep understanding with: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    def _load_llm(self, model_id: str, device: str, max_new_tokens: int, trust_remote_code: bool = True):
        logger.info(f"Invoking wisdom from: {model_id}")
        self.llm = HuggingFaceLLM(
            model_name=model_id,
            tokenizer_name=model_id,
            device_map=device,
            max_new_tokens=max_new_tokens,
            tokenizer_kwargs={"trust_remote_code": trust_remote_code},
            model_kwargs={"trust_remote_code": trust_remote_code},
        )

    def _create_text_splitter(self, chunk_size: int, chunk_overlap: int):
        logger.info("Unfolding knowledge into digestible insights...")
        self.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _create_index_and_query_engine(self):
        logger.info("Assembling sacred insights for guidance...")
        split_docs = []
        for doc in self.documents:
            chunks = self.text_splitter.split_text(doc.text)
            for chunk in chunks:
                split_docs.append(Document(text=chunk, metadata=doc.metadata))

        logger.info(f"Total passages for reflection: {len(split_docs)}")

        Settings.embed_model = self.embedding_model
        Settings.llm = self.llm
        Settings.node_parser = self.text_splitter

        self.index = VectorStoreIndex.from_documents(split_docs)
        self.query_engine = self.index.as_query_engine()
        logger.info("KrishnaChat is attuned with the Gita's wisdom.")

    @staticmethod
    def _truncate_answer(text: str, max_sentences: int) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= max_sentences:
            return text
        return ' '.join(sentences[:max_sentences])

    @staticmethod
    def _validate_verse(chapter: int, verse: int) -> bool:
        return 1 <= chapter <= MAX_CHAPTER and 1 <= verse <= MAX_VERSES[chapter - 1]

    @staticmethod
    def _decode_metadata(meta: Dict) -> str:
        type_map = {"skrt": "Sanskrit", "trs": "Translation", "pur": "Purport"}
        return f"{meta.get('s', 'unknown')}, Chapter {meta.get('c', '?')}, Verse {meta.get('v', '?')}, {type_map.get(meta.get('t', ''), 'unknown')}"

    @staticmethod
    def _get_text_type(text_type: str) -> str:
        type_map = {"skrt": "Sanskrit", "trs": "Translation", "pur": "Purport"}
        return type_map.get(text_type, text_type)

    @staticmethod
    def _extract_core_english(text: str) -> str:
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if (
                line
                and not line.startswith('Bg. ')
                and not re.fullmatch(r'[A-Z]{2,}', line)
                and not line == line.upper()
                and not line.startswith('\\')
                and not line.startswith('verse ')
                and not re.match(r'^\d+\.\d+', line)
                and not re.match(r'^chapter\s+\d+', line, re.IGNORECASE)
            ):
                lines.append(line)
        return '\n'.join(lines).strip()

    def generate_response(self, question: str, max_sentences: int = 4, include_full_verse: bool = True) -> str:
        if not question.strip():
            return "Please share your question."

        question = re.sub(r'\[[^\]]+\]', '', question).strip().lower()

        logger.info(f"Reflecting on your inquiry: {question}")
        result = self.query_engine.query(question)
        logger.info("Divine insight received.")

        answer = str(result).strip()
        if not answer or answer.lower() == "empty response":
            answer = "I am listening, but I have not yet found guidance for your question."

        if "karma" in question and "kama" in answer.lower():
            answer = answer.replace("kama", "karma")
        if "yoga" in question and "yogi" in answer.lower():
            answer = answer.replace("yogi", "yoga")

        answer = self._truncate_answer(answer, max_sentences)

        source_info = []
        full_texts = []
        key_verses = []

        if hasattr(result, "source_nodes") and result.source_nodes:
            for node in result.source_nodes:
                meta = node.metadata
                if not meta or 'c' not in meta or 'v' not in meta:
                    continue
                try:
                    chapter = int(meta['c'])
                    verse = int(meta['v'])
                    text_type = meta.get('t', '')
                    source = meta.get('s', 'unknown')

                    if not self._validate_verse(chapter, verse):
                        continue

                    key_verses.append((chapter, verse, text_type, node.text))
                    ref = self._decode_metadata(meta)
                    if ref not in source_info:
                        source_info.append(ref)
                except (ValueError, KeyError):
                    continue

        for chapter, verse, text_type, text in key_verses:
            if include_full_verse:
                core = self._extract_core_english(text)
                if core:
                    full_texts.append(f"**Chapter {chapter}, Verse {verse} ({self._get_text_type(text_type)}):**\n{core}\n")

        references = "; ".join(source_info) if source_info else "sacred scripture"

        if "karma" in question:
            answer += (
                "\n\n**Practical Guidance:**\n"
                "Karma yoga is one of the Gita’s most practical teachings. Perform your duties sincerely, "
                "without attachment to personal gain or anxiety over results. See your daily activities "
                "as an offering to the Divine, and gradually cultivate inner peace and wisdom."
            )
        elif "yoga" in question:
            answer += (
                "\n\n**Practical Guidance:**\n"
                "Yoga, according to Krishna, is about steadying the mind and seeking union with the Supreme. "
                "Begin each day with a moment of quiet reflection; act mindfully, offer your work, "
                "and see the Divine in others. Even small consistent steps in this mindset bring growth."
            )
        elif "devotion" in question or "bhakti" in question:
            answer += (
                "\n\n**Practical Guidance:**\n"
                "Make your heart a sacred offering. Engage in prayer, service, or remembrance, "
                "cultivating selfless love for the Divine. Meditate on the Gita’s wisdom throughout your day, "
                "and look for opportunities to serve others with genuine care."
            )
        else:
            answer += (
                "\n\n**Practical Guidance:**\n"
                "Take a moment today to reflect on your life’s purpose. Remind yourself of the Gita’s teachings "
                "as you go about your daily duties. Even simple awareness of these truths brings light to your path."
            )

        response = [
            answer,
            f"\n\n(Source(s): {references})"
        ]

        if include_full_verse and full_texts:
            response.append("\n\n**Verses from the Gita:**")
            response.append("\n\n" + "----------\n\n".join(full_texts))

        return "\n".join(response)

if __name__ == "__main__":
    welcome = """
I'm KrishnaChat, an intelligent, compassionate assistant inspired by Krishna’s teachings.
I draw from the Bhagavad Gita (stored in krishna_pdfs) to offer grounded, practical,
and uplifting guidance for life, mindfulness, ethics, and personal growth.
Let’s reflect together.
    """
    print(welcome)
    bot = KrishnaSLMRAGModel()

    print("\nKrishnaChat is ready to support your journey!\nAsk anything about life, purpose, wisdom, or the Gita.")
    while True:
        query = input("\nAsk Krishna: ").strip()
        if not query:
            print("Please share your question.")
            continue
        try:
            response = bot.generate_response(query, include_full_verse=True)
            print("\nKrishna offers:\n")
            print(response)
        except Exception as e:
            print("🙏 We experience temporary uncertainty. Let's try again.")
            logger.error(str(e))
