# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List
# import uuid
# import fitz  # PyMuPDF
# import re
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from ai_utils import rewrite_query
# import os
# import json
# from dotenv import load_dotenv
# from langchain.schema import OutputParserException 
# from langchain_openai import ChatOpenAI
# from langchain_chroma import Chroma
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ChromaDB config
# chroma_path = "./chroma_store"
# client = chromadb.PersistentClient(path=chroma_path)

# # PDF storage directory
# upload_dir = "./uploads"
# os.makedirs(upload_dir, exist_ok=True)

# # SentenceTransformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # LLM for profile extraction
# llm = ChatOpenAI(model_name="gpt-4o")

# # Split text into clean chunks
# def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
#     paragraphs = re.split(r"\n{2,}", text)
#     chunks = []
#     for para in paragraphs:
#         sentences = re.split(r"(?<=[.!?]) +", para.strip())
#         chunk = ""
#         for sentence in sentences:
#             if len(chunk) + len(sentence) < chunk_size:
#                 chunk += sentence + " "
#             else:
#                 chunks.append(chunk.strip())
#                 chunk = sentence + " "
#         if chunk:
#             chunks.append(chunk.strip())
#     return [chunk for chunk in chunks if len(chunk) > 30]

# # Extract text from PDF
# def extract_text_from_pdf(file: UploadFile) -> str:
#     with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
#         text = "\n".join([page.get_text("text", sort=True) for page in doc])
#         logger.info(f"Extracted text (first 500 chars): {text[:500]}")
#         return text

# # Extract candidate profile
# def extract_profile(text: str) -> dict:
#     prompt = f"""Extract structured data from the following resume text. Analyze the text carefully and identify relevant information even if it's not explicitly labeled. Return a JSON object with the following fields:
#     - skills: A list of technical and soft skills (e.g., Python, teamwork, AWS).
#     - experience: Total years of professional work experience, estimated from job history or explicitly stated years. If unclear, estimate conservatively or return 0.
#     - education: A list of degrees or educational qualifications (e.g., "B.S. Computer Science, XYZ University, 2020").
#     - certifications: A list of certifications (e.g., "AWS Certified Solutions Architect").
#     - contact_details: A dictionary containing contact information, including email, phone number, LinkedIn URL, and GitHub URL if available (e.g., {{"email": "example@domain.com", "phone": "+1234567890", "linkedin": "linkedin.com/in/example", "github": "github.com/example"}}). If a contact field is missing, exclude it from the dictionary.
#     If a field cannot be determined, return an empty list, 0, or empty dictionary as appropriate. Ensure the output is valid JSON.

#     Resume text:
#     {text}
#     """
#     try:
#         logger.info(f"Processing resume text (first 500 chars): {text[:500]}")
#         response = llm.invoke(prompt)
#         content = response.content.strip()
#         logger.info(f"LLM response: {content}")
#         # Remove markdown code block if present
#         if content.startswith("```json"):
#             content = content.replace("```json", "", 1).rstrip("```").strip()
#         # Check if response starts with JSON-like structure
#         if not content.startswith("{"):
#             logger.error(f"LLM response is not valid JSON after cleaning: {content[:100]}")
#             return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
#         parsed = json.loads(content)
#         # Validate expected keys
#         expected_keys = {"skills", "experience", "education", "certifications", "contact_details"}
#         if not all(key in parsed for key in expected_keys):
#             logger.error(f"Missing expected keys in JSON: {parsed}")
#             return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
#         logger.info(f"Successfully parsed profile: {parsed}")
#         return parsed
#     except Exception as e:
#         logger.error(f"Error extracting profile: {str(e)}")
#         return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}

# # Upload endpoint with PDF storage and profile extraction
# @app.post("/upload-resumes")
# async def upload_resumes(files: List[UploadFile] = File(...)):
#     collection = client.get_or_create_collection(name="resumes")
#     added = []
#     for file in files:
#         file_content = await file.read()
#         file_path = os.path.join(upload_dir, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(file_content)
#         file.file.seek(0)
#         text = extract_text_from_pdf(file)
#         chunks = split_into_chunks(text)
#         profile = extract_profile(text)
#         for idx, chunk in enumerate(chunks):
#             doc_id = str(uuid.uuid4())
#             embedding = embedding_model.encode(chunk).tolist()
#             collection.add(
#                 ids=[doc_id],
#                 documents=[chunk],
#                 embeddings=[embedding],
#                 metadatas=[{
#                     "filename": file.filename,
#                     "chunk_index": idx,
#                     "pdf_path": file_path,
#                     "profile": json.dumps(profile)
#                 }]
#             )
#         added.append({"filename": file.filename, "chunks": len(chunks)})
#     return {"status": "success", "added": added}

# # Query model
# class QueryItem(BaseModel):
#     query: str
#     top_k: int = 5

# # Classic semantic search with AI query rewriting
# @app.post("/search-resumes-ai")
# def search_resumes_ai(item: QueryItem):
#     collection = client.get_or_create_collection(name="resumes")
#     refined_query = rewrite_query(item.query)
#     query_embedding = embedding_model.encode(refined_query).tolist()
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=item.top_k,
#         include=["documents", "metadatas"]
#     )
#     return {
#         "original_query": item.query,
#         "refined_query": refined_query,
#         "results": results
#     }

# # LLM-based search
# @app.post("/search-resumes-llm")
# def search_resumes_llm(item: QueryItem):
#     try:
#         vectordb = Chroma(
#             collection_name="resumes",
#             embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#             persist_directory=chroma_path,
#             client=client
#         )
#         retriever = vectordb.as_retriever(search_kwargs={"k": item.top_k})
#         qa = RetrievalQA.from_chain_type(
#             llm=ChatOpenAI(model_name="gpt-4o-mini"),
#             retriever=retriever,
#             return_source_documents=True
#         )
#         result = qa.invoke({"query": item.query})
#         sources = [
#             {
#                 "filename": doc.metadata.get("filename"),
#                 "snippet": doc.page_content,
#                 "pdf_url": f"/get-pdf?filename={doc.metadata.get('filename')}",
#                 "candidate_details": json.loads(doc.metadata.get("profile", "{}"))
#             } for doc in result["source_documents"]
#         ]
#         logger.info(f"Retrieved sources: {sources}")
#         return {
#             "query": item.query,
#             "answer": result["result"],
#             "sources": sources
#         }
#     except Exception as e:
#         logger.error(f"Error in search_resumes_llm: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Serve PDF file
# @app.get("/get-pdf")
# def get_pdf(filename: str):
#     file_path = os.path.join(upload_dir, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="PDF not found")
#     return FileResponse(file_path, media_type="application/pdf", filename=filename)

# # View all stored resumes
# @app.get("/all-resumes")
# def list_all():
#     collection = client.get_or_create_collection(name="resumes")
#     return collection.get(include=["metadatas"])

# # Clear collection and PDFs
# @app.delete("/clear-resumes")
# def clear():
#     client.delete_collection("resumes")
#     for file in os.listdir(upload_dir):
#         os.remove(os.path.join(upload_dir, file))
#     return {"status": "collection and PDFs cleared"}

# # Preview specific resume by filename
# @app.get("/preview-resume")
# def preview_resume(filename: str):
#     collection = client.get_or_create_collection(name="resumes")
#     results = collection.get(include=["documents", "metadatas"])
#     documents = results["documents"]
#     metadatas = results["metadatas"]
#     filtered = [
#         (doc, meta) for doc, meta in zip(documents, metadatas)
#         if meta.get("filename") == filename
#     ]
#     if not filtered:
#         raise HTTPException(status_code=404, detail="Resume not found")
#     sorted_chunks = sorted(filtered, key=lambda x: x[1].get("chunk_index", 0))
#     full_text = "\n\n".join(chunk[0] for chunk in sorted_chunks)
#     return {"filename": filename, "content": full_text}


# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List
# import uuid
# import fitz  # PyMuPDF
# import re
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from ai_utils import rewrite_query
# import os
# import json
# from dotenv import load_dotenv
# from langchain.schema import OutputParserException 
# from langchain_openai import ChatOpenAI
# from langchain_chroma import Chroma
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ChromaDB config
# chroma_path = "./chroma_store"
# client = chromadb.PersistentClient(path=chroma_path)

# # PDF storage directory
# upload_dir = "./uploads"
# os.makedirs(upload_dir, exist_ok=True)

# # SentenceTransformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # LLM for profile extraction
# llm = ChatOpenAI(model_name="gpt-4o")

# # Split text into clean chunks
# def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
#     paragraphs = re.split(r"\n{2,}", text)
#     chunks = []
#     for para in paragraphs:
#         sentences = re.split(r"(?<=[.!?]) +", para.strip())
#         chunk = ""
#         for sentence in sentences:
#             if len(chunk) + len(sentence) < chunk_size:
#                 chunk += sentence + " "
#             else:
#                 chunks.append(chunk.strip())
#                 chunk = sentence + " "
#         if chunk:
#             chunks.append(chunk.strip())
#     return [chunk for chunk in chunks if len(chunk) > 30]

# # Extract text from PDF
# def extract_text_from_pdf(file: UploadFile) -> str:
#     with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
#         text = "\n".join([page.get_text("text", sort=True) for page in doc])
#         logger.info(f"Extracted text (first 500 chars): {text[:500]}")
#         return text

# # Extract candidate profile
# def extract_profile(text: str) -> dict:
#     prompt = f"""Extract structured data from the following resume text. Analyze the text carefully and identify relevant information even if it's not explicitly labeled. Return a JSON object with the following fields:
#     - skills: A list of technical and soft skills (e.g., Python, teamwork, AWS).
#     - experience: Total years of professional work experience, estimated from job history or explicitly stated years. If unclear, estimate conservatively or return 0.
#     - education: A list of degrees or educational qualifications (e.g., "B.S. Computer Science, XYZ University, 2020").
#     - certifications: A list of certifications (e.g., "AWS Certified Solutions Architect").
#     - contact_details: A dictionary containing contact information, including email, phone number, LinkedIn URL, and GitHub URL if available (e.g., {{"email": "example@domain.com", "phone": "+1234567890", "linkedin": "linkedin.com/in/example", "github": "github.com/example"}}). If a contact field is missing, exclude it from the dictionary.
#     If a field cannot be determined, return an empty list, 0, or empty dictionary as appropriate. Ensure the output is valid JSON.

#     Resume text:
#     {text}
#     """
#     try:
#         logger.info(f"Processing resume text (first 500 chars): {text[:500]}")
#         response = llm.invoke(prompt)
#         content = response.content.strip()
#         logger.info(f"LLM response: {content}")
#         # Remove markdown code block if present
#         if content.startswith("```json"):
#             content = content.replace("```json", "", 1).rstrip("```").strip()
#         # Check if response starts with JSON-like structure
#         if not content.startswith("{"):
#             logger.error(f"LLM response is not valid JSON after cleaning: {content[:100]}")
#             return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
#         parsed = json.loads(content)
#         # Validate expected keys
#         expected_keys = {"skills", "experience", "education", "certifications", "contact_details"}
#         if not all(key in parsed for key in expected_keys):
#             logger.error(f"Missing expected keys in JSON: {parsed}")
#             return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
#         logger.info(f"Successfully parsed profile: {parsed}")
#         return parsed
#     except Exception as e:
#         logger.error(f"Error extracting profile: {str(e)}")
#         return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}

# # Upload endpoint with PDF storage and profile extraction
# @app.post("/upload-resumes")
# async def upload_resumes(files: List[UploadFile] = File(...)):
#     collection = client.get_or_create_collection(name="resumes")
#     added = []
#     for file in files:
#         file_content = await file.read()
#         file_path = os.path.join(upload_dir, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(file_content)
#         file.file.seek(0)
#         text = extract_text_from_pdf(file)
#         chunks = split_into_chunks(text)
#         profile = extract_profile(text)
#         for idx, chunk in enumerate(chunks):
#             doc_id = str(uuid.uuid4())
#             embedding = embedding_model.encode(chunk).tolist()
#             collection.add(
#                 ids=[doc_id],
#                 documents=[chunk],
#                 embeddings=[embedding],
#                 metadatas=[{
#                     "filename": file.filename,
#                     "chunk_index": idx,
#                     "pdf_path": file_path,
#                     "profile": json.dumps(profile)
#                 }]
#             )
#         added.append({"filename": file.filename, "chunks": len(chunks)})
#     return {"status": "success", "added": added}

# # Query model
# class QueryItem(BaseModel):
#     query: str
#     top_k: int = 5

# # Classic semantic search with AI query rewriting
# @app.post("/search-resumes-ai")
# def search_resumes_ai(item: QueryItem):
#     collection = client.get_or_create_collection(name="resumes")
#     refined_query = rewrite_query(item.query)
#     query_embedding = embedding_model.encode(refined_query).tolist()
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=item.top_k,
#         include=["documents", "metadatas"]
#     )
#     return {
#         "original_query": item.query,
#         "refined_query": refined_query,
#         "results": results
#     }

# # LLM-based search with experience filtering
# @app.post("/search-resumes-llm")
# def search_resumes_llm(item: QueryItem):
#     try:
#         # Parse query to extract minimum experience requirement
#         min_experience = 0
#         experience_match = re.search(r"(\d+)\s*(?:years?|yrs?)\s*(?:of\s*experience)?", item.query, re.IGNORECASE)
#         if experience_match:
#             min_experience = float(experience_match.group(1))
#             logger.info(f"Extracted minimum experience from query: {min_experience} years")

#         vectordb = Chroma(
#             collection_name="resumes",
#             embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#             persist_directory=chroma_path,
#             client=client
#         )
#         retriever = vectordb.as_retriever(search_kwargs={"k": item.top_k})
#         qa = RetrievalQA.from_chain_type(
#             llm=ChatOpenAI(model_name="gpt-4o-mini"),
#             retriever=retriever,
#             return_source_documents=True
#         )
#         result = qa.invoke({"query": item.query})
        
#         # Filter sources based on minimum experience
#         filtered_sources = []
#         for doc in result["source_documents"]:
#             profile = json.loads(doc.metadata.get("profile", "{}"))
#             experience = profile.get("experience", 0)
#             if experience >= min_experience:
#                 filtered_sources.append({
#                     "filename": doc.metadata.get("filename"),
#                     "snippet": doc.page_content,
#                     "pdf_url": f"/get-pdf?filename={doc.metadata.get('filename')}",
#                     "candidate_details": profile
#                 })
#             else:
#                 logger.info(f"Filtered out resume {doc.metadata.get('filename')} with {experience} years (required: {min_experience})")
        
#         # Limit to top_k results after filtering
#         filtered_sources = filtered_sources[:item.top_k]
#         logger.info(f"Retrieved and filtered sources: {filtered_sources}")
        
#         return {
#             "query": item.query,
#             "answer": result["result"],
#             "sources": filtered_sources
#         }
#     except Exception as e:
#         logger.error(f"Error in search_resumes_llm: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Serve PDF file
# @app.get("/get-pdf")
# def get_pdf(filename: str):
#     file_path = os.path.join(upload_dir, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="PDF not found")
#     return FileResponse(file_path, media_type="application/pdf", filename=filename)

# # View all stored resumes
# @app.get("/all-resumes")
# def list_all():
#     collection = client.get_or_create_collection(name="resumes")
#     return collection.get(include=["metadatas"])

# # Clear collection and PDFs
# @app.delete("/clear-resumes")
# def clear():
#     client.delete_collection("resumes")
#     for file in os.listdir(upload_dir):
#         os.remove(os.path.join(upload_dir, file))
#     return {"status": "collection and PDFs cleared"}

# # Preview specific resume by filename
# @app.get("/preview-resume")
# def preview_resume(filename: str):
#     collection = client.get_or_create_collection(name="resumes")
#     results = collection.get(include=["documents", "metadatas"])
#     documents = results["documents"]
#     metadatas = results["metadatas"]
#     filtered = [
#         (doc, meta) for doc, meta in zip(documents, metadatas)
#         if meta.get("filename") == filename
#     ]
#     if not filtered:
#         raise HTTPException(status_code=404, detail="Resume not found")
#     sorted_chunks = sorted(filtered, key=lambda x: x[1].get("chunk_index", 0))
#     full_text = "\n\n".join(chunk[0] for chunk in sorted_chunks)
#     return {"filename": filename, "content": full_text}


# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List
# import uuid
# import fitz  # PyMuPDF
# import re
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from ai_utils import rewrite_query
# import os
# import json
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_chroma import Chroma
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from langchain.chains import LLMChain
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ChromaDB config
# chroma_path = "./chroma_store"
# client = chromadb.PersistentClient(path=chroma_path)

# # PDF storage directory
# upload_dir = "./uploads"
# os.makedirs(upload_dir, exist_ok=True)

# # SentenceTransformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # LLM for profile extraction and query parsing
# llm = ChatOpenAI(model_name="gpt-4o")

# # Split text into clean chunks
# def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
#     paragraphs = re.split(r"\n{2,}", text)
#     chunks = []
#     for para in paragraphs:
#         sentences = re.split(r"(?<=[.!?]) +", para.strip())
#         chunk = ""
#         for sentence in sentences:
#             if len(chunk) + len(sentence) < chunk_size:
#                 chunk += sentence + " "
#             else:
#                 chunks.append(chunk.strip())
#                 chunk = sentence + " "
#         if chunk:
#             chunks.append(chunk.strip())
#     return [chunk for chunk in chunks if len(chunk) > 30]

# # Extract text from PDF
# def extract_text_from_pdf(file: UploadFile) -> str:
#     with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
#         text = "\n".join([page.get_text("text", sort=True) for page in doc])
#         logger.info(f"Extracted text (first 500 chars): {text[:500]}")
#         return text

# # Extract candidate profile
# def extract_profile(text: str) -> dict:
#     prompt = f"""Extract structured data from the following resume text. Analyze the text carefully and identify relevant information even if it's not explicitly labeled. Return a JSON object with the following fields:
#     - skills: A list of technical and soft skills (e.g., Python, teamwork, AWS).
#     - experience: Total years of professional work experience, estimated from job history or explicitly stated years. If unclear, estimate conservatively or return 0.
#     - education: A list of degrees or educational qualifications (e.g., "B.S. Computer Science, XYZ University, 2020").
#     - certifications: A list of certifications (e.g., "AWS Certified Solutions Architect").
#     - contact_details: A dictionary containing contact information, including email, phone number, LinkedIn URL, and GitHub URL if available (e.g., {{"email": "example@domain.com", "phone": "+1234567890", "linkedin": "linkedin.com/in/example", "github": "github.com/example"}}). If a contact field is missing, exclude it from the dictionary.
#     If a field cannot be determined, return an empty list, 0, or empty dictionary as appropriate. Ensure the output is valid JSON.

#     Resume text:
#     {text}
#     """
#     try:
#         logger.info(f"Processing resume text (first 500 chars): {text[:500]}")
#         response = llm.invoke(prompt)
#         content = response.content.strip()
#         logger.info(f"LLM response: {content}")
#         # Remove markdown code block if present
#         if content.startswith("```json"):
#             content = content.replace("```json", "", 1).rstrip("```").strip()
#         # Check if response starts with JSON-like structure
#         if not content.startswith("{"):
#             logger.error(f"LLM response is not valid JSON after cleaning: {content[:100]}")
#             return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
#         parsed = json.loads(content)
#         # Validate expected keys
#         expected_keys = {"skills", "experience", "education", "certifications", "contact_details"}
#         if not all(key in parsed for key in expected_keys):
#             logger.error(f"Missing expected keys in JSON: {parsed}")
#             return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
#         logger.info(f"Successfully parsed profile: {parsed}")
#         return parsed
#     except Exception as e:
#         logger.error(f"Error extracting profile: {str(e)}")
#         return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}

# # Upload endpoint with PDF storage and profile extraction
# @app.post("/upload-resumes")
# async def upload_resumes(files: List[UploadFile] = File(...)):
#     collection = client.get_or_create_collection(name="resumes")
#     added = []
#     for file in files:
#         file_content = await file.read()
#         file_path = os.path.join(upload_dir, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(file_content)
#         file.file.seek(0)
#         text = extract_text_from_pdf(file)
#         chunks = split_into_chunks(text)
#         profile = extract_profile(text)
#         for idx, chunk in enumerate(chunks):
#             doc_id = str(uuid.uuid4())
#             embedding = embedding_model.encode(chunk).tolist()
#             collection.add(
#                 ids=[doc_id],
#                 documents=[chunk],
#                 embeddings=[embedding],
#                 metadatas=[{
#                     "filename": file.filename,
#                     "chunk_index": idx,
#                     "pdf_path": file_path,
#                     "profile": json.dumps(profile),
#                     "skills": json.dumps([skill.lower() for skill in profile.get("skills", [])]),
#                     "experience": float(profile.get("experience", 0)),
#                     "certifications": json.dumps([cert.lower() for cert in profile.get("certifications", [])])
#                 }]
#             )
#         added.append({"filename": file.filename, "chunks": len(chunks)})
#     return {"status": "success", "added": added}

# # Query model
# class QueryItem(BaseModel):
#     query: str
#     top_k: int = 5

# # Classic semantic search with AI query rewriting
# @app.post("/search-resumes-ai")
# def search_resumes_ai(item: QueryItem):
#     collection = client.get_or_create_collection(name="resumes")
#     refined_query = rewrite_query(item.query)
#     query_embedding = embedding_model.encode(refined_query).tolist()
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=item.top_k,
#         include=["documents", "metadatas"]
#     )
#     return {
#         "original_query": item.query,
#         "refined_query": refined_query,
#         "results": results
#     }

# # LLM-based search with intelligent query parsing
# @app.post("/search-resumes-llm")
# def search_resumes_llm(item: QueryItem):
#     try:
#         # Parse query to extract search criteria using LLM
#         prompt = f"""Analyze the following query and extract search criteria for filtering resumes. Return a JSON object with the following fields:
# - min_experience: float, minimum years of experience required (0 if not specified).
# - required_skills: list of strings, specific skills mentioned (empty list if none). Convert skills to lowercase.
# - required_certifications: list of strings, specific certifications mentioned (empty list if none). Convert certifications to lowercase.
# - other_criteria: string, any additional requirements not captured above (empty string if none).
# Ensure the output is valid JSON. Interpret the query flexibly to handle various phrasings (e.g., "3 years," "at least three years," "Python skills," "AWS certified"). If no specific 
# are mentioned, return defaults.

# Query: {item.query}
# """
#         response = llm.invoke(prompt)
#         content = response.content.strip()
#         if content.startswith("```json"):
#             content = content.replace("```json", "", 1).rstrip("```").strip()
#         try:
#             criteria = json.loads(content)
#         except json.JSONDecodeError:
#             logger.error(f"Failed to parse LLM criteria response: {content}")
#             criteria = {"min_experience": 0, "required_skills": [], "required_certifications": [], "other_criteria": item.query}
#         logger.info(f"Parsed query criteria: {criteria}")

#         # Validate criteria
#         min_experience = float(criteria.get("min_experience", 0))
#         required_skills = [skill.lower() for skill in criteria.get("required_skills", [])]
#         required_certifications = [cert.lower() for cert in criteria.get("required_certifications", [])]
#         other_criteria = criteria.get("other_criteria", "")

#         # Build ChromaDB where clause for metadata filtering
#         where_clauses = []
#         if min_experience > 0:
#             where_clauses.append({"experience": {"$gte": min_experience}})
#         else:
#             where_clauses.append({"experience": {"$gte": 0}})

#         # Set where_filter based on number of clauses
#         if len(where_clauses) > 1:
#             where_filter = {"$and": where_clauses}
#         elif where_clauses:
#             where_filter = where_clauses[0]
#         else:
#             where_filter = {}
#         logger.info(f"ChromaDB where filter: {where_filter}")

#         # Initialize vector store
#         vectordb = Chroma(
#             collection_name="resumes",
#             embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#             persist_directory=chroma_path,
#             client=client
#         )

#         # Perform direct query with where filter
#         query_embedding = embedding_model.encode(item.query).tolist()
#         query_results = vectordb._collection.query(
#             query_embeddings=[query_embedding],
#             n_results=item.top_k * 2,  # Retrieve more results to account for post-filtering
#             where=where_filter,
#             include=["documents", "metadatas"]
#         )

#         # Convert query results to LangChain Documents and filter by skills/certifications
#         documents = []
#         for doc, meta in zip(query_results["documents"][0], query_results["metadatas"][0]):
#             # Parse skills and certifications from metadata
#             meta_skills = json.loads(meta.get("skills", "[]"))
#             meta_certs = json.loads(meta.get("certifications", "[]"))
            
#             # Check if all required skills and certifications are present
#             skills_match = all(skill in meta_skills for skill in required_skills)
#             certs_match = all(cert in meta_certs for cert in required_certifications)
            
#             if skills_match and certs_match:
#                 documents.append(Document(page_content=doc, metadata=meta))
            
#             # Limit to top_k documents
#             if len(documents) >= item.top_k:
#                 break
#         logger.info(f"Retrieved documents after filtering: {len(documents)}")

#         # Create custom LLMChain with tailored prompt
#         qa_prompt = PromptTemplate(
#             input_variables=["context", "query"],
#             template="""You are a resume search assistant. Based on the provided query and retrieved resumes, return a concise summary of the matching candidates. If no resumes match the criteria, state: "No resumes found matching the criteria." Do not ask for clarification or additional information.

#         Query: {query}
#         Context: {context}

#         Provide a summary of the matching resumes, including key details like experience and skills if specified in the query. If other criteria (e.g., location) are mentioned, include them if available in the context.
#         """
#         )
#         qa = LLMChain(
#             llm=ChatOpenAI(model_name="gpt-4o-mini"),
#             prompt=qa_prompt
#         )

#         # Construct context from documents
#         context = "\n\n".join([doc.page_content for doc in documents])

#         # Include other_criteria in the query if present
#         query = item.query
#         if other_criteria:
#             query += f" {other_criteria}"

#         # Invoke LLMChain
#         result = qa.invoke({"query": query, "context": context})

#         # Format response
#         sources = [
#             {
#                 "filename": doc.metadata.get("filename"),
#                 "snippet": doc.page_content,
#                 "pdf_url": f"/get-pdf?filename={doc.metadata.get('filename')}",
#                 "candidate_details": json.loads(doc.metadata.get("profile", "{}"))
#             } for doc in documents
#         ]
#         logger.info(f"Retrieved sources: {sources}")

#         # Customize answer if no sources
#         answer = result["text"] if sources else "No resumes found matching the criteria."
#         return {
#             "query": item.query,
#             "answer": answer,
#             "sources": sources
#         }
#     except Exception as e:
#         logger.error(f"Error in search_resumes_llm: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Serve PDF file
# @app.get("/get-pdf")
# def get_pdf(filename: str):
#     file_path = os.path.join(upload_dir, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="PDF not found")
#     return FileResponse(file_path, media_type="application/pdf", filename=filename)

# # View all stored resumes
# @app.get("/all-resumes")
# def list_all():
#     collection = client.get_or_create_collection(name="resumes")
#     return collection.get(include=["metadatas"])

# # Clear collection and PDFs
# @app.delete("/clear-resumes")
# def clear():
#     client.delete_collection("resumes")
#     for file in os.listdir(upload_dir):
#         os.remove(os.path.join(upload_dir, file))
#     return {"status": "collection and PDFs cleared"}

# # Preview specific resume by filename
# @app.get("/preview-resume")
# def preview_resume(filename: str):
#     collection = client.get_or_create_collection(name="resumes")
#     results = collection.get(include=["documents", "metadatas"])
#     documents = results["documents"]
#     metadatas = results["metadatas"]
#     filtered = [
#         (doc, meta) for doc, meta in zip(documents, metadatas)
#         if meta.get("filename") == filename
#     ]
#     if not filtered:
#         raise HTTPException(status_code=404, detail="Resume not found")
#     sorted_chunks = sorted(filtered, key=lambda x: x[1].get("chunk_index", 0))
#     full_text = "\n\n".join(chunk[0] for chunk in sorted_chunks)
#     return {"filename": filename, "content": full_text}



from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uuid
import fitz  # PyMuPDF
import re
import chromadb
from sentence_transformers import SentenceTransformer
from ai_utils import rewrite_query, expand_query
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
import logging
from langgraph.graph import StateGraph, END
from graph import ResumeSearchGraph
from typing import TypedDict, List, Dict, Annotated

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB config
chroma_path = "./chroma_store"
client = chromadb.PersistentClient(path=chroma_path)

# PDF storage directory
upload_dir = "./uploads"
os.makedirs(upload_dir, exist_ok=True)

# SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM for profile extraction and query parsing
llm = ChatOpenAI(model_name="gpt-4o")

# Split text into clean chunks
def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?]) +", para.strip())
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) < chunk_size:
                chunk += sentence + " "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
        if chunk:
            chunks.append(chunk.strip())
    return [chunk for chunk in chunks if len(chunk) > 30]

# Extract text from PDF
def extract_text_from_pdf(file: UploadFile) -> str:
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text("text", sort=True) for page in doc])
        logger.info(f"Extracted text (first 500 chars): {text[:500]}")
        return text

# Extract candidate profile
def extract_profile(text: str) -> dict:
    prompt = f"""Extract structured data from the following resume text. Analyze the text carefully and identify relevant information even if it's not explicitly labeled. Return a JSON object with the following fields:
    - skills: A list of technical and soft skills (e.g., Python, teamwork, AWS).
    - experience: Total years of professional work experience, estimated from job history or explicitly stated years. If unclear, estimate conservatively or return 0.
    - education: A list of degrees or educational qualifications (e.g., "B.S. Computer Science, XYZ University, 2020").
    - certifications: A list of certifications (e.g., "AWS Certified Solutions Architect").
    - contact_details: A dictionary containing contact information, including email, phone number, LinkedIn URL, and GitHub URL if available (e.g., {{"email": "example@domain.com", "phone": "+1234567890", "linkedin": "linkedin.com/in/example", "github": "github.com/example"}}). If a contact field is missing, exclude it from the dictionary.
    If a field cannot be determined, return an empty list, 0, or empty dictionary as appropriate. Ensure the output is valid JSON.

    Resume text:
    {text}
    """
    try:
        logger.info(f"Processing resume text (first 500 chars): {text[:500]}")
        response = llm.invoke(prompt)
        content = response.content.strip()
        logger.info(f"LLM response: {content}")
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).rstrip("```").strip()
        if not content.startswith("{"):
            logger.error(f"LLM response is not valid JSON after cleaning: {content[:100]}")
            return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
        parsed = json.loads(content)
        expected_keys = {"skills", "experience", "education", "certifications", "contact_details"}
        if not all(key in parsed for key in expected_keys):
            logger.error(f"Missing expected keys in JSON: {parsed}")
            return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}
        logger.info(f"Successfully parsed profile: {parsed}")
        return parsed
    except Exception as e:
        logger.error(f"Error extracting profile: {str(e)}")
        return {"skills": [], "experience": 0, "education": [], "certifications": [], "contact_details": {}}

# Upload endpoint with PDF storage and profile extraction
@app.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    collection = client.get_or_create_collection(name="resumes")
    added = []
    for file in files:
        file_content = await file.read()
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        file.file.seek(0)
        text = extract_text_from_pdf(file)
        chunks = split_into_chunks(text)
        profile = extract_profile(text)
        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "filename": file.filename,
                    "chunk_index": idx,
                    "pdf_path": file_path,
                    "profile": json.dumps(profile),
                    "skills": json.dumps([skill.lower() for skill in profile.get("skills", [])]),
                    "experience": float(profile.get("experience", 0)),
                    "certifications": json.dumps([cert.lower() for cert in profile.get("certifications", [])])
                }]
            )
            # In /upload-resumes, after collection.add
            logger.info(f"Added document {doc_id} with metadata: {json.dumps({
                'filename': file.filename,
                'chunk_index': idx,
                'skills': [skill.lower() for skill in profile.get('skills', [])],
                'experience': float(profile.get('experience', 0))
            })}")
        added.append({"filename": file.filename, "chunks": len(chunks)})
    return {"status": "success", "added": added}

# Query model for classic search
class QueryItem(BaseModel):
    query: str
    top_k: int = 5

# Classic semantic search with AI query rewriting
@app.post("/search-resumes-ai")
def search_resumes_ai(item: QueryItem):
    collection = client.get_or_create_collection(name="resumes")
    refined_query = rewrite_query(item.query)
    query_embedding = embedding_model.encode(refined_query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=item.top_k,
        include=["documents", "metadatas"]
    )
    return {
        "original_query": item.query,
        "refined_query": refined_query,
        "results": results
    }

# LLM-based search with intelligent query parsing
# LLM-based search with intelligent query parsing
@app.post("/search-resumes-llm")
def search_resumes_llm(item: QueryItem):
    try:
        # Parse query to extract search criteria using LLM
        prompt = f"""Analyze the following query and extract search criteria for filtering resumes. Return a JSON object with the following fields:
- min_experience: float, minimum years of experience required (0 if not specified).
- required_skills: list of strings, specific skills mentioned (empty list if none). Convert skills to lowercase.
- required_certifications: list of strings, specific certifications mentioned (empty list if none). Convert certifications to lowercase.
- other_criteria: string, any additional requirements not captured above (empty string if none).
Ensure the output is valid JSON. Interpret the query flexibly to handle various phrasings (e.g., "3 years," "at least three years," "Python skills," "AWS certified"). If no specific criteria are mentioned, return defaults.

Query: {item.query}
"""
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).rstrip("```").strip()
        try:
            criteria = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM criteria response: {content}")
            criteria = {"min_experience": 0, "required_skills": [], "required_certifications": [], "other_criteria": item.query}
        logger.info(f"Parsed query criteria: {criteria}")

        # Validate criteria
        min_experience = float(criteria.get("min_experience", 0))
        required_skills = [skill.lower() for skill in criteria.get("required_skills", [])]
        required_certifications = [cert.lower() for cert in criteria.get("required_certifications", [])]
        other_criteria = criteria.get("other_criteria", "")

        # Build ChromaDB where clause for metadata filtering
        where_clauses = []
        if min_experience > 0:
            where_clauses.append({"experience": {"$gte": min_experience}})
        else:
            where_clauses.append({"experience": {"$gte": 0}})

        # Set where_filter based on number of clauses
        if len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}
        elif where_clauses:
            where_filter = where_clauses[0]
        else:
            where_filter = {}
        logger.info(f"ChromaDB where filter: {where_filter}")

        # Initialize vector store
        vectordb = Chroma(
            collection_name="resumes",
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory=chroma_path,
            client=client
        )

        # Perform direct query with where filter
        query_embedding = embedding_model.encode(item.query).tolist()
        query_results = vectordb._collection.query(
            query_embeddings=[query_embedding],
            n_results=item.top_k * 2,  # Retrieve more results to account for post-filtering
            where=where_filter,
            include=["documents", "metadatas"]
        )

        # Convert query results to LangChain Documents and filter by skills/certifications
        documents = []
        for doc, meta in zip(query_results["documents"][0], query_results["metadatas"][0]):
            # Parse skills and certifications from metadata
            meta_skills = json.loads(meta.get("skills", "[]"))
            meta_certs = json.loads(meta.get("certifications", "[]"))
            
            # Check if all required skills and certifications are present
            skills_match = all(skill in meta_skills for skill in required_skills)
            certs_match = all(cert in meta_certs for cert in required_certifications)
            
            if skills_match and certs_match:
                documents.append(Document(page_content=doc, metadata=meta))
            
            # Limit to top_k documents
            if len(documents) >= item.top_k:
                break
        logger.info(f"Retrieved documents after filtering: {len(documents)}")

        # Create custom LLMChain with tailored prompt
        qa_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a resume search assistant. Based on the provided query and retrieved resumes, return a concise summary of the matching candidates. If no resumes match the criteria, state: "No resumes found matching the criteria." Do not ask for clarification or additional information.

        Query: {query}
        Context: {context}

        Provide a summary of the matching resumes, including key details like experience and skills if specified in the query. If other criteria (e.g., location) are mentioned, include them if available in the context.
        """
        )
        qa = LLMChain(
            llm=ChatOpenAI(model_name="gpt-4o-mini"),
            prompt=qa_prompt
        )

        # Construct context from documents
        context = "\n\n".join([doc.page_content for doc in documents])

        # Include other_criteria in the query if present
        query = item.query
        if other_criteria:
            query += f" {other_criteria}"

        # Invoke LLMChain
        result = qa.invoke({"query": query, "context": context})

        # Format response
        sources = [
            {
                "filename": doc.metadata.get("filename"),
                "snippet": doc.page_content,
                "pdf_url": f"/get-pdf?filename={doc.metadata.get('filename')}",
                "candidate_details": json.loads(doc.metadata.get("profile", "{}"))
            } for doc in documents
        ]
        logger.info(f"Retrieved sources: {sources}")

        # Customize answer if no sources
        answer = result["text"] if sources else "No resumes found matching the criteria."
        return {
            "query": item.query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in search_resumes_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve PDF file
@app.get("/get-pdf")
def get_pdf(filename: str):
    file_path = os.path.join(upload_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

# View all stored resumes
@app.get("/all-resumes")
def list_all():
    collection = client.get_or_create_collection(name="resumes")
    return collection.get(include=["metadatas"])

# Clear collection and PDFs
@app.delete("/clear-resumes")
def clear():
    client.delete_collection("resumes")
    for file in os.listdir(upload_dir):
        os.remove(os.path.join(upload_dir, file))
    return {"status": "collection and PDFs cleared"}

# Preview specific resume by filename
@app.get("/preview-resume")
def preview_resume(filename: str):
    collection = client.get_or_create_collection(name="resumes")
    results = collection.get(include=["documents", "metadatas"])
    documents = results["documents"]
    metadatas = results["metadatas"]
    filtered = [
        (doc, meta) for doc, meta in zip(documents, metadatas)
        if meta.get("filename") == filename
    ]
    if not filtered:
        raise HTTPException(status_code=404, detail="Resume not found")
    sorted_chunks = sorted(filtered, key=lambda x: x[1].get("chunk_index", 0))
    full_text = "\n\n".join(chunk[0] for chunk in sorted_chunks)
    return {"filename": filename, "content": full_text}

# New conversational endpoint
class ChatInput(BaseModel):
    message: str
    thread_id: str
    top_k: int = 5

@app.post("/chat-resumes")
async def chat_resumes(input: ChatInput):
    try:
        collection = client.get_or_create_collection(name="resumes")
        count = collection.count()
        if count == 0:
            logger.info("No resumes found in the collection")
            return {
                "thread_id": input.thread_id,
                "response": "No resumes are available in the database. Please upload resumes using the /upload-resumes endpoint.",
                "history": [
                    {"role": "user", "content": input.message},
                    {"role": "assistant", "content": "No resumes are available in the database. Please upload resumes using the /upload-resumes endpoint."}
                ],
                "suggestions": [
                    "Upload a resume to start searching.",
                    "Try a different query after uploading resumes."
                ],
                "results": []
            }

        graph = ResumeSearchGraph(client, chroma_path, llm, embedding_model)
        config = {"configurable": {"thread_id": input.thread_id}}
        response = await graph.graph.ainvoke(
            {"messages": [{"role": "user", "content": input.message}], "top_k": input.top_k},
            config=config
        )
        return {
            "thread_id": input.thread_id,
            "response": response["response"],
            "history": response["messages"],
            "suggestions": response.get("suggestions", []),
            "results": response.get("results", [])
        }
    except Exception as e:
        logger.error(f"Error in chat_resumes: {str(e)}")
        return {
            "thread_id": input.thread_id,
            "response": f"An error occurred while processing your query: {str(e)}. Please try again or upload relevant resumes.",
            "history": [
                {"role": "user", "content": input.message},
                {"role": "assistant", "content": f"An error occurred while processing your query: {str(e)}. Please try again or upload relevant resumes."}
            ],
            "suggestions": [
                "Upload resumes if none are available.",
                "Refine your query with specific criteria (e.g., skills, experience).",
                "Check the server logs for more details."
            ],
            "results": []
        }