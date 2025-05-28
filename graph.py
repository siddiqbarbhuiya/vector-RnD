from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json
from typing import TypedDict, List, Dict
from ai_utils import expand_query
import logging
import re
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    messages: List[Dict[str, str]]
    query: str
    top_k: int
    results: List[Dict]
    response: str
    suggestions: List[str]
    criteria: Dict
    query_type: str
    agent_mode: str

class ResumeSearchGraph:
    def __init__(self, client, chroma_path: str, llm, embedding_model):
        self.client = client
        self.chroma_path = chroma_path
        self.llm = llm
        self.embedding_model = embedding_model
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("single_agent_process", self.single_agent_process)
        workflow.add_node("parse_query_agent", self.parse_query_agent)
        workflow.add_node("filter_resumes_agent", self.filter_resumes_agent)
        workflow.add_node("generate_response_agent", self.generate_response_agent)
        workflow.add_node("count_resumes", self.count_resumes)

        workflow.set_entry_point("classify_query")
        workflow.add_conditional_edges(
            "classify_query",
            lambda state: state["agent_mode"] if state["query_type"] != "count" else "count",
            {
                "single": "single_agent_process",
                "multi": "parse_query_agent",
                "count": "count_resumes"
            }
        )
        workflow.add_edge("parse_query_agent", "filter_resumes_agent")
        workflow.add_edge("filter_resumes_agent", "generate_response_agent")
        workflow.add_edge("single_agent_process", "generate_response_agent")
        workflow.add_edge("count_resumes", "generate_response_agent")
        workflow.add_edge("generate_response_agent", END)
        return workflow.compile()

    def classify_query(self, state: GraphState) -> GraphState:
        message = state["messages"][-1]["content"].lower()
        top_k = state["top_k"]
        prompt = """Analyze the query to determine its type and complexity. Return a JSON object with:
- query_type: 'count' if asking about number of resumes, else 'search'
- agent_mode: 'single' for simple queries, 'multi' for complex queries

Query: {message}
"""
        try:
            response = self.llm.invoke(prompt.format(message=message))
            content = response.content.strip()
            logger.debug(f"Raw LLM response for query classification: {content}")
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).rstrip("```").strip()
            parsed = json.loads(content)
            query_type = parsed.get("query_type", "search")
            agent_mode = parsed.get("agent_mode", "single")
            logger.info(f"Classified query: type={query_type}, mode={agent_mode}")
            return {
                "messages": state["messages"],
                "query": message,
                "top_k": top_k,
                "results": [],
                "response": "",
                "suggestions": [],
                "criteria": {},
                "query_type": query_type,
                "agent_mode": agent_mode
            }
        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            return {
                "messages": state["messages"],
                "query": message,
                "top_k": top_k,
                "results": [],
                "response": f"Error classifying query: {str(e)}",
                "suggestions": [],
                "criteria": {},
                "query_type": "search",
                "agent_mode": "single"
            }

    def parse_query_agent(self, state: GraphState) -> GraphState:
        message = state["messages"][-1]["content"].lower()
        top_k = state["top_k"]
        logger.info(f"Parse Query Agent: Parsing query: {message}")

        role_skills = {
            "front-end developer": ["react js", "javascript", "html", "css", "typescript", "frontend development"],
            "software engineer": ["python", "java", "javascript", "sql", "software development", "programming"],
            "teacher": ["teaching", "education", "curriculum development", "classroom management"],
            "developer": ["programming", "software development", "coding"]
        }

        prompt = """Extract resume search criteria from the query. Map roles to skills. Return JSON with:
- needs_clarification: bool
- suggestions: list of strings
- criteria: dict with:
  - min_experience: float
  - required_skills: list of strings
  - required_certifications: list of strings
  - other_criteria: dict
Ensure valid JSON.

Query: {message}
"""
        try:
            response = self.llm.invoke(prompt.format(message=message))
            content = response.content.strip()
            logger.debug(f"Raw LLM response for parse_query: {content}")
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).rstrip("```").strip()
            parsed = json.loads(content)
            logger.debug(f"Parsed LLM response: {parsed}")

            needs_clarification = parsed.get("needs_clarification", False)
            if needs_clarification:
                return {
                    "messages": state["messages"],
                    "query": message,
                    "top_k": top_k,
                    "results": [],
                    "response": "Please clarify your query.",
                    "suggestions": parsed.get("suggestions", []),
                    "criteria": {},
                    "query_type": state["query_type"],
                    "agent_mode": state["agent_mode"]
                }

            criteria = parsed.get("criteria", {})
            criteria.setdefault("min_experience", 0.0)
            criteria.setdefault("required_skills", [])
            criteria.setdefault("required_certifications", [])
            criteria.setdefault("other_criteria", {})
            expanded_query = expand_query(message, criteria)
            logger.debug(f"Expanded query: {expanded_query}")
            return {
                "messages": state["messages"],
                "query": expanded_query,
                "top_k": top_k,
                "results": [],
                "response": "",
                "suggestions": [],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            skills = []
            min_experience = 0.0
            other_criteria = {}
            for skill in ["react js", "javascript", "typescript", "html", "css", "python"]:
                if skill in message:
                    skills.append(skill)
            for role, role_skill_list in role_skills.items():
                if role in message:
                    skills.extend([s for s in role_skill_list if s not in skills])
                    other_criteria["role"] = role
            exp_match = re.search(r"(?:at least )?(\d+(?:\.\d+)?)\s*(year|years)", message)
            if exp_match:
                min_experience = float(exp_match.group(1))

            criteria = {
                "min_experience": min_experience,
                "required_skills": skills or [message.strip()],
                "required_certifications": [],
                "other_criteria": other_criteria
            }
            return {
                "messages": state["messages"],
                "query": message,
                "top_k": top_k,
                "results": [],
                "response": "",
                "suggestions": [],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }

    def filter_resumes_agent(self, state: GraphState) -> GraphState:
        query = state["query"]
        top_k = state["top_k"]
        criteria = state["criteria"]
        logger.info(f"Filter Resumes Agent: Filtering with query: {query}, criteria: {criteria}")

        if not query or not isinstance(query, str):
            logger.error(f"Invalid or missing query: {query}")
            return {
                "messages": state["messages"],
                "query": query,
                "top_k": top_k,
                "results": [],
                "response": "Error: Invalid or missing query.",
                "suggestions": ["Provide a valid search query.", "Specify skills or experience."],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }

        try:
            vectordb = Chroma(
                collection_name="resumes",
                embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                persist_directory=self.chroma_path,
                client=self.client
            )

            collection = self.client.get_collection(name="resumes")
            count = collection.count()
            logger.info(f"Collection has {count} documents")
            if count == 0:
                return {
                    "messages": state["messages"],
                    "query": query,
                    "top_k": top_k,
                    "results": [],
                    "response": "No resumes found in the database.",
                    "suggestions": ["Upload resumes using /upload-resumes.", "Try again after adding resumes."],
                    "criteria": criteria,
                    "query_type": state["query_type"],
                    "agent_mode": state["agent_mode"]
                }

            search_kwargs = {"k": top_k * 2}
            qa_prompt = PromptTemplate(
                input_variables=["context", "query"],
                template="""You are a resume search assistant. Return a summary of matching candidates, including experience and skills. If no matches, return an empty string.

                Query: {query}
                Context: {context}

                Summary:
                """
            )
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o-mini"),
                chain_type="stuff",
                retriever=vectordb.as_retriever(search_kwargs=search_kwargs),
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt}
            )

            logger.debug(f"Invoking QA with input: {{'query': {query}}}")
            try:
                result = qa.invoke({"query": query})
            except Exception as invoke_err:
                logger.warning(f"QA invoke failed: {str(invoke_err)}. Trying alternative invocation.")
                result = qa({"query": "find me candidate who has maximum number of skills mentioned"})  # Fallback invocation
            logger.info(f"Retrieved {len(result['source_documents'])} documents for query: {query}")

            sources = []
            min_experience = float(criteria.get("min_experience", 0))
            required_skills = [s.lower() for s in criteria.get("required_skills", [])]
            for doc in result['source_documents']:
                meta = doc.metadata
                meta_skills = json.loads(meta.get("skills", "[]"))
                meta_exp = float(meta.get("experience", 0))
                logger.debug(f"Checking document: skills={meta_skills}, experience={meta_exp}")
                skills_match = all(
                    any(fuzz.ratio(skill, meta_skill.lower()) > 70 for meta_skill in meta_skills)  # Lowered threshold
                    for skill in required_skills
                ) if required_skills else True
                exp_match = meta_exp >= max(0, min_experience - 0.5) if min_experience else True
                if skills_match and exp_match:
                    sources.append({
                        "filename": meta.get("filename", "unknown"),
                        "snippet": doc.page_content[:200],
                        "pdf_url": f"/get-pdf?filename={meta.get('filename', 'unknown')}",
                        "candidate_details": json.loads(meta.get("profile", "{}"))
                    })
                if len(sources) >= top_k:
                    break

            if not sources:
                logger.info("No matching resumes found after filtering.")
                return {
                    "messages": state["messages"],
                    "query": query,
                    "top_k": top_k,
                    "results": [],
                    "response": "No resumes found matching the criteria.",
                    "suggestions": [
                        "Broaden your search (e.g., fewer skills or lower experience).",
                        f"Try related skills like {' or '.join(required_skills or ['javascript'])}.",
                        "Check if skill names match resume metadata (e.g., 'ReactJS' vs. 'React JS')."
                    ],
                    "criteria": criteria,
                    "query_type": state["query_type"],
                    "agent_mode": state["agent_mode"]
                }

            return {
                "messages": state["messages"],
                "query": query,
                "top_k": top_k,
                "results": sources,
                "response": result["result"],
                "suggestions": [],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }
        except Exception as e:
            logger.error(f"Error filtering resumes: {str(e)}")
            return {
                "messages": state["messages"],
                "query": query,
                "top_k": top_k,
                "results": [],
                "response": f"Error filtering resumes: {str(e)}",
                "suggestions": ["Try again or rephrase query.", "Check resume metadata for skill names."],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }

    def single_agent_process(self, state: GraphState) -> GraphState:
        message = state["messages"][-1]["content"].lower()
        top_k = state["top_k"]
        logger.info(f"Single Agent: Processing query: {message}")

        role_skills = {
            "front-end developer": ["react js", "javascript", "html", "css", "typescript", "frontend development"],
            "software engineer": ["python", "java", "javascript", "sql", "software development", "programming"]
        }

        skills = []
        min_experience = 0.0
        for skill in ["react js", "javascript", "typescript", "html", "css", "python"]:
            if skill in message:
                skills.append(skill)
        for role, role_skill_list in role_skills.items():
            if role in message:
                skills.extend([s for s in role_skill_list if s not in skills])
        exp_match = re.search(r"(?:at least )?(\d+(?:\.\d+)?)\s*(year|years)", message)
        if exp_match:
            min_experience = float(exp_match.group(1))

        criteria = {
            "min_experience": min_experience,
            "required_skills": skills or [message.strip()],
            "required_certifications": [],
            "other_criteria": {}
        }
        query = message

        try:
            vectordb = Chroma(
                collection_name="resumes",
                embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                persist_directory=self.chroma_path,
                client=self.client
            )
            collection = self.client.get_collection(name="resumes")
            count = collection.count()
            logger.info(f"Collection has {count} documents")
            if count == 0:
                return {
                    "messages": state["messages"],
                    "query": query,
                    "top_k": top_k,
                    "results": [],
                    "response": "No resumes found in the database.",
                    "suggestions": ["Upload resumes using /upload-resumes.", "Try again after adding resumes."],
                    "criteria": criteria,
                    "query_type": state["query_type"],
                    "agent_mode": state["agent_mode"]
                }

            search_kwargs = {"k": top_k}
            qa_prompt = PromptTemplate(
                input_variables=["context", "query"],
                template="""You are a resume search assistant. Return a summary of matching candidates, including experience and skills.

                Query: {query}
                Context: {context}

                Summary:
                """
            )
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o-mini"),
                chain_type="stuff",
                retriever=vectordb.as_retriever(search_kwargs=search_kwargs),
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt}
            )

            logger.debug(f"Invoking QA with input: {{'query': {query}}}")
            try:
                result = qa.invoke({"query": query})
            except Exception as invoke_err:
                logger.warning(f"QA invoke failed: {str(invoke_err)}. Trying alternative invocation.")
                result = qa({"query": "find me candidate who has maximum number of skills mentioned"})
            logger.info(f"Retrieved {len(result['source_documents'])} documents for query: {query}")

            sources = []
            for doc in result['source_documents']:
                meta = doc.metadata
                meta_skills = json.loads(meta.get("skills", "[]"))
                meta_exp = float(meta.get("experience", 0))
                logger.debug(f"Checking document: skills={meta_skills}, experience={meta_exp}")
                skills_match = all(
                    any(fuzz.ratio(skill, meta_skill.lower()) > 70 for meta_skill in meta_skills)
                    for skill in skills
                ) if skills else True
                exp_match = meta_exp >= max(0, min_experience - 0.5) if min_experience else True
                if skills_match and exp_match:
                    sources.append({
                        "filename": meta.get("filename", "unknown"),
                        "snippet": doc.page_content[:200],
                        "pdf_url": f"/get-pdf?filename={meta.get('filename', 'unknown')}",
                        "candidate_details": json.loads(meta.get("profile", "{}"))
                    })
                if len(sources) >= top_k:
                    break

            if not sources:
                return {
                    "messages": state["messages"],
                    "query": query,
                    "top_k": top_k,
                    "results": [],
                    "response": "No resumes found matching the criteria.",
                    "suggestions": [
                        "Broaden your search (e.g., fewer skills or lower experience).",
                        f"Try related skills like {' or '.join(skills or ['javascript'])}.",
                        "Check if skill names match resume metadata (e.g., 'ReactJS' vs. 'React JS')."
                    ],
                    "criteria": criteria,
                    "query_type": state["query_type"],
                    "agent_mode": state["agent_mode"]
                }

            return {
                "messages": state["messages"],
                "query": query,
                "top_k": top_k,
                "results": sources,
                "response": result["result"],
                "suggestions": [],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }
        except Exception as e:
            logger.error(f"Error in single agent process: {str(e)}")
            return {
                "messages": state["messages"],
                "query": query,
                "top_k": top_k,
                "results": [],
                "response": f"Error processing query: {str(e)}",
                "suggestions": ["Try again.", "Check resume metadata for skill names."],
                "criteria": criteria,
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }

    def count_resumes(self, state: GraphState) -> GraphState:
        try:
            collection = self.client.get_collection(name="resumes")
            count = collection.count()
            response = f"There are {count} resumes available in the database."
            logger.info(f"Counted {count} resumes")
            return {
                "messages": state["messages"],
                "query": state["query"],
                "top_k": state["top_k"],
                "results": [],
                "response": response,
                "suggestions": [
                    "Search for specific resumes (e.g., by role or experience)?",
                    "List types of resumes available?",
                    "Filter by specific skills or experience?"
                ],
                "criteria": state["criteria"],
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }
        except Exception as e:
            logger.error(f"Error counting resumes: {str(e)}")
            return {
                "messages": state["messages"],
                "query": state["query"],
                "top_k": state["top_k"],
                "results": [],
                "response": f"Error counting resumes: {str(e)}",
                "suggestions": [],
                "criteria": state["criteria"],
                "query_type": state["query_type"],
                "agent_mode": state["agent_mode"]
            }

    def generate_response_agent(self, state: GraphState) -> GraphState:
        results = state["results"]
        response = state["response"]
        messages = state["messages"]
        criteria = state["criteria"]
        query_type = state["query_type"]
        agent_mode = state["agent_mode"]
        top_k = state["top_k"]
        logger.info(f"Generate Response Agent: Mode={agent_mode}, Results={len(results)}")

        if not response or "error" in response.lower():
            response = "No resumes matched your criteria." if not results else response

        try:
            collection = self.client.get_collection(name="resumes")
            all_metadata = collection.get(include=["metadatas"])["metadatas"]
            available_skills = set()
            for meta in all_metadata:
                skills = json.loads(meta.get("skills", "[]"))
                available_skills.update(skills)
            skills_suggestions = list(available_skills)[:3]
        except Exception as e:
            logger.error(f"Error fetching available skills: {str(e)}")
            skills_suggestions = ["javascript", "python", "teaching"]

        prompt = """Generate a conversational response based on search results or count. For search queries, summarize results, highlighting experience and skills. If no results, explain why and suggest alternatives using available skills ({skills}). For count queries, confirm the number of resumes and offer actions. Use history for context.

        History: {history}
        Search Response: {response}
        Results: {results}
        Criteria: {criteria}
        Query Type: {query_type}
        Agent Mode: {agent_mode}
        """
        prompt = prompt.format(
            skills=", ".join(skills_suggestions),
            history=json.dumps(messages),
            response=response,
            results=json.dumps(results),
            criteria=json.dumps(criteria),
            query_type=query_type,
            agent_mode=agent_mode
        )
        try:
            llm_response = self.llm.invoke(prompt)
            content = llm_response.content.strip()

            suggestions_prompt = """Suggest 2-3 follow-up questions or actions based on the query history and results. For search queries, suggest refinements using available skills (e.g., {skills}). For count queries, suggest searches or uploads. Return a JSON list of strings.

            History: {history}
            Results: {results}
            Criteria: {criteria}
            Query Type: {query_type}
            Agent Mode: {agent_mode}
            """
            suggestions_prompt = suggestions_prompt.format(
                skills=", ".join(skills_suggestions),
                history=json.dumps(messages),
                results=json.dumps(results),
                criteria=json.dumps(criteria),
                query_type=query_type,
                agent_mode=agent_mode
            )
            suggestions_response = self.llm.invoke(suggestions_prompt)
            suggestions_content = suggestions_response.content.strip()
            try:
                if suggestions_content.startswith("```json"):
                    suggestions_content = suggestions_content.replace("```json", "", 1).rstrip("```").strip()
                suggestions = json.loads(suggestions_content)
                if not isinstance(suggestions, list) or not all(isinstance(s, str) for s in suggestions):
                    raise ValueError("Suggestions must be a list of strings")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid suggestions JSON: {suggestions_content}, error: {str(e)}")
                suggestions = [
                    f"Search for skills like {skills_suggestions[0]}.",
                    "Try a broader experience range.",
                    "Check if skill names match resume metadata."
                ]

            return {
                "messages": messages + [{"role": "assistant", "content": content}],
                "query": state["query"],
                "top_k": top_k,
                "results": results,
                "response": content,
                "suggestions": suggestions,
                "criteria": criteria,
                "query_type": query_type,
                "agent_mode": agent_mode
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "messages": messages + [{"role": "assistant", "content": f"Error generating response: {str(e)}"}],
                "query": state["query"],
                "top_k": top_k,
                "results": results,
                "response": f"Error generating response: {str(e)}",
                "suggestions": ["Try again.", "Refine your query."],
                "criteria": criteria,
                "query_type": query_type,
                "agent_mode": agent_mode
            }