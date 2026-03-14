import json
import time
from typing import List, Dict, Any

# ==========================================
# OPENAI HELPER FOR MULTI-AGENT
# ==========================================

def get_chat_completion(messages, temperature=0.0):
    """Helper to call OpenAI using the notebook's existing client and model."""
    import sys
    
    # Check if running in Streamlit, __main__ might not have what we need
    # We will try __main__ first, then mp3_assignment, lastly default to env globals
    try:
        import mp3_assignment as app_module
    except ImportError:
        import __main__ as app_module
        
    client = getattr(app_module, 'client', None)
    if client is None:
        import __main__
        client = getattr(__main__, 'client', None)
        
    model = getattr(app_module, 'ACTIVE_MODEL', "gpt-4o-mini")
    if getattr(app_module, 'ACTIVE_MODEL', None) is None:
        import __main__
        model = getattr(__main__, 'ACTIVE_MODEL', "gpt-4o-mini")
    
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

def get_all_schemas():
    try:
        import mp3_assignment as app_module
    except ImportError:
        import __main__ as app_module
    schemas = getattr(app_module, 'ALL_SCHEMAS', None)
    if schemas is None:
        import __main__
        return getattr(__main__, 'ALL_SCHEMAS', [])
    return schemas

# ==========================================
# 1. DATABASE AGENT
# ==========================================

DATABASE_AGENT_PROMPT = """
You are a Database Specialist Agent.
Your role is to query the local database to find stock tickers based on sectors or industries.
Use the `get_tickers_by_sector` tool to find requested companies.
Only return information based on the data retrieved from the database. Do not hallucinate tickers.
"""

def run_database_agent(task: str, schemas: list, verbose: bool = True):
    try:
        from mp3_assignment import run_specialist_agent as run_sp
    except ImportError:
        import __main__
        run_sp = getattr(__main__, 'run_specialist_agent', None)
    if not run_sp: raise ImportError("Missing run_specialist_agent in global scope")

    valid_tools = ["get_tickers_by_sector", "query_local_db"]
    agent_schemas = [s for s in schemas if s.get("function", {}).get("name") in valid_tools]
    if not agent_schemas and schemas: agent_schemas = schemas
    
    return run_sp(
        agent_name="Database Agent",
        system_prompt=DATABASE_AGENT_PROMPT,
        task=task,
        tool_schemas=agent_schemas,
        max_iters=5,
        verbose=verbose
    )

# ==========================================
# 2. FUNDAMENTALS AGENT
# ==========================================

FUNDAMENTALS_AGENT_PROMPT = """
You are a Fundamentals Specialist Agent.
Your role is to retrieve static or point-in-time company metrics, such as P/E ratio, EPS, market cap, and 52-week high/low.
Use the `get_company_overview` tool to fetch this data for specific stock tickers.
Never invent financial numbers. If the data is missing or returns an error, state that clearly.
"""

def run_fundamentals_agent(task: str, schemas: list, verbose: bool = True):
    try:
        from mp3_assignment import run_specialist_agent as run_sp
    except ImportError:
        import __main__
        run_sp = getattr(__main__, 'run_specialist_agent', None)
    
    valid_tools = ["get_company_overview"]
    agent_schemas = [s for s in schemas if s.get("function", {}).get("name") in valid_tools]
    if not agent_schemas and schemas: agent_schemas = schemas
    
    return run_sp(
        agent_name="Fundamentals Agent",
        system_prompt=FUNDAMENTALS_AGENT_PROMPT,
        task=task,
        tool_schemas=agent_schemas,
        max_iters=5,
        verbose=verbose
    )

# ==========================================
# 3. TECHNICAL/PRICE AGENT
# ==========================================

TECHNICAL_AGENT_PROMPT = """
You are a Technical and Price Specialist Agent.
Your role is to handle historical price performance and check current market open/close status.
Use the `get_price_performance` tool to evaluate how a stock or list of stocks performed over time.
Use the `get_market_status` tool to check if the market is open.
Do not hallucinate percentage changes or prices. Base all answers strictly on tool outputs.
"""

def run_technical_agent(task: str, schemas: list, verbose: bool = True):
    try:
        from mp3_assignment import run_specialist_agent as run_sp
    except ImportError:
        import __main__
        run_sp = getattr(__main__, 'run_specialist_agent', None)
    
    valid_tools = ["get_price_performance", "get_market_status"]
    agent_schemas = [s for s in schemas if s.get("function", {}).get("name") in valid_tools]
    if not agent_schemas and schemas: agent_schemas = schemas

    return run_sp(
        agent_name="Technical/Price Agent",
        system_prompt=TECHNICAL_AGENT_PROMPT,
        task=task,
        tool_schemas=agent_schemas,
        max_iters=5,
        verbose=verbose
    )

# ==========================================
# 4. SENTIMENT AGENT
# ==========================================

SENTIMENT_AGENT_PROMPT = """
You are a Sentiment Specialist Agent.
Your role is to retrieve and analyze recent news headlines and sentiment for specific tickers.
Use the `get_news_sentiment` tool to fetch news articles and their sentiment scores.
Summarize the sentiment accurately without fabricating news stories.
"""

def run_sentiment_agent(task: str, schemas: list, verbose: bool = True):
    try:
        from mp3_assignment import run_specialist_agent as run_sp
    except ImportError:
        import __main__
        run_sp = getattr(__main__, 'run_specialist_agent', None)
    
    valid_tools = ["get_news_sentiment"]
    agent_schemas = [s for s in schemas if s.get("function", {}).get("name") in valid_tools]
    if not agent_schemas and schemas: agent_schemas = schemas

    return run_sp(
        agent_name="Sentiment Agent",
        system_prompt=SENTIMENT_AGENT_PROMPT,
        task=task,
        tool_schemas=agent_schemas,
        max_iters=5,
        verbose=verbose
    )

# ==========================================
# ORCHESTRATOR
# ==========================================

ORCHESTRATOR_PROMPT = """
You are the Orchestrator Agent. 
Your job is to read the user's financial question and create a step-by-step sequential plan to answer it.
You have access to the following specialist agents:
- Database Agent: Finds stock tickers based on sectors or industries from the local database.
- Fundamentals Agent: Gets company metrics (P/E, EPS, market cap, 52-week high/low).
- Technical/Price Agent: Gets historical price performance and market status.
- Sentiment Agent: Gets recent news sentiment.

Output your plan as a valid JSON LIST of objects, where each object has:
- "agent": The exact name of the agent to use (must be one of the four above).
- "task": The specific instruction for that agent.

Since agents are executed sequentially, later agents can rely on data found by earlier agents (e.g., getting tickers first, then getting price for those tickers). So, if a question involves multiple domains, ensure the agents are ordered logically.
IMPORTANT: Return ONLY raw JSON list, without markdown fences.
Example:
[
  {"agent": "Database Agent", "task": "Find tickers for the energy sector."},
  {"agent": "Technical/Price Agent", "task": "Fetch 6-month price performance for the tickers found."}
]
"""

def get_orchestrator_plan(question: str, verbose: bool = True) -> list:
    response = get_chat_completion([
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {"role": "user", "content": question}
    ])
    content = response.choices[0].message.content.strip()
    
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        return json.loads(content)
    except Exception as e:
        if verbose: print(f"Failed to parse orchestrator plan: {e}. Running all agents individually.")
        return [
            {"agent": "Database Agent", "task": question},
            {"agent": "Technical/Price Agent", "task": question},
            {"agent": "Fundamentals Agent", "task": question},
            {"agent": "Sentiment Agent", "task": question}
        ]

# ==========================================
# CRITIC
# ==========================================

CRITIC_PROMPT = """
You are the Critic Agent.
Your job is to fact-check a specialist agent's answer against the raw data it retrieved.
Input format:
Agent Answer: <answer>
Raw Data: <JSON dump of tool outputs>

Tasks:
1. Verify that every number, price, sentiment label or definitive claim in the answer is supported by the raw data.
2. Identify any hallucinations or unsupported numbers.
3. Determine a confidence score from 0.0 (completely hallucinated/unsupported) to 1.0 (perfectly supported or clearly explains why data was unavailable).
4. List any specific issues found.

Output your review as a valid JSON object strictly matching this format:
{
  "confidence": 0.95,
  "issues_found": ["issue 1 here..."] // empty list if none
}
IMPORTANT: Return ONLY raw JSON, without markdown fences.
"""

def run_critic(agent_result, verbose: bool = True):
    content = f"Agent Answer: {agent_result.answer}\nRaw Data: {json.dumps(agent_result.raw_data)}"
    response = get_chat_completion([
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": content}
    ])
    res_text = response.choices[0].message.content.strip()
    
    if res_text.startswith("```json"):
        res_text = res_text[7:-3].strip()
    elif res_text.startswith("```"):
        res_text = res_text[3:-3].strip()
        
    try:
        parsed = json.loads(res_text)
        agent_result.confidence = float(parsed.get("confidence", 0.0))
        agent_result.issues_found = parsed.get("issues_found", [])
    except Exception as e:
        if verbose: print(f"Critic parse error: {e}")
        agent_result.confidence = 0.5
        agent_result.issues_found = ["Critic evaluation failed to parse"]

# ==========================================
# SYNTHESIZER
# ==========================================

SYNTHESIZER_PROMPT = """
You are the Synthesizer Agent.
Your job is to answer the user's original question using ONLY the provided verified findings from specialist agents.
1. Formulate a cohesive, clear final answer.
2. Directly address the user's question.
3. If the findings lack information, clearly state what couldn't be found.
4. Do not invent details beyond what the specialists provided.
"""

def run_synthesizer(question: str, agent_results: list, verbose: bool = True) -> str:
    findings = []
    for r in agent_results:
        conf = getattr(r, 'confidence', 0.0)
        iss = getattr(r, 'issues_found', [])
        findings.append(
            f"--- {r.agent_name} (Confidence: {conf:.2f}) ---\n"
            f"Answer: {r.answer}\n"
            f"Issues: {', '.join(iss) if iss else 'None'}"
        )
        
    content = f"Original Question: {question}\n\nSpecialist Findings:\n" + "\n\n".join(findings)
    
    response = get_chat_completion([
        {"role": "system", "content": SYNTHESIZER_PROMPT},
        {"role": "user", "content": content}
    ])
    return response.choices[0].message.content

# ==========================================
# MULTI-AGENT RUNNER
# ==========================================

def run_multi_agent(question: str, verbose: bool = True) -> dict:
    start_time = time.time()
    
    schemas = get_all_schemas()
    
    if verbose: print(f"[Multi-Agent] Generating orchestration plan...")
    plan = get_orchestrator_plan(question, verbose=verbose)
    
    agent_results = []
    accumulated_context = []
    
    agent_funcs = {
        "Database Agent": run_database_agent,
        "Fundamentals Agent": run_fundamentals_agent,
        "Technical/Price Agent": run_technical_agent,
        "Sentiment Agent": run_sentiment_agent
    }
    
    for step in plan:
        agent_name = step.get("agent")
        sub_task = step.get("task")
        
        if agent_name not in agent_funcs:
            continue
            
        full_task = f"Original Question: {question}\n\nYour Specific Task: {sub_task}"
        if accumulated_context:
            full_task += "\n\nContext from previous agents (use this data if needed!):\n" + "\n".join(accumulated_context)
            
        if verbose: print(f"--- Running {agent_name} ---")
        
        # Execute specialist
        res = agent_funcs[agent_name](full_task, schemas=schemas, verbose=verbose)
        
        if verbose: print(f"--- Critic Evaluating {agent_name} ---")
        run_critic(res, verbose=verbose)
        
        agent_results.append(res)
        accumulated_context.append(f"[{agent_name} output]: {res.answer}")
        
    if verbose: print("--- Synthesizer Generating Final Answer ---")
    final_answer = run_synthesizer(question, agent_results, verbose=verbose)
    
    elapsed = time.time() - start_time
    
    return {
        "final_answer": final_answer,
        "agent_results": agent_results,
        "elapsed_sec": elapsed,
        "architecture": "orchestrator-critic"
    }

# ==========================================
# LLM EVALUATOR (LLM-AS-JUDGE)
# ==========================================

EVALUATOR_SYSTEM_PROMPT = """
You are an expert LLM-as-judge evaluator for a financial AI assistant.
Your job is to read a question, the expected answer description, and an agent's actual answer, then score it.
You MUST output your evaluation strictly as a valid JSON object matching this structure exactly:
{
    "score"                 : 0,
    "max_score"             : 3,
    "reasoning"             : "one sentence explaining the score",
    "hallucination_detected": false,
    "key_issues"            : []
}
IMPORTANT: Return ONLY raw JSON, with no markdown fences, prefixes, or suffixes.
"""

def run_evaluator(question: str, expected_answer: str, agent_answer: str) -> dict:
    import json
    
    fallback = {
        "score": 0, "max_score": 3, "reasoning": "evaluator parse error",
        "hallucination_detected": False, "key_issues": ["evaluator failed to parse"]
    }
    
    rubric = """
### Scoring Rubric:
3 — Fully correct:    all required data present, numbers accurate, conditions met
2 — Partially correct: key data present but incomplete, gaps, or minor inaccuracies
1 — Mostly wrong:     attempted but wrong numbers, missed required conditions,
                      or claims that appear fabricated
0 — Complete failure: refused to answer, said data unavailable without trying tools,
                      or answer has no relevance to the question

### Hallucination Detection Rules:
NOTE: You can only see the agent's final text answer, not its tool execution traces. You MUST ASSUME the agent successfully called tools if it confidently provides the requested specific numbers, unless stated otherwise.
Flag hallucination_detected as `true` ONLY if you see:
- The agent explicitly states it is guessing, estimating, or doesn't have access to the data, but provides a specific number anyway.
- Stock tickers that don't exist or aren't relevant to the question.
- The answer clearly contradicts the expected answer type or invents unrelated information.
"""
    
    content = f"{rubric}\n\nQuestion: {question}\n\nExpected Answer: {expected_answer}\n\nAgent's Actual Answer: {agent_answer}"
    
    try:
        response = get_chat_completion([
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ])
        
        res_text = response.choices[0].message.content.strip()
        
        if res_text.startswith("```json"):
            res_text = res_text[7:-3].strip()
        elif res_text.startswith("```"):
            res_text = res_text[3:-3].strip()
            
        parsed = json.loads(res_text)
        
        # Ensure all required keys exist and have appropriate types
        return {
            "score": int(parsed.get("score", 0)),
            "max_score": 3,
            "reasoning": str(parsed.get("reasoning", "")),
            "hallucination_detected": bool(parsed.get("hallucination_detected", False)),
            "key_issues": list(parsed.get("key_issues", []))
        }
    except Exception as e:
        print(f"Evaluator error: {e}")
        return fallback



