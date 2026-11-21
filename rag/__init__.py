"""Agentic RAG over SEC filings.

    config      settings: models, chunking, K, thresholds
    retrieval   the retrieval strategies
    agent       the router, the diagnosis, the policy and the LangGraph graph
    decide      the decision-model client
    ladder      the baseline agent (no decision model)
    evaluate    retrieval metrics, the judge and the cost accounting
"""
