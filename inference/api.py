from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.router import AdaptiveRAGRouter
import os

app = FastAPI()
router = AdaptiveRAGRouter(model_path='models/rag_router')

class QueryRequest(BaseModel):
    query: str

@app.get('/')
def read_root():
    return {"msg": "Adaptive RAG Router API"}

@app.post('/classify')
def classify_query(req: QueryRequest):
    result = router.classify(req.query)
    action = 'escalate'
    # Basic threshold decision
    if result['confidence'] >= 0.85:
        if result['intent'] == 'conversational':
            action = 'skip_rag'
        elif result['intent'] == 'factual':
            action = 'vector_search'
        elif result['intent'] == 'analytical':
            action = 'sql_rag'
        elif result['intent'] == 'temporal':
            action = 'web_search_api'
    else:
        action = 'escalate_to_gpt4'
    return {
        'intent': result['intent'],
        'confidence': result['confidence'],
        'latency_ms': round(result['latency_ms'], 2),
        'action': action
    }

def main():
    import uvicorn
    uvicorn.run('src.inference.api:app', host='0.0.0.0', port=8000, reload=False)

if __name__ == '__main__':
    main()