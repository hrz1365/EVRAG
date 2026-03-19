
import json
import time
import pandas as pd
import glob
import os
from src.pipelines.rag_pipeline import RAGPipeline,VectorStore,Embedder
# from src.pipelines import PDFLoader, TextChunker, Embedder, VectorStore, LLMEngineGemini
import numpy as np
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


import json
import pandas as pd

def load_and_flatten(json_path):
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    
    rows = []
    # print(data)
    for query_id, content in data.items():
        
        row = {
            "query_id": query_id,
            "question": content.get("question"),
            "answer": content.get("answer"),
            "context":content.get("context"),
            "section_number":content.get("section_number"), #"section_name","file_name": "","question_type": "",
            "section_name":content.get("section_name"), 
            "file_name":content.get("file_name"), 
            "question_type":content.get("question_type")

        }
        print(row["question"])
        rows.append(row)
    combined_df = pd.DataFrame(rows)
    combined_df.drop_duplicates(subset=['question'], keep='first', inplace=True)
    print(combined_df.columns, len(combined_df))
    return combined_df

# Example usage:
# Replace 'your_folder_path' with the actual path to your folder

folder_path = './data/eval_dataset_set.json'
all_data_df = load_and_flatten(folder_path)
all_data_df.to_csv('validationset.csv')
# exit()
# all_data_df= pd.read_csv('validationset.csv')
rag = RAGPipeline(pdf_path='./data/reports')
rag.build_index()
# Display the resulting DataFrame
print(all_data_df.head())
print(f"\nTotal number of rows in the combined DataFrame: {len(all_data_df)}")
if not 'rag_output' in all_data_df.columns:
    all_data_df['rag_output']=np.nan
    all_data_df['retrieved_contexts']=np.nan
embedder = Embedder("all-MiniLM-L6-v2").get()
vector_store= VectorStore(embedder)
retriever_obj = vector_store.retriever(k=7)

compressor = FlashrankRerank(odel="ms-marco-MiniLM-L-12-v2", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever_obj
)

# compressed_docs = compression_retriever.invoke(
#     "What did the president say about Ketanji Jackson Brown"
# )

for  index, row in all_data_df.iterrows():
    query= row["question"]
    print("Building index from PDF...")
    if  np.isnan(row['retrieved_contexts']):
        print("Retrieving for ", query)
        
        results = compression_retriever.invoke(query)
        
        source={}
        for i,doc in enumerate(results):
            #['producer', 'creator', 'creationdate', 'author', 'keywords', 'moddate', 'title', 'trapped', 'source', 'total_pages', 'page', 'page_label']
            source[str(i)]={
                    "content": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "title": doc.metadata.get('title')
                    
                }

    row["retrieved_contexts"]=json.dumps(source)
    print(row["retrieved_contexts"])
    exit()
    # rows.append(row)
       



#     # try:
#     #     answer, content = rag.query(query)
#     #     print(f"question: {row["question"]} \n\n Expected answer: {row["answer"]}, \n\n RAG Answer: {answer}\n\n content: {content}")
#     #     row["rag_output"]=answer
#     #     row["retrieved_context"]=content
#     #     time.sleep(60) 
#     # except Exception as e:
#     #     print(e)
#     #     time.sleep(60) 
# combined_df = pd.DataFrame(rows)
# combined_df = pd.DataFrame(rows)
# all_data_df.to_csv('validationset.csv')




