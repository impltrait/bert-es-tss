from extract_feature import BertVector
from elasticsearch import Elasticsearch
import json
from elasticsearch.helpers import bulk

indexName = 'cvc'
bert = BertVector()


def get_es_client():
    return Elasticsearch("127.0.0.1", http_auth=("elastic", "elastic"),
                         port="9200")


def embed_text(sentences):
    """
    将所有的句子转化为向量
    """
    resp = []
    for s in sentences:
        cc = bert.encode([s])[0].tolist()
        resp.append(cc)
    return resp


def bulk_index_data():
    """
    将数据索引到es中，且其中包含描述的特征向量字段
    """
    print("begin embed index data to vector")
    with open("../data/data.json", encoding="utf-8") as file:
        load_dict = json.load(file)
    features = [doc["title"] for doc in load_dict]
    print("number of lines to embed:", len(features))
    features_vectors = embed_text(features)
    print("begin index data to es")
    requests = []
    for i, doc in enumerate(load_dict):
        request = {'_op_type': 'index',  # 操作 index update create delete
                   '_index': indexName,  # index
                   '_id': doc["id"],
                   '_source':
                       {
                           'title': doc["title"],
                           'keyword': doc["keyword"],
                           'content': doc["content"],
                           'feature_vector': features_vectors[i],
                       }
                   }
        requests.append(request)

    success, _ = bulk(get_es_client(), requests, index=indexName, raise_on_error=True)
    print(success)
    print("end index data to es")


def create_index():
    print("begin create index")
    setting = {
        "settings": {
            "number_of_replicas": 0,
            "number_of_shards": 2
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "keyword"
                },
                "keyword": {
                    "type": "keyword"
                },
                "content": {
                    "type": "text"
                },
                "feature_vector": {
                    "type": "dense_vector",
                    "dims": 768
                }
            }
        }
    }
    get_es_client().indices.create(index=indexName, body=setting)
    print("end create index")


def test():
    es = get_es_client()
    while True:
        try:
            query = input("Enter query: ")
            input_vector = bert.encode([query])[0].tolist()
            resp = es.search(index=indexName, body={
                "_source": ["title", "content"],
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.queryVector, doc['feature_vector'])+1",
                            "params": {
                                "queryVector": input_vector
                            }
                        }
                    }
                }
            })
            print("可能获得的疾病是：", end=" ")
            for hit in resp["hits"]["hits"]:
                print(hit["_source"]["title"], end="\t")
            print("\n")
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    # create_index()
    # bulk_index_data()
    test()
