import requests
from json import loads

tagtogAPIUrl = "https://www.tagtog.net/-api/documents/v1"

auth = requests.auth.HTTPBasicAuth(username="Paul-2", password="dumb_password")
common_params = {"owner": "LouisDlms", "project": "fake_news"}

def get_tagtog_id(doc_id: int) -> str:
    filename = f"{doc_id}.txt"
    
    params = common_params.copy()
    params["search"] = "folder:pool"
    
    # Get number of pages
    response = requests.get(tagtogAPIUrl, params=params, auth=auth)
    response = loads(response.text)
    num_pages = response["pages"]["numPages"]

    # Paginated search
    for page in range(num_pages):
        params["page"] = page

        response = requests.get(tagtogAPIUrl, params=params, auth=auth)
        response = loads(response.text)

        search_result = [doc for doc in response["docs"] if doc["filename"] == filename]

        if search_result:
            doc = search_result[0]
            print(f"{filename} found on page {page}/{num_pages}")
            return doc["id"]
        
    raise Exception(f"The document #{doc_id} doesn't exist on tagtog.")

def get_ann_json(tagtog_id: str) -> dict:
    params = common_params.copy()
    params = {"owner": "LouisDlms", "project": "fake_news", 'ids':tagtog_id, "output": "ann.json"}
    response = requests.get(tagtogAPIUrl, params=params, auth=auth)
    
    try:
        ann = loads(response.text)
        return ann
    except Exception as e:
        raise Exception(f"Tagtog ID {tagtog_id} is incorrect. Use get_tagtog_id.")

def get_ann_from_article_id(id: int) -> dict:
    tagtog_id = get_tagtog_id(id)
    ann = get_ann_json(tagtog_id)
    return ann

def get_ann_legend() -> dict:
    url = "https://www.tagtog.net/-api/settings/v1/annotationsLegend?owner=LouisDlms&project=fake_news"
    response = requests.get(url, auth=auth)

    legend = loads(response.text)
    # Better to reverse for our purpose
    reversed_legend = {v: k for k, v in legend.items()}
    return reversed_legend

def get_text_from_id(id: int) -> str:
    tagtog_id = get_tagtog_id(id)
    return get_text(tagtog_id)

def get_text(tagtog_id: int) -> str:
    params = common_params.copy()
    params['ids'] = tagtog_id
    params["output"] = "orig"

    response = requests.get(tagtogAPIUrl, params=params, auth=auth)
    if response.status_code == 200:
        return response.content.decode()

def remove_articles(article_id: int):
    params = common_params.copy()
    params['search'] = f'filename:{article_id}.txt'

    response = requests.delete(tagtogAPIUrl, params=params, auth=auth)
    print(response.text)

if __name__ == '__main__':
    # Run an example. You can play with the article_id parameter
    article_id = 6 #2790 on page 53

    tagtog_id = get_tagtog_id(article_id)
    ann = get_ann_json(tagtog_id)

    legend = get_ann_legend()
    text = get_text_from_id(article_id)

    print(f"ID: {article_id}")
    print(f"Tagtog ID: {tagtog_id}")
    print(f"ann.json: {ann}")
    print(f"Legend: {legend}")
    print(f"Text: {text}")