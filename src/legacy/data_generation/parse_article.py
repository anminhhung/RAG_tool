import scipdf
from tqdm import tqdm
import json
import pandas as pd
import argparse

def parse_paper(paper_id):
    try:
        article_dict = scipdf.parse_pdf_to_dict(f'https://arxiv.org/pdf/{paper_id}.pdf') # return dictionary
        return article_dict
    except Exception as e:
        print(f'Error in parsing {paper_id}: {e}')
        return None

def main(args):

    cols = ['id', 'title', 'abstract', 'categories']
    data = []
    file_name = args.file_name


    with open(file_name, encoding='latin-1') as f:
        for line in f:
            doc = json.loads(line)
            lst = [doc['id'], doc['title'], doc['abstract'], doc['categories']]
            data.append(lst)

    df_data = pd.DataFrame(data=data, columns=cols)
    topics = ['cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL'] # filter the topics that are not about AI & DS

    filtered_data = df_data[df_data['categories'].isin(topics)]
    filtered_data = filtered_data.iloc[-20000:]

    all_articles = []
    with open('../output/parsed_article.json', 'r') as f:
        all_articles = json.load(f)


    for row in tqdm(filtered_data.iterrows(), total=len(filtered_data)):
        paper_id = row[1]['id']
        article_dict = parse_paper(paper_id)
        all_articles.append(article_dict)
        with open(args.output_path, 'w') as f:
            json.dump(all_articles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='../data/arxiv-metadata-oai-snapshot.json')
    parser.add_argument('--output_path', type=str, default='../output/parsed_article.json')
    main()