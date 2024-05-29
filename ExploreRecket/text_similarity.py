from sentence_transformers.util import cos_sim
import pickle
import numpy
import re


# Preprocess text
def filter_chinese(text):
    # result = re.findall('[\u4e00-\u9fa50-9,.!?，。！？]', text)
    result = re.findall('[\u4e00-\u9fa5]', text)
    output = ''.join(result)
    return output


def get_sim_score(pre_model, text):
    with open('ExploreRecket/resources/red_packet_text.txt', 'r', encoding='UTF-8') as f:
        samples = f.read().split()
    with open('ExploreRecket/resources/red_packet_text.pkl', 'rb') as f:
        samples_embedding = pickle.load(f)
    pre_text = filter_chinese(text)
    # print(pre_text)
    # Encode text into vectors
    embedding = pre_model.encode(pre_text)
    cosine_sim = cos_sim(embedding, samples_embedding)
    sim_scores = cosine_sim[0].numpy()
    # Get the score for the most similar text
    max_score = max(sim_scores)
    # print('Maximum similarity score: ', max_score)
    index = numpy.where(sim_scores == max_score)
    # Get the most similar red packet text
    sim_text = samples[index[0][0]]
    # print('The most similar red packet text：', sim_text)
    return max_score, sim_text
